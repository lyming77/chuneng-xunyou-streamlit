# -*- coding: utf-8 -*-
"""多项目「方案 A」寻优:读各项目收资表格 → 交叉字段补全缺失 → 全网格 → 每项目独立文件夹输出。

用法:
  python3 脚本/寻优_多项目.py                 # 跑全部 6 个新项目
  python3 脚本/寻优_多项目.py 东盘风电场      # 只跑单个项目(便于调试)

产出(结果/<项目名>/):
  方案A_全网格.csv / 方案A_最优配置.md / 时序_补全后.csv
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import multiprocessing as mp

sys.path.insert(0, "脚本")
import 寻优 as M
from 方案B_滚动MPC import solve_mpc
from 读取收资表格 import read_kv, parse_ranges, read_tou_table, zone_of, build_tou, MONTH_GROUP, COEF_DEFAULT

SRC_DIR = "收资表格 2"
OUT_ROOT = "结果"
H_SET = [2, 4, 6, 8, 10]
RATIO_BY_TYPE = {"光伏": 0.5, "风电": 0.3}
N_JOBS = 8

PROJECTS = ["东盘风电场", "六合龙源光伏电站", "冯家风电场",
            "山东玉山风电场", "滨海风电场", "陈公风电场"]

_DF = None
_PSTATION = None
_NYEAR = None
_LOAN_TERM = None
_RAMP = None
_COST_PARAMS = None


def _init(df, p_station, n_year, loan_term, ramp, cost_params):
    global _DF, _PSTATION, _NYEAR, _LOAN_TERM, _RAMP, _COST_PARAMS
    _DF = df
    _PSTATION = p_station
    _NYEAR = n_year
    _LOAN_TERM = loan_term
    _RAMP = ramp
    _COST_PARAMS = cost_params


def _worker(ph):
    P, H = ph
    R_year, r_cap, _, schedule = solve_mpc(
        P, H, _DF, P_STATION=_PSTATION, ramp=_RAMP,
        return_schedule=True)
    annual_cycles = M.DT * float(
        (schedule["cc"] + schedule["cg"]).sum()) / (P * H)
    f = M.project_finance(P, H, R_year, r_cap, n_year=_NYEAR,
                          loan_term=_LOAN_TERM, cost_params=_COST_PARAMS)
    return dict(P=P, H=H, E=P * H,
                annual_cycles=annual_cycles,
                R_year=R_year / 1e4, R_cap=r_cap / 1e4,
                capex=f["capex"] / 1e4, capex_rep=f["capex_rep"] / 1e4,
                fi_irr=f["fi_irr"] * 100, fi_npv=f["fi_npv"] / 1e4,
                eq_irr=f["eq_irr"] * 100, eq_npv=f["eq_npv"] / 1e4)


def fill_power(ts):
    """交叉字段补全三个功率字段,返回 (ts, 填充统计)。"""
    c_act, c_avail, c_fcst = "实际发电功率(MW)", "超短期预测功率(MW)", "预测发电功率(日前,MW)"
    ts = ts.copy()
    stats = {c: int(ts[c].isna().sum()) for c in (c_act, c_avail, c_fcst)}
    # 实际 ← 超短期 ← 日前;超短期 ← 实际 ← 日前;日前 ← 超短期 ← 实际
    ts[c_act] = ts[c_act].fillna(ts[c_avail]).fillna(ts[c_fcst])
    ts[c_avail] = ts[c_avail].fillna(ts[c_act]).fillna(ts[c_fcst])
    ts[c_fcst] = ts[c_fcst].fillna(ts[c_avail]).fillna(ts[c_act])
    # 实际发电功率为毛发电(≥0),负值毛刺截到 0
    ts[c_act] = ts[c_act].clip(lower=0.0)
    return ts, stats


def is_sharp_of(zones, month, hour):
    return any(a <= hour < b for a, b in zones[MONTH_GROUP[month]]["尖峰"])


def build_df(ts, params):
    """构建与 寻优.load_data 同结构的模型 df。"""
    t = pd.to_datetime(ts["时间戳"])
    df = pd.DataFrame({"time": t})
    df["avail"] = ts["超短期预测功率(MW)"].astype(float)
    df["act_mw"] = ts["实际发电功率(MW)"].astype(float)
    df["forecast_mw"] = ts["预测发电功率(日前,MW)"].astype(float)
    df["mkt"] = ts["实时市场价格(元/MWh)"].astype(float)
    df["mkt_fc"] = ts["预测价格(元/MWh)"].astype(float)

    p_aux = float(params["station"]["厂用电功率"])
    tou = build_tou(params["flat"], params["coef"], params["zones"], t)
    df = df.merge(tou, on="time", how="left")
    df["hour"] = df["time"].dt.hour
    df["month"] = df["time"].dt.month
    df["is_sharp"] = [is_sharp_of(params["zones"], m, h) for m, h in zip(df["month"], df["hour"])]

    avail, act, mkt = df["avail"].values, df["act_mw"].values, df["mkt"].values
    df["curt"] = np.where(mkt <= M.CURT_PRICE, np.maximum(avail - act, 0.0), 0.0)
    df["grid_need"] = np.maximum(0, p_aux - act)
    return df


def read_project(path):
    st = read_kv(path, "场站参数")
    tech = read_kv(path, "技术经济参数")
    flat = {int(str(k).replace("月", "")): float(v) for k, v in read_kv(path, "平段电价").items()}
    coef = {k: float(v) for k, v in read_kv(path, "峰谷平系数").items()} if read_kv(path, "峰谷平系数") else COEF_DEFAULT
    zones = read_tou_table(path)
    ts = pd.read_excel(path, sheet_name="时序数据")
    ts.columns = [c.strip() for c in ts.columns]
    ts = ts.sort_values("时间戳").reset_index(drop=True)
    return dict(station=st, tech=tech, flat=flat, coef=coef, zones=zones, ts=ts)


def cost_params_from_tech(tech):
    """把收资表格造价字段映射为核心模型参数;旧表缺少的新字段自动采用默认值。"""
    mapping = {
        "battery_per_mwh": ("电池造价 c_电池", 1.0),
        "land_per_mwh": ("土地造价 c_土地", 1.0),
        "pcs_per_mw": ("PCS 造价 c_PCS", 1.0),
        "station_equipment_per_mw": ("其他造价 c_其他", 1.0),
        "fixed_predevelopment": ("前期研究及申报费", 1.0),
        "install_rate": ("建筑安装工程费率", 0.01),
        "engineering_rate": ("建设管理设计监理费率", 0.01),
        "contingency_rate": ("基本预备费率", 0.01),
        "battery_unit_mwh": ("电池舱单体容量", 1.0),
        "pcs_unit_mw": ("PCS单套功率", 1.0),
    }
    result = {}
    for target, (source, factor) in mapping.items():
        value = tech.get(source)
        if value not in (None, ""):
            result[target] = float(value) * factor
    return result


def run_grid(df, p_station, n_year, loan_term, p_range, ramp, cost_params=None):
    combos = [(P, H) for P in p_range for H in H_SET]
    ctx = mp.get_context("spawn")
    with ctx.Pool(N_JOBS, initializer=_init,
                  initargs=(df, p_station, n_year, loan_term, ramp, cost_params or {}),
                  maxtasksperchild=2) as pool:
        rows = pool.map(_worker, combos, chunksize=1)
    return pd.DataFrame(rows)


def report(res, df, name, station, tech, p_station, n_year, fill_stats, p_range):
    st_type = station["场站类型"]
    by_fi = res.loc[res["fi_npv"].idxmax()]
    by_eq = res.loc[res["eq_npv"].idxmax()]
    irr_row = res.loc[res["fi_irr"].idxmax()]
    eq_irr_row = res.loc[res["eq_irr"].idxmax()]
    n_fi_neg = int((res["fi_npv"] < 0).sum())
    n_eq_neg = int((res["eq_npv"] < 0).sum())
    n_irr_nan = int(res["fi_irr"].isna().sum())
    r_peak = res.loc[res["R_year"].idxmax()]

    fill_lines = "\n".join(
        f"- **{c}**:缺 {fill_stats[c]} 点(已用同场站最相近字段交叉补全)" for c in fill_stats)

    lines = [
        f"# {name} 方案A寻优结果",
        "",
        f"> **场站**:{name}({st_type} · 装机 {p_station} MW · 厂用电 {station['厂用电功率']} MW · 项目年限 {n_year} 年)",
        f"> **数据**:{df['time'].min().date()} ~ {df['time'].max().date()}(全年 {len(df)} 点,15min)",
        f"> **网格**:P∈[{p_range[0]},{p_range[-1]}] MW(1~5@1MW、>5@5MW,上限=装机×{RATIO_BY_TYPE[st_type]:.0%})× H∈{{2,4,6,8,10}},共 {len(p_range)*len(H_SET)} 组合",
        f"> **口径**:{n_year} 年财务口径({'' if n_year > M.DEP_BATT else '老项目,不换电池;'}弃电 mkt≤−80 · 付费充电逐笔预测价差≥120 · 免费电量预测放电价值≥120 · 跨日库存保留原门槛)",
        "",
        "## 一、补全说明(数据不完整项)",
        "",
        fill_lines,
        "",
        "## 二、方案A 最优(全网格)",
        "",
        "| 指标 | 全投资NPV最优 | 资本金NPV最优 | 全投资IRR最优 | 资本金IRR最优 |",
        "|---|---|---|---|---|",
        f"| 储能功率 P | {by_fi['P']:.0f} MW | {by_eq['P']:.0f} MW | {irr_row['P']:.0f} MW | {eq_irr_row['P']:.0f} MW |",
        f"| 持续时长 H | {by_fi['H']:.0f} h | {by_eq['H']:.0f} h | {irr_row['H']:.0f} h | {eq_irr_row['H']:.0f} h |",
        f"| 装机 E | {by_fi['E']:.0f} MWh | {by_eq['E']:.0f} MWh | {irr_row['E']:.0f} MWh | {eq_irr_row['E']:.0f} MWh |",
        f"| 全投资 NPV | {by_fi['fi_npv']:.1f} 万 | {by_eq['fi_npv']:.1f} 万 | {irr_row['fi_npv']:.1f} 万 | {eq_irr_row['fi_npv']:.1f} 万 |",
        f"| 全投资 IRR | {by_fi['fi_irr']:.2f}% | {by_eq['fi_irr']:.2f}% | {irr_row['fi_irr']:.2f}% | {eq_irr_row['fi_irr']:.2f}% |",
        f"| 资本金 IRR | {by_fi['eq_irr']:.2f}% | {by_eq['eq_irr']:.2f}% | {irr_row['eq_irr']:.2f}% | {eq_irr_row['eq_irr']:.2f}% |",
        "",
        f"- **全投资 NPV 峰值** P={by_fi['P']:.0f}/H={by_fi['H']:.0f}({by_fi['fi_npv']:.1f} 万);**资本金 NPV 峰值** P={by_eq['P']:.0f}/H={by_eq['H']:.0f}({by_eq['eq_npv']:.1f} 万)。",
        f"- **全投资IRR峰值** P={irr_row['P']:.0f}/H={irr_row['H']:.0f}({irr_row['fi_irr']:.2f}%);**资本金IRR峰值** P={eq_irr_row['P']:.0f}/H={eq_irr_row['H']:.0f}({eq_irr_row['eq_irr']:.2f}%)。",
        f"- {len(res)} 组合中 {n_fi_neg} 个全投资 NPV<0、{n_eq_neg} 个资本金 NPV<0、{n_irr_nan} 个全投资 IRR 为空(NaN);全投资 IRR 最低 {res['fi_irr'].min():.2f}%。",
        f"- R_year 峰值 {r_peak['R_year']:.1f} 万(P={r_peak['P']:.0f}/H={r_peak['H']:.0f})。",
        "",
        "完整数据见 `方案A_全网格.csv`，可在 Excel 中按目标指标自行排序。",
    ]
    return "\n".join(lines)


def process(name):
    path = os.path.join(SRC_DIR, f"{name}_收资表格.xlsx")
    t0 = time.time()
    print(f"\n===== {name} =====", flush=True)
    p = read_project(path)
    p_station = float(p["station"]["装机容量"])
    st_type = p["station"]["场站类型"]
    n_year = int(float(p["tech"]["项目年限 n"]))
    loan_term = int(float(p["tech"]["贷款期限"]))
    ts, fill_stats = fill_power(p["ts"])
    df = build_df(ts, p)

    p_max = p_station * RATIO_BY_TYPE[st_type]
    p_range = list(range(1, 6)) + list(range(10, int(p_max) + 1, 5))
    print(f"  装机 {p_station} MW({st_type}) · 年限 {n_year} · 厂用电 {p['station']['厂用电功率']} MW · "
          f"P∈[{p_range[0]},{p_range[-1]}] 共 {len(p_range)} 点 · 限电点 {int((df['curt']>0).sum())} · "
          f"尖峰点 {int(df['is_sharp'].sum())} · 组合 {len(p_range)*len(H_SET)}", flush=True)

    ramp = None  # 方案A不设置爬坡约束
    cost_params = cost_params_from_tech(p["tech"])
    res = run_grid(df, p_station, n_year, loan_term, p_range, ramp, cost_params)

    out_dir = os.path.join(OUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)
    cols = ["P", "H", "E", "annual_cycles", "R_year", "R_cap", "capex", "capex_rep",
            "fi_irr", "fi_npv", "eq_irr", "eq_npv"]
    res[cols].to_csv(os.path.join(out_dir, "方案A_全网格.csv"), index=False, encoding="utf-8-sig")
    # 补全后时序(供审计)
    ts[["时间戳", "实际发电功率(MW)", "超短期预测功率(MW)", "预测发电功率(日前,MW)",
        "实时市场价格(元/MWh)", "预测价格(元/MWh)"]].to_csv(
        os.path.join(out_dir, "时序_补全后.csv"), index=False, encoding="utf-8-sig")

    md = report(res, df, name, p["station"], p["tech"], p_station, n_year, fill_stats, p_range)
    with open(os.path.join(out_dir, "方案A_最优配置.md"), "w", encoding="utf-8") as fp:
        fp.write(md)

    best = res.loc[res["fi_npv"].idxmax()]
    best_eq_irr = res.loc[res["eq_irr"].idxmax()]
    print(f"  资本金IRR最优 P={best_eq_irr['P']:.0f}/H={best_eq_irr['H']:.0f} "
          f"IRR={best_eq_irr['eq_irr']:.2f}%；全投资NPV最优 P={best['P']:.0f}/H={best['H']:.0f} "
          f"NPV={best['fi_npv']:.1f}万 · 耗时 {time.time()-t0:.0f}s", flush=True)


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    projects = [only] if only else PROJECTS
    for name in projects:
        process(name)
    print("\n全部完成。结果见 结果/<项目名>/", flush=True)


if __name__ == "__main__":
    main()
