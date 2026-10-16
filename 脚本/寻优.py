# -*- coding: utf-8 -*-
"""新能源配建储能「容量·收益寻优」模型。

外层:(P, H) 网格搜索,枚举储能功率与持续时长。
内层:储能充放电仿真(充放互斥 MILP),得全年收益。两种口径:
      - solve_day/simulate: 每天一个 LP(96 点,15min),日末强制 SOC=5%(日独立,遗留)。
      - solve_year: 全年耦合一个 MILP(35040 点,带充放互斥二进制),允许跨日结转(当前默认)。
      本文件为「全知上界」版本:决策与结算都用实际值,量化理想上界。

依赖: pulp + highspy(HiGHS 求解器,`pip install highspy`)。
用法: python3 脚本/寻优.py
"""
import numpy as np
import pandas as pd
import pulp

# ---------- 已确认参数 ----------
DT = 0.25            # 时间步长 h
SOC_MIN, SOC_MAX = 0.05, 0.98   # 荷电状态上下限
ETA = 0.93           # 储能利用率(损耗全在放电侧)
P_AUX = 0.8          # 厂用电 MW
P_STATION = 480.0    # 阜山龙源光伏电站装机容量 MW(与收资表格一致)
STATION_TYPE = "光伏"   # 场站类型:光伏/风电,决定全网格 P 上限比例(见 §1.3 工作流程)
RATIO_BY_TYPE = {"光伏": 0.5, "风电": 0.3}   # 配储功率上限 = 装机 × 比例
P_MAX = P_STATION * RATIO_BY_TYPE[STATION_TYPE]   # 全网格 P 上限 MW(光伏 480×0.5=240)
CURT_PRICE = -80.0   # 限电电价阈值(元/MWh):电价 ≤ 该值判为限电,否则 avail−act 视为预测/设备误差
ARB_MIN = 120.0      # 充放价差门槛(元/MWh):方案A在日前模型中逐笔约束预测放电价−预测充电价≥120
CHG_PRICE_MAX = 250.0  # 充电电价上限(元/MWh):电价高于此值禁发电充电 cg(弃电充电 cc 仅在 mkt≤−80 时发生,天然满足)


def ramp_limit(p_station, st_type):
    """15min 有功功率变化限值(MW):线性爬坡下 15min 相邻变化 ΔP 需同时满足 1min、10min 两个滑动窗口,
    取 min(15·L1, 1.5·L10)。

    GB/T 19963(风电)1min/10min 限值;GB/T 19964(光伏)仅 1min≤10%装机、无 10min 限值。
      - 风电 <30MW:   min(15×3, 1.5×10) = 15 MW
      - 风电 30~150MW: min(1.5·装机, 0.5·装机) = 0.5·装机
      - 风电 >150MW:   min(15×15, 1.5×50) = 75 MW
      - 光伏:         15×10% = 1.5·装机(超过装机,无有效约束)
    """
    if st_type == "光伏":
        return 1.5 * p_station
    if p_station < 30:
        return 15.0
    if p_station <= 150:
        return 0.5 * p_station
    return 75.0

# 容量补偿
R_MECH = 0.8         # 机制电量/上网电量(被动参与=80%,容量补偿计 (1−R)=20%)
P_CAP = 30000.0      # 容量补偿电价 元/MW·月(已确认 3 万)

# 造价基础单价。详细造价在 investment_breakdown() 中按设备、建安、工程费和预备费展开。
C_BATT = 500000.0    # 电池系统 0.5 元/Wh = 50 万/MWh,第 10 年更换
C_LAND = 35000.0     # 场地/土地 0.035 元/Wh = 3.5 万/MWh
C_PCS = 270000.0     # PCS 0.27 元/W = 27 万/MW
C_OTHER = 200000.0   # 站内升压单元、开关、保护、消防等 0.2 元/W = 20 万/MW

# 全口径工程造价默认参数。费率以五莲 10MW/40MWh 可研的费用结构作校核,
# 但保留本项目已确认的设备基础单价;标准边界不含新建送出线路和主变扩容。
COST_MODEL = "detailed"
C_FIXED_PREDEV = 1500000.0                    # 前期研究、申报、接入及电能质量研究(元/项目)
RATE_INSTALL = 210.0 / 2703.75               # 建筑安装工程费/设备购置费
RATE_ENGINEERING = 136.95 / 2703.75          # 建设管理、勘察设计、监理/设备购置费
RATE_CONTINGENCY = 0.03                      # 基本预备费/预备费前投资
BATTERY_UNIT_MWH = 5.0                       # 设备离散校核:单舱容量
PCS_UNIT_MW = 2.5                            # 设备离散校核:单套 PCS/升压单元功率

# 分类折旧与经济寿命。税务折旧用于所得税,经济寿命用于评价期末资产余值。
TAX_LIFE_BATTERY = 10
TAX_LIFE_POWER = 10
TAX_LIFE_CIVIL = 20
AMORT_LIFE_PREDEV = 10
ECON_LIFE_BATTERY = 10
ECON_LIFE_POWER = 15
ECON_LIFE_CIVIL = 20
TERMINAL_RECOVERY_POWER = 1.0                # 剩余经济价值可回收比例(项目可覆盖)
TERMINAL_RECOVERY_CIVIL = 1.0
DEFAULT_COST_PARAMS = {
    "battery_per_mwh": C_BATT,
    "land_per_mwh": C_LAND,
    "pcs_per_mw": C_PCS,
    "station_equipment_per_mw": C_OTHER,
    "fixed_predevelopment": C_FIXED_PREDEV,
    "install_rate": RATE_INSTALL,
    "engineering_rate": RATE_ENGINEERING,
    "contingency_rate": RATE_CONTINGENCY,
    "battery_unit_mwh": BATTERY_UNIT_MWH,
    "pcs_unit_mw": PCS_UNIT_MW,
}
N_YEAR = 20          # 长周期项目年限(计算期)

# 融资与财务(全生命周期现金流口径)
EQUITY_RATIO = 0.30  # 资本金比例
LOAN_RATE = 0.03     # 贷款利率(年)
LOAN_TERM = 15       # 长周期项目首次贷款期限(年,等额本息)
LOAN_TERM_SHORT = 7  # 短周期项目首次贷款期限(年,等额本息)
LOAN_TERM_REP = 7    # 更换电池追加投资贷款期限(年,等额本息)
EQUITY_RETURN = 0.08 # 资本金回报率(资本金NPV折现率)
DEP_BATT = 10        # 电池折旧年限(直线,10+10 两段)
DEGR = 0.02          # 电池年容量衰减率(线性 2%/年,第 10 年换电池后重置;用户口径)
DEP_AC = TAX_LIFE_CIVIL  # 兼容旧输出字段;详细模式按资产分类折旧
SALVAGE = 0.0        # 兼容旧接口;详细模式的期末余值由经济寿命单独计算
OM_RATIO = 0.01      # 生产费用 = 初始投资 × 1%/年
TAX_RATE = 0.25      # 所得税率

# 加权平均资本成本(全投资NPV折现率)
WACC = EQUITY_RATIO * EQUITY_RETURN + (1 - EQUITY_RATIO) * LOAN_RATE * (1 - TAX_RATE)

# 尖峰时段(小时,左闭右开)——山东 2026 分时电价
SHARP_HOURS = {
    1: (16, 17, 18), 2: (16, 17, 18), 12: (16, 17, 18),
    3: (17, 18, 19), 4: (17, 18, 19), 5: (17, 18, 19),
    6: (17, 18, 19, 20, 21), 7: (17, 18, 19, 20, 21), 8: (17, 18, 19, 20, 21),
    9: (17, 18), 10: (17, 18), 11: (17, 18),
}


def load_data():
    """加载并对齐 35040 点时序数据。"""
    pv = pd.read_csv("15min数据/阜山龙源光伏_15min.csv", parse_dates=["time"])
    fc = pd.read_csv("15min数据/预测发电功率_15min.csv", parse_dates=["time"])
    mp = pd.read_csv("15min数据/实时价格_15min.csv", parse_dates=["time"])
    fp = pd.read_csv("15min数据/预测价格_15min.csv", parse_dates=["time"])
    tou = pd.read_csv("15min数据/工商业分时电价_15min.csv", parse_dates=["time"])

    df = pv[["time"]].copy()
    df = df.merge(pv[["time", "avail_kw", "act_mw"]], on="time")
    df = df.merge(fc[["time", "forecast_mw"]], on="time")
    df = df.merge(mp[["time", "price"]].rename(columns={"price": "mkt"}), on="time")
    df = df.merge(fp[["time", "price"]].rename(columns={"price": "mkt_fc"}), on="time")
    df = df.merge(tou[["time", "tou_price"]].rename(columns={"tou_price": "tou"}), on="time")
    df["avail"] = df["avail_kw"] / 1000.0
    df = df.drop(columns=["avail_kw"])

    # 尖峰时段标记
    df["hour"] = df["time"].dt.hour
    df["month"] = df["time"].dt.month
    df["is_sharp"] = [h in SHARP_HOURS[m] for m, h in zip(df["month"], df["hour"])]

    # 限电判断:电价 ≤ CURT_PRICE(深度负价=真限电)→ 弃电=avail−act;否则 avail−act 是预测/设备误差,不算弃电
    avail, act, mkt = df["avail"].values, df["act_mw"].values, df["mkt"].values
    curt = np.where(mkt <= CURT_PRICE, np.maximum(avail - act, 0.0), 0.0)
    df["curt"] = curt

    # 下网需求(厂用电扣除自发后仍需下网)
    df["grid_need"] = np.maximum(0, P_AUX - act)
    return df


def solve_day(p_h, day_arrays):
    """解一天(96 点)的 LP(无充放互斥 binary)。

    假套利(弃电充电与放电同刻)在光伏场景影响 <0.2%(弃电在午间、放电在傍晚),忽略。
    返回: (obj 元, P_c 序列(弃电+发电), P_cgen 序列(仅发电充电), P_d 序列)
    """
    N = 96
    P, H = p_h
    E_cap = P * H
    E_lo, E_hi = SOC_MIN * E_cap, SOC_MAX * E_cap
    act, curt = day_arrays["act"], day_arrays["curt"]
    mkt, tou, grid_need = day_arrays["mkt"], day_arrays["tou"], day_arrays["grid_need"]

    prob = pulp.LpProblem("day", pulp.LpMaximize)
    idx = range(N)

    pc_curt = pulp.LpVariable.dicts("c_c", idx, 0)   # 弃电充电(免费)
    pc_gen = pulp.LpVariable.dicts("c_g", idx, 0)    # 发电充电(按市场价)
    pu = pulp.LpVariable.dicts("u", idx, 0)          # 抵厂用电放电
    pv = pulp.LpVariable.dicts("v", idx, 0)          # 套利放电
    E = pulp.LpVariable.dicts("E", idx, E_lo, E_hi)  # 电量

    for t in idx:
        # 充电来源:弃电免费(≤弃电)、发电计价(≤发电,不购电充电)、合计 ≤ P
        # 注:原「充电≤0.8×新能源」系数已移除——它是日前防预测偏差的余量,运行日由 AGC
        #     按实际可用功率重分配,偏差由考核兜底;且本模型为全知上界(用实际发电),无预测偏差可防。
        #     正确上限 = min(P, 发电+弃电=可用功率)。
        prob += pc_curt[t] <= curt[t]
        prob += pc_gen[t] <= max(0.0, act[t])
        prob += pc_curt[t] + pc_gen[t] <= P
        # 放电: ≤ P、≤ 装机−发电; 抵厂用电 ≤ 下网需求
        prob += pu[t] + pv[t] <= min(P, P_STATION - act[t])
        prob += pu[t] <= grid_need[t]

    # SOC 动态与每日循环: E[t] = 时刻 t 末电量,覆盖全部 96 时刻
    prob += E[0] == E_lo + (pc_curt[0] + pc_gen[0]) * DT - (pu[0] + pv[0]) / ETA * DT
    for t in range(1, N):
        prob += E[t] == E[t - 1] + (pc_curt[t] + pc_gen[t]) * DT \
                 - (pu[t] + pv[t]) / ETA * DT
    prob += E[N - 1] == E_lo

    # 目标: 放电价值 − 充电成本
    prob += pulp.lpSum(DT * (tou[t] * pu[t] + mkt[t] * pv[t] - mkt[t] * pc_gen[t]) for t in idx)

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if prob.status != pulp.LpStatusOptimal:
        return 0.0, np.zeros(N), np.zeros(N), np.zeros(N)

    obj = pulp.value(prob.objective)
    pc = np.array([pc_curt[t].value() + pc_gen[t].value() for t in idx])
    pc_g = np.array([pc_gen[t].value() for t in idx])
    pd_ = np.array([pu[t].value() + pv[t].value() for t in idx])
    return obj, pc, pc_g, pd_


def _capacity_comp(df, pd_arr, pc_gen_arr):
    """储能容量补偿(元/年):储能充放对「全场」容量补偿的增量。

    容量补偿奖励的是「全场」在尖峰时段提供功率的能力,储能收入 = 充放带来的增量:
    - 放电 → 全场尖峰功率 + → 增量 +
    - 从发电充电(吞掉本该上网的电)→ 全场尖峰功率 − → 增量 −
    - 从弃电充电(本就被限电、不上网)→ 全场尖峰功率不变 → 增量 0
    故增量功率 = 放电 − 发电充电;月有效容量 = 该月尖峰时段平均(放电−发电充电),
    × (1−R) × P容量,12 个日历月求和。
    """
    tmp = df[["month", "is_sharp"]].copy()
    tmp["contrib"] = pd_arr - pc_gen_arr
    monthly = tmp.loc[tmp["is_sharp"]].groupby("month")["contrib"].mean()
    return float((1 - R_MECH) * P_CAP * monthly.sum())


def solve_year(p_h, df, return_schedule=False, ramp=None):
    """全年(35040 点)耦合 MILP,允许跨日结转(替代逐日独立求解)。

    与 solve_day 的区别:仅年初 E[0]=5% 与年末 E[N-1]=5%(年际稳态)作为边界,
    中间不再强制日末 SOC=5%,储能可跨日结转(低价日囤电、留到高价日放)。
    含充放互斥二进制 z[t]:同一时刻只能充电(z=1)或放电(z=0),杜绝同刻充放的假套利。
    返回: (R_year 元, 容量补偿 元/年, (抵厂用电收益, 套利收益, 充电成本))
    """
    P, H = p_h
    E_cap = P * H
    E_lo, E_hi = SOC_MIN * E_cap, SOC_MAX * E_cap
    act = df["act_mw"].values
    curt = df["curt"].values
    mkt = df["mkt"].values
    tou = df["tou"].values
    grid_need = df["grid_need"].values
    N = len(df)
    prob = pulp.LpProblem("year", pulp.LpMaximize)
    idx = range(N)
    pc_c = pulp.LpVariable.dicts("c_c", idx, 0)
    pc_g = pulp.LpVariable.dicts("c_g", idx, 0)
    pu = pulp.LpVariable.dicts("u", idx, 0)
    pv = pulp.LpVariable.dicts("v", idx, 0)
    E = pulp.LpVariable.dicts("E", idx, E_lo, E_hi)

    for t in idx:
        prob += pc_c[t] <= curt[t]
        prob += pc_g[t] <= (max(0.0, act[t]) if mkt[t] <= CHG_PRICE_MAX else 0.0)
        prob += pc_c[t] + pc_g[t] <= P
        prob += pu[t] + pv[t] <= min(P, P_STATION - act[t])
        prob += pu[t] <= grid_need[t]

    # 充放互斥:同一时刻只能充电(z=1)或放电(z=0)
    z = pulp.LpVariable.dicts("z", idx, cat="Binary")
    for t in idx:
        prob += pc_c[t] + pc_g[t] <= P * z[t]
        prob += pu[t] + pv[t] <= P * (1 - z[t])

    prob += E[0] == E_lo + (pc_c[0] + pc_g[0]) * DT - (pu[0] + pv[0]) / ETA * DT
    for t in range(1, N):
        prob += E[t] == E[t - 1] + (pc_c[t] + pc_g[t]) * DT - (pu[t] + pv[t]) / ETA * DT
    prob += E[N - 1] == E_lo   # 年际稳态(年初=年末=5%)

    # 场站有功功率爬坡约束(15min 相邻变化 ≤ ramp):场站总功率 = 新能源 − 充电 + 放电
    if ramp is not None:
        for t in range(1, N):
            ch = pc_c[t] + pc_g[t]; dis = pu[t] + pv[t]
            ch_p = pc_c[t - 1] + pc_g[t - 1]; dis_p = pu[t - 1] + pv[t - 1]
            g = float(act[t]) - ch + dis
            gp = float(act[t - 1]) - ch_p + dis_p
            prob += g - gp <= ramp
            prob += gp - g <= ramp

    # 充放价差门槛:发电充电 pc_g 有效成本 = 实际价 + ARB_MIN(影子),仅当放电价−充电价 ≥120 才套利;结算仍按真实价。
    prob += pulp.lpSum(DT * (tou[t] * pu[t] + mkt[t] * pv[t] - (mkt[t] + ARB_MIN) * pc_g[t]) for t in idx)
    # gapRel=1%:小功率点求最优性极慢(无 gap 时 >52min),按 1% 相对间隙提前停,目标值即最优可行解(保守)
    prob.solve(pulp.HiGHS(msg=False, gapRel=0.01, timeLimit=120, threads=1))
    obj = pulp.value(prob.objective)
    if obj is None:
        if return_schedule:
            z = np.zeros(N)
            return 0.0, 0.0, (0.0, 0.0, 0.0), dict(cc=z, cg=z, pu=z, pv=z, soc=z)
        return 0.0, 0.0, (0.0, 0.0, 0.0)
    pc = np.array([pc_c[t].value() + pc_g[t].value() for t in idx])
    pc_gen = np.array([pc_g[t].value() for t in idx])
    pd_ = np.array([pu[t].value() + pv[t].value() for t in idx])
    u = np.minimum(pd_, grid_need)
    rev_tou = DT * float((tou * u).sum())
    rev_mkt = DT * float((mkt * (pd_ - u)).sum())
    cost_chg = DT * float((mkt * np.maximum(pc - curt, 0)).sum())
    r_cap = _capacity_comp(df, pd_, pc_gen)
    R_year = rev_tou + rev_mkt - cost_chg   # 真实收益(结算用真实价,不含 ARB_MIN 影子门槛)
    if return_schedule:
        cc = np.array([pc_c[t].value() for t in idx])
        cg = np.array([pc_g[t].value() for t in idx])
        pu_arr = np.array([pu[t].value() for t in idx])
        pv_arr = np.array([pv[t].value() for t in idx])
        soc = np.array([E[t].value() for t in idx])   # 绝对能量(MWh),与方案A/B一致(看板统一 /E_cap 归一化)
        sched = dict(cc=cc, cg=cg, pu=pu_arr, pv=pv_arr, soc=soc)
        return R_year, r_cap, (rev_tou, rev_mkt, cost_chg), sched
    return R_year, r_cap, (rev_tou, rev_mkt, cost_chg)


def simulate(p_h, df):
    """全年仿真一个 (P,H),返回 (R_year, 容量补偿, 收益构成)."""
    days = df.groupby(df["time"].dt.date)
    total = 0.0
    rev_tou, rev_mkt, cost_chg = 0.0, 0.0, 0.0
    pd_all = np.zeros(len(df))
    pc_gen_all = np.zeros(len(df))
    for _, g in days:
        arr = {
            "act": g["act_mw"].values,
            "curt": g["curt"].values,
            "mkt": g["mkt"].values,
            "tou": g["tou"].values,
            "grid_need": g["grid_need"].values,
        }
        obj, pc, pc_gen, pd_ = solve_day(p_h, arr)
        total += obj
        pd_all[g.index.values] = pd_
        pc_gen_all[g.index.values] = pc_gen
        # 收益构成分解(放电抵厂用电 / 套利 / 充电成本)
        u = np.minimum(pd_, g["grid_need"].values)
        rev_tou += DT * (g["tou"].values * u).sum()
        rev_mkt += DT * (g["mkt"].values * (pd_ - u)).sum()
        curt = g["curt"].values
        cost_chg += DT * (g["mkt"].values * np.maximum(pc - curt, 0)).sum()
    r_cap = _capacity_comp(df, pd_all, pc_gen_all)
    return total, r_cap, (rev_tou, rev_mkt, cost_chg)


def _npv(cf, rate):
    """现金流净现值。cf[0] 为第 0 年(期初)投入。"""
    n = len(cf) - 1
    return cf[0] + np.sum(cf[1:] / (1 + rate) ** np.arange(1, n + 1))


def _irr(cf):
    """二分法求内部收益率(单一符号变化现金流,期初投入为负)。"""
    cf = np.asarray(cf, dtype=float)

    def npv_at(r):
        return cf[0] + np.sum(cf[1:] / (1 + r) ** np.arange(1, len(cf)))

    if npv_at(0.0) <= 0:
        return float("nan")          # 不盈利,无正 IRR
    hi = 1.0
    while npv_at(hi) > 0 and hi < 1e6:
        hi *= 2
    if npv_at(hi) > 0:
        return float("nan")
    lo = 0.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if npv_at(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _annuity(principal, rate, term):
    """等额本息年还款额(利率 0 时退化为等额本金)。"""
    if rate <= 0:
        return principal / term
    return principal * rate * (1 + rate) ** term / ((1 + rate) ** term - 1)


def investment_breakdown(P, H, capex_mult=1.0, cost_model=None, discrete=False,
                         cost_params=None):
    """按功率、容量和工程费用驱动因素计算初始投资。

    detailed:标准全口径。包含站内设备、土地/场地、建安、前期、设计监理和预备费;
             不含新建送出线路和主变扩容。
    legacy:  历史四项线性造价,仅用于复核旧结果。
    discrete:按 5MWh 电池舱、2.5MW PCS/升压单元向上取整,用于选定方案工程校核;
             全网格默认使用连续估算,避免设备规格假设主导容量寻优。
    cost_params:项目级造价覆盖字典,键名见 DEFAULT_COST_PARAMS。
    """
    model = COST_MODEL if cost_model is None else str(cost_model)
    if model not in {"detailed", "legacy"}:
        raise ValueError("造价模式必须为 detailed 或 legacy")
    if P <= 0 or H <= 0:
        raise ValueError("储能功率和时长必须大于0")

    params = DEFAULT_COST_PARAMS.copy()
    if cost_params:
        unknown = set(cost_params) - set(params)
        if unknown:
            raise ValueError(f"未知造价参数:{'、'.join(sorted(unknown))}")
        params.update(cost_params)

    E = float(P) * float(H)
    unit_E = float(params["battery_unit_mwh"])
    unit_P = float(params["pcs_unit_mw"])
    priced_E = np.ceil(E / unit_E) * unit_E if discrete else E
    priced_P = np.ceil(float(P) / unit_P) * unit_P if discrete else float(P)

    battery = float(params["battery_per_mwh"]) * priced_E
    land = float(params["land_per_mwh"]) * E
    pcs = float(params["pcs_per_mw"]) * priced_P
    station_equipment = float(params["station_equipment_per_mw"]) * priced_P
    equipment = battery + pcs + station_equipment

    if model == "legacy":
        install = predevelopment = engineering = contingency = 0.0
    else:
        install = equipment * float(params["install_rate"])
        predevelopment = float(params["fixed_predevelopment"])
        engineering = equipment * float(params["engineering_rate"])
        pre_contingency = equipment + land + install + predevelopment + engineering
        contingency = pre_contingency * float(params["contingency_rate"])

    raw = dict(
        battery_system=battery,
        land=land,
        pcs=pcs,
        station_equipment=station_equipment,
        construction_install=install,
        predevelopment=predevelopment,
        management_design_supervision=engineering,
        contingency=contingency,
    )
    components = {name: value * capex_mult for name, value in raw.items()}
    total = sum(components.values())

    # 将间接费分配到可形成经济价值的资产池;前期费单独摊销、期末不回收。
    battery_base = components["battery_system"]
    power_base = components["pcs"] + components["station_equipment"]
    civil_base = components["land"] + components["construction_install"]
    predev_base = components["predevelopment"] + components["management_design_supervision"]
    alloc_base = battery_base + power_base + civil_base + predev_base
    contingency_total = components["contingency"]
    if alloc_base > 0:
        battery_asset = battery_base + contingency_total * battery_base / alloc_base
        power_asset = power_base + contingency_total * power_base / alloc_base
        civil_asset = civil_base + contingency_total * civil_base / alloc_base
        predev_asset = predev_base + contingency_total * predev_base / alloc_base
    else:
        battery_asset = power_asset = civil_asset = predev_asset = 0.0

    return {
        "cost_model": model,
        "discrete": bool(discrete),
        "rated_power_mw": float(P),
        "rated_energy_mwh": E,
        "priced_power_mw": float(priced_P),
        "priced_energy_mwh": float(priced_E),
        "cost_params": params,
        **components,
        "total": total,
        "replacement_battery": components["battery_system"],
        "asset_pools": {
            "battery": battery_asset,
            "power_equipment": power_asset,
            "civil_land": civil_asset,
            "predevelopment": predev_asset,
        },
    }


def _depreciation_schedule(cost, life, years, start=0):
    """直线折旧/摊销计划;start=0 表示第1年开始。"""
    result = np.zeros(years)
    if cost <= 0 or life <= 0 or start >= years:
        return result
    end = min(years, start + int(life))
    result[start:end] = cost / float(life)
    return result


def _remaining_economic_value(cost, economic_life, age, recovery_rate):
    """按剩余经济寿命计算评价期末可回收价值。"""
    if cost <= 0 or economic_life <= 0:
        return 0.0
    remaining = max(0.0, 1.0 - float(age) / float(economic_life))
    return cost * remaining * recovery_rate


def project_finance(P, H, R_year, r_cap, capex_mult=1.0, degr=None,
                    wacc=None, equity_return=None, n_year=None,
                    loan_term=None, replacement_loan_term=None,
                    equity_ratio=None, cost_model=None, discrete_capex=False,
                    cost_params=None):
    """全生命周期现金流(全投资 + 资本金两套口径),返回指标 dict。

    capex_mult / degr / wacc / equity_return / n_year / loan_term / equity_ratio
    为敏感性或项目覆盖参数。cost_model 可选 detailed/legacy;cost_params 可逐项目覆盖造价;
    discrete_capex=True 用设备规格向上取整。equity_ratio=0 用于无资本金、全额贷款情景。

    R_year: 年经营净收益(套利 + 抵厂用电 − 充电成本), 元/年
    r_cap:  容量补偿(元/年,已按逐月尖峰平均放电功率 ×(1−R)×P容量 计算)

    口径(用户拍板):
    - detailed 造价:站内设备+土地/场地+建安+前期+设计监理+预备费;
      不含新建送出线路和主变扩容。legacy 保留历史四项造价用于对照。
    - 项目年限:长周期 20 年、短周期 10 年;长周期项目第 10 年更换电池并追加 capex_rep=电池 50万×E。
    - detailed 分类折旧:电池10年、功率设备10年、土建/场地20年、前期费10年;
      10年项目期末按剩余经济寿命回收功率设备和土建/场地价值,电池残值0。
      legacy 严格保留旧口径:电池及非电池资产均在项目期内折完、期末余值0。
    - 生产费用 = 初始投资 × 1%/年;所得税 25%。
    - 融资:资本金 30%,贷款 70% 等额本息;长周期首次贷款 15 年、短周期首次贷款 7 年;
      更换电池追加投资贷款 7 年。
    """
    n = N_YEAR if n_year is None else int(n_year)
    if loan_term is None:
        loan_term_1 = LOAN_TERM if n > DEP_BATT else LOAN_TERM_SHORT
    else:
        loan_term_1 = int(loan_term)
    loan_term_1 = min(max(1, loan_term_1), n)
    replacement_loan_term_ = LOAN_TERM_REP if replacement_loan_term is None else int(replacement_loan_term)
    eqr = EQUITY_RETURN if equity_return is None else equity_return
    equity_ratio_ = EQUITY_RATIO if equity_ratio is None else float(equity_ratio)
    if not 0.0 <= equity_ratio_ <= 1.0:
        raise ValueError("资本金比例必须在0至1之间")
    if wacc is None:
        wacc_ = (WACC if equity_ratio is None else
                 equity_ratio_ * eqr + (1 - equity_ratio_) * LOAN_RATE * (1 - TAX_RATE))
    else:
        wacc_ = wacc                                       # 折现率覆盖(敏感性)
    dg = DEGR if degr is None else degr                     # 电池年衰减率覆盖(敏感性)
    cost = investment_breakdown(P, H, capex_mult, cost_model, discrete_capex, cost_params)
    model = cost["cost_model"]
    assets = cost["asset_pools"]
    capex_batt = cost["replacement_battery"]                # 长周期项目第10年更换的电池系统
    capex_0 = cost["total"]                                  # 初始投资(元)
    replace = n > DEP_BATT                                   # 项目年限 > 电池折旧年限才换电池
    capex_rep = capex_batt if replace else 0.0               # 第 10 年更换电池追加投资(元)

    # 电池容量逐年线性衰减(DEGR/年);换电池年(索引 DEP_BATT)后重置,老项目(n≤DEP_BATT)不重置。
    if replace:
        cap_factor = np.array([max(0.0, 1 - dg * (y % DEP_BATT)) for y in range(n)])
    else:
        cap_factor = np.array([max(0.0, 1 - dg * y) for y in range(n)])
    revenue = (R_year + r_cap) * cap_factor                  # 逐年经营收入(元/年, 第 1 年满容量)
    om = OM_RATIO * capex_0                                  # 生产费用(元/年)

    # 税务折旧与期末经济余值分开。数组索引 t 对应第 t+1 年。
    if model == "legacy":
        capex_ac = capex_0 - capex_batt
        dep = np.full(n, capex_ac / n)
        if replace:
            dep[:DEP_BATT] += capex_batt / DEP_BATT
            dep[DEP_BATT:] += capex_batt / DEP_BATT
        else:
            dep += capex_batt / n
        terminal_value = 0.0
        terminal_tax_book = 0.0
    else:
        dep_battery = _depreciation_schedule(assets["battery"], TAX_LIFE_BATTERY, n)
        dep_power = _depreciation_schedule(assets["power_equipment"], TAX_LIFE_POWER, n)
        dep_civil = _depreciation_schedule(assets["civil_land"], TAX_LIFE_CIVIL, n)
        dep_predev = _depreciation_schedule(assets["predevelopment"], AMORT_LIFE_PREDEV, n)
        dep = dep_battery + dep_power + dep_civil + dep_predev
        if replace:
            dep += _depreciation_schedule(capex_rep, TAX_LIFE_BATTERY, n, start=DEP_BATT)

        terminal_value = (
            _remaining_economic_value(assets["power_equipment"], ECON_LIFE_POWER, n,
                                      TERMINAL_RECOVERY_POWER)
            + _remaining_economic_value(assets["civil_land"], ECON_LIFE_CIVIL, n,
                                        TERMINAL_RECOVERY_CIVIL)
        )
        terminal_tax_book = max(0.0, assets["battery"] - dep_battery.sum())
        terminal_tax_book += max(0.0, assets["power_equipment"] - dep_power.sum())
        terminal_tax_book += max(0.0, assets["civil_land"] - dep_civil.sum())
        terminal_tax_book += max(0.0, assets["predevelopment"] - dep_predev.sum())
        if replace:
            replacement_dep = _depreciation_schedule(capex_rep, TAX_LIFE_BATTERY, n,
                                                      start=DEP_BATT)
            terminal_tax_book += max(0.0, capex_rep - replacement_dep.sum())

    terminal_tax = max(0.0, terminal_value - terminal_tax_book) * TAX_RATE
    terminal_net = terminal_value - terminal_tax

    # 更换投资现金流:仅换电池项目,第 10 年(索引 9)
    capex_cf = np.zeros(n)
    equity_inj = np.zeros(n)
    if replace:
        capex_cf[DEP_BATT - 1] = capex_rep
        equity_inj[DEP_BATT - 1] = equity_ratio_ * capex_rep  # 第 10 年资本金追加

    # 全投资口径(利息不抵税)
    fi_tax = np.maximum(0, revenue - om - dep) * TAX_RATE
    fi_tax[-1] += terminal_tax
    fi_ncf = revenue - om - fi_tax - capex_cf
    fi_ncf[-1] += terminal_value
    fi_cf = np.concatenate([[-capex_0], fi_ncf])

    # 资本金口径(利息抵税):首次贷款按项目口径 15/7 年;更换电池贷款 7 年。
    loan1 = (1 - equity_ratio_) * capex_0
    interest = np.zeros(n)
    principal = np.zeros(n)
    a1 = _annuity(loan1, LOAN_RATE, loan_term_1)
    bal1 = loan1
    for t in range(loan_term_1):
        interest[t] += bal1 * LOAN_RATE
        principal[t] += a1 - bal1 * LOAN_RATE
        bal1 = max(0.0, bal1 - (a1 - bal1 * LOAN_RATE))
    if replace:
        loan2 = (1 - equity_ratio_) * capex_rep
        term2 = min(max(1, replacement_loan_term_), n - DEP_BATT)
        a2 = _annuity(loan2, LOAN_RATE, term2)
        bal2 = loan2
        for t in range(DEP_BATT, DEP_BATT + term2):         # 更换次年起连续偿还 term2 年
            interest[t] += bal2 * LOAN_RATE
            principal[t] += a2 - bal2 * LOAN_RATE
            bal2 = max(0.0, bal2 - (a2 - bal2 * LOAN_RATE))

    equity_cf0 = equity_ratio_ * capex_0
    eq_tax = np.maximum(0, revenue - om - dep - interest) * TAX_RATE
    eq_tax[-1] += terminal_tax
    eq_ncf = revenue - om - eq_tax - (principal + interest) - equity_inj
    eq_ncf[-1] += terminal_value
    eq_cf = np.concatenate([[-equity_cf0], eq_ncf])

    yearly = []
    for y in range(n):
        yearly.append(dict(y=y + 1, cap_factor=round(float(cap_factor[y]), 6),
                           revenue=float(revenue[y]), om=float(om), dep=float(dep[y]),
                           fi_tax=float(fi_tax[y]), fi_ncf=float(fi_ncf[y]),
                           interest=float(interest[y]), principal=float(principal[y]),
                           eq_tax=float(eq_tax[y]), eq_ncf=float(eq_ncf[y]),
                           capex_cf=float(capex_cf[y]), equity_inj=float(equity_inj[y]),
                           terminal_value=float(terminal_value if y == n - 1 else 0.0),
                           terminal_tax=float(terminal_tax if y == n - 1 else 0.0)))
    eq_irr = float("nan") if equity_cf0 == 0 else _irr(eq_cf)
    return dict(capex=capex_0, capex_rep=capex_rep, r_cap=r_cap, revenue=R_year + r_cap, om=om,
                fi_npv=_npv(fi_cf, wacc_), fi_irr=_irr(fi_cf),
                eq_npv=_npv(eq_cf, eqr), eq_irr=eq_irr, yearly=yearly,
                equity_ratio=equity_ratio_, wacc=wacc_, loan1=loan1,
                loan2=((1 - equity_ratio_) * capex_rep if replace else 0.0),
                cost_model=model, capex_breakdown=cost,
                terminal_value=terminal_value, terminal_tax_book=terminal_tax_book,
                terminal_tax=terminal_tax, terminal_net=terminal_net)


def search(df, p_range, h_set, simulate_fn=simulate):
    """网格搜索,返回 DataFrame 每格一行,及最优 (P,H)(按全投资 NPV 排序)。

    simulate_fn 可选 solve_year(全年耦合/跨日结转)或 simulate(逐日独立/日末归 5%)。
    """
    rows = []
    for P in p_range:
        for H in h_set:
            R_year, r_cap, _ = simulate_fn((P, H), df)
            f = project_finance(P, H, R_year, r_cap)
            rows.append(dict(P=P, H=H, E=P * H,
                             R_year=R_year / 1e4, R_cap=f["r_cap"] / 1e4,
                             capex=f["capex"] / 1e4,
                             fi_npv=f["fi_npv"] / 1e4, fi_irr=f["fi_irr"],
                             eq_npv=f["eq_npv"] / 1e4, eq_irr=f["eq_irr"]))
            print(f"  P={P:6.1f} H={H}  R_year={R_year/1e4:8.1f}万 "
                  f"容量补偿={f['r_cap']/1e4:7.1f}万 造价={f['capex']/1e4:6.0f}万 "
                  f"全投资NPV={f['fi_npv']/1e4:8.1f}万 IRR={f['fi_irr']*100:5.1f}% "
                  f"资本金NPV={f['eq_npv']/1e4:8.1f}万 IRR={f['eq_irr']*100:5.1f}%", flush=True)
    res = pd.DataFrame(rows)
    best = res.loc[res["fi_npv"].idxmax()]
    return res, best


if __name__ == "__main__":
    df = load_data()
    print(f"数据: {len(df)} 点, {df['time'].dt.date.nunique()} 天, "
          f"时间范围 {df['time'].min().date()} ~ {df['time'].max().date()}")
    print(f"限电时段占比 {100*(df['curt']>0).mean():.1f}%, 尖峰时段点数 {df['is_sharp'].sum()}")
    print(f"WACC={WACC*100:.2f}%  资本金比例={EQUITY_RATIO:.0%}  贷款利率={LOAN_RATE:.1%}  "
          f"折旧=电池{TAX_LIFE_BATTERY}/功率设备{TAX_LIFE_POWER}/土建场地{TAX_LIFE_CIVIL}年  "
          f"生产费用={OM_RATIO:.1%}初始投资/年  "
          f"所得税={TAX_RATE:.0%}  限电价={CURT_PRICE:.0f}")

    # 全年耦合(跨日结转)搜索——上界口径
    print("\n=== 全年耦合搜索(允许跨日结转) ===")
    res, best = search(df, p_range=[20, 60, 100, 140, 180, 220, 260, 300, 340, 380, 420],
                       h_set=[2, 4, 6, 8], simulate_fn=solve_year)

    print("\n=== 最优(全年耦合) ===")
    print(best.to_string())
    print(f"\n最优 P={best['P']:.1f} MW, H={best['H']} h, 装机 {best['E']:.0f} MWh, "
          f"全投资NPV={best['fi_npv']:.1f} 万, 全投资IRR={best['fi_irr']*100:.1f}%, "
          f"资本金IRR={best['eq_irr']*100:.1f}%")
