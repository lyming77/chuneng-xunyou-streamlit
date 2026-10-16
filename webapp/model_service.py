from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "脚本"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import 寻优 as M
import 寻优_多项目 as MP
from 方案B_滚动MPC import solve_mpc


REQUIRED_COLUMNS = [
    "时间戳",
    "实际发电功率(MW)",
    "超短期预测功率(MW)",
    "预测发电功率(日前,MW)",
    "实时市场价格(元/MWh)",
    "预测价格(元/MWh)",
]
ALL_COLUMNS = REQUIRED_COLUMNS[:3] + ["理论功率(MW)"] + REQUIRED_COLUMNS[3:]
ALIASES = {
    "时间": "时间戳",
    "实际发电(MW)": "实际发电功率(MW)",
    "实际功率(MW)": "实际发电功率(MW)",
    "可用功率(MW)": "超短期预测功率(MW)",
    "日前预测功率(MW)": "预测发电功率(日前,MW)",
    "预测发电功率(MW)": "预测发电功率(日前,MW)",
    "实时电价(元/MWh)": "实时市场价格(元/MWh)",
    "实际电价(元/MWh)": "实时市场价格(元/MWh)",
    "日前预测电价(元/MWh)": "预测价格(元/MWh)",
    "预测电价(元/MWh)": "预测价格(元/MWh)",
}

DEFAULT_FLAT = {
    1: 721.57, 2: 705.77, 3: 656.87, 4: 653.87, 5: 665.17, 6: 650.36,
    7: 691.67, 8: 638.77, 9: 665.57, 10: 730.87, 11: 786.77, 12: 749.26,
}
DEFAULT_COEF = {"尖峰": 2.0, "高峰": 1.7, "平段": 1.0, "低谷": 0.3, "深谷": 0.1}
DEFAULT_ZONE_TEXT = {
    "1-2月、12月": {"尖峰": "16-19", "高峰": "7-9,19-21", "低谷": "2-6,10-11,14-15", "深谷": "11-14"},
    "3-5月": {"尖峰": "17-20", "高峰": "20-22", "低谷": "10-11,14-15", "深谷": "11-14"},
    "6月": {"尖峰": "17-22", "高峰": "16-17,22-23", "低谷": "7-12", "深谷": "—"},
    "7-8月": {"尖峰": "17-22", "高峰": "16-17,22-23", "低谷": "1-6", "深谷": "—"},
    "9-11月": {"尖峰": "17-19", "高峰": "16-17,19-21", "低谷": "10-11,14-15", "深谷": "11-14"},
}

_MODEL_LOCK = threading.RLock()


def default_config() -> dict:
    return {
        "project_name": "新能源项目",
        "project_type": "风电",
        "station_power_mw": 100.0,
        "aux_power_mw": 0.0,
        "mechanism_ratio_pct": 80.0,
        "capacity_compensation_yuan_per_mw_month": 30000.0,
        "soc_min_pct": 5.0,
        "soc_max_pct": 98.0,
        "efficiency_pct": 93.0,
        "minimum_spread_yuan_per_mwh": 120.0,
        "maximum_charge_price_yuan_per_mwh": 250.0,
        "curtailment_price_yuan_per_mwh": -80.0,
        "project_years": 20,
        "loan_term_years": 15,
        "replacement_loan_term_years": 7,
        "equity_ratio_pct": 30.0,
        "loan_rate_pct": 3.0,
        "equity_discount_rate_pct": 8.0,
        "battery_degradation_pct": 2.0,
        "om_ratio_pct": 1.0,
        "income_tax_pct": 25.0,
        "battery_cost_wan_per_mwh": 50.0,
        "land_cost_wan_per_mwh": 3.5,
        "pcs_cost_wan_per_mw": 27.0,
        "station_equipment_cost_wan_per_mw": 20.0,
        "installation_rate_pct": 7.766,
        "include_land_cost": False,
        "include_predevelopment_cost": False,
        "include_engineering_cost": False,
        "predevelopment_cost_wan": 150.0,
        "engineering_rate_pct": 5.065,
        "contingency_rate_pct": 3.0,
        "battery_unit_mwh": 5.0,
        "pcs_unit_mw": 2.5,
        "flat_prices": copy.deepcopy(DEFAULT_FLAT),
        "tou_coefficients": copy.deepcopy(DEFAULT_COEF),
        "zone_text": copy.deepcopy(DEFAULT_ZONE_TEXT),
    }


def config_signature(config: dict, data_bytes: bytes) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, default=float).encode("utf-8")
    return hashlib.sha256(data_bytes + payload).hexdigest()


def upload_template_csv(start: str = "2026-01-01") -> bytes:
    timestamps = pd.date_range(start, periods=365 * 96, freq="15min")
    frame = pd.DataFrame({"时间戳": timestamps.strftime("%Y-%m-%d %H:%M")})
    for column in ALL_COLUMNS[1:]:
        frame[column] = ""
    return frame.to_csv(index=False).encode("utf-8-sig")


def _read_csv(data: bytes) -> pd.DataFrame:
    error = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=encoding)
        except UnicodeDecodeError as exc:
            error = exc
    raise ValueError(f"CSV编码无法识别：{error}")


def read_uploaded_timeseries(data: bytes, filename: str) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        raw = _read_csv(data)
    elif suffix in {".xlsx", ".xlsm"}:
        excel = pd.ExcelFile(io.BytesIO(data))
        sheet = "时序数据" if "时序数据" in excel.sheet_names else excel.sheet_names[0]
        raw = pd.read_excel(excel, sheet_name=sheet)
    else:
        raise ValueError("仅支持CSV或XLSX文件")
    return normalize_timeseries(raw, require_full_year=True)


def normalize_timeseries(raw: pd.DataFrame, require_full_year: bool = True) -> pd.DataFrame:
    frame = raw.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.rename(columns={key: value for key, value in ALIASES.items() if key in frame.columns})
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError("上传文件缺少字段：" + "、".join(missing))
    if "理论功率(MW)" not in frame.columns:
        frame["理论功率(MW)"] = np.nan
    frame = frame[ALL_COLUMNS].copy()
    frame["时间戳"] = pd.to_datetime(frame["时间戳"], errors="coerce")
    if frame["时间戳"].isna().any():
        raise ValueError(f"时间戳存在{int(frame['时间戳'].isna().sum())}个无法识别的值")
    frame = frame.sort_values("时间戳").reset_index(drop=True)
    if frame["时间戳"].duplicated().any():
        raise ValueError("时间戳存在重复值")
    interval = frame["时间戳"].diff().dt.total_seconds().iloc[1:]
    if not interval.eq(900).all():
        bad = int((interval != 900).sum())
        raise ValueError(f"时间戳不是连续15分钟序列，共{bad}处断点")
    if require_full_year:
        allowed = {365 * 96, 366 * 96}
        if len(frame) not in allowed:
            raise ValueError(f"时序数据为{len(frame)}行；完整年度应为35040行，闰年可为35136行")
        counts = frame.groupby(frame["时间戳"].dt.date).size()
        if not counts.eq(96).all():
            raise ValueError("部分日期不是96个15分钟点")
    numeric_columns = ALL_COLUMNS[1:]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    price_columns = ["实时市场价格(元/MWh)", "预测价格(元/MWh)"]
    bad_prices = {column: int(frame[column].isna().sum()) for column in price_columns
                  if frame[column].isna().any()}
    if bad_prices:
        detail = "、".join(f"{column}缺{count}点" for column, count in bad_prices.items())
        raise ValueError("电价字段存在空值或非数字：" + detail)

    frame, fill_stats = MP.fill_power(frame)
    power_columns = ["实际发电功率(MW)", "超短期预测功率(MW)", "预测发电功率(日前,MW)"]
    unfilled = {column: int(frame[column].isna().sum()) for column in power_columns
                if frame[column].isna().any()}
    if unfilled:
        detail = "、".join(f"{column}仍缺{count}点" for column, count in unfilled.items())
        raise ValueError("同一时点的三个功率字段均缺失，无法交叉补全：" + detail)
    for column in power_columns:
        frame[column] = frame[column].clip(lower=0.0)
    frame.attrs["power_fill_stats"] = fill_stats
    return frame


def parse_ranges(value: str) -> list[list[float]]:
    if value in (None, "", "—", "-"):
        return []
    ranges = []
    for segment in str(value).replace("，", ",").split(","):
        segment = segment.strip()
        if not segment:
            continue
        parts = segment.split("-")
        if len(parts) != 2:
            raise ValueError(f"分时区间格式错误：{segment}")
        start, end = map(float, parts)
        if not 0 <= start < end <= 24:
            raise ValueError(f"分时区间超出0—24小时：{segment}")
        ranges.append([start, end])
    return ranges


def build_zones(zone_text: dict) -> dict:
    return {
        group: {category: parse_ranges(value) for category, value in categories.items()}
        for group, categories in zone_text.items()
    }


def build_model_frame(timeseries: pd.DataFrame, config: dict) -> pd.DataFrame:
    params = {
        "station": {"厂用电功率": float(config["aux_power_mw"])},
        "flat": {int(month): float(value) for month, value in config["flat_prices"].items()},
        "coef": {key: float(value) for key, value in config["tou_coefficients"].items()},
        "zones": build_zones(config["zone_text"]),
    }
    return MP.build_df(timeseries, params)


def cost_params(config: dict) -> dict:
    return {
        "battery_per_mwh": float(config["battery_cost_wan_per_mwh"]) * 1e4,
        "land_per_mwh": (float(config["land_cost_wan_per_mwh"]) * 1e4
                         if config.get("include_land_cost", False) else 0.0),
        "pcs_per_mw": float(config["pcs_cost_wan_per_mw"]) * 1e4,
        "station_equipment_per_mw": float(config["station_equipment_cost_wan_per_mw"]) * 1e4,
        "fixed_predevelopment": (float(config["predevelopment_cost_wan"]) * 1e4
                                 if config.get("include_predevelopment_cost", False) else 0.0),
        "install_rate": float(config["installation_rate_pct"]) / 100,
        "engineering_rate": (float(config["engineering_rate_pct"]) / 100
                             if config.get("include_engineering_cost", False) else 0.0),
        "contingency_rate": float(config["contingency_rate_pct"]) / 100,
        "battery_unit_mwh": float(config["battery_unit_mwh"]),
        "pcs_unit_mw": float(config["pcs_unit_mw"]),
    }


def cost_benchmark(config: dict, power_mw: float = 10.0, hours: float = 2.0) -> dict:
    """按给定功率和时长计算连续估算造价及元/W、元/Wh指标。"""
    breakdown = M.investment_breakdown(
        float(power_mw), float(hours), cost_model="detailed", discrete=False,
        cost_params=cost_params(config))
    total = float(breakdown["total"])
    energy_mwh = float(power_mw) * float(hours)
    return {
        "power_mw": float(power_mw),
        "hours": float(hours),
        "energy_mwh": energy_mwh,
        "total_wan": total / 1e4,
        "yuan_per_w": total / (float(power_mw) * 1e6),
        "yuan_per_wh": total / (energy_mwh * 1e6),
        "breakdown": breakdown,
    }


@contextmanager
def temporary_model_settings(config: dict):
    names = [
        "SOC_MIN", "SOC_MAX", "ETA", "ARB_MIN", "CHG_PRICE_MAX", "CURT_PRICE",
        "R_MECH", "P_CAP", "EQUITY_RATIO", "LOAN_RATE", "EQUITY_RETURN", "DEGR",
        "OM_RATIO", "TAX_RATE", "WACC", "LOAN_TERM_REP",
    ]
    previous = {name: getattr(M, name) for name in names}
    equity_ratio = float(config["equity_ratio_pct"]) / 100
    loan_rate = float(config["loan_rate_pct"]) / 100
    equity_return = float(config["equity_discount_rate_pct"]) / 100
    tax_rate = float(config["income_tax_pct"]) / 100
    values = {
        "SOC_MIN": float(config["soc_min_pct"]) / 100,
        "SOC_MAX": float(config["soc_max_pct"]) / 100,
        "ETA": float(config["efficiency_pct"]) / 100,
        "ARB_MIN": float(config["minimum_spread_yuan_per_mwh"]),
        "CHG_PRICE_MAX": float(config["maximum_charge_price_yuan_per_mwh"]),
        "CURT_PRICE": float(config["curtailment_price_yuan_per_mwh"]),
        "R_MECH": float(config["mechanism_ratio_pct"]) / 100,
        "P_CAP": float(config["capacity_compensation_yuan_per_mw_month"]),
        "EQUITY_RATIO": equity_ratio,
        "LOAN_RATE": loan_rate,
        "EQUITY_RETURN": equity_return,
        "DEGR": float(config["battery_degradation_pct"]) / 100,
        "OM_RATIO": float(config["om_ratio_pct"]) / 100,
        "TAX_RATE": tax_rate,
        "WACC": equity_ratio * equity_return + (1 - equity_ratio) * loan_rate * (1 - tax_rate),
        "LOAN_TERM_REP": int(config["replacement_loan_term_years"]),
    }
    try:
        for name, value in values.items():
            setattr(M, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(M, name, value)


def make_power_grid(station_power: float, project_type: str, maximum_ratio_pct: float | None = None) -> list[float]:
    default_ratio = 50.0 if project_type == "光伏" else 30.0
    ratio = default_ratio if maximum_ratio_pct is None else float(maximum_ratio_pct)
    maximum = max(1.0, float(station_power) * ratio / 100)
    values = [float(value) for value in range(1, min(5, int(math.floor(maximum))) + 1)]
    if maximum >= 10:
        values.extend(float(value) for value in range(10, int(math.floor(maximum / 5)) * 5 + 1, 5))
    if not values:
        values = [round(maximum, 3)]
    return sorted(set(values))


def _finance(P: float, H: int, annual_revenue: float, capacity_revenue: float, config: dict) -> dict:
    return M.project_finance(
        P, H, annual_revenue, capacity_revenue,
        n_year=int(config["project_years"]),
        loan_term=int(config["loan_term_years"]),
        replacement_loan_term=int(config["replacement_loan_term_years"]),
        equity_ratio=float(config["equity_ratio_pct"]) / 100,
        equity_return=float(config["equity_discount_rate_pct"]) / 100,
        cost_params=cost_params(config),
    )


def _result_row(P: float, H: int, frame: pd.DataFrame, config: dict,
                day_progress: Callable[[int, int, object], None] | None = None) -> tuple[dict, dict]:
    annual_revenue, capacity_revenue, parts, schedule = solve_mpc(
        P, H, frame, return_schedule=True,
        P_STATION=float(config["station_power_mw"]), ramp=None,
        day_progress=day_progress)
    charge = np.asarray(schedule["cc"]) + np.asarray(schedule["cg"])
    discharge = np.asarray(schedule["pu"]) + np.asarray(schedule["pv"])
    energy = P * H
    finance = _finance(P, H, annual_revenue, capacity_revenue, config)
    row = {
        "P": float(P), "H": int(H), "E": float(energy),
        "annual_cycles": float(charge.sum() * M.DT / energy),
        "charge_mwh": float(charge.sum() * M.DT),
        "discharge_mwh": float(discharge.sum() * M.DT),
        "free_charge_mwh": float(np.asarray(schedule["cc"]).sum() * M.DT),
        "paid_charge_mwh": float(np.asarray(schedule["cg"]).sum() * M.DT),
        "R_year": annual_revenue / 1e4,
        "R_cap": capacity_revenue / 1e4,
        "total_revenue": (annual_revenue + capacity_revenue) / 1e4,
        "capex": finance["capex"] / 1e4,
        "capex_rep": finance["capex_rep"] / 1e4,
        "fi_irr": finance["fi_irr"] * 100,
        "fi_npv": finance["fi_npv"] / 1e4,
        "eq_irr": finance["eq_irr"] * 100,
        "eq_npv": finance["eq_npv"] / 1e4,
    }
    detail = {
        "annual_revenue": annual_revenue,
        "capacity_revenue": capacity_revenue,
        "parts": parts,
        "schedule": schedule,
        "finance": finance,
    }
    return row, detail


def best_valid_row(frame: pd.DataFrame, column: str) -> pd.Series | None:
    """返回指定指标的最大有效值所在行；整列为空或无穷时返回 None。"""
    if frame.empty or column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if not values.notna().any():
        return None
    return frame.loc[values.idxmax()]


def run_grid(frame: pd.DataFrame, config: dict, powers: list[float], hours: list[int],
             progress: Callable[[float, int, str], None] | None = None,
             partial_result: Callable[[pd.DataFrame], None] | None = None) -> pd.DataFrame:
    combinations = [(float(power), int(duration)) for power in powers for duration in hours]
    rows = []
    with _MODEL_LOCK, temporary_model_settings(config):
        for index, (power, duration) in enumerate(combinations, 1):
            label = f"{power:g}MW/{power * duration:g}MWh"
            if progress:
                progress(index - 1, len(combinations), f"{label}｜准备计算")
            def update_day(done, total_days, date):
                if progress:
                    overall = index - 1 + done / total_days
                    progress(overall, len(combinations),
                             f"{label}｜{date}｜{done}/{total_days}天")
            try:
                row, _ = _result_row(power, duration, frame, config, update_day)
                row["error"] = ""
            except Exception as exc:
                row = {"P": power, "H": duration, "E": power * duration, "error": str(exc)}
            rows.append(row)
            if partial_result:
                partial_result(pd.DataFrame(rows))
            if progress:
                progress(index, len(combinations), label)
    result = pd.DataFrame(rows)
    for column in ["annual_cycles", "charge_mwh", "discharge_mwh", "free_charge_mwh",
                   "paid_charge_mwh", "R_year", "R_cap", "total_revenue", "capex",
                   "capex_rep", "fi_irr", "fi_npv", "eq_irr", "eq_npv"]:
        if column not in result:
            result[column] = np.nan
    return result


def run_configuration(frame: pd.DataFrame, config: dict, power: float, hours: int) -> dict:
    with _MODEL_LOCK, temporary_model_settings(config):
        row, detail = _result_row(float(power), int(hours), frame, config)
        return build_detail(frame, config, row, detail)


def build_detail(frame: pd.DataFrame, config: dict, row: dict, detail: dict) -> dict:
    power, hours, energy = row["P"], row["H"], row["E"]
    schedule = detail["schedule"]
    free_charge = np.asarray(schedule["cc"], dtype=float)
    paid_charge = np.asarray(schedule["cg"], dtype=float)
    utility_discharge = np.asarray(schedule["pu"], dtype=float)
    market_discharge = np.asarray(schedule["pv"], dtype=float)
    charge = free_charge + paid_charge
    discharge = utility_discharge + market_discharge
    actual = np.maximum(frame["act_mw"].to_numpy(dtype=float), 0.0)
    curtailed = np.maximum(frame["curt"].to_numpy(dtype=float), 0.0)
    price = frame["mkt"].to_numpy(dtype=float)
    curtailed_charge = np.minimum(free_charge, curtailed)
    supplemental_charge = np.maximum(0.0, free_charge - curtailed_charge)
    actual_generation = float(actual.sum() * M.DT)
    curtailed_energy = float(curtailed.sum() * M.DT)
    paid_charge_energy = float(paid_charge.sum() * M.DT)
    discharge_energy = float(discharge.sum() * M.DT)
    post_export = actual_generation - paid_charge_energy + discharge_energy
    base_market_revenue_yuan = float((actual * price).sum() * M.DT)
    rev_tou, rev_market, charging_cost = map(float, detail["parts"])
    market_revenue_after_yuan = base_market_revenue_yuan + rev_market - charging_cost
    finance = detail["finance"]
    metrics = {
        **row,
        "project_name": config["project_name"],
        "project_type": config["project_type"],
        "station_power_mw": float(config["station_power_mw"]),
        "points": len(frame),
        "start": pd.Timestamp(frame["time"].iloc[0]),
        "end": pd.Timestamp(frame["time"].iloc[-1]),
        "actual_generation_mwh": actual_generation,
        "curtailed_energy_mwh": curtailed_energy,
        "curtailed_charge_mwh": float(curtailed_charge.sum() * M.DT),
        "supplemental_charge_mwh": float(supplemental_charge.sum() * M.DT),
        "paid_charge_mwh": paid_charge_energy,
        "charge_mwh": float(charge.sum() * M.DT),
        "discharge_mwh": discharge_energy,
        "post_export_mwh": post_export,
        "curtailment_capture_pct": (float(curtailed_charge.sum() * M.DT) / curtailed_energy * 100
                                     if curtailed_energy else 0.0),
        "market_price_before": base_market_revenue_yuan / actual_generation if actual_generation else np.nan,
        "market_price_after": market_revenue_after_yuan / post_export if post_export else np.nan,
        "rev_tou_wan": rev_tou / 1e4,
        "rev_market_wan": rev_market / 1e4,
        "charging_cost_wan": charging_cost / 1e4,
        "average_paid_charge_price": charging_cost / paid_charge_energy if paid_charge_energy else np.nan,
        "average_discharge_value": ((rev_tou + rev_market) / discharge_energy if discharge_energy else np.nan),
        "terminal_net_wan": finance["terminal_net"] / 1e4,
    }
    time = frame["time"].dt
    schedule_frame = pd.DataFrame({
        "年": time.year,
        "月": time.month,
        "日": time.day,
        "时刻": time.strftime("%H:%M"),
        "预测发电功率(MW)": frame["forecast_mw"].to_numpy(),
        "实际发电功率(MW)": frame["act_mw"].to_numpy(),
        "可用功率(MW)": frame["avail"].to_numpy(),
        "弃电功率(MW)": frame["curt"].to_numpy(),
        "免费充电功率(MW)": free_charge,
        "发电充电功率(MW)": paid_charge,
        "执行充电功率(MW)": charge,
        "执行放电功率(MW)": discharge,
        "预测电价(元/MWh)": frame["mkt_fc"].to_numpy(),
        "实时电价(元/MWh)": price,
        "SOC(%)": np.asarray(schedule["soc"]) / energy * 100,
    })
    slots = np.arange(len(frame)) % 96
    average_profile = pd.DataFrame({
        "点": slots,
        "时刻": [f"{slot // 4:02d}:{(slot % 4) * 15:02d}" for slot in slots],
        "储能功率(MW)": discharge - charge,
        "实时电价(元/MWh)": price,
    }).groupby("点", as_index=False).agg({
        "时刻": "first", "储能功率(MW)": "mean", "实时电价(元/MWh)": "mean"})
    return {
        "metrics": metrics,
        "schedule": schedule_frame,
        "average_profile": average_profile,
        "finance": finance,
    }


def grid_csv_bytes(grid: pd.DataFrame) -> bytes:
    return grid.to_csv(index=False).encode("utf-8-sig")


def schedule_excel_bytes(detail: dict) -> bytes:
    output = io.BytesIO()
    metrics = detail["metrics"]
    summary = pd.DataFrame([
        ["项目", metrics["project_name"]],
        ["储能功率(MW)", metrics["P"]],
        ["储能容量(MWh)", metrics["E"]],
        ["全年充电量(MWh)", metrics["charge_mwh"]],
        ["全年放电量(MWh)", metrics["discharge_mwh"]],
        ["等效满循环次数", metrics["annual_cycles"]],
        ["首年总收入(万元)", metrics["total_revenue"]],
        ["资本金IRR(%)", metrics["eq_irr"]],
    ], columns=["指标", "数值"])
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="全年汇总", index=False)
        detail["schedule"].to_excel(writer, sheet_name="逐15分钟调度", index=False)
        pd.DataFrame(detail["finance"]["yearly"]).to_excel(writer, sheet_name="财务现金流", index=False)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for column in sheet.columns:
                sheet.column_dimensions[column[0].column_letter].width = min(26, max(12, len(str(column[0].value or "")) + 4))
    return output.getvalue()
