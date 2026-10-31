"""投资分析报告插图；图表字体随代码提供，云端无需安装系统中文字体。"""
from __future__ import annotations

import io
import threading
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text


FONT = Path(__file__).parent / "assets/report-chart-font.ttf"
GREEN, GOLD, TERRA, SAGE = "#105848", "#e8ad3c", "#a04028", "#7aa08f"
_PLOT_LOCK = threading.Lock()


def _save(fig):
    prop = FontProperties(fname=str(FONT), size=10)
    for text in fig.findobj(Text):
        text.set_fontproperties(prop)
    for ax in fig.axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e7e4db", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0.12, 1, 1), pad=1.4)
    buffer = io.BytesIO()
    FigureCanvasAgg(fig).print_png(buffer)
    return buffer.getvalue()


def build_charts(detail, financing, stresses, comparison):
    with _PLOT_LOCK:
        return _build_charts(detail, financing, stresses, comparison)


def _build_charts(detail, financing, stresses, comparison):
    charts = {}
    profile = detail["average_profile"]
    fig = Figure(figsize=(8, 3.7), dpi=180)
    ax = fig.subplots()
    power = profile["储能功率(MW)"].to_numpy()
    x = np.arange(len(profile))
    ax.bar(x, power, color=np.where(power >= 0, GOLD, GREEN), width=0.85,
           label="储能功率：充电为负、放电为正")
    ax.axhline(0, color="#89928a", linewidth=0.7)
    ax.set_ylabel("储能功率（MW）")
    ticks = np.arange(0, len(profile), 12)
    ax.set_xticks(ticks, profile["时刻"].iloc[ticks])
    ax.set_xlabel("一天96个15分钟点（全年同一时刻均值）")
    right = ax.twinx()
    right.plot(x, profile["实时电价(元/MWh)"], color=TERRA, linewidth=1.8, label="实时市场电价")
    right.set_ylabel("电价（元/MWh）")
    handles, labels = ax.get_legend_handles_labels()
    h2, l2 = right.get_legend_handles_labels()
    fig.legend(handles + h2, labels + l2, loc="lower center", ncol=2, frameon=False)
    charts["average"] = _save(fig)

    m = detail["metrics"]
    values = [m["rev_tou_wan"], m["rev_market_wan"], -m["charging_cost_wan"], m["R_cap"]]
    labels = ["抵厂用电收益", "市场放电收入", "充电成本（扣减）", "容量补偿"]
    fig = Figure(figsize=(8, 3.4), dpi=180)
    ax = fig.subplots()
    ax.barh(labels, values, color=[GREEN, GOLD, TERRA, SAGE])
    ax.axvline(0, color="#89928a", linewidth=0.7)
    ax.set_xlabel("首年金额（万元）")
    for y, v in enumerate(values):
        ax.annotate(f"{v:,.1f}", (v, y), xytext=(5 if v >= 0 else -5, 0),
                    textcoords="offset points", ha="left" if v >= 0 else "right", va="center")
    ax.margins(x=0.2)
    charts["revenue"] = _save(fig)

    schedule = detail["schedule"]
    monthly = schedule.groupby(["年", "月"])[["执行充电功率(MW)", "执行放电功率(MW)"]].sum() * 0.25
    fig = Figure(figsize=(8, 3.6), dpi=180)
    ax = fig.subplots()
    x = np.arange(len(monthly))
    ax.bar(x - 0.18, monthly.iloc[:, 0], width=0.36, color=GREEN, label="充电量")
    ax.bar(x + 0.18, monthly.iloc[:, 1], width=0.36, color=GOLD, label="放电量")
    ax.set_xticks(x, [f"{y}-{month:02d}" for y, month in monthly.index], rotation=45, ha="right")
    ax.set_ylabel("电量（MWh）")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
    charts["monthly"] = _save(fig)

    fig = Figure(figsize=(8, 3.7), dpi=180)
    ax = fig.subplots()
    ax.bar(financing["year"], financing["equity_ncf_wan"], color=GREEN, label="资本金净现金流")
    ax.plot(financing["year"], financing["debt_service_wan"], color=GOLD, label="贷款本息")
    ax.set_xlabel("运营年份")
    ax.set_ylabel("年度金额（万元）")
    right = ax.twinx()
    right.plot(financing["year"], financing["cumulative_wan"], color=TERRA, label="累计资本金现金流")
    right.set_ylabel("累计金额（万元）")
    handles, labels = ax.get_legend_handles_labels()
    h2, l2 = right.get_legend_handles_labels()
    fig.legend(handles + h2, labels + l2, loc="lower center", ncol=3, frameon=False)
    charts["cash"] = _save(fig)

    fig = Figure(figsize=(8, 4.0), dpi=180)
    ax = fig.subplots()
    x = np.arange(len(stresses))
    ax.barh(x - 0.18, stresses["全投资NPV"], height=0.34, color=GOLD, label="全投资净现值")
    ax.barh(x + 0.18, stresses["资本金NPV"], height=0.34, color=GREEN, label="资本金净现值")
    ax.set_yticks(x, stresses["情景"])
    ax.invert_yaxis()
    ax.axvline(0, color="#89928a", linewidth=0.7)
    ax.set_xlabel("净现值（万元）")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
    charts["sensitivity"] = _save(fig)

    if len(comparison) > 1:
        fig = Figure(figsize=(8, 3.6), dpi=180)
        ax = fig.subplots()
        ax.plot(comparison["P"], comparison["total_revenue"], color=GOLD, marker="o", label="首年总收入")
        ax.set_xlabel("同一持续时长下的储能功率（MW）")
        ax.set_ylabel("首年总收入（万元）")
        right = ax.twinx()
        right.plot(comparison["P"], comparison["eq_irr"], color=GREEN, marker="s", label="资本金内部收益率")
        right.set_ylabel("资本金内部收益率（%）")
        handles, labels = ax.get_legend_handles_labels()
        h2, l2 = right.get_legend_handles_labels()
        fig.legend(handles + h2, labels + l2, loc="lower center", ncol=2, frameon=False)
        charts["capacity"] = _save(fig)
    return charts
