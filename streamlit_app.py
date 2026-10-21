from __future__ import annotations

import math
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from webapp.model_service import (
    DEFAULT_COEF,
    DEFAULT_FLAT,
    DEFAULT_ZONE_TEXT,
    best_valid_row,
    build_model_frame,
    config_signature,
    cost_benchmark,
    default_config,
    grid_csv_bytes,
    make_power_grid,
    read_uploaded_timeseries,
    run_configuration,
    run_grid,
    schedule_excel_bytes,
    upload_template_csv,
)
from webapp.report_builder import build_report_bytes


st.set_page_config(
    page_title="新能源配储寻优平台",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --green:#105848; --gold:#e8ad3c; --cream:#f3f0e8; --ink:#1c2a22; }
    .stApp { background:#f3f0e8; color:var(--ink); }
    [data-testid="stHeader"] { background:rgba(243,240,232,.92); }
    .hero { background:#105848; color:white; border-radius:18px; padding:28px 34px; margin-bottom:16px; }
    .hero h1 { margin:0 0 6px; font-size:2.15rem; letter-spacing:.03em; }
    .hero p { margin:0; color:#dcebe4; }
    .workflow-note { background:#fff8e6; border:1px solid #ead9aa; border-radius:10px;
                     padding:10px 14px; margin:4px 0 12px; }
    .workflow-note b { color:#105848; }
    div[role="tablist"] { display:flex; gap:10px; }
    div[data-testid="stTab"] {
        flex:1 1 0; height:60px; justify-content:center; white-space:normal;
        background:white; border:1px solid #d8d5ca; border-radius:12px;
        color:#105848; font-size:1rem; font-weight:700;
    }
    div[data-testid="stTab"] p { color:inherit; font-weight:700; }
    div[data-testid="stTab"][aria-selected="true"] {
        background:#105848; border-color:#105848; color:white;
    }
    div[data-testid="stTab"] .react-aria-SelectionIndicator { display:none; }
    [data-testid="stMetric"] { background:white; border:1px solid #dedcd2; padding:14px; border-radius:12px; }
    div[data-testid="stDataFrame"] { border:1px solid #dedcd2; border-radius:10px; overflow:hidden; }
    .boundary { background:#fff8e6; border-left:4px solid #e8ad3c; padding:10px 14px; border-radius:6px; }
    div[data-testid="stTextInput"] label,
    div[data-testid="stNumberInput"] label,
    div[data-testid="stSelectbox"] label { min-height:2.4rem; align-items:flex-end; }
    .template-copy { padding:.2rem 0; }
    .template-copy b { color:#105848; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>新能源配储寻优 · 云端测算平台</h1>
      <p>方案A｜全年15分钟数据｜全口径工程造价｜全网格容量比较｜选定方案报告</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def state_default(name, value):
    if name not in st.session_state:
        st.session_state[name] = value
    return st.session_state[name]


defaults = state_default("defaults", default_config())
state_default("saved_signature", None)
state_default("grid_result", None)
state_default("detail_result", None)
state_default("report_bytes", None)
state_default("schedule_bytes", None)


@st.cache_data(show_spinner=False)
def cached_read_upload(data: bytes, filename: str) -> pd.DataFrame:
    return read_uploaded_timeseries(data, filename)


def number(label, key, value, *, step=1.0, min_value=None, max_value=None, help=None,
           disabled=False):
    kwargs = dict(label=label, value=value, step=step, key=key, help=help, disabled=disabled)
    if min_value is not None:
        kwargs["min_value"] = min_value
    if max_value is not None:
        kwargs["max_value"] = max_value
    return st.number_input(**kwargs)


def render_parameters(base: dict) -> dict:
    config = dict(base)
    st.subheader("项目基本信息")
    row1 = st.columns(3, vertical_alignment="bottom")
    with row1[0]:
        config["project_name"] = st.text_input("项目名称", value=base["project_name"], key="p_project_name")
    with row1[1]:
        config["project_type"] = st.selectbox("新能源类型", ["风电", "光伏"],
                                               index=0 if base["project_type"] == "风电" else 1,
                                               key="p_project_type")
    with row1[2]:
        config["station_power_mw"] = number("新能源装机容量（MW）", "p_station", base["station_power_mw"],
                                             step=1.0, min_value=0.1)

    row2 = st.columns(3, vertical_alignment="bottom")
    with row2[0]:
        config["aux_power_mw"] = number("厂用电功率（MW）", "p_aux", base["aux_power_mw"],
                                         step=0.1, min_value=0.0)
    with row2[1]:
        config["mechanism_ratio_pct"] = number("机制电量/上网电量比（%）", "p_mech", base["mechanism_ratio_pct"],
                                                step=1.0, min_value=0.0, max_value=100.0)
    with row2[2]:
        config["capacity_compensation_yuan_per_mw_month"] = number(
            "容量补偿电价（元/MW·月）", "p_capacity_price",
            base["capacity_compensation_yuan_per_mw_month"], step=1000.0, min_value=0.0)

    with st.expander("调度与储能参数", expanded=True):
        row1 = st.columns(3, vertical_alignment="bottom")
        with row1[0]:
            config["soc_min_pct"] = number("SOC下限（%）", "p_soc_min", base["soc_min_pct"],
                                            step=1.0, min_value=0.0, max_value=99.0)
        with row1[1]:
            config["soc_max_pct"] = number("SOC上限（%）", "p_soc_max", base["soc_max_pct"],
                                            step=1.0, min_value=1.0, max_value=100.0)
        with row1[2]:
            config["efficiency_pct"] = number("储能综合效率（%）", "p_eta", base["efficiency_pct"],
                                               step=1.0, min_value=1.0, max_value=100.0)

        row2 = st.columns(3, vertical_alignment="bottom")
        with row2[0]:
            config["minimum_spread_yuan_per_mwh"] = number(
                "最低预测充放价差（元/MWh）", "p_spread", base["minimum_spread_yuan_per_mwh"],
                step=10.0, min_value=0.0)
        with row2[1]:
            config["maximum_charge_price_yuan_per_mwh"] = number(
                "最高预测充电价格（元/MWh）", "p_charge_price", base["maximum_charge_price_yuan_per_mwh"],
                step=10.0)
        with row2[2]:
            config["curtailment_price_yuan_per_mwh"] = number(
                "限电统计电价阈值（元/MWh）", "p_curt_price", base["curtailment_price_yuan_per_mwh"],
                step=10.0, help="只用于限电量统计，不是充电资格限制。")

    with st.expander("工程造价参数", expanded=True):
        st.markdown('<div class="boundary"><b>标准工程边界：</b>站内储能及接入工程；不含新建送出线路和主变扩容。</div>',
                    unsafe_allow_html=True)
        st.caption("固定计入：电池系统、PCS升压一体化系统、站内公用及二次系统、建筑安装工程、基本预备费。")

        st.markdown("**项目可选费用**")
        o1, o2, o3 = st.columns(3, vertical_alignment="top", border=True)
        with o1:
            config["include_land_cost"] = st.checkbox(
                "计入土地及场地投资", value=bool(base.get("include_land_cost", False)), key="p_include_land")
            config["land_cost_wan_per_mwh"] = number(
                "土地及场地（万元/MWh）", "p_land_cost", base["land_cost_wan_per_mwh"],
                step=0.5, min_value=0.0, disabled=not config["include_land_cost"])
        with o2:
            config["include_predevelopment_cost"] = st.checkbox(
                "计入前期研究及申报费", value=bool(base.get("include_predevelopment_cost", False)),
                key="p_include_predev")
            config["predevelopment_cost_wan"] = number(
                "前期研究申报费（万元）", "p_predev", base["predevelopment_cost_wan"],
                step=10.0, min_value=0.0, disabled=not config["include_predevelopment_cost"])
        with o3:
            config["include_engineering_cost"] = st.checkbox(
                "计入管理、设计及监理费", value=bool(base.get("include_engineering_cost", False)),
                key="p_include_engineering")
            config["engineering_rate_pct"] = number(
                "管理设计监理费率（%）", "p_engineering", base["engineering_rate_pct"],
                step=0.1, min_value=0.0, disabled=not config["include_engineering_cost"])

        st.markdown("**固定计入费用**")
        row1 = st.columns(3, vertical_alignment="bottom")
        with row1[0]:
            config["battery_cost_wan_per_mwh"] = number(
                "电池系统（万元/MWh）", "p_battery_cost", base["battery_cost_wan_per_mwh"],
                step=1.0, min_value=0.0,
                help="包含电芯/电池簇、BMS、热管理、电池舱及舱内消防等。")
        with row1[1]:
            config["pcs_cost_wan_per_mw"] = number(
                "PCS升压一体化系统（万元/MW）", "p_pcs_cost", base["pcs_cost_wan_per_mw"],
                step=1.0, min_value=0.0,
                help="包含PCS、升压变压器及配套开关设备。")
        with row1[2]:
            config["station_equipment_cost_wan_per_mw"] = number(
                "站内公用及二次系统（万元/MW）", "p_station_cost",
                base["station_equipment_cost_wan_per_mw"], step=1.0, min_value=0.0,
                help="包含EMS/AGC/AVC、通信、计量、保护、站级消防、交直流电源及线缆等；不重复计取PCS包内设备。")

        row2 = st.columns(4, vertical_alignment="bottom")
        with row2[0]:
            config["installation_rate_pct"] = number(
                "建筑安装费率（%）", "p_install", base["installation_rate_pct"], step=0.1, min_value=0.0)
        with row2[1]:
            config["contingency_rate_pct"] = number(
                "基本预备费率（%）", "p_contingency", base["contingency_rate_pct"], step=0.1, min_value=0.0)
        with row2[2]:
            config["battery_unit_mwh"] = number(
                "电池舱单体容量（MWh/舱）", "p_battery_unit", base["battery_unit_mwh"], step=0.5, min_value=0.1)
        with row2[3]:
            config["pcs_unit_mw"] = number(
                "PCS单套功率（MW/套）", "p_pcs_unit", base["pcs_unit_mw"], step=0.5, min_value=0.1)

        benchmark_header, benchmark_input = st.columns([3, 1], vertical_alignment="bottom")
        with benchmark_header:
            st.markdown("**2小时储能造价快速校核**")
            st.caption("按两小时配置即时检查当前造价参数是否处于合理水平。")
        with benchmark_input:
            check_power = number("校核功率（MW）", "p_cost_check_power", 10.0,
                                 step=1.0, min_value=0.1,
                                 help="按该功率和2小时容量连续估算；固定费用会随校核规模摊薄。")
        benchmark = cost_benchmark(config, check_power, 2.0)
        b1, b2, b3, b4 = st.columns(4, vertical_alignment="top")
        b1.metric("校核配置", f"{check_power:g}MW / {benchmark['energy_mwh']:g}MWh")
        b2.metric("估算总投资", f"{benchmark['total_wan']:,.1f}万元")
        b3.metric("功率单位造价", f"{benchmark['yuan_per_w']:.3f}元/W")
        b4.metric("容量单位造价", f"{benchmark['yuan_per_wh']:.3f}元/Wh")
        st.caption("校核采用连续造价，不执行电池舱和PCS套数向上取整；用于快速判断当前输入造价水平。")

    with st.expander("财务参数", expanded=False):
        row1 = st.columns(4, vertical_alignment="bottom")
        with row1[0]:
            config["project_years"] = int(number("项目评价期（年）", "p_years", int(base["project_years"]),
                                                  step=1, min_value=1, max_value=40))
        with row1[1]:
            config["loan_term_years"] = int(number("首次贷款期限（年）", "p_loan_term", int(base["loan_term_years"]),
                                                    step=1, min_value=1, max_value=40))
        with row1[2]:
            config["replacement_loan_term_years"] = int(number(
                "更换电池贷款期限（年）", "p_rep_loan", int(base["replacement_loan_term_years"]),
                step=1, min_value=1, max_value=20))
        with row1[3]:
            config["equity_ratio_pct"] = number(
                "资本金比例（%）", "p_equity", base["equity_ratio_pct"], step=1.0, min_value=0.0, max_value=100.0)

        row2 = st.columns(4, vertical_alignment="bottom")
        with row2[0]:
            config["loan_rate_pct"] = number(
                "贷款利率（%）", "p_loan_rate", base["loan_rate_pct"], step=0.1, min_value=0.0)
        with row2[1]:
            config["equity_discount_rate_pct"] = number(
                "资本金折现率（%）", "p_equity_rate", base["equity_discount_rate_pct"], step=0.5, min_value=0.0)
        with row2[2]:
            config["battery_degradation_pct"] = number(
                "电池年衰减率（%）", "p_degr", base["battery_degradation_pct"], step=0.1, min_value=0.0)
        with row2[3]:
            config["om_ratio_pct"] = number(
                "年生产费用/初始投资（%）", "p_om", base["om_ratio_pct"], step=0.1, min_value=0.0)

        row3 = st.columns(4, vertical_alignment="bottom")
        with row3[0]:
            config["income_tax_pct"] = number(
                "所得税率（%）", "p_tax", base["income_tax_pct"], step=1.0, min_value=0.0, max_value=100.0)

    with st.expander("工商业分时电价", expanded=False):
        st.caption(
            "默认采用山东省自2026年10月1日起执行的工商业分时电价时段；"
            "高峰、低谷、尖峰和深谷浮动比例分别为+70%、-70%、+100%和-90%。"
        )
        flat_default = pd.DataFrame({"月份": list(range(1, 13)),
                                     "平段电价（元/MWh）": [base["flat_prices"][m] for m in range(1, 13)]})
        flat_edited = st.data_editor(
            flat_default, hide_index=True, width="stretch", key="p_flat_editor",
            disabled=["月份"], column_config={
                "月份": st.column_config.NumberColumn(format="%d月"),
                "平段电价（元/MWh）": st.column_config.NumberColumn(min_value=0.0, format="%.2f"),
            })
        config["flat_prices"] = {
            int(row["月份"]): float(row["平段电价（元/MWh）"])
            for _, row in flat_edited.iterrows()
        }
        coeff_default = pd.DataFrame({"时段": list(DEFAULT_COEF),
                                      "系数": [base["tou_coefficients"][key] for key in DEFAULT_COEF]})
        coeff_edited = st.data_editor(
            coeff_default, hide_index=True, width="stretch", key="p_coeff_editor",
            disabled=["时段"], column_config={"系数": st.column_config.NumberColumn(min_value=0.0, format="%.3f")})
        config["tou_coefficients"] = {str(row["时段"]): float(row["系数"])
                                       for _, row in coeff_edited.iterrows()}
        zone_default = pd.DataFrame([
            {"月份": group, **categories} for group, categories in base["zone_text"].items()
        ])
        zone_edited = st.data_editor(zone_default, hide_index=True, width="stretch",
                                     key="p_zone_editor", disabled=["月份"])
        config["zone_text"] = {
            str(row["月份"]): {
                key: (str(row[key]).strip() if pd.notna(row[key]) and str(row[key]).strip() else "—")
                for key in ["尖峰", "高峰", "低谷", "深谷"]
            }
            for _, row in zone_edited.iterrows()
        }
    return config


with st.sidebar:
    st.markdown("### 测算口径")
    st.caption("当前前端仅运行方案A。上传数据和计算结果仅保存在当前会话，云端不长期留存。")
    st.markdown("**固定规则**")
    st.write("- 15分钟、96点/日")
    st.write("- 同一时段禁止同时充放电")
    st.write("- 放电功率＋新能源功率≤场站装机")
    st.write("- 全网格采用连续造价估算")


with st.expander("储能调度逻辑（方案A）", expanded=False):
    dispatch_config = st.session_state.get("config", defaults)
    dispatch_spread = float(st.session_state.get(
        "p_spread", dispatch_config["minimum_spread_yuan_per_mwh"]))
    dispatch_charge_price = float(st.session_state.get(
        "p_charge_price", dispatch_config["maximum_charge_price_yuan_per_mwh"]))
    dispatch_soc_min = float(st.session_state.get(
        "p_soc_min", dispatch_config["soc_min_pct"]))
    dispatch_soc_max = float(st.session_state.get(
        "p_soc_max", dispatch_config["soc_max_pct"]))
    dispatch_efficiency = float(st.session_state.get(
        "p_eta", dispatch_config["efficiency_pct"]))
    dispatch_curtailment_price = float(st.session_state.get(
        "p_curt_price", dispatch_config["curtailment_price_yuan_per_mwh"]))
    st.markdown("**日前排计划 → 运行日按实际修正 → 按实际执行量结算收益**")
    st.markdown(f"""
**1. 日前：比较全天96个点，形成充放电计划**

使用预测电价和预测发电功率，在功率、容量及电量平衡约束下，统一比较全天可实现的充放电收益，安排较低价格充电、较高价值放电。充电同时考虑发电电量和免费限电电量，放电同时考虑市场售电和抵厂用电。

- **发电充电**：预测充电价格不高于 **{dispatch_charge_price:g}元/MWh**；每笔充电必须与当天后续放电配对，对应预测放电价值减去预测充电价格不低于 **{dispatch_spread:g}元/MWh**。价差门槛用于判断充放电资格，充电成本按预测电价计算。抵厂用电按对应分时电价衡量价值。
- **免费电量**：按0元/MWh计成本，预测放电价值至少达到 **{dispatch_spread:g}元/MWh** 才能放电。跨日剩余电量继续保留原来的最低预测放电价格。

**2. 运行日：优先执行日前计划，按实际能力修正**

- 实际发电不足时，计划充电只能按实际可实现功率执行。少充的发电充电电量形成当天缺口；后续利用“可用功率－实际发电功率”的正偏差免费弥补，只补已有缺口，不挤占本时点计划，计划放电时不补充电，未补缺口当天清零。
- 补充电不受限电统计电价阈值约束。计划限电充电的可用量，当前按实时电价不高于 **{dispatch_curtailment_price:g}元/MWh** 时的“可用功率－实际发电功率”正偏差计算。
- 按计划放电时，同时核对实际库存、电量来源对应的价格门槛及并网余量。电量不足就相应少放，市场上网放电还受“新能源实际功率＋储能上网放电功率不超过场站装机”限制。

**3. 全过程约束与收益结算**

同一15分钟点禁止同时充放电，充、放电功率均不超过储能额定功率；荷电状态保持在 **{dispatch_soc_min:g}%～{dispatch_soc_max:g}%**，综合效率按 **{dispatch_efficiency:g}%** 计。缺口当天清零，实际剩余库存可跨日，电量来源及价格门槛继续保留。

市场售电收入和发电充电成本按实时市场电价及实际执行量结算，抵厂用电按分时电价计收益，并另行计算容量补偿。以上数值随第一步填写的参数变化，测算采用保存后的参数。
""")
    st.caption("当前测算使用上传数据计算的限电量作为日前免费充电可用量；预测电价用于排计划，实时电价用于结算。")


st.markdown(
    '<div class="workflow-note"><b>操作顺序：</b>先在①上传数据并保存参数；'
    '再点击②运行全网格测算；测算完成后点击③选择配置并下载结果。</div>',
    unsafe_allow_html=True,
)
tab_input, tab_grid, tab_report = st.tabs([
    "① 上传数据并保存参数",
    "② 运行全网格测算",
    "③ 选定方案并下载报告",
])

with tab_input:
    st.subheader("上传全年生产与市场时序")
    with st.container(border=True):
        template_copy, template_action = st.columns([3, 1], vertical_alignment="center")
        with template_copy:
            st.markdown(
                '<div class="template-copy"><b>第一步：下载标准模板</b><br>'
                '仅在模板中填写全年生产和市场时序数据，其他项目参数在本页录入。</div>',
                unsafe_allow_html=True,
            )
        with template_action:
            st.download_button("下载全年时序CSV模板", upload_template_csv(),
                               file_name="全年生产与市场时序模板.csv", mime="text/csv",
                               width="stretch")
    upload = st.file_uploader(
        "第二步：上传已填写的CSV或XLSX文件（XLSX优先读取“时序数据”工作表）",
        type=["csv", "xlsx", "xlsm"], key="timeseries_upload")
    timeseries = None
    upload_bytes = b""
    if upload is not None:
        upload_bytes = upload.getvalue()
        try:
            timeseries = cached_read_upload(upload_bytes, upload.name)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("时序点数", f"{len(timeseries):,}")
            c2.metric("完整天数", f"{timeseries['时间戳'].dt.date.nunique():,}")
            c3.metric("起始时间", timeseries["时间戳"].min().strftime("%Y-%m-%d"))
            c4.metric("结束时间", timeseries["时间戳"].max().strftime("%Y-%m-%d"))
            fill_stats = timeseries.attrs.get("power_fill_stats", {})
            filled = {column: count for column, count in fill_stats.items() if count}
            if filled:
                detail = "、".join(f"{column}补全{count}点" for column, count in filled.items())
                st.warning("数据校验通过；功率字段已按主模型交叉补全规则处理：" + detail + "。")
            else:
                st.success("数据校验通过：时间连续、每天96点、必填字段完整。")
            with st.expander("查看上传数据前96点"):
                st.dataframe(timeseries.head(96), width="stretch", height=360)
        except Exception as exc:
            st.error(str(exc))
    config_now = render_parameters(defaults)
    if st.button("保存数据与参数", type="primary", width="stretch", disabled=timeseries is None):
        if config_now["soc_min_pct"] >= config_now["soc_max_pct"]:
            st.error("SOC下限必须低于SOC上限。")
        elif config_now["loan_term_years"] > config_now["project_years"]:
            st.error("首次贷款期限不得超过项目评价期。")
        else:
            try:
                frame = build_model_frame(timeseries, config_now)
                signature = config_signature(config_now, upload_bytes)
                changed = signature != st.session_state.saved_signature
                st.session_state.config = config_now
                st.session_state.timeseries = timeseries
                st.session_state.model_frame = frame
                st.session_state.upload_bytes = upload_bytes
                st.session_state.saved_signature = signature
                if changed:
                    st.session_state.grid_result = None
                    st.session_state.detail_result = None
                    st.session_state.report_bytes = None
                    st.session_state.schedule_bytes = None
                st.success("第一步已完成。请点击页面上方“② 运行全网格测算”进入下一步。")
            except Exception as exc:
                st.error(f"参数校验失败：{exc}")

with tab_grid:
    if st.session_state.saved_signature is None:
        st.info("第二步尚未解锁：请先点击上方“① 上传数据并保存参数”，完成上传和保存。")
    else:
        config = st.session_state.config
        station_power = float(config["station_power_mw"])
        default_ratio = 50.0 if config["project_type"] == "光伏" else 30.0
        st.subheader("设置全网格范围")
        c1, c2 = st.columns([1, 2], vertical_alignment="bottom")
        with c1:
            maximum_ratio = st.number_input("储能功率上限/新能源装机（%）", min_value=1.0,
                                            max_value=100.0, value=default_ratio, step=1.0,
                                            key="grid_ratio")
        with c2:
            durations = st.pills(
                "持续时长H（小时）", [2, 4, 6, 8, 10], selection_mode="multi",
                default=[2, 4, 6, 8, 10], key="grid_hours",
                format_func=lambda value: f"{value}小时", width="stretch", wrap=False,
            )
        powers = make_power_grid(station_power, config["project_type"], maximum_ratio)
        st.caption("功率网格：1—5MW按1MW递增，10MW以上按5MW递增。"
                   f"当前共{len(powers)}个功率点、{len(durations)}个时长点、"
                   f"{len(powers) * len(durations)}个组合。")
        st.info("全年方案A需要逐日求解365个日前计划，云端算力波动较大。"
                "开始后页面会持续显示已完成天数和预计剩余时间，每完成一个组合立即输出一行结果。"
                "计算期间请勿重复提交或关闭页面。")
        if st.button("运行全网格测算", type="primary", width="stretch",
                     disabled=not durations):
            progress_bar = st.progress(0.0)
            status = st.empty()
            partial_table = st.empty()
            started_at = time.monotonic()
            def update_progress(done, total, label):
                fraction = min(1.0, max(0.0, done / total))
                progress_bar.progress(fraction)
                elapsed = time.monotonic() - started_at
                eta = elapsed * (1 - fraction) / fraction if fraction > 0 else None
                eta_text = f"｜预计剩余约{eta / 60:.1f}分钟" if eta is not None else ""
                status.caption(
                    f"正在计算：{label}｜总进度{fraction:.1%}｜"
                    f"已用{elapsed / 60:.1f}分钟{eta_text}")
            def show_partial(partial):
                columns = ["P", "H", "E", "annual_cycles", "total_revenue",
                           "fi_npv", "eq_irr", "error"]
                available = [column for column in columns if column in partial.columns]
                partial_table.dataframe(partial[available], hide_index=True, width="stretch")
            try:
                result = run_grid(st.session_state.model_frame, config, powers, durations,
                                  update_progress, show_partial)
            except Exception as exc:
                st.error(f"全网格测算中断：{exc}")
                st.stop()
            st.session_state.grid_result = result
            st.session_state.detail_result = None
            st.session_state.report_bytes = None
            st.session_state.schedule_bytes = None
            status.empty()
            progress_bar.empty()
            partial_table.empty()
            st.success("第二步已完成。请点击页面上方“③ 选定方案并下载报告”进入下一步。")

        grid = st.session_state.grid_result
        if grid is not None:
            failed = grid[grid["error"].fillna("").ne("")] if "error" in grid else pd.DataFrame()
            success = grid[grid["error"].fillna("").eq("")].copy() if "error" in grid else grid.copy()
            if success.empty:
                st.error("全网格没有成功结果，请检查错误信息。")
                st.dataframe(grid[["P", "H", "error"]], width="stretch")
            else:
                best_eq = best_valid_row(success, "eq_irr")
                best_npv = best_valid_row(success, "fi_npv")
                c1, c2, c3, c4 = st.columns(4)
                if best_eq is None:
                    c1.metric("资本金IRR最高配置", "无有效值")
                    c2.metric("最高资本金IRR", "不可计算")
                    st.warning("本次成功组合的资本金现金流均未形成可计算的有效IRR根，请结合资本金NPV和全投资NPV判断。")
                else:
                    c1.metric("资本金IRR最高配置", f"{best_eq.P:g}MW / {best_eq.H:g}h")
                    c2.metric("最高资本金IRR", f"{best_eq.eq_irr:.2f}%")
                if best_npv is None:
                    c3.metric("全投资NPV最高配置", "无有效值")
                    c4.metric("最高全投资NPV", "不可计算")
                else:
                    c3.metric("全投资NPV最高配置", f"{best_npv.P:g}MW / {best_npv.H:g}h")
                    c4.metric("最高全投资NPV", f"{best_npv.fi_npv:,.1f}万元")
                if not failed.empty:
                    st.warning(f"另有{len(failed)}个组合未完成，可在结果表的错误列查看原因。")
                display = success.sort_values("eq_irr", ascending=False, na_position="last").rename(columns={
                    "P": "功率(MW)", "H": "时长(h)", "E": "容量(MWh)",
                    "annual_cycles": "年充电次数", "charge_mwh": "年充电量(MWh)",
                    "discharge_mwh": "年放电量(MWh)", "R_year": "经营净收益(万元)",
                    "R_cap": "容量补偿(万元)", "total_revenue": "首年总收入(万元)",
                    "capex": "初始投资(万元)", "fi_npv": "全投资NPV(万元)",
                    "fi_irr": "全投资IRR(%)", "eq_npv": "资本金NPV(万元)",
                    "eq_irr": "资本金IRR(%)",
                })
                st.dataframe(display, hide_index=True, width="stretch", height=520)
                heat = success.copy()
                heat["配置"] = heat.apply(lambda row: f"{row.P:g}MW/{row.E:g}MWh", axis=1)
                heat_column = "eq_irr" if best_eq is not None else "fi_npv"
                heat_title = "资本金IRR（%）" if best_eq is not None else "全投资NPV（万元）"
                heat_tooltip = "资本金IRR" if best_eq is not None else "全投资NPV"
                tooltips = ["配置:N", alt.Tooltip(f"{heat_column}:Q", title=heat_tooltip, format=".2f")]
                if best_eq is not None:
                    tooltips.append(alt.Tooltip("fi_npv:Q", title="全投资NPV", format=",.1f"))
                tooltips.append(alt.Tooltip("annual_cycles:Q", title="年充电次数", format=".1f"))
                heatmap = alt.Chart(heat).mark_rect(cornerRadius=3).encode(
                    x=alt.X("P:O", title="储能功率（MW）"),
                    y=alt.Y("H:O", title="持续时长（h）"),
                    color=alt.Color(f"{heat_column}:Q", title=heat_title, scale=alt.Scale(scheme="yellowgreenblue")),
                    tooltip=tooltips,
                ).properties(height=300)
                st.altair_chart(heatmap, width="stretch")
                st.download_button("下载全网格CSV", grid_csv_bytes(grid),
                                   file_name=f"{config['project_name']}_方案A_全网格.csv",
                                   mime="text/csv", width="stretch")

with tab_report:
    grid = st.session_state.grid_result
    if grid is None:
        st.info("第三步尚未解锁：请先点击上方“② 运行全网格测算”，完成容量组合计算。")
    else:
        success = grid[grid["error"].fillna("").eq("")].copy() if "error" in grid else grid.copy()
        if success.empty:
            st.error("没有可供选择的成功配置。")
        else:
            success = success.sort_values(["P", "H"])
            options = [(float(row.P), int(row.H)) for row in success.itertuples()]
            best = best_valid_row(success, "eq_irr")
            if best is None:
                best = best_valid_row(success, "fi_npv")
            if best is None:
                best = success.iloc[0]
            default_option = (float(best.P), int(best.H))
            selected = st.selectbox(
                "选择储能功率和时长",
                options,
                index=options.index(default_option),
                format_func=lambda item: f"{item[0]:g}MW / {item[0] * item[1]:g}MWh（{item[1]}h）",
                key="selected_config")
            power, duration = selected
            selected_row = success[(success["P"] == power) & (success["H"] == duration)].iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("初始投资", f"{selected_row.capex:,.1f}万元")
            c2.metric("首年总收入", f"{selected_row.total_revenue:,.1f}万元")
            c3.metric("全投资NPV", f"{selected_row.fi_npv:,.1f}万元")
            c4.metric("资本金IRR", "—" if not np.isfinite(selected_row.eq_irr) else f"{selected_row.eq_irr:.2f}%")
            if st.button("计算选定配置并生成报告", type="primary", width="stretch"):
                with st.spinner("正在重新计算全年调度并生成报告……"):
                    detail = run_configuration(st.session_state.model_frame, st.session_state.config,
                                               power, duration)
                    st.session_state.detail_result = detail
                    st.session_state.schedule_bytes = schedule_excel_bytes(detail)
                    st.session_state.report_bytes = build_report_bytes(st.session_state.config, grid, detail)
                st.success("选定配置计算完成。")

            detail = st.session_state.detail_result
            if detail is not None:
                metrics = detail["metrics"]
                if not math.isclose(metrics["P"], power) or metrics["H"] != duration:
                    st.warning("当前展示的是上一次生成的配置，请点击按钮重新生成所选配置。")
                else:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("全年充电量", f"{metrics['charge_mwh']:,.1f}MWh")
                    c2.metric("全年放电量", f"{metrics['discharge_mwh']:,.1f}MWh")
                    c3.metric("等效满循环", f"{metrics['annual_cycles']:,.1f}次")
                    c4.metric("限电电量利用率", f"{metrics['curtailment_capture_pct']:,.2f}%")
                    profile = detail["average_profile"]
                    power_chart = alt.Chart(profile).mark_bar().encode(
                        x=alt.X("点:O", title="一天96个15分钟点", axis=alt.Axis(labelAngle=0, values=list(range(0, 96, 8)))),
                        y=alt.Y("储能功率(MW):Q", title="储能功率（MW）"),
                        color=alt.condition(alt.datum["储能功率(MW)"] >= 0,
                                            alt.value("#e8ad3c"), alt.value("#105848")),
                        tooltip=["时刻:N", alt.Tooltip("储能功率(MW):Q", format=".2f")],
                    )
                    price_chart = alt.Chart(profile).mark_line(color="#a04028", strokeWidth=2.2).encode(
                        x=alt.X("点:O"),
                        y=alt.Y("实时电价(元/MWh):Q", title="实时电价（元/MWh）"),
                        tooltip=["时刻:N", alt.Tooltip("实时电价(元/MWh):Q", format=".2f")],
                    )
                    chart = alt.layer(power_chart, price_chart).resolve_scale(y="independent").properties(
                        height=380, title="全年96点平均储能功率与实时市场电价")
                    st.altair_chart(chart, width="stretch")
                    col1, col2 = st.columns(2)
                    safe_name = "".join(character if character.isalnum() or character in "-_" else "_"
                                        for character in st.session_state.config["project_name"])
                    with col1:
                        st.download_button(
                            "下载全年充放电明细", st.session_state.schedule_bytes,
                            file_name=f"{safe_name}_{power:g}MW_{power*duration:g}MWh_充放电情况.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            width="stretch")
                    with col2:
                        st.download_button(
                            "下载投资分析报告", st.session_state.report_bytes,
                            file_name=f"{safe_name}_{power:g}MW_{power*duration:g}MWh_投资分析报告.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            width="stretch")

st.caption("测算结果为输入数据、方案A调度规则和当前参数下的历史回测结果。")
