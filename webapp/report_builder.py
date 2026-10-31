from __future__ import annotations

import io
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from webapp.model_service import DEFAULT_BATTERY_REPLACEMENT_YEAR, report_sensitivity
from webapp.report_charts import build_charts


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "报告模板/投资分析报告_标准格式.docx"
FONT_BODY = "仿宋_GB2312"
FONT_HEAD = "黑体"
FONT_SECOND = "楷体_GB2312"
FONT_TITLE = "STSong"


def _number(value: float) -> str:
    value = float(value)
    return str(int(value)) if value.is_integer() else f"{value:g}"


def _fmt(value, decimals=2, suffix="") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    return f"{number:,.{decimals}f}{suffix}"


def _clear_body(document: Document) -> None:
    body = document._element.body
    for child in list(body):
        if child.tag != qn("w:sectPr"):
            body.remove(child)


def _set_font(run, name=FONT_BODY, size=16, bold=False) -> None:
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), name)
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor(0, 0, 0)


def _add_text(document, text="", *, font=FONT_BODY, size=16, bold=False,
              align=WD_ALIGN_PARAGRAPH.JUSTIFY, indent=True, keep=False):
    paragraph = document.add_paragraph()
    paragraph.alignment = align
    paragraph.paragraph_format.first_line_indent = Pt(32 if indent else 0)
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
    paragraph.paragraph_format.line_spacing = Pt(28.95)
    paragraph.paragraph_format.keep_with_next = keep
    _set_font(paragraph.add_run(text), font, size, bold)
    return paragraph


def _add_heading(document, text: str, level=1):
    paragraph = document.add_paragraph(style=f"Heading {level}")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    paragraph.paragraph_format.first_line_indent = Pt(32)
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
    paragraph.paragraph_format.line_spacing = Pt(28.95)
    paragraph.paragraph_format.keep_with_next = True
    _set_font(paragraph.add_run(text), FONT_HEAD if level == 1 else FONT_SECOND, 16, level == 2)
    return paragraph


def _set_cell(cell, text, bold=False, align=WD_ALIGN_PARAGRAPH.LEFT):
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.alignment = align
    paragraph.paragraph_format.first_line_indent = Pt(0)
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
    paragraph.paragraph_format.line_spacing = Pt(18)
    _set_font(paragraph.add_run(str(text)), FONT_HEAD if bold else FONT_BODY, 10.5, bold)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    properties = cell._tc.get_or_add_tcPr()
    margins = OxmlElement("w:tcMar")
    for name, value in (("top", 60), ("start", 70), ("bottom", 60), ("end", 70)):
        node = OxmlElement(f"w:{name}")
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")
        margins.append(node)
    properties.append(margins)


def _add_table(document, headers, rows, widths=None, aligns=None):
    body = document._element.body
    if len(body) > 1 and body[-2].tag == qn("w:tbl"):
        # Word会合并紧邻的同样式表格；用无可见内容的小段落隔开，保留各自重复表头。
        spacer = document.add_paragraph()
        spacer.paragraph_format.space_before = Pt(0)
        spacer.paragraph_format.space_after = Pt(0)
        spacer.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
        spacer.paragraph_format.line_spacing = Pt(1)
        _set_font(spacer.add_run(), size=1)
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    table.rows[0]._tr.get_or_add_trPr().append(repeat)
    for index, header in enumerate(headers):
        _set_cell(table.rows[0].cells[index], header, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    for row in rows:
        cells = table.add_row().cells
        cant_split = OxmlElement("w:cantSplit")
        table.rows[-1]._tr.get_or_add_trPr().append(cant_split)
        for index, value in enumerate(row):
            _set_cell(cells[index], value,
                      align=aligns[index] if aligns else (
                          WD_ALIGN_PARAGRAPH.LEFT if index == 0 else WD_ALIGN_PARAGRAPH.RIGHT))
    if widths:
        section = document.sections[-1]
        available_width = (section.page_width - section.left_margin - section.right_margin) / Cm(1)
        widths = [width * available_width / sum(widths) for width in widths]
        for column, width in zip(table.columns, widths):
            column.width = Cm(width)
        for row in table.rows:
            for index, width in enumerate(widths):
                row.cells[index].width = Cm(width)
    return table


def _add_picture(document, image_bytes, caption):
    section = document.sections[-1]
    width = section.page_width - section.left_margin - section.right_margin
    paragraph = _add_text(document, indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, keep=True)
    paragraph.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    paragraph.add_run().add_picture(io.BytesIO(image_bytes), width=width)
    _add_text(document, caption, font=FONT_SECOND, size=12,
              indent=False, align=WD_ALIGN_PARAGRAPH.CENTER)


def _financing_rows(finance):
    cumulative = -finance["capex"] * finance["equity_ratio"]
    rows = []
    for item in finance["yearly"]:
        debt = item["principal"] + item["interest"]
        cumulative += item["eq_ncf"]
        rows.append({"year": item["y"], "revenue_wan": item["revenue"] / 1e4,
                     "debt_service_wan": debt / 1e4, "equity_injection_wan": item["equity_inj"] / 1e4,
                     "equity_ncf_wan": item["eq_ncf"] / 1e4, "cumulative_wan": cumulative / 1e4,
                     "dscr": (item["revenue"] - item["om"] - item["eq_tax"]) / debt if debt > 1e-9 else np.nan})
    return pd.DataFrame(rows)


def _new_document() -> Document:
    document = Document(TEMPLATE) if TEMPLATE.exists() else Document()
    _clear_body(document)
    normal = document.styles["Normal"]
    normal.font.name = FONT_BODY
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), FONT_BODY)
    normal.font.size = Pt(16)
    normal.paragraph_format.first_line_indent = Pt(32)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(0)
    normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
    normal.paragraph_format.line_spacing = Pt(28.95)
    return document


def _add_cover(document, config, metrics):
    _add_text(document, "", align=WD_ALIGN_PARAGRAPH.CENTER, indent=False)
    _add_text(document, config["project_name"], font=FONT_TITLE, size=22,
              align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, keep=True)
    _add_text(document, f"{_number(metrics['P'])}MW/{_number(metrics['E'])}MWh配建储能项目",
              font=FONT_TITLE, size=22, align=WD_ALIGN_PARAGRAPH.CENTER,
              indent=False, keep=True)
    _add_text(document, "投资分析报告", font=FONT_TITLE, size=22,
              align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, keep=True)
    _add_text(document, "（方案A测算）", font=FONT_SECOND, size=16,
              align=WD_ALIGN_PARAGRAPH.CENTER, indent=False)
    today = date.today()
    _add_text(document, f"{today.year}年{today.month}月{today.day}日", size=16,
              align=WD_ALIGN_PARAGRAPH.CENTER, indent=False)
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.first_line_indent = Pt(0)
    paragraph.add_run().add_break(WD_BREAK.PAGE)


def build_report_bytes(config: dict, grid: pd.DataFrame, detail: dict) -> bytes:
    metrics = detail["metrics"]
    finance = detail["finance"]
    investment = finance["capex_breakdown"]
    replacement_year = int(config.get("battery_replacement_year", DEFAULT_BATTERY_REPLACEMENT_YEAR))
    schedule = detail["schedule"]
    financing = _financing_rows(finance)
    stresses = report_sensitivity(config, metrics)
    successful = grid[grid["error"].fillna("").eq("")].copy() if "error" in grid else grid.copy()
    selected = successful[np.isclose(successful["P"], metrics["P"]) &
                          np.isclose(successful["H"], metrics["H"])]
    grid_consistent = not selected.empty and np.isclose(selected.iloc[0]["capex"], metrics["capex"], rtol=1e-6)
    if not grid_consistent:
        successful = pd.DataFrame([metrics])
    comparison = successful[successful["H"].eq(metrics["H"])].sort_values("P")
    charts = build_charts(detail, financing, stresses, comparison)
    document = _new_document()
    _add_cover(document, config, metrics)
    station = float(config["station_power_mw"])
    before = metrics["actual_generation_mwh"]
    after = metrics["post_export_mwh"]
    export_after = metrics["post_market_export_mwh"]
    curtailed = metrics["curtailed_energy_mwh"]
    captured = metrics["curtailed_charge_mwh"]
    curtailed_after = max(0, curtailed - captured)
    available_energy = before + curtailed
    before_rate = curtailed / available_energy * 100 if available_energy else 0
    after_rate = curtailed_after / available_energy * 100 if available_energy else 0
    revenue_before = metrics["base_market_revenue_wan"]
    comprehensive_after = (revenue_before + metrics["total_revenue"]) * 1e4 / after if after else np.nan
    uplift = metrics["total_revenue"] * 1e4 / after if after else np.nan
    income_increase = metrics["total_revenue"] / revenue_before * 100 if revenue_before else np.nan
    utility = metrics["utility_discharge_mwh"]
    has_utility = utility > 1e-8
    supply_label = "整体供电量（含抵厂用电）" if has_utility else "整体上网电量"
    eta = float(config["efficiency_pct"]) / 100
    energy_delta = (float(schedule["SOC(%)"].iloc[-1]) - float(config["soc_min_pct"])) / 100 * metrics["E"]
    conversion_loss = (metrics["charge_mwh"] - energy_delta) * (1 - eta)
    paid = schedule["发电充电功率(MW)"].to_numpy()
    actual = schedule["实际发电功率(MW)"].to_numpy()
    price = schedule["实时电价(元/MWh)"].to_numpy()
    negative = price < 0
    low_positive = (price >= 0) & (price <= float(config["maximum_charge_price_yuan_per_mwh"]))
    negative_available = float(actual[negative].sum() * 0.25)
    negative_charged = float(paid[negative].sum() * 0.25)
    positive_charged = float(paid[low_positive].sum() * 0.25)
    title_config = f"{_number(metrics['P'])}MW/{_number(metrics['E'])}MWh"

    _add_heading(document, "一、报告摘要")
    _add_table(document, ["核心指标", "测算结果", "单位"], [
        ["初始总投资", _fmt(metrics["capex"], 1), "万元"],
        ["首年经营净收益", _fmt(metrics["R_year"], 1), "万元"],
        ["首年容量补偿", _fmt(metrics["R_cap"], 1), "万元"],
        ["首年总收入", _fmt(metrics["total_revenue"], 1), "万元"],
        ["全投资净现值", _fmt(metrics["fi_npv"], 1), "万元"],
        ["全投资内部收益率", _fmt(metrics["fi_irr"]), "%"],
        ["资本金净现值", _fmt(metrics["eq_npv"], 1), "万元"],
        ["资本金内部收益率", _fmt(metrics["eq_irr"]), "%"],
    ], widths=[6.2, 5, 3.2])
    _add_text(document,
        f"本报告依据{metrics['start']:%Y年%m月%d日}至{metrics['end']:%Y年%m月%d日}的"
        f"{metrics['points']:,}个15分钟生产与市场数据点，对{title_config}配置进行方案A历史回测。"
        f"初始投资为{metrics['capex']:,.1f}万元，首年经营净收益{metrics['R_year']:,.1f}万元，"
        f"加上容量补偿后首年总收入{metrics['total_revenue']:,.1f}万元。")
    _add_text(document,
        f"全年充电{metrics['charge_mwh']:,.1f}MWh、放电{metrics['discharge_mwh']:,.1f}MWh，"
        f"等效满循环{metrics['annual_cycles']:,.1f}次。储能利用限电电量{captured:,.1f}MWh，"
        f"限电比例由{before_rate:.2f}%变为{after_rate:.2f}%；项目{ supply_label }变化"
        f"{after-before:+,.1f}MWh，等效收益单价提升{_fmt(uplift)}元/MWh。"
        "以上为当前数据和输入条件下的测算事实，不作投资方向判断。")

    _add_heading(document, "二、项目概况与测算范围")
    _add_text(document,
        f"{config['project_name']}为{config['project_type']}项目，装机{station:g}MW，"
        f"厂用电功率按{config['aux_power_mw']:g}MW输入。本次储能功率{metrics['P']:g}MW、"
        f"容量{metrics['E']:g}MWh，额定持续时长{metrics['H']:g}小时，占新能源装机"
        f"{metrics['P']/station*100:.2f}%。评价期为{config['project_years']}年。")
    _add_table(document, ["项目", "输入条件"], [
        ["样本期间", f"{metrics['start']:%Y-%m-%d}至{metrics['end']:%Y-%m-%d}"],
        ["可用功率来源", metrics["available_power_source"]],
        ["新能源装机／厂用电", f"{station:g}MW／{config['aux_power_mw']:g}MW"],
        ["储能配置", title_config],
        ["综合效率／SOC范围", f"{config['efficiency_pct']:g}%／{config['soc_min_pct']:g}%—{config['soc_max_pct']:g}%"],
        ["预测价差／充电价格上限", f"不低于{config['minimum_spread_yuan_per_mwh']:g}／不高于{config['maximum_charge_price_yuan_per_mwh']:g}元/MWh"],
        ["项目评价期／首次贷款期", f"{config['project_years']}年／{min(config['loan_term_years'],config['project_years'])}年"],
        ["电池更换安排", f"第{replacement_year}年末更换一次" if replacement_year < config['project_years'] else "评价期内不更换"],
    ], widths=[6, 8.4])
    _add_text(document,
        "工程范围为站内储能及接入工程，不含新建送出线路和主变扩容。报告分别列示调度、运行影响、"
        "工程投资和财务现金流；可选费用以网页勾选结果为准。一个样本年度用于首年运行回测，"
        "后续年度收入按电池容量衰减比例折算，未逐年重新预测市场价格和求解调度。")
    _add_text(document,
        f"可用功率来源为“{metrics['available_power_source']}”。可用功率应考虑检修、故障等影响，"
        "理论功率不能替代可用功率。若来源标注为超短期预测估算，限电及偏差弥补指标也属于该估算口径，"
        "不能视为独立实测可用功率的统计结果。")

    _add_heading(document, "三、技术方案与调度原则")
    principles = [
        "日前使用预测电价和预测发电功率形成每天96点充放电计划，运行日执行既定计划并按实际发电能力修正，结算使用实时市场电价和实际执行量。",
        f"付费充电的预测价格不高于{config['maximum_charge_price_yuan_per_mwh']:g}元/MWh，逐笔对应当日后续预测放电价格高出至少{config['minimum_spread_yuan_per_mwh']:g}元/MWh的机会；免费库存按零成本执行相应预测放电门槛。价差门槛作为资格约束，不加在充电成本上。",
        "发电能力不足导致少充时，记录当日充电缺口；后续可用功率高于实际发电功率时，仅用该偏差免费弥补缺口，缺口当天清零。偏差弥补不受限电统计电价阈值限制。",
        f"同一15分钟时段禁止同时充放电。放电受实际库存、{metrics['P']:g}MW额定功率、逐笔库存价差以及新能源功率与储能放电功率合计不超过{station:g}MW等条件约束。",
        f"荷电状态运行范围为{config['soc_min_pct']:g}%—{config['soc_max_pct']:g}%，综合效率按{config['efficiency_pct']:g}%计算。预测价格满足条件不代表实时结算一定获利，预测误差会影响实际收入。",
    ]
    for i, text in enumerate(principles, 1):
        _add_text(document, f"{i}. {text}")
    _add_picture(document, charts["average"], "图3-1  全年96点平均日储能功率与实时市场电价")
    _add_text(document,
        "图中横轴为一天96个时点，柱形为全年同一时刻的储能平均功率，充电为负、放电为正，"
        "折线为实时市场电价均值。全年平均曲线用于观察充放电的时间分布；不同日期平均后可能抵消，"
        "不能据此判断某一天是否充满或存在同时充放电，逐点核查应以配套充放电文件为准。")

    _add_heading(document, "四、首年运行与收益测算")
    _add_heading(document, "（一）配储建设前后新能源运行指标", 2)
    _add_table(document, ["指标", "建设前", "建设后", "变化"], [
        [f"{supply_label}(MWh)", _fmt(before,1), _fmt(after,1), _fmt(after-before,1)],
        ["市场上网电量(MWh)", _fmt(before,1), _fmt(export_after,1), _fmt(export_after-before,1)],
        ["等效利用小时(h)", _fmt(before/station), _fmt(after/station), _fmt((after-before)/station)],
        ["限电量(MWh)", _fmt(curtailed,1), _fmt(curtailed_after,1), _fmt(-captured,1)],
        ["限电比例(%)", _fmt(before_rate), _fmt(after_rate), _fmt(after_rate-before_rate)],
        ["实时市场平均结算电价(元/MWh)", _fmt(metrics['market_price_before']), _fmt(metrics['market_price_after']), _fmt(metrics['market_price_after']-metrics['market_price_before'])],
        ["含容量补偿等的综合收益单价(元/MWh)", _fmt(metrics['market_price_before']), _fmt(comprehensive_after), _fmt(comprehensive_after-metrics['market_price_before'])],
    ], widths=[5.4,3,3,3])
    _add_text(document,
        f"配储前项目发电量为{before:,.1f}MWh，限电电量{curtailed:,.1f}MWh。储能将其中"
        f"{captured:,.1f}MWh充入电池，限电量利用率为{metrics['curtailment_capture_pct']:.2f}%，"
        f"剩余限电量为{curtailed_after:,.1f}MWh，限电比例下降{before_rate-after_rate:.2f}个百分点。"
        "已充入储能的限电电量属于已发出的电量，不能再把储能转换损耗计回限电量。")
    _add_text(document,
        f"项目{ supply_label }按原发电量减去发电充电量、加上储能放电量计算，为"
        f"{before:,.1f}－{metrics['paid_charge_mwh']:,.1f}＋{metrics['discharge_mwh']:,.1f}＝{after:,.1f}MWh。"
        f"对应等效利用小时由{before/station:,.2f}变为{after/station:,.2f}小时。"
        f"其中抵厂用电放电{utility:,.1f}MWh，市场放电{metrics['market_discharge_mwh']:,.1f}MWh；"
        "市场结算电价仅采用市场上网电量加权，抵厂用电收益单独列示。")
    _add_text(document,
        f"建设前实时市场平均结算电价为{metrics['market_price_before']:.2f}元/MWh，建设后为"
        f"{metrics['market_price_after']:.2f}元/MWh。计入抵厂用电收益和容量补偿后的综合收益单价为"
        f"{comprehensive_after:.2f}元/MWh。储能首年增量收入相当于原实时市场收入的"
        f"{_fmt(income_increase)}%；增量收入除以配储后{ supply_label }折算为{_fmt(uplift)}元/MWh。"
        "实时市场结算电价和含补偿的收益折算值为不同指标，不能混用。")

    _add_heading(document, "（二）储能运行指标", 2)
    _add_table(document, ["指标", "测算值"], [
        ["全年充电量", f"{metrics['charge_mwh']:,.1f}MWh"],
        ["其中：限电电量充电", f"{captured:,.1f}MWh"],
        ["其中：偏差弥补", f"{metrics['supplemental_charge_mwh']:,.1f}MWh"],
        ["其中：付费发电充电", f"{metrics['paid_charge_mwh']:,.1f}MWh"],
        ["付费充电中：实时负价／非负且不高于充电上限", f"{negative_charged:,.1f}／{positive_charged:,.1f}MWh"],
        ["全年放电量", f"{metrics['discharge_mwh']:,.1f}MWh"],
        ["等效满循环（充电量÷额定容量）", f"{metrics['annual_cycles']:,.1f}次"],
        ["限电电量利用率", f"{metrics['curtailment_capture_pct']:.2f}%"],
        ["付费充电平均成本", f"{_fmt(metrics['average_paid_charge_price'])}元/MWh"],
        ["平均放电收益（含抵厂用电）", f"{_fmt(metrics['average_discharge_value'])}元/MWh"],
    ], widths=[8.8,5.6])
    _add_text(document,
        f"限电充电占全年充电量{captured/metrics['charge_mwh']*100 if metrics['charge_mwh'] else 0:.2f}%，"
        f"偏差弥补{metrics['supplemental_charge_mwh']:,.1f}MWh按免费电量处理，"
        f"付费充电{metrics['paid_charge_mwh']:,.1f}MWh按实时价格结算。名称中的“付费”表示计入市场充电成本，"
        "当实时价格为负时，该部分成本也可以为负。")
    _add_text(document,
        f"样本中实时负价时段的实际发电量为{negative_available:,.1f}MWh，储能在这些时段利用"
        f"发电电量充电{negative_charged:,.1f}MWh，占{negative_charged/negative_available*100 if negative_available else 0:.2f}%。"
        "该比例使用实际电量统计，不代表这些电量在日前均具备可预测、可充入且可于合格时段放出的条件。"
        "实时价格高于预测充电上限时仍可能执行原日前计划，因此实时分类不等于计划资格判断。")
    _add_table(document, ["月份", "充电(MWh)", "放电(MWh)", "等效循环(次)"], [
        [f"{y}-{month:02d}", _fmt(group['执行充电功率(MW)'].sum()*0.25,1),
         _fmt(group['执行放电功率(MW)'].sum()*0.25,1),
         _fmt(group['执行充电功率(MW)'].sum()*0.25/metrics['E'],1)]
        for (y,month),group in schedule.groupby(["年","月"])
    ], widths=[3.4,3.7,3.7,3.6])
    _add_picture(document, charts["monthly"], "图4-1  分月充放电量")
    _add_text(document,
        "分月结果反映样本年度的发电和价格季节差异。等效满循环以充电量除以额定容量计算，"
        "并非用放电量或可用SOC容量作分母。各月均值不能替代全年逐15分钟约束下的实际执行。")

    _add_heading(document, "（三）首年收益", 2)
    _add_table(document, ["收益项目", "金额(万元/年)"], [
        ["抵厂用电收益", _fmt(metrics['rev_tou_wan'],1)],
        ["市场放电收入", _fmt(metrics['rev_market_wan'],1)],
        ["减：充电成本", _fmt(metrics['charging_cost_wan'],1)],
        ["经营净收益", _fmt(metrics['R_year'],1)],
        ["容量补偿", _fmt(metrics['R_cap'],1)],
        ["首年总收入", _fmt(metrics['total_revenue'],1)],
    ], widths=[8,6.4])
    _add_text(document,
        f"市场放电收入为{metrics['rev_market_wan']:,.1f}万元，抵厂用电收益"
        f"{metrics['rev_tou_wan']:,.1f}万元，扣除充电成本{metrics['charging_cost_wan']:,.1f}万元后，"
        f"经营净收益为{metrics['R_year']:,.1f}万元。容量补偿为{metrics['R_cap']:,.1f}万元，"
        f"占首年总收入的{metrics['R_cap']/metrics['total_revenue']*100 if metrics['total_revenue'] else 0:.2f}%。"
        "此处总收入已扣充电成本，尚未扣生产费用、所得税和贷款本息。")
    _add_picture(document, charts["revenue"], "图4-2  首年收益构成")

    _add_heading(document, "五、投资估算与资金筹措")
    labels = [("电池系统（含舱内配套）","battery_system"),("土地及场地","land"),
              ("PCS升压一体化系统","pcs"),("站内公用及二次系统","station_equipment"),
              ("建筑安装工程","construction_install"),("前期研究及申报","predevelopment"),
              ("建设管理、勘察设计及监理","management_design_supervision"),("基本预备费","contingency")]
    _add_table(document, ["投资项目","金额(万元)","占比(%)"], [
        [label,_fmt(investment[key]/1e4,1),_fmt(investment[key]/finance['capex']*100,1)]
        for label,key in labels
    ] + [["合计",_fmt(metrics['capex'],1),"100.0"]], widths=[7.4,3.6,3.4])
    _add_text(document,
        f"本次初始投资{metrics['capex']:,.1f}万元，折合{finance['capex']/(metrics['P']*1e6):.3f}元/W、"
        f"{finance['capex']/(metrics['E']*1e6):.3f}元/Wh。电池系统单价采用网页输入的"
        f"{config['battery_cost_wan_per_mwh']:g}万元/MWh，PCS采用{config['pcs_cost_wan_per_mw']:g}万元/MW，"
        f"站内设备采用{config['station_equipment_cost_wan_per_mw']:g}万元/MW。"
        f"建筑安装费率{config['installation_rate_pct']:g}%，基本预备费率{config['contingency_rate_pct']:g}%。")
    optional = [("土地及场地", "include_land_cost"),("前期研究及申报", "include_predevelopment_cost"),
                ("建设管理、勘察设计及监理", "include_engineering_cost")]
    _add_text(document,
        "电池系统含BMS、热管理、电池舱及舱内消防；PCS系统含PCS、升压变压器及配套开关；"
        "站内公用系统含EMS、通信、计量、保护及站级消防。可选费用本次计入："
        + ("、".join(label for label,key in optional if config[key]) or "无") + "；未计入："
        + ("、".join(label for label,key in optional if not config[key]) or "无") + "。新建送出线路和主变扩容不在投资范围内。")
    replacement = finance.get('battery_replacement_year', replacement_year if finance['capex_rep'] > 0 else None) is not None
    initial_equity = finance['capex'] * finance['equity_ratio'] / 1e4
    _add_table(document, ["资金项目","金额(万元)","条件"], [
        ["初始资本金",_fmt(initial_equity,1),f"比例{config['equity_ratio_pct']:g}%"],
        ["初始贷款",_fmt(finance['loan1']/1e4,1),f"利率{config['loan_rate_pct']:g}%，偿还{min(config['loan_term_years'],config['project_years'])}年"],
        ["电池更换投资",_fmt(finance['capex_rep']/1e4,1),f"第{replacement_year}年末" if replacement else "评价期内不更换"],
        ["更换电池资本金",_fmt(finance['capex_rep']*finance['equity_ratio']/1e4,1),f"比例{config['equity_ratio_pct']:g}%"],
        ["更换电池贷款",_fmt(finance['loan2']/1e4,1),f"偿还{min(config['replacement_loan_term_years'],max(0,config['project_years']-replacement_year))}年"],
    ], widths=[5.4,3.6,5.4])
    _add_text(document,
        f"初始资本金为{initial_equity:,.1f}万元，贷款{finance['loan1']/1e4:,.1f}万元，按等额本息偿还。"
        + (f"第{replacement_year}年末更换电池，追加投资{finance['capex_rep']/1e4:,.1f}万元，"
           f"第{replacement_year+1}年容量重置；更换投资按相同资本金比例筹措。" if replacement else
           "项目评价期内不更换电池；期末余值按各类资产剩余经济寿命计算，不将所有资产统一按零残值处理。")
        + f"年生产费用按初始投资的{config['om_ratio_pct']:g}%计，为{finance['om']/1e4:,.1f}万元，"
          f"评价期末税后净回收价值{metrics['terminal_net_wan']:,.1f}万元。")

    _add_heading(document, "六、财务评价")
    _add_heading(document, "（一）财务指标", 2)
    _add_table(document, ["指标","全投资口径","资本金口径"], [
        ["净现值(万元)",_fmt(metrics['fi_npv'],1),_fmt(metrics['eq_npv'],1)],
        ["内部收益率(%)",_fmt(metrics['fi_irr']),_fmt(metrics['eq_irr'])],
        ["评价折现率(%)",_fmt(finance['wacc']*100),_fmt(config['equity_discount_rate_pct'])],
    ], widths=[5.6,4.4,4.4])
    _add_text(document,
        f"全投资内部收益率为{_fmt(metrics['fi_irr'])}%，全投资净现值{_fmt(metrics['fi_npv'],1)}万元；"
        f"资本金内部收益率为{_fmt(metrics['eq_irr'])}%，资本金净现值{_fmt(metrics['eq_npv'],1)}万元。"
        "两套指标分别对应项目总投资现金流和扣除贷款本息后的资本金现金流。资本金收益率包含杠杆影响，"
        "不能据此推断每年的可分配现金均为正；无法形成有效内部收益率的情形以“—”列示。")
    _add_heading(document, "（二）融资偿付与现金流", 2)
    negative_years = financing.loc[financing['equity_ncf_wan'] < -1e-8,'year'].astype(int).tolist()
    recovered = financing[financing['cumulative_wan'] >= 0]
    recovery_text = f"第{int(recovered.iloc[0]['year'])}年" if not recovered.empty else "评价期内未转正"
    _add_table(document, ["现金流指标","测算结果"], [
        ["最低偿债备付率",_fmt(financing['dscr'].min())+"倍"],
        ["资本金净现金流为负年份","、".join(map(str,negative_years)) or "无"],
        ["累计资本金现金流首次转正",recovery_text],
        ["评价期末累计资本金净现金流",_fmt(financing.iloc[-1]['cumulative_wan'],1)+"万元"],
    ], widths=[8,6.4])
    _add_table(document, ["年","收入","贷款本息","追加资本金","资本金净现金流","累计现金流"], [
        [str(int(r.year)),_fmt(r.revenue_wan,1),_fmt(r.debt_service_wan,1),
         _fmt(r.equity_injection_wan,1),_fmt(r.equity_ncf_wan,1),_fmt(r.cumulative_wan,1)]
        for r in financing.itertuples()
    ], widths=[0.9,2.2,2.5,2.7,3.1,3.0])
    _add_text(document,
        f"上表金额单位为万元，累计现金流包含期初资本金{initial_equity:,.1f}万元的流出。"
        f"累计资本金现金流首次转正为{recovery_text}，运营期资本金净现金流为负的年份为"
        f"{'、'.join(map(str,negative_years)) or '无'}。"
        "电池更换会产生追加资本金，其后可能同时偿还两笔贷款，年度现金流可能因此下降。"
        "偿债备付率按收入减生产费用及资本金口径所得税后的现金除以当年贷款本息计算。"
        "累计余额按前期现金留存在项目内计算，不等同于已经分红后的账户余额。")
    _add_picture(document, charts['cash'], "图6-1  全生命周期资本金现金流及贷款本息")

    _add_heading(document, "七、容量方案比较")
    display = comparison
    if len(display)>10:
        display = display.loc[(display['P']-metrics['P']).abs().nsmallest(10).index].sort_values('P')
    _add_table(document, ["功率(MW)","容量(MWh)","循环(次)","收入(万元)","投资(万元)","资本金IRR(%)"], [
        [_fmt(r.P,1),_fmt(r.E,1),_fmt(r.annual_cycles,1),_fmt(r.total_revenue,1),_fmt(r.capex,1),_fmt(r.eq_irr)]
        for r in display.itertuples()
    ], widths=[2.1,2.3,2.1,2.6,2.6,2.7])
    if len(comparison)<=1:
        _add_text(document,
            "本次相同条件下仅有一个同持续时长配置的测算结果，不能据此判断容量增大后的变化趋势，"
            "也不能将该配置称为全部容量组合中的最优方案。容量比较图不绘制未经计算的数据。"
            + ("所提供网格的选定配置造价与详细报告不一致，因此未作为同条件容量比较依据。" if not grid_consistent else ""))
    else:
        valid = successful[np.isfinite(successful['eq_irr'])]
        best = valid.loc[valid['eq_irr'].idxmax()] if not valid.empty else None
        _add_text(document,
            f"本次网格成功计算{len(successful)}种组合，其中{metrics['H']:g}小时配置{len(comparison)}种。"
            + (f"已计算范围内资本金内部收益率最高为{best['eq_irr']:.2f}%，对应{best['P']:g}MW/{best['E']:g}MWh。" if best is not None else "本次网格无有效资本金内部收益率。")
            + "表中列示与所选功率最接近的同持续时长组合；图中采用全部已算的同持续时长结果。收入增加与收益率增加含义不同，须同时对照投资和循环次数。")
        _add_picture(document, charts['capacity'], "图7-1  同持续时长下的收入与资本金收益率")

    _add_heading(document, "八、敏感性分析")
    _add_table(document, ["情景","全投资NPV","全投资IRR(%)","资本金NPV","资本金IRR(%)"], [
        [r['情景'],_fmt(r['全投资NPV'],1),_fmt(r['全投资IRR']),_fmt(r['资本金NPV'],1),_fmt(r['资本金IRR'])]
        for _,r in stresses.iterrows()
    ], widths=[4.0,2.6,2.6,2.6,2.6])
    _add_text(document,
        f"表中净现值单位为万元。各情景只改变列明的单一条件，保持本次{title_config}首年调度不变。"
        "造价变化同步影响初始投资、更换投资、折旧、生产费用和融资；经营收益变化仅调整经营净收益，"
        "首年容量补偿不随该比例变化；衰减情景按模型规则作用于后续年度收入。"
        "以上为财务敏感性分析，不代表新的电价预测或重新调度。")
    _add_text(document,
        f"上述情景下资本金净现值介于{stresses['资本金NPV'].min():,.1f}至{stresses['资本金NPV'].max():,.1f}万元，"
        f"全投资净现值介于{stresses['全投资NPV'].min():,.1f}至{stresses['全投资NPV'].max():,.1f}万元。"
        "内部收益率、净现值和贷款偿付现金应结合阅读；某个情景的收益率较高并不代表负现金流年份消失。")
    _add_picture(document, charts['sensitivity'], "图8-1  单因素财务敏感性分析")

    _add_heading(document, "九、主要不确定性与测算范围")
    _add_table(document, ["事项","本次口径及影响"], [
        ["市场与预测误差","用上传年度的预测安排计划、实时价格结算；其他年度价格结构和预测质量变化会改变收益。"],
        ["可用功率",f"来源为{metrics['available_power_source']}；估算数据会影响限电量及免费偏差弥补统计。"],
        ["容量补偿",f"输入标准{config['capacity_compensation_yuan_per_mw_month']:g}元/MW·月，机制比例{config['mechanism_ratio_pct']:g}%；按逐月尖峰平均执行放电功率等模型规则测算。"],
        ["造价与融资","采用本次输入值及勾选费用；未选项目不计入，不含新建送出线路、主变扩容及建设期资金成本。"],
        ["税费","计入输入的所得税和生产费用；未单独完整测算增值税、附加税、保险及税收优惠。"],
        ["电池性能","按固定效率、线性年衰减和设定更换年测算，未逐年重新求解衰减后的调度。"],
    ], widths=[3.3,11.1], aligns=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT])
    _add_text(document,
        "本报告使用用户上传的时序数据及网页填写条件，不用单一年份回测承诺未来收益。"
        "工程造价属于容量组合测算口径，具体采购报价、接入条件和合同结算差异应另行核对。"
        "报告只陈述条件、计算过程和结果，不作推进、投资或建设建议。")

    _add_heading(document, "十、测算结论与适用范围")
    _add_text(document,
        f"在当前条件下，{title_config}初始投资{metrics['capex']:,.1f}万元，首年总收入"
        f"{metrics['total_revenue']:,.1f}万元，资本金内部收益率{_fmt(metrics['eq_irr'])}%，"
        f"资本金净现值{metrics['eq_npv']:,.1f}万元。储能限电充电{captured:,.1f}MWh、"
        f"偏差弥补{metrics['supplemental_charge_mwh']:,.1f}MWh；配储后{ supply_label }变化"
        f"{after-before:+,.1f}MWh，实时市场平均结算电价变化"
        f"{metrics['market_price_after']-metrics['market_price_before']:+.2f}元/MWh。")
    _add_text(document,
        "上述结论仅对应本次数据、所选配置、调度规则和经济条件。完整逐15分钟充放电及逐年现金流"
        "保存在配套Excel文件中，可用来核查单日计划执行和年度财务结果。")

    paragraph = document.add_paragraph()
    paragraph.paragraph_format.first_line_indent = Pt(0)
    paragraph.add_run().add_break(WD_BREAK.PAGE)
    _add_heading(document, "附录  主要测算口径与资料来源")
    items = [
        f"数据来源为本次上传的全年生产与市场时序数据、方案A执行结果及同条件网格。样本含{metrics['points']:,}点，来源说明：{metrics['available_power_source']}。",
        f"评价期{config['project_years']}年，电池年衰减{config['battery_degradation_pct']:g}%；" + (f"第{replacement_year}年末更换电池，第{replacement_year+1}年容量重置。" if replacement else "评价期内不更换电池。"),
        f"资本金比例{config['equity_ratio_pct']:g}%，贷款利率{config['loan_rate_pct']:g}%，首次贷款输入期限{config['loan_term_years']}年，更换贷款输入期限{config['replacement_loan_term_years']}年，实际期限不超过剩余评价期。",
        f"资本金折现率{config['equity_discount_rate_pct']:g}%，加权平均资本成本{finance['wacc']*100:.2f}%，所得税率{config['income_tax_pct']:g}%，年生产费用为初始投资的{config['om_ratio_pct']:g}%。",
        "电池、功率设备、土建场地和前期费用分别按相应税务寿命折旧或摊销，更换年份与税务折旧寿命分别设置。原电池提前更换按零处置收入测算，未折完账面价值计入更换当年费用；评价期末按非电池资产剩余经济寿命估算可回收价值，税务账面价值与经济余值分开。",
    ]
    for i,text in enumerate(items,1):
        _add_text(document, f"{i}. {text}")
    _add_heading(document, "（一）运行与收益指标计算口径", 2)
    _add_text(document,
        f"限电量按实时价格不高于{config['curtailment_price_yuan_per_mwh']:g}元/MWh的时段，"
        "对可用功率减实际发电功率的正差额乘0.25小时累加。建设后限电量＝原限电量－限电充电量，"
        f"即{curtailed:,.1f}－{captured:,.1f}＝{curtailed_after:,.1f}MWh。限电比例分母为原发电量加原限电量"
        f"{available_energy:,.1f}MWh；效率损耗不重新计为限电，偏差弥补单独列示。")
    _add_text(document,
        f"整体供电量＝原发电量－付费发电充电量＋储能全部放电量；市场上网电量仅加市场放电量。"
        f"全年充放电转换损耗{conversion_loss:,.1f}MWh，年末相较年初增加库存{energy_delta:,.1f}MWh。"
        "整体供电增量也可用免费充电量减去转换损耗、再减库存增加量核对，避免遗漏偏差弥补或残余库存。")
    _add_text(document,
        f"实时市场平均结算电价：建设前为原实时市场收入{revenue_before:,.1f}万元×10,000÷"
        f"{before:,.1f}MWh；建设后为“原市场收入＋市场放电收入－付费充电成本”÷市场上网电量"
        f"{export_after:,.1f}MWh，结果{metrics['market_price_after']:.2f}元/MWh。容量补偿及抵厂用电收益"
        "不计入实时市场电价；两者计入综合收益单价。")
    _add_text(document,
        f"等效收益单价提升＝储能首年增量收入{metrics['total_revenue']:,.1f}万元×10,000÷配储后"
        f"{supply_label}{after:,.1f}MWh＝{_fmt(uplift)}元/MWh。等效满循环＝全年充电量"
        f"{metrics['charge_mwh']:,.1f}MWh÷额定容量{metrics['E']:g}MWh＝{metrics['annual_cycles']:,.1f}次。")

    document.core_properties.title = f"{config['project_name']}{title_config}投资分析报告"
    document.core_properties.subject = "配储寻优方案A测算"
    document.core_properties.author = ""
    output = io.BytesIO()
    document.save(output)
    return output.getvalue()
