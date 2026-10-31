import sys
import io
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from webapp.model_service import (
    DATA_SCHEMA_VERSION,
    DEFAULT_COEF,
    DEFAULT_ZONE_TEXT,
    best_valid_row,
    build_zones,
    build_model_frame,
    cost_benchmark,
    cost_params,
    default_config,
    make_power_grid,
    normalize_timeseries,
    run_configuration,
    run_grid,
    schedule_excel_bytes,
    upload_template_csv,
)
from webapp.report_builder import build_report_bytes
from 读取收资表格 import zone_of


class WebAppTests(unittest.TestCase):
    def setUp(self):
        n = 96
        timestamp = pd.date_range("2026-01-01", periods=n, freq="15min")
        price = np.r_[np.zeros(48), np.full(48, 300.0)]
        self.timeseries = normalize_timeseries(pd.DataFrame({
            "时间戳": timestamp,
            "实际发电功率(MW)": np.full(n, 2.0),
            "超短期预测功率(MW)": np.full(n, 2.0),
            "可用功率(MW)": np.full(n, 2.0),
            "预测发电功率(日前,MW)": np.full(n, 2.0),
            "实时市场价格(元/MWh)": price,
            "预测价格(元/MWh)": price,
        }), require_full_year=False)
        self.config = default_config()
        self.config.update({
            "project_name": "网页测试项目",
            "station_power_mw": 10.0,
            "aux_power_mw": 0.0,
        })

    def test_power_grid_matches_existing_rule(self):
        self.assertEqual(make_power_grid(76.5, "风电", 30),
                         [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0])

    def test_power_bounds_allow_single_and_multiple_configurations(self):
        self.assertEqual(make_power_grid(480, "光伏", minimum_power_mw=100,
                                         maximum_power_mw=100), [100.0])
        self.assertEqual(make_power_grid(100, "风电", minimum_power_mw=10,
                                         maximum_power_mw=20), [10.0, 15.0, 20.0])
        self.assertEqual(make_power_grid(95, "风电", minimum_power_mw=6.5,
                                         maximum_power_mw=6.5), [6.5])
        self.assertEqual(make_power_grid(100, "风电", minimum_power_mw=7,
                                         maximum_power_mw=12), [7.0, 10.0, 12.0])
        for lower, upper in [(10, 5), (0, 5), (1, np.inf), (np.nan, 5)]:
            with self.subTest(lower=lower, upper=upper), self.assertRaises(ValueError):
                make_power_grid(100, "风电", minimum_power_mw=lower, maximum_power_mw=upper)

    def test_selected_power_range_controls_combinations_actually_calculated(self):
        frame = build_model_frame(self.timeseries, self.config)
        powers = make_power_grid(10, "风电", minimum_power_mw=2, maximum_power_mw=2)
        result = run_grid(frame, self.config, powers, [2, 4])
        self.assertEqual(list(zip(result["P"], result["H"])), [(2.0, 2), (2.0, 4)])
        self.assertTrue(result["error"].eq("").all())

    def test_battery_replacement_cash_flow_is_in_year_ten(self):
        frame = build_model_frame(self.timeseries, self.config)
        detail = run_configuration(frame, self.config, 1.0, 2)
        yearly = detail["finance"]["yearly"]
        self.assertEqual([row["y"] for row in yearly if row["capex_cf"] > 0], [10])
        self.assertEqual(yearly[10]["cap_factor"], 1.0)

    def test_missing_power_is_filled_by_existing_rule(self):
        raw = self.timeseries.copy()
        raw.loc[0, "实际发电功率(MW)"] = np.nan
        raw.loc[1, "超短期预测功率(MW)"] = np.nan
        raw.loc[2, "预测发电功率(日前,MW)"] = np.nan
        filled = normalize_timeseries(raw, require_full_year=False)
        self.assertFalse(filled[["实际发电功率(MW)", "超短期预测功率(MW)",
                                 "预测发电功率(日前,MW)"]].isna().any().any())
        self.assertEqual(filled.attrs["power_fill_stats"]["实际发电功率(MW)"], 1)

    def test_available_power_is_distinct_from_theoretical_and_forecast_power(self):
        raw = self.timeseries.copy()
        raw["实际发电功率(MW)"] = 2.0
        raw["可用功率(MW)"] = 3.0
        raw["超短期预测功率(MW)"] = 8.0
        raw["理论功率(MW)"] = 10.0
        raw["实时市场价格(元/MWh)"] = -100.0
        normalized = normalize_timeseries(raw, require_full_year=False)
        frame = build_model_frame(normalized, self.config)
        np.testing.assert_allclose(frame["avail"], 3.0)
        np.testing.assert_allclose(frame["curt"], 1.0)
        np.testing.assert_allclose(normalized["超短期预测功率(MW)"], 8.0)
        self.assertFalse(frame.attrs["available_power_is_estimate"])

    def test_legacy_upload_is_explicitly_marked_as_estimated(self):
        raw = self.timeseries.drop(columns="可用功率(MW)")
        raw["理论功率(MW)"] = 10.0
        with self.assertWarnsRegex(UserWarning, "理论功率未被采用"):
            normalized = normalize_timeseries(raw, require_full_year=False)
        frame = build_model_frame(normalized, self.config)
        np.testing.assert_allclose(frame["avail"], 2.0)
        self.assertTrue(frame.attrs["available_power_is_estimate"])
        self.assertIn("估算", frame.attrs["available_power_source"])

    def test_incomplete_available_power_is_not_filled_from_other_power(self):
        for value in (np.nan, np.inf, -1.0, "错误"):
            with self.subTest(value=value):
                raw = self.timeseries.copy()
                raw["可用功率(MW)"] = raw["可用功率(MW)"].astype(object)
                raw.loc[0, "可用功率(MW)"] = value
                raw["理论功率(MW)"] = 10.0
                with self.assertRaisesRegex(ValueError, "可用功率.*请补齐真实可用功率"):
                    normalize_timeseries(raw, require_full_year=False)

    def test_available_power_alias_does_not_overwrite_short_term_forecast(self):
        raw = self.timeseries.rename(columns={"可用功率(MW)": "可用发电功率(MW)"})
        raw["可用发电功率(MW)"] = 3.0
        normalized = normalize_timeseries(raw, require_full_year=False)
        np.testing.assert_allclose(normalized["可用功率(MW)"], 3.0)
        np.testing.assert_allclose(normalized["超短期预测功率(MW)"], 2.0)

    def test_template_requests_available_power_instead_of_theoretical_power(self):
        columns = pd.read_csv(io.BytesIO(upload_template_csv()), nrows=0).columns
        self.assertIn("可用功率(MW)", columns)
        self.assertNotIn("理论功率(MW)", columns)

    def test_schema_update_clears_old_saved_results_before_calculation(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(ROOT / "streamlit_app.py"))
        app.session_state["data_schema_version"] = "old-theoretical-power"
        app.session_state["saved_signature"] = "old-saved-data"
        for key in ("grid_result", "detail_result", "report_bytes", "schedule_bytes",
                    "timeseries", "model_frame", "upload_bytes"):
            app.session_state[key] = "old-result"
        app.run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["data_schema_version"], DATA_SCHEMA_VERSION)
        for key in ("saved_signature", "grid_result", "detail_result", "report_bytes", "schedule_bytes"):
            self.assertIsNone(app.session_state[key])
        for key in ("timeseries", "model_frame", "upload_bytes"):
            self.assertNotIn(key, app.session_state)
        self.assertTrue(any("旧会话测算结果已清除" in item.value for item in app.info))

    def test_saving_yearly_upload_unlocks_power_and_duration_selection(self):
        from streamlit.testing.v1 import AppTest
        yearly = pd.concat([self.timeseries] * 365, ignore_index=True)
        yearly["时间戳"] = pd.date_range("2026-01-01", periods=len(yearly), freq="15min")
        data = yearly.to_csv(index=False).encode("utf-8-sig")
        upload = SimpleNamespace(name="available-power-test.csv", getvalue=lambda: data)
        with patch("streamlit.file_uploader", return_value=upload):
            app = AppTest.from_file(str(ROOT / "streamlit_app.py")).run(timeout=30)
            next(w for w in app.number_input if w.label == "新能源装机容量（MW）").set_value(10.0)
            next(w for w in app.button if w.label == "保存数据与参数").click().run(timeout=30)
        self.assertFalse(app.exception)
        np.testing.assert_allclose(app.session_state["model_frame"]["avail"], 2.0)
        self.assertEqual(len(app.session_state["model_frame"]), 35040)
        labels = {w.label for w in app.number_input}
        self.assertIn("储能功率下限（MW）", labels)
        self.assertIn("储能功率上限（MW）", labels)
        self.assertTrue(any("第一步已完成" in item.value for item in app.success))

    def test_supplemental_charge_uses_available_power_not_theoretical_power(self):
        from 方案B_滚动MPC import solve_mpc
        plans = [np.zeros(96) for _ in range(4)]
        plans[1][0] = 4.0  # 首点实际少充1MWh,下一点可免费补缺口。
        raw = self.timeseries.copy()
        raw["实际发电功率(MW)"] = 0.0
        raw["可用功率(MW)"] = 0.0
        raw.loc[1, "可用功率(MW)"] = 1.0
        raw["超短期预测功率(MW)"] = 8.0
        raw["理论功率(MW)"] = 10.0
        frame = build_model_frame(normalize_timeseries(raw, require_full_year=False), self.config)
        with patch("方案B_滚动MPC.solve_window_a", return_value=plans):
            _, _, _, schedule = solve_mpc(4, 4, frame, return_schedule=True, P_STATION=10)
        self.assertAlmostEqual(schedule["cc"][1], 1.0)
        self.assertAlmostEqual(schedule["cc"].sum() * 0.25, 0.25)

    def test_one_day_configuration_and_downloads(self):
        frame = build_model_frame(self.timeseries, self.config)
        detail = run_configuration(frame, self.config, 1.0, 2)
        metrics = detail["metrics"]
        self.assertGreater(metrics["charge_mwh"], 0)
        self.assertGreater(metrics["discharge_mwh"], 0)
        self.assertEqual(len(detail["schedule"]), 96)
        excel = schedule_excel_bytes(detail)
        self.assertTrue(excel.startswith(b"PK"))
        summary = pd.read_excel(io.BytesIO(excel), sheet_name="全年汇总").set_index("指标")
        self.assertEqual(summary.loc["可用功率来源", "数值"], "上传的可用功率")
        grid = pd.DataFrame([{**{key: metrics.get(key, np.nan) for key in (
            "P", "H", "E", "annual_cycles", "total_revenue", "capex", "fi_npv", "eq_irr")},
                                  "error": ""}])
        report = build_report_bytes(self.config, grid, detail)
        self.assertTrue(report.startswith(b"PK"))

    def test_grid_reports_day_progress_and_partial_rows(self):
        frame = build_model_frame(self.timeseries, self.config)
        progress = []
        partial = []
        result = run_grid(
            frame, self.config, [1.0], [2],
            lambda done, total, label: progress.append((done, total, label)),
            lambda rows: partial.append(rows.copy()),
        )
        self.assertEqual(len(result), 1)
        self.assertTrue(any("天" in label for _, _, label in progress))
        self.assertEqual(len(partial), 1)
        self.assertEqual(len(partial[0]), 1)

    def test_best_valid_row_handles_all_missing_values(self):
        frame = pd.DataFrame({"P": [1.0, 2.0], "eq_irr": [np.nan, np.inf],
                              "fi_npv": [-20.0, -10.0]})
        self.assertIsNone(best_valid_row(frame, "eq_irr"))
        self.assertEqual(best_valid_row(frame, "fi_npv")["P"], 2.0)

    def test_optional_cost_items_follow_project_switches(self):
        params = cost_params(self.config)
        self.assertEqual(params["land_per_mwh"], 0.0)
        self.assertEqual(params["fixed_predevelopment"], 0.0)
        self.assertEqual(params["engineering_rate"], 0.0)
        self.config.update({
            "include_land_cost": True,
            "include_predevelopment_cost": True,
            "include_engineering_cost": True,
        })
        params = cost_params(self.config)
        self.assertGreater(params["land_per_mwh"], 0.0)
        self.assertGreater(params["fixed_predevelopment"], 0.0)
        self.assertGreater(params["engineering_rate"], 0.0)

    def test_two_hour_cost_benchmark_reports_power_and_energy_unit_prices(self):
        benchmark = cost_benchmark(self.config, power_mw=10.0, hours=2.0)
        self.assertGreater(benchmark["total_wan"], 0.0)
        self.assertAlmostEqual(benchmark["yuan_per_w"], benchmark["yuan_per_wh"] * 2.0)

    def test_new_shandong_tou_periods_are_the_defaults(self):
        zones = build_zones(DEFAULT_ZONE_TEXT)
        self.assertEqual(zone_of(zones, 3, 8 * 4), "低谷")
        self.assertEqual(zone_of(zones, 3, 10 * 4), "深谷")
        self.assertEqual(zone_of(zones, 6, 22 * 4), "高峰")
        self.assertEqual(zone_of(zones, 7, 10 * 4), "平段")
        self.assertEqual(zone_of(zones, 10, 19 * 4), "尖峰")
        self.assertEqual(zone_of(zones, 10, 21 * 4), "高峰")
        self.assertEqual(DEFAULT_COEF,
                         {"尖峰": 2.0, "高峰": 1.7, "平段": 1.0, "低谷": 0.3, "深谷": 0.1})


if __name__ == "__main__":
    unittest.main()
