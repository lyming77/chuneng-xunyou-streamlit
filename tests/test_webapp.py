import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from webapp.model_service import (
    best_valid_row,
    build_model_frame,
    default_config,
    make_power_grid,
    normalize_timeseries,
    run_configuration,
    run_grid,
    schedule_excel_bytes,
)
from webapp.report_builder import build_report_bytes


class WebAppTests(unittest.TestCase):
    def setUp(self):
        n = 96
        timestamp = pd.date_range("2026-01-01", periods=n, freq="15min")
        price = np.r_[np.zeros(48), np.full(48, 300.0)]
        self.timeseries = normalize_timeseries(pd.DataFrame({
            "时间戳": timestamp,
            "实际发电功率(MW)": np.full(n, 2.0),
            "超短期预测功率(MW)": np.full(n, 2.0),
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

    def test_missing_power_is_filled_by_existing_rule(self):
        raw = self.timeseries.copy()
        raw.loc[0, "实际发电功率(MW)"] = np.nan
        raw.loc[1, "超短期预测功率(MW)"] = np.nan
        raw.loc[2, "预测发电功率(日前,MW)"] = np.nan
        filled = normalize_timeseries(raw, require_full_year=False)
        self.assertFalse(filled[["实际发电功率(MW)", "超短期预测功率(MW)",
                                 "预测发电功率(日前,MW)"]].isna().any().any())
        self.assertEqual(filled.attrs["power_fill_stats"]["实际发电功率(MW)"], 1)

    def test_one_day_configuration_and_downloads(self):
        frame = build_model_frame(self.timeseries, self.config)
        detail = run_configuration(frame, self.config, 1.0, 2)
        metrics = detail["metrics"]
        self.assertGreater(metrics["charge_mwh"], 0)
        self.assertGreater(metrics["discharge_mwh"], 0)
        self.assertEqual(len(detail["schedule"]), 96)
        excel = schedule_excel_bytes(detail)
        self.assertTrue(excel.startswith(b"PK"))
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


if __name__ == "__main__":
    unittest.main()
