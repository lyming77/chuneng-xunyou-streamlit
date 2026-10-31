import sys
from pathlib import Path
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '脚本'))
import 寻优 as M


class InvestmentCostTests(unittest.TestCase):
    def test_legacy_cost_reproduces_old_10mw_40mwh_result(self):
        cost = M.investment_breakdown(10, 4, cost_model="legacy")
        self.assertAlmostEqual(cost["total"] / 1e4, 2610.0, places=6)

    def test_detailed_cost_adds_full_engineering_scope(self):
        cost = M.investment_breakdown(10, 4, cost_model="detailed")
        self.assertAlmostEqual(cost["total"] / 1e4, 3169.2634285714287, places=6)
        self.assertGreater(cost["construction_install"], 0)
        self.assertGreater(cost["predevelopment"], 0)
        self.assertGreater(cost["management_design_supervision"], 0)
        self.assertGreater(cost["contingency"], 0)
        self.assertNotIn("outgoing_line", cost)
        self.assertNotIn("main_transformer_expansion", cost)

    def test_fixed_cost_makes_small_system_unit_cost_higher(self):
        small = M.investment_breakdown(1, 4, cost_model="detailed")
        large = M.investment_breakdown(10, 4, cost_model="detailed")
        self.assertGreater(small["total"] / 4, large["total"] / 40)

    def test_discrete_check_rounds_equipment_units_up(self):
        cost = M.investment_breakdown(3, 4, cost_model="detailed", discrete=True)
        self.assertEqual(cost["priced_energy_mwh"], 15.0)
        self.assertEqual(cost["priced_power_mw"], 5.0)

    def test_project_cost_parameters_can_override_defaults(self):
        base = M.investment_breakdown(10, 4, cost_model="detailed")
        changed = M.investment_breakdown(
            10, 4, cost_model="detailed",
            cost_params={"land_per_mwh": 0.0, "fixed_predevelopment": 1000000.0})
        self.assertLess(changed["total"], base["total"])
        self.assertEqual(changed["land"], 0.0)
        self.assertEqual(changed["predevelopment"], 1000000.0)


class ResidualValueTests(unittest.TestCase):
    def test_ten_year_project_keeps_non_battery_terminal_value(self):
        finance = M.project_finance(10, 4, 0, 0, n_year=10, cost_model="detailed")
        self.assertAlmostEqual(finance["terminal_value"] / 1e4, 332.2666666666667, places=6)
        self.assertAlmostEqual(finance["terminal_net"] / 1e4, 291.925, places=6)
        self.assertEqual(finance["yearly"][-1]["terminal_value"], finance["terminal_value"])

    def test_twenty_year_project_has_no_terminal_value_under_default_lives(self):
        finance = M.project_finance(10, 4, 0, 0, n_year=20, cost_model="detailed")
        self.assertAlmostEqual(finance["terminal_value"], 0.0, places=6)

    def test_legacy_mode_keeps_zero_terminal_value(self):
        finance = M.project_finance(10, 4, 0, 0, n_year=10, cost_model="legacy")
        self.assertAlmostEqual(finance["terminal_value"], 0.0, places=6)
        self.assertAlmostEqual(sum(row["dep"] for row in finance["yearly"]), finance["capex"], places=6)


class ReplacementTimingTests(unittest.TestCase):
    def test_eight_year_replacement_moves_investment_reset_and_loan(self):
        finance = M.project_finance(2, 4, 700000, 50000, n_year=20,
                                    loan_term=5, replacement_loan_term=7,
                                    battery_replacement_year=8)
        rows = finance["yearly"]
        self.assertEqual(finance["battery_replacement_year"], 8)
        self.assertEqual([r['y'] for r in rows if r['capex_cf'] > 0], [8])
        self.assertEqual([r['y'] for r in rows if r['equity_inj'] > 0], [8])
        self.assertAlmostEqual(rows[7]['cap_factor'], 0.86)
        self.assertEqual(rows[8]['cap_factor'], 1.0)
        self.assertAlmostEqual(rows[16]['cap_factor'], 0.84)  # 不在第17年无投资重置
        self.assertEqual(rows[7]['principal'], 0)
        self.assertGreater(rows[8]['principal'], 0)
        self.assertGreater(rows[14]['principal'], 0)
        self.assertEqual(rows[15]['principal'], 0)
        self.assertAlmostEqual(sum(r['principal'] for r in rows), finance['loan1'] + finance['loan2'])

    def test_replacement_on_or_after_project_end_is_omitted(self):
        for year in (10, 15):
            finance = M.project_finance(2, 4, 700000, 50000, n_year=10,
                                        battery_replacement_year=year)
            self.assertIsNone(finance['battery_replacement_year'])
            self.assertEqual(finance['capex_rep'], 0)
            self.assertEqual(finance['loan2'], 0)
            self.assertFalse(any(r['capex_cf'] for r in finance['yearly']))
            self.assertAlmostEqual(finance['yearly'][-1]['cap_factor'], 0.82)

    def test_early_replacement_stops_original_battery_depreciation(self):
        finance = M.project_finance(2, 4, 700000, 50000, n_year=20,
                                    battery_replacement_year=8)
        self.assertAlmostEqual(sum(r['dep'] for r in finance['yearly']),
                               finance['capex'] + finance['capex_rep'])
        self.assertGreater(finance['yearly'][7]['dep'], finance['yearly'][6]['dep'])


if __name__ == "__main__":
    unittest.main()
