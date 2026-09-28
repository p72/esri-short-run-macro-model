"""ソルバーの異常系と回帰テスト（Issue #17）.

実行: python -m unittest discover -s tests -v
"""
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import model as M  # noqa: E402
from model import Eq, Model  # noqa: E402

IDX = pd.period_range("2020Q1", periods=3, freq="Q")


def toy_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.DataFrame({"x": [1.0, 2.0, 3.0], "z": [1.0, 2.0, 3.0]}, index=IDX)
    af = pd.DataFrame({"x": [0.0, 10.0, 20.0]}, index=IDX)
    return d, af


class SolverGuards(unittest.TestCase):
    def test_nan_rhs_is_error(self):
        d, af = toy_data()
        with self.assertRaisesRegex(ValueError, "右辺が有限値ではありません"):
            Model([Eq(1, "x", "level", lambda v: np.nan)]).solve(d, "2020Q1", "2020Q3", af)

    def test_nan_reference_is_error(self):
        # max(0.001, NaN) のように下限関数が欠損を黙って置き換えるのを防ぐ
        d, af = toy_data()
        d.loc[IDX[1], "z"] = np.nan
        with self.assertRaisesRegex(ValueError, "z"):
            Model([Eq(1, "x", "level", lambda v: max(0.001, v("z")))]).solve(d, "2020Q1", "2020Q3", af)

    def test_nan_initial_value_is_error(self):
        d, af = toy_data()
        d.loc[IDX[0], "x"] = np.nan
        with self.assertRaisesRegex(ValueError, "初期値"):
            Model([Eq(1, "x", "level", lambda v: v("z"))]).solve(d, "2020Q1", "2020Q3", af)

    def test_lag_before_data_is_error(self):
        d, af = toy_data()
        with self.assertRaisesRegex(IndexError, "範囲外"):
            Model([Eq(1, "x", "level", lambda v: v("z", 1))]).solve(d, "2020Q1", "2020Q1", af)

    def test_lead_after_data_is_error(self):
        d, af = toy_data()
        with self.assertRaisesRegex(IndexError, "範囲外"):
            Model([Eq(1, "x", "level", lambda v: v("z", -1))]).solve(d, "2020Q3", "2020Q3", af)

    def test_add_factors_aligned_by_label(self):
        d, af = toy_data()
        m = Model([Eq(1, "x", "level", lambda v: v("z"))])
        a = m.solve(d, "2020Q1", "2020Q3", af)["x"].tolist()
        b = m.solve(d, "2020Q1", "2020Q3", af.iloc[::-1])["x"].tolist()
        self.assertEqual(a, [1.0, 12.0, 23.0])
        self.assertEqual(a, b)

    def test_missing_add_factor_period_is_error(self):
        d, af = toy_data()
        with self.assertRaisesRegex(ValueError, "誤差項に 1 期の欠落"):
            Model([Eq(1, "x", "level", lambda v: v("z"))]).solve(d, "2020Q1", "2020Q3", af.iloc[:2])

    def test_non_consecutive_index_is_error(self):
        d, af = toy_data()
        for bad in (d.iloc[::-1], d.iloc[[0, 2]]):
            with self.assertRaisesRegex(ValueError, "昇順・連続"):
                Model([Eq(1, "x", "level", lambda v: v("z"))]).solve(bad, bad.index[0].strftime("%YQ%q"),
                                                                     bad.index[-1].strftime("%YQ%q"), af)

    def test_non_quarterly_index_is_error(self):
        d, af = toy_data()
        d.index = pd.RangeIndex(3)
        with self.assertRaisesRegex(ValueError, "四半期"):
            Model([Eq(1, "x", "level", lambda v: v("z"))])._check_index(d)

    def test_fixed_path_with_gap_is_error(self):
        d, af = toy_data()
        with self.assertRaisesRegex(ValueError, "固定値 x"):
            Model([Eq(1, "x", "level", lambda v: v("z"))]).solve(
                d, "2020Q1", "2020Q3", af, fixed={"x": pd.Series([1.0, 2.0], index=IDX[:2])})

    def test_simultaneous_system_residual_zero(self):
        # x = 0.5 y + z, y = 0.5 x（同時方程式）: 解いた後の残差検査を通る
        d = pd.DataFrame({"x": [1.0] * 3, "y": [1.0] * 3, "z": [1.0, 2.0, 3.0]}, index=IDX)
        af = pd.DataFrame(0.0, index=IDX, columns=["x", "y"])
        m = Model([Eq(1, "x", "level", lambda v: 0.5 * v("y") + v("z")), Eq(2, "y", "level", lambda v: 0.5 * v("x"))])
        out = m.solve(d, "2020Q1", "2020Q3", af)
        np.testing.assert_allclose(out["x"], d["z"] / 0.75, rtol=1e-9)


class FullModel(unittest.TestCase):
    """同梱データで、標準解が実績を再現し、11シナリオの保存値と一致することを確かめる."""

    @classmethod
    def setUpClass(cls):
        import simulate as S
        cls.S = S
        data = pd.read_csv(ROOT / "data" / "processed" / "model_data.csv", index_col="period")
        data.index = pd.PeriodIndex(data.index, freq="Q")
        cls.data = data

    def test_baseline_reproduces_data(self):
        m = Model()
        af = m.add_factors(self.data, self.S.SOLVE_START, self.S.END)
        base = m.solve(self.data, self.S.SOLVE_START, self.S.END, af)
        w = slice(self.S.SOLVE_START, self.S.END)
        for c in ["GDP", "CP", "IFP", "GDPV", "PCP", "UR", "YDV", "BGV"]:
            np.testing.assert_allclose(base.loc[w, c], self.data.loc[w, c], rtol=1e-8, err_msg=c)

    def test_failed_scenario_writes_nothing(self):
        S = self.S
        orig = S.build_scenarios

        def broken(base, af):
            sc = orig(base, af)[:1]
            sc.append(S.Scenario(99, "壊れたシナリオ", overrides={"GDP": Eq(2, "GDP", "level", lambda v: np.nan)}))
            return sc
        tag = "_unittest_fail"
        S.build_scenarios = broken
        argv = sys.argv
        sys.argv = ["simulate.py", "--tag", tag]
        try:
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as cm:
                S.main()
            self.assertIn("99", str(cm.exception.code))
        finally:
            S.build_scenarios = orig
            sys.argv = argv
        self.assertFalse(list((ROOT / "output").glob(f"*{tag}*")))


if __name__ == "__main__":
    unittest.main()
