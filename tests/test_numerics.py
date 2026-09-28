"""Numerical regressions: run with python -m unittest discover -s tests -v."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import model as M
from experiment_fidelity import eq134_printed
import ecm
import simulate
import check_transcription


class SolverTests(unittest.TestCase):
    def setUp(self):
        self.idx = pd.period_range("2020Q1", periods=3, freq="Q")
        self.data = pd.DataFrame({"x": [1., 2., 3.], "z": [1., 2., 3.]}, index=self.idx)
        self.model = M.Model([M.Eq(1, "x", "level", lambda v: v("z"))])
        self.af = pd.DataFrame({"x": [0., 10., 20.]}, index=self.idx)

    def test_add_factors_align_by_quarter(self):
        normal = self.model.solve(self.data, "2020Q1", "2020Q3", self.af)
        shuffled = self.model.solve(self.data, "2020Q1", "2020Q3", self.af.iloc[::-1])
        pd.testing.assert_frame_equal(normal, shuffled)

    def test_missing_factor_quarter_rejected(self):
        with self.assertRaises(ValueError):
            self.model.solve(self.data, "2020Q1", "2020Q3", self.af.iloc[1:])

    def test_nonfinite_rhs_rejected(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                model = M.Model([M.Eq(1, "x", "level", lambda v: value)])
                model.solve(self.data, "2020Q1", "2020Q3", self.af)

    def test_nonfinite_input_cannot_hide_behind_max(self):
        self.data.loc[self.idx[0], "z"] = np.nan
        model = M.Model([M.Eq(113, "x", "level", lambda v: max(0.001, v("z")))])
        for operation in ("solve", "add_factors"):
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                if operation == "solve":
                    model.solve(self.data, "2020Q1", "2020Q3", self.af)
                else:
                    model.add_factors(self.data, "2020Q1", "2020Q3")

    def test_nan_initial_value_rejected(self):
        self.data.loc[self.idx[0], "x"] = np.nan
        with self.assertRaises(ValueError):
            self.model.solve(self.data, "2020Q1", "2020Q3", self.af)

    def test_fixed_nan_rejected_even_without_dependents(self):
        fixed = pd.Series([1., np.nan, 3.], index=self.idx)
        with self.assertRaises(ValueError):
            self.model.solve(self.data, "2020Q1", "2020Q3", self.af, fixed={"x": fixed})

    def test_out_of_range_lags_and_leads_rejected(self):
        for lag, period in ((1, "2020Q1"), (-1, "2020Q3")):
            model = M.Model([M.Eq(1, "x", "level", lambda v: v("z", lag))])
            for operation in ("solve", "add_factors"):
                with self.subTest(lag=lag, operation=operation), self.assertRaises(ValueError):
                    if operation == "solve":
                        model.solve(self.data, period, period, self.af)
                    else:
                        model.add_factors(self.data, period, period)

    def test_invalid_time_axis_rejected(self):
        for data in (self.data.iloc[::-1], self.data.iloc[[0, 2]], self.data.iloc[[0, 0, 2]]):
            with self.subTest(index=str(data.index)), self.assertRaises(ValueError):
                self.model.add_factors(data, "2020Q1", "2020Q3")
        with self.assertRaises(ValueError):
            self.model.solve(self.data, "2020Q3", "2020Q1", self.af)

    def test_valid_lag_lead_and_coupled_system(self):
        model = M.Model([M.Eq(1, "x", "level", lambda v: 0.5 * v("z") + v("x", 1)),
                         M.Eq(2, "z", "level", lambda v: 0.25 * v("x") + v("z", -1))])
        af = pd.DataFrame(0., index=self.idx, columns=["x", "z"])
        result = model.solve(self.data, "2020Q2", "2020Q2", af)
        # x = z/2 + 1; z = x/4 + 3 => x = 20/7, z = 26/7.
        self.assertAlmostEqual(result.loc["2020Q2", "x"], 20 / 7, places=8)
        self.assertAlmostEqual(result.loc["2020Q2", "z"], 26 / 7, places=8)


class PaperTests(unittest.TestCase):
    def test_transcription_rejects_missing_identity(self):
        # An identity has no non-structural decimal coefficient: empty sets
        # alone cannot detect that the whole equation has disappeared.
        paper = {n: [] for n in range(1, 153)}
        implementation = {n: set() for n in range(1, 153) if n != 2}
        with patch.object(Path, "exists", return_value=True), \
                patch.object(check_transcription, "paper_blocks", return_value=paper), \
                patch.object(check_transcription, "model_blocks", return_value=implementation):
            with self.assertRaisesRegex(SystemExit, "欠落 \\[2\\]"):
                check_transcription.main()

    def test_equation_134_printed_typo_only_in_dummy_term(self):
        # RN72 printed pp.66-67: the first LOG uses multiplication; only
        # the LOG multiplied by -0.011036*DTCIC2**2 uses 1+RTCI+PRTIF.
        vals = dict(RTCI=.08, PRTCP=.52, PRTIF=.18, PRTIH=.8, PRTCG=.48,
                    PRTIG=.45, CPV=300., IFPV=90., IHPV=20., CGV=100., IGV=25., D972C=1.)
        corrected = next(e for e in M.build_equations() if e.no == 134)
        good = sum(.08 / (1 + .08 * vals[p]) * vals[n] for n, p in
                   [("CPV", "PRTCP"), ("IFPV", "PRTIF"), ("IHPV", "PRTIH"),
                    ("CGV", "PRTCG"), ("IGV", "PRTIG")])
        typo = good - .08 / (1 + .08 * .18) * 90 + .08 / (1 + .08 + .18) * 90
        for dummy in (0., 1.):
            vals["DTCIC2"] = dummy
            get = lambda n, k=0: vals[n]
            expected = .915210 * np.log(good) - .011036 * dummy ** 2 * np.log(typo) + .191987
            self.assertAlmostEqual(eq134_printed().rhs(get), expected, places=12)
            if dummy == 0:
                self.assertEqual(eq134_printed().rhs(get), corrected.rhs(get))


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.data = pd.read_csv(root / "data/processed/model_data.csv", index_col="period")
        cls.data.index = pd.PeriodIndex(cls.data.index, freq="Q")

    def test_all_equation_transforms_round_trip(self):
        # Includes the custom inversions for ratios, stocks and effective tax rates.
        t = self.data.index.get_loc(pd.Period("2018Q1", "Q"))
        arrays = M.Model._arrays(self.data)
        for itr in ("dlog", "level"):
            with patch.object(M, "ITR_FORM", itr):
                for eq in M.build_equations():
                    with self.subTest(itr=itr, equation=eq.no):
                        get = M.Model._getter(arrays, self.data.index, t)
                        y = eq.lhs_value(get) + .001
                        val = eq.invert(get, y)
                        changed = lambda n, k=0: val if n == eq.name and k == 0 else get(n, k)
                        self.assertAlmostEqual(eq.lhs_value(changed), y, places=10)

    def test_all_scenarios_satisfy_equations(self):
        start, end = simulate.SOLVE_START, simulate.END
        model = M.Model()
        af = model.add_factors(self.data, start, end)
        base = model.solve(self.data, start, end, af)
        np.testing.assert_allclose(base.loc[start:end, model.endog],
                                   self.data.loc[start:end, model.endog], rtol=1e-10, atol=1e-9)
        over, cols = ecm.frozen_overrides(model, base, simulate.solve_mask(base.index))
        for scenario in simulate.build_scenarios(base, af):
            with self.subTest(scenario=scenario.no):
                data = base.assign(**{**scenario.data, **cols})
                overrides = {**over, **scenario.overrides}
                result = model.solve(data, start, end, af, fixed=scenario.fixed,
                                     shocks=scenario.shocks, overrides=overrides)
                arrays = model._arrays(result)
                # Evaluate every equation against the completed simultaneous solution,
                # not against the partially updated Gauss-Seidel sweep.
                for t in model._positions(result, start, end):
                    get = model._getter(arrays, result.index, t)
                    for original in model.eqs:
                        eq = overrides.get(original.name, original)
                        if eq.name in scenario.fixed:
                            self.assertEqual(get(eq.name), scenario.fixed[eq.name].iloc[t])
                            continue
                        a = af[eq.name].iloc[t] if eq.name in af else 0.
                        s = scenario.shocks[eq.name].iloc[t] if eq.name in scenario.shocks else 0.
                        residual = abs(get(eq.name) - eq.invert(get, eq.rhs(get) + a + s))
                        self.assertLess(residual / (1 + abs(get(eq.name))), 1e-8,
                                        f"scenario {scenario.no}, {result.index[t]}, eq{eq.no}")

    def test_cli_failure_does_not_write_partial_results(self):
        real_solve = M.Model.solve
        calls = 0

        def fail_third_call(model, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:  # baseline, scenario 1, then scenario 2 fails
                raise ValueError("injected scenario failure")
            return real_solve(model, *args, **kwargs)

        with patch.object(sys, "argv", ["simulate.py"]), patch.object(M.Model, "solve", fail_third_call), \
                patch.object(pd.DataFrame, "to_csv") as write, patch("builtins.print"):
            with self.assertRaisesRegex(RuntimeError, "injected scenario failure"):
                simulate.main()
            write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
