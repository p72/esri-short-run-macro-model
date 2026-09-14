"""Issue #1: 法人税減税シナリオ(5)で資本コスト UCC が論文と逆符号になる原因の分解.

式121〜125（資本コスト）は入力（TT, PERR, RGB, RCD, PIFPAT, PIFPATSUM, PGDPAT, RRFP）の
非線形関数なので、シナリオ解の各入力を1つずつ標準解に差し替えて寄与を測る。
あわせて、仮定値（REQU, ROR, SLRATIO, TINCR）と水準（RRFP, PERR）の感応度を出す。

出力: output/experiment_ucc.csv（四半期別の寄与）, output/experiment_ucc_sensitivity.csv（年乗数の感応度）
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ecm as E  # noqa: E402
import model as M  # noqa: E402
import simulate as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ["TT", "PERR", "RGB", "RCD", "PIFPAT", "PIFPATSUM", "PGDPAT", "RRFP"]


def ucc(r: pd.Series) -> tuple[float, float, float]:
    """式121〜125を1期分の入力から評価（RRFP4 = RRFP の当期〜3期前の和）."""
    pvdp = -1 / 18 * math.log(0.1) * (1 - 0.1 * math.exp(-r.RGB / 100 * 18)) / (r.RGB / 100 - math.log(0.1) / 18)
    f = r.PIFPAT / (1 - r.TT) * (1 - r.TT * pvdp - r.TINCR)
    db = f * ((1 - r.TT) * (r.RCD + r.SLRATIO * r.RGB) / (1 + r.SLRATIO) + r.RRFP * 400 - 0.925751 * r.PIFPATSUM)
    de = f * ((1 - r.PERR * r.ROR / 100) / r.PERR * 100 + r.RRFP4 * 100 - 0.925751 * r.PIFPATSUM)
    pf = (1 - r.REQU) * db + r.REQU * de
    return pf / 100 / r.PGDPAT, db, de


def rows_at(df: pd.DataFrame, t: str) -> pd.Series:
    r = df.loc[t].copy()
    r["RRFP4"] = sum(df["RRFP"].shift(k).loc[t] for k in range(4))
    return r


def solve_scenario5() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(ROOT / "data" / "processed" / "model_data.csv", index_col="period")
    data.index = pd.PeriodIndex(data.index, freq="Q")
    model = M.Model()
    af = model.add_factors(data, S.SOLVE_START, S.END)
    base = model.solve(data, S.SOLVE_START, S.END, af)
    over, cols = E.frozen_overrides(model, base, S.solve_mask(base.index))
    sc = next(s for s in S.build_scenarios(base, af) if s.no == 5)
    d = base.copy()
    for k, v in {**sc.data, **cols}.items():
        d[k] = v
    shock = model.solve(d, S.SOLVE_START, S.END, af, fixed=sc.fixed,
                        overrides={**over, **sc.overrides}, shocks=sc.shocks)
    return base, shock


def main() -> None:
    base, shock = solve_scenario5()
    pub = pd.read_csv(ROOT / "output" / "published_multipliers.csv")
    pub_ucc = pub[(pub.scenario == 5) & (pub.variable == "UCC") & (pub.quarter > 0)]
    pub_ucc = {f"{y}Q{q}": v for y, q, v in zip(pub_ucc.year, pub_ucc.quarter, pub_ucc.value)}

    # ---- 四半期別の寄与分解
    rows = []
    for p in pd.period_range(S.START, S.END, freq="Q"):
        t = str(p)
        b, s = rows_at(base, t), rows_at(shock, t)
        u0, db0, de0 = ucc(b)
        u1, db1, de1 = ucc(s)
        row = {"period": t, "paper": pub_ucc.get(t, np.nan), "repro": (u1 / u0 - 1) * 100,
               "model_check": (shock.loc[t, "UCC"] / base.loc[t, "UCC"] - 1) * 100,
               "UCCDB": (db1 / db0 - 1) * 100, "UCCDE": (de1 / de0 - 1) * 100}
        for k in INPUTS:
            c = b.copy()
            c[k] = s[k]
            if k == "RRFP":
                c["RRFP4"] = s["RRFP4"]
            row[f"contrib_{k}"] = (ucc(c)[0] / u0 - 1) * 100
        row["dTT"], row["dETT"] = s.TT - b.TT, s.ETT - b.ETT
        row["PERR_base"], row["PERR_shock"] = b.PERR, s.PERR
        rows.append(row)
    contrib = pd.DataFrame(rows).set_index("period")
    contrib.round(6).to_csv(ROOT / "output" / "experiment_ucc.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print("UCC 乖離率(%)の寄与分解（シナリオ5、入力を1つずつシナリオ解に差し替え）")
    print(contrib[["paper", "repro", "UCCDB", "UCCDE"] + [f"contrib_{k}" for k in INPUTS]].round(2).to_string())

    # ---- 仮定値・水準の感応度（UCC の年乗数）
    def annual(modify) -> list[float]:
        out = []
        for y in (2018, 2019, 2020):
            vals = []
            for q in range(1, 5):
                b, s = rows_at(base, f"{y}Q{q}"), rows_at(shock, f"{y}Q{q}")
                modify(b, s)
                vals.append((ucc(s)[0] / ucc(b)[0] - 1) * 100)
            out.append(float(np.mean(vals)))
        return out

    def setter(name, val):
        def f(b, s):
            b[name] = s[name] = val
        return f

    def scaler(names, m):
        def f(b, s):
            for x in (b, s):
                for n in names:
                    x[n] *= m
        return f

    def tt_fixed(b, s):
        s["TT"] = b["TT"]

    cases = [("論文", None)]
    cases += [(f"REQU={v:.2f}", setter("REQU", v)) for v in (0.0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)]
    cases += [(f"ROR={v:.1f}", setter("ROR", v)) for v in (0.0, 2.0, 4.0)]
    cases += [(f"SLRATIO={v:.1f}", setter("SLRATIO", v)) for v in (0.0, 0.5, 2.0)]
    cases += [(f"TINCR={v:.2f}", setter("TINCR", v)) for v in (0.0, 0.05)]
    cases += [(f"RRFP×{m}", scaler(["RRFP", "RRFP4"], m)) for m in (0.5, 1.0, 2.0)]
    cases += [(f"PERR水準×{m}", scaler(["PERR"], m)) for m in (0.5, 1.0, 2.0)]
    cases += [("TTを動かさない(ETTのみ)", tt_fixed)]
    pub_annual = pub[(pub.scenario == 5) & (pub.variable == "UCC") & (pub.quarter == 0)].sort_values("year")["value"].tolist()
    sens = []
    for name, fn in cases:
        vals = pub_annual if fn is None else annual(fn)
        sens.append({"case": name, "2018": vals[0], "2019": vals[1], "2020": vals[2]})
    sens = pd.DataFrame(sens).set_index("case")
    sens.round(4).to_csv(ROOT / "output" / "experiment_ucc_sensitivity.csv", encoding="utf-8-sig")
    print("\nUCC 年乗数(%) の感応度（既定: REQU 0.40, ROR 2.0, SLRATIO 0.5, TINCR 0）")
    print(sens.round(2).to_string())


if __name__ == "__main__":
    main()
