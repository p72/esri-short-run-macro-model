"""仮説検証: ESRI の乗数計算では誤差修正項（ECM）がベースライン値で固定されているか.

各推定式の右辺を「元の右辺 − ECM(ショック解) + ECM(標準解)」に差し替えて11シナリオを解き、
(a) ECM が動く実装、(b) 全ECM固定、(c) 貿易・価格ECMのみ固定 の3通りを論文と比べる。
誤差修正項の定義と固定の仕組みは src/ecm.py にある。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ecm as ECMOD  # noqa: E402
import model as M  # noqa: E402
import simulate as SIM  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ECM = ECMOD.ECM_TERMS
TRADE_PRICE = ["XGS", "FUEL", "PXGS", "PFUELAT", "PNFMGSAT"]


def run(variant: str, frozen: list[str]) -> pd.DataFrame:
    """frozen に挙げた誤差修正項を固定して11シナリオの乗数を計算する."""
    data = pd.read_csv(ROOT / "data/processed/model_data.csv", index_col="period")
    data.index = pd.PeriodIndex(data.index, freq="Q")
    model = M.Model()
    af = model.add_factors(data, SIM.SOLVE_START, SIM.END)
    base = model.solve(data, SIM.SOLVE_START, SIM.END, af)
    live = frozenset(n for n in ECM if n not in frozen)
    over, cols = ECMOD.frozen_overrides(model, base, SIM.solve_mask(base.index), live=live)
    out = []
    for s in SIM.build_scenarios(base, af):
        d = base.copy()
        for k, v in {**s.data, **cols}.items():
            d[k] = v
        sh = model.solve(d, SIM.SOLVE_START, SIM.END, af, fixed=s.fixed, overrides={**over, **s.overrides}, shocks=s.shocks)
        out.append(SIM.multipliers(sh, base, s.no))
    return pd.concat(out).assign(variant=variant)


def main() -> None:
    pub = pd.read_csv(ROOT / "output/published_multipliers.csv")
    res = pd.concat([run("a_ECM有効", []), run("b_全ECM固定", list(ECM)), run("c_貿易価格ECMのみ固定", TRADE_PRICE)])
    res.to_csv(ROOT / "output/experiment_ecm.csv", index=False, encoding="utf-8-sig")
    m = res.merge(pub, on=["scenario", "variable", "year", "quarter"], suffixes=("", "_paper"))
    m["abs"] = (m["value"] - m["value_paper"]).abs()
    ann = m[m.quarter == 0]
    print("== 年乗数の平均絶対誤差（全変数）: シナリオ × 設定")
    print(ann.pivot_table(index="scenario", columns="variant", values="abs", aggfunc="mean").round(3).to_string())
    print("\n== 実質GDP 年乗数")
    g = ann[ann.variable == "GDP"].pivot_table(index=["scenario", "year"], columns="variant", values="value")
    g["論文"] = ann[ann.variable == "GDP"].groupby(["scenario", "year"])["value_paper"].first()
    print(g.round(2).to_string())
    q = m[(m.quarter > 0) & (m.scenario.isin([4, 9])) & (m.variable.isin(["XGS", "PXGS", "IHP"]))]
    q = q.assign(p=q.year.astype(str) + "Q" + q.quarter.astype(str))
    print("\n== 四半期経路（(4)所得税の IHP、(9)為替の XGS/PXGS）")
    t = q.pivot_table(index=["scenario", "variable", "p"], columns="variant", values="value")
    t["論文"] = q.groupby(["scenario", "variable", "p"])["value_paper"].first()
    keep = {(4, "IHP"), (9, "XGS"), (9, "PXGS")}
    print(t[[(i[0], i[1]) in keep for i in t.index]].round(2).to_string())


if __name__ == "__main__":
    main()
