"""誤差修正項をどれだけ固定すべきかの探索（docs/fidelity.md の根拠）.

(1) 14本すべて有効（印刷どおり）から1本ずつ固定したときの一致度
(2) 貪欲法: 最も改善する項から順に固定し、改善が止まるまで続ける（既定 ecm.DEFAULT_LIVE の導出）
(3) 論文で長期均衡を入れ子括弧「LOG(x) − ( … )」で書いている式のうち ecm.py に含めていない
    3本（NFMGS, PCGAT, PIGAT）を固定しても結果が変わらないこと、YWV を有効にした場合の差

出力: output/experiment_ecm_freeze_one.csv, experiment_ecm_greedy.csv, experiment_ecm_nested.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ecm as ECMOD  # noqa: E402
import experiment_fidelity as F  # noqa: E402
from model import ln  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# 入れ子形の長期均衡項をもつが係数が小さく ecm.py の対象にしていない3本
EXTRA_TERMS = {
    "NFMGS": lambda v: -0.000380 * (ln(v("NFMGS", 1))
                                    - (-21.50892 + 0.448526 * ln(v("IFP", 1) + v("IHP", 1) + v("IG", 1))
                                       + 2.054694 * ln(v("CP", 1) + v("CG", 1))
                                       + 0.196224 * ln(v("PNFMGSAT", 1) / v("PGDPAT", 1)) + 0.006261 * v("TIME", 1))),
    "PCGAT": lambda v: -0.012528 * (ln(v("PCGAT", 1) / v("PCPAT", 1))
                                    - (0.246639 * ln(v("WIPH", 1) / v("PCPAT", 1)) - 0.911111)),
    "PIGAT": lambda v: -0.007312 * (ln(v("PIGAT", 1) / v("PIFPAT", 1))
                                    - (1.088554 * ln(v("PIHPAT", 1) / v("PIFPAT", 1))
                                       - 0.438980 * ln(v("PCGAT", 1) / v("PIFPAT", 1)) + 0.007716)),
}


def main() -> None:
    pub = pd.read_csv(ROOT / "output/published_multipliers.csv")
    pub = pub[pub.quarter == 0]
    eqs = F.equations()

    def score(live: set[str]) -> tuple[float, float]:
        res = F.run(eqs, frozenset(live))
        m = res[res.quarter == 0].merge(pub, on=["scenario", "variable", "year"], suffixes=("", "_paper"))
        m["abs"] = (m["value"] - m["value_paper"]).abs()
        return float(m["abs"].mean()), float((m["abs"] <= 0.1).mean())

    # (1) 全有効から1本固定
    allt = set(ECMOD.ECM_TERMS)
    mae0, w0 = score(allt)
    print(f"全有効: MAE {mae0:.4f} ±0.1以内 {w0:.1%}", flush=True)
    one = []
    for x in sorted(allt):
        mae, w = score(allt - {x})
        one.append({"frozen_one": x, "MAE": mae, "within_0.1": w})
        print(f"  {x:9s} を固定 → MAE {mae:.4f} ±0.1以内 {w:.1%}", flush=True)
    pd.DataFrame(one).sort_values("MAE").to_csv(ROOT / "output/experiment_ecm_freeze_one.csv", index=False, encoding="utf-8-sig")

    # (2) 貪欲法
    live, mae, w = set(allt), mae0, w0
    steps = [{"step": 0, "frozen": "", "MAE": mae, "within_0.1": w}]
    while True:
        best = min(((x, *score(live - {x})) for x in sorted(live)), key=lambda r: r[1])
        if best[1] >= mae - 1e-6:
            break
        live.discard(best[0])
        mae, w = best[1], best[2]
        steps.append({"step": len(steps), "frozen": best[0], "MAE": mae, "within_0.1": w})
        print(f"step {len(steps) - 1}: {best[0]} を固定 → MAE {mae:.4f} ±0.1以内 {w:.1%}", flush=True)
    pd.DataFrame(steps).to_csv(ROOT / "output/experiment_ecm_greedy.csv", index=False, encoding="utf-8-sig")
    print("貪欲法の結果 有効:", sorted(live), "（ecm.DEFAULT_LIVE:", sorted(ECMOD.DEFAULT_LIVE), "）")

    # (3) 入れ子形の追加3本と YWV
    ECMOD.ECM_TERMS.update(EXTRA_TERMS)
    D = set(ECMOD.DEFAULT_LIVE)
    ex = set(EXTRA_TERMS)
    cases = {"既定（追加3本は式に埋め込み＝有効）": D | ex, "NFMGS 固定": D | {"PCGAT", "PIGAT"},
             "PCGAT 固定": D | {"NFMGS", "PIGAT"}, "PIGAT 固定": D | {"NFMGS", "PCGAT"}, "3本とも固定": D,
             "既定 + YWV 有効": D | ex | {"YWV"}, "3本固定 + YWV 有効": D | {"YWV"}}
    rows = []
    for k, lv in cases.items():
        mae, w = score(lv)
        rows.append({"case": k, "MAE": mae, "within_0.1": w})
        print(f"{k:32s} MAE {mae:.5f} ±0.1以内 {w:.2%}", flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "output/experiment_ecm_nested.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
