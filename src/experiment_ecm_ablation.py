"""ECM 固定仮説の切り分け: 全ECM固定から1本ずつ「動く」状態に戻し、論文との誤差の変化を見る."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment_ecm import ECM, run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    pub = pd.read_csv(ROOT / "output/published_multipliers.csv")
    allnames = list(ECM)
    configs = {"全固定": allnames, **{f"{n}だけ有効": [x for x in allnames if x != n] for n in allnames}}
    rows = []
    for label, names in configs.items():
        r = run(label, names)
        m = r.merge(pub, on=["scenario", "variable", "year", "quarter"], suffixes=("", "_paper"))
        m = m[m.quarter == 0]
        m["abs"] = (m["value"] - m["value_paper"]).abs()
        gdp = m[m.variable == "GDP"]
        rows.append({"設定": label, "全変数MAE": m["abs"].mean(), "GDP_MAE": gdp["abs"].mean(),
                     "GDP_最大誤差": gdp["abs"].max(),
                     **{f"S{s}": v for s, v in m.groupby("scenario")["abs"].mean().items()}})
        print(f"{label:16s} 全変数MAE={rows[-1]['全変数MAE']:.4f} GDP_MAE={rows[-1]['GDP_MAE']:.4f}", flush=True)
    out = pd.DataFrame(rows).sort_values("全変数MAE")
    out.to_csv(ROOT / "output/experiment_ecm_ablation.csv", index=False, encoding="utf-8-sig")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
