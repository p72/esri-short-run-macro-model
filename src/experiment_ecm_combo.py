"""ECM 固定仮説: CP を有効にしたうえで、改善候補の組み合わせを比較する."""
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
    live_sets = {"CPのみ": {"CP"}, "CP+YICV": {"CP", "YICV"}, "CP+PCPAT": {"CP", "PCPAT"},
                 "CP+YICV+PCPAT": {"CP", "YICV", "PCPAT"}, "CP+IHP": {"CP", "IHP"}}
    rows, gdp_tab = [], {}
    for label, live in live_sets.items():
        r = run(label, [n for n in allnames if n not in live])
        m = r.merge(pub, on=["scenario", "variable", "year", "quarter"], suffixes=("", "_paper"))
        m = m[m.quarter == 0]
        m["abs"] = (m["value"] - m["value_paper"]).abs()
        g = m[m.variable == "GDP"]
        rows.append({"有効なECM": label, "全変数MAE": m["abs"].mean(), "±0.1以内": (m["abs"] <= 0.1).mean(),
                     "GDP_MAE": g["abs"].mean(), "GDP_最大誤差": g["abs"].max(),
                     **{f"S{s}": v for s, v in m.groupby("scenario")["abs"].mean().items()}})
        gdp_tab[label] = g.set_index(["scenario", "year"])["value"]
        gdp_tab["論文"] = g.set_index(["scenario", "year"])["value_paper"]
        print(label, round(rows[-1]["全変数MAE"], 4), round(rows[-1]["GDP_MAE"], 4), flush=True)
    out = pd.DataFrame(rows).sort_values("全変数MAE")
    out.to_csv(ROOT / "output/experiment_ecm_combo.csv", index=False, encoding="utf-8-sig")
    print(out.round(3).to_string(index=False))
    print(pd.DataFrame(gdp_tab).round(2).to_string())


if __name__ == "__main__":
    main()
