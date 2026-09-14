"""実質GDP乗数の四半期経路（11シナリオ）: 論文 vs 再現 の図を描く.

入力: output/published_multipliers.csv, output/multipliers_reproduced.csv
出力: output/gdp_multiplier_paths.png（白背景・英語表記）
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "gdp_multiplier_paths.png"

rep = pd.read_csv(ROOT / "output/multipliers_reproduced.csv")
pub = pd.read_csv(ROOT / "output/published_multipliers.csv")

TITLES = {
    1: "(1) Public investment +1% of real GDP",
    2: "(2) Same, short rate fixed",
    3: "(3) Nominal public investment +1% of GDP",
    4: "(4) Personal income tax cut, 1% of GDP",
    5: "(5) Corporate income tax cut, 1% of GDP",
    6: "(6) Consumption tax rate +1pt",
    7: "(7) Short-term interest rate +1pt",
    8: "(8) Money supply -1%",
    9: "(9) Yen depreciation 10%",
    10: "(10) Crude oil price +20%",
    11: "(11) World demand +1%",
}
BLUE, ORANGE = "#2a78d6", "#eb6834"      # validated categorical slots 1, 2
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

def path(df, sc):
    d = df[(df.variable == "GDP") & (df.scenario == sc) & (df.quarter > 0)].sort_values(["year", "quarter"])
    return [f"{y}Q{q}" for y, q in zip(d.year, d.quarter)], d.value.to_numpy()

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
fig, axes = plt.subplots(3, 4, figsize=(13, 8.2), sharex=True, sharey=False)
fig.patch.set_facecolor("white")

for ax, sc in zip(axes.flat, range(1, 12)):
    labels, yp = path(pub, sc)
    _, yr = path(rep, sc)
    x = range(len(labels))
    ax.axhline(0, color=GRID, lw=1, zorder=1)
    ax.plot(x, yp, color=BLUE, lw=2, marker="o", ms=4.5, zorder=3, label="Paper (ESRI RN No.72)")
    ax.plot(x, yr, color=ORANGE, lw=2, ls=(0, (4, 2)), zorder=4, label="Reproduction")
    ax.set_title(TITLES[sc], loc="left", fontsize=9.5, color=INK, pad=6)
    # direct labels: last value of each series
    ax.annotate(f"{yp[-1]:+.2f}", (x[-1], yp[-1]), xytext=(5, 4), textcoords="offset points",
                color=INK2, fontsize=8, va="bottom")
    ax.annotate(f"{yr[-1]:+.2f}", (x[-1], yr[-1]), xytext=(5, -4), textcoords="offset points",
                color=INK2, fontsize=8, va="top")
    ax.set_xticks([0, 4, 8, 11]); ax.set_xticklabels(["2018Q1", "2019Q1", "2020Q1", "20Q4"])
    ax.set_xlim(-0.3, 12.6)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    lo, hi = min(yp.min(), yr.min()), max(yp.max(), yr.max())
    pad = 0.12 * (hi - lo + 0.2)
    ax.set_ylim(min(lo - pad, -0.05), max(hi + pad, 0.05))

# 12th panel: legend + notes
ax = axes.flat[11]
ax.axis("off")
h, l = axes.flat[0].get_legend_handles_labels()
ax.legend(h, l, loc="upper left", frameon=False, fontsize=9.5, bbox_to_anchor=(0, 0.95))
ax.text(0, 0.55, "Deviation of real GDP from baseline, %\n(quarterly, shock starts 2018Q1)\n\n"
        "Reproduction: 152 published equations,\nbaseline = actual data, add factors fixed;\n"
        "96.1% of table cells within 0.1 of the paper.",
        transform=ax.transAxes, va="top", fontsize=8.5, color=INK2, linespacing=1.5)

fig.suptitle("ESRI Short-Run Macroeconometric Model (2022): real GDP multiplier paths",
             x=0.01, ha="left", fontsize=13, color=INK, fontweight="bold", y=0.985)
fig.text(0.01, 0.945, "Paper values vs. Python reproduction, 11 scenarios, 2018Q1-2020Q4",
         fontsize=10, color=INK2)
fig.text(0.01, 0.01, "Source: ESRI Research Note No.72 (Dec 2022), Appendix I multiplier tables; "
         "reproduction from output/multipliers_reproduced.csv", fontsize=7.5, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 0.93), h_pad=1.8, w_pad=1.2)
fig.savefig(OUT, dpi=160, facecolor="white")
print(OUT)
