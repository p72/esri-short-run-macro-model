"""名目GDP水準目標（experiment_ngdplt.py の結果）での実質GDP の推移の図.

左: 実質GDP の水準（兆円、2020年基準、季節調整済み年率）。実績と、NGDPLT 3%・4%・5% をモデルに組み込んで解いた値
右: 実質GDP の前年比（%）
モデルの解の水準は、実績の実質GDP に基準解からの乖離（%）を掛けて作る（2019年以前は実績のまま）。

出力: output/ngdplt_gdp.png（日本語フォント IPAPGothic または IPAGothic が必要）
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
FONT = next((f for f in ("IPAPGothic", "IPAGothic") if any(f == x.name for x in font_manager.fontManager.ttflist)), "sans-serif")
BLUE, ORANGE, AQUA, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e6e5e1"
CASES = (("3", AQUA), ("4", BLUE), ("5", ORANGE))


def series() -> list[tuple[str, pd.Series, str]]:
    d = pd.read_csv(OUT / "experiment_ngdplt.csv", index_col=0)
    gdp = pd.read_csv(ROOT / "data" / "processed" / "model_data_v2024.csv", index_col=0)["GDP"].loc["2019Q1":"2024Q4"] / 1000
    q = list(d.index)
    out = [("実績", gdp, INK)]
    for g, c in CASES:
        lv = gdp.copy()
        lv.loc[q] = gdp.loc[q].values * (1 + d[f"モデル NGDPLT{g}% 実質GDP（基準解比 %）"].values / 100)
        out.append((f"NGDPLT {g}%", lv, c))
    return out


def place(ax, x_end, items, gap, fmt):
    items = sorted(items)
    prev = None
    for v, c in items:
        yy = v if prev is None else max(v, prev + gap)
        prev = yy
        ax.annotate(fmt.format(v), (x_end, v), xytext=(x_end + 0.4, yy), textcoords="data", color=c, fontsize=9.5, va="center")


def main() -> None:
    ss = series()
    q = [p for p in ss[0][1].index if p >= "2020Q1"]
    x = np.arange(len(q))
    plt.rcParams.update({"font.family": FONT, "font.size": 10.5, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.unicode_minus": False})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8))
    fig.patch.set_facecolor("white")
    ends1, ends2 = [], []
    for lab, lv, c in ss:
        y1 = lv.loc[q]
        y2 = ((lv / lv.shift(4) - 1) * 100).loc[q]
        w = 2.6 if lab == "実績" else 2.3
        ax1.plot(x, y1.values, color=c, lw=w, label=lab)
        ax2.plot(x, y2.values, color=c, lw=w, label=lab)
        ends1.append((y1.iloc[-1], c))
        ends2.append((y2.iloc[-1], c))
    place(ax1, x[-1], ends1, 0.035 * np.diff(ax1.get_ylim())[0], "{:.0f}")
    place(ax2, x[-1], ends2, 0.045 * np.diff(ax2.get_ylim())[0], "{:+.1f}%")
    ax2.axhline(0, color=INK2, lw=0.9)
    ax1.set_title("実質GDP（兆円、2020年基準、季節調整済み年率）", loc="left", fontsize=11.5, color=INK)
    ax2.set_title("実質GDP の前年比（%）", loc="left", fontsize=11.5, color=INK)
    ax1.legend(loc="lower right", frameon=False)
    for ax in (ax1, ax2):
        ax.set_xticks(x[::4])
        ax.set_xticklabels([f"{s[:4]}年" for s in q[::4]])
        ax.set_xlim(-0.4, len(q) + 1.4)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("名目GDP水準目標なら、実質GDP はどうなったか", x=0.01, ha="left", fontsize=14, color=INK, fontweight="bold", y=0.98)
    fig.text(0.01, 0.915, "NGDPLT はルール（i ＝ 0.71×前期の i ＋ 0.29×（g ＋ 名目GDPギャップ%））をモデルに組み込んで解いた値。金利の下限なし。",
             fontsize=9.5, color=INK2)
    fig.text(0.01, 0.015, "内閣府 短期日本経済マクロ計量モデル（2022年版、ESRI Research Note No.72）の Python 再現、2024年版データ（国民経済計算 2024年度年次推計など）。",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.9), w_pad=2.5)
    fig.savefig(OUT / "ngdplt_gdp.png", dpi=160, facecolor="white")
    for lab, lv, _ in ss:
        y = lv.loc["2020Q1":]
        g = (lv / lv.shift(4) - 1).loc["2020Q1":] * 100
        print(lab, "水準", y.groupby(y.index.str[:4]).mean().round(1).tolist(), "前年比", g.groupby(g.index.str[:4]).mean().round(2).tolist())


if __name__ == "__main__":
    main()
