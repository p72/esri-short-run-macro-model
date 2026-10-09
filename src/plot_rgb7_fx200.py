"""長期金利7%・ドル円200円シナリオ（experiment_rgb7_fx200.py の結果）の図.

6つのパネル: 実質GDP、実質消費、住宅投資、消費デフレーター、財政収支/名目GDP、政府の純財産所得/名目GDP（四半期、基準解からの差）

出力: output/rgb7_fx200.png（日本語フォント IPAPGothic が必要）
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"
STYLE = {"A 円安のみ（ドル円200円）": ("円安のみ（ドル円200円）", ORANGE, (0, (5, 2))),
         "B 長期金利のみ（7%）": ("長期金利のみ（7%）", BLUE, (0, (1, 1.3))),
         "C 長期金利7%・ドル円200円": ("長期金利7%・ドル円200円", INK, "-")}
PANELS = [("実質GDP（%）", "実質GDP（%）"), ("実質消費（%）", "実質消費（%）"), ("住宅投資（%）", "住宅投資（%）"),
          ("消費デフレーター（%）", "消費者物価（消費デフレーター、%）"), ("財政収支/名目GDP（%pt）", "財政収支/名目GDP（%pt）"),
          ("政府の純財産所得/名目GDP（%pt）", "政府の純財産所得（利払いを含む）/名目GDP（%pt）")]


def main() -> None:
    p = pd.read_csv(OUT / "experiment_rgb7_fx200_paths.csv")
    plt.rcParams.update({"font.family": "IPAPGothic", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.unicode_minus": False})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
    fig.patch.set_facecolor("white")
    q = p["四半期"].unique()
    x = np.arange(len(q))
    for ax, (col, title) in zip(axes.flat, PANELS):
        ends = []
        for key, (lab, c, ls) in STYLE.items():
            y = p.loc[p["シナリオ"] == key, col].to_numpy()
            ax.plot(x, y, color=c, lw=2.3, ls=ls, label=lab)
            ends.append((y[-1], c))
        span = np.diff(ax.get_ylim())[0]
        prev = None
        for yv, c in sorted(ends):  # 近い値のラベルは上下にずらす
            dy = 9 if prev is not None and abs(yv - prev) < 0.06 * span else 0
            ax.annotate(f"{yv:+.1f}", (x[-1], yv), xytext=(5, dy), textcoords="offset points", color=c, fontsize=9, va="center")
            prev = yv
        ax.axhline(0, color=INK2, lw=0.9)
        ax.set_xticks(x[::4])
        ax.set_xticklabels([f"{s[:4]}年" for s in q[::4]])
        ax.set_xlim(-0.4, len(q) + 0.6)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.01, 0.925), ncol=3, frameon=False, fontsize=10)
    fig.suptitle("長期金利7%・ドル円200円シナリオ（ESRI 短期マクロモデル）", x=0.01, ha="left", fontsize=14, color=INK,
                 fontweight="bold", y=0.985)
    fig.text(0.01, 0.945, "2022Q1 から恒久的に、長期金利を7%（実績 0.2〜1.0%）、ドル円を200円（実績 116〜156円）に固定。基準解（2022〜24年の実績）からの差。",
             fontsize=9.5, color=INK2)
    note = ("内閣府 短期日本経済マクロ計量モデル（2022年版、ESRI Research Note No.72）の Python 再現、2024年版データ。短期金利はテイラー・ルール（式114）。"
            "長期金利のみのケースは為替を基準解に固定。誤差修正項は既定（3本のみ有効）。\n"
            "長期金利の上昇で家計の財産所得（利子の受け取り）が増え、可処分所得と消費が増える（式104）。政府の利払いは式138 でゆっくり増える。"
            "ショックはモデルの推定範囲を大きく超えるので、結果は目安。")
    fig.text(0.01, 0.012, note, fontsize=8, color=INK2, linespacing=1.55, va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 0.9), h_pad=2.2, w_pad=2.0)
    fig.savefig(OUT / "rgb7_fx200.png", dpi=160, facecolor="white")
    print(OUT / "rgb7_fx200.png")


if __name__ == "__main__":
    main()
