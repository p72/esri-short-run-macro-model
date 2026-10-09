"""名目GDP1%規模の消費税減税と税収（experiment_ctax_revenue.py の結果）の図.

左: 法人所得税の増減（兆円、四半期・年率）の経路。消費税減税と同額の給付金、2021年版と2024年版
右: 税目別の増減（兆円、3年平均）の積み上げと、財源の戻り（税収の自然増 ÷ 事前の減税額）

表: note 記事（docs/note_ctax_revenue.md）用に、3年平均の主な数値を表の画像にする（数値は CSV から読む）

出力: output/ctax_revenue.png, output/ctax_revenue_table.png（日本語フォント IPAPGothic が必要）
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
BLUE, ORANGE, AQUA, VIOLET, INK, INK2, GRID, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a63d2", "#0b0b0b", "#52514e", "#e6e5e1", "#b9b8b3"
V21, V24 = "2021年版（2018〜20年）", "2024年版（2022〜24年）"
CASES = [("消費税減税", BLUE), ("給付金（所得減税）", ORANGE)]


def table(a: pd.DataFrame) -> None:
    avg = a.groupby(["版", "政策"], sort=False).mean(numeric_only=True)
    keys = [(v, c) for v in (V21, V24) for c, _ in CASES]
    rows = [("法人所得税（兆円）", "法人所得税（兆円）", "{:+.2f}"), ("企業所得（%）", "企業所得 YCV（基準解比 %）", "{:+.1f}"),
            ("消費税（兆円）", "消費税（兆円）", "{:+.1f}"), ("個人所得税（兆円）", "個人所得税（兆円）", "{:+.1f}"),
            ("税収の合計（兆円）", "税収の合計（兆円）", "{:+.1f}"), ("財源の戻り（%）", "財源の戻り（%）", "{:.0f}"),
            ("実質GDP（%）", "実質GDP（%）", "{:+.2f}")]
    head = ["3年平均（基準解からの差）"] + [f"{c.replace('（所得減税）', '')}\n{v.split('（')[0]}" for v, c in keys]
    widths = [3.6, 1.9, 1.9, 1.9, 1.9]
    W, rh, hh = sum(widths), 0.6, 0.84
    h = hh + rh * len(rows) + 1.5
    fig = plt.figure(figsize=(W, h))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(h, 0)
    ax.axis("off")
    ax.text(0.25, 0.5, "名目GDP1%規模の消費税減税と同額の給付金：税収への影響", fontsize=15, color=INK, va="center")
    y = 1.0
    ax.add_patch(plt.Rectangle((0, y), W, hh, color="#f3f2ee", lw=0))
    x = 0
    for txt, w in zip(head, widths):
        ax.text(x + (0.2 if x == 0 else w / 2), y + hh / 2, txt, fontsize=11, color=INK2,
                ha="left" if x == 0 else "center", va="center", linespacing=1.3)
        x += w
    y += hh
    for lab, col, fmt in rows:
        ax.text(0.2, y + rh / 2, lab, fontsize=12, color=INK, va="center")
        x = widths[0]
        for k, w in zip(keys, widths[1:]):
            ax.text(x + w / 2, y + rh / 2, fmt.format(avg.loc[k, col]), fontsize=12, color=INK, ha="center", va="center")
            x += w
        ax.plot([0, W], [y + rh, y + rh], color=GRID, lw=1)
        y += rh
    ax.text(0.25, y + 0.35, "財源の戻り＝（税収の合計の変化＋事前の減税額）÷事前の減税額。給付金は個人所得税の減税として与える。",
            fontsize=9.5, color=INK2, va="center")
    fig.savefig(OUT / "ctax_revenue_table.png", dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    a = pd.read_csv(OUT / "experiment_ctax_revenue.csv")
    p = pd.read_csv(OUT / "experiment_ctax_revenue_paths.csv")
    plt.rcParams.update({"font.family": "IPAPGothic", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.unicode_minus": False})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.6), gridspec_kw={"width_ratios": [1, 1.15]})
    fig.patch.set_facecolor("white")

    # 左: 法人所得税の経路（実施からの四半期）
    for case, c in CASES:
        for v, ls, lab in ((V21, "-", "2021年版"), (V24, (0, (4, 2)), "2024年版")):
            y = p.loc[(p["版"] == v) & (p["政策"] == case), "法人所得税（兆円）"].to_numpy()
            ax1.plot(np.arange(1, len(y) + 1), y, color=c, lw=2.3, ls=ls, label=f"{case}・{lab}")
    ax1.axhline(0, color=INK2, lw=0.9)
    ax1.set_xticks(range(1, 13))
    ax1.set_xlabel("実施からの四半期")
    ax1.set_title("法人所得税の増減（兆円、年率）", loc="left", fontsize=11.5, color=INK)
    ax1.legend(loc="lower right", frameon=False, fontsize=9.5)
    ax1.annotate("企業所得が増えてから\n約1年遅れて税収が増える", (4.6, 0.7), xytext=(7.0, 0.6), fontsize=10, color=INK, va="center",
                 arrowprops=dict(arrowstyle="->", color=INK2, lw=1))

    # 右: 税目別の増減（3年平均）
    parts = [("消費税（兆円）", "消費税", BLUE), ("個人所得税（兆円）", "個人所得税", ORANGE),
             ("法人所得税（兆円）", "法人所得税", AQUA), ("その他の間接税（兆円）", "その他の間接税", GRAY)]
    avg = a.groupby(["版", "政策"], sort=False).mean(numeric_only=True)
    keys = [(v, c) for v in (V21, V24) for c, _ in CASES]
    x = np.arange(len(keys))
    pos, neg = np.zeros(len(keys)), np.zeros(len(keys))
    for col, lab, c in parts:
        vals = np.array([avg.loc[k, col] for k in keys])
        bottom = np.where(vals >= 0, pos, neg)
        ax2.bar(x, vals, 0.6, bottom=bottom, color=c, label=lab, edgecolor="white", lw=0.8)
        pos += np.where(vals >= 0, vals, 0)
        neg += np.where(vals < 0, vals, 0)
    tot = np.array([avg.loc[k, "税収の合計（兆円）"] for k in keys])
    ret = np.array([avg.loc[k, "財源の戻り（%）"] for k in keys])
    corp = np.array([avg.loc[k, "法人所得税（兆円）"] for k in keys])
    ax2.scatter(x, tot, color=INK, s=34, zorder=4, label="税収の合計")
    for xi, t, r, cp, nb in zip(x, tot, ret, corp, neg):
        ax2.annotate(f"合計 {t:+.1f}兆円\n財源の戻り {r:.0f}%\n法人税 {cp:+.2f}兆円", (xi, nb), xytext=(0, -6),
                     textcoords="offset points", ha="center", va="top", fontsize=9, color=INK)
    ax2.axhline(0, color=INK2, lw=0.9)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{c}\n{v.split('（')[0]}" for v, c in keys], fontsize=9.5)
    ax2.set_ylim(neg.min() * 1.75, max(pos.max(), 0.5) * 1.6)
    ax2.set_title("税目別の増減（兆円、3年平均）", loc="left", fontsize=11.5, color=INK)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False, fontsize=9, ncol=5)

    for ax in (ax1, ax2):
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    x21 = a.loc[(a["版"] == V21) & (a["政策"] == "消費税減税"), "消費税率の引下げ幅（%pt）"].iloc[0]
    x24 = a.loc[(a["版"] == V24) & (a["政策"] == "消費税減税"), "消費税率の引下げ幅（%pt）"].iloc[0]
    fig.suptitle("名目GDP1%規模の消費税減税で、法人税収はどうなるか", x=0.01, ha="left", fontsize=14, color=INK,
                 fontweight="bold", y=0.985)
    fig.text(0.01, 0.935, f"消費税率を {x21:.2f}%pt（2021年版）／{x24:.2f}%pt（2024年版）恒久的に引下げ（事前の税収減＝名目GDPの1%）。"
             "比較は同額の給付金（個人所得税の減税）。基準解からの差。", fontsize=9.5, color=INK2)
    note = ("内閣府 短期日本経済マクロ計量モデル（2022年版、ESRI Research Note No.72）の Python 再現。法人所得税は SNA の「所得・富等に課される経常税」の企業分"
            "（国税の法人税に、法人住民税・法人事業税の所得割などを含む）。\n法人所得税＝実効税率×前期から4期分の企業所得の平均（式131）。"
            "財源の戻り＝（税収の合計の変化＋事前の減税額）÷事前の減税額。誤差修正項は既定（3本のみ有効）。")
    fig.text(0.01, 0.012, note, fontsize=8, color=INK2, linespacing=1.55, va="bottom")
    fig.tight_layout(rect=(0, 0.07, 1, 0.92), w_pad=2.5)
    fig.savefig(OUT / "ctax_revenue.png", dpi=160, facecolor="white")
    table(a)
    print(OUT / "ctax_revenue.png", OUT / "ctax_revenue_table.png", sep="\n")


if __name__ == "__main__":
    main()
