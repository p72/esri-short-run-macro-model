"""税収中立の税の組み替え（experiment_tax_swap.py の結果）の図.

左上: 実質GDP の経路（A・B と参考の単独ケース）
右上: 消費者物価（消費デフレーター）の経路
左下: 3年平均の主な指標を A と B で比べる横棒
右下: どちらが良いかの判定（指標ごと）
2024年版（2022〜24年）を主に描く。2021年版（2018〜20年）はほぼ同じ値になる。

出力: output/tax_swap.png（日本語フォント IPAPGothic または IPAGothic が必要）
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
BLUE, ORANGE, INK, INK2, GRID, GRAY = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1", "#b9b8b3"
V = "2024年版（2022〜24年）"
A, B = "A 消費税増税＋所得税減税", "B 所得税増税＋消費税減税"
STYLE = {A: ("A 消費税↑・所得税↓", ORANGE, "-", 2.6), B: ("B 所得税↑・消費税↓", BLUE, "-", 2.6),
         "参考: 消費税減税のみ": ("参考: 消費税減税のみ", BLUE, (0, (1, 1.3)), 1.6),
         "参考: 所得税減税のみ": ("参考: 所得税減税のみ", ORANGE, (0, (1, 1.3)), 1.6)}
BARS = [("実質GDP（%）", "実質GDP（%）", True), ("実質消費（%）", "実質消費（%）", True),
        ("住宅投資（%）", "住宅投資（%）", True), ("設備投資（%）", "設備投資（%）", True),
        ("消費者物価（消費デフレーター、%）", "消費者物価（%）", False), ("実質賃金（%）", "実質賃金（%）", True),
        ("実質可処分所得（%）", "実質可処分所得（%）", True), ("財政収支/名目GDP（%pt）", "財政収支/名目GDP（%pt）", True)]


def main() -> None:
    q = pd.read_csv(OUT / "experiment_tax_swap.csv")
    a = pd.read_csv(OUT / "experiment_tax_swap_annual.csv")
    q = q[q["版"] == V]
    avg = a[a["版"] == V].drop(columns=["年"]).groupby("政策").mean(numeric_only=True)
    a21 = a[a["版"] != V].drop(columns=["年"]).groupby("政策").mean(numeric_only=True)
    plt.rcParams.update({"font.family": FONT, "font.size": 10.5, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.unicode_minus": False})
    fig = plt.figure(figsize=(15, 10.5))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], width_ratios=[1, 1])

    quarters = q["四半期"].unique()
    x = np.arange(len(quarters))
    impl = list(quarters).index("2022Q1")
    for k, (col, title) in enumerate([("実質GDP（%）", "実質GDP（基準解からの乖離、%）"),
                                      ("消費者物価（消費デフレーター、%）", "消費者物価（消費デフレーター、基準解からの乖離、%）")]):
        ax = fig.add_subplot(gs[0, k])
        ax.axvspan(-0.5, impl - 0.5, color="#f6f5f2", zorder=0)
        for case, (lab, c, ls, lw) in STYLE.items():
            if col.startswith("消費者物価") and case.startswith("参考"):
                continue
            y = q.loc[q["政策"] == case, col].to_numpy()
            ax.plot(x, y, color=c, ls=ls, lw=lw, label=lab)
            ax.annotate(f"{y[-1]:+.2f}", (x[-1], y[-1]), xytext=(5, 0), textcoords="offset points", color=c, fontsize=9, va="center")
        ax.axhline(0, color=INK2, lw=0.9)
        ax.set_xticks(x[::4])
        ax.set_xticklabels([s[:4] + "年" for s in quarters[::4]])
        ax.set_xlim(-0.5, len(x) + 1)
        ax.set_title(title, loc="left", fontsize=11.5, color=INK)
        ax.text(impl - 0.4, ax.get_ylim()[0] + 0.05 * np.diff(ax.get_ylim())[0], "実施前（駆け込み・買い控え）", fontsize=8.5, color=INK2, ha="right")
        if k == 0:
            ax.legend(loc="upper right", frameon=False, fontsize=9)

    # 3年平均の比較
    ax = fig.add_subplot(gs[1, 0])
    y = np.arange(len(BARS))[::-1]
    va = np.array([avg.loc[A, c] for c, _, _ in BARS])
    vb = np.array([avg.loc[B, c] for c, _, _ in BARS])
    ax.barh(y + 0.19, va, 0.36, color=ORANGE, label="A 消費税↑・所得税↓")
    ax.barh(y - 0.19, vb, 0.36, color=BLUE, label="B 所得税↑・消費税↓")
    for yy, v1, v2 in zip(y, va, vb):
        for off, v, c in ((0.19, v1, ORANGE), (-0.19, v2, BLUE)):
            ax.text(v + (0.06 if v >= 0 else -0.06), yy + off, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=8.8, color=c)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab, _ in BARS])
    ax.axvline(0, color=INK2, lw=0.9)
    lim = max(abs(va).max(), abs(vb).max()) * 1.35
    ax.set_xlim(-lim, lim)
    ax.set_title("3年平均（2022〜24年）の比較", loc="left", fontsize=11.5, color=INK)
    ax.legend(loc="lower left", frameon=False, fontsize=9)

    # 判定
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    rows = [("景気（実質GDP・消費）", "B", f"B は実質GDP {avg.loc[B, '実質GDP（%）']:+.2f}%、A は {avg.loc[A, '実質GDP（%）']:+.2f}%"),
            ("物価", "B", f"B は消費者物価を{avg.loc[B, '消費者物価（消費デフレーター、%）']:+.1f}%、A は{avg.loc[A, '消費者物価（消費デフレーター、%）']:+.1f}%"),
            ("実質賃金", "B", f"物価が下がる B が {avg.loc[B, '実質賃金（%）']:+.1f}%"),
            ("財政収支", "B", f"B は{avg.loc[B, '財政収支/名目GDP（%pt）']:+.2f}%pt（物価が下がり政府支出の名目額が減る）"),
            ("家計の手取り（実質可処分所得）", "A", f"所得税が軽い A が {avg.loc[A, '実質可処分所得（%）']:+.1f}%、B は {avg.loc[B, '実質可処分所得（%）']:+.1f}%"),
            ("効果の持続", "―", "どちらも3年目には実質GDP の差が ±0.1〜0.2% に縮む")]
    ax.text(0.0, 0.97, "どちらが良い政策か（このモデルの答え）", fontsize=13, color=INK, fontweight="bold", va="top")
    yy = 0.86
    for item, win, why in rows:
        c = BLUE if win == "B" else ORANGE if win == "A" else GRAY
        ax.add_patch(plt.Rectangle((0.0, yy - 0.045), 0.07, 0.07, color=c, lw=0))
        ax.text(0.035, yy - 0.01, win, color="white" if win != "―" else INK, fontsize=12, fontweight="bold", ha="center", va="center")
        ax.text(0.09, yy + 0.005, item, fontsize=11, color=INK, va="center")
        ax.text(0.09, yy - 0.04, why, fontsize=9, color=INK2, va="center")
        yy -= 0.115
    ax.text(0.0, yy + 0.01, "→ 短期の景気・物価・財政では B（所得税増税＋消費税減税）が良い。\n"
            "   ただし効果は小さく（実質GDP +0.1〜0.3%）、時間とともに薄れる。", fontsize=10.5, color=INK, va="top", linespacing=1.6)

    fig.suptitle("税収中立の組み替え：A 消費税増税＋所得税減税 vs B 所得税増税＋消費税減税（規模は名目GDPの1%ずつ）",
                 x=0.01, ha="left", fontsize=14, color=INK, fontweight="bold", y=0.985)
    fig.text(0.01, 0.948, f"消費税率は A で+{avg.loc[A, '消費税率の変化（%pt）']:.2f}%pt、B で{avg.loc[B, '消費税率の変化（%pt）']:.2f}%pt（事前の税収変化＝名目GDPの±1%）。"
             "所得税は名目GDPの∓1%。2022Q1 から恒久。2024年版データ（2022〜24年）。", fontsize=9.5, color=INK2)
    note = ("内閣府 短期日本経済マクロ計量モデル（2022年版、ESRI Research Note No.72）の Python 再現。誤差修正項は既定（3本のみ有効）。"
            f"論文と同じ2021年版（2018〜20年）でもほぼ同じ（3年平均の実質GDP：A {a21.loc[A, '実質GDP（%）']:+.2f}%、B {a21.loc[B, '実質GDP（%）']:+.2f}%）。\n"
            "モデルの消費税の転嫁率は0.52（残りは企業の取り分）。財政収支の改善には、実質一定の政府支出の名目額が物価とともに動く会計上の効果を含む。"
            "所得分布（逆進性）や労働供給への影響はモデルにない。")
    fig.text(0.01, 0.012, note, fontsize=8, color=INK2, linespacing=1.55, va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 0.93), h_pad=2.5, w_pad=3)
    fig.savefig(OUT / "tax_swap.png", dpi=150, facecolor="white")
    print(OUT / "tax_swap.png")


if __name__ == "__main__":
    main()
