"""交易損失と金融政策の枠組み（experiment_tot_policy.py の結果）の図.

上段: 名目雇用者報酬・消費デフレーター・実質GDP の基準解からの乖離（四半期）
下段: 短期金利の乖離、GDP デフレーターの寄与度分解（分配面・支出面、3年平均）

出力: output/tot_policy.png（日本語フォント IPAPGothic が必要）
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
BLUE, ORANGE, AQUA, VIOLET, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#8a63d2", "#0b0b0b", "#52514e", "#e6e5e1"
STYLE = {  # 政策 → (短い名前, 色, 線種)
    "緩和継続（金利を固定）": ("緩和継続", BLUE, "-"),
    "モデルどおり（テイラー・ルール）": ("モデルどおり", INK2, (0, (5, 2))),
    "物価重視の引締め（消費者物価に反応）": ("物価重視の引締め", ORANGE, "-"),
    "参考: 緩和継続＋パススルーを推定し直した式": ("参考: パススルー更新", AQUA, (0, (1, 1.3))),
}


def main() -> None:
    p = pd.read_csv(OUT / "experiment_tot_policy_paths.csv")
    a = pd.read_csv(OUT / "experiment_tot_policy.csv")
    rate = a["輸入価格の上昇率（%）"].iloc[0]
    avg = a.drop(columns=["年", "輸入価格の上昇率（%）"]).groupby("政策", sort=False).mean()

    plt.rcParams.update({"font.family": "IPAPGothic", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.unicode_minus": False})
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.2))
    fig.patch.set_facecolor("white")

    quarters = p["四半期"].unique()
    x = np.arange(len(quarters))
    for ax, col, title in [(axes[0, 0], "名目雇用者報酬", "名目雇用者報酬（%）"),
                           (axes[0, 1], "消費デフレーター", "消費者物価（消費デフレーター、%）"),
                           (axes[0, 2], "実質GDP", "実質GDP（%）"),
                           (axes[1, 0], "短期金利（%pt）", "短期金利（%pt）")]:
        for pol, (short, c, ls) in STYLE.items():
            y = p.loc[p["政策"] == pol, col].to_numpy()
            ax.plot(x, y, color=c, lw=2.2, ls=ls, label=short)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xticks(x[::4])
        ax.set_xticklabels([q[:4] for q in quarters[::4]])
    axes[0, 1].legend(loc="upper left", frameon=False, fontsize=9.5)

    # 寄与度分解（3年平均）
    pols = list(STYLE)
    xb = np.arange(len(pols))
    for ax, parts, title in [
        (axes[1, 1], [("寄与_単位労働コスト", "単位労働コスト", BLUE), ("寄与_単位利潤", "単位利潤", ORANGE),
                      ("寄与_固定資本減耗", "固定資本減耗", AQUA), ("寄与_純間接税", "純間接税", VIOLET)],
         "GDPデフレーターの分配面の寄与（%pt、3年平均）"),
        (axes[1, 2], [("寄与_国内需要デフレーター", "国内需要デフレーター", BLUE), ("寄与_交易条件", "交易条件", ORANGE)],
         "GDPデフレーターの支出面の寄与（%pt、3年平均）")]:
        pos, neg = np.zeros(len(pols)), np.zeros(len(pols))
        for col, lab, c in parts:
            v = avg.loc[pols, col].to_numpy()
            bottom = np.where(v >= 0, pos, neg)
            ax.bar(xb, v, 0.6, bottom=bottom, color=c, label=lab, edgecolor="white", lw=0.8)
            pos += np.where(v >= 0, v, 0)
            neg += np.where(v < 0, v, 0)
        tot = avg.loc[pols, "GDPデフレーター"].to_numpy()
        ax.scatter(xb, tot, color=INK, zorder=4, s=28, label="GDPデフレーター")
        for xi, t in zip(xb, tot):
            ax.annotate(f"{t:+.2f}", (xi, t), xytext=(14, 0), textcoords="offset points", fontsize=9, color=INK, va="center")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(xb)
        ax.set_xticklabels([STYLE[k][0].replace("参考: ", "参考:\n") for k in pols], fontsize=9)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.legend(loc="lower left", frameon=False, fontsize=8.5, ncol=2)
        ax.set_ylim(min(neg.min(), tot.min()) * 2.0, max(pos.max(), 0.3) * 1.4)

    for ax in axes.flat:
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("交易損失は賃金を下げるのか：同じ交易ショックを金融政策の設定を変えて解く", x=0.01, ha="left",
                 fontsize=14, color=INK, fontweight="bold", y=0.985)
    fig.text(0.01, 0.945, f"2022Q1 から原油価格と海外の輸入価格を +{rate:.0f}%（1年目の交易条件要因が2022年の実績 −2.46%pt と同じ規模）。"
             "基準解（実績、2022〜24年）からの乖離。", fontsize=9.5, color=INK2)
    note = ("内閣府 短期日本経済マクロ計量モデル（2022年版、ESRI Research Note No.72）の Python 再現、2024年版データ。緩和継続は短期・長期金利を固定、"
            "モデルどおりはテイラー・ルール（式114）、物価重視の引締めは消費者物価の前年比の乖離×1.5 の利上げ。\n"
            "参考は緩和継続のまま、輸入物価（式68）と消費デフレーター（式56）を2011〜2024年で推定し直した式。誤差修正項は既定（3本のみ有効）。"
            "単位利潤は名目GDP から雇用者報酬・固定資本減耗・純間接税を除いた残差（統計上の不突合を含む）。")
    fig.text(0.01, 0.012, note, fontsize=8, color=INK2, linespacing=1.55, va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94), h_pad=2.2)
    fig.savefig(OUT / "tot_policy.png", dpi=160, facecolor="white")
    print(OUT / "tot_policy.png")


if __name__ == "__main__":
    main()
