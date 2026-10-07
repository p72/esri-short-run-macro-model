"""円安の物価押し上げ効果（為替パススルー）の図.

左: CPI の2年累積パススルー（名目実効為替レートが1%円安のとき、2年後の CPI 上昇率%）の10年移動窓の推移と90%信頼区間。
    モデル（論文の円10%減価シナリオ）の 2年目の消費デフレーター効果（円1%あたり約0.016%）を参照線で示す。
右: 2024年版データ（基準解 2022〜24年）で円10%減価を解いたときの消費デフレーター PCP の経路。
    既定の係数と、式56・68 を2011〜2024年で推定し直した係数を比べる。

入力: output/experiment_fx_passthrough_{rolling,paths,model}.csv（experiment_fx_passthrough.py が作る）
出力: output/fx_passthrough.png（日本語フォント IPAPGothic が必要）
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
BLUE, ORANGE, AQUA, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e6e5e1"
Z90 = 1.645


def main() -> None:
    roll = pd.read_csv(OUT / "experiment_fx_passthrough_rolling.csv")
    paths = pd.read_csv(OUT / "experiment_fx_passthrough_paths.csv", index_col=0)
    model = pd.read_csv(OUT / "experiment_fx_passthrough_model.csv")
    ref = model.loc[(model["基準解"] == "既定版（2018〜20年）") & (model["ケース"] == "既定の係数"), "PCP_2年目"].iloc[0] / 10

    plt.rcParams.update({"font.family": "IPAPGothic", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.unicode_minus": False})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw={"width_ratios": [1.35, 1]})
    fig.patch.set_facecolor("white")

    # 左: 移動窓のパススルー
    for name, lab, c in (("CPI 財", "CPI 財", ORANGE), ("CPI 生鮮食品を除く総合", "CPI 生鮮食品を除く総合（コア）", BLUE)):
        r = roll[roll["系列"] == name].copy()
        r["t"] = pd.PeriodIndex(r["窓の終わり"], freq="Q").to_timestamp(how="end")
        r = r[r["t"] >= "1995-01-01"]
        ax1.fill_between(r["t"], r["係数"] - Z90 * r["標準誤差"], r["係数"] + Z90 * r["標準誤差"], color=c, alpha=0.13, lw=0)
        ax1.plot(r["t"], r["係数"], color=c, lw=2.2, label=lab)
        last = r.iloc[-1]
        ax1.annotate(f"{last['係数']:.2f}", (last["t"], last["係数"]), xytext=(6, 0), textcoords="offset points",
                     color=c, fontsize=10, va="center")
    ax1.axhline(ref, color=INK, lw=1.2, ls=(0, (5, 3)))
    ax1.text(pd.Timestamp("1995-06-01"), ref + 0.012, f"内閣府モデル（論文の円10%減価、2年目）: 円1%あたり {ref:.3f}%",
             color=INK, fontsize=9)
    ax1.axhline(0, color=INK2, lw=0.8)
    ax1.set_ylabel("2年累積の上昇率（円1%安あたり、%）")
    ax1.set_title("CPI の為替パススルー（10年移動窓、窓の終わりの時点で表示）", loc="left", fontsize=11, color=INK)
    ax1.legend(loc="upper left", frameon=False)

    # 右: モデルの PCP 経路
    x = range(1, len(paths) + 1)
    series = [("2024年版（2022〜24年）|既定の係数", "既定の係数（論文、推定 1990年代〜2020年）", INK2, "-"),
              ("2024年版（2022〜24年）|式68 を推定し直す", "式68（非燃料輸入物価）だけ推定し直す", AQUA, (0, (1, 1.3))),
              ("2024年版（2022〜24年）|式56 を推定し直す", "式56（消費デフレーター）だけ推定し直す", ORANGE, (0, (5, 2))),
              ("2024年版（2022〜24年）|式56・68 を推定し直す", "式56・68 を推定し直す（2011〜2024年）", BLUE, "-")]
    for (col, lab, c, ls), dy in zip(series, (-5, 5, 0, 0)):  # 近い2本のラベルを上下にずらす
        ax2.plot(x, paths[col], color=c, lw=2.2, ls=ls, label=lab)
        ax2.annotate(f"{paths[col].iloc[-1]:.2f}", (len(paths), paths[col].iloc[-1]), xytext=(6, dy),
                     textcoords="offset points", color=c, fontsize=10, va="center")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels([str(i) for i in x])
    ax2.set_xlabel("円10%減価からの四半期")
    ax2.set_ylabel("消費デフレーター PCP（基準解からの乖離、%）")
    ax2.set_ylim(0, None)
    ax2.set_title("モデルで円10%減価（基準解 2022〜24年）", loc="left", fontsize=11, color=INK)
    ax2.legend(loc="upper left", frameon=False, fontsize=9)

    for ax in (ax1, ax2):
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("円安の物価押し上げ効果は大きくなったか", x=0.01, ha="left", fontsize=14, color=INK, fontweight="bold", y=0.975)
    note = ("左: 日銀 名目実効為替レート・輸入物価指数（契約通貨ベース）、総務省 CPI（2020年基準）の四半期平均。CPI の8四半期の対数変化を、円安（実効レートの逆数）・"
            "契約通貨建て輸入物価・消費税率の8四半期変化で回帰。帯は Newey-West（ラグ8）の90%信頼区間。\n"
            "右: 内閣府 短期日本経済マクロ計量モデル（2022年版、ESRI Research Note No.72）の Python 再現。2024年版データ。誤差修正項は既定（消費デフレーターなど3本のみ有効）。"
            "推定し直した式は論文と同じ形の OLS。")
    fig.text(0.01, 0.012, note, fontsize=8, color=INK2, linespacing=1.55, va="bottom")
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    fig.savefig(OUT / "fx_passthrough.png", dpi=160, facecolor="white")
    print(OUT / "fx_passthrough.png")


if __name__ == "__main__":
    main()
