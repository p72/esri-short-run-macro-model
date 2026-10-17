"""名目GDP水準目標（NGDPLT）シナリオ（experiment_ngdplt.py の結果）の図.

左: 名目GDP（実績、目標経路 4%・5%、NGDPLT をモデルに組み込んだ解）
中: 政策金利（実績、ルールが示すシャドー金利、モデルに組み込んだ解、比較のインフレ目標ルール）
右: モデルに組み込んだ解の実質GDP・消費デフレーター（実績からの乖離）

出力: output/ngdplt.png（日本語フォント IPAPGothic または IPAGothic が必要）
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
BLUE, ORANGE, AQUA, VIOLET, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#8a63d2", "#0b0b0b", "#52514e", "#e6e5e1"
FONT = next((f for f in ("IPAPGothic", "IPAGothic") if any(f == x.name for x in font_manager.fontManager.ttflist)), "sans-serif")


def main() -> None:
    d = pd.read_csv(OUT / "experiment_ngdplt.csv", index_col=0)
    plt.rcParams.update({"font.family": FONT, "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.unicode_minus": False})
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.4))
    fig.patch.set_facecolor("white")
    x = np.arange(len(d))

    def end_label(ax, y, c, dy=0, fmt="{:+.1f}"):
        ax.annotate(fmt.format(y.iloc[-1]), (x[-1], y.iloc[-1]), xytext=(5, dy), textcoords="offset points",
                    color=c, fontsize=9, va="center")

    # 左: 名目GDP
    ax = axes[0]
    ax.plot(x, d["名目GDP 実績（兆円）"], color=INK, lw=2.4, label="実績")
    for g, c in (("4", BLUE), ("5", ORANGE)):
        ax.plot(x, d[f"名目GDP 目標{g}%（兆円）"], color=c, lw=1.4, ls=(0, (2, 2)), label=f"目標経路 年{g}%")
        ax.plot(x, d[f"モデル NGDPLT{g}% 名目GDP（兆円）"], color=c, lw=2.4, label=f"NGDPLT {g}%（モデルの解）")
        end_label(ax, d[f"名目GDP 目標{g}%（兆円）"], c, fmt="{:.0f}")
        end_label(ax, d[f"モデル NGDPLT{g}% 名目GDP（兆円）"], c, fmt="{:.0f}")
    end_label(ax, d["名目GDP 実績（兆円）"], INK, dy=-6, fmt="{:.0f}")
    ax.set_title("名目GDP（兆円、季節調整済み年率）", loc="left", fontsize=11.5, color=INK)
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    # 中: 政策金利
    ax = axes[1]
    ax.plot(x, d["短期金利 実績（%）"], color=INK, lw=2.4, label="実績")
    ax.plot(x, d["インフレ目標（消費デフレーター）のシャドー金利（%）"], color=VIOLET, lw=2.0, ls=(0, (5, 2)),
            label="インフレ目標ルール（消費者物価）")
    for g, c in (("4", BLUE), ("5", ORANGE)):
        ax.plot(x, d[f"NGDPLT{g}% φ1 シャドー金利（%）"], color=c, lw=1.6, ls=(0, (2, 2)), label=f"NGDPLT {g}% ルール（実績の経済のまま）")
        ax.plot(x, d[f"モデル NGDPLT{g}% 短期金利（%）"], color=c, lw=2.4, label=f"NGDPLT {g}%（モデルの解）")
        end_label(ax, d[f"NGDPLT{g}% φ1 シャドー金利（%）"], c)
        end_label(ax, d[f"モデル NGDPLT{g}% 短期金利（%）"], c)
    end_label(ax, d["インフレ目標（消費デフレーター）のシャドー金利（%）"], VIOLET)
    ax.axhline(0, color=INK2, lw=0.9)
    ax.set_title("政策金利（短期金利、%）", loc="left", fontsize=11.5, color=INK)
    ax.legend(loc="lower left", frameon=False, fontsize=8.6)
    i22 = list(d.index).index("2022Q2")
    ax.annotate("原油高・円安でも\nNGDPLT は利上げしない", (i22, d["NGDPLT4% φ1 シャドー金利（%）"].iloc[i22]),
                xytext=(0.3, 4.6), fontsize=9.5, color=INK, arrowprops=dict(arrowstyle="->", color=INK2, lw=1))

    # 右: 実質GDP・消費デフレーター
    ax = axes[2]
    for g, c in (("4", BLUE), ("5", ORANGE)):
        ax.plot(x, d[f"モデル NGDPLT{g}% 実質GDP（基準解比 %）"], color=c, lw=2.4, label=f"実質GDP（NGDPLT {g}%）")
        ax.plot(x, d[f"モデル NGDPLT{g}% 消費デフレーター（基準解比 %）"], color=c, lw=1.8, ls=(0, (1, 1.3)),
                label=f"消費者物価（NGDPLT {g}%）")
        end_label(ax, d[f"モデル NGDPLT{g}% 実質GDP（基準解比 %）"], c, dy=6 if g == "4" else 0)
        end_label(ax, d[f"モデル NGDPLT{g}% 消費デフレーター（基準解比 %）"], c, dy=-6 if g == "5" else 0)
    ax.axhline(0, color=INK2, lw=0.9)
    ax.set_title("NGDPLT のモデルの解（実績からの乖離、%）", loc="left", fontsize=11.5, color=INK)
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    for ax in axes:
        ax.set_xticks(x[::4])
        ax.set_xticklabels([f"{s[:4]}年" for s in d.index[::4]])
        ax.set_xlim(-0.4, len(d) + 1.4)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("2020年から名目GDP水準目標（年4%・5%）を採っていたら、政策金利は何%か", x=0.01, ha="left", fontsize=14,
                 color=INK, fontweight="bold", y=0.985)
    fig.text(0.01, 0.935, "目標経路＝2019年の名目GDP（平均）から年4%・5%で伸ばした水準。ルール：i ＝ 0.71×前期の i ＋ 0.29×（g ＋ 名目GDPギャップ%）。"
             "金利の下限は置かない（シャドー金利）。", fontsize=9.5, color=INK2)
    note = ("内閣府 短期日本経済マクロ計量モデル（2022年版、ESRI Research Note No.72）の Python 再現、2024年版データ（国民経済計算 2024年度年次推計など）。"
            "点線のルールは実績の経済のまま計算、実線はモデルの式113 を差し替えて解いた値。\n"
            "インフレ目標ルールはモデルの式114 と同じ形（潜在成長率＋2＋1.68×（物価上昇率−2）＋0.15×GDPギャップ）で、前年比の消費デフレーターを使う。"
            "長期金利は式116 で短期金利に連動し、下限 0 にかかる。誤差修正項は既定（3本のみ有効）。")
    fig.text(0.01, 0.012, note, fontsize=8, color=INK2, linespacing=1.55, va="bottom")
    fig.tight_layout(rect=(0, 0.07, 1, 0.92), w_pad=2.2)
    fig.savefig(OUT / "ngdplt.png", dpi=160, facecolor="white")
    print(OUT / "ngdplt.png")


if __name__ == "__main__":
    main()
