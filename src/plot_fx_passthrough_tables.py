"""note 記事（docs/note_fx_passthrough.md）に貼る表の画像を作る.

数値は output/experiment_fx_passthrough_{model,cpi,coef}.csv（experiment_fx_passthrough.py の結果）を丸めて書き写したもの。
出力: output/fx_passthrough_table{1..4}_*.png（日本語フォント IPAPGothic が必要）
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
out = Path(__file__).resolve().parents[1] / "output"
INK, INK2, GRID, HEAD, HL = "#0b0b0b", "#52514e", "#e6e5e1", "#f3f2ee", "#fdeee6"
plt.rcParams.update({"font.family": "IPAPGothic"})
def table(name, title, header, rows, widths, hl_cells=(), note=None):
    W = sum(widths); rh = 0.62; h = rh * (len(rows) + 1) + 1.0 + (0.45 if note else 0.25)
    fig = plt.figure(figsize=(W, h)); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(h, 0); ax.axis("off")
    ax.text(0.25, 0.5, title, fontsize=15, color=INK, va="center")
    y0 = 1.0
    for r, row in enumerate([header] + rows):
        y = y0 + r * rh; x = 0
        ax.add_patch(plt.Rectangle((0, y), W, rh, color=HEAD if r == 0 else "white", lw=0))
        for c, (txt, w) in enumerate(zip(row, widths)):
            if (r, c) in hl_cells:
                ax.add_patch(plt.Rectangle((x, y), w, rh, color=HL, lw=0))
            bold = (r, c) in hl_cells
            ax.text(x + (0.2 if c == 0 else w / 2), y + rh / 2, txt, fontsize=12.5 if r else 12,
                    color=INK if (r and c) or bold else INK2 if r == 0 else INK,
                    ha="left" if c == 0 else "center", va="center", fontweight="bold" if bold else "normal")
            x += w
        ax.plot([0, W], [y + rh, y + rh], color=GRID, lw=1)
    if note:
        ax.text(0.25, y0 + (len(rows) + 1) * rh + 0.3, note, fontsize=10, color=INK2, va="center")
    fig.savefig(out / f"fx_passthrough_{name}.png", dpi=180, facecolor="white"); plt.close(fig)

table("table1_model", "内閣府モデル：円10%安のときの消費者物価", ["", "1年目", "2年目", "3年目"],
      [["消費者物価（家計消費デフレーター）", "+0.15%", "+0.16%", "+0.24%"]], [4.6, 1.5, 1.5, 1.5],
      note="円1%あたり約0.02%。再現の結果（論文は3年目 +0.23%で、ほぼ同じ）。")
table("table2_cpi", "円が1%安くなったとき、2年間で物価が何%上がったか", ["", "1995〜2012年", "2013〜2020年", "2021年〜"],
      [["コアCPI（生鮮食品を除く総合）", "−0.01", "0.03", "0.18（0.12）"],
       ["コアコアCPI（生鮮食品・エネルギーを除く）", "−0.01", "−0.00", "0.21（0.10）"],
       ["CPI 財", "−0.01", "0.04", "0.28（0.13）"],
       ["CPI サービス", "0.00", "0.02", "0.11（0.09）"]], [4.9, 1.8, 1.8, 1.9],
      hl_cells={(1, 3), (2, 3), (3, 3)},
      note="括弧は標準誤差（Newey-West）。日本銀行・総務省のデータから推定。")
table("table3_coef", "最近のデータ（2011〜2024年）で係数を推定し直すと", ["", "推定し直した値", "論文"],
      [["輸入物価の円安への反応（式68、長期）", "0.51", "0.25"],
       ["消費者物価の輸入物価への反応（式56、長期）", "0.30", "0.021"]], [5.2, 2.1, 1.6],
      hl_cells={(2, 1)})
table("table4_rerun", "推定し直した係数で、円10%安を計算し直す", ["", "1年目", "2年目", "3年目"],
      [["論文の係数", "+0.15%", "+0.17%", "+0.23%"],
       ["推定し直した係数", "+0.24%", "+0.49%", "+0.72%"]], [3.4, 1.5, 1.5, 1.5],
      hl_cells={(2, 2), (2, 3)},
      note="2022〜24年の経済を出発点に計算。円1%あたり、2年目で約0.05%、3年目で約0.07%。")
print(out / "fx_passthrough_table*.png")
