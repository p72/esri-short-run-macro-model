"""モデルと2022〜24年の実績の違い（experiment_expost_2022.py の結果）の図.

上段: 実績とモデルの予測（2022Q1 から誤差項を平時の平均にして解いた値）の水準（2021Q4=100）
     GDPデフレーター、消費デフレーター、名目雇用者報酬
下段: 2024年の実績−予測を、主な式ごとの寄与に分けた横棒（GDPデフレーター、名目雇用者報酬）と、
     誤差項が平時から大きく外れた式（z 値）

出力: output/expost_2022.png（日本語フォント IPAPGothic が必要）
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
BLUE, ORANGE, AQUA, INK, INK2, GRID, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e6e5e1", "#b9b8b3"
LABEL = {  # 式 → 図の名前
    "式55 PGDPAT": "式55 国内物価（GDPデフレーター）",
    "式56 PCPAT": "式56 消費デフレーター",
    "式65 PXGS": "式65 輸出物価",
    "式57 CGPIAT": "式57 企業物価",
    "式67 PFUELAT": "式67 燃料輸入物価",
    "式68 PNFMGSAT": "式68 非燃料輸入物価",
    "式89 YWV": "式89 雇用者報酬（労働分配率）",
    "式47 LF": "式47 労働力人口",
    "式3 CP": "式3 消費",
}
ZNAME = {"CGPI": "式74 企業物価（税込み）", "PGDPAT": "式55 国内物価", "PLAND": "式119 地価", "LF": "式47 労働力人口",
         "PNFMGSAT": "式68 非燃料輸入物価", "YOLIV": "式86 雇主の社会負担", "ITR": "式129 所得税の実効税率",
         "CSSV": "式137 社会保険料", "CCAVG": "式139 政府の固定資本減耗", "RSHARE": "式117 株式収益率",
         "IHP": "式5 住宅投資", "RGBX": "式116 長期金利", "PFUELAT": "式67 燃料輸入物価", "PXGS": "式65 輸出物価",
         "XGS": "式9 輸出", "CGPIAT": "式57 企業物価", "YWV": "式89 雇用者報酬", "PCPAT": "式56 消費デフレーター"}


def main() -> None:
    lv = pd.read_csv(OUT / "experiment_expost_2022_levels.csv", index_col=0)
    d = pd.read_csv(OUT / "experiment_expost_2022.csv")
    z = pd.read_csv(OUT / "experiment_expost_2022_residuals.csv")
    zc = [c for c in z.columns if c.startswith("z値")][0]

    plt.rcParams.update({"font.family": "IPAPGothic", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.unicode_minus": False})
    fig = plt.figure(figsize=(15, 10))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15])

    x = np.arange(len(lv))
    q = list(lv.index)
    start = q.index("2022Q1")
    for i, var in enumerate(["GDPデフレーター", "消費デフレーター", "名目雇用者報酬"]):
        ax = fig.add_subplot(gs[0, i])
        ax.axvspan(start - 0.5, len(q) - 0.5, color="#f6f5f2", zorder=0)
        # 名目雇用者報酬は、式89 が見る国民所得 NIV の四半期の振れ（季節調整の不安定さ）を予測が拾うので4四半期移動平均
        sm = (lambda ser: ser.rolling(4).mean()) if var == "名目雇用者報酬" else (lambda ser: ser)
        act = sm(lv[f"{var}|実績"])
        ax.plot(x, act, color=BLUE, lw=2.4, label="実績")
        pred = sm(lv[f"{var}|モデルの予測"].where(lv.index >= "2022Q1", lv[f"{var}|実績"])).to_numpy()
        ax.plot(x[start - 1:], pred[start - 1:], color=ORANGE, lw=2.4, ls=(0, (5, 2)), label="モデルの予測")
        for nm, c in (("実績", BLUE), ("モデルの予測", ORANGE)):
            yv = sm(lv[f"{var}|{nm}"].where(lv.index >= "2022Q1", lv[f"{var}|実績"])).iloc[-1]
            ax.annotate(f"{yv:.1f}", (x[-1], yv), xytext=(6, 0), textcoords="offset points", color=c, fontsize=10, va="center")
        ax.axhline(100, color=INK2, lw=0.8)
        ax.set_xticks(x[::4])
        ax.set_xticklabels([s[:4] for s in q[::4]])
        ax.set_xlim(-0.5, len(q) + 1.2)
        ttl = f"{var}（{'4期移動平均、' if var == '名目雇用者報酬' else ''}2021Q4=100）".replace('名目雇用者報酬（', '雇用者報酬（')
        ax.set_title(ttl, loc="left", fontsize=11.5, color=INK)
        if i == 0:
            ax.legend(loc="upper left", frameon=False, fontsize=10)
            ax.text(start - 0.2, ax.get_ylim()[0] + 0.4, "2022年〜 予測", fontsize=9, color=INK2)

    # 2024年の実績−予測の、主な式ごとの寄与
    y24 = d[d["年"] == 2024].set_index("区分")
    total = y24.loc["実績−モデルの予測（全体）"]
    singles = {k.replace(" だけの寄与", ""): y24.loc[k] for k in y24.index if k.endswith("だけの寄与")}
    for j, (col, title) in enumerate([("GDPデフレーター", "2024年の GDPデフレーターの差（実績−予測、%）"),
                                      ("名目雇用者報酬", "2024年の名目雇用者報酬の差（実績−予測、%）")]):
        ax = fig.add_subplot(gs[1, j])
        items = sorted(((LABEL[k], v[col]) for k, v in singles.items()), key=lambda t: t[1])
        items.append(("その他の式・交差項", total[col] - sum(v for _, v in items)))
        items.append(("合計（実績−予測）", total[col]))
        y = np.arange(len(items))
        vals = np.array([v for _, v in items])
        colors = [INK if lab.startswith("合計") else GRAY if lab.startswith("その他") else (BLUE if v >= 0 else ORANGE)
                  for lab, v in items]
        ax.barh(y, vals, 0.62, color=colors)
        for yi, v in zip(y, vals):
            ax.text(v + (0.2 if v >= 0 else -0.2), yi, f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right",
                    fontsize=9.5, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels([lab for lab, _ in items], fontsize=9.5)
        ax.axvline(0, color=INK2, lw=0.9)
        lo, hi = vals.min(), vals.max()
        ax.set_xlim(lo - 3, hi + 3)
        ax.set_title(title, loc="left", fontsize=11.5, color=INK)
        ax.grid(axis="x", color=GRID, lw=0.8)

    # 誤差項の z 値
    ax = fig.add_subplot(gs[1, 2])
    top = z[z["変数"].isin(ZNAME)].head(10).iloc[::-1]
    y = np.arange(len(top))
    vals = top[zc].to_numpy()
    ax.barh(y, vals, 0.62, color=[BLUE if v >= 0 else ORANGE for v in vals])
    for yi, v in zip(y, vals):
        ax.text(v + (0.15 if v >= 0 else -0.15), yi, f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right", fontsize=9.5)
    ax.set_yticks(y)
    ax.set_yticklabels([ZNAME[v] for v in top["変数"]], fontsize=9.5)
    ax.axvline(0, color=INK2, lw=0.9)
    ax.set_xlim(vals.min() - 1.5, vals.max() + 1.5)
    ax.set_title("誤差項が平時から外れた式（z 値、上位10）", loc="left", fontsize=11.5, color=INK)
    ax.grid(axis="x", color=GRID, lw=0.8)

    for ax in fig.axes:
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if ax not in fig.axes[:3]:
            continue
        ax.grid(axis="y", color=GRID, lw=0.8)
    fig.suptitle("モデルと2022〜24年の実績は、どこで違うのか：国内物価の式が説明できない値上げ", x=0.01, ha="left",
                 fontsize=14, color=INK, fontweight="bold", y=0.985)
    fig.text(0.01, 0.945, "モデルの予測＝2022Q1 から、外生変数・為替・金利は実績のまま、各式の誤差項だけを平時（2014〜19年）の平均にして解いた値。"
             "実績との差は、その式の仕組みでは説明できない動き。", fontsize=9.5, color=INK2)
    note = ("内閣府 短期日本経済マクロ計量モデル（2022年版、ESRI Research Note No.72）の Python 再現、2024年版データ（国民経済計算 2026年4-6月期2次QE・2024年度年次推計）。"
            "誤差修正項はすべて有効。\n式ごとの寄与は「その式の誤差項だけ実績に戻す」と「全部実績にしてその式だけ平時にする」の平均。"
            "z 値は（2022〜24年の誤差項の平均−平時の平均）/平時の標準偏差。"
            "名目雇用者報酬の予測は、国民所得の四半期の振れを拾うので4四半期移動平均で示す。")
    fig.text(0.01, 0.012, note, fontsize=8, color=INK2, linespacing=1.55, va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94), h_pad=2.5, w_pad=2.0)
    fig.savefig(OUT / "expost_2022.png", dpi=160, facecolor="white")
    print(OUT / "expost_2022.png")


if __name__ == "__main__":
    main()
