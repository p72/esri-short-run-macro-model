"""交易損失と金融政策の枠組み（experiment_tot_policy.py）の解説図3枚.

1. モデルの答えは2008年型: GDPデフレーターの支出面の寄与（1年目）を、記事の実績（2008年・2022年）と並べる
2. 金融政策の違いはほとんど効かない: 名目雇用者報酬と短期金利の四半期経路
3. 交易損失は誰が負担したか: 交易損失を100とした負担の内訳（3年平均）

実績の数値は note 記事「交易損失は賃金を下げるのか」（内閣府 国民経済計算 2020年基準からの計算、前年比の寄与度）。
実績は前年比、モデルは基準解からの乖離なので、厳密な比較ではなく「形」の比較。

表: note 記事（docs/note_tot_policy.md）用に、3年平均の主な数値を表の画像にする（数値は CSV から読む）

出力: output/tot_policy_explain{1,2,3}.png, output/tot_policy_table.png（日本語フォント IPAPGothic が必要）
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
GRAY = "#b9b8b3"
# 記事の実績（GDPデフレーター前年比の寄与度、%pt）: (国内需要デフレーター, 交易条件, GDPデフレーター)
ACTUAL = {"実績 2008年": (0.71, -1.61, -0.90), "実績 2022年": (3.16, -2.46, 0.70)}
SHORT = {"緩和継続（金利を固定）": "モデル: 緩和継続",
         "モデルどおり（テイラー・ルール）": "モデル: テイラー・ルール",
         "物価重視の引締め（消費者物価に反応）": "モデル: 物価重視の引締め",
         "参考: 緩和継続＋パススルーを推定し直した式": "参考: パススルー更新"}
SRC = ("内閣府 短期日本経済マクロ計量モデル（2022年版、ESRI Research Note No.72）の Python 再現、2024年版データ。"
       "2022Q1 から原油価格と海外の輸入価格を +25%。")


def style() -> None:
    plt.rcParams.update({"font.family": "IPAPGothic", "font.size": 10.5, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.unicode_minus": False})


def finish(ax) -> None:
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def header(fig, title: str, sub: str) -> None:
    fig.suptitle(title, x=0.02, ha="left", fontsize=16, color=INK, fontweight="bold", y=0.975)
    fig.text(0.02, 0.93, sub, fontsize=10.5, color=INK2, va="top", linespacing=1.5)


def fig1(a: pd.DataFrame) -> None:
    y1 = a[a["年"].str.startswith("1年目")].set_index("政策")
    rows = list(ACTUAL.items()) + [(SHORT[k], (y1.loc[k, "寄与_国内需要デフレーター"], y1.loc[k, "寄与_交易条件"],
                                              y1.loc[k, "GDPデフレーター"])) for k in SHORT]
    labels = [r[0] for r in rows]
    dd = np.array([r[1][0] for r in rows])
    tot = np.array([r[1][1] for r in rows])
    gdpd = np.array([r[1][2] for r in rows])
    fig, ax = plt.subplots(figsize=(12, 6.6))
    fig.patch.set_facecolor("white")
    y = np.arange(len(rows))[::-1]
    ax.barh(y, tot, 0.6, color=ORANGE, label="交易条件（輸入コストの上昇）")
    ax.barh(y, dd, 0.6, color=BLUE, label="国内需要デフレーター（国内の物価）")
    ax.scatter(gdpd, y, color=INK, s=40, zorder=4, label="GDPデフレーター（合計）")
    for yi, d_, t_, g_ in zip(y, dd, tot, gdpd):
        ax.text(d_ + 0.08, yi, f"国内物価 {d_:+.2f}", va="center", fontsize=9.5, color=BLUE)
        ax.text(t_ - 0.08, yi, f"{t_:+.2f}", va="center", ha="right", fontsize=9.5, color=ORANGE)
        ax.annotate(f"{g_:+.2f}", (g_, yi), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=9, color=INK)
    ax.axvline(0, color=INK2, lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.axhspan(y[1] - 0.5, y[1] + 0.5, color="#fdeee6", zorder=0)
    ax.axhspan(y[0] - 0.5, y[0] + 0.5, color="#eef4fc", zorder=0)
    ax.text(4.55, y[0], "国内物価で\nほとんど吸収されない", fontsize=10.5, color=INK, va="center", ha="left")
    ax.text(4.55, y[1], "国内物価が上がって\n交易損失を分担", fontsize=10.5, color=INK, va="center", ha="left")
    ax.text(4.55, (y[2] + y[5]) / 2, "モデルは\nどの政策でも\n2008年に近い形", fontsize=10.5, color=INK, va="center", ha="left")
    ax.set_xlim(-3.2, 6.4)
    ax.set_xlabel("GDPデフレーターへの寄与（%pt）")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=10)
    finish(ax)
    header(fig, "① モデルの答えは「2008年型」：輸入コストの上昇が国内の物価に回らない",
           "GDPデフレーター＝国内需要デフレーター（国内の物価）＋交易条件。交易損失の規模は2022年の実績と同じ（1年目）。")
    fig.text(0.02, 0.015, "実績は note 記事「交易損失は賃金を下げるのか」（内閣府 国民経済計算から計算、前年比の寄与度）。"
             "モデルは基準解（実績）からの乖離（1年目の平均）で、厳密な比較ではなく形の比較。\n" + SRC,
             fontsize=8.3, color=INK2, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.07, 1, 0.89))
    fig.savefig(OUT / "tot_policy_explain1.png", dpi=160, facecolor="white")


def fig2(p: pd.DataFrame) -> None:
    pols = [("緩和継続（金利を固定）", "緩和継続（金利を固定）", BLUE, "-"),
            ("モデルどおり（テイラー・ルール）", "モデルどおり（テイラー・ルール）", INK2, (0, (5, 2))),
            ("物価重視の引締め（消費者物価に反応）", "物価重視の引締め（消費者物価×1.5で利上げ）", ORANGE, "-")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.8))
    fig.patch.set_facecolor("white")
    q = p["四半期"].unique()
    x = np.arange(len(q))
    for ax, col, title in [(ax1, "名目雇用者報酬", "名目の賃金（雇用者報酬、基準解からの乖離 %）"),
                           (ax2, "短期金利（%pt）", "短期金利（基準解からの乖離 %pt）")]:
        for k, lab, c, ls in pols:
            ax.plot(x, p.loc[p["政策"] == k, col].to_numpy(), color=c, lw=2.4, ls=ls, label=lab)
        ax.axhline(0, color=INK2, lw=0.9)
        ax.set_xticks(x[::4])
        ax.set_xticklabels([f"{s[:4]}年" for s in q[::4]])
        ax.set_title(title, loc="left", fontsize=11.5, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.8)
        finish(ax)
    ax1.set_ylim(-3.2, 1.1)
    ax1.annotate("3つの政策でほぼ同じ。\nどれでも名目賃金は約2%下がる", (6, -2.1), xytext=(4.2, -0.9), fontsize=10.5, color=INK,
                 arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
    ax2.annotate("引締めでも利上げは最大0.4%pt。\n消費者物価がほとんど上がらないので\n上げる理由も小さい",
                 (3, 0.41), xytext=(4.6, 0.25), fontsize=10.5, color=INK, arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
    ax2.annotate("テイラー・ルールは金利がほぼ0で\nほとんど動かない", (6, 0.06), xytext=(5.2, 0.13), fontsize=10.5, color=INK,
                 arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
    ax1.legend(loc="upper left", frameon=False, fontsize=9.5)
    header(fig, "② 金融政策の違いはほとんど効かない",
           "金利の下限に近い2022年の出発点では、金利で名目需要を支える余地も、物価を抑えるために上げる理由も小さい。")
    fig.text(0.02, 0.015, SRC + "緩和継続は短期・長期金利を固定、テイラー・ルールはモデルの式114。", fontsize=8.3, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.89))
    fig.savefig(OUT / "tot_policy_explain2.png", dpi=160, facecolor="white")


def fig3(a: pd.DataFrame) -> None:
    avg = a.drop(columns=["年", "輸入価格の上昇率（%）"]).groupby("政策", sort=False).mean()
    keys = ["緩和継続（金利を固定）", "物価重視の引締め（消費者物価に反応）", "参考: 緩和継続＋パススルーを推定し直した式"]
    loss = -avg.loc[keys, "寄与_交易条件"]
    parts = [("国内の物価の上昇\n（消費者・買い手が負担）", avg.loc[keys, "寄与_国内需要デフレーター"], BLUE),
             ("賃金の減少\n（単位労働コスト）", -avg.loc[keys, "寄与_単位労働コスト"], ORANGE),
             ("利潤の減少\n（単位利潤）", -avg.loc[keys, "寄与_単位利潤"], AQUA),
             ("その他\n（固定資本減耗・純間接税）", -(avg.loc[keys, "寄与_固定資本減耗"] + avg.loc[keys, "寄与_純間接税"]), GRAY)]
    shares = [(lab, (v / loss * 100).to_numpy(), c) for lab, v, c in parts]
    total = sum(s for _, s, _ in shares)
    assert np.allclose(total, 100), total
    fig, ax = plt.subplots(figsize=(13, 5.6))
    fig.patch.set_facecolor("white")
    y = np.arange(len(keys))[::-1]
    pos, neg = np.zeros(len(keys)), np.zeros(len(keys))
    for lab, s, c in shares:
        left = np.where(s >= 0, pos, neg + s)
        ax.barh(y, np.abs(s), 0.58, left=left, color=c, label=lab.replace("\n", ""), edgecolor="white", lw=1)
        for yi, l_, w_ in zip(y, left, s):
            if abs(w_) >= 6:
                ax.text(l_ + abs(w_) / 2, yi, f"{w_:.0f}%", ha="center", va="center", fontsize=11,
                        color="white" if c != GRAY else INK, fontweight="bold")
        pos += np.where(s >= 0, s, 0)
        neg += np.where(s < 0, s, 0)
    ax.axvline(0, color=INK2, lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[k] for k in keys], fontsize=11)
    ax.set_xlim(min(neg.min(), 0) - 2, 100)
    ax.set_xlabel("交易損失（GDPデフレーターの交易条件要因）を100とした負担の割合（%、3年平均）")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4, frameon=False, fontsize=9.5)
    finish(ax)
    header(fig, "③ 交易損失は誰が負担したか：モデルでは、ほとんど賃金と利潤",
           "交易損失 ＝ 国内の物価の上昇 ＋ 賃金の減少 ＋ 利潤の減少 ＋ その他。金融政策を変えても内訳はほとんど変わらない。\n"
           "価格転嫁が強い版（参考）では、物価の負担が約13%→約32%に増え、賃金・利潤の負担が減る。")
    fig.text(0.02, 0.015, SRC + "\n参考は緩和継続のまま、輸入物価と消費デフレーターの式を2011〜2024年で推定し直した版。"
             "その他は固定資本減耗の増加で、わずかにマイナス。", fontsize=8.3, color=INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 0.84))
    fig.savefig(OUT / "tot_policy_explain3.png", dpi=160, facecolor="white")


def table(a: pd.DataFrame) -> None:
    avg = a.drop(columns=["年", "輸入価格の上昇率（%）"]).groupby("政策", sort=False).mean()
    keys = list(SHORT)
    rows = [("GDPデフレーター", "GDPデフレーター", "%"), ("　国内需要デフレーターの寄与", "寄与_国内需要デフレーター", "pt"),
            ("　交易条件の寄与", "寄与_交易条件", "pt"), ("名目雇用者報酬", "名目雇用者報酬", "%"),
            ("消費者物価（消費デフレーター）", "消費デフレーター", "%"), ("実質賃金", "実質賃金", "%"),
            ("実質GDP", "実質GDP", "%")]
    head = ["3年平均（基準解からの乖離）"] + [SHORT[k].replace("モデル: ", "").replace("参考: ", "参考:\n") for k in keys]
    widths = [4.4, 1.9, 1.9, 1.9, 1.9]
    W, rh = sum(widths), 0.6
    h = rh * (len(rows) + 1.4) + 1.5
    fig = plt.figure(figsize=(W, h))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(h, 0)
    ax.axis("off")
    ax.text(0.25, 0.5, "交易損失（輸入価格 +25%）を与えたときのモデルの結果", fontsize=15, color=INK, va="center")
    y = 1.0
    hh = rh * 1.4  # 見出しは2行
    ax.add_patch(plt.Rectangle((0, y), W, hh, color="#f3f2ee", lw=0))
    x = 0
    for txt, w in zip(head, widths):
        ax.text(x + (0.2 if x == 0 else w / 2), y + hh / 2, txt, fontsize=11, color=INK2,
                ha="left" if x == 0 else "center", va="center", linespacing=1.3)
        x += w
    y += hh
    for lab, col, unit in rows:
        x = 0
        ax.text(0.2, y + rh / 2, lab, fontsize=12, color=INK, va="center")
        x += widths[0]
        for k, w in zip(keys, widths[1:]):
            v = avg.loc[k, col]
            ax.text(x + w / 2, y + rh / 2, f"{v:+.2f}{'%' if unit == '%' else ''}", fontsize=12, color=INK,
                    ha="center", va="center")
            x += w
        ax.plot([0, W], [y + rh, y + rh], color=GRID, lw=1)
        y += rh
    ax.text(0.25, y + 0.35, "%は基準解（実績、2022〜24年）からの乖離、寄与は%pt。参考は輸入物価と消費デフレーターの式を推定し直した版。",
            fontsize=9.5, color=INK2, va="center")
    fig.savefig(OUT / "tot_policy_table.png", dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    style()
    a = pd.read_csv(OUT / "experiment_tot_policy.csv")
    p = pd.read_csv(OUT / "experiment_tot_policy_paths.csv")
    fig1(a)
    fig2(p)
    fig3(a)
    table(a)
    print(*(OUT / f"tot_policy_explain{i}.png" for i in (1, 2, 3)), OUT / "tot_policy_table.png", sep="\n")


if __name__ == "__main__":
    main()
