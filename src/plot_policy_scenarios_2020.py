"""2020Q4 からの大型経済対策シナリオ（消費税廃止・公共投資・現金給付）を、実績（現状維持）と比べて描く.

日経NEEDSモデルを用いた試算（小野盛司氏）として出回っている図と同じ4シナリオを、このモデル（ESRI 短期モデル
2022年版の再現）で解く。版は ESRI_VINTAGE=2024（2020年基準SNA、2010Q1〜2024Q4）。

- 現状維持: 実績値（標準解）。元の図の「政策無し」は NEEDS の見通しで、実績とは異なる。
- 消費税廃止: 2020Q4 から消費税率 RTCI を 0 にする（標準・軽減の区別がないモデルなので全品目）。
- 公共投資: 2020Q4 から名目の公的固定資本形成を年20兆円増やす（論文シナリオ(3)と同じ与え方）。
- 現金給付: 2020Q4 から1人年80万円（総人口 × 80万円 ≈ 年100兆円）を、個人所得税の減税として与える。
いずれも予告なしに 2020Q4 から恒久的に実施（解く期間を 2020Q4 から始め、実施前の駆け込み・買い控えは出さない）。
誤差修正項は既定（消費・個人企業所得・消費デフレーターの3本のみ有効）。

元の図の「実質国民所得」はこのモデルに無い（海外からの純所得を持たない）ため、水準の近い実質GDP（2020年基準、兆円・季調年率）で描く。

出力: output/policy_scenarios_2020.csv, output/policy_scenarios_2020.png（日本語フォント IPAPGothic が必要）
"""
import os
import sys
from pathlib import Path

os.environ["ESRI_VINTAGE"] = "2024"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import ecm as E  # noqa: E402
import model as M  # noqa: E402
import vintage as VT  # noqa: E402

START, IMPL, END = "2020Q1", "2020Q4", "2023Q1"
POP = 1.2615e8          # 総人口（2020年10月1日 国勢調査 1億2,615万人）
PER_HEAD = 800_000      # 円/人/年
BENEFIT = POP * PER_HEAD / 1e9   # 10億円/年（≈ 100.9兆円）
PUBLIC_INV = 20_000.0   # 10億円/年（20兆円）


def main() -> None:
    data = pd.read_csv(VT.processed("model_data.csv"), index_col="period")
    data.index = pd.PeriodIndex(data.index, freq="Q")
    m = M.Model()
    af = m.add_factors(data, IMPL, END)
    base = m.solve(data, IMPL, END, af)
    idx = base.index
    mask = (idx >= pd.Period(IMPL, "Q")) & (idx <= pd.Period(END, "Q"))
    over, cols = E.frozen_overrides(m, base, mask)
    d0 = base.copy()
    for k, v in cols.items():
        d0[k] = v
    on = pd.Series((idx >= pd.Period(IMPL, "Q")).astype(float), index=idx)

    def run(data_over=None, **kw):
        d = d0.copy()
        for k, v in (data_over or {}).items():
            d[k] = v
        kw["overrides"] = {**over, **kw.get("overrides", {})}
        return m.solve(d, IMPL, END, af, **kw)

    runs = {
        # 消費税率 0（対数の課税ベースが 0 にならないよう 1e-9 とする。税収はほぼ 0 になる）
        "ctax": run({"RTCI": base["RTCI"] * (1 - on) + 1e-9 * on}),
        "pubinv": run(fixed={"IGV": base["IGV"] + PUBLIC_INV * on},
                      overrides={"IGV": M.Eq(19, "IG", "level", lambda v: v("IGV") / v("PIG"))}),
        "benefit": run(shocks={"TYPV": -BENEFIT * on}),
    }
    win = slice(START, END)
    out = pd.DataFrame({"base": base.loc[win, "GDP"] / 1000, **{k: r.loc[win, "GDP"] / 1000 for k, r in runs.items()}})
    for k, r in runs.items():
        out[f"{k}_BGVATGDPV"] = (r.loc[win, "BGVATGDPV"] - base.loc[win, "BGVATGDPV"])
        out[f"{k}_PCP"] = (r.loc[win, "PCP"] / base.loc[win, "PCP"] - 1) * 100
    out.index = out.index.astype(str)
    out.to_csv(ROOT / "output" / "policy_scenarios_2020.csv", float_format="%.6f")
    pd.set_option("display.width", 200)
    print(out.round(2).to_string())
    plot(out)


def plot(d: pd.DataFrame) -> None:
    BLUE, ORANGE, AQUA, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e6e5e1"
    plt.rcParams.update({"font.family": "IPAPGothic", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.unicode_minus": False})
    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    fig.patch.set_facecolor("white")
    x = range(len(d))
    impl = list(d.index).index(IMPL)
    series = [("ctax", "消費税廃止", BLUE, "-", "o"), ("pubinv", "公共投資（名目で年20兆円増加）", ORANGE, (0, (5, 2)), "s"),
              ("benefit", "現金給付（1人年80万円、年約101兆円）", AQUA, (0, (1, 1.3)), "^"),
              ("base", "現状維持（実績）", INK2, "-", "x")]
    for key, lab, c, ls, mk in series:
        ax.plot(x, d[key], color=c, lw=2.2, ls=ls, marker=mk, ms=5, label=lab, zorder=3)
    ends = sorted(((d[k].iloc[-1], k) for k, *_ in series))
    for i, (yv, k) in enumerate(ends):
        ax.annotate(f"{yv:.1f}", (len(d) - 1, yv), xytext=(8, (i - 1.5) * 4), textcoords="offset points",
                    color=INK, fontsize=9.5, va="center")
    ax.annotate(f"{d['base'].iloc[impl - 1]:.1f}", (impl - 1, d["base"].iloc[impl - 1]), xytext=(-10, 10),
                textcoords="offset points", color=INK, fontsize=9.5, ha="right")
    ax.axvline(impl - 0.5, color=INK2, lw=0.8, ls=":")
    ax.text(impl - 0.4, ax.get_ylim()[0] + 3, f"{IMPL}〜 実施（予告なし・恒久）", color=INK2, fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(list(d.index))
    ax.set_xlim(-0.4, len(d) - 0.2 + 0.8)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    fig.suptitle("実質GDP（兆円、季節調整済み年率、2020年基準）：大型経済対策の試算", x=0.01, ha="left",
                 fontsize=14, color=INK, fontweight="bold", y=0.985)
    fig.text(0.01, 0.935, "内閣府 短期日本経済マクロ計量モデル（2022年版、ESRI Research Note No.72）の Python 再現で計算。"
             "2024年版データ、基準解＝実績（現状維持）。", fontsize=9, color=INK2)
    last = d.iloc[-1]
    note = ("注: 日経NEEDSモデルによる試算（小野盛司氏）の図と同じ4シナリオを、別のモデルで解いたもの。元の図の「実質国民所得」はこのモデルに無いため、水準の近い実質GDPで描いた。"
            "元の図の「政策無し」は見通し、ここでは実績。\n"
            "消費税廃止は全品目の税率を0に（モデルは標準・軽減の区別なし）。現金給付は個人所得税の減税として与える（非課税の給付と同じ）。誤差修正項は既定（3本のみ有効）。"
            "大きなショックのため、推定期間の範囲を超えた外挿を含む。\n"
            f"2023Q1 の財政収支/名目GDP（基準解との差）: 消費税廃止 {last['ctax_BGVATGDPV']:+.1f}%pt、公共投資 {last['pubinv_BGVATGDPV']:+.1f}%pt、"
            f"現金給付 {last['benefit_BGVATGDPV']:+.1f}%pt。消費者物価（消費デフレーター）: 消費税廃止 {last['ctax_PCP']:+.1f}%、"
            f"公共投資 {last['pubinv_PCP']:+.1f}%、現金給付 {last['benefit_PCP']:+.1f}%。")
    fig.text(0.01, 0.012, note, fontsize=8, color=INK2, linespacing=1.55, va="bottom")
    fig.tight_layout(rect=(0, 0.12, 1, 0.92))
    fig.savefig(ROOT / "output" / "policy_scenarios_2020.png", dpi=160, facecolor="white")
    print(ROOT / "output" / "policy_scenarios_2020.png")


if __name__ == "__main__":
    main()
