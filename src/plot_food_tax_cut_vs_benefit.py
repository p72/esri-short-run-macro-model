"""食料品の消費税率 8%→1% vs 同額の給付金: 実質GDPと財政収支/GDP の3年間の四半期経路を計算して描く.

版は ESRI_VINTAGE=2024 に固定する（標準10%・軽減8%の複数税率が基準解に入っている 2022Q1〜2024Q4 を使う）。

規模（事前の税収減）
  ESRI 2024年度年次推計「家計の目的別最終消費支出の構成（名目）」の 1.食料・非アルコール飲料 F（税込み）から
  減収額 = F × (0.08 − 0.01) / 1.08。暦年の F / 国内家計最終消費支出 の比を各四半期の CPV に掛けて四半期化する。
  外食（10%）と酒類は対象外。テイクアウト・出前は SNA では外食・宿泊に入るため含めない（その分は過小）。
  --loss-tn 4.4 のように与えると、事前の減収額の3年平均がその額（兆円/年）になるよう税率の引下げ幅を決める
  （報道・木内氏コラムの 4.4兆円 = 食料品ゼロ税率の約5兆円 × 7/8。軽減税率の対象全体・2026年ごろの物価水準に相当）。

モデルでの与え方
  モデルの消費税は標準税率 RTCI 1本なので、消費だけに効く実効税率を2本追加して差し替え式で解く。
  - RTCICP: 消費関数(3)の税率リード・ラグ項と消費デフレーター(73)に使う。引下げ幅は、食料品価格に全額転嫁されたときの
    消費デフレーターの低下（事前の減収額 ÷ 名目消費。SNA 基準では食料品の比率 × 7/108）と、式73の低下が2022年に一致するよう決める。
  - RTCIREV: 消費税収(134)の消費部分に使う。引下げ幅は、2022年の事前の税収減（消費以外は基準解のまま）が上の減収額と一致するよう決める。
  （1本で両方を合わせると、式134の消費部分の税収が実際より小さいため物価の低下が約2割過大になる）
  住宅投資(5)の税率リード項、設備・住宅・政府支出の税率は動かさない。どちらも 2022Q1 から恒久的に与える。
  給付金は各四半期の消費税の事前の減収額と同額の個人所得税減税として与える。

入力: data/processed/model_data_v2024.csv, data/raw/vintage2024/2024s12n_jp.xlsx
出力: output/food_tax_cut_vs_benefit{,_4.4tn}.csv, .png（--loss-tn 指定時は _{値}tn が付く。日本語フォント IPAPGothic が必要）
"""
import argparse
import os
import sys
from pathlib import Path

os.environ["ESRI_VINTAGE"] = "2024"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ecm as E
import model as M
import simulate as S
import vintage as VT

IMPL, END, SOLVE_START, WIN0 = "2022Q1", "2024Q4", "2021Q3", "2021Q3"
S.START, S.END, S.SOLVE_START = IMPL, END, SOLVE_START
R_FOOD_OLD, R_FOOD_NEW = 0.08, 0.01
ap = argparse.ArgumentParser()
ap.add_argument("--loss-tn", type=float, default=None, help="事前の減収額（兆円/年）。省略時は SNA の食料支出×7/108")
ARGS = ap.parse_args()
SUFFIX = "" if ARGS.loss_tn is None else f"_{ARGS.loss_tn:g}tn"

# ---- 食料品支出（ESRI 家計の目的別最終消費支出、名目・暦年）
tab = pd.read_excel(VT.V["dir"] / "2024s12n_jp.xlsx", sheet_name="暦年", header=None)
years = tab.iloc[6, 1:].astype(int).tolist()
row = lambda key: pd.Series(tab[tab[0].astype(str).str.contains(key)].iloc[0, 1:].astype(float).values, index=years)
food, hhc = row("食料・非アルコール"), row("国内家計最終消費支出")
food_share = food / hhc

# ---- データと基準解
data = pd.read_csv(VT.processed("model_data.csv"), index_col="period")
data.index = pd.PeriodIndex(data.index, freq="Q")
need = pd.Period(END, "Q") + 2
data = data.reindex(pd.period_range(data.index.min(), need, freq="Q"))
data["RTCI"] = data["RTCI"].ffill()
data = data.copy().assign(RTCICP=data["RTCI"], RTCIREV=data["RTCI"])

m = M.Model()
af = m.add_factors(data, SOLVE_START, END)
base = m.solve(data, SOLVE_START, END, af)
over, cols = E.frozen_overrides(m, base, S.solve_mask(base.index))
d0 = base.copy()
for k, v in cols.items():
    d0[k] = v
idx = base.index
after = pd.Series((idx >= pd.Period(IMPL, "Q")).astype(float), index=idx)

# ---- 消費だけに効く税率 RTCICP を使う差し替え式
eqs = {e.name: e for e in m.eqs}
assert "TCIV" not in over and "PCP" not in over


def cp_rate(v):
    return lambda n, k=0: v("RTCICP" if n == "RTCI" else n, k)


for name in ("CP", "PCP"):
    e = over.get(name, eqs[name])
    over[name] = M.Eq(e.no, e.name, e.kind, (lambda v, _e=e: _e.rhs(cp_rate(v))), e.lhs, e.inv)


def tci_base_food(v):
    r, rc, p = v("RTCI"), v("RTCIREV"), v("PRTCP")
    return M.tci_base(v) - r / (1 + r * p) * v("CPV") + rc / (1 + rc * p) * v("CPV")


e134 = eqs["TCIV"]
over["TCIV"] = M.Eq(e134.no, "TCIV", e134.kind, lambda v: (
    0.915210 * M.ln(tci_base_food(v))
    - 0.011036 * v("DTCIC2") * v("DTCIC2") * M.ln(tci_base_food(v))
    + 0.191987 * v("D972C")), e134.lhs, e134.inv)

# ---- 規模: 事前の減収額（年率、10億円）と、それに一致する RTCICP の引下げ幅
q = base.loc[IMPL:END]
target = pd.Series([food_share[p.year] * q.at[p, "CPV"] * (R_FOOD_OLD - R_FOOD_NEW) / (1 + R_FOOD_OLD) for p in q.index],
                   index=q.index)
if ARGS.loss_tn is not None:
    target = pd.Series(ARGS.loss_tn * 1000.0, index=q.index)  # 税率一定なので各期の減収は名目消費とともに変わる


def exante_loss(delta):
    """RTCICP を delta 下げたときの式134の事前の減収額（CPV など他は基準解のまま）."""
    r, p, cpv = q["RTCI"], q["PRTCP"], q["CPV"]
    tb = (q["TCIV"] / np.exp(af.loc[q.index, "TCIV"] + 0.191987 * q["D972C"])) ** (1 / 0.915210)
    rc = r - delta
    tb_new = tb - r / (1 + r * p) * cpv + rc / (1 + rc * p) * cpv
    return q["TCIV"] * (1 - (tb_new / tb) ** 0.915210)


y1 = slice(IMPL, "2022Q4") if ARGS.loss_tn is None else slice(IMPL, END)  # --loss-tn は3年平均で合わせる
lo, hi = 0.0, 0.10
for _ in range(60):
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if exante_loss(mid).loc[y1].sum() < target.loc[y1].sum() else (lo, mid)
delta = (lo + hi) / 2
loss = exante_loss(delta)
# 物価: 食料品価格に全額転嫁 → 消費デフレーター低下率 = 食料品比率 × 7/108。式73 で同じ低下になる RTCICP の引下げ幅
q22 = q.loc[y1]  # 物価の合わせ込みも同じ期間
dp = (target.loc[y1] / q22["CPV"]).mean()
r0, p0 = q22["RTCI"].mean(), q22["PRTCP"].mean()
delta_p = dp * (1 + r0 * p0) / p0
amount = loss.reindex(idx).fillna(0.0) * after


def run(data_over=None, shocks=None):
    d = d0.copy()
    for k, v in (data_over or {}).items():
        d[k] = v
    return m.solve(d, SOLVE_START, END, af, overrides=over, shocks=shocks)


chk = run()
assert np.allclose(chk.loc[WIN0:END, "GDP"], base.loc[WIN0:END, "GDP"], rtol=1e-8), "差し替え式で基準解が再現されない"
runs = {"food": run(data_over={"RTCICP": d0["RTCICP"] - delta_p * after, "RTCIREV": d0["RTCIREV"] - delta * after}),
        "benefit": run(shocks={"TYPV": -amount})}
win = slice(WIN0, END)
out = {}
for k, sh in runs.items():
    out[(k, "GDP")] = (sh.loc[win, "GDP"] / base.loc[win, "GDP"] - 1) * 100
    out[(k, "BGV")] = sh.loc[win, "BGVATGDPV"] - base.loc[win, "BGVATGDPV"]
    out[(k, "CP")] = (sh.loc[win, "CP"] / base.loc[win, "CP"] - 1) * 100
    out[(k, "PCP")] = (sh.loc[win, "PCP"] / base.loc[win, "PCP"] - 1) * 100
out[("scale", "loss_bn")] = amount.loc[win]
out[("scale", "target_bn")] = target.reindex(base.loc[win].index)
out[("scale", "loss_pct_gdp")] = amount.loc[win] / base.loc[win, "GDPV"] * 100
df = pd.DataFrame(out)
df.index = df.index.astype(str)
df.to_csv(ROOT / f"output/food_tax_cut_vs_benefit{SUFFIX}.csv", float_format="%.8f")

ann = df.loc[IMPL:END].copy()
ann.index = pd.PeriodIndex(ann.index, freq="Q").year
ann = ann.groupby(level=0).mean()
print(f"食料・非アルコール飲料（暦年、兆円）: " + ", ".join(f"{y} {food[y] / 1000:.2f}" for y in (2022, 2023, 2024)))
print(f"実効税率の引下げ幅: 物価用 RTCICP {delta_p * 100:.3f}%pt（消費デフレーター −{dp * 100:.3f}%）、税収用 RTCIREV {delta * 100:.3f}%pt")
print("年平均:\n", ann.round(3).to_string())

# ---- 作図
d = df
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.family": "IPAPGothic", "font.size": 9.5, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.unicode_minus": False})
fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2))
fig.patch.set_facecolor("white")
x = list(range(len(d)))
labels = list(d.index)
impl = labels.index(IMPL)
loss_tn = ann[("scale", "loss_bn")] / 1000
loss_pct = ann[("scale", "loss_pct_gdp")]
for ax, (var, title, unit) in zip(axes, [("GDP", "実質GDP", "基準解からの乖離、%"),
                                         ("BGV", "財政収支／名目GDP", "基準解からの乖離、%ポイント")]):
    ax.axhline(0, color=GRID, lw=1)
    ax.axvspan(-0.5, impl - 0.5, color="#f3f2ee", zorder=0)
    ax.axvline(impl - 0.5, color=INK2, lw=0.8, ls=":")
    ax.plot(x, d[("food", var)], color=BLUE, lw=2.2, marker="o", ms=3.5, label="食料品の消費税 8%→1%（恒久）")
    ax.plot(x, d[("benefit", var)], color=ORANGE, lw=2.2, ls=(0, (4, 2)), marker="o", ms=3.5,
            label="同額の給付金（所得減税として）")
    ya, yb = d[("food", var)].iloc[-1], d[("benefit", var)].iloc[-1]
    off = 6 if ya >= yb else -6
    ax.annotate(f"{ya:+.2f}", (x[-1], ya), xytext=(6, off), textcoords="offset points", color=BLUE, fontsize=9, va="center")
    ax.annotate(f"{yb:+.2f}", (x[-1], yb), xytext=(6, -off), textcoords="offset points", color=ORANGE, fontsize=9, va="center")
    ax.set_title(f"{title}（{unit}）", loc="left", fontsize=11, color=INK, pad=8)
    ticks = [i for i, l in enumerate(labels) if l.endswith("Q1")] + [len(labels) - 1]
    ax.set_xticks(ticks)
    ax.set_xticklabels([labels[i] for i in ticks])
    ax.set_xlim(-0.5, len(labels) + 0.8)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
ax = axes[0]
ymin, ymax = ax.get_ylim()
ax.text(impl - 0.3, ymin + 0.03 * (ymax - ymin), f"実施（{IMPL}〜、恒久）", color=INK2, fontsize=8.5, va="bottom", ha="left")
ax.annotate("実施前の買い控え\n（消費関数のリード項）", (1, d[("food", "GDP")].iloc[1]), xytext=(2.6, ymin + 0.2 * (ymax - ymin)),
            textcoords="data", color=INK2, fontsize=8.5, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.annotate("実施直後の反動増", (impl, d[("food", "GDP")].iloc[impl]), xytext=(impl + 1.2, d[("food", "GDP")].iloc[impl] - 0.02),
            textcoords="data", color=INK2, fontsize=8.5, va="center", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.01, 0.905), ncol=2, frameon=False, fontsize=9.5)
fig.suptitle("食料品の消費税 8%→1% vs 同額の給付金" + (f"（減収 {ARGS.loss_tn:g}兆円/年）" if ARGS.loss_tn else "") + "：実質GDPと財政収支の3年間の経路", x=0.01, ha="left",
             fontsize=13.5, color=INK, fontweight="bold", y=0.985)
fig.text(0.01, 0.935, "内閣府 短期日本経済マクロ計量モデル（2022年版、ESRI Research Note No.72）の Python 再現で計算。"
         f"2024年版データ（2020年基準SNA）、基準解＝{IMPL}〜{END} の実績、ショックは恒久。", fontsize=9, color=INK2)
g = lambda k, var: "/".join(f"{v:+.2f}" for v in ann[(k, var)])
scale = (f"規模: 事前の減収額＝食料・非アルコール飲料の家計消費（ESRI 年次推計、税込み）×7/108。年平均 {'/'.join(f'{v:.2f}' for v in loss_tn)} 兆円"
         f"（名目GDP比 {'/'.join(f'{v:.2f}' for v in loss_pct)}%）。外食・酒類は対象外、テイクアウトは含めない。給付金は各期の消費税の事前減収と同額。\n")
if ARGS.loss_tn is not None:
    scale = (f"規模: 事前の減収額の3年平均を {ARGS.loss_tn:g}兆円/年とした（報道・試算の値＝食料品ゼロ税率の約5兆円×7/8）。年平均 {'/'.join(f'{v:.2f}' for v in loss_tn)} 兆円、"
             f"基準解の名目GDP比 {'/'.join(f'{v:.2f}' for v in loss_pct)}%。給付金は同額。\n")
note = (scale +

        f"モデルでは消費だけに効く実効税率を、物価（食料品に全額転嫁、消費デフレーター −{dp * 100:.2f}%）で {delta_p * 100:.2f}%pt、税収で {delta * 100:.2f}%pt 下げる"
        "（消費関数・消費デフレーター・消費税収の式のみ差し替え）。住宅・設備・政府支出の税率は不変。\n"
        f"年平均（1/2/3年目）の実質GDP: 食料品減税 {g('food', 'GDP')}、給付金 {g('benefit', 'GDP')}。"
        f"財政収支/GDP: 食料品減税 {g('food', 'BGV')}、給付金 {g('benefit', 'BGV')}。誤差修正項は消費・個人企業所得・消費デフレーターの3本のみ有効。")
fig.text(0.01, 0.012, note, fontsize=8, color=INK2, linespacing=1.55, va="bottom")
fig.tight_layout(rect=(0, 0.14, 1, 0.86), w_pad=2.5)
fig.savefig(ROOT / f"output/food_tax_cut_vs_benefit{SUFFIX}.png", dpi=160, facecolor="white")
print(ROOT / f"output/food_tax_cut_vs_benefit{SUFFIX}.png")
