"""名目GDP水準目標（NGDPLT）を2021年から始めていたら（目標経路の起点＝2020Q4 の名目GDP）.

experiment_ngdplt.py（起点＝2019年の名目GDP、2020Q1 から実施）の変種。コロナ禍の落ち込みのあと、
2020Q4 の名目GDP を起点に年3%・4%・5%で伸ばす目標経路を置き、2021Q1 から同じルール
i = 0.71 i(-1) + 0.29 × (g + 名目GDPギャップ%)（下限なし）で政策金利を決める。
目標どおり（ギャップ0）でも、政策金利はルールの中立金利 g に向かって上がっていく。

計算は2通り（experiment_ngdplt.py と同じ）: 実績の経済のままルールに当てはめた金利（点線）と、
式113 を差し替えてモデルに組み込んで解いた値（実線）。誤差修正項は既定（3本のみ有効）。

出力: output/ngdplt_2021.png（4パネル）, output/experiment_ngdplt_2021.csv（年平均）
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from matplotlib import font_manager  # noqa: E402
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import ecm as E, model as M  # noqa: E401
from experiment_fx_passthrough import load
START, END, RHO = "2021Q1", "2024Q4", 0.71
data = load("model_data_v2024.csv")
data = data.reindex(pd.period_range(data.index.min(), pd.Period(END, "Q") + 2, freq="Q")); data["RTCI"] = data["RTCI"].ffill()
m = M.Model(); af = m.add_factors(data, START, END); base = m.solve(data, START, END, af)
idx = base.index; win = (idx >= pd.Period(START, "Q")) & (idx <= pd.Period(END, "Q"))
over, cols = E.frozen_overrides(m, base, win)
d0 = base.copy()
for k, v in cols.items(): d0[k] = v
lvl0 = base.loc["2020Q4", "GDPV"]
tq = np.array([p.year + (p.quarter - 0.5) / 4 for p in idx]); t0 = 2020 + 3.5 / 4
res = {}
for g in (3.0, 4.0, 5.0):
    tp = pd.Series(lvl0 * (1 + g / 100) ** (tq - t0), index=idx)
    d = d0.copy(); d["NGDP_TARGET"] = tp
    rule = (lambda v, _g=g: RHO * v("RCD", 1) + (1 - RHO) * (_g + 100 * np.log(v("GDPV") / v("NGDP_TARGET"))))
    s = m.solve(d, START, END, af, overrides={**over, "RCD": M.Eq(113, "RCD", "level", rule)})
    # 実績の経済のまま当てはめたルール
    gap = 100 * np.log(base["GDPV"] / tp); i, sh = base.loc["2020Q4", "RCD"], {}
    for p in idx[win]:
        i = RHO * i + (1 - RHO) * (g + gap[p]); sh[p] = i
    res[g] = dict(s=s, tp=tp, shadow=pd.Series(sh))
w = slice("2020Q1", END); q = [str(p) for p in idx if pd.Period("2020Q1", "Q") <= p <= pd.Period(END, "Q")]
x = np.arange(len(q)); x0 = q.index(START)
BLUE, ORANGE, AQUA, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e6e5e1"
C = {3.0: AQUA, 4.0: BLUE, 5.0: ORANGE}
FONT = next((f for f in ("IPAPGothic", "IPAGothic") if any(f == x.name for x in font_manager.fontManager.ttflist)), "sans-serif")
plt.rcParams.update({"font.family": FONT, "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.unicode_minus": False})
fig, axs = plt.subplots(2, 2, figsize=(14, 9.6)); fig.patch.set_facecolor("white")
yoy = lambda s: (s / s.shift(4) - 1) * 100
labels = {}
def lab(ax, v, c, fmt):
    labels.setdefault(ax, []).append((float(v), fmt.format(v), c))
# 名目GDP
ax = axs[0, 0]; ax.plot(x, base.loc[w, "GDPV"] / 1000, color=INK, lw=2.6, label="実績"); lab(ax, base.loc[END, "GDPV"] / 1000, INK, "{:.0f}")
for g, r in res.items():
    ax.plot(x[x0 - 1:], r["tp"].loc["2020Q4":END] / 1000, color=C[g], lw=1.3, ls=(0, (2, 2)), label=f"目標経路 年{g:g}%")
    ax.plot(x, r["s"].loc[w, "GDPV"] / 1000, color=C[g], lw=2.3, label=f"NGDPLT {g:g}%（モデルの解）")
    lab(ax, r["tp"].loc[END] / 1000, C[g], "{:.0f}"); lab(ax, r["s"].loc[END, "GDPV"] / 1000, C[g], "{:.0f}")
ax.set_title("名目GDP（兆円）", loc="left", fontsize=11.5, color=INK); ax.legend(loc="upper left", frameon=False, fontsize=8.8)
# 政策金利
ax = axs[0, 1]; ax.plot(x, base.loc[w, "RCD"], color=INK, lw=2.6, label="実績"); lab(ax, base.loc[END, "RCD"], INK, "{:+.1f}")
for g, r in res.items():
    ax.plot(x[x0:], r["shadow"].values, color=C[g], lw=1.5, ls=(0, (2, 2)), label=f"NGDPLT {g:g}% ルール（実績の経済のまま）")
    ax.plot(x, r["s"].loc[w, "RCD"], color=C[g], lw=2.3, label=f"NGDPLT {g:g}%（モデルの解）")
    lab(ax, r["shadow"].iloc[-1], C[g], "{:+.1f}"); lab(ax, r["s"].loc[END, "RCD"], C[g], "{:+.1f}")
ax.axhline(0, color=INK2, lw=0.9); ax.set_title("政策金利（短期金利、%）", loc="left", fontsize=11.5, color=INK); ax.set_ylim(-0.9, 6.2); ax.legend(loc="upper left", frameon=False, fontsize=8.3, ncol=2)
# 実質GDP
ax = axs[1, 0]; ax.plot(x, base.loc[w, "GDP"] / 1000, color=INK, lw=2.6, label="実績"); lab(ax, base.loc[END, "GDP"] / 1000, INK, "{:.0f}")
for g, r in res.items():
    ax.plot(x, r["s"].loc[w, "GDP"] / 1000, color=C[g], lw=2.3, label=f"NGDPLT {g:g}%"); lab(ax, r["s"].loc[END, "GDP"] / 1000, C[g], "{:.0f}")
ax.set_title("実質GDP（兆円、2020年基準）", loc="left", fontsize=11.5, color=INK); ax.legend(loc="lower right", frameon=False)
# 消費者物価
ax = axs[1, 1]; ax.plot(x, yoy(base["PCP"]).loc[w], color=INK, lw=2.6, label="実績"); lab(ax, yoy(base["PCP"]).loc[END], INK, "{:.1f}%")
for g, r in res.items():
    y = yoy(r["s"]["PCP"]).loc[w]; ax.plot(x, y, color=C[g], lw=2.3, label=f"NGDPLT {g:g}%"); lab(ax, y.iloc[-1], C[g], "{:.1f}%")
ax.axhline(0, color=INK2, lw=0.9); ax.axhline(2, color=INK2, lw=0.8, ls=(0, (4, 3)))
ax.set_title("消費者物価（消費デフレーター前年比、%）", loc="left", fontsize=11.5, color=INK); ax.legend(loc="upper left", frameon=False)
for ax in axs.flat:
    ax.axvspan(-0.4, x0 - 0.5, color="#f6f5f2", zorder=0)
    ax.set_xticks(x[::4]); ax.set_xticklabels([f"{s[:4]}年" for s in q[::4]]); ax.set_xlim(-0.4, len(q) + 1.6)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
for ax, items in labels.items():
    lo, hi = ax.get_ylim(); gap = 0.045 * (hi - lo); items = sorted(items); prev = None
    for v, t, c in items:
        yy = v if prev is None else max(v, prev + gap); prev = yy
        ax.annotate(t, (x[-1], v), xytext=(x[-1] + 0.4, yy), textcoords="data", color=c, fontsize=9, va="center")
fig.suptitle("名目GDP水準目標を2021年から始めていたら（目標の起点＝2020年10-12月期の名目GDP）", x=0.01, ha="left", fontsize=14, color=INK, fontweight="bold", y=0.985)
fig.text(0.01, 0.948, "目標経路＝2020Q4 の名目GDP から年3%・4%・5%で伸ばした水準。ルール：i ＝ 0.71×前期の i ＋ 0.29×（g ＋ 名目GDPギャップ%）、2021Q1 から。金利の下限なし。灰色の帯は実施前。", fontsize=9.5, color=INK2)
fig.text(0.01, 0.012, "内閣府 短期日本経済マクロ計量モデル（2022年版、ESRI Research Note No.72）の Python 再現、2024年版データ。実線はルールをモデルに組み込んで解いた値。物価は家計消費デフレーター。", fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 0.94), h_pad=2.2, w_pad=2.2); fig.savefig(ROOT / "output" / "ngdplt_2021.png", dpi=150, facecolor="white")
rows = []
for g, r in res.items():
    s = r["s"].loc[START:END]
    yr = s.index.year
    rows.append(pd.DataFrame({"目標": f"{g:g}%", "年": sorted(set(yr)),
                              "名目GDP 目標経路（兆円）": (r["tp"].loc[START:END] / 1000).groupby(yr).mean().values,
                              "名目GDP（兆円）": (s.GDPV / 1000).groupby(yr).mean().values,
                              "政策金利 モデルの解（%）": s.RCD.groupby(yr).mean().values,
                              "政策金利 実績の経済のまま（%）": r["shadow"].groupby(r["shadow"].index.year).mean().values,
                              "実質GDP（兆円）": (s.GDP / 1000).groupby(yr).mean().values,
                              "消費デフレーター前年比（%）": yoy(r["s"]["PCP"]).loc[START:END].groupby(yr).mean().values}))
b = base.loc[START:END]
yr = b.index.year
rows.append(pd.DataFrame({"目標": "実績", "年": sorted(set(yr)), "名目GDP（兆円）": (b.GDPV / 1000).groupby(yr).mean().values,
                          "政策金利 モデルの解（%）": b.RCD.groupby(yr).mean().values, "実質GDP（兆円）": (b.GDP / 1000).groupby(yr).mean().values,
                          "消費デフレーター前年比（%）": yoy(base["PCP"]).loc[START:END].groupby(yr).mean().values}))
out = pd.concat(rows, ignore_index=True)
out.to_csv(ROOT / "output" / "experiment_ngdplt_2021.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
print(out.round(2).to_string(index=False))
