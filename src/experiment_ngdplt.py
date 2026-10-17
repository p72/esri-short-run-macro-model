"""2020年から名目GDP水準目標（NGDPLT、年4%・5%）を採っていたら、政策金利はどう推移したか.

名目GDP水準目標は、名目GDP の「水準」を年 g% で伸びる目標経路に乗せる金融政策。
原油高のような供給ショックでは、物価が上がる分だけ実質の落ち込みを許すので、インフレ目標のように
物価に反応して利上げし、実質産出をさらに犠牲にすることを避けられる、とされる。

データは2024年版（2020年基準SNA、2024Q4まで）。目標経路は、2019年の名目GDP の平均（2019年央）を起点に年 g% で伸ばす。

1. ルールが示す政策金利（実績の経済のまま、フィードバックなし）
   モデルのテイラー・ルール（式114）と同じ形（前期の金利の重み 0.71）に、目標を当てはめる。成長率と物価上昇率は前年比。
   - NGDPLT:            i = 0.71 i(-1) + 0.29 × (g + φ × 名目GDPギャップ)
     名目GDPギャップ = 100 × ln(名目GDP / 目標経路)。中立金利は g（＝実質の潜在成長率 + 目標経路と整合的な物価上昇率）
     φ = 1.0（参考に 0.5 も）
   - インフレ目標（比較）: i = 0.71 i(-1) + 0.29 × (潜在成長率 + 2 + 1.68 × (物価上昇率 − 2) + 0.15 × GDPギャップ)
     物価は GDPデフレーター（モデルの式114と同じ）と消費デフレーター（消費者物価に近い）の2通り
   いずれも金利の下限は置かない（マイナス金利を含む「シャドー金利」）。
2. モデルに組み込んで解く（フィードバックあり）
   2020Q1 から、式113 の短期金利を NGDPLT のルール（下限なし）に差し替えて解き、名目GDP がどれだけ目標に近づくかを見る。
   長期金利は式116 で短期金利に連動する。誤差修正項は既定（消費・個人企業所得・消費デフレーターの3本のみ有効）。

出力: output/experiment_ngdplt.csv（四半期）, output/experiment_ngdplt_annual.csv（年平均）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import ecm as E  # noqa: E402
import model as M  # noqa: E402
from experiment_fx_passthrough import load  # noqa: E402

OUT = ROOT / "output"
FNAME, START, END = "model_data_v2024.csv", "2020Q1", "2024Q4"
RHO = 0.71
TARGETS = (4.0, 5.0)
PHIS = (1.0, 0.5)


def target_path(idx: pd.PeriodIndex, level2019: float, g: float) -> pd.Series:
    t = np.array([p.year + (p.quarter - 0.5) / 4 for p in idx])  # 四半期の中央
    return pd.Series(level2019 * (1 + g / 100) ** (t - 2019.5), index=idx)


def rule_path(start_rate: float, target_rate: pd.Series) -> pd.Series:
    """i = RHO i(-1) + (1 − RHO) × target_rate を START から回す（下限なし）."""
    out, prev = {}, start_rate
    for p, x in target_rate.items():
        prev = RHO * prev + (1 - RHO) * x
        out[p] = prev
    return pd.Series(out)


def main() -> None:
    data = load(FNAME)
    need = pd.Period(END, "Q") + 2  # 消費・住宅の式が RTCI の2期先を見る
    data = data.reindex(pd.period_range(data.index.min(), need, freq="Q"))
    data["RTCI"] = data["RTCI"].ffill()
    m = M.Model()
    af = m.add_factors(data, START, END)
    base = m.solve(data, START, END, af)
    idx = base.index
    w = slice(START, END)
    lvl2019 = base.loc["2019Q1":"2019Q4", "GDPV"].mean()
    yoy = lambda s: (s / s.shift(4) - 1) * 100  # noqa: E731
    pot_g = yoy(base["GDPPOT"])
    i0 = base.loc[pd.Period(START, "Q") - 1, "RCD"]

    out = pd.DataFrame(index=idx[(idx >= pd.Period(START, "Q")) & (idx <= pd.Period(END, "Q"))])
    out["名目GDP 実績（兆円）"] = base.loc[w, "GDPV"] / 1000
    out["短期金利 実績（%）"] = base.loc[w, "RCD"]
    out["長期金利 実績（%）"] = base.loc[w, "RGB"]
    out["消費デフレーター前年比（%）"] = yoy(base["PCP"]).loc[w]
    out["GDPデフレーター前年比（%）"] = yoy(base["PGDPAT"]).loc[w]
    # インフレ目標（比較）のシャドー金利
    for name, p in (("GDPデフレーター", "PGDPAT"), ("消費デフレーター", "PCP")):
        tgt = pot_g + 2 + 1.68 * (yoy(base[p]) - 2) + 0.15 * base["GDPGAP"]
        out[f"インフレ目標（{name}）のシャドー金利（%）"] = rule_path(i0, tgt.loc[w])

    over, cols = E.frozen_overrides(m, base, (idx >= pd.Period(START, "Q")) & (idx <= pd.Period(END, "Q")))
    d0 = base.copy()
    for k, v in cols.items():
        d0[k] = v
    for g in TARGETS:
        tp = target_path(idx, lvl2019, g)
        out[f"名目GDP 目標{g:g}%（兆円）"] = tp.loc[w] / 1000
        out[f"名目GDPギャップ 目標{g:g}%（%）"] = 100 * np.log(base.loc[w, "GDPV"] / tp.loc[w])
        for phi in PHIS:
            gap = 100 * np.log(base["GDPV"] / tp)
            out[f"NGDPLT{g:g}% φ{phi:g} シャドー金利（%）"] = rule_path(i0, (g + phi * gap).loc[w])
        # 2. モデルに組み込んで解く（φ=1、下限なし）
        d = d0.copy()
        d["NGDP_TARGET"] = tp
        rule = (lambda v, _g=g: RHO * v("RCD", 1) + (1 - RHO) * (_g + 100 * np.log(v("GDPV") / v("NGDP_TARGET"))))
        ov = {**over, "RCD": M.Eq(113, "RCD", "level", rule)}
        try:
            s = m.solve(d, START, END, af, overrides=ov)
            out[f"モデル NGDPLT{g:g}% 短期金利（%）"] = s.loc[w, "RCD"]
            out[f"モデル NGDPLT{g:g}% 長期金利（%）"] = s.loc[w, "RGB"]
            out[f"モデル NGDPLT{g:g}% 名目GDP（兆円）"] = s.loc[w, "GDPV"] / 1000
            out[f"モデル NGDPLT{g:g}% 実質GDP（基準解比 %）"] = (s.loc[w, "GDP"] / base.loc[w, "GDP"] - 1) * 100
            out[f"モデル NGDPLT{g:g}% 消費デフレーター（基準解比 %）"] = (s.loc[w, "PCP"] / base.loc[w, "PCP"] - 1) * 100
        except Exception as err:  # 解けない場合は記録して続ける
            print(f"目標{g:g}%: モデルに組み込んだ解が求まらない: {err!r}")
    out.index = out.index.astype(str)
    out.to_csv(OUT / "experiment_ngdplt.csv", encoding="utf-8-sig", float_format="%.4f")
    ann = out.groupby(out.index.str[:4]).mean()
    ann.to_csv(OUT / "experiment_ngdplt_annual.csv", encoding="utf-8-sig", float_format="%.4f")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(ann.round(2).T.to_string())


if __name__ == "__main__":
    main()
