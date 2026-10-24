"""税収中立の税の組み替え: A 消費税増税＋所得税減税 と B 所得税増税＋消費税減税.

どちらも、事前の税収の増減がそれぞれ名目GDPの1%（足して0）になるように組み合わせ、実施から恒久的に続ける。
- 消費税: 標準税率 RTCI を変える。幅は、初年の税収ベースで事前の消費税収の変化が名目GDPの1%になるように決める
  （plot_tax_cut_vs_benefit.py と同じ方法。増税と減税で幅はわずかに違う）。
- 所得税: 個人所得税 TYPV に名目GDPの1%のショックを与える（論文シナリオ(4)と同じ）。
誤差修正項は既定（消費・個人企業所得・消費デフレーターの3本のみ有効、他は標準解で固定）。
消費税率の変更は予告されるので、消費・住宅の式のリード項で、実施前の2四半期に駆け込み（増税）や買い控え（減税）が出る。

版は2つ。
- 2021年版（論文と同じ版）: 実施 2018Q1〜2020Q4（解く期間 2017Q3〜）
- 2024年版: 実施 2022Q1〜2024Q4（解く期間 2021Q3〜）

出力: output/experiment_tax_swap.csv（四半期）, output/experiment_tax_swap_annual.csv（年平均）
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
import simulate as S  # noqa: E402

OUT = ROOT / "output"
VINTAGES = {  # 版: (データ, 解き始め, 実施開始, 終わり)
    "2021年版（2018〜20年）": ("model_data.csv", "2017Q3", "2018Q1", "2020Q4"),
    "2024年版（2022〜24年）": ("model_data_v2024.csv", "2021Q3", "2022Q1", "2024Q4"),
}
CASES = {"A 消費税増税＋所得税減税": (+1, -1), "B 所得税増税＋消費税減税": (-1, +1),
         "参考: 消費税減税のみ": (-1, 0), "参考: 所得税減税のみ": (0, -1)}


def rate_change(base: pd.DataFrame, start: str, sign: int) -> float:
    """事前の消費税収の変化が名目GDPの1%になる税率の変化幅（sign=+1 増税、-1 減税）."""
    b1 = base.loc[start:f"{start[:4]}Q4"]
    tciv, gdpv, prt, r0 = b1.TCIV.mean(), b1.GDPV.mean(), b1.PRTCP.iloc[0], b1.RTCI.mean()
    f = lambda r: r / (1 + r * prt)  # noqa: E731  課税ベースに掛かる実効的な税率
    target = (1 + sign * 0.01 * gdpv / tciv) ** (1 / 0.915) * f(r0)
    grid = np.arange(0.005, 0.20, 0.00001)
    r_new = grid[np.argmin(np.abs(f(grid) - target))]
    return r_new - r0


def measures(s: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    pct = lambda v: (s[v] / b[v] - 1) * 100  # noqa: E731
    return pd.DataFrame({
        "実質GDP（%）": pct("GDP"), "実質消費（%）": pct("CP"), "設備投資（%）": pct("IFP"), "住宅投資（%）": pct("IHP"),
        "消費者物価（消費デフレーター、%）": pct("PCP"), "実質賃金（%）": ((s.W / s.PCP) / (b.W / b.PCP) - 1) * 100,
        "実質可処分所得（%）": ((s.YDV / s.PCP) / (b.YDV / b.PCP) - 1) * 100,
        "失業率（%pt）": s.UR - b.UR,
        "財政収支/名目GDP（%pt）": s.BGVATGDPV - b.BGVATGDPV,
        "消費税収/名目GDP（%pt）": (s.TCIV / s.GDPV - b.TCIV / b.GDPV) * 100,
        "所得税収/名目GDP（%pt）": (s.TYPV / s.GDPV - b.TYPV / b.GDPV) * 100,
    })


def main() -> None:
    q_all, a_all = [], []
    for label, (fname, ss, start, end) in VINTAGES.items():
        data = pd.read_csv(ROOT / "data" / "processed" / fname, index_col="period")
        data.index = pd.PeriodIndex(data.index, freq="Q")
        need = pd.Period(end, "Q") + 2
        if data.index.max() < need:
            data = data.reindex(pd.period_range(data.index.min(), need, freq="Q"))
            data["RTCI"] = data["RTCI"].ffill()
        S.SOLVE_START, S.START, S.END = ss, start, end
        m = M.Model()
        af = m.add_factors(data, ss, end)
        base = m.solve(data, ss, end, af)
        over, cols = E.frozen_overrides(m, base, S.solve_mask(base.index))
        d0 = base.copy()
        for k, v in cols.items():
            d0[k] = v
        idx = base.index
        on = pd.Series(S.sim_mask(idx).astype(float), index=idx)
        after = pd.Series((idx >= pd.Period(start, "Q")).astype(float), index=idx)
        dx = {+1: rate_change(base, start, +1), -1: rate_change(base, start, -1)}
        for case, (c_sign, i_sign) in CASES.items():
            d = d0.copy()
            if c_sign:
                d["RTCI"] = base["RTCI"] + dx[c_sign] * after
            shocks = {"TYPV": i_sign * 0.01 * base["GDPV"] * on} if i_sign else None
            s = m.solve(d, ss, end, af, overrides=over, shocks=shocks)
            r = measures(s.loc[ss:end], base.loc[ss:end])
            r.insert(0, "四半期", r.index.astype(str))
            r.insert(0, "政策", case)
            r.insert(0, "版", label)
            r.insert(3, "消費税率の変化（%pt）", dx[c_sign] * 100 if c_sign else 0.0)
            q_all.append(r)
            a = r.loc[start:end].drop(columns=["版", "政策", "四半期"]).groupby(r.loc[start:end].index.year).mean()
            a.insert(0, "年", [f"{i}年目（{y}）" for i, y in enumerate(a.index, 1)])
            a.insert(0, "政策", case)
            a.insert(0, "版", label)
            a_all.append(a)
    q = pd.concat(q_all, ignore_index=True)
    a = pd.concat(a_all, ignore_index=True)
    q.to_csv(OUT / "experiment_tax_swap.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    a.to_csv(OUT / "experiment_tax_swap_annual.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 30)
    print(a.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
