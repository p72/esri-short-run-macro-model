"""名目GDP1%規模の消費税減税で、法人税収などの税収はどうなるか（税目別の増減と、税収の自然増による財源の戻り）.

シナリオは plot_tax_cut_vs_benefit.py と同じ。
- 消費税減税: このモデルの税収ベース（初年、論文期間は税率8%）で事前の税収減が名目GDPの1%になる引下げ幅を恒久的に与える
- 比較: 同額の給付金（個人所得税を名目GDPの1%減税、論文シナリオ(4)）
誤差修正項は既定（消費・個人企業所得・消費デフレーターの3本のみ有効、他は標準解で固定）。

版は2つ。
- 2021年版（論文と同じ版）: 実施 2018Q1〜2020Q4（解く期間 2017Q3〜）
- 2024年版: 実施 2022Q1〜2024Q4（解く期間 2021Q3〜）。基準解の税率は標準10%・軽減8%の実効税率

法人所得税は式131 TYCV = ETT × 前期から4期分の企業所得 YCV の平均、ETT は式132（法定実効税率 TT と GDP ギャップ）。
税収の合計は式128 TAXV = TYPV + TYCV + ITAXV（ITAXV = 消費税 TCIV + 関税等 TCSTV + その他の間接税 OITAXV）。
財源の戻り = （税収の合計の変化 + 事前の減税額）/ 事前の減税額。事前の減税額は基準解の名目GDP の1%。

出力: output/experiment_ctax_revenue.csv（年平均）, output/experiment_ctax_revenue_paths.csv（四半期）
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
TAXES = [("消費税", lambda x: x.TCIV), ("その他の間接税", lambda x: x.ITAXV - x.TCIV),
         ("個人所得税", lambda x: x.TYPV), ("法人所得税", lambda x: x.TYCV), ("税収の合計", lambda x: x.TAXV)]


def solve_vintage(fname: str, solve_start: str, start: str, end: str) -> dict[str, pd.DataFrame]:
    data = pd.read_csv(ROOT / "data" / "processed" / fname, index_col="period")
    data.index = pd.PeriodIndex(data.index, freq="Q")
    need = pd.Period(end, "Q") + 2  # 消費・住宅の式が RTCI の2期先を見る
    if data.index.max() < need:
        data = data.reindex(pd.period_range(data.index.min(), need, freq="Q"))
        data["RTCI"] = data["RTCI"].ffill()
    S.SOLVE_START, S.START, S.END = solve_start, start, end
    m = M.Model()
    af = m.add_factors(data, solve_start, end)
    base = m.solve(data, solve_start, end, af)
    over, cols = E.frozen_overrides(m, base, S.solve_mask(base.index))
    d0 = base.copy()
    for k, v in cols.items():
        d0[k] = v
    idx = base.index
    on = pd.Series(S.sim_mask(idx).astype(float), index=idx)
    after = pd.Series((idx >= pd.Period(start, "Q")).astype(float), index=idx)

    # 事前の税収減が名目GDPの1%になる税率の引下げ幅（plot_tax_cut_vs_benefit.py と同じ方法。初年の税収ベースで決める）
    b1 = base.loc[start:f"{start[:4]}Q4"]
    tciv, gdpv, prt, r0 = b1.TCIV.mean(), b1.GDPV.mean(), b1.PRTCP.iloc[0], b1.RTCI.mean()
    f = lambda r: r / (1 + r * prt)  # noqa: E731
    target = (1 - 0.01 * gdpv / tciv) ** (1 / 0.915) * f(r0)
    x_eq = r0 - max(r for r in np.arange(0.01, r0, 0.00001) if f(r) <= target)

    def run(data_over=None, shocks=None):
        d = d0.copy()
        for k, v in (data_over or {}).items():
            d[k] = v
        return m.solve(d, solve_start, end, af, overrides=over, shocks=shocks)

    return {"base": base, "消費税減税": run(data_over={"RTCI": base["RTCI"] - x_eq * after}),
            "給付金（所得減税）": run(shocks={"TYPV": -0.01 * base["GDPV"] * on}), "x_eq": x_eq}


def measures(s: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    cost = 0.01 * b.GDPV  # 事前の減税額（年率、10億円）
    out = {}
    for name, f in TAXES:
        out[f"{name}（兆円）"] = (f(s) - f(b)) / 1000
        out[f"{name}（名目GDP比 %pt）"] = (f(s) - f(b)) / b.GDPV * 100
    out["法人所得税（基準解比 %）"] = (s.TYCV / b.TYCV - 1) * 100
    out["企業所得 YCV（基準解比 %）"] = (s.YCV / b.YCV - 1) * 100
    out["法人実効税率 ETT（%pt）"] = (s.ETT - b.ETT) * 100
    out["名目GDP（%）"] = (s.GDPV / b.GDPV - 1) * 100
    out["実質GDP（%）"] = (s.GDP / b.GDP - 1) * 100
    out["事前の減税額（兆円）"] = cost / 1000
    out["財源の戻り（%）"] = ((s.TAXV - b.TAXV) + cost) / cost * 100
    out["うち法人所得税の増収（%）"] = (s.TYCV - b.TYCV) / cost * 100
    return pd.DataFrame(out)


def main() -> None:
    annual, paths = [], []
    for label, (fname, ss, start, end) in VINTAGES.items():
        r = solve_vintage(fname, ss, start, end)
        print(f"{label}: 消費税率の引下げ幅 {r['x_eq'] * 100:.2f}%pt")
        for case in ("消費税減税", "給付金（所得減税）"):
            mm = measures(r[case].loc[start:end], r["base"].loc[start:end])
            a = mm.groupby(mm.index.year).mean()
            a.insert(0, "年", [f"{i}年目（{y}）" for i, y in enumerate(a.index, 1)])
            a.insert(0, "政策", case)
            a.insert(0, "版", label)
            a.insert(3, "消費税率の引下げ幅（%pt）", r["x_eq"] * 100 if case == "消費税減税" else np.nan)
            annual.append(a)
            paths.append(mm.assign(版=label, 政策=case, 四半期=mm.index.astype(str)))
    res = pd.concat(annual, ignore_index=True)
    # 恒等式の検査: 税目の合計 = 税収の合計
    parts = res[[f"{n}（兆円）" for n, _ in TAXES[:-1]]].sum(axis=1)
    assert np.allclose(parts, res["税収の合計（兆円）"], atol=1e-6), "税目の合計が税収の合計と一致しない"
    res.to_csv(OUT / "experiment_ctax_revenue.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    p = pd.concat(paths, ignore_index=True)
    p = p[["版", "政策", "四半期"] + [c for c in p.columns if c not in ("版", "政策", "四半期")]]
    p.to_csv(OUT / "experiment_ctax_revenue_paths.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    show = ["版", "政策", "年", "法人所得税（兆円）", "法人所得税（基準解比 %）", "企業所得 YCV（基準解比 %）", "消費税（兆円）",
            "個人所得税（兆円）", "税収の合計（兆円）", "事前の減税額（兆円）", "財源の戻り（%）", "うち法人所得税の増収（%）",
            "実質GDP（%）"]
    print(res[show].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
