"""交易損失は賃金を下げるのか：同じ交易ショックを、金融政策の設定を変えて解く.

note 記事「交易損失は賃金を下げるのか」（GDPデフレーターの寄与度分解で、2008年はデフレに戻り、2022年は物価上昇で
分担されて賃金上昇に至ったとし、分かれ目はマクロ政策の枠組みだと論じる）の反事実をモデルで試算する。

モデルのデータは2010Q1からなので、2008年そのものは解けない。2024年版データの2022〜24年を基準解（実績）にして、
2022Q1 から原油価格 POILD と海外の非燃料輸入価格 WD_PI を同じ率で恒久的に上げ、金融政策だけを3通りに変える。
率は、1年目の交易条件要因（GDPデフレーターと国内需要デフレーターの変化率の差）が記事の2022年の値 −2.46%pt に
なるよう校正する。

- 緩和継続: 短期金利 RCD と長期金利 RGB を基準解に固定（2022年の日銀に近い）
- モデルどおり: 既定のテイラー・ルール（式114、GDPデフレーター PGDPAT と GDP ギャップに反応、下限 0.001）
- 物価重視の引締め: 輸入インフレにも利上げで応じる。式113 を差し替え、基準解の短期金利に、
  消費デフレーター PCP の前年比の基準解からの乖離の1.5倍を足す（下限は既定と同じ 0.001）
- 参考: 緩和継続のまま、輸入物価の式68 と消費デフレーターの式56 を2011〜2024年で推定し直した版
  （experiment_fx_passthrough.py の reestimate。輸入コストが国内の物価に転嫁されやすい）

GDPデフレーターの分解は記事と同じ考え方。
- 分配面: 単位労働コスト（雇用者報酬/実質GDP）、固定資本減耗、純間接税（間接税−補助金）、単位利潤（残差）
- 支出面: 国内需要デフレーター（(名目GDP−名目純輸出)/(実質GDP−実質純輸出)）と、交易条件要因（残差）
寄与は「基準解の GDP デフレーターに対する各項目の変化」（%pt）で、合計は GDP デフレーターの乖離（%）に一致する。

出力: output/experiment_tot_policy.csv（年平均）, output/experiment_tot_policy_paths.csv（四半期）
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
from experiment_fx_passthrough import load, reestimate  # noqa: E402

OUT = ROOT / "output"
FNAME, START, END = "model_data_v2024.csv", "2022Q1", "2024Q4"
TARGET_TOT = -2.46   # 記事の2022年の交易条件要因（%pt）
TIGHT_K = 1.5        # 物価重視の引締め: 消費者物価の前年比の乖離1%pt あたりの利上げ幅（%pt）
POLICIES = {"ease": "緩和継続（金利を固定）", "model": "モデルどおり（テイラー・ルール）",
            "tight": "物価重視の引締め（消費者物価に反応）",
            "pass": "参考: 緩和継続＋パススルーを推定し直した式"}


def setup(variant: dict[str, M.Eq] | None = None, terms: dict | None = None):
    """基準解と、誤差修正項を固定した差し替え式を作る. variant は差し替える式（式56・68 を推定し直した版）."""
    data = load(FNAME)
    need = pd.Period(END, "Q") + 2  # 消費の式が RTCI の2期先を見る
    data = data.reindex(pd.period_range(data.index.min(), need, freq="Q"))
    data["RTCI"] = data["RTCI"].ffill()
    eqs = M.build_equations()
    if variant:
        eqs = [variant.get(e.name, e) for e in eqs]
    m = M.Model(eqs)
    af = m.add_factors(data, START, END)
    base = m.solve(data, START, END, af)
    idx = base.index
    mask = (idx >= pd.Period(START, "Q")) & (idx <= pd.Period(END, "Q"))
    over, cols = E.frozen_overrides(m, base, mask)
    # 差し替えた式の誤差修正項（既定で固定するもの）は、差し替えた式の項で固定し直す（experiment_fx_passthrough と同じ）
    for name, term in (terms or {}).items():
        if name in E.DEFAULT_LIVE:
            continue
        X = {c: base[c].to_numpy(dtype=float) for c in base.columns}
        vals = np.full(len(base), np.nan)
        for t in np.where(mask)[0]:
            vals[t] = term(lambda nm, k=0, _t=t: X[nm][_t - k])
        col = f"ECMBASE_{name}"
        cols[col] = pd.Series(vals, index=base.index)
        e = variant[name]
        over[name] = M.Eq(e.no, e.name, e.kind, (lambda v, _e=e, _t=term, _c=col: _e.rhs(v) - _t(v) + v(_c)), e.lhs, e.inv)
    d0 = base.copy()
    for k, v in cols.items():
        d0[k] = v
    # 物価重視の引締めのルールが読む基準解の値
    d0["BASE_RCD"] = base["RCD"]
    d0["BASE_PCPINF"] = (base["PCP"] / base["PCP"].shift(4) - 1) * 100
    return m, af, base, d0, over


def tight_rcd(v):
    infl = (v("PCP") / v("PCP", 4) - 1) * 100
    return max(0.001, v("BASE_RCD") + TIGHT_K * (infl - v("BASE_PCPINF")))


def run(m, af, base, d0, over, rate: float, policy: str) -> pd.DataFrame:
    d = d0.copy()
    on = pd.Series((d.index >= pd.Period(START, "Q")).astype(float), index=d.index)
    for var in ("POILD", "WD_PI"):
        d[var] = base[var] * (1 + rate * on)
    fixed, ov = {}, dict(over)
    if policy in ("ease", "pass"):
        fixed = {"RCD": base["RCD"], "RGB": base["RGB"]}
    elif policy == "tight":
        ov["RCD"] = M.Eq(113, "RCD", "level", tight_rcd)
    return m.solve(d, START, END, af, fixed=fixed, overrides=ov)


def measures(s: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """基準解からの乖離（四半期）と、GDP デフレーターの寄与度分解."""
    w = slice(START, END)
    s, b = s.loc[w], b.loc[w]
    pgdp = b.GDPV / b.GDP

    def per_gdp(x):  # 実質GDP 1単位あたりの名目額
        return x.GDPV / x.GDP, x.YWV / x.GDP, x.CCAV / x.GDP, (x.ITAXV - x.SUBV) / x.GDP

    (p_s, u_s, c_s, t_s), (p_b, u_b, c_b, t_b) = per_gdp(s), per_gdp(b)
    pdd = lambda x: (x.GDPV - x.BFV) / (x.GDP - x.BF)  # noqa: E731
    out = pd.DataFrame({
        "GDPデフレーター": (p_s / p_b - 1) * 100,
        "寄与_単位労働コスト": (u_s - u_b) / pgdp * 100,
        "寄与_単位利潤": ((p_s - u_s - c_s - t_s) - (p_b - u_b - c_b - t_b)) / pgdp * 100,
        "寄与_固定資本減耗": (c_s - c_b) / pgdp * 100,
        "寄与_純間接税": (t_s - t_b) / pgdp * 100,
        "寄与_国内需要デフレーター": (pdd(s) / pdd(b) - 1) * 100,
    })
    out["寄与_交易条件"] = out["GDPデフレーター"] - out["寄与_国内需要デフレーター"]
    for name, var in [("名目雇用者報酬", "YWV"), ("1人あたり賃金", "W"), ("消費デフレーター", "PCP"),
                      ("実質GDP", "GDP"), ("名目GDP", "GDPV"), ("実質消費", "CP")]:
        out[name] = (s[var] / b[var] - 1) * 100
    out["実質賃金"] = ((s.W / s.PCP) / (b.W / b.PCP) - 1) * 100
    out["失業率（%pt）"] = s.UR - b.UR
    out["短期金利（%pt）"] = s.RCD - b.RCD
    out["長期金利（%pt）"] = s.RGB - b.RGB
    return out


def main() -> None:
    m, af, base, d0, over = setup()
    # 校正: 1%の上昇で1年目の交易条件要因を測り、比例させる（モデルどおりの政策で）
    probe = 0.10
    tot1 = measures(run(m, af, base, d0, over, probe, "model"), base)["寄与_交易条件"].iloc[:4].mean()
    rate = probe * TARGET_TOT / tot1
    print(f"校正: 輸入価格 +{probe:.0%} で1年目の交易条件要因 {tot1:.3f}%pt → 目標 {TARGET_TOT} には +{rate:.1%}")

    # 参考ケース: 式56・68 を2011〜2024年で推定し直した版（#24）。基準解は実績のままで変わらない
    eqs_new, terms_new, _ = reestimate(load(FNAME))
    alt = setup(eqs_new, terms_new)
    annual, paths = [], []
    for key, label in POLICIES.items():
        mm, aa, bb, dd, oo = alt if key == "pass" else (m, af, base, d0, over)
        r = measures(run(mm, aa, bb, dd, oo, rate, key), bb)
        r.insert(0, "政策", label)
        paths.append(r.assign(四半期=r.index.astype(str)))
        a = r.drop(columns="政策").groupby(r.index.year).mean()
        a.insert(0, "政策", label)
        a.insert(1, "年", [f"{i}年目（{y}）" for i, y in enumerate(a.index, 1)])
        annual.append(a)
    res = pd.concat(annual, ignore_index=True)
    res.insert(2, "輸入価格の上昇率（%）", rate * 100)
    p = pd.concat(paths, ignore_index=True)
    p = p[["政策", "四半期"] + [c for c in p.columns if c not in ("政策", "四半期")]]

    # 恒等式の検査: 分配面・支出面の寄与の合計が GDP デフレーターの乖離と一致
    dist = res[[c for c in res.columns if c.startswith("寄与_") and c not in ("寄与_国内需要デフレーター", "寄与_交易条件")]]
    gap = (dist.sum(axis=1) - res["GDPデフレーター"]).abs().max()
    assert gap < 1e-9, f"分配面の寄与の合計が一致しない: {gap}"

    res.to_csv(OUT / "experiment_tot_policy.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    p.to_csv(OUT / "experiment_tot_policy_paths.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(res.drop(columns="輸入価格の上昇率（%）").round(2).to_string(index=False))


if __name__ == "__main__":
    main()
