"""円安の物価押し上げ効果（為替パススルー）の検証.

論文（ESRI Research Note No.72）の円10%減価シナリオ(9)では、家計消費デフレーター PCP の上昇は +0.15/+0.16/+0.23%
（円1%あたり約0.02%）。これが近年の日本経済でも成り立つかを3つの方法で確かめる。

1. モデル内の経路分解（式67・68の輸入物価 → 式56の消費デフレーター、または国内物価 PGDPAT 経由）
   既定版（基準解 2018〜20年）と2024年版（2022〜24年）で、円10%減価の PCP 効果を、
   輸入物価を基準解に固定した場合・国内物価 PGDPAT を固定した場合と比べて分ける。
2. 近年データでの推定（日銀 輸入物価指数・名目実効為替レート、総務省 CPI）
   CPI の8四半期（2年）の対数変化を、円安（名目実効為替レートの逆数）の2年変化・契約通貨建て輸入物価の2年変化・
   消費税率の変化で回帰し、2年累積のパススルーを期間別と10年の移動窓で出す。標準誤差は Newey-West（ラグ8）。
3. 係数を推定し直したモデル（2024年版データ、2011〜2024年）で円10%減価を解き直す
   式68（非燃料輸入物価）と式56（消費デフレーター）を論文と同じ形で OLS 推定し直し、差し替える（既定のモデルは変えない）。
   誤差項は推定し直した式で計算し直すので、基準解は実績のまま。式68 の誤差修正項は既定どおり標準解で固定、式56 は有効。

入力: data/processed/model_data.csv, model_data_v2024.csv, data/raw/boj_passthrough.csv（fetch_boj.py --passthrough）,
      data/raw/estat_cpi_0003427113.csv（fetch_estat.py cpi）
出力: output/experiment_fx_passthrough_model.csv（1・3）, output/experiment_fx_passthrough_cpi.csv（2 期間別）,
      output/experiment_fx_passthrough_rolling.csv（2 移動窓）, output/experiment_fx_passthrough_coef.csv（3 推定した係数）,
      output/experiment_fx_passthrough_paths.csv（3 四半期の経路、作図用）
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
from model import dl, ln  # noqa: E402

OUT = ROOT / "output"
H = 8  # 2年（四半期）
FX_SHOCK = 0.10
WINDOWS = {"既定版（2018〜20年）": ("model_data.csv", "2018Q1", "2020Q4"),
           "2024年版（2022〜24年）": ("model_data_v2024.csv", "2022Q1", "2024Q4")}
EST_SAMPLE = ("2011Q1", "2024Q4")


# ---------------------------------------------------------------------------
# 1・3: モデルで円10%減価を解く
# ---------------------------------------------------------------------------
def load(fname: str) -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data" / "processed" / fname, index_col="period")
    d.index = pd.PeriodIndex(d.index, freq="Q")
    return d


def ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.column_stack([np.ones(len(y)), X])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    se = np.sqrt(np.diag(e @ e / (len(y) - X.shape[1]) * np.linalg.inv(X.T @ X)))
    return b, se


def reestimate(d: pd.DataFrame) -> tuple[dict[str, M.Eq], dict[str, callable], pd.DataFrame]:
    """式56・68 を論文と同じ形で推定し直し、差し替え式と（式68 の）誤差修正項を返す."""
    L = np.log
    s = slice(*EST_SAMPLE)
    # 式68: dlog(PNFMGSAT) = a·(log PNFMGSAT(-1) − b·log(WD_PI·FXS)(-1) − c·TIME(-1) − c2·TIME(-1)^2) + k·dlog(WD_PI·FXS) + 定数
    w = L(d.WD_PI * d.FXS)
    x68 = pd.DataFrame({"y": L(d.PNFMGSAT).diff(), "lp": L(d.PNFMGSAT).shift(1), "wl": w.shift(1),
                        "t": d.TIME.shift(1), "t2": d.TIME.shift(1) ** 2, "dw": w.diff()}).loc[s].dropna()
    b68, se68 = ols(x68.y.values, x68[["lp", "wl", "t", "t2", "dw"]].values)
    c0, a, bw, bt, bt2, k = b68

    def term68(v):
        return a * ln(v("PNFMGSAT", 1)) + bw * ln(v("WD_PI", 1) * v("FXS", 1)) + bt * v("TIME", 1) + bt2 * v("TIME", 1) ** 2

    def rhs68(v):
        return c0 + term68(v) + k * (ln(v("WD_PI") * v("FXS")) - ln(v("WD_PI", 1) * v("FXS", 1)))

    # 式56: dlog(PCPAT) = a·(log(PCPAT/PGDPAT)(-1) − b·log(PMGSAT/PGDPAT)(-1) − c·TIME(-1)) + g0·dlog PGDPAT + g1·dlog PGDPAT(-1) + f·dlog PFUELAT + 定数
    x56 = pd.DataFrame({"y": L(d.PCPAT).diff(), "ep": (L(d.PCPAT) - L(d.PGDPAT)).shift(1),
                        "em": (L(d.PMGSAT) - L(d.PGDPAT)).shift(1), "t": d.TIME.shift(1),
                        "g0": L(d.PGDPAT).diff(), "g1": L(d.PGDPAT).diff().shift(1), "f0": L(d.PFUELAT).diff()}).loc[s].dropna()
    b56, se56 = ols(x56.y.values, x56[["ep", "em", "t", "g0", "g1", "f0"]].values)
    e0, ep, em, et, g0, g1, f0 = b56

    def rhs56(v):
        return (e0 + ep * ln(v("PCPAT", 1) / v("PGDPAT", 1)) + em * ln(v("PMGSAT", 1) / v("PGDPAT", 1)) + et * v("TIME", 1)
                + g0 * dl(v, "PGDPAT") + g1 * dl(v, "PGDPAT", 1) + f0 * dl(v, "PFUELAT"))

    eqs = {"PNFMGSAT": M.Eq(68, "PNFMGSAT", "dlog", rhs68), "PCPAT": M.Eq(56, "PCPAT", "dlog", rhs56)}
    coef = pd.DataFrame([
        ("式68", "誤差修正の速度", a, se68[1], -0.360246),
        ("式68", "長期: 円建て海外価格への弾力性", -bw / a, np.nan, 0.250637),
        ("式68", "短期: 円建て海外価格", k, se68[5], 0.336684),
        ("式56", "誤差修正の速度", ep, se56[1], -0.048746),
        ("式56", "長期: 輸入物価/国内物価への弾力性", -em / ep, np.nan, 0.021043),
        ("式56", "短期: 国内物価（当期）", g0, se56[4], 0.567277),
        ("式56", "短期: 燃料輸入物価", f0, se56[6], 0.018442),
    ], columns=["式", "項目", f"推定値（{EST_SAMPLE[0]}〜{EST_SAMPLE[1]}）", "標準誤差", "論文"])
    return eqs, {"PNFMGSAT": term68}, coef


def fx_scenario(fname: str, start: str, end: str, variant: dict[str, M.Eq] | None = None,
                variant_terms: dict | None = None, hold: tuple[str, ...] = ()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """円10%減価（start から恒久）を解き、基準解とショック解を返す. hold の変数は基準解に固定する."""
    data = load(fname)
    # 消費の式が RTCI の2期先を見るので、データの最終期の先まで税率を延ばす（simulate.py と同じ）
    need = pd.Period(end, "Q") + 2
    if data.index.max() < need:
        data = data.reindex(pd.period_range(data.index.min(), need, freq="Q"))
        data["RTCI"] = data["RTCI"].ffill()
    eqs = M.build_equations()
    if variant:
        eqs = [variant.get(e.name, e) for e in eqs]
    m = M.Model(eqs)
    af = m.add_factors(data, start, end)
    base = m.solve(data, start, end, af)
    mask = (base.index >= pd.Period(start, "Q")) & (base.index <= pd.Period(end, "Q"))
    over, cols = E.frozen_overrides(m, base, mask)
    # 差し替えた式68 の誤差修正項は、差し替えた式の項で固定し直す
    for name, term in (variant_terms or {}).items():
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
    d = base.copy()
    for k_, v_ in cols.items():
        d[k_] = v_
    on = pd.Series((base.index >= pd.Period(start, "Q")).astype(float), index=base.index)
    fixed = {"FXS": base["FXS"] * (1 + FX_SHOCK * on), **{h: base[h] for h in hold}}
    shock = m.solve(d, start, end, af, fixed=fixed, overrides=over)
    return base, shock


def annual(base: pd.DataFrame, shock: pd.DataFrame, var: str, start: str, end: str) -> list[float]:
    dev = (shock.loc[start:end, var] / base.loc[start:end, var] - 1) * 100
    return dev.groupby(dev.index.year).mean().tolist()


def model_part() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, paths = [], {}
    eqs_new, terms_new, coef = reestimate(load("model_data_v2024.csv"))
    cases = [("既定の係数", None, None, ()), ("既定の係数・輸入物価を固定", None, None, ("PNFMGSAT", "PFUELAT")),
             ("既定の係数・国内物価を固定", None, None, ("PGDPAT",))]
    for label, (fname, start, end) in WINDOWS.items():
        for case, var, terms, hold in cases:
            b, s = fx_scenario(fname, start, end, var, terms, hold)
            row = {"基準解": label, "ケース": case}
            for v_ in ("PCP", "PNFMGSAT", "PFUELAT", "PMGS", "PGDPAT", "GDP", "CP"):
                for i, x in enumerate(annual(b, s, v_, start, end), 1):
                    row[f"{v_}_{i}年目"] = x
            rows.append(row)
            if case == "既定の係数":
                paths[(label, case)] = (s.loc[start:end, "PCP"] / b.loc[start:end, "PCP"] - 1) * 100
    # 3: 推定し直した式（2024年版の基準解のみ）
    fname, start, end = WINDOWS["2024年版（2022〜24年）"]
    for case, var in [("式68 を推定し直す", {"PNFMGSAT": eqs_new["PNFMGSAT"]}),
                      ("式56 を推定し直す", {"PCPAT": eqs_new["PCPAT"]}),
                      ("式56・68 を推定し直す", eqs_new)]:
        terms = {k_: v_ for k_, v_ in terms_new.items() if k_ in var}
        b, s = fx_scenario(fname, start, end, var, terms)
        row = {"基準解": "2024年版（2022〜24年）", "ケース": case}
        for v_ in ("PCP", "PNFMGSAT", "PFUELAT", "PMGS", "PGDPAT", "GDP", "CP"):
            for i, x in enumerate(annual(b, s, v_, start, end), 1):
                row[f"{v_}_{i}年目"] = x
        rows.append(row)
        paths[("2024年版（2022〜24年）", case)] = (s.loc[start:end, "PCP"] / b.loc[start:end, "PCP"] - 1) * 100
    res = pd.DataFrame(rows)
    p = pd.DataFrame({f"{k[0]}|{k[1]}": v.reset_index(drop=True) for k, v in paths.items()})
    p.index = [f"{i + 1}四半期目" for i in range(len(p))]
    return res, coef, p


# ---------------------------------------------------------------------------
# 2: 近年データでの推定
# ---------------------------------------------------------------------------
def quarterly_boj(b: pd.DataFrame, name: str) -> pd.Series:
    s = b[b["name"] == name].set_index("date")["value"].astype(float)
    s.index = pd.PeriodIndex([f"{d[:4]}-{d[4:]}" for d in s.index], freq="M")
    return s.groupby(s.index.asfreq("Q")).mean()


def quarterly_cpi(c: pd.DataFrame, code: str) -> pd.Series:
    x = c[c["cat01_code"] == code]
    s = pd.Series(x["value"].astype(float).values,
                  index=pd.PeriodIndex([f"{t[:4]}-{t[-2:]}" for t in x["time_code"]], freq="M")).sort_index()
    q = s.groupby(s.index.asfreq("Q")).agg(["mean", "count"])
    return q.loc[q["count"] == 3, "mean"]  # 3か月そろった四半期だけ


def newey_west(y: np.ndarray, X: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    X = np.column_stack([np.ones(len(y)), X])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    xi = np.linalg.inv(X.T @ X)
    u = X * e[:, None]
    S = u.T @ u
    for lag in range(1, lags + 1):
        g = u[lag:].T @ u[:-lag]
        S += (1 - lag / (lags + 1)) * (g + g.T)
    return b, np.sqrt(np.diag(xi @ S @ xi))


def tax_rate(idx: pd.PeriodIndex) -> pd.Series:
    r = pd.Series(0.0, index=idx)
    for p, v in [("1989Q2", 0.03), ("1997Q2", 0.05), ("2014Q2", 0.08), ("2019Q4", 0.10)]:
        r[r.index >= pd.Period(p, "Q")] = v
    return np.log(1 + r)


def regress(df: pd.DataFrame) -> tuple[float, float, int]:
    """2年変化の回帰。窓の中で消費税率が変わらないときは税の項を落とす."""
    cols = ["e", "pc"] + (["t"] if df["t"].abs().max() > 0 else [])
    b, se = newey_west(df["y"].values, df[cols].values, H)
    return b[1], se[1], len(df)


def cpi_part() -> tuple[pd.DataFrame, pd.DataFrame]:
    b = pd.read_csv(ROOT / "data" / "raw" / "boj_passthrough.csv", dtype={"date": str})
    c = pd.read_csv(ROOT / "data" / "raw" / "estat_cpi_0003427113.csv", dtype=str)
    e = -np.log(quarterly_boj(b, "NEER"))           # 上昇＝円安
    pc = np.log(quarterly_boj(b, "IMP_CONTRACT"))   # 契約通貨建ての輸入物価（海外の価格）
    py = np.log(quarterly_boj(b, "IMP_YEN"))
    targets = {"輸入物価（円ベース）": py, "CPI 総合": None, "CPI 生鮮食品を除く総合": None,
               "CPI 生鮮食品及びエネルギーを除く総合": None, "CPI 財": None, "CPI サービス": None}
    codes = {"CPI 総合": "0001", "CPI 生鮮食品を除く総合": "0161", "CPI 生鮮食品及びエネルギーを除く総合": "0178",
             "CPI 財": "0202", "CPI サービス": "0220"}
    for k_, code in codes.items():
        targets[k_] = np.log(quarterly_cpi(c, code))
    periods = {"1995〜2012年": ("1995Q1", "2012Q4"), "2013〜2020年": ("2013Q1", "2020Q4"),
               "2021年〜": ("2021Q1", "2030Q4"), "1995年〜全期間": ("1995Q1", "2030Q4")}
    rows, roll = [], []
    for name, y in targets.items():
        idx = y.index
        df = pd.DataFrame({"y": y - y.shift(H), "e": e - e.shift(H), "pc": pc - pc.shift(H)}, index=idx)
        df["t"] = (tax_rate(idx) - tax_rate(idx).shift(H)) * (0 if name.startswith("輸入") else 1)
        df = df.dropna()
        for pname, (a, z) in periods.items():
            beta, se, n = regress(df.loc[a:z])
            rows.append({"系列": name, "期間": pname, "推定期間": f"{df.loc[a:z].index[0]}〜{df.loc[a:z].index[-1]}",
                         "観測数": n, "2年累積パススルー（円1%あたり%）": beta, "標準誤差": se})
        if name in ("CPI 生鮮食品を除く総合", "CPI 財", "輸入物価（円ベース）"):
            for end_i in range(40, len(df) + 1):
                w = df.iloc[end_i - 40:end_i]
                beta, se, _ = regress(w)
                roll.append({"系列": name, "窓の終わり": str(w.index[-1]), "係数": beta, "標準誤差": se})
    return pd.DataFrame(rows), pd.DataFrame(roll)


def main() -> None:
    pd.set_option("display.width", 220)
    res, coef, paths = model_part()
    res.to_csv(OUT / "experiment_fx_passthrough_model.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    coef.to_csv(OUT / "experiment_fx_passthrough_coef.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    paths.to_csv(OUT / "experiment_fx_passthrough_paths.csv", encoding="utf-8-sig", float_format="%.6f")
    show = ["基準解", "ケース"] + [f"PCP_{i}年目" for i in (1, 2, 3)] + [f"PNFMGSAT_{i}年目" for i in (1, 3)] + ["GDP_3年目"]
    print("円10%減価の効果（基準解からの乖離、%）\n", res[show].round(3).to_string(index=False))
    print("\n推定し直した係数\n", coef.round(4).to_string(index=False))
    cpi, roll = cpi_part()
    cpi.to_csv(OUT / "experiment_fx_passthrough_cpi.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    roll.to_csv(OUT / "experiment_fx_passthrough_rolling.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    print("\n2年累積パススルー（円＝名目実効為替レートが1%下がったときの上昇率%）\n", cpi.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
