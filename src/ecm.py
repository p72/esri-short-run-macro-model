"""誤差修正項（ECM）の定義と、乗数計算時にベースライン値で固定する仕組み.

論文の乗数表は「消費関数以外の誤差修正項がベースライン値のまま」の計算と最もよく一致する
（output/experiment_ecm_ablation.csv, experiment_ecm_combo.csv）。
例: 円10%減価シナリオの輸出は、誤差修正項なしで計算した経路と小数第2位まで一致する。

各項は model.py の該当式の右辺から切り出したもの。check_separation() で、前期の水準変数を
動かしても「右辺 − 誤差修正項」が変わらない（＝その変数の影響がすべて誤差修正項側にある）ことを検査する。
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

import model as M
from model import BETA, FXS2015, ln

Getter = Callable[..., float]


def _ydr(v, k):
    return v("YDV", k) / v("PCP", k)


def _rpx(v, k):
    return 100 * v("PXGS", k) / ((v("FXS", k) / FXS2015) * v("WD_PX", k))


def _lshare(v, k):
    return (v("YWV", k) + v("YICV", k)) / (v("NIV", k) + v("CCAV", k))


# 式の内生変数名 → 誤差修正項（係数×長期均衡からの乖離）
ECM_TERMS: dict[str, Callable[[Getter], float]] = {
    "CP": lambda v: -0.042671 * (ln(v("CP", 1)) - 0.986981 * ln(sum(_ydr(v, k) for k in range(1, 6)) / 5)
                                 - 0.007910 * ln(v("NWCV", 1) / v("PCP", 1))),
    "IHP": lambda v: -0.401905 * (ln(v("KHP", 1)) - (0.547149 * ln(sum(_ydr(v, k) for k in range(1, 6)) / 5)
                                                     + 0.003366 * v("TIME", 1) - 3.10e-05 * v("TIME", 1) ** 2
                                                     + 1.076611 * ln(v("HH", 1)) - 3.135839)),
    "XGS": lambda v: -0.229111 * (ln(v("XGS", 1) / v("WD_YVI", 1))
                                  - (6.657787 - 0.198561 * ln(_rpx(v, 1)) + 0.008266 * v("TIME", 1)
                                     - 4.82e-05 * v("TIME", 1) ** 2)),
    "FUEL": lambda v: -0.171986 * (ln(v("FUEL", 1) / v("GDP", 1))
                                   - (-0.002409 * ln(v("PFUELAT", 1) / v("PGDPAT", 1)) - 0.001171 * v("TIME", 1)
                                      - 3.000749)),
    "PXGS": lambda v: -0.161846 * (ln(v("PXGS", 1) / v("FXS", 1))
                                   - (0.725350 * ln(v("CGPIAT", 1) / v("FXS", 1)) - 0.002055 * v("TIME") - 0.985034)),
    "PFUELAT": lambda v: -0.329477 * (ln(v("PFUELAT", 1)) - (0.879168 * ln(v("POILD", 1) * v("FXS", 1)) - 7.984062)),
    "PNFMGSAT": lambda v: -0.360246 * (ln(v("PNFMGSAT", 1))
                                       - (-0.010084 * v("TIME", 1) + 2.81e-05 * v("TIME", 1) ** 2
                                          + 0.250637 * ln(v("WD_PI", 1) * v("FXS", 1)) - 1.496870)),
    "PCPAT": lambda v: -0.048746 * (ln(v("PCPAT", 1) / v("PGDPAT", 1)) + 0.092200
                                    - 0.021043 * ln(v("PMGSAT", 1) / v("PGDPAT", 1)) - 0.000646 * v("TIME", 1)),
    "CGPIAT": lambda v: -0.033739 * (ln(v("CGPIAT", 1) / v("PMGSAT", 1))
                                     - (0.650307 * ln(v("PGDPAT", 1) / v("PMGSAT", 1)) - 0.000317 * v("TIME", 1)
                                        + 0.014447)),
    "PIFPAT": lambda v: -0.022185 * (ln(v("PIFPAT", 1) / v("PNFMGSAT", 1))
                                     - (0.754583 * ln(v("PGDPAT", 1) / v("PNFMGSAT", 1)) - 0.000304 * v("TIME", 1)
                                        + 0.057748)),
    "LHX": lambda v: -0.198209 * (ln(v("LHX", 1)) - (-0.000837 * v("TIME", 1) - 0.057861 * v("D9203C", 1) + 4.770799)),
    "YWV": lambda v: -0.136934 * ln(_lshare(v, 1) / BETA),
    "YICV": lambda v: -0.539627 * (ln(v("YICV", 1)) - 0.857041 * ln(v("YWV", 1)) - (1 - 0.857041) * ln(v("YCV", 1))
                                   + 0.001211 * v("TIME", 1) + 1.650417),
    "PLAND": lambda v: -0.027890 * (ln(v("PLAND", 1) / v("GDPV", 1))
                                    - (0.110679 * ln(v("PSHARE", 1) / v("GDPV", 1))
                                       - 0.013395 * v("TIME70Q1", 1) - 10.62183)),
}

# 検査用: 右辺のうち誤差修正項にだけ現れる「前期の水準変数」
_PROBE = {"CP": "CP", "IHP": "KHP", "XGS": "XGS", "FUEL": "FUEL", "PXGS": "PXGS", "PFUELAT": "PFUELAT",
          "PNFMGSAT": "PNFMGSAT", "PCPAT": "PCPAT", "CGPIAT": "CGPIAT", "PIFPAT": "PIFPAT", "LHX": "LHX",
          "YICV": "YICV", "PLAND": "PLAND"}

# 論文の乗数表と最も整合する既定: 消費関数の誤差修正項のみ動かす
DEFAULT_LIVE = frozenset({"CP"})


def frozen_overrides(model: M.Model, base: pd.DataFrame, mask: np.ndarray,
                     live: frozenset[str] = DEFAULT_LIVE) -> tuple[dict[str, M.Eq], dict[str, pd.Series]]:
    """live 以外の誤差修正項を標準解の値で固定する差し替え式と、固定値を入れるデータ列を返す."""
    eqs = {e.name: e for e in model.eqs}
    X = {c: base[c].to_numpy(dtype=float) for c in base.columns}
    over: dict[str, M.Eq] = {}
    cols: dict[str, pd.Series] = {}
    for name, term in ECM_TERMS.items():
        if name in live:
            continue
        vals = np.full(len(base), np.nan)
        for t in np.where(mask)[0]:
            vals[t] = term(lambda nm, k=0, _t=t: X[nm][_t - k])
        col = f"ECMBASE_{name}"
        cols[col] = pd.Series(vals, index=base.index)
        e = eqs[name]
        over[name] = M.Eq(e.no, e.name, e.kind,
                          (lambda v, _e=e, _term=term, _c=col: _e.rhs(v) - _term(v) + v(_c)), e.lhs, e.inv)
    return over, cols


def check_separation(data: pd.DataFrame, period: str = "2018Q1", eps: float = 0.01) -> pd.DataFrame:
    """前期水準を ±eps だけ動かしたとき、右辺全体は変わり「右辺−誤差修正項」は変わらないことを確かめる."""
    model = M.Model()
    eqs = {e.name: e for e in model.eqs}
    t0 = data.index.get_loc(pd.Period(period, "Q"))
    X = {c: data[c].to_numpy(dtype=float).copy() for c in data.columns}
    rows = []
    for name, probe in _PROBE.items():
        e, term = eqs[name], ECM_TERMS[name]

        def v(nm, k=0, _t=t0):
            return X[nm][_t - k]

        rest0, rhs0 = e.rhs(v) - term(v), e.rhs(v)
        X[probe][t0 - 1] *= 1 + eps
        rest1, rhs1 = e.rhs(v) - term(v), e.rhs(v)
        X[probe][t0 - 1] /= 1 + eps
        rows.append({"式": name, "検査変数": f"{probe}(-1)", "右辺の変化": rhs1 - rhs0,
                     "右辺−ECMの変化": rest1 - rest0, "合格": abs(rest1 - rest0) < 1e-12 < abs(rhs1 - rhs0)})
    return pd.DataFrame(rows)
