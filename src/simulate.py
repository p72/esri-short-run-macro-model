"""論文の11の乗数シミュレーションを実行し、公表乗数表と比較する.

入力: data/processed/model_data.csv, output/published_multipliers.csv
出力: output/multipliers_reproduced.csv（シナリオ×変数×四半期／年平均）
      output/multipliers_comparison.csv（年平均の再現値と論文値）
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model as M  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
START, END = "2018Q1", "2020Q4"

# 乗数表の変数と表示（pt=乖離幅、それ以外は乖離率%）
PT_VARS = {"GDPD", "UR", "GDPGAP", "RCD", "RGB", "PERR", "BGVATGDPV", "SBGVATGDPV", "BCVATGDPV"}
ALIAS = {"LH": "LHX", "KFP": "KFP", "KHP": "KHP", "KG": "KG"}
TABLE_VARS = ["GDP", "CP", "IFP", "IHP", "IG", "CG", "XGS", "MGS", "GDPD", "PGDP", "PCP", "PIFP", "PIHP",
              "PXGS", "PMGS", "CGPI", "GDPV", "CPV", "IFPV", "IHPV", "IGV", "CGV", "XGSV", "MGSV", "NIV",
              "YWV", "YCV", "YDV", "TAXV", "TYPV", "TYCV", "TCIV", "WIPH", "LE", "LF", "LH", "UR", "CUX",
              "GDPPOT", "GDPGAP", "RCD", "RGB", "PERR", "UCC", "FXS", "PSHARE", "PLAND", "M2CD",
              "BGVATGDPV", "SBGVATGDPV", "BCVATGDPV", "KFP", "KHP", "KG"]


@dataclass
class Scenario:
    no: int
    title: str
    data: dict[str, pd.Series] = field(default_factory=dict)       # 外生変数の差し替え
    fixed: dict[str, pd.Series] = field(default_factory=dict)      # 内生変数の固定
    overrides: dict[str, M.Eq] = field(default_factory=dict)      # 式の差し替え
    shocks: dict[str, pd.Series] = field(default_factory=dict)     # 変換後左辺への加算


def sim_mask(index: pd.PeriodIndex) -> np.ndarray:
    return (index >= pd.Period(START, "Q")) & (index <= pd.Period(END, "Q"))


def build_scenarios(base: pd.DataFrame, af: pd.DataFrame) -> list[Scenario]:
    idx = base.index
    on = pd.Series(sim_mask(idx).astype(float), index=idx)
    # 外生変数のショックは期間後も継続させる（式にリード項があり、期末で途切れると反動が出る）
    after = pd.Series((idx >= pd.Period(START, "Q")).astype(float), index=idx)
    first = pd.Series((idx == pd.Period(START, "Q")).astype(float), index=idx)

    def add(var: str, amount: pd.Series) -> pd.Series:
        return base[var] + amount * after

    sc: list[Scenario] = []
    # (1) 実質公的固定資本形成を実質GDPの1%だけ継続的に拡大
    sc.append(Scenario(1, "実質公共投資 +実質GDP1%", data={"IG": add("IG", 0.01 * base["GDP"])}))
    # (2) 同上、短期金利固定
    sc.append(Scenario(2, "実質公共投資 +実質GDP1%（短期金利固定）",
                       data={"IG": add("IG", 0.01 * base["GDP"])}, fixed={"RCD": base["RCD"]}))
    # (3) 名目公的固定資本形成を名目GDPの1%だけ拡大: 名目IGVを外生、実質IGを内生化
    sc.append(Scenario(3, "名目公共投資 +名目GDP1%",
                       fixed={"IGV": add("IGV", 0.01 * base["GDPV"])},
                       overrides={"IGV": M.Eq(19, "IG", "level", lambda v: v("IGV") / v("PIG"))}))
    # (4) 個人所得税を名目GDPの1%減税
    sc.append(Scenario(4, "個人所得税 名目GDP1%減税", shocks={"TYPV": -0.01 * base["GDPV"] * on}))
    # (5) 法人所得税を名目GDPの1%減税: 実効税率ETTを税収が1%GDP減る分だけ引き下げ
    ycv_avg = base["YCV"].shift(1).rolling(4).mean()
    ett = base["ETT"] - on * 0.01 * base["GDPV"] / ycv_avg
    # 資本コスト式の法定実効税率 TT も同じ比率で引き下げる（論文表で資本コストが低下しているため）
    tt = base["TT"] * (ett / base["ETT"]).where(on > 0, 1.0)
    sc.append(Scenario(5, "法人所得税 名目GDP1%減税", fixed={"ETT": ett}, data={"TT": tt}))
    # (6) 消費税率 +1%pt
    # 論文の四半期乗数は、増税初期に「当期の税率変化」と「先行(駆け込み)項」が同時に効いた形と一致する
    # （消費 −1.300+0.725、住宅 −2.239+0.326+0.655）。初期四半期にリード項分を加えて再現する。
    sc.append(Scenario(6, "消費税率 +1%pt", data={"RTCI": add("RTCI", 0.01)},
                       shocks={"CP": 0.725444 * 0.01 * first,
                               "IHP": (0.326110 + 0.655187) * 0.01 * first}))
    # (7) 短期金利 +1%pt
    sc.append(Scenario(7, "短期金利 +1%pt", fixed={"RCD": add("RCD", 1.0)}))
    # (8) 貨幣供給量: 1年目に四半期0.25%ずつ減らし、以後 -1% を維持。金利は貨幣需要式を逆算
    k = pd.Series(np.cumsum(sim_mask(idx)).astype(float), index=idx).clip(upper=4) * on
    m2 = base["M2CD"] * (1 - 0.0025 * k)

    def rcd_from_money(v):
        # 式111 log(M2CD) = 1.000279·log(GDP·PGDPAT) − 0.035292·RCD + 0.006374·TIME − 0.335434 + 誤差項 を RCD について解く
        return (1.000279 * math.log(v("GDP") * v("PGDPAT")) + 0.006374 * v("TIME") - 0.335434
                + v("AF_M2CD") - math.log(v("M2CD"))) / 0.035292

    # M2CD を固定（式111は解かない）、式113（RCD）を貨幣需要の逆算式に置き換える。
    # RCD 式の誤差項は標準解で 0 なので、逆算式にそのまま足しても影響しない。
    sc.append(Scenario(8, "貨幣供給量 -1%", fixed={"M2CD": m2}, data={"AF_M2CD": af["M2CD"]},
                       overrides={"RCD": M.Eq(113, "RCD", "level", rcd_from_money)}))
    # (9) 円の対ドル10%減価（為替外生化）
    sc.append(Scenario(9, "円10%減価", fixed={"FXS": base["FXS"] * (1 + 0.10 * on)}))
    # (10) 原油価格 +20%
    sc.append(Scenario(10, "原油価格 +20%", data={"POILD": base["POILD"] * (1 + 0.20 * on)}))
    # (11) 世界需要 +1%
    sc.append(Scenario(11, "世界需要 +1%", data={"WD_YVI": base["WD_YVI"] * (1 + 0.01 * on)}))
    return sc


def multipliers(shock: pd.DataFrame, base: pd.DataFrame, no: int) -> pd.DataFrame:
    rows = []
    mask = sim_mask(base.index)
    for tv in TABLE_VARS:
        var = ALIAS.get(tv, tv)
        if var not in base:
            continue
        b, s = base.loc[mask, var], shock.loc[mask, var]
        # GDPD（成長率・年率）は内生変数なので、そのまま乖離幅をとる
        dev = (s - b) if tv in PT_VARS else (s / b - 1) * 100
        for p, val in dev.items():
            rows.append((no, tv, p.year, p.quarter, val))
    df = pd.DataFrame(rows, columns=["scenario", "variable", "year", "quarter", "value"])
    agg = df.groupby(["scenario", "variable", "year"], as_index=False)["value"].mean().assign(quarter=0)
    return pd.concat([df, agg], ignore_index=True)


def main() -> None:
    import argparse

    import ecm as ECMOD

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ecm", choices=["esri", "live"], default="esri",
                    help="esri: 消費関数以外の誤差修正項を標準解の値で固定（論文と整合、既定） / live: すべて動かす")
    args = ap.parse_args()

    data = pd.read_csv(ROOT / "data" / "processed" / "model_data.csv", index_col="period")
    data.index = pd.PeriodIndex(data.index, freq="Q")
    model = M.Model()
    af = model.add_factors(data, START, END)
    base = model.solve(data, START, END, af)
    core = ["GDP", "CP", "IFP", "GDPV", "PCP", "UR", "RCD", "FXS", "YDV", "BCV"]
    err = (base.loc[START:END, core] / data.loc[START:END, core] - 1).abs().max()
    print("標準解と実績の最大乖離率:", err.map(lambda x: f"{x:.1e}").to_dict())

    if args.ecm == "esri":
        ecm_over, ecm_cols = ECMOD.frozen_overrides(model, base, sim_mask(base.index))
        print(f"誤差修正項: {sorted(ECMOD.DEFAULT_LIVE)} のみ有効、{len(ecm_over)} 本を標準解の値で固定")
    else:
        ecm_over, ecm_cols = {}, {}
        print("誤差修正項: すべて有効")

    results = []
    for s in build_scenarios(base, af):
        d = base.copy()
        for k, val in {**s.data, **ecm_cols}.items():
            d[k] = val
        try:
            shock = model.solve(d, START, END, af, fixed=s.fixed,
                                overrides={**ecm_over, **s.overrides}, shocks=s.shocks)
        except Exception as e:  # noqa: BLE001 - 失敗したシナリオは報告して続行
            print(f"({s.no}) {s.title}: 失敗 {e}")
            continue
        results.append(multipliers(shock, base, s.no))
        g = results[-1].query("variable == 'GDP' and quarter == 0")["value"].round(2).tolist()
        print(f"({s.no:2d}) {s.title:28s} 実質GDP 年乗数 {g}")

    rep = pd.concat(results, ignore_index=True)
    out = ROOT / "output"
    suffix = "" if args.ecm == "esri" else "_ecmlive"
    rep.to_csv(out / f"multipliers_reproduced{suffix}.csv", index=False, encoding="utf-8-sig")
    pub = pd.read_csv(out / "published_multipliers.csv")
    cmp_ = (rep[rep.quarter == 0].merge(pub[pub.quarter == 0], on=["scenario", "variable", "year"],
                                         suffixes=("_repro", "_paper"))
            .drop(columns=["quarter_repro", "quarter_paper"]))
    cmp_["diff"] = cmp_["value_repro"] - cmp_["value_paper"]
    cmp_.to_csv(out / f"multipliers_comparison{suffix}.csv", index=False, encoding="utf-8-sig")
    print(f"論文と±0.1以内の割合（全変数×3年）: {(cmp_['diff'].abs() <= 0.1).mean():.1%}")
    key = cmp_[cmp_.variable.isin(["GDP", "CP", "IFP", "PCP", "UR", "RGB", "FXS", "BCVATGDPV"])]
    print(key.pivot_table(index=["scenario", "variable"], columns="year",
                          values=["value_repro", "value_paper"]).round(2).to_string())


if __name__ == "__main__":
    main()
