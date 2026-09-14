"""論文への忠実性の検証: 印刷どおりの式・別解釈・誤植の修正前の式で11シナリオを解き、乗数表との一致を比べる.

既定の実装（model.py）は、論文の印刷から次の点で離れている。
  - 誤植の修正 7 件（[fix] 注記。係数の値はすべて論文どおり）
  - 式129（所得実効税率）を水準式ではなく DLOG 型で実装
  - 11 本の推定式の誤差修正項を標準解の値で固定（ecm.py）
本スクリプトは、それぞれを印刷どおり（または別の読み方）に戻した場合の乗数を計算し、
論文の乗数表との一致率・平均絶対誤差と、影響の大きい変数を出力する。

出力: output/experiment_fidelity.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ecm as ECMOD  # noqa: E402
import model as M  # noqa: E402
import simulate as SIM  # noqa: E402
from model import dl, ex, ln  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FXS2011 = 79.78  # 2011年平均 円/ドル（日銀 FM08/FXERM07 の月中平均）


def equations(itr_form: str = "dlog", replace: dict[str, M.Eq] | None = None) -> list[M.Eq]:
    old = M.ITR_FORM
    M.ITR_FORM = itr_form
    try:
        eqs = M.build_equations()
    finally:
        M.ITR_FORM = old
    replace = replace or {}
    return [replace.get(e.name, e) for e in eqs]


def run(eqs: list[M.Eq], live: frozenset[str]) -> pd.DataFrame:
    data = pd.read_csv(ROOT / "data/processed/model_data.csv", index_col="period")
    data.index = pd.PeriodIndex(data.index, freq="Q")
    model = M.Model(eqs)
    af = model.add_factors(data, SIM.SOLVE_START, SIM.END)
    base = model.solve(data, SIM.SOLVE_START, SIM.END, af)
    over, cols = ECMOD.frozen_overrides(model, base, SIM.solve_mask(base.index), live=live)
    out = []
    for s in SIM.build_scenarios(base, af):
        d = base.copy()
        for k, v in {**s.data, **cols}.items():
            d[k] = v
        sh = model.solve(d, SIM.SOLVE_START, SIM.END, af, fixed=s.fixed,
                         overrides={**over, **s.overrides}, shocks=s.shocks)
        out.append(SIM.multipliers(sh, base, s.no))
    return pd.concat(out, ignore_index=True)


# ---- 印刷どおり・別解釈の式 ------------------------------------------------
def eq9_fxs2011() -> M.Eq:
    """式9: 誤差修正項の相対価格に FXS2011 を使う（印刷どおり。定数の違いは誤差項に吸収される）."""
    def rpx(v, k, fx):
        return 100 * v("PXGS", k) / ((v("FXS", k) / fx) * v("WD_PX", k))
    return M.Eq(9, "XGS", "dlog", lambda v: (
        -0.229111 * (ln(v("XGS", 1) / v("WD_YVI", 1))
                     - (6.657787 - 0.198561 * ln(rpx(v, 1, FXS2011))
                        + 0.008266 * v("TIME", 1) - 4.82e-05 * v("TIME", 1) ** 2))
        + 1.297672 * dl(v, "WD_YVI")
        - 0.079268 * sum(ln(rpx(v, 1 + j, M.FXS2015)) - ln(rpx(v, 5 + j, M.FXS2015)) for j in range(5)) / 5
        - 0.367667 * v("D091") - 0.229254 * v("D202")))


def eq47_wiph() -> M.Eq:
    """式47: 未定義の WPHX を WPH ではなく WIPH（賃金・俸給ベース）と読む."""
    return M.Eq(47, "LF", "custom",
                lambda v: (-0.053590 * ln(v("POP65") / v("POP")) - 0.026685 * ln(v("UR", 2))
                           + 0.422674 * ln((v("WIPH") / v("PCPAT")) / v("WPHXREQ")) - 0.536525),
                lhs=lambda v: ln(v("LF") / v("POP")), inv=lambda v, y: v("POP") * ex(y))


def eq57_plus_const() -> M.Eq:
    """式57: 印刷の "--0.000488" を +0.000488 と読む（定数のみの違い）."""
    return M.Eq(57, "CGPIAT", "dlog", lambda v: (
        -0.033739 * (ln(v("CGPIAT", 1) / v("PMGSAT", 1))
                     - (0.650307 * ln(v("PGDPAT", 1) / v("PMGSAT", 1)) - 0.000317 * v("TIME", 1) + 0.014447))
        + 0.191675 * dl(v, "PMGSAT") + 0.483802 * dl(v, "PGDPAT") + 0.019649 * dl(v, "CUX", 1) + 0.000488))


def eq59_literal() -> M.Eq:
    """式59: 印刷の括弧をそのまま読む  (PIFPAT − PIFPAT(−4)/PIFPAT(−4))×100 = (PIFPAT − 1)×100."""
    return M.Eq(59, "PIFPATGR", "level", lambda v: (v("PIFPAT") - v("PIFPAT", 4) / v("PIFPAT", 4)) * 100)


def eq116_x400() -> M.Eq:
    """式116: D(DLOG(PCPAT(−1)*400)) を 400×D(DLOG(PCPAT(−1)))（年率インフレ率の変化）と読む."""
    return M.Eq(116, "RGBX", "d", lambda v: (
        -0.048253 * (v("RGB", 1) - sum(v("RCD", 1 + j) for j in range(8)) / 8)
        - 0.086901 * (v("RGB", 1) - v("RGB", 2)) + 0.510110 * (v("RCD") - v("RCD", 1))
        + 1.815275 * 400 * (dl(v, "PCPAT", 1) - dl(v, "PCPAT", 2))))


def eq134_printed() -> M.Eq:
    """式134: 印刷どおり設備投資の項を RTCI/(1+RTCI+PRTIF) とする."""
    def base(v):
        r = v("RTCI")
        return (r / (1 + r * v("PRTCP")) * v("CPV") + r / (1 + r + v("PRTIF")) * v("IFPV")
                + r / (1 + r * v("PRTIH")) * v("IHPV") + r / (1 + r * v("PRTCG")) * v("CGV")
                + r / (1 + r * v("PRTIG")) * v("IGV"))
    return M.Eq(134, "TCIV", "log", lambda v: (
        0.915210 * ln(base(v)) - 0.011036 * v("DTCIC2") * v("DTCIC2") * ln(base(v)) + 0.191987 * v("D972C")))


ALL_LIVE = frozenset(ECMOD.ECM_TERMS)
VARIANTS: list[tuple[str, str, dict, frozenset]] = [
    # (名前, 説明, 式の差し替え, 動かす誤差修正項)
    ("既定", "model.py の既定（式129 DLOG型、誤差修正項は CP・YICV・PCPAT のみ有効）", {}, ECMOD.DEFAULT_LIVE),
    ("式129 印刷どおり", "所得実効税率を印刷どおりの水準式にする（Issue #2）", {"__itr__": "level"}, ECMOD.DEFAULT_LIVE),
    ("誤差修正項 全有効", "14本の誤差修正項をすべて動かす（印刷どおり）", {}, ALL_LIVE),
    ("印刷どおり（誤植修正のみ）", "式129 水準式 + 誤差修正項 全有効 + 式9 FXS2011", {"__itr__": "level", "XGS": eq9_fxs2011()}, ALL_LIVE),
    ("式9 FXS2011", "誤差修正項の相対価格に FXS2011（印刷どおり）", {"XGS": eq9_fxs2011()}, ECMOD.DEFAULT_LIVE),
    ("式47 WPHX=WIPH", "未定義変数 WPHX を WIPH と読む", {"LF": eq47_wiph()}, ECMOD.DEFAULT_LIVE),
    ("式57 定数 +0.000488", "印刷の '--0.000488' を正の定数と読む", {"CGPIAT": eq57_plus_const()}, ECMOD.DEFAULT_LIVE),
    ("式59 括弧を字義どおり", "PIFPATGR = (PIFPAT − 1)×100 と読む", {"PIFPATGR": eq59_literal()}, ECMOD.DEFAULT_LIVE),
    ("式116 ×400", "インフレ率の変化を年率（×400）で入れる", {"RGBX": eq116_x400()}, ECMOD.DEFAULT_LIVE),
    ("式134 印刷どおり", "消費税の課税ベースに '1+RTCI+PRTIF' を使う", {"TCIV": eq134_printed()}, ECMOD.DEFAULT_LIVE),
]


def main() -> None:
    pub = pd.read_csv(ROOT / "output/published_multipliers.csv")
    pub = pub[pub.quarter == 0]
    rows, ref = [], None
    for name, desc, rep, live in VARIANTS:
        itr = rep.pop("__itr__", "dlog") if "__itr__" in rep else "dlog"
        rep = {k: v for k, v in rep.items() if k != "__itr__"}
        try:
            res = run(equations(itr, rep), live)
        except Exception as e:  # noqa: BLE001
            rows.append({"variant": name, "description": desc, "error": repr(e)})
            print(f"{name}: 失敗 {e!r}")
            continue
        m = res[res.quarter == 0].merge(pub, on=["scenario", "variable", "year"], suffixes=("", "_paper"))
        m["abs"] = (m["value"] - m["value_paper"]).abs()
        by_var = m.groupby("variable")["abs"].mean()
        if ref is None:
            ref = by_var
        delta = (by_var - ref).sort_values()
        g = m[m.variable == "GDP"]
        row = {"variant": name, "description": desc,
               "within_0.1": (m["abs"] <= 0.1).mean(), "MAE_all": m["abs"].mean(),
               "MAE_GDP": g["abs"].mean(), "max_err_GDP": g["abs"].max(),
               "worse_vars": ", ".join(f"{k}(+{v:.2f})" for k, v in delta.tail(3)[::-1].items() if v > 0.005),
               "better_vars": ", ".join(f"{k}({v:.2f})" for k, v in delta.head(3).items() if v < -0.005)}
        rows.append(row)
        print(f"{name:20s} ±0.1以内 {row['within_0.1']:.1%}  MAE {row['MAE_all']:.4f}  GDP MAE {row['MAE_GDP']:.4f}"
              f"  悪化: {row['worse_vars'] or '—'}  改善: {row['better_vars'] or '—'}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "output/experiment_fidelity.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
