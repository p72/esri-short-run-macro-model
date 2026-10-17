"""長期金利7%・ドル円200円シナリオ.

2024年版データ（基準解＝2022Q1〜2024Q4 の実績。長期金利は0.2〜1.0%、ドル円は116〜156円）に対して、
2022Q1 から次のショックを恒久的に与える。誤差修正項は既定（消費・個人企業所得・消費デフレーターの3本のみ有効）。

- A 円安のみ: ドル円 FXS を200円に固定。長期金利・短期金利はモデルどおり（短期金利はテイラー・ルール、下限0.001）
- B 長期金利のみ: 長期金利 RGB を7%に固定。為替は基準解に固定する（モデルの為替の式は日米金利差で円高に動くため、
  金利の効果だけを見るために止める）。短期金利はモデルどおり
- C 両方: 長期金利7%・ドル円200円

長期金利は、設備投資（資本コスト）、住宅投資、政府の利払い（式138、政府の純財産所得の比率が RGB に反応）、
家計の財産所得（式104 YIEV、家計は利子の受け取りが多いので金利上昇で可処分所得が増える）などを通じて効く。
ショックはモデルの推定範囲を大きく超えるので、結果は目安。特に、個人企業所得の式（91）は長期金利の2期差に反応するため、
金利を一度に6%pt以上上げると1年目に個人企業所得が大きく減り、その分が雇用者報酬（式89、両者の合計で決まる）に移る。

出力: output/experiment_rgb7_fx200.csv（年平均）, output/experiment_rgb7_fx200_paths.csv（四半期）
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import model as M  # noqa: E402
from experiment_tot_policy import setup, START, END  # noqa: E402  2024年版・2022Q1〜2024Q4・誤差修正項は既定

OUT = ROOT / "output"
RGB_LEVEL, FX_LEVEL = 7.0, 200.0
CASES = {"A 円安のみ（ドル円200円）": ("fx",), "B 長期金利のみ（7%）": ("rgb",), "C 長期金利7%・ドル円200円": ("fx", "rgb")}


def measures(s: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    pct = lambda v: (s[v] / b[v] - 1) * 100  # noqa: E731
    return pd.DataFrame({
        "実質GDP（%）": pct("GDP"), "実質消費（%）": pct("CP"), "設備投資（%）": pct("IFP"), "住宅投資（%）": pct("IHP"),
        "輸出（%）": pct("XGS"), "輸入（%）": pct("MGS"),
        "消費デフレーター（%）": pct("PCP"), "GDPデフレーター（%）": pct("PGDP"), "企業物価（%）": pct("CGPIAT"),
        "名目GDP（%）": pct("GDPV"), "名目雇用者報酬（%）": pct("YWV"),
        "実質賃金（%）": ((s.W / s.PCP) / (b.W / b.PCP) - 1) * 100,
        "失業率（%pt）": s.UR - b.UR,
        "短期金利（%pt）": s.RCD - b.RCD, "長期金利（%pt）": s.RGB - b.RGB, "ドル円（円）": s.FXS - b.FXS,
        "財政収支/名目GDP（%pt）": s.BGVATGDPV - b.BGVATGDPV,
        "政府の純財産所得/名目GDP（%pt）": (s.YIGV / s.GDPV - b.YIGV / b.GDPV) * 100,
        "政府純債務/名目GDP（%pt）": s.SBGVATGDPV - b.SBGVATGDPV,
    })


def main() -> None:
    m, af, base, d0, over = setup()
    idx = base.index
    on = idx >= pd.Period(START, "Q")
    annual, paths = [], []
    for label, shocks in CASES.items():
        fixed = {}
        if "fx" in shocks:
            fixed["FXS"] = base["FXS"].where(~on, FX_LEVEL)
        if "rgb" in shocks:
            fixed["RGB"] = base["RGB"].where(~on, RGB_LEVEL)
            fixed.setdefault("FXS", base["FXS"])  # 金利のみのケースは為替を基準解に固定
        ov = dict(over)
        s = m.solve(d0, START, END, af, fixed=fixed, overrides=ov)
        r = measures(s.loc[START:END], base.loc[START:END])
        paths.append(r.assign(シナリオ=label, 四半期=r.index.astype(str)))
        a = r.groupby(r.index.year).mean()
        a.insert(0, "年", [f"{i}年目（{y}）" for i, y in enumerate(a.index, 1)])
        a.insert(0, "シナリオ", label)
        annual.append(a)
    res = pd.concat(annual, ignore_index=True)
    res.to_csv(OUT / "experiment_rgb7_fx200.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    p = pd.concat(paths, ignore_index=True)
    p = p[["シナリオ", "四半期"] + [c for c in p.columns if c not in ("シナリオ", "四半期")]]
    p.to_csv(OUT / "experiment_rgb7_fx200_paths.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    print(res.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
