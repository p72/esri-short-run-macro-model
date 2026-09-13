"""論文の付属資料I（乗数詳細表）をテキストから抽出して CSV にする.

入力: reference/rn72_2022model.txt（PDF から抽出したテキスト）
出力: output/published_multipliers.csv
      列 = scenario, variable, year, quarter(0=年平均 AGGREGATE), value
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "reference" / "rn72_2022model.txt"
OUT = ROOT / "output" / "published_multipliers.csv"

SCENARIOS = {
    1: "実質公的固定資本形成 +実質GDP1%",
    2: "実質公的固定資本形成 +実質GDP1%（短期金利固定）",
    3: "名目公的固定資本形成 +名目GDP1%",
    4: "個人所得税 名目GDP1%減税",
    5: "法人所得税 名目GDP1%減税",
    6: "消費税率 +1%pt",
    7: "短期金利 +1%pt",
    8: "貨幣供給量 -1%",
    9: "円の対ドル10%減価",
    10: "原油価格 +20%",
    11: "世界需要 +1%",
}

_title = re.compile(r"^\((\d+)\)\s*\S")
_name = re.compile(r"^[A-Z][A-Z0-9_]*$")
_float = re.compile(r"^-?\d+\.\d+$")


def parse(text: str) -> pd.DataFrame:
    body = text.split("===== PAGE 24 =====", 1)[1].split("===== PAGE 46 =====", 1)[0]
    rows = []
    scen = None
    header: list[str] = []
    collecting = False
    year = quarter = None
    agg = False
    buf: list[float] = []

    warnings: list[str] = []

    def flush():
        nonlocal buf
        for nm, val in zip(header, buf):
            rows.append((scen, nm, year, 0 if agg else quarter, val))
        buf = []

    def boundary(where: str):
        # 行の区切りで値が余っていたら表の崩れ。持ち越さずに捨てて警告する
        nonlocal buf
        if buf:
            warnings.append(f"scenario {scen} {header[:1]} {year}/{quarter} {where}: {len(buf)}個の値を破棄 {buf}")
            buf = []

    for raw in body.splitlines():
        s = raw.strip()
        if not s:
            continue
        m = _title.match(s)
        if m:
            boundary("title")
            scen = int(m.group(1))
            continue
        if s == "AGGREGATE":
            boundary("aggregate")
            agg = True
            continue
        if _name.match(s) and s not in {"ESRI"}:
            if not collecting:
                boundary("header")
                header, collecting, agg = [], True, False
            header.append(s)
            continue
        if re.fullmatch(r"\d{4}", s):
            boundary("year")
            collecting = False
            year = int(s)
            if agg:
                quarter = 0
            continue
        if re.fullmatch(r"[1-4]", s) and not agg:
            boundary("quarter")
            collecting = False
            quarter = int(s)
            continue
        if _float.match(s) or s == "#REF!":
            buf.append(float(s) if s != "#REF!" else float("nan"))
            if len(buf) == len(header):
                flush()
    df = pd.DataFrame(rows, columns=["scenario", "variable", "year", "quarter", "value"])
    return df.dropna(subset=["value"])


def main() -> None:
    df = parse(SRC.read_text(encoding="utf-8"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    n_var = df.groupby("scenario")["variable"].nunique()
    print(f"{len(df)} 行を書き出しました: {OUT}")
    print("シナリオ別の変数数:", n_var.to_dict())
    gdp = df[(df.variable == "GDP") & (df.quarter == 0)].pivot(index="scenario", columns="year", values="value")
    print("\n実質GDP 年乗数（論文）:\n", gdp)


if __name__ == "__main__":
    main()
