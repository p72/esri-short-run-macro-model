"""SNA データの版（vintage）の切り替え.

論文（2022年12月）は「令和3年に公表された令和2年度国民経済計算年次推計（平成27年基準）」を使っている。
QE は 2021年10-12月期2次速報（2022年3月公表、2020年度年次推計ベース）が論文の公共投資乗数から
逆算した IG/GDP 比と 0.001%pt で一致した（2021年7-9月期2次は 0.02、2025年版は 0.04 ずれる）。
その後の年次推計で 2019〜2020年の値は改定されているため、既定はこの版とする。

環境変数 ESRI_VINTAGE=2025 で、2025年4-6月期2次QE + 2022年度年次推計の版に切り替えられる。
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

VINTAGES = {
    # 論文と同じ版（既定）
    "2021": dict(dir=RAW / "vintage2021", qe="2142", annual="2020", suffix="",
                 label="2021年10-12月期2次QE（2022年3月公表）+ 2020年度年次推計（2021年12月公表）"),
    # 取得時点の最新版
    "2025": dict(dir=RAW, qe="2522", annual="2022", suffix="_v2025",
                 label="2025年4-6月期2次QE + 2022年度年次推計"),
    # 2024Q4 まで延長した版（2020年基準）: 2026年4-6月期2次QE + 2024年度年次推計
    "2024": dict(dir=RAW / "vintage2024", qe="2622", annual="2024", suffix="_v2024", end="2024Q4", raw_end_year=2025,
                 label="2026年4-6月期2次QE（2020年基準）+ 2024年度年次推計（2025年12月公表）"),
}

NAME = os.environ.get("ESRI_VINTAGE", "2021")
if NAME not in VINTAGES:
    raise SystemExit(f"ESRI_VINTAGE は {sorted(VINTAGES)} のいずれか: {NAME}")
V = VINTAGES[NAME]
END = V.get("end", "2021Q4")  # モデル用データの最終四半期
RAW_END_YEAR = V.get("raw_end_year", 2020)  # e-Stat 貿易統計など、年単位で追加取得する原データの使用最終年


def qe_file(series: str) -> Path:
    """四半期GDP速報 CSV（series は gaku-jk / gaku-mk / def-qk / kshotoku-q）."""
    return V["dir"] / f"{series}{V['qe']}.csv"


def annual_file(table: str) -> Path:
    """年次推計 Excel（table は qom2 / i4 / i5 / ss4n / ss1 / ss5 / si4 / s6_2）."""
    return V["dir"] / f"{V['annual']}{table}_jp.xlsx"


def processed(name: str) -> Path:
    """data/processed 配下の出力名（既定版以外はサフィックス付き）."""
    stem, ext = name.rsplit(".", 1)
    return ROOT / "data" / "processed" / f"{stem}{V['suffix']}.{ext}"
