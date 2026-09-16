"""ESRI から SNA のファイル（四半期GDP速報 CSV と年次推計 Excel）を版のフォルダに取得する.

使い方: ESRI_VINTAGE=2024 python src/fetch_sna.py
  vintage.py の版定義（qe=速報の4桁コード、annual=年次推計の年度、dir=保存先）に従って、
  gaku-jk/gaku-mk/def-qk/kshotoku-q の CSV と qom2/i4/i5/ss4n/ss1/ss5/si4/s6_2 の Excel を取得する。
  QE のフォルダは sna/menu.html から最新公表分のリンクを探し、版のコードと一致するものを使う
  （一致しなければ命名規則 files/{year}/qe{yyq}{_2}/tables/ から組み立てる）。

論文と同じ版（2021）のファイルは ESRI のアーカイブ構成が異なるため同梱のものを使う。
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vintage as VT  # noqa: E402

ESRI = "https://www.esri.cao.go.jp"
QE_FILES = ["gaku-jk", "gaku-mk", "def-qk", "kshotoku-q"]
ANNUAL_TABLES = ["qom2", "i4", "i5", "ss4n", "ss1", "ss5", "si4", "s6_2"]


def qe_folder(code: str) -> str:
    """速報コード（例 2622 = 2026年4-6月期2次）の tables/ フォルダ URL."""
    html = requests.get(f"{ESRI}/jp/sna/menu.html", timeout=60).content.decode("utf-8", "replace")
    for m in re.finditer(r'href="(/jp/sna/data/data_list/sokuhou/files/\d{4}/qe\d+(?:_2)?/tables/)[a-z\-]+(\d{4})\.csv"', html):
        if m.group(2) == code:
            return ESRI + m.group(1)
    yy, q, n = code[:2], code[2], code[3]
    year = 2000 + int(yy)
    return f"{ESRI}/jp/sna/data/data_list/sokuhou/files/{year}/qe{yy}{q}{'_2' if n == '2' else ''}/tables/"


def download(url: str, dest: Path) -> None:
    r = requests.get(url, timeout=120)
    if r.status_code != 200:
        print(f"  取得できません ({r.status_code}): {url}")
        return
    dest.write_bytes(r.content)
    print(f"  {dest.name} ({len(r.content):,} bytes)")
    time.sleep(1)


def main() -> None:
    v = VT.V
    out = v["dir"]
    out.mkdir(parents=True, exist_ok=True)
    print(f"版 {VT.NAME}: {v['label']} → {out}")
    folder = qe_folder(v["qe"])
    for s in QE_FILES:
        download(f"{folder}{s}{v['qe']}.csv", out / f"{s}{v['qe']}.csv")
    base = f"{ESRI}/jp/sna/data/data_list/kakuhou/files/{v['annual']}/tables/"
    for t in ANNUAL_TABLES:
        download(f"{base}{v['annual']}{t}_jp.xlsx", out / f"{v['annual']}{t}_jp.xlsx")


if __name__ == "__main__":
    main()
