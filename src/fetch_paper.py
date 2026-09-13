"""論文PDF（ESRI Research Note No.72）を内閣府サイトから取得し、テキストを抽出する.

論文は著作物のためリポジトリに同梱していない。src/published.py の前に実行する。
出力: reference/e_rnote072.pdf, reference/rn72_2022model.txt
"""
from __future__ import annotations

from pathlib import Path

import pymupdf
import requests

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "reference"
URL = "https://www.esri.cao.go.jp/jp/esri/archive/e_rnote/e_rnote080/e_rnote072.pdf"


def main() -> None:
    REF.mkdir(exist_ok=True)
    pdf = REF / "e_rnote072.pdf"
    if not pdf.exists():
        r = requests.get(URL, timeout=120)
        r.raise_for_status()
        pdf.write_bytes(r.content)
        print(f"取得: {URL} → {pdf} ({len(r.content):,} bytes)")
    doc = pymupdf.open(pdf)
    out = REF / "rn72_2022model.txt"
    with out.open("w", encoding="utf-8") as f:
        for i, page in enumerate(doc):
            f.write(f"\n===== PAGE {i + 1} =====\n")
            f.write(page.get_text())
    print(f"抽出: {doc.page_count} ページ → {out}")


if __name__ == "__main__":
    main()
