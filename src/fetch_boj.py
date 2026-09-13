"""日本銀行 時系列統計データ検索サイト API から月次・四半期系列を取得する.

出力: data/raw/boj_<DB>_<CODE>.json（生データ）と data/raw/boj_series.csv（縦持ち）
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
BASE = "https://www.stat-search.boj.or.jp/api/v1"
HEAD = {"Accept-Encoding": "gzip"}

FIXED = {
    # model name: (DB, series code)
    "FXS": ("FM08", "FXERM07"),               # ドル円 17時 月中平均
    "RCD": ("FM02", "STRAK_STRACDN2DB"),      # 譲渡性預金平均金利（新規発行分）総合
    "M2CD_OLD": ("MD02", "MAMS3ANM2C"),       # M2+CD 平残（〜2008/04）
    "M2_NEW": ("MD02", "MAM1NAM2M2MO"),       # M2 平残（2003/04〜）
    "CGPIAT": ("PR01", "PRCG20_1200000000"),  # 消費税を除く国内企業物価 総平均
    "CGPI": ("PR01", "PRCG20_2200000000"),    # 国内企業物価 総平均
    "PFUEL_IMP": ("PR01", "PRCG20_2600520001"),  # 輸入物価（円ベース）類別 石油・石炭・天然ガス
    "GOV_FA": ("FF", "FOF_FFAS420A900"),      # 資金循環 一般政府 資産合計（ストック）
    "GOV_FL": ("FF", "FOF_FFAS420L900"),      # 同 負債合計
    "GOV_NFA": ("FF", "FOF_FFAS420L700"),     # 同 金融資産・負債差額
    "HH_SHARE": ("FF", "FOF_FFAS430A330"),    # 資金循環 家計 株式等（ストック）
}


def get(path: str, **params) -> dict:
    for attempt in range(3):
        r = requests.get(f"{BASE}/{path}", params=params, headers=HEAD, timeout=120)
        if r.status_code == 200:
            data = r.json()
            if data.get("STATUS") == 200:
                return data
            raise ValueError(f"{path} {params}: {data.get('MESSAGEID')} {data.get('MESSAGE')}")
        time.sleep(5 * (attempt + 1))
    r.raise_for_status()


def search(db: str, must: list[str], freq: str | None = None) -> list[dict]:
    meta = get("getMetadata", db=db, lang="jp", format="json")["RESULTSET"]
    time.sleep(1.5)
    hits = [s for s in meta
            if all(k in (s.get("NAME_OF_TIME_SERIES_J") or "") for k in must)
            and (freq is None or s.get("FREQUENCY") == freq)
            and "更新停止" not in (s.get("NAME_OF_TIME_SERIES_J") or "")]
    return hits


def fetch(db: str, code: str) -> pd.Series:
    data = get("getDataCode", db=db, code=code, lang="jp", format="json")
    (RAW / f"boj_{db}_{code.replace('%', 'pct')}.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8")
    s = data["RESULTSET"][0]
    vals = s["VALUES"]
    out = pd.Series(pd.to_numeric(vals["VALUES"], errors="coerce"), index=[str(d) for d in vals["SURVEY_DATES"]])
    out.attrs["name"] = s.get("NAME_OF_TIME_SERIES_J")
    out.attrs["freq"] = s.get("FREQUENCY")
    time.sleep(1.5)
    return out


def main() -> None:
    targets = dict(FIXED)

    # 輸入物価指数（円ベース）石油・石炭・天然ガス
    hits = search("PR01", ["輸入物価指数/円ベース", "石油・石炭・天然ガス"], "MONTHLY")
    hits = [h for h in hits if h["SERIES_CODE"].startswith("PRCG20_26") and "前年比" not in h["NAME_OF_TIME_SERIES_J"]]
    print("輸入物価 候補:", [(h["SERIES_CODE"], h["NAME_OF_TIME_SERIES_J"]) for h in hits[:6]])
    if hits:
        targets["PFUEL_IMP"] = ("PR01", hits[0]["SERIES_CODE"])

    # 資金循環: 一般政府の金融資産・負債差額（ストック）
    ff = search("FF", ["一般政府", "ストック"], "QUARTERLY")
    for key, words in {"GOV_FA": ["資産・金融資産合計"], "GOV_FL": ["負債・負債合計"],
                       "GOV_NFA": ["金融資産・負債差額"]}.items():
        c = [h for h in ff if all(w in h["NAME_OF_TIME_SERIES_J"] for w in words)
             and h["NAME_OF_TIME_SERIES_J"].split("／")[1:2] == ["一般政府"]]
        print(key, "候補:", [(h["SERIES_CODE"], h["NAME_OF_TIME_SERIES_J"]) for h in c[:5]])
        if c:
            targets[key] = ("FF", c[0]["SERIES_CODE"])

    rows = []
    for name, (db, code) in targets.items():
        s = fetch(db, code)
        print(f"{name:10s} {db}/{code:22s} {s.attrs['freq']:10s} {s.index[0]}–{s.index[-1]}  {s.attrs['name']}")
        rows.append(pd.DataFrame({"name": name, "db": db, "code": code, "date": s.index, "value": s.values}))
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(RAW / "boj_series.csv", index=False, encoding="utf-8-sig")
    print(f"{len(df)} 行 → {RAW / 'boj_series.csv'}")


if __name__ == "__main__":
    main()
