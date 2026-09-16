"""e-Stat API から 2024年版の延長に必要な系列を取得する.

出力（data/raw/）:
  estat_cux_2020base.csv                製造工業 稼働率指数 季節調整済 月次 2020年=100（2018年1月〜）
  estat_trade_crude_0003425296.csv      普通貿易統計 概況品別国別表 輸入 原油及び粗油 2021〜2025年（月別 数量・金額）
  estat_trade_fuel_0003425296.csv       同 鉱物性燃料
  estat_lfs_monthly_sa.csv              労働力調査 基本集計 月次 季節調整値（労働力人口・就業者・雇用者・非労働力人口・完全失業率）

e-Stat API は連続リクエストで 403 を返すことがあるため、各リクエストの間に sleep を入れ、403 は待って再試行する。
APP ID は環境変数 ESTAT_APP_ID で与える。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
API = "https://api.e-stat.go.jp/rest/3.0/app/json/getStatsData"
APP = os.environ.get("ESTAT_APP_ID")


def get_stats_data(stats_id: str, **params) -> pd.DataFrame:
    """統計表を全件取得して縦持ち DataFrame（各分類の code/name 列 + value + unit）にする."""
    if not APP:
        raise SystemExit("環境変数 ESTAT_APP_ID に e-Stat の APP ID を設定してください")
    rows, start = [], 1
    names: dict[str, dict[str, str]] = {}
    while True:
        p = dict(appId=APP, statsDataId=stats_id, limit=100000, startPosition=start, **params)
        for attempt in range(4):
            r = requests.get(API, params=p, timeout=120)
            if r.status_code == 200:
                break
            time.sleep(20)
        r.raise_for_status()
        j = r.json()["GET_STATS_DATA"]["STATISTICAL_DATA"]
        for c in j["CLASS_INF"]["CLASS_OBJ"]:
            cls = c["CLASS"] if isinstance(c["CLASS"], list) else [c["CLASS"]]
            names[c["@id"]] = {x["@code"]: x["@name"] for x in cls}
        vals = j["DATA_INF"]["VALUE"]
        rows.extend(vals if isinstance(vals, list) else [vals])
        nxt = j["RESULT_INF"].get("NEXT_KEY")
        if not nxt:
            break
        start = int(nxt)
        time.sleep(3)
    df = pd.DataFrame(rows)
    out = pd.DataFrame()
    for cid in names:
        col = f"@{cid}"
        if col in df:
            out[f"{cid}_code"] = df[col]
            out[f"{cid}_name"] = df[col].map(names[cid])
    out["value"] = pd.to_numeric(df["$"], errors="coerce")
    out["unit"] = df.get("@unit")
    time.sleep(3)
    return out


def main() -> None:
    which = sys.argv[1:] or ["cux", "trade", "lfs"]
    if "cux" in which:
        # 総合季節調整済指数【月次】 稼働率（2020＝100.0）: 製造工業（1100000000）の全期間
        d = get_stats_data("0004052231", cdCat02="1100000000")
        d = d[d["time_name"].str.fullmatch(r"\d{6}")]
        d = d.rename(columns={"time_name": "cat01_name", "time_code": "cat01_code"})  # 2015年基準ファイルと同じ列名に揃える
        d.to_csv(RAW / "estat_cux_2020base.csv", index=False, encoding="utf-8-sig")
        print("cux:", d.cat01_name.min(), "〜", d.cat01_name.max(), len(d))
    if "trade" in which:
        for key, code in {"crude": "30301000", "fuel": "30000000"}.items():
            d = get_stats_data("0003425296", cdCat01=code)
            d = d[~d["cat02_name"].str.startswith(("合計", "単位"))]
            d.to_csv(RAW / f"estat_trade_{key}_0003425296.csv", index=False, encoding="utf-8-sig")
            print(f"trade {key}:", sorted(d.time_name.unique()), len(d))
    if "lfs" in which:
        sid = os.environ.get("ESTAT_LFS_ID")
        if not sid:
            print("lfs: ESTAT_LFS_ID が未設定のためスキップ")
        else:
            d = get_stats_data(sid)
            d.to_csv(RAW / "estat_lfs_monthly_sa.csv", index=False, encoding="utf-8-sig")
            print("lfs:", d.shape)


if __name__ == "__main__":
    main()
