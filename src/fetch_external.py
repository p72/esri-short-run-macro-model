"""FRED の代わりに ESRI 景気動向指数・OECD から外部変数を取得する.

出力: data/raw/external_series.csv（縦持ち: name, date[YYYY-MM or YYYYQn], value, source）
  PSHARE  TOPIX 月平均               ESRI 景気動向指数 先行系列 L9（data/raw/esri_ci1_*.xlsx）
  RGB     10年国債新発債流通利回り    同 L10B
  US_RGB  米国長期金利                OECD DF_FINMARK IRLT
  US_WPI  米国生産者物価              OECD DF_KEI PP
  WD_PX   競争国輸出価格の代理        OECD DF_KEI PP（G7→なければ米国）
  WD_PI   競争国輸入価格の代理        同上
  WD_YVI  世界GDPの代理               OECD DF_QNA OECD計 実質GDP
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OECD = "https://sdmx.oecd.org/public/rest/data"


def esri_ci1() -> list[pd.DataFrame]:
    f = sorted(RAW.glob("esri_ci1_*.xlsx"))[-1]
    df = pd.read_excel(f, sheet_name=0, header=None).iloc[5:]
    df = df[pd.to_numeric(df[1], errors="coerce").notna() & pd.to_numeric(df[2], errors="coerce").notna()]
    date = [f"{int(y):04d}-{int(m):02d}" for y, m in zip(df[1], df[2])]
    out = []
    for name, col in {"PSHARE": 11, "RGB": 14}.items():
        out.append(pd.DataFrame({"name": name, "date": date, "value": pd.to_numeric(df[col], errors="coerce"),
                                 "source": f"ESRI {f.name} col{col}"}))
    return out


def oecd(flow: str, key: str, start: str) -> pd.DataFrame:
    url = f"{OECD}/{flow}/{key}?startPeriod={start}&format=csvfilewithlabels&dimensionAtObservation=AllDimensions"
    for attempt in range(3):
        r = requests.get(url, timeout=120)
        if r.status_code == 200 and len(r.text) > 100:
            (RAW / f"oecd_{flow.split(',')[1].replace('@', '_')}_{key.replace('+', '-').replace('.', '_')[:40]}.csv").write_text(r.text, encoding="utf-8")
            return pd.read_csv(io.StringIO(r.text))
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"OECD {r.status_code}: {url}\n{r.text[:300]}")


def pick(df: pd.DataFrame, **cond) -> pd.DataFrame:
    for col, val in cond.items():
        if col in df.columns:
            df = df[df[col].astype(str) == val]
    return df


def series(df: pd.DataFrame, name: str, source: str) -> pd.DataFrame:
    s = df[["TIME_PERIOD", "OBS_VALUE"]].dropna().drop_duplicates("TIME_PERIOD").sort_values("TIME_PERIOD")
    date = s["TIME_PERIOD"].astype(str).str.replace("-Q", "Q", regex=False)
    return pd.DataFrame({"name": name, "date": date, "value": s["OBS_VALUE"], "source": source})


def main() -> None:
    parts = esri_ci1()

    fin = oecd("OECD.SDD.STES,DSD_STES@DF_FINMARK,4.0", "USA.M.IRLT.PA.....", "2008-01")
    parts.append(series(fin, "US_RGB", "OECD FINMARK USA IRLT"))
    time.sleep(2)

    kei = oecd("OECD.SDD.STES,DSD_KEI@DF_KEI,4.0", "USA+G7+OECD.M.PP.IX...", "2008-01")
    print("KEI PP 組合せ:", kei.groupby(["REF_AREA", "ACTIVITY", "ADJUSTMENT", "TRANSFORMATION"]).size().to_dict())
    usa = pick(kei, REF_AREA="USA", TRANSFORMATION="_Z")
    usa = usa[usa["ACTIVITY"].astype(str).isin(["_T", "BTE", "C"])].sort_values("ACTIVITY")
    usa = usa[usa["ACTIVITY"] == usa["ACTIVITY"].iloc[0]]
    parts.append(series(usa, "US_WPI", f"OECD KEI USA PP activity={usa['ACTIVITY'].iloc[0]}"))
    g7 = pick(kei, REF_AREA="G7", TRANSFORMATION="_Z")
    world = g7 if len(g7) else usa
    world = world[world["ACTIVITY"] == world["ACTIVITY"].iloc[0]]
    tag = f"OECD KEI {world['REF_AREA'].iloc[0]} PP（代理）"
    parts.append(series(world, "WD_PX", tag))
    parts.append(series(world, "WD_PI", tag))
    time.sleep(2)

    qna = oecd("OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA,1.1", "Q.Y.OECD.S1.S1.B1GQ._Z._Z._Z....", "2008-Q1")
    print("QNA 組合せ:", qna.groupby(["UNIT_MEASURE", "PRICE_BASE", "TRANSFORMATION", "TABLE_IDENTIFIER"]).size().to_dict())
    vol = qna[(qna["PRICE_BASE"].isin(["LR", "L"])) & (qna["TRANSFORMATION"] == "N")]
    vol = vol[vol["UNIT_MEASURE"] == vol["UNIT_MEASURE"].iloc[0]]
    parts.append(series(vol, "WD_YVI", f"OECD QNA OECD計 実質GDP {vol['UNIT_MEASURE'].iloc[0]}/{vol['PRICE_BASE'].iloc[0]}（代理）"))

    out = pd.concat(parts, ignore_index=True)
    out.to_csv(RAW / "external_series.csv", index=False, encoding="utf-8-sig")
    print(out.groupby("name").agg(n=("value", "size"), first=("date", "min"), last=("date", "max"),
                                  source=("source", "first")).to_string())


if __name__ == "__main__":
    main()
