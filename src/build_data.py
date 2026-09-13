"""モデル用の四半期データセットを組み立てる.

入力: data/processed/sna_quarterly.csv, sna_annual.csv（src/sna.py の出力）と data/raw/ の各種ファイル
出力: data/processed/model_data.csv   … 2010Q1〜2021Q4、モデル変数名
      data/processed/data_notes.csv   … 変数ごとの出所・加工・仮定

方針: 乗数は「ショック解 − 標準解」なので、誤差項で吸収される水準の違いより
      構成比・ショックと非線形に絡む変数の妥当性を優先する。代用・仮定は notes に必ず記録する。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model as M  # noqa: E402
from sna import seasonal_adjust  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
START, END = "2010Q1", "2021Q4"
IDX = pd.period_range(START, END, freq="Q")

NOTES: dict[str, str] = {}


def note(names: str, text: str) -> None:
    for n in names.split():
        NOTES[n] = text


def to_q(s: pd.Series, how: str = "mean") -> pd.Series:
    """月次（PeriodIndex[M]）→ 四半期."""
    g = s.groupby(s.index.asfreq("Q"))
    out = g.mean() if how == "mean" else g.sum()
    cnt = g.count()
    return out[cnt == 3] if how == "sum" else out


def rebase(s: pd.Series, year: int = 2015, level: float = 1.0) -> pd.Series:
    base = s[[p.year == year for p in s.index]].mean()
    return s / base * level


# ---------------------------------------------------------------------------
# 読み込み
# ---------------------------------------------------------------------------
def load_sna() -> tuple[pd.DataFrame, pd.DataFrame]:
    q = pd.read_csv(PROC / "sna_quarterly.csv", index_col="period")
    q.index = pd.PeriodIndex(q.index, freq="Q")
    a = pd.read_csv(PROC / "sna_annual.csv", index_col="year")
    return q, a


def load_boj() -> dict[str, pd.Series]:
    df = pd.read_csv(RAW / "boj_series.csv", dtype={"date": str})
    out = {}
    for name, g in df.groupby("name"):
        vals = pd.to_numeric(g["value"], errors="coerce").to_numpy()
        if g["db"].iloc[0] == "FF":  # YYYYQQ
            idx = pd.PeriodIndex([pd.Period(year=int(d[:4]), quarter=int(d[4:]), freq="Q") for d in g["date"]])
        else:  # YYYYMM
            idx = pd.PeriodIndex([pd.Period(year=int(d[:4]), month=int(d[4:]), freq="M") for d in g["date"]])
        out[name] = pd.Series(vals, index=idx).dropna()
    return out


def load_lfs() -> pd.DataFrame:
    rows = list(openpyxl.load_workbook(RAW / "estat_lfs_000040115411.xlsx", read_only=True, data_only=True)
                ["季節調整値"].iter_rows(values_only=True))
    recs, k = [], 0
    for r in rows[10:]:
        if r[1] is None or not re.fullmatch(r"\d+月", str(r[1]).strip()):
            continue
        per = pd.Period("1953-01", "M") + k
        if int(str(r[1]).strip()[:-1]) != per.month:
            raise ValueError(f"労働力調査の月の並びが崩れています: {per} vs {r[1]}")
        k += 1
        num = lambda v: pd.to_numeric(str(v).strip(), errors="coerce")  # noqa: E731
        recs.append((per, num(r[4]), num(r[7]), num(r[10]), num(r[16]), num(r[19])))
    df = pd.DataFrame(recs, columns=["p", "LF", "LE", "LW", "NILF", "UR"]).set_index("p")
    return df.apply(to_q)


def load_pop65() -> pd.Series:
    d = pd.read_csv(RAW / "estat_lfs_pop65_q.csv")
    # 65歳以上・男女計など内訳の「総数」だけを残す（cat03/cat04 は性別等の内訳）
    d = d[(d["cat02_code"].astype(int) == 7) & (d["cat03_code"].astype(int) == 0) & (d["cat04_code"].astype(int) == 0)]
    ym = d["time_name"].str.extract(r"(\d{4})年(\d+)～").astype(int)
    idx = pd.PeriodIndex([pd.Period(year=y, quarter=(m + 2) // 3, freq="Q") for y, m in zip(ym[0], ym[1])])
    s = pd.Series(d["value"].to_numpy(dtype=float), index=idx).sort_index()
    if s.index.duplicated().any():
        raise ValueError("65歳以上人口に重複した四半期があります（内訳の絞り込みを確認）")
    return s


def load_cux() -> pd.Series:
    d = pd.read_csv(RAW / "estat_cux_2015base.csv", dtype={"cat01_name": str})
    d = d[(d["cat02_code"] == 1100000000) & d["cat01_name"].str.fullmatch(r"\d{6}")]
    idx = pd.PeriodIndex([pd.Period(year=int(s[:4]), month=int(s[4:]), freq="M") for s in d["cat01_name"]])
    return to_q(pd.Series(d["value"].to_numpy(dtype=float), index=idx).sort_index())


def load_hours() -> pd.Series:
    """毎月勤労統計 長期時系列表29（TLシート）の指数ブロックから四半期原指数を読む（列4〜7が第1〜第4四半期）."""
    df = pd.read_excel(RAW / "estat_maikin_t29_hours_5plus.xls", sheet_name="TL", header=None)
    vals: dict[pd.Period, float] = {}
    for _, r in df.iloc[9:].iterrows():
        y = pd.to_numeric(r[0], errors="coerce")
        if pd.isna(y):
            if vals:  # 指数ブロックの終わり（下に増減率ブロックが続く）
                break
            continue
        for q in range(1, 5):
            v = pd.to_numeric(r[3 + q], errors="coerce")
            if pd.notna(v):
                vals[pd.Period(year=int(y), quarter=q, freq="Q")] = float(v)
    return pd.Series(vals).sort_index()


def load_trade(prefix: str) -> pd.DataFrame | None:
    files = sorted(RAW.glob(f"estat_trade_{prefix}_*.csv"))
    if not files:
        return None
    d = pd.concat([pd.read_csv(f) for f in files])
    d["month"] = d["cat02_name"].str.extract(r"(\d+)月").astype(int)
    d["kind"] = np.where(d["cat02_name"].str.contains("金額"), "value", "qty")
    d["year"] = d["time_name"].str.extract(r"(\d{4})").astype(int)
    g = d.groupby(["year", "month", "kind"])["value"].sum().unstack("kind")
    g.index = pd.PeriodIndex([pd.Period(year=y, month=m, freq="M") for y, m in g.index])
    return g.sort_index()


def load_external() -> dict[str, pd.Series]:
    """代替取得先（ESRI景気動向指数、OECD 等）の縦持ちCSV: name,date(YYYY-MM または YYYYQn),value."""
    f = RAW / "external_series.csv"
    if not f.exists():
        return {}
    d = pd.read_csv(f, dtype={"date": str})
    out = {}
    for name, g in d.groupby("name"):
        freq = "Q" if "Q" in g["date"].iloc[0] else "M"
        s = pd.Series(g["value"].to_numpy(dtype=float), index=pd.PeriodIndex(g["date"], freq=freq)).sort_index()
        out[name] = to_q(s) if freq == "M" else s
    return out


# ---------------------------------------------------------------------------
# 加工の部品
# ---------------------------------------------------------------------------
def fy_to_q(values: pd.Series) -> pd.Series:
    """年度値を四半期（年度内一定）に割り付ける。年度y = yQ2〜(y+1)Q1."""
    out = pd.Series(np.nan, index=IDX)
    for p in IDX:
        fy = p.year if p.quarter >= 2 else p.year - 1
        if fy in values.index:
            out[p] = values[fy]
    return out


def calendar_step(schedule: list[tuple[str, float]]) -> pd.Series:
    """[(開始四半期, 値), ...] の階段関数."""
    out = pd.Series(np.nan, index=IDX)
    for start, val in schedule:
        out[out.index >= pd.Period(start, "Q")] = val
    return out


def stock_from_flows(year_end: pd.Series, flow: pd.Series, price: pd.Series) -> tuple[pd.Series, pd.Series]:
    """モデルの蓄積式 K_t = K_{t-1}·g_t·(1−r) + I_t/4（g=P_t/P_{t-1}）で四半期ストックを作る.

    除却率 r は暦年ごとに一定とし、第4四半期末が SNA の暦年末残高に一致するよう二分法で決める。
    """
    K = pd.Series(np.nan, index=IDX)
    R = pd.Series(np.nan, index=IDX)
    for y in range(IDX[0].year, IDX[-1].year + 1):
        if y - 1 not in year_end.index or y not in year_end.index:
            continue
        qs = [pd.Period(year=y, quarter=k, freq="Q") for k in range(1, 5)]
        prev = pd.Period(year=y - 1, quarter=4, freq="Q")
        if any(q not in flow.index or pd.isna(flow[q]) for q in qs) or prev not in price.index:
            continue

        def path(r: float) -> list[float]:
            k, pp, res = year_end[y - 1], price[prev], []
            for q in qs:
                k = k * (price[q] / pp) * (1 - r) + flow[q] / 4
                pp = price[q]
                res.append(k)
            return res

        lo, hi = -0.2, 0.2
        for _ in range(80):
            mid = (lo + hi) / 2
            if path(mid)[-1] > year_end[y]:
                lo = mid
            else:
                hi = mid
        for q, k in zip(qs, path((lo + hi) / 2)):
            K[q] = k
            R[q] = (lo + hi) / 2
        if prev in IDX and pd.isna(K[prev]):
            K[prev] = year_end[y - 1]
    return K, R


def interp_year_end(year_end: pd.Series, log: bool = False) -> pd.Series:
    pts = pd.Series({pd.Period(year=int(y), quarter=4, freq="Q"): v for y, v in year_end.items()})
    full = pd.period_range(pts.index.min(), pts.index.max(), freq="Q")
    s = pts.reindex(full)
    s = np.exp(np.log(s).interpolate()) if log else s.interpolate()
    return s.reindex(IDX)


def fill_identities(df: pd.DataFrame, max_pass: int = 30) -> pd.DataFrame:
    """データの無い内生変数を、水準型の定義式を実績値に当てはめて埋める."""
    model = M.Model()
    X = {c: df[c].to_numpy(dtype=float).copy() for c in df.columns}
    n = len(df)
    for e in model.eqs:
        X.setdefault(e.name, np.full(n, np.nan))
    for _ in range(max_pass):
        changed = 0
        for e in model.eqs:
            if e.kind != "level":
                continue
            arr = X[e.name]
            for t in np.where(np.isnan(arr))[0]:
                def v(nm, k=0, _t=t):
                    i = _t - k
                    if i < 0 or i >= n:
                        return np.nan
                    return X[nm][i]
                try:
                    val = e.rhs(v)
                except (KeyError, ValueError, ZeroDivisionError, OverflowError, TypeError):
                    continue
                if val is not None and np.isfinite(val):
                    arr[t] = val
                    changed += 1
        if changed == 0:
            break
    return pd.DataFrame(X, index=df.index)


# ---------------------------------------------------------------------------
# 組み立て
# ---------------------------------------------------------------------------
def build() -> pd.DataFrame:
    sna, ann = load_sna()
    boj = load_boj()
    ext = load_external()
    D = sna.reindex(IDX).copy()
    D = D.drop(columns=[c for c in ["NFIV", "GNIV", "TPIV", "TINCGV", "YINPV", "TAXNETV"] if c in D])
    note("GDP CP IHP IFP INP CG IG ING BF XGS MGS KAISA", "ESRI 2015年基準QE(2025Q2 2次) 実質季調年率")
    note("GDPV CPV IHPV IFPV INPV CGV IGV INGV BFV XGSV MGSV RTRIV PTRIV", "ESRI 2015年基準QE 名目季調年率")
    note("YWV YWIV YOLIV YIGV YIEV YICV NIV CCAVG TCIV TCSTV SUBV CSSV TYPV YDV BSSV TYCV OITAXV",
         "2022年度年次推計 四半期原系列を移動平均比率法で季調し×4")
    note("CCAV", "GDP−間接税+補助金+海外純所得−国民所得（不突合を含む）")
    note("SDV", "0（不突合はCCAVに含めた）")

    # ---- 暦・ダミー
    D = D.join(M.make_calendar(IDX))

    # ---- 消費税（税率と転嫁率）
    D["RTCI"] = calendar_step([("2010Q1", 0.05), ("2014Q2", 0.08), ("2019Q4", 0.10)])
    note("RTCI", "消費税率（標準税率）")
    prt = dict(PRTCP=0.52, PRTIF=0.18, PRTIH=0.80, PRTGP=0.86, PRTIG=0.45, PRTCG=0.48,
               PRTFU=0.0, PRTNF=0.0, PRTMG=0.0)
    prt["PRTNP"], prt["PRTNG"] = prt["PRTGP"], prt["PRTIG"]
    for k, val in prt.items():
        D[k] = val
    note(" ".join(prt), "論文 乗数表(6)消費税+1%pt の2018Q1デフレータ反応から逆算した転嫁率")

    # ---- デフレーター（名目÷実質で精度を確保）と消費税抜き
    for p, (nom, real) in {"PGDP": ("GDPV", "GDP"), "PCP": ("CPV", "CP"), "PIFP": ("IFPV", "IFP"),
                           "PIHP": ("IHPV", "IHP"), "PCG": ("CGV", "CG"), "PIG": ("IGV", "IG"),
                           "PXGS": ("XGSV", "XGS"), "PMGS": ("MGSV", "MGS")}.items():
        D[p] = D[nom] / D[real]
    for at, (p, r) in {"PCPAT": ("PCP", "PRTCP"), "PIFPAT": ("PIFP", "PRTIF"), "PIHPAT": ("PIHP", "PRTIH"),
                       "PCGAT": ("PCG", "PRTCG"), "PIGAT": ("PIG", "PRTIG")}.items():
        D[at] = D[p] / (1 + D["RTCI"] * D[r])
    D["CGPI"] = rebase(to_q(boj["CGPI"])).reindex(IDX)
    D["CGPIAT"] = rebase(to_q(boj["CGPIAT"])).reindex(IDX)
    D["PINP"], D["PING"] = D["CGPI"], D["PIG"]
    D["PINPAT"] = D["PINP"] / (1 + D["RTCI"] * D["PRTNP"])
    D["ERRINPV"] = D["INPV"] - D["INP"] * D["PINP"]
    D["ERRINGV"] = D["INGV"] - D["ING"] * D["PING"]
    note("PINPAT", "PINP /(1+RTCI·PRTNP)")
    note("CGPI CGPIAT", "日銀 国内企業物価（含む／除く消費税）四半期平均 2015=1")
    note("PINP PING", "在庫デフレーターの代理（PINP=企業物価, PING=公的固定資本形成デフレーター）")

    # ---- 輸入の燃料・非燃料分割
    fuel = load_trade("fuel")
    D["PFUEL"] = rebase(to_q(boj["PFUEL_IMP"])).reindex(IDX)
    if fuel is not None:
        fq = to_q(fuel["value"], how="sum") / 1e6 * 4  # 千円→10億円、年率
        D["FUELV"] = seasonal_adjust(fq).reindex(IDX)
        note("FUELV", "普通貿易統計 鉱物性燃料 輸入額（全国合計, 四半期計×4, 季調）")
    D["FUEL"] = D["FUELV"] / D["PFUEL"] if "FUELV" in D else np.nan
    D["NFMGSV"] = D["MGSV"] - D["FUELV"]
    D["NFMGS"] = D["MGS"] - D["FUEL"]
    D["PNFMGS"] = D["NFMGSV"] / D["NFMGS"]
    D["PFUELAT"], D["PNFMGSAT"], D["PMGSAT"] = D["PFUEL"], D["PNFMGS"], D["PMGS"]  # 輸入の転嫁率は0
    # 消費税抜きGDPデフレーター: 式26の名目(消費税除く)を実質GDPで割る
    D["PGDPAT"] = (D["CP"] * D["PCPAT"] + D["IFP"] * D["PIFPAT"] + D["IHP"] * D["PIHPAT"]
                   + D["INPV"] / (1 + D["RTCI"] * D["PRTNP"]) + D["INGV"] / (1 + D["RTCI"] * D["PRTNG"])
                   + D["CG"] * D["PCGAT"] + D["IG"] * D["PIGAT"] + D["XGSV"] - D["MGS"] * D["PMGSAT"]) / D["GDP"]
    note("PGDPAT", "式26の名目GDP(消費税除く)÷実質GDP")
    note("PFUEL", "日銀 輸入物価（円ベース）石油・石炭・天然ガス 2015=1")
    note("NFMGS NFMGSV PNFMGS", "輸入計−鉱物性燃料（連鎖の非加法性は無視）")

    crude = load_trade("crude")
    D["FXS"] = to_q(boj["FXS"]).reindex(IDX)
    if crude is not None and "qty" in crude:
        cq = to_q(crude["value"], how="sum") * 1000 / to_q(crude["qty"], how="sum")  # 円/kL
        D["POILD"] = (cq.reindex(IDX) / D["FXS"]) / 6.2898
        note("POILD", "通関 原粗油 輸入単価（円/kL）÷円ドル÷6.2898（ドル/バレル）")

    # ---- 労働
    lfs = load_lfs().reindex(IDX)
    D["LF"], D["LE"], D["LW"], D["UR"] = lfs["LF"], lfs["LE"], lfs["LW"], lfs["UR"]
    D["POP"] = lfs["LF"] + lfs["NILF"]
    D["RLEW"] = D["LW"] / D["LE"]
    note("LF LE LW UR POP", "労働力調査 長期時系列 季節調整値 四半期平均（POP=労働力+非労働力）")
    pop65 = load_pop65().reindex(IDX)
    share = (pop65 / D["POP"]).dropna()
    D["POP65"] = pop65.where(pop65.notna(), D["POP"] * share.iloc[0])
    note("POP65", "労働力調査 年齢階級別（2018Q1〜）。それ以前は2018Q1の比率で延長（LF式は当期値のみ使用）")
    D["HH"] = D["POP"]
    note("HH", "代理: 15歳以上人口（住宅投資式の対数項は誤差項に吸収され乗数に影響しない）")
    D["CUX"] = load_cux().reindex(IDX)
    note("CUX", "製造工業稼働率指数 季調 2015=100 四半期平均（e-Stat）")
    D["LHX"] = rebase(seasonal_adjust(load_hours()), level=100).reindex(IDX)
    note("LHX", "毎月勤労統計 長期時系列表29 総実労働時間指数（5人以上・就業形態計・原指数）を季調し2015=100")

    # ---- 賃金
    D["WI"] = D["YWIV"] / D["LW"]
    D["W"] = D["YWV"] / D["LW"]

    # ---- 金融
    old, new = to_q(boj["M2CD_OLD"]), to_q(boj["M2_NEW"])
    ov = old.index.intersection(new.index)
    ratio = (old[ov] / new[ov]).mean()
    D["M2CD"] = (new * ratio).reindex(IDX) / 10
    note("M2CD", f"日銀 M2平残×{ratio:.4f}（M2+CD との重複期間平均比）÷10（億円→10億円）")
    D["RCD"] = to_q(boj["RCD"]).reindex(IDX).clip(lower=0.001)
    D["RCDX"] = D["RCD"]
    note("RCD", "日銀 譲渡性預金平均金利（新規発行分）総合 四半期平均（下限0.001）")
    if "RGB" in ext:
        D["RGB"] = ext["RGB"].reindex(IDX).clip(lower=0.001)
        D["RGBX"] = D["RGB"]
        note("RGB", "10年国債利回り 四半期平均（マイナス期間は式の下限0.001で置換）")

    # ---- ストック
    D["KFPV"], D["RRFPV"] = stock_from_flows(ann["KFPV"], D["IFPV"], D["PIFP"])
    D["KHPV"], D["RRHPV"] = stock_from_flows(ann["KHPV"], D["IHPV"], D["PIHP"])
    D["KGV"], D["RRKGV"] = stock_from_flows(ann["KGV"], D["IGV"], D["PIG"])
    D["KNPV"], D["RRNPV"] = stock_from_flows(ann["KNPV"], D["INPV"], D["PINP"])
    D["RRFP"] = D["RRFPV"]
    note("KFPV KHPV KGV KNPV", "SNA 固定資本ストック/在庫の暦年末残高を、モデルの蓄積式で四半期化")
    note("RRFPV RRHPV RRKGV RRNPV RRFP", "暦年末残高に一致させる年内一定の除却率（逆算）")

    # ---- 政府
    D["RKGV"] = D["RRKGV"] * D["KGV"].shift(1) * 4
    D["RKG"] = D["RKGV"] / D["PIG"].shift(1)
    D["CCAVGR"] = (D["CCAVG"] / D["PCG"]) / D["RKG"]
    D["CGXRKG"] = D["CG"] - D["CCAVGR"] * D["RKG"]
    fy_ig = pd.Series({fy: D.loc[[p for p in IDX if (p.year if p.quarter >= 2 else p.year - 1) == fy], "IGV"].mean()
                       for fy in range(IDX[0].year, IDX[-1].year)})
    D["IGVR"] = fy_to_q(ann["GOVGFCF_FY"] / fy_ig)
    D["IGVR"] = D["IGVR"].ffill().bfill()
    note("IGVR", "GFS 一般政府 固定資産の純取得(年度)÷公的固定資本形成(年度)")
    base = sum(D[c] for c in ["YWV", "BSSV", "YIEV", "YICV", "OTYDV"])
    D["ITR"] = D["TYPV"] / base.rolling(4).mean()
    D["ITREQ"] = D["ITR"].loc["2010Q4":"2019Q4"].mean()
    D["ETT"] = D["TYCV"] / D["YCV"].shift(1).rolling(4).mean()
    D["TT"] = fy_to_q(pd.Series({2010: .4069, 2011: .4069, 2012: .3801, 2013: .3801, 2014: .3464,
                                 2015: .3211, 2016: .2997, 2017: .2997, 2018: .2974, 2019: .2974,
                                 2020: .2974, 2021: .2974}))
    note("TT", "法人実効税率（財務省公表, 年度）")
    D["RTCST"] = D["TCSTV"] / D["MGSV"]
    D["SR"] = calendar_step([("2010Q1", .16058), ("2010Q4", .16412), ("2011Q4", .16766), ("2012Q4", .17120),
                             ("2013Q4", .17474), ("2014Q4", .17828), ("2015Q4", .18182), ("2016Q4", .18300)])
    note("SR", "厚生年金保険料率（毎年9月改定→第4四半期から）")
    D["IR"] = 1.0
    note("IR", "所得代替率: 1（社会保障給付式の対数定数に吸収）")
    # 論文の乗数表で累積赤字/GDP の初期変化が「比率×GDP変化率」に一致するには比率≈214%が必要で、
    # 純債務(≈125%)ではなく総債務に近い。資金循環の一般政府「負債合計」を使う。
    # 資金循環の「負債・合計」は金融資産・負債差額を含むバランス項目で資産合計に一致するため、
    # 負債そのものは 資産合計 − 金融資産・負債差額 で求める（2018年でGDP比約231%）
    D["SBGV"] = ((boj["GOV_FA"] - boj["GOV_NFA"]) / 10).reindex(IDX)
    note("SBGV", "資金循環 一般政府 負債（資産合計−金融資産・負債差額, ストック）÷10。論文乗数表の比率変化から総債務ベースと判断")
    D["OTNGV"] = 0.0

    # ---- 海外
    D["BCV"] = D["BFV"] + D["RTRIV"] - D["PTRIV"]
    D["ERRBCV"] = 0.0
    D["FASSTV"] = interp_year_end(ann["FASSTV"], log=True)
    D["SBCV"] = interp_year_end(ann["SBCV"])
    D["FLIABV"] = D["FASSTV"] - D["SBCV"]
    D["RSBCV"] = (D["SBCV"] - D["BCV"] / 4) / D["SBCV"].shift(1)
    note("FASSTV SBCV", "SNA 対外資産・対外純資産 暦年末残高を四半期補間")
    for name in ["US_RGB", "US_WPI", "WD_YVI", "WD_PX", "WD_PI"]:
        if name in ext:
            D[name] = ext[name].reindex(IDX)

    # ---- 資産
    D["LANDT"] = interp_year_end(ann["LANDT"])
    # 家計保有の土地は SNA 家計部門の期末貸借対照表から。利上げ時の家計の利子収入（式104）は
    # 家計純資産 NWCV の水準に比例するため、全額家計保有と仮定すると所得効果が過大になる
    D["LANDV"] = interp_year_end(ann["LANDV_HH"])
    D["PROLA"] = D["LANDT"] / D["LANDV"]
    D["PLAND"] = rebase(D["LANDT"])
    D["RLAND"] = D["LANDV"] / D["PLAND"]
    note("LANDV", "SNA 家計（個人企業を含む）期末貸借対照表 土地の暦年末残高を四半期補間")
    note("PLAND PROLA", "代理: SNA 土地残高を指数化（市街地価格指数の代わり）。PROLA＝土地全体÷家計保有")
    D["SHARETV"] = interp_year_end(ann["SHARETV"])
    if "PSHARE" in ext:
        D["PSHARE"] = rebase(ext["PSHARE"]).reindex(IDX)
        D["RSHARET"] = D["SHARETV"] / D["PSHARE"]
    if "HH_SHARE" in boj:
        D["SHAREV"] = (boj["HH_SHARE"] / 10).reindex(IDX)
        note("SHAREV", "資金循環 家計 株式等（ストック）÷10（億円→10億円）")
        if "PSHARE" in D:
            D["RSHARE"] = D["SHAREV"] / D["PSHARE"]

    # ---- 資本コスト関係（乗数への影響は小さい: 設備投資式の均衡資本比率の係数 −0.0017）
    D["REQU"], D["SLRATIO"], D["TINCR"], D["ROR"] = 0.40, 0.50, 0.0, 2.0
    note("REQU SLRATIO TINCR ROR", "仮定値（資本コストは設備投資式でほぼ効かないため簡便に設定）")

    # ---- 均衡値（対数差で入るものは水準が乗数に影響しない）
    D["UREQ"] = 3.0
    note("UREQ", "仮定: 3.0%（JILPT 均衡失業率の概ねの水準）")
    D["CUXEQ"] = D["CUX"].loc["2010Q1":"2019Q4"].mean()
    if "LHX" in D:
        D["LHXEQ"] = D["LHX"].rolling(13, center=True, min_periods=7).mean()
    D["WPH"] = 100 * D["W"] / D["LHX"] if "LHX" in D else np.nan
    D["WPHXREQ"] = (D["WPH"] / D["PCPAT"]).loc["2010Q1":"2019Q4"].mean()

    # ---- 生産関数の誤差項（式54が実績の稼働率を再現するように）
    if "LHX" in D:
        tfp = np.exp(M.GAM + M.RAM1 * D["TIME"] + M.RAM2 * D["TIME96Q2"])
        kfp = D["KFPV"] / D["PIFP"]
        D["ERRPFU"] = D["GDP"] - tfp * (D["LE"] * D["LHX"]) ** M.BETA * (kfp.shift(1) * D["CUX"] / 100) ** (1 - M.BETA)
        note("ERRPFU", "式54を実績CUXで満たすように逆算（GDP比は data_notes の print を参照）")

    D["RSBGV"] = np.nan  # 定義式を埋めた後に計算
    D = fill_identities(D)
    D["RSBGV"] = D["SBGV"] - D["SBGV"].shift(1) + D["BGV"] / 4
    return D


def main() -> None:
    D = build()
    PROC.mkdir(parents=True, exist_ok=True)
    D.to_csv(PROC / "model_data.csv", encoding="utf-8-sig", index_label="period")
    pd.Series(NOTES, name="note").rename_axis("variable").to_csv(PROC / "data_notes.csv", encoding="utf-8-sig")

    model = M.Model()
    need = set(model.endog)
    import inspect
    src = inspect.getsource(M)
    need |= set(re.findall(r'v\("([A-Z0-9_]+)"', src))
    win = D.loc["2015Q1":"2020Q4"]
    missing = sorted(n for n in need if n not in D.columns or win[n].isna().any())
    print(f"model_data.csv: {D.shape}")
    print(f"2015Q1〜2020Q4 で欠損がある変数 ({len(missing)}):", missing)
    if "ERRPFU" in D:
        print("ERRPFU/GDP (2015–2019 平均):", round((D["ERRPFU"] / D["GDP"]).loc["2015Q1":"2019Q4"].mean(), 4))
    cols = [c for c in ["GDP", "FUELV", "POILD", "LF", "UR", "CUX", "M2CD", "RCD", "KFPV", "RRFPV",
                        "IGVR", "ITR", "ETT", "SBGV", "SBCV"] if c in D]
    print(D.loc["2017Q4":"2020Q4", cols].round(4).to_string())


if __name__ == "__main__":
    main()
