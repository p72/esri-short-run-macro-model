"""変数台帳（Issue #3）: 論文の変数定義と、実際に使った系列・加工・単位換算・欠損処理を突き合わせる.

入力: reference/rn72_2022model.txt（src/fetch_paper.py の出力。付属資料II 変数名一覧）
      data/processed/model_data.csv, src/model.py
出力: data/processed/variable_ledger.csv（model_data の全列 × 台帳項目）

検査（失敗すると終了コード 1）:
  - model_data の全列に台帳の記載がある
  - 論文の変数が model_data に揃っている（無いものは一覧表示）
  - 内生変数の式番号が論文と一致する
  - 「定数」と記した変数が実際に定数で、値も一致する。記していない変数は定数でない（Issue #3 第4項）
  - data/raw の全ファイルがいずれかの変数から参照されている
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model as M  # noqa: E402
import vintage as VT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "reference" / "rn72_2022model.txt"
WINDOW = ("2010Q1", VT.END)   # model_data の期間
SOLVE = ("2017Q3", "2020Q4")    # simulate.py が解く期間

SOURCE_ABBR = {
    "BOJ,BPM": "日本銀行 国際収支統計", "BOJ,ESM": "日本銀行 金融経済統計月報", "BOJ,PIM": "日本銀行 企業物価指数",
    "CAO,SNA": "内閣府 国民経済計算・固定資本ストック速報", "IMF,IFS": "IMF International Financial Statistics",
    "IMF,WEO": "IMF World Economic Outlook", "JTA,SRTJ": "日本関税協会 外国貿易概況",
    "METI,ISM": "経済産業省 鉱工業生産指数", "SBSC,LFS": "総務省 労働力調査報告", "SBSC,BRR": "総務省 住民基本台帳",
    "MOF,MBFS": "財務省 財政金融統計月報", "MHLW,MLS": "厚生労働省 毎月勤労統計調査報告", "MHLW": "厚生労働省",
    "TSE,ASR": "東京証券取引所 統計月報", "JILPT,ULS": "労働政策研究・研修機構 ユースフル労働統計",
    "Author": "内閣府経済社会総合研究所作成", "不動産研究所": "日本不動産研究所", "日本証券業協会": "日本証券業協会",
}
UNITS = ["10億円", "2015年＝100", "2015年＝1", "2015年=1", "1万人", "1万", "10万円", "円／ドル", "ドル／バレル", "倍", "％"]
_Z2H = str.maketrans("ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ０１２３４５６７８９", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


# ---------------------------------------------------------------------------
# 論文 付属資料II のパース
# ---------------------------------------------------------------------------
def parse_paper(path: Path = PAPER) -> pd.DataFrame:
    """付属資料II（内生変数表・外生変数表）→ variable, paper_role, eq_paper, paper_name, paper_unit, paper_source."""
    lines = path.read_text(encoding="utf-8").splitlines()
    i0 = next(i for i, ln in enumerate(lines) if "（付属資料II）" in ln and "変数名一覧" in ln)
    i1 = next(i for i in range(i0, len(lines)) if "（付属資料 III" in lines[i] or "（付属資料III" in lines[i])
    skip_prefix = ("===== PAGE", "ESRI Research Note", "短期日本経済マクロ計量モデル", "記　号", "式番号",
                   "名　　　称", "単　位", "出　所")
    sym_re = re.compile(r"^[A-Z][A-Z0-9_]*$")
    rows, group, role, prev_title = [], [], None, False
    def flush() -> None:
        if not group:
            return
        sym, rest = group[0], group[1:]
        if not rest or rest[-1] not in SOURCE_ABBR:
            raise ValueError(f"付属資料II の {sym} の出所が読めません: {rest}")
        src, rest = rest[-1], rest[:-1]
        eq = ""
        if role == "endog":
            m = re.match(r"^(\d+(?:,\s*\d+)*)\s*(\S.*)?$", rest[0])
            if m is None:
                raise ValueError(f"付属資料II の {sym} の式番号が読めません: {rest[0]!r}")
            eq = m.group(1).replace(" ", "")
            rest = ([m.group(2)] if m.group(2) else []) + rest[1:]
        name, unit = " ".join(rest), ""
        if len(rest) == 2 and rest[1] in UNITS:
            name, unit = rest
        else:
            for u in UNITS:  # 単位が名称の行末に連結している場合（PRTNF, RLAND）
                if name.endswith(u) and len(name) > len(u):
                    name, unit = name[:-len(u)], u
                    break
        rows.append(dict(variable=sym, paper_role=role, eq_paper=eq, paper_name=name.strip(),
                         paper_unit=unit.replace("＝", "="), paper_source=src))
    for ln in lines[i0:i1]:
        ln = ln.strip().translate(_Z2H)
        if not ln:
            continue
        if ln.startswith("(1)"):
            role = "endog"
            continue
        if ln.startswith("(2)"):
            flush(); group = []
            role = "exog"
            continue
        if role is None:
            continue
        if ln.startswith(skip_prefix):
            prev_title = ln.startswith("短期日本経済マクロ計量モデル")
            continue
        if prev_title and ln.isdigit():  # ページ番号
            prev_title = False
            continue
        prev_title = False
        if sym_re.match(ln) and ln not in SOURCE_ABBR:
            flush()
            group = [ln]
        elif group:
            group.append(ln)
    flush()
    return pd.DataFrame(rows).set_index("variable")


# ---------------------------------------------------------------------------
# 実際に使った系列・加工（build_data.py / sna.py / fetch_*.py を要約）
# ---------------------------------------------------------------------------
V = VT.V
ANN = f"{V['annual']}年度年次推計"
RAWDIR = f"data/raw/{V['dir'].name}" if V["dir"] != VT.RAW else "data/raw"
EXT = pd.Period(VT.END, "Q") > pd.Period("2022Q4", "Q")   # 2022年以降に延長した版（稼働率・貿易・米国PPI・労働力調査の追加ファイルを使う）
LFS_FILE = f"{RAWDIR}/estat_lfs_000031831358.xlsx" if EXT else "data/raw/estat_lfs_000040115411.xlsx"
BASEYEAR = "2020" if EXT else "2015"   # QE の基準年（2024年版は2020年基準）

# (変数, 区分, 使用した系列・出所, 原ファイル, 加工, 単位換算, 欠損処理, 定数値, 関連Issue)
ENTRIES: list[tuple] = [
    ("GDP CP IHP IFP INP CG IG ING BF XGS MGS", "論文どおり",
     f"ESRI {BASEYEAR}年基準 四半期GDP速報 実質（連鎖）季節調整系列 {V['label']}",
     f"{RAWDIR}/gaku-jk{V['qe']}.csv", "そのまま", "10億円・年率（公表単位のまま）", "なし", None, ""),
    ("KAISA", "論文にない補助変数",
     "ESRI 四半期GDP速報 実質 開差（連鎖方式の不突合）", f"{RAWDIR}/gaku-jk{V['qe']}.csv",
     "式2（実質GDPの定義式）を実績で閉じるために使用", "10億円・年率", "なし", None, ""),
    ("GDPV CPV IHPV IFPV INPV CGV IGV INGV BFV XGSV MGSV RTRIV PTRIV", "論文どおり",
     f"ESRI {BASEYEAR}年基準 四半期GDP速報 名目 季節調整系列 {V['label']}",
     f"{RAWDIR}/gaku-mk{V['qe']}.csv", "そのまま", "10億円・年率", "なし", None, ""),
    ("PGDP PCP PIHP PIFP PCG PIG PXGS PMGS", "論文どおり",
     "ESRI 四半期GDP速報 デフレーター（季調系列）", f"{RAWDIR}/def-qk{V['qe']}.csv, gaku-mk{V['qe']}.csv, gaku-jk{V['qe']}.csv",
     "sna.py が def-qk を読むが、build_data.py で名目÷実質に置き換える（公表値と一致、丸め誤差を避ける）", f"{BASEYEAR}年=100 → {BASEYEAR}年=1", "なし", None, ""),
    ("YWV", "同種統計・加工差",
     "ESRI 四半期GDP速報 雇用者報酬 名目 季節調整系列（kshotoku 表）", f"{RAWDIR}/kshotoku-q{V['qe']}.csv",
     "公式季調値をそのまま使用（自前の移動平均比率法との差 0.15%）", "10億円・年率", "公表のない四半期は年次推計の原系列を自前で季調", None, ""),
    ("YWIV YOLIV YIGV YICV CCAVG TCIV TCSTV SUBV CSSV TYPV YDV BSSV", "同種統計・加工差",
     f"ESRI {ANN} 国民所得の分配（qom2）・一般政府 所得支出勘定（i4）・家計 所得支出勘定（i5）の四半期原系列",
     f"{RAWDIR}/{V['annual']}qom2_jp.xlsx, {V['annual']}i4_jp.xlsx, {V['annual']}i5_jp.xlsx",
     "移動平均比率法（2×4中心化移動平均、季節因子は5年移動平均）で季調。YIGV・SUBV は加法型、YICV は加法型",
     "10億円（四半期原系列）×4 → 年率", "なし（論文と同じ版は 2021Q4 まで）", None, ""),
    ("TYCV", "同種統計・加工差",
     f"ESRI {ANN} 一般政府 所得・富等に課される経常税（受取）− 家計 同（支払）", f"{RAWDIR}/{V['annual']}i4_jp.xlsx, {V['annual']}i5_jp.xlsx",
     "差をとってから移動平均比率法で季調", "10億円×4 → 年率", "なし", None, "#3（ESRI の季調との四半期パターン差）"),
    ("OITAXV", "同種統計・加工差",
     f"ESRI {ANN} 一般政府 生産・輸入品に課される税 − 付加価値型税 − 輸入関税", f"{RAWDIR}/{V['annual']}i4_jp.xlsx",
     "差をとってから移動平均比率法で季調", "10億円×4 → 年率", "なし", None, ""),
    ("ITAXV", "定義式", "TCIV + TCSTV + OITAXV（式133）", "", "", "10億円・年率", "なし", None, ""),
    ("TAXV", "定義式", "TYPV + TYCV + ITAXV（式128）", "", "", "10億円・年率", "なし", None, ""),
    ("CCAV", "同種統計・加工差",
     "GDPV − ITAXV + SUBV + 海外からの純所得（NFIV）− 国民所得（NIV、年次推計の四半期原系列を季調）の残差",
     f"{RAWDIR}/gaku-mk{V['qe']}.csv, {V['annual']}qom2_jp.xlsx",
     "暦年平均を保ちながら年央値を3次スプラインで補間（季調残差が法人企業所得に乗らないようにする）", "10億円・年率", "なし", None, ""),
    ("YIEV", "定義の解釈",
     f"ESRI {ANN} 家計 財産所得（純、受取）− うち賃貸料", f"{RAWDIR}/{V['annual']}qom2_jp.xlsx",
     "賃貸料を除いた金融的財産所得（利子・配当・その他）。論文乗数表から逆算した法人企業所得の水準と式104の定式化から特定",
     "10億円×4 → 年率", "なし", None, ""),
    ("YIV", "定義式", "YIEV + YIGV（式105）", "", "", "10億円・年率", "なし", None, ""),
    ("YCV", "定義の解釈",
     f"ESRI {ANN} 企業所得（法人＋公的企業＋個人企業）− 個人企業所得 + 家計賃貸料 + 対家計民間非営利団体財産所得",
     f"{RAWDIR}/{V['annual']}qom2_jp.xlsx",
     "各項を加法型で季調して合成。論文乗数表から逆算した ESRI の四半期経路と相関0.96・水準比0.998", "10億円×4 → 年率", "なし", None,
     "#3（季調パターン差）"),
    ("ENTV RENTHH", "中間系列（モデル未使用）",
     f"ESRI {ANN} 企業所得、家計財産所得のうち賃貸料", f"{RAWDIR}/{V['annual']}qom2_jp.xlsx",
     "YCV・YIEV の作成に使用。model.py からは参照されない", "10億円×4 → 年率", "なし", None, ""),
    ("NIV", "定義式", "YWV + YIV + YICV + YCV（式83。年次推計の国民所得を直接季調すると法人企業所得に季節性が残るため定義式で作る）",
     "", "", "10億円・年率", "なし", None, ""),
    ("SDV", "残差", "GDPV − CCAV − ITAXV + SUBV + NFIV − NIV（式83・84を閉じる残差）", "",
     "CCAV を暦年で平滑化した分の四半期変動を含む。2018〜20年は GDP 比 −1〜+2%", "10億円・年率", "なし", None, "#3"),
    ("OTYDV", "残差", "YDV − (YWV + BSSV + YIEV + YICV − TYPV − CSSV)（式90を実績で閉じる）", "", "", "10億円・年率", "なし", None, ""),
    # ---- 暦・ダミー
    ("TIME", "暦", "1980Q1 = 1 の四半期通番（式110・111の定数から逆算）", "", "model.make_calendar", "四半期数", "なし", None, ""),
    ("TIME96Q2", "暦", "1996Q2 以降 1,2,3,… それ以前 0（生産関数の TFP トレンド屈折）", "", "model.make_calendar", "四半期数", "なし", None, ""),
    ("TIME70Q1", "暦", "1970Q1 = 1 の四半期通番", "", "model.make_calendar", "四半期数", "なし", None, ""),
    ("D202 D203 D204 D954C963 D0703 D091C093 D091 D092 D9203C D084 D142 D151 D191 D972C D952 D984 D961", "ダミー",
     "推定式に現れる期間ダミー（名前の年・四半期で 1。C は以降継続、C+年四半期は区間）", "", "model.make_calendar", "0/1",
     "なし", "名前の期間外は 0", ""),
    ("DTCIC2", "ダミー（論文どおり）", "消費税率引上げダミー 1997Q2–1998Q1, 2014Q2–2015Q1, 2019Q4–2020Q3", "",
     "model.make_calendar", "0/1", "なし", None, ""),
    # ---- 消費税
    ("RTCI", "論文どおり", "消費税 標準税率（2010Q1 5%, 2014Q2 8%, 2019Q4 10%）", "", "階段関数（手入力）",
     "% → 小数（0.05 など）", "なし", None, ""),
    ("PRTCP PRTIF PRTIH PRTGP PRTIG PRTCG PRTFU PRTNF PRTMG", "逆算（Author 系列の代替）",
     "論文 乗数表(6) 消費税率+1%pt の 2018Q1 デフレーター反応から逆算した転嫁率", "output/published_multipliers.csv",
     "全期間一定。PRTMG は論文にない（輸入デフレーターの転嫁率、0）", "小数", "なし", "CONST", "#3"),
    ("PRTNP", "逆算（Author 系列の代替）", "= PRTGP（企業物価の転嫁率を在庫デフレーターに流用）", "", "全期間一定", "小数", "なし", "CONST", "#3"),
    ("PRTNG", "逆算（Author 系列の代替）", "= PRTIG（公的固定資本形成の転嫁率を公的在庫に流用）", "", "全期間一定", "小数", "なし", "CONST", "#3"),
    ("PCPAT PIFPAT PIHPAT PCGAT PIGAT", "定義式", "デフレーター ÷ (1 + RTCI × 転嫁率)（式56,58,61,63,64）", "", "", "2015年=1", "なし", None, ""),
    ("PINPAT", "定義式", "PINP ÷ (1 + RTCI × PRTNP)（式62）", "", "", "2015年=1", "なし", None, ""),
    ("PGDPAT", "定義式", "式26 の名目GDP（消費税除く）÷ 実質GDP（式55）", "", "", "2015年=1", "なし", None, ""),
    ("CGPI", "論文どおり", "日本銀行 国内企業物価指数 総平均（PR01/PRCG20_2200000000）", "data/raw/boj_series.csv",
     "月次→四半期平均→2015年平均=1", "2015年=100 → 2015年=1", "なし", None, ""),
    ("CGPIAT", "論文どおり", "日本銀行 国内企業物価指数 総平均 消費税を除く（PR01/PRCG20_1200000000）", "data/raw/boj_series.csv",
     "月次→四半期平均→2015年平均=1", "2015年=100 → 2015年=1", "なし", None, ""),
    ("PINP", "代用", "= CGPI（民間在庫デフレーターは QE で公表されないため企業物価で代用）", "", "", "2015年=1", "なし", None, "#3"),
    ("PING", "代用", "= PIG（公的在庫デフレーターを公的固定資本形成デフレーターで代用）", "", "", "2015年=1", "なし", None, "#3"),
    ("ERRINPV", "残差（Author 系列の代替）", "INPV − INP × PINP（式17を実績で閉じる）", "", "", "10億円・年率", "なし", None, ""),
    ("ERRINGV", "残差（Author 系列の代替）", "INGV − ING × PING（式20を実績で閉じる）", "", "", "10億円・年率", "なし", None, ""),
    # ---- 輸入の燃料分割・為替・原油
    ("PFUEL", "代用", "日本銀行 輸入物価指数 円ベース 石油・石炭・天然ガス（PR01/PRCG20_2600520001）。論文は SNA の鉱物性燃料輸入デフレーター",
     "data/raw/boj_series.csv", "月次→四半期平均→2015年平均=1", "2015年=100 → 2015年=1", "なし", None, "#3"),
    ("FUELV", "同種統計・加工差", "財務省 普通貿易統計（e-Stat）鉱物性燃料 輸入額 全国合計（論文は日本関税協会 外国貿易概況）",
     "data/raw/estat_trade_fuel_0003228199.csv, estat_trade_fuel_0003313968.csv" + (", estat_trade_fuel_0003425296.csv" if EXT else ""),
     "月次→四半期合計→移動平均比率法で季調", "千円 ÷1e6 ×4 → 10億円・年率", "なし", None, ""),
    ("FUEL", "定義式", "FUELV ÷ PFUEL（式11）", "", "", "10億円・年率", "なし", None, ""),
    ("NFMGSV NFMGS PNFMGS", "定義式", "MGSV − FUELV、MGS − FUEL、NFMGSV ÷ NFMGS（式25,12,82。連鎖の非加法性は無視）", "", "", "10億円・年率／2015年=1", "なし", None, ""),
    ("PFUELAT PNFMGSAT PMGSAT", "定義式", "= PFUEL, PNFMGS, PMGS（式67,68,66。輸入の転嫁率 PRTFU・PRTNF は 0）", "", "", "2015年=1", "なし", None, ""),
    ("FXS", "論文どおり", "日本銀行 東京市場 ドル・円スポット 17時時点 月中平均（FM08/FXERM07）", "data/raw/boj_series.csv",
     "月次→四半期平均", "円/ドル", "なし", None, ""),
    ("POILD", "同種統計・加工差", "財務省 普通貿易統計（e-Stat）原油及び粗油 輸入金額÷数量（論文は日本関税協会 外国貿易概況）",
     "data/raw/estat_trade_crude_0003228199.csv, estat_trade_crude_0003313968.csv" + (", estat_trade_crude_0003425296.csv" if EXT else ""),
     "月次→四半期合計で単価", "千円/kL ×1000 → 円/kL ÷ FXS → ドル/kL ÷ 6.2898 → ドル/バレル", "なし", None, ""),
    # ---- 労働
    ("LF LE LW UR", "論文どおり", "総務省 労働力調査 長期時系列表1-a-1 主要項目 月次 季節調整値（e-Stat）",
     LFS_FILE, "月次→四半期平均", "万人・%（公表単位のまま）", "なし", None, ""),
    ("POP", "論文どおり", "総務省 労働力調査 労働力人口 + 非労働力人口（15歳以上人口）", LFS_FILE,
     "月次季調値の和→四半期平均", "万人", "なし", None, ""),
    ("RLEW", "定義式（逆算）", "LW ÷ LE（式53 を実績で閉じる）", "", "", "小数", "なし", None, ""),
    ("POP65", "同種統計・加工差", "総務省 労働力調査 基本集計 15歳以上人口 65歳以上 四半期（e-Stat 0003005798 系）",
     "data/raw/estat_lfs_pop65_q.csv", "四半期原系列（季調なし）", "万人",
     "2018Q1 より前は POP × (2018Q1 の POP65/POP 比) で延長（式47・140 は当期値のみ使用）", None, "#3"),
    ("HH", "代用", "= POP（15歳以上人口）。論文は住民基本台帳の世帯数", "", "", "万人（論文は1万世帯）", "なし", None, "#3"),
    ("CUX", "論文どおり", "経済産業省 製造工業 稼働率指数 季節調整済（e-Stat。2015年基準" + ("、2018年1月以降は2020年基準を重複期間の平均比で接続" if EXT else "") + "）",
     "data/raw/estat_cux_2015base.csv" + (", estat_cux_2020base.csv" if EXT else ""),
     "月次→四半期平均", "2015年=100（論文の単位表記は %）", "なし", None, ""),
    ("LHX", "同種統計・加工差", "厚生労働省 毎月勤労統計 長期時系列表29 総実労働時間指数 事業所規模5人以上 調査産業計 就業形態計 原指数",
     "data/raw/estat_maikin_t29_hours_5plus.xls", "四半期原指数→移動平均比率法で季調→2015年平均=100", "2015年=100", "なし", None, ""),
    ("WI W", "定義式", "YWIV ÷ LW、YWV ÷ LW（式69, 95）", "", "", "10億円/万人 = 10万円", "なし", None, ""),
    ("WPH", "定義式", "100 × W ÷ LHX（式96）", "", "", "10万円", "なし", None, ""),
    # ---- 金融
    ("M2CD", "同種統計・加工差", "日本銀行 マネーストック M2 平均残高（MD02/MAM1NAM2M2MO）。M2+CD（MD02/MAMS3ANM2C、2008年4月で公表終了）に接続",
     "data/raw/boj_series.csv", "月次→四半期平均。重複期間（2003/04〜2008/04）の平均比 1.0050 を M2 に乗じる", "億円 ÷10 → 10億円", "なし", None, "#3"),
    ("RCD", "論文どおり", "日本銀行 譲渡性預金 平均金利 新規発行分 総合（FM02/STRAK_STRACDN2DB）", "data/raw/boj_series.csv",
     "月次→四半期平均。式113 の下限 0.001 で切り上げ（マイナス金利期）", "%", "なし", None, ""),
    ("RCDX", "定義式", "= RCD（式114 の実績値。式113 RCD = max(0.001, RCDX)）", "", "", "%", "なし", None, ""),
    ("RGB", "同種統計", "10年国債 新発債流通利回り 月平均（ESRI 景気動向指数 先行系列の採用系列、原資料は日本証券業協会・日本相互証券）",
     "data/raw/esri_ci1_0907.xlsx → external_series.csv", "月次→四半期平均。式115 の下限 0.001 で切り上げ", "%", "なし", None, ""),
    ("RGBX", "定義式", "= RGB（式116 の実績値）", "", "", "%", "なし", None, ""),
    ("PSHARE", "論文どおり", "東証株価指数 TOPIX 月平均（ESRI 景気動向指数 先行系列の採用系列、原資料は東京証券取引所）",
     "data/raw/esri_ci1_0907.xlsx → external_series.csv", "月次→四半期平均→2015年平均=1", "ポイント → 2015年=1", "なし", None, ""),
    # ---- ストック
    ("KFPV", "同種統計・加工差", "ESRI 国民経済計算 固定資産（名目、暦年末）民間部門（非金融法人・金融機関の民間、家計、NPISH）合計 − 住宅",
     f"{RAWDIR}/{V['annual']}ss4n_jp.xlsx", "モデルの蓄積式 K_t = K_{t-1}·(P_t/P_{t-1})·(1−r) + I_t/4 で四半期化（r は暦年内一定、年末残高に一致するよう二分法）",
     "10億円", "なし", None, ""),
    ("KHPV", "同種統計・加工差", "ESRI 国民経済計算 固定資産 住宅（名目、暦年末）民間部門合計", f"{RAWDIR}/{V['annual']}ss4n_jp.xlsx",
     "同上（IHPV, PIHP で四半期化）", "10億円", "なし", None, ""),
    ("KGV", "同種統計・加工差", "ESRI 国民経済計算 固定資産（名目、暦年末）公的部門（非金融法人・金融機関の公的、一般政府）合計",
     f"{RAWDIR}/{V['annual']}ss4n_jp.xlsx", "同上（IGV, PIG で四半期化）", "10億円", "なし", None, ""),
    ("KNPV", "同種統計・加工差", "ESRI 国民資産・負債残高 在庫（暦年末）", f"{RAWDIR}/{V['annual']}ss1_jp.xlsx",
     "同上（INPV, PINP で四半期化）", "10億円", "なし", None, ""),
    ("RRFPV RRHPV RRKGV RRNPV", "逆算（Author 系列の代替）", "暦年末残高に一致させる年内一定の除却率（上の四半期化で二分法により決定）", "",
     "", "小数（四半期率）", "なし", None, ""),
    ("RRFP", "逆算（Author 系列の代替）", "= RRFPV（実質の除却率を名目と同じとする）", "", "", "小数", "なし", None, ""),
    ("RKGV RKG", "定義式", "RRKGV × KGV(-1) × 4、RKGV ÷ PIG(-1)（式46, 45）", "", "", "10億円・年率", "なし", None, ""),
    ("CCAVGR", "逆算（Author 系列の代替）", "(CCAVG ÷ PCG) ÷ RKG（一般政府固定資本減耗の公的固定資本除却比）", "", "", "小数", "なし", None, ""),
    ("CGXRKG", "残差（Author 系列の代替）", "CG − CCAVGR × RKG（式7 を実績で閉じる）", "", "", "10億円・年率", "なし", None, ""),
    # ---- 政府
    ("IGVR", "同種統計・加工差", "ESRI 国民経済計算 一般政府の部門別勘定（GFS）固定資産の純取得 + 固定資本減耗（年度）÷ 公的固定資本形成 IGV（年度平均）",
     f"{RAWDIR}/{V['annual']}s6_2_jp.xlsx", "年度値を年度内一定で四半期に割付", "小数", "年度値のない端の期間は前後の値で埋める（ffill/bfill）", None, ""),
    ("ITR", "定義式", "TYPV ÷ (YWV + BSSV + YIEV + YICV + OTYDV) の4期移動平均（式129 の税率定義）", "", "", "小数", "4期移動平均のため 2010Q1〜Q3 は欠損", None, "#2"),
    ("ITREQ", "逆算（Author 系列の代替）", "ITR の 2010Q4〜2019Q4 平均（全期間一定）", "", "", "小数", "なし", "CONST", "#2"),
    ("ETT", "定義式", "TYCV ÷ YCV の前期からの4期移動平均（式132）", "", "", "小数", "4期移動平均のため 2010Q1〜Q4 は欠損", None, "#1"),
    ("TT", "論文どおり", "財務省 法人実効税率（国・地方、年度。2010年度 40.69% … 2018年度以降 29.74%）", "", "年度値を階段関数で手入力",
     "% → 小数", "なし", None, "#1"),
    ("RTCST", "定義式（逆算）", "TCSTV ÷ MGSV（式135 を実績で閉じる）", "", "", "小数", "なし", None, ""),
    ("SR", "論文どおり", "厚生年金保険料率（総報酬ベース、毎年9月改定。2017年9月以降 18.3%）", "", "階段関数で手入力（改定は第4四半期から）", "% → 小数", "なし", None, ""),
    ("IR", "仮定値", "所得代替率 = 1（式140 は log(IR·WI·POP65) で入り、定数は誤差項に吸収）", "", "全期間一定", "小数", "なし", "CONST", "#3"),
    ("SBGV", "定義の解釈", "日本銀行 資金循環統計 一般政府 資産合計 − 金融資産・負債差額（ストック、FF/FOF_FFAS420A900, FOF_FFAS420L700）= 負債",
     "data/raw/boj_series.csv", "四半期末残高そのまま。論文乗数表の累積赤字/GDP 比の初期変化から総債務ベース（GDP 比約231%）と判断",
     "億円 ÷10 → 10億円", "なし", None, ""),
    ("OTNGV", "仮定値", "一般政府財政バランス残余項目 = 0（式126 の残差は BGV に含まれる）", "", "全期間一定", "10億円", "なし", "CONST", "#3"),
    ("RSBGV", "残差（Author 系列の代替）", "SBGV − SBGV(-1) + BGV/4（式141 を実績で閉じる評価調整額）", "", "", "10億円", "2010Q1 は欠損（ラグ）", None, ""),
    # ---- 海外
    ("BCV", "代用", "BFV + RTRIV − PTRIV（SNA ベースの純輸出＋海外との所得の純受取）。論文は国際収支統計の経常収支（第二次所得収支を含む）", "",
     "式148 で ERRBCV = 0 として算出", "10億円・年率", "なし", None, "#3"),
    ("ERRBCV", "仮定値", "経常収支誤差項 = 0", "", "全期間一定", "10億円", "なし", "CONST", "#3"),
    ("FASSTV", "同種統計・加工差", "ESRI 国民経済計算 対外資産（暦年末）", f"{RAWDIR}/{V['annual']}ss5_jp.xlsx", "暦年末値を対数線形で四半期補間", "10億円", "なし", None, ""),
    ("SBCV", "同種統計・加工差", "ESRI 国民経済計算 対外純資産（暦年末）", f"{RAWDIR}/{V['annual']}ss5_jp.xlsx", "暦年末値を線形で四半期補間", "10億円", "なし", None, ""),
    ("FLIABV", "定義式", "FASSTV − SBCV（式150）", "", "", "10億円", "なし", None, ""),
    ("RSBCV", "逆算（Author 系列の代替）", "(SBCV − BCV/4) ÷ SBCV(-1)（式149 を実績で閉じる累積経常収支調整項）", "", "", "小数", "2010Q1 は欠損（ラグ）", None, ""),
    ("US_RGB", "同種統計", "OECD Financial market statistics 米国 長期金利 IRLT（10年国債利回り、月次）。論文は IMF IFS",
     "data/raw/oecd_DSD_STES_DF_FINMARK_USA_M_IRLT_PA_____.csv → external_series.csv", "月次→四半期平均", "%", "なし", None, ""),
    ("US_WPI", "同種統計",
     ("FRED 米国製造業 生産者物価指数 PCUOMFGOMFG（BLS、月次）。論文は IMF IFS" if EXT else
      "OECD Key short-term economic indicators 米国 生産者物価 PP（製造業 activity=C、月次、2015年=100）。論文は IMF IFS"),
     ("data/raw/fred_PCUOMFGOMFG.csv" if EXT else "data/raw/oecd_DSD_KEI_DF_KEI_USA-G7-OECD_M_PP_IX___.csv → external_series.csv"),
     "月次→四半期平均" + ("→2015年平均=100" if EXT else ""),
     "2015年=100 のまま（論文は 2015年=1。式147 には対数比で入り、水準差は誤差項に吸収）",
     "なし" if EXT else "2023年以降は OECD 未公表（モデル期間外）", None, "#3"),
    ("WD_PX WD_PI", "代用",
     "= 米国 生産者物価（US_WPI と同じ系列）。論文は競争国の輸出・輸入価格の加重平均（IMF IFS）。" + ("OECD の系列が2022年で終了したため FRED の米国PPIで代用" if EXT else "OECD の G7 集計が取得できず米国で代用"),
     ("data/raw/fred_PCUOMFGOMFG.csv" if EXT else "data/raw/oecd_DSD_KEI_DF_KEI_USA-G7-OECD_M_PP_IX___.csv → external_series.csv"),
     "月次→四半期平均", "2015年=100（式9, 65, 80 には対数差・比で入る）",
     "なし" if EXT else "2023年以降は OECD 未公表（モデル期間外）", None, "#3"),
    ("WD_YVI", "代用", "OECD Quarterly National Accounts OECD 加盟国計 実質GDP 指数（2015年=100、季調）。論文は世界GDP（日本除く、IMF WEO）",
     "data/raw/oecd_DSD_NAMAIN1_DF_QNA_Q_Y_OECD_S1_S1_B1GQ__Z__Z__Z____.csv → external_series.csv", "四半期そのまま（日本を含む）", "2015年=100", "なし", None, "#3"),
    # ---- 資産
    ("LANDT", "同種統計・加工差", "ESRI 国民資産・負債残高 土地（暦年末、国全体）", f"{RAWDIR}/{V['annual']}ss1_jp.xlsx", "暦年末値を線形で四半期補間", "10億円", "なし", None, ""),
    ("LANDV", "同種統計・加工差", "ESRI 家計（個人企業を含む）期末貸借対照表 土地（暦年末）", f"{RAWDIR}/{V['annual']}si4_jp.xlsx", "暦年末値を線形で四半期補間", "10億円", "なし", None, ""),
    ("PROLA", "定義式（逆算）", "LANDT ÷ LANDV（式102 を実績で閉じる。論文の「家計保有分比率」の逆数にあたる定義）", "", "", "小数（>1）", "なし", None, ""),
    ("PLAND", "代用", "LANDT を 2015年平均=1 に指数化。論文は日本不動産研究所 市街地価格指数", "", "土地残高の変化率を地価の変化率とみなす", "2015年=1", "なし", None, "#3"),
    ("RLAND", "定義式（逆算）", "LANDV ÷ PLAND（式103 を実績で閉じる）", "", "", "10億円/指数", "なし", None, ""),
    ("SHARETV", "定義の解釈", "ESRI 国民資産・負債残高 負債側 うち株式（居住者発行残高、暦年末）。資産側（保有分）だと株価収益率の水準が論文の含意より約2割低い",
     f"{RAWDIR}/{V['annual']}ss1_jp.xlsx", "暦年末値を線形で四半期補間", "10億円", "なし", None, "#1"),
    ("RSHARET", "定義式（逆算）", "SHARETV ÷ PSHARE（式100 を実績で閉じる）", "", "", "10億円/指数", "なし", None, ""),
    ("SHAREV", "同種統計", "日本銀行 資金循環統計 家計 株式等（ストック、FF/FOF_FFAS430A330）。論文は SNA の家計保有株式（si4 期末貸借対照表にも同項目あり、未使用）",
     "data/raw/boj_series.csv", "四半期末残高そのまま", "億円 ÷10 → 10億円", "なし", None, "#3"),
    ("RSHARE", "定義式（逆算）", "SHAREV ÷ PSHARE（式101 を実績で閉じる）", "", "", "10億円/指数", "なし", None, ""),
    # ---- 資本コスト
    ("REQU", "仮定値", "自己資本比率 = 0.40 で全期間一定（法人企業統計の自己資本比率は概ね 40%）。論文は Author", "", "定数", "小数（論文は %）", "なし", "CONST", "#1 #3"),
    ("SLRATIO", "仮定値", "短期・長期負債比率 = 0.50 で全期間一定。論文は財務省 財政金融統計月報（法人企業統計）", "", "定数", "小数（論文は %）", "なし", "CONST", "#1 #3"),
    ("TINCR", "仮定値", "投資税額控除率 = 0（論文は Author）", "", "定数", "小数", "なし", "CONST", "#3"),
    ("ROR", "仮定値", "株価平均利回り = 2.0% で全期間一定。論文は東証統計月報（東証一部 有配当会社）", "", "定数", "%", "なし", "CONST", "#1 #3"),
    # ---- 均衡値
    ("UREQ", "仮定値", "均衡失業率 = 3.0% で全期間一定（論文は JILPT ユースフル労働統計）。式48・50 に log(UR/UREQ) 等で入る", "", "定数", "%", "なし", "CONST", "#3"),
    ("CUXEQ", "逆算（Author 系列の代替）", "CUX の 2010Q1〜2019Q4 平均（全期間一定）", "", "定数", "2015年=100", "なし", "CONST", "#3"),
    ("LHXEQ", "逆算（Author 系列の代替）", "LHX の 13期中心化移動平均（端は最低7期）", "", "", "2015年=100", "端の期間は min_periods=7 で計算", None, "#3"),
    ("WPHXREQ", "逆算（Author 系列の代替）", "(WPH ÷ PCPAT) の 2010Q1〜2019Q4 平均（全期間一定）", "", "定数", "10万円（実質）", "なし", "CONST", "#3"),
    ("ERRPFU", "逆算（Author 系列の代替）", "式54（生産関数）を実績の稼働率で満たす残差の GDP 比（2015〜19年平均）× GDP", "",
     "四半期ごとの残差は稼働率のノイズを含むため GDP 比を一定にする", "10億円・年率", "なし", None, "#3"),
]

# 上で個別に書かない内生変数（水準型の定義式で実績から埋める）
DEFAULT_IDENTITY = ("定義式", "model.py 式{no} を実績値に当てはめて算出（build_data.fill_identities）", "", "", "", "なし", None, "")

MODEL_UNIT = {}
for names, u in [
    ("GDP CP IHP IFP INP CG IG ING BF XGS MGS KAISA GDPV CPV IHPV IFPV INPV CGV IGV INGV BFV XGSV MGSV RTRIV PTRIV "
     "YWV YWIV YOLIV YIGV YIEV YICV NIV ENTV RENTHH CCAVG TCIV TCSTV SUBV CSSV TYPV YDV BSSV TYCV OITAXV ITAXV CCAV TAXV YIV YCV SDV OTYDV "
     "ERRINPV ERRINGV FUELV FUEL NFMGSV NFMGS RKGV RKG CGXRKG BCV BGV ERRPFU GDPVEXCT CPVEXCT IFPVEXCT IHPVEXCT INPVEXCT CGVEXCT "
     "IGVEXCT INGVEXCT MGSVEXCT RFP RFPV RHPV RNPV GDPPOT YCVAT INPVA YLV YL", "10億円・季調年率"),
    ("KFPV KHPV KGV KNPV FASSTV FLIABV SBCV LANDT LANDV SHARETV SHAREV SBGV M2CD KFP KFPSTA KHP KNP KG NWCV NWTV FNWV KPV OTNGV ERRBCV RSBGV", "10億円（ストック・期末）"),
    ("PGDP PCP PIHP PIFP PCG PIG PXGS PMGS PCPAT PIFPAT PIHPAT PCGAT PIGAT PINPAT PGDPAT CGPI CGPIAT PINP PING PFUEL PNFMGS PFUELAT PNFMGSAT PMGSAT PSHARE PLAND", "2015年=1"),
    ("CUX CUXEQ LHX LHXEQ US_WPI WD_PX WD_PI WD_YVI", "2015年=100"),
    ("RCD RCDX RGB RGBX UR UREQ US_RGB ROR GDPD PGDPD GDPGAP", "%"),
    ("RTCI PRTCP PRTIF PRTIH PRTGP PRTIG PRTCG PRTFU PRTNF PRTMG PRTNP PRTNG SR TT ETT ITR ITREQ REQU SLRATIO TINCR RLEW IGVR RTCST CCAVGR PROLA RSBCV "
     "RRFPV RRHPV RRKGV RRNPV RRFP IR UCC UCCDB UCCDE UCCPF MK KNGDEQ PIFPATGR PIFPATSUM PVDP BGVATGDPV SBGVATGDPV BCVATGDPV SBCVATGDPV", "小数（比率）"),
    ("LF LE LW POP POP65 HH", "万人"),
    ("W WI WPH WIPH WPHXREQ", "10万円"),
    ("FXS", "円/ドル"), ("POILD", "ドル/バレル"), ("PERR", "倍"), ("RSHARET RSHARE RLAND", "10億円/指数"),
    ("TIME TIME96Q2 TIME70Q1", "四半期数"),
    ("D202 D203 D204 D954C963 D0703 D091C093 D091 D092 D9203C D084 D142 D151 D191 D972C D952 D984 D961 DTCIC2", "0/1"),
]:
    for n in names.split():
        MODEL_UNIT[n] = u
# 比率系のうち GDP 比は % 表示（乗数表と同じ）
for n in ["BGVATGDPV", "SBGVATGDPV", "BCVATGDPV", "SBCVATGDPV"]:
    MODEL_UNIT[n] = "%"


# ---------------------------------------------------------------------------
# 台帳の組み立てと検査
# ---------------------------------------------------------------------------
def build_ledger(data: pd.DataFrame, paper: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    problems: list[str] = []
    model = M.Model()
    eq_no = {e.name: e.no for e in model.eqs}
    eq_kind = {e.name: e.kind for e in model.eqs}
    import inspect
    used = set(re.findall(r'v\("([A-Z0-9_]+)"', inspect.getsource(M)))

    manual: dict[str, tuple] = {}
    for names, *rest in ENTRIES:
        for n in names.split():
            if n in manual:
                problems.append(f"台帳に {n} が二重に記載されています")
            manual[n] = tuple(rest)

    win = data.loc[WINDOW[0]:WINDOW[1]]
    solve = data.loc[SOLVE[0]:SOLVE[1]]
    rows = []
    for var in data.columns:
        if var in manual:
            status, source, raw, transform, unit_conv, missing, const, issue = manual[var]
        elif var in eq_no and eq_kind[var] == "level":
            status, source, raw, transform, unit_conv, missing, const, issue = DEFAULT_IDENTITY
            source = source.format(no=eq_no[var])
        else:
            problems.append(f"{var}: 台帳に記載がなく、水準型の定義式でもありません")
            status = source = raw = transform = unit_conv = missing = issue = ""
            const = None
        role = "内生" if var in eq_no else ("外生" if var in used else "補助（未使用）")
        s = win[var]
        nun = s.dropna().nunique()
        is_const = nun == 1
        const_val = float(s.dropna().iloc[0]) if is_const else np.nan
        if const == "CONST" and not is_const:
            problems.append(f"{var}: 台帳では定数だが実データは {nun} 通りの値をとる")
        if const is None and is_const and status not in ("ダミー", "ダミー（論文どおり）"):
            problems.append(f"{var}: 台帳に定数と書いていないが実データは定数 {const_val}")
        p = paper.loc[var] if var in paper.index else None
        if p is not None and var in eq_no and p["eq_paper"] and str(eq_no[var]) not in p["eq_paper"].split(","):
            problems.append(f"{var}: 式番号が論文 {p['eq_paper']} と model.py {eq_no[var]} で異なります")
        rows.append(dict(
            variable=var,
            role=role,
            eq_model=eq_no.get(var, ""),
            eq_paper=p["eq_paper"] if p is not None else "",
            paper_role={"endog": "内生", "exog": "外生"}.get(p["paper_role"], "") if p is not None else "論文にない",
            paper_name=p["paper_name"] if p is not None else "",
            paper_unit=p["paper_unit"] if p is not None else "",
            paper_source=p["paper_source"] if p is not None else "",
            paper_source_full=SOURCE_ABBR.get(p["paper_source"], "") if p is not None else "",
            model_unit=MODEL_UNIT.get(var, ""),
            status=status,
            source_used=source,
            raw_file=raw,
            transform=transform,
            unit_conversion=unit_conv,
            missing_handling=missing,
            constant_value=const_val if is_const else "",
            first_valid=str(s.first_valid_index() or ""),
            last_valid=str(s.last_valid_index() or ""),
            n_missing_window=int(s.isna().sum()),
            n_missing_solve_period=int(solve[var].isna().sum()) if solve[var].size else 0,
            issue=issue,
        ))
        if var not in MODEL_UNIT:
            problems.append(f"{var}: model_unit が未設定")
    ledger = pd.DataFrame(rows).set_index("variable")

    missing_in_data = [v for v in paper.index if v not in data.columns]
    if missing_in_data:
        problems.append(f"論文の変数が model_data にありません: {missing_in_data}")
    return ledger, problems


def check_raw_files(ledger: pd.DataFrame) -> list[str]:
    """data/raw の全ファイルが台帳から参照されているか（未参照ファイルは棚卸し漏れ）."""
    referenced = set()
    for txt in ledger["raw_file"]:
        for part in re.split(r"[,、→]\s*", str(txt)):
            name = part.strip().split("/")[-1].split(" ")[0]
            if name:
                referenced.add(name)
    # 台帳は現在の版で書かれるので、もう一方の版の SNA ファイル（src/vintage.py, sna.py が読む）も参照済みとみなす
    for v in VT.VINTAGES.values():
        referenced |= {f"{s}{v['qe']}.csv" for s in ["gaku-jk", "gaku-mk", "def-qk", "kshotoku-q"]}
        referenced |= {f"{v['annual']}{t}_jp.xlsx" for t in ["qom2", "i4", "i5", "ss4n", "ss1", "ss5", "si4", "s6_2"]}
        referenced.add(f"{v['annual']}s12n_jp.xlsx")  # 家計の目的別消費（plot_food_tax_cut_vs_benefit.py の食料品支出）
    unref = []
    for f in sorted((ROOT / "data" / "raw").rglob("*")):
        if f.is_file() and not f.name.startswith("_") and f.name not in referenced \
                and not (f.name.startswith("boj_") and f.suffix == ".json"):  # fetch_boj.py の生JSON（boj_series.csv に集約）
            unref.append(str(f.relative_to(ROOT)))
    return unref


def main() -> None:
    if not PAPER.exists():
        raise SystemExit(f"{PAPER} がありません。先に python src/fetch_paper.py を実行してください")
    paper = parse_paper()
    data = pd.read_csv(VT.processed("model_data.csv"), index_col="period")
    data.index = pd.PeriodIndex(data.index, freq="Q")
    ledger, problems = build_ledger(data, paper)
    out = VT.processed("variable_ledger.csv")
    ledger.to_csv(out, encoding="utf-8-sig")

    print(f"論文 付属資料II: 内生 {(paper.paper_role == 'endog').sum()} 変数、外生 {(paper.paper_role == 'exog').sum()} 変数")
    print(f"台帳: {len(ledger)} 変数 → {out}")
    print("\n区分別の変数数:")
    print(ledger["status"].value_counts().to_string())
    extra = ledger.index[ledger["paper_role"] == "論文にない"].tolist()
    print(f"\n論文にない変数（{len(extra)}）: {extra}")
    print("\n論文と定義・出所が異なる変数（代用・仮定値・定義の解釈）:")
    sub = ledger[ledger["status"].str.contains("代用|仮定値|定義の解釈")]
    print(sub[["paper_name", "paper_source", "status", "source_used", "issue"]].to_string())
    unref = check_raw_files(ledger)
    if unref:
        print(f"\nどの変数からも参照されていない data/raw のファイル（{len(unref)}、版の比較・調査用に取得したもの）:")
        for f in unref:
            print("  ", f)
    if problems:
        print("\n検査で見つかった問題:")
        for p in problems:
            print(" -", p)
        raise SystemExit(1)
    print("\n検査: すべて通過（記載漏れなし、式番号一致、定数の記載と実データが一致）")


if __name__ == "__main__":
    main()
