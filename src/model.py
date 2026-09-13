"""ESRI 短期日本経済マクロ計量モデル（2022年版）の Python 実装.

出典: 酒巻・鈴木・中尾・北川・符川・仲島・堀 (2022)
      「短期日本経済マクロ計量モデル(2022年版)の構造と乗数分析」ESRI Research Note No.72
      付属資料III 方程式体系（152本、うち推定式47本）

変数名は論文の ``JA_`` 接頭辞を外したもの（WD_*, US_*, PRT* はそのまま）。
各式は「変換後の左辺 = 右辺 + 誤差項(アドファクター)」の形で持ち、
実績値から逆算した誤差項を足し戻すことで標準解＝実績値を再現する。

論文記載からの修正点（誤植と判断したもの）は各式のコメントに ``[fix]`` で明記。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

ln = math.log
ex = math.exp

# 4) 生産関数のパラメータ
GAM = -0.634751
RAM1 = 0.003502
RAM2 = -0.001466
BETA = 0.597440
AVRDL = BETA
FXS2015 = 121.04  # 2015年平均 円/ドル（対数内の定数なので乗数には影響しない）

# 式129 所得実効税率: 論文の印刷どおり水準式にすると乗数表(1)の TYPV 経路と合わない。
# DLOG 型（誤差修正型）と解釈すると四半期乗数とほぼ一致するため "dlog" を既定とする。
ITR_FORM = "dlog"

Getter = Callable[..., float]


@dataclass
class Eq:
    no: int
    name: str
    kind: str  # "level" | "log" | "dlog" | "d" | "custom"
    rhs: Callable[[Getter], float]
    lhs: Callable[[Getter], float] | None = None  # custom のみ
    inv: Callable[[Getter, float], float] | None = None  # custom のみ

    def lhs_value(self, v: Getter) -> float:
        n = self.name
        if self.kind == "level":
            return v(n)
        if self.kind == "log":
            return ln(v(n))
        if self.kind == "dlog":
            return ln(v(n)) - ln(v(n, 1))
        if self.kind == "d":
            return v(n) - v(n, 1)
        return self.lhs(v)

    def invert(self, v: Getter, y: float) -> float:
        n = self.name
        if self.kind == "level":
            return y
        if self.kind == "log":
            return ex(y)
        if self.kind == "dlog":
            return v(n, 1) * ex(y)
        if self.kind == "d":
            return v(n, 1) + y
        return self.inv(v, y)


# ---------------------------------------------------------------------------
# 補助関数（k はラグ次数。負ならリード）
# ---------------------------------------------------------------------------
def dl(v: Getter, n: str, k: int = 0) -> float:
    return ln(v(n, k)) - ln(v(n, k + 1))


def dd(v: Getter, n: str, k: int = 0) -> float:
    return v(n, k) - v(n, k + 1)


def ydr(v, k):  # 実質可処分所得
    return v("YDV", k) / v("PCP", k)


def lshare(v, k):  # 労働分配率（雇用者報酬＋個人企業所得）/（国民所得＋固定資本減耗）
    return (v("YWV", k) + v("YICV", k)) / (v("NIV", k) + v("CCAV", k))


def typ_base(v, k):  # 個人所得税の課税ベース
    return v("YWV", k) + v("BSSV", k) + v("YIEV", k) + v("YICV", k) + v("OTYDV", k)


def tci_base(v):  # 消費税の課税ベース
    r = v("RTCI")
    return (r / (1 + r * v("PRTCP")) * v("CPV")
            + r / (1 + r * v("PRTIF")) * v("IFPV")  # [fix] 論文 "1+JA_RTCI+PRTIF"
            + r / (1 + r * v("PRTIH")) * v("IHPV")
            + r / (1 + r * v("PRTCG")) * v("CGV")
            + r / (1 + r * v("PRTIG")) * v("IGV"))


def build_equations() -> list[Eq]:
    E: list[Eq] = []

    def add(no, name, kind, rhs, lhs=None, inv=None):
        E.append(Eq(no, name, kind, rhs, lhs, inv))

    # ---------------- 1. 需要 -------------------------------------------------
    add(1, "GDPD", "level", lambda v: (v("GDP") - v("GDP", 1)) / v("GDP", 1) * 400)
    add(2, "GDP", "level", lambda v: v("CP") + v("IFP") + v("IHP") + v("INP") + v("CG")
        + v("IG") + v("ING") + v("BF") + v("KAISA"))

    def rr_gdp(v, k):
        return v("RGB", k) - 400 * ln(v("PGDPAT", k) / v("PGDPAT", k + 1))

    add(3, "CP", "dlog", lambda v: (
        -0.042671 * (ln(v("CP", 1))
                     - 0.986981 * ln(sum(ydr(v, k) for k in range(1, 6)) / 5)
                     - 0.007910 * ln(v("NWCV", 1) / v("PCP", 1)))
        + 0.208950 * (ln(ydr(v, 1)) - ln(ydr(v, 2)))
        + 0.251115 * (ln(ydr(v, 2)) - ln(ydr(v, 3)))
        + 0.000506 * (rr_gdp(v, 0) - rr_gdp(v, 1))
        + 0.725444 * (v("RTCI", -1) - v("RTCI", 0))
        - 1.300309 * (v("RTCI", 0) - v("RTCI", 1))
        + 0.358144 * (v("RTCI", 1) - v("RTCI", 2))
        - 0.089450 * v("D202") + 0.030462 * v("D203")))

    def rr_cd(v, k):
        return v("RCD", k) - 400 * ln(v("PGDPAT", k) / v("PGDPAT", k + 1))

    add(4, "IFP", "dlog", lambda v: (
        -0.001676 * ln(v("KFP", 1) / v("KFPSTA", 1))
        + 0.202322 * sum(dl(v, "PSHARE", k) for k in range(4)) / 4
        - 0.004495 * (rr_cd(v, 0) - rr_cd(v, 1))
        - 0.003368 * (rr_cd(v, 1) - rr_cd(v, 2))
        - 0.130921 * sum(dl(v, "ETT", k) for k in range(4)) / 4))

    def rr_ih(v, k):
        return v("RGB", k) - 100 * ln(v("PIHPAT", k) / v("PIHPAT", k + 4))

    add(5, "IHP", "dlog", lambda v: (
        -0.401905 * (ln(v("KHP", 1))
                     - (0.547149 * ln(sum(ydr(v, k) for k in range(1, 6)) / 5)
                        + 0.003366 * v("TIME", 1) - 3.10e-05 * v("TIME", 1) ** 2
                        + 1.076611 * ln(v("HH", 1)) - 3.135839))
        - 0.007125 * (rr_ih(v, 0) - rr_ih(v, 1))
        - 0.001411 * (rr_ih(v, 1) - rr_ih(v, 2))
        + 0.192584 * (ln(ydr(v, 0)) - ln(ydr(v, 1)))
        + 0.144948 * (ln(ydr(v, 2)) - ln(ydr(v, 3)))
        - 2.239347 * (v("RTCI", 0) - v("RTCI", 1))
        + 0.326110 * (v("RTCI", -1) - v("RTCI", 0))
        + 0.655187 * (v("RTCI", -2) - v("RTCI", -1))
        + 0.036148 * v("D954C963") - 0.095167 * v("D0703") - 0.067775 * v("D091C093")
        - 0.004362))

    def rr_in(v, k):
        return v("RCD", k) - 100 * ln(v("PINPAT", k) / v("PINPAT", k + 4))

    add(6, "INP", "custom",
        lambda v: (-0.007311 * ln((v("KNPV", 1) / v("GDPV", 1)) / v("KNGDEQ", 1))
                   - 0.499038 * (v("INP", 1) / v("GDP", 1) - v("INP", 2) / v("GDP", 2))
                   - 0.000893 * (rr_in(v, 2) - rr_in(v, 3))
                   + 0.000227 * dd(v, "GDPGAP")),
        lhs=lambda v: v("INP") / v("GDP") - v("INP", 1) / v("GDP", 1),
        inv=lambda v, y: v("GDP") * (v("INP", 1) / v("GDP", 1) + y))

    add(7, "CG", "level", lambda v: v("CGXRKG") + v("CCAVGR") * v("RKG"))
    add(8, "BF", "level", lambda v: v("XGS") - v("MGS"))

    def rpx(v, k):  # 輸出の相対価格
        return 100 * v("PXGS", k) / ((v("FXS", k) / FXS2015) * v("WD_PX", k))

    add(9, "XGS", "dlog", lambda v: (
        -0.229111 * (ln(v("XGS", 1) / v("WD_YVI", 1))
                     - (6.657787 - 0.198561 * ln(rpx(v, 1))  # [fix] FXS2011→FXS2015（定数のみ）
                        + 0.008266 * v("TIME", 1) - 4.82e-05 * v("TIME", 1) ** 2))
        + 1.297672 * dl(v, "WD_YVI")
        # @MOVSUM(DLOG(x(-1),0,4),5)/5 : 前年同期比対数差の5期移動平均
        - 0.079268 * sum(ln(rpx(v, 1 + j)) - ln(rpx(v, 5 + j)) for j in range(5)) / 5
        - 0.367667 * v("D091") - 0.229254 * v("D202")))

    add(10, "MGS", "level", lambda v: v("FUEL") + v("NFMGS"))

    def rpf(v, k):
        return v("PFUELAT", k) / v("PGDPAT", k)

    add(11, "FUEL", "dlog", lambda v: (
        -0.171986 * (ln(v("FUEL", 1) / v("GDP", 1))
                     - (-0.002409 * ln(rpf(v, 1)) - 0.001171 * v("TIME", 1) - 3.000749))
        + 0.047406 * dl(v, "GDP")
        + 0.172695 * (ln(rpf(v, 1)) - ln(rpf(v, 2)))
        - 0.175121 * (ln(rpf(v, 2)) - ln(rpf(v, 3)))
        + 0.116953 * (ln(rpf(v, 3)) - ln(rpf(v, 4)))
        - 0.036201 * (ln(rpf(v, 4)) - ln(rpf(v, 5)))))

    def inv3(v, k):
        return v("IFP", k) + v("IHP", k) + v("IG", k)

    def rpn(v, k):
        return v("PNFMGSAT", k) / v("PGDPAT", k)

    add(12, "NFMGS", "dlog", lambda v: (
        -0.000380 * (ln(v("NFMGS", 1))
                     - (-21.50892 + 0.448526 * ln(inv3(v, 1))
                        + 2.054694 * ln(v("CP", 1) + v("CG", 1))
                        + 0.196224 * ln(rpn(v, 1)) + 0.006261 * v("TIME", 1)))
        + 0.034363 * (ln(inv3(v, 0)) - ln(inv3(v, 1)))
        + 0.957629 * dl(v, "CP") + 0.787116 * dl(v, "CP", 1)
        - 0.109475 * (ln(rpn(v, 2)) - ln(rpn(v, 3)))
        - 0.162353 * v("D091")))

    # ---------------- 名目値 -------------------------------------------------
    add(13, "GDPV", "level", lambda v: v("CPV") + v("IFPV") + v("IHPV") + v("INPV") + v("CGV")
        + v("IGV") + v("INGV") + v("BFV"))
    add(14, "CPV", "level", lambda v: v("CP") * v("PCP"))
    add(15, "IFPV", "level", lambda v: v("IFP") * v("PIFP"))
    add(16, "IHPV", "level", lambda v: v("IHP") * v("PIHP"))
    add(17, "INPV", "level", lambda v: v("INP") * v("PINP") + v("ERRINPV"))
    add(18, "CGV", "level", lambda v: v("CG") * v("PCG"))
    add(19, "IGV", "level", lambda v: v("IG") * v("PIG"))
    add(20, "INGV", "level", lambda v: v("ING") * v("PING") + v("ERRINGV"))
    add(21, "BFV", "level", lambda v: v("XGSV") - v("MGSV"))
    add(22, "XGSV", "level", lambda v: v("XGS") * v("PXGS"))
    add(23, "MGSV", "level", lambda v: v("FUELV") + v("NFMGSV"))
    add(24, "FUELV", "level", lambda v: v("FUEL") * v("PFUEL"))
    add(25, "NFMGSV", "level", lambda v: v("NFMGS") * v("PNFMGS"))
    add(26, "GDPVEXCT", "level", lambda v: v("CPVEXCT") + v("IFPVEXCT") + v("IHPVEXCT")
        + v("INPVEXCT") + v("INGVEXCT") + v("CGVEXCT") + v("IGVEXCT") + v("XGSV") - v("MGSVEXCT"))
    add(27, "CPVEXCT", "level", lambda v: v("CP") * v("PCPAT"))
    add(28, "IFPVEXCT", "level", lambda v: v("IFP") * v("PIFPAT"))
    add(29, "IHPVEXCT", "level", lambda v: v("IHP") * v("PIHPAT"))
    add(30, "INPVEXCT", "level", lambda v: v("INPV") / (1 + v("RTCI") * v("PRTNP")))
    add(31, "CGVEXCT", "level", lambda v: v("CG") * v("PCGAT"))
    add(32, "IGVEXCT", "level", lambda v: v("IG") * v("PIGAT"))
    add(33, "INGVEXCT", "level", lambda v: v("INGV") / (1 + v("RTCI") * v("PRTNG")))
    add(34, "MGSVEXCT", "level", lambda v: v("MGS") * v("PMGSAT"))

    # ---------------- ストック -----------------------------------------------
    add(35, "KFP", "level", lambda v: v("KFPV") / v("PIFP"))
    add(36, "RFP", "level", lambda v: v("RFPV") / v("PIFP", 1))
    add(37, "RFPV", "level", lambda v: v("RRFPV") * v("KFPV", 1) * 4)
    add(38, "KFPSTA", "level", lambda v: (1 - AVRDL) * v("GDP") / v("UCC"))
    add(39, "KHP", "level", lambda v: v("KHPV") / v("PIHP"))
    add(40, "RHPV", "level", lambda v: v("RRHPV") * v("KHPV", 1) * 4)
    add(41, "KNP", "level", lambda v: v("KNPV") / v("PINP"))
    add(42, "RNPV", "level", lambda v: v("RRNPV") * v("KNPV", 1) * 4)
    add(43, "KG", "level", lambda v: v("KGV") / v("PIG"))
    add(44, "KGV", "level", lambda v: v("KGV", 1) * (v("PIG") / v("PIG", 1)) + v("IGV") / 4
        - (v("RKGV") / 4) * (v("PIG") / v("PIG", 1)))
    add(45, "RKG", "level", lambda v: v("RKGV") / v("PIG", 1))
    add(46, "RKGV", "level", lambda v: v("RRKGV") * v("KGV", 1) * 4)

    # ---------------- 2. 労働・供給 -------------------------------------------
    add(47, "LF", "custom",
        lambda v: (-0.053590 * ln(v("POP65") / v("POP"))
                   - 0.026685 * ln(v("UR", 2))
                   + 0.422674 * ln((v("WPH") / v("PCPAT")) / v("WPHXREQ"))  # [fix] WPHX→WPH
                   - 0.536525),
        lhs=lambda v: ln(v("LF") / v("POP")),
        inv=lambda v, y: v("POP") * ex(y))

    add(48, "UR", "d", lambda v: (
        -0.004683 * ln(v("UR", 1) / v("UREQ", 1))
        - 1.082884 * dl(v, "CUX")
        + 0.327986 * dd(v, "UR", 1) + 0.168706 * dd(v, "UR", 2)
        + 0.959006 * (ln(lshare(v, 1) / BETA) - ln(lshare(v, 2) / BETA))
        + 0.433098 * v("D092")))

    add(49, "LHX", "dlog", lambda v: (
        -0.198209 * (ln(v("LHX", 1))
                     - (-0.000837 * v("TIME", 1) - 0.057861 * v("D9203C", 1) + 4.770799))
        + 0.327645 * dl(v, "LE")
        + 0.107178 * (v("CUX") / 100 - v("CUX", 1) / 100)))

    add(50, "GDPPOT", "level", lambda v: (
        ex(GAM) * ex(RAM1 * v("TIME")) * ex(RAM2 * v("TIME96Q2"))
        * (v("LF") * (1 - v("UREQ") / 100) * v("LHXEQ")) ** BETA
        * (v("KFP", 1) * v("CUXEQ") / 100) ** (1 - BETA)))
    add(51, "GDPGAP", "level", lambda v: (v("GDP") - v("GDPPOT")) / v("GDPPOT") * 100)
    add(52, "LE", "level", lambda v: v("LF") * (1 - v("UR") / 100))
    add(53, "LW", "level", lambda v: v("LE") * v("RLEW"))  # [fix] LA_LE→LE

    add(54, "CUX", "log", lambda v: (
        ln((((v("GDP") - v("ERRPFU"))
             / (ex(GAM + RAM1 * v("TIME") + RAM2 * v("TIME96Q2")) * (v("LE") * v("LHX")) ** BETA))
            ** (1 / (1 - BETA))) / v("KFP", 1))
        + 4.605170))

    # ---------------- 物価 ---------------------------------------------------
    gap_w = (0.015069, 0.022603, 0.022603, 0.015069)
    add(55, "PGDPAT", "dlog", lambda v: (
        0.135358 * dl(v, "PGDPAT", 1)
        + 0.017695 * (ln(v("M2CD") / v("GDP")) - ln(v("M2CD", 1) / v("GDP", 1)))
        - 1.06e-05 * v("TIME")
        + 0.013820 * v("D084") + 0.004652 * v("D142") + 0.010386 * v("D151") + 0.008754 * v("D191")
        + sum(w * ln((v("GDPGAP", k + 1) + 100) / 100) for k, w in enumerate(gap_w))))

    add(56, "PCPAT", "dlog", lambda v: (
        -0.048746 * (ln(v("PCPAT", 1) / v("PGDPAT", 1)) + 0.092200
                     - 0.021043 * ln(v("PMGSAT", 1) / v("PGDPAT", 1)) - 0.000646 * v("TIME", 1))
        + 0.567277 * dl(v, "PGDPAT") + 0.198024 * dl(v, "PGDPAT", 1)
        + 0.018442 * dl(v, "PFUELAT")))

    add(57, "CGPIAT", "dlog", lambda v: (
        -0.033739 * (ln(v("CGPIAT", 1) / v("PMGSAT", 1))
                     - (0.650307 * ln(v("PGDPAT", 1) / v("PMGSAT", 1)) - 0.000317 * v("TIME", 1) + 0.014447))
        + 0.191675 * dl(v, "PMGSAT") + 0.483802 * dl(v, "PGDPAT")
        + 0.019649 * dl(v, "CUX", 1)
        - 0.000488))  # [fix] 論文 "--0.000488"（t値は負）

    add(58, "PIFPAT", "dlog", lambda v: (
        -0.022185 * (ln(v("PIFPAT", 1) / v("PNFMGSAT", 1))
                     - (0.754583 * ln(v("PGDPAT", 1) / v("PNFMGSAT", 1)) - 0.000304 * v("TIME", 1) + 0.057748))
        + 0.654523 * dl(v, "PGDPAT")
        + 0.001742 * dd(v, "GDPGAP") + 0.000653 * dd(v, "GDPGAP", 1)
        + 0.266259 * dl(v, "PIFPAT", 2) + 0.000252 * dl(v, "PNFMGSAT", 2)
        - 0.000604))

    # [fix] 括弧の対応を修正
    add(59, "PIFPATGR", "level", lambda v: (v("PIFPAT") - v("PIFPAT", 4)) / v("PIFPAT", 4) * 100)
    add(60, "PIFPATSUM", "level", lambda v: (v("PIFPATGR") + v("PIFPATGR", 1)) / 2)

    add(61, "PIHPAT", "dlog", lambda v: (
        0.591046 * dl(v, "PGDPAT") + 0.323149 * dl(v, "PGDPAT", 1)
        + 0.070799 * dl(v, "PMGSAT") + 0.036542 * dl(v, "PMGSAT", 1)
        - 0.017771 * dl(v, "WIPH") + 0.001777))

    add(62, "PINPAT", "dlog", lambda v: 0.477662 * dl(v, "CGPIAT") + 0.322910 * dl(v, "CGPIAT", 1))

    add(63, "PCGAT", "dlog", lambda v: (
        -0.012528 * (ln(v("PCGAT", 1) / v("PCPAT", 1))
                     - (0.246639 * ln(v("WIPH", 1) / v("PCPAT", 1)) - 0.911111))
        - 0.284893 * dl(v, "PCGAT", 1) + 0.256729 * dl(v, "PCGAT", 2)
        + 0.851041 * dl(v, "PCPAT", 1) + 2.54e-05))

    add(64, "PIGAT", "dlog", lambda v: (
        -0.007312 * (ln(v("PIGAT", 1) / v("PIFPAT", 1))
                     - (1.088554 * ln(v("PIHPAT", 1) / v("PIFPAT", 1))
                        - 0.438980 * ln(v("PCGAT", 1) / v("PIFPAT", 1)) + 0.007716))
        + 0.884220 * dl(v, "PIFPAT") + 0.242857 * dl(v, "PIHPAT") + 0.066865 * dl(v, "PCGAT")
        + 0.002070))

    add(65, "PXGS", "dlog", lambda v: (
        -0.161846 * (ln(v("PXGS", 1) / v("FXS", 1))
                     - (0.725350 * ln(v("CGPIAT", 1) / v("FXS", 1)) - 0.002055 * v("TIME") - 0.985034))
        + 0.399314 * dl(v, "FXS") + 0.231007 * dl(v, "WD_PX")))

    # PRTMG は外生変数表に無いが式に現れる（既定 0）
    add(66, "PMGSAT", "level", lambda v: v("PMGS") / (1 + v("RTCI") * v("PRTMG")))

    add(67, "PFUELAT", "dlog", lambda v: (
        -0.329477 * (ln(v("PFUELAT", 1)) - (0.879168 * ln(v("POILD", 1) * v("FXS", 1)) - 7.984062))
        + 0.667851 * dl(v, "POILD") + 0.894119 * dl(v, "FXS") + 0.000210))

    def wpi_yen(v, k):
        return v("WD_PI", k) * v("FXS", k)

    add(68, "PNFMGSAT", "dlog", lambda v: (
        -0.360246 * (ln(v("PNFMGSAT", 1))
                     - (-0.010084 * v("TIME", 1) + 2.81e-05 * v("TIME", 1) ** 2
                        + 0.250637 * ln(wpi_yen(v, 1)) - 1.496870))
        + 0.336684 * (ln(wpi_yen(v, 0)) - ln(wpi_yen(v, 1)))))

    add(69, "WI", "level", lambda v: v("YWIV") / v("LW"))
    add(70, "WIPH", "level", lambda v: 100 * v("WI") / v("LHX"))
    add(71, "PGDP", "level", lambda v: v("GDPV") / v("GDP"))
    add(72, "PGDPD", "level", lambda v: (v("PGDP") - v("PGDP", 1)) / v("PGDP", 1) * 400)
    add(73, "PCP", "level", lambda v: v("PCPAT") * (1 + v("RTCI") * v("PRTCP")))
    add(74, "CGPI", "level", lambda v: v("CGPIAT") * (1 + v("RTCI") * v("PRTGP")))
    add(75, "PIFP", "level", lambda v: v("PIFPAT") * (1 + v("RTCI") * v("PRTIF")))
    add(76, "PIHP", "level", lambda v: v("PIHPAT") * (1 + v("RTCI") * v("PRTIH")))
    add(77, "PINP", "level", lambda v: v("PINPAT") * (1 + v("RTCI") * v("PRTNP")))
    add(78, "PCG", "level", lambda v: v("PCGAT") * (1 + v("RTCI") * v("PRTCG")))
    add(79, "PIG", "level", lambda v: v("PIGAT") * (1 + v("RTCI") * v("PRTIG")))
    add(80, "PMGS", "level", lambda v: v("MGSV") / v("MGS"))
    add(81, "PFUEL", "level", lambda v: v("PFUELAT") * (1 + v("RTCI") * v("PRTFU")))
    add(82, "PNFMGS", "level", lambda v: v("PNFMGSAT") * (1 + v("RTCI") * v("PRTNF")))

    # ---------------- 所得分配 -----------------------------------------------
    add(83, "NIV", "level", lambda v: v("GDPV") - v("CCAV") - v("ITAXV") + v("SUBV")
        + (v("RTRIV") - v("PTRIV")) - v("SDV"))
    add(84, "YCV", "level", lambda v: v("NIV") - v("YWV") - (v("YIV") + v("YICV")))
    add(85, "YCVAT", "level", lambda v: sum(v("YCV", k) for k in range(4)) / 4 * (1 - v("ETT")))
    add(86, "YOLIV", "log", lambda v: 0.662644 * ln(v("SR") * v("YWIV")) + 3.570176)
    add(87, "CCAV", "log", lambda v: 1.001451 * ln(v("RFPV") + v("RHPV") + v("RKGV")))
    add(88, "INPVA", "level", lambda v: 1.009059 * dd(v, "PINP") * v("KNP", 1) * 4)

    add(89, "YWV", "custom",
        lambda v: (-0.136934 * ln(lshare(v, 1) / BETA)
                   - 0.379185 * (ln(lshare(v, 1)) - ln(lshare(v, 2)))
                   - 0.171113 * (ln(lshare(v, 2)) - ln(lshare(v, 3)))
                   - 0.310936 * (ln(lshare(v, 3)) - ln(lshare(v, 4)))
                   - 0.171401 * dl(v, "CUX")
                   - 0.000742),
        lhs=lambda v: ln(lshare(v, 0)) - ln(lshare(v, 1)),
        inv=lambda v, y: lshare(v, 1) * ex(y) * (v("NIV") + v("CCAV")) - v("YICV"))

    add(90, "YDV", "level", lambda v: v("YWV") + v("BSSV") + v("YIEV") + v("YICV") + v("OTYDV")
        - v("TYPV") - v("CSSV"))

    add(91, "YICV", "dlog", lambda v: (
        -0.539627 * (ln(v("YICV", 1)) - 0.857041 * ln(v("YWV", 1))
                     - (1 - 0.857041) * ln(v("YCV", 1))
                     + 0.001211 * v("TIME", 1) + 1.650417)
        - 0.096934 * dd(v, "RGB", 2)
        - 0.057300 * sum(dl(v, "PMGSAT", k) for k in range(3)) / 3))

    add(92, "YLV", "level", lambda v: v("YDV") - v("YIEV") - v("OTYDV"))
    add(93, "YL", "level", lambda v: v("YLV") / v("PGDP"))
    add(94, "YWIV", "level", lambda v: v("YWV") - v("YOLIV"))
    add(95, "W", "level", lambda v: v("YWV") / v("LW"))
    add(96, "WPH", "level", lambda v: 100 * v("W") / v("LHX"))
    add(97, "NWCV", "level", lambda v: v("FNWV") + v("LANDV") + v("SHAREV") + v("KHPV"))
    add(98, "NWTV", "level", lambda v: v("FNWV") + v("LANDT") + v("SHARETV") + v("KHPV"))
    add(99, "FNWV", "level", lambda v: v("SBGV") + v("SBCV"))
    add(100, "SHARETV", "level", lambda v: v("PSHARE") * v("RSHARET"))
    add(101, "SHAREV", "level", lambda v: v("PSHARE") * v("RSHARE"))
    add(102, "LANDT", "level", lambda v: v("PROLA") * v("LANDV"))
    add(103, "LANDV", "level", lambda v: v("PLAND") * v("RLAND"))

    add(104, "YIEV", "custom",
        lambda v: (0.034806 * (v("NIV", 1) / v("NWCV", 2) - v("NIV", 2) / v("NWCV", 3))
                   + 0.000677 * dd(v, "RCD")
                   + 0.000345 * dd(v, "RGB", 1)
                   + 0.000520 * dd(v, "RGB", 3)),
        lhs=lambda v: v("YIEV") / v("NWCV", 1) - v("YIEV", 1) / v("NWCV", 2),
        inv=lambda v, y: v("NWCV", 1) * (v("YIEV", 1) / v("NWCV", 2) + y))

    add(105, "YIV", "level", lambda v: v("YIEV") + v("YIGV"))
    add(106, "KPV", "level", lambda v: v("KFPV") + v("KHPV") + v("KNPV"))
    add(107, "KFPV", "level", lambda v: v("KFPV", 1) * (v("PIFP") / v("PIFP", 1)) + v("IFPV") / 4
        - (v("RFPV") / 4) * (v("PIFP") / v("PIFP", 1)))
    add(108, "KHPV", "level", lambda v: v("KHPV", 1) * (v("PIHP") / v("PIHP", 1)) + v("IHPV") / 4
        - (v("RHPV") / 4) * (v("PIHP") / v("PIHP", 1)))
    add(109, "KNPV", "level", lambda v: v("KNPV", 1) * (v("PINP") / v("PINP", 1)) + v("INPV") / 4
        - (v("RNPV") / 4) * (v("PINP") / v("PINP", 1)))
    add(110, "KNGDEQ", "level", lambda v: -0.000614 * v("TIME") + 2.03e-06 * v("TIME") ** 2 + 0.161341)

    # ---------------- 3. 貨幣市場 ---------------------------------------------
    add(111, "M2CD", "log", lambda v: (1.000279 * ln(v("GDP") * v("PGDPAT")) - 0.035292 * v("RCD")
                                       + 0.006374 * v("TIME") - 0.335434))
    add(112, "MK", "level", lambda v: v("M2CD") / v("GDP"))
    add(113, "RCD", "level", lambda v: max(0.001, v("RCDX")))
    # テイラー・ルール型の政策反応関数
    add(114, "RCDX", "level", lambda v: (
        0.71 * v("RCD", 1)
        + (1 - 0.71) * (100 * (v("GDPPOT") - v("GDPPOT", 1)) / v("GDPPOT", 1) + 2
                        + 1.68 * (100 * (v("PGDPAT") - v("PGDPAT", 1)) / v("PGDPAT", 1) - 2)
                        + 0.15 * v("GDPGAP"))))
    add(115, "RGB", "level", lambda v: max(0.001, v("RGBX")))
    add(116, "RGBX", "d", lambda v: (
        -0.048253 * (v("RGB", 1) - sum(v("RCD", 1 + j) for j in range(8)) / 8)
        - 0.086901 * dd(v, "RGB", 1)
        + 0.510110 * dd(v, "RCD")
        # D(DLOG(PCPAT(-1)*400)): 定数倍は対数差で消える（印刷どおり）
        + 1.815275 * (dl(v, "PCPAT", 1) - dl(v, "PCPAT", 2))))

    add(117, "RSHARE", "custom",
        lambda v: (0.002510 * ln(v("SHAREV") / v("GDPV")) - 0.003548 * v("TIME")
                   + 9.42e-07 * v("TIME") ** 2 - 1.192846),
        lhs=lambda v: ln(v("RSHARE") / v("RSHARET")),
        inv=lambda v, y: v("RSHARET") * ex(y))
    add(118, "PERR", "level", lambda v: v("SHARETV") / v("YCVAT"))

    def rr_cg(v, k):
        return v("RGB", k) - 400 * ln(v("CGPIAT", k) / v("CGPIAT", k + 1))

    add(119, "PLAND", "dlog", lambda v: (
        -0.027890 * (ln(v("PLAND", 1) / v("GDPV", 1))
                     - (0.110679 * ln(v("PSHARE", 1) / v("GDPV", 1))
                        - 0.013395 * v("TIME70Q1", 1) - 10.62183))
        - 0.009485 * ln(v("PSHARE", 1))
        - 0.000174 * (rr_cg(v, 0) - rr_cg(v, 1))))

    rcd_w = (0.038272, 0.061236, 0.068890, 0.061236, 0.038272)
    fxs_w = (0.066904, 0.089205, 0.066904)
    add(120, "PSHARE", "custom",
        lambda v: (0.186344 * dl(v, "GDPV") + 4.095267 * dl(v, "WD_YVI") - 0.036622
                   - sum(w * dd(v, "RCD", k) for k, w in enumerate(rcd_w))
                   + sum(w * dl(v, "FXS", k) for k, w in enumerate(fxs_w))),
        lhs=lambda v: ln(v("PSHARE") * v("RSHARET")) - ln(v("PSHARE", 1) * v("RSHARET", 1)),
        inv=lambda v, y: v("PSHARE", 1) * v("RSHARET", 1) * ex(y) / v("RSHARET"))

    # ---------------- 資本コスト ---------------------------------------------
    add(121, "PVDP", "level", lambda v: (
        -1 / 18 * ln(0.1) * (1 - 0.1 * ex(-1 * v("RGB") / 100 * 18)) / (v("RGB") / 100 - ln(0.1) / 18)))
    # [fix] 括弧の対応を補完、SLRATI0→SLRATIO
    add(122, "UCCDB", "level", lambda v: (
        v("PIFPAT") / (1 - v("TT"))
        * ((1 - v("TT")) * (v("RCD") + v("SLRATIO") * v("RGB")) / (1 + v("SLRATIO"))
           + v("RRFP") * 400 - 0.925751 * v("PIFPATSUM"))
        * (1 - v("TT") * v("PVDP") - v("TINCR"))))
    add(123, "UCCDE", "level", lambda v: (
        v("PIFPAT") / (1 - v("TT"))
        * ((1 - v("PERR") * v("ROR") / 100) / v("PERR") * 100
           + sum(v("RRFP", k) for k in range(4)) * 100
           - 0.925751 * v("PIFPATSUM"))
        * (1 - v("TT") * v("PVDP") - v("TINCR"))))
    add(124, "UCCPF", "level", lambda v: (1 - v("REQU")) * v("UCCDB") + v("REQU") * v("UCCDE"))
    add(125, "UCC", "level", lambda v: v("UCCPF") / 100 / v("PGDPAT"))

    # ---------------- 4. 財政 ------------------------------------------------
    add(126, "BGV", "level", lambda v: (v("TAXV") + v("CSSV") + v("YIGV") + v("CCAVG") - v("BSSV")
                                        - v("CGV") - v("IGVR") * v("IGV") - v("SUBV") + v("OTNGV")))
    add(127, "BGVATGDPV", "level", lambda v: v("BGV") / v("GDPV") * 100)
    add(128, "TAXV", "level", lambda v: v("TYPV") + v("TYCV") + v("ITAXV"))

    if ITR_FORM == "dlog":
        add(129, "ITR", "custom",
            lambda v: -0.958520 * ln(v("ITR", 1) / v("ITREQ", 1)) + 0.625328 * v("GDPGAP") / 100,
            lhs=lambda v: ln(v("ITR") / v("ITREQ")) - ln(v("ITR", 1) / v("ITREQ", 1)),
            inv=lambda v, y: v("ITREQ") * (v("ITR", 1) / v("ITREQ", 1)) * ex(y))
    else:  # 論文の印刷どおり
        add(129, "ITR", "custom",
            lambda v: -0.958520 * ln(v("ITR", 1) / v("ITREQ")) + 0.625328 * v("GDPGAP") / 100,
            lhs=lambda v: ln(v("ITR") / v("ITREQ")),
            inv=lambda v, y: v("ITREQ") * ex(y))

    add(130, "TYPV", "level", lambda v: v("ITR") * sum(typ_base(v, k) for k in range(4)) / 4)
    add(131, "TYCV", "level", lambda v: v("ETT") * sum(v("YCV", k) for k in range(1, 5)) / 4)
    add(132, "ETT", "custom",
        lambda v: (0.871359 * ln(v("ETT", 1) / v("TT", 1)) + 0.168727 * v("GDPGAP") / 100
                   + 0.251152 * v("D203") + 0.261928 * v("D204") + 0.007970),
        lhs=lambda v: ln(v("ETT") / v("TT")),
        inv=lambda v, y: v("TT") * ex(y))
    add(133, "ITAXV", "level", lambda v: v("TCIV") + v("TCSTV") + v("OITAXV"))
    add(134, "TCIV", "log", lambda v: (
        0.915210 * ln(tci_base(v))
        - 0.011036 * v("DTCIC2") * v("DTCIC2") * ln(tci_base(v))
        + 0.191987 * v("D972C")))
    add(135, "TCSTV", "level", lambda v: v("RTCST") * v("MGSV"))
    add(136, "OITAXV", "dlog", lambda v: -0.230742 * ln(v("OITAXV", 1) / v("GDPV", 1)) - 0.714290)
    add(137, "CSSV", "log", lambda v: 1.544466 * ln(v("YOLIV")) - 5.258649)
    add(138, "YIGV", "custom",
        lambda v: 0.000101 * v("RGB") + 0.937011 * v("YIGV", 1) / (-v("SBGV", 2)) + 4.38e-05,
        lhs=lambda v: v("YIGV") / (-v("SBGV", 1)),
        inv=lambda v, y: y * (-v("SBGV", 1)))
    add(139, "CCAVG", "level", lambda v: (0.182353 * v("RKGV") + 0.160804 * v("RKGV", 1)
                                          + 0.159536 * v("RKGV", 2) + 0.178468 * v("RKGV", 3)))
    add(140, "BSSV", "log", lambda v: 0.959556 * ln(v("IR") * v("WI") * v("POP65")) + 0.536611)
    add(141, "SBGV", "level", lambda v: v("SBGV", 1) - v("BGV") / 4 + v("RSBGV"))
    add(142, "SBGVATGDPV", "level", lambda v: v("SBGV") / v("GDPV") * 100)

    # ---------------- 5. 海外 ------------------------------------------------
    add(143, "RTRIV", "dlog", lambda v: (
        -0.009322 * ln((200 * v("RTRIV", 1)) / (v("FASSTV", 2) * (v("US_RGB", 2) + v("US_RGB", 1))))
        + 0.546269 * dl(v, "FASSTV", 1)
        + 0.014788 * dd(v, "US_RGB", 2) + 0.054964 * dd(v, "US_RGB", 3)
        + 0.473896 * dl(v, "FXS")))
    add(144, "PTRIV", "dlog", lambda v: (
        -0.003783 * ln((200 * v("PTRIV", 1)) / (v("FLIABV", 2) * (v("RGB", 2) + v("RGB", 1))))
        + 0.078589 * dl(v, "PTRIV", 2) + 0.106503 * dl(v, "PTRIV", 3)
        + 0.722067 * dl(v, "FLIABV", 1)
        + 0.003603 * dd(v, "RGB", 1) + 0.012019 * dd(v, "RGB", 2)
        + 0.447491 * dl(v, "FXS") - 0.857802 * v("D961")))
    add(145, "FASSTV", "dlog", lambda v: (
        0.206646 * dl(v, "SBCV") + 0.196878 * dl(v, "FXS")
        + 0.242347 * dl(v, "GDPV") + 0.128465 * dl(v, "GDPV", 1) + 0.008120))
    add(146, "RSBCV", "log", lambda v: (0.014175 * ln(v("RSBCV", 1)) + 0.736952 * ln(v("FXS") / v("FXS", 1))
                                        - 0.005499))
    add(147, "FXS", "dlog", lambda v: (
        -0.022012 * ln(v("FXS", 1) / (v("CGPIAT", 1) / v("US_WPI", 1)))
        + 0.004459 * (v("US_RGB") - v("RGB"))
        + 0.001060 * (v("US_RGB", 2) - v("RGB", 2))
        + 0.057818 * (ln(v("CGPIAT") / v("US_WPI")) - ln(v("CGPIAT", 1) / v("US_WPI", 1)))
        + 0.177620 * dl(v, "FXS", 1)
        - 0.126159 * v("D952") - 0.180614 * v("D984") + 0.187360))
    add(148, "BCV", "level", lambda v: v("BFV") + (v("RTRIV") - v("PTRIV")) + v("ERRBCV"))
    add(149, "SBCV", "level", lambda v: v("SBCV", 1) * v("RSBCV") + v("BCV") / 4)
    add(150, "FLIABV", "level", lambda v: v("FASSTV") - v("SBCV"))
    add(151, "BCVATGDPV", "level", lambda v: 100 * v("BCV") / v("GDPV"))
    add(152, "SBCVATGDPV", "level", lambda v: v("SBCV") / v("GDPV") * 100)

    return E


# ---------------------------------------------------------------------------
# 暦・ダミー変数
# ---------------------------------------------------------------------------
# 例: D202=2020Q2, D0703=2007Q3（四半期は2桁のこともある）, D954C963=1995Q4〜1996Q3, D9203C=1992Q3以降
_DUMMY_RE = re.compile(r"^D(\d{2})(0?[1-4])(?:C(?:(\d{2})(0?[1-4]))?)?$")


def _year(yy: str) -> int:
    y = int(yy)
    return 1900 + y if y >= 50 else 2000 + y


def make_calendar(index: pd.PeriodIndex) -> pd.DataFrame:
    """TIME（1980Q1=1）、トレンド、推定式に現れるダミー変数を作る."""
    ordinal = np.array([p.year * 4 + p.quarter - 1 for p in index], dtype=float)
    base80 = 1980 * 4 + 0  # 1980Q1
    out = pd.DataFrame(index=index)
    out["TIME"] = ordinal - base80 + 1
    out["TIME96Q2"] = np.maximum(0.0, ordinal - (1996 * 4 + 1) + 1)
    out["TIME70Q1"] = ordinal - 1970 * 4 + 1
    names = ["D202", "D203", "D204", "D954C963", "D0703", "D091C093", "D091", "D092",
             "D9203C", "D084", "D142", "D151", "D191", "D972C", "D952", "D984", "D961"]
    for nm in names:
        m = _DUMMY_RE.match(nm)
        y1, q1 = _year(m.group(1)), int(m.group(2))
        start = y1 * 4 + q1 - 1
        if "C" not in nm:
            end = start
        elif m.group(3) is None:
            end = 10 ** 9
        else:
            end = _year(m.group(3)) * 4 + int(m.group(4)) - 1
        out[nm] = ((ordinal >= start) & (ordinal <= end)).astype(float)
    dt = np.zeros(len(index))
    for (ya, qa), (yb, qb) in [((1997, 2), (1998, 1)), ((2014, 2), (2015, 1)), ((2019, 4), (2020, 3))]:
        dt[(ordinal >= ya * 4 + qa - 1) & (ordinal <= yb * 4 + qb - 1)] = 1.0
    out["DTCIC2"] = dt
    return out


# ---------------------------------------------------------------------------
# ソルバー
# ---------------------------------------------------------------------------
class Model:
    def __init__(self, equations: list[Eq] | None = None):
        self.eqs = equations if equations is not None else build_equations()
        self.endog = [e.name for e in self.eqs]
        dup = {n for n in self.endog if self.endog.count(n) > 1}
        if dup:
            raise ValueError(f"重複した内生変数: {dup}")

    @staticmethod
    def _arrays(data: pd.DataFrame) -> dict[str, np.ndarray]:
        return {c: data[c].to_numpy(dtype=float).copy() for c in data.columns}

    def _positions(self, data: pd.DataFrame, start: str, end: str) -> range:
        idx = data.index
        return range(idx.get_loc(pd.Period(start, "Q")), idx.get_loc(pd.Period(end, "Q")) + 1)

    def add_factors(self, data: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
        """実績値に対する各式の残差（CERR*ERTS_xx に相当）を計算する."""
        X = self._arrays(data)
        pos = self._positions(data, start, end)
        af = pd.DataFrame(0.0, index=data.index, columns=self.endog)
        problems = []
        for t in pos:
            def v(n, k=0, _t=t):
                return X[n][_t - k]
            for e in self.eqs:
                try:
                    r = e.lhs_value(v) - e.rhs(v)
                except (KeyError, ValueError, ZeroDivisionError, OverflowError) as err:
                    problems.append((str(data.index[t]), e.no, e.name, repr(err)))
                    continue
                if not np.isfinite(r):
                    problems.append((str(data.index[t]), e.no, e.name, "non-finite"))
                af.iloc[t, af.columns.get_loc(e.name)] = r
        if problems:
            msg = "\n".join(f"{p} eq{no} {nm}: {err}" for p, no, nm, err in problems[:40])
            raise ValueError(f"誤差項を計算できない式があります（先頭40件）:\n{msg}")
        return af

    def solve(self, data: pd.DataFrame, start: str, end: str, af: pd.DataFrame,
              fixed: dict[str, pd.Series] | None = None,
              overrides: dict[str, Eq] | None = None,
              shocks: dict[str, pd.Series] | None = None,
              tol: float = 1e-10, maxit: int = 1000) -> pd.DataFrame:
        """ガウス＝ザイデル法で期ごとに同時方程式を解く.

        fixed:     内生変数を所与の経路で固定（式を外生化）
        overrides: 式を差し替え（例: 貨幣供給量外生化時の金利決定式）
        shocks:    変換後の左辺に加えるインパクト（アドファクターと同じ次元）
        """
        fixed = fixed or {}
        overrides = overrides or {}
        shocks = shocks or {}
        X = self._arrays(data)
        for n, s in fixed.items():
            X[n] = s.reindex(data.index).to_numpy(dtype=float).copy()
        eqs = [overrides.get(e.name, e) for e in self.eqs]
        A = {n: af[n].to_numpy() for n in af.columns}
        S = {n: s.reindex(data.index).fillna(0.0).to_numpy() for n, s in shocks.items()}
        pos = self._positions(data, start, end)
        for t in pos:
            def v(n, k=0, _t=t):
                return X[n][_t - k]
            for it in range(maxit):
                worst = 0.0
                for e in eqs:
                    if e.name in fixed:
                        continue
                    # 差し替え式で新たに内生化した変数には誤差項がない（0とする）
                    a = A[e.name][t] if e.name in A else 0.0
                    y = e.rhs(v) + a + (S[e.name][t] if e.name in S else 0.0)
                    new = e.invert(v, y)
                    old = X[e.name][t]
                    worst = max(worst, abs(new - old) / (1.0 + abs(old)))
                    X[e.name][t] = new
                if worst < tol:
                    break
            else:
                raise RuntimeError(f"{data.index[t]} で収束しませんでした (最大変化 {worst:.2e})")
        return pd.DataFrame(X, index=data.index)
