"""式129（所得実効税率 ITR）の左辺が水準か階差（DLOG）かを、印刷された推定統計量から検討する（Issue #2, #19）.

論文（2022年版、Research Note No.72）の印刷:
    LOG(ITR/ITREQ) = -0.958520*LOG(ITR(-1)/ITREQ) + 0.625328*GDPGAP/100   [OLS 1990Q1-2020Q4, RSQ 0.478962]
前の版（2018年版、Research Note No.41）の印刷:
    LOG(ITR/ITREQ) =  0.504631*LOG(ITR(-1)/ITREQ) + 0.458211*GDPGAP/100   [OLS 1990Q1-2016Q4, RSQ 0.287657]
論文の凡例（付属資料III）では RSQ は自由度修正済み決定係数。

1本の説明変数に近い回帰では、水準式の決定係数はおよそ（前期の係数）^2 になる。
2018年版は 0.50^2 ≈ 0.25 と RSQ 0.29 が整合するが、2022年版は (-0.96)^2 ≈ 0.92 に対して RSQ 0.48 で整合しない。
左辺が DLOG なら、前期の係数 ≈ -1・決定係数 ≈ 0.5 になる（水準の自己相関がほぼ0の系列）。

方法: 両方の形のデータ生成過程で、印刷の係数・誤差の標準偏差（SER 0.062475）を使って 1989Q4〜2020Q4 の125期を作り、
ラグをとった 1990Q1〜2020Q4 の124期で、切片・前期・GDPギャップの3係数（k=3）を OLS で推定する。
自由度修正済み決定係数 1 − (SSE/(n−k)) / (SST/(n−1)) を論文の RSQ と比べる。
GDPギャップはランダムウォーク（標準偏差0.3%）、誤差は独立な正規分布、初期値は0と仮定する。
原データによる再推定ではないので、結果は「この仮定のもとでの整合性」であり、誤植の確証ではない。

あわせて、同梱の ESRI 年次推計の四半期データ（1994年〜）で ITR を作り、1995Q1〜2020Q4 で両方の形を推定し直す。
ITR の作り方は3通り（季節調整済み＝モデルと同じ、原系列×課税ベース4期平均、原系列×同じ期の課税ベース）。
1990年代前半の四半期データと、同じ期間の GDP ギャップ（モデルの定義）がないため、GDP ギャップは説明変数に入れない（k=2）。

出力: output/experiment_itr_form.csv（シミュレーション）、output/experiment_itr_reestimate.csv（再推定）
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
B_LAG, B_GAP, SER = -0.958520, 0.625328, 0.062475  # 2022年版の印刷
N = 125  # 1989Q4〜2020Q4 を生成し、ラグをとって 1990Q1〜2020Q4 の124期で推定する
REPS = 2000


def fit(y: np.ndarray, cols: list[np.ndarray]) -> tuple[float, float]:
    """前期の係数と、自由度修正済み決定係数（論文の RSQ と同じ定義）を返す."""
    X = np.column_stack([np.ones(len(y))] + cols)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    n, k = X.shape
    return b[1], 1 - (e @ e / (n - k)) / ((y - y.mean()) @ (y - y.mean()) / (n - 1))


def main() -> None:
    rng = np.random.default_rng(0)
    rows = []
    # 真の形ごとに、水準の自己回帰係数を決める（DLOG型なら 1 + B_LAG、水準式なら B_LAG）
    for truth, rho in (("DLOG型が正しい", 1 + B_LAG), ("水準式が正しい", B_LAG)):
        est = []
        for _ in range(REPS):
            gap = np.cumsum(rng.normal(0, 0.3, N)) / 100  # GDPギャップ（%→小数、持続的な系列）
            gap -= gap.mean()
            x = np.zeros(N)
            for t in range(1, N):
                x[t] = rho * x[t - 1] + B_GAP * gap[t] + rng.normal(0, SER)
            lag = x[:-1]
            b_lv, r2_lv = fit(x[1:], [lag, gap[1:]])
            b_dl, r2_dl = fit(x[1:] - lag, [lag, gap[1:]])
            est.append((b_lv, r2_lv, b_dl, r2_dl))
        a = np.array(est)
        m, lo, hi = a.mean(axis=0), np.percentile(a, 5, axis=0), np.percentile(a, 95, axis=0)
        rows.append(dict(真の形=truth, 推定の観測数=N - 1,
                         水準式で推定した前期の係数=m[0], 水準式で推定した自由度修正RSQ=m[1],
                         水準式RSQ_5パーセント点=lo[1], 水準式RSQ_95パーセント点=hi[1],
                         DLOG型で推定した前期の係数=m[2], DLOG型で推定した自由度修正RSQ=m[3],
                         DLOG型RSQ_5パーセント点=lo[3], DLOG型RSQ_95パーセント点=hi[3]))
    rows.append(dict(真の形="論文2022年版の印刷（左辺の表記は LOG）", 推定の観測数=124,
                     DLOG型で推定した前期の係数=B_LAG, DLOG型で推定した自由度修正RSQ=0.478962))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "output" / "experiment_itr_form.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))
    print("\n仮定したデータ生成過程のもとでは、印刷の（係数 -0.959, RSQ 0.479）は DLOG 型で推定した値の範囲に入り、"
          "水準式で推定した値の範囲（RSQ ≈ 0.8〜0.95）から大きく外れる。DLOG 解釈を強く支持する補助的な証拠だが、"
          "原データの再推定・正誤表・著者資料による確証ではない。")



def reestimate() -> pd.DataFrame:
    """同梱の四半期データで ITR を作り、1995Q1〜2020Q4 で水準式と DLOG 型を推定し直す."""
    import sna  # ESRI 年次推計の四半期シート（原系列）

    inc = sna.income_block().loc["1994Q1":"2020Q4"]
    otyd = inc.YDV - (inc.YWV + inc.BSSV + inc.YIEV + inc.YICV - inc.TYPV - inc.CSSV)
    base_raw = inc.YWV + inc.BSSV + inc.YIEV + inc.YICV + otyd
    sa = pd.read_csv(ROOT / "data" / "processed" / "sna_quarterly.csv", index_col=0)
    sa.index = pd.PeriodIndex(sa.index, freq="Q")
    sa = sa.loc["1994Q1":"2020Q4"]
    base_sa = sa.YWV + sa.BSSV + sa.YIEV + sa.YICV + sa.OTYDV
    series = {
        "季節調整済み・課税ベース4期平均（モデルと同じ）": sa.TYPV / base_sa.rolling(4).mean(),
        "原系列・課税ベース4期平均": inc.TYPV / base_raw.rolling(4).mean(),
        "原系列・同じ期の課税ベース": inc.TYPV / base_raw,
    }
    rows = []
    for name, itr in series.items():
        x = np.log(itr / itr.loc["1995Q1":"2020Q4"].mean()).loc["1994Q4":"2020Q4"].dropna()
        y, lag = x.values[1:], x.values[:-1]
        for form, lhs in (("水準式", y), ("DLOG型", y - lag)):
            X = np.column_stack([np.ones(len(lhs)), lag])
            b = np.linalg.lstsq(X, lhs, rcond=None)[0]
            e = lhs - X @ b
            n, k = X.shape
            r2 = 1 - (e @ e / (n - k)) / (((lhs - lhs.mean()) ** 2).sum() / (n - 1))
            rows.append(dict(ITRの作り方=name, 形=form, 推定期間=f"{x.index[1]}〜{x.index[-1]}", 観測数=n,
                             前期の係数=b[1], 自由度修正RSQ=r2, SER=np.sqrt(e @ e / (n - k))))
    rows.append(dict(ITRの作り方="論文2022年版の印刷", 形="（左辺の表記は LOG）", 推定期間="1990Q1〜2020Q4", 観測数=124,
                     前期の係数=B_LAG, 自由度修正RSQ=0.478962, SER=SER))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "output" / "experiment_itr_reestimate.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
    print(out.round(3).to_string(index=False))
    print("\nどの作り方でも、印刷の（係数 -0.959, RSQ 0.479, SER 0.062）の組は再現できない。ESRI の ITR の作り方"
          "（季節調整の方法・税の計上時期など）が同梱データから再現できないため、原データでの確証は得られない。")
    return out


if __name__ == "__main__":
    main()
    print()
    reestimate()
