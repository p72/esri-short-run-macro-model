"""式129（所得実効税率 ITR）の左辺が水準か階差（DLOG）かを、印刷された推定統計量から判定する（Issue #2）.

論文（2022年版、Research Note No.72）の印刷:
    LOG(ITR/ITREQ) = -0.958520*LOG(ITR(-1)/ITREQ) + 0.625328*GDPGAP/100   [OLS 1990Q1-2020Q4, RSQ 0.478962]
前の版（2018年版、Research Note No.41）の印刷:
    LOG(ITR/ITREQ) =  0.504631*LOG(ITR(-1)/ITREQ) + 0.458211*GDPGAP/100   [OLS 1990Q1-2016Q4, RSQ 0.287657]

1本の説明変数に近い回帰では、水準式の決定係数はおよそ（前期の係数）^2 になる。
2018年版は 0.50^2 ≈ 0.25 と RSQ 0.29 が整合するが、2022年版は (-0.96)^2 ≈ 0.92 に対して RSQ 0.48 で整合しない。
左辺が DLOG なら、前期の係数 ≈ -1・決定係数 ≈ 0.5 になる（水準の自己相関がほぼ0の系列）。
印刷された係数と RSQ から、どちらの形で推定したときの統計量かを、両方の形のデータ生成過程で回帰を繰り返して確かめる。

出力: output/experiment_itr_form.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
B_LAG, B_GAP, SER = -0.958520, 0.625328, 0.062475  # 2022年版の印刷
N = 124  # 1990Q1〜2020Q4
REPS = 2000


def fit(y: np.ndarray, cols: list[np.ndarray]) -> tuple[float, float]:
    X = np.column_stack([np.ones(len(y))] + cols)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    return b[1], 1 - e @ e / ((y - y.mean()) @ (y - y.mean()))


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
        m = np.array(est).mean(axis=0)
        rows.append(dict(真の形=truth, 水準式で推定した前期の係数=m[0], 水準式で推定したRSQ=m[1],
                         DLOG型で推定した前期の係数=m[2], DLOG型で推定したRSQ=m[3]))
    rows.append(dict(真の形="論文2022年版の印刷", 水準式で推定した前期の係数=np.nan, 水準式で推定したRSQ=np.nan,
                     DLOG型で推定した前期の係数=B_LAG, DLOG型で推定したRSQ=0.478962))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "output" / "experiment_itr_form.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))
    print("\n印刷の（係数 -0.959, RSQ 0.479）は、DLOG型で推定したときの値と一致し、水準式の推定値（RSQ ≈ 0.9）とは一致しない。")
    print("2018年版（水準式、係数 +0.50, RSQ 0.29）は水準式で整合する。2022年版で推定の形が DLOG に変わり、印刷の 'D' が落ちたと判断する。")


if __name__ == "__main__":
    main()
