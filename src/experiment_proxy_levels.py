"""代用系列の水準・傾きが乗数に影響しないことを確かめる（Issue #3 の項目3）.

世帯数 HH（式5 の長期均衡に対数で入る）と市街地価格指数 PLAND（式119 の長期均衡に対数で入る）について、
系列に定数倍と年率8%のトレンドを掛けた極端なデータで11シナリオを解き直し、既定のデータとの乗数の差を出す。
誤差修正項が既定（11本固定）の場合と、すべて動かす場合の両方で確かめる。
対数で入る外生・内生変数の水準の違いは、各期の誤差項（アドファクター）に吸収されるので、差は0になるはず。

出力: output/experiment_proxy_levels.csv
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"


def distorted(d: pd.DataFrame, var: str) -> pd.DataFrame:
    d = d.copy()
    t = np.arange(len(d))
    d[var] = d[var] * 1.5 * np.exp(0.02 * t)
    if var == "PLAND":  # 式103 LANDV = PLAND × RLAND を実績で閉じ直す
        d["RLAND"] = d["LANDV"] / d["PLAND"]
    return d


def main() -> None:
    base = pd.read_csv(ROOT / "data" / "processed" / "model_data.csv", index_col="period")
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for var in ("HH", "PLAND"):
            f = Path(tmp) / f"model_data_{var}.csv"
            distorted(base, var).to_csv(f)
            for ecm in ("esri", "live"):
                tag = f"_proxytest{var}"
                subprocess.run([sys.executable, str(ROOT / "src" / "simulate.py"), "--data", str(f), "--tag", tag, "--ecm", ecm],
                               check=True, capture_output=True)
                sfx = "" if ecm == "esri" else "_ecmlive"
                a = pd.read_csv(OUT / f"multipliers_reproduced{sfx}.csv")
                new_rep, new_cmp = OUT / f"multipliers_reproduced{tag}{sfx}.csv", OUT / f"multipliers_comparison{tag}{sfx}.csv"
                b = pd.read_csv(new_rep)
                keys = [c for c in a.columns if c != "value"]
                m = a.merge(b, on=keys)
                c = pd.read_csv(new_cmp).dropna(subset=["value_paper"])
                rows.append(dict(変数=var, 誤差修正項=ecm, 歪め方="×1.5×exp(0.02t)",
                                 乗数の最大差=float((m.value_x - m.value_y).abs().max()),
                                 一致率=round((c["diff"].abs() <= 0.1 + 1e-9).mean() * 100, 2)))
                new_rep.unlink()
                new_cmp.unlink()
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "experiment_proxy_levels.csv", index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
