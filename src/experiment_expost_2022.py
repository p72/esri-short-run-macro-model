"""モデルと2022〜24年の実績は、どこで違うのか（事後シミュレーションと誤差項の診断）.

experiment_tot_policy.py では、2022年規模の交易損失を与えると、モデルは「2008年型」（国内物価がほとんど上がらず、
名目賃金が下がる）になった。実際の2022〜24年は、国内物価が上がり、賃金も上がった。その違いがモデルのどの式から
来るのかを調べる。

基準解は、各式に誤差項（Model.add_factors）を足して実績を再現している。そこで、

1. 誤差項の診断: 各式の2022〜24年の誤差項の平均を、平時（2014〜2019年）の平均・標準偏差と比べる（z 値）。
2. 事後シミュレーション: 2022Q1 から、外生変数は実績のまま、誤差項だけを平時の平均に置き換えて解く
   （＝モデルが予測する2022〜24年）。誤差修正項はすべて有効（モデル本来の動き）。
   為替 FXS と金利（RCD・RCDX・RGB・RGBX）は、実績の誤差項のままにする（金融市場は実績を与える）。
3. 要因分解: 式のグループごとに誤差項を実績に戻し、実績との差がどれだけ埋まるかを測る。
   - 1つずつ戻す: 事後シミュレーションに、そのグループの誤差項だけ実績を入れる
   - 全部戻してから1つずつ外す: 実績（全グループ実績）から、そのグループだけ平時の平均にする
   2つの方法の平均を寄与とし、残り（交差項）も示す。

指標は experiment_tot_policy.measures と同じ（実績の、モデルの予測からの乖離。GDP デフレーターの分配面・支出面の寄与）。

出力: output/experiment_expost_2022_residuals.csv（誤差項の診断）, output/experiment_expost_2022.csv（実績−予測と要因分解、年平均）,
      output/experiment_expost_2022_paths.csv（四半期の経路）,
      output/experiment_expost_2022_levels.csv（実績と予測の水準、2021Q4=100）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import model as M  # noqa: E402
from experiment_fx_passthrough import load  # noqa: E402
from experiment_tot_policy import measures  # noqa: E402

OUT = ROOT / "output"
FNAME, START, END = "model_data_v2024.csv", "2022Q1", "2024Q4"
NORMAL = ("2014Q1", "2019Q4")  # 稼働率 CUX のデータが2013Q1からのため、ラグを含めて2014年から
KEEP = {"FXS", "RCD", "RCDX", "RGB", "RGBX"}  # 金融市場は実績を与える
GROUPS = {  # 式番号の範囲 → グループ
    "需要・生産・雇用（式1〜54）": range(1, 55),
    "物価（式55〜82）": range(55, 83),
    "所得・分配（式83〜110）": range(83, 111),
    "金融・財政・その他（式111〜152）": range(111, 153),
}
KEY_EQS = ["PGDPAT", "PCPAT", "CGPIAT", "PNFMGSAT", "PFUELAT", "PXGS", "YWV", "LF", "CP"]  # 1本ずつの寄与も見る式
VARS = ["GDPデフレーター", "寄与_単位労働コスト", "寄与_単位利潤", "寄与_国内需要デフレーター", "寄与_交易条件",
        "消費デフレーター", "名目雇用者報酬", "1人あたり賃金", "実質賃金", "名目GDP", "実質GDP", "実質消費"]


def main() -> None:
    data = load(FNAME)
    need = pd.Period(END, "Q") + 2  # 消費の式が RTCI の2期先を見る
    data = data.reindex(pd.period_range(data.index.min(), need, freq="Q"))
    data["RTCI"] = data["RTCI"].ffill()
    m = M.Model()
    eqs = {e.name: e for e in m.eqs}
    af = m.add_factors(data, NORMAL[0], END)
    actual = m.solve(data, START, END, af)

    # 1. 誤差項の診断
    norm = af.loc[NORMAL[0]:NORMAL[1]]
    recent = af.loc[START:END]
    rows = []
    for name, e in eqs.items():
        sd = norm[name].std()
        if not np.isfinite(sd) or sd < 1e-12:
            continue  # 恒等式（誤差項が常に0）
        yearly = recent[name].groupby(recent.index.year).mean()
        rows.append({"式": e.no, "変数": name, "形": e.kind, "平時の平均": norm[name].mean(), "平時の標準偏差": sd,
                     **{f"{y}年の平均": v for y, v in yearly.items()},
                     "2022〜24年の平均": recent[name].mean(),
                     "z値（2022〜24年平均−平時平均）/平時標準偏差": (recent[name].mean() - norm[name].mean()) / sd})
    res = pd.DataFrame(rows).sort_values("z値（2022〜24年平均−平時平均）/平時標準偏差", key=abs, ascending=False)
    res.to_csv(OUT / "experiment_expost_2022_residuals.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

    # 2. 事後シミュレーション
    window = (af.index >= pd.Period(START, "Q")) & (af.index <= pd.Period(END, "Q"))
    normal_mean = norm.mean()

    def af_names(names: set[str]) -> pd.DataFrame:
        """names の式の誤差項を、2022〜24年だけ平時の平均に置き換える."""
        a = af.copy()
        for name in names - KEEP:
            a.loc[window, name] = normal_mean[name]
        return a

    def members(g: str) -> set[str]:
        return {name for name, e in eqs.items() if e.no in GROUPS[g]}

    def af_with(normal_groups: set[str]) -> pd.DataFrame:
        return af_names(set().union(*(members(g) for g in normal_groups)) if normal_groups else set())

    groups = list(GROUPS)
    expost = m.solve(data, START, END, af_with(set(groups)))
    check = m.solve(data, START, END, af_with(set()))
    gap = max((check[v] - actual[v]).abs().loc[START:END].max() / actual[v].abs().loc[START:END].mean()
              for v in ("GDP", "PGDP", "YWV", "PCP"))
    assert gap < 1e-6, f"誤差項を実績に戻しても実績を再現しない: {gap}"

    def annual(x: pd.DataFrame) -> pd.DataFrame:
        return x[VARS].groupby(x.index.year).mean()

    total = measures(actual, expost)  # 実績のモデル予測からの乖離
    out = [annual(total).assign(区分="実績−モデルの予測（全体）")]
    paths = [total[VARS].assign(区分="実績−モデルの予測（全体）", 四半期=total.index.astype(str))]
    contrib_sum = 0
    for g in groups:
        one_on = measures(m.solve(data, START, END, af_with(set(groups) - {g})), expost)  # 1つずつ戻す
        one_off = measures(actual, m.solve(data, START, END, af_with({g})))               # 全部戻して1つ外す
        c = (one_on[VARS] + one_off[VARS]) / 2
        contrib_sum = contrib_sum + c
        out.append(annual(c).assign(区分=f"寄与: {g}"))
        out.append(annual(one_on).assign(区分=f"（1つずつ戻す）{g}"))
        out.append(annual(one_off).assign(区分=f"（全部戻して1つ外す）{g}"))
        paths.append(c.assign(区分=f"寄与: {g}", 四半期=c.index.astype(str)))
    # 主な式を1本ずつ（他の式は、1つずつ戻す方法では平時、全部戻して1つ外す方法では実績）
    every = set(eqs)
    for name in KEY_EQS:
        one_on = measures(m.solve(data, START, END, af_names(every - {name})), expost)
        one_off = measures(actual, m.solve(data, START, END, af_names({name})))
        c = (one_on[VARS] + one_off[VARS]) / 2
        out.append(annual(c).assign(区分=f"式{eqs[name].no} {name} だけの寄与"))
    inter = total[VARS] - contrib_sum
    out.append(annual(inter).assign(区分="残り（グループ間の交差項）"))
    df = pd.concat(out)
    df.insert(0, "年", df.index)
    df = df[["区分", "年"] + VARS]
    df.to_csv(OUT / "experiment_expost_2022.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    p = pd.concat(paths, ignore_index=True)
    p = p[["区分", "四半期"] + VARS]
    p.to_csv(OUT / "experiment_expost_2022_paths.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

    # 水準（2021Q4=100）: 実績とモデルの予測
    lv = {}
    w = slice("2019Q1", END)
    for lab, f in [("GDPデフレーター", lambda x: x.GDPV / x.GDP), ("消費デフレーター", lambda x: x.PCP),
                   ("名目雇用者報酬", lambda x: x.YWV), ("実質賃金", lambda x: x.W / x.PCP), ("実質GDP", lambda x: x.GDP)]:
        for nm, x in (("実績", actual), ("モデルの予測", expost)):
            ser = f(x).loc[w]
            lv[f"{lab}|{nm}"] = ser / f(actual).loc["2021Q4"] * 100
    lvd = pd.DataFrame(lv)
    lvd.index = lvd.index.astype(str)
    lvd.to_csv(OUT / "experiment_expost_2022_levels.csv", encoding="utf-8-sig", float_format="%.6f")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("誤差項の z 値が大きい式（上位20）")
    print(res.head(20).round(4).to_string(index=False))
    print("\n実績−モデルの予測と、式のグループ別の寄与（年平均）")
    show = df[~df["区分"].str.startswith("（")]
    print(show.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
