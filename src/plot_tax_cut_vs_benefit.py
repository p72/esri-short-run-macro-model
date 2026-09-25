"""名目GDP1%規模の消費税減税 vs 給付金（所得減税）: 実質GDPと財政収支/GDP の3年間の四半期経路を計算して描く.

- 消費税減税: このモデルの税収ベース（2018年、税率8%）で事前の税収減が名目GDPの1%になる引下げ幅（2.21%pt）を恒久的に与える
  （参考として 2.0%pt も計算。論文の乗数表の線形換算に相当）
- 給付金: 個人所得税を名目GDPの1%減税（論文シナリオ(4)と同じ。モデルでは給付金と可処分所得への効き方が同じ）
- 解く期間は 2017Q3〜2020Q4。実施前の2四半期にはリード項による買い控えが出る

入力: data/processed/model_data.csv（既定の版）
出力: output/tax_cut_vs_benefit.csv（四半期経路）, output/tax_cut_vs_benefit.png（日本語フォント IPAPGothic が必要）
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import pandas as pd, numpy as np
import model as M, simulate as S, ecm as E
data=pd.read_csv(ROOT / 'data/processed/model_data.csv', index_col='period'); data.index=pd.PeriodIndex(data.index,freq='Q')
m=M.Model(); af=m.add_factors(data,S.SOLVE_START,S.END); base=m.solve(data,S.SOLVE_START,S.END,af)
over,cols=E.frozen_overrides(m,base,S.solve_mask(base.index)); d0=base.copy()
for k,v in cols.items(): d0[k]=v
idx=base.index; on=pd.Series(S.sim_mask(idx).astype(float),index=idx); after=pd.Series((idx>=pd.Period("2018Q1","Q")).astype(float),index=idx)
b18=base.loc['2018Q1':'2018Q4']; tciv,gdpv,prt=b18.TCIV.mean(),b18.GDPV.mean(),b18.PRTCP.iloc[0]
f=lambda r: r/(1+r*prt); target=(1-0.01*gdpv/tciv)**(1/0.915)*f(0.08)
x_eq=0.08-max(r for r in np.arange(0.03,0.08,0.00001) if f(r)<=target)
def run(data_over=None,shocks=None):
    d=d0.copy()
    for k,v in (data_over or {}).items(): d[k]=v
    return m.solve(d,S.SOLVE_START,S.END,af,overrides=over,shocks=shocks)
runs={"benefit":run(shocks={"TYPV":-0.01*base["GDPV"]*on}),
      "ctax":run(data_over={"RTCI":base["RTCI"]-x_eq*after}),
      "ctax2":run(data_over={"RTCI":base["RTCI"]-0.02*after})}
win=slice("2017Q3","2020Q4")
out={}
for k,sh in runs.items():
    out[(k,"GDP")]=((sh.loc[win,"GDP"]/base.loc[win,"GDP"]-1)*100)
    out[(k,"BGV")]=(sh.loc[win,"BGVATGDPV"]-base.loc[win,"BGVATGDPV"])
    out[(k,"CP")]=((sh.loc[win,"CP"]/base.loc[win,"CP"]-1)*100)
df=pd.DataFrame(out); df.index=df.index.astype(str); df.to_csv(ROOT / 'output/tax_cut_vs_benefit.csv')
print("消費税の引下げ幅（事前コスト=名目GDP1%）:", round(x_eq*100,2), "pt")
pd.set_option("display.width",200); print(df.round(3).to_string())


# ---- 作図 ----
d=pd.read_csv(ROOT / "output/tax_cut_vs_benefit.csv",header=[0,1],index_col=0)
BLUE,ORANGE,INK,INK2,GRID="#2a78d6","#eb6834","#0b0b0b","#52514e","#e6e5e1"
plt.rcParams.update({"font.family":"IPAPGothic","font.size":9.5,"axes.edgecolor":GRID,"axes.labelcolor":INK2,"xtick.color":INK2,"ytick.color":INK2,"axes.unicode_minus":False})
fig,axes=plt.subplots(1,2,figsize=(12.5,6.2)); fig.patch.set_facecolor("white")
x=range(len(d)); labels=list(d.index); impl=labels.index("2018Q1")
for ax,(var,title,unit) in zip(axes,[("GDP","実質GDP","基準解からの乖離、%"),("BGV","財政収支／名目GDP","基準解からの乖離、%ポイント")]):
    ax.axhline(0,color=GRID,lw=1)
    ax.axvspan(-0.5,impl-0.5,color="#f3f2ee",zorder=0)
    ax.axvline(impl-0.5,color=INK2,lw=0.8,ls=":")
    ax.plot(x,d[("ctax",var)],color=BLUE,lw=2.2,marker="o",ms=3.5,label="消費税減税（−2.2%pt、事前の税収減＝名目GDP1%）")
    ax.plot(x,d[("benefit",var)],color=ORANGE,lw=2.2,ls=(0,(4,2)),marker="o",ms=3.5,label="給付金・所得減税（名目GDP1%）")
    for key,col,dy in [("ctax",BLUE,5),("benefit",ORANGE,-5)]:
        ax.annotate(f"{d[(key,var)].iloc[-1]:+.2f}",(x[-1],d[(key,var)].iloc[-1]),xytext=(6,dy),textcoords="offset points",color=col,fontsize=9,va="center")
    ax.set_title(f"{title}（{unit}）",loc="left",fontsize=11,color=INK,pad=8)
    ticks=[i for i,l in enumerate(labels) if l.endswith("Q1")]+[len(labels)-1]
    ax.set_xticks(ticks); ax.set_xticklabels([labels[i] for i in ticks]); ax.set_xlim(-0.5,len(labels)+0.8)
    ax.grid(axis="y",color=GRID,lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
ax=axes[0]
ax.annotate("実施前の買い控え\n（消費関数の\nリード項）",(1,d[("ctax","GDP")].iloc[1]),xytext=(2.1,-0.62),textcoords="data",color=INK2,fontsize=8.5,arrowprops=dict(arrowstyle="-",color=INK2,lw=0.8))
ax.text(impl-0.3,-0.12,"実施（2018Q1〜、恒久）",color=INK2,fontsize=8.5,va="top")
ax.annotate("駆け込み反動と\n実質所得増が同時に",(2,d[("ctax","GDP")].iloc[2]),xytext=(3.6,0.66),textcoords="data",color=INK2,fontsize=8.5,arrowprops=dict(arrowstyle="-",color=INK2,lw=0.8))
ax=axes[1]
ax.annotate("消費税分の税収減は名目GDP比 −0.9〜−1.0。\n差の大半（約0.2）は、実質固定の政府消費・\n公共投資の名目額が物価低下で減る会計効果",(4,d[("ctax","BGV")].iloc[4]),xytext=(1.6,-0.20),textcoords="data",color=INK2,fontsize=8.5,arrowprops=dict(arrowstyle="-",color=INK2,lw=0.8))
ax.annotate("2019Q4〜: 基準解の税率が10%に上がり\n同じ2.2%pt の税収比が縮小（コロナ期は税収基盤が変動）",(9,d[("ctax","BGV")].iloc[9]),xytext=(7.2,-0.30),textcoords="data",color=INK2,fontsize=8.5,arrowprops=dict(arrowstyle="-",color=INK2,lw=0.8))
h,l=axes[0].get_legend_handles_labels(); fig.legend(h,l,loc="upper left",bbox_to_anchor=(0.01,0.905),ncol=2,frameon=False,fontsize=9.5)
fig.suptitle("名目GDP1%規模の消費税減税 vs 給付金：実質GDPと財政収支の3年間の経路",x=0.01,ha="left",fontsize=13.5,color=INK,fontweight="bold",y=0.985)
fig.text(0.01,0.935,"内閣府 短期日本経済マクロ計量モデル（2022年版、ESRI Research Note No.72）の Python 再現で計算。基準解＝2018Q1〜2020Q4 の実績、ショックは恒久。",fontsize=9,color=INK2)
_dd = d.loc["2018Q1":"2020Q4"].copy(); _dd.index = [i[:4] for i in _dd.index]
_ann = lambda k, v: "/".join(f"{x:+.2f}".replace("-", "−") for x in _dd[(k, v)].groupby(level=0).mean())
note=("注: 消費税減税は税率 10%→8% 相当ではなく、このモデルの税収ベース（2018年、税率8%）で事前の税収減が名目GDPの1%になる 2.21%pt の引下げ。2.0%pt なら実質GDP +0.44/+0.40/+0.36（年平均）。\n"
      "給付金は個人所得税の減税として与える（モデルでは可処分所得への効き方が同じ）。消費税のデフレーター転嫁率は 0.52（残りは企業の取り分）。誤差修正項は消費・個人企業所得・消費デフレーターの3本を有効化、他は標準解で固定。\n"
      f"年平均の実質GDP乗数: 消費税減税 {_ann('ctax', 'GDP')}、給付金 {_ann('benefit', 'GDP')}。財政収支/GDP: 消費税減税 {_ann('ctax', 'BGV')}、給付金 {_ann('benefit', 'BGV')}。")
fig.text(0.01,0.012,note,fontsize=8,color=INK2,linespacing=1.55,va="bottom")
fig.tight_layout(rect=(0,0.14,1,0.86),w_pad=2.5); fig.savefig(ROOT / "output/tax_cut_vs_benefit.png",dpi=160,facecolor="white"); print(ROOT / "output/tax_cut_vs_benefit.png")
