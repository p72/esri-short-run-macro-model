"""名目GDP水準目標（experiment_ngdplt.py の結果）での消費者物価の上昇率の図.

消費デフレーター（家計消費の物価）の前年比を、実績と、NGDPLT 3%・4%・5% をモデルに組み込んで解いた値で比べる。
モデルの解の水準は、実績の消費デフレーターに基準解からの乖離（%）を掛けて作る（2019年以前は実績のまま）。

出力: output/ngdplt_cpi.png（日本語フォント IPAPGothic または IPAGothic が必要）
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FONT = next((f for f in ("IPAPGothic", "IPAGothic") if any(f == x.name for x in font_manager.fontManager.ttflist)), "sans-serif")
d=pd.read_csv(ROOT / "output" / "experiment_ngdplt.csv",index_col=0)
m=pd.read_csv(ROOT / "data" / "processed" / "model_data_v2024.csv",index_col=0)["PCP"].loc["2019Q1":"2024Q4"]
BLUE,ORANGE,AQUA,INK,INK2,GRID="#2a78d6","#eb6834","#1baf7a","#0b0b0b","#52514e","#e6e5e1"
plt.rcParams.update({"font.family":FONT,"font.size":10.5,"axes.edgecolor":GRID,"axes.labelcolor":INK2,"xtick.color":INK2,"ytick.color":INK2,"axes.unicode_minus":False})
fig,ax=plt.subplots(figsize=(10,5.6)); fig.patch.set_facecolor("white")
q=list(d.index); x=np.arange(len(q))
def yoy(lv): return (lv/lv.shift(4)-1)*100
act=yoy(m).loc[q]
ser=[("実績",act,INK,"-")]
for g,c in (("3",AQUA),("4",BLUE),("5",ORANGE)):
    dev=d[f"モデル NGDPLT{g}% 消費デフレーター（基準解比 %）"]
    lv=m.copy(); lv.loc[q]=m.loc[q].values*(1+dev.values/100)
    ser.append((f"NGDPLT {g}%",yoy(lv).loc[q],c,"-"))
ends=[]
for lab,y,c,ls in ser:
    ax.plot(x,y.values,color=c,lw=2.4 if lab!="実績" else 2.6,label=lab); ends.append((y.iloc[-1],c))
ends.sort(); prev=None
for v,c in ends:
    yy=v if prev is None else max(v,prev+0.22); prev=yy
    ax.annotate(f"{v:.1f}%",(x[-1],v),xytext=(x[-1]+0.35,yy),textcoords="data",color=c,fontsize=10,va="center")
ax.axhline(0,color=INK2,lw=0.9); ax.axhline(2,color=INK2,lw=0.8,ls=(0,(4,3)))
ax.text(0.2,2.08,"2%",color=INK2,fontsize=9)
ax.set_xticks(x[::4]); ax.set_xticklabels([f"{s[:4]}年" for s in q[::4]]); ax.set_xlim(-0.4,len(q)+1.3)
ax.grid(axis="y",color=GRID,lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
for sp in ("top","right"): ax.spines[sp].set_visible(False)
ax.legend(loc="upper left",frameon=False)
fig.suptitle("名目GDP水準目標なら、消費者物価の上昇率はどうなったか",x=0.01,ha="left",fontsize=14,color=INK,fontweight="bold",y=0.98)
fig.text(0.01,0.915,"消費デフレーター（家計消費の物価）の前年比、%。NGDPLT はルールをモデルに組み込んで解いた値。",fontsize=9.5,color=INK2)
fig.text(0.01,0.015,"内閣府 短期日本経済マクロ計量モデル（2022年版）の Python 再現、2024年版データ。モデルの物価は CPI ではなく、国民経済計算の家計消費デフレーター。",fontsize=8,color=INK2)
fig.tight_layout(rect=(0,0.04,1,0.9)); fig.savefig(ROOT / "output" / "ngdplt_cpi.png",dpi=160,facecolor="white")
for lab,y,c,ls in ser: print(lab, y.groupby(y.index.str[:4]).mean().round(2).tolist())
