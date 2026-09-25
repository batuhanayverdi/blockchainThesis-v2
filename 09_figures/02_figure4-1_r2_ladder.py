import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
labels=["(1) Claim\ncategories only","(2) + post\ncontrols","(3) + category and\nmonth FE","(4) + author\nstanding","(5) + author\nFE"]
vals=[0.016,0.345,0.359,0.645,0.891]
fig,ax=plt.subplots(figsize=(15/2.54,7/2.54),dpi=300)
cols=["#c9d3e0","#9fb3cc","#7592b7","#3f6aa0","#1f3a5f"]
bars=ax.bar(range(5),vals,color=cols,width=0.62)
for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+0.015,f"{v:.3f}",ha="center",va="bottom",fontsize=7.5)
ax.set_xticks(range(5)); ax.set_xticklabels(labels,fontsize=6.8)
ax.set_ylabel("R-squared",fontsize=7.5); ax.set_ylim(0,1.0); ax.tick_params(axis="y",labelsize=7)
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.annotate("",xy=(2.72,0.64),xytext=(2.28,0.40),arrowprops=dict(arrowstyle="->",color="#b8741a",lw=1.2))
ax.text(2.0,0.72,"+0.286 from the four\nauthor standing measures",fontsize=6.5,color="#b8741a",ha="center")
plt.tight_layout(); fig.savefig("r2.png",dpi=300,facecolor="white"); print("ok")
