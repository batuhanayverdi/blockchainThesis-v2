import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
W, H = 17.0, 11.6
fig = plt.figure(figsize=(W/2.54, H/2.54), dpi=300)
ax = fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis("off")
NAVY="#1f3a5f"; BLUE="#2f6db3"; GREY="#f3f5f8"; ORANGE="#b8741a"; GREEN="#3d7a4a"
TS=6.4; BS=5.7
def box(x,y,w,h,title,body,fc=GREY,ec=NAVY,tc=NAVY):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.1",fc=fc,ec=ec,lw=0.7))
    ax.text(x+w/2,y+h-0.22,title,ha="center",va="top",fontsize=TS,fontweight="bold",color=tc,linespacing=1.05)
    ax.text(x+w/2,y+0.12,body,ha="center",va="bottom",fontsize=BS,color="#222222",linespacing=1.15)
def arrow(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=6,lw=0.7,color=NAVY,shrinkA=0,shrinkB=0))
def label(x,y,t): ax.text(x,y,t,fontsize=BS+0.4,color=BLUE,fontweight="bold",ha="left",va="center")
w=3.08; g=0.3; x0=0.3; xs=[x0+i*(w+g) for i in range(5)]
# Row A
yA=9.2; hA=1.75
label(x0,11.2,"A   Data collection and preprocessing (Section 3.1)")
box(xs[0],yA,w,hA,"Retrieval (HiveSQL)","all posts, 1 Sep 2024 to\n20 Jan 2025, > 100 characters\nn = 331,178")
box(xs[1],yA,w,hA,"Language filter","fastText, at least 90%\nEnglish characters\nn = 200,821")
box(xs[2],yA,w,hA,"Cleaning, topic labels","footers, code, tags removed;\n18 content categories\n(gpt-4o-mini)")
box(xs[3],yA,w,hA,"Stage 1: check-worthy?","ClaimBuster (DeBERTa-v2);\nhighest sentence score > 0.85\nn = 43,277",fc="#e8eef7")
box(xs[4],yA,w,hA,"Stratified sample","all 3,121 political posts +\nmatched non-political posts\nn = 6,242")
for i in range(4): arrow(xs[i]+w,yA+hA/2,xs[i+1],yA+hA/2)
# Row B
yB=6.05; hB=1.75
label(x0,8.1,"B   Content accuracy pipeline (Section 3.2)")
box(xs[0],yB,w,hB,"Stage 2: claims, queries","GPT-5-mini selects up to 3\ncheck-worthy claims and\nwrites one query each",fc="#e8eef7")
box(xs[1],yB,w,hB,"Stage 3: web evidence","Google via Serper API;\n10 organic results, snippet\nand knowledge panel",fc="#e8eef7")
box(xs[2],yB,w,hB,"Stage 4: verdict, stance","GPT-5-mini; six-level scale\n(true ... pants on fire,\nunverified); asserted or quoted",fc="#e8eef7")
box(xs[3],yB,w,hB,"Analysis sample","at least one asserted claim\nwith a verdict; n = 5,245\nposts by 1,650 authors")
box(xs[4],yB,w,hB,"Misleading-claim count","asserted false-side claims\nper post: 0, 1, 2 or 3\n(independent variable)",fc="#fbeede",ec=ORANGE,tc=ORANGE)
for i in range(4): arrow(xs[i]+w,yB+hB/2,xs[i+1],yB+hB/2)
ax.plot([xs[4]+w/2,xs[4]+w/2,xs[0]+w/2,xs[0]+w/2],[yA,yA-0.4,yA-0.4,yB+hB+0.2],color=NAVY,lw=0.7)
arrow(xs[0]+w/2,yB+hB+0.2,xs[0]+w/2,yB+hB)
# Row C: measures
yC=2.95; hC=1.75; wc=(W-x0-0.85-3*g)/4; xc=[x0+i*(wc+g) for i in range(4)]
label(x0,4.95,"C   Further measures (Section 3.3) and model inputs (Section 3.5)")
box(xc[0],yC,wc,hC,"Emotional language","DistilRoBERTa emotion classifier\non the same cleaned text;\nsix emotions, z-standardised",fc="#fbeede",ec=ORANGE,tc=ORANGE)
box(xc[1],yC,wc,hC,"Sentence-level dynamics","volatility and mean of each\nemotion across sentences\n(pysbd segmentation)",fc="#fbeede",ec=ORANGE,tc=ORANGE)
box(xc[2],yC,wc,hC,"Engagement outcomes","read from the chain after the\nseven-day window: net votes,\nreblogs, payout",fc="#e7f0e8",ec=GREEN,tc=GREEN)
box(xc[3],yC,wc,hC,"Controls","length, images, links, tags,\npolitical, weekend; author\nstanding before 1 Sep 2024",fc="#e7f0e8",ec=GREEN,tc=GREEN)
# Row D: model bar
yD=0.3; hD=1.35
box(x0,yD,W-2*x0,hD,"Regression models","OLS with HC1 robust standard errors; category, month and author fixed effects;\nfour specifications per outcome; emotion x misleading-claim interactions for H2",fc="#1f3a5f",ec=NAVY,tc="white")
ax.texts[-1].set_color("white")
for i in range(4): arrow(xc[i]+wc/2,yC,xc[i]+wc/2,yD+hD)
# misleading-claim count down to model bar (route right of row C)
xr=W-0.45
ax.plot([xs[4]+w/2,xs[4]+w/2,xr,xr],[yB,yB-0.35,yB-0.35,yD+hD+0.2],color=ORANGE,lw=0.8)
ax.add_patch(FancyArrowPatch((xr,yD+hD+0.2),(xr,yD+hD),arrowstyle="-|>",mutation_scale=6,lw=0.8,color=ORANGE,shrinkA=0,shrinkB=0))
fig.savefig("pipeline.png",dpi=300,facecolor="white"); print("ok")
