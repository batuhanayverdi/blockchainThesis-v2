"""
Visualize the Hive sample: monthly and categorical distributions
================================================================
Reads hive_sample_master.parquet and writes verification charts to sample_viz/:

  1. monthly_distribution.png        political vs non-political counts per month
                                     (the two series should track each other,
                                     since the non-political draw is monthly-matched)
  2. nonpolitical_categories.png     the 7 non-political categories vs the even
                                     target line (technology may sit lower)
  3. political_categories.png        political category composition (not designed
                                     to be even; this is just what the data is)
  4. nonpol_month_category_heatmap.png   month x category for the non-political draw

It also prints each underlying table so you can check the numbers directly.

Run:  python visualize_sample.py
Deps: pip install pandas pyarrow matplotlib numpy
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


MASTER = "hive_sample_master.parquet"
OUTDIR = "sample_viz"

TUM_BLUE = "#0065BD"
TUM_ORANGE = "#E37222"
GRID = "#d0d0d0"

POLITICAL_ORDER = [
    "election_campaign", "election_fraud", "trump_administration",
    "biden_administration", "foreign_policy", "domestic_policy",
    "political_violence", "political_commentary", "other_political",
    "non_us_politics",
]
NONPOL_ORDER = [
    "travel_lifestyle", "community_platform", "personal", "gaming",
    "entertainment", "crypto_finance", "technology",
]


def _label_bars(ax, bars):
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:,.0f}", (b.get_x() + b.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=9,
                    xytext=(0, 2), textcoords="offset points")


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def main():
    if not os.path.exists(MASTER):
        print(f"Missing {MASTER}.")
        return
    os.makedirs(OUTDIR, exist_ok=True)

    df = pd.read_parquet(MASTER)
    if "month" not in df.columns:
        df["month"] = pd.to_datetime(df["created"], errors="coerce").dt.strftime("%Y-%m")
    months = sorted(df["month"].dropna().unique())

    pol = df[df["sample_stratum"] == "political"]
    npl = df[df["sample_stratum"] == "nonpolitical"]
    print(f"Sample: {len(df):,}  (political {len(pol):,}, non-political {len(npl):,})")

    # -- 1. monthly grouped bars --
    pol_m = pol.groupby("month").size().reindex(months, fill_value=0)
    npl_m = npl.groupby("month").size().reindex(months, fill_value=0)
    x = np.arange(len(months))
    w = 0.4
    fig, ax = plt.subplots(figsize=(max(8, len(months) * 1.1), 5))
    _label_bars(ax, ax.bar(x - w / 2, pol_m.values, w, label="Political", color=TUM_BLUE))
    _label_bars(ax, ax.bar(x + w / 2, npl_m.values, w, label="Non-political", color=TUM_ORANGE))
    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha="right")
    ax.set_ylabel("Posts")
    ax.set_title("Monthly distribution: political vs non-political")
    ax.legend()
    _despine(ax)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/monthly_distribution.png", dpi=150)
    plt.close(fig)
    print("\nMonthly counts:")
    print(pd.DataFrame({"political": pol_m, "non_political": npl_m}))

    # -- 2. non-political categories vs even target --
    nc = npl["category"].value_counts().reindex(NONPOL_ORDER, fill_value=0)
    target = len(npl) / len(NONPOL_ORDER)
    fig, ax = plt.subplots(figsize=(10, 5))
    _label_bars(ax, ax.bar(range(len(nc)), nc.values, color=TUM_ORANGE))
    ax.axhline(target, color=TUM_BLUE, ls="--", lw=1.5,
               label=f"even target = {target:.0f}")
    ax.set_xticks(range(len(nc)))
    ax.set_xticklabels(nc.index, rotation=45, ha="right")
    ax.set_ylabel("Posts")
    ax.set_title("Non-political sample by category")
    ax.legend()
    _despine(ax)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/nonpolitical_categories.png", dpi=150)
    plt.close(fig)
    print("\nNon-political by category:")
    print(nc)

    # -- 3. political categories --
    pc_raw = pol["category"].value_counts()
    order = ([c for c in POLITICAL_ORDER if c in pc_raw.index]
             + [c for c in pc_raw.index if c not in POLITICAL_ORDER])
    pc = pc_raw.reindex(order)
    fig, ax = plt.subplots(figsize=(11, 5))
    _label_bars(ax, ax.bar(range(len(pc)), pc.values, color=TUM_BLUE))
    ax.set_xticks(range(len(pc)))
    ax.set_xticklabels(pc.index, rotation=45, ha="right")
    ax.set_ylabel("Posts")
    ax.set_title("Political sample by category")
    _despine(ax)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/political_categories.png", dpi=150)
    plt.close(fig)
    print("\nPolitical by category:")
    print(pc)

    # -- 4. non-political month x category heatmap --
    piv = (pd.crosstab(npl["category"], npl["month"])
           .reindex(index=NONPOL_ORDER, columns=months, fill_value=0))
    fig, ax = plt.subplots(figsize=(max(8, len(months) * 1.0), 5))
    im = ax.imshow(piv.values, aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, rotation=45, ha="right")
    ax.set_yticks(range(len(NONPOL_ORDER)))
    ax.set_yticklabels(NONPOL_ORDER)
    vmax = piv.values.max() if piv.values.size else 1
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, str(v), ha="center", va="center", fontsize=8,
                    color="white" if v > vmax * 0.6 else "black")
    ax.set_title("Non-political sample: month x category")
    fig.colorbar(im, ax=ax, label="Posts")
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/nonpol_month_category_heatmap.png", dpi=150)
    plt.close(fig)

    print(f"\nWrote 4 charts to {OUTDIR}/")


if __name__ == "__main__":
    main()
