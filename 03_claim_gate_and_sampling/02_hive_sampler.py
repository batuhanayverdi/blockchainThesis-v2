"""
Hive sampler - builds the balanced political / non-political sample for the LLM stage
=====================================================================================
Turns the ClaimBuster scores into the gated, balanced sample that the Model 3 and
Model 1 fact-check pipelines will run on.

Logic:
  - Passers = posts with cb_max_score > CB_THRESHOLD.
  - is_political = classification == "YES", OR category in the 9 political
    categories, OR (toggle) category == "non_us_politics".
  - Political set = ALL political passers (optionally capped by MAX_POLITICAL).
  - Non-political set = exactly the political count, drawn from the non-political
    pool (the 7 NONPOL_CATEGORIES; other_noise and non_us_politics excluded),
    monthly-matched to the political distribution computed after the gate, split
    as evenly as availability allows across the 7 categories, with shortfalls in
    thin cells redistributed. Fixed seed.

Inputs:
  hive_cb_scores.parquet   (doc_id, cb_max_score, ...)
  hive_cleaned.parquet     (author, permlink, title, body_for_analysis, created,
                            category, classification, net_votes, payout_combined, ...)

Output:
  hive_sample_master.parquet   the sampled posts with body_for_analysis and all
                               metadata + is_political + sample_stratum + month.
                               This is the input to the Model 3 / Model 1 stages
                               and the master table for the regression merge.

Run:  python hive_sampler.py
Deps: pip install pandas pyarrow numpy
"""

import os
import numpy as np
import pandas as pd


# ============================ CONFIG ========================================
SCORES_FILE = "hive_cb_scores.parquet"
CLEANED_FILE = "hive_cleaned.parquet"
OUT_MASTER = "hive_sample_master.parquet"

CB_THRESHOLD = 0.85
SEED = 42
INCLUDE_NON_US_POLITICS = True     # count category == non_us_politics as political
MAX_POLITICAL = None               # cap the political set (None = all passers)

POLITICAL_CATEGORIES = {
    "election_campaign", "election_fraud", "trump_administration",
    "biden_administration", "foreign_policy", "domestic_policy",
    "political_violence", "political_commentary", "other_political",
}
NONPOL_CATEGORIES = [
    "travel_lifestyle", "community_platform", "personal", "gaming",
    "entertainment", "crypto_finance", "technology",
]

# Columns dropped from the master to keep it lean (raw bodies not needed
# downstream; body_for_analysis is retained).
DROP_COLS = ["body", "body_clean"]


# ===================== STRATIFIED APPORTIONMENT =============================
def capped_apportion(total, weights, caps):
    """Allocate `total` units across keys by `weights`, each capped at caps[k],
    redistributing any overflow to keys with remaining capacity. Allocation is
    capped at the total available capacity."""
    keys = list(weights)
    alloc = {k: 0 for k in keys}
    total = int(min(total, sum(caps.values())))
    while sum(alloc.values()) < total:
        need = total - sum(alloc.values())
        active = [k for k in keys if alloc[k] < caps[k]]
        if not active:
            break
        wsum = sum(weights[k] for k in active)
        added = 0
        ideals = {}
        if wsum > 0:
            ideals = {k: need * weights[k] / wsum for k in active}
            for k in active:
                g = min(int(ideals[k]), caps[k] - alloc[k])
                if g > 0:
                    alloc[k] += g
                    added += g
        if added == 0:
            order = (sorted(active, key=lambda k: ideals[k] - int(ideals[k]),
                            reverse=True) if ideals else active)
            for k in order:
                if sum(alloc.values()) >= total:
                    break
                if alloc[k] < caps[k]:
                    alloc[k] += 1
    return alloc


def make_doc_id(df):
    return df["author"].astype(str) + "/" + df["permlink"].astype(str)


# ============================== MAIN ======================================
def main():
    for f in (SCORES_FILE, CLEANED_FILE):
        if not os.path.exists(f):
            print(f"Missing {f}.")
            return

    scores = pd.read_parquet(SCORES_FILE).drop_duplicates("doc_id")
    df = pd.read_parquet(CLEANED_FILE)
    df["doc_id"] = make_doc_id(df)
    df = df.drop_duplicates("doc_id")

    df = df.merge(scores[["doc_id", "cb_max_score"]], on="doc_id", how="inner")
    passers = df[df["cb_max_score"] > CB_THRESHOLD].copy()
    print(f"Passers (cb_max_score > {CB_THRESHOLD}): {len(passers):,}")

    passers["is_political"] = (
        (passers["classification"] == "YES")
        | passers["category"].isin(POLITICAL_CATEGORIES)
        | (INCLUDE_NON_US_POLITICS & (passers["category"] == "non_us_politics"))
    )
    passers["month"] = pd.to_datetime(
        passers["created"], errors="coerce"
    ).dt.strftime("%Y-%m")

    rng = np.random.default_rng(SEED)

    # -- political set --
    political = passers[passers["is_political"]].copy()
    if MAX_POLITICAL is not None and len(political) > MAX_POLITICAL:
        keep = rng.choice(political.index.to_numpy(), MAX_POLITICAL, replace=False)
        political = political.loc[keep]
    n_pol = len(political)
    print(f"Political passers (set): {n_pol:,}")

    # -- non-political pool --
    nonpol_pool = passers[
        (~passers["is_political"]) & (passers["category"].isin(NONPOL_CATEGORIES))
    ].copy()
    print(f"Non-political pool (7 categories): {len(nonpol_pool):,}")

    # -- monthly targets from the political distribution --
    pol_month = political["month"].value_counts().to_dict()
    avail_month = nonpol_pool["month"].value_counts().to_dict()
    months = sorted(set(pol_month) | set(avail_month))
    target_month = capped_apportion(
        n_pol,
        {m: pol_month.get(m, 0) for m in months},
        {m: avail_month.get(m, 0) for m in months},
    )

    # -- within each month, even split across categories with redistribution --
    chosen, realized_cat = [], {c: 0 for c in NONPOL_CATEGORIES}
    realized_month = {}
    for m in months:
        tm = target_month.get(m, 0)
        if tm <= 0:
            realized_month[m] = 0
            continue
        cell = nonpol_pool[nonpol_pool["month"] == m]
        caps_c = {c: int((cell["category"] == c).sum()) for c in NONPOL_CATEGORIES}
        alloc_c = capped_apportion(tm, {c: 1.0 for c in NONPOL_CATEGORIES}, caps_c)
        got = 0
        for c in NONPOL_CATEGORIES:
            k = alloc_c[c]
            if k <= 0:
                continue
            pool_c = cell[cell["category"] == c]
            pick = rng.choice(pool_c.index.to_numpy(), k, replace=False)
            chosen.extend(pick.tolist())
            realized_cat[c] += k
            got += k
        realized_month[m] = got

    nonpol_sample = nonpol_pool.loc[chosen].copy()
    n_nonpol = len(nonpol_sample)

    # -- assemble + write --
    political["sample_stratum"] = "political"
    nonpol_sample["sample_stratum"] = "nonpolitical"
    sample = pd.concat([political, nonpol_sample], ignore_index=True)
    sample = sample.drop(columns=[c for c in DROP_COLS if c in sample.columns])
    sample.to_parquet(OUT_MASTER, index=False)

    # -- report --
    print("\n" + "=" * 64)
    print("Sample report")
    print("=" * 64)
    print(f"\n  Monthly match (political distribution vs non-political draw):")
    print(f"    {'month':<9s}{'pol':>8s}{'pol %':>8s}{'np target':>11s}{'np real':>9s}")
    for m in months:
        pc = pol_month.get(m, 0)
        print(f"    {m:<9s}{pc:>8,}{pc / max(1, n_pol) * 100:>7.1f}%"
              f"{target_month.get(m, 0):>11,}{realized_month.get(m, 0):>9,}")
    print(f"    {'TOTAL':<9s}{n_pol:>8,}{'100.0%':>8s}{sum(target_month.values()):>11,}"
          f"{n_nonpol:>9,}")

    print(f"\n  Non-political draw by category (target even = {n_nonpol / 7:.0f} each):")
    for c in NONPOL_CATEGORIES:
        print(f"    {c:<20s}{realized_cat[c]:>8,}")

    print(f"\n  Sample size: {len(sample):,}  "
          f"(political {n_pol:,} + non-political {n_nonpol:,})")
    if n_nonpol < n_pol:
        print(f"  NOTE: non-political pool could not fully match the political "
              f"count ({n_nonpol:,} < {n_pol:,}); the pool is the limit.")
    print(f"\n  Wrote {OUT_MASTER}. This is the input to the Model 3 and Model 1 "
          f"stages, and the master table for the regression merge.")
    print(f"  Each of these {len(sample):,} posts goes through Stage 2 to 4 of "
          f"both pipelines, so this count drives the LLM-stage cost.")


if __name__ == "__main__":
    main()
