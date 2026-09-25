"""
Hive political identity language, v2  (Rathje et al. 2021 dictionaries)
=======================================================================
Changes from v1, after the false-positive scan:
  1. MIN_ENTRY_CHARS = 3. The comma-splitting loader had turned suffixes like
     "Jr" from "Donald Trump, Jr." into standalone entries; "jr" then matched
     150 times in unrelated text. Entries shorter than 3 characters are now
     dropped and listed in the console.
  2. Deduplicated side totals. Names appearing in two subdictionaries (e.g.
     a senator in both the famous and congressional lists) were counted twice
     in pol_total. Side totals are now computed from a union pattern per side,
     so each mention counts once. Subdictionary counts remain as descriptive
     columns and may sum to more than the deduplicated total.

FRAMING UNCHANGED: this measures political identity references overall, NOT
in-group vs out-group; author ideology is unknown on Hive. US-centric
dictionaries and polysemous identity terms are accepted for fidelity to the
source method and noted as caveats.

Outputs (in OUT_DIR):
  hive_political_identity.parquet   one row per post:
    pol_dem_identity, pol_rep_identity      per-subdictionary counts
    pol_dem_famous,   pol_rep_famous        (descriptive; may double count
    pol_dem_congress, pol_rep_congress       names present in two lists)
    pol_dem_total,    pol_rep_total         DEDUPLICATED side totals
    pol_total                               deduplicated overall count
    pol_log1p, pol_density100, pol_any      transformations
    n_words                                 word count of the matched text

Requirements: pandas, numpy, pyarrow, openpyxl (installed). Runs in ~1 min.
"""

import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

DICT_DIR = r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_hive_pipeline\10. Political Identity Language\OSF repository"
SAMPLE_FILE = r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_hive_pipeline\hive sample preparation (politic-non-politic)\hive_sample_master.parquet"
DOC_ID_FILTER = r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_hive_pipeline\9. Sentence Emotion Scoring\doc_id.xlsx"
OUT_DIR = r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_hive_pipeline\10. Political Identity Language"

TEXT_COLUMN = "body_for_analysis"
TEXT_FALLBACKS = ["body_english_extracted", "body_clean", "body"]

DICT_FILES = {
    ("dem", "identity"): "LiberalIdentity.txt",
    ("rep", "identity"): "ConservativeIdentity.txt",
    ("dem", "famous"):   "MostFamousDemocrats.txt",
    ("rep", "famous"):   "MostFamousRepublicans.txt",
    ("dem", "congress"): "DemocratCongress.txt",
    ("rep", "congress"): "RepublicansCongress.txt",
}

MIN_ENTRY_CHARS = 3
TOP_MATCHES_TO_SHOW = 25


# ============================================================================
# DICTIONARY LOADING AND NORMALISATION
# ============================================================================

def norm(s: str) -> str:
    """Lowercase, punctuation to spaces, collapse whitespace. Applied to both
    dictionary entries and post text so phrase matching is consistent."""
    s = s.lower()
    s = re.sub(r"[^a-z0-9@ ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_dictionary(path: Path, dropped_log: list):
    """One entry per line; commas within a line split into separate entries.
    Entries shorter than MIN_ENTRY_CHARS are dropped and logged."""
    entries = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            for part in line.split(","):
                e = norm(part)
                if not e:
                    continue
                if len(e) < MIN_ENTRY_CHARS:
                    dropped_log.append((path.name, e))
                    continue
                entries.append(e)
    return sorted(set(entries), key=len, reverse=True)


def compile_pattern(entries):
    alts = "|".join(re.escape(e) for e in entries)
    return re.compile(rf"\b(?:{alts})\b")


def main() -> None:
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hive_political_identity.parquet"

    print("=" * 72)
    print("Political identity language v2 (Rathje et al. 2021 dictionaries)")
    print("=" * 72)

    # ---- load dictionaries ----
    dropped_log = []
    dicts = {}
    for (side, kind), fname in DICT_FILES.items():
        fpath = Path(DICT_DIR) / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Dictionary not found: {fpath}")
        dicts[(side, kind)] = load_dictionary(fpath, dropped_log)
        print(f"  {fname:<28s} {len(dicts[(side, kind)]):>4d} entries")

    if dropped_log:
        print(f"\n  Entries dropped by the {MIN_ENTRY_CHARS}-character rule:")
        for fname, e in dropped_log:
            print(f"    {fname:<28s} '{e}'")
    else:
        print("\n  No entries dropped by the length rule.")

    # per-subdictionary patterns (descriptive counts)
    sub_patterns = {k: compile_pattern(v) for k, v in dicts.items()}

    # deduplicated union per side (authoritative totals)
    side_entries = {
        "dem": sorted(set(dicts[("dem", "identity")]) |
                      set(dicts[("dem", "famous")]) |
                      set(dicts[("dem", "congress")]),
                      key=len, reverse=True),
        "rep": sorted(set(dicts[("rep", "identity")]) |
                      set(dicts[("rep", "famous")]) |
                      set(dicts[("rep", "congress")]),
                      key=len, reverse=True),
    }
    side_patterns = {s: compile_pattern(e) for s, e in side_entries.items()}
    overlap_dem = (len(dicts[("dem", "identity")]) +
                   len(dicts[("dem", "famous")]) +
                   len(dicts[("dem", "congress")]) -
                   len(side_entries["dem"]))
    overlap_rep = (len(dicts[("rep", "identity")]) +
                   len(dicts[("rep", "famous")]) +
                   len(dicts[("rep", "congress")]) -
                   len(side_entries["rep"]))
    print(f"\n  Union dedup: dem {len(side_entries['dem'])} entries "
          f"({overlap_dem} duplicates removed), "
          f"rep {len(side_entries['rep'])} entries "
          f"({overlap_rep} duplicates removed)")

    # ---- load posts and filter to the regression sample ----
    df = pd.read_parquet(SAMPLE_FILE)
    if "doc_id" in df.columns:
        df["_key"] = df["doc_id"].astype(str).str.strip()
    else:
        df["_key"] = df["author"].astype(str) + "/" + df["permlink"].astype(str)
    ids = pd.read_excel(DOC_ID_FILTER, usecols=[0])
    keep_ids = set(ids[ids.columns[0]].dropna().astype(str).str.strip())
    before = len(df)
    df = df[df["_key"].isin(keep_ids)].copy()
    print(f"\n  Sample filter: {before:,} -> {len(df):,} posts (expected 5,565)")

    text_col = TEXT_COLUMN if TEXT_COLUMN in df.columns else None
    if text_col is None:
        for c in TEXT_FALLBACKS:
            if c in df.columns:
                text_col = c
                break
    print(f"  Text column: {text_col}")

    # ---- count ----
    print("  Counting ...")
    match_counter = Counter()
    naive_total_sum = 0
    rows = []
    for _, row in df.iterrows():
        text = norm(str(row.get(text_col) or ""))
        n_words = len(text.split())
        rec = {"doc_id": row["_key"], "n_words": n_words}

        # descriptive subdictionary counts (may overlap across lists)
        for (side, kind), pat in sub_patterns.items():
            rec[f"pol_{side}_{kind}"] = len(pat.findall(text))

        # authoritative deduplicated side totals
        for side, pat in side_patterns.items():
            hits = pat.findall(text)
            rec[f"pol_{side}_total"] = len(hits)
            match_counter.update(hits)

        rec["pol_total"] = rec["pol_dem_total"] + rec["pol_rep_total"]
        naive_total_sum += (rec["pol_dem_identity"] + rec["pol_dem_famous"] +
                            rec["pol_dem_congress"] + rec["pol_rep_identity"] +
                            rec["pol_rep_famous"] + rec["pol_rep_congress"])
        rec["pol_log1p"] = float(np.log1p(rec["pol_total"]))
        rec["pol_density100"] = (100.0 * rec["pol_total"] / n_words
                                 if n_words else 0.0)
        rec["pol_any"] = int(rec["pol_total"] > 0)
        rows.append(rec)

    out = pd.DataFrame(rows)
    out.to_parquet(out_path, index=False)
    print(f"  Wrote {len(out):,} rows to {out_path}")

    # ---- diagnostics ----
    s = out["pol_total"]
    dedup_sum = int(s.sum())
    print(f"\n  Dedup effect: naive subdictionary sum {naive_total_sum:,} vs "
          f"deduplicated total {dedup_sum:,} "
          f"({naive_total_sum - dedup_sum:,} double counts removed)")
    print("\n  pol_total distribution (deduplicated):")
    print(f"    zero share: {(s == 0).mean() * 100:.1f}%   any-mention posts: "
          f"{(s > 0).sum():,}")
    print(f"    mean {s.mean():.2f}   p50 {s.quantile(0.50):.0f}   "
          f"p90 {s.quantile(0.90):.0f}   p99 {s.quantile(0.99):.0f}   "
          f"max {s.max():.0f}")
    print(f"    side split among mentions: dem "
          f"{out['pol_dem_total'].sum():,} vs rep "
          f"{out['pol_rep_total'].sum():,}")

    print(f"\n  Top {TOP_MATCHES_TO_SHOW} matched entries "
          f"(scan again for false positives):")
    for term, n in match_counter.most_common(TOP_MATCHES_TO_SHOW):
        print(f"    {term:<30s} {n:>7,}")

    print("\nDone. Send the console output back. If the list is clean, "
          "Table G (pol_log1p primary, pol_any variant) is next.")


if __name__ == "__main__":
    main()
