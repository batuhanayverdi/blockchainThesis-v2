"""
Hive descriptive post features, v2  (local, no SQL)
===================================================
v2: reads hive_cleaned.parquet (same January 2025 scrape, all needed
columns present) instead of the 2.1 GB consolidated archive; loads only
the required columns.
Builds the descriptive extras (image presence, link presence, tag count,
posting time) for the 5,565 regression posts WITHOUT querying HiveSQL.

Why no SQL: HiveSQL's Comments and Accounts tables are state tables holding
CURRENT (2026) values for mutable fields, so a fresh pull is temporally
invalid for a 2024-2025 window. The original scrape (0-hive-scraping-
everything.py) already captured `created` (immutable) and `json_metadata`
plus raw `body` as of January 2025, the closest historically valid snapshot.
This script reads that archive.

Caveat to carry into the thesis text: posts are editable, so metadata
reflects each post as of the scrape date, not necessarily the moment of
creation. The scrape sits at the window's end, so this is the best
available approximation.

Inputs:
  FULL_PARQUET (preferred)  the consolidated hive_full_posts_full.parquet
  CHUNK_DIR (fallback)      the hive_full_chunks folder of weekly parquets
  DOC_ID_FILTER             doc_id.xlsx, column A, the 5,565 sample

Output (in OUT_DIR):
  hive_post_features.parquet   one row per post:
    has_image, n_images        json_metadata 'image' list, plus body regex
                                fallback (![...](...) and <img>)
    has_link, n_links           http(s) occurrences in raw body, excluding
                                image URLs already counted
    n_tags                      json_metadata 'tags' length
    created_utc                 immutable creation timestamp
    post_hour                   0-23, UTC
    post_dow                    0=Monday ... 6=Sunday
    is_weekend                  Saturday or Sunday

Requirements: pandas, numpy, pyarrow, openpyxl (installed). Runs in ~1-3 min.
"""

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

# hive_cleaned.parquet derives from the same January 2025 scrape and
# carries body, created, and json_metadata (verified by find_archive.py),
# so it replaces the 2.1 GB consolidated archive as the input.
FULL_PARQUET = r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_hive_pipeline\hive sample preparation (politic-non-politic)\hive_cleaned.parquet"
CHUNK_DIR = r"C:\Users\batuh\PycharmProjects\Thesis\2- Step - Hive Data Scraping\hive_full_chunks"
DOC_ID_FILTER = r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_hive_pipeline\9. Sentence Emotion Scoring\doc_id.xlsx"
OUT_DIR = r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_hive_pipeline\12. Post Features"

NEEDED_COLS = ["author", "permlink", "created", "json_metadata", "body"]

IMG_MD_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
IMG_HTML_RE = re.compile(r"<img[^>]*>", re.IGNORECASE)
URL_RE = re.compile(r"https?://[^\s)\"'<>\]]+", re.IGNORECASE)


# ============================================================================
# LOADING
# ============================================================================

def load_archive():
    if FULL_PARQUET and os.path.exists(FULL_PARQUET):
        print(f"  Archive: {FULL_PARQUET}")
        return pd.read_parquet(FULL_PARQUET, columns=NEEDED_COLS)
    chunks = sorted(Path(CHUNK_DIR).glob("*.parquet"))
    if not chunks:
        raise FileNotFoundError(
            "Neither the consolidated parquet nor weekly chunks were found. "
            "Set FULL_PARQUET or CHUNK_DIR to the scrape archive.")
    print(f"  Archive: {len(chunks)} weekly chunks in {CHUNK_DIR}")
    parts = []
    for p in chunks:
        df = pd.read_parquet(p)
        keep = [c for c in NEEDED_COLS if c in df.columns]
        parts.append(df[keep])
    return pd.concat(parts, ignore_index=True)


def parse_meta(raw):
    """json_metadata is author-supplied and often malformed; parse
    defensively. Returns (n_tags, n_images_meta, links_meta_count)."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return 0, 0, 0
    try:
        meta = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(meta, dict):
            return 0, 0, 0
    except Exception:
        return 0, 0, 0
    tags = meta.get("tags", [])
    n_tags = len(tags) if isinstance(tags, list) else 0
    imgs = meta.get("image", [])
    n_img = len(imgs) if isinstance(imgs, list) else 0
    links = meta.get("links", [])
    n_lnk = len(links) if isinstance(links, list) else 0
    return n_tags, n_img, n_lnk


def main() -> None:
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hive_post_features.parquet"

    print("=" * 72)
    print("Descriptive post features from the January 2025 scrape archive")
    print("=" * 72)

    df = load_archive()
    print(f"  Archive rows: {len(df):,}")
    missing = [c for c in NEEDED_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Archive lacks columns {missing}; found "
                       f"{list(df.columns)[:20]} ...")

    # ---- filter to the regression sample ----
    ids = pd.read_excel(DOC_ID_FILTER, usecols=[0])
    keys = ids[ids.columns[0]].dropna().astype(str).str.strip()
    sp = keys.str.split("/", n=1, expand=True)
    want = set(zip(sp[0], sp[1]))
    df["_pair"] = list(zip(df["author"].astype(str), df["permlink"].astype(str)))
    before = len(df)
    df = df[df["_pair"].isin(want)].drop_duplicates("_pair").copy()
    print(f"  Sample filter: {before:,} -> {len(df):,} posts (expected 5,565)")
    if len(df) < len(want):
        print(f"  WARNING: {len(want) - len(df)} sample posts not found in "
              f"the archive; check that the archive covers the full window.")

    # ---- metadata features ----
    print("  Parsing json_metadata ...")
    meta = df["json_metadata"].apply(parse_meta)
    df["n_tags"] = [m[0] for m in meta]
    n_img_meta = np.array([m[1] for m in meta])
    n_lnk_meta = np.array([m[2] for m in meta])
    meta_ok = df["json_metadata"].notna() & (df["n_tags"] + n_img_meta +
                                             n_lnk_meta > 0)
    print(f"    posts with usable metadata: {meta_ok.sum():,} "
          f"({meta_ok.mean() * 100:.1f}%)")

    # ---- body regex features (fallback and complement) ----
    print("  Scanning raw body ...")
    body = df["body"].fillna("").astype(str)
    n_img_body = (body.str.count(IMG_MD_RE) + body.str.count(IMG_HTML_RE))
    all_urls = body.str.findall(URL_RE)
    img_urls_in_md = body.str.findall(
        re.compile(r"!\[[^\]]*\]\((https?://[^)]+)\)", re.IGNORECASE))
    n_urls_total = all_urls.apply(len).to_numpy()
    n_urls_img = img_urls_in_md.apply(len).to_numpy()

    df["n_images"] = np.maximum(n_img_meta, n_img_body.to_numpy())
    df["has_image"] = (df["n_images"] > 0).astype(int)
    df["n_links"] = np.maximum(n_urls_total - n_urls_img, 0)
    df["has_link"] = (df["n_links"] > 0).astype(int)

    # ---- posting time (immutable) ----
    created = pd.to_datetime(df["created"], errors="coerce", utc=True)
    df["created_utc"] = created
    df["post_hour"] = created.dt.hour
    df["post_dow"] = created.dt.dayofweek
    df["is_weekend"] = created.dt.dayofweek.isin([5, 6]).astype(int)

    df["doc_id"] = df["author"].astype(str) + "/" + df["permlink"].astype(str)
    out_cols = ["doc_id", "author", "permlink", "has_image", "n_images",
                "has_link", "n_links", "n_tags", "created_utc",
                "post_hour", "post_dow", "is_weekend"]
    df[out_cols].to_parquet(out_path, index=False)
    print(f"  Wrote {len(df):,} rows to {out_path}")

    # ---- diagnostics ----
    print("\n  Feature distributions:")
    print(f"    has_image: {df['has_image'].mean() * 100:.1f}%   "
          f"n_images p50 {df['n_images'].median():.0f}  "
          f"p90 {df['n_images'].quantile(0.90):.0f}  "
          f"max {df['n_images'].max():.0f}")
    print(f"    has_link:  {df['has_link'].mean() * 100:.1f}%   "
          f"n_links  p50 {df['n_links'].median():.0f}  "
          f"p90 {df['n_links'].quantile(0.90):.0f}  "
          f"max {df['n_links'].max():.0f}")
    print(f"    n_tags:    p50 {df['n_tags'].median():.0f}  "
          f"p90 {df['n_tags'].quantile(0.90):.0f}  "
          f"max {df['n_tags'].max():.0f}   "
          f"zero-tag posts {(df['n_tags'] == 0).sum():,}")
    print(f"    is_weekend: {df['is_weekend'].mean() * 100:.1f}%")
    hr = df["post_hour"].value_counts().sort_index()
    print(f"    busiest posting hours (UTC): "
          f"{', '.join(str(h) for h in hr.nlargest(3).index.tolist())}")
    print(f"    created range: {created.min()} to {created.max()}")

    print("\nDone. Send the console output back. Next: join to the "
          "regression data and decide how these enter the models "
          "(controls or a compact descriptive family).")


if __name__ == "__main__":
    main()
