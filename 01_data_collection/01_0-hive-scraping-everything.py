"""
Hive full-scope scraper for thesis Study 1 (no keyword filter).

Pulls ALL posts from Sep 1, 2024 - Jan 20, 2025 in weekly chunks.
After each week completes, the chunk is saved to disk as parquet so progress
is never lost. Excel exports are produced in 1M-row splits because Excel
cannot hold more than 1,048,576 rows per file.

Usage:
    1. Fill in HIVESQL_PASSWORD
    2. Run: python hive_full_scraper_weekly.py
    3. If interrupted, rerun - completed weeks are cached and skipped

Expected volume: 2-4 hundred thousand posts across ~20 weeks.
Expected runtime: 2-3 hours depending on HiveSQL load.
Expected disk usage: 2-4 GB parquet, multiple Excel files of ~250 MB each.
"""

import pyodbc
import pandas as pd
import re
import os
import time
from datetime import datetime, timedelta
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

HIVESQL_PASSWORD = os.getenv("HIVESQL_PASSWORD")

SCOPE_START = datetime(2024, 9, 1)
SCOPE_END = datetime(2025, 1, 21)
MIN_BODY_LENGTH = 100  # skip near-empty posts

CHECKPOINT_DIR = "hive_full_chunks"
EXCEL_OUT_PREFIX = "hive_full_posts"
EXCEL_ROW_LIMIT = 1_000_000  # Excel hard limit is 1,048,576

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# =============================================================================
# CONNECTION
# =============================================================================

def open_connection():
    conn = pyodbc.connect(
        "DRIVER={SQL Server};"
        "SERVER=vip.hivesql.io;"
        "DATABASE=DBHive;"
        "UID=Hive-batuhanayverdi01;"
        f"PWD={HIVESQL_PASSWORD};",
        timeout=30,
    )
    conn.timeout = 3600  # 1 hour per query - weekly chunks may be large
    return conn


# =============================================================================
# WEEK BOUNDARIES
# =============================================================================

def generate_weeks():
    weeks = []
    current = SCOPE_START
    i = 1
    while current < SCOPE_END:
        week_end = min(current + timedelta(days=7), SCOPE_END)
        label = f"week{i:02d}_{current.strftime('%Y-%m-%d')}_to_{week_end.strftime('%Y-%m-%d')}"
        weeks.append((i, label, current, week_end))
        current = week_end
        i += 1
    return weeks


# =============================================================================
# FETCH ONE WEEK
# =============================================================================

def fetch_week(conn, start: datetime, end: datetime) -> pd.DataFrame:
    """Pull all top-level posts within the date range."""
    q = f"""
    SELECT
        author, permlink, title, body, created,
        net_votes, children,
        total_payout_value, pending_payout_value,
        promoted, body_length, json_metadata
    FROM Comments
    WHERE
        created >= '{start.strftime('%Y-%m-%d %H:%M:%S')}'
        AND created < '{end.strftime('%Y-%m-%d %H:%M:%S')}'
        AND depth = 0
        AND LEN(body) > {MIN_BODY_LENGTH}
    """
    return pd.read_sql(q, conn)


# =============================================================================
# POSTPROCESS
# =============================================================================

def clean_body(text):
    if pd.isna(text):
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"#{1,6}\s", "", text)
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_payout(val):
    if pd.isna(val):
        return 0.0
    try:
        return float(str(val).split()[0])
    except (ValueError, IndexError):
        return 0.0


def postprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["body_clean"] = df["body"].apply(clean_body)
    df["link"] = "https://peakd.com/@" + df["author"] + "/" + df["permlink"]
    df["total_payout_num"] = df["total_payout_value"].apply(parse_payout)
    df["pending_payout_num"] = df["pending_payout_value"].apply(parse_payout)
    df["payout_combined"] = df["total_payout_num"] + df["pending_payout_num"]
    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    weeks = generate_weeks()
    print(f"Scope: {SCOPE_START.date()} to {SCOPE_END.date()}")
    print(f"Total weeks: {len(weeks)}")
    print(f"Checkpoint directory: {CHECKPOINT_DIR}/")
    print()

    # Check which weeks are already done (resume support)
    completed = []
    pending = []
    for week_idx, label, start, end in weeks:
        ckpt = os.path.join(CHECKPOINT_DIR, f"{label}.parquet")
        if os.path.exists(ckpt):
            completed.append((week_idx, label, start, end))
        else:
            pending.append((week_idx, label, start, end))

    if completed:
        print(f"Already cached weeks: {len(completed)}")
        for w in completed:
            ckpt = os.path.join(CHECKPOINT_DIR, f"{w[1]}.parquet")
            size_mb = os.path.getsize(ckpt) / (1024 * 1024)
            print(f"  [SKIP] {w[1]} ({size_mb:.1f} MB on disk)")
        print()

    if not pending:
        print("All weeks cached. Moving to export step.")
    else:
        print(f"Weeks to scrape: {len(pending)}\n")
        conn = open_connection()

        try:
            for week_idx, label, start, end in tqdm(pending, desc="Weeks"):
                ckpt = os.path.join(CHECKPOINT_DIR, f"{label}.parquet")
                tqdm.write(f"\n  [{label}] querying {start.date()} -> {end.date()} ...")

                t0 = time.time()
                try:
                    df_week = fetch_week(conn, start, end)
                except Exception as e:
                    tqdm.write(f"  [{label}] ERROR: {e}")
                    tqdm.write(f"  Reconnecting...")
                    try:
                        conn.close()
                    except Exception:
                        pass
                    conn = open_connection()
                    continue

                elapsed = time.time() - t0
                tqdm.write(f"  [{label}] fetched {len(df_week):,} posts in {elapsed:.0f}s")

                # Save immediately - parquet is compact and fast
                df_week.to_parquet(ckpt, index=False)
                size_mb = os.path.getsize(ckpt) / (1024 * 1024)
                tqdm.write(f"  [{label}] checkpoint saved: {size_mb:.1f} MB")

                # Brief pause to be gentle on the shared SQL server
                time.sleep(2)
        finally:
            try:
                conn.close()
            except Exception:
                pass

    # =========================================================================
    # EXPORT: merge all weekly parquets into Excel splits
    # =========================================================================

    print("\n" + "=" * 70)
    print("EXPORT: merging weekly chunks and writing Excel files")
    print("=" * 70)

    all_parquets = sorted(
        os.path.join(CHECKPOINT_DIR, f)
        for f in os.listdir(CHECKPOINT_DIR)
        if f.endswith(".parquet")
    )
    if not all_parquets:
        print("No parquet chunks found. Nothing to export.")
        return

    print(f"Loading {len(all_parquets)} weekly chunks...")
    dfs = []
    for p in all_parquets:
        dfs.append(pd.read_parquet(p))
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total raw rows: {len(df):,}")

    # Deduplicate (unlikely but safe)
    before = len(df)
    df = df.drop_duplicates(subset=["author", "permlink"])
    if len(df) < before:
        print(f"Deduplicated: {before:,} -> {len(df):,}")

    print("Post-processing (body cleaning, link building, payout parsing)...")
    df = postprocess(df)

    # Write Excel splits
    # Truncate body for Excel cell limit (32,767 chars)
    df["body_clean_excel"] = df["body_clean"].str.slice(0, 32000)

    export_cols = [
        "author", "permlink", "title", "body_clean_excel", "created",
        "net_votes", "children",
        "total_payout_num", "pending_payout_num", "payout_combined",
        "promoted", "body_length", "link",
    ]
    export_df = df[export_cols].rename(columns={"body_clean_excel": "body"})

    n_files = (len(export_df) + EXCEL_ROW_LIMIT - 1) // EXCEL_ROW_LIMIT
    print(f"\nWriting {n_files} Excel file(s), each up to {EXCEL_ROW_LIMIT:,} rows:")

    for i in range(n_files):
        start_i = i * EXCEL_ROW_LIMIT
        end_i = min(start_i + EXCEL_ROW_LIMIT, len(export_df))
        chunk = export_df.iloc[start_i:end_i]
        out_path = f"{EXCEL_OUT_PREFIX}_part{i+1:02d}.xlsx"
        chunk.to_excel(out_path, index=False)
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"  {out_path}: rows {start_i:,}-{end_i:,} ({size_mb:.1f} MB)")

    # Also save the full dataset as parquet (much faster to load later)
    full_parquet = f"{EXCEL_OUT_PREFIX}_full.parquet"
    df.to_parquet(full_parquet, index=False)
    size_mb = os.path.getsize(full_parquet) / (1024 * 1024)
    print(f"\nFull dataset parquet: {full_parquet} ({size_mb:.1f} MB)")

    # Summary
    print(f"\n=== Summary ===")
    print(f"Total posts:      {len(df):,}")
    print(f"Unique authors:   {df['author'].nunique():,}")
    print(f"Date range:       {df['created'].min()} to {df['created'].max()}")
    print(f"\n=== Engagement ===")
    print(f"  Votes   mean/med/max: {df['net_votes'].mean():.1f} / {df['net_votes'].median()} / {df['net_votes'].max()}")
    print(f"  Replies mean/med/max: {df['children'].mean():.1f} / {df['children'].median()} / {df['children'].max()}")
    print(f"  Payout  mean/med/max: ${df['payout_combined'].mean():.2f} / ${df['payout_combined'].median():.2f} / ${df['payout_combined'].max():.2f}")


if __name__ == "__main__":
    main()