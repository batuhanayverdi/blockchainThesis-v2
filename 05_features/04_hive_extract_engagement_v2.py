"""
Hive engagement-variable extraction v2 (with tqdm progress)
===========================================================
Same purpose as before: richer reward / curation / diffusion fields for the SAME
6,242 posts and the SAME window, joined on (author, permlink). This version runs
the per-post queries in batches so tqdm shows progress, and the column names are
pinned to the schema this HiveSQL instance actually exposes.

Outputs (checkpointed; reruns skip finished steps):
  eng_comments.parquet, eng_votes.parquet, eng_accounts.parquet,
  eng_reblogs.parquet, hive_engagement_extended.parquet
"""

import os
import time
import warnings
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore", message=".*SQLAlchemy.*")

# ============================================================================
# CONFIGURATION
# ============================================================================

HIVESQL_PASSWORD = os.getenv("HIVESQL_PASSWORD")

POSTS_FILE = r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_hive_pipeline\6. regression\hive_regression_dataset.xlsx"
OUTPUT_DIR = r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_hive_pipeline\7. hive new dataset"

SCOPE_START = "2024-09-01 00:00:00"
REBLOG_END  = "2025-01-28 00:00:00"

COMMENTS_CHUNK = 1000
VOTES_CHUNK    = 500
DO_REBLOGS = True

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# CONNECTION + POSTS
# ============================================================================

def open_conn():
    conn = pyodbc.connect(
        "DRIVER={SQL Server};SERVER=vip.hivesql.io;DATABASE=DBHive;"
        f"UID=Hive-batuhanayverdi01;PWD={HIVESQL_PASSWORD};", timeout=30)
    conn.timeout = 3600
    return conn


def load_posts():
    if POSTS_FILE.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(POSTS_FILE)
    else:
        df = pd.read_parquet(POSTS_FILE)
    df.columns = [c.lower() for c in df.columns]
    if not {"author", "permlink"} <= set(df.columns):
        sp = df["doc_id"].astype(str).str.split("/", n=1, expand=True)
        df["author"], df["permlink"] = sp[0], sp[1]
    posts = df[["author", "permlink"]].dropna().drop_duplicates().reset_index(drop=True)
    print(f"Posts to enrich: {len(posts):,}")
    return posts


def make_temp(conn, posts):
    cur = conn.cursor()
    cur.execute("IF OBJECT_ID('tempdb..#posts') IS NOT NULL DROP TABLE #posts;")
    cur.execute("CREATE TABLE #posts (rn INT, author NVARCHAR(20), permlink NVARCHAR(300));")
    cur.fast_executemany = True
    rows = [(i + 1, a, p) for i, (a, p) in enumerate(posts.itertuples(index=False, name=None))]
    cur.executemany("INSERT INTO #posts (rn, author, permlink) VALUES (?, ?, ?)", rows)
    cur.execute("CREATE INDEX ix_ap ON #posts(author, permlink);")
    cur.execute("CREATE INDEX ix_rn ON #posts(rn);")
    conn.commit()
    n = pd.read_sql("SELECT COUNT(*) n FROM #posts", conn)["n"][0]
    print(f"#posts temp table loaded: {n:,} rows\n")
    return int(n)


def batched(conn, total, build_sql, chunk, desc):
    """Run build_sql(rn_lo, rn_hi) over rn ranges with a tqdm bar; concat."""
    parts = []
    n_batches = (total + chunk - 1) // chunk
    for k in tqdm(range(n_batches), desc=desc, unit="batch"):
        a, b = k * chunk + 1, (k + 1) * chunk
        parts.append(pd.read_sql(build_sql(a, b), conn))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ============================================================================
# STEPS
# ============================================================================

def step_comments(conn, total):
    out = os.path.join(OUTPUT_DIR, "eng_comments.parquet")
    if os.path.exists(out):
        print("eng_comments.parquet exists, skipping"); return pd.read_parquet(out)
    cols = ["author", "permlink", "net_votes", "children", "net_rshares",
            "abs_rshares", "vote_rshares", "total_vote_weight", "total_payout_value",
            "curator_payout_value", "pending_payout_value", "author_rewards",
            "promoted", "percent_hbd", "created"]
    sel = ", ".join(f"c.[{c}]" for c in cols)
    build = lambda a, b: (
        f"SELECT {sel} FROM Comments c INNER JOIN #posts p "
        f"ON p.author = c.author AND p.permlink = c.permlink "
        f"WHERE c.depth = 0 AND p.rn BETWEEN {a} AND {b}")
    df = batched(conn, total, build, COMMENTS_CHUNK, "Comments")
    df.to_parquet(out, index=False)
    print(f"  -> {len(df):,} rows")
    return df


def step_votes(conn, total):
    out = os.path.join(OUTPUT_DIR, "eng_votes.parquet")
    if os.path.exists(out):
        print("eng_votes.parquet exists, skipping"); return pd.read_parquet(out)
    build = lambda a, b: (
        "SELECT v.author, v.permlink, COUNT(*) AS n_votes_raw, "
        "SUM(CASE WHEN v.weight > 0 THEN 1 ELSE 0 END) AS upvotes, "
        "SUM(CASE WHEN v.weight < 0 THEN 1 ELSE 0 END) AS downvotes, "
        "COUNT(DISTINCT v.voter) AS distinct_voters "
        "FROM TxVotes v INNER JOIN #posts p "
        "ON p.author = v.author AND p.permlink = v.permlink "
        f"WHERE p.rn BETWEEN {a} AND {b} "
        "GROUP BY v.author, v.permlink")
    df = batched(conn, total, build, VOTES_CHUNK, "Votes")
    df.to_parquet(out, index=False)
    print(f"  -> {len(df):,} posts with vote rows")
    return df


def step_accounts(conn):
    out = os.path.join(OUTPUT_DIR, "eng_accounts.parquet")
    if os.path.exists(out):
        print("eng_accounts.parquet exists, skipping"); return pd.read_parquet(out)
    cols = ["reputation", "reputation_ui", "vesting_shares",
            "received_vesting_shares", "delegated_vesting_shares", "post_count"]
    sel = ", ".join(f"a.[{c}]" for c in cols)
    q = (f"SELECT a.name AS author, {sel} FROM Accounts a "
         f"INNER JOIN (SELECT DISTINCT author FROM #posts) p ON p.author = a.name")
    t0 = time.time()
    print("Author standing (Accounts) ...", end="", flush=True)
    df = pd.read_sql(q, conn)
    print(f" {len(df):,} authors in {time.time()-t0:.0f}s")
    df.to_parquet(out, index=False)
    return df


def step_reblogs(conn):
    out = os.path.join(OUTPUT_DIR, "eng_reblogs.parquet")
    if os.path.exists(out):
        print("eng_reblogs.parquet exists, skipping"); return pd.read_parquet(out)
    # weekly windows so the scan shows progress
    cur = datetime.strptime(SCOPE_START, "%Y-%m-%d %H:%M:%S")
    end = datetime.strptime(REBLOG_END, "%Y-%m-%d %H:%M:%S")
    wins = []
    while cur < end:
        nxt = min(cur + timedelta(days=7), end)
        wins.append((cur.strftime("%Y-%m-%d %H:%M:%S"), nxt.strftime("%Y-%m-%d %H:%M:%S")))
        cur = nxt
    parts = []
    for ws, we in tqdm(wins, desc="Reblogs (weekly)", unit="wk"):
        q = ("SELECT r.author, r.permlink, COUNT(*) AS reblogs FROM ("
             "  SELECT COALESCE(JSON_VALUE(t.json,'$.author'), JSON_VALUE(t.json,'$[1].author')) AS author, "
             "         COALESCE(JSON_VALUE(t.json,'$.permlink'), JSON_VALUE(t.json,'$[1].permlink')) AS permlink "
             f"  FROM TxCustoms t WHERE t.tid = 'reblog' AND t.timestamp >= '{ws}' AND t.timestamp < '{we}'"
             ") r INNER JOIN #posts p ON p.author = r.author AND p.permlink = r.permlink "
             "GROUP BY r.author, r.permlink")
        try:
            parts.append(pd.read_sql(q, conn))
        except Exception as e:
            print(f"\n  Reblog query failed ({e}). Verify TxCustoms JSON support; "
                  f"the rest is unaffected.")
            return None
    if not parts:
        return None
    df = (pd.concat(parts, ignore_index=True)
          .groupby(["author", "permlink"], as_index=False)["reblogs"].sum())
    df.to_parquet(out, index=False)
    print(f"  -> {len(df):,} posts had >=1 reblog")
    return df


# ============================================================================
# MAIN
# ============================================================================

def main():
    posts = load_posts()
    conn = open_conn()
    try:
        total = make_temp(conn, posts)
        comments = step_comments(conn, total)
        votes    = step_votes(conn, total)
        accounts = step_accounts(conn)
        reblogs  = step_reblogs(conn) if DO_REBLOGS else None
    finally:
        try: conn.close()
        except Exception: pass

    merged = posts.copy()
    for name, df, on in [("comments", comments, ["author", "permlink"]),
                         ("votes", votes, ["author", "permlink"]),
                         ("reblogs", reblogs, ["author", "permlink"]),
                         ("accounts", accounts, ["author"])]:
        if df is not None and len(df):
            merged = merged.merge(df, on=on, how="left", suffixes=("", f"_{name}"))
    for c in ("reblogs", "upvotes", "downvotes", "distinct_voters", "n_votes_raw"):
        if c in merged.columns:
            merged[c] = merged[c].fillna(0)

    out = os.path.join(OUTPUT_DIR, "hive_engagement_extended.parquet")
    merged.to_parquet(out, index=False)

    print("\n" + "=" * 64 + "\nVERIFICATION\n" + "=" * 64)
    print(f"Posts in sample: {len(posts):,}   Rows in output: {len(merged):,}")
    print("\nCoverage (non-null) per new field:")
    for c in [c for c in merged.columns if c not in ("author", "permlink")]:
        nn = merged[c].notna().sum()
        print(f"  {c:<26s} {nn:>6,} / {len(merged):,}  ({100*nn/len(merged):.1f}%)")
    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()
