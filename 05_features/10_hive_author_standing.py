"""
Hive author standing, strictly pre-window  (HiveSQL, pyodbc)
============================================================
Builds historically valid author-standing measures for the 1,716 sample
authors, using only blockchain operations dated BEFORE the observation
window (2024-09-01). Replaces the temporally invalid state-table
snapshots (reputation_ui, effective_vesting) in the R analysis.

Why this is valid where the snapshot was not: HiveSQL operation tables
(Tx*) and virtual-operation tables (VO*) are append-only histories with
timestamps. Summing them with `timestamp < @window_start` yields the
exact cumulative value as of that date. State tables hold CURRENT
values and are used here only for the immutable `created` field.

Reputation is deliberately NOT reconstructed: it is a recursive
function of all incoming vote rshares weighted by voter reputation at
vote time, with no historical snapshots available. Cumulative author
rewards serve as the standing proxy instead.

Outputs (in OUT_DIR):
  hive_author_standing.parquet   one row per author:
    author
    acct_created_utc          immutable account creation timestamp
    acct_age_days             age in days at 2024-09-01 (>= 0)
    pre_author_rew_hbd        cumulative HBD author rewards  (< window)
    pre_author_rew_hive       cumulative HIVE author rewards (< window)
    pre_author_rew_vests      cumulative vesting author rewards (< window)
    pre_author_rew_events     number of rewarded posts/comments (< window)
    pre_curation_vests        cumulative curation rewards in vests (< window)
    pre_root_posts            distinct root posts authored (< window)

Requirements: pandas, numpy, pyarrow, openpyxl, pyodbc (all installed).
Credentials: free HiveSQL registration; the login carries the "Hive-"
prefix. Set HIVESQL_PASSWORD below or as an environment variable.

Run time: a few minutes; each component checkpoints to its own parquet
and is skipped on re-run if present, so re-running after a timeout is
safe and cheap.
"""

import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyodbc


# ============================================================================
# CONFIGURATION
# ============================================================================

DOC_ID_FILTER = r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_hive_pipeline\9. Sentence Emotion Scoring\doc_id.xlsx"
OUT_DIR       = r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_hive_pipeline\12. Post Features"

WINDOW_START = "2024-09-01"          # observation window start (UTC)

HIVESQL_SERVER   = os.environ.get("HIVESQL_SERVER",   "vip.hivesql.io")
HIVESQL_DATABASE = os.environ.get("HIVESQL_DATABASE", "DBHive")
HIVESQL_USER     = os.environ.get("HIVESQL_USER",     "Hive-batuhanayverdi01")
HIVESQL_PASSWORD = os.environ.get("HIVESQL_PASSWORD", "bsR9mnbF5Zyzxk4PjrB3")

BATCH_SIZE = 300                     # accounts per IN() list
RUN_STAGE2 = False                   # vesting reconstruction; see bottom

ACCOUNT_RE = re.compile(r"^[a-z0-9.\-]{3,16}$")   # Hive account name rules


# ============================================================================
# HELPERS
# ============================================================================

def connect():
    conn = pyodbc.connect(
        "DRIVER={SQL Server};"
        f"SERVER={HIVESQL_SERVER};DATABASE={HIVESQL_DATABASE};"
        f"UID={HIVESQL_USER};PWD={HIVESQL_PASSWORD};", timeout=30)
    conn.timeout = 3600   # per-query timeout in seconds
    return conn


def load_authors() -> list:
    ids = pd.read_excel(DOC_ID_FILTER, usecols=[0])
    keys = ids[ids.columns[0]].dropna().astype(str).str.strip()
    authors = sorted({k.split("/", 1)[0] for k in keys if "/" in k})
    bad = [a for a in authors if not ACCOUNT_RE.match(a)]
    if bad:
        print(f"  WARNING: {len(bad)} names fail the account-name pattern "
              f"and are dropped: {bad[:5]} ...")
        authors = [a for a in authors if ACCOUNT_RE.match(a)]
    print(f"  Authors: {len(authors)} (expected ~1,716)")
    return authors


def in_list(batch) -> str:
    # Names validated by ACCOUNT_RE above, so direct quoting is safe.
    return ",".join(f"'{a}'" for a in batch)


def batched_query(conn, sql_template: str, authors: list,
                  colnames: list) -> pd.DataFrame:
    """Run sql_template (with {IN} placeholder) over author batches."""
    parts = []
    n_batches = (len(authors) - 1) // BATCH_SIZE + 1
    for i in range(0, len(authors), BATCH_SIZE):
        batch = authors[i:i + BATCH_SIZE]
        sql = sql_template.format(IN=in_list(batch), WS=WINDOW_START)
        cur = conn.cursor()
        cur.execute(sql)
        rows = [tuple(r) for r in cur.fetchall()]   # pyodbc Row -> tuple
        cur.close()
        parts.append(pd.DataFrame(rows, columns=colnames))
        print(f"    batch {i // BATCH_SIZE + 1}/{n_batches}: {len(rows)} rows")
    return pd.concat(parts, ignore_index=True) if parts else \
        pd.DataFrame(columns=colnames)


def probe_columns(conn, table: str) -> None:
    """Print the actual schema; verifies column names before querying."""
    cur = conn.cursor()
    cur.execute(
        "SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS "
        f"WHERE TABLE_NAME = '{table}' ORDER BY ORDINAL_POSITION")
    cols = [tuple(r) for r in cur.fetchall()]
    cur.close()
    print(f"  [{table}] " + ", ".join(f"{c}({t})" for c, t in cols))


def checkpoint(out_dir: Path, name: str):
    """Load the component if already saved, else compute and save."""
    path = out_dir / f"_ckpt_{name}.parquet"

    def run(fn):
        if path.exists():
            print(f"  [{name}] checkpoint found, skipping query.")
            return pd.read_parquet(path)
        df = fn()
        df.to_parquet(path, index=False)
        print(f"  [{name}] saved checkpoint ({len(df)} rows).")
        return df
    return run


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hive_author_standing.parquet"

    print("=" * 72)
    print("Strictly pre-window author standing from HiveSQL")
    print(f"Window start (exclusive upper bound for history): {WINDOW_START}")
    print("=" * 72)

    authors = load_authors()
    conn = connect()
    print("  Connected to HiveSQL.")

    # ---- 0. schema probe (verify column names once; harmless to keep) ----
    print("\nSchema probe:")
    for t in ["Accounts", "VOAuthorRewards", "VOCurationRewards", "TxComments"]:
        probe_columns(conn, t)

    # ---- 1. account creation (immutable, safe from the state table) ----
    print("\n[1/4] Account creation dates")
    acct = checkpoint(out_dir, "accounts")(lambda: batched_query(
        conn,
        "SELECT name, created FROM Accounts WHERE name IN ({IN})",
        authors, ["author", "acct_created_utc"]))

    # ---- 2. pre-window author rewards ----
    print("\n[2/4] Cumulative author rewards before the window")
    arw = checkpoint(out_dir, "author_rewards")(lambda: batched_query(
        conn,
        "SELECT author, "
        "       SUM(hbd_payout)     AS hbd, "
        "       SUM(hive_payout)    AS hive, "
        "       SUM(vesting_payout) AS vests, "
        "       COUNT(*)            AS events "
        "FROM VOAuthorRewards "
        "WHERE author IN ({IN}) AND timestamp < '{WS}' "
        "GROUP BY author",
        authors, ["author", "pre_author_rew_hbd", "pre_author_rew_hive",
                  "pre_author_rew_vests", "pre_author_rew_events"]))

    # ---- 3. pre-window curation rewards (stake-activity proxy) ----
    print("\n[3/4] Cumulative curation rewards before the window")
    crw = checkpoint(out_dir, "curation_rewards")(lambda: batched_query(
        conn,
        "SELECT curator, SUM(reward) AS vests "
        "FROM VOCurationRewards "
        "WHERE curator IN ({IN}) AND timestamp < '{WS}' "
        "GROUP BY curator",
        authors, ["author", "pre_curation_vests"]))

    # ---- 4. pre-window root-post count (dedup: edits repeat rows) ----
    print("\n[4/4] Distinct root posts authored before the window")
    pst = checkpoint(out_dir, "root_posts")(lambda: batched_query(
        conn,
        "SELECT author, COUNT(DISTINCT permlink) AS n "
        "FROM TxComments "
        "WHERE author IN ({IN}) AND parent_author = '' "
        "      AND timestamp < '{WS}' "
        "GROUP BY author",
        authors, ["author", "pre_root_posts"]))

    conn.close()

    # ---- assemble ----
    df = pd.DataFrame({"author": authors})
    df = (df.merge(acct, on="author", how="left")
            .merge(arw,  on="author", how="left")
            .merge(crw,  on="author", how="left")
            .merge(pst,  on="author", how="left"))
    fill0 = ["pre_author_rew_hbd", "pre_author_rew_hive",
             "pre_author_rew_vests", "pre_author_rew_events",
             "pre_curation_vests", "pre_root_posts"]
    df[fill0] = df[fill0].apply(pd.to_numeric, errors="coerce").fillna(0)

    ws = datetime.fromisoformat(WINDOW_START).replace(tzinfo=timezone.utc)
    created = pd.to_datetime(df["acct_created_utc"], errors="coerce", utc=True)
    df["acct_age_days"] = ((ws - created).dt.days).clip(lower=0)

    df.to_parquet(out_path, index=False)
    print(f"\nWrote {len(df)} rows to {out_path}")

    # ---- diagnostics ----
    print("\nDiagnostics:")
    print(f"  accounts matched in Accounts table: {created.notna().sum()} "
          f"of {len(df)}")
    print(f"  accounts created after window start (zeros expected): "
          f"{(created > ws).sum()}")
    for c in ["pre_author_rew_vests", "pre_curation_vests",
              "pre_root_posts", "acct_age_days"]:
        q = df[c].quantile([0.5, 0.9]).round(1).tolist()
        print(f"  {c:<24} p50 {q[0]:>12}  p90 {q[1]:>12}  "
              f"zero-share {(df[c] == 0).mean() * 100:.1f}%")

    print("\nDone. Send the console output back (including the schema "
          "probe). Next: join to the regression data and swap the "
          "standing variables in sections 7.3 and 7.4 of the qmd.")


# ============================================================================
# STAGE 2 (OPTIONAL, OFF BY DEFAULT): effective vesting at window start
# ============================================================================
# Backward reconstruction:
#   own_vests(T) = own_vests(now)
#                  - claimed reward vests            (T..now)  TxClaimRewardBalances
#                  - power-up vests received         (T..now)  VOTransferToVestingCompleteds
#                  + power-down vests withdrawn      (T..now)  VOFillVestingWithdraws
#   delegated/received at T: for each (delegator, delegatee) pair, the
#   last TxDelegateVestingShares operation before T states the absolute
#   delegation; sum by direction.
# This is implementable but has more failure modes (column-name drift,
# rare flows such as escrow releases in vests). Enable RUN_STAGE2 only
# after the schema probe confirms the column names of these tables, and
# treat the result as a robustness variant of the pre-window standing
# measures above, not as the primary variable.
# ============================================================================


if __name__ == "__main__":
    main()
