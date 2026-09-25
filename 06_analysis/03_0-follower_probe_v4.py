"""
Follower probe v4: one scan for ALL authors (server-side extraction)
====================================================================
Insight this tests
------------------
Strategy A failed because it scanned TxCustoms once PER AUTHOR (1,650
scans). This probe tests scanning once PER MONTH for ALL authors at the
same time: SQL Server extracts the 'following' value from each payload
during the scan (CHARINDEX + SUBSTRING) and compares it against the full
author list with IN. If a month costs roughly what one author-month LIKE
cost in v2 (~25s), the full history is ~1-4 hours of checkpointed
queries instead of ~1,171 hours.

What it measures
----------------
  1. One recent month (2024-08), all 1,650 authors, timed. Sanity check:
     the tdvtv subset should match what the v2 LIKE query found.
  2. One heavy bot-era month (2017-06, ~4.1M ops), timed. This is the
     worst case that decides whether early history is affordable.
  3. Whether #temp tables are permitted (fallback join mechanism if the
     large IN list performs badly).
  4. A weighted full-history extrapolation and a clear verdict.

Notes
-----
  - tid='follow' also carries reblog payloads with no 'following' key;
    the CHARINDEX guard drops them inside the scan.
  - Payloads with a JSON list in 'following' or unusual spacing are not
    matched by the extraction; the probe reports how many follow ops in
    the month carry no extractable name so this residual is quantified.

Run:  python 0-follower_probe_v4.py
Deps: pip install pyodbc pandas pyarrow
"""

import json
import os
import queue
import re
import sys
import threading
import time

import pandas as pd
import pyodbc

# ============================================================================
# CONFIGURATION
# ============================================================================

HIVESQL_SERVER = os.environ.get("HIVESQL_SERVER", "vip.hivesql.io")
HIVESQL_DATABASE = os.environ.get("HIVESQL_DATABASE", "DBHive")
HIVESQL_USER = os.environ.get("HIVESQL_USER", "Hive-batuhanayverdi01")
HIVESQL_PASSWORD = os.environ.get("HIVESQL_PASSWORD", "bsR9mnbF5Zyzxk4PjrB3")

REGRESSION_PARQUET = (
    r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_hive_pipeline"
    r"\6. regression\hive_regression_dataset_main.parquet"
)

RECENT_MONTH = ("2024-08-01", "2024-09-01")
HEAVY_MONTH = ("2017-06-01", "2017-07-01")
SANITY_AUTHOR = "tdvtv"          # v2 found 1 matching op in 2024-08

BUDGET_RECENT = 300              # seconds
BUDGET_HEAVY = 900               # seconds; 2017-06 has ~4.1M follow ops
HEARTBEAT_SECONDS = 15

# Rough monthly volume profile from v2 for the extrapolation: months are
# grouped into eras with approximate follow-op volumes relative to 2024-08.
# (2017-06: 4.1M, 2021-01: 147K, 2024-08: 109K.)
ERA_MONTHS = [
    ("2016-2018 bot era", 30),
    ("2019-2024 normal era", 70),
]

ACCOUNT_RE = re.compile(r"^[a-z0-9.\-]{3,16}$")


def say(msg=""):
    print(msg, flush=True)


def connect(query_timeout):
    if not HIVESQL_PASSWORD:
        say("ERROR: set the HIVESQL_PASSWORD environment variable.")
        sys.exit(1)
    conn = pyodbc.connect(
        "DRIVER={SQL Server};"
        f"SERVER={HIVESQL_SERVER};DATABASE={HIVESQL_DATABASE};"
        f"UID={HIVESQL_USER};PWD={HIVESQL_PASSWORD};",
        timeout=30,
    )
    conn.timeout = query_timeout
    return conn


def timed_query(label, sql, budget_seconds):
    """Heartbeat wrapper from v2: fresh connection per query, worker
    thread, elapsed prints every 15s, hard budget."""
    result_q = queue.Queue()

    def worker():
        conn = None
        try:
            conn = connect(budget_seconds)
            cur = conn.cursor()
            t0 = time.time()
            cur.execute(sql)
            rows = [tuple(r) for r in cur.fetchall()]
            result_q.put(("ok", time.time() - t0, rows))
            cur.close()
        except Exception as e:                       # noqa: BLE001
            result_q.put(("err", None, str(e)))
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:                    # noqa: BLE001
                    pass

    say(f"  [{label}] started (budget {budget_seconds}s)")
    t_start = time.time()
    threading.Thread(target=worker, daemon=True).start()
    while True:
        try:
            status, elapsed, payload = result_q.get(
                timeout=HEARTBEAT_SECONDS)
            break
        except queue.Empty:
            waited = time.time() - t_start
            if waited > budget_seconds + 30:
                say(f"  [{label}] OVER BUDGET after {waited:.0f}s; "
                    f"abandoning.")
                return None, None
            say(f"  [{label}] ... still running ({waited:.0f}s)")
    if status == "err":
        say(f"  [{label}] FAILED: {str(payload)[:250]}")
        return None, None
    say(f"  [{label}] finished in {elapsed:.1f}s ({len(payload):,} rows)")
    return elapsed, payload


def load_analysis_authors():
    df = pd.read_parquet(
        REGRESSION_PARQUET,
        columns=["author", "analysis_sample_body"],
    )
    mask = df["analysis_sample_body"].fillna(False).astype(bool)
    authors = sorted(
        df.loc[mask, "author"].dropna().astype(str).str.strip().unique()
    )
    authors = [a for a in authors if ACCOUNT_RE.match(a)]
    say(f"  Analysis-sample authors: {len(authors)} (expected 1,650)")
    return authors


# ============================================================================
# QUERY BUILDER: one scan, all authors
# ============================================================================

def single_scan_query(authors, start, end):
    in_list = ",".join(f"'{a}'" for a in authors)
    # s = position of the '"following":"' marker; +13 lands on the first
    # character of the name; e = position of the closing quote.
    return f"""
SELECT t.timestamp, t.json,
       SUBSTRING(t.json, x.s + 13, y.e - (x.s + 13)) AS following_name
FROM TxCustoms t
CROSS APPLY (SELECT CHARINDEX('"following":"', t.json) AS s) x
CROSS APPLY (SELECT CASE WHEN x.s > 0
                         THEN CHARINDEX('"', t.json, x.s + 13)
                         ELSE 0 END AS e) y
WHERE t.tid = 'follow'
  AND t.timestamp >= '{start}' AND t.timestamp < '{end}'
  AND x.s > 0
  AND y.e > x.s + 13
  AND SUBSTRING(t.json, x.s + 13, y.e - (x.s + 13)) IN ({in_list})
"""


def residual_count_query(start, end):
    """Follow ops in the month with NO extractable 'following' marker
    (reblogs plus unusual payloads); quantifies what the extraction
    cannot see."""
    return (
        "SELECT COUNT(*) FROM TxCustoms "
        f"WHERE tid = 'follow' AND timestamp >= '{start}' "
        f"AND timestamp < '{end}' "
        "AND CHARINDEX('\"following\":\"', json) = 0"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():
    say("Loading analysis-sample authors ...")
    authors = load_analysis_authors()

    # ---- 1. recent month, all authors ------------------------------------
    say("\n" + "=" * 70)
    say("TEST 1: one scan, all authors, recent month "
        f"({RECENT_MONTH[0][:7]})")
    say("=" * 70)
    t_recent, rows_recent = timed_query(
        "scan 2024-08 all-authors",
        single_scan_query(authors, *RECENT_MONTH),
        BUDGET_RECENT,
    )
    if rows_recent is not None:
        n_sanity = sum(1 for r in rows_recent if r[2] == SANITY_AUTHOR)
        say(f"  Sanity: ops targeting {SANITY_AUTHOR}: {n_sanity} "
            f"(v2 LIKE found 1)")
        parsed_ok = 0
        for _, payload, _ in rows_recent[:200]:
            try:
                obj = json.loads(payload)
                if isinstance(obj, list) and len(obj) == 2:
                    obj = obj[1]
                if isinstance(obj, dict) and obj.get("follower"):
                    parsed_ok += 1
            except Exception:                        # noqa: BLE001
                pass
        say(f"  Parse check: {parsed_ok}/{min(len(rows_recent), 200)} "
            f"returned rows parse as follow ops")

    # residual: ops the extraction cannot see (mostly reblogs)
    _, res = timed_query(
        "residual count 2024-08",
        residual_count_query(*RECENT_MONTH),
        120,
    )
    if res:
        say(f"  Ops without a 'following' marker in the month: "
            f"{res[0][0]:,} (reblogs and unusual payloads)")

    # ---- 2. heavy bot-era month ------------------------------------------
    say("\n" + "=" * 70)
    say(f"TEST 2: heavy month ({HEAVY_MONTH[0][:7]}, ~4.1M follow ops)")
    say("=" * 70)
    t_heavy, rows_heavy = timed_query(
        "scan 2017-06 all-authors",
        single_scan_query(authors, *HEAVY_MONTH),
        BUDGET_HEAVY,
    )

    # ---- 3. temp table availability (fallback mechanism) ------------------
    say("\n" + "=" * 70)
    say("TEST 3: are #temp tables permitted? (fallback join mechanism)")
    say("=" * 70)
    try:
        conn = connect(60)
        cur = conn.cursor()
        cur.execute("SELECT 1 AS x INTO #probe; SELECT COUNT(*) FROM #probe;")
        # advance to the second result set
        while cur.description is None:
            if not cur.nextset():
                break
        say("  #temp tables: PERMITTED")
        cur.close()
        conn.close()
    except Exception as e:                           # noqa: BLE001
        say(f"  #temp tables: NOT permitted ({str(e)[:120]})")

    # ---- 4. verdict --------------------------------------------------------
    say("\n" + "=" * 70)
    say("VERDICT")
    say("=" * 70)
    if t_recent is None:
        say("  Recent-month scan failed or exceeded budget: the "
            "single-scan approach is not viable either. The reward-based "
            "measure stands, now with three documented failed routes.")
        return

    heavy = t_heavy if t_heavy is not None else BUDGET_HEAVY
    normal_h = t_recent * ERA_MONTHS[1][1] / 3600
    bot_h = heavy * ERA_MONTHS[0][1] / 3600
    total_h = normal_h + bot_h
    say(f"  Recent month: {t_recent:.0f}s x {ERA_MONTHS[1][1]} months "
        f"= {normal_h:.1f} h")
    say(f"  Bot-era month: {heavy:.0f}s"
        f"{' (budget cap; real cost unknown)' if t_heavy is None else ''}"
        f" x {ERA_MONTHS[0][1]} months = {bot_h:.1f} h")
    say(f"  Estimated full history: ~{total_h:.1f} h of checkpointed "
        f"monthly queries")
    say("")
    if total_h <= 4:
        say("  VIABLE. Next step: full reconstruction script with "
            "per-month parquet checkpoints, client-side latest-state "
            "resolution, and a pre_followers column for the builder.")
    elif total_h <= 10:
        say("  BORDERLINE. Possible but a long babysat run on a shared "
            "server; decide whether the upgrade justifies the time.")
    else:
        say("  NOT viable. The reward-based measure stands.")


if __name__ == "__main__":
    main()
