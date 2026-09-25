"""
Spot-check Over-Cleaned Posts (v3 - current pipeline)
=====================================================

Same as your v2, with one change: the "before" text is read from `body_clean`
(the scraper's lightly-cleaned body that hive_cleaning.py cleans further), since
the current pipeline no longer produces `body_english_extracted`.

Pulls posts where cleaning removed the most content, so you can confirm the
cleaner is stripping boilerplate rather than real prose. Three slices are
written, with the high-engagement one mattering most for H1/H2.

Reads:  hive_cleaned.parquet
Run:    python spot_check_overcleaned.py
Deps:   pip install pandas pyarrow numpy
"""

import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# -- Configuration -----------------------------------------------------------

CONFIG = {
    "input_file":   r"hive_cleaned.parquet",
    "output_dir":   r"overcleaning_inspection",

    "before_column":   "body_clean",     # was body_english_extracted
    "before_fallback": "body",

    "n_top_pct":    20,
    "n_empty":      20,
    "n_engagement": 15,

    "empty_threshold_chars": 50,
    "min_chars_before":      500,
    "engagement_min_votes":  10,

    "print_chars": 3000,
}


def resolve_before_col(df: pd.DataFrame, cfg: Dict) -> str:
    if cfg["before_column"] in df.columns:
        return cfg["before_column"]
    if cfg["before_fallback"] in df.columns:
        return cfg["before_fallback"]
    raise KeyError(f"Neither '{cfg['before_column']}' nor '{cfg['before_fallback']}' "
                   f"in columns: {list(df.columns)}")


def fmt_post_block(row: pd.Series, before: str, after: str, cfg: Dict) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append(
        f"author={row.get('author', '?')}  "
        f"permlink={str(row.get('permlink', '?'))[:40]}  "
        f"status={row.get('lang_status', '?')}  "
        f"category={row.get('category', '?')}"
    )
    lines.append(
        f"len_before={len(before):,}  len_after={len(after):,}  "
        f"reduction={(len(before) - len(after)) / max(1, len(before)) * 100:.1f}%  "
        f"votes={int(row.get('net_votes', 0))}  "
        f"payout=${row.get('payout_combined', 0):.2f}"
    )
    title = row.get("title", "")
    if isinstance(title, str) and title:
        lines.append(f"title: {title[:120]}")
    lines.append("-" * 80)
    lines.append("BEFORE:")
    lines.append(before[:cfg["print_chars"]])
    if len(before) > cfg["print_chars"]:
        lines.append(f"... [{len(before) - cfg['print_chars']:,} more chars truncated]")
    lines.append("-" * 80)
    lines.append("AFTER:")
    if not after.strip():
        lines.append("[empty after cleaning]")
    else:
        lines.append(after[:cfg["print_chars"]])
        if len(after) > cfg["print_chars"]:
            lines.append(f"... [{len(after) - cfg['print_chars']:,} more chars truncated]")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    cfg = CONFIG
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {cfg['input_file']} ...")
    df = pd.read_parquet(cfg["input_file"])
    print(f"  {len(df):,} rows loaded")
    before_col = resolve_before_col(df, cfg)
    print(f"  Using before-column: '{before_col}'")

    sub = df[df["lang_status"].isin(["kept", "kept_bilingual"])].copy()
    sub["before_len"] = sub[before_col].fillna("").astype(str).str.len()
    sub["after_len"] = sub["body_for_analysis_len"].fillna(0).astype(int)
    sub = sub[sub["before_len"] > 0]
    sub["reduction_pct"] = (sub["before_len"] - sub["after_len"]) / sub["before_len"] * 100
    print(f"  Analyzable posts with non-empty input: {len(sub):,}")

    def write_slice(frame, before_c, fname):
        blocks = [fmt_post_block(r, str(r[before_c]), str(r["body_for_analysis"]), cfg)
                  for _, r in frame.iterrows()]
        (out_dir / fname).write_text("\n".join(blocks) if blocks else "[no posts matched]",
                                     encoding="utf-8")
        print(f"  Saved {len(blocks)} posts to {out_dir / fname}")

    print("\nSlice 1: top % reduction ...")
    write_slice(sub.nlargest(cfg["n_top_pct"], "reduction_pct"), before_col,
                "top_pct_reduction.txt")

    print("\nSlice 2: started long, became empty ...")
    became_empty = sub[(sub["before_len"] >= cfg["min_chars_before"])
                       & (sub["after_len"] < cfg["empty_threshold_chars"])].copy()
    became_empty["abs_drop"] = became_empty["before_len"] - became_empty["after_len"]
    write_slice(became_empty.nlargest(cfg["n_empty"], "abs_drop"), before_col,
                "became_empty.txt")

    print("\nSlice 3: high-engagement over-cleaned ...")
    threshold = sub["reduction_pct"].quantile(0.95)
    high_eng = sub[(sub["reduction_pct"] >= threshold)
                   & (sub["net_votes"] >= cfg["engagement_min_votes"])].copy()
    write_slice(high_eng.nlargest(cfg["n_engagement"], "reduction_pct"), before_col,
                "high_engagement_overcleaned.txt")

    summary = {
        "n_analyzable": int(len(sub)),
        "p95_reduction_pct": round(float(sub["reduction_pct"].quantile(0.95)), 2),
        "p99_reduction_pct": round(float(sub["reduction_pct"].quantile(0.99)), 2),
        "n_became_empty_from_500": int(((sub["before_len"] >= cfg["min_chars_before"])
                                        & (sub["after_len"] < cfg["empty_threshold_chars"])).sum()),
        "n_high_engagement_over": int(((sub["reduction_pct"] >= threshold)
                                       & (sub["net_votes"] >= cfg["engagement_min_votes"])).sum()),
    }
    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k:<30s}  {v}")
    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False)
    print(f"\nDone. Inspection files in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
