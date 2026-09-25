"""
Spot check: VAD preprocessing and lexicon matching
===================================================
Verifies that the cleaning applied before NRC-VAD scoring preserves the
substance of Hive posts, and that the 48.5% coverage figure reflects
sensible matching rather than a broken pipeline.

It imports clean_and_lemmatize and score_text directly from
6-hive_vad_scores.py, so what is inspected here is the same code that
produced hive_vad_scores.parquet, not a copy that could drift.

Three diagnostics
-----------------
  1. SIDE BY SIDE: raw versus cleaned text for a purposive sample
     (random posts, the lowest-coverage posts, the fewest-matched posts,
     and the valence extremes), with word counts and scores.
  2. RETENTION: across a larger random subset, what share of words
     survives cleaning. Heavy loss would indicate the cleaner is
     stripping content rather than noise.
  3. UNMATCHED VOCABULARY: the most frequent cleaned tokens absent from
     the lexicon. If these are Hive and cryptocurrency jargon, the 48.5%
     coverage is expected. If they are ordinary English words, then
     lemmatization or matching is at fault.

Output:
  vad_spot_check.xlsx    full raw and cleaned text for manual reading

Run:  python 7-vad_spot_check.py
Deps: pandas, numpy, pyarrow, openpyxl, nltk, tqdm (already installed)
"""

import importlib.util
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE = r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_hive_pipeline"

SCORER_PATH = rf"{BASE}\17. NRC_VAD\6-hive_vad_scores.py"
INPUT_FILE = (
    rf"{BASE}\hive sample preparation (politic-non-politic)"
    r"\hive_sample_master.parquet"
)
VAD_SCORES = rf"{BASE}\17. NRC_VAD\hive_vad_scores.parquet"
OUT_DIR = rf"{BASE}\17. NRC_VAD"

N_RANDOM = 10          # random posts
N_EXTREME = 5          # per extreme category
N_RETENTION = 500      # posts used for the retention statistic
N_UNMATCHED = 40       # unmatched tokens to list
RANDOM_SEED = 20260101

RAW_EXCERPT = 400      # characters printed to console
CLEAN_EXCERPT = 400


# ============================================================================
# IMPORT THE SCORER MODULE BY PATH
# ============================================================================

def load_scorer(path: str):
    p = Path(path)
    if not p.exists():
        # try a couple of likely alternatives before giving up
        for alt in [
            Path(OUT_DIR) / "6-hive_vad_scores.py",
            Path.cwd() / "6-hive_vad_scores.py",
        ]:
            if alt.exists():
                p = alt
                break
        else:
            raise FileNotFoundError(
                f"Could not find the scoring script. Set SCORER_PATH; "
                f"tried {path}"
            )
    spec = importlib.util.spec_from_file_location("vad_scorer", p)
    module = importlib.util.module_from_spec(spec)
    sys.modules["vad_scorer"] = module
    spec.loader.exec_module(module)          # main() is guarded, so safe
    print(f"  Imported cleaning and scoring functions from: {p}")
    return module


# ============================================================================
# HELPERS
# ============================================================================

def ensure_doc_id(df: pd.DataFrame) -> pd.DataFrame:
    if "doc_id" in df.columns:
        return df
    if {"author", "permlink"} <= set(df.columns):
        df = df.copy()
        df["doc_id"] = df["author"].astype(str) + "/" + df["permlink"].astype(str)
        return df
    raise KeyError("Need doc_id, or author and permlink.")


def word_count(text) -> int:
    if not isinstance(text, str):
        return 0
    return len(re.findall(r"\b\w+\b", text))


def pick_sample(scores: pd.DataFrame) -> pd.DataFrame:
    """Random posts plus the cases most likely to expose a problem."""
    rng = np.random.default_rng(RANDOM_SEED)
    scored = scores[scores["valence"].notna()].copy()

    picks = []

    idx = rng.choice(scored.index, size=min(N_RANDOM, len(scored)),
                     replace=False)
    picks.append(scored.loc[idx].assign(pick_reason="random"))

    picks.append(
        scored.nsmallest(N_EXTREME, "vad_coverage")
        .assign(pick_reason="lowest coverage")
    )
    picks.append(
        scored.nsmallest(N_EXTREME, "n_matched")
        .assign(pick_reason="fewest matched tokens")
    )
    picks.append(
        scored.nsmallest(N_EXTREME, "valence")
        .assign(pick_reason="lowest valence")
    )
    picks.append(
        scored.nlargest(N_EXTREME, "valence")
        .assign(pick_reason="highest valence")
    )

    out = pd.concat(picks, ignore_index=True)
    return out.drop_duplicates("doc_id", keep="first")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("VAD preprocessing spot check")
    print("=" * 72)

    scorer = load_scorer(SCORER_PATH)
    scorer.ensure_nltk()
    vad_map = scorer.load_vad_lexicon(scorer.VAD_CSV)

    posts = ensure_doc_id(pd.read_parquet(INPUT_FILE))
    text_col = (
        scorer.TEXT_COLUMN
        if scorer.TEXT_COLUMN in posts.columns
        else next(c for c in scorer.TEXT_FALLBACKS if c in posts.columns)
    )
    scores = pd.read_parquet(VAD_SCORES)
    print(f"  Posts: {len(posts):,}   scored rows: {len(scores):,}   "
          f"text column: {text_col}")

    merged = scores.merge(
        posts[["doc_id", text_col]], on="doc_id", how="left"
    ).rename(columns={text_col: "raw_text"})

    # ---- 1. side by side ---------------------------------------------------
    sample = pick_sample(merged)
    print("\n" + "=" * 72)
    print(f"1. SIDE BY SIDE ({len(sample)} posts)")
    print("=" * 72)

    rows = []
    for r in sample.itertuples(index=False):
        cleaned = scorer.clean_and_lemmatize(r.raw_text)
        v, a, d, n_tok, n_match = scorer.score_text(cleaned, vad_map)
        raw_wc, clean_wc = word_count(r.raw_text), word_count(cleaned)
        retention = clean_wc / raw_wc if raw_wc else np.nan

        print("\n" + "-" * 72)
        print(f"[{r.pick_reason}] {r.doc_id}")
        print(f"  words {raw_wc} -> {clean_wc} "
              f"(retention {retention * 100:.0f}%)   "
              f"matched {n_match}/{n_tok} ({(n_match / n_tok * 100) if n_tok else 0:.0f}%)")
        print(f"  valence {v:.3f}   arousal {a:.3f}   dominance {d:.3f}")
        print(f"  RAW    : {str(r.raw_text)[:RAW_EXCERPT]}")
        print(f"  CLEANED: {cleaned[:CLEAN_EXCERPT]}")

        rows.append(
            {
                "doc_id": r.doc_id,
                "pick_reason": r.pick_reason,
                "raw_words": raw_wc,
                "cleaned_words": clean_wc,
                "retention": retention,
                "n_tokens": n_tok,
                "n_matched": n_match,
                "coverage": (n_match / n_tok) if n_tok else np.nan,
                "valence": v,
                "arousal": a,
                "dominance": d,
                "raw_text": r.raw_text,
                "cleaned_text": cleaned,
            }
        )

    # ---- 2. retention across a larger subset -------------------------------
    print("\n" + "=" * 72)
    print(f"2. RETENTION across {N_RETENTION} random posts")
    print("=" * 72)
    rng = np.random.default_rng(RANDOM_SEED)
    pool = merged[merged["raw_text"].notna()]
    idx = rng.choice(pool.index, size=min(N_RETENTION, len(pool)),
                     replace=False)
    ret, unmatched = [], Counter()
    for r in pool.loc[idx].itertuples(index=False):
        cleaned = scorer.clean_and_lemmatize(r.raw_text)
        raw_wc, clean_wc = word_count(r.raw_text), word_count(cleaned)
        if raw_wc:
            ret.append(clean_wc / raw_wc)
        toks = [re.sub(r"[^a-z]", "", t) for t in cleaned.lower().split()]
        unmatched.update(t for t in toks if t and t not in vad_map)

    ret = pd.Series(ret)
    print(f"  Word retention after cleaning: mean {ret.mean() * 100:.1f}%  "
          f"p10 {ret.quantile(0.10) * 100:.1f}%  "
          f"p50 {ret.quantile(0.50) * 100:.1f}%  "
          f"p90 {ret.quantile(0.90) * 100:.1f}%")
    print("  Interpretation: high retention means the cleaner removes "
          "formatting and non-ASCII characters rather than prose. "
          "Very low retention would indicate over-cleaning.")

    # ---- 3. unmatched vocabulary -------------------------------------------
    print("\n" + "=" * 72)
    print(f"3. MOST FREQUENT UNMATCHED TOKENS (top {N_UNMATCHED})")
    print("=" * 72)
    print("  Expected: function words, platform and crypto jargon, names.")
    print("  Concerning: ordinary English content words, which would "
          "suggest a lemmatization or matching fault.\n")
    top = unmatched.most_common(N_UNMATCHED)
    for i in range(0, len(top), 4):
        print("   " + "   ".join(
            f"{w:<16}{n:>6}" for w, n in top[i:i + 4]
        ))

    # ---- write -------------------------------------------------------------
    out_path = out_dir / "vad_spot_check.xlsx"
    detail = pd.DataFrame(rows)
    unmatched_df = pd.DataFrame(
        unmatched.most_common(500), columns=["token", "count"]
    )
    try:
        with pd.ExcelWriter(out_path) as writer:
            detail.to_excel(writer, sheet_name="side_by_side", index=False)
            unmatched_df.to_excel(writer, sheet_name="unmatched", index=False)
        print(f"\nWrote: {out_path}")
    except Exception as e:                             # pragma: no cover
        print(f"\n(Excel write skipped: {e})")

    print("\nDone. Read a few RAW versus CLEANED pairs yourself: the "
          "question is whether the cleaned text still carries the meaning "
          "of the original post.")


if __name__ == "__main__":
    main()
