"""
Hive Language Filter (90% English, text unaltered)
==================================================

Purpose
-------
Keep only posts whose body is at least 90% English, and DROP the rest.
Unlike the earlier version, this script does NOT extract or modify the post
text. A kept post keeps its original body in full, so the text remains
suitable for sentiment and emotional-language analysis.

How the English share is measured
---------------------------------
For each post, the body is split into sentences. Each sentence (after light
preprocessing that removes URLs, code, hashtags, crypto tickers, and wallet
or transaction hashes) is language-identified with fastText. The English
share is the number of characters in English sentences divided by the total
number of characters in detectable sentences. A post is kept when that share
is >= ENGLISH_SHARE_THRESHOLD.

Columns added (the body is left untouched)
------------------------------------------
  pct_english    English share by character count, in [0, 1]
  dominant_lang  most frequent language by characters (diagnostic)
  lang_status    one of:
      'kept'                  >= threshold English (retained)
      'dropped_low_english'   below threshold (removed)
      'too_short'             too little detectable text (removed)
      'empty'                 body missing or empty (removed)

Output
------
Only the kept posts are written to OUTPUT_FILE, with the original columns
(body unchanged) plus the three columns above. The run also prints and saves:
  - the remaining post count,
  - how many of the remaining posts are political (broad definition),
  - a monthly breakdown of remaining and political posts.

Dependencies
------------
  pip install fasttext-langdetect pandas pyarrow tqdm

Run
---
  python hive_language_filter_90.py
"""

import os
import re
import time
from typing import Dict, List
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm


# -- Configuration ----------------------------------------------------------

CONFIG = {
    # Input: the classified corpus (output of classify-full-dataset.py).
    "input_file":  r"hive_classified.parquet",
    # Output: kept posts only (body unaltered).
    "output_file": r"hive_english_filtered.parquet",
    # Summary files.
    "report_file":  r"language_filter_report.csv",
    "monthly_file": r"monthly_breakdown.csv",

    # Body column to read for detection, and the fallback if it is absent.
    # Detection reads this column; the column itself is never modified.
    "body_column":   "body_clean",
    "body_fallback": "body",

    # Keep a post when at least this share of its detectable text is English.
    "english_share_threshold": 0.90,

    # Minimum total detectable characters needed to judge a post; below this
    # the post is marked 'too_short'.
    "min_chars_for_detection": 30,

    # Minimum sentence length (preprocessed characters) considered for
    # detection. Very short fragments are ignored to reduce detector noise.
    "sentence_min_chars": 15,

    # Parallelism. None uses all logical cores.
    "n_workers": None,
}


# -- Political category definitions -----------------------------------------
# A post counts as political (broad sense) when it was classified YES, or
# when its category is one of the political categories below. 'non_us_politics'
# is included explicitly because the classifier files it under the
# non-political subset.

POLITICAL_CATEGORIES = {
    "election_campaign", "election_fraud", "trump_administration",
    "biden_administration", "foreign_policy", "domestic_policy",
    "political_violence", "political_commentary", "other_political",
}
POLITICAL_BROAD = POLITICAL_CATEGORIES | {"non_us_politics"}


# -- Worker globals (initialized once per process) --------------------------

_DETECTOR = None


def _worker_init():
    """Load the fastText detector once per worker process."""
    global _DETECTOR
    from ftlangdetect import detect as _detect
    _DETECTOR = _detect


# -- Preprocessing and sentence splitting -----------------------------------

_RE_CODE_BLOCK    = re.compile(r"```[\s\S]*?```")
_RE_INLINE_CODE   = re.compile(r"`[^`]+`")
_RE_URL           = re.compile(r"https?://\S+|www\.\S+")
_RE_HASHTAG       = re.compile(r"#\w+")
_RE_CRYPTO_TICKER = re.compile(r"\$[A-Z]{2,10}\b")
_RE_LONG_ALPHANUM = re.compile(r"\b[a-zA-Z0-9]{25,}\b")
_RE_WHITESPACE    = re.compile(r"\s+")

_RE_PARA_BREAK = re.compile(r"\n\s*\n+")
_RE_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _preprocess(text: str) -> str:
    """Strip non-linguistic noise and collapse whitespace for detection only.

    This is applied to a temporary copy used purely to measure the language
    share. It does not change the stored post body. Collapsing whitespace
    also removes newlines, which the detector requires.
    """
    s = _RE_CODE_BLOCK.sub(" ", text)
    s = _RE_INLINE_CODE.sub(" ", s)
    s = _RE_URL.sub(" ", s)
    s = _RE_HASHTAG.sub(" ", s)
    s = _RE_CRYPTO_TICKER.sub(" ", s)
    s = _RE_LONG_ALPHANUM.sub(" ", s)
    s = _RE_WHITESPACE.sub(" ", s)
    return s.strip()


def _segment_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter (blank lines, newlines, then .!? )."""
    out: List[str] = []
    for para in _RE_PARA_BREAK.split(text):
        para = para.strip()
        if not para:
            continue
        for line in para.split("\n"):
            line = line.strip()
            if not line:
                continue
            for part in _RE_SENT_SPLIT.split(line):
                part = part.strip()
                if part:
                    out.append(part)
    return out


# -- Per-post worker --------------------------------------------------------

def process_post(args):
    """Measure the English share of one post. The body is not modified.

    Returns (pos, pct_english, dominant_lang, lang_status).
    """
    pos, body, share_thresh, min_det_chars, sent_min_chars = args

    if not isinstance(body, str) or not body.strip():
        return (pos, 0.0, "", "empty")

    eng = 0
    tot = 0
    lang_chars: Dict[str, int] = {}
    for sent in _segment_sentences(body):
        cleaned = _preprocess(sent)
        if len(cleaned) < sent_min_chars:
            continue
        try:
            rs = _DETECTOR(cleaned, low_memory=True)
        except Exception:
            continue
        lang = rs["lang"]
        length = len(cleaned)
        tot += length
        lang_chars[lang] = lang_chars.get(lang, 0) + length
        if lang == "en":
            eng += length

    if tot < min_det_chars:
        return (pos, 0.0, "", "too_short")

    pct = eng / tot
    dominant = max(lang_chars, key=lang_chars.get) if lang_chars else "und"
    status = "kept" if pct >= share_thresh else "dropped_low_english"
    return (pos, pct, dominant, status)


# -- Main processing --------------------------------------------------------

def run(cfg: Dict) -> pd.DataFrame:
    path = cfg["input_file"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input parquet not found: {path}")

    print(f"Loading {path} ...")
    df = pd.read_parquet(path)
    print(f"  {len(df):,} rows loaded")

    body_col = cfg["body_column"] if cfg["body_column"] in df.columns \
        else cfg["body_fallback"]
    if body_col not in df.columns:
        raise KeyError(
            f"Neither '{cfg['body_column']}' nor '{cfg['body_fallback']}' "
            f"found. Columns: {list(df.columns)}"
        )
    print(f"  Reading body from column: '{body_col}' (not modified)")
    print(f"  English share threshold: {cfg['english_share_threshold']}")

    n_workers = cfg["n_workers"] or os.cpu_count() or 4
    print(f"  Workers: {n_workers}\n")

    bodies = df[body_col].tolist()
    n = len(bodies)

    args_iter = (
        (i,
         bodies[i] if isinstance(bodies[i], str) else "",
         cfg["english_share_threshold"],
         cfg["min_chars_for_detection"],
         cfg["sentence_min_chars"])
        for i in range(n)
    )

    pcts     = np.empty(n, dtype=np.float32)
    doms     = np.empty(n, dtype=object)
    statuses = np.empty(n, dtype=object)

    chunksize = max(50, n // (n_workers * 50))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers,
                             initializer=_worker_init) as ex:
        with tqdm(total=n, desc="Detecting", unit="post") as pbar:
            for (pos, pct, dom, status) in ex.map(
                process_post, args_iter, chunksize=chunksize
            ):
                pcts[pos]     = pct
                doms[pos]     = dom
                statuses[pos] = status
                pbar.update(1)
    elapsed = time.time() - t0
    print(f"\n  Detection complete in {elapsed:.1f}s "
          f"({elapsed / max(1, n) * 1000:.2f} ms per post)")

    df = df.copy()
    df["pct_english"]   = pcts
    df["dominant_lang"] = doms
    df["lang_status"]   = statuses
    return df


# -- Report (counts, political share, monthly breakdown) --------------------

def write_report(df: pd.DataFrame, cfg: Dict) -> None:
    print("\n" + "=" * 72)
    print("Language filter report")
    print("=" * 72)

    n_total = len(df)
    counts = df["lang_status"].value_counts(dropna=False)
    order = ["kept", "dropped_low_english", "too_short", "empty"]

    print(f"\n  Status distribution (N = {n_total:,}):")
    for s in order:
        c = int(counts.get(s, 0))
        print(f"    {s:<22s}  {c:>10,}  ({c / n_total * 100:>6.2f}%)")

    kept = df[df["lang_status"] == "kept"].copy()
    kept_n = len(kept)
    print(f"\n  Remaining posts (>= {cfg['english_share_threshold']:.0%} "
          f"English): {kept_n:,}  ({kept_n / n_total * 100:.2f}% of {n_total:,})")

    # Political share among the remaining posts.
    if "classification" in kept.columns and "category" in kept.columns:
        kept["is_political"] = (
            (kept["classification"] == "YES")
            | kept["category"].isin(POLITICAL_BROAD)
        )
        pol_n = int(kept["is_political"].sum())
        print(f"  Of which political (broad): {pol_n:,}  "
              f"({pol_n / max(1, kept_n) * 100:.2f}% of remaining)")
    else:
        kept["is_political"] = False
        pol_n = 0
        print("  (classification/category columns absent; "
              "political count skipped)")

    # Top languages among dropped posts, for auditing.
    dropped = df[df["lang_status"] == "dropped_low_english"]
    if len(dropped) > 0:
        print(f"\n  Top dominant languages among dropped posts:")
        for lang, c in dropped["dominant_lang"].value_counts().head(8).items():
            print(f"    {str(lang):<6s}  {c:>10,}  "
                  f"({c / len(dropped) * 100:>5.1f}%)")

    # Monthly breakdown of remaining posts (total and political).
    kept["month"] = (
        pd.to_datetime(kept["created"], errors="coerce").dt.strftime("%Y-%m")
    )
    monthly = (
        kept.groupby("month", dropna=False)
        .agg(remaining=("permlink", "size"),
             political=("is_political", "sum"))
        .reset_index()
        .sort_values("month")
    )
    monthly["political_pct"] = (
        monthly["political"] / monthly["remaining"].replace(0, np.nan) * 100
    ).round(1)

    print(f"\n  Monthly breakdown of remaining posts:")
    print(f"    {'month':<10s}  {'remaining':>12s}  {'political':>12s}  "
          f"{'pol %':>7s}")
    for _, r in monthly.iterrows():
        print(f"    {str(r['month']):<10s}  {int(r['remaining']):>12,}  "
              f"{int(r['political']):>12,}  {r['political_pct']:>6.1f}%")
    print(f"    {'TOTAL':<10s}  {kept_n:>12,}  {pol_n:>12,}  "
          f"{pol_n / max(1, kept_n) * 100:>6.1f}%")

    # Save summaries.
    rows = [{"status": s, "n": int(counts.get(s, 0)),
             "pct": round(counts.get(s, 0) / n_total * 100, 3)} for s in order]
    rows.append({"status": "remaining_total", "n": kept_n,
                 "pct": round(kept_n / n_total * 100, 3)})
    rows.append({"status": "remaining_political", "n": pol_n,
                 "pct": round(pol_n / max(1, kept_n) * 100, 3)})
    pd.DataFrame(rows).to_csv(cfg["report_file"], index=False)
    monthly.to_csv(cfg["monthly_file"], index=False)
    print(f"\n  Saved: {cfg['report_file']} and {cfg['monthly_file']}")


# -- Atomic write (kept posts only) -----------------------------------------

def write_kept(df: pd.DataFrame, cfg: Dict) -> None:
    kept = df[df["lang_status"] == "kept"].copy()
    out_path = cfg["output_file"]
    tmp_path = out_path + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    print(f"\nWriting kept posts: {tmp_path}")
    kept.to_parquet(tmp_path, index=False)
    size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    print(f"  Wrote {len(kept):,} rows ({size_mb:.1f} MB)")

    print("Verifying ...")
    check = pd.read_parquet(tmp_path)
    if len(check) != len(kept):
        os.remove(tmp_path)
        raise RuntimeError(
            f"Row count mismatch: wrote {len(kept):,}, read back {len(check):,}"
        )
    print("  Verification passed.")

    if os.path.exists(out_path):
        os.remove(out_path)
    os.rename(tmp_path, out_path)
    print(f"\nFinal output: {out_path}")
    print(f"  Size: {os.path.getsize(out_path) / (1024 * 1024):.1f} MB")


# -- Entry point ------------------------------------------------------------

def main() -> None:
    cfg = CONFIG
    print("=" * 72)
    print("Hive language filter (90% English, text unaltered)")
    print("=" * 72)
    df = run(cfg)
    write_report(df, cfg)
    write_kept(df, cfg)
    print("\nDone. The output contains only the kept posts, with bodies "
          "unchanged, ready for sentiment and emotional-language analysis.")


if __name__ == "__main__":
    # On Windows, ProcessPoolExecutor requires the entry point guard.
    main()
