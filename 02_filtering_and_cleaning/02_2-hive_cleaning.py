"""
Hive Post Cleaning (v3 - wired to the current pipeline)
=======================================================

Same cleaning logic as your v2 (footer/boilerplate/code/markdown handling with
the over-cleaning fixes). The only changes are the inputs, to match the current
pipeline:

  - Reads hive_english_filtered.parquet (output of 1-hive_language_filter_90.py),
    where the body is kept WHOLE and the only status is 'kept'.
  - Cleans the `body_clean` column (the scraper already stripped HTML and images
    into it; this step removes footers, boilerplate, code, tables, mention spam,
    crypto tickers, etc.) into `body_for_analysis`.

The result, `body_for_analysis`, is the single cleaned text used by BOTH the
ClaimBuster gate / fact-check pipeline and the emotional-language model.

Output: hive_cleaned.parquet (all original columns + body_for_analysis +
        body_for_analysis_len), plus a report and a before/after sample file.

Run:  python hive_cleaning.py
Deps: pip install pandas pyarrow numpy tqdm
"""

import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm


# -- Configuration -----------------------------------------------------------

CONFIG = {
    "input_file":  r"hive_english_filtered.parquet",
    "output_file": r"hive_cleaned.parquet",
    "report_file":  r"cleaning_report.csv",
    "samples_file": r"cleaning_samples.txt",
    "n_samples":    25,

    # The current language filter keeps the body whole and marks only 'kept'.
    "analyzable_statuses": ["kept", "kept_bilingual"],

    # Clean from body_clean (HTML/images already stripped by the scraper);
    # fall back to the raw body if body_clean is absent.
    "body_column":   "body_clean",
    "body_fallback": "body",

    "n_workers": None,
}


def resolve_body_col(df: pd.DataFrame, cfg: Dict) -> str:
    if cfg["body_column"] in df.columns:
        return cfg["body_column"]
    if cfg["body_fallback"] in df.columns:
        return cfg["body_fallback"]
    raise KeyError(f"Neither '{cfg['body_column']}' nor '{cfg['body_fallback']}' "
                   f"in columns: {list(df.columns)}")


# -- Code-block handling ----------------------------------------------------

_PROG_LANGS = {
    "python", "py", "ipython",
    "javascript", "js", "typescript", "ts",
    "bash", "sh", "shell", "zsh", "fish", "powershell", "ps", "cmd",
    "sql", "mysql", "postgresql", "sqlite",
    "json", "yaml", "yml", "toml", "ini", "xml",
    "java", "kotlin", "scala", "groovy",
    "cpp", "c++", "c", "csharp", "cs", "c#",
    "rust", "rs", "go", "golang",
    "php", "ruby", "rb", "perl", "lua",
    "html", "css", "scss", "sass", "less",
    "dockerfile", "makefile", "cmake",
    "r", "matlab", "octave",
    "diff", "patch",
    "regex", "regexp",
}

_RE_CODE_BLOCK_FULL = re.compile(
    r"```([a-zA-Z0-9_+#-]*)\s*\n?([\s\S]*?)```",
    re.MULTILINE,
)


def _code_block_replacer(match: re.Match) -> str:
    """Remove code blocks that contain code; keep ones that wrap prose."""
    lang = match.group(1).strip().lower()
    content = match.group(2)
    if lang in _PROG_LANGS:
        return " "
    if len(content) < 100:
        return " "
    return content


# -- Footer handling (tail-only, non-greedy) --------------------------------

_HIVE_FOOTER_PHRASES = [
    r"thanks?\s+(?:so\s+much\s+)?for\s+(?:reading|stopping\s+by|visiting|watching|taking\s+the\s+time)",
    r"thank\s+you\s+(?:so\s+much\s+)?for\s+(?:reading|stopping\s+by|visiting|watching|taking\s+the\s+time)",
    r"have\s+a\s+(?:great|wonderful|nice|good|blessed|amazing|productive)\s+(?:day|week|weekend|evening|night|morning)",
    r"see\s+you\s+(?:tomorrow|soon|next\s+time|in\s+the\s+next\s+post)",
    r"until\s+(?:next\s+time|tomorrow|the\s+next\s+post)",
    r"don'?t\s+forget\s+to\s+(?:upvote|comment|follow|reblog|like|subscribe)",
    r"if\s+you\s+(?:liked|enjoyed)\s+this\s+post",
    r"translated\s+(?:with|using)\s+(?:deepl|google\s+translate|chatgpt)",
    r"for\s+the\s+best\s+experience\s+view\s+this\s+post\s+on",
]
_RE_FOOTER_IN_TAIL = re.compile(
    r"(?i)\b(?:" + "|".join(_HIVE_FOOTER_PHRASES) + r")\b"
)
_FOOTER_TAIL_WINDOW = 400

def _strip_trailing_footer(text: str) -> str:
    if len(text) < 200:
        return text
    tail_start = max(0, len(text) - _FOOTER_TAIL_WINDOW)
    m = _RE_FOOTER_IN_TAIL.search(text[tail_start:])
    if m is None:
        return text
    trim_at = tail_start + m.start()
    # Only treat it as a footer if it sits in the latter part of the post,
    # so an opening greeting cannot erase the whole body of a short post.
    if trim_at < 0.5 * len(text):
        return text
    return text[:trim_at].rstrip(" \t.,;:-\n")


# -- Other compiled regexes (unchanged from v2) -----------------------------

_RE_INLINE_CODE    = re.compile(r"`[^`\n]+`")
_RE_URL            = re.compile(r"https?://\S+|www\.\S+")

_RE_TABLE_SEP      = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$", re.MULTILINE)
_RE_TABLE_ROW      = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
_RE_HRULE          = re.compile(r"^\s*(?:[-*_]\s*){3,}\s*$", re.MULTILINE)
_RE_MD_HEADER      = re.compile(r"^\s*#{1,6}\s+", re.MULTILINE)
_RE_BLOCKQUOTE     = re.compile(r"^>\s*", re.MULTILINE)
_RE_LIST_MARKER    = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+", re.MULTILINE)
_RE_FOOTNOTE_DEF   = re.compile(r"^\s*\[\^[^\]]+\]:.*$", re.MULTILINE)

_RE_POSTED_VIA     = re.compile(
    r"(?im)^\s*posted\s+(?:via|using|with|from|on)\s+.*$"
)

_RE_MENTION_SPAM   = re.compile(
    r"(?:@[a-zA-Z0-9.\-]{3,16}[\s,;]+){4,}@?[a-zA-Z0-9.\-]{0,16}"
)
_RE_CRYPTO_TICKER  = re.compile(r"\$[A-Z]{2,10}\b")
_RE_LONG_ALPHANUM  = re.compile(r"\b[a-zA-Z0-9]{25,}\b")
_RE_FOOTNOTE_REF   = re.compile(r"\[\^[^\]]+\]")
_RE_STRIKETHROUGH  = re.compile(r"~~([^~]+)~~")
_RE_END_HASHTAGS   = re.compile(r"(?:#\w+\s*){2,}\s*$")

_RE_BOLD_ITALIC    = re.compile(r"\*{1,3}([^*\n]+?)\*{1,3}")
_RE_UNDERSCORE_EM  = re.compile(r"(?<!\w)_{1,3}([^_\n]+?)_{1,3}(?!\w)")
_RE_RESIDUAL_MD_LINK = re.compile(r"\[([^\]]+)\]\([^\)]+\)")

_RE_MULTI_NEWLINE  = re.compile(r"\n{3,}")
_RE_WHITESPACE     = re.compile(r"[ \t]+")


# -- Per-post cleaner -------------------------------------------------------

def clean_post(text) -> str:
    if not isinstance(text, str) or not text:
        return ""

    s = text

    s = _RE_CODE_BLOCK_FULL.sub(_code_block_replacer, s)
    s = _RE_INLINE_CODE.sub(" ", s)
    s = _RE_URL.sub(" ", s)

    s = _RE_TABLE_SEP.sub("", s)
    s = _RE_TABLE_ROW.sub(" ", s)
    s = _RE_HRULE.sub("", s)
    s = _RE_MD_HEADER.sub("", s)
    s = _RE_BLOCKQUOTE.sub("", s)
    s = _RE_LIST_MARKER.sub("", s)
    s = _RE_FOOTNOTE_DEF.sub("", s)

    s = _RE_POSTED_VIA.sub("", s)

    s = _RE_MENTION_SPAM.sub(" ", s)
    s = _RE_CRYPTO_TICKER.sub(" ", s)
    s = _RE_LONG_ALPHANUM.sub(" ", s)
    s = _RE_FOOTNOTE_REF.sub("", s)
    s = _RE_STRIKETHROUGH.sub(r"\1", s)
    s = _RE_END_HASHTAGS.sub("", s)

    s = _RE_RESIDUAL_MD_LINK.sub(r"\1", s)
    s = _RE_BOLD_ITALIC.sub(r"\1", s)
    s = _RE_UNDERSCORE_EM.sub(r"\1", s)

    s = _RE_MULTI_NEWLINE.sub("\n\n", s)
    s = _RE_WHITESPACE.sub(" ", s)
    s = "\n".join(line.strip() for line in s.split("\n"))
    s = _RE_MULTI_NEWLINE.sub("\n\n", s)
    s = s.strip()

    # Footer trim runs last so it sees the cleaned tail
    s = _strip_trailing_footer(s)

    return s


def clean_batch(texts: List[str]) -> List[str]:
    return [clean_post(t) for t in texts]


# -- Main --------------------------------------------------------------------

def run_cleaning(cfg: Dict) -> pd.DataFrame:
    print(f"Loading {cfg['input_file']} ...")
    df = pd.read_parquet(cfg["input_file"])
    print(f"  {len(df):,} rows loaded")

    body_col = resolve_body_col(df, cfg)
    print(f"  Cleaning from column: '{body_col}'")

    analyzable_mask = df["lang_status"].isin(cfg["analyzable_statuses"])
    n_analyzable = int(analyzable_mask.sum())
    print(f"  Analyzable posts (lang_status in {cfg['analyzable_statuses']}): "
          f"{n_analyzable:,}")

    df = df.copy()
    df["body_for_analysis"] = ""
    df["body_for_analysis_len"] = 0

    if n_analyzable == 0:
        print("  Nothing to clean.")
        return df

    n_workers = cfg["n_workers"] or os.cpu_count() or 4
    print(f"\nUsing {n_workers} worker processes")

    analyzable_idx = df.index[analyzable_mask].tolist()
    texts = df.loc[analyzable_mask, body_col].fillna("").astype(str).tolist()

    chunk_size = max(200, len(texts) // (n_workers * 50))
    chunks = [texts[i: i + chunk_size] for i in range(0, len(texts), chunk_size)]
    print(f"  Submitted {len(chunks):,} chunks of size ~{chunk_size}")

    cleaned: List[str] = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        with tqdm(total=len(texts), desc="Cleaning", unit="post") as pbar:
            for result_chunk in executor.map(clean_batch, chunks):
                cleaned.extend(result_chunk)
                pbar.update(len(result_chunk))

    elapsed = time.time() - t0
    print(f"  Cleaning complete in {elapsed:.1f}s "
          f"({elapsed / max(1, n_analyzable) * 1000:.2f} ms per post)")

    print("\nApplying results to DataFrame (vectorized) ...")
    cleaned_arr = np.array(cleaned, dtype=object)
    cleaned_lens = np.array([len(c) for c in cleaned], dtype="int64")
    df.loc[analyzable_idx, "body_for_analysis"] = cleaned_arr
    df.loc[analyzable_idx, "body_for_analysis_len"] = cleaned_lens
    return df


# -- Report ------------------------------------------------------------------

def write_report(df: pd.DataFrame, cfg: Dict) -> None:
    print("\n" + "=" * 72)
    print("Cleaning report")
    print("=" * 72)

    body_col = resolve_body_col(df, cfg)
    analyzable = df[df["lang_status"].isin(cfg["analyzable_statuses"])].copy()
    n = len(analyzable)

    before_lens = analyzable[body_col].fillna("").astype(str).str.len()
    after_lens = analyzable["body_for_analysis_len"]
    reduction_pct = (
        (before_lens - after_lens) / before_lens.replace(0, np.nan) * 100
    )

    print(f"\n  Analyzable posts cleaned: {n:,}")
    print(f"\n  Body length (characters):")
    print(f"    {'metric':<10s}  {'before':>10s}  {'after':>10s}")
    print(f"    {'mean':<10s}  {before_lens.mean():>10,.0f}  {after_lens.mean():>10,.0f}")
    print(f"    {'median':<10s}  {int(before_lens.median()):>10,}  {int(after_lens.median()):>10,}")

    print(f"\n  Character reduction from cleaning:")
    print(f"    mean   : {reduction_pct.mean():>6.2f}%")
    print(f"    median : {reduction_pct.median():>6.2f}%")
    print(f"    p95    : {reduction_pct.quantile(0.95):>6.2f}%")

    thresholds = [50, 100, 200, 500]
    print(f"\n  Posts with very short body_for_analysis:")
    for t in thresholds:
        n_short = int((after_lens < t).sum())
        print(f"    < {t:>4d} chars: {n_short:>8,}  ({n_short / max(1, n) * 100:.2f}%)")

    rows = []
    for t in thresholds:
        n_short = int((after_lens < t).sum())
        rows.append({"metric": f"posts_under_{t}_chars", "value": n_short,
                     "pct": round(n_short / max(1, n) * 100, 3)})
    rows.append({"metric": "mean_reduction_pct",
                 "value": round(reduction_pct.mean(), 2), "pct": None})
    rows.append({"metric": "median_reduction_pct",
                 "value": round(reduction_pct.median(), 2), "pct": None})
    pd.DataFrame(rows).to_csv(cfg["report_file"], index=False)
    print(f"\n  Report saved to: {cfg['report_file']}")


def save_samples(df: pd.DataFrame, cfg: Dict) -> None:
    body_col = resolve_body_col(df, cfg)
    analyzable = df[df["lang_status"].isin(cfg["analyzable_statuses"])]
    if len(analyzable) == 0:
        return

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(analyzable.index,
                            min(cfg["n_samples"], len(analyzable)), replace=False)

    lines: List[str] = []
    for i, idx in enumerate(sample_idx, start=1):
        row = df.loc[idx]
        before = str(row.get(body_col, ""))
        after = str(row.get("body_for_analysis", ""))
        lines.append("=" * 80)
        lines.append(f"SAMPLE {i} | author={row.get('author', '?')} | "
                     f"status={row.get('lang_status', '?')} | "
                     f"len_before={len(before)} | len_after={len(after)}")
        lines.append("-" * 80)
        lines.append("BEFORE:")
        lines.append(before[:2000])
        lines.append("-" * 80)
        lines.append("AFTER:")
        lines.append(after[:2000])
        lines.append("")

    Path(cfg["samples_file"]).write_text("\n".join(lines), encoding="utf-8")
    print(f"  Samples saved to: {cfg['samples_file']}")


def write_atomically(df: pd.DataFrame, cfg: Dict) -> None:
    out_path = cfg["output_file"]
    tmp_path = out_path + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    print(f"\nWriting: {tmp_path}")
    df.to_parquet(tmp_path, index=False)
    print(f"  Wrote: {os.path.getsize(tmp_path) / 1024 / 1024:.1f} MB")

    print("Verifying ...")
    check = pd.read_parquet(tmp_path)
    if len(check) != len(df):
        os.remove(tmp_path)
        raise RuntimeError(f"Row count mismatch: {len(df):,} vs {len(check):,}")
    for required in ("body_for_analysis", "body_for_analysis_len"):
        if required not in check.columns:
            os.remove(tmp_path)
            raise RuntimeError(f"Required column '{required}' missing")
    print("  Verification passed.")

    if os.path.exists(out_path):
        os.remove(out_path)
    os.rename(tmp_path, out_path)
    print(f"\nFinal output: {out_path}")


def main() -> None:
    cfg = CONFIG
    print("=" * 72)
    print("Hive post cleaning (v3 - current pipeline)")
    print("=" * 72)
    df = run_cleaning(cfg)
    write_report(df, cfg)
    save_samples(df, cfg)
    write_atomically(df, cfg)
    print("\nDone. Run spot_check_overcleaned.py to verify high-engagement "
          "posts were not mangled, then upload hive_cleaned.parquet to Kaggle "
          "and run the ClaimBuster gate.")


if __name__ == "__main__":
    main()
