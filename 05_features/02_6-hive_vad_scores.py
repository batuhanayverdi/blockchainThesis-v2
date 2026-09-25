"""
Hive valence, arousal, and dominance scoring (NRC-VAD v1, lexicon only)
=======================================================================
Produces an alternative operationalization of emotional language for the
H2 robustness extension. No model is involved: each token is looked up in
the NRC-VAD Lexicon (Mohammad, 2018) and the matched scores are averaged,
which makes this measure independent of the DistilRoBERTa classifier and
of the GPT accuracy pipeline.

Preprocessing follows the ECIS approach (Transcript Cleaning for
Arousal.py): lowercase, strip URLs, hashtags, emojis and non-ASCII
characters, drop punctuation, then POS-tag and lemmatize with WordNet
before tokenizing with TreebankWordTokenizer. YouTube-specific steps
(speaker labels, timestamps, subscribe-and-like phrases) are dropped
because they do not occur in Hive posts, and body_for_analysis has
already been stripped of markdown, footers, and crypto noise upstream.

Two deliberate differences from the ECIS script
-----------------------------------------------
1. Posts with no lexicon matches receive NaN, not 0.0. A score of 0.0 is
   a valid position on the 0-1 scale (maximally negative valence), so
   coding unmeasured posts as 0.0 would insert extreme false values into
   the regression. NaN lets them drop explicitly.
2. All three dimensions are scored in one pass, and per-post coverage is
   recorded so the lexicon's fit to Hive vocabulary can be reported.

Input:
  hive_sample_master.parquet   the same file and text column used for
                               the DistilRoBERTa scores, so the post set
                               and text basis are identical
  The NRC VAD Lexicon _Full Data_data.csv

Output (in OUT_DIR):
  hive_vad_scores.parquet      one row per post:
    doc_id, valence, arousal, dominance
    n_tokens        tokens after cleaning
    n_matched       tokens found in the lexicon
    vad_coverage    n_matched / n_tokens
  hive_vad_scores.xlsx         same, for inspection

Citation: Mohammad, S. M. (2018). Obtaining reliable human ratings of
valence, arousal, and dominance for 20,000 English words. Proceedings of
the 56th Annual Meeting of the Association for Computational Linguistics.
The lexicon is licensed for non-commercial research use and must not be
redistributed, so keep the CSV out of any public repository.

Requirements: pandas, numpy, pyarrow, openpyxl, nltk, tqdm
Run time: a few minutes on CPU for 6,242 posts.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import nltk
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer, word_tokenize


# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = (
    r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_hive_pipeline"
    r"\hive sample preparation (politic-non-politic)\hive_sample_master.parquet"
)
VAD_CSV = (
    r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_hive_pipeline"
    r"\17. NRC_VAD\The NRC VAD Lexicon _Full Data_data.csv"
)
EMOTION_FILE = (
    r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_hive_pipeline"
    r"\hive_emotion_scores.parquet"
)
OUT_DIR = (
    r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_hive_pipeline"
    r"\17. NRC_VAD"
)

TEXT_COLUMN = "body_for_analysis"
TEXT_FALLBACKS = ["body_english_extracted", "body_clean", "body"]

# Posts with very few matched tokens give unstable means; scores below this
# threshold are kept but flagged in the diagnostics so the sensitivity of
# the results to sparse matching can be checked in R.
MIN_MATCHED_FLAG = 10


# ============================================================================
# NLTK RESOURCES
# ============================================================================

def ensure_nltk() -> None:
    """NLTK 3.9 renamed two resources; request both names so the script
    works on old and new installations alike."""
    resources = [
        "punkt", "punkt_tab",
        "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
        "wordnet", "omw-1.4",
    ]
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception:                              # pragma: no cover
            pass


# ============================================================================
# LEXICON
# ============================================================================

def load_vad_lexicon(path: str) -> dict:
    """Return {word: (valence, arousal, dominance)}. Column names in the
    exported CSV carry suffixes such as 'arousal (a.scores.tsv)', so they
    are matched by prefix rather than assumed."""
    lex = pd.read_csv(path)
    cols = {c.strip().lower(): c for c in lex.columns}

    def find(prefix):
        for key, original in cols.items():
            if key.startswith(prefix):
                return original
        raise KeyError(
            f"No column starting with '{prefix}' in {path}. "
            f"Columns found: {list(lex.columns)}"
        )

    term_col = find("term")
    v_col, a_col, d_col = find("valence"), find("arousal"), find("dominance")

    lex = lex[[term_col, v_col, a_col, d_col]].copy()
    lex.columns = ["word", "valence", "arousal", "dominance"]
    lex["word"] = lex["word"].astype(str).str.strip().str.lower()
    for c in ["valence", "arousal", "dominance"]:
        lex[c] = pd.to_numeric(lex[c], errors="coerce")
    lex = lex.dropna(subset=["word", "valence", "arousal", "dominance"])
    lex = lex.drop_duplicates("word", keep="first")

    print(f"  Lexicon entries loaded: {len(lex):,}")
    for c in ["valence", "arousal", "dominance"]:
        print(f"    {c}: min {lex[c].min():.3f}  max {lex[c].max():.3f}  "
              f"mean {lex[c].mean():.3f}")
    return {
        row.word: (row.valence, row.arousal, row.dominance)
        for row in lex.itertuples(index=False)
    }


# ============================================================================
# CLEANING (ECIS approach, adapted to Hive)
# ============================================================================

lemmatizer = WordNetLemmatizer()
treebank = TreebankWordTokenizer()


def get_wordnet_pos(treebank_tag: str):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("N"):
        return wordnet.NOUN
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def clean_and_lemmatize(text: str) -> str:
    """Same sequence as the ECIS cleaner, minus the YouTube-only steps."""
    if not isinstance(text, str):
        return ""
    text = text.lower()

    # bracketed asides and parentheticals
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)

    # URLs and hashtags
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"#\w+", "", text)

    # emojis and any non-ASCII characters
    text = re.sub(r"[^\x00-\x7F]+", "", text)

    # repeated dashes and angle brackets
    text = re.sub(r">+", "", text)
    text = re.sub(r"[-]{2,}", "", text)

    # punctuation except sentence markers
    text = re.sub(r"[^\w\s.,!?]", "", text)

    # short repeated stutters, e.g. "ha ha ha"
    text = re.sub(r"\b(\w{1,3})(\s+\1){2,}\b", "", text)

    # stray single characters other than a and i
    text = re.sub(r"\b(?![ai]\b)[a-zA-Z]\b", "", text)

    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""

    tokens = word_tokenize(text)
    if not tokens:
        return ""
    tagged = pos_tag(tokens)
    lemmas = [
        lemmatizer.lemmatize(tok, get_wordnet_pos(pos)) for tok, pos in tagged
    ]
    return " ".join(lemmas)


# ============================================================================
# SCORING
# ============================================================================

def score_text(text: str, vad_map: dict):
    """Mean valence, arousal, and dominance over matched tokens.
    Returns NaN scores when nothing matches, never 0.0."""
    tokens = treebank.tokenize(text.lower())
    cleaned = [re.sub(r"[^a-z]", "", t) for t in tokens]
    cleaned = [t for t in cleaned if t]

    hits = [vad_map[t] for t in cleaned if t in vad_map]
    n_tokens, n_matched = len(cleaned), len(hits)

    if n_matched == 0:
        return np.nan, np.nan, np.nan, n_tokens, 0

    arr = np.asarray(hits, dtype=float)
    return arr[:, 0].mean(), arr[:, 1].mean(), arr[:, 2].mean(), n_tokens, n_matched


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Hive VAD scoring (NRC-VAD v1, lexicon only)")
    print("=" * 72)

    ensure_nltk()
    print("\nLoading lexicon ...")
    vad_map = load_vad_lexicon(VAD_CSV)

    print(f"\nLoading posts: {INPUT_FILE}")
    df = pd.read_parquet(INPUT_FILE)
    print(f"  Rows loaded: {len(df):,}")

    text_col = TEXT_COLUMN if TEXT_COLUMN in df.columns else None
    if text_col is None:
        text_col = next((c for c in TEXT_FALLBACKS if c in df.columns), None)
    if text_col is None:
        raise KeyError(
            f"No text column found. Looked for {TEXT_COLUMN} and {TEXT_FALLBACKS}."
        )
    print(f"  Text column: {text_col}")

    if "doc_id" not in df.columns:
        if {"author", "permlink"} <= set(df.columns):
            df["doc_id"] = (df["author"].astype(str) + "/"
                            + df["permlink"].astype(str))
        else:
            raise KeyError("Need doc_id, or author and permlink, to build keys.")

    records = []
    for doc_id, raw in tqdm(
        zip(df["doc_id"].astype(str), df[text_col]),
        total=len(df),
        desc="Scoring",
    ):
        cleaned = clean_and_lemmatize(raw)
        v, a, d, n_tokens, n_matched = score_text(cleaned, vad_map)
        records.append(
            {
                "doc_id": doc_id,
                "valence": v,
                "arousal": a,
                "dominance": d,
                "n_tokens": n_tokens,
                "n_matched": n_matched,
            }
        )

    out = pd.DataFrame(records)
    out["vad_coverage"] = np.where(
        out["n_tokens"] > 0, out["n_matched"] / out["n_tokens"].replace(0, np.nan), np.nan
    )

    # ---- diagnostics -------------------------------------------------------
    print("\n" + "=" * 72)
    print("DIAGNOSTICS")
    print("=" * 72)
    scored = out["valence"].notna()
    print(f"  Posts scored: {int(scored.sum()):,} of {len(out):,}")
    print(f"  Posts with no lexicon match (NaN): {int((~scored).sum()):,}")
    print(f"  Posts with fewer than {MIN_MATCHED_FLAG} matched tokens: "
          f"{int((out['n_matched'] < MIN_MATCHED_FLAG).sum()):,}")
    cov = out.loc[scored, "vad_coverage"]
    print(f"  Coverage (matched / tokens): mean {cov.mean() * 100:.1f}%  "
          f"p10 {cov.quantile(0.10) * 100:.1f}%  "
          f"p50 {cov.quantile(0.50) * 100:.1f}%  "
          f"p90 {cov.quantile(0.90) * 100:.1f}%")
    for c in ["valence", "arousal", "dominance"]:
        s = out.loc[scored, c]
        print(f"  {c:<10} mean {s.mean():.3f}  sd {s.std():.3f}  "
              f"min {s.min():.3f}  max {s.max():.3f}")

    # convergent validity against the DistilRoBERTa scores, if available
    if Path(EMOTION_FILE).exists():
        emo = pd.read_parquet(EMOTION_FILE)
        if "doc_id" not in emo.columns and {"author", "permlink"} <= set(emo.columns):
            emo["doc_id"] = (emo["author"].astype(str) + "/"
                             + emo["permlink"].astype(str))
        emo_cols = [
            c for c in ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
            if c in emo.columns
        ]
        if emo_cols:
            merged = out.merge(
                emo[["doc_id", *emo_cols]], on="doc_id", how="inner"
            )
            print("\n  Correlations with the DistilRoBERTa emotion scores")
            print("  (valence should be negative for the negative emotions "
                  "and positive for joy):")
            for c in emo_cols:
                r_v = merged["valence"].corr(merged[c])
                r_a = merged["arousal"].corr(merged[c])
                r_d = merged["dominance"].corr(merged[c])
                print(f"    {c:<9} valence {r_v:+.3f}   arousal {r_a:+.3f}   "
                      f"dominance {r_d:+.3f}")

    # ---- write -------------------------------------------------------------
    parquet_path = out_dir / "hive_vad_scores.parquet"
    xlsx_path = out_dir / "hive_vad_scores.xlsx"
    out.to_parquet(parquet_path, index=False)
    print(f"\nWrote: {parquet_path}")
    try:
        out.to_excel(xlsx_path, index=False)
        print(f"Wrote: {xlsx_path}")
    except Exception as e:                             # pragma: no cover
        print(f"(Excel write skipped: {e})")

    print("\nDone. Send the console output back, especially the coverage "
          "figures and the correlations, before the variables go into the "
          "regression dataset.")


if __name__ == "__main__":
    main()
