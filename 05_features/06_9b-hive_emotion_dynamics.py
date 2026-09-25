"""
Stage 2: post-level emotional dynamics for the six discrete emotions
====================================================================
Thesis: Rewarding Deception? The Impact of Tokenized Engagement on
Misinformation Diffusion in Blockchain-Based Social Media.

Purpose
-------
Turns the sentence-level scores from Stage 1 into one row per post. For each
of the six discrete emotions, three measures are computed:

  <emotion>_volatility   The standard deviation of the differences between
                         adjacent sentences, following Berger et al. (2021).
                         With sentence scores s_1 ... s_n, the differences are
                         d_i = s_(i+1) - s_i, and volatility is SD(d) with
                         ddof = 1. It captures how much a post moves between
                         emotional states rather than how emotional it is on
                         average. Undefined for posts with fewer than three
                         sentences, since two differences are the minimum for
                         a sample standard deviation.

  <emotion>_sd           The standard deviation across all sentences, that is,
                         the overall dispersion of the emotion within the post
                         irrespective of ordering. Undefined for posts with
                         fewer than two sentences.

  <emotion>_mean         The average across sentences, the level measure.

Berger et al. (2021) enter volatility together with dispersion and the mean so
that the volatility coefficient is identified net of how varied and how
emotional the post is overall. The regression models follow the same
convention: each emotion is estimated in its own model containing all three of
its measures, and the six emotions are never entered jointly.

Boilerplate filtering
---------------------
A review of the most divergent posts indicated that video-description posts
carry long runs of boilerplate, including donation links, social media
handles, and affiliate lists. Segmented into sentences, this material forms
neutral fragments that dilute the mean and inject swings that are artefacts of
formatting rather than of emotional language. This script therefore flags such
fragments and reports two versions of every measure:

  filtered      the primary measures, computed on the kept sentences
  unfiltered    the same measures on all sentences, reported as a robustness
                check so that the filter's influence is visible rather than
                assumed

Alignment safeguard
-------------------
Flagging requires the sentence text, which the scored parquet does not store.
Because pysbd is deterministic, re-splitting the same source text reproduces
the same sentences in the same order. The script verifies this per post by
comparing the recomputed sentence count with the count in the scored file. Any
post that fails the check is left entirely unfiltered and is listed in the
console output, so nothing is misaligned silently.

Note on scaling
---------------
The measures are written on their raw scale. Standardisation is applied in the
regression script, so that z-scores are computed on the final analytic sample
rather than on this file, which may contain posts that later drop out.

Inputs
------
hive_sentence_emotions.parquet   one row per (doc_id, sent_idx), from Stage 1
hive_sample_master.parquet       source text, needed to re-split for flagging

Outputs
-------
hive_emotion_dynamics.parquet             primary, boilerplate-filtered
hive_emotion_dynamics_unfiltered.parquet  robustness, all sentences
boilerplate_flags.parquet                 per-sentence flags, for audit

Requirements: pandas, numpy, pyarrow, pysbd. Runs in about a minute.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pysbd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(
    r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_hive_pipeline")

SAMPLE_FILE = (BASE_DIR / "hive sample preparation (politic-non-politic)"
               / "hive_sample_master.parquet")
STAGE_DIR = BASE_DIR / "9. Sentence Emotion Scoring"

SENTENCE_FILE = STAGE_DIR / "hive_sentence_emotions.parquet"
FILTERED_FILE = STAGE_DIR / "hive_emotion_dynamics.parquet"
UNFILTERED_FILE = STAGE_DIR / "hive_emotion_dynamics_unfiltered.parquet"
FLAGS_FILE = STAGE_DIR / "boilerplate_flags.parquet"

TEXT_COLUMN = "body_for_analysis"
TEXT_FALLBACKS = ["body_english_extracted", "body_clean", "body"]

# Must match Stage 1 exactly, in both content and order.
EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
MIN_SENT_CHARS = 3

MIN_SENTENCES_VOLATILITY = 3    # two adjacent differences at minimum
MIN_SENTENCES_DISPERSION = 2


# ---------------------------------------------------------------------------
# Boilerplate heuristics
# ---------------------------------------------------------------------------

URL_PATTERN = re.compile(r"(https?://|www\.)", re.IGNORECASE)
BULLET_PATTERN = re.compile(r"^[\W_]*[◯∴•▶►★☆✔✅➤·\-=~*]+\s")
HASHTAG_PATTERN = re.compile(r"^\s*#\w")


def is_boilerplate(sentence):
    """Flag fragments that carry formatting rather than emotional language.

    A fragment is flagged if any of the following holds:
      1. it contains fewer than three alphabetic words
      2. fewer than half of its non-space characters are letters
      3. it contains a URL
      4. it is a short label ending in a colon
      5. it opens with a bullet, a decorative character, or a hashtag
    """
    sentence = sentence.strip()

    alphabetic_words = re.findall(r"[A-Za-z]+", sentence)
    if len(alphabetic_words) < 3:
        return True

    without_spaces = re.sub(r"\s", "", sentence)
    if without_spaces:
        letter_share = sum(c.isalpha() for c in without_spaces) / len(without_spaces)
        if letter_share < 0.50:
            return True

    if URL_PATTERN.search(sentence):
        return True
    if sentence.endswith(":") and len(sentence) < 60:
        return True
    if BULLET_PATTERN.match(sentence) or HASHTAG_PATTERN.match(sentence):
        return True

    return False


# ---------------------------------------------------------------------------
# Text preparation, mirroring Stage 1
# ---------------------------------------------------------------------------

def split_sentences(text, segmenter):
    """Reproduce the Stage 1 segmentation exactly."""
    if not text:
        return []
    try:
        sentences = segmenter.segment(text)
    except Exception:
        sentences = [s for line in text.split("\n") for s in line.split(". ")]
    return [s.strip() for s in sentences if len(s.strip()) >= MIN_SENT_CHARS]


def resolve_text_column(frame):
    if TEXT_COLUMN in frame.columns:
        return TEXT_COLUMN
    for candidate in TEXT_FALLBACKS:
        if candidate in frame.columns:
            return candidate
    raise KeyError(
        f"No text column found. Looked for {TEXT_COLUMN} and {TEXT_FALLBACKS}.")


def load_source_texts():
    """Return a doc_id indexed series of post text."""
    sample = pd.read_parquet(SAMPLE_FILE)
    if "doc_id" in sample.columns:
        sample["doc_id"] = sample["doc_id"].astype(str).str.strip()
    else:
        sample["doc_id"] = (sample["author"].astype(str) + "/"
                            + sample["permlink"].astype(str))
    text_column = resolve_text_column(sample)
    return sample.set_index("doc_id")[text_column]


# ---------------------------------------------------------------------------
# Measures
# ---------------------------------------------------------------------------

def volatility(values):
    """SD of the differences between adjacent sentences, ddof = 1."""
    if len(values) < MIN_SENTENCES_VOLATILITY:
        return np.nan
    return float(np.std(np.diff(values), ddof=1))


def dispersion(values):
    """SD across sentences, ddof = 1."""
    if len(values) < MIN_SENTENCES_DISPERSION:
        return np.nan
    return float(np.std(values, ddof=1))


def aggregate(sentences, count_column):
    """Collapse sentence rows to one row per post.

    `sentences` must already be sorted by doc_id and sent_idx, since
    volatility depends on the order in which sentences appear.
    """
    records = []
    for doc_id, group in sentences.groupby("doc_id", sort=False):
        record = {"doc_id": doc_id, count_column: len(group)}
        for emotion in EMOTIONS:
            values = group[emotion].to_numpy(dtype=float)
            record[f"{emotion}_volatility"] = volatility(values)
            record[f"{emotion}_sd"] = dispersion(values)
            record[f"{emotion}_mean"] = float(np.mean(values))
        records.append(record)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Flagging
# ---------------------------------------------------------------------------

def build_flags(sentences, texts):
    """Return per-sentence boilerplate flags and the list of misaligned posts.

    Posts whose recomputed sentence count differs from the scored count are
    left unfiltered, so a segmentation mismatch can never shift flags onto the
    wrong sentences.
    """
    segmenter = pysbd.Segmenter(language="en", clean=False)
    counts = sentences.groupby("doc_id", sort=False).size()

    frames = []
    misaligned = []
    for doc_id, n_scored in counts.items():
        text = str(texts.get(doc_id, "") or "").strip()
        recomputed = split_sentences(text, segmenter)

        if len(recomputed) != n_scored:
            misaligned.append(doc_id)
            flags = [False] * n_scored
        else:
            flags = [is_boilerplate(s) for s in recomputed]

        frames.append(pd.DataFrame({"doc_id": doc_id,
                                    "sent_idx": range(n_scored),
                                    "boilerplate": flags}))

    return pd.concat(frames, ignore_index=True), misaligned


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_coverage(frame, label):
    print(f"\n  Coverage, {label}:")
    print(f"    posts: {len(frame):,}")
    for emotion in EMOTIONS[:1]:
        defined = frame[f"{emotion}_volatility"].notna().sum()
        print(f"    volatility defined: {defined:,} "
              f"({len(frame) - defined:,} posts have fewer than three "
              f"usable sentences)")


def report_distributions(frame, label):
    print(f"\n  Distributions, {label}:")
    header = f"    {'emotion':<10s} {'volatility':>22s} {'sd':>22s} {'mean':>22s}"
    print(header)
    for emotion in EMOTIONS:
        cells = []
        for measure in ["volatility", "sd", "mean"]:
            column = frame[f"{emotion}_{measure}"].dropna()
            cells.append(f"{column.mean():8.4f} ({column.std():7.4f})")
        print(f"    {emotion:<10s} {cells[0]:>22s} {cells[1]:>22s} "
              f"{cells[2]:>22s}")
    print("    Cells report the mean with the standard deviation in "
          "parentheses.")


def report_collinearity(frame, label):
    """The check that determines whether all three measures can be entered.

    Berger et al. (2021) report that volatility and dispersion could be
    entered together for one of their studies but not for another, where the
    correlation reached .89. Discrete emotion probabilities are concentrated
    near zero for most sentences, so the correlation may be higher here than
    it was for a composite measure. Any emotion showing a very high value
    needs a specification decision before the models are finalised.
    """
    print(f"\n  Collinearity among the three measures, {label}:")
    print(f"    {'emotion':<10s} {'vol x sd':>10s} {'vol x mean':>12s} "
          f"{'sd x mean':>11s}")
    for emotion in EMOTIONS:
        vol = frame[f"{emotion}_volatility"]
        sd = frame[f"{emotion}_sd"]
        mean = frame[f"{emotion}_mean"]
        print(f"    {emotion:<10s} {vol.corr(sd):10.3f} "
              f"{vol.corr(mean):12.3f} {sd.corr(mean):11.3f}")
    print("    Values approaching .90 indicate that volatility and dispersion "
          "cannot be separately identified for that emotion.")


def report_agreement(filtered, unfiltered):
    """How far the filter moves each measure."""
    merged = filtered.merge(unfiltered, on="doc_id", suffixes=("_f", "_u"))
    print("\n  Correlation between the filtered and unfiltered measures:")
    print(f"    {'emotion':<10s} {'volatility':>11s} {'sd':>8s} {'mean':>8s}")
    for emotion in EMOTIONS:
        values = []
        for measure in ["volatility", "sd", "mean"]:
            values.append(merged[f"{emotion}_{measure}_f"].corr(
                merged[f"{emotion}_{measure}_u"]))
        print(f"    {emotion:<10s} {values[0]:11.3f} {values[1]:8.3f} "
              f"{values[2]:8.3f}")
    print("    High values indicate that the filter changes the ranking of "
          "posts only modestly, which supports reporting the unfiltered "
          "version as a robustness check rather than as a rival measure.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("Stage 2: post-level emotional dynamics")
    print("=" * 72)

    if not SENTENCE_FILE.exists():
        raise FileNotFoundError(f"Stage 1 output not found: {SENTENCE_FILE}")

    sentences = pd.read_parquet(SENTENCE_FILE)
    missing = [e for e in EMOTIONS if e not in sentences.columns]
    if missing:
        raise KeyError(f"Emotion columns missing from Stage 1 output: {missing}")

    # Only the six emotions are read. Any composite or valence columns left in
    # the file by earlier versions of the pipeline are ignored.
    keep = ["doc_id", "author", "permlink", "sent_idx"] + EMOTIONS
    sentences = sentences[keep].sort_values(
        ["doc_id", "sent_idx"]).reset_index(drop=True)

    print(f"  Sentence rows: {len(sentences):,}")
    print(f"  Posts: {sentences['doc_id'].nunique():,}")
    print(f"  Emotions: {', '.join(EMOTIONS)}")

    # One row per post, carrying the identifiers that later merges rely on.
    post_index = (sentences.groupby("doc_id", as_index=False)
                  .agg(author=("author", "first"),
                       permlink=("permlink", "first"),
                       n_sentences_total=("sent_idx", "size")))

    # ---- unfiltered measures, the robustness version ----
    unfiltered = post_index.merge(
        aggregate(sentences, "n_sentences_used"), on="doc_id", how="left")
    unfiltered.to_parquet(UNFILTERED_FILE, index=False)
    print(f"\n  Wrote {len(unfiltered):,} rows to {UNFILTERED_FILE.name}")

    # ---- boilerplate flagging ----
    print("\n  Re-splitting the source text to flag boilerplate ...")
    texts = load_source_texts()
    flags, misaligned = build_flags(sentences, texts)
    flags.to_parquet(FLAGS_FILE, index=False)

    if misaligned:
        print(f"  WARNING: {len(misaligned)} posts failed the alignment check "
              f"and were left unfiltered:")
        for doc_id in misaligned[:10]:
            print(f"    {doc_id}")
        if len(misaligned) > 10:
            print(f"    ... and {len(misaligned) - 10} more")
    else:
        print("  Alignment verified for every post.")

    sentences = sentences.merge(flags, on=["doc_id", "sent_idx"], how="left")
    sentences["boilerplate"] = sentences["boilerplate"].fillna(False)

    n_flagged = int(sentences["boilerplate"].sum())
    share_per_post = sentences.groupby("doc_id")["boilerplate"].mean()
    print(f"\n  Sentences flagged: {n_flagged:,} "
          f"({n_flagged / len(sentences) * 100:.1f}% of all sentences)")
    print(f"  Share dropped per post: median {share_per_post.median():.2f}, "
          f"p90 {share_per_post.quantile(0.90):.2f}")
    print(f"  Posts losing more than half their sentences: "
          f"{(share_per_post > 0.5).sum():,}")

    # ---- filtered measures, the primary version ----
    kept = sentences[~sentences["boilerplate"]]
    print(f"  Sentences retained: {len(kept):,}")

    filtered = post_index.merge(
        aggregate(kept, "n_sentences_kept"), on="doc_id", how="left")
    filtered["n_sentences_kept"] = filtered["n_sentences_kept"].fillna(0).astype(int)
    filtered["n_sentences_dropped"] = (filtered["n_sentences_total"]
                                       - filtered["n_sentences_kept"])
    filtered.to_parquet(FILTERED_FILE, index=False)
    print(f"  Wrote {len(filtered):,} rows to {FILTERED_FILE.name}")

    all_flagged = int((filtered["n_sentences_kept"] == 0).sum())
    if all_flagged:
        print(f"  Posts with every sentence flagged, measures set to missing: "
              f"{all_flagged:,}")

    # ---- diagnostics ----
    report_coverage(unfiltered, "unfiltered")
    report_coverage(filtered, "filtered, primary")
    report_distributions(filtered, "filtered, primary")
    report_collinearity(filtered, "filtered, primary")
    report_agreement(filtered, unfiltered)

    print("\n  Note on sample size: this file has one row per scored post. "
          "The analytic sample is set when it is merged into the regression "
          "dataset, and posts without a usable volatility value drop from the "
          "volatility models only. Report both counts in the thesis.")

    print("\nDone. Send the console output back before the regression models "
          "are specified.")


if __name__ == "__main__":
    main()
