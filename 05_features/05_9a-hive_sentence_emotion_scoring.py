"""
Stage 1: sentence-level discrete emotion scoring
================================================
Thesis: Rewarding Deception? The Impact of Tokenized Engagement on
Misinformation Diffusion in Blockchain-Based Social Media.

Purpose
-------
The H2 analysis models emotional language as six discrete emotions rather
than as a single composite score. This script produces the sentence-level
input for the emotional dynamics measures used in that analysis. It splits
each post in the regression sample into sentences and assigns every sentence
a probability for each of the six emotions.

Design decisions
----------------
* Model: j-hartmann/emotion-english-distilroberta-base, the same model used
  for the post-level emotion scores, so that the dynamics measures and the
  level measures remain directly comparable.

* Unit of analysis: the sentence. Emotional dynamics are defined as movement
  between adjacent sentences within a post, following Berger et al. (2021),
  which requires a score for each sentence rather than for the post as a
  whole. Individual sentences very rarely exceed the model's 512-token
  limit; the few that do are truncated.

* Variables retained: the six discrete emotions (anger, disgust, fear, joy,
  sadness, surprise). The model also returns a neutral probability, which is
  written to the output for diagnostic purposes only and does not enter any
  model. Earlier versions of this pipeline derived a single composite score
  as one minus the neutral probability; that composite has been removed, as
  has the VADER valence score, so that the analysis rests on the discrete
  emotions alone.

* Scale: probabilities are left on the raw softmax scale and are not
  renormalised across the six retained categories. This means the six
  probabilities are compositionally dependent, since a sentence dominated by
  neutral language suppresses all six at once. The models are estimated one
  emotion at a time, which limits but does not remove this dependence, and
  it is acknowledged as a measurement limitation.

* Segmentation: pysbd, which is deterministic. Stage 2 depends on this
  property. It re-splits the same text to align boilerplate flags with the
  scored sentences, which avoids a second pass of model inference.

* Batched inference is padded with an attention mask, so batched scoring is
  numerically equivalent to scoring one sentence at a time.

* The run is resumable. Each post is written and flushed to the checkpoint as
  soon as it is scored, so an interrupted run continues where it stopped.

Inputs
------
hive_sample_master.parquet   post text, in the body_for_analysis column
doc_id.xlsx                  the doc_ids of the regression sample

Outputs
-------
hive_sentence_emotions.parquet   one row per (doc_id, sent_idx)
sentence_checkpoint.jsonl        incremental checkpoint, safe to delete once
                                 the parquet has been written

Requirements: torch, transformers, pandas, pyarrow, openpyxl, pysbd, tqdm.
"""

import json
import os
from pathlib import Path

import pandas as pd
import pysbd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(
    r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_hive_pipeline")

SAMPLE_FILE = (BASE_DIR / "hive sample preparation (politic-non-politic)"
               / "hive_sample_master.parquet")
STAGE_DIR = BASE_DIR / "9. Sentence Emotion Scoring"
DOC_ID_FILE = STAGE_DIR / "doc_id.xlsx"

SENTENCE_FILE = STAGE_DIR / "hive_sentence_emotions.parquet"
CHECKPOINT_FILE = STAGE_DIR / "sentence_checkpoint.jsonl"

MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEXT_COLUMN = "body_for_analysis"
TEXT_FALLBACKS = ["body_english_extracted", "body_clean", "body"]

# The six analysis variables, in a fixed order used by every later script.
EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

# Scored and stored, but never modelled. Kept only so that the share of
# neutral language in the corpus can be reported.
DIAGNOSTIC_LABEL = "neutral"

MAX_TOKENS = 512        # model limit, almost never binding at sentence level
BATCH_SIZE = 32         # sentences per forward pass
MIN_SENT_CHARS = 3      # fragments shorter than this are dropped
FSYNC_EVERY = 200       # posts between forced fsyncs


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model():
    """Load the tokenizer and classifier and verify the expected label set."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()

    id_to_label = model.config.id2label
    label_names = [id_to_label[i] for i in range(len(id_to_label))]

    missing = [e for e in EMOTIONS + [DIAGNOSTIC_LABEL]
               if e not in label_names]
    if missing:
        raise ValueError(
            f"The model does not return these expected labels: {missing}. "
            f"Labels returned: {label_names}")

    return tokenizer, model, label_names


def score_sentences(sentences, tokenizer, model, label_names):
    """Score a post's sentences and return one dictionary per sentence.

    Each dictionary holds the six emotion probabilities plus the neutral
    probability. Columns are selected by label name rather than by position,
    so a change in the model's label ordering cannot silently mislabel a
    variable.
    """
    wanted = EMOTIONS + [DIAGNOSTIC_LABEL]
    column_of = {name: i for i, name in enumerate(label_names)}

    scored = []
    for start in range(0, len(sentences), BATCH_SIZE):
        batch = sentences[start:start + BATCH_SIZE]
        encoded = tokenizer(batch, truncation=True, max_length=MAX_TOKENS,
                            padding=True, return_tensors="pt")
        encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

        with torch.no_grad():
            probabilities = F.softmax(model(**encoded).logits, dim=1).cpu()

        for row in range(len(batch)):
            scored.append({name: round(float(probabilities[row,
                                                           column_of[name]]), 4)
                           for name in wanted})
    return scored


# ---------------------------------------------------------------------------
# Text preparation
# ---------------------------------------------------------------------------

def split_sentences(text, segmenter):
    """Split a post into sentences, discarding degenerate fragments.

    Stage 2 reproduces this function exactly. Any change made here must be
    mirrored there, otherwise the boilerplate flags will not align with the
    scored sentences.
    """
    if not text:
        return []
    try:
        sentences = segmenter.segment(text)
    except Exception:
        # pysbd occasionally fails on badly formed markdown; fall back to a
        # crude split so that the post is still scored rather than skipped.
        sentences = [s for line in text.split("\n") for s in line.split(". ")]
    return [s.strip() for s in sentences if len(s.strip()) >= MIN_SENT_CHARS]


def resolve_text_column(frame):
    """Return the name of the text column, preferring body_for_analysis."""
    if TEXT_COLUMN in frame.columns:
        return TEXT_COLUMN
    for candidate in TEXT_FALLBACKS:
        if candidate in frame.columns:
            return candidate
    raise KeyError(
        f"No text column found. Looked for {TEXT_COLUMN} and {TEXT_FALLBACKS}. "
        f"Columns present: {list(frame.columns)}")


def add_doc_key(frame):
    """Add the doc_id key used to join every stage of the pipeline."""
    if "doc_id" in frame.columns:
        frame["doc_id"] = frame["doc_id"].astype(str).str.strip()
    else:
        frame["doc_id"] = (frame["author"].astype(str) + "/"
                           + frame["permlink"].astype(str))
    return frame


def restrict_to_regression_sample(frame):
    """Keep only the posts listed in the regression sample doc_id file."""
    if not DOC_ID_FILE.exists():
        print("  WARNING: no doc_id file found, scoring every row in the "
              "sample file")
        return frame

    ids = pd.read_excel(DOC_ID_FILE, usecols=[0])
    wanted = set(ids[ids.columns[0]].dropna().astype(str).str.strip())

    before = len(frame)
    frame = frame[frame["doc_id"].isin(wanted)].copy()
    print(f"  Regression sample filter: {before:,} -> {len(frame):,} posts")

    not_found = len(wanted) - frame["doc_id"].nunique()
    if not_found:
        print(f"  WARNING: {not_found} doc_ids in the filter file were not "
              f"found in the sample file; check the key format")
    return frame


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def load_completed_ids(path):
    """Return the doc_ids already present in the checkpoint."""
    completed = set()
    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    completed.add(json.loads(line)["doc_id"])
                except Exception:
                    continue
    return completed


def checkpoint_to_parquet(checkpoint_path, output_path):
    """Explode the per-post checkpoint into one row per sentence."""
    rows = []
    with open(checkpoint_path, "r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except Exception:
                continue
            for sentence in record.get("sentences", []):
                rows.append({"doc_id": record["doc_id"],
                             "author": record.get("author"),
                             "permlink": record.get("permlink"),
                             **sentence})

    frame = pd.DataFrame(rows)
    frame.to_parquet(output_path, index=False)
    return frame


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def report(frame):
    """Print the descriptive checks needed before Stage 2."""
    if frame.empty:
        print("  No sentences were scored.")
        return

    per_post = frame.groupby("doc_id").size()
    print("\n  Sentences per post: "
          f"median {per_post.median():.0f}, "
          f"p10 {per_post.quantile(0.10):.0f}, "
          f"p90 {per_post.quantile(0.90):.0f}")
    print(f"  Posts with fewer than 3 sentences (volatility undefined): "
          f"{(per_post < 3).sum():,} of {per_post.size:,}")

    print("\n  Sentence-level emotion probabilities:")
    for emotion in EMOTIONS:
        column = frame[emotion]
        print(f"    {emotion:<10s} mean {column.mean():6.4f}  "
              f"sd {column.std():6.4f}  "
              f"share above .50 {(column > 0.50).mean() * 100:5.1f}%")

    neutral = frame[DIAGNOSTIC_LABEL]
    print(f"\n  Neutral probability (diagnostic only): "
          f"mean {neutral.mean():.4f}, "
          f"share of sentences above .50 {(neutral > 0.50).mean() * 100:.1f}%")
    print("    A high neutral share means the six probabilities are "
          "compressed toward zero, which is expected and is handled by "
          "z-scoring in the regression models.")

    print("\n  Correlations among the six emotions at sentence level:")
    print(frame[EMOTIONS].corr().round(3).to_string())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    STAGE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Stage 1: sentence-level discrete emotion scoring")
    print("=" * 72)
    print(f"  Model:  {MODEL_NAME}")
    print(f"  Device: {DEVICE}")
    print(f"  Emotions retained: {', '.join(EMOTIONS)}")

    tokenizer, model, label_names = load_model()
    segmenter = pysbd.Segmenter(language="en", clean=False)

    if not SAMPLE_FILE.exists():
        raise FileNotFoundError(f"Sample file not found: {SAMPLE_FILE}")
    posts = pd.read_parquet(SAMPLE_FILE)
    print(f"\n  Sample file: {SAMPLE_FILE}")
    print(f"  Rows loaded: {len(posts):,}")

    posts = add_doc_key(posts)
    posts = restrict_to_regression_sample(posts)

    text_column = resolve_text_column(posts)
    print(f"  Text column: {text_column}")
    print(f"  Posts to score: {len(posts):,}")

    completed = load_completed_ids(CHECKPOINT_FILE)
    if completed:
        print(f"  Resuming: {len(completed):,} posts already scored")

    n_scored = n_empty = 0
    checkpoint = open(CHECKPOINT_FILE, "a", encoding="utf-8")
    try:
        for _, post in tqdm(posts.iterrows(), total=len(posts),
                            desc="Scoring", unit="post"):
            doc_id = post["doc_id"]
            if doc_id in completed:
                continue

            text = str(post.get(text_column) or "").strip()
            sentences = split_sentences(text, segmenter)

            record = {"doc_id": doc_id,
                      "author": post.get("author"),
                      "permlink": post.get("permlink"),
                      "sentences": []}

            if sentences:
                scores = score_sentences(sentences, tokenizer, model,
                                         label_names)
                for index, score in enumerate(scores):
                    record["sentences"].append({"sent_idx": index, **score})
            else:
                n_empty += 1

            checkpoint.write(json.dumps(record, ensure_ascii=False) + "\n")
            checkpoint.flush()

            n_scored += 1
            if n_scored % FSYNC_EVERY == 0:
                os.fsync(checkpoint.fileno())
    finally:
        checkpoint.flush()
        try:
            os.fsync(checkpoint.fileno())
        except Exception:
            pass
        checkpoint.close()

    print(f"\n  Newly scored: {n_scored:,}  (posts with no usable text: "
          f"{n_empty:,})")
    print("  Consolidating the checkpoint into parquet ...")
    sentences = checkpoint_to_parquet(CHECKPOINT_FILE, SENTENCE_FILE)
    print(f"  Wrote {len(sentences):,} sentence rows to {SENTENCE_FILE}")

    report(sentences)

    print("\nDone. Stage 2 aggregates these rows into the post-level "
          "volatility, dispersion, and mean measures.")


if __name__ == "__main__":
    main()
