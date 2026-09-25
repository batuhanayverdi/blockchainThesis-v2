"""
Hive moral outrage scoring via DOC, v2  (Brady et al. 2021, local, resumable)
=============================================================================
v3: replaces the pretrained_model_predict wrapper with the package's
documented manual pipeline (WordEmbed + _load_crockett_model + predict),
loading tokenizer and model once and caching them. Fixes the v2 return-type
error and avoids reloading the model on every batch.
Prerequisites in .venv_doc: pip install tf-keras keras_preprocessing emoji==1.7.0.
Applies the CrockettLab Digital Outrage Classifier (DOC) to the regression
sample, sentence by sentence, and aggregates to post level.

RUN THIS IN THE SEPARATE .venv_doc ENVIRONMENT, never in the thesis venv:
old TensorFlow can conflict with the torch/transformers stack.

TWO PHASES:
  Phase 0 (default): smoke test. Loads the model, scores a few invented
  example sentences, prints probabilities. Send me this output first.
  Phase 1: set RUN_FULL = True after the smoke test looks sane. Scores all
  kept sentences (same pysbd split and same boilerplate filter as the
  emotional dynamics phase), checkpointed per post, then aggregates.

DOMAIN CAVEAT (Brady et al., 2021): DOC is trained on US political Twitter
discourse and requires validity checks on new samples. The script therefore
writes a spot-check file of classified sentences for manual review BEFORE the
variable is used in any model.

Outputs (in OUT_DIR):
  doc_checkpoint.jsonl              per-post checkpoint (resume-safe)
  hive_outrage_sentences.parquet    one row per kept sentence: text hashopt,
                                    outrage probability, binary label
  hive_outrage_posts.parquet        one row per post: outrage_share,
                                    outrage_mean_prob, outrage_any,
                                    n_sentences_scored
  outrage_spot_check.xlsx           60 sentences (30 classified outrage,
                                    30 not), shuffled, for manual agree /
                                    disagree annotation

ADAPTER NOTE: the exact package API may differ by version. All package
interaction is isolated in load_doc() and predict_probs() below. If either
raises, the script stops with instructions; paste the full error back to me
and I will adjust the adapter.
"""

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")


def _ensure_keras2():
    """DOC was written against Keras 2 (keras.preprocessing.text). Modern
    TensorFlow ships Keras 3, which removed that module. If the Keras 2 API
    is not importable, alias the tf-keras compatibility package under the
    'keras' name BEFORE outrageclf is imported. The explicit submodule
    mappings also let joblib unpickle the fitted Tokenizer inside
    word_embed.joblib, whose pickle references the old module paths."""
    import sys
    try:
        import keras.preprocessing.text  # noqa: F401
        return "native keras 2 API found"
    except Exception:
        pass
    try:
        import tf_keras
        import tf_keras.preprocessing.text
        import tf_keras.models
        sys.modules["keras"] = tf_keras
        sys.modules["keras.preprocessing"] = tf_keras.preprocessing
        sys.modules["keras.preprocessing.text"] = tf_keras.preprocessing.text
        sys.modules["keras.models"] = tf_keras.models
        return "tf-keras shim installed (Keras 2 API aliased as keras)"
    except ImportError as e:
        raise ImportError(
            "Neither Keras 2 nor tf-keras is available. Run: "
            "pip install tf-keras   in the .venv_doc environment."
        ) from e


import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pysbd
from tqdm import tqdm


# ============================================================================
# CONFIGURATION
# ============================================================================

# Pretrained files downloaded per the repository README. FILL THESE IN.
MODEL_PATH = r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_hive_pipeline\11. Moral Outrage\Shareable Model Files\GRU.h5"
EMBED_PATH = r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_hive_pipeline\11. Moral Outrage\Shareable Model Files\26k_training_data.joblib"

SAMPLE_FILE = r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_hive_pipeline\hive sample preparation (politic-non-politic)\hive_sample_master.parquet"
DOC_ID_FILTER = r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_hive_pipeline\9. Sentence Emotion Scoring\doc_id.xlsx"
OUT_DIR = r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_hive_pipeline\11. Moral Outrage"

TEXT_COLUMN = "body_for_analysis"
TEXT_FALLBACKS = ["body_english_extracted", "body_clean", "body"]
MIN_SENT_CHARS = 3            # identical to the dynamics phase
OUTRAGE_THRESHOLD = 0.5       # probability -> binary label
BATCH = 256                   # sentences per predict call
SAVE_EVERY = 200

RUN_FULL = True              # set True after the smoke test looks sane

SMOKE_SENTENCES = [
    # invented examples spanning the construct, NOT from any dataset
    "This is absolutely disgusting and everyone responsible should be "
    "ashamed and punished.",
    "How dare they lie to us like this, they belong in prison.",
    "I cannot believe the corruption, these people are a disgrace.",
    "The meeting is scheduled for Tuesday afternoon in the main hall.",
    "I planted tomatoes this weekend and the weather was lovely.",
    "The quarterly report shows revenue increased by three percent.",
]


# ============================================================================
# PACKAGE ADAPTER (all outrageclf interaction lives here)
# ============================================================================

_DOC_CACHE = {}


def load_doc():
    """Load the tokenizer and GRU model ONCE via the package's documented
    manual pipeline (repository README), instead of the
    pretrained_model_predict wrapper. Two reasons: the wrapper's return is
    not a plain probability array (it surfaced a function object in v2), and
    the smoke-test log shows it reloads the tokenizer and model on EVERY
    call, which would be prohibitively slow across per-post batches."""
    status = _ensure_keras2()
    import tensorflow as tf
    print(f"    keras compatibility: {status}")
    print(f"    tensorflow {tf.__version__}")
    from outrageclf.preprocessing import WordEmbed, get_lemmatize_hashtag
    from outrageclf.classifier import _load_crockett_model
    word_embed = WordEmbed()
    word_embed._get_pretrained_tokenizer(EMBED_PATH)
    model = _load_crockett_model(MODEL_PATH)
    _DOC_CACHE["word_embed"] = word_embed
    _DOC_CACHE["model"] = model
    _DOC_CACHE["lemmatize"] = get_lemmatize_hashtag
    print("    tokenizer and GRU model loaded and cached")
    return True


def predict_probs(sentences):
    """Outrage probabilities via the documented pipeline:
    lemmatize -> embed -> model.predict. Uses the cached objects."""
    if "model" not in _DOC_CACHE:
        load_doc()
    lemmatized = _DOC_CACHE["lemmatize"](list(sentences))
    embedded = _DOC_CACHE["word_embed"]._get_embedded_vector(lemmatized)
    preds = np.asarray(_DOC_CACHE["model"].predict(embedded, verbose=0))
    if preds.ndim == 2 and preds.shape[1] == 2:
        probs = preds[:, 1]      # two-column output: assume outrage col 1;
                                 # the smoke-test ordering validates this
    else:
        probs = preds.ravel()    # single sigmoid unit (expected case)
    if len(probs) != len(sentences):
        raise RuntimeError(
            f"Prediction length {len(probs)} != input {len(sentences)}; "
            "paste this message back for an adapter fix.")
    return [float(p) for p in probs]


# ============================================================================
# SENTENCE SPLITTING AND BOILERPLATE FILTER (identical to Stage 2b)
# ============================================================================

URL_RE = re.compile(r"(https?://|www\.)", re.IGNORECASE)
BULLET_RE = re.compile(r"^[\W_]*[◯∴•▶►★☆✔✅➤·\-=~*]+\s")
HASHTAG_RE = re.compile(r"^\s*#\w")


def is_boilerplate(s: str) -> bool:
    s = s.strip()
    if len(re.findall(r"[A-Za-z]+", s)) < 3:
        return True
    nonspace = re.sub(r"\s", "", s)
    if nonspace and sum(c.isalpha() for c in nonspace) / len(nonspace) < 0.50:
        return True
    if URL_RE.search(s):
        return True
    if s.endswith(":") and len(s) < 60:
        return True
    if BULLET_RE.match(s) or HASHTAG_RE.match(s):
        return True
    return False


def split_kept(text: str, seg) -> list:
    try:
        sents = seg.segment(text)
    except Exception:
        sents = [s for chunk in text.split("\n") for s in chunk.split(". ")]
    sents = [s.strip() for s in sents if len(s.strip()) >= MIN_SENT_CHARS]
    return [s for s in sents if not is_boilerplate(s)]


# ============================================================================
# CHECKPOINT HELPERS
# ============================================================================

def load_done_keys(ckpt_path: Path):
    done = set()
    if ckpt_path.exists():
        with open(ckpt_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["doc_id"])
                except Exception:
                    continue
    return done


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "doc_checkpoint.jsonl"

    print("=" * 72)
    print("Moral outrage scoring via DOC (Brady et al., 2021)")
    print("=" * 72)

    # ---- Phase 0: load model and smoke test ----
    print("  Loading DOC ...")
    try:
        load_doc()
    except Exception as e:
        print("\n  DOC FAILED TO LOAD. This is the decision point we planned "
              "for.\n  Paste the full traceback below back to me:\n")
        raise

    print("  Smoke test (invented sentences, first three should score "
          "clearly higher):")
    try:
        probs = predict_probs(SMOKE_SENTENCES)
    except Exception as e:
        print("\n  DOC LOADED BUT PREDICTION FAILED. Paste the full "
              "traceback back to me:\n")
        raise
    for s, p in zip(SMOKE_SENTENCES, probs):
        print(f"    {p:.3f}   {s[:70]}")

    if not RUN_FULL:
        print("\nSmoke test complete. If the ordering above looks sane "
              "(outrage-style sentences clearly above the neutral ones), "
              "set RUN_FULL = True and run again for the full corpus.")
        return

    # ---- Phase 1: full corpus ----
    df = pd.read_parquet(SAMPLE_FILE)
    if "doc_id" in df.columns:
        df["_key"] = df["doc_id"].astype(str).str.strip()
    else:
        df["_key"] = df["author"].astype(str) + "/" + df["permlink"].astype(str)
    ids = pd.read_excel(DOC_ID_FILTER, usecols=[0])
    keep_ids = set(ids[ids.columns[0]].dropna().astype(str).str.strip())
    df = df[df["_key"].isin(keep_ids)].copy()
    print(f"\n  Posts to score: {len(df):,} (expected 5,565)")

    text_col = TEXT_COLUMN if TEXT_COLUMN in df.columns else None
    if text_col is None:
        for c in TEXT_FALLBACKS:
            if c in df.columns:
                text_col = c
                break

    seg = pysbd.Segmenter(language="en", clean=False)
    done = load_done_keys(ckpt_path)
    if done:
        print(f"  Resuming: {len(done):,} posts already scored")

    n_new = 0
    ckpt = open(ckpt_path, "a", encoding="utf-8")
    try:
        for _, row in tqdm(df.iterrows(), total=len(df),
                           desc="Scoring", unit="post"):
            doc_id = row["_key"]
            if doc_id in done:
                continue
            text = str(row.get(text_col) or "").strip()
            kept = split_kept(text, seg)
            rec = {"doc_id": doc_id, "sentences": []}
            if kept:
                probs = []
                for start in range(0, len(kept), BATCH):
                    probs.extend(predict_probs(kept[start:start + BATCH]))
                for idx, (s, p) in enumerate(zip(kept, probs)):
                    rec["sentences"].append({
                        "sent_idx": idx,
                        "sentence": s,
                        "outrage_prob": round(p, 4),
                        "outrage": int(p >= OUTRAGE_THRESHOLD),
                    })
            ckpt.write(json.dumps(rec, ensure_ascii=False) + "\n")
            ckpt.flush()
            n_new += 1
            if n_new % SAVE_EVERY == 0:
                os.fsync(ckpt.fileno())
    finally:
        ckpt.flush()
        try:
            os.fsync(ckpt.fileno())
        except Exception:
            pass
        ckpt.close()

    # ---- consolidate ----
    print("  Consolidating checkpoint ...")
    sent_rows, post_rows = [], []
    with open(ckpt_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            probs = [s["outrage_prob"] for s in rec["sentences"]]
            labels = [s["outrage"] for s in rec["sentences"]]
            for s in rec["sentences"]:
                sent_rows.append({"doc_id": rec["doc_id"], **s})
            post_rows.append({
                "doc_id": rec["doc_id"],
                "n_sentences_scored": len(probs),
                "outrage_share": (float(np.mean(labels)) if labels else np.nan),
                "outrage_mean_prob": (float(np.mean(probs)) if probs
                                      else np.nan),
                "outrage_any": (int(any(labels)) if labels else 0),
            })

    sent_df = pd.DataFrame(sent_rows)
    post_df = pd.DataFrame(post_rows)
    sent_df.to_parquet(out_dir / "hive_outrage_sentences.parquet", index=False)
    post_df.to_parquet(out_dir / "hive_outrage_posts.parquet", index=False)
    print(f"  Wrote {len(sent_df):,} sentence rows and {len(post_df):,} "
          f"post rows")

    # ---- diagnostics ----
    print("\n  Sentence-level: share classified outrage "
          f"{sent_df['outrage'].mean() * 100:.1f}%   "
          f"mean prob {sent_df['outrage_prob'].mean():.3f}")
    s = post_df["outrage_share"].dropna()
    print(f"  Post-level outrage_share: mean {s.mean():.3f}   "
          f"p50 {s.quantile(0.50):.3f}   p90 {s.quantile(0.90):.3f}   "
          f"any-outrage posts: {post_df['outrage_any'].sum():,}")

    # ---- spot-check file (validity, per Brady et al. 2021) ----
    rng = np.random.default_rng(42)
    pos = sent_df[sent_df["outrage"] == 1]
    neg = sent_df[sent_df["outrage"] == 0]
    take_pos = pos.sample(min(30, len(pos)), random_state=42)
    take_neg = neg.sample(min(30, len(neg)), random_state=42)
    spot = pd.concat([take_pos, take_neg]).sample(frac=1, random_state=42)
    spot = spot[["doc_id", "sentence", "outrage_prob", "outrage"]].copy()
    spot["your_verdict_agree_1_disagree_0"] = ""
    spot.to_excel(out_dir / "outrage_spot_check.xlsx", index=False)
    print(f"\n  Spot-check file written: outrage_spot_check.xlsx "
          f"({len(spot)} sentences). Fill the verdict column and send back "
          "the agreement rates before we model anything.")


if __name__ == "__main__":
    main()
