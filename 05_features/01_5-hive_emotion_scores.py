"""
Hive emotional-language scoring  (local CPU, scoped to the regression sample)
============================================================================
Same model and same chunk-and-average method as your YouTube sentiment script
(j-hartmann/emotion-english-distilroberta-base, 512-token windows averaged
across chunks). Built to run on your own machine on the CPU, on the 6,242-post
sample, which is small enough that CPU is fine (expect roughly 15-60 minutes
depending on your machine and how long the posts are).

Crash-safety: every post is written AND flushed to disk immediately, so if the
run freezes and you stop it, or it errors partway, everything scored up to that
point is already saved. Just run the script again and it resumes, skipping the
posts already in the checkpoint.

The text is scored exactly as it sits in body_for_analysis, emoji included.

Requirements:
  pip install torch transformers pandas pyarrow tqdm

Outputs (in OUTPUT_DIR):
  hive_emotion_scores.parquet      one row per post: 7 emotion scores + extras
  emotion_checkpoint.jsonl         incremental checkpoint (safe to delete after)
"""

import json
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


# ============================================================================
# CONFIGURATION
# ============================================================================

# Put this script in the same folder as the parquet, or set an absolute path,
# e.g. r"C:\Users\batuh\PycharmProjects\Thesis\...\hive_sample_master.parquet"
INPUT_FILE = r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_hive_pipeline\hive sample preparation (politic-non-politic)\hive_sample_master.parquet"
OUTPUT_DIR = "."

DEVICE = "cpu"            # this build is for CPU; set to "cuda" if you have a GPU

MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
TEXT_COLUMN = "body_for_analysis"      # cleaned English text; same as the fact-check input
TEXT_FALLBACKS = ["body_english_extracted", "body_clean", "body"]

CHUNK_SIZE = 512          # same window size as the YouTube measure
MAX_CHUNKS = 64           # cap windows per post to bound memory on very long posts
SAVE_EVERY = 200          # how often to force an fsync (every post is flushed anyway)

ANALYZABLE_STATUSES = ["kept", "kept_bilingual"]


# ============================================================================
# CORE: one post -> averaged emotion scores  (verbatim from your script)
# ============================================================================

def analyze_text(text, tokenizer, model, device, labels):
    """Tokenize without truncation, split into CHUNK_SIZE windows, run the
    model on each window, average the softmax probabilities across windows.
    Padding with an attention mask makes batched inference numerically equal
    to per-chunk inference."""
    enc = tokenizer(text, truncation=False, add_special_tokens=True)
    ids = enc["input_ids"]
    if not ids:
        return None, 0

    chunks = [ids[i:i + CHUNK_SIZE] for i in range(0, len(ids), CHUNK_SIZE)]
    if MAX_CHUNKS:
        chunks = chunks[:MAX_CHUNKS]

    maxlen = max(len(c) for c in chunks)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    input_ids = torch.full((len(chunks), maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((len(chunks), maxlen), dtype=torch.long)
    for r, c in enumerate(chunks):
        input_ids[r, :len(c)] = torch.tensor(c, dtype=torch.long)
        attn[r, :len(c)] = 1

    input_ids = input_ids.to(device)
    attn = attn.to(device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attn).logits
        probs = F.softmax(logits, dim=1)          # (n_chunks, n_labels)
    mean = probs.mean(dim=0).cpu().tolist()        # average across windows

    scores = {labels[i]: round(float(mean[i]), 4) for i in range(len(mean))}
    return scores, len(chunks)


# ============================================================================
# CHECKPOINT HELPERS (resumable)
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


def ckpt_to_parquet(ckpt_path: Path, out_path: Path):
    rows = []
    with open(ckpt_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    return df


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "emotion_checkpoint.jsonl"
    out_path = out_dir / "hive_emotion_scores.parquet"

    device = DEVICE
    print("=" * 72)
    print("Hive emotional-language scoring (local)")
    print("=" * 72)
    print(f"  Model:  {MODEL_NAME}")
    print(f"  Device: {device}" + ("  (CPU; this is the slow step, but the sample "
                                    "is small)" if device == "cpu" else ""))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    labels = model.config.id2label
    label_names = [labels[i] for i in range(len(labels))]
    print(f"  Emotions: {', '.join(label_names)}")

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input not found: {INPUT_FILE}. Put this script next to "
            f"hive_sample_master.parquet, or set INPUT_FILE to its full path.")
    df = pd.read_parquet(INPUT_FILE)
    print(f"\n  Input: {INPUT_FILE}")
    print(f"  Rows loaded: {len(df):,}")

    # The sample is already the analysis set, so score it whole. A full corpus
    # would be filtered to the analyzable English set first.
    if "sample_stratum" in df.columns:
        print(f"  This is the regression sample; scoring all {len(df):,} posts.")
    elif "cb_processed" in df.columns:
        df = df[df["cb_processed"]].copy()
        print(f"  Filtered on cb_processed: {len(df):,}")
    elif "lang_status" in df.columns:
        df = df[df["lang_status"].isin(ANALYZABLE_STATUSES)].copy()
        print(f"  Filtered on lang_status: {len(df):,}")

    text_col = TEXT_COLUMN if TEXT_COLUMN in df.columns else None
    if text_col is None:
        for c in TEXT_FALLBACKS:
            if c in df.columns:
                text_col = c
                break
    if text_col is None:
        raise KeyError(f"No text column found. Looked for {TEXT_COLUMN} and "
                       f"{TEXT_FALLBACKS}. Columns: {list(df.columns)}")
    have_doc_id = "doc_id" in df.columns
    print(f"  Text column: {text_col}")
    print(f"  Posts to score: {len(df):,}")

    done = load_done_keys(ckpt_path)
    if done:
        print(f"  Resuming: {len(done):,} already scored, skipping those")

    n_new = n_empty = 0
    ckpt = open(ckpt_path, "a", encoding="utf-8")
    try:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring", unit="post"):
            author = row.get("author")
            permlink = row.get("permlink")
            doc_id = row["doc_id"] if have_doc_id and pd.notna(row.get("doc_id")) \
                else f"{author}/{permlink}"
            if doc_id in done:
                continue

            text = str(row.get(text_col) or "").strip()
            rec = {
                "doc_id": doc_id,
                "post_key": doc_id,          # alias, same value, for older joins
                "author": author,
                "permlink": permlink,
                "num_chunks": 0,
            }
            for name in label_names:
                rec[name] = None

            if text:
                scores, n_chunks = analyze_text(text, tokenizer, model, device, labels)
                if scores is not None:
                    rec.update(scores)
                    rec["num_chunks"] = n_chunks
                    rec["emotional_language"] = round(
                        1.0 - float(scores.get("neutral", 0.0)), 4)
                    rec["dominant_emotion"] = max(
                        label_names, key=lambda n: scores.get(n, 0.0))
                else:
                    n_empty += 1
            else:
                n_empty += 1

            ckpt.write(json.dumps(rec, ensure_ascii=False) + "\n")
            ckpt.flush()                     # save every post immediately
            n_new += 1
            if n_new % SAVE_EVERY == 0:
                os.fsync(ckpt.fileno())      # force to physical disk periodically
    finally:
        ckpt.flush()
        try:
            os.fsync(ckpt.fileno())
        except Exception:
            pass
        ckpt.close()

    print(f"\n  Newly scored: {n_new:,}  (empty text: {n_empty:,})")
    print("  Consolidating checkpoint into parquet ...")
    final = ckpt_to_parquet(ckpt_path, out_path)
    print(f"  Wrote {len(final):,} rows to {out_path}")

    present = [c for c in label_names if c in final.columns]
    if present and len(final):
        means = final[present].mean(numeric_only=True)
        print("\n  Mean emotion scores across posts:")
        for name in present:
            print(f"    {name:<10s} {means[name]:.4f}")
        if "emotional_language" in final.columns:
            print(f"    emotional_language (1 - neutral): "
                  f"{final['emotional_language'].mean():.4f}")

    print("\nDone. Join hive_emotion_scores.parquet on doc_id to the Model 4 "
          "posts.csv and hive_sample_master.parquet for the H1/H2 models.")


if __name__ == "__main__":
    main()
