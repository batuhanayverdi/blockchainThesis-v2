"""
Hive ClaimBuster gate (GPU, resumable) - Stage 1 for the Hive run
=================================================================
Scores every analyzable Hive post for check-worthiness and writes the MAX score
per post, so the 0.85 threshold can be tuned later without re-scoring.

  - Reads hive_cleaned.parquet and scores `body_for_analysis` (already cleaned by
    hive_cleaning.py). No further text cleaning happens here.
  - Runs on the GPU (Kaggle T4), batching sentences ACROSS posts for speed.
  - Resumable: posts are processed in chunks; each chunk writes a shard parquet
    and is recorded in a progress file. If the Kaggle session drops, just rerun
    and it skips finished chunks.
  - Kaggle-aware paths: if hive_cleaned.parquet is not in the working directory it
    is auto-found under /kaggle/input/<dataset>/, and outputs are written to
    /kaggle/working when that exists. No path editing needed.

doc_id = author + "/" + permlink. Everything else (category, classification,
created, engagement, title, clean_text) is joined back by doc_id in the sampler.

Reads:  hive_cleaned.parquet   (author, permlink, body_for_analysis, lang_status)
Writes: hive_cb_scores.parquet (doc_id, cb_max_score, cb_n_checkworthy, n_sentences)
        cb_shards/shard_*.parquet, cb_progress.json   (resume state)

Run on Kaggle: enable the T4 GPU and turn Internet ON (the model downloads from
Hugging Face), add hive_cleaned.parquet as a dataset, then run this file.
Deps: transformers, torch, pandas, pyarrow (preinstalled on Kaggle GPU images).
"""

import os
import re
import json
import glob

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================ CONFIG ========================================
# Output base: Kaggle working dir if present, else current directory.
OUT_BASE = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."

INPUT_FILE = "hive_cleaned.parquet"        # auto-resolved to the Kaggle mount if absent
OUTPUT_FILE = os.path.join(OUT_BASE, "hive_cb_scores.parquet")
SHARD_DIR = os.path.join(OUT_BASE, "cb_shards")
PROGRESS_FILE = os.path.join(OUT_BASE, "cb_progress.json")

TEXT_COL = "body_for_analysis"
ANALYZABLE_STATUSES = ["kept", "kept_bilingual"]

CB_MODEL = "lucafrost/ClaimBuster-DeBERTaV2"
CB_POSITIVE_INDEX = 2          # CFS (Check-worthy Factual Statement)
CB_THRESHOLD = 0.85            # used only for the n_checkworthy count + final stats
MIN_SENT_CHARS = 15

CHUNK_POSTS = 2000             # posts per chunk (one shard + one checkpoint each)
BATCH_SIZE = 128               # sentences per forward pass on the GPU
MAX_LEN = 256
USE_FP16 = True                # autocast on CUDA. If scores come out constant or
                               # NaN, set this to False (DeBERTa-v2 + fp16 edge case).

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Module-level handles, set in main().
_TOK = _MODEL = _POS = _NUM = None


# ===================== SENTENCE SEGMENTATION ===============================
_ABBREV = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
           "inc", "ltd", "co", "u.s", "u.k", "e.g", "i.e", "a.m", "p.m",
           "no", "vol", "fig", "approx", "dept"}


def split_sentences(text):
    text = (text or "").strip()
    if not text:
        return []
    raw = re.split(r'(?<=[.!?])\s+', text)
    merged, buf = [], ""
    for part in raw:
        buf = (buf + " " + part).strip() if buf else part
        tokens = buf.split()
        last = re.sub(r'[^a-zA-Z.]', '', tokens[-1]).lower().rstrip('.') if tokens else ""
        if last in _ABBREV:
            continue
        merged.append(buf)
        buf = ""
    if buf:
        merged.append(buf)
    return [s.strip() for s in merged if len(s.strip()) >= MIN_SENT_CHARS]


# ========================= CLAIMBUSTER ====================================
def load_claimbuster():
    print(f"Loading ClaimBuster: {CB_MODEL} on {DEVICE}")
    if DEVICE == "cpu":
        print("  WARNING: no GPU detected. On Kaggle, enable the T4 under "
              "Settings > Accelerator, or this will be very slow.")
    tok = AutoTokenizer.from_pretrained(CB_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(CB_MODEL)
    model.to(DEVICE)
    model.eval()
    num = model.config.num_labels
    pos = CB_POSITIVE_INDEX if CB_POSITIVE_INDEX is not None else (1 if num == 2 else num - 1)
    print(f"  num_labels={num}  positive_index={pos}  id2label={model.config.id2label}")
    return tok, model, pos, num


@torch.no_grad()
def score_batch(sentences):
    """Score a flat list of sentences -> list of check-worthiness probabilities."""
    if not sentences:
        return []
    scores = []
    for i in range(0, len(sentences), BATCH_SIZE):
        batch = sentences[i:i + BATCH_SIZE]
        enc = _TOK(batch, return_tensors="pt", truncation=True,
                   padding=True, max_length=MAX_LEN)
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        if DEVICE == "cuda" and USE_FP16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = _MODEL(**enc).logits
        else:
            logits = _MODEL(**enc).logits
        logits = logits.float()
        if _NUM == 1:
            probs = torch.sigmoid(logits).squeeze(-1)
        else:
            probs = torch.softmax(logits, dim=-1)[:, _POS]
        scores.extend(probs.detach().cpu().tolist())
    return scores


def process_chunk(posts, score_fn, threshold=CB_THRESHOLD):
    """posts: list of {doc_id, text}. Returns score rows for the chunk."""
    flat, spans = [], []
    for p in posts:
        sents = split_sentences(p["text"])
        spans.append((len(flat), len(flat) + len(sents)))
        flat.extend(sents)
    scores = score_fn(flat) if flat else []
    rows = []
    for p, (a, b) in zip(posts, spans):
        sc = scores[a:b]
        mx = max(sc) if sc else 0.0
        rows.append({
            "doc_id": p["doc_id"],
            "cb_max_score": round(float(mx), 4),
            "cb_n_checkworthy": int(sum(1 for x in sc if x > threshold)),
            "n_sentences": b - a,
        })
    return rows


# ============================ RESUME ======================================
def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f).get("done_chunks", []))
    return set()


def save_progress(done):
    tmp = PROGRESS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"done_chunks": sorted(done)}, f)
    os.replace(tmp, PROGRESS_FILE)


def resolve_input(path):
    if os.path.exists(path):
        return path
    hits = sorted(glob.glob("/kaggle/input/**/hive_cleaned.parquet", recursive=True))
    return hits[0] if hits else path


# ============================== MAIN ======================================
def main():
    global _TOK, _MODEL, _POS, _NUM

    in_path = resolve_input(INPUT_FILE)
    if not os.path.exists(in_path):
        print(f"Could not find {INPUT_FILE}. Contents of /kaggle/input:")
        listing = glob.glob("/kaggle/input/**", recursive=True)
        if not listing:
            print("  (/kaggle/input is empty: the dataset is not attached to this "
                  "notebook. Click '+ Add Input' in the sidebar and add your "
                  "hive-cleaned dataset, then rerun.)")
        else:
            for p in listing[:50]:
                print("  ", p)
        return
    print(f"Reading: {in_path}")

    df = pd.read_parquet(in_path,
                         columns=["author", "permlink", TEXT_COL, "lang_status"])
    df = df[df["lang_status"].isin(ANALYZABLE_STATUSES)].copy()
    df[TEXT_COL] = df[TEXT_COL].fillna("")
    df = df[df[TEXT_COL].str.len() > 0].reset_index(drop=True)
    df["doc_id"] = (df["author"].astype(str) + "/" + df["permlink"].astype(str))
    n = len(df)
    n_dupe = int(df["doc_id"].duplicated().sum())
    if n_dupe:
        print(f"  WARNING: {n_dupe} duplicate doc_ids (author/permlink); keeping all.")
    print(f"Gating {n:,} analyzable posts.")

    os.makedirs(SHARD_DIR, exist_ok=True)
    done = load_progress()
    n_chunks = (n + CHUNK_POSTS - 1) // CHUNK_POSTS
    print(f"{n_chunks} chunks of {CHUNK_POSTS}; {len(done)} already finished.")

    _TOK, _MODEL, _POS, _NUM = load_claimbuster()

    for ci in range(n_chunks):
        if ci in done:
            continue
        start = ci * CHUNK_POSTS
        chunk = df.iloc[start:start + CHUNK_POSTS]
        posts = [{"doc_id": r.doc_id, "text": getattr(r, TEXT_COL)}
                 for r in chunk.itertuples(index=False)]
        rows = process_chunk(posts, score_batch)
        shard_path = os.path.join(SHARD_DIR, f"shard_{ci:05d}.parquet")
        pd.DataFrame(rows).to_parquet(shard_path, index=False)
        done.add(ci)
        save_progress(done)
        n_pass = sum(1 for x in rows if x["cb_max_score"] > CB_THRESHOLD)
        print(f"  chunk {ci + 1}/{n_chunks}: {len(rows)} posts, "
              f"{n_pass} pass>{CB_THRESHOLD}")

    shards = sorted(glob.glob(os.path.join(SHARD_DIR, "shard_*.parquet")))
    scores = pd.concat((pd.read_parquet(s) for s in shards), ignore_index=True)
    scores.to_parquet(OUTPUT_FILE, index=False)

    n_pass = int((scores["cb_max_score"] > CB_THRESHOLD).sum())
    print(f"\nGate done. Wrote {OUTPUT_FILE} with {len(scores):,} scored posts.")
    print(f"  pass at >{CB_THRESHOLD}: {n_pass:,} ({n_pass / max(1, len(scores)) * 100:.1f}%)")
    print("  cb_max_score percentiles:")
    for q in (0.50, 0.75, 0.90, 0.95, 0.99):
        print(f"    p{int(q * 100)}: {scores['cb_max_score'].quantile(q):.3f}")
    print("\nNext: download hive_cb_scores.parquet; the sampler applies the "
          "threshold and joins category / classification / created / engagement.")


if __name__ == "__main__":
    main()
