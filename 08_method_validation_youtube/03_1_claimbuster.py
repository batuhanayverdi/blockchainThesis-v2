"""
Stage 1 of 4: ClaimBuster gate (local lucafrost/ClaimBuster-DeBERTaV2)
======================================================================
Cleans each post / transcript, scores every sentence with the local ClaimBuster
model on CPU, and keeps a document only if at least one sentence scores above the
threshold. Passing documents are written with their cleaned text; GPT does the
actual claim selection in stage 2.

Reads:  pilot_posts.csv, pilot_transcripts.json   (templates written if missing)
Writes: pilot_outputs/stage1_passed.json
        -> {"documents": [...passed docs with clean_text + cb metadata...]}

Run:    python 1_claimbuster.py
Deps:   pip install transformers torch pandas
No API keys needed for this stage. Runs on CPU.
"""

import os
import re
import json

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================ CONFIG ========================================
CB_MODEL = "lucafrost/ClaimBuster-DeBERTaV2"
CB_POSITIVE_INDEX = 2        # CFS (Check-worthy Factual Statement); set explicitly
                             # for lucafrost (labels: 0=NFS, 1=UFS, 2=CFS)
CB_THRESHOLD = 0.85          # keep the post if any sentence scores strictly above
MIN_SENT_CHARS = 15          # ignore fragments shorter than this before scoring

POSTS_FILE = "pilot_posts.csv"
TRANSCRIPTS_FILE = "pilot_transcripts.json"
OUTPUT_DIR = "pilot_outputs"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "stage1_passed.json")


# ===================== CLEANING + SEGMENTATION =============================
_TIGHT_CTA = [
    r'\blike (and )?subscribe\b', r'\bcomment below\b',
    r'\bhit (the )?(notification )?bell\b',
    r'\b(please |make sure to )?ring (the )?bell\b', r'\bstay tuned\b',
    r'\bturn on notifications\b',
    r'\bclick the link (below|in the description)\b',
    r'\bsmash (that |the )?like button\b', r'\bthanks for watching\b',
    r'\b(I\'ll |we\'ll )?see you (next time|in the next (video|one))\b',
]
_BOUNDED_CTA = [
    r"\bdon'?t forget to [^.!?\n]{0,60}", r'\bsupport us on [^.!?\n]{0,40}',
    r'\bcheck out (our|my) [^.!?\n]{0,40}',
    r'\bsubscribe to [^.!?\n]{0,25}? channel\b',
    r'\bleave (your |a )?(opinion |comment )?below[^.!?\n]{0,40}',
]


def clean_transcript(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\[[^\]]*\]', ' ', text)
    text = re.sub(r'\([^)]*\)', ' ', text)
    text = re.sub(r'\b\d{1,2}:\d{2}(?::\d{2})?\b', ' ', text)
    text = re.sub(r'(>>\s*)?[A-Z][A-Z\s\.\-]{1,30}:', ' ', text)
    text = re.sub(r'\s*>>\s*', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'#\w+', ' ', text)
    for pattern in _TIGHT_CTA + _BOUNDED_CTA:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(\w{1,3})(\s+\1){2,}\b', r'\1 \1', text)
    text = re.sub(r'-{2,}', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_post(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'```.*?```', ' ', text, flags=re.DOTALL)
    text = re.sub(r'`[^`]*`', ' ', text)
    text = re.sub(r'!\[[^\]]*\]\([^)]*\)', ' ', text)
    text = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'[#>*_~`|]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


_ABBREV = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
           "inc", "ltd", "co", "u.s", "u.k", "e.g", "i.e", "a.m", "p.m",
           "no", "vol", "fig", "approx", "dept"}


def split_sentences(text: str) -> list:
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
    print(f"Loading ClaimBuster model: {CB_MODEL}")
    tok = AutoTokenizer.from_pretrained(CB_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(CB_MODEL)
    model.eval()
    id2label = model.config.id2label
    num_labels = model.config.num_labels
    print(f"  num_labels={num_labels}  id2label={id2label}")

    pos_idx = CB_POSITIVE_INDEX
    if pos_idx is None and num_labels > 1:
        for idx, lab in id2label.items():
            if any(k in str(lab).lower()
                   for k in ["check", "cfs", "worthy"]):   # NOT "factual": it is
                pos_idx = int(idx)                          # inside "Non-Factual" too
                break
        if pos_idx is None:
            pos_idx = 1 if num_labels == 2 else max(int(i) for i in id2label)
    print(f"  check-worthy class index = {pos_idx} "
          f"({'sigmoid(single logit)' if num_labels == 1 else id2label.get(pos_idx)})")
    return tok, model, pos_idx, num_labels


@torch.no_grad()
def score_sentences(sentences, tok, model, pos_idx, num_labels, batch_size=16):
    scores = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", truncation=True,
                  padding=True, max_length=256)
        logits = model(**enc).logits
        if num_labels == 1:
            probs = torch.sigmoid(logits).squeeze(-1)
        else:
            probs = torch.softmax(logits, dim=-1)[:, pos_idx]
        scores.extend(probs.tolist())
    return scores


# ============================== IO ========================================
def load_posts(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path).fillna("")
    return [{
        "doc_id": str(r.get("id", "")), "source_type": "hive_post",
        "title": str(r.get("title", "")),
        "author": str(r.get("author", "") or "a Hive user"),
        "date": str(r.get("created", "") or "unknown date"),
        "clean_text": clean_post(str(r.get("body", ""))),
    } for _, r in df.iterrows()]


def load_transcripts(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    docs = []
    for item in data:
        raw = item.get("cleaned_transcript") or item.get("transcript") or ""
        docs.append({
            "doc_id": str(item.get("video_id", "")), "source_type": "youtube_transcript",
            "title": str(item.get("title", "")),
            "author": str(item.get("channel") or item.get("author") or "a YouTube channel"),
            "date": str(item.get("published") or item.get("upload_date") or "unknown date"),
            "clean_text": clean_transcript(raw),
        })
    return docs


def write_templates():
    pd.DataFrame([
        {"id": "post1", "title": "Example title", "author": "alice",
         "created": "2024-10-15", "body": "Put the post text here."}
    ]).to_csv(POSTS_FILE, index=False, encoding="utf-8")
    with open(TRANSCRIPTS_FILE, "w", encoding="utf-8") as f:
        json.dump([{"video_id": "vid1", "title": "Example video",
                    "channel": "Some Channel", "published": "2024-10-15",
                    "transcript": "Put the transcript text here."}],
                  f, ensure_ascii=False, indent=2)
    print(f"Templates written: {POSTS_FILE}, {TRANSCRIPTS_FILE}. Fill and re-run.")


# ============================== MAIN ======================================
def main():
    posts = load_posts(POSTS_FILE)
    transcripts = load_transcripts(TRANSCRIPTS_FILE)
    if posts is None and transcripts is None:
        write_templates()
        return

    docs = (posts or []) + (transcripts or [])
    print(f"Loaded {len(docs)} documents "
          f"({len(posts or [])} posts, {len(transcripts or [])} transcripts)")

    tok, model, pos_idx, num_labels = load_claimbuster()

    passed = []
    for doc in docs:
        sentences = split_sentences(doc["clean_text"])
        scored = []
        if sentences:
            cb = score_sentences(sentences, tok, model, pos_idx, num_labels)
            scored = list(zip(sentences, cb))
        max_score = max((sc for _, sc in scored), default=0.0)
        keep = max_score > CB_THRESHOLD
        print(f"  {doc['source_type']} {doc['doc_id']}: "
              f"{len(sentences)} sentences, max_score={max_score:.3f} "
              f"-> {'KEEP' if keep else 'drop'}")
        if keep:
            doc["cb_max_score"] = round(max_score, 4)
            doc["cb_checkworthy_sentences"] = [   # metadata for your reference only
                {"text": s, "score": round(sc, 4)}
                for s, sc in scored if sc > CB_THRESHOLD
            ]
            passed.append(doc)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"documents": passed}, f, ensure_ascii=False, indent=2)
    print(f"\nStage 1 done. {len(passed)}/{len(docs)} documents passed the "
          f"{CB_THRESHOLD} gate.\nWrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
