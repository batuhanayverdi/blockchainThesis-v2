"""
Transcript preprocessing for ClaimBuster: clean + restore punctuation (CPU, PARALLEL)
=====================================================================================
Same output as the single-process version, processed across worker processes.

What changed from the first parallel version (fixes the BrokenProcessPool crash):
  - Model LOADING is serialized with a lock, so only one worker loads at a time.
    Six 2.2GB models loading at once spiked past RAM and the OS killed a worker.
    Loading one at a time caps the spike; inference still runs fully in parallel.
  - The model is loaded explicitly as float32, so the "falling back to float32"
    path (which briefly doubles memory during load) never triggers.

If a worker is still killed, lower WORKERS to 4 (the only knob you need).

Install: pip install deepmultilingualpunctuation
"""

import os
import re
import json
import time
from pathlib import Path
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor

# -- transformers compatibility shim (runs in every worker process on import) --
import deepmultilingualpunctuation.punctuationmodel as _pm
from transformers import pipeline as _hf_pipeline


def _patched_init(self, model="oliverguhr/fullstop-punctuation-multilang-large"):
    # Load explicitly as float32 to avoid the dtype fallback that doubles memory.
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    mdl = AutoModelForTokenClassification.from_pretrained(model, torch_dtype=torch.float32)
    tok = AutoTokenizer.from_pretrained(model)
    self.pipe = _hf_pipeline("ner", model=mdl, tokenizer=tok,
                             aggregation_strategy="none")


_pm.PunctuationModel.__init__ = _patched_init
from deepmultilingualpunctuation import PunctuationModel  # noqa: E402


# ============================ CONFIG ========================================
INPUT_FILE = "Filtered_Transcripts_from_Tutuk et al (2026).json"
OUTPUT_FILE = "Transcripts_ready_for_claimbuster.json"
RECAPITALIZE = True
SAVE_EVERY = 20
WORKERS = 6        # lower to 4 if a worker is still killed during loading


# ===================== CLEANING (your v2, unchanged) =======================
TIGHT_CTA = [
    r'\blike (and )?subscribe\b', r'\bcomment below\b',
    r'\bhit (the )?(notification )?bell\b',
    r'\b(please |make sure to )?ring (the )?bell\b', r'\bstay tuned\b',
    r'\bturn on notifications\b',
    r'\bclick the link (below|in the description)\b',
    r'\bsmash (that |the )?like button\b', r'\bthanks for watching\b',
    r'\b(I\'ll |we\'ll )?see you (next time|in the next (video|one))\b',
]
BOUNDED_CTA = [
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
    for pattern in TIGHT_CTA + BOUNDED_CTA:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(\w{1,3})(\s+\1){2,}\b', r'\1 \1', text)
    text = re.sub(r'-{2,}', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def recapitalize(text: str) -> str:
    if not text:
        return text
    text = re.sub(r'^\s*([a-z])', lambda m: m.group(1).upper(), text)
    text = re.sub(r'([.!?]\s+)([a-z])',
                  lambda m: m.group(1) + m.group(2).upper(), text)
    return text


# ========================= WORKER (one model per process) =================
_MODEL = None


def _init_worker(load_lock):
    import torch
    torch.set_num_threads(1)             # pin each worker to one thread
    global _MODEL
    with load_lock:                      # only one worker loads at a time
        _MODEL = PunctuationModel()


def process_item(item: dict) -> dict:
    cleaned = clean_transcript(item.get("transcript", "") or "")
    item["cleaned_transcript"] = cleaned
    if not cleaned:
        item["punctuated_transcript"] = ""
        return item
    try:
        restored = _MODEL.restore_punctuation(cleaned)
        if RECAPITALIZE:
            restored = recapitalize(restored)
        item["punctuated_transcript"] = restored
    except Exception as e:               # noqa: BLE001
        item["punctuated_transcript"] = cleaned
        item["punctuation_error"] = str(e)
    return item


# ============================== MAIN ======================================
def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} transcripts from {INPUT_FILE}")
    print(f"Using {WORKERS} worker process(es) on {os.cpu_count()} CPU(s)")

    out_path = Path(OUTPUT_FILE)
    t0 = time.time()
    results = []
    n_empty = n_err = 0
    term_before = term_after = 0
    n_was_single = n_now_split = 0

    manager = Manager()
    load_lock = manager.Lock()           # serialize model loading across workers

    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_worker,
                             initargs=(load_lock,)) as ex:
        for i, res in enumerate(ex.map(process_item, data, chunksize=1)):
            results.append(res)
            vid = res.get("video_id", "?")
            cleaned = res.get("cleaned_transcript", "") or ""
            restored = res.get("punctuated_transcript", "") or ""

            if not cleaned:
                n_empty += 1
                print(f"[{i+1}/{len(data)}] {vid} -> empty")
            elif "punctuation_error" in res:
                n_err += 1
                print(f"[{i+1}/{len(data)}] {vid} -> ERROR: {res['punctuation_error']}")
            else:
                b = sum(cleaned.count(c) for c in ".!?")
                a = sum(restored.count(c) for c in ".!?")
                term_before += b
                term_after += a
                if b == 0:
                    n_was_single += 1
                    if a > 0:
                        n_now_split += 1
                print(f"[{i+1}/{len(data)}] {vid} -> terminators {b} -> {a}")

            if (i + 1) % SAVE_EVERY == 0:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"Done in {time.time() - t0:.1f}s. empty={n_empty}, errors={n_err}")
    print(f"Sentence terminators total: {term_before:,} -> {term_after:,}")
    print(f"Transcripts with no terminators before: {n_was_single} "
          f"(now split: {n_now_split})")
    print(f"Output: {out_path.resolve()}")


if __name__ == "__main__":      # required on Windows for multiprocessing
    main()
