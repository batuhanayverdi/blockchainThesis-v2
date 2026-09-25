"""
HIVE MODEL 1 - Stage 2 of 4: query from title (the title IS the claim)
======================================================================
Model 1 treats each post's title as the claim and asks GPT for one neutral
search query per title. Prompt and logic are identical to the validated Model 1
Stage 2; the title now comes from hive_sample_master.parquet (no Excel lookup),
and execution is threaded with resume.

Reads:  hive_sample_master.parquet
Writes: hive_outputs/model1/stage2_claims.json   ({"documents": [...], "claims": [...]})
        hive_outputs/model1/stage2_queries.csv
        hive_outputs/model1/_ckpt_select.json     (resume state)

Run:    python hive_model1_select_query.py
Deps:   pip install openai pandas pyarrow
Key:    OPENAI_API_KEY  (read from the environment, never hard-coded)
"""

import os
import re
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI


# ============================ CONFIG ========================================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GPT_MODEL = "gpt-5-mini"
SELECT_REASONING = "low"
SELECT_MAX_TOKENS = 2000
N_WORKERS = 8
SAVE_EVERY = 50
MAX_RETRIES = 3
REQUEST_PAUSE = 0.4

INPUT_FILE = r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_hive_pipeline\hive sample preparation (politic-non-politic)\hive_sample_master.parquet"
OUTDIR = os.path.join("hive_outputs", "model1")
OUTPUT_FILE = os.path.join(OUTDIR, "stage2_claims.json")
QUERIES_CSV = os.path.join(OUTDIR, "stage2_queries.csv")
CKPT_FILE = os.path.join(OUTDIR, "_ckpt_select.json")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else OpenAI()
_DROP = set()
_lock = threading.Lock()


def call_gpt(messages, reasoning_effort=None, max_tokens=2000, json_mode=False):
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            kwargs = {"model": GPT_MODEL, "messages": messages,
                      "max_completion_tokens": max_tokens}
            if reasoning_effort and "reasoning_effort" not in _DROP:
                kwargs["reasoning_effort"] = reasoning_effort
            if json_mode and "response_format" not in _DROP:
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs)
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:               # noqa: BLE001
            msg = str(e).lower()
            dropped = False
            for arg in ("reasoning_effort", "response_format"):
                if arg in msg and arg not in _DROP:
                    _DROP.add(arg)
                    dropped = True
            if dropped:
                continue
            last_err = e
            time.sleep(REQUEST_PAUSE * (attempt + 1))
    print(f"  ! GPT call failed: {last_err}")
    return ""


def parse_json(text):
    if not text:
        return None
    t = re.sub(r'^```(?:json)?', '', text.strip())
    t = re.sub(r'```$', '', t.strip()).strip()
    try:
        return json.loads(t)
    except Exception:                        # noqa: BLE001
        m = re.search(r'\{.*\}', t, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:                # noqa: BLE001
                return None
    return None


# ===================== HIVE SAMPLE LOADER ==================================
def load_documents(path):
    df = pd.read_parquet(path)
    if "doc_id" not in df.columns:
        df["doc_id"] = df["author"].astype(str) + "/" + df["permlink"].astype(str)
    docs = []
    for r in df.itertuples(index=False):
        docs.append({
            "doc_id": getattr(r, "doc_id"),
            "source_type": "hive_post",
            "title": str(getattr(r, "title", "") or ""),
            "author": str(getattr(r, "author", "") or ""),
            "date": str(getattr(r, "created", "") or ""),
            "cb_max_score": getattr(r, "cb_max_score", None),
            "category": str(getattr(r, "category", "") or ""),
            "is_political": bool(getattr(r, "is_political", False)),
            "sample_stratum": str(getattr(r, "sample_stratum", "") or ""),
        })
    return docs


def load_ckpt(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_ckpt(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


# ===================== PROMPT (query from title) ===========================
QUERY_SYSTEM = (
    "You write one neutral web search query to help fact-check the factual "
    "content of a short post title. Respond with JSON only."
)


def query_from_title(title):
    usr = (
        "Post title:\n\n" + title + "\n\n"
        "Write ONE short Google query (about 4 to 8 words) that would help "
        "verify the factual content of this title. Name the key entities and the "
        "disputed fact. Keep it NEUTRAL: do not copy the title wholesale, do not "
        "add rhetorical or narrative phrasing, and do not presuppose whether the "
        "title is true. Do not use quotation marks. If the title contains no "
        "clearly checkable factual content, still produce the most reasonable "
        "query from whatever it mentions.\n\n"
        'Return JSON exactly as: {"query": "..."}.'
    )
    out = call_gpt([{"role": "system", "content": QUERY_SYSTEM},
                    {"role": "user", "content": usr}],
                   reasoning_effort=SELECT_REASONING,
                   max_tokens=SELECT_MAX_TOKENS, json_mode=True)
    data = parse_json(out) or {}
    query = (data.get("query") or "").strip().strip('"')
    return query


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Missing {INPUT_FILE}. Run the sampler first.")
        return
    documents = load_documents(INPUT_FILE)
    print(f"Loaded {len(documents)} sampled posts.")
    os.makedirs(OUTDIR, exist_ok=True)

    # Only posts that actually have a title can be title-checked.
    titled = [d for d in documents if d["title"].strip()]
    n_missing = len(documents) - len(titled)

    def work(d):
        q = query_from_title(d["title"])
        return q or d["title"]               # fall back to the raw title

    ckpt = load_ckpt(CKPT_FILE)
    todo = [d for d in titled if d["doc_id"] not in ckpt]
    print(f"  select: {len(titled)} titled posts, {len(ckpt)} cached, "
          f"{len(todo)} to do ({n_missing} had no title).")
    done = 0
    if todo:
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            futs = {ex.submit(work, d): d["doc_id"] for d in todo}
            for fut in as_completed(futs):
                k = futs[fut]
                try:
                    res = fut.result()
                except Exception as e:       # noqa: BLE001
                    res = {"_error": str(e)}
                with _lock:
                    ckpt[k] = res
                    done += 1
                    if done % SAVE_EVERY == 0:
                        save_ckpt(CKPT_FILE, ckpt)
                        print(f"    select: {done}/{len(todo)} done")
        save_ckpt(CKPT_FILE, ckpt)

    all_claims, flat, registry = [], [], []
    for doc in titled:
        query = ckpt.get(doc["doc_id"])
        if not isinstance(query, str) or not query:
            query = doc["title"]
        title = doc["title"]
        all_claims.append({
            "doc_id": doc["doc_id"], "source_type": doc["source_type"],
            "title": title, "author": doc["author"], "date": doc["date"],
            "claim": title, "claim_normalized": title, "query": query,
        })
        flat.append({"doc_id": doc["doc_id"], "claim": title, "query": query})
        registry.append({
            "doc_id": doc["doc_id"], "source_type": doc["source_type"],
            "title": title, "author": doc["author"], "date": doc["date"],
            "cb_max_score": doc["cb_max_score"], "category": doc["category"],
            "is_political": doc["is_political"], "sample_stratum": doc["sample_stratum"],
            "clean_text": title,             # title-only context for stage 4 stance
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"documents": registry, "claims": all_claims},
                  f, ensure_ascii=False, indent=2)
    pd.DataFrame(flat).to_csv(QUERIES_CSV, index=False, encoding="utf-8")
    print(f"\nModel 1 Stage 2 done. {len(all_claims)} title-claims "
          f"({n_missing} posts had no title).\nWrote {OUTPUT_FILE} and {QUERIES_CSV}")


if __name__ == "__main__":
    main()
