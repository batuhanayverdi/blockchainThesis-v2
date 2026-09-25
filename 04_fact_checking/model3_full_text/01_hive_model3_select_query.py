"""
HIVE MODEL 3 - Stage 2 of 4: claim selection + query generation (full body)
===========================================================================
Selects up to three check-worthy claims from each post's body_for_analysis and
writes one neutral search query per claim. Prompt and logic are identical to the
validated Model 3 Stage 2; only the input (hive_sample_master.parquet) and the
threaded-with-resume execution differ.

Reads:  hive_sample_master.parquet
Writes: hive_outputs/model3/stage2_claims.json   ({"documents": [...], "claims": [...]})
        hive_outputs/model3/stage2_queries.csv
        hive_outputs/model3/_ckpt_select.json     (resume state)

Run:    python hive_model3_select_query.py
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
SELECT_MAX_TOKENS = 6000
MAX_CLAIMS_PER_POST = 3
N_WORKERS = 8
SAVE_EVERY = 50
MAX_RETRIES = 3
REQUEST_PAUSE = 0.4

INPUT_FILE = r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_hive_pipeline\hive sample preparation (politic-non-politic)\hive_sample_master.parquet"
OUTDIR = os.path.join("hive_outputs", "model3")
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
            "body_for_analysis": str(getattr(r, "body_for_analysis", "") or ""),
            "cb_max_score": getattr(r, "cb_max_score", None),
            "category": str(getattr(r, "category", "") or ""),
            "is_political": bool(getattr(r, "is_political", False)),
            "sample_stratum": str(getattr(r, "sample_stratum", "") or ""),
        })
    return docs


# ===================== RESUME (threaded checkpoint) ========================
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


def run_threaded(items, key_fn, work_fn, ckpt_path, label):
    ckpt = load_ckpt(ckpt_path)
    todo = [it for it in items if key_fn(it) not in ckpt]
    print(f"  {label}: {len(items)} total, {len(ckpt)} cached, {len(todo)} to do.")
    done = 0
    if todo:
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            futs = {ex.submit(work_fn, it): key_fn(it) for it in todo}
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
                        save_ckpt(ckpt_path, ckpt)
                        print(f"    {label}: {done}/{len(todo)} done")
        save_ckpt(ckpt_path, ckpt)
    return ckpt


# ===================== PROMPT (claim selection + neutral query) ============
SELECT_SYSTEM = (
    "You are a fact-checking assistant. You find the most check-worthy factual "
    "claims in a document and write one neutral search query to verify each. A "
    "check-worthy claim is a specific, verifiable statement of fact (who, what, "
    "when, where, how many). Ignore opinions, questions, predictions, jokes, and "
    "calls to action. Respond with JSON only."
)

SELECT_EXAMPLES = (
    'Examples of the desired output:\n'
    '{"claims": [\n'
    '  {"claim": "Actually, Nasa uses orange filters on Mars Bash.",\n'
    '   "claim_normalized": "NASA uses orange filters on images of Mars.",\n'
    '   "query": "NASA Mars image color filters"},\n'
    '  {"claim": "And they have a special Mountain Dew at Applebee\'s.",\n'
    '   "claim_normalized": "Applebee\'s serves an exclusive Mountain Dew.",\n'
    '   "query": "Applebee\'s exclusive Mountain Dew"}\n'
    ']}'
)


def select_and_query(clean_text):
    usr = (
        "Document:\n\n" + clean_text + "\n\n"
        "Select at most THREE of the most check-worthy factual claims, ordered "
        "from most to least check-worthy. For each claim return three fields:\n"
        '  - "claim": the sentence copied verbatim from the document.\n'
        '  - "claim_normalized": the same claim rewritten as ONE clean, '
        "self-contained declarative sentence. Fix obvious transcription errors "
        "and remove filler, but do NOT add any facts that are not in the "
        "original sentence.\n"
        '  - "query": a short Google query (about 4 to 8 words) built from the '
        "normalized claim. Name the key entities and the disputed fact. Keep it "
        "NEUTRAL: do not copy the claim's wording, do not add narrative or "
        "rhetorical phrasing, and do not presuppose whether the claim is true. "
        "Do not use quotation marks in the query.\n\n"
        + SELECT_EXAMPLES + "\n\n"
        'Return JSON exactly as: {"claims": [{"claim": "...", '
        '"claim_normalized": "...", "query": "..."}]}. '
        'If there are no check-worthy factual claims, return {"claims": []}.'
    )
    out = call_gpt([{"role": "system", "content": SELECT_SYSTEM},
                    {"role": "user", "content": usr}],
                   reasoning_effort=SELECT_REASONING,
                   max_tokens=SELECT_MAX_TOKENS, json_mode=True)
    data = parse_json(out)
    if not data or "claims" not in data:
        return []
    claims = []
    for c in data["claims"]:
        claim = (c.get("claim") or "").strip()
        normalized = (c.get("claim_normalized") or "").strip()
        query = (c.get("query") or "").strip()
        if not normalized:
            normalized = claim
        if claim and query:
            claims.append({"claim": claim, "claim_normalized": normalized,
                           "query": query})
    return claims[:MAX_CLAIMS_PER_POST]


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Missing {INPUT_FILE}. Run the sampler first.")
        return
    documents = load_documents(INPUT_FILE)
    print(f"Loaded {len(documents)} sampled posts.")
    os.makedirs(OUTDIR, exist_ok=True)

    ckpt = run_threaded(
        documents,
        key_fn=lambda d: d["doc_id"],
        work_fn=lambda d: select_and_query(d["body_for_analysis"]),
        ckpt_path=CKPT_FILE, label="select",
    )

    all_claims, flat, registry = [], [], []
    for doc in documents:
        selected = ckpt.get(doc["doc_id"], [])
        if isinstance(selected, dict):       # error sentinel
            selected = []
        for c in selected:
            all_claims.append({
                "doc_id": doc["doc_id"], "source_type": doc["source_type"],
                "title": doc["title"], "author": doc["author"], "date": doc["date"],
                "claim": c["claim"], "claim_normalized": c["claim_normalized"],
                "query": c["query"],
            })
            flat.append({"doc_id": doc["doc_id"], "claim": c["claim"],
                         "claim_normalized": c["claim_normalized"],
                         "query": c["query"]})
        registry.append({
            "doc_id": doc["doc_id"], "source_type": doc["source_type"],
            "title": doc["title"], "author": doc["author"], "date": doc["date"],
            "cb_max_score": doc["cb_max_score"], "category": doc["category"],
            "is_political": doc["is_political"], "sample_stratum": doc["sample_stratum"],
            "clean_text": doc["body_for_analysis"],
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"documents": registry, "claims": all_claims},
                  f, ensure_ascii=False, indent=2)
    pd.DataFrame(flat).to_csv(QUERIES_CSV, index=False, encoding="utf-8")
    print(f"\nModel 3 Stage 2 done. {len(all_claims)} claims across "
          f"{len(documents)} posts.\nWrote {OUTPUT_FILE} and {QUERIES_CSV}")


if __name__ == "__main__":
    main()
