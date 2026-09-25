"""
MODEL 3 - Stage 2 of 4: claim selection + query generation
==========================================================
Identical claim-selection logic to your 2_select_query.py (same prompt, same
examples, same up-to-three claims per document), repointed to a model3/ folder.
Keeping the prompt identical to Model 2 means Models 2 and 3 select the SAME
claims, so any later difference between them reflects only the verdict step.

Reads:  pilot_outputs/stage1_passed.json
Writes: pilot_outputs/model3/stage2_claims.json   ({"documents": [...], "claims": [...]})
        pilot_outputs/model3/stage2_queries.csv

NOTE: Because this is identical to Model 2's selection, which you have already
run, you can skip this script (and model3_search.py) and instead point
model3_fact_check.py at Model 2's existing pilot_outputs/stage3_search.json. That
reuses Model 2's claims and evidence exactly and saves a full re-run. Running
these two scripts is only needed if you want Model 3 fully self-contained.

Run:    python model3_select_query.py
Deps:   pip install openai pandas
Key:    OPENAI_API_KEY   (read from the environment, not hard-coded)
"""

import os
import re
import json
import time

import pandas as pd
from openai import OpenAI


# ============================ CONFIG ========================================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")   # set this in your shell
GPT_MODEL = "gpt-5-mini"
SELECT_REASONING = "low"
SELECT_MAX_TOKENS = 6000
MAX_CLAIMS_PER_POST = 3
REQUEST_PAUSE = 0.4
MAX_RETRIES = 3

INPUT_FILE = r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_pipeline\pilot_outputs\stage1_passed.json"

OUTDIR = os.path.join("pilot_outputs", "model3")
OUTPUT_FILE = os.path.join(OUTDIR, "stage2_claims.json")
QUERIES_CSV = os.path.join(OUTDIR, "stage2_queries.csv")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else OpenAI()
_DROP = set()


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
                    print(f"  ({arg} not accepted here; continuing without it)")
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
        print(f"Missing {INPUT_FILE}. Run the ClaimBuster gate (stage 1) first.")
        return
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        documents = json.load(f)["documents"]
    print(f"Loaded {len(documents)} passed documents from stage 1.")

    os.makedirs(OUTDIR, exist_ok=True)
    all_claims, flat = [], []
    for doc in documents:
        selected = select_and_query(doc["clean_text"])
        time.sleep(REQUEST_PAUSE)
        print(f"  {doc['doc_id']}: GPT selected {len(selected)} claim(s)")
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

    registry = [{"doc_id": d["doc_id"], "source_type": d["source_type"],
                 "title": d["title"], "author": d["author"], "date": d["date"],
                 "cb_max_score": d.get("cb_max_score"),
                 "clean_text": d.get("clean_text", "")} for d in documents]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"documents": registry, "claims": all_claims},
                  f, ensure_ascii=False, indent=2)
    pd.DataFrame(flat).to_csv(QUERIES_CSV, index=False, encoding="utf-8")
    print(f"\nModel 3 Stage 2 done. {len(all_claims)} claims across "
          f"{len(documents)} docs.\nWrote {OUTPUT_FILE} and {QUERIES_CSV}")


if __name__ == "__main__":
    main()
