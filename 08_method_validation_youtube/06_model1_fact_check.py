"""
MODEL 1 - Stage 4 of 4: verdict (True / False / Unverified) + stance + roll-up
==============================================================================
Same verdict logic and post roll-up as your 4_factcheck.py. The only difference
is the input: Model 1's claim is the title, and the post context is also the
title (set in Model 1 Stage 2), so stance will almost always come back
'asserted'. That collapse is the expected title-only limitation.

Post roll-up (unchanged from your pipeline):
  - any claim FALSE and asserted        -> misinformation
  - at least one TRUE, none of the above -> true
  - only unverified / quoted-false      -> unverified
  - no claim                            -> no_checkworthy_claim

The output keeps the same schema as your Model 2 posts.csv (final_label +
verdict_counts), so you score Model 1 the same way: map final_label to a binary
at evaluation (misinformation -> 1, true -> 0, unverified / no_checkworthy ->
abstain).

Reads:  pilot_outputs/model1/stage3_search.json
Writes: pilot_outputs/model1/stage4_factcheck.json
        pilot_outputs/model1/claims.csv
        pilot_outputs/model1/posts.csv

Run:    python model1_fact_check.py
Deps:   pip install openai pandas
Key:    OPENAI_API_KEY   (read from the environment, not hard-coded)
"""

import os
import re
import json
import time
from collections import Counter, defaultdict

import pandas as pd
from openai import OpenAI


# ============================ CONFIG ========================================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")   # set this in your shell
GPT_MODEL = "gpt-5-mini"
VERDICT_REASONING = "medium"
VERDICT_MAX_TOKENS = 4000
REQUEST_PAUSE = 0.4
MAX_RETRIES = 3

CONTEXT_RADIUS = 1200
REQUIRE_ASSERTED_FOR_MISINFO = True

OUTDIR = os.path.join("pilot_outputs", "model1")
INPUT_FILE = os.path.join(OUTDIR, "stage3_search.json")
OUTPUT_JSON = os.path.join(OUTDIR, "stage4_factcheck.json")
CLAIMS_CSV = os.path.join(OUTDIR, "claims.csv")
POSTS_CSV = os.path.join(OUTDIR, "posts.csv")

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


# ============================ CONTEXT ======================================
def context_window(text, claim, radius=CONTEXT_RADIUS):
    if not text:
        return ""
    if claim:
        idx = text.find(claim)
        if idx == -1:
            idx = text.find(claim[:60])
        if idx != -1:
            start = max(0, idx - radius)
            end = min(len(text), idx + len(claim) + radius)
            return text[start:end]
    return text[: 2 * radius]


# ============================ VERDICT =====================================
def normalize_verdict(raw):
    v = (raw or "").strip().lower()
    if v in ("true", "mostly true", "accurate", "supported"):
        return "true"
    if v in ("false", "mostly false", "incorrect", "misinformation", "refuted"):
        return "false"
    return "unverified"


def normalize_stance(raw):
    s = (raw or "").strip().lower()
    if s in ("asserted", "assert", "endorsed", "stated"):
        return "asserted"
    if s in ("quoted_or_questioned", "quoted", "questioned", "attributed",
             "quoted or questioned"):
        return "quoted_or_questioned"
    return "unclear"


def get_verdict(claim, author, date, formatted_results, context):
    sys = ("You are a careful fact-checking expert. You judge claims using only "
           "the provided search results, and you weigh the reliability and "
           "independence of sources rather than how confident or recent they "
           "sound.")
    usr = (
        "Judge the following claim.\n\n"
        f"Claim: '{claim}' (stated by {author} on {date}).\n\n"
        "Surrounding text from the source (for context only, not evidence):\n"
        f"{context}\n\n"
        f"Google results:\n{formatted_results}\n\n"
        "Return two judgments:\n"
        "1) verdict: Is the claim True, False, or Unverified based on the "
        "results? Rely on reliable, independent sources that support or refute "
        "the proposition. Some results may be labelled [Featured snippet] or "
        "[Knowledge panel]; these are summaries Google surfaces prominently, not "
        "verified verdicts, so judge them by their underlying source like any "
        "other result and give them no extra weight for being prominent. A "
        "[Featured snippet] often repeats one of the organic results from the "
        "same domain; when it does, count that source once, not twice. If "
        "several results merely repeat the claim without independent support, "
        "treat that as Unverified rather than True. Do not defer to a source "
        "just because it sounds authoritative or recent. If the evidence is "
        "insufficient or mixed, choose Unverified.\n"
        "2) stance: In the surrounding text, does the source present this claim "
        "as its OWN factual assertion ('asserted'), or does it quote, attribute, "
        "or question the claim without endorsing it ('quoted_or_questioned')? If "
        "unclear, choose 'unclear'.\n\n"
        'Respond with JSON only: {"verdict": "True | False | Unverified", '
        '"stance": "asserted | quoted_or_questioned | unclear", '
        '"justification": "<concise justification citing the domains of the '
        'relevant results>"}.'
    )
    out = call_gpt([{"role": "system", "content": sys},
                    {"role": "user", "content": usr}],
                   reasoning_effort=VERDICT_REASONING,
                   max_tokens=VERDICT_MAX_TOKENS, json_mode=True)
    data = parse_json(out) or {}
    return (normalize_verdict(data.get("verdict")),
            normalize_stance(data.get("stance")),
            (data.get("justification") or "").strip(), out)


# ========================= AGGREGATION ====================================
def aggregate(results):
    if not results:
        return {"final_label": "no_checkworthy_claim", "verdict_counts": {}}
    counts = Counter(v for v, _ in results)
    misinfo = any(
        v == "false" and (s == "asserted" or not REQUIRE_ASSERTED_FOR_MISINFO)
        for v, s in results
    )
    if misinfo:
        label = "misinformation"
    elif counts.get("true", 0) > 0:
        label = "true"
    else:
        label = "unverified"
    return {"final_label": label, "verdict_counts": dict(counts)}


# ============================== MAIN ======================================
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Missing {INPUT_FILE}. Run model1_search.py first.")
        return
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    documents, claims = data["documents"], data["claims"]
    print(f"Loaded {len(claims)} claims from Model 1 Stage 3.")

    clean_by_doc = {d["doc_id"]: d.get("clean_text", "") for d in documents}

    results_by_doc = defaultdict(list)
    for n, c in enumerate(claims, start=1):
        ctx = context_window(clean_by_doc.get(c["doc_id"], ""), c.get("claim", ""))
        verdict, stance, justification, raw = get_verdict(
            c["claim"], c["author"], c["date"],
            c.get("formatted_results", ""), ctx)
        time.sleep(REQUEST_PAUSE)
        c["verdict"] = verdict
        c["stance"] = stance
        c["justification"] = justification
        c["verdict_raw"] = raw
        results_by_doc[c["doc_id"]].append((verdict, stance))
        print(f"  [{n}/{len(claims)}] {c['doc_id']} -> {verdict} ({stance})")

    post_rows = []
    for d in documents:
        agg = aggregate(results_by_doc.get(d["doc_id"], []))
        row = {k: v for k, v in d.items() if k != "clean_text"}
        post_rows.append({**row, **agg})

    os.makedirs(OUTDIR, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"documents": documents, "claims": claims, "posts": post_rows},
                  f, ensure_ascii=False, indent=2)

    claims_df = pd.DataFrame([{k: v for k, v in c.items()
                               if k not in ("results", "formatted_results")}
                              for c in claims])
    posts_df = pd.DataFrame(post_rows)
    claims_df.to_csv(CLAIMS_CSV, index=False, encoding="utf-8")
    posts_df.to_csv(POSTS_CSV, index=False, encoding="utf-8")

    print(f"\nModel 1 Stage 4 done. Wrote {OUTPUT_JSON}, {CLAIMS_CSV}, {POSTS_CSV}")
    if not posts_df.empty:
        print("\nFinal labels:")
        print(posts_df["final_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
