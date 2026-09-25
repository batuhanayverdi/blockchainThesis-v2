"""
HIVE MODEL 3 - Stage 4 via Batch API: STEP 1 of 2 (submit)
==========================================================
Builds a Batch request file whose every request body is byte-identical to the
synchronous hive_model3_fact_check.py: same model (gpt-5-mini), same get_verdict
system + user message (context window + formatted results + claim), same
reasoning_effort, max_completion_tokens, and response_format. It then uploads the
file and creates the batch, and records the batch id.

It only submits claims that do NOT already have a verdict in the shared
checkpoint (_ckpt_verdict.json, the same file the synchronous script uses), so:
  - first run submits every claim,
  - after a partial collect it submits only the ones still missing,
  - claims you already verdicted synchronously are skipped.

Run STEP 1:  python hive_model3_fact_check_batch_submit.py
Then later:  python hive_model3_fact_check_batch_collect.py
Deps:        pip install openai
Key:         OPENAI_API_KEY  (read from the environment, never hard-coded)
"""

import os
import json
import time

from openai import OpenAI


# ============================ CONFIG ========================================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GPT_MODEL = "gpt-5-mini"
VERDICT_REASONING = "medium"      # identical to the synchronous script
VERDICT_MAX_TOKENS = 4000         # identical to the synchronous script
CONTEXT_RADIUS = 1200             # identical to the synchronous script
COMPLETION_WINDOW = "24h"

# If a batch is already in flight, this script refuses so you do not stack two.
# Set True only to abandon the recorded batch and submit a fresh one.
FORCE_RESUBMIT = False

OUTDIR = os.path.join("hive_outputs", "model3")
INPUT_FILE = os.path.join(OUTDIR, "stage3_search.json")
CKPT_FILE = os.path.join(OUTDIR, "_ckpt_verdict.json")     # shared with sync script
STATE_FILE = os.path.join(OUTDIR, "_batch_state.json")
BATCH_INPUT = os.path.join(OUTDIR, "_batch_input.jsonl")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else OpenAI()


# =============== prompt + context: copied verbatim from the sync script =====
_PUNCT = {"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
          "\u2014": "-", "\u2013": "-", "\u00a0": " "}


def _normalize_with_map(s):
    out, idx_map, prev_space = [], [], False
    for i, ch in enumerate(s):
        ch = _PUNCT.get(ch, ch)
        if ch.isspace():
            if prev_space:
                continue
            out.append(" "); idx_map.append(i); prev_space = True
        else:
            out.append(ch); idx_map.append(i); prev_space = False
    return "".join(out), idx_map


def context_window(text, claim, radius=CONTEXT_RADIUS):
    if not text:
        return ""
    if claim:
        idx = text.find(claim)
        if idx == -1:
            idx = text.find(claim[:60])
        if idx == -1:
            norm_text, idx_map = _normalize_with_map(text)
            key = _normalize_with_map(claim)[0].strip()
            j = norm_text.find(key) if key else -1
            if j == -1 and len(key) > 80:
                j = norm_text.find(key[:80])
            if j != -1:
                idx = idx_map[j]
        if idx != -1:
            start = max(0, idx - radius)
            end = min(len(text), idx + len(claim) + radius)
            return text[start:end]
    return text[: 2 * radius]


def build_verdict_messages(claim, author, date, formatted_results, context):
    """The exact [system, user] messages the synchronous get_verdict sends."""
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
        "1) verdict: Choose EXACTLY ONE label for how accurate the claim is, "
        "based on the results:\n"
        "   - 'true': accurate, with nothing significant missing.\n"
        "   - 'mostly_true': accurate but needs clarification or additional "
        "information.\n"
        "   - 'mostly_false': contains an element of truth but ignores critical "
        "facts that would give a different impression.\n"
        "   - 'false': not accurate.\n"
        "   - 'pants_on_fire': not accurate AND makes an absurd or egregious "
        "claim with no basis in fact.\n"
        "   - 'unverified': the evidence is insufficient or mixed to determine "
        "accuracy.\n"
        "   Rely on reliable, independent sources that support or refute the "
        "proposition. Some results may be labelled [Featured snippet] or "
        "[Knowledge panel]; these are summaries Google surfaces prominently, not "
        "verified verdicts, so judge them by their underlying source like any "
        "other result and give them no extra weight for being prominent. A "
        "[Featured snippet] often repeats one of the organic results from the "
        "same domain; when it does, count that source once, not twice. If "
        "several results merely repeat the claim without independent support, "
        "choose 'unverified' rather than 'true'. Do not defer to a source just "
        "because it sounds authoritative or recent.\n"
        "2) stance: In the surrounding text, does the source present this claim "
        "as its OWN factual assertion ('asserted'), or does it quote, attribute, "
        "or question the claim without endorsing it ('quoted_or_questioned')? If "
        "unclear, choose 'unclear'.\n\n"
        'Respond with JSON only: {"verdict": "true | mostly_true | mostly_false '
        '| false | pants_on_fire | unverified", "stance": "asserted | '
        'quoted_or_questioned | unclear", "justification": "<concise '
        'justification citing the domains of the relevant results>"}.'
    )
    return [{"role": "system", "content": sys},
            {"role": "user", "content": usr}]


def build_body(messages):
    """The exact request body the synchronous call_gpt sends for the verdict."""
    return {"model": GPT_MODEL, "messages": messages,
            "max_completion_tokens": VERDICT_MAX_TOKENS,
            "reasoning_effort": VERDICT_REASONING,
            "response_format": {"type": "json_object"}}


def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def main():
    if not OPENAI_API_KEY:
        print("OPENAI_API_KEY is not set.")
        return
    if not os.path.exists(INPUT_FILE):
        print(f"Missing {INPUT_FILE}. Run hive_model3_search.py first.")
        return

    # refuse to stack a second batch on top of an unconsumed one
    if os.path.exists(STATE_FILE) and not FORCE_RESUBMIT:
        st = load_json(STATE_FILE, {})
        print(f"A batch is already recorded (id {st.get('batch_id')}). Run "
              f"hive_model3_fact_check_batch_collect.py first to consume it, or "
              f"set FORCE_RESUBMIT = True to abandon it and submit a new one.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    documents, claims = data["documents"], data["claims"]
    clean_by_doc = {d["doc_id"]: d.get("clean_text", "") for d in documents}

    ckpt = load_json(CKPT_FILE, {})
    todo = [i for i in range(len(claims)) if str(i) not in ckpt]
    print(f"{len(claims)} claims | {len(ckpt)} already have a verdict | "
          f"{len(todo)} to submit.")
    if not todo:
        print("Nothing to submit. Run the collect script to aggregate and write "
              "the outputs.")
        return

    os.makedirs(OUTDIR, exist_ok=True)
    with open(BATCH_INPUT, "w", encoding="utf-8") as f:
        for i in todo:
            c = claims[i]
            ctx = context_window(clean_by_doc.get(c["doc_id"], ""), c.get("claim", ""))
            body = build_body(build_verdict_messages(
                c["claim"], c["author"], c["date"],
                c.get("formatted_results", ""), ctx))
            f.write(json.dumps({"custom_id": f"claim-{i}", "method": "POST",
                                "url": "/v1/chat/completions", "body": body},
                               ensure_ascii=False) + "\n")
    print(f"Wrote {len(todo)} requests to {BATCH_INPUT}")

    up = client.files.create(file=open(BATCH_INPUT, "rb"), purpose="batch")
    batch = client.batches.create(input_file_id=up.id,
                                  endpoint="/v1/chat/completions",
                                  completion_window=COMPLETION_WINDOW,
                                  metadata={"description": "hive model3 verdicts"})
    state = {"batch_id": batch.id, "input_file_id": up.id,
             "n_submitted": len(todo), "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    print(f"\nSubmitted batch {batch.id} ({len(todo)} requests, window "
          f"{COMPLETION_WINDOW}).")
    print(f"State saved to {STATE_FILE}.")
    print("Come back later and run hive_model3_fact_check_batch_collect.py to "
          "check status and, once finished, write the outputs.")


if __name__ == "__main__":
    main()
