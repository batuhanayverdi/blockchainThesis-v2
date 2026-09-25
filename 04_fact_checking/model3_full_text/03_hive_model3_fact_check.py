"""
HIVE MODEL 3 - Stage 4 of 4: 6-class verdict + stance + ordinal roll-up
=======================================================================
Verdict prompt, stance question, and ordinal post roll-up are identical to the
validated Model 3 Stage 4. NO binary collapse here; you collapse at the
regression/eval step:
    false_side = {pants_on_fire, false, mostly_false}  -> misinformation (1)
    true_side  = {mostly_true, true}                   -> not (0)
    abstain    = {unverified, no_checkworthy_claim}     -> excluded

Reads:  hive_outputs/model3/stage3_search.json
Writes: hive_outputs/model3/stage4_factcheck.json
        hive_outputs/model3/claims.csv
        hive_outputs/model3/posts.csv
        hive_outputs/model3/_ckpt_verdict.json   (resume state)

Run:    python hive_model3_fact_check.py
Deps:   pip install openai pandas
Key:    OPENAI_API_KEY  (read from the environment, never hard-coded)
"""

import os
import re
import json
import time
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI


# ============================ CONFIG ========================================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GPT_MODEL = "gpt-5-mini"
VERDICT_REASONING = "medium"
VERDICT_MAX_TOKENS = 4000
N_WORKERS = 8
SAVE_EVERY = 50
MAX_RETRIES = 3
REQUEST_PAUSE = 0.4

CONTEXT_RADIUS = 1200
REQUIRE_ASSERTED_FOR_MISINFO = True

OUTDIR = os.path.join("hive_outputs", "model3")
INPUT_FILE = os.path.join(OUTDIR, "stage3_search.json")
OUTPUT_JSON = os.path.join(OUTDIR, "stage4_factcheck.json")
CLAIMS_CSV = os.path.join(OUTDIR, "claims.csv")
POSTS_CSV = os.path.join(OUTDIR, "posts.csv")
CKPT_FILE = os.path.join(OUTDIR, "_ckpt_verdict.json")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else OpenAI()
_DROP = set()
_lock = threading.Lock()

ACC_LEVEL = {"pants_on_fire": 1, "false": 2, "mostly_false": 3,
             "mostly_true": 4, "true": 5}
LEVEL_LABEL = {1: "pants_on_fire", 2: "false", 3: "mostly_false",
               4: "mostly_true", 5: "true"}


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


_PUNCT = {"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
          "\u2014": "-", "\u2013": "-", "\u00a0": " "}


def _normalize_with_map(s):
    """Collapse runs of whitespace and normalize quotes/dashes. Returns
    (normalized_string, idx_map) where idx_map[k] is the ORIGINAL index of
    normalized character k, so a match offset maps back to the source exactly."""
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
            # whitespace/punctuation-tolerant fallback, exact offset mapping
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


def normalize_verdict6(raw):
    v = (raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "true": "true",
        "mostly_true": "mostly_true", "mostlytrue": "mostly_true",
        "mostly_false": "mostly_false", "mostlyfalse": "mostly_false",
        "false": "false",
        "pants_on_fire": "pants_on_fire", "pantsonfire": "pants_on_fire",
        "pants": "pants_on_fire",
        "unverified": "unverified", "unverifiable": "unverified",
    }
    return mapping.get(v, "unverified")


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
    out = call_gpt([{"role": "system", "content": sys},
                    {"role": "user", "content": usr}],
                   reasoning_effort=VERDICT_REASONING,
                   max_tokens=VERDICT_MAX_TOKENS, json_mode=True)
    data = parse_json(out) or {}
    return {"verdict": normalize_verdict6(data.get("verdict")),
            "stance": normalize_stance(data.get("stance")),
            "justification": (data.get("justification") or "").strip(),
            "verdict_raw": out}


def aggregate(results):
    """results: list of (label6, stance). Returns 6-class label + ordinal."""
    if not results:
        return {"final_label": "no_checkworthy_claim", "accuracy_ordinal": "",
                "verdict_counts": {}}
    counts = Counter(l for l, _ in results)
    counting = []
    for label6, stance in results:
        lvl = ACC_LEVEL.get(label6)
        if lvl is None:                       # unverified -> abstain at claim level
            continue
        if lvl in (1, 2, 3):                  # false-side: stance-gated
            if stance == "asserted" or not REQUIRE_ASSERTED_FOR_MISINFO:
                counting.append(lvl)
        else:                                 # true-side: counts regardless of stance
            counting.append(lvl)
    if counting:
        lvl = min(counting)
        return {"final_label": LEVEL_LABEL[lvl], "accuracy_ordinal": lvl,
                "verdict_counts": dict(counts)}
    return {"final_label": "unverified", "accuracy_ordinal": "",
            "verdict_counts": dict(counts)}


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


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Missing {INPUT_FILE}. Run hive_model3_search.py first.")
        return
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    documents, claims = data["documents"], data["claims"]
    print(f"Loaded {len(claims)} claims from Model 3 Stage 3.")

    clean_by_doc = {d["doc_id"]: d.get("clean_text", "") for d in documents}

    def verdict_one(i):
        c = claims[i]
        ctx = context_window(clean_by_doc.get(c["doc_id"], ""), c.get("claim", ""))
        return get_verdict(c["claim"], c["author"], c["date"],
                           c.get("formatted_results", ""), ctx)

    ckpt = load_ckpt(CKPT_FILE)
    todo = [i for i in range(len(claims)) if str(i) not in ckpt]
    print(f"  verdict: {len(claims)} total, {len(ckpt)} cached, {len(todo)} to do.")
    done = 0
    if todo:
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            futs = {ex.submit(verdict_one, i): i for i in todo}
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    res = fut.result()
                except Exception as e:       # noqa: BLE001
                    res = {"verdict": "unverified", "stance": "unclear",
                           "justification": "", "verdict_raw": "", "_error": str(e)}
                with _lock:
                    ckpt[str(i)] = res
                    done += 1
                    if done % SAVE_EVERY == 0:
                        save_ckpt(CKPT_FILE, ckpt)
                        print(f"    verdict: {done}/{len(todo)} done")
        save_ckpt(CKPT_FILE, ckpt)

    results_by_doc = defaultdict(list)
    for i, c in enumerate(claims):
        r = ckpt.get(str(i), {})
        c["verdict"] = r.get("verdict", "unverified")
        c["stance"] = r.get("stance", "unclear")
        c["justification"] = r.get("justification", "")
        c["verdict_raw"] = r.get("verdict_raw", "")
        results_by_doc[c["doc_id"]].append((c["verdict"], c["stance"]))

    post_rows = []
    for d in documents:
        agg = aggregate(results_by_doc.get(d["doc_id"], []))
        row = {k: v for k, v in d.items() if k != "clean_text"}
        post_rows.append({**row, **agg})

    os.makedirs(OUTDIR, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"documents": documents, "claims": claims, "posts": post_rows},
                  f, ensure_ascii=False, indent=2)
    pd.DataFrame([{k: v for k, v in c.items()
                   if k not in ("results", "formatted_results")}
                  for c in claims]).to_csv(CLAIMS_CSV, index=False, encoding="utf-8")
    posts_df = pd.DataFrame(post_rows)
    posts_df.to_csv(POSTS_CSV, index=False, encoding="utf-8")

    print(f"\nModel 3 Stage 4 done. Wrote {OUTPUT_JSON}, {CLAIMS_CSV}, {POSTS_CSV}")
    if not posts_df.empty:
        print("\nFinal labels (6-class):")
        print(posts_df["final_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
