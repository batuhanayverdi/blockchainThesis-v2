"""
HIVE MODEL 3 - Stage 4 via Batch API: STEP 2 of 2 (collect + aggregate)
=======================================================================
Checks the submitted batch. While it is still running, prints status and exits.
Once finished, it downloads the results, parses each one through the SAME
normalization as the synchronous script (normalize_verdict6 / normalize_stance),
merges successes into the shared checkpoint (_ckpt_verdict.json), then runs the
SAME ordinal, stance-gated aggregation and writes the SAME outputs:
    stage4_factcheck.json, claims.csv, posts.csv

Failed or empty lines are reported and simply left uncached, so re-running the
submit script sends just those, and collecting again fills them in. The outputs
are always (re)written from the current checkpoint; a loud notice prints if any
claim is still without a verdict, so you know the labels are partial.

Run STEP 2:  python hive_model3_fact_check_batch_collect.py
Deps:        pip install openai pandas
Key:         OPENAI_API_KEY  (read from the environment, never hard-coded)
"""

import os
import re
import json
from collections import Counter, defaultdict

import pandas as pd
from openai import OpenAI


# ============================ CONFIG ========================================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
REQUIRE_ASSERTED_FOR_MISINFO = True     # identical to the synchronous script

OUTDIR = os.path.join("hive_outputs", "model3")
INPUT_FILE = os.path.join(OUTDIR, "stage3_search.json")
CKPT_FILE = os.path.join(OUTDIR, "_ckpt_verdict.json")     # shared with sync script
STATE_FILE = os.path.join(OUTDIR, "_batch_state.json")
OUTPUT_JSON = os.path.join(OUTDIR, "stage4_factcheck.json")
CLAIMS_CSV = os.path.join(OUTDIR, "claims.csv")
POSTS_CSV = os.path.join(OUTDIR, "posts.csv")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else OpenAI()

ACC_LEVEL = {"pants_on_fire": 1, "false": 2, "mostly_false": 3,
             "mostly_true": 4, "true": 5}
LEVEL_LABEL = {1: "pants_on_fire", 2: "false", 3: "mostly_false",
               4: "mostly_true", 5: "true"}

TERMINAL = {"completed", "failed", "expired", "cancelled"}


# =============== parsing + aggregation: copied verbatim from sync ===========
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


# =============== batch result parsing =======================================
def parse_result_content(content):
    """Same post-call normalization as the synchronous get_verdict."""
    data = parse_json(content) or {}
    return {"verdict": normalize_verdict6(data.get("verdict")),
            "stance": normalize_stance(data.get("stance")),
            "justification": (data.get("justification") or "").strip(),
            "verdict_raw": content}


def _download_text(file_id):
    r = client.files.content(file_id)
    if hasattr(r, "text"):
        return r.text
    if hasattr(r, "read"):
        data = r.read()
    elif hasattr(r, "content"):
        data = r.content
    else:
        return str(r)
    return data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else data


def merge_output_into_ckpt(output_text, ckpt):
    filled = failed = 0
    for line in output_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:                    # noqa: BLE001
            failed += 1
            continue
        cid = str(obj.get("custom_id", ""))
        idx = cid.split("-")[-1]
        resp = obj.get("response") or {}
        if obj.get("error") or resp.get("status_code") != 200:
            failed += 1
            continue
        body = resp.get("body") or {}
        choices = body.get("choices") or [{}]
        content = ((choices[0].get("message") or {}).get("content") or "")
        if not content.strip():
            failed += 1
            continue
        ckpt[idx] = parse_result_content(content)
        filled += 1
    return filled, failed


def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def write_outputs(documents, claims, ckpt):
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
    return posts_df


def finish(documents, claims, ckpt):
    posts_df = write_outputs(documents, claims, ckpt)
    missing = sum(1 for i in range(len(claims)) if str(i) not in ckpt)
    print(f"\nWrote {OUTPUT_JSON}, {CLAIMS_CSV}, {POSTS_CSV}")
    if not posts_df.empty:
        print("\nFinal labels (6-class):")
        print(posts_df["final_label"].value_counts().to_string())
    if missing:
        print(f"\nNOTICE: {missing} claim(s) still have no verdict, so these "
              f"outputs are PARTIAL. Run the submit script again (it will send "
              f"only the missing ones), then run this collect script again.")
    else:
        print(f"\nAll {len(claims)} claims have a verdict. Outputs are complete.")


def main():
    if not OPENAI_API_KEY:
        print("OPENAI_API_KEY is not set.")
        return
    if not os.path.exists(INPUT_FILE):
        print(f"Missing {INPUT_FILE}. Run hive_model3_search.py first.")
        return
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    documents, claims = data["documents"], data["claims"]
    ckpt = load_json(CKPT_FILE, {})

    # No active batch: aggregate if already complete, else point to submit.
    if not os.path.exists(STATE_FILE):
        if all(str(i) in ckpt for i in range(len(claims))) and claims:
            print("No active batch; all claims already have verdicts. Writing "
                  "outputs from the checkpoint.")
            finish(documents, claims, ckpt)
        else:
            print("No active batch recorded. Run "
                  "hive_model3_fact_check_batch_submit.py first.")
        return

    state = load_json(STATE_FILE, {})
    batch_id = state.get("batch_id")
    batch = client.batches.retrieve(batch_id)
    status = getattr(batch, "status", "unknown")
    counts = getattr(batch, "request_counts", None)
    cinfo = ""
    if counts is not None:
        cinfo = (f"  (total {getattr(counts, 'total', '?')}, completed "
                 f"{getattr(counts, 'completed', '?')}, failed "
                 f"{getattr(counts, 'failed', '?')})")
    print(f"Batch {batch_id}: status = {status}{cinfo}")

    if status not in TERMINAL:
        print("Not finished yet. Come back later and run this script again.")
        return

    if status == "completed":
        out_id = getattr(batch, "output_file_id", None)
        err_id = getattr(batch, "error_file_id", None)
        filled = failed = 0
        if out_id:
            filled, failed = merge_output_into_ckpt(_download_text(out_id), ckpt)
        n_err = 0
        if err_id:
            n_err = sum(1 for ln in _download_text(err_id).splitlines() if ln.strip())
        save_json(CKPT_FILE, ckpt)
        os.remove(STATE_FILE)                # batch consumed
        print(f"Collected: {filled} verdicts added, {failed} result lines "
              f"unusable, {n_err} error lines.")
        finish(documents, claims, ckpt)
    else:
        os.remove(STATE_FILE)                # dead batch consumed
        print(f"Batch ended as '{status}'. Nothing collected from it. Run the "
              f"submit script again to retry the outstanding claims.")
        finish(documents, claims, ckpt)


if __name__ == "__main__":
    main()
