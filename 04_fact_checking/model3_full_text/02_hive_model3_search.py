"""
HIVE MODEL 3 - Stage 3 of 4: Google search via Serper (single key)
==================================================================
Same Serper formatting as the validated Model 3 Stage 3, threaded with resume.
Only SUCCESSFUL searches are checkpointed, so anything that fails (rate limit
exhausted, or the key out of credits) stays uncached and is retried on a rerun.

Process the claims in numeric slices with CLAIM_RANGE so you can do a batch when
you have time and continue later; prior results are always kept.

Reads:  hive_outputs/model3/stage2_claims.json
Writes: hive_outputs/model3/stage3_search.json
        hive_outputs/model3/stage3_results.csv
        hive_outputs/model3/_ckpt_search.json   (resume state, successes only)

Run:    python hive_model3_search.py
Deps:   pip install requests pandas
"""

import os
import json
import time
import threading
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd


# ============================ KEY ===========================================
# Paste your Serper key between the quotes, or leave it empty to read the
# SERPER_API_KEY environment variable instead.
SERPER_API_KEY = "enter key"
if not SERPER_API_KEY:
    SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")

# ===================== CLAIM RANGE (this run only) ==========================
# Process only a 1-indexed, inclusive slice of this model's claim list this run,
# e.g. [1, 2500]. The next run, e.g. [2501, 5000], keeps everything already
# searched (the checkpoint) and only fetches the new slice. Set to None to search
# every remaining claim. Claim numbering starts at 1. The range is over the claim
# count printed on load (Model 3 keeps up to 3 per post, so it is below 18,646).
CLAIM_RANGE = [17501,18300]

# ============================ CONFIG ========================================
SERPER_URL = "https://google.serper.dev/search"
SERPER_NUM = 10
N_WORKERS = 12          # lower this if you see many rate-limit (429) messages
SAVE_EVERY = 100
MAX_RETRIES = 3
REQUEST_PAUSE = 0.4

OUTDIR = os.path.join("hive_outputs", "model3")
INPUT_FILE = os.path.join(OUTDIR, "stage2_claims.json")
OUTPUT_FILE = os.path.join(OUTDIR, "stage3_search.json")
RESULTS_CSV = os.path.join(OUTDIR, "stage3_results.csv")
CKPT_FILE = os.path.join(OUTDIR, "_ckpt_search.json")

_lock = threading.Lock()


def serper_search(query):
    """Returns (ok, payload). ok is False on an invalid / out-of-credits key
    (401/403) or after retries are exhausted; 429 rate limits are backed off and
    retried on the same key."""
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    body = {"q": query, "num": SERPER_NUM}
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(SERPER_URL, headers=headers, json=body, timeout=30)
            sc = r.status_code
            if sc in (401, 403):
                return False, {}                         # invalid / out of credits
            if sc == 429:                                # rate limited: back off, retry
                time.sleep(REQUEST_PAUSE * (attempt + 2))
                continue
            r.raise_for_status()
            return True, r.json()
        except Exception:                                # noqa: BLE001
            time.sleep(REQUEST_PAUSE * (attempt + 1))
    return False, {}


def _domain(url):
    return urlparse(url or "").netloc.replace("www.", "")


def format_results(payload):
    lines, raw = [], []

    answer_box = payload.get("answerBox") or {}
    if answer_box:
        text = (answer_box.get("answer") or answer_box.get("snippet") or "").strip()
        if text:
            domain = _domain(answer_box.get("link", ""))
            title = answer_box.get("title", "").strip()
            lines.append(f"[Featured snippet] {domain}: {title}. {text}".strip())
            raw.append({"type": "answer_box", "domain": domain,
                        "link": answer_box.get("link", ""), "title": title,
                        "date": "", "snippet": text})

    kg = payload.get("knowledgeGraph") or {}
    if kg:
        desc = (kg.get("description") or "").strip()
        if desc:
            domain = (kg.get("descriptionSource", "")
                      or _domain(kg.get("descriptionLink", "")))
            title = kg.get("title", "").strip()
            typ = kg.get("type", "").strip()
            head = f"{title} ({typ})" if typ else title
            lines.append(f"[Knowledge panel] {domain}: {head}. {desc}".strip())
            raw.append({"type": "knowledge_graph", "domain": domain,
                        "link": kg.get("descriptionLink", ""), "title": title,
                        "date": "", "snippet": desc})

    organic = payload.get("organic", [])[:SERPER_NUM]
    for i, res in enumerate(organic, start=1):
        link = res.get("link", "")
        domain = _domain(link)
        title = res.get("title", "").strip()
        snippet = res.get("snippet", "").strip()
        date = res.get("date", "").strip()
        date_part = f"({date}): " if date else ""
        lines.append(f"{i}. {domain}: {date_part}{title}. {snippet}")
        raw.append({"type": "organic", "domain": domain, "link": link,
                    "title": title, "date": date, "snippet": snippet})

    return "\n".join(lines), raw


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


def search_one(claim):
    ok, payload = serper_search(claim["query"])
    if not ok:
        return {"_ok": False, "formatted_results": "", "results": [], "n_results": 0}
    formatted, raw = format_results(payload)
    n_organic = sum(1 for x in raw if x.get("type") == "organic")
    return {"_ok": True, "formatted_results": formatted, "results": raw,
            "n_results": n_organic}


def main():
    if not SERPER_API_KEY:
        print("No Serper key. Paste it into SERPER_API_KEY at the top of this "
              "file, or set the SERPER_API_KEY environment variable.")
        return
    if not os.path.exists(INPUT_FILE):
        print(f"Missing {INPUT_FILE}. Run hive_model3_select_query.py first.")
        return
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    documents, claims = data["documents"], data["claims"]

    n = len(claims)
    if CLAIM_RANGE is None:
        lo, hi = 1, n
    else:
        lo, hi = int(CLAIM_RANGE[0]), int(CLAIM_RANGE[1])
        lo, hi = max(1, lo), min(n, hi)
    in_range = list(range(lo - 1, hi)) if lo <= hi else []   # 1-indexed, inclusive

    ckpt = load_ckpt(CKPT_FILE)
    todo = [i for i in in_range if str(i) not in ckpt]
    print(f"Loaded {n} claims | this run searches slice [{lo}, {hi}] "
          f"({len(in_range)} claims): {len(in_range) - len(todo)} already done, "
          f"{len(todo)} to do.")
    print(f"Overall searched so far: {len(ckpt)}/{n}.")

    ok = fail = 0
    if todo:
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            futs = {ex.submit(search_one, claims[i]): i for i in todo}
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    res = fut.result()
                except Exception:            # noqa: BLE001
                    res = {"_ok": False, "formatted_results": "", "results": [],
                           "n_results": 0}
                with _lock:
                    if res.get("_ok"):
                        ckpt[str(i)] = {"formatted_results": res["formatted_results"],
                                        "results": res["results"],
                                        "n_results": res["n_results"]}
                        ok += 1
                        if ok % SAVE_EVERY == 0:
                            save_ckpt(CKPT_FILE, ckpt)
                            print(f"    searched {ok}/{len(todo)}")
                    else:
                        fail += 1
        save_ckpt(CKPT_FILE, ckpt)

    flat, missing = [], 0
    for i, c in enumerate(claims):
        res = ckpt.get(str(i))
        if res is None:
            c["formatted_results"], c["results"], c["n_results"] = "", [], 0
            missing += 1
        else:
            c["formatted_results"] = res.get("formatted_results", "")
            c["results"] = res.get("results", [])
            c["n_results"] = res.get("n_results", 0)
        flat.append({"doc_id": c["doc_id"], "claim": c["claim"], "query": c["query"],
                     "n_results": c["n_results"], "searched": res is not None})

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"documents": documents, "claims": claims},
                  f, ensure_ascii=False, indent=2)
    pd.DataFrame(flat).to_csv(RESULTS_CSV, index=False, encoding="utf-8")

    in_range_missing = sum(1 for i in in_range if str(i) not in ckpt)
    other_missing = missing - in_range_missing
    print(f"\nModel 3 Stage 3 done. Wrote {OUTPUT_FILE} and {RESULTS_CSV}")
    print(f"This run: {ok} searched OK, {fail} failed.")
    print(f"Overall: {n - missing}/{n} searched, {missing} remaining "
          f"({in_range_missing} in slice [{lo}, {hi}], {other_missing} in other slices).")
    if in_range_missing:
        print("  Some claims inside this slice are unsearched (rate limit or key "
              "out of credits). Rerun this slice to fill them.")
    if other_missing:
        print("  Set CLAIM_RANGE to the next slice and rerun; prior results kept.")
    if missing == 0:
        print("  All claims searched. Next: hive_model3_fact_check.py.")


if __name__ == "__main__":
    main()
