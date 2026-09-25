"""
MODEL 1 - Stage 3 of 4: Google search via Serper
=================================================
Same search logic as your 3_search.py, pointed at Model 1's title-based queries.
Keeps the top organic results plus any labelled featured snippet (answerBox) or
knowledge panel (knowledgeGraph), with their source domains.

Reads:  pilot_outputs/model1/stage2_claims.json
Writes: pilot_outputs/model1/stage3_search.json
        pilot_outputs/model1/stage3_results.csv

Run:    python model1_search.py
Deps:   pip install requests pandas
Key:    SERPER_API_KEY   (read from the environment, not hard-coded)
"""

import os
import json
import time
from urllib.parse import urlparse

import requests
import pandas as pd


# ============================ CONFIG ========================================
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")   # set this in your shell
SERPER_URL = "https://google.serper.dev/search"
SERPER_NUM = 10
REQUEST_PAUSE = 0.4
MAX_RETRIES = 3

OUTDIR = os.path.join("pilot_outputs", "model1")
INPUT_FILE = os.path.join(OUTDIR, "stage2_claims.json")
OUTPUT_FILE = os.path.join(OUTDIR, "stage3_search.json")
RESULTS_CSV = os.path.join(OUTDIR, "stage3_results.csv")


def serper_search(query):
    if not SERPER_API_KEY:
        raise RuntimeError("SERPER_API_KEY is not set.")
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": query, "num": SERPER_NUM}
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(SERPER_URL, headers=headers, json=payload, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:               # noqa: BLE001
            last_err = e
            time.sleep(REQUEST_PAUSE * (attempt + 1))
    print(f"  ! Serper call failed: {last_err}")
    return {}


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


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Missing {INPUT_FILE}. Run model1_select_query.py first.")
        return
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    documents, claims = data["documents"], data["claims"]
    print(f"Loaded {len(claims)} title-claims from Model 1 Stage 2.")

    flat = []
    for n, c in enumerate(claims, start=1):
        payload = serper_search(c["query"])
        formatted, raw = format_results(payload)
        time.sleep(REQUEST_PAUSE)
        n_organic = sum(1 for x in raw if x.get("type") == "organic")
        has_box = any(x.get("type") in ("answer_box", "knowledge_graph") for x in raw)
        c["formatted_results"] = formatted
        c["results"] = raw
        c["n_results"] = n_organic
        print(f"  [{n}/{len(claims)}] {c['doc_id']} | organic={n_organic}"
              f"{' +box' if has_box else ''} | {c['query']}")
        flat.append({"doc_id": c["doc_id"], "claim": c["claim"],
                     "query": c["query"], "n_results": n_organic,
                     "has_box": has_box})

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"documents": documents, "claims": claims},
                  f, ensure_ascii=False, indent=2)
    pd.DataFrame(flat).to_csv(RESULTS_CSV, index=False, encoding="utf-8")
    print(f"\nModel 1 Stage 3 done. Wrote {OUTPUT_FILE} and {RESULTS_CSV}")


if __name__ == "__main__":
    main()
