"""
MODEL 1 - Stage 2 of 4: query from title (the title IS the claim)
=================================================================
Model 1 treats the video TITLE as the claim. For each document that passed the
shared ClaimBuster gate, this stage looks up its title in the Final Dataset
workbook (matched by video ID) and asks GPT for ONE neutral search query per
title. No claim or normalized claim is requested, because the title is the claim.

Weak titles (e.g. "My morning routine") are kept on purpose so Model 1's
title-only limitation is visible in the results.

Reads:  pilot_outputs/stage1_passed.json   (the shared gate output)
        Final Scrap - Final Dataset.xlsx    (sheet 'Final Dataset', col B=Video ID, col C=title)
Writes: pilot_outputs/model1/stage2_claims.json   ({"documents": [...], "claims": [...]})
        pilot_outputs/model1/stage2_queries.csv

Run:    python model1_select_query.py
Deps:   pip install openai pandas openpyxl
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
SELECT_MAX_TOKENS = 2000
REQUEST_PAUSE = 0.4
MAX_RETRIES = 3

STAGE1_FILE = r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_pipeline\pilot_outputs\stage1_passed.json"

# Title source workbook (col B = Video ID, col C = title, in that order).
EXCEL_FILE = r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_pipeline\Final Scrap - Final Dataset.xlsx"
EXCEL_SHEET = "Final Dataset"
EXCEL_USECOLS = "B,C"

OUTDIR = os.path.join("pilot_outputs", "model1")
OUTPUT_FILE = os.path.join(OUTDIR, "stage2_claims.json")
QUERIES_CSV = os.path.join(OUTDIR, "stage2_queries.csv")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else OpenAI()
_DROP = set()   # arguments this endpoint rejects (reasoning_effort/response_format)


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


# ============================ TITLES ========================================
def load_titles(path, sheet, usecols):
    df = pd.read_excel(path, sheet_name=sheet, usecols=usecols, dtype=str)
    cols = list(df.columns)
    if len(cols) < 2:
        raise RuntimeError(f"Expected two columns from {usecols}, got {cols}")
    vid_col, title_col = cols[0], cols[1]            # B then C
    print(f"  Title columns read as: id='{vid_col}', title='{title_col}'")
    titles = {}
    for _, r in df.iterrows():
        vid = ("" if pd.isna(r[vid_col]) else str(r[vid_col])).strip()
        title = ("" if pd.isna(r[title_col]) else str(r[title_col])).strip()
        if vid and vid.lower() != "nan":
            titles[vid] = title
    return titles


# ===================== PROMPT (query from title) ===========================
QUERY_SYSTEM = (
    "You write one neutral web search query to help fact-check the factual "
    "content of a short video title. Respond with JSON only."
)


def query_from_title(title):
    usr = (
        "Video title:\n\n" + title + "\n\n"
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
    return query or title           # fall back to the raw title if GPT returns nothing


def main():
    if not os.path.exists(STAGE1_FILE):
        print(f"Missing {STAGE1_FILE}. Run the ClaimBuster gate (stage 1) first.")
        return
    with open(STAGE1_FILE, "r", encoding="utf-8") as f:
        documents = json.load(f)["documents"]
    print(f"Loaded {len(documents)} passed documents from stage 1.")

    titles = load_titles(EXCEL_FILE, EXCEL_SHEET, EXCEL_USECOLS)
    print(f"Loaded {len(titles)} titles from the workbook.")

    os.makedirs(OUTDIR, exist_ok=True)
    all_claims, flat, registry = [], [], []
    missing_title = 0
    for doc in documents:
        doc_id = doc["doc_id"]
        title = titles.get(doc_id) or doc.get("title", "")
        if not title:
            missing_title += 1
            print(f"  ! no title for {doc_id}; skipping")
            continue
        query = query_from_title(title)
        time.sleep(REQUEST_PAUSE)
        print(f"  {doc_id}: {query}")
        all_claims.append({
            "doc_id": doc_id, "source_type": doc.get("source_type", ""),
            "title": title, "author": doc.get("author", ""),
            "date": doc.get("date", ""),
            "claim": title, "claim_normalized": title, "query": query,
        })
        flat.append({"doc_id": doc_id, "claim": title, "query": query})
        registry.append({
            "doc_id": doc_id, "source_type": doc.get("source_type", ""),
            "title": title, "author": doc.get("author", ""),
            "date": doc.get("date", ""),
            "cb_max_score": doc.get("cb_max_score"),
            "clean_text": title,            # title-only context for stage 4 stance
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"documents": registry, "claims": all_claims},
                  f, ensure_ascii=False, indent=2)
    pd.DataFrame(flat).to_csv(QUERIES_CSV, index=False, encoding="utf-8")
    print(f"\nModel 1 Stage 2 done. {len(all_claims)} title-claims "
          f"({missing_title} documents had no title)."
          f"\nWrote {OUTPUT_FILE} and {QUERIES_CSV}")


if __name__ == "__main__":
    main()
