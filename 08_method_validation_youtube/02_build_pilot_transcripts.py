"""
Build pilot_transcripts.json
============================
Joins two sources on video_id to produce the transcript input for stage 1:
  - video_id + transcript text  <- Transcripts_ready_for_claimbuster.json
                                   ('punctuated_transcript' field)
  - title, channel, published   <- 'youtube_Scrap' sheet of the Excel workbook
                                   (columns 'title', 'channel', 'publishedAt',
                                    matched on the 'video_id' column)

Output record schema (what stage 1 reads):
  {"video_id": ..., "title": ..., "channel": ..., "published": ...,
   "transcript": <punctuated_transcript>}

Run:    python build_pilot_transcripts.py
Deps:   pip install pandas openpyxl
"""

import json
import pandas as pd


# ============================ CONFIG ========================================
JSON_FILE = "Transcripts_ready_for_claimbuster.json"
XLSX_FILE = r"C:\Users\batuh\PycharmProjects\Thesis\Update\Final Scrap - Final Dataset.xlsx"
SHEET = "youtube_Scrap"
OUTPUT_FILE = "pilot_transcripts.json"

COL_VIDEO_ID = "video_id"
COL_TITLE = "title"
COL_CHANNEL = "channel"
COL_PUBLISHED = "publishedAt"


def _as_text(value):
    if pd.isna(value):
        return ""
    if hasattr(value, "strftime"):          # datetime / Timestamp -> YYYY-MM-DD
        return value.strftime("%Y-%m-%d")
    return str(value).strip()


def load_metadata():
    df = pd.read_excel(XLSX_FILE, sheet_name=SHEET, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    needed = [COL_VIDEO_ID, COL_TITLE, COL_CHANNEL, COL_PUBLISHED]

    if all(c in df.columns for c in needed):
        sub = df[needed].copy()
        print("Matched Excel columns by name.")
    else:
        missing = [c for c in needed if c not in df.columns]
        print(f"Named columns missing {missing}; falling back to columns B, C, D, E.")
        sub = df.iloc[:, [1, 2, 3, 4]].copy()   # B, C, D, E
    sub.columns = ["video_id", "title", "channel", "published"]

    meta = {}
    for _, r in sub.iterrows():
        vid = _as_text(r["video_id"])
        if vid and vid.lower() != "nan":
            meta[vid] = {"title": _as_text(r["title"]),
                         "channel": _as_text(r["channel"]),
                         "published": _as_text(r["published"])}
    return meta


def main():
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        transcripts = json.load(f)
    meta = load_metadata()
    print(f"Loaded {len(transcripts)} transcripts and {len(meta)} metadata rows.")

    records, unmatched, n_empty = [], [], 0
    for item in transcripts:
        vid = str(item.get("video_id", "")).strip()
        text = item.get("punctuated_transcript", "") or ""
        if not text:
            n_empty += 1
        if vid not in meta:
            unmatched.append(vid)
        m = meta.get(vid, {})
        records.append({
            "video_id": vid,
            "title": m.get("title", ""),
            "channel": m.get("channel", ""),
            "published": m.get("published", ""),
            "transcript": text,
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {len(records)} records to {OUTPUT_FILE}")
    if n_empty:
        print(f"  {n_empty} record(s) had an empty punctuated_transcript")
    if unmatched:
        shown = ", ".join(unmatched[:10]) + ("..." if len(unmatched) > 10 else "")
        print(f"  {len(unmatched)} video_id(s) not found in the sheet: {shown}")


if __name__ == "__main__":
    main()
