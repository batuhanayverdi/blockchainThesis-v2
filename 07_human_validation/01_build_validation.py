"""
Build the human-validation materials for checking GPT's veracity labels
=======================================================================
Draws a balanced sample of 100 posts (20 per accuracy category) from the final
regression pool (analysis_sample == True), splits them 50/50 across two raters so
each rater sees 10 posts per category, highlights each Model 3 claim's source
sentence in yellow, and writes a single self-contained annotation web page plus a
private answer key.

What you get
------------
  validation_app.html     ONE blog-styled page for BOTH raters. The rater picks
                          "Rater 1" or "Rater 2" on entry, reads the rules, then
                          reviews their 50 posts. Six labels + one checkbox per
                          post. Progress is saved in the browser, so they can
                          pause, close, and re-enter where they left off. An
                          Export button downloads validation_<rater>.csv.
  master_key.xlsx         YOUR copy only. Maps each post id to its model label,
                          category, ordinal, political flag, title/author/date.
                          NEVER give this to the raters.

Blinding: the HTML embeds only post id, title, author, date, body, and the
highlighted claim sentences. No model label / verdict / stance / justification is
ever placed on the page.

Highlighting: each of the up to three Model 3 claims is located in the post body
by exact match (after folding curly quotes and dashes), with a strict fuzzy pass
for lightly edited claims, and the exact claim text is highlighted. A claim that
cannot be located confidently is left unhighlighted and counted in the run
summary, rather than painted onto an unrelated sentence.

Reads
-----
  hive_review.xlsx               sheets "by_post" and "by_claim_model3"
  hive_sample_master.parquet     restricted to analysis_sample == True; supplies
                                 the FULL post body (by_post stores only a
                                 2000-char snippet)

Run:  python build_validation.py
Deps: pip install pandas openpyxl pyarrow
"""

import os
import re
import json
import html
import difflib
import pandas as pd


# ============================ CONFIG ========================================
# --- inputs (edit paths to match your machine) ---
REVIEW_XLSX = r"hive_review.xlsx"
MASTER_PARQUET = r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_hive_pipeline\hive sample preparation (politic-non-politic)\hive_cleaned.parquet"

# --- outputs ---
OUT_HTML = "validation_app.html"
OUT_KEY = "master_key.xlsx"

# --- sampling ---
SEED = 20260630              # fixed for reproducibility
PER_CATEGORY = 20           # 20 per category x 5 = 100 posts (20-20-20-20-20)
RATERS = ["Rater 1", "Rater 2"]   # 100 / 2 = 50 each, 10 per category each

# The five accuracy categories, ordered worst -> best (model3_label values).
# "unverified" is excluded from the sample but offered to raters as a 6th option.
CATEGORIES = ["pants_on_fire", "false", "mostly_false", "mostly_true", "true"]

# --- restriction to the analysis pool ---
# The 5,565 regression posts are the by_post rows whose final decision is not
# "abstain". "ok" and "misinfo" are kept; "abstain" (unverified) is dropped.
DECISION_COL = "model4_binary"        # in by_post: abstain / misinfo / ok
EXCLUDE_DECISIONS = {"abstain"}

# --- column names (verify these against your files) ---
LABEL_COL = "model3_label"        # in by_post: the 5 accuracy categories
ORDINAL_COL = "model3_ordinal"    # in by_post: accuracy ordinal (kept in key)
POLITICAL_COL = "is_political"    # in by_post
DOCID_COL = "doc_id"              # in by_post / by_claim (parquet has author+permlink)
AUTHOR_COL = "author"            # in parquet (col A)
PERMLINK_COL = "permlink"        # in parquet (col B)
# body_for_analysis is the exact text ClaimBuster scored and Model 3 extracted
# and fact-checked from, so the Column G claims align with it. body_clean is a
# lighter clean the claims were NOT taken from (misaligned highlights, stray URL
# fragments); kept only as a fallback.
TEXT_COL_CANDIDATES = ["body_for_analysis", "body_clean"]

# --- highlighting ---
MAX_CLAIMS_PER_POST = 3     # Model 3 keeps up to 3

# --- eligibility filter -----------------------------------------------------
# Drop posts whose veracity is not independently checkable from public sources:
# automated/templated posts, activity logs, token/reward mechanics, and so on.
# Applied to the pool BEFORE sampling. These patterns catch the mechanical cases;
# opinion, personal diaries, and fiction usually need the manual drop list below.
# Judge eligibility from CONTENT, not from the model label, to stay unbiased.
EXCLUDE_AUTHORS = {"lolzbot", "pimp.token", "voice.direct", "actifit"}
EXCLUDE_PERMLINK_SUBSTR = [
    "actifit-", "dividends-report", "daily-post", "burn-post",
    "power-up-day", "hpud", "report-card",
]
EXCLUDE_DATE_PERMLINK = True                 # permlinks that are just a date
MANUAL_DROP_IDS = set()                      # e.g. {"author/permlink", ...}
DROP_IDS_FILE = "validation_drop_ids.txt"    # optional, one post_id per line

DISPLAY = {
    "pants_on_fire": "Pants on Fire",
    "false": "False",
    "mostly_false": "Mostly False",
    "mostly_true": "Mostly True",
    "true": "True",
    "unverified": "Unverified or cannot determine",
}


# ============================ HELPERS =======================================
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}([t_-]|$)", re.I)


def is_eligible(doc_id, author, permlink, drop_ids):
    """Content-based eligibility. Returns False for non-verifiable posts."""
    if doc_id in drop_ids:
        return False
    a = (author or "").lower().strip()
    p = (permlink or "").lower().strip()
    if a in EXCLUDE_AUTHORS:
        return False
    if any(s in p for s in EXCLUDE_PERMLINK_SUBSTR):
        return False
    if EXCLUDE_DATE_PERMLINK and DATE_RE.match(p):
        return False
    return True


def _norm_docid(df=None):
    """Return a doc_id Series: use the column if present, else author/permlink."""
    if DOCID_COL in df.columns:
        return df[DOCID_COL].astype(str).str.strip()
    if AUTHOR_COL in df.columns and PERMLINK_COL in df.columns:
        return (df[AUTHOR_COL].astype(str).str.strip() + "/" +
                df[PERMLINK_COL].astype(str).str.strip())
    raise KeyError(f"Cannot build doc_id: need '{DOCID_COL}' or "
                   f"'{AUTHOR_COL}'+'{PERMLINK_COL}'.")


# Fold typographic variants to ASCII so a verbatim claim still matches the body
# even when one side uses curly quotes, en/em dashes, or non-breaking spaces.
# All mappings are one character to one character so offset maps stay aligned.
_FOLD = {
    "\u2018": "'", "\u2019": "'", "\u201b": "'", "\u2032": "'", "\u00b4": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u2033": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
    "\u2015": "-", "\u2212": "-", "\u00a0": " ",
}

COVERAGE_MIN = 0.85         # min share of a claim that must align to highlight it


def _fold_lower(ch):
    return _FOLD.get(ch, ch).lower()


def _norm_map(s):
    """Fold + lowercase + collapse whitespace, keeping a map from each normalized
    character back to its original index (used to locate a claim span)."""
    out, idx_map, prev_space = [], [], False
    for i, ch in enumerate(s):
        if ch.isspace():
            if prev_space:
                continue
            out.append(" ")
            idx_map.append(i)
            prev_space = True
        else:
            out.append(_fold_lower(ch))
            idx_map.append(i)
            prev_space = False
    return "".join(out), idx_map


def _norm_str(s):
    """Same normalization as _norm_map, without the offset map (for the claim)."""
    out, prev_space = [], False
    for ch in s:
        if ch.isspace():
            if not prev_space:
                out.append(" ")
                prev_space = True
        else:
            out.append(_fold_lower(ch))
            prev_space = False
    return "".join(out).strip()


def _find_claim_span(body, body_norm, body_map, claim):
    """Locate the model's verbatim claim. Returns (start, end) or None. Tries a
    literal find, then a fold/case/whitespace-normalized find."""
    j = body.find(claim)
    if j >= 0:
        return (j, j + len(claim))
    cnorm = _norm_str(claim)
    if cnorm:
        k = body_norm.find(cnorm)
        if k >= 0:
            return (body_map[k], body_map[k + len(cnorm) - 1] + 1)
    return None


def _fuzzy_span(body_norm, body_map, cnorm):
    """For a claim the model lightly edited: find where it aligns in the body via
    contiguous matching blocks. Returns (start, end) only if most of the claim
    aligns in a tight region, else None (so we never paint an unrelated line)."""
    if len(cnorm) < 12:
        return None
    sm = difflib.SequenceMatcher(None, body_norm, cnorm, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size >= 4]
    if not blocks:
        return None
    coverage = sum(b.size for b in blocks) / len(cnorm)
    a_start = blocks[0].a
    a_end = blocks[-1].a + blocks[-1].size
    if coverage < COVERAGE_MIN or (a_end - a_start) > len(cnorm) * 1.6:
        return None
    return (body_map[a_start], body_map[min(a_end, len(body_map)) - 1] + 1)


_PARA = "\x00P\x00"


def _render_with_marks(body, spans):
    """Render body to HTML, wrapping each (start, end) span in <mark>. Blank
    lines become paragraphs; single newlines become spaces."""
    parts, cur = [], 0
    for s, e in spans:
        if s > cur:
            parts.append(("plain", body[cur:s]))
        parts.append(("mark", body[s:e]))
        cur = e
    if cur < len(body):
        parts.append(("plain", body[cur:]))

    chunks = []
    for kind, text in parts:
        if kind == "mark":
            t = html.escape(re.sub(r"\s+", " ", text).strip())
            chunks.append(f'<mark class="claim">{t}</mark>')
        else:
            t = html.escape(text)
            t = re.sub(r"\n\s*\n", _PARA, t).replace("\n", " ")
            chunks.append(t)
    joined = "".join(chunks)
    return "\n".join(f"<p>{p.strip()}</p>"
                     for p in joined.split(_PARA) if p.strip())


def build_body_html(body, claims):
    """Highlight each model claim where it appears in body_for_analysis. Claims
    are copied verbatim, so they are located as substrings (after folding curly
    quotes and dashes); a strict fuzzy pass covers lightly edited ones. A claim
    that cannot be located confidently is left unhighlighted rather than guessed.
    Returns (html, n_matched)."""
    body = body or ""
    body_norm, body_map = _norm_map(body)
    spans, n_matched = [], 0
    for c in claims:
        c = (c or "").strip()
        if not c:
            continue
        span = _find_claim_span(body, body_norm, body_map, c)
        if span is None:
            span = _fuzzy_span(body_norm, body_map, _norm_str(c))
        if span:
            spans.append(span)
            n_matched += 1

    # sort and merge overlaps so nested or adjacent claims render as valid HTML
    spans.sort()
    merged = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return _render_with_marks(body, [(s, e) for s, e in merged]), n_matched


# ============================ MAIN ==========================================
def main():
    if not os.path.exists(REVIEW_XLSX):
        raise SystemExit(f"Missing {REVIEW_XLSX}")

    by_post = pd.read_excel(REVIEW_XLSX, sheet_name="by_post", dtype=str).fillna("")
    by_claim = pd.read_excel(REVIEW_XLSX, sheet_name="by_claim_model3",
                             dtype=str).fillna("")
    by_post["doc_id"] = _norm_docid(by_post)
    by_claim["doc_id"] = _norm_docid(by_claim)

    # ---- claims per post (up to 3), computed up front so we can require >=1 ----
    claims_by_doc = {}
    for did, grp in by_claim.groupby("doc_id"):
        claims_by_doc[did] = [c for c in grp["claim"].tolist()
                              if c.strip()][:MAX_CLAIMS_PER_POST]

    # ---- drop abstain (unverified) so we keep the ~5,565 analysis pool ----
    if DECISION_COL in by_post.columns:
        before = len(by_post)
        keep = ~by_post[DECISION_COL].str.strip().str.lower().isin(EXCLUDE_DECISIONS)
        by_post = by_post[keep].copy()
        print(f"  Dropped {DECISION_COL} in {EXCLUDE_DECISIONS}: "
              f"{len(by_post)} of {before} by_post rows kept.")
    else:
        print(f"  WARNING: '{DECISION_COL}' not in by_post; available: "
              f"{list(by_post.columns)}")

    # ---- match to hive_cleaned.parquet for full body_clean text ----
    full_text = {}
    if os.path.exists(MASTER_PARQUET):
        mdf = pd.read_parquet(MASTER_PARQUET)
        mdf["doc_id"] = _norm_docid(mdf)
        text_col = next((c for c in TEXT_COL_CANDIDATES if c in mdf.columns), None)
        if text_col is None:
            print(f"  WARNING: none of {TEXT_COL_CANDIDATES} in parquet; "
                  f"available: {list(mdf.columns)}")
        else:
            print(f"  Full-body column: {text_col} ({len(mdf)} parquet rows).")
            full_text = {d: t for d, t in zip(mdf["doc_id"],
                                              mdf[text_col].astype(str))
                         if str(t).strip()}
        pool_ids = set(full_text)
        before = len(by_post)
        by_post = by_post[by_post["doc_id"].isin(pool_ids)].copy()
        print(f"  Matched to parquet (non-empty body): {len(by_post)} of "
              f"{before} posts.")
    else:
        raise SystemExit(f"  Parquet not found: {MASTER_PARQUET}. Needed for "
                         f"full body text; fix MASTER_PARQUET.")

    # ---- eligibility filter (drop non-verifiable content) ----
    drop_ids = set(MANUAL_DROP_IDS)
    if os.path.exists(DROP_IDS_FILE):
        with open(DROP_IDS_FILE, encoding="utf-8") as fh:
            drop_ids |= {ln.strip() for ln in fh
                         if ln.strip() and not ln.startswith("#")}

    def _elig(row):
        did = row["doc_id"]
        if not claims_by_doc.get(did):        # no model claim = nothing to check
            return False
        perm = did.split("/", 1)[1] if "/" in did else did
        author = did.split("/", 1)[0] if "/" in did else ""
        return is_eligible(did, author, perm, drop_ids)

    mask = by_post.apply(_elig, axis=1)
    print(f"\n  Eligibility filter dropped {int((~mask).sum())} of "
          f"{len(by_post)} posts (manual drops listed: {len(drop_ids)}).")
    by_post = by_post[mask].copy()

    # ---- keep only the five accuracy categories ----
    by_post = by_post[by_post[LABEL_COL].isin(CATEGORIES)].copy()
    counts = by_post[LABEL_COL].value_counts()
    print("\n  Available per category (eligible pool):")
    for c in CATEGORIES:
        avail = int(counts.get(c, 0))
        flag = "  <-- under PER_CATEGORY" if avail < PER_CATEGORY else ""
        print(f"    {c:<14} {avail}{flag}")

    # ---- stratified sample: PER_CATEGORY per category ----
    picked = []
    for c in CATEGORIES:
        pool = by_post[by_post[LABEL_COL] == c]
        n = min(PER_CATEGORY, len(pool))
        if n < PER_CATEGORY:
            print(f"  WARNING: only {n} '{c}' posts; 20-20-20-20-20 not met.")
        picked.append(pool.sample(n=n, random_state=SEED))
    sample = pd.concat(picked, ignore_index=True)

    # ---- assign to raters: PER_CATEGORY/len(RATERS) per category each ----
    per_rater_per_cat = PER_CATEGORY // len(RATERS)
    sample["rater"] = ""
    for c in CATEGORIES:
        idx = (sample.loc[sample[LABEL_COL] == c]
               .sample(frac=1.0, random_state=SEED + 1).index.tolist())
        for k, r in enumerate(RATERS):
            chunk = idx[k * per_rater_per_cat:(k + 1) * per_rater_per_cat]
            sample.loc[chunk, "rater"] = r

    # ---- build per-rater post payloads (blinded) ----
    data = {r: [] for r in RATERS}
    key_rows = []
    tot_claims = tot_matched = 0
    miss_list = []
    for _, row in sample.iterrows():
        did = row["doc_id"]
        claims = claims_by_doc.get(did, [])
        body = full_text.get(did, "")
        body_html, n_match = build_body_html(body, claims)
        tot_claims += len(claims)
        tot_matched += n_match
        if n_match < len(claims):
            miss_list.append((did, n_match, len(claims)))

        data[row["rater"]].append({
            "post_id": did,
            "title": row.get("title", ""),
            "author": row.get("author", ""),
            "date": row.get("date", ""),
            "body_html": body_html,
        })
        key_rows.append({
            "post_id": did,
            "rater": row["rater"],
            "model_label": row[LABEL_COL],
            "model_ordinal": row.get(ORDINAL_COL, ""),
            "model4_binary": row.get(DECISION_COL, ""),
            "is_political": row.get(POLITICAL_COL, ""),
            "n_claims": len(claims),
            "title": row.get("title", ""),
            "author": row.get("author", ""),
            "date": row.get("date", ""),
        })

    # interleave categories within each rater's display order
    for r in RATERS:
        df = pd.DataFrame(data[r]).sample(frac=1.0, random_state=SEED + 2)
        data[r] = df.to_dict("records")

    # ---- write the answer key (your eyes only) ----
    pd.DataFrame(key_rows).to_excel(OUT_KEY, index=False)
    print(f"\n  Wrote {OUT_KEY} ({len(key_rows)} rows). Keep this private.")
    unmatched = tot_claims - tot_matched
    print(f"  Claim highlighting: {tot_matched}/{tot_claims} claims located, "
          f"{unmatched} not confidently located.")
    if miss_list:
        print(f"  {len(miss_list)} posts highlight fewer claims than they have "
              f"(inspect the body; the claim text may not sit in it verbatim):")
        for did, m, c in miss_list:
            print(f"    {did}  ({m}/{c})")

    # ---- write the single annotation page ----
    page = HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(data))
    page = page.replace("__RATERS_JSON__", json.dumps(RATERS))
    page = page.replace("__LABELS_JSON__", json.dumps(
        [(k, DISPLAY[k]) for k in
         ["pants_on_fire", "false", "mostly_false", "mostly_true", "true",
          "unverified"]]))
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"  Wrote {OUT_HTML}. Posts per rater: "
          f"{ {r: len(data[r]) for r in RATERS} }")
    print("  Send validation_app.html to both raters.")


# ============================ HTML TEMPLATE =================================
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Post review</title>
<style>
  :root { --bg:#f4f1ea; --card:#fffdf8; --ink:#222; --muted:#777;
          --accent:#1b3a5b; --mark:#fff27a; --line:#e3ddd0; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font-family:Georgia,"Times New Roman",serif; line-height:1.65; }
  .wrap { max-width:720px; margin:0 auto; padding:24px 18px 120px; }
  .screen { display:none; }
  .screen.active { display:block; }
  h1 { font-size:28px; margin:0 0 6px; }
  .byline { color:var(--muted); font-size:14px; margin-bottom:22px;
            font-family:Arial,Helvetica,sans-serif; }
  .post { background:var(--card); border:1px solid var(--line); border-radius:10px;
          padding:28px 30px; box-shadow:0 1px 3px rgba(0,0,0,.05); }
  .post p { margin:0 0 16px; font-size:18px; }
  mark.claim { background:var(--mark); padding:1px 2px; border-radius:2px; }
  .panel { background:var(--card); border:1px solid var(--line); border-radius:10px;
           padding:20px 24px; margin-top:18px;
           font-family:Arial,Helvetica,sans-serif; }
  .panel h3 { margin:0 0 12px; font-size:16px; }
  .opt { display:block; padding:10px 12px; margin:6px 0; border:1px solid var(--line);
         border-radius:8px; cursor:pointer; font-size:15px; }
  .opt:hover { background:#f7f3ea; }
  .opt input { margin-right:10px; }
  .check { margin-top:14px; font-size:15px; }
  .nav { position:fixed; bottom:0; left:0; right:0; background:#fff;
         border-top:1px solid var(--line); padding:12px 18px;
         display:flex; align-items:center; gap:14px; justify-content:center;
         font-family:Arial,Helvetica,sans-serif; }
  .bar { flex:0 0 220px; height:8px; background:var(--line); border-radius:5px;
         overflow:hidden; }
  .bar > div { height:100%; background:var(--accent); width:0; }
  button { font:inherit; font-family:Arial,Helvetica,sans-serif; cursor:pointer;
           border:1px solid var(--accent); background:var(--accent); color:#fff;
           padding:9px 18px; border-radius:8px; }
  button.ghost { background:#fff; color:var(--accent); }
  button:disabled { opacity:.4; cursor:not-allowed; }
  .center { text-align:center; }
  .rules { font-family:Arial,Helvetica,sans-serif; font-size:15px; }
  .rules li { margin:8px 0; }
  .raterbtns button { margin:8px; min-width:140px; }
  .small { font-size:13px; color:var(--muted); font-family:Arial,Helvetica,sans-serif; }
</style>
</head>
<body>
<div class="wrap">

  <!-- ENTRY -->
  <div id="entry" class="screen active center">
    <h1>Post review</h1>
    <p class="small">Thank you for helping. First, choose who you are.</p>
    <div class="raterbtns" id="raterbtns"></div>
    <p style="margin-top:28px">
      <button class="ghost" onclick="resetAll()">Reset saved progress</button></p>
    <p class="small">For testing only. This clears every saved answer in this
       browser so you can start fresh.</p>
  </div>

  <!-- RULES -->
  <div id="rules" class="screen">
    <h1>How to review</h1>
    <div class="rules">
      <p>You will read <b id="nposts"></b> short blog posts. In each post, the
         sentence(s) we want you to check are <mark class="claim">highlighted in
         yellow</mark>. Judge the highlighted statement(s) together.</p>
      <ol>
        <li>Read the post and the highlighted claim(s).</li>
        <li>Open Google in another tab and search for the claim. Check the news
            as well, the way you would if you were verifying it yourself.</li>
        <li>Decide how accurate the highlighted claim(s) are, taken together,
            and choose one label.</li>
        <li>An error does not by itself make a claim <b>False</b>. If the claim
            still carries a real element of truth but leaves a misleading
            impression, it is <b>Mostly false</b>. Use <b>False</b> when the
            claim is simply not accurate.</li>
        <li>Choose <b>Unverified or cannot determine</b> only when, after
            searching, you still cannot tell whether the claim is true or false,
            because reliable evidence is missing or sources genuinely conflict.
            Do not use it for a claim that is part true and part false; that is
            Mostly false or False.</li>
        <li>Tick the checkbox if the highlighted texts are checkable factual
            claims (statements that could be true or false), rather than
            opinions or questions.</li>
      </ol>
      <p><b>What the labels mean</b></p>
      <ul>
        <li><b>True:</b> accurate, with nothing significant missing.</li>
        <li><b>Mostly true:</b> accurate but needs clarification or additional
            information.</li>
        <li><b>Mostly false:</b> contains an element of truth but ignores
            critical facts that would give a different impression.</li>
        <li><b>False:</b> not accurate.</li>
        <li><b>Pants on fire:</b> not accurate, and additionally makes an absurd
            or egregious claim with no basis in fact.</li>
        <li><b>Unverified or cannot determine:</b> after searching, the evidence
            is missing or genuinely conflicting, so you cannot tell.</li>
      </ul>
      <p class="small">Your progress is saved automatically in this browser. You
         can close the page and come back later on the same computer to continue
         where you left off.</p>
    </div>
    <p class="center"><button onclick="startAnnotating()">Start reviewing</button></p>
  </div>

  <!-- POST -->
  <div id="annot" class="screen">
    <article class="post">
      <h1 id="p_title"></h1>
      <div class="byline" id="p_byline"></div>
      <div id="p_body"></div>
    </article>

    <div class="panel">
      <h3>How accurate are the highlighted claim(s)?</h3>
      <div id="options"></div>
      <label class="check">
        <input type="checkbox" id="claimcheck">
        The highlighted texts are checkable factual claims.
      </label>
    </div>
  </div>

  <!-- DONE -->
  <div id="done" class="screen center">
    <h1>All done</h1>
    <p class="rules">You have reviewed every post. Please click the button below
       to download your results, then send the file back.</p>
    <p><button onclick="exportCSV()">Export my results</button></p>
  </div>

</div>

<!-- NAV BAR -->
<div class="nav" id="navbar" style="display:none">
  <button class="ghost" id="backbtn" onclick="go(-1)">Back</button>
  <div class="bar"><div id="barfill"></div></div>
  <span class="small" id="progress"></span>
  <button id="nextbtn" onclick="go(1)">Next</button>
  <button class="ghost" onclick="exportCSV()">Export</button>
  <button class="ghost" onclick="resetCurrent()">Reset</button>
</div>

<script>
const DATA = __DATA_JSON__;
const RATERS = __RATERS_JSON__;
const LABELS = __LABELS_JSON__;   // [[value, display], ...]

let STATE = { rater:null, idx:0, posts:[], answers:{}, seen:{} };

function lsKey(){ return "hive_val_" + STATE.rater; }
function save(){
  localStorage.setItem(lsKey(), JSON.stringify({
    idx: STATE.idx, answers: STATE.answers, seen: STATE.seen }));
}
function load(){
  try {
    const raw = localStorage.getItem(lsKey());
    if (raw){ const o = JSON.parse(raw);
      STATE.answers = o.answers || {}; STATE.seen = o.seen || {};
      STATE.idx = o.idx || 0; }
  } catch(e){}
}

function show(id){
  document.querySelectorAll(".screen").forEach(s=>s.classList.remove("active"));
  document.getElementById(id).classList.add("active");
  document.getElementById("navbar").style.display = (id==="annot")?"flex":"none";
}

const rb = document.getElementById("raterbtns");
RATERS.forEach(r=>{
  const b = document.createElement("button");
  b.textContent = r; b.onclick = ()=>pickRater(r); rb.appendChild(b);
});

function pickRater(r){
  STATE.rater = r; STATE.posts = DATA[r] || []; load();
  document.getElementById("nposts").textContent = STATE.posts.length;
  show("rules");
}

function startAnnotating(){
  let i = STATE.posts.findIndex(p => !(p.post_id in STATE.answers));
  STATE.idx = (i === -1) ? STATE.posts.length - 1 : i;
  render(); show("annot"); window.scrollTo(0,0);
}

function render(){
  const p = STATE.posts[STATE.idx];
  if (!p){ show("done"); return; }
  if (!STATE.seen[p.post_id]) STATE.seen[p.post_id] = Date.now();

  document.getElementById("p_title").textContent = p.title || "(untitled)";
  document.getElementById("p_byline").textContent =
    "by " + (p.author||"unknown") + (p.date ? "  -  " + p.date : "");
  document.getElementById("p_body").innerHTML = p.body_html || "";

  const a = STATE.answers[p.post_id] || {};
  const opts = document.getElementById("options");
  opts.innerHTML = LABELS.map(([val,disp])=>
    '<label class="opt"><input type="radio" name="lab" value="'+val+'"'+
    (a.human_label===val?" checked":"")+'>'+disp+'</label>').join("");
  opts.querySelectorAll('input[name="lab"]').forEach(el=>{ el.onchange = record; });
  document.getElementById("claimcheck").checked = !!a.is_checkable_claim;
  document.getElementById("claimcheck").onchange = record;

  document.getElementById("backbtn").disabled = (STATE.idx===0);
  const last = (STATE.idx === STATE.posts.length-1);
  document.getElementById("nextbtn").textContent = last ? "Finish" : "Next";
  const ans = Object.keys(STATE.answers).length;
  document.getElementById("progress").textContent =
    (STATE.idx+1)+" / "+STATE.posts.length+"  (answered "+ans+")";
  document.getElementById("barfill").style.width = (100*ans/STATE.posts.length)+"%";
  save();
}

function record(){
  const p = STATE.posts[STATE.idx];
  const sel = document.querySelector('input[name="lab"]:checked');
  STATE.answers[p.post_id] = {
    human_label: sel ? sel.value : "",
    is_checkable_claim: document.getElementById("claimcheck").checked,
    order_shown: STATE.idx,
    seconds_spent: STATE.seen[p.post_id]
      ? Math.round((Date.now()-STATE.seen[p.post_id])/1000) : "",
    answered_at: new Date().toISOString()
  };
  save(); render();
}

function go(d){
  const p = STATE.posts[STATE.idx];
  if (d>0 && !(STATE.answers[p.post_id] && STATE.answers[p.post_id].human_label)){
    alert("Please choose a label before continuing."); return;
  }
  STATE.idx += d;
  if (STATE.idx >= STATE.posts.length){ show("done"); window.scrollTo(0,0); return; }
  if (STATE.idx < 0) STATE.idx = 0;
  render();
  window.scrollTo(0,0);
}

function exportCSV(){
  const cols = ["rater","post_id","order_shown","human_label",
                "is_checkable_claim","seconds_spent","answered_at"];
  const rows = [cols.join(",")];
  STATE.posts.forEach(p=>{
    const a = STATE.answers[p.post_id]; if (!a) return;
    rows.push([STATE.rater, p.post_id, a.order_shown, a.human_label,
               a.is_checkable_claim, a.seconds_spent, a.answered_at]
              .map(csv).join(","));
  });
  const blob = new Blob([rows.join("\n")], {type:"text/csv"});
  const url = URL.createObjectURL(blob);
  const fn = "validation_" + STATE.rater.replace(/\s+/g,"").toLowerCase() + ".csv";
  const a = document.createElement("a"); a.href = url; a.download = fn; a.click();
  URL.revokeObjectURL(url);
}

function csv(v){
  v = (v===undefined||v===null) ? "" : String(v);
  return /[",\n]/.test(v) ? '"'+v.replace(/"/g,'""')+'"' : v;
}

function resetAll(){
  if (!confirm("Clear ALL saved answers in this browser? This cannot be undone."))
    return;
  Object.keys(localStorage).filter(k=>k.indexOf("hive_val_")===0)
    .forEach(k=>localStorage.removeItem(k));
  location.reload();
}

function resetCurrent(){
  if (!confirm("Clear "+STATE.rater+"'s saved answers and start over?")) return;
  localStorage.removeItem(lsKey());
  STATE.answers = {}; STATE.seen = {}; STATE.idx = 0;
  show("entry");
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
