"""
Evaluate the pipeline against gold labels (accuracy / precision / recall / F1)
=============================================================================
Gold: Final Dataset sheet, "Video id" (col B) + "Final Rating Decision" (col K).
Pred: posts.csv -> final_label, matched on video id.

The pipeline only makes a misinfo-vs-ok call when it commits to 'misinformation'
or 'true'. The other labels are abstentions and are EXCLUDED from precision /
recall, because scoring an "I don't know" as FN or FP would distort the metrics.
Which labels count as abstentions depends on the view:

  - FULL BENCHMARK : abstain = {unverified}. 'no_checkworthy_claim' still counts
                     as ok, and videos with no prediction are filled with it, so
                     gate / coverage failures stay visible as misses. This is the
                     honest end-to-end picture.
  - MATCHED ONLY   : abstain = {unverified, no_checkworthy_claim}. Only the
                     checked videos where the pipeline actually committed to
                     misinformation or true are scored. This is verdict quality.

Every excluded post is counted and printed, so the abstention rule is transparent.
A gold x prediction crosstab is also printed: it is the fastest way to see both
over-flagging (gold ok predicted misinformation) and abstention-masking (gold
misinfo predicted unverified, which scoring excludes rather than penalises).

Output: metrics table to console + pilot_outputs/eval_metrics.csv (fresh each run;
        copy it to a baseline name between runs to compare schemes).

Run:  python evaluate_f1.py
Deps: pip install pandas scikit-learn openpyxl
"""

import os
import pandas as pd
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             accuracy_score, confusion_matrix)


# ============================ CONFIG ========================================
PRED_FILE = "pilot_outputs/posts.csv"
PRED_ID_COL = "doc_id"
PRED_LABEL_COL = "final_label"

GOLD_FILE = r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_pipeline\Final Scrap - Final Dataset.xlsx"
GOLD_SHEET = "Final Dataset"
GOLD_ID_COL = "Video id"
GOLD_LABEL_COL = "Final Rating Decision"

METRICS_CSV = "pilot_outputs/eval_metrics.csv"
MISSING_PRED = "no_checkworthy_claim"   # fills videos with no prediction (full view)

PRED_MAP = {
    "misinformation": "misinfo",
    "true": "ok",
    "no_checkworthy_claim": "ok",       # only reached in the FULL view
}
GOLD_MAP = {                            # keys are matched case-insensitively
    "FALSE CLAIM": "misinfo",
    "TRUE CLAIM": "ok",
}
LABELS = ["misinfo", "ok"]             # positive class = misinfo

ABSTAIN_FULL = {"unverified"}
ABSTAIN_MATCHED = {"unverified", "no_checkworthy_claim"}


def load_gold():
    df = pd.read_excel(GOLD_FILE, sheet_name=GOLD_SHEET)
    df.columns = [str(c).strip() for c in df.columns]
    gid = GOLD_ID_COL if GOLD_ID_COL in df.columns else df.columns[1]
    glab = GOLD_LABEL_COL if GOLD_LABEL_COL in df.columns else df.columns[10]
    out = df[[gid, glab]].copy()
    out.columns = ["video_id", "gold"]
    out["video_id"] = out["video_id"].astype(str).str.strip()
    out["gold"] = out["gold"].astype(str).str.strip().str.upper()  # case-robust
    return out[out["video_id"].str.lower().ne("nan") & (out["video_id"] != "")]


def evaluate(sub, title, rows, abstain):
    """Score one view. Posts whose prediction is in `abstain` are excluded and
    counted; only committed misinfo/true calls enter the confusion matrix."""
    sub = sub.copy()
    sub["pred"] = sub["pred"].astype(str)

    excl = {lab: int((sub["pred"] == lab).sum()) for lab in sorted(abstain)}
    n_excl = sum(excl.values())

    y_true = sub["gold"].astype(str).map(GOLD_MAP)
    y_pred = sub["pred"].where(~sub["pred"].isin(abstain)).map(PRED_MAP)
    bad = y_true.isna() | y_pred.isna()
    yt, yp = y_true[~bad], y_pred[~bad]
    n = len(yt)

    acc = accuracy_score(yt, yp) if n else 0.0
    p = precision_score(yt, yp, pos_label="misinfo", zero_division=0)
    r = recall_score(yt, yp, pos_label="misinfo", zero_division=0)
    f1 = f1_score(yt, yp, pos_label="misinfo", zero_division=0)
    mf1 = f1_score(yt, yp, labels=LABELS, average="macro", zero_division=0)

    rows.append({
        "view": title,
        "n_scored": n,
        "n_excluded": n_excl,
        "accuracy": round(float(acc), 3),
        "precision_misinfo": round(float(p), 3),
        "recall_misinfo": round(float(r), 3),
        "f1_misinfo": round(float(f1), 3),
        "macro_f1": round(float(mf1), 3),
    })

    print(f"\n----- {title} -----")
    excl_str = ", ".join(f"{k}={v}" for k, v in excl.items()) or "none"
    print(f"scored n={n} | excluded (abstentions): {excl_str}")
    if n:
        print("confusion (rows = gold, cols = pred):", LABELS)
        print(confusion_matrix(yt, yp, labels=LABELS))


def main():
    if not os.path.exists(PRED_FILE):
        print(f"Missing {PRED_FILE}. Run 4_factcheck.py first.")
        return
    pred = pd.read_csv(PRED_FILE).fillna("")
    pred[PRED_ID_COL] = pred[PRED_ID_COL].astype(str).str.strip()
    gold = load_gold()

    print("prediction labels:", sorted(pred[PRED_LABEL_COL].astype(str).unique()))
    print("gold labels       :", sorted(gold["gold"].unique()))
    unmapped = sorted(set(gold["gold"].unique()) - set(GOLD_MAP))
    if unmapped:
        n_unmapped = int(gold["gold"].isin(unmapped).sum())
        print(f"gold labels with no mapping (excluded from scoring): {unmapped} "
              f"-> {n_unmapped} video(s)")

    df = gold.merge(pred[[PRED_ID_COL, PRED_LABEL_COL]],
                    left_on="video_id", right_on=PRED_ID_COL, how="left")
    missing = df[PRED_LABEL_COL].isna()
    print(f"\ngold videos: {len(gold)} | with a prediction: {(~missing).sum()} "
          f"| missing: {missing.sum()}")
    print("missing videos by gold label:")
    print(df[missing]["gold"].value_counts().to_string())

    rows = []
    full = df.copy()
    full["pred"] = full[PRED_LABEL_COL].fillna(MISSING_PRED)

    # Gold x raw prediction: shows over-flagging and abstention-masking at a glance.
    print("\ngold x raw pipeline label (full view, missing filled):")
    print(pd.crosstab(full["gold"], full["pred"]).to_string())

    evaluate(full, "FULL BENCHMARK", rows, ABSTAIN_FULL)

    matched = df[~missing].copy()
    matched["pred"] = matched[PRED_LABEL_COL]
    evaluate(matched, "MATCHED ONLY (committed calls)", rows, ABSTAIN_MATCHED)

    table = pd.DataFrame(rows, columns=[
        "view", "n_scored", "n_excluded", "accuracy",
        "precision_misinfo", "recall_misinfo", "f1_misinfo", "macro_f1"])
    print("\n================= METRICS TABLE =================")
    print(table.to_string(index=False))

    os.makedirs(os.path.dirname(METRICS_CSV), exist_ok=True)
    table.to_csv(METRICS_CSV, index=False, encoding="utf-8")
    print(f"\nWrote {METRICS_CSV}")


if __name__ == "__main__":
    main()
