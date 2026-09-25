"""
Compare the three models against gold (full + matched views)
============================================================
This is your evaluate_f1.py generalized to all three models. The gold schema,
the two views, the abstention rules, and the metric definitions are unchanged.
The only addition is the prediction mapping: Model 3 emits six labels, so the
false-side ones (pants_on_fire / false / mostly_false) map to 'misinfo' and the
true-side ones (mostly_true / true) to 'ok'. With that one extension the same
scorer handles Model 1 (3-class), Model 2 (3-class) and Model 3 (6-class).

Gold: Final Dataset sheet, "Video id" (col B) + "Final Rating Decision" (col K).
Pred: each model's posts.csv -> final_label, matched on video id.

Views (identical to evaluate_f1.py):
  - FULL BENCHMARK : abstain = {unverified}. 'no_checkworthy_claim' counts as ok,
                     and videos with no prediction are filled with it, so gate /
                     coverage failures stay visible as misses. Honest end-to-end.
  - MATCHED ONLY   : abstain = {unverified, no_checkworthy_claim}. Only checked
                     videos where the model committed are scored. Verdict quality.

Output: per-model crosstab + both views to console, and a combined metrics table
        to console + pilot_outputs/model_comparison_metrics.csv.

Run:  python compare_model.py
Deps: pip install pandas scikit-learn openpyxl
"""

import os
import pandas as pd
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             accuracy_score, confusion_matrix)


# ============================ CONFIG ========================================
MODEL_FILES = {
    "Model 1 (title)":
        r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_pipeline\1. model - just title\pilot_outputs\model1\posts.csv",
    "Model 2 (full, 3-class)":
        r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_pipeline\pilot_outputs\posts.csv",
    "Model 3 (full, 6-class)":
        r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_pipeline\3. model - full transkript (6 verdict)\pilot_outputs\model3\posts.csv",
    "Model 4 (cascade 3+1)":
        r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_pipeline\4. model - cascade 3+1\pilot_outputs\model4\posts.csv",
}
PRED_ID_COL = "doc_id"
PRED_LABEL_COL = "final_label"

GOLD_FILE = r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_pipeline\Final Scrap - Final Dataset.xlsx"
GOLD_SHEET = "Final Dataset"
GOLD_ID_COL = "Video id"
GOLD_LABEL_COL = "Final Rating Decision"

METRICS_CSV = "pilot_outputs/model_comparison_metrics.csv"
MISSING_PRED = "no_checkworthy_claim"   # fills videos with no prediction (full view)

# Prediction -> binary. Covers 3-class (Model 1/2) and 6-class (Model 3).
PRED_MAP = {
    "misinformation": "misinfo",
    "pants_on_fire": "misinfo",
    "false": "misinfo",
    "mostly_false": "misinfo",
    "true": "ok",
    "mostly_true": "ok",
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
    gid = GOLD_ID_COL if GOLD_ID_COL in df.columns else df.columns[1]    # col B
    glab = GOLD_LABEL_COL if GOLD_LABEL_COL in df.columns else df.columns[10]  # col K
    out = df[[gid, glab]].copy()
    out.columns = ["video_id", "gold"]
    out["video_id"] = out["video_id"].astype(str).str.strip()
    out["gold"] = out["gold"].astype(str).str.strip().str.upper()       # case-robust
    return out[out["video_id"].str.lower().ne("nan") & (out["video_id"] != "")]


def evaluate(sub, model, view, rows, abstain):
    """Score one model under one view. Posts whose prediction is in `abstain`
    are excluded and counted; only committed misinfo/ok calls are scored."""
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
        "model": model, "view": view,
        "n_scored": n, "n_excluded": n_excl,
        "accuracy": round(float(acc), 3),
        "precision_misinfo": round(float(p), 3),
        "recall_misinfo": round(float(r), 3),
        "f1_misinfo": round(float(f1), 3),
        "macro_f1": round(float(mf1), 3),
    })

    excl_str = ", ".join(f"{k}={v}" for k, v in excl.items()) or "none"
    print(f"  [{view}] scored n={n} | excluded (abstentions): {excl_str}")
    if n:
        print("    confusion (rows = gold, cols = pred):", LABELS)
        for line in str(confusion_matrix(yt, yp, labels=LABELS)).splitlines():
            print("    " + line)


def evaluate_model(name, path, gold, rows):
    if not os.path.exists(path):
        print(f"\n##### {name}: file not found: {path}")
        return
    pred = pd.read_csv(path).fillna("")
    if PRED_ID_COL not in pred.columns or PRED_LABEL_COL not in pred.columns:
        print(f"\n##### {name}: missing '{PRED_ID_COL}' or '{PRED_LABEL_COL}'. "
              f"Columns: {list(pred.columns)}")
        return
    pred[PRED_ID_COL] = pred[PRED_ID_COL].astype(str).str.strip()

    print(f"\n############### {name} ###############")
    pred_labels = sorted(pred[PRED_LABEL_COL].astype(str).unique())
    print("  prediction labels:", pred_labels)
    unmapped_pred = [l for l in pred_labels
                     if l not in PRED_MAP and l not in (ABSTAIN_FULL | ABSTAIN_MATCHED)]
    if unmapped_pred:
        print(f"  WARNING: prediction labels with no mapping (excluded): {unmapped_pred}")

    df = gold.merge(pred[[PRED_ID_COL, PRED_LABEL_COL]],
                    left_on="video_id", right_on=PRED_ID_COL, how="left")
    missing = df[PRED_LABEL_COL].isna()
    print(f"  gold videos: {len(gold)} | with a prediction: {int((~missing).sum())} "
          f"| missing/gate-dropped: {int(missing.sum())}")

    full = df.copy()
    full["pred"] = full[PRED_LABEL_COL].fillna(MISSING_PRED)
    print("  gold x raw pipeline label (full view, missing filled):")
    for line in pd.crosstab(full["gold"], full["pred"]).to_string().splitlines():
        print("    " + line)
    evaluate(full, name, "FULL", rows, ABSTAIN_FULL)

    matched = df[~missing].copy()
    matched["pred"] = matched[PRED_LABEL_COL]
    evaluate(matched, name, "MATCHED", rows, ABSTAIN_MATCHED)


def main():
    if not os.path.exists(GOLD_FILE):
        print(f"Missing gold file: {GOLD_FILE}")
        return
    gold = load_gold()
    print("gold labels:", sorted(gold["gold"].unique()))
    unmapped = sorted(set(gold["gold"].unique()) - set(GOLD_MAP))
    if unmapped:
        n_unmapped = int(gold["gold"].isin(unmapped).sum())
        print(f"gold labels with no mapping (excluded from scoring): {unmapped} "
              f"-> {n_unmapped} video(s)")

    rows = []
    for name, path in MODEL_FILES.items():
        evaluate_model(name, path, gold, rows)

    table = pd.DataFrame(rows, columns=[
        "model", "view", "n_scored", "n_excluded", "accuracy",
        "precision_misinfo", "recall_misinfo", "f1_misinfo", "macro_f1"])
    print("\n==================== METRICS TABLE ====================")
    print(table.to_string(index=False))

    os.makedirs(os.path.dirname(METRICS_CSV), exist_ok=True)
    table.to_csv(METRICS_CSV, index=False, encoding="utf-8")
    print(f"\nWrote {METRICS_CSV}")
    return table


if __name__ == "__main__":
    main()
