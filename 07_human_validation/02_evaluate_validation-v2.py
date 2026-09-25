"""
Human validation summary, two raters
====================================
Reads master_key.xlsx and both rater exports, merges them, and prints the
agreement numbers. The two raters reviewed disjoint sets of 50 posts each,
so this is model-human agreement over 100 posts. No inter-rater
reliability is computed, because no post was seen by both raters.

Writes into results/:
  validation_merged.csv       one row per rated post
  validation_summary.csv      pooled and per-rater headline numbers
  validation_confusion.csv    5x5 model label by rater label

Run:  python evaluate_validation.py
Deps: pip install pandas numpy openpyxl
"""

import os
import csv
import glob
import numpy as np
import pandas as pd

# ============================ CONFIG ========================================
BASE = r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_hive_pipeline\8. human validation"
KEY_XLSX = os.path.join(BASE, "master_key.xlsx")
OUT_DIR = os.path.join(BASE, "results")
RATER_GLOB = os.path.join(OUT_DIR, "validation_rater*.csv")

CATEGORIES = ["pants_on_fire", "false", "mostly_false", "mostly_true", "true"]
ORD = {c: i for i, c in enumerate(CATEGORIES)}
MISLEADING = {"pants_on_fire", "false", "mostly_false"}
UNVERIFIED = "unverified"

# Excel rewrites the export: "true"/"false" become TRUE/FALSE (or WAHR/FALSCH
# in a German locale) and the delimiter becomes ";". Both are reversed here.
LABEL_FIX = {
    "true": "true", "TRUE": "true", "True": "true", "WAHR": "true", "wahr": "true",
    "false": "false", "FALSE": "false", "False": "false",
    "FALSCH": "false", "falsch": "false",
    "pants_on_fire": "pants_on_fire", "mostly_false": "mostly_false",
    "mostly_true": "mostly_true", "unverified": "unverified",
}
BOOL_FIX = {"true": True, "TRUE": True, "True": True, "WAHR": True, "1": True,
            "false": False, "FALSE": False, "False": False, "FALSCH": False,
            "0": False}


# ============================ LOADING =======================================
def read_rater(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        head = fh.read(4096)
    try:
        sep = csv.Sniffer().sniff(head, delimiters=",;\t").delimiter
    except csv.Error:
        sep = ";" if head.count(";") > head.count(",") else ","

    df = pd.read_csv(path, sep=sep, dtype=str, encoding="utf-8-sig").fillna("")
    df.columns = [c.strip() for c in df.columns]
    raw = df["human_label"].astype(str).str.strip()
    df["human_label"] = raw.map(LABEL_FIX)
    bad = raw[df["human_label"].isna()].unique().tolist()
    if bad:
        print(f"  WARNING [{os.path.basename(path)}]: unrecognised labels {bad}, "
              f"those rows dropped.")
        df = df[df["human_label"].notna()].copy()
    if "is_checkable_claim" in df.columns:
        df["is_checkable_claim"] = (df["is_checkable_claim"].astype(str)
                                    .str.strip().map(BOOL_FIX))
    for c in ("seconds_spent", "order_shown"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["post_id"] = df["post_id"].astype(str).str.strip()
    df["source_file"] = os.path.basename(path)
    print(f"  Read {os.path.basename(path)}: {len(df)} rows (delimiter '{sep}').")
    return df


# ============================ STATISTICS ====================================
def binary_kappa(model, human):
    """Cohen's kappa for the misleading vs accurate split.

    Kept because raw agreement on a sample that is balanced by design can
    look higher than it is. Delete this and the b_kappa line below if you
    do not want to report it.
    """
    n = len(model)
    if n == 0:
        return np.nan
    po = float(np.mean(model == human))
    pe = 0.0
    for lab in ("misleading", "accurate"):
        pe += float(np.mean(model == lab)) * float(np.mean(human == lab))
    return np.nan if pe == 1 else (po - pe) / (1 - pe)


def summarise(df, label):
    """Headline numbers for one group of rows."""
    ok = df[df["human_label"] != UNVERIFIED]
    okb = df[df["human_bin"] != "unverified"]
    m, h = okb["model_bin"].to_numpy(), okb["human_bin"].to_numpy()

    tp = int(((m == "misleading") & (h == "misleading")).sum())
    fp = int(((m == "misleading") & (h == "accurate")).sum())
    fn = int(((m == "accurate") & (h == "misleading")).sum())
    tn = int(((m == "accurate") & (h == "accurate")).sum())

    return {
        "group": label,
        "n_posts": len(df),
        "n_unverified": int((df["human_label"] == UNVERIFIED).sum()),
        "exact_match": float((ok["model_label"] == ok["human_label"]).mean()),
        "within_one": float((ok["abs_err"] <= 1).mean()),
        "signed_difference": float((ok["human_ord"] - ok["model_ord"]).mean()),
        "binary_agreement": (tp + tn) / len(okb) if len(okb) else np.nan,
        "binary_kappa": binary_kappa(m, h),
        "binary_precision": tp / (tp + fp) if (tp + fp) else np.nan,
        "binary_recall": tp / (tp + fn) if (tp + fn) else np.nan,
        "checkable_share": (float(df["is_checkable_claim"].fillna(False).mean())
                            if "is_checkable_claim" in df.columns else np.nan),
    }


def show(row):
    print(f"\n  {row['group']}  (n = {row['n_posts']}, "
          f"unverified = {row['n_unverified']})")
    print(f"    Same label as the model      {row['exact_match']:.0%}")
    print(f"    Within one category          {row['within_one']:.0%}")
    print(f"    Signed difference            {row['signed_difference']:+.2f} "
          f"(positive = rater judged the post more accurate)")
    print(f"    Misleading vs accurate       {row['binary_agreement']:.0%} "
          f"agreement, kappa {row['binary_kappa']:.2f}")
    print(f"      model flags, rater agrees  {row['binary_precision']:.0%}")
    print(f"      rater flags, model caught  {row['binary_recall']:.0%}")
    if not np.isnan(row["checkable_share"]):
        print(f"    Confirmed checkable claim    {row['checkable_share']:.0%}")


# ============================ MAIN ==========================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    key = pd.read_excel(KEY_XLSX, dtype=str).fillna("")
    key["post_id"] = key["post_id"].astype(str).str.strip()
    key["model_label"] = key["model_label"].astype(str).str.strip()
    print(f"  Read master_key.xlsx: {len(key)} rows.")

    paths = sorted(glob.glob(RATER_GLOB))
    if not paths:
        raise SystemExit(f"No rater files matched {RATER_GLOB}")
    hum = pd.concat([read_rater(p) for p in paths], ignore_index=True)

    df = hum.merge(key, on="post_id", how="left", suffixes=("", "_key"))
    missing = df["model_label"].isna() | (df["model_label"] == "")
    if missing.any():
        print(f"  WARNING: {int(missing.sum())} rated posts absent from the key, "
              f"dropped.")
        df = df[~missing].copy()

    overlap = int(df["post_id"].duplicated().sum())
    print(f"  Posts rated by more than one rater: {overlap}")

    # Rater name comes from the file, not from the button the rater clicked.
    names = {f: f"Rater {i + 1}" for i, f in enumerate(sorted(df["source_file"].unique()))}
    if "rater" in df.columns:
        for f, sub in df.groupby("source_file"):
            vals = sub["rater"].astype(str).str.strip()
            if vals.any():
                names[f] = vals.mode().iloc[0]
    if len(set(names.values())) < len(names):
        names = {f: f for f in names}          # names collided, use file names
    df["rater_id"] = df["source_file"].map(names)

    if "rater" in df.columns and "rater_key" in df.columns:
        mism = (df["rater"].astype(str).str.strip()
                != df["rater_key"].astype(str).str.strip())
        if mism.any():
            print(f"  WARNING: {int(mism.sum())} posts answered under a different "
                  f"rater name than assigned in the key. Check before pooling.")

    df["model_ord"] = df["model_label"].map(ORD)
    df["human_ord"] = df["human_label"].map(ORD)
    df["abs_err"] = (df["human_ord"] - df["model_ord"]).abs()
    df["model_bin"] = np.where(df["model_label"].isin(MISLEADING),
                               "misleading", "accurate")
    df["human_bin"] = np.where(df["human_label"].isin(MISLEADING),
                               "misleading", "accurate")
    df.loc[df["human_label"] == UNVERIFIED, "human_bin"] = "unverified"

    # ---- summary ----
    rows = [summarise(df, "Pooled")]
    for name, sub in df.groupby("rater_id", sort=True):
        rows.append(summarise(sub, name))
    for r in rows:
        show(r)

    conf = (pd.crosstab(df["model_label"], df["human_label"])
            .reindex(index=CATEGORIES, columns=CATEGORIES + [UNVERIFIED])
            .fillna(0).astype(int))
    print("\n  Model label (rows) by rater label (columns)\n")
    print(conf.to_string())

    worst = (df[(df["human_label"] != UNVERIFIED) & (df["abs_err"] >= 2)]
             .sort_values("abs_err", ascending=False)
             [["abs_err", "model_label", "human_label", "rater_id", "post_id"]])
    print(f"\n  Disagreements of two categories or more: {len(worst)}")
    if len(worst):
        print(worst.head(10).to_string(index=False))

    # ---- files ----
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "validation_summary.csv"),
                              index=False, encoding="utf-8-sig")
    conf.to_csv(os.path.join(OUT_DIR, "validation_confusion.csv"),
                encoding="utf-8-sig")
    keep = [c for c in ["post_id", "rater_id", "model_label", "human_label",
                        "abs_err", "model_bin", "human_bin", "is_checkable_claim",
                        "n_claims", "order_shown", "seconds_spent", "title",
                        "author", "date"] if c in df.columns]
    df[keep].to_csv(os.path.join(OUT_DIR, "validation_merged.csv"),
                    index=False, encoding="utf-8-sig")
    print(f"\n  Wrote validation_summary.csv, validation_confusion.csv and "
          f"validation_merged.csv to {OUT_DIR}")


if __name__ == "__main__":
    main()
