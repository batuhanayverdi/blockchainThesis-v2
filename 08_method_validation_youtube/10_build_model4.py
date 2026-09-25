"""
Build Model 4 (cascade): Model 3 leads, Model 1 fills 'unverified'
==================================================================
Per video: take Model 3's label, EXCEPT where Model 3 returned 'unverified', in
which case substitute Model 1's label (the title verdict). Model 3 is never
overridden when it commits, so this only raises coverage, it never replaces a
committed full-text call. Where Model 3 is unverified and Model 1 has no row, or
Model 1 is also unverified, the result stays unverified.

The output has the same shape as the other posts.csv files (doc_id + final_label),
so you can drop it straight into compare_model.py. A `label_source` column records
where each label came from, and `accuracy_ordinal` is blanked on fallback rows
since Model 1 has no ordinal.

Reads:  Model 3 posts.csv (primary), Model 1 posts.csv (fallback)
Writes: model4/posts.csv

Run:  python build_model4.py
Deps: pip install pandas
"""

import os
import pandas as pd


# ============================ CONFIG ========================================
MODEL3_FILE = r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_pipeline\3. model - full transkript (6 verdict)\pilot_outputs\model3\posts.csv"
MODEL1_FILE = r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_pipeline\1. model - just title\pilot_outputs\model1\posts.csv"
OUTPUT_FILE = r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_pipeline\4. model - cascade 3+1\pilot_outputs\model4\posts.csv"

ID_COL = "doc_id"
LABEL_COL = "final_label"
FALLBACK_TRIGGER = {"unverified"}   # Model 3 labels that hand off to Model 1


def main():
    if not os.path.exists(MODEL3_FILE):
        print(f"Missing {MODEL3_FILE}")
        return
    if not os.path.exists(MODEL1_FILE):
        print(f"Missing {MODEL1_FILE}")
        return
    m3 = pd.read_csv(MODEL3_FILE, dtype=str).fillna("")
    m1 = pd.read_csv(MODEL1_FILE, dtype=str).fillna("")
    m1_label = {str(r[ID_COL]).strip(): str(r[LABEL_COL]).strip()
                for _, r in m1.iterrows()}

    rows = []
    n_m3 = n_fb = n_no_fb = 0
    fb_breakdown = {}
    for _, r in m3.iterrows():
        row = dict(r)
        vid = str(r[ID_COL]).strip()
        lab3 = str(r[LABEL_COL]).strip()

        if lab3 in FALLBACK_TRIGGER and vid in m1_label:
            lab4 = m1_label[vid]
            source = "model1_fallback"
            n_fb += 1
            fb_breakdown[lab4] = fb_breakdown.get(lab4, 0) + 1
            if "accuracy_ordinal" in row:
                row["accuracy_ordinal"] = ""      # Model 1 has no ordinal
        elif lab3 in FALLBACK_TRIGGER:
            lab4 = lab3
            source = "model3_unverified_no_model1_row"
            n_no_fb += 1
        else:
            lab4 = lab3
            source = "model3"
            n_m3 += 1

        row[LABEL_COL] = lab4
        row["label_source"] = source
        rows.append(row)

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"Model 4 written: {len(out)} videos -> {OUTPUT_FILE}")
    print(f"  kept from Model 3            : {n_m3}")
    print(f"  filled by Model 1 (fallback) : {n_fb}")
    print(f"  unverified, no Model 1 row   : {n_no_fb}")
    if fb_breakdown:
        print("  what the fallback supplied   :")
        for k, v in sorted(fb_breakdown.items(), key=lambda kv: -kv[1]):
            tag = " (still unverified, no rescue)" if k in FALLBACK_TRIGGER else ""
            print(f"      {k}: {v}{tag}")
    print("\nFinal label distribution:")
    print(out[LABEL_COL].value_counts().to_string())
    print("\nNow add this path to compare_model.py MODEL_FILES and rerun it.")


if __name__ == "__main__":
    main()
