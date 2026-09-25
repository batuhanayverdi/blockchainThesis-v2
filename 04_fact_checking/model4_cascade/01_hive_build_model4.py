"""
Build HIVE MODEL 4 (cascade): Model 3 leads, Model 1 fills 'unverified'
=======================================================================
Per post: take Model 3's label, EXCEPT where Model 3 returned 'unverified', in
which case substitute Model 1's title verdict. Model 3 is never overridden when
it commits, so this only raises coverage. Identical logic to the validated
build_model4, repointed to the Hive outputs.

Reads:  hive_outputs/model3/posts.csv  (primary)
        hive_outputs/model1/posts.csv  (fallback)
Writes: hive_outputs/model4/posts.csv

The output keeps doc_id + final_label (+ a label_source column, and a blanked
accuracy_ordinal on fallback rows since Model 1 has no ordinal). Merge it onto
hive_sample_master.parquet by doc_id for the regression.

Run:  python hive_build_model4.py
Deps: pip install pandas
"""

import os
import pandas as pd


# ============================ CONFIG ========================================
MODEL3_FILE = os.path.join(r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_hive_pipeline\4.3 model full transkript (6 verdict)", "hive_outputs", "model3", "posts.csv")
MODEL1_FILE = os.path.join(r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_hive_pipeline\4.1-model (just title)", "hive_outputs", "model1", "posts.csv")
OUTPUT_FILE = os.path.join("hive_outputs", "model4", "posts.csv")

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
        did = str(r[ID_COL]).strip()
        lab3 = str(r[LABEL_COL]).strip()

        if lab3 in FALLBACK_TRIGGER and did in m1_label:
            lab4 = m1_label[did]
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

    print(f"Model 4 written: {len(out)} posts -> {OUTPUT_FILE}")
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
    print("\nNext: collapse to binary at the regression step and merge onto "
          "hive_sample_master.parquet by doc_id.")


if __name__ == "__main__":
    main()
