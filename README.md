# Rewarding Deception? — Study 1 (Hive) Code and Data

Code, intermediate data, and results for **Study 1** of the master's thesis
*"Rewarding Deception? The Impact of Tokenized Engagement on Misinformation
Diffusion in Blockchain-Based Social Media."*

- **Author:** Batuhan Ayverdi (batuhan.ayverdi@tum.de)
- **Supervisor:** Dr. Sara Alida Volkmer
- **Platform studied:** [Hive](https://hive.io), a blockchain-based social media platform
- **Observation window:** 1 September 2024 – 20 January 2025

The study asks whether posts containing misleading claims are rewarded with more
engagement (net votes, reblogs, payout) than accurate posts, and how emotional
language moderates that relationship.

---

## 1. Start here

If you only want to look at the output rather than run anything, these three
files give the fullest picture and open in a browser or Excel:

| File | What it shows |
|---|---|
| [`results/r-studio-result/hive_regression (4).html`](results/r-studio-result/) | The complete rendered regression analysis — every table, figure, and diagnostic in one scrollable document. **This is the main result.** |
| [`01_data_collection/hive_review.xlsx`](01_data_collection/) | Post-by-post inspection workbook. One row per post with the model's verdict, every extracted claim, the search evidence, and the justification. Colour-coded, sortable. Best way to judge whether the fact-checking pipeline is doing something sensible. |
| [`07_human_validation/human-validation_app.html`](07_human_validation/) | The blind annotation interface used by the two human raters. Opens offline in any browser; you can click through it exactly as the raters did. |

Everything else in this README explains how those were produced.

---

## 2. Repository layout

Folders are numbered in pipeline order. Within each folder, files are numbered in
execution order.

```
thesis_code_for_alida/
├── 01_data_collection/          Scraping Hive + topic classification
├── 02_filtering_and_cleaning/   Language filter + text cleaning
├── 03_claim_gate_and_sampling/  ClaimBuster gate + stratified sampling
├── 04_fact_checking/            LLM fact-checking (3 model variants)
│   ├── model1_title_only/       Title treated as the claim
│   ├── model3_full_text/        Full body, 6-class verdict
│   └── model4_cascade/          Model 3 lead, Model 1 fallback  ← used in thesis
├── 05_features/                 Emotion, VAD, engagement, author standing, ...
├── 06_analysis/                 Regression dataset build + Quarto analysis
├── 07_human_validation/         Human coding of 100 posts against the model
├── 08_method_validation_youtube/ Method validation on a YouTube gold-label set
├── 09_figures/                  Thesis figures
├── results/                     All analysis output (tables, figures, metrics)
└── MANIFEST.csv                 Maps every file here to its original location
                                 in the working project tree
```

`MANIFEST.csv` exists because the working project used descriptive folder names
(`"4.3. model full transkript (6 verdict)"`, `"17. NRC_VAD"`, …). Scripts still
contain those original paths internally — see [§6](#6-known-gaps-and-caveats).

---

## 3. The pipeline

Sample sizes below match Figure 1 of the thesis
([`09_figures/01_figure1_pipeline.py`](09_figures/01_figure1_pipeline.py)).

### Stage 01 — Data collection

| Script | Reads | Writes |
|---|---|---|
| `01_0-hive-scraping-everything.py` | HiveSQL (`vip.hivesql.io`) | weekly parquet chunks → `hive_full_posts_full.parquet` |
| `02_0-classify-full-dataset.py` | `hive_full_posts_full.parquet` | **`hive_classified.parquet`** ✓ *included* |

All Hive posts in the window longer than 100 characters: **n = 331,178**.
Each post is then labelled political/non-political plus one of 18 content
categories by `gpt-4o-mini`.

### Stage 02 — Filtering and cleaning

| Script | Reads | Writes |
|---|---|---|
| `01_1-hive_language_filter_90.py` | `hive_classified.parquet` | `hive_english_filtered.parquet` |
| `02_2-hive_cleaning.py` | `hive_english_filtered.parquet` | `hive_cleaned.parquet` |
| `03_2-spot_check_overcleaned.py` | `hive_cleaned.parquet` | inspection samples |

fastText keeps posts that are ≥ 90 % English by character count: **n = 200,821**.
Cleaning strips footers, boilerplate, code blocks, tables, mention spam, and
crypto tickers into the column `body_for_analysis`, which is the single text
basis used by *both* the fact-checking and the emotion pipelines.

The spot-check script exists to verify the cleaner removes boilerplate rather
than prose — a deliberate audit step, not a dead end.

### Stage 03 — Claim gate and sampling

| Script | Reads | Writes |
|---|---|---|
| `01_3-hive_claimbuster_gate.py` | `hive_cleaned.parquet` | `hive_cb_scores.parquet` |
| `02_hive_sampler.py` | `hive_cb_scores` + `hive_cleaned` | **`hive_sample_master.parquet`** ✓ *included* |
| `03_visualize_sample.py` | `hive_sample_master.parquet` | sampling verification charts |

ClaimBuster (DeBERTa-v2) scores every sentence; the post keeps its **maximum**
sentence score, so the 0.85 threshold can be re-tuned without re-scoring.
Posts above threshold: **n = 43,277**.

The sample takes **all 3,121 political** posts above the gate, plus an equal
number of non-political posts, month-matched to the political distribution and
spread across 7 non-political categories (seed 42). **n = 6,242**.

> `hive_sample_master.parquet` is the backbone of the project — six later
> scripts read it. It is included here (30 MB) and currently sits in
> `01_data_collection/` for convenience, although it is produced at Stage 03.

### Stage 04 — Fact-checking

Three model variants, each a four-stage chain. Stage 1 (the gate) is shared.

```
Stage 2  claim selection + query generation   (gpt-5-mini)
Stage 3  web evidence                          (Google via Serper API)
Stage 4  verdict + stance                      (gpt-5-mini)
```

| Variant | Claim unit | Verdict scale |
|---|---|---|
| **Model 1** (`model1_title_only/`) | The post title | 3-class (true / false / unverified) |
| **Model 3** (`model3_full_text/`) | Up to 3 claims from the body | 6-class (`pants_on_fire` … `true`, + `unverified`) |
| **Model 4** (`model4_cascade/`) | Model 3, with Model 1 filling `unverified` | 6-class + fallback |

**Model 4 is the variant used in the thesis.** Model 3 is never overridden when
it commits, so the cascade only raises coverage.

Two decisions matter for interpretation:

- **Stance gating.** A false-side verdict counts as misinformation only when the
  post *asserts* the claim. Quoted or questioned claims are excluded and counted
  separately in the audit.
- **No binary collapse at this stage.** The 6-class label and its ordinal are
  preserved; collapsing to binary happens at the regression step, where
  `unverified` is an abstention and is dropped rather than coded as accurate.

Model 3 also ships a **Batch API** submit/collect pair (`04_`/`05_`) that is
byte-identical in request body to the synchronous script and shares the same
checkpoint file. It exists to cut cost on the full run; either path yields the
same output.

`02_build_hive_review_excel.py` produces `hive_review.xlsx` — the inspection
workbook listed in [§1](#1-start-here).

**Analysis sample:** posts with at least one asserted claim carrying a verdict —
**n = 5,245 posts by 1,650 authors**.
Independent variable: count of asserted false-side claims per post (0, 1, 2, 3).

### Stage 05 — Feature construction

All keyed on `doc_id` (= `author/permlink`).

| # | Script | Produces | Used in thesis? |
|---|---|---|---|
| 01 | `01_5-hive_emotion_scores.py` | 6 emotions, post level (DistilRoBERTa) | ✅ H2 main |
| 02 | `02_6-hive_vad_scores.py` | Valence / arousal / dominance (NRC-VAD lexicon) | ✅ H2 robustness |
| 03 | `03_7-vad_spot_check.py` | VAD preprocessing audit | ✅ appendix |
| 04 | `04_hive_extract_engagement_v2.py` | Extended engagement from HiveSQL | ✅ outcomes |
| 05 | `05_9a-hive_sentence_emotion_scoring.py` | Sentence-level emotions (pysbd) | ✅ feeds 06 |
| 06 | `06_9b-hive_emotion_dynamics.py` | Volatility / SD / mean per emotion | ✅ H2 dynamics |
| 07 | `07_hive_political_identity_v2.py` | Political identity counts (Rathje et al. 2021) | ⚠️ exploratory only |
| 08 | `08_hive_doc_outrage_scoring_v3.py` | Moral outrage (Brady et al. 2021 DOC) | ⚠️ exploratory only |
| 09 | `09_hive_post_features_v2.py` | Images, links, tags, posting time | ✅ controls |
| 10 | `10_hive_author_standing.py` | Pre-window author standing | ✅ controls |

Two measurement choices worth noting:

- **VAD scores are `NaN`, not `0.0`, when no lexicon token matches.** Zero is a
  valid position on the 0–1 scale (maximally negative valence), so coding
  unmeasured posts as zero would inject extreme false values into the
  regression.
- **Author standing is reconstructed strictly pre-window.** HiveSQL state tables
  hold *current* (2026) values and are temporally invalid for a 2024–25 window.
  Standing is therefore summed from append-only operation tables with
  `timestamp < 2024-09-01`. Reputation is deliberately *not* reconstructed (it
  is recursive with no historical snapshot); cumulative author rewards serve as
  the proxy.

Scripts 07 and 08 were computed but **do not enter any reported model** — they
are exploratory branches kept for transparency.

### Stage 06 — Analysis

| Script | Reads | Writes |
|---|---|---|
| `01_build_regression_dataset_final.py` | Stage 04 verdicts + all Stage 05 features | **`hive_regression_dataset_main.parquet`** ✓ *included* |
| `02_hive_regression (4).qmd` | `hive_regression_dataset_main.parquet` | everything in `results/r-studio-result/` |
| `03_0-follower_probe_v4.py` | HiveSQL | feasibility probe (see below) |

The Quarto document is the analysis of record: `fixest` models with HC1 robust
standard errors, seed `20260101`. It also reads
`hive_emotion_dynamics.parquet` for the H2 dynamics section.

**The follower probe is a negative result, kept on purpose.** Follower counts
exist on Hive only as ~147 million `custom_json` operations with no index;
per-author scans extrapolate to roughly 1,171 hours. The probe documents why
author *popularity* is operationalised as pre-window reward per post instead of
follower count.

### Stage 07 — Human validation

100 posts (20 per accuracy category) drawn from the analysis sample, split
50/50 across two raters, presented blind — the annotation page carries only post
id, title, author, date, body, and highlighted claim sentences. No model label,
verdict, stance, or justification is ever rendered.

| Metric (pooled, n = 100) | Value |
|---|---|
| Exact category match | 0.360 |
| Within one category | 0.813 |
| Binary agreement | 0.709 |
| Binary Cohen's κ | 0.445 |
| Binary precision | 0.593 |
| Binary recall | 0.914 |

The two raters saw **disjoint** sets, so this is model–human agreement, not
inter-rater reliability; no IRR is computed and none is claimed.

Output: `results/human_validation/`.

### Stage 08 — Method validation (YouTube)

Before the Hive run, the same four-stage architecture was benchmarked against a
YouTube dataset with **gold labels**, to establish that the pipeline detects
misinformation at a defensible rate.

Scored under two views: **FULL** (end-to-end, gate failures visible) and
**MATCHED** (verdict quality on committed cases only). Abstentions are excluded
rather than scored as errors.

| Model | View | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Model 1 (title) | MATCHED | 0.875 | 0.981 | 0.729 | 0.837 |
| Model 3 (full, 6-class) | MATCHED | 0.739 | 0.761 | 0.637 | 0.693 |
| **Model 4 (cascade)** | **MATCHED** | **0.744** | **0.771** | **0.643** | **0.701** |

Full table: [`08_method_validation_youtube/model_comparison_metrics.csv`](08_method_validation_youtube/model_comparison_metrics.csv).
Model 4 was selected for the Hive run: it has the lowest abstention count
(4 excluded vs 20 for Model 3) at comparable verdict quality.

### Stage 09 — Figures

`01_figure1_pipeline.py` draws the thesis pipeline diagram.
`02_figure4-1_r2_ladder.py` draws the R² ladder showing incremental explanatory
power (0.016 → 0.891 across five specifications), of which +0.286 comes from the
four author-standing measures.

---

## 4. Data files

### Included

| File | Location | Size | Rows |
|---|---|---|---|
| `hive_classified.parquet` | `01_data_collection/` | 2.29 GB | 331,178 |
| `hive_sample_master.parquet` | `01_data_collection/` | 30 MB | 6,242 |
| `hive_review.xlsx` | `01_data_collection/` | 42 MB | — |
| `hive_regression_dataset_main.parquet` | `06_analysis/` | 1.2 MB | 6,242 × 71 cols |
| `hive_regression_dataset_main.xlsx` | `06_analysis/` | 2.5 MB | + audit sheet |
| `master_key.xlsx` | `07_human_validation/` | 17 KB | validation answer key |
| `human-validation_app.html` | `07_human_validation/` | 356 KB | annotation interface |
| `model_comparison_metrics.csv` | `08_method_validation_youtube/` | 625 B | benchmark results |
| NRC-VAD lexicon | `results/vad_check/` | 518 KB | see licence note below |

**`hive_regression_dataset_main.parquet` is self-sufficient** — the Quarto
document runs from it alone (plus `hive_emotion_dynamics.parquet` for the
dynamics section). Nothing upstream needs to be re-run to reproduce the tables.

### Not included

Intermediate artefacts, omitted for size or because they are reconstructible:

`hive_full_posts_full.parquet` · `hive_english_filtered.parquet` ·
`hive_cleaned.parquet` (1.4 GB) · `hive_cb_scores.parquet` ·
`hive_outputs/model{1,3}/` raw search + verdict JSON (168 MB / 666 MB) ·
`doc_id.xlsx` · `hive_emotion_scores.parquet` · `hive_vad_scores.parquet` ·
`hive_sentence_emotions.parquet` · `hive_emotion_dynamics.parquet` ·
`hive_engagement_extended.parquet` · `hive_post_features.parquet` ·
`hive_author_standing.parquet`

Third-party resources not redistributed here: the Rathje et al. (2021) political
identity dictionaries (OSF), and the Brady et al. (2021) DOC model weights.

Available on request — most are only a few MB.

> **Licence note.** The NRC-VAD Lexicon (Mohammad, 2018) is licensed for
> non-commercial research use and **must not be redistributed**. It is included
> here for supervision purposes only; remove it before any public release of
> this repository.

---

## 5. Running the code

### Environment

Python 3.9–3.11. No 3.10+ syntax is used.

```bash
pip install pandas numpy pyarrow openpyxl tqdm matplotlib
pip install torch transformers          # emotion + ClaimBuster scoring
pip install openai requests             # fact-checking stages
pip install pyodbc                      # HiveSQL access (Windows)
pip install nltk pysbd fasttext-langdetect
```

The R analysis needs Quarto plus `tidyverse`, `arrow`, `fixest`,
`modelsummary`, `gt`, `scales`, `ggcorrplot`, `broom`, `forcats`.

Two components need special environments:

- **`05_features/08_hive_doc_outrage_scoring_v3.py`** requires a *separate*
  virtualenv with legacy TensorFlow (`tf-keras`, `keras_preprocessing`,
  `emoji==1.7.0`). Old TF conflicts with the torch/transformers stack. This
  script feeds no reported model, so it can be skipped entirely.
- **`03_claim_gate_and_sampling/01_3-hive_claimbuster_gate.py`** was run on a
  Kaggle T4 GPU and auto-detects `/kaggle/working`. It runs on CPU but slowly.

### Credentials

All API and database credentials are read from environment variables. Set the
ones you need before running:

```bash
export OPENAI_API_KEY="..."      # stages 01 (classification), 04 (fact-checking)
export SERPER_API_KEY="..."      # stage 04 (web evidence)
export HIVESQL_PASSWORD="..."    # stages 01, 05 (HiveSQL)
export HIVESQL_USER="..."        # HiveSQL login (carries a "Hive-" prefix)
```

HiveSQL registration is free at [hivesql.io](https://hivesql.io).

**Cost warning:** re-running Stage 04 on all 6,242 posts means tens of thousands
of `gpt-5-mini` calls plus Serper queries. Stage 01 classification over 331,178
posts costs roughly $2–3 on `gpt-4o-mini`. The included
`hive_regression_dataset_main.parquet` makes all of this unnecessary for
reproducing the results.

---

## 6. Known gaps and caveats

Listed openly so they are not mistaken for oversights.

**Paths are absolute.** Roughly 30 scripts contain hard-coded
`C:\Users\...\fact_checked_hive_pipeline\...` paths at the top of their config
block, reflecting the working project tree. They must be edited before any
script will run elsewhere. `MANIFEST.csv` maps every file to its original
location, which makes the mapping mechanical.

**`doc_id.xlsx` is not produced by any script here.** Five Stage 05 scripts read
it as a filter listing the analysis-sample `doc_id`s. It was exported manually
from the sample master during the working run. It can be regenerated from
`hive_sample_master.parquet`.

**Stage 05 script 04 reads a Stage 06 output.** `04_hive_extract_engagement_v2.py`
points at an earlier `hive_regression_dataset.xlsx`, so the numbering does not
reflect true dependency order at that one point. It also has a missing `pyodbc`
import.

**Model 2 scripts are absent.** `08_method_validation_youtube/13_compare_model.py`
and the metrics CSV report a "Model 2 (full, 3-class)" variant. Model 2 was the
3-class full-text predecessor of Model 3; its scripts are not in this bundle
because it was superseded and is reported only as a benchmark comparison.

**The YouTube branch is upstream-incomplete.** Transcript acquisition and the
gold-label workbook (`Final Scrap - Final Dataset.xlsx`) sit outside this
bundle; Stage 08 begins from already-collected transcripts.

**Figure values are hard-coded.** `09_figures/02_figure4-1_r2_ladder.py` contains
the five R² values as literals, transcribed from the Quarto output rather than
read from it. The figure is a presentation artefact, not a computation.

**Filenames retain legacy numbering.** `03_2-spot_check…`, `01_3-claimbuster…`,
`05_9a-…` carry both the new pipeline order (prefix) and the original working
number (suffix). When the two disagree, **the prefix is authoritative**.

**Posts are editable on Hive.** Metadata reflects each post. The scrape sits at the
window's end, which is the closest historically valid snapshot available.

**Personal data.** Hive is a public blockchain and all author names and post
permalinks in these files are public record. They are nonetheless real
identifiers and should be handled accordingly.

---

## 7. Where the thesis results come from

| Thesis element | Source |
|---|---|
| Figure 1 (pipeline) | `09_figures/01_figure1_pipeline.py` |
| Figure 4.1 (R² ladder) | `09_figures/02_figure4-1_r2_ladder.py` |
| Table 4.1 (descriptives) | `results/r-studio-result/{html,pdf}_tables/Table_4_1_*` |
| H1 (misleading claims → engagement) | `Table_1A_H1_net_votes`, `Table_1B_H1_any_reblog` |
| H2 (emotion moderation) | `Table_2_*` (net votes), `Table_3_*` (reblogs) |
| H2 robustness — VAD | `Table_6A`–`Table_6D` |
| H2 robustness — dynamics | `Table_7A`–`Table_7E` |
| Author popularity / standing | `Table_4A`–`Table_4C`, `Table_5A`–`Table_5B` |
| Robustness — title, payout | `Table_R1A`–`Table_R3` |
| Descriptive t-tests | `Table_D1*`, `Table_D2*` |
| Diagnostic figures | `results/r-studio-result/figures_{png,pdf}/D1`–`D8` |
| Human validation | `results/human_validation/` |
| Method validation | `08_method_validation_youtube/model_comparison_metrics.csv` |
| Sample descriptives | `results/descriptives/` |

---

## References

- Berger, J., Rocklage, M. D., & Packard, G. (2021). Expression modalities and emotional dynamics.
- Brady, W. J., McLoughlin, K., Doan, T. N., & Crockett, M. J. (2021). Digital Outrage Classifier.
- Hartmann, J. (2022). `emotion-english-distilroberta-base`.
- Mohammad, S. M. (2018). Obtaining reliable human ratings of valence, arousal, and dominance for 20,000 English words. *ACL 2021*.
- Rathje, S., Van Bavel, J. J., & van der Linden, S. (2021). Out-group animosity drives engagement on social media. *PNAS*.
