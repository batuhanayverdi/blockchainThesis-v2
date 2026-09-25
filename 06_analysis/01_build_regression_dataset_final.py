"""
Build the MAIN post-level regression dataset for Study 1 (Hive)
================================================================

Purpose
-------
Creates one row per post for the simplified main regression analysis.

Main predictors:
  1. Body misinformation intensity as a categorical count:
       0, 1, 2, or 3 asserted misleading claims
  2. A separate title_misleading indicator from Model 1
  3. Six separate DistilRoBERTa emotion scores:
       anger, disgust, fear, joy, sadness, surprise

Main outcomes:
  - net_votes
  - any_reblog / reblogs

Main controls:
  - log_len
  - is_political
  - n_images
  - n_links
  - n_tags
  - is_weekend

Important rules:
  - Only claims with stance == "asserted" enter the main claim measures.
  - Quoted claims are excluded and counted for the audit.
  - unverified / blank verdicts are NOT coded as accurate.
  - The title is a separate predictor; it is not added as a fourth body claim.
  - One observation remains one post.

Outputs:
  - hive_regression_dataset_main.parquet
  - hive_regression_dataset_main.xlsx
      * data sheet
      * audit sheet
  - hive_regression_dataset_main_audit.csv

Run:
  python build_regression_dataset.py

Dependencies:
  pip install pandas numpy pyarrow openpyxl
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(
    r"C:\Users\batuh\PycharmProjects\Thesis\fact_checked_hive_pipeline"
)

# Claim-level fact-check outputs
# The current cascade folder is tried first. Older locations are kept only as
# fallbacks so the script survives folder renames.
CASCADE_DIR = BASE_DIR / "4.4. model cascade (model3,1)"

M3_JSON_CANDIDATES = [
    CASCADE_DIR / "hive_outputs" / "model3" / "stage4_factcheck.json",
    BASE_DIR / "4.3 model full transkript (6 verdict)" / "hive_outputs" / "model3" / "stage4_factcheck.json",
]

M1_JSON_CANDIDATES = [
    CASCADE_DIR / "hive_outputs" / "model1" / "stage4_factcheck.json",
    BASE_DIR / "4.1-model (just title)" / "hive_outputs" / "model1" / "stage4_factcheck.json",
]

# Post-level inputs
SAMPLE_FILE = (
    BASE_DIR
    / "hive sample preparation (politic-non-politic)"
    / "hive_sample_master.parquet"
)

EMOTION_FILE = BASE_DIR / "hive_emotion_scores.parquet"
VAD_FILE = BASE_DIR / "17. NRC_VAD" / "hive_vad_scores.parquet"

EXTENDED_FILE = (
    BASE_DIR
    / "7. hive new dataset"
    / "hive_engagement_extended.parquet"
)

POST_FEATURES_FILE = (
    BASE_DIR
    / "12. Post Features"
    / "hive_post_features.parquet"
)

OUTPUT_DIR = BASE_DIR / "6. regression"
OUTPUT_STEM = "hive_regression_dataset_main"


# =============================================================================
# ANALYSIS DEFINITIONS
# =============================================================================

FALSE_SIDE = {"pants_on_fire", "false", "mostly_false"}
TRUE_SIDE = {"mostly_true", "true"}
CLASSIFIED_VERDICTS = FALSE_SIDE | TRUE_SIDE

EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

# NRC-VAD lexicon dimensions (Mohammad, 2018). An alternative, model-free
# operationalization of emotional language used for the H2 robustness check.
VAD_VARS = ["valence", "arousal", "dominance"]

POST_FEATURES = ["n_images", "n_links", "n_tags", "is_weekend"]
PREWINDOW_STANDING_VARS = [
    "log_pre_rewards",
    "log_pre_curation",
    "log_pre_posts",
    "log_acct_age",
]

# Author popularity for the Eckert specifications: pre-window average author
# reward per pre-window root post, computed from the RAW standing columns.
# Follower counts were probed on HiveSQL and ruled out: follow relationships
# exist only as ~147 million custom_json operations, and per-author scans
# extrapolate to ~1,171 hours. The reward-based measure is therefore the
# popularity variable; the probe output documents this choice.
POPULARITY_VARS = [
    "pre_avg_reward",
    "log_pre_avg_reward",
    "pre_root_posts",
]

# Fallback candidates used only if the extended engagement file is unavailable.
VOTES_CANDS = ["net_votes", "vote_count", "num_votes"]
REBLOG_CANDS = ["reblogs", "reblog_count", "num_reblogs"]
COMMENTS_CANDS = ["comments", "children", "comment_count", "num_comments"]
UPVOTES_CANDS = ["upvotes", "up_votes", "positive_votes"]
DOWNVOTES_CANDS = ["downvotes", "down_votes", "negative_votes"]
DISTINCT_VOTERS_CANDS = ["distinct_voters", "unique_voters", "n_distinct_voters"]
PAYOUT_CANDS = [
    "payout_combined",
    "total_payout_value",
    "payout",
    "author_payout",
]

PREWINDOW_STANDING_FILE = (
    BASE_DIR
    / "12. Post Features"
    / "hive_author_standing.parquet"
)


try:
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
except Exception:  # pragma: no cover
    ILLEGAL_CHARACTERS_RE = re.compile(
        r"[\000-\010]|[\013-\014]|[\016-\037]"
    )


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def normalise(value: Any) -> str:
    """Return a lowercase, stripped string; missing values become an empty string."""
    if value is None:
        return ""
    return str(value).strip().lower()


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found:\n{path}")


def resolve_factcheck_path(model_name: str, candidates: list[Path]) -> Path:
    """Resolve a Model 1/3 stage4 JSON without silently choosing the wrong run."""
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = [
        path
        for path in BASE_DIR.rglob("stage4_factcheck.json")
        if path.parent.name.lower() == model_name.lower()
    ]

    if len(matches) == 1:
        return matches[0]

    tried = "\n".join(f"  - {path}" for path in candidates)

    if not matches:
        raise FileNotFoundError(
            f"Could not find {model_name} stage4_factcheck.json.\n"
            f"Tried these expected locations:\n{tried}\n\n"
            f"Also searched recursively under:\n  {BASE_DIR}"
        )

    found = "\n".join(f"  - {path}" for path in matches)
    raise RuntimeError(
        f"Found multiple possible {model_name} stage4_factcheck.json files.\n"
        f"Please keep the intended path first in its candidate list.\n{found}"
    )


def ensure_doc_id(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Guarantee a string doc_id column."""
    if "doc_id" in df.columns:
        out = df.copy()
        out["doc_id"] = out["doc_id"].astype(str).str.strip()
        return out

    if "post_key" in df.columns:
        out = df.copy()
        out["doc_id"] = out["post_key"].astype(str).str.strip()
        return out

    if {"author", "permlink"} <= set(df.columns):
        out = df.copy()
        out["doc_id"] = (
            out["author"].astype(str).str.strip()
            + "/"
            + out["permlink"].astype(str).str.strip()
        )
        return out

    raise KeyError(
        f"{name}: expected doc_id, post_key, or author + permlink. "
        f"Available columns: {list(df.columns)}"
    )


def assert_unique(df: pd.DataFrame, name: str) -> None:
    """Stop rather than silently creating a many-to-many merge."""
    duplicated = df["doc_id"].duplicated(keep=False)
    if duplicated.any():
        examples = df.loc[duplicated, "doc_id"].head(10).tolist()
        raise ValueError(
            f"{name}: doc_id is not unique. Example duplicate IDs: {examples}"
        )


def ensure_author_unique(df: pd.DataFrame, name: str) -> None:
    """Stop rather than silently creating a many-to-many merge at the author level."""
    if "author" not in df.columns:
        raise KeyError(f"{name}: expected an author column. Available columns: {list(df.columns)}")
    duplicated = df["author"].astype(str).str.strip().duplicated(keep=False)
    if duplicated.any():
        examples = df.loc[duplicated, "author"].astype(str).head(10).tolist()
        raise ValueError(
            f"{name}: author is not unique. Example duplicate authors: {examples}"
        )

def first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def to_num(series: pd.Series) -> pd.Series:
    """Parse numeric values, including strings such as '1.23 HBD'."""
    numeric = pd.to_numeric(series, errors="coerce")
    needs_parsing = numeric.isna() & series.notna()

    if needs_parsing.any():
        extracted = (
            series.loc[needs_parsing]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.extract(r"(-?\d+(?:\.\d+)?)")[0]
        )
        numeric.loc[needs_parsing] = pd.to_numeric(
            extracted, errors="coerce"
        )

    return numeric


def to01(value: Any) -> float:
    if pd.isna(value):
        return np.nan

    if isinstance(value, (bool, np.bool_)):
        return float(value)

    text = normalise(value)

    if text in {"1", "true", "t", "yes", "y"}:
        return 1.0
    if text in {"0", "false", "f", "no", "n"}:
        return 0.0

    try:
        return float(float(text) != 0)
    except (TypeError, ValueError):
        return np.nan


def coalesce_text(primary: pd.Series, fallback: pd.Series) -> pd.Series:
    """Use primary unless it is missing or blank."""
    primary_text = primary.astype("string")
    missing = primary_text.isna() | primary_text.str.strip().eq("")
    return primary_text.mask(missing, fallback.astype("string"))


def sanitize_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in out.columns:
        out[column] = out[column].map(
            lambda value: (
                ILLEGAL_CHARACTERS_RE.sub("", value)
                if isinstance(value, str)
                else value
            )
        )
    return out


# =============================================================================
# FACT-CHECK HELPERS
# =============================================================================

def load_factcheck(
    path: Path,
) -> tuple[dict[str, dict], dict[str, dict], dict[str, list[dict]]]:
    """Load post, document, and claim objects keyed by doc_id."""
    require_file(path, "Fact-check JSON")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    posts = {
        str(post["doc_id"]).strip(): post
        for post in payload.get("posts", [])
    }

    documents = {
        str(document["doc_id"]).strip(): document
        for document in payload.get("documents", [])
    }

    claims_by_doc: dict[str, list[dict]] = {}
    for claim in payload.get("claims", []):
        doc_id = str(claim.get("doc_id", "")).strip()
        if doc_id:
            claims_by_doc.setdefault(doc_id, []).append(claim)

    return posts, documents, claims_by_doc


def summarise_claims(claims: list[dict]) -> dict[str, int]:
    """
    Summarise claim status.

    Only asserted claims can contribute to the misinformation count.
    Every explicit non-asserted stance is excluded as a quoted/attributed
    claim. Missing stance values are excluded and audited separately.
    """
    asserted = [
        claim
        for claim in claims
        if normalise(claim.get("stance")) == "asserted"
    ]

    # The pipeline may use a non-asserted label other than the literal
    # string "quoted" (for example "quote", "not_asserted", etc.).
    # For the main analysis, every explicit non-empty stance other than
    # "asserted" is excluded and counted here.
    quoted = [
        claim
        for claim in claims
        if normalise(claim.get("stance")) not in {"", "asserted"}
    ]

    # Missing/blank stance is kept separate so it is never silently described
    # as a quote. These claims are also excluded from the asserted-only sample.
    other_nonasserted = [
        claim
        for claim in claims
        if normalise(claim.get("stance")) == ""
    ]

    asserted_misleading = [
        claim
        for claim in asserted
        if normalise(claim.get("verdict")) in FALSE_SIDE
    ]

    asserted_nonmisleading = [
        claim
        for claim in asserted
        if normalise(claim.get("verdict")) in TRUE_SIDE
    ]

    asserted_classified = [
        claim
        for claim in asserted
        if normalise(claim.get("verdict")) in CLASSIFIED_VERDICTS
    ]

    asserted_unverified = [
        claim
        for claim in asserted
        if normalise(claim.get("verdict")) not in CLASSIFIED_VERDICTS
    ]

    # These additional counts make the asserted-only sample loss transparent.
    # A claim is classified here regardless of stance, which reproduces the
    # pre-stance eligibility concept without reintroducing non-asserted claims
    # into the final misinformation measure.
    classified_any_stance = [
        claim
        for claim in claims
        if normalise(claim.get("verdict")) in CLASSIFIED_VERDICTS
    ]
    classified_explicit_nonasserted = [
        claim
        for claim in claims
        if normalise(claim.get("stance")) not in {"", "asserted"}
        and normalise(claim.get("verdict")) in CLASSIFIED_VERDICTS
    ]
    classified_blank_stance = [
        claim
        for claim in claims
        if normalise(claim.get("stance")) == ""
        and normalise(claim.get("verdict")) in CLASSIFIED_VERDICTS
    ]

    return {
        "n_claims_total": len(claims),
        "n_asserted_claims": len(asserted),
        "n_quoted_claims": len(quoted),
        "n_other_nonasserted_claims": len(other_nonasserted),
        "n_classified_claims_any_stance": len(classified_any_stance),
        "n_classified_explicit_nonasserted_claims": len(
            classified_explicit_nonasserted
        ),
        "n_classified_blank_stance_claims": len(classified_blank_stance),
        "n_asserted_classified_claims": len(asserted_classified),
        "n_asserted_unverified_claims": len(asserted_unverified),
        "n_asserted_misleading_claims": len(asserted_misleading),
        "n_asserted_nonmisleading_claims": len(asserted_nonmisleading),
    }


def derive_title_measure(claims: list[dict]) -> dict[str, Any]:
    """
    Build a separate title-level misinformation measure.

    If there is at least one asserted false-side title claim, title_misleading = 1.
    Otherwise, if there is at least one asserted true-side title claim,
    title_misleading = 0.
    Otherwise it remains missing.

    Model 1 normally returns one title claim, but this also behaves safely if
    multiple claims are present.
    """
    summary = summarise_claims(claims)

    misleading_n = summary["n_asserted_misleading_claims"]
    nonmisleading_n = summary["n_asserted_nonmisleading_claims"]

    if misleading_n > 0:
        title_misleading = 1.0
    elif nonmisleading_n > 0:
        title_misleading = 0.0
    else:
        title_misleading = np.nan

    first_claim = claims[0] if claims else {}

    return {
        "title_n_claims_total": summary["n_claims_total"],
        "title_n_asserted_claims": summary["n_asserted_claims"],
        "title_n_quoted_claims": summary["n_quoted_claims"],
        "title_n_asserted_classified_claims": (
            summary["n_asserted_classified_claims"]
        ),
        "title_n_asserted_unverified_claims": (
            summary["n_asserted_unverified_claims"]
        ),
        "title_n_asserted_misleading_claims": misleading_n,
        "title_n_asserted_nonmisleading_claims": nonmisleading_n,
        "title_mixed_classification": int(
            misleading_n > 0 and nonmisleading_n > 0
        ),
        "title_misleading": title_misleading,
        "title_first_verdict": normalise(first_claim.get("verdict")),
        "title_first_stance": normalise(first_claim.get("stance")),
    }


def build_stance_audit(
    m3_json: Path,
    m1_json: Path,
) -> pd.DataFrame:
    """List the exact stance values present in the Model 3 and Model 1 JSONs."""
    _, _, m3_claims_by_doc = load_factcheck(m3_json)
    _, _, m1_claims_by_doc = load_factcheck(m1_json)

    rows: list[dict[str, Any]] = []

    for source, claims_by_doc in [
        ("body_model3", m3_claims_by_doc),
        ("title_model1", m1_claims_by_doc),
    ]:
        counts: dict[str, int] = {}

        for claims in claims_by_doc.values():
            for claim in claims:
                stance = normalise(claim.get("stance"))
                display_value = stance if stance else "(blank/missing)"
                counts[display_value] = counts.get(display_value, 0) + 1

        for stance_value, count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            rows.append(
                {
                    "source": source,
                    "stance_value": stance_value,
                    "claim_count": count,
                    "kept_in_asserted_only_sample": (
                        stance_value == "asserted"
                    ),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# INPUT BUILDERS
# =============================================================================

def build_claim_table(m3_json: Path, m1_json: Path) -> pd.DataFrame:
    """Build the post-level body and title claim measures."""
    m3_posts, _, m3_claims_by_doc = load_factcheck(m3_json)
    m1_posts, _, m1_claims_by_doc = load_factcheck(m1_json)

    # Keep every post represented in Model 3 posts or claims.
    doc_ids = sorted(set(m3_posts) | set(m3_claims_by_doc))

    rows: list[dict[str, Any]] = []

    for doc_id in doc_ids:
        m3_post = m3_posts.get(doc_id, {})
        m1_post = m1_posts.get(doc_id, {})

        body_claims = m3_claims_by_doc.get(doc_id, [])
        title_claims = m1_claims_by_doc.get(doc_id, [])

        body = summarise_claims(body_claims)
        title = derive_title_measure(title_claims)

        misleading_count = body["n_asserted_misleading_claims"]

        # Claim extraction is capped at three claims per post
        # (MAX_CLAIMS_PER_POST = 3 in hive_model3_select_query.py), so more than
        # three asserted misleading body claims is structurally impossible. Stop
        # rather than silently top-code: the predictor is right-censored at three,
        # and inventing a "3 or more" category would misrepresent that ceiling.
        if misleading_count > 3:
            raise ValueError(
                f"{doc_id} has {misleading_count} asserted misleading body "
                "claims, but claim extraction is capped at three. Investigate the "
                "fact-check inputs before rebuilding."
            )

        rows.append(
            {
                "doc_id": doc_id,
                # Metadata retained for traceability and fallback.
                "author_m3": m3_post.get("author", ""),
                "title": m3_post.get("title", ""),
                "date_m3": m3_post.get("date", ""),
                "category_m3": m3_post.get("category", ""),
                "is_political_m3": m3_post.get("is_political", ""),
                "sample_stratum_m3": m3_post.get("sample_stratum", ""),
                "m3_final_label": m3_post.get("final_label", ""),
                "m3_accuracy_ordinal": m3_post.get(
                    "accuracy_ordinal", np.nan
                ),
                "m1_final_label": m1_post.get("final_label", ""),
                # Body-claim audit.
                "n_body_claims_total": body["n_claims_total"],
                "n_asserted_claims": body["n_asserted_claims"],
                "n_quoted_claims_removed": body["n_quoted_claims"],
                "n_other_nonasserted_claims_removed": (
                    body["n_other_nonasserted_claims"]
                ),
                "n_classified_claims_any_stance": (
                    body["n_classified_claims_any_stance"]
                ),
                "n_classified_explicit_nonasserted_claims": (
                    body["n_classified_explicit_nonasserted_claims"]
                ),
                "n_classified_blank_stance_claims": (
                    body["n_classified_blank_stance_claims"]
                ),
                "n_asserted_classified_claims": (
                    body["n_asserted_classified_claims"]
                ),
                "n_asserted_unverified_claims": (
                    body["n_asserted_unverified_claims"]
                ),
                "n_asserted_misleading_claims": (
                    body["n_asserted_misleading_claims"]
                ),
                "n_asserted_nonmisleading_claims": (
                    body["n_asserted_nonmisleading_claims"]
                ),
                # Main categorical body IV.
                "misleading_claim_count": misleading_count,
                "one_misleading_claim": int(misleading_count == 1),
                "two_misleading_claims": int(misleading_count == 2),
                "three_misleading_claims": int(misleading_count == 3),
                # Separate title IV and its audit fields.
                **title,
            }
        )

    claims = pd.DataFrame(rows)

    # JSON may contain a mixture of integers, numeric strings, and blanks.
    # Keeping this as object makes pyarrow infer conflicting types at export.
    claims["m3_accuracy_ordinal"] = pd.to_numeric(
        claims["m3_accuracy_ordinal"],
        errors="coerce",
    )

    assert_unique(claims, "Claim table")

    # Pre-stance eligibility: at least one claim had a classified verdict,
    # irrespective of whether the post asserted or merely quoted that claim.
    # This flag is audit-only and is never used as the final analysis sample.
    claims["analysis_sample_pre_stance"] = (
        claims["n_classified_claims_any_stance"] > 0
    )

    # Final body sample: at least one asserted claim received a classified
    # true-side or false-side verdict.
    claims["analysis_sample_body"] = (
        claims["n_asserted_classified_claims"] > 0
    )
    claims["lost_due_to_asserted_only_filter"] = (
        claims["analysis_sample_pre_stance"]
        & ~claims["analysis_sample_body"]
    )

    # Title is no longer required for the main regression. This restricted
    # sample is retained only for the later title robustness analysis.
    claims["analysis_sample_title_robustness"] = (
        claims["analysis_sample_body"]
        & claims["title_misleading"].notna()
    )

    return claims


def load_sample_controls() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the selected sample controls plus the full sample for fallback use."""
    require_file(SAMPLE_FILE, "Sample parquet")

    sample_full = ensure_doc_id(
        pd.read_parquet(SAMPLE_FILE),
        "Sample parquet",
    )
    assert_unique(sample_full, "Sample parquet")

    wanted = [
        "doc_id",
        "author",
        "category",
        "is_political",
        "sample_stratum",
        "month",
        "created",
        "body_for_analysis_len",
        "log_len",
    ]

    sample = sample_full[
        [column for column in wanted if column in sample_full.columns]
    ].copy()

    if "is_political" in sample.columns:
        sample["is_political"] = sample["is_political"].map(to01)

    if "created" in sample.columns:
        created = pd.to_datetime(
            sample["created"],
            errors="coerce",
            utc=True,
        )
        sample["created"] = created.dt.tz_convert(None)

        if "month" not in sample.columns:
            sample["month"] = created.dt.strftime("%Y-%m")

    if (
        "log_len" not in sample.columns
        and "body_for_analysis_len" in sample.columns
    ):
        body_len = pd.to_numeric(
            sample["body_for_analysis_len"],
            errors="coerce",
        ).clip(lower=0)

        sample["log_len"] = np.log1p(body_len)

    sample = sample.drop(
        columns=["body_for_analysis_len"],
        errors="ignore",
    )

    return sample, sample_full


def load_emotions() -> pd.DataFrame:
    """Load only the six discrete DistilRoBERTa emotion scores."""
    require_file(EMOTION_FILE, "Emotion parquet")

    emotions = ensure_doc_id(
        pd.read_parquet(EMOTION_FILE),
        "Emotion parquet",
    )
    assert_unique(emotions, "Emotion parquet")

    missing = [emotion for emotion in EMOTIONS if emotion not in emotions.columns]
    if missing:
        raise KeyError(
            "Emotion parquet is missing required main-regression columns: "
            f"{missing}"
        )

    emotions = emotions[["doc_id", *EMOTIONS]].copy()

    for emotion in EMOTIONS:
        emotions[emotion] = to_num(emotions[emotion])

    return emotions


def load_vad() -> pd.DataFrame:
    """Load the NRC-VAD lexicon scores (Mohammad, 2018).

    These provide an alternative operationalization of emotional language
    that uses no model at all, so the H2 robustness extension does not
    depend on the DistilRoBERTa classifier. The file is optional: if it is
    absent the builder proceeds without the columns rather than failing,
    because the main models do not require them.

    n_matched is carried through so the R analysis can drop posts whose
    scores rest on very few matched tokens.
    """
    if not VAD_FILE.exists():
        print(f"  VAD parquet not found at {VAD_FILE}; skipping VAD columns.")
        return pd.DataFrame(columns=["doc_id", *VAD_VARS, "vad_n_matched"])

    vad = ensure_doc_id(pd.read_parquet(VAD_FILE), "VAD parquet")
    assert_unique(vad, "VAD parquet")

    missing = [column for column in VAD_VARS if column not in vad.columns]
    if missing:
        raise KeyError(f"VAD parquet is missing required columns: {missing}")

    keep = ["doc_id", *VAD_VARS]
    if "n_matched" in vad.columns:
        keep.append("n_matched")

    vad = vad[keep].copy()
    if "n_matched" in vad.columns:
        vad = vad.rename(columns={"n_matched": "vad_n_matched"})

    for column in VAD_VARS:
        vad[column] = to_num(vad[column])
    if "vad_n_matched" in vad.columns:
        vad["vad_n_matched"] = to_num(vad["vad_n_matched"])

    return vad


def load_post_features() -> pd.DataFrame:
    """Load the selected post-format controls used in the main model ladder."""
    require_file(POST_FEATURES_FILE, "Post-features parquet")

    features = ensure_doc_id(
        pd.read_parquet(POST_FEATURES_FILE),
        "Post-features parquet",
    )
    assert_unique(features, "Post-features parquet")

    missing = [
        feature for feature in POST_FEATURES if feature not in features.columns
    ]
    if missing:
        raise KeyError(
            "Post-features parquet is missing required columns: "
            f"{missing}"
        )

    features = features[["doc_id", *POST_FEATURES]].copy()

    for column in ["n_images", "n_links", "n_tags"]:
        features[column] = to_num(features[column])

    features["is_weekend"] = features["is_weekend"].map(to01)

    return features



def load_engagement(sample_full: pd.DataFrame) -> pd.DataFrame:
    """
    Load finalized engagement outcomes.

    The extended HiveSQL pull is preferred. A small sample-file fallback exists
    only so the script fails gracefully if the extended file has been moved.
    """
    if EXTENDED_FILE.exists():
        engagement = ensure_doc_id(
            pd.read_parquet(EXTENDED_FILE),
            "Extended engagement parquet",
        )
        assert_unique(engagement, "Extended engagement parquet")

        if "children" in engagement.columns and "comments" not in engagement.columns:
            engagement = engagement.rename(columns={"children": "comments"})

        required = ["net_votes", "reblogs"]
        missing = [
            column for column in required if column not in engagement.columns
        ]
        if missing:
            raise KeyError(
                "Extended engagement parquet is missing required outcomes: "
                f"{missing}"
            )

        comments_col = first_present(engagement, COMMENTS_CANDS)
        upvotes_col = first_present(engagement, UPVOTES_CANDS)
        downvotes_col = first_present(engagement, DOWNVOTES_CANDS)
        distinct_voters_col = first_present(engagement, DISTINCT_VOTERS_CANDS)

        keep = [
            "doc_id",
            "net_votes",
            "reblogs",
            "total_payout_value",
            "pending_payout_value",
            comments_col,
            upvotes_col,
            downvotes_col,
            distinct_voters_col,
        ]
        keep = [column for column in keep if column is not None and column in engagement.columns]
        engagement = engagement[keep].copy()

        rename_map = {}
        if comments_col is not None:
            rename_map[comments_col] = "comments"
        if upvotes_col is not None:
            rename_map[upvotes_col] = "upvotes"
        if downvotes_col is not None:
            rename_map[downvotes_col] = "downvotes"
        if distinct_voters_col is not None:
            rename_map[distinct_voters_col] = "distinct_voters"
        engagement = engagement.rename(columns=rename_map)

        for column in engagement.columns:
            if column != "doc_id":
                engagement[column] = to_num(engagement[column])

        # Missing reblog counts inside the authoritative panel represent zero.
        engagement["reblogs"] = engagement["reblogs"].fillna(0)

        total = (
            engagement["total_payout_value"]
            if "total_payout_value" in engagement.columns
            else pd.Series(0.0, index=engagement.index)
        )
        pending = (
            engagement["pending_payout_value"]
            if "pending_payout_value" in engagement.columns
            else pd.Series(0.0, index=engagement.index)
        )

        engagement["payout"] = total.fillna(0) + pending.fillna(0)

        print("Engagement source: extended finalized HiveSQL pull")

    else:
        print(
            "WARNING: extended engagement file not found. "
            "Using the sample-file fallback."
        )

        votes_col = first_present(sample_full, VOTES_CANDS)
        reblogs_col = first_present(sample_full, REBLOG_CANDS)
        payout_col = first_present(sample_full, PAYOUT_CANDS)
        comments_col = first_present(sample_full, COMMENTS_CANDS)
        upvotes_col = first_present(sample_full, UPVOTES_CANDS)
        downvotes_col = first_present(sample_full, DOWNVOTES_CANDS)
        distinct_voters_col = first_present(sample_full, DISTINCT_VOTERS_CANDS)

        if votes_col is None or reblogs_col is None:
            raise KeyError(
                "Fallback sample does not contain both a votes and a reblogs "
                "column."
            )

        engagement = sample_full[["doc_id"]].copy()
        engagement["net_votes"] = to_num(sample_full[votes_col])
        engagement["reblogs"] = to_num(sample_full[reblogs_col]).fillna(0)
        engagement["comments"] = to_num(sample_full[comments_col]) if comments_col is not None else np.nan
        engagement["upvotes"] = to_num(sample_full[upvotes_col]) if upvotes_col is not None else np.nan
        engagement["downvotes"] = to_num(sample_full[downvotes_col]) if downvotes_col is not None else np.nan
        engagement["distinct_voters"] = to_num(sample_full[distinct_voters_col]) if distinct_voters_col is not None else np.nan

        if payout_col is not None:
            engagement["payout"] = to_num(sample_full[payout_col])
        else:
            engagement["payout"] = np.nan

    for column in ["comments", "upvotes", "downvotes", "distinct_voters"]:
        if column not in engagement.columns:
            engagement[column] = np.nan

    engagement["any_reblog"] = np.where(
        engagement["reblogs"].notna(),
        (engagement["reblogs"] > 0).astype(int),
        np.nan,
    )

    return engagement[
        [
            "doc_id",
            "net_votes",
            "reblogs",
            "any_reblog",
            "payout",
            "comments",
            "upvotes",
            "downvotes",
            "distinct_voters",
        ]
    ]


def load_prewindow_author_standing() -> pd.DataFrame:
    """
    Load author standing measured strictly before 2024-09-01.

    Source:
      12. Post Features/hive_author_standing.parquet

    The source stores the raw pre-window measures. They are converted here to
    the four logged variables used in the R-squared decomposition:
      - log_pre_rewards  = log1p(pre_author_rew_vests)
      - log_pre_curation = log1p(pre_curation_vests)
      - log_pre_posts    = log1p(pre_root_posts)
      - log_acct_age     = log1p(acct_age_days)
    """
    require_file(
        PREWINDOW_STANDING_FILE,
        "Pre-window author standing parquet",
    )

    print(f"Loading pre-window author standing: {PREWINDOW_STANDING_FILE}")
    standing = pd.read_parquet(PREWINDOW_STANDING_FILE)

    if "author" not in standing.columns:
        raise KeyError(
            "Pre-window author standing file must contain an 'author' column. "
            f"Available columns: {list(standing.columns)}"
        )

    raw_columns = {
        "log_pre_rewards": "pre_author_rew_vests",
        "log_pre_curation": "pre_curation_vests",
        "log_pre_posts": "pre_root_posts",
        "log_acct_age": "acct_age_days",
    }

    # Accept either the already-transformed columns or the original raw columns.
    missing_both = [
        target
        for target, raw in raw_columns.items()
        if target not in standing.columns and raw not in standing.columns
    ]
    if missing_both:
        raise KeyError(
            "Pre-window author standing file is missing both transformed and "
            f"raw source columns for: {missing_both}. Available columns: "
            f"{list(standing.columns)}"
        )

    out = pd.DataFrame({
        "author": standing["author"].astype(str).str.strip()
    })

    for target, raw in raw_columns.items():
        source = target if target in standing.columns else raw
        values = to_num(standing[source]).clip(lower=0)
        out[target] = values if source == target else np.log1p(values)

    # Author popularity (Eckert): average pre-window author reward per
    # pre-window root post, from the RAW columns. Authors with zero
    # pre-window root posts receive zero; pre_root_posts is carried through
    # so the R analysis can flag them.
    popularity_sources = {"pre_author_rew_vests", "pre_root_posts"}
    if popularity_sources <= set(standing.columns):
        rewards = to_num(standing["pre_author_rew_vests"]).clip(lower=0).fillna(0)
        n_posts = to_num(standing["pre_root_posts"]).clip(lower=0).fillna(0)
        out["pre_root_posts"] = n_posts
        out["pre_avg_reward"] = np.where(
            n_posts > 0,
            rewards / np.where(n_posts > 0, n_posts, 1),
            0.0,
        )
        out["log_pre_avg_reward"] = np.log1p(out["pre_avg_reward"])
    else:
        missing_pop = sorted(popularity_sources - set(standing.columns))
        print(
            "  WARNING: raw columns missing for the popularity measure "
            f"({missing_pop}); pre_avg_reward will be absent."
        )
        for column in POPULARITY_VARS:
            out[column] = np.nan

    ensure_author_unique(out, "Pre-window author standing")

    print("Pre-window standing source audit:")
    print(f"  Authors in standing file: {len(out):,}")
    for column in PREWINDOW_STANDING_VARS + POPULARITY_VARS:
        print(
            f"  {column}: {int(out[column].notna().sum()):,} non-missing, "
            f"{int((out[column] == 0).sum()):,} zeros"
        )

    if not any(out[column].notna().any() for column in PREWINDOW_STANDING_VARS):
        raise ValueError(
            "The pre-window standing file was found, but all four derived "
            "standing variables are missing. Inspect the raw source columns."
        )

    return out


def normalise_output_dtypes(data: pd.DataFrame) -> pd.DataFrame:
    """
    Give parquet a stable schema.

    JSON fields can mix numbers, numeric strings, blanks, and missing values.
    Pandas then stores them as object columns, which pyarrow may be unable to
    convert. Explicit typing here prevents that class of export error.
    """
    out = data.copy()

    integer_columns = [
        "n_body_claims_total",
        "n_asserted_claims",
        "n_quoted_claims_removed",
        "n_other_nonasserted_claims_removed",
        "n_classified_claims_any_stance",
        "n_classified_explicit_nonasserted_claims",
        "n_classified_blank_stance_claims",
        "n_asserted_classified_claims",
        "n_asserted_unverified_claims",
        "n_asserted_misleading_claims",
        "n_asserted_nonmisleading_claims",
        "misleading_claim_count",
        "one_misleading_claim",
        "two_misleading_claims",
        "three_misleading_claims",
        "title_n_claims_total",
        "title_n_asserted_claims",
        "title_n_quoted_claims",
        "title_n_asserted_classified_claims",
        "title_n_asserted_unverified_claims",
        "title_n_asserted_misleading_claims",
        "title_n_asserted_nonmisleading_claims",
        "title_mixed_classification",
        "any_reblog",
        "n_images",
        "n_links",
        "n_tags",
    ]

    numeric_columns = [
        "m3_accuracy_ordinal",
        "title_misleading",
        *EMOTIONS,
        *VAD_VARS,
        "vad_n_matched",
        "net_votes",
        "reblogs",
        "payout",
        "comments",
        "upvotes",
        "downvotes",
        "distinct_voters",
        "log_len",
        "is_political",
        "is_weekend",
        *PREWINDOW_STANDING_VARS,
        *POPULARITY_VARS,
    ]

    boolean_columns = [
        "analysis_sample_pre_stance",
        "analysis_sample_body",
        "lost_due_to_asserted_only_filter",
        "analysis_sample_title_robustness",
    ]

    datetime_columns = ["created"]

    string_columns = [
        "doc_id",
        "author",
        "title",
        "category",
        "month",
        "sample_stratum",
        "title_first_verdict",
        "title_first_stance",
        "m3_final_label",
        "m1_final_label",
    ]

    for column in integer_columns:
        if column in out.columns:
            out[column] = pd.to_numeric(
                out[column],
                errors="coerce",
            ).astype("Int64")

    for column in numeric_columns:
        if column in out.columns:
            out[column] = pd.to_numeric(
                out[column],
                errors="coerce",
            ).astype("Float64")

    for column in boolean_columns:
        if column in out.columns:
            out[column] = out[column].astype("boolean")

    for column in datetime_columns:
        if column in out.columns:
            out[column] = pd.to_datetime(
                out[column],
                errors="coerce",
            )

    for column in string_columns:
        if column in out.columns:
            out[column] = out[column].astype("string")

    return out


# =============================================================================
# AUDIT
# =============================================================================

def build_audit(data: pd.DataFrame) -> pd.DataFrame:
    pre_stance_sample = data["analysis_sample_pre_stance"].fillna(False)
    body_sample = data["analysis_sample_body"].fillna(False)
    stance_loss = data["lost_due_to_asserted_only_filter"].fillna(False)
    title_sample = data["analysis_sample_title_robustness"].fillna(False)

    any_asserted_body = data["n_asserted_claims"] > 0
    valid_title = data["title_misleading"].notna()
    any_asserted_title = data["title_n_asserted_claims"] > 0

    metrics: list[tuple[str, Any]] = [
        ("posts_in_model3", len(data)),
        ("body_claims_before_filter", int(data["n_body_claims_total"].sum())),
        ("asserted_body_claims_kept", int(data["n_asserted_claims"].sum())),
        (
            "non_asserted_body_claims_removed_quoted_questioned_unclear",
            int(data["n_quoted_claims_removed"].sum()),
        ),
        (
            "blank_or_missing_stance_body_claims_removed",
            int(data["n_other_nonasserted_claims_removed"].sum()),
        ),
        (
            "classified_body_claims_before_stance_filter",
            int(data["n_classified_claims_any_stance"].sum()),
        ),
        (
            "classified_explicit_nonasserted_body_claims_removed",
            int(data["n_classified_explicit_nonasserted_claims"].sum()),
        ),
        (
            "classified_blank_stance_body_claims_removed",
            int(data["n_classified_blank_stance_claims"].sum()),
        ),
        (
            "asserted_classified_body_claims_after_stance_filter",
            int(data["n_asserted_classified_claims"].sum()),
        ),
        (
            "asserted_unverified_body_claims",
            int(data["n_asserted_unverified_claims"].sum()),
        ),
        ("posts_with_any_asserted_body_claim", int(any_asserted_body.sum())),
        (
            "posts_with_classified_body_claim_before_stance_filter",
            int(pre_stance_sample.sum()),
        ),
        (
            "posts_lost_specifically_due_to_asserted_only_filter",
            int(stance_loss.sum()),
        ),
        (
            "posts_with_asserted_body_but_no_classified_verdict",
            int((any_asserted_body & ~body_sample).sum()),
        ),
        (
            "body_only_main_sample",
            int(body_sample.sum()),
        ),
        (
            "posts_dropped_from_body_main_sample",
            int((~body_sample).sum()),
        ),
        (
            "posts_with_any_asserted_title_claim",
            int(any_asserted_title.sum()),
        ),
        (
            "posts_with_valid_title_classification",
            int(valid_title.sum()),
        ),
        (
            "posts_with_asserted_title_but_no_classified_verdict",
            int((any_asserted_title & ~valid_title).sum()),
        ),
        (
            "title_robustness_sample",
            int(title_sample.sum()),
        ),
        (
            "body_sample_posts_lost_when_title_is_required",
            int(body_sample.sum() - title_sample.sum()),
        ),
        (
            "valid_title_misleading_0",
            int((data["title_misleading"] == 0).sum()),
        ),
        (
            "valid_title_misleading_1",
            int((data["title_misleading"] == 1).sum()),
        ),
        (
            "title_measure_missing",
            int(data["title_misleading"].isna().sum()),
        ),
    ]

    for count in range(4):
        metrics.append(
            (
                f"body_main_sample_posts_with_{count}_misleading_body_claims",
                int(
                    (
                        body_sample
                        & (data["misleading_claim_count"] == count)
                    ).sum()
                ),
            )
        )

    for column in [
        *EMOTIONS,
        *VAD_VARS,
        "net_votes",
        "reblogs",
        "comments",
        "upvotes",
        "downvotes",
        "distinct_voters",
        "log_len",
        "is_political",
        *POST_FEATURES,
        *PREWINDOW_STANDING_VARS,
    ]:
        if column in data.columns:
            metrics.append(
                (
                    f"nonmissing_{column}_within_body_main_sample",
                    int(data.loc[body_sample, column].notna().sum()),
                )
            )

    return pd.DataFrame(metrics, columns=["metric", "value"])


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parquet_path = OUTPUT_DIR / f"{OUTPUT_STEM}.parquet"
    xlsx_path = OUTPUT_DIR / f"{OUTPUT_STEM}.xlsx"
    audit_csv_path = OUTPUT_DIR / f"{OUTPUT_STEM}_audit.csv"

    print("=" * 72)
    print("OUTPUT LOCATION")
    print("=" * 72)
    print(f"Output folder : {OUTPUT_DIR.resolve()}")
    print(f"Parquet file  : {parquet_path.resolve()}")
    print(f"Excel file    : {xlsx_path.resolve()}")
    print(f"Audit CSV     : {audit_csv_path.resolve()}")
    print("=" * 72)
    print()

    m3_json = resolve_factcheck_path("model3", M3_JSON_CANDIDATES)
    m1_json = resolve_factcheck_path("model1", M1_JSON_CANDIDATES)

    print("Resolved fact-check files:")
    print(f"  Model 3: {m3_json}")
    print(f"  Model 1: {m1_json}")

    print("Auditing exact stance values...")
    stance_audit = build_stance_audit(m3_json, m1_json)
    print(stance_audit.to_string(index=False))

    print("\nBuilding claim measures...")
    claims = build_claim_table(m3_json, m1_json)

    print("Loading sample controls...")
    sample, sample_full = load_sample_controls()

    print("Loading six discrete emotion scores...")
    emotions = load_emotions()

    print("Loading NRC-VAD lexicon scores...")
    vad = load_vad()

    print("Loading post-format controls...")
    features = load_post_features()

    print("Loading engagement outcomes...")
    engagement = load_engagement(sample_full)

    standing = load_prewindow_author_standing()

    # All joins are explicitly one-to-one.
    data = claims.merge(
        sample,
        on="doc_id",
        how="left",
        validate="one_to_one",
    )

    data = data.merge(
        emotions,
        on="doc_id",
        how="left",
        validate="one_to_one",
    )

    if not vad.empty:
        data = data.merge(
            vad,
            on="doc_id",
            how="left",
            validate="one_to_one",
        )
    else:
        for column in [*VAD_VARS, "vad_n_matched"]:
            data[column] = np.nan

    data = data.merge(
        features,
        on="doc_id",
        how="left",
        validate="one_to_one",
    )

    data = data.merge(
        engagement,
        on="doc_id",
        how="left",
        validate="one_to_one",
    )

    # Canonical metadata: prefer the sample file, fall back to Model 3.
    if "author" in data.columns:
        data["author"] = coalesce_text(data["author"], data["author_m3"])
    else:
        data["author"] = data["author_m3"].astype("string")

    if not standing.empty:
        # Hive usernames are case-insensitive. Join on a normalized key so
        # whitespace or capitalization differences cannot silently erase the
        # pre-window standing merge.
        data["_author_key"] = (
            data["author"].astype("string").str.strip().str.lower()
        )
        standing["_author_key"] = (
            standing["author"].astype("string").str.strip().str.lower()
        )

        duplicated_keys = standing["_author_key"].duplicated(keep=False)
        if duplicated_keys.any():
            examples = standing.loc[duplicated_keys, "author"].head(10).tolist()
            raise ValueError(
                "Pre-window standing has duplicate normalized author keys. "
                f"Examples: {examples}"
            )

        popularity_present = [
            column for column in POPULARITY_VARS if column in standing.columns
        ]
        standing_for_merge = standing[
            ["_author_key", *PREWINDOW_STANDING_VARS, *popularity_present]
        ]
        data = data.merge(
            standing_for_merge,
            on="_author_key",
            how="left",
            validate="many_to_one",
        )

        matched_any = data[PREWINDOW_STANDING_VARS].notna().any(axis=1)
        matched_all = data[PREWINDOW_STANDING_VARS].notna().all(axis=1)
        body_mask = data["analysis_sample_body"].fillna(False).astype(bool)

        total_posts = len(data)
        total_authors = int(data["_author_key"].nunique())
        matched_posts = int(matched_any.sum())
        matched_authors = int(
            data.loc[matched_any, "_author_key"].nunique()
        )

        body_posts = int(body_mask.sum())
        body_authors = int(data.loc[body_mask, "_author_key"].nunique())
        body_posts_complete = int((body_mask & matched_all).sum())
        body_authors_complete = int(
            data.loc[body_mask & matched_all, "_author_key"].nunique()
        )

        print("Pre-window standing merge audit:")
        print(f"  Entire parquet matched posts: {matched_posts:,} of {total_posts:,}")
        print(f"  Entire parquet matched authors: {matched_authors:,} of {total_authors:,}")
        print(f"  Body-only sample posts with all standing measures: {body_posts_complete:,} of {body_posts:,}")
        print(f"  Body-only sample authors with all standing measures: {body_authors_complete:,} of {body_authors:,}")

        # Popularity preview at the author level (body-only sample). The R
        # analysis performs the actual median split and top-20% cut; these
        # numbers exist to sanity-check the distribution beforehand.
        if "pre_avg_reward" in data.columns:
            author_pop = (
                data.loc[body_mask, ["_author_key", "pre_avg_reward",
                                     "pre_root_posts"]]
                .dropna(subset=["pre_avg_reward"])
                .drop_duplicates("_author_key")
            )
            if len(author_pop):
                pop = author_pop["pre_avg_reward"]
                quantiles = pop.quantile([0.5, 0.8, 0.9]).round(1)
                print("  Author popularity (pre-window avg reward, "
                      "body-only sample authors):")
                print(f"    authors: {len(author_pop):,}  "
                      f"zero-popularity share: {(pop == 0).mean() * 100:.1f}%")
                print(f"    p50 {quantiles.iloc[0]:,.1f}  "
                      f"p80 {quantiles.iloc[1]:,.1f}  "
                      f"p90 {quantiles.iloc[2]:,.1f}")
                print(f"    authors with zero pre-window root posts: "
                      f"{int((author_pop['pre_root_posts'] == 0).sum()):,}")

        missing_body = body_mask & ~matched_all
        if missing_body.any():
            missing_authors = (
                data.loc[missing_body, "_author_key"]
                .dropna()
                .drop_duplicates()
                .head(20)
                .tolist()
            )
            raise RuntimeError(
                "Pre-window standing coverage is incomplete in the body-only "
                "main sample. Missing authors include: "
                f"{missing_authors}"
            )

    if "category" in data.columns:
        data["category"] = coalesce_text(
            data["category"],
            data["category_m3"],
        )
    else:
        data["category"] = data["category_m3"].astype("string")

    if "sample_stratum" in data.columns:
        data["sample_stratum"] = coalesce_text(
            data["sample_stratum"],
            data["sample_stratum_m3"],
        )
    else:
        data["sample_stratum"] = data["sample_stratum_m3"].astype("string")

    if "is_political" in data.columns:
        fallback_political = data["is_political_m3"].map(to01)
        data["is_political"] = data["is_political"].fillna(
            fallback_political
        )
    else:
        data["is_political"] = data["is_political_m3"].map(to01)

    if "created" not in data.columns:
        created = pd.to_datetime(
            data["date_m3"],
            errors="coerce",
            utc=True,
        )
        data["created"] = created.dt.tz_convert(None)

    if "month" not in data.columns:
        data["month"] = pd.to_datetime(
            data["created"],
            errors="coerce",
        ).dt.strftime("%Y-%m")

    # Keep the output compact and ordered for the main R analysis.
    final_columns = [
        # Keys and metadata
        "doc_id",
        "author",
        "title",
        "created",
        "category",
        "month",
        "sample_stratum",
        # Analysis flags
        "analysis_sample_pre_stance",
        "analysis_sample_body",
        "lost_due_to_asserted_only_filter",
        "analysis_sample_title_robustness",
        # Claim audit
        "n_body_claims_total",
        "n_asserted_claims",
        "n_quoted_claims_removed",
        "n_other_nonasserted_claims_removed",
        "n_classified_claims_any_stance",
        "n_classified_explicit_nonasserted_claims",
        "n_classified_blank_stance_claims",
        "n_asserted_classified_claims",
        "n_asserted_unverified_claims",
        "n_asserted_misleading_claims",
        "n_asserted_nonmisleading_claims",
        # Main categorical body IV
        "misleading_claim_count",
        "one_misleading_claim",
        "two_misleading_claims",
        "three_misleading_claims",
        # Separate title IV and audit
        "title_misleading",
        "title_n_claims_total",
        "title_n_asserted_claims",
        "title_n_quoted_claims",
        "title_n_asserted_classified_claims",
        "title_n_asserted_unverified_claims",
        "title_n_asserted_misleading_claims",
        "title_n_asserted_nonmisleading_claims",
        "title_mixed_classification",
        "title_first_verdict",
        "title_first_stance",
        # Six separate DistilRoBERTa emotions
        *EMOTIONS,
        # NRC-VAD lexicon dimensions (H2 robustness)
        *VAD_VARS,
        "vad_n_matched",
        # Main outcomes
        "net_votes",
        "reblogs",
        "any_reblog",
        "comments",
        "upvotes",
        "downvotes",
        "distinct_voters",
        # Stored for later robustness; not required in the main QMD
        "payout",
        # Main controls and fixed-effect keys
        "log_len",
        "is_political",
        "n_images",
        "n_links",
        "n_tags",
        "is_weekend",
        # Optional pre-window author standing
        *PREWINDOW_STANDING_VARS,
        # Author popularity for the Eckert specifications
        *POPULARITY_VARS,
        # Traceability only
        "m3_final_label",
        "m3_accuracy_ordinal",
        "m1_final_label",
    ]

    present = [column for column in final_columns if column in data.columns]
    missing = [column for column in final_columns if column not in data.columns]

    data = data[present].sort_values("doc_id").reset_index(drop=True)
    data = normalise_output_dtypes(data)
    audit = build_audit(data)

    data.to_parquet(parquet_path, index=False)
    audit.to_csv(audit_csv_path, index=False)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        sanitize_for_excel(data).to_excel(
            writer,
            sheet_name="data",
            index=False,
        )
        sanitize_for_excel(audit).to_excel(
            writer,
            sheet_name="audit",
            index=False,
        )
        sanitize_for_excel(stance_audit).to_excel(
            writer,
            sheet_name="stance_audit",
            index=False,
        )

        data_ws = writer.book["data"]
        data_ws.freeze_panes = "A2"
        data_ws.auto_filter.ref = data_ws.dimensions

        audit_ws = writer.book["audit"]
        audit_ws.freeze_panes = "A2"
        audit_ws.auto_filter.ref = audit_ws.dimensions

        stance_ws = writer.book["stance_audit"]
        stance_ws.freeze_panes = "A2"
        stance_ws.auto_filter.ref = stance_ws.dimensions

    pre_stance_sample = data["analysis_sample_pre_stance"].fillna(False)
    body_sample = data["analysis_sample_body"].fillna(False)
    stance_loss = data["lost_due_to_asserted_only_filter"].fillna(False)
    title_sample = data["analysis_sample_title_robustness"].fillna(False)

    posts_any_asserted_body = int((data["n_asserted_claims"] > 0).sum())
    posts_body_classified = int(body_sample.sum())
    posts_any_asserted_title = int((data["title_n_asserted_claims"] > 0).sum())
    posts_title_classified = int(data["title_misleading"].notna().sum())
    posts_body_and_title = int(title_sample.sum())

    print("\n" + "=" * 72)
    print("SAMPLE FUNNEL")
    print("=" * 72)
    print(f"Posts written                                      : {len(data):,}")
    print(f"Posts with >=1 asserted body claim                 : {posts_any_asserted_body:,}")
    print(f"Posts with >=1 classified claim before stance rule : {int(pre_stance_sample.sum()):,}")
    print(f"Posts lost specifically to asserted-only rule      : {int(stance_loss.sum()):,}")
    print(f"Posts with >=1 classified asserted body claim      : {posts_body_classified:,}")
    print(f"Posts with >=1 asserted title claim                : {posts_any_asserted_title:,}")
    print(f"Posts with a classified title measure              : {posts_title_classified:,}")
    print(f"Body-only MAIN sample                              : {posts_body_classified:,}")
    print(f"Body + title ROBUSTNESS sample                     : {posts_body_and_title:,}")
    print(
        "Body-sample posts lost when title is required     : "
        f"{posts_body_classified - posts_body_and_title:,}"
    )

    print("\nMisleading body-claim count in BODY-ONLY main sample:")
    print(
        data.loc[
            body_sample,
            "misleading_claim_count",
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nTitle misinformation in TITLE robustness sample:")
    print(
        data.loc[
            title_sample,
            "title_misleading",
        ]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )

    if missing:
        print(f"\nNOTE: expected output columns not found: {missing}")

    print("\n" + "=" * 72)
    print("FILES CREATED SUCCESSFULLY")
    print("=" * 72)
    print(f"Output folder : {OUTPUT_DIR.resolve()}")
    print(f"Parquet file  : {parquet_path.resolve()}")
    print(f"Excel file    : {xlsx_path.resolve()}")
    print(f"Audit CSV     : {audit_csv_path.resolve()}")
    print("=" * 72)


if __name__ == "__main__":
    main()
