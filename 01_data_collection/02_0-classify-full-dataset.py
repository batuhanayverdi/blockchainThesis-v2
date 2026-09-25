"""
Classifier v2: Returns classification (YES/NO) AND category label.

Output columns in classification_progress.csv:
    idx, classification, category

Categories for political posts (when classification=YES):
    election_campaign, election_fraud, trump_administration,
    biden_administration, foreign_policy, domestic_policy,
    political_violence, political_commentary, other_political

Categories for non-political posts (when classification=NO):
    crypto_finance, gaming, travel_lifestyle, entertainment,
    technology, personal, non_us_politics, community_platform,
    other_noise

Cost impact: ~$2-3 extra for the full dataset (more output tokens).
Same resume logic as before.
"""

import os
import sys
import time
import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

API_KEY = "API key"  # PASTE YOUR KEY HERE

INPUT_PARQUET = "hive_full_posts_full.parquet"
PROGRESS_CSV = "classification_progress.csv"
FINAL_OUTPUT = "hive_classified.parquet"

MODEL = "gpt-4o-mini"
MAX_BODY_CHARS = 1500
N_WORKERS = 20
SAVE_EVERY = 500
MAX_RETRIES = 3

# =============================================================================
# VALID CATEGORIES
# =============================================================================

POLITICAL_CATEGORIES = {
    "election_campaign", "election_fraud", "trump_administration",
    "biden_administration", "foreign_policy", "domestic_policy",
    "political_violence", "political_commentary", "other_political",
}

NONPOLITICAL_CATEGORIES = {
    "crypto_finance", "gaming", "travel_lifestyle", "entertainment",
    "technology", "personal", "non_us_politics", "community_platform",
    "other_noise",
}

ALL_CATEGORIES = POLITICAL_CATEGORIES | NONPOLITICAL_CATEGORIES


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are a content classifier for an academic study on the 2024 US presidential election.

You will receive a social media post. Output two things separated by a pipe character:
1. classification: YES or NO - whether the post is PRIMARILY about US politics during the 2024 election cycle (Sep 2024 - Jan 2025)
2. category: a single snake_case label from the lists below

Output format (exact): CLASSIFICATION|CATEGORY

CLASSIFICATION RULES:

YES if the post:
- Discusses the 2024 US presidential election, primaries, or campaigns
- Discusses US political figures (Trump, Harris, Biden, Musk as political actor, Walz, Vance) in a political, policy, or electoral context
- Discusses US government actions, executive orders, Congress, Supreme Court, or federal policy
- Discusses political events of the period (Jan 6 aftermath, Trump assassination attempts, Hurricane Helene/Milton government response, Luigi Mangione / UnitedHealthcare CEO shooting as political story)
- Takes a political stance on US domestic issues (immigration, abortion, economy as policy, foreign policy)
- Discusses election fraud claims, voting, ballots, or electoral processes in the US
- Discusses US foreign, trade, or tariff policy TOWARD other countries (Trump's tariffs on China, US sanctions on Iran, US aid to Ukraine)

NO if the post:
- Is about cryptocurrency, Bitcoin, Hive token economy, NFTs, trading, or blockchain (even if Trump or Musk is mentioned)
- Is about technology, gaming, Splinterlands, or Hive platform activities
- Is about travel, food, lifestyle, personal diary entries, pet stories
- Is about sports, entertainment, film reviews, music, art
- Mentions political figures only incidentally
- Is PRIMARILY about non-US politics (Canadian, European, etc.). Mere mention of US figures does not make it US political.
- Is about science, space weather, climate (as natural phenomena)
- Is commercial, promotional, or a daily activity report

PRIMARY SUBJECT test: if US political references were removed, would the post still have substantial content? If yes, classify as NO.

CATEGORY RULES:

If YES, pick ONE of these:
- election_campaign: campaigns, rallies, debates, voting processes, candidate endorsements
- election_fraud: fraud claims, ballot disputes, stolen election narratives, election integrity
- trump_administration: Trump's policies, cabinet picks, inauguration, post-election decisions
- biden_administration: Biden's final months, executive orders, transition period actions
- foreign_policy: US foreign relations, sanctions, aid, tariffs toward other countries
- domestic_policy: immigration, abortion, economy, healthcare as policy debates
- political_violence: assassination attempts, Jan 6, political attacks, threats
- political_commentary: partisan opinion, memes, reactions, commentary on politicians
- other_political: political content not fitting above

If NO, pick ONE of these:
- crypto_finance: cryptocurrency, Bitcoin, trading, blockchain, Hive tokens, NFTs
- gaming: Splinterlands, video games, game reviews
- travel_lifestyle: travel, food, cooking, lifestyle, home
- entertainment: film, TV, music, sports, art, books
- technology: tech news, gadgets, software, science, space
- personal: personal diary, family, pets, emotions, experiences
- non_us_politics: other countries' politics, leaders, elections
- community_platform: Hive community posts, giveaways, curation reports, platform updates
- other_noise: anything not fitting above

Respond with exactly: YES|category_name or NO|category_name
No other text. No explanations."""


# =============================================================================
# PARSE RESPONSE
# =============================================================================

def parse_response(raw: str):
    """Parse 'YES|category' or 'NO|category' into (classification, category)."""
    raw = raw.strip()
    if "|" not in raw:
        return ("PARSE_ERROR", raw[:50])

    parts = raw.split("|", 1)
    cls = parts[0].strip().upper()
    cat = parts[1].strip().lower().replace(" ", "_")

    # Normalize classification
    if cls.startswith("YES"):
        cls = "YES"
    elif cls.startswith("NO"):
        cls = "NO"
    else:
        return ("PARSE_ERROR", raw[:50])

    # Validate category
    if cat not in ALL_CATEGORIES:
        # Try to map common variants
        for valid_cat in ALL_CATEGORIES:
            if valid_cat in cat or cat in valid_cat:
                cat = valid_cat
                break
        else:
            cat = "other_political" if cls == "YES" else "other_noise"

    return (cls, cat)


# =============================================================================
# CLASSIFY ONE POST
# =============================================================================

def classify_one(client, idx, title, body):
    """Returns (idx, classification, category)."""
    user_msg = f"Title: {str(title)[:300]}\n\nBody:\n{str(body)[:MAX_BODY_CHARS]}"
    last_err = None

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0,
                max_tokens=20,  # a bit more to fit "YES|election_campaign"
            )
            raw = resp.choices[0].message.content.strip()
            cls, cat = parse_response(raw)
            return (idx, cls, cat)
        except Exception as e:
            last_err = e
            time.sleep(2 ** (attempt + 1))

    return (idx, "ERROR", str(last_err)[:80])


# =============================================================================
# PROGRESS HANDLING
# =============================================================================

def load_progress():
    """Returns dict of {idx: (classification, category)}."""
    if not os.path.exists(PROGRESS_CSV):
        return {}
    df = pd.read_csv(PROGRESS_CSV)
    # Only keep rows with valid classification
    valid = df[df["classification"].isin(["YES", "NO"])]
    return {
        row["idx"]: (row["classification"], row.get("category", ""))
        for _, row in valid.iterrows()
    }


def save_progress(results_dict):
    """Write current results to disk atomically."""
    rows = [
        {"idx": idx, "classification": cls, "category": cat}
        for idx, (cls, cat) in results_dict.items()
    ]
    df = pd.DataFrame(rows)
    tmp = PROGRESS_CSV + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, PROGRESS_CSV)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"Loading {INPUT_PARQUET}...")
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"  {len(df):,} total posts")

    progress = load_progress()
    print(f"  {len(progress):,} already classified (resuming)")

    pending_indices = [i for i in df.index if i not in progress]
    print(f"  {len(pending_indices):,} remaining to process")

    if not pending_indices:
        print("\nAll posts already classified. Writing final output.")
    else:
        avg_tokens_in = 550
        avg_tokens_out = 6  # ~6 tokens for "YES|category_name"
        est_cost = len(pending_indices) * avg_tokens_in * 0.15 / 1_000_000
        est_cost += len(pending_indices) * avg_tokens_out * 0.60 / 1_000_000
        print(f"  Estimated remaining cost: ~${est_cost:.2f}")
        print(f"  Checkpoint saves every {SAVE_EVERY} posts")
        print(f"  Using {N_WORKERS} parallel workers")
        print()

        client = OpenAI(api_key=API_KEY)

        completed_since_save = 0
        progress_bar = tqdm(total=len(pending_indices), desc="Classifying", unit="post")

        with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
            future_to_idx = {}
            for idx in pending_indices:
                row = df.loc[idx]
                future = executor.submit(
                    classify_one,
                    client,
                    idx,
                    row.get("title", ""),
                    row.get("body_clean", row.get("body", "")),
                )
                future_to_idx[future] = idx

            try:
                for future in as_completed(future_to_idx):
                    idx, cls, cat = future.result()
                    progress[idx] = (cls, cat)
                    completed_since_save += 1
                    progress_bar.update(1)

                    if completed_since_save >= SAVE_EVERY:
                        save_progress(progress)
                        completed_since_save = 0
                        progress_bar.set_postfix_str(f"last save: {len(progress):,}")
            except KeyboardInterrupt:
                print(f"\n\nInterrupted. Saving current progress...")
                save_progress(progress)
                print(f"Saved {len(progress):,} results to {PROGRESS_CSV}")
                print(f"Re-run this script to resume.")
                progress_bar.close()
                return
            except Exception as e:
                print(f"\n\nUnexpected error: {e}")
                save_progress(progress)
                print(f"Saved {len(progress):,} results. Re-run to resume.")
                progress_bar.close()
                return

        progress_bar.close()
        save_progress(progress)
        print(f"\nAll processing complete. Saved {len(progress):,} results.")

    # =========================================================================
    # FINAL OUTPUT
    # =========================================================================

    print(f"\nMerging with original data...")
    df["classification"] = df.index.map({i: p[0] for i, p in progress.items()}).fillna("MISSING")
    df["category"] = df.index.map({i: p[1] for i, p in progress.items()}).fillna("")
    df.to_parquet(FINAL_OUTPUT, index=False)
    print(f"Saved: {FINAL_OUTPUT}")

    print(f"\n=== Classification stats ===")
    print(df["classification"].value_counts())

    print(f"\n=== Category distribution (YES) ===")
    print(df[df["classification"] == "YES"]["category"].value_counts())

    print(f"\n=== Category distribution (NO) ===")
    print(df[df["classification"] == "NO"]["category"].value_counts())

    df_yes = df[df["classification"] == "YES"].copy()
    print(f"\nYES (political) posts: {len(df_yes):,}")

    body_col = "body_clean" if "body_clean" in df_yes.columns else "body"
    df_yes[body_col] = df_yes[body_col].astype(str).str.slice(0, 32000)

    export_cols = [
        "author", "permlink", "title", body_col, "created",
        "net_votes", "children", "payout_combined", "link",
        "classification", "category",
    ]
    available = [c for c in export_cols if c in df_yes.columns]

    if len(df_yes) <= 1_000_000:
        out = "hive_classified_political.xlsx"
        df_yes[available].to_excel(out, index=False)
        print(f"Excel saved: {out}")
    else:
        for i in range(0, len(df_yes), 1_000_000):
            part = df_yes.iloc[i:i+1_000_000]
            out = f"hive_classified_political_part{i//1_000_000 + 1:02d}.xlsx"
            part[available].to_excel(out, index=False)
            print(f"  {out}: {len(part):,} rows")


if __name__ == "__main__":
    main()