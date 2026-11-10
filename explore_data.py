
import pandas as pd
from datasets import load_dataset

# ============================================================
# 1. LOAD DATASET
# ============================================================

dataset = load_dataset(
    "lmarena-ai/arena-human-preference-140k"
)

df = dataset["train"].to_pandas()

print("\n========== DATASET OVERVIEW ==========")
print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst five observations:")
print(df.head().to_string())


# ============================================================
# 2. HUMAN PREFERENCE OUTCOMES
# ============================================================

print("\n========== WINNER DISTRIBUTION ==========")
print(df["winner"].value_counts())

print("\nWinner percentages:")
print(
    df["winner"].value_counts(normalize=True).mul(100).round(2)
)


# ============================================================
# 3. EXPLORE LLM MODELS
# ============================================================

all_models = pd.concat(
    [df["model_a"], df["model_b"]],
    ignore_index=True
)

print("\n========== MODEL INFORMATION ==========")
print("Number of unique models:", all_models.nunique())

print("\nTop 10 most frequently evaluated models:")
print(all_models.value_counts().head(10))


# ============================================================
# 4. CALCULATE WINS BY MODEL
# ============================================================

# Keep comparisons with a decisive winner
decisive = df[
    df["winner"].isin(["model_a", "model_b"])
].copy()

# Identify the actual model that won each comparison
decisive["winning_model"] = decisive.apply(
    lambda row: (
        row["model_a"]
        if row["winner"] == "model_a"
        else row["model_b"]
    ),
    axis=1
)

# Count wins for each model
model_wins = decisive["winning_model"].value_counts()

# Count participation in decisive comparisons
model_appearances = pd.concat(
    [decisive["model_a"], decisive["model_b"]],
    ignore_index=True
).value_counts()

# Combine wins and appearances
model_stats = pd.DataFrame({
    "wins": model_wins,
    "comparisons": model_appearances
}).fillna(0)

model_stats["wins"] = model_stats["wins"].astype(int)
model_stats["comparisons"] = model_stats["comparisons"].astype(int)

model_stats["win_rate"] = (
    model_stats["wins"] / model_stats["comparisons"]
)

print("\n========== MODEL WIN RATES ==========")
print(
    model_stats.sort_values(
        "win_rate", ascending=False
    ).head(15).to_string()
)

# Note: Raw win rates do not account for opponent strength.


# ============================================================
# 5. EXPLORE TASK CATEGORIES
# ============================================================

print("\n========== TASK CATEGORIES ==========")

print("\nFirst category tag:")
print(df["category_tag"].iloc[0])

# Extract category labels
df["math"] = df["category_tag"].apply(
    lambda x: x.get("math_v0.1", {}).get("math", False)
)

df["creative_writing"] = df["category_tag"].apply(
    lambda x: x.get("creative_writing_v0.1", {})
               .get("creative_writing", False)
)

df["coding"] = df["is_code"]

# Count task categories
print("\nTask category counts:")

for category in ["math", "coding", "creative_writing"]:
    count = df[category].sum()
    percentage = 100 * count / len(df)

    print(f"{category}: {count:,} ({percentage:.2f}%)")

# Winner outcomes by task category
print("\nWinner distribution by task category:")

for category in ["math", "coding", "creative_writing"]:
    print(f"\n{category.upper()}")

    subset = df[df[category]]
    print(subset["winner"].value_counts())


# ============================================================
# 6. LANGUAGE DISTRIBUTION
# ============================================================

print("\n========== LANGUAGE DISTRIBUTION ==========")

print("\nTop 10 languages:")
print(df["language"].value_counts().head(10))


# ============================================================
# 7. MISSING VALUES
# ============================================================

print("\n========== MISSING VALUES ==========")

print(
    df[
        ["model_a", "model_b", "winner",
         "category_tag", "is_code", "language"]
    ].isna().sum()
)


# ============================================================
# 8. MODEL MATCHUP FREQUENCIES
# ============================================================

print("\n========== MODEL MATCHUPS ==========")

# A vs B and B vs A are treated as the same matchup
df["pair"] = [
    tuple(sorted((a, b)))
    for a, b in zip(df["model_a"], df["model_b"])
]

pair_counts = df["pair"].value_counts()

print("\nNumber of unique observed model pairs:")
print(len(pair_counts))

print("\nTop 10 most frequent matchups:")
print(pair_counts.head(10).to_string())

print("\n10 least frequent observed matchups:")
print(pair_counts.tail(10).to_string())

# Number of theoretically possible distinct pairs
n_models = all_models.nunique()
possible_pairs = n_models * (n_models - 1) // 2

print("\nTheoretically possible model pairs:", possible_pairs)
print("Observed model pairs:", len(pair_counts))

# Check whether all potential matchups are represented
print(
    "Percentage of possible pairs observed:",
    round(100 * len(pair_counts) / possible_pairs, 2)
)


# ============================================================
# 9. SAVE CLEANED DATA FOR FUTURE RESEARCH
# ============================================================

# Save only columns needed for initial BTL research
research_df = df[
    [
        "id",
        "model_a",
        "model_b",
        "winner",
        "evaluation_session_id",
        "math",
        "coding",
        "creative_writing",
        "language"
    ]
].copy()

research_df.to_csv(
    "arena_research_data.csv",
    index=False
)

print("\n========== COMPLETE ==========")
print("Saved cleaned data to arena_research_data.csv")