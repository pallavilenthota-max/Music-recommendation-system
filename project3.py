
# MUSIC RECOMMENDATION SYSTEM
# Compare three content-based recommendation methods

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

# --------------------------------------------------
# STEP 1: DATASET PATH
# --------------------------------------------------

MUSIC_PATH = r"C:\Users\palla\OneDrive\Documents\ML PROJECTVDATSETS\MUSIC RECOMMENDATION SYSTEM\spotify_millsongdata.csv"

OUTPUT_DIR = "music_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# STEP 2: LOAD DATASET
# --------------------------------------------------

df = pd.read_csv(MUSIC_PATH)

df.columns = df.columns.astype(str).str.strip()

print("\nMUSIC RECOMMENDATION DATASET")
print("-" * 60)

print("\nFirst five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

# --------------------------------------------------
# STEP 3: CHECK REQUIRED COLUMNS
# --------------------------------------------------

required_columns = ["artist", "song", "text"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            "Required column not found: " + column +
            ". Check the column names printed above."
        )

# --------------------------------------------------
# STEP 4: CLEAN DATA
# --------------------------------------------------

df = df.dropna(
    subset=["artist", "song", "text"]
).copy()

df["artist"] = df["artist"].astype(str).str.strip()
df["song"] = df["song"].astype(str).str.strip()
df["text"] = df["text"].astype(str).str.strip()

df = df[
    (df["artist"] != "") &
    (df["song"] != "") &
    (df["text"] != "")
].copy()

# Remove duplicate songs by the same artist
df = df.drop_duplicates(
    subset=["artist", "song"]
).reset_index(drop=True)

print("\nDataset after cleaning:", df.shape)
print("Number of artists:", df["artist"].nunique())
print("Number of songs:", df["song"].nunique())

# --------------------------------------------------
# STEP 5: SELECT SONGS
# --------------------------------------------------
# A smaller subset helps the program run faster in Spyder.
# Increase this number if your computer has sufficient memory.

MAX_SONGS = 6000

if len(df) > MAX_SONGS:
    df = df.sample(
        n=MAX_SONGS,
        random_state=42
    ).reset_index(drop=True)

print("\nSongs selected for evaluation:", len(df))

# --------------------------------------------------
# STEP 6: PREPARE SONG FEATURES
# --------------------------------------------------

# Combine song title and lyrics.
# Artist names are not included in the text features.

df["combined_text"] = (
    df["song"] + " " + df["text"]
)

corpus = df["combined_text"].tolist()
artists = df["artist"].to_numpy()

# --------------------------------------------------
# STEP 7: DEFINE RECOMMENDATION METHODS
# --------------------------------------------------

vectorizers = {

    "TF-IDF Word": TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=15000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    ),

    "TF-IDF Character": TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        max_features=10000,
        min_df=2,
        sublinear_tf=True
    ),

    "Count Vectorizer": CountVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=15000,
        min_df=2,
        max_df=0.95
    )
}

# --------------------------------------------------
# STEP 8: SELECT SONGS FOR EVALUATION
# --------------------------------------------------

K = 10

QUERY_COUNT = min(200, len(df))

rng = np.random.default_rng(42)

query_indices = rng.choice(
    len(df),
    size=QUERY_COUNT,
    replace=False
)

# --------------------------------------------------
# STEP 9: TRAIN AND EVALUATE METHODS
# --------------------------------------------------

results = []
recommendation_objects = {}

for name, vectorizer in vectorizers.items():

    print("\nEvaluating method:", name)

    # Convert songs into numerical feature vectors
    song_matrix = vectorizer.fit_transform(corpus)

    # Find songs with similar content
    recommender = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_neighbors=min(K + 1, len(df))
    )

    recommender.fit(song_matrix)

    # Evaluate selected songs
    distances, indices = recommender.kneighbors(
        song_matrix[query_indices]
    )

    precision_scores = []
    recall_scores = []
    f1_scores = []

    for query_idx, neighbors in zip(
        query_indices, indices
    ):

        # Exclude the input song itself
        recommended_indices = [
            idx for idx in neighbors
            if idx != query_idx
        ][:K]

        if len(recommended_indices) == 0:
            continue

        # Same-artist songs are treated as relevant
        relevant_count = sum(
            artists[idx] == artists[query_idx]
            for idx in recommended_indices
        )

        precision = relevant_count / len(recommended_indices)

        total_relevant = max(
            int(np.sum(artists == artists[query_idx])) - 1,
            0
        )

        if total_relevant > 0:
            recall = relevant_count / total_relevant
        else:
            recall = 0.0

        if precision + recall > 0:
            f1 = (
                2 * precision * recall /
                (precision + recall)
            )
        else:
            f1 = 0.0

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    results.append({
        "Model": name,
        "Precision@10": np.mean(precision_scores),
        "Recall@10": np.mean(recall_scores),
        "F1-score@10": np.mean(f1_scores)
    })

    recommendation_objects[name] = (
        vectorizer,
        recommender,
        song_matrix
    )

# --------------------------------------------------
# STEP 10: DISPLAY MODEL COMPARISON
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1-score@10",
    ascending=False
).reset_index(drop=True)

print("\n" + "=" * 70)
print("MUSIC RECOMMENDATION MODEL COMPARISON")
print("=" * 70)

print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "music_model_comparison.csv"
    ),
    index=False
)

# --------------------------------------------------
# STEP 11: MODEL COMPARISON GRAPH
# --------------------------------------------------

results_df.set_index("Model")[
    ["Precision@10", "Recall@10", "F1-score@10"]
].plot(
    kind="bar",
    figsize=(10, 6),
    ylim=(0, 1)
)

plt.title("Music Recommendation - Model Comparison")
plt.xlabel("Recommendation Method")
plt.ylabel("Evaluation Score")
plt.xticks(rotation=20, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "music_model_comparison.png"
    ),
    dpi=300
)

plt.show()

# --------------------------------------------------
# STEP 12: SELECT BEST METHOD
# --------------------------------------------------

best_model_name = results_df.iloc[0]["Model"]

vectorizer, recommender, song_matrix = (
    recommendation_objects[best_model_name]
)

print("\nBEST RECOMMENDATION METHOD:")
print(best_model_name)

# --------------------------------------------------
# STEP 13: GENERATE SONG RECOMMENDATIONS
# --------------------------------------------------

query_idx = int(query_indices[0])

distances, indices = recommender.kneighbors(
    song_matrix[query_idx],
    n_neighbors=min(K + 1, len(df))
)

recommended_indices = [
    idx for idx in indices[0]
    if idx != query_idx
][:K]

recommendations = df.iloc[
    recommended_indices
][["song", "artist"]].copy()

recommendations.insert(
    0,
    "Rank",
    range(1, len(recommendations) + 1)
)

print("\nINPUT SONG:")
print(df.loc[query_idx, "song"])

print("\nARTIST:")
print(df.loc[query_idx, "artist"])

print("\nRECOMMENDED SONGS:")
print(recommendations.to_string(index=False))

recommendations.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "example_recommendations.csv"
    ),
    index=False
)

# --------------------------------------------------
# STEP 14: TOP 10 ARTISTS GRAPH
# --------------------------------------------------

top_artists = df["artist"].value_counts().head(10)

top_artists.sort_values().plot(
    kind="barh",
    figsize=(9, 5)
)

plt.title("Top 10 Artists in the Selected Music Dataset")
plt.xlabel("Number of Songs")
plt.ylabel("Artist")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "music_top_artists.png"
    ),
    dpi=300
)

plt.show()

# --------------------------------------------------
# STEP 15: FINAL OUTPUT
# --------------------------------------------------

print("\nMUSIC RECOMMENDATION PROJECT COMPLETED")

print(
    "\nResults and graphs saved in:",
    os.path.abspath(OUTPUT_DIR)
)

print(
    "\nNote: Evaluation treats songs by the same artist "
    "as relevant recommendations. This is only a proxy "
    "and does not measure actual listener preferences."
)

