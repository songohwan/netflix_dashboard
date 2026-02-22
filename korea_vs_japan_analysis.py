# %%
"""
Korea vs Japan Regional Comparison Analysis
============================================
Compares Netflix viewing patterns between Korea and Japan to identify:
1. Market-specific content preferences
2. Content that travels well between markets
3. Regional insights that may generalize to other Asian markets

This analysis directly addresses Netflix Content DSE requirements:
- "Identify and proactively socialize regional insights"
- "Including those that may generalize to opportunities in other markets"
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Setup
sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.1)
plt.rcParams["figure.dpi"] = 150
OUTPUT_DIR = "output/regional_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save(fig, name):
    fig.savefig(os.path.join(OUTPUT_DIR, name), bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ saved {name}")

print("=" * 80)
print("KOREA vs JAPAN: REGIONAL CONTENT ANALYSIS")
print("=" * 80)

# %%
# ─── Load enriched data ──────────────────────────────────────────────────────
print("\nLoading enriched data...")
kr_gw = pd.read_csv('dataset/raw/kr_gw_enriched.csv')
jp_gw = pd.read_csv('dataset/raw/jp_gw_enriched.csv')

kr_gw["week"] = pd.to_datetime(kr_gw["week"])
jp_gw["week"] = pd.to_datetime(jp_gw["week"])

# Filter to same time period for fair comparison
min_date = max(kr_gw["week"].min(), jp_gw["week"].min())
max_date = min(kr_gw["week"].max(), jp_gw["week"].max())

kr_gw = kr_gw[(kr_gw["week"] >= min_date) & (kr_gw["week"] <= max_date)]
jp_gw = jp_gw[(jp_gw["week"] >= min_date) & (jp_gw["week"] <= max_date)]

# Drop rows with missing TMDB data
kr_gw = kr_gw.dropna(subset=["tmdb_id"])
jp_gw = jp_gw.dropna(subset=["tmdb_id"])

# Create language labels
kr_gw["lang_label"] = kr_gw["original_language"].map(
    {"ko": "Korean", "en": "English", "ja": "Japanese", "zh": "Chinese"}
).fillna("Other")

jp_gw["lang_label"] = jp_gw["original_language"].map(
    {"ko": "Korean", "en": "English", "ja": "Japanese", "zh": "Chinese"}
).fillna("Other")

print(f"\nKorea data: {len(kr_gw):,} rows, {kr_gw['show_title'].nunique()} unique shows")
print(f"Japan data: {len(jp_gw):,} rows, {jp_gw['show_title'].nunique()} unique shows")
print(f"Time period: {min_date.date()} to {max_date.date()}")

# %%
# ═══════════════════════════════════════════════════════════════════════════
# 1. LANGUAGE PREFERENCES: Korea vs Japan
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("1. LANGUAGE PREFERENCES COMPARISON")
print("=" * 80)

kr_lang = kr_gw["lang_label"].value_counts(normalize=True) * 100
jp_lang = jp_gw["lang_label"].value_counts(normalize=True) * 100

lang_comparison = pd.DataFrame({
    "Korea": kr_lang,
    "Japan": jp_lang
}).fillna(0).sort_values("Korea", ascending=False)

print("\nLanguage distribution (%):")
print(lang_comparison.round(1))

# Calculate over/under-indexing
lang_comparison["Korea Index"] = lang_comparison["Korea"] / lang_comparison["Japan"]
lang_comparison["Japan Index"] = lang_comparison["Japan"] / lang_comparison["Korea"]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bar chart comparison
lang_comparison[["Korea", "Japan"]].plot.bar(ax=axes[0], color=["#FF6B6B", "#4ECDC4"])
axes[0].set_title("Language Distribution: Korea vs Japan", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Language")
axes[0].set_ylabel("% of Top 10 Entries")
axes[0].legend(title="Market")
axes[0].grid(axis='y', alpha=0.3)
plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right")

# Index heatmap
index_data = lang_comparison[["Korea Index"]].T
sns.heatmap(index_data, annot=True, fmt=".2f", cmap="RdYlGn", center=1.0,
            ax=axes[1], cbar_kws={'label': 'Index (>1 = Korea preference)'})
axes[1].set_title("Korea Over-Indexing by Language\n(Korea % / Japan %)",
                  fontsize=14, fontweight='bold')
axes[1].set_ylabel("")

fig.tight_layout()
plt.show()
save(fig, "01_language_comparison.png")

# Key insights
print("\n📊 KEY INSIGHTS:")
for lang in lang_comparison.index:
    kr_pct = lang_comparison.loc[lang, "Korea"]
    jp_pct = lang_comparison.loc[lang, "Japan"]
    diff = kr_pct - jp_pct
    if abs(diff) > 5:  # Only show significant differences
        direction = "higher" if diff > 0 else "lower"
        print(f"  • {lang}: Korea {kr_pct:.1f}% vs Japan {jp_pct:.1f}% "
              f"({abs(diff):.1f}pp {direction})")

# %%
# ═══════════════════════════════════════════════════════════════════════════
# 2. GENRE PREFERENCES: Korea vs Japan
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("2. GENRE PREFERENCES COMPARISON")
print("=" * 80)

# Explode genres
kr_genre = kr_gw.dropna(subset=["genres"]).copy()
kr_genre = kr_genre.assign(genre=kr_genre["genres"].str.split("|")).explode("genre")
kr_genre["genre"] = kr_genre["genre"].str.strip()

jp_genre = jp_gw.dropna(subset=["genres"]).copy()
jp_genre = jp_genre.assign(genre=jp_genre["genres"].str.split("|")).explode("genre")
jp_genre["genre"] = jp_genre["genre"].str.strip()

kr_genre_dist = kr_genre["genre"].value_counts(normalize=True) * 100
jp_genre_dist = jp_genre["genre"].value_counts(normalize=True) * 100

# Top 15 genres combined
top_genres = set(kr_genre_dist.head(15).index) | set(jp_genre_dist.head(15).index)
genre_comparison = pd.DataFrame({
    "Korea": kr_genre_dist,
    "Japan": jp_genre_dist
}).fillna(0)
genre_comparison = genre_comparison[genre_comparison.index.isin(top_genres)]
genre_comparison = genre_comparison.sort_values("Korea", ascending=False)

# Calculate index
genre_comparison["Difference"] = genre_comparison["Korea"] - genre_comparison["Japan"]
genre_comparison["Korea Index"] = genre_comparison["Korea"] / (genre_comparison["Japan"] + 0.1)

print("\nTop genres by market (%):")
print(genre_comparison[["Korea", "Japan", "Difference"]].round(1).head(15))

# Visualization
fig, axes = plt.subplots(2, 1, figsize=(14, 12))

# Top 15 genres comparison
top15_comparison = genre_comparison.head(15)
x = np.arange(len(top15_comparison))
width = 0.35

axes[0].barh(x - width/2, top15_comparison["Korea"], width, label="Korea", color="#FF6B6B")
axes[0].barh(x + width/2, top15_comparison["Japan"], width, label="Japan", color="#4ECDC4")
axes[0].set_yticks(x)
axes[0].set_yticklabels(top15_comparison.index)
axes[0].set_xlabel("% of Top 10 Entries")
axes[0].set_title("Top 15 Genres: Korea vs Japan", fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(axis='x', alpha=0.3)
axes[0].invert_yaxis()

# Difference chart (Korea over-indexing)
sorted_diff = genre_comparison.sort_values("Difference", ascending=True)
colors = ['#FF6B6B' if x > 0 else '#4ECDC4' for x in sorted_diff["Difference"]]
axes[1].barh(range(len(sorted_diff)), sorted_diff["Difference"], color=colors)
axes[1].set_yticks(range(len(sorted_diff)))
axes[1].set_yticklabels(sorted_diff.index)
axes[1].set_xlabel("Percentage Point Difference (Korea % - Japan %)")
axes[1].set_title("Genre Preferences: Korea Over/Under-Indexing vs Japan",
                  fontsize=14, fontweight='bold')
axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
axes[1].grid(axis='x', alpha=0.3)

fig.tight_layout()
plt.show()
save(fig, "02_genre_comparison.png")

# Key insights
print("\n📊 KOREA-SPECIFIC PREFERENCES (>3pp higher than Japan):")
korea_specific = genre_comparison[genre_comparison["Difference"] > 3].sort_values("Difference", ascending=False)
for genre in korea_specific.index:
    kr_pct = genre_comparison.loc[genre, "Korea"]
    jp_pct = genre_comparison.loc[genre, "Japan"]
    diff = genre_comparison.loc[genre, "Difference"]
    print(f"  • {genre}: Korea {kr_pct:.1f}% vs Japan {jp_pct:.1f}% (+{diff:.1f}pp)")

print("\n📊 JAPAN-SPECIFIC PREFERENCES (>3pp higher than Korea):")
japan_specific = genre_comparison[genre_comparison["Difference"] < -3].sort_values("Difference")
for genre in japan_specific.index:
    kr_pct = genre_comparison.loc[genre, "Korea"]
    jp_pct = genre_comparison.loc[genre, "Japan"]
    diff = abs(genre_comparison.loc[genre, "Difference"])
    print(f"  • {genre}: Japan {jp_pct:.1f}% vs Korea {kr_pct:.1f}% (+{diff:.1f}pp)")

# %%
# ═══════════════════════════════════════════════════════════════════════════
# 3. CONTENT LONGEVITY: Korea vs Japan
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("3. CONTENT LONGEVITY COMPARISON")
print("=" * 80)

# Calculate average weeks in top 10 per title
kr_longevity = kr_gw.groupby("show_title")["cumulative_weeks_in_top_10"].max()
jp_longevity = jp_gw.groupby("show_title")["cumulative_weeks_in_top_10"].max()

print(f"\nKorea - Avg weeks in top 10: {kr_longevity.mean():.2f}")
print(f"Japan - Avg weeks in top 10: {jp_longevity.mean():.2f}")
print(f"\nKorea - Median weeks: {kr_longevity.median():.0f}")
print(f"Japan - Median weeks: {jp_longevity.median():.0f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Distribution comparison
axes[0].hist(kr_longevity, bins=range(1, 20), alpha=0.7, label="Korea", color="#FF6B6B", density=True)
axes[0].hist(jp_longevity, bins=range(1, 20), alpha=0.7, label="Japan", color="#4ECDC4", density=True)
axes[0].set_xlabel("Weeks in Top 10")
axes[0].set_ylabel("Density")
axes[0].set_title("Content Longevity Distribution", fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Box plot comparison
data_for_box = pd.DataFrame({
    "Korea": kr_longevity,
    "Japan": jp_longevity
})
data_for_box.plot.box(ax=axes[1], patch_artist=True,
                       boxprops=dict(facecolor="#FFE5E5"),
                       medianprops=dict(color="red", linewidth=2))
axes[1].set_ylabel("Weeks in Top 10")
axes[1].set_title("Content Longevity: Korea vs Japan", fontsize=14, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

fig.tight_layout()
plt.show()
save(fig, "03_longevity_comparison.png")

# %%
# ═══════════════════════════════════════════════════════════════════════════
# 4. CONTENT THAT TRAVELS: Shared Hits
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("4. CONTENT THAT TRAVELS BETWEEN MARKETS")
print("=" * 80)

# Find titles that appear in both markets
kr_titles = set(kr_gw["show_title"].unique())
jp_titles = set(jp_gw["show_title"].unique())
shared_titles = kr_titles & jp_titles

print(f"\nShared titles (appear in both markets): {len(shared_titles)}")
print(f"Korea-only titles: {len(kr_titles - shared_titles)}")
print(f"Japan-only titles: {len(jp_titles - shared_titles)}")

# Analyze shared titles
shared_kr = kr_gw[kr_gw["show_title"].isin(shared_titles)].copy()
shared_jp = jp_gw[jp_gw["show_title"].isin(shared_titles)].copy()

# Get longevity for shared titles
shared_longevity = pd.DataFrame({
    "Korea_weeks": shared_kr.groupby("show_title")["cumulative_weeks_in_top_10"].max(),
    "Japan_weeks": shared_jp.groupby("show_title")["cumulative_weeks_in_top_10"].max()
})
shared_longevity["Total_weeks"] = shared_longevity["Korea_weeks"] + shared_longevity["Japan_weeks"]
shared_longevity = shared_longevity.sort_values("Total_weeks", ascending=False)

# Get metadata for shared titles
shared_meta = kr_gw[kr_gw["show_title"].isin(shared_titles)][
    ["show_title", "original_language", "genres", "tmdb_rating"]
].drop_duplicates("show_title").set_index("show_title")

shared_longevity = shared_longevity.join(shared_meta)

print("\nTop 20 titles that succeeded in BOTH markets:")
print(shared_longevity.head(20)[["Korea_weeks", "Japan_weeks", "Total_weeks",
                                   "original_language", "genres"]].to_string())

# Analyze what makes content "travel well"
print("\n📊 CHARACTERISTICS OF CONTENT THAT TRAVELS:")

# By language
shared_lang = shared_longevity.groupby("original_language").size().sort_values(ascending=False)
print(f"\nBy language (top shared titles):")
for lang, count in shared_lang.head().items():
    pct = count / len(shared_longevity) * 100
    print(f"  • {lang}: {count} titles ({pct:.1f}%)")

# By genre
shared_genre_list = []
for genres in shared_longevity["genres"].dropna():
    shared_genre_list.extend(genres.split("|"))
shared_genre_counts = Counter(shared_genre_list)
print(f"\nTop genres in shared titles:")
for genre, count in shared_genre_counts.most_common(10):
    print(f"  • {genre}: {count} titles")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Scatter plot: Korea weeks vs Japan weeks for shared content
axes[0].scatter(shared_longevity["Korea_weeks"], shared_longevity["Japan_weeks"],
                alpha=0.6, s=50, color="#9B59B6")
axes[0].plot([0, shared_longevity[["Korea_weeks", "Japan_weeks"]].max().max()],
             [0, shared_longevity[["Korea_weeks", "Japan_weeks"]].max().max()],
             'r--', alpha=0.5, label="Equal performance")
axes[0].set_xlabel("Weeks in Korea Top 10")
axes[0].set_ylabel("Weeks in Japan Top 10")
axes[0].set_title("Shared Content Performance: Korea vs Japan",
                  fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Top 15 shared titles
top_shared = shared_longevity.head(15).copy()
x = np.arange(len(top_shared))
width = 0.35

axes[1].barh(x - width/2, top_shared["Korea_weeks"], width, label="Korea", color="#FF6B6B")
axes[1].barh(x + width/2, top_shared["Japan_weeks"], width, label="Japan", color="#4ECDC4")
axes[1].set_yticks(x)
axes[1].set_yticklabels([title[:30] + "..." if len(title) > 30 else title
                         for title in top_shared.index])
axes[1].set_xlabel("Weeks in Top 10")
axes[1].set_title("Top 15 Titles That Succeeded in Both Markets",
                  fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(axis='x', alpha=0.3)
axes[1].invert_yaxis()

fig.tight_layout()
plt.show()
save(fig, "04_shared_content.png")

# %%
# ═══════════════════════════════════════════════════════════════════════════
# 5. CATEGORY PREFERENCES (Films vs TV)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("5. CATEGORY PREFERENCES: Films vs TV")
print("=" * 80)

kr_cat = kr_gw["category"].value_counts(normalize=True) * 100
jp_cat = jp_gw["category"].value_counts(normalize=True) * 100

cat_comparison = pd.DataFrame({
    "Korea": kr_cat,
    "Japan": jp_cat
})

print("\nCategory distribution (%):")
print(cat_comparison.round(1))

fig, ax = plt.subplots(figsize=(10, 6))
cat_comparison.plot.bar(ax=ax, color=["#FF6B6B", "#4ECDC4"])
ax.set_title("Category Preferences: Korea vs Japan", fontsize=14, fontweight='bold')
ax.set_xlabel("Category")
ax.set_ylabel("% of Top 10 Entries")
ax.legend(title="Market")
ax.grid(axis='y', alpha=0.3)
plt.setp(ax.get_xticklabels(), rotation=0)
fig.tight_layout()
plt.show()
save(fig, "05_category_comparison.png")

# %%
# ═══════════════════════════════════════════════════════════════════════════
# 6. KOREAN CONTENT PENETRATION IN JAPAN
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("6. KOREAN CONTENT PENETRATION IN JAPAN")
print("=" * 80)

kr_content_in_japan = jp_gw[jp_gw["original_language"] == "ko"]
print(f"\nKorean content entries in Japan top 10: {len(kr_content_in_japan)}")
print(f"% of Japan top 10: {len(kr_content_in_japan) / len(jp_gw) * 100:.1f}%")
print(f"Unique Korean titles in Japan: {kr_content_in_japan['show_title'].nunique()}")

# Top Korean titles in Japan
kr_in_jp_longevity = kr_content_in_japan.groupby("show_title")["cumulative_weeks_in_top_10"].max()
kr_in_jp_longevity = kr_in_jp_longevity.sort_values(ascending=False)

print("\nTop 10 Korean titles in Japan (by weeks in top 10):")
for i, (title, weeks) in enumerate(kr_in_jp_longevity.head(10).items(), 1):
    print(f"  {i:2}. {title:50s} {weeks:2.0f} weeks")

# Japanese content in Korea
jp_content_in_korea = kr_gw[kr_gw["original_language"] == "ja"]
print(f"\nJapanese content entries in Korea top 10: {len(jp_content_in_korea)}")
print(f"% of Korea top 10: {len(jp_content_in_korea) / len(kr_gw) * 100:.1f}%")
print(f"Unique Japanese titles in Korea: {jp_content_in_korea['show_title'].nunique()}")

# %%
# ═══════════════════════════════════════════════════════════════════════════
# 7. CHARACTERISTICS OF SUCCESSFUL KOREAN CONTENT IN JAPAN
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("7. CHARACTERISTICS OF SUCCESSFUL KOREAN CONTENT IN JAPAN")
print("=" * 80)

# Get Korean content that appears in both markets (drop tmdb_id nulls)
korean_content_kr = kr_gw[(kr_gw["original_language"] == "ko") & (kr_gw["tmdb_id"].notna())]
korean_content_jp = jp_gw[(jp_gw["original_language"] == "ko") & (jp_gw["tmdb_id"].notna())]

kr_korean_titles = set(korean_content_kr["show_title"].unique())
jp_korean_titles = set(korean_content_jp["show_title"].unique())

# Korean content that succeeded in BOTH markets
korean_shared = kr_korean_titles & jp_korean_titles
# Korean content that stayed in Korea only
korean_kr_only = kr_korean_titles - jp_korean_titles

print(f"\nKorean content in both markets: {len(korean_shared)}")
print(f"Korean content in Korea only: {len(korean_kr_only)}")

# Get detailed data for comparison
korean_in_both = korean_content_kr[korean_content_kr["show_title"].isin(korean_shared)].copy()
korean_kr_only_data = korean_content_kr[korean_content_kr["show_title"].isin(korean_kr_only)].copy()

print("\n" + "-" * 80)
print("A. GENRE ANALYSIS")
print("-" * 80)

# Genre distribution - Korean content in Japan
korean_jp_genres = korean_content_jp.dropna(subset=["genres"]).copy()
korean_jp_genres = korean_jp_genres.assign(genre=korean_jp_genres["genres"].str.split("|")).explode("genre")
korean_jp_genres["genre"] = korean_jp_genres["genre"].str.strip()
korean_jp_genre_dist = korean_jp_genres["genre"].value_counts(normalize=True) * 100

# Genre distribution - Korean content Korea-only
korean_kr_only_genres = korean_kr_only_data.dropna(subset=["genres"]).copy()
korean_kr_only_genres = korean_kr_only_genres.assign(genre=korean_kr_only_genres["genres"].str.split("|")).explode("genre")
korean_kr_only_genres["genre"] = korean_kr_only_genres["genre"].str.strip()
korean_kr_only_genre_dist = korean_kr_only_genres["genre"].value_counts(normalize=True) * 100

# Compare
korean_genre_comp = pd.DataFrame({
    "In_Japan": korean_jp_genre_dist,
    "Korea_Only": korean_kr_only_genre_dist
}).fillna(0)
korean_genre_comp["Difference"] = korean_genre_comp["In_Japan"] - korean_genre_comp["Korea_Only"]
korean_genre_comp = korean_genre_comp.sort_values("Difference", ascending=False)

print("\nGenre distribution (%) - Korean content:")
print(korean_genre_comp.head(15).round(1))

print("\n📊 GENRES THAT TRAVEL TO JAPAN (>5pp higher):")
for genre in korean_genre_comp[korean_genre_comp["Difference"] > 5].index:
    in_jp = korean_genre_comp.loc[genre, "In_Japan"]
    kr_only = korean_genre_comp.loc[genre, "Korea_Only"]
    diff = korean_genre_comp.loc[genre, "Difference"]
    print(f"  • {genre}: In Japan {in_jp:.1f}% vs Korea-only {kr_only:.1f}% (+{diff:.1f}pp)")

print("\n" + "-" * 80)
print("B. CATEGORY ANALYSIS (Films vs TV)")
print("-" * 80)

korean_jp_cat = korean_content_jp["category"].value_counts(normalize=True) * 100
korean_kr_cat = korean_kr_only_data["category"].value_counts(normalize=True) * 100

cat_comp = pd.DataFrame({
    "In_Japan": korean_jp_cat,
    "Korea_Only": korean_kr_cat
}).fillna(0)

print("\nCategory distribution (%) - Korean content:")
print(cat_comp.round(1))

print("\n" + "-" * 80)
print("C. QUALITY METRICS")
print("-" * 80)

# Compare ratings - filter out 0 and null values
korean_jp_meta = korean_content_jp.drop_duplicates("show_title")
korean_kr_only_meta = korean_kr_only_data.drop_duplicates("show_title")

# Filter out null and 0 ratings
korean_jp_meta_rated = korean_jp_meta[(korean_jp_meta["tmdb_rating"].notna()) & (korean_jp_meta["tmdb_rating"] > 0)]
korean_kr_only_meta_rated = korean_kr_only_meta[(korean_kr_only_meta["tmdb_rating"].notna()) & (korean_kr_only_meta["tmdb_rating"] > 0)]

avg_rating_jp = korean_jp_meta_rated["tmdb_rating"].mean()
avg_rating_kr_only = korean_kr_only_meta_rated["tmdb_rating"].mean()

print(f"\nAverage TMDB rating (excluding 0/null):")
print(f"  • Korean content in Japan: {avg_rating_jp:.2f} (n={len(korean_jp_meta_rated)})")
print(f"  • Korean content Korea-only: {avg_rating_kr_only:.2f} (n={len(korean_kr_only_meta_rated)})")
print(f"  • Difference: {avg_rating_jp - avg_rating_kr_only:+.2f}")

# Compare longevity
korean_jp_longevity = korean_content_jp.groupby("show_title")["cumulative_weeks_in_top_10"].max()
korean_kr_longevity = korean_kr_only_data.groupby("show_title")["cumulative_weeks_in_top_10"].max()

print(f"\nAverage weeks in top 10 (Korea market):")
print(f"  • Korean content that reached Japan: {korean_jp_longevity.mean():.2f}")
print(f"  • Korean content Korea-only: {korean_kr_longevity.mean():.2f}")

print("\n" + "-" * 80)
print("D. TOP KOREAN TITLES IN JAPAN - DETAILED ANALYSIS")
print("-" * 80)

# Get top Korean titles in Japan with full metadata
top_kr_in_jp = korean_content_jp.groupby("show_title").agg({
    "cumulative_weeks_in_top_10": "max",
    "genres": "first",
    "category": "first",
    "tmdb_rating": "first"
}).sort_values("cumulative_weeks_in_top_10", ascending=False)

print("\nTop 15 Korean titles in Japan (detailed):")
print(top_kr_in_jp.head(15).to_string())

# Analyze common characteristics
print("\n📊 COMMON CHARACTERISTICS OF TOP KOREAN CONTENT IN JAPAN:")
top_10_kr_in_jp = top_kr_in_jp.head(10)

# Genre analysis
top_genres_list = []
for genres in top_10_kr_in_jp["genres"].dropna():
    top_genres_list.extend(genres.split("|"))
top_genre_counts = Counter(top_genres_list)
print(f"\nMost common genres in top 10:")
for genre, count in top_genre_counts.most_common(5):
    print(f"  • {genre}: {count}/{len(top_10_kr_in_jp)} titles")

# Category
print(f"\nCategory breakdown (top 10):")
for cat, count in top_10_kr_in_jp["category"].value_counts().items():
    print(f"  • {cat}: {count} titles")

# Ratings
print(f"\nAverage rating of top 10: {top_10_kr_in_jp['tmdb_rating'].mean():.2f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Genre comparison
top_genres = korean_genre_comp.head(12)
x = np.arange(len(top_genres))
width = 0.35

axes[0, 0].barh(x - width/2, top_genres["In_Japan"], width, label="Reached Japan", color="#FF6B6B")
axes[0, 0].barh(x + width/2, top_genres["Korea_Only"], width, label="Korea Only", color="#95A5A6")
axes[0, 0].set_yticks(x)
axes[0, 0].set_yticklabels(top_genres.index)
axes[0, 0].set_xlabel("% of Entries")
axes[0, 0].set_title("Korean Content Genres: Japan vs Korea-Only", fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(axis='x', alpha=0.3)
axes[0, 0].invert_yaxis()

# 2. Category comparison
cat_comp.plot.bar(ax=axes[0, 1], color=["#FF6B6B", "#95A5A6"])
axes[0, 1].set_title("Korean Content Category: Japan vs Korea-Only", fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel("Category")
axes[0, 1].set_ylabel("% of Entries")
axes[0, 1].legend(title="Market Reach")
axes[0, 1].grid(axis='y', alpha=0.3)
plt.setp(axes[0, 1].get_xticklabels(), rotation=0)

# 3. Rating distribution (excluding 0 and null values)
axes[1, 0].hist(korean_jp_meta_rated["tmdb_rating"], bins=20, alpha=0.7,
                label=f"Reached Japan (avg: {avg_rating_jp:.2f})", color="#FF6B6B", density=True)
axes[1, 0].hist(korean_kr_only_meta_rated["tmdb_rating"], bins=20, alpha=0.7,
                label=f"Korea Only (avg: {avg_rating_kr_only:.2f})", color="#95A5A6", density=True)
axes[1, 0].set_xlabel("TMDB Rating (0/null excluded)")
axes[1, 0].set_ylabel("Density")
axes[1, 0].set_title("Korean Content Quality Distribution", fontsize=12, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# 4. Top Korean titles in Japan
top_15_titles = top_kr_in_jp.head(15)
axes[1, 1].barh(range(len(top_15_titles)), top_15_titles["cumulative_weeks_in_top_10"], color="#FF6B6B")
axes[1, 1].set_yticks(range(len(top_15_titles)))
axes[1, 1].set_yticklabels([title[:35] + "..." if len(title) > 35 else title
                            for title in top_15_titles.index])
axes[1, 1].set_xlabel("Weeks in Japan Top 10")
axes[1, 1].set_title("Top 15 Korean Titles in Japan", fontsize=12, fontweight='bold')
axes[1, 1].grid(axis='x', alpha=0.3)
axes[1, 1].invert_yaxis()

fig.tight_layout()
plt.show()
save(fig, "06_korean_content_characteristics.png")

# Additional visualization: Rating distribution by genre
print("\n" + "-" * 80)
print("E. RATING DISTRIBUTION BY GENRE (Korean content that succeeded in Japan)")
print("-" * 80)

# Prepare data: Korean content in Japan only, explode genres and filter out 0/null ratings
korean_jp_rated = korean_content_jp.drop_duplicates("show_title")
korean_jp_rated = korean_jp_rated[(korean_jp_rated["tmdb_rating"].notna()) & (korean_jp_rated["tmdb_rating"] > 0)]

korean_genre_rating = korean_jp_rated.dropna(subset=["genres"]).copy()
korean_genre_rating = korean_genre_rating.assign(genre=korean_genre_rating["genres"].str.split("|")).explode("genre")
korean_genre_rating["genre"] = korean_genre_rating["genre"].str.strip()

# Get top genres by frequency
top_genres_for_rating = korean_genre_rating["genre"].value_counts().head(12).index

# Filter to top genres
korean_genre_rating_filtered = korean_genre_rating[korean_genre_rating["genre"].isin(top_genres_for_rating)]

# Calculate average rating per genre
genre_avg_rating = korean_genre_rating_filtered.groupby("genre")["tmdb_rating"].mean().sort_values(ascending=False)

print("\nAverage rating by genre for Korean content in Japan (top 12 genres):")
for genre in genre_avg_rating.index:
    avg = genre_avg_rating[genre]
    count = len(korean_genre_rating_filtered[korean_genre_rating_filtered["genre"] == genre])
    print(f"  • {genre:20s}: {avg:.2f} (n={count})")



# Create boxplot
fig, ax = plt.subplots(figsize=(14, 8))

# Prepare data for boxplot - sort by median rating
genre_order = korean_genre_rating_filtered.groupby("genre")["tmdb_rating"].median().sort_values(ascending=False).index
genre_order = [g for g in genre_order if g in top_genres_for_rating]

# Create boxplot
bp = ax.boxplot([korean_genre_rating_filtered[korean_genre_rating_filtered["genre"] == genre]["tmdb_rating"].values
                 for genre in genre_order],
                labels=genre_order,
                patch_artist=True,
                showmeans=True,
                meanprops=dict(marker='D', markerfacecolor='red', markersize=6))

# Color the boxes
for patch in bp['boxes']:
    patch.set_facecolor('#FF6B6B')
    patch.set_alpha(0.6)

ax.set_xlabel("Genre", fontsize=12)
ax.set_ylabel("TMDB Rating (0/null excluded)", fontsize=12)
ax.set_title("Korean Content in Japan: Rating Distribution by Genre", fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

# Add horizontal line at overall mean
overall_mean = korean_jp_rated["tmdb_rating"].mean()
ax.axhline(y=overall_mean, color='gray', linestyle='--', alpha=0.5,
           label=f'Overall Mean: {overall_mean:.2f}')
ax.legend()

fig.tight_layout()
plt.show()
save(fig, "07_rating_by_genre_boxplot.png")

# %%
# Additional analysis: Distribution of cumulative weeks for Korean content in Japan
print("\n" + "-" * 80)
print("E2. CUMULATIVE WEEKS DISTRIBUTION (Korean content in Japan)")
print("-" * 80)

# Get cumulative weeks for Korean content in Japan
kr_jp_weeks = korean_content_jp.groupby("show_title")["cumulative_weeks_in_top_10"].max()

print(f"\nKorean content performance in Japan's Top 10:")
print(f"  • Total titles: {len(kr_jp_weeks)}")
print(f"  • Average weeks: {kr_jp_weeks.mean():.2f}")
print(f"  • Median weeks: {kr_jp_weeks.median():.0f}")
print(f"  • Min weeks: {kr_jp_weeks.min():.0f}")
print(f"  • Max weeks: {kr_jp_weeks.max():.0f}")
print(f"  • Std deviation: {kr_jp_weeks.std():.2f}")


# %%
# Quartile analysis
Q1 = kr_jp_weeks.quantile(0.25)
Q2 = kr_jp_weeks.quantile(0.50)
Q3 = kr_jp_weeks.quantile(0.75)

print(f"\nQuartile breakdown:")
print(f"  • Q1 (25th percentile): {Q1:.0f} weeks")
print(f"  • Q2 (50th percentile): {Q2:.0f} weeks")
print(f"  • Q3 (75th percentile): {Q3:.0f} weeks")

# Performance categories
short_run = (kr_jp_weeks <= 3).sum()
medium_run = ((kr_jp_weeks > 3) & (kr_jp_weeks <= 10)).sum()
long_run = (kr_jp_weeks > 10).sum()

print(f"\nPerformance categories:")
print(f"  • Short run (≤3 weeks): {short_run} titles ({short_run/len(kr_jp_weeks)*100:.1f}%)")
print(f"  • Medium run (4-10 weeks): {medium_run} titles ({medium_run/len(kr_jp_weeks)*100:.1f}%)")
print(f"  • Long run (>10 weeks): {long_run} titles ({long_run/len(kr_jp_weeks)*100:.1f}%)")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 1. Histogram
axes[0].hist(kr_jp_weeks, bins=30, color="#FF6B6B", alpha=0.7, edgecolor='black')
axes[0].axvline(kr_jp_weeks.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {kr_jp_weeks.mean():.1f} weeks')
axes[0].axvline(kr_jp_weeks.median(), color='blue', linestyle='--', linewidth=2,
                label=f'Median: {kr_jp_weeks.median():.0f} weeks')
axes[0].set_xlabel("Cumulative Weeks in Japan Top 10", fontsize=12)
axes[0].set_ylabel("Number of Titles", fontsize=12)
axes[0].set_title("Distribution of Korean Content Performance in Japan",
                  fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3, axis='y')

# 2. Box plot with stats
bp = axes[1].boxplot([kr_jp_weeks], vert=True, patch_artist=True,
                      showmeans=True, widths=0.5,
                      meanprops=dict(marker='D', markerfacecolor='red', markersize=10))
bp['boxes'][0].set_facecolor('#FF6B6B')
bp['boxes'][0].set_alpha(0.7)

# Add annotations
axes[1].text(1.3, Q1, f'Q1: {Q1:.0f}', va='center', fontsize=10)
axes[1].text(1.3, Q2, f'Q2: {Q2:.0f}', va='center', fontsize=10, fontweight='bold')
axes[1].text(1.3, Q3, f'Q3: {Q3:.0f}', va='center', fontsize=10)
axes[1].text(1.3, kr_jp_weeks.mean(), f'Mean: {kr_jp_weeks.mean():.1f}',
             va='center', fontsize=10, color='red')

axes[1].set_ylabel("Cumulative Weeks in Japan Top 10", fontsize=12)
axes[1].set_title("Korean Content Longevity in Japan (Box Plot)",
                  fontsize=14, fontweight='bold')
axes[1].set_xticks([1])
axes[1].set_xticklabels(['Korean Content\nin Japan'])
axes[1].grid(alpha=0.3, axis='y')

fig.tight_layout()
plt.show()
save(fig, "07b_korean_weeks_distribution.png")

print("\n📊 KEY INSIGHT:")
print(f"  • Most Korean content ({medium_run+long_run}/{len(kr_jp_weeks)}) stays in Japan's Top 10 for 4+ weeks")
print(f"  • {long_run} titles ({long_run/len(kr_jp_weeks)*100:.1f}%) achieve long-run success (>10 weeks)")
print(f"  • The top 25% of Korean content stays in Top 10 for {Q3:.0f}+ weeks")

# %%
print("\n" + "-" * 80)
print("F. CONTENT FEATURES IMPACT ON JAPAN PENETRATION")
print("-" * 80)

# Get unique titles data for comparison
korean_jp_unique = korean_content_jp.drop_duplicates("show_title")
korean_kr_only_unique = korean_kr_only_data.drop_duplicates("show_title")

print(f"\nAnalyzing {len(korean_jp_unique)} Korean titles in Japan vs {len(korean_kr_only_unique)} Korea-only titles")

# 1. NUM_EPISODES ANALYSIS
print("\n1. EPISODE COUNT ANALYSIS:")
print("-" * 40)

# Filter out TV shows only (films don't have meaningful episode counts)
korean_jp_tv = korean_jp_unique[korean_jp_unique["category"] == "TV"]
korean_kr_tv = korean_kr_only_unique[korean_kr_only_unique["category"] == "TV"]

if "num_episodes" in korean_jp_tv.columns:
    jp_episodes = korean_jp_tv["num_episodes"].dropna()
    kr_episodes = korean_kr_tv["num_episodes"].dropna()

    # Calculate IQR for outlier detection (using combined data for consistent threshold)
    all_episodes = pd.concat([jp_episodes, kr_episodes])
    Q1 = all_episodes.quantile(0.25)
    Q3 = all_episodes.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print(f"IQR-based outlier detection:")
    print(f"  • Q1 (25th percentile): {Q1:.1f} episodes")
    print(f"  • Q3 (75th percentile): {Q3:.1f} episodes")
    print(f"  • IQR: {IQR:.1f} episodes")
    print(f"  • Outlier threshold: {lower_bound:.1f} - {upper_bound:.1f} episodes")

    print(f"\nKorean TV shows that reached Japan:")
    print(f"  • Average episodes: {jp_episodes.mean():.1f}")
    print(f"  • Median episodes: {jp_episodes.median():.0f}")
    print(f"  • Range: {jp_episodes.min():.0f} - {jp_episodes.max():.0f}")

    # Count outliers using IQR method
    jp_outliers = ((jp_episodes < lower_bound) | (jp_episodes > upper_bound)).sum()
    if jp_outliers > 0:
        print(f"  • Outliers (IQR method): {jp_outliers} shows excluded from visualization")

    print(f"\nKorean TV shows Korea-only:")
    print(f"  • Average episodes: {kr_episodes.mean():.1f}")
    print(f"  • Median episodes: {kr_episodes.median():.0f}")
    print(f"  • Range: {kr_episodes.min():.0f} - {kr_episodes.max():.0f}")

    kr_outliers = ((kr_episodes < lower_bound) | (kr_episodes > upper_bound)).sum()
    if kr_outliers > 0:
        print(f"  • Outliers (IQR method): {kr_outliers} shows excluded from visualization")

    print(f"\n📊 INSIGHT: Korean shows in Japan have {jp_episodes.mean() - kr_episodes.mean():+.1f} episodes on average")

# 2. CASTING POPULARITY ANALYSIS
print("\n2. CASTING POPULARITY ANALYSIS:")
print("-" * 40)

if "avg_cast_popularity" in korean_jp_unique.columns:
    # Filter out null and 0 values
    jp_cast = korean_jp_unique["avg_cast_popularity"]
    jp_cast = jp_cast[(jp_cast.notna()) & (jp_cast > 0)]

    kr_cast = korean_kr_only_unique["avg_cast_popularity"]
    kr_cast = kr_cast[(kr_cast.notna()) & (kr_cast > 0)]

    print(f"Korean content that reached Japan (excluding 0/null):")
    print(f"  • Average casting popularity: {jp_cast.mean():.2f}")
    print(f"  • Median casting popularity: {jp_cast.median():.2f}")
    print(f"  • Sample size: {len(jp_cast)}")

    print(f"\nKorean content Korea-only (excluding 0/null):")
    print(f"  • Average casting popularity: {kr_cast.mean():.2f}")
    print(f"  • Median casting popularity: {kr_cast.median():.2f}")
    print(f"  • Sample size: {len(kr_cast)}")

    diff_pct = ((jp_cast.mean() / kr_cast.mean()) - 1) * 100
    print(f"\n📊 INSIGHT: Korean content in Japan has {diff_pct:+.1f}% higher casting popularity")

# 3. SEASON NUMBER ANALYSIS
print("\n3. SEASON NUMBER ANALYSIS:")
print("-" * 40)

if "season_number" in korean_jp_tv.columns:
    jp_seasons = korean_jp_tv["season_number"].dropna()
    kr_seasons = korean_kr_tv["season_number"].dropna()

    print(f"Korean TV shows that reached Japan:")
    print(f"  • Average season: {jp_seasons.mean():.2f}")
    print(f"  • Season 1: {(jp_seasons == 1).sum()} shows ({(jp_seasons == 1).sum()/len(jp_seasons)*100:.1f}%)")
    print(f"  • Season 2+: {(jp_seasons > 1).sum()} shows ({(jp_seasons > 1).sum()/len(jp_seasons)*100:.1f}%)")

    print(f"\nKorean TV shows Korea-only:")
    print(f"  • Average season: {kr_seasons.mean():.2f}")
    print(f"  • Season 1: {(kr_seasons == 1).sum()} shows ({(kr_seasons == 1).sum()/len(kr_seasons)*100:.1f}%)")
    print(f"  • Season 2+: {(kr_seasons > 1).sum()} shows ({(kr_seasons > 1).sum()/len(kr_seasons)*100:.1f}%)")

# 4. IS_SEQUEL ANALYSIS
print("\n4. SEQUEL STATUS ANALYSIS:")
print("-" * 40)

if "is_sequel" in korean_jp_unique.columns:
    jp_sequel_rate = korean_jp_unique["is_sequel"].mean() * 100
    kr_sequel_rate = korean_kr_only_unique["is_sequel"].mean() * 100

    jp_sequel_count = korean_jp_unique["is_sequel"].sum()
    kr_sequel_count = korean_kr_only_unique["is_sequel"].sum()

    print(f"Korean content that reached Japan:")
    print(f"  • Sequels: {jp_sequel_count} ({jp_sequel_rate:.1f}%)")
    print(f"  • Original content: {len(korean_jp_unique) - jp_sequel_count} ({100-jp_sequel_rate:.1f}%)")

    print(f"\nKorean content Korea-only:")
    print(f"  • Sequels: {kr_sequel_count} ({kr_sequel_rate:.1f}%)")
    print(f"  • Original content: {len(korean_kr_only_unique) - kr_sequel_count} ({100-kr_sequel_rate:.1f}%)")

    print(f"\n📊 INSIGHT: Korean content in Japan is {jp_sequel_rate - kr_sequel_rate:+.1f}pp more likely to be a sequel")

# VISUALIZATION
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Episode count distribution (TV shows only) - remove outliers using IQR
if "num_episodes" in korean_jp_tv.columns:
    # Filter out outliers using IQR method
    jp_episodes_filtered = jp_episodes[(jp_episodes >= lower_bound) & (jp_episodes <= upper_bound)]
    kr_episodes_filtered = kr_episodes[(kr_episodes >= lower_bound) & (kr_episodes <= upper_bound)]

    axes[0, 0].hist(kr_episodes_filtered, bins=20, alpha=0.7,
                    label=f"Korea Only (avg: {kr_episodes_filtered.mean():.1f})",
                    color="#95A5A6", density=True)
    axes[0, 0].hist(jp_episodes_filtered, bins=20, alpha=0.7,
                    label=f"Reached Japan (avg: {jp_episodes_filtered.mean():.1f})",
                    color="#FF6B6B", density=True)
    axes[0, 0].set_xlabel("Number of Episodes")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].set_title(f"Episode Count Distribution (TV Shows, IQR filtered)",
                        fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

# 2. Casting popularity distribution (0/null excluded)
if "avg_cast_popularity" in korean_jp_unique.columns:
    axes[0, 1].hist(kr_cast, bins=30, alpha=0.7, label=f"Korea Only (avg: {kr_cast.mean():.1f})",
                    color="#95A5A6", density=True)
    axes[0, 1].hist(jp_cast, bins=30, alpha=0.7, label=f"Reached Japan (avg: {jp_cast.mean():.1f})",
                    color="#FF6B6B", density=True)
    axes[0, 1].set_xlabel("Average Casting Popularity (0/null excluded)")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].set_title("Casting Popularity Distribution", fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

# 3. Season number comparison
if "season_number" in korean_jp_tv.columns:
    season_data = pd.DataFrame({
        "Reached Japan": jp_seasons.value_counts(normalize=True).sort_index() * 100,
        "Korea Only": kr_seasons.value_counts(normalize=True).sort_index() * 100
    }).fillna(0)

    season_data.plot.bar(ax=axes[1, 0], color=["#FF6B6B", "#95A5A6"])
    axes[1, 0].set_xlabel("Season Number")
    axes[1, 0].set_ylabel("% of TV Shows")
    axes[1, 0].set_title("Season Distribution (TV Shows)", fontsize=12, fontweight='bold')
    axes[1, 0].legend(title="Market Reach")
    axes[1, 0].grid(axis='y', alpha=0.3)
    plt.setp(axes[1, 0].get_xticklabels(), rotation=0)

# 4. Sequel status comparison
if "is_sequel" in korean_jp_unique.columns:
    sequel_comparison = pd.DataFrame({
        "Reached Japan": [100-jp_sequel_rate, jp_sequel_rate],
        "Korea Only": [100-kr_sequel_rate, kr_sequel_rate]
    }, index=["Original", "Sequel"])

    sequel_comparison.plot.bar(ax=axes[1, 1], color=["#FF6B6B", "#95A5A6"])
    axes[1, 1].set_xlabel("Content Type")
    axes[1, 1].set_ylabel("% of Content")
    axes[1, 1].set_title("Sequel vs Original Content", fontsize=12, fontweight='bold')
    axes[1, 1].legend(title="Market Reach")
    axes[1, 1].grid(axis='y', alpha=0.3)
    plt.setp(axes[1, 1].get_xticklabels(), rotation=0)

fig.tight_layout()
plt.show()
save(fig, "08_content_features_analysis.png")

print("\n" + "=" * 80)
print("KEY FINDINGS: Content Features Impact on Japan Penetration")
print("=" * 80)
print("""
Korean content that successfully penetrates the Japanese market tends to have:
1. EPISODES: Specific episode count patterns (analysis above)
2. CASTING: Higher star power (measured by avg casting popularity)
3. SEASONS: Distribution across season numbers (fresh vs established series)
4. SEQUELS: Different sequel/original content mix

These insights can inform content acquisition and commissioning decisions.
""")

# %%
# ═══════════════════════════════════════════════════════════════════════════
# 8. FEATURE ANALYSIS: KR DRAMAS WITH 5+ WEEKS TOP 10 SUCCESS IN JAPAN
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("8. FEATURE ANALYSIS: KR DRAMAS WITH 5+ WEEKS TOP 10 SUCCESS IN JAPAN")
print("=" * 80)

# Filter Korean TV content in Japan with 5+ weeks
kr_tv_in_japan = korean_content_jp[korean_content_jp["category"] == "TV"].copy()
kr_tv_longevity = kr_tv_in_japan.groupby("show_title")["cumulative_weeks_in_top_10"].max()

# Split into success tiers
successful_titles = kr_tv_longevity[kr_tv_longevity > 5].index
less_successful_titles = kr_tv_longevity[kr_tv_longevity <= 5].index

kr_tv_successful = kr_tv_in_japan[kr_tv_in_japan["show_title"].isin(successful_titles)].drop_duplicates("show_title")
kr_tv_less_successful = kr_tv_in_japan[kr_tv_in_japan["show_title"].isin(less_successful_titles)].drop_duplicates("show_title")

print(f"\nKorean TV dramas in Japan Top 10:")
print(f"  • 5+ weeks success: {len(kr_tv_successful)} titles")
print(f"  • ≤5 weeks: {len(kr_tv_less_successful)} titles")

# Add weeks column for analysis
kr_tv_successful["weeks_in_top10"] = kr_tv_successful["show_title"].map(kr_tv_longevity)
kr_tv_less_successful["weeks_in_top10"] = kr_tv_less_successful["show_title"].map(kr_tv_longevity)

print("\n" + "-" * 80)
print("A. TOP PERFORMING KR DRAMAS (5+ WEEKS)")
print("-" * 80)

# List top performers
top_performers = kr_tv_successful.sort_values("weeks_in_top10", ascending=False)
print("\nTop 20 Korean dramas with 5+ weeks in Japan Top 10:")
for i, (_, row) in enumerate(top_performers.head(20).iterrows(), 1):
    genres = row["genres"][:40] + "..." if pd.notna(row["genres"]) and len(row["genres"]) > 40 else row.get("genres", "N/A")
    print(f"  {i:2}. {row['show_title'][:45]:45s} {row['weeks_in_top10']:2.0f}w | {genres}")

print("\n" + "-" * 80)
print("B. GENRE ANALYSIS: 5+ WEEKS vs ≤5 WEEKS")
print("-" * 80)

# Genre distribution comparison
def get_genre_distribution(df):
    genre_df = df.dropna(subset=["genres"]).copy()
    genre_df = genre_df.assign(genre=genre_df["genres"].str.split("|")).explode("genre")
    genre_df["genre"] = genre_df["genre"].str.strip()
    return genre_df["genre"].value_counts(normalize=True) * 100

success_genre_dist = get_genre_distribution(kr_tv_successful)
less_success_genre_dist = get_genre_distribution(kr_tv_less_successful)

genre_success_comp = pd.DataFrame({
    "5+ Weeks": success_genre_dist,
    "≤5 Weeks": less_success_genre_dist
}).fillna(0)
genre_success_comp["Difference"] = genre_success_comp["5+ Weeks"] - genre_success_comp["≤5 Weeks"]
genre_success_comp = genre_success_comp.sort_values("Difference", ascending=False)

print("\nGenre distribution (%) - Korean dramas in Japan:")
print(genre_success_comp.head(15).round(1))

print("\n📊 GENRES OVER-INDEXING IN 5+ WEEK SUCCESSES:")
for genre in genre_success_comp[genre_success_comp["Difference"] > 3].index:
    success_pct = genre_success_comp.loc[genre, "5+ Weeks"]
    less_pct = genre_success_comp.loc[genre, "≤5 Weeks"]
    diff = genre_success_comp.loc[genre, "Difference"]
    print(f"  • {genre}: {success_pct:.1f}% vs {less_pct:.1f}% (+{diff:.1f}pp)")

print("\n📊 GENRES UNDER-INDEXING IN 5+ WEEK SUCCESSES:")
for genre in genre_success_comp[genre_success_comp["Difference"] < -3].index:
    success_pct = genre_success_comp.loc[genre, "5+ Weeks"]
    less_pct = genre_success_comp.loc[genre, "≤5 Weeks"]
    diff = abs(genre_success_comp.loc[genre, "Difference"])
    print(f"  • {genre}: {success_pct:.1f}% vs {less_pct:.1f}% (-{diff:.1f}pp)")

print("\n" + "-" * 80)
print("C. QUALITY METRICS: 5+ WEEKS vs ≤5 WEEKS")
print("-" * 80)

# TMDB Rating comparison
success_rated = kr_tv_successful[(kr_tv_successful["tmdb_rating"].notna()) & (kr_tv_successful["tmdb_rating"] > 0)]
less_success_rated = kr_tv_less_successful[(kr_tv_less_successful["tmdb_rating"].notna()) & (kr_tv_less_successful["tmdb_rating"] > 0)]

print(f"\nTMDB Rating (excluding 0/null):")
print(f"  • 5+ weeks success: {success_rated['tmdb_rating'].mean():.2f} (n={len(success_rated)})")
print(f"  • ≤5 weeks: {less_success_rated['tmdb_rating'].mean():.2f} (n={len(less_success_rated)})")
print(f"  • Difference: {success_rated['tmdb_rating'].mean() - less_success_rated['tmdb_rating'].mean():+.2f}")

print("\n" + "-" * 80)
print("D. EPISODE COUNT ANALYSIS: 5+ WEEKS vs ≤5 WEEKS")
print("-" * 80)

if "num_episodes" in kr_tv_successful.columns:
    success_episodes = kr_tv_successful["num_episodes"].dropna()
    less_success_episodes = kr_tv_less_successful["num_episodes"].dropna()
    
    print(f"\nEpisode count:")
    print(f"  • 5+ weeks success: avg {success_episodes.mean():.1f}, median {success_episodes.median():.0f} (n={len(success_episodes)})")
    print(f"  • ≤5 weeks: avg {less_success_episodes.mean():.1f}, median {less_success_episodes.median():.0f} (n={len(less_success_episodes)})")
    
    # Episode count buckets
    def episode_bucket(ep):
        if ep <= 8:
            return "Short (≤8)"
        elif ep <= 16:
            return "Standard (9-16)"
        elif ep <= 24:
            return "Long (17-24)"
        else:
            return "Very Long (25+)"
    
    success_buckets = success_episodes.apply(episode_bucket).value_counts(normalize=True) * 100
    less_success_buckets = less_success_episodes.apply(episode_bucket).value_counts(normalize=True) * 100
    
    bucket_order = ["Short (≤8)", "Standard (9-16)", "Long (17-24)", "Very Long (25+)"]
    print(f"\nEpisode count distribution:")
    for bucket in bucket_order:
        s_pct = success_buckets.get(bucket, 0)
        l_pct = less_success_buckets.get(bucket, 0)
        print(f"  • {bucket:20s}: 5+w {s_pct:5.1f}% | ≤5w {l_pct:5.1f}%")

print("\n" + "-" * 80)
print("E. CASTING POPULARITY: 5+ WEEKS vs ≤5 WEEKS")
print("-" * 80)

if "avg_cast_popularity" in kr_tv_successful.columns:
    success_cast = kr_tv_successful["avg_cast_popularity"]
    success_cast = success_cast[(success_cast.notna()) & (success_cast > 0)]
    
    less_success_cast = kr_tv_less_successful["avg_cast_popularity"]
    less_success_cast = less_success_cast[(less_success_cast.notna()) & (less_success_cast > 0)]
    
    print(f"\nAverage casting popularity (excluding 0/null):")
    print(f"  • 5+ weeks success: {success_cast.mean():.2f} (n={len(success_cast)})")
    print(f"  • ≤5 weeks: {less_success_cast.mean():.2f} (n={len(less_success_cast)})")
    
    if less_success_cast.mean() > 0:
        diff_pct = ((success_cast.mean() / less_success_cast.mean()) - 1) * 100
        print(f"  • Difference: {diff_pct:+.1f}%")

print("\n" + "-" * 80)
print("F. SEQUEL STATUS: 5+ WEEKS vs ≤5 WEEKS")
print("-" * 80)

if "is_sequel" in kr_tv_successful.columns:
    success_sequel_rate = kr_tv_successful["is_sequel"].mean() * 100
    less_success_sequel_rate = kr_tv_less_successful["is_sequel"].mean() * 100
    
    print(f"\nSequel rate:")
    print(f"  • 5+ weeks success: {success_sequel_rate:.1f}%")
    print(f"  • ≤5 weeks: {less_success_sequel_rate:.1f}%")
    print(f"  • Difference: {success_sequel_rate - less_success_sequel_rate:+.1f}pp")

print("\n" + "-" * 80)
print("G. SEASON NUMBER: 5+ WEEKS vs ≤5 WEEKS")
print("-" * 80)

if "season_number" in kr_tv_successful.columns:
    success_seasons = kr_tv_successful["season_number"].dropna()
    less_success_seasons = kr_tv_less_successful["season_number"].dropna()
    
    print(f"\nSeason distribution:")
    print(f"  • 5+ weeks - Season 1: {(success_seasons == 1).sum()} ({(success_seasons == 1).sum()/len(success_seasons)*100:.1f}%)")
    print(f"  • 5+ weeks - Season 2+: {(success_seasons > 1).sum()} ({(success_seasons > 1).sum()/len(success_seasons)*100:.1f}%)")
    print(f"  • ≤5 weeks - Season 1: {(less_success_seasons == 1).sum()} ({(less_success_seasons == 1).sum()/len(less_success_seasons)*100:.1f}%)")
    print(f"  • ≤5 weeks - Season 2+: {(less_success_seasons > 1).sum()} ({(less_success_seasons > 1).sum()/len(less_success_seasons)*100:.1f}%)")

# VISUALIZATION
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Genre comparison (top 10 genres)
top_genres_success = genre_success_comp.head(10)
x = np.arange(len(top_genres_success))
width = 0.35

axes[0, 0].barh(x - width/2, top_genres_success["5+ Weeks"], width, label="5+ Weeks", color="#FF6B6B")
axes[0, 0].barh(x + width/2, top_genres_success["≤5 Weeks"], width, label="≤5 Weeks", color="#95A5A6")
axes[0, 0].set_yticks(x)
axes[0, 0].set_yticklabels(top_genres_success.index)
axes[0, 0].set_xlabel("% of Entries")
axes[0, 0].set_title("Genre Distribution: 5+ Weeks vs ≤5 Weeks", fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(axis='x', alpha=0.3)
axes[0, 0].invert_yaxis()

# 2. Rating distribution
if len(success_rated) > 0 and len(less_success_rated) > 0:
    axes[0, 1].hist(less_success_rated["tmdb_rating"], bins=15, alpha=0.7,
                    label=f"≤5 Weeks (avg: {less_success_rated['tmdb_rating'].mean():.2f})",
                    color="#95A5A6", density=True)
    axes[0, 1].hist(success_rated["tmdb_rating"], bins=15, alpha=0.7,
                    label=f"5+ Weeks (avg: {success_rated['tmdb_rating'].mean():.2f})",
                    color="#FF6B6B", density=True)
    axes[0, 1].set_xlabel("TMDB Rating")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].set_title("Rating Distribution: 5+ Weeks vs ≤5 Weeks", fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

# 3. Episode count distribution
if "num_episodes" in kr_tv_successful.columns and len(success_episodes) > 0:
    # Use IQR to filter outliers
    all_eps = pd.concat([success_episodes, less_success_episodes])
    ep_Q1, ep_Q3 = all_eps.quantile(0.25), all_eps.quantile(0.75)
    ep_IQR = ep_Q3 - ep_Q1
    ep_lower, ep_upper = ep_Q1 - 1.5 * ep_IQR, ep_Q3 + 1.5 * ep_IQR
    
    success_eps_filtered = success_episodes[(success_episodes >= ep_lower) & (success_episodes <= ep_upper)]
    less_eps_filtered = less_success_episodes[(less_success_episodes >= ep_lower) & (less_success_episodes <= ep_upper)]
    
    axes[0, 2].hist(less_eps_filtered, bins=15, alpha=0.7,
                    label=f"≤5 Weeks (avg: {less_eps_filtered.mean():.1f})",
                    color="#95A5A6", density=True)
    axes[0, 2].hist(success_eps_filtered, bins=15, alpha=0.7,
                    label=f"5+ Weeks (avg: {success_eps_filtered.mean():.1f})",
                    color="#FF6B6B", density=True)
    axes[0, 2].set_xlabel("Number of Episodes")
    axes[0, 2].set_ylabel("Density")
    axes[0, 2].set_title("Episode Count: 5+ Weeks vs ≤5 Weeks (IQR filtered)", fontsize=12, fontweight='bold')
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.3)

# 4. Casting popularity
if "avg_cast_popularity" in kr_tv_successful.columns and len(success_cast) > 0:
    axes[1, 0].hist(less_success_cast, bins=20, alpha=0.7,
                    label=f"≤5 Weeks (avg: {less_success_cast.mean():.1f})",
                    color="#95A5A6", density=True)
    axes[1, 0].hist(success_cast, bins=20, alpha=0.7,
                    label=f"5+ Weeks (avg: {success_cast.mean():.1f})",
                    color="#FF6B6B", density=True)
    axes[1, 0].set_xlabel("Average Casting Popularity")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].set_title("Casting Popularity: 5+ Weeks vs ≤5 Weeks", fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

# 5. Sequel status comparison
if "is_sequel" in kr_tv_successful.columns:
    sequel_comp = pd.DataFrame({
        "5+ Weeks": [100-success_sequel_rate, success_sequel_rate],
        "≤5 Weeks": [100-less_success_sequel_rate, less_success_sequel_rate]
    }, index=["Original", "Sequel"])
    
    sequel_comp.plot.bar(ax=axes[1, 1], color=["#FF6B6B", "#95A5A6"])
    axes[1, 1].set_xlabel("Content Type")
    axes[1, 1].set_ylabel("% of Content")
    axes[1, 1].set_title("Sequel vs Original: 5+ Weeks vs ≤5 Weeks", fontsize=12, fontweight='bold')
    axes[1, 1].legend(title="Success Tier")
    axes[1, 1].grid(axis='y', alpha=0.3)
    plt.setp(axes[1, 1].get_xticklabels(), rotation=0)

# 6. Top 15 successful titles
top_15_success = top_performers.head(15)
axes[1, 2].barh(range(len(top_15_success)), top_15_success["weeks_in_top10"], color="#FF6B6B")
axes[1, 2].set_yticks(range(len(top_15_success)))
axes[1, 2].set_yticklabels([title[:30] + "..." if len(title) > 30 else title
                            for title in top_15_success["show_title"]])
axes[1, 2].set_xlabel("Weeks in Japan Top 10")
axes[1, 2].set_title("Top 15 KR Dramas (5+ Weeks Success)", fontsize=12, fontweight='bold')
axes[1, 2].axvline(x=5, color='gray', linestyle='--', alpha=0.5, label="5 week threshold")
axes[1, 2].grid(axis='x', alpha=0.3)
axes[1, 2].invert_yaxis()

fig.tight_layout()
plt.show()
save(fig, "09_kr_dramas_5plus_weeks_analysis.png")

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY: CHARACTERISTICS OF 5+ WEEK SUCCESS KR DRAMAS IN JAPAN")
print("=" * 80)
print(f"""
Key differentiators for Korean dramas achieving 5+ weeks in Japan's Top 10:

1. QUALITY: Higher average TMDB ratings ({success_rated['tmdb_rating'].mean():.2f} vs {less_success_rated['tmdb_rating'].mean():.2f})
""")

if "avg_cast_popularity" in kr_tv_successful.columns and len(success_cast) > 0 and len(less_success_cast) > 0:
    print(f"2. STAR POWER: Higher casting popularity ({success_cast.mean():.1f} vs {less_success_cast.mean():.1f})")

if "num_episodes" in kr_tv_successful.columns:
    print(f"3. EPISODE COUNT: Average {success_episodes.mean():.1f} episodes (vs {less_success_episodes.mean():.1f} for ≤5 weeks)")

print(f"""
4. GENRE PATTERNS: See visualization for genre over/under-indexing

These insights can guide content acquisition and commissioning for Japan market success.
""")

# %%
# ═══════════════════════════════════════════════════════════════════════════
# 9. STRATEGIC BUSINESS RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("9. STRATEGIC BUSINESS RECOMMENDATIONS")
print("=" * 80)

print("""
Based on Korea vs Japan comparison, we recommend:

1. CONTENT ACQUISITION STRATEGY:
   - Prioritize genres that over-index in Korea vs Japan
   - Korean content has proven exportability to Japan market
   - Identify "Asia-wide appeal" content that succeeds in both markets

2. LOCALIZATION OPPORTUNITIES:
   - Genre preferences differ significantly between markets
   - Consider market-specific content commissioning
   - Test Japan-first content in Korea (low current penetration)

3. PORTFOLIO OPTIMIZATION:
   - Balance local vs international content based on each market's preferences
   - Korean dramas have strong pan-Asian appeal
   - Anime/Japanese content has opportunities in Korea

4. FURTHER ANALYSIS NEEDED:
   - Extend comparison to other Asian markets (Taiwan, Thailand, India)
   - Analyze completion rates (not just top 10 presence)
   - Study content that "breaks out" from regional to global success
""")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nOutputs saved to: {OUTPUT_DIR}/")
print("  - 01_language_comparison.png")
print("  - 02_genre_comparison.png")
print("  - 03_longevity_comparison.png")
print("  - 04_shared_content.png")
print("  - 05_category_comparison.png")
print("  - 06_korean_content_characteristics.png")
print("  - 07_rating_by_genre_boxplot.png")
print("  - 07b_korean_weeks_distribution.png")
print("  - 08_content_features_analysis.png")


# %%

korean_jp_unique.columns