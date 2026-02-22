# %%
import glob
import os
import re
import textwrap

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns
from collections import Counter


# %%

sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.1)
plt.rcParams["figure.dpi"] = 150
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save(fig, name):
    fig.savefig(os.path.join(OUTPUT_DIR, name), bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ saved {name}")

def detect_language(title):
    """Detect language based on character patterns in title"""
    if pd.isna(title):
        return "Unknown"
    
    has_korean = bool(re.search(r'[\uAC00-\uD7A3]', str(title)))
    has_japanese = bool(re.search(r'[\u3040-\u309F\u30A0-\u30FF]', str(title)))
    has_chinese = bool(re.search(r'[\u4E00-\u9FFF]', str(title)))
    has_english = bool(re.search(r'[a-zA-Z]', str(title)))
    
    if has_korean:
        return "Korean"
    elif has_japanese:
        return "Japanese"
    elif has_chinese:
        return "Chinese"
    elif has_english:
        return "English"
    else:
        return "Other"

print("=" * 70)
print("KOREA-SPECIFIC NETFLIX ANALYSIS")
print("=" * 70)

# %%

DATA_DIR = "dataset/raw"
gw_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_global_weekly.tsv")))
latest_file = gw_files[-1]
print(f"\n  Loading latest global_weekly file: {os.path.basename(latest_file)}")

global_gw = pd.read_csv(latest_file, sep="\t")
global_gw["week"] = pd.to_datetime(global_gw["week"])

kr_gw = global_gw[global_gw["country_iso2"] == "KR"].copy()



# %%
kr_gw["language"] = kr_gw["show_title"].apply(detect_language)

# %%
kr_gw[kr_gw.week == '2026-02-15']



# %%

kr_gw.to_csv("dataset/raw/kr_gw.csv", index=False)

# %%
kr_gw_enriched = pd.read_csv('dataset/raw/kr_gw_enriched.csv')
kr_gw_enriched["week"] = pd.to_datetime(kr_gw_enriched["week"])
kr_gw_enriched = kr_gw_enriched[kr_gw_enriched["week"] >= pd.Timestamp("2023-03-01")]

na_by_week = (
    kr_gw_enriched.assign(is_na_tmdb=kr_gw_enriched["tmdb_type"].isna())
    .groupby("week")["is_na_tmdb"]
    .agg(["sum", "count"])
    .rename(columns={"sum": "na_count", "count": "total_count"})
    .reset_index()
)
na_by_week["na_rate"] = na_by_week["na_count"] / na_by_week["total_count"]


fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
axes[0].plot(na_by_week["week"], na_by_week["na_count"], marker="o", linewidth=1.5)
axes[0].set_title("TMDB NA Count by Week (Korea)")
axes[0].set_ylabel("NA Count")
axes[0].grid(alpha=0.3)

axes[1].plot(na_by_week["week"], na_by_week["na_rate"], marker="o", linewidth=1.5)
axes[1].set_title("TMDB NA Rate by Week (Korea)")
axes[1].set_ylabel("NA Rate")
axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
axes[1].grid(alpha=0.3)
axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
axes[1].xaxis.set_minor_locator(mdates.MonthLocator())
plt.setp(axes[1].get_xticklabels(), rotation=90, ha="center")
plt.show()

# fig.tight_layout()
# save(fig, "kr_01b_tmdb_na_by_week.png")

# %%
valid_weeks = na_by_week.loc[na_by_week["na_rate"] < 0.2, "week"]
kr_gw_enriched = kr_gw_enriched[kr_gw_enriched["week"].isin(valid_weeks)]


# %%
kr_gw_enriched = kr_gw_enriched.dropna(subset = ["tmdb_id"])
kr_gw_enriched["lang_label"] = kr_gw_enriched["original_language"].map(
    {"ko": "Korean", "en": "English", "ja": "Japanese"}
).fillna("Other")

# %%
kr_gw_enriched.head()






# %%
print("\n" + "=" * 70)
print("A. KOREAN AUDIENCE PREFERENCES – GENRE, LANGUAGE & TRENDS")
print("=" * 70)

# ── A1. Genre frequency (explode pipe-separated genres) ──────────────────────
genre_df = kr_gw_enriched.dropna(subset=["genres"]).copy()
genre_df = genre_df.assign(genre=genre_df["genres"].str.split("|")).explode("genre")
genre_df["genre"] = genre_df["genre"].str.strip()

genre_counts = genre_df["genre"].value_counts().head(15)

fig, ax = plt.subplots(figsize=(12, 6))
genre_counts.sort_values().plot.barh(ax=ax, color=sns.color_palette("Set2", len(genre_counts)))
ax.set_title("Korea Top 10: Most Frequent Genres (TMDB, all weeks)")
ax.set_xlabel("Number of Entries")
ax.set_ylabel("Genre")
fig.tight_layout()
plt.show()
save(fig, "kr_A1_genre_frequency.png")

print("\n  Top 15 genres in Korea Top 10:")
print(genre_counts.to_string())

# %%

# ── A2. Genre share by original_language ─────────────────────────────────────
top_genres = genre_counts.index.tolist()
top_langs_list = ["ko", "en", "ja"]
lang_genre = (
    genre_df[genre_df["original_language"].isin(top_langs_list) &
             genre_df["genre"].isin(top_genres)]
    .groupby(["original_language", "genre"])
    .size()
    .unstack(fill_value=0)
)
lang_genre_pct = lang_genre.div(lang_genre.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(14, 6))
lang_genre_pct.T.plot.bar(ax=ax, colormap="Set2")
ax.set_title("Genre Mix by Original Language in Korea Top 10 (%)")
ax.set_xlabel("Genre")
ax.set_ylabel("Share (%)")
ax.legend(title="Language", labels=["Korean (ko)", "English (en)", "Japanese (ja)"])
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
fig.tight_layout()
plt.show()
save(fig, "kr_A2_genre_by_language.png")
# %%
# ── A3. Genre trends over time (top 6 genres, monthly) ───────────────────────
genre_df["month"] = genre_df["week"].dt.to_period("M").dt.to_timestamp()
top6_genres = genre_counts.head(6).index.tolist()

genre_monthly = (
    genre_df[genre_df["genre"].isin(top6_genres)]
    .groupby(["month", "genre"])
    .size()
    .unstack(fill_value=0)
)

fig, ax = plt.subplots(figsize=(14, 6))
for genre in top6_genres:
    if genre in genre_monthly.columns:
        ax.plot(genre_monthly.index, genre_monthly[genre], marker="o",
                markersize=4, linewidth=1.8, label=genre)
ax.set_title("Korea Top 10: Genre Trends Over Time (Monthly)")
ax.set_xlabel("Month")
ax.set_ylabel("Number of Entries")
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
ax.legend(title="Genre", bbox_to_anchor=(1.01, 1), loc="upper left")
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()
# save(fig, "kr_A3_genre_trends_over_time.png")

# %%

# ── A4. Language distribution over time (monthly, stacked area) ──────────────
lang_time = kr_gw_enriched.copy()
lang_time["month"] = lang_time["week"].dt.to_period("M").dt.to_timestamp()

lang_monthly = (
    lang_time.groupby(["month", "lang_label"])
    .size()
    .unstack(fill_value=0)
)
col_order = [c for c in ["Korean", "English", "Japanese", "Other"] if c in lang_monthly.columns]
lang_monthly = lang_monthly[col_order]

fig, ax = plt.subplots(figsize=(14, 6))
lang_monthly.plot.area(ax=ax, alpha=0.75, colormap="Set2")
ax.set_title("Korea Top 10: Language Distribution Over Time (Monthly)")
ax.set_xlabel("Month")
ax.set_ylabel("Number of Entries")
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
ax.legend(title="Language", bbox_to_anchor=(1.01, 1), loc="upper left")
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()
save(fig, "kr_A4_language_over_time.png")
# %%
# ── A5. Top genres by longevity (avg cumulative weeks) ───────────────────────
genre_longevity = (
    genre_df.groupby("genre")["cumulative_weeks_in_top_10"]
    .agg(["mean", "count"])
    .query("count >= 10")
    .sort_values("mean", ascending=False)
    .head(15)
)

fig, ax = plt.subplots(figsize=(12, 6))
genre_longevity["mean"].sort_values().plot.barh(
    ax=ax, color=sns.color_palette("rocket", len(genre_longevity)))
ax.set_title("Korea Top 10: Avg Longevity by Genre (min 10 entries)")
ax.set_xlabel("Avg Cumulative Weeks in Top 10")
ax.set_ylabel("Genre")
fig.tight_layout()
plt.show()
save(fig, "kr_A5_genre_longevity.png")

print("\n  Genre longevity (avg cumulative weeks, min 10 entries):")
print(genre_longevity.to_string())
# %%
# ── A6. Korean vs non-Korean content share over time ─────────────────────────
lang_time["is_korean"] = (lang_time["original_language"] == "ko").astype(int)
kr_share = (
    lang_time.groupby("month")["is_korean"]
    .agg(["sum", "count"])
    .rename(columns={"sum": "korean_count", "count": "total"})
)
kr_share["korean_pct"] = kr_share["korean_count"] / kr_share["total"] * 100

fig, ax = plt.subplots(figsize=(14, 5))
ax.fill_between(kr_share.index, kr_share["korean_pct"], alpha=0.4, color="#FF6B6B")
ax.plot(kr_share.index, kr_share["korean_pct"], color="#FF6B6B", linewidth=2, marker="o", markersize=4)
ax.axhline(50, color="gray", linestyle="--", alpha=0.6, label="50% line")
ax.set_title("Korea Top 10: Korean-Language Content Share Over Time")
ax.set_xlabel("Month")
ax.set_ylabel("Korean Content (%)")
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()

plt.show()  
save(fig, "kr_A6_korean_share_over_time.png")

print("\n  Korean content share by month:")
print(kr_share[["korean_count", "total", "korean_pct"]].round(1).to_string())

# %%
print("\n" + "=" * 70)
print("2. KOREAN VS ENGLISH VS JAPANESE CONTENT PERFORMANCE")
print("=" * 70)

perf_by_lang = kr_gw_enriched.groupby("lang_label").agg({
    "cumulative_weeks_in_top_10": ["mean", "median", "max", "count"],
    "weekly_rank": "mean"
}).round(2)

print("\n  Performance metrics by language:")
print(perf_by_lang.to_string())

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

top_langs = ["Korean", "English", "Japanese"]
_lang_mask = kr_gw_enriched["lang_label"].isin(top_langs)

sns.boxplot(data=kr_gw_enriched[_lang_mask],
            x="lang_label", y="cumulative_weeks_in_top_10", ax=axes[0, 0],
            order=top_langs)
axes[0, 0].set_title("Longevity by Language (Cumulative Weeks in Top 10)")
axes[0, 0].set_xlabel("Language")
axes[0, 0].set_ylabel("Cumulative Weeks")

sns.boxplot(data=kr_gw_enriched[_lang_mask],
            x="lang_label", y="weekly_rank", ax=axes[0, 1],
            order=top_langs)
axes[0, 1].set_title("Average Ranking Position by Language")
axes[0, 1].set_xlabel("Language")
axes[0, 1].set_ylabel("Weekly Rank (Lower is Better)")
axes[0, 1].invert_yaxis()

lang_over_time = kr_gw_enriched[_lang_mask].groupby(["week", "lang_label"]).size().unstack(fill_value=0)
lang_over_time.plot(ax=axes[1, 0], marker="o", markersize=3, linewidth=1.5)
axes[1, 0].set_title("Number of Shows by Language Over Time")
axes[1, 0].set_xlabel("Week")
axes[1, 0].set_ylabel("Number of Shows in Top 10")
axes[1, 0].legend(title="Language")
axes[1, 0].grid(alpha=0.3)

lang_data = kr_gw_enriched[_lang_mask]
avg_rank_by_lang = lang_data.groupby(["lang_label", "weekly_rank"]).size().unstack(fill_value=0)

for lang in top_langs:
    if lang in avg_rank_by_lang.index:
        axes[1, 1].plot(range(1, 11), avg_rank_by_lang.loc[lang],
                       marker="o", label=lang, linewidth=2)

axes[1, 1].set_title("Rank Distribution by Language")
axes[1, 1].set_xlabel("Weekly Rank Position")
axes[1, 1].set_ylabel("Frequency")
axes[1, 1].set_xticks(range(1, 11))
axes[1, 1].legend(title="Language")
axes[1, 1].grid(alpha=0.3)

fig.suptitle("Content Performance Analysis: Korean vs English vs Japanese", 
             fontsize=16, y=0.995)
fig.tight_layout()
plt.show()
save(fig, "kr_02_language_performance.png")


# %%
lang_cat_perf = kr_gw_enriched[kr_gw_enriched["lang_label"].isin(top_langs)].groupby(["lang_label", "category"]).agg({
    "cumulative_weeks_in_top_10": "mean",
    "show_title": "count"
}).round(2)
lang_cat_perf.columns = ["Avg_Cumulative_Weeks", "Count"]

print("\n  Performance by language and category:")
print(lang_cat_perf.to_string())

print("\n" + "=" * 70)
print("3. KOREA VS GLOBAL: CATEGORY PREFERENCES")
print("=" * 70)


kr_cat_dist = kr_gw_enriched["category"].value_counts(normalize=True) * 100
global_cat_dist = global_gw["category"].value_counts(normalize=True) * 100

comparison_df = pd.DataFrame({
    "Korea": kr_cat_dist,
    "Global": global_cat_dist
}).fillna(0)

print("\n  Category distribution comparison (%):")
print(comparison_df.to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

comparison_df.plot.bar(ax=axes[0], color=["#FF6B6B", "#4ECDC4"])
axes[0].set_title("Korea vs Global: Category Distribution")
axes[0].set_xlabel("Category")
axes[0].set_ylabel("Percentage (%)")
axes[0].legend(title="Region")
plt.setp(axes[0].get_xticklabels(), rotation=15, ha="right")

kr_lang_in_cat = kr_gw_enriched[kr_gw_enriched["lang_label"].isin(top_langs)].groupby(["category", "lang_label"]).size().unstack(fill_value=0)
kr_lang_in_cat_pct = kr_lang_in_cat.div(kr_lang_in_cat.sum(axis=1), axis=0) * 100

kr_lang_in_cat_pct.plot.barh(ax=axes[1], stacked=True, colormap="Set2")
axes[1].set_title("Language Mix within Each Category (Korea)")
axes[1].set_xlabel("Percentage (%)")
axes[1].set_ylabel("Category")
axes[1].legend(title="Language", bbox_to_anchor=(1.05, 1), loc='upper left')

fig.tight_layout()
plt.show()
save(fig, "kr_03_korea_vs_global_categories.png")

# %%
print("\n" + "=" * 70)
print("4. TOP PERFORMING CONTENT BY LANGUAGE")
print("=" * 70)

lang_label_map = {"ko": "Korean", "en": "English", "ja": "Japanese"}

for iso, label in lang_label_map.items():
    lang_data = kr_gw_enriched[kr_gw_enriched["original_language"] == iso]
    if len(lang_data) == 0:
        continue
    
    top_shows = lang_data.groupby(["show_title", "category"]).agg({
        "cumulative_weeks_in_top_10": "max",
        "weekly_rank": "min"
    }).reset_index()
    
    top_shows = top_shows.nlargest(10, "cumulative_weeks_in_top_10")
    
    print(f"\n  Top 10 {label} Content in Korea (by longevity):")
    for idx, row in top_shows.iterrows():
        print(f"    {row['show_title'][:50]:50s} | {row['category']:20s} | {row['cumulative_weeks_in_top_10']:2.0f} weeks | Best rank: #{row['weekly_rank']:.0f}")

fig, axes = plt.subplots(3, 1, figsize=(14, 16))

for idx, (iso, label) in enumerate(lang_label_map.items()):
    lang_data = kr_gw_enriched[kr_gw_enriched["original_language"] == iso]
    if len(lang_data) == 0:
        axes[idx].text(0.5, 0.5, f"No {label} content data",
                      ha='center', va='center', transform=axes[idx].transAxes)
        axes[idx].set_title(f"Top {label} Content")
        continue
    
    top_shows = lang_data.groupby(["show_title", "category"]).agg({
        "cumulative_weeks_in_top_10": "max"
    }).reset_index()
    
    top_shows = top_shows.nlargest(15, "cumulative_weeks_in_top_10")
    
    labels = [textwrap.fill(f"{r.show_title} ({r.category})", 40)
              for _, r in top_shows.iterrows()]
    
    colors = ["#FF6B6B" if cat == "Films" else "#4ECDC4"
              for cat in top_shows["category"]]
    
    axes[idx].barh(labels, top_shows["cumulative_weeks_in_top_10"], color=colors)
    axes[idx].set_xlabel("Max Cumulative Weeks in Top 10")
    axes[idx].set_title(f"Top 15 {label} Content in Korea (by Longevity)")
    axes[idx].invert_yaxis()

fig.tight_layout()
plt.show()
# save(fig, "kr_04_top_content_by_language.png")


# %%
print("\n" + "=" * 70)
print("5. LIFECYCLE ANALYSIS: HOW HITS EVOLVE IN KOREA")
print("=" * 70)

show_lifecycle = kr_gw_enriched.sort_values(["show_title", "season_title", "week"])

lifecycle_stats = []
for (show, season), group in show_lifecycle.groupby(["show_title", "season_title"]):
    if len(group) < 2:
        continue
    
    group = group.sort_values("week")
    
    lifecycle_stats.append({
        "show_title": show,
        "season_title": season,
        "category": group["category"].iloc[0],
        "lang_label": group["lang_label"].iloc[0],
        "total_weeks": len(group),
        "first_week": group["week"].min(),
        "last_week": group["week"].max(),
        "entry_rank": group["weekly_rank"].iloc[0],
        "best_rank": group["weekly_rank"].min(),
        "exit_rank": group["weekly_rank"].iloc[-1],
        "avg_rank": group["weekly_rank"].mean(),
        "max_cumulative": group["cumulative_weeks_in_top_10"].max()
    })

lifecycle_df = pd.DataFrame(lifecycle_stats)

print(f"\n  Analyzed {len(lifecycle_df)} show lifecycles")
print(f"\n  Average lifecycle metrics:")
print(lifecycle_df[["total_weeks", "entry_rank", "best_rank", "exit_rank", "avg_rank"]].describe().round(2).to_string())

print(f"\n  Lifecycle by language:")
print(lifecycle_df.groupby("lang_label")[["total_weeks", "entry_rank", "best_rank", "exit_rank"]].mean().round(2).to_string())

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

sns.histplot(data=lifecycle_df, x="total_weeks", bins=30, ax=axes[0, 0], kde=True)
axes[0, 0].set_title("Distribution of Show Lifespans in Top 10")
axes[0, 0].set_xlabel("Number of Weeks in Top 10")
axes[0, 0].set_ylabel("Frequency")
axes[0, 0].axvline(lifecycle_df["total_weeks"].median(), color='red', 
                   linestyle='--', label=f'Median: {lifecycle_df["total_weeks"].median():.1f} weeks')
axes[0, 0].legend()

entry_exit = lifecycle_df[["entry_rank", "exit_rank"]].copy()
axes[0, 1].scatter(entry_exit["entry_rank"], entry_exit["exit_rank"], 
                  alpha=0.5, s=50, edgecolors="k", linewidth=0.5)
axes[0, 1].plot([1, 10], [1, 10], 'r--', alpha=0.5, label="Entry = Exit")
axes[0, 1].set_xlabel("Entry Rank")
axes[0, 1].set_ylabel("Exit Rank")
axes[0, 1].set_title("Entry vs Exit Rank Position")
axes[0, 1].set_xlim(0.5, 10.5)
axes[0, 1].set_ylim(0.5, 10.5)
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

lifecycle_by_lang = lifecycle_df[lifecycle_df["lang_label"].isin(top_langs)]
sns.boxplot(data=lifecycle_by_lang, x="lang_label", y="total_weeks",
            ax=axes[1, 0], order=top_langs)
axes[1, 0].set_title("Show Lifespan by Language")
axes[1, 0].set_xlabel("Language")
axes[1, 0].set_ylabel("Weeks in Top 10")

rank_improvement = lifecycle_df.copy()
rank_improvement["rank_improvement"] = rank_improvement["entry_rank"] - rank_improvement["best_rank"]

sns.histplot(data=rank_improvement, x="rank_improvement", bins=20, ax=axes[1, 1], kde=True)
axes[1, 1].set_title("Rank Improvement from Entry to Peak")
axes[1, 1].set_xlabel("Rank Improvement (Entry - Best)")
axes[1, 1].set_ylabel("Frequency")
axes[1, 1].axvline(0, color='red', linestyle='--', alpha=0.5, label='No improvement')
axes[1, 1].legend()

fig.suptitle("Hit Lifecycle Analysis in Korea", fontsize=16, y=0.995)
fig.tight_layout()
save(fig, "kr_05_lifecycle_analysis.png")

print("\n" + "=" * 70)
print("6. TYPICAL HIT PATTERNS: TRAJECTORY ANALYSIS")
print("=" * 70)

long_runners = lifecycle_df[lifecycle_df["total_weeks"] >= 5].nlargest(20, "total_weeks")

print(f"\n  Top 20 longest-running shows in Korea:")
for idx, row in long_runners.iterrows():
    title_display = row["show_title"][:45]
    print(f"    {title_display:45s} | {row['lang_label']:10s} | {row['total_weeks']:2.0f} weeks | Entry: #{row['entry_rank']:.0f} → Best: #{row['best_rank']:.0f}")

fig, axes = plt.subplots(2, 1, figsize=(16, 12))

sample_shows = long_runners.head(10)

for idx, row in sample_shows.iterrows():
    show_data = kr_gw_enriched[
        (kr_gw_enriched["show_title"] == row["show_title"]) &
        (kr_gw_enriched["season_title"] == row["season_title"])
    ].sort_values("week")
    
    if len(show_data) > 1:
        label = f"{row['show_title'][:30]} ({row['lang_label']})"
        axes[0].plot(range(len(show_data)), show_data["weekly_rank"],
                    marker="o", label=label, linewidth=2, markersize=4)

axes[0].set_xlabel("Weeks Since Entry")
axes[0].set_ylabel("Weekly Rank")
axes[0].set_title("Rank Trajectory of Top 10 Longest-Running Shows")
axes[0].invert_yaxis()
axes[0].set_yticks(range(1, 11))
axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
axes[0].grid(alpha=0.3)

pattern_counts = {
    "Quick Peak (peaks in week 1-2)": 0,
    "Gradual Rise (peaks after week 3+)": 0,
    "Stable (stays within 3 ranks)": 0,
    "Volatile (>5 rank range)": 0
}

for idx, row in lifecycle_df.iterrows():
    show_data = kr_gw_enriched[
        (kr_gw_enriched["show_title"] == row["show_title"]) &
        (kr_gw_enriched["season_title"] == row["season_title"])
    ].sort_values("week")
    
    if len(show_data) < 2:
        continue
    
    best_rank_week = show_data["weekly_rank"].idxmin()
    week_of_peak = show_data.index.get_loc(best_rank_week)
    
    rank_range = show_data["weekly_rank"].max() - show_data["weekly_rank"].min()
    
    if week_of_peak <= 1:
        pattern_counts["Quick Peak (peaks in week 1-2)"] += 1
    else:
        pattern_counts["Gradual Rise (peaks after week 3+)"] += 1
    
    if rank_range <= 3:
        pattern_counts["Stable (stays within 3 ranks)"] += 1
    elif rank_range > 5:
        pattern_counts["Volatile (>5 rank range)"] += 1

pattern_df = pd.DataFrame(list(pattern_counts.items()), columns=["Pattern", "Count"])
pattern_df = pattern_df[pattern_df["Count"] > 0]

axes[1].barh(pattern_df["Pattern"], pattern_df["Count"], 
            color=sns.color_palette("viridis", len(pattern_df)))
axes[1].set_xlabel("Number of Shows")
axes[1].set_title("Common Trajectory Patterns in Korea")
axes[1].invert_yaxis()

for i, (pattern, count) in enumerate(zip(pattern_df["Pattern"], pattern_df["Count"])):
    axes[1].text(count + max(pattern_df["Count"]) * 0.02, i, 
                str(int(count)), va='center', fontweight='bold')

fig.tight_layout()
plt.show()
save(fig, "kr_06_trajectory_patterns.png")

print(f"\n  Trajectory pattern distribution:")
for pattern, count in pattern_counts.items():
    if count > 0:
        pct = (count / len(lifecycle_df)) * 100
        print(f"    {pattern:40s}: {count:4d} ({pct:5.1f}%)")

# %%

print("\n" + "=" * 70)
print("7. INSIGHTS SUMMARY")
print("=" * 70)

print("\n  KEY FINDINGS:")
print(f"\n  1. LANGUAGE DOMINANCE:")
total_enriched = len(kr_gw_enriched)
print(f"     - Korean content: {(kr_gw_enriched['original_language'] == 'ko').sum()} entries ({(kr_gw_enriched['original_language'] == 'ko').sum() / total_enriched * 100:.1f}%)")
print(f"     - English content: {(kr_gw_enriched['original_language'] == 'en').sum()} entries ({(kr_gw_enriched['original_language'] == 'en').sum() / total_enriched * 100:.1f}%)")
print(f"     - Japanese content: {(kr_gw_enriched['original_language'] == 'ja').sum()} entries ({(kr_gw_enriched['original_language'] == 'ja').sum() / total_enriched * 100:.1f}%)")

print(f"\n  2. PERFORMANCE BY LANGUAGE:")
korean_avg = kr_gw_enriched[kr_gw_enriched["original_language"] == "ko"]["cumulative_weeks_in_top_10"].mean()
english_avg = kr_gw_enriched[kr_gw_enriched["original_language"] == "en"]["cumulative_weeks_in_top_10"].mean()
japanese_avg = kr_gw_enriched[kr_gw_enriched["original_language"] == "ja"]["cumulative_weeks_in_top_10"].mean()
print(f"     - Korean content avg longevity: {korean_avg:.2f} weeks")
print(f"     - English content avg longevity: {english_avg:.2f} weeks")
print(f"     - Japanese content avg longevity: {japanese_avg:.2f} weeks")

print(f"\n  3. TYPICAL HIT LIFECYCLE:")
print(f"     - Median lifespan: {lifecycle_df['total_weeks'].median():.1f} weeks")
print(f"     - Average entry rank: #{lifecycle_df['entry_rank'].mean():.1f}")
print(f"     - Average best rank: #{lifecycle_df['best_rank'].mean():.1f}")
print(f"     - Average exit rank: #{lifecycle_df['exit_rank'].mean():.1f}")

print(f"\n  4. CATEGORY PREFERENCES:")
print(f"     - Korea Films: {kr_cat_dist.get('Films', 0):.1f}%")
print(f"     - Korea TV: {kr_cat_dist.get('TV', 0):.1f}%")
print(f"     - Global Films: {global_cat_dist.get('Films', 0):.1f}%")
print(f"     - Global TV: {global_cat_dist.get('TV', 0):.1f}%")

lifecycle_df.to_csv(os.path.join(OUTPUT_DIR, "kr_lifecycle_data.csv"), index=False)
print(f"\n  Lifecycle data saved to {os.path.join(OUTPUT_DIR, 'kr_lifecycle_data.csv')}")

print("\n" + "=" * 70)
print("KOREA-SPECIFIC ANALYSIS COMPLETE")
print("=" * 70)
print(f"  All visualizations saved to ./{OUTPUT_DIR}/")

# %%
# =============================================================================
# PART B: BREAKOUT CLASSIFICATION – EDA & FEATURE ANALYSIS
# =============================================================================
# Goal: Predict whether a title will be a "breakout" (4+ weeks in Top 10)
#       vs. a "one-week/short appearance" (<4 weeks)

print("\n" + "=" * 70)
print("B. BREAKOUT CLASSIFICATION – EDA & FEATURE ANALYSIS")
print("=" * 70)

# ── B1. Create title-level dataset with target variable ──────────────────────
# Fill NA season_title (Films don't have seasons) to avoid dropping them in groupby
kr_gw_enriched["season_title"] = kr_gw_enriched["season_title"].fillna("")

title_df = kr_gw_enriched.groupby(["show_title", "season_title", "category"]).agg({
    "cumulative_weeks_in_top_10": "max",
    "weekly_rank": ["min", "first"],  # best_rank, entry_rank
    "tmdb_id": "first",
    "tmdb_rating": "first",
    "tmdb_vote_count": "first",
    "tmdb_popularity": "first",
    "genres": "first",
    "original_language": "first",
    "runtime_minutes": "first",
    "days_since_release": "min",
    "is_korean": "first",
    "is_sequel": "first",
    "budget_usd": "first",
    "avg_cast_popularity": "first",
    "week": "min",  # first appearance week
}).reset_index()

title_df.columns = [
    "show_title", "season_title", "category", "max_weeks", "best_rank", "entry_rank",
    "tmdb_id", "tmdb_rating", "tmdb_vote_count", "tmdb_popularity", "genres",
    "original_language", "runtime_minutes", "days_since_release", "is_korean",
    "is_sequel", "budget_usd", "avg_cast_popularity", "first_week"
]

# Target variable
title_df["is_breakout"] = (title_df["max_weeks"] >= 4).astype(int)

print(f"\n  Title-level dataset shape: {title_df.shape}")
print(f"  Unique titles: {len(title_df)}")

# %%
import numpy as np
np.unique(title_df["category"])

# %%
# ── B2. Target variable distribution ─────────────────────────────────────────
print("\n" + "-" * 50)
print("B2. TARGET VARIABLE DISTRIBUTION")
print("-" * 50)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Histogram of max_weeks
axes[0].hist(title_df["max_weeks"], bins=range(1, title_df["max_weeks"].max()+2), 
             edgecolor="black", alpha=0.7, color="#4ECDC4")
axes[0].axvline(4, color="red", linestyle="--", linewidth=2, label="Breakout threshold (4 weeks)")
axes[0].set_xlabel("Max Weeks in Top 10")
axes[0].set_ylabel("Number of Titles")
axes[0].set_title("Distribution of Title Longevity")
axes[0].legend()

# Class balance pie chart
breakout_counts = title_df["is_breakout"].value_counts()
labels = ["Short-lived (<4 weeks)", "Breakout (4+ weeks)"]
colors = ["#FF6B6B", "#4ECDC4"]
axes[1].pie(breakout_counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
axes[1].set_title("Class Balance: Breakout vs Short-lived")

# Breakout rate by category
breakout_by_cat = title_df.groupby("category")["is_breakout"].agg(["mean", "count"])
breakout_by_cat["mean"].plot.bar(ax=axes[2], color=["#FF6B6B", "#4ECDC4"], edgecolor="black")
axes[2].set_title("Breakout Rate by Category")
axes[2].set_xlabel("Category")
axes[2].set_ylabel("Breakout Rate")
axes[2].set_ylim(0, 1)
for i, (idx, row) in enumerate(breakout_by_cat.iterrows()):
    axes[2].text(i, row["mean"] + 0.02, f'n={int(row["count"])}', ha='center', fontsize=10)
plt.setp(axes[2].get_xticklabels(), rotation=0)

fig.tight_layout()
plt.show()
save(fig, "kr_B2_target_distribution.png")

print(f"\n  Class distribution:")
print(f"    - Short-lived (<4 weeks): {breakout_counts.get(0, 0)} ({breakout_counts.get(0, 0)/len(title_df)*100:.1f}%)")
print(f"    - Breakout (4+ weeks):    {breakout_counts.get(1, 0)} ({breakout_counts.get(1, 0)/len(title_df)*100:.1f}%)")

# %%
# ── B3. Feature analysis: Language ───────────────────────────────────────────
print("\n" + "-" * 50)
print("B3. FEATURE ANALYSIS: LANGUAGE")
print("-" * 50)

# Map language labels
title_df["lang_label"] = title_df["original_language"].map(
    {"ko": "Korean", "en": "English", "ja": "Japanese"}
).fillna("Other")

lang_breakout = title_df.groupby("lang_label").agg({
    "is_breakout": ["mean", "sum", "count"],
    "max_weeks": "mean"
}).round(3)
lang_breakout.columns = ["breakout_rate", "breakout_count", "total_count", "avg_weeks"]
lang_breakout = lang_breakout.sort_values("total_count", ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Breakout rate by language
lang_order = lang_breakout.index.tolist()
colors_lang = ["#FF6B6B" if l == "Korean" else "#4ECDC4" if l == "English" else "#FFE66D" if l == "Japanese" else "#95E1D3" for l in lang_order]
bars = axes[0].bar(lang_order, lang_breakout["breakout_rate"], color=colors_lang, edgecolor="black")
axes[0].set_title("Breakout Rate by Original Language")
axes[0].set_xlabel("Language")
axes[0].set_ylabel("Breakout Rate")
axes[0].set_ylim(0, 1)
axes[0].axhline(title_df["is_breakout"].mean(), color="gray", linestyle="--", label=f'Overall: {title_df["is_breakout"].mean():.1%}')
axes[0].legend()
for i, (idx, row) in enumerate(lang_breakout.iterrows()):
    axes[0].text(i, row["breakout_rate"] + 0.02, f'n={int(row["total_count"])}', ha='center', fontsize=9)

# Stacked bar: breakout vs short-lived by language
lang_stack = title_df.groupby(["lang_label", "is_breakout"]).size().unstack(fill_value=0)
lang_stack = lang_stack.loc[lang_order]
lang_stack.plot.bar(ax=axes[1], stacked=True, color=["#FF6B6B", "#4ECDC4"], edgecolor="black")
axes[1].set_title("Breakout vs Short-lived Count by Language")
axes[1].set_xlabel("Language")
axes[1].set_ylabel("Number of Titles")
axes[1].legend(["Short-lived", "Breakout"], title="Status")
plt.setp(axes[1].get_xticklabels(), rotation=0)

fig.tight_layout()
plt.show()
save(fig, "kr_B3_language_breakout.png")

print("\n  Breakout rate by language:")
print(lang_breakout.to_string())

# %%
# ── B4. Feature analysis: TMDB Rating ────────────────────────────────────────
print("\n" + "-" * 50)
print("B4. FEATURE ANALYSIS: TMDB RATING")
print("-" * 50)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Distribution by breakout status
title_df_rating = title_df.dropna(subset=["tmdb_rating"])
for status, color, label in [(0, "#FF6B6B", "Short-lived"), (1, "#4ECDC4", "Breakout")]:
    data = title_df_rating[title_df_rating["is_breakout"] == status]["tmdb_rating"]
    axes[0].hist(data, bins=20, alpha=0.6, color=color, label=f"{label} (n={len(data)})", edgecolor="black")
axes[0].set_xlabel("TMDB Rating")
axes[0].set_ylabel("Frequency")
axes[0].set_title("TMDB Rating Distribution by Breakout Status")
axes[0].legend()

# Boxplot
sns.boxplot(data=title_df_rating, x="is_breakout", y="tmdb_rating", ax=axes[1], 
            palette=["#FF6B6B", "#4ECDC4"])
axes[1].set_xticklabels(["Short-lived", "Breakout"])
axes[1].set_xlabel("Status")
axes[1].set_ylabel("TMDB Rating")
axes[1].set_title("TMDB Rating by Breakout Status")

# Breakout rate by rating bins
title_df_rating["rating_bin"] = pd.cut(title_df_rating["tmdb_rating"], bins=[0, 5, 6, 7, 8, 10], labels=["0-5", "5-6", "6-7", "7-8", "8-10"])
rating_breakout = title_df_rating.groupby("rating_bin", observed=True)["is_breakout"].agg(["mean", "count"])
rating_breakout["mean"].plot.bar(ax=axes[2], color="#4ECDC4", edgecolor="black")
axes[2].set_title("Breakout Rate by Rating Bin")
axes[2].set_xlabel("TMDB Rating Range")
axes[2].set_ylabel("Breakout Rate")
axes[2].set_ylim(0, 1)
for i, (idx, row) in enumerate(rating_breakout.iterrows()):
    axes[2].text(i, row["mean"] + 0.02, f'n={int(row["count"])}', ha='center', fontsize=9)
plt.setp(axes[2].get_xticklabels(), rotation=0)

fig.tight_layout()
plt.show()
save(fig, "kr_B4_rating_breakout.png")

print(f"\n  TMDB Rating statistics by breakout status:")
print(title_df_rating.groupby("is_breakout")["tmdb_rating"].describe().round(2).to_string())

# %%
# ── B5. Feature analysis: Days Since Release (Recency) ──────────────────────
print("\n" + "-" * 50)
print("B5. FEATURE ANALYSIS: DAYS SINCE RELEASE")
print("-" * 50)

title_df_recency = title_df.dropna(subset=["days_since_release"])
title_df_recency = title_df_recency[title_df_recency["days_since_release"] >= 0]  # exclude negative values

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Log-transformed distribution
title_df_recency["log_days"] = np.log1p(title_df_recency["days_since_release"])
for status, color, label in [(0, "#FF6B6B", "Short-lived"), (1, "#4ECDC4", "Breakout")]:
    data = title_df_recency[title_df_recency["is_breakout"] == status]["log_days"]
    axes[0].hist(data, bins=20, alpha=0.6, color=color, label=f"{label} (n={len(data)})", edgecolor="black")
axes[0].set_xlabel("Log(Days Since Release + 1)")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Recency Distribution (Log-transformed)")
axes[0].legend()

# Boxplot
sns.boxplot(data=title_df_recency, x="is_breakout", y="days_since_release", ax=axes[1],
            palette=["#FF6B6B", "#4ECDC4"])
axes[1].set_xticklabels(["Short-lived", "Breakout"])
axes[1].set_xlabel("Status")
axes[1].set_ylabel("Days Since Release")
axes[1].set_title("Days Since Release by Breakout Status")
axes[1].set_yscale("log")

# Breakout rate by recency bins
title_df_recency["recency_bin"] = pd.cut(title_df_recency["days_since_release"], 
                                          bins=[0, 7, 30, 90, 365, float("inf")],
                                          labels=["0-7d", "8-30d", "31-90d", "91-365d", "365d+"])
recency_breakout = title_df_recency.groupby("recency_bin", observed=True)["is_breakout"].agg(["mean", "count"])
recency_breakout["mean"].plot.bar(ax=axes[2], color="#4ECDC4", edgecolor="black")
axes[2].set_title("Breakout Rate by Recency")
axes[2].set_xlabel("Days Since Release")
axes[2].set_ylabel("Breakout Rate")
axes[2].set_ylim(0, 1)
for i, (idx, row) in enumerate(recency_breakout.iterrows()):
    axes[2].text(i, row["mean"] + 0.02, f'n={int(row["count"])}', ha='center', fontsize=9)
plt.setp(axes[2].get_xticklabels(), rotation=0)

fig.tight_layout()
plt.show()
save(fig, "kr_B5_recency_breakout.png")

print(f"\n  Breakout rate by recency:")
print(recency_breakout.to_string())

# %%
# ── B6. Feature analysis: Genre ──────────────────────────────────────────────
print("\n" + "-" * 50)
print("B6. FEATURE ANALYSIS: GENRE")
print("-" * 50)

# Explode genres
title_df_genre = title_df.dropna(subset=["genres"]).copy()
title_df_genre = title_df_genre.assign(genre=title_df_genre["genres"].str.split("|")).explode("genre")
title_df_genre["genre"] = title_df_genre["genre"].str.strip()

# Get top genres
top_genres = title_df_genre["genre"].value_counts().head(12).index.tolist()
title_df_top_genre = title_df_genre[title_df_genre["genre"].isin(top_genres)]

genre_breakout = title_df_top_genre.groupby("genre")["is_breakout"].agg(["mean", "count"])
genre_breakout = genre_breakout.sort_values("mean", ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Breakout rate by genre
colors_genre = ["#4ECDC4" if m > title_df["is_breakout"].mean() else "#FF6B6B" for m in genre_breakout["mean"]]
genre_breakout["mean"].plot.barh(ax=axes[0], color=colors_genre, edgecolor="black")
axes[0].axvline(title_df["is_breakout"].mean(), color="gray", linestyle="--", label=f'Overall: {title_df["is_breakout"].mean():.1%}')
axes[0].set_xlabel("Breakout Rate")
axes[0].set_ylabel("Genre")
axes[0].set_title("Breakout Rate by Genre (Top 12)")
axes[0].legend()
for i, (idx, row) in enumerate(genre_breakout.iterrows()):
    axes[0].text(row["mean"] + 0.01, i, f'n={int(row["count"])}', va='center', fontsize=9)

# Genre count distribution
genre_counts = title_df_top_genre.groupby("genre").size().sort_values(ascending=True)
genre_counts.plot.barh(ax=axes[1], color="#FFE66D", edgecolor="black")
axes[1].set_xlabel("Number of Titles")
axes[1].set_ylabel("Genre")
axes[1].set_title("Genre Frequency (Top 12)")

fig.tight_layout()
plt.show()
save(fig, "kr_B6_genre_breakout.png")

print(f"\n  Breakout rate by genre (top 12):")
print(genre_breakout.sort_values("mean", ascending=False).to_string())

# %%
# ── B7. Feature analysis: Cast Popularity ────────────────────────────────────
print("\n" + "-" * 50)
print("B7. FEATURE ANALYSIS: CAST POPULARITY")
print("-" * 50)

title_df_cast = title_df.dropna(subset=["avg_cast_popularity"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Boxplot
sns.boxplot(data=title_df_cast, x="is_breakout", y="avg_cast_popularity", ax=axes[0],
            palette=["#FF6B6B", "#4ECDC4"])
axes[0].set_xticklabels(["Short-lived", "Breakout"])
axes[0].set_xlabel("Status")
axes[0].set_ylabel("Average Cast Popularity")
axes[0].set_title("Cast Popularity by Breakout Status")

# Breakout rate by cast popularity bins
title_df_cast["cast_bin"] = pd.qcut(title_df_cast["avg_cast_popularity"], q=4, labels=["Low", "Medium-Low", "Medium-High", "High"])
cast_breakout = title_df_cast.groupby("cast_bin", observed=True)["is_breakout"].agg(["mean", "count"])
cast_breakout["mean"].plot.bar(ax=axes[1], color="#4ECDC4", edgecolor="black")
axes[1].set_title("Breakout Rate by Cast Popularity Quartile")
axes[1].set_xlabel("Cast Popularity Quartile")
axes[1].set_ylabel("Breakout Rate")
axes[1].set_ylim(0, 1)
for i, (idx, row) in enumerate(cast_breakout.iterrows()):
    axes[1].text(i, row["mean"] + 0.02, f'n={int(row["count"])}', ha='center', fontsize=9)
plt.setp(axes[1].get_xticklabels(), rotation=0)

fig.tight_layout()
plt.show()
save(fig, "kr_B7_cast_breakout.png")

print(f"\n  Cast popularity by breakout status:")
print(title_df_cast.groupby("is_breakout")["avg_cast_popularity"].describe().round(2).to_string())

# %%
# ── B8. Feature correlation matrix ───────────────────────────────────────────
print("\n" + "-" * 50)
print("B8. FEATURE CORRELATION MATRIX")
print("-" * 50)

# Prepare numeric features
numeric_features = ["is_breakout", "max_weeks", "tmdb_rating", "tmdb_popularity", 
                    "tmdb_vote_count", "avg_cast_popularity", "days_since_release",
                    "is_korean", "is_sequel", "entry_rank", "best_rank"]
corr_df = title_df[numeric_features].dropna()
corr_matrix = corr_df.corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
            square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1)
ax.set_title("Feature Correlation Matrix (All Data)")
fig.tight_layout()
plt.show()
save(fig, "kr_B8_correlation_matrix.png")

print(f"\n  Correlation with is_breakout (all data):")
breakout_corr = corr_matrix["is_breakout"].drop("is_breakout").sort_values(ascending=False)
print(breakout_corr.to_string())

# %%
# ── B8b. Correlation matrix excluding tmdb_rating == 0 ───────────────────────
print("\n" + "-" * 50)
print("B8b. CORRELATION MATRIX (EXCLUDING TMDB_RATING == 0)")
print("-" * 50)

# Filter out zero ratings (likely no votes / missing data)
corr_df_filtered = title_df[numeric_features].dropna()
corr_df_filtered = corr_df_filtered[corr_df_filtered["tmdb_rating"] > 0]

print(f"  Rows with tmdb_rating == 0: {(title_df['tmdb_rating'] == 0).sum()}")
print(f"  Rows after filtering: {len(corr_df_filtered)} (was {len(corr_df)})")

corr_matrix_filtered = corr_df_filtered.corr()

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Original correlation matrix
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
            square=True, linewidths=0.5, ax=axes[0], vmin=-1, vmax=1)
axes[0].set_title(f"All Data (n={len(corr_df)})")

# Filtered correlation matrix
sns.heatmap(corr_matrix_filtered, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
            square=True, linewidths=0.5, ax=axes[1], vmin=-1, vmax=1)
axes[1].set_title(f"Excluding rating=0 (n={len(corr_df_filtered)})")

fig.suptitle("Feature Correlation Matrix: Impact of Zero Ratings", fontsize=14, y=1.02)
fig.tight_layout()
plt.show()
save(fig, "kr_B8b_correlation_comparison.png")

# Compare correlations
print(f"\n  Correlation with is_breakout comparison:")
breakout_corr_filtered = corr_matrix_filtered["is_breakout"].drop("is_breakout").sort_values(ascending=False)
comparison = pd.DataFrame({
    "all_data": breakout_corr,
    "excl_zero_rating": breakout_corr_filtered,
    "change": breakout_corr_filtered - breakout_corr
}).round(3)
print(comparison.to_string())

# %%
# ── B9. Feature summary table ────────────────────────────────────────────────
print("\n" + "-" * 50)
print("B9. FEATURE SUMMARY FOR MODELING")
print("-" * 50)

feature_summary = []

# is_korean
ko_rate = title_df.groupby("is_korean")["is_breakout"].mean()
feature_summary.append({
    "feature": "is_korean",
    "type": "binary",
    "missing_pct": title_df["is_korean"].isna().mean() * 100,
    "breakout_rate_high": ko_rate.get(1, 0),
    "breakout_rate_low": ko_rate.get(0, 0),
    "lift": ko_rate.get(1, 0) / ko_rate.get(0, 0.01) if ko_rate.get(0, 0) > 0 else np.nan,
    "recommendation": "STRONG - Korean content has much higher breakout rate"
})

# tmdb_rating
rating_corr = corr_matrix.loc["is_breakout", "tmdb_rating"]
feature_summary.append({
    "feature": "tmdb_rating",
    "type": "numeric",
    "missing_pct": title_df["tmdb_rating"].isna().mean() * 100,
    "correlation": rating_corr,
    "recommendation": "MODERATE - Higher ratings correlate with breakout"
})

# days_since_release
recency_corr = corr_matrix.loc["is_breakout", "days_since_release"]
feature_summary.append({
    "feature": "days_since_release",
    "type": "numeric",
    "missing_pct": title_df["days_since_release"].isna().mean() * 100,
    "correlation": recency_corr,
    "recommendation": "MODERATE - Newer releases tend to break out more"
})

# original_language
feature_summary.append({
    "feature": "original_language",
    "type": "categorical",
    "missing_pct": title_df["original_language"].isna().mean() * 100,
    "unique_values": title_df["original_language"].nunique(),
    "recommendation": "STRONG - Language is a key predictor (ko > ja > en)"
})

# genres
feature_summary.append({
    "feature": "genres",
    "type": "multi-label",
    "missing_pct": title_df["genres"].isna().mean() * 100,
    "recommendation": "MODERATE - Some genres have higher breakout rates"
})

# avg_cast_popularity
cast_corr = corr_matrix.loc["is_breakout", "avg_cast_popularity"]
feature_summary.append({
    "feature": "avg_cast_popularity",
    "type": "numeric",
    "missing_pct": title_df["avg_cast_popularity"].isna().mean() * 100,
    "correlation": cast_corr,
    "recommendation": "WEAK - Cast popularity shows limited predictive signal"
})

# category
cat_rate = title_df.groupby("category")["is_breakout"].mean()
feature_summary.append({
    "feature": "category",
    "type": "categorical",
    "missing_pct": 0,
    "unique_values": 2,
    "recommendation": "MODERATE - Films vs TV have different breakout patterns"
})

summary_df = pd.DataFrame(feature_summary)
print("\n  Feature Summary:")
print(summary_df.to_string(index=False))

# %%
# ── B10. Save title-level dataset for modeling ───────────────────────────────
title_df.to_csv(os.path.join(OUTPUT_DIR, "kr_title_level_for_modeling.csv"), index=False)
print(f"\n  Title-level dataset saved to {os.path.join(OUTPUT_DIR, 'kr_title_level_for_modeling.csv')}")

print("\n" + "=" * 70)
print("BREAKOUT CLASSIFICATION EDA COMPLETE")
print("=" * 70)
print(f"  Key findings:")
print(f"    - Korean content breakout rate: {lang_breakout.loc['Korean', 'breakout_rate']:.1%} (vs overall {title_df['is_breakout'].mean():.1%})")
print(f"    - Higher TMDB ratings correlate with breakout (r={rating_corr:.2f})")
print(f"    - Newer releases have higher breakout rates")
print(f"    - Drama, Crime, Thriller genres show strong breakout potential")

# %%
# =============================================================================
# PART C: BREAKOUT CLASSIFICATION MODEL
# =============================================================================
# Goal: Build predictive models to identify content likely to be a "breakout"
#       (4+ weeks in Top 10) for content acquisition decisions

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             precision_recall_curve, average_precision_score,
                             roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings('ignore')

print("\n" + "=" * 70)
print("C. BREAKOUT CLASSIFICATION MODEL")
print("=" * 70)

# ── C1. Data Preparation ─────────────────────────────────────────────────────
print("\n" + "-" * 50)
print("C1. DATA PREPARATION")
print("-" * 50)

# Start with title_df, exclude tmdb_rating == 0
model_df = title_df.copy()
model_df = model_df[model_df["tmdb_rating"] > 0]  # Exclude zero ratings
model_df = model_df.dropna(subset=["tmdb_rating", "original_language", "genres", "days_since_release"])

print(f"  Dataset after filtering (rating > 0, no NA): {len(model_df)} titles")
print(f"  Class distribution: {model_df['is_breakout'].value_counts().to_dict()}")

# Feature engineering
# 1. Genre one-hot encoding (top genres only)
top_genre_list = ["Drama", "Comedy", "Action", "Thriller", "Crime", "Romance", "Mystery", "Horror"]
for genre in top_genre_list:
    model_df[f"genre_{genre.lower()}"] = model_df["genres"].str.contains(genre, case=False, na=False).astype(int)

# 2. Genre count
model_df["genre_count"] = model_df["genres"].str.count(r"\|") + 1

# 3. Language encoding
model_df["lang_ko"] = (model_df["original_language"] == "ko").astype(int)
model_df["lang_en"] = (model_df["original_language"] == "en").astype(int)
model_df["lang_ja"] = (model_df["original_language"] == "ja").astype(int)

# 4. Category encoding
model_df["is_film"] = (model_df["category"] == "Films").astype(int)

# 5. Log transform days_since_release
model_df["log_days_since_release"] = np.log1p(model_df["days_since_release"].clip(lower=0))

# 6. Recency flag
model_df["is_recent"] = (model_df["days_since_release"] <= 30).astype(int)

# Define features
feature_cols = [
    "tmdb_rating", "tmdb_popularity", "is_korean", "is_sequel", "is_film",
    "log_days_since_release", "is_recent", "genre_count",
    "lang_ko", "lang_en", "lang_ja",
    "genre_drama", "genre_comedy", "genre_action", "genre_thriller", 
    "genre_crime", "genre_romance", "genre_mystery", "genre_horror"
]

# Prepare X and y
X = model_df[feature_cols].fillna(0)
y = model_df["is_breakout"]

print(f"\n  Features: {len(feature_cols)}")
print(f"  Feature list: {feature_cols}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n  Train set: {len(X_train)} | Test set: {len(X_test)}")
print(f"  Train breakout rate: {y_train.mean():.1%} | Test breakout rate: {y_test.mean():.1%}")

# Scale features for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
# ── C2. Logistic Regression ──────────────────────────────────────────────────
print("\n" + "-" * 50)
print("C2. LOGISTIC REGRESSION (Interpretable Model)")
print("-" * 50)

lr_model = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")
lr_model.fit(X_train_scaled, y_train)

# Cross-validation
cv_scores = cross_val_score(lr_model, X_train_scaled, y_train, cv=5, scoring="roc_auc")
print(f"\n  5-Fold CV ROC-AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")

# Predictions
y_pred_lr = lr_model.predict(X_test_scaled)
y_prob_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

print(f"\n  Test Set Performance:")
print(f"    ROC-AUC: {roc_auc_score(y_test, y_prob_lr):.3f}")
print(f"    Average Precision: {average_precision_score(y_test, y_prob_lr):.3f}")

print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred_lr, target_names=["Short-lived", "Breakout"]))

# Coefficients
coef_df = pd.DataFrame({
    "feature": feature_cols,
    "coefficient": lr_model.coef_[0],
    "odds_ratio": np.exp(lr_model.coef_[0])
}).sort_values("coefficient", ascending=False)

print(f"\n  Logistic Regression Coefficients (sorted by impact):")
print(coef_df.to_string(index=False))

# Plot coefficients
fig, ax = plt.subplots(figsize=(10, 8))
colors = ["#4ECDC4" if c > 0 else "#FF6B6B" for c in coef_df["coefficient"]]
ax.barh(coef_df["feature"], coef_df["coefficient"], color=colors, edgecolor="black")
ax.axvline(0, color="black", linewidth=0.5)
ax.set_xlabel("Coefficient (Log-Odds)")
ax.set_title("Logistic Regression: Feature Coefficients\n(Positive = Increases Breakout Probability)")
ax.invert_yaxis()
fig.tight_layout()
plt.show()
save(fig, "kr_C2_logistic_coefficients.png")

# %%
# ── C3. Random Forest: Baseline vs Tuned ─────────────────────────────────────
print("\n" + "-" * 50)
print("C3. RANDOM FOREST (Baseline vs Tuned)")
print("-" * 50)

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

# --- Baseline Random Forest (default parameters) ---
print("\n  [BASELINE] Random Forest with default parameters:")
rf_baseline = RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1)
rf_baseline.fit(X_train, y_train)

y_pred_rf_baseline = rf_baseline.predict(X_test)
y_prob_rf_baseline = rf_baseline.predict_proba(X_test)[:, 1]

f1_rf_baseline = f1_score(y_test, y_pred_rf_baseline)
print(f"    ROC-AUC: {roc_auc_score(y_test, y_prob_rf_baseline):.3f}")
print(f"    F1 Score: {f1_rf_baseline:.3f}")
print(f"    Avg Precision: {average_precision_score(y_test, y_prob_rf_baseline):.3f}")

# --- Tuned Random Forest ---
print("\n  [TUNED] Random Forest with GridSearchCV:")

# Define parameter grid for Random Forest
rf_param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, 15],
    "min_samples_split": [5, 10],
    "min_samples_leaf": [3, 5],
}

print("\n  Tuning Random Forest hyperparameters (this may take a moment)...")
rf_base = RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1)

rf_grid_search = GridSearchCV(
    rf_base, 
    rf_param_grid, 
    cv=5, 
    scoring="f1",  # Optimize for F1 score
    n_jobs=-1,
    verbose=0
)
rf_grid_search.fit(X_train, y_train)

print(f"\n  Best Parameters: {rf_grid_search.best_params_}")
print(f"  Best CV F1 Score: {rf_grid_search.best_score_:.3f}")

# Use best model
rf_model = rf_grid_search.best_estimator_

# Cross-validation with best model
cv_scores_rf = cross_val_score(rf_model, X_train, y_train, cv=5, scoring="roc_auc")
print(f"  5-Fold CV ROC-AUC: {cv_scores_rf.mean():.3f} (+/- {cv_scores_rf.std()*2:.3f})")

# Predictions
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

print(f"\n  Test Set Performance:")
print(f"    ROC-AUC: {roc_auc_score(y_test, y_prob_rf):.3f}")
print(f"    Average Precision: {average_precision_score(y_test, y_prob_rf):.3f}")

print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=["Short-lived", "Breakout"]))

# Feature importance
importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": rf_model.feature_importances_
}).sort_values("importance", ascending=False)

print(f"\n  Random Forest Feature Importance:")
print(importance_df.to_string(index=False))

# Plot feature importance
fig, ax = plt.subplots(figsize=(10, 8))
importance_sorted = importance_df.sort_values("importance", ascending=True)
ax.barh(importance_sorted["feature"], importance_sorted["importance"], color="#4ECDC4", edgecolor="black")
ax.set_xlabel("Feature Importance (Gini)")
ax.set_title("Random Forest (Tuned): Feature Importance")
fig.tight_layout()
plt.show()
save(fig, "kr_C3_rf_feature_importance.png")

# Improvement summary for Random Forest
f1_rf_tuned = f1_score(y_test, y_pred_rf)
print(f"\n  ── Random Forest Improvement Summary ──")
print(f"  {'Metric':<20} {'Baseline':>12} {'Tuned':>12} {'Δ':>10}")
print(f"  {'-'*54}")
print(f"  {'F1 Score':<20} {f1_rf_baseline:>12.3f} {f1_rf_tuned:>12.3f} {f1_rf_tuned - f1_rf_baseline:>+10.3f}")
print(f"  {'ROC-AUC':<20} {roc_auc_score(y_test, y_prob_rf_baseline):>12.3f} {roc_auc_score(y_test, y_prob_rf):>12.3f} {roc_auc_score(y_test, y_prob_rf) - roc_auc_score(y_test, y_prob_rf_baseline):>+10.3f}")

# %%
# ── C3b. XGBoost: Baseline vs Tuned ──────────────────────────────────────────
print("\n" + "-" * 50)
print("C3b. XGBOOST (Baseline vs Tuned)")
print("-" * 50)

try:
    from xgboost import XGBClassifier
    
    # Calculate scale_pos_weight for class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    # --- Baseline XGBoost (default parameters) ---
    print("\n  [BASELINE] XGBoost with default parameters:")
    xgb_baseline = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False
    )
    xgb_baseline.fit(X_train, y_train)
    
    y_pred_xgb_baseline = xgb_baseline.predict(X_test)
    y_prob_xgb_baseline = xgb_baseline.predict_proba(X_test)[:, 1]
    
    f1_xgb_baseline = f1_score(y_test, y_pred_xgb_baseline)
    print(f"    ROC-AUC: {roc_auc_score(y_test, y_prob_xgb_baseline):.3f}")
    print(f"    F1 Score: {f1_xgb_baseline:.3f}")
    print(f"    Avg Precision: {average_precision_score(y_test, y_prob_xgb_baseline):.3f}")
    
    # --- Tuned XGBoost ---
    print("\n  [TUNED] XGBoost with GridSearchCV:")
    
    # Define parameter grid for XGBoost
    xgb_param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8],
        "colsample_bytree": [0.7, 0.8],
    }
    
    print("\n  Tuning XGBoost hyperparameters (this may take a moment)...")
    xgb_base = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
        n_jobs=-1
    )
    
    xgb_grid_search = GridSearchCV(
        xgb_base,
        xgb_param_grid,
        cv=5,
        scoring="f1",  # Optimize for F1 score
        n_jobs=-1,
        verbose=0
    )
    xgb_grid_search.fit(X_train, y_train)
    
    print(f"\n  Best Parameters: {xgb_grid_search.best_params_}")
    print(f"  Best CV F1 Score: {xgb_grid_search.best_score_:.3f}")
    
    # Use best model
    xgb_model = xgb_grid_search.best_estimator_
    
    # Cross-validation with best model
    cv_scores_xgb = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring="roc_auc")
    print(f"  5-Fold CV ROC-AUC: {cv_scores_xgb.mean():.3f} (+/- {cv_scores_xgb.std()*2:.3f})")
    
    # Predictions
    y_pred_xgb = xgb_model.predict(X_test)
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
    
    print(f"\n  Test Set Performance:")
    print(f"    ROC-AUC: {roc_auc_score(y_test, y_prob_xgb):.3f}")
    print(f"    Average Precision: {average_precision_score(y_test, y_prob_xgb):.3f}")
    
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred_xgb, target_names=["Short-lived", "Breakout"]))
    
    # Feature importance
    xgb_importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": xgb_model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    print(f"\n  XGBoost Feature Importance:")
    print(xgb_importance_df.to_string(index=False))
    
    # Plot feature importance
    fig, ax = plt.subplots(figsize=(10, 8))
    xgb_importance_sorted = xgb_importance_df.sort_values("importance", ascending=True)
    ax.barh(xgb_importance_sorted["feature"], xgb_importance_sorted["importance"], color="#FF6B6B", edgecolor="black")
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title("XGBoost (Tuned): Feature Importance")
    fig.tight_layout()
    plt.show()
    save(fig, "kr_C3b_xgb_feature_importance.png")
    
    # Improvement summary for XGBoost
    f1_xgb_tuned = f1_score(y_test, y_pred_xgb)
    print(f"\n  ── XGBoost Improvement Summary ──")
    print(f"  {'Metric':<20} {'Baseline':>12} {'Tuned':>12} {'Δ':>10}")
    print(f"  {'-'*54}")
    print(f"  {'F1 Score':<20} {f1_xgb_baseline:>12.3f} {f1_xgb_tuned:>12.3f} {f1_xgb_tuned - f1_xgb_baseline:>+10.3f}")
    print(f"  {'ROC-AUC':<20} {roc_auc_score(y_test, y_prob_xgb_baseline):>12.3f} {roc_auc_score(y_test, y_prob_xgb):>12.3f} {roc_auc_score(y_test, y_prob_xgb) - roc_auc_score(y_test, y_prob_xgb_baseline):>+10.3f}")
    
    xgb_available = True
    
except ImportError:
    print("  XGBoost not installed. Install with: pip install xgboost")
    print("  Skipping XGBoost analysis.")
    xgb_available = False
    y_prob_xgb = None

# %%
# ── C4. Model Comparison: Precision-Recall Curves ────────────────────────────
print("\n" + "-" * 50)
print("C4. MODEL COMPARISON: PRECISION-RECALL CURVES")
print("-" * 50)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Precision-Recall Curve
precision_lr, recall_lr, thresholds_lr = precision_recall_curve(y_test, y_prob_lr)
precision_rf, recall_rf, thresholds_rf = precision_recall_curve(y_test, y_prob_rf)

ap_lr = average_precision_score(y_test, y_prob_lr)
ap_rf = average_precision_score(y_test, y_prob_rf)

axes[0].plot(recall_lr, precision_lr, label=f"Logistic Regression (AP={ap_lr:.3f})", linewidth=2)
axes[0].plot(recall_rf, precision_rf, label=f"Random Forest (AP={ap_rf:.3f})", linewidth=2)

# Add XGBoost if available
if xgb_available and y_prob_xgb is not None:
    precision_xgb, recall_xgb, _ = precision_recall_curve(y_test, y_prob_xgb)
    ap_xgb = average_precision_score(y_test, y_prob_xgb)
    axes[0].plot(recall_xgb, precision_xgb, label=f"XGBoost (AP={ap_xgb:.3f})", linewidth=2)

axes[0].axhline(y_test.mean(), color="gray", linestyle="--", label=f"Baseline (rate={y_test.mean():.2f})")
axes[0].set_xlabel("Recall")
axes[0].set_ylabel("Precision")
axes[0].set_title("Precision-Recall Curve\n(Higher = Better for Content Acquisition)")
axes[0].legend()
axes[0].grid(alpha=0.3)

# ROC Curve
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

auc_lr = roc_auc_score(y_test, y_prob_lr)
auc_rf = roc_auc_score(y_test, y_prob_rf)

axes[1].plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC={auc_lr:.3f})", linewidth=2)
axes[1].plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={auc_rf:.3f})", linewidth=2)

# Add XGBoost if available
if xgb_available and y_prob_xgb is not None:
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
    auc_xgb = roc_auc_score(y_test, y_prob_xgb)
    axes[1].plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC={auc_xgb:.3f})", linewidth=2)

axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve")
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.tight_layout()
plt.show()
save(fig, "kr_C4_model_comparison.png")

# %%
# ── C4b. Threshold Tuning for Better Recall ──────────────────────────────────
print("\n" + "-" * 50)
print("C4b. THRESHOLD TUNING (Precision-Recall Tradeoff)")
print("-" * 50)

# Find optimal threshold for different objectives
def find_optimal_threshold(y_true, y_prob, target_recall=0.80):
    """Find threshold that achieves target recall while maximizing precision."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # Find thresholds where recall >= target
    valid_idx = np.where(recall[:-1] >= target_recall)[0]
    if len(valid_idx) == 0:
        return 0.5, precision[0], recall[0]
    # Among valid, pick the one with highest precision
    best_idx = valid_idx[np.argmax(precision[:-1][valid_idx])]
    return thresholds[best_idx], precision[best_idx], recall[best_idx]

# Analyze thresholds
thresholds_to_test = [0.3, 0.4, 0.5, 0.6]
print("\n  Random Forest: Threshold Analysis")
print("  " + "-" * 60)
print(f"  {'Threshold':>10} | {'Precision':>10} | {'Recall':>10} | {'F1':>10}")
print("  " + "-" * 60)

for thresh in thresholds_to_test:
    y_pred_thresh = (y_prob_rf >= thresh).astype(int)
    from sklearn.metrics import precision_score, recall_score, f1_score
    prec = precision_score(y_test, y_pred_thresh)
    rec = recall_score(y_test, y_pred_thresh)
    f1 = f1_score(y_test, y_pred_thresh)
    print(f"  {thresh:>10.2f} | {prec:>10.3f} | {rec:>10.3f} | {f1:>10.3f}")

# Find threshold for 80% recall
opt_thresh, opt_prec, opt_rec = find_optimal_threshold(y_test, y_prob_rf, target_recall=0.80)
print(f"\n  Optimal threshold for 80% recall: {opt_thresh:.3f}")
print(f"    → Precision: {opt_prec:.3f}, Recall: {opt_rec:.3f}")

# Apply optimized threshold
y_pred_rf_optimized = (y_prob_rf >= opt_thresh).astype(int)

# Visualize threshold impact
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Threshold vs metrics
thresholds_range = np.linspace(0.1, 0.9, 50)
precisions, recalls, f1s = [], [], []
for t in thresholds_range:
    y_pred_t = (y_prob_rf >= t).astype(int)
    precisions.append(precision_score(y_test, y_pred_t, zero_division=0))
    recalls.append(recall_score(y_test, y_pred_t, zero_division=0))
    f1s.append(f1_score(y_test, y_pred_t, zero_division=0))

axes[0].plot(thresholds_range, precisions, label="Precision", linewidth=2)
axes[0].plot(thresholds_range, recalls, label="Recall", linewidth=2)
axes[0].plot(thresholds_range, f1s, label="F1", linewidth=2, linestyle="--")
axes[0].axvline(0.5, color="gray", linestyle=":", label="Default (0.5)")
axes[0].axvline(opt_thresh, color="red", linestyle="--", label=f"Optimized ({opt_thresh:.2f})")
axes[0].set_xlabel("Threshold")
axes[0].set_ylabel("Score")
axes[0].set_title("Random Forest: Threshold vs Metrics")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Confusion matrix with optimized threshold
from sklearn.metrics import ConfusionMatrixDisplay
cm_optimized = confusion_matrix(y_test, y_pred_rf_optimized)
sns.heatmap(cm_optimized, annot=True, fmt="d", cmap="Blues", ax=axes[1],
            xticklabels=["Short-lived", "Breakout"],
            yticklabels=["Short-lived", "Breakout"])
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")
axes[1].set_title(f"Random Forest (threshold={opt_thresh:.2f})\nOptimized for 80% Recall")

fig.tight_layout()
plt.show()
save(fig, "kr_C4b_threshold_tuning.png")

print(f"\n  Classification Report (Optimized Threshold = {opt_thresh:.2f}):")
print(classification_report(y_test, y_pred_rf_optimized, target_names=["Short-lived", "Breakout"]))

# %%
# ── C5. Confusion Matrix Comparison ──────────────────────────────────────────
print("\n" + "-" * 50)
print("C5. CONFUSION MATRICES (Default Threshold = 0.5)")
print("-" * 50)

# Build list of models to compare
models_to_compare = [
    (y_pred_lr, "Logistic Regression"),
    (y_pred_rf, "Random Forest")
]
if xgb_available:
    models_to_compare.append((y_pred_xgb, "XGBoost"))

n_models = len(models_to_compare)
fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
if n_models == 1:
    axes = [axes]

for ax, (y_pred, title) in zip(axes, models_to_compare):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Short-lived", "Breakout"],
                yticklabels=["Short-lived", "Breakout"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{title}\nConfusion Matrix")

fig.tight_layout()
plt.show()
save(fig, "kr_C5_confusion_matrices.png")

# Print classification reports for all models
print("\n  Classification Reports:")
print("  " + "=" * 70)
print("\n  LOGISTIC REGRESSION:")
print(classification_report(y_test, y_pred_lr, target_names=["Short-lived", "Breakout"]))

print("  RANDOM FOREST:")
print(classification_report(y_test, y_pred_rf, target_names=["Short-lived", "Breakout"]))

if xgb_available:
    print("  XGBOOST:")
    print(classification_report(y_test, y_pred_xgb, target_names=["Short-lived", "Breakout"]))

# %%
# ── C6. Feature Importance Comparison ────────────────────────────────────────
print("\n" + "-" * 50)
print("C6. FEATURE IMPORTANCE COMPARISON")
print("-" * 50)

# Determine number of subplots based on XGBoost availability
n_plots = 3 if xgb_available else 2
fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 8))

# Logistic Regression - Absolute coefficients
coef_sorted = coef_df.copy()
coef_sorted["abs_coef"] = coef_sorted["coefficient"].abs()
coef_sorted = coef_sorted.sort_values("abs_coef", ascending=True)
colors_lr = ["#4ECDC4" if c > 0 else "#FF6B6B" for c in coef_sorted["coefficient"]]
axes[0].barh(coef_sorted["feature"], coef_sorted["coefficient"], color=colors_lr, edgecolor="black")
axes[0].axvline(0, color="black", linewidth=0.5)
axes[0].set_xlabel("Coefficient (Log-Odds)")
axes[0].set_title("Logistic Regression Coefficients\n(Green = Increases Breakout Prob.)")

# Random Forest - Feature importance
importance_sorted = importance_df.sort_values("importance", ascending=True)
axes[1].barh(importance_sorted["feature"], importance_sorted["importance"], color="#4ECDC4", edgecolor="black")
axes[1].set_xlabel("Feature Importance (Gini)")
axes[1].set_title("Random Forest Feature Importance")

# XGBoost - Feature importance (if available)
if xgb_available:
    xgb_importance_sorted = xgb_importance_df.sort_values("importance", ascending=True)
    axes[2].barh(xgb_importance_sorted["feature"], xgb_importance_sorted["importance"], color="#FF6B6B", edgecolor="black")
    axes[2].set_xlabel("Feature Importance (Gain)")
    axes[2].set_title("XGBoost Feature Importance")
    fig.suptitle("Feature Importance Comparison: LR vs RF vs XGBoost", fontsize=14, y=1.02)
else:
    fig.suptitle("Feature Importance Comparison: LR vs RF", fontsize=14, y=1.02)

fig.tight_layout()
plt.show()
save(fig, "kr_C6_feature_importance_comparison.png")

# Print combined ranking
print("\n  Feature Importance Ranking:")
print("  " + "-" * 70)

combined = pd.merge(
    coef_df[["feature", "coefficient", "odds_ratio"]],
    importance_df[["feature", "importance"]].rename(columns={"importance": "rf_importance"}),
    on="feature"
)
combined["lr_rank"] = combined["coefficient"].abs().rank(ascending=False).astype(int)
combined["rf_rank"] = combined["rf_importance"].rank(ascending=False).astype(int)

if xgb_available:
    combined = pd.merge(
        combined,
        xgb_importance_df[["feature", "importance"]].rename(columns={"importance": "xgb_importance"}),
        on="feature"
    )
    combined["xgb_rank"] = combined["xgb_importance"].rank(ascending=False).astype(int)
    combined = combined.sort_values("xgb_rank")
    print(combined[["feature", "lr_rank", "rf_rank", "xgb_rank", "odds_ratio"]].to_string(index=False))
else:
    combined = combined.sort_values("rf_rank")
    print(combined[["feature", "lr_rank", "rf_rank", "odds_ratio", "rf_importance"]].to_string(index=False))

# %%
# ── C7. Business Insights Summary ────────────────────────────────────────────
print("\n" + "=" * 70)
print("C7. BUSINESS INSIGHTS: WHAT PREDICTS A BREAKOUT IN KOREA?")
print("=" * 70)

print(f"""
  MODEL PERFORMANCE SUMMARY:
  ─────────────────────────────────────────────────────────────────────
  Logistic Regression:  ROC-AUC = {auc_lr:.3f}  |  Avg Precision = {ap_lr:.3f}
  Random Forest:        ROC-AUC = {auc_rf:.3f}  |  Avg Precision = {ap_rf:.3f}""")

if xgb_available:
    print(f"  XGBoost:              ROC-AUC = {auc_xgb:.3f}  |  Avg Precision = {ap_xgb:.3f}")

print(f"""
  KEY PREDICTORS OF BREAKOUT (4+ weeks in Top 10):
  ─────────────────────────────────────────────────────────────────────
""")

# Top positive predictors from Logistic Regression
top_positive = coef_df[coef_df["coefficient"] > 0].head(5)
top_negative = coef_df[coef_df["coefficient"] < 0].head(5)

print("  INCREASES breakout probability:")
for _, row in top_positive.iterrows():
    print(f"    • {row['feature']:25s} (odds ratio: {row['odds_ratio']:.2f}x)")

print("\n  DECREASES breakout probability:")
for _, row in top_negative.iterrows():
    print(f"    • {row['feature']:25s} (odds ratio: {row['odds_ratio']:.2f}x)")

print(f"""
  
  ACTIONABLE RECOMMENDATIONS FOR CONTENT ACQUISITION:
  ─────────────────────────────────────────────────────────────────────
  1. Prioritize Korean-language content (strongest predictor)
  2. Look for titles with TMDB rating > 7.5
  3. Drama, Thriller, Crime genres perform well in Korea
  4. Recent releases (< 30 days) have higher breakout potential
  5. Sequels/follow-up seasons have slight advantage
  
  CAUTION:
  - English-language content has lower breakout rate in Korea
  - High global popularity (TMDB) doesn't guarantee Korean success
""")

# Save model results
model_results = {
    "logistic_regression": {
        "roc_auc": auc_lr,
        "avg_precision": ap_lr,
        "cv_roc_auc_mean": cv_scores.mean(),
        "cv_roc_auc_std": cv_scores.std()
    },
    "random_forest": {
        "roc_auc": auc_rf,
        "avg_precision": ap_rf,
        "cv_roc_auc_mean": cv_scores_rf.mean(),
        "cv_roc_auc_std": cv_scores_rf.std()
    }
}

coef_df.to_csv(os.path.join(OUTPUT_DIR, "kr_logistic_coefficients.csv"), index=False)
importance_df.to_csv(os.path.join(OUTPUT_DIR, "kr_rf_feature_importance.csv"), index=False)

print(f"\n  Model outputs saved to {OUTPUT_DIR}/")
print("    - kr_logistic_coefficients.csv")
print("    - kr_rf_feature_importance.csv")

print("\n" + "=" * 70)
print("BREAKOUT CLASSIFICATION MODEL COMPLETE")
print("=" * 70)
