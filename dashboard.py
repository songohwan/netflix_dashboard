"""
Netflix Korea vs Japan: Regional Content Analysis Dashboard
===========================================================
Interactive Streamlit dashboard for exploring Netflix Top 10 content
differences between Korea and Japan.

Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

# ──────────────────────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Netflix: Korea vs Japan",
    page_icon="🎬",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    kr = pd.read_csv("dataset/raw/kr_gw_enriched.csv")
    jp = pd.read_csv("dataset/raw/jp_gw_enriched.csv")

    kr["week"] = pd.to_datetime(kr["week"])
    jp["week"] = pd.to_datetime(jp["week"])

    # Align to overlapping period
    min_d = max(kr["week"].min(), jp["week"].min())
    max_d = min(kr["week"].max(), jp["week"].max())
    kr = kr[(kr["week"] >= min_d) & (kr["week"] <= max_d)]
    jp = jp[(jp["week"] >= min_d) & (jp["week"] <= max_d)]

    kr = kr.dropna(subset=["tmdb_id"])
    jp = jp.dropna(subset=["tmdb_id"])

    lang_map = {"ko": "Korean", "en": "English", "ja": "Japanese", "zh": "Chinese"}
    kr["lang_label"] = kr["original_language"].map(lang_map).fillna("Other")
    jp["lang_label"] = jp["original_language"].map(lang_map).fillna("Other")

    return kr, jp, min_d, max_d


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def explode_genres(df):
    out = df.dropna(subset=["genres"]).copy()
    out = out.assign(genre=out["genres"].str.split("|")).explode("genre")
    out["genre"] = out["genre"].str.strip()
    return out


def title_longevity(df):
    return df.groupby("show_title")["cumulative_weeks_in_top_10"].max()


def all_genres(kr, jp):
    g = set()
    for df in [kr, jp]:
        for genres in df["genres"].dropna():
            g.update(x.strip() for x in genres.split("|"))
    return sorted(g)


# ──────────────────────────────────────────────────────────────
# Tab 1 — Regional Explorer
# ──────────────────────────────────────────────────────────────
def render_tab1(kr, jp):
    st.header("Regional Explorer")
    st.caption("Select a region and explore what's trending in Netflix's Top 10.")

    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        region = st.selectbox("Region", ["Korea", "Japan"], key="t1_region")
    with col_ctrl2:
        cat_filter = st.selectbox("Category", ["All", "Films", "TV"], key="t1_cat")

    df = kr.copy() if region == "Korea" else jp.copy()
    if cat_filter != "All":
        df = df[df["category"] == cat_filter]

    if df.empty:
        st.warning("No data for the selected filters.")
        return

    # — KPI row ——————————————————————————————————————————————
    longevity = title_longevity(df)
    genre_exp = explode_genres(df)
    top_genre = genre_exp["genre"].value_counts().idxmax() if not genre_exp.empty else "N/A"
    dominant_lang = df["lang_label"].value_counts(normalize=True)
    dominant_lang_name = dominant_lang.idxmax()
    dominant_lang_pct = dominant_lang.iloc[0] * 100

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Unique Titles", f"{df['show_title'].nunique():,}")
    k2.metric("Avg Weeks in Top 10", f"{longevity.mean():.1f}")
    k3.metric("Top Genre", top_genre)
    k4.metric("Dominant Language", f"{dominant_lang_name} ({dominant_lang_pct:.0f}%)")

    # — Row 1: Genre Heatmap + Language Donut ———————————————
    c1, c2 = st.columns([3, 2])

    with c1:
        st.subheader("Genre Performance")
        if not genre_exp.empty:
            stats = genre_exp.groupby("genre").agg(
                frequency=("genre", "size"),
                avg_weeks=("cumulative_weeks_in_top_10", "mean"),
                avg_rating=("tmdb_rating", lambda x: x[x > 0].mean()),
            )
            stats["freq_pct"] = stats["frequency"] / len(df) * 100
            stats = stats.sort_values("frequency", ascending=False).head(15)

            # Normalise for heatmap colouring
            matrix = stats[["freq_pct", "avg_weeks", "avg_rating"]].copy()
            matrix.columns = ["Frequency %", "Avg Weeks", "Avg Rating"]
            normed = matrix.apply(lambda s: (s - s.min()) / (s.max() - s.min() + 1e-9))

            fig = go.Figure(
                data=go.Heatmap(
                    z=normed.values,
                    x=matrix.columns.tolist(),
                    y=matrix.index.tolist(),
                    text=matrix.round(1).values,
                    texttemplate="%{text}",
                    colorscale="YlOrRd",
                    showscale=False,
                )
            )
            fig.update_layout(height=500, yaxis=dict(autorange="reversed"),
                              margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Language Distribution")
        lang_dist = df["lang_label"].value_counts()
        fig = px.pie(
            names=lang_dist.index,
            values=lang_dist.values,
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_traces(textinfo="label+percent")
        fig.update_layout(height=500, showlegend=False,
                          margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # — Row 2: Longevity Distribution + Trending Table ————
    c3, c4 = st.columns([2, 3])

    with c3:
        st.subheader("Top 10 Longevity Distribution")
        fig = px.histogram(
            longevity,
            nbins=20,
            labels={"value": "Weeks in Top 10", "count": "Titles"},
            color_discrete_sequence=["#FF6B6B" if region == "Korea" else "#4ECDC4"],
        )
        fig.update_layout(
            showlegend=False, bargap=0.1,
            xaxis_title="Weeks in Top 10", yaxis_title="Number of Titles",
            margin=dict(l=0, r=0, t=10, b=0), height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.subheader("Top Trending Titles")
        top_titles = (
            df.sort_values("cumulative_weeks_in_top_10", ascending=False)
            .drop_duplicates("show_title")
            .head(20)[["show_title", "category", "genres", "cumulative_weeks_in_top_10",
                        "tmdb_rating", "lang_label"]]
            .rename(columns={
                "show_title": "Title",
                "category": "Category",
                "genres": "Genres",
                "cumulative_weeks_in_top_10": "Weeks in Top 10",
                "tmdb_rating": "TMDB Rating",
                "lang_label": "Language",
            })
        )
        st.dataframe(top_titles, use_container_width=True, hide_index=True, height=400)


# ──────────────────────────────────────────────────────────────
# Tab 2 — Cross-Border Travel Tool
# ──────────────────────────────────────────────────────────────
def render_tab2(kr, jp):
    st.header("Cross-Border Travel Tool")
    st.caption("Discover content that succeeds in both Korea and Japan simultaneously.")

    kr_titles = set(kr["show_title"].unique())
    jp_titles = set(jp["show_title"].unique())
    shared = kr_titles & jp_titles

    # — Controls ————————————————————————————————————————————
    c1, c2, c3 = st.columns(3)
    with c1:
        genre_options = all_genres(kr, jp)
        genre_filter = st.multiselect("Filter by Genre", genre_options, key="t2_genre")
    with c2:
        min_weeks = st.slider("Min Combined Weeks in Top 10", 1, 40, 1, key="t2_weeks")
    with c3:
        cat_filter = st.radio("Category", ["All", "Films", "TV"], horizontal=True, key="t2_cat")

    # — Build shared longevity table ————————————————————————
    kr_long = title_longevity(kr[kr["show_title"].isin(shared)])
    jp_long = title_longevity(jp[jp["show_title"].isin(shared)])
    longevity = pd.DataFrame({"Korea_weeks": kr_long, "Japan_weeks": jp_long}).dropna()
    longevity["Total_weeks"] = longevity["Korea_weeks"] + longevity["Japan_weeks"]

    meta = (
        kr[kr["show_title"].isin(shared)]
        [["show_title", "original_language", "genres", "tmdb_rating",
          "category", "num_episodes", "avg_cast_popularity", "lang_label"]]
        .drop_duplicates("show_title")
        .set_index("show_title")
    )
    longevity = longevity.join(meta)

    # Apply filters
    if genre_filter:
        mask = longevity["genres"].apply(
            lambda g: any(gf in str(g) for gf in genre_filter) if pd.notna(g) else False
        )
        longevity = longevity[mask]
    if cat_filter != "All":
        longevity = longevity[longevity["category"] == cat_filter]
    longevity = longevity[longevity["Total_weeks"] >= min_weeks]

    # — Metrics ————————————————————————————————————————————
    m1, m2, m3 = st.columns(3)
    m1.metric("Shared Titles", f"{len(shared):,}")
    m2.metric("Korea-Only Titles", f"{len(kr_titles - shared):,}")
    m3.metric("Japan-Only Titles", f"{len(jp_titles - shared):,}")

    if longevity.empty:
        st.info("No titles match the current filters. Try relaxing your criteria.")
        return

    # — Scatter plot ————————————————————————————————————————
    st.subheader("Performance Comparison: Korea vs Japan")
    scatter_df = longevity.reset_index()
    fig = px.scatter(
        scatter_df,
        x="Korea_weeks",
        y="Japan_weeks",
        color="lang_label",
        size="tmdb_rating",
        hover_name="show_title",
        hover_data={"genres": True, "category": True, "Korea_weeks": True,
                    "Japan_weeks": True, "tmdb_rating": ":.1f", "lang_label": False},
        color_discrete_map={
            "Korean": "#FF6B6B", "English": "#4ECDC4",
            "Japanese": "#9B59B6", "Chinese": "#F1C40F", "Other": "#95A5A6",
        },
        labels={"Korea_weeks": "Weeks in Korea Top 10",
                "Japan_weeks": "Weeks in Japan Top 10",
                "lang_label": "Language"},
    )
    max_val = max(scatter_df[["Korea_weeks", "Japan_weeks"]].max().max(), 5) + 2
    fig.add_shape(
        type="line", x0=0, y0=0, x1=max_val, y1=max_val,
        line=dict(color="gray", dash="dash", width=1),
    )
    fig.add_annotation(
        x=max_val * 0.85, y=max_val * 0.65,
        text="Korea<br>outperforms", showarrow=False,
        font=dict(size=10, color="gray"),
    )
    fig.add_annotation(
        x=max_val * 0.35, y=max_val * 0.85,
        text="Japan<br>outperforms", showarrow=False,
        font=dict(size=10, color="gray"),
    )
    fig.update_layout(height=550, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # — Genre profile & Table ———————————————————————————————
    g1, g2 = st.columns([2, 3])

    with g1:
        st.subheader("Genre Profile of Shared Titles")
        shared_genres = explode_genres(
            longevity.reset_index().rename(columns={"show_title": "show_title_idx"})
            .assign(show_title=longevity.index)
        )
        if not shared_genres.empty:
            genre_counts = shared_genres["genre"].value_counts().head(12)
            fig = px.bar(
                x=genre_counts.values,
                y=genre_counts.index,
                orientation="h",
                labels={"x": "Count", "y": "Genre"},
                color_discrete_sequence=["#9B59B6"],
            )
            fig.update_layout(yaxis=dict(autorange="reversed"), height=400,
                              margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)

    with g2:
        st.subheader("Shared Titles Detail")
        display = (
            longevity.reset_index()
            .sort_values("Total_weeks", ascending=False)
            [["show_title", "Korea_weeks", "Japan_weeks", "Total_weeks",
              "genres", "category", "lang_label"]]
            .rename(columns={
                "show_title": "Title",
                "Korea_weeks": "KR Weeks",
                "Japan_weeks": "JP Weeks",
                "Total_weeks": "Total",
                "genres": "Genres",
                "category": "Category",
                "lang_label": "Language",
            })
        )
        st.dataframe(display, use_container_width=True, hide_index=True, height=400)


# ──────────────────────────────────────────────────────────────
# Tab 3 — Content Acquisition Simulator
# ──────────────────────────────────────────────────────────────

# Scoring weights derived from logistic regression coefficients
# (kr_logistic_coefficients.csv) and random forest feature importances
_GENRE_COEFS = {
    "Action": 0.384, "Thriller": 0.199, "Drama": 0.122,
    "Comedy": 0.120, "Horror": 0.037, "Mystery": 0.024,
    "Crime": 0.006, "Romance": -0.248,
    "Animation": 0.0, "Adventure": 0.0, "Fantasy": 0.0,
    "Science Fiction": 0.0, "Documentary": -0.10,
    "Family": 0.0, "War": 0.0,
}

_LANG_SCORES_JP = {"Korean": 6, "Japanese": 10, "English": 3, "Chinese": 0, "Other": -2}
_LANG_SCORES_KR = {"Korean": 12, "English": 3, "Japanese": -1, "Chinese": 0, "Other": -3}


def calculate_success_probability(
    target_market, category, tmdb_rating, episode_count,
    cast_popularity, language, genres, is_sequel, days_since_release,
):
    score = 50.0
    contribs = {}

    # 1. TMDB Rating  (LR coeff: 0.049, RF rank: 3)
    r = (tmdb_rating - 7.0) * 5.0
    r = max(-15, min(15, r))
    score += r
    contribs["TMDB Rating"] = r

    # 2. Episode count (TV only)
    if category == "TV":
        if 8 <= episode_count <= 16:
            e = 6.0
        elif 17 <= episode_count <= 24:
            e = 2.0
        elif episode_count <= 7:
            e = -2.0
        else:
            e = -5.0
        score += e
        contribs["Episode Count"] = e

    # 3. Cast popularity
    c = (cast_popularity - 2.5) * 2.0
    c = max(-8, min(10, c))
    score += c
    contribs["Cast Popularity"] = c

    # 4. Language fit (LR coeff: 0.344 for is_korean)
    lang_map = _LANG_SCORES_JP if target_market == "Japan" else _LANG_SCORES_KR
    l = lang_map.get(language, 0)
    score += l
    contribs["Language Fit"] = l

    # 5. Genre mix (from LR coefficients)
    g = sum(_GENRE_COEFS.get(gn, 0) for gn in genres) * 10
    g = max(-10, min(12, g))
    score += g
    contribs["Genre Mix"] = g

    # 6. Category (LR coeff: -0.480 for is_film)
    if category == "Films":
        ct = -5.0
    else:
        ct = 3.0
    score += ct
    contribs["Category"] = ct

    # 7. Sequel (LR coeff: +0.249)
    sq = 4.0 if is_sequel else 0.0
    score += sq
    contribs["Sequel Bonus"] = sq

    # 8. Recency (LR coeff: -0.189 for log_days)
    if days_since_release <= 14:
        rc = 5.0
    elif days_since_release <= 30:
        rc = 3.0
    elif days_since_release <= 90:
        rc = 0.0
    elif days_since_release <= 180:
        rc = -3.0
    else:
        rc = -6.0
    score += rc
    contribs["Recency"] = rc

    score = max(0, min(100, score))
    return score, contribs


def generate_recommendations(score, contribs, inputs):
    recs = []

    if inputs["tmdb_rating"] < 7.0:
        recs.append(("warning", "Quality Concern",
                      f"TMDB rating of {inputs['tmdb_rating']:.1f} is below the 7.0 threshold. "
                      f"Korean dramas achieving 5+ weeks in Japan's Top 10 average 7.8+."))
    elif inputs["tmdb_rating"] >= 8.0:
        recs.append(("success", "Strong Quality Signal",
                      f"Rating of {inputs['tmdb_rating']:.1f} places this in the top tier "
                      f"(odds ratio 1.05x per rating point)."))

    if inputs["category"] == "TV" and inputs["episode_count"] > 24:
        recs.append(("warning", "Episode Count Risk",
                      f"{inputs['episode_count']} episodes exceeds the optimal 9-16 range. "
                      f"Longer series show lower cross-border success rates."))
    elif inputs["category"] == "TV" and 8 <= inputs["episode_count"] <= 16:
        recs.append(("success", "Optimal Episode Count",
                      "9-16 episodes is the sweet spot for Korean dramas in Japan."))

    if inputs["target_market"] == "Japan" and inputs["language"] == "Korean":
        recs.append(("info", "Korean Content in Japan",
                      "Korean content represents ~11% of Japan's Top 10 entries. "
                      "Drama and Romance genres travel best. Star power is a key differentiator."))

    if inputs["cast_popularity"] < 2.0:
        recs.append(("warning", "Low Star Power",
                      "Cast popularity below 2.0 is below the dataset average. "
                      "Content reaching Japan averages ~30% higher cast popularity."))
    elif inputs["cast_popularity"] >= 5.0:
        recs.append(("success", "High Star Power",
                      "Strong cast popularity boosts discoverability and cross-border appeal."))

    if inputs["days_since_release"] > 180:
        recs.append(("info", "Catalog Title",
                      "Titles older than 180 days have lower odds of re-entering the Top 10. "
                      "Consider pairing with a marketing event or sequel announcement."))

    if score >= 70:
        recs.append(("success", "Strong Acquisition Candidate",
                      "This content profile aligns well with historically successful titles "
                      f"in the {inputs['target_market']} market."))
    elif score < 35:
        recs.append(("error", "High Risk Profile",
                      "Multiple factors score below average. Consider adjusting content "
                      "characteristics or targeting a different market."))

    return recs


def render_tab3(kr, jp):
    st.header("Content Acquisition Simulator")
    st.caption(
        "Adjust content characteristics to estimate Top 10 success probability. "
        "Scoring is derived from logistic regression on historical Netflix data."
    )

    col_in, col_out = st.columns([1, 2])

    with col_in:
        st.subheader("Content Characteristics")
        target_market = st.selectbox("Target Market", ["Japan", "Korea"], key="t3_market")
        category = st.selectbox("Category", ["TV", "Films"], key="t3_cat")
        tmdb_rating = st.slider("TMDB Rating", 1.0, 10.0, 7.0, 0.1, key="t3_rating")

        ep_disabled = category == "Films"
        episode_count = st.slider(
            "Episode Count", 1, 50, 12, 1,
            key="t3_ep", disabled=ep_disabled,
            help="Only applicable for TV content",
        )

        cast_popularity = st.slider("Avg Cast Popularity", 0.0, 15.0, 3.0, 0.1, key="t3_cast")
        language = st.selectbox(
            "Original Language",
            ["Korean", "English", "Japanese", "Chinese", "Other"],
            key="t3_lang",
        )

        genre_options = sorted(_GENRE_COEFS.keys())
        selected_genres = st.multiselect("Genres", genre_options, default=["Drama"], key="t3_genre")
        is_sequel = st.checkbox("Sequel / follow-up season", key="t3_sequel")
        days_since_release = st.slider("Days Since Release", 0, 365, 14, key="t3_days")

    # — Compute score ————————————————————————————————————————
    score, contribs = calculate_success_probability(
        target_market, category, tmdb_rating, episode_count,
        cast_popularity, language, selected_genres, is_sequel,
        days_since_release,
    )

    with col_out:
        # — Gauge ——————————————————————————————————————————
        st.subheader("Success Probability")

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=score,
                number={"suffix": "%", "font": {"size": 48}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": "#333"},
                    "steps": [
                        {"range": [0, 35], "color": "#FF6B6B"},
                        {"range": [35, 65], "color": "#F9E07F"},
                        {"range": [65, 100], "color": "#6BCB77"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.8,
                        "value": score,
                    },
                },
            )
        )
        fig.update_layout(height=280, margin=dict(l=30, r=30, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

        # — Factor contributions ——————————————————————————
        st.subheader("Factor Contributions")
        contrib_df = pd.DataFrame(
            {"Factor": list(contribs.keys()), "Contribution": list(contribs.values())}
        ).sort_values("Contribution")

        colors = ["#6BCB77" if v >= 0 else "#FF6B6B" for v in contrib_df["Contribution"]]
        fig = go.Figure(
            go.Bar(
                x=contrib_df["Contribution"],
                y=contrib_df["Factor"],
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.1f}" for v in contrib_df["Contribution"]],
                textposition="outside",
            )
        )
        fig.update_layout(
            height=max(250, len(contribs) * 38),
            xaxis_title="Points (base = 50)",
            yaxis_title="",
            margin=dict(l=0, r=40, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        # — Recommendations ———————————————————————————————
        st.subheader("Recommendations")
        inputs = {
            "target_market": target_market,
            "category": category,
            "tmdb_rating": tmdb_rating,
            "episode_count": episode_count,
            "cast_popularity": cast_popularity,
            "language": language,
            "genres": selected_genres,
            "is_sequel": is_sequel,
            "days_since_release": days_since_release,
        }
        recs = generate_recommendations(score, contribs, inputs)
        for rtype, title, text in recs:
            if rtype == "success":
                st.success(f"**{title}:** {text}")
            elif rtype == "warning":
                st.warning(f"**{title}:** {text}")
            elif rtype == "error":
                st.error(f"**{title}:** {text}")
            else:
                st.info(f"**{title}:** {text}")


# ──────────────────────────────────────────────────────────────
# Tab 4 — Korean Content Feature Analysis
# ──────────────────────────────────────────────────────────────
def render_tab4(kr, jp):
    st.header("Korean Content Feature Analysis")
    st.caption("Deep dive into characteristics of Korean content that succeeds in Japan's Top 10.")

    # Filter Korean content
    korean_content_jp = jp[(jp["original_language"] == "ko") & (jp["tmdb_id"].notna())].copy()
    korean_content_kr = kr[(kr["original_language"] == "ko") & (kr["tmdb_id"].notna())].copy()

    kr_korean_titles = set(korean_content_kr["show_title"].unique())
    jp_korean_titles = set(korean_content_jp["show_title"].unique())
    korean_shared = kr_korean_titles & jp_korean_titles
    korean_kr_only = kr_korean_titles - jp_korean_titles

    # — KPI Row ————————————————————————————————————————————
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Korean Titles in Japan", len(jp_korean_titles))
    k2.metric("Korean Titles in Both Markets", len(korean_shared))
    k3.metric("Korea-Only Korean Titles", len(korean_kr_only))
    
    kr_pct_in_jp = len(korean_content_jp) / len(jp) * 100 if len(jp) > 0 else 0
    k4.metric("% of Japan Top 10", f"{kr_pct_in_jp:.1f}%")

    st.divider()

    # — Sub-tabs for different analyses ————————————————————
    subtab1, subtab2 = st.tabs(["📊 Japan Penetration Analysis", "🏆 5+ Weeks Success Analysis"])

    # ══════════════════════════════════════════════════════
    # Sub-tab 1: Korean Content in Japan vs Korea-Only
    # ══════════════════════════════════════════════════════
    with subtab1:
        st.subheader("Korean Content: Japan vs Korea-Only Comparison")

        korean_kr_only_data = korean_content_kr[korean_content_kr["show_title"].isin(korean_kr_only)].copy()

        # Genre comparison
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Genre Distribution")
            # Korean content in Japan
            jp_genres = explode_genres(korean_content_jp)
            kr_only_genres = explode_genres(korean_kr_only_data)

            if not jp_genres.empty and not kr_only_genres.empty:
                jp_genre_dist = jp_genres["genre"].value_counts(normalize=True) * 100
                kr_genre_dist = kr_only_genres["genre"].value_counts(normalize=True) * 100

                genre_comp = pd.DataFrame({
                    "Reached Japan": jp_genre_dist,
                    "Korea Only": kr_genre_dist
                }).fillna(0).head(12)

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=genre_comp.index,
                    x=genre_comp["Reached Japan"],
                    name="Reached Japan",
                    orientation="h",
                    marker_color="#FF6B6B"
                ))
                fig.add_trace(go.Bar(
                    y=genre_comp.index,
                    x=genre_comp["Korea Only"],
                    name="Korea Only",
                    orientation="h",
                    marker_color="#95A5A6"
                ))
                fig.update_layout(
                    barmode="group",
                    height=400,
                    xaxis_title="% of Entries",
                    yaxis=dict(autorange="reversed"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Category Distribution")
            jp_cat = korean_content_jp["category"].value_counts(normalize=True) * 100
            kr_cat = korean_kr_only_data["category"].value_counts(normalize=True) * 100

            cat_comp = pd.DataFrame({
                "Reached Japan": jp_cat,
                "Korea Only": kr_cat
            }).fillna(0)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=cat_comp.index,
                y=cat_comp["Reached Japan"],
                name="Reached Japan",
                marker_color="#FF6B6B"
            ))
            fig.add_trace(go.Bar(
                x=cat_comp.index,
                y=cat_comp["Korea Only"],
                name="Korea Only",
                marker_color="#95A5A6"
            ))
            fig.update_layout(
                barmode="group",
                height=400,
                yaxis_title="% of Entries",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)

        # Quality & Cast metrics
        col3, col4 = st.columns(2)

        with col3:
            st.markdown("#### Quality Metrics (TMDB Rating)")
            jp_meta = korean_content_jp.drop_duplicates("show_title")
            kr_only_meta = korean_kr_only_data.drop_duplicates("show_title")

            jp_rated = jp_meta[(jp_meta["tmdb_rating"].notna()) & (jp_meta["tmdb_rating"] > 0)]
            kr_rated = kr_only_meta[(kr_only_meta["tmdb_rating"].notna()) & (kr_only_meta["tmdb_rating"] > 0)]

            if len(jp_rated) > 0 and len(kr_rated) > 0:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=jp_rated["tmdb_rating"],
                    name=f"Reached Japan (avg: {jp_rated['tmdb_rating'].mean():.2f})",
                    marker_color="#FF6B6B",
                    opacity=0.7,
                    histnorm="probability density"
                ))
                fig.add_trace(go.Histogram(
                    x=kr_rated["tmdb_rating"],
                    name=f"Korea Only (avg: {kr_rated['tmdb_rating'].mean():.2f})",
                    marker_color="#95A5A6",
                    opacity=0.7,
                    histnorm="probability density"
                ))
                fig.update_layout(
                    barmode="overlay",
                    height=350,
                    xaxis_title="TMDB Rating",
                    yaxis_title="Density",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)

                # Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Avg Rating (Japan)", f"{jp_rated['tmdb_rating'].mean():.2f}")
                m2.metric("Avg Rating (Korea Only)", f"{kr_rated['tmdb_rating'].mean():.2f}")
                m3.metric("Difference", f"{jp_rated['tmdb_rating'].mean() - kr_rated['tmdb_rating'].mean():+.2f}")

        with col4:
            st.markdown("#### Casting Popularity")
            if "avg_cast_popularity" in jp_meta.columns:
                jp_cast = jp_meta["avg_cast_popularity"]
                jp_cast = jp_cast[(jp_cast.notna()) & (jp_cast > 0)]

                kr_cast = kr_only_meta["avg_cast_popularity"]
                kr_cast = kr_cast[(kr_cast.notna()) & (kr_cast > 0)]

                if len(jp_cast) > 0 and len(kr_cast) > 0:
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=jp_cast,
                        name=f"Reached Japan (avg: {jp_cast.mean():.1f})",
                        marker_color="#FF6B6B",
                        opacity=0.7,
                        histnorm="probability density"
                    ))
                    fig.add_trace(go.Histogram(
                        x=kr_cast,
                        name=f"Korea Only (avg: {kr_cast.mean():.1f})",
                        marker_color="#95A5A6",
                        opacity=0.7,
                        histnorm="probability density"
                    ))
                    fig.update_layout(
                        barmode="overlay",
                        height=350,
                        xaxis_title="Avg Cast Popularity",
                        yaxis_title="Density",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Avg Cast Pop (Japan)", f"{jp_cast.mean():.1f}")
                    m2.metric("Avg Cast Pop (Korea Only)", f"{kr_cast.mean():.1f}")
                    diff_pct = ((jp_cast.mean() / kr_cast.mean()) - 1) * 100 if kr_cast.mean() > 0 else 0
                    m3.metric("Difference", f"{diff_pct:+.1f}%")

        # Top Korean titles in Japan
        st.markdown("#### Top Korean Titles in Japan")
        top_kr_jp = korean_content_jp.groupby("show_title").agg({
            "cumulative_weeks_in_top_10": "max",
            "genres": "first",
            "category": "first",
            "tmdb_rating": "first"
        }).sort_values("cumulative_weeks_in_top_10", ascending=False).head(15)

        fig = px.bar(
            x=top_kr_jp["cumulative_weeks_in_top_10"],
            y=top_kr_jp.index,
            orientation="h",
            labels={"x": "Weeks in Japan Top 10", "y": "Title"},
            color_discrete_sequence=["#FF6B6B"]
        )
        fig.update_layout(
            height=450,
            yaxis=dict(autorange="reversed"),
            margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════
    # Sub-tab 2: 5+ Weeks Success Analysis
    # ══════════════════════════════════════════════════════
    with subtab2:
        st.subheader("Korean Dramas: 5+ Weeks vs ≤5 Weeks in Japan Top 10")

        # Filter Korean TV content
        kr_tv_jp = korean_content_jp[korean_content_jp["category"] == "TV"].copy()
        kr_tv_longevity = kr_tv_jp.groupby("show_title")["cumulative_weeks_in_top_10"].max()

        # Split by success threshold
        weeks_threshold = st.slider("Success Threshold (weeks)", 3, 10, 5, key="t4_threshold")

        successful_titles = kr_tv_longevity[kr_tv_longevity > weeks_threshold].index
        less_successful_titles = kr_tv_longevity[kr_tv_longevity <= weeks_threshold].index

        kr_tv_success = kr_tv_jp[kr_tv_jp["show_title"].isin(successful_titles)].drop_duplicates("show_title")
        kr_tv_less = kr_tv_jp[kr_tv_jp["show_title"].isin(less_successful_titles)].drop_duplicates("show_title")

        # KPIs
        k1, k2, k3 = st.columns(3)
        k1.metric(f">{weeks_threshold} Weeks Success", len(kr_tv_success))
        k2.metric(f"≤{weeks_threshold} Weeks", len(kr_tv_less))
        success_rate = len(kr_tv_success) / (len(kr_tv_success) + len(kr_tv_less)) * 100 if (len(kr_tv_success) + len(kr_tv_less)) > 0 else 0
        k3.metric("Success Rate", f"{success_rate:.1f}%")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"#### Genre Distribution: >{weeks_threshold}w vs ≤{weeks_threshold}w")
            success_genres = explode_genres(kr_tv_success)
            less_genres = explode_genres(kr_tv_less)

            if not success_genres.empty and not less_genres.empty:
                success_dist = success_genres["genre"].value_counts(normalize=True) * 100
                less_dist = less_genres["genre"].value_counts(normalize=True) * 100

                genre_comp = pd.DataFrame({
                    f">{weeks_threshold} Weeks": success_dist,
                    f"≤{weeks_threshold} Weeks": less_dist
                }).fillna(0)
                genre_comp["Difference"] = genre_comp[f">{weeks_threshold} Weeks"] - genre_comp[f"≤{weeks_threshold} Weeks"]
                genre_comp = genre_comp.sort_values("Difference", ascending=False).head(10)

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=genre_comp.index,
                    x=genre_comp[f">{weeks_threshold} Weeks"],
                    name=f">{weeks_threshold} Weeks",
                    orientation="h",
                    marker_color="#FF6B6B"
                ))
                fig.add_trace(go.Bar(
                    y=genre_comp.index,
                    x=genre_comp[f"≤{weeks_threshold} Weeks"],
                    name=f"≤{weeks_threshold} Weeks",
                    orientation="h",
                    marker_color="#95A5A6"
                ))
                fig.update_layout(
                    barmode="group",
                    height=400,
                    xaxis_title="% of Entries",
                    yaxis=dict(autorange="reversed"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Rating Distribution")
            success_rated = kr_tv_success[(kr_tv_success["tmdb_rating"].notna()) & (kr_tv_success["tmdb_rating"] > 0)]
            less_rated = kr_tv_less[(kr_tv_less["tmdb_rating"].notna()) & (kr_tv_less["tmdb_rating"] > 0)]

            if len(success_rated) > 0 and len(less_rated) > 0:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=success_rated["tmdb_rating"],
                    name=f">{weeks_threshold}w (avg: {success_rated['tmdb_rating'].mean():.2f})",
                    marker_color="#FF6B6B",
                    opacity=0.7,
                    histnorm="probability density"
                ))
                fig.add_trace(go.Histogram(
                    x=less_rated["tmdb_rating"],
                    name=f"≤{weeks_threshold}w (avg: {less_rated['tmdb_rating'].mean():.2f})",
                    marker_color="#95A5A6",
                    opacity=0.7,
                    histnorm="probability density"
                ))
                fig.update_layout(
                    barmode="overlay",
                    height=400,
                    xaxis_title="TMDB Rating",
                    yaxis_title="Density",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)

        # Episode count & Cast popularity
        col3, col4 = st.columns(2)

        with col3:
            st.markdown("#### Episode Count")
            if "num_episodes" in kr_tv_success.columns:
                success_eps = kr_tv_success["num_episodes"].dropna()
                less_eps = kr_tv_less["num_episodes"].dropna()

                if len(success_eps) > 0 and len(less_eps) > 0:
                    # IQR filtering
                    all_eps = pd.concat([success_eps, less_eps])
                    q1, q3 = all_eps.quantile(0.25), all_eps.quantile(0.75)
                    iqr = q3 - q1
                    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

                    success_eps_f = success_eps[(success_eps >= lower) & (success_eps <= upper)]
                    less_eps_f = less_eps[(less_eps >= lower) & (less_eps <= upper)]

                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=success_eps_f,
                        name=f">{weeks_threshold}w (avg: {success_eps_f.mean():.1f})",
                        marker_color="#FF6B6B",
                        opacity=0.7,
                        histnorm="probability density"
                    ))
                    fig.add_trace(go.Histogram(
                        x=less_eps_f,
                        name=f"≤{weeks_threshold}w (avg: {less_eps_f.mean():.1f})",
                        marker_color="#95A5A6",
                        opacity=0.7,
                        histnorm="probability density"
                    ))
                    fig.update_layout(
                        barmode="overlay",
                        height=350,
                        xaxis_title="Number of Episodes",
                        yaxis_title="Density",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    st.plotly_chart(fig, use_container_width=True)

        with col4:
            st.markdown("#### Casting Popularity")
            if "avg_cast_popularity" in kr_tv_success.columns:
                success_cast = kr_tv_success["avg_cast_popularity"]
                success_cast = success_cast[(success_cast.notna()) & (success_cast > 0)]

                less_cast = kr_tv_less["avg_cast_popularity"]
                less_cast = less_cast[(less_cast.notna()) & (less_cast > 0)]

                if len(success_cast) > 0 and len(less_cast) > 0:
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=success_cast,
                        name=f">{weeks_threshold}w (avg: {success_cast.mean():.1f})",
                        marker_color="#FF6B6B",
                        opacity=0.7,
                        histnorm="probability density"
                    ))
                    fig.add_trace(go.Histogram(
                        x=less_cast,
                        name=f"≤{weeks_threshold}w (avg: {less_cast.mean():.1f})",
                        marker_color="#95A5A6",
                        opacity=0.7,
                        histnorm="probability density"
                    ))
                    fig.update_layout(
                        barmode="overlay",
                        height=350,
                        xaxis_title="Avg Cast Popularity",
                        yaxis_title="Density",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # Sequel analysis
        col5, col6 = st.columns(2)

        with col5:
            st.markdown("#### Sequel vs Original")
            if "is_sequel" in kr_tv_success.columns:
                success_sequel = kr_tv_success["is_sequel"].mean() * 100
                less_sequel = kr_tv_less["is_sequel"].mean() * 100

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=["Original", "Sequel"],
                    y=[100 - success_sequel, success_sequel],
                    name=f">{weeks_threshold} Weeks",
                    marker_color="#FF6B6B"
                ))
                fig.add_trace(go.Bar(
                    x=["Original", "Sequel"],
                    y=[100 - less_sequel, less_sequel],
                    name=f"≤{weeks_threshold} Weeks",
                    marker_color="#95A5A6"
                ))
                fig.update_layout(
                    barmode="group",
                    height=350,
                    yaxis_title="% of Content",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)

        with col6:
            st.markdown(f"#### Top Titles (>{weeks_threshold} Weeks)")
            kr_tv_success["weeks_in_top10"] = kr_tv_success["show_title"].map(kr_tv_longevity)
            top_success = kr_tv_success.sort_values("weeks_in_top10", ascending=False).head(10)

            fig = px.bar(
                x=top_success["weeks_in_top10"],
                y=top_success["show_title"],
                orientation="h",
                labels={"x": "Weeks in Japan Top 10", "y": "Title"},
                color_discrete_sequence=["#FF6B6B"]
            )
            fig.add_vline(x=weeks_threshold, line_dash="dash", line_color="gray",
                          annotation_text=f"{weeks_threshold}w threshold")
            fig.update_layout(
                height=350,
                yaxis=dict(autorange="reversed"),
                margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)

        # Summary insights
        st.divider()
        st.markdown("### Key Insights")

        insights = []
        if len(success_rated) > 0 and len(less_rated) > 0:
            rating_diff = success_rated['tmdb_rating'].mean() - less_rated['tmdb_rating'].mean()
            insights.append(f"**Quality:** Successful titles have {rating_diff:+.2f} higher avg TMDB rating")

        if "avg_cast_popularity" in kr_tv_success.columns and len(success_cast) > 0 and len(less_cast) > 0:
            cast_diff_pct = ((success_cast.mean() / less_cast.mean()) - 1) * 100
            insights.append(f"**Star Power:** Successful titles have {cast_diff_pct:+.1f}% higher casting popularity")

        if "num_episodes" in kr_tv_success.columns and len(success_eps) > 0 and len(less_eps) > 0:
            ep_diff = success_eps.mean() - less_eps.mean()
            insights.append(f"**Episode Count:** Successful titles average {ep_diff:+.1f} episodes difference")

        for insight in insights:
            st.info(insight)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    kr, jp, min_d, max_d = load_data()

    # Sidebar
    with st.sidebar:
        st.title("Netflix KR vs JP")
        st.markdown(f"**Data period:** {min_d.date()} to {max_d.date()}")
        st.metric("Korea Titles", kr["show_title"].nunique())
        st.metric("Japan Titles", jp["show_title"].nunique())
        st.divider()
        st.caption(
            "Dashboard derived from Netflix Top 10 weekly data "
            "enriched with TMDB metadata."
        )

    st.title("Netflix Content Analysis: Korea vs Japan")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Regional Explorer",
        "🌏 Cross-Border Travel Tool",
        "🎯 Content Acquisition Simulator",
        "🇰🇷 Korean Content Analysis",
    ])

    with tab1:
        render_tab1(kr, jp)
    with tab2:
        render_tab2(kr, jp)
    with tab3:
        render_tab3(kr, jp)
    with tab4:
        render_tab4(kr, jp)


if __name__ == "__main__":
    main()
