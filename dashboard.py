
from pathlib import Path
import json
import re
import ast

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data" / "ml-latest-small"
OUTPUT_DIR = ROOT / "outputs"

MOVIES_CSV = DATA_DIR / "movies.csv"
RATINGS_CSV = DATA_DIR / "ratings.csv"

GOLD_DIR = OUTPUT_DIR / "gold"
MODEL_DIR = OUTPUT_DIR / "model"

USER_RECOMMENDATIONS_DIR = MODEL_DIR / "user_recommendations"
ITEM_RECOMMENDATIONS_DIR = MODEL_DIR / "item_recommendations"
METRICS_DIR = MODEL_DIR / "metrics"


# ============================================================
# 2. STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Movie Recommendation System using Big Data Analytics",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 3. CUSTOM DARK THEME
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at top right, #182344 0%, transparent 32%),
            linear-gradient(135deg, #080d1b 0%, #0b1225 55%, #10182e 100%);
        color: #f3f6ff;
    }

    [data-testid="stSidebar"] {
        background: #090f20;
        border-right: 1px solid #202b47;
    }

    [data-testid="stSidebar"] * {
        color: #e8edff;
    }

    .main-title {
        font-size: 38px;
        font-weight: 800;
        letter-spacing: -1.2px;
        color: #ffffff;
        margin-bottom: 4px;
    }

    .main-subtitle {
        color: #9daaca;
        font-size: 15px;
        margin-bottom: 26px;
    }

    .hero-card {
        padding: 30px;
        border-radius: 20px;
        border: 1px solid #28385c;
        background: linear-gradient(125deg, #162443, #111b33);
        margin-bottom: 25px;
    }

    .hero-card h2 {
        color: #ffffff;
        font-size: 30px;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .hero-card p {
        color: #bac7e5;
        font-size: 15px;
        line-height: 1.7;
    }

    .metric-card {
        background: linear-gradient(145deg, #141f39, #10182d);
        border: 1px solid #293858;
        padding: 20px;
        border-radius: 16px;
        min-height: 125px;
    }

    .metric-label {
        font-size: 13px;
        color: #a5b3d3;
        margin-bottom: 12px;
    }

    .metric-value {
        font-size: 29px;
        font-weight: 800;
        color: #ffffff;
    }

    .metric-note {
        color: #7dd3fc;
        font-size: 12px;
        margin-top: 6px;
    }

    .section-title {
        color: #ffffff;
        font-size: 23px;
        font-weight: 750;
        margin-top: 18px;
        margin-bottom: 15px;
    }

    .info-card {
        background: #111a30;
        border: 1px solid #283653;
        border-radius: 15px;
        padding: 20px;
        min-height: 135px;
    }

    .info-card h4 {
        color: #7dd3fc;
        margin-bottom: 8px;
    }

    .info-card p {
        color: #bdc8e1;
        font-size: 14px;
        line-height: 1.65;
    }

    .movie-card {
        background: #121d35;
        border: 1px solid #2a3959;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 10px;
    }

    .movie-card h4 {
        color: #ffffff;
        font-size: 16px;
        margin-bottom: 7px;
    }

    .movie-card p {
        color: #aebbd8;
        font-size: 13px;
    }

    div[data-testid="stMetric"] {
        background: #121c33;
        border: 1px solid #2a3959;
        padding: 16px;
        border-radius: 14px;
    }

    div[data-testid="stMetricLabel"] {
        color: #a8b5d4;
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff;
    }

    .footer {
        text-align: center;
        color: #7888aa;
        font-size: 12px;
        padding: 25px 0 10px 0;
        border-top: 1px solid #26324e;
        margin-top: 40px;
    }

    .stButton > button {
        background: #38bdf8;
        color: #081225;
        border: none;
        border-radius: 9px;
        font-weight: 700;
    }

    .stButton > button:hover {
        background: #7dd3fc;
        color: #081225;
    }

    div[data-baseweb="select"] > div {
        background-color: #111b32;
        border-color: #344565;
    }

    .stTextInput input,
    .stNumberInput input {
        background-color: #111b32;
        color: #ffffff;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 4. GENERAL HELPER FUNCTIONS
# ============================================================

def format_number(value):
    """Format a number for dashboard metric cards."""
    try:
        if pd.isna(value):
            return "0"
        return f"{float(value):,.0f}"
    except (TypeError, ValueError):
        return str(value)


def format_decimal(value, digits=3):
    """Format decimal values safely."""
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def find_parquet_files(path):
    """Find Parquet files in a file or output directory."""
    path = Path(path)

    if not path.exists():
        return []

    if path.is_file() and path.suffix.lower() == ".parquet":
        return [path]

    return sorted(path.rglob("*.parquet"))


def read_parquet_folder(path):
    """
    Read Parquet files from a folder or a single Parquet file.
    Returns an empty DataFrame when no data is available.
    """
    files = find_parquet_files(path)

    if not files:
        return pd.DataFrame()

    frames = []

    for file_path in files:
        try:
            frames.append(pd.read_parquet(file_path))
        except Exception as error:
            st.warning(
                f"Could not read {file_path.name}: {error}"
            )

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def read_metrics():
    """Load model evaluation metrics saved by the training script."""
    possible_files = [
        METRICS_DIR / "part-00000.json",
        MODEL_DIR / "metrics.json",
        MODEL_DIR / "model_metrics.json",
    ]

    for file_path in possible_files:
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    return json.load(file)
            except (OSError, json.JSONDecodeError):
                continue

    # Also support a metrics folder containing JSON output.
    if METRICS_DIR.exists():
        for file_path in METRICS_DIR.rglob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    return json.load(file)
            except (OSError, json.JSONDecodeError):
                continue

    return {}


def get_metric(metrics, possible_names, default=None):
    """Find a metric even when its key uses different capitalization."""
    if not isinstance(metrics, dict):
        return default

    normalized = {
        str(key).lower().replace(" ", "_"): value
        for key, value in metrics.items()
    }

    for name in possible_names:
        key = name.lower().replace(" ", "_")
        if key in normalized:
            return normalized[key]

    return default


# ============================================================
# 5. LOAD MOVIES AND RATINGS
# ============================================================

@st.cache_data(show_spinner=False)
def load_movies():
    """Load movie information from the source CSV or Gold output."""
    possible_paths = [
        GOLD_DIR / "movies",
        GOLD_DIR / "movies.parquet",
    ]

    for path in possible_paths:
        df = read_parquet_folder(path)
        if not df.empty and "movieId" in df.columns:
            return df

    if MOVIES_CSV.exists():
        try:
            return pd.read_csv(MOVIES_CSV)
        except Exception:
            pass

    return pd.DataFrame(columns=["movieId", "title", "genres"])


@st.cache_data(show_spinner=False)
def load_ratings():
    """Load rating data from the source CSV or Gold output."""
    possible_paths = [
        GOLD_DIR / "ratings",
        GOLD_DIR / "ratings.parquet",
    ]

    for path in possible_paths:
        df = read_parquet_folder(path)
        if not df.empty and "rating" in df.columns:
            return df

    if RATINGS_CSV.exists():
        try:
            return pd.read_csv(RATINGS_CSV)
        except Exception:
            pass

    return pd.DataFrame(columns=["userId", "movieId", "rating"])


@st.cache_data(show_spinner=False)
def load_user_recommendations():
    """Load existing ALS user recommendations."""
    return read_parquet_folder(USER_RECOMMENDATIONS_DIR)


@st.cache_data(show_spinner=False)
def load_item_recommendations():
    """Load item-based recommendations if generated."""
    return read_parquet_folder(ITEM_RECOMMENDATIONS_DIR)


movies_df = load_movies()
ratings_df = load_ratings()
recs = load_user_recommendations()
item_recs = load_item_recommendations()
metrics = read_metrics()


# ============================================================
# 6. CLEAN DATA TYPES
# ============================================================

if not movies_df.empty:
    if "movieId" in movies_df.columns:
        movies_df["movieId"] = pd.to_numeric(
            movies_df["movieId"], errors="coerce"
        )

if not ratings_df.empty:
    for column in ["userId", "movieId", "rating"]:
        if column in ratings_df.columns:
            ratings_df[column] = pd.to_numeric(
                ratings_df[column], errors="coerce"
            )

if not recs.empty and "userId" in recs.columns:
    recs["userId"] = pd.to_numeric(
        recs["userId"], errors="coerce"
    )


# ============================================================
# 7. RECOMMENDATION PARSING FIX
# ============================================================

def parse_recommendations(raw_recommendations):
    """
    Convert saved ALS recommendations into ordinary Python records.

    Supports:
    - NumPy arrays
    - Lists of dictionaries
    - Lists of [movieId, predicted_rating]
    - Strings representing NumPy arrays, such as:
      array([[6818.0, 6.14], [33649.0, 5.98]], dtype=object)
    """
    if raw_recommendations is None:
        return []

    if isinstance(raw_recommendations, (float, np.floating)):
        if pd.isna(raw_recommendations):
            return []

    # Convert native NumPy arrays to lists.
    if isinstance(raw_recommendations, np.ndarray):
        raw_recommendations = raw_recommendations.tolist()

    # Parse a saved string.
    if isinstance(raw_recommendations, str):
        raw_text = raw_recommendations.strip()

        if not raw_text:
            return []

        # First try ordinary Python list/dictionary syntax.
        try:
            parsed = ast.literal_eval(raw_text)
            if isinstance(parsed, (list, tuple, np.ndarray)):
                raw_recommendations = parsed
        except (ValueError, SyntaxError):
            pass

        # If literal_eval could not parse the NumPy representation,
        # extract the numerical movieId/score pairs directly.
        if isinstance(raw_recommendations, str):
            number_pattern = (
                r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
                r"(?:[eE][-+]?\d+)?"
            )

            numbers = re.findall(number_pattern, raw_text)

            try:
                values = [float(number) for number in numbers]
            except ValueError:
                return []

            if len(values) < 2:
                return []

            # Every recommendation should have two values:
            # movieId and predicted rating.
            if len(values) % 2 != 0:
                values = values[:-1]

            raw_recommendations = [
                [values[index], values[index + 1]]
                for index in range(0, len(values), 2)
            ]

    if isinstance(raw_recommendations, np.ndarray):
        raw_recommendations = raw_recommendations.tolist()

    if not isinstance(raw_recommendations, (list, tuple)):
        return []

    records = []

    for item in raw_recommendations:
        movie_id = None
        score = None

        if isinstance(item, dict):
            movie_id = item.get(
                "movieId",
                item.get("movie_id")
            )
            score = item.get(
                "rating",
                item.get(
                    "prediction",
                    item.get("score")
                )
            )

        elif isinstance(item, (list, tuple, np.ndarray)):
            if len(item) >= 2:
                movie_id = item[0]
                score = item[1]

        else:
            continue

        try:
            movie_id_number = float(movie_id)
            score_number = float(score)

            if not np.isfinite(movie_id_number):
                continue

            if not np.isfinite(score_number):
                continue

            records.append({
                "movieId": int(movie_id_number),
                "Predicted score": score_number,
            })

        except (TypeError, ValueError, OverflowError):
            continue

    return records


def get_user_recommendations(user_rows, movies_df):
    """
    Convert one user's saved ALS recommendations to a DataFrame
    and attach movie titles and genres.
    """
    if user_rows.empty or "recommendations" not in user_rows.columns:
        return pd.DataFrame()

    raw_recommendations = user_rows.iloc[0]["recommendations"]

    records = parse_recommendations(raw_recommendations)

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records)

    # Attach movie titles and genres.
    if (
        movies_df is not None
        and not movies_df.empty
        and "movieId" in movies_df.columns
    ):
        columns_to_use = [
            column
            for column in ["movieId", "title", "genres"]
            if column in movies_df.columns
        ]

        movie_lookup = movies_df[columns_to_use].copy()

        movie_lookup["movieId"] = pd.to_numeric(
            movie_lookup["movieId"], errors="coerce"
        )

        movie_lookup = movie_lookup.dropna(subset=["movieId"])
        movie_lookup["movieId"] = movie_lookup["movieId"].astype(int)

        result = result.merge(
            movie_lookup.drop_duplicates(subset=["movieId"]),
            on="movieId",
            how="left",
        )

    if "title" not in result.columns:
        result["title"] = "Unknown movie"

    result["title"] = result["title"].fillna("Unknown movie")

    if "genres" in result.columns:
        result["genres"] = result["genres"].fillna("Unknown")

    return result.sort_values(
        "Predicted score",
        ascending=False,
    ).reset_index(drop=True)


# ============================================================
# 8. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 12px 0 24px 0;">
            <div style="font-size: 34px;">🎬</div>
            <div style="font-size: 25px; font-weight: 800;">
                Movie Recommendation System using Big Data Analytics
            </div>
            <div style="font-size: 12px; color: #8fa1c7;">
                MOVIE ANALYTICS PLATFORM
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Explore Movies",
            "Personalized Recommendations",
            "Model Performance",
        ],
        label_visibility="visible",
    )

    st.divider()

    st.caption("PROJECT TECHNOLOGY")
    st.markdown(
        """
        - Python
        - PySpark
        - Spark MLlib ALS
        - Streamlit
        - MovieLens dataset
        """
    )

    st.divider()

    st.caption("Data source: MovieLens")


# ============================================================
# 9. PAGE HEADER
# ============================================================

st.markdown(
    '<div class="main-title">Movie Recommendation System using Big Data Analytics</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-subtitle">'
    'Movie Recommendation & Big Data Analytics Dashboard'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# 10. OVERVIEW PAGE
# ============================================================

if page == "Overview":

    st.markdown(
        """
        <div class="hero-card">
            <h2>Discover your next favourite movie.</h2>
            <p>
                Movie Recommendation System using Big Data Analytics uses collaborative filtering and Spark MLlib ALS
                to generate personalized movie recommendations from
                MovieLens user-rating data. Explore the dataset, inspect
                recommendations, and evaluate model performance in one place.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    total_movies = (
        movies_df["movieId"].nunique()
        if "movieId" in movies_df.columns and not movies_df.empty
        else 0
    )

    total_ratings = len(ratings_df)

    total_users = (
        ratings_df["userId"].nunique()
        if "userId" in ratings_df.columns and not ratings_df.empty
        else 0
    )

    average_rating = (
        ratings_df["rating"].mean()
        if "rating" in ratings_df.columns and not ratings_df.empty
        else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Movies", format_number(total_movies))

    with col2:
        st.metric("Total Ratings", format_number(total_ratings))

    with col3:
        st.metric("Unique Users", format_number(total_users))

    with col4:
        st.metric(
            "Average Rating",
            format_decimal(average_rating, 2),
        )

    st.markdown(
        '<div class="section-title">How Movie Recommendation System using Big Data Analytics Works</div>',
        unsafe_allow_html=True,
    )

    card1, card2, card3 = st.columns(3)

    with card1:
        st.markdown(
            """
            <div class="info-card">
                <h4>01 · Data Processing</h4>
                <p>
                    MovieLens data is loaded and processed using
                    PySpark, with Bronze, Silver, and Gold output layers.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with card2:
        st.markdown(
            """
            <div class="info-card">
                <h4>02 · ALS Recommendation</h4>
                <p>
                    Spark MLlib Alternating Least Squares learns
                    patterns in user ratings to estimate preferences.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with card3:
        st.markdown(
            """
            <div class="info-card">
                <h4>03 · Model Evaluation</h4>
                <p>
                    Test RMSE helps measure the difference between
                    predicted ratings and actual ratings.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-title">Rating Distribution</div>',
        unsafe_allow_html=True,
    )

    if (
        not ratings_df.empty
        and "rating" in ratings_df.columns
    ):
        chart_data = (
            ratings_df["rating"]
            .dropna()
            .value_counts()
            .sort_index()
        )

        st.bar_chart(chart_data)

    else:
        st.info(
            "Ratings data is not available. Check the MovieLens CSV files."
        )

    st.markdown(
        '<div class="section-title">Top Rated Movies by Average Rating</div>',
        unsafe_allow_html=True,
    )

    if (
        not ratings_df.empty
        and not movies_df.empty
        and {"movieId", "rating"}.issubset(ratings_df.columns)
        and {"movieId", "title"}.issubset(movies_df.columns)
    ):
        movie_summary = (
            ratings_df.groupby("movieId")
            .agg(
                Average_Rating=("rating", "mean"),
                Rating_Count=("rating", "count"),
            )
            .reset_index()
        )

        # Exclude movies with very few ratings from this overview.
        movie_summary = movie_summary[
            movie_summary["Rating_Count"] >= 20
        ]

        movie_summary = movie_summary.merge(
            movies_df[["movieId", "title"]].drop_duplicates("movieId"),
            on="movieId",
            how="left",
        )

        movie_summary = movie_summary.sort_values(
            ["Average_Rating", "Rating_Count"],
            ascending=[False, False],
        ).head(10)

        if not movie_summary.empty:
            movie_summary["Average_Rating"] = (
                movie_summary["Average_Rating"].round(2)
            )

            st.dataframe(
                movie_summary[
                    ["title", "Average_Rating", "Rating_Count"]
                ].rename(
                    columns={
                        "title": "Movie",
                        "Average_Rating": "Average Rating",
                        "Rating_Count": "Number of Ratings",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    else:
        st.info("Movie-rating summary is not available.")


# ============================================================
# 11. EXPLORE MOVIES PAGE
# ============================================================

elif page == "Explore Movies":

    st.markdown(
        '<div class="section-title">Explore the Movie Dataset</div>',
        unsafe_allow_html=True,
    )

    if movies_df.empty:
        st.warning(
            "Movies data was not found. Check data/ml-latest-small/movies.csv."
        )

    else:
        search_query = st.text_input(
            "Search by movie title",
            placeholder="Example: Toy Story",
        )

        genres_series = pd.Series(dtype="object")

        if "genres" in movies_df.columns:
            genres_series = (
                movies_df["genres"]
                .dropna()
                .astype(str)
                .str.split("|")
                .explode()
            )

            genre_options = sorted(
                genre
                for genre in genres_series.unique()
                if genre and genre != "(no genres listed)"
            )
        else:
            genre_options = []

        selected_genre = st.selectbox(
            "Filter by genre",
            ["All Genres"] + genre_options,
        )

        filtered_movies = movies_df.copy()

        if search_query and "title" in filtered_movies.columns:
            filtered_movies = filtered_movies[
                filtered_movies["title"]
                .astype(str)
                .str.contains(
                    search_query,
                    case=False,
                    na=False,
                    regex=False,
                )
            ]

        if (
            selected_genre != "All Genres"
            and "genres" in filtered_movies.columns
        ):
            filtered_movies = filtered_movies[
                filtered_movies["genres"]
                .fillna("")
                .astype(str)
                .str.contains(
                    re.escape(selected_genre),
                    case=False,
                    na=False,
                )
            ]

        st.caption(
            f"{len(filtered_movies):,} movie records match your filters."
        )

        display_columns = [
            column
            for column in ["movieId", "title", "genres"]
            if column in filtered_movies.columns
        ]

        st.dataframe(
            filtered_movies[display_columns],
            use_container_width=True,
            hide_index=True,
            height=450,
        )

        csv_movies = filtered_movies[display_columns].to_csv(index=False)

        st.download_button(
            "Download filtered movies CSV",
            data=csv_movies,
            file_name="Movie Recommendation System using Big Data Analytics_movies.csv",
            mime="text/csv",
        )

        st.markdown(
            '<div class="section-title">Genre Distribution</div>',
            unsafe_allow_html=True,
        )

        if "genres" in movies_df.columns:
            genre_counts = (
                movies_df["genres"]
                .dropna()
                .astype(str)
                .str.split("|")
                .explode()
            )

            genre_counts = genre_counts[
                genre_counts != "(no genres listed)"
            ].value_counts().head(15)

            st.bar_chart(genre_counts)


# ============================================================
# 12. PERSONALIZED RECOMMENDATIONS PAGE
# ============================================================

elif page == "Personalized Recommendations":

    st.markdown(
        '<div class="section-title">Personalized Recommendations</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        Choose a MovieLens user ID to view movies recommended by the
        trained ALS model. Higher predicted scores indicate stronger
        estimated preference, not a guarantee of enjoyment.
        """
    )

    if recs.empty:
        st.warning(
            "No saved user recommendations were found."
        )

        st.code(
            "Expected location: outputs/model/user_recommendations/"
        )

    elif "userId" not in recs.columns:
        st.error(
            "The recommendations data does not contain a userId column."
        )

        st.write("Available columns:", recs.columns.tolist())

    else:
        # Ensure user IDs are numeric before building the dropdown.
        recs["userId"] = pd.to_numeric(
            recs["userId"], errors="coerce"
        )

        user_ids = sorted(
            recs["userId"]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        if not user_ids:
            st.warning(
                "No valid user IDs were found in the recommendations data."
            )

        else:
            selected_user = st.selectbox(
                "Select a MovieLens user ID",
                user_ids,
                index=0,
                help="These are IDs from the MovieLens dataset, not account logins.",
            )

            # IMPORTANT:
            # Create user_rows before passing it to the function.
            user_rows = recs[
                recs["userId"] == selected_user
            ]

            user_recommendations = get_user_recommendations(
                user_rows,
                movies_df,
            )

            if not user_recommendations.empty:

                metric1, metric2 = st.columns(2)

                with metric1:
                    st.metric(
                        "Selected User",
                        str(selected_user),
                    )

                with metric2:
                    st.metric(
                        "Recommendations Found",
                        format_number(len(user_recommendations)),
                    )

                st.markdown(
                    '<div class="section-title">Recommended Movies</div>',
                    unsafe_allow_html=True,
                )

                display_columns = [
                    column
                    for column in [
                        "title",
                        "genres",
                        "Predicted score",
                    ]
                    if column in user_recommendations.columns
                ]

                display_df = user_recommendations[
                    display_columns
                ].copy()

                if "Predicted score" in display_df.columns:
                    display_df["Predicted score"] = (
                        display_df["Predicted score"].round(3)
                    )

                st.dataframe(
                    display_df.rename(
                        columns={
                            "title": "Movie",
                            "genres": "Genres",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                csv_data = user_recommendations.to_csv(index=False)

                st.download_button(
                    "Download recommendations as CSV",
                    data=csv_data,
                    file_name=(
                        f"Movie Recommendation System using Big Data Analytics_recommendations_user_{selected_user}.csv"
                    ),
                    mime="text/csv",
                )

                st.markdown(
                    '<div class="section-title">Predicted Preference Scores</div>',
                    unsafe_allow_html=True,
                )

                if "title" in user_recommendations.columns:
                    chart_df = user_recommendations[
                        ["title", "Predicted score"]
                    ].copy()

                    chart_df = chart_df.set_index("title")

                    st.bar_chart(chart_df)

                st.caption(
                    "Predicted scores are model estimates. "
                    "They may be outside the usual rating range and "
                    "should not be interpreted as guaranteed ratings."
                )

            else:
                st.warning(
                    "No recommendation records could be converted for "
                    f"user {selected_user}."
                )

                # Helpful diagnostic information if the format changes.
                with st.expander("Debug recommendation data"):
                    st.write("Columns:", recs.columns.tolist())
                    st.write("Selected user:", selected_user)
                    st.write("Matching rows:", len(user_rows))

                    if not user_rows.empty:
                        raw_value = user_rows.iloc[0].get(
                            "recommendations"
                        )

                        st.write("Saved value type:", type(raw_value).__name__)
                        st.write("Example saved record:", user_rows.iloc[0].to_dict())

                        st.write(
                            "Parsed recommendation records:",
                            parse_recommendations(raw_value),
                        )


# ============================================================
# 13. MODEL PERFORMANCE PAGE
# ============================================================

elif page == "Model Performance":

    st.markdown(
        '<div class="section-title">ALS Model Performance</div>',
        unsafe_allow_html=True,
    )

    rmse = get_metric(
        metrics,
        ["test_rmse", "rmse", "Test RMSE", "RMSE"],
    )

    training_rows = get_metric(
        metrics,
        ["training_rows", "train_rows", "Training rows"],
    )

    test_rows = get_metric(
        metrics,
        ["test_rows", "Testing rows", "Test rows"],
    )

    model_columns = st.columns(3)

    with model_columns[0]:
        st.metric(
            "Test RMSE",
            format_decimal(rmse, 4)
            if rmse is not None
            else "N/A",
        )

    with model_columns[1]:
        st.metric(
            "Training Rows",
            format_number(training_rows)
            if training_rows is not None
            else "N/A",
        )

    with model_columns[2]:
        st.metric(
            "Test Rows",
            format_number(test_rows)
            if test_rows is not None
            else "N/A",
        )

    st.markdown(
        '<div class="section-title">Understanding the Model</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        **Alternating Least Squares (ALS)** is a collaborative-filtering
        algorithm available in Spark MLlib.

        - **Training data:** user–movie ratings used to learn latent factors.
        - **Test data:** held-out ratings used to evaluate predictions.
        - **RMSE:** Root Mean Squared Error; lower values generally indicate
          predictions closer to the actual ratings on the same test split.
        - **User recommendations:** movies ranked using the model's
          estimated ratings for a user.
        """
    )

    if rmse is not None:
        st.info(
            f"The saved test RMSE is {format_decimal(rmse, 4)}. "
            "Use this value to compare model versions evaluated on "
            "the same test split."
        )
    else:
        st.warning(
            "Model metrics could not be loaded. Check the JSON file "
            "inside outputs/model/metrics/."
        )

    with st.expander("View saved model metrics"):
        if metrics:
            st.json(metrics)
        else:
            st.write("No model metrics JSON file was found.")

    st.markdown(
        '<div class="section-title">Generated Model Outputs</div>',
        unsafe_allow_html=True,
    )

    output_status = [
        (
            "User factors",
            MODEL_DIR / "user_factors",
        ),
        (
            "Item factors",
            MODEL_DIR / "item_factors",
        ),
        (
            "User recommendations",
            USER_RECOMMENDATIONS_DIR,
        ),
        (
            "Item recommendations",
            ITEM_RECOMMENDATIONS_DIR,
        ),
    ]

    status_rows = []

    for name, path in output_status:
        exists = path.exists()

        status_rows.append({
            "Output": name,
            "Status": "Available" if exists else "Not found",
            "Location": str(path.relative_to(ROOT)),
        })

    st.dataframe(
        pd.DataFrame(status_rows),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# 14. FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <strong>Movie Recommendation System using Big Data Analytics</strong> · Movie Recommendation & Big Data Analytics<br>
        Built with Streamlit, PySpark, Spark MLlib ALS, and MovieLens data.<br>
        Educational prototype — model predictions are not guarantees of preference.
    </div>
    """,
    unsafe_allow_html=True,
)
