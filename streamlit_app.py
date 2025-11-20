# streamlit_app.py
# Clean, stable version without syntax errors & fully Python 3.13 compatible

import os
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import traceback

# ------------------------------------------------------------
# Helper 1: Load DATABASE_URL from Streamlit Secrets or env
# ------------------------------------------------------------
def get_database_url():
    db = None
    try:
        db = st.secrets["DATABASE_URL"]
    except Exception:
        db = os.getenv("DATABASE_URL")
    return db


# ------------------------------------------------------------
# Helper 2: Create SQLAlchemy engine with safe tests
# ------------------------------------------------------------
@st.cache_resource
def make_engine(database_url: str):
    if not database_url:
        return None, "DATABASE_URL is not set anywhere."

    try:
        engine = create_engine(
            database_url,
            connect_args={"connect_timeout": 10},
            pool_pre_ping=True,
        )
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine, None
    except Exception as e:
        return None, f"Engine creation failed: {e}\n\n{traceback.format_exc()}"


# ------------------------------------------------------------
# Helper 3: Load data from DB
# ------------------------------------------------------------
@st.cache_data
def load_data_from_db(engine, limit=500):
    q = f"SELECT * FROM tweets ORDER BY created_at DESC LIMIT {limit};"
    return pd.read_sql(q, engine)


# ------------------------------------------------------------
# Helper 4: Load sample CSV if DB not available
# ------------------------------------------------------------
def load_sample_csv(path="sample_data/labeled_tweets.csv"):
    df = pd.read_csv(path)

    # Fix label column if needed
    if "label" in df.columns and "sentiment" not in df.columns:
        df = df.rename(columns={"label": "sentiment"})

    # Add missing columns
    required_cols = [
        "id", "username", "created_at", "text", "lang",
        "retweet_count", "reply_count", "like_count", "quote_count",
        "sentiment", "sentiment_score"
    ]
    for c in required_cols:
        if c not in df.columns:
            df[c] = None

    # Ensure created_at is datetime
    try:
        df["created_at"] = pd.to_datetime(df["created_at"])
    except:
        df["created_at"] = pd.Timestamp.now()

    return df


# ------------------------------------------------------------
# Helper 5: VADER sentiment
# ------------------------------------------------------------
def prepare_vader():
    try:
        import nltk
        nltk.download("vader_lexicon")
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
    except Exception:
        raise RuntimeError("NLTK/VADER not available in this environment.")


def vader_sentiment(text):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    s = sid.polarity_scores(text)
    score = float(s["compound"])
    if score >= 0.05:
        label = "positive"
    elif score <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return label, score


# ------------------------------------------------------------
# UI START
# ------------------------------------------------------------

st.set_page_config(layout="wide", page_title="Twitter Sentiment Explorer")
st.title("Twitter Sentiment Explorer")

with st.sidebar:
    st.header("Controls")

    database_url = get_database_url()
    st.write("DATABASE_URL detected:", bool(database_url))

    limit = st.number_input("Rows to load", min_value=10, max_value=3000, value=500, step=50)

    run_vader = st.button("Run VADER sentiment on data (in-memory)")


# ------------------------------------------------------------
# DATABASE / DATA LOAD FLOW
# ------------------------------------------------------------

engine, engine_err = make_engine(database_url)

if engine is None:
    st.warning("DB not available. Reason: ")
    st.code(engine_err)
    st.info("Falling back to sample CSV / read-only mode.")
    df = load_sample_csv()
else:
    try:
        df = load_data_from_db(engine, limit=int(limit))
        st.success(f"Loaded {len(df)} rows from DB.")
    except Exception as e:
        st.error("Failed reading from DB. Falling back to sample CSV.")
        st.code(str(e))
        st.code(traceback.format_exc())
        df = load_sample_csv()


# ------------------------------------------------------------
# Display data
# ------------------------------------------------------------
st.subheader("Sentiment distribution")
if "sentiment" in df.columns:
    st.bar_chart(df["sentiment"].value_counts())
else:
    st.info("No sentiment column yet.")

st.subheader("Tweets")
st.dataframe(df[["created_at", "username", "text", "sentiment"]].head(200), use_container_width=True)


# ------------------------------------------------------------
# Run VADER
# ------------------------------------------------------------
if run_vader:
    try:
        prepare_vader()
        st.info("Running VADER...")

        labels, scores = [], []
        for t in df["text"].fillna("").astype(str):
            lab, sc = vader_sentiment(t)
            labels.append(lab)
            scores.append(sc)

        df["sentiment"] = labels
        df["sentiment_score"] = scores

        st.success("VADER completed. Showing updated sentiment:")
        st.dataframe(df[["text", "sentiment", "sentiment_score"]])

    except Exception as e:
        st.error("VADER ERROR: " + str(e))
        st.code(traceback.format_exc())


# ------------------------------------------------------------
# Download CSV button
# ------------------------------------------------------------
st.markdown("---")
csv = df.to_csv(index=False)
st.download_button(
    "Download data as CSV",
    data=csv,
    file_name="tweets_sentiment.csv",
    mime="text/csv",
)
