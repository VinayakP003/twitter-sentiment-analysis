# streamlit_app.py
# Streamlit dashboard for Twitter Sentiment Explorer
# - reads DATABASE_URL from st.secrets or env
# - safe DB connection with helpful error messages
# - fallback to sample CSV if DB not reachable
# - simple VADER sentiment runner

import os
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import traceback

# --- Helpers: DB engine with safe connect ---
def get_database_url():
    # Streamlit secrets recommended for Cloud; fallback to env var
    db = None
    try:
        db = st.secrets["DATABASE_URL"]
    except Exception:
        db = os.getenv("DATABASE_URL")
    return db

@st.cache_resource
def make_engine(database_url: str):
    # create engine with short timeout and pool_pre_ping to avoid stale connections
    if database_url is None or database_url == "":
        return None, "DATABASE_URL not set"
    try:
        # Add connect_timeout and enable pool_pre_ping
        engine = create_engine(database_url, connect_args={"connect_timeout": 10}, pool_pre_ping=True)
        # quick test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine, None
    except Exception as e:
        return None, str(e) + "\n\n" + traceback.format_exc()

# --- Simple VADER sentiment (fast) ---
def ensure_nltk_vader():
    try:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
    except Exception:
        # try to (silently) install nltk in environments where it's not present
        raise RuntimeError("NLTK not available. Install `nltk` and download vader_lexicon.")

def vader_sentiment(text):
    # note: ensure you have `nltk` and vader_lexicon downloaded in the environment
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

# --- Load data (from DB or fallback to sample CSV) ---
@st.cache_data
def load_data_from_db(engine, limit=500):
    q = f"SELECT * FROM tweets ORDER BY created_at DESC LIMIT {limit};"
    return pd.read_sql(q, engine)

def load_sample_csv(path="sample_data/labeled_tweets.csv"):
    df = pd.read_csv(path)
    # Normalize columns expected by app
    if "label" in df.columns and "sentiment" not in df.columns:
        df = df.rename(columns={"label": "sentiment"})
    # add required columns if missing
    for col in ["id", "username", "created_at", "text", "lang", "retweet_count", "reply_count", "like_count", "quote_count", "sentiment", "sentiment_score"]:
        if col not in df.columns:
            df[col] = None
    # ensure created_at is datetime
    try:
        df["created_at"] = pd.to_datetime(df["created_at"])
    except Exception:
        df["created_at"] = pd.Timestamp.now()
    return df

# --- UI ---
st.set_page_config(layout="wide", page_title="Twitter Sentiment Explorer")
st.title("Twitter Sentiment Explorer")

with st.sidebar:
    st.header("Controls")
    st.write("Data source & actions")
    database_url = get_database_url()
    st.write("DATABASE configured:", bool(database_url))
    collect_btn = st.button("(Local) Load sample CSV into view")
    run_vader_btn = st.button("Run VADER on loaded tweets")
    limit = st.number_input("How many rows to load (view)", min_value=10, max_value=5000, value=500, step=50)
    st.markdown("---")
    st.markdown("**Deploy notes**:")
    st.markdown("- On Streamlit Cloud: add `DATABASE_URL` in _Settings → Secrets_.")
    st.markdown("- For many managed Postgres, add `?sslmode=require` to the URL if needed.")

# Attempt to create engine (safe)
engine, engine_err = make_engine(database_url)

if engine is None:
    st.warning("Database engine not available. Reason: " + (engine_err or "No DATABASE_URL found."))
    st.info("Falling back to sample CSV for browsing. Streams that require DB write will be disabled.")
    df = load_sample_csv()
else:
    try:
        df = load_data_from_db(engine, limit=int(limit))
        st.success(f"Loaded {len(df)} rows from DB.")
    except Exception as e:
        st.error("Error loading from DB. Falling back to sample CSV (see logs).")
        st.text(str(e))
        st.text(traceback.format_exc())
        df = load_sample_csv()

# Show stats and quick viz
st.subheader("Sentiment breakdown (if present)")
if "sentiment" in df.columns:
    vc = df["sentiment"].value_counts()
    st.bar_chart(vc)
else:
    st.info("No 'sentiment' column in data yet.")

st.subheader("Sample tweets")
st.dataframe(df[["created_at", "username", "text", "sentiment"]].head(200), use_container_width=True)

# Run VADER if requested
if run_vader_btn:
    try:
        ensure_nltk_vader()
    except Exception as e:
        st.error("NLTK not installed in this environment. Install `nltk` and download vader_lexicon.")
        st.stop()

    st.info("Running VADER on loaded tweets (in-memory)...")
    labels, scores = [], []
    for t in df["text"].fillna("").astype(str).tolist():
        lab, sc = vader_sentiment(t)
        labels.append(lab)
        scores.append(sc)
    df["sentiment"] = labels
    df["sentiment_score"] = scores
    st.success("VADER completed (in-memory). You can download or push to DB if desired.")

    # If DB available, offer to write sentiment back
    if engine is not None:
        if st.confirmation_dialog := st.button("Write sentiment back to DB (append/upsert)"):
            st.info("Writing sentiment back to DB...")
            try:
                # simple upsert approach: create temp table and insert non-duplicates
                temp_name = "tmp_streamlit_sent"
                with engine.begin() as conn:
                    conn.execute(text(f"CREATE TEMP TABLE {temp_name} (LIKE tweets INCLUDING ALL);"))
                    # write only id, sentiment, sentiment_score
                    updf = df[["id", "sentiment", "sentiment_score"]].dropna(subset=["id"])
                    if not updf.empty:
                        updf.to_sql(temp_name, conn, index=False, if_exists="append", method="multi")
                        conn.execute(
                            text(f"""
                            INSERT INTO tweets (id, sentiment, sentiment_score)
                            SELECT id, sentiment, sentiment_score FROM {temp_name}
                            ON CONFLICT (id) DO UPDATE SET sentiment=EXCLUDED.sentiment, sentiment_score=EXCLUDED.sentiment_score;
                            """)
                        )
                        st.success("Sentiment upserted to DB.")
                    else:
                        st.warning("No valid ids present to upsert.")
            except Exception as e:
                st.error("Failed to write to DB: " + str(e))
                st.text(traceback.format_exc())
    else:
        st.info("DB not configured; skipping DB write.")

# Download CSV of current view
st.markdown("---")
csv = df.to_csv(index=False)
st.download_button("Download current data (CSV)", data=csv, file_name="tweets_view.csv", mime="text/csv")
