# Twitter Sentiment Analysis

---

## ✅ Project overview

This repo contains an end-to-end pipeline to:

* Collect tweets using `snscrape` (or the official Twitter/X API) and persist raw records to Postgres.
* Run preprocessing and sentiment classification using lightweight rule-based (VADER) or transformer-based models.
* Optionally train a supervised classifier (TF-IDF + LogisticRegression) from labeled data.
* Serve an interactive Streamlit dashboard for exploratory analysis and on-demand re-processing.

It is designed to be easy to run locally and straightforward to harden for production (BigQuery/Snowflake support, upserts, retries, and CI / deployment pipelines).

---

## 🔧 Features

* Raw tweet ingestion (snscrape; template available for Twitter API integration)
* Postgres schema and Docker Compose for local data warehouse
* NLP: VADER (fast) and optional HuggingFace transformer pipeline (higher quality)
* Optional supervised training pipeline (TF-IDF + Logistic Regression)
* Streamlit app for visualization, manual reclassification, and light ETL triggers
* Sample labeled dataset and scripts to train and save models

---

## 🗂️ Repository layout

```
twitter-sentiment/
├─ README.md                    # (This file)
├─ requirements.txt
├─ docker-compose.yml
├─ db/init.sql
├─ etl.py                       # Scraper + loader
├─ nlp.py                       # preprocessing & sentiment functions
├─ model_train.py               # optional supervised trainer
├─ streamlit_app.py             # Streamlit dashboard
├─ utils.py
├─ sample_data/
│  └─ labeled_tweets.csv
└─ .gitignore
```

---

## 🚀 Quickstart — local (developer)

1. Install Docker (for Postgres) and Python 3.10+.

2. Start the database:

```bash
docker compose up -d
```

3. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate   # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

4. Run the ETL to collect tweets (example using `snscrape`):

```bash
python etl.py --source snscrape --query "#ai lang:en" --max 200
```

5. Run the Streamlit dashboard:

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser.

---

## 🔐 Configuration & secrets

* Local Postgres connection defaults to `postgresql://postgres:postgres@localhost:5432/tweetsdb`.
* For production, set `DATABASE_URL` environment variable to point to your Postgres/Cloud SQL or to a SQLAlchemy-compatible connection string.
* If you use the Twitter API, create a developer app and set the appropriate keys as environment variables (place them in a `.env` file and load via `python-dotenv`).

---

## 🧪 Modeling & NLP

* `nlp.py` provides a fast rule-based option (VADER) and a heavier transformer-based option (HuggingFace pipeline). VADER is ideal for quick runs and smaller machines.
* `model_train.py` demonstrates a TF-IDF + LogisticRegression supervised approach. Replace or extend this with more advanced models as needed.

---

## 📦 Deployment recommendations

* Small deployments: Dockerize the Streamlit app and deploy on services like Railway, Fly.io, or Render.
* Production data warehousing: export raw JSON to Parquet and load into BigQuery or Snowflake for large-scale analytics.
* For long-running scraping at scale: implement robust backoff, rate-limit handling, proxy rotation, and deduplication.

---


