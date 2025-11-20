# load_sample.py
import os
import pandas as pd
from sqlalchemy import create_engine, text

DB = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/tweetsdb")
engine = create_engine(DB)

def ensure_table(engine):
    create_sql = """
    CREATE TABLE IF NOT EXISTS tweets (
      id BIGINT PRIMARY KEY,
      username TEXT,
      created_at TIMESTAMP,
      text TEXT,
      lang TEXT,
      retweet_count INT,
      reply_count INT,
      like_count INT,
      quote_count INT,
      scraped_at TIMESTAMP DEFAULT now(),
      sentiment TEXT,
      sentiment_score FLOAT
    );
    """
    with engine.connect() as conn:
        conn.execute(text(create_sql))
        conn.commit()

def load_csv(path):
    df = pd.read_csv(path)

    # Normalize columns: rename label -> sentiment (if present)
    if 'label' in df.columns and 'sentiment' not in df.columns:
        df = df.rename(columns={'label': 'sentiment'})

    # add any missing columns the DB expects (fill with None/defaults)
    if 'id' not in df.columns:
        df['id'] = range(1, len(df) + 1)
    if 'created_at' not in df.columns:
        df['created_at'] = pd.Timestamp.now()
    if 'username' not in df.columns:
        df['username'] = None
    if 'lang' not in df.columns:
        df['lang'] = None
    for col in ['retweet_count','reply_count','like_count','quote_count']:
        if col not in df.columns:
            df[col] = None
    if 'scraped_at' not in df.columns:
        df['scraped_at'] = pd.Timestamp.now()
    if 'sentiment_score' not in df.columns:
        df['sentiment_score'] = None

    # Only keep columns that exist in the DB table (this prevents accidental extra columns)
    cols = ['id','username','created_at','text','lang','retweet_count','reply_count','like_count','quote_count','scraped_at','sentiment','sentiment_score']
    df = df[cols]

    # Insert (append)
    df.to_sql('tweets', engine, if_exists='append', index=False, method='multi')
    print("Inserted", len(df), "rows")

if __name__ == "__main__":
    ensure_table(engine)
    load_csv("sample_data/labeled_tweets.csv")
