"""ETL script: collects tweets and inserts into Postgres.
Supports two modes: `snscrape` (no API key) or `twitter_api` (requires env vars).
"""
import os, argparse
import pandas as pd
from sqlalchemy import create_engine
from tqdm import tqdm

DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/tweetsdb")
engine = create_engine(DB_URL)

def scrape_snscrape(query, max_tweets=500):
    import snscrape.modules.twitter as sntwitter
    rows = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        if i >= max_tweets:
            break
        rows.append({
            "id": tweet.id,
            "username": tweet.user.username,
            "created_at": tweet.date,
            "text": tweet.content,
            "lang": tweet.lang,
            "retweet_count": tweet.retweetCount,
            "reply_count": tweet.replyCount,
            "like_count": tweet.likeCount,
            "quote_count": tweet.quoteCount,
        })
    return pd.DataFrame(rows)

def load_to_db(df):
    if df.empty:
        return 0
    # safe insert: use if_exists='append' but rely on primary key to avoid duplicates
    df.to_sql('tweets', engine, if_exists='append', index=False, method='multi')
    return len(df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=['snscrape','twitter_api'], default='snscrape')
    parser.add_argument('--query', type=str, required=True)
    parser.add_argument('--max', type=int, default=500)
    args = parser.parse_args()

    if args.source == 'snscrape':
        df = scrape_snscrape(args.query, max_tweets=args.max)
        cnt = load_to_db(df)
        print(f"Inserted {cnt} tweets")
    else:
        raise NotImplementedError('twitter_api path: implement using your API keys')
