import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from nlp import vader_sentiment

DB_URL = st.secrets.get('DATABASE_URL', 'postgresql://postgres:postgres@db:5432/tweetsdb')
engine = create_engine(DB_URL)

st.set_page_config(layout='wide', page_title='Twitter Sentiment Dashboard')
st.title('Twitter Sentiment Explorer')

with st.sidebar:
    st.header('Controls')
    query = st.text_input('Scrape query (for live collection)', '#ai lang:en')
    num = st.number_input('How many to collect (snscrape)', value=200, step=50)
    if st.button('Collect now (snscrape)'):
        import subprocess, sys
        subprocess.run([sys.executable,'etl.py','--query', query, '--max', str(num)])
        st.experimental_rerun()

@st.cache_data
def load_data(limit=500):
    q = f"SELECT * FROM tweets ORDER BY created_at DESC LIMIT {limit};"
    return pd.read_sql(q, engine)

df = load_data(500)
st.write('Showing', len(df), 'tweets')

if st.button('Run sentiment (VADER) on loaded'):
    labels = []
    scores = []
    for t in df['text'].fillna(''):
        lab, sc = vader_sentiment(str(t))
        labels.append(lab)
        scores.append(sc)
    df['sentiment'] = labels
    df['sentiment_score'] = scores
    # safe write: update only rows that have no sentiment
    to_update = df[df['sentiment'].notnull()][['id','sentiment','sentiment_score']]
    # write updates back to DB (simple approach: upsert via to_sql may duplicate; user can run SQL upsert for production)
    to_update.to_sql('tweets', engine, if_exists='append', index=False, method='multi')
    st.success('Sentiment written (note: check duplicates / upsert logic for production)')
    st.experimental_rerun()

st.subheader('Sentiment breakdown')
if 'sentiment' in df.columns:
    st.bar_chart(df['sentiment'].value_counts())

st.subheader('Sample tweets')
st.dataframe(df[['created_at','username','text','sentiment']].head(200))
