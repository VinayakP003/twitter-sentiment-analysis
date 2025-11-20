import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except:
    nltk.download('vader_lexicon')
    nltk.download('punkt')

sid = SentimentIntensityAnalyzer()

def vader_sentiment(text):
    s = sid.polarity_scores(text)
    score = s['compound']
    if score >= 0.05:
        label = 'positive'
    elif score <= -0.05:
        label = 'negative'
    else:
        label = 'neutral'
    return label, float(score)

# Optional: transformers-based sentiment (higher quality but heavier)
try:
    from transformers import pipeline
    _transformer_pipeline = None
    def transformer_sentiment(text):
        global _transformer_pipeline
        if _transformer_pipeline is None:
            _transformer_pipeline = pipeline('sentiment-analysis')
        out = _transformer_pipeline(text[:512])[0]
        label = out['label'].lower()
        score = float(out['score'])
        return label, score
except Exception:
    def transformer_sentiment(text):
        raise RuntimeError('transformers not available or failed to load')
