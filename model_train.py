# Example supervised trainer: TF-IDF + LogisticRegression
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from joblib import dump

# expects CSV with columns text,label in sample_data/
df = pd.read_csv('sample_data/labeled_tweets.csv')
X_train, X_test, y_train, y_test = train_test_split(df.text, df.label, test_size=0.2, random_state=42)
pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
    ('clf', LogisticRegression(max_iter=1000))
])
pipe.fit(X_train, y_train)
print('train score', pipe.score(X_train,y_train))
print('test score', pipe.score(X_test,y_test))
dump(pipe, 'model.joblib')
