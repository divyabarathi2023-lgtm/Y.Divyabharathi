import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

# Load dataset
df = pd.read_csv("news.csv")

# Selecting required columns
df = df[['text', 'label']]

# Split data
x = df['text']
y = df['label']

tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf.fit_transform(x)

model = PassiveAggressiveClassifier(max_iter=50)
model.fit(tfidf_train, y)

# Save Model
pickle.dump(model, open("fake_news_model.pkl", "wb"))
pickle.dump(tfidf, open("tfidf_vectorizer.pkl", "wb"))

print("Model Saved Successfully ✔")
