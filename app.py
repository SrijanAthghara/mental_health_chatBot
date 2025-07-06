from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer
import re
import string

# Download resources
nltk.download('wordnet')

# Init
app = Flask(__name__)
lemmatizer = WordNetLemmatizer()

# Load CSV
df2 = pd.read_csv('Dataset/mentalhealth.csv')

# Preprocess questions
def preprocess(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and len(w) > 1]
    return " ".join(tokens)

df2['clean_questions'] = df2['Questions'].apply(preprocess)

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df2['clean_questions'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['message']
    clean_input = preprocess(user_input)
    user_vector = vectorizer.transform([clean_input])

    similarity = cosine_similarity(user_vector, tfidf_matrix)
    best_match_idx = similarity.argmax()
    final_response = df2.iloc[best_match_idx]['Answers']

    return render_template('index.html', user_input=user_input, bot_response=final_response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)


