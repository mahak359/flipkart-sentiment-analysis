import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")

# Load trained sentiment model
pipeline = joblib.load("NaiveBayes_sentiment_model.pkl")

out_dict = {
    0: "❌ Negative Review",
    1: "✅ Positive Review"
}

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

st.title("🛒 Flipkart Review Sentiment Analyzer")
st.write("Enter a Flipkart product review to predict sentiment")

review = st.text_input("Enter Review", placeholder="Type your review here...")

btn_click = st.button("Predict")

if btn_click:
    if review.strip() != "":
        cleaned = clean_text(review)
        pred = pipeline.predict([cleaned])[0]
        st.success(out_dict[pred])
    else:
        st.error("Please enter a review")
