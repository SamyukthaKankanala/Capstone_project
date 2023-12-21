import streamlit as st
import pandas as pd
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
def tokenize(text):
    return word_tokenize(text)

def to_lowercase(tokens):
    return [token.lower() for token in tokens]

def remove_special_characters(tokens):
    return [token for token in tokens if token.isalnum()]

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token not in stop_words]

def lemmatize(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token, pos='v') for token in tokens]

def preprocess_text(text):
    tokens = tokenize(text)
    tokens = to_lowercase(tokens)
    tokens = remove_special_characters(tokens)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return ' '.join(tokens)


logreg_Tfidf = joblib.load('logreg_Tfidf.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

def predict_sentiment(text):
    transformed_text = tfidf_vectorizer.transform([text])
    prediction = logreg_Tfidf.predict(transformed_text)
    return prediction[0]

def main():
    st.title("Product Review Sentiment Analysis")
    review_text = st.text_area("Enter a product review:")

    if st.button("Analyze"):
        sentiment = predict_sentiment(review_text)
        st.write(f"The predicted sentiment is: {sentiment}")

if __name__ == "__main__":
    main()

