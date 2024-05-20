import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.corpus import stopwords
import re
from joblib import load

nltk.download('stopwords')

# Function to clean and preprocess the text
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text

# Function to scrape news articles
def scrape_news_articles(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    articles = []
    for item in soup.find_all('article'):
        headline = item.find('h1', class_='headline')
        if headline:
            headline_text = headline.get_text(strip=True)
            article_url = headline.find('a')['href']
            article_response = requests.get(article_url)
            article_soup = BeautifulSoup(article_response.content, 'html.parser')
            article_content = article_soup.find('div', class_='article-content')
            if article_content:
                article_text = article_content.get_text(strip=True)
                articles.append((headline_text, article_text))
    return articles

# Function to classify the difficulty of the text
def classify_difficulty(text, model, vectorizer):
    processed_text = preprocess_text(text)
    X = vectorizer.transform([processed_text])
    predicted_class = model.predict(X)
    return predicted_class[0]

# Load pre-trained classifier and vectorizer
vectorizer = load('models/vectorizer.joblib')
model = load('models/model.joblib')

# Streamlit app
st.title('French News Article Recommender')
st.write('Enter the desired level of language difficulty:')

difficulty_level = st.selectbox('Difficulty Level', ['Easy', 'Medium', 'Hard'])

if st.button('Find Article'):
    url = 'https://newswebsite.com/latest'  # Replace with the actual news website URL
    articles = scrape_news_articles(url)
    
    if articles:
        for headline, article in articles:
            difficulty = classify_difficulty(article, model, vectorizer)
            if difficulty == difficulty_level.lower():
                st.write(f'### {headline}')
                st.write(article)
                break
        else:
            st.write('No articles found for the desired difficulty level.')
    else:
        st.write('No articles found.')
