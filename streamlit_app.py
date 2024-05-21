import streamlit as st
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import torch
import pandas as pd
import random

# Load model and tokenizer
model_dir = "model"
try:
    st.write("Loading tokenizer...")
    tokenizer = CamembertTokenizer.from_pretrained(model_dir)
    st.write("Tokenizer loaded successfully")
    
    # Log special tokens
    special_tokens = tokenizer.special_tokens_map
    st.write(f"Special tokens: {special_tokens}")

    st.write("Loading model...")
    model = CamembertForSequenceClassification.from_pretrained(model_dir)
    st.write("Model loaded successfully")

    st.success("Model and tokenizer loaded successfully")
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model and tokenizer: {e}")
    st.stop()

# Load the dataframe from GitHub
@st.cache
def load_dataframe():
    url = "https://raw.githubusercontent.com/JakobFrh/DS-ML_Final/main/Video_french.csv"
    df = pd.read_csv(url)
    return df

df = load_dataframe()

# Streamlit app
st.title("Camembert Model for Sequence Classification")

text = st.text_area("Enter text:", "")

if st.button("Classify"):
    if not text:
        st.error("Please enter some text.")
    else:
        try:
            st.write(f"Input text: {text}")
            inputs = tokenizer(text, return_tensors="pt")
            st.write(f"Tokenized inputs: {inputs}")

            outputs = model(**inputs)
            logits = outputs.logits
            st.write(f"Model outputs (logits): {logits}")

            predicted_class = torch.argmax(logits, dim=1).item()
            st.write(f"Predicted class: {predicted_class}")

            # Retrieve videos that match the predicted difficulty level
            matching_videos = df[df['difficulty_level'] == predicted_class]
            if not matching_videos.empty:
                selected_video = matching_videos.sample(n=1).iloc[0]
                player_name = selected_video["player_name"]
                video_url = selected_video["video_url"]
                st.write(f"Your French level corresponds to: {player_name}")
                st.video(video_url)
            else:
                st.error("No videos found for the predicted difficulty level.")
        except Exception as e:
            st.error(f"An error occurred during classification: {e}")
