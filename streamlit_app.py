import streamlit as st
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import torch
import pandas as pd
import random
import urllib.error

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
@st.cache_data
def load_dataframe(url):
    try:
        df = pd.read_csv(url)
        return df
    except urllib.error.HTTPError as e:
        st.error(f"HTTP error: {e.code} - {e.reason}")
    except urllib.error.URLError as e:
        st.error(f"URL error: {e.reason}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
    return None

url = "https://raw.githubusercontent.com/JakobFrh/DS-ML_Final/Video_french.csv"
df = load_dataframe(url)

if df is not None:
    st.write("Dataframe loaded successfully. Here are the columns:")
    st.write(df.columns.tolist())
else:
    st.error("Failed to load the dataframe.")
    st.stop()

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

            # Check and use the correct column name for difficulty level
            if 'Level' in df.columns:
                matching_videos = df[df['Level'] == predicted_class]
            elif 'Level' in df.columns:  # If the column name is 'Level'
                matching_videos = df[df['Level'] == predicted_class]
            else:
                st.error("No matching column for difficulty level found in the dataframe.")
                st.stop()

            if not matching_videos.empty:
                selected_video = matching_videos.sample(n=1).iloc[0]
                player_name = selected_video["Player"]
                video_url = selected_video["Video"]
                st.write(f"Your French level corresponds to: {player_name}")
                st.video(video_url)
            else:
                st.error("No videos found for the predicted difficulty level.")
        except Exception as e:
            st.error(f"An error occurred during classification: {e}")
