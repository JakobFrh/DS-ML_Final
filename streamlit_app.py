import streamlit as st
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import torch
import pandas as pd
import random
import urllib.error

# Load model and tokenizer
model_dir = "model"
try:
    tokenizer = CamembertTokenizer.from_pretrained(model_dir)
    model = CamembertForSequenceClassification.from_pretrained(model_dir)
except ImportError as e:
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

url = "https://raw.githubusercontent.com/JakobFrh/DS-ML_Final/main/Video_french.csv"
df = load_dataframe(url)

# if df is not None:
    # st.sidebar.success("Welcome to RiBERTy, your French assessment assistant.")
    # st.sidebar.write("Use this app to assess your French level and get matched with a video based on your proficiency.")
# else:
    # st.error("Failed to load the dataframe.")
    # st.stop()

# Streamlit app
st.title("RiBERTy")
st.write("...when the beautiful language meets the beautiful game.")

st.markdown(
    """
    <style>
    .big-font {
        font-size:20px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="big-font">Enter a french sentence below, so RiBERTy can asses you french level.</p>', unsafe_allow_html=True)

text = st.text_area("Enter text:", "")

if st.button("Let's go"):
    if not text:
        st.error("Please enter some text.")
    else:
        try:
            inputs = tokenizer(text, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()

            if 'Level' in df.columns:
                matching_videos = df[df['Level'] == predicted_class]
            else:
                st.error("No matching column for difficulty level found in the dataframe.")
                st.stop()

            if not matching_videos.empty:
                selected_video = matching_videos.sample(n=1).iloc[0]
                player_name = selected_video["Player"]
                video_url = selected_video["Video"]
                st.write(f"Your French level corresponds to: **{player_name}**")
                st.video(video_url)
            else:
                st.error("No videos found for the predicted difficulty level.")
        except Exception as e:
            st.error(f"An error occurred during classification: {e}")

# st.sidebar.title("About")
# st.sidebar.info(
#     """
#     This application uses a CamemBERT model to classify text into different levels of French proficiency. 
#     Based on the predicted level, a video is recommended to help improve your French skills.
#     """
# )

st.sidebar.title("Instructions")
st.sidebar.write(
    """
    1. Enter a piece of French text in the text area.
    2. Click the 'Let's go' button to assess your level.
    3. Watch the recommended video to improve your French!
    """
)

