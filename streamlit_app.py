import streamlit as st
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import torch


# Load model and tokenizer
model_dir = "model"
try:
    tokenizer = CamembertTokenizer.from_pretrained(model_dir)
    model = CamembertForSequenceClassification.from_pretrained(model_dir)
    st.success("Model and tokenizer loaded successfully")
except ImportError as e:
    st.error(f"Import error: {e}")
except Exception as e:
    st.error(f"An error occurred: {e}")

# Streamlit app
st.title("Camembert Model for Sequence Classification")

text = st.text_area("Enter text:", "")

if st.button("Classify"):
    try:
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        st.write(f"Predicted class: {predicted_class}")
    except Exception as e:
        st.error(f"An error occurred during classification: {e}")
