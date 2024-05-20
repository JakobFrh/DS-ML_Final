import streamlit as st
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import torch
import os

# Define the model directory
model_dir = "model"

# Verify that the model directory exists
if not os.path.isdir(model_dir):
    st.error(f"Model directory {model_dir} does not exist.")
else:
    # Load model and tokenizer
    try:
        tokenizer = CamembertTokenizer.from_pretrained(model_dir)
        model = CamembertForSequenceClassification.from_pretrained(model_dir)
    except Exception as e:
        st.error(f"Error loading model: {e}")
    else:
        # Streamlit app
        st.title("Camembert Model for Sequence Classification")

        text = st.text_area("Enter text:", "")

        if st.button("Classify"):
            inputs = tokenizer(text, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
            st.write(f"Predicted class: {predicted_class}")
