import streamlit as st
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import torch

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
        except Exception as e:
            st.error(f"An error occurred during classification: {e}")
