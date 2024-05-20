import streamlit as st
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import torch

# Load model and tokenizer
model_name = "model"  # directory where your model files are located
tokenizer = CamembertTokenizer.from_pretrained(model_name)
model = CamembertForSequenceClassification.from_pretrained(model_name)

# Streamlit app
st.title('Camembert Model for Sequence Classification')

text = st.text_area('Enter text:', '')

if st.button('Classify'):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    st.write(f'Predicted class: {predicted_class}')
