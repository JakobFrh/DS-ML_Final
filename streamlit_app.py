import streamlit as st
import torch
from transformers import CamembertForSequenceClassification, CamembertTokenizer
import sentencepiece as spm

# Load the SentencePiece model
sp = spm.SentencePieceProcessor()
sp.load('/mnt/data/sentencepiece.bpe.model')

# Load pre-trained CamemBERT model and tokenizer
tokenizer = CamembertTokenizer.from_pretrained('models/camembert_tokenizer')
model = CamembertForSequenceClassification.from_pretrained('models/camembert_model')

# Function to tokenize text using SentencePiece
def tokenize_with_sentencepiece(text):
    pieces = sp.encode_as_pieces(text)
    ids = sp.encode_as_ids(text)
    return pieces, ids

# Streamlit app
st.title('French Sentence Difficulty Classifier with CamemBERT and SentencePiece')
st.write('Enter a French sentence to classify its difficulty level:')

sentence = st.text_input('Sentence')
if st.button('Classify'):
    if sentence:
        pieces, ids = tokenize_with_sentencepiece(sentence)
        inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
        
        # Assuming class indices: 0 - Easy, 1 - Medium, 2 - Hard
        difficulty_map = {0: 'Easy', 1: 'Medium', 2: 'Hard'}
        st.write(f'The difficulty level of the sentence is: {difficulty_map[predicted_class]}')
        st.write(f'Tokenized Sentence: {pieces}')
    else:
        st.write('Please enter a sentence to classify.')
