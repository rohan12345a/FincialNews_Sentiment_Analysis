import streamlit as st
from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Load the fine-tuned model and tokenizer
model = BertForSequenceClassification.from_pretrained("C:\\Users\\Lenovo\\Downloads\\aibf-folder\\AIBF_Model")
tokenizer = BertTokenizer.from_pretrained("C:\\Users\\Lenovo\\Downloads\\aibf-folder\\AIBF_Model")

# Function to predict sentiment
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    sentiment = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    return sentiment[predictions.item()]

# Streamlit interface
st.title("Financial Sentiment Analysis")
st.write("Enter a financial text to analyze its sentiment:")

# Text input for user
user_input = st.text_area("Input Text:", height=150)

# Button to predict sentiment
if st.button("Predict"):
    if user_input:
        sentiment = predict_sentiment(user_input)
        st.success(f"The predicted sentiment is: **{sentiment}**")
    else:
        st.warning("Please enter some text to analyze.")


with st.container():
    with st.sidebar:
        members = [
            {"name": "Rohan Saraswat", "email": "rohan.saraswat2003@gmail. com", "linkedin": "https://www.linkedin.com/in/rohan-saraswat-a70a2b225/"},
        ]

        # Define the page title and heading
        st.markdown("<h1 style='font-size:28px'>Author</h1>", unsafe_allow_html=True)

        # Iterate over the list of members and display their details
        for member in members:
            st.write(f"Name: {member['name']}")
            st.write(f"Email: {member['email']}")
            st.write(f"LinkedIn: {member['linkedin']}")
            st.write("")