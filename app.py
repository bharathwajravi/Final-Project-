import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
import os

# Path to the sentiment model files
sentiment_model_path = r'G:\Data Science Projects\Final Project\Model Files\Sentimental model file\content\sentiment_model'

# Load the sentiment model and tokenizer
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_path)
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_path)

def predict_sentiment(text):
    inputs = sentiment_tokenizer(text, return_tensors="pt")
    outputs = sentiment_model(**inputs)
    probs = outputs.logits.softmax(dim=-1)
    return probs

# Path to the text summarization model files (replace with actual path)
summarization_model_path = r'G:\Data Science Projects\Final Project\Model Files\Text Summarization model file\content\text_summarization_model'

# Ensure the path exists
if os.path.exists(summarization_model_path):
    # Load the text summarization model and tokenizer
    summarization_model = AutoModelForSeq2SeqLM.from_pretrained(summarization_model_path)
    summarization_tokenizer = AutoTokenizer.from_pretrained(summarization_model_path)
else:
    summarization_model = None
    summarization_tokenizer = None

def summarize_text(text):
    inputs = summarization_tokenizer(text, return_tensors="pt", truncation=True)
    summary_ids = summarization_model.generate(inputs.input_ids)
    summary = summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Streamlit App
st.title("NLP and Image Generation Model Application")

task = st.sidebar.selectbox("Select a Task", ["Sentiment Analysis", "Text Summarization", "Image Generation"])

if task == "Sentiment Analysis":
    user_input = st.text_input("Enter text for sentiment analysis:")
    if st.button("Analyze"):
        sentiment = predict_sentiment(user_input)
        st.write("Sentiment probabilities:", sentiment)
elif task == "Text Summarization":
    if summarization_model and summarization_tokenizer:
        user_input = st.text_area("Enter text for summarization:")
        if st.button("Summarize"):
            summary = summarize_text(user_input)
            st.write(summary)
    else:
        st.write("Text Summarization model path is incorrect or model is not available.")
elif task == "Image Generation":
    st.write("Image Generation task not implemented yet.")
