# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 17:52:46 2023

@author: elsli
"""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

access_token = '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'
model = AutoModelForSequenceClassification.from_pretrained("elsliew/autotrain-skillsync2-69166137722", use_auth_token=access_token)
tokenizer = AutoTokenizer.from_pretrained("elsliew/autotrain-skillsync2-69166137722", use_auth_token=access_token)

def analyze_sentiment(input_data):
    tokens = tokenizer.encode(input_data, return_tensors='pt')
    results = model(tokens)
    logits = results.logits.squeeze(0)
    probabilities = torch.softmax(logits, dim=0)
    sentiment_score = torch.dot(torch.tensor([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]), probabilities)
    skillsync = round(sentiment_score.item(), 2)
    return skillsync

def main():
    st.title("Sentiment Analysis API")

    input_text = st.text_input("Enter the text")
    if st.button("Analyze"):
        sentiment_score = analyze_sentiment(input_text)
        st.write("Sentiment Score:", sentiment_score)

if __name__ == '__main__':
    main()