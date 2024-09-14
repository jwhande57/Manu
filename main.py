# streamlit_app.py

import streamlit as st
from textblob import TextBlob
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(page_title="Sentiment Analysis", layout="centered")

# App Title
st.title("Sentiment Analyzer")

# Main function to perform sentiment analysis
def perform_sentiment_analysis(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity  # Returns a score between -1 (negative) to 1 (positive)

# Function to generate gauge chart
def generate_gauge_chart(score):
    # Define the sentiment label based on the score
    sentiment_label = ""
    if score > 0.2:
        sentiment_label = "Positive 😊"
    elif score < -0.2:
        sentiment_label = "Negative 😔"
    else:
        sentiment_label = "Neutral 😐"
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        gauge={'axis': {'range': [-1, 1]}, 
               'bar': {'color': "lightblue"},
               'steps': [
                   {'range': [-1, -0.2], 'color': "red"},
                   {'range': [-0.2, 0.2], 'color': "gray"},
                   {'range': [0.2, 1], 'color': "green"}],
              },
        title={'text': f"Sentiment: {sentiment_label}"}
    ))

    return fig, sentiment_label

# Input field for text and button for analysis
st.write("Enter the comment to analyze its sentiment:")

with st.form(key='sentiment_form'):
    user_input = st.text_area("Input text:", height=100)
    analyze_button = st.form_submit_button("Analyze Sentiment")

# Analysis result
if analyze_button and user_input:
    score = perform_sentiment_analysis(user_input)
    fig, sentiment_label = generate_gauge_chart(score)
    
    # Display the gauge chart
    st.plotly_chart(fig)
    
    # Display the sentiment score and label
    st.write(f"Sentiment Score: {score:.2f} ({sentiment_label})")
    
    # Maintain an analysis log
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    st.session_state.history.append({
        "text": user_input,
        "score": score,
        "label": sentiment_label
    })

# Show analysis history
if 'history' in st.session_state and st.session_state.history:
    st.subheader("Analysis History")
    for idx, entry in enumerate(reversed(st.session_state.history)):
        st.write(f"**{idx + 1}.** Text: *{entry['text']}* | Sentiment: {entry['label']} (Score: {entry['score']:.2f})")
