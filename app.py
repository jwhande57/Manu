import streamlit as st
import pandas as pd
from textblob import TextBlob
import plotly.graph_objects as go
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

# Set page configuration
st.set_page_config(page_title="Mobile Network Feedback Analyzer", layout="centered")

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("network_feedback.csv")

df = load_data()

# Preprocess text: remove URLs, special characters, and emojis
def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text

# Perform sentiment analysis
def perform_sentiment_analysis(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Generate gauge chart for sentiment analysis
def generate_gauge_chart(score):
    sentiment_label = ""
    if score > 0.2:
        sentiment_label = "Positive 😀"
    elif score < -0.2:
        sentiment_label = "Negative 😞"
    else:
        sentiment_label = "Neutral 😐"
    
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


# Extract top keywords from the text using sklearn's built-in stop words
def extract_keywords(texts):
    vectorizer = CountVectorizer(stop_words='english')  # Use sklearn's built-in stop words
    word_counts = vectorizer.fit_transform(texts)
    keywords = vectorizer.get_feature_names_out()
    word_freq = word_counts.toarray().sum(axis=0)
    return dict(zip(keywords, word_freq))

# Generate a word cloud for visualization
def generate_word_cloud(keywords_dict):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(keywords_dict)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Display the dataset
st.subheader("Network Feedback Data")
st.write(df)

# Input form for text-based sentiment analysis
st.write("Enter a custom comment to analyze its sentiment:")

with st.form(key='sentiment_form'):
    user_input = st.text_area("Input text:", height=100)
    analyze_button = st.form_submit_button("Analyze Feedback")

# Custom comment analysis result
if analyze_button and user_input:
    # Preprocess the input text
    cleaned_input = preprocess_text(user_input)
    
    # Sentiment analysis
    score = perform_sentiment_analysis(cleaned_input)
    fig, sentiment_label = generate_gauge_chart(score)
    
    # Display sentiment analysis result
    st.plotly_chart(fig)
    st.write(f"Sentiment Score: {score:.2f} ({sentiment_label})")
    
    # Maintain analysis history
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    st.session_state.history.append({
        "text": user_input,
        "score": score,
        "label": sentiment_label
    })

# Add a multiselect option to select multiple ISPs for comparison
st.subheader("Compare ISPs")

selected_isps = st.multiselect(
    "Select ISPs to compare", 
    options=['Econet', 'Netone', 'Telone', 'Liquid', 'Utande'], 
    default=['Econet', 'Netone']
)

if selected_isps:
    # Filter the dataset based on the selected ISPs
    filtered_df = df[df['ISP'].isin(selected_isps)]
    
    # Preprocess comments for the filtered dataset
    filtered_df['Cleaned_Comment'] = filtered_df['Comment'].apply(preprocess_text)
    filtered_df['Sentiment'] = filtered_df['Cleaned_Comment'].apply(perform_sentiment_analysis)

    # Display sentiment distribution for each selected ISP
    st.subheader("Sentiment Distribution by ISP")

    sentiment_dist_by_isp = filtered_df.groupby(['ISP']).apply(
        lambda x: pd.Series({
            'Positive': (x['Sentiment'] > 0.2).sum(),
            'Neutral': ((x['Sentiment'] >= -0.2) & (x['Sentiment'] <= 0.2)).sum(),
            'Negative': (x['Sentiment'] < -0.2).sum()
        })
    ).reset_index()

    st.dataframe(sentiment_dist_by_isp)

    # Plot sentiment distribution comparison
    fig = go.Figure()

    for isp in selected_isps:
        isp_data = sentiment_dist_by_isp[sentiment_dist_by_isp['ISP'] == isp]
        fig.add_trace(go.Bar(
            x=['Positive', 'Neutral', 'Negative'],
            y=isp_data.iloc[0, 1:],
            name=isp
        ))

    fig.update_layout(barmode='group', title="Sentiment Comparison by ISP", xaxis_title="Sentiment", yaxis_title="Count")
    st.plotly_chart(fig)

    # Keyword extraction for each ISP
    st.subheader("Keyword Analysis by ISP")

    for isp in selected_isps:
        isp_comments = filtered_df[filtered_df['ISP'] == isp]['Cleaned_Comment'].tolist()
        if isp_comments:
            st.write(f"**Keywords for {isp}**")
            isp_keywords = extract_keywords(isp_comments)
            generate_word_cloud(isp_keywords)
        else:
            st.write(f"No comments available for {isp}")

# Show dataset with sentiment scores for the selected ISPs
st.subheader(f"Feedback Data for Selected ISPs")
st.write(filtered_df[['Date', 'ISP', 'Comment', 'Sentiment']])

# Show analysis history for custom comments
if 'history' in st.session_state and st.session_state.history:
    st.subheader("Custom Comment Analysis History")
    for idx, entry in enumerate(reversed(st.session_state.history)):
        st.write(f"**{idx + 1}.** Text: *{entry['text']}* | Sentiment: {entry['label']} (Score: {entry['score']:.2f})")
