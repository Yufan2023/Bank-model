from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

def analyze_sentiment(news_headlines):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = [analyzer.polarity_scores(headline)['compound'] for headline in news_headlines]
    sentiment_dates = pd.date_range(start='2023-01-01', periods=len(sentiment_scores), freq='ME')
    sentiment_df = pd.DataFrame({'Date': sentiment_dates, 'Sentiment_Score': sentiment_scores})
    return sentiment_df

def merge_sentiment_with_data(data, sentiment_df):
    # Ensure date columns are in the same format
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date']).dt.tz_localize(None)

    # Merge sentiment data with the main dataset
    data = pd.merge(data, sentiment_df, on='Date', how='left')
    data['Sentiment_Score'].fillna(0, inplace=True)  # Fill missing sentiment scores with 0
    return data
