from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

def analyze_sentiment(headlines):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = [analyzer.polarity_scores(headline)['compound'] for headline in headlines]
    sentiment_dates = pd.date_range(start='2023-01-01', periods=len(sentiment_scores), freq='ME')
    sentiment_df = pd.DataFrame({'Date': sentiment_dates, 'Sentiment_Score': sentiment_scores})
    return sentiment_df
