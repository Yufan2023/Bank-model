import unittest
from src.sentiment_analysis import analyze_sentiment

class TestSentimentAnalysis(unittest.TestCase):
    def test_sentiment_score(self):
        sample_headline = "Positive economic news boosts market"
        score = analyze_sentiment(sample_headline)
        self.assertTrue(isinstance(score, float))

if __name__ == '__main__':
    unittest.main()
