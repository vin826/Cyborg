import requests
import json
class SentimentSignal:
    def __init__(self, api_key, source='eodhd'):
        
        self.api_key = api_key
        self.source = source

    def get_sentiment(self, symbol='BTC'):
        try:
            url = f"https://eodhd.com/api/news?s={symbol}&api_token={self.api_key}&fmt=json"
            response = requests.get(url)
            data = response.json()

            # Debug: show top-level data
            print("[API Raw]:", data)

            if isinstance(data, list) and len(data) > 0:
                # Grab the sentiment dict from first article
                sentiment_data = data[0].get("sentiment", {})
                polarity = sentiment_data.get("polarity", 0.0)
                print(f"[Extracted Polarity]: {polarity}")
                return round(float(polarity), 3)
            else:
                print("[Sentiment]: Empty or malformed list.")
                return 0.0

        except Exception as e:
            print(f"[Sentiment API Error]: {e}")
            return 0.0
            
sentiment = SentimentSignal(api_key='685df1e87e5521.89870388')


print(sentiment.get_sentiment(symbol='BTC'))
