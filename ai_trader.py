import ccxt
import pandas as pd
from utils.indicators import compute_rsi, compute_mfi, compute_wavetrend, compute_atr
from sentiment_signal import SentimentSignal

class AITrader:
    def __init__(self, api_key: str, api_secret: str, model=None):
        self.exchange = ccxt.mexc({
            'mx0vglG2gFRGKGNGsd': api_key,
            '3fa8e335eb4b40c0a4fde65edcff401c': api_secret,
        })
        self.model = model
        self.sentiment = SentimentSignal(api_key='685df1e87e5521.89870388')  # Inject sentiment class
        self.trade_log = []

    def fetch_data(self, symbol='BTC/USDT:USDT', timeframe='1h', limit=500) -> pd.DataFrame:
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df['rsi'] = compute_rsi(df['close'])
        df['mfi'] = compute_mfi(df)
        wt = compute_wavetrend(df)
        df['wt1'], df['wt2'] = wt['wt1'], wt['wt2']
        df['atr'] = compute_atr(df)
        return df

    def enrich_with_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        df['sentiment'] = self.sentiment.get_sentiment()
        print(f"[Sentiment Injected] → {df['sentiment'].iloc[-1]}")

        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        for lag in range(1, 4):
            df[f'return_lag_{lag}'] = df['close'].pct_change(lag)
        df = self.compute_indicators(df)
        df = self.enrich_with_sentiment(df)
        return df.dropna()

    def predict(self, df: pd.DataFrame):
        if not self.model:
            print("No model provided.")
            return None
        features = ['close', 'volume', 'rsi', 'mfi', 'wt1', 'wt2', 'atr',
                    'return_lag_1', 'return_lag_2', 'return_lag_3', 'sentiment']
        latest = df[features].iloc[[-1]]
        pred = self.model.predict(latest)
        return "BUY" if pred[0] == 1 else "SELL"

    def get_current_price(self, symbol='BTC/USDT:USDT') -> float:
        try:
            return self.exchange.fetch_ticker(symbol)['last']
        except Exception as e:
            print(f"Error fetching current price: {e}")
            return None