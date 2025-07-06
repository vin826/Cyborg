import ccxt
import pandas as pd

def load_ohlcv(symbol='BTC/USDT.P', timeframe='1h', limit=500, api_key='', api_secret=''):
    exchange = ccxt.mexc({
        'mx0vglG2gFRGKGNGsd': api_key,
        '3fa8e335eb4b40c0a4fde65edcff401c': api_secret
    })

    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"[Data Loader] Error fetching OHLCV: {e}")
        return pd.DataFrame()