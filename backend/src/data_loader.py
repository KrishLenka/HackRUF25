from typing import List
import pandas as pd
import os

# Prefer Alpaca if environment provides keys; otherwise fallback to yfinance
ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

def get_historical_prices_yf(symbols: List[str], start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    import yfinance as yf
    # returns DataFrame with columns like 'AAPL', 'MSFT' representing adjusted close
    out = {}
    for s in symbols:
        df = yf.download(s, start=start, end=end, interval=interval, progress=False)
        if df.empty:
            raise RuntimeError(f"No data for {s}")
        # Use 'Adj Close' if available else 'Close'
        col = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
        out[s] = col
    prices = pd.DataFrame(out)
    prices.index = pd.to_datetime(prices.index)
    return prices

def get_historical_prices(symbols: List[str], start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    if ALPACA_KEY and ALPACA_SECRET:
        try:
            from alpaca_trade_api.rest import REST
            client = REST(ALPACA_KEY, ALPACA_SECRET, ALPACA_BASE, api_version='v2')
            # Alpaca get_bars returns different structures; we'll fetch daily close for simplicity
            frames = []
            for s in symbols:
                bars = client.get_bars(s, timeframe=interval, start=start, end=end).df
                if bars.empty:
                    raise RuntimeError(f"No Alpaca data for {s}")
                # normalize index, use close
                series = bars['close']
                series.name = s
                frames.append(series)
            df = pd.concat(frames, axis=1)
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            print("Alpaca fetch failed, falling back to yfinance:", e)
    return get_historical_prices_yf(symbols, start, end, interval)
