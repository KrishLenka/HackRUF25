import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from models import Portfolio, MarketPredictionResult
from data_loader import get_historical_prices


class MarketPredictor:
    """
    Predicts portfolio performance based on current market conditions using
    time series analysis, machine learning, and Monte Carlo simulations.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.rng = np.random.RandomState(42)
    
    def fetch_market_data(
        self, 
        symbols: List[str], 
        lookback_days: int = 365
    ) -> pd.DataFrame:
        """Fetch historical market data for analysis."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        try:
            prices_df = get_historical_prices(
                symbols,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d")
            )
            return prices_df
        except Exception as e:
            print(f"Error fetching market data: {e}")
            # Return dummy data for demo purposes
            dates = pd.date_range(end=end_date, periods=lookback_days, freq='D')
            data = {symbol: np.random.randn(lookback_days).cumsum() + 100 
                   for symbol in symbols}
            return pd.DataFrame(data, index=dates)
    
    def calculate_technical_indicators(self, prices: pd.Series) -> pd.DataFrame:
        """Calculate technical indicators for a price series."""
        df = pd.DataFrame()
        df['price'] = prices
        
        # Moving averages
        df['sma_20'] = prices.rolling(window=20).mean()
        df['sma_50'] = prices.rolling(window=50).mean()
        df['sma_200'] = prices.rolling(window=200).mean()
        
        # Exponential moving average
        df['ema_12'] = prices.ewm(span=12).mean()
        df['ema_26'] = prices.ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # RSI (Relative Strength Index)
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = prices.rolling(window=20).mean()
        bb_std = prices.rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Volatility
        df['volatility'] = prices.pct_change().rolling(window=30).std()
        
        # Momentum
        df['momentum'] = prices.pct_change(periods=10)
        
        return df.dropna()
    
    def calculate_market_indicators(self, prices_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate overall market health indicators."""
        returns = prices_df.pct_change().dropna()
        
        indicators = {
            "market_volatility": float(returns.std().mean() * np.sqrt(252)),
            "average_momentum": float(returns.mean().mean() * 252),
            "correlation_to_market": float(returns.corr().values[np.triu_indices_from(returns.corr().values, k=1)].mean()),
            "trend_strength": self._calculate_trend_strength(prices_df),
            "risk_sentiment": self._calculate_risk_sentiment(returns),
        }
        
        return indicators
    
    def _calculate_trend_strength(self, prices_df: pd.DataFrame) -> float:
        """
        Calculate overall market trend strength.
        Returns value between -1 (strong downtrend) and 1 (strong uptrend).
        """
        trends = []
        for col in prices_df.columns:
            prices = prices_df[col].dropna()
            if len(prices) > 50:
                sma_20 = prices.rolling(window=20).mean().iloc[-1]
                sma_50 = prices.rolling(window=50).mean().iloc[-1]
                current = prices.iloc[-1]
                
                # Compare current price to moving averages
                if sma_20 > sma_50 and current > sma_20:
                    trends.append(1)  # Uptrend
                elif sma_20 < sma_50 and current < sma_20:
                    trends.append(-1)  # Downtrend
                else:
                    trends.append(0)  # Neutral
        
        return float(np.mean(trends)) if trends else 0.0
    
    def _calculate_risk_sentiment(self, returns: pd.DataFrame) -> float:
        """
        Calculate market risk sentiment.
        Returns value between 0 (low risk) and 1 (high risk).
        """
        # Use volatility and skewness as proxies for risk sentiment
        volatility = returns.std().mean()
        skewness = returns.skew().mean()  # Negative skew indicates more downside risk
        
        # Normalize to 0-1 scale
        risk_score = min(1.0, volatility * 10 + abs(min(0, skewness)) * 0.2)
        return float(risk_score)
    
    def monte_carlo_simulation(
        self,
        current_value: float,
        daily_returns: np.ndarray,
        forecast_days: int,
        n_simulations: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Monte Carlo simulation to predict future portfolio values.
        Returns: (simulated_paths, daily_statistics)
        """
        # Calculate statistics from historical returns
        mean_return = np.mean(daily_returns)
        std_return = np.std(daily_returns)
        
        # Generate random returns based on historical statistics
        simulated_paths = np.zeros((n_simulations, forecast_days))
        simulated_paths[:, 0] = current_value
        
        for day in range(1, forecast_days):
            random_returns = self.rng.normal(mean_return, std_return, n_simulations)
            simulated_paths[:, day] = simulated_paths[:, day - 1] * (1 + random_returns)
        
        # Calculate statistics for each day
        daily_stats = np.percentile(simulated_paths, [5, 50, 95], axis=0)
        
        return simulated_paths, daily_stats
    
    def predict_individual_assets(
        self,
        prices_df: pd.DataFrame,
        forecast_days: int
    ) -> List[Dict[str, float]]:
        """Predict individual asset performance using technical analysis."""
        predictions = []
        
        for symbol in prices_df.columns:
            try:
                prices = prices_df[symbol].dropna()
                
                # Calculate technical indicators
                indicators = self.calculate_technical_indicators(prices)
                
                if len(indicators) < 30:
                    continue
                
                # Use simple momentum and trend for prediction
                current_price = float(prices.iloc[-1])
                sma_50 = float(indicators['sma_50'].iloc[-1])
                rsi = float(indicators['rsi'].iloc[-1])
                momentum = float(indicators['momentum'].iloc[-1])
                
                # Simple prediction model
                trend_factor = 1.0
                if current_price > sma_50:
                    trend_factor += 0.01 * (forecast_days / 30)  # Slight uptrend
                else:
                    trend_factor -= 0.01 * (forecast_days / 30)  # Slight downtrend
                
                # Adjust based on RSI (overbought/oversold)
                if rsi > 70:  # Overbought
                    trend_factor -= 0.02
                elif rsi < 30:  # Oversold
                    trend_factor += 0.02
                
                predicted_price = current_price * trend_factor
                expected_return = ((predicted_price - current_price) / current_price) * 100
                
                predictions.append({
                    "symbol": symbol,
                    "current_price": current_price,
                    "predicted_price": predicted_price,
                    "expected_return_pct": expected_return,
                    "confidence": min(100, 70 + abs(momentum) * 1000),  # Higher confidence with stronger momentum
                    "trend": "bullish" if trend_factor > 1 else "bearish",
                    "rsi": rsi
                })
            
            except Exception as e:
                print(f"Error predicting {symbol}: {e}")
                continue
        
        return predictions
    
    def predict(
        self,
        portfolio: Portfolio,
        forecast_days: int,
        confidence_level: float = 0.95
    ) -> MarketPredictionResult:
        """
        Main prediction function that forecasts portfolio performance.
        """
        # Get portfolio symbols
        symbols = [asset.symbol for asset in portfolio.assets]
        
        # Fetch historical data
        prices_df = self.fetch_market_data(symbols, lookback_days=365)
        
        # Calculate current portfolio value
        current_prices = prices_df.iloc[-1].to_dict()
        current_value = portfolio.cash
        
        for asset in portfolio.assets:
            current_value += asset.quantity * current_prices.get(asset.symbol, 0)
        
        # Calculate portfolio returns
        returns_df = prices_df.pct_change().dropna()
        
        # Calculate weighted portfolio returns
        weights = []
        for asset in portfolio.assets:
            asset_value = asset.quantity * current_prices.get(asset.symbol, 0)
            weights.append(asset_value / current_value)
        
        portfolio_returns = (returns_df * weights).sum(axis=1).values
        
        # Run Monte Carlo simulation
        simulated_paths, daily_stats = self.monte_carlo_simulation(
            current_value,
            portfolio_returns,
            forecast_days,
            n_simulations=1000
        )
        
        # Extract predictions
        predicted_lower = float(daily_stats[0, -1])  # 5th percentile
        predicted_mean = float(daily_stats[1, -1])   # 50th percentile (median)
        predicted_upper = float(daily_stats[2, -1])  # 95th percentile
        
        expected_return_pct = ((predicted_mean - current_value) / current_value) * 100
        
        # Calculate risk score (0-100)
        downside_risk = max(0, current_value - predicted_lower) / current_value
        volatility_risk = returns_df.std().mean()
        risk_score = min(100, (downside_risk * 50 + volatility_risk * 500))
        
        # Generate daily predictions
        predicted_daily_values = []
        start_date = datetime.now()
        
        for day in range(forecast_days):
            date = start_date + timedelta(days=day)
            predicted_daily_values.append({
                "date": date.strftime("%Y-%m-%d"),
                "mean": float(daily_stats[1, day]),
                "lower": float(daily_stats[0, day]),
                "upper": float(daily_stats[2, day]),
                "day": day + 1
            })
        
        # Predict individual assets
        asset_predictions = self.predict_individual_assets(prices_df, forecast_days)
        
        # Calculate market indicators
        market_indicators = self.calculate_market_indicators(prices_df)
        
        return MarketPredictionResult(
            portfolio_current_value=current_value,
            predicted_value_mean=predicted_mean,
            predicted_value_upper=predicted_upper,
            predicted_value_lower=predicted_lower,
            confidence_level=confidence_level,
            expected_return_pct=expected_return_pct,
            risk_score=risk_score,
            predicted_daily_values=predicted_daily_values,
            asset_predictions=asset_predictions,
            market_indicators=market_indicators
        )
