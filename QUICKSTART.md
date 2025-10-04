# 🚀 Quick Start Guide - Portfolio Crisis Simulator

## Setup Instructions

### 1. Set up Alpaca API Keys

Create a `.env` file in the root directory:

```bash
# Copy the template
cp backend/env_template.txt .env

# Edit the file and add your Alpaca API keys from https://alpaca.markets/
# For hackathon/testing, use paper trading keys (free)
```

Your `.env` should look like:
```
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
DEBUG=True
```

### 2. Install Dependencies

```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (most should already be installed)
pip install fastapi uvicorn pydantic python-dotenv requests alpaca-trade-api yfinance
```

### 3. Run the Backend

```bash
cd backend/src
python main.py
```

The API will start at: **http://localhost:8000**

- API Docs: http://localhost:8000/docs
- Alternative Docs: http://localhost:8000/redoc

### 4. Test the API

In a new terminal:

```bash
cd backend/src
python test_api.py
```

Or test with curl:

```bash
# Health check
curl http://localhost:8000/api/v1/

# Get crisis types
curl http://localhost:8000/api/v1/crisis-types
```

## Quick API Examples

### 1. Simulate a Tech Bubble Crisis

```bash
curl -X POST http://localhost:8000/api/v1/simulate-crisis \
  -H "Content-Type: application/json" \
  -d '{
    "portfolio": {
      "name": "Tech Portfolio",
      "assets": [
        {"symbol": "AAPL", "quantity": 10},
        {"symbol": "MSFT", "quantity": 5},
        {"symbol": "GOOGL", "quantity": 3}
      ],
      "cash": 5000.0
    },
    "crisis_type": "tech_bubble",
    "intensity": 2.0,
    "duration_days": 60
  }'
```

### 2. Predict Portfolio Performance

```bash
curl -X POST http://localhost:8000/api/v1/predict-performance \
  -H "Content-Type: application/json" \
  -d '{
    "portfolio": {
      "name": "Balanced Portfolio",
      "assets": [
        {"symbol": "AAPL", "quantity": 5},
        {"symbol": "JPM", "quantity": 10}
      ],
      "cash": 10000.0
    },
    "forecast_days": 30,
    "confidence_level": 0.95
  }'
```

### 3. Get Portfolio Value

```bash
curl -X POST http://localhost:8000/api/v1/portfolio-value \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Portfolio",
    "assets": [
      {"symbol": "AAPL", "quantity": 10}
    ],
    "cash": 1000.0
  }'
```

## Available Crisis Types

1. **banking_crisis** - 2008-style financial meltdown
2. **tech_bubble** - Dot-com bubble burst
3. **energy_crisis** - Oil shock scenario
4. **pandemic** - COVID-19 style disruption
5. **geopolitical** - War/conflict scenario
6. **inflation_spike** - Rapid inflation
7. **interest_rate_shock** - Fed rate hikes
8. **housing_crash** - Real estate collapse
9. **supply_chain** - Global supply disruption
10. **currency_crisis** - Currency devaluation

## Crisis Intensity Guide

- **0.1 - 0.5**: Mild (minor market correction)
- **0.5 - 1.5**: Moderate (typical recession)
- **1.5 - 3.0**: Severe (major crisis like 2008)
- **3.0 - 5.0**: Extreme (worst-case scenario)
- **5.0 - 10.0**: Catastrophic (unprecedented)

## Response Structure

### Crisis Simulation Response

```json
{
  "crisis_type": "tech_bubble",
  "intensity": 2.0,
  "portfolio_initial_value": 50000.0,
  "portfolio_final_value": 35000.0,
  "total_loss_pct": -30.0,
  "total_loss_usd": 15000.0,
  "asset_impacts": [
    {
      "symbol": "AAPL",
      "current_price": 150.0,
      "simulated_price": 105.0,
      "price_change_pct": -30.0,
      "sector": "Technology",
      "impact_reason": "Tech sector declined significantly..."
    }
  ],
  "metrics_before": {
    "volatility": 0.25,
    "sharpe_ratio": 1.5,
    "max_drawdown": -0.15,
    ...
  },
  "metrics_after": {...},
  "daily_values": [...],
  "recommendations": [
    "⚠️ SEVERE RISK: Portfolio highly vulnerable...",
    "Consider diversifying away from tech stocks..."
  ]
}
```

### Market Prediction Response

```json
{
  "portfolio_current_value": 50000.0,
  "predicted_value_mean": 52500.0,
  "predicted_value_upper": 58000.0,
  "predicted_value_lower": 47000.0,
  "confidence_level": 0.95,
  "expected_return_pct": 5.0,
  "risk_score": 45.2,
  "predicted_daily_values": [...],
  "asset_predictions": [
    {
      "symbol": "AAPL",
      "current_price": 150.0,
      "predicted_price": 157.5,
      "expected_return_pct": 5.0,
      "confidence": 85.0,
      "trend": "bullish",
      "rsi": 55.0
    }
  ],
  "market_indicators": {
    "market_volatility": 0.18,
    "average_momentum": 0.12,
    "correlation_to_market": 0.75,
    "trend_strength": 0.6,
    "risk_sentiment": 0.45
  }
}
```

## Portfolio Metrics Explained

- **Volatility**: Annual price variation (lower is more stable)
- **Sharpe Ratio**: Risk-adjusted return (higher is better, >1 is good)
- **Max Drawdown**: Largest peak-to-trough decline
- **Value at Risk (VaR)**: Expected loss in worst 5% of cases
- **Beta**: Sensitivity to market movements (1.0 = market average)
- **RSI**: Relative Strength Index (70+ overbought, 30- oversold)

## Troubleshooting

**Error: "No module named 'dotenv'"**
```bash
pip install python-dotenv
```

**Error: "Could not connect to API"**
- Make sure the server is running: `cd backend/src && python main.py`
- Check the URL: http://localhost:8000

**Error: Alpaca API errors**
- Check your API keys in `.env`
- System will automatically fall back to Yahoo Finance

**Slow responses**
- First request may be slow (fetching historical data)
- Subsequent requests are faster

## Next Steps

1. **Integrate with Frontend**: Use the API endpoints from your React/Vue/Angular app
2. **Add More Features**: Extend crisis scenarios or add new metrics
3. **Deploy**: Use Docker or deploy to a cloud platform

## Project Structure

```
backend/src/
├── main.py                    # FastAPI app entry point
├── config.py                  # Configuration
├── models.py                  # Pydantic models
├── data_loader.py             # Alpaca/yfinance integration
├── api/
│   └── routes.py              # API endpoints
└── services/
    ├── crisis_simulator.py   # Crisis simulation engine
    └── market_predictor.py   # Market prediction engine
```

## Features

✅ 10 different crisis scenarios  
✅ Real-time stock data from Alpaca API  
✅ Monte Carlo simulations (1000+ iterations)  
✅ Technical analysis (RSI, MACD, Bollinger Bands)  
✅ Portfolio risk metrics  
✅ Sector-based impact modeling  
✅ Actionable recommendations  
✅ Interactive API documentation  

---

**Built for HackRU F25 🚀**

Need help? Check:
- API Docs: http://localhost:8000/docs
- Backend README: backend/README.md
