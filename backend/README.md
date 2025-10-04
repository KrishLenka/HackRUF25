# Portfolio Crisis Simulator - Backend

A sophisticated Python backend for simulating portfolio performance during market crises and predicting future returns using the Alpaca API.

## 🚀 Features

### Crisis Simulation
Simulate how your portfolio performs during 10 different crisis scenarios:
- **Banking Crisis** - Systemic banking failures (e.g., 2008)
- **Tech Bubble Burst** - Tech valuation collapse (e.g., Dot-com 2000)
- **Energy Crisis** - Energy price shocks (e.g., 1970s Oil Crisis)
- **Pandemic** - Global health crises (e.g., COVID-19)
- **Geopolitical Crisis** - International conflicts
- **Inflation Spike** - Rapid price increases
- **Interest Rate Shock** - Sudden rate hikes
- **Housing Market Crash** - Real estate collapse
- **Supply Chain Disruption** - Global supply issues
- **Currency Crisis** - Currency devaluation

### Market Prediction
- Monte Carlo simulations with 1000+ iterations
- Technical indicator analysis (RSI, MACD, Bollinger Bands)
- Confidence intervals for predictions
- Risk scoring and portfolio metrics
- Individual asset predictions

### Advanced Analytics
- Portfolio risk metrics (Sharpe ratio, VaR, max drawdown, volatility)
- Sector-based impact analysis
- Real-time data from Alpaca API (with yfinance fallback)
- Timeline visualization of crisis impacts
- Actionable recommendations

## 📦 Project Structure

```
backend/
├── src/
│   ├── main.py                    # FastAPI application entry point
│   ├── config.py                  # Configuration and settings
│   ├── models.py                  # Pydantic models for API
│   ├── data_loader.py             # Alpaca API integration
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py              # API endpoints
│   ├── services/
│   │   ├── __init__.py
│   │   ├── crisis_simulator.py   # Crisis simulation engine
│   │   └── market_predictor.py   # Market prediction engine
│   └── test_api.py                # API test script
├── .env.example                   # Environment variables template
└── README.md                      # This file
```

## 🛠️ Installation

### 1. Clone the repository
```bash
cd backend
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r ../requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
# Edit .env and add your Alpaca API credentials
```

Get your Alpaca API keys from: https://alpaca.markets/

## 🚀 Running the Server

### Development mode (with auto-reload)
```bash
cd src
python main.py
```

Or using uvicorn directly:
```bash
cd src
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API Base URL**: http://localhost:8000/api/v1
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📚 API Usage Examples

### 1. Get Available Crisis Types
```bash
curl http://localhost:8000/api/v1/crisis-types
```

### 2. Simulate a Crisis
```bash
curl -X POST http://localhost:8000/api/v1/simulate-crisis \
  -H "Content-Type: application/json" \
  -d '{
    "portfolio": {
      "name": "My Portfolio",
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

### 3. Predict Portfolio Performance
```bash
curl -X POST http://localhost:8000/api/v1/predict-performance \
  -H "Content-Type: application/json" \
  -d '{
    "portfolio": {
      "name": "Balanced Portfolio",
      "assets": [
        {"symbol": "AAPL", "quantity": 5},
        {"symbol": "JPM", "quantity": 10},
        {"symbol": "JNJ", "quantity": 8}
      ],
      "cash": 10000.0
    },
    "forecast_days": 30,
    "confidence_level": 0.95
  }'
```

### 4. Get Portfolio Value
```bash
curl -X POST http://localhost:8000/api/v1/portfolio-value \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Portfolio",
    "assets": [
      {"symbol": "AAPL", "quantity": 10}
    ],
    "cash": 1000.0
  }'
```

### 5. Batch Simulation (Multiple Crises)
```bash
curl -X POST "http://localhost:8000/api/v1/batch-simulate?intensity=1.5&duration_days=30" \
  -H "Content-Type: application/json" \
  -d '{
    "portfolio": {
      "name": "Diverse Portfolio",
      "assets": [
        {"symbol": "AAPL", "quantity": 5},
        {"symbol": "JPM", "quantity": 10}
      ],
      "cash": 5000.0
    },
    "crisis_types": ["banking_crisis", "tech_bubble", "pandemic"]
  }'
```

## 🧪 Testing

Run the comprehensive test suite:
```bash
cd src
python test_api.py
```

This will test all endpoints and display formatted results.

## 🔧 Configuration

Edit `.env` file to configure:

```env
# Alpaca API (get from https://alpaca.markets/)
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Application
DEBUG=True

# Optional
DATABASE_URL=sqlite:///./portfolio_simulator.db
REDIS_HOST=localhost
REDIS_PORT=6379
```

## 📊 Response Models

### Crisis Simulation Response
```json
{
  "crisis_type": "tech_bubble",
  "intensity": 2.0,
  "portfolio_initial_value": 50000.00,
  "portfolio_final_value": 35000.00,
  "total_loss_pct": -30.0,
  "total_loss_usd": 15000.00,
  "asset_impacts": [...],
  "metrics_before": {...},
  "metrics_after": {...},
  "daily_values": [...],
  "recommendations": [...]
}
```

### Market Prediction Response
```json
{
  "portfolio_current_value": 50000.00,
  "predicted_value_mean": 52500.00,
  "predicted_value_upper": 58000.00,
  "predicted_value_lower": 47000.00,
  "confidence_level": 0.95,
  "expected_return_pct": 5.0,
  "risk_score": 45.2,
  "predicted_daily_values": [...],
  "asset_predictions": [...],
  "market_indicators": {...}
}
```

## 🎯 Crisis Parameters

### Intensity Levels
- `0.1 - 0.5`: Mild crisis
- `0.5 - 1.5`: Moderate crisis
- `1.5 - 3.0`: Severe crisis
- `3.0 - 5.0`: Extreme crisis
- `5.0 - 10.0`: Catastrophic crisis

### Duration
- Short-term: 7-30 days
- Medium-term: 30-90 days
- Long-term: 90-365 days

## 🔍 Technical Details

### Data Sources
1. **Primary**: Alpaca Markets API (real-time and historical data)
2. **Fallback**: Yahoo Finance (yfinance)

### Algorithms Used
- **Crisis Simulation**: Sector-based impact modeling with stochastic volatility
- **Market Prediction**: Monte Carlo + Technical Analysis + Machine Learning
- **Risk Metrics**: Industry-standard formulas (Sharpe, VaR, drawdown)

### Supported Symbols
Any stock symbol available on Alpaca/Yahoo Finance, including:
- Individual stocks (AAPL, MSFT, GOOGL, etc.)
- ETFs (SPY, QQQ, etc.)
- Sector coverage across all major industries

## 🚨 Error Handling

The API includes comprehensive error handling:
- Invalid symbols → Returns error with details
- API rate limits → Automatic fallback to yfinance
- Invalid parameters → Validation errors with clear messages
- Server errors → Logged with full context

## 📈 Performance

- Crisis simulation: ~1-3 seconds per request
- Market prediction: ~2-5 seconds (includes 1000 Monte Carlo runs)
- Batch operations: Parallel processing for multiple crises

## 🤝 For Hackathon Judges

This backend demonstrates:
- ✅ Complex financial modeling and simulations
- ✅ Real-time data integration (Alpaca API)
- ✅ Advanced analytics (ML, Monte Carlo, technical analysis)
- ✅ Production-ready code structure
- ✅ Comprehensive API documentation
- ✅ Error handling and fallbacks
- ✅ Scalable architecture

## 📝 Next Steps

To integrate with frontend:
1. Use the API endpoints from your React/Vue/Angular app
2. All responses are JSON with detailed data
3. CORS is enabled for development
4. Interactive API docs available at `/docs`

## 🐛 Troubleshooting

**API not responding?**
- Check if server is running: `curl http://localhost:8000/api/v1/`
- Check logs in terminal

**Alpaca API errors?**
- Verify API keys in `.env`
- Check if using paper trading URL
- System falls back to yfinance automatically

**Slow responses?**
- First request may be slow (fetching historical data)
- Subsequent requests use cached data
- Consider adding Redis for production

## 📄 License

MIT License - Built for HackRU F25

---

**Happy Hacking! 🚀**
