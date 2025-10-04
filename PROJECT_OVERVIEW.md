# 📊 Portfolio Crisis Simulator - Complete Overview

## What You Have Now

I've created a **complete, sophisticated Python backend** for your investment portfolio crisis simulator. Here's everything that's been built:

## 📂 Complete File Tree

```
HackRUF25/
├── .env                          # Your API keys (needs configuration)
├── requirements.txt              # Python dependencies (updated)
├── QUICKSTART.md                 # Quick start guide
├── SETUP_SUMMARY.md             # Detailed setup instructions
├── PROJECT_OVERVIEW.md          # This file
│
└── backend/
    ├── README.md                # Comprehensive backend documentation
    ├── env_template.txt         # Environment variables template
    │
    └── src/
        ├── main.py              # ⭐ FastAPI app entry point
        ├── config.py            # Configuration management
        ├── models.py            # Pydantic data models
        ├── data_loader.py       # Alpaca API integration (existing)
        ├── check_setup.py       # Setup verification script
        ├── test_api.py          # Comprehensive test suite
        │
        ├── api/
        │   ├── __init__.py
        │   └── routes.py        # ⭐ All API endpoints
        │
        └── services/
            ├── __init__.py
            ├── crisis_simulator.py    # ⭐ Crisis simulation engine
            └── market_predictor.py    # ⭐ Market prediction engine
```

## 🎯 Core Components

### 1. **main.py** - Application Entry Point
- FastAPI application setup
- CORS configuration
- Logging configuration
- Startup/shutdown events
- **Run with**: `python main.py`

### 2. **crisis_simulator.py** - The Heart of Crisis Simulation
**Lines of Code**: ~550

**Features**:
- 10 crisis types with unique sector impacts
- Sector-based impact modeling (11 sectors covered)
- Crisis timeline generation
- Portfolio metrics calculation (Sharpe, VaR, drawdown, volatility)
- Recommendations engine
- Stochastic price modeling with realistic volatility

**Crisis Types Implemented**:
1. Banking Crisis (2008-style) 
2. Tech Bubble Burst (Dot-com)
3. Energy Crisis (Oil shock)
4. Pandemic (COVID-style)
5. Geopolitical Crisis (War/conflict)
6. Inflation Spike
7. Interest Rate Shock
8. Housing Market Crash
9. Supply Chain Disruption
10. Currency Crisis

**Sector Coverage**:
- Technology
- Financial Services
- Healthcare
- Energy
- Consumer Cyclical
- Consumer Defensive
- Industrials
- Real Estate
- Utilities
- Basic Materials
- Communication Services

### 3. **market_predictor.py** - Advanced Market Prediction
**Lines of Code**: ~350

**Features**:
- Monte Carlo simulation (1000+ iterations)
- Technical indicators:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - SMA (Simple Moving Averages: 20, 50, 200 day)
  - EMA (Exponential Moving Averages)
  - Momentum indicators
- Market health indicators
- Individual asset predictions
- Risk scoring (0-100)
- Confidence intervals

### 4. **routes.py** - API Endpoints
**Lines of Code**: ~250

**Endpoints**:
```
GET  /api/v1/                    # Health check
GET  /api/v1/crisis-types        # List crisis types + descriptions
POST /api/v1/simulate-crisis     # Run crisis simulation
POST /api/v1/predict-performance # Predict portfolio performance
POST /api/v1/batch-simulate      # Run multiple crisis simulations
POST /api/v1/portfolio-value     # Get current portfolio value
```

### 5. **models.py** - Data Models
**Lines of Code**: ~150

**Models Defined**:
- `Portfolio` - Portfolio structure
- `PortfolioAsset` - Individual asset
- `CrisisType` - Enum of crisis types
- `CrisisSimulationRequest` - Crisis sim input
- `CrisisSimulationResult` - Crisis sim output
- `MarketPredictionRequest` - Prediction input
- `MarketPredictionResult` - Prediction output
- `PortfolioMetrics` - Risk metrics
- `AssetImpact` - Individual asset impact

## 📊 Example API Flows

### Crisis Simulation Flow
```
User Input → API Endpoint → Crisis Simulator
                               ↓
                        Fetch Current Prices (Alpaca/yfinance)
                               ↓
                        Map Stocks to Sectors
                               ↓
                        Apply Crisis Impact Matrix
                               ↓
                        Generate Timeline (stochastic)
                               ↓
                        Calculate Metrics
                               ↓
                        Generate Recommendations
                               ↓
                        Return JSON Response → Frontend
```

### Market Prediction Flow
```
User Input → API Endpoint → Market Predictor
                               ↓
                        Fetch Historical Data (1 year)
                               ↓
                        Calculate Technical Indicators
                               ↓
                        Run Monte Carlo (1000 simulations)
                               ↓
                        Calculate Confidence Intervals
                               ↓
                        Predict Individual Assets
                               ↓
                        Assess Market Health
                               ↓
                        Return JSON Response → Frontend
```

## 🔬 Technical Sophistication

### Crisis Impact Matrix
Each crisis has a unique impact profile across sectors:

```python
# Example: Tech Bubble
{
    "Technology": -0.45,           # Tech stocks hit hardest
    "Communication Services": -0.30,
    "Healthcare": 0.05,            # Defensive sector benefits
    "Utilities": 0.08,             # Flight to safety
    ...
}
```

### Monte Carlo Simulation
```
For each simulation (1000x):
  - Sample from historical return distribution
  - Apply random walk
  - Calculate portfolio value at each time step
  
Aggregate:
  - 5th percentile (worst case)
  - 50th percentile (expected)
  - 95th percentile (best case)
```

### Risk Metrics
- **Sharpe Ratio**: Risk-adjusted returns
- **Value at Risk (VaR)**: Expected loss in worst 5% of cases
- **Max Drawdown**: Largest peak-to-trough decline
- **Volatility**: Annualized standard deviation
- **Beta**: Sensitivity to market movements

## 💻 Usage Examples

### Example 1: Simulate Tech Bubble on Tech Portfolio
```bash
curl -X POST http://localhost:8000/api/v1/simulate-crisis \
  -H "Content-Type: application/json" \
  -d '{
    "portfolio": {
      "name": "Tech Growth",
      "assets": [
        {"symbol": "AAPL", "quantity": 10},
        {"symbol": "MSFT", "quantity": 15},
        {"symbol": "GOOGL", "quantity": 5},
        {"symbol": "NVDA", "quantity": 8}
      ],
      "cash": 5000.0
    },
    "crisis_type": "tech_bubble",
    "intensity": 2.5,
    "duration_days": 60
  }'
```

**Expected Output**:
- Significant losses (tech stocks drop 30-45%)
- Daily timeline showing crash progression
- Warnings about tech concentration
- Recommendations to diversify

### Example 2: Predict Balanced Portfolio
```bash
curl -X POST http://localhost:8000/api/v1/predict-performance \
  -H "Content-Type: application/json" \
  -d '{
    "portfolio": {
      "name": "Balanced",
      "assets": [
        {"symbol": "AAPL", "quantity": 5},
        {"symbol": "JPM", "quantity": 10},
        {"symbol": "JNJ", "quantity": 8},
        {"symbol": "XOM", "quantity": 12}
      ],
      "cash": 10000.0
    },
    "forecast_days": 30,
    "confidence_level": 0.95
  }'
```

**Expected Output**:
- Expected return with confidence intervals
- Risk score
- Technical analysis for each stock
- Market health indicators

## 📈 Response Examples

### Crisis Simulation Response
```json
{
  "crisis_type": "tech_bubble",
  "intensity": 2.5,
  "portfolio_initial_value": 75000.00,
  "portfolio_final_value": 52500.00,
  "total_loss_pct": -30.0,
  "total_loss_usd": 22500.00,
  
  "asset_impacts": [
    {
      "symbol": "AAPL",
      "current_price": 180.00,
      "simulated_price": 126.00,
      "price_change_pct": -30.0,
      "sector": "Technology",
      "impact_reason": "Technology sector declined significantly..."
    }
  ],
  
  "metrics_before": {
    "volatility": 0.25,
    "sharpe_ratio": 1.5,
    "max_drawdown": -0.15,
    "value_at_risk": -1200.00
  },
  
  "metrics_after": {
    "volatility": 0.42,
    "sharpe_ratio": 0.3,
    "max_drawdown": -0.35,
    "value_at_risk": -3500.00
  },
  
  "daily_values": [
    {"date": "2025-10-04", "value": 75000, "day": 1},
    {"date": "2025-10-05", "value": 73200, "day": 2},
    ...
  ],
  
  "recommendations": [
    "⚠️ SEVERE RISK: Portfolio highly vulnerable to tech bubble",
    "Consider reducing exposure to tech stocks...",
    "Most vulnerable: AAPL, MSFT, GOOGL",
    "Increase allocation to defensive sectors"
  ]
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
  
  "asset_predictions": [
    {
      "symbol": "AAPL",
      "current_price": 180.00,
      "predicted_price": 189.00,
      "expected_return_pct": 5.0,
      "confidence": 82.0,
      "trend": "bullish",
      "rsi": 55.2
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

## 🚀 Getting Started

### Step 1: Install Dependencies
```bash
# Free up disk space first
pip cache purge

# Install packages
pip install fastapi uvicorn pydantic python-dotenv requests pandas numpy yfinance alpaca-trade-api scikit-learn
```

### Step 2: Configure API Keys
```bash
# Edit .env file
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

Get free keys at: **https://alpaca.markets/**

### Step 3: Run Server
```bash
cd backend/src
python main.py
```

Server starts at: **http://localhost:8000**

### Step 4: Test
```bash
# Check health
curl http://localhost:8000/api/v1/

# Run full test suite
python test_api.py
```

### Step 5: View Documentation
Open browser: **http://localhost:8000/docs**

## 🎨 Frontend Integration

### React Example
```javascript
const simulateCrisis = async () => {
  const response = await fetch('http://localhost:8000/api/v1/simulate-crisis', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      portfolio: {
        assets: [
          { symbol: "AAPL", quantity: 10 },
          { symbol: "MSFT", quantity: 5 }
        ],
        cash: 5000
      },
      crisis_type: "tech_bubble",
      intensity: 2.0,
      duration_days: 60
    })
  });
  
  const data = await response.json();
  console.log('Total Loss:', data.total_loss_pct + '%');
  console.log('Recommendations:', data.recommendations);
};
```

## 📊 Statistics

- **Total Files Created**: 11
- **Total Lines of Code**: ~1,800
- **API Endpoints**: 6
- **Crisis Types**: 10
- **Sectors Covered**: 11
- **Technical Indicators**: 10+
- **Risk Metrics**: 6

## 🏆 Why This Is Complex

1. **Multi-Factor Modeling**: Combines sector analysis, technical indicators, and stochastic processes
2. **Real Market Data**: Integrates with professional trading API (Alpaca)
3. **Advanced Math**: Monte Carlo, Sharpe ratios, correlation matrices
4. **Production Quality**: Proper error handling, validation, logging
5. **Scalable Architecture**: Clean separation, easily extensible
6. **Comprehensive**: 10 different crisis scenarios, each with unique sector impacts

## 🎯 Ready for Hackathon!

You now have:
✅ Complete backend with all crisis simulations  
✅ Advanced market prediction engine  
✅ Real-time stock data integration  
✅ Professional financial metrics  
✅ Interactive API documentation  
✅ Test suite  
✅ Comprehensive guides  

**Next**: Install dependencies → Configure API keys → Run server → Build frontend!

---

**Questions?** Check:
- `SETUP_SUMMARY.md` for detailed setup
- `QUICKSTART.md` for quick start guide
- `backend/README.md` for backend documentation
- http://localhost:8000/docs for API docs (when running)

**Good luck with your hackathon! 🚀**
