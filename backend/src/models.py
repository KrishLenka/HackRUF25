from typing import List, Dict, Optional
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime


class CrisisType(str, Enum):
    BANKING_CRISIS = "banking_crisis"
    TECH_BUBBLE = "tech_bubble"
    ENERGY_CRISIS = "energy_crisis"
    PANDEMIC = "pandemic"
    GEOPOLITICAL = "geopolitical"
    INFLATION_SPIKE = "inflation_spike"
    INTEREST_RATE_SHOCK = "interest_rate_shock"
    HOUSING_CRASH = "housing_crash"
    SUPPLY_CHAIN = "supply_chain"
    CURRENCY_CRISIS = "currency_crisis"


class PortfolioAsset(BaseModel):
    symbol: str = Field(..., description="Stock ticker symbol")
    quantity: float = Field(..., gt=0, description="Number of shares")
    
    @validator('symbol')
    def validate_symbol(cls, v):
        return v.upper().strip()


class Portfolio(BaseModel):
    name: str = Field("My Portfolio", description="Portfolio name")
    assets: List[PortfolioAsset] = Field(..., min_items=1)
    cash: float = Field(0, ge=0, description="Cash holdings in USD")


class CrisisSimulationRequest(BaseModel):
    portfolio: Portfolio
    crisis_type: CrisisType
    intensity: float = Field(..., ge=0.1, le=10.0, description="Crisis intensity (0.1-10)")
    duration_days: int = Field(30, ge=1, le=365, description="Crisis duration in days")
    

class MarketPredictionRequest(BaseModel):
    portfolio: Portfolio
    forecast_days: int = Field(30, ge=1, le=180, description="Days to forecast")
    confidence_level: float = Field(0.95, ge=0.8, le=0.99)


class PortfolioMetrics(BaseModel):
    total_value: float
    daily_return: float
    volatility: float
    sharpe_ratio: Optional[float]
    max_drawdown: float
    value_at_risk: float  # VaR 95%
    beta: Optional[float]  # vs SPY


class AssetImpact(BaseModel):
    symbol: str
    current_price: float
    simulated_price: float
    price_change_pct: float
    sector: Optional[str]
    impact_reason: str


class CrisisSimulationResult(BaseModel):
    crisis_type: CrisisType
    intensity: float
    portfolio_initial_value: float
    portfolio_final_value: float
    total_loss_pct: float
    total_loss_usd: float
    asset_impacts: List[AssetImpact]
    metrics_before: PortfolioMetrics
    metrics_after: PortfolioMetrics
    daily_values: List[Dict[str, float]]  # Date -> portfolio value
    recommendations: List[str]


class MarketPredictionResult(BaseModel):
    portfolio_current_value: float
    predicted_value_mean: float
    predicted_value_upper: float
    predicted_value_lower: float
    confidence_level: float
    expected_return_pct: float
    risk_score: float  # 0-100
    predicted_daily_values: List[Dict[str, float]]
    asset_predictions: List[Dict[str, float]]
    market_indicators: Dict[str, float]
