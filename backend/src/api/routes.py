from fastapi import APIRouter, HTTPException, status
from typing import List
import logging

from models import (
    Portfolio,
    CrisisSimulationRequest,
    CrisisSimulationResult,
    MarketPredictionRequest,
    MarketPredictionResult,
    CrisisType,
)
from services.crisis_simulator import CrisisSimulator
from services.market_predictor import MarketPredictor

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services
crisis_simulator = CrisisSimulator()
market_predictor = MarketPredictor()


@router.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "Portfolio Crisis Simulator API",
        "version": "1.0.0"
    }


@router.get("/crisis-types")
async def get_crisis_types():
    """Get all available crisis types with descriptions."""
    crisis_descriptions = {
        CrisisType.BANKING_CRISIS: {
            "name": "Banking Crisis",
            "description": "Systemic failure in banking sector with credit freezes and bank runs",
            "example": "2008 Financial Crisis",
            "severity": "Very High",
            "typical_duration": "6-18 months"
        },
        CrisisType.TECH_BUBBLE: {
            "name": "Tech Bubble Burst",
            "description": "Collapse in technology stock valuations due to overvaluation",
            "example": "Dot-com Bubble (2000)",
            "severity": "High",
            "typical_duration": "12-24 months"
        },
        CrisisType.ENERGY_CRISIS: {
            "name": "Energy Crisis",
            "description": "Sharp increases in energy prices affecting economy-wide costs",
            "example": "1970s Oil Crisis",
            "severity": "High",
            "typical_duration": "6-12 months"
        },
        CrisisType.PANDEMIC: {
            "name": "Pandemic",
            "description": "Global health crisis disrupting economic activity and supply chains",
            "example": "COVID-19 (2020)",
            "severity": "Very High",
            "typical_duration": "12-36 months"
        },
        CrisisType.GEOPOLITICAL: {
            "name": "Geopolitical Crisis",
            "description": "International conflicts affecting trade, energy, and market sentiment",
            "example": "Ukraine Conflict",
            "severity": "Medium-High",
            "typical_duration": "Variable"
        },
        CrisisType.INFLATION_SPIKE: {
            "name": "Inflation Spike",
            "description": "Rapid increase in consumer prices eroding purchasing power",
            "example": "2021-2022 Inflation",
            "severity": "Medium",
            "typical_duration": "12-24 months"
        },
        CrisisType.INTEREST_RATE_SHOCK: {
            "name": "Interest Rate Shock",
            "description": "Sudden large increases in interest rates by central banks",
            "example": "Volcker Shock (1980s)",
            "severity": "Medium-High",
            "typical_duration": "6-18 months"
        },
        CrisisType.HOUSING_CRASH: {
            "name": "Housing Market Crash",
            "description": "Collapse in real estate prices with widespread defaults",
            "example": "Subprime Mortgage Crisis",
            "severity": "Very High",
            "typical_duration": "24-48 months"
        },
        CrisisType.SUPPLY_CHAIN: {
            "name": "Supply Chain Disruption",
            "description": "Major disruptions in global supply chains causing shortages",
            "example": "Post-COVID Supply Crisis",
            "severity": "Medium",
            "typical_duration": "6-18 months"
        },
        CrisisType.CURRENCY_CRISIS: {
            "name": "Currency Crisis",
            "description": "Rapid devaluation of currency causing capital flight",
            "example": "Asian Financial Crisis (1997)",
            "severity": "High",
            "typical_duration": "12-24 months"
        },
    }
    
    return {
        "crisis_types": [
            {
                "value": crisis_type.value,
                "details": crisis_descriptions[crisis_type]
            }
            for crisis_type in CrisisType
        ]
    }


@router.post("/simulate-crisis", response_model=CrisisSimulationResult)
async def simulate_crisis(request: CrisisSimulationRequest):
    """
    Simulate how a portfolio would perform during a specific crisis event.
    
    - **portfolio**: Portfolio with assets and cash holdings
    - **crisis_type**: Type of crisis to simulate
    - **intensity**: Crisis intensity from 0.1 (mild) to 10.0 (severe)
    - **duration_days**: How long the crisis lasts (1-365 days)
    """
    try:
        logger.info(f"Simulating {request.crisis_type} crisis with intensity {request.intensity}")
        
        result = crisis_simulator.simulate(
            portfolio=request.portfolio,
            crisis_type=request.crisis_type,
            intensity=request.intensity,
            duration_days=request.duration_days
        )
        
        return result
    
    except Exception as e:
        logger.error(f"Error simulating crisis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to simulate crisis: {str(e)}"
        )


@router.post("/predict-performance", response_model=MarketPredictionResult)
async def predict_performance(request: MarketPredictionRequest):
    """
    Predict portfolio performance based on current market conditions.
    
    - **portfolio**: Portfolio with assets and cash holdings
    - **forecast_days**: Number of days to forecast (1-180)
    - **confidence_level**: Statistical confidence level (0.8-0.99)
    """
    try:
        logger.info(f"Predicting portfolio performance for {request.forecast_days} days")
        
        result = market_predictor.predict(
            portfolio=request.portfolio,
            forecast_days=request.forecast_days,
            confidence_level=request.confidence_level
        )
        
        return result
    
    except Exception as e:
        logger.error(f"Error predicting performance: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to predict performance: {str(e)}"
        )


@router.post("/batch-simulate")
async def batch_simulate(
    portfolio: Portfolio,
    crisis_types: List[CrisisType],
    intensity: float = 1.0,
    duration_days: int = 30
):
    """
    Run multiple crisis simulations in batch to compare portfolio vulnerability.
    
    Returns results for all specified crisis types.
    """
    try:
        results = {}
        
        for crisis_type in crisis_types:
            result = crisis_simulator.simulate(
                portfolio=portfolio,
                crisis_type=crisis_type,
                intensity=intensity,
                duration_days=duration_days
            )
            results[crisis_type.value] = result
        
        return {
            "portfolio_name": portfolio.name,
            "intensity": intensity,
            "duration_days": duration_days,
            "simulations": results
        }
    
    except Exception as e:
        logger.error(f"Error in batch simulation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run batch simulation: {str(e)}"
        )


@router.post("/portfolio-value")
async def get_portfolio_value(portfolio: Portfolio):
    """
    Get current value of a portfolio based on real-time prices.
    """
    try:
        from data_loader import get_historical_prices
        from datetime import datetime, timedelta
        
        symbols = [asset.symbol for asset in portfolio.assets]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        
        prices_df = get_historical_prices(
            symbols,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )
        
        current_prices = prices_df.iloc[-1].to_dict()
        
        total_value = portfolio.cash
        asset_values = []
        
        for asset in portfolio.assets:
            price = current_prices.get(asset.symbol, 0)
            value = asset.quantity * price
            total_value += value
            
            asset_values.append({
                "symbol": asset.symbol,
                "quantity": asset.quantity,
                "price": price,
                "value": value,
                "percentage": 0  # Will calculate after total
            })
        
        # Calculate percentages
        for asset_val in asset_values:
            asset_val["percentage"] = (asset_val["value"] / total_value) * 100
        
        return {
            "total_value": total_value,
            "cash": portfolio.cash,
            "invested_value": total_value - portfolio.cash,
            "assets": asset_values,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error calculating portfolio value: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate portfolio value: {str(e)}"
        )
