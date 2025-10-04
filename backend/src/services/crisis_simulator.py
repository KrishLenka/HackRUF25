import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from models import (
    CrisisType, Portfolio, CrisisSimulationResult, 
    AssetImpact, PortfolioMetrics
)
from data_loader import get_historical_prices


# Crisis impact matrix: how each crisis affects different sectors
SECTOR_CRISIS_IMPACT = {
    CrisisType.BANKING_CRISIS: {
        "Financial Services": -0.35,
        "Real Estate": -0.25,
        "Consumer Cyclical": -0.20,
        "Basic Materials": -0.15,
        "Industrials": -0.15,
        "Technology": -0.10,
        "Healthcare": -0.05,
        "Consumer Defensive": -0.03,
        "Utilities": 0.02,
        "Communication Services": -0.08,
        "Energy": -0.12,
    },
    CrisisType.TECH_BUBBLE: {
        "Technology": -0.45,
        "Communication Services": -0.30,
        "Consumer Cyclical": -0.15,
        "Industrials": -0.10,
        "Financial Services": -0.08,
        "Healthcare": 0.05,
        "Consumer Defensive": 0.03,
        "Utilities": 0.08,
        "Energy": 0.02,
        "Real Estate": -0.05,
        "Basic Materials": -0.05,
    },
    CrisisType.ENERGY_CRISIS: {
        "Energy": 0.30,  # Energy stocks may benefit
        "Utilities": -0.25,
        "Industrials": -0.20,
        "Basic Materials": -0.18,
        "Consumer Cyclical": -0.15,
        "Communication Services": -0.10,
        "Technology": -0.12,
        "Financial Services": -0.10,
        "Real Estate": -0.08,
        "Healthcare": -0.05,
        "Consumer Defensive": -0.08,
    },
    CrisisType.PANDEMIC: {
        "Travel & Leisure": -0.50,
        "Real Estate": -0.30,
        "Financial Services": -0.25,
        "Consumer Cyclical": -0.20,
        "Industrials": -0.18,
        "Energy": -0.28,
        "Basic Materials": -0.10,
        "Communication Services": 0.05,
        "Technology": 0.15,
        "Healthcare": 0.20,
        "Consumer Defensive": 0.10,
        "Utilities": 0.02,
    },
    CrisisType.GEOPOLITICAL: {
        "Energy": 0.15,
        "Healthcare": 0.10,
        "Utilities": 0.08,
        "Consumer Defensive": 0.05,
        "Financial Services": -0.20,
        "Technology": -0.15,
        "Industrials": -0.18,
        "Consumer Cyclical": -0.15,
        "Communication Services": -0.10,
        "Real Estate": -0.12,
        "Basic Materials": -0.10,
    },
    CrisisType.INFLATION_SPIKE: {
        "Consumer Defensive": -0.15,
        "Consumer Cyclical": -0.25,
        "Real Estate": -0.20,
        "Technology": -0.22,
        "Financial Services": -0.10,
        "Healthcare": -0.08,
        "Industrials": -0.12,
        "Basic Materials": 0.05,
        "Energy": 0.18,
        "Utilities": -0.10,
        "Communication Services": -0.15,
    },
    CrisisType.INTEREST_RATE_SHOCK: {
        "Real Estate": -0.35,
        "Utilities": -0.25,
        "Technology": -0.30,
        "Consumer Cyclical": -0.20,
        "Financial Services": -0.15,
        "Industrials": -0.15,
        "Communication Services": -0.12,
        "Consumer Defensive": -0.10,
        "Healthcare": -0.08,
        "Basic Materials": -0.10,
        "Energy": -0.05,
    },
    CrisisType.HOUSING_CRASH: {
        "Real Estate": -0.45,
        "Financial Services": -0.30,
        "Consumer Cyclical": -0.25,
        "Industrials": -0.20,
        "Basic Materials": -0.18,
        "Technology": -0.12,
        "Communication Services": -0.10,
        "Consumer Defensive": -0.05,
        "Healthcare": -0.03,
        "Utilities": 0.02,
        "Energy": -0.08,
    },
    CrisisType.SUPPLY_CHAIN: {
        "Industrials": -0.30,
        "Consumer Cyclical": -0.25,
        "Basic Materials": -0.20,
        "Technology": -0.22,
        "Consumer Defensive": -0.15,
        "Healthcare": -0.12,
        "Energy": -0.10,
        "Communication Services": -0.08,
        "Financial Services": -0.10,
        "Real Estate": -0.08,
        "Utilities": -0.05,
    },
    CrisisType.CURRENCY_CRISIS: {
        "Financial Services": -0.30,
        "Real Estate": -0.25,
        "Consumer Cyclical": -0.20,
        "Communication Services": -0.15,
        "Technology": -0.18,
        "Industrials": -0.15,
        "Basic Materials": -0.12,
        "Consumer Defensive": -0.10,
        "Healthcare": -0.08,
        "Utilities": -0.05,
        "Energy": 0.10,
    },
}

# Simplified sector mapping for common stocks
STOCK_SECTOR_MAP = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology", 
    "GOOG": "Technology", "AMZN": "Consumer Cyclical", "META": "Communication Services",
    "TSLA": "Consumer Cyclical", "NVDA": "Technology", "JPM": "Financial Services",
    "BAC": "Financial Services", "WFC": "Financial Services", "GS": "Financial Services",
    "MS": "Financial Services", "C": "Financial Services", "V": "Financial Services",
    "MA": "Financial Services", "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare", "ABBV": "Healthcare",
    "MRK": "Healthcare", "LLY": "Healthcare", "PG": "Consumer Defensive",
    "KO": "Consumer Defensive", "PEP": "Consumer Defensive", "WMT": "Consumer Defensive",
    "COST": "Consumer Defensive", "HD": "Consumer Cyclical", "MCD": "Consumer Cyclical",
    "NKE": "Consumer Cyclical", "DIS": "Communication Services", "NFLX": "Communication Services",
    "T": "Communication Services", "VZ": "Communication Services", "CMCSA": "Communication Services",
    "BA": "Industrials", "CAT": "Industrials", "GE": "Industrials", "MMM": "Industrials",
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities", "D": "Utilities",
    "SPY": "Market Index", "QQQ": "Market Index", "DIA": "Market Index",
}


class CrisisSimulator:
    def __init__(self):
        self.rng = np.random.RandomState(42)
    
    def get_stock_sector(self, symbol: str) -> str:
        """Get the sector for a given stock symbol."""
        return STOCK_SECTOR_MAP.get(symbol, "Unknown")
    
    def calculate_crisis_impact(
        self, 
        symbol: str, 
        crisis_type: CrisisType, 
        intensity: float,
        current_price: float
    ) -> Tuple[float, str]:
        """
        Calculate the impact of a crisis on a specific stock.
        Returns (final_price, impact_reason)
        """
        sector = self.get_stock_sector(symbol)
        
        # Get base impact from crisis-sector matrix
        base_impact = SECTOR_CRISIS_IMPACT.get(crisis_type, {}).get(sector, -0.10)
        
        # Apply intensity multiplier (0.1 to 10)
        adjusted_impact = base_impact * intensity
        
        # Add some randomness for volatility (±20% of impact)
        noise = self.rng.normal(0, abs(adjusted_impact) * 0.2)
        final_impact = adjusted_impact + noise
        
        # Cap the impact between -90% and +100%
        final_impact = np.clip(final_impact, -0.90, 1.0)
        
        final_price = current_price * (1 + final_impact)
        
        impact_reason = self._generate_impact_reason(
            sector, crisis_type, final_impact
        )
        
        return final_price, impact_reason
    
    def _generate_impact_reason(
        self, 
        sector: str, 
        crisis_type: CrisisType, 
        impact: float
    ) -> str:
        """Generate a human-readable reason for the impact."""
        direction = "surged" if impact > 0 else "declined"
        magnitude = "significantly" if abs(impact) > 0.2 else "moderately"
        
        reasons = {
            CrisisType.BANKING_CRISIS: f"{sector} sector {direction} {magnitude} due to banking system instability",
            CrisisType.TECH_BUBBLE: f"{sector} stocks {direction} as tech valuation concerns spread",
            CrisisType.ENERGY_CRISIS: f"{sector} affected by energy price volatility and supply constraints",
            CrisisType.PANDEMIC: f"{sector} impacted by pandemic-related demand and supply changes",
            CrisisType.GEOPOLITICAL: f"{sector} responded to geopolitical tensions and trade uncertainty",
            CrisisType.INFLATION_SPIKE: f"{sector} pressured by inflation eroding margins and demand",
            CrisisType.INTEREST_RATE_SHOCK: f"{sector} reacted to sudden interest rate changes",
            CrisisType.HOUSING_CRASH: f"{sector} affected by housing market collapse and credit concerns",
            CrisisType.SUPPLY_CHAIN: f"{sector} disrupted by supply chain bottlenecks and delays",
            CrisisType.CURRENCY_CRISIS: f"{sector} impacted by currency volatility and capital flight",
        }
        
        return reasons.get(crisis_type, f"{sector} affected by {crisis_type}")
    
    def simulate_crisis_timeline(
        self,
        initial_value: float,
        final_value: float,
        duration_days: int
    ) -> List[Dict[str, float]]:
        """
        Generate a realistic timeline of portfolio values during the crisis.
        Crisis typically follows a pattern: slow decline -> crash -> slight recovery
        """
        timeline = []
        
        # Generate crisis curve
        t = np.linspace(0, 1, duration_days)
        
        # Crisis pattern: exponential decay with some recovery at the end
        crisis_curve = np.exp(-3 * t) * (1 - t) + t * 0.3
        
        # Normalize to go from initial to final value
        crisis_curve = crisis_curve / crisis_curve[0]
        min_point = crisis_curve.min()
        
        # Adjust so it ends at the final value
        final_ratio = final_value / initial_value
        crisis_curve = initial_value + (crisis_curve - 1) * initial_value * (1 - final_ratio) / (1 - min_point)
        
        # Add daily noise
        noise = self.rng.normal(0, initial_value * 0.02, duration_days)
        crisis_curve = crisis_curve + noise
        
        start_date = datetime.now()
        for day, value in enumerate(crisis_curve):
            date = start_date + timedelta(days=day)
            timeline.append({
                "date": date.strftime("%Y-%m-%d"),
                "value": float(value),
                "day": day + 1
            })
        
        return timeline
    
    def calculate_portfolio_metrics(
        self,
        portfolio_value: float,
        daily_returns: np.ndarray,
        market_returns: Optional[np.ndarray] = None
    ) -> PortfolioMetrics:
        """Calculate various risk and performance metrics for a portfolio."""
        
        volatility = float(np.std(daily_returns) * np.sqrt(252))  # Annualized
        mean_return = float(np.mean(daily_returns))
        
        # Sharpe ratio (assuming 3% risk-free rate)
        risk_free_rate = 0.03 / 252  # Daily risk-free rate
        if volatility > 0:
            sharpe_ratio = float((mean_return - risk_free_rate) / (volatility / np.sqrt(252)))
        else:
            sharpe_ratio = 0.0
        
        # Maximum drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = float(drawdown.min())
        
        # Value at Risk (95% confidence)
        var_95 = float(np.percentile(daily_returns, 5) * portfolio_value)
        
        # Beta (if market returns provided)
        beta = None
        if market_returns is not None and len(market_returns) == len(daily_returns):
            covariance = np.cov(daily_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            if market_variance > 0:
                beta = float(covariance / market_variance)
        
        return PortfolioMetrics(
            total_value=portfolio_value,
            daily_return=mean_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            value_at_risk=var_95,
            beta=beta
        )
    
    def generate_recommendations(
        self,
        crisis_type: CrisisType,
        asset_impacts: List[AssetImpact],
        total_loss_pct: float
    ) -> List[str]:
        """Generate actionable recommendations based on simulation results."""
        recommendations = []
        
        # General crisis-specific advice
        crisis_advice = {
            CrisisType.BANKING_CRISIS: "Consider reducing exposure to financial sector stocks and increasing allocation to defensive sectors like utilities and healthcare.",
            CrisisType.TECH_BUBBLE: "Diversify away from tech-heavy positions. Consider value stocks and stable dividend payers.",
            CrisisType.ENERGY_CRISIS: "Energy stocks may provide a hedge. Consider increasing exposure to traditional energy companies.",
            CrisisType.PANDEMIC: "Healthcare and technology sectors often show resilience. Avoid travel and hospitality stocks.",
            CrisisType.GEOPOLITICAL: "Focus on domestic companies with less international exposure. Consider defensive stocks and commodities.",
            CrisisType.INFLATION_SPIKE: "Look at inflation-protected securities (TIPS), commodities, and real estate. Reduce bonds.",
            CrisisType.INTEREST_RATE_SHOCK: "Avoid high-growth tech and REITs. Consider financial stocks that benefit from higher rates.",
            CrisisType.HOUSING_CRASH: "Reduce real estate and construction exposure. Increase cash positions and defensive stocks.",
            CrisisType.SUPPLY_CHAIN: "Favor companies with localized supply chains and strong inventory management.",
            CrisisType.CURRENCY_CRISIS: "Consider multinational companies that can benefit from currency moves and commodity-based stocks.",
        }
        
        recommendations.append(crisis_advice.get(crisis_type, "Diversify your portfolio across sectors."))
        
        # Loss-based recommendations
        if total_loss_pct > 30:
            recommendations.append("⚠️ SEVERE RISK: Your portfolio is highly vulnerable. Immediate diversification needed.")
        elif total_loss_pct > 20:
            recommendations.append("⚠️ HIGH RISK: Significant losses expected. Consider hedging strategies.")
        elif total_loss_pct > 10:
            recommendations.append("⚠️ MODERATE RISK: Notable exposure to crisis. Review sector allocation.")
        else:
            recommendations.append("✓ LOW RISK: Portfolio shows resilience to this crisis scenario.")
        
        # Asset-specific recommendations
        worst_performers = sorted(asset_impacts, key=lambda x: x.price_change_pct)[:3]
        if worst_performers:
            symbols = ", ".join([a.symbol for a in worst_performers])
            recommendations.append(f"Most vulnerable assets: {symbols}. Consider reducing positions.")
        
        # Sector diversification
        sectors = [a.sector for a in asset_impacts if a.sector]
        if len(set(sectors)) < 3:
            recommendations.append("Insufficient sector diversification. Consider adding stocks from defensive sectors.")
        
        return recommendations
    
    def simulate(
        self,
        portfolio: Portfolio,
        crisis_type: CrisisType,
        intensity: float,
        duration_days: int
    ) -> CrisisSimulationResult:
        """
        Main simulation function that runs a crisis scenario on a portfolio.
        """
        # Get current prices for all assets
        symbols = [asset.symbol for asset in portfolio.assets]
        
        try:
            # Get last 60 days of data for metrics calculation
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)
            
            prices_df = get_historical_prices(
                symbols,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d")
            )
            
            current_prices = prices_df.iloc[-1].to_dict()
            
            # Calculate returns for metrics
            returns_df = prices_df.pct_change().dropna()
            
        except Exception as e:
            # Fallback: use dummy data if API fails
            print(f"Warning: Could not fetch real prices: {e}")
            current_prices = {symbol: 100.0 for symbol in symbols}
            returns_df = pd.DataFrame({symbol: np.random.randn(60) * 0.01 for symbol in symbols})
        
        # Calculate initial portfolio value
        initial_value = portfolio.cash
        for asset in portfolio.assets:
            initial_value += asset.quantity * current_prices.get(asset.symbol, 0)
        
        # Simulate crisis impact on each asset
        asset_impacts = []
        final_value = portfolio.cash
        
        for asset in portfolio.assets:
            current_price = current_prices.get(asset.symbol, 0)
            simulated_price, impact_reason = self.calculate_crisis_impact(
                asset.symbol, crisis_type, intensity, current_price
            )
            
            price_change_pct = ((simulated_price - current_price) / current_price) * 100
            
            asset_impacts.append(AssetImpact(
                symbol=asset.symbol,
                current_price=current_price,
                simulated_price=simulated_price,
                price_change_pct=price_change_pct,
                sector=self.get_stock_sector(asset.symbol),
                impact_reason=impact_reason
            ))
            
            final_value += asset.quantity * simulated_price
        
        # Calculate portfolio metrics before and after
        portfolio_returns = returns_df.mean(axis=1).values
        metrics_before = self.calculate_portfolio_metrics(
            initial_value,
            portfolio_returns
        )
        
        # Simulate post-crisis returns (more volatile)
        crisis_returns = portfolio_returns * (1 + intensity * 0.5)
        metrics_after = self.calculate_portfolio_metrics(
            final_value,
            crisis_returns
        )
        
        # Generate timeline
        daily_values = self.simulate_crisis_timeline(
            initial_value,
            final_value,
            duration_days
        )
        
        # Calculate totals
        total_loss_usd = initial_value - final_value
        total_loss_pct = (total_loss_usd / initial_value) * 100
        
        # Generate recommendations
        recommendations = self.generate_recommendations(
            crisis_type,
            asset_impacts,
            total_loss_pct
        )
        
        return CrisisSimulationResult(
            crisis_type=crisis_type,
            intensity=intensity,
            portfolio_initial_value=initial_value,
            portfolio_final_value=final_value,
            total_loss_pct=total_loss_pct,
            total_loss_usd=total_loss_usd,
            asset_impacts=asset_impacts,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            daily_values=daily_values,
            recommendations=recommendations
        )
