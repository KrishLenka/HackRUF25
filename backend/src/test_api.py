"""
Test script to demonstrate the Portfolio Crisis Simulator API
Run this after starting the server with: python main.py
"""

import requests
import json
from pprint import pprint

BASE_URL = "http://localhost:8000/api/v1"


def test_health_check():
    """Test if the API is running."""
    print("\n" + "="*60)
    print("Testing Health Check...")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    pprint(response.json())


def test_get_crisis_types():
    """Get all available crisis types."""
    print("\n" + "="*60)
    print("Getting Available Crisis Types...")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/crisis-types")
    data = response.json()
    
    print(f"\nAvailable Crisis Types ({len(data['crisis_types'])}):")
    for crisis in data['crisis_types']:
        print(f"\n  • {crisis['details']['name']}")
        print(f"    Value: {crisis['value']}")
        print(f"    Description: {crisis['details']['description']}")
        print(f"    Example: {crisis['details']['example']}")
        print(f"    Severity: {crisis['details']['severity']}")


def test_crisis_simulation():
    """Test crisis simulation with a sample portfolio."""
    print("\n" + "="*60)
    print("Testing Crisis Simulation...")
    print("="*60)
    
    # Sample portfolio with tech stocks
    portfolio = {
        "name": "Tech Growth Portfolio",
        "assets": [
            {"symbol": "AAPL", "quantity": 10},
            {"symbol": "MSFT", "quantity": 15},
            {"symbol": "GOOGL", "quantity": 5},
            {"symbol": "TSLA", "quantity": 8},
        ],
        "cash": 5000.0
    }
    
    request_data = {
        "portfolio": portfolio,
        "crisis_type": "tech_bubble",
        "intensity": 2.5,
        "duration_days": 60
    }
    
    print(f"\nSimulating Tech Bubble (Intensity: 2.5) for 60 days...")
    print(f"Portfolio: {portfolio['name']}")
    
    response = requests.post(
        f"{BASE_URL}/simulate-crisis",
        json=request_data
    )
    
    if response.status_code == 200:
        result = response.json()
        
        print(f"\n{'='*60}")
        print("SIMULATION RESULTS")
        print(f"{'='*60}")
        
        print(f"\nCrisis Type: {result['crisis_type'].replace('_', ' ').title()}")
        print(f"Intensity: {result['intensity']}")
        
        print(f"\nPortfolio Performance:")
        print(f"  Initial Value: ${result['portfolio_initial_value']:,.2f}")
        print(f"  Final Value:   ${result['portfolio_final_value']:,.2f}")
        print(f"  Total Loss:    ${result['total_loss_usd']:,.2f} ({result['total_loss_pct']:.2f}%)")
        
        print(f"\nMetrics Before Crisis:")
        metrics_before = result['metrics_before']
        print(f"  Volatility:     {metrics_before['volatility']*100:.2f}%")
        print(f"  Sharpe Ratio:   {metrics_before['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:   {metrics_before['max_drawdown']*100:.2f}%")
        
        print(f"\nMetrics After Crisis:")
        metrics_after = result['metrics_after']
        print(f"  Volatility:     {metrics_after['volatility']*100:.2f}%")
        print(f"  Sharpe Ratio:   {metrics_after['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:   {metrics_after['max_drawdown']*100:.2f}%")
        
        print(f"\nAsset-Level Impact:")
        for asset in result['asset_impacts']:
            print(f"\n  {asset['symbol']} ({asset['sector']}):")
            print(f"    Current Price:  ${asset['current_price']:.2f}")
            print(f"    Simulated Price: ${asset['simulated_price']:.2f}")
            print(f"    Change:         {asset['price_change_pct']:+.2f}%")
            print(f"    Reason: {asset['impact_reason']}")
        
        print(f"\nRecommendations:")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print(f"\nFirst 5 days of timeline:")
        for day in result['daily_values'][:5]:
            print(f"  Day {day['day']}: ${day['value']:,.2f}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


def test_market_prediction():
    """Test market prediction with a sample portfolio."""
    print("\n" + "="*60)
    print("Testing Market Prediction...")
    print("="*60)
    
    # Sample diversified portfolio
    portfolio = {
        "name": "Balanced Portfolio",
        "assets": [
            {"symbol": "AAPL", "quantity": 5},
            {"symbol": "JPM", "quantity": 10},
            {"symbol": "JNJ", "quantity": 8},
            {"symbol": "XOM", "quantity": 12},
        ],
        "cash": 10000.0
    }
    
    request_data = {
        "portfolio": portfolio,
        "forecast_days": 30,
        "confidence_level": 0.95
    }
    
    print(f"\nPredicting performance for 30 days...")
    print(f"Portfolio: {portfolio['name']}")
    
    response = requests.post(
        f"{BASE_URL}/predict-performance",
        json=request_data
    )
    
    if response.status_code == 200:
        result = response.json()
        
        print(f"\n{'='*60}")
        print("PREDICTION RESULTS")
        print(f"{'='*60}")
        
        print(f"\nCurrent Portfolio Value: ${result['portfolio_current_value']:,.2f}")
        
        print(f"\nPredicted Values (30 days):")
        print(f"  Best Case (95th percentile):  ${result['predicted_value_upper']:,.2f}")
        print(f"  Expected (50th percentile):   ${result['predicted_value_mean']:,.2f}")
        print(f"  Worst Case (5th percentile):  ${result['predicted_value_lower']:,.2f}")
        
        print(f"\nExpected Return: {result['expected_return_pct']:+.2f}%")
        print(f"Risk Score: {result['risk_score']:.1f}/100")
        
        print(f"\nMarket Indicators:")
        for key, value in result['market_indicators'].items():
            print(f"  {key.replace('_', ' ').title()}: {value:.4f}")
        
        print(f"\nIndividual Asset Predictions:")
        for asset in result['asset_predictions']:
            print(f"\n  {asset['symbol']} - {asset['trend'].upper()}")
            print(f"    Current:  ${asset['current_price']:.2f}")
            print(f"    Predicted: ${asset['predicted_price']:.2f} ({asset['expected_return_pct']:+.2f}%)")
            print(f"    RSI: {asset['rsi']:.1f}")
            print(f"    Confidence: {asset['confidence']:.1f}%")
        
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


def test_batch_simulation():
    """Test batch simulation across multiple crisis types."""
    print("\n" + "="*60)
    print("Testing Batch Simulation...")
    print("="*60)
    
    portfolio = {
        "name": "Diverse Portfolio",
        "assets": [
            {"symbol": "AAPL", "quantity": 5},
            {"symbol": "JPM", "quantity": 10},
            {"symbol": "XOM", "quantity": 8},
        ],
        "cash": 5000.0
    }
    
    crisis_types = ["banking_crisis", "tech_bubble", "energy_crisis"]
    
    response = requests.post(
        f"{BASE_URL}/batch-simulate",
        params={
            "intensity": 1.5,
            "duration_days": 30
        },
        json={
            "portfolio": portfolio,
            "crisis_types": crisis_types
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        
        print(f"\nBatch Simulation Results:")
        print(f"Portfolio: {result['portfolio_name']}")
        
        for crisis_type, sim_result in result['simulations'].items():
            print(f"\n  {crisis_type.replace('_', ' ').title()}:")
            print(f"    Loss: ${sim_result['total_loss_usd']:,.2f} ({sim_result['total_loss_pct']:.2f}%)")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


def test_portfolio_value():
    """Test getting current portfolio value."""
    print("\n" + "="*60)
    print("Testing Portfolio Valuation...")
    print("="*60)
    
    portfolio = {
        "name": "Test Portfolio",
        "assets": [
            {"symbol": "AAPL", "quantity": 10},
            {"symbol": "MSFT", "quantity": 5},
        ],
        "cash": 1000.0
    }
    
    response = requests.post(
        f"{BASE_URL}/portfolio-value",
        json=portfolio
    )
    
    if response.status_code == 200:
        result = response.json()
        
        print(f"\nPortfolio Valuation:")
        print(f"  Total Value: ${result['total_value']:,.2f}")
        print(f"  Cash: ${result['cash']:,.2f}")
        print(f"  Invested: ${result['invested_value']:,.2f}")
        
        print(f"\nAsset Breakdown:")
        for asset in result['assets']:
            print(f"\n  {asset['symbol']}:")
            print(f"    Quantity: {asset['quantity']}")
            print(f"    Price: ${asset['price']:.2f}")
            print(f"    Value: ${asset['value']:,.2f}")
            print(f"    Allocation: {asset['percentage']:.2f}%")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("PORTFOLIO CRISIS SIMULATOR - API TEST SUITE")
    print("="*60)
    
    try:
        test_health_check()
        test_get_crisis_types()
        test_portfolio_value()
        test_crisis_simulation()
        test_market_prediction()
        test_batch_simulation()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED!")
        print("="*60 + "\n")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to the API server.")
        print("Make sure the server is running with: python main.py")
        print("Or: uvicorn main:app --reload\n")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}\n")


if __name__ == "__main__":
    main()
