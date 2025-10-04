import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Alpaca API Configuration
    ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
    ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
    ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    
    # Application Configuration
    APP_NAME = "Portfolio Crisis Simulator"
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # Database Configuration (for future use)
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./portfolio_simulator.db")
    
    # Cache Configuration
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    
    # Simulation Parameters
    DEFAULT_LOOKBACK_DAYS = 365 * 3  # 3 years of historical data
    MONTE_CARLO_SIMULATIONS = 1000
    
settings = Settings()
