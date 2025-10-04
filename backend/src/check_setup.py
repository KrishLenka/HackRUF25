"""
Simple script to check if the backend is properly set up.
Run this before starting the server.
"""

import sys
import os

def check_module(module_name, package_name=None):
    """Check if a module can be imported."""
    if package_name is None:
        package_name = module_name
    
    try:
        __import__(module_name)
        print(f"✓ {package_name} installed")
        return True
    except ImportError:
        print(f"✗ {package_name} NOT installed - run: pip install {package_name}")
        return False

def check_env_file():
    """Check if .env file exists."""
    env_path = os.path.join(os.path.dirname(__file__), '../../.env')
    if os.path.exists(env_path):
        print(f"✓ .env file found")
        
        # Check if it has Alpaca keys
        from dotenv import load_dotenv
        load_dotenv(env_path)
        
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        if api_key and secret_key and 'your_' not in api_key:
            print(f"✓ Alpaca API keys configured")
        else:
            print(f"⚠ Alpaca API keys not set (will use yfinance fallback)")
        return True
    else:
        print(f"⚠ .env file not found - copy backend/env_template.txt to .env")
        return False

def main():
    print("="*60)
    print("BACKEND SETUP CHECK")
    print("="*60)
    print()
    
    # Required modules
    required = [
        ('fastapi', 'fastapi'),
        ('uvicorn', 'uvicorn'),
        ('pydantic', 'pydantic'),
        ('dotenv', 'python-dotenv'),
        ('requests', 'requests'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
    ]
    
    # Optional but recommended
    optional = [
        ('alpaca_trade_api', 'alpaca-trade-api'),
        ('yfinance', 'yfinance'),
        ('sklearn', 'scikit-learn'),
        ('scipy', 'scipy'),
        ('statsmodels', 'statsmodels'),
    ]
    
    print("Required Packages:")
    required_ok = all(check_module(mod, pkg) for mod, pkg in required)
    
    print("\nOptional Packages:")
    optional_ok = all(check_module(mod, pkg) for mod, pkg in optional)
    
    print("\nConfiguration:")
    env_ok = check_env_file()
    
    print("\n" + "="*60)
    
    if required_ok and optional_ok and env_ok:
        print("✓ ALL CHECKS PASSED - Ready to start!")
        print("\nRun the server with:")
        print("  python main.py")
        print("\nOr:")
        print("  uvicorn main:app --reload")
        return 0
    elif required_ok:
        print("⚠ PARTIALLY READY - Some optional packages missing")
        print("\nYou can start the server, but some features may not work")
        print("Install missing packages with:")
        print("  pip install -r ../../requirements.txt")
        return 0
    else:
        print("✗ SETUP INCOMPLETE - Install required packages first")
        print("\nRun:")
        print("  pip install fastapi uvicorn pydantic python-dotenv requests pandas numpy")
        return 1

if __name__ == "__main__":
    sys.exit(main())
