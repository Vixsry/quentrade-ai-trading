#!/usr/bin/env python3
"""
Quentrade Launcher
Simplified script to start Quentrade with proper error handling
"""

import os
import sys
import logging
from dotenv import load_dotenv

def check_requirements():
    """Check if all requirements are met"""
    required_files = ['.env', 'requirements.txt', 'config.ini']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"Error: Missing required files: {', '.join(missing_files)}")
        print("Please run ./setup.sh first")
        return False
    
    return True

def check_api_keys():
    """Check if API keys are configured"""
    load_dotenv()
    
    required_keys = [
        'BYBIT_API_KEY',
        'BYBIT_API_SECRET',
        'COINMARKETCAP_API_KEY',
        'FRED_API_KEY',
        'NEWS_API_KEY'
    ]
    
    missing_keys = []
    
    for key in required_keys:
        if not os.getenv(key) or os.getenv(key) == 'your_api_key_here':
            missing_keys.append(key)
    
    if missing_keys:
        print(f"Error: Missing or unconfigured API keys: {', '.join(missing_keys)}")
        print("Please edit the .env file with your actual API keys")
        return False
    
    return True

def setup_logging():
    """Setup logging configuration"""
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/quentrade.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """Main function to start Quentrade"""
    try:
        # Check requirements
        if not check_requirements():
            sys.exit(1)
        
        # Check API keys
        if not check_api_keys():
            sys.exit(1)
        
        # Setup logging
        setup_logging()
        
        # Import and run Quentrade
        print("Starting Quentrade AI Trading Terminal...")
        from quentrade import QuentradeAI
        import asyncio
        
        app = QuentradeAI()
        asyncio.run(app.run())
        
    except ImportError as e:
        print(f"Error: Failed to import required modules: {e}")
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)
    
    except KeyboardInterrupt:
        print("\nQuentrade terminated by user")
        sys.exit(0)
    
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        print(f"Fatal error: {e}")
        print("Check logs/quentrade.log for details")
        sys.exit(1)

if __name__ == "__main__":
    main()