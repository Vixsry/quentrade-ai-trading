#!/bin/bash

# Quentrade Setup Script
# Run this script to set up Quentrade on your system

echo "================================"
echo "   Quentrade Setup Script       "
echo "================================"

# Check if Python 3.8+ is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3.8+ is required but not installed. Please install Python first."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if [ "$(printf '%s\n' "3.8" "$PYTHON_VERSION" | sort -V | head -n1)" != "3.8" ]; then
    echo "Python 3.8+ is required. Found version $PYTHON_VERSION"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p models
mkdir -p logs
mkdir -p data
mkdir -p strategies
mkdir -p backtest_results

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << EOL
# Quentrade API Keys
BYBIT_API_KEY=your_api_key_here
BYBIT_API_SECRET=your_api_secret_here
COINMARKETCAP_API_KEY=your_api_key_here
FRED_API_KEY=your_api_key_here
NEWS_API_KEY=your_api_key_here
QUIKNODE_API_KEY=your_api_key_here

# Optional notification settings
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
DISCORD_WEBHOOK_URL=your_webhook_url_here

# Trading settings
RISK_PER_TRADE=0.02
MAX_POSITION_SIZE=0.1
USE_TESTNET=false
EOL
    echo ".env file created. Please update it with your API keys."
fi

# Create database
echo "Initializing database..."
python3 -c "from quentrade import QuentradeAI; app = QuentradeAI(); app.initialize_database()"

# Create log file
echo "Setting up logging..."
touch logs/quentrade.log

# Make main script executable
chmod +x quentrade.py

echo "================================"
echo "   Setup complete!              "
echo "================================"
echo ""
echo "Next steps:"
echo "1. Edit the .env file with your API keys"
echo "2. Run: source venv/bin/activate"
echo "3. Run: python quentrade.py"
echo ""
echo "For more information, see README.md"