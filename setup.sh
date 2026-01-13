#!/bin/bash
# Finstreet Trading System Setup Script
# Ensures reproducible environment across machines

set -e

echo "========================================"
echo "Finstreet Trading System - Setup"
echo "========================================"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.10"

if [[ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]]; then
    echo "Warning: Python $REQUIRED_VERSION+ recommended. Current: $PYTHON_VERSION"
fi

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data/raw data/processed models reports/figures

# Setup .env template if not exists
if [ ! -f ".env" ]; then
    echo "Creating .env template..."
    cat > .env << 'EOF'
# FYERS API Configuration
FYERS_CLIENT_ID=your_client_id
FYERS_SECRET_KEY=your_secret_key
FYERS_ACCESS_TOKEN=your_access_token

# Optional: Logging level
LOG_LEVEL=INFO
EOF
    echo "Please update .env with your FYERS credentials"
fi

echo "========================================"
echo "Setup complete!"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run the full pipeline:"
echo "  python run.py all"
echo ""
echo "Available commands:"
echo "  python run.py fetch      - Fetch data"
echo "  python run.py ensemble   - Train models"
echo "  python run.py backtest   - Run backtest"
echo "  python run.py predict    - Generate predictions"
echo "  python run.py visualize  - Generate charts"
echo "========================================"
