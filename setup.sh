#!/bin/bash
set -e

echo "========================================"
echo "Finstreet Trading System - Setup"
echo "========================================"

PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.10"

if [[ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]]; then
    echo "Warning: Python $REQUIRED_VERSION+ recommended. Current: $PYTHON_VERSION"
fi

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

mkdir -p data/raw data/processed models reports/figures

if [ ! -f ".env" ]; then
    echo "Creating .env template..."
    cat > .env << 'EOF'
FYERS_CLIENT_ID=your_client_id
FYERS_SECRET_KEY=your_secret_key
FYERS_ACCESS_TOKEN=your_access_token
EOF
    echo "Update .env with your FYERS credentials"
fi

echo "========================================"
echo "Setup complete!"
echo ""
echo "Activate: source venv/bin/activate"
echo ""
echo "Commands:"
echo "  python run.py all        - Full pipeline"
echo "  python run.py fetch      - Fetch data"
echo "  python run.py ensemble   - Train models"
echo "  python run.py backtest   - Run backtest"
echo "  python run.py predict    - Generate predictions"
echo "  python run.py visualize  - Generate charts"
echo "========================================"
