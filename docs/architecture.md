# System Architecture

## Overview

This document describes the architecture of the algorithmic trading system, including data flow, component interactions, and execution pipeline.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRADING SYSTEM ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  DATA LAYER  │───▶│   FEATURES   │───▶│    MODEL     │───▶│  EXECUTION   │
│              │    │   PIPELINE   │    │   PIPELINE   │    │   ENGINE     │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ • Fyers API  │    │ • Indicators │    │ • Training   │    │ • Order Mgmt │
│ • Raw Data   │    │ • Labeling   │    │ • Prediction │    │ • Risk Mgmt  │
│ • Processing │    │ • Advanced   │    │ • Validation │    │ • Position   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

## Component Details

### 1. Data Layer (`src/data/`)

Responsible for market data acquisition and storage.

```
┌─────────────────────────────────────────┐
│              DATA LAYER                 │
├─────────────────────────────────────────┤
│                                         │
│  ┌─────────────┐    ┌─────────────┐    │
│  │ fyers_client│───▶│ fetch_data  │    │
│  │    .py      │    │    .py      │    │
│  └─────────────┘    └──────┬──────┘    │
│                            │           │
│                            ▼           │
│                    ┌─────────────┐     │
│                    │  data/raw/  │     │
│                    │  data/proc/ │     │
│                    └─────────────┘     │
└─────────────────────────────────────────┘
```

**Components:**
- `fyers_client.py` - API client for Fyers broker
- `fetch_data.py` - Data fetching and storage logic
- `generate_token.py` - Authentication token management

### 2. Feature Pipeline (`src/features/`)

Transforms raw market data into model-ready features.

```
┌─────────────────────────────────────────────────────────────────┐
│                     FEATURE PIPELINE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Raw OHLCV ──▶ ┌────────────┐ ──▶ ┌────────────┐ ──▶ Features   │
│                │ indicators │     │  advanced  │                │
│                │    .py     │     │ features.py│                │
│                └────────────┘     └────────────┘                │
│                      │                  │                        │
│                      ▼                  ▼                        │
│              ┌─────────────────────────────────┐                 │
│              │      preprocessing.py           │                 │
│              │  • Normalization                │                 │
│              │  • Missing value handling       │                 │
│              │  • Feature scaling              │                 │
│              └─────────────────────────────────┘                 │
│                              │                                   │
│                              ▼                                   │
│                      ┌─────────────┐                             │
│                      │ labeling.py │                             │
│                      │ (Target Gen)│                             │
│                      └─────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

**Feature Categories:**
- Technical indicators (RSI, MACD, Bollinger Bands)
- Price-derived features (returns, volatility)
- Volume analysis
- Trend detection

### 3. Model Pipeline (`src/model/`)

Handles model training, validation, and prediction.

```
┌─────────────────────────────────────────────────────────────────┐
│                      MODEL PIPELINE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Features ──▶ ┌────────────┐ ──▶ ┌────────────┐ ──▶ Signals    │
│               │  train.py  │     │predictor.py│                 │
│               └────────────┘     └────────────┘                 │
│                     │                  │                         │
│                     ▼                  ▼                         │
│              ┌─────────────┐    ┌─────────────┐                  │
│              │   Model     │    │  Inference  │                  │
│              │  Artifacts  │    │   Engine    │                  │
│              └─────────────┘    └─────────────┘                  │
│                                                                  │
│  Model Outputs:                                                  │
│  • feature_importance.csv                                        │
│  • trained model weights                                         │
└─────────────────────────────────────────────────────────────────┘
```

### 4. Strategy Layer (`src/strategy/`)

Converts model signals into trading decisions.

```
┌─────────────────────────────────────────┐
│           STRATEGY LAYER                │
├─────────────────────────────────────────┤
│                                         │
│  ┌─────────────────────────────────┐   │
│  │         strategy.py             │   │
│  │  • Signal interpretation        │   │
│  │  • Position sizing              │   │
│  │  • Risk management              │   │
│  └─────────────────────────────────┘   │
│                  │                      │
│                  ▼                      │
│  ┌─────────────────────────────────┐   │
│  │      trend_momentum.py          │   │
│  │  • Trend following logic        │   │
│  │  • Momentum filters             │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

### 5. Execution Engine (`src/execution/`)

Handles order management and trade execution.

```
┌─────────────────────────────────────────┐
│          EXECUTION ENGINE               │
├─────────────────────────────────────────┤
│                                         │
│  Strategy ──▶ ┌─────────────┐ ──▶ Broker│
│   Signal      │  trader.py  │    Orders │
│               └─────────────┘           │
│                     │                   │
│                     ▼                   │
│              ┌─────────────┐            │
│              │ Order Queue │            │
│              │ • Validation│            │
│              │ • Execution │            │
│              │ • Tracking  │            │
│              └─────────────┘            │
└─────────────────────────────────────────┘
```

### 6. Backtesting (`src/backtest/`)

Historical strategy evaluation and performance analysis.

```
┌─────────────────────────────────────────────────────────────────┐
│                     BACKTEST ENGINE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Historical ──▶ ┌────────────┐ ──▶ ┌────────────┐ ──▶ Reports  │
│     Data        │backtest.py │     │visualization│               │
│                 └────────────┘     └────────────┘               │
│                       │                  │                       │
│                       ▼                  ▼                       │
│               ┌─────────────┐    ┌─────────────┐                 │
│               │  Simulated  │    │ Performance │                 │
│               │   Trades    │    │  Metrics    │                 │
│               └─────────────┘    └─────────────┘                 │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
                              COMPLETE DATA FLOW
                              
    ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
    │  Fyers  │────▶│   Raw   │────▶│Processed│────▶│Features │
    │   API   │     │  Data   │     │  Data   │     │         │
    └─────────┘     └─────────┘     └─────────┘     └────┬────┘
                                                         │
    ┌─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│  Model  │────▶│ Signal  │────▶│Strategy │────▶│  Order  │
│Prediction│    │Generation│    │ Logic   │     │Execution│
└─────────┘     └─────────┘     └─────────┘     └─────────┘
```

## Directory Structure

```
finstreet/
├── config/                 # Configuration files
│   └── settings.py
├── data/
│   ├── raw/               # Raw market data
│   └── processed/         # Cleaned and processed data
├── docs/                  # Documentation
│   └── architecture.md
├── models/                # Trained models and artifacts
├── reports/               # Generated reports and figures
│   └── figures/
├── src/
│   ├── backtest/          # Backtesting engine
│   ├── data/              # Data acquisition
│   ├── execution/         # Trade execution
│   ├── features/          # Feature engineering
│   ├── model/             # ML model training
│   ├── strategy/          # Trading strategies
│   └── visualization/     # Performance visualization
├── requirements.txt
├── run.py                 # Main entry point
└── README.md
```

## Key Interfaces

### Data Interface
```python
# Expected OHLCV format
columns = ["timestamp", "open", "high", "low", "close", "volume"]
```

### Signal Interface
```python
# Model output format
signal = {
    "direction": 1 | -1 | 0,  # Long, Short, Neutral
    "confidence": float,       # 0.0 to 1.0
    "timestamp": datetime
}
```

### Trade Interface
```python
# Trade record format
trade = {
    "entry_time": datetime,
    "exit_time": datetime,
    "direction": str,
    "entry_price": float,
    "exit_price": float,
    "quantity": int,
    "pnl": float
}
```
