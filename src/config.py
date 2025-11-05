"""
Configuration settings for the Macro-Technical Sentiment Forex Classifier
"""
import os
from pathlib import Path

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API Keys (Load from environment variables)
OANDA_API_KEY = os.getenv("OANDA_API_KEY", "")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
TRADING_ECONOMICS_API_KEY = os.getenv("TRADING_ECONOMICS_API_KEY", "")

# Currency Pairs Configuration
CURRENCY_PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"]
PRIMARY_PAIR = "EUR_USD"

# Data Acquisition Settings
FX_DATA_GRANULARITY = "M5"  # 5-minute candles for high-fidelity
FX_TRAINING_TIMEFRAME = "H4"  # 4-hour prediction timeframe
HISTORICAL_YEARS = 5  # Years of historical data to fetch

# Technical Indicators Configuration
TECHNICAL_INDICATORS = {
    "EMA_PERIODS": [50, 100, 200],
    "RSI_PERIOD": 14,
    "ATR_PERIOD": 14,
    "STOCHASTIC_PERIODS": {"fastk_period": 14, "slowk_period": 3, "slowd_period": 3},
    "MACD_PERIODS": {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
    "BOLLINGER_PERIOD": 20,
    "BOLLINGER_STD": 2,
}

# Macroeconomic Events Configuration
HIGH_IMPACT_EVENTS = [
    "Non-Farm Payroll",
    "CPI",
    "Interest Rate Decision",
    "GDP",
    "Unemployment Rate",
    "Retail Sales",
]

# NLP/Sentiment Configuration
SENTIMENT_MODEL = "ProsusAI/finbert"  # FinBERT from HuggingFace
SENTIMENT_EMA_PERIODS = [3, 7, 14]  # Days for time-weighted sentiment
MAX_NEWS_PER_DAY = 50
LDA_NUM_TOPICS = 10  # Latent Dirichlet Allocation topics

# COT Data Configuration
COT_FUTURES_CONTRACTS = {
    "EUR": "Euro FX",
    "GBP": "British Pound",
    "JPY": "Japanese Yen",
    "AUD": "Australian Dollar",
}
COT_NORMALIZATION_WINDOW = 156  # 3 years of weekly data

# Feature Engineering
LAGGED_PERIODS = [1, 2, 3, 5, 10]  # Lag periods for autoregressive features
PCA_VARIANCE_THRESHOLD = 0.95  # Variance to retain in PCA
VOLATILITY_WINDOWS = [5, 10, 22, 50]  # Windows for realized volatility

# Model Architecture - Stacking Ensemble (XGBoost + LSTM + XGBoost Meta)
ENSEMBLE_CONFIG = {
    "base_learners": {
        "xgboost": {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0.1,
            "reg_alpha": 0.05,
            "reg_lambda": 1.0,
            "random_state": 42,
        },
        "lstm": {
            "sequence_length": 22,  # ~1 month of trading days
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.3,
            "learning_rate": 0.001,
            "batch_size": 64,
            "epochs": 100,
            "early_stopping_patience": 10,
        },
    },
    "meta_learner": {
        "type": "xgboost",  # XGBoost meta-classifier
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "batch_size": 64,
        "early_stopping_patience": 10,
    },
}

# Walk-Forward Optimization Configuration
WFO_CONFIG = {
    "train_window_years": 2,  # In-sample period
    "test_window_months": 6,  # Out-of-sample period
    "step_months": 6,  # Rolling step size
    "min_train_samples": 5000,
    "cv_folds": 5,  # For nested cross-validation
}

# Hyperparameter Optimization
OPTUNA_CONFIG = {
    "n_trials": 100,
    "timeout": 7200,  # 2 hours
    "n_jobs": -1,
    "optimization_metric": "profit_factor",  # or "sharpe_ratio"
}

# Risk Management Parameters
RISK_MANAGEMENT = {
    "base_confidence_threshold": 0.70,
    "confidence_sensitivity": 0.10,  # ATR multiplier for dynamic threshold
    "stop_loss_multiplier": 1.5,
    "risk_reward_ratio": 2.0,
    "max_time_bars": 6,  # Maximum 24 hours (6 x 4H bars)
    "max_position_size": 0.02,  # 2% of capital per trade
    "max_drawdown_threshold": 0.15,  # 15% max drawdown alert
}

# Trade Entry Optimization Ranges (for WFO)
OPTIMIZATION_RANGES = {
    "prediction_horizon": [4, 8, 12, 24],  # Hours
    "confidence_threshold": [0.65, 0.85],
    "stop_loss_multiplier": [1.0, 3.0],
    "risk_reward_ratio": [1.5, 3.0],
    "time_based_exit": [4, 12],  # Bars
}

# Execution Simulation (Backtesting)
EXECUTION_CONFIG = {
    "spread_pips": {"EUR_USD": 0.8, "GBP_USD": 1.2, "USD_JPY": 0.8, "AUD_USD": 1.0},
    "slippage_pips": 1.5,
    "commission_per_lot": 3.5,  # USD per standard lot
    "initial_capital": 100000,  # USD
}

# Data Quality Thresholds
DATA_QUALITY = {
    "min_accuracy": 0.98,
    "max_gap_hours": 24,
    "outlier_std_threshold": 5.0,
}

# Monitoring and Alerts
MONITORING_CONFIG = {
    "performance_check_interval": "daily",
    "mdd_alert_threshold": 0.50,  # 50% of backtested MDD
    "profit_factor_min": 1.2,
    "feature_importance_shift_threshold": 0.20,  # 20% shift in top features
    "auto_retrain_trigger": True,
}

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}"

# Class Imbalance Handling
IMBALANCE_CONFIG = {
    "strategy": "SMOTE",  # or "class_weights", "threshold_adjustment"
    "smote_k_neighbors": 5,
    "cost_sensitive_weights": {"Buy": 2.0, "Sell": 2.0, "Hold": 1.0},
    "target_threshold_atr": 1.0,  # Min ATR move to classify as Buy/Sell
}

# Target Variable Configuration
TARGET_CONFIG = {
    # "ternary" (Buy/Sell/Hold) or "binary" (Up/Down)
    "classification_type": "ternary",
    "forward_window_hours": 24,
    "min_move_threshold_pips": 10,  # Minimum move to classify as directional
}
