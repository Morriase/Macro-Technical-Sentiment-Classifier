"""
Configuration settings for the Macro-Technical Sentiment Forex Classifier
"""
import torch
import os
from pathlib import Path
from dotenv import load_dotenv

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent

# Load environment variables from .env file
load_dotenv(PROJECT_ROOT / ".env")

# Detect if running on Kaggle
IS_KAGGLE = os.path.exists('/kaggle/input')

# LSTM toggle: Enable for full hybrid ensemble
# T4 GPU on Kaggle has enough VRAM (16GB) for LSTM training
USE_LSTM = True

# BASELINE_MODE: Set True to run XGBoost-only baseline for comparison
# Expert recommendation: compare LSTM+XGB vs XGB-only to see if LSTM adds value
BASELINE_MODE = False  # Set True to skip LSTM entirely

if IS_KAGGLE:
    # Kaggle paths - updated dataset location
    DATA_DIR = Path("/kaggle/input/macros-and-ohlc")
    MODELS_DIR = Path("/kaggle/working/models")
    LOGS_DIR = Path("/kaggle/working/logs")
    RESULTS_DIR = Path("/kaggle/working/results")
else:
    # Local paths
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    LOGS_DIR = PROJECT_ROOT / "logs"
    RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist (except input data on Kaggle)
for directory in [MODELS_DIR, LOGS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
if not IS_KAGGLE:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

# API Keys (Load from environment variables)
OANDA_API_KEY = os.getenv("OANDA_API_KEY", "")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
TRADING_ECONOMICS_API_KEY = os.getenv("TRADING_ECONOMICS_API_KEY", "")
# Free tier: 100 requests/day
MARKETAUX_API_KEY = os.getenv("MARKETAUX_API_KEY", "")
# FRED API (Federal Reserve Economic Data) - 120 requests/minute
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# Currency Pairs Configuration
# First batch (training now)
CURRENCY_PAIRS = ["EUR_USD"]
# Second batch (train later): ["XAU_USD", "USD_CAD", "USD_CHF", "NZD_USD"]

PRIMARY_PAIR = "EUR_USD"  # Primary pair for training and evaluation

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

# Live Sentiment Configuration (for inference server)
# Disable by default for performance (adds 1-2s latency)
ENABLE_LIVE_SENTIMENT = False

# Cache duration for live sentiment (minutes)
# Higher values reduce API calls for high-frequency EAs:
# - 5 min: ~288 requests/day if predicting every 5 min → OK for free tier (100/day)
# - 60 min: ~24 requests/day → Very safe for free tier
# Sentiment doesn't change rapidly, so longer cache is fine for trading
SENTIMENT_CACHE_MINUTES = 60  # Default to hourly cache for rate limiting safety

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
# OPTIMIZED FOR: 30GB RAM, prevent overfitting, stable loss convergence
ENSEMBLE_CONFIG = {
    "base_learners": {
        "xgboost": {
            # Tree parameters - conservative for generalization
            "n_estimators": 1000,       # Increased for better convergence
            "max_depth": 6,             # Increased from 4 to capture more complexity
            "learning_rate": 0.05,      # Increased from 0.02 for faster learning
            "min_child_weight": 3,      # Reduced to allow more specific splits
            "gamma": 0.1,               # Reduced - less conservative
            # Sampling - aggressive subsampling to reduce variance
            "subsample": 0.8,           # Increased slightly
            "colsample_bytree": 0.8,    # Increased slightly
            "colsample_bylevel": 0.8,   # Increased slightly
            # Regularization - strong L1/L2 to prevent overfitting
            "reg_alpha": 0.1,           # Reduced L1
            "reg_lambda": 1.0,          # Reduced L2
            # Training settings
            "random_state": 42,
            "early_stopping_rounds": 50,  # Stop if no improvement
            "eval_metric": "mlogloss",
            # Memory optimization
            "max_bin": 256,             # Standard bins for better accuracy
            "tree_method": "hist",      # Histogram-based for memory efficiency
            # BINARY CLASSIFICATION UPDATE
            "num_class": 2,             # Changed from 3 to 2 (Buy/Sell)
        },
        "lstm": {
            # Architecture - EXTENDED HORIZON for better pattern recognition
            # FIX: 30 steps × 5min = 2.5 hours is too short for macro-technical patterns
            # NEW: 100 steps × 5min = 8.3 hours (matches forward_window prediction horizon)
            "sequence_length": 100,     # CRITICAL FIX: Extended from 30 to 100
            "hidden_size": 48,          # Smaller - prevents memorization
            "num_layers": 1,            # Single layer - simpler = better generalization
            "bidirectional": False,
            "hidden_activation": None,  # NO activation - LSTM gates provide non-linearity

            # Regularization - AGGRESSIVE to prevent overfitting
            "use_batch_norm": True,     # ENABLED - stabilizes training
            "dropout": 0.3,             # INCREASED - strong regularization needed

            # Weight regularization - STRONGER
            "l1_lambda": 1e-5,          # Increased L1 for sparsity
            "l2_lambda": 1e-2,          # 10x stronger L2 to prevent overfitting

            "label_smoothing": 0.1,     # Increased - forex is noisy

            # Learning rate - LOWER for more stable learning
            "learning_rate": 5e-5,      # Reduced from 1e-4
            "lr_warmup_epochs": 3,      # Shorter warmup
            "lr_min_factor": 0.1,       # Higher minimum to prevent overfitting

            # Training schedule - LARGER BATCHES for better generalization
            "batch_size": 16384,        # Larger batch = smoother gradients
            "epochs": 500,
            "early_stopping_patience": 8,  # CRITICAL: Stop much earlier

            # Optimizer
            "optimizer": "adamw",
            "beta1": 0.9,
            "beta2": 0.999,
            "max_grad_norm": 0.5,       # Tighter gradient clipping

            # Classification
            "num_classes": 2,
            "gradient_accumulation_steps": 1,
        },
    },
    "meta_learner": {
        "type": "xgboost",
        "n_estimators": 150,            # Slightly more for meta learning
        "max_depth": 3,                 # Shallow - only 6 input features
        "learning_rate": 0.03,          # Conservative
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.2,
        "reg_lambda": 1.5,
        "random_state": 42,
        "early_stopping_rounds": 30,
    },
    # Memory management settings - 30GB RAM limit
    "memory": {
        "max_train_samples": 1500000,   # Can use more with 30GB RAM
        "max_val_samples": 300000,      # More validation samples
        "use_float32": True,            # Use float32 (50% RAM savings)
        "aggressive_gc": True,          # Garbage collect after each fold
    },
}

# Walk-Forward Optimization Configuration
# OPTIMIZED FOR: 30GB RAM with both XGBoost and LSTM training
WFO_CONFIG = {
    "train_window_months": 4,       # Reduced from 6 - less data per fold
    "test_window_months": 1,        # Reduced from 2 - faster iteration
    "step_months": 1,               # Smaller steps for more folds
    "min_train_samples": 2000,      # Reduced minimum
    "cv_folds": 3,                  # 3 folds for OOF generation
    "max_samples_per_fold": 1000000,  # Increased to use full folds
}

# Hyperparameter Optimization
OPTUNA_CONFIG = {
    # Reduced to 2 for Kaggle RAM/time; set to 5+ locally
    "n_trials": 2 if IS_KAGGLE else 5,
    "timeout": 3600,  # 1 hour
    "n_jobs": 1,
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
    # Fuzzy Logic Quality Scoring
    "use_fuzzy_quality": True,  # Enable fuzzy logic signal quality filtering
    "min_quality_threshold": 40,  # Minimum quality score to trade (0-100)
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

# GPU/CUDA Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = torch.cuda.is_available()
CUDA_DEVICE_COUNT = torch.cuda.device_count() if USE_CUDA else 0

# GPU Optimization Settings
# OPTIMIZED FOR: 2x NVIDIA T4 GPUs (30GB RAM)
# Goal: Maximize GPU utilization - make them SCREAM
GPU_CONFIG = {
    "device": DEVICE,
    "use_cuda": USE_CUDA,
    "num_gpus": CUDA_DEVICE_COUNT,
    # Data loading - USE WORKERS to keep GPUs fed
    # 4 workers to prefetch data (CPU → GPU pipeline)
    "num_workers": 4,
    "pin_memory": True,             # Pin memory for faster CPU→GPU transfer
    "prefetch_factor": 2,           # Prefetch 2 batches per worker
    # Mixed precision - CRITICAL for T4 (Tensor Cores love FP16)
    "mixed_precision": USE_CUDA,
    "cudnn_benchmark": USE_CUDA,    # Enable cuDNN auto-tuner
    # NO gradient accumulation - we want BIG batches to saturate GPUs
    "gradient_accumulation_steps": 1,
    # Memory limits - T4 has 16GB each, use it!
    "max_gpu_memory_fraction": 0.95,   # Use 95% of GPU memory
    "empty_cache_frequency": 50,       # Less frequent cache clearing
}

# Set cuDNN optimization flags
if USE_CUDA:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False  # Trade determinism for speed

# Class Imbalance Handling
IMBALANCE_CONFIG = {
    "strategy": "SMOTE",  # or "class_weights", "threshold_adjustment"
    "smote_k_neighbors": 5,
    # Increased from 2.0 to reduce HOLD bias
    "cost_sensitive_weights": {"Buy": 3.0, "Sell": 3.0, "Hold": 1.0},
    "target_threshold_atr": 1.0,  # Min ATR move to classify as Buy/Sell
}

# Target Variable Configuration
TARGET_CONFIG = {
    # "ternary" (Buy/Sell/Hold) or "binary" (Up/Down)
    "classification_type": "binary",
    # REDUCED from 24h to 8h per expert: shorter horizons more predictable from M5 data
    "forward_window_hours": 8,
    # CRITICAL FIX: Set to 0.0 to keep ALL data continuous
    # Filtering 73% of rows breaks LSTM's temporal learning (jumps in time)
    # Instead, use class_weights to de-prioritize small moves
    "min_move_threshold_pips": 0.0,
    "atr_multiplier": 0.6,  # Only used if min_move_threshold_pips is None
}
