"""
Hybrid Stacking Ensemble: XGBoost + LSTM + XGBoost Meta-Classifier
Architecture from required.txt:
- Level-0: XGBoost (tabular features) + LSTM (sequence modeling)
- Level-1: XGBoost Meta-Classifier (combines base learner outputs)
"""
from src.models.lstm_model import LSTMSequenceModel
from src.config import ENSEMBLE_CONFIG, USE_LSTM, IS_KAGGLE, BASELINE_MODE
from loguru import logger
from typing import Dict, List, Tuple, Optional
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class HybridEnsemble:
    """
    Two-level stacking ensemble combining XGBoost and LSTM
    OPTIMIZED: 30GB RAM, proper regularization, stable training
    """

    def __init__(
        self,
        xgb_params: Optional[Dict] = None,
        lstm_params: Optional[Dict] = None,
        meta_xgb_params: Optional[Dict] = None,
        n_folds: int = 3,
        random_state: int = 42,
        memory_config: Optional[Dict] = None,
    ):
        """
        Initialize hybrid ensemble

        Args:
            xgb_params: XGBoost base learner hyperparameters
            lstm_params: LSTM hyperparameters
            meta_xgb_params: Meta-classifier XGBoost hyperparameters
            n_folds: Number of folds for OOF predictions
            random_state: Random seed
            memory_config: Memory management settings
        """
        self.n_folds = n_folds
        self.random_state = random_state

        # Memory management settings
        self.memory_config = memory_config or ENSEMBLE_CONFIG.get("memory", {
            "max_train_samples": 40000,
            "max_val_samples": 10000,
            "use_float32": True,
            "aggressive_gc": True,
        })

        # Base Learner 1: XGBoost for tabular features
        # OPTIMIZED: Conservative parameters to prevent overfitting
        xgb_config = ENSEMBLE_CONFIG.get(
            "base_learners", {}).get("xgboost", {})
        self.xgb_params = xgb_params or {
            "objective": "multi:softprob",
            "num_class": 3,
            "max_depth": xgb_config.get("max_depth", 4),
            "learning_rate": xgb_config.get("learning_rate", 0.02),
            "n_estimators": xgb_config.get("n_estimators", 500),
            "subsample": xgb_config.get("subsample", 0.7),
            "colsample_bytree": xgb_config.get("colsample_bytree", 0.7),
            "colsample_bylevel": xgb_config.get("colsample_bylevel", 0.7),
            "min_child_weight": xgb_config.get("min_child_weight", 5),
            "gamma": xgb_config.get("gamma", 0.2),
            "reg_alpha": xgb_config.get("reg_alpha", 0.3),
            "reg_lambda": xgb_config.get("reg_lambda", 2.0),
            "max_bin": xgb_config.get("max_bin", 128),
            "scale_pos_weight": 1.0,
            "eval_metric": "mlogloss",
            "early_stopping_rounds": xgb_config.get("early_stopping_rounds", 50),
            "random_state": random_state,
            "tree_method": "hist",
            "device": "cuda" if USE_CUDA else "cpu",
            "n_jobs": 1 if USE_CUDA else -1,
        }

        self.xgb_base = xgb.XGBClassifier(**self.xgb_params)

        # Base Learner 2: LSTM for sequence modeling
        # OPTIMIZED: MQL5_LSTM.mq5 architecture parameters
        # Structure: Input → BatchNorm → LSTM(40) → Output(3)
        # MQL5 test file has Output(2), but production uses 3 classes: Buy(2), Sell(0), Hold(1)
        lstm_config = ENSEMBLE_CONFIG.get("base_learners", {}).get("lstm", {})
        self.lstm_params = lstm_params or {
            # BarsToLine
            "sequence_length": lstm_config.get("sequence_length", 40),
            # HiddenLayer
            "hidden_size": lstm_config.get("hidden_size", 40),
            # 1 LSTM layer
            "num_layers": lstm_config.get("num_layers", 1),
            "num_classes": 3,
            # No dropout (BatchNorm replaces)
            "dropout": lstm_config.get("dropout", 0.0),
            # MQL5: 3e-5
            "learning_rate": lstm_config.get("learning_rate", 3e-5),
            # MQL5: 10000
            "batch_size": lstm_config.get("batch_size", 10000),
            # MQL5: 500
            "epochs": lstm_config.get("epochs", 500),
            # MQL5: 20
            "early_stopping_patience": lstm_config.get("early_stopping_patience", 20),
            "l1_lambda": lstm_config.get("l1_lambda", 1e-7),
            "l2_lambda": lstm_config.get("l2_lambda", 1e-5),
            "label_smoothing": lstm_config.get("label_smoothing", 0.1),
            "lr_warmup_epochs": lstm_config.get("lr_warmup_epochs", 3),
            "lr_min_factor": lstm_config.get("lr_min_factor", 0.01),
            "max_grad_norm": lstm_config.get("max_grad_norm", 1.0),
            "gradient_accumulation_steps": lstm_config.get("gradient_accumulation_steps", 1),
            "bidirectional": lstm_config.get("bidirectional", False),
            # MQL5: BatchNorm enabled
            "use_batch_norm": lstm_config.get("use_batch_norm", True),
            # MQL5: Swish
            "hidden_activation": lstm_config.get("hidden_activation", "swish"),
        }

        self.lstm_base = None  # Initialized during fit

        # Meta-Classifier: XGBoost
        meta_config = ENSEMBLE_CONFIG.get("meta_learner", {})
        self.meta_xgb_params = meta_xgb_params or {
            "objective": "multi:softprob",
            "num_class": 3,
            "max_depth": meta_config.get("max_depth", 3),
            "learning_rate": meta_config.get("learning_rate", 0.03),
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "eval_metric": "mlogloss",
            "random_state": random_state,
            # Use GPU for training (XGBoost 2.0+: use 'hist' with device='cuda')
            "tree_method": "hist",
            "device": "cuda",
            "n_jobs": 1,  # Must be 1 when using GPU
        }

        self.meta_classifier = xgb.XGBClassifier(**self.meta_xgb_params)

        # Scaling
        self.scaler = StandardScaler()

        # Tracking
        self.is_fitted = False
        self.feature_importance_ = None

        # Reduced logging for cleaner output
        # logger.info("Hybrid Ensemble initialized: XGBoost + LSTM + XGBoost Meta")

    def _xgb_predict_proba(self, model: xgb.XGBClassifier, X: np.ndarray) -> np.ndarray:
        """
        GPU-compatible XGBoost prediction using DMatrix.
        Avoids device mismatch warning when model is on CUDA.
        """
        dmatrix = xgb.DMatrix(X)
        # get_booster().predict returns flattened probabilities, reshape to (n_samples, n_classes)
        preds = model.get_booster().predict(dmatrix, output_margin=False)
        n_classes = model.n_classes_
        return preds.reshape(-1, n_classes).astype(np.float32)

    def generate_out_of_fold_predictions(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate out-of-fold predictions from base learners
        Prevents data leakage to meta-learner
        MEMORY OPTIMIZED: Process one model at a time, aggressive cleanup

        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)

        Returns:
            Tuple of (xgb_oof_proba, lstm_oof_proba)
            Each has shape (n_samples, n_classes)
        """
        import gc
        import torch

        n_samples, n_features = X.shape
        n_classes = 3

        # Initialize OOF prediction arrays
        xgb_oof_proba = np.zeros((n_samples, n_classes), dtype=np.float32)
        lstm_oof_proba = np.zeros((n_samples, n_classes), dtype=np.float32)

        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=self.n_folds)

        # Get memory limits from config
        max_train_samples = self.memory_config.get("max_train_samples", 40000)
        max_val_samples = self.memory_config.get("max_val_samples", 10000)
        aggressive_gc = self.memory_config.get("aggressive_gc", True)

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(
                f"OOF fold {fold_idx + 1}/{self.n_folds} - Train: {len(train_idx):,} samples, Val: {len(val_idx):,} samples")

            # Subsample for memory efficiency on large datasets
            if len(train_idx) > max_train_samples:
                # Take last N samples (most recent data for time series)
                train_idx = train_idx[-max_train_samples:]
                logger.info(
                    f"    Subsampled train to {len(train_idx):,} samples")

            if len(val_idx) > max_val_samples:
                # Take last N samples for validation too
                val_idx = val_idx[-max_val_samples:]
                logger.info(f"    Subsampled val to {len(val_idx):,} samples")

            # Use float32 to reduce memory
            X_train_fold = X[train_idx].astype(np.float32)
            y_train_fold = y[train_idx]
            X_val_fold = X[val_idx].astype(np.float32)
            y_val_fold = y[val_idx]

            # ===== XGBoost FIRST (then delete) =====
            logger.info(f"  → Training XGBoost (fold {fold_idx + 1})...")
            from sklearn.utils.class_weight import compute_sample_weight
            fold_sample_weights = compute_sample_weight(
                class_weight={0: 3.0, 1: 3.0, 2: 1.0},
                y=y_train_fold
            )

            xgb_fold = xgb.XGBClassifier(**self.xgb_params)
            xgb_fold.fit(
                X_train_fold, y_train_fold,
                sample_weight=fold_sample_weights,
                # Only val set to save memory
                eval_set=[(X_val_fold, y[val_idx])],
                verbose=50,  # Log every 50 rounds for OOF folds
            )
            # Use helper for GPU-compatible prediction
            xgb_oof_proba[val_idx] = self._xgb_predict_proba(
                xgb_fold, X_val_fold)

            # Log XGBoost fold accuracy
            xgb_pred = np.argmax(xgb_oof_proba[val_idx], axis=1)
            xgb_acc = (xgb_pred == y[val_idx]).mean()
            logger.info(
                f"    XGBoost fold {fold_idx + 1} Val Accuracy: {xgb_acc:.4f}")

            # DELETE XGBoost immediately
            del xgb_fold, fold_sample_weights
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # ===== LSTM SECOND (then delete) =====
            logger.info(f"  → Training LSTM (fold {fold_idx + 1})...")
            lstm_fold = LSTMSequenceModel(
                input_size=n_features, **self.lstm_params
            )
            lstm_fold.fit(X_train_fold, y_train_fold, X_val_fold, y[val_idx])
            lstm_proba_fold = lstm_fold.predict_proba(X_val_fold)

            # Log LSTM fold accuracy
            lstm_pred = np.argmax(lstm_proba_fold, axis=1)
            # Align with val_idx (LSTM has sequence offset)
            y_val_aligned = y[val_idx][-len(lstm_pred):]
            lstm_acc = (lstm_pred == y_val_aligned).mean()
            logger.info(
                f"    LSTM fold {fold_idx + 1} Val Accuracy: {lstm_acc:.4f}")

            # Handle shape mismatch
            seq_len = self.lstm_params.get("sequence_length", 10)
            if len(lstm_proba_fold) < len(val_idx):
                n_missing = len(val_idx) - len(lstm_proba_fold)
                uniform_proba = np.full(
                    (n_missing, lstm_proba_fold.shape[1]), 1.0 / lstm_proba_fold.shape[1], dtype=np.float32)
                lstm_proba_fold = np.vstack([uniform_proba, lstm_proba_fold])

            lstm_oof_proba[val_idx] = lstm_proba_fold.astype(np.float32)

            # DELETE LSTM and fold data immediately
            del lstm_fold, lstm_proba_fold
            del X_train_fold, y_train_fold, X_val_fold
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"Fold {fold_idx + 1} completed, memory cleared")

        return xgb_oof_proba, lstm_oof_proba

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        save_plots_path: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Train the hybrid ensemble

        Args:
            X: Training features
            y: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            save_plots_path: Path prefix for saving training plots (optional)
            feature_names: List of feature column names (CRITICAL for inference alignment)
        """
        # logger.info("Training Hybrid Ensemble")
        # logger.info(f"Training samples: {len(X)}, Features: {X.shape[1]}")
        logger.info(
            f"Hybrid Ensemble fit: X.shape={X.shape}, y.shape={y.shape}")

        # Store feature metadata (CRITICAL for inference server validation)
        self.n_features_ = X.shape[1]
        self.feature_names_ = feature_names if feature_names is not None else [
            f"feature_{i}" for i in range(self.n_features_)]

        if len(self.feature_names_) != self.n_features_:
            raise ValueError(
                f"Feature names length ({len(self.feature_names_)}) must match X.shape[1] ({self.n_features_})")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Memory check - skip OOF for large datasets
        max_samples_for_oof = 50000
        if len(X) > max_samples_for_oof:
            logger.info(
                f"Dataset too large ({len(X):,} samples) for OOF - using simple split")

            # CRITICAL: Add gap between train/val to prevent LSTM sequence leakage
            # Gap must be >= sequence_length + forward_window to prevent ANY overlap
            # sequence_length = lookback window, forward_window = prediction horizon
            seq_length = self.lstm_params.get('sequence_length', 40)
            # forward_window_hours=8 / 4h_bars = 2 bars, but we use M5 so: 8*12=96 M5 bars
            # 8 hours of M5 data (expert recommendation: test shorter horizon)
            forward_bars = 96
            gap = seq_length + forward_bars  # 40 + 96 = 136 samples minimum

            # Use last 20% as holdout for meta-learner training
            split_idx = int(len(X_scaled) * 0.8)
            X_train_full = X_scaled[:split_idx - gap]  # Leave gap before split
            y_train_full = y[:split_idx - gap]
            X_holdout_full = X_scaled[split_idx:]  # Holdout starts after gap
            y_holdout_full = y[split_idx:]

            logger.info(
                f"  Train/Val gap: {gap} samples (prevents LSTM sequence leakage)")

            # Subsample using STRATIFIED RANDOM SAMPLING (not tail!) for balanced classes
            max_train = 50_000  # Reduced for Kaggle RAM
            max_holdout = 10_000

            if len(X_train_full) > max_train:
                # Use stratified sampling to maintain class balance
                from sklearn.model_selection import train_test_split
                _, X_train_base, _, y_train_base = train_test_split(
                    X_train_full, y_train_full,
                    test_size=max_train,
                    stratify=y_train_full,
                    random_state=self.random_state
                )
                logger.info(
                    f"  Stratified sample: {max_train:,} training samples")
            else:
                X_train_base = X_train_full
                y_train_base = y_train_full

            if len(X_holdout_full) > max_holdout:
                # Take from END of holdout (most recent data for validation)
                X_holdout = X_holdout_full[-max_holdout:]
                y_holdout = y_holdout_full[-max_holdout:]
                logger.info(f"  Holdout: last {max_holdout:,} samples")
            else:
                X_holdout = X_holdout_full
                y_holdout = y_holdout_full

            # Clean up large arrays
            del X_train_full, y_train_full, X_holdout_full, y_holdout_full
            import gc
            gc.collect()

            # Train base learners on subsampled data
            logger.info("Training XGBoost base learner...")
            from sklearn.utils.class_weight import compute_sample_weight
            # Strong class weights to match 74/13/13 imbalance (74/13 ≈ 5.7)
            sample_weights = compute_sample_weight(
                class_weight={0: 5.5, 1: 5.5, 2: 1.0},
                y=y_train_base
            )
            self.xgb_base.fit(
                X_train_base, y_train_base,
                sample_weight=sample_weights,
                # Very small eval set to reduce memory
                eval_set=[(X_holdout[:3000], y_holdout[:3000])],
                verbose=50,
            )

            # LSTM training (skip if BASELINE_MODE or USE_LSTM=False)
            # Expert: compare XGBoost-only vs hybrid to see if LSTM adds value
            if USE_LSTM and not BASELINE_MODE:
                logger.info("Training LSTM base learner...")
                self.lstm_base = LSTMSequenceModel(
                    input_size=X_scaled.shape[1], **self.lstm_params
                )
                self.lstm_base.fit(
                    X_train_base,
                    y_train_base,
                    X_holdout[:5000],
                    y_holdout[:5000],
                    save_plots_path=save_plots_path,
                )
            else:
                if BASELINE_MODE:
                    logger.info(
                        "Skipping LSTM (BASELINE_MODE=True - XGBoost-only experiment)")
                else:
                    logger.info("Skipping LSTM (USE_LSTM=False)")
                self.lstm_base = None

            # Generate meta-features from holdout predictions
            logger.info("Generating meta-features from holdout set...")
            xgb_holdout_proba = self._xgb_predict_proba(
                self.xgb_base, X_holdout)

            if USE_LSTM and not BASELINE_MODE and self.lstm_base is not None:
                lstm_holdout_proba = self.lstm_base.predict_proba(X_holdout)
                # Handle LSTM sequence offset
                if len(lstm_holdout_proba) < len(xgb_holdout_proba):
                    n_missing = len(xgb_holdout_proba) - \
                        len(lstm_holdout_proba)
                    uniform_proba = np.full(
                        (n_missing, 3), 1.0/3, dtype=np.float32)
                    lstm_holdout_proba = np.vstack(
                        [uniform_proba, lstm_holdout_proba])
                meta_features = np.hstack(
                    [xgb_holdout_proba, lstm_holdout_proba])
            else:
                # XGBoost-only: meta-features are just XGBoost probabilities
                meta_features = xgb_holdout_proba

            # Train meta-classifier
            logger.info("Training meta-classifier...")
            self.meta_classifier.fit(meta_features, y_holdout, verbose=10)

        else:
            # Original OOF approach for smaller datasets
            xgb_oof_proba, lstm_oof_proba = self.generate_out_of_fold_predictions(
                X_scaled, y
            )
            meta_features = np.hstack([xgb_oof_proba, lstm_oof_proba])

            # Train base learners on full training set
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weights = compute_sample_weight(
                class_weight={0: 3.0, 1: 3.0, 2: 1.0},
                y=y
            )

            logger.info("Training XGBoost base learner on full dataset")
            self.xgb_base.fit(
                X_scaled, y, sample_weight=sample_weights, verbose=50)

            logger.info("Training LSTM base learner on full dataset")
            self.lstm_base = LSTMSequenceModel(
                input_size=X_scaled.shape[1], **self.lstm_params
            )
            self.lstm_base.fit(X_scaled, y, save_plots_path=save_plots_path)

            # Train meta-classifier
            logger.info("Training meta-classifier...")
            self.meta_classifier.fit(meta_features, y, verbose=10)

        # Extract feature importance
        self.feature_importance_ = {
            "xgb_base": self.xgb_base.feature_importances_,
            "meta": self.meta_classifier.feature_importances_,
        }

        self.is_fitted = True
        logger.info("Hybrid Ensemble training completed")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Class probabilities (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Get base learner predictions (use helper for GPU-compatible XGBoost)
        xgb_proba = self._xgb_predict_proba(self.xgb_base, X_scaled)

        if self.lstm_base is not None:
            lstm_proba = self.lstm_base.predict_proba(X_scaled)
            # Handle LSTM sequence length mismatch (drops first sequence_length-1 samples)
            if len(lstm_proba) < len(xgb_proba):
                n_missing = len(xgb_proba) - len(lstm_proba)
                n_classes = lstm_proba.shape[1]
                # Pad with uniform probabilities for missing samples
                uniform_proba = np.full(
                    (n_missing, n_classes), 1.0 / n_classes)
                lstm_proba = np.vstack([uniform_proba, lstm_proba])
            # Concatenate for meta-learner
            meta_features = np.hstack([xgb_proba, lstm_proba])
        else:
            # XGBoost-only mode
            meta_features = xgb_proba

        # Meta-classifier prediction (use helper for GPU-compatible XGBoost)
        final_proba = self._xgb_predict_proba(
            self.meta_classifier, meta_features)

        return final_proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Predicted labels (n_samples,)
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def get_base_learner_predictions(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get individual base learner predictions for analysis

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Tuple of (xgb_predictions, lstm_predictions)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)

        # Use helper for GPU-compatible XGBoost prediction
        xgb_proba = self._xgb_predict_proba(self.xgb_base, X_scaled)
        lstm_proba = self.lstm_base.predict_proba(X_scaled)

        return xgb_proba, lstm_proba

    def save_model(self, filepath: str):
        """
        Save ensemble to disk

        Args:
            filepath: Base path (will create multiple files)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Cannot save.")

        # Save base learners
        joblib.dump(self.xgb_base, f"{filepath}_xgb_base.pkl")
        if self.lstm_base is not None:
            self.lstm_base.save_model(f"{filepath}_lstm_base.pth")

        # Save meta-classifier
        joblib.dump(self.meta_classifier, f"{filepath}_meta.pkl")

        # Save scaler and config (INCLUDING FEATURE METADATA - CRITICAL!)
        joblib.dump(
            {
                "scaler": self.scaler,
                "xgb_params": self.xgb_params,
                "lstm_params": self.lstm_params,
                "meta_xgb_params": self.meta_xgb_params,
                "n_folds": self.n_folds,
                "random_state": self.random_state,
                "feature_importance_": self.feature_importance_,
                "n_features_": self.n_features_,
                "feature_names_": self.feature_names_,
            },
            f"{filepath}_config.pkl",
        )

        logger.info(f"Hybrid Ensemble saved to {filepath}_*")

    def load_model(self, filepath: str):
        """
        Load ensemble from disk

        Args:
            filepath: Base path
        """
        # Load config
        config = joblib.load(f"{filepath}_config.pkl")
        self.scaler = config["scaler"]
        self.xgb_params = config["xgb_params"]
        self.lstm_params = config["lstm_params"]
        self.meta_xgb_params = config["meta_xgb_params"]
        self.n_folds = config["n_folds"]
        self.random_state = config["random_state"]
        self.feature_importance_ = config["feature_importance_"]

        # Load feature metadata (CRITICAL for inference validation)
        self.n_features_ = config.get("n_features_", None)
        self.feature_names_ = config.get("feature_names_", None)

        # Load base learners
        self.xgb_base = joblib.load(f"{filepath}_xgb_base.pkl")

        # Initialize LSTM and load weights (if LSTM was used during training)
        lstm_path = Path(f"{filepath}_lstm_base.pth")
        if lstm_path.exists():
            input_size = self.xgb_base.n_features_in_
            self.lstm_base = LSTMSequenceModel(
                input_size=input_size, **self.lstm_params
            )
            self.lstm_base.load_model(str(lstm_path))
        else:
            self.lstm_base = None
            logger.info("No LSTM model found - using XGBoost-only mode")

        # Load meta-classifier
        self.meta_classifier = joblib.load(f"{filepath}_meta.pkl")

        self.is_fitted = True
        logger.info(f"Hybrid Ensemble loaded from {filepath}_*")

    def get_feature_importance(self) -> Dict:
        """
        Get feature importance from all models

        Returns:
            Dictionary with importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.feature_importance_


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    # Generate sample time series data
    X, y = make_classification(
        n_samples=5000,
        n_features=50,
        n_informative=30,
        n_classes=3,
        n_clusters_per_class=2,
        random_state=42,
    )

    # Split with temporal ordering preserved
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Initialize ensemble
    ensemble = HybridEnsemble(
        n_folds=3,  # Reduced for faster testing
        random_state=42,
    )

    # Override LSTM params for faster testing
    ensemble.lstm_params["epochs"] = 10
    ensemble.lstm_params["sequence_length"] = 10

    # Train
    ensemble.fit(X_train, y_train, X_test, y_test)

    # Predictions
    y_pred_proba = ensemble.predict_proba(X_test)
    y_pred = ensemble.predict(X_test)

    # Evaluate
    print("\n" + "=" * 50)
    print("HYBRID ENSEMBLE EVALUATION")
    print("=" * 50)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Compare base learner predictions
    xgb_proba, lstm_proba = ensemble.get_base_learner_predictions(X_test)
    xgb_pred = np.argmax(xgb_proba, axis=1)
    lstm_pred = np.argmax(lstm_proba, axis=1)

    print(f"\nXGBoost Accuracy: {accuracy_score(y_test, xgb_pred):.4f}")
    print(f"LSTM Accuracy: {accuracy_score(y_test, lstm_pred):.4f}")
    print(f"Ensemble Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Feature importance
    importance = ensemble.get_feature_importance()
    print(f"\nXGBoost base learner top 5 features:")
    top_features = np.argsort(importance["xgb_base"])[-5:][::-1]
    for idx in top_features:
        print(f"  Feature {idx}: {importance['xgb_base'][idx]:.4f}")
