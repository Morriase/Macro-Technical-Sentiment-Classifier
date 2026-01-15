"""
Hybrid Stacking Ensemble: XGBoost + LSTM + XGBoost Meta-Classifier
Architecture from required.txt:
- Level-0: XGBoost (tabular features) + LSTM (sequence modeling)
- Level-1: XGBoost Meta-Classifier (combines base learner outputs)
"""
from src.models.lstm_model import LSTMSequenceModel
from src.config import ENSEMBLE_CONFIG
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
    """

    def __init__(
        self,
        xgb_params: Optional[Dict] = None,
        lstm_params: Optional[Dict] = None,
        meta_xgb_params: Optional[Dict] = None,
        n_folds: int = 5,
        random_state: int = 42,
    ):
        """
        Initialize hybrid ensemble

        Args:
            xgb_params: XGBoost base learner hyperparameters
            lstm_params: LSTM hyperparameters
            meta_xgb_params: Meta-classifier XGBoost hyperparameters
            n_folds: Number of folds for OOF predictions
            random_state: Random seed
        """
        self.n_folds = n_folds
        self.random_state = random_state

        # Base Learner 1: XGBoost for tabular features
        self.xgb_params = xgb_params or {
            "objective": "multi:softprob",
            "num_class": 3,
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 300,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0.1,
            "reg_alpha": 0.05,
            "reg_lambda": 1.0,
            "scale_pos_weight": 1.0,
            "eval_metric": "mlogloss",
            "random_state": random_state,
            "tree_method": "gpu_hist",  # Use GPU for training (10-100x faster!)
            "device": "cuda",  # Explicitly use CUDA
            "n_jobs": -1,
        }

        self.xgb_base = xgb.XGBClassifier(**self.xgb_params)

        # Base Learner 2: LSTM-CNN Hybrid for sequence modeling
        self.lstm_params = lstm_params or {
            "sequence_length": 22,  # ~1 month of trading days
            "hidden_size": 128,  # Increased for hybrid capacity
            "num_layers": 2,
            "num_classes": 3,
            "dropout": 0.2,  # Reduced per research paper
            "learning_rate": 0.001,  # Paper recommendation
            "batch_size": 64,  # Paper recommendation
            "epochs": 100,
            "early_stopping_patience": 15,  # Increased for hybrid convergence
            # CNN parameters for hybrid architecture
            "cnn_filters": 64,
            "kernel_size": 3,
        }

        self.lstm_base = None  # Initialized during fit

        # Meta-Classifier: XGBoost
        self.meta_xgb_params = meta_xgb_params or {
            "objective": "multi:softprob",
            "num_class": 3,
            "max_depth": 3,  # Shallower than base
            "learning_rate": 0.05,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "eval_metric": "mlogloss",
            "random_state": random_state,
            "tree_method": "gpu_hist",  # Use GPU for training
            "device": "cuda",  # Explicitly use CUDA
            "n_jobs": -1,
        }

        self.meta_classifier = xgb.XGBClassifier(**self.meta_xgb_params)

        # Scaling
        self.scaler = StandardScaler()

        # Tracking
        self.is_fitted = False
        self.feature_importance_ = None

        # Reduced logging for cleaner output
        # logger.info("Hybrid Ensemble initialized: XGBoost + LSTM + XGBoost Meta")

    def generate_out_of_fold_predictions(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate out-of-fold predictions from base learners
        Prevents data leakage to meta-learner

        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)

        Returns:
            Tuple of (xgb_oof_proba, lstm_oof_proba)
            Each has shape (n_samples, n_classes)
        """
        # logger.info(f"Generating OOF predictions with {self.n_folds} folds")

        n_samples, n_features = X.shape
        n_classes = 3

        # Initialize OOF prediction arrays
        xgb_oof_proba = np.zeros((n_samples, n_classes))
        lstm_oof_proba = np.zeros((n_samples, n_classes))

        # CRITICAL: Use Time Series Split instead of StratifiedKFold
        # TimeSeriesSplit respects temporal ordering (no shuffle!)
        # This prevents data leakage and aligns with Walk-Forward Validation spec
        # StratifiedKFold was causing overfitting by training on future data
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=self.n_folds)

        # logger.info(f"Generating OOF predictions with {self.n_folds} folds...")
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            # Only log first and last fold to reduce clutter
            if fold_idx == 0 or (fold_idx + 1) == self.n_folds:
                logger.info(
                    f"OOF prediction on fold {fold_idx + 1}/{self.n_folds}")

            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X[val_idx]

            # Compute sample weights for this fold
            from sklearn.utils.class_weight import compute_sample_weight
            fold_sample_weights = compute_sample_weight(
                class_weight={0: 3.0, 1: 3.0, 2: 1.0},  # BUY=0, SELL=1, HOLD=2
                y=y_train_fold
            )

            # XGBoost base learner
            xgb_fold = xgb.XGBClassifier(**self.xgb_params)
            xgb_fold.fit(
                X_train_fold,
                y_train_fold,
                sample_weight=fold_sample_weights,
                eval_set=[(X_val_fold, y[val_idx])],
                verbose=False,
            )
            xgb_oof_proba[val_idx] = xgb_fold.predict_proba(X_val_fold)

            # LSTM base learner
            lstm_fold = LSTMSequenceModel(
                input_size=n_features, **self.lstm_params
            )
            lstm_fold.fit(X_train_fold, y_train_fold, X_val_fold, y[val_idx])

            # LSTM requires sequence length, so it returns fewer predictions
            # (drops first sequence_length-1 samples)
            lstm_proba_fold = lstm_fold.predict_proba(X_val_fold)

            # Handle shape mismatch: LSTM drops first sequence_length-1 samples
            seq_len = self.lstm_params.get("sequence_length", 22)
            if len(lstm_proba_fold) < len(val_idx):
                # Pad with uniform probabilities for first few samples
                n_missing = len(val_idx) - len(lstm_proba_fold)
                uniform_proba = np.full(
                    (n_missing, lstm_proba_fold.shape[1]), 1.0 / lstm_proba_fold.shape[1])
                lstm_proba_fold = np.vstack([uniform_proba, lstm_proba_fold])

            lstm_oof_proba[val_idx] = lstm_proba_fold

            # Only log first and last fold
            if fold_idx == 0 or (fold_idx + 1) == self.n_folds:
                logger.info(f"Fold {fold_idx + 1}/{self.n_folds} completed")

        # logger.info("OOF predictions generation completed")
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

        # Store feature metadata (CRITICAL for inference server validation)
        self.n_features_ = X.shape[1]
        self.feature_names_ = feature_names if feature_names is not None else [
            f"feature_{i}" for i in range(self.n_features_)]

        if len(self.feature_names_) != self.n_features_:
            raise ValueError(
                f"Feature names length ({len(self.feature_names_)}) must match X.shape[1] ({self.n_features_})")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Step 1: Generate OOF predictions for meta-learner training
        xgb_oof_proba, lstm_oof_proba = self.generate_out_of_fold_predictions(
            X_scaled, y
        )

        # Step 2: Train base learners on full training set
        # Compute sample weights to combat class imbalance
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weights = compute_sample_weight(
            class_weight={0: 3.0, 1: 3.0, 2: 1.0},  # BUY=0, SELL=1, HOLD=2
            y=y
        )

        logger.info("Training XGBoost base learner on full dataset")
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            self.xgb_base.fit(
                X_scaled,
                y,
                sample_weight=sample_weights,
                eval_set=[(X_val_scaled, y_val)],
                verbose=True,
            )
        else:
            self.xgb_base.fit(X_scaled, y, sample_weight=sample_weights)

        logger.info("Training LSTM base learner on full dataset")
        self.lstm_base = LSTMSequenceModel(
            input_size=X_scaled.shape[1], **self.lstm_params
        )

        if X_val is not None and y_val is not None:
            self.lstm_base.fit(X_scaled, y, X_val_scaled,
                               y_val, save_plots_path=save_plots_path)
        else:
            self.lstm_base.fit(X_scaled, y, save_plots_path=save_plots_path)

        # Step 3: Prepare meta-features (OOF predictions)
        # Concatenate probability predictions from both base learners
        # Shape: (n_samples, n_classes * 2)
        meta_features = np.hstack([xgb_oof_proba, lstm_oof_proba])

        logger.info(f"Meta-features shape: {meta_features.shape}")

        # Step 4: Train meta-classifier on OOF predictions
        logger.info("Training XGBoost meta-classifier")
        if X_val is not None and y_val is not None:
            # Generate meta-features for validation set
            xgb_val_proba = self.xgb_base.predict_proba(X_val_scaled)
            lstm_val_proba = self.lstm_base.predict_proba(X_val_scaled)

            # Handle LSTM sequence length mismatch (padding fix)
            if len(xgb_val_proba) != len(lstm_val_proba):
                n_missing = len(xgb_val_proba) - len(lstm_val_proba)
                n_classes = lstm_val_proba.shape[1]
                uniform_proba = np.full(
                    (n_missing, n_classes), 1.0 / n_classes)
                lstm_val_proba = np.vstack([uniform_proba, lstm_val_proba])

            meta_features_val = np.hstack([xgb_val_proba, lstm_val_proba])

            self.meta_classifier.fit(
                meta_features,
                y,
                eval_set=[(meta_features_val, y_val)],
                verbose=True,
            )
        else:
            self.meta_classifier.fit(meta_features, y)

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

        # Get base learner predictions
        xgb_proba = self.xgb_base.predict_proba(X_scaled)
        lstm_proba = self.lstm_base.predict_proba(X_scaled)

        # Handle LSTM sequence length mismatch (drops first sequence_length-1 samples)
        if len(lstm_proba) < len(xgb_proba):
            n_missing = len(xgb_proba) - len(lstm_proba)
            n_classes = lstm_proba.shape[1]
            # Pad with uniform probabilities for missing samples
            uniform_proba = np.full((n_missing, n_classes), 1.0 / n_classes)
            lstm_proba = np.vstack([uniform_proba, lstm_proba])

        # Concatenate for meta-learner
        meta_features = np.hstack([xgb_proba, lstm_proba])

        # Meta-classifier prediction
        final_proba = self.meta_classifier.predict_proba(meta_features)

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

        xgb_proba = self.xgb_base.predict_proba(X_scaled)
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

        # Initialize LSTM and load weights
        # Input size is inferred from XGBoost
        input_size = self.xgb_base.n_features_in_
        self.lstm_base = LSTMSequenceModel(
            input_size=input_size, **self.lstm_params
        )
        self.lstm_base.load_model(f"{filepath}_lstm_base.pth")

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
