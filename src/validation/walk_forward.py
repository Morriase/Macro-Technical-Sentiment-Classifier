"""
Walk-Forward Optimization Framework
Implements time-series aware train/test splits with rolling windows
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Generator, Dict, Optional
from datetime import datetime, timedelta
from loguru import logger
import optuna
from sklearn.metrics import balanced_accuracy_score, f1_score

from src.config import WFO_CONFIG, OPTUNA_CONFIG, RISK_MANAGEMENT


class WalkForwardSplitter:
    """
    Time-series aware train/test splitting
    Maintains temporal ordering to prevent data leakage
    """

    def __init__(
        self,
        train_window_months: int = 12,
        test_window_months: int = 6,
        step_months: int = 6,
        min_train_samples: int = 5000,
    ):
        """
        Initialize walk-forward splitter

        Args:
            train_window_months: In-sample training period (months)
            test_window_months: Out-of-sample testing period (months)
            step_months: Rolling step size (months)
            min_train_samples: Minimum training samples required
        """
        self.train_window_months = train_window_months
        self.test_window_months = test_window_months
        self.step_months = step_months
        self.min_train_samples = min_train_samples

    def split(
        self, df: pd.DataFrame, date_column: str = "timestamp"
    ) -> Generator[Tuple[pd.Index, pd.Index], None, None]:
        """
        Generate walk-forward train/test splits

        Args:
            df: DataFrame with temporal index or date column
            date_column: Name of date column (if not using index)

        Yields:
            Tuples of (train_indices, test_indices)
        """
        if date_column in df.columns:
            dates = pd.to_datetime(df[date_column])
        else:
            dates = pd.to_datetime(df.index)

        min_date = dates.min()
        max_date = dates.max()

        # Initial train window end
        train_end = min_date + pd.DateOffset(months=self.train_window_months)

        split_count = 0

        while train_end < max_date:
            # Test window
            test_start = train_end
            test_end = test_start + \
                pd.DateOffset(months=self.test_window_months)

            if test_end > max_date:
                break

            # Train indices
            train_mask = (dates >= min_date) & (dates < train_end)
            train_indices = df.index[train_mask]

            # Test indices
            test_mask = (dates >= test_start) & (dates < test_end)
            test_indices = df.index[test_mask]

            # Check minimum samples
            if len(train_indices) < self.min_train_samples:
                logger.warning(
                    f"Insufficient training samples: {len(train_indices)} < {self.min_train_samples}"
                )
                break

            split_count += 1
            logger.info(
                f"Split {split_count}: Train [{train_indices[0]} to {train_indices[-1]}], "
                f"Test [{test_indices[0]} to {test_indices[-1]}]"
            )

            yield train_indices, test_indices

            # Roll forward
            train_end += pd.DateOffset(months=self.step_months)


class WalkForwardOptimizer:
    """
    Walk-Forward Optimization with nested cross-validation
    Optimizes hyperparameters within each WF window
    """

    def __init__(
        self,
        model_class,
        wfo_config: Dict = WFO_CONFIG,
        optuna_config: Dict = OPTUNA_CONFIG,
    ):
        """
        Initialize WFO optimizer

        Args:
            model_class: Model class to optimize
            wfo_config: Walk-forward configuration
            optuna_config: Optuna optimization configuration
        """
        self.model_class = model_class
        self.wfo_config = wfo_config
        self.optuna_config = optuna_config

        self.splitter = WalkForwardSplitter(
            train_window_months=wfo_config["train_window_months"],
            test_window_months=wfo_config["test_window_months"],
            step_months=wfo_config["step_months"],
            min_train_samples=wfo_config["min_train_samples"],
        )

        self.results = []

    def objective(
        self,
        trial: optuna.Trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        """
        Optuna objective function for hyperparameter optimization

        Args:
            trial: Optuna trial
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Optimization metric value
        """
        # Suggest hyperparameters for XGBoost base learner
        xgb_params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
            "gamma": trial.suggest_float("gamma", 0.0, 0.5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 0.1),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.0),
            "random_state": self.splitter.min_train_samples,
        }

        # Train model with suggested parameters
        model = self.model_class(xgb_params=xgb_params)
        model.fit(X_train, y_train)

        # Evaluate on validation set
        y_pred = model.predict(X_val)

        # Calculate optimization metric
        metric_name = self.optuna_config["optimization_metric"]

        if metric_name == "balanced_accuracy":
            score = balanced_accuracy_score(y_val, y_pred)
        elif metric_name == "f1":
            score = f1_score(y_val, y_pred, average="weighted")
        elif metric_name == "profit_factor":
            # Calculate profit factor from predictions
            # This requires backtesting - simplified here
            score = self._calculate_profit_factor(y_val, y_pred)
        else:
            score = balanced_accuracy_score(y_val, y_pred)

        return score

    def _calculate_profit_factor(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> float:
        """
        Calculate profit factor metric (simplified)

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Profit factor
        """
        # Simplified: assume correct predictions = profit, incorrect = loss
        correct = (y_true == y_pred).sum()
        incorrect = (y_true != y_pred).sum()

        if incorrect == 0:
            return float('inf')

        profit_factor = correct / max(incorrect, 1)
        return profit_factor

    def optimize_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict:
        """
        Optimize hyperparameters using Optuna

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Best hyperparameters
        """
        logger.info("Starting hyperparameter optimization")

        def objective_wrapper(trial):
            return self.objective(trial, X_train, y_train, X_val, y_val)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        study.optimize(
            objective_wrapper,
            n_trials=self.optuna_config["n_trials"],
            timeout=self.optuna_config["timeout"],
            n_jobs=self.optuna_config["n_jobs"],
            show_progress_bar=True,
        )

        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best value: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")

        return study.best_params

    def run_walk_forward_optimization(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target_column: str = "target",
        optimize_each_window: bool = True,
        save_plots_path: Optional[str] = None,
    ) -> List[Dict]:
        """
        Run complete walk-forward optimization

        Args:
            df: Complete dataset with features and target
            feature_columns: List of feature column names
            target_column: Target column name
            optimize_each_window: Whether to re-optimize in each window
            save_plots_path: Optional base path for saving training plots (fold number will be appended)

        Returns:
            List of results for each WF window
        """
        logger.info("Starting Walk-Forward Optimization")

        results = []
        best_params = None

        for fold_idx, (train_idx, test_idx) in enumerate(self.splitter.split(df)):
            logger.info(f"\n{'='*60}")
            logger.info(f"Walk-Forward Fold {fold_idx + 1}")
            logger.info(f"{'='*60}")

            # Extract data
            X_train = df.loc[train_idx, feature_columns].values
            y_train = df.loc[train_idx, target_column].values
            X_test = df.loc[test_idx, feature_columns].values
            y_test = df.loc[test_idx, target_column].values

            # Further split training data for validation (nested CV)
            val_size = int(len(X_train) * 0.2)
            X_train_opt, X_val = X_train[:-val_size], X_train[-val_size:]
            y_train_opt, y_val = y_train[:-val_size], y_train[-val_size:]

            # Hyperparameter optimization (if enabled)
            if optimize_each_window or best_params is None:
                best_params = self.optimize_hyperparameters(
                    X_train_opt, y_train_opt, X_val, y_val
                )

            # Train final model on full training data
            logger.info("Training model with optimized parameters")
            # Convert flat params dict to xgb_params structure
            xgb_params = {
                "objective": "multi:softprob",
                "num_class": 3,
                "learning_rate": best_params["learning_rate"],
                "max_depth": best_params["max_depth"],
                "n_estimators": best_params["n_estimators"],
                "subsample": best_params["subsample"],
                "colsample_bytree": best_params["colsample_bytree"],
                "min_child_weight": best_params["min_child_weight"],
                "gamma": best_params["gamma"],
                "reg_alpha": best_params["reg_alpha"],
                "reg_lambda": best_params["reg_lambda"],
                "random_state": self.splitter.min_train_samples,
            }
            model = self.model_class(xgb_params=xgb_params)

            # Generate plot path with fold number if base path provided
            fold_plot_path = None
            if save_plots_path is not None:
                fold_plot_path = f"{save_plots_path}_fold{fold_idx + 1}"

            # Pass feature names for inference server alignment (CRITICAL!)
            model.fit(X_train, y_train, save_plots_path=fold_plot_path,
                      feature_names=feature_columns)

            # Evaluate on OOS test set
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)

            # Calculate metrics
            ba_score = balanced_accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")

            logger.info(f"OOS Balanced Accuracy: {ba_score:.4f}")
            logger.info(f"OOS F1 Score: {f1:.4f}")

            # Store results
            fold_results = {
                "fold": fold_idx + 1,
                "train_start": df.loc[train_idx].index[0],
                "train_end": df.loc[train_idx].index[-1],
                "test_start": df.loc[test_idx].index[0],
                "test_end": df.loc[test_idx].index[-1],
                "train_samples": len(train_idx),
                "test_samples": len(test_idx),
                "best_params": best_params.copy(),
                "balanced_accuracy": ba_score,
                "f1_score": f1,
                "predictions": y_pred,
                "probabilities": y_pred_proba,
                "true_labels": y_test,
                "model": model,
            }

            results.append(fold_results)

        self.results = results
        logger.info(f"\nCompleted {len(results)} Walk-Forward folds")

        return results

    def aggregate_results(self) -> pd.DataFrame:
        """
        Aggregate results across all WF folds

        Returns:
            DataFrame with aggregated metrics
        """
        if not self.results:
            logger.warning("No results to aggregate")
            return pd.DataFrame()

        summary = []

        for result in self.results:
            summary.append({
                "fold": result["fold"],
                "test_start": result["test_start"],
                "test_end": result["test_end"],
                "test_samples": result["test_samples"],
                "balanced_accuracy": result["balanced_accuracy"],
                "f1_score": result["f1_score"],
            })

        summary_df = pd.DataFrame(summary)

        # Calculate overall statistics
        logger.info("\n" + "="*60)
        logger.info("Walk-Forward Optimization Summary")
        logger.info("="*60)
        logger.info(f"Total Folds: {len(summary_df)}")
        logger.info(
            f"Avg Balanced Accuracy: {summary_df['balanced_accuracy'].mean():.4f} ± {summary_df['balanced_accuracy'].std():.4f}")
        logger.info(
            f"Avg F1 Score: {summary_df['f1_score'].mean():.4f} ± {summary_df['f1_score'].std():.4f}")
        logger.info("="*60)

        return summary_df


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    import pandas as pd

    # Generate sample time-series data
    n_samples = 10000
    X, y = make_classification(
        n_samples=n_samples,
        n_features=50,
        n_informative=30,
        n_classes=3,
        random_state=42,
    )

    # Create DataFrame with temporal index
    dates = pd.date_range(start="2018-01-01", periods=n_samples, freq="4H")
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y
    df["timestamp"] = dates
    df.set_index("timestamp", inplace=True)

    # Test splitter
    splitter = WalkForwardSplitter(
        train_window_years=2,
        test_window_months=6,
        step_months=6,
    )

    for train_idx, test_idx in splitter.split(df):
        print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")
        print(f"Train period: {train_idx[0]} to {train_idx[-1]}")
        print(f"Test period: {test_idx[0]} to {test_idx[-1]}")
        print("-" * 60)
