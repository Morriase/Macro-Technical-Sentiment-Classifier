"""
Extract and save feature metadata for existing trained models
This avoids retraining - we regenerate feature names from the training pipeline
"""
from src.feature_engineering.technical_features import TechnicalFeatureEngineer
from src.data_acquisition.kaggle_loader import KaggleFXDataLoader
from src.config import CURRENCY_PAIRS, MODELS_DIR, DATA_DIR, IS_KAGGLE
import sys
import json
import joblib
from pathlib import Path
from loguru import logger
from datetime import datetime

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


logger.info("=" * 80)
logger.info("EXTRACTING FEATURE METADATA FROM EXISTING MODELS")
logger.info("=" * 80)


def extract_feature_columns_from_data(pair: str) -> list:
    """
    Recreate the exact feature columns used during training
    by processing the same data through the feature engineering pipeline
    """
    logger.info(f"\nProcessing {pair}...")

    try:
        # Load data (same as training)
        if IS_KAGGLE:
            loader = KaggleFXDataLoader()
            df_price, df_events = loader.load_currency_pair(pair)
        else:
            # Load from Kaggle dataset directory
            fx_file = DATA_DIR / "kaggle_dataset" / \
                "fx_data" / f"{pair}_M5.parquet"
            if not fx_file.exists():
                raise FileNotFoundError(f"FX data not found: {fx_file}")

            import pandas as pd
            df_price = pd.read_parquet(fx_file)
            logger.info(f"Loaded {len(df_price)} price candles for {pair}")

        # Engineer features (same pipeline as training)
        tech_engineer = TechnicalFeatureEngineer()
        df_features = tech_engineer.calculate_all_features(df_price.copy())
        df_features = tech_engineer.calculate_feature_crosses(df_features)

        # Drop NaN
        df_features.dropna(inplace=True)

        # Extract feature columns (SAME LOGIC AS main.py train_model)
        exclude_cols = [
            "open", "high", "low", "close", "volume",
            "forward_close", "forward_return", "forward_return_pips",
            "target", "target_class"
        ]

        feature_cols = [
            col for col in df_features.columns
            if col not in exclude_cols
        ]

        logger.success(f"{pair}: Extracted {len(feature_cols)} features")
        return feature_cols

    except Exception as e:
        logger.error(f"Failed to extract features for {pair}: {e}")
        return None


def update_model_config_with_features(pair: str, feature_names: list):
    """
    Load existing model config, add feature metadata, and re-save
    """
    config_path = MODELS_DIR / f"{pair}_model.pth_config.pkl"

    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        return False

    # Load existing config
    config = joblib.load(config_path)

    # Add feature metadata
    config["n_features_"] = len(feature_names)
    config["feature_names_"] = feature_names

    # Re-save with metadata
    joblib.dump(config, config_path)
    logger.success(f"Updated {config_path} with feature metadata")

    return True


def save_feature_schema_json(pair: str, feature_names: list):
    """
    Save feature schema as JSON for inference server
    """
    schema = {
        "currency_pair": pair,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "feature_order": {name: idx for idx, name in enumerate(feature_names)},
        "model_version": "1.0",
        "extracted_date": datetime.now().isoformat(),
        "note": "Extracted from training pipeline (not retrained)",
    }

    schema_path = MODELS_DIR / f"{pair}_feature_schema.json"
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)

    logger.success(f"Saved feature schema: {schema_path}")
    return True


def main():
    """Extract metadata for all trained models"""

    # Find all trained models
    trained_pairs = []
    for pair in CURRENCY_PAIRS:
        config_file = MODELS_DIR / f"{pair}_model.pth_config.pkl"
        if config_file.exists():
            trained_pairs.append(pair)

    if not trained_pairs:
        logger.error("No trained models found!")
        return

    logger.info(f"Found trained models for: {', '.join(trained_pairs)}")

    # Process each pair
    success_count = 0
    for pair in trained_pairs:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing {pair}")
        logger.info(f"{'=' * 60}")

        # Extract feature names from data pipeline
        feature_names = extract_feature_columns_from_data(pair)

        if feature_names is None:
            logger.warning(f"Skipping {pair} - feature extraction failed")
            continue

        # Verify feature count matches model
        config_path = MODELS_DIR / f"{pair}_model.pth_config.pkl"
        config = joblib.load(config_path)

        # Check XGBoost base model feature count
        xgb_path = MODELS_DIR / f"{pair}_model.pth_xgb_base.pkl"
        xgb_model = joblib.load(xgb_path)
        expected_features = xgb_model.n_features_in_

        if len(feature_names) != expected_features:
            logger.error(
                f"MISMATCH! Extracted {len(feature_names)} features, "
                f"but model expects {expected_features} features"
            )
            logger.error("Model needs retraining or data mismatch!")
            continue

        logger.success(
            f"✓ Feature count matches: {len(feature_names)} features")

        # Update model config
        if update_model_config_with_features(pair, feature_names):
            # Save JSON schema for inference server
            save_feature_schema_json(pair, feature_names)
            success_count += 1

    # Summary
    logger.info("\n" + "=" * 80)
    if success_count == len(trained_pairs):
        logger.success(
            f"✓ ALL {success_count} models updated with feature metadata!")
        logger.info("\nNext steps:")
        logger.info(
            "1. Models now have feature_names_ and n_features_ in config")
        logger.info("2. JSON schemas created for inference server validation")
        logger.info("3. You can now build the inference server")
    else:
        logger.warning(
            f"⚠ Only {success_count}/{len(trained_pairs)} models updated successfully"
        )
        logger.info("Check errors above - you may need to retrain some pairs")


if __name__ == "__main__":
    main()
