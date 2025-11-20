"""
Compare old vs new models and decide if deployment is warranted
"""
import argparse
import json
import pickle
from pathlib import Path
import pandas as pd


def load_model_metrics(models_dir: Path, pair: str) -> dict:
    """Load model performance metrics"""
    try:
        # Load WFO summary
        wfo_path = models_dir / "results" / f"{pair}_wfo_summary.csv"
        if not wfo_path.exists():
            return None
        
        df = pd.read_csv(wfo_path)
        
        # Calculate key metrics
        metrics = {
            "accuracy": df["accuracy"].mean(),
            "precision": df["precision"].mean(),
            "recall": df["recall"].mean(),
            "f1": df["f1"].mean(),
            "sharpe": df.get("sharpe_ratio", pd.Series([0])).mean(),
            "total_return": df.get("total_return", pd.Series([0])).mean(),
        }
        
        return metrics
    except Exception as e:
        print(f"Error loading metrics for {pair}: {e}")
        return None


def compare_pair_models(old_dir: Path, new_dir: Path, pair: str) -> dict:
    """Compare old vs new model for a single pair"""
    old_metrics = load_model_metrics(old_dir, pair)
    new_metrics = load_model_metrics(new_dir, pair)
    
    if not old_metrics or not new_metrics:
        return {
            "pair": pair,
            "improved": False,
            "reason": "Missing metrics",
            "old_metrics": old_metrics,
            "new_metrics": new_metrics
        }
    
    # Calculate improvement score (weighted)
    weights = {
        "accuracy": 0.2,
        "f1": 0.3,
        "sharpe": 0.3,
        "total_return": 0.2
    }
    
    improvement_score = 0
    for metric, weight in weights.items():
        old_val = old_metrics.get(metric, 0)
        new_val = new_metrics.get(metric, 0)
        
        if old_val > 0:
            pct_change = (new_val - old_val) / old_val
            improvement_score += pct_change * weight
    
    # Require at least 2% improvement to deploy
    improved = improvement_score > 0.02
    
    return {
        "pair": pair,
        "improved": improved,
        "improvement_score": improvement_score,
        "old_metrics": old_metrics,
        "new_metrics": new_metrics,
        "reason": f"{improvement_score*100:.2f}% improvement" if improved else "Insufficient improvement"
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=str, required=True, help="Old models directory")
    parser.add_argument("--new", type=str, required=True, help="New models directory")
    parser.add_argument("--output", type=str, default="comparison.json", help="Output JSON file")
    args = parser.parse_args()
    
    old_dir = Path(args.old)
    new_dir = Path(args.new)
    
    pairs = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"]
    
    results = []
    for pair in pairs:
        result = compare_pair_models(old_dir, new_dir, pair)
        results.append(result)
        print(f"\n{pair}:")
        print(f"  Improved: {result['improved']}")
        print(f"  Reason: {result['reason']}")
    
    # Deploy if at least 2 out of 4 models improved
    improved_count = sum(1 for r in results if r["improved"])
    should_deploy = improved_count >= 2
    
    output = {
        "should_deploy": should_deploy,
        "improved_count": improved_count,
        "total_pairs": len(pairs),
        "results": results,
        "summary": f"{improved_count}/{len(pairs)} models improved"
    }
    
    # Save to JSON
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"DECISION: {'DEPLOY ✅' if should_deploy else 'SKIP ⏭️'}")
    print(f"Reason: {output['summary']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
