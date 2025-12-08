#!/usr/bin/env python3
"""
Feature Importance Analyzer for OI Gemini Models

Analyzes feature importance from trained models and displays the most relevant features.
Works with regime-specific models (LightGBM/XGBoost) and feature selectors.

Prerequisites:
    pip install lightgbm xgboost joblib numpy

Usage:
    python scripts/analyze_feature_importance.py [NSE|BSE] [--top N] [--regime R]
    
Examples:
    python scripts/analyze_feature_importance.py NSE
    python scripts/analyze_feature_importance.py NSE --top 30
    python scripts/analyze_feature_importance.py BSE --regime 0
    python scripts/analyze_feature_importance.py NSE --all-regimes
    python scripts/analyze_feature_importance.py NSE --export nse_features.json
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any

try:
    import joblib
except ImportError:
    joblib = None
    print("WARNING: joblib not installed. Some model formats may not load.")

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import numpy as np
except ImportError:
    np = None
    print("ERROR: numpy is required. Please install it: pip install numpy")
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)


def load_model_files(exchange: str) -> Tuple[List[str], Dict, Any]:
    """Load model files from disk."""
    model_dir = Path(f"models/{exchange}")
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Load feature names
    feature_path = model_dir / "model_features.pkl"
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")
    
    feature_names = pickle.load(open(feature_path, "rb"))
    if not isinstance(feature_names, list):
        raise ValueError(f"Expected list of feature names, got {type(feature_names)}")
    
    LOGGER.info(f"✓ Loaded {len(feature_names)} feature names")
    
    # Load regime models
    regime_path = model_dir / "regime_models.pkl"
    if not regime_path.exists():
        LOGGER.warning(f"Regime models file not found: {regime_path}")
        regime_models = {}
    else:
        try:
            if joblib:
                regime_models = joblib.load(regime_path)
            else:
                regime_models = pickle.load(open(regime_path, "rb"))
            LOGGER.info(f"✓ Loaded regime models: {list(regime_models.keys())}")
        except ImportError as e:
            LOGGER.error(f"Missing dependency to load models: {e}")
            LOGGER.info("Please install required packages: pip install lightgbm xgboost")
            raise
        except Exception as e:
            LOGGER.warning(f"Could not load regime models: {e}")
            regime_models = {}
    
    # Load feature selector (optional)
    selector_path = model_dir / "feature_selector.pkl"
    feature_selector = None
    if selector_path.exists():
        try:
            if joblib:
                feature_selector = joblib.load(selector_path)
            else:
                feature_selector = pickle.load(open(selector_path, "rb"))
            LOGGER.info("✓ Loaded feature selector")
        except Exception as e:
            LOGGER.warning(f"Could not load feature selector: {e}")
    
    return feature_names, regime_models, feature_selector


def extract_feature_importance(model: Any, feature_names: List[str]) -> List[Tuple[str, float]]:
    """Extract feature importance from a model."""
    importance_scores = []
    
    # LightGBM models
    if hasattr(model, 'feature_importances_'):
        scores = model.feature_importances_
        if len(scores) == len(feature_names):
            importance_scores = list(zip(feature_names, scores))
        elif hasattr(model, 'feature_name_'):
            # Model was trained on selected features
            # feature_name_ is a property (list), not a method
            selected_features = model.feature_name_ if isinstance(model.feature_name_, list) else list(model.feature_name_)
            scores_dict = dict(zip(selected_features, scores))
            # Map to full feature list
            importance_scores = [(f, scores_dict.get(f, 0.0)) for f in feature_names]
        else:
            LOGGER.warning(f"Feature count mismatch: model has {len(scores)}, expected {len(feature_names)}")
            return []
    
    # XGBoost Booster
    elif hasattr(model, 'get_score'):
        scores_dict = model.get_score(importance_type='gain')
        # XGBoost uses f0, f1, f2... as keys
        if scores_dict and any(k.startswith('f') for k in scores_dict.keys()):
            # Convert f0, f1, ... to feature names
            for i, feat_name in enumerate(feature_names):
                score = scores_dict.get(f'f{i}', 0.0)
                importance_scores.append((feat_name, score))
        else:
            # Direct feature name mapping
            importance_scores = [(f, scores_dict.get(f, 0.0)) for f in feature_names]
    
    # XGBoost sklearn API
    elif hasattr(model, 'feature_importances_'):
        scores = model.feature_importances_
        if len(scores) == len(feature_names):
            importance_scores = list(zip(feature_names, scores))
    
    # Feature selector (SelectFromModel)
    elif hasattr(model, 'get_support'):
        # This is a feature selector, not a model
        support = model.get_support()
        selected_features = [f for f, s in zip(feature_names, support) if s]
        LOGGER.info(f"Feature selector selected {len(selected_features)} features")
        # Return selected features with importance 1.0
        importance_scores = [(f, 1.0) for f in selected_features]
    
    else:
        LOGGER.warning(f"Model type {type(model)} does not support feature importance extraction")
        return []
    
    # Sort by importance (descending)
    importance_scores.sort(key=lambda x: x[1], reverse=True)
    return importance_scores


def aggregate_importance(regime_models: Dict, feature_names: List[str]) -> List[Tuple[str, float]]:
    """Aggregate feature importance across all regimes."""
    aggregated = {f: 0.0 for f in feature_names}
    regime_counts = {f: 0 for f in feature_names}
    
    for regime_id, model in regime_models.items():
        importance = extract_feature_importance(model, feature_names)
        for feat_name, score in importance:
            aggregated[feat_name] += score
            regime_counts[feat_name] += 1
    
    # Average across regimes
    for feat_name in aggregated:
        if regime_counts[feat_name] > 0:
            aggregated[feat_name] /= regime_counts[feat_name]
    
    # Sort by aggregated importance
    sorted_features = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)
    return sorted_features


def print_feature_importance(
    features: List[Tuple[str, float]], 
    title: str, 
    top_n: int = 30,
    show_percent: bool = True
):
    """Print feature importance in a formatted table."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'Feature Name':<50} {'Importance':<15}")
    print(f"{'-'*80}")
    
    total_importance = sum(score for _, score in features)
    
    for rank, (feat_name, score) in enumerate(features[:top_n], 1):
        if show_percent and total_importance > 0:
            pct = (score / total_importance) * 100
            print(f"{rank:<6} {feat_name:<50} {score:<15.6f} ({pct:.2f}%)")
        else:
            print(f"{rank:<6} {feat_name:<50} {score:<15.6f}")
    
    if len(features) > top_n:
        print(f"\n... and {len(features) - top_n} more features")
    
    print(f"{'='*80}\n")


def analyze_feature_selector(selector: Any, feature_names: List[str], top_n: int = 30):
    """Analyze feature selector to see which features were selected."""
    if selector is None:
        return
    
    print(f"\n{'='*80}")
    print("  Feature Selector Analysis")
    print(f"{'='*80}")
    
    if hasattr(selector, 'get_support'):
        support = selector.get_support()
        selected = [f for f, s in zip(feature_names, support) if s]
        print(f"Selected {len(selected)} out of {len(feature_names)} features ({len(selected)/len(feature_names)*100:.1f}%)")
        print(f"\nTop {min(top_n, len(selected))} Selected Features:")
        for i, feat in enumerate(selected[:top_n], 1):
            print(f"  {i}. {feat}")
    elif hasattr(selector, 'feature_importances_'):
        # Selector is a model itself
        importance = extract_feature_importance(selector, feature_names)
        print_feature_importance(importance, "Feature Selector Importance", top_n)
    else:
        print("Feature selector format not recognized")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze feature importance from trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/analyze_feature_importance.py NSE
  python scripts/analyze_feature_importance.py NSE --top 50
  python scripts/analyze_feature_importance.py BSE --regime 0
  python scripts/analyze_feature_importance.py NSE --export nse_features.json
        """
    )
    parser.add_argument('exchange', choices=['NSE', 'BSE'], help='Exchange to analyze')
    parser.add_argument('--top', type=int, default=30, help='Number of top features to display (default: 30)')
    parser.add_argument('--regime', type=int, help='Show importance for specific regime only')
    parser.add_argument('--export', type=str, help='Export results to JSON file')
    parser.add_argument('--all-regimes', action='store_true', help='Show importance for all regimes separately')
    
    args = parser.parse_args()
    
    try:
        # Load model files
        feature_names, regime_models, feature_selector = load_model_files(args.exchange)
        
        results = {
            'exchange': args.exchange,
            'total_features': len(feature_names),
            'regimes': list(regime_models.keys()),
            'feature_importance': {}
        }
        
        # Analyze feature selector
        if feature_selector is not None:
            analyze_feature_selector(feature_selector, feature_names, args.top)
        
        # Analyze per-regime or aggregate
        if args.regime is not None:
            # Specific regime
            if args.regime not in regime_models:
                print(f"ERROR: Regime {args.regime} not found. Available regimes: {list(regime_models.keys())}")
                return
            
            model = regime_models[args.regime]
            importance = extract_feature_importance(model, feature_names)
            print_feature_importance(importance, f"Regime {args.regime} Feature Importance", args.top)
            results['feature_importance'][f'regime_{args.regime}'] = {
                feat: float(score) for feat, score in importance
            }
        
        elif args.all_regimes:
            # All regimes separately
            for regime_id, model in regime_models.items():
                importance = extract_feature_importance(model, feature_names)
                print_feature_importance(importance, f"Regime {regime_id} Feature Importance", args.top)
                results['feature_importance'][f'regime_{regime_id}'] = {
                    feat: float(score) for feat, score in importance
                }
        
        else:
            # Aggregate across all regimes
            aggregated = aggregate_importance(regime_models, feature_names)
            print_feature_importance(aggregated, f"{args.exchange} - Overall Feature Importance (Aggregated)", args.top)
            results['feature_importance']['aggregated'] = {
                feat: float(score) for feat, score in aggregated
            }
            
            # Also show top features per regime if multiple regimes exist
            if len(regime_models) > 1:
                print(f"\n{'='*80}")
                print("  Top Features by Regime (Top 10)")
                print(f"{'='*80}")
                for regime_id, model in regime_models.items():
                    importance = extract_feature_importance(model, feature_names)
                    top_10 = importance[:10]
                    print(f"\nRegime {regime_id} - Top 10:")
                    for rank, (feat, score) in enumerate(top_10, 1):
                        print(f"  {rank}. {feat:<50} {score:.6f}")
        
        # Export to JSON if requested
        if args.export:
            with open(args.export, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ Results exported to {args.export}")
        
        # Summary statistics
        print(f"\n{'='*80}")
        print("  Summary")
        print(f"{'='*80}")
        print(f"Exchange: {args.exchange}")
        print(f"Total Features: {len(feature_names)}")
        print(f"Regimes: {len(regime_models)} ({list(regime_models.keys())})")
        if feature_selector and hasattr(feature_selector, 'get_support'):
            selected_count = feature_selector.get_support().sum()
            print(f"Selected Features: {selected_count} ({selected_count/len(feature_names)*100:.1f}%)")
        print(f"{'='*80}\n")
        
    except FileNotFoundError as e:
        LOGGER.error(f"Model files not found: {e}")
        LOGGER.info("Make sure you have trained models in models/{exchange}/ directory")
        return 1
    except Exception as e:
        LOGGER.error(f"Error analyzing features: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

