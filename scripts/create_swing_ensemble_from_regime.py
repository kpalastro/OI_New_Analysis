#!/usr/bin/env python3
"""
Create a swing_ensemble.pkl from existing regime_models.pkl.

This bridges the gap between the old regime-specific model format
and the new SwingTradingEnsemble format expected by MultiHorizonEnsemble.
"""
import sys
from pathlib import Path
import joblib
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.swing_ensemble import SwingTradingEnsemble

logging.basicConfig(level=logging.INFO)

def create_swing_ensemble(exchange: str):
    """Create swing_ensemble.pkl from regime_models.pkl."""
    model_dir = Path(f"models/{exchange}")
    regime_models_path = model_dir / "regime_models.pkl"
    swing_ensemble_path = model_dir / "swing_ensemble.pkl"
    
    if not regime_models_path.exists():
        logging.error(f"regime_models.pkl not found at {regime_models_path}")
        return False
    
    # Load regime models
    regime_models = joblib.load(regime_models_path)
    logging.info(f"Loaded regime models: {list(regime_models.keys())}")
    
    # Create SwingTradingEnsemble with exchange to load feature selector
    swing_ensemble = SwingTradingEnsemble(exchange=exchange)
    
    # If we have regime models, we can use the first one (or combine them)
    # For now, we'll use the most common regime model (regime 0 or 1)
    # In production, you'd want to use regime-aware prediction
    
    if len(regime_models) > 0:
        # Use the first available model as a fallback
        # In a proper implementation, you'd want to use regime-aware selection
        first_regime = list(regime_models.keys())[0]
        model = regime_models[first_regime]
        
        # Check if it's XGBoost or LightGBM
        model_type = type(model).__name__.lower()
        
        if 'lgbm' in model_type or 'lightgbm' in model_type:
            if swing_ensemble.models.get('lightgbm'):
                swing_ensemble.models['lightgbm'] = model
                swing_ensemble._is_fitted = True
                logging.info(f"Using LightGBM model from regime {first_regime}")
        elif 'xgb' in model_type or 'xgboost' in model_type:
            if swing_ensemble.models.get('xgboost'):
                swing_ensemble.models['xgboost'] = model
                swing_ensemble._is_fitted = True
                logging.info(f"Using XGBoost model from regime {first_regime}")
        
        # If we have multiple regime models, we could combine them
        # For now, we'll just use one as a simple solution
        if len(regime_models) > 1:
            logging.warning(f"Multiple regime models found ({len(regime_models)}). Using regime {first_regime} only.")
            logging.warning("For proper regime-aware prediction, you should train a unified swing_ensemble.pkl")
    
    # Save the ensemble
    swing_ensemble_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(swing_ensemble, swing_ensemble_path)
    logging.info(f"✓ Saved swing_ensemble.pkl to {swing_ensemble_path}")
    
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create swing_ensemble.pkl from regime_models.pkl")
    parser.add_argument("--exchange", default="NSE", choices=["NSE", "BSE"])
    args = parser.parse_args()
    
    success = create_swing_ensemble(args.exchange)
    sys.exit(0 if success else 1)

