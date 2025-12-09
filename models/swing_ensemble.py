import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    import joblib
except ImportError:
    joblib = None

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

class SwingTradingEnsemble:
    """
    Ensemble of Tree-based models (XGBoost + LightGBM) for Swing Trading (1-3 days).
    Uses weighted average of probabilities from constituent models.
    """
    def __init__(self, exchange: Optional[str] = None):
        self.models = {}
        self.weights = {}
        self._is_fitted = False
        self.feature_selector = None
        self.feature_columns = None
        self.exchange = exchange
        
        if XGBClassifier:
            self.models['xgboost'] = XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                objective='multi:softprob',
                n_jobs=-1
            )
            self.weights['xgboost'] = 0.5
            
        if LGBMClassifier:
            self.models['lightgbm'] = LGBMClassifier(
                n_estimators=500,
                num_leaves=31,
                learning_rate=0.05,
                objective='multiclass',
                n_jobs=-1
            )
            self.weights['lightgbm'] = 0.5
            
        if not self.models:
            logging.warning("No tree-based models available (XGBoost/LightGBM missing). Swing ensemble disabled.")
        
        # Load feature selector if exchange is provided
        if exchange and joblib:
            self._load_feature_selector(exchange)
    
    def _load_feature_selector(self, exchange: str):
        """Load feature selector and feature columns for this exchange."""
        model_dir = Path(f"models/{exchange}")
        selector_path = model_dir / "feature_selector.pkl"
        features_path = model_dir / "model_features.pkl"
        
        if selector_path.exists():
            try:
                self.feature_selector = joblib.load(selector_path)
                logging.debug(f"Loaded feature selector for {exchange}")
            except Exception as e:
                logging.warning(f"Failed to load feature selector: {e}")
        
        if features_path.exists():
            try:
                self.feature_columns = joblib.load(features_path)
                logging.debug(f"Loaded feature columns for {exchange}")
            except Exception as e:
                logging.warning(f"Failed to load feature columns: {e}")

    def fit(self, X, y):
        """Train all sub-models."""
        for name, model in self.models.items():
            logging.info(f"Training {name} for swing ensemble...")
            model.fit(X, y)
        self._is_fitted = True

    def predict_proba(self, X) -> np.ndarray:
        """
        Weighted average of prediction probabilities.
        Returns: (n_samples, n_classes) array
        """
        if not self._is_fitted:
            # Return uniform probabilities if not fitted
            return np.ones((X.shape[0], 3)) / 3.0
            
        final_probs = np.zeros((X.shape[0], 3))
        total_weight = 0.0
        
        for name, model in self.models.items():
            try:
                # Check if model is actually trained (has feature_importances_ or n_estimators > 0)
                if hasattr(model, 'feature_importances_') or (hasattr(model, 'n_estimators') and model.n_estimators > 0):
                    probs = model.predict_proba(X)
                    weight = self.weights[name]
                    final_probs += probs * weight
                    total_weight += weight
                else:
                    logging.debug(f"Skipping {name} - model not trained (no feature_importances_)")
            except Exception as e:
                logging.warning(f"Error predicting with {name}: {e}")
                
        if total_weight > 0:
            final_probs /= total_weight
        else:
            # If no models worked, return uniform probabilities
            logging.warning("No trained models available, returning uniform probabilities")
            return np.ones((X.shape[0], 3)) / 3.0
            
        return final_probs

    def predict(self, features: Any) -> Dict[str, Any]:
        """
        Single sample inference.
        Args:
            features: Array-like (1, n_features), DataFrame, or dict of feature names
        """
        # Handle different input types
        if isinstance(features, dict):
            # Convert dict to array using feature_columns order
            if self.feature_columns:
                X = np.array([[features.get(col, 0.0) for col in self.feature_columns]])
            else:
                # Fallback: use dict values in order
                X = np.array([list(features.values())])
        elif hasattr(features, 'values'):
            # DataFrame
            if self.feature_columns:
                # Select only the columns used during training
                missing = [col for col in self.feature_columns if col not in features.columns]
                if missing:
                    for col in missing:
                        features[col] = 0.0
                X = features[self.feature_columns].values
            else:
                X = features.values
        else:
            # Array-like
            X = np.array(features)
            
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Apply feature selector if available
        if self.feature_selector is not None and hasattr(self.feature_selector, 'transform'):
            try:
                X = self.feature_selector.transform(X)
            except Exception as e:
                logging.warning(f"Feature selection failed: {e}, using all features")
             
        probs = self.predict_proba(X)[0]
        max_prob = float(max(probs))
        predicted_class = int(probs.argmax())
        
        # DEBUG: Log if we detect the bug pattern
        if predicted_class == 1:  # HOLD
            hold_prob = float(probs[1])
            if abs(max_prob - (1 - hold_prob)) < 0.0001 and hold_prob > 0.5:
                logging.error(f"[SwingEnsemble] BUG DETECTED: max_prob={max_prob:.6f} equals 1-HOLD_prob, probs={probs.tolist()}")
        
        return {
            'class': predicted_class,
            'probabilities': probs.tolist()
        }

