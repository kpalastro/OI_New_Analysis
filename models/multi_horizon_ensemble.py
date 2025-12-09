import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import torch
except ImportError:
    torch = None

try:
    import joblib
except ImportError:
    joblib = None

from models.intraday_lstm import IntradayLSTMModel
from models.swing_ensemble import SwingTradingEnsemble
from models.expiry_transformer import ExpiryDayTransformer
from models.horizon_router import HorizonRouter

LOGGER = logging.getLogger(__name__)


class MultiHorizonEnsemble:
    """
    Unified interface for multi-horizon predictions.
    Delegates to specialized models based on the HorizonRouter.
    """
    def __init__(self, exchange: str):
        self.exchange = exchange
        self.router = HorizonRouter()
        self.weights_loaded = False
        
        # Initialize Sub-Models
        try:
            self.intraday_model = IntradayLSTMModel()
        except ImportError:
            LOGGER.warning("IntradayLSTMModel unavailable (missing torch).")
            self.intraday_model = None
            
        self.swing_model = SwingTradingEnsemble(exchange=exchange)
        
        try:
            self.expiry_model = ExpiryDayTransformer()
        except ImportError:
            LOGGER.warning("ExpiryDayTransformer unavailable (missing torch).")
            self.expiry_model = None
        
        self._load_weights()

    def _load_weights(self):
        """
        Load pre-trained weights for all models from disk.
        
        Expected file structure:
            models/{exchange}/intraday_lstm.pt      - PyTorch LSTM weights
            models/{exchange}/expiry_transformer.pt - PyTorch Transformer weights
            models/{exchange}/swing_ensemble.pkl    - Joblib serialized ensemble
        """
        model_dir = Path(f"models/{self.exchange}")
        
        if not model_dir.exists():
            LOGGER.info(f"Model directory {model_dir} not found. Using untrained models.")
            return
        
        loaded_count = 0
        
        # 1. Load LSTM weights (PyTorch)
        lstm_path = model_dir / "intraday_lstm.pt"
        if lstm_path.exists() and self.intraday_model is not None and torch is not None:
            try:
                state_dict = torch.load(lstm_path, map_location='cpu')
                self.intraday_model.load_state_dict(state_dict)
                self.intraday_model.eval()  # Set to evaluation mode
                LOGGER.info(f"✓ Loaded LSTM weights from {lstm_path}")
                loaded_count += 1
            except Exception as e:
                LOGGER.warning(f"Failed to load LSTM weights: {e}")
        
        # 2. Load Transformer weights (PyTorch)
        transformer_path = model_dir / "expiry_transformer.pt"
        if transformer_path.exists() and self.expiry_model is not None and torch is not None:
            try:
                state_dict = torch.load(transformer_path, map_location='cpu')
                self.expiry_model.load_state_dict(state_dict)
                self.expiry_model.eval()  # Set to evaluation mode
                LOGGER.info(f"✓ Loaded Transformer weights from {transformer_path}")
                loaded_count += 1
            except Exception as e:
                LOGGER.warning(f"Failed to load Transformer weights: {e}")
        
        # 3. Load Swing Ensemble (Joblib - XGBoost/LightGBM)
        swing_path = model_dir / "swing_ensemble.pkl"
        if swing_path.exists() and self.swing_model is not None and joblib is not None:
            try:
                saved_data = joblib.load(swing_path)
                
                # Handle different save formats
                if isinstance(saved_data, dict):
                    # Format: {'models': {...}, 'weights': {...}, '_is_fitted': True}
                    if 'models' in saved_data:
                        self.swing_model.models = saved_data['models']
                    if 'weights' in saved_data:
                        self.swing_model.weights = saved_data['weights']
                    self.swing_model._is_fitted = saved_data.get('_is_fitted', True)
                elif isinstance(saved_data, SwingTradingEnsemble):
                    # Direct object serialization
                    self.swing_model = saved_data
                    # Ensure exchange is set for feature selector loading
                    if not hasattr(self.swing_model, 'exchange') or not self.swing_model.exchange:
                        self.swing_model.exchange = self.exchange
                
                # Initialize feature_selector and feature_columns if they don't exist
                # (they may not have been saved with the object)
                if not hasattr(self.swing_model, 'feature_selector'):
                    self.swing_model.feature_selector = None
                if not hasattr(self.swing_model, 'feature_columns'):
                    self.swing_model.feature_columns = None
                
                # Reload feature selector and feature columns (they may not be in pickle)
                # This ensures the selector is available even if it wasn't saved with the object
                if hasattr(self.swing_model, '_load_feature_selector') and self.swing_model.exchange:
                    self.swing_model._load_feature_selector(self.swing_model.exchange)
                    
                LOGGER.info(f"✓ Loaded Swing Ensemble from {swing_path}")
                loaded_count += 1
            except Exception as e:
                LOGGER.warning(f"Failed to load Swing Ensemble: {e}")
        
        self.weights_loaded = loaded_count > 0
        
        if loaded_count > 0:
            LOGGER.info(f"Model loading complete: {loaded_count} model(s) loaded for {self.exchange}")
        else:
            LOGGER.info(f"No pre-trained weights found for {self.exchange}. Models will use default initialization.")

    def predict(self, features: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main prediction entry point.
        
        Args:
            features: Feature vector (DataFrame/Array) for Swing, or Sequence for LSTM.
                      Ideally, this is a tuple (feature_vector, feature_sequence) or similar.
                      For compatibility, we assume ml_core passes a dict with keys 'vector' and 'sequence'.
            context: Additional context (strategy mode, time, time_to_expiry_hours).
            
        Returns:
            Dict containing signal, probabilities, and used horizon.
        """
        # Unwrap input if it's our special struct
        feature_vector = features
        feature_sequence = None
        
        if isinstance(features, dict) and 'vector' in features:
            feature_vector = features['vector']
            feature_sequence = features.get('sequence')
            
        # Determine horizon
        # We need a dict for determination, if feature_vector is DF, convert first row to dict
        feature_dict = {}
        if hasattr(feature_vector, 'iloc'):
            feature_dict = feature_vector.iloc[0].to_dict()
        elif context:
            feature_dict = context # Fallback
            
        horizon = self.router.determine_horizon(feature_dict, context)
        
        result = {
            'horizon': horizon,
            'signal': 'HOLD',
            'confidence': 0.0,
            'probabilities': [0.0, 1.0, 0.0] # [SELL, HOLD, BUY] default
        }
        
        try:
            # 1. Intraday (LSTM)
            if horizon == 'intraday':
                if self.intraday_model and feature_sequence is not None:
                    # Expect sequence to be (seq_len, features)
                    # Convert to tensor if needed inside predict_single
                    pred = self.intraday_model.predict_single(feature_sequence)
                    result.update({
                        'signal': ['SELL', 'HOLD', 'BUY'][pred['class']], # Assuming 0=Sell, 1=Hold, 2=Buy? Check mapping
                        'probabilities': pred['probabilities'],
                        'confidence': max(pred['probabilities'])
                    })
                    return result
                else:
                    # Fallback to swing if model missing or sequence missing
                    logging.debug("Intraday fallback to swing (model or sequence missing)")
                    horizon = 'swing'
                    result['horizon'] = 'swing'
            
            # 2. Expiry (Transformer)
            if horizon == 'expiry':
                if self.expiry_model:
                    # Convert to array if DF
                    if hasattr(feature_vector, 'values'):
                        fv = feature_vector.values
                    else:
                        fv = np.array(feature_vector)
                        
                    if fv.ndim == 2:
                        fv = fv[0] # Take first row if batch
                        
                    pred = self.expiry_model.predict(fv)
                    result.update({
                        'signal': ['SELL', 'HOLD', 'BUY'][pred['direction']],
                        'probabilities': pred['probabilities'],
                        'confidence': max(pred['probabilities'])
                    })
                else:
                    horizon = 'swing'
                    result['horizon'] = 'swing'
                    
            # 3. Swing (Ensemble)
            if horizon == 'swing':
                if self.swing_model:
                    pred = self.swing_model.predict(feature_vector)
                    # Mapping: 0=SELL, 1=HOLD, 2=BUY is standard for our pipeline?
                    # Check MLSignalGenerator logic: {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}
                    # Usually classifiers output 0, 1, 2 indices. 
                    # Let's map 0->SELL, 1->HOLD, 2->BUY for now, need verification with train script.
                    classes = ['SELL', 'HOLD', 'BUY']
                    probs = pred['probabilities']
                    predicted_class = pred['class']
                    signal = classes[predicted_class]
                    
                    # CRITICAL FIX: Confidence MUST ALWAYS be max(probabilities)
                    # This is the fundamental definition of confidence - the maximum probability
                    # Never use 1 - probability or any other calculation
                    confidence = float(max(probs))
                    
                    # Additional safety: For HOLD signals, ensure confidence equals HOLD probability
                    # (which should be max(probs) if HOLD is the predicted class)
                    if predicted_class == 1:  # HOLD signal
                        hold_prob = float(probs[1])
                        # If confidence doesn't match HOLD probability, something is wrong
                        if abs(confidence - hold_prob) > 0.0001:
                            logging.warning(f"[{self.exchange}] HOLD signal: confidence={confidence:.6f} != HOLD_prob={hold_prob:.6f}, fixing")
                            confidence = hold_prob
                        # Also check if confidence was calculated as 1 - HOLD_prob (common bug)
                        wrong_conf = 1.0 - hold_prob
                        if abs(confidence - wrong_conf) < 0.0001:
                            logging.error(f"[{self.exchange}] BUG DETECTED: confidence={confidence:.6f} equals 1-HOLD_prob, correcting to {hold_prob:.6f}")
                            confidence = hold_prob
                    
                    # FINAL SAFETY: Before updating result, ensure confidence is correct
                    # This is the last chance to fix it before returning
                    final_max_prob = float(max(probs))
                    if abs(confidence - final_max_prob) > 0.0001:
                        # Confidence is wrong - force correct value
                        if predicted_class == 1:  # HOLD
                            hold_prob = float(probs[1])
                            wrong_conf = 1.0 - hold_prob
                            if abs(confidence - wrong_conf) < 0.0001:
                                logging.error(f"[{self.exchange}] 🐛 FINAL FIX: confidence={confidence:.6f} was 1-HOLD_prob, correcting to {final_max_prob:.6f}")
                            else:
                                logging.error(f"[{self.exchange}] FINAL FIX: confidence={confidence:.6f} != max_prob {final_max_prob:.6f}, correcting")
                        else:
                            logging.error(f"[{self.exchange}] FINAL FIX: confidence={confidence:.6f} != max_prob {final_max_prob:.6f}, correcting")
                        confidence = final_max_prob
                    
                    result.update({
                        'signal': signal,
                        'probabilities': probs,
                        'confidence': confidence
                    })
                
        except Exception as e:
            logging.error(f"Error in MultiHorizonEnsemble predict ({horizon}): {e}", exc_info=True)
            # On error, return default HOLD signal
            result = {
                'horizon': horizon,
                'signal': 'HOLD',
                'confidence': 0.0,
                'probabilities': [0.0, 1.0, 0.0]
            }
            
        # FINAL ABSOLUTE FIX: Before returning, ensure confidence ALWAYS equals max(probabilities)
        # This is the last line of defense against the confidence bug
        probs_final = result.get('probabilities', [])
        if probs_final and len(probs_final) >= 3:
            max_prob_final = float(max(probs_final))
            current_conf = float(result.get('confidence', 0.0))
            
            # If confidence doesn't match max_prob, fix it
            if abs(current_conf - max_prob_final) > 0.0001:
                # Check if it's the 1 - HOLD_prob bug
                if result.get('signal') == 'HOLD' and len(probs_final) > 1:
                    hold_prob_final = float(probs_final[1])
                    wrong_conf_final = 1.0 - hold_prob_final
                    if abs(current_conf - wrong_conf_final) < 0.0001:
                        logging.error(f"[{self.exchange}] 🐛 ABSOLUTE FIX: confidence={current_conf:.6f} was 1-HOLD_prob={wrong_conf_final:.6f}, correcting to {max_prob_final:.6f}")
                    else:
                        logging.error(f"[{self.exchange}] ABSOLUTE FIX: confidence={current_conf:.6f} != max_prob {max_prob_final:.6f}, correcting")
                else:
                    logging.error(f"[{self.exchange}] ABSOLUTE FIX: confidence={current_conf:.6f} != max_prob {max_prob_final:.6f}, correcting")
                
                result['confidence'] = max_prob_final
        
        return result

