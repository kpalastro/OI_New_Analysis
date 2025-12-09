"""
ml_core.py

Runtime ML signal engine for OI Gemini.
Integrates Multi-Horizon Models (Phase 2) and Enhanced Regime Detection (Phase 5).
"""
from __future__ import annotations

import json
import logging
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import database_new as db
from risk_manager import get_optimal_position_size
from time_utils import now_ist
from models.multi_horizon_ensemble import MultiHorizonEnsemble
from regime_analysis import MarketRegimeDetector

# Optional RL support
try:
    from models.reinforcement_learning import RLStrategy, RLState
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    RLStrategy = None
    RLState = None

SIGNAL_MAP = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}


class MLSignalGenerator:
    """
    Core ML engine that loads models and generates trading signals.
    """
    def __init__(self, exchange: str):
        self.exchange = exchange
        
        # Phase 2: Multi-Horizon Ensemble
        self.model_ensemble = MultiHorizonEnsemble(exchange)
        self.models_loaded = True 

        # Phase 5: Enhanced Regime Detector
        self.regime_detector = MarketRegimeDetector(exchange)
        
        # Reinforcement Learning (optional)
        self.rl_strategy: Optional[RLStrategy] = None
        self.use_rl: bool = False  # Enable via config or method
        if RL_AVAILABLE:
            try:
                # Try to load RL model (PPO or DQN)
                rl_model_path = Path(f"models/{exchange}/rl")
                # Check for PPO first, then DQN
                ppo_path = list(rl_model_path.glob("ppo_strategy_*.zip"))
                dqn_path = list(rl_model_path.glob("dqn_strategy_*.zip"))
                
                if ppo_path:
                    self.rl_strategy = RLStrategy(exchange, model_path=str(ppo_path[0]), algorithm="PPO")
                    if self.rl_strategy.model_loaded:
                        self.use_rl = True
                        logging.info(f"[{exchange}] RL model loaded: {ppo_path[0]}")
                elif dqn_path:
                    self.rl_strategy = RLStrategy(exchange, model_path=str(dqn_path[0]), algorithm="DQN")
                    if self.rl_strategy.model_loaded:
                        self.use_rl = True
                        logging.info(f"[{exchange}] RL model loaded: {dqn_path[0]}")
                else:
                    logging.debug(f"[{exchange}] No RL model found in {rl_model_path}")
            except Exception as e:
                logging.warning(f"[{exchange}] RL model loading failed: {e}")
        
        self.strategy_metrics = {
            'win_rate': 0.58,
            'avg_w_l_ratio': 1.55,
        }
        
        # State tracking
        self.signal_history = deque(maxlen=50)
        self.feedback_window = deque(maxlen=100)
        self.accuracy_history = deque(maxlen=50)
        self.pending_predictions: Dict[str, Dict[str, Any]] = {}
        self.degrade_threshold = 0.55
        self.needs_retrain = False
        self.last_feedback_timestamp: Optional[datetime] = None
        self.signal_sequence = 0
        
        # Rolling buffer for Sequence Models (if needed by ensemble)
        self.regime_feature_buffer: deque = deque(maxlen=60)
        
        # RL state tracking (for position/portfolio context)
        self.rl_position: float = 0.0
        self.rl_portfolio_value: float = 1_000_000.0  # Normalized

    def generate_signal(self, features_dict: Dict[str, Any]) -> Tuple[str, float, str, Dict]:
        """
        Generate a trading signal based on input features.
        """
        try:
            # DEBUG: Track confidence through the entire function
            debug_trace = []
            
            # 1. Detect Regime (Phase 5)
            # We pass the raw feature dict; the detector handles extraction
            current_regime = self.regime_detector.detect_regime(features_dict)
            regime_config = self.regime_detector.get_strategy_weights(current_regime)
            debug_trace.append(f"Step 1: Regime={current_regime}")
            
            # 2. Prepare Features for Ensemble
            # Convert dict to array (assuming feature names are consistent with training)
            # Ideally, we should enforce order. For now, we trust the features_dict keys or 
            # the ensemble should handle named inputs. 
            # In Phase 2 implementation, we passed a vector.
            # Let's flatten values. Note: strict ordering is crucial here. 
            # A robust implementation would use a schema.
            # For this context, we assume features_dict comes from feature_engineering 
            # which produces a consistent dict.
            
            # For sequence models, update buffer
            # (Simplified vector creation - in prod, ensure column order matches training)
            feature_values = list(features_dict.values())
            self.regime_feature_buffer.append(feature_values)
            
            ensemble_input = {
                'vector': np.array(feature_values).reshape(1, -1),
                'sequence': np.array(list(self.regime_feature_buffer)) if len(self.regime_feature_buffer) > 0 else None,
                'time_to_expiry_hours': features_dict.get('time_to_expiry_hours', 24.0)
            }
            
            # 3. Get Prediction from Ensemble (Phase 2) or RL
            debug_trace.append(f"Step 2: Prepared ensemble_input, vector shape={ensemble_input['vector'].shape}")
            
            if self.use_rl and self.rl_strategy and self.rl_strategy.model_loaded:
                # Use RL model for prediction
                try:
                    # Prepare state vector for RL
                    # RL models are typically trained on a subset of key features
                    # Use core features that are most relevant for trading decisions
                    core_features = [
                        'pcr_total_oi',
                        'futures_premium',
                        'vix',
                        'underlying_price',
                        'atm_shift_intensity',
                        'put_call_iv_skew',
                        'net_gamma_exposure',
                    ]
                    
                    # Extract core features from features_dict
                    rl_feature_values = []
                    for feat in core_features:
                        val = features_dict.get(feat, 0.0)
                        if pd.isna(val) if 'pd' in globals() else (val is None or (isinstance(val, float) and np.isnan(val))):
                            val = 0.0
                        rl_feature_values.append(float(val))
                    
                    # Create state vector: 7 core features + position + portfolio = 9 features
                    state_vector = np.array(rl_feature_values, dtype=np.float32)
                    # Add position and portfolio info
                    state_vector = np.append(state_vector, [
                        float(self.rl_position),
                        float(self.rl_portfolio_value / 1_000_000.0)  # Normalized
                    ])
                    
                    # Ensure shape matches model expectation (9,)
                    if state_vector.shape[0] != 9:
                        logging.warning(f"[{self.exchange}] RL state vector shape mismatch: {state_vector.shape}, expected (9,). Using ensemble instead.")
                        raise ValueError(f"State vector shape {state_vector.shape} != (9,)")
                    
                    # Get RL action
                    action = self.rl_strategy.predict(state_vector)
                    
                    # Map RL action to signal
                    signal = SIGNAL_MAP.get(action.signal, 'HOLD')
                    confidence = abs(action.position_size)  # Use position size as confidence
                    horizon = 'rl'
                    probabilities = [0.0, 0.0, 0.0]  # RL doesn't provide probabilities
                    if action.signal == 1:
                        probabilities[2] = confidence  # BUY
                    elif action.signal == -1:
                        probabilities[0] = confidence  # SELL
                    else:
                        probabilities[1] = 1.0 - confidence  # HOLD
                    
                    ensemble_result = {
                        'signal': signal,
                        'confidence': confidence,
                        'horizon': horizon,
                        'probabilities': probabilities,
                        'source': 'rl',
                        'position_size': action.position_size,
                    }
                    
                    # Update RL state
                    self.rl_position = action.position_size if action.signal > 0 else -action.position_size if action.signal < 0 else self.rl_position
                    
                except Exception as e:
                    logging.warning(f"[{self.exchange}] RL prediction failed, falling back to ensemble: {e}")
                    # Fallback to ensemble
                    ensemble_result = self.model_ensemble.predict(ensemble_input)
                    signal = ensemble_result.get('signal', 'HOLD')
                    confidence = ensemble_result.get('confidence', 0.0)
                    horizon = ensemble_result.get('horizon', 'unknown')
                    probabilities = ensemble_result.get('probabilities', [0.0, 0.0, 0.0])
            else:
                # Use standard ensemble
                ensemble_result = self.model_ensemble.predict(ensemble_input)
                signal = ensemble_result.get('signal', 'HOLD')
                original_confidence = ensemble_result.get('confidence', 0.0)
                horizon = ensemble_result.get('horizon', 'unknown')
                probabilities = ensemble_result.get('probabilities', [0.0, 0.0, 0.0]) # Sell, Hold, Buy
                
                # CRITICAL FIX: Always recalculate confidence from probabilities
                # Confidence MUST be max(probabilities) - this is the definition
                # Never trust the confidence value from ensemble_result - recalculate it
                if probabilities and len(probabilities) >= 3:
                    # Force recalculation - this is the correct confidence
                    confidence = float(max(probabilities))
                    
                    # Log if we're fixing a bug
                    if abs(original_confidence - confidence) > 0.0001:
                        if signal == 'HOLD' and len(probabilities) > 1:
                            hold_prob = float(probabilities[1])
                            wrong_conf = 1.0 - hold_prob
                            if abs(original_confidence - wrong_conf) < 0.0001:
                                logging.error(f"[{self.exchange}] ✅ FIXED CONFIDENCE BUG: was {original_confidence:.6f} (1-HOLD_prob={wrong_conf:.6f}), corrected to {confidence:.6f}")
                            else:
                                logging.warning(f"[{self.exchange}] Confidence corrected: {original_confidence:.6f} -> {confidence:.6f}")
                        else:
                            logging.warning(f"[{self.exchange}] Confidence corrected: {original_confidence:.6f} -> {confidence:.6f}")
                else:
                    # Fallback if probabilities are invalid
                    confidence = original_confidence
                    if not probabilities or len(probabilities) < 3:
                        logging.warning(f"[{self.exchange}] Cannot recalculate confidence: probabilities={probabilities}")
                
                debug_trace.append(f"Step 3: Ensemble result - signal={signal}, confidence={confidence:.6f} (was {original_confidence:.6f}), probabilities={probabilities}")
            
            # 4. Regime Adjustment (Phase 5)
            # Downgrade signal if regime is hostile
            # Store original values before any modifications
            original_signal = signal
            original_confidence = confidence
            
            # FINAL ABSOLUTE FIX: Before regime check, ensure confidence is correct
            # This is a last-ditch effort to fix the confidence bug
            if probabilities and len(probabilities) >= 3:
                max_prob_final = float(max(probabilities))
                if abs(confidence - max_prob_final) > 0.0001:
                    # Confidence is wrong - fix it now
                    if signal == 'HOLD' and len(probabilities) > 1:
                        hold_prob_final = float(probabilities[1])
                        wrong_conf_final = 1.0 - hold_prob_final
                        if abs(confidence - wrong_conf_final) < 0.0001:
                            logging.error(f"[{self.exchange}] 🚨 ABSOLUTE FIX BEFORE REGIME: confidence={confidence:.6f} was 1-HOLD_prob={wrong_conf_final:.6f}, correcting to {max_prob_final:.6f}")
                        else:
                            logging.error(f"[{self.exchange}] ABSOLUTE FIX BEFORE REGIME: confidence={confidence:.6f} != max_prob {max_prob_final:.6f}, correcting")
                    else:
                        logging.error(f"[{self.exchange}] ABSOLUTE FIX BEFORE REGIME: confidence={confidence:.6f} != max_prob {max_prob_final:.6f}, correcting")
                    confidence = max_prob_final
            
            debug_trace.append(f"Step 4a: Before regime check - signal={signal}, confidence={confidence:.6f}")
            
            # Only apply regime filters for truly extreme conditions
            if current_regime == 'HIGH_VOL_CRASH' and signal == 'BUY':
                # Suppress BUY signals during crashes
                signal = 'HOLD'
                confidence = 0.0
                rationale = "Signal suppressed by High Vol Crash regime."
                debug_trace.append(f"Step 4b: HIGH_VOL_CRASH filter applied - confidence set to 0.0")
            elif current_regime == 'LOW_VOL_COMPRESSION' and confidence < 0.6:
                # Only filter very weak signals (< 0.6) in low vol compression
                # High confidence signals (>= 0.6) should pass through
                signal = 'HOLD'
                # Keep original confidence even when filtering to HOLD (for debugging)
                # Don't modify confidence here
                rationale = "Weak signal filtered in Low Vol regime."
                debug_trace.append(f"Step 4b: LOW_VOL_COMPRESSION filter applied - signal=HOLD, confidence={confidence:.6f} (unchanged)")
            else:
                # All other cases: use ensemble prediction as-is
                rationale = f"Horizon {horizon} | Regime {current_regime} | Conf {confidence:.1%} | Signal {signal}"
                debug_trace.append(f"Step 4b: No regime filter - signal={signal}, confidence={confidence:.6f}")
            
            # Safety check: ensure confidence wasn't accidentally modified
            if signal == 'HOLD' and original_confidence > 0.5 and confidence != original_confidence and confidence != 0.0:
                # If signal was changed to HOLD but confidence was unexpectedly modified, restore it
                logging.warning(f"[{self.exchange}] Confidence unexpectedly changed from {original_confidence:.6f} to {confidence:.6f} when setting HOLD. Restoring.")
                confidence = original_confidence
                debug_trace.append(f"Step 4c: Confidence restored to {confidence:.6f}")
            
            debug_trace.append(f"Step 4d: After regime check - signal={signal}, confidence={confidence:.6f}")

            # 5. Risk Sizing
            debug_trace.append(f"Step 5a: Before risk sizing - signal={signal}, confidence={confidence:.6f}")
            risk_payload = {'fraction': 0.0, 'recommended_lots': 0, 'kelly_fraction': 0.0}
            if signal != 'HOLD':
                current_vol = float(features_dict.get('vix', 20.0)) / 100.0
                
                risk_payload = get_optimal_position_size(
                    ml_confidence=confidence,
                    win_rate=self.strategy_metrics['win_rate'],
                    avg_win_loss_ratio=self.strategy_metrics['avg_w_l_ratio'],
                    current_volatility=current_vol,
                    regime_risk_scale=regime_config['risk_scale'] # Phase 5 Scaling
                )
                debug_trace.append(f"Step 5b: After risk sizing - confidence={confidence:.6f} (should be unchanged)")

            metadata = {
                'regime': current_regime,
                'horizon': horizon,
                'buy_prob': probabilities[2] if len(probabilities) > 2 else 0.0,
                'sell_prob': probabilities[0] if len(probabilities) > 0 else 0.0,
                'position_size_frac': risk_payload.get('fraction', 0.0),
                'kelly_fraction': risk_payload.get('kelly_fraction', 0.0),
                'recommended_lots': risk_payload.get('recommended_lots', 0),
                'confidence': confidence,
                'regime_risk_scale': regime_config['risk_scale'],
                'rolling_accuracy': self._rolling_accuracy(),
                'last_feedback_at': self.last_feedback_timestamp.isoformat() if self.last_feedback_timestamp else None,
                'model_source': ensemble_result.get('source', 'ensemble'),  # 'rl' or 'ensemble'
            }
            
            # Add RL-specific metadata if using RL
            if self.use_rl and ensemble_result.get('source') == 'rl':
                metadata['rl_position_size'] = ensemble_result.get('position_size', 0.0)
                metadata['rl_algorithm'] = self.rl_strategy.algorithm if self.rl_strategy else 'unknown'

            self.signal_history.append({'signal': signal, 'confidence': confidence, 'regime': current_regime})
            metadata['signal_history'] = list(self.signal_history)[-5:]

            signal_id = self._register_prediction(signal, metadata.get('buy_prob'), metadata.get('sell_prob'))
            metadata['signal_id'] = signal_id
            
            debug_trace.append(f"Step 6: Final - signal={signal}, confidence={confidence:.6f}")
            
            # If confidence is suspiciously constant (0.251179), log the full trace
            if abs(confidence - 0.251179) < 0.0001:
                logging.error(f"[{self.exchange}] SUSPICIOUS CONFIDENCE VALUE DETECTED: {confidence:.6f}")
                logging.error(f"[{self.exchange}] Debug trace:\n" + "\n".join(debug_trace))
                logging.error(f"[{self.exchange}] Original ensemble confidence: {original_confidence:.6f}")
                logging.error(f"[{self.exchange}] Probabilities: {probabilities}")
                logging.error(f"[{self.exchange}] Metadata confidence: {metadata.get('confidence', 'N/A')}")

            return signal, confidence, rationale, metadata

        except Exception as err:
            logging.error(f"[{self.exchange}] Error during signal generation: {err}", exc_info=True)
            return 'HOLD', 0.0, 'Error during inference.', {}

    def predict_and_learn(self, features_dict: Dict[str, Any], actual_outcome: Optional[int] = None):
        """Single entry point for prediction with optional immediate feedback."""
        signal, confidence, rationale, metadata = self.generate_signal(features_dict)
        if actual_outcome is not None and metadata.get('signal_id'):
            self.record_feedback(metadata['signal_id'], actual_outcome)
        return signal, confidence, rationale, metadata

    def record_feedback(self, signal_id: str, actual_outcome: int) -> Optional[Dict[str, Any]]:
        """Record realised outcome for a prior prediction to update rolling accuracy."""
        if signal_id not in self.pending_predictions:
            logging.warning("[%s] Feedback received for unknown signal id %s", self.exchange, signal_id)
            return None

        predicted = self.pending_predictions.pop(signal_id)
        predicted_direction = predicted.get('direction', 0)
        
        success = 1.0 if actual_outcome == predicted_direction else 0.0
        self.feedback_window.append(success)
        rolling_accuracy = self._rolling_accuracy()
        self.accuracy_history.append(rolling_accuracy)
        self.last_feedback_timestamp = now_ist()
        
        degrade_triggered = (
            len(self.feedback_window) == self.feedback_window.maxlen
            and rolling_accuracy < self.degrade_threshold
        )
        self.needs_retrain = degrade_triggered

        summary = {
            'exchange': self.exchange,
            'signal_id': signal_id,
            'rolling_accuracy': rolling_accuracy,
            'degrade_triggered': degrade_triggered,
        }
        return summary

    def _register_prediction(self, signal: str, buy_prob: float | None, sell_prob: float | None) -> str:
        """Store latest prediction metadata for future feedback correlation."""
        self.signal_sequence += 1
        signal_id = f"{self.exchange}-{int(time.time() * 1000)}-{self.signal_sequence}"
        direction = 1 if signal == 'BUY' else -1 if signal == 'SELL' else 0
        
        self.pending_predictions[signal_id] = {
            'direction': direction,
            'timestamp': now_ist().isoformat(),
            'buy_prob': buy_prob,
            'sell_prob': sell_prob,
        }
        if len(self.pending_predictions) > 500:
            stale_keys = list(self.pending_predictions.keys())[:-500]
            for key in stale_keys:
                self.pending_predictions.pop(key, None)
        return signal_id

    def _rolling_accuracy(self) -> float:
        if not self.feedback_window:
            return 0.0
        return float(sum(self.feedback_window) / len(self.feedback_window))
