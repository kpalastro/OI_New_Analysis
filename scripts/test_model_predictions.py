#!/usr/bin/env python3
"""
Test script to debug why models always output the same confidence.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import joblib
from models.swing_ensemble import SwingTradingEnsemble

def test_swing_ensemble():
    exchange = "NSE"
    print(f"\n{'='*60}")
    print(f"Testing SwingTradingEnsemble for {exchange}")
    print(f"{'='*60}\n")
    
    # Load ensemble
    ensemble = SwingTradingEnsemble(exchange=exchange)
    
    print(f"1. Model Status:")
    print(f"   _is_fitted: {ensemble._is_fitted}")
    print(f"   Models available: {list(ensemble.models.keys())}")
    print(f"   Feature selector loaded: {ensemble.feature_selector is not None}")
    print(f"   Feature columns loaded: {ensemble.feature_columns is not None}")
    
    if ensemble.feature_columns:
        print(f"   Feature columns count: {len(ensemble.feature_columns)}")
        print(f"   First 10 columns: {ensemble.feature_columns[:10]}")
    
    # Check if models are actually fitted
    print(f"\n2. Model Fitting Status:")
    for name, model in ensemble.models.items():
        if hasattr(model, 'n_estimators'):
            print(f"   {name}: n_estimators={model.n_estimators}")
        if hasattr(model, 'get_params'):
            params = model.get_params()
            if 'n_estimators' in params:
                print(f"   {name}: configured n_estimators={params['n_estimators']}")
        # Check if model has been trained
        if hasattr(model, 'feature_importances_'):
            print(f"   {name}: ✓ Has feature_importances_ (trained)")
            print(f"   {name}: Feature importance shape: {model.feature_importances_.shape}")
        else:
            print(f"   {name}: ✗ No feature_importances_ (not trained)")
    
    # Test with sample features
    print(f"\n3. Testing Predictions:")
    
    # Create sample feature dict matching REQUIRED_FEATURE_COLUMNS
    from feature_engineering import REQUIRED_FEATURE_COLUMNS
    
    # Test 1: Random features
    print(f"\n   Test 1: Random features")
    features1 = {col: np.random.randn() * 0.1 for col in REQUIRED_FEATURE_COLUMNS[:50]}
    for i in range(50, len(REQUIRED_FEATURE_COLUMNS)):
        features1[REQUIRED_FEATURE_COLUMNS[i]] = 0.0
    
    try:
        pred1 = ensemble.predict(features1)
        print(f"   Result: class={pred1['class']}, probs={pred1['probabilities']}")
        print(f"   Confidence: {max(pred1['probabilities']):.3f}")
    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Different random features
    print(f"\n   Test 2: Different random features")
    features2 = {col: np.random.randn() * 0.2 for col in REQUIRED_FEATURE_COLUMNS[:50]}
    for i in range(50, len(REQUIRED_FEATURE_COLUMNS)):
        features2[REQUIRED_FEATURE_COLUMNS[i]] = 0.0
    
    try:
        pred2 = ensemble.predict(features2)
        print(f"   Result: class={pred2['class']}, probs={pred2['probabilities']}")
        print(f"   Confidence: {max(pred2['probabilities']):.3f}")
    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Check feature selector
    if ensemble.feature_selector is not None:
        print(f"\n4. Feature Selector Analysis:")
        try:
            # Create a sample array
            if ensemble.feature_columns:
                sample_X = np.array([[0.0] * len(ensemble.feature_columns)])
            else:
                sample_X = np.array([[0.0] * 100])
            
            X_selected = ensemble.feature_selector.transform(sample_X)
            print(f"   Input shape: {sample_X.shape}")
            print(f"   Output shape: {X_selected.shape}")
            print(f"   Features selected: {X_selected.shape[1]}")
            
            # Check if selector has a mask
            if hasattr(ensemble.feature_selector, 'get_support'):
                support = ensemble.feature_selector.get_support()
                print(f"   Features kept: {support.sum()} out of {len(support)}")
        except Exception as e:
            print(f"   Error analyzing selector: {e}")
            import traceback
            traceback.print_exc()
    
    # Test 4: Direct model prediction
    print(f"\n5. Direct Model Prediction Test:")
    if ensemble.models and ensemble._is_fitted:
        try:
            # Create a simple test array
            test_X = np.random.randn(1, 50)  # Small test
            if ensemble.feature_selector:
                test_X = ensemble.feature_selector.transform(test_X)
            
            for name, model in ensemble.models.items():
                if hasattr(model, 'predict_proba'):
                    try:
                        probs = model.predict_proba(test_X)
                        print(f"   {name}: probs={probs[0]}, max={probs[0].max():.3f}")
                    except Exception as e:
                        print(f"   {name}: Error - {e}")
        except Exception as e:
            print(f"   Error in direct prediction: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_swing_ensemble()

