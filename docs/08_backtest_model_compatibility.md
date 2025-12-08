# Backtest Model Compatibility Issue

## Problem

Backtests generate "No qualifying trades" because:
1. Models were trained with sklearn 1.4.2
2. Current environment has sklearn 1.7.2
3. Version mismatch causes segmentation faults when loading/predicting
4. Models crash before generating signals

## Root Cause

The `regime_models.pkl` and `feature_selector.pkl` were saved with sklearn 1.4.2, but the current environment has 1.7.2. This causes:
- InconsistentVersionWarning when loading
- Segmentation faults when predicting
- Models fail silently, returning only HOLD signals

## Solutions

### Solution 1: Retrain Models (Recommended)

Retrain models with the current sklearn version:

```bash
python3 train_model.py NSE
```

This will:
- Create new models compatible with sklearn 1.7.2
- Generate `swing_ensemble.pkl` automatically
- Ensure feature selector compatibility

**Time**: ~10-30 minutes depending on data size

### Solution 2: Downgrade sklearn

Match the sklearn version used during training:

```bash
pip3 install scikit-learn==1.4.2
```

**Warning**: This may break other parts of the system that depend on newer sklearn features.

### Solution 3: Use Lower Confidence Threshold (Temporary)

If you can't retrain immediately, lower the confidence threshold to see if any signals are generated:

```bash
python3 backtesting/run.py \
    --exchange NSE \
    --start 2025-11-24 \
    --end 2025-12-08 \
    --min-confidence 0.25 \
    --output reports/backtests/NSE.json
```

**Note**: This may not work if models are crashing due to version mismatch.

## What We Fixed

1. ✅ **Installed ML dependencies** (xgboost, lightgbm, catboost)
2. ✅ **Fixed date handling** in `load_historical_data_for_ml`
3. ✅ **Created swing_ensemble.pkl** from regime models
4. ✅ **Added feature selector support** to `SwingTradingEnsemble`
5. ⚠️ **Still need**: Compatible sklearn version or retrained models

## Verification

After retraining or fixing sklearn version, verify models work:

```bash
# Test signal generation
python3 -c "
from ml_core import MLSignalGenerator
import numpy as np

gen = MLSignalGenerator('NSE')
features = {f'feature_{i}': np.random.randn() for i in range(112)}
signal, conf, rationale, meta = gen.generate_signal(features)
print(f'Signal: {signal}, Confidence: {conf:.3f}')
"
```

Expected output:
- Signal should be BUY, SELL, or HOLD (not always HOLD)
- Confidence should vary (not always 0.333)
- No segmentation faults

## Next Steps

1. **Retrain models** with current sklearn version
2. **Run backtest** with proper date range
3. **Verify trades** are generated
4. **Review results** in `reports/backtests/NSE.json`

## Files Modified

- `models/swing_ensemble.py` - Added feature selector support
- `models/multi_horizon_ensemble.py` - Pass exchange to SwingTradingEnsemble
- `scripts/create_swing_ensemble_from_regime.py` - Updated to use exchange parameter

## Related Issues

- sklearn version mismatch warnings
- Segmentation faults when loading models
- All signals are HOLD with confidence 0.333
- No trades generated in backtests

---

**Status**: Models need retraining for compatibility
**Priority**: High - Required for backtesting to work
**Last Updated**: 2025-12-09

