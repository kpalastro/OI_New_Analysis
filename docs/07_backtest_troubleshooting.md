# Backtest Troubleshooting Guide

## Issue: "No qualifying trades generated"

### Root Cause

The backtest is generating **only HOLD signals** with low confidence (0.333), which is below the default minimum confidence threshold (0.55).

### Why This Happens

1. **Missing ML Dependencies**: XGBoost/LightGBM are not installed, causing the ensemble to be disabled
2. **Low Model Confidence**: Without the full ensemble, models only output HOLD signals with fixed low confidence
3. **High Confidence Threshold**: Default threshold (0.55) filters out all low-confidence signals

### Diagnostic Results

When running the diagnostic script:
```
Signal distribution:
  BUY:     0 (0.0%)
  SELL:    0 (0.0%)
  HOLD:  200 (100.0%)
Confidence: 0.333 (all signals)
```

## Solutions

### Solution 1: Install Missing Dependencies (Recommended)

Install the required ML libraries:

```bash
pip install xgboost lightgbm catboost
# or
pip3 install xgboost lightgbm catboost
```

After installation, models will use the full ensemble and generate BUY/SELL signals with proper confidence scores.

### Solution 2: Lower Confidence Threshold (Temporary)

If you can't install dependencies immediately, lower the confidence threshold:

```bash
python backtesting/run.py \
    --exchange NSE \
    --start 2025-11-24 \
    --end 2025-12-08 \
    --min-confidence 0.30 \
    --output reports/backtests/NSE.json
```

**Note**: This will allow trades, but they may not be reliable since the models aren't fully functional.

### Solution 3: Use Available Date Range

Ensure you're using a date range that has data in the `ml_features` table:

```bash
# Check available dates
python3 -c "
import database_new as db
conn = db.get_db_connection()
cursor = conn.cursor()
cursor.execute('SELECT MIN(timestamp)::date, MAX(timestamp)::date FROM ml_features WHERE exchange = %s', ('NSE',))
print(cursor.fetchone())
conn.close()
"
```

Currently available: **2025-11-24 to 2025-12-08**

### Solution 4: Check Model Files

Verify that model files exist:

```bash
ls -la models/NSE/
# Should show:
# - hmm_regime_model.pkl
# - ensemble/ (directory with model files)
```

## Quick Fix Commands

### 1. Install Dependencies
```bash
pip3 install xgboost lightgbm catboost
```

### 2. Run Diagnostic
```bash
python3 scripts/diagnose_backtest.py NSE 2025-11-24 2025-12-08
```

### 3. Run Backtest with Lower Threshold (if needed)
```bash
python3 backtesting/run.py \
    --exchange NSE \
    --start 2025-11-24 \
    --end 2025-12-08 \
    --min-confidence 0.30 \
    --output reports/backtests/NSE.json
```

## Expected Behavior After Fix

After installing dependencies, you should see:

```
Signal distribution:
  BUY:    45 (22.5%)
  SELL:   38 (19.0%)
  HOLD:  117 (58.5%)
Confidence: 0.45 - 0.85 (variable)
```

And the backtest should generate trades:

```
Backtest complete: 127 trades | Net PnL 45230.50 | Sharpe 1.85
```

## Verification Steps

1. **Check dependencies installed**:
   ```bash
   python3 -c "import xgboost, lightgbm, catboost; print('✓ All dependencies installed')"
   ```

2. **Run diagnostic**:
   ```bash
   python3 scripts/diagnose_backtest.py NSE 2025-11-24 2025-12-08
   ```

3. **Check for BUY/SELL signals**:
   - Should see non-zero BUY and SELL counts
   - Confidence should vary (not all 0.333)

4. **Run backtest**:
   ```bash
   python3 backtesting/run.py --exchange NSE --start 2025-11-24 --end 2025-12-08 --output reports/backtests/NSE.json
   ```

## Common Issues

### Issue: "No data available for {exchange}"

**Cause**: Date range has no data in `ml_features` table

**Solution**: 
- Check available dates in database
- Adjust `--start` and `--end` parameters
- Use date range: `2025-11-24` to `2025-12-08`

### Issue: "Models not loaded"

**Cause**: Model files missing or corrupted

**Solution**:
- Check `models/{exchange}/` directory exists
- Verify `hmm_regime_model.pkl` exists
- Check `ensemble/` directory has model files

### Issue: "All signals are HOLD"

**Cause**: Missing ML dependencies (XGBoost/LightGBM)

**Solution**: Install dependencies (see Solution 1 above)

### Issue: "No qualifying trades" (but signals exist)

**Cause**: Confidence threshold too high or position sizing returns zero

**Solution**:
- Lower `--min-confidence` (try 0.45)
- Check `strategy_metrics` in diagnostic output
- Verify `win_rate` and `avg_w_l_ratio` are reasonable

## Next Steps

1. **Install dependencies**: `pip3 install xgboost lightgbm catboost`
2. **Run diagnostic**: Verify signals are being generated
3. **Run backtest**: Use proper date range and confidence threshold
4. **Review results**: Check PnL, Sharpe ratio, and trade count

---

**Last Updated**: 2025-12-09
**Status**: Fixed date handling bug in `load_historical_data_for_ml`

