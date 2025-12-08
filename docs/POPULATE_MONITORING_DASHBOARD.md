# How to Populate the Model Monitoring Dashboard

The Model Monitoring Dashboard displays training diagnostics, cross-validation metrics, backtest results, and online learning statistics. This guide explains how to generate the required data files.

## Dashboard Data Requirements

The dashboard loads data from the following sources:

1. **Auto ML Summary** - `models/{exchange}/reports/auto_ml_summary.json` ✅ (Already exists)
2. **Training Report** - `models/{exchange}/reports/training_report.json` ❌ (Missing)
3. **Backtest Results** - `reports/backtests/{exchange}.json` ❌ (Missing)
4. **Online Learning State** - `reports/online_learning_state.json` ❌ (Missing)

## Step 1: Generate Backtest Results

Backtest results populate the "Equity & Drawdown" panel and "Backtest Snapshot" sections.

### Run Backtest for NSE:
```bash
python3 backtesting/run.py \
    --exchange NSE \
    --start 2025-11-01 \
    --end 2025-12-05 \
    --output reports/backtests/NSE.json
```

### Run Backtest for BSE:
```bash
python3 backtesting/run.py \
    --exchange BSE \
    --start 2025-11-01 \
    --end 2025-12-05 \
    --output reports/backtests/BSE.json
```

**Note:** The `reports/backtests/` directory will be created automatically if it doesn't exist.

### Backtest Parameters:
- `--start`: Start date (YYYY-MM-DD)
- `--end`: End date (YYYY-MM-DD)
- `--strategy`: Strategy ID (default: `ml_signal`)
- `--holding-period`: Holding window in minutes (default: 15)
- `--min-confidence`: Minimum ML confidence to trade (default: 0.55)
- `--account-size`: Account notional in INR (default: 1,000,000)
- `--max-risk`: Max risk per trade as fraction (default: 0.02)

## Step 2: Generate Training Reports

Training reports provide detailed training diagnostics. These are typically generated during model training.

### Option A: Run Training (if not already done):
```bash
python3 train_model.py NSE
python3 train_model.py BSE
```

### Option B: Create Minimal Training Report

If training has already been done but the report is missing, you can create a minimal report:

```python
# Create models/NSE/reports/training_report.json
{
  "exchange": "NSE",
  "training_date": "2025-12-08",
  "dataset_size": 1667,
  "features_count": 112,
  "model_type": "CatBoost",
  "cv_score": 0.52,
  "status": "completed"
}
```

## Step 3: Generate Online Learning State

Online learning statistics show feedback and performance over time. This is generated automatically when the application runs with online learning enabled.

### Enable Online Learning:
The online learning state is automatically saved to `reports/online_learning_state.json` when:
1. The main application (`oi_tracker_new.py`) is running
2. Online learning service is active
3. Feedback is collected from paper trading

### Manual Creation (if needed):
```python
# Create reports/online_learning_state.json
{
  "NSE": {
    "total_feedback": 0,
    "positive_feedback": 0,
    "negative_feedback": 0,
    "last_updated": "2025-12-08T08:00:00+05:30"
  },
  "BSE": {
    "total_feedback": 0,
    "positive_feedback": 0,
    "negative_feedback": 0,
    "last_updated": "2025-12-08T08:00:00+05:30"
  }
}
```

## Step 4: Verify Dashboard Data

After generating the required files, refresh the monitoring dashboard at:
```
http://localhost:5050/monitoring
```

The dashboard should now display:
- ✅ **Target Distribution** - From training reports
- ✅ **Optuna Best Params** - From auto_ml_summary.json
- ✅ **Walk-Forward AutoML** - From auto_ml_summary.json
- ✅ **Backtest Snapshot** - From backtest JSON files
- ✅ **Equity & Drawdown Chart** - From backtest equity curve
- ✅ **Rolling Accuracy** - From online learning state
- ✅ **Time-Series CV** - From auto_ml_summary.json segments

## Quick Start Script

Create a script to generate all required data:

```bash
#!/bin/bash
# populate_dashboard.sh

# Create directories
mkdir -p reports/backtests

# Run backtests
echo "Running NSE backtest..."
python3 backtesting/run.py --exchange NSE --start 2025-11-01 --end 2025-12-05 --output reports/backtests/NSE.json

echo "Running BSE backtest..."
python3 backtesting/run.py --exchange BSE --start 2025-11-01 --end 2025-12-05 --output reports/backtests/BSE.json

echo "✓ Dashboard data populated!"
echo "Visit http://localhost:5050/monitoring to view the dashboard"
```

## Troubleshooting

### Dashboard shows "No CV data"
- **Cause:** `auto_ml_summary.json` exists but may be empty or incomplete
- **Solution:** Re-run training to regenerate the summary:
  ```bash
  python3 train_model.py NSE
  python3 train_model.py BSE
  ```

### Backtest shows "PnL ₹0 · Sharpe 0.00"
- **Cause:** No trades were generated in the backtest period
- **Solution:** 
  - Check if models are loaded correctly
  - Verify date range has sufficient data
  - Lower `--min-confidence` threshold
  - Check database has data for the date range

### Equity chart is empty
- **Cause:** Backtest JSON file doesn't contain `equity_curve` data
- **Solution:** Ensure backtest was run with `--output` flag pointing to `reports/backtests/{exchange}.json`

### Online Learning shows "No feedback yet"
- **Cause:** Online learning hasn't collected feedback yet
- **Solution:** 
  - Run the main application to collect paper trading feedback
  - Feedback is automatically saved to `reports/online_learning_state.json`
  - This requires the application to be running and making trades

## File Structure

After populating all data, your directory structure should look like:

```
OI_Newdb/
├── models/
│   ├── NSE/
│   │   └── reports/
│   │       ├── auto_ml_summary.json ✅
│   │       └── training_report.json ✅
│   └── BSE/
│       └── reports/
│           ├── auto_ml_summary.json ✅
│           └── training_report.json ✅
└── reports/
    ├── backtests/
    │   ├── NSE.json ✅
    │   └── BSE.json ✅
    └── online_learning_state.json ✅
```

## Next Steps

1. **Run backtests** to populate equity curves and PnL metrics
2. **Generate training reports** if missing
3. **Enable online learning** to collect feedback over time
4. **Refresh dashboard** to see updated metrics

---

**Note:** The dashboard automatically refreshes data when you reload the page. No server restart is needed after generating new data files.

