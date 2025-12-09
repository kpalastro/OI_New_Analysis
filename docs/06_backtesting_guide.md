# Backtesting Guide

This guide explains how to run backtests on your ML trading models to evaluate their performance on historical data.

## Overview

The backtesting engine:
- Loads historical minute-level data from the database
- Replays ML signals through the same `MLSignalGenerator` used in production
- Simulates trades with configurable transaction costs and slippage
- Calculates performance metrics (PnL, Sharpe ratio, drawdown, etc.)
- Generates equity curves and trade records

## Quick Start

### Basic Backtest

Run a simple backtest for NSE:

```bash
python backtesting/run.py \
    --exchange NSE \
    --start 2025-11-01 \
    --end 2025-12-05 \
    --output reports/backtests/NSE.json
```

### Backtest for BSE

```bash
python backtesting/run.py \
    --exchange BSE \
    --start 2025-11-01 \
    --end 2025-12-05 \
    --output reports/backtests/BSE.json
```

## Command-Line Parameters

### Required Parameters

- `--exchange`: Exchange to backtest (`NSE` or `BSE`)
- `--start`: Start date in `YYYY-MM-DD` format
- `--end`: End date in `YYYY-MM-DD` format

### Optional Parameters

#### Strategy Configuration

- `--strategy`: Strategy ID (default: `ml_signal`)
- `--holding-period`: Holding window in minutes (default: `15`)
  - How long to hold a position before closing
- `--min-confidence`: Minimum ML confidence to execute a trade (default: `0.55`)
  - Only trades with confidence >= this value will be executed
  - Range: 0.0 to 1.0

#### Cost & Slippage

- `--cost-bps`: Transaction cost in basis points (default: `2.0`)
  - 1 basis point = 0.01%
  - Example: `2.0` bps = 0.02% transaction cost
- `--slippage-bps`: Slippage in basis points (default: `1.0`)
  - Estimated price slippage when entering/exiting trades
  - Example: `1.5` bps = 0.015% slippage

#### Risk Management

- `--account-size`: Account notional in INR (default: `1,000,000`)
  - Starting capital for the backtest
- `--margin-per-lot`: Margin required per index lot in INR (default: `75,000`)
  - Used to calculate position sizing
- `--max-risk`: Maximum risk per trade as fraction (default: `0.02`)
  - Example: `0.02` = 2% of account size per trade
  - Range: 0.0 to 1.0

#### Advanced Options

- `--max-trades`: Upper bound on number of trades (default: `None` = unlimited)
  - Useful for quick tests or limiting trade count
- `--limit-rows`: Optional cap on data rows for dry-runs (default: `None`)
  - Limits the number of historical rows processed
  - Useful for quick validation tests
- `--output`: Path to save JSON results file (default: `None`)
  - If not specified, results are only printed to console
- `--log-level`: Logging level (default: `INFO`)
  - Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`

## Examples

### Example 1: Conservative Strategy

Low confidence threshold, longer holding period, higher costs:

```bash
python backtesting/run.py \
    --exchange NSE \
    --start 2025-11-01 \
    --end 2025-12-05 \
    --min-confidence 0.65 \
    --holding-period 30 \
    --cost-bps 3.0 \
    --slippage-bps 2.0 \
    --output reports/backtests/NSE_conservative.json
```

### Example 2: Aggressive Strategy

High confidence threshold, shorter holding, lower costs:

```bash
python backtesting/run.py \
    --exchange NSE \
    --start 2025-11-01 \
    --end 2025-12-05 \
    --min-confidence 0.50 \
    --holding-period 10 \
    --cost-bps 1.5 \
    --slippage-bps 0.5 \
    --max-risk 0.03 \
    --output reports/backtests/NSE_aggressive.json
```

### Example 3: Quick Test (Limited Rows)

Test with limited data for fast validation:

```bash
python backtesting/run.py \
    --exchange NSE \
    --start 2025-12-01 \
    --end 2025-12-05 \
    --limit-rows 1000 \
    --max-trades 50 \
    --output reports/backtests/NSE_quick_test.json
```

### Example 4: Large Account Size

Simulate with larger capital:

```bash
python backtesting/run.py \
    --exchange NSE \
    --start 2025-11-01 \
    --end 2025-12-05 \
    --account-size 5000000 \
    --margin-per-lot 75000 \
    --output reports/backtests/NSE_large_account.json
```

## Understanding Results

### Console Output

After running a backtest, you'll see:

```
Backtest complete: 127 trades | Net PnL 45230.50 | Sharpe 1.85
Saved backtest report to reports/backtests/NSE.json
```

- **Number of trades**: Total executed trades
- **Net PnL**: Total profit/loss after costs (in INR)
- **Sharpe ratio**: Risk-adjusted return metric (higher is better)

### JSON Output Structure

The output JSON file contains:

```json
{
  "config": {
    "exchange": "NSE",
    "start": "2025-11-01",
    "end": "2025-12-05",
    "strategy": "ml_signal",
    "holding_period_minutes": 15,
    "transaction_cost_bps": 2.0,
    "slippage_bps": 1.0,
    "min_confidence": 0.55,
    "account_size": 1000000.0,
    "margin_per_lot": 75000.0,
    "max_risk_per_trade": 0.02
  },
  "metrics": {
    "num_trades": 127,
    "gross_total_pnl": 48500.25,
    "net_total_pnl": 45230.50,
    "total_costs": 3269.75,
    "sharpe_ratio": 1.85,
    "max_drawdown": -12500.00,
    "win_rate": 0.58,
    "avg_win": 1250.50,
    "avg_loss": -850.25,
    "profit_factor": 1.47
  },
  "equity_curve": [
    {
      "timestamp": "2025-11-01T09:15:00+05:30",
      "gross_equity": 0.0,
      "net_equity": 0.0
    },
    ...
  ],
  "trades": [
    {
      "timestamp": "2025-11-01T09:30:00+05:30",
      "signal": "BUY",
      "confidence": 0.72,
      "position_size": 50000.0,
      "gross_pnl": 1250.50,
      "net_pnl": 1200.25,
      "future_return": 0.025
    },
    ...
  ],
  "monte_carlo_report": {
    "num_simulations": 1000,
    "median_pnl": 42000.00,
    "percentile_5": -15000.00,
    "percentile_95": 95000.00
  }
}
```

### Key Metrics Explained

#### Performance Metrics

- **num_trades**: Total number of executed trades
- **gross_total_pnl**: Total PnL before transaction costs
- **net_total_pnl**: Total PnL after transaction costs and slippage
- **total_costs**: Sum of all transaction costs and slippage
- **sharpe_ratio**: Risk-adjusted return
  - > 1.0: Good
  - > 2.0: Excellent
  - < 0.5: Poor

#### Risk Metrics

- **max_drawdown**: Largest peak-to-trough decline (negative value)
- **win_rate**: Percentage of profitable trades (0.0 to 1.0)
- **avg_win**: Average profit per winning trade
- **avg_loss**: Average loss per losing trade
- **profit_factor**: Ratio of gross profit to gross loss
  - > 1.0: Profitable
  - > 1.5: Good
  - < 1.0: Losing strategy

#### Monte Carlo Analysis

- **num_simulations**: Number of random simulations run
- **median_pnl**: Median PnL across all simulations
- **percentile_5**: 5th percentile (worst case scenario)
- **percentile_95**: 95th percentile (best case scenario)

## Interpreting Results

### Good Results

✅ **Positive net PnL** with Sharpe > 1.0
✅ **Win rate > 50%** with profit factor > 1.2
✅ **Max drawdown < 20%** of account size
✅ **Consistent equity curve** (steady upward trend)

### Warning Signs

⚠️ **Negative net PnL** - Strategy is losing money
⚠️ **Sharpe < 0.5** - Poor risk-adjusted returns
⚠️ **Win rate < 40%** - Too many losing trades
⚠️ **Large drawdowns** - High volatility/risk
⚠️ **Erratic equity curve** - Unstable performance

### Common Issues

#### No Trades Generated

**Problem**: `num_trades: 0`

**Possible Causes**:
- Date range has no data in database
- `--min-confidence` threshold too high
- Models not loaded correctly
- All signals are `HOLD`

**Solutions**:
- Check database has data for the date range
- Lower `--min-confidence` (try 0.45)
- Verify models exist in `models/{exchange}/`
- Check logs for model loading errors

#### Zero or Negative PnL

**Problem**: `net_total_pnl: 0.0` or negative

**Possible Causes**:
- Transaction costs too high relative to profits
- Poor model performance
- Slippage eating into profits
- Wrong signal direction

**Solutions**:
- Reduce `--cost-bps` and `--slippage-bps`
- Check model accuracy metrics
- Review individual trades in JSON output
- Verify model is generating correct signals

#### High Drawdown

**Problem**: `max_drawdown` is very negative

**Possible Causes**:
- Large losing streaks
- Position sizing too aggressive
- Market regime changes

**Solutions**:
- Reduce `--max-risk` per trade
- Increase `--min-confidence` threshold
- Review trades during drawdown periods
- Consider regime-aware position sizing

## Best Practices

### 1. Start with Recent Data

Test on recent data first (last 1-2 months) before running longer backtests:

```bash
python backtesting/run.py \
    --exchange NSE \
    --start 2025-11-01 \
    --end 2025-12-05 \
    --output reports/backtests/NSE_recent.json
```

### 2. Use Realistic Costs

Don't underestimate transaction costs and slippage:
- **Transaction costs**: 2-3 bps for index futures
- **Slippage**: 1-2 bps in normal markets, 3-5 bps in volatile markets

### 3. Test Multiple Scenarios

Run backtests with different parameters:

```bash
# Conservative
python backtesting/run.py --exchange NSE --start 2025-11-01 --end 2025-12-05 \
    --min-confidence 0.65 --cost-bps 3.0 --output reports/backtests/NSE_conservative.json

# Moderate
python backtesting/run.py --exchange NSE --start 2025-11-01 --end 2025-12-05 \
    --min-confidence 0.55 --cost-bps 2.0 --output reports/backtests/NSE_moderate.json

# Aggressive
python backtesting/run.py --exchange NSE --start 2025-11-01 --end 2025-12-05 \
    --min-confidence 0.45 --cost-bps 1.5 --output reports/backtests/NSE_aggressive.json
```

### 4. Compare Across Exchanges

Run the same strategy on both NSE and BSE:

```bash
python backtesting/run.py --exchange NSE --start 2025-11-01 --end 2025-12-05 \
    --output reports/backtests/NSE.json

python backtesting/run.py --exchange BSE --start 2025-11-01 --end 2025-12-05 \
    --output reports/backtests/BSE.json
```

### 5. Review Individual Trades

Open the JSON output and examine individual trades:
- Look for patterns in winning vs losing trades
- Check if confidence correlates with profitability
- Identify time periods with poor performance

## Walk-Forward Analysis

For more robust testing, use walk-forward analysis to test on multiple time periods:

```bash
python -m backtesting.walk_forward \
    --exchange NSE \
    --start 2025-01-01 \
    --end 2025-12-05 \
    --train-days 60 \
    --test-days 30 \
    --step-days 15
```

This splits the data into training and testing periods, retraining models on each training window.

## Integration with Dashboard

Backtest results automatically appear in the Model Monitoring Dashboard:

1. Run backtest and save to `reports/backtests/{exchange}.json`
2. Open dashboard at `http://localhost:5050/monitoring`
3. View "Backtest Snapshot" and "Equity & Drawdown" panels

## Troubleshooting

### Error: "No data available for {exchange}"

**Solution**: Check database has data for the date range:
```sql
SELECT MIN(timestamp), MAX(timestamp) 
FROM multi_resolution_bars 
WHERE exchange = 'NSE';
```

### Error: "Models not loaded for exchange {exchange}"

**Solution**: Ensure models exist:
```bash
ls models/NSE/
# Should show: hmm_regime_model.pkl, ensemble/, etc.
```

### Error: "ModuleNotFoundError"

**Solution**: Install dependencies:
```bash
pip install -r requirements.txt
```

### Backtest Takes Too Long

**Solutions**:
- Use `--limit-rows` for quick tests
- Use `--max-trades` to limit trade count
- Reduce date range
- Use `--log-level ERROR` to reduce logging overhead

## Next Steps

1. **Run initial backtest** on recent data
2. **Analyze results** and identify issues
3. **Adjust parameters** and re-run
4. **Compare strategies** (conservative vs aggressive)
5. **Run walk-forward analysis** for robust validation
6. **View results in dashboard** for visualization

---

**Note**: Backtesting is not a guarantee of future performance. Always use proper risk management and start with paper trading before going live.

