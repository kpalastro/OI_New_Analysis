# RL Model Integration Guide

## Overview

This guide explains how to integrate trained RL models into the main trading system and view their predictions in the monitoring dashboard.

---

## Current Status

✅ **RL Framework**: Implemented (`models/reinforcement_learning.py`)  
✅ **RL Training**: Available (`scripts/train_rl.py`)  
✅ **RL Integration**: Added to `MLSignalGenerator`  
✅ **Dashboard Support**: RL predictions appear in monitoring dashboard  

---

## How RL Models Are Used

### 1. Automatic Detection

When `MLSignalGenerator` initializes, it automatically:

1. **Checks for RL models** in `models/{exchange}/rl/`
2. **Loads PPO model** if found: `ppo_strategy_{exchange}.zip`
3. **Falls back to DQN** if PPO not found: `dqn_strategy_{exchange}.zip`
4. **Enables RL** if model loads successfully

### 2. Prediction Flow

```
Features → MLSignalGenerator.generate_signal()
    ↓
Check if RL enabled
    ↓
Yes → RL Model → Action (signal + position_size)
    ↓
No → Multi-Horizon Ensemble → Signal + Confidence
    ↓
Return signal with metadata['model_source'] = 'rl' or 'ensemble'
```

### 3. Signal Generation

**RL Prediction:**
- Input: State vector (features + position + portfolio_value)
- Output: `RLAction` with `signal` (-1/0/1) and `position_size` (0-1)
- Confidence: `abs(position_size)`
- Source: `'rl'` in metadata

**Ensemble Prediction (Fallback):**
- Input: Feature dictionary
- Output: Signal + confidence from Multi-Horizon Ensemble
- Source: `'ensemble'` in metadata

---

## Enabling RL Models

### Step 1: Train RL Model

```bash
# Train PPO model
python scripts/train_rl.py \
    --exchange NSE \
    --algorithm PPO \
    --timesteps 100000 \
    --mode strategy

# Or train DQN model
python scripts/train_rl.py \
    --exchange NSE \
    --algorithm DQN \
    --timesteps 100000 \
    --mode strategy
```

**Output:**
```
models/NSE/rl/
├── ppo_strategy_nse.zip  # or dqn_strategy_nse.zip
├── best_model/
└── checkpoints/
```

### Step 2: Verify Model Location

The system looks for models in:
```
models/{EXCHANGE}/rl/ppo_strategy_{exchange}.zip
models/{EXCHANGE}/rl/dqn_strategy_{exchange}.zip
```

**Example for NSE:**
```
models/NSE/rl/ppo_strategy_nse.zip
```

### Step 3: Restart Application

RL models are loaded at startup. Restart the application:

```bash
python oi_tracker_new.py
```

**Check logs for:**
```
[NSE] RL model loaded: models/NSE/rl/ppo_strategy_nse.zip
```

---

## Monitoring RL Predictions

### 1. Metrics Collection

RL predictions are automatically logged with `source='rl'`:

```python
collector.record_model_performance(
    signal=signal,
    confidence=confidence,
    source='rl',  # Identifies RL predictions
    metadata={
        'model_source': 'rl',
        'rl_algorithm': 'PPO',
        'rl_position_size': 0.5,
        ...
    }
)
```

### 2. Dashboard Display

The monitoring dashboard (`/monitoring`) shows:

**Model Performance Section:**
- Total signals (includes RL predictions)
- Signal breakdown (BUY/SELL/HOLD)
- Average confidence
- Confidence distribution

**RL-Specific Metrics:**
- RL predictions are tracked separately via `source='rl'`
- Can be filtered in metrics queries

### 3. Metrics Files

RL predictions are logged to:
```
metrics/phase2_metrics/{EXCHANGE}_metrics.jsonl
```

**Example entry:**
```json
{
  "type": "model_performance",
  "exchange": "NSE",
  "timestamp": "2025-12-08T15:30:00+05:30",
  "signal": "BUY",
  "confidence": 0.75,
  "source": "rl",
  "metadata": {
    "model_source": "rl",
    "rl_algorithm": "PPO",
    "rl_position_size": 0.75,
    "regime": "LOW_VOL_COMPRESSION",
    ...
  }
}
```

### 4. Querying RL Metrics

**Database Query:**
```sql
-- RL predictions only
SELECT 
    timestamp,
    signal,
    confidence,
    metadata->>'rl_algorithm' as algorithm,
    metadata->>'rl_position_size' as position_size
FROM ml_performance
WHERE source = 'rl'
ORDER BY timestamp DESC
LIMIT 100;
```

**Metrics File Query:**
```python
import json

# Read RL predictions from metrics file
with open('metrics/phase2_metrics/NSE_metrics.jsonl') as f:
    for line in f:
        data = json.loads(line)
        if data.get('source') == 'rl':
            print(f"RL Signal: {data['signal']}, Confidence: {data['confidence']}")
```

---

## Configuration

### Enable/Disable RL

**Automatic (Default):**
- RL is enabled if model file exists
- No configuration needed

**Manual Control:**
```python
# In ml_core.py, you can add config option:
use_rl = config.get('use_rl_models', True)  # Enable/disable RL
```

### Model Priority

**Current Priority:**
1. **RL Model** (if available and enabled)
2. **Multi-Horizon Ensemble** (fallback)

**Future Enhancement:**
- Ensemble voting (combine RL + Ensemble)
- Weighted combination
- Regime-based selection

---

## RL vs Ensemble Comparison

| Aspect | RL Model | Ensemble Model |
|--------|----------|----------------|
| **Training** | Reinforcement learning | Supervised learning |
| **Input** | State vector + position + portfolio | Feature dictionary |
| **Output** | Action (signal + position_size) | Signal + confidence |
| **Confidence** | Position size magnitude | Model probability |
| **Adaptation** | Learns from rewards | Requires retraining |
| **Use Case** | Optimal position sizing | Signal generation |

---

## Troubleshooting

### Issue: RL Model Not Loading

**Check:**
1. Model file exists: `ls models/NSE/rl/`
2. File name matches: `ppo_strategy_nse.zip` or `dqn_strategy_nse.zip`
3. Model is valid: Try loading manually

**Solution:**
```python
from models.reinforcement_learning import RLStrategy
strategy = RLStrategy("NSE", model_path="models/NSE/rl/ppo_strategy_nse.zip", algorithm="PPO")
print(f"Loaded: {strategy.model_loaded}")
```

### Issue: RL Predictions Not Appearing

**Check:**
1. Is RL enabled? Check logs for "RL model loaded"
2. Are predictions being made? Check `metadata['model_source']`
3. Are metrics being logged? Check `metrics/phase2_metrics/`

**Solution:**
- Verify model loaded successfully
- Check that `use_rl` is `True`
- Ensure metrics collector is recording

### Issue: RL Performance Poor

**Possible Causes:**
- Model not trained enough (increase timesteps)
- Reward function mismatch
- State representation incorrect
- Overfitting to training data

**Solutions:**
- Retrain with more timesteps
- Adjust reward function
- Verify feature engineering
- Use ensemble as fallback

---

## Best Practices

### 1. Model Selection

- **Use RL** when you want optimal position sizing
- **Use Ensemble** when you want signal generation
- **Use Both** for ensemble voting (future feature)

### 2. Monitoring

- Monitor RL predictions separately from ensemble
- Compare RL vs ensemble performance
- Track RL-specific metrics (position sizing accuracy)

### 3. Retraining

- Retrain RL models weekly (like ensemble models)
- Use same training data period
- Compare performance before deploying

---

## Future Enhancements

### Planned Features

1. **Ensemble Voting**: Combine RL + Ensemble predictions
2. **RL-Specific Dashboard**: Dedicated RL metrics view
3. **Online RL Learning**: Update RL models with live feedback
4. **Multi-Agent RL**: Different agents for different regimes

---

## Summary

**RL Integration Status:**
- ✅ Models can be trained (`scripts/train_rl.py`)
- ✅ Models auto-load if present
- ✅ Predictions appear in system
- ✅ Metrics logged with `source='rl'`
- ✅ Dashboard shows RL predictions

**To Use RL:**
1. Train model: `python scripts/train_rl.py --exchange NSE --algorithm PPO --timesteps 100000`
2. Restart application: `python oi_tracker_new.py`
3. Check logs: Look for "RL model loaded"
4. Monitor: View predictions in dashboard/metrics

**RL predictions will:**
- Appear in signal generation
- Be logged to metrics files
- Show up in monitoring dashboard
- Be tracked separately from ensemble predictions

