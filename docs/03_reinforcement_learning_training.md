# Reinforcement Learning Training Guide

## Overview

OI Gemini includes a reinforcement learning (RL) framework for:
1. **Strategy Training**: Learning optimal trading signals and position sizing
2. **Execution Optimization**: Optimizing order placement to minimize slippage

This guide explains how to train RL agents using the provided framework.

---

## Prerequisites

### Required Packages

```bash
pip install stable-baselines3[extra] gymnasium numpy pandas
```

### Optional (for TensorBoard visualization)

```bash
pip install tensorboard
```

---

## RL Framework Components

### 1. TradingEnvironment

**Purpose**: Gym-style environment for training trading strategies

**State Space**:
- Market features (PCR, VIX, OI metrics, etc.)
- Current position size (-1 to 1)
- Portfolio value (normalized)

**Action Space**:
- `signal`: -1 (SELL), 0 (HOLD), 1 (BUY)
- `position_size`: Fraction of capital (0 to 1)

**Reward Function**:
- Based on PnL normalized by initial capital
- Includes transaction costs (0.02% per trade)
- Reward = (Position × Future Return × Portfolio Value) / Initial Capital

**Location**: `models/reinforcement_learning.py::TradingEnvironment`

### 2. ExecutionEnvironment

**Purpose**: RL environment for order execution optimization

**State Space**:
- Spread (bid-ask)
- Order book imbalance
- Volatility
- Time remaining

**Action Space**:
- `price_offset`: Continuous (-2.0 to +2.0 ticks from mid-price)
- `aggression`: Discrete (0=Passive, 1=Aggressive)

**Reward Function**:
- Implementation Shortfall (slippage vs benchmark)
- Penalizes spread cost for aggressive orders

**Location**: `models/reinforcement_learning.py::ExecutionEnvironment`

---

## Training Strategy Agents

### Basic Training Command

```bash
python scripts/train_rl.py \
    --exchange NSE \
    --algorithm PPO \
    --timesteps 100000 \
    --mode strategy
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--exchange` | ✅ Yes | - | Exchange: `NSE` or `BSE` |
| `--algorithm` | ❌ No | `PPO` | Algorithm: `PPO` or `DQN` |
| `--timesteps` | ❌ No | `100000` | Number of training timesteps |
| `--days` | ❌ No | `90` | Days of historical data |
| `--mode` | ❌ No | `strategy` | Mode: `strategy` or `execution` |
| `--output` | ❌ No | `models/{exchange}/rl/` | Output directory |

### Training Examples

#### Quick Training (Testing)

```bash
# Fast training with fewer timesteps
python scripts/train_rl.py \
    --exchange NSE \
    --algorithm PPO \
    --timesteps 20000 \
    --days 30
```

#### Production Training

```bash
# Full training with more timesteps
python scripts/train_rl.py \
    --exchange NSE \
    --algorithm PPO \
    --timesteps 500000 \
    --days 180
```

#### DQN Training

```bash
# Use DQN algorithm (discrete action space)
python scripts/train_rl.py \
    --exchange NSE \
    --algorithm DQN \
    --timesteps 200000
```

### Training Process

1. **Data Loading**: Loads historical features from database
2. **Environment Creation**: Creates `TradingEnvironment` with features
3. **Model Initialization**: Initializes PPO or DQN agent
4. **Training Loop**: Agent interacts with environment, learns from rewards
5. **Model Saving**: Saves trained model to `models/{exchange}/rl/`

### Output Files

```
models/NSE/rl/
├── ppo_strategy_nse.zip          # Final trained model
├── best_model/                   # Best model during training
│   └── best_model.zip
├── checkpoints/                   # Periodic checkpoints
│   ├── rl_model_10000_steps.zip
│   ├── rl_model_20000_steps.zip
│   └── ...
├── eval_logs/                    # Evaluation logs
│   └── evaluations.npz
└── tensorboard/                  # TensorBoard logs
    └── PPO_1/
```

### Monitoring Training

#### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir models/NSE/rl/tensorboard

# Open browser to http://localhost:6006
```

**Metrics to Monitor**:
- `rollout/ep_rew_mean`: Average episode reward
- `train/value_loss`: Value function loss
- `train/policy_loss`: Policy loss
- `train/entropy_loss`: Exploration entropy

---

## Training Execution Agents

### Basic Training Command

```bash
python scripts/train_rl.py \
    --exchange NSE \
    --algorithm PPO \
    --timesteps 50000 \
    --mode execution
```

**Note**: Execution training requires tick-level order book data. Currently, the script uses placeholder data. You need to implement actual tick data loading from your database or files.

### Implementation Steps

1. **Load Tick Data**: Query order book snapshots from database
2. **Format Data**: Convert to format expected by `ExecutionEnvironment`
3. **Train Agent**: Use PPO (required for continuous action space)

---

## Algorithm Comparison

### PPO (Proximal Policy Optimization)

**Advantages**:
- Stable training
- Works with continuous and discrete actions
- Good sample efficiency
- Recommended for most use cases

**Use When**:
- Training strategy agents (signal + position size)
- Training execution agents (price offset is continuous)

### DQN (Deep Q-Network)

**Advantages**:
- Simpler architecture
- Good for discrete action spaces
- Well-studied algorithm

**Use When**:
- Simple discrete actions only
- Want to experiment with different algorithms

---

## Training Best Practices

### 1. Data Quality

- **Minimum Data**: At least 60 days of historical data
- **Feature Completeness**: Ensure all features are populated
- **No Gaps**: Avoid missing timestamps in training data

### 2. Hyperparameter Tuning

**PPO Defaults** (in `train_rl.py`):
- `learning_rate`: 3e-4
- `n_steps`: 2048
- `batch_size`: 64
- `n_epochs`: 10
- `gamma`: 0.99 (discount factor)
- `clip_range`: 0.2

**Tuning Tips**:
- Lower `learning_rate` (1e-4) for more stable training
- Increase `n_steps` for better sample efficiency
- Adjust `gamma` based on trading horizon (0.95 for short-term, 0.99 for long-term)

### 3. Training Duration

| Timesteps | Training Time | Use Case |
|-----------|---------------|----------|
| 20,000 | ~5-10 min | Quick testing |
| 100,000 | ~30-60 min | Standard training |
| 500,000 | ~3-5 hours | Production training |

### 4. Evaluation

The training script automatically:
- Evaluates model every 5,000 timesteps
- Saves best model based on evaluation performance
- Logs evaluation metrics to TensorBoard

### 5. Checkpointing

Models are saved at regular intervals (every 10,000 timesteps) to allow:
- Resuming training if interrupted
- Comparing models at different training stages
- Rolling back if performance degrades

---

## Using Trained Models

### Loading Strategy Model

```python
from models.reinforcement_learning import RLStrategy

# Load trained model
strategy = RLStrategy(
    exchange="NSE",
    model_path="models/NSE/rl/ppo_strategy_nse.zip",
    algorithm="PPO"
)

# Generate signal
state = np.array([...])  # Feature vector
action = strategy.predict(state)
print(f"Signal: {action.signal}, Position Size: {action.position_size}")
```

### Loading Execution Model

```python
from models.reinforcement_learning import RLExecutor

# Load trained model
executor = RLExecutor(
    exchange="NSE",
    model_path="models/NSE/rl/ppo_execution_nse.zip"
)

# Decide order placement
placement = executor.decide_placement(
    symbol="NIFTY25DECFUT",
    current_price=26000.0,
    spread=0.5,
    imbalance=1.2
)
print(f"Price Offset: {placement.price_offset}, Aggression: {placement.aggression}")
```

---

## Troubleshooting

### Issue: "stable-baselines3 not available"

**Solution**:
```bash
pip install stable-baselines3[extra]
```

### Issue: Training is slow

**Solutions**:
- Reduce `--timesteps`
- Reduce `--days` (less data)
- Use fewer parallel environments (reduce `n_envs` in code)

### Issue: Model not learning (reward not improving)

**Possible Causes**:
- Insufficient training data
- Reward function too sparse
- Learning rate too high/low
- Action space too large

**Solutions**:
- Increase training timesteps
- Adjust reward function (add intermediate rewards)
- Tune learning rate
- Simplify action space

### Issue: "Insufficient data" error

**Solution**:
- Ensure database has at least 60 days of historical data
- Check that features are being populated correctly
- Verify data loading function in `train_rl.py`

---

## Integration with Main System

### Current Status

The RL framework is **implemented but not yet integrated** into the main trading system. To integrate:

1. **Train Models**: Use `scripts/train_rl.py` to train agents
2. **Load in MLSignalGenerator**: Modify `ml_core.py` to optionally use RL models
3. **Add Configuration**: Add RL model paths to config
4. **Test in Paper Trading**: Validate RL signals before live trading

### Future Enhancements

- **Online Learning**: Update RL models with live feedback
- **Multi-Agent RL**: Multiple agents for different market regimes
- **Hierarchical RL**: High-level strategy + low-level execution
- **Transfer Learning**: Pre-train on historical data, fine-tune on recent data

---

## References

- **Stable Baselines3**: https://stable-baselines3.readthedocs.io/
- **PPO Paper**: https://arxiv.org/abs/1707.06347
- **DQN Paper**: https://arxiv.org/abs/1312.5602
- **Gymnasium**: https://gymnasium.farama.org/

---

## Summary

RL training in OI Gemini:

1. **Install dependencies**: `pip install stable-baselines3[extra] gymnasium`
2. **Train strategy agent**: `python scripts/train_rl.py --exchange NSE --algorithm PPO --timesteps 100000`
3. **Monitor with TensorBoard**: `tensorboard --logdir models/NSE/rl/tensorboard`
4. **Load and use**: Use `RLStrategy` or `RLExecutor` classes to load trained models

The RL framework provides a foundation for learning optimal trading strategies through trial and error, complementing the supervised learning models already in use.

