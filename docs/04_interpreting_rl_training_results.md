# Interpreting RL Training Results

## Overview

This guide explains how to interpret TensorBoard metrics from reinforcement learning training for trading strategies.

---

## Key Metrics Explained

### 1. eval/mean_reward (Most Important)

**What it measures:**
- Average reward per episode during evaluation
- Higher = better trading performance
- Reward = (Position × Future Return × Portfolio Value) / Initial Capital

**How to interpret:**
- **Positive values**: Agent is making profitable trades
- **Negative values**: Agent is losing money
- **Increasing trend**: Agent is learning and improving
- **Flat line**: Agent has converged or stopped learning

**Example interpretations:**
- `0.8` = Agent makes 80% of initial capital as profit (very good)
- `0.15` = Agent makes 15% of initial capital as profit (moderate)
- `-0.1` = Agent loses 10% of initial capital (bad)

**What to look for:**
- ✅ Steady upward trend
- ✅ Consistent positive rewards
- ✅ Low variance (stable performance)
- ❌ Declining rewards over time
- ❌ High variance (unstable)

---

### 2. eval/mean_ep_length

**What it measures:**
- Average number of steps per episode
- In trading: How many time steps the agent trades before episode ends

**How to interpret:**
- **Long episodes** (30k+ steps): Agent uses most/all of the training data
- **Short episodes** (<1k steps): Episodes end early (possibly due to done conditions)
- **Consistent length**: Environment is stable

**For trading:**
- Episodes typically last the full training period
- ~35,000 steps = ~90 days × 390 minutes/day (expected)

---

### 3. rollout/exploration_rate (DQN only)

**What it measures:**
- Epsilon-greedy exploration rate
- How often agent explores random actions vs exploits learned policy

**How to interpret:**
- **High rate** (0.5+): Agent explores a lot (early training)
- **Low rate** (0.05-0.1): Agent mostly exploits learned policy (later training)
- **Constant**: Fixed exploration schedule

**DQN exploration schedule:**
- Starts high, decays over time
- Final rate: 0.05 (5% random actions)

**Note:** PPO doesn't use epsilon-greedy, so this metric doesn't apply.

---

### 4. train/value_loss

**What it measures:**
- Value function loss (how well agent estimates future rewards)
- Lower = better value estimation

**How to interpret:**
- **Decreasing**: Agent is learning to predict rewards better
- **Stable low**: Agent has good value estimates
- **Increasing**: Agent may be overfitting or unstable

---

### 5. train/policy_loss

**What it measures:**
- Policy loss (how well agent's actions match optimal policy)
- Lower = better policy

**How to interpret:**
- **Decreasing**: Agent is improving its policy
- **Stable**: Policy has converged
- **Oscillating**: Policy may be unstable

---

## Comparing Algorithms

### DQN vs PPO

| Metric | DQN | PPO | Winner |
|--------|-----|-----|--------|
| **Final Reward** | Higher (0.77) | Lower (0.12) | DQN |
| **Stability** | More variance | More stable | PPO |
| **Training Speed** | Faster | Slower | DQN |
| **Sample Efficiency** | Lower | Higher | PPO |

**When to use DQN:**
- Discrete action spaces
- Want faster training
- Can tolerate more variance
- Simple environments

**When to use PPO:**
- Continuous action spaces
- Need stable performance
- Limited training data
- Complex environments

---

## Common Patterns and What They Mean

### Pattern 1: High Initial Reward, Then Drop

**What you see:**
```
Reward: 0.8 → 0.0 → 0.8
```

**Possible causes:**
- Agent overfits to early data
- Exploration phase causes temporary performance drop
- Environment changes (regime shift)

**What to do:**
- Continue training (may recover)
- Check if reward function is correct
- Verify data quality

---

### Pattern 2: Steady Improvement

**What you see:**
```
Reward: 0.0 → 0.1 → 0.2 → 0.3
```

**Interpretation:**
- ✅ Agent is learning correctly
- ✅ Stable training
- ✅ Good sign

**What to do:**
- Continue training
- May need more timesteps to reach optimal performance

---

### Pattern 3: Oscillating Rewards

**What you see:**
```
Reward: 0.5 → 0.2 → 0.6 → 0.3 → 0.7
```

**Possible causes:**
- Learning rate too high
- Environment is non-stationary
- Reward function is noisy

**What to do:**
- Lower learning rate
- Increase training timesteps
- Smooth reward function

---

### Pattern 4: Flat Line (No Learning)

**What you see:**
```
Reward: 0.0 → 0.0 → 0.0 (constant)
```

**Possible causes:**
- Learning rate too low
- Reward function always returns 0
- Agent stuck in local minimum
- Insufficient exploration

**What to do:**
- Increase learning rate
- Check reward function
- Increase exploration
- Verify environment is working

---

## Performance Benchmarks

### For Trading Strategies

| Reward Range | Performance | Action |
|--------------|-------------|--------|
| **> 0.5** | Excellent | Use in production |
| **0.2 - 0.5** | Good | Continue training, may improve |
| **0.0 - 0.2** | Moderate | Needs more training or tuning |
| **< 0.0** | Poor | Fix reward function or environment |

### Episode Length

| Length | Interpretation |
|--------|----------------|
| **30k+ steps** | Normal (full trading period) |
| **10k - 30k** | Partial episodes (check done conditions) |
| **< 1k steps** | Episodes ending too early (check environment) |

---

## Troubleshooting

### Issue: Rewards Not Improving

**Check:**
1. Is reward function correct?
2. Are features being calculated correctly?
3. Is environment resetting properly?
4. Is learning rate appropriate?

**Solutions:**
- Verify reward calculation
- Check feature engineering
- Adjust hyperparameters
- Increase training timesteps

---

### Issue: High Variance in Rewards

**Check:**
1. Is exploration rate too high?
2. Is environment stochastic?
3. Is reward function noisy?

**Solutions:**
- Reduce exploration rate
- Smooth reward function
- Increase batch size
- Use more stable algorithm (PPO)

---

### Issue: Rewards Declining Over Time

**Check:**
1. Is agent overfitting?
2. Is data distribution changing?
3. Is reward function correct?

**Solutions:**
- Add regularization
- Use more diverse training data
- Verify reward function
- Check for data leakage

---

## Best Practices

### 1. Monitor Multiple Metrics

Don't just look at reward:
- Check value loss (convergence)
- Check policy loss (learning)
- Check episode length (environment health)

### 2. Compare Multiple Runs

- Run each algorithm multiple times
- Compare average performance
- Check variance (stability)

### 3. Use Evaluation Metrics

- `eval/mean_reward` is more reliable than `rollout/ep_rew_mean`
- Evaluation uses deterministic policy (no exploration)
- Better reflects actual trading performance

### 4. Check Training vs Evaluation

- Training rewards may be higher (exploration)
- Evaluation rewards reflect actual performance
- Focus on evaluation metrics for deployment decisions

---

## Example: Your Current Results

Based on your TensorBoard plots:

### DQN Performance
- **Final Reward**: 0.77 (excellent)
- **Trend**: Recovered from early drop
- **Stability**: Some variance, but strong final performance
- **Verdict**: ✅ Good candidate for production

### PPO Performance
- **Final Reward**: 0.12 (moderate)
- **Trend**: Steady improvement
- **Stability**: More stable than DQN
- **Verdict**: ⚠️ Needs more training or tuning

### Recommendation
1. **Continue training DQN**: Already performing well
2. **Train PPO longer**: May catch up with more timesteps
3. **Compare on validation set**: Test on unseen data
4. **Consider ensemble**: Combine both algorithms

---

## Next Steps

1. **Extend Training**: Run for more timesteps (200k-500k)
2. **Hyperparameter Tuning**: Optimize learning rate, batch size, etc.
3. **Validation**: Test on held-out data
4. **Backtesting**: Test on historical data
5. **Paper Trading**: Test in live environment with paper trades

---

## Summary

**Key Takeaways:**
- `eval/mean_reward` is the most important metric
- Higher reward = better trading performance
- DQN shows better final performance in your case
- PPO is more stable but needs more training
- Monitor multiple metrics for complete picture
- Compare algorithms on evaluation metrics, not training metrics

**Your Results:**
- DQN: 0.77 reward (excellent) ✅
- PPO: 0.12 reward (needs improvement) ⚠️
- Recommendation: Continue training both, focus on DQN for now

