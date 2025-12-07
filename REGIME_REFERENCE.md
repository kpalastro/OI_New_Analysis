# Complete Regime Reference Guide (R0, R1, R2, R3)

## Overview

The HMM (Hidden Markov Model) automatically discovers **4 distinct market regimes** from historical data. Each regime represents a unique combination of market conditions based on 5 key features.

## Regime Features

The HMM uses these 5 features to identify regimes:

```python
REGIME_FEATURES = [
    'vix',                      # Volatility Index (0-100+)
    'realized_vol_5m',         # 5-minute realized volatility (0-1+)
    'pcr_total_oi_zscore',    # Put-Call Ratio Z-score (-3 to +3)
    'price_roc_30m',          # 30-minute price rate of change (%)
    'breadth_divergence'      # ITM Call vs Put breadth difference (-1 to +1)
]
```

## Code Locations

### 1. Regime Definition (Training)
**File**: `train_model.py`
**Lines**: 67-133

```python
class RegimeHMMTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components: int = 4, ...):
        self.n_components = n_components  # 4 regimes
        ...
    
    def fit(self, X):
        # Extract regime features
        X_regime = X[REGIME_FEATURES].copy()
        
        # Create HMM with 4 components
        self.model = hmm.GaussianHMM(
            n_components=4,  # R0, R1, R2, R3
            covariance_type="full",
            ...
        )
        self.model.fit(X_regime)
```

### 2. Regime Prediction (Live)
**File**: `ml_core.py`
**Lines**: 240-276

```python
def _predict_regime_with_fallback(self, hmm_input: pd.DataFrame) -> int:
    """
    Predicts current regime: Returns 0, 1, 2, or 3
    """
    if len(self.regime_feature_buffer) >= 2:
        buffer_array = np.array(list(self.regime_feature_buffer))
        predicted_states = self.hmm_model.predict(buffer_array)
        return int(predicted_states[-1])  # Returns R0, R1, R2, or R3
```

### 3. Regime-Specific Model Selection
**File**: `ml_core.py`
**Lines**: 319-324

```python
# Predict current regime
current_regime = self._predict_regime_with_fallback(hmm_input)  # 0, 1, 2, or 3

# Get model trained for this regime
regime_model = self.regime_models.get(current_regime)  # Uses R0, R1, R2, or R3 model
```

### 4. Regime Model Training
**File**: `train_model.py`
**Lines**: 289-302

```python
unique_regimes = np.unique(regimes)  # [0, 1, 2, 3]

for r in unique_regimes:
    mask = (df['regime'] == r)  # Filter data for regime R0, R1, R2, or R3
    X_r = selector.transform(df.loc[mask, feature_cols])
    y_r = df.loc[mask, 'target']
    
    model = lgb.LGBMClassifier(**DEFAULT_MODEL_PARAMS)
    model.fit(X_r, y_r)
    regime_models[int(r)] = model  # Store as R0, R1, R2, or R3
```

## Regime Characteristics

### How to Determine Regime Characteristics

Run the analysis script to see actual regime characteristics:

```bash
python regime_analysis.py --exchange NSE
```

This will show:
- Mean feature values for each regime (centers)
- Covariance matrices (spread and correlations)
- Transition probabilities (how regimes change)
- Interpretations based on feature values

### Typical Regime Interpretations

Based on HMM analysis, regimes typically represent:

#### **R0 (Regime 0)**
**Typical Characteristics:**
- Low to moderate VIX (< 18)
- Low realized volatility (< 0.12)
- Neutral PCR Z-score (around 0)
- Small price movements
- Balanced breadth

**Market Behavior:**
- Calm, range-bound markets
- Sideways/choppy trading
- Low uncertainty

**Trading Strategy:**
- Range trading
- Mean reversion
- Small position sizes
- Tight stops

**Code Reference:**
```python
# In logs, you'll see:
{"regime": 0, "signal": "SELL", "confidence": 0.96}
```

---

#### **R1 (Regime 1)**
**Typical Characteristics:**
- Moderate VIX (15-22)
- Moderate realized volatility (0.10-0.18)
- Slight PCR bias (positive or negative)
- Moderate price momentum
- Some breadth divergence

**Market Behavior:**
- Trending markets with controlled volatility
- Directional moves
- Moderate uncertainty

**Trading Strategy:**
- Trend following
- Breakout strategies
- Moderate position sizes
- Standard stops

**Code Reference:**
```python
# In logs, you'll see:
{"regime": 1, "signal": "BUY", "confidence": 0.90}
```

---

#### **R2 (Regime 2)**
**Typical Characteristics:**
- High VIX (20-30)
- High realized volatility (0.15-0.25)
- Strong PCR bias (extreme values)
- Strong price momentum
- Significant breadth divergence

**Market Behavior:**
- High volatility trending markets
- Large price swings
- Strong directional bias
- High uncertainty

**Trading Strategy:**
- Momentum trading
- Larger position sizes (with high confidence)
- Wider stops
- Trend continuation plays

**Code Reference:**
```python
# In logs, you'll see:
{"regime": 2, "signal": "BUY", "confidence": 0.99}
```

---

#### **R3 (Regime 3)**
**Typical Characteristics:**
- Very high VIX (> 25)
- Very high realized volatility (> 0.20)
- Extreme PCR values
- Extreme price movements
- Extreme breadth divergence

**Market Behavior:**
- Crisis/panic conditions
- Extreme volatility
- Unpredictable moves
- Very high uncertainty

**Trading Strategy:**
- Defensive positioning
- Reduced position sizes
- Very wide stops or avoid trading
- Wait for regime change

**Code Reference:**
```python
# In logs, you'll see:
{"regime": 3, "signal": "HOLD", "confidence": 0.50}
```

## Regime Transition Patterns

The HMM learns transition probabilities between regimes:

```python
# Example transition matrix (from training)
# Probability of moving FROM row TO column:

#        R0    R1    R2    R3
# R0  [ 0.85, 0.10, 0.04, 0.01 ]  # R0 tends to stay in R0
# R1  [ 0.15, 0.70, 0.12, 0.03 ]  # R1 can move to R0 or R2
# R2  [ 0.05, 0.20, 0.65, 0.10 ]  # R2 can move to R1 or R3
# R3  [ 0.10, 0.15, 0.20, 0.55 ]  # R3 tends to persist but can calm down
```

**Key Insights:**
- Regimes tend to persist (diagonal values are high)
- Transitions are gradual (adjacent regimes more likely)
- Extreme regimes (R3) eventually calm down

## Using Regimes in Code

### Check Current Regime
```python
# In ml_core.py, generate_signal() method
metadata = {
    'regime': current_regime,  # 0, 1, 2, or 3
    'hmm_valid': self.hmm_valid,
    ...
}
```

### Access Regime-Specific Model
```python
# Get model for specific regime
regime_0_model = regime_models[0]  # R0 model
regime_1_model = regime_models[1]  # R1 model
regime_2_model = regime_models[2]  # R2 model
regime_3_model = regime_models[3]  # R3 model
```

### Regime-Based Position Sizing
```python
# Adjust position size based on regime
if current_regime == 0:  # Low vol
    max_size = 2
elif current_regime == 1:  # Moderate
    max_size = 3
elif current_regime == 2:  # High vol
    max_size = 4  # But require higher confidence
elif current_regime == 3:  # Extreme
    max_size = 1  # Defensive
```

## Real-World Examples from Logs

From `logs/recommendations/2025-12-05.jsonl`:

```json
// R0 Example (Low Volatility)
{"regime": 0, "signal": "SELL", "confidence": 0.96}

// R1 Example (Moderate Volatility)  
{"regime": 1, "signal": "BUY", "confidence": 0.90}

// R2 Example (High Volatility)
{"regime": 2, "signal": "BUY", "confidence": 0.99}
```

## Analyzing Your Specific Regimes

To see the exact characteristics of regimes in your trained model:

1. **Run the analysis script:**
   ```bash
   python regime_analysis.py --exchange NSE
   ```

2. **Check the output:**
   - Mean feature values for each regime
   - Covariance matrices
   - Transition probabilities
   - Interpretations

3. **Compare with typical patterns:**
   - R0 usually has lowest VIX/volatility
   - R2 usually has highest VIX/volatility
   - R1 and R3 are intermediate

## Summary Table

| Regime | Code | Typical VIX | Typical Vol | Market State | Strategy |
|--------|------|-------------|-------------|--------------|----------|
| **R0** | `0` | < 18 | < 0.12 | Low Vol / Sideways | Range Trading |
| **R1** | `1` | 15-22 | 0.10-0.18 | Moderate / Trending | Trend Following |
| **R2** | `2` | 20-30 | 0.15-0.25 | High Vol / Strong Trend | Momentum Trading |
| **R3** | `3` | > 25 | > 0.20 | Extreme Vol / Crisis | Defensive / Avoid |

## Important Notes

1. **Regimes are discovered automatically** - The exact characteristics depend on your training data
2. **Regime IDs (0,1,2,3) are arbitrary** - R0 in one model may not match R0 in another
3. **Run `regime_analysis.py`** to see your specific regime characteristics
4. **Regimes persist** - Markets don't switch regimes every minute
5. **Use regime-specific models** - Each regime has its own trained LightGBM model

## Code Constants

```python
# Number of regimes
N_COMPONENTS = 4  # train_model.py line 72

# Regime feature names
REGIME_FEATURES = [
    'vix',
    'realized_vol_5m', 
    'pcr_total_oi_zscore',
    'price_roc_30m',
    'breadth_divergence'
]  # train_model.py line 49, ml_core.py line 28
```

