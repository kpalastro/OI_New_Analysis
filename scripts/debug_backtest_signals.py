#!/usr/bin/env python3
"""
Debug script to see why backtest generates no trades.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtesting.engine import BacktestEngine, BacktestConfig
from datetime import date
from feature_engineering import REQUIRED_FEATURE_COLUMNS
import numpy as np

def main():
    config = BacktestConfig(
        exchange='NSE',
        start=date(2025, 11, 24),
        end=date(2025, 12, 8),
        min_confidence=0.55,
        max_risk_per_trade=0.02,
        account_size=1_000_000.0,
        margin_per_lot=75_000.0
    )
    
    engine = BacktestEngine(config)
    
    print("="*60)
    print("Backtest Signal Debug")
    print("="*60)
    
    # Prepare frame
    frame = engine._prepare_frame()
    print(f"\n1. Data Frame:")
    print(f"   Shape: {frame.shape if frame is not None and not frame.empty else 'EMPTY'}")
    
    if frame is None or frame.empty:
        print("   ❌ No data available")
        return
    
    print(f"   Rows: {len(frame)}")
    print(f"   Date range: {frame['timestamp'].min()} to {frame['timestamp'].max()}")
    
    # Test signal generation
    print(f"\n2. Signal Generation Test (first 50 rows):")
    signals = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
    confidences = []
    qualifying_signals = []
    
    for idx, row in frame.head(50).iterrows():
        features = {col: float(row.get(col, 0.0)) for col in REQUIRED_FEATURE_COLUMNS}
        signal, confidence, rationale, metadata = engine.signal_engine.generate_signal(features)
        
        signals[signal] = signals.get(signal, 0) + 1
        confidences.append(confidence)
        
        # Check if it would qualify
        qualifies = (signal != 'HOLD' and confidence >= config.min_confidence)
        if qualifies:
            qualifying_signals.append({
                'idx': idx,
                'signal': signal,
                'confidence': confidence,
                'rationale': rationale[:60]
            })
    
    print(f"   Signal distribution:")
    print(f"     BUY:  {signals['BUY']:3d}")
    print(f"     SELL: {signals['SELL']:3d}")
    print(f"     HOLD: {signals['HOLD']:3d}")
    
    if confidences:
        print(f"\n   Confidence statistics:")
        print(f"     Min:  {min(confidences):.3f}")
        print(f"     Max:  {max(confidences):.3f}")
        print(f"     Mean: {sum(confidences)/len(confidences):.3f}")
        print(f"     Above {config.min_confidence}: {sum(1 for c in confidences if c >= config.min_confidence)}")
    
    print(f"\n   Qualifying signals (non-HOLD, confidence >= {config.min_confidence}): {len(qualifying_signals)}")
    if qualifying_signals:
        print(f"   First 5 qualifying signals:")
        for qs in qualifying_signals[:5]:
            print(f"     {qs}")
    
    # Test position sizing for qualifying signals
    print(f"\n3. Position Sizing Test:")
    from risk_manager import get_optimal_position_size
    
    tested = 0
    zero_position_count = 0
    
    for idx, row in frame.head(100).iterrows():
        features = {col: float(row.get(col, 0.0)) for col in REQUIRED_FEATURE_COLUMNS}
        signal, confidence, rationale, metadata = engine.signal_engine.generate_signal(features)
        
        if signal != 'HOLD' and confidence >= config.min_confidence:
            tested += 1
            if tested > 5:
                break
            
            risk = get_optimal_position_size(
                ml_confidence=confidence,
                win_rate=engine.signal_engine.strategy_metrics['win_rate'],
                avg_win_loss_ratio=engine.signal_engine.strategy_metrics['avg_w_l_ratio'],
                max_risk=config.max_risk_per_trade,
                account_size=config.account_size,
                margin_per_lot=config.margin_per_lot,
            )
            capital = risk.get('capital_allocated', 0.0)
            
            if capital <= 0.0:
                zero_position_count += 1
                print(f"   ❌ Signal: {signal}, Conf: {confidence:.3f}, Capital: {capital:.2f}")
            else:
                print(f"   ✅ Signal: {signal}, Conf: {confidence:.3f}, Capital: {capital:.2f}")
    
    if zero_position_count > 0:
        print(f"\n   ⚠️  {zero_position_count} signals had zero position size")
    
    # Summary
    print(f"\n4. Summary:")
    total_qualifying = len(qualifying_signals)
    if total_qualifying == 0:
        print(f"   ❌ NO QUALIFYING TRADES")
        print(f"   Reason: All signals are HOLD or confidence < {config.min_confidence}")
    else:
        print(f"   ✅ {total_qualifying} qualifying signals found")
        print(f"   (But may be filtered by position sizing)")

if __name__ == "__main__":
    main()

