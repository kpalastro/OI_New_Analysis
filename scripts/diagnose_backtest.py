#!/usr/bin/env python3
"""
Diagnostic script to identify why backtests generate no trades.
"""
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import database_new as db
from feature_engineering import REQUIRED_FEATURE_COLUMNS, prepare_training_features
from ml_core import MLSignalGenerator
from risk_manager import get_optimal_position_size

def check_data_availability(exchange: str, start: str, end: str):
    """Check if data exists in database."""
    print(f"\n{'='*60}")
    print(f"1. Checking Data Availability")
    print(f"{'='*60}")
    
    try:
        conn = db.get_db_connection()
        cursor = conn.cursor()
        
        # Check data in multi_resolution_bars
        cursor.execute('''
            SELECT 
                MIN(timestamp) as min_date,
                MAX(timestamp) as max_date,
                COUNT(*) as row_count,
                COUNT(DISTINCT DATE(timestamp)) as day_count
            FROM multi_resolution_bars 
            WHERE exchange = %s
            AND timestamp >= %s::date
            AND timestamp <= %s::date
        ''', (exchange, start, end))
        
        result = cursor.fetchone()
        print(f"  Exchange: {exchange}")
        print(f"  Date Range: {start} to {end}")
        print(f"  Min date in DB: {result[0]}")
        print(f"  Max date in DB: {result[1]}")
        print(f"  Total rows: {result[2]:,}")
        print(f"  Days with data: {result[3]}")
        
        if result[2] == 0:
            print(f"  ❌ NO DATA FOUND in database!")
            return False
        else:
            print(f"  ✅ Data found")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"  ❌ Error checking data: {e}")
        return False


def check_models(exchange: str):
    """Check if models are loaded."""
    print(f"\n{'='*60}")
    print(f"2. Checking Model Loading")
    print(f"{'='*60}")
    
    try:
        signal_engine = MLSignalGenerator(exchange)
        print(f"  Exchange: {exchange}")
        print(f"  Models loaded: {signal_engine.models_loaded}")
        
        if not signal_engine.models_loaded:
            print(f"  ❌ MODELS NOT LOADED!")
            print(f"  Check if models exist in: models/{exchange}/")
            return False
        
        print(f"  ✅ Models loaded successfully")
        print(f"  Strategy metrics: {signal_engine.strategy_metrics}")
        return True
        
    except Exception as e:
        print(f"  ❌ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_feature_preparation(exchange: str, start: str, end: str):
    """Check if features can be prepared."""
    print(f"\n{'='*60}")
    print(f"3. Checking Feature Preparation")
    print(f"{'='*60}")
    
    try:
        raw = db.load_historical_data_for_ml(exchange, start, end)
        
        if raw is None or raw.empty:
            print(f"  ❌ No raw data returned from database")
            return None
        
        print(f"  Raw data rows: {len(raw):,}")
        print(f"  Raw data columns: {len(raw.columns)}")
        
        feature_frame = prepare_training_features(raw, required_columns=REQUIRED_FEATURE_COLUMNS)
        
        if feature_frame.empty:
            print(f"  ❌ Feature frame is empty after preparation")
            return None
        
        print(f"  Feature frame rows: {len(feature_frame):,}")
        print(f"  Feature frame columns: {len(feature_frame.columns)}")
        print(f"  ✅ Features prepared successfully")
        
        # Check for required columns
        missing = [col for col in REQUIRED_FEATURE_COLUMNS if col not in feature_frame.columns]
        if missing:
            print(f"  ⚠️  Missing required columns: {missing[:10]}...")
        else:
            print(f"  ✅ All required columns present")
        
        return feature_frame
        
    except Exception as e:
        print(f"  ❌ Error preparing features: {e}")
        import traceback
        traceback.print_exc()
        return None


def check_signal_generation(exchange: str, feature_frame, sample_size: int = 100):
    """Check what signals are being generated."""
    print(f"\n{'='*60}")
    print(f"4. Checking Signal Generation")
    print(f"{'='*60}")
    
    try:
        signal_engine = MLSignalGenerator(exchange)
        
        if not signal_engine.models_loaded:
            print(f"  ❌ Models not loaded, cannot generate signals")
            return
        
        # Sample rows
        sample = feature_frame.head(sample_size)
        print(f"  Testing {len(sample)} sample rows...")
        
        signals = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        confidences = []
        low_confidence_count = 0
        min_confidence = 0.55
        
        for idx, row in sample.iterrows():
            features = {col: float(row.get(col, 0.0)) for col in REQUIRED_FEATURE_COLUMNS}
            signal, confidence, rationale, metadata = signal_engine.generate_signal(features)
            
            signals[signal] = signals.get(signal, 0) + 1
            confidences.append(confidence)
            
            if confidence < min_confidence:
                low_confidence_count += 1
        
        print(f"  Signal distribution:")
        print(f"    BUY:  {signals['BUY']:4d} ({signals['BUY']/len(sample)*100:.1f}%)")
        print(f"    SELL: {signals['SELL']:4d} ({signals['SELL']/len(sample)*100:.1f}%)")
        print(f"    HOLD: {signals['HOLD']:4d} ({signals['HOLD']/len(sample)*100:.1f}%)")
        
        if confidences:
            print(f"  Confidence statistics:")
            print(f"    Min:  {min(confidences):.3f}")
            print(f"    Max:  {max(confidences):.3f}")
            print(f"    Mean: {sum(confidences)/len(confidences):.3f}")
            print(f"    Below {min_confidence}: {low_confidence_count} ({low_confidence_count/len(sample)*100:.1f}%)")
        
        # Check position sizing
        qualifying_signals = signals['BUY'] + signals['SELL']
        if qualifying_signals == 0:
            print(f"  ❌ NO QUALIFYING SIGNALS (all HOLD or below confidence)")
        else:
            print(f"  ✅ {qualifying_signals} qualifying signals found")
            
            # Test position sizing for a few signals
            print(f"\n  Testing position sizing...")
            test_count = 0
            zero_position_count = 0
            
            for idx, row in sample.iterrows():
                features = {col: float(row.get(col, 0.0)) for col in REQUIRED_FEATURE_COLUMNS}
                signal, confidence, rationale, metadata = signal_engine.generate_signal(features)
                
                if signal != 'HOLD' and confidence >= min_confidence:
                    test_count += 1
                    if test_count > 5:  # Test first 5 qualifying signals
                        break
                    
                    risk = get_optimal_position_size(
                        ml_confidence=confidence,
                        win_rate=signal_engine.strategy_metrics.get('win_rate', 0.5),
                        avg_win_loss_ratio=signal_engine.strategy_metrics.get('avg_w_l_ratio', 1.0),
                        max_risk=0.02,
                        account_size=1_000_000.0,
                        margin_per_lot=75_000.0,
                    )
                    capital_allocated = risk.get('capital_allocated', 0.0)
                    
                    if capital_allocated <= 0.0:
                        zero_position_count += 1
                        print(f"    Signal: {signal}, Confidence: {confidence:.3f}, Capital: {capital_allocated:.2f} ❌")
                    else:
                        print(f"    Signal: {signal}, Confidence: {confidence:.3f}, Capital: {capital_allocated:.2f} ✅")
            
            if zero_position_count > 0:
                print(f"  ⚠️  {zero_position_count} signals had zero position size")
        
    except Exception as e:
        print(f"  ❌ Error generating signals: {e}")
        import traceback
        traceback.print_exc()


def main():
    import sys
    
    # Allow command line args or use defaults
    if len(sys.argv) >= 4:
        exchange = sys.argv[1]
        start = sys.argv[2]
        end = sys.argv[3]
    else:
        exchange = "NSE"
        # Auto-detect available date range
        conn = db.get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT MIN(timestamp)::date, MAX(timestamp)::date
            FROM multi_resolution_bars 
            WHERE exchange = %s
        ''', (exchange,))
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0] and result[1]:
            start = str(result[0])
            end = str(result[1])
            print(f"Auto-detected date range: {start} to {end}")
        else:
            start = "2025-12-07"
            end = "2025-12-08"
    
    print(f"\n{'='*60}")
    print(f"Backtest Diagnostic Tool")
    print(f"{'='*60}")
    print(f"Exchange: {exchange}")
    print(f"Date Range: {start} to {end}")
    
    # Step 1: Check data
    has_data = check_data_availability(exchange, start, end)
    if not has_data:
        print(f"\n❌ DIAGNOSIS: No data in database for this date range")
        print(f"   Solution: Check database or adjust date range")
        return
    
    # Step 2: Check models
    models_loaded = check_models(exchange)
    if not models_loaded:
        print(f"\n❌ DIAGNOSIS: Models not loaded")
        print(f"   Solution: Train models or check model files exist")
        return
    
    # Step 3: Check features
    feature_frame = check_feature_preparation(exchange, start, end)
    if feature_frame is None or feature_frame.empty:
        print(f"\n❌ DIAGNOSIS: Cannot prepare features")
        print(f"   Solution: Check data quality and feature engineering")
        return
    
    # Step 4: Check signals
    check_signal_generation(exchange, feature_frame, sample_size=200)
    
    print(f"\n{'='*60}")
    print(f"Diagnostic Complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

