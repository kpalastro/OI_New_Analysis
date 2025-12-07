#!/usr/bin/env python3
"""
Check database schema and add missing tables/columns.
Compares existing schema with expected schema from database_new.py
"""

import logging
from database_new import get_db_connection, release_db_connection, get_config, initialize_database, migrate_database

logging.basicConfig(level=logging.INFO, format='%(message)s')

def get_table_columns(conn, table_name):
    """Get all columns for a table."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        ORDER BY ordinal_position
    """, (table_name,))
    return cursor.fetchall()

def check_and_fix_schema():
    """Check database schema and fix any missing tables/columns."""
    config = get_config()
    
    print("=" * 70)
    print("DATABASE SCHEMA CHECKER")
    print("=" * 70)
    print(f"Database: {config.db_name}@{config.db_host}:{config.db_port}")
    print()
    
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get all existing tables
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name
        """)
        existing_tables = {row[0] for row in cursor.fetchall()}
        
        # Expected tables
        expected_tables = {
            'option_chain_snapshots',
            'ml_features',
            'exchange_metadata',
            'training_batches',
            'vix_term_structure',
            'macro_signals',
            'order_book_depth_snapshots',
            'paper_trading_metrics',
            'multi_resolution_bars'
        }
        
        print("1. Checking tables...")
        missing_tables = expected_tables - existing_tables
        if missing_tables:
            print(f"   ⚠️  Missing tables: {missing_tables}")
        else:
            print(f"   ✓ All {len(expected_tables)} tables exist")
        
        print()
        print("2. Checking table columns...")
        
        # Define expected columns for each table
        expected_columns = {
            'option_chain_snapshots': [
                'id', 'timestamp', 'exchange', 'strike', 'option_type', 'symbol',
                'oi', 'ltp', 'token', 'underlying_price', 'moneyness',
                'time_to_expiry_seconds', 'pct_change_3m', 'pct_change_5m',
                'pct_change_10m', 'pct_change_15m', 'pct_change_30m',
                'iv', 'volume', 'best_bid', 'best_ask', 'bid_quantity',
                'ask_quantity', 'spread', 'order_book_imbalance',
                'created_at', 'updated_at'
            ],
            'ml_features': [
                'timestamp', 'exchange', 'pcr_total_oi', 'pcr_itm_oi',
                'pcr_total_volume', 'futures_premium', 'time_to_expiry_hours',
                'vix', 'underlying_price', 'underlying_future_price',
                'underlying_future_oi', 'total_itm_oi_ce', 'total_itm_oi_pe',
                'atm_shift_intensity', 'itm_ce_breadth', 'itm_pe_breadth',
                'percent_oichange_fut_3m', 'itm_oi_ce_pct_change_3m_wavg',
                'itm_oi_pe_pct_change_3m_wavg', 'dealer_vanna_exposure',
                'dealer_charm_exposure', 'net_gamma_exposure', 'gamma_flip_level',
                'ce_volume_to_oi_ratio', 'pe_volume_to_oi_ratio',
                'news_sentiment_score', 'sentiment_score_50', 'sentiment_score_100',
                'trin_50', 'trin_100', 'created_at', 'feature_payload'
            ],
            'exchange_metadata': [
                'exchange', 'last_update_time', 'last_atm_strike',
                'last_underlying_price', 'last_future_price', 'last_future_oi',
                'updated_at'
            ],
            'training_batches': [
                'id', 'exchange', 'start_timestamp', 'end_timestamp',
                'model_hash', 'artifact_path', 'csv_path', 'parquet_path',
                'metadata', 'created_at', 'dataset_version'
            ],
            'vix_term_structure': [
                'id', 'timestamp', 'exchange', 'front_month_price',
                'next_month_price', 'contango_pct', 'backwardation_pct',
                'current_vix', 'realized_vol', 'vix_ma_5d', 'vix_ma_20d',
                'vix_trend_1d', 'vix_trend_5d', 'source', 'created_at'
            ],
            'macro_signals': [
                'id', 'timestamp', 'exchange', 'fii_flow', 'dii_flow',
                'fii_dii_net', 'usdinr', 'usdinr_trend', 'crude_price',
                'crude_trend', 'banknifty_correlation', 'macro_spread',
                'risk_on_score', 'metadata', 'sentiment_score_50',
                'sentiment_confidence_50', 'trin_50', 'sentiment_score_100',
                'sentiment_confidence_100', 'trin_100', 'created_at'
            ],
            'order_book_depth_snapshots': [
                'id', 'timestamp', 'exchange', 'depth_buy_total',
                'depth_sell_total', 'depth_imbalance_ratio', 'source', 'created_at'
            ],
            'paper_trading_metrics': [
                'id', 'timestamp', 'exchange', 'executed', 'reason', 'signal',
                'confidence', 'quantity_lots', 'pnl', 'constraint_violation',
                'created_at'
            ],
            'multi_resolution_bars': [
                'id', 'timestamp', 'exchange', 'resolution', 'token', 'symbol',
                'open_price', 'high_price', 'low_price', 'close_price',
                'volume', 'oi', 'oi_change', 'vwap', 'trade_count',
                'spread_avg', 'imbalance_avg', 'created_at'
            ]
        }
        
        missing_columns = {}
        for table in expected_tables:
            if table not in existing_tables:
                continue
            
            existing_cols = {row[0] for row in get_table_columns(conn, table)}
            expected_cols = set(expected_columns.get(table, []))
            missing = expected_cols - existing_cols
            
            if missing:
                missing_columns[table] = missing
                print(f"   ⚠️  {table}: Missing {len(missing)} columns: {missing}")
            else:
                print(f"   ✓ {table}: All {len(expected_cols)} columns present")
        
        print()
        
        # If there are missing tables or columns, run initialization and migration
        if missing_tables or missing_columns:
            print("3. Fixing schema...")
            print("   Running initialize_database()...")
            try:
                initialize_database()
                print("   ✓ Database initialization completed")
            except Exception as e:
                print(f"   ⚠️  Initialization warning: {e}")
            
            print("   Running migrate_database()...")
            try:
                migrate_database()
                print("   ✓ Database migration completed")
            except Exception as e:
                print(f"   ⚠️  Migration warning: {e}")
            
            # Re-check after migration
            print()
            print("4. Re-checking after fixes...")
            conn2 = get_db_connection()
            cursor2 = conn2.cursor()
            cursor2.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                ORDER BY table_name
            """)
            existing_tables_after = {row[0] for row in cursor2.fetchall()}
            
            if expected_tables - existing_tables_after:
                print(f"   ⚠️  Still missing tables: {expected_tables - existing_tables_after}")
            else:
                print(f"   ✓ All tables now exist")
            
            # Check columns again
            still_missing = {}
            for table in expected_tables:
                if table not in existing_tables_after:
                    continue
                existing_cols = {row[0] for row in get_table_columns(conn2, table)}
                expected_cols = set(expected_columns.get(table, []))
                missing = expected_cols - existing_cols
                if missing:
                    still_missing[table] = missing
                    print(f"   ⚠️  {table}: Still missing: {missing}")
                else:
                    print(f"   ✓ {table}: All columns present")
            
            release_db_connection(conn2)
            
            if not still_missing and not (expected_tables - existing_tables_after):
                print()
                print("=" * 70)
                print("✓ SCHEMA IS COMPLETE!")
                print("=" * 70)
            else:
                print()
                print("=" * 70)
                print("⚠️  SOME ISSUES REMAIN - Please check manually")
                print("=" * 70)
        else:
            print("3. Schema is complete - no fixes needed!")
            print()
            print("=" * 70)
            print("✓ ALL TABLES AND COLUMNS ARE PRESENT")
            print("=" * 70)
        
        # Show table statistics
        print()
        print("5. Table Statistics:")
        for table in sorted(expected_tables):
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"   {table}: {count:,} records")
            except Exception as e:
                print(f"   {table}: Error - {e}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if conn:
            release_db_connection(conn)

if __name__ == '__main__':
    check_and_fix_schema()

