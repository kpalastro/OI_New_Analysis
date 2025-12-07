#!/usr/bin/env python3
"""
Check database indexes, constraints, and hypertables.
"""

import logging
from database_new import get_db_connection, release_db_connection, get_config

logging.basicConfig(level=logging.INFO, format='%(message)s')

def check_indexes_and_constraints():
    """Check indexes, constraints, and hypertables."""
    config = get_config()
    
    print("=" * 70)
    print("DATABASE INDEXES, CONSTRAINTS & HYPERTABLES CHECKER")
    print("=" * 70)
    print(f"Database: {config.db_name}@{config.db_host}:{config.db_port}")
    print()
    
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 1. Check indexes
        print("1. Checking indexes...")
        cursor.execute("""
            SELECT 
                tablename,
                indexname
            FROM pg_indexes
            WHERE schemaname = 'public'
            ORDER BY tablename, indexname
        """)
        indexes = cursor.fetchall()
        
        expected_indexes = {
            'option_chain_snapshots': ['idx_snapshots_ts_exchange'],
            'ml_features': ['idx_ml_features_ts_exchange'],
            'training_batches': ['idx_training_batches_exchange'],
            'vix_term_structure': ['idx_vix_term_structure_ts'],
            'macro_signals': ['idx_macro_signals_exchange'],
            'order_book_depth_snapshots': ['idx_depth_snapshots_exchange'],
            'paper_trading_metrics': ['idx_paper_trading_metrics_exchange_ts'],
            'multi_resolution_bars': [
                'idx_multi_res_bars_resolution_time',
                'idx_multi_res_bars_token_time'
            ]
        }
        
        indexes_by_table = {}
        for tablename, indexname in indexes:
            if tablename not in indexes_by_table:
                indexes_by_table[tablename] = []
            indexes_by_table[tablename].append(indexname)
        
        for table, expected_idx_list in expected_indexes.items():
            existing = set(indexes_by_table.get(table, []))
            expected = set(expected_idx_list)
            missing = expected - existing
            if missing:
                print(f"   ⚠️  {table}: Missing indexes: {missing}")
            else:
                print(f"   ✓ {table}: All indexes present ({len(existing)} total)")
        
        # Check for primary key indexes (auto-created)
        print()
        print("2. Checking primary keys...")
        cursor.execute("""
            SELECT
                tc.table_name,
                kc.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kc
                ON tc.constraint_name = kc.constraint_name
            WHERE tc.constraint_type = 'PRIMARY KEY'
                AND tc.table_schema = 'public'
            ORDER BY tc.table_name
        """)
        primary_keys = cursor.fetchall()
        
        expected_pks = {
            'option_chain_snapshots': ['id'],
            'ml_features': ['timestamp', 'exchange'],
            'exchange_metadata': ['exchange'],
            'training_batches': ['id'],
            'vix_term_structure': ['id'],
            'macro_signals': ['id'],
            'order_book_depth_snapshots': ['id'],
            'paper_trading_metrics': ['id'],
            'multi_resolution_bars': ['id']
        }
        
        pks_by_table = {}
        for table_name, column_name in primary_keys:
            if table_name not in pks_by_table:
                pks_by_table[table_name] = []
            pks_by_table[table_name].append(column_name)
        
        for table, expected_pk in expected_pks.items():
            existing = set(pks_by_table.get(table, []))
            expected = set(expected_pk)
            if existing == expected:
                print(f"   ✓ {table}: Primary key correct ({', '.join(existing)})")
            else:
                print(f"   ⚠️  {table}: Expected PK {expected}, found {existing}")
        
        # Check unique constraints
        print()
        print("3. Checking unique constraints...")
        cursor.execute("""
            SELECT
                tc.table_name,
                string_agg(kc.column_name, ', ' ORDER BY kc.ordinal_position) as columns
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kc
                ON tc.constraint_name = kc.constraint_name
            WHERE tc.constraint_type = 'UNIQUE'
                AND tc.table_schema = 'public'
            GROUP BY tc.table_name, tc.constraint_name
            ORDER BY tc.table_name
        """)
        unique_constraints = cursor.fetchall()
        
        expected_uniques = {
            'option_chain_snapshots': ['timestamp, exchange, strike, option_type'],
            'multi_resolution_bars': ['timestamp, exchange, resolution, token']
        }
        
        uniques_by_table = {}
        for table_name, columns in unique_constraints:
            if table_name not in uniques_by_table:
                uniques_by_table[table_name] = []
            uniques_by_table[table_name].append(columns)
        
        for table, expected_uniq_list in expected_uniques.items():
            existing = set(uniques_by_table.get(table, []))
            expected = set(expected_uniq_list)
            if existing & expected:  # At least one matches
                print(f"   ✓ {table}: Unique constraint present")
            else:
                print(f"   ⚠️  {table}: Expected unique constraint on {expected}, found {existing}")
        
        # Check hypertables (TimescaleDB)
        print()
        print("4. Checking TimescaleDB hypertables...")
        try:
            cursor.execute("""
                SELECT hypertable_name
                FROM timescaledb_information.hypertables
                WHERE hypertable_schema = 'public'
                ORDER BY hypertable_name
            """)
            hypertables = [row[0] for row in cursor.fetchall()]
            
            expected_hypertables = {
                'option_chain_snapshots',
                'ml_features',
                'vix_term_structure',
                'macro_signals',
                'order_book_depth_snapshots',
                'paper_trading_metrics',
                'multi_resolution_bars'
            }
            
            existing_hypertables = set(hypertables)
            missing = expected_hypertables - existing_hypertables
            
            if missing:
                print(f"   ⚠️  Missing hypertables: {missing}")
            else:
                print(f"   ✓ All {len(expected_hypertables)} expected hypertables are present")
                for ht in sorted(hypertables):
                    print(f"      - {ht}")
        except Exception as e:
            print(f"   ⚠️  Could not check hypertables (TimescaleDB may not be installed): {e}")
        
        print()
        print("=" * 70)
        print("✓ CHECK COMPLETE")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if conn:
            release_db_connection(conn)

if __name__ == '__main__':
    check_indexes_and_constraints()

