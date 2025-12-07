#!/usr/bin/env python3
"""
Complete fix for TimescaleDB hypertables - handles primary keys properly.
"""

import logging
from database_new import get_db_connection, release_db_connection, get_config

logging.basicConfig(level=logging.INFO, format='%(message)s')

def fix_hypertables_complete():
    """Fix all hypertables, handling primary key constraints properly."""
    config = get_config()
    
    print("=" * 70)
    print("COMPLETE HYPERTABLE FIX")
    print("=" * 70)
    print(f"Database: {config.db_name}@{config.db_host}:{config.db_port}")
    print()
    
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get existing hypertables
        try:
            cursor.execute("""
                SELECT hypertable_name
                FROM timescaledb_information.hypertables
                WHERE hypertable_schema = 'public'
            """)
            existing_hypertables = {row[0] for row in cursor.fetchall()}
        except Exception as e:
            print(f"⚠️  Could not query hypertables: {e}")
            existing_hypertables = set()
        
        # Tables that should be hypertables
        tables_to_hypertable = {
            'option_chain_snapshots': {
                'has_unique_with_timestamp': True,  # UNIQUE(timestamp, exchange, strike, option_type)
                'can_drop_pk': True
            },
            'ml_features': {
                'has_unique_with_timestamp': True,  # PRIMARY KEY (timestamp, exchange)
                'can_drop_pk': False  # PK already includes timestamp
            },
            'vix_term_structure': {
                'has_unique_with_timestamp': False,
                'can_drop_pk': False  # No unique constraint with timestamp
            },
            'macro_signals': {
                'has_unique_with_timestamp': False,
                'can_drop_pk': False
            },
            'order_book_depth_snapshots': {
                'has_unique_with_timestamp': False,
                'can_drop_pk': False
            },
            'paper_trading_metrics': {
                'has_unique_with_timestamp': False,
                'can_drop_pk': False
            },
            'multi_resolution_bars': {
                'has_unique_with_timestamp': True,  # UNIQUE(timestamp, exchange, resolution, token)
                'can_drop_pk': True
            }
        }
        
        print("Processing tables...")
        created_count = 0
        skipped_count = 0
        failed_count = 0
        
        for table, config_info in tables_to_hypertable.items():
            if table in existing_hypertables:
                print(f"   ⊘ {table}: Already a hypertable (skipping)")
                skipped_count += 1
                continue
            
            try:
                # Check if table exists
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = %s
                    )
                """, (table,))
                
                if not cursor.fetchone()[0]:
                    print(f"   ⚠️  {table}: Table does not exist (skipping)")
                    continue
                
                # Check row count
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                row_count = cursor.fetchone()[0]
                
                # Check current primary key
                cursor.execute("""
                    SELECT kc.column_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kc
                        ON tc.constraint_name = kc.constraint_name
                    WHERE tc.table_schema = 'public'
                        AND tc.table_name = %s
                        AND tc.constraint_type = 'PRIMARY KEY'
                    ORDER BY kc.ordinal_position
                """, (table,))
                pk_columns = [row[0] for row in cursor.fetchall()]
                
                # Strategy: For tables with unique constraints that include timestamp,
                # we can drop the primary key and use the unique constraint
                if config_info['has_unique_with_timestamp'] and config_info['can_drop_pk']:
                    if 'timestamp' not in pk_columns and pk_columns:
                        # Drop primary key constraint
                        pk_constraint_name = f"{table}_pkey"
                        print(f"   → {table}: Dropping primary key constraint...")
                        try:
                            cursor.execute(f"ALTER TABLE {table} DROP CONSTRAINT {pk_constraint_name};")
                            conn.commit()
                            print(f"      ✓ Dropped primary key")
                        except Exception as e:
                            print(f"      ⚠️  Could not drop PK: {e}")
                            conn.rollback()
                            # Try to find the actual constraint name
                            cursor.execute("""
                                SELECT constraint_name
                                FROM information_schema.table_constraints
                                WHERE table_schema = 'public'
                                    AND table_name = %s
                                    AND constraint_type = 'PRIMARY KEY'
                            """, (table,))
                            result = cursor.fetchone()
                            if result:
                                actual_name = result[0]
                                try:
                                    cursor.execute(f"ALTER TABLE {table} DROP CONSTRAINT {actual_name};")
                                    conn.commit()
                                    print(f"      ✓ Dropped primary key (found name: {actual_name})")
                                except Exception as e2:
                                    print(f"      ❌ Failed: {e2}")
                                    failed_count += 1
                                    continue
                
                # Now create hypertable
                if row_count > 0:
                    print(f"   → {table}: Has {row_count:,} rows, migrating data...")
                    query = f"SELECT create_hypertable('{table}', 'timestamp', if_not_exists => TRUE, migrate_data => TRUE);"
                else:
                    query = f"SELECT create_hypertable('{table}', 'timestamp', if_not_exists => TRUE);"
                
                cursor.execute(query)
                conn.commit()
                print(f"   ✓ {table}: Hypertable created")
                created_count += 1
                
            except Exception as e:
                error_msg = str(e)
                if 'cannot create a unique index' in error_msg.lower():
                    print(f"   ⚠️  {table}: Cannot create hypertable - primary key issue")
                    print(f"      This table has a primary key on 'id' which conflicts with partitioning.")
                    print(f"      Options:")
                    print(f"      1. Keep as regular table (recommended if data exists)")
                    print(f"      2. Manually modify: ALTER TABLE {table} DROP CONSTRAINT {table}_pkey;")
                    print(f"         Then: SELECT create_hypertable('{table}', 'timestamp', migrate_data => TRUE);")
                else:
                    print(f"   ⚠️  {table}: Error - {error_msg}")
                conn.rollback()
                failed_count += 1
        
        print()
        print("=" * 70)
        if created_count > 0:
            print(f"✓ Created {created_count} hypertable(s)")
        if skipped_count > 0:
            print(f"⊘ Skipped {skipped_count} (already exist)")
        if failed_count > 0:
            print(f"⚠️  {failed_count} table(s) need manual intervention")
        print("=" * 70)
        
        # Final verification
        print()
        print("Final status:")
        try:
            cursor.execute("""
                SELECT hypertable_name
                FROM timescaledb_information.hypertables
                WHERE hypertable_schema = 'public'
                ORDER BY hypertable_name
            """)
            final_hypertables = [row[0] for row in cursor.fetchall()]
            
            if final_hypertables:
                print(f"✓ {len(final_hypertables)} hypertable(s) active:")
                for ht in final_hypertables:
                    print(f"   - {ht}")
            
            # Show which tables are still regular tables
            all_tables = set(tables_to_hypertable.keys())
            regular_tables = all_tables - set(final_hypertables)
            if regular_tables:
                print(f"\n⊘ {len(regular_tables)} table(s) remain as regular tables:")
                for rt in sorted(regular_tables):
                    print(f"   - {rt}")
                    print(f"     (These can still benefit from indexes for time-series queries)")
        except Exception as e:
            print(f"⚠️  Could not verify: {e}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if conn:
            release_db_connection(conn)

if __name__ == '__main__':
    fix_hypertables_complete()

