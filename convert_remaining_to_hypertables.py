#!/usr/bin/env python3
"""
Convert remaining tables to hypertables by dropping primary keys.
WARNING: This will drop the 'id' primary key column constraint.
Make sure no foreign keys reference these tables.
"""

import logging
from database_new import get_db_connection, release_db_connection, get_config

logging.basicConfig(level=logging.INFO, format='%(message)s')

def convert_remaining_to_hypertables():
    """Convert remaining tables to hypertables."""
    config = get_config()
    
    print("=" * 70)
    print("CONVERTING REMAINING TABLES TO HYPERTABLES")
    print("=" * 70)
    print("⚠️  WARNING: This will drop primary key constraints on 'id' columns")
    print("⚠️  Make sure no foreign keys reference these tables")
    print()
    print(f"Database: {config.db_name}@{config.db_host}:{config.db_port}")
    print()
    
    # Tables that need conversion
    tables_to_convert = [
        'vix_term_structure',
        'macro_signals',
        'order_book_depth_snapshots',
        'paper_trading_metrics'
    ]
    
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
        
        print("Processing tables...")
        converted_count = 0
        skipped_count = 0
        failed_count = 0
        
        for table in tables_to_convert:
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
                
                # Find primary key constraint name
                cursor.execute("""
                    SELECT constraint_name
                    FROM information_schema.table_constraints
                    WHERE table_schema = 'public'
                        AND table_name = %s
                        AND constraint_type = 'PRIMARY KEY'
                """, (table,))
                pk_result = cursor.fetchone()
                
                if not pk_result:
                    print(f"   ⚠️  {table}: No primary key found (skipping)")
                    continue
                
                pk_constraint_name = pk_result[0]
                
                # Drop primary key
                print(f"   → {table}: Dropping primary key '{pk_constraint_name}'...")
                cursor.execute(f"ALTER TABLE {table} DROP CONSTRAINT {pk_constraint_name};")
                conn.commit()
                print(f"      ✓ Primary key dropped")
                
                # Create hypertable
                if row_count > 0:
                    print(f"   → {table}: Has {row_count:,} rows, migrating data...")
                    query = f"SELECT create_hypertable('{table}', 'timestamp', if_not_exists => TRUE, migrate_data => TRUE);"
                else:
                    query = f"SELECT create_hypertable('{table}', 'timestamp', if_not_exists => TRUE);"
                
                cursor.execute(query)
                conn.commit()
                print(f"   ✓ {table}: Hypertable created")
                converted_count += 1
                
            except Exception as e:
                print(f"   ❌ {table}: Error - {e}")
                conn.rollback()
                failed_count += 1
        
        print()
        print("=" * 70)
        if converted_count > 0:
            print(f"✓ Converted {converted_count} table(s) to hypertables")
        if skipped_count > 0:
            print(f"⊘ Skipped {skipped_count} (already hypertables)")
        if failed_count > 0:
            print(f"❌ Failed {failed_count} table(s)")
        print("=" * 70)
        
        # Final verification
        print()
        print("Final hypertable status:")
        try:
            cursor.execute("""
                SELECT hypertable_name
                FROM timescaledb_information.hypertables
                WHERE hypertable_schema = 'public'
                ORDER BY hypertable_name
            """)
            final_hypertables = [row[0] for row in cursor.fetchall()]
            
            if final_hypertables:
                print(f"✓ {len(final_hypertables)} hypertable(s):")
                for ht in final_hypertables:
                    print(f"   - {ht}")
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
    convert_remaining_to_hypertables()

