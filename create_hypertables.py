#!/usr/bin/env python3
"""
Create TimescaleDB hypertables for time-series tables.
"""

import logging
from database_new import get_db_connection, release_db_connection, get_config

logging.basicConfig(level=logging.INFO, format='%(message)s')

def create_hypertables():
    """Create TimescaleDB hypertables for time-series tables."""
    config = get_config()
    
    print("=" * 70)
    print("CREATING TIMESCALEDB HYPERTABLES")
    print("=" * 70)
    print(f"Database: {config.db_name}@{config.db_host}:{config.db_port}")
    print()
    
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if TimescaleDB is available
        try:
            cursor.execute("SELECT extname FROM pg_extension WHERE extname = 'timescaledb'")
            if not cursor.fetchone():
                print("⚠️  TimescaleDB extension not installed. Attempting to create...")
                try:
                    cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
                    conn.commit()
                    print("✓ TimescaleDB extension created")
                except Exception as e:
                    print(f"❌ Could not create TimescaleDB extension: {e}")
                    print("   Please install TimescaleDB manually:")
                    print("   https://docs.timescale.com/install/latest/")
                    return
        except Exception as e:
            print(f"⚠️  Could not check TimescaleDB extension: {e}")
            return
        
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
        tables_to_hypertable = [
            'option_chain_snapshots',
            'ml_features',
            'vix_term_structure',
            'macro_signals',
            'order_book_depth_snapshots',
            'paper_trading_metrics',
            'multi_resolution_bars'
        ]
        
        print("Creating hypertables...")
        created_count = 0
        skipped_count = 0
        
        for table in tables_to_hypertable:
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
                
                # Special handling for multi_resolution_bars
                if table == 'multi_resolution_bars':
                    # Check if it's empty
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    row_count = cursor.fetchone()[0]
                    
                    if row_count == 0:
                        # Empty table - we can modify the primary key
                        # Drop the existing primary key
                        cursor.execute("ALTER TABLE multi_resolution_bars DROP CONSTRAINT multi_resolution_bars_pkey;")
                        conn.commit()
                        print(f"   → {table}: Dropped old primary key")
                        
                        # Create hypertable (will use the unique constraint)
                        query = "SELECT create_hypertable('multi_resolution_bars', 'timestamp', if_not_exists => TRUE);"
                        cursor.execute(query)
                        conn.commit()
                        print(f"   ✓ {table}: Hypertable created")
                        created_count += 1
                    else:
                        print(f"   ⚠️  {table}: Has {row_count:,} rows. Cannot modify primary key on non-empty table.")
                        print(f"      This table needs manual intervention to become a hypertable.")
                        print(f"      Consider: ALTER TABLE multi_resolution_bars DROP CONSTRAINT multi_resolution_bars_pkey;")
                        print(f"      Then: SELECT create_hypertable('multi_resolution_bars', 'timestamp', migrate_data => TRUE);")
                    continue
                
                # Check if table has data
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                row_count = cursor.fetchone()[0]
                
                if row_count > 0:
                    # Table has data, need to migrate
                    print(f"   → {table}: Has {row_count:,} rows, migrating data...")
                    query = f"SELECT create_hypertable('{table}', 'timestamp', if_not_exists => TRUE, migrate_data => TRUE);"
                else:
                    # Empty table, no migration needed
                    query = f"SELECT create_hypertable('{table}', 'timestamp', if_not_exists => TRUE);"
                
                cursor.execute(query)
                conn.commit()
                print(f"   ✓ {table}: Hypertable created")
                created_count += 1
                
            except Exception as e:
                print(f"   ⚠️  {table}: Error - {e}")
                conn.rollback()
        
        print()
        print("=" * 70)
        if created_count > 0:
            print(f"✓ Created {created_count} hypertable(s)")
        if skipped_count > 0:
            print(f"⊘ Skipped {skipped_count} (already exist)")
        print("=" * 70)
        
        # Verify final state
        print()
        print("Verifying hypertables...")
        try:
            cursor.execute("""
                SELECT hypertable_name
                FROM timescaledb_information.hypertables
                WHERE hypertable_schema = 'public'
                ORDER BY hypertable_name
            """)
            final_hypertables = [row[0] for row in cursor.fetchall()]
            
            if final_hypertables:
                print(f"✓ Found {len(final_hypertables)} hypertable(s):")
                for ht in final_hypertables:
                    print(f"   - {ht}")
            else:
                print("⚠️  No hypertables found")
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
    create_hypertables()

