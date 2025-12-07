#!/usr/bin/env python3
"""
Migrate all TIMESTAMP columns to TIMESTAMPTZ (timestamp with time zone).
This follows TimescaleDB best practices for time-series data.
"""

import logging
from database_new import get_db_connection, release_db_connection, get_config

logging.basicConfig(level=logging.INFO, format='%(message)s')

def migrate_to_timestamptz():
    """Convert all TIMESTAMP columns to TIMESTAMPTZ."""
    config = get_config()
    
    print("=" * 70)
    print("MIGRATING TIMESTAMP TO TIMESTAMPTZ")
    print("=" * 70)
    print("This will convert all TIMESTAMP columns to TIMESTAMPTZ")
    print("(TimescaleDB best practice for time-series data)")
    print()
    print(f"Database: {config.db_name}@{config.db_host}:{config.db_port}")
    print()
    
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get all TIMESTAMP columns
        cursor.execute("""
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public'
                AND data_type = 'timestamp without time zone'
            ORDER BY table_name, column_name
        """)
        timestamp_columns = cursor.fetchall()
        
        if not timestamp_columns:
            print("✓ No TIMESTAMP columns found - all are already TIMESTAMPTZ")
            return
        
        print(f"Found {len(timestamp_columns)} TIMESTAMP columns to convert:")
        for table, column, dtype in timestamp_columns:
            print(f"   - {table}.{column}")
        
        print()
        print("Converting columns...")
        
        converted_count = 0
        failed_count = 0
        
        for table, column, dtype in timestamp_columns:
            try:
                # Check if column has a default value
                cursor.execute("""
                    SELECT column_default
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                        AND table_name = %s
                        AND column_name = %s
                """, (table, column))
                default_result = cursor.fetchone()
                has_default = default_result and default_result[0]
                
                # Convert TIMESTAMP to TIMESTAMPTZ
                # PostgreSQL will automatically convert the data
                print(f"   → {table}.{column}: Converting to TIMESTAMPTZ...")
                
                # Build ALTER COLUMN statement
                alter_sql = f"ALTER TABLE {table} ALTER COLUMN {column} TYPE TIMESTAMPTZ"
                
                # If it has a default (like NOW()), preserve it
                if has_default:
                    # Extract the default value
                    default_value = default_result[0]
                    # For NOW() defaults, we need to cast it
                    if 'now()' in default_value.lower():
                        alter_sql += f" USING {column}::timestamptz"
                        # The default will be preserved automatically
                    else:
                        alter_sql += f" USING {column}::timestamptz"
                else:
                    alter_sql += f" USING {column}::timestamptz"
                
                cursor.execute(alter_sql)
                conn.commit()
                print(f"      ✓ Converted")
                converted_count += 1
                
            except Exception as e:
                print(f"      ❌ Error: {e}")
                conn.rollback()
                failed_count += 1
        
        print()
        print("=" * 70)
        if converted_count > 0:
            print(f"✓ Converted {converted_count} column(s)")
        if failed_count > 0:
            print(f"❌ Failed {failed_count} column(s)")
        print("=" * 70)
        
        # Verify conversion
        print()
        print("Verifying conversion...")
        cursor.execute("""
            SELECT COUNT(*)
            FROM information_schema.columns
            WHERE table_schema = 'public'
                AND data_type = 'timestamp without time zone'
        """)
        remaining_timestamp = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(*)
            FROM information_schema.columns
            WHERE table_schema = 'public'
                AND data_type = 'timestamp with time zone'
        """)
        timestamptz_count = cursor.fetchone()[0]
        
        if remaining_timestamp == 0:
            print(f"✓ All TIMESTAMP columns converted to TIMESTAMPTZ")
            print(f"  Total TIMESTAMPTZ columns: {timestamptz_count}")
        else:
            print(f"⚠️  {remaining_timestamp} TIMESTAMP columns still remain")
            print(f"  TIMESTAMPTZ columns: {timestamptz_count}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if conn:
            release_db_connection(conn)

if __name__ == '__main__':
    migrate_to_timestamptz()

