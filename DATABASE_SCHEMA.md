# Database Schema Documentation

This document lists all database tables and fields required to run the OI Tracker application.

## Database Type
- **PostgreSQL** (with TimescaleDB extension recommended)
- Database name: `oi_db_new` (configurable via `OI_TRACKER_DB_NAME`)

---

## Tables Overview

The database consists of **9 main tables**:

1. `option_chain_snapshots` - Raw option chain data
2. `ml_features` - Machine learning features
3. `exchange_metadata` - Exchange state tracking
4. `training_batches` - Training dataset metadata
5. `vix_term_structure` - VIX term structure data
6. `macro_signals` - Macroeconomic signals
7. `order_book_depth_snapshots` - Order book depth metrics
8. `paper_trading_metrics` - Paper trading execution logs
9. `multi_resolution_bars` - Multi-resolution OHLCV bars

---

## Table 1: `option_chain_snapshots`

**Purpose**: Stores raw option chain snapshot data at each timestamp.

**Fields**:

| Field Name | Data Type | Nullable | Description |
|------------|-----------|----------|-------------|
| `id` | SERIAL | NO | Primary key (auto-increment) |
| `timestamp` | TIMESTAMPTZ | NO | Snapshot timestamp (with timezone) |
| `exchange` | TEXT | NO | Exchange name (NSE, BSE, etc.) |
| `strike` | DOUBLE PRECISION | NO | Strike price |
| `option_type` | TEXT | NO | Option type (CE or PE) |
| `symbol` | TEXT | NO | Option symbol |
| `oi` | BIGINT | YES | Open interest |
| `ltp` | DOUBLE PRECISION | YES | Last traded price |
| `token` | BIGINT | NO | Instrument token |
| `underlying_price` | DOUBLE PRECISION | YES | Underlying asset price |
| `moneyness` | TEXT | YES | Moneyness (ITM, ATM, OTM) |
| `time_to_expiry_seconds` | INTEGER | YES | Time to expiry in seconds |
| `pct_change_3m` | DOUBLE PRECISION | YES | % OI change in 3 minutes |
| `pct_change_5m` | DOUBLE PRECISION | YES | % OI change in 5 minutes |
| `pct_change_10m` | DOUBLE PRECISION | YES | % OI change in 10 minutes |
| `pct_change_15m` | DOUBLE PRECISION | YES | % OI change in 15 minutes |
| `pct_change_30m` | DOUBLE PRECISION | YES | % OI change in 30 minutes |
| `iv` | DOUBLE PRECISION | YES | Implied volatility |
| `volume` | BIGINT | YES | Trading volume |
| `best_bid` | DOUBLE PRECISION | YES | Best bid price |
| `best_ask` | DOUBLE PRECISION | YES | Best ask price |
| `bid_quantity` | DOUBLE PRECISION | YES | Bid quantity |
| `ask_quantity` | DOUBLE PRECISION | YES | Ask quantity |
| `spread` | DOUBLE PRECISION | YES | Bid-ask spread |
| `order_book_imbalance` | DOUBLE PRECISION | YES | Order book imbalance ratio |
| `created_at` | TIMESTAMPTZ | NO | Record creation timestamp (default: NOW()) |
| `updated_at` | TIMESTAMPTZ | NO | Record update timestamp (default: NOW()) |

**Constraints**:
- PRIMARY KEY: `id`
- UNIQUE: `(timestamp, exchange, strike, option_type)`

**Indexes**:
- `idx_snapshots_ts_exchange` on `(timestamp, exchange)`

**Hypertable**: Yes (TimescaleDB)

---

## Table 2: `ml_features`

**Purpose**: Stores engineered ML features aggregated at timestamp level.

**Fields**:

| Field Name | Data Type | Nullable | Description |
|------------|-----------|----------|-------------|
| `timestamp` | TIMESTAMPTZ | NO | Feature timestamp (IST) |
| `exchange` | TEXT | NO | Exchange name |
| `pcr_total_oi` | DOUBLE PRECISION | YES | Put-Call Ratio (total OI) |
| `pcr_itm_oi` | DOUBLE PRECISION | YES | Put-Call Ratio (ITM OI) |
| `pcr_total_volume` | DOUBLE PRECISION | YES | Put-Call Ratio (total volume) |
| `futures_premium` | DOUBLE PRECISION | YES | Futures premium over spot |
| `time_to_expiry_hours` | DOUBLE PRECISION | YES | Time to expiry in hours |
| `vix` | DOUBLE PRECISION | YES | VIX value |
| `underlying_price` | DOUBLE PRECISION | YES | Underlying asset price |
| `underlying_future_price` | DOUBLE PRECISION | YES | Underlying future price |
| `underlying_future_oi` | DOUBLE PRECISION | YES | Underlying future OI |
| `total_itm_oi_ce` | DOUBLE PRECISION | YES | Total ITM Call OI |
| `total_itm_oi_pe` | DOUBLE PRECISION | YES | Total ITM Put OI |
| `atm_shift_intensity` | DOUBLE PRECISION | YES | ATM shift intensity |
| `itm_ce_breadth` | DOUBLE PRECISION | YES | ITM Call breadth |
| `itm_pe_breadth` | DOUBLE PRECISION | YES | ITM Put breadth |
| `percent_oichange_fut_3m` | DOUBLE PRECISION | YES | % OI change in futures (3m) |
| `itm_oi_ce_pct_change_3m_wavg` | DOUBLE PRECISION | YES | ITM CE OI % change (3m, weighted avg) |
| `itm_oi_pe_pct_change_3m_wavg` | DOUBLE PRECISION | YES | ITM PE OI % change (3m, weighted avg) |
| `dealer_vanna_exposure` | DOUBLE PRECISION | YES | Dealer vanna exposure |
| `dealer_charm_exposure` | DOUBLE PRECISION | YES | Dealer charm exposure |
| `net_gamma_exposure` | DOUBLE PRECISION | YES | Net gamma exposure |
| `gamma_flip_level` | DOUBLE PRECISION | YES | Gamma flip level |
| `ce_volume_to_oi_ratio` | DOUBLE PRECISION | YES | Call volume to OI ratio |
| `pe_volume_to_oi_ratio` | DOUBLE PRECISION | YES | Put volume to OI ratio |
| `news_sentiment_score` | DOUBLE PRECISION | YES | News sentiment score |
| `sentiment_score_50` | DOUBLE PRECISION | YES | NIFTY50 sentiment score |
| `sentiment_score_100` | DOUBLE PRECISION | YES | NIFTY100 sentiment score |
| `trin_50` | DOUBLE PRECISION | YES | NIFTY50 TRIN value |
| `trin_100` | DOUBLE PRECISION | YES | NIFTY100 TRIN value |
| `created_at` | TIMESTAMPTZ | NO | Record creation timestamp (default: NOW()) |
| `feature_payload` | TEXT | YES | JSON payload with additional features |

**Constraints**:
- PRIMARY KEY: `(timestamp, exchange)`

**Indexes**:
- `idx_ml_features_ts_exchange` on `(timestamp, exchange)`

**Hypertable**: Yes (TimescaleDB)

---

## Table 3: `exchange_metadata`

**Purpose**: Tracks current state of each exchange (last update time, ATM strike, etc.).

**Fields**:

| Field Name | Data Type | Nullable | Description |
|------------|-----------|----------|-------------|
| `exchange` | TEXT | NO | Exchange name (PRIMARY KEY) |
| `last_update_time` | TIMESTAMPTZ | NO | Last update timestamp |
| `last_atm_strike` | DOUBLE PRECISION | YES | Last ATM strike price |
| `last_underlying_price` | DOUBLE PRECISION | YES | Last underlying price |
| `last_future_price` | DOUBLE PRECISION | YES | Last future price |
| `last_future_oi` | BIGINT | YES | Last future OI |
| `updated_at` | TIMESTAMPTZ | NO | Record update timestamp (default: NOW()) |

**Constraints**:
- PRIMARY KEY: `exchange`

**Indexes**: None (small table, primary key is sufficient)

---

## Table 4: `training_batches`

**Purpose**: Tracks training dataset exports and model artifacts.

**Fields**:

| Field Name | Data Type | Nullable | Description |
|------------|-----------|----------|-------------|
| `id` | SERIAL | NO | Primary key (auto-increment) |
| `exchange` | TEXT | NO | Exchange name |
| `start_timestamp` | TIMESTAMPTZ | NO | Training window start |
| `end_timestamp` | TIMESTAMPTZ | NO | Training window end |
| `model_hash` | TEXT | YES | Model hash/version |
| `artifact_path` | TEXT | YES | Path to model artifact |
| `csv_path` | TEXT | YES | Path to CSV export |
| `parquet_path` | TEXT | YES | Path to Parquet export |
| `metadata` | TEXT | YES | JSON metadata |
| `created_at` | TIMESTAMPTZ | NO | Record creation timestamp (default: NOW()) |
| `dataset_version` | TEXT | YES | Dataset version identifier |

**Constraints**:
- PRIMARY KEY: `id`

**Indexes**:
- `idx_training_batches_exchange` on `(exchange, start_timestamp)`

---

## Table 5: `vix_term_structure`

**Purpose**: Stores VIX term structure data (front month, next month, contango/backwardation).

**Fields**:

| Field Name | Data Type | Nullable | Description |
|------------|-----------|----------|-------------|
| `id` | SERIAL | NO | Primary key (auto-increment) |
| `timestamp` | TIMESTAMPTZ | NO | Snapshot timestamp |
| `exchange` | TEXT | NO | Exchange name |
| `front_month_price` | DOUBLE PRECISION | YES | Front month VIX price |
| `next_month_price` | DOUBLE PRECISION | YES | Next month VIX price |
| `contango_pct` | DOUBLE PRECISION | YES | Contango percentage |
| `backwardation_pct` | DOUBLE PRECISION | YES | Backwardation percentage |
| `current_vix` | DOUBLE PRECISION | YES | Current VIX value |
| `realized_vol` | DOUBLE PRECISION | YES | Realized volatility |
| `vix_ma_5d` | DOUBLE PRECISION | YES | VIX 5-day moving average |
| `vix_ma_20d` | DOUBLE PRECISION | YES | VIX 20-day moving average |
| `vix_trend_1d` | DOUBLE PRECISION | YES | VIX 1-day trend |
| `vix_trend_5d` | DOUBLE PRECISION | YES | VIX 5-day trend |
| `source` | TEXT | YES | Data source identifier |
| `created_at` | TIMESTAMPTZ | NO | Record creation timestamp (default: NOW()) |

**Constraints**:
- PRIMARY KEY: `id`

**Indexes**:
- `idx_vix_term_structure_ts` on `(timestamp DESC)`

**Hypertable**: Yes (TimescaleDB)

---

## Table 6: `macro_signals`

**Purpose**: Stores macroeconomic signals (FII/DII flows, USD/INR, crude, sentiment).

**Fields**:

| Field Name | Data Type | Nullable | Description |
|------------|-----------|----------|-------------|
| `id` | SERIAL | NO | Primary key (auto-increment) |
| `timestamp` | TIMESTAMPTZ | NO | Snapshot timestamp |
| `exchange` | TEXT | NO | Exchange name |
| `fii_flow` | DOUBLE PRECISION | YES | FII net flow (₹ Cr) |
| `dii_flow` | DOUBLE PRECISION | YES | DII net flow (₹ Cr) |
| `fii_dii_net` | DOUBLE PRECISION | YES | Combined FII+DII net flow |
| `usdinr` | DOUBLE PRECISION | YES | USD/INR exchange rate |
| `usdinr_trend` | DOUBLE PRECISION | YES | USD/INR trend (% change) |
| `crude_price` | DOUBLE PRECISION | YES | Crude oil price |
| `crude_trend` | DOUBLE PRECISION | YES | Crude oil trend (% change) |
| `banknifty_correlation` | DOUBLE PRECISION | YES | BankNifty correlation |
| `macro_spread` | DOUBLE PRECISION | YES | Macro spread metric |
| `risk_on_score` | DOUBLE PRECISION | YES | Risk-on score |
| `metadata` | TEXT | YES | JSON metadata |
| `sentiment_score_50` | DOUBLE PRECISION | YES | NIFTY50 sentiment score (0-100) |
| `sentiment_confidence_50` | DOUBLE PRECISION | YES | NIFTY50 sentiment confidence (0-100) |
| `trin_50` | DOUBLE PRECISION | YES | NIFTY50 TRIN value |
| `sentiment_score_100` | DOUBLE PRECISION | YES | NIFTY100 sentiment score (0-100) |
| `sentiment_confidence_100` | DOUBLE PRECISION | YES | NIFTY100 sentiment confidence (0-100) |
| `trin_100` | DOUBLE PRECISION | YES | NIFTY100 TRIN value |
| `created_at` | TIMESTAMPTZ | NO | Record creation timestamp (default: NOW()) |

**Constraints**:
- PRIMARY KEY: `id`

**Indexes**:
- `idx_macro_signals_exchange` on `(exchange, timestamp DESC)`

**Hypertable**: Yes (TimescaleDB)

---

## Table 7: `order_book_depth_snapshots`

**Purpose**: Stores aggregated order book depth metrics.

**Fields**:

| Field Name | Data Type | Nullable | Description |
|------------|-----------|----------|-------------|
| `id` | SERIAL | NO | Primary key (auto-increment) |
| `timestamp` | TIMESTAMPTZ | NO | Snapshot timestamp |
| `exchange` | TEXT | NO | Exchange name |
| `depth_buy_total` | DOUBLE PRECISION | YES | Total buy depth |
| `depth_sell_total` | DOUBLE PRECISION | YES | Total sell depth |
| `depth_imbalance_ratio` | DOUBLE PRECISION | YES | Depth imbalance ratio |
| `source` | TEXT | YES | Data source identifier |
| `created_at` | TIMESTAMPTZ | NO | Record creation timestamp (default: NOW()) |

**Constraints**:
- PRIMARY KEY: `id`

**Indexes**:
- `idx_depth_snapshots_exchange` on `(exchange, timestamp DESC)`

**Hypertable**: Yes (TimescaleDB)

---

## Table 8: `paper_trading_metrics`

**Purpose**: Logs paper trading execution decisions and results.

**Fields**:

| Field Name | Data Type | Nullable | Description |
|------------|-----------|----------|-------------|
| `id` | SERIAL | NO | Primary key (auto-increment) |
| `timestamp` | TIMESTAMPTZ | NO | Signal timestamp |
| `exchange` | TEXT | NO | Exchange name |
| `executed` | BOOLEAN | NO | Whether trade was executed |
| `reason` | TEXT | YES | Execution reason or rejection reason |
| `signal` | TEXT | YES | Signal type (BUY/SELL) |
| `confidence` | DOUBLE PRECISION | YES | Model confidence (0-1) |
| `quantity_lots` | INTEGER | YES | Quantity in lots |
| `pnl` | DOUBLE PRECISION | YES | Profit/Loss (if closed) |
| `constraint_violation` | BOOLEAN | NO | Whether risk constraint was violated (default: FALSE) |
| `created_at` | TIMESTAMPTZ | NO | Record creation timestamp (default: NOW()) |

**Constraints**:
- PRIMARY KEY: `id`

**Indexes**:
- `idx_paper_trading_metrics_exchange_ts` on `(exchange, timestamp DESC)`

**Hypertable**: Yes (TimescaleDB)

---

## Table 9: `multi_resolution_bars`

**Purpose**: Stores OHLCV bars at multiple resolutions (1min, 5min, 15min, 1D).

**Fields**:

| Field Name | Data Type | Nullable | Description |
|------------|-----------|----------|-------------|
| `id` | SERIAL | NO | Primary key (auto-increment) |
| `timestamp` | TIMESTAMPTZ | NO | Bar timestamp |
| `exchange` | TEXT | NO | Exchange name |
| `resolution` | TEXT | NO | Bar resolution (1min, 5min, 15min, 1D) |
| `token` | INTEGER | NO | Instrument token |
| `symbol` | TEXT | YES | Instrument symbol |
| `open_price` | DOUBLE PRECISION | YES | Open price |
| `high_price` | DOUBLE PRECISION | YES | High price |
| `low_price` | DOUBLE PRECISION | YES | Low price |
| `close_price` | DOUBLE PRECISION | YES | Close price |
| `volume` | BIGINT | YES | Trading volume |
| `oi` | BIGINT | YES | Open interest (for options/futures) |
| `oi_change` | BIGINT | YES | Change in OI for the bar |
| `vwap` | DOUBLE PRECISION | YES | Volume-weighted average price |
| `trade_count` | INTEGER | YES | Number of trades in the bar |
| `spread_avg` | DOUBLE PRECISION | YES | Average bid-ask spread |
| `imbalance_avg` | DOUBLE PRECISION | YES | Average order book imbalance |
| `created_at` | TIMESTAMPTZ | NO | Record creation timestamp (default: NOW()) |

**Constraints**:
- PRIMARY KEY: `id`
- UNIQUE: `(timestamp, exchange, resolution, token)`

**Indexes**:
- `idx_multi_res_bars_resolution_time` on `(exchange, resolution, timestamp DESC)`
- `idx_multi_res_bars_token_time` on `(token, timestamp DESC)`

**Hypertable**: Yes (TimescaleDB)

---

## Database Initialization

The database schema is automatically initialized when `database_new.py` is imported. The initialization process:

1. **Creates all tables** (if they don't exist)
2. **Enables TimescaleDB extension** (if available)
3. **Creates hypertables** for time-series tables
4. **Creates indexes** for performance
5. **Runs migrations** to add any missing columns

To manually initialize:

```python
from database_new import initialize_database, migrate_database

# Initialize schema
initialize_database()

# Run migrations (adds missing columns)
migrate_database()
```

---

## Required Database Extensions

### TimescaleDB (Recommended)
- Enables efficient time-series storage and queries
- Auto-compression of old data
- Better performance for time-based queries

To install TimescaleDB:
```sql
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
```

---

## Database Configuration

Database connection settings are configured via environment variables (see `config.py`):

- `OI_TRACKER_DB_TYPE` - Database type (default: `postgres`)
- `OI_TRACKER_DB_HOST` - Database host (default: `localhost`)
- `OI_TRACKER_DB_PORT` - Database port (default: `5432`)
- `OI_TRACKER_DB_NAME` - Database name (default: `oi_db_new`)
- `OI_TRACKER_DB_USER` - Database user (default: `dilip`)
- `OI_TRACKER_DB_PASSWORD` - Database password

---

## Notes

1. **All timestamps use TIMESTAMPTZ (timestamp with time zone)** - This follows TimescaleDB best practices and ensures proper timezone handling
2. **TimescaleDB hypertables** are created automatically for time-series tables
3. **Indexes** are created for common query patterns
4. **Migrations** are run automatically to add missing columns
5. **PostgreSQL is required** - SQLite support has been removed

---

## Data Retention

Recommended retention periods (configurable via cleanup scripts):

- `option_chain_snapshots`: 90 days
- `ml_features`: 120 days
- `multi_resolution_bars`: 60 days
- `macro_signals`: 180 days
- `vix_term_structure`: 90 days
- `paper_trading_metrics`: Permanent (for analysis)
- `training_batches`: Permanent (for traceability)
- `exchange_metadata`: Permanent (current state only)

---

## Related Files

- `database_new.py` - Database schema and operations
- `config.py` - Database configuration
- `delete_all_records.py` - Cleanup script
- `delete_records.py` - Selective cleanup
- `export_to_csv.py` - Data export utilities
- `view_database.py` - Database viewer

