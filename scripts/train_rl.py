#!/usr/bin/env python3
"""
Reinforcement Learning Training Script for OI Gemini

Trains RL agents (PPO/DQN) for trading strategy and execution optimization.

Prerequisites:
    pip install stable-baselines3[extra] gymnasium

Usage:
    python scripts/train_rl.py --exchange NSE --algorithm PPO --timesteps 100000
    python scripts/train_rl.py --exchange NSE --algorithm DQN --timesteps 50000 --mode strategy
    python scripts/train_rl.py --exchange NSE --algorithm PPO --timesteps 20000 --mode execution
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

try:
    from stable_baselines3 import PPO, DQN
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("ERROR: stable-baselines3 not installed. Install with: pip install stable-baselines3[extra]")
    sys.exit(1)

from models.reinforcement_learning import TradingEnvironment, ExecutionEnvironment, RLAction
import database_new as db
from config import get_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)


def load_training_data(exchange: str, days: int = 90) -> pd.DataFrame:
    """Load historical features for RL training."""
    LOGGER.info(f"Loading {days} days of historical data for {exchange}...")
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Load features from database
    # This should match the feature engineering pipeline
    try:
        # Get features from multi_resolution_bars or option_chain_snapshots
        # For now, we'll use a simplified approach - you may need to adapt this
        # based on your actual feature storage
        
        # Example: Load from database
        # features_df = db.get_historical_features(exchange, start_date, end_date)
        
        # For demonstration, we'll create a placeholder
        # In production, replace this with actual database query
        LOGGER.warning("Using placeholder data loading. Implement actual database query.")
        
        # Placeholder: Create sample features
        # You should replace this with actual database query
        n_samples = days * 390  # ~390 minutes per trading day
        feature_cols = [
            'pcr_total_oi', 'futures_premium', 'vix', 'underlying_price',
            'atm_shift_intensity', 'put_call_iv_skew', 'net_gamma_exposure',
            # Add more features as needed
        ]
        
        # Generate synthetic data for demonstration
        # In production, load from database
        np.random.seed(42)
        data = {}
        for col in feature_cols:
            data[col] = np.random.randn(n_samples)
        
        # Add future_return for reward calculation
        data['future_return'] = np.random.randn(n_samples) * 0.01
        
        features_df = pd.DataFrame(data)
        features_df['timestamp'] = pd.date_range(start=start_date, periods=n_samples, freq='1min')
        
        LOGGER.info(f"Loaded {len(features_df)} samples with {len(feature_cols)} features")
        return features_df
        
    except Exception as e:
        LOGGER.error(f"Failed to load training data: {e}")
        raise


def train_strategy_agent(
    exchange: str,
    algorithm: str,
    timesteps: int,
    features_df: pd.DataFrame,
    output_dir: Path
):
    """Train RL agent for trading strategy."""
    LOGGER.info(f"Training {algorithm} strategy agent for {exchange}...")
    
    # Create environment factory function
    # Each call must create a NEW environment instance
    def make_env():
        return TradingEnvironment(
            exchange=exchange,
            features_df=features_df.copy(),  # Copy dataframe to avoid sharing state
            initial_capital=1_000_000.0
        )
    
    # Create vectorized environment for faster training
    vec_env = make_vec_env(make_env, n_envs=4)
    
    # Initialize model
    if algorithm.upper() == "PPO":
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            verbose=1,
            tensorboard_log=str(output_dir / "tensorboard")
        )
    elif algorithm.upper() == "DQN":
        model = DQN(
            "MlpPolicy",
            vec_env,
            learning_rate=1e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=32,
            gamma=0.99,
            target_update_interval=1000,
            exploration_fraction=0.1,
            exploration_final_eps=0.05,
            verbose=1,
            tensorboard_log=str(output_dir / "tensorboard")
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Use PPO or DQN")
    
    # Setup callbacks
    eval_callback = EvalCallback(
        vec_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=str(output_dir / "checkpoints"),
        name_prefix="rl_model"
    )
    
    # Train model
    LOGGER.info(f"Starting training for {timesteps} timesteps...")
    model.learn(
        total_timesteps=timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save final model
    model_path = output_dir / f"{algorithm.lower()}_strategy_{exchange.lower()}.zip"
    model.save(str(model_path))
    LOGGER.info(f"✓ Model saved to {model_path}")
    
    return model, model_path


def train_execution_agent(
    exchange: str,
    algorithm: str,
    timesteps: int,
    tick_data: list,
    output_dir: Path
):
    """Train RL agent for order execution optimization."""
    LOGGER.info(f"Training {algorithm} execution agent for {exchange}...")
    
    # Create environment
    env = ExecutionEnvironment(tick_data=tick_data, target_quantity=1)
    
    # Initialize model (PPO for continuous action space)
    if algorithm.upper() != "PPO":
        LOGGER.warning("Execution optimization requires PPO (continuous actions). Switching to PPO.")
        algorithm = "PPO"
    
    model = PPO(
        "MultiInputPolicy",  # For Dict action space
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log=str(output_dir / "tensorboard")
    )
    
    # Setup callbacks
    eval_callback = EvalCallback(
        env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=1000,
        deterministic=True
    )
    
    # Train model
    LOGGER.info(f"Starting training for {timesteps} timesteps...")
    model.learn(
        total_timesteps=timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    
    # Save final model
    model_path = output_dir / f"{algorithm.lower()}_execution_{exchange.lower()}.zip"
    model.save(str(model_path))
    LOGGER.info(f"✓ Model saved to {model_path}")
    
    return model, model_path


def main():
    parser = argparse.ArgumentParser(
        description="Train RL agents for trading strategy and execution",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--exchange', type=str, required=True, choices=['NSE', 'BSE'],
                       help='Exchange to train for')
    parser.add_argument('--algorithm', type=str, default='PPO', choices=['PPO', 'DQN'],
                       help='RL algorithm to use (default: PPO)')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Number of training timesteps (default: 100000)')
    parser.add_argument('--days', type=int, default=90,
                       help='Days of historical data to use (default: 90)')
    parser.add_argument('--mode', type=str, default='strategy', choices=['strategy', 'execution'],
                       help='Training mode: strategy or execution (default: strategy)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (default: models/{exchange}/rl/)')
    
    args = parser.parse_args()
    
    if not SB3_AVAILABLE:
        LOGGER.error("stable-baselines3 is required for RL training")
        return 1
    
    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(f"models/{args.exchange}/rl/")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    LOGGER.info(f"Output directory: {output_dir}")
    
    try:
        if args.mode == 'strategy':
            # Load training data
            features_df = load_training_data(args.exchange, args.days)
            
            # Train strategy agent
            model, model_path = train_strategy_agent(
                exchange=args.exchange,
                algorithm=args.algorithm,
                timesteps=args.timesteps,
                features_df=features_df,
                output_dir=output_dir
            )
            
            LOGGER.info(f"✓ Strategy training complete. Model: {model_path}")
            
        elif args.mode == 'execution':
            # For execution, we need tick-level order book data
            # This is a placeholder - implement actual tick data loading
            LOGGER.warning("Execution training requires tick-level order book data.")
            LOGGER.warning("Implement tick data loading from database or files.")
            
            # Placeholder tick data
            tick_data = [
                {'bid': 100.0, 'ask': 100.1, 'bid_size': 100, 'ask_size': 100}
                for _ in range(1000)
            ]
            
            # Train execution agent
            model, model_path = train_execution_agent(
                exchange=args.exchange,
                algorithm=args.algorithm,
                timesteps=args.timesteps,
                tick_data=tick_data,
                output_dir=output_dir
            )
            
            LOGGER.info(f"✓ Execution training complete. Model: {model_path}")
        
        LOGGER.info("="*80)
        LOGGER.info("Training Summary")
        LOGGER.info("="*80)
        LOGGER.info(f"Exchange: {args.exchange}")
        LOGGER.info(f"Algorithm: {args.algorithm}")
        LOGGER.info(f"Mode: {args.mode}")
        LOGGER.info(f"Timesteps: {args.timesteps}")
        LOGGER.info(f"Output: {output_dir}")
        LOGGER.info("="*80)
        
        return 0
        
    except Exception as e:
        LOGGER.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())

