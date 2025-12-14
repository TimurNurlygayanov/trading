#!/usr/bin/env python3
"""
Train Hybrid Transformer + RL Trading System.

This script:
1. Trains a Transformer model for price prediction
2. Trains an RL agent using transformer predictions + mean reversion signals
3. Evaluates the hybrid system with dynamic exit strategy

Key innovation: No fixed R:R ratio - the model learns when to exit
based on predicted profitability and market conditions.
"""
import sys
import warnings
import logging
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch


def main():
    from data.downloaders.forex_downloader import ForexDownloader
    from strategies.transformer.model import PriceTransformer, TransformerTrainer
    from strategies.transformer.hybrid_env import TransformerRLEnv
    from strategies.rl_ppo.environment_enhanced import EnhancedForexEnv

    print('=' * 75)
    print('     HYBRID TRANSFORMER + RL TRADING SYSTEM')
    print('     Dynamic Exit Strategy - No Fixed R:R Ratio')
    print('=' * 75)
    print()

    # ===== LOAD DATA =====
    logger.info("Loading real EURUSD data from Polygon API...")

    downloader = ForexDownloader(source='polygon')

    try:
        train_df = pd.read_parquet('data/storage/historical/EURUSD_1h_train.parquet')
        val_df = pd.read_parquet('data/storage/historical/EURUSD_1h_val.parquet')
        test_df = pd.read_parquet('data/storage/historical/EURUSD_1h_test.parquet')
        logger.info("Loaded data from cache")
    except:
        logger.info("Downloading fresh data...")
        train_df = downloader.download('EURUSD', '1h', '2021-01-01', '2023-06-30')
        val_df = downloader.download('EURUSD', '1h', '2023-07-01', '2023-12-31')
        test_df = downloader.download('EURUSD', '1h', '2024-01-01', '2024-12-01')

    print()
    print(f'Training data:   {len(train_df):,} bars')
    print(f'Validation data: {len(val_df):,} bars')
    print(f'Test data:       {len(test_df):,} bars')
    print()

    # ===== PHASE 1: TRAIN TRANSFORMER =====
    print('=' * 75)
    print('PHASE 1: Training Transformer Price Predictor')
    print('=' * 75)
    print()

    # Create transformer model
    transformer = PriceTransformer(
        input_dim=1,
        d_model=64,
        nhead=4,
        num_encoder_layers=3,
        dim_feedforward=256,
        dropout=0.1,
        prediction_horizon=12,
        seq_length=60
    )

    trainer = TransformerTrainer(transformer, learning_rate=1e-4)

    # Get price arrays
    train_prices = train_df['close'].values
    val_prices = val_df['close'].values

    logger.info("Training transformer (100 epochs)...")
    start_time = datetime.now()

    history = trainer.train(
        train_prices=train_prices,
        val_prices=val_prices,
        epochs=100,
        batch_size=64,
        early_stopping_patience=15
    )

    transformer_time = datetime.now() - start_time
    print(f'\nTransformer training completed in {transformer_time}')

    # Save transformer
    transformer_path = Path('models/transformer')
    transformer_path.mkdir(parents=True, exist_ok=True)
    trainer.save(str(transformer_path / 'price_predictor.pt'))
    logger.info(f"Transformer saved to {transformer_path}")

    # Evaluate transformer predictions
    print()
    print('Transformer Prediction Quality:')
    transformer.eval()
    test_prices = test_df['close'].values

    # Get predictions on test data
    with torch.no_grad():
        test_predictions = transformer.predict_returns(test_prices[-100:])

    # Calculate prediction accuracy (direction)
    actual_returns = np.diff(test_prices[-100:]) / test_prices[-100:-1]
    if len(actual_returns) >= len(test_predictions[0]):
        pred_direction = test_predictions[0][:len(actual_returns)] > 0
        actual_direction = actual_returns[:len(pred_direction)] > 0
        direction_accuracy = (pred_direction == actual_direction).mean()
        print(f'  Direction Accuracy: {direction_accuracy:.1%}')

    # ===== PHASE 2: TRAIN ENHANCED RL (Mean Reversion Signals) =====
    print()
    print('=' * 75)
    print('PHASE 2: Training Enhanced RL with Mean Reversion Signals')
    print('=' * 75)
    print()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from stable_baselines3.common.callbacks import EvalCallback
        from stable_baselines3.common.monitor import Monitor
    except ImportError:
        raise ImportError("stable-baselines3 required. Run: pip install stable-baselines3")

    # Create directories
    model_dir = Path('models/hybrid')
    log_dir = Path('logs/hybrid')
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create environments
    def make_enhanced_env(df):
        def _init():
            env = EnhancedForexEnv(
                df,
                transformer_model=None,  # First train without transformer
                initial_balance=100_000,
                max_daily_drawdown=5_000,
                risk_per_trade=2_000,
                window_size=60,
                use_dynamic_exit=True
            )
            return Monitor(env, str(log_dir))
        return _init

    logger.info("Creating vectorized environments...")
    train_env = DummyVecEnv([make_enhanced_env(train_df)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = DummyVecEnv([make_enhanced_env(val_df)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(log_dir),
        eval_freq=10000,
        n_eval_episodes=3,
        deterministic=True,
        verbose=1
    )

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        verbose=1,
        tensorboard_log=str(log_dir)
    )

    logger.info("Training Enhanced RL agent (500,000 timesteps)...")
    start_time = datetime.now()

    model.learn(
        total_timesteps=500_000,
        callback=eval_callback,
        progress_bar=True
    )

    rl_time = datetime.now() - start_time
    print(f'\nEnhanced RL training completed in {rl_time}')

    # Save model
    model.save(str(model_dir / 'enhanced_rl_mr'))
    train_env.save(str(model_dir / 'enhanced_vec_normalize.pkl'))

    # ===== PHASE 3: TRAIN HYBRID TRANSFORMER+RL =====
    print()
    print('=' * 75)
    print('PHASE 3: Training Hybrid Transformer + RL System')
    print('=' * 75)
    print()

    # Create hybrid environment with transformer
    def make_hybrid_env(df, transformer_model):
        def _init():
            env = TransformerRLEnv(
                df,
                transformer_model=transformer_model,
                initial_balance=100_000,
                max_daily_drawdown=5_000,
                max_position_risk=3_000,
                window_size=60,
                prediction_horizon=12,
                max_hold_time=100
            )
            return Monitor(env, str(log_dir / 'hybrid'))
        return _init

    (log_dir / 'hybrid').mkdir(exist_ok=True)

    logger.info("Creating hybrid environments with transformer...")
    hybrid_train_env = DummyVecEnv([make_hybrid_env(train_df, transformer)])
    hybrid_train_env = VecNormalize(hybrid_train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    hybrid_eval_env = DummyVecEnv([make_hybrid_env(val_df, transformer)])
    hybrid_eval_env = VecNormalize(hybrid_eval_env, norm_obs=True, norm_reward=False, training=False)

    # Hybrid evaluation callback
    hybrid_eval_callback = EvalCallback(
        hybrid_eval_env,
        best_model_save_path=str(model_dir / 'hybrid_best'),
        log_path=str(log_dir / 'hybrid'),
        eval_freq=10000,
        n_eval_episodes=3,
        deterministic=True,
        verbose=1
    )

    (model_dir / 'hybrid_best').mkdir(exist_ok=True)

    # Create hybrid model
    hybrid_model = PPO(
        "MlpPolicy",
        hybrid_train_env,
        learning_rate=2e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.02,  # More exploration for complex action space
        verbose=1,
        tensorboard_log=str(log_dir / 'hybrid')
    )

    logger.info("Training Hybrid model (500,000 timesteps)...")
    start_time = datetime.now()

    hybrid_model.learn(
        total_timesteps=500_000,
        callback=hybrid_eval_callback,
        progress_bar=True
    )

    hybrid_time = datetime.now() - start_time
    print(f'\nHybrid training completed in {hybrid_time}')

    # Save hybrid model
    hybrid_model.save(str(model_dir / 'hybrid_transformer_rl'))
    hybrid_train_env.save(str(model_dir / 'hybrid_vec_normalize.pkl'))

    # ===== EVALUATION =====
    print()
    print('=' * 75)
    print('EVALUATION ON TEST DATA (2024)')
    print('=' * 75)
    print()

    # Evaluate Enhanced RL (Mean Reversion signals only)
    print('1. Enhanced RL (Mean Reversion Signals):')
    enhanced_env = EnhancedForexEnv(test_df, initial_balance=100_000)
    enhanced_results = evaluate_model(model, enhanced_env, n_episodes=5)
    print_results(enhanced_results)

    # Evaluate Hybrid (Transformer + RL)
    print()
    print('2. Hybrid Transformer + RL:')
    hybrid_env = TransformerRLEnv(test_df, transformer_model=transformer, initial_balance=100_000)
    hybrid_results = evaluate_model(hybrid_model, hybrid_env, n_episodes=5)
    print_results(hybrid_results)

    # Compare with random
    print()
    print('3. Random Baseline:')
    random_results = evaluate_random(hybrid_env, n_episodes=5)
    print_results(random_results)

    # Summary
    print()
    print('=' * 75)
    print('COMPARISON SUMMARY')
    print('=' * 75)
    print()
    print(f'{"Strategy":<30} {"Mean Profit":>15} {"Win Rate":>12} {"Avg Trade":>12}')
    print('-' * 75)
    print(f'{"Enhanced RL (MR Signals)":<30} ${enhanced_results["mean_profit"]:>14,.2f} {enhanced_results["win_rate"]:>11.1%} ${enhanced_results["avg_trade"]:>11,.2f}')
    print(f'{"Hybrid Transformer+RL":<30} ${hybrid_results["mean_profit"]:>14,.2f} {hybrid_results["win_rate"]:>11.1%} ${hybrid_results["avg_trade"]:>11,.2f}')
    print(f'{"Random Baseline":<30} ${random_results["mean_profit"]:>14,.2f} {random_results["win_rate"]:>11.1%} ${random_results["avg_trade"]:>11,.2f}')

    print()
    print('=' * 75)
    print('Training Complete!')
    print(f'  Transformer:  {transformer_time}')
    print(f'  Enhanced RL:  {rl_time}')
    print(f'  Hybrid:       {hybrid_time}')
    print()
    print('Models saved to: models/hybrid/')
    print('=' * 75)


def evaluate_model(model, env, n_episodes=5):
    """Evaluate a trained model."""
    profits = []
    win_rates = []
    avg_trades = []
    all_trades = 0

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            # Convert action to scalar if it's an array
            if hasattr(action, '__len__'):
                action = int(action[0]) if len(action) > 0 else int(action)
            else:
                action = int(action)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        profits.append(info['total_profit'])
        win_rates.append(info.get('win_rate', 0))
        if info['num_trades'] > 0:
            avg_trades.append(info['total_profit'] / info['num_trades'])
        all_trades += info['num_trades']

    return {
        'mean_profit': np.mean(profits),
        'std_profit': np.std(profits),
        'win_rate': np.mean(win_rates) if win_rates else 0,
        'avg_trade': np.mean(avg_trades) if avg_trades else 0,
        'total_trades': all_trades // n_episodes
    }


def evaluate_random(env, n_episodes=5):
    """Evaluate random baseline."""
    profits = []
    win_rates = []
    avg_trades = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        profits.append(info['total_profit'])
        win_rates.append(info.get('win_rate', 0))
        if info['num_trades'] > 0:
            avg_trades.append(info['total_profit'] / info['num_trades'])

    return {
        'mean_profit': np.mean(profits),
        'std_profit': np.std(profits),
        'win_rate': np.mean(win_rates) if win_rates else 0,
        'avg_trade': np.mean(avg_trades) if avg_trades else 0,
        'total_trades': 0
    }


def print_results(results):
    """Print evaluation results."""
    print(f'  Mean Profit:  ${results["mean_profit"]:,.2f} (+/- ${results["std_profit"]:,.2f})')
    print(f'  Win Rate:     {results["win_rate"]:.1%}')
    print(f'  Avg Trade:    ${results["avg_trade"]:,.2f}')
    print(f'  Trades/Ep:    {results["total_trades"]}')


if __name__ == '__main__':
    main()
