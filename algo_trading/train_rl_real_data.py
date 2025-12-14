#!/usr/bin/env python3
"""
Train RL PPO model on real EURUSD data from Polygon/Massive API.

Training configuration:
- Account: $100,000
- Max Daily Drawdown: $5,000
- Timesteps: 1,000,000
- Data: Real EURUSD 1h from Polygon API
"""
import sys
import warnings
import logging
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd


def main():
    from data.downloaders.forex_downloader import ForexDownloader
    from strategies.rl_ppo.training import RLTrainer
    from strategies.rl_ppo.environment import ForexTradingEnv

    print('=' * 70)
    print('     PPO FOREX TRADING AGENT - TRAINING ON REAL DATA')
    print('     Account: $100,000 | Max Daily DD: $5,000 | 1M Steps')
    print('=' * 70)
    print()

    # Initialize downloader with Polygon API
    logger.info("Initializing Polygon API downloader...")
    downloader = ForexDownloader(source='polygon')

    if not downloader.polygon_api_key:
        raise ValueError("Polygon API key not found!")

    # Download training data (2021-2023)
    logger.info("Downloading training data (2021-01-01 to 2023-06-30)...")
    train_df = downloader.download(
        symbol='EURUSD',
        timeframe='1h',
        start_date='2021-01-01',
        end_date='2023-06-30'
    )
    logger.info(f"Training data: {len(train_df)} bars")

    # Download validation data (2023-07 to 2023-12)
    logger.info("Downloading validation data (2023-07-01 to 2023-12-31)...")
    val_df = downloader.download(
        symbol='EURUSD',
        timeframe='1h',
        start_date='2023-07-01',
        end_date='2023-12-31'
    )
    logger.info(f"Validation data: {len(val_df)} bars")

    # Download test data (2024)
    logger.info("Downloading test data (2024-01-01 to 2024-12-01)...")
    test_df = downloader.download(
        symbol='EURUSD',
        timeframe='1h',
        start_date='2024-01-01',
        end_date='2024-12-01'
    )
    logger.info(f"Test data: {len(test_df)} bars")

    # Save data for future use
    data_dir = Path('data/storage/historical')
    data_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(data_dir / 'EURUSD_1h_train.parquet')
    val_df.to_parquet(data_dir / 'EURUSD_1h_val.parquet')
    test_df.to_parquet(data_dir / 'EURUSD_1h_test.parquet')
    logger.info(f"Data saved to {data_dir}")

    print()
    print('Data Summary:')
    print(f'  Training:   {len(train_df):,} bars ({train_df.index[0].date()} to {train_df.index[-1].date()})')
    print(f'  Validation: {len(val_df):,} bars ({val_df.index[0].date()} to {val_df.index[-1].date()})')
    print(f'  Test:       {len(test_df):,} bars ({test_df.index[0].date()} to {test_df.index[-1].date()})')
    print()

    # Initialize trainer
    trainer = RLTrainer(
        model_dir='models/rl_real',
        log_dir='logs/rl_real'
    )

    # Train with 1M timesteps
    print('=' * 70)
    print('Starting PPO Training (1,000,000 timesteps)...')
    print('This will take approximately 30-60 minutes.')
    print('=' * 70)
    print()

    start_time = datetime.now()

    model = trainer.train(
        train_df=train_df,
        val_df=val_df,
        total_timesteps=1_000_000,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        filter_trading_hours=False,
        early_stopping=False  # Train for full 1M steps
    )

    training_time = datetime.now() - start_time
    print()
    print(f'Training completed in {training_time}')
    print()

    # Evaluate on test data
    print('=' * 70)
    print('EVALUATING ON UNSEEN TEST DATA (2024)')
    print('=' * 70)
    print()

    print('Running 10 evaluation episodes...')
    eval_results = trainer.evaluate(model, test_df, n_episodes=10)

    print()
    print('EVALUATION RESULTS (Test Data - 2024):')
    print('-' * 50)
    for key, value in eval_results.items():
        print(f'  {key}: {value:.4f}')

    # Run detailed analysis
    print()
    print('=' * 70)
    print('DETAILED EPISODE ANALYSIS')
    print('=' * 70)
    print()

    env = ForexTradingEnv(test_df)
    obs, _ = env.reset()
    done = False
    actions = []
    rewards = []
    balances = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        actions.append(action)
        rewards.append(reward)
        balances.append(info['balance'])
        done = terminated or truncated

    print(f'Episode Summary:')
    print(f'  Total Steps: {len(actions)}')
    print(f'  Total Reward: {sum(rewards):.4f}')
    print(f'  Number of Trades: {info["num_trades"]}')
    print(f'  Total Profit: ${info["total_profit"]:,.2f}')
    print(f'  Final Balance: ${info["balance"]:,.2f}')
    print(f'  Return: {(info["balance"] - 100000) / 100000 * 100:.2f}%')
    print(f'  Account Blown: {info.get("account_blown", False)}')

    # Action distribution
    unique, counts = np.unique(actions, return_counts=True)
    action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
    print()
    print('Action Distribution:')
    for u, c in zip(unique, counts):
        pct = c / len(actions) * 100
        print(f'  {action_names.get(u, u)}: {c} ({pct:.1f}%)')

    # Compare with random baseline
    print()
    print('=' * 70)
    print('COMPARISON: RL AGENT vs RANDOM AGENT')
    print('=' * 70)

    rl_profits = []
    random_profits = []

    for _ in range(10):
        # RL Agent
        env = ForexTradingEnv(test_df)
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        rl_profits.append(info['total_profit'])

        # Random Agent
        env = ForexTradingEnv(test_df)
        obs, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        random_profits.append(info['total_profit'])

    print()
    print(f'                        RL Agent       Random Agent')
    print(f'  --------------------------------------------------------')
    print(f'  Mean Profit:      ${np.mean(rl_profits):10,.2f}     ${np.mean(random_profits):10,.2f}')
    print(f'  Std Profit:       ${np.std(rl_profits):10,.2f}     ${np.std(random_profits):10,.2f}')
    print(f'  Best Profit:      ${max(rl_profits):10,.2f}     ${max(random_profits):10,.2f}')
    print(f'  Worst Profit:     ${min(rl_profits):10,.2f}     ${min(random_profits):10,.2f}')

    improvement = np.mean(rl_profits) - np.mean(random_profits)
    print()
    print(f'  RL Agent Improvement: ${improvement:,.2f}')
    print()

    print('=' * 70)
    print(f'Model saved to: models/rl_real/ppo_forex_final.zip')
    print(f'Training time: {training_time}')
    print('=' * 70)


if __name__ == '__main__':
    main()
