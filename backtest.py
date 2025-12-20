#!/usr/bin/env python3
"""
Run backtest of the Momentum + Fundamentals Strategy.

Usage:
    python3 backtest.py

This will:
1. Load historical data from EODHD (cached)
2. Train a CatBoost model on 2010-2024 data
3. Backtest on 2025 data
4. Save the trained model for use by get_recommendations.py
"""

from run_strategy import main

if __name__ == '__main__':
    results = main()
    print("\nBacktest complete! Model saved for recommendations.")
