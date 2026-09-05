"""
Step 2 of 6 -- turn raw box scores into a training set.

Reads nba_data.csv and writes X_features.npy / y_target.npy, where each row
describes one matchup and the label is 1 if the home team won.

The feature vector itself lives in nba_features.py so that this script, the
predictor, the UI and the backtest all agree on what a matchup looks like.

    python 02_prepare_features.py
"""

import numpy as np

import nba_features


if __name__ == "__main__":
    print("Preparing features")
    print("-" * 60)

    try:
        games = nba_features.load_games('nba_data.csv')
    except FileNotFoundError:
        print("nba_data.csv not found. Run 01_get_data.py first.")
        raise SystemExit(1)

    print(f"  Loaded {len(games)} rows covering {games['GAME_ID'].nunique()} games")

    X, y = nba_features.build_training_dataset(games)

    if len(X) == 0:
        print("\nNo training rows produced. Check nba_data.csv.")
        raise SystemExit(1)

    np.save('X_features.npy', X)
    np.save('y_target.npy', y)

    print("-" * 60)
    print(f"Saved X_features.npy {X.shape} and y_target.npy {y.shape}")
