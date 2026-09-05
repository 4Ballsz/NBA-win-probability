"""
Step 6 of 6 -- backtest the 2024-25 season.

Answers the only question that matters: if we had used this model all season,
how often would it have been right?

Two rules keep the answer honest:

  1. The model is retrained here on games played BEFORE 2024-10-22. The
     nba_model.json from step 3 was fitted on every season including 2024-25,
     so scoring 2024-25 with it would be grading the model on its own homework.

  2. Team form still comes from the full dataset, because features only ever
     read games played strictly before the game being predicted. Restricting
     form to pre-season data instead would freeze every team's stats at last
     June and quietly wreck the predictions.

    python 06_season_backtest.py
"""

from datetime import datetime

import pandas as pd
import xgboost as xgb

import nba_features

SEASON_START = datetime(2024, 10, 22)
SEASON_END = datetime(2025, 6, 15)

RESULTS_CSV = 'backtest_results_2024_25.csv'


def train_on_prior_seasons(games):
    """Fit a model using only games played before the test season."""
    prior = games[games['GAME_DATE'] < SEASON_START]
    print(f"  Training rows: {len(prior)} ({prior['GAME_DATE'].min().date()} "
          f"to {prior['GAME_DATE'].max().date()})")

    X, y = nba_features.build_training_dataset(prior)

    model = xgb.XGBClassifier(**nba_features.MODEL_PARAMS)
    model.fit(X, y)
    return model


def predict_season(model, games):
    """Predict every game of the test season. Returns one row per game."""
    team_index = nba_features.index_by_team(games)

    season = games[(games['GAME_DATE'] >= SEASON_START) & (games['GAME_DATE'] <= SEASON_END)]

    rows = []
    for _, pair in season.groupby('GAME_ID'):
        if len(pair) != 2:
            continue

        home_row, away_row = nba_features.split_home_away(pair)
        if home_row is None:
            continue

        home, away = home_row['TEAM'], away_row['TEAM']
        game_date = home_row['GAME_DATE']

        prediction = nba_features.predict_matchup(model, team_index, home, away, game_date)
        home_won = home_row['PTS'] > away_row['PTS']
        actual_winner = home if home_won else away

        rows.append({
            'date': game_date,
            'home_team': home,
            'away_team': away,
            'home_prob': prediction['home_prob'],
            'confidence': prediction['confidence'],
            'predicted_winner': prediction['predicted_winner'],
            'actual_winner': actual_winner,
            'home_won': int(home_won),
            'correct': prediction['predicted_winner'] == actual_winner,
            'home_score': int(home_row['PTS']),
            'away_score': int(away_row['PTS']),
            'margin': abs(int(home_row['PTS']) - int(away_row['PTS'])),
        })

    return pd.DataFrame(rows)


def report(results):
    """Print accuracy, a breakdown by confidence, and a calibration table."""
    print("\nOverall")
    print("-" * 60)
    accuracy = results['correct'].mean()

    # The bar to beat: home teams win roughly 58% of NBA games, so a model that
    # blindly picks the home team already scores about that.
    home_baseline = results['home_won'].mean()

    print(f"  Games:              {len(results)}")
    print(f"  Correct:            {results['correct'].sum()}")
    print(f"  Accuracy:           {accuracy:.1%}")
    print(f"  Always pick home:   {home_baseline:.1%}")
    print(f"  Edge over baseline: {accuracy - home_baseline:+.1%}")

    print("\nBy confidence")
    print("-" * 60)
    bands = [
        ('HIGH   (>60%)', results['confidence'] > 0.6),
        ('MEDIUM (30-60%)', results['confidence'].between(0.3, 0.6)),
        ('LOW    (<30%)', results['confidence'] < 0.3),
    ]
    for label, mask in bands:
        subset = results[mask]
        if len(subset):
            print(f"  {label:<16} {subset['correct'].sum():>4}/{len(subset):<5} "
                  f"{subset['correct'].mean():.1%}")

    print("\nCalibration -- when the model says X%, how often does the home team win?")
    print("-" * 60)
    for low in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7]:
        high = low + 0.1 if low >= 0.3 else 0.3
        band = results[(results['home_prob'] >= low) & (results['home_prob'] < high)]
        if len(band):
            print(f"  predicted {low:.0%}-{high:.0%}:  actual {band['home_won'].mean():>5.1%}  "
                  f"(n={len(band)})")

    # A well-calibrated model's actual rate tracks its predicted rate. Big gaps
    # mean the probabilities are overconfident even if accuracy looks fine.


if __name__ == "__main__":
    print("Backtesting the 2024-25 season")
    print("-" * 60)

    try:
        games = nba_features.load_games()
    except FileNotFoundError:
        print("nba_data.csv not found. Run 01_get_data.py first.")
        raise SystemExit(1)

    model = train_on_prior_seasons(games)
    results = predict_season(model, games)

    if results.empty:
        print("\nNo 2024-25 games found in nba_data.csv.")
        raise SystemExit(1)

    report(results)

    results.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved {RESULTS_CSV}")
