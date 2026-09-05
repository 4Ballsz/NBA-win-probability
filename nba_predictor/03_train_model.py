"""
Step 3 of 6 -- train the XGBoost classifier.

Reads X_features.npy / y_target.npy and writes nba_model.json.

Validation uses TimeSeriesSplit rather than a random split: the rows are in
chronological order, so a random split would let the model train on future
games and report an accuracy it could never reach in practice.

    python 03_train_model.py
"""

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit

import nba_features
from nba_features import MODEL_PARAMS

N_SPLITS = 5


def cross_validate(X, y, n_splits=N_SPLITS):
    """Score the model on rolling time-ordered folds. Returns (accuracies, brier scores)."""
    print(f"\n{n_splits}-fold time series cross-validation")
    print("-" * 60)

    accuracies = []
    brier_scores = []

    for fold, (train_idx, val_idx) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(X), 1):
        model = xgb.XGBClassifier(**MODEL_PARAMS)
        model.fit(X[train_idx], y[train_idx])

        probabilities = model.predict_proba(X[val_idx])[:, 1]
        predictions = (probabilities > 0.5).astype(int)

        accuracy = accuracy_score(y[val_idx], predictions)
        brier = brier_score_loss(y[val_idx], probabilities)

        # Predicting "home team wins" every time is the bar any model must clear.
        home_baseline = y[val_idx].mean()

        accuracies.append(accuracy)
        brier_scores.append(brier)

        print(f"  fold {fold}: train={len(train_idx):>6}  val={len(val_idx):>5}  "
              f"accuracy={accuracy:.3f}  (always-home={home_baseline:.3f})  brier={brier:.3f}")

    return accuracies, brier_scores


def show_feature_importance(model, top_n=10):
    """List the features the model leaned on most."""
    print(f"\nTop {top_n} features")
    print("-" * 60)

    importance = model.feature_importances_
    for rank, index in enumerate(np.argsort(importance)[::-1][:top_n], 1):
        print(f"  {rank:>2}. {nba_features.FEATURE_NAMES[index]:<20} {importance[index]:.4f}")


if __name__ == "__main__":
    print("Training model")
    print("-" * 60)

    try:
        X = np.load('X_features.npy')
        y = np.load('y_target.npy')
    except FileNotFoundError:
        print("Feature files not found. Run 02_prepare_features.py first.")
        raise SystemExit(1)

    print(f"  Samples:       {X.shape[0]}")
    print(f"  Features:      {X.shape[1]}")
    print(f"  Home win rate: {y.mean():.1%}")

    if X.shape[1] != len(nba_features.FEATURE_NAMES):
        print(f"\nExpected {len(nba_features.FEATURE_NAMES)} features but found {X.shape[1]}.")
        print("Re-run 02_prepare_features.py so the data matches nba_features.py.")
        raise SystemExit(1)

    accuracies, brier_scores = cross_validate(X, y)

    print("-" * 60)
    print(f"  Accuracy: {np.mean(accuracies):.3f} +/- {np.std(accuracies):.3f}")
    print(f"  Brier:    {np.mean(brier_scores):.3f} +/- {np.std(brier_scores):.3f}   (lower is better)")

    print("\nFitting final model on all data...")
    final_model = xgb.XGBClassifier(**MODEL_PARAMS)
    final_model.fit(X, y)

    show_feature_importance(final_model)

    final_model.save_model('nba_model.json')
    print("\nSaved nba_model.json")
