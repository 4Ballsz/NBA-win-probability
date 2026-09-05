# NBA Game Predictor — pipeline

Predicts NBA game winners with XGBoost, using each team's last 5 games.

## Setup

```bash
pip install -r requirements.txt
```

## Run in order

Each script writes a file the next one reads, so run them top to bottom the
first time. After that you can jump straight to step 4, 5 or 6.

| Step | Script | Reads | Writes |
|---|---|---|---|
| 1 | `01_get_data.py` | NBA API | `nba_data.csv` |
| 2 | `02_prepare_features.py` | `nba_data.csv` | `X_features.npy`, `y_target.npy` |
| 3 | `03_train_model.py` | the `.npy` files | `nba_model.json` |
| 4 | `04_predict.py` | model + data | prints one prediction |
| 5 | `05_interactive_ui.py` | model + data | desktop window |
| 6 | `06_season_backtest.py` | `nba_data.csv` | `backtest_results_2024_25.csv` |

```bash
python 01_get_data.py          # downloads 2019-2025, takes a couple of minutes
python 02_prepare_features.py
python 03_train_model.py
python 05_interactive_ui.py    # or 04_predict.py for the terminal version
```

`nba_features.py` is not a step. It holds the pieces every script shares — the
team list, the feature vector, and the code that reads a team's recent form.

## Why there is a shared module

The model matches features **by position**, not by name. If step 2 puts
rebound differential in slot 3 and step 4 puts assists there instead, nothing
crashes — the model simply reads assists as though they were rebounds and the
predictions quietly become nonsense.

So the feature vector is written down exactly once, in
`build_feature_vector()`. Every script calls it. If you add a feature, add it
there and to `FEATURE_NAMES` in the same edit, then re-run steps 2 and 3.

## The 24 features

Six are differences between the two teams, and the rest are each team's raw
form so the model can judge *how* two similar-looking teams differ.

| Group | Features |
|---|---|
| Differences | points, effective FG%, rebounds, wins in last 5, rest days |
| Context | home advantage (always 1) |
| Per team | win %, points, eFG%, steals, blocks, assists, turnovers, plus/minus |
| Schedule | back-to-back flag for each team |

**Effective field goal percentage** (`efg_pct`) is used instead of raw FG%
because it credits a three-pointer as worth 1.5 two-pointers, which is what it
is actually worth on the scoreboard.

## Reading the backtest

`06_season_backtest.py` is the honest scorecard. Two things make it
trustworthy, and both are easy to get wrong:

1. **It retrains the model** on games before 2024-10-22 rather than loading
   `nba_model.json`, which was fitted on every season *including* 2024-25.
   Scoring a model on data it trained on is grading its own homework.

2. **Team form still comes from the full dataset**, because features only read
   games played strictly before the game being predicted. Restricting form to
   pre-season data instead would freeze every team's stats at the previous June.

Read the output in this order:

- **Accuracy vs. "always pick home"** — home teams win ~58% of NBA games. If
  the model is not clearly above that line, it has learned nothing useful.
- **By confidence** — the model should be right more often on the games it is
  confident about. If HIGH and LOW accuracy are the same, its confidence is
  meaningless.
- **Calibration** — when the model says 70%, the home team should win about 70%
  of those games. A model can be accurate and still badly overconfident.

## Files the pipeline generates

`nba_data.csv`, `X_features.npy`, `y_target.npy`, `nba_model.json` and
`backtest_results_2024_25.csv` are all rebuildable from the scripts and are
gitignored.
