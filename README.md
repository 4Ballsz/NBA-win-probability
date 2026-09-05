# NBA Game Predictor

Predicts the winner of an NBA game from how both teams have been playing over
their last five games, using an XGBoost classifier trained on six seasons of
box scores (2019–2025).

```bash
cd nba_predictor
pip install -r requirements.txt
python 01_get_data.py          # downloads 2019-2025 from the NBA Stats API
python 02_prepare_features.py
python 03_train_model.py
python 05_interactive_ui.py    # or 04_predict.py for the terminal version
```

See [`nba_predictor/README.md`](nba_predictor/README.md) for the full pipeline
walkthrough, the feature set, and how to read the backtest.

## How it works

Each matchup is described by 24 features comparing the two teams' recent
form — scoring, shooting efficiency, rebounding, turnovers, rest days — plus a
home-court flag. The model is validated with time-series cross-validation and
backtested on a full season it never trained on.

Two details do most of the work:

- **Home advantage.** NBA home teams win roughly 58% of games. Any model has
  to beat that number to be worth anything, which is why the backtest reports
  it right next to the model's own accuracy.
- **No peeking.** A team's form is only ever computed from games played
  *strictly before* the game being predicted, so nothing about the outcome
  leaks into the features.

## Data source

Game data comes from the public NBA Stats API through the
[`nba_api`](https://github.com/swar/nba_api) package. No API key is needed,
but the endpoints rate-limit and time out often, so the scripts retry and
pause between calls.

## Requirements

Python 3.9+, plus the packages in
[`nba_predictor/requirements.txt`](nba_predictor/requirements.txt).

## Known limitations

- **No injury data.** A star sitting out is invisible to the model, and it is
  probably the single biggest source of wrong predictions.
- **Form only.** There is no notion of team quality beyond the last 5 games,
  so a good team on a cold streak is underrated and vice versa.
- **No travel or altitude.** Rest days are counted, but not distance flown.
- **Season boundaries.** A team's "last 5 games" can reach back into the
  previous season for games played in October.

## License

[MIT](LICENSE)
