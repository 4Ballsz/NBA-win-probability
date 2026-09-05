"""
Shared building blocks for the NBA game predictor.

Every script in this folder imports from here so that the feature vector is
defined exactly once. Training, prediction, the UI and the backtest must all
describe a matchup the same way -- if they drift apart, the model is fed
numbers at prediction time that mean something different from what it learned.
"""

import numpy as np
import pandas as pd
import xgboost as xgb

# How many recent games summarize a team's current form.
RECENT_GAMES = 5

# A team needs at least this many played games before we trust its form.
MIN_HISTORY = 3

# Rest is capped so that the gap between seasons (or a long injury layoff)
# doesn't enter the model as a 120-day outlier that dwarfs every other feature.
MAX_REST_DAYS = 7

TEAMS = {
    'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BKN': 'Brooklyn Nets',
    'CHA': 'Charlotte Hornets', 'CHI': 'Chicago Bulls', 'CLE': 'Cleveland Cavaliers',
    'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets', 'DET': 'Detroit Pistons',
    'GSW': 'Golden State Warriors', 'HOU': 'Houston Rockets', 'IND': 'Indiana Pacers',
    'LAC': 'LA Clippers', 'LAL': 'Los Angeles Lakers', 'MEM': 'Memphis Grizzlies',
    'MIA': 'Miami Heat', 'MIL': 'Milwaukee Bucks', 'MIN': 'Minnesota Timberwolves',
    'NOP': 'New Orleans Pelicans', 'NYK': 'New York Knicks', 'OKC': 'Oklahoma City Thunder',
    'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers', 'PHX': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs',
    'TOR': 'Toronto Raptors', 'UTA': 'Utah Jazz', 'WAS': 'Washington Wizards',
}

# Order matters: it must match build_feature_vector() element for element.
FEATURE_NAMES = [
    'pts_diff', 'efg_diff', 'reb_diff', 'wins_diff', 'rest_diff',
    'home_advantage', 'home_win_pct', 'away_win_pct', 'home_pts', 'away_pts',
    'home_efg', 'away_efg', 'home_stl', 'away_stl', 'home_blk', 'away_blk',
    'home_ast', 'away_ast', 'home_tov', 'away_tov', 'home_plus_minus',
    'away_plus_minus', 'home_back_to_back', 'away_back_to_back',
]

# Shared so that the model trained in step 3 and the model the backtest fits in
# step 6 are the same model, and a tuning change only has to be made once.
MODEL_PARAMS = {
    'n_estimators': 300,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
}

# League-average-ish placeholders, used only when a team has too little history.
LEAGUE_AVERAGE_FORM = {
    'pts_avg': 112.0, 'efg_pct': 0.535, 'reb_avg': 44.0, 'ast_avg': 25.0,
    'tov_avg': 14.0, 'stl_avg': 7.5, 'blk_avg': 5.0, 'wins': 2,
    'win_pct': 0.5, 'plus_minus_avg': 0.0,
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_games(path='nba_data.csv'):
    """Load the CSV written by 01_get_data.py, normalized for the rest of the pipeline."""
    games = pd.read_csv(path)
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])

    if 'TEAM_ABBREVIATION' in games.columns:
        games = games.rename(columns={'TEAM_ABBREVIATION': 'TEAM'})

    return games.sort_values('GAME_DATE').reset_index(drop=True)


def load_model(path='nba_model.json'):
    """Load a trained XGBoost classifier."""
    model = xgb.XGBClassifier()
    model.load_model(path)
    return model


def index_by_team(games):
    """Split games into one date-sorted frame per team, for fast history lookups."""
    return {
        team: frame.sort_values('GAME_DATE').reset_index(drop=True)
        for team, frame in games.groupby('TEAM')
    }


# ---------------------------------------------------------------------------
# Reading a team's recent form
# ---------------------------------------------------------------------------

def recent_games_before(team_frame, game_date, count=RECENT_GAMES):
    """
    The team's most recent `count` games played strictly before `game_date`,
    ordered newest first.

    `team_frame` must be sorted by GAME_DATE ascending (index_by_team does this),
    which lets us binary-search the cutoff instead of scanning every row.
    """
    if team_frame is None or team_frame.empty:
        return pd.DataFrame()

    cutoff = int(team_frame['GAME_DATE'].searchsorted(game_date, side='left'))
    window = team_frame.iloc[max(0, cutoff - count):cutoff]
    return window.iloc[::-1]


def summarize_form(recent):
    """
    Condense a team's recent games into the numbers the model cares about.

    `recent` comes from recent_games_before(), so it is already trimmed to the
    right window and ordered newest first.
    """
    if len(recent) < MIN_HISTORY:
        return dict(LEAGUE_AVERAGE_FORM)

    field_goals_made = recent['FGM'].sum()
    field_goals_attempted = recent['FGA'].sum()
    threes_made = recent['FG3M'].sum()
    wins = int((recent['WL'] == 'W').sum())

    return {
        'pts_avg': recent['PTS'].mean(),
        # Effective FG% credits a three as 1.5 field goals, since it is worth 1.5x.
        'efg_pct': ((field_goals_made + 0.5 * threes_made) / field_goals_attempted
                    if field_goals_attempted > 0 else LEAGUE_AVERAGE_FORM['efg_pct']),
        'reb_avg': recent['REB'].mean(),
        'ast_avg': recent['AST'].mean(),
        'tov_avg': recent['TOV'].mean(),
        'stl_avg': recent['STL'].mean(),
        'blk_avg': recent['BLK'].mean(),
        'wins': wins,
        'win_pct': wins / len(recent),
        'plus_minus_avg': recent['PLUS_MINUS'].mean(),
    }


def days_of_rest(recent, game_date):
    """Days between the team's previous game and this one, capped at MAX_REST_DAYS."""
    if len(recent) == 0:
        return 2

    gap = (game_date - recent.iloc[0]['GAME_DATE']).days
    return int(min(max(gap, 0), MAX_REST_DAYS))


# ---------------------------------------------------------------------------
# The feature vector
# ---------------------------------------------------------------------------

def build_feature_vector(home_form, away_form, home_rest, away_rest):
    """
    Describe one matchup as the 24 numbers the model is trained on.

    This is the single definition of the feature vector. Keep it in step with
    FEATURE_NAMES -- the model matches features by position, not by name, so a
    reordering here silently corrupts every prediction.
    """
    return [
        home_form['pts_avg'] - away_form['pts_avg'],
        home_form['efg_pct'] - away_form['efg_pct'],
        home_form['reb_avg'] - away_form['reb_avg'],
        home_form['wins'] - away_form['wins'],
        home_rest - away_rest,
        1,  # home_advantage: constant, but lets the model learn a home baseline
        home_form['win_pct'],
        away_form['win_pct'],
        home_form['pts_avg'],
        away_form['pts_avg'],
        home_form['efg_pct'],
        away_form['efg_pct'],
        home_form['stl_avg'],
        away_form['stl_avg'],
        home_form['blk_avg'],
        away_form['blk_avg'],
        home_form['ast_avg'],
        away_form['ast_avg'],
        home_form['tov_avg'],
        away_form['tov_avg'],
        home_form['plus_minus_avg'],
        away_form['plus_minus_avg'],
        1 if home_rest == 0 else 0,
        1 if away_rest == 0 else 0,
    ]


def features_for_matchup(team_index, home_team, away_team, game_date):
    """
    Build the feature vector for one matchup, plus the form summaries behind it.

    Only games played strictly before `game_date` are used, so this is safe to
    call on a full dataset without leaking the result of the game itself.
    """
    home_recent = recent_games_before(team_index.get(home_team), game_date)
    away_recent = recent_games_before(team_index.get(away_team), game_date)

    home_form = summarize_form(home_recent)
    away_form = summarize_form(away_recent)

    home_rest = days_of_rest(home_recent, game_date)
    away_rest = days_of_rest(away_recent, game_date)

    features = build_feature_vector(home_form, away_form, home_rest, away_rest)

    context = {
        'home_form': home_form,
        'away_form': away_form,
        'home_rest': home_rest,
        'away_rest': away_rest,
        'home_played': len(home_recent),
        'away_played': len(away_recent),
    }
    return features, context


# ---------------------------------------------------------------------------
# Reading the game log
# ---------------------------------------------------------------------------

def split_home_away(pair):
    """
    Given the two rows the API returns for one game, say which is the home team.

    MATCHUP reads "LAL vs. BOS" for the home team and "BOS @ LAL" for the away
    team, so the separator alone identifies the side.
    """
    home = pair[pair['MATCHUP'].str.contains('vs.', regex=False)]
    away = pair[pair['MATCHUP'].str.contains('@', regex=False)]

    if len(home) != 1 or len(away) != 1:
        return None, None

    return home.iloc[0], away.iloc[0]


# ---------------------------------------------------------------------------
# Predicting, and checking the prediction
# ---------------------------------------------------------------------------

def predict_matchup(model, team_index, home_team, away_team, game_date):
    """
    Run one matchup through the model.

    Returns the win probabilities plus the recent-form numbers behind them, so
    a caller can explain the prediction instead of only reporting it.
    """
    game_date = pd.to_datetime(game_date)
    features, context = features_for_matchup(team_index, home_team, away_team, game_date)

    home_prob = float(model.predict_proba(np.array([features], dtype=np.float32))[0, 1])
    home_form, away_form = context['home_form'], context['away_form']
    winner = home_team if home_prob > 0.5 else away_team

    return {
        'home_team': home_team,
        'away_team': away_team,
        'home_name': TEAMS.get(home_team, home_team),
        'away_name': TEAMS.get(away_team, away_team),
        'home_prob': home_prob,
        'away_prob': 1 - home_prob,
        'predicted_winner': winner,
        'predicted_winner_name': TEAMS.get(winner, winner),
        # How far from a coin flip the model is, scaled to 0-100%.
        'confidence': abs(home_prob - 0.5) * 2,
        'home_record': f"{home_form['wins']}-{context['home_played'] - home_form['wins']}",
        'away_record': f"{away_form['wins']}-{context['away_played'] - away_form['wins']}",
        'home_ppg': round(home_form['pts_avg'], 1),
        'away_ppg': round(away_form['pts_avg'], 1),
        'home_rest': context['home_rest'],
        'away_rest': context['away_rest'],
    }


def confidence_label(confidence):
    """Bucket a confidence score into a word."""
    if confidence > 0.6:
        return 'HIGH'
    return 'MEDIUM' if confidence > 0.3 else 'LOW'


def actual_result(games, home_team, away_team, game_date):
    """
    Look up how a game really ended, or None if it isn't in the data.

    The winner is decided by comparing the two scores rather than by reading a
    WL column, so it does not matter which team's row we happen to find first.
    """
    game_date = pd.to_datetime(game_date)
    same_day = games[games['GAME_DATE'] == game_date]

    home_row = same_day[same_day['TEAM'] == home_team]
    away_row = same_day[same_day['TEAM'] == away_team]

    if home_row.empty or away_row.empty:
        return None

    home_score = int(home_row.iloc[0]['PTS'])
    away_score = int(away_row.iloc[0]['PTS'])

    return {
        'winner': home_team if home_score > away_score else away_team,
        'home_score': home_score,
        'away_score': away_score,
        'margin': abs(home_score - away_score),
    }


# ---------------------------------------------------------------------------
# Turning raw games into a training set
# ---------------------------------------------------------------------------

def build_training_dataset(games, verbose=True):
    """
    Turn raw game rows into (X, y) where y is 1 when the home team won.

    Each game appears twice in the raw data (once per team), so we group by
    GAME_ID and handle each matchup once.
    """
    team_index = index_by_team(games)

    feature_rows = []
    labels = []
    skipped = 0

    for _, pair in games.groupby('GAME_ID'):
        if len(pair) != 2:
            skipped += 1
            continue

        home_row, away_row = split_home_away(pair)
        if home_row is None:
            skipped += 1
            continue

        game_date = home_row['GAME_DATE']
        features, context = features_for_matchup(
            team_index, home_row['TEAM'], away_row['TEAM'], game_date
        )

        # Early-season games have no form to speak of; they would just add noise.
        if context['home_played'] < MIN_HISTORY or context['away_played'] < MIN_HISTORY:
            skipped += 1
            continue

        feature_rows.append(features)
        labels.append(1 if home_row['PTS'] > away_row['PTS'] else 0)

    X = np.array(feature_rows, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    if verbose:
        print(f"  Usable games:  {len(X)}")
        print(f"  Skipped:       {skipped} (not enough history, or incomplete record)")
        if len(y):
            print(f"  Home win rate: {y.mean():.1%}")
            if not 0.50 <= y.mean() <= 0.65:
                print(f"  WARNING: {y.mean():.1%} is well off the ~58% NBA home win rate. "
                      f"Home and away may be swapped somewhere.")

    return X, y
