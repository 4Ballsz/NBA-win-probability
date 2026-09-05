"""
Step 1 of 6 -- download raw NBA box scores.

Pulls every game from the 2019-20 through 2024-25 seasons and writes them to
nba_data.csv. Each game produces two rows, one from each team's perspective.

    python 01_get_data.py
"""

import time

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

import nba_features

# The NBA regular season and playoffs for each season we train on. The API is
# queried by date range rather than by season name so that playoff games are
# included alongside the regular season.
SEASON_DATES = {
    '2019-20': ('2019-10-22', '2020-10-11'),
    '2020-21': ('2020-12-22', '2021-07-20'),
    '2021-22': ('2021-10-19', '2022-06-16'),
    '2022-23': ('2022-10-18', '2023-06-12'),
    '2023-24': ('2023-10-24', '2024-06-17'),
    '2024-25': ('2024-10-22', '2025-06-15'),
}

# LeagueGameFinder covers every league NBA Stats tracks. '00' is the NBA itself;
# without it the results also include G League and WNBA games.
NBA_LEAGUE_ID = '00'

NUMERIC_COLUMNS = [
    'PTS', 'PLUS_MINUS', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
    'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF',
]


def fetch_season(season, attempts=3):
    """Download one season. The NBA API is flaky, so failures are retried."""
    start_date, end_date = SEASON_DATES[season]
    print(f"  {season} ({start_date} to {end_date})...", end=' ', flush=True)

    for attempt in range(1, attempts + 1):
        try:
            finder = leaguegamefinder.LeagueGameFinder(
                league_id_nullable=NBA_LEAGUE_ID,
                date_from_nullable=start_date,
                date_to_nullable=end_date,
            )
            games = finder.get_data_frames()[0]
            break
        except Exception as error:
            print(f"\n    attempt {attempt}/{attempts} failed: {error}")
            if attempt == attempts:
                return pd.DataFrame()
            time.sleep(2)

    if games.empty:
        print("no games returned")
        return pd.DataFrame()

    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
    games['SEASON'] = season

    for column in NUMERIC_COLUMNS:
        if column in games.columns:
            games[column] = pd.to_numeric(games[column], errors='coerce')

    print(f"{len(games)} rows")
    return games


def drop_exhibition_games(games):
    """
    Remove All-Star weekend games (Team LeBron vs. Team Giannis, USA vs.
    World, Rising Stars, etc).

    LeagueGameFinder has no flag for these -- they come back mixed in with
    real games, under made-up team codes that never appear in a normal
    season. Dropping anything outside the 30 real teams also drops any
    row where a team's own history is too short to be exhibition-only,
    since a fake team with a code like "TMG" never accumulates one.
    """
    real_games = games[games['TEAM_ABBREVIATION'].isin(nba_features.TEAMS)]

    dropped = len(games) - len(real_games)
    if dropped:
        print(f"  Dropped {dropped} exhibition-game rows (All-Star weekend, etc.)")

    return real_games


def collect_all_seasons():
    """Download every season listed in SEASON_DATES and stack them together."""
    print("Downloading NBA games")
    print("-" * 60)

    seasons = []
    for season in SEASON_DATES:
        games = fetch_season(season)
        if not games.empty:
            seasons.append(games)
        time.sleep(0.5)  # be polite to the API

    if not seasons:
        return pd.DataFrame()

    return drop_exhibition_games(pd.concat(seasons, ignore_index=True))


if __name__ == "__main__":
    all_games = collect_all_seasons()

    if all_games.empty:
        print("\nNo data collected. Check your network connection and try again.")
        raise SystemExit(1)

    all_games.to_csv('nba_data.csv', index=False)

    print("-" * 60)
    print(f"Rows:    {len(all_games)}  (2 per game, one per team)")
    print(f"Teams:   {all_games['TEAM_ABBREVIATION'].nunique()}")
    print(f"Dates:   {all_games['GAME_DATE'].min().date()} to {all_games['GAME_DATE'].max().date()}")
    print("Saved to nba_data.csv")
