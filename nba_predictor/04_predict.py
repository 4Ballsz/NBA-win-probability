"""
Step 4 of 6 -- predict a single game from the command line.

Asks for two teams and a date, prints the model's probabilities, and -- if the
game is already in the data -- shows what actually happened.

    python 04_predict.py
"""

from datetime import datetime

import nba_features


def print_prediction(prediction, actual, date_str):
    """Print one prediction, and the real result when we have it."""
    print()
    print("=" * 55)
    print(f"{prediction['away_name']} @ {prediction['home_name']}")
    print(f"{date_str}")
    print("=" * 55)

    print("\nLast 5 games")
    print(f"  {prediction['home_name']:<24} {prediction['home_record']}  "
          f"{prediction['home_ppg']} PPG  {prediction['home_rest']}d rest")
    print(f"  {prediction['away_name']:<24} {prediction['away_record']}  "
          f"{prediction['away_ppg']} PPG  {prediction['away_rest']}d rest")

    print("\nModel prediction")
    print(f"  {prediction['home_name']:<24} {prediction['home_prob']:.1%}")
    print(f"  {prediction['away_name']:<24} {prediction['away_prob']:.1%}")

    confidence = prediction['confidence']
    print(f"\n  Winner:     {prediction['predicted_winner_name']}")
    print(f"  Confidence: {nba_features.confidence_label(confidence)} ({confidence:.0%})")

    print("\nActual result")
    if actual is None:
        print("  Not in the dataset -- either a future game or outside 2019-2025.")
        return

    verdict = "correct" if actual['winner'] == prediction['predicted_winner'] else "wrong"
    print(f"  Winner:     {nba_features.TEAMS.get(actual['winner'], actual['winner'])}")
    print(f"  Score:      {actual['home_score']} - {actual['away_score']} "
          f"(home - away, margin {actual['margin']})")
    print(f"  Model was:  {verdict}")


def ask_for_team(role):
    """Prompt until the user gives a real team code."""
    while True:
        code = input(f"{role} team code: ").upper().strip()
        if code in nba_features.TEAMS:
            return code
        print(f"  '{code}' is not an NBA team code. Try again.")


if __name__ == "__main__":
    try:
        model = nba_features.load_model()
        games = nba_features.load_games()
    except Exception as error:
        # XGBoost raises its own error type for a missing model file, so this
        # catches both that and a missing nba_data.csv.
        print(f"Could not load model or data: {error}")
        print("Run steps 1-3 first.")
        raise SystemExit(1)

    team_index = nba_features.index_by_team(games)

    print(f"Loaded {games['GAME_ID'].nunique()} games, "
          f"{games['GAME_DATE'].min().date()} to {games['GAME_DATE'].max().date()}")
    print("\nTeams:", ', '.join(sorted(nba_features.TEAMS)))
    print()

    home_team = ask_for_team("Home")
    away_team = ask_for_team("Away")
    date_str = input("Game date (YYYY-MM-DD): ").strip()

    try:
        game_date = datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        print("Date must look like 2024-11-01.")
        raise SystemExit(1)

    prediction = nba_features.predict_matchup(model, team_index, home_team, away_team, game_date)
    actual = nba_features.actual_result(games, home_team, away_team, game_date)

    print_prediction(prediction, actual, date_str)
