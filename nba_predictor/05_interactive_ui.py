"""
Step 5 of 6 -- the same predictor as step 4, with a desktop window.

Pick two teams and a date, press Predict, and the model's reasoning appears
alongside the real result when the game is in the dataset.

    python 05_interactive_ui.py
"""

import tkinter as tk
from datetime import datetime
from tkinter import messagebox, ttk

import nba_features

BACKGROUND = '#1a1a2e'
PANEL = '#16213e'
TEXT_AREA = '#0f3460'
ACCENT = '#e94560'
SUCCESS = '#2ecc71'
FAILURE = '#e74c3c'
WORKING = '#f39c12'

TEAM_CHOICES = [f"{code} - {name}" for code, name in sorted(nba_features.TEAMS.items())]


class PredictorWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("NBA Game Predictor")
        self.root.geometry("950x800")
        self.root.configure(bg=BACKGROUND)

        self.model = None
        self.games = None
        self.team_index = None

        self._build_widgets()
        self._load_resources()

    # -- setup ------------------------------------------------------------

    def _build_widgets(self):
        tk.Label(self.root, text="NBA GAME PREDICTOR", font=('Arial', 20, 'bold'),
                 bg=BACKGROUND, fg=ACCENT).pack(pady=(20, 5))

        self.status = tk.Label(self.root, text="Loading...", font=('Arial', 10),
                               bg=BACKGROUND, fg='white')
        self.status.pack(pady=(0, 10))

        main = tk.Frame(self.root, bg=BACKGROUND)
        main.pack(fill='both', expand=True, padx=30, pady=10)

        teams = tk.LabelFrame(main, text="Teams", font=('Arial', 12, 'bold'),
                              bg=PANEL, fg='white', relief='raised', bd=2)
        teams.pack(fill='x', pady=10)

        self.home_var = tk.StringVar()
        self.away_var = tk.StringVar()

        for row, (label, variable) in enumerate([("Home Team:", self.home_var),
                                                 ("Away Team:", self.away_var)]):
            tk.Label(teams, text=label, font=('Arial', 11), bg=PANEL, fg='white').grid(
                row=row, column=0, padx=10, pady=10, sticky='w')
            ttk.Combobox(teams, textvariable=variable, values=TEAM_CHOICES,
                         width=35, state='readonly').grid(row=row, column=1, padx=10, pady=10)

        tk.Button(teams, text="Swap", command=self.swap_teams, bg=ACCENT, fg='white',
                  font=('Arial', 10), width=10).grid(row=0, column=2, rowspan=2, padx=20)

        dates = tk.LabelFrame(main, text="Game Date", font=('Arial', 12, 'bold'),
                              bg=PANEL, fg='white', relief='raised', bd=2)
        dates.pack(fill='x', pady=10)

        tk.Label(dates, text="YYYY-MM-DD:", font=('Arial', 11), bg=PANEL, fg='white').pack(
            side='left', padx=10, pady=10)

        self.date_entry = tk.Entry(dates, font=('Arial', 11), width=15)
        self.date_entry.pack(side='left', padx=10, pady=10)
        self.date_entry.insert(0, datetime.now().strftime('%Y-%m-%d'))

        self.predict_button = tk.Button(main, text="MAKE PREDICTION", command=self.make_prediction,
                                        bg=SUCCESS, fg='white', font=('Arial', 14, 'bold'),
                                        height=2, state='disabled')
        self.predict_button.pack(fill='x', pady=20)

        results = tk.LabelFrame(main, text="Result", font=('Arial', 12, 'bold'),
                                bg=PANEL, fg='white', relief='raised', bd=2)
        results.pack(fill='both', expand=True, pady=10)

        # The scrollbar is a sibling of the text box, not a child of it, so that
        # it sits beside the text rather than floating on top of the content.
        scrollbar = tk.Scrollbar(results)
        scrollbar.pack(side='right', fill='y', pady=10)

        self.output = tk.Text(results, bg=TEXT_AREA, fg='white', font=('Consolas', 11),
                              wrap='word', height=18, yscrollcommand=scrollbar.set)
        self.output.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        scrollbar.config(command=self.output.yview)

        self._write("Select two teams and a date, then press MAKE PREDICTION.")

    def _load_resources(self):
        self.root.update()
        try:
            self.model = nba_features.load_model()
            self.games = nba_features.load_games()
            self.team_index = nba_features.index_by_team(self.games)
        except Exception as error:
            self.status.config(text="Could not load nba_model.json or nba_data.csv", fg=FAILURE)
            self._write("Run 01_get_data.py through 03_train_model.py first, "
                        f"then reopen this window.\n\nDetails:\n{error}")
            return

        self.status.config(text="Ready", fg=SUCCESS)
        self.predict_button.config(state='normal')

    # -- helpers ----------------------------------------------------------

    def _write(self, text):
        self.output.config(state='normal')
        self.output.delete('1.0', tk.END)
        self.output.insert('1.0', text)
        self.output.config(state='disabled')

    def swap_teams(self):
        home, away = self.home_var.get(), self.away_var.get()
        self.home_var.set(away)
        self.away_var.set(home)

    # -- prediction -------------------------------------------------------

    def make_prediction(self):
        home_choice, away_choice = self.home_var.get(), self.away_var.get()
        date_str = self.date_entry.get().strip()

        if not home_choice or not away_choice:
            messagebox.showwarning("Missing team", "Please select both teams.")
            return
        if home_choice == away_choice:
            messagebox.showwarning("Same team", "Please select two different teams.")
            return

        try:
            game_date = datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            messagebox.showerror("Invalid date", "Use YYYY-MM-DD, for example 2024-11-01.")
            return

        home_team = home_choice.split(' - ')[0]
        away_team = away_choice.split(' - ')[0]

        self.status.config(text="Predicting...", fg=WORKING)
        self.root.update()

        prediction = nba_features.predict_matchup(
            self.model, self.team_index, home_team, away_team, game_date)
        actual = nba_features.actual_result(self.games, home_team, away_team, game_date)

        self._write(self._format_report(prediction, actual, date_str))
        self.status.config(text="Ready", fg=SUCCESS)

    def _format_report(self, prediction, actual, date_str):
        confidence = prediction['confidence']

        lines = [
            f"{prediction['away_name']} @ {prediction['home_name']}",
            date_str,
            "=" * 52,
            "",
            "LAST 5 GAMES",
            f"  {prediction['home_name']:<24} {prediction['home_record']}  "
            f"{prediction['home_ppg']} PPG  {prediction['home_rest']}d rest",
            f"  {prediction['away_name']:<24} {prediction['away_record']}  "
            f"{prediction['away_ppg']} PPG  {prediction['away_rest']}d rest",
            "",
            "MODEL PREDICTION",
            f"  {prediction['home_name']:<24} {prediction['home_prob']:.1%}",
            f"  {prediction['away_name']:<24} {prediction['away_prob']:.1%}",
            "",
            f"  Winner:     {prediction['predicted_winner_name']}",
            f"  Confidence: {nba_features.confidence_label(confidence)} ({confidence:.0%})",
            "",
            "ACTUAL RESULT",
        ]

        if actual is None:
            lines.append("  Not in the dataset -- a future game, or outside 2019-2025.")
        else:
            verdict = "correct" if actual['winner'] == prediction['predicted_winner'] else "wrong"
            winner_name = nba_features.TEAMS.get(actual['winner'], actual['winner'])
            lines += [
                f"  Winner:     {winner_name}",
                f"  Score:      {actual['home_score']} - {actual['away_score']} "
                f"(home - away, margin {actual['margin']})",
                f"  Model was:  {verdict}",
            ]

        return "\n".join(lines)


if __name__ == "__main__":
    root = tk.Tk()
    PredictorWindow(root)
    root.mainloop()
