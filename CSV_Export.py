import csv

def export_schedule_to_csv(schedule, team_names, filename="tournament_schedule.csv"):
    """
    schedule: dict[int -> list[tuple(away_team_id, home_team_id)]], 0-based rounds
    team_names: list[str] indexed by team id
    filename: output CSV path
    """
    with open(filename, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Round", "Away_ID", "Away_Name", "Home_ID", "Home_Name", "Match"])
        for r in sorted(schedule.keys()):
            for away_id, home_id in schedule[r]:
                w.writerow([
                    r + 1,
                    away_id,
                    team_names[away_id],
                    home_id,
                    team_names[home_id],
                    f"{team_names[away_id]} @ {team_names[home_id]}"
                ])
    print(f"Saved schedule to: {filename}")
