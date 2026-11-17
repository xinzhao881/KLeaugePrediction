"""Predict K League matches with the saved goal model (no retraining).

Usage
  python predict.py                      # every unfinished scheduled match
  python predict.py 울산 서울             # any matchup: HOME AWAY
  python predict.py 울산 서울 전북 포항    # several matchups (pairs)
  python predict.py 울산 서울 --threshold 0.1

The model is trained and saved by feature.py (models/goal_model.joblib).
For a custom matchup, each team's features are its CURRENT pre-match state:
rolling mean of its last ROLL_WINDOW finished matches and the mean of its
finished matches in the latest season, i.e. the same values pruning.py would
give the team's next fixture. Only data from finished matches is used.
"""

import argparse
import re

import numpy as np
import pandas as pd

import feature as f
import pruning as pr

RESULT_KR = {"H": "홈 승", "D": "무승부", "A": "원정 승"}
FEATURE_RE = re.compile(r"^(roll(\d+)|season)_(att_|def_|str_)?(.+)$")


def load_bundle():
    if not f.MODEL_PATH.exists():
        raise SystemExit(f"[ERROR] {f.MODEL_PATH} not found: run `python feature.py` first")
    return f.load_model()


def load_team_matches():
    files = sorted(f.PROCESSED_DIR.glob("*/team_match_stats.csv"))
    if not files:
        raise SystemExit("[ERROR] team_match_stats.csv not found: run `python pruning.py` first")
    tm = pd.concat([pd.read_csv(p, encoding="utf-8-sig") for p in files], ignore_index=True)
    tm["kst_date"] = pd.to_datetime(tm["kst_date"])
    tm["finished"] = tm["finished"].astype(str).str.lower() == "true"
    return tm.sort_values(["team", "kst_date", "kst_time"]).reset_index(drop=True)


def team_state(tm, team, cols):
    """Current pre-match feature values of one team, keyed without the side prefix."""
    season = tm["season"].max()  # latest season in the data
    played = tm[(tm["team"] == team) & tm["finished"]]
    if played.empty:
        raise SystemExit(f"[ERROR] no finished matches for {team}")
    this_season = played[played["season"] == season]
    if this_season.empty:
        print(f"[WARNING] {team} has no {season} matches: season_* features are missing "
              f"(filled with the training median)")

    state = {}
    for name in sorted({c.split("_", 1)[1] for c in cols}):
        if name == "season_matches_played":
            state[name] = float(len(this_season))
            continue
        m = FEATURE_RE.match(name)
        stat = m.group(4) if m else None
        if stat not in played:
            state[name] = np.nan  # e.g. schedule features of a hypothetical match
            continue
        if m.group(2):  # rolling window across seasons, like prior_state(window=N)
            state[name] = played[stat].tail(int(m.group(2))).mean()
        else:
            state[name] = this_season[stat].mean()
    return state, played["kst_date"].max()


def predict_matchups(bundle, pairs, threshold):
    model, tm = bundle["model"], load_team_matches()
    rows = []
    for home, away in pairs:
        home, away = pr.normalize_team_name(home), pr.normalize_team_name(away)
        if home == away:
            raise SystemExit(f"[ERROR] same team on both sides: {home}")
        hs, h_last = team_state(tm, home, model.cols)
        as_, a_last = team_state(tm, away, model.cols)
        row = {c: (hs if c.startswith("home_") else as_)[c.split("_", 1)[1]]
               for c in model.cols}
        rows.append({"home_team": home, "away_team": away, "home_data_through": h_last,
                     "away_data_through": a_last, **row})
    X = pd.DataFrame(rows)
    h, a = model.predict_goals(X)
    probs = f.poisson_outcome_probs(h, a)
    out = X[["home_team", "away_team"]].copy()
    out["pred_home_goals"], out["pred_away_goals"] = h, a
    out["pred_goal_diff"] = h - a
    out["pred"] = f.classify(h - a, threshold)
    out["P_win"], out["P_draw"], out["P_loss"] = probs[:, 0], probs[:, 1], probs[:, 2]
    out["data_through"] = [f"{max(x, y):%Y-%m-%d}" for x, y in
                           zip(X["home_data_through"], X["away_data_through"])]
    return out


def print_matchups(out, threshold):
    for r in out.itertuples():
        print(f"\n{r.home_team} (홈) vs {r.away_team} (원정)   [data through {r.data_through}]")
        print(f"  예상 득점   {r.pred_home_goals:.2f} : {r.pred_away_goals:.2f}  "
              f"(득실차 {r.pred_goal_diff:+.2f}, 무승부 기준 ±{threshold:.2f})")
        print(f"  예측 결과   {RESULT_KR[r.pred]}")
        print(f"  확률(참고)  홈 승 {r.P_win:.1%} / 무 {r.P_draw:.1%} / 원정 승 {r.P_loss:.1%}")


def main():
    ap = argparse.ArgumentParser(description="Predict K League matches with the saved model")
    ap.add_argument("teams", nargs="*", help="HOME AWAY [HOME AWAY ...]")
    ap.add_argument("--threshold", type=float, default=None,
                    help="draw band on |predicted goal diff| (default: tuned value)")
    args = ap.parse_args()

    bundle = load_bundle()
    threshold = bundle["threshold"] if args.threshold is None else args.threshold
    print(f"[Model] features={bundle['feature_set']} ({len(bundle['model'].cols)}), "
          f"alpha={bundle['model'].alpha:.1f}, threshold={threshold:.2f}, "
          f"trained through {bundle['trained_through']} ({bundle['n_matches']} matches, "
          f"test accuracy {bundle['test_accuracy']:.1%})")

    df = f.load_match_features()
    latest = df.loc[df["finished"], "kst_date"].max()
    if latest > pd.Timestamp(bundle["trained_through"]):
        print(f"[WARNING] data has matches through {latest:%Y-%m-%d}; "
              f"run `python feature.py` to retrain")

    if args.teams:
        if len(args.teams) % 2:
            raise SystemExit("[ERROR] give teams in pairs: HOME AWAY [HOME AWAY ...]")
        pairs = list(zip(args.teams[::2], args.teams[1::2]))
        print_matchups(predict_matchups(bundle, pairs, threshold), threshold)
        return

    upcoming = df[~df["finished"]]
    if upcoming.empty:
        print("No unfinished matches in the data. Give a matchup, e.g. `python predict.py 울산 서울`")
        return
    out = f.prediction_frame(upcoming, bundle["model"], threshold)
    out["pred"] = out["pred"].map(RESULT_KR)
    pd.set_option("display.width", 200)
    print(f"\n[Upcoming] {len(out)} matches")
    print(out.drop(columns=["season"]).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
