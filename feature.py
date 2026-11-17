"""Predict home/away goals, then classify W/D/L with a goal-difference threshold.

Input : kleague_csv/processed/{year}/match_features.csv (built by pruning.py;
        every feature is a pre-match rolling / season-to-date mean)
Split : chronological TRAIN_RATIO / rest of finished matches (no shuffling)

Model
  1. Multiple linear regression (Ridge) predicts home goals and away goals.
  2. pred_goal_diff = pred_home_goals - pred_away_goals
       |pred_goal_diff| <  threshold -> D (draw)
        pred_goal_diff  >= threshold -> H (home win)
        pred_goal_diff  <= -threshold -> A (away win)
  The threshold is tuned on OUT-OF-FOLD train predictions (TimeSeriesSplit),
  so it is not fitted to in-sample predictions and never sees the test period.
  A fixed threshold of 0.5 (the reference paper) is reported for comparison.

Feature sets
  core : form + xG / chance-quality stats that were stable over time
  all  : every home_/away_ feature in match_features.csv

The final setup (feature set, alpha, threshold) is chosen on the train period,
refit on all finished matches, and predicts the upcoming matches.
P_win / P_draw / P_loss (independent Poisson on the predicted goals) are
saved as supplementary columns for the betting backtest.

Outputs
  feature_importance/model_metrics.csv          test metrics per setup
  feature_importance/model_coefficients.csv     standardized coefficients
  feature_importance/permutation_importance.csv final model, goal-diff MSE
  predictions/test_predictions.csv              test-period predictions
  predictions/upcoming_predictions.csv          unfinished matches
  models/goal_model.joblib                      final model for predict.py
"""

import re

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import pruning as pr

PROCESSED_DIR = pr.BASE_DIR / "kleague_csv" / "processed"
IMPORTANCE_DIR = pr.BASE_DIR / "feature_importance"
PRED_DIR = pr.BASE_DIR / "predictions"
MODEL_DIR = pr.BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "goal_model.joblib"  # used by predict.py

TRAIN_RATIO = 0.7
CV_SPLITS = 5
RIDGE_ALPHAS = np.logspace(-1, 4, 26)
THRESHOLDS = np.round(np.arange(0.0, 1.01, 0.01), 2)
PAPER_THRESHOLD = 0.5
MAX_GOALS = 10            # Poisson grid size
MIN_LAMBDA = 0.05         # floor for predicted goals
PERM_REPEATS = 30
RESULTS = ["H", "D", "A"]  # home win, draw, away win

# Stats (team-match column names) in the core set; FotMob only, so 2026
# (no K League portal data yet) is fully covered.
CORE_STATS = [
    "goals_for", "goals_against", "points", "xg_diff",
    "fm_expected_goals_open_play",
    "opp_fm_expected_goals_non_penalty",
    "opp_fm_expected_goals_set_play",
    "opp_fm_big_chance",
    "opp_fm_shotsontarget",
    "opp_fm_shots_inside_box",
    "opp_fm_clearances",
]
CORE_CONTEXT = ["season_matches_played"]


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_match_features():
    files = sorted(PROCESSED_DIR.glob("*/match_features.csv"))
    if not files:
        raise FileNotFoundError("match_features.csv not found, run pruning.py first")
    df = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in files], ignore_index=True)
    df["kst_date"] = pd.to_datetime(df["kst_date"])
    df["finished"] = df["finished"].astype(str).str.lower() == "true"
    return df.sort_values(["kst_date", "kst_time", "match_id"]).reset_index(drop=True)


def feature_sets(df):
    """Linear models: home_/away_ columns only (diff_ is a linear combination)."""
    side = [c for c in df.columns if c.startswith(("home_", "away_"))
            and pd.api.types.is_numeric_dtype(df[c])
            and c not in ("home_team_id", "away_team_id", "home_goals", "away_goals")]
    stat_re = "|".join(re.escape(s) for s in CORE_STATS)
    core_re = re.compile(rf"^(home|away)_(roll\d+|season)_(att_|def_|str_)?({stat_re})$")
    core = [c for c in side if core_re.match(c)]
    core += [f"{s}_{c}" for c in CORE_CONTEXT for s in ("home", "away") if f"{s}_{c}" in df]
    found = {core_re.match(c).group(4) for c in core if core_re.match(c)}
    missing = sorted(set(CORE_STATS) - found)
    if missing:
        print(f"[WARNING] core stats not in match_features.csv: {missing}")
    return {"core": core, "all": side}


def chronological_split(df):
    done = df[df["finished"]]
    n_train = int(len(done) * TRAIN_RATIO)
    return done.iloc[:n_train], done.iloc[n_train:], df[~df["finished"]]


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class GoalModel:
    """Two Ridge regressions: home goals and away goals."""

    def __init__(self, alpha, cols):
        self.alpha, self.cols = alpha, cols

    def _pipe(self):
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             Ridge(alpha=self.alpha))

    def fit(self, d):
        self.home_ = self._pipe().fit(d[self.cols], d["home_goals"])
        self.away_ = self._pipe().fit(d[self.cols], d["away_goals"])
        return self

    def predict_goals(self, d):
        return (np.clip(self.home_.predict(d[self.cols]), MIN_LAMBDA, None),
                np.clip(self.away_.predict(d[self.cols]), MIN_LAMBDA, None))

    def coefficients(self):
        return pd.DataFrame({"feature": self.cols,
                             "coef_home_goals": self.home_[-1].coef_,
                             "coef_away_goals": self.away_[-1].coef_})


def load_model(path=None):
    """Load the bundle saved by main(); bundle['model'] is a fitted GoalModel."""
    bundle = joblib.load(path or MODEL_PATH)
    model = GoalModel(bundle["alpha"], bundle["cols"])
    model.home_, model.away_ = bundle["home_pipeline"], bundle["away_pipeline"]
    bundle["model"] = model
    return bundle


def classify(goal_diff, threshold):
    d = np.asarray(goal_diff)
    return np.where(d >= threshold, "H", np.where(d <= -threshold, "A", "D"))


def poisson_outcome_probs(lam_h, lam_a):
    """Supplementary P(H), P(D), P(A) from independent Poisson goals."""
    goals = np.arange(MAX_GOALS + 1)
    ph = poisson.pmf(goals[None, :], np.asarray(lam_h)[:, None])
    pa = poisson.pmf(goals[None, :], np.asarray(lam_a)[:, None])
    grid = ph[:, :, None] * pa[:, None, :]          # [match, home goals, away goals]
    lower = np.tril(np.ones((len(goals),) * 2), -1)  # home > away
    probs = np.stack([(grid * lower).sum((1, 2)),
                      np.trace(grid, axis1=1, axis2=2),
                      (grid * lower.T).sum((1, 2))], axis=1)
    return probs / probs.sum(1, keepdims=True)


def out_of_fold(train, cols, alpha):
    """Out-of-fold predicted goals on the train period (expanding folds)."""
    oof = pd.DataFrame(index=train.index, columns=["home", "away"], dtype=float)
    for tr, va in TimeSeriesSplit(n_splits=CV_SPLITS).split(train):
        h, a = GoalModel(alpha, cols).fit(train.iloc[tr]).predict_goals(train.iloc[va])
        oof.iloc[va] = np.column_stack([h, a])
    return oof.dropna()


def tune_alpha(train, cols):
    """Alpha with the lowest out-of-fold squared error of home + away goals."""
    def mse(alpha):
        oof = out_of_fold(train, cols, alpha)
        t = train.loc[oof.index]
        return np.mean((oof["home"] - t["home_goals"]) ** 2
                       + (oof["away"] - t["away_goals"]) ** 2)
    scores = [mse(a) for a in RIDGE_ALPHAS]
    return RIDGE_ALPHAS[int(np.argmin(scores))]


def tune_threshold(goal_diff, result):
    """Threshold maximizing accuracy (ties -> smallest threshold)."""
    acc = [np.mean(classify(goal_diff, t) == np.asarray(result)) for t in THRESHOLDS]
    best = int(np.argmax(acc))
    return THRESHOLDS[best], acc[best]


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def goal_metrics(d, pred_h, pred_a):
    gd, pgd = d["home_goals"] - d["away_goals"], pred_h - pred_a
    return {
        "mae_home_goals": np.mean(np.abs(pred_h - d["home_goals"])),
        "mae_away_goals": np.mean(np.abs(pred_a - d["away_goals"])),
        "mae_goal_diff": np.mean(np.abs(pgd - gd)),
        "r2_goal_diff": 1 - np.sum((gd - pgd) ** 2) / np.sum((gd - gd.mean()) ** 2),
        "corr_goal_diff": np.corrcoef(pgd, gd)[0, 1],
    }


def result_metrics(pred, result):
    pred, result = np.asarray(pred), np.asarray(result)
    out = {"accuracy": np.mean(pred == result), "pred_draw_share": np.mean(pred == "D")}
    for r in RESULTS:  # recall: share of actual r predicted as r
        out[f"recall_{r}"] = np.mean(pred[result == r] == r) if (result == r).any() else np.nan
    return out


def confusion(pred, result):
    return (pd.crosstab(pd.Series(np.asarray(pred), name="pred"),
                        pd.Series(np.asarray(result), name="actual"))
            .reindex(index=RESULTS, columns=RESULTS, fill_value=0))


def permutation_importance(model, test, base_mse, rng):
    """Increase of goal-difference MSE when one feature is shuffled."""
    gd = (test["home_goals"] - test["away_goals"]).values
    rows = []
    for c in model.cols:
        losses = []
        for _ in range(PERM_REPEATS):
            shuffled = test.copy()
            shuffled[c] = rng.permutation(shuffled[c].values)
            h, a = model.predict_goals(shuffled)
            losses.append(np.mean((h - a - gd) ** 2))
        rows.append({"feature": c, "importance": np.mean(losses) - base_mse,
                     "std": np.std(losses)})
    return pd.DataFrame(rows).sort_values("importance", ascending=False)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def prediction_frame(d, model, threshold):
    out = d[["season", "round", "kst_date", "kst_time", "home_team", "away_team"]].copy()
    out["kst_date"] = out["kst_date"].dt.strftime("%Y-%m-%d")
    h, a = model.predict_goals(d)
    out["pred_home_goals"], out["pred_away_goals"] = h, a
    out["pred_goal_diff"] = h - a
    out["pred"] = classify(h - a, threshold)
    probs = poisson_outcome_probs(h, a)
    out["P_win"], out["P_draw"], out["P_loss"] = probs[:, 0], probs[:, 1], probs[:, 2]
    if d["finished"].all():
        out["home_goals"], out["away_goals"] = d["home_goals"], d["away_goals"]
        out["result"] = d["result"]
        out["correct"] = out["pred"] == out["result"]
    return out


def main():
    IMPORTANCE_DIR.mkdir(exist_ok=True)
    PRED_DIR.mkdir(exist_ok=True)
    df = load_match_features()
    sets = feature_sets(df)
    train, test, upcoming = chronological_split(df)
    print(f"Train: {train.kst_date.min():%Y-%m-%d} ~ {train.kst_date.max():%Y-%m-%d} "
          f"({len(train)} matches)")
    print(f"Test : {test.kst_date.min():%Y-%m-%d} ~ {test.kst_date.max():%Y-%m-%d} "
          f"({len(test)} matches)")
    print(f"Upcoming: {len(upcoming)} matches")
    print("Feature sets: " + ", ".join(f"{k} {len(v)}" for k, v in sets.items()))

    rows = [{"setup": "baseline_always_home", **result_metrics(np.full(len(test), "H"),
                                                               test["result"])}]
    setups, coefs = {}, []
    for set_name, cols in sets.items():
        alpha = tune_alpha(train, cols)
        oof = out_of_fold(train, cols, alpha)
        threshold, cv_acc = tune_threshold(oof["home"] - oof["away"],
                                           train.loc[oof.index, "result"])
        model = GoalModel(alpha, cols).fit(train)
        h, a = model.predict_goals(test)
        gm = goal_metrics(test, h, a)
        coefs.append(model.coefficients().assign(features=set_name))
        for label, thr in [("tuned", threshold), ("paper_0.5", PAPER_THRESHOLD)]:
            key = f"{set_name}/{label}"
            setups[key] = {"model": model, "threshold": thr, "cv_accuracy": cv_acc}
            rows.append({"setup": f"ridge/{key}", "features": set_name,
                         "n_features": len(cols), "alpha": alpha, "threshold": thr,
                         "cv_accuracy": cv_acc if label == "tuned" else np.nan,
                         **result_metrics(classify(h - a, thr), test["result"]), **gm})

    res = pd.DataFrame(rows)
    res.round(4).to_csv(IMPORTANCE_DIR / "model_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(coefs).round(5).to_csv(IMPORTANCE_DIR / "model_coefficients.csv",
                                     index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print("\n[Test] W/D/L by goal-difference threshold")
    print(res[["setup", "alpha", "threshold", "cv_accuracy", "accuracy", "pred_draw_share",
               "recall_H", "recall_D", "recall_A"]].round(3).to_string(index=False))
    print("\n[Test] goal prediction")
    print(res.dropna(subset=["mae_goal_diff"]).drop_duplicates("features")[
        ["features", "mae_home_goals", "mae_away_goals", "mae_goal_diff",
         "r2_goal_diff", "corr_goal_diff"]].round(3).to_string(index=False))

    # Final setup: tuned threshold, feature set with the best train CV accuracy
    best_key = max((k for k in setups if k.endswith("/tuned")),
                   key=lambda k: setups[k]["cv_accuracy"])
    model, threshold = setups[best_key]["model"], setups[best_key]["threshold"]
    test_pred = prediction_frame(test, model, threshold)
    print(f"\n[Final] {best_key}: alpha={model.alpha:.1f}, threshold={threshold:.2f} "
          f"-> test accuracy {test_pred['correct'].mean():.3f}")
    print(confusion(test_pred["pred"], test_pred["result"]))
    test_pred.round(4).to_csv(PRED_DIR / "test_predictions.csv",
                              index=False, encoding="utf-8-sig")

    gd = (test["home_goals"] - test["away_goals"]).values
    base_mse = np.mean((test_pred["pred_goal_diff"].values - gd) ** 2)
    imp = permutation_importance(model, test, base_mse, np.random.default_rng(0))
    imp.round(5).to_csv(IMPORTANCE_DIR / "permutation_importance.csv",
                        index=False, encoding="utf-8-sig")
    print("\n[Permutation importance] goal-diff MSE increase, top 10")
    print(imp.head(10).round(4).to_string(index=False))

    # Refit on every finished match, save it, and predict the upcoming ones
    finished = pd.concat([train, test])
    final = GoalModel(model.alpha, model.cols).fit(finished)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({  # plain sklearn objects only (rebuilt by load_model)
        "home_pipeline": final.home_,
        "away_pipeline": final.away_,
        "alpha": final.alpha,
        "cols": final.cols,
        "threshold": threshold,
        "feature_set": best_key.split("/")[0],
        "trained_through": f"{finished.kst_date.max():%Y-%m-%d}",
        "n_matches": len(finished),
        "test_accuracy": float(test_pred["correct"].mean()),
    }, MODEL_PATH)
    print(f"\n[Saved model] {MODEL_PATH} (trained on {len(finished)} matches through "
          f"{finished.kst_date.max():%Y-%m-%d})")
    if len(upcoming):
        up = prediction_frame(upcoming, final, threshold)
        up.round(3).to_csv(PRED_DIR / "upcoming_predictions.csv",
                           index=False, encoding="utf-8-sig")
        print(f"\n[Upcoming] {len(up)} matches")
        print(up.drop(columns=["season"]).round(2).to_string(index=False))

    print(f"\nSaved: {IMPORTANCE_DIR}/model_metrics.csv, model_coefficients.csv, "
          f"permutation_importance.csv; {PRED_DIR}/test_predictions.csv, "
          f"upcoming_predictions.csv")


if __name__ == "__main__":
    main()
