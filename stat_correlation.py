"""Select and classify team stats into attack / defense groups by correlation.

Uses the chronological TRAIN period only (first TRAIN_RATIO of matches), so the
feature selection itself does not peek at the test period.

For every candidate stat (own stats `kl_*`/`fm_*`, conceded stats `opp_*`):
  * predictive r : pre-match mean of the stat (rolling ROLL_WINDOW and
                   season-to-date) vs goals scored / conceded in the next match
  * same-match r : stat vs goals in the same match (reference only)

Classification uses the PREDICTIVE correlation. Same-match correlation is
distorted by game state (a leading team sits back: possession and passes
correlate negatively with goals in the same match but positively with future
goals), so it is only reported.

  attack   : stat predicts goals scored more than goals conceded
  defense  : stat predicts goals conceded more than goals scored
  strength : predicts both about equally (overall team quality)
  context : schedule features (rest days, kickoff, ...) that predict points

Outputs (feature_importance/):
  stat_correlation.csv  every candidate with correlations and drop reason
  stat_groups.csv       selected: stat, group, ... (read by pruning.py)
"""

import numpy as np
import pandas as pd

import pruning as pr

OUT_DIR = pr.BASE_DIR / "feature_importance"
TRAIN_RATIO = 0.7          # same chronological split as the model
MIN_PRED_CORR = 0.05       # ~2 standard errors with ~1,500 team-matches
MIN_SIGN_STABILITY = 0.75  # share of train seasons with the same sign
MIN_COVERAGE = 0.8
DUPLICATE_CORR = 0.8       # between pre-match means; keep the stronger one
GROUP_MARGIN = 0.02        # |r_gf| and |r_ga| closer than this -> 'strength'


def spearman(x, y):
    mask = x.notna() & y.notna()
    if mask.sum() < 30:
        return np.nan
    return x[mask].corr(y[mask], method="spearman")


def train_mask(tm):
    """Chronological first TRAIN_RATIO of finished matches."""
    times = tm.loc[tm["finished"]].drop_duplicates("match_id")["utc_time"].sort_values()
    n_train = int(len(times) * TRAIN_RATIO)
    cutoff = times.iloc[n_train - 1]
    print(f"Train period: {times.iloc[0]:%Y-%m-%d} ~ {cutoff:%Y-%m-%d} ({n_train} matches)")
    return tm["finished"] & (tm["utc_time"] <= cutoff)


def load_all_stats():
    """Team-match table with every stat and its conceded (opp_) version."""
    kl = pr.load_kleague_long(None)
    fm_all = pr.load_fotmob_long(pr.fotmob_seasons(), None)
    own = [c for c in list(kl.columns) + list(fm_all.columns)
           if c.startswith(("kl_", "fm_"))]
    kl_stats = sorted({c[3:] for c in own if c.startswith("kl_")})
    fm_stats = sorted({c[3:] for c in own if c.startswith("fm_")})
    return pr.build_team_match_table(kl_stats, fm_stats, sorted(set(own)))


def season_sign_stability(values, target, train, seasons, overall):
    rs = [spearman(values[train & (seasons == s)], target[train & (seasons == s)])
          for s in sorted(seasons[train].unique())]
    rs = [r for r in rs if pd.notna(r)]
    return np.mean([np.sign(r) == np.sign(overall) for r in rs]) if rs else np.nan


def analyze_stats(tm, train):
    cands = [c for c in tm.columns
             if (c.startswith(("kl_", "fm_", "opp_")) and c != "opp_id")
             or c == "goals_minus_xg"]
    cands = [c for c in cands if pd.api.types.is_numeric_dtype(tm[c])]

    roll = pr.prior_state(tm, cands, ["team_id"], window=pr.ROLL_WINDOW)
    season = pr.prior_state(tm, cands, ["team_id", "season"])
    gf, ga = tm["goals_for"].astype(float), tm["goals_against"].astype(float)

    rows = []
    for c in cands:
        r = {
            "stat": c,
            "coverage": tm.loc[train, c].notna().mean(),
            "pred_roll_gf": spearman(roll.loc[train, c], gf[train]),
            "pred_roll_ga": spearman(roll.loc[train, c], ga[train]),
            "pred_season_gf": spearman(season.loc[train, c], gf[train]),
            "pred_season_ga": spearman(season.loc[train, c], ga[train]),
            "same_gf": spearman(tm.loc[train, c], gf[train]),
            "same_ga": spearman(tm.loc[train, c], ga[train]),
        }
        r["pred_gf"] = np.nanmean([r["pred_roll_gf"], r["pred_season_gf"]])
        r["pred_ga"] = np.nanmean([r["pred_roll_ga"], r["pred_season_ga"]])
        attack = abs(r["pred_gf"]) >= abs(r["pred_ga"])
        if abs(abs(r["pred_gf"]) - abs(r["pred_ga"])) < GROUP_MARGIN:
            r["group"] = "strength"  # predicts both: overall team quality
        else:
            r["group"] = "attack" if attack else "defense"
        r["pred_r"] = r["pred_gf"] if attack else r["pred_ga"]
        target = gf if attack else ga
        r["roll_season_agree"] = (np.sign(r["pred_roll_gf" if attack else "pred_roll_ga"])
                                  == np.sign(r["pred_season_gf" if attack else "pred_season_ga"]))
        r["sign_stability"] = season_sign_stability(roll[c], target, train, tm["season"],
                                                    r["pred_r"])
        r["game_state_reversal"] = (np.sign(r["same_gf" if attack else "same_ga"])
                                    != np.sign(r["pred_r"]))
        rows.append(r)
    res = pd.DataFrame(rows)

    def reason(row):
        if row.coverage < MIN_COVERAGE:
            return "low_coverage"
        if not abs(row.pred_r) >= MIN_PRED_CORR:
            return "not_predictive"
        if not row.roll_season_agree:
            return "roll_season_disagree"
        if row.sign_stability < MIN_SIGN_STABILITY:
            return "unstable_sign"
        return ""
    res["drop_reason"] = res.apply(reason, axis=1)

    # Remove near-duplicates (by correlation of the pre-match rolling means)
    cand = res[res["drop_reason"] == ""].assign(abs_r=lambda d: d["pred_r"].abs())
    cand = cand.sort_values("abs_r", ascending=False)
    corr = roll.loc[train, cand["stat"]].corr(method="spearman").abs()
    kept = []
    for s in cand["stat"]:
        dup = next((k for k in kept if corr.loc[s, k] >= DUPLICATE_CORR), None)
        if dup:
            res.loc[res["stat"] == s, "drop_reason"] = f"duplicate_of:{dup}"
        else:
            kept.append(s)
    return res


# Context features shared by both teams of a match. At team level their
# correlation with points cancels out (winner 3 + loser 0), so they are
# evaluated against the HOME team's points (= do they change home advantage).
SHARED_CONTEXT = {"is_weekend", "kickoff_hour"}


def analyze_context(tm, train):
    """Match-level check: home-away difference (or shared value) vs home points."""
    ctx = pr.compute_context_features(tm)
    cols = list(ctx.columns)
    df = pd.concat([tm[["match_id", "is_home", "season", "points"]], ctx], axis=1)
    df["train"] = train
    home = df[df["is_home"] == 1].set_index("match_id")
    away = df[df["is_home"] == 0].set_index("match_id").reindex(home.index)

    pts = home["points"].astype(float)
    m_train = home["train"]
    rows = []
    for c in cols:
        x = home[c] if c in SHARED_CONTEXT else home[c] - away[c]
        r_pts = spearman(x[m_train], pts[m_train])
        stab = season_sign_stability(x, pts, m_train, home["season"], r_pts)
        rows.append({"stat": c, "group": "context", "coverage": x[m_train].notna().mean(),
                     "pred_r": r_pts, "sign_stability": stab,
                     "context_test": "home value" if c in SHARED_CONTEXT else "home - away"})
    res = pd.DataFrame(rows)
    res["drop_reason"] = np.where(
        res["stat"] == "season_matches_played", "",  # kept: tells the model how
        np.where(res["pred_r"].abs() < MIN_PRED_CORR, "not_predictive",  # reliable season_* is
                 np.where(res["sign_stability"] < MIN_SIGN_STABILITY, "unstable_sign", "")))
    return res


def main():
    OUT_DIR.mkdir(exist_ok=True)
    tm = load_all_stats()
    train = train_mask(tm)

    stats = analyze_stats(tm, train)
    context = analyze_context(tm, train)
    res = pd.concat([stats, context], ignore_index=True)
    res["selected"] = res["drop_reason"] == ""
    res = res.assign(abs_r=res["pred_r"].abs()).sort_values(
        ["selected", "group", "abs_r"], ascending=[False, True, False]).drop(columns="abs_r")
    res.round(4).to_csv(OUT_DIR / "stat_correlation.csv", index=False, encoding="utf-8-sig")

    sel = res[res["selected"]]
    sel[["stat", "group", "pred_r", "pred_gf", "pred_ga", "same_gf", "same_ga"]].round(4).to_csv(
        OUT_DIR / "stat_groups.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 220)
    cols = ["stat", "group", "pred_gf", "pred_ga", "same_gf", "same_ga",
            "sign_stability", "game_state_reversal"]
    print("\n[Selected]")
    print(sel[cols].round(3).to_string(index=False))
    print(f"\nSelected {len(sel)} / {len(res)}: "
          + ", ".join(f"{g} {n}" for g, n in sel["group"].value_counts().items()))
    print("Dropped:", res.loc[~res["selected"], "drop_reason"].str.split(":").str[0]
          .value_counts().to_dict())
    print(f"Saved: {OUT_DIR / 'stat_correlation.csv'}, {OUT_DIR / 'stat_groups.csv'}")


if __name__ == "__main__":
    main()
