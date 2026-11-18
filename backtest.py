"""Walk-forward betting backtest of the goal model against Betman W/D/L odds.

Usage
  python backtest.py                              # 2024-01-01 ~, monthly retrain
  python backtest.py --start 2025-01-01 --retrain week
  python backtest.py --features core --min-edge 0.05 --kelly 0.25
  python backtest.py --reuse                      # skip refitting, reuse predictions.csv

Leakage control
  The data is split into retrain blocks (week / month / season). For every
  block the model is fit ONLY on matches finished before the block starts:
  alpha, draw threshold and feature set are re-tuned with the same
  out-of-fold procedure as feature.py, then the block's matches are
  predicted. Features themselves are pre-match rolling stats (pruning.py).

Odds
  betman_kleague.csv (betman.py). These are the final odds of the round, so
  the backtest assumes every bet is placed at those odds. Betman's margin is
  ~13-15% (sum of 1/odds ~ 1.14), which a bet has to overcome.
  NOTE: most Betman Proto games require a 2+ match combination; this
  backtest evaluates single bets as an upper bound on model value.

Strategies (one bet per match at most)
  value_flat   1 unit on the outcome with the highest EV if EV >= min_edge
  value_kelly  fractional Kelly stake on that outcome (bankroll fraction)
  model_pick   1 unit on the model's W/D/L classification (no odds used)
  favorite     1 unit on the lowest-odds outcome (baseline)
  home         1 unit on the home win (baseline)

Outputs (backtest_results/)
  predictions.csv     per-match model probabilities, market probabilities, odds
  bets.csv            every bet of every strategy with stake / profit / bankroll
  summary.csv         ROI, hit rate, MDD, bootstrap ROI CI per strategy
  season_summary.csv  ROI per strategy and season
  edge_sweep.csv      value_flat ROI by min_edge (sensitivity, not for tuning)
  equity.png          bankroll curves (if matplotlib is installed)
"""

import argparse

import numpy as np
import pandas as pd

import feature as f
import pruning as pr

ODDS_PATH = pr.BASE_DIR / "betman_kleague.csv"
OUT_DIR = pr.BASE_DIR / "backtest_results"
PRED_PATH = OUT_DIR / "predictions.csv"

OUTCOMES = ["H", "D", "A"]
P_COLS = ["P_win", "P_draw", "P_loss"]
ODDS_COLS = ["odds_home", "odds_draw", "odds_away"]
MKT_COLS = ["mkt_H", "mkt_D", "mkt_A"]
EDGE_GRID = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
BOOTSTRAP = 2000


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_odds():
    """Betman K League 1 W/D/L odds with K League short team names."""
    b = pd.read_csv(ODDS_PATH, encoding="utf-8-sig")
    b = b[(b[ODDS_COLS] > 1).all(axis=1)].copy()  # drop TBD / unpriced rows
    for side in ("home", "away"):
        # Betman occasionally prefixes a team with 'XX' (data-entry artifact)
        b[side] = b[side].astype(str).str.replace(r"^XX", "", regex=True)
        b[side] = b[side].map(pr.normalize_team_name)
    dt = pd.to_datetime(b["date"])
    b["kst_date"] = dt.dt.normalize()
    # same match sold in several rounds -> keep the latest round's odds
    b = b.sort_values("gmTs").drop_duplicates(["kst_date", "home", "away"], keep="last")
    return b.rename(columns={"home": "home_team", "away": "away_team"})[
        ["kst_date", "home_team", "away_team", "gmTs", *ODDS_COLS]]


def merge_odds(df, odds):
    m = df.merge(odds, on=["kst_date", "home_team", "away_team"], how="left")
    finished = m[m["finished"]]
    lo, hi = odds["kst_date"].min(), odds["kst_date"].max()
    in_range = finished[finished["kst_date"].between(lo, hi)]
    missing = in_range[in_range["odds_home"].isna()]
    print(f"[Odds] {odds.shape[0]} Betman matches; matched {in_range['odds_home'].notna().sum()}"
          f"/{len(in_range)} finished matches in {lo:%Y-%m-%d} ~ {hi:%Y-%m-%d}")
    if len(missing):
        print(f"[WARNING] {len(missing)} finished matches without odds, e.g.")
        print(missing[["kst_date", "home_team", "away_team"]].head(5).to_string(index=False))
    return m


# --------------------------------------------------------------------------
# Walk-forward predictions
# --------------------------------------------------------------------------
def block_starts(dates, how):
    if how == "season":
        return dates.dt.year.map(lambda y: pd.Timestamp(year=y, month=1, day=1))
    freq = {"week": "W-SUN", "month": "M"}[how]
    return dates.dt.to_period(freq).dt.start_time


def fit_setup(train, sets, feature_choice):
    """Same model selection as feature.py, using only `train`."""
    names = list(sets) if feature_choice == "auto" else [feature_choice]
    best = None
    for name in names:
        cols = sets[name]
        alpha = f.tune_alpha(train, cols)
        oof = f.out_of_fold(train, cols, alpha)
        thr, cv_acc = f.tune_threshold(oof["home"] - oof["away"], train.loc[oof.index, "result"])
        if best is None or cv_acc > best["cv_accuracy"]:
            best = {"features": name, "cols": cols, "alpha": alpha,
                    "threshold": thr, "cv_accuracy": cv_acc}
    best["model"] = f.GoalModel(best["alpha"], best["cols"]).fit(train)
    return best


def walk_forward(df, start, retrain, feature_choice, min_train):
    sets = f.feature_sets(df)
    target = df[df["finished"] & (df["kst_date"] >= start) & df["odds_home"].notna()].copy()
    target["block"] = block_starts(target["kst_date"], retrain)
    finished = df[df["finished"]]
    parts = []
    blocks = sorted(target["block"].unique())
    for i, block in enumerate(blocks, 1):
        train = finished[finished["kst_date"] < block]
        if len(train) < min_train:
            print(f"  skip block {block:%Y-%m-%d}: only {len(train)} training matches")
            continue
        s = fit_setup(train, sets, feature_choice)
        d = target[target["block"] == block]
        h, a = s["model"].predict_goals(d)
        probs = f.poisson_outcome_probs(h, a)
        out = d[["season", "round", "kst_date", "kst_time", "home_team", "away_team",
                 "home_goals", "away_goals", "result", "gmTs", *ODDS_COLS]].copy()
        out["pred_home_goals"], out["pred_away_goals"] = h, a
        out["pred"] = f.classify(h - a, s["threshold"])
        out[P_COLS] = probs
        out["train_until"] = train["kst_date"].max()
        out["n_train"] = len(train)
        out["feature_set"], out["alpha"], out["threshold"] = s["features"], s["alpha"], s["threshold"]
        parts.append(out)
        print(f"  [{i}/{len(blocks)}] block {block:%Y-%m-%d}: train {len(train)} "
              f"(<= {train['kst_date'].max():%Y-%m-%d}), predict {len(d)}, "
              f"{s['features']} alpha={s['alpha']:.0f} thr={s['threshold']:.2f}")
    if not parts:
        raise SystemExit("[ERROR] nothing to backtest (check --start / odds coverage)")
    pred = pd.concat(parts, ignore_index=True)
    implied = 1 / pred[ODDS_COLS].to_numpy()
    pred["overround"] = implied.sum(1)
    pred[MKT_COLS] = implied / implied.sum(1, keepdims=True)  # margin removed proportionally
    return pred


# --------------------------------------------------------------------------
# Strategies
# --------------------------------------------------------------------------
def best_ev(pred):
    p = pred[P_COLS].to_numpy()
    o = pred[ODDS_COLS].to_numpy()
    ev = p * o - 1
    k = ev.argmax(1)
    idx = np.arange(len(pred))
    return k, ev[idx, k], p[idx, k], o[idx, k]


def make_bets(pred, strategy, min_edge, kelly, max_stake):
    """Rows: chosen outcome and stake rule ('unit' or Kelly bankroll fraction)."""
    o = pred[ODDS_COLS].to_numpy()
    idx = np.arange(len(pred))
    if strategy in ("value_flat", "value_kelly"):
        k, ev, p, odds = best_ev(pred)
        mask = ev >= min_edge
        if strategy == "value_flat":
            frac = np.full(len(pred), np.nan)
        else:
            frac = np.clip(kelly * ev / (odds - 1), 0, max_stake)
            mask &= frac > 0
    else:
        if strategy == "model_pick":
            k = pred["pred"].map(OUTCOMES.index).to_numpy()
        elif strategy == "favorite":
            k = o.argmin(1)
        elif strategy == "home":
            k = np.zeros(len(pred), dtype=int)
        else:
            raise ValueError(strategy)
        mask = np.ones(len(pred), dtype=bool)
        frac = np.full(len(pred), np.nan)
        ev = pred[P_COLS].to_numpy()[idx, k] * o[idx, k] - 1
    bets = pred.loc[mask, ["season", "kst_date", "kst_time", "home_team", "away_team",
                           "result"]].copy()
    bets["strategy"] = strategy
    bets["pick"] = np.array(OUTCOMES)[k[mask]]
    bets["odds"] = o[idx, k][mask]
    bets["model_p"] = pred[P_COLS].to_numpy()[idx, k][mask]
    bets["market_p"] = pred[MKT_COLS].to_numpy()[idx, k][mask]
    bets["ev"] = ev[mask]
    bets["kelly_frac"] = frac[mask]
    bets["won"] = bets["pick"] == bets["result"]
    return bets


def settle(bets, flat_bankroll):
    """Sequential bankroll. Stakes for one day use that day's opening bankroll,
    so a result is never used to size a bet on a match played the same day."""
    bets = bets.sort_values(["kst_date", "kst_time"]).copy()
    kelly = bets["kelly_frac"].notna().any()
    bank = 1.0 if kelly else float(flat_bankroll)
    stakes, banks = [], []
    for _, day in bets.groupby("kst_date", sort=False):
        day_stakes = (day["kelly_frac"] * bank).to_numpy() if kelly else np.ones(len(day))
        profit = np.where(day["won"], day_stakes * (day["odds"] - 1), -day_stakes)
        running = bank + np.cumsum(profit)
        bank = running[-1]
        stakes.extend(day_stakes)
        banks.extend(running)
    bets["stake"] = stakes
    bets["profit"] = np.where(bets["won"], bets["stake"] * (bets["odds"] - 1), -bets["stake"])
    bets["bankroll"] = banks
    return bets


def max_drawdown(bankroll, start, relative):
    """Largest peak-to-trough fall: fraction of the peak (Kelly) or units (flat)."""
    curve = np.concatenate([[start], np.asarray(bankroll)])
    peak = np.maximum.accumulate(curve)
    fall = peak - curve
    return float((fall / peak).max() if relative else fall.max())


def longest_losing_streak(won):
    best = cur = 0
    for w in won:
        cur = 0 if w else cur + 1
        best = max(best, cur)
    return best


def bootstrap_roi_ci(bets, rng):
    """95% CI of flat ROI (profit per unit) resampling bets with replacement."""
    if len(bets) < 2:
        return np.nan, np.nan
    r = (np.where(bets["won"], bets["odds"] - 1, -1.0))
    sims = rng.choice(r, size=(BOOTSTRAP, len(r)), replace=True).mean(1)
    return tuple(np.percentile(sims, [2.5, 97.5]))


def summarize(bets, n_matches, flat_bankroll, rng):
    if bets.empty:
        return {"n_matches": n_matches, "n_bets": 0}
    kelly = bets["kelly_frac"].notna().any()
    start = 1.0 if kelly else float(flat_bankroll)
    lo, hi = bootstrap_roi_ci(bets, rng)
    return {
        "n_matches": n_matches,
        "n_bets": len(bets),
        "bet_share": len(bets) / n_matches,
        "hit_rate": bets["won"].mean(),
        "avg_odds": bets["odds"].mean(),
        "avg_model_p": bets["model_p"].mean(),
        "avg_market_p": bets["market_p"].mean(),
        "staked": bets["stake"].sum(),
        "profit": bets["profit"].sum(),
        "roi": bets["profit"].sum() / bets["stake"].sum(),
        "flat_roi_ci_low": lo,
        "flat_roi_ci_high": hi,
        "start_bankroll": start,
        "final_bankroll": bets["bankroll"].iloc[-1],
        # flat: units lost from the peak; Kelly: fraction of the peak bankroll
        "max_dd_units": np.nan if kelly else max_drawdown(bets["bankroll"], start, False),
        "max_dd_pct": max_drawdown(bets["bankroll"], start, True) if kelly else np.nan,
        "longest_losing_streak": longest_losing_streak(bets["won"]),
        "picks_H/D/A": "/".join(str((bets["pick"] == r).sum()) for r in OUTCOMES),
    }


# --------------------------------------------------------------------------
# Probability quality: model vs market
# --------------------------------------------------------------------------
def prob_scores(pred):
    y = np.stack([(pred["result"] == r).to_numpy() for r in OUTCOMES], 1).astype(float)
    base = pred["result"].value_counts(normalize=True).reindex(OUTCOMES).to_numpy()
    rows = []
    for name, p in [("model", pred[P_COLS].to_numpy()),
                    ("market (1/odds, margin removed)", pred[MKT_COLS].to_numpy()),
                    ("base rate (in-sample)", np.tile(base, (len(pred), 1)))]:
        p = np.clip(p, 1e-12, 1)
        rows.append({"probabilities": name,
                     "log_loss": -np.mean(np.log((p * y).sum(1))),
                     "brier": np.mean(((p - y) ** 2).sum(1)),
                     "accuracy(argmax)": np.mean(p.argmax(1) == y.argmax(1))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def plot_equity(all_bets, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[INFO] matplotlib not installed: skip equity.png")
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for s, b in all_bets.groupby("strategy"):
        ax = axes[1] if b["kelly_frac"].notna().any() else axes[0]
        ax.plot(pd.to_datetime(b["kst_date"]), b["profit"].cumsum() if ax is axes[0]
                else b["bankroll"], label=s)
    axes[0].set_title("Flat 1-unit strategies: cumulative profit (units)")
    axes[1].set_title("Fractional Kelly: bankroll (start = 1.0)")
    for ax in axes:
        ax.axhline(0 if ax is axes[0] else 1, color="grey", lw=0.8)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"Saved: {path}")


def main():
    ap = argparse.ArgumentParser(description="Walk-forward betting backtest")
    ap.add_argument("--start", default="2024-01-01", help="first match date to bet on")
    ap.add_argument("--retrain", choices=["week", "month", "season"], default="month")
    ap.add_argument("--features", choices=["auto", "core", "all"], default="auto",
                    help="auto = pick by train CV accuracy each refit (like feature.py)")
    ap.add_argument("--min-train", type=int, default=200, help="min training matches")
    ap.add_argument("--min-edge", type=float, default=0.05, help="EV threshold for value bets")
    ap.add_argument("--kelly", type=float, default=0.25, help="Kelly fraction")
    ap.add_argument("--max-stake", type=float, default=0.05, help="max bankroll fraction per bet")
    ap.add_argument("--flat-bankroll", type=float, default=100, help="start units for flat bets")
    ap.add_argument("--reuse", action="store_true", help="reuse backtest_results/predictions.csv")
    args = ap.parse_args()
    OUT_DIR.mkdir(exist_ok=True)

    if args.reuse and PRED_PATH.exists():
        pred = pd.read_csv(PRED_PATH, encoding="utf-8-sig", parse_dates=["kst_date"])
        print(f"[Reuse] {PRED_PATH} ({len(pred)} matches)")
    else:
        df = merge_odds(f.load_match_features(), load_odds())
        print(f"[Walk-forward] start={args.start}, retrain={args.retrain}, "
              f"features={args.features}")
        pred = walk_forward(df, pd.Timestamp(args.start), args.retrain,
                            args.features, args.min_train)
        pred.round(5).to_csv(PRED_PATH, index=False, encoding="utf-8-sig")
        print(f"Saved: {PRED_PATH}")

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    n = len(pred)
    print(f"\nBacktest: {pred['kst_date'].min():%Y-%m-%d} ~ {pred['kst_date'].max():%Y-%m-%d}, "
          f"{n} matches, mean overround {pred['overround'].mean():.3f}")
    print(f"Model W/D/L accuracy: {(pred['pred'] == pred['result']).mean():.3f}")

    print("\n[Probability quality] lower log loss / Brier is better")
    print(prob_scores(pred).round(4).to_string(index=False))

    rng = np.random.default_rng(0)
    strategies = ["value_flat", "value_kelly", "model_pick", "favorite", "home"]
    all_bets, rows, season_rows = [], [], []
    for s in strategies:
        bets = settle(make_bets(pred, s, args.min_edge, args.kelly, args.max_stake),
                      args.flat_bankroll)
        all_bets.append(bets)
        rows.append({"strategy": s, **summarize(bets, n, args.flat_bankroll, rng)})
        for season, b in bets.groupby("season"):
            season_rows.append({"strategy": s, "season": season, "n_bets": len(b),
                                "hit_rate": b["won"].mean(),
                                "roi": b["profit"].sum() / b["stake"].sum()})
    all_bets = pd.concat(all_bets, ignore_index=True)
    summary = pd.DataFrame(rows)
    seasons = pd.DataFrame(season_rows)

    sweep = []
    for e in EDGE_GRID:
        b = make_bets(pred, "value_flat", e, args.kelly, args.max_stake)
        r = np.where(b["won"], b["odds"] - 1, -1.0)
        sweep.append({"min_edge": e, "n_bets": len(b),
                      "hit_rate": b["won"].mean() if len(b) else np.nan,
                      "roi": r.mean() if len(b) else np.nan})
    sweep = pd.DataFrame(sweep)

    print(f"\n[Strategies] min_edge={args.min_edge}, kelly={args.kelly}, "
          f"max_stake={args.max_stake}")
    print(summary.round(3).to_string(index=False))
    print("\n[ROI by season]")
    print(seasons.pivot(index="strategy", columns="season", values="roi")
          .reindex(strategies).round(3).to_string())
    print("\n[value_flat ROI by min_edge] sensitivity only - picking the best row is overfitting")
    print(sweep.round(3).to_string(index=False))

    all_bets.round(5).to_csv(OUT_DIR / "bets.csv", index=False, encoding="utf-8-sig")
    summary.round(5).to_csv(OUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    seasons.round(5).to_csv(OUT_DIR / "season_summary.csv", index=False, encoding="utf-8-sig")
    sweep.round(5).to_csv(OUT_DIR / "edge_sweep.csv", index=False, encoding="utf-8-sig")
    print(f"\nSaved: {OUT_DIR}/bets.csv, summary.csv, season_summary.csv, edge_sweep.csv")
    plot_equity(all_bets, OUT_DIR / "equity.png")


if __name__ == "__main__":
    main()
