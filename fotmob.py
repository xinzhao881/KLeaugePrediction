"""Scrape K League 1 fixtures, match stats and lineups from FotMob with Selenium.

FotMob is a Next.js app, so every page embeds its data in the `__NEXT_DATA__`
script tag. We read that JSON instead of parsing the rendered DOM.

Stage 1 (fixtures): one page load per season returns every match (all rounds).
Stage 2 (details):  one page load per finished match (`/match/{id}`) returns
                    team stats (xG, possession, shots, ...), lineups and
                    per-player stats. A trimmed JSON is cached per match so the
                    run can be resumed.

NOTE: the `/matches/{slug}/{code}#{id}` URL ignores the `#id` server-side and
returns the latest match of that pairing, so always use `/match/{id}`.

Outputs:
    fotmob_csv/{season}/matches.csv      fixtures and results
    fotmob_csv/{season}/match_stats.csv  one row per match, home_/away_ stats
    fotmob_csv/{season}/lineups.csv      one row per player per match
    fotmob_csv/raw/{season}/{id}.json    trimmed raw cache

WARNING (data leakage): match_stats/lineups are POST-match data. Use them only
to build rolling/prior-match features, never as same-match features.
"""

import argparse
import json
import os
import random
import re
import time

import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

LEAGUE_ID = 9080
FIXTURES_URL = (
    "https://www.fotmob.com/ko/leagues/{league_id}/fixtures/k-league-1"
    "?season={season}&group=by-round&round=0"
)
MATCH_URL = "https://www.fotmob.com/match/{match_id}"
SEASONS = [2022, 2023, 2024, 2025, 2026]
OUTPUT_ROOT = "fotmob_csv"
RAW_ROOT = os.path.join(OUTPUT_ROOT, "raw")
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/157.0.0.0 Safari/537.36"
)
DRIVER_RESTART_EVERY = 150  # restart Chrome periodically to limit memory growth


# --------------------------------------------------------------------------- #
# Selenium helpers
# --------------------------------------------------------------------------- #
def build_driver(headless=True):
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--lang=ko-KR")
    options.add_argument(f"user-agent={USER_AGENT}")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--window-size=1400,1000")
    # Skip images to speed up page loads; we only need the embedded JSON.
    options.add_experimental_option(
        "prefs", {"profile.managed_default_content_settings.images": 2}
    )
    return webdriver.Chrome(options=options)


def load_next_data(driver, url):
    driver.get(url)
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.ID, "__NEXT_DATA__"))
    )
    raw = driver.execute_script(
        "return document.getElementById('__NEXT_DATA__').textContent;"
    )
    return json.loads(raw)


# --------------------------------------------------------------------------- #
# Stage 1: fixtures
# --------------------------------------------------------------------------- #
def fetch_fixtures_data(driver, season, retries=3):
    url = FIXTURES_URL.format(league_id=LEAGUE_ID, season=season)
    for attempt in range(1, retries + 1):
        try:
            return load_next_data(driver, url)
        except (TimeoutException, json.JSONDecodeError) as exc:
            print(f"[WARN] fixtures {season} attempt {attempt} failed: {exc}")
            time.sleep(3 * attempt)
    raise RuntimeError(f"Failed to load FotMob fixtures for season {season}")


def parse_score(score_str):
    """'2 - 1' -> (2, 1); returns (None, None) for unplayed matches."""
    if not score_str or "-" not in score_str:
        return None, None
    try:
        home, away = (int(s.strip()) for s in score_str.split("-", 1))
        return home, away
    except ValueError:
        return None, None


def match_to_row(season, match):
    status = match.get("status", {})
    finished = bool(status.get("finished"))
    home_goals, away_goals = parse_score(status.get("scoreStr")) if finished else (None, None)

    if home_goals is None:
        result = None
    elif home_goals > away_goals:
        result = "H"
    elif home_goals < away_goals:
        result = "A"
    else:
        result = "D"

    utc_time = pd.to_datetime(status.get("utcTime"), utc=True)
    kst_time = utc_time.tz_convert("Asia/Seoul") if pd.notna(utc_time) else pd.NaT

    return {
        "season": season,
        "round": int(match["round"]) if str(match.get("round", "")).isdigit() else match.get("round"),
        "match_id": match.get("id"),
        "utc_time": status.get("utcTime"),
        "kst_date": kst_time.strftime("%Y-%m-%d") if pd.notna(kst_time) else None,
        "kst_time": kst_time.strftime("%H:%M") if pd.notna(kst_time) else None,
        "home_team": match["home"].get("name"),
        "home_team_id": match["home"].get("id"),
        "away_team": match["away"].get("name"),
        "away_team_id": match["away"].get("id"),
        "home_goals": home_goals,
        "away_goals": away_goals,
        "result": result,
        "finished": finished,
        "started": bool(status.get("started")),
        "cancelled": bool(status.get("cancelled")),
        "awarded": bool(status.get("awarded")),
        "status_short": (status.get("reason") or {}).get("short"),
        "page_url": MATCH_URL.format(match_id=match.get("id")),
    }


def scrape_fixtures(driver, season):
    data = fetch_fixtures_data(driver, season)
    page_props = data["props"]["pageProps"]

    available = page_props.get("allAvailableSeasons", [])
    if available and str(season) not in available:
        print(f"[WARN] season {season} not in available seasons: {available}")

    matches = page_props.get("fixtures", {}).get("allMatches", [])
    df = pd.DataFrame([match_to_row(season, m) for m in matches])
    if not df.empty:
        df[["home_goals", "away_goals"]] = df[["home_goals", "away_goals"]].astype("Int64")
        df = df.sort_values(["round", "utc_time", "match_id"]).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Stage 2: match details (fetch + cache)
# --------------------------------------------------------------------------- #
def raw_path(season, match_id):
    return os.path.join(RAW_ROOT, str(season), f"{match_id}.json")


def trim_match_data(page_props):
    """Keep only the parts we parse, to keep the cache small."""
    content = page_props.get("content", {}) or {}
    match_facts = content.get("matchFacts", {}) or {}
    return {
        "general": page_props.get("general"),
        "header": page_props.get("header"),
        "infoBox": match_facts.get("infoBox"),
        "stats": content.get("stats"),
        "lineup": content.get("lineup"),
        "playerStats": content.get("playerStats"),
    }


def fetch_match_detail(driver, match_id, retries=3):
    url = MATCH_URL.format(match_id=match_id)
    for attempt in range(1, retries + 1):
        try:
            page_props = load_next_data(driver, url)["props"]["pageProps"]
            got_id = str((page_props.get("general") or {}).get("matchId"))
            if got_id != str(match_id):
                raise ValueError(f"expected match {match_id}, got {got_id}")
            return trim_match_data(page_props)
        except (TimeoutException, json.JSONDecodeError, KeyError, ValueError) as exc:
            print(f"[WARN] match {match_id} attempt {attempt} failed: {exc}")
            time.sleep(3 * attempt)
    return None


def scrape_details(fixtures, headless=True):
    """Fetch and cache every finished match that is not cached yet."""
    targets = fixtures[fixtures["finished"] & ~fixtures["cancelled"]]
    todo = [
        (row.season, row.match_id)
        for row in targets.itertuples()
        if not os.path.exists(raw_path(row.season, row.match_id))
    ]
    print(f"Match details: {len(targets) - len(todo)} cached, {len(todo)} to fetch")
    if not todo:
        return

    driver = build_driver(headless=headless)
    failed = []
    try:
        for i, (season, match_id) in enumerate(todo, start=1):
            if i > 1 and (i - 1) % DRIVER_RESTART_EVERY == 0:
                driver.quit()
                driver = build_driver(headless=headless)

            try:
                detail = fetch_match_detail(driver, match_id)
            except WebDriverException as exc:
                print(f"[WARN] driver error on {match_id}, restarting: {exc.msg}")
                driver.quit()
                driver = build_driver(headless=headless)
                detail = fetch_match_detail(driver, match_id)

            if detail is None:
                failed.append(match_id)
            else:
                path = raw_path(season, match_id)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(detail, f, ensure_ascii=False)

            if i % 20 == 0 or i == len(todo):
                print(f"  [{i}/{len(todo)}] fetched (failed: {len(failed)})")
            time.sleep(random.uniform(1.5, 3.0))  # rate limiting
    finally:
        driver.quit()

    if failed:
        print(f"[WARN] {len(failed)} matches failed, re-run to retry: {failed}")


# --------------------------------------------------------------------------- #
# Stage 2: parsing cached details into tables
# --------------------------------------------------------------------------- #
def clean_key(key):
    """'matchstats.headers.tackles' -> 'tackles', 'BallPossesion' -> 'ballpossesion'."""
    key = str(key).split(".")[-1]
    key = re.sub(r"[^0-9a-zA-Z]+", "_", key).strip("_")
    return key.lower()


VALUE_PCT_RE = re.compile(r"^\s*(-?[\d.]+)\s*\((\d+)%\)\s*$")


def parse_stat_value(value):
    """Return (value, pct): '287 (74%)' -> (287, 74), '2.11' -> (2.11, None)."""
    if value is None:
        return None, None
    if isinstance(value, (int, float)):
        return value, None
    text = str(value).strip()
    match = VALUE_PCT_RE.match(text)
    if match:
        return float(match.group(1)), float(match.group(2))
    try:
        return float(text.rstrip("%")), None
    except ValueError:
        return text, None


def parse_team_stats(detail, period="All"):
    """Flatten team stats of one period into {home_<key>: v, away_<key>: v}."""
    row = {}
    periods = (detail.get("stats") or {}).get("Periods") or {}
    for group in (periods.get(period) or {}).get("stats", []):
        for stat in group.get("stats", []):
            if stat.get("type") == "title":
                continue
            key = clean_key(stat.get("key") or stat.get("title"))
            values = stat.get("stats") or [None, None]
            for side, raw in zip(("home", "away"), values):
                col = f"{side}_{key}"
                if col in row:  # same stat repeated in several groups
                    continue
                value, pct = parse_stat_value(raw)
                row[col] = value
                if pct is not None:
                    row[f"{col}_pct"] = pct
    return row


def parse_match_stats(season, match_id, detail):
    general = detail.get("general") or {}
    info = detail.get("infoBox") or {}
    lineup = detail.get("lineup") or {}
    header_teams = (detail.get("header") or {}).get("teams") or [{}, {}]

    row = {
        "season": season,
        "match_id": str(match_id),
        "round": general.get("matchRound"),
        "league_name": general.get("leagueName"),
        "coverage_level": general.get("coverageLevel"),
        "home_team": (general.get("homeTeam") or {}).get("name"),
        "home_team_id": (general.get("homeTeam") or {}).get("id"),
        "away_team": (general.get("awayTeam") or {}).get("name"),
        "away_team_id": (general.get("awayTeam") or {}).get("id"),
        "home_score": header_teams[0].get("score"),
        "away_score": header_teams[1].get("score") if len(header_teams) > 1 else None,
        "stadium": (info.get("Stadium") or {}).get("name"),
        "referee": (info.get("Referee") or {}).get("text"),
        "attendance": info.get("Attendance"),
    }

    for side in ("home", "away"):
        team = lineup.get(f"{side}Team") or {}
        row[f"{side}_formation"] = team.get("formation")
        row[f"{side}_team_rating"] = team.get("rating")
        row[f"{side}_avg_starter_age"] = team.get("averageStarterAge")
        row[f"{side}_starter_market_value"] = team.get("totalStarterMarketValue")
        row[f"{side}_coach"] = (team.get("coach") or {}).get("name")

    row.update(parse_team_stats(detail, "All"))
    return row


def flatten_player_stats(player_stats):
    """Flatten FotMob per-player stat groups into {key: value, key_total: total}."""
    out = {}
    for group in (player_stats or {}).get("stats", []):
        for title, item in (group.get("stats") or {}).items():
            stat = item.get("stat") or {}
            if stat.get("type") == "boolean":
                continue
            key = clean_key(item.get("key") or title)
            if key in out:
                continue
            out[key] = stat.get("value")
            if "total" in stat:
                out[f"{key}_total"] = stat.get("total")
    return out


def parse_lineups(season, match_id, detail):
    lineup = detail.get("lineup") or {}
    player_stats = detail.get("playerStats") or {}
    rows = []

    for side in ("home", "away"):
        team = lineup.get(f"{side}Team") or {}
        groups = [
            ("starter", team.get("starters") or []),
            ("sub", team.get("subs") or []),
            ("unavailable", team.get("unavailable") or []),
        ]
        for role, players in groups:
            for p in players:
                perf = p.get("performance") or {}
                sub_events = perf.get("substitutionEvents") or []
                sub_in = next((e.get("time") for e in sub_events if e.get("type") == "subIn"), None)
                sub_out = next((e.get("time") for e in sub_events if e.get("type") == "subOut"), None)

                row = {
                    "season": season,
                    "match_id": str(match_id),
                    "side": side,
                    "team_id": team.get("id"),
                    "team": team.get("name"),
                    "formation": team.get("formation"),
                    "role": role,
                    "player_id": p.get("id"),
                    "player_name": p.get("name"),
                    "shirt_number": p.get("shirtNumber"),
                    "position_id": p.get("positionId"),
                    "usual_position_id": p.get("usualPlayingPositionId"),
                    "age": p.get("age"),
                    "country": p.get("countryCode"),
                    "market_value": p.get("marketValue"),
                    "rating": perf.get("rating"),
                    "sub_in_minute": sub_in,
                    "sub_out_minute": sub_out,
                    "unavailable_reason": ((p.get("unavailability") or {}).get("type")
                                           if role == "unavailable" else None),
                }
                row.update(flatten_player_stats(player_stats.get(str(p.get("id")))))
                rows.append(row)
    return rows


def build_detail_tables(season, fixtures):
    stats_rows, lineup_rows = [], []
    for match_id in fixtures.loc[fixtures["season"] == season, "match_id"]:
        path = raw_path(season, match_id)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            detail = json.load(f)
        stats_rows.append(parse_match_stats(season, match_id, detail))
        lineup_rows.extend(parse_lineups(season, match_id, detail))

    stats_df = pd.DataFrame(stats_rows)
    lineup_df = pd.DataFrame(lineup_rows)

    # Attach round/date from fixtures (detail pages use split-round league ids).
    if not stats_df.empty:
        meta = fixtures[["match_id", "round", "kst_date"]].astype({"match_id": str})
        stats_df = stats_df.drop(columns=["round"]).merge(meta, on="match_id", how="left")
        front = ["season", "round", "kst_date", "match_id"]
        stats_df = stats_df[front + [c for c in stats_df.columns if c not in front]]
        stats_df = stats_df.sort_values(["round", "kst_date", "match_id"]).reset_index(drop=True)
    return stats_df, lineup_df


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def save_csv(df, season, name):
    out_dir = os.path.join(OUTPUT_ROOT, str(season))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def main(seasons=SEASONS, headless=True, details=True):
    driver = build_driver(headless=headless)
    all_fixtures = []
    try:
        for i, season in enumerate(seasons):
            if i:
                time.sleep(random.uniform(3, 6))  # rate limiting
            df = scrape_fixtures(driver, season)
            path = save_csv(df, season, "matches.csv")
            print(f"Saved: {path} ({len(df)} matches, "
                  f"{df['round'].nunique()} rounds, {int(df['finished'].sum())} finished)")
            all_fixtures.append(df)
    finally:
        driver.quit()

    if not details:
        return

    fixtures = pd.concat(all_fixtures, ignore_index=True)
    scrape_details(fixtures, headless=headless)

    for season in seasons:
        stats_df, lineup_df = build_detail_tables(season, fixtures)
        p1 = save_csv(stats_df, season, "match_stats.csv")
        p2 = save_csv(lineup_df, season, "lineups.csv")
        print(f"Saved: {p1} ({len(stats_df)} rows), {p2} ({len(lineup_df)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape K League 1 data from FotMob")
    parser.add_argument("--seasons", type=int, nargs="+", default=SEASONS)
    parser.add_argument("--no-details", action="store_true",
                        help="only scrape fixtures/results")
    parser.add_argument("--show-browser", action="store_true",
                        help="run Chrome with a visible window")
    args = parser.parse_args()
    main(seasons=args.seasons, headless=not args.show_browser, details=not args.no_details)
