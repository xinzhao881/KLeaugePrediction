
import pandas as pd
from pathlib import Path
import re
import html

# ==========================================
# 1. 경로 설정
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "kleague_csv"
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)
META_COLUMNS = ["Rnd","대회","H/A"]

# ==========================================
# 3. CSV 읽기
# ==========================================
def read_csv(path):
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp949")

    # HTML entity 디코딩 및 컬럼명 정리
    df.columns = [
        html.unescape(str(col)).replace("**", "").strip()
        for col in df.columns
    ]

    return df

def add_match_key(df):
    """라운드와 두 팀 이름으로 경기 식별 키 생성"""
    match_teams = df["대회"].apply(lambda x: x.split("vs"))

    df["TeamA"] = match_teams.apply(lambda x: x[0])
    df["TeamB"] = match_teams.apply(lambda x: x[1])

    df["MatchKey"] = df.apply(
        lambda row: (
            str(row["Rnd."]).strip(),
            *sorted([row["TeamA"], row["TeamB"]])
        ),
        axis=1
    )

    return df


def process_season(year_dir):
    year = year_dir.name
    print(f"\n[{year}] Processing...")

    team_dfs = []

    # 해당 시즌의 구단 CSV만 읽기
    for csv_path in year_dir.glob("*.csv"):
        team_name = csv_path.stem

        df = read_csv(csv_path)
        df["Team"] = team_name

        team_dfs.append(df)

    if not team_dfs:
        return

    # 시즌 내부에서만 임시로 합쳐서 상대팀을 찾음
    all_teams = pd.concat(team_dfs, ignore_index=True)

    all_teams = add_match_key(all_teams)

    # 상대팀 이름 추출
    def find_opponent(row):
        if row["Team"] == row["TeamA"]:
            return row["TeamB"]
        elif row["Team"] == row["TeamB"]:
            return row["TeamA"]
        else:
            raise ValueError(
                f"팀 이름 불일치: {row['Team']} / {row['대회']}"
            )

    all_teams["Opponent"] = all_teams.apply(
        find_opponent,
        axis=1
    )

    # 상대팀 데이터 준비
    opponent_data = all_teams.drop(
        columns=[
            "TeamA",
            "TeamB",
            "H/A",
            "Opponent"  # 기존 Opponent 컬럼 제거
        ],
        errors="ignore"
    ).copy()

    # Team을 Opponent로 변경
    opponent_data = opponent_data.rename(
        columns={"Team": "Opponent"}
    )

    # 상대팀 통계 컬럼에 접두사 추가
    opponent_data = opponent_data.rename(
        columns={
            col: f"상대_{col}"
            for col in opponent_data.columns
            if col not in ["Rnd.", "MatchKey", "Opponent"]
        }
    )

    # 필요한 컬럼만 선택
    opponent_data = opponent_data[
        ["Rnd.", "MatchKey", "Opponent"]
        + [
            col for col in opponent_data.columns
            if col.startswith("상대_")
        ]
    ]

        # 중복 경기 확인
    duplicate = opponent_data.duplicated(
        subset=["Rnd.", "MatchKey", "Opponent"],
        keep=False
    )

    if duplicate.any():
        print(f"[WARNING] {year}: 중복된 상대팀 경기 데이터 발견")
        print(
            opponent_data.loc[
                duplicate,
                ["Rnd.", "MatchKey", "Opponent"]
            ]
        )
        raise ValueError("경기 매칭 키가 중복됩니다.")

    # 구단별로 다시 분리하여 상대팀 데이터 추가
    output_dir = OUTPUT_DIR / year
    output_dir.mkdir(parents=True, exist_ok=True)

    for team_name, team_df in all_teams.groupby("Team"):

        result = team_df.merge(
            opponent_data,
            on=["Rnd.", "MatchKey", "Opponent"],
            how="left",
            validate="one_to_one",
            indicator=True
        )

        unmatched = (result["_merge"] != "both").sum()

        if unmatched:
            print(
                f"[WARNING] {year}/{team_name}: "
                f"{unmatched}경기 상대팀 데이터 매칭 실패"
            )

        result = result.drop(
            columns=[
                "_merge",
                "TeamA",
                "TeamB",
                "MatchKey",
                "Opponent"
            ],
            errors="ignore"
        )

        result.to_csv(
            output_dir / f"{team_name}.csv",
            index=False,
            encoding="utf-8-sig"
        )

        print(f"Saved: {year}/{team_name}.csv")


# ==========================================
# 4. K League + FotMob merge & pre-match features
# ==========================================
FOTMOB_DIR = BASE_DIR / "fotmob_csv"

# Centralized team mapping: FotMob team_id -> K League portal short name.
# Use ids, not names: FotMob shows the name used at that time
# (e.g. 'Ulsan Hyundai' -> 'Ulsan HD FC', 'Jeju United' -> 'Jeju SK').
TEAM_ID_TO_KR = {
    133896: "울산",
    46038: "전북",
    109373: "포항",
    92630: "서울",
    164734: "강원",
    245905: "광주",
    133901: "김천",
    133897: "대구",
    133900: "대전",
    187951: "수원FC",
    429440: "안양",
    133898: "제주",
    133895: "인천",
    88517: "수원",
    429441: "부천",
    6614: "성남",
}

# K League stats are the primary source for event stats that both sources have.
KL_STATS = [
    "슈팅", "유효 슈팅", "PA내 슈팅", "키패스",
    "패스_성공%", "공격진영 패스_시도", "크로스_시도",
    "경합 지상_성공%", "경합 공중_성공%",
    "태클_성공", "인터셉트", "획득", "볼미스", "파울", "코너킥",
]
# FotMob-only stats (not available in the K League portal data).
FM_STATS = [
    "expected_goals", "expected_goals_non_penalty", "expected_goals_on_target",
    "ballpossesion", "big_chance", "touches_opp_box", "keeper_saves", "team_rating",
]
# Opponent's stats in the same match = what the team conceded.
AGAINST_STATS = [
    "kl_슈팅", "kl_유효 슈팅", "kl_PA내 슈팅", "kl_키패스",
    "fm_expected_goals", "fm_expected_goals_non_penalty",
    "fm_big_chance", "fm_touches_opp_box",
]
ROLL_WINDOW = 5
REST_DAYS_CAP = 30  # off-season gaps are not informative beyond this


def team_kr(team_id):
    name = TEAM_ID_TO_KR.get(int(team_id))
    if name is None:
        raise ValueError(f"Unmapped FotMob team_id: {team_id} (add to TEAM_ID_TO_KR)")
    return name


# Other spellings (K League portal full names, Betman, user input) -> short name.
# Keys are compared without spaces, case-insensitive.
TEAM_ALIASES = {
    "울산HD": "울산", "울산현대": "울산", "울산HDFC": "울산",
    "전북현대": "전북", "전북현대모터스": "전북",
    "포항스틸러스": "포항",
    "FC서울": "서울",
    "강원FC": "강원",
    "광주FC": "광주",
    "김천상무": "김천", "김천상무FC": "김천",
    "대구FC": "대구",
    "대전하나": "대전", "대전하나시티즌": "대전",
    "수원삼성": "수원", "수원삼성블루윙즈": "수원",
    "FC안양": "안양",
    "제주SK": "제주", "제주유나이티드": "제주", "제주SKFC": "제주",
    "인천유나이티드": "인천",
    "부천FC": "부천", "부천FC1995": "부천",
    "성남FC": "성남",
}


def normalize_team_name(name):
    """Map any known spelling to the K League short name (e.g. '울산 HD' -> '울산')."""
    key = str(name).replace(" ", "").upper()
    canonical = {n.upper(): n for n in TEAM_ID_TO_KR.values()}
    if key in canonical:
        return canonical[key]
    aliases = {k.replace(" ", "").upper(): v for k, v in TEAM_ALIASES.items()}
    if key in aliases:
        return aliases[key]
    raise ValueError(f"Unknown team name: {name!r} (known: {sorted(set(TEAM_ID_TO_KR.values()))})")


def fotmob_seasons():
    return sorted(
        int(p.name) for p in FOTMOB_DIR.iterdir()
        if p.is_dir() and p.name.isdigit() and (p / "matches.csv").exists()
    )


# FotMob columns that are not team performance stats
FM_NON_STATS = {"team", "team_id", "score", "formation", "coach", "starter_market_value"}


def load_fotmob_long(seasons, fm_stats=None):
    """One row per (match, team) from FotMob fixtures + team stats.

    fm_stats: FotMob stat names to load (None = every numeric team stat).
    """
    frames = []
    for season in seasons:
        fixtures = pd.read_csv(FOTMOB_DIR / str(season) / "matches.csv",
                               encoding="utf-8-sig", dtype={"match_id": str})
        stats_path = FOTMOB_DIR / str(season) / "match_stats.csv"
        if stats_path.exists():
            stats = pd.read_csv(stats_path, encoding="utf-8-sig", dtype={"match_id": str})
            if fm_stats is None:
                fm_stats = [
                    c[5:] for c in stats.columns
                    if c.startswith("home_") and c[5:] not in FM_NON_STATS
                    and pd.api.types.is_numeric_dtype(stats[c])
                ]
            keep = ["match_id"] + [
                f"{side}_{s}" for side in ("home", "away") for s in fm_stats
                if f"{side}_{s}" in stats.columns
            ]
            fixtures = fixtures.merge(stats[keep], on="match_id", how="left")
        fixtures = fixtures[~fixtures["cancelled"]]

        for side, opp in (("home", "away"), ("away", "home")):
            d = pd.DataFrame({
                "season": fixtures["season"],
                "round": fixtures["round"],
                "match_id": fixtures["match_id"],
                "utc_time": pd.to_datetime(fixtures["utc_time"], utc=True),
                "kst_date": pd.to_datetime(fixtures["kst_date"]),
                "kst_time": fixtures["kst_time"],
                "finished": fixtures["finished"],
                "team_id": fixtures[f"{side}_team_id"],
                "opp_id": fixtures[f"{opp}_team_id"],
                "is_home": int(side == "home"),
                "goals_for": fixtures[f"{side}_goals"],
                "goals_against": fixtures[f"{opp}_goals"],
            })
            for s in fm_stats or []:
                d[f"fm_{s}"] = pd.to_numeric(fixtures.get(f"{side}_{s}"), errors="coerce")
            frames.append(d)

    long_df = pd.concat(frames, ignore_index=True)
    long_df["team"] = long_df["team_id"].map(team_kr)
    long_df["opponent"] = long_df["opp_id"].map(team_kr)
    return long_df


KL_NON_STATS = {"Rnd.", "대회", "H/A", "득점", "도움", "탈압박"}  # goals/assists = target leak


def load_kleague_long(kl_stats=None):
    """One row per (match, team) from raw K League team CSVs.

    kl_stats: K League column names to load (None = every numeric stat column).
    """
    frames = []
    for year_dir in sorted(DATA_DIR.iterdir()):
        if not (year_dir.is_dir() and year_dir.name.isdigit()):
            continue
        for csv_path in year_dir.glob("*.csv"):
            df = read_csv(csv_path)
            teams = df["대회"].str.split("vs")
            out = pd.DataFrame({
                "season": int(year_dir.name),
                "round": df["Rnd."].astype(int),
                "team": csv_path.stem,
                "opponent": [
                    b if a == csv_path.stem else a for a, b in teams
                ],
            })
            cols = kl_stats if kl_stats is not None else [
                c for c in df.columns
                if c not in KL_NON_STATS and pd.api.types.is_numeric_dtype(df[c])
            ]
            for col in cols:
                if col not in df.columns:
                    print(f"[WARNING] {year_dir.name}/{csv_path.name}: missing '{col}'")
                out[f"kl_{col}"] = pd.to_numeric(df.get(col), errors="coerce")
            frames.append(out)
    return pd.concat(frames, ignore_index=True)


def build_team_match_table(kl_stats=KL_STATS, fm_stats=FM_STATS, against_stats=AGAINST_STATS):
    """Merge FotMob and K League per (match, team) and attach opponent stats.

    Pass kl_stats/fm_stats=None to load every available stat (for analysis).
    """
    seasons = fotmob_seasons()
    fm = load_fotmob_long(seasons, fm_stats)
    kl = load_kleague_long(kl_stats)

    merged = fm.merge(kl, on=["season", "round", "team", "opponent"],
                      how="left", validate="one_to_one", indicator=True)

    # Report finished matches that have no K League stats (e.g. season not scraped)
    kl_seasons = set(kl["season"])
    missing = merged[(merged["_merge"] == "left_only") & merged["finished"]
                     & merged["season"].isin(kl_seasons)]
    if len(missing):
        print(f"[WARNING] {len(missing)} FotMob team-matches without K League stats")
        print(missing[["season", "round", "team", "opponent"]].head(10))
    unmatched_kl = len(kl) - (merged["_merge"] == "both").sum()
    if unmatched_kl:
        print(f"[WARNING] {unmatched_kl} K League rows did not match FotMob")
    for season in sorted(set(seasons) - kl_seasons):
        print(f"[INFO] {season}: no K League data, kl_* features will be NaN")
    merged = merged.drop(columns="_merge")

    # Opponent's stats in the same match (= conceded)
    opp = merged[["match_id", "team_id"] + list(against_stats)].rename(
        columns={"team_id": "opp_id", **{c: f"opp_{c}" for c in against_stats}}
    )
    merged = merged.merge(opp, on=["match_id", "opp_id"], how="left", validate="one_to_one")

    finished = merged["finished"]
    merged["points"] = pd.NA
    merged.loc[finished, "points"] = (
        (merged.loc[finished, "goals_for"] > merged.loc[finished, "goals_against"]) * 3
        + (merged.loc[finished, "goals_for"] == merged.loc[finished, "goals_against"]) * 1
    )
    merged["points"] = pd.to_numeric(merged["points"])
    merged["goal_diff"] = merged["goals_for"] - merged["goals_against"]
    merged["xg_diff"] = merged["fm_expected_goals"] - merged["opp_fm_expected_goals"]
    merged["goals_minus_xg"] = merged["goals_for"] - merged["fm_expected_goals"]

    return merged.sort_values(["team_id", "utc_time"]).reset_index(drop=True)


def load_starting_xi(seasons):
    frames = []
    for season in seasons:
        path = FOTMOB_DIR / str(season) / "lineups.csv"
        if path.exists():
            lu = pd.read_csv(path, encoding="utf-8-sig", low_memory=False,
                             dtype={"match_id": str})
            frames.append(lu.loc[lu["role"] == "starter", ["match_id", "team_id", "player_id"]])
    if not frames:
        return pd.DataFrame(columns=["match_id", "team_id", "xi"])
    xi = pd.concat(frames).groupby(["match_id", "team_id"])["player_id"].apply(frozenset)
    return xi.rename("xi").reset_index()


def prior_state(df, cols, group_cols, window=None):
    """Value of a rolling/expanding mean as of BEFORE each match.

    The mean is computed over finished matches only (state AFTER each match).
    Each row then takes the state of the last finished match strictly before
    it in the same group, so unplayed future fixtures are covered too.
    `df` must be sorted chronologically within each group.
    """
    played = df[df["finished"]]
    grouped = played.groupby(group_cols)[cols]
    if window:
        state = grouped.rolling(window, min_periods=1).mean()
    else:
        state = grouped.expanding().mean()
    state = state.reset_index(level=list(range(len(group_cols))), drop=True)

    # Row label of the last finished match before each row (within the group)
    keys = [df[c] for c in group_cols]
    played_label = pd.Series(df.index, index=df.index).where(df["finished"])
    last_played = played_label.groupby(keys).shift(1).groupby(keys).ffill()

    out = state.reindex(last_played.values)
    out.index = df.index
    return out[cols]


def compute_context_features(tm):
    """Schedule/context features known before kickoff (sorted by team, time)."""
    ctx = pd.DataFrame(index=tm.index)

    # Matches played this season before this match
    ctx["season_matches_played"] = (
        tm["finished"].astype(int).groupby([tm["team_id"], tm["season"]]).cumsum()
        - tm["finished"].astype(int)
    )

    # Rest days since the previous finished match (capped)
    last_date = tm["kst_date"].where(tm["finished"])
    last_date = last_date.groupby(tm["team_id"]).shift(1).groupby(tm["team_id"]).ffill()
    ctx["rest_days"] = (tm["kst_date"] - last_date).dt.days.clip(upper=REST_DAYS_CAP)

    ctx["is_weekend"] = tm["kst_date"].dt.dayofweek.isin([5, 6]).astype(int)
    ctx["kickoff_hour"] = pd.to_numeric(tm["kst_time"].str[:2], errors="coerce")

    # Rotation: starters changed vs previous match. Uses the announced XI of the
    # target match (published ~1h before kickoff) -> near-kickoff feature only.
    xi = load_starting_xi(sorted(tm["season"].unique()))
    with_xi = tm[["match_id", "team_id"]].merge(xi, on=["match_id", "team_id"], how="left")
    prev_xi = with_xi.groupby("team_id")["xi"].shift(1)
    ctx["xi_changes"] = pd.to_numeric(pd.Series([
        11 - len(cur & prev) if isinstance(cur, frozenset) and isinstance(prev, frozenset)
        else None
        for cur, prev in zip(with_xi["xi"], prev_xi)
    ], index=tm.index))
    return ctx


GROUP_TAG = {"base": "", "attack": "att_", "defense": "def_", "strength": "str_"}


def add_prematch_features(tm, stat_groups, context_cols):
    """Add pre-match features to the team-match table (sorted by team, time).

    stat_groups: {team-match column: 'base' | 'attack' | 'defense'}; each column
                 gets a rolling and a season-to-date pre-match mean.
    context_cols: context features to keep (see compute_context_features).
    """
    hist_cols = list(stat_groups)
    names = [f"{GROUP_TAG[stat_groups[c]]}{c}" for c in hist_cols]

    roll = prior_state(tm, hist_cols, ["team_id"], window=ROLL_WINDOW)
    roll.columns = [f"roll{ROLL_WINDOW}_{n}" for n in names]
    season = prior_state(tm, hist_cols, ["team_id", "season"])
    season.columns = [f"season_{n}" for n in names]

    ctx = compute_context_features(tm)[list(context_cols)]
    feats = pd.concat([roll, season, ctx], axis=1)
    return pd.concat([tm, feats.set_index(tm.index)], axis=1), list(feats.columns)


def build_match_features(tm, feature_cols):
    """Pivot the team-match table into one row per match (home_/away_/diff_)."""
    meta = ["season", "round", "match_id", "kst_date", "kst_time", "finished"]
    home = tm[tm["is_home"] == 1]
    away = tm[tm["is_home"] == 0]

    left = home[meta + ["team", "team_id", "goals_for", "goals_against"] + feature_cols].rename(
        columns={"team": "home_team", "team_id": "home_team_id",
                 "goals_for": "home_goals", "goals_against": "away_goals",
                 **{c: f"home_{c}" for c in feature_cols}}
    )
    right = away[["match_id", "team", "team_id"] + feature_cols].rename(
        columns={"team": "away_team", "team_id": "away_team_id",
                 **{c: f"away_{c}" for c in feature_cols}}
    )
    matches = left.merge(right, on="match_id", how="inner", validate="one_to_one")

    roll_cols = [c for c in feature_cols if c.startswith(f"roll{ROLL_WINDOW}_")]
    diffs = pd.DataFrame({
        f"diff_{c}": matches[f"home_{c}"] - matches[f"away_{c}"]
        for c in roll_cols + [c for c in ["rest_days"] if c in feature_cols]
    })
    matches = pd.concat([matches, diffs], axis=1)

    hg, ag = matches["home_goals"], matches["away_goals"]
    matches["result"] = None
    matches.loc[matches["finished"] & (hg > ag), "result"] = "H"
    matches.loc[matches["finished"] & (hg == ag), "result"] = "D"
    matches.loc[matches["finished"] & (hg < ag), "result"] = "A"

    front = ["season", "round", "kst_date", "kst_time", "match_id",
             "home_team", "away_team", "home_team_id", "away_team_id",
             "finished", "home_goals", "away_goals", "result"]
    matches = matches[front + [c for c in matches.columns if c not in front]]
    return matches.sort_values(["kst_date", "kst_time", "match_id"]).reset_index(drop=True)


STAT_GROUPS_PATH = BASE_DIR / "feature_importance" / "stat_groups.csv"
# Always-kept rolling stats (team strength / form)
BASE_STATS = ["goals_for", "goals_against", "points", "xg_diff"]
CONTEXT_FEATURES = ["season_matches_played", "rest_days", "is_weekend",
                    "kickoff_hour", "xi_changes"]


def load_feature_config():
    """Read stat groups selected by stat_correlation.py (falls back to defaults)."""
    if STAT_GROUPS_PATH.exists():
        cfg = pd.read_csv(STAT_GROUPS_PATH, encoding="utf-8-sig")
        is_ctx = cfg["group"] == "context"
        groups = dict(zip(cfg.loc[~is_ctx, "stat"], cfg.loc[~is_ctx, "group"]))
        context = cfg.loc[is_ctx, "stat"].tolist()
        print(f"[INFO] Feature config: {STAT_GROUPS_PATH.name} "
              f"({len(groups)} stats, {len(context)} context)")
    else:
        print("[WARNING] stat_groups.csv not found, run stat_correlation.py; using defaults")
        groups = {f"fm_{s}": "attack" for s in FM_STATS}
        groups.update({f"kl_{s}": "attack" for s in KL_STATS})
        groups.update({f"opp_{s}": "defense" for s in AGAINST_STATS})
        context = CONTEXT_FEATURES
    stat_groups = {c: "base" for c in BASE_STATS}
    stat_groups.update({c: g for c, g in groups.items() if c not in stat_groups})
    return stat_groups, context


def required_stats(stat_groups):
    """Raw stats to load for the selected team-match columns."""
    own = {c for c in stat_groups if c.startswith(("kl_", "fm_"))}
    against = {c[4:] for c in stat_groups if c.startswith("opp_")}
    against.add("fm_expected_goals")  # needed for xg_diff
    own |= against | {"fm_expected_goals"}
    kl_stats = sorted(c[3:] for c in own if c.startswith("kl_"))
    fm_stats = sorted(c[3:] for c in own if c.startswith("fm_"))
    return kl_stats, fm_stats, sorted(against)


def build_features():
    print("\n[Merge] K League + FotMob")
    stat_groups, context = load_feature_config()
    kl_stats, fm_stats, against = required_stats(stat_groups)
    tm = build_team_match_table(kl_stats, fm_stats, against)
    tm, feature_cols = add_prematch_features(tm, stat_groups, context)
    matches = build_match_features(tm, feature_cols)

    for season, df in matches.groupby("season"):
        out_dir = OUTPUT_DIR / str(season)
        out_dir.mkdir(parents=True, exist_ok=True)
        tm_season = tm[tm["season"] == season].drop(columns=["utc_time"])
        tm_season.to_csv(out_dir / "team_match_stats.csv", index=False, encoding="utf-8-sig")
        df.to_csv(out_dir / "match_features.csv", index=False, encoding="utf-8-sig")
        print(f"Saved: {season}/match_features.csv ({len(df)} matches, "
              f"{int(df['finished'].sum())} finished), {season}/team_match_stats.csv")


def main():
    for year_dir in sorted(DATA_DIR.iterdir()):
        if year_dir.is_dir() and year_dir.name.isdigit():
            process_season(year_dir)
    build_features()


if __name__ == "__main__":
    main()