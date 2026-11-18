"""베트맨 프로토 승부식(G101) 회차별 데이터를 수집해 K리그 승무패 배당/결과를 CSV로 저장한다.

- 원본 응답은 betman_raw/{gmTs}.json 에 캐시 (재실행 시 네트워크 요청 생략)
- 추출 결과는 betman_kleague.csv
"""
import csv
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

URL = "https://www.betman.co.kr/buyPsblGame/gameInfoInq.do"
HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.betman.co.kr/",
}
RAW_DIR = Path("betman_raw")
OUT_CSV = Path("betman_kleague.csv")
KST = timezone(timedelta(hours=9))

YEARS = ["2023", "2024", "2025", "2026"]
MAX_ROUND = 200          # 연간 최대 회차 탐색 범위
MAX_MISSES = 5           # 연속으로 회차가 없으면 해당 연도 종료
SLEEP_SEC = 0.5          # 서버 부하 방지
RETRIES = 3              # 네트워크 오류 시 재시도 횟수

RESULT_MAP = {"0": "H", "1": "D", "2": "A"}  # 0=홈승, 1=무, 2=원정승 (추정, 검증 필요)

FIELDS = [
    "gmTs", "matchSeq", "date", "leagueName", "home", "away",
    "odds_home", "odds_draw", "odds_away", "score", "home_goals", "away_goals",
    "gameResult", "result",
]


def fetch_round(session, gm_ts):
    """한 회차 응답(JSON)을 반환. 캐시가 있으면 캐시 사용. 회차가 없으면 None."""
    cache = RAW_DIR / f"{gm_ts}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    payload = {
        "gameYear": "",
        "gmId": "G101",
        "gmTs": gm_ts,
        "_sbmInfo": {"_sbmInfo": {"debugMode": "false"}},
    }
    for attempt in range(RETRIES):
        try:
            resp = session.post(URL, json=payload, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            break
        except (requests.RequestException, ValueError):
            if attempt == RETRIES - 1:
                raise
            time.sleep(2 * (attempt + 1))
    time.sleep(SLEEP_SEC)

    # 존재하지 않는 회차면 다른 회차가 오거나 일정이 비어 있음
    lottery = data.get("currentLottery") or {}
    if lottery.get("gmTs") != gm_ts or not (data.get("compSchedules") or {}).get("datas"):
        return None

    RAW_DIR.mkdir(exist_ok=True)
    cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


SEEN_LEAGUES = set()  # 처음 보는 축구 리그명을 로그로 남겨 K리그 표기 확인용


def is_kleague(row):
    if row.get("itemCode") != "SC":
        return False
    name = f"{row.get('leagueName') or ''} {row.get('leagueShortName') or ''}".strip()
    if name not in SEEN_LEAGUES:
        SEEN_LEAGUES.add(name)
        print(f"    새 축구 리그: {name}")
    # 베트맨 표기: 'K리그1' / 'K리그2' / 'K승강PO' → 1부 리그만 사용
    compact = name.upper().replace(" ", "")
    return "K리그1" in compact


def extract_kleague(data, gm_ts):
    """compSchedules 에서 K리그 '승무패' 행만 추출."""
    cs = data["compSchedules"]
    out = []
    for values in cs["datas"]:
        row = dict(zip(cs["keys"], values))
        # 2023~24 응답은 betTypNm 이 None 이라 handi 코드로 판별 (0=승무패, 2=핸디캡, 9=언더오버, 27=홀짝)
        if not is_kleague(row) or row.get("handi") != 0:
            continue
        # From 2026 (round 260046) a first-half W/D/L market ('축구 전반 승무패') is also
        # sold with handi=0; keep the full-time market only.
        if "전반" in (row.get("betNm") or ""):
            continue

        score = row.get("mchScore") or ""
        hg, ag = (score.split(":") + [""])[:2] if ":" in score else ("", "")
        date = datetime.fromtimestamp(row["gameDate"] / 1000, KST).strftime("%Y-%m-%d %H:%M")
        out.append({
            "gmTs": gm_ts,
            "matchSeq": row.get("matchSeq"),
            "date": date,
            "leagueName": row.get("leagueName"),
            "home": row.get("homeName"),
            "away": row.get("awayName"),
            "odds_home": row.get("winAllot"),
            "odds_draw": row.get("drawAllot"),
            "odds_away": row.get("loseAllot"),
            "score": score,
            "home_goals": hg,
            "away_goals": ag,
            "gameResult": row.get("gameResult"),
            "result": RESULT_MAP.get(str(row.get("gameResult")), ""),
        })
    return out


def main():
    # --offline: rebuild the CSV from betman_raw/ only (no network requests)
    if "--offline" in sys.argv:
        all_rows = []
        for path in sorted(RAW_DIR.glob("*.json")):
            gm_ts = int(path.stem)
            all_rows.extend(extract_kleague(json.loads(path.read_text(encoding="utf-8")), gm_ts))
        save_csv(all_rows)
        return

    session = requests.Session()
    all_rows = []

    for year in YEARS:
        misses = 0
        for n in range(1, MAX_ROUND + 1):
            gm_ts = int(f"{year[2:]}{n:04d}")
            try:
                data = fetch_round(session, gm_ts)
            except (requests.RequestException, ValueError) as e:
                # 네트워크 오류는 '회차 없음'으로 세지 않음 (캐시 안 됐으므로 재실행 시 다시 시도)
                print(f"[{gm_ts}] 요청 실패(재실행 시 재시도): {e}")
                continue

            if data is None:
                misses += 1
                if misses >= MAX_MISSES:
                    print(f"{year}: 회차 {n} 부근에서 종료")
                    break
                continue
            misses = 0

            rows = extract_kleague(data, gm_ts)
            all_rows.extend(rows)
            print(f"[{gm_ts}] K리그 승무패 {len(rows)}경기")

    save_csv(all_rows)


def save_csv(all_rows):
    # 같은 경기가 여러 회차에 걸쳐 발매될 수 있으므로 (날짜, 홈, 원정) 기준 마지막 회차만 유지
    dedup = {}
    for r in all_rows:
        dedup[(r["date"], r["home"], r["away"])] = r
    rows = sorted(dedup.values(), key=lambda r: (r["date"], r["home"]))

    with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"저장 완료: {OUT_CSV} ({len(rows)}경기)")


if __name__ == "__main__":
    main()
