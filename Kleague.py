from io import StringIO
import os

import requests
import json
from bs4 import BeautifulSoup
import pandas as pd

URL = "https://portal.kleague.com/data/addition/team/overallAdditionData.do"
xg_url="https://portal.kleague.com/data/addition/team/xgPointTeamRank.do"


def flatten_columns(columns):
    """Flatten pandas' multi-row table headers into readable CSV column names."""
    flattened = []
    for column in columns:
        parts = [
            str(part).strip()
            for part in (column if isinstance(column, tuple) else (column,))
            if str(part).strip() and not str(part).startswith("Unnamed:")
        ]

        # Simple columns such as ('년도', '년도') should only appear once.
        unique_parts = []
        for part in parts:
            if not unique_parts or part != unique_parts[-1]:
                unique_parts.append(part)

        name = "_".join(unique_parts) if unique_parts else "unnamed"
        flattened.append(name)

    # Avoid duplicate column names if the source table has any.
    counts = {}
    unique_names = []
    for name in flattened:
        counts[name] = counts.get(name, 0) + 1
        unique_names.append(name if counts[name] == 1 else f"{name}_{counts[name]}")
    return unique_names

def get_team_list(year):
    url = "https://portal.kleague.com/data/addition/common/searchMeetAndTeamInfoByYear.do"
    menu_info = {"menuCd":"0197",
                 "menuNm":"년도별 팀기록",
                 "parentMenuCd":"0195",
                 "menuUrl":"/data/addition/team/additionTeamDataBySeason.do",
                 "chartYn":"N",
                 "btnQry":"Y",
                 "btnExl":"N",
                 "btnSave":"N",
                 "btnAdd":"N",
                 "btnDel":"N",
                 "btnPrt":"N",
                 "btnClr":"N",
                 "btnDwn":"N",
                 "searchChildrenYn":"Y",
                 "helpFileSize":0,
                 "regUserId":"platform",
                 "updUserId":"platform",
                 "searchCondition":"",
                 "searchKeyword":""
                 }
    payload = {
        "menuVoJson": json.dumps(menu_info, ensure_ascii=False),
        "selectYear": year,
        "SelectTeamId":""
    }
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/130.0.0.0 Safari/537.36"
        ),
        "Referer": "https://portal.kleague.com/",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "X-Requested-With": "XMLHttpRequest"
    }
    response = requests.post(url, data=payload, headers=headers)
    data = response.json()
    team_list = data.get("teamNameList", [])
    if team_list:
        return [{"teamSName": team["teamSName"], "teamId": team["teamId"]} for team in team_list]


def get_team_stats(year):
    # 알려주신 menuVoJson을 JSON 객체로 작성합니다.
    team_list = get_team_list(year)

    menu_info = {
        "menuCd": "0196",
        "menuNm": "경기별 팀기록",
        "parentMenuCd": "0195",
        "menuUrl": "/data/addition/team/overallAdditionData.do",
        "uriPath": "data/addition/team/",
        "chartYn": "N",
        "btnQry": "Y",
        "btnExl": "N",
        "btnSave": "N",
        "btnAdd": "N",
        "btnDel": "N",
        "btnPrt": "N",
        "btnClr": "N",
        "btnDwn": "N",
        "searchChildrenYn": "Y",
        "helpFileSize": 0,
        "regUserId": "platform",
        "updUserId": "platform",
        "searchCondition": "",
        "searchKeyword": ""
    }
    for team in team_list:
        # POST 데이터
        payload = {
            "menuVoJson": json.dumps(menu_info, ensure_ascii=False),
            "selectYear": year,  # 지정된 연도 선택
            "selectMeetSeq" : "1",
            "selectTeamId": team["teamId"]
        }

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/130.0.0.0 Safari/537.36"
            ),
            "Referer": "https://portal.kleague.com/",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "X-Requested-With": "XMLHttpRequest"
        }
        # POST 요청을 보내고 응답을 받습니다.
        response = requests.post(URL, data=payload, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")
        target = soup.find("div", class_="searchDataset sub-team-table")
        tables = pd.read_html(StringIO(str(target)))
        df = tables[0]
        df.columns = flatten_columns(df.columns)
        file_path = f"./kleague_csv/{year}/{team['teamSName']}.csv"
        folder_path = f"./kleague_csv/{year}"
        os.makedirs(folder_path, exist_ok=True)
        df.to_csv(file_path, index=False, encoding="utf-8-sig")
# def prune_teams():
#     #합계,평균 행 제거하고 opponent_team 컬럼 추가해서 csv돌아가면서 수정
 
def main():
    years = ["2023", "2024", "2025"]
    for year in years:
        get_team_stats(year)
if __name__ == "__main__":
    main()
