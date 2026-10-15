import requests
import json

URL = "https://portal.kleague.com/data/addition/team/additionTeamDataBySeason.do"

# 알려주신 menuVoJson을 JSON 객체로 작성합니다.
menu_info = {
    "menuCd": "0197",
    "menuNm": "년도별 팀기록",
    "parentMenuCd": "0195",
    "menuUrl": "/data/addition/team/additionTeamDataBySeason.do",
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

# POST 데이터
payload = {
    "menuVoJson": json.dumps(menu_info, ensure_ascii=False),
    "selectYear": "2025",
    "selectMeetSeq" : "1",
    "selectTeamId": ""
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

def main():
    session = requests.Session()
    session.headers.update(headers)

    response = session.post(URL, data=payload, timeout=30)

    print("=" * 80)
    print("HTTP Status:", response.status_code)
    print("Response URL:", response.url)
    print("Response Content-Type:", response.headers.get("Content-Type"))
    print("Response Length:", len(response.text))
    print("=" * 80)

    response.raise_for_status()

    # JSON 응답이면 보기 좋게 출력하고, 아니면 원문 전체를 출력합니다.
    try:
        result = response.json()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except requests.exceptions.JSONDecodeError:
        print(response.text)

if __name__ == "__main__":
    main()