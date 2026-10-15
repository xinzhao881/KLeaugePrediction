import requests

url = "https://www.betman.co.kr/buyPsblGame/gameInfoInq.do"

payload = {
    "gmId": "G101",
    "gmTs": 260121,
    "gameYear": "",
    "_sbmInfo": {
        "_sbmInfo": {
            "debugMode": "false"
        }
    }
}

headers = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.betman.co.kr/",
}

response = requests.post(
    url,
    json=payload,
    headers=headers,
    timeout=15,
)

response.raise_for_status()

print("Status:", response.status_code)
print("Content-Type:", response.headers.get("Content-Type"))
print(response.text[:5000])