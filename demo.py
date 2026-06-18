import requests

TOKEN = "ws_a7dbbf9b62ea50bcaeedb732188407c1f34175b0110a656371cbdcfbd2ed74fe"
URL = "https://api.wingobot.com/v2/1-min-game-history"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/json"
}

try:
    response = requests.get(URL, headers=headers, timeout=15)
    response.raise_for_status()

    data = response.json()

    if data.get("success"):

        current = data.get("current", {})
        print("\n===== CURRENT PERIOD =====")
        print("Period :", current.get("issueNumber"))
        print("Number :", current.get("number"))
        print("Colour :", current.get("colour"))
        print("Premium:", current.get("premium"))
        print()

        print("===== HISTORY =====")

        for row in data.get("history", []):
            print(
                f'{row.get("issueNumber")} | '
                f'{row.get("number")} | '
                f'{row.get("colour")} | '
                f'{row.get("premium")} | '
                f'{row.get("sum")}'
            )

        stats = data.get("stats", {})

        print("\n===== STATS =====")
        print("Fetched      :", stats.get("fetched"))
        print("Last Updated :", stats.get("last_updated"))

    else:
        print("Error:", data.get("error", "Unknown error"))

except requests.exceptions.Timeout:
    print("Request timed out.")

except requests.exceptions.RequestException as e:
    print("Request failed:", e)

except Exception as e:
    print("Unexpected error:", e)