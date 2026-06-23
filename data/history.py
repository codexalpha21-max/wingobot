import json
import time
from datetime import datetime, UTC

import requests

HEADERS = {
    "accept": "application/json",
    "user-agent": "Mozilla/5.0",
}


def get_current_period_1min():
    now = datetime.now(UTC)
    date = now.strftime("%Y%m%d")
    start = datetime(now.year, now.month, now.day, 0, 0, 0, tzinfo=UTC)
    minutes = int((now - start).total_seconds() // 60)
    return f"{date}1000{10001 + minutes}"


def nearby_periods(current_period, lookback=80):
    current = int(current_period)
    for i in range(0, lookback):
        yield str(current - i)


def build_url(period):
    return (
        f"https://wingo.oss-ap-southeast-7.aliyuncs.com/"
        f"WinGo_1_{period}_past100_draws?r={int(time.time() * 1000)}"
    )


def category_from_number(number):
    number_int = int(number)
    return "SMALL" if number_int <= 4 else "BIG"


def parse_create_time(create_time):
    if not create_time:
        return int(time.time())
    try:
        dt = datetime.fromisoformat(str(create_time).replace("Z", "+00:00"))
        return int(dt.timestamp())
    except Exception:
        return int(time.time())


def fetch_period(period, timeout=10):
    url = build_url(period)
    response = requests.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()

    text = response.text.strip()
    if not text.startswith("["):
        return []

    data = response.json()
    if not isinstance(data, list):
        return []

    rows = []
    for item in data:
        content = item.get("content", {})
        issue = content.get("issueNumber", "")
        number = content.get("number", "")
        colour = content.get("colour", "")
        premium = content.get("premium", "")
        create_time = item.get("createTime", "")

        if issue == "" or number == "":
            continue

        try:
            number_int = int(number)
        except Exception:
            continue

        rows.append({
            "period": str(issue),
            "number": number_int,
            "category": category_from_number(number_int),
            "colour": str(colour).upper(),
            "timestamp": parse_create_time(create_time),
            "premium": premium,
            "createTime": create_time,
        })

    return rows


def fetch_latest_available(timeout=10, lookback=80):
    current_period = get_current_period_1min()
    for period in nearby_periods(current_period, lookback=lookback):
        try:
            rows = fetch_period(period, timeout=timeout)
        except Exception:
            rows = []

        if rows:
            return period, rows

    return current_period, []


def public_row(row):
    create_time = row.get("createTime")
    if not create_time and row.get("timestamp"):
        try:
            create_time = datetime.fromtimestamp(
                int(row.get("timestamp")),
                UTC
            ).strftime("%Y-%m-%d %H:%M:%S UTC")
        except Exception:
            create_time = ""

    return {
        "issueNumber": row.get("period"),
        "number": row.get("number"),
        "colour": row.get("colour"),
        "category": row.get("category"),
        "premium": row.get("premium", ""),
        "createTime": create_time,
    }


def fetch_latest_20():
    latest_period, rows = fetch_latest_available(timeout=10, lookback=100)
    latest_20 = rows[:20]

    big_count = sum(1 for r in latest_20 if r.get("category") == "BIG")
    small_count = sum(1 for r in latest_20 if r.get("category") == "SMALL")

    payload = {
        "success": True,
        "currentPeriod": get_current_period_1min(),
        "latestAvailablePeriod": latest_period,
        "fetched": len(latest_20),
        "statistics": {
            "total": len(latest_20),
            "big": big_count,
            "small": small_count,
        },
        "history": [public_row(row) for row in latest_20],
    }

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


if __name__ == "__main__":
    fetch_latest_20()
