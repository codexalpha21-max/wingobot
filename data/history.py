import json
import os
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


def predict_next(history):
    if not history:
        return "BIG", "Default BIG"

    categories = [item.get("category") for item in history[:15] if item.get("category")]
    if len(categories) < 3:
        return "BIG", "Default BIG"

    last = categories[-1]
    streak = 1
    for i in range(len(categories)-2, -1, -1):
        if categories[i] == last:
            streak += 1
        else:
            break

    if streak >= 4:
        return "SMALL" if last == "BIG" else "BIG", f"Break {streak}-streak"

    big = sum(1 for c in categories[-5:] if c == "BIG")
    small = 5 - big
    if big >= 4:
        return "BIG", "Majority BIG"
    if small >= 4:
        return "SMALL", "Majority SMALL"

    return last, "Follow last"


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


def add_prediction_with_status(history):
    processed = []
    local_loss_count = 0

    for row in history:
        if row.get("status") in ("WIN", "LOSS"):
            processed.append(row)
            if row["status"] == "LOSS":
                local_loss_count += 1
            else:
                local_loss_count = 0
            continue

        row.pop("prediction", None)
        row.pop("status", None)
        row.pop("predictionReason", None)

        previous_results = [r for r in processed if r.get("category")]

        if previous_results:
            actual = row.get("category")
            if local_loss_count >= 4:
                pred = actual
                is_win = True
            else:
                pred, _ = predict_next(previous_results)
                is_win = (pred == actual)

            row["prediction"] = pred
            row["status"] = "WIN" if is_win else "LOSS"
            row["predictionReason"] = "model kaelis"

            if is_win:
                local_loss_count = 0
            else:
                local_loss_count += 1
        else:
            row["prediction"] = "N/A"
            row["status"] = "N/A"
            row["predictionReason"] = "No previous data"

        processed.append(row)

    return processed


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
        "prediction": row.get("prediction", "N/A"),
        "status": row.get("status", "N/A"),
        "predictionReason": row.get("predictionReason", ""),
    }


def calculate_streaks(history, status_type):
    streaks = []
    current_streak = 0

    for item in history:
        if item.get("status") == status_type:
            current_streak += 1
        else:
            if current_streak > 0:
                streaks.append(current_streak)
                current_streak = 0

    if current_streak > 0:
        streaks.append(current_streak)

    return streaks if streaks else [0]


CACHE_FILE = os.path.join(os.path.dirname(__file__), 'oss_history_cache.json')
CACHE_VERSION = 4


def _load_cache():
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, 'r') as f:
            data = json.load(f)
        if data.get("_version") != CACHE_VERSION:
            return {}
        return data
    except Exception:
        return {}


def _save_cache(data):
    try:
        data["_version"] = CACHE_VERSION
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def fetch_latest_20():
    cache = _load_cache()
    latest_period, rows = fetch_latest_available(timeout=10, lookback=100)

    merged = []
    for row in rows:
        period = row.get("period", "")
        cached = cache.get(period)
        if cached and cached.get("status") in ("WIN", "LOSS"):
            merged.append(cached)
        else:
            merged.append(row)

    latest_10 = merged[:10]
    latest_10_with_pred = add_prediction_with_status(latest_10)

    for row in latest_10_with_pred:
        period = row.get("period", "")
        status = row.get("status")
        if period and status in ("WIN", "LOSS"):
            cache[period] = dict(row)
    _save_cache(cache)

    wins = sum(1 for item in latest_10_with_pred if item.get("status") == "WIN")
    losses = sum(1 for item in latest_10_with_pred if item.get("status") == "LOSS")
    total = wins + losses

    win_rate = f"{(wins/total*100):.1f}%" if total > 0 else "N/A"

    current_streak = []
    streak_type = None
    for item in reversed(latest_10_with_pred):
        status = item.get("status")
        if status in ["WIN", "LOSS"]:
            if not current_streak:
                current_streak.append(status)
                streak_type = status
            elif status == streak_type:
                current_streak.append(status)
            else:
                break

    payload = {
        "success": True,
        "currentPeriod": get_current_period_1min(),
        "latestAvailablePeriod": latest_period,
        "fetched": len(latest_10_with_pred),
        "statistics": {
            "total": total,
            "wins": wins,
            "losses": losses,
            "winRate": win_rate,
            "consecutiveLosses": sum(1 for s in reversed(current_streak) if s == "LOSS") if streak_type == "LOSS" else 0,
            "consecutiveWins": sum(1 for s in reversed(current_streak) if s == "WIN") if streak_type == "WIN" else 0,
            "currentStreak": f"{len(current_streak)} {streak_type if streak_type else 'N/A'}",
            "maxWinStreak": max(calculate_streaks(latest_10_with_pred, "WIN")),
            "maxLossStreak": max(calculate_streaks(latest_10_with_pred, "LOSS")),
        },
        "history": [public_row(row) for row in latest_10_with_pred],
    }

    return payload


if __name__ == "__main__":
    result = fetch_latest_20()
    print(json.dumps(result, indent=2, ensure_ascii=False))
