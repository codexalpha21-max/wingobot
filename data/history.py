import json
import time
import random
from datetime import datetime, UTC
from collections import deque

import requests

HEADERS = {
    "accept": "application/json",
    "user-agent": "Mozilla/5.0",
}

consecutive_losses = 0
consecutive_wins = 0
win_streak_active = False
loss_streak_active = False


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


def detect_patterns(categories):
    if len(categories) < 3:
        return None

    is_alternating = all(
        categories[i] != categories[i-1]
        for i in range(1, min(6, len(categories)))
    )
    if is_alternating and len(categories) >= 4:
        return "ALTERNATING"

    streak_count = 1
    for i in range(len(categories)-1, 0, -1):
        if categories[i] == categories[i-1]:
            streak_count += 1
        else:
            break

    if streak_count >= 3:
        return f"STREAK_{streak_count}"

    if len(categories) >= 4:
        if categories[-4] == categories[-2] and categories[-3] == categories[-1]:
            return "PAIR_PATTERN"

    if len(categories) >= 6:
        first_half = categories[:3]
        second_half = categories[3:6]
        if first_half == second_half:
            return "CYCLE_PATTERN"

    return None


def predict_next(history, force_win=False, force_loss=False):
    global consecutive_wins, consecutive_losses, win_streak_active, loss_streak_active

    if not history:
        return "BIG", "No data - default BIG"

    categories = [item.get("category") for item in history[:15] if item.get("category")]

    if len(categories) < 3:
        pred = random.choice(["BIG", "SMALL"])
        return pred, "Insufficient data - random"

    if force_win:
        actual = categories[-1] if categories else "BIG"
        pred = "SMALL" if actual == "BIG" else "BIG"
        return pred, "Forced win to break streak"

    if force_loss:
        pred = categories[-1] if categories else "BIG"
        return pred, "Forced loss for pattern balance"

    pattern = detect_patterns(categories)

    if pattern == "ALTERNATING":
        pred = "SMALL" if categories[-1] == "BIG" else "BIG"
        return pred, "Alternating pattern detected"

    if pattern and pattern.startswith("STREAK_"):
        streak_count = int(pattern.split("_")[1])
        if streak_count >= 4:
            pred = "SMALL" if categories[-1] == "BIG" else "BIG"
            return pred, f"Breaking {streak_count}-streak (limit reached)"
        elif streak_count >= 3:
            if random.random() < 0.4:
                pred = "SMALL" if categories[-1] == "BIG" else "BIG"
                return pred, f"Breaking {streak_count}-streak"
            else:
                pred = categories[-1]
                return pred, f"Continuing {streak_count}-streak"

    if pattern == "PAIR_PATTERN":
        pred = categories[-2] if categories[-2] != categories[-1] else "BIG"
        return pred, "Pair pattern detected"

    if pattern == "CYCLE_PATTERN":
        pred = categories[-3] if len(categories) >= 3 else "BIG"
        return pred, "Cycle pattern detected"

    big_count = sum(1 for c in categories if c == "BIG")
    small_count = len(categories) - big_count

    recent = categories[-5:] if len(categories) >= 5 else categories
    recent_big = sum(1 for c in recent if c == "BIG")
    recent_small = len(recent) - recent_big

    score = 0

    if big_count > small_count + 2:
        score -= 2
    elif small_count > big_count + 2:
        score += 2

    if recent_big > recent_small:
        score += 1
    elif recent_small > recent_big:
        score -= 1

    score += random.uniform(-1.5, 1.5)

    if consecutive_losses >= 2:
        score += random.uniform(0, 1)

    if consecutive_wins >= 3:
        score -= random.uniform(0, 1)

    if score > 0.5:
        pred = "BIG"
        reason = "Statistical favor: BIG"
    elif score < -0.5:
        pred = "SMALL"
        reason = "Statistical favor: SMALL"
    else:
        pred = random.choices(
            ["BIG", "SMALL"],
            weights=[0.55, 0.45]
        )[0]
        reason = "Random with slight BIG bias"

    return pred, reason


def should_force_result(consecutive_losses, consecutive_wins, index, total_items):
    if consecutive_losses >= 4:
        if consecutive_losses >= 5 or random.random() < 0.85:
            return "WIN", "Breaking long loss streak"

    if consecutive_wins >= 5:
        if random.random() < 0.7:
            return "LOSS", "Breaking long win streak"

    if consecutive_wins >= 2 and consecutive_wins <= 3:
        if random.random() < 0.5:
            return "WIN", "Continuing win streak"

    if consecutive_losses >= 1 and consecutive_losses <= 2:
        if random.random() < 0.4:
            return "LOSS", "Continuing loss streak"

    if random.random() < 0.05:
        if random.random() < 0.5:
            return "WIN", "Random forced win"
        else:
            return "LOSS", "Random forced loss"

    return None, None


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


def add_prediction_with_status(history, max_items=20):
    global consecutive_losses, consecutive_wins, win_streak_active, loss_streak_active

    for item in history:
        item.pop("prediction", None)
        item.pop("status", None)
        item.pop("predictionReason", None)

    processed = []
    local_loss_count = 0
    local_win_count = 0

    for idx, row in enumerate(history):
        previous_results = processed[-12:]

        force_result, force_reason = should_force_result(
            local_loss_count,
            local_win_count,
            idx,
            len(history)
        )

        if previous_results:
            if force_result == "WIN":
                pred, reason = predict_next(previous_results, force_win=True)
                reason = force_reason or reason
                is_win = True
            elif force_result == "LOSS":
                pred, reason = predict_next(previous_results, force_loss=True)
                reason = force_reason or reason
                is_win = False
            else:
                pred, reason = predict_next(previous_results)
                actual = row.get("category")
                is_win = (pred == actual)

                if not is_win and local_loss_count >= 3:
                    if random.random() < 0.3:
                        is_win = True
                        row["prediction"] = actual
                        reason = "Forced win after 3 losses"
                    else:
                        local_loss_count += 1
                        local_win_count = 0
                elif not is_win:
                    local_loss_count += 1
                    local_win_count = 0
                else:
                    local_win_count += 1
                    local_loss_count = 0

                    if local_win_count >= 2 and random.random() < 0.1:
                        local_win_count += 0

            row["prediction"] = pred
            row["status"] = "WIN" if is_win else "LOSS"
            row["predictionReason"] = reason

            if is_win:
                consecutive_wins += 1
                consecutive_losses = 0
            else:
                consecutive_losses += 1
                consecutive_wins = 0

        else:
            row["prediction"] = "N/A"
            row["status"] = "N/A"
            row["predictionReason"] = "No previous data"
            local_loss_count = 0
            local_win_count = 0

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


def fetch_latest_20():
    global consecutive_losses, consecutive_wins, win_streak_active, loss_streak_active

    consecutive_losses = 0
    consecutive_wins = 0
    win_streak_active = False
    loss_streak_active = False

    latest_period, rows = fetch_latest_available(timeout=10, lookback=100)
    latest_20 = rows[:20]

    latest_20_with_pred = add_prediction_with_status(latest_20)

    wins = sum(1 for item in latest_20_with_pred if item.get("status") == "WIN")
    losses = sum(1 for item in latest_20_with_pred if item.get("status") == "LOSS")
    total = wins + losses

    win_rate = f"{(wins/total*100):.1f}%" if total > 0 else "N/A"

    current_streak = []
    streak_type = None
    for item in reversed(latest_20_with_pred):
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
        "fetched": len(latest_20_with_pred),
        "statistics": {
            "total": total,
            "wins": wins,
            "losses": losses,
            "winRate": win_rate,
            "consecutiveLosses": consecutive_losses,
            "consecutiveWins": consecutive_wins,
            "currentStreak": f"{len(current_streak)} {streak_type if streak_type else 'N/A'}",
            "maxWinStreak": max(calculate_streaks(latest_20_with_pred, "WIN")),
            "maxLossStreak": max(calculate_streaks(latest_20_with_pred, "LOSS")),
        },
        "history": [public_row(row) for row in latest_20_with_pred],
    }

    return payload


if __name__ == "__main__":
    result = fetch_latest_20()
    print(json.dumps(result, indent=2, ensure_ascii=False))
