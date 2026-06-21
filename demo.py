import argparse
import json
import time
from datetime import UTC, datetime

import requests

from helpers import (
    DAILY_1K_HISTORY_CSV,
    get_current_period_1min,
    load_daily_1k_history,
    nearby_periods,
    save_daily_1k_history,
)


HEADERS = {
    "accept": "application/json",
    "user-agent": "Mozilla/5.0",
}


def build_url(period):
    return f"https://wingo.oss-ap-southeast-7.aliyuncs.com/WinGo_1_{period}_past100_draws?r={int(time.time() * 1000)}"


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
            "patternUsed": "demo_oss_fetch",
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


def fetch_daily_backfill(anchor_period, first_rows, timeout=10, max_pages=40):
    day = str(anchor_period)[:8]
    seen = set()
    all_rows = []

    for row in first_rows:
        period = str(row.get("period") or "")
        if period.startswith(day) and period not in seen:
            seen.add(period)
            all_rows.append(row)

    if all_rows:
        oldest = min(all_rows, key=lambda row: int(str(row.get("period"))))
        anchor_period = str(int(str(oldest.get("period"))) - 1)

    for _ in range(max_pages):
        try:
            rows = fetch_period(anchor_period, timeout=timeout)
        except Exception:
            break
        rows = [row for row in rows if str(row.get("period") or "").startswith(day)]
        new_rows = []
        for row in rows:
            period = str(row.get("period") or "")
            if period and period not in seen:
                seen.add(period)
                new_rows.append(row)
        if not new_rows:
            break
        all_rows.extend(new_rows)
        oldest = min(new_rows, key=lambda row: int(str(row.get("period"))))
        anchor_period = str(int(str(oldest.get("period"))) - 1)

    all_rows.sort(key=lambda row: int(str(row.get("period") or 0)), reverse=True)
    return all_rows


def public_row(row):
    create_time = row.get("createTime")
    if not create_time and row.get("timestamp"):
        try:
            create_time = datetime.fromtimestamp(int(row.get("timestamp")), UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
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


def sync_once(limit=20, full_backfill=True):
    before = len(load_daily_1k_history(limit=None))
    latest_period, latest_rows = fetch_latest_available(timeout=10)
    rows_to_save = fetch_daily_backfill(latest_period, latest_rows) if full_backfill else latest_rows
    merged = save_daily_1k_history([*load_daily_1k_history(limit=None), *rows_to_save])
    latest = merged[:limit]
    payload = {
        "success": True,
        "period": get_current_period_1min(),
        "latestAvailablePeriod": latest_period,
        "savedFile": DAILY_1K_HISTORY_CSV,
        "fetched": len(rows_to_save),
        "totalSaved": len(merged),
        "newSaved": max(0, len(merged) - before),
        "duplicatesRemoved": True,
        "noLimitSave": True,
        "history": [public_row(row) for row in latest],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def main():
    parser = argparse.ArgumentParser(description="Fetch OSS Wingo history and save daily_1k_history.csv")
    parser.add_argument("--watch", action="store_true", help="Keep syncing every 4 seconds")
    parser.add_argument("--fast", action="store_true", help="Fetch latest past100 only")
    parser.add_argument("--limit", type=int, default=20, help="Rows to print in JSON output")
    args = parser.parse_args()

    while True:
        try:
            sync_once(limit=args.limit, full_backfill=not args.fast)
        except Exception as e:
            print(json.dumps({"success": False, "error": str(e)}, indent=2))
        if not args.watch:
            break
        time.sleep(4)


if __name__ == "__main__":
    main()
