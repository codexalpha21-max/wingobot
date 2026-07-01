import urllib.request
import time
import json
import os
import threading

SERVERS = [
    'https://cloud-apis.com',
]
ROUTES = [
    '/v2/free',
    '/model/kaelis',
    '/v2/history',
    '/v2/history/1m',
    '/v2/history/30s',
]
# Routes that require the model credentials payload
PROTECTED_ROUTES = {'/v2/free', '/model/kaelis', '/v2/history', '/v2/ml/patterns', '/v2/ml/status'}
MODEL_PAYLOAD = json.dumps({
    'model_name': 'kaelis',
    'model_key': 'kaelis.ai/paid/models',
}).encode()
PING_SECONDS = 5


def current_period_1m():
    t = time.gmtime()
    date_str = time.strftime('%Y%m%d', t)
    total_minutes = t.tm_hour * 60 + t.tm_min
    return f"{date_str}1000{total_minutes + 10001:05d}"


def nearby_periods(period=None, lookback=3):
    period = period or current_period_1m()
    prefix = period[:-5]
    number = int(period[-5:])
    return [f"{prefix}{number - offset:05d}" for offset in range(lookback + 1)]


def oss_url(period=None):
    return 'https://api.nexapk.in/wingo1min.php'


def ping_json(url, label, payload=None):
    try:
        method = 'POST' if payload else 'GET'
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                'User-Agent': 'Mozilla/5.0',
                'Accept': 'application/json,text/html,*/*',
                **({
                    'Content-Type': 'application/json',
                    'Content-Length': str(len(payload)),
                } if payload else {}),
            },
            method=method,
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            body = resp.read().decode()
        try:
            data = json.loads(body)
        except Exception:
            print(f"[{time.strftime('%H:%M:%S')}] {label} -> {resp.status} html")
            return body
        if isinstance(data, dict) and 'data' in data:
            items = data['data'].get('list', [])
            latest = items[0].get('issueNumber', '') if items else ''
            print(f"[{time.strftime('%H:%M:%S')}] {label} -> {resp.status} {latest}")
            return items
        pr = data.get('predictionResult') or data.get('prediction', {})
        per = pr.get('period', '') if isinstance(pr, dict) else ''
        print(f"[{time.strftime('%H:%M:%S')}] {label} -> {resp.status} {per}")
        return data
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] {label} -> ERROR: {e}")
        return None


def ping():
    # Deprecated single ping in favor of fast/slow loops
    pass


def ping_async(url, label, payload=None):
    threading.Thread(target=ping_json, args=(url, label, payload), daemon=True).start()


FAST_ROUTES = ['/v2/free', '/model/kaelis']
SLOW_ROUTES = ['/v2/history', '/v2/history/1m', '/v2/history/30s']
FAST_PING_SECONDS = 3
SLOW_PING_SECONDS = 20


def fast_ping_loop():
    print(f"[WARM] Fast ping loop started (every {FAST_PING_SECONDS}s) for prediction routes...")
    while True:
        for base in SERVERS:
            for route in FAST_ROUTES:
                payload = MODEL_PAYLOAD
                ping_async(base + route, base + route, payload=payload)
        time.sleep(FAST_PING_SECONDS)


def slow_ping_loop():
    print(f"[WARM] Slow ping loop started (every {SLOW_PING_SECONDS}s) for history/OSS...")
    while True:
        # Check lottery01 API
        data = ping_json(oss_url(), "lottery01")
        if isinstance(data, list) and data:
            pass
        
        # Check slower routes
        for base in SERVERS:
            for route in SLOW_ROUTES:
                ping_async(base + route, base + route, payload=None)
        time.sleep(SLOW_PING_SECONDS)


if __name__ == '__main__':
    print(f"[WARM] Starting local/cloud API ping loops...")
    print(f"  Server: {SERVERS[0]}")
    
    t_fast = threading.Thread(target=fast_ping_loop, daemon=True)
    t_slow = threading.Thread(target=slow_ping_loop, daemon=True)
    
    t_fast.start()
    t_slow.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[WARM] Shutting down ping loops.")
