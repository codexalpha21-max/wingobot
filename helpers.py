import os
import csv
import time
import warnings
import requests
import urllib3

warnings.filterwarnings('ignore', message='Unverified HTTPS request')
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from config import *

if 'DAILY_1K_HISTORY_CSV' not in globals():
    DAILY_1K_HISTORY_CSV = os.path.join(os.path.dirname(__file__), 'data', '1m', 'daily_1k_history.csv')

DAILY_1K_HEADER = ['period', 'number', 'category', 'colour', 'timestamp', 'patternUsed']


def get_current_period_1min():
    t = time.gmtime()
    date_str = time.strftime('%Y%m%d', t)
    total_minutes = t.tm_hour * 60 + t.tm_min
    period_number = str(total_minutes + 10001).zfill(4)
    return f"{date_str}1000{period_number}"


def _period_sort_key(period):
    try:
        return int(str(period))
    except Exception:
        return 0


def load_daily_1k_history(limit=None):
    if not os.path.exists(DAILY_1K_HISTORY_CSV):
        return []
    rows = []
    try:
        with open(DAILY_1K_HISTORY_CSV, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                period = str(row.get('period') or '')
                category = row.get('category')
                if not period or category not in ('BIG', 'SMALL'):
                    continue
                try:
                    number = int(float(row.get('number')))
                except Exception:
                    number = None
                rows.append({
                    'period': period,
                    'number': number,
                    'category': category,
                    'colour': row.get('colour') or None,
                    'timestamp': int(float(row.get('timestamp') or time.time())),
                    'patternUsed': row.get('patternUsed') or 'daily_1k_history',
                })
    except Exception:
        return []
    rows.sort(key=lambda item: _period_sort_key(item.get('period')), reverse=True)
    return rows[:limit] if limit else rows


def save_daily_1k_history(rows, max_rows=10000):
    by_period = {}
    for item in rows or []:
        period = str(item.get('period') or '')
        category = item.get('category')
        if not period or category not in ('BIG', 'SMALL'):
            continue
        by_period[period] = {
            'period': period,
            'number': item.get('number'),
            'category': category,
            'colour': item.get('colour') or item.get('color') or '',
            'timestamp': item.get('timestamp') or int(time.time()),
            'patternUsed': item.get('patternUsed') or 'daily_1k_history',
        }
    merged = list(by_period.values())
    merged.sort(key=lambda item: _period_sort_key(item.get('period')), reverse=True)
    merged = merged[:max_rows]
    tmp = DAILY_1K_HISTORY_CSV + '.tmp'
    os.makedirs(os.path.dirname(DAILY_1K_HISTORY_CSV), exist_ok=True)
    try:
        with open(tmp, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=DAILY_1K_HEADER)
            writer.writeheader()
            for row in sorted(merged, key=lambda item: _period_sort_key(item.get('period'))):
                writer.writerow({key: row.get(key, '') for key in DAILY_1K_HEADER})
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, DAILY_1K_HISTORY_CSV)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    return merged


def build_verify_api_url(period=None):
    period = period or get_current_period_1min()
    return f"{VERIFY_API_URL.format(period=period)}?r={int(time.time() * 1000)}"


def nearby_periods(period=None, lookback=12):
    base = period or get_current_period_1min()
    periods = []
    try:
        prefix = str(base)[:-5]
        number = int(str(base)[-5:])
        for offset in range(lookback + 1):
            periods.append(f"{prefix}{number - offset:05d}")
    except Exception:
        periods.append(base)
    return periods


def normalize_wingo_draw(item):
    content = item.get('content') or {}
    period = content.get('issueNumber') or item.get('issueNumber')
    number = content.get('number') or item.get('number')
    colour = content.get('colour') or item.get('colour')
    if number is None:
        return None
    try:
        number_int = int(number)
    except Exception:
        return None
    return {
        'period': period,
        'category': 'SMALL' if number_int <= 4 else 'BIG',
        'number': str(number_int),
        'colour': colour,
        'timestamp': int(time.time()),
    }


def normalize_wingo_draws(decoded):
    if not isinstance(decoded, list):
        return []
    draws = []
    for item in decoded:
        if not isinstance(item, dict):
            continue
        draw = normalize_wingo_draw(item)
        if draw and draw.get('period'):
            draws.append(draw)
    return draws


def build_default_user_state():
    patterns = [
        'zigZag', 'skipPattern', 'trendBased', 'cyclePattern', 'longPattern',
        'markovChain', 'entropyBased', 'numberBased', 'neural',
        'streakMomentum', 'markov2', 'trendStatistics'
    ]
    pattern_stats = {}
    for p in patterns:
        pattern_stats[p] = {
            'wins': 0, 'total': 0, 'successRate': 0,
            'recentWins': 0, 'recentTotal': 0, 'consecutiveLosses': 0
        }

    number_patterns = {}
    number_repetition = {}
    for i in range(10):
        number_patterns[i] = {
            'BIG': {'count': 0, 'successRate': 0},
            'SMALL': {'count': 0, 'successRate': 0},
            'total': 0
        }
        number_repetition[i] = {'count': 0, 'recentCount': 0, 'lastSeen': 0}

    return {
        'showHigher': True,
        'autoToggle': True,
        'lastAdjustment': 0,
        'patternStatsNormal': pattern_stats,
        'patternStatsAdvanced': pattern_stats,
        'numberPatterns': number_patterns,
        'numberRepetition': number_repetition,
        'transitionMatrix': {
            'BIG': {'BIG': 0, 'SMALL': 0},
            'SMALL': {'BIG': 0, 'SMALL': 0}
        },
        'entropyHistory': [],
        'neuralWeights': [0.0] * 10,
        'bias': 0.0,
        'learningRate': 0.1,
        'lastProcessedPeriod': '',
        'lossRecovery': {
            'consecutiveLosses': 0,
            'totalSkipsThisRun': 0,
            'lastSkipPeriod': '',
            'skipCooldownUntil': 0,
            'recoveryMode': False,
            'recoveryModeStart': 0,
            'lastFiveResults': [],
            'forcedFlipActive': False,
            'forcedFlipCount': 0,
            'lossGuardActive': False,
            'lossGuardReason': '',
            'lastSkipReason': '',
        }
    }


def fetch_api_data_raw(retries=1, timeout=2):
    headers = {
        'Content-Type': 'application/json;charset=UTF-8',
        'Accept': 'application/json, text/plain, */*'
    }
    last_error = 'No draw data found'
    for period in nearby_periods():
        url = build_verify_api_url(period)
        for i in range(retries):
            try:
                r = requests.get(url, headers=headers, timeout=timeout, verify=False)
                decoded = r.json()
                draws = normalize_wingo_draws(decoded)
                if draws:
                    return draws
            except Exception as e:
                last_error = str(e)
                if i < retries - 1:
                    time.sleep(0.3)
    return {'error': 'Failed after retries'}


def fetch_api_data(retries=1, timeout=3, bypass_cache=False):
    import json
    cache_file = os.path.join(DATA_DIR, 'api_data_cache.json')
    now = int(time.time())
    if not bypass_cache and os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache = json.load(f)
        if cache and 'timestamp' in cache and (now - cache['timestamp']) < 20:
            return cache['data']

    data = fetch_api_data_raw(retries, timeout)
    
    # Fallback to Wingobot daily history if OSS bucket URL fails or returns 404
    if not data or 'error' in data:
        try:
            fallback_data = fetch_wingobot_daily_history(retries=retries, timeout=timeout + 2, limit=150)
            if fallback_data and isinstance(fallback_data, list) and 'error' not in fallback_data:
                # Normalize fallback data format to match expected API format
                normalized_fallback = []
                for item in fallback_data:
                    normalized_fallback.append({
                        'period': item.get('period'),
                        'category': item.get('category'),
                        'number': str(item.get('number')) if item.get('number') is not None else None,
                        'colour': item.get('colour'),
                        'timestamp': item.get('timestamp', now),
                    })
                data = normalized_fallback
        except Exception as fallback_exc:
            print(f"[FALLBACK] fetch_wingobot_daily_history failed: {fallback_exc}")

    if data and 'error' not in data:
        with open(cache_file, 'w') as f:
            json.dump({'timestamp': now, 'data': data}, f)
        return data

    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache = json.load(f)
        if cache and 'data' in cache:
            return cache['data']

    return data


def fetch_trend_statistics_raw(retries=1, timeout=2):
    params = {
        'gameCode': 'WinGo_1M', 'pageNo': 1, 'pageSize': 10,
        'language': 'en', 'random': '739791024272',
        'signature': 'CE224F61135E94EE84483A803F1DD0C8',
        'timestamp': str(int(time.time()))
    }
    headers = {
        'Accept': 'application/json, text/plain, */*',
        'User-Agent': 'Mozilla/5.0 (Linux; Android 10)',
        'Referer': 'https://51gameq.com/#/saasLottery/WinGo'
    }
    for i in range(retries):
        try:
            r = requests.get(TREND_STATS_API_URL, params=params,
                             headers=headers, timeout=timeout, verify=False)
            decoded = r.json()
            if decoded.get('code') == 0 and isinstance(decoded.get('data'), list):
                return decoded['data']
        except Exception as e:
            if i == retries - 1:
                return {'error': str(e)}
            time.sleep(0.3)
    return {'error': 'Failed after retries'}


def fetch_trend_statistics(retries=0, timeout=2, bypass_cache=False):
    import json
    cache_file = os.path.join(DATA_DIR, 'trend_stats_cache.json')
    now = int(time.time())
    if not bypass_cache and os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache = json.load(f)
        if cache and 'timestamp' in cache and (now - cache['timestamp']) < 60:
            return cache['data']

    data = fetch_trend_statistics_raw(retries, timeout)
    if data and 'error' not in data:
        with open(cache_file, 'w') as f:
            json.dump({'timestamp': now, 'data': data}, f)
        return data

    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache = json.load(f)
        if cache and 'data' in cache:
            return cache['data']

    return data


def fetch_game_history_raw(retries=1, timeout=3):
    headers = {
        'Content-Type': 'application/json;charset=UTF-8',
        'Accept': 'application/json, text/plain, */*'
    }
    last_error = 'No draw data found'
    for period in nearby_periods(lookback=12):
        url = build_verify_api_url(period)
        for i in range(retries):
            try:
                r = requests.get(url, headers=headers, timeout=timeout, verify=False)
                decoded = r.json()
                draws = normalize_wingo_draws(decoded)
                if draws:
                    return draws
            except Exception as e:
                last_error = str(e)
                if i < retries - 1:
                    time.sleep(0.3)
    return {'error': last_error}


def fetch_wingobot_history(retries=1, timeout=5):
    headers = {
        'Authorization': f'Bearer {WINGOBOT_TOKEN}',
        'Accept': 'application/json',
    }
    for i in range(retries):
        try:
            r = requests.get(WINGOBOT_API_URL, headers=headers, timeout=timeout, verify=False)
            decoded = r.json()
            if decoded.get('success'):
                history = decoded.get('history', [])
                current = decoded.get('current', {})
                items = []
                if current and current.get('issueNumber'):
                    items.append({
                        'period': current.get('issueNumber', ''),
                        'number': current.get('number'),
                        'category': 'SMALL' if (current.get('number') is not None and int(current['number']) <= 4) else 'BIG',
                        'colour': current.get('colour'),
                        'timestamp': int(time.time()),
                    })
                for item in history:
                    items.append({
                        'period': item.get('issueNumber', ''),
                        'number': item.get('number'),
                        'category': 'SMALL' if (item.get('number') is not None and int(item['number']) <= 4) else 'BIG',
                        'colour': item.get('colour'),
                        'timestamp': int(time.time()),
                    })
                return items
            else:
                if i == retries - 1:
                    return {'error': decoded.get('error', 'Wingobot API error')}
        except Exception as e:
            if i == retries - 1:
                return {'error': str(e)}
            time.sleep(0.3)
    return {'error': 'Failed after retries'}


def _fetch_auth_history_items(headers, timeout=10):
    """Fetch history from WINGOBOT_API_URL (auth endpoint) and return normalized items."""
    try:
        r = requests.get(WINGOBOT_API_URL, headers=headers, timeout=timeout, verify=False)
        decoded = r.json()
        if not decoded.get('success'):
            return []
        items = []
        current = decoded.get('current') or {}
        if current.get('issueNumber'):
            items.append(current)
        items.extend(decoded.get('history') or [])
        normalized = []
        for item in items:
            period = str(item.get('issueNumber') or item.get('period') or '')
            if not period:
                continue
            number = item.get('number')
            try:
                number_int = int(float(number))
            except Exception:
                continue
            normalized.append({
                'period': period,
                'number': number_int,
                'category': 'SMALL' if number_int <= 4 else 'BIG',
                'colour': item.get('colour') or item.get('color'),
                'timestamp': int(time.time()),
                'patternUsed': 'daily_1k_history',
            })
        return normalized
    except Exception:
        return []


def fetch_wingobot_daily_history(retries=1, timeout=15, limit=None):
    cached_rows = load_daily_1k_history(limit=None)
    headers = {
        'Authorization': f'Bearer {WINGOBOT_TOKEN}',
        'Accept': 'application/json',
    }

    # --- Pull from auth API (WINGOBOT_API_URL) first for extra history rows ---
    auth_items = _fetch_auth_history_items(headers, timeout=min(timeout, 10))

    for i in range(retries):
        try:
            r = requests.get(WINGOBOT_DAILY_HISTORY_URL, headers=headers, timeout=timeout, verify=False)
            decoded = r.json()
            if not decoded.get('success') and 'history' not in decoded:
                if i == retries - 1:
                    # Still merge auth items even if daily URL failed
                    if auth_items or cached_rows:
                        merged = save_daily_1k_history([*cached_rows, *auth_items])
                        return merged[:limit] if limit else merged
                    return {'error': decoded.get('error', 'Wingobot daily history API error')}
                time.sleep(0.3)
                continue

            items = []
            current = decoded.get('current') or {}
            if current.get('issueNumber'):
                items.append(current)
            items.extend(decoded.get('history') or [])

            # Collect all periods seen so far to avoid duplicates
            seen = set()
            normalized = []

            # Add auth items first (may contain unique periods)
            for item in auth_items:
                period = str(item.get('period') or '')
                if period and period not in seen:
                    seen.add(period)
                    normalized.append(item)

            # Add daily-history items
            for item in items:
                period = str(item.get('issueNumber') or item.get('period') or '')
                if not period or period in seen:
                    continue
                seen.add(period)
                number = item.get('number')
                try:
                    number_int = int(float(number))
                except Exception:
                    continue
                normalized.append({
                    'period': period,
                    'number': number_int,
                    'category': 'SMALL' if number_int <= 4 else 'BIG',
                    'colour': item.get('colour') or item.get('color'),
                    'timestamp': int(time.time()),
                    'patternUsed': 'daily_1k_history',
                })

            merged = save_daily_1k_history([*cached_rows, *normalized])
            return merged[:limit] if limit else merged
        except Exception as e:
            if i == retries - 1:
                # Fallback: at least save auth items
                if auth_items:
                    merged = save_daily_1k_history([*cached_rows, *auth_items])
                    return merged[:limit] if limit else merged
                if cached_rows:
                    return cached_rows[:limit] if limit else cached_rows
                return {'error': str(e)}
            time.sleep(0.3)
    return {'error': 'Failed after retries'}
