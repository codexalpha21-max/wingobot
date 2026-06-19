import csv
import json
import os
import threading
import time
import requests

from helpers import get_current_period_1min
from ml import get_model_summary, predict_ml, train_model
from storage import load_prediction_history_entries
from free_prediction import load_free_history
from config import DATA_DIR

DAILY_1K_CSV = os.path.join(DATA_DIR, '1m', 'daily_1k_history.csv')
MODEL_HISTORY_CSV = os.path.join(DATA_DIR, 'model', 'model_prediction_history.csv')
MODEL_HISTORY_BACKUP_CSV = MODEL_HISTORY_CSV + '.backup'
MODEL_CACHE_FILE = os.path.join(DATA_DIR, 'model_cache.json')
MODEL_HISTORY_LIMIT = 10
PAYLOAD_CACHE_SECONDS = 10
TRAINING_ROWS_REQUIRED = 2000
DAILY_1K_HEADER = ['period', 'number', 'category', 'colour', 'timestamp', 'patternUsed']

HEADER = ['id', 'period', 'prediction', 'status', 'confidence', 'actual', 'number', 'patternused', 'timestamp', 'skipped', 'skipreason', 'created_at']

_lock = threading.RLock()
_payload_cache = None
_payload_cache_time = 0
_last_train_time = 0
_bg_refresh_thread = None
_bg_refresh_lock = threading.Lock()
_fetch_thread = None
_fetch_running = True
_verified_periods = set()
_verified_periods_lock = threading.Lock()

# ---------------------------------------------------------------------------
# 5-second API fetcher – saves new periods to daily_1k_history.csv
# ---------------------------------------------------------------------------

def _build_api_url(period=None):
    if period is None:
        period = get_current_period_1min()
    ts = int(time.time() * 1000)
    return f"https://wingo.oss-ap-southeast-7.aliyuncs.com/WinGo_1_{period}_past100_draws?r={ts}"

def _fetch_and_save():
    try:
        url = _build_api_url()
        headers = {"accept": "application/json, text/plain, */*"}
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code != 200:
            return
        data = resp.json()
        if not isinstance(data, list):
            return
        os.makedirs(os.path.dirname(DAILY_1K_CSV), exist_ok=True)
        existing = set()
        if os.path.exists(DAILY_1K_CSV):
            try:
                with open(DAILY_1K_CSV, 'r', newline='') as f:
                    for row in csv.DictReader(f):
                        existing.add(str(row.get('period', '')))
            except Exception:
                pass
        new_rows = []
        for item in data:
            period = str(item.get('issueNumber', ''))
            if not period or period in existing:
                continue
            number = item.get('content', {}).get('number')
            colour = item.get('content', {}).get('colour', '')
            category = 'BIG' if int(number) >= 5 else 'SMALL' if number is not None else ''
            if not category:
                continue
            new_rows.append({'period': period, 'number': str(number), 'category': category, 'colour': colour, 'timestamp': str(int(time.time())), 'patternUsed': 'api_fetch'})
            existing.add(period)
        if new_rows:
            mode = 'a' if os.path.exists(DAILY_1K_CSV) else 'w'
            with open(DAILY_1K_CSV, 'a' if mode == 'a' else 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=DAILY_1K_HEADER)
                if mode == 'w':
                    w.writeheader()
                w.writerows(new_rows)
            print(f'[FETCH] saved {len(new_rows)} new periods to daily_1k_history.csv')
    except Exception as exc:
        pass

def _fetch_loop():
    while _fetch_running:
        _fetch_and_save()
        time.sleep(5)

def _start_fetch_loop():
    global _fetch_thread
    if _fetch_thread and _fetch_thread.is_alive():
        return
    t = threading.Thread(target=_fetch_loop, daemon=True, name='api_fetch_5s')
    _fetch_thread = t
    t.start()
    print('[FETCH] 5s API fetch loop started')

# ---------------------------------------------------------------------------
# Collect ALL data from every source
# ---------------------------------------------------------------------------

def _load_csv(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            return [row for row in reader if row.get('period')]
    except Exception:
        return []

def _collect_all_data():
    by_period = {}
    for row in _load_csv(DAILY_1K_CSV):
        p = str(row.get('period', ''))
        cat = row.get('category', '')
        if p and cat in ('BIG', 'SMALL'):
            by_period[p] = {'period': p, 'actual': cat, 'number': row.get('number'), 'prediction': None, 'status': 'TRAINING', 'source': 'daily_1k'}
    for path, src in [(os.path.join(DATA_DIR, 'predict', 'prediction_history.csv'), 'v2_predict'),
                      (os.path.join(DATA_DIR, 'free', 'free_prediction_history.csv'), 'v2_free'),
                      (MODEL_HISTORY_CSV, 'model_predict'),
                      (MODEL_HISTORY_BACKUP_CSV, 'model_predict_backup')]:
        for row in _load_csv(path):
            p = str(row.get('period', ''))
            actual = row.get('actual') or row.get('category', '')
            if p and actual in ('BIG', 'SMALL') and p not in by_period:
                by_period[p] = {'period': p, 'actual': actual, 'number': row.get('number'), 'prediction': row.get('prediction') or None, 'status': row.get('status', 'TRAINING'), 'source': src}
    for entry in load_prediction_history_entries(limit=None):
        p = str(entry.get('period', ''))
        if p and entry.get('status') in ('WIN', 'LOSS') and entry.get('actual') in ('BIG', 'SMALL') and p not in by_period:
            by_period[p] = {'period': p, 'actual': entry['actual'], 'number': entry.get('number'), 'prediction': entry.get('prediction'), 'status': entry['status'], 'source': 'v2_predict_memory'}
    for entry in load_free_history(limit=None):
        p = str(entry.get('period', ''))
        if p and entry.get('status') in ('WIN', 'LOSS') and entry.get('actual') in ('BIG', 'SMALL') and p not in by_period:
            by_period[p] = {'period': p, 'actual': entry['actual'], 'number': entry.get('number'), 'prediction': entry.get('prediction'), 'status': entry['status'], 'source': 'v2_free_memory'}
    rows = sorted(by_period.values(), key=lambda r: int(str(r.get('period', '0'))[-12:] or 0))
    return rows

# ---------------------------------------------------------------------------
# Verify pending entries
# ---------------------------------------------------------------------------

def _verify(entries):
    current_period = get_current_period_1min()
    changed = False
    for e in entries:
        if e.get('status') != 'Pending' or e.get('actual') in ('BIG', 'SMALL'):
            continue
        if str(e.get('period', '')) >= current_period:
            continue
        pred = e.get('prediction', '')
        if not pred:
            continue
        if str(e.get('skipped', '')).lower() in ('1', 'true'):
            continue
        with _verified_periods_lock:
            if e.get('period') in _verified_periods:
                continue
        try:
            url = _build_api_url(e.get('period'))
            resp = requests.get(url, headers={"accept": "application/json"}, timeout=3)
            if resp.status_code != 200:
                continue
            data = resp.json()
            if not isinstance(data, list):
                continue
            match = None
            for item in data:
                if str(item.get('issueNumber', '')) == str(e.get('period', '')):
                    match = item
                    break
            if not match:
                continue
            actual_num = match.get('content', {}).get('number')
            if actual_num is None:
                continue
            actual = 'BIG' if int(actual_num) >= 5 else 'SMALL'
            e['actual'] = actual
            e['number'] = str(actual_num)
            e['status'] = 'WIN' if pred == actual else 'LOSS'
            e['skipped'] = '0'
            _upsert(e)
            with _verified_periods_lock:
                _verified_periods.add(e.get('period'))
            changed = True
        except Exception:
            continue
    return _entries() if changed else entries

# ---------------------------------------------------------------------------
# CSV read/write helpers
# ---------------------------------------------------------------------------

def _period_key(p):
    try:
        return int(str(p))
    except Exception:
        return 0

def _csv_value(v):
    return '' if v is None else str(v)

def _entries():
    rows = _load_csv(MODEL_HISTORY_CSV)
    rows += _load_csv(MODEL_HISTORY_BACKUP_CSV)
    by_period = {}
    for r in rows:
        by_period[str(r.get('period', ''))] = r
    result = sorted(by_period.values(), key=lambda r: _period_key(r.get('period')), reverse=True)
    return result

def _upsert(entry):
    period = str(entry.get('period', ''))
    if not period:
        return
    row = {k: _csv_value(entry.get(k, '')) for k in HEADER}
    rows = _entries()
    found = False
    for idx, old in enumerate(rows):
        if str(old.get('period', '')) == period:
            old_status = old.get('status', '')
            new_status = row.get('status', '')
            if old_status in ('WIN', 'LOSS') and new_status == 'Pending':
                return
            rows[idx] = row
            found = True
            break
    if not found:
        rows.append(row)
    rows_sorted = sorted(rows, key=lambda r: _period_key(r.get('period')))
    os.makedirs(os.path.dirname(MODEL_HISTORY_CSV), exist_ok=True)
    for path in (MODEL_HISTORY_CSV, MODEL_HISTORY_BACKUP_CSV):
        try:
            with open(path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=HEADER)
                w.writeheader()
                w.writerows(rows_sorted)
        except Exception:
            pass

def _public_entry(row):
    number = None
    try:
        number = int(float(str(row.get('number', ''))))
    except Exception:
        pass
    return {'period': row.get('period', ''), 'prediction': row.get('prediction') or '', 'status': row.get('status', 'Pending'), 'confidence': round(float(row.get('confidence') or 0), 2), 'actual': row.get('actual'), 'number': number, 'skipped': str(row.get('skipped', '')).lower() in ('1', 'true'), 'skipReason': row.get('skipreason') or row.get('skipReason') or ''}

def _stats(history):
    w = sum(1 for h in history if h.get('status') == 'WIN')
    l = sum(1 for h in history if h.get('status') == 'LOSS')
    p = sum(1 for h in history if h.get('status') == 'Pending')
    s = sum(1 for h in history if h.get('skipped'))
    settled = w + l
    return {'wins': w, 'losses': l, 'pending': p, 'skipped': s, 'total': len(history), 'winRate': round((w / max(settled, 1)) * 100, 2), 'settled': settled}

def _training_rows():
    all_data = _collect_all_data()
    return [r for r in all_data if r.get('actual') in ('BIG', 'SMALL')]

# ---------------------------------------------------------------------------
# Main prediction – pure ML, no custom brain/analysis/ensemble
# ---------------------------------------------------------------------------

def get_model_payload():
    global _payload_cache, _payload_cache_time, _last_train_time

    now = time.time()
    if _payload_cache and now - _payload_cache_time < PAYLOAD_CACHE_SECONDS:
        return _payload_cache
    if not _lock.acquire(blocking=False):
        if _payload_cache:
            c = dict(_payload_cache)
            c['stale'] = True
            c['staleReason'] = 'refresh_in_progress'
            return c
    else:
        _lock.release()

    with _lock:
        _start_fetch_loop()
        rows = _verify(_entries())

        current_period = get_current_period_1min()
        current = next((e for e in rows if str(e.get('period', '')) == current_period), None)
        training_data = _training_rows()

        if len(training_data) >= TRAINING_ROWS_REQUIRED and (now - _last_train_time >= 60 or not current):
            train_model(training_data, force=True)
            _last_train_time = time.time()

        summary = get_model_summary()
        slice_data = [r for r in training_data[-200:] if r.get('actual') in ('BIG', 'SMALL')]
        current_slice = [{'category': r['actual'], 'number': r.get('number')} for r in reversed(slice_data[:80])]
        ml_result = predict_ml(training_data, current_slice) if current_slice else None

        selected_prediction = None
        model_ready = ml_result and ml_result.get('samples', 0) >= TRAINING_ROWS_REQUIRED
        if model_ready:
            selected_prediction = {
                'prediction': ml_result['prediction'],
                'confidence': ml_result['confidence'],
                'model': ml_result.get('selectedModel', 'ensemble'),
                'validationAccuracy': ml_result.get('selectedModelAccuracy'),
                'allPredictions': ml_result.get('modelPredictions', []),
            }

        if not current:
            if selected_prediction:
                current = {'period': current_period, 'prediction': selected_prediction['prediction'], 'status': 'Pending', 'confidence': selected_prediction['confidence'], 'actual': '', 'number': '', 'patternused': selected_prediction.get('model', 'ml'), 'timestamp': str(int(time.time())), 'skipped': '0', 'skipreason': '', 'created_at': time.strftime('%Y-%m-%d %H:%M:%S')}
            else:
                current = {'period': current_period, 'prediction': '', 'status': 'Pending', 'confidence': 0, 'actual': '', 'number': '', 'patternused': 'waiting_for_data', 'timestamp': str(int(time.time())), 'skipped': '1', 'skipreason': 'Collecting training data', 'created_at': time.strftime('%Y-%m-%d %H:%M:%S')}
            _upsert(current)
            rows = _entries()

        rows.sort(key=lambda r: _period_key(r.get('period')), reverse=True)
        history = [_public_entry(r) for r in rows[:MODEL_HISTORY_LIMIT]]

        payload = {
            'predictionResult': {'period': current.get('period'), 'prediction': current.get('prediction') or '', 'status': current.get('status', 'Pending'), 'skipped': current.get('skipped') == '1', 'skipReason': current.get('skipreason') or ''},
            'predictionDetails': {'gameType': 'Wingo 1 Min Model', 'confidence': round(float(current.get('confidence') or 0), 2), 'actual': current.get('actual') or None, 'number': _number(current.get('number')), 'mlPrediction': ml_result, 'selectedModel': selected_prediction.get('model') if selected_prediction else None},
            'modelDecision': selected_prediction,
            'stats': _stats(history),
            'history': history,
            'learning': {'learnedRows': len(training_data), 'sources': ['daily_1k_history.csv', 'prediction_history.csv', 'free_prediction_history.csv', 'model_prediction_history.csv', 'live_api_5s']},
            'ml': {'trained': summary.get('totalSamples', 0) >= TRAINING_ROWS_REQUIRED, 'samples': summary.get('totalSamples', 0), 'accuracy': summary.get('lastAccuracy'), 'models': summary.get('models', [])},
        }
        _payload_cache = payload
        _payload_cache_time = time.time()
        return payload

def _number(value):
    try:
        return int(float(str(value)))
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def _save_cache(payload):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        tmp = MODEL_CACHE_FILE + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump({'timestamp': time.time(), 'payload': payload}, f)
        os.replace(tmp, MODEL_CACHE_FILE)
    except Exception:
        pass

def _load_cache():
    if not os.path.exists(MODEL_CACHE_FILE):
        return None, None
    try:
        with open(MODEL_CACHE_FILE, 'r', encoding='utf-8') as f:
            d = json.load(f)
        return d.get('payload'), time.time() - float(d.get('timestamp', 0))
    except Exception:
        return None, None

def get_cached_model_payload():
    p, age = _load_cache()
    if age is None or age > 10:
        _ensure_bg_refresh()
    if p is None:
        live = get_model_payload()
        _save_cache(live)
        return live
    if age <= 120:
        return p
    stale = dict(p)
    stale['stale'] = True
    stale['staleReason'] = f'cache_age_{int(age)}s'
    return stale

def _bg_worker():
    try:
        p = get_model_payload()
        _save_cache(p)
    except Exception as exc:
        print(f'[MODEL_BG] {exc}')

def _ensure_bg_refresh():
    global _bg_refresh_thread
    with _bg_refresh_lock:
        if _bg_refresh_thread and _bg_refresh_thread.is_alive():
            return
        t = threading.Thread(target=_bg_worker, daemon=True, name='model_bg')
        _bg_refresh_thread = t
        t.start()

def start_model_bg_refresh_loop():
    global _fetch_running
    _fetch_running = True
    _start_fetch_loop()
    def _loop():
        while True:
            try:
                p = get_model_payload()
                _save_cache(p)
            except Exception as e:
                print(f'[MODEL_BG] loop: {e}')
            time.sleep(10)
    t = threading.Thread(target=_loop, daemon=True, name='model_bg_loop')
    t.start()
    print('[MODEL] Background refresh started (10s) + 5s API fetch')
