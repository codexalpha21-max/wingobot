import json
import os
import csv
import threading
import time
import traceback
import urllib.request
import requests as http_requests
from fastapi import FastAPI, APIRouter, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, ORJSONResponse, Response
from helpers import get_current_period_1min

try:
    import orjson  # type: ignore[import-not-found]  # noqa: F401
    DEFAULT_JSON_RESPONSE = ORJSONResponse
except Exception:
    DEFAULT_JSON_RESPONSE = JSONResponse

free_router = APIRouter()
predict_router = APIRouter()
main_router = APIRouter()

app = FastAPI(
    title='Wingo Prediction API',
    default_response_class=DEFAULT_JSON_RESPONSE,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)



DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
PREDICTION_HISTORY_CSV = os.path.join(DATA_DIR, 'predict', 'prediction_history.csv')
PREDICTION_HISTORY_BACKUP_CSV = PREDICTION_HISTORY_CSV + '.backup'
LEETS_FILE = os.path.join(DATA_DIR, 'leets.json')
PUBLIC_HISTORY_LIMIT = 20
REAL_HISTORY_LIMIT = 20
REAL_HISTORY_URLS = {
    '1m': 'https://wingo.oss-ap-southeast-7.aliyuncs.com/WinGo_1_{period}_past100_draws',
    '30': 'https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json',
}
_history_snapshot = []
_history_snapshot_lock = threading.Lock()
_real_history_snapshot = {'1m': [], '30': []}
_real_history_snapshot_lock = threading.Lock()
_prediction_cycle_lock_handle = None


def acquire_prediction_cycle_lock():
    global _prediction_cycle_lock_handle
    os.makedirs(DATA_DIR, exist_ok=True)
    lock_path = os.path.join(DATA_DIR, 'prediction_cycle.lock')
    handle = open(lock_path, 'a+')
    try:
        if os.name == 'nt':
            import msvcrt
            handle.seek(0)
            handle.write('0')
            handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, IOError):
        handle.close()
        return False
    _prediction_cycle_lock_handle = handle
    return True


def start_prediction_cycle():
    from run import main_loop
    if not acquire_prediction_cycle_lock():
        print("[RUN] Prediction cycle already active in another process")
        return
    t = threading.Thread(target=main_loop, daemon=True)
    t.start()
    print("[RUN] Prediction cycle started in background thread")


@app.on_event('startup')
def startup_event():
    try:
        _invalidate_history_snapshot()
    except Exception as exc:
        print(f"[STARTUP] history snapshot reload error: {exc}")
    start_prediction_cycle()
    try:
        from model_prediction import start_model_bg_refresh_loop
        start_model_bg_refresh_loop()
    except Exception as exc:
        print(f"[MODEL_BG] startup error: {exc}")
    try:
        from model_kaelis import start_kaelis_bg_refresh_loop
        start_kaelis_bg_refresh_loop()
    except Exception as exc:
        print(f"[KAELIS_BG] startup error: {exc}")
    try:
        start_data_sync_worker()
    except Exception as exc:
        print(f"[DATA_SYNC] startup error: {exc}")
    _start_warp()


def _start_warp():
    """Background thread: ping /warp every 20s to prevent Railway idle sleep."""
    t = threading.Thread(target=_warp_loop, daemon=True, name='warp')
    t.start()
    print('[WARP] Keep-alive ping active (every 20s)')


def _warp_loop():
    time.sleep(5)
    port = os.environ.get('PORT', '8000')
    while True:
        try:
            http_requests.get(f'http://127.0.0.1:{port}/warp', timeout=3)
        except Exception:
            pass
        time.sleep(20)


@main_router.get('/warp')
@main_router.post('/warp')
async def warp_endpoint():
    return {'warp': True, 'ts': int(time.time())}


def _data_sync_worker():
    """Background thread: every 30s fetch wingobot history to warm LSTM (1000 rows needed)."""
    import time as _t
    _t.sleep(5)  # let server fully start first
    while True:
        try:
            from helpers import fetch_wingobot_daily_history, load_daily_1k_history
            from ml import train_model
            from storage import load_predictions_csv

            old_count = len(load_daily_1k_history())
            rows = fetch_wingobot_daily_history(retries=2, timeout=15, limit=None)
            new_count = len(rows) if isinstance(rows, list) else 0
            added = max(0, new_count - old_count)

            if added > 0:
                print(f"[DATA_SYNC] +{added} new rows -> {new_count} total in daily_1k_history")
                try:
                    all_preds = load_predictions_csv()
                    train_model(all_preds, force=False)
                except Exception as train_err:
                    print(f"[DATA_SYNC] train error: {train_err}")
            else:
                print(f"[DATA_SYNC] {new_count} rows (no new data)")
        except Exception as exc:
            print(f"[DATA_SYNC] error: {exc}")
        _t.sleep(30)


def start_data_sync_worker():
    t = threading.Thread(target=_data_sync_worker, daemon=True)
    t.start()
    print("[DATA_SYNC] Background data sync started (every 30s)")



def read_unique_csv(filepath, limit=20):
    global _history_snapshot
    seen = {}
    with _history_snapshot_lock:
        for row in _history_snapshot:
            period = str(row.get('period', ''))
            if period:
                seen[period] = dict(row)

    sources = []
    if filepath == PREDICTION_HISTORY_CSV:
        sources.append(PREDICTION_HISTORY_BACKUP_CSV)
    sources.append(filepath)

    for source in sources:
        if not os.path.exists(source):
            continue
        for attempt in range(3):
            try:
                with open(source, 'r', newline='') as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    if not header or 'period' not in header:
                        raise ValueError('History CSV header is incomplete')
                    for row in reader:
                        if not row or len(row) < len(header):
                            continue
                        item = dict(zip(header, row))
                        period = item.get('period', '')
                        if period:
                            seen[period] = item
                break
            except (OSError, csv.Error, ValueError):
                if attempt < 2:
                    time.sleep(0.01)

    result = list(seen.values())
    result.sort(key=lambda r: period_sort_key(r.get('period', '')), reverse=True)
    if result:
        with _history_snapshot_lock:
            _history_snapshot = [dict(row) for row in result]
    return result[:limit]


def period_sort_key(period):
    try:
        return int(str(period))
    except Exception:
        return 0


def to_int(val):
    try:
        return int(float(str(val)))
    except Exception:
        return val


def blank_to_none(val):
    return None if val == '' or val is None else val


def get_number_color(number):
    number = to_int(number)
    if not isinstance(number, int):
        return None
    if number == 0:
        return 'RED,VIOLET'
    if number == 5:
        return 'GREEN,VIOLET'
    return 'GREEN' if number % 2 else 'RED'


def get_number_size(number):
    number = to_int(number)
    if not isinstance(number, int):
        return None
    return 'SMALL' if number <= 4 else 'BIG'


def clean_real_history_item(item):
    content = item.get('content') or {}
    number = content.get('number') if content else item.get('number')
    number = to_int(number) if blank_to_none(number) is not None else None
    return {
        'period': content.get('issueNumber') or item.get('issueNumber') or item.get('period') or '',
        'number': number,
        'color': (content.get('colour') or item.get('colour') or get_number_color(number)),
        'size': get_number_size(number),
    }


def fetch_real_history(game, limit=REAL_HISTORY_LIMIT):
    url = REAL_HISTORY_URLS[game]
    periods = [None]
    if '{period}' in url:
        base = read_leets().get('currentPeriod') or time.strftime('%Y%m%d100010001', time.gmtime())
        prefix = str(base)[:-5]
        number = int(str(base)[-5:])
        periods = [f"{prefix}{number - offset:05d}" for offset in range(13)]

    last_error = None
    for period in periods:
        final_url = url.format(period=period) if period else url
        req = urllib.request.Request(
            f"{final_url}?r={int(time.time() * 1000)}",
            headers={
                'Accept': 'application/json, text/plain, */*',
                'Content-Type': 'application/json;charset=UTF-8',
                'User-Agent': 'Mozilla/5.0',
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                decoded = json.loads(resp.read().decode())
            items = decoded if isinstance(decoded, list) else decoded.get('data', {}).get('list', [])
            history = [clean_real_history_item(item) for item in items[:limit]]
            if history:
                with _real_history_snapshot_lock:
                    _real_history_snapshot[game] = [dict(row) for row in history]
                return history
        except Exception as e:
            last_error = e

    # Fallback for 1m game to the robust fetch_api_data (which pings WingoBot API)
    if game == '1m':
        try:
            from helpers import fetch_api_data
            game_data = fetch_api_data(retries=2, timeout=5, bypass_cache=True)
            if game_data and isinstance(game_data, list) and 'error' not in game_data:
                history = [clean_real_history_item(item) for item in game_data[:limit]]
                if history:
                    with _real_history_snapshot_lock:
                        _real_history_snapshot[game] = [dict(row) for row in history]
                    return history
        except Exception as fallback_exc:
            print(f"[FETCH_REAL_HISTORY] fallback to fetch_api_data failed: {fallback_exc}")

    if last_error:
        raise last_error
    return []


def clean_history_entry(row, prediction=None, confidence=None, status=None, actual=None, number=None):
    actual_value = blank_to_none(actual if actual is not None else row.get('actual'))
    if actual_value == 'Unknown':
        actual_value = None
    number_value = blank_to_none(number if number is not None else row.get('number'))
    number_value = to_int(number_value) if number_value is not None else None
    color = get_number_color(number_value)
    status_value = status if status is not None else row.get('status', 'Pending')
    if str(status_value).upper() == 'ERROR':
        status_value = 'Pending'
        
    pred_val = prediction if prediction not in (None, '') else row.get('prediction')
    if not pred_val or pred_val == '':
        pred_val = 'SKIP' if str(status_value).upper() == 'SKIP' or str(row.get('status', '')).upper() == 'SKIP' else 'BIG'

    return {
        'period': row.get('period', ''),
        'prediction': pred_val,
        'status': status_value,
        'actual': actual_value,
        'number': number_value,
        'actualNumber': number_value,
        'color': color,
        'actualColor': color,
        'confidence': to_int(confidence if confidence is not None else row.get('confidence')),
        'skipped': str(row.get('skipped', '')).lower() in ('1', 'true') or str(
            status if status is not None else row.get('status', '')
        ).upper() == 'SKIP',
        'skipReason': row.get('skipreason') or row.get('skipReason') or None,
    }


def build_history_stats(history):
    wins = sum(1 for h in history if str(h.get('status', '')).upper() == 'WIN')
    losses = sum(1 for h in history if str(h.get('status', '')).upper() == 'LOSS')
    pending = sum(1 for h in history if str(h.get('status', '')).upper() == 'PENDING')
    skipped = sum(1 for h in history if str(h.get('status', '')).upper() == 'SKIP' or h.get('skipped'))
    settled = wins + losses
    win_rate = round((wins / settled) * 100, 2) if settled else 0
    recent = [h for h in history if str(h.get('status', '')).upper() in ('WIN', 'LOSS')][:10]
    recent_wins = sum(1 for h in recent if str(h.get('status', '')).upper() == 'WIN')
    recent_accuracy = round((recent_wins / len(recent)) * 100, 2) if recent else 0

    streak_status = None
    streak_count = 0
    for item in history:
        status = str(item.get('status', '')).upper()
        if status not in ('WIN', 'LOSS'):
            continue
        if streak_status is None:
            streak_status = status
            streak_count = 1
        elif status == streak_status:
            streak_count += 1
        else:
            break

    return {
        'pending': pending,
        'skipped': skipped,
        'streak': f"{streak_count} {streak_status or 'None'}",
        'totalLosses': losses,
        'totalPredictions': len(history),
        'settledPredictions': settled,
        'totalWins': wins,
        'winRate': win_rate,
        'accuracy': win_rate,
        'recentAccuracy': recent_accuracy,
    }


def build_real_draw_stats(history):
    total = len(history)
    big = sum(1 for h in history if str(h.get('size', '')).upper() == 'BIG')
    small = sum(1 for h in history if str(h.get('size', '')).upper() == 'SMALL')
    colors = {}
    numbers = {}
    for item in history:
        color = item.get('color')
        number = item.get('number')
        if color:
            colors[str(color)] = colors.get(str(color), 0) + 1
        if number is not None:
            numbers[str(number)] = numbers.get(str(number), 0) + 1
    return {
        'total': total,
        'big': big,
        'small': small,
        'bigRate': round((big / total) * 100, 2) if total else 0,
        'smallRate': round((small / total) * 100, 2) if total else 0,
        'colors': colors,
        'numbers': numbers,
        'source': 'live draw history',
    }


def build_csv_learning_payload(history):
    settled = [h for h in history if str(h.get('status', '')).upper() in ('WIN', 'LOSS')]
    losses = [h for h in settled if str(h.get('status', '')).upper() == 'LOSS']
    wins = len(settled) - len(losses)
    loss_by_prediction = {'BIG': 0, 'SMALL': 0}
    for item in losses:
        pred = str(item.get('prediction', '')).upper()
        if pred in loss_by_prediction:
            loss_by_prediction[pred] += 1

    consecutive_losses = 0
    consecutive_wins = 0
    for item in settled:
        status = str(item.get('status', '')).upper()
        if status == 'LOSS':
            if consecutive_wins:
                break
            consecutive_losses += 1
        elif status == 'WIN':
            if consecutive_losses:
                break
            consecutive_wins += 1

    recent = settled[:10]
    recent_wins = sum(1 for item in recent if str(item.get('status', '')).upper() == 'WIN')
    accuracy = round((wins / len(settled)) * 100, 2) if settled else 0
    recent_accuracy = round((recent_wins / len(recent)) * 100, 2) if recent else 0

    return {
        'source': 'prediction_history.csv',
        'live': True,
        'learnedRows': len(settled),
        'accuracy': accuracy,
        'recentAccuracy': recent_accuracy,
        'wins': wins,
        'losses': len(losses),
        'pending': sum(1 for h in history if str(h.get('status', '')).upper() == 'PENDING'),
        'consecutiveWins': consecutive_wins,
        'consecutiveLosses': consecutive_losses,
        'lossByPrediction': loss_by_prediction,
        'guardActive': consecutive_losses >= 2,
        'guardReason': (
            f"{consecutive_losses} back-to-back losses from prediction_history.csv; skip next risky round."
            if consecutive_losses >= 2 else ''
        ),
    }


def build_ml_payload(ml):
    samples = to_int((ml.get('totalSamples') or ml.get('samples') or 0) if ml else 0)
    if not isinstance(samples, int):
        samples = 0
    train_cycles = to_int(ml.get('trainCycles', 0) if ml else 0)
    if not isinstance(train_cycles, int):
        train_cycles = 0
    accuracy = to_int(ml.get('accuracy', 0) if ml else 0) if samples > 0 else None
    recent_accuracy = to_int(ml.get('recentAccuracy', 0) if ml else 0) if samples > 0 else None
    validation_samples = to_int(ml.get('validationSamples', 0) if ml else 0)
    validation = ml.get('validationAccuracy', {}) if isinstance(ml, dict) else {}
    if samples >= 100 and train_cycles > 0:
        strength = 'strong'
    elif samples >= 40 and train_cycles > 0:
        strength = 'learning'
    elif samples >= 10:
        strength = 'warming'
    else:
        strength = 'not_ready'
    return {
        'trained': train_cycles > 0 and samples > 0,
        'learning': samples > 0,
        'strength': strength,
        'accuracy': accuracy,
        'recentAccuracy': recent_accuracy,
        'samples': samples,
        'trainCycles': train_cycles,
        'lastTrainTime': ml.get('lastTrainTime') if isinstance(ml, dict) else None,
        'modelVersion': ml.get('modelVersion') if isinstance(ml, dict) else None,
        'models': ml.get('models', []) if isinstance(ml, dict) else [],
        'lightgbmAvailable': ml.get('lightgbmAvailable', False) if isinstance(ml, dict) else False,
        'xgboostAvailable': ml.get('xgboostAvailable', False) if isinstance(ml, dict) else False,
        'validationAccuracy': validation,
        'validationSamples': validation_samples if isinstance(validation_samples, int) else 0,
        'message': (
            'Model learning from verified history.'
            if samples > 0 else
            'Model has no verified training samples yet.'
        ),
    }


def build_autolearn_payload(al, history=None):
    al = al or {}
    total = to_int(al.get('totalPredictions', 0))
    correct = to_int(al.get('totalCorrect', 0))
    if isinstance(total, int) and total > 0:
        return {
            'overallAccuracy': al.get('overallAccuracy', 0),
            'totalPredictions': total,
            'totalCorrect': correct if isinstance(correct, int) else 0,
            'perPattern': al.get('perPattern', {}),
        }

    verified = []
    for row in history or []:
        status = str(row.get('status', '')).upper()
        prediction = row.get('prediction')
        actual = row.get('actual')
        if status in ('WIN', 'LOSS') and prediction and actual in ('BIG', 'SMALL'):
            verified.append(row)
    total = len(verified)
    correct = sum(1 for row in verified if str(row.get('prediction')).upper() == str(row.get('actual')).upper())
    overall = round((correct / total) * 100, 1) if total else 0
    return {
        'overallAccuracy': overall,
        'totalPredictions': total,
        'totalCorrect': correct,
        'perPattern': {
            'ensemble': {
                'accuracy': overall,
                'total': total,
                'correct': correct,
            }
        } if total else {},
    }


def _read_all_verified_predictions():
    """Read all verified (WIN/LOSS) predictions from CSV files."""
    rows = []
    sources = [
        PREDICTION_HISTORY_CSV,
        PREDICTION_HISTORY_BACKUP_CSV,
        os.path.join(DATA_DIR, 'predict', 'predictions.csv'),
        os.path.join(DATA_DIR, 'model', 'model_prediction_history.csv'),
    ]
    seen = set()
    for path in sources:
        if not os.path.exists(path):
            continue
        try:
            with open(path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    period = str(row.get('period') or '').strip()
                    status = str(row.get('status') or '').upper().strip()
                    if not period or period in seen:
                        continue
                    if status in ('WIN', 'LOSS'):
                        seen.add(period)
                        rows.append({
                            'period': period,
                            'prediction': row.get('prediction', ''),
                            'actual': row.get('actual', ''),
                            'status': status,
                            'pattern': str(row.get('patternused') or row.get('patternUsed') or 'ensemble'),
                        })
        except Exception:
            pass
    return rows


def _compute_winloss(verified_rows):
    wins = sum(1 for r in verified_rows if r['status'] == 'WIN')
    losses = sum(1 for r in verified_rows if r['status'] == 'LOSS')
    total = wins + losses
    win_rate = round((wins / total) * 100, 2) if total else 0.0
    recent = verified_rows[:50]
    recent_wins = sum(1 for r in recent if r['status'] == 'WIN')
    recent_total = len(recent)
    recent_acc = round((recent_wins / recent_total) * 100, 2) if recent_total else 0.0
    return {
        'wins': wins,
        'losses': losses,
        'total': total,
        'winRate': win_rate,
        'recentAccuracy': recent_acc,
    }


def _model_progression(total_samples, train_cycles):
    if total_samples >= 100 and train_cycles > 0:
        return 'strong'
    if total_samples >= 40 and train_cycles > 0:
        return 'learning'
    if total_samples >= 10:
        return 'warming'
    return 'not_ready'


def _model_status_label(progression):
    return {
        'strong': 'ready',
        'learning': 'active',
        'warming': 'initializing',
        'not_ready': 'pending',
    }.get(progression, 'unknown')


def build_model_status_section(model_names, leets=None, verified_rows=None):
    """Build per-model status for a list of model names."""
    leets = leets or {}
    if verified_rows is None:
        verified_rows = _read_all_verified_predictions()
    winloss = _compute_winloss(verified_rows)
    ml = leets.get('mlModel', {})
    ml_samples = ml.get('totalSamples') or ml.get('samples') or 0
    ml_cycles = ml.get('trainCycles') or 0

    from model_brain import get_model_knowledge, load_brain
    load_brain()

    analytical_models = {'pattern', 'transition', 'number', 'sequence', 'loss', 'brain'}
    models_out = []
    for mname in model_names:
        knowledge = get_model_knowledge(mname) or {}
        prog = _model_progression(ml_samples if mname in ('ml', 'ensemble') else winloss['total'], ml_cycles)

        brain_rate = knowledge.get('winRate')
        if brain_rate is not None:
            wins = knowledge.get('wins', 0)
            losses = knowledge.get('losses', 0)
            total = wins + losses
            win_rate = round(brain_rate * 100, 2)
        else:
            wins = 0; losses = 0; total = 0; win_rate = 0.0

        accuracy = win_rate

        entry = {
            'name': mname,
            'status': _model_status_label(prog),
            'progression': prog,
            'totalData': total,
            'wins': wins,
            'losses': losses,
            'winRate': win_rate,
            'accuracy': accuracy,
            'learningSaved': bool(knowledge),
            'knowledge': knowledge if knowledge else None,
        }
        if mname == 'ml' and knowledge.get('subModels'):
            entry['subModels'] = list(knowledge['subModels'].keys())
        models_out.append(entry)
    return models_out


def compute_realtime_confidence(prediction, leets=None, recent_results=None, model_name='ensemble'):
    from model_brain import get_model_knowledge, load_brain
    load_brain()

    ml = (leets or {}).get('mlModel', {})
    ml_acc = ml.get('accuracy') or ml.get('recentAccuracy') or 50
    pattern_k = get_model_knowledge('pattern') or {}
    transition_k = get_model_knowledge('transition') or {}
    sequence_k = get_model_knowledge('sequence') or {}
    loss_k = get_model_knowledge('loss') or {}

    seq_conf = 0.0
    if recent_results and len(recent_results) >= 2 and sequence_k:
        for length in [2, 3, 4]:
            if len(recent_results) >= length:
                key = '_'.join(recent_results[:length])
                sd = (sequence_k.get(f'seq{length}', {}) or {}).get(key)
                if sd:
                    p = sd.get('toBig' if prediction == 'BIG' else 'toSmall', 0.5)
                    seq_conf += (p - 0.5) * (1.5 ** length) * 20

    trans_conf = 0.0
    if recent_results and len(recent_results) > 0:
        last = recent_results[0]
        td = (transition_k.get(last) or {}) if last in transition_k else {}
        if td:
            p = td.get('toBig' if prediction == 'BIG' else 'toSmall', 0.5)
            trans_conf = (p - 0.5) * 40

    big_ratio = pattern_k.get('bigRatio', 0.5)
    pattern_conf = (big_ratio - 0.5) * 40 if prediction == 'BIG' else (0.5 - big_ratio) * 40

    loss_conf = 0.0
    if loss_k:
        bal = loss_k.get('bigAfterLossRatio', 0.5)
        sal = loss_k.get('smallAfterLossRatio', 0.5)
        if prediction == 'BIG':
            loss_conf = (bal - 0.5) * 30
        else:
            loss_conf = (sal - 0.5) * 30

    base = 55.0
    total = base + seq_conf + trans_conf + pattern_conf + loss_conf + (ml_acc - 50) * 0.3
    total = max(50.0, min(95.0, total))
    return round(total, 2)


def build_public_history(limit=PUBLIC_HISTORY_LIMIT, current=None, fallback_rows=None):
    csv_rows = read_unique_csv(PREDICTION_HISTORY_CSV, limit=5000)
    merged_rows = {}
    for row in fallback_rows or []:
        pattern_used = row.get('patternused') or row.get('patternUsed') or ''
        if str(pattern_used).lower() == 'imported':
            continue
        period = str(row.get('period', ''))
        if period:
            merged_rows[period] = row
    for row in csv_rows:
        pattern_used = row.get('patternused') or row.get('patternUsed') or ''
        if str(pattern_used).lower() == 'imported':
            continue
        period = str(row.get('period', ''))
        if period:
            merged_rows[period] = row
    history_raw = list(merged_rows.values())
    history_raw.sort(key=lambda r: period_sort_key(r.get('period', '')), reverse=True)
    current = current or {}
    curr_period = current.get('period')
    curr_pred = current.get('prediction')
    curr_conf = current.get('confidence')
    curr_status = current.get('status', 'Pending') or 'Pending'
    curr_actual = current.get('actual')
    curr_number = current.get('number')

    history = []
    added_current = False
    for row in history_raw:
        if curr_period and row.get('period') == curr_period:
            row_status = str(row.get('status', '')).upper()
            settled = row_status in ('WIN', 'LOSS', 'SKIP', 'ERROR')
            history.append(clean_history_entry(
                row,
                prediction=row.get('prediction') or curr_pred,
                confidence=row.get('confidence') or curr_conf,
                status=row.get('status') if settled else curr_status,
                actual=row.get('actual') if settled else curr_actual,
                number=row.get('number') if settled else curr_number,
            ))
            added_current = True
        else:
            history.append(clean_history_entry(row))

    if not added_current and curr_period and curr_pred:
        history.insert(0, clean_history_entry({
            'period': curr_period,
            'prediction': curr_pred,
            'status': curr_status,
            'actual': curr_actual,
            'number': curr_number,
            'confidence': curr_conf,
        }))

    history.sort(key=lambda r: period_sort_key(r.get('period', '')), reverse=True)
    return history[:limit]


def fill_visible_history_gaps(history, limit=PUBLIC_HISTORY_LIMIT):
    if not history:
        return history
    by_period = {str(row.get('period')): dict(row) for row in history if row.get('period')}
    sorted_rows = sorted(by_period.values(), key=lambda r: period_sort_key(r.get('period')), reverse=True)
    filled = []
    for idx, row in enumerate(sorted_rows):
        filled.append(row)
        if len(filled) >= limit:
            break
        if idx >= len(sorted_rows) - 1:
            continue
        current_key = period_sort_key(row.get('period'))
        next_key = period_sort_key(sorted_rows[idx + 1].get('period'))
        gap = current_key - next_key
        if gap <= 1 or gap > 10:
            continue
        fallback_prediction = row.get('prediction') if row.get('prediction') in ('BIG', 'SMALL') else sorted_rows[idx + 1].get('prediction')
        fallback_confidence = row.get('confidence') or sorted_rows[idx + 1].get('confidence') or 55
        for missing_key in range(current_key - 1, next_key, -1):
            missing_period = str(missing_key)
            if missing_period in by_period:
                continue
            filled.append(clean_history_entry({
                'period': missing_period,
                'prediction': fallback_prediction if fallback_prediction in ('BIG', 'SMALL') else 'BIG',
                'status': 'Pending',
                'confidence': fallback_confidence,
                'actual': None,
                'number': None,
                'skipped': False,
                'skipReason': 'Auto-filled missing visible history period.',
            }))
            if len(filled) >= limit:
                break
    filled.sort(key=lambda r: period_sort_key(r.get('period', '')), reverse=True)
    return filled[:limit]


def _invalidate_history_snapshot():
    """Reload the in-memory snapshot from disk so subsequent reads see fresh data.
    Never clears to empty — that would cause read_unique_csv to return nothing
    if a concurrent file operation briefly makes the CSV unavailable."""
    global _history_snapshot
    seen = {}
    with _history_snapshot_lock:
        for row in _history_snapshot:
            period = str(row.get('period', ''))
            if period:
                seen[period] = dict(row)

    for source in (PREDICTION_HISTORY_BACKUP_CSV, PREDICTION_HISTORY_CSV):
        if not os.path.exists(source):
            continue
        try:
            with open(source, 'r', newline='') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header or 'period' not in header:
                    continue
                for row in reader:
                    if not row or len(row) < len(header):
                        continue
                    item = dict(zip(header, row))
                    period = item.get('period', '')
                    if period:
                        seen[period] = item
        except Exception:
            continue
    with _history_snapshot_lock:
        fresh = list(seen.values())
        if fresh:
            _history_snapshot = fresh
        # If we couldn't read anything, keep old snapshot rather than wiping it


def safe_persist_prediction(period, prediction, status, confidence, actual, number,
                             pattern_used, skipped, skip_reason):
    """
    Persist a prediction to prediction_history.csv ONLY if the period is not already there.
    This prevents history gaps when the background runner is briefly offline.
    After writing, invalidates the snapshot cache.
    """
    if not period or str(prediction).upper() not in ('BIG', 'SMALL', 'SKIP'):
        return
    try:
        from storage import upsert_prediction_history_csv, load_prediction_history_entries
        # Check if period already exists
        existing = load_prediction_history_entries(limit=None)
        already = any(str(e.get('period', '')) == str(period) for e in existing)
        if not already:
            upsert_prediction_history_csv({
                'period': str(period),
                'prediction': prediction,
                'status': status or 'Pending',
                'confidence': confidence or 0,
                'actual': actual,
                'number': number,
                'patternUsed': pattern_used or 'ensemble',
                'timestamp': int(time.time()),
                'skipped': bool(skipped),
                'skipReason': skip_reason or '',
            })
            _invalidate_history_snapshot()
    except Exception:
        pass


def hydrate_history_with_live_results(history):
    pending = [
        item for item in history
        if item.get('period')
        and item.get('actual') is None
        and str(item.get('status', '')).upper() in ('PENDING', 'SKIP')
    ]
    if not pending:
        return history

    try:
        from helpers import fetch_api_data, load_daily_1k_history
        from storage import upsert_prediction_history_csv
        game_data = fetch_api_data(retries=2, timeout=5, bypass_cache=False)
    except Exception:
        return history

    # Build lookup from live API results
    by_period = {}
    if isinstance(game_data, list):
        for item in game_data:
            p = str(item.get('period', ''))
            if p:
                by_period[p] = item

    # Also load daily_1k_history as fallback for older periods not in live API
    try:
        daily = load_daily_1k_history(limit=None)
        for item in daily:
            p = str(item.get('period', ''))
            if p and p not in by_period:
                # Normalise daily history entry to match game_data shape
                by_period[p] = {
                    'period': p,
                    'category': item.get('category'),
                    'number': item.get('number'),
                }
    except Exception:
        pass

    changed = False
    for item in history:
        period = str(item.get('period', ''))
        row_status = str(item.get('status', '')).upper()
        if not period or row_status not in ('PENDING', 'SKIP') or item.get('actual') is not None:
            continue

        match = by_period.get(period)
        if not match:
            # Try suffix match (last 3 digits) as last resort against live data only
            suffix = period[-3:]
            if isinstance(game_data, list):
                match = next(
                    (row for row in game_data if str(row.get('period', '')).endswith(suffix)),
                    None,
                )
        if not match:
            continue

        actual = match.get('category')
        number = to_int(match.get('number')) if blank_to_none(match.get('number')) is not None else None
        if actual not in ('BIG', 'SMALL'):
            continue

        item['actual'] = actual
        item['number'] = number
        item['actualNumber'] = number
        item['color'] = get_number_color(number)
        item['actualColor'] = get_number_color(number)
        if item.get('prediction') and str(item.get('prediction')).upper() in ('BIG', 'SMALL'):
            item['status'] = 'WIN' if item.get('prediction') == actual else 'LOSS'
        else:
            item['status'] = 'SKIP'

        # Preserve original patternUsed; fall back to 'ensemble' only if missing
        original_pattern = item.get('patternUsed') or item.get('patternused') or 'ensemble'
        if str(original_pattern).lower() == 'imported':
            original_pattern = 'ensemble'

        try:
            upsert_prediction_history_csv({
                'period': period,
                'prediction': item.get('prediction'),
                'status': item.get('status'),
                'confidence': item.get('confidence', 0),
                'actual': actual,
                'number': number,
                'patternUsed': original_pattern,
                'timestamp': int(time.time()),
                'skipped': item.get('skipped', False),
                'skipReason': item.get('skipReason') or '',
            })
        except Exception:
            pass
        changed = True

    if changed:
        _invalidate_history_snapshot()
        history.sort(key=lambda r: period_sort_key(r.get('period', '')), reverse=True)
    return history


def read_leets():
    if not os.path.exists(LEETS_FILE):
        return {}
    try:
        with open(LEETS_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


REQUIRED_MODEL_NAME = 'kaelis'
REQUIRED_MODEL_KEY = 'kaelis.ai/paid/models'


def _verify_payload(body: dict) -> bool:
    return body.get('model_name') == REQUIRED_MODEL_NAME and body.get('model_key') == REQUIRED_MODEL_KEY


_ACCESS_DENIED = JSONResponse(
    status_code=403,
    content={
        'success': False,
        'error': 'Access denied',
    },
)


@free_router.get('/v2/free')
@free_router.post('/v2/free')
async def v2_free(request: Request):
    body = {}
    if request.method == 'POST':
        try:
            body = await request.json()
        except Exception:
            body = {}
    if not _verify_payload(body):
        return _ACCESS_DENIED
    from free_prediction import get_free_payload
    payload = get_free_payload()
    leets = read_leets()
    payload['models'] = build_model_status_section(
        ['pattern', 'transition', 'number', 'sequence'], leets=leets,
    )
    return Response(
        content=json.dumps(payload, indent=2, ensure_ascii=False),
        media_type="application/json",
    )


@main_router.get('/v2/colour')
@main_router.get('/v2/color')
@main_router.post('/v2/colour')
@main_router.post('/v2/color')
async def v2_colour(request: Request):
    body = {}
    if request.method == 'POST':
        try:
            body = await request.json()
        except Exception:
            body = {}
    if not _verify_payload(body):
        return _ACCESS_DENIED
    try:
        from color_prediction import get_color_prediction_payload
        payload = get_color_prediction_payload(body)
        leets = read_leets()
        payload['models'] = build_model_status_section(
            ['lstm', 'bilstm', 'ml', 'pattern', 'sequence'], leets=leets,
        )
        pred = payload.get('predictionResult', {}).get('prediction')
        hist = payload.get('history', [])
        recent_results = [h.get('actual') for h in hist if h.get('actual') in ('BIG', 'SMALL')][:10]
        rt_conf = compute_realtime_confidence(pred, leets=leets, recent_results=recent_results)
        if payload.get('predictionDetails'):
            payload['predictionDetails']['confidence'] = rt_conf
        return payload
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                'success': False,
                'route': '/v2/colour',
                'error': str(exc),
                'trace': traceback.format_exc().splitlines()[-8:],
            },
        )


@main_router.post('/model/predict')
async def v2_model(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not _verify_payload(body):
        return _ACCESS_DENIED
    try:
        from model_prediction import get_cached_model_payload
        payload = get_cached_model_payload()
        leets = read_leets()
        payload['models'] = build_model_status_section(
            ['lstm', 'bilstm', 'ml', 'ensemble'], leets=leets,
        )
        pred = payload.get('predictionResult', {}).get('prediction')
        if payload.get('predictionDetails'):
            hist = payload.get('history', [])
            recent_results = [h.get('actual') for h in hist if h.get('actual') in ('BIG', 'SMALL')][:10]
            rt_conf = compute_realtime_confidence(pred, leets=leets, recent_results=recent_results)
            payload['predictionDetails']['confidence'] = rt_conf
        payload['boostingModels'] = ['XGBoost', 'LightGBM', 'CatBoost']
        ml_section = payload.get('ml', {})
        if ml_section:
            ml_section['boostingModels'] = ['XGBoost', 'LightGBM', 'CatBoost']
            ml_section['catboostAvailable'] = True
        return payload
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                'success': False,
                'route': '/model/predict',
                'error': str(exc),
                'trace': traceback.format_exc().splitlines()[-8:],
            },
        )


@main_router.get('/model/kaelis')
@main_router.post('/model/kaelis')
async def v2_kaelis(request: Request):
    try:
        from model_kaelis import get_cached_kaelis_payload
        payload = get_cached_kaelis_payload()
        return payload
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                'success': False,
                'route': '/model/kaelis',
                'error': str(exc),
                'trace': traceback.format_exc().splitlines()[-8:],
            },
        )


def read_json_file(filepath):
    if not os.path.exists(filepath):
        return {"success": True, "current": {}, "stats": {}, "api_status": {}, "history": []}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


@main_router.get('/v2/history')
def history(request: Request):
    if not _verify_payload(dict(request.query_params)):
        return _ACCESS_DENIED
    leets = read_leets()
    history_rows = build_public_history(
        limit=PUBLIC_HISTORY_LIMIT,
        fallback_rows=leets.get('history', []),
    )
    history_rows = hydrate_history_with_live_results(history_rows)
    history_rows = history_rows[:PUBLIC_HISTORY_LIMIT]
    verified_rows = _read_all_verified_predictions()
    return {
        'success': True,
        'history': history_rows,
        'stats': build_history_stats(history_rows),
        'historySource': {
            'file': 'prediction_history.csv',
            'live': True,
            'rows': len(history_rows),
            'limit': PUBLIC_HISTORY_LIMIT,
        },
        'models': build_model_status_section(
            ['pattern', 'transition', 'number', 'sequence', 'loss', 'brain'],
            leets=leets, verified_rows=verified_rows,
        ),
    }


@main_router.get('/v2/history/1m')
def history_1m():
    try:
        history_rows = fetch_real_history('1m')
        stale = False
        if not history_rows:
            with _real_history_snapshot_lock:
                history_rows = [dict(row) for row in _real_history_snapshot.get('1m', [])]
            stale = bool(history_rows)
        return {
            'success': True,
            'game': '1m',
            'limit': REAL_HISTORY_LIMIT,
            'rows': len(history_rows),
            'stale': stale,
            'history': history_rows,
            'stats': build_real_draw_stats(history_rows),
        }
    except Exception as e:
        with _real_history_snapshot_lock:
            history_rows = [dict(row) for row in _real_history_snapshot.get('1m', [])]
        if history_rows:
            return {
                'success': True,
                'game': '1m',
                'limit': REAL_HISTORY_LIMIT,
                'rows': len(history_rows),
                'stale': True,
                'warning': str(e),
                'history': history_rows,
                'stats': build_real_draw_stats(history_rows),
            }
        return JSONResponse(
            status_code=502,
            content={'success': False, 'game': '1m', 'error': str(e), 'history': []},
        )


@main_router.get('/v2/history/30s')
def history_30s():
    try:
        history_rows = fetch_real_history('30')
        stale = False
        if not history_rows:
            with _real_history_snapshot_lock:
                history_rows = [dict(row) for row in _real_history_snapshot.get('30', [])]
            stale = bool(history_rows)
        return {
            'success': True,
            'game': '30s',
            'limit': REAL_HISTORY_LIMIT,
            'rows': len(history_rows),
            'stale': stale,
            'history': history_rows,
            'stats': build_real_draw_stats(history_rows),
        }
    except Exception as e:
        with _real_history_snapshot_lock:
            history_rows = [dict(row) for row in _real_history_snapshot.get('30', [])]
        if history_rows:
            return {
                'success': True,
                'game': '30s',
                'limit': REAL_HISTORY_LIMIT,
                'rows': len(history_rows),
                'stale': True,
                'warning': str(e),
                'history': history_rows,
                'stats': build_real_draw_stats(history_rows),
            }
        return JSONResponse(
            status_code=502,
            content={'success': False, 'game': '30s', 'error': str(e), 'history': []},
        )

@main_router.get('/v2/ml/patterns')
async def v2_ml_patterns(request: Request):
    """Return live pattern analysis: which N-gram sequences are winning vs losing,
    the current trend regime, loss concentration by prediction side, and HOT/COLD
    pattern flags. Useful for debugging the ML self-learning system."""
    if not _verify_payload(dict(request.query_params)):
        return _ACCESS_DENIED
    try:
        from ml import get_pattern_analysis, build_sequence_training_rows
        from storage import load_predictions_csv
        all_preds = load_predictions_csv()
        rows = build_sequence_training_rows(all_preds)
        analysis = get_pattern_analysis(rows, n=50)
        leets = read_leets()
        return {
            'success': True,
            'totalHistoryRows': len(rows),
            'patternAnalysis': analysis,
            'models': build_model_status_section(
                ['ml', 'pattern', 'sequence'], leets=leets,
            ),
        }
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                'success': False,
                'error': str(exc),
                'trace': traceback.format_exc().splitlines()[-6:],
            },
        )


@main_router.get('/v2/ml/status')
async def v2_ml_status(request: Request):
    """Return a concise ML model status summary including training cycle info,
    validation accuracy per model, and loss learning profile."""
    if not _verify_payload(dict(request.query_params)):
        return _ACCESS_DENIED
    try:
        from ml import get_model_summary, get_pattern_analysis, build_sequence_training_rows
        from storage import load_predictions_csv
        summary = get_model_summary()
        all_preds = load_predictions_csv()
        rows = build_sequence_training_rows(all_preds)
        analysis = get_pattern_analysis(rows, n=50)
        leets = read_leets()
        return {
            'success': True,
            'ml': summary,
            'patternSummary': {
                'regime':        analysis.get('regime'),
                'recentBigRate': analysis.get('recentBigRate'),
                'currentStreak': analysis.get('currentStreak'),
                'lossConcentration': analysis.get('lossConcentration'),
                'topFailingCount': len(analysis.get('topFailingPatterns', [])),
                'topWinningCount': len(analysis.get('topWinningPatterns', [])),
            },
            'models': build_model_status_section(
                ['lstm', 'bilstm', 'ml', 'ensemble'], leets=leets,
            ),
        }
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={'success': False, 'error': str(exc)},
        )


app.include_router(main_router)
app.include_router(free_router)
app.include_router(predict_router)


@main_router.get('/demo/kaelis')
async def serve_kaelis_demo():
    html_path = os.path.join(os.path.dirname(__file__), 'demo_kaelis.html')
    if os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read(), status_code=200)
    return HTMLResponse('<h2>demo_kaelis.html not found</h2>', status_code=404)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    workers = int(os.environ.get('WORKERS', 1))
    print(f"[API] FastAPI server on port {port} with {workers} worker(s)")
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=port, workers=workers)
