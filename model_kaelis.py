import csv
import json
import os
import threading
import time
import traceback
import pickle
import math
import concurrent.futures

from collections import Counter, defaultdict
import numpy as np
from helpers import fetch_api_data, fetch_wingobot_daily_history, get_current_period_1min, get_oss_data_status, normalize_side, verified_outcome
from ml import predict_ml, predict_lstm_bilstm, train_model, get_model_summary
from config import DATA_DIR


BASE_DIR = os.path.dirname(__file__)
KAELIS_HISTORY_CSV = os.path.join(DATA_DIR, 'model', 'kaelis_prediction_history.csv')
KAELIS_HISTORY_LIMIT = 20
KAELIS_CACHE_FILE = os.path.join(DATA_DIR, 'kaelis_cache.json')
KAELIS_CACHE_STALE_SECONDS = 120
KAELIS_BG_REFRESH_INTERVAL = 30
KAELIS_BRAIN_FILE = os.path.join(DATA_DIR, 'model_brain', 'kaelis_brain.pkl')
KAELIS_MODEL_FILE = os.path.join(DATA_DIR, 'model_brain', 'kaelis_model.pkl')
KAELIS_MODEL_NAMES = ['XGBoost', 'LightGBM', 'LSTM']

HEADER = [
    'id', 'period', 'prediction', 'status', 'confidence',
    'actual', 'number', 'patternused', 'timestamp',
    'skipped', 'skipreason', 'created_at'
]

_lock = threading.RLock()
_history_snapshot = None  # None = not loaded; [] = empty after load
_payload_cache = None
_payload_cache_time = 0
_PAYLOAD_CACHE_SECONDS = 12
_bg_refresh_thread = None
_bg_refresh_lock = threading.Lock()
_last_predict_time = 0
_last_period = None
_active_period_prediction = None
# In-memory entry store (always source of truth, CSV is backup)
_memory_entries = {}  # period -> dict
_memory_entries_lock = threading.Lock()
_verified_periods = set()  # periods already verified — never re-verify
_verified_periods_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Kaelis Learner – smart self-learning engine
# ---------------------------------------------------------------------------

class KaelisLearner:
    def __init__(self):
        self.models = {'XGBoost': {'wins':0,'losses':0,'total':0,'acc':50.0,'recent_wins':0,'recent_losses':0,'recent_acc':50.0,'consecutive_losses':0,'consecutive_wins':0},
                       'LightGBM': {'wins':0,'losses':0,'total':0,'acc':50.0,'recent_wins':0,'recent_losses':0,'recent_acc':50.0,'consecutive_losses':0,'consecutive_wins':0},
                       'LSTM': {'wins':0,'losses':0,'total':0,'acc':50.0,'recent_wins':0,'recent_losses':0,'recent_acc':50.0,'consecutive_losses':0,'consecutive_wins':0}}
        self.weights = {'XGBoost': 0.35, 'LightGBM': 0.35, 'LSTM': 0.30}
        self.total_predictions = 0
        self.total_wins = 0
        self.total_losses = 0
        self.recent_results = []
        self.consecutive_losses = 0
        self.loss_recovery_mode = False
        self.recovery_side = None
        self.pattern_memory = defaultdict(lambda: {'wins':0,'losses':0,'total':0})
        self.regime_memory = defaultdict(lambda: {'wins':0,'losses':0,'total':0})
        self.side_memory = defaultdict(lambda: {'wins':0,'losses':0,'total':0})  # BIG/SMALL performance
        self.zigzag_memory = defaultdict(lambda: {'wins':0,'losses':0,'total':0})  # zigzag pattern memory
        self.streak_memory = defaultdict(lambda: {'wins':0,'losses':0,'total':0})  # streak pattern memory
        self.last_learn_time = 0

    def learn(self, model_name, prediction, actual, won):
        if model_name not in self.models:
            return
        m = self.models[model_name]
        m['total'] += 1
        if won:
            m['wins'] += 1
            m['recent_wins'] += 1
            m['consecutive_wins'] += 1
            m['consecutive_losses'] = 0
        else:
            m['losses'] += 1
            m['recent_losses'] += 1
            m['consecutive_losses'] += 1
            m['consecutive_wins'] = 0
        t = m['wins'] + m['losses']
        m['acc'] = round((m['wins']/t)*100, 2) if t else 50.0
        rt = m['recent_wins'] + m['recent_losses']
        if rt >= 15:
            m['recent_acc'] = round((m['recent_wins']/rt)*100, 2)
            m['recent_wins'] = 0
            m['recent_losses'] = 0
        _save_model_brain(model_name, dict(m))

    def learn_outcome(self, prediction, actual, model_details, pattern_name=None, regime=None, streak_len=0, zigzag_count=0, prev_actual=None):
        if not actual or not prediction or actual not in ('BIG','SMALL') or prediction not in ('BIG','SMALL'):
            return
        won = prediction == actual
        self.total_predictions += 1
        if won:
            self.total_wins += 1
        else:
            self.total_losses += 1
        self.recent_results.insert(0, actual)
        if len(self.recent_results) > 100:
            self.recent_results = self.recent_results[:100]

        if not won:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
            self.loss_recovery_mode = False
            self.recovery_side = None

        for d in model_details:
            mn = d.get('model','')
            mp = d.get('prediction')
            if mp in ('BIG','SMALL') and mp == prediction:
                self.learn(mn, mp, actual, won)

        self._adjust_weights()
        self._check_loss_recovery(prediction, actual)

        # Learn predicted side performance
        side_key = f"predict_{prediction}"
        sm = self.side_memory[side_key]
        sm['total'] += 1
        if won: sm['wins'] += 1
        else: sm['losses'] += 1

        # Learn actual side performance  
        actual_key = f"actual_{actual}"
        am = self.side_memory[actual_key]
        am['total'] += 1
        if won: am['wins'] += 1
        else: am['losses'] += 1

        # Learn pattern memory
        if pattern_name:
            pm = self.pattern_memory[pattern_name]
            pm['total'] += 1
            if won: pm['wins'] += 1
            else: pm['losses'] += 1

        # Learn regime memory
        if regime:
            rm = self.regime_memory[regime]
            rm['total'] += 1
            if won: rm['wins'] += 1
            else: rm['losses'] += 1

        # Learn zigzag pattern
        z_key = 'zigzag_active' if zigzag_count >= 3 else 'zigzag_inactive'
        zm = self.zigzag_memory[z_key]
        zm['total'] += 1
        if won: zm['wins'] += 1
        else: zm['losses'] += 1

        # Learn streak pattern
        if streak_len >= 3:
            s_key = f"streak_{streak_len}_{actual}"
            sm2 = self.streak_memory[s_key]
            sm2['total'] += 1
            if won: sm2['wins'] += 1
            else: sm2['losses'] += 1

        # Learn alternation (zigzag) pattern
        if prev_actual and prev_actual != actual:
            alt_key = f"alternate_{prev_actual}_to_{actual}"
            am2 = self.zigzag_memory[alt_key]
            am2['total'] += 1
            if won: am2['wins'] += 1
            else: am2['losses'] += 1

    def _adjust_weights(self):
        total_acc = 0
        accs = {}
        for name in self.weights:
            m = self.models[name]
            if m['total'] >= 5:
                a = m['recent_acc'] if m['recent_wins']+m['recent_losses'] >= 10 else m['acc']
            else:
                a = 50.0
            accs[name] = a
            total_acc += a
        if total_acc > 0 and len(accs) >= 2:
            for name in self.weights:
                self.weights[name] = max(0.05, min(0.70, accs[name]/total_acc * 3.0))
            wsum = sum(self.weights.values())
            if wsum > 0:
                for name in self.weights:
                    self.weights[name] /= wsum

    def _check_loss_recovery(self, prediction, actual):
        if self.consecutive_losses >= 1:
            if not self.loss_recovery_mode:
                self.loss_recovery_mode = True
            self.recovery_side = actual

    def get_recovery_adjustment(self):
        if self.loss_recovery_mode and self.recovery_side:
            return {
                'active': True,
                'side': self.recovery_side,
                'consecutive_losses': self.consecutive_losses,
                'boost': min(18 + self.consecutive_losses * 10, 45),
            }
        return {'active': False, 'side': None, 'consecutive_losses': self.consecutive_losses, 'boost': 0}

    def get_pattern_accuracy(self, pattern_name):
        pm = self.pattern_memory.get(pattern_name)
        if not pm or pm['total'] < 3:
            return None
        return round((pm['wins']/pm['total'])*100, 1)

    def get_side_accuracy(self, side):
        sm = self.side_memory.get(f"predict_{side}")
        if not sm or sm['total'] < 3:
            return None
        return round((sm['wins']/sm['total'])*100, 1)

    def get_regime_accuracy(self, regime):
        rm = self.regime_memory.get(regime)
        if not rm or rm['total'] < 3:
            return None
        return round((rm['wins']/rm['total'])*100, 1)

    def get_zigzag_accuracy(self):
        zm = self.zigzag_memory.get('zigzag_active')
        if not zm or zm['total'] < 3:
            return None
        return round((zm['wins']/zm['total'])*100, 1)

    def get_stats(self):
        best_pattern = max(self.pattern_memory.items(), key=lambda x: x[1]['wins']/max(x[1]['total'],1)) if self.pattern_memory else None
        best_side = max(self.side_memory.items(), key=lambda x: x[1]['wins']/max(x[1]['total'],1)) if self.side_memory else None
        return {
            'totalPredictions': self.total_predictions,
            'totalWins': self.total_wins,
            'totalLosses': self.total_losses,
            'winRate': round((self.total_wins/max(self.total_wins+self.total_losses,1))*100, 2),
            'consecutiveLosses': self.consecutive_losses,
            'lossRecoveryMode': self.loss_recovery_mode,
            'recoverySide': self.recovery_side,
            'models': {k: {'accuracy':v['acc'],'recentAccuracy':v['recent_acc'],'total':v['total'],'weight':round(self.weights[k],4)} for k,v in self.models.items()},
            'weights': {k: round(v,4) for k,v in self.weights.items()},
            'bestPattern': {'name': best_pattern[0], 'accuracy': round((best_pattern[1]['wins']/max(best_pattern[1]['total'],1))*100,1)} if best_pattern and best_pattern[1]['total']>=3 else None,
            'bestSide': {'side': best_side[0], 'accuracy': round((best_side[1]['wins']/max(best_side[1]['total'],1))*100,1)} if best_side and best_side[1]['total']>=3 else None,
            'patternMemory': {k: {'wins':v['wins'],'losses':v['losses'],'total':v['total'],'acc':round((v['wins']/max(v['total'],1))*100,1)} for k,v in sorted(self.pattern_memory.items(), key=lambda x:-x[1]['total'])[:10]},
            'sideMemory': {k: {'wins':v['wins'],'losses':v['losses'],'total':v['total'],'acc':round((v['wins']/max(v['total'],1))*100,1)} for k,v in sorted(self.side_memory.items(), key=lambda x:-x[1]['total'])[:10]},
            'zigzagAccuracy': self.get_zigzag_accuracy(),
        }


_learner = None
_learner_lock = threading.Lock()

def _get_learner():
    global _learner
    if _learner is not None:
        return _learner
    with _learner_lock:
        if _learner is not None:
            return _learner
        try:
            os.makedirs(os.path.dirname(KAELIS_BRAIN_FILE), exist_ok=True)
            if os.path.exists(KAELIS_BRAIN_FILE):
                with open(KAELIS_BRAIN_FILE, 'rb') as f:
                    _learner = pickle.load(f)
                return _learner
        except Exception:
            pass
        _learner = KaelisLearner()
        return _learner

def _save_learner():
    global _learner
    if _learner is None:
        return
    try:
        os.makedirs(os.path.dirname(KAELIS_BRAIN_FILE), exist_ok=True)
        tmp = KAELIS_BRAIN_FILE + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(_learner, f)
        os.replace(tmp, KAELIS_BRAIN_FILE)
    except Exception:
        pass

_MODEL_BRAIN_DIR = os.path.join(DATA_DIR, 'model_brain', 'kaelis_models')

def _save_model_brain(model_name, data):
    try:
        os.makedirs(_MODEL_BRAIN_DIR, exist_ok=True)
        path = os.path.join(_MODEL_BRAIN_DIR, f'{model_name}.pkl')
        tmp = path + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(data, f)
        os.replace(tmp, path)
    except Exception:
        pass

def _load_model_brains():
    brains = {}
    if not os.path.isdir(_MODEL_BRAIN_DIR):
        return brains
    for fname in os.listdir(_MODEL_BRAIN_DIR):
        if fname.endswith('.pkl'):
            model_name = fname[:-4]
            try:
                with open(os.path.join(_MODEL_BRAIN_DIR, fname), 'rb') as f:
                    brains[model_name] = pickle.load(f)
            except Exception:
                pass
    return brains

def _get_model_accuracy(model_name):
    brains = _load_model_brains()
    return brains.get(model_name)

# ---------------------------------------------------------------------------
# Load ALL history from every source
# ---------------------------------------------------------------------------

def _discover_all_csvs():
    csvs = set()
    # Scan data/ and its subdirs recursively
    if os.path.isdir(DATA_DIR):
        for root, dirs, files in os.walk(DATA_DIR):
            for f in files:
                if f.endswith('.csv') and f != '.gitkeep':
                    csvs.add(os.path.join(root, f))
    # Scan project root
    for d in [BASE_DIR, os.path.join(BASE_DIR, 'predict'), os.path.join(BASE_DIR, 'free')]:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith('.csv'):
                    csvs.add(os.path.join(d, f))
    return sorted(csvs)


def _load_all_history():
    by_period = {}
    for path in _discover_all_csvs():
        if not os.path.exists(path):
            continue
        try:
            with open(path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    period = str(row.get('period', '')).strip()
                    if not period:
                        continue
                    actual = row.get('actual', '')
                    if actual not in ('BIG', 'SMALL'):
                        continue
                    candidate = {
                        'period': period,
                        'prediction': row.get('prediction', ''),
                        'status': row.get('status', 'WIN'),
                        'actual': actual,
                        'number': row.get('number', ''),
                        'confidence': float(row.get('confidence', 100)),
                        'patternUsed': row.get('patternused') or row.get('patternUsed') or '',
                        'source': os.path.basename(path),
                    }
                    existing = by_period.get(period)
                    candidate_is_own = os.path.abspath(path) == os.path.abspath(KAELIS_HISTORY_CSV)
                    existing_has_prediction = bool(existing and existing.get('prediction') in ('BIG', 'SMALL'))
                    candidate_has_prediction = candidate['prediction'] in ('BIG', 'SMALL')
                    if (not existing or candidate_is_own or
                            (candidate_has_prediction and not existing_has_prediction)):
                        by_period[period] = candidate
        except Exception:
            pass
    all_rows = list(by_period.values())
    all_rows.sort(key=lambda r: int(r['period']) if r['period'].isdigit() else 0)
    return all_rows


def _learning_source_summary(rows=None):
    rows = rows if rows is not None else _load_all_history()
    counts = defaultdict(int)
    for row in rows:
        counts[row.get('source') or 'unknown'] += 1
    return {
        'totalRows': len(rows),
        'files': dict(sorted(counts.items(), key=lambda item: item[0])),
        'displayHistoryLimit': KAELIS_HISTORY_LIMIT,
        'fullHistoryUsedForTraining': True,
    }


# ---------------------------------------------------------------------------
# Ensure directories & files exist; bootstrap from daily history if empty
# ---------------------------------------------------------------------------

def _boostrap_memory_from_csvs():
    with _memory_entries_lock:
        if _memory_entries:
            return
        path = KAELIS_HISTORY_CSV
        if os.path.exists(path):
            try:
                with open(path, 'r', newline='', encoding='utf-8') as f:
                    for row in csv.DictReader(f):
                        period = str(row.get('period', '')).strip()
                        if not period:
                            continue
                        actual = row.get('actual', '')
                        if actual not in ('BIG', 'SMALL') and row.get('status') not in ('Pending', 'TRAINING'):
                            continue
                        candidate = {
                            'period': period,
                            'prediction': row.get('prediction', ''),
                            'status': row.get('status', 'Pending'),
                            'confidence': float(row.get('confidence') or 0),
                            'actual': actual if actual in ('BIG', 'SMALL') else '',
                            'number': row.get('number', ''),
                            'patternused': row.get('patternused') or row.get('patternUsed') or '',
                            'timestamp': int(row.get('timestamp') or 0),
                            'skipped': row.get('skipped') in ('True', 'true', True),
                            'skipreason': row.get('skipreason') or row.get('skipReason') or '',
                        }
                        existing = _memory_entries.get(period)
                        if existing and existing.get('prediction') in ('BIG', 'SMALL'):
                            continue
                        _memory_entries[period] = candidate
            except Exception:
                pass


def _ensure_files():
    os.makedirs(os.path.dirname(KAELIS_HISTORY_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(KAELIS_BRAIN_FILE), exist_ok=True)
    if not os.path.exists(KAELIS_HISTORY_CSV):
        with open(KAELIS_HISTORY_CSV, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=HEADER)
            w.writeheader()


def _bootstrap_from_daily():
    _ensure_files()
    _boostrap_memory_from_csvs()
    with _memory_entries_lock:
        if _memory_entries:
            return
    csv_has_data = False
    if os.path.exists(KAELIS_HISTORY_CSV):
        try:
            with open(KAELIS_HISTORY_CSV,'r',newline='') as f:
                for _ in csv.DictReader(f):
                    csv_has_data = True
                    break
        except:
            pass
    if csv_has_data:
        return
    daily = fetch_wingobot_daily_history(retries=1, timeout=5, limit=None)
    if not isinstance(daily, list):
        return
    count = 0
    for item in daily:
        period = str(item.get('period', ''))
        category = item.get('category')
        number = item.get('number')
        if period and category in ('BIG', 'SMALL'):
            _upsert({
                'period': period,
                'prediction': '',
                'status': 'TRAINING',
                'confidence': 0,
                'actual': category,
                'number': number,
                'patternused': 'daily_bootstrap',
                'timestamp': int(time.time()),
                'skipped': False,
                'skipreason': '',
            })
            with _verified_periods_lock:
                _verified_periods.add(period)
            count += 1


# ---------------------------------------------------------------------------
# Fast verification: runs every 8s in dedicated thread, independent of prediction
# ---------------------------------------------------------------------------

_verify_last_run = 0
_verify_thread = None
_verify_thread_lock = threading.Lock()


def _verify_memory_entries():
    """Quick-verify unverified periods against live API. Each period verified ONCE only."""
    global _verify_last_run
    now = time.time()
    if now - _verify_last_run < 3:
        return
    _verify_last_run = now
    current_period = get_current_period_1min()

    # Find periods that are Pending, have no actual, and haven't been verified yet
    with _memory_entries_lock:
        with _verified_periods_lock:
            pending = [e for e in _memory_entries.values()
                       if e.get('period') < current_period
                       and e.get('status') == 'Pending'
                       and not e.get('actual')
                       and e.get('period') not in _verified_periods]
    if not pending:
        return

    game_data = fetch_api_data(retries=1, timeout=3, bypass_cache=True)
    if not isinstance(game_data, list) or not game_data:
        print(f"[KAELIS_VERIFY] fetch_api_data returned {type(game_data).__name__}: {str(game_data)[:100]}")
        return
    by_period = {str(_period_key(item.get('period', ''))): item for item in game_data if item.get('period')}
    learner = _get_learner()
    updated = 0
    with _memory_entries_lock:
        all_actuals = [e.get('actual') for e in _memory_entries.values() if e.get('actual') in ('BIG','SMALL')]
        for entry in _memory_entries.values():
            per = str(entry.get('period', ''))
            # Skip already verified, current period, or not in our pending list
            with _verified_periods_lock:
                if per in _verified_periods:
                    continue
            if entry.get('status') in ('WIN', 'LOSS', 'SKIP'):
                continue
            if entry.get('actual'):
                continue
            if entry.get('period') >= current_period:
                continue
            m = by_period.get(per)
            if not m or m.get('category') not in ('BIG','SMALL'):
                continue
            actual = normalize_side(m.get('category'))
            entry['actual'] = actual
            entry['number'] = str(m.get('number', ''))
            entry['status'] = verified_outcome(entry.get('prediction'), actual, not entry.get('prediction'))
            entry['skipped'] = False
            entry['skipReason'] = ''
            updated += 1
            with _verified_periods_lock:
                _verified_periods.add(per)
            if entry.get('prediction') in ('BIG','SMALL'):
                pattern_name = entry.get('patternUsed') or entry.get('patternused') or 'kaelis_ensemble'
                model_name = _model_from_pattern(pattern_name)
                all_actuals.insert(0, actual)
                regime, streak_len, zigzag_count = _detect_regime(all_actuals)
                prev_actual = all_actuals[1] if len(all_actuals) > 1 else None
                learner.learn_outcome(
                    entry.get('prediction'), actual,
                    [{'model': model_name, 'prediction': entry.get('prediction')}],
                    pattern_name=pattern_name,
                    regime=regime,
                    streak_len=streak_len,
                    zigzag_count=zigzag_count,
                    prev_actual=prev_actual,
                )
    if updated:
        _save_learner()
        _invalidate_snapshot()
        _write_entries([])  # writes all current memory to CSV (lock-safe)


def _verify_loop():
    """Dedicated thread: verify pending entries every ~8s, never blocks prediction."""
    while True:
        try:
            _verify_memory_entries()
        except Exception:
            pass
        time.sleep(8)


def _start_verify_loop():
    global _verify_thread
    with _verify_thread_lock:
        if _verify_thread and _verify_thread.is_alive():
            return
        t = threading.Thread(target=_verify_loop, daemon=True, name='kaelis_verify')
        _verify_thread = t
        t.start()


# Keep old _verify_pending for backward compat (now just delegates to memory verify + _entries)
def _verify_pending(entries):
    _verify_memory_entries()
    return _entries()

def _write_entries(entries):
    # Write to memory (non-destructive: only updates, never clears)
    with _memory_entries_lock:
        for e in entries:
            p = str(e.get('period',''))
            if p:
                _memory_entries[p] = dict(e)
    _invalidate_snapshot()
    # CSV backup from all current memory entries
    try:
        with _memory_entries_lock:
            all_rows = [{k: _csv_value(e.get(k, '')) for k in HEADER} for e in _memory_entries.values()]
        all_rows.sort(key=lambda r: _period_key(r.get('period')), reverse=False)
        os.makedirs(os.path.dirname(KAELIS_HISTORY_CSV), exist_ok=True)
        with open(KAELIS_HISTORY_CSV, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=HEADER)
            w.writeheader()
            w.writerows(all_rows)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Learn from ALL history into learner
# ---------------------------------------------------------------------------

def _detect_regime(actuals):
    if len(actuals) < 5:
        return 'UNKNOWN', 0, 0
    recent = actuals[:12]
    streak = 0
    streak_side = recent[0]
    for a in recent:
        if a == streak_side: streak += 1
        else: break
    alternations = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i-1])
    alt_ratio = alternations / max(len(recent)-1, 1)
    zigzag_count = 0
    for i in range(2, len(recent)):
        if recent[i] != recent[i-1] and recent[i-1] != recent[i-2]:
            zigzag_count += 1
    if streak >= 4:
        return 'STREAK', streak, zigzag_count
    elif alt_ratio >= 0.7:
        return 'ZIGZAG', streak, zigzag_count
    elif alt_ratio >= 0.4:
        return 'CHOPPY', streak, zigzag_count
    return 'MIXED', streak, zigzag_count


def _model_from_pattern(pattern):
    p = pattern.lower()
    if 'xgb' in p or 'xgboost' in p:
        return 'XGBoost'
    elif 'lgbm' in p or 'lightgbm' in p or 'lgb' in p:
        return 'LightGBM'
    elif 'lstm' in p or 'bilstm' in p or 'sequence' in p:
        return 'LSTM'
    return 'XGBoost'


def _learn_from_history(learner):
    all_rows = _load_all_history()
    actuals = [r['actual'] for r in all_rows]
    for i, row in enumerate(all_rows):
        if row.get('status') not in ('WIN','LOSS'):
            continue
        if row.get('prediction') not in ('BIG','SMALL'):
            continue
        # Skip bootstrap entries (artificial data, not real predictions)
        if row.get('patternUsed') == 'daily_bootstrap':
            continue
        pattern_name = row.get('patternUsed', '') or 'unknown'
        model_name = _model_from_pattern(pattern_name)
        prev_actual = actuals[i-1] if i > 0 else None
        regime, streak_len, zigzag_count = _detect_regime(actuals[i:])
        learner.learn_outcome(
            row['prediction'], row['actual'],
            [{'model': model_name, 'prediction': row['prediction']}],
            pattern_name=pattern_name,
            regime=regime,
            streak_len=streak_len,
            zigzag_count=zigzag_count,
            prev_actual=prev_actual,
        )


def _learn_recovery_strategy(learner, settled, losses, recent_actuals=None):
    def opposite(side):
        return 'SMALL' if side == 'BIG' else 'BIG' if side == 'SMALL' else None

    def add_score(scores, name, side, target):
        if side not in ('BIG', 'SMALL') or target not in ('BIG', 'SMALL'):
            return
        s = scores.setdefault(name, {'side': side, 'wins': 0, 'total': 0})
        s['side'] = side
        s['total'] += 1
        if side == target:
            s['wins'] += 1

    chronological = list(reversed(settled))
    scores = {}
    for i, row in enumerate(chronological[:-1]):
        if row.get('status') != 'LOSS':
            continue
        target = chronological[i + 1].get('actual')
        last_pred = row.get('prediction')
        last_actual = row.get('actual')
        context = [r.get('actual') for r in chronological[max(0, i - 11):i + 1]]
        context = list(reversed([x for x in context if x in ('BIG', 'SMALL')]))
        regime, streak_len, zigzag_count = _detect_regime(context)
        add_score(scores, 'follow_last_actual', last_actual, target)
        add_score(scores, 'opposite_failed_prediction', opposite(last_pred), target)
        add_score(scores, 'opposite_last_actual', opposite(last_actual), target)
        add_score(scores, 'repeat_failed_prediction', last_pred, target)
        if regime == 'STREAK' and streak_len >= 3:
            add_score(scores, 'streak_follow', last_actual, target)
        if regime == 'ZIGZAG' and zigzag_count >= 2:
            add_score(scores, 'zigzag_reverse', opposite(last_actual), target)

    last_loss = losses[0] if losses else {}
    last_pred = last_loss.get('prediction')
    last_actual = last_loss.get('actual')
    live_actuals = [x for x in (recent_actuals or []) if x in ('BIG', 'SMALL')]
    if not live_actuals:
        live_actuals = [r.get('actual') for r in reversed(settled[-12:]) if r.get('actual') in ('BIG', 'SMALL')]
    regime, streak_len, zigzag_count = _detect_regime(live_actuals)
    strategy_sides = {
        'follow_last_actual': last_actual,
        'opposite_failed_prediction': opposite(last_pred),
        'opposite_last_actual': opposite(last_actual),
        'repeat_failed_prediction': last_pred,
        'streak_follow': last_actual if regime == 'STREAK' and streak_len >= 3 else None,
        'zigzag_reverse': opposite(last_actual) if regime == 'ZIGZAG' and zigzag_count >= 2 else None,
    }
    big_acc = learner.get_side_accuracy('BIG') or 50
    small_acc = learner.get_side_accuracy('SMALL') or 50
    strategy_sides['best_side_memory'] = 'BIG' if big_acc >= small_acc else 'SMALL'

    best = None
    for name, side in strategy_sides.items():
        if side not in ('BIG', 'SMALL'):
            continue
        stat = scores.get(name, {'wins': 0, 'total': 0})
        acc = (stat['wins'] / stat['total'] * 100) if stat['total'] else 50.0
        weight = stat['total']
        candidate = {
            'strategy': name,
            'prediction': side,
            'accuracy': round(acc, 2),
            'samples': weight,
            'regime': regime,
            'streakLen': streak_len,
            'zigzagCount': zigzag_count,
        }
        if best is None or (acc, weight) > (best['accuracy'], best['samples']):
            best = candidate
    return best or {'strategy': 'follow_last_actual', 'prediction': last_actual, 'accuracy': 50.0, 'samples': 0, 'regime': regime, 'streakLen': streak_len, 'zigzagCount': zigzag_count}


# ---------------------------------------------------------------------------
# Core prediction: XGBoost + LightGBM + LSTM weighted ensemble
# ---------------------------------------------------------------------------

def _model_loss_manager(learner, training_rows, model_predictions, big_votes, small_votes):
    all_rows = _load_all_history()
    settled = [
        r for r in all_rows
        if r.get('status') in ('WIN', 'LOSS')
        and r.get('prediction') in ('BIG', 'SMALL')
        and r.get('actual') in ('BIG', 'SMALL')
    ]
    settled = list(reversed(settled))
    losses = []
    for row in settled:
        if row.get('status') != 'LOSS':
            break
        losses.append(row)
    signal = {
        'active': False,
        'prediction': None,
        'consecutiveLosses': len(losses),
        'reason': '',
        'boost': 0,
        'confidence': 0,
    }
    if len(losses) < 1:
        return signal

    recent_losses = losses[:10]
    last_actual = recent_losses[0].get('actual') if recent_losses else None
    if last_actual not in ('BIG', 'SMALL'):
        return signal

    learned_recovery = _learn_recovery_strategy(learner, settled, losses)
    pred_sides = [r.get('prediction') for r in recent_losses[:6]]
    alternates = all(pred_sides[i] != pred_sides[i+1] for i in range(min(len(pred_sides)-1, 4)))

    if learned_recovery.get('samples', 0) >= 3 and learned_recovery.get('accuracy', 0) >= 52:
        recovery_side = learned_recovery['prediction']
        reason = f"learned_{learned_recovery['strategy']}"
    elif alternates and len(pred_sides) >= 3:
        recovery_side = 'SMALL' if last_actual == 'BIG' else 'BIG'
        reason = 'anti_whipsaw_opposite'
    else:
        recovery_side = last_actual
        reason = 'reactive_follow_last_actual'

    boost = min(0.78, 0.42 + len(losses) * 0.12)
    return {
        **signal,
        'active': True,
        'prediction': recovery_side,
        'reason': reason,
        'boost': round(boost, 4),
        'confidence': round(min(96, 76 + len(losses) * 5), 2),
        'learnedRecovery': learned_recovery,
    }

def _predict(learner, training_rows, current_slice, daily_history):
    global _active_period_prediction

    if not current_slice:
        return None

    ml_result = predict_ml(training_rows, current_slice)

    lstm_result = predict_lstm_bilstm(training_rows, current_slice, daily_history if isinstance(daily_history,list) else [])

    # Extract XGBoost and LightGBM from ml_result
    xgb_pred = None
    lgbm_pred = None
    if ml_result and ml_result.get('modelPredictions'):
        for mp in ml_result['modelPredictions']:
            if mp.get('model') == 'XGBClassifier' and mp.get('prediction') in ('BIG','SMALL'):
                xgb_pred = {'prediction': mp['prediction'], 'confidence': float(mp.get('confidence',50)), 'probability': float(mp.get('bigProbability',50))}
            if mp.get('model') == 'LGBMClassifier' and mp.get('prediction') in ('BIG','SMALL'):
                lgbm_pred = {'prediction': mp['prediction'], 'confidence': float(mp.get('confidence',50)), 'probability': float(mp.get('bigProbability',50))}

    lstm_pred = None
    if lstm_result and lstm_result.get('ready') and lstm_result.get('prediction') in ('BIG','SMALL'):
        lstm_pred = {'prediction': lstm_result['prediction'], 'confidence': float(lstm_result.get('confidence',50)), 'probability': float(lstm_result.get('bigProbability',50))}

    # Weighted voting (BIG vs SMALL, no bias)
    model_predictions = []
    total_weight = 0.0
    big_votes = 0.0
    small_votes = 0.0

    for name, pred, default_w in [('XGBoost', xgb_pred, 0.35), ('LightGBM', lgbm_pred, 0.35), ('LSTM', lstm_pred, 0.30)]:
        if pred:
            w = learner.weights.get(name, default_w)
            m = learner.models.get(name)
            if m and m['total'] >= 5:
                acc_factor = m['recent_acc']/100.0 if m['recent_wins']+m['recent_losses'] >= 5 else m['acc']/100.0
                w *= (0.5 + 0.5 * acc_factor)
            total_weight += w
            if pred['prediction'] == 'BIG':
                big_votes += w * (pred['confidence'] / 100.0)
            else:
                small_votes += w * (pred['confidence'] / 100.0)
            model_predictions.append({'model': name, 'prediction': pred['prediction'], 'confidence': round(pred['confidence'],2), 'probability': pred['probability'], 'weight': round(w,4)})

    if not model_predictions:
        if ml_result and ml_result.get('prediction') in ('BIG','SMALL'):
            conf = float(ml_result.get('confidence', 50))
            model_predictions.append({'model': 'Ensemble', 'prediction': ml_result['prediction'], 'confidence': conf, 'probability': float(ml_result.get('bigProbability', 50)), 'weight': 1.0})
            total_weight = 1.0
            if ml_result['prediction'] == 'BIG':
                big_votes = 1.0 * (conf / 100.0)
            else:
                small_votes = 1.0 * (conf / 100.0)
        else:
            return None

    # Loss recovery handled entirely by _model_loss_manager below
    recovery = learner.get_recovery_adjustment()

    all_actuals = [r.get('actual') for r in reversed(training_rows) if r.get('actual') in ('BIG','SMALL')]
    regime, streak_len, zigzag_count = _detect_regime(all_actuals)

    # Regime-aware adjustment (vote-based)
    regime_acc = learner.get_regime_accuracy(regime)
    if regime_acc and regime_acc > 55:
        boost = (regime_acc - 50) * 0.3
        if regime == 'STREAK' and streak_len >= 3:
            latest = all_actuals[0] if all_actuals else None
            if latest == 'BIG':
                big_votes += boost * total_weight * 0.4
            elif latest == 'SMALL':
                small_votes += boost * total_weight * 0.4
        elif regime == 'ZIGZAG' and zigzag_count >= 3:
            opposite = 'SMALL' if (all_actuals[0] if all_actuals else 'BIG') == 'BIG' else 'BIG'
            if opposite == 'BIG':
                big_votes += boost * total_weight * 0.3
            else:
                small_votes += boost * total_weight * 0.3
    elif regime == 'STREAK' and streak_len >= 3:
        latest = all_actuals[0] if all_actuals else None
        if latest == 'BIG':
            big_votes += 0.15 * total_weight
        elif latest == 'SMALL':
            small_votes += 0.15 * total_weight
    elif regime == 'ZIGZAG' and zigzag_count >= 2:
        opposite = 'SMALL' if (all_actuals[0] if all_actuals else 'BIG') == 'BIG' else 'BIG'
        if opposite == 'BIG':
            big_votes += 0.12 * total_weight
        else:
            small_votes += 0.12 * total_weight

    loss_manager = _model_loss_manager(learner, training_rows, model_predictions, big_votes, small_votes)
    if loss_manager['active'] and loss_manager['prediction'] in ('BIG', 'SMALL'):
        if loss_manager['prediction'] == 'BIG':
            big_votes += loss_manager['boost'] * max(total_weight, 1.0)
        else:
            small_votes += loss_manager['boost'] * max(total_weight, 1.0)
        model_predictions.append({
            'model': 'DeepLossManager',
            'prediction': loss_manager['prediction'],
            'confidence': loss_manager['confidence'],
            'probability': 50,
            'weight': round(loss_manager['boost'], 4),
        })
        total_weight += loss_manager['boost']

    pred = 'BIG' if big_votes >= small_votes else 'SMALL'

    if loss_manager['active'] and loss_manager['consecutiveLosses'] >= 1:
        pred = loss_manager['prediction']

    # REAL confidence = historical win rate of the predicted side
    if learner.total_predictions >= 5:
        real_conf = learner.get_stats()['winRate']
        if loss_manager['active'] and loss_manager['consecutiveLosses'] >= 1:
            real_conf = max(real_conf, loss_manager['confidence'])
    else:
        real_conf = 50.0

    # Side-specific accuracy boost
    side_acc = learner.get_side_accuracy(pred)
    if side_acc and side_acc > 55:
        real_conf += (side_acc - 50) * 0.3

    # Adjust confidence based on model agreement & edge
    agreeing = sum(1 for mp in model_predictions if mp['prediction'] == pred)
    total_models = len(model_predictions)
    agreement_ratio = agreeing / max(total_models, 1)
    total_votes = big_votes + small_votes
    edge = abs(big_votes - small_votes) / max(total_votes, 0.01) * 50

    confidence = real_conf + (edge * 0.3) + (agreement_ratio * 5 - 2.5)
    confidence = max(50.0, min(95.0, confidence))
    big_pct = round((big_votes / max(total_votes, 0.01)) * 100, 2)

    return {
        'prediction': pred,
        'confidence': round(confidence, 2),
        'bigProbability': big_pct,
        'modelPredictions': model_predictions,
        'recovery': recovery,
        'regime': regime,
        'streakLen': streak_len,
        'zigzagCount': zigzag_count,
        'lossManager': loss_manager,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _period_key(p):
    try: return int(str(p))
    except: return 0

def _csv_value(v):
    return '' if v is None else str(v)

def _entries():
    global _history_snapshot
    if _history_snapshot is not None:
        return _history_snapshot
    # Load from CSV if available, merge with memory
    rows = []
    if os.path.exists(KAELIS_HISTORY_CSV):
        try:
            with open(KAELIS_HISTORY_CSV,'r',newline='') as f:
                r = csv.DictReader(f)
                rows = [row for row in r if row.get('period')]
        except:
            pass
    with _memory_entries_lock:
        for p, entry in _memory_entries.items():
            existing = [i for i, row in enumerate(rows) if row.get('period') == p]
            if existing:
                rows[existing[0]] = {k: _csv_value(v) for k, v in entry.items()}
            else:
                rows.append({k: _csv_value(v) for k, v in entry.items()})
    rows.sort(key=lambda r: _period_key(r.get('period')), reverse=True)
    seen = set()
    deduped = []
    for r in rows:
        p = r.get('period')
        if p and p not in seen:
            seen.add(p)
            deduped.append(r)
    _history_snapshot = deduped
    return deduped

def _invalidate_snapshot():
    global _history_snapshot
    _history_snapshot = None

def _public_entry(row):
    skipped = row.get('skipped') == 'True' or row.get('skipped') is True
    status = verified_outcome(row.get('prediction'), row.get('actual'), skipped)
    return {'period':row.get('period'),'prediction':normalize_side(row.get('prediction')) or row.get('prediction'),'status':status or row.get('status','Pending'),'confidence':float(row.get('confidence') or 0),'actual':normalize_side(row.get('actual')) or row.get('actual'),'number':row.get('number'),'patternUsed':row.get('patternUsed') or row.get('patternused') or '','skipped':skipped,'skipReason':row.get('skipReason') or row.get('skipreason') or '','timestamp':int(row.get('timestamp') or 0)}

def _stats(history):
    t = len(history)
    w = sum(1 for h in history if h.get('status')=='WIN')
    l = sum(1 for h in history if h.get('status')=='LOSS')
    s = sum(1 for h in history if h.get('skipped'))
    return {'total':t,'wins':w,'losses':l,'skipped':s,'winRate':round((w/max(w+l,1))*100,2)}

def _is_settled(entry):
    return str((entry or {}).get('status', '')).upper() in ('WIN', 'LOSS')

def _upsert(entry):
    period = str(entry.get('period',''))
    if not period:
        return
    with _memory_entries_lock:
        existing = _memory_entries.get(period)
        if _is_settled(existing):
            return
        merged = dict(existing or {})
        merged.update(dict(entry))
        _memory_entries[period] = merged
    _invalidate_snapshot()
    # Write to CSV as best-effort backup
    try:
        rows = []
        with _memory_entries_lock:
            for p, e in _memory_entries.items():
                rows.append({k: _csv_value(e.get(k, '')) for k in HEADER})
        rows.sort(key=lambda r: _period_key(r.get('period')), reverse=False)
        os.makedirs(os.path.dirname(KAELIS_HISTORY_CSV), exist_ok=True)
        with open(KAELIS_HISTORY_CSV,'w',newline='') as f:
            w = csv.DictWriter(f, fieldnames=HEADER)
            w.writeheader()
            w.writerows(rows)
    except:
        pass

def _make_training_rows(all_history, game_data, daily_history):
    by_p = {}
    for row in all_history:
        p = str(row.get('period',''))
        if p and row.get('actual') in ('BIG','SMALL'):
            by_p[p] = row
    for src in [game_data, daily_history]:
        if isinstance(src, list):
            for item in src:
                p = str(item.get('period',''))
                if p and item.get('category') in ('BIG','SMALL') and p not in by_p:
                    by_p[p] = {'period':p,'prediction':'','status':'WIN','actual':item.get('category'),'number':item.get('number',''),'confidence':100}
    return sorted(by_p.values(), key=lambda r: _period_key(r.get('period',0)))


def _data_fallback_prediction(rows=None, period=None):
    try:
        live = fetch_api_data(retries=1, timeout=5, bypass_cache=True)
        if isinstance(live, list):
            live_actuals = [r.get('category') for r in live if r.get('category') in ('BIG', 'SMALL')]
            if len(live_actuals) >= 3:
                streak_side = live_actuals[0]
                streak_count = 0
                for side in live_actuals:
                    if side == streak_side:
                        streak_count += 1
                    else:
                        break
                if streak_count >= 3:
                    return streak_side
                alternations = sum(1 for i in range(1, min(len(live_actuals), 6)) if live_actuals[i] != live_actuals[i - 1])
                if alternations >= min(len(live_actuals), 6) - 2:
                    return 'SMALL' if live_actuals[0] == 'BIG' else 'BIG'
                return 'SMALL' if live_actuals[0] == 'BIG' else 'BIG'
    except Exception:
        pass
    rows = rows if rows is not None else _load_all_history()
    actuals = [r.get('actual') for r in rows if r.get('actual') in ('BIG', 'SMALL')]
    if not actuals:
        return 'BIG' if _period_key(period or get_current_period_1min()) % 2 else 'SMALL'
    return 'SMALL' if actuals[-1] == 'BIG' else 'BIG'


def _make_payload(current_period, current, learner, entries, result):
    entries_list = list(entries) if isinstance(entries, list) else list(_memory_entries.values())
    entries_list.sort(key=lambda r: _period_key(r.get('period')), reverse=True)
    history = [_public_entry(r) for r in entries_list[:KAELIS_HISTORY_LIMIT]]
    learner_stats = learner.get_stats() if learner else {'totalPredictions': 0, 'totalWins': 0, 'totalLosses': 0, 'winRate': 0}
    all_history = _load_all_history()

    model_accuracies = {}
    if learner:
        for model_name in KAELIS_MODEL_NAMES:
            m = learner.models.get(model_name)
            brain_data = _get_model_accuracy(model_name)
            if m and m['total'] > 0:
                model_accuracies[model_name] = {
                    'accuracy': m['acc'],
                    'recentAccuracy': m['recent_acc'],
                    'totalPredictions': m['total'],
                    'wins': m['wins'],
                    'losses': m['losses'],
                    'consecutiveLosses': m['consecutive_losses'],
                    'consecutiveWins': m['consecutive_wins'],
                    'currentWeight': round(learner.weights.get(model_name, 0), 4),
                    'lastSavedBrain': brain_data is not None,
                }
    if result and isinstance(result, dict):
        for mp in (result.get('modelPredictions') or []):
            mn = mp.get('model', '')
            if mn and mn not in model_accuracies:
                model_accuracies[mn] = {
                    'accuracy': mp.get('confidence', 0) if mp.get('available', True) else 0,
                    'recentAccuracy': mp.get('confidence', 0) if mp.get('available', True) else 0,
                    'totalPredictions': 0 if not mp.get('available', True) else 1,
                    'wins': 0, 'losses': 0,
                    'consecutiveLosses': 0, 'consecutiveWins': 0,
                    'currentWeight': mp.get('weight', 0),
                    'lastSavedBrain': _get_model_accuracy(mn) is not None,
                }

    if not current:
        current = {'period': current_period or '', 'prediction': _data_fallback_prediction(all_history, current_period), 'status': 'Pending', 'confidence': 51.0,
                   'actual': None, 'number': None, 'patternused': 'kaelis_fallback',
                   'timestamp': int(time.time()), 'skipped': False, 'skipreason': ''}
    all_stats = _stats(entries_list)
    return {
        'predictionResult': {
            'period': current.get('period'),
            'prediction': current.get('prediction') or '',
            'status': current.get('status', 'Pending'),
            'skipped': False, 'skipReason': '',
        },
        'predictionDetails': {
            'gameType': 'Wingo 1 Min Kaelis',
            'confidence': round(float(current.get('confidence') or 0), 2),
            'actual': current.get('actual'),
            'number': current.get('number'),
        },
        'modelDecision': {
            'period': current.get('period'),
            'prediction': current.get('prediction') or '',
            'confidence': round(float(current.get('confidence') or 0), 2),
            'modelResult': result,
            'learnerStats': learner_stats,
            'modelAccuracies': model_accuracies,
            'trainedFromRows': len(all_history),
        },
        'learningSources': _learning_source_summary(all_history),
        'history': history[:KAELIS_HISTORY_LIMIT],
        'stats': all_stats,
        'ossStatus': get_oss_data_status(),
    }


def get_kaelis_payload():
    global _payload_cache, _payload_cache_time, _last_period, _active_period_prediction, _last_predict_time

    now = time.time()
    if _payload_cache and now - _payload_cache_time < _PAYLOAD_CACHE_SECONDS:
        return _inject_history(_payload_cache)
    if not _lock.acquire(blocking=False):
        if _payload_cache:
            c = dict(_payload_cache)
            c['stale'] = True
            c['staleReason'] = 'kaelis_refresh_in_progress'
            return _inject_history(c)
        return _skeleton_payload()

    try:
        _ensure_files()
        _bootstrap_from_daily()

        learner = _get_learner()
        _learn_from_history(learner)

        entries = _verify_pending(_entries())
        current_period = get_current_period_1min()
        current = next((e for e in entries if e.get('period') == current_period), None)

        should_predict = current_period != _last_period or not current

        result = _active_period_prediction
        if should_predict:
            try:
                game_data = fetch_api_data(retries=1, timeout=5, bypass_cache=False)
                daily_history = fetch_wingobot_daily_history(retries=1, timeout=5, limit=None)
                current_slice = []
                if isinstance(game_data, list):
                    current_slice = [{'category': r.get('category'), 'number': r.get('number')} for r in game_data if r.get('category') in ('BIG', 'SMALL')]
                if not current_slice and isinstance(daily_history, list):
                    current_slice = [{'category': r.get('category'), 'number': r.get('number')} for r in daily_history[:150] if r.get('category') in ('BIG', 'SMALL')]
                all_history = _load_all_history()
                training_rows = _make_training_rows(all_history, game_data, daily_history)
                sm = get_model_summary()
                if sm.get('totalSamples', 0) == 0 and len(training_rows) >= 10:
                    threading.Thread(target=train_model, args=(training_rows,), kwargs={'force': True}, daemon=True).start()
                pred_result = _predict(learner, training_rows, current_slice, daily_history)
                if pred_result:
                    result = pred_result
                    _active_period_prediction = result
                    _last_period = current_period
                    _last_predict_time = time.time()
            except Exception as e:
                print(f'[KAELIS_PREDICT] error: {e}')

        if not current:
            if result and result.get('prediction') in ('BIG', 'SMALL'):
                current = {'period': current_period, 'prediction': result['prediction'], 'status': 'Pending',
                           'confidence': result.get('confidence', 51), 'actual': None, 'number': None,
                           'patternused': 'kaelis_ensemble', 'timestamp': int(time.time()),
                           'skipped': False, 'skipreason': ''}
            else:
                current = {'period': current_period, 'prediction': _data_fallback_prediction(period=current_period), 'status': 'Pending', 'confidence': 51.0,
                           'actual': None, 'number': None, 'patternused': 'kaelis_default_fallback',
                           'timestamp': int(time.time()), 'skipped': False, 'skipreason': ''}
            _upsert(current)
            _invalidate_snapshot()
            entries = _entries()

        _save_learner()

        _payload_cache = _make_payload(current_period, current, learner, entries, result)
        _payload_cache_time = time.time()
        return _payload_cache
    except Exception as e:
        print(f'[KAELIS_COMPUTE] error: {e}\n{traceback.format_exc()}')
        try:
            _payload_cache = _make_payload(get_current_period_1min(), None, None, [], result)
            _payload_cache_time = time.time()
            return _payload_cache
        except Exception:
            return _skeleton_payload()
    finally:
        try:
            _lock.release()
        except RuntimeError:
            pass


# ---------------------------------------------------------------------------
# Cache & background refresh
# ---------------------------------------------------------------------------

def _convert_native(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {k: _convert_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_native(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _save_cache(payload):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        tmp = KAELIS_CACHE_FILE + '.tmp'
        with open(tmp,'w',encoding='utf-8') as f:
            json.dump({'timestamp':time.time(),'payload':_convert_native(payload)}, f)
        os.replace(tmp, KAELIS_CACHE_FILE)
    except Exception as e:
        print(f'[KAELIS_CACHE] save error: {e}')

def _load_cache():
    if not os.path.exists(KAELIS_CACHE_FILE):
        return None, None
    try:
        with open(KAELIS_CACHE_FILE,'r',encoding='utf-8') as f:
            d = json.load(f)
        return d.get('payload'), time.time()-float(d.get('timestamp',0))
    except: return None, None

def _get_fast_history():
    with _memory_entries_lock:
        rows = [dict(e) for e in _memory_entries.values()]
    if not rows:
        rows = _entries()
    rows.sort(key=lambda r: _period_key(r.get('period')), reverse=True)
    public = [_public_entry(r) for r in rows[:KAELIS_HISTORY_LIMIT]]
    stats = _stats(rows)
    return public, stats


def _inject_history(payload):
    """Replace history in payload with live in-memory data (latest 20)."""
    p = dict(payload)
    cp = get_current_period_1min()
    pr = dict(p.get('predictionResult') or {})
    md = dict(p.get('modelDecision') or {})
    dp = dict(p.get('predictionDetails') or {})
    h, s = _get_fast_history()
    current_entry = next((row for row in h if str(row.get('period')) == str(cp)), None)
    if pr.get('period') != cp:
        pred = current_entry.get('prediction') if current_entry else None
        if pred not in ('BIG', 'SMALL'):
            pred = pr.get('prediction') if pr.get('prediction') in ('BIG', 'SMALL') else md.get('prediction')
        if pred not in ('BIG', 'SMALL'):
            pred = _data_fallback_prediction(period=cp)
        entry_status = current_entry.get('status', 'Pending') if current_entry else 'Pending'
        pr.update({'period': cp, 'prediction': pred, 'status': entry_status, 'skipped': False, 'skipReason': ''})
        md.update({'period': cp, 'prediction': pred})
        p['currentized'] = True
        if not current_entry:
            _upsert({
                'period': cp,
                'prediction': pred,
                'status': 'Pending',
                'confidence': md.get('confidence') or dp.get('confidence') or 51,
                'actual': None,
                'number': None,
                'patternused': 'kaelis_ensemble',
                'timestamp': int(time.time()),
                'skipped': False,
                'skipreason': '',
            })
            h, s = _get_fast_history()
    elif pr.get('prediction') in ('BIG', 'SMALL'):
        if current_entry and current_entry.get('prediction') != pr.get('prediction'):
            pr['prediction'] = current_entry['prediction']
            pr['status'] = current_entry.get('status', 'Pending')
        elif current_entry and current_entry.get('prediction') == pr.get('prediction') and current_entry.get('status') in ('WIN', 'LOSS'):
            pr['status'] = current_entry['status']
        md.update({'period': cp, 'prediction': pr.get('prediction')})
    # Refresh details from live memory
    if current_entry:
        dp['confidence'] = round(float(current_entry.get('confidence') or 0), 2)
        dp['actual'] = current_entry.get('actual')
        dp['number'] = current_entry.get('number')
        md['confidence'] = dp['confidence']
    else:
        dp['confidence'] = round(float(dp.get('confidence') or 0), 2)
        md['confidence'] = dp['confidence']
    # Refresh learner stats from live learner
    try:
        learner = _get_learner()
        if learner:
            md['learnerStats'] = learner.get_stats()
            model_acc = {}
            for model_name in KAELIS_MODEL_NAMES:
                m = learner.models.get(model_name)
                brain_data = _get_model_accuracy(model_name)
                if m and m['total'] > 0:
                    model_acc[model_name] = {
                        'accuracy': m['acc'], 'recentAccuracy': m['recent_acc'],
                        'totalPredictions': m['total'], 'wins': m['wins'], 'losses': m['losses'],
                        'consecutiveLosses': m['consecutive_losses'],
                        'consecutiveWins': m['consecutive_wins'],
                        'currentWeight': round(learner.weights.get(model_name, 0), 4),
                        'lastSavedBrain': brain_data is not None,
                    }
            for mp in (md.get('modelResult') or {}).get('modelPredictions') or []:
                mn = mp.get('model', '')
                if mn and mn not in model_acc:
                    model_acc[mn] = {
                        'accuracy': mp.get('confidence', 0) if mp.get('available', True) else 0,
                        'recentAccuracy': mp.get('confidence', 0) if mp.get('available', True) else 0,
                        'totalPredictions': 0 if not mp.get('available', True) else 1,
                        'wins': 0, 'losses': 0,
                        'consecutiveLosses': 0, 'consecutiveWins': 0,
                        'currentWeight': mp.get('weight', 0),
                        'lastSavedBrain': _get_model_accuracy(mn) is not None,
                    }
            md['modelAccuracies'] = model_acc
            md['trainedFromRows'] = len(_load_all_history())
    except Exception:
        pass
    # Safety: always sync predictionResult from live memory if entry exists
    if current_entry and current_entry.get('prediction') in ('BIG', 'SMALL'):
        pr['prediction'] = current_entry['prediction']
        pr['status'] = current_entry.get('status', 'Pending')
    p['predictionResult'] = pr
    p['modelDecision'] = md
    p['predictionDetails'] = dp
    p['history'] = h[:KAELIS_HISTORY_LIMIT]
    p['stats'] = s
    p.setdefault('learningSources', _learning_source_summary())
    return p


def _skeleton_payload():
    cp = get_current_period_1min()
    h, s = _get_fast_history()
    current_entry = next((row for row in h if str(row.get('period')) == str(cp)), None)
    pred = current_entry.get('prediction') if current_entry else None
    if pred not in ('BIG', 'SMALL'):
        pred = 'BIG' if _period_key(cp) % 2 else 'SMALL'
    status = current_entry.get('status', 'Pending') if current_entry else 'Pending'
    confidence = round(float(current_entry.get('confidence') or 0), 2) if current_entry else 0
    actual = current_entry.get('actual') if current_entry else None
    number = current_entry.get('number') if current_entry else None
    return {
        'predictionResult': {'period': cp, 'prediction': pred, 'status': status, 'skipped': False, 'skipReason': ''},
        'predictionDetails': {'gameType': 'Wingo 1 Min Kaelis', 'confidence': confidence, 'actual': actual, 'number': number},
        'modelDecision': {'period': cp, 'prediction': pred, 'confidence': confidence, 'modelResult': None, 'learnerStats': None, 'modelAccuracies': {}, 'trainedFromRows': 0},
        'learningSources': _learning_source_summary(),
        'history': h[:KAELIS_HISTORY_LIMIT],
        'stats': s,
        'warming': True, 'warmingReason': 'First load — background refresh in progress',
    }


def _predict_for_period():
    """Only called in background thread, never in request path."""
    learner = _get_learner()
    game_data = fetch_api_data(retries=1, timeout=4, bypass_cache=False)
    daily_history = fetch_wingobot_daily_history(retries=1, timeout=6, limit=None)
    current_slice = []
    if isinstance(game_data, list):
        current_slice = [{'category':r.get('category'),'number':r.get('number')} for r in game_data if r.get('category') in ('BIG','SMALL')]
    if not current_slice and isinstance(daily_history, list):
        current_slice = [{'category':r.get('category'),'number':r.get('number')} for r in daily_history[:150] if r.get('category') in ('BIG','SMALL')]
    all_history = _load_all_history()
    training_rows = _make_training_rows(all_history, game_data, daily_history)
    if len(training_rows) >= 15 and get_model_summary().get('totalSamples', 0) == 0:
        threading.Thread(target=train_model, args=(training_rows,), kwargs={'force': True}, daemon=True).start()
    result = _predict(learner, training_rows, current_slice, daily_history)
    return result, current_slice, training_rows, learner


def get_cached_kaelis_payload():
    p, age = _load_cache()

    if not _memory_entries:
        _boostrap_memory_from_csvs()

    if p is None:
        payload = _skeleton_payload()
        _payload_cache = payload
        _payload_cache_time = time.time()
        _bg_refresh()
        return payload

    if age > KAELIS_BG_REFRESH_INTERVAL:
        _bg_refresh()

    result = _inject_history(p)

    if age > KAELIS_CACHE_STALE_SECONDS:
        result['stale'] = True
        result['staleReason'] = f'cache_age_{int(age)}s_bg_refresh_running'

    return result

_bg_thread = None
_bg_lock = threading.Lock()

def _bg_refresh():
    global _bg_thread
    with _bg_lock:
        if _bg_thread and _bg_thread.is_alive():
            return
        t = threading.Thread(target=_bg_worker_with_timeout, daemon=True, name='kaelis_bg')
        _bg_thread = t; t.start()

def _bg_worker():
    try:
        p = get_kaelis_payload()
    except Exception as e:
        print(f'[KAELIS_BG] {e}\n{traceback.format_exc()}')

def _bg_worker_with_timeout():
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_bg_worker)
            fut.result(timeout=25)
    except concurrent.futures.TimeoutError:
        print('[KAELIS_BG] worker timed out')

def start_kaelis_bg_refresh_loop():
    _start_verify_loop()
    def _auto_train_loop():
        while True:
            try:
                from ml import train_model, get_model_summary
                training_rows = _load_all_history()
                if training_rows and len(training_rows) >= 10:
                    train_model(training_rows, force=True)
            except Exception:
                pass
            time.sleep(10)
    t_train = threading.Thread(target=_auto_train_loop, daemon=True, name='kaelis_autotrain')
    t_train.start()
    def _loop():
        while True:
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    fut = pool.submit(_bg_worker)
                    fut.result(timeout=25)
            except concurrent.futures.TimeoutError:
                print('[KAELIS_BG] compute timed out, skipping this cycle')
            except Exception as e:
                print(f'[KAELIS_BG] loop error: {e}')
            time.sleep(KAELIS_BG_REFRESH_INTERVAL)
    t = threading.Thread(target=_loop, daemon=True, name='kaelis_bg_loop')
    t.start()
    print('[KAELIS_BG] Background refresh + verify loop started')
