import csv
import json
import os
import threading
import time
import traceback
import pickle
import math

from collections import Counter, defaultdict
import numpy as np
from helpers import fetch_api_data, fetch_wingobot_daily_history, get_current_period_1min
from ml import predict_ml, predict_lstm_bilstm, train_model, get_model_summary


BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
KAELIS_HISTORY_CSV = os.path.join(DATA_DIR, 'model', 'kaelis_prediction_history.csv')
KAELIS_HISTORY_LIMIT = 20
KAELIS_CACHE_FILE = os.path.join(DATA_DIR, 'kaelis_cache.json')
KAELIS_CACHE_STALE_SECONDS = 120
KAELIS_BG_REFRESH_INTERVAL = 30
KAELIS_BRAIN_FILE = os.path.join(DATA_DIR, 'model_brain', 'kaelis_brain.pkl')
KAELIS_MODEL_FILE = os.path.join(DATA_DIR, 'model_brain', 'kaelis_model.pkl')

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
        if self.consecutive_losses >= 2 and not self.loss_recovery_mode:
            self.loss_recovery_mode = True
            self.recovery_side = actual

    def get_recovery_adjustment(self):
        if self.loss_recovery_mode and self.recovery_side:
            return {
                'active': True,
                'side': self.recovery_side,
                'consecutive_losses': self.consecutive_losses,
                'boost': min(self.consecutive_losses * 8, 30),
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
    all_rows = []
    seen = set()
    for path in _discover_all_csvs():
        if not os.path.exists(path):
            continue
        try:
            with open(path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    period = str(row.get('period', '')).strip()
                    if not period or period in seen:
                        continue
                    actual = row.get('actual', '')
                    if actual not in ('BIG', 'SMALL'):
                        continue
                    seen.add(period)
                    all_rows.append({
                        'period': period,
                        'prediction': row.get('prediction', ''),
                        'status': row.get('status', 'WIN'),
                        'actual': actual,
                        'number': row.get('number', ''),
                        'confidence': float(row.get('confidence', 100)),
                        'patternUsed': row.get('patternused') or row.get('patternUsed') or '',
                        'source': os.path.basename(path),
                    })
        except Exception:
            pass
    all_rows.sort(key=lambda r: int(r['period']) if r['period'].isdigit() else 0)
    return all_rows


# ---------------------------------------------------------------------------
# Ensure directories & files exist; bootstrap from daily history if empty
# ---------------------------------------------------------------------------

def _ensure_files():
    os.makedirs(os.path.dirname(KAELIS_HISTORY_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(KAELIS_BRAIN_FILE), exist_ok=True)
    if not os.path.exists(KAELIS_HISTORY_CSV):
        with open(KAELIS_HISTORY_CSV, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=HEADER)
            w.writeheader()


def _bootstrap_from_daily():
    _ensure_files()
    with _memory_entries_lock:
        if _memory_entries:
            return
    # Also check if CSV already has data (first run after restart)
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
    daily = fetch_wingobot_daily_history(retries=2, timeout=8, limit=None)
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
                'prediction': category,
                'status': 'WIN',
                'confidence': 100,
                'actual': category,
                'number': number,
                'patternUsed': 'daily_bootstrap',
                'timestamp': int(time.time()),
                'skipped': False,
                'skipReason': '',
            })
            count += 1


# ---------------------------------------------------------------------------
# Verify pending predictions against live API
# ---------------------------------------------------------------------------

def _verify_pending(entries):
    current_period = get_current_period_1min()
    learner = _get_learner()
    pending = [e for e in entries if e.get('period') < current_period and e.get('status') in ('Pending','SKIP') and not e.get('actual')]
    if not pending:
        return entries
    game_data = fetch_api_data(retries=1, timeout=4, bypass_cache=True)
    if not isinstance(game_data, list):
        return entries
    by_period = {str(item.get('period','')): item for item in game_data if item.get('period')}
    updated = False
    all_actuals = [e.get('actual') for e in entries if e.get('actual') in ('BIG','SMALL')]
    for entry in pending:
        per = str(entry.get('period',''))
        m = by_period.get(per)
        if m and m.get('category') in ('BIG','SMALL'):
            actual = m['category']
            entry['actual'] = actual
            entry['number'] = str(m.get('number',''))
            # Always set WIN/LOSS when we have actual — NEVER SKIP after verification
            if entry.get('prediction') in ('BIG','SMALL'):
                entry['status'] = 'WIN' if entry.get('prediction') == actual else 'LOSS'
            else:
                entry['status'] = 'WIN'
            entry['skipped'] = False
            entry['skipReason'] = ''
            updated = True
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
    _save_learner()
    if updated:
        _write_entries(entries)
        _invalidate_snapshot()
    return _entries()

def _write_entries(entries):
    with _memory_entries_lock:
        _memory_entries.clear()
        for e in entries:
            p = str(e.get('period',''))
            if p:
                _memory_entries[p] = dict(e)
    _invalidate_snapshot()
    try:
        rows = [{k: _csv_value(e.get(k, '')) for k in HEADER} for e in entries]
        rows.sort(key=lambda r: _period_key(r.get('period')), reverse=False)
        os.makedirs(os.path.dirname(KAELIS_HISTORY_CSV), exist_ok=True)
        with open(KAELIS_HISTORY_CSV, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=HEADER)
            w.writeheader()
            w.writerows(rows)
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


# ---------------------------------------------------------------------------
# Core prediction: XGBoost + LightGBM + LSTM weighted ensemble
# ---------------------------------------------------------------------------

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

    # Weighted voting
    model_predictions = []
    total_weight = 0
    big_votes = 0
    confidences = []

    for name, pred, default_w in [('XGBoost', xgb_pred, 0.35), ('LightGBM', lgbm_pred, 0.35), ('LSTM', lstm_pred, 0.30)]:
        if pred:
            w = learner.weights.get(name, default_w)
            # Adjust weight by recent accuracy
            m = learner.models.get(name)
            if m and m['total'] >= 5:
                acc_factor = m['recent_acc']/100.0 if m['recent_wins']+m['recent_losses'] >= 5 else m['acc']/100.0
                w *= (0.5 + 0.5 * acc_factor)
            total_weight += w
            p = pred['probability']/100.0 if pred['prediction'] == 'BIG' else 1.0 - pred['probability']/100.0
            big_votes += w * p
            confidences.append(pred['confidence'])
            model_predictions.append({'model': name, 'prediction': pred['prediction'], 'confidence': round(pred['confidence'],2), 'probability': pred['probability'], 'weight': round(w,4)})

    if not model_predictions:
        # Fallback: use ml_result's own prediction if available
        if ml_result and ml_result.get('prediction') in ('BIG','SMALL'):
            conf = float(ml_result.get('confidence', 50))
            bp = float(ml_result.get('bigProbability', 50))
            return {
                'prediction': ml_result['prediction'],
                'confidence': min(92, max(50, conf)),
                'bigProbability': bp,
                'modelPredictions': [{'model': 'Ensemble', 'prediction': ml_result['prediction'], 'confidence': conf, 'probability': bp, 'weight': 1.0}],
                'recovery': learner.get_recovery_adjustment(),
            }
        return None

    # Loss recovery adjustment
    recovery = learner.get_recovery_adjustment()
    if recovery['active'] and recovery['side']:
        recovery_boost = recovery['boost']/100.0
        if recovery['side'] == 'BIG':
            big_votes += recovery_boost * total_weight * 0.5
        else:
            big_votes -= recovery_boost * total_weight * 0.5
        model_predictions.append({'model':'LossRecovery','prediction':recovery['side'],'confidence':round(55+recovery['boost'],2),'probability':50+recovery['boost'],'weight':round(recovery['boost']/100,4)})
        total_weight += recovery['boost']/100 * 0.5

    if total_weight == 0:
        return None

    final_bp = (big_votes/total_weight)*100
    pred = 'BIG' if final_bp >= 50 else 'SMALL'

    # Detect market regime from training_rows actuals
    all_actuals = [r.get('actual') for r in training_rows if r.get('actual') in ('BIG','SMALL')]
    regime, streak_len, zigzag_count = _detect_regime(all_actuals)

    # Regime-aware adjustment
    regime_acc = learner.get_regime_accuracy(regime)
    if regime_acc and regime_acc > 55:
        regime_boost = (regime_acc - 50) * 0.3
        if regime == 'STREAK' and streak_len >= 3:
            latest_side = all_actuals[0] if all_actuals else None
            if latest_side:
                if latest_side == 'BIG':
                    big_votes += regime_boost * total_weight * 0.4
                else:
                    big_votes -= regime_boost * total_weight * 0.4
        elif regime == 'ZIGZAG' and zigzag_count >= 3:
            opposite = 'SMALL' if (all_actuals[0] if all_actuals else 'BIG') == 'BIG' else 'BIG'
            if opposite == 'BIG':
                big_votes += regime_boost * total_weight * 0.3
            else:
                big_votes -= regime_boost * total_weight * 0.3
    elif regime == 'STREAK' and streak_len >= 3:
        latest_side = all_actuals[0] if all_actuals else None
        if latest_side:
            if latest_side == 'BIG':
                big_votes += 0.15 * total_weight
            else:
                big_votes -= 0.15 * total_weight

    final_bp = (big_votes/total_weight)*100
    pred = 'BIG' if final_bp >= 50 else 'SMALL'

    # REAL confidence = historical win rate of the predicted side
    if learner.total_predictions >= 5:
        real_conf = learner.get_stats()['winRate']
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
    edge = abs(final_bp - 50)

    confidence = real_conf + (edge * 0.3) + (agreement_ratio * 5 - 2.5)
    confidence = max(50.0, min(95.0, confidence))

    return {
        'prediction': pred,
        'confidence': round(confidence, 2),
        'bigProbability': round(final_bp, 2),
        'modelPredictions': model_predictions,
        'recovery': recovery,
        'regime': regime,
        'streakLen': streak_len,
        'zigzagCount': zigzag_count,
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
    _history_snapshot = rows
    return rows

def _invalidate_snapshot():
    global _history_snapshot
    _history_snapshot = None

def _public_entry(row):
    return {'period':row.get('period'),'prediction':row.get('prediction'),'status':row.get('status','Pending'),'confidence':float(row.get('confidence') or 0),'actual':row.get('actual'),'number':row.get('number'),'patternUsed':row.get('patternUsed') or row.get('patternused') or '','skipped':row.get('skipped')=='True' or row.get('skipped') is True,'skipReason':row.get('skipReason') or row.get('skipreason') or '','timestamp':int(row.get('timestamp') or 0)}

def _stats(history):
    t = len(history)
    w = sum(1 for h in history if h.get('status')=='WIN')
    l = sum(1 for h in history if h.get('status')=='LOSS')
    s = sum(1 for h in history if h.get('skipped'))
    return {'total':t,'wins':w,'losses':l,'skipped':s,'winRate':round((w/max(w+l,1))*100,2)}

def _upsert(entry):
    period = str(entry.get('period',''))
    if not period:
        return
    with _memory_entries_lock:
        _memory_entries[period] = dict(entry)
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


# ---------------------------------------------------------------------------
# Main prediction cycle (self-contained)
# ---------------------------------------------------------------------------

def get_kaelis_payload():
    global _payload_cache, _payload_cache_time, _last_period, _active_period_prediction, _last_predict_time

    now = time.time()
    if _payload_cache and now-_payload_cache_time<_PAYLOAD_CACHE_SECONDS:
        return _payload_cache
    if not _lock.acquire(blocking=False):
        if _payload_cache:
            c = dict(_payload_cache)
            c['stale']=True; c['staleReason']='kaelis_refresh_in_progress'
            return c
    else:
        _lock.release()

    with _lock:
        _ensure_files()
        _bootstrap_from_daily()

        learner = _get_learner()
        _learn_from_history(learner)

        entries = _verify_pending(_entries())
        current_period = get_current_period_1min()
        current = next((e for e in entries if e.get('period')==current_period), None)

        should_predict = current_period != _last_period or not current

        if should_predict:
            # Use cached API data (bypass_cache=False), shorter timeouts
            game_data = fetch_api_data(retries=1, timeout=4, bypass_cache=False)
            daily_history = fetch_wingobot_daily_history(retries=1, timeout=6, limit=None)

            current_slice = []
            if isinstance(game_data, list):
                current_slice = [{'category':r.get('category'),'number':r.get('number')} for r in game_data if r.get('category') in ('BIG','SMALL')]
            if not current_slice and isinstance(daily_history, list):
                current_slice = [{'category':r.get('category'),'number':r.get('number')} for r in daily_history[:150] if r.get('category') in ('BIG','SMALL')]

            all_history = _load_all_history()
            training_rows = _make_training_rows(all_history, game_data, daily_history)

            # Only train if not yet trained
            summary = get_model_summary()
            if summary.get('totalSamples', 0) == 0 and len(training_rows) >= 15:
                train_model(training_rows, force=True)

            result = _predict(learner, training_rows, current_slice, daily_history)

            if result:
                _active_period_prediction = result
                _last_period = current_period
                _last_predict_time = time.time()

        result = _active_period_prediction

        if not current:
            if result and result.get('prediction') in ('BIG','SMALL'):
                current = {'period':current_period,'prediction':result['prediction'],'status':'Pending','confidence':result['confidence'],'actual':None,'number':None,'patternUsed':'kaelis_ensemble','timestamp':int(time.time()),'skipped':False,'skipReason':''}
            else:
                current = {'period':current_period,'prediction':'BIG','status':'Pending','confidence':51.0,'actual':None,'number':None,'patternUsed':'kaelis_default_fallback','timestamp':int(time.time()),'skipped':False,'skipReason':''}
            _upsert(current)
            _invalidate_snapshot()
            entries = _entries()

        _save_learner()

        entries.sort(key=lambda r: _period_key(r.get('period')), reverse=True)
        history = [_public_entry(r) for r in entries[:KAELIS_HISTORY_LIMIT]]
        learner_stats = learner.get_stats()

        payload = {
            'predictionResult': {
                'period': current.get('period'),
                'prediction': current.get('prediction') or '',
                'status': current.get('status','Pending'),
                'skipped': False,
                'skipReason': '',
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
                'trainedFromRows': len(_load_all_history()),
            },
            'history': history,
            'stats': _stats(history),
        }

        _payload_cache = payload
        _payload_cache_time = time.time()
        return payload


# ---------------------------------------------------------------------------
# Cache & background refresh
# ---------------------------------------------------------------------------

def _save_cache(payload):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        tmp = KAELIS_CACHE_FILE + '.tmp'
        with open(tmp,'w',encoding='utf-8') as f:
            json.dump({'timestamp':time.time(),'payload':payload}, f)
        os.replace(tmp, KAELIS_CACHE_FILE)
    except: pass

def _load_cache():
    if not os.path.exists(KAELIS_CACHE_FILE):
        return None, None
    try:
        with open(KAELIS_CACHE_FILE,'r',encoding='utf-8') as f:
            d = json.load(f)
        return d.get('payload'), time.time()-float(d.get('timestamp',0))
    except: return None, None

def _skeleton_payload():
    cp = get_current_period_1min()
    return {
        'predictionResult': {'period': cp, 'prediction': 'BIG', 'status': 'Pending', 'skipped': False, 'skipReason': ''},
        'predictionDetails': {'gameType': 'Wingo 1 Min Kaelis', 'confidence': 0, 'actual': None, 'number': None},
        'modelDecision': {'period': cp, 'prediction': 'BIG', 'confidence': 0, 'modelResult': None, 'learnerStats': None, 'trainedFromRows': 0},
        'history': [], 'stats': {'total': 0, 'wins': 0, 'losses': 0, 'skipped': 0, 'winRate': 0},
        'warming': True, 'warmingReason': 'First load — background refresh in progress',
    }


def _get_fast_history():
    """Return history from in-memory store instantly (no CSV)."""
    with _memory_entries_lock:
        rows = [dict(e) for e in _memory_entries.values()]
    rows.sort(key=lambda r: _period_key(r.get('period')), reverse=True)
    public = [_public_entry(r) for r in rows[:KAELIS_HISTORY_LIMIT]]
    stats = _stats(public)
    return public, stats


def _fast_stats_only():
    _, stats = _get_fast_history()
    return stats


def _predict_for_period():
    """Lightweight prediction: only called in background thread, never in request path."""
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
    train_model(training_rows, force=len(training_rows)>=15 and get_model_summary().get('totalSamples',0)==0)
    result = _predict(learner, training_rows, current_slice, daily_history)
    return result, current_slice, training_rows, learner


def get_cached_kaelis_payload():
    p, age = _load_cache()
    now = time.time()

    # No cache at all → return skeleton immediately, trigger bg refresh
    if p is None:
        _bg_refresh()
        return _skeleton_payload()

    # Stale → trigger bg refresh (non-blocking), return stale data
    if age > KAELIS_BG_REFRESH_INTERVAL:
        _bg_refresh()

    # Serve cached if fresh enough
    if age <= KAELIS_CACHE_STALE_SECONDS:
        return p

    sp = dict(p)
    # Inject live history from memory (always up-to-date)
    history, stats = _get_fast_history()
    sp['history'] = history
    sp['stats'] = stats
    sp['stale'] = True
    sp['staleReason'] = f'cache_age_{int(age)}s_bg_refresh_running'
    return sp

_bg_thread = None
_bg_lock = threading.Lock()

def _bg_refresh():
    global _bg_thread
    with _bg_lock:
        if _bg_thread and _bg_thread.is_alive():
            return
        t = threading.Thread(target=_bg_worker, daemon=True, name='kaelis_bg')
        _bg_thread = t; t.start()

def _bg_worker():
    try:
        p = get_kaelis_payload(); _save_cache(p)
    except Exception as e:
        print(f'[KAELIS_BG] {e}\n{traceback.format_exc()}')

def start_kaelis_bg_refresh_loop():
    def _loop():
        while True:
            try:
                p = get_kaelis_payload(); _save_cache(p)
            except Exception as e:
                print(f'[KAELIS_BG] loop error: {e}')
            time.sleep(KAELIS_BG_REFRESH_INTERVAL)
    t = threading.Thread(target=_loop, daemon=True, name='kaelis_bg_loop')
    t.start()
    print('[KAELIS_BG] Background refresh loop started (every 30s)')
