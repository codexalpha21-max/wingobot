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
_history_snapshot = []
_payload_cache = None
_payload_cache_time = 0
_PAYLOAD_CACHE_SECONDS = 12
_bg_refresh_thread = None
_bg_refresh_lock = threading.Lock()
_last_predict_time = 0
_last_period = None
_active_period_prediction = None

# ---------------------------------------------------------------------------
# Kaelis Learner – smart self-learning engine
# ---------------------------------------------------------------------------

class KaelisLearner:
    def __init__(self):
        # Per-model performance
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
        self.pattern_memory = defaultdict(lambda: {'wins':0,'losses':0})
        self.regime_memory = defaultdict(lambda: {'wins':0,'losses':0})
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

    def learn_outcome(self, prediction, actual, model_details):
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

        # Learn per-model
        for d in model_details:
            mn = d.get('model','')
            mp = d.get('prediction')
            if mp in ('BIG','SMALL') and mp == prediction:
                self.learn(mn, mp, actual, won)

        # Update weights dynamically
        self._adjust_weights()

        # Loss recovery logic
        self._check_loss_recovery(prediction, actual)

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
            # Loss streak detected – enter recovery mode
            self.loss_recovery_mode = True
            # Follow the actual (market) side
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

    def get_stats(self):
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

def _load_all_history():
    all_rows = []
    seen = set()
    sources = [
        KAELIS_HISTORY_CSV,
        os.path.join(DATA_DIR, 'model', 'model_prediction_history.csv'),
        os.path.join(DATA_DIR, 'predict', 'prediction_history.csv'),
        os.path.join(DATA_DIR, 'predict', 'prediction_history.csv.backup'),
        os.path.join(DATA_DIR, 'free', 'free_prediction_history.csv'),
        os.path.join(DATA_DIR, 'predict', 'predictions.csv'),
    ]
    for path in sources:
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
                    })
        except Exception:
            pass
    all_rows.sort(key=lambda r: int(r['period']) if r['period'].isdigit() else 0)
    return all_rows


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
    for entry in pending:
        per = str(entry.get('period',''))
        m = by_period.get(per)
        if m and m.get('category') in ('BIG','SMALL'):
            actual = m['category']
            entry['actual'] = actual
            entry['number'] = str(m.get('number',''))
            is_skip = entry.get('skipped') or entry.get('prediction') == 'SKIP'
            entry['status'] = 'SKIP' if is_skip else ('WIN' if entry.get('prediction') == actual else 'LOSS')
            updated = True
            if not is_skip and entry.get('prediction') in ('BIG','SMALL'):
                pattern = (entry.get('patternUsed') or '').lower()
                model = 'XGBoost'
                if 'lgbm' in pattern or 'lightgbm' in pattern:
                    model = 'LightGBM'
                elif 'lstm' in pattern:
                    model = 'LSTM'
                learner.learn_outcome(entry.get('prediction'), actual, [{'model': model, 'prediction': entry.get('prediction')}])
    _save_learner()
    if updated:
        _write_entries(entries)
        _invalidate_snapshot()
    return _entries()

def _write_entries(entries):
    os.makedirs(os.path.dirname(KAELIS_HISTORY_CSV), exist_ok=True)
    try:
        with open(KAELIS_HISTORY_CSV, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=HEADER)
            w.writeheader()
            for row in entries:
                w.writerow({k: _csv_value(v) for k,v in row.items()})
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Learn from ALL history into learner
# ---------------------------------------------------------------------------

def _learn_from_history(learner):
    sources = [
        KAELIS_HISTORY_CSV,
        os.path.join(DATA_DIR, 'model', 'model_prediction_history.csv'),
        os.path.join(DATA_DIR, 'predict', 'prediction_history.csv'),
        os.path.join(DATA_DIR, 'free', 'free_prediction_history.csv'),
    ]
    for path in sources:
        if not os.path.exists(path):
            continue
        try:
            with open(path, 'r', newline='', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    if row.get('status') in ('WIN','LOSS') and row.get('prediction') in ('BIG','SMALL') and row.get('actual') in ('BIG','SMALL'):
                        pattern = (row.get('patternUsed') or row.get('patternused') or '').lower()
                        if 'xgb' in pattern or 'xgboost' in pattern:
                            model = 'XGBoost'
                        elif 'lgbm' in pattern or 'lightgbm' in pattern or 'lgb' in pattern:
                            model = 'LightGBM'
                        elif 'lstm' in pattern or 'bilstm' in pattern or 'sequence' in pattern:
                            model = 'LSTM'
                        else:
                            model = 'XGBoost'
                        learner.learn_outcome(row.get('prediction'), row.get('actual'), [{'model': model, 'prediction': row.get('prediction')}])
        except Exception:
            pass


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

    # REAL confidence = historical win rate of the predicted side
    if learner.total_predictions >= 5:
        real_conf = learner.get_stats()['winRate']
    else:
        real_conf = 50.0

    # Adjust confidence based on model agreement & edge
    agreeing = sum(1 for mp in model_predictions if mp['prediction'] == pred)
    total_models = len(model_predictions)
    agreement_ratio = agreeing / max(total_models, 1)
    edge = abs(final_bp - 50)

    # Confidence = base win rate + edge/agreement boost (capped)
    confidence = real_conf + (edge * 0.3) + (agreement_ratio * 5 - 2.5)
    confidence = max(50.0, min(92.0, confidence))

    return {
        'prediction': pred,
        'confidence': round(confidence, 2),
        'bigProbability': round(final_bp, 2),
        'modelPredictions': model_predictions,
        'recovery': recovery,
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
    if _history_snapshot: return _history_snapshot
    _history_snapshot = _read_rows(KAELIS_HISTORY_CSV)
    return _history_snapshot

def _invalidate_snapshot():
    global _history_snapshot
    _history_snapshot = []

def _read_rows(path):
    if not os.path.exists(path): return []
    try:
        with open(path,'r',newline='') as f:
            r = csv.DictReader(f)
            return [row for row in r if row.get('period')]
    except: return []

def _public_entry(row):
    return {'period':row.get('period'),'prediction':row.get('prediction'),'status':row.get('status','Pending'),'confidence':float(row.get('confidence') or 0),'actual':row.get('actual'),'number':row.get('number'),'patternUsed':row.get('patternUsed') or row.get('patternused') or '','skipped':row.get('skipped')=='True' or row.get('skipped') is True,'skipReason':row.get('skipReason') or row.get('skipreason') or '','timestamp':int(row.get('timestamp') or 0)}

def _stats(history):
    t = len(history)
    w = sum(1 for h in history if h.get('status')=='WIN')
    l = sum(1 for h in history if h.get('status')=='LOSS')
    s = sum(1 for h in history if h.get('skipped'))
    return {'total':t,'wins':w,'losses':l,'skipped':s,'winRate':round((w/max(w+l,1))*100,2)}

def _upsert(entry):
    rows = _read_rows(KAELIS_HISTORY_CSV)
    period = str(entry.get('period',''))
    existing = [row for row in rows if row.get('period')!=period]
    existing.append({k:_csv_value(v) for k,v in entry.items()})
    existing.sort(key=lambda r: _period_key(r.get('period')), reverse=True)
    if len(existing)>500: existing=existing[:500]
    os.makedirs(os.path.dirname(KAELIS_HISTORY_CSV), exist_ok=True)
    try:
        with open(KAELIS_HISTORY_CSV,'w',newline='') as f:
            w = csv.DictWriter(f, fieldnames=HEADER)
            w.writeheader(); w.writerows(existing)
    except: pass

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
        learner = _get_learner()
        _learn_from_history(learner)

        entries = _verify_pending(_entries())
        current_period = get_current_period_1min()
        current = next((e for e in entries if e.get('period')==current_period), None)

        # Only predict once per period
        should_predict = current_period != _last_period or not current

        if should_predict:
            game_data = fetch_api_data(retries=2, timeout=5)
            daily_history = fetch_wingobot_daily_history(retries=1, timeout=8, limit=None)

            current_slice = []
            if isinstance(game_data, list):
                current_slice = [{'category':r.get('category'),'number':r.get('number')} for r in game_data if r.get('category') in ('BIG','SMALL')]
            if not current_slice and isinstance(daily_history, list):
                current_slice = [{'category':r.get('category'),'number':r.get('number')} for r in daily_history[:150] if r.get('category') in ('BIG','SMALL')]

            all_history = _load_all_history()
            training_rows = _make_training_rows(all_history, game_data, daily_history)

            # Train models if needed
            train_model(training_rows, force=len(training_rows)>=15 and get_model_summary().get('totalSamples',0)==0)

            result = _predict(learner, training_rows, current_slice, daily_history)

            if result:
                _active_period_prediction = result
                _last_period = current_period
                _last_predict_time = time.time()

        result = _active_period_prediction

        # Decision: confidence >= 68
        should_skip = True
        skip_reason = ''
        if result and result['prediction'] and result['confidence'] >= 68:
            should_skip = False
        elif not result:
            skip_reason = 'Models not ready yet'
        else:
            skip_reason = f'Low confidence ({result["confidence"]}% < 68%)'

        if not current:
            if not should_skip:
                current = {'period':current_period,'prediction':result['prediction'],'status':'Pending','confidence':result['confidence'],'actual':None,'number':None,'patternUsed':'kaelis_ensemble','timestamp':int(time.time()),'skipped':False,'skipReason':''}
            else:
                current = {'period':current_period,'prediction':'SKIP','status':'SKIP','confidence':0,'actual':None,'number':None,'patternUsed':'kaelis_ensemble','timestamp':int(time.time()),'skipped':True,'skipReason':skip_reason}
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
                'skipped': current.get('skipped',False),
                'skipReason': current.get('skipReason','') or '',
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

def get_cached_kaelis_payload():
    p, age = _load_cache()
    if age is None or age > KAELIS_BG_REFRESH_INTERVAL:
        _bg_refresh()
    if p is None:
        live = get_kaelis_payload()
        _save_cache(live)
        return live
    if age <= KAELIS_CACHE_STALE_SECONDS:
        return p
    sp = dict(p)
    sp['stale']=True; sp['staleReason']=f'cache_age_{int(age)}s'
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
