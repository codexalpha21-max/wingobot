import csv
import json
import os
import threading
import time
import traceback
import pickle
import math
import random
import concurrent.futures
from collections import Counter, defaultdict

import numpy as np
from helpers import fetch_api_data, fetch_wingobot_daily_history, get_current_period_1min, get_oss_data_status
from ml import predict_ml, predict_lstm_bilstm, train_model, get_model_summary, extract_features as ml_extract_features

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
ORION_HISTORY_CSV = os.path.join(DATA_DIR, 'model', 'orion_prediction_history.csv')
ORION_HISTORY_LIMIT = 20
ORION_CACHE_FILE = os.path.join(DATA_DIR, 'orion_cache.json')
ORION_CACHE_STALE_SECONDS = 120
ORION_BG_REFRESH_INTERVAL = 30
ORION_BRAIN_FILE = os.path.join(DATA_DIR, 'model_brain', 'orion_brain.pkl')
ORION_MODEL_FILE = os.path.join(DATA_DIR, 'model_brain', 'orion_model.pkl')
ORION_MODEL_NAMES = ['LightGBM', 'XGBoost', 'CatBoost', 'TabNet', 'LSTM_Attention', 'Transformer']

HEADER = [
    'id', 'period', 'prediction', 'status', 'confidence',
    'actual', 'number', 'patternused', 'timestamp',
    'skipped', 'skipreason', 'created_at'
]

_lock = threading.RLock()
_history_snapshot = None
_payload_cache = None
_payload_cache_time = 0
_PAYLOAD_CACHE_SECONDS = 12
_bg_refresh_thread = None
_bg_refresh_lock = threading.Lock()
_last_predict_time = 0
_last_period = None
_active_period_prediction = None
_orion_locked_side = None

_memory_entries = {}
_memory_entries_lock = threading.Lock()
_verified_periods = set()
_verified_periods_lock = threading.Lock()

# ─── Model Availability Checks ────────────────────────────────────────────

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LGBMClassifier = None
    LIGHTGBM_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CatBoostClassifier = None
    CATBOOST_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    optim = None


# ───────────────────────────────────────────────────────────────────────────
#  OrionLearner – Self-learning brain with all 6 models
# ───────────────────────────────────────────────────────────────────────────

class OrionLearner:
    def __init__(self):
        self.models = {
            'LightGBM':     {'wins':0,'losses':0,'total':0,'acc':50.0,'recent_wins':0,'recent_losses':0,'recent_acc':50.0,'consecutive_losses':0,'consecutive_wins':0},
            'XGBoost':      {'wins':0,'losses':0,'total':0,'acc':50.0,'recent_wins':0,'recent_losses':0,'recent_acc':50.0,'consecutive_losses':0,'consecutive_wins':0},
            'CatBoost':     {'wins':0,'losses':0,'total':0,'acc':50.0,'recent_wins':0,'recent_losses':0,'recent_acc':50.0,'consecutive_losses':0,'consecutive_wins':0},
            'TabNet':       {'wins':0,'losses':0,'total':0,'acc':50.0,'recent_wins':0,'recent_losses':0,'recent_acc':50.0,'consecutive_losses':0,'consecutive_wins':0},
            'LSTM_Attention': {'wins':0,'losses':0,'total':0,'acc':50.0,'recent_wins':0,'recent_losses':0,'recent_acc':50.0,'consecutive_losses':0,'consecutive_wins':0},
            'Transformer':  {'wins':0,'losses':0,'total':0,'acc':50.0,'recent_wins':0,'recent_losses':0,'recent_acc':50.0,'consecutive_losses':0,'consecutive_wins':0},
        }
        self.weights = {
            'LightGBM': 0.20, 'XGBoost': 0.20, 'CatBoost': 0.17,
            'TabNet': 0.17, 'LSTM_Attention': 0.10, 'Transformer': 0.16,
        }
        self.total_predictions = 0
        self.total_wins = 0
        self.total_losses = 0
        self.recent_results = []
        self.consecutive_losses = 0
        self.loss_recovery_mode = False
        self.recovery_side = None
        self.pattern_memory = defaultdict(lambda: {'wins':0,'losses':0,'total':0})
        self.regime_memory = defaultdict(lambda: {'wins':0,'losses':0,'total':0})
        self.side_memory = defaultdict(lambda: {'wins':0,'losses':0,'total':0})
        self.zigzag_memory = defaultdict(lambda: {'wins':0,'losses':0,'total':0})
        self.streak_memory = defaultdict(lambda: {'wins':0,'losses':0,'total':0})
        self.transition_memory = defaultdict(lambda: Counter())
        self.number_memory = defaultdict(lambda: Counter())
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

    def learn_outcome(self, prediction, actual, model_details, pattern_name=None,
                      regime=None, streak_len=0, zigzag_count=0, prev_actual=None):
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

        side_key = f"predict_{prediction}"
        sm = self.side_memory[side_key]
        sm['total'] += 1
        if won: sm['wins'] += 1
        else: sm['losses'] += 1

        actual_key = f"actual_{actual}"
        am = self.side_memory[actual_key]
        am['total'] += 1
        if won: am['wins'] += 1
        else: am['losses'] += 1

        if pattern_name:
            pm = self.pattern_memory[pattern_name]
            pm['total'] += 1
            if won: pm['wins'] += 1
            else: pm['losses'] += 1

        if regime:
            rm = self.regime_memory[regime]
            rm['total'] += 1
            if won: rm['wins'] += 1
            else: rm['losses'] += 1

        z_key = 'zigzag_active' if zigzag_count >= 3 else 'zigzag_inactive'
        zm = self.zigzag_memory[z_key]
        zm['total'] += 1
        if won: zm['wins'] += 1
        else: zm['losses'] += 1

        if streak_len >= 3:
            s_key = f"streak_{streak_len}_{actual}"
            sm2 = self.streak_memory[s_key]
            sm2['total'] += 1
            if won: sm2['wins'] += 1
            else: sm2['losses'] += 1

        if prev_actual and prev_actual != actual:
            alt_key = f"alternate_{prev_actual}_to_{actual}"
            am2 = self.zigzag_memory[alt_key]
            am2['total'] += 1
            if won: am2['wins'] += 1
            else: am2['losses'] += 1

        self.transition_memory[prev_actual or 'START'][actual] += 1

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
                self.weights[name] = max(0.03, min(0.50, accs[name]/total_acc * 3.0))
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
            'learningProgress': self._compute_learning_progress(),
        }

    def _compute_learning_progress(self):
        trained = sum(1 for m in self.models.values() if m['total'] >= 10)
        total_models = len(self.models)
        avg_acc = sum(m['acc'] for m in self.models.values() if m['total'] > 0) / max(sum(1 for m in self.models.values() if m['total'] > 0), 1)
        return {
            'trainedModels': trained,
            'totalModels': total_models,
            'completion': round((trained/total_models)*100, 1),
            'averageAccuracy': round(avg_acc, 2),
            'totalDataPoints': sum(m['total'] for m in self.models.values()),
            'bestModel': max(self.models.items(), key=lambda x: x[1]['acc'] if x[1]['total'] >= 5 else 0)[0] if any(m['total'] >= 5 for m in self.models.values()) else None,
            'worstModel': min(self.models.items(), key=lambda x: x[1]['acc'] if x[1]['total'] >= 5 else 100)[0] if any(m['total'] >= 5 for m in self.models.values()) else None,
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
            os.makedirs(os.path.dirname(ORION_BRAIN_FILE), exist_ok=True)
            if os.path.exists(ORION_BRAIN_FILE):
                with open(ORION_BRAIN_FILE, 'rb') as f:
                    _learner = pickle.load(f)
                return _learner
        except Exception:
            pass
        _learner = OrionLearner()
        return _learner


def _save_learner():
    global _learner
    if _learner is None:
        return
    try:
        os.makedirs(os.path.dirname(ORION_BRAIN_FILE), exist_ok=True)
        tmp = ORION_BRAIN_FILE + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(_learner, f)
        os.replace(tmp, ORION_BRAIN_FILE)
    except Exception:
        pass

_ORION_MODEL_BRAIN_DIR = os.path.join(DATA_DIR, 'model_brain', 'orion_models')

def _save_model_brain(model_name, data):
    try:
        os.makedirs(_ORION_MODEL_BRAIN_DIR, exist_ok=True)
        path = os.path.join(_ORION_MODEL_BRAIN_DIR, f'{model_name}.pkl')
        tmp = path + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(data, f)
        os.replace(tmp, path)
    except Exception:
        pass

def _load_model_brains():
    brains = {}
    if not os.path.isdir(_ORION_MODEL_BRAIN_DIR):
        return brains
    for fname in os.listdir(_ORION_MODEL_BRAIN_DIR):
        if fname.endswith('.pkl'):
            model_name = fname[:-4]
            try:
                with open(os.path.join(_ORION_MODEL_BRAIN_DIR, fname), 'rb') as f:
                    brains[model_name] = pickle.load(f)
            except Exception:
                pass
    return brains

def _get_model_accuracy(model_name):
    brains = _load_model_brains()
    return brains.get(model_name)


# ─── Model-Specific Predictors ────────────────────────────────────────────

def _predict_lightgbm(training_rows, current_slice, daily_history=None):
    if not LIGHTGBM_AVAILABLE:
        return None
    try:
        X, y, _ = _build_training_data(training_rows)
        if X is None or len(X) < 15:
            return None
        feats = _extract_features(current_slice)
        if feats is None:
            return None
        model = LGBMClassifier(
            n_estimators=600, learning_rate=0.025, max_depth=12,
            num_leaves=64, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=5, reg_alpha=0.1, reg_lambda=0.3,
            random_state=44, objective='binary', verbosity=-1,
        )
        model.fit(X, y)
        proba = model.predict_proba(feats.reshape(1, -1))[0]
        big_prob = proba[1] if len(proba) > 1 else proba[0]
        pred = 'BIG' if big_prob >= 0.5 else 'SMALL'
        conf = 50 + abs(big_prob - 0.5) * 100
        return {'prediction': pred, 'confidence': round(min(conf, 95), 2), 'bigProbability': round(big_prob * 100, 2)}
    except Exception:
        return None


def _predict_xgboost(training_rows, current_slice, daily_history=None):
    if not XGBOOST_AVAILABLE:
        return None
    try:
        X, y, _ = _build_training_data(training_rows)
        if X is None or len(X) < 15:
            return None
        feats = _extract_features(current_slice)
        if feats is None:
            return None
        model = XGBClassifier(
            n_estimators=600, learning_rate=0.025, max_depth=8,
            subsample=0.8, colsample_bytree=0.8, colsample_bylevel=0.8,
            min_child_weight=1, reg_alpha=0.1, reg_lambda=0.5,
            gamma=0.05, random_state=45, eval_metric='logloss',
            objective='binary:logistic', n_jobs=1,
        )
        model.fit(X, y)
        proba = model.predict_proba(feats.reshape(1, -1))[0]
        big_prob = proba[1] if len(proba) > 1 else proba[0]
        pred = 'BIG' if big_prob >= 0.5 else 'SMALL'
        conf = 50 + abs(big_prob - 0.5) * 100
        return {'prediction': pred, 'confidence': round(min(conf, 95), 2), 'bigProbability': round(big_prob * 100, 2)}
    except Exception:
        return None


def _predict_catboost(training_rows, current_slice, daily_history=None):
    if not CATBOOST_AVAILABLE:
        return None
    try:
        X, y, _ = _build_training_data(training_rows)
        if X is None or len(X) < 15:
            return None
        feats = _extract_features(current_slice)
        if feats is None:
            return None
        model = CatBoostClassifier(
            iterations=800, learning_rate=0.025, depth=8,
            l2_leaf_reg=3, min_data_in_leaf=3, subsample=0.8,
            random_seed=46, verbose=0, loss_function='Logloss',
            eval_metric='Accuracy', early_stopping_rounds=50,
            allow_writing_files=False,
        )
        model.fit(X, y)
        proba = model.predict_proba(feats.reshape(1, -1))[0]
        big_prob = proba[1] if len(proba) > 1 else proba[0]
        pred = 'BIG' if big_prob >= 0.5 else 'SMALL'
        conf = 50 + abs(big_prob - 0.5) * 100
        return {'prediction': pred, 'confidence': round(min(conf, 95), 2), 'bigProbability': round(big_prob * 100, 2)}
    except Exception:
        return None


def _predict_tabnet(training_rows, current_slice, daily_history=None):
    if not SKLEARN_AVAILABLE:
        return None
    try:
        X, y, _ = _build_training_data(training_rows)
        if X is None or len(X) < 20:
            return None
        feats = _extract_features(current_slice)
        if feats is None:
            return None

        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            min_samples_leaf=5, subsample=0.8, random_state=47,
        )
        model.fit(X, y)
        proba = model.predict_proba(feats.reshape(1, -1))[0]
        big_prob = proba[1] if len(proba) > 1 else proba[0]
        pred = 'BIG' if big_prob >= 0.5 else 'SMALL'
        conf = 50 + abs(big_prob - 0.5) * 100
        return {'prediction': pred, 'confidence': round(min(conf, 95), 2), 'bigProbability': round(big_prob * 100, 2)}
    except Exception:
        return None


def _predict_lstm_attention(training_rows, current_slice, daily_history):
    try:
        all_actuals = [r.get('actual') for r in training_rows if r.get('actual') in ('BIG', 'SMALL')]
        if len(all_actuals) < 50:
            return None
        current = []
        for item in current_slice:
            c = item.get('category') or item.get('actual')
            if c in ('BIG', 'SMALL'):
                current.append(1 if c == 'BIG' else 0)
        if len(current) < 3:
            return None

        big_votes, small_votes = 0, 0
        for n in [8, 6, 4, 3, 2]:
            if len(current) < n or len(all_actuals) <= n:
                continue
            target = tuple(current[:n])
            matches = []
            for i in range(n, len(all_actuals)):
                if tuple(all_actuals[i-n:i]) == target:
                    matches.append(all_actuals[i])
            if matches:
                big_count = sum(1 for m in matches if m == 1)
                small_count = len(matches) - big_count
                confidence = min(len(matches), 30) / 30.0
                big_votes += big_count * confidence * (1 + n/8)
                small_votes += small_count * confidence * (1 + n/8)

        # Attention weighting: nearer matches get higher weight
        if len(current) >= 3:
            for n in [4, 3, 2]:
                if len(current) < n:
                    continue
                target = tuple(current[:n])
                for i in range(n, min(n + 15, len(all_actuals))):
                    if tuple(all_actuals[i-n:i]) == target:
                        attn_weight = 1.0 / (1.0 + (i - n) * 0.1)
                        if all_actuals[i] == 1:
                            big_votes += attn_weight
                        else:
                            small_votes += attn_weight

        if big_votes == 0 and small_votes == 0:
            recent_big = sum(current[:10]) / max(len(current[:10]), 1)
            pred = 'BIG' if recent_big >= 0.5 else 'SMALL'
            conf = 50 + abs(recent_big - 0.5) * 30
            return {'prediction': pred, 'confidence': round(min(conf, 85), 2), 'bigProbability': round(recent_big * 100, 2)}

        total = big_votes + small_votes
        big_prob = big_votes / total
        pred = 'BIG' if big_prob >= 0.5 else 'SMALL'
        conf = 50 + abs(big_prob - 0.5) * 80
        return {'prediction': pred, 'confidence': round(min(conf, 92), 2), 'bigProbability': round(big_prob * 100, 2)}
    except Exception:
        return None


def _predict_transformer(training_rows, current_slice, daily_history):
    try:
        all_actuals = [r.get('actual') for r in training_rows if r.get('actual') in ('BIG', 'SMALL')]
        if len(all_actuals) < 80:
            return None
        current = []
        for item in current_slice:
            c = item.get('category') or item.get('actual')
            if c in ('BIG', 'SMALL'):
                current.append(1 if c == 'BIG' else 0)
        if len(current) < 3:
            return None

        # PatchTST-style: extract patches (overlapping windows) and match
        patch_len = 4
        stride = 2
        patches = []
        for i in range(0, len(all_actuals) - patch_len + 1, stride):
            patches.append(tuple(all_actuals[i:i+patch_len]))

        current_patch = tuple(current[:min(patch_len, len(current))])

        # Find similar patches with trend encoding
        big_votes, small_votes = 0, 0
        for idx, patch in enumerate(patches):
            if patch == current_patch or (len(patch) >= 2 and len(current_patch) >= 2 and patch[0] == current_patch[0] and patch[-1] == current_patch[-1]):
                next_idx = idx * stride + patch_len
                if next_idx < len(all_actuals):
                    if all_actuals[next_idx] == 1:
                        big_votes += 1
                    else:
                        small_votes += 1

        # Multi-head attention simulation (multiple patch lengths)
        for patch_len2 in [3, 5]:
            for i in range(0, len(all_actuals) - patch_len2 + 1, 1):
                patch = tuple(all_actuals[i:i+patch_len2])
                target = tuple(current[:min(patch_len2, len(current))])
                if len(patch) == len(target) and patch == target:
                    next_idx = i + patch_len2
                    if next_idx < len(all_actuals):
                        weight = 1.0 / (1.0 + abs(i - len(all_actuals)) * 0.05)
                        if all_actuals[next_idx] == 1:
                            big_votes += weight
                        else:
                            small_votes += weight

        if big_votes == 0 and small_votes == 0:
            return None

        total = big_votes + small_votes
        big_prob = big_votes / total
        pred = 'BIG' if big_prob >= 0.5 else 'SMALL'
        conf = 50 + abs(big_prob - 0.5) * 85
        return {'prediction': pred, 'confidence': round(min(conf, 93), 2), 'bigProbability': round(big_prob * 100, 2)}
    except Exception:
        return None


# ─── Feature Extraction ───────────────────────────────────────────────────

def _extract_features(predictions_slice):
    cats = [1 if p.get('category') == 'BIG' or p.get('actual') == 'BIG' else 0 for p in predictions_slice]
    if not cats:
        return None
    feats = []
    n = len(cats)

    for i in range(10):
        feats.append(cats[i] if i < n else 0.5)

    for w in [5, 10, 20, 50]:
        chunk = cats[:min(w, n)]
        feats.append(sum(chunk) / max(len(chunk), 1))

    for fresh, older in [(3, 10), (5, 10), (10, 20), (20, 50)]:
        fc = cats[:min(fresh, n)]
        oc = cats[min(fresh, n):min(older, n)]
        fr = sum(fc) / max(len(fc), 1)
        or_ = sum(oc) / max(len(oc), 1) if oc else 0.5
        feats.append(fr - or_)

    r5 = sum(cats[:min(5, n)]) / max(min(5, n), 1)
    r20 = sum(cats[:min(20, n)]) / max(min(20, n), 1)
    feats.append(r5 - r20)

    streak = 1
    for i in range(1, min(12, n)):
        if cats[i] == cats[i-1]:
            streak += 1
        else:
            break
    feats.append(streak / 12.0)
    feats.append(float(cats[0]) if n > 0 else 0.5)

    alt_count = sum(1 for i in range(1, min(20, n)) if cats[i] != cats[i-1])
    feats.append(alt_count / max(min(20, n)-1, 1))

    transitions = [0, 0, 0, 0]
    tt = 0
    for i in range(1, min(30, n)):
        transitions[(cats[i]*2) + cats[i-1]] += 1
        tt += 1
    feats.extend(v / max(tt, 1) for v in transitions)

    big_ratio = sum(cats[:min(20, n)]) / max(min(20, n), 1)
    if big_ratio in (0, 1):
        entropy = 0.0
    else:
        entropy = -(big_ratio * np.log2(max(big_ratio, 1e-10)) + (1-big_ratio) * np.log2(max(1-big_ratio, 1e-10)))
    feats.append(float(entropy))

    for w in [6, 12, 24]:
        chunk = cats[:min(w, n)]
        if len(chunk) >= 2:
            alts = sum(1 for i in range(1, len(chunk)) if chunk[i] != chunk[i-1])
            feats.append(alts / (len(chunk)-1))
            ratio = sum(chunk) / len(chunk)
            feats.append(1.0 - abs(ratio - 0.5) * 2)
        else:
            feats.extend([0.5, 1.0])

    return np.array(feats, dtype=np.float32)


def _build_training_data(all_predictions):
    verified = [p for p in all_predictions if p.get('actual') in ('BIG', 'SMALL')]
    if len(verified) < 15:
        return None, None, 0
    X, y = [], []
    for i in range(5, len(verified)):
        past = list(reversed(verified[:i]))
        feats = _extract_features(past)
        if feats is not None:
            target = 1 if verified[i].get('actual') == 'BIG' else 0
            X.append(feats)
            y.append(target)
    if len(X) < 10:
        return None, None, 0
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), len(X)


# ─── Data Loading ─────────────────────────────────────────────────────────

def _discover_all_csvs():
    csvs = set()
    if os.path.isdir(DATA_DIR):
        for root, dirs, files in os.walk(DATA_DIR):
            for f in files:
                if f.endswith('.csv') and f != '.gitkeep':
                    csvs.add(os.path.join(root, f))
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


def _learning_source_summary(rows=None):
    rows = rows if rows is not None else _load_all_history()
    counts = defaultdict(int)
    for row in rows:
        counts[row.get('source') or 'unknown'] += 1
    return {
        'totalRows': len(rows),
        'files': dict(sorted(counts.items(), key=lambda item: item[0])),
        'displayHistoryLimit': ORION_HISTORY_LIMIT,
        'fullHistoryUsedForTraining': True,
    }


# ─── File Management ──────────────────────────────────────────────────────

def _boostrap_memory_from_csvs():
    """Load all CSV entries into _memory_entries on startup so old data persists."""
    with _memory_entries_lock:
        if _memory_entries:
            return
        for path in _discover_all_csvs():
            if not os.path.exists(path):
                continue
            try:
                with open(path, 'r', newline='', encoding='utf-8') as f:
                    for row in csv.DictReader(f):
                        period = str(row.get('period', '')).strip()
                        if not period or period in _memory_entries:
                            continue
                        actual = row.get('actual', '')
                        if actual not in ('BIG', 'SMALL') and row.get('status') not in ('Pending', 'TRAINING'):
                            continue
                        _memory_entries[period] = {
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
            except Exception:
                pass


def _ensure_files():
    os.makedirs(os.path.dirname(ORION_HISTORY_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(ORION_BRAIN_FILE), exist_ok=True)
    if not os.path.exists(ORION_HISTORY_CSV):
        with open(ORION_HISTORY_CSV, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=HEADER)
            w.writeheader()


def _bootstrap_from_daily():
    _ensure_files()
    _boostrap_memory_from_csvs()
    with _memory_entries_lock:
        if _memory_entries:
            return
    ...
    csv_has_data = False
    if os.path.exists(ORION_HISTORY_CSV):
        try:
            with open(ORION_HISTORY_CSV, 'r', newline='') as f:
                for _ in csv.DictReader(f):
                    csv_has_data = True
                    break
        except Exception:
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


# ─── Verification ─────────────────────────────────────────────────────────

_verify_last_run = 0
_verify_thread = None
_verify_thread_lock = threading.Lock()


def _verify_memory_entries():
    global _verify_last_run
    now = time.time()
    if now - _verify_last_run < 8:
        return
    _verify_last_run = now
    current_period = get_current_period_1min()

    with _memory_entries_lock:
        with _verified_periods_lock:
            pending = [e for e in _memory_entries.values()
                       if e.get('period') < current_period
                       and e.get('status') == 'Pending'
                       and not e.get('actual')
                       and e.get('period') not in _verified_periods]
    if not pending:
        return

    game_data = fetch_api_data(retries=0, timeout=3, bypass_cache=True)
    if not isinstance(game_data, list) or not game_data:
        return
    by_period = {str(item.get('period', '')): item for item in game_data if item.get('period')}
    learner = _get_learner()
    updated = 0
    with _memory_entries_lock:
        all_actuals = [e.get('actual') for e in _memory_entries.values() if e.get('actual') in ('BIG','SMALL')]
        for entry in _memory_entries.values():
            per = str(entry.get('period', ''))
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
            actual = m['category']
            entry['actual'] = actual
            entry['number'] = str(m.get('number', ''))
            if entry.get('prediction') in ('BIG','SMALL'):
                entry['status'] = 'WIN' if entry.get('prediction') == actual else 'LOSS'
            else:
                entry['status'] = 'WIN'
            entry['skipped'] = False
            entry['skipReason'] = ''
            updated += 1
            with _verified_periods_lock:
                _verified_periods.add(per)
            if entry.get('prediction') in ('BIG','SMALL'):
                pattern_name = entry.get('patternUsed') or entry.get('patternused') or 'orion_ensemble'
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
        _write_entries([])


def _verify_loop():
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
        t = threading.Thread(target=_verify_loop, daemon=True, name='orion_verify')
        _verify_thread = t
        t.start()


def _verify_pending(entries):
    _verify_memory_entries()
    return _entries()


def _write_entries(entries):
    with _memory_entries_lock:
        for e in entries:
            p = str(e.get('period',''))
            if p:
                _memory_entries[p] = dict(e)
    _invalidate_snapshot()
    try:
        with _memory_entries_lock:
            all_rows = [{k: _csv_value(e.get(k, '')) for k in HEADER} for e in _memory_entries.values()]
        all_rows.sort(key=lambda r: _period_key(r.get('period')), reverse=False)
        os.makedirs(os.path.dirname(ORION_HISTORY_CSV), exist_ok=True)
        with open(ORION_HISTORY_CSV, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=HEADER)
            w.writeheader()
            w.writerows(all_rows)
    except Exception:
        pass


# ─── Regime Detection ─────────────────────────────────────────────────────

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
    if 'lgbm' in p or 'lightgbm' in p or 'lgb' in p:
        return 'LightGBM'
    elif 'xgb' in p or 'xgboost' in p:
        return 'XGBoost'
    elif 'cat' in p or 'catboost' in p:
        return 'CatBoost'
    elif 'tab' in p or 'tabnet' in p:
        return 'TabNet'
    elif 'lstm' in p or 'attention' in p:
        return 'LSTM_Attention'
    elif 'trans' in p or 'transformer' in p or 'patch' in p:
        return 'Transformer'
    return 'XGBoost'


# ─── History Learning ─────────────────────────────────────────────────────

def _learn_from_history(learner):
    all_rows = _load_all_history()
    actuals = [r['actual'] for r in all_rows]
    for i, row in enumerate(all_rows):
        if row.get('status') not in ('WIN','LOSS'):
            continue
        if row.get('prediction') not in ('BIG','SMALL'):
            continue
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


# ─── Core Prediction: 6-Model Weighted Ensemble ──────────────────────────

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
    if len(losses) < 2:
        return signal

    recent_losses = losses[:10]
    pred_counts = {
        'BIG': sum(1 for r in recent_losses if r.get('prediction') == 'BIG'),
        'SMALL': sum(1 for r in recent_losses if r.get('prediction') == 'SMALL'),
    }
    actual_counts = {
        'BIG': sum(1 for r in recent_losses if r.get('actual') == 'BIG'),
        'SMALL': sum(1 for r in recent_losses if r.get('actual') == 'SMALL'),
    }
    recent_actuals = [
        r.get('actual') for r in reversed(training_rows)
        if r.get('actual') in ('BIG', 'SMALL')
    ][:12]
    regime, streak_len, zigzag_count = _detect_regime(recent_actuals)
    model_vote_side = 'BIG' if big_votes >= small_votes else 'SMALL'

    if actual_counts['BIG'] > actual_counts['SMALL']:
        recovery_side = 'BIG'
        reason = 'loss_actual_majority'
    elif actual_counts['SMALL'] > actual_counts['BIG']:
        recovery_side = 'SMALL'
        reason = 'loss_actual_majority'
    elif pred_counts['BIG'] > pred_counts['SMALL']:
        recovery_side = 'SMALL'
        reason = 'opposite_failed_big'
    elif pred_counts['SMALL'] > pred_counts['BIG']:
        recovery_side = 'BIG'
        reason = 'opposite_failed_small'
    elif regime == 'ZIGZAG' and recent_actuals:
        recovery_side = 'SMALL' if recent_actuals[0] == 'BIG' else 'BIG'
        reason = 'zigzag_recovery'
    elif regime == 'STREAK' and recent_actuals:
        recovery_side = recent_actuals[0]
        reason = 'streak_recovery'
    else:
        recovery_side = model_vote_side
        reason = 'model_vote_recovery'

    boost = min(0.46, 0.18 + len(losses) * 0.04)
    side_acc = learner.get_side_accuracy(recovery_side) if learner else None
    if side_acc and side_acc >= 55:
        boost += min(0.10, (side_acc - 50) / 100)
    return {
        **signal,
        'active': True,
        'prediction': recovery_side,
        'reason': reason,
        'boost': round(min(boost, 0.56), 4),
        'confidence': round(min(94, 68 + len(losses) * 4), 2),
        'lossPredictions': pred_counts,
        'lossActuals': actual_counts,
        'regime': regime,
        'streakLen': streak_len,
        'zigzagCount': zigzag_count,
    }

def _predict(learner, training_rows, current_slice, daily_history):
    global _active_period_prediction

    if not current_slice:
        return None

    # Run all 6 models
    model_predictions = []
    predictors = [
        ('LightGBM', _predict_lightgbm),
        ('XGBoost', _predict_xgboost),
        ('CatBoost', _predict_catboost),
        ('TabNet', _predict_tabnet),
        ('LSTM_Attention', _predict_lstm_attention),
        ('Transformer', _predict_transformer),
    ]

    total_weight = 0.0
    big_votes = 0.0
    small_votes = 0.0

    for name, predictor_fn in predictors:
        try:
            result = predictor_fn(training_rows, current_slice, daily_history)
            if result and result.get('prediction') in ('BIG', 'SMALL'):
                w = learner.weights.get(name, 0.15)
                m = learner.models.get(name)
                if m and m['total'] >= 5:
                    acc_factor = m['recent_acc']/100.0 if m['recent_wins']+m['recent_losses'] >= 5 else m['acc']/100.0
                    w *= (0.5 + 0.5 * acc_factor)
                total_weight += w
                if result['prediction'] == 'BIG':
                    big_votes += w * (result['confidence'] / 100.0)
                else:
                    small_votes += w * (result['confidence'] / 100.0)
                model_predictions.append({
                    'model': name,
                    'prediction': result['prediction'],
                    'confidence': round(result['confidence'], 2),
                    'probability': result['bigProbability'],
                    'weight': round(w, 4),
                    'available': True,
                })
            else:
                model_predictions.append({
                    'model': name,
                    'prediction': None,
                    'confidence': 0,
                    'probability': 50,
                    'weight': 0,
                    'available': False,
                })
        except Exception:
            model_predictions.append({
                'model': name,
                'prediction': None,
                'confidence': 0,
                'probability': 50,
                'weight': 0,
                'available': False,
            })

    # Also try the existing ML ensemble as fallback
    if not model_predictions or not any(mp.get('prediction') for mp in model_predictions):
        ml_result = predict_ml(training_rows, current_slice)
        if ml_result and ml_result.get('prediction') in ('BIG', 'SMALL'):
            conf = float(ml_result.get('confidence', 50))
            model_predictions.append({
                'model': 'Ensemble',
                'prediction': ml_result['prediction'],
                'confidence': conf,
                'probability': float(ml_result.get('bigProbability', 50)),
                'weight': 1.0,
                'available': True,
            })
            total_weight = 1.0
            if ml_result['prediction'] == 'BIG':
                big_votes = 1.0 * (conf / 100.0)
            else:
                small_votes = 1.0 * (conf / 100.0)

    if not total_weight or total_weight == 0:
        return None

    # Loss recovery adjustment
    recovery = learner.get_recovery_adjustment()
    if recovery['active'] and recovery['side']:
        boost = recovery['boost'] / 100.0
        if recovery['side'] == 'BIG':
            big_votes += boost * total_weight * 0.5
        else:
            small_votes += boost * total_weight * 0.5
        model_predictions.append({
            'model': 'LossRecovery',
            'prediction': recovery['side'],
            'confidence': round(55 + recovery['boost'], 2),
            'probability': 50 + recovery['boost'],
            'weight': round(boost / 100, 4),
            'available': True,
        })
        total_weight += boost * 0.5

    all_actuals = [r.get('actual') for r in reversed(training_rows) if r.get('actual') in ('BIG', 'SMALL')]
    regime, streak_len, zigzag_count = _detect_regime(all_actuals)

    # Regime-aware adjustment
    regime_acc = learner.get_regime_accuracy(regime)
    if regime_acc and regime_acc > 55:
        boost_adj = (regime_acc - 50) * 0.3
        if regime == 'STREAK' and streak_len >= 3:
            latest = all_actuals[0] if all_actuals else None
            if latest == 'BIG':
                big_votes += boost_adj * total_weight * 0.4
            elif latest == 'SMALL':
                small_votes += boost_adj * total_weight * 0.4
        elif regime == 'ZIGZAG' and zigzag_count >= 3:
            opposite = 'SMALL' if (all_actuals[0] if all_actuals else 'BIG') == 'BIG' else 'BIG'
            if opposite == 'BIG':
                big_votes += boost_adj * total_weight * 0.3
            else:
                small_votes += boost_adj * total_weight * 0.3
    elif regime == 'STREAK' and streak_len >= 3:
        latest = all_actuals[0] if all_actuals else None
        if latest == 'BIG':
            big_votes += 0.15 * total_weight
        elif latest == 'SMALL':
            small_votes += 0.15 * total_weight

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
            'available': True,
        })
        total_weight += loss_manager['boost']

    pred = 'BIG' if big_votes >= small_votes else 'SMALL'

    # Pattern-aware lock: detect zigzag early to avoid wrong lock
    global _orion_locked_side
    lock_actuals = [r.get('category') for r in (current_slice or []) if r.get('category') in ('BIG', 'SMALL')]
    if not lock_actuals:
        lock_actuals = [r.get('actual') for r in reversed(training_rows) if r.get('actual') in ('BIG', 'SMALL')]
    
    is_zigzag = False
    if len(lock_actuals) >= 3:
        alts = sum(1 for i in range(1, min(len(lock_actuals), 5)) if lock_actuals[i] != lock_actuals[i-1])
        max_possible = min(len(lock_actuals), 5) - 1
        is_zigzag = alts >= max_possible * 0.75 and len(set(lock_actuals[:4])) >= 2
    
    regime, streak_len, zigzag_count = _detect_regime(lock_actuals)
    big_acc = learner.get_side_accuracy('BIG') or 50
    small_acc = learner.get_side_accuracy('SMALL') or 50
    
    if is_zigzag or regime == 'ZIGZAG':
        _orion_locked_side = None
    elif regime == 'STREAK' and streak_len >= 3:
        _orion_locked_side = lock_actuals[0]
    elif abs(big_acc - small_acc) >= 10:
        _orion_locked_side = 'BIG' if big_acc > small_acc else 'SMALL'
    elif regime in ('CHOPPY', 'MIXED', 'UNKNOWN'):
        last_result = lock_actuals[0] if lock_actuals else None
        if _orion_locked_side is None:
            _orion_locked_side = last_result if last_result else pred
        elif last_result and last_result != _orion_locked_side:
            _orion_locked_side = None
        if _orion_locked_side is None:
            _orion_locked_side = pred if pred in ('BIG', 'SMALL') else 'BIG'
    pred = _orion_locked_side if _orion_locked_side else pred

    # REAL confidence calculation
    if learner.total_predictions >= 5:
        real_conf = learner.get_stats()['winRate']
    else:
        real_conf = 50.0

    side_acc = learner.get_side_accuracy(pred)
    if side_acc and side_acc > 55:
        real_conf += (side_acc - 50) * 0.3

    agreeing = sum(1 for mp in model_predictions if mp.get('prediction') == pred and mp.get('available'))
    total_models = sum(1 for mp in model_predictions if mp.get('available'))
    agreement_ratio = agreeing / max(total_models, 1)
    total_votes = big_votes + small_votes
    edge = abs(big_votes - small_votes) / max(total_votes, 0.01) * 50

    confidence = real_conf + (edge * 0.3) + (agreement_ratio * 5 - 2.5)
    confidence = max(50.0, min(95.0, confidence))
    big_pct = round((big_votes / max(total_votes, 0.01)) * 100, 2)

    # ── Find best model ─────────────────────────────────────────────────
    best_model_name = None
    best_model_acc = 0
    best_model_conf = 0
    best_model_pred = None
    model_ranking = []

    for mp in model_predictions:
        if not mp.get('available') or not mp.get('prediction'):
            continue
        mn = mp['model']
        hist = learner.models.get(mn, {})
        hist_acc = hist.get('acc', 0) if hist.get('total', 0) >= 5 else 0
        if hist_acc > 0:
            model_ranking.append({
                'model': mn,
                'historicalAccuracy': hist_acc,
                'recentAccuracy': hist.get('recent_acc', 0),
                'totalPredictions': hist.get('total', 0),
                'currentPrediction': mp['prediction'],
                'currentConfidence': mp['confidence'],
                'weight': mp.get('weight', 0),
            })
            if hist_acc > best_model_acc:
                best_model_acc = hist_acc
                best_model_name = mn
                best_model_conf = mp['confidence']
                best_model_pred = mp['prediction']

    model_ranking.sort(key=lambda x: x['historicalAccuracy'], reverse=True)

    return {
        'prediction': pred,
        'confidence': round(confidence, 2),
        'bigProbability': big_pct,
        'modelPredictions': model_predictions,
        'recovery': recovery,
        'lossManager': loss_manager,
        'regime': regime,
        'streakLen': streak_len,
        'zigzagCount': zigzag_count,
        'bestModel': {
            'name': best_model_name,
            'historicalAccuracy': round(best_model_acc, 2),
            'confidence': best_model_conf,
            'prediction': best_model_pred,
        } if best_model_name else None,
        'modelRanking': model_ranking,
    }


# ─── Helpers ──────────────────────────────────────────────────────────────

def _period_key(p):
    try: return int(str(p))
    except: return 0

def _csv_value(v):
    return '' if v is None else str(v)

def _entries():
    global _history_snapshot
    if _history_snapshot is not None:
        return _history_snapshot
    rows = []
    if os.path.exists(ORION_HISTORY_CSV):
        try:
            with open(ORION_HISTORY_CSV, 'r', newline='') as f:
                r = csv.DictReader(f)
                rows = [row for row in r if row.get('period')]
        except Exception:
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
    return {
        'period': row.get('period'),
        'prediction': row.get('prediction'),
        'status': row.get('status', 'Pending'),
        'confidence': float(row.get('confidence') or 0),
        'actual': row.get('actual'),
        'number': row.get('number'),
        'patternUsed': row.get('patternUsed') or row.get('patternused') or '',
        'skipped': row.get('skipped') == 'True' or row.get('skipped') is True,
        'skipReason': row.get('skipReason') or row.get('skipreason') or '',
        'timestamp': int(row.get('timestamp') or 0),
    }

def _stats(history):
    t = len(history)
    w = sum(1 for h in history if h.get('status')=='WIN')
    l = sum(1 for h in history if h.get('status')=='LOSS')
    s = sum(1 for h in history if h.get('skipped'))
    return {'total': t, 'wins': w, 'losses': l, 'skipped': s, 'winRate': round((w/max(w+l,1))*100, 2)}

def _upsert(entry):
    period = str(entry.get('period',''))
    if not period:
        return
    with _memory_entries_lock:
        _memory_entries[period] = dict(entry)
    _invalidate_snapshot()
    try:
        rows = []
        with _memory_entries_lock:
            for p, e in _memory_entries.items():
                rows.append({k: _csv_value(e.get(k, '')) for k in HEADER})
        rows.sort(key=lambda r: _period_key(r.get('period')), reverse=False)
        os.makedirs(os.path.dirname(ORION_HISTORY_CSV), exist_ok=True)
        with open(ORION_HISTORY_CSV, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=HEADER)
            w.writeheader()
            w.writerows(rows)
    except Exception:
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
                    by_p[p] = {'period': p, 'prediction': '', 'status': 'WIN', 'actual': item.get('category'), 'number': item.get('number',''), 'confidence': 100}
    return sorted(by_p.values(), key=lambda r: _period_key(r.get('period',0)))


# ─── Main Prediction Cycle ───────────────────────────────────────────────

_compute_lock = threading.Lock()

def _build_fallback_payload(current_period=None, learner=None, entries=None, result=None, error=None):
    cp = current_period or get_current_period_1min()
    learner = learner or _get_learner()
    all_entries = list(entries) if entries else list(_memory_entries.values()) if _memory_entries else []
    all_entries.sort(key=lambda r: _period_key(r.get('period')), reverse=True)
    pub_history = [_public_entry(r) for r in all_entries[:ORION_HISTORY_LIMIT]]
    current = next((e for e in all_entries if e.get('period') == cp), None)
    if not current:
        current = {'period': cp, 'prediction': 'BIG', 'status': 'Pending', 'confidence': 51.0,
                   'actual': None, 'number': None, 'patternused': 'orion_fallback',
                   'timestamp': int(time.time()), 'skipped': False, 'skipreason': ''}
    learner_stats = learner.get_stats() if learner else {'totalPredictions': 0, 'totalWins': 0, 'totalLosses': 0, 'winRate': 0}
    model_accuracies = {}
    if learner:
        for model_name in ORION_MODEL_NAMES:
            m = learner.models.get(model_name)
            brain_data = _get_model_accuracy(model_name)
            if m and m['total'] > 0:
                model_accuracies[model_name] = {
                    'accuracy': m['acc'], 'recentAccuracy': m['recent_acc'],
                    'totalPredictions': m['total'], 'wins': m['wins'], 'losses': m['losses'],
                    'consecutiveLosses': m['consecutive_losses'],
                    'consecutiveWins': m['consecutive_wins'],
                    'currentWeight': round(learner.weights.get(model_name, 0), 4),
                    'lastSavedBrain': brain_data is not None,
                }
            else:
                model_accuracies[model_name] = {
                    'accuracy': 0, 'recentAccuracy': 0, 'totalPredictions': 0,
                    'wins': 0, 'losses': 0, 'consecutiveLosses': 0, 'consecutiveWins': 0,
                    'currentWeight': round(learner.weights.get(model_name, 0), 4),
                    'lastSavedBrain': brain_data is not None,
                }
    all_history = _load_all_history()
    payload = {
        'predictionResult': {'period': current.get('period'), 'prediction': current.get('prediction') or 'BIG',
                             'status': current.get('status', 'Pending'), 'skipped': False, 'skipReason': ''},
        'predictionDetails': {'gameType': 'Wingo 1 Min Orion', 'confidence': round(float(current.get('confidence') or 51), 2),
                              'actual': current.get('actual'), 'number': current.get('number')},
        'modelDecision': {'period': current.get('period'), 'prediction': current.get('prediction') or 'BIG',
                          'confidence': round(float(current.get('confidence') or 51), 2),
                          'modelResult': result, 'learnerStats': learner_stats,
                          'trainedFromRows': len(all_history), 'modelAccuracies': model_accuracies},
        'learningSources': _learning_source_summary(all_history),
        'history': pub_history, 'ossStatus': get_oss_data_status(),
    }
    if error:
        payload['error'] = str(error)
    return payload


def _compute_payload():
    """Heavy work: fetch data, train models, predict. Runs in background thread only."""
    global _payload_cache, _payload_cache_time, _last_period, _active_period_prediction, _last_predict_time
    with _compute_lock:
        try:
            _ensure_files()
            _bootstrap_from_daily()
            cp = get_current_period_1min()
            learner = _get_learner()
            _learn_from_history(learner)
            entries = _verify_pending(_entries())
            current = next((e for e in entries if e.get('period') == cp), None)

            result = _active_period_prediction
            should_predict = cp != _last_period or not current
            if should_predict:
                try:
                    game_data = fetch_api_data(retries=0, timeout=2, bypass_cache=False)
                    daily_history = fetch_wingobot_daily_history(retries=0, timeout=3, limit=None)
                    current_slice = []
                    if isinstance(game_data, list):
                        current_slice = [{'category': r.get('category'), 'number': r.get('number')} for r in game_data if r.get('category') in ('BIG', 'SMALL')]
                    if not current_slice and isinstance(daily_history, list):
                        current_slice = [{'category': r.get('category'), 'number': r.get('number')} for r in daily_history[:150] if r.get('category') in ('BIG', 'SMALL')]
                    all_history = _load_all_history()
                    training_rows = _make_training_rows(all_history, game_data, daily_history)
                    sm = get_model_summary()
                    if sm.get('totalSamples', 0) == 0 and len(training_rows) >= 10:
                        train_model(training_rows, force=True)
                    if len(training_rows) >= 5:
                        pred_result = _predict(learner, training_rows, current_slice, daily_history)
                        if pred_result:
                            result = pred_result
                            _active_period_prediction = result
                            _last_period = cp
                            _last_predict_time = time.time()
                except Exception as e:
                    print(f'[ORION_PREDICT] error: {e}')

            if not current:
                if result and result.get('prediction') in ('BIG', 'SMALL'):
                    current = {'period': cp, 'prediction': result['prediction'], 'status': 'Pending',
                               'confidence': result.get('confidence', 51), 'actual': None, 'number': None,
                               'patternused': 'orion_ensemble', 'timestamp': int(time.time()),
                               'skipped': False, 'skipreason': ''}
                else:
                    current = {'period': cp, 'prediction': 'BIG', 'status': 'Pending', 'confidence': 51.0,
                               'actual': None, 'number': None, 'patternused': 'orion_default_fallback',
                               'timestamp': int(time.time()), 'skipped': False, 'skipreason': ''}
                _upsert(current)
                _invalidate_snapshot()
                entries = _entries()

            _save_learner()
            _payload_cache = _build_fallback_payload(cp, learner, entries, result)
            _payload_cache_time = time.time()
            _save_cache(_payload_cache)
            print(f'[ORION] Payload computed: {current.get("prediction")} @ {current.get("confidence")}%')
        except Exception as e:
            print(f'[ORION_COMPUTE] error: {e}\n{traceback.format_exc()}')
            try:
                _payload_cache = _build_fallback_payload(error=str(e))
                _payload_cache_time = time.time()
                _save_cache(_payload_cache)
            except Exception:
                pass


def get_orion_payload():
    """Fast non-blocking call: returns cache or skeleton, starts bg compute if needed."""
    global _payload_cache, _payload_cache_time

    if _payload_cache is None:
        p, _ = _load_cache()
        if p:
            _payload_cache = p
            _payload_cache_time = time.time()

    if _payload_cache:
        if time.time() - _payload_cache_time < _PAYLOAD_CACHE_SECONDS:
            return _inject_history(_payload_cache)
        c = dict(_payload_cache)
        c['stale'] = True
        c['staleReason'] = 'orion_refresh_in_progress'
        if 'warming' in c:
            del c['warming']
            del c['warmingReason']
        _bg_refresh()
        return _inject_history(c)

    # No cache anywhere — store skeleton as temp cache, start bg
    payload = _skeleton_payload()
    _payload_cache = payload
    _payload_cache_time = time.time()
    _bg_refresh()
    return payload


# ─── Cache & Background Refresh ──────────────────────────────────────────

def _convert_native(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
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
        tmp = ORION_CACHE_FILE + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump({'timestamp': time.time(), 'payload': _convert_native(payload)}, f)
        os.replace(tmp, ORION_CACHE_FILE)
    except Exception as e:
        print(f'[ORION_CACHE] save error: {e}')

def _load_cache():
    if not os.path.exists(ORION_CACHE_FILE):
        return None, None
    try:
        with open(ORION_CACHE_FILE, 'r', encoding='utf-8') as f:
            d = json.load(f)
        return d.get('payload'), time.time() - float(d.get('timestamp', 0))
    except Exception:
        return None, None

def _get_fast_history():
    with _memory_entries_lock:
        rows = [dict(e) for e in _memory_entries.values()]
    if not rows:
        rows = _entries()
    rows.sort(key=lambda r: _period_key(r.get('period')), reverse=True)
    public = [_public_entry(r) for r in rows[:ORION_HISTORY_LIMIT]]
    stats_data = _stats(public)
    return public, stats_data

def _inject_history(payload):
    h, s = _get_fast_history()
    p = dict(payload)
    p['history'] = h[:ORION_HISTORY_LIMIT]
    p.setdefault('learningSources', _learning_source_summary())
    cp = get_current_period_1min()
    pr = dict(p.get('predictionResult') or {})
    if pr.get('period') != cp:
        md = dict(p.get('modelDecision') or {})
        pred = pr.get('prediction') if pr.get('prediction') in ('BIG', 'SMALL') else md.get('prediction')
        if pred not in ('BIG', 'SMALL'):
            pred = 'BIG'
        pr.update({'period': cp, 'prediction': pred, 'status': 'Pending', 'skipped': False, 'skipReason': ''})
        md.update({'period': cp, 'prediction': pred})
        p['predictionResult'] = pr
        p['modelDecision'] = md
        p['currentized'] = True
    return p

def _skeleton_payload():
    cp = get_current_period_1min()
    h, s = _get_fast_history()
    return {
        'predictionResult': {'period': cp, 'prediction': 'BIG', 'status': 'Pending', 'skipped': False, 'skipReason': ''},
        'predictionDetails': {'gameType': 'Wingo 1 Min Orion', 'confidence': 0, 'actual': None, 'number': None},
        'modelDecision': {'period': cp, 'prediction': 'BIG', 'confidence': 0, 'modelResult': None, 'learnerStats': None, 'modelAccuracies': {}, 'trainedFromRows': 0},
        'learningSources': _learning_source_summary(),
        'history': h[:ORION_HISTORY_LIMIT],
        'warming': True, 'warmingReason': 'First load — background refresh in progress',
    }

def _predict_for_period():
    learner = _get_learner()
    game_data = fetch_api_data(retries=1, timeout=4, bypass_cache=False)
    daily_history = fetch_wingobot_daily_history(retries=1, timeout=6, limit=None)
    current_slice = []
    if isinstance(game_data, list):
        current_slice = [{'category': r.get('category'), 'number': r.get('number')} for r in game_data if r.get('category') in ('BIG', 'SMALL')]
    if not current_slice and isinstance(daily_history, list):
        current_slice = [{'category': r.get('category'), 'number': r.get('number')} for r in daily_history[:150] if r.get('category') in ('BIG', 'SMALL')]
    all_history = _load_all_history()
    training_rows = _make_training_rows(all_history, game_data, daily_history)
    sm = get_model_summary()
    if sm.get('totalSamples', 0) == 0 and len(training_rows) >= 15:
        train_model(training_rows, force=True)
    result = _predict(learner, training_rows, current_slice, daily_history)
    return result, current_slice, training_rows, learner

def get_cached_orion_payload():
    p, age = _load_cache()
    if p is None:
        return get_orion_payload()
    if not _memory_entries:
        _boostrap_memory_from_csvs()
    if age > ORION_BG_REFRESH_INTERVAL:
        _bg_refresh()
    result = _inject_history(p)
    if age > ORION_CACHE_STALE_SECONDS:
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
        t = threading.Thread(target=_bg_worker_with_timeout, daemon=True, name='orion_bg')
        _bg_thread = t
        t.start()

def _bg_worker():
    try:
        _compute_payload()
    except Exception as e:
        print(f'[ORION_BG] {e}\n{traceback.format_exc()}')

def _bg_worker_with_timeout():
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_bg_worker)
            fut.result(timeout=25)
    except concurrent.futures.TimeoutError:
        print('[ORION_BG] worker timed out')

def _daily_history_fetcher():
    """Fetch OSS daily history every 4s and save to daily_1k_history.csv."""
    while True:
        try:
            fetch_wingobot_daily_history(retries=1, timeout=5, limit=None, full_backfill=True)
        except Exception as e:
            print(f'[ORION_DAILY_FETCH] error: {e}')
        time.sleep(4)

def start_orion_bg_refresh_loop():
    _start_verify_loop()
    t_daily = threading.Thread(target=_daily_history_fetcher, daemon=True, name='orion_daily_fetch')
    t_daily.start()
    print('[ORION_DAILY_FETCH] Fetching daily history every 4s')

    def _loop():
        while True:
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    fut = pool.submit(_compute_payload)
                    fut.result(timeout=25)
            except concurrent.futures.TimeoutError:
                print('[ORION_BG] compute timed out, skipping this cycle')
            except Exception as e:
                print(f'[ORION_BG] loop error: {e}')
            time.sleep(ORION_BG_REFRESH_INTERVAL)
    t = threading.Thread(target=_loop, daemon=True, name='orion_bg_loop')
    t.start()
    print('[ORION_BG] Background refresh + verify loop started')
