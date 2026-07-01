import csv
import time
import json
import os
import numpy as np
from collections import deque

try:
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

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

from config import DATA_DIR
from helpers import build_default_user_state

MODEL_FILE = os.path.join(DATA_DIR, 'ml_model.json')
ML_METADATA_FILE = os.path.join(DATA_DIR, 'ml_metadata.json')
BASE_DIR = os.path.dirname(__file__)
LSTM_MIN_HISTORY = 1000

_model = None
_retrain_threshold = 1
_retrain_interval = 14400  # 4 hours
_accuracy_cache = {
    'accuracy': 50,
    'recentAccuracy': 50,
    'total': 0,
    'cached_at': 0,
    'data_count': 0
}
_RETRAIN_META_FILE = ML_METADATA_FILE  # reuse same file
_last_train_count = 0
MODEL_VERSION = 7   # force retrain on latest data

# ---------------------------------------------------------------------------
# Pattern Failure Tracker – tracks which N-gram patterns are currently losing
# so the model can self-correct in real-time without waiting for a retrain.
# ---------------------------------------------------------------------------
_PATTERN_TRACKER_FILE = os.path.join(DATA_DIR, 'pattern_failure_tracker.json')
_pattern_tracker_cache = {}
_pattern_tracker_last_load = 0


def _load_pattern_tracker():
    global _pattern_tracker_cache, _pattern_tracker_last_load
    now = time.time()
    if now - _pattern_tracker_last_load < 30:
        return _pattern_tracker_cache
    try:
        if os.path.exists(_PATTERN_TRACKER_FILE):
            with open(_PATTERN_TRACKER_FILE, 'r') as f:
                _pattern_tracker_cache = json.load(f)
    except Exception:
        _pattern_tracker_cache = {}
    _pattern_tracker_last_load = now
    return _pattern_tracker_cache


def _save_pattern_tracker(tracker):
    global _pattern_tracker_cache
    try:
        tmp = _PATTERN_TRACKER_FILE + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(tracker, f)
        os.replace(tmp, _PATTERN_TRACKER_FILE)
        _pattern_tracker_cache = tracker
    except Exception:
        pass


def update_pattern_tracker(rows):
    """Update the pattern-failure tracker from a list of history rows.
    Tracks win/loss per 2-gram, 3-gram, and 4-gram sequences.
    """
    verified = [
        r for r in (rows or [])
        if r.get('status') in ('WIN', 'LOSS')
        and r.get('prediction') in ('BIG', 'SMALL')
        and r.get('actual') in ('BIG', 'SMALL')
    ]
    if len(verified) < 10:
        return
    tracker = _load_pattern_tracker()
    cats = [1 if r.get('actual') == 'BIG' else 0 for r in verified]
    statuses = [r.get('status') for r in verified]
    for gram in (2, 3, 4):
        for i in range(gram, len(cats)):
            key = ''.join('B' if c == 1 else 'S' for c in cats[i - gram:i])
            outcome = statuses[i]  # WIN or LOSS of the prediction at position i
            if outcome not in ('WIN', 'LOSS'):
                continue
            entry = tracker.setdefault(key, {'wins': 0, 'losses': 0})
            if outcome == 'WIN':
                entry['wins'] += 1
            else:
                entry['losses'] += 1
    # Trim tracker to prevent unbounded growth (keep patterns seen ≥2 times)
    tracker = {k: v for k, v in tracker.items() if v['wins'] + v['losses'] >= 2}
    _save_pattern_tracker(tracker)


def get_pattern_analysis(rows, n=30):
    """Return a structured analysis of which patterns are failing most.
    Returns top failing patterns, top winning patterns, trend regime, and
    loss concentration stats. Exposed via /v2/ml/patterns API.
    """
    verified = [
        r for r in (rows or [])
        if r.get('status') in ('WIN', 'LOSS')
        and r.get('prediction') in ('BIG', 'SMALL')
        and r.get('actual') in ('BIG', 'SMALL')
    ]
    recent = list(reversed(verified[-n:]))
    cats = [1 if r.get('actual') == 'BIG' else 0 for r in verified]
    
    # --- Trend regime (bull=BIG dominant, bear=SMALL dominant) ---
    recent_cats = cats[-20:] if len(cats) >= 20 else cats
    long_cats   = cats[-100:] if len(cats) >= 100 else cats
    recent_big_rate = sum(recent_cats) / max(len(recent_cats), 1)
    long_big_rate   = sum(long_cats)   / max(len(long_cats),   1)
    regime = 'neutral'
    if recent_big_rate > 0.62:
        regime = 'BIG_DOMINANT'
    elif recent_big_rate < 0.38:
        regime = 'SMALL_DOMINANT'
    elif recent_big_rate > long_big_rate + 0.08:
        regime = 'SHIFTING_TO_BIG'
    elif recent_big_rate < long_big_rate - 0.08:
        regime = 'SHIFTING_TO_SMALL'
    
    # --- Win/Loss streaks ---
    current_streak_type = None
    current_streak_len  = 0
    for r in recent:
        s = r.get('status')
        if current_streak_type is None:
            current_streak_type = s
            current_streak_len  = 1
        elif s == current_streak_type:
            current_streak_len += 1
        else:
            break
    
    # --- Where are losses concentrated? ---
    big_pred_total  = sum(1 for r in recent if r.get('prediction') == 'BIG')
    sml_pred_total  = sum(1 for r in recent if r.get('prediction') == 'SMALL')
    big_losses      = sum(1 for r in recent if r.get('prediction') == 'BIG'   and r.get('status') == 'LOSS')
    sml_losses      = sum(1 for r in recent if r.get('prediction') == 'SMALL' and r.get('status') == 'LOSS')
    big_loss_pct    = round(big_losses / max(big_pred_total, 1) * 100, 1)
    sml_loss_pct    = round(sml_losses / max(sml_pred_total, 1) * 100, 1)
    
    # --- Pattern failure from tracker ---
    tracker = _load_pattern_tracker()
    pattern_list = []
    for key, v in tracker.items():
        total = v['wins'] + v['losses']
        if total < 3:
            continue
        loss_rate = v['losses'] / total
        pattern_list.append({
            'pattern': key,
            'wins':    v['wins'],
            'losses':  v['losses'],
            'total':   total,
            'lossRate': round(loss_rate * 100, 1),
            'winRate':  round((1 - loss_rate) * 100, 1),
        })
    pattern_list.sort(key=lambda x: x['lossRate'], reverse=True)
    top_failing  = pattern_list[:8]
    top_winning  = sorted(pattern_list, key=lambda x: x['winRate'], reverse=True)[:8]
    
    return {
        'totalVerified':      len(verified),
        'recentChecked':      len(recent),
        'regime':             regime,
        'recentBigRate':      round(recent_big_rate * 100, 1),
        'longBigRate':        round(long_big_rate   * 100, 1),
        'currentStreak': {
            'type':   current_streak_type,
            'length': current_streak_len,
        },
        'lossConcentration': {
            'BIG_predLoss':   big_loss_pct,
            'SMALL_predLoss': sml_loss_pct,
            'worstSide':      'BIG' if big_loss_pct > sml_loss_pct else 'SMALL' if sml_loss_pct > big_loss_pct else 'BALANCED',
        },
        'topFailingPatterns': top_failing,
        'topWinningPatterns': top_winning,
    }


def _default_meta():
    return {
        'lastTrainCount': 0,
        'lastTrainTime': 0,
        'totalTrainCycles': 0,
        'accuracyHistory': [],
        'modelVersion': MODEL_VERSION
    }


def _load_meta():
    try:
        if os.path.exists(_RETRAIN_META_FILE):
            with open(_RETRAIN_META_FILE, 'r') as f:
                meta = json.load(f)
            meta.setdefault('lastTrainCount', 0)
            meta.setdefault('lastTrainTime', 0)
            meta.setdefault('totalTrainCycles', 0)
            meta.setdefault('accuracyHistory', [])
            meta.setdefault('modelVersion', 1)
            return meta
    except Exception:
        pass
    return _default_meta()


def _save_meta(meta):
    try:
        tmp = _RETRAIN_META_FILE + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(meta, f, indent=2)
        os.replace(tmp, _RETRAIN_META_FILE)
    except Exception:
        pass


def _encode_category(cat):
    return 1 if cat == 'BIG' else 0


def _decode_category(val):
    return 'BIG' if val >= 0.5 else 'SMALL'


def _safe_number(value):
    try:
        number = int(float(value))
        return number if 0 <= number <= 9 else None
    except Exception:
        return None


def _ratio(count, total, default=0.0):
    return count / total if total else default


def _get_historic_data(all_predictions):
    by_period = {}
    
    # 1. Base input predictions (passed from memory)
    for p in all_predictions or []:
        period = str(p.get('period') or '')
        if period and p.get('actual') in ('BIG', 'SMALL') and p.get('status') in ('WIN', 'LOSS'):
            by_period[period] = {
                'period': period,
                'prediction': p.get('prediction') or 'BIG',
                'status': p.get('status'),
                'confidence': float(p.get('confidence') or 100),
                'actual': p.get('actual'),
                'number': p.get('number'),
                'patternUsed': p.get('patternUsed') or 'ensemble',
                'timestamp': int(float(p.get('timestamp') or time.time()))
            }

    # 2. Main prediction history csv files
    paths = (
        (os.path.join(DATA_DIR, 'predict', 'prediction_history.csv'), 'v2_predict_csv'),
        (os.path.join(DATA_DIR, 'free', 'free_prediction_history.csv'), 'v2_free_csv'),
        (os.path.join(DATA_DIR, 'model', 'model_prediction_history.csv'), 'model_predict_csv'),
    )
    for path, source in paths:
        if os.path.exists(path):
            try:
                with open(path, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        period = str(row.get('period') or '')
                        actual = row.get('actual')
                        status = row.get('status') or 'Pending'
                        if period and actual in ('BIG', 'SMALL') and status in ('WIN', 'LOSS'):
                            by_period.setdefault(period, {
                                'period': period,
                                'prediction': row.get('prediction') or actual,
                                'status': status,
                                'confidence': float(row.get('confidence') or 100),
                                'actual': actual,
                                'number': row.get('number'),
                                'patternUsed': row.get('patternused') or row.get('patternUsed') or source,
                                'timestamp': int(float(row.get('timestamp') or time.time()))
                            })
            except Exception:
                pass

    # 3. Daily draws API history
    from config import DAILY_1K_HISTORY_CSV
    if os.path.exists(DAILY_1K_HISTORY_CSV):
        try:
            with open(DAILY_1K_HISTORY_CSV, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    category = row.get('category')
                    period = str(row.get('period') or row.get('issueNumber') or '')
                    if period and category in ('BIG', 'SMALL'):
                        by_period.setdefault(period, {
                            'period': period,
                            'prediction': category,
                            'status': 'WIN',
                            'confidence': 100.0,
                            'actual': category,
                            'number': row.get('number'),
                            'patternUsed': 'daily_1k_history',
                            'timestamp': int(float(row.get('timestamp') or time.time()))
                        })
        except Exception:
            pass

    # Sort in ascending chronological order for feature extraction windows
    verified = list(by_period.values())
    verified.sort(key=lambda x: int(str(x.get('period') or '0')[-12:] or 0))
    return verified




def extract_features(predictions_slice):
    feats = []
    cats = [1 if p.get('category') == 'BIG' or p.get('actual') == 'BIG' else 0 for p in predictions_slice]
    if not cats:
        return None

    n = len(cats)

    # ── 1. Raw last-10 one-hot ─────────────────────────────────────────────────
    for i in range(10):
        feats.append(cats[i] if i < n else 0.5)

    # ── 2. Multi-window BIG ratio ──────────────────────────────────────────────
    windows = [5, 10, 20, 50]
    for w in windows:
        chunk = cats[:min(w, n)]
        feats.append(sum(chunk) / max(len(chunk), 1))

    # ── 3. Fresh-vs-older market pressure momentum ─────────────────────────────
    for fresh, older in ((3, 10), (5, 10), (10, 20), (20, 50)):
        fresh_chunk = cats[:min(fresh, n)]
        older_chunk = cats[min(fresh, n):min(older, n)]
        fresh_ratio = sum(fresh_chunk) / max(len(fresh_chunk), 1)
        older_ratio = sum(older_chunk) / max(len(older_chunk), 1) if older_chunk else 0.5
        feats.append(fresh_ratio - older_ratio)   # positive = BIG trend accelerating

    # ── 4. Trend regime: is the market trending or reversing? ─────────────────
    recent5  = sum(cats[:min(5,  n)]) / max(min(5,  n), 1)
    recent20 = sum(cats[:min(20, n)]) / max(min(20, n), 1)
    recent50 = sum(cats[:min(50, n)]) / max(min(50, n), 1)
    feats.append(recent5  - recent20)   # short-term vs mid momentum
    feats.append(recent20 - recent50)   # mid vs long momentum
    feats.append(1.0 if recent5 > 0.60 else 0.0)   # BIG dominant now
    feats.append(1.0 if recent5 < 0.40 else 0.0)   # SMALL dominant now
    feats.append(1.0 if recent5 > recent50 + 0.10 else 0.0)  # shifting to BIG
    feats.append(1.0 if recent5 < recent50 - 0.10 else 0.0)  # shifting to SMALL

    # ── 5. Current streak (length + direction) ────────────────────────────────
    streak = 1
    for i in range(1, min(12, n)):
        if cats[i] == cats[i - 1]:
            streak += 1
        else:
            break
    streak_dir = cats[0] if n > 0 else 0
    feats.append(streak / 12.0)
    feats.append(float(streak_dir))
    feats.append(1.0 if streak_dir == 1 else 0.0)  # currently in BIG streak
    feats.append(1.0 if streak_dir == 0 else 0.0)  # currently in SMALL streak
    feats.append(1.0 if streak >= 3 and streak_dir == 1 else 0.0)  # long BIG streak
    feats.append(1.0 if streak >= 3 and streak_dir == 0 else 0.0)  # long SMALL streak
    feats.append(1.0 if streak >= 5 else 0.0)  # very long streak (reversal risk)

    # ── 6. Run-length encoding: lengths of the last 6 runs ─────────────────────
    # Captures the rhythm: e.g. 3×BIG, 1×SMALL, 4×BIG → [3,1,4,...]
    run_lengths = []
    run_dirs    = []
    if n > 0:
        cur_dir = cats[0]
        cur_len = 1
        for i in range(1, min(80, n)):
            if cats[i] == cur_dir:
                cur_len += 1
            else:
                run_lengths.append(cur_len)
                run_dirs.append(cur_dir)
                cur_dir = cats[i]
                cur_len = 1
        run_lengths.append(cur_len)
        run_dirs.append(cur_dir)
    for j in range(6):
        feats.append(min(run_lengths[j], 10) / 10.0 if j < len(run_lengths) else 0.0)
        feats.append(float(run_dirs[j]) if j < len(run_dirs) else 0.5)
    avg_run = sum(run_lengths) / max(len(run_lengths), 1)
    feats.append(min(avg_run, 8) / 8.0)  # average run length

    # ── 7. Number distribution features ───────────────────────────────────────
    nums = []
    for p in predictions_slice[:80]:
        number = _safe_number(p.get('number'))
        if number is not None:
            nums.append(number)
    for d in range(10):
        feats.append(nums.count(d) / max(len(nums), 1))

    recent_nums = nums[:20]
    older_nums  = nums[20:60]
    for group in (recent_nums, older_nums):
        feats.append(float(np.mean(group) / 9.0) if group else 0.5)
        feats.append(float(np.std(group) / 4.5)  if group else 0.0)
        feats.append(_ratio(sum(1 for num in group if num >= 5), len(group), 0.5))
        feats.append(_ratio(sum(1 for num in group if num % 2), len(group), 0.5))
    if recent_nums:
        last_num = recent_nums[0]
        feats.append(last_num / 9.0)
        feats.append(_ratio(recent_nums.count(last_num), len(recent_nums), 0.0))
    else:
        feats.extend([0.5, 0.0])

    # ── 8. Zigzag / alternation features ──────────────────────────────────────
    zigzag = 0
    if n >= 5:
        zz = True
        for i in range(1, 5):
            if cats[i] == cats[i - 1]:
                zz = False
                break
        zigzag = 1 if zz else 0
    feats.append(zigzag)

    alt_count = 0
    for i in range(1, min(20, n)):
        if cats[i] != cats[i - 1]:
            alt_count += 1
    feats.append(alt_count / max(min(20, n) - 1, 1))

    # ── 9. First-order transition matrix ──────────────────────────────────────
    transitions = [0, 0, 0, 0]  # SS, SB, BS, BB
    transition_total = 0
    for i in range(1, min(30, n)):
        previous = cats[i]
        current  = cats[i - 1]
        transitions[(previous * 2) + current] += 1
        transition_total += 1
    feats.extend(v / max(transition_total, 1) for v in transitions)

    # ── 10. Second-order transition matrix ────────────────────────────────────
    second_order = [0] * 8
    second_total = 0
    for i in range(2, min(40, n)):
        state = (cats[i] * 4) + (cats[i - 1] * 2) + cats[i - 2]
        second_order[state] += 1
        second_total += 1
    feats.extend(v / max(second_total, 1) for v in second_order)

    # ── 11. Shannon entropy of recent 20 ─────────────────────────────────────
    big_ratio = sum(cats[:min(20, n)]) / max(min(20, n), 1)
    if big_ratio in (0, 1):
        entropy = 0.0
    else:
        entropy = -(
            big_ratio * np.log2(big_ratio)
            + (1 - big_ratio) * np.log2(1 - big_ratio)
        )
    feats.append(float(entropy))

    # ── 12. Volatility / balance markers ──────────────────────────────────────
    for w in (6, 12, 24):
        chunk = cats[:min(w, n)]
        if len(chunk) >= 2:
            alternations = sum(1 for i in range(1, len(chunk)) if chunk[i] != chunk[i - 1])
            feats.append(alternations / (len(chunk) - 1))
            ratio = sum(chunk) / len(chunk)
            feats.append(1.0 - abs(ratio - 0.5) * 2)
        else:
            feats.extend([0.5, 1.0])

    # ── 13. Own prediction W/L features ──────────────────────────────────────
    settled = [p for p in predictions_slice if p.get('status') in ('WIN', 'LOSS')]
    for w in (5, 10, 20):
        chunk      = settled[:min(w, len(settled))]
        losses     = sum(1 for p in chunk if p.get('status') == 'LOSS')
        wins       = sum(1 for p in chunk if p.get('status') == 'WIN')
        big_losses = sum(1 for p in chunk if p.get('status') == 'LOSS' and p.get('prediction') == 'BIG')
        sml_losses = sum(1 for p in chunk if p.get('status') == 'LOSS' and p.get('prediction') == 'SMALL')
        feats.extend([
            _ratio(wins,       len(chunk), 0.5),
            _ratio(losses,     len(chunk), 0.0),
            _ratio(big_losses, max(losses, 1), 0.0),
            _ratio(sml_losses, max(losses, 1), 0.0),
        ])

    # ── 14. Consecutive win/loss streak in own predictions ────────────────────
    cons_loss = 0
    cons_win  = 0
    for p in settled:
        if p.get('status') == 'LOSS' and cons_win == 0:
            cons_loss += 1
        elif p.get('status') == 'WIN' and cons_loss == 0:
            cons_win += 1
        else:
            break
    feats.append(min(cons_loss, 8) / 8.0)
    feats.append(min(cons_win,  8) / 8.0)

    # ── 15. Per-side loss rate from pattern failure tracker ───────────────────
    # Build key from last 2-gram and 3-gram actual sequence
    tracker = _load_pattern_tracker()
    for gram in (2, 3, 4):
        if n >= gram:
            key = ''.join('B' if c == 1 else 'S' for c in cats[:gram])
            entry = tracker.get(key, {})
            total = entry.get('wins', 0) + entry.get('losses', 0)
            if total > 0:
                feats.append(entry.get('losses', 0) / total)   # pattern loss rate
                feats.append(entry.get('wins',   0) / total)   # pattern win rate
                feats.append(min(total, 50) / 50.0)            # confidence in estimate
            else:
                feats.extend([0.5, 0.5, 0.0])
        else:
            feats.extend([0.5, 0.5, 0.0])

    # ── 16. HOT/COLD pattern flags ────────────────────────────────────────────
    # Is the current 2/3-gram among the top-failing or top-winning patterns?
    hot_threshold  = 0.65  # win rate ≥ 65% → HOT
    cold_threshold = 0.60  # loss rate ≥ 60% → COLD
    is_hot  = 0.0
    is_cold = 0.0
    for gram in (2, 3):
        if n >= gram:
            key   = ''.join('B' if c == 1 else 'S' for c in cats[:gram])
            entry = tracker.get(key, {})
            total = entry.get('wins', 0) + entry.get('losses', 0)
            if total >= 3:
                win_r  = entry.get('wins',   0) / total
                loss_r = entry.get('losses', 0) / total
                if win_r  >= hot_threshold:
                    is_hot  = 1.0
                if loss_r >= cold_threshold:
                    is_cold = 1.0
    feats.append(is_hot)
    feats.append(is_cold)

    # ── 17. Market volatility: std dev of recent BIG counts in rolling-5 ──────
    roll5_big = []
    for start in range(0, min(40, n) - 4, 1):
        chunk5 = cats[start:start + 5]
        roll5_big.append(sum(chunk5) / 5.0)
    feats.append(float(np.std(roll5_big)) if len(roll5_big) >= 2 else 0.0)

    return np.array(feats, dtype=np.float32)


def build_training_data(all_predictions):
    verified = _get_historic_data(all_predictions)
    if len(verified) < 15:
        return None, None, 0

    X, y = [], []
    for i in range(5, len(verified)):
        # Only use outcomes that existed before this target period.
        past = list(reversed(verified[:i]))
        feats = extract_features(past)
        if feats is not None:
            target = _encode_category(verified[i].get('actual', 'SMALL'))
            X.append(feats)
            y.append(target)

    if len(X) < 10:
        return None, None, 0

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), len(X)


def should_retrain(all_predictions):
    """Check if retrain is needed — count threshold OR 4-hour schedule."""
    meta = _load_meta()
    verified_count = len(_get_historic_data(all_predictions))
    now = int(time.time())
    version_ready = meta.get('modelVersion', 1) < MODEL_VERSION
    model_missing = not os.path.exists(MODEL_FILE)
    count_ready = verified_count - meta['lastTrainCount'] >= _retrain_threshold
    time_ready = (now - meta['lastTrainTime']) >= _retrain_interval and meta['lastTrainTime'] > 0
    return version_ready or model_missing or count_ready or time_ready


import threading

_training_thread = None
_training_lock = threading.Lock()
_TRAINING_LOCK_FILE = os.path.join(DATA_DIR, 'ml_training.lock')


def train_model(all_predictions, force=False):
    global _model, _accuracy_cache, _training_thread
    
    if os.path.exists(_TRAINING_LOCK_FILE):
        try:
            mtime = os.path.getmtime(_TRAINING_LOCK_FILE)
            if time.time() - mtime < 600:
                return False
        except Exception:
            pass

    with _training_lock:
        if _training_thread is not None and _training_thread.is_alive():
            return False

        if not force and not should_retrain(all_predictions):
            return False

        def _target():
            global _model, _accuracy_cache
            try:
                try:
                    with open(_TRAINING_LOCK_FILE, 'w') as lf:
                        lf.write(str(os.getpid()))
                except Exception:
                    pass

                meta = _load_meta()
                verified_count = len(_get_historic_data(all_predictions))
                X, y, count = build_training_data(all_predictions)
                if X is None or count < 10:
                    return

                validation_accuracy = None
                validation_samples = 0
                if SKLEARN_AVAILABLE and len(np.unique(y)) > 1:
                    split = max(10, int(len(X) * 0.8))
                    split = min(split, len(X) - 5)
                    X_train, X_valid = X[:split], X[split:]
                    y_train, y_valid = y[:split], y[split:]

                    candidates = []
                    if LIGHTGBM_AVAILABLE:
                        candidates.append(
                            LGBMClassifier(
                                n_estimators=600,
                                learning_rate=0.025,
                                max_depth=12,
                                num_leaves=64,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                min_child_samples=5,
                                reg_alpha=0.1,
                                reg_lambda=0.3,
                                min_split_gain=0.01,
                                random_state=44,
                                objective='binary',
                                verbosity=-1,
                            )
                        )
                    if XGBOOST_AVAILABLE:
                        candidates.append(
                            XGBClassifier(
                                n_estimators=600,
                                learning_rate=0.025,
                                max_depth=8,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                colsample_bylevel=0.8,
                                min_child_weight=1,
                                reg_alpha=0.1,
                                reg_lambda=0.5,
                                gamma=0.05,
                                random_state=45,
                                eval_metric='logloss',
                                objective='binary:logistic',
                                n_jobs=1,
                            )
                        )
                    if CATBOOST_AVAILABLE:
                        candidates.append(
                            CatBoostClassifier(
                                iterations=800,
                                learning_rate=0.025,
                                depth=8,
                                l2_leaf_reg=3,
                                min_data_in_leaf=3,
                                subsample=0.8,
                                random_seed=46,
                                verbose=0,
                                loss_function='Logloss',
                                eval_metric='Accuracy',
                                early_stopping_rounds=50,
                                allow_writing_files=False,
                            )
                        )
                    candidates += [
                        RandomForestClassifier(
                            n_estimators=300,
                            max_depth=8,
                            min_samples_leaf=4,
                            random_state=42,
                            n_jobs=1,
                            class_weight='balanced_subsample',
                        ),
                        ExtraTreesClassifier(
                            n_estimators=300,
                            max_depth=10,
                            min_samples_leaf=3,
                            random_state=43,
                            n_jobs=1,
                            class_weight='balanced',
                        ),
                        LogisticRegression(
                            C=0.5,
                            max_iter=1000,
                            class_weight='balanced',
                            random_state=42,
                        ),
                    ]
                    weights = []
                    validation = {}
                    for candidate in candidates:
                        candidate.fit(X_train, y_train)
                        score = accuracy_score(y_valid, candidate.predict(X_valid))
                        name = candidate.__class__.__name__
                        validation[name] = round(float(score) * 100, 2)
                        weights.append(max(float(score) - 0.45, 0.05))
                        candidate.fit(X, y)
                    local_model = _WeightedEnsemble(candidates, weights, count)
                    meta['validationAccuracy'] = validation
                    validation_accuracy = round(
                        sum(validation.values()) / len(validation), 2
                    )
                    validation_samples = len(y_valid)
                    meta['validationSamples'] = validation_samples
                else:
                    local_model = _SimpleModel()
                    local_model.fit(X, y)

                _model = local_model

                meta['lastTrainCount'] = verified_count
                meta['lastTrainTime'] = int(time.time())
                meta['totalTrainCycles'] += 1
                meta['modelVersion'] = MODEL_VERSION

                if validation_accuracy is not None:
                    acc = {
                        'accuracy': validation_accuracy,
                        'recentAccuracy': validation_accuracy,
                        'total': validation_samples,
                    }
                else:
                    acc = get_ml_accuracy(all_predictions)
                meta['accuracyHistory'].append({
                    'timestamp': meta['lastTrainTime'],
                    'accuracy': acc['accuracy'],
                    'recentAccuracy': acc['recentAccuracy'],
                    'samples': acc['total'],
                    'dataCount': verified_count,
                    'trainCycles': meta['totalTrainCycles'],
                })
                if len(meta['accuracyHistory']) > 365:
                    meta['accuracyHistory'] = meta['accuracyHistory'][-365:]

                _accuracy_cache = {'accuracy': 50, 'recentAccuracy': 50, 'total': 0, 'cached_at': 0, 'data_count': 0}
                _save_model(meta)
                _save_meta(meta)
                print(f"  [ML] Trained in background: {count} samples, acc={acc['accuracy']}%, recent={acc['recentAccuracy']}%, cycle={meta['totalTrainCycles']}")
            except Exception as e:
                print(f"  [ML] Background training error: {e}")
            finally:
                try:
                    if os.path.exists(_TRAINING_LOCK_FILE):
                        os.remove(_TRAINING_LOCK_FILE)
                except Exception:
                    pass

        _training_thread = threading.Thread(target=_target, daemon=True)
        _training_thread.start()
        return True


def predict_ml(all_predictions, current_predictions_slice):
    global _model
    if _model is None:
        if not _load_model():
            if len(current_predictions_slice) >= 5:
                cats = [
                    1 if p.get('category') == 'BIG' or p.get('actual') == 'BIG' else 0
                    for p in current_predictions_slice[:20]
                ]
                if cats:
                    big_r = sum(cats) / len(cats)
                    pred = 'BIG' if big_r > 0.5 else 'SMALL'
                    conf = calibrate_confidence(55 + abs(big_r - 0.5) * 20)
                    return {
                        'prediction': pred,
                        'confidence': round(conf, 1),
                        'bigProbability': round(big_r * 100, 1),
                        'mlScore': round(conf, 1),
                        'samples': 0,
                        'selectedModel': 'FallbackRatio',
                        'selectedModelAccuracy': None,
                        'modelPredictions': [],
                    }
            return None

    feats = extract_features(current_predictions_slice)
    if feats is None:
        return None

    feats = feats.reshape(1, -1)
    big_prob = 0.5  # default

    if SKLEARN_AVAILABLE and hasattr(_model, 'predict_proba'):
        proba = _model.predict_proba(feats)[0]
        big_prob = proba[1] if len(proba) > 1 else proba[0]
        pred = _decode_category(big_prob)
        conf = calibrate_confidence(min(50 + abs(big_prob - 0.5) * 100, 92))
    elif hasattr(_model, 'predict_proba'):
        proba = _model.predict_proba(feats)[0]
        big_prob = proba[1] if len(proba) > 1 else proba[0]
        pred = _decode_category(big_prob)
        conf = calibrate_confidence(min(50 + abs(big_prob - 0.5) * 100, 92))
    else:
        pred_val = _model.predict(feats)[0]
        pred = _decode_category(pred_val)
        conf = calibrate_confidence(65)

    samples = _model.get_sample_count() if hasattr(_model, 'get_sample_count') else (
        _last_train_count if _last_train_count > 0 else 0
    )
    model_predictions = []
    selected_model = 'ensemble'
    selected_accuracy = None
    if hasattr(_model, 'model_details'):
        validation = _load_meta().get('validationAccuracy', {})
        model_predictions = _model.model_details(feats, validation)
        if model_predictions:
            selected = max(
                model_predictions,
                key=lambda item: (
                    item.get('validationAccuracy') if item.get('validationAccuracy') is not None else -1,
                    item.get('confidence', 0),
                ),
            )
            if selected.get('validationAccuracy') is not None:
                selected_model = selected['model']
                selected_accuracy = selected.get('validationAccuracy')

    return {
        'prediction': pred,
        'confidence': round(conf, 1),
        'bigProbability': round(float(big_prob) * 100, 1) if isinstance(big_prob, (int, float)) else 50,
        'mlScore': round(conf * (1.2 if conf > 70 else 1.0), 1),
        'samples': samples,
        'selectedModel': selected_model,
        'selectedModelAccuracy': selected_accuracy,
        'modelPredictions': model_predictions,
    }


def get_ml_accuracy(all_predictions):
    global _accuracy_cache
    verified = _get_historic_data(all_predictions)
    data_count = len(verified)
    now = time.time()

    if data_count < 10:
        _accuracy_cache = {
            'accuracy': 50, 'recentAccuracy': 50, 'total': 0,
            'cached_at': now, 'data_count': data_count
        }
        return _accuracy_cache

    if (
        _accuracy_cache['data_count'] == data_count
        and now - _accuracy_cache['cached_at'] < 300
        and _accuracy_cache['total'] > 0
    ):
        return _accuracy_cache

    if _model is None:
        _accuracy_cache = {
            'accuracy': 50, 'recentAccuracy': 50, 'total': data_count,
            'cached_at': now, 'data_count': data_count
        }
        return _accuracy_cache

    meta = _load_meta()
    validation = meta.get('validationAccuracy', {})
    validation_samples = meta.get('validationSamples', 0)
    if validation and validation_samples:
        validation_accuracy = round(
            sum(validation.values()) / len(validation), 2
        )
        _accuracy_cache = {
            'accuracy': validation_accuracy,
            'recentAccuracy': validation_accuracy,
            'total': validation_samples,
            'cached_at': now,
            'data_count': data_count,
        }
        return _accuracy_cache

    correct = 0
    recent_correct = 0
    total = 0
    max_check = min(data_count, 500)

    chronological = verified[:max_check]
    for i in range(5, len(chronological)):
        past = list(reversed(chronological[:i]))
        feats = extract_features(past)
        if feats is None:
            continue
        feats = feats.reshape(1, -1)

        if SKLEARN_AVAILABLE and hasattr(_model, 'predict'):
            p = _model.predict(feats)[0]
            pred = _decode_category(p)
        elif hasattr(_model, 'predict'):
            p = _model.predict(feats)[0]
            pred = _decode_category(p)
        else:
            pred = 'SMALL'

        actual = chronological[i].get('actual', 'SMALL')
        if pred == actual:
            correct += 1
            if total < 20:
                recent_correct += 1
        total += 1

    _accuracy_cache = {
        'accuracy': round((correct / total) * 100, 2) if total > 0 else 50,
        'recentAccuracy': round((recent_correct / min(total, 20)) * 100, 2) if total > 0 else 50,
        'total': total,
        'cached_at': now,
        'data_count': data_count,
    }
    return _accuracy_cache


def calibrate_confidence(raw_confidence):
    """Adjust confidence based on model's historical accuracy."""
    meta = _load_meta()
    hist = meta.get('accuracyHistory', [])
    if not hist:
        return raw_confidence
    last_acc = hist[-1].get('accuracy', 50)
    # confidence <= actual accuracy + 15%, clamped to [55, 90]
    cap = min(max(last_acc + 15, 55), 90)
    return min(raw_confidence, cap)


def _csv_training_rows(path, source):
    if not os.path.exists(path):
        return []
    rows = []
    try:
        with open(path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                actual = (row.get('actual') or '').upper()
                status = (row.get('status') or '').upper()
                period = str(row.get('period') or '')
                if not period or status not in ('WIN', 'LOSS') or actual not in ('BIG', 'SMALL'):
                    continue
                rows.append({
                    'period': period,
                    'prediction': (row.get('prediction') or '').upper() or None,
                    'status': status,
                    'confidence': float(row.get('confidence') or 0),
                    'actual': actual,
                    'number': row.get('number') or None,
                    'patternUsed': row.get('patternused') or source,
                    'timestamp': int(float(row.get('timestamp') or time.time())),
                    'sourceRoute': source,
                })
    except Exception:
        return []
    return rows



def _daily_1k_training_rows(path):
    """Read daily_1k_history.csv which uses 'category' not 'actual'."""
    if not os.path.exists(path):
        return []
    rows = []
    try:
        with open(path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                category = (row.get('category') or '').upper()
                period = str(row.get('period') or '')
                if not period or category not in ('BIG', 'SMALL'):
                    continue
                rows.append({
                    'period': period,
                    'prediction': None,
                    'status': 'TRAINING',
                    'confidence': 0,
                    'actual': category,
                    'number': row.get('number') or None,
                    'patternUsed': row.get('patternUsed') or 'daily_1k_history',
                    'timestamp': int(float(row.get('timestamp') or time.time())),
                    'sourceRoute': 'daily_1k_csv',
                })
    except Exception:
        return []
    return rows


def build_sequence_training_rows(base_rows=None, market_rows=None):
    by_period = {}
    for row in _get_historic_data(base_rows or []):
        period = str(row.get('period') or '')
        if period and row.get('actual') in ('BIG', 'SMALL'):
            item = dict(row)
            item['sourceRoute'] = item.get('sourceRoute') or 'v2_predict_memory'
            by_period[period] = item
    paths = (
        (os.path.join(DATA_DIR, 'predict', 'prediction_history.csv'), 'v2_predict_csv'),
        (os.path.join(DATA_DIR, 'free', 'free_prediction_history.csv'), 'v2_free_csv'),
        (os.path.join(DATA_DIR, 'model', 'model_prediction_history.csv'), 'model_predict_csv'),
    )
    for path, source in paths:
        for row in _csv_training_rows(path, source):
            by_period.setdefault(str(row.get('period')), row)
    # --- Also learn from daily_1k_history.csv (data/1m/) ---
    for row in _daily_1k_training_rows(os.path.join(DATA_DIR, '1m', 'daily_1k_history.csv')):
        by_period.setdefault(str(row.get('period')), row)
    for row in market_rows or []:
        period = str(row.get('period') or '')
        actual = row.get('actual') or row.get('category')
        if period and actual in ('BIG', 'SMALL'):
            by_period.setdefault(period, {
                'period': period,
                'prediction': None,
                'status': 'TRAINING',
                'confidence': 0,
                'actual': actual,
                'number': row.get('number'),
                'patternUsed': row.get('patternUsed') or 'daily_1k_history',
                'timestamp': int(row.get('timestamp') or time.time()),
                'sourceRoute': 'daily_1k_history',
            })
    rows = list(by_period.values())
    rows.sort(key=lambda row: int(str(row.get('period') or '0')[-12:] or 0))
    return rows


def _sequence_prob(rows, recent_latest_first, bidirectional=False):
    cats = [1 if row.get('actual') == 'BIG' else 0 for row in rows if row.get('actual') in ('BIG', 'SMALL')]
    recent = [
        1 if item.get('category') == 'BIG' or item.get('actual') == 'BIG' else 0
        for item in recent_latest_first
        if item.get('category') in ('BIG', 'SMALL') or item.get('actual') in ('BIG', 'SMALL')
    ]
    if len(cats) < 50 or len(recent) < 3:
        return 0.5
    votes = []
    weights = []
    for n in (16, 12, 8, 5, 3, 2, 1):
        if len(recent) < n or len(cats) <= n:
            continue
        target_state = tuple(reversed(recent[:n]))
        big = small = 0
        for i in range(n, len(cats)):
            state = tuple(cats[i - n:i])
            if state == target_state:
                if cats[i] == 1:
                    big += 1
                else:
                    small += 1
        total = big + small
        if total:
            votes.append((big + 1) / (total + 2))
            weights.append(min(total, 40) * (1 + n / 16))
        if bidirectional and len(recent) >= n:
            reverse_state = tuple(recent[:n])
            big = small = 0
            rev_cats = list(reversed(cats))
            for i in range(n, len(rev_cats)):
                state = tuple(rev_cats[i - n:i])
                if state == reverse_state:
                    if rev_cats[i] == 1:
                        big += 1
                    else:
                        small += 1
            total = big + small
            if total:
                votes.append((big + 1) / (total + 2))
                weights.append(min(total, 30) * (1 + n / 20))
    if not votes:
        recent_big = sum(recent[:20]) / max(min(len(recent), 20), 1)
        long_big = sum(cats[-200:]) / max(min(len(cats), 200), 1)
        return (recent_big * 0.65) + (long_big * 0.35)
    return float(np.average(votes, weights=weights))


def _sequence_validation(rows, bidirectional=False):
    if len(rows) < LSTM_MIN_HISTORY:
        return None
    start = max(60, len(rows) - 80)
    correct = total = 0
    for idx in range(start, len(rows)):
        if (idx - start) % 2:
            continue
        prior = rows[max(0, idx - 1200):idx]
        recent = list(reversed(prior[-80:]))
        prob = _sequence_prob(prior, recent, bidirectional=bidirectional)
        pred = 'BIG' if prob >= 0.5 else 'SMALL'
        if pred == rows[idx].get('actual'):
            correct += 1
        total += 1
    return round((correct / total) * 100, 2) if total else None


def _loss_learning_profile(rows):
    verified = [
        row for row in rows
        if row.get('status') in ('WIN', 'LOSS')
        and row.get('prediction') in ('BIG', 'SMALL')
        and row.get('actual') in ('BIG', 'SMALL')
    ]
    recent = list(reversed(verified[-80:]))
    profile = {
        'totalVerified': len(verified),
        'recentChecked': len(recent),
        'sideLossRate': {'BIG': 0, 'SMALL': 0},
        'sideAccuracy': {'BIG': 50, 'SMALL': 50},
        'consecutiveLosses': 0,
        'inverseTrapSide': None,
        'inverseTrapStrength': 0,
        'lossPressure': 0,
        'adjustment': 0,
        'patternAdjustment': 0,
        'reason': 'not_enough_verified_prediction_loss_data',
    }
    if not recent:
        return profile

    # ── Per-side WIN/LOSS rate over recent 80 ────────────────────────────────
    for side in ('BIG', 'SMALL'):
        side_rows = [row for row in recent if row.get('prediction') == side]
        losses = sum(1 for row in side_rows if row.get('status') == 'LOSS')
        wins   = sum(1 for row in side_rows if row.get('status') == 'WIN')
        total  = wins + losses
        if total:
            profile['sideLossRate'][side]  = round((losses / total) * 100, 2)
            profile['sideAccuracy'][side]  = round((wins   / total) * 100, 2)

    # ── Consecutive loss streak + inverse trap analysis ───────────────────────
    consecutive_losses = 0
    inverse_counts = {'BIG': 0, 'SMALL': 0}
    recent_actuals_in_loss = []
    for row in recent:
        if row.get('status') != 'LOSS':
            break
        consecutive_losses += 1
        pred   = row.get('prediction')
        actual = row.get('actual')
        if pred in ('BIG', 'SMALL') and actual in ('BIG', 'SMALL') and pred != actual:
            inverse_counts[actual] += 1
            recent_actuals_in_loss.append(actual)
    profile['consecutiveLosses'] = consecutive_losses

    trap_side     = max(inverse_counts, key=inverse_counts.get)
    trap_strength = inverse_counts[trap_side]
    if trap_strength >= 1:
        profile['inverseTrapSide']     = trap_side
        profile['inverseTrapStrength'] = trap_strength

    # ── Market actual majority during loss streak ─────────────────────────────
    market_big_in_loss   = recent_actuals_in_loss.count('BIG')
    market_small_in_loss = recent_actuals_in_loss.count('SMALL')

    # ── Pattern-specific loss adjustment from tracker ─────────────────────────
    # If the current sequence pattern has historically been a loser, add extra
    # adjustment toward the winning side.
    tracker = _load_pattern_tracker()
    pattern_adj = 0.0
    actuals_seq = [1 if r.get('actual') == 'BIG' else 0 for r in recent]
    for gram in (2, 3, 4):
        if len(actuals_seq) >= gram:
            key   = ''.join('B' if c == 1 else 'S' for c in actuals_seq[:gram])
            entry = tracker.get(key, {})
            total = entry.get('wins', 0) + entry.get('losses', 0)
            if total >= 4:
                loss_rate = entry.get('losses', 0) / total
                win_rate  = entry.get('wins',   0) / total
                # Losing pattern: push away from it (adjust towards the winning opposite)
                if loss_rate > 0.60:
                    # This gram pattern tends to lose for predictions → flip bias
                    latest_pred = recent[0].get('prediction') if recent else None
                    if latest_pred == 'BIG':
                        pattern_adj -= (loss_rate - 0.50) * 0.30  # favour SMALL
                    elif latest_pred == 'SMALL':
                        pattern_adj += (loss_rate - 0.50) * 0.30  # favour BIG
                elif win_rate > 0.65:
                    # Winning pattern: reinforce current direction
                    latest_pred = recent[0].get('prediction') if recent else None
                    if latest_pred == 'BIG':
                        pattern_adj += (win_rate - 0.50) * 0.20
                    elif latest_pred == 'SMALL':
                        pattern_adj -= (win_rate - 0.50) * 0.20
    pattern_adj = max(-0.25, min(0.25, pattern_adj))
    profile['patternAdjustment'] = round(pattern_adj, 4)

    # ── Compute probability adjustment ───────────────────────────────────────
    big_loss   = profile['sideLossRate']['BIG']
    small_loss = profile['sideLossRate']['SMALL']
    loss_gap   = small_loss - big_loss   # positive = SMALL is losing more → favor BIG
    adjustment = 0.0

    # Base adjustment from side-loss imbalance
    if abs(loss_gap) >= 8:
        adjustment += max(-0.18, min(0.18, loss_gap / 100 * 0.6))

    # Inverse trap: the side that kept coming in actuals → boost that side
    if profile['inverseTrapSide'] == 'BIG':
        adjustment += min(0.22, 0.07 * trap_strength)
    elif profile['inverseTrapSide'] == 'SMALL':
        adjustment -= min(0.22, 0.07 * trap_strength)

    # Pattern-specific correction
    adjustment += pattern_adj * 0.6

    # During consecutive loss streak: scale adjustment by streak depth
    if consecutive_losses >= 1:
        scale = 1 + consecutive_losses * 0.18
        adjustment = adjustment * min(scale, 2.8)

        # If market was consistently outputting one side during losses, follow it
        if consecutive_losses >= 2 and recent_actuals_in_loss:
            market_bias = (market_big_in_loss - market_small_in_loss) / len(recent_actuals_in_loss)
            adjustment += market_bias * 0.15 * min(consecutive_losses, 4)

    # Clamp to [-0.45, +0.45]
    adjustment = max(-0.45, min(0.45, adjustment))
    profile['adjustment']   = round(adjustment, 4)
    profile['lossPressure'] = round(max(big_loss, small_loss), 2)
    profile['reason'] = (
        'loss_aware_probability_adjustment'
        if adjustment else
        'loss_profile_observed_no_adjustment_needed'
    )
    # Also update pattern tracker while we are here (side-effect, low-cost)
    update_pattern_tracker(rows)
    return profile


def _apply_loss_learning(probability, loss_profile):
    """Apply the learned loss correction to the raw probability.
    Uses a tanh-scaled curve so small adjustments have gentle effect
    but large adjustments (deep loss streaks) genuinely shift the output.
    """
    adjustment = float((loss_profile or {}).get('adjustment') or 0)
    if not adjustment:
        return probability
    # tanh scaling: adjustment of 0.40 → full ~0.38 shift; 0.18 → ~0.18 shift
    import math
    scaled = math.tanh(adjustment * 2.0) * 0.45
    return max(0.03, min(0.97, probability + scaled))


def predict_lstm_bilstm(all_predictions, current_predictions_slice, market_rows=None):
    rows = build_sequence_training_rows(all_predictions, market_rows)
    source_counts = {}
    for row in rows:
        source = row.get('sourceRoute') or 'unknown'
        source_counts[source] = source_counts.get(source, 0) + 1
    if len(rows) < LSTM_MIN_HISTORY:
        return {
            'ready': False,
            'prediction': None,
            'confidence': 0,
            'samples': len(rows),
            'requiredSamples': LSTM_MIN_HISTORY,
            'selectedModel': None,
            'modelPredictions': [],
            'sourceCounts': source_counts,
            'reason': f'LSTM/Bi-LSTM need {LSTM_MIN_HISTORY} history rows.',
        }
    loss_profile = _loss_learning_profile(rows)
    lstm_raw_prob = _sequence_prob(rows, current_predictions_slice, bidirectional=False)
    bilstm_raw_prob = _sequence_prob(rows, current_predictions_slice, bidirectional=True)
    lstm_prob = _apply_loss_learning(lstm_raw_prob, loss_profile)
    bilstm_prob = _apply_loss_learning(bilstm_raw_prob, loss_profile)
    lstm_acc = _sequence_validation(rows, bidirectional=False)
    bilstm_acc = _sequence_validation(rows, bidirectional=True)
    details = []
    for name, prob, acc in (
        ('LSTMSequenceModel', lstm_prob, lstm_acc),
        ('BiLSTMSequenceModel', bilstm_prob, bilstm_acc),
    ):
        confidence = min(95, max(55, 50 + abs(prob - 0.5) * 100))
        details.append({
            'model': name,
            'prediction': _decode_category(prob),
            'confidence': round(confidence, 2),
            'bigProbability': round(prob * 100, 2),
            'rawBigProbability': round((lstm_raw_prob if name == 'LSTMSequenceModel' else bilstm_raw_prob) * 100, 2),
            'validationAccuracy': acc,
            'lossAdjustment': loss_profile.get('adjustment', 0),
        })
    selected = max(details, key=lambda row: ((row.get('validationAccuracy') or 0), row.get('confidence') or 0))
    return {
        'ready': True,
        'prediction': selected['prediction'],
        'confidence': selected['confidence'],
        'bigProbability': selected['bigProbability'],
        'samples': len(rows),
        'requiredSamples': LSTM_MIN_HISTORY,
        'selectedModel': selected['model'],
        'selectedModelAccuracy': selected.get('validationAccuracy'),
        'modelPredictions': details,
        'sourceCounts': source_counts,
        'lossLearning': loss_profile,
        'mode': 'LSTM_BILSTM_ONLY',
    }


def _save_model(meta=None):
    if _model is None:
        return
    import pickle
    try:
        tmp = MODEL_FILE + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(_model, f)
        os.replace(tmp, MODEL_FILE)
    except Exception as e:
        print(f"  [ML] Error pickling model: {e}")
    if meta is None:
        meta = _load_meta()
    meta['sklearn'] = SKLEARN_AVAILABLE
    meta['timestamp'] = int(time.time())
    meta['samples'] = _model.get_sample_count() if hasattr(_model, 'get_sample_count') else meta.get('lastTrainCount', 0)
    try:
        tmp = ML_METADATA_FILE + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(meta, f, indent=2)
        os.replace(tmp, ML_METADATA_FILE)
    except Exception:
        pass


def _load_model():
    global _model
    if not os.path.exists(MODEL_FILE):
        return False
    try:
        meta = _load_meta()
        if meta.get('modelVersion', 1) < MODEL_VERSION:
            return False
        import pickle
        with open(MODEL_FILE, 'rb') as f:
            _model = pickle.load(f)
        return True
    except Exception:
        return False


def get_accuracy_history():
    """Return accuracy history for monitoring."""
    meta = _load_meta()
    return meta.get('accuracyHistory', [])


def get_model_summary():
    """Return human-readable model summary."""
    meta = _load_meta()
    hist = meta.get('accuracyHistory', [])
    last = hist[-1] if hist else {}
    return {
        'totalTrainCycles': meta['totalTrainCycles'],
        'lastTrainTime': meta['lastTrainTime'],
        'lastTrainCount': meta['lastTrainCount'],
        'lastAccuracy': last.get('accuracy', 50),
        'lastRecentAccuracy': last.get('recentAccuracy', 50),
        'totalSamples': last.get('samples', 0),
        'historyLength': len(hist),
        'sklearn': SKLEARN_AVAILABLE,
        'modelVersion': meta.get('modelVersion', 1),
        'validationAccuracy': meta.get('validationAccuracy', {}),
        'validationSamples': meta.get('validationSamples', 0),
        'models': [
            *(['LGBMClassifier'] if LIGHTGBM_AVAILABLE else []),
            *(['XGBClassifier'] if XGBOOST_AVAILABLE else []),
            *(['CatBoostClassifier'] if CATBOOST_AVAILABLE else []),
            'RandomForestClassifier',
            'ExtraTreesClassifier',
            'LogisticRegression',
        ]
        if meta.get('modelVersion', 1) >= 4 and SKLEARN_AVAILABLE else ['SimpleModel'],
        'lightgbmAvailable': LIGHTGBM_AVAILABLE,
        'xgboostAvailable': XGBOOST_AVAILABLE,
        'catboostAvailable': CATBOOST_AVAILABLE,
    }


class _WeightedEnsemble:
    def __init__(self, models, weights, sample_count):
        self.models = models
        self.weights = np.array(weights, dtype=np.float64)
        self.sample_count = sample_count

    def predict_proba(self, X):
        probabilities = []
        for model in self.models:
            proba = model.predict_proba(X)
            if proba.shape[1] == 1:
                cls = int(model.classes_[0])
                big_prob = np.ones(len(X)) if cls == 1 else np.zeros(len(X))
                proba = np.column_stack((1 - big_prob, big_prob))
            probabilities.append(proba)
        normalized = self.weights / max(self.weights.sum(), 1e-9)
        combined = sum(
            weight * proba
            for weight, proba in zip(normalized, probabilities)
        )
        return combined

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int32)

    def get_sample_count(self):
        return self.sample_count

    def model_details(self, X, validation_accuracy=None):
        validation_accuracy = validation_accuracy or {}
        details = []
        normalized = self.weights / max(self.weights.sum(), 1e-9)
        for model, weight in zip(self.models, normalized):
            name = model.__class__.__name__
            proba = model.predict_proba(X)[0]
            if len(proba) == 1:
                cls = int(model.classes_[0])
                big_prob = 1.0 if cls == 1 else 0.0
            else:
                big_prob = float(proba[1])
            confidence = min(50 + abs(big_prob - 0.5) * 100, 92)
            details.append({
                'model': name,
                'prediction': _decode_category(big_prob),
                'confidence': round(confidence, 1),
                'bigProbability': round(big_prob * 100, 1),
                'validationAccuracy': validation_accuracy.get(name),
                'ensembleWeight': round(float(weight), 4),
            })
        return details


class _SimpleModel:
    def __init__(self):
        self.weights = None
        self.bias = 0.0
        self.n_features = 0
        self.sample_count = 0

    def fit(self, X, y):
        self.n_features = X.shape[1]
        n = X.shape[0]
        self.weights = np.zeros(self.n_features, dtype=np.float32)
        self.bias = 0.0
        lr = 0.01
        epochs = 50
        for epoch in range(epochs):
            idxs = np.random.permutation(n)
            for idx in idxs:
                x = X[idx]
                target = y[idx]
                linear = np.dot(x, self.weights) + self.bias
                prob = 1 / (1 + np.exp(-linear))
                err = target - prob
                self.weights += lr * err * x
                self.bias += lr * err
        self.sample_count = n

    def predict_proba(self, X):
        linear = np.dot(X, self.weights) + self.bias
        prob = 1 / (1 + np.exp(-linear))
        return np.column_stack((1 - prob, prob))

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(np.int32)

    def get_sample_count(self):
        return self.sample_count
