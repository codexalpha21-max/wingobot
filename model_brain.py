import os
import time
import json
import pickle
import csv
import threading
from collections import defaultdict, Counter

from config import DATA_DIR

BRAIN_DIR = os.path.join(DATA_DIR, 'model_brain')
os.makedirs(BRAIN_DIR, exist_ok=True)

MODEL_NAMES = ['ensemble', 'ml', 'lstm', 'bilstm', 'autolearn',
               'pattern', 'transition', 'number', 'sequence', 'loss', 'brain']

_brain_lock = threading.RLock()
_brain_cache = {}
_recent_actuals = []


def _model_folder(model_name):
    folder = os.path.join(BRAIN_DIR, model_name)
    os.makedirs(folder, exist_ok=True)
    return folder


def _model_knowledge_path(model_name):
    return os.path.join(_model_folder(model_name), 'knowledge.pkl')


def _model_meta_path(model_name):
    return os.path.join(_model_folder(model_name), 'meta.pkl')


def _save_pkl(path, data):
    tmp = path + '.tmp'
    try:
        with open(tmp, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        return True
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        return False


def _load_pkl(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


def _collect_all_data():
    verified = []
    sources = [
        os.path.join(DATA_DIR, '1m', 'daily_1k_history.csv'),
        os.path.join(DATA_DIR, 'predict', 'predictions.csv'),
        os.path.join(DATA_DIR, 'predict', 'prediction_history.csv'),
        os.path.join(DATA_DIR, 'model', 'model_prediction_history.csv'),
    ]
    seen_periods = set()
    for path in sources:
        if not os.path.exists(path):
            continue
        try:
            with open(path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    period = str(row.get('period') or '').strip()
                    actual = row.get('actual') or row.get('category') or ''
                    actual = actual.upper().strip()
                    prediction = row.get('prediction') or ''
                    prediction = prediction.upper().strip()
                    status = (row.get('status') or '').upper().strip()
                    if not period or period in seen_periods:
                        continue
                    if actual not in ('BIG', 'SMALL'):
                        continue
                    number = row.get('number')
                    try:
                        number = int(float(number))
                    except (ValueError, TypeError):
                        number = None
                    seen_periods.add(period)
                    verified.append({
                        'period': period,
                        'prediction': prediction if prediction in ('BIG', 'SMALL') else None,
                        'actual': actual, 'number': number,
                        'status': status if status in ('WIN', 'LOSS') else ('WIN' if actual in ('BIG', 'SMALL') else 'PENDING'),
                        'timestamp': int(float(row.get('timestamp') or time.time())),
                    })
        except Exception:
            pass
    verified.sort(key=lambda x: int(str(x.get('period') or '0')[-12:] or 0))
    return verified


def _learn_pattern_knowledge(data):
    cats = [d['actual'] for d in data if d['actual'] in ('BIG', 'SMALL')]
    if len(cats) < 10:
        return {}
    n = len(cats)
    big_streaks, small_streaks = [], []
    current, side = 0, None
    for c in cats:
        if c == 'BIG':
            if side == 'BIG': current += 1
            else:
                if side == 'SMALL' and current > 0: small_streaks.append(current)
                current, side = 1, 'BIG'
        else:
            if side == 'SMALL': current += 1
            else:
                if side == 'BIG' and current > 0: big_streaks.append(current)
                current, side = 1, 'SMALL'
    if side == 'BIG' and current > 0: big_streaks.append(current)
    elif side == 'SMALL' and current > 0: small_streaks.append(current)
    return {
        'total': n, 'bigCount': cats.count('BIG'), 'smallCount': cats.count('SMALL'),
        'bigRatio': round(cats.count('BIG') / n, 4),
        'smallRatio': round(cats.count('SMALL') / n, 4),
        'bigStreaks': big_streaks[-20:], 'smallStreaks': small_streaks[-20:],
        'avgBigStreak': round(sum(big_streaks) / max(len(big_streaks), 1), 2) if big_streaks else 0,
        'avgSmallStreak': round(sum(small_streaks) / max(len(small_streaks), 1), 2) if small_streaks else 0,
        'longestBigStreak': max(big_streaks) if big_streaks else 0,
        'longestSmallStreak': max(small_streaks) if small_streaks else 0,
    }


def _learn_transition_knowledge(data):
    trans = defaultdict(lambda: {'BIG': 0, 'SMALL': 0})
    cats = [d['actual'] for d in data if d['actual'] in ('BIG', 'SMALL')]
    for i in range(len(cats) - 1):
        trans[cats[i]][cats[i + 1]] += 1
    return {k: {'toBig': round(v['BIG'] / max(sum(v.values()), 1), 4),
                'toSmall': round(v['SMALL'] / max(sum(v.values()), 1), 4),
                'total': sum(v.values())} for k, v in trans.items()}


def _learn_number_knowledge(data):
    counts, after_big, after_small = Counter(), Counter(), Counter()
    for i in range(len(data)):
        d = data[i]; n = d.get('number')
        if n is not None and d['actual'] in ('BIG', 'SMALL'):
            counts[n] += 1
            if i > 0:
                prev = data[i - 1]['actual']
                if prev == 'BIG': after_big[n] += 1
                elif prev == 'SMALL': after_small[n] += 1
    return {'counts': dict(counts.most_common()), 'mostCommon': counts.most_common(3) if counts else [],
            'afterBig': dict(after_big.most_common(5)) if after_big else {},
            'afterSmall': dict(after_small.most_common(5)) if after_small else {}}


def _learn_sequence_knowledge(data):
    seqs, cats = {}, [d['actual'] for d in data if d['actual'] in ('BIG', 'SMALL')]
    for length in [2, 3, 4]:
        seq_counts = defaultdict(lambda: {'BIG': 0, 'SMALL': 0})
        for i in range(len(cats) - length):
            key = tuple(cats[i:i + length])
            nxt = cats[i + length] if i + length < len(cats) else None
            if nxt: seq_counts[key][nxt] += 1
        learned = {}
        for seq, outcomes in seq_counts.items():
            total = outcomes['BIG'] + outcomes['SMALL']
            if total >= 2:
                learned['_'.join(seq)] = {'toBig': round(outcomes['BIG'] / total, 4),
                                          'toSmall': round(outcomes['SMALL'] / total, 4), 'total': total}
        if learned: seqs[f'seq{length}'] = learned
    return seqs


def _learn_loss_knowledge(data):
    li = {'totalLosses': 0, 'bigToSmallLosses': 0, 'smallToBigLosses': 0, 'afterLossResults': []}
    for i in range(len(data) - 1):
        curr, nxt = data[i], data[i + 1]
        if curr.get('prediction') and curr['actual'] in ('BIG', 'SMALL') and curr['status'] == 'LOSS':
            li['totalLosses'] += 1
            if curr['prediction'] == 'BIG' and curr['actual'] == 'SMALL': li['bigToSmallLosses'] += 1
            elif curr['prediction'] == 'SMALL' and curr['actual'] == 'BIG': li['smallToBigLosses'] += 1
            li['afterLossResults'].append(nxt['actual'])
    entries = len(li['afterLossResults'])
    if entries > 0:
        big_after = li['afterLossResults'].count('BIG')
        small_after = li['afterLossResults'].count('SMALL')
        li['bigAfterLoss'] = big_after; li['smallAfterLoss'] = small_after
        li['bigAfterLossRatio'] = round(big_after / entries, 4)
        li['smallAfterLossRatio'] = round(small_after / entries, 4)
    return li


def _model_predictions_from_csv(csv_path, model_name_filter=None):
    rows = []
    if not os.path.exists(csv_path):
        return rows
    try:
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pu = (row.get('patternused') or row.get('patternUsed') or '').strip()
                if model_name_filter and pu != model_name_filter:
                    continue
                status = (row.get('status') or '').strip().upper()
                prediction = (row.get('prediction') or '').strip().upper()
                actual = (row.get('actual') or row.get('category') or '').strip().upper()
                number = row.get('number')
                try:
                    number = int(float(number))
                except (ValueError, TypeError):
                    number = None
                if actual in ('BIG', 'SMALL'):
                    rows.append({
                        'prediction': prediction if prediction in ('BIG', 'SMALL') else None,
                        'actual': actual,
                        'status': status if status in ('WIN', 'LOSS') else ('WIN' if prediction == actual else 'LOSS'),
                        'number': number,
                    })
    except Exception:
        pass
    return rows


def _learn_model_stats_from_rows(rows):
    if not rows:
        return {'totalPredictions': 0, 'wins': 0, 'losses': 0, 'winRate': 0, 'status': 'no_data'}
    wins = sum(1 for r in rows if r['status'] == 'WIN')
    losses = sum(1 for r in rows if r['status'] == 'LOSS')
    total = wins + losses
    win_rate = round(wins / total, 4) if total > 0 else 0
    big_preds = sum(1 for r in rows if r.get('prediction') == 'BIG')
    small_preds = sum(1 for r in rows if r.get('prediction') == 'SMALL')
    big_wins = sum(1 for r in rows if r.get('prediction') == 'BIG' and r['status'] == 'WIN')
    small_wins = sum(1 for r in rows if r.get('prediction') == 'SMALL' and r['status'] == 'WIN')
    streak = 0
    for r in reversed(rows):
        if r['status'] == 'WIN':
            streak += 1
        else:
            break
    recent = rows[-30:] if len(rows) >= 30 else rows
    recent_wins = sum(1 for r in recent if r['status'] == 'WIN')
    recent_rate = round(recent_wins / max(len(recent), 1), 4)
    return {
        'totalPredictions': total,
        'wins': wins,
        'losses': losses,
        'winRate': win_rate,
        'recentWinRate': recent_rate,
        'currentStreak': streak,
        'bigPredictions': big_preds,
        'smallPredictions': small_preds,
        'bigWins': big_wins,
        'smallWins': small_wins,
        'status': 'active' if total > 0 else 'no_data',
    }


def _learn_ensemble_knowledge():
    rows = _model_predictions_from_csv(
        os.path.join(DATA_DIR, 'predict', 'prediction_history.csv'),
        model_name_filter='ensemble'
    )
    return _learn_model_stats_from_rows(rows)


def _learn_ml_knowledge():
    csv_path = os.path.join(DATA_DIR, 'model', 'model_prediction_history.csv')
    sub_models = {}
    all_rows = []
    if not os.path.exists(csv_path):
        return _learn_model_stats_from_rows([])
    try:
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pu = (row.get('patternused') or '').strip()
                if pu in ('BiLSTMSequenceModel',):
                    continue
                status = (row.get('status') or '').strip().upper()
                prediction = (row.get('prediction') or '').strip().upper()
                actual = (row.get('actual') or row.get('category') or '').strip().upper()
                number = row.get('number')
                try:
                    number = int(float(number))
                except (ValueError, TypeError):
                    number = None
                if actual not in ('BIG', 'SMALL'):
                    continue
                r = {
                    'prediction': prediction if prediction in ('BIG', 'SMALL') else None,
                    'actual': actual,
                    'status': status if status in ('WIN', 'LOSS') else ('WIN' if prediction == actual else 'LOSS'),
                    'number': number,
                }
                all_rows.append(r)
                if pu not in sub_models:
                    sub_models[pu] = []
                sub_models[pu].append(r)
    except Exception:
        pass
    base = _learn_model_stats_from_rows(all_rows)
    sub_stats = {}
    for name, srows in sub_models.items():
        sub_stats[name] = _learn_model_stats_from_rows(srows)
    base['subModels'] = sub_stats
    return base


def _learn_bilstm_knowledge():
    rows = _model_predictions_from_csv(
        os.path.join(DATA_DIR, 'model', 'model_prediction_history.csv'),
        model_name_filter='BiLSTMSequenceModel'
    )
    return _learn_model_stats_from_rows(rows)


def _learn_lstm_knowledge():
    return None


def _learn_autolearn_knowledge():
    return None


def _learn_inverse_patterns(data):
    """Learn from past pattern outcomes — the brain analyzes what usually comes next
    after different market patterns and stores the learned prediction."""
    cats = [d['actual'] for d in data if d['actual'] in ('BIG', 'SMALL')]
    if len(cats) < 10:
        return {}
    learned = {'alternating': {}, 'consecutive': {}, 'reversal': {}}
    # Alternating pattern: after a run of alternating BIG/SMALL, what happens?
    i = 0
    while i < len(cats) - 3:
        if cats[i] != cats[i+1]:
            alt_len = 2
            while i + alt_len < len(cats) and cats[i + alt_len] != cats[i + alt_len - 1]:
                alt_len += 1
            if alt_len >= 3:
                key = f"alt_{alt_len}"
                if key not in learned['alternating']:
                    learned['alternating'][key] = {'big': 0, 'small': 0, 'total': 0}
                nxt = cats[i + alt_len] if i + alt_len < len(cats) else None
                if nxt:
                    learned['alternating'][key][nxt.lower()] += 1
                    learned['alternating'][key]['total'] += 1
                i += alt_len
                continue
        i += 1
    # Consecutive pattern: after a run of same side (BIG BIG BIG), what happens?
    i = 0
    while i < len(cats) - 2:
        if cats[i] == cats[i+1]:
            same_len = 2
            while i + same_len < len(cats) and cats[i + same_len] == cats[i + same_len - 1]:
                same_len += 1
            if same_len >= 2:
                key = f"cons_{cats[i]}_{same_len}"
                if key not in learned['consecutive']:
                    learned['consecutive'][key] = {'big': 0, 'small': 0, 'total': 0}
                nxt = cats[i + same_len] if i + same_len < len(cats) else None
                if nxt:
                    learned['consecutive'][key][nxt.lower()] += 1
                    learned['consecutive'][key]['total'] += 1
                i += same_len
                continue
        i += 1
    # Reversal pattern: after a SINGLE flip (e.g. BIG, SMALL or SMALL, BIG),
    # what does the market do? Continue the new side or flip back?
    for i in range(len(cats) - 2):
        if cats[i] != cats[i+1]:
            key = f"rev_{cats[i]}_to_{cats[i+1]}"
            if key not in learned['reversal']:
                learned['reversal'][key] = {'big': 0, 'small': 0, 'total': 0}
            nxt = cats[i + 2] if i + 2 < len(cats) else None
            if nxt:
                learned['reversal'][key][nxt.lower()] += 1
                learned['reversal'][key]['total'] += 1
    # Compute learnedPrediction for each pattern
    for category in learned:
        for key, stats in learned[category].items():
            t = stats['total']
            stats['bigRate'] = round(stats['big'] / t, 4) if t else 0
            stats['smallRate'] = round(stats['small'] / t, 4) if t else 0
            stats['confidence'] = round(abs(stats['bigRate'] - 0.5) * 200, 2) if t >= 3 else 0
            if stats['big'] > stats['small'] and t >= 2:
                stats['learnedPrediction'] = 'BIG'
            elif stats['small'] > stats['big'] and t >= 2:
                stats['learnedPrediction'] = 'SMALL'
            else:
                stats['learnedPrediction'] = None
    return learned


def learn_all():
    data = _collect_all_data()
    if len(data) < 10:
        return False
    now = int(time.time())
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())
    meta = {'lastLearned': timestamp, 'totalRows': len(data), 'timestamp': now}
    raw_map = {
        'pattern': _learn_pattern_knowledge(data),
        'transition': _learn_transition_knowledge(data),
        'number': _learn_number_knowledge(data),
        'sequence': _learn_sequence_knowledge(data),
        'loss': _learn_loss_knowledge(data),
        'brain': {**meta, **_learn_inverse_patterns(data)},
        'ensemble': _learn_ensemble_knowledge(),
        'ml': _learn_ml_knowledge(),
        'lstm': _learn_lstm_knowledge(),
        'bilstm': _learn_bilstm_knowledge(),
        'autolearn': _learn_autolearn_knowledge(),
    }
    knowledge_map = {k: v for k, v in raw_map.items() if v is not None}
    with _brain_lock:
        global _brain_cache
        ok = True
        for model_name in MODEL_NAMES:
            if model_name in knowledge_map:
                ok &= _save_pkl(_model_knowledge_path(model_name), knowledge_map[model_name])
                ok &= _save_pkl(_model_meta_path(model_name), meta)
                _brain_cache[model_name] = knowledge_map[model_name]
        return ok


def load_brain():
    with _brain_lock:
        global _brain_cache
        if not _brain_cache:
            for model_name in MODEL_NAMES:
                k = _load_pkl(_model_knowledge_path(model_name)) or {}
                _brain_cache[model_name] = k
        return _brain_cache


def get_model_knowledge(model_name):
    return load_brain().get(model_name, {})


def _warm_recent_actuals():
    global _recent_actuals
    if _recent_actuals:
        return
    data = _collect_all_data()
    cats = [d['actual'] for d in data if d['actual'] in ('BIG', 'SMALL')]
    _recent_actuals = cats[-200:] if len(cats) > 200 else cats


def brain_learn_from_result(actual):
    if actual not in ('BIG', 'SMALL'):
        return False
    with _brain_lock:
        global _brain_cache, _recent_actuals
        _warm_recent_actuals()
        _recent_actuals.append(actual)
        if len(_recent_actuals) > 500:
            _recent_actuals = _recent_actuals[-500:]
        if len(_recent_actuals) < 4:
            return False
        if 'brain' not in _brain_cache:
            load_brain()
        brain_data = _brain_cache.get('brain', {})
        if not isinstance(brain_data, dict):
            brain_data = {}
        for section in ('alternating', 'consecutive', 'reversal'):
            if section not in brain_data:
                brain_data[section] = {}
        cats = _recent_actuals
        prev = cats[:-1]
        last_actual = actual
        updated = False

        # Alternating pattern
        alt_len = 0
        if len(prev) >= 2 and prev[-1] != prev[-2]:
            alt_len = 2
            for j in range(len(prev) - 3, -1, -1):
                if prev[j] != prev[j + 1]:
                    alt_len += 1
                else:
                    break
        if alt_len >= 3:
            key = f"alt_{alt_len}"
            if key not in brain_data['alternating']:
                brain_data['alternating'][key] = {'big': 0, 'small': 0, 'total': 0}
            brain_data['alternating'][key][last_actual.lower()] += 1
            brain_data['alternating'][key]['total'] += 1
            t = brain_data['alternating'][key]['total']
            brain_data['alternating'][key]['bigRate'] = round(brain_data['alternating'][key]['big'] / t, 4) if t else 0
            brain_data['alternating'][key]['smallRate'] = round(brain_data['alternating'][key]['small'] / t, 4) if t else 0
            brain_data['alternating'][key]['confidence'] = round(abs(brain_data['alternating'][key]['bigRate'] - 0.5) * 200, 2) if t >= 3 else 0
            if brain_data['alternating'][key]['big'] > brain_data['alternating'][key]['small'] and t >= 2:
                brain_data['alternating'][key]['learnedPrediction'] = 'BIG'
            elif brain_data['alternating'][key]['small'] > brain_data['alternating'][key]['big'] and t >= 2:
                brain_data['alternating'][key]['learnedPrediction'] = 'SMALL'
            else:
                brain_data['alternating'][key]['learnedPrediction'] = None
            updated = True

        # Consecutive pattern
        cons_len = 0
        cons_side = None
        if len(prev) >= 2 and prev[-1] == prev[-2]:
            cons_len = 2
            cons_side = prev[-1]
            for j in range(len(prev) - 3, -1, -1):
                if prev[j] == prev[j + 1]:
                    cons_len += 1
                else:
                    break
        if cons_len >= 2 and cons_side:
            key = f"cons_{cons_side}_{cons_len}"
            if key not in brain_data['consecutive']:
                brain_data['consecutive'][key] = {'big': 0, 'small': 0, 'total': 0}
            brain_data['consecutive'][key][last_actual.lower()] += 1
            brain_data['consecutive'][key]['total'] += 1
            t = brain_data['consecutive'][key]['total']
            brain_data['consecutive'][key]['bigRate'] = round(brain_data['consecutive'][key]['big'] / t, 4) if t else 0
            brain_data['consecutive'][key]['smallRate'] = round(brain_data['consecutive'][key]['small'] / t, 4) if t else 0
            brain_data['consecutive'][key]['confidence'] = round(abs(brain_data['consecutive'][key]['bigRate'] - 0.5) * 200, 2) if t >= 3 else 0
            if brain_data['consecutive'][key]['big'] > brain_data['consecutive'][key]['small'] and t >= 2:
                brain_data['consecutive'][key]['learnedPrediction'] = 'BIG'
            elif brain_data['consecutive'][key]['small'] > brain_data['consecutive'][key]['big'] and t >= 2:
                brain_data['consecutive'][key]['learnedPrediction'] = 'SMALL'
            else:
                brain_data['consecutive'][key]['learnedPrediction'] = None
            updated = True

        # Reversal pattern
        if len(prev) >= 2:
            last_two = prev[-2:]
            if last_two[0] != last_two[1]:
                key = f"rev_{last_two[0]}_to_{last_two[1]}"
                if key not in brain_data['reversal']:
                    brain_data['reversal'][key] = {'big': 0, 'small': 0, 'total': 0}
                brain_data['reversal'][key][last_actual.lower()] += 1
                brain_data['reversal'][key]['total'] += 1
                t = brain_data['reversal'][key]['total']
                brain_data['reversal'][key]['bigRate'] = round(brain_data['reversal'][key]['big'] / t, 4) if t else 0
                brain_data['reversal'][key]['smallRate'] = round(brain_data['reversal'][key]['small'] / t, 4) if t else 0
                brain_data['reversal'][key]['confidence'] = round(abs(brain_data['reversal'][key]['bigRate'] - 0.5) * 200, 2) if t >= 3 else 0
                if brain_data['reversal'][key]['big'] > brain_data['reversal'][key]['small'] and t >= 2:
                    brain_data['reversal'][key]['learnedPrediction'] = 'BIG'
                elif brain_data['reversal'][key]['small'] > brain_data['reversal'][key]['big'] and t >= 2:
                    brain_data['reversal'][key]['learnedPrediction'] = 'SMALL'
                else:
                    brain_data['reversal'][key]['learnedPrediction'] = None
                updated = True

        if updated:
            _brain_cache['brain'] = brain_data
            _save_pkl(_model_knowledge_path('brain'), brain_data)
        return updated


def brain_think(current_prediction, current_confidence, consecutive_losses, last_prediction, last_actual, recent_results):
    brain = load_brain()
    patterns = brain.get('pattern', {})
    transitions = brain.get('transition', {})
    loss_info = brain.get('loss', {})
    sequence_info = brain.get('sequence', {})

    seq_score = {'BIG': 0.0, 'SMALL': 0.0}
    if recent_results and len(recent_results) >= 2:
        for length in [2, 3, 4]:
            if len(recent_results) >= length:
                key = '_'.join(recent_results[:length])
                sd = (sequence_info.get(f'seq{length}', {}) or {}).get(key)
                if sd:
                    seq_score['BIG'] += sd.get('toBig', 0) * (1.5 ** length)
                    seq_score['SMALL'] += sd.get('toSmall', 0) * (1.5 ** length)

    trans_score = {'BIG': 0.0, 'SMALL': 0.0}
    if recent_results and len(recent_results) > 0 and recent_results[0] in transitions:
        td = transitions[recent_results[0]]
        trans_score['BIG'] = td.get('toBig', 0.5)
        trans_score['SMALL'] = td.get('toSmall', 0.5)

    big_edge = patterns.get('bigRatio', 0.5)
    loss_score = {'BIG': 0.0, 'SMALL': 0.0}
    if consecutive_losses > 0:
        bal = loss_info.get('bigAfterLossRatio', 0.5)
        sal = loss_info.get('smallAfterLossRatio', 0.5)
        loss_score['BIG'] = bal * (1 + 0.2 * consecutive_losses)
        loss_score['SMALL'] = sal * (1 + 0.2 * consecutive_losses)
        if last_prediction and last_actual and last_prediction != last_actual:
            loss_score[last_actual] += 0.5 * consecutive_losses

    combined = {'BIG': 0.0, 'SMALL': 0.0}
    combined['BIG'] = big_edge * 15 + seq_score['BIG'] * 25 + trans_score['BIG'] * 20 + loss_score['BIG'] * 25 + (current_confidence / 100) * 15
    combined['SMALL'] = (1 - big_edge) * 15 + seq_score['SMALL'] * 25 + trans_score['SMALL'] * 20 + loss_score['SMALL'] * 25 + (1 - current_confidence / 100) * 15

    total_c = combined['BIG'] + combined['SMALL']
    big_prob = combined['BIG'] / max(total_c, 0.001)
    prediction = 'BIG' if big_prob >= 0.5 else 'SMALL'
    confidence = min(94, max(60, 55 + abs(big_prob - 0.5) * 80))

    if consecutive_losses >= 1:
        confidence = min(94, max(confidence, 75))
        if last_prediction and last_actual and last_prediction != last_actual:
            ba = loss_info.get('bigAfterLoss', 0); sa = loss_info.get('smallAfterLoss', 0)
            if ba > sa and last_actual == 'BIG':
                prediction, confidence = 'BIG', min(94, confidence + 5)
            elif sa > ba and last_actual == 'SMALL':
                prediction, confidence = 'SMALL', min(94, confidence + 5)
    if consecutive_losses >= 2:
        confidence = min(94, max(confidence, 80))
        if seq_score[prediction] < 0.1 and trans_score.get(prediction, 0) < 0.4:
            alt = 'SMALL' if prediction == 'BIG' else 'BIG'
            if seq_score[alt] > seq_score[prediction] or trans_score.get(alt, 0) > trans_score.get(prediction, 0):
                prediction, confidence = alt, min(94, confidence + 3)

    return prediction, round(confidence, 2), {
        'bigProbability': round(big_prob * 100, 2),
        'seqScore': {k: round(v, 4) for k, v in seq_score.items()},
        'transScore': {k: round(v, 4) for k, v in trans_score.items()},
        'lossScore': {k: round(v, 4) for k, v in loss_score.items()},
        'combined': {k: round(v, 4) for k, v in combined.items()},
        'edge': round(abs(big_prob - 0.5) * 2, 4),
    }
