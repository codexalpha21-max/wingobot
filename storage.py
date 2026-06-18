import os
import csv
import json
import time
import threading

from config import *
from helpers import build_default_user_state

_file_locks = {}
PREDICTION_HISTORY_BACKUP_CSV = PREDICTION_HISTORY_CSV + '.backup'


def _get_lock(name):
    if name not in _file_locks:
        _file_locks[name] = threading.Lock()
    return _file_locks[name]


def _atomic_write_csv(filepath, header, rows):
    if os.path.dirname(filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    tmp = f"{filepath}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        with open(tmp, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, filepath)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _load_merged_history_rows(header):
    current_header = header
    by_period = {}
    for filepath in (PREDICTION_HISTORY_BACKUP_CSV, PREDICTION_HISTORY_CSV):
        if not os.path.exists(filepath):
            continue
        with open(filepath, 'r', newline='') as f:
            reader = csv.reader(f)
            file_header = next(reader, None)
            if file_header and 'period' in file_header:
                current_header = file_header
            for row in reader:
                if not row:
                    continue
                while len(row) < len(current_header):
                    row.append('')
                if len(row) > 1 and row[1]:
                    by_period[row[1]] = row
    rows = list(by_period.values())
    rows.sort(
        key=lambda row: int(row[1]) if len(row) > 1 and row[1].isdigit() else 0
    )
    return current_header, rows


def load_predictions_csv(limit=500):
    if not os.path.exists(PREDICTIONS_CSV):
        return []
    predictions = []
    read_limit = 5000
    try:
        with open(PREDICTIONS_CSV, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return []
            rows = []
            for row in reader:
                if len(row) < 8:
                    continue
                rows.append(row)
                if len(rows) > read_limit:
                    rows.pop(0)
        for row in reversed(rows):
            predictions.append({
                'period': row[0],
                'prediction': row[1] if row[1] else None,
                'status': row[2],
                'confidence': float(row[3]),
                'actual': row[4] if row[4] else None,
                'number': row[5] if row[5] else None,
                'patternUsed': row[6],
                'timestamp': int(row[7]),
                'skipped': row[8] == '1' if len(row) > 8 else False,
                'skipReason': row[9] if len(row) > 9 else '',
            })
    except Exception:
        return []
    if len(predictions) > limit:
        predictions = predictions[:limit]
    return predictions


def append_prediction_csv(entry):
    is_new = not os.path.exists(PREDICTIONS_CSV)
    lock = _get_lock('predictions')
    with lock:
        with open(PREDICTIONS_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow([
                    'period', 'prediction', 'status', 'confidence',
                    'actual', 'number', 'patternUsed', 'timestamp',
                    'skipped', 'skipReason'
                ])
            writer.writerow([
                entry.get('period', ''),
                entry.get('prediction', '') if entry.get('prediction') is not None else '',
                entry.get('status', 'Pending'),
                entry.get('confidence', 0),
                entry.get('actual', '') if entry.get('actual') is not None else '',
                entry.get('number', '') if entry.get('number') is not None else '',
                entry.get('patternUsed', 'ensemble'),
                entry.get('timestamp', int(time.time())),
                '1' if entry.get('skipped', False) else '0',
                entry.get('skipReason', ''),
            ])


def rewrite_predictions_csv(predictions):
    sorted_preds = list(reversed(predictions))
    if len(sorted_preds) > MAX_PREDICTIONS_CSV:
        sorted_preds = sorted_preds[-MAX_PREDICTIONS_CSV:]
    lock = _get_lock('predictions_rewrite')
    with lock:
        tmp = PREDICTIONS_CSV + '.tmp'
        with open(tmp, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'period', 'prediction', 'status', 'confidence',
                'actual', 'number', 'patternUsed', 'timestamp',
                'skipped', 'skipReason'
            ])
            for entry in sorted_preds:
                writer.writerow([
                    entry.get('period', ''),
                    entry.get('prediction', '') if entry.get('prediction') is not None else '',
                    entry.get('status', 'Pending'),
                    entry.get('confidence', 0),
                    entry.get('actual', '') if entry.get('actual') is not None else '',
                    entry.get('number', '') if entry.get('number') is not None else '',
                    entry.get('patternUsed', 'ensemble'),
                    entry.get('timestamp', int(time.time())),
                    '1' if entry.get('skipped', False) else '0',
                    entry.get('skipReason', ''),
                ])
        os.replace(tmp, PREDICTIONS_CSV)


def append_stats_csv(stats_row):
    is_new = not os.path.exists(STATS_CSV)
    lock = _get_lock('stats')
    with lock:
        with open(STATS_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow([
                    'timestamp', 'date', 'totalWins', 'totalLosses',
                    'winRate', 'accuracy', 'recentAccuracy',
                    'totalPredictions', 'streak',
                    'consecutiveLosses', 'consecutiveWins'
                ])
            writer.writerow([
                stats_row.get('timestamp', int(time.time())),
                stats_row.get('date', time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())),
                stats_row.get('totalWins', 0),
                stats_row.get('totalLosses', 0),
                stats_row.get('winRate', 0),
                stats_row.get('accuracy', 0),
                stats_row.get('recentAccuracy', 0),
                stats_row.get('totalPredictions', 0),
                stats_row.get('streak', '0 None'),
                stats_row.get('consecutiveLosses', 0),
                stats_row.get('consecutiveWins', 0),
            ])
    _trim_csv(STATS_CSV, MAX_STATS_CSV)


def _trim_csv(filepath, max_rows):
    if not os.path.exists(filepath):
        return
    lock = _get_lock(f'trim_{os.path.basename(filepath)}')
    with lock:
        with open(filepath, 'r', newline='') as f:
            lines = f.readlines()
        if len(lines) <= max_rows + 1:
            return
        header = lines[0]
        data_lines = lines[1:]
        trimmed = data_lines[-max_rows:]
        tmp = filepath + '.trim.tmp'
        with open(tmp, 'w', newline='') as f:
            f.write(header)
            f.writelines(trimmed)
        os.replace(tmp, filepath)


def load_pending_predictions_csv(max_rows=200):
    if not os.path.exists(PENDING_PREDICTIONS_CSV):
        return []
    pending = []
    try:
        with open(PENDING_PREDICTIONS_CSV, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return []
            for row in reader:
                if len(row) < 10:
                    continue
                entry = {
                    'period': row[0],
                    'prediction': row[1] if row[1] else None,
                    'status': row[2],
                    'confidence': float(row[3]),
                    'actual': row[4] if row[4] else None,
                    'number': row[5] if row[5] else None,
                    'patternUsed': row[6],
                    'timestamp': int(row[7]),
                    'skipped': row[8] == '1',
                    'skipReason': row[9] if len(row) > 9 else '',
                }
                if len(row) > 10 and row[10]:
                    try:
                        entry['patternPredictions'] = json.loads(row[10])
                    except Exception:
                        entry['patternPredictions'] = None
                else:
                    entry['patternPredictions'] = None
                pending.append(entry)
                if len(pending) >= max_rows:
                    break
    except Exception:
        return []
    return pending


def save_pending_predictions_csv(pending):
    pending_only = []
    verified = []
    for entry in pending:
        if entry.get('status') == 'Pending' and not entry.get('locked'):
            pending_only.append(entry)
        else:
            verified.append(entry)
    verified = verified[-100:]
    combined = pending_only + verified
    lock = _get_lock('pending')
    with lock:
        tmp = PENDING_PREDICTIONS_CSV + '.tmp'
        with open(tmp, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'period', 'prediction', 'status', 'confidence',
                'actual', 'number', 'patternUsed', 'timestamp',
                'skipped', 'skipReason', 'patternPredictions'
            ])
            for entry in combined:
                pp = entry.get('patternPredictions')
                pp_json = json.dumps(pp, ensure_ascii=False) if pp else ''
                writer.writerow([
                    entry.get('period', ''),
                    entry.get('prediction', '') if entry.get('prediction') is not None else '',
                    entry.get('status', 'Pending'),
                    entry.get('confidence', 0),
                    entry.get('actual', '') if entry.get('actual') is not None else '',
                    entry.get('number', '') if entry.get('number') is not None else '',
                    entry.get('patternUsed', 'ensemble'),
                    entry.get('timestamp', int(time.time())),
                    '1' if entry.get('skipped', False) else '0',
                    entry.get('skipReason', ''),
                    pp_json,
                ])
        os.replace(tmp, PENDING_PREDICTIONS_CSV)


def load_pattern_stats_csv():
    default = build_default_user_state()['patternStatsAdvanced']
    if not os.path.exists(PATTERN_STATS_CSV):
        return default
    stats = {}
    try:
        with open(PATTERN_STATS_CSV, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return default
            for row in reader:
                if len(row) < 7:
                    continue
                stats[row[0]] = {
                    'wins': int(row[1]),
                    'total': int(row[2]),
                    'successRate': float(row[3]),
                    'recentWins': int(row[4]),
                    'recentTotal': int(row[5]),
                    'consecutiveLosses': int(row[6]),
                }
    except Exception:
        return default
    for p, val in default.items():
        stats.setdefault(p, val)
    return stats


def save_pattern_stats_csv(stats):
    lock = _get_lock('pattern_stats')
    with lock:
        tmp = PATTERN_STATS_CSV + '.tmp'
        with open(tmp, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'pattern', 'wins', 'total', 'successRate',
                'recentWins', 'recentTotal', 'consecutiveLosses'
            ])
            for pattern, row in stats.items():
                writer.writerow([
                    pattern,
                    row.get('wins', 0),
                    row.get('total', 0),
                    row.get('successRate', 0),
                    row.get('recentWins', 0),
                    row.get('recentTotal', 0),
                    row.get('consecutiveLosses', 0),
                ])
        os.replace(tmp, PATTERN_STATS_CSV)


def load_number_stats_csv():
    def_patterns = build_default_user_state()['numberPatterns']
    def_rep = build_default_user_state()['numberRepetition']
    if not os.path.exists(NUMBER_STATS_CSV):
        return {'patterns': def_patterns, 'repetition': def_rep}
    patterns = {}
    repetition = {}
    try:
        with open(NUMBER_STATS_CSV, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return {'patterns': def_patterns, 'repetition': def_rep}
            for row in reader:
                if len(row) < 9:
                    continue
                num = int(row[0])
                patterns[num] = {
                    'BIG': {'count': float(row[1]), 'successRate': float(row[2])},
                    'SMALL': {'count': float(row[3]), 'successRate': float(row[4])},
                    'total': float(row[5])
                }
                repetition[num] = {
                    'count': int(row[6]),
                    'recentCount': int(row[7]),
                    'lastSeen': int(row[8])
                }
    except Exception:
        return {'patterns': def_patterns, 'repetition': def_rep}
    for i in range(10):
        patterns.setdefault(i, def_patterns[i])
        repetition.setdefault(i, def_rep[i])
    return {'patterns': patterns, 'repetition': repetition}


def save_number_stats_csv(patterns, repetition):
    lock = _get_lock('number_stats')
    with lock:
        tmp = NUMBER_STATS_CSV + '.tmp'
        with open(tmp, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'number', 'big_count', 'big_successRate',
                'small_count', 'small_successRate', 'total',
                'rep_count', 'rep_recentCount', 'rep_lastSeen'
            ])
            for i in range(10):
                p = patterns.get(i, {
                    'BIG': {'count': 0, 'successRate': 0},
                    'SMALL': {'count': 0, 'successRate': 0},
                    'total': 0
                })
                r = repetition.get(i, {'count': 0, 'recentCount': 0, 'lastSeen': 0})
                writer.writerow([
                    i,
                    p['BIG'].get('count', 0),
                    p['BIG'].get('successRate', 0),
                    p['SMALL'].get('count', 0),
                    p['SMALL'].get('successRate', 0),
                    p.get('total', 0),
                    r.get('count', 0),
                    r.get('recentCount', 0),
                    r.get('lastSeen', 0),
                ])
        os.replace(tmp, NUMBER_STATS_CSV)


def load_transition_matrix_csv():
    default = {'BIG': {'BIG': 0, 'SMALL': 0}, 'SMALL': {'BIG': 0, 'SMALL': 0}}
    if not os.path.exists(TRANSITION_MATRIX_CSV):
        return default
    matrix = {}
    try:
        with open(TRANSITION_MATRIX_CSV, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return default
            for row in reader:
                if len(row) < 3:
                    continue
                matrix[row[0]] = {'BIG': float(row[1]), 'SMALL': float(row[2])}
    except Exception:
        return default
    for s in ('BIG', 'SMALL'):
        matrix.setdefault(s, default[s])
    return matrix


def save_transition_matrix_csv(matrix):
    lock = _get_lock('transition_matrix')
    with lock:
        tmp = TRANSITION_MATRIX_CSV + '.tmp'
        with open(tmp, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['state', 'BIG', 'SMALL'])
            for state in ('BIG', 'SMALL'):
                row = matrix.get(state, {'BIG': 0, 'SMALL': 0})
                writer.writerow([state, row.get('BIG', 0), row.get('SMALL', 0)])
        os.replace(tmp, TRANSITION_MATRIX_CSV)


def load_entropy_history_csv():
    if not os.path.exists(ENTROPY_HISTORY_CSV):
        return []
    history = []
    try:
        with open(ENTROPY_HISTORY_CSV, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return []
            for row in reader:
                if row:
                    history.append(float(row[0]))
    except Exception:
        return []
    return history


def save_entropy_history_csv(history):
    lock = _get_lock('entropy_history')
    with lock:
        tmp = ENTROPY_HISTORY_CSV + '.tmp'
        with open(tmp, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['entropy'])
            for val in history:
                writer.writerow([val])
        os.replace(tmp, ENTROPY_HISTORY_CSV)


def load_neural_states_csv():
    default = {
        'weights': [0.0] * 10,
        'bias': 0.0,
        'learningRate': 0.1,
        'showHigher': True,
        'autoToggle': True,
        'lastAdjustment': 0,
        'lastProcessedPeriod': '',
    }
    if not os.path.exists(NEURAL_STATES_CSV):
        return default
    kv = {}
    try:
        with open(NEURAL_STATES_CSV, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return default
            for row in reader:
                if len(row) < 2:
                    continue
                kv[row[0]] = row[1]
    except Exception:
        return default

    states = dict(default)
    if 'bias' in kv:
        states['bias'] = float(kv['bias'])
    if 'learningRate' in kv:
        states['learningRate'] = float(kv['learningRate'])
    if 'showHigher' in kv:
        states['showHigher'] = kv['showHigher'] == '1'
    if 'autoToggle' in kv:
        states['autoToggle'] = kv['autoToggle'] == '1'
    if 'lastAdjustment' in kv:
        states['lastAdjustment'] = int(kv['lastAdjustment'])
    if 'lastProcessedPeriod' in kv:
        states['lastProcessedPeriod'] = kv['lastProcessedPeriod']
    for i in range(10):
        key = f'weight_{i}'
        if key in kv:
            states['weights'][i] = float(kv[key])
    return states


def save_neural_states_csv(states):
    lock = _get_lock('neural_states')
    with lock:
        tmp = NEURAL_STATES_CSV + '.tmp'
        with open(tmp, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['key', 'value'])
            writer.writerow(['bias', states.get('bias', 0.0)])
            writer.writerow(['learningRate', states.get('learningRate', 0.1)])
            writer.writerow(['showHigher', '1' if states.get('showHigher', True) else '0'])
            writer.writerow(['autoToggle', '1' if states.get('autoToggle', True) else '0'])
            writer.writerow(['lastAdjustment', states.get('lastAdjustment', 0)])
            writer.writerow(['lastProcessedPeriod', states.get('lastProcessedPeriod', '')])
            weights = states.get('weights', [0.0] * 10)
            for i in range(10):
                writer.writerow([f'weight_{i}', weights[i]])
        os.replace(tmp, NEURAL_STATES_CSV)


def load_loss_recovery_csv():
    default = build_default_user_state()['lossRecovery']
    if not os.path.exists(LOSS_RECOVERY_CSV):
        return default
    kv = {}
    try:
        with open(LOSS_RECOVERY_CSV, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return default
            for row in reader:
                if len(row) < 2:
                    continue
                kv[row[0]] = row[1]
    except Exception:
        return default

    lr = dict(default)
    if 'consecutiveLosses' in kv:
        lr['consecutiveLosses'] = int(kv['consecutiveLosses'])
    if 'totalSkipsThisRun' in kv:
        lr['totalSkipsThisRun'] = int(kv['totalSkipsThisRun'])
    if 'lastSkipPeriod' in kv:
        lr['lastSkipPeriod'] = kv['lastSkipPeriod']
    if 'skipCooldownUntil' in kv:
        lr['skipCooldownUntil'] = int(kv['skipCooldownUntil'])
    if 'recoveryMode' in kv:
        lr['recoveryMode'] = kv['recoveryMode'] == '1'
    if 'recoveryModeStart' in kv:
        lr['recoveryModeStart'] = int(kv['recoveryModeStart'])
    if 'lastFiveResults' in kv:
        lr['lastFiveResults'] = kv['lastFiveResults'].split(',') if kv['lastFiveResults'] else []
    if 'forcedFlipActive' in kv:
        lr['forcedFlipActive'] = kv['forcedFlipActive'] == '1'
    if 'forcedFlipCount' in kv:
        lr['forcedFlipCount'] = int(kv['forcedFlipCount'])
    if 'lossGuardActive' in kv:
        lr['lossGuardActive'] = kv['lossGuardActive'] == '1'
    if 'lossGuardReason' in kv:
        lr['lossGuardReason'] = kv['lossGuardReason']
    if 'lastSkipReason' in kv:
        lr['lastSkipReason'] = kv['lastSkipReason']
    return lr


def save_loss_recovery_csv(lr):
    lock = _get_lock('loss_recovery')
    with lock:
        tmp = LOSS_RECOVERY_CSV + '.tmp'
        with open(tmp, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['key', 'value'])
            writer.writerow(['consecutiveLosses', lr.get('consecutiveLosses', 0)])
            writer.writerow(['totalSkipsThisRun', lr.get('totalSkipsThisRun', 0)])
            writer.writerow(['lastSkipPeriod', lr.get('lastSkipPeriod', '')])
            writer.writerow(['skipCooldownUntil', lr.get('skipCooldownUntil', 0)])
            writer.writerow(['recoveryMode', '1' if lr.get('recoveryMode', False) else '0'])
            writer.writerow(['recoveryModeStart', lr.get('recoveryModeStart', 0)])
            l5 = ','.join(lr.get('lastFiveResults', []))
            writer.writerow(['lastFiveResults', l5])
            writer.writerow(['forcedFlipActive', '1' if lr.get('forcedFlipActive', False) else '0'])
            writer.writerow(['forcedFlipCount', lr.get('forcedFlipCount', 0)])
            writer.writerow(['lossGuardActive', '1' if lr.get('lossGuardActive', False) else '0'])
            writer.writerow(['lossGuardReason', lr.get('lossGuardReason', '')])
            writer.writerow(['lastSkipReason', lr.get('lastSkipReason', '')])
        os.replace(tmp, LOSS_RECOVERY_CSV)


def load_all_states():
    ns = load_neural_states_csv()
    ps = load_pattern_stats_csv()
    num_stats = load_number_stats_csv()
    tm = load_transition_matrix_csv()
    eh = load_entropy_history_csv()
    lr = load_loss_recovery_csv()
    pending = load_pending_predictions_csv()

    user_states = {
        'user': {
            'showHigher': ns['showHigher'],
            'autoToggle': ns['autoToggle'],
            'lastAdjustment': ns['lastAdjustment'],
            'patternStatsNormal': ps,
            'patternStatsAdvanced': ps,
            'numberPatterns': num_stats['patterns'],
            'numberRepetition': num_stats['repetition'],
            'transitionMatrix': tm,
            'entropyHistory': eh,
            'neuralWeights': ns['weights'],
            'bias': ns['bias'],
            'learningRate': ns['learningRate'],
            'lastProcessedPeriod': ns['lastProcessedPeriod'],
            'lossRecovery': lr,
        }
    }
    return {'userStates': user_states, 'pendingPredictions': pending}


def save_all_states(user_states, pending):
    user = user_states.get('user', {})
    ns = {
        'weights': user.get('neuralWeights', [0.0] * 10),
        'bias': user.get('bias', 0.0),
        'learningRate': user.get('learningRate', 0.1),
        'showHigher': user.get('showHigher', True),
        'autoToggle': user.get('autoToggle', True),
        'lastAdjustment': user.get('lastAdjustment', 0),
        'lastProcessedPeriod': user.get('lastProcessedPeriod', ''),
    }
    save_neural_states_csv(ns)
    save_pattern_stats_csv(user.get('patternStatsAdvanced', {}))
    save_number_stats_csv(user.get('numberPatterns', {}), user.get('numberRepetition', {}))
    save_transition_matrix_csv(user.get('transitionMatrix', {}))
    save_entropy_history_csv(user.get('entropyHistory', []))
    save_loss_recovery_csv(user.get('lossRecovery', {}))
    save_pending_predictions_csv(pending)


def migrate_json_to_csv():
    old_predictions = os.path.join(os.path.dirname(__file__), 'all_predictions.json')
    old_stats = os.path.join(os.path.dirname(__file__), 'all_stats.json')

    if os.path.exists(old_predictions) and not os.path.exists(PREDICTIONS_CSV):
        try:
            with open(old_predictions, 'r') as f:
                data = json.load(f)
            if isinstance(data, list) and data:
                data.reverse()
                with open(PREDICTIONS_CSV, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'period', 'prediction', 'status', 'confidence',
                        'actual', 'number', 'patternUsed', 'timestamp',
                        'skipped', 'skipReason'
                    ])
                    for entry in data:
                        writer.writerow([
                            entry.get('period', ''),
                            entry.get('prediction', '') if entry.get('prediction') is not None else '',
                            entry.get('status', 'Pending'),
                            entry.get('confidence', 0),
                            entry.get('actual', '') if entry.get('actual') is not None else '',
                            entry.get('number', '') if entry.get('number') is not None else '',
                            entry.get('patternUsed', 'ensemble'),
                            entry.get('timestamp', int(time.time())),
                            '1' if entry.get('skipped', False) else '0',
                            entry.get('skipReason', ''),
                        ])
        except Exception:
            pass

    old_state = os.path.join(os.path.dirname(__file__), 'prediction_data.json')
    if os.path.exists(old_state) and not os.path.exists(NEURAL_STATES_CSV):
        try:
            with open(old_state, 'r') as f:
                data = json.load(f)
            if isinstance(data, dict):
                user = data.get('userStates', {}).get('user', {})
                ns = {
                    'weights': user.get('neuralWeights', [0.0] * 10),
                    'bias': user.get('bias', 0.0),
                    'learningRate': user.get('learningRate', 0.1),
                    'showHigher': user.get('showHigher', True),
                    'autoToggle': user.get('autoToggle', True),
                    'lastAdjustment': user.get('lastAdjustment', 0),
                    'lastProcessedPeriod': user.get('lastProcessedPeriod', ''),
                }
                save_neural_states_csv(ns)
                save_pattern_stats_csv(
                    user.get('patternStatsAdvanced', build_default_user_state()['patternStatsAdvanced'])
                )
                np_data = user.get('numberPatterns', build_default_user_state()['numberPatterns'])
                nr_data = user.get('numberRepetition', build_default_user_state()['numberRepetition'])
                save_number_stats_csv(np_data, nr_data)
                save_transition_matrix_csv(
                    user.get('transitionMatrix', build_default_user_state()['transitionMatrix'])
                )
                save_entropy_history_csv(user.get('entropyHistory', []))
                save_loss_recovery_csv(
                    user.get('lossRecovery', build_default_user_state()['lossRecovery'])
                )
                save_pending_predictions_csv(data.get('pendingPredictions', []))
        except Exception:
            pass


def append_prediction_history_csv(entry):
    is_new = not os.path.exists(PREDICTION_HISTORY_CSV)
    lock = _get_lock('prediction_history')
    with lock:
        with open(PREDICTION_HISTORY_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow([
                    'id', 'period', 'prediction', 'status', 'confidence',
                    'actual', 'number', 'patternused', 'timestamp',
                    'skipped', 'skipreason', 'created_at'
                ])
            writer.writerow([
                '',  # id (auto)
                entry.get('period', ''),
                entry.get('prediction', ''),
                entry.get('status', 'Pending'),
                entry.get('confidence', 0),
                entry.get('actual', ''),
                entry.get('number', ''),
                entry.get('patternUsed', 'ensemble'),
                entry.get('timestamp', int(time.time())),
                '1' if entry.get('skipped', False) else '0',
                entry.get('skipReason', ''),
                time.strftime('%Y-%m-%d %H:%M:%S'),
            ])


def period_exists_in_history_csv(period):
    if not os.path.exists(PREDICTION_HISTORY_CSV):
        return False
    lock = _get_lock('prediction_history')
    with lock:
        with open(PREDICTION_HISTORY_CSV, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row and len(row) > 1 and row[1] == period:
                    return True
    return False


def safe_append_prediction(entry):
    period = entry.get('period', '')
    if not period:
        return
    lock = _get_lock('prediction_history')
    with lock:
        if not os.path.exists(PREDICTION_HISTORY_CSV):
            with open(PREDICTION_HISTORY_CSV, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'period', 'prediction', 'status', 'confidence',
                                 'actual', 'number', 'patternused', 'timestamp',
                                 'skipped', 'skipreason', 'created_at'])
        with open(PREDICTION_HISTORY_CSV, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row and len(row) > 1 and row[1] == period:
                    return
        with open(PREDICTION_HISTORY_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                '',
                period,
                entry.get('prediction', '') if entry.get('prediction') is not None else '',
                entry.get('status', 'Pending'),
                entry.get('confidence', 0),
                entry.get('actual', '') if entry.get('actual') is not None else '',
                entry.get('number', '') if entry.get('number') is not None else '',
                entry.get('patternUsed', 'ensemble'),
                entry.get('timestamp', int(time.time())),
                '1' if entry.get('skipped', False) else '0',
                entry.get('skipReason', ''),
                time.strftime('%Y-%m-%d %H:%M:%S'),
            ])


def upsert_prediction_history_csv(entry):
    period = str(entry.get('period', ''))
    if not period or str(entry.get('patternUsed', '')).lower() == 'imported':
        return

    header = ['id', 'period', 'prediction', 'status', 'confidence',
              'actual', 'number', 'patternused', 'timestamp',
              'skipped', 'skipreason', 'created_at']

    def csv_value(value):
        return '' if value is None else str(value)

    lock = _get_lock('prediction_history')
    with lock:
        current_header, rows = _load_merged_history_rows(header)
        found = False
        for row in rows:
            if len(row) > 1 and row[1] == period:
                row[2] = csv_value(entry.get('prediction'))
                row[3] = entry.get('status', 'Pending')
                row[4] = csv_value(entry.get('confidence', 0))
                row[5] = csv_value(entry.get('actual'))
                row[6] = csv_value(entry.get('number'))
                row[7] = entry.get('patternUsed', 'ensemble')
                row[8] = csv_value(entry.get('timestamp', int(time.time())))
                row[9] = '1' if entry.get('skipped', False) else '0'
                row[10] = entry.get('skipReason', '') or ''
                if not row[11]:
                    row[11] = time.strftime('%Y-%m-%d %H:%M:%S')
                found = True

        if not found:
            rows.append([
                '',
                period,
                csv_value(entry.get('prediction')),
                entry.get('status', 'Pending'),
                csv_value(entry.get('confidence', 0)),
                csv_value(entry.get('actual')),
                csv_value(entry.get('number')),
                entry.get('patternUsed', 'ensemble'),
                csv_value(entry.get('timestamp', int(time.time()))),
                '1' if entry.get('skipped', False) else '0',
                entry.get('skipReason', '') or '',
                time.strftime('%Y-%m-%d %H:%M:%S'),
            ])

        _atomic_write_csv(PREDICTION_HISTORY_CSV, current_header, rows)
        _atomic_write_csv(PREDICTION_HISTORY_BACKUP_CSV, current_header, rows)


def load_prediction_history_entries(limit=None):
    header = ['id', 'period', 'prediction', 'status', 'confidence',
              'actual', 'number', 'patternused', 'timestamp',
              'skipped', 'skipreason', 'created_at']
    try:
        current_header, rows = _load_merged_history_rows(header)
    except Exception:
        return []

    entries = []
    selected_rows = rows if limit is None else rows[-limit:]
    for row in reversed(selected_rows):
        while len(row) < len(current_header):
            row.append('')
        item = dict(zip(current_header, row))
        pattern_used = item.get('patternused') or item.get('patternUsed') or 'ensemble'
        if str(pattern_used).lower() == 'imported':
            continue
        try:
            confidence = float(item.get('confidence') or 0)
        except Exception:
            confidence = 0
        try:
            timestamp = int(float(item.get('timestamp') or time.time()))
        except Exception:
            timestamp = int(time.time())
        entries.append({
            'period': item.get('period', ''),
            'prediction': item.get('prediction') or None,
            'status': item.get('status') or 'Pending',
            'confidence': confidence,
            'actual': item.get('actual') or None,
            'number': item.get('number') or None,
            'patternUsed': pattern_used,
            'timestamp': timestamp,
            'skipped': str(item.get('skipped', '')).lower() in ('1', 'true'),
            'skipReason': item.get('skipreason') or item.get('skipReason') or '',
        })
    return entries


def import_all_to_history_csv():
    if not os.path.exists(PREDICTIONS_CSV):
        return
    lock = _get_lock('prediction_history')
    with lock:
        existing = {}
        if os.path.exists(PREDICTION_HISTORY_CSV):
            with open(PREDICTION_HISTORY_CSV, 'r', newline='') as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if row and len(row) > 1 and row[1]:
                        existing[row[1]] = True

        new_rows = []
        with open(PREDICTIONS_CSV, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) < 8:
                    continue
                period = row[0]
                pattern_used = row[6] if row[6] else 'ensemble'
                if pattern_used.lower() == 'imported':
                    continue
                if period and period not in existing:
                    new_rows.append({
                        'period': period,
                        'prediction': row[1] if row[1] else None,
                        'status': row[2],
                        'confidence': float(row[3]) if row[3] else 0,
                        'actual': row[4] if row[4] else None,
                        'number': row[5] if row[5] else None,
                        'patternUsed': pattern_used,
                        'timestamp': int(row[7]) if len(row) > 7 and row[7] else int(time.time()),
                        'skipped': row[8] == '1' if len(row) > 8 else False,
                        'skipReason': row[9] if len(row) > 9 else '',
                    })

        if new_rows:
            with open(PREDICTION_HISTORY_CSV, 'a', newline='') as f:
                writer = csv.writer(f)
                for entry in new_rows:
                    writer.writerow([
                        '',
                        entry.get('period', ''),
                        entry.get('prediction', '') if entry.get('prediction') is not None else '',
                        entry.get('status', 'Pending'),
                        entry.get('confidence', 0),
                        entry.get('actual', '') if entry.get('actual') is not None else '',
                        entry.get('number', '') if entry.get('number') is not None else '',
                        entry.get('patternUsed', 'ensemble'),
                        entry.get('timestamp', int(time.time())),
                        '1' if entry.get('skipped', False) else '0',
                        entry.get('skipReason', ''),
                        time.strftime('%Y-%m-%d %H:%M:%S'),
                    ])
            print(f"  [Migrate] Added {len(new_rows)} old predictions to history CSV")


def update_prediction_history_csv(period, status, actual, number):
    def csv_value(value):
        return '' if value is None else str(value)

    header = ['id', 'period', 'prediction', 'status', 'confidence',
              'actual', 'number', 'patternused', 'timestamp',
              'skipped', 'skipreason', 'created_at']
    lock = _get_lock('prediction_history')
    with lock:
        current_header, rows = _load_merged_history_rows(header)
        found = False
        for row in rows:
            if len(row) > 1 and row[1] == period:
                row[3] = status
                row[5] = csv_value(actual)
                row[6] = csv_value(number)
                found = True
        if not found:
            rows.append([
                '',
                period,
                '',
                status,
                '',
                csv_value(actual),
                csv_value(number),
                '',
                str(int(time.time())),
                '0',
                '',
                time.strftime('%Y-%m-%d %H:%M:%S'),
            ])
        _atomic_write_csv(PREDICTION_HISTORY_CSV, current_header, rows)
        _atomic_write_csv(PREDICTION_HISTORY_BACKUP_CSV, current_header, rows)
