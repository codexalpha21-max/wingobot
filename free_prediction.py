import csv
import json
import os
import threading
import time

from analyzers import (
    analyze_cycle_pattern,
    analyze_entropy_based,
    analyze_long_pattern,
    analyze_markov_2nd_order,
    analyze_markov_chain,
    analyze_number_patterns,
    analyze_prediction_accuracy,
    analyze_skip_pattern,
    analyze_streak_momentum,
    analyze_trend_based,
    analyze_win_loss_tracker,
    analyze_zigzag_pattern,
    detect_market_regime,
    get_anti_bias_correction,
    get_number_based_prediction,
)
from helpers import build_default_user_state, fetch_api_data, get_current_period_1min
from config import DATA_DIR


BASE_DIR = os.path.dirname(__file__)
FREE_HISTORY_CSV = os.path.join(DATA_DIR, 'free', 'free_prediction_history.csv')
FREE_HISTORY_BACKUP_CSV = FREE_HISTORY_CSV + '.backup'
FREE_STATE_FILE = os.path.join(DATA_DIR, 'free_state.json')
FREE_LOCK_FILE = os.path.join(DATA_DIR, 'free_prediction.lock')
FREE_HISTORY_LIMIT = 20

HEADER = [
    'id', 'period', 'prediction', 'status', 'confidence',
    'actual', 'number', 'patternused', 'timestamp',
    'skipped', 'skipreason', 'created_at'
]

_lock = threading.RLock()
_history_snapshot = []


def _period_key(period):
    try:
        return int(str(period))
    except Exception:
        return 0


def _csv_value(value):
    return '' if value is None else str(value)


def _read_rows_from(path):
    if not os.path.exists(path):
        return []
    rows = []
    try:
        with open(path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or 'period' not in reader.fieldnames:
                return []
            for row in reader:
                if row.get('period'):
                    rows.append(row)
    except Exception:
        return []
    return rows


def load_free_history(limit=None):
    global _history_snapshot
    by_period = {}
    for path in (FREE_HISTORY_BACKUP_CSV, FREE_HISTORY_CSV):
        for row in _read_rows_from(path):
            by_period[str(row.get('period', ''))] = row
    rows = list(by_period.values())
    rows.sort(key=lambda row: _period_key(row.get('period')), reverse=True)
    if rows:
        _history_snapshot = [dict(row) for row in rows]
    elif _history_snapshot:
        rows = [dict(row) for row in _history_snapshot]
    return rows[:limit] if limit else rows


def _write_rows(rows):
    global _history_snapshot
    os.makedirs(os.path.dirname(FREE_HISTORY_CSV), exist_ok=True)
    rows_asc = sorted(rows, key=lambda row: _period_key(row.get('period')))
    for path in (FREE_HISTORY_CSV, FREE_HISTORY_BACKUP_CSV):
        tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
        try:
            with open(tmp, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=HEADER)
                writer.writeheader()
                for row in rows_asc:
                    writer.writerow({key: row.get(key, '') for key in HEADER})
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
    _history_snapshot = [
        dict(row) for row in sorted(rows, key=lambda row: _period_key(row.get('period')), reverse=True)
    ]


def upsert_free_history(entry):
    period = str(entry.get('period', ''))
    if not period:
        return
    row = {
        'id': '',
        'period': period,
        'prediction': _csv_value(entry.get('prediction')),
        'status': entry.get('status', 'Pending'),
        'confidence': _csv_value(entry.get('confidence', 0)),
        'actual': _csv_value(entry.get('actual')),
        'number': _csv_value(entry.get('number')),
        'patternused': entry.get('patternUsed', 'free_ensemble'),
        'timestamp': _csv_value(entry.get('timestamp', int(time.time()))),
        'skipped': '1' if entry.get('skipped') else '0',
        'skipreason': entry.get('skipReason', '') or '',
        'created_at': entry.get('created_at') or time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with _lock:
        rows = load_free_history()
        found = False
        for idx, old in enumerate(rows):
            if str(old.get('period', '')) == period:
                row['created_at'] = old.get('created_at') or row['created_at']
                rows[idx] = row
                found = True
                break
        if not found:
            rows.append(row)
        _write_rows(rows)


def _row_to_entry(row):
    return {
        'period': str(row.get('period', '')),
        'prediction': row.get('prediction') or None,
        'status': row.get('status') or 'Pending',
        'confidence': float(row.get('confidence') or 0),
        'actual': row.get('actual') or None,
        'number': row.get('number') or None,
        'patternUsed': row.get('patternused') or row.get('patternUsed') or 'free_ensemble',
        'timestamp': int(float(row.get('timestamp') or time.time())),
        'skipped': str(row.get('skipped', '')).lower() in ('1', 'true'),
        'skipReason': row.get('skipreason') or row.get('skipReason') or '',
    }


def _clean_number(number):
    if number in (None, ''):
        return None
    try:
        return int(float(str(number)))
    except Exception:
        return None


def _number_color(number):
    number = _clean_number(number)
    if number is None:
        return None
    if number == 0:
        return 'RED,VIOLET'
    if number == 5:
        return 'GREEN,VIOLET'
    return 'GREEN' if number % 2 else 'RED'


def _public_entry(entry):
    number = _clean_number(entry.get('number'))
    raw_pred = entry.get('prediction') or ''
    is_skip = entry.get('skipped', False) or str(entry.get('status', '')).upper() == 'SKIP'
    if not raw_pred or raw_pred not in ('BIG', 'SMALL', 'SKIP'):
        raw_pred = 'SKIP' if is_skip else 'BIG'
    return {
        'period': entry.get('period', ''),
        'prediction': raw_pred,
        'status': entry.get('status', 'Pending'),
        'confidence': round(float(entry.get('confidence') or 0), 2),
        'actual': entry.get('actual'),
        'number': number,
        'actualNumber': number,
        'color': _number_color(number),
        'actualColor': _number_color(number),
        'skipped': bool(entry.get('skipped', False)),
        'skipReason': entry.get('skipReason') or None,
    }


def _load_state():
    default = {'user': build_default_user_state()}
    if not os.path.exists(FREE_STATE_FILE):
        return default
    try:
        with open(FREE_STATE_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get('user'), dict):
            merged = default['user']
            merged.update(data['user'])
            return {
                'user': merged,
                'freeDecision': data.get('freeDecision', {}),
            }
    except Exception:
        pass
    return default


def _save_state(state):
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = f"{FREE_STATE_FILE}.{os.getpid()}.tmp"
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(state, f, separators=(',', ':'))
    os.replace(tmp, FREE_STATE_FILE)


def _entries():
    return [_row_to_entry(row) for row in load_free_history()]


def _update_loss_state(state, entries):
    lr = state['user'].setdefault('lossRecovery', build_default_user_state()['lossRecovery'])
    recent = [e['status'] for e in entries if e.get('status') in ('WIN', 'LOSS')][:10]
    lr['lastFiveResults'] = recent[:5]
    cons_loss = 0
    for status in recent:
        if status == 'LOSS':
            cons_loss += 1
        else:
            break
    cons_win = 0
    for status in recent:
        if status == 'WIN':
            cons_win += 1
        else:
            break
    lr['consecutiveLosses'] = cons_loss
    lr['recoveryMode'] = cons_loss >= 1 and cons_win == 0
    lr['lossGuardActive'] = cons_loss >= 2
    lr['lossGuardReason'] = (
        f"Free recovery: {cons_loss} consecutive losses, prediction will be guarded."
        if cons_loss >= 2 else ''
    )
    if lr['recoveryMode'] and not lr.get('recoveryModeStart'):
        lr['recoveryModeStart'] = int(time.time())


def verify_free_pending(entries, state):
    current_period = get_current_period_1min()
    pending = [
        e for e in entries
        if e.get('status') in ('Pending', 'SKIP')
        and e.get('actual') not in ('BIG', 'SMALL')
        and str(e.get('period', '')) < current_period
    ]
    if not pending:
        _update_loss_state(state, entries)
        return entries

    game_data = fetch_api_data(retries=2, timeout=5, bypass_cache=False)
    if isinstance(game_data, dict) and game_data.get('error'):
        _update_loss_state(state, entries)
        return entries

    by_period = {str(_period_key(item.get('period', ''))): item for item in game_data if item.get('period')}
    changed = False
    for entry in entries:
        if (
            entry.get('status') not in ('Pending', 'SKIP')
            or entry.get('actual') in ('BIG', 'SMALL')
            or str(entry.get('period', '')) >= current_period
        ):
            continue
        match = by_period.get(str(entry.get('period')))
        if not match:
            suffix = str(entry.get('period', ''))[-3:]
            match = next((item for item in game_data if str(item.get('period', '')).endswith(suffix)), None)
        if not match:
            continue
        actual = match.get('category')
        entry['actual'] = actual
        entry['number'] = match.get('number')
        entry['status'] = (
            'SKIP'
            if entry.get('skipped') or not entry.get('prediction')
            else 'WIN' if entry.get('prediction') == actual else 'LOSS'
        )
        entry['timestamp'] = entry.get('timestamp') or int(time.time())
        upsert_free_history(entry)
        changed = True

    if changed:
        entries = _entries()
    _update_loss_state(state, entries)
    return entries


def _add_vote(scores, pattern_predictions, name, prediction, confidence, score, weight, anti_bias):
    if prediction not in ('BIG', 'SMALL'):
        return
    confidence = float(confidence or 50)
    score = float(score or confidence)
    pattern_predictions[name] = {
        'prediction': prediction,
        'confidence': round(confidence, 2),
        'score': round(score, 2),
    }
    penalty = anti_bias.get('multiplier', 1.0)
    if anti_bias.get('correction') == 'PENALIZE_BIG' and prediction == 'BIG':
        weight *= penalty
    if anti_bias.get('correction') == 'PENALIZE_SMALL' and prediction == 'SMALL':
        weight *= penalty
    scores[prediction] += weight * confidence * (score / 100)


def _free_loss_signal(entries):
    settled = [
        entry for entry in entries
        if entry.get('status') in ('WIN', 'LOSS')
        and entry.get('prediction') in ('BIG', 'SMALL')
        and entry.get('actual') in ('BIG', 'SMALL')
    ]
    losses = []
    for entry in settled:
        if entry.get('status') != 'LOSS':
            break
        losses.append(entry)

    signal = {
        'consecutiveLosses': len(losses),
        'prediction': None,
        'reason': None,
        'confidence': 0,
    }
    if not losses:
        return signal

    if len(losses) >= 2:
        predictions = [entry.get('prediction') for entry in losses[:6]]
        actuals = [entry.get('actual') for entry in losses[:6]]
        repeated_inverse = (
            len(set(predictions)) == 1
            and len(set(actuals)) == 1
            and predictions[0] != actuals[0]
        )
        alternating_actuals = all(
            actuals[index] != actuals[index - 1]
            for index in range(1, len(actuals))
        )
        if repeated_inverse:
            signal.update({
                'prediction': actuals[0],
                'reason': 'free_repeated_inverse_recovery',
                'confidence': min(92, 80 + len(losses) * 4),
            })
        elif alternating_actuals:
            signal.update({
                'prediction': 'SMALL' if actuals[0] == 'BIG' else 'BIG',
                'reason': 'free_alternating_loss_recovery',
                'confidence': min(88, 76 + len(losses) * 3),
            })
    return signal


def _free_risk_assessment(
    confidence,
    scores,
    pattern_predictions,
    loss_signal,
    entries,
    regime,
):
    total_score = scores['BIG'] + scores['SMALL']
    edge = abs(scores['BIG'] - scores['SMALL']) / max(total_score, 0.001)
    winner = 'BIG' if scores['BIG'] >= scores['SMALL'] else 'SMALL'
    votes = [
        data.get('prediction') for data in pattern_predictions.values()
        if data.get('prediction') in ('BIG', 'SMALL')
    ]
    agreement = votes.count(winner) / max(len(votes), 1)
    risk = 0
    reasons = []

    if confidence < 60:
        risk += 25
        reasons.append(f'Low confidence ({round(confidence, 2)}%).')
    elif confidence < 66:
        risk += 12
        reasons.append(f'Moderate confidence ({round(confidence, 2)}%).')
    if edge < 0.10:
        risk += 25
        reasons.append('Pattern scores are close to a tie.')
    elif edge < 0.18:
        risk += 12
        reasons.append('Pattern score edge is narrow.')
    if agreement < 0.50:
        risk += 22
        reasons.append(f'Pattern agreement is weak ({round(agreement * 100, 2)}%).')
    elif agreement < 0.62:
        risk += 10
        reasons.append(f'Pattern agreement is mixed ({round(agreement * 100, 2)}%).')
    if regime == 'MIXED':
        risk += 10
        reasons.append('Market regime is mixed.')

    consecutive_losses = int(loss_signal.get('consecutiveLosses') or 0)
    if consecutive_losses:
        risk += min(24, consecutive_losses * 8)
        reasons.append(f'Active loss streak: {consecutive_losses}.')
    strong_recovery = loss_signal.get('prediction') in ('BIG', 'SMALL')
    if strong_recovery:
        risk = max(0, risk - 30)
        reasons.append('Verified loss-recovery pattern is active.')

    recent_skip = any(
        entry.get('skipped') or str(entry.get('status', '')).upper() == 'SKIP'
        for entry in entries[:2]
    )
    risk = min(100, risk)
    level = 'HIGH' if risk >= 55 else 'MEDIUM' if risk >= 35 else 'LOW'
    should_skip = risk >= 55 and not recent_skip and not (
        strong_recovery and confidence >= 68
    )
    if recent_skip and risk >= 55:
        reasons.append('Skip cooldown active.')
    return {
        'score': risk,
        'level': level,
        'skip': should_skip,
        'edge': round(edge, 4),
        'agreement': round(agreement * 100, 2),
        'winner': winner,
        'strongRecovery': strong_recovery,
        'recentSkipCooldown': recent_skip,
        'reasons': reasons,
    }


def _build_prediction(entries, state):
    game_data = fetch_api_data(retries=2, timeout=5)
    results = [] if isinstance(game_data, dict) and game_data.get('error') else game_data[:150]
    if not results:
        settled = [e for e in entries if e.get('actual') in ('BIG', 'SMALL')]
        results = [
            {'period': e['period'], 'category': e['actual'], 'number': e.get('number')}
            for e in settled[:150]
        ]

    latest_result = results[0]['category'] if results else 'SMALL'
    latest_number = _clean_number(results[0].get('number')) if results else None
    win_loss = analyze_win_loss_tracker(entries)
    accuracy = analyze_prediction_accuracy(entries)
    anti_bias = get_anti_bias_correction(entries)

    if results:
        analyze_number_patterns(results, state)
    trend = analyze_trend_based(results)
    zigzag = analyze_zigzag_pattern(results)
    skip = analyze_skip_pattern(results)
    cycle = analyze_cycle_pattern(results)
    long_pat = analyze_long_pattern(results, win_loss)
    markov = analyze_markov_chain(results, state)
    entropy = analyze_entropy_based(results, state)
    number_based = get_number_based_prediction(
        latest_number,
        state['user'].get('numberPatterns', {}),
        state['user'].get('numberRepetition', {}),
    )
    streak = analyze_streak_momentum(results)
    markov2 = analyze_markov_2nd_order(results)
    regime = detect_market_regime(results).get('regime', 'MIXED')

    scores = {'BIG': 0.0, 'SMALL': 0.0}
    pattern_predictions = {}
    streak_weight = 32 if regime == 'STREAK' else 14 if regime == 'ZIGZAG' else 24
    markov2_weight = 28 if regime == 'STREAK' else 16 if regime == 'ZIGZAG' else 22

    _add_vote(scores, pattern_predictions, 'streakMomentum', streak.get('prediction'), streak.get('confidence'), streak.get('streakScore'), streak_weight, anti_bias)
    _add_vote(scores, pattern_predictions, 'markov2', markov2.get('prediction'), markov2.get('confidence'), markov2.get('markov2Score'), markov2_weight, anti_bias)
    _add_vote(scores, pattern_predictions, 'markovChain', markov.get('prediction'), markov.get('confidence'), markov.get('markovScore'), 16, anti_bias)
    _add_vote(scores, pattern_predictions, 'entropyBased', entropy.get('prediction'), entropy.get('confidence'), entropy.get('entropyScore'), 14, anti_bias)
    _add_vote(scores, pattern_predictions, 'numberBased', number_based.get('prediction'), number_based.get('confidence'), number_based.get('numberScore'), 12, anti_bias)

    trend_pred = trend.get('trend')
    if trend_pred == 'NEUTRAL':
        trend_pred = 'SMALL' if latest_result == 'BIG' else 'BIG'
    _add_vote(scores, pattern_predictions, 'trendBased', trend_pred, 58 + accuracy.get('recentAccuracy', 0) / 10, trend.get('trendScore'), 18, anti_bias)

    if zigzag.get('isZigZag') and not zigzag.get('isBroken'):
        _add_vote(scores, pattern_predictions, 'zigZag', zigzag.get('nextPrediction'), zigzag.get('confidence'), zigzag.get('zigZagScore'), 16, anti_bias)
    if skip.get('isSkipPattern'):
        _add_vote(scores, pattern_predictions, 'skipPattern', skip.get('nextPrediction') or ('SMALL' if latest_result == 'BIG' else 'BIG'), skip.get('confidence'), skip.get('skipScore') * 10, 12, anti_bias)
    if cycle.get('isCycle'):
        _add_vote(scores, pattern_predictions, 'cyclePattern', cycle.get('nextPrediction'), cycle.get('confidence'), cycle.get('cycleScore') * 10, 10, anti_bias)
    if long_pat.get('isLongPattern'):
        _add_vote(scores, pattern_predictions, 'longPattern', long_pat.get('lastCategory'), long_pat.get('confidence'), long_pat.get('longPatternScore') * 10, 10, anti_bias)

    prediction = 'BIG' if scores['BIG'] > scores['SMALL'] else 'SMALL'
    dominant = max(scores.values()) or 1
    confidence = max(55, min(88, abs(scores['BIG'] - scores['SMALL']) / dominant * 100))

    recent10 = results[:10]
    if recent10:
        big_ratio = sum(1 for row in recent10 if row.get('category') == 'BIG') / len(recent10)
        if prediction == 'BIG' and big_ratio > 0.80:
            prediction = 'SMALL'
            confidence = max(confidence - 10, 60)
        elif prediction == 'SMALL' and big_ratio < 0.20:
            prediction = 'BIG'
            confidence = max(confidence - 10, 60)

    lr = state['user'].get('lossRecovery', {})
    cons_loss = max(lr.get('consecutiveLosses', 0), win_loss.get('consecutiveLosses', 0))
    loss_signal = _free_loss_signal(entries)
    if loss_signal.get('prediction') in ('BIG', 'SMALL'):
        prediction = loss_signal['prediction']
        confidence = max(confidence, loss_signal.get('confidence', 0))
    elif cons_loss == 1:
        confidence = min(confidence, 82)

    risk = _free_risk_assessment(
        confidence,
        scores,
        pattern_predictions,
        loss_signal,
        entries,
        regime,
    )
    decision = {
        'lossSignal': loss_signal,
        'risk': risk,
        'scores': {key: round(value, 3) for key, value in scores.items()},
        'regime': regime,
        'recentAccuracy': accuracy.get('recentAccuracy', 0),
    }
    state['freeDecision'] = decision
    if risk.get('skip'):
        return {
            'period': get_current_period_1min(),
            'prediction': None,
            'status': 'SKIP',
            'confidence': 0,
            'actual': None,
            'number': None,
            'patternUsed': 'free_risk_guard',
            'patternPredictions': pattern_predictions,
            'timestamp': int(time.time()),
            'skipped': True,
            'skipReason': (
                f"Free risk guard ({risk['score']}% {risk['level']}): "
                + '; '.join(risk.get('reasons', [])[:3])
            ),
            'decision': decision,
        }

    return {
        'period': get_current_period_1min(),
        'prediction': prediction,
        'status': 'Pending',
        'confidence': round(confidence, 2),
        'actual': None,
        'number': None,
        'patternUsed': 'free_ensemble',
        'patternPredictions': pattern_predictions,
        'timestamp': int(time.time()),
        'skipped': False,
        'skipReason': '',
        'decision': decision,
    }


def _stats(history):
    wins = sum(1 for row in history if row.get('status') == 'WIN')
    losses = sum(1 for row in history if row.get('status') == 'LOSS')
    pending = sum(1 for row in history if row.get('status') == 'Pending')
    skipped = sum(
        1 for row in history
        if row.get('status') == 'SKIP' or row.get('skipped')
    )
    settled = wins + losses
    win_rate = round((wins / settled) * 100, 2) if settled else 0
    recent = [row for row in history if row.get('status') in ('WIN', 'LOSS')][:10]
    recent_wins = sum(1 for row in recent if row.get('status') == 'WIN')
    recent_accuracy = round((recent_wins / len(recent)) * 100, 2) if recent else 0
    streak_status = None
    streak = 0
    for row in history:
        if row.get('status') not in ('WIN', 'LOSS'):
            continue
        if streak_status is None:
            streak_status = row['status']
            streak = 1
        elif row['status'] == streak_status:
            streak += 1
        else:
            break
    return {
        'pending': pending,
        'skipped': skipped,
        'streak': f"{streak} {streak_status or 'None'}",
        'totalLosses': losses,
        'totalPredictions': len(history),
        'settledPredictions': settled,
        'totalWins': wins,
        'winRate': win_rate,
        'accuracy': win_rate,
        'recentAccuracy': recent_accuracy,
        'source': 'free_prediction_history.csv',
    }


def get_free_payload():
    os.makedirs(DATA_DIR, exist_ok=True)
    with _lock:
        state = _load_state()
        entries = verify_free_pending(_entries(), state)
        current_period = get_current_period_1min()
        current = next((e for e in entries if e.get('period') == current_period), None)
        if not current:
            current = _build_prediction(entries, state)
            upsert_free_history(current)
            entries = _entries()
        _update_loss_state(state, entries)
        _save_state(state)

    entries.sort(key=lambda row: _period_key(row.get('period')), reverse=True)
    public_history = [
        _public_entry(row) for row in entries[:FREE_HISTORY_LIMIT]
        if not (row.get('skipped') or str(row.get('status', '')).upper() == 'SKIP')
    ]
    decision = current.get('decision') or state.get('freeDecision') or {}

    if not current.get('prediction') or current.get('prediction') not in ('BIG', 'SMALL') or current.get('skipped') or str(current.get('status', '')).upper() == 'SKIP':
        for entry in entries:
            if entry.get('prediction') in ('BIG', 'SMALL') and not entry.get('skipped') and str(entry.get('status', '')).upper() != 'SKIP':
                current = entry
                break
        else:
            current = {'prediction': 'BIG', 'status': 'Pending', 'confidence': 55, 'skipped': False, 'skipReason': '', 'period': get_current_period_1min()}

    return {
        'predictionResult': {
            'period': current.get('period'),
            'prediction': current.get('prediction'),
            'status': current.get('status', 'Pending'),
            'skipped': False,
            'skipReason': '',
        },
        'predictionDetails': {
            'gameType': 'Wingo 1 Min Free',
            'confidence': round(float(current.get('confidence') or 0), 2),
            'actual': current.get('actual'),
            'number': _clean_number(current.get('number')),
            'actualNumber': _clean_number(current.get('number')),
            'color': _number_color(current.get('number')),
            'actualColor': _number_color(current.get('number')),
            'lossRecovery': state['user'].get('lossRecovery', {}),
            'lossSignal': decision.get('lossSignal', {}),
            'lossRisk': decision.get('risk', {}),
            'patternScores': decision.get('scores', {}),
            'marketRegime': decision.get('regime'),
            'recentAccuracy': decision.get('recentAccuracy', 0),
        },
        'stats': _stats(public_history),
        'history': public_history,
        'historySource': {
            'file': 'free_prediction_history.csv',
            'live': True,
            'rows': len(public_history),
            'limit': FREE_HISTORY_LIMIT,
        },
    }
