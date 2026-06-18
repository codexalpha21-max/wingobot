import csv
import json
import os
import threading
import time
import traceback

from helpers import fetch_api_data, fetch_wingobot_daily_history, get_current_period_1min
from ml import get_model_summary, predict_lstm_bilstm, predict_ml, train_model
from storage import load_prediction_history_entries
from free_prediction import load_free_history


BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_HISTORY_CSV = os.path.join(DATA_DIR, 'model', 'model_prediction_history.csv')
MODEL_HISTORY_BACKUP_CSV = MODEL_HISTORY_CSV + '.backup'
MODEL_HISTORY_LIMIT = 20
MODEL_CACHE_FILE = os.path.join(DATA_DIR, 'model_cache.json')
MODEL_CACHE_STALE_SECONDS = 120  # serve stale cache up to 2 min
MODEL_BG_REFRESH_INTERVAL = 30  # background refresh every 30 sec

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
_last_feedback_train_period = ''
MODEL_ONLY_EXCLUDED_MODELS = {'RandomForestClassifier', 'ExtraTreesClassifier', 'LogisticRegression'}
_bg_refresh_thread = None
_bg_refresh_lock = threading.Lock()


def _period_key(period):
    try:
        return int(str(period))
    except Exception:
        return 0


def _csv_value(value):
    return '' if value is None else str(value)


def _read_rows(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or 'period' not in reader.fieldnames:
                return []
            return [row for row in reader if row.get('period')]
    except Exception:
        return []


def load_model_history(limit=None):
    global _history_snapshot
    by_period = {}
    for path in (MODEL_HISTORY_BACKUP_CSV, MODEL_HISTORY_CSV):
        for row in _read_rows(path):
            by_period[str(row.get('period', ''))] = row
    rows = list(by_period.values())
    rows.sort(key=lambda row: _period_key(row.get('period')), reverse=True)
    if rows:
        _history_snapshot = [dict(row) for row in rows]
    elif _history_snapshot:
        rows = [dict(row) for row in _history_snapshot]
    return rows[:limit] if limit else rows


def _write_history(rows):
    global _history_snapshot
    os.makedirs(os.path.dirname(MODEL_HISTORY_CSV), exist_ok=True)
    rows = sorted(rows, key=lambda row: _period_key(row.get('period')))
    for path in (MODEL_HISTORY_CSV, MODEL_HISTORY_BACKUP_CSV):
        tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
        try:
            with open(tmp, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=HEADER)
                writer.writeheader()
                for row in rows:
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


def upsert_model_history(entry):
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
        'patternused': entry.get('patternUsed', 'model_ensemble'),
        'timestamp': _csv_value(entry.get('timestamp', int(time.time()))),
        'skipped': '1' if entry.get('skipped') else '0',
        'skipreason': entry.get('skipReason', '') or '',
        'created_at': entry.get('created_at') or time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with _lock:
        rows = load_model_history()
        found = False
        for idx, old in enumerate(rows):
            if str(old.get('period', '')) == period:
                row['created_at'] = old.get('created_at') or row['created_at']
                rows[idx] = row
                found = True
                break
        if not found:
            rows.append(row)
        _write_history(rows)


def _row_to_entry(row):
    try:
        confidence = float(row.get('confidence') or 0)
    except Exception:
        confidence = 0
    try:
        timestamp = int(float(row.get('timestamp') or time.time()))
    except Exception:
        timestamp = int(time.time())
    payload = {
        'period': str(row.get('period', '')),
        'prediction': row.get('prediction') or None,
        'status': row.get('status') or 'Pending',
        'confidence': confidence,
        'actual': row.get('actual') or None,
        'number': row.get('number') or None,
        'patternUsed': row.get('patternused') or 'model_ensemble',
        'timestamp': timestamp,
        'skipped': str(row.get('skipped', '')).lower() in ('1', 'true'),
        'skipReason': row.get('skipreason') or '',
    }
    return payload


def _entries():
    return [_row_to_entry(row) for row in load_model_history()]


def _number(value):
    try:
        return int(float(str(value)))
    except Exception:
        return None


def _color(number):
    number = _number(number)
    if number is None:
        return None
    if number == 0:
        return 'RED,VIOLET'
    if number == 5:
        return 'GREEN,VIOLET'
    return 'GREEN' if number % 2 else 'RED'


def _public_entry(entry):
    number = _number(entry.get('number'))
    return {
        'period': entry.get('period', ''),
        'prediction': entry.get('prediction') or '',
        'status': entry.get('status', 'Pending'),
        'confidence': round(float(entry.get('confidence') or 0), 2),
        'actual': entry.get('actual'),
        'number': number,
        'actualNumber': number,
        'color': _color(number),
        'actualColor': _color(number),
        'skipped': bool(entry.get('skipped', False)),
        'skipReason': entry.get('skipReason') or None,
        'patternUsed': entry.get('patternUsed', 'model_ensemble'),
    }


def _stats(history):
    wins = sum(1 for row in history if row.get('status') == 'WIN')
    losses = sum(1 for row in history if row.get('status') == 'LOSS')
    pending = sum(1 for row in history if row.get('status') == 'Pending')
    skipped = sum(1 for row in history if row.get('status') == 'SKIP')
    settled = wins + losses
    recent = [row for row in history if row.get('status') in ('WIN', 'LOSS')][:10]
    recent_wins = sum(1 for row in recent if row.get('status') == 'WIN')
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
        'winRate': round((wins / settled) * 100, 2) if settled else 0,
        'accuracy': round((wins / settled) * 100, 2) if settled else 0,
        'recentAccuracy': round((recent_wins / len(recent)) * 100, 2) if recent else 0,
        'source': 'model_prediction_history.csv',
    }


def _model_performance(entries):
    settled = [
        row for row in entries
        if row.get('status') in ('WIN', 'LOSS')
        and row.get('prediction') in ('BIG', 'SMALL')
        and row.get('actual') in ('BIG', 'SMALL')
    ]
    performance = {}
    for row in settled:
        model_name = row.get('patternUsed') or 'model_ensemble'
        stats = performance.setdefault(model_name, {
            'wins': 0,
            'losses': 0,
            'total': 0,
            'recentWins': 0,
            'recentTotal': 0,
            'consecutiveLosses': 0,
            'lossWhenPredictingBIG': 0,
            'lossWhenPredictingSMALL': 0,
        })
        stats['total'] += 1
        if row.get('status') == 'WIN':
            stats['wins'] += 1
        else:
            stats['losses'] += 1
            key = (
                'lossWhenPredictingBIG'
                if row.get('prediction') == 'BIG'
                else 'lossWhenPredictingSMALL'
            )
            stats[key] += 1

    for model_name, stats in performance.items():
        recent = [
            row for row in settled
            if (row.get('patternUsed') or 'model_ensemble') == model_name
        ][:20]
        stats['recentTotal'] = len(recent)
        stats['recentWins'] = sum(1 for row in recent if row.get('status') == 'WIN')
        for row in recent:
            if row.get('status') != 'LOSS':
                break
            stats['consecutiveLosses'] += 1
        stats['accuracy'] = round(
            (stats['wins'] / stats['total']) * 100, 2
        ) if stats['total'] else 0
        stats['recentAccuracy'] = round(
            (stats['recentWins'] / stats['recentTotal']) * 100, 2
        ) if stats['recentTotal'] else None
    return performance


def _current_loss_pattern(entries):
    settled = [
        row for row in entries
        if row.get('status') in ('WIN', 'LOSS')
        and row.get('prediction') in ('BIG', 'SMALL')
        and row.get('actual') in ('BIG', 'SMALL')
    ]
    losses = []
    for row in settled:
        if row.get('status') != 'LOSS':
            break
        losses.append(row)

    signal = {
        'consecutiveLosses': len(losses),
        'prediction': None,
        'reason': None,
    }
    if len(losses) < 2:
        return signal

    predictions = [row.get('prediction') for row in losses[:6]]
    actuals = [row.get('actual') for row in losses[:6]]
    same_wrong_side = len(set(predictions)) == 1 and len(set(actuals)) == 1
    alternating_actuals = all(
        actuals[index] != actuals[index - 1]
        for index in range(1, len(actuals))
    )
    if same_wrong_side and predictions[0] != actuals[0]:
        signal.update({
            'prediction': actuals[0],
            'reason': 'repeated_inverse_loss',
        })
    elif alternating_actuals:
        signal.update({
            'prediction': 'SMALL' if actuals[0] == 'BIG' else 'BIG',
            'reason': 'alternating_loss_recovery',
        })
    return signal


def _loss_manager_signal(entries, candidates):
    settled = [
        row for row in entries
        if row.get('status') in ('WIN', 'LOSS')
        and row.get('prediction') in ('BIG', 'SMALL')
        and row.get('actual') in ('BIG', 'SMALL')
    ]
    losses = []
    for row in settled:
        if row.get('status') != 'LOSS':
            break
        losses.append(row)
    signal = {
        'active': False,
        'consecutiveLosses': len(losses),
        'prediction': None,
        'confidenceBoost': 0,
        'reason': '',
        'lossPredictions': {},
        'lossActuals': {},
    }
    if len(losses) < 2:
        return signal

    recent_losses = losses[:10]
    pred_counts = {
        'BIG': sum(1 for row in recent_losses if row.get('prediction') == 'BIG'),
        'SMALL': sum(1 for row in recent_losses if row.get('prediction') == 'SMALL'),
    }
    actual_counts = {
        'BIG': sum(1 for row in recent_losses if row.get('actual') == 'BIG'),
        'SMALL': sum(1 for row in recent_losses if row.get('actual') == 'SMALL'),
    }
    signal['lossPredictions'] = pred_counts
    signal['lossActuals'] = actual_counts

    model_votes = {'BIG': 0.0, 'SMALL': 0.0}
    for row in candidates or []:
        pred = row.get('prediction')
        if pred in model_votes:
            weight = (
                float(row.get('validationAccuracy') or 50) * 0.45
                + float(row.get('confidence') or 50) * 0.35
                + float(row.get('adaptiveScore') or 50) * 0.20
            )
            model_votes[pred] += max(weight, 1)

    if actual_counts['BIG'] > actual_counts['SMALL']:
        recovery_side = 'BIG'
        reason = 'loss_manager_actual_majority_recovery'
    elif actual_counts['SMALL'] > actual_counts['BIG']:
        recovery_side = 'SMALL'
        reason = 'loss_manager_actual_majority_recovery'
    elif pred_counts['BIG'] > pred_counts['SMALL']:
        recovery_side = 'SMALL'
        reason = 'loss_manager_opposite_failed_side'
    elif pred_counts['SMALL'] > pred_counts['BIG']:
        recovery_side = 'BIG'
        reason = 'loss_manager_opposite_failed_side'
    else:
        recovery_side = 'BIG' if model_votes['BIG'] >= model_votes['SMALL'] else 'SMALL'
        reason = 'loss_manager_model_consensus_recovery'

    signal.update({
        'active': True,
        'prediction': recovery_side,
        'confidenceBoost': min(18, 8 + len(losses) * 2),
        'reason': reason,
        'modelVotes': {k: round(v, 2) for k, v in model_votes.items()},
    })
    return signal


def _adaptive_model_decision(ml_prediction, model_entries):
    if not ml_prediction:
        return None
    model_rows = [
        row for row in (ml_prediction.get('modelPredictions') or [])
        if row.get('model') not in MODEL_ONLY_EXCLUDED_MODELS
    ]
    if not model_rows:
        return ml_prediction

    performance = _model_performance(model_entries)
    candidates = []
    for row in model_rows:
        if row.get('prediction') not in ('BIG', 'SMALL'):
            continue
        validation = float(row.get('validationAccuracy') or 50)
        confidence = float(row.get('confidence') or 50)
        live = performance.get(row.get('model'), {})
        live_total = int(live.get('recentTotal') or 0)
        live_accuracy = (
            float(live.get('recentAccuracy'))
            if live.get('recentAccuracy') is not None
            else 50
        )
        live_weight = min(live_total / 8, 1)
        adjusted_live = (live_accuracy * live_weight) + (50 * (1 - live_weight))
        loss_penalty = min(int(live.get('consecutiveLosses') or 0) * 9, 27)
        score = (
            validation * 0.50
            + adjusted_live * 0.35
            + confidence * 0.15
            - loss_penalty
        )
        candidates.append({
            **row,
            'adaptiveScore': round(score, 2),
            'liveAccuracy': live.get('accuracy'),
            'liveRecentAccuracy': live.get('recentAccuracy'),
            'liveSamples': live.get('total', 0),
            'consecutiveLosses': live.get('consecutiveLosses', 0),
        })

    if not candidates:
        return ml_prediction
    selected = max(
        candidates,
        key=lambda row: (
            row.get('adaptiveScore', 0),
            row.get('validationAccuracy') or 0,
            row.get('confidence') or 0,
        ),
    )

    lgbm = next((row for row in candidates if row.get('model') == 'LGBMClassifier'), None)
    xgb = next((row for row in candidates if row.get('model') == 'XGBClassifier'), None)
    cbt = next((row for row in candidates if row.get('model') == 'CatBoostClassifier'), None)
    loss_pattern = _current_loss_pattern(model_entries)
    loss_manager = _loss_manager_signal(model_entries, candidates)
    boost_candidates = [m for m in (lgbm, xgb, cbt) if m is not None]
    boost_ready = bool(
        len(boost_candidates) >= 2
        and int(ml_prediction.get('samples') or 0) >= 40
        and all(
            float(m.get('validationAccuracy') or 0) >= 54
            for m in boost_candidates
        )
    )
    selection_reason = 'adaptive_model_score'
    if boost_ready and loss_pattern['consecutiveLosses'] >= 1:
        big_probability = sum(
            float(m.get('bigProbability') or 50) for m in boost_candidates
        ) / len(boost_candidates)
        boost_model_names = '_'.join(
            m['model'] for m in boost_candidates
        )
        selected = {
            'model': f'{boost_model_names}_Recovery',
            'prediction': 'BIG' if big_probability >= 50 else 'SMALL',
            'confidence': round(min(94, 55 + abs(big_probability - 50)), 2),
            'bigProbability': round(big_probability, 2),
            'validationAccuracy': round(
                sum(
                    float(m.get('validationAccuracy') or 0)
                    for m in boost_candidates
                ) / len(boost_candidates),
                2,
            ),
            'adaptiveScore': round(
                sum(
                    m.get('adaptiveScore', 0) for m in boost_candidates
                ) / len(boost_candidates),
                2,
            ),
        }
        selection_reason = 'boost_ensemble_after_loss'

    if loss_pattern.get('prediction'):
        selected = {
            **selected,
            'prediction': loss_pattern['prediction'],
            'confidence': round(max(float(selected.get('confidence') or 0), 82), 2),
        }
        selection_reason = loss_pattern['reason']
    elif loss_manager.get('active') and loss_manager.get('prediction') in ('BIG', 'SMALL'):
        selected = {
            **selected,
            'prediction': loss_manager['prediction'],
            'confidence': round(min(
                92,
                max(float(selected.get('confidence') or 0), 70)
                + float(loss_manager.get('confidenceBoost') or 0),
            ), 2),
        }
        selection_reason = loss_manager['reason']

    return {
        **selected,
        'selectionReason': selection_reason,
        'lossPattern': loss_pattern,
        'lossManager': loss_manager,
        'modelPerformance': performance,
        'rankedModels': sorted(
            candidates,
            key=lambda row: row.get('adaptiveScore', 0),
            reverse=True,
        ),
    }


def _model_loss_risk(decision, ml_prediction, model_entries):
    risk = 0
    reasons = []
    if not decision or not ml_prediction:
        return {
            'score': 100,
            'level': 'HIGH',
            'skip': True,
            'reasons': ['Model decision is not ready.'],
        }

    samples = int(ml_prediction.get('samples') or 0)
    ranked = decision.get('rankedModels') or []
    validation = float(decision.get('validationAccuracy') or 0)
    confidence = float(decision.get('confidence') or 0)
    big_probability = float(decision.get('bigProbability') or 50)
    live_recent = decision.get('liveRecentAccuracy')
    consecutive_losses = int(decision.get('consecutiveLosses') or 0)
    loss_pattern = decision.get('lossPattern') or {}

    if samples < 10:
        risk += 35
        reasons.append(f'Only {samples} trained samples.')
    elif samples < 30:
        risk += 12
        reasons.append(f'Model is still learning ({samples} samples).')
    if validation < 54:
        risk += 25
        reasons.append(f'Validation accuracy is weak ({round(validation, 2)}%).')
    if confidence < 58:
        risk += 20
        reasons.append(f'Model confidence is weak ({round(confidence, 2)}%).')
    if abs(big_probability - 50) < 7:
        risk += 25
        reasons.append('Probability is too close to 50/50.')
    if live_recent is not None and float(live_recent) < 45:
        risk += 25
        reasons.append(f'Recent live accuracy is low ({round(float(live_recent), 2)}%).')
    if consecutive_losses >= 2:
        risk += min(35, consecutive_losses * 12)
        reasons.append(f'Selected model has {consecutive_losses} consecutive losses.')

    reliable = [
        row for row in ranked
        if float(row.get('validationAccuracy') or 0) >= 54
    ]
    if len(reliable) >= 2:
        sides = {row.get('prediction') for row in reliable[:3]}
        if len(sides) > 1:
            risk += 20
            reasons.append('Validated models disagree on direction.')

    if (
        loss_pattern.get('prediction') in ('BIG', 'SMALL')
        and decision.get('prediction') == loss_pattern.get('prediction')
    ):
        risk = max(0, risk - 25)
        reasons.append('Verified loss-recovery pattern supports this direction.')

    recent_risk_skip = any(
        row.get('skipped')
        or row.get('status') == 'SKIP'
        or row.get('patternUsed') == 'model_risk_guard'
        for row in model_entries[:3]
    )
    latest_shadow = next(
        (
            row for row in model_entries[:5]
            if str(row.get('patternUsed', '')).startswith('SHADOW_')
            and row.get('actual') in ('BIG', 'SMALL')
        ),
        None,
    )
    shadow_prediction = (
        str(latest_shadow.get('patternUsed')).rsplit('_', 1)[-1]
        if latest_shadow else None
    )
    shadow_passed = bool(
        latest_shadow
        and shadow_prediction in ('BIG', 'SMALL')
        and shadow_prediction == latest_shadow.get('actual')
    )
    current_loss_pattern = decision.get('lossPattern') or {}
    model_loss_run = int(current_loss_pattern.get('consecutiveLosses') or 0)
    selected_model_losses = int(decision.get('consecutiveLosses') or 0)
    consecutive_losses = max(model_loss_run, selected_model_losses)
    selection_reason = str(decision.get('selectionReason') or '')
    validated_recovery = bool(
        recent_risk_skip
        and shadow_passed
        and confidence >= 60
        and validation >= 54
        and (
            selection_reason in (
                'boost_ensemble_after_loss',
                'repeated_inverse_loss',
                'alternating_loss_recovery',
            )
            or current_loss_pattern.get('prediction') == decision.get('prediction')
        )
    )
    hard_loss_guard = False
    risk = min(100, risk)
    level = 'HIGH' if risk >= 55 else 'MEDIUM' if risk >= 35 else 'LOW'
    extreme_risk = (
        risk >= 85
        and confidence < 58
        and validation < 54
        and abs(big_probability - 50) < 6
    )
    loss_streak_risk = (
        consecutive_losses >= 3
        and risk >= 75
        and not validated_recovery
        and confidence < 62
    )
    if extreme_risk or loss_streak_risk:
        should_skip = not recent_risk_skip
        if should_skip:
            level = 'HIGH'
            reasons.insert(
                0,
                'Model-only skip: extreme uncertainty after loss analysis.',
            )
        else:
            reasons.append('Skip cooldown active; model prediction is allowed this round.')
    else:
        should_skip = False
        if consecutive_losses >= 2:
            reasons.append('Loss streak detected; model keeps predicting unless risk is extreme.')
    return {
        'score': risk,
        'level': level,
        'skip': should_skip,
        'skipCooldown': recent_risk_skip,
        'hardLossGuard': hard_loss_guard,
        'extremeRisk': extreme_risk,
        'lossStreakRisk': loss_streak_risk,
        'validatedRecovery': validated_recovery,
        'consecutiveLosses': consecutive_losses,
        'shadowPrediction': shadow_prediction,
        'shadowActual': latest_shadow.get('actual') if latest_shadow else None,
        'shadowPassed': shadow_passed,
        'reasons': reasons,
    }


def _analysis_snapshot(learning_rows, current_slice, ml_prediction):
    actuals = [
        row.get('actual') for row in learning_rows
        if row.get('actual') in ('BIG', 'SMALL')
    ]
    recent_actuals = actuals[:20]
    current_cats = [
        row.get('category') for row in current_slice
        if row.get('category') in ('BIG', 'SMALL')
    ]
    source_cats = current_cats[:20] or recent_actuals

    streak = 0
    streak_side = None
    if source_cats:
        streak_side = source_cats[0]
        for cat in source_cats:
            if cat == streak_side:
                streak += 1
            else:
                break

    alternations = 0
    for idx in range(1, len(source_cats)):
        if source_cats[idx] != source_cats[idx - 1]:
            alternations += 1
    alternation_ratio = round(alternations / max(len(source_cats) - 1, 1), 3) if len(source_cats) >= 2 else 0

    big_recent = recent_actuals.count('BIG')
    small_recent = recent_actuals.count('SMALL')
    transitions = {'BIG->BIG': 0, 'BIG->SMALL': 0, 'SMALL->BIG': 0, 'SMALL->SMALL': 0}
    for idx in range(1, len(actuals[:80])):
        prev = actuals[idx]
        cur = actuals[idx - 1]
        key = f"{prev}->{cur}"
        if key in transitions:
            transitions[key] += 1

    losses = [
        row for row in learning_rows
        if row.get('status') == 'LOSS'
        and row.get('prediction') in ('BIG', 'SMALL')
        and row.get('actual') in ('BIG', 'SMALL')
    ][:30]
    loss_bias = {
        'BIG': sum(1 for row in losses if row.get('prediction') == 'BIG'),
        'SMALL': sum(1 for row in losses if row.get('prediction') == 'SMALL'),
    }

    numbers = []
    for row in learning_rows[:80]:
        number = _number(row.get('number'))
        if number is not None:
            numbers.append(number)
    number_frequency = {str(num): numbers.count(num) for num in range(10)}

    visible_model_predictions = [
        row for row in (ml_prediction.get('modelPredictions', []) if ml_prediction else [])
        if row.get('model') not in MODEL_ONLY_EXCLUDED_MODELS
    ]
    best_model = None
    if visible_model_predictions:
        best_model = max(
            visible_model_predictions,
            key=lambda item: (
                item.get('validationAccuracy') if item.get('validationAccuracy') is not None else -1,
                item.get('confidence', 0),
            ),
        )

    return {
        'learnedVerifiedRows': len(learning_rows),
        'currentMarketWindow': len(current_cats),
        'recentActualBig': big_recent,
        'recentActualSmall': small_recent,
        'recentActualBias': 'BIG' if big_recent > small_recent else 'SMALL' if small_recent > big_recent else 'BALANCED',
        'activeStreakSide': streak_side,
        'activeStreakLength': streak,
        'alternationRatio': alternation_ratio,
        'regime': 'ZIGZAG' if alternation_ratio >= 0.7 else 'STREAK' if streak >= 2 else 'MIXED',
        'transitions': transitions,
        'lossBiasByPrediction': loss_bias,
        'numberFrequency': number_frequency,
        'bestModel': best_model,
        'allModelPredictions': visible_model_predictions,
    }


def _learning_rows(model_entries):
    by_period = {}
    for entry in load_prediction_history_entries(limit=None):
        if entry.get('period') and entry.get('status') in ('WIN', 'LOSS'):
            row = dict(entry)
            row['sourceRoute'] = 'v2_predict'
            by_period[str(entry['period'])] = row
    for entry in load_free_history(limit=None):
        if entry.get('period') and entry.get('status') in ('WIN', 'LOSS'):
            row = {
                'period': str(entry.get('period') or ''),
                'prediction': entry.get('prediction') or None,
                'status': entry.get('status'),
                'confidence': entry.get('confidence') or 0,
                'actual': entry.get('actual') or None,
                'number': entry.get('number') or None,
                'patternUsed': entry.get('patternused') or entry.get('patternUsed') or 'free_ensemble',
                'timestamp': int(float(entry.get('timestamp') or time.time())),
                'skipped': str(entry.get('skipped', '')).lower() in ('1', 'true'),
                'skipReason': entry.get('skipreason') or entry.get('skipReason') or '',
                'sourceRoute': 'v2_free',
            }
            if row['period']:
                by_period.setdefault(row['period'], row)
    for entry in model_entries:
        if entry.get('period') and entry.get('status') in ('WIN', 'LOSS'):
            row = dict(entry)
            row['sourceRoute'] = 'model_predict'
            by_period[str(entry['period'])] = row
    rows = list(by_period.values())
    rows.sort(key=lambda row: _period_key(row.get('period')), reverse=True)
    return rows


def _market_training_rows(game_data, source='market_api'):
    rows = []
    if not isinstance(game_data, list):
        return rows
    for item in game_data:
        period = str(item.get('period') or '')
        actual = item.get('category')
        if not period or actual not in ('BIG', 'SMALL'):
            continue
        rows.append({
            'period': period,
            'prediction': None,
            'status': 'TRAINING',
            'confidence': 0,
            'actual': actual,
            'number': item.get('number'),
            'patternUsed': 'daily_1k_history' if source == 'daily_1k_history' else 'market_training',
            'timestamp': int(time.time()),
            'skipped': False,
            'skipReason': '',
            'sourceRoute': source,
        })
    rows.sort(key=lambda row: _period_key(row.get('period')), reverse=True)
    return rows


def _merge_training_rows(verified_rows, market_rows):
    by_period = {}
    for row in market_rows:
        period = str(row.get('period') or '')
        if period:
            by_period[period] = row
    for row in verified_rows:
        period = str(row.get('period') or '')
        if period:
            by_period[period] = row
    rows = list(by_period.values())
    rows.sort(key=lambda row: _period_key(row.get('period')), reverse=True)
    return rows


def _training_source_counts(rows):
    counts = {
        'v2_predict': 0,
        'v2_free': 0,
        'model_predict': 0,
        'market_api': 0,
        'daily_1k_history': 0,
        'unknown': 0,
    }
    for row in rows:
        source = row.get('sourceRoute') or 'unknown'
        counts[source if source in counts else 'unknown'] += 1
    return counts


def verify_model_pending(entries):
    current_period = get_current_period_1min()
    pending = [
        e for e in entries
        if e.get('status') in ('Pending', 'SKIP')
        and e.get('actual') not in ('BIG', 'SMALL')
        and str(e.get('period', '')) < current_period
    ]
    if not pending:
        return entries

    game_data = fetch_api_data(retries=2, timeout=5, bypass_cache=False)
    if not isinstance(game_data, list):
        return entries
    by_period = {str(item.get('period')): item for item in game_data if item.get('period')}
    changed = False
    for entry in entries:
        if (
            entry.get('status') not in ('Pending', 'SKIP')
            or entry.get('actual') in ('BIG', 'SMALL')
            or str(entry.get('period', '')) >= current_period
        ):
            continue
        period = str(entry.get('period', ''))
        match = by_period.get(period)
        if not match:
            suffix = period[-3:]
            match = next((item for item in game_data if str(item.get('period', '')).endswith(suffix)), None)
        if not match:
            continue
        actual = match.get('category')
        number = match.get('number')
        if actual not in ('BIG', 'SMALL'):
            continue
        entry['actual'] = actual
        entry['number'] = number
        entry['status'] = (
            'SKIP'
            if entry.get('skipped') or str(entry.get('patternUsed', '')).startswith('SHADOW_')
            else 'WIN' if entry.get('prediction') == actual else 'LOSS'
        )
        upsert_model_history(entry)
        changed = True
    return _entries() if changed else entries


def _ml_payload(summary):
    samples = int(summary.get('totalSamples') or 0)
    train_cycles = int(summary.get('totalTrainCycles') or 0)
    if samples >= 100 and train_cycles:
        strength = 'strong'
    elif samples >= 40 and train_cycles:
        strength = 'learning'
    elif samples >= 10:
        strength = 'warming'
    else:
        strength = 'not_ready'
    validation_accuracy = {
        name: accuracy
        for name, accuracy in summary.get('validationAccuracy', {}).items()
        if name not in MODEL_ONLY_EXCLUDED_MODELS
    }
    return {
        'trained': train_cycles > 0 and samples > 0,
        'learning': samples > 0,
        'strength': strength,
        'accuracy': summary.get('lastAccuracy') if samples else None,
        'recentAccuracy': summary.get('lastRecentAccuracy') if samples else None,
        'samples': samples,
        'trainCycles': train_cycles,
        'lastTrainTime': summary.get('lastTrainTime'),
        'modelVersion': summary.get('modelVersion'),
        'models': [
            name for name in summary.get('models', [])
            if name not in MODEL_ONLY_EXCLUDED_MODELS
        ],
        'lightgbmAvailable': summary.get('lightgbmAvailable', False),
        'xgboostAvailable': summary.get('xgboostAvailable', False),
        'validationAccuracy': validation_accuracy,
        'validationSamples': summary.get('validationSamples', 0),
    }


def get_model_payload():
    global _last_feedback_train_period, _payload_cache, _payload_cache_time
    now = time.time()
    if _payload_cache and now - _payload_cache_time < _PAYLOAD_CACHE_SECONDS:
        return _payload_cache
    if not _lock.acquire(blocking=False):
        if _payload_cache:
            cached = dict(_payload_cache)
            cached['stale'] = True
            cached['staleReason'] = 'model_refresh_in_progress'
            return cached
    else:
        _lock.release()
    with _lock:
        entries = verify_model_pending(_entries())
        learning_rows = _learning_rows(entries)

        current_period = get_current_period_1min()
        current = next((e for e in entries if e.get('period') == current_period), None)
        game_data = fetch_api_data(retries=2, timeout=5)
        daily_history = fetch_wingobot_daily_history(retries=1, timeout=8, limit=None)
        daily_training_rows = _market_training_rows(daily_history, source='daily_1k_history')
        market_training_rows = _market_training_rows(game_data, source='market_api') + daily_training_rows
        training_rows = _merge_training_rows(learning_rows, market_training_rows)
        training_source_counts = _training_source_counts(training_rows)
        latest_feedback_period = next(
            (
                str(row.get('period'))
                for row in entries
                if row.get('status') in ('WIN', 'LOSS')
            ),
            '',
        )
        force_feedback_train = bool(
            latest_feedback_period
            and latest_feedback_period != _last_feedback_train_period
        )
        train_started = train_model(
            training_rows,
            force=(
                len(training_rows) >= 15
                and (
                    get_model_summary().get('totalSamples', 0) == 0
                    or force_feedback_train
                )
            ),
        )
        if train_started and latest_feedback_period:
            _last_feedback_train_period = latest_feedback_period
        current_slice = []
        if isinstance(game_data, list):
            current_slice = [
                {'category': row.get('category'), 'number': row.get('number')}
                for row in game_data
                if row.get('category') in ('BIG', 'SMALL')
            ]
        if not current_slice and isinstance(daily_history, list):
            current_slice = [
                {'category': row.get('category'), 'number': row.get('number')}
                for row in daily_history[:150]
                if row.get('category') in ('BIG', 'SMALL')
            ]

        summary = get_model_summary()
        ml_prediction = predict_ml(training_rows, current_slice) if current_slice else None
        selected_prediction = _adaptive_model_decision(ml_prediction, entries)
        sequence_prediction = predict_lstm_bilstm(
            training_rows,
            current_slice,
            daily_history if isinstance(daily_history, list) else [],
        ) if current_slice else None
        if (
            (not selected_prediction or int((ml_prediction or {}).get('samples') or 0) < 10)
            and sequence_prediction
            and sequence_prediction.get('ready')
            and sequence_prediction.get('prediction') in ('BIG', 'SMALL')
        ):
            model_rows = sequence_prediction.get('modelPredictions') or []
            selected_prediction = {
                'model': sequence_prediction.get('selectedModel') or 'BiLSTMSequenceModel',
                'prediction': sequence_prediction.get('prediction'),
                'confidence': sequence_prediction.get('confidence', 0),
                'bigProbability': sequence_prediction.get('bigProbability', 50),
                'validationAccuracy': sequence_prediction.get('selectedModelAccuracy') or 50,
                'selectionReason': 'lstm_bilstm_warm_model_fallback',
                'rankedModels': model_rows,
                'modelPerformance': {},
                'lossPattern': sequence_prediction.get('lossLearning', {}),
                'consecutiveLosses': int((sequence_prediction.get('lossLearning') or {}).get('consecutiveLosses') or 0),
                'allModelPredictions': model_rows,
            }
            ml_prediction = {
                'prediction': sequence_prediction.get('prediction'),
                'confidence': sequence_prediction.get('confidence', 0),
                'bigProbability': sequence_prediction.get('bigProbability', 50),
                'mlScore': sequence_prediction.get('confidence', 0),
                'samples': sequence_prediction.get('samples', len(training_rows)),
                'selectedModel': selected_prediction['model'],
                'selectedModelAccuracy': selected_prediction['validationAccuracy'],
                'modelPredictions': model_rows,
                'sourceCounts': sequence_prediction.get('sourceCounts', {}),
                'lossLearning': sequence_prediction.get('lossLearning', {}),
                'mode': 'LSTM_BILSTM_FALLBACK',
            }
        model_ready = bool(
            selected_prediction
            and selected_prediction.get('prediction') in ('BIG', 'SMALL')
            and ml_prediction
            and ml_prediction.get('samples', 0) >= 10
        )
        loss_risk = _model_loss_risk(selected_prediction, ml_prediction, entries)

        can_replace_not_ready = (
            current
            and current.get('status') == 'SKIP'
            and str(current.get('skipReason', '')).startswith(('Model not ready', 'Model warming'))
            and (model_ready or len(training_rows) >= 1000)
        )
        if not current or can_replace_not_ready:
            if model_ready and not loss_risk.get('skip'):
                current = {
                    'period': current_period,
                    'prediction': selected_prediction['prediction'],
                    'status': 'Pending',
                    'confidence': selected_prediction.get('confidence', 0),
                    'actual': None,
                    'number': None,
                    'patternUsed': selected_prediction.get('model', 'model_ensemble'),
                    'timestamp': int(time.time()),
                    'skipped': False,
                    'skipReason': '',
                }
            elif model_ready:
                current = {
                    'period': current_period,
                    'prediction': 'SKIP',
                    'status': 'SKIP',
                    'confidence': 0,
                    'actual': None,
                    'number': None,
                    'patternUsed': (
                        f"SHADOW_{selected_prediction.get('model', 'MODEL')}_"
                        f"{selected_prediction['prediction']}"
                    ),
                    'timestamp': int(time.time()),
                    'skipped': True,
                    'skipReason': (
                        f"Model risk guard ({loss_risk['score']}% {loss_risk['level']}): "
                        + '; '.join(loss_risk.get('reasons', [])[:3])
                    ),
                }
            else:
                current = {
                    'period': current_period,
                    'prediction': 'SKIP',
                    'status': 'SKIP',
                    'confidence': 0,
                    'actual': None,
                    'number': None,
                    'patternUsed': 'model_ensemble',
                    'timestamp': int(time.time()),
                    'skipped': True,
                    'skipReason': (
                        f"Model warming: training from {len(training_rows)} historical draw rows."
                    ),
                }
            upsert_model_history(current)
            entries = _entries()
        analysis_snapshot = _analysis_snapshot(training_rows, current_slice, ml_prediction)

    entries.sort(key=lambda row: _period_key(row.get('period')), reverse=True)
    history = [_public_entry(row) for row in entries[:MODEL_HISTORY_LIMIT]]
    summary = get_model_summary()
    model_accuracy = selected_prediction.get('validationAccuracy') if selected_prediction else None
    if model_accuracy is None:
        model_accuracy = summary.get('lastAccuracy')
    all_model_predictions = [
        row for row in (ml_prediction.get('modelPredictions', []) if ml_prediction else [])
        if row.get('model') not in MODEL_ONLY_EXCLUDED_MODELS
    ]
    payload = {
        'predictionResult': {
            'period': current.get('period'),
            'prediction': current.get('prediction') or '',
            'status': current.get('status', 'Pending'),
            'skipped': current.get('skipped', False),
            'skipReason': current.get('skipReason', '') or '',
        },
        'predictionDetails': {
            'gameType': 'Wingo 1 Min Model',
            'confidence': round(float(current.get('confidence') or 0), 2),
            'actual': current.get('actual'),
            'number': _number(current.get('number')),
            'actualNumber': _number(current.get('number')),
            'color': _color(current.get('number')),
            'actualColor': _color(current.get('number')),
            'modelOnly': True,
            'mlPrediction': ml_prediction,
            'selectedModel': selected_prediction.get('model') if selected_prediction else None,
            'selectedModelAccuracy': model_accuracy,
            'selectedModelPrediction': selected_prediction,
            'selectionReason': selected_prediction.get('selectionReason') if selected_prediction else None,
            'lossRisk': loss_risk,
        },
        'modelDecision': {
            'period': current.get('period'),
            'prediction': current.get('prediction') or '',
            'confidence': round(float(current.get('confidence') or 0), 2),
            'selectedModel': selected_prediction.get('model') if selected_prediction else None,
            'selectedModelAccuracy': model_accuracy,
            'selectedModelPrediction': selected_prediction,
            'selectionReason': selected_prediction.get('selectionReason') if selected_prediction else None,
            'lossPattern': selected_prediction.get('lossPattern') if selected_prediction else None,
            'modelPerformance': selected_prediction.get('modelPerformance') if selected_prediction else {},
            'rankedModels': selected_prediction.get('rankedModels') if selected_prediction else [],
            'lossRisk': loss_risk,
            'allModelPredictions': all_model_predictions,
            'trainedFromRows': len(training_rows),
            'verifiedPredictionRows': len(learning_rows),
            'marketBootstrapRows': len(market_training_rows),
            'dailyHistoryRows': len(daily_training_rows),
            'trainingSourceCounts': training_source_counts,
            'usesFullHistoryForTraining': True,
            'displayHistoryLimit': MODEL_HISTORY_LIMIT,
        },
        'lossAnalysis': {
            'currentPattern': (
                selected_prediction.get('lossPattern') if selected_prediction else None
            ),
            'byModel': (
                selected_prediction.get('modelPerformance') if selected_prediction else {}
            ),
            'selectionReason': (
                selected_prediction.get('selectionReason') if selected_prediction else None
            ),
            'risk': loss_risk,
            'improvementPolicy': [
                'Penalize models with consecutive live losses.',
                'Blend LightGBM and XGBoost after a verified loss when both are validated.',
                'Override repeated inverse-loss patterns.',
                'Prefer live model accuracy plus validation accuracy over validation alone.',
            ],
        },
        'modelAccuracy': {
            'selectedModelAccuracy': model_accuracy,
            'overallAccuracy': summary.get('lastAccuracy'),
            'recentAccuracy': summary.get('lastRecentAccuracy'),
            'validationAccuracy': {
                name: accuracy
                for name, accuracy in summary.get('validationAccuracy', {}).items()
                if name not in MODEL_ONLY_EXCLUDED_MODELS
            },
            'validationSamples': summary.get('validationSamples', 0),
            'samples': summary.get('totalSamples', 0),
            'trainCycles': summary.get('totalTrainCycles', 0),
        },
        'stats': _stats(history),
        'history': history,
        'historySource': {
            'file': 'model_prediction_history.csv',
            'live': True,
            'rows': len(history),
            'limit': MODEL_HISTORY_LIMIT,
        },
        'learning': {
            'source': [
                'prediction_history.csv',
                'free_prediction_history.csv',
                'model_prediction_history.csv',
                'live_market_api',
            ],
            'learnedRows': len(training_rows),
            'verifiedPredictionRows': len(learning_rows),
            'marketBootstrapRows': len(market_training_rows),
            'dailyHistoryRows': len(daily_training_rows),
            'trainingSourceCounts': training_source_counts,
            'modelOnly': True,
            'analysis': analysis_snapshot,
            'deepLearning': {
                'marketPattern': analysis_snapshot['regime'],
                'streakSide': analysis_snapshot['activeStreakSide'],
                'streakLength': analysis_snapshot['activeStreakLength'],
                'alternationRatio': analysis_snapshot['alternationRatio'],
                'lossBiasByPrediction': analysis_snapshot['lossBiasByPrediction'],
                'bestModel': analysis_snapshot['bestModel'],
                'primaryModels': ['LGBMClassifier', 'XGBClassifier', 'CatBoostClassifier'],
                'boostRecovery': analysis_snapshot.get('boostRecovery', False),
                'boostingDepth': 8,
            },
        },
        'ml': _ml_payload(summary),
    }
    _payload_cache = payload
    _payload_cache_time = time.time()
    return payload


def _save_model_cache(payload):
    """Save computed payload to disk cache (atomic write)."""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        tmp = MODEL_CACHE_FILE + '.tmp'
        data = {'timestamp': time.time(), 'payload': payload}
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        os.replace(tmp, MODEL_CACHE_FILE)
    except Exception:
        pass


def _load_model_cache():
    """Load payload from disk cache. Returns (payload, age_seconds) or (None, None)."""
    if not os.path.exists(MODEL_CACHE_FILE):
        return None, None
    try:
        with open(MODEL_CACHE_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        age = time.time() - float(data.get('timestamp', 0))
        return data.get('payload'), age
    except Exception:
        return None, None


def get_cached_model_payload():
    """
    Fast path: read from disk cache.
    - If cache is fresh (< MODEL_CACHE_STALE_SECONDS), return it directly.
    - If cache is stale, return it with stale=True flag so client knows.
    - If no cache exists at all, fall back to live compute (first-boot only).
    Always triggers a background refresh if cache is older than BG_REFRESH_INTERVAL.
    """
    payload, age = _load_model_cache()

    # Trigger background refresh if cache is old or missing
    if age is None or age > MODEL_BG_REFRESH_INTERVAL:
        _ensure_bg_refresh()

    if payload is None:
        # First boot — no cache yet, compute live and save
        live = get_model_payload()
        _save_model_cache(live)
        return live

    if age <= MODEL_CACHE_STALE_SECONDS:
        return payload

    # Cache too old — still return it with stale flag rather than blocking
    stale_payload = dict(payload)
    stale_payload['stale'] = True
    stale_payload['staleReason'] = f'cache_age_{int(age)}s_bg_refresh_pending'
    return stale_payload


def _bg_refresh_worker():
    """Background thread: compute get_model_payload() and save to cache."""
    global _bg_refresh_thread
    try:
        payload = get_model_payload()
        _save_model_cache(payload)
    except Exception as exc:
        print(f'[MODEL_BG] refresh error: {exc}\n{traceback.format_exc()}')
    finally:
        with _bg_refresh_lock:
            _bg_refresh_thread = None


def _ensure_bg_refresh():
    """Spawn a background refresh thread if one is not already running."""
    global _bg_refresh_thread
    with _bg_refresh_lock:
        if _bg_refresh_thread is not None and _bg_refresh_thread.is_alive():
            return
        t = threading.Thread(target=_bg_refresh_worker, daemon=True, name='model_bg_refresh')
        _bg_refresh_thread = t
        t.start()


def start_model_bg_refresh_loop():
    """Called once on app startup. Runs a periodic refresh loop in background."""
    def _loop():
        while True:
            try:
                payload = get_model_payload()
                _save_model_cache(payload)
            except Exception as exc:
                print(f'[MODEL_BG] loop error: {exc}')
            time.sleep(MODEL_BG_REFRESH_INTERVAL)
    t = threading.Thread(target=_loop, daemon=True, name='model_bg_refresh_loop')
    t.start()
    print('[MODEL_BG] Background refresh loop started (every 30s)')
