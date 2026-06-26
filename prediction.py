import time
import math
import json
import os
from config import *
from helpers import *
from analyzers import *
from storage import *
from ml import train_model, predict_ml, get_ml_accuracy, predict_lstm_bilstm
from autolearn import detect_active_pattern, load_memory, save_memory, match_old_pattern, auto_correct, get_accuracy, build_prediction, update_from_verification
from model_brain import learn_all, load_brain, brain_think, brain_learn_from_result

def import_game_history(all_predictions):
    cache_file = os.path.join(DATA_DIR, 'game_history_cache.json')
    now = int(time.time())
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache = json.load(f)
            if cache and cache.get('timestamp', 0) > now - 10:
                return 0
        except Exception:
            pass
    data = fetch_game_history_raw()
    if 'error' in data:
        return 0
    with open(cache_file, 'w') as f:
        json.dump({'timestamp': now}, f)
    if not isinstance(data, list) or len(data) == 0:
        return 0
    existing = {}
    for p in all_predictions:
        if p.get('period'):
            existing[p['period']] = True
    to_add = []
    for entry in data:
        per = entry['period']
        if per and per not in existing and entry.get('category') in ('BIG', 'SMALL'):
            existing[per] = True
            to_add.append(entry)
    if not to_add:
        return 0
    for entry in to_add:
        pred_entry = {            'period': entry['period'],            'prediction': entry['category'],            'status': 'WIN',            'confidence': 100,            'actual': entry['category'],            'number': entry['number'],            'patternUsed': 'imported',            'timestamp': entry.get('timestamp', now),            'skipped': False,            'skipReason': '',        }
        all_predictions.insert(0, pred_entry)
        append_prediction_csv(pred_entry)
    all_predictions[:] = all_predictions[:MAX_PREDICTIONS_CSV]
    try:
        train_model(all_predictions, force=True)
    except Exception:
        pass
    return len(to_add)

def update_loss_recovery_state(user_states, all_predictions):
    lr = user_states['user']['lossRecovery']
    recent = []
    recent_full = []
    for entry in _own_verified_rows(all_predictions):
        if entry.get('status') in ('WIN', 'LOSS'):
            recent.append(entry['status'])
            recent_full.append(entry)
            if len(recent) >= 20:
                break
    lr['lastFiveResults'] = recent[:5]
    lr['lastTwentyResults'] = recent[:20]
    cons_loss = 0
    for r in recent:
        if r == 'LOSS':
            cons_loss += 1
        else:
            break
    cons_win = 0
    for r in recent:
        if r == 'WIN':
            cons_win += 1
        else:
            break
    lr['consecutiveLosses'] = cons_loss
    lr['consecutiveWins'] = cons_win
    loss_pattern = []
    for e in recent_full[:5]:
        if e.get('status') == 'LOSS':
            loss_pattern.append({'prediction': e.get('prediction'), 'actual': e.get('actual')})
    lr['lossPattern'] = loss_pattern
    lr['lossGuardActive'] = cons_loss >= 1
    lr['lossGuardReason'] = (
        f"Active loss: {cons_loss} consecutive. Preventing back-to-back."
        if cons_loss >= 1 else ''
    )
    if cons_loss >= 2 and not lr.get('recoveryMode'):
        lr['recoveryMode'] = True
        lr['recoveryModeStart'] = int(time.time())
        lr['forcedFlipActive'] = True
        lr['forcedFlipCount'] = 0
        if loss_pattern:
            last_wrong = loss_pattern[0]
            lr['recoveryDirection'] = last_wrong.get('actual')
    if lr.get('recoveryMode'):
        if cons_win >= 1:
            lr['recoveryMode'] = False
            lr['forcedFlipActive'] = False
            lr['forcedFlipCount'] = 0
            lr['lossGuardActive'] = False
            lr['lossGuardReason'] = ''
            lr['recoveryDirection'] = None



def build_loss_learning_log(pattern_stats, win_loss_tracker, own_eval, correction, confidence):
    weak_patterns = []
    for pat, stats in pattern_stats.items():
        recent_total = stats.get('recentTotal', 0)
        if recent_total <= 0:
            continue
        recent_rate = round((stats.get('recentWins', 0) / recent_total) * 100, 2)
        if stats.get('consecutiveLosses', 0) >= 2 or (recent_total >= 3 and recent_rate < 40):
            weak_patterns.append({
                'pattern': pat,
                'consecutiveLosses': stats.get('consecutiveLosses', 0),
                'recentWinRate': recent_rate,
                'recentTotal': recent_total,
            })
    weak_patterns.sort(key=lambda p: (p['consecutiveLosses'], -p['recentTotal']), reverse=True)
    return {
        'consecutiveLosses': win_loss_tracker.get('consecutiveLosses', 0),
        'consecutiveWins': win_loss_tracker.get('consecutiveWins', 0),
        'ownWinRate': own_eval.get('winRate', 0),
        'ownVerified': own_eval.get('totalVerified', 0),
        'correction': correction,
        'confidenceAfterLearning': round(confidence, 2),
        'weakPatterns': weak_patterns[:5],
    }


def build_recovery_stats(win_loss_tracker, confidence, consensus_ratio, loss_signal, market_signal):
    cons_loss = win_loss_tracker.get('consecutiveLosses', 0)
    active = cons_loss >= 2
    signal = loss_signal or market_signal or {}
    signal_conf = signal.get('confidence', 0) if isinstance(signal, dict) else 0
    recovery_score = 0
    if active:
        recovery_score = min(
            95,
            round((min(confidence, 90) * 0.45) + (consensus_ratio * 100 * 0.25) + (signal_conf * 0.30), 2),
        )
    return {
        'active': active,
        'mode': 'HIGH_QUALITY_ONLY' if active else 'NORMAL',
        'consecutiveLosses': cons_loss,
        'targetWinChance': 90 if active else None,
        'estimatedRecoveryScore': recovery_score,
        'signal': signal.get('reason') if isinstance(signal, dict) else None,
        'signalConfidence': signal_conf,
        'confidence': round(confidence, 2),
        'consensus': round(consensus_ratio * 100, 2),
    }





def brain_inverse_guard(prediction, confidence, all_predictions, consecutive_losses=0):
    """Check brain knowledge for learned patterns and adjust prediction based on
    what the brain has learned from historical data — no hardcoded flips."""
    try:
        from model_brain import get_model_knowledge
        brain = get_model_knowledge('brain') or {}
    except Exception:
        return prediction, confidence, {'applied': False, 'reason': 'brain_unavailable'}
    verified = _own_verified_rows(all_predictions)
    recent = [r.get('actual') for r in verified[:8] if r.get('actual') in ('BIG', 'SMALL')]
    if len(recent) < 2:
        return prediction, confidence, {'applied': False, 'reason': 'too_few_actuals'}
    # Build pattern keys from reversed recent (most recent first)
    alt_len = 0
    for i in range(1, len(recent)):
        if recent[i] != recent[i - 1]:
            alt_len += 1
        else:
            break
    alt_key = f"alt_{alt_len + 1}" if alt_len >= 2 else None
    learned_pred = None
    learned_conf = 0
    source = None
    if alt_key and brain.get('alternating', {}).get(alt_key, {}).get('learnedPrediction'):
        alt_data = brain['alternating'][alt_key]
        learned_pred = alt_data['learnedPrediction']
        learned_conf = alt_data['confidence']
        source = f'brain_alternating_{alt_key}'
    # Check consecutive patterns if no alternating match
    if not learned_pred:
        for length in [4, 3, 2]:
            if len(recent) >= length:
                key = f"cons_{recent[0]}_{length}"
                cons_data = brain.get('consecutive', {}).get(key, {})
                if cons_data.get('learnedPrediction'):
                    learned_pred = cons_data['learnedPrediction']
                    learned_conf = cons_data['confidence']
                    source = f'brain_consecutive_{key}'
                    break
    # Check reversal patterns if still no match
    if not learned_pred and len(recent) >= 3:
        rev_key = f"rev_{recent[1]}_to_{recent[0]}"
        rev_data = brain.get('reversal', {}).get(rev_key, {})
        if rev_data.get('learnedPrediction'):
            learned_pred = rev_data['learnedPrediction']
            learned_conf = rev_data['confidence']
            source = f'brain_reversal_{rev_key}'
    # Check sequence patterns if still no match
    if not learned_pred and len(recent) >= 2:
        seq_key = '_'.join(recent[:2])
        for slen in [2, 3, 4]:
            seq_info = brain.get(f'seq{slen}', {})
            if seq_key in seq_info:
                sd = seq_info[seq_key]
                if sd.get('total', 0) >= 3:
                    learned_pred = 'BIG' if sd.get('toBig', 0) > sd.get('toSmall', 0) else 'SMALL'
                    learned_conf = round(abs(sd.get('toBig', 0.5) - 0.5) * 200, 2)
                    source = f'brain_sequence_{slen}_{seq_key}'
                    break
    if not learned_pred:
        return prediction, confidence, {'applied': False, 'reason': 'no_learned_pattern'}
    # Stronger override when consecutive losses are high
    loss_amplifier = 1 + (consecutive_losses * 0.4)
    effective_conf = learned_conf * loss_amplifier
    if learned_pred == prediction:
        # Same direction — just boost confidence
        new_conf = min(94, max(confidence, confidence + effective_conf * 0.4))
        return prediction, new_conf, {
            'applied': True, 'reason': f'{source}_confirm',
            'brainConfidence': learned_conf, 'learnedPrediction': learned_pred,
        }
    # Learned prediction differs — apply with strength
    if consecutive_losses >= 1:
        # During losses, brain is authoritative unless it has very low confidence
        if learned_conf < 5 and consecutive_losses < 3:
            return prediction, confidence, {'applied': False, 'reason': 'brain_confidence_too_low'}
        new_conf = min(94, max(confidence + effective_conf * 0.6, 72))
        return learned_pred, new_conf, {
            'applied': True, 'reason': source + '_override_on_loss',
            'brainConfidence': learned_conf, 'learnedPrediction': learned_pred,
        }
    new_conf = min(94, max(confidence, confidence + effective_conf * 0.5))
    return learned_pred, new_conf, {
        'applied': True, 'reason': source,
        'brainConfidence': learned_conf, 'learnedPrediction': learned_pred,
    }


def _detect_alternating_pattern(all_predictions, depth=6, tolerance=1):
    """Check if recent actuals form a mostly alternating BIG/SMALL pattern.
    tolerance = max number of allowed repeats (same adjacent values)."""
    verified = _own_verified_rows(all_predictions)
    actuals = [r.get('actual') for r in verified[:depth] if r.get('actual') in ('BIG', 'SMALL')]
    if len(actuals) < 3:
        return False
    repeats = sum(1 for i in range(1, len(actuals)) if actuals[i] == actuals[i - 1])
    return repeats <= tolerance


def apply_loss_recovery(user_states, prediction, confidence, win_loss_tracker, explanation, all_predictions=None):
    """Let model_brain's learned knowledge guide recovery — no hardcoded flips."""
    cons_loss = win_loss_tracker.get('consecutiveLosses', 0)
    if cons_loss == 0:
        return prediction, confidence, ''
    # Boost confidence when losses mount — prediction stays from brain/ML
    boost = min(18, cons_loss * 4)
    confidence = min(confidence + boost, 94)
    # At 3+ consecutive losses, check if market is consistently producing one side
    if cons_loss >= 3 and all_predictions:
        verified = _own_verified_rows(all_predictions)
        recent_actuals = [r.get('actual') for r in verified[:cons_loss+2] if r.get('actual') in ('BIG', 'SMALL')]
        if len(recent_actuals) >= 4:
            last_actual = recent_actuals[0]
            same_side_count = sum(1 for a in recent_actuals[:cons_loss] if a == last_actual)
            # If all recent losses show market strongly biased toward one side
            if same_side_count >= min(cons_loss - 1, 2) and last_actual != prediction:
                confidence = min(94, confidence + 5)
                return last_actual, confidence, f' loss_trend_follow_{last_actual}_{cons_loss}loss'
    return prediction, confidence, f' recovery_confidence_boost_{cons_loss}loss'


def calculate_loss_psychological_profile(all_predictions, win_loss_tracker):
    """Build a psychological profile of our prediction behavior vs market reality."""
    verified = _own_verified_rows(all_predictions)
    if len(verified) < 3:
        return {}
    profile = {
        'bigPredictions': 0, 'smallPredictions': 0,
        'bigWins': 0, 'smallWins': 0,
        'bigLosses': 0, 'smallLosses': 0,
        'consecutiveBigLosses': 0, 'consecutiveSmallLosses': 0,
        'bigToSmallLosses': 0, 'smallToBigLosses': 0,
    }
    cons_big_loss = 0
    cons_small_loss = 0
    for e in verified:
        pred = e.get('prediction')
        actual = e.get('actual')
        status = e.get('status')
        if pred == 'BIG':
            profile['bigPredictions'] += 1
            if status == 'WIN':
                profile['bigWins'] += 1
                cons_big_loss = 0
            else:
                profile['bigLosses'] += 1
                cons_big_loss += 1
                if actual == 'SMALL':
                    profile['bigToSmallLosses'] += 1
        elif pred == 'SMALL':
            profile['smallPredictions'] += 1
            if status == 'WIN':
                profile['smallWins'] += 1
                cons_small_loss = 0
            else:
                profile['smallLosses'] += 1
                cons_small_loss += 1
                if actual == 'BIG':
                    profile['smallToBigLosses'] += 1
    profile['consecutiveBigLosses'] = cons_big_loss
    profile['consecutiveSmallLosses'] = cons_small_loss
    profile['bigWinRate'] = round((profile['bigWins'] / max(profile['bigPredictions'], 1)) * 100, 2)
    profile['smallWinRate'] = round((profile['smallWins'] / max(profile['smallPredictions'], 1)) * 100, 2)
    return profile


def brain_like_deep_recovery(all_predictions, results, prediction, confidence, win_loss_tracker):
    """Deep psychological recovery: analyze WHY we lost and what the market is doing."""
    cons_loss = win_loss_tracker.get('consecutiveLosses', 0)
    if cons_loss == 0:
        return prediction, confidence, ''
    profile = calculate_loss_psychological_profile(all_predictions, win_loss_tracker)
    cats = [r.get('category') for r in (results or []) if r.get('category') in ('BIG', 'SMALL')]
    if not cats:
        return prediction, confidence, ''
    # Analyze the last N actual results for transition patterns
    win_loss_learnt = {'BIG': 0.0, 'SMALL': 0.0}
    for i in range(1, min(len(cats), 8)):
        if cats[i] == 'BIG':
            win_loss_learnt['SMALL'] += 0.5 ** i
        else:
            win_loss_learnt['BIG'] += 0.5 ** i
    flip_threshold = 1 - (0.5 ** cons_loss)
    # Use profile to determine if we have a directional bias problem
    big_accuracy_issue = profile.get('bigWinRate', 50) < 40 and profile.get('bigPredictions', 0) >= 3
    small_accuracy_issue = profile.get('smallWinRate', 50) < 40 and profile.get('smallPredictions', 0) >= 3
    if cons_loss >= 2:
        last_pred = win_loss_tracker.get('lastPrediction')
        last_actual = win_loss_tracker.get('lastActual')
        if last_pred == 'BIG' and last_actual == 'SMALL':
            if big_accuracy_issue:
                win_loss_learnt['SMALL'] += 2.0
            else:
                win_loss_learnt[last_actual] += 1.5
        elif last_pred == 'SMALL' and last_actual == 'BIG':
            if small_accuracy_issue:
                win_loss_learnt['BIG'] += 2.0
            else:
                win_loss_learnt[last_actual] += 1.5
        rec_trend = get_recent_actual_trend(all_predictions)
        if rec_trend and rec_trend in ('BIG', 'SMALL'):
            win_loss_learnt[rec_trend] += 1.0
    final_pred = 'BIG' if win_loss_learnt['BIG'] >= win_loss_learnt['SMALL'] else 'SMALL'
    edge = abs(win_loss_learnt['BIG'] - win_loss_learnt['SMALL']) / max(sum(win_loss_learnt.values()), 0.01)
    boost = min(cons_loss * 4, 20)
    recover_confidence = min(92, max(66, confidence + boost + edge * 30))
    if final_pred != prediction:
        return final_pred, recover_confidence, f' brain_recovery_flip_{cons_loss}loss'
    return final_pred, recover_confidence, f' brain_confirms_{cons_loss}loss'


def self_learning_mistake_profile(all_predictions):
    """Track every mistake to learn and self-improve like a brain."""
    verified = _own_verified_rows(all_predictions)
    if len(verified) < 5:
        return {'learntPatterns': {}, 'mistakeAvoidance': 0}
    mistake_patterns = {}
    for i in range(len(verified) - 1):
        curr = verified[i]
        nxt = verified[i + 1] if i + 1 < len(verified) else None
        if curr.get('status') == 'LOSS' and nxt:
            key = f"{curr.get('prediction')}_to_{curr.get('actual')}"
            if key not in mistake_patterns:
                mistake_patterns[key] = {'count': 0, 'wins': 0, 'losses': 0, 'nextActuals': []}
            mistake_patterns[key]['count'] += 1
            mistake_patterns[key]['nextActuals'].append(nxt.get('actual'))
            if nxt.get('status') == 'WIN':
                mistake_patterns[key]['wins'] += 1
            else:
                mistake_patterns[key]['losses'] += 1
    best_recovery = None
    best_recovery_rate = 0
    for key, data in mistake_patterns.items():
        if data['count'] >= 2:
            rate = data['wins'] / max(data['count'], 1)
            if rate > best_recovery_rate:
                best_recovery_rate = rate
                most_common_next = max(set(data['nextActuals']), key=data['nextActuals'].count)
                best_recovery = {'pattern': key, 'recoveryRate': round(rate * 100, 2), 'nextBest': most_common_next}
    return {
        'learntPatterns': mistake_patterns,
        'bestRecovery': best_recovery,
        'mistakeAvoidance': round(best_recovery_rate * 100, 2) if best_recovery_rate > 0 else 0,
    }


def _own_verified_rows(all_predictions):
    by_period = {}
    no_period = []
    for entry in all_predictions:
        if entry.get('status') not in ('WIN', 'LOSS'):
            continue
        if str(entry.get('patternUsed') or entry.get('patternused') or '').lower() == 'imported':
            continue
        if entry.get('prediction') not in ('BIG', 'SMALL'):
            continue
        if entry.get('actual') not in ('BIG', 'SMALL'):
            continue
        period = str(entry.get('period') or '')
        if period:
            by_period.setdefault(period, entry)
        else:
            no_period.append(entry)
    rows = list(by_period.values()) + no_period
    rows.sort(
        key=lambda entry: int(str(entry.get('period')))
        if str(entry.get('period', '')).isdigit()
        else int(entry.get('timestamp') or 0),
        reverse=True,
    )
    return rows


def get_recent_actual_trend(all_predictions):
    ver = _own_verified_rows(all_predictions)
    recent = ver[:15]
    if len(recent) < 3:
        return None
    bc = sum(1 for e in recent if e['actual'] == 'BIG')
    sc = len(recent) - bc
    return 'BIG' if bc > sc else 'SMALL'


def get_market_structure_signal(results, all_predictions):
    cats = [r.get('category') for r in (results or []) if r.get('category') in ('BIG', 'SMALL')]
    if len(cats) < 2:
        cats = [
            e.get('actual') for e in all_predictions
            if e.get('status') in ('WIN', 'LOSS') and e.get('actual') in ('BIG', 'SMALL')
        ][:12]
    if len(cats) < 2:
        return None

    streak = 1
    for cat in cats[1:8]:
        if cat == cats[0]:
            streak += 1
        else:
            break
    verified = _own_verified_rows(all_predictions)
    last_verified = verified[0] if verified else {}
    if (
        last_verified.get('status') == 'LOSS'
        and last_verified.get('prediction') == cats[0]
    ):
        return None
    if streak >= 2:
        return {
            'prediction': cats[0],
            'confidence': min(78 + streak * 3, 90),
            'reason': f'same_trend_{streak}',
        }

    alt = sum(1 for i in range(1, min(len(cats), 8)) if cats[i] != cats[i - 1])
    alt_ratio = alt / max(min(len(cats), 8) - 1, 1)
    if len(cats) >= 5 and alt_ratio >= 0.75:
        return {
            'prediction': None,
            'confidence': 0,
            'reason': 'zigzag_detected_let_brain_decide',
        }

    recent = cats[:6]
    big_count = recent.count('BIG')
    small_count = recent.count('SMALL')
    if big_count >= 5:
        return {'prediction': 'BIG', 'confidence': 78, 'reason': 'big_pressure'}
    if small_count >= 5:
        return {'prediction': 'SMALL', 'confidence': 78, 'reason': 'small_pressure'}
    return None


def get_loss_correction_signal(all_predictions):
    verified = _own_verified_rows(all_predictions)
    if not verified:
        return None

    latest = verified[0]
    if latest.get('status') == 'LOSS' and latest.get('actual') in ('BIG', 'SMALL'):
        return {
            'prediction': latest['actual'],
            'confidence': 84,
            'reason': f"latest_loss_follow_actual_{latest['actual'].lower()}",
        }

    consecutive_losses = []
    for entry in verified:
        if entry.get('status') == 'LOSS':
            consecutive_losses.append(entry)
        else:
            break

    recent_losses = [e for e in verified[:8] if e.get('status') == 'LOSS']
    loss_count = max(len(consecutive_losses), len(recent_losses))
    if loss_count < 2:
        return None

    big_wrong = sum(1 for e in recent_losses if e.get('prediction') == 'BIG' and e.get('actual') == 'SMALL')
    small_wrong = sum(1 for e in recent_losses if e.get('prediction') == 'SMALL' and e.get('actual') == 'BIG')
    if big_wrong >= 2 and big_wrong >= small_wrong:
        return {
            'prediction': 'SMALL',
            'confidence': 82 if len(consecutive_losses) >= 2 else 76,
            'reason': f'loss_flip_big_to_small_{big_wrong}',
        }
    if small_wrong >= 2 and small_wrong > big_wrong:
        return {
            'prediction': 'BIG',
            'confidence': 82 if len(consecutive_losses) >= 2 else 76,
            'reason': f'loss_flip_small_to_big_{small_wrong}',
        }

    if len(consecutive_losses) >= 2:
        actuals = [e.get('actual') for e in consecutive_losses[:3]]
        if actuals and actuals.count(actuals[0]) == len(actuals):
            return {
                'prediction': actuals[0],
                'confidence': 80,
                'reason': f'follow_loss_actual_{actuals[0].lower()}',
            }
    return None


def get_inverse_loss_trap_signal(all_predictions, candidate_prediction=None):
    """Detect if all recent losses are inversions (pred != actual).
    Returns detection info only — the model decides what to do."""
    verified = _own_verified_rows(all_predictions)
    if len(verified) < 2:
        return None

    recent = verified[:8]
    loss_run = []
    for entry in recent:
        if entry.get('status') == 'LOSS' and entry.get('prediction') != entry.get('actual'):
            loss_run.append(entry)
        else:
            break
    if len(loss_run) < 2:
        return None

    preds = [entry.get('prediction') for entry in loss_run]
    actuals = [entry.get('actual') for entry in loss_run]
    inverse_count = sum(1 for entry in loss_run if entry.get('prediction') != entry.get('actual'))
    inverse_rate = inverse_count / max(len(loss_run), 1)
    if inverse_rate < 0.85:
        return None

    actuals_set = {a for a in actuals if a in ('BIG', 'SMALL')}
    if len(actuals_set) == 1:
        trap_pred = 'SMALL' if list(actuals_set)[0] == 'BIG' else 'BIG'
    elif candidate_prediction == 'BIG':
        trap_pred = 'SMALL'
    else:
        trap_pred = 'BIG'
    return {
        'detected': True,
        'prediction': trap_pred,
        'confidence': round(65 + inverse_rate * 25, 1),
        'reason': f"inverse_trap_{len(loss_run)}_losses_{inverse_rate:.0%}",
        'lossRun': len(loss_run),
        'inverseRate': inverse_rate,
        'lastPredictions': preds[:6],
        'lastActuals': actuals[:6],
    }


def apply_model_recovery(
    prediction,
    confidence,
    ml_prediction,
    all_predictions,
):
    tracker = analyze_win_loss_tracker(all_predictions)
    consecutive_losses = tracker.get('consecutiveLosses', 0)
    diagnostics = {
        'active': consecutive_losses >= 1,
        'applied': False,
        'consecutiveLosses': consecutive_losses,
        'inputPrediction': prediction,
        'outputPrediction': prediction,
        'reason': 'model_not_ready',
    }
    if not ml_prediction or prediction not in ('BIG', 'SMALL'):
        return prediction, confidence, diagnostics

    samples = int(ml_prediction.get('samples') or 0)
    model_rows = ml_prediction.get('modelPredictions') or []
    # Raise validation threshold during loss streaks to avoid ML overriding brain
    min_validation = 52 + min(consecutive_losses * 2, 10)
    eligible = [
        row for row in model_rows
        if row.get('prediction') in ('BIG', 'SMALL')
        and row.get('validationAccuracy') is not None
        and float(row.get('validationAccuracy') or 0) >= min_validation
    ]
    diagnostics['samples'] = samples
    diagnostics['eligibleModels'] = len(eligible)
    diagnostics['selectedModel'] = ml_prediction.get('selectedModel')
    diagnostics['selectedModelAccuracy'] = ml_prediction.get('selectedModelAccuracy')
    if samples < 20 or not eligible:
        diagnostics['reason'] = 'insufficient_validated_models'
        return prediction, confidence, diagnostics

    votes = {'BIG': 0.0, 'SMALL': 0.0}
    counts = {'BIG': 0, 'SMALL': 0}
    for row in eligible:
        side = row['prediction']
        validation = float(row.get('validationAccuracy') or 50)
        model_confidence = float(row.get('confidence') or 50)
        reliability = max(1.0, validation - 49)
        votes[side] += reliability * (model_confidence / 100)
        counts[side] += 1

    model_side = 'BIG' if votes['BIG'] >= votes['SMALL'] else 'SMALL'
    total_votes = votes['BIG'] + votes['SMALL']
    vote_share = votes[model_side] / max(total_votes, 0.001)
    model_count_share = counts[model_side] / max(len(eligible), 1)
    best_accuracy = max(float(row.get('validationAccuracy') or 0) for row in eligible)
    agreeing_rows = [row for row in eligible if row.get('prediction') == model_side]
    avg_model_confidence = (
        sum(float(row.get('confidence') or 50) for row in agreeing_rows) / len(agreeing_rows)
        if agreeing_rows else 50
    )
    strong = (
        samples >= 30
        and (
            (best_accuracy >= 56 and vote_share >= 0.58)
            or (len(eligible) >= 3 and model_count_share >= 0.67)
        )
    )
    recovery_ready = consecutive_losses >= 1 and strong
    lightgbm_row = next(
        (row for row in model_rows if row.get('model') == 'LGBMClassifier'),
        None,
    )
    xgboost_row = next(
        (row for row in model_rows if row.get('model') == 'XGBClassifier'),
        None,
    )
    # Raise ML override thresholds during loss streaks
    loss_penalty = min(consecutive_losses * 3, 12)
    min_lightgbm_acc = 55 + loss_penalty
    min_lightgbm_conf = 58 + loss_penalty
    min_pair_acc = 54 + loss_penalty
    min_pair_conf = 56 + loss_penalty
    min_samples = 25 + (consecutive_losses * 5) if consecutive_losses >= 2 else 25
    lightgbm_ready = bool(
        lightgbm_row
        and samples >= min_samples
        and lightgbm_row.get('prediction') in ('BIG', 'SMALL')
        and float(lightgbm_row.get('validationAccuracy') or 0) >= min_lightgbm_acc
        and float(lightgbm_row.get('confidence') or 0) >= min_lightgbm_conf
    )
    boost_pair_ready = bool(
        lightgbm_row
        and xgboost_row
        and samples >= max(min_samples, 40)
        and float(lightgbm_row.get('validationAccuracy') or 0) >= min_pair_acc
        and float(xgboost_row.get('validationAccuracy') or 0) >= min_pair_acc
        and float(lightgbm_row.get('confidence') or 0) >= min_pair_conf
        and float(xgboost_row.get('confidence') or 0) >= min_pair_conf
    )
    boost_pair_prediction = None
    boost_pair_big_probability = None
    boost_pair_confidence = None
    if boost_pair_ready:
        lgbm_big_probability = float(lightgbm_row.get('bigProbability') or 50)
        xgb_big_probability = float(xgboost_row.get('bigProbability') or 50)
        boost_pair_big_probability = (lgbm_big_probability + xgb_big_probability) / 2
        boost_pair_prediction = 'BIG' if boost_pair_big_probability >= 50 else 'SMALL'
        boost_pair_confidence = min(
            92,
            50 + abs(boost_pair_big_probability - 50),
        )

    diagnostics.update({
        'modelPrediction': model_side,
        'bestValidationAccuracy': round(best_accuracy, 2),
        'voteShare': round(vote_share * 100, 2),
        'modelAgreement': round(model_count_share * 100, 2),
        'modelConfidence': round(avg_model_confidence, 2),
        'votes': {side: round(value, 3) for side, value in votes.items()},
        'strong': strong,
        'lightgbm': {
            'available': bool(lightgbm_row),
            'ready': lightgbm_ready,
            'prediction': lightgbm_row.get('prediction') if lightgbm_row else None,
            'confidence': lightgbm_row.get('confidence') if lightgbm_row else None,
            'validationAccuracy': lightgbm_row.get('validationAccuracy') if lightgbm_row else None,
        },
        'boostEnsemble': {
            'ready': boost_pair_ready,
            'prediction': boost_pair_prediction,
            'bigProbability': (
                round(boost_pair_big_probability, 2)
                if boost_pair_big_probability is not None else None
            ),
            'confidence': (
                round(boost_pair_confidence, 2)
                if boost_pair_confidence is not None else None
            ),
            'lightgbmProbability': (
                lightgbm_row.get('bigProbability') if lightgbm_row else None
            ),
            'xgboostProbability': (
                xgboost_row.get('bigProbability') if xgboost_row else None
            ),
            'lightgbmAccuracy': (
                lightgbm_row.get('validationAccuracy') if lightgbm_row else None
            ),
            'xgboostAccuracy': (
                xgboost_row.get('validationAccuracy') if xgboost_row else None
            ),
        },
    })

    if consecutive_losses >= 1 and boost_pair_ready:
        output_confidence = min(
            92,
            max(
                60,
                boost_pair_confidence,
                (
                    float(lightgbm_row.get('validationAccuracy') or 54)
                    + float(xgboost_row.get('validationAccuracy') or 54)
                ) / 2,
            ),
        )
        diagnostics.update({
            'applied': boost_pair_prediction != prediction,
            'outputPrediction': boost_pair_prediction,
            'reason': (
                'lightgbm_xgboost_first_loss_recovery'
                if boost_pair_prediction != prediction
                else 'lightgbm_xgboost_confirm_after_loss'
            ),
        })
        return boost_pair_prediction, round(output_confidence, 2), diagnostics

    if consecutive_losses >= 1 and lightgbm_ready:
        lightgbm_side = lightgbm_row['prediction']
        lightgbm_confidence = float(lightgbm_row.get('confidence') or 58)
        lightgbm_accuracy = float(lightgbm_row.get('validationAccuracy') or 55)
        if lightgbm_side != prediction:
            output_confidence = min(
                90,
                max(60, (lightgbm_confidence * 0.60) + (lightgbm_accuracy * 0.40)),
            )
            diagnostics.update({
                'applied': True,
                'outputPrediction': lightgbm_side,
                'reason': 'lightgbm_first_loss_recovery',
            })
            return lightgbm_side, round(output_confidence, 2), diagnostics
        diagnostics.update({
            'outputPrediction': prediction,
            'reason': 'lightgbm_confirms_after_loss',
        })
        return prediction, round(min(90, max(confidence, lightgbm_confidence)), 2), diagnostics

    if recovery_ready and model_side != prediction:
        output_confidence = min(
            90,
            max(62, (avg_model_confidence * 0.55) + (best_accuracy * 0.45)),
        )
        diagnostics.update({
            'applied': True,
            'outputPrediction': model_side,
            'reason': 'validated_model_loss_recovery',
        })
        return model_side, round(output_confidence, 2), diagnostics

    if model_side == prediction and strong:
        diagnostics.update({
            'outputPrediction': prediction,
            'reason': 'validated_model_confirms_prediction',
        })
        return prediction, round(min(92, max(confidence, avg_model_confidence)), 2), diagnostics

    diagnostics['reason'] = 'model_gate_not_met'
    return prediction, confidence, diagnostics


def is_v2_model_ready(ml_prediction):
    if not ml_prediction or ml_prediction.get('prediction') not in ('BIG', 'SMALL'):
        return False
    samples = int(ml_prediction.get('samples') or 0)
    if samples <= 0:
        return False
    selected_model = str(ml_prediction.get('selectedModel') or '')
    if selected_model in ('', 'FallbackRatio'):
        return False
    selected_accuracy = ml_prediction.get('selectedModelAccuracy')
    if selected_accuracy is not None:
        return True
    for row in ml_prediction.get('modelPredictions') or []:
        if (
            row.get('prediction') in ('BIG', 'SMALL')
            and row.get('validationAccuracy') is not None
        ):
            return True
    return selected_model not in ('FallbackRatio',)


def assess_v2_loss_risk(
    confidence,
    consensus_ratio,
    ml_prediction,
    model_recovery,
    fusion_log,
    market_signal,
    loss_signal,
    win_loss_tracker,
    pending_predictions,
):
    risk = 0
    reasons = []
    consecutive_losses = int(win_loss_tracker.get('consecutiveLosses', 0) or 0)

    if confidence < 60:
        risk += 25
        reasons.append(f'Low confidence ({round(confidence, 2)}%).')
    elif confidence < 66:
        risk += 12
        reasons.append(f'Moderate confidence ({round(confidence, 2)}%).')

    if consensus_ratio < 0.52:
        risk += 25
        reasons.append(f'Weak pattern consensus ({round(consensus_ratio * 100, 2)}%).')
    elif consensus_ratio < 0.60:
        risk += 12
        reasons.append(f'Mixed pattern consensus ({round(consensus_ratio * 100, 2)}%).')

    fusion_edge = float((fusion_log or {}).get('edge') or 0)
    if fusion_edge < 0.08:
        risk += 20
        reasons.append('Advanced fusion is close to a tie.')
    elif fusion_edge < 0.16:
        risk += 10
        reasons.append('Advanced fusion edge is narrow.')

    samples = int((ml_prediction or {}).get('samples') or 0)
    model_rows = (ml_prediction or {}).get('modelPredictions') or []
    reliable_models = [
        row for row in model_rows
        if row.get('prediction') in ('BIG', 'SMALL')
        and float(row.get('validationAccuracy') or 0) >= 54
    ]
    model_ready = is_v2_model_ready(ml_prediction)
    if not model_ready:
        risk += 18
        reasons.append(f'ML not ready yet ({samples} trained samples).')
    if len(reliable_models) >= 2:
        reliable_sides = {row.get('prediction') for row in reliable_models}
        if len(reliable_sides) > 1:
            risk += 20
            reasons.append('Validated ML models disagree.')

    boost = (model_recovery or {}).get('boostEnsemble') or {}
    boost_probability = boost.get('bigProbability')
    if boost.get('ready') and boost_probability is not None:
        if abs(float(boost_probability) - 50) < 7:
            risk += 22
            reasons.append('LightGBM/XGBoost average is near 50/50.')

    if (
        market_signal
        and loss_signal
        and market_signal.get('prediction') in ('BIG', 'SMALL')
        and loss_signal.get('prediction') in ('BIG', 'SMALL')
        and market_signal.get('prediction') != loss_signal.get('prediction')
    ):
        risk += 18
        reasons.append('Market and loss-recovery signals conflict.')

    if consecutive_losses >= 1:
        risk += min(24, consecutive_losses * 8)
        reasons.append(f'Active loss streak: {consecutive_losses}.')

    recovery_reason = str((model_recovery or {}).get('reason') or '')
    strong_recovery = recovery_reason in (
        'model_and_inverse_loss_agree',
        'inverse_loss_guard_overrides_model',
        'lightgbm_xgboost_first_loss_recovery',
        'lightgbm_xgboost_confirm_after_loss',
    )
    if strong_recovery:
        risk = max(0, risk - 30)
        reasons.append('Strong validated recovery signal is active.')

    recent_skip = any(
        entry.get('skipped') or str(entry.get('status', '')).upper() == 'SKIP'
        for entry in pending_predictions[:2]
    )
    consecutive_shadow_skips = 0
    for entry in pending_predictions:
        if (
            entry.get('skipped')
            or str(entry.get('status', '')).upper() == 'SKIP'
            or str(entry.get('patternUsed', '')).startswith('SHADOW_')
        ):
            consecutive_shadow_skips += 1
        else:
            break
    latest_shadow = next(
        (
            entry for entry in pending_predictions[:5]
            if str(entry.get('patternUsed', '')).startswith('SHADOW_')
            and entry.get('actual') in ('BIG', 'SMALL')
        ),
        None,
    )
    shadow_prediction = (
        str(latest_shadow.get('patternUsed')).replace('SHADOW_', '', 1)
        if latest_shadow else None
    )
    shadow_passed = bool(
        latest_shadow
        and shadow_prediction in ('BIG', 'SMALL')
        and shadow_prediction == latest_shadow.get('actual')
    )
    risk = min(100, risk)
    level = 'HIGH' if risk >= 55 else 'MEDIUM' if risk >= 35 else 'LOW'
    custom_fallback_mode = not model_ready
    model_ramp_mode = False
    model_has_prediction = bool(
        model_ready
        and ml_prediction
        and ml_prediction.get('prediction') in ('BIG', 'SMALL')
    )
    hard_loss_guard = (
        consecutive_losses >= 2
        and model_ready
        and not model_has_prediction
    )
    model_validated_recovery = bool(
        model_ready
        and
        strong_recovery
        and confidence >= 68
        and shadow_passed
        and (
            (model_recovery or {}).get('strong')
            or ((model_recovery or {}).get('boostEnsemble') or {}).get('ready')
            or (model_recovery or {}).get('inverseLossSignal')
        )
    )
    logic_validated_recovery = bool(
        not model_ready
        and shadow_passed
        and confidence >= 68
        and consensus_ratio >= 0.62
        and fusion_edge >= 0.16
        and (
            (model_recovery or {}).get('inverseLossSignal')
            or (loss_signal and loss_signal.get('prediction') in ('BIG', 'SMALL'))
            or (market_signal and market_signal.get('confidence', 0) >= 78)
        )
    )
    forced_custom_resume = bool(
        consecutive_shadow_skips >= 2
        and confidence >= 60
        and consensus_ratio >= 0.52
        and fusion_edge >= 0.08
    )
    model_agreement = float((model_recovery or {}).get('voteShare') or 0)
    model_accuracy = float((model_recovery or {}).get('bestValidationAccuracy') or 0)
    boost = (model_recovery or {}).get('boostEnsemble') or {}
    boost_probability = boost.get('bigProbability')
    boost_edge = (
        abs(float(boost_probability) - 50) * 2
        if boost_probability is not None else 0
    )
    if model_ready:
        estimated_win_chance = (
            min(confidence, 92) * 0.25
            + min(model_accuracy, 100) * 0.30
            + min(model_agreement, 100) * 0.25
            + min(boost_edge, 100) * 0.20
        )
        recovery_source = 'MODEL'
        recovery_threshold = 64
    else:
        signal_confidence = max(
            float((loss_signal or {}).get('confidence') or 0),
            float((market_signal or {}).get('confidence') or 0),
        )
        estimated_win_chance = (
            min(confidence, 92) * 0.35
            + min(consensus_ratio * 100, 100) * 0.30
            + min(fusion_edge * 100, 100) * 0.20
            + min(signal_confidence, 100) * 0.15
        )
        recovery_source = 'CUSTOM_LOGIC'
        recovery_threshold = 66
    estimated_win_chance = round(min(95, estimated_win_chance), 2)
    recovery_signal_ready = bool(
        estimated_win_chance >= recovery_threshold
        and confidence >= 60
        and consensus_ratio >= 0.52
        and (
            custom_fallback_mode
            or (model_recovery or {}).get('strong')
            or boost.get('ready')
        )
    )
    validated_recovery = (
        model_validated_recovery
        or logic_validated_recovery
        or (forced_custom_resume and recovery_signal_ready)
        or recovery_signal_ready
    )
    should_skip = False
    if model_has_prediction:
        reasons.append('ML ready; model-only prediction is active.')
    elif custom_fallback_mode:
        reasons.append('ML warmup; custom ensemble used while model learns.')

    return {
        'score': risk,
        'level': level,
        'skip': should_skip,
        'strongRecovery': strong_recovery,
        'customFallbackMode': custom_fallback_mode,
        'modelReady': model_ready,
        'validatedRecovery': validated_recovery,
        'modelValidatedRecovery': model_validated_recovery,
        'logicValidatedRecovery': logic_validated_recovery,
        'estimatedWinChance': estimated_win_chance,
        'recoverySource': recovery_source,
        'reasons': reasons,
    }


def advanced_pattern_fusion(pattern_predictions, pattern_stats, all_predictions, base_prediction,
                            base_confidence, market_signal=None, loss_signal=None):
    scores = {'BIG': 0.0, 'SMALL': 0.0}
    diagnostics = {}
    pattern_count = {'BIG': 0, 'SMALL': 0}
    total_reliability = 0.0

    for name, data in pattern_predictions.items():
        pred = data.get('prediction')
        if pred not in ('BIG', 'SMALL'):
            continue
        confidence = float(data.get('confidence', 50) or 50)
        score = float(data.get('score', confidence) or confidence)
        stats = pattern_stats.get(name, {}) if isinstance(pattern_stats, dict) else {}
        total = stats.get('total', 0) or 0
        recent_total = stats.get('recentTotal', 0) or 0
        success = stats.get('successRate', 50) if total >= 3 else 50
        recent_success = (
            (stats.get('recentWins', 0) / recent_total) * 100
            if recent_total >= 3 else success
        )
        # Human-like: weigh recent success more heavily than long-term
        reliability = max(0.20, min(1.8, ((success * 0.30) + (recent_success * 0.70)) / 52))
        if stats.get('consecutiveLosses', 0) >= 2:
            reliability *= 0.30
        if stats.get('consecutiveLosses', 0) >= 3:
            reliability *= 0.20
        # Boost patterns that have been consistently winning recently
        if recent_total >= 5 and recent_success > 65:
            reliability *= 1.25
        elif recent_total >= 5 and recent_success > 55:
            reliability *= 1.10
        vote = (confidence / 100) * (score / 100) * reliability
        scores[pred] += vote
        pattern_count[pred] += 1
        total_reliability += reliability
        diagnostics[name] = {
            'prediction': pred,
            'confidence': round(confidence, 2),
            'score': round(score, 2),
            'reliability': round(reliability, 3),
            'vote': round(vote, 4),
        }

    for signal_name, signal in (('market', market_signal), ('lossPattern', loss_signal)):
        if signal and signal.get('prediction') in ('BIG', 'SMALL'):
            conf = float(signal.get('confidence', 70) or 70)
            weight = 1.35 if signal_name == 'market' else 1.20
            scores[signal['prediction']] += (conf / 100) * weight
            diagnostics[signal_name] = {
                'prediction': signal['prediction'],
                'confidence': conf,
                'reason': signal.get('reason'),
                'vote': round((conf / 100) * weight, 4),
            }

    if scores['BIG'] == 0 and scores['SMALL'] == 0:
        return base_prediction, base_confidence, {'reason': 'base', 'scores': scores, 'patterns': diagnostics}

    total_votes = scores['BIG'] + scores['SMALL']
    edge = abs(scores['BIG'] - scores['SMALL']) / max(total_votes, 0.001)

    # Human-like reasoning: check if majority of patterns agree
    total_patterns = pattern_count['BIG'] + pattern_count['SMALL']
    majority_side = 'BIG' if pattern_count['BIG'] > pattern_count['SMALL'] else 'SMALL'
    majority_ratio = max(pattern_count['BIG'], pattern_count['SMALL']) / max(total_patterns, 1)

    score_winner = 'BIG' if scores['BIG'] >= scores['SMALL'] else 'SMALL'
    prediction = score_winner

    # Human-like: if majority of patterns agree AND score agrees, trust it more
    if majority_ratio >= 0.70 and majority_side == score_winner:
        edge = max(edge, 0.30)

    # Human-like: if score is close but count is decisive, slightly favor count
    if edge < 0.15 and majority_ratio >= 0.65:
        if majority_side != score_winner:
            prediction = majority_side
            edge = 0.15

    confidence = min(92, max(58, 55 + edge * 45))
    if prediction == base_prediction:
        confidence = min(92, max(confidence, base_confidence))
    else:
        confidence = min(88, max(confidence, base_confidence - 3))

    inverse_trap = get_inverse_loss_trap_signal(all_predictions, prediction)
    if inverse_trap:
        prediction = inverse_trap['prediction']
        confidence = max(confidence, inverse_trap['confidence'])
        diagnostics['inverseLossTrap'] = inverse_trap
        return prediction, round(min(confidence, 94), 2), {
            'reason': inverse_trap['reason'],
            'scores': {k: round(v, 4) for k, v in scores.items()},
            'edge': round(edge, 4),
            'patterns': diagnostics,
        }
    return prediction, round(confidence, 2), {
        'reason': 'advanced_pattern_fusion',
        'scores': {k: round(v, 4) for k, v in scores.items()},
        'edge': round(edge, 4),
        'patterns': diagnostics,
    }


def allow_same_prediction(win_loss_tracker, pending_predictions, prediction, all_predictions=None):
    cons_loss = win_loss_tracker.get('consecutiveLosses', 0)
    cons_win = win_loss_tracker.get('consecutiveWins', 0)
    if cons_win >= 2 and win_loss_tracker.get('lastPrediction') == prediction:
        return True
    recent = [e for e in pending_predictions[:5] if e.get('status') in ('WIN', 'LOSS')]
    if not recent:
        recent = pending_predictions[:5]
    same_count = sum(1 for e in recent if e.get('prediction') == prediction)
    if cons_loss >= 2:
        return False
    if cons_loss >= 1:
        return same_count < 2
    return same_count < 4

def evaluate_own_predictions(pending_predictions):
    """Analyze how well our OWN predictions have performed recently."""
    ver = [e for e in pending_predictions if e.get('status') in ('WIN', 'LOSS')]
    total = len(ver)
    wins = sum(1 for e in ver if e['status'] == 'WIN')
    cons_losses = 0
    for e in pending_predictions:
        if e.get('status') == 'LOSS':
            cons_losses += 1
        elif e.get('status') == 'WIN':
            break
    last_preds = [e.get('prediction') for e in pending_predictions[:5] if e.get('prediction')]
    last_actuals = [e.get('actual') for e in pending_predictions[:5] if e.get('actual')]
    return {        'totalVerified': total,        'totalWins': wins,        'winRate': round((wins / total) * 100, 2) if total > 0 else 0,        'consecutiveLosses': cons_losses,        'lastPredictions': last_preds,        'lastActuals': last_actuals,    }

def auto_correct_strategy(pattern_stats, win_loss_tracker, own_eval, all_predictions):
    """Auto-detect problems and return corrected pattern weights."""
    weights = calculate_pattern_weights(pattern_stats, win_loss_tracker)
    cons_loss = own_eval.get('consecutiveLosses', 0)
    wr = own_eval.get('winRate', 50)
    # If 3+ consecutive losses, enter auto-correction
    if cons_loss >= 3:
        # Drop patterns on a losing streak
        for pat, st in pattern_stats.items():
            if st.get('consecutiveLosses', 0) >= 2 and pat in weights:
                weights[pat] = weights[pat] * 0.1
        # Boost the best pattern
        best_pat = max(pattern_stats.items(), key=lambda x: x[1].get('successRate', 0) * x[1].get('total', 0))[0] if pattern_stats else None
        if best_pat and best_pat in weights:
            weights[best_pat] = weights[best_pat] * 3
    # If win rate < 40% over last predictions, reduce confidence
    correction = 'none'
    if cons_loss >= 3:
        correction = f'loss_streak_{cons_loss}'
    elif wr < 40 and own_eval.get('totalVerified', 0) >= 3:
        correction = 'low_win_rate'
    return weights, correction

def rac_algorithm(pattern_predictions, all_predictions, pattern_stats):
    rac_scores = {}
    recent = all_predictions[:50]
    for pattern, pred_data in pattern_predictions.items():
        pp = pred_data['prediction']
        pc = pred_data['confidence']
        stats = pattern_stats.get(pattern, {'wins': 0, 'total': 0, 'successRate': 0, 'recentWins': 0, 'recentTotal': 0})
        hist_acc = stats.get('successRate', 50)
        rec_acc = (stats['recentWins'] / stats['recentTotal']) * 100 if stats.get('recentTotal', 0) > 0 else hist_acc
        pm = pt = 0
        for he in recent:
            if he.get('patternUsed') == pattern:
                pt += 1
                if he.get('status') == 'WIN':
                    pm += 1
        pmr = (pm / pt) * 100 if pt > 0 else rec_acc
        rs = (hist_acc * 0.3) + (rec_acc * 0.4) + (pmr * 0.2) + (min(pc, 95) * 0.1)
        if rec_acc > 60 and pt >= 5:
            rs *= 1.2
        if rec_acc < 45 and pt >= 5:
            rs *= 0.8
        rac_scores[pattern] = {'score': min(100, rs), 'prediction': pp, 'confidence': pc}
    return rac_scores

def calculate_pattern_weights(pattern_stats, win_loss_tracker):
    weights = {}
    for pattern, stats in pattern_stats.items():
        sr = stats.get('successRate', 50)
        rr = (stats['recentWins'] / stats['recentTotal']) * 100 if stats.get('recentTotal', 0) > 0 else sr
        recent_total = stats.get('recentTotal', 0)
        total = stats.get('total', 0)
        cons_loss = stats.get('consecutiveLosses', 0)
        if cons_loss >= 3:
            penalty = 0.05
        elif cons_loss == 2:
            penalty = 0.2
        elif cons_loss == 1:
            penalty = 0.5
        else:
            penalty = 1.0
        if recent_total >= 5 and rr > 65:
            perf = 1.5
        elif recent_total >= 3 and rr > 50:
            perf = 1.0
        elif recent_total >= 3:
            perf = 0.5
        elif total >= 10 and sr > 55:
            perf = 0.8
        else:
            perf = 0.3
        base = sr * 0.3 + rr * 0.7 if recent_total >= 3 else sr
        weights[pattern] = base * penalty * perf
    total_w = sum(weights.values()) or 1
    weights = {k: v for k, v in weights.items() if (v / total_w) > 0.3}
    tw = sum(weights.values()) or 1
    return {k: (v / tw) * 100 for k, v in weights.items()}

def verify_pending_predictions(pending_predictions, all_predictions, user_states):
    current_period = get_current_period_1min()
    need_verify = False
    for entry in pending_predictions:
        if (
            entry['period'] < current_period
            and entry.get('status') in ('Pending', 'SKIP')
            and not entry.get('actual')
        ):
            need_verify = True
            break
    if not need_verify:
        update_loss_recovery_state(user_states, all_predictions)
        return pending_predictions
    game_data = fetch_api_data(retries=2, timeout=5, bypass_cache=False)
    analyze_number_patterns(game_data, user_states) if 'error' not in game_data else None
    updated = False
    for entry in pending_predictions:
        if (
            entry['period'] < current_period
            and entry.get('status') in ('Pending', 'SKIP')
            and not entry.get('actual')
        ):
            matching = None
            per = entry['period']
            # Try external API first
            if 'error' not in game_data:
                by_period = {
                    str(item.get('period', '')): item
                    for item in game_data
                    if item.get('period')
                }
                matching = by_period.get(str(per))
            if not matching and 'error' not in game_data:
                pl3 = str(per)[-3:]
                for item in game_data:
                    if item.get('period') and str(item['period'])[-3:] == pl3:
                        matching = item
                        break
            # Fallback: check all_predictions for known results
            if not matching:
                for ae in all_predictions:
                    if ae['period'] == per and ae.get('actual'):
                        matching = {'category': ae['actual'], 'number': ae.get('number')}
                        break
            if matching:
                ar = matching['category']
                pattern_used = str(entry.get('patternUsed', ''))
                is_shadow = entry.get('skipped') or pattern_used.startswith('SHADOW_')
                entry['status'] = (
                    'SKIP'
                    if is_shadow
                    else 'WIN' if entry.get('prediction') == ar else 'LOSS'
                )
                entry['actual'] = ar
                entry['locked'] = True
                entry['number'] = matching.get('number')
                entry['confidence'] = round(entry.get('confidence', 0), 2)
                try:
                    brain_learn_from_result(ar)
                except Exception:
                    pass
                try:
                    upsert_prediction_history_csv(entry)
                except Exception:
                    pass
                pu = entry.get('patternUsed', 'ensemble')
                if is_shadow:
                    for ae in all_predictions:
                        if ae['period'] == entry['period']:
                            ae['status'] = 'SKIP'
                            ae['actual'] = entry['actual']
                            ae['number'] = entry.get('number')
                            ae['patternUsed'] = pu
                            break
                    updated = True
                    continue
                pk = 'patternStatsAdvanced'
                if entry.get('patternPredictions'):
                    for pat_name, pat_data in entry['patternPredictions'].items():
                        if pat_data.get('prediction') and pat_data['prediction'] != 'NEUTRAL':
                            ps = user_states['user'][pk].setdefault(pat_name, {                                'wins': 0, 'total': 0, 'successRate': 0,                                'recentWins': 0, 'recentTotal': 0, 'consecutiveLosses': 0                            })
                            pat_win = pat_data['prediction'] == ar
                            ps['total'] += 1
                            ps['recentTotal'] += 1
                            if pat_win:
                                ps['wins'] += 1
                                ps['recentWins'] += 1
                                ps['consecutiveLosses'] = 0
                            else:
                                ps['consecutiveLosses'] += 1
                            ps['successRate'] = round((ps['wins'] / ps['total']) * 100, 2) if ps['total'] > 0 else 0
                ps_ens = user_states['user'][pk].setdefault(pu, {                    'wins': 0, 'total': 0, 'successRate': 0,                    'recentWins': 0, 'recentTotal': 0, 'consecutiveLosses': 0                })
                ps_ens['total'] += 1
                ps_ens['recentTotal'] += 1
                if entry['status'] == 'WIN':
                    ps_ens['wins'] += 1
                    ps_ens['recentWins'] += 1
                    ps_ens['consecutiveLosses'] = 0
                else:
                    ps_ens['consecutiveLosses'] += 1
                ps_ens['successRate'] = round((ps_ens['wins'] / ps_ens['total']) * 100, 2) if ps_ens['total'] > 0 else 0
                for ae in all_predictions:
                    if ae['period'] == entry['period']:
                        ae['status'] = entry['status']
                        ae['actual'] = entry['actual']
                        ae['number'] = entry.get('number')
                        ae['patternUsed'] = pu
                        break
                updated = True
            else:
                entry['verifyRetries'] = entry.get('verifyRetries', 0) + 1
                entry['lastVerifyAttempt'] = int(time.time())
                entry['skipReason'] = ''
                for ae in all_predictions:
                    if ae['period'] == entry['period']:
                        ae['status'] = 'Pending'
                        ae['actual'] = None
                        break
                updated = True
    update_loss_recovery_state(user_states, all_predictions)
    if updated:
        save_all_states(user_states, pending_predictions)
        rewrite_predictions_csv(all_predictions)
        try:
            auto_mem = load_memory()
            update_from_verification(all_predictions, auto_mem)
        except Exception:
            pass
        try:
            train_model(all_predictions, force=True)
        except Exception:
            pass
        fs = get_stats(pending_predictions, all_predictions, user_states)
        ad = analyze_prediction_accuracy(all_predictions)
        wlt = analyze_win_loss_tracker(all_predictions)
        append_stats_csv({            'timestamp': int(time.time()),            'date': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),            'totalWins': fs['totalWins'],            'totalLosses': fs['totalLosses'],            'winRate': fs['winRate'],            'accuracy': ad['accuracy'],            'recentAccuracy': ad['recentAccuracy'],            'totalPredictions': ad['totalPredictions'],            'streak': fs['streak'],            'consecutiveLosses': wlt['consecutiveLosses'],            'consecutiveWins': wlt['consecutiveWins'],        })
    return pending_predictions

def get_stats(pending_predictions, all_predictions, user_states=None):
    verified = [e for e in all_predictions if e.get('status') in ('WIN', 'LOSS')]
    tw = sum(1 for e in verified if e['status'] == 'WIN')
    tl = sum(1 for e in verified if e['status'] == 'LOSS')
    tot = tw + tl
    wr = round((tw / tot) * 100, 2) if tot > 0 else 0
    streak = 0
    st = None
    for e in all_predictions:
        if e.get('status') in ('WIN', 'LOSS'):
            if st is None:
                st = e['status']
                streak = 1
            elif st == e['status']:
                streak += 1
            else:
                break
    l10 = [e.get('actual', 'Pending') for e in pending_predictions[:10]]
    user = user_states.get('user', {}) if isinstance(user_states, dict) else {}
    return {        'operation': 'getStats',        'totalWins': tw,        'totalLosses': tl,        'winRate': wr,        'streak': f"{streak} {st or 'None'}",        'lastTen': l10,        'patternStats': user.get('patternStatsAdvanced', {}),        'numberPatterns': user.get('numberPatterns', {}),        'numberRepetition': user.get('numberRepetition', {}),        'transitionMatrix': user.get('transitionMatrix', {}),        'entropyHistory': user.get('entropyHistory', []),        'neuralWeights': user.get('neuralWeights', []),        'bias': user.get('bias', 0.0),        'lossRecovery': user.get('lossRecovery', {}),        'summary': f"Win rate: {wr}%. Streak: {streak} {st or 'None'}."    }

def auto_import_wingobot_history(all_predictions, user_states):
    cache_file = os.path.join(DATA_DIR, 'wingobot_import_cache.json')
    now = int(time.time())
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache = json.load(f)
            if cache and cache.get('timestamp', 0) > now - 15:
                return 0
        except Exception:
            pass
    imported = fetch_wingobot_history()
    if 'error' in imported:
        return 0
    with open(cache_file, 'w') as f:
        json.dump({'timestamp': now}, f)
    if not isinstance(imported, list) or len(imported) == 0:
        return 0
    existing = {}
    for p in all_predictions:
        if p.get('period'):
            existing[p['period']] = True
    to_add = []
    for entry in imported:
        per = entry['period']
        if per not in existing and entry.get('category') in ('BIG', 'SMALL'):
            to_add.append(entry)
            existing[per] = True
    if not to_add:
        return 0
    for entry in to_add:
        pred_entry = {            'period': entry['period'],            'prediction': entry['category'],            'status': 'WIN',            'confidence': 100,            'actual': entry['category'],            'number': entry['number'],            'patternUsed': 'imported',            'timestamp': entry.get('timestamp', now),            'skipped': False,            'skipReason': '',        }
        all_predictions.insert(0, pred_entry)
        append_prediction_csv(pred_entry)
    all_predictions[:] = all_predictions[:MAX_PREDICTIONS_CSV]
    all_results = []
    for p in all_predictions:
        if p.get('actual') and p.get('number') is not None and p['number'] != '':
            try:
                all_results.append({'number': int(p['number']), 'category': p['actual'], 'period': p['period']})
            except (ValueError, TypeError):
                pass
    user = user_states['user']
    default = build_default_user_state()
    user['numberPatterns'] = default['numberPatterns']
    user['numberRepetition'] = default['numberRepetition']
    user['transitionMatrix'] = default['transitionMatrix']
    user['entropyHistory'] = []
    user['neuralWeights'] = [0.0] * 10
    user['bias'] = 0.0
    user['patternStatsAdvanced'] = default['patternStatsAdvanced']
    analyze_number_patterns(all_results, user_states)
    if len(to_add) > 0:
        train_model(all_predictions, force=True)
    return len(to_add)

def analyze_deep_history(all_predictions, latest_category, user_states):
    results = []
    for p in all_predictions:
        if p.get('actual') and p.get('number') is not None and p['number'] != '':
            try:
                results.append({'number': int(p['number']), 'category': p['actual'], 'period': p['period']})
            except (ValueError, TypeError):
                pass
    total = len(results)
    if total < 20:
        return {'prediction': 'NEUTRAL', 'confidence': 50, 'deepScore': 50,                'longTermBigRatio': 0.5, 'historySize': total}
    big_count = sum(1 for r in results if r['category'] == 'BIG')
    big_ratio = big_count / total
    # Multi-window analysis: look at short, medium, and long-term trends
    short = results[:15]
    medium = results[:50]
    long_view = results[:200]
    short_big = sum(1 for r in short if r['category'] == 'BIG')
    medium_big = sum(1 for r in medium if r['category'] == 'BIG')
    long_big = sum(1 for r in long_view if r['category'] == 'BIG')
    short_ratio = short_big / max(len(short), 1)
    medium_ratio = medium_big / max(len(medium), 1)
    long_ratio = long_big / max(len(long_view), 1)
    num_counts = [0] * 10
    for r in results:
        n = int(r['number'])
        if 0 <= n <= 9:
            num_counts[n] += 1
    max_num = num_counts.index(max(num_counts))
    min_num = num_counts.index(min(num_counts))
    deviation = abs(big_ratio - 0.5)
    confidence = min(50 + deviation * 100, 88)
    # Determine direction: use recent trends to detect shifts
    if short_ratio > 0.65 and short_ratio > medium_ratio:
        trend_dir = 'BIG'
        confidence = min(confidence + 10, 90)
    elif short_ratio < 0.35 and short_ratio < medium_ratio:
        trend_dir = 'SMALL'
        confidence = min(confidence + 10, 90)
    elif medium_ratio > 0.60:
        trend_dir = 'BIG'
        confidence = min(confidence + 6, 88)
    elif medium_ratio < 0.40:
        trend_dir = 'SMALL'
        confidence = min(confidence + 6, 88)
    elif long_ratio > 0.55:
        trend_dir = 'BIG'
    elif long_ratio < 0.45:
        trend_dir = 'SMALL'
    else:
        trend_dir = 'BIG' if big_ratio > 0.5 else 'SMALL'
    consistency = 1.0 - abs(short_ratio - medium_ratio)
    score = confidence * (1 + deviation) * (0.8 + consistency * 0.4)
    if total > 500:
        score *= 1.12
    if total > 2000:
        score *= 1.08
    return {        'prediction': trend_dir,        'confidence': confidence,        'deepScore': min(score, 100),        'longTermBigRatio': round(big_ratio, 4),        'shortBigRatio': round(short_ratio, 4),        'mediumBigRatio': round(medium_ratio, 4),        'mostFrequentNumber': max_num,        'leastFrequentNumber': min_num,        'historySize': total,    }

def generate_prediction(user_states, pending_predictions, all_predictions):
    auto_period = get_current_period_1min()
    current_time = int(time.time())
    ml_acc_data = get_ml_accuracy(all_predictions)
    ml_acc = ml_acc_data.get('accuracy', 50) if ml_acc_data else 50
    for entry in pending_predictions:
        if entry['period'] == auto_period:
            return {                'gameType': 'Wingo 1 Min', 'period': auto_period,                'prediction': entry.get('prediction'), 'confidence': entry.get('confidence', 60),                'locked': True, 'skipped': entry.get('skipped', False),                'lossRecovery': user_states['user']['lossRecovery'],                'explanation': 'Cached prediction for this period.',                'timestamp': entry.get('timestamp', current_time),                'mlAccuracy': round(ml_acc, 1) if ml_acc_data and ml_acc_data.get('total', 0) >= 10 else None,                'mlSamples': ml_acc_data.get('total', 0) if ml_acc_data else 0,            }
    game_data = fetch_api_data()
    trend_stats = fetch_trend_statistics()
    trend_stats_analysis = None
    if 'error' not in trend_stats:
        trend_stats_analysis = analyze_trend_statistics(trend_stats)
    latest_result = 'SMALL'
    latest_number = None
    results = []
    if 'error' in game_data or not game_data:
        for e in pending_predictions:
            if e.get('actual'):
                latest_result = e['actual']
                break
    else:
        results = game_data[:150]
        latest_result = results[0].get('category', 'SMALL')
        latest_number = int(results[0]['number']) if results[0].get('number') is not None else None
    wlt = analyze_win_loss_tracker(all_predictions) if results else {'consecutiveWins': 0, 'lastPrediction': None}
    zigzag_analysis = analyze_zigzag_pattern(results) if results else {'isZigZag': False, 'isBroken': False, 'zigZagScore': 0}
    skip_analysis = analyze_skip_pattern(results) if results else {'isSkipPattern': False, 'skipScore': 0}
    trend_analysis = analyze_trend_based(results) if results else {'bigRatio': 0.5, 'movingAverage': 0.5, 'trend': 'NEUTRAL', 'trendScore': 50}
    cycle_analysis = analyze_cycle_pattern(results) if results else {'isCycle': False, 'cycleScore': 0}
    long_pattern_analysis = analyze_long_pattern(results, wlt) if results else {'isLongPattern': False, 'lastCategory': None, 'longPatternScore': 0}
    number_analysis = get_number_based_prediction(latest_number, user_states['user']['numberPatterns'], user_states['user']['numberRepetition'])
    markov_analysis = analyze_markov_chain(results, user_states) if results else {'prediction': None, 'confidence': 50, 'bigProbability': 50, 'markovScore': 50}
    entropy_analysis = analyze_entropy_based(results, user_states) if results else {'prediction': None, 'confidence': 50, 'entropy': 0, 'avgEntropy': 0, 'entropyScore': 50}
    neural_analysis = analyze_neural_network(results, user_states) if results else {'prediction': None, 'confidence': 50, 'output': 0.5, 'neuralScore': 50}
    streak_analysis = analyze_streak_momentum(results) if results else {'prediction': 'NEUTRAL', 'confidence': 50, 'streakScore': 50, 'streakLength': 0}
    markov2_analysis = analyze_markov_2nd_order(results) if results else {'prediction': 'NEUTRAL', 'confidence': 50, 'markov2Score': 50}
    regime_analysis = detect_market_regime(results) if results else {'regime': 'MIXED', 'streakiness': 0.5}
    deep_history_analysis = analyze_deep_history(all_predictions, latest_result, user_states)

    autolearn_memory = load_memory()
    autolearn_detected = detect_active_pattern(results) if results else {'patternType': 'unknown', 'confidence': 0, 'direction': None, 'score': 0, 'details': {}}
    autolearn_pred = build_prediction(autolearn_detected, autolearn_memory, all_predictions)
    autolearn_acc_data = get_accuracy(autolearn_memory)

    ml_prediction = None
    ml_acc = None
    current_slice = []
    if results:
        current_slice = [{'category': r['category'], 'number': r.get('number')} for r in results]
    elif pending_predictions:
        for e in pending_predictions:
            if e.get('actual'):
                current_slice.append({'category': e['actual'], 'number': e.get('number')})
    if current_slice:
        train_model(all_predictions)
        ml_prediction = predict_ml(all_predictions, current_slice)
    try:
        learn_all()
    except Exception:
        pass
    daily_history = fetch_wingobot_daily_history(retries=1, timeout=8, limit=None)
    daily_rows = daily_history if isinstance(daily_history, list) else []
    sequence_prediction = predict_lstm_bilstm(all_predictions, current_slice, daily_rows)
    ml_acc_data = get_ml_accuracy(all_predictions)
    ml_acc = ml_acc_data.get('accuracy', 50) if ml_acc_data else 50
    if not sequence_prediction.get('ready'):
        lstm_big_prob = float(sequence_prediction.get('bigProbability', 50))
        lstm_samples = int(sequence_prediction.get('samples', 0))
        lstm_confidence = max(50, min(70, 50 + abs(lstm_big_prob - 50) * 0.4 + min(lstm_samples / 20, 10)))
        prediction = 'BIG' if lstm_big_prob >= 50 else 'SMALL'
        confidence = lstm_confidence
    else:
        prediction = sequence_prediction['prediction']
        confidence = sequence_prediction['confidence']
    win_loss_tracker = analyze_win_loss_tracker(all_predictions)
    accuracy_analysis = analyze_prediction_accuracy(all_predictions)
    pattern_stats = user_states['user']['patternStatsAdvanced']
    own_eval = evaluate_own_predictions(pending_predictions)
    pattern_weights, correction = auto_correct_strategy(pattern_stats, win_loss_tracker, own_eval, all_predictions)
    anti_bias = get_anti_bias_correction(all_predictions)
    regime = regime_analysis['regime']
    streak_weight = {'STREAK': 35, 'ZIGZAG': 10}.get(regime, 20)
    markov2_weight = {'STREAK': 30, 'ZIGZAG': 15}.get(regime, 22)
    big_score = small_score = 0
    pattern_predictions = {}
    pattern_contributions = {}
    if streak_analysis['prediction'] != 'NEUTRAL':
        sc = streak_analysis['confidence']
        pattern_predictions['streakMomentum'] = {'prediction': streak_analysis['prediction'], 'confidence': sc, 'score': streak_analysis['streakScore']}
        vote = streak_analysis['prediction']
        bm = anti_bias['multiplier'] if (anti_bias['correction'] == 'PENALIZE_BIG' and vote == 'BIG') or (anti_bias['correction'] == 'PENALIZE_SMALL' and vote == 'SMALL') else 1.0
        if vote == 'BIG':
            big_score += streak_weight * sc * bm
        else:
            small_score += streak_weight * sc * bm
        pattern_contributions['streakMomentum'] = {'prediction': vote, 'weight': streak_weight, 'confidence': sc}
    if markov2_analysis['prediction'] != 'NEUTRAL':
        mc = markov2_analysis['confidence']
        pattern_predictions['markov2'] = {'prediction': markov2_analysis['prediction'], 'confidence': mc, 'score': markov2_analysis['markov2Score']}
        vote = markov2_analysis['prediction']
        bm = anti_bias['multiplier'] if (anti_bias['correction'] == 'PENALIZE_BIG' and vote == 'BIG') or (anti_bias['correction'] == 'PENALIZE_SMALL' and vote == 'SMALL') else 1.0
        if vote == 'BIG':
            big_score += markov2_weight * mc * bm
        else:
            small_score += markov2_weight * mc * bm
        pattern_contributions['markov2'] = {'prediction': vote, 'weight': markov2_weight, 'confidence': mc}
    if trend_stats_analysis and 'prediction' in trend_stats_analysis:
        tw = 25
        tc = trend_stats_analysis['confidence']
        ts = tc * 1.5
        pattern_predictions['trendStatistics'] = {'prediction': trend_stats_analysis['prediction'], 'confidence': tc, 'score': ts}
        if trend_stats_analysis['prediction'] == 'BIG':
            big_score += tw * tc * (ts / 100)
        else:
            small_score += tw * tc * (ts / 100)
        pattern_contributions['trendStatistics'] = {'prediction': trend_stats_analysis['prediction'], 'weight': tw, 'confidence': tc}
    if deep_history_analysis['prediction'] != 'NEUTRAL' and deep_history_analysis['historySize'] >= 20:
        dw = 20
        dc = deep_history_analysis['confidence']
        pattern_predictions['deepHistory'] = {'prediction': deep_history_analysis['prediction'], 'confidence': dc, 'score': deep_history_analysis['deepScore']}
        bm_d = anti_bias['multiplier'] if (anti_bias['correction'] == 'PENALIZE_BIG' and deep_history_analysis['prediction'] == 'BIG') or (anti_bias['correction'] == 'PENALIZE_SMALL' and deep_history_analysis['prediction'] == 'SMALL') else 1.0
        if deep_history_analysis['prediction'] == 'BIG':
            big_score += dw * dc * (deep_history_analysis['deepScore'] / 100) * bm_d
        else:
            small_score += dw * dc * (deep_history_analysis['deepScore'] / 100) * bm_d
        pattern_contributions['deepHistory'] = {'prediction': deep_history_analysis['prediction'], 'weight': dw, 'confidence': dc}
    ml_samples_available = int((ml_prediction or {}).get('samples') or 0)
    model_ready_for_v2 = is_v2_model_ready(ml_prediction)
    if ml_prediction and ml_prediction['prediction'] != 'NEUTRAL' and model_ready_for_v2:
        ml_samples = ml_prediction.get('samples', 0)
        ml_recent = ml_acc_data.get('recentAccuracy', ml_acc) if ml_acc_data else ml_acc
        if ml_samples < 20:
            mw = 6
        elif ml_acc < 52 and ml_recent < 52:
            mw = 8
        elif ml_acc >= 60 or ml_recent >= 60:
            mw = max(14, min(38, int(max(ml_acc, ml_recent) * 0.42)))
        else:
            mw = max(10, min(24, int(max(ml_acc, ml_recent) * 0.30)))
        mc = ml_prediction['confidence']
        if win_loss_tracker.get('consecutiveLosses', 0) >= 2:
            mc = min(mc, 72)
        pattern_predictions['ml'] = {'prediction': ml_prediction['prediction'], 'confidence': mc, 'score': ml_prediction['mlScore']}
        bm_m = anti_bias['multiplier'] if (anti_bias['correction'] == 'PENALIZE_BIG' and ml_prediction['prediction'] == 'BIG') or (anti_bias['correction'] == 'PENALIZE_SMALL' and ml_prediction['prediction'] == 'SMALL') else 1.0
        if ml_prediction['prediction'] == 'BIG':
            big_score += mw * mc * (ml_prediction['mlScore'] / 100) * bm_m
        else:
            small_score += mw * mc * (ml_prediction['mlScore'] / 100) * bm_m
        pattern_contributions['ml'] = {
            'prediction': ml_prediction['prediction'],
            'weight': mw,
            'confidence': mc,
            'samples': ml_samples,
            'accuracy': ml_acc,
            'recentAccuracy': ml_recent,
        }
    if autolearn_pred and autolearn_pred['prediction'] != 'NEUTRAL':
        al_acc_val = autolearn_acc_data.get('overallAccuracy', 0) if autolearn_acc_data else 0
        aw = max(10, min(30, int(al_acc_val * 0.3)))
        ac = autolearn_pred['confidence']
        pattern_predictions['autolearn'] = {'prediction': autolearn_pred['prediction'], 'confidence': ac, 'score': autolearn_pred['patternConfidence'], 'patternType': autolearn_pred.get('patternType')}
        bm_a = anti_bias['multiplier'] if (anti_bias['correction'] == 'PENALIZE_BIG' and autolearn_pred['prediction'] == 'BIG') or (anti_bias['correction'] == 'PENALIZE_SMALL' and autolearn_pred['prediction'] == 'SMALL') else 1.0
        if autolearn_pred['prediction'] == 'BIG':
            big_score += aw * ac * (autolearn_pred['patternConfidence'] / 100) * bm_a
        else:
            small_score += aw * ac * (autolearn_pred['patternConfidence'] / 100) * bm_a
        pattern_contributions['autolearn'] = {'prediction': autolearn_pred['prediction'], 'weight': aw, 'confidence': ac}
    for pattern, weight in pattern_weights.items():
        pat_pred = 'NEUTRAL'
        pat_conf = 50
        pat_score = 50
        if pattern == 'longPattern':
            if long_pattern_analysis['isLongPattern']:
                pat_pred = long_pattern_analysis['lastCategory']
                pat_conf = 40 + (accuracy_analysis['recentAccuracy'] / 5)
                pat_score = long_pattern_analysis['longPatternScore']
        elif pattern == 'zigZag':
            if zigzag_analysis['isZigZag'] and not zigzag_analysis['isBroken']:
                pat_pred = 'SMALL' if zigzag_analysis['lastCategory'] == 'BIG' else 'BIG'
                pat_conf = 35 + (accuracy_analysis['recentAccuracy'] / 6)
                pat_score = zigzag_analysis['zigZagScore'] * 10
        elif pattern == 'skipPattern':
            if skip_analysis['isSkipPattern']:
                pat_pred = 'SMALL' if latest_result == 'BIG' else 'BIG'
                pat_conf = 30 + (accuracy_analysis['recentAccuracy'] / 7)
                pat_score = skip_analysis['skipScore'] * 10
        elif pattern == 'cyclePattern':
            if cycle_analysis['isCycle']:
                l4 = [r['category'] for r in results[:4]] if results else []
                pat_pred = 'SMALL' if l4[1] == 'BIG' else 'BIG' if len(l4) > 1 else 'NEUTRAL'
                pat_conf = 35 + (accuracy_analysis['recentAccuracy'] / 6)
                pat_score = cycle_analysis['cycleScore'] * 10
        elif pattern == 'numberBased':
            na = get_number_based_prediction(latest_number, user_states['user']['numberPatterns'], user_states['user']['numberRepetition'])
            pat_pred = na['prediction']
            pat_conf = na['confidence']
            pat_score = na['numberScore']
        elif pattern == 'markovChain':
            pat_pred = markov_analysis['prediction']
            pat_conf = markov_analysis['confidence']
            pat_score = markov_analysis['markovScore']
        elif pattern == 'entropyBased':
            pat_pred = entropy_analysis['prediction']
            pat_conf = entropy_analysis['confidence']
            pat_score = entropy_analysis['entropyScore']
        elif pattern == 'neural':
            pat_pred = neural_analysis['prediction']
            pat_conf = neural_analysis['confidence']
            pat_score = neural_analysis['neuralScore']
        else:
            pat_pred = trend_analysis['trend'] if trend_analysis['trend'] != 'NEUTRAL' else ('SMALL' if latest_result == 'BIG' else 'BIG')
            pat_conf = 40 + (accuracy_analysis['recentAccuracy'] / 5)
            pat_score = trend_analysis['trendScore']
        pattern_predictions[pattern] = {'prediction': pat_pred, 'confidence': pat_conf, 'score': pat_score}
        pattern_contributions[pattern] = {'prediction': pat_pred, 'weight': weight, 'confidence': pat_conf}
        if pat_pred == 'BIG':
            bm = anti_bias['multiplier'] if anti_bias['correction'] == 'PENALIZE_BIG' else 1.0
            big_score += weight * pat_conf * (pat_score / 100) * bm
        elif pat_pred == 'SMALL':
            sm = anti_bias['multiplier'] if anti_bias['correction'] == 'PENALIZE_SMALL' else 1.0
            small_score += weight * pat_conf * (pat_score / 100) * sm
    # ─── Intelligent human-like voting ───
    # Count how many patterns vote each way, weighted by their recent accuracy
    big_votes = small_votes = 0
    big_weighted = small_weighted = 0.0
    total_weight = 0.0
    for pat_name, pat_data in pattern_predictions.items():
        if pat_data.get('prediction') not in ('BIG', 'SMALL'):
            continue
        pat_acc = 50
        if pat_name in pattern_stats:
            ps = pattern_stats[pat_name]
            if ps.get('total', 0) >= 3:
                pat_acc = ps.get('successRate', 50)
        pat_conf = pat_data.get('confidence', 50)
        vote_weight = (pat_conf / 100.0) * (pat_acc / 100.0)
        total_weight += vote_weight
        if pat_data['prediction'] == 'BIG':
            big_votes += 1
            big_weighted += vote_weight
        else:
            small_votes += 1
            small_weighted += vote_weight
    raw_pred = 'BIG' if big_weighted > small_weighted else 'SMALL'
    raw_conf = min(90, max(55, (abs(big_weighted - small_weighted) / max(total_weight, 0.01)) * 100)) if total_weight > 0 else 55
    trend = get_recent_actual_trend(all_predictions)
    if trend and trend == raw_pred:
        raw_conf = min(raw_conf + 6, 92)
    elif trend and trend != raw_pred and win_loss_tracker.get('consecutiveLosses', 0) == 0:
        raw_conf = max(raw_conf - 4, 55)
    # Cross-check: if both regime and momentum agree, boost confidence
    regime_align = False
    if regime == 'STREAK' and streak_analysis.get('prediction') == raw_pred:
        regime_align = True
    if regime == 'ZIGZAG' and 'zigZag' in pattern_predictions and pattern_predictions['zigZag'].get('prediction') == raw_pred:
        regime_align = True
    if regime_align:
        raw_conf = min(raw_conf + 8, 92)
    # Finalize prediction
    prediction = raw_pred
    confidence = raw_conf
    # RAC algorithm as secondary check
    rac_scores = rac_algorithm(pattern_predictions, all_predictions, pattern_stats)
    rac_big = rac_small = total_rac = 0
    for pattern, rd in rac_scores.items():
        if pattern in pattern_predictions:
            total_rac += rd['score']
            if rd['prediction'] == 'BIG':
                rac_big += rd['score']
            elif rd['prediction'] == 'SMALL':
                rac_small += rd['score']
    if total_rac > 0:
        rnb = (rac_big / total_rac) * 100
        rns = (rac_small / total_rac) * 100
        if rnb > 65 and prediction == 'SMALL':
            prediction = 'BIG'
            confidence = min(confidence + 5, 88)
        elif rns > 65 and prediction == 'BIG':
            prediction = 'SMALL'
            confidence = min(confidence + 5, 88)
    rec_acc = accuracy_analysis.get('recentAccuracy', 50)
    if rec_acc > 0 and rec_acc < 100:
        if rec_acc > 65:
            confidence = min(confidence + 5, 92)
        elif rec_acc < 40:
            confidence = max(confidence - 8, 50)
    ml_info = f" ML:{ml_prediction['prediction']}@{ml_prediction['confidence']}" if ml_prediction and model_ready_for_v2 else ""
    ml_acc_str = f" mlAcc:{ml_acc:.0f}%/{ml_acc_data.get('recentAccuracy', ml_acc):.0f}%" if ml_prediction and model_ready_for_v2 and ml_acc_data else ""
    corr_str = f" C:{correction}" if correction != 'none' else ""
    al_acc = autolearn_acc_data['overallAccuracy']
    al_str = f" AL:{autolearn_detected['patternType']}@{autolearn_detected['confidence']}% Acc:{al_acc}%" if autolearn_detected['patternType'] != 'unknown' and autolearn_detected['confidence'] > 0 else ""
    explanation = f"R:{regime} S:{streak_analysis['streakLength']} H:{deep_history_analysis['historySize']}{ml_info}{ml_acc_str}{al_str}{corr_str} B:{anti_bias['correction']} Big:{round(big_score)} Sm:{round(small_score)} => {prediction}@{confidence}%"
    agreeing = total_signals = 0
    for p in pattern_predictions.values():
        if p['prediction'] != 'NEUTRAL':
            total_signals += 1
            if p['prediction'] == prediction:
                agreeing += 1
    consensus_ratio = agreeing / total_signals if total_signals > 0 else 0.5
    if consensus_ratio >= 0.60:
        if neural_analysis.get('confidence', 0) > 80 and neural_analysis.get('prediction') == prediction:
            confidence = min(confidence + 5, 90)
        if number_analysis.get('confidence', 0) > 80 and number_analysis.get('prediction') == prediction:
            confidence = min(confidence + 5, 90)
        if trend_analysis['trend'] != 'NEUTRAL' and trend_analysis['trend'] == prediction:
            confidence = min(confidence + 5, 90)
        if trend_stats_analysis and 'prediction' in trend_stats_analysis and trend_stats_analysis['prediction'] == prediction:
            confidence = min(confidence + 6, 90)
        if deep_history_analysis['prediction'] == prediction and deep_history_analysis['historySize'] > 100:
            confidence = min(confidence + 5, 90)
        if ml_prediction and model_ready_for_v2 and ml_prediction['prediction'] == prediction:
            confidence = min(confidence + 6, 90)
    if not allow_same_prediction(win_loss_tracker, pending_predictions, prediction, all_predictions):
        trend = get_recent_actual_trend(all_predictions)
        if trend and trend != prediction:
            prediction = trend
        confidence = max(confidence - 15, 70)
    if accuracy_analysis['recentAccuracy'] < 50 and accuracy_analysis['recentAccuracy'] > 0:
        confidence = max(confidence - 8, 55)
    if results:
        rec10 = results[:10]
        rb = sum(1 for r in rec10 if r['category'] == 'BIG')
        rbr = rb / len(rec10) if rec10 else 0.5
        if rbr > 0.75:
            prediction = 'BIG'
            confidence = max(confidence + 5, 78)
        elif rbr < 0.25:
            prediction = 'SMALL'
            confidence = max(confidence + 5, 78)
        elif prediction == 'SMALL' and rbr > 0.65:
            prediction = 'BIG'
            confidence = max(confidence, 72)
        elif prediction == 'BIG' and rbr < 0.35:
            prediction = 'SMALL'
            confidence = max(confidence, 72)
        elif win_loss_tracker.get('consecutiveWins', 0) >= 3:
            confidence = min(confidence + 8, 92)
    cons_loss = win_loss_tracker.get('consecutiveLosses', 0)
    if cons_loss >= 2:
        recovery_trend = get_recent_actual_trend(all_predictions)
        if recovery_trend and recovery_trend != prediction and consensus_ratio < 0.70:
            prediction = recovery_trend
        confidence = min(max(confidence - 8, 55), 75)
    elif cons_loss == 1:
        lpred = win_loss_tracker.get('lastPrediction')
        lact = win_loss_tracker.get('lastActual')
        if lpred and lact and lpred == prediction and lpred != lact:
            trend = get_recent_actual_trend(all_predictions)
            if trend and trend != prediction:
                prediction = trend
                confidence = min(confidence + 12, 90)
    ver = _own_verified_rows(all_predictions)
    if len(ver) >= 5:
        recent_act = [e.get('actual') for e in ver[:10] if e.get('actual') in ('BIG', 'SMALL')]
        if len(recent_act) >= 5:
            big_c = recent_act.count('BIG')
            sm_c = recent_act.count('SMALL')
            if big_c / len(recent_act) > 0.70 and prediction == 'SMALL':
                prediction = 'BIG'
                confidence = max(confidence + 3, 65)
            elif sm_c / len(recent_act) > 0.70 and prediction == 'BIG':
                prediction = 'SMALL'
                confidence = max(confidence + 3, 65)
    loss_signal = get_loss_correction_signal(all_predictions)
    market_signal = get_market_structure_signal(results, all_predictions)
    prediction, confidence, fusion_log = advanced_pattern_fusion(
        pattern_predictions,
        pattern_stats,
        all_predictions,
        prediction,
        confidence,
        market_signal,
        loss_signal,
    )
    # Don't let ML override during active loss streaks — brain/patterns should lead
    consecutive_losses = win_loss_tracker.get('consecutiveLosses', 0)
    if model_ready_for_v2 and ml_prediction and ml_prediction.get('prediction') in ('BIG', 'SMALL'):
        if consecutive_losses >= 1:
            # During losses, only use ML if it's predicting a DIFFERENT direction than what kept losing
            last_loss_actual = win_loss_tracker.get('lastActual')
            if last_loss_actual and ml_prediction['prediction'] == last_loss_actual:
                # ML agrees with the market's actual direction — use it
                prediction = ml_prediction['prediction']
                confidence = float(ml_prediction.get('confidence') or confidence)
            # Otherwise keep the pattern-based prediction (ML keeps repeating the losing direction)
        else:
            prediction = ml_prediction['prediction']
            confidence = float(ml_prediction.get('confidence') or confidence)

    # Brain inverse guard: check learned patterns BEFORE recovery
    brain_guard_pred, brain_guard_conf, brain_guard_log = brain_inverse_guard(
        prediction, confidence, all_predictions, consecutive_losses,
    )
    if brain_guard_log.get('applied'):
        prediction = brain_guard_pred
        confidence = brain_guard_conf

    prediction, confidence, model_recovery_log = apply_model_recovery(
        prediction,
        confidence,
        ml_prediction if model_ready_for_v2 else None,
        all_predictions,
    )
    if brain_guard_log.get('applied'):
        model_recovery_log['brainGuard'] = brain_guard_log
        # If model recovery overrode brain guard, check if brain should win
        brain_guard_overridden = (
            model_recovery_log.get('outputPrediction') != brain_guard_pred
            and model_recovery_log.get('reason', '') not in ('model_not_ready', 'insufficient_validated_models', 'model_gate_not_met')
        )
        if brain_guard_overridden:
            # During active losses, brain guard is authoritative unless ML is exceptional
            override_accuracy = model_recovery_log.get('bestValidationAccuracy', 0) or 0
            if consecutive_losses >= 2 or override_accuracy < 60:
                prediction = brain_guard_pred
                confidence = brain_guard_conf
                model_recovery_log['outputPrediction'] = brain_guard_pred
                model_recovery_log['reason'] = 'brain_guard_overrides_model_recovery'
                model_recovery_log['brainGuard'] = brain_guard_log

    if model_ready_for_v2 and ml_prediction and ml_prediction.get('prediction') in ('BIG', 'SMALL'):
        model_recovery_log['mode'] = 'MODEL_ONLY'
        fusion_log['decisionMode'] = 'MODEL_ONLY'
    else:
        model_recovery_log['mode'] = 'CUSTOM_ONLY_ML_LEARNING'
    correction_reason = fusion_log.get('reason', 'advanced_pattern_fusion')
    explanation += f" AF:{correction_reason} MR:{model_recovery_log.get('reason')}"
    loss_log = build_loss_learning_log(pattern_stats, win_loss_tracker, own_eval, correction, confidence)
    loss_log['advancedFusion'] = fusion_log
    loss_log['modelRecovery'] = model_recovery_log
    recovery_stats = build_recovery_stats(win_loss_tracker, confidence, consensus_ratio, loss_signal, market_signal)
    loss_log['recoveryStats'] = recovery_stats
    v2_loss_risk = assess_v2_loss_risk(
        confidence,
        consensus_ratio,
        ml_prediction,
        model_recovery_log,
        fusion_log,
        market_signal,
        loss_signal,
        win_loss_tracker,
        pending_predictions,
    )
    loss_log['lossRisk'] = v2_loss_risk
    mistake_profile = self_learning_mistake_profile(all_predictions)
    loss_log['mistakeProfile'] = mistake_profile
    prediction, confidence, recovery_suffix = apply_loss_recovery(user_states, prediction, confidence, win_loss_tracker, explanation, all_predictions)
    prediction, confidence, brain_suffix = brain_like_deep_recovery(all_predictions, results, prediction, confidence, win_loss_tracker)
    try:
        cons_loss = win_loss_tracker.get('consecutiveLosses', 0)
        recent_cats = [r.get('category') for r in (results or [])[:8] if r.get('category') in ('BIG', 'SMALL')]
        brain_pred, brain_conf, brain_log = brain_think(
            prediction, confidence, cons_loss,
            win_loss_tracker.get('lastPrediction'),
            win_loss_tracker.get('lastActual'),
            recent_cats,
        )
        if brain_conf > confidence:
            prediction = brain_pred
            confidence = brain_conf
            brain_suffix = f' brain_boost_{brain_pred}@{brain_conf}'
    except Exception:
        pass
    # Final brain guard reinforcement: if brain had a learned pattern prediction,
    # ensure it's respected when consecutive losses are active
    if brain_guard_log.get('applied') and consecutive_losses >= 1:
        if prediction != brain_guard_pred and brain_guard_conf >= confidence:
            prediction = brain_guard_pred
            confidence = brain_guard_conf
            brain_suffix = f' brain_final_guard_{brain_guard_pred}@{brain_guard_conf}'
    recovery_suffix = recovery_suffix or brain_suffix
    if brain_suffix:
        recovery_suffix = brain_suffix
    explanation += recovery_suffix
    if autolearn_pred:
        pattern_predictions['autolearn'] = autolearn_pred
    loss_log = build_loss_learning_log(pattern_stats, win_loss_tracker, own_eval, correction, confidence)
    loss_log['recoveryStats'] = build_recovery_stats(win_loss_tracker, confidence, consensus_ratio, loss_signal, market_signal)
    loss_log['advancedFusion'] = fusion_log
    loss_log['modelRecovery'] = model_recovery_log
    loss_log['lossRisk'] = v2_loss_risk
    prediction_entry = {        'period': auto_period, 'prediction': prediction, 'status': 'Pending',        'confidence': round(confidence, 2), 'locked': True, 'latestResult': latest_result,        'patternUsed': 'ensemble', 'patternPredictions': pattern_predictions, 'lossLog': loss_log,        'timestamp': current_time, 'skipped': False,    }
    pending_predictions.insert(0, prediction_entry)
    all_predictions.insert(0, prediction_entry)
    append_prediction_csv(prediction_entry)
    save_all_states(user_states, pending_predictions)
    fs = get_stats(pending_predictions, all_predictions, user_states)
    ad = analyze_prediction_accuracy(all_predictions)
    wlt2 = analyze_win_loss_tracker(all_predictions)
    append_stats_csv({        'timestamp': current_time, 'date': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),        'totalWins': fs['totalWins'], 'totalLosses': fs['totalLosses'],        'winRate': fs['winRate'], 'accuracy': ad['accuracy'],        'recentAccuracy': ad['recentAccuracy'], 'totalPredictions': ad['totalPredictions'],        'streak': fs['streak'], 'consecutiveLosses': wlt2['consecutiveLosses'],        'consecutiveWins': wlt2['consecutiveWins'],    })
    return {        'gameType': 'Wingo 1 Min', 'period': auto_period,        'prediction': prediction, 'confidence': confidence,        'locked': True, 'latestResult': latest_result, 'timestamp': current_time,        'lossRecovery': user_states['user']['lossRecovery'], 'lossLog': loss_log,        'skipped': False, 'explanation': explanation,        'recommendation': f"Bet: {prediction} (Confidence: {confidence}%). {recovery_suffix}",        'mlAccuracy': round(ml_acc, 1) if ml_acc_data and ml_acc_data.get('total', 0) >= 10 else None,        'mlSamples': ml_acc_data.get('total', 0) if ml_acc_data else 0,        'autolearnAccuracy': autolearn_acc_data['overallAccuracy'],        'autolearnPattern': autolearn_detected['patternType'],        'autolearnConfidence': autolearn_detected['confidence'],    }
