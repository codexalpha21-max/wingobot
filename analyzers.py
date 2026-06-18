import math
import time


def analyze_trend_statistics(trend_data):
    if not isinstance(trend_data, dict) and not isinstance(trend_data, list):
        return None
    if isinstance(trend_data, dict) and 'error' in trend_data:
        return None
    if not trend_data:
        return None

    number_stats = {}
    for item in trend_data:
        number = int(item.get('number', -1))
        if 0 <= number <= 9:
            number_stats[number] = {
                'missingCount': int(item.get('missingCount', 0)),
                'avgMissing': float(item.get('avgMissing', 0)),
                'openCount': int(item.get('openCount', 0)),
                'maxContinuous': int(item.get('maxContinuous', 0))
            }

    prediction_scores = {}
    for num, stats in number_stats.items():
        score = 0
        if stats['missingCount'] > stats['avgMissing']:
            score += (stats['missingCount'] - stats['avgMissing']) * 10
        if stats['missingCount'] > stats['avgMissing'] * 1.5:
            score += 30
        if stats['openCount'] > 12:
            score += 15
        if stats['maxContinuous'] >= 2:
            score += 10
        if stats['openCount'] < 3 and stats['missingCount'] < stats['avgMissing'] * 0.5:
            score -= 20
        prediction_scores[num] = score

    sorted_scores = sorted(prediction_scores.items(), key=lambda x: x[1], reverse=True)
    top_numbers = [k for k, v in sorted_scores[:3]]

    big_count = small_count = 0
    for num in top_numbers:
        if num >= 5:
            big_count += prediction_scores[num]
        else:
            small_count += prediction_scores[num]

    prediction = 'BIG' if big_count > small_count else 'SMALL'
    denom = max(big_count + small_count, 1)
    confidence = min(95, 60 + abs(big_count - small_count) / denom * 30)

    high_missing = [num for num, stats in number_stats.items() if stats['missingCount'] > stats['avgMissing'] * 1.2]
    low_missing = [num for num, stats in number_stats.items() if stats['missingCount'] < stats['avgMissing'] * 0.8]
    high_open = [num for num, stats in number_stats.items() if stats['openCount'] > 10]

    return {
        'prediction': prediction,
        'confidence': round(confidence, 2),
        'topNumbers': top_numbers,
        'numberScores': prediction_scores,
        'numberStats': number_stats,
        'highMissingNumbers': high_missing,
        'lowMissingNumbers': low_missing,
        'analysis': {
            'avgMissing': sum(s['avgMissing'] for s in number_stats.values()) / max(len(number_stats), 1),
            'totalOpenCount': sum(s['openCount'] for s in number_stats.values()),
            'mostFrequent': high_open
        }
    }


def analyze_zigzag_pattern(results):
    cats = [r['category'] for r in results[:15]]
    if len(cats) < 4:
        return {
            'isZigZag': False,
            'isBroken': False,
            'lastCategory': cats[0] if cats else None,
            'zigZagScore': 0,
            'confidence': 50
        }

    alt_count = 0
    for i in range(1, len(cats)):
        if cats[i] != cats[i - 1]:
            alt_count += 1

    total_pairs = len(cats) - 1
    alt_ratio = alt_count / total_pairs if total_pairs > 0 else 0
    is_zigzag = alt_ratio >= 0.7 and len(cats) >= 5

    streak_len = 1
    for i in range(1, min(6, len(cats))):
        if cats[i] == cats[i - 1]:
            streak_len += 1
        else:
            break

    is_broken = streak_len >= 3
    last = cats[0] if cats else None
    next_pred = 'SMALL' if last == 'BIG' else 'BIG'

    if is_zigzag:
        confidence = min(85, 55 + alt_ratio * 30)
    elif is_broken:
        confidence = min(75, 50 + streak_len * 5)
    else:
        if alt_ratio >= 0.5:
            confidence = 55 + alt_ratio * 15
        else:
            confidence = 50

    score = int(alt_ratio * 100)

    return {
        'isZigZag': is_zigzag,
        'isBroken': is_broken,
        'lastCategory': last,
        'nextPrediction': next_pred,
        'zigZagScore': score,
        'confidence': round(confidence),
        'alternationRatio': round(alt_ratio, 2),
        'streakLength': streak_len
    }


def analyze_skip_pattern(results):
    cats = [r['category'] for r in results[:12]]
    if len(cats) < 5:
        return {'isSkipPattern': False, 'skipScore': 0, 'confidence': 50, 'nextPrediction': None}

    scores = []
    patterns = [
        (['SMALL', 'SMALL', 'BIG', 'SMALL', 'SMALL', 'BIG'], 'SMALL'),
        (['BIG', 'BIG', 'SMALL', 'BIG', 'BIG', 'SMALL'], 'BIG'),
        (['SMALL', 'BIG', 'SMALL', 'BIG', 'SMALL', 'BIG'], 'SMALL'),
        (['BIG', 'SMALL', 'BIG', 'SMALL', 'BIG', 'SMALL'], 'BIG'),
    ]

    next_pred = None
    best_score = 0
    for pattern, nxt in patterns:
        if len(cats) >= len(pattern):
            matches = sum(1 for i in range(len(pattern)) if i < len(cats) and cats[i] == pattern[i])
            match_ratio = matches / len(pattern)
            if match_ratio >= 0.7:
                weight = match_ratio * 100
                if matches == len(pattern):
                    weight += 30
                scores.append(weight)
                if match_ratio > best_score:
                    best_score = match_ratio
                    next_pred = nxt

    is_skip = best_score >= 0.7
    confidence = min(85, 50 + best_score * 40) if best_score > 0 else 50

    return {
        'isSkipPattern': is_skip,
        'skipScore': int(best_score * 10) if best_score > 0 else 0,
        'confidence': round(confidence),
        'nextPrediction': next_pred,
        'matchRatio': round(best_score, 2)
    }


def analyze_trend_based(results):
    results = results[:150]
    total = len(results)
    if total == 0:
        return {
            'bigRatio': 0.5,
            'movingAverage': 0.5,
            'trend': 'NEUTRAL',
            'trendScore': 50,
            'momentum': 0,
            'rsi': 50
        }

    cats = [1 if r['category'] == 'BIG' else 0 for r in results]
    weights = [pow(0.92, i) for i in range(total)]
    wsum = sum(weights)
    big_r = sum(c * w for c, w in zip(cats, weights)) / wsum if wsum > 0 else 0.5

    ema_short = 0
    ema_long = 0
    alpha_s = 0.3
    alpha_l = 0.1
    for i in range(min(10, total)):
        ema_short = alpha_s * cats[i] + (1 - alpha_s) * ema_short
    for i in range(min(30, total)):
        ema_long = alpha_l * cats[i] + (1 - alpha_l) * ema_long

    momentum = ema_short - ema_long

    gains = 0
    losses = 0
    for i in range(1, min(14, total)):
        diff = cats[i - 1] - cats[i]
        if diff > 0:
            gains += diff
        else:
            losses += abs(diff)

    avg_gain = gains / min(14, total) if min(14, total) > 0 else 0.5
    avg_loss = losses / min(14, total) if min(14, total) > 0 else 0.5
    rsi = 50
    if avg_loss > 0:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = 70

    ma = sum(cats[:20]) / min(20, total) if total > 0 else 0.5
    base_ts = (big_r * 0.5 + ma * 0.3) * 100
    momentum_factor = momentum * 20
    ts = base_ts + momentum_factor

    if rsi > 65:
        ts -= 8
    elif rsi < 35:
        ts += 8

    ts = max(0, min(100, ts))
    trend = 'BIG' if ts > 58 else ('SMALL' if ts < 42 else 'NEUTRAL')

    return {
        'bigRatio': big_r,
        'movingAverage': ma,
        'trend': trend,
        'trendScore': round(ts, 1),
        'momentum': round(momentum, 3),
        'rsi': round(rsi, 1),
        'emaShort': round(ema_short, 3),
        'emaLong': round(ema_long, 3)
    }


def analyze_cycle_pattern(results):
    cats = [r['category'] for r in results[:12]]
    if len(cats) < 6:
        return {'isCycle': False, 'cycleScore': 0, 'confidence': 50, 'nextPrediction': None}

    phase_hits = 0
    phase_total = min(6, len(cats) // 2)
    for i in range(phase_total):
        if i + 2 < len(cats):
            if cats[i] == cats[i + 2]:
                phase_hits += 1

    phase_ratio = phase_hits / phase_total if phase_total > 0 else 0
    is_cycle = phase_ratio >= 0.7

    next_pred = None
    if is_cycle and len(cats) >= 4:
        next_pred = cats[1]

    conf = 55 + phase_ratio * 35

    return {
        'isCycle': is_cycle,
        'cycleScore': int(phase_ratio * 10),
        'confidence': round(min(90, conf)),
        'nextPrediction': next_pred,
        'phaseRatio': round(phase_ratio, 2)
    }


def analyze_long_pattern(results, win_loss_tracker):
    cats = [r['category'] for r in results[:15]]
    last = results[0]['category'] if results else None
    if not cats:
        return {'isLongPattern': False, 'lastCategory': None, 'longPatternScore': 0, 'confidence': 50}

    streak = 1
    for i in range(1, len(cats)):
        if cats[i] == cats[i - 1]:
            streak += 1
        else:
            break

    total_wins = win_loss_tracker.get('consecutiveWins', 0)
    total_losses = win_loss_tracker.get('consecutiveLosses', 0)
    is_long = streak >= 3 or (streak >= 2 and total_wins >= 2)
    long_confidence = min(85, 50 + streak * 8 + (5 if total_wins >= 2 else 0))

    if streak >= 5:
        long_confidence -= 10
        is_long = False
    if total_losses >= 3:
        is_long = False
        long_confidence = 40

    score = streak * 2 + (3 if total_wins >= 2 else 0)

    return {
        'isLongPattern': is_long,
        'lastCategory': last,
        'longPatternScore': score,
        'confidence': round(long_confidence),
        'streakLength': streak
    }


def analyze_markov_chain(results, user_states):
    tm = user_states['user'].get('transitionMatrix', {})
    results = results[:150]
    df = 0.95

    for i in range(1, len(results)):
        cur = results[i].get('category')
        nxt_val = results[i - 1].get('category')
        if cur and nxt_val:
            w = pow(df, i - 1)
            tm.setdefault(cur, {'BIG': 0, 'SMALL': 0})
            tm[cur][nxt_val] = tm[cur].get(nxt_val, 0) + w

    cc = results[0].get('category', 'BIG') if results else 'BIG'
    bc = tm.get(cc, {}).get('BIG', 0)
    sc = tm.get(cc, {}).get('SMALL', 0)
    tot = bc + sc
    bp = (bc / tot) * 100 if tot > 0 else 50
    pred = 'BIG' if bp > 50 else 'SMALL'
    conf = min(abs(bp - 50) * 2 + 50, 95)

    return {
        'prediction': pred,
        'confidence': conf,
        'bigProbability': bp,
        'markovScore': conf * (1.2 if tot > 10 else 1.0),
        'samples': tot
    }


def analyze_entropy_based(results, user_states):
    cats = [r['category'] for r in results[:150]]
    total = len(cats)
    if total == 0:
        return {
            'prediction': 'SMALL',
            'confidence': 50,
            'entropy': 0,
            'avgEntropy': 0,
            'entropyScore': 50
        }

    counts = {}
    for c in cats:
        counts[c] = counts.get(c, 0) + 1

    entropy = 0
    for c, count in counts.items():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    big_r = counts.get('BIG', 0) / total if total > 0 else 0.5
    history = user_states['user'].setdefault('entropyHistory', [])
    history.append(entropy)
    if len(history) > 200:
        user_states['user']['entropyHistory'] = history[-200:]

    avg = sum(history) / max(len(history), 1)
    entropy_ratio = entropy / avg if avg > 0 else 1
    last = results[0].get('category', 'BIG') if results else 'BIG'

    if entropy_ratio > 1.15:
        pred = 'SMALL' if last == 'BIG' else 'BIG'
        conf = 70
    elif entropy_ratio < 0.85:
        pred = last
        conf = 82
    else:
        if abs(big_r - 0.5) > 0.15:
            pred = 'BIG' if big_r > 0.5 else 'SMALL'
            conf = 65 + abs(big_r - 0.5) * 30
        else:
            pred = last
            conf = 60

    return {
        'prediction': pred,
        'confidence': round(min(92, conf)),
        'entropy': round(entropy, 3),
        'avgEntropy': round(avg, 3),
        'entropyScore': round(conf * (0.9 if entropy_ratio > 1.1 else 1.1)),
        'entropyRatio': round(entropy_ratio, 2),
        'bigRatio': round(big_r, 2)
    }


def analyze_neural_network(results, user_states):
    results = results[:15]
    inputs = [1 if r['category'] == 'BIG' else 0 for r in results]
    weights = user_states['user'].setdefault('neuralWeights', [0.0] * 10)
    bias = user_states['user'].setdefault('bias', 0.0)
    lr = user_states['user'].setdefault('learningRate', 0.1)

    total_w = bias
    for i in range(min(len(inputs), len(weights))):
        total_w += inputs[i] * weights[i]

    output = 1 / (1 + math.exp(-total_w))
    pred = 'BIG' if output > 0.5 else 'SMALL'
    conf = min(92, 50 + abs(output - 0.5) * 100)

    if len(inputs) > 1:
        actual = 1 if results[0].get('category', 'BIG') == 'BIG' else 0
        err = actual - output
        momentum = user_states['user'].setdefault('neuralMomentum', 0.0)
        momentum = 0.9 * momentum + 0.1 * err
        adapt_lr = lr * (1 + abs(momentum))
        for i in range(min(len(inputs), len(weights))):
            weights[i] += adapt_lr * err * inputs[i]
        user_states['user']['bias'] = bias + adapt_lr * err
        user_states['user']['neuralMomentum'] = momentum

    vals = {
        'prediction': pred,
        'confidence': round(conf),
        'output': round(output, 4),
        'neuralScore': round(conf * (1.3 if len(inputs) > 5 else 1.0), 1)
    }
    return vals


def analyze_number_patterns(results, user_states):
    np = user_states['user'].setdefault('numberPatterns', {})
    nr = user_states['user'].setdefault('numberRepetition', {})
    results = results[:150]
    df = 0.95
    ct = int(time.time())

    for i in range(1, len(results)):
        cn = results[i].get('number')
        cn = int(cn) if cn is not None else None
        nc = results[i - 1].get('category')
        if cn is not None and nc:
            np.setdefault(cn, {
                'BIG': {'count': 0, 'successRate': 0},
                'SMALL': {'count': 0, 'successRate': 0},
                'total': 0
            })
            w = pow(df, i - 1)
            np[cn]['total'] += w
            np[cn][nc]['count'] += w
            t = np[cn]['total']
            np[cn][nc]['successRate'] = round((np[cn][nc]['count'] / t) * 100, 2) if t > 0 else 0

        if cn is not None:
            nr.setdefault(cn, {'count': 0, 'recentCount': 0, 'lastSeen': 0})
            nr[cn]['count'] += 1
            nr[cn]['recentCount'] += 1
            nr[cn]['lastSeen'] = ct

    for num, data in nr.items():
        if ct - data.get('lastSeen', 0) > 3600:
            data['recentCount'] = 0


def get_number_based_prediction(latest_number, number_patterns, number_repetition):
    if latest_number is None or latest_number not in number_patterns:
        return {'prediction': 'NEUTRAL', 'confidence': 50, 'numberScore': 50}

    p = number_patterns[latest_number]
    if p.get('total', 0) < 5:
        return {'prediction': 'NEUTRAL', 'confidence': 50, 'numberScore': 50}

    br = p['BIG']['successRate']
    sr = p['SMALL']['successRate']
    pred = 'BIG' if br > sr else 'SMALL'
    conf = max(br, sr) + 15
    rep = number_repetition.get(latest_number, {})
    multiplier = 1.4 if rep.get('recentCount', 0) > 2 else 1.0

    if br > 70:
        conf += 5
    if sr > 70:
        conf += 5

    return {
        'prediction': pred,
        'confidence': round(min(92, conf)),
        'numberScore': round(min(conf, 92) * multiplier, 1)
    }


def analyze_streak_momentum(results):
    cats = [r['category'] for r in results[:20]]
    if len(cats) < 3:
        return {'prediction': 'NEUTRAL', 'confidence': 50, 'streakLength': 0, 'streakScore': 50}

    streak_len = 1
    streak_dir = cats[0]
    for i in range(1, len(cats)):
        if cats[i] == streak_dir:
            streak_len += 1
        else:
            break

    big5 = sum(1 for c in cats[:5] if c == 'BIG')
    small5 = 5 - big5
    streak_probs = {1: 48, 2: 45, 3: 40, 4: 35, 5: 28, 6: 22, 7: 15}
    break_prob = streak_probs.get(streak_len, 10) if streak_len <= 7 else 5

    if streak_len >= 5:
        opposite = 'SMALL' if streak_dir == 'BIG' else 'BIG'
        conf = min(80, 50 + (streak_len - 5) * 8)
        return {
            'prediction': opposite,
            'confidence': round(conf),
            'streakLength': streak_len,
            'streakScore': round(conf),
            'breakProbability': break_prob
        }

    if streak_len >= 3:
        if big5 >= 4 or small5 >= 4:
            return {
                'prediction': streak_dir,
                'confidence': 65 + streak_len * 5,
                'streakLength': streak_len,
                'streakScore': 65 + streak_len * 5,
                'breakProbability': break_prob
            }

    conf_map = {1: 55, 2: 62, 3: 70, 4: 75}
    base_conf = conf_map.get(streak_len, 65)
    if big5 >= 4:
        base_conf += 5
    if small5 >= 4:
        base_conf += 5

    return {
        'prediction': streak_dir,
        'confidence': round(min(85, base_conf)),
        'streakLength': streak_len,
        'streakScore': round(min(85, base_conf)),
        'breakProbability': break_prob
    }


def analyze_markov_2nd_order(results):
    cats = [r['category'] for r in results[:80]]
    if len(cats) < 5:
        return {'prediction': 'NEUTRAL', 'confidence': 50, 'markov2Score': 50}

    transitions = {}
    for i in range(2, len(cats)):
        state = f"{cats[i]}_{cats[i - 1]}"
        nxt_val = cats[i - 2]
        if state not in transitions:
            transitions[state] = {'BIG': 0, 'SMALL': 0}
        transitions[state][nxt_val] += 1

    alt_count = sum(1 for i in range(1, len(cats)) if cats[i] != cats[i - 1])
    alt_ratio = alt_count / max(len(cats) - 1, 1)

    cur_state = f"{cats[0]}_{cats[1]}"
    if cur_state not in transitions:
        if alt_ratio > 0.6:
            pred = 'SMALL' if cats[0] == 'BIG' else 'BIG'
            return {'prediction': pred, 'confidence': 55, 'markov2Score': 55, 'state': cur_state, 'bigProb': 50}
        return {'prediction': cats[0], 'confidence': 55, 'markov2Score': 55, 'state': cur_state, 'bigProb': 50}

    big_c = transitions[cur_state]['BIG']
    small_c = transitions[cur_state]['SMALL']
    total = big_c + small_c
    if total < 2:
        return {'prediction': cats[0], 'confidence': 55, 'markov2Score': 55, 'state': cur_state, 'bigProb': 50}

    big_prob = big_c / total
    pred = 'BIG' if big_prob > 0.5 else 'SMALL'
    conf = min(50 + abs(big_prob - 0.5) * 80, 90)

    return {
        'prediction': pred,
        'confidence': round(conf),
        'markov2Score': round(conf),
        'state': cur_state,
        'bigProb': round(big_prob * 100, 1),
        'samples': total
    }


def detect_market_regime(results):
    cats = [r['category'] for r in results[:25]]
    if len(cats) < 6:
        return {'regime': 'UNKNOWN', 'streakiness': 0.5, 'regimeScore': 50}

    alternations = 0
    sames = 0
    for i in range(1, len(cats)):
        if cats[i] != cats[i - 1]:
            alternations += 1
        else:
            sames += 1

    total_pairs = alternations + sames
    streakiness = sames / total_pairs if total_pairs > 0 else 0.5

    last6 = cats[:6]
    recent_alt = 0
    for i in range(1, len(last6)):
        if last6[i] != last6[i - 1]:
            recent_alt += 1
    recent_ratio = recent_alt / max(len(last6) - 1, 1)

    big_count = cats.count('BIG')
    big_ratio = big_count / len(cats)

    if streakiness > 0.55 and recent_ratio < 0.4:
        regime = 'STREAK'
    elif streakiness < 0.45 and recent_ratio > 0.5:
        regime = 'ZIGZAG'
    elif abs(big_ratio - 0.5) > 0.25:
        regime = 'TREND'
    elif streakiness > 0.5 and recent_ratio > 0.5:
        regime = 'CHOPPY'
    else:
        regime = 'MIXED'

    return {
        'regime': regime,
        'streakiness': round(streakiness, 2),
        'regimeScore': round(streakiness * 100),
        'recentAlternation': round(recent_ratio, 2),
        'bigRatio': round(big_ratio, 2)
    }


def get_anti_bias_correction(all_predictions):
    filtered = [
        e for e in all_predictions
        if e.get('status') in ('WIN', 'LOSS')
        and str(e.get('patternUsed') or e.get('patternused') or '').lower() != 'imported'
        and e.get('prediction') in ('BIG', 'SMALL')
        and e.get('actual') in ('BIG', 'SMALL')
    ]
    if len(filtered) < 5:
        return {'correction': 'NONE', 'multiplier': 1.0, 'biasedToward': None}

    recent = filtered[:15]
    lost_big = lost_small = 0
    for e in recent:
        if e['status'] == 'LOSS':
            if e.get('prediction') == 'BIG':
                lost_big += 1
            else:
                lost_small += 1

    big_preds = sum(1 for e in recent if e.get('prediction') == 'BIG' and e['status'] == 'LOSS')
    sm_preds = sum(1 for e in recent if e.get('prediction') == 'SMALL' and e['status'] == 'LOSS')
    total_loss = lost_big + lost_small

    if total_loss < 2:
        return {'correction': 'NONE', 'multiplier': 1.0, 'biasedToward': None}

    if lost_big / total_loss >= 0.55:
        return {'correction': 'PENALIZE_BIG', 'multiplier': 0.35, 'biasedToward': 'BIG'}
    if lost_small / total_loss >= 0.55:
        return {'correction': 'PENALIZE_SMALL', 'multiplier': 0.35, 'biasedToward': 'SMALL'}

    return {'correction': 'NONE', 'multiplier': 1.0, 'biasedToward': None}


def analyze_win_loss_tracker(predictions):
    verified = [
        e for e in predictions
        if e.get('status') in ('WIN', 'LOSS')
        and str(e.get('patternUsed') or e.get('patternused') or '').lower() != 'imported'
        and e.get('prediction') in ('BIG', 'SMALL')
        and e.get('actual') in ('BIG', 'SMALL')
    ]
    cons_loss = 0
    cons_win = 0
    last_pred = None
    last_act = None
    l4_losses = 0
    streak_score = 0

    if verified:
        latest_status = verified[0]['status']
        last_pred = verified[0].get('prediction')
        last_act = verified[0].get('actual')
        for e in verified:
            if e.get('status') != latest_status:
                break
            if latest_status == 'LOSS':
                cons_loss += 1
                streak_score -= 2
            else:
                cons_win += 1
                streak_score += 2

    for e in verified[:4]:
        if e['status'] == 'LOSS':
            l4_losses += 1
            streak_score -= 1

    return {
        'consecutiveLosses': cons_loss,
        'consecutiveWins': cons_win,
        'lastPrediction': last_pred,
        'lastActual': last_act,
        'lastFourLosses': l4_losses,
        'streakScore': streak_score
    }


def analyze_prediction_accuracy(all_predictions):
    verified = [
        e for e in all_predictions
        if e.get('status') in ('WIN', 'LOSS')
        and str(e.get('patternUsed') or e.get('patternused') or '').lower() != 'imported'
        and e.get('prediction') in ('BIG', 'SMALL')
        and e.get('actual') in ('BIG', 'SMALL')
    ]
    total = len(verified)
    wins = sum(1 for e in verified if e['status'] == 'WIN')
    recent = verified[:20]
    rw = sum(1 for e in recent if e['status'] == 'WIN')

    return {
        'accuracy': round((wins / total) * 100, 2) if total > 0 else 0,
        'recentAccuracy': round((rw / len(recent)) * 100, 2) if recent else 0,
        'totalPredictions': total,
        'wins': wins
    }
