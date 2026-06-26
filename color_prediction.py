import os
import csv
import time
import math
import json
import random
import threading
import numpy as np
from collections import defaultdict, Counter
from config import DATA_DIR
from helpers import get_current_period_1min, fetch_api_data, load_daily_1k_history
from storage import load_predictions_csv
from ml import build_sequence_training_rows, extract_features

COLOR_PREDICTION_HISTORY_CSV = os.path.join(DATA_DIR, 'predict', 'color_prediction_history.csv')
COLOR_CLASSES = ['RED', 'GREEN', 'RED,VIOLET', 'GREEN,VIOLET']
COLOR_LABELS = {c: i for i, c in enumerate(COLOR_CLASSES)}

_color_learner = None
_color_learner_lock = threading.Lock()

COLOR_1M_HISTORY_CSV = os.path.join(DATA_DIR, '1m', 'daily_1k_history.csv')
COLOR_PREDICTIONS_CSV = os.path.join(DATA_DIR, 'predict', 'predictions.csv')
COLOR_PREDICTION_HISTORY_CSV2 = os.path.join(DATA_DIR, 'predict', 'prediction_history.csv')
COLOR_MODEL_HISTORY_CSV = os.path.join(DATA_DIR, 'model', 'model_prediction_history.csv')


def to_int(val):
    try:
        return int(float(str(val)))
    except (ValueError, TypeError):
        return None

def get_number_color(number):
    number = to_int(number)
    if number is None:
        return None
    if number == 0:
        return 'RED,VIOLET'
    if number == 5:
        return 'GREEN,VIOLET'
    return 'GREEN' if number % 2 else 'RED'

def is_color_match(predicted, actual):
    if not predicted or not actual:
        return False
    if predicted == actual:
        return True
    pred_parts = [p.strip() for p in predicted.split(',')]
    act_parts = [p.strip() for p in actual.split(',')]
    for p in pred_parts:
        if p in act_parts:
            return True
    return False

def load_color_predictions_csv(limit=500):
    if not os.path.exists(COLOR_PREDICTION_HISTORY_CSV):
        return []
    predictions = []
    try:
        with open(COLOR_PREDICTION_HISTORY_CSV, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return []
            for row in reader:
                if len(row) < 8:
                    continue
                predictions.append({
                    'period': row[0],
                    'prediction': row[1] if row[1] else None,
                    'status': row[2],
                    'confidence': float(row[3]) if row[3] else 0.0,
                    'actual': row[4] if row[4] else None,
                    'number': row[5] if row[5] else None,
                    'patternUsed': row[6],
                    'timestamp': int(row[7]) if row[7] else int(time.time()),
                    'skipped': row[8] == '1' if len(row) > 8 else False,
                    'skipReason': row[9] if len(row) > 9 else '',
                })
    except Exception:
        return []
    predictions.sort(key=lambda x: int(x['period']) if x['period'].isdigit() else 0, reverse=True)
    return predictions[:limit]

def upsert_color_prediction_history(entry):
    predictions = load_color_predictions_csv(limit=5000)
    by_period = {p['period']: p for p in predictions}
    period = entry['period']
    if period in by_period:
        by_period[period] = {**by_period[period], **entry}
    else:
        by_period[period] = entry
    sorted_preds = sorted(by_period.values(), key=lambda x: int(x['period']) if x['period'].isdigit() else 0)
    tmp = COLOR_PREDICTION_HISTORY_CSV + '.tmp'
    os.makedirs(os.path.dirname(COLOR_PREDICTION_HISTORY_CSV), exist_ok=True)
    try:
        with open(tmp, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['period', 'prediction', 'status', 'confidence', 'actual', 'number', 'patternUsed', 'timestamp', 'skipped', 'skipReason'])
            for p in sorted_preds:
                writer.writerow([p.get('period', ''), p.get('prediction') or '', p.get('status', 'Pending'), p.get('confidence', 0.0), p.get('actual') or '', p.get('number') if p.get('number') is not None else '', p.get('patternUsed', 'ensemble'), p.get('timestamp', int(time.time())), '1' if p.get('skipped') else '0', p.get('skipReason') or ''])
        os.replace(tmp, COLOR_PREDICTION_HISTORY_CSV)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass

def hydrate_color_history(history, game_data):
    if not game_data:
        return history, False
    by_period = {}
    for item in game_data:
        if item.get('period'):
            key = str(int(str(item['period'])[-5:]))
            by_period[key] = item
    changed = False
    for item in history:
        period = str(item.get('period', ''))
        row_status = str(item.get('status', '')).upper()
        if not period or row_status not in ('PENDING', 'SKIP') or item.get('actual') is not None:
            continue
        match = by_period.get(period)
        if not match:
            suffix = period[-3:]
            match = next((row for row in game_data if str(row.get('period', '')).endswith(suffix)), None)
        if not match:
            continue
        number = match.get('number')
        if number is None:
            continue
        try:
            number = int(float(str(number)))
        except ValueError:
            continue
        actual_color = get_number_color(number)
        item['actual'] = actual_color
        item['number'] = number
        item['actualColor'] = actual_color
        if item.get('prediction'):
            item['status'] = 'WIN' if is_color_match(item.get('prediction'), actual_color) else 'LOSS'
        else:
            item['status'] = 'SKIP'
        try:
            upsert_color_prediction_history(item)
        except Exception:
            pass
        changed = True
    return history, changed


def _load_all_numbers():
    numbers = []
    seen = set()
    sources = [
        COLOR_PREDICTION_HISTORY_CSV,
        COLOR_PREDICTION_HISTORY_CSV2,
        COLOR_PREDICTIONS_CSV,
        COLOR_MODEL_HISTORY_CSV,
        COLOR_1M_HISTORY_CSV,
    ]
    for path in sources:
        if not os.path.exists(path):
            continue
        try:
            with open(path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    period = str(row.get('period') or '').strip()
                    if not period or period in seen:
                        continue
                    num = row.get('number') or row.get('winningNumber') or ''
                    try:
                        num = int(float(num))
                        seen.add(period)
                        numbers.append(num)
                    except (ValueError, TypeError):
                        pass
        except Exception:
            pass
    return numbers


class DeepColorAnalyzer:
    def __init__(self):
        self.colors = []
        self.numbers = []
        self.cache = {}

    def feed(self, colors, numbers):
        self.colors = colors
        self.numbers = numbers

    def analyze_sequence_matching(self, recent, depth=20):
        if len(self.colors) < 10 or len(recent) < 2:
            return {c: 0.25 for c in COLOR_CLASSES}
        votes = {c: 0.0 for c in COLOR_CLASSES}
        total_weight = 0.0
        for n in range(1, depth + 1):
            if len(recent) < n:
                continue
            target = tuple(reversed(recent[:n]))
            counts = {c: 0 for c in COLOR_CLASSES}
            for i in range(n, len(self.colors)):
                if tuple(self.colors[i - n:i]) == target:
                    nxt = self.colors[i]
                    if nxt in counts:
                        counts[nxt] += 1
            tot = sum(counts.values())
            if tot:
                weight = min(tot, 80) * (1.5 ** (n / 4))
                if tot >= 3:
                    weight *= 1.3
                for c in COLOR_CLASSES:
                    votes[c] += ((counts[c] + 1) / (tot + 4)) * weight
                total_weight += weight
            rtarget = tuple(recent[:n])
            rcounts = {c: 0 for c in COLOR_CLASSES}
            rcolors = list(reversed(self.colors))
            for i in range(n, len(rcolors)):
                if tuple(rcolors[i - n:i]) == rtarget:
                    nxt = rcolors[i]
                    if nxt in rcounts:
                        rcounts[nxt] += 1
            rtot = sum(rcounts.values())
            if rtot:
                rweight = min(rtot, 60) * (1.3 ** (n / 4)) * 0.8
                for c in COLOR_CLASSES:
                    votes[c] += ((rcounts[c] + 1) / (rtot + 4)) * rweight
                total_weight += rweight
        if total_weight > 0:
            return {c: votes[c] / total_weight for c in COLOR_CLASSES}
        counts = {c: self.colors.count(c) for c in COLOR_CLASSES}
        t = sum(counts.values()) or 1
        return {c: counts[c] / t for c in COLOR_CLASSES}

    def analyze_trend_regime(self, recent):
        if len(recent) < 6:
            return 'MIXED', {}
        streak_color, streak_len = None, 0
        for c in recent:
            if streak_color is None:
                streak_color, streak_len = c, 1
            elif c == streak_color:
                streak_len += 1
            else:
                break
        alt_count = 0
        for i in range(1, min(len(recent), 12)):
            if recent[i] != recent[i - 1]:
                alt_count += 1
        alt_ratio = alt_count / max(len(recent[:12]) - 1, 1)
        if streak_len >= 4:
            regime = 'STREAK'
        elif alt_ratio >= 0.7:
            regime = 'ZIGZAG'
        else:
            regime = 'MIXED'
        big_count = sum(1 for c in recent if c in ('RED', 'RED,VIOLET'))
        green_count = sum(1 for c in recent if c in ('GREEN', 'GREEN,VIOLET'))
        bias = 'RED' if big_count > green_count else 'GREEN' if green_count > big_count else 'NEUTRAL'
        return regime, {'streakColor': streak_color, 'streakLen': streak_len, 'altRatio': round(alt_ratio, 2), 'bias': bias}

    def analyze_number_frequency(self, numbers_hist, window=200):
        if not numbers_hist:
            return {c: 0.25 for c in COLOR_CLASSES}
        recent_nums = numbers_hist[-min(window, len(numbers_hist)):]
        color_counts = Counter(get_number_color(n) for n in recent_nums if get_number_color(n) is not None)
        total = sum(color_counts.values()) or 1
        probs = {}
        for c in COLOR_CLASSES:
            freq = color_counts.get(c, 0) / total
            gaps = []
            gap = 0
            for n in reversed(numbers_hist):
                gc = get_number_color(n)
                if gc == c:
                    gaps.append(gap)
                    gap = 0
                else:
                    gap += 1
            avg_gap = sum(gaps) / max(len(gaps), 1) if gaps else 999
            expected_gap = total / max(color_counts.get(c, 1), 1)
            due = max(0, (avg_gap - expected_gap) / expected_gap) if expected_gap > 0 else 0
            due_boost = min(due, 2.0)
            probs[c] = freq * (1 + due_boost * 0.5)
        t = sum(probs.values()) or 1
        return {c: probs[c] / t for c in COLOR_CLASSES}

    def analyze_momentum(self, recent):
        if len(recent) < 10:
            return {c: 0.0 for c in COLOR_CLASSES}
        windows = [3, 5, 8, 12, 20]
        momentum = {}
        for c in COLOR_CLASSES:
            score = 0.0
            for w in windows:
                if len(recent) < w:
                    continue
                recent_w = recent[:w]
                older_w = recent[w:min(w * 2, len(recent))]
                if not older_w:
                    continue
                r_rate = recent_w.count(c) / len(recent_w)
                o_rate = older_w.count(c) / len(older_w)
                diff = r_rate - o_rate
                weight = math.sqrt(w)
                if r_rate > 0.5 and diff > 0:
                    score += diff * weight * 2
                elif r_rate < 0.3 and diff < -0.1:
                    score -= abs(diff) * weight
                else:
                    score += diff * weight
            momentum[c] = score
        return momentum

    def analyze_pattern_clusters(self, recent):
        if len(self.colors) < 20 or len(recent) < 3:
            return {c: 0.25 for c in COLOR_CLASSES}
        recent_str = ''.join(recent[:6])
        pattern_scores = {c: 0.0 for c in COLOR_CLASSES}
        match_count = 0
        for i in range(3, len(self.colors) - 1):
            end = min(i + 6, len(self.colors))
            window = self.colors[i - 3:end]
            window_str = ''.join(window[:len(recent_str)])
            if window_str == recent_str:
                nxt = self.colors[end] if end < len(self.colors) else None
                if nxt and nxt in pattern_scores:
                    pattern_scores[nxt] += 1
                    match_count += 1
        if match_count > 0:
            t = sum(pattern_scores.values()) or 1
            return {c: pattern_scores[c] / t for c in COLOR_CLASSES}
        return {c: 0.25 for c in COLOR_CLASSES}

    def analyze_exhaustive(self, recent, numbers_hist):
        seq = self.analyze_sequence_matching(recent, depth=25)
        freq = self.analyze_number_frequency(numbers_hist)
        momentum = self.analyze_momentum(recent)
        clusters = self.analyze_pattern_clusters(recent)
        regime, regime_info = self.analyze_trend_regime(recent)
        combined = {}
        for c in COLOR_CLASSES:
            s = seq.get(c, 0.25)
            f = freq.get(c, 0.25)
            m = max(0, momentum.get(c, 0) * 0.02 + 0.25)
            cl = clusters.get(c, 0.25)
            if regime == 'STREAK' and regime_info.get('streakColor') == c:
                m = max(m, 0.35)
            elif regime == 'ZIGZAG':
                alt = [x for x in COLOR_CLASSES if x != c]
                if any(recent[i] != recent[i - 1] for i in range(1, min(4, len(recent)))):
                    pass
            combined[c] = s * 0.30 + f * 0.20 + m * 0.25 + cl * 0.25
        t = sum(combined.values()) or 1
        return {c: combined[c] / t for c in COLOR_CLASSES}, regime, regime_info


class ColorDeepLearner:
    def __init__(self):
        self.reset()

    def reset(self):
        self.per_model = defaultdict(lambda: {'wins': 0, 'losses': 0, 'recent': []})
        self.per_color = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'losses': 0}))
        self.consecutive_losses = 0
        self.loss_pattern = []
        self.total_wins = 0
        self.total_losses = 0
        self.weights = {'deep_seq': 30, 'xgboost': 25, 'frequency': 15, 'markov': 15, 'exhaustive': 35}
        self.recent_predictions = []
        self.regime_history = []

    def learn(self, prediction, actual, model_name='ensemble'):
        if not actual or not prediction:
            return
        win = 1 if is_color_match(prediction, actual) else 0
        if win:
            self.total_wins += 1
            self.per_model[model_name]['wins'] += 1
            self.per_color[model_name][prediction]['wins'] += 1
            self.consecutive_losses = 0
        else:
            self.total_losses += 1
            self.per_model[model_name]['losses'] += 1
            self.per_color[model_name][prediction]['losses'] += 1
            self.consecutive_losses += 1
            self.loss_pattern.append({'pred': prediction, 'actual': actual})
            if len(self.loss_pattern) > 30:
                self.loss_pattern.pop(0)
        self.per_model[model_name]['total'] = self.per_model[model_name]['wins'] + self.per_model[model_name]['losses']
        self.per_model[model_name]['recent'].append(win)
        if len(self.per_model[model_name]['recent']) > 100:
            self.per_model[model_name]['recent'] = self.per_model[model_name]['recent'][-100:]
        self.recent_predictions.append({'pred': prediction, 'actual': actual, 'win': win})
        if len(self.recent_predictions) > 200:
            self.recent_predictions = self.recent_predictions[-200:]
        self._adjust_weights()

    def _adjust_weights(self):
        for model in self.weights:
            acc = self.per_model[model]
            if acc['total'] < 10:
                continue
            recent = acc['recent'][-30:]
            rw = sum(recent)
            rt = len(recent)
            ra = rw / max(rt, 1)
            self.weights[model] = max(5, min(50, 8 + ra * 45))

    def get_weight(self, name):
        return self.weights.get(name, 20)

    def total_weight(self):
        return sum(self.weights.values())

    def get_accuracy(self, n=50):
        recent = []
        for m in self.weights:
            recent.extend(self.per_model[m]['recent'][-n:])
        if not recent:
            return 50.0
        return round((sum(recent) / len(recent)) * 100, 2)

    def get_loss_correction(self, combined_prob):
        if self.consecutive_losses == 0 or not self.loss_pattern:
            return None
        recent_losses = [lp for lp in self.loss_pattern[-5:] if lp['pred'] != lp['actual']]
        if not recent_losses:
            return None
        wrong_colors = [lp['pred'] for lp in recent_losses]
        counter = Counter(wrong_colors)
        worst = counter.most_common(1)[0][0]
        if counter[worst] >= 2:
            alt = sorted([c for c in COLOR_CLASSES if c != worst], key=lambda c: combined_prob.get(c, 0), reverse=True)
            if alt:
                return alt[0]
        return None

    def get_worst_color(self):
        if self.total_wins + self.total_losses < 20:
            return None
        rates = {}
        for c in COLOR_CLASSES:
            w = self.per_color.get('deep_ensemble', {}).get(c, {}).get('wins', 0)
            l = self.per_color.get('deep_ensemble', {}).get(c, {}).get('losses', 0)
            t = w + l
            if t >= 5:
                rates[c] = w / t
        if not rates:
            return None
        worst = min(rates, key=rates.get)
        if rates[worst] < 0.4:
            return worst
        return None

    def record_regime(self, regime):
        self.regime_history.append(regime)
        if len(self.regime_history) > 100:
            self.regime_history = self.regime_history[-100:]

    def best_regime_strategy(self):
        if len(self.regime_history) < 10:
            return None
        recent_regimes = self.regime_history[-20:]
        streak_count = sum(1 for r in recent_regimes if r == 'STREAK')
        zigzag_count = sum(1 for r in recent_regimes if r == 'ZIGZAG')
        if streak_count > zigzag_count * 2:
            return 'STREAK'
        if zigzag_count > streak_count * 2:
            return 'ZIGZAG'
        return 'MIXED'


def _load_all_numbers_init():
    numbers = []
    seen = set()
    sources = [
        COLOR_PREDICTION_HISTORY_CSV,
        COLOR_PREDICTION_HISTORY_CSV2,
        COLOR_PREDICTIONS_CSV,
        COLOR_MODEL_HISTORY_CSV,
        COLOR_1M_HISTORY_CSV,
    ]
    for path in sources:
        if not os.path.exists(path):
            continue
        try:
            with open(path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    period = str(row.get('period') or '').strip()
                    if not period or period in seen:
                        continue
                    num = row.get('number') or row.get('winningNumber') or ''
                    try:
                        num = int(float(num))
                        seen.add(period)
                        numbers.append(num)
                    except (ValueError, TypeError):
                        pass
        except Exception:
            pass
    return numbers


def _get_learner():
    global _color_learner
    with _color_learner_lock:
        if _color_learner is None:
            _color_learner = ColorDeepLearner()
            csv_history = load_color_predictions_csv(limit=2000)
            for h in csv_history:
                if h.get('status') in ('WIN', 'LOSS') and h.get('prediction') and h.get('actual'):
                    act = h['actual']
                    if act in COLOR_CLASSES:
                        _color_learner.learn(h['prediction'], act, h.get('patternUsed', 'deep_ensemble'))
            all_numbers = _load_all_numbers_init()
            if all_numbers:
                colors_from_nums = [get_number_color(n) for n in all_numbers if get_number_color(n) is not None]
                cc = Counter(colors_from_nums)
                for c in COLOR_CLASSES:
                    _color_learner.per_color['history'][c]['wins'] = cc.get(c, 0)
                    _color_learner.per_color['history'][c]['losses'] = max(1, len(colors_from_nums) - cc.get(c, 0))
                    for _ in range(min(cc.get(c, 0), 50)):
                        _color_learner.learn(c, c, 'history')
        return _color_learner


def get_color_prediction_payload(body):
    game_data = fetch_api_data()
    daily_rows = load_daily_1k_history(limit=None)
    all_predictions = load_predictions_csv()
    rows = build_sequence_training_rows(all_predictions, daily_rows)
    history = load_color_predictions_csv(limit=500)
    if 'error' not in game_data and game_data:
        history, _ = hydrate_color_history(history, game_data)
    current_period = get_current_period_1min()
    existing_pred = next((h for h in history if str(h['period']) == str(current_period)), None)
    learner = _get_learner()

    if existing_pred and (existing_pred.get('actual') or existing_pred.get('status') in ('WIN', 'LOSS')):
        pr = existing_pred
    else:
        pr = _build_prediction(rows, history, game_data, learner, current_period)

    if pr.get('actual') and pr.get('prediction'):
        learner.learn(pr['prediction'], pr['actual'], pr.get('patternUsed', 'deep_ensemble'))

    stats_h = [h for h in history if h.get('status') in ('WIN', 'LOSS')]
    wins = sum(1 for h in stats_h if h['status'] == 'WIN')
    losses = sum(1 for h in stats_h if h['status'] == 'LOSS')
    settled = wins + losses
    wr = round((wins / settled) * 100, 2) if settled else 0.0
    sc, scnt = None, 0
    for item in history:
        s = str(item.get('status', '')).upper()
        if s not in ('WIN', 'LOSS'):
            continue
        if sc is None:
            sc, scnt = s, 1
        elif s == sc:
            scnt += 1
        else:
            break

    worst = learner.get_worst_color()

    stats = {
        'totalWins': wins, 'totalLosses': losses, 'winRate': wr,
        'streak': f"{scnt} {sc or 'None'}", 'pending': sum(1 for h in history if h.get('status') == 'Pending'),
        'totalPredictions': len(history), 'learnerAccuracy': learner.get_accuracy(),
        'autoWeights': dict(learner.weights), 'consecutiveLosses': learner.consecutive_losses,
        'worstColor': worst, 'totalLearned': learner.total_wins + learner.total_losses,
    }

    fh = []
    for h in history[:20]:
        num = h.get('number')
        ac = get_number_color(num) if num is not None else None
        fh.append({
            'period': h['period'], 'prediction': h['prediction'], 'status': h['status'],
            'actual': h['actual'], 'number': to_int(num), 'actualNumber': to_int(num),
            'color': h['prediction'], 'actualColor': ac, 'confidence': h['confidence'],
            'patternUsed': h['patternUsed'], 'timestamp': h['timestamp'],
        })

    return {
        'predictionResult': {
            'period': pr.get('period'), 'prediction': pr.get('prediction'),
            'status': pr.get('status'), 'skipped': pr.get('skipped', False),
            'skipReason': pr.get('skipReason') or '',
        },
        'predictionDetails': {
            'gameType': 'Wingo 1 Min Color',
            'confidence': pr.get('confidence'), 'actual': pr.get('actual'),
            'number': to_int(pr.get('number')), 'actualNumber': to_int(pr.get('number')),
            'color': pr.get('prediction'), 'actualColor': pr.get('actual'),
            'explanation': pr.get('explanation', ''),
            'recommendation': f"Place bet on {pr.get('prediction')}.",
            'lossLog': {
                'consecutiveLosses': learner.consecutive_losses,
                'learnerAccuracy': learner.get_accuracy(),
                'autoWeights': dict(learner.weights),
                'worstColor': worst,
            },
        },
        'stats': stats,
        'history': fh,
        'historySource': {'file': 'color_prediction_history.csv', 'live': True, 'rows': len(history), 'limit': 20},
    }


def _prepare_data(rows, game_data):
    colors = []
    numbers = []
    for row in rows:
        num = row.get('number')
        try:
            num = int(float(num))
            numbers.append(num)
            c = get_number_color(num)
            if c:
                colors.append(c)
        except (ValueError, TypeError):
            pass
    if game_data and 'error' not in game_data:
        for item in game_data:
            num = item.get('number')
            try:
                num = int(float(num))
                if num not in numbers:
                    numbers.append(num)
                    c = get_number_color(num)
                    if c:
                        colors.append(c)
            except (ValueError, TypeError):
                pass
    return colors, numbers


def _build_prediction(rows, history, game_data, learner, current_period):
    colors, numbers = _prepare_data(rows, game_data)

    recent_slice = []
    if 'error' not in game_data and game_data:
        recent_slice = [get_number_color(r.get('number')) for r in game_data[:150] if r.get('number') is not None]
    else:
        recent_slice = [c for c in colors[-150:]]
    recent_slice = [c for c in recent_slice if c is not None]

    analyzer = DeepColorAnalyzer()
    analyzer.feed(colors, numbers)

    # ── 1. Deep sequence matching ──
    prob_seq = analyzer.analyze_sequence_matching(recent_slice, depth=25)

    # ── 2. Exhaustive analysis (regime + momentum + frequency + clusters) ──
    prob_exhaustive, regime, regime_info = analyzer.analyze_exhaustive(recent_slice, numbers)
    learner.record_regime(regime)

    # ── 3. XGBoost ──
    prob_xgb = _predict_xgboost(rows, recent_slice)

    # ── 4. Frequency ──
    prob_freq = _predict_frequency(colors, numbers)

    # ── 5. Markov ──
    prob_markov = _predict_markov(colors)

    # ── 6. Trend override ──
    prob_trend = _trend_analysis(colors, recent_slice, regime, regime_info)

    # ── Fusion ──
    strategy = learner.best_regime_strategy()
    w_seq = learner.get_weight('deep_seq')
    w_xgb = learner.get_weight('xgboost')
    w_freq = learner.get_weight('frequency')
    w_markov = learner.get_weight('markov')
    w_exh = learner.get_weight('exhaustive')
    w_trend = 22

    if strategy == 'STREAK':
        w_seq += 10; w_markov += 5; w_trend += 10
    elif strategy == 'ZIGZAG':
        w_exh += 10; w_seq += 5

    combined = {}
    for c in COLOR_CLASSES:
        combined[c] = (
            w_seq * prob_seq.get(c, 0.25) +
            w_xgb * prob_xgb.get(c, 0.25) +
            w_freq * prob_freq.get(c, 0.25) +
            w_markov * prob_markov.get(c, 0.25) +
            w_exh * prob_exhaustive.get(c, 0.25) +
            w_trend * prob_trend.get(c, 0.25)
        )
    total_w = sum([w_seq, w_xgb, w_freq, w_markov, w_exh, w_trend])
    combined = {c: combined[c] / total_w for c in COLOR_CLASSES}

    predicted = max(COLOR_CLASSES, key=lambda c: combined[c])
    max_prob = combined[predicted]
    second_prob = sorted(combined.values(), reverse=True)[1] if len(combined) > 1 else 0
    edge = max_prob - second_prob

    correction = learner.get_loss_correction(combined)
    if correction:
        predicted = correction
        max_prob = combined.get(correction, 0.30)

    worst = learner.get_worst_color()
    if worst and worst == predicted and learner.consecutive_losses >= 1:
        alt = sorted([c for c in COLOR_CLASSES if c != worst], key=lambda c: combined[c], reverse=True)
        if alt and combined.get(alt[0], 0) > 0.15:
            predicted = alt[0]
            max_prob = combined[alt[0]]

    # ── confidence ──
    conf = 55.0 + (max_prob - 0.25) * 110.0 + edge * 60.0
    acc = learner.get_accuracy(30)
    conf += (acc - 50) * 0.3
    if learner.consecutive_losses >= 1:
        conf += 8
    if learner.consecutive_losses >= 2:
        conf += 5
    conf = max(55.0, min(97.0, conf))

    explanations = []
    for name, prob in [('DeepSeq', prob_seq), ('XGBoost', prob_xgb), ('Frequency', prob_freq), ('Markov', prob_markov), ('Exhaustive', prob_exhaustive), ('Trend', prob_trend)]:
        pct = round(prob.get(predicted, 0) * 100, 1)
        explanations.append(f"{name}={pct}%")

    explanation = f"Deep ensemble → {predicted} | Regime={regime} | " + " | ".join(explanations)
    if correction:
        explanation += f" | Loss-corrected from {learner.loss_pattern[-1]['pred'] if learner.loss_pattern else '?'}"

    model_name = f"deep_{regime.lower()}_{strategy.lower() if strategy else 'adaptive'}"

    pr = {
        'period': current_period, 'prediction': predicted, 'status': 'Pending',
        'confidence': round(conf, 2), 'actual': None, 'number': None,
        'patternUsed': model_name, 'timestamp': int(time.time()),
        'skipped': False, 'skipReason': '', 'explanation': explanation,
    }
    upsert_color_prediction_history(pr)
    return pr


def _predict_xgboost(rows, recent_slice):
    try:
        X, y = _build_xg_data(rows)
        if X is None or len(np.unique(y)) < 2:
            return {c: 0.25 for c in COLOR_CLASSES}
        try:
            from xgboost import XGBClassifier
            clf = XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.1, random_state=42, eval_metric='mlogloss', objective='multi:softprob', num_class=4, n_jobs=1)
            clf.fit(X, y)
        except Exception:
            from sklearn.ensemble import ExtraTreesClassifier
            clf = ExtraTreesClassifier(n_estimators=300, max_depth=8, random_state=42, n_jobs=1)
            clf.fit(X, y)
        feats = _build_query_feats(recent_slice)
        if feats is None:
            return {c: 0.25 for c in COLOR_CLASSES}
        probs = clf.predict_proba(feats.reshape(1, -1))[0]
        return {c: float(probs[i]) if i < len(probs) else 0.25 for i, c in enumerate(COLOR_CLASSES)}
    except Exception:
        return {c: 0.25 for c in COLOR_CLASSES}


def _build_xg_data(rows):
    if len(rows) < 30:
        return None, None
    X, y = [], []
    for i in range(15, len(rows)):
        past = rows[:i]
        target_num = rows[i].get('number')
        if target_num is None:
            continue
        try:
            target_num = int(float(str(target_num)))
        except ValueError:
            continue
        tc = get_number_color(target_num)
        if tc not in COLOR_LABELS:
            continue
        feats = extract_features(past[-100:])
        if feats is not None:
            X.append(feats)
            y.append(COLOR_LABELS[tc])
    if len(X) < 10:
        return None, None
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def _build_query_feats(recent_slice):
    if not recent_slice:
        return None
    cats = []
    for c in recent_slice[:100]:
        if c in ('RED', 'RED,VIOLET'):
            cats.append(1)
        elif c in ('GREEN', 'GREEN,VIOLET'):
            cats.append(0)
    if not cats:
        return None
    mock = [{'category': 'BIG' if v else 'SMALL', 'actual': 'BIG' if v else 'SMALL'} for v in cats]
    return extract_features(mock)


def _predict_frequency(colors, numbers):
    if not colors:
        return {c: 0.25 for c in COLOR_CLASSES}
    n = len(colors)
    short = colors[-min(20, n):]
    med = colors[-min(100, n):]
    long_term = colors[-min(2000, n):]
    probs = {}
    for c in COLOR_CLASSES:
        fs = short.count(c) / max(len(short), 1)
        fm = med.count(c) / max(len(med), 1)
        fl = long_term.count(c) / max(len(long_term), 1)
        gaps = []
        g = 0
        for item in reversed(colors):
            if item == c:
                gaps.append(g)
                g = 0
            else:
                g += 1
        avg_gap = sum(gaps) / max(len(gaps), 1) if gaps else 999
        exp_gap = 1.0 / max(fl, 0.01)
        due = max(0, (avg_gap - exp_gap) / max(exp_gap, 1))
        due_boost = min(due, 2.5)
        base = 0.15 * fs + 0.25 * fm + 0.60 * fl
        probs[c] = base * (1 + due_boost * 0.3)
    t = sum(probs.values()) or 1
    return {c: probs[c] / t for c in COLOR_CLASSES}


def _predict_markov(colors):
    if len(colors) < 10:
        return {c: 0.25 for c in COLOR_CLASSES}
    t1 = {c: {c2: 0 for c2 in COLOR_CLASSES} for c in COLOR_CLASSES}
    t2, t3 = {}, {}
    for i in range(1, len(colors)):
        p, q = colors[i - 1], colors[i]
        if p in t1 and q in t1[p]:
            t1[p][q] += 1
    for i in range(2, len(colors)):
        s = (colors[i - 2], colors[i - 1]); nx = colors[i]
        if s not in t2:
            t2[s] = {c2: 0 for c2 in COLOR_CLASSES}
        if nx in t2[s]:
            t2[s][nx] += 1
    for i in range(3, len(colors)):
        s = (colors[i - 3], colors[i - 2], colors[i - 1]); nx = colors[i]
        if s not in t3:
            t3[s] = {c2: 0 for c2 in COLOR_CLASSES}
        if nx in t3[s]:
            t3[s][nx] += 1

    mc = {c: 0.0 for c in COLOR_CLASSES}
    w = 0.0
    if len(colors) >= 3:
        s3 = (colors[-3], colors[-2], colors[-1])
        if s3 in t3:
            ct = t3[s3]; tot = sum(ct.values())
            if tot >= 2:
                for c in COLOR_CLASSES:
                    mc[c] = (ct[c] + 1) / (tot + 4)
                w = 3.0
    if w < 2 and len(colors) >= 2:
        s2 = (colors[-2], colors[-1])
        if s2 in t2:
            ct = t2[s2]; tot = sum(ct.values())
            if tot >= 3:
                for c in COLOR_CLASSES:
                    mc[c] = (ct[c] + 1) / (tot + 4)
                w = 2.0
    if w < 1:
        ct = t1.get(colors[-1], {}); tot = sum(ct.values())
        if tot > 0:
            for c in COLOR_CLASSES:
                mc[c] = (ct[c] + 1) / (tot + 4)
        else:
            for c in COLOR_CLASSES:
                mc[c] = 0.25
        w = 1.0

    pb = {c: 0.0 for c in COLOR_CLASSES}
    if len(colors) >= 6:
        if len(set(colors[-6:])) == 1:
            opp = 'GREEN' if colors[-1] == 'RED' else 'RED'
            pb[opp] += 0.60; pb[colors[-1]] += 0.10
        elif len(set(colors[-4:])) == 1:
            opp = 'GREEN' if colors[-1] == 'RED' else 'RED'
            pb[opp] += 0.40; pb[colors[-1]] += 0.10
        if colors[-4] == colors[-2] and colors[-3] == colors[-1] and colors[-2] != colors[-1]:
            pb[colors[-2]] += 0.30
    combined = {c: mc[c] + pb[c] for c in COLOR_CLASSES}
    t = sum(combined.values()) or 1
    return {c: combined[c] / t for c in COLOR_CLASSES}


def _trend_analysis(colors, recent, regime, regime_info):
    if not colors:
        return {c: 0.25 for c in COLOR_CLASSES}
    probs = {c: 0.25 for c in COLOR_CLASSES}
    if regime == 'STREAK' and regime_info.get('streakColor'):
        sc = regime_info['streakColor']
        sl = regime_info.get('streakLen', 0)
        opp = 'GREEN' if 'RED' in sc else 'RED'
        if sl >= 5:
            probs[opp] = 0.60
            if 'VIOLET' in sc:
                probs[opp + ',VIOLET'] = 0.25 if opp + ',VIOLET' in probs else 0.20
        elif sl >= 3:
            probs[opp] = 0.45
            probs[sc] = 0.30
        else:
            probs[sc] = 0.50
    elif regime == 'ZIGZAG':
        last = recent[0] if recent else None
        if last:
            opp = 'GREEN' if 'RED' in last else 'RED'
            probs[opp] = 0.60
            if 'VIOLET' in last:
                probs[opp + ',VIOLET'] = 0.30 if opp + ',VIOLET' in probs else 0.20
    else:
        bias = regime_info.get('bias', 'NEUTRAL')
        if bias == 'RED':
            probs['RED'] = 0.35
            probs['RED,VIOLET'] = 0.25
            probs['GREEN'] = 0.25
            probs['GREEN,VIOLET'] = 0.15
        elif bias == 'GREEN':
            probs['GREEN'] = 0.35
            probs['GREEN,VIOLET'] = 0.25
            probs['RED'] = 0.25
            probs['RED,VIOLET'] = 0.15
    t = sum(probs.values()) or 1
    return {c: probs[c] / t for c in COLOR_CLASSES}
