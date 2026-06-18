import json
import os
import time
from collections import defaultdict, Counter

from config import DATA_DIR

AUTOLEARN_FILE = os.path.join(DATA_DIR, 'autolearn_patterns.json')
PATTERN_MEMORY_FILE = os.path.join(DATA_DIR, 'pattern_memory.json')

KNOWN_PATTERNS = [
    'streak', 'zigzag', 'alternating', 'repeating_2', 'repeating_3',
    'trend_up', 'trend_down', 'cycle_3', 'cycle_4', 'skip_2', 'skip_3',
    'sequential_up', 'sequential_down', 'mirror', 'random'
]


def _default_memory():
    return {
        'patterns': {},
        'history': [],
        'corrections': [],
        'accuracy': {},
        'totalPredictions': 0,
        'totalCorrect': 0,
        'processedPeriods': [],
        'version': 2,
        'lastUpdated': 0,
    }


def load_memory():
    if os.path.exists(AUTOLEARN_FILE):
        try:
            with open(AUTOLEARN_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return _default_memory()


def save_memory(mem):
    try:
        tmp = AUTOLEARN_FILE + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(mem, f, indent=2)
        os.replace(tmp, AUTOLEARN_FILE)
    except Exception:
        pass


def detect_active_pattern(results):
    cats = [r['category'] for r in results[:30]] if results else []
    if len(cats) < 5:
        return {'patternType': 'unknown', 'confidence': 0, 'direction': None, 'score': 0, 'details': {}}

    n = len(cats)
    details = {}

    alt_count = sum(1 for i in range(1, n) if cats[i] != cats[i-1])
    alt_ratio = alt_count / (n - 1) if n > 1 else 0
    details['altRatio'] = round(alt_ratio, 3)

    streak_len = 1
    for i in range(1, min(n, 15)):
        if cats[i] == cats[i-1]:
            streak_len += 1
        else:
            break
    details['streakLen'] = streak_len

    big_count = cats.count('BIG')
    big_ratio = big_count / n
    details['bigRatio'] = round(big_ratio, 3)

    # Sequential patterns (1234, 5678 etc.)
    nums = []
    for r in results[:15]:
        try:
            nums.append(int(r.get('number', -1)))
        except Exception:
            nums.append(-1)
    nums = [n for n in nums if 0 <= n <= 9]
    details['numCount'] = len(nums)

    seq_up = seq_down = 0
    for i in range(1, len(nums)):
        if nums[i] > nums[i-1]:
            seq_up += 1
        elif nums[i] < nums[i-1]:
            seq_down += 1
    seq_ratio = seq_up / max(len(nums)-1, 1) if len(nums) > 1 else 0
    details['seqUpRatio'] = round(seq_up / max(len(nums)-1, 1), 3) if len(nums) > 1 else 0
    details['seqDownRatio'] = round(seq_down / max(len(nums)-1, 1), 3) if len(nums) > 1 else 0

    # Detect pattern type
    detected_type = 'random'
    direction = None
    score = 0

    # Streak detection
    if streak_len >= 4:
        detected_type = 'streak'
        direction = cats[0]
        score = min(50 + streak_len * 10, 95)
    elif streak_len >= 3 and alt_ratio < 0.4:
        detected_type = 'streak'
        direction = cats[0]
        score = min(50 + streak_len * 8, 90)
    # Alternating / zigzag
    elif alt_ratio >= 0.75 and n >= 6:
        detected_type = 'zigzag'
        direction = 'SMALL' if cats[0] == 'BIG' else 'BIG'
        score = min(60 + alt_ratio * 30, 92)
    elif alt_ratio >= 0.6 and n >= 6:
        detected_type = 'alternating'
        direction = 'SMALL' if cats[0] == 'BIG' else 'BIG'
        score = min(50 + alt_ratio * 25, 85)
    # Repeating 2 pattern (ABABAB)
    elif n >= 6:
        ab_match = sum(1 for i in range(2, n) if cats[i] == cats[i-2])
        ab_ratio = ab_match / max(n-2, 1)
        details['abRatio'] = round(ab_ratio, 3)
        if ab_ratio >= 0.8:
            detected_type = 'repeating_2'
            direction = cats[1] if len(cats) > 1 else cats[0]
            score = min(55 + ab_ratio * 35, 92)
    # Repeating 3 pattern (ABCABC)
    if n >= 9 and detected_type == 'random':
        abc_match = sum(1 for i in range(3, n) if cats[i] == cats[i-3])
        abc_ratio = abc_match / max(n-3, 1)
        details['abcRatio'] = round(abc_ratio, 3)
        if abc_ratio >= 0.75:
            detected_type = 'repeating_3'
            direction = cats[2] if len(cats) > 2 else cats[0]
            score = min(50 + abc_ratio * 40, 90)
    # Skip-2 pattern
    if n >= 6 and detected_type == 'random':
        skip2_match = sum(1 for i in range(2, n) if cats[i] == cats[i-2] and cats[i] != cats[i-1])
        skip2_ratio = skip2_match / max(n-2, 1)
        details['skip2Ratio'] = round(skip2_ratio, 3)
        if skip2_ratio >= 0.7:
            detected_type = 'skip_2'
            direction = 'SMALL' if cats[0] == 'BIG' else 'BIG'
            score = min(50 + skip2_ratio * 30, 85)
    # Cycle-3 pattern
    if n >= 9 and detected_type == 'random':
        cycle3 = sum(1 for i in range(3, n) if cats[i] == cats[i-3])
        c3_ratio = cycle3 / max(n-3, 1)
        details['c3Ratio'] = round(c3_ratio, 3)
        if c3_ratio >= 0.7:
            detected_type = 'cycle_3'
            direction = cats[2] if len(cats) > 2 else cats[0]
            score = min(50 + c3_ratio * 30, 85)
    # Sequential number patterns
    if len(nums) >= 5 and detected_type == 'random':
        if seq_ratio > 0.7 and len(nums) >= 4:
            detected_type = 'sequential_up'
            direction = 'BIG' if nums[0] >= 5 else 'SMALL'
            score = min(55 + seq_ratio * 30, 88)
        elif seq_down / max(len(nums)-1, 1) > 0.7:
            detected_type = 'sequential_down'
            direction = 'SMALL' if nums[0] <= 4 else 'BIG'
            score = min(55 + (seq_down / max(len(nums)-1, 1)) * 30, 88)
    # Trend detection
    if detected_type == 'random' and n >= 10:
        recent_big = sum(1 for c in cats[:5] if c == 'BIG')
        older_big = sum(1 for c in cats[5:10] if c == 'BIG')
        if recent_big >= 4 and older_big <= 2:
            detected_type = 'trend_up'
            direction = 'BIG'
            score = min(55 + (recent_big - older_big) * 10, 88)
        elif recent_big <= 1 and older_big >= 3:
            detected_type = 'trend_down'
            direction = 'SMALL'
            score = min(55 + (older_big - recent_big) * 10, 88)
    # Mirror pattern (BIG->SMALL->BIG->SMALL perfectly)
    if n >= 4 and detected_type == 'random':
        mirror = True
        for i in range(1, min(n, 8)):
            if cats[i] == cats[i-1]:
                mirror = False
                break
        if mirror:
            detected_type = 'mirror'
            direction = 'SMALL' if cats[0] == 'BIG' else 'BIG'
            score = 70
            details['mirror'] = True

    details['detectedType'] = detected_type

    return {
        'patternType': detected_type,
        'confidence': round(min(score, 95), 1),
        'direction': direction,
        'score': round(score, 1),
        'details': details,
    }


def match_old_pattern(detected, memory):
    ptype = detected['patternType']
    pat_stats = memory.get('patterns', {}).get(ptype, {})
    if not pat_stats:
        return {
            'matched': False,
            'historicalAccuracy': 0,
            'historicalCount': 0,
            'mapping': None,
        }

    total = pat_stats.get('total', 0)
    correct = pat_stats.get('correct', 0)
    acc = round((correct / total) * 100, 1) if total > 0 else 0

    # Get the most common outcome for this pattern
    outcomes = pat_stats.get('outcomes', {})
    best_outcome = None
    best_outcome_count = 0
    for outcome, count in outcomes.items():
        if count > best_outcome_count:
            best_outcome_count = count
            best_outcome = outcome

    return {
        'matched': total > 0,
        'historicalAccuracy': acc,
        'historicalCount': total,
        'mapping': best_outcome,
        'mappingConfidence': round((best_outcome_count / total) * 100, 1) if total > 0 and best_outcome else 0,
    }


def auto_correct(detected, actual_result, memory):
    ptype = detected['patternType']
    mem = memory.setdefault('patterns', {})
    if ptype not in mem:
        mem[ptype] = {'total': 0, 'correct': 0, 'outcomes': {}, 'lastUsed': 0}

    mem[ptype]['total'] = mem[ptype].get('total', 0) + 1
    mem[ptype]['lastUsed'] = int(time.time())

    was_correct = detected.get('direction') == actual_result
    if was_correct:
        mem[ptype]['correct'] = mem[ptype].get('correct', 0) + 1

    outcomes = mem[ptype].setdefault('outcomes', {})
    outcomes[actual_result] = outcomes.get(actual_result, 0) + 1

    memory['totalPredictions'] = memory.get('totalPredictions', 0) + 1
    if was_correct:
        memory['totalCorrect'] = memory.get('totalCorrect', 0) + 1

    correction_entry = {
        'timestamp': int(time.time()),
        'patternType': ptype,
        'detectedDirection': detected.get('direction'),
        'actualResult': actual_result,
        'wasCorrect': was_correct,
    }
    memory.setdefault('corrections', []).append(correction_entry)
    if len(memory['corrections']) > 1000:
        memory['corrections'] = memory['corrections'][-1000:]

    memory['lastUpdated'] = int(time.time())
    save_memory(memory)

    return was_correct


def get_accuracy(memory):
    patterns = memory.get('patterns', {})
    pat_acc = {}
    for ptype, stats in patterns.items():
        total = stats.get('total', 0)
        correct = stats.get('correct', 0)
        if total > 0:
            pat_acc[ptype] = {
                'accuracy': round((correct / total) * 100, 1),
                'total': total,
                'correct': correct,
            }

    total_preds = memory.get('totalPredictions', 0)
    total_correct = memory.get('totalCorrect', 0)
    overall = round((total_correct / total_preds) * 100, 1) if total_preds > 0 else 0

    return {
        'overallAccuracy': overall,
        'totalPredictions': total_preds,
        'totalCorrect': total_correct,
        'perPattern': pat_acc,
    }


def build_prediction(detected, memory, all_predictions=None):
    ptype = detected['patternType']
    if ptype == 'unknown' or detected['confidence'] < 30:
        return None

    match_info = match_old_pattern(detected, memory)
    direction = detected.get('direction')
    confidence = detected['confidence']

    if match_info['matched'] and match_info['mapping']:
        hist_acc = match_info['historicalAccuracy']
        map_conf = match_info['mappingConfidence']
        if hist_acc > 50:
            direction = match_info['mapping']
            confidence = min(confidence + (hist_acc - 50) * 0.3, 95)
            if map_conf > 60:
                confidence = min(confidence + 5, 95)
        elif hist_acc < 40 and match_info['mapping'] != direction:
            direction = match_info['mapping']
            confidence = max(confidence - 10, 35)
    else:
        # Use rule-based mapping
        if alt_ratio := detected['details'].get('altRatio'):
            if alt_ratio > 0.75:
                pass

    if direction is None:
        return None

    if all_predictions:
        verified = [e for e in all_predictions if e.get('status') in ('WIN', 'LOSS')]
        if len(verified) >= 10:
            recent_wins = sum(1 for e in verified[:10] if e['status'] == 'WIN')
            recent_rate = recent_wins / min(len(verified), 10)
            if recent_rate < 0.3:
                confidence = max(confidence - 10, 30)

    return {
        'prediction': direction,
        'confidence': round(confidence, 1),
        'patternType': ptype,
        'patternConfidence': detected['confidence'],
        'historicalMatched': match_info['matched'],
        'historicalAccuracy': match_info['historicalAccuracy'],
    }


def update_from_verification(all_predictions, memory):
    verified = [e for e in all_predictions if e.get('status') in ('WIN', 'LOSS')]
    recent_verified = verified[:30]
    processed = set(memory.get('processedPeriods', []))
    updated = False
    for entry in recent_verified:
        period = str(entry.get('period', ''))
        if not period or period in processed or entry.get('patternUsed') == 'imported':
            continue
        pred = entry.get('prediction')
        actual = entry.get('actual')
        if not pred or not actual:
            continue
        pp = entry.get('patternPredictions')
        if pp and 'autolearn' in pp:
            al_data = pp['autolearn']
            detected_type = al_data.get('patternType')
            detected_dir = al_data.get('prediction')
        else:
            detected_type = 'ensemble'
            detected_dir = pred
        if detected_type and detected_dir:
            mem = memory.setdefault('patterns', {})
            if detected_type not in mem:
                mem[detected_type] = {'total': 0, 'correct': 0, 'outcomes': {}, 'lastUsed': 0}
            mem[detected_type]['total'] = mem[detected_type].get('total', 0) + 1
            was_correct = detected_dir == actual
            if was_correct:
                mem[detected_type]['correct'] = mem[detected_type].get('correct', 0) + 1
            outcomes = mem[detected_type].setdefault('outcomes', {})
            outcomes[actual] = outcomes.get(actual, 0) + 1
            memory['totalPredictions'] = memory.get('totalPredictions', 0) + 1
            if was_correct:
                memory['totalCorrect'] = memory.get('totalCorrect', 0) + 1
            processed.add(period)
            updated = True
    if updated:
        memory['processedPeriods'] = list(processed)[-5000:]
        save_memory(memory)
