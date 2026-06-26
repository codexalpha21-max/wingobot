import json
import os
import time
import traceback
from collections import OrderedDict

from config import *
from helpers import get_current_period_1min
from storage import *
from prediction import *
from ml import get_ml_accuracy, train_model, get_model_summary
from autolearn import load_memory, get_accuracy as get_autolearn_accuracy, update_from_verification
from model_brain import learn_all as brain_learn

LEETS_FILE = os.path.join(DATA_DIR, 'leets.json')
PUBLIC_HISTORY_LIMIT = 20


def get_number_color(number):
    try:
        number = int(float(str(number)))
    except Exception:
        return None
    if number == 0:
        return 'RED,VIOLET'
    if number == 5:
        return 'GREEN,VIOLET'
    return 'GREEN' if number % 2 else 'RED'


def build_history_stats(history):
    wins = sum(1 for h in history if str(h.get('status', '')).upper() == 'WIN')
    losses = sum(1 for h in history if str(h.get('status', '')).upper() == 'LOSS')
    pending = sum(1 for h in history if str(h.get('status', '')).upper() == 'PENDING')
    settled = wins + losses
    win_rate = round((wins / settled) * 100, 2) if settled else 0

    streak_status = None
    streak_count = 0
    for item in history:
        status = str(item.get('status', '')).upper()
        if status not in ('WIN', 'LOSS'):
            continue
        if streak_status is None:
            streak_status = status
            streak_count = 1
        elif status == streak_status:
            streak_count += 1
        else:
            break

    return OrderedDict([
        ('totalWins', wins),
        ('totalLosses', losses),
        ('winRate', win_rate),
        ('streak', f"{streak_count} {streak_status or 'None'}"),
        ('pending', pending),
        ('totalPredictions', len(history)),
    ])


def build_autolearn_info(al_acc, history):
    total = al_acc.get('totalPredictions', 0) if al_acc else 0
    if total:
        return OrderedDict([
            ('overallAccuracy', al_acc.get('overallAccuracy', 0)),
            ('totalPredictions', total),
            ('totalCorrect', al_acc.get('totalCorrect', 0)),
            ('perPattern', OrderedDict(
                sorted(al_acc.get('perPattern', {}).items(), key=lambda x: x[1].get('total', 0), reverse=True)
            ) if al_acc.get('perPattern') else {}),
        ])

    verified = [
        h for h in history
        if str(h.get('status', '')).upper() in ('WIN', 'LOSS')
        and h.get('prediction')
        and h.get('actual') in ('BIG', 'SMALL')
    ]
    real_total = len(verified)
    real_correct = sum(
        1 for h in verified
        if str(h.get('prediction')).upper() == str(h.get('actual')).upper()
    )
    real_accuracy = round((real_correct / real_total) * 100, 1) if real_total else 0
    return OrderedDict([
        ('overallAccuracy', real_accuracy),
        ('totalPredictions', real_total),
        ('totalCorrect', real_correct),
        ('perPattern', OrderedDict([
            ('ensemble', {
                'accuracy': real_accuracy,
                'total': real_total,
                'correct': real_correct,
            })
        ]) if real_total else {}),
    ])


def period_sort_key(period):
    try:
        return int(str(period))
    except Exception:
        return 0


def format_history_entry(entry):
    number = entry.get('number')
    color = get_number_color(number)
    return {
        'period': entry.get('period', ''),
        'prediction': entry.get('prediction'),
        'status': entry.get('status', 'Pending'),
        'actual': entry.get('actual'),
        'number': number,
        'actualNumber': number,
        'color': color,
        'actualColor': color,
        'confidence': entry.get('confidence', 0),
        'patternUsed': entry.get('patternUsed', 'ensemble'),
        'timestamp': entry.get('timestamp', 0),
    }


def save_leet(data):
    tmp = LEETS_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, LEETS_FILE)


def build_clean_pending(pending_predictions):
    return [
        {
            'period': p.get('period', ''),
            'prediction': p.get('prediction'),
            'status': p.get('status', 'Pending'),
            'confidence': round(p.get('confidence', 0), 2) if isinstance(p.get('confidence'), (int, float)) else p.get('confidence', 0),
            'actual': p.get('actual'),
            'number': p.get('number'),
            'patternUsed': p.get('patternUsed', 'ensemble'),
            'skipped': p.get('skipped', False),
            'skipReason': p.get('skipReason'),
        }
        for p in pending_predictions[:MAX_PENDING]
    ]


def sync_learning_from_history(all_predictions):
    history_entries = load_prediction_history_entries(limit=None)
    if not history_entries:
        return all_predictions, 0

    by_period = {
        str(entry.get('period', '')): entry
        for entry in all_predictions
        if entry.get('period')
    }
    learned_rows = 0
    for entry in history_entries:
        period = str(entry.get('period', ''))
        if not period:
            continue
        if entry.get('status') in ('WIN', 'LOSS') and entry.get('actual') in ('BIG', 'SMALL'):
            learned_rows += 1
        existing = by_period.get(period, {})
        merged = {**existing, **entry}
        by_period[period] = merged

    merged_predictions = sorted(
        by_period.values(),
        key=lambda e: period_sort_key(e.get('period', '')),
        reverse=True,
    )[:MAX_PREDICTIONS_CSV]

    try:
        memory = load_memory()
        update_from_verification(merged_predictions, memory)
    except Exception as e:
        print(f"  [Error] autolearn history sync: {e}")

    try:
        train_model(history_entries)
    except Exception as e:
        print(f"  [Error] ML history sync: {e}")

    return merged_predictions, learned_rows


def cycle():
    migrate_json_to_csv()

    try:
        import_all_to_history_csv()
    except Exception as e:
        print(f"  [Error] import_all_to_history_csv: {e}")

    current_period = get_current_period_1min()
    print(f"\n[{time.strftime('%H:%M:%S')}] New period: {current_period}")

    all_predictions = load_predictions_csv()
    state_data = load_all_states()
    user_states = state_data['userStates']
    pending_predictions = state_data['pendingPredictions']

    # Recover any Pending entries not in pending_predictions
    pending_periods = {str(e.get('period', '')) for e in pending_predictions}
    for source in (all_predictions, load_prediction_history_entries(limit=500)):
        for ap in source:
            p = str(ap.get('period', ''))
            if (p and p not in pending_periods
                    and ap.get('status') == 'Pending'
                    and p < current_period
                    and not ap.get('actual')
                    and ap.get('prediction') in ('BIG', 'SMALL')):
                pending_predictions.append({
                    'period': p,
                    'prediction': ap.get('prediction'),
                    'status': 'Pending',
                    'confidence': ap.get('confidence', 0),
                    'actual': None,
                    'number': None,
                    'patternUsed': ap.get('patternUsed', 'ensemble'),
                    'timestamp': ap.get('timestamp', int(time.time())),
                    'skipped': ap.get('skipped', False),
                    'skipReason': ap.get('skipReason', ''),
                    'patternPredictions': None,
                })
                pending_periods.add(p)

    try:
        import_game_history(all_predictions)
        auto_import_wingobot_history(all_predictions, user_states)
    except Exception:
        pass

    try:
        pending_predictions = verify_pending_predictions(pending_predictions, all_predictions, user_states)
    except Exception as e:
        print(f"  [Error] verify_pending_predictions: {e}")

    try:
        brain_learn()
    except Exception as e:
        print(f"  [Error] brain_learn: {e}")

    try:
        for e in pending_predictions:
            upsert_prediction_history_csv(e)
    except Exception as ex:
        print(f"  [Error] live history sync: {ex}")

    try:
        all_predictions, _ = sync_learning_from_history(all_predictions)
    except Exception as e:
        print(f"  [Error] pre-prediction history learning sync: {e}")

    verified = [e for e in pending_predictions if e.get('status') in ('WIN', 'LOSS', 'SKIP', 'ERROR')]
    if not verified:
        needs_v = any(
            e['period'] < current_period and e.get('status') == 'Pending'
            for e in pending_predictions
        )
        if needs_v:
            print(
                f"  [Info] {sum(1 for e in pending_predictions if e['period'] < current_period and e.get('status') == 'Pending')}"
                " old entries still Pending (API might be unreachable)"
            )

    pr = None
    already_has = any(e['period'] == current_period for e in pending_predictions)
    if not already_has:
        try:
            pr = generate_prediction(user_states, pending_predictions, all_predictions)
        except Exception as e:
            print(f"  [Error] generate_prediction: {e}")

    cp_entry = None
    for e in pending_predictions:
        if e['period'] == current_period:
            cp_entry = e
            break

    if not cp_entry and pr:
        cp_entry = {
            'period': pr['period'],
            'prediction': pr.get('prediction'),
            'status': 'SKIP' if pr.get('skipped') else 'Pending',
            'confidence': pr.get('confidence', 0),
            'actual': None,
            'number': None,
            'patternUsed': 'ensemble',
            'timestamp': int(time.time()),
            'skipped': 1 if pr.get('skipped') else 0,
            'skipReason': pr.get('skipReason', ''),
        }

    if not cp_entry:
        cp_entry = {
            'period': current_period,
            'prediction': None,
            'status': 'Pending',
            'confidence': 0,
            'actual': None,
            'number': None,
            'patternUsed': 'ensemble',
            'timestamp': int(time.time()),
            'skipped': 0,
            'skipReason': 'fallback',
        }

    try:
        upsert_prediction_history_csv(cp_entry)
    except Exception as e:
        print(f"  [Error] history csv: {e}")

    try:
        all_predictions, learned_rows = sync_learning_from_history(all_predictions)
    except Exception as e:
        print(f"  [Error] history learning sync: {e}")

    try:
        ad = analyze_prediction_accuracy(all_predictions)
        fs = get_stats(pending_predictions, all_predictions, user_states)
        ml_acc_data = get_ml_accuracy(all_predictions)
        verified_pp = [e for e in pending_predictions if e.get('status') in ('WIN', 'LOSS')]
        pp_tw = sum(1 for e in verified_pp if e['status'] == 'WIN')
        pp_tl = sum(1 for e in verified_pp if e['status'] == 'LOSS')
        pp_tot = pp_tw + pp_tl
        if pp_tot > 0:
            fs['totalWins'] = pp_tw
            fs['totalLosses'] = pp_tl
            fs['winRate'] = round((pp_tw / pp_tot) * 100, 2)
            streak = 0
            st = None
            for e in pending_predictions:
                if e.get('status') in ('WIN', 'LOSS'):
                    if st is None:
                        st = e['status']
                        streak = 1
                    elif st == e['status']:
                        streak += 1
                    else:
                        break
            fs['streak'] = f"{streak} {st or 'None'}"
            fs['summary'] = f"Win rate: {fs['winRate']}%. Streak: {fs['streak']}."
    except Exception as e:
        print(f"  [Error] stats analysis: {e}")
        ad = {'accuracy': 0, 'recentAccuracy': 0, 'totalPredictions': 0, 'pending': 0}
        fs = {'totalWins': 0, 'totalLosses': 0, 'winRate': 0, 'streak': '0 None', 'lastTen': [], 'summary': ''}
        ml_acc_data = {'accuracy': 0, 'recentAccuracy': 0, 'total': 0}

    try:
        training_source = load_prediction_history_entries(limit=None) or all_predictions
        if train_model(training_source):
            sm = get_model_summary()
            print(
                f"  [ML] Model: {sm['totalTrainCycles']} cycles,"
                f" acc={sm['lastAccuracy']}%,"
                f" recent={sm['lastRecentAccuracy']}%,"
                f" samples={sm['totalSamples']}"
            )
    except Exception as e:
        print(f"  [Error] auto train: {e}")

    history_by_period = {}
    for entry in load_predictions_csv(limit=PUBLIC_HISTORY_LIMIT):
        if str(entry.get('patternUsed', '')).lower() == 'imported':
            continue
        period = entry.get('period', '')
        if period:
            history_by_period[period] = entry
    for entry in pending_predictions:
        if str(entry.get('patternUsed', '')).lower() == 'imported':
            continue
        period = entry.get('period', '')
        if period:
            history_by_period[period] = {**history_by_period.get(period, {}), **entry}
    if cp_entry and cp_entry.get('period'):
        period = cp_entry['period']
        history_by_period[period] = {**history_by_period.get(period, {}), **cp_entry}

    history_entries = sorted(
        history_by_period.values(),
        key=lambda e: period_sort_key(e.get('period', '')),
        reverse=True,
    )
    history = [format_history_entry(entry) for entry in history_entries[:PUBLIC_HISTORY_LIMIT]]
    real_stats = build_history_stats(history)

    try:
        ms = get_model_summary()
        ml_samples = ms['totalSamples']
        ml_accuracy = ms['lastAccuracy'] if ml_samples else 0
        ml_recent_accuracy = ms['lastRecentAccuracy'] if ml_samples else 0
        if ml_samples >= 100 and ms['totalTrainCycles'] > 0:
            ml_strength = 'strong'
        elif ml_samples >= 40 and ms['totalTrainCycles'] > 0:
            ml_strength = 'learning'
        elif ml_samples >= 10:
            ml_strength = 'warming'
        else:
            ml_strength = 'not_ready'
        ml_model_info = OrderedDict([
            ('trained', ms['totalTrainCycles'] > 0 and ml_samples > 0),
            ('learning', ml_samples > 0),
            ('strength', ml_strength),
            ('trainCycles', ms['totalTrainCycles']),
            ('lastTrainTime', ms['lastTrainTime']),
            ('accuracy', ml_accuracy),
            ('recentAccuracy', ml_recent_accuracy),
            ('totalSamples', ml_samples),
            ('dataCount', len(all_predictions)),
            ('modelVersion', ms.get('modelVersion', 1)),
            ('models', ms.get('models', [])),
            ('lightgbmAvailable', ms.get('lightgbmAvailable', False)),
            ('xgboostAvailable', ms.get('xgboostAvailable', False)),
            ('validationAccuracy', ms.get('validationAccuracy', {})),
            ('validationSamples', ms.get('validationSamples', 0)),
            ('message', 'Model learning from verified history.' if ml_samples else 'Model has no verified training samples yet.'),
        ])
    except Exception:
        ml_model_info = None

    try:
        al_mem = load_memory()
        al_acc = get_autolearn_accuracy(al_mem)
        autolearn_info = build_autolearn_info(al_acc, history)
    except Exception:
        autolearn_info = None

    try:
        ps = user_states['user'].get('patternStatsAdvanced', {})
        sorted_pats = sorted(ps.items(), key=lambda x: x[1].get('total', 0), reverse=True)[:10]
        pattern_perf = []
        for pat, st in sorted_pats:
            tot = st.get('total', 0)
            if tot >= 3:
                pattern_perf.append(OrderedDict([
                    ('name', pat),
                    ('wins', st.get('wins', 0)),
                    ('total', tot),
                    ('successRate', st.get('successRate', 0)),
                    ('recentWins', st.get('recentWins', 0)),
                    ('recentTotal', st.get('recentTotal', 0)),
                    ('consecutiveLosses', st.get('consecutiveLosses', 0)),
                ]))
    except Exception:
        pattern_perf = []

    output = {
        'lastUpdated': int(time.time()),
        'currentPeriod': current_period,
        'stats': OrderedDict([
            ('totalWins', real_stats['totalWins']),
            ('totalLosses', real_stats['totalLosses']),
            ('winRate', real_stats['winRate']),
            ('accuracy', ad.get('accuracy', 0)),
            ('recentAccuracy', ad.get('recentAccuracy', 0)),
            ('streak', real_stats['streak']),
            ('pending', real_stats['pending']),
            ('lastTen', fs.get('lastTen', [])),
            ('summary', fs.get('summary', '')),
            ('mlAccuracy', ml_acc_data.get('accuracy') if ml_acc_data.get('total', 0) else None),
            ('mlRecentAccuracy', ml_acc_data.get('recentAccuracy') if ml_acc_data.get('total', 0) else None),
            ('mlSamples', ml_acc_data.get('total', 0)),
            ('totalPredictions', real_stats['totalPredictions']),
        ]),
        'mlModel': ml_model_info,
        'autolearn': autolearn_info,
        'patternPerformance': pattern_perf,
        'history': history,
    }

    try:
        save_leet(output)
        pred_str = cp_entry.get('prediction', 'None') if cp_entry else 'None'
        conf_str = cp_entry.get('confidence', 0) if cp_entry else 0
        status_str = cp_entry.get('status', 'Pending') if cp_entry else 'Pending'
        det = cp_entry.get('period', '') if cp_entry else ''
        st = output.get('stats', {})
        pw = st.get('totalWins', 0)
        pl = st.get('totalLosses', 0)
        wr = st.get('winRate', 0)
        sk = st.get('streak', '0 None')
        al = output.get('autolearn', {})
        al_acc = al.get('overallAccuracy', 0) if al else 0
        pp_len = len(output.get('patternPerformance', []))
        hl = len(output.get('history', []))
        is_new = "NEW" if det != current_period else "SAME"
        print(f"  [{is_new}] {pred_str} @ {conf_str}% ({status_str}) | W:{pw} L:{pl} ({wr}%) | Streak:{sk} | ML:{st.get('mlAccuracy', '?')}% AL:{al_acc}% | Hist:{hl} PP:{pp_len}")
    except Exception as e:
        print(f"  [Error] saving leets.json: {e}")


def main_loop():
    print("[System] Wingo 1 Min Auto — starting 2s cycle")
    print()
    while True:
        try:
            cycle()
        except KeyboardInterrupt:
            print("\nShutdown")
            break
        except Exception as e:
            print(f"  Error: {e}")
            traceback.print_exc()
        time.sleep(2)


if __name__ == '__main__':
    main_loop()
