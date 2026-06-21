import os
import time

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
STATE_FILE = os.path.join(DATA_DIR, 'predict', 'state.json')
PREDICTIONS_CSV = os.path.join(DATA_DIR, 'predict', 'predictions.csv')
STATS_CSV = os.path.join(DATA_DIR, 'predict', 'stats.csv')
LOCK_DIR = os.path.join(DATA_DIR, 'locks')

PENDING_PREDICTIONS_CSV = os.path.join(DATA_DIR, 'predict', 'pending_predictions.csv')
PATTERN_STATS_CSV = os.path.join(DATA_DIR, 'predict', 'pattern_stats.csv')
NUMBER_STATS_CSV = os.path.join(DATA_DIR, 'predict', 'number_stats.csv')
NEURAL_STATES_CSV = os.path.join(DATA_DIR, 'predict', 'neural_states.csv')
TRANSITION_MATRIX_CSV = os.path.join(DATA_DIR, 'predict', 'transition_matrix.csv')
ENTROPY_HISTORY_CSV = os.path.join(DATA_DIR, 'predict', 'entropy_history.csv')
LOSS_RECOVERY_CSV = os.path.join(DATA_DIR, 'predict', 'loss_recovery.csv')
PREDICTION_HISTORY_CSV = os.path.join(DATA_DIR, 'predict', 'prediction_history.csv')
DAILY_1K_HISTORY_CSV = os.path.join(DATA_DIR, '1m', 'daily_1k_history.csv')

VERIFY_API_URL = 'https://wingo.oss-ap-southeast-7.aliyuncs.com/WinGo_1_{period}_past100_draws'
TREND_STATS_API_URL = 'https://api.ar-lottery01.com/api/Lottery/GetTrendStatistics'

MAX_PREDICTIONS_CSV = 5000
MAX_STATS_CSV = 5000
MAX_PENDING = 50

API_ACCESS_KEY = 'enzo'

WINGOBOT_TOKEN = os.getenv(
    'WINGOBOT_TOKEN',
    'ws_a7dbbf9b62ea50bcaeedb732188407c1f34175b0110a656371cbdcfbd2ed74fe',
)

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOCK_DIR, exist_ok=True)
