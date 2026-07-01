import os
import shutil
import time

PROJECT_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
_configured_data_dir = (
    os.getenv('PYAPI_DATA_DIR')
    or os.getenv('RAILWAY_VOLUME_MOUNT_PATH')
    or PROJECT_DATA_DIR
)
DATA_DIR = os.path.abspath(os.path.expanduser(_configured_data_dir))
PERSISTENT_DATA_ENABLED = os.path.normcase(DATA_DIR) != os.path.normcase(os.path.abspath(PROJECT_DATA_DIR))


def _seed_persistent_data():
    """Copy bundled seed files once; never overwrite files already on the volume."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not PERSISTENT_DATA_ENABLED or not os.path.isdir(PROJECT_DATA_DIR):
        return
    for root, _, files in os.walk(PROJECT_DATA_DIR):
        relative = os.path.relpath(root, PROJECT_DATA_DIR)
        target_root = DATA_DIR if relative == '.' else os.path.join(DATA_DIR, relative)
        os.makedirs(target_root, exist_ok=True)
        for filename in files:
            if filename.endswith(('.tmp', '.lock')):
                continue
            source = os.path.join(root, filename)
            target = os.path.join(target_root, filename)
            if not os.path.exists(target):
                try:
                    shutil.copy2(source, target)
                except FileExistsError:
                    pass


_seed_persistent_data()
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

VERIFY_API_URL = 'https://api.nexapk.in/wingo1min.php'
TREND_STATS_API_URL = 'https://api.ar-lottery01.com/api/Lottery/GetTrendStatistics'

LOTTERY01_30S_URL = 'https://api.nexapk.in/wingo30s.php'
LOTTERY01_1M_URL = 'https://api.nexapk.in/wingo1min.php'

MAX_PREDICTIONS_CSV = 5000
MAX_STATS_CSV = 5000
MAX_PENDING = 50

API_ACCESS_KEY = 'enzo'

WINGOBOT_TOKEN = os.getenv('WINGOBOT_TOKEN')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOCK_DIR, exist_ok=True)


def get_storage_status():
    return {
        'dataDir': DATA_DIR,
        'persistent': PERSISTENT_DATA_ENABLED,
        'provider': 'railway-volume' if os.getenv('RAILWAY_VOLUME_MOUNT_PATH') else (
            'custom' if os.getenv('PYAPI_DATA_DIR') else 'local-ephemeral'
        ),
    }
