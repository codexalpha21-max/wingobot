#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${APP_DIR:-${SCRIPT_DIR}}"
SERVICE_NAME="${SERVICE_NAME:-cloud-apis}"
WARM_SERVICE_NAME="${WARM_SERVICE_NAME:-cloud-apis-warm}"
ENV_FILE="${ENV_FILE:-/etc/cloud-apis.env}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-2}"

echo "==> Deploying ${SERVICE_NAME} from ${APP_DIR}"
cd "${APP_DIR}"

echo "==> Creating/updating virtualenv"
if [ ! -d "venv" ]; then
  "${PYTHON_BIN}" -m venv venv
fi

source venv/bin/activate
python -m ensurepip --upgrade || true
python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade -r requirements.txt
python -m pip install --upgrade fastapi "uvicorn[standard]" gunicorn orjson
python -m pip check || true
python - <<'PYEOF'
import importlib
for name in ("fastapi", "uvicorn", "gunicorn", "orjson"):
    importlib.import_module(name)
print("Python deps OK: fastapi uvicorn gunicorn orjson")
PYEOF

if [ ! -f "${ENV_FILE}" ]; then
  echo "==> Creating ${ENV_FILE}"
  cat > "${ENV_FILE}" <<ENVEOF
# App settings.
PORT=${PORT}
ENVEOF
else
  echo "==> Keeping existing ${ENV_FILE}"
fi
chmod 600 "${ENV_FILE}"

if ! command -v systemctl >/dev/null 2>&1; then
  echo "==> systemctl not found, running all files locally from ${APP_DIR}"
  if [ -f ".env" ]; then
    set -a
    # shellcheck disable=SC1091
    source ".env"
    set +a
  fi
  if [ "${ENV_FILE}" ]; then
    set -a
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    set +a
  fi

  echo "  server -> port ${PORT} (${WORKERS} workers)"
  echo "  warm   -> pings all routes every 5s"

  # Start server with workers
  if command -v gunicorn &>/dev/null; then
    echo "  using gunicorn + uvicorn workers"
    gunicorn -k uvicorn.workers.UvicornWorker server:app \
      --bind 0.0.0.0:${PORT} --workers ${WORKERS} --timeout 120 \
      --access-logfile - --error-logfile - &
  else
    echo "  using uvicorn directly (workers=${WORKERS})"
    export WORKERS
    python server.py &
  fi
  SRV_PID=$!

  # Start warm.py in background
  python warm.py &
  WARM_PID=$!

  echo "==> Server (PID $SRV_PID) + Warm (PID $WARM_PID) running"
  echo "    Press Ctrl+C to stop both"

  # Trap Ctrl+C to kill both
  cleanup() {
    echo; echo "==> Shutting down..."
    kill $SRV_PID $WARM_PID 2>/dev/null
    wait $SRV_PID $WARM_PID 2>/dev/null
    echo "==> All stopped"
  }
  trap cleanup SIGINT SIGTERM

  wait $SRV_PID
  exit 0
fi

echo "==> Writing systemd service: ${SERVICE_NAME}"
cat > "/etc/systemd/system/${SERVICE_NAME}.service" <<EOF
[Unit]
Description=Cloud APIs FastAPI App
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=${APP_DIR}
EnvironmentFile=-${ENV_FILE}
Environment="PATH=${APP_DIR}/venv/bin"
ExecStart=${APP_DIR}/venv/bin/gunicorn -k uvicorn.workers.UvicornWorker server:app --bind ${HOST}:${PORT} --workers ${WORKERS} --timeout 120 --access-logfile - --error-logfile -
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

echo "==> Writing systemd service: ${WARM_SERVICE_NAME}"
cat > "/etc/systemd/system/${WARM_SERVICE_NAME}.service" <<EOF
[Unit]
Description=Cloud APIs route warmer
After=${SERVICE_NAME}.service network-online.target
Wants=${SERVICE_NAME}.service network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=${APP_DIR}
EnvironmentFile=-${ENV_FILE}
Environment="PATH=${APP_DIR}/venv/bin"
ExecStart=${APP_DIR}/venv/bin/python warm.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

echo "==> Reloading and restarting services"
systemctl daemon-reload
systemctl enable "${SERVICE_NAME}" "${WARM_SERVICE_NAME}"
systemctl restart "${SERVICE_NAME}"
systemctl restart "${WARM_SERVICE_NAME}"

echo "==> Service status"
systemctl --no-pager --full status "${SERVICE_NAME}" || true
systemctl --no-pager --full status "${WARM_SERVICE_NAME}" || true

echo
echo "Done."
echo "After editing ${ENV_FILE}, run: systemctl restart ${SERVICE_NAME} ${WARM_SERVICE_NAME}"
