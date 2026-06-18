import subprocess
import sys
import os
import time
import signal

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable
SERVER = os.path.join(BASE_DIR, 'server.py')

processes = []


def start_server(server_type, port, workers=1):
    env = os.environ.copy()
    env['SERVER_TYPE'] = server_type
    env['PORT'] = str(port)
    env['WORKERS'] = str(workers)
    p = subprocess.Popen(
        [PYTHON, '-m', 'uvicorn', 'server:app',
         '--host', '0.0.0.0', '--port', str(port),
         '--workers', str(workers)],
        cwd=BASE_DIR, env=env,
    )
    processes.append(p)
    return p


if __name__ == '__main__':
    print("Starting all API servers...")
    print("  [4 CPU / 16 GB] allocating workers...")

    start_server('free', port=5001, workers=1)
    print("  [1/3] Free server on :5001 (1 worker)")

    start_server('predict', port=5002, workers=1)
    print("  [2/3] Predict V2 server on :5002 (1 worker)")

    start_server('main', port=5000, workers=2)
    print("  [3/3] Main server on :5000 (2 workers)")
    print("  Total: 4 workers across 3 servers")

    time.sleep(2)

    try:
        while True:
            alive = sum(1 for p in processes if p.poll() is None)
            if alive < len(processes):
                for i, p in enumerate(processes):
                    if p.poll() is not None:
                        print(f"  [WARN] Server {i+1} exited (code={p.returncode})")
                        # restart
                        start_server(['free', 'predict', 'main'][i],
                                      [5001, 5002, 5000][i],
                                      [1, 1, 2][i])
                        print(f"  [RESTART] Server {i+1} restarted")
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nShutting down...")
        for p in processes:
            p.terminate()
        for p in processes:
            try:
                p.wait(timeout=5)
            except Exception:
                p.kill()
        print("All servers stopped.")
