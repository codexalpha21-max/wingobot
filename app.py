import subprocess
import sys
import os

base = os.path.dirname(os.path.abspath(__file__))


def load_env_file(path):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_env_file(os.path.join(base, ".env"))

procs = []

scripts = [
    ('server.py', 'FastAPI API + prediction cycle'),
    ('warm.py', 'ping OSS + all API routes every 5s'),
]

for script, desc in scripts:
    try:
        procs.append(subprocess.Popen([sys.executable, os.path.join(base, script)], cwd=base))
        print(f"[RUN] {script} ({desc}) started")
    except Exception as e:
        print(f"[RUN] {script} FAILED: {e}")

print("[RUN] All running. Press Ctrl+C to stop all.")

try:
    for p in procs:
        p.wait()
except KeyboardInterrupt:
    print("\n[RUN] Shutting down...")
    for p in procs:
        try:
            p.terminate()
            p.wait(timeout=5)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
    print("[RUN] All stopped.")
