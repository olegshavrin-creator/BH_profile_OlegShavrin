#!/usr/bin/env bash
# Start the BS 2.0 web UI inside WSL (detached). Own port, log and job folder, independent of bs 1.0 on :7860.
# Usage: run_web2.sh [PORT]      Stop: pkill -f "bs2 web"
set -uo pipefail
PORT="${1:-7870}"
mkdir -p "$HOME/bs2_data/logs"
cd "$HOME"
export PYTHONWARNINGS=ignore
export no_proxy="localhost,127.0.0.1,0.0.0.0,${no_proxy:-}" NO_PROXY="localhost,127.0.0.1,0.0.0.0,${NO_PROXY:-}"
export GRADIO_ANALYTICS_ENABLED=False
for i in $(seq 1 10); do
  pgrep -f "bs2 web" >/dev/null || break
  if curl -s -o /dev/null --max-time 2 "http://localhost:$PORT/"; then echo "already running and answering on :$PORT"; exit 0; fi
  sleep 2
done
pgrep -f "bs2 web" >/dev/null && { echo "old server still shutting down; try again"; exit 1; }
setsid nohup "$HOME/bs/venv/bin/bs2" web --port "$PORT" > "$HOME/bs2_data/logs/web.log" 2>&1 < /dev/null &
for i in $(seq 1 30); do
  sleep 2
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 "http://localhost:$PORT/" || true)
  [ "$code" = "200" ] && break
done
tail -3 "$HOME/bs2_data/logs/web.log" | grep -vE "SyntaxWarning|invalid escape"
echo "http://localhost:$PORT -> HTTP ${code:-000} after $((i*2))s"
