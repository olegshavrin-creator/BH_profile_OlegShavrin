#!/usr/bin/env bash
# Start the BS Profiler 3.1 web UI inside WSL (detached). Own port, log, job folder and Gradio temp folder;
# независим от BS 2.0 на :7870.
# Usage: run_web3.sh [PORT]      Stop: pkill -f "bin/bs3 we[b]"
set -uo pipefail
PORT="${1:-7880}"
mkdir -p "$HOME/bs3_data/logs"
export GRADIO_TEMP_DIR="$HOME/bs3_data/gradio_tmp"
mkdir -p "$GRADIO_TEMP_DIR"
cd "$HOME"
export PYTHONWARNINGS=ignore
export no_proxy="localhost,127.0.0.1,0.0.0.0,${no_proxy:-}" NO_PROXY="localhost,127.0.0.1,0.0.0.0,${NO_PROXY:-}"
export GRADIO_ANALYTICS_ENABLED=False
for i in $(seq 1 10); do
  pgrep -f "bin/bs3 we[b]" >/dev/null || break
  if curl -s -o /dev/null --max-time 2 "http://localhost:$PORT/"; then echo "already running and answering on :$PORT"; exit 0; fi
  sleep 2
done
pgrep -f "bin/bs3 we[b]" >/dev/null && { echo "old server still shutting down; try again"; exit 1; }
# uploads and rendered files of earlier runs: drop the ones older than a day, then the empty PDF-download folders left
# behind (Gradio removes the file it served but not the folder pdf_for_download put it in)
find "$GRADIO_TEMP_DIR" -type f -mtime +0 -delete 2>/dev/null
find "$GRADIO_TEMP_DIR" -mindepth 1 -type d -empty -delete 2>/dev/null
echo "=== $(date '+%Y-%m-%d %H:%M:%S') bs3 web start on :$PORT ===" >> "$HOME/bs3_data/logs/web.log"
setsid nohup "$HOME/bs/venv/bin/bs3" web --port "$PORT" >> "$HOME/bs3_data/logs/web.log" 2>&1 < /dev/null &
for i in $(seq 1 30); do
  sleep 2
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 "http://localhost:$PORT/" || true)
  [ "$code" = "200" ] && break
done
tail -3 "$HOME/bs3_data/logs/web.log" | grep -vE "SyntaxWarning|invalid escape"
echo "http://localhost:$PORT -> HTTP ${code:-000} after $((i*2))s"
