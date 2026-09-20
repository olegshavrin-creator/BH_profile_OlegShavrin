#!/usr/bin/env bash
# Start (or restart) the UI preview server on :7871 with the newest finished job. Stop: pkill -f "ui_previe[w]"
pkill -f "ui_previe[w].py" 2>/dev/null; sleep 2
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$HOME/bs2_data/logs"; cd "$HOME"
export PYTHONWARNINGS=ignore GRADIO_ANALYTICS_ENABLED=False no_proxy="localhost,127.0.0.1,0.0.0.0"
setsid nohup "$HOME/bs/venv/bin/python" "$ROOT/scripts/ui_preview.py" "$@" \
  > "$HOME/bs2_data/logs/preview.log" 2>&1 < /dev/null &
for i in $(seq 1 30); do
  sleep 2
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 "http://localhost:7871/" || true)
  [ "$code" = "200" ] && break
done
grep -a "preview job" "$HOME/bs2_data/logs/preview.log"
echo "http://localhost:7871 -> HTTP ${code:-000}"
