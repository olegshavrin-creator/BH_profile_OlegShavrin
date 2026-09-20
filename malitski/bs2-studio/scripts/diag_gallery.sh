#!/usr/bin/env bash
# Diagnose the key-frame gallery: latest jobs, whether Gradio copied the JPEGs into its cache, server start time.
set -uo pipefail
echo "--- jobs (newest first)"
ls -1t "$HOME/bs2_data/web_jobs" | head -4
echo "--- server process"
for p in $(pgrep -f "bs2 w[e]b"); do ps -o pid=,lstart= -p "$p"; done
echo "--- gradio cache: key frames"
find /tmp/gradio -name "key_*.jpg" -printf "%TT %s %p\n" 2>/dev/null | sort | tail -6
echo "--- GRADIO_TEMP_DIR=${GRADIO_TEMP_DIR:-unset}"
echo "--- web.log errors"
grep -aiE "error|exception|traceback|invalid|not allowed" "$HOME/bs2_data/logs/web.log" | tail -8 | cut -c1-220
