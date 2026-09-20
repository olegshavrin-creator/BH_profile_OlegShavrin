#!/usr/bin/env bash
# Does the running BS 2.0 server serve key-frame JPEGs to the gallery? Usage: check_gallery.sh [PORT]
set -uo pipefail
PORT="${1:-7870}"
f=$(ls -t "$HOME"/bs2_data/web_jobs/*/explain/key_*.jpg 2>/dev/null | head -1)
[ -z "$f" ] && { echo "no key frames found"; exit 1; }
echo "file: $f ($(stat -c%s "$f") bytes)"
for url in "http://localhost:$PORT/gradio_api/file=$f" "http://localhost:$PORT/file=$f"; do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 8 "$url")
  echo "$code  $url"
done
