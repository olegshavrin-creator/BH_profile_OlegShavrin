"""Serve the BS 2.0 page with an already finished job pre-rendered (for UI checks without the GPU analysis).
Usage: ui_preview.py [JOB_DIR] [PORT]     default: newest job in ~/bs2_data/web_jobs, port 7871
"""
import sys
from pathlib import Path

from bs2.pipeline import Studio
from bs2.webapp import build_app

jobs = Path.home() / "bs2_data" / "web_jobs"
job = sys.argv[1] if len(sys.argv) > 1 else str(sorted(p for p in jobs.iterdir() if (p / "result.json").exists())[-1])
port = int(sys.argv[2]) if len(sys.argv) > 2 else 7871
print("preview job:", job, "port:", port, flush=True)
demo = build_app(Studio(), jobs, preview_job=job)
demo.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=port, show_api=False, show_error=True,
                                                allowed_paths=[str(jobs)])
