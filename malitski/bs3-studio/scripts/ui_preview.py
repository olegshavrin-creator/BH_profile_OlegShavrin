"""Serve the BS Profiler 3.1 page with an already finished job pre-rendered (for UI checks without the GPU analysis).
Usage: ui_preview.py [JOB_DIR] [PORT]     default: newest job in ~/bs3_data/web_jobs, port 7882
JOB_DIR must lie in ~/bs3_data/web_jobs: the page and the PDF button write into the job folder, and 3.1 never writes
into the 2.0 work dir (design 13.3). A 2.0 job is copied there first with scripts/import_job.py.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # bs3-studio/: `import bs3` is this working tree
from bs3 import settings  # noqa: E402  (standard library only: gradio is not imported yet)


def _inside(path: Path, root: Path) -> bool:
    path, root = path.resolve(), root.resolve()
    return path == root or root in path.parents


jobs = settings.JOBS_DIR
job = sys.argv[1] if len(sys.argv) > 1 else str(sorted(p for p in jobs.iterdir() if (p / "result.json").exists())[-1])
if not _inside(Path(job).expanduser(), jobs):
    sys.exit(f"refused: {job} is outside {jobs}; copy it there first: scripts/import_job.py {job}")
port = int(sys.argv[2]) if len(sys.argv) > 2 else settings.PREVIEW_PORT

# own Gradio temp folder (the shared /tmp/gradio is cleaned by another service); must be set before gradio is imported
settings.apply_process_env()
Path(os.environ["GRADIO_TEMP_DIR"]).mkdir(parents=True, exist_ok=True)

from bs3.pipeline import Studio  # noqa: E402
from bs3.web.app import build_app  # noqa: E402

print("preview job:", job, "port:", port, flush=True)
demo = build_app(Studio(), jobs, preview_job=job)
# no allowed_paths: the page shows no file of the job folder (key frames are data URIs, charts are srcdoc), and the PDF
# is handed to Gradio from its own temp folder (web.app.pdf_for_download)
demo.queue(default_concurrency_limit=settings.QUEUE_CONCURRENCY).launch(server_name=settings.HOST, server_port=port,
                                                                         show_api=False, show_error=True)
