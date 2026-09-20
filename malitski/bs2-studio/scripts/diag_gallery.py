"""Reproduce what Gradio does with the key-frame gallery value: postprocess + move to cache, print the file URLs.
Usage: diag_gallery.py [JOB_DIR]
"""
import json
import sys
from pathlib import Path

import gradio as gr
from gradio import processing_utils

jobs = Path.home() / "bs2_data" / "web_jobs"
job = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(p for p in jobs.iterdir() if (p / "result.json").exists())[-1]
rep = json.loads((job / "result.json").read_text(encoding="utf-8"))
frames = [(p, f"кадр {Path(p).stem.split('_frame')[-1]}") for p in rep.get("key_frames", [])]
print("job:", job, "| key_frames:", len(frames), [Path(p).name for p, _ in frames][:3])
print("explain dir:", sorted(x.name for x in (job / "explain").glob("key_*.jpg")) if (job / "explain").exists() else None)

with gr.Blocks() as demo:
    g = gr.Gallery(columns=5, height=300, object_fit="contain")
demo.allowed_paths = [str(jobs)]
data = g.postprocess(frames)
print("postprocess ->", type(data).__name__, "| first:", data.root[0].model_dump() if data.root else None)
try:
    moved = processing_utils.move_files_to_cache(data, g, postprocess=True)
    first = moved.root[0].image if hasattr(moved, "root") else moved[0]["image"]
    print("moved first:", first.path if hasattr(first, "path") else first)
    print("url:", getattr(first, "url", None))
except Exception as e:  # noqa: BLE001
    print("move_files_to_cache FAILED:", type(e).__name__, e)
