"""The files of a BS Profiler 3.1 job folder in one place: their names, reading and writing.

A job folder <work dir>/<job id>/ holds

  result.json                  RESULT                     the result of the analysis (pipeline.run_analysis)
  explain/explanation.json     EXPLAIN_DIR, EXPLANATION   the explanations of AMLAI 1.0
  explain/key_*.jpg            KEY_FRAME_GLOB             the key frames
  charts/                      CHARTS_DIR                 the chart images of the PDF
  segments/                    SEGMENTS_DIR               the segment clips of a long video
  input.<ext>                  INPUT_GLOB                 the uploaded video

The files are always found from the folder being opened, never from the absolute paths result.json stores (`job_dir`,
`key_frames`): a job folder that was moved to another data root or copied (an imported 2.0 job, the copy of
scripts/compare_baseline.py) opens the same way. Writes are atomic, so a page render and a PDF export that read the
same file at the same time never see half of it."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path, PurePosixPath

RESULT = "result.json"
EXPLAIN_DIR = "explain"
EXPLANATION = "explanation.json"
KEY_FRAME_GLOB = "key_*.jpg"
CHARTS_DIR = "charts"
SEGMENTS_DIR = "segments"
INPUT_GLOB = "input.*"


def explanation_path(job_dir: str | Path) -> Path:
    """<job>/explain/explanation.json (it may not exist: OCEAN-AI builds no explanations)."""
    return Path(job_dir) / EXPLAIN_DIR / EXPLANATION


def read_json(path: str | Path) -> dict | None:
    """The JSON object stored in `path`; None when the file is missing, cannot be read or holds no JSON object."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write_json(path: str | Path, data: dict) -> None:
    """Writes `data` as result.json and explanation.json have always been written (indent 2, UTF-8 without escapes)
    into a temporary file next to `path`, then puts it in place with one rename. Raises on failure and leaves no
    temporary file; an existing file then stays as it was. The temporary name is per process and thread, so two
    writers of the same file never write into one temporary file."""
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def load_job(job_dir: str | Path) -> tuple[dict, dict | None]:
    """(result.json, explanation.json or None) of the job folder `job_dir`. `rep["job_dir"]` is set to that folder,
    whatever the file stores. Raises FileNotFoundError without result.json, ValueError when it holds no JSON object."""
    job = Path(job_dir)
    rep = read_json(job / RESULT)
    if rep is None:
        if not (job / RESULT).is_file():
            raise FileNotFoundError(f"no {RESULT} in {job}")
        raise ValueError(f"{RESULT} in {job} is not a JSON object")
    rep["job_dir"] = str(job)
    return rep, read_json(explanation_path(job))


def key_frame_paths(job_dir: str | Path, rep: dict) -> list[Path]:
    """The key frames of the job in <job>/explain/, in the order result.json lists them (`key_frames`), each found by
    its file name (the stored path may name another folder); a listed frame that is not there is left out. A job whose
    result.json has no list gets the frames on disk in sorted order."""
    ex = Path(job_dir) / EXPLAIN_DIR
    listed = rep.get("key_frames")
    if isinstance(listed, list):
        paths = [ex / PurePosixPath(p).name for p in listed if isinstance(p, str) and PurePosixPath(p).name]
        return [p for p in paths if p.is_file()]
    return sorted(ex.glob(KEY_FRAME_GLOB)) if ex.is_dir() else []


def input_file(job_dir: str | Path) -> Path | None:
    """The uploaded video copied into the job folder (input.<ext>); None when it has none (an imported 2.0 job)."""
    return next(iter(sorted(Path(job_dir).glob(INPUT_GLOB))), None)
