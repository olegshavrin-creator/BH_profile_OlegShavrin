"""scripts/clean_jobs.py (owner decision of 2026-09-27; refactoring stage 22a): a dry run by default lists what it
would free and removes nothing; --apply removes only the segment clips of finished jobs and the folders of runs that
never finished (no result.json, quiet for a day). A fresh unfinished folder survives; input.*, result.json, explain/,
charts/ and the PDF of a finished job are kept; the script stays inside the work dir and never follows a symlink out
of it."""
from __future__ import annotations

import importlib.util
import io
import json
import os
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "clean_jobs.py"


def _mod():
    spec = importlib.util.spec_from_file_location("bs3_clean_jobs", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _finished(job: Path, *, segments: bool = True) -> None:
    """A finished job: result.json, input.*, explain/, charts/ and a PDF, plus (optionally) segments/ with clips."""
    job.mkdir(parents=True)
    (job / "result.json").write_text(json.dumps({"job_dir": str(job)}), encoding="utf-8")
    (job / "input.mp4").write_bytes(b"video-bytes")
    (job / "explain").mkdir()
    (job / "explain" / "key_01_frame10.jpg").write_bytes(b"jpg")
    (job / "explain" / "explanation.json").write_text("{}", encoding="utf-8")
    (job / "charts").mkdir()
    (job / "charts" / "radar.png").write_bytes(b"png-bytes")
    (job / f"{job.name}.pdf").write_bytes(b"%PDF-1.4")
    if segments:
        (job / "segments").mkdir()
        (job / "segments" / "seg01_0-20s.mp4").write_bytes(b"x" * 1000)
        (job / "segments" / "timeline.json").write_text("[]", encoding="utf-8")


def _unfinished(job: Path, age_hours: float) -> None:
    """A folder without result.json whose newest entry is `age_hours` old (a run that never finished)."""
    job.mkdir(parents=True)
    (job / "input.mp4").write_bytes(b"video-bytes")
    (job / "segments").mkdir()
    (job / "segments" / "seg01_0-20s.mp4").write_bytes(b"x" * 500)
    old = time.time() - age_hours * 3600
    for p in [*job.rglob("*"), job]:
        os.utime(p, (old, old))


def _tree(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*")}


def test_inside_guard_never_leaves_the_work_dir():
    m = _mod()
    with tempfile.TemporaryDirectory() as d:
        root = (Path(d) / "web_jobs").resolve()
        root.mkdir()
        assert m._inside(root, root)
        assert m._inside(root / "a" / "b", root)
        assert not m._inside(root.parent, root)
        assert not m._inside(root.parent / "elsewhere", root)


def test_plan_finds_only_segments_and_stale_unfinished():
    m = _mod()
    with tempfile.TemporaryDirectory() as d:
        work = Path(d).resolve()
        _finished(work / "finished_with_segments")
        _finished(work / "finished_no_segments", segments=False)
        _unfinished(work / "stale_unfinished", age_hours=48)
        _unfinished(work / "fresh_unfinished", age_hours=1)
        (work / "loose_file.txt").write_text("not a job", encoding="utf-8")
        paths = {r.path for r in m.plan(work)}
        assert paths == {work / "finished_with_segments" / "segments", work / "stale_unfinished"}


def test_dry_run_removes_nothing():
    m = _mod()
    with tempfile.TemporaryDirectory() as d:
        work = Path(d).resolve()
        _finished(work / "fin")
        _unfinished(work / "stale", age_hours=48)
        _unfinished(work / "fresh", age_hours=1)
        before = _tree(work)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = m.main(["--work-dir", str(work)])
        assert rc == 0
        assert _tree(work) == before                       # a dry run changes nothing on disk
        text = buf.getvalue()
        assert "would remove" in text and "--apply" in text
        assert "fresh" not in text                          # the fresh unfinished folder is not even listed


def test_apply_removes_exactly_segments_and_stale_folders():
    m = _mod()
    with tempfile.TemporaryDirectory() as d:
        work = Path(d).resolve()
        _finished(work / "fin")
        _unfinished(work / "stale", age_hours=48)
        _unfinished(work / "fresh", age_hours=1)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = m.main(["--work-dir", str(work), "--apply"])
        assert rc == 0
        # (a) the finished job keeps everything a job needs, but its segment clips are gone
        fin = work / "fin"
        for kept in ("result.json", "input.mp4", "explain/explanation.json", "explain/key_01_frame10.jpg",
                     "charts/radar.png", "fin.pdf"):
            assert (fin / kept).is_file(), kept
        assert not (fin / "segments").exists()
        # (b) the stale unfinished folder is gone; a fresh unfinished folder survives whole
        assert not (work / "stale").exists()
        assert (work / "fresh" / "input.mp4").is_file() and (work / "fresh" / "segments").is_dir()


def test_a_fresh_unfinished_folder_survives_even_with_apply():
    m = _mod()
    with tempfile.TemporaryDirectory() as d:
        work = Path(d).resolve()
        _unfinished(work / "just_started", age_hours=0.1)   # a run that may still be going
        with redirect_stdout(io.StringIO()):
            assert m.main(["--work-dir", str(work), "--apply"]) == 0
        assert (work / "just_started" / "input.mp4").is_file()


def test_a_symlink_pointing_outside_the_work_dir_is_never_followed():
    m = _mod()
    with tempfile.TemporaryDirectory() as d:
        d = Path(d).resolve()
        work = d / "web_jobs"
        work.mkdir()
        outside = d / "outside"
        outside.mkdir()
        (outside / "precious.txt").write_text("keep me", encoding="utf-8")
        try:
            (work / "linkjob").symlink_to(outside, target_is_directory=True)      # a whole job that is a symlink out
            fin = work / "fin"
            _finished(fin, segments=False)
            (fin / "segments").symlink_to(outside, target_is_directory=True)      # a finished job's segments symlinked out
        except (OSError, NotImplementedError):
            raise unittest.SkipTest("symlinks are not available here")
        assert m.plan(work) == []                            # both are symlinks out: nothing is planned for removal
        with redirect_stdout(io.StringIO()):
            assert m.main(["--work-dir", str(work), "--apply"]) == 0
        assert (outside / "precious.txt").is_file()          # the outside folder and its file are untouched
        assert (work / "linkjob").is_symlink() and (fin / "segments").is_symlink()


def test_a_work_dir_that_is_not_a_folder_is_refused():
    m = _mod()
    with tempfile.TemporaryDirectory() as d:
        missing = Path(d) / "no_such_dir"
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            rc = m.main(["--work-dir", str(missing)])
        assert rc == 2
