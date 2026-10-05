"""scripts/import_job.py on a synthetic job in a temporary folder (task T16): the copy has result.json and explain/,
its paths point into the copy, the video and the segments are not copied, the source is unchanged, and a target
outside the 3.0 work dir is refused."""
from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "import_job.py"


def _mod():
    spec = importlib.util.spec_from_file_location("bs3_import_job", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _fake_job(base: Path) -> Path:
    job = base / "old" / "20000101_000000"
    (job / "explain").mkdir(parents=True)
    (job / "segments").mkdir()
    (job / "segments" / "seg001.mp4").write_bytes(b"x" * 10)
    (job / "input.mp4").write_bytes(b"video")
    frame = job / "explain" / "key_01_frame10.jpg"
    frame.write_bytes(b"jpg")
    rep = {"job_dir": str(job), "key_frames": [str(frame)], "input": str(job / "input.mp4"),
           "timeline": [{"segment": 1, "file": str(job / "segments" / "seg001.mp4")}], "traits": {}}
    (job / "result.json").write_text(json.dumps(rep), encoding="utf-8")
    (job / "explain" / "explanation.json").write_text(
        json.dumps({"frames": {"key_frame_files": [str(frame)]}, "input": str(job / "input.mp4")}), encoding="utf-8")
    return job


def test_import_copies_and_rewrites_paths():
    m = _mod()
    with tempfile.TemporaryDirectory() as d:
        base = Path(d).resolve()
        job = _fake_job(base)
        root = base / "bs3"
        before = m.tree_sha256(job)
        assert "segments/seg001.mp4" not in before and "input.mp4" in before
        dest = m.import_job(job, root / "web_jobs", root=root)
        assert dest == root / "web_jobs" / job.name
        assert sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*")) == [
            "explain", "explain/explanation.json", "explain/key_01_frame10.jpg", "result.json"]
        rep = json.loads((dest / "result.json").read_text(encoding="utf-8"))
        assert rep["job_dir"] == str(dest)
        assert rep["key_frames"] == [str(dest / "explain" / "key_01_frame10.jpg")]
        assert Path(rep["key_frames"][0]).is_file()
        assert rep["input"] == str(job / "input.mp4")                     # only its name is read
        expl = json.loads((dest / "explain" / "explanation.json").read_text(encoding="utf-8"))
        assert expl["frames"]["key_frame_files"] == rep["key_frames"]
        assert m.tree_sha256(job) == before
        # a second import needs --force
        try:
            m.import_job(job, root / "web_jobs", root=root)
            raise AssertionError("an existing copy was replaced without force")
        except m.ImportRefused:
            pass
        assert m.import_job(job, root / "web_jobs", root=root, force=True) == dest
        assert not [p for p in (root / "web_jobs").iterdir() if p.name.startswith(".import_")]


def test_import_refuses_targets_outside_the_work_dir():
    m = _mod()
    with tempfile.TemporaryDirectory() as d:
        base = Path(d).resolve()
        job = _fake_job(base)
        root = base / "bs3"
        before = m.tree_sha256(job)
        for target in (base / "elsewhere", root / ".." / "elsewhere", job.parent):
            try:
                m.import_job(job, target, root=root)
                raise AssertionError(f"not refused: {target}")
            except m.ImportRefused:
                pass
        assert not (base / "elsewhere").exists()
        assert m.tree_sha256(job) == before
        assert m.DEFAULT_DEST == m.BS3_ROOT / "web_jobs" and m.BS3_ROOT == Path.home() / "bs3_data"
