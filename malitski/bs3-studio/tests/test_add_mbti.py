"""scripts/add_mbti.py (design 7.2, task T28) on synthetic jobs in a temporary folder: the section is written only for
a job under the 3.0 jobs root, a job elsewhere is refused and its file stays byte for byte the same."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import tempfile
from pathlib import Path

from samples import rep

from bs3 import mbti

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "add_mbti.py"
JOB_ID = "20000101_000000"


def _mod():
    spec = importlib.util.spec_from_file_location("bs3_add_mbti", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _job(base: Path, r: dict) -> Path:
    job = base / JOB_ID
    job.mkdir(parents=True)
    (job / "result.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    return job


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_writes_section_under_root():
    m = _mod()
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "web_jobs"
        job = _job(root, rep("B"))
        assert m.add_mbti(job, root=root) == "written"
        saved = json.loads((job / "result.json").read_text(encoding="utf-8"))
        sec = saved["mbti"]
        assert sec["schema_version"] == 3 and sec["type"] == "ENFJ" and sec["type_strict"] == "ENFJ"
        assert sec["computed_by"] == "BS Profiler 3.1 3.1.0a1" and "computed_on_render" not in sec
        assert sec["model"] == "oceanai" and "second" not in sec and "agreement" not in sec
        assert mbti.get_mbti(saved) == sec                 # the page now shows the stored section as it is
        before = _sha(job / "result.json")
        assert m.add_mbti(job, root=root) == "kept"        # stored already: not recomputed without --force
        assert _sha(job / "result.json") == before
        assert m.add_mbti(job, root=root, force=True) == "written"
        # a section of schema 1 (reference group) or 2 (second opinion, agreement) is replaced without --force
        for old_schema in (1, 2):
            old = {**saved, "mbti": {**sec, "schema_version": old_schema, "type": "EXFJ", "second": [], "agreement": None}}
            (job / "result.json").write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")
            assert mbti.get_mbti(old)["type"] == "ENFJ"         # the old section is not shown
            assert m.add_mbti(job, root=root) == "written"
            assert json.loads((job / "result.json").read_text(encoding="utf-8"))["mbti"]["schema_version"] == 3


def test_refuses_outside_root_and_leaves_file_unchanged():
    m = _mod()
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "web_jobs"
        root.mkdir()
        job = _job(Path(d) / "other", rep("A"))
        before = _sha(job / "result.json")
        for target in (job, root / ".." / "other" / JOB_ID, root):
            try:
                m.add_mbti(target, root=root)
            except m.AddRefused:
                pass
            else:
                raise AssertionError(f"not refused: {target}")
        assert _sha(job / "result.json") == before
        err = io.StringIO()
        with contextlib.redirect_stderr(err):              # main() prints the refusal to stderr
            rc = m.main([str(job)])
        assert rc == 2                                     # the default root is ~/bs3_data/web_jobs: refused too
        assert f"{JOB_ID}: refused: " in err.getvalue() and " is not a job under " in err.getvalue()
        assert _sha(job / "result.json") == before


def test_no_big_five_no_section():
    m = _mod()
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "web_jobs"
        r = rep("B")
        r["variant_scores"] = {}
        for k in r["traits"]:
            r["traits"][k]["score"] = None
        job = _job(root, r)
        before = _sha(job / "result.json")
        assert m.add_mbti(job, root=root) == "no Big Five"
        assert _sha(job / "result.json") == before
