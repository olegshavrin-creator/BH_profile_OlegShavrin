"""tests/run.py itself: a test that raises unittest.SkipTest is counted and reported as skipped with its reason, never
as passed, and BS3_TESTS_STRICT=1 turns a skip into a failed run (the checks before a commit run strict)."""
from __future__ import annotations

import contextlib
import io
import os
import tempfile
from pathlib import Path

import run

PROBE = '''import unittest


def test_passes():
    assert True


def test_skips():
    raise unittest.SkipTest("no tool here")
'''


def _run(path: Path, strict: str | None) -> tuple[int, str]:
    old = os.environ.pop("BS3_TESTS_STRICT", None)
    if strict is not None:
        os.environ["BS3_TESTS_STRICT"] = strict
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            rc = run.main([str(path)])
    finally:
        os.environ.pop("BS3_TESTS_STRICT", None)
        if old is not None:
            os.environ["BS3_TESTS_STRICT"] = old
    return rc, out.getvalue()


def test_skip_is_counted_and_reported():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "test_runner_probe.py"
        path.write_text(PROBE, encoding="utf-8")
        rc, text = _run(path, None)
        assert rc == 0, text
        assert "SKIP test_runner_probe.py::test_skips: no tool here" in text, text
        assert "1 passed, 1 skipped, 0 failed in " in text, text
        rc, text = _run(path, "1")
        assert rc == 1, text
        assert "1 passed, 1 skipped, 0 failed in " in text, text
