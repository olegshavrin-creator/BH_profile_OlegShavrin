"""The offline scripts of BS Profiler 3.1 (scripts/*.py): every one of them compiles, and the ones the checks and the
owner run answer --help from the bs3-studio folder. The refactoring moves names these scripts import; a broken import
shows up here, not at the next manual run."""
from __future__ import annotations

import os
import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]            # bs3-studio/
SCRIPTS = ROOT / "scripts"
HELP = ("import_job", "add_mbti", "rerender_samples", "check_job", "compare_baseline", "clean_jobs")
DEPLOY = ROOT / "deploy"


def test_every_script_compiles():
    files = sorted(SCRIPTS.glob("*.py"))
    assert {f.stem for f in files} >= set(HELP), files
    with tempfile.TemporaryDirectory() as d:          # the .pyc files go there, not into scripts/__pycache__
        for f in files:
            py_compile.compile(str(f), cfile=str(Path(d) / f"{f.stem}.pyc"), doraise=True)


def test_scripts_answer_help():
    env = {**os.environ, "GRADIO_ANALYTICS_ENABLED": "False"}
    for name in HELP:
        r = subprocess.run([sys.executable, str(SCRIPTS / f"{name}.py"), "--help"], cwd=str(ROOT), env=env,
                           capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, (name, r.returncode, r.stdout[-500:], r.stderr[-2000:])
        assert r.stdout.strip(), name


def test_launchers_are_self_contained():
    """The launch scripts and the service template after the 3.1 cleanup (stage 26): run_web3.sh no longer matches its
    own caller, run_preview.sh hard-codes no absolute path, and the unit template names version 3.1."""
    web = (SCRIPTS / "run_web3.sh").read_text(encoding="utf-8")
    assert 'pgrep -f "bs3 web"' not in web, "run_web3.sh must not match its own caller; use bin/bs3 we[b]"
    prev = (SCRIPTS / "run_preview.sh").read_text(encoding="utf-8")
    assert "/mnt/c/" not in prev, "run_preview.sh must not hard-code an absolute /mnt/c/ path"
    unit = (DEPLOY / "bs3-web.service").read_text(encoding="utf-8")
    desc = [ln for ln in unit.splitlines() if ln.startswith("Description=")]
    assert desc and "3.1" in desc[0], desc
