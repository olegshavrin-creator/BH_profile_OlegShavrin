"""Test runner of BS Profiler 3.1 on the standard library only (pytest is not installed into the shared venv).

Usage:  ~/bs/venv/bin/python bs3-studio/tests/run.py [test_file.py ...] [-k SUBSTRING]
Finds tests/test_*.py, calls every module-level function named test_* in the order of definition, prints one line per
failure and per skip (a test skips by raising unittest.SkipTest, with the reason), the summary
«N passed, M skipped, K failed in Xs» and the three slowest files. Exit code 1 if anything failed, and also if anything
was skipped when BS3_TESTS_STRICT=1 (the checks before a commit: a skipped PDF test checks nothing).
No test loads the translation model or calls Ollama: before the test files are loaded, bs3.translate says Ollama is
unreachable and refuses to load Marian. The files stay pytest-compatible (plain asserts, no fixtures).
"""
from __future__ import annotations

import importlib.util
import inspect
import os
import sys
import time
import traceback
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                   # bs3-studio/: `import bs3` resolves to this working tree
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(f"bs3_tests_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _no_models() -> None:
    """During the tests the translation model is never loaded and the shared Ollama is never called: bs3.translate
    reports Ollama unreachable and raises from _get instead of loading Marian (a caller that catches that error falls
    back to no translation, so it does not always abort the test); a test that needs a translation stubs the call
    itself. bs3.translate calls Ollama's availability through its own name ollama_available; bs3.ollama.available
    answers «no» from its last check, which is dated never to expire (test_ollama sets its own state)."""
    import bs3.ollama
    import bs3.translate

    def _get(*_a, **_k):
        raise RuntimeError("model load in tests")

    bs3.translate.ollama_available = lambda *a, **k: False
    bs3.translate._get = _get
    bs3.ollama._state.update(t=float("inf"), ok=False)


def main(argv: list[str]) -> int:
    key = None
    files: list[Path] = []
    it = iter(argv)
    for a in it:
        if a == "-k":
            key = next(it, None)
        else:
            p = Path(a)
            files.append(p if p.is_absolute() or p.exists() else HERE / p.name)
    if not files:
        files = sorted(HERE.glob("test_*.py"))
    strict = os.environ.get("BS3_TESTS_STRICT") == "1"
    t0 = time.time()
    _no_models()
    passed, failed, skipped, took = 0, [], [], {}
    for f in files:
        t_file = time.time()
        try:
            mod = _load(f)
        except unittest.SkipTest as e:
            skipped.append((f.name, "<import>", str(e)))
            print(f"SKIP {f.name}: {e}")
            continue
        except Exception:
            failed.append((f.name, "<import>", traceback.format_exc()))
            print(f"FAIL {f.name}: import error")
            continue
        tests = [(n, fn) for n, fn in vars(mod).items()
                 if n.startswith("test_") and inspect.isfunction(fn) and fn.__module__ == mod.__name__]
        tests.sort(key=lambda t: t[1].__code__.co_firstlineno)
        for name, fn in tests:
            if key and key not in name:
                continue
            try:
                fn()
                passed += 1
            except unittest.SkipTest as e:
                skipped.append((f.name, name, str(e)))
                print(f"SKIP {f.name}::{name}: {e}")
            except Exception:
                failed.append((f.name, name, traceback.format_exc()))
                print(f"FAIL {f.name}::{name}")
        took[f.name] = time.time() - t_file
    for fname, name, tb in failed:
        print(f"\n===== {fname}::{name}\n{tb}")
    slow = sorted(took.items(), key=lambda kv: -kv[1])[:3]
    if slow:
        print("\nslowest: " + ", ".join(f"{n} {s:.1f}s" for n, s in slow))
    print(f"\n{passed} passed, {len(skipped)} skipped, {len(failed)} failed in {time.time() - t0:.1f}s")
    if skipped and strict:
        print("BS3_TESTS_STRICT=1: a skipped test fails the run")
    return 1 if failed or (skipped and strict) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
