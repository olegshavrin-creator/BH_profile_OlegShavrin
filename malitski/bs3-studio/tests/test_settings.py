"""bs3/settings.py: every runtime setting is defined once, with the values 3.1 used before the module existed;
importing it loads nothing else; apply_process_env keeps what the caller set; the modules take their defaults from it
and the package does not repeat the values anywhere else."""
from __future__ import annotations

import ast
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

from bs3 import cli, frame_captions, frame_phrase, journal, longvideo, pipeline, settings, translate

ROOT = Path(__file__).resolve().parents[1]            # bs3-studio/
HOME = Path.home()
ENV_NAMES = ("BS3_DATA_DIR", "BS3_JOURNAL", "BS3_OLLAMA_URL", "BS3_KEEP_FAILED_JOBS", "BS3_KEEP_SEGMENTS")

# the literals of 3.1 before bs3/settings.py (the table of the refactoring plan, stage 7)
DEFAULTS = {
    "DATA_DIR": str(HOME / "bs3_data"),
    "JOBS_DIR": str(HOME / "bs3_data" / "web_jobs"),
    "LOGS_DIR": str(HOME / "bs3_data" / "logs"),
    "JOURNAL_PATH": str(HOME / "bs3_data" / "logs" / "journal.txt"),
    "GRADIO_TMP": str(HOME / "bs3_data" / "gradio_tmp"),
    "MM_CHECKPOINTS": str(HOME / "bs" / "mm_runs_seeds" / "seed*" / "best.pt"),
    "OCEANAI_MODELS_DIR": str(HOME / "bs" / "models"),
    "ASR_MODEL": "openai/whisper-large-v3-turbo",
    "OLLAMA_MODEL": "qwen2.5vl:7b",
    "OLLAMA_URL": "",
    "OLLAMA_PORT": 11434,
    "OLLAMA_KEEP_ALIVE": "30m",
    "OLLAMA_DESCRIBE_TIMEOUT": 900,
    "OLLAMA_DESCRIBE_ATTEMPTS": 3,
    "OLLAMA_TRANSLATE_TIMEOUT": 300,
    "OLLAMA_DICTIONARY_TIMEOUT": 180,
    "OLLAMA_PROBE_TTL": 60,
    "OLLAMA_PROBE_TIMEOUT": 3,
    "PHRASE_TIMEOUT": 45,
    "PHRASE_BUDGET": 30,
    "LLM_PARALLEL": 2,
    "PORT": 7880,
    "PREVIEW_PORT": 7882,
    "HOST": "0.0.0.0",
    "QUEUE_CONCURRENCY": 1,
    "NO_PROXY": "localhost,127.0.0.1,0.0.0.0",
    "SEGMENT_SEC": 20.0,
    "SINGLE_CLIP_MAX_SEC": 30.0,
    "MIN_TAIL_SEC": 6.0,
    "KEEP_FAILED_JOBS": False,             # BS3_KEEP_FAILED_JOBS=1 keeps a failed run's folder (stage 22)
    "KEEP_SEGMENTS": False,                # BS3_KEEP_SEGMENTS=1 keeps a finished job's segment clips (stage 22a)
}

# no other module of bs3/ may write these: a default lives in settings.py only
FORBIDDEN = ("qwen2.5vl:7b", "whisper-large-v3-turbo", "mm_runs_seeds", "11434", "7880")
FORBIDDEN_EXACT = ("30m",)
FORBIDDEN_INTS = (11434, 7880)
# (file, literal): not a setting. The audio_whisper feature encoder of AMLAI 1.0 is the model its checkpoints were
# trained with; it must not follow the Whisper of the transcript (settings.ASR_MODEL)
PINNED = {("bs3/mm/extractors_audio.py", "whisper-large-v3-turbo")}


def _run(code: str, **env) -> str:
    clean = {k: v for k, v in os.environ.items() if k not in ENV_NAMES}
    clean.update(env)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=clean, cwd=str(ROOT),
                       timeout=60)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip().splitlines()[-1]


def _values(**env) -> dict:
    code = ("import json, bs3.settings as s; "
            "print(json.dumps({k: v if isinstance(v, (int, float)) else str(v) for k, v in vars(s).items() "
            "if k.isupper()}))")
    return json.loads(_run(code, **env))


def test_defaults_are_the_values_of_3_1():
    got = _values()
    assert got == DEFAULTS, {k: (got.get(k), DEFAULTS.get(k)) for k in set(got) | set(DEFAULTS)
                             if got.get(k) != DEFAULTS.get(k)}
    assert "POOL_DIR" not in got                          # the pool was removed (stage P)
    assert isinstance(settings.MM_CHECKPOINTS, str) and isinstance(settings.OCEANAI_MODELS_DIR, str)
    assert all(isinstance(getattr(settings, k), Path) for k in ("DATA_DIR", "JOBS_DIR", "LOGS_DIR", "JOURNAL_PATH",
                                                                 "GRADIO_TMP"))


def test_environment_variables():
    got = _values(BS3_DATA_DIR="/srv/bs3", BS3_OLLAMA_URL=" http://10.0.0.5:11500/ ")
    assert got["DATA_DIR"] == "/srv/bs3" and got["JOBS_DIR"] == "/srv/bs3/web_jobs"
    assert got["LOGS_DIR"] == "/srv/bs3/logs" and got["JOURNAL_PATH"] == "/srv/bs3/logs/journal.txt"
    assert got["GRADIO_TMP"] == "/srv/bs3/gradio_tmp" and got["OLLAMA_URL"] == "http://10.0.0.5:11500"
    got = _values(BS3_JOURNAL="/srv/j/journal.txt")
    assert got["JOURNAL_PATH"] == "/srv/j/journal.txt" and got["JOBS_DIR"] == DEFAULTS["JOBS_DIR"]
    got = _values(BS3_DATA_DIR="", BS3_JOURNAL="", BS3_OLLAMA_URL="")    # empty = not set
    assert got == DEFAULTS


def test_every_setting_is_documented():
    """docs/config.md (stage 27) names every public upper-case setting of bs3.settings and every environment
    variable, so a new setting is not shipped undocumented."""
    doc = (ROOT / "docs" / "config.md").read_text(encoding="utf-8")
    missing = [k for k in vars(settings) if k.isupper() and k not in doc]
    assert not missing, f"settings missing from docs/config.md: {missing}"
    missing_env = [e for e in ENV_NAMES if e not in doc]
    assert not missing_env, f"env vars missing from docs/config.md: {missing_env}"


def test_import_loads_nothing_else():
    code = ("import json, sys; before = set(sys.modules); import bs3.settings; "
            "print(json.dumps(sorted(set(sys.modules) - before)))")
    new = json.loads(_run(code))
    other = [m for m in new if m not in ("bs3", "bs3.settings") and m.split(".")[0] not in sys.stdlib_module_names]
    assert not other, other
    assert not [m for m in new if m.startswith("bs3.") and m != "bs3.settings"], new
    tree = ast.parse((ROOT / "bs3" / "settings.py").read_text(encoding="utf-8"))
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert imported <= {"__future__", "os", "pathlib"}, imported


def test_apply_process_env_keeps_what_is_set():
    names = ("GRADIO_TEMP_DIR", "GRADIO_ANALYTICS_ENABLED", "no_proxy", "NO_PROXY")
    saved = {n: os.environ.get(n) for n in names}
    try:
        for n in names:
            os.environ[n] = f"mine {n}"
        settings.apply_process_env()
        assert {n: os.environ[n] for n in names} == {n: f"mine {n}" for n in names}
        for n in names:
            os.environ.pop(n, None)
        settings.apply_process_env()
        assert os.environ["GRADIO_TEMP_DIR"] == str(settings.GRADIO_TMP)
        assert os.environ["GRADIO_ANALYTICS_ENABLED"] == "False"
        assert os.environ["no_proxy"] == os.environ["NO_PROXY"] == "localhost,127.0.0.1,0.0.0.0"
    finally:
        for n, v in saved.items():
            if v is None:
                os.environ.pop(n, None)
            else:
                os.environ[n] = v


def _docstrings(tree: ast.AST) -> set[int]:
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                    and isinstance(first.value.value, str):
                out.add(id(first.value))
    return out


def test_values_are_written_only_in_settings():
    """The literals of the defaults appear in bs3/ only in settings.py: in code, that is (strings, numbers, help
    texts); docstrings and comments may name them."""
    bad = []
    for path in sorted((ROOT / "bs3").rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if rel == "bs3/settings.py" or "__pycache__" in rel:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docs = _docstrings(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or id(node) in docs:
                continue
            v = node.value
            if isinstance(v, str):
                hits = [s for s in FORBIDDEN if s in v] + [s for s in FORBIDDEN_EXACT if v == s]
            elif isinstance(v, int) and not isinstance(v, bool):
                hits = [str(v)] if v in FORBIDDEN_INTS else []
            else:
                hits = []
            bad += [f"{rel}:{node.lineno}: {h}" for h in hits if (rel, h) not in PINNED]
    assert not bad, "defaults written outside bs3/settings.py:\n" + "\n".join(bad)


def test_segment_lengths_match_the_texts():
    texts = ("pdf/appendix.py and pdf/sections.py: «по ~20 с» (once each); pdf/build.py: «Ролик короче 30 с оценивается "
             "целиком»; web.app: «до ~20 с» (the stop button); caveats C8 and C19")
    assert settings.SEGMENT_SEC == 20, f"SEGMENT_SEC changed: reword {texts}"
    assert settings.SINGLE_CLIP_MAX_SEC == 30, f"SINGLE_CLIP_MAX_SEC changed: reword {texts}"


def _defaults(fn) -> dict:
    return {k: p.default for k, p in inspect.signature(fn).parameters.items() if p.default is not inspect.Parameter.empty}


def _ast_defaults(rel: str, owner: str, member: str | None = None) -> dict:
    """{name: default expression} of a dataclass (`owner`) or of a method (`owner`.`member`), read from the source
    (the backends import torch, OpenCV and the model code)."""
    tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
    if member is None:
        return {n.target.id: ast.unparse(n.value) for n in cls.body if isinstance(n, ast.AnnAssign) and n.value}
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == member)
    args = fn.args.args[len(fn.args.args) - len(fn.args.defaults):]
    return {a.arg: ast.unparse(d) for a, d in zip(args, fn.args.defaults)}


def _def_source(rel: str, *names: str) -> str:
    """The source of a function (`names` = its name) or of a method (class name, method name), read without importing
    the module."""
    src = (ROOT / rel).read_text(encoding="utf-8")
    node = ast.parse(src)
    for name in names:
        node = next(n for n in node.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
    return ast.get_source_segment(src, node)


def test_modules_take_their_defaults_from_settings():
    web = cli.parse_args(["web"])
    assert (web.port, web.host, web.asr_model, web.ollama_model) == (settings.PORT, settings.HOST, settings.ASR_MODEL,
                                                                     settings.OLLAMA_MODEL)
    assert cli.parse_args(["infer", "v.mp4"]).segment == settings.SEGMENT_SEC
    assert journal.PATH == settings.JOURNAL_PATH
    assert _defaults(pipeline.Studio) == {"asr_model": settings.ASR_MODEL, "ollama_model": settings.OLLAMA_MODEL,
                                          "mm_ckpt": None, "models_dir": None}
    assert _defaults(longvideo.plan_segments) == {"seg_len": settings.SEGMENT_SEC, "min_seg": settings.MIN_TAIL_SEC}
    an = _defaults(longvideo.LongVideoAnalyzer)
    assert (an["seg_len"], an["single_max"], an["asr_model"]) == (settings.SEGMENT_SEC, settings.SINGLE_CLIP_MAX_SEC,
                                                                   settings.ASR_MODEL)
    assert _defaults(translate._ollama_json)["timeout"] == settings.OLLAMA_TRANSLATE_TIMEOUT
    # the frame-phrase requests read their time limits from settings, and no module keeps a copy of the two numbers
    assert "timeout=settings.PHRASE_TIMEOUT" in _def_source("bs3/backend_mm.py", "MMBackend", "describe_frame")
    assert _def_source("bs3/mm/explain.py", "key_frame_info").count("settings.PHRASE_BUDGET") == 2
    assert not [(m.__name__, n) for m in (frame_captions, frame_phrase) for n in ("PHRASE_TIMEOUT", "PHRASE_BUDGET")
                if hasattr(m, n)]
    mm = _ast_defaults("bs3/backend_mm.py", "MMConfig")
    assert (mm["checkpoint"], mm["asr_model"], mm["ollama_model"]) == (
        "settings.MM_CHECKPOINTS", "settings.ASR_MODEL", "settings.OLLAMA_MODEL"), mm
    call = _ast_defaults("bs3/backend_mm.py", "MMBackend", "_ollama")
    assert call == {"timeout": "settings.OLLAMA_DESCRIBE_TIMEOUT", "attempts": "settings.OLLAMA_DESCRIBE_ATTEMPTS"}
    oa = _ast_defaults("bs3/backend_oceanai.py", "BackendConfig")
    assert (oa["models_dir"], oa["asr_model"]) == ("settings.OCEANAI_MODELS_DIR", "settings.ASR_MODEL"), oa
    src = (ROOT / "bs3" / "web" / "app.py").read_text(encoding="utf-8")
    main = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    body = ast.get_source_segment(src, main)
    assert "settings.apply_process_env()" in body and "default_concurrency_limit=settings.QUEUE_CONCURRENCY" in body
    kw = dict(zip([a.arg for a in main.args.args][-len(main.args.defaults):], map(ast.unparse, main.args.defaults)))
    assert (kw["port"], kw["host"], kw["asr_model"], kw["ollama_model"]) == (
        "settings.PORT", "settings.HOST", "settings.ASR_MODEL", "settings.OLLAMA_MODEL"), kw
