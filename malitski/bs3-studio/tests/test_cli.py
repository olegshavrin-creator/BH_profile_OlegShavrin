"""The command line of BS Profiler 3.1 (bs3.cli): `web`, `infer` and `explain` with the options that are used (the
FIV2 evaluation is research code in training/, test_training.py). The speech language is not an option: it is
bs3.LANG. Nothing loads a model: `web.app.main` and the backend modules are replaced by fakes, `cli._backend` by a
canned one."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import bs3
from bs3 import cli, media
from bs3.errors import UserFacingError
from bs3.norms import TRAIT_KEYS

ROOT = Path(__file__).resolve().parents[1]            # bs3-studio/


def _exit_code(argv: list[str]) -> int | None:
    """The exit code argparse gives `argv` (None: parsed)."""
    try:
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            cli.parse_args(argv)
    except SystemExit as e:
        return e.code
    return None


@contextlib.contextmanager
def _modules(**mods):
    saved = {name: sys.modules.get(name) for name in mods}
    sys.modules.update(mods)
    try:
        yield
    finally:
        for name, old in saved.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def test_subcommands_are_web_infer_explain():
    ap = cli.build_parser()
    sub = next(a for a in ap._actions if isinstance(a, argparse._SubParsersAction))
    assert set(sub.choices) == {"web", "infer", "explain"}, sorted(sub.choices)
    for gone in ("setup-weights", "infer-dir", "eval-fiv2"):
        assert _exit_code([gone, "x"]) == 2, gone


def test_web_options():
    a = cli.parse_args(["web"])
    assert a.port == 7880 and a.host == "0.0.0.0" and a.work_dir is None and a.share is False
    assert a.asr_model == "openai/whisper-large-v3-turbo" and a.ollama_model == "qwen2.5vl:7b"
    assert a.mm_ckpt is None and a.models_dir is None and a.verbose is False
    assert set(vars(a)) == {"cmd", "fn", "port", "host", "work_dir", "share", "asr_model", "ollama_model", "mm_ckpt",
                            "models_dir", "verbose"}, sorted(vars(a))
    for opt in (["--lang", "ru"], ["--backend", "mm"], ["--corpus", "mupta"], ["--ensemble-members", "mm"],
                ["--primary", "mm"], ["--sslmepr-ckpt", "x"]):
        assert _exit_code(["web", *opt]) == 2, opt


def test_cmd_web_forwards_every_option():
    seen = []
    fake = types.ModuleType("bs3.web.app")
    fake.main = lambda **kw: seen.append(kw)
    with _modules(**{"bs3.web.app": fake}):
        a = cli.parse_args(["web", "--port", "7999", "--host", "127.0.0.1", "--work-dir", "/tmp/jobs", "--share",
                            "--asr-model", "asr-x", "--ollama-model", "qwen-x", "--mm-ckpt", "/tmp/c.pt",
                            "--models-dir", "/tmp/models", "-v"])
        a.fn(a)
        a = cli.parse_args(["web"])
        a.fn(a)
    assert seen[0] == {"port": 7999, "host": "127.0.0.1", "work_dir": "/tmp/jobs", "share": True, "asr_model": "asr-x",
                       "ollama_model": "qwen-x", "mm_ckpt": "/tmp/c.pt", "models_dir": "/tmp/models"}, seen[0]
    assert seen[1] == {"port": 7880, "host": "0.0.0.0", "work_dir": None, "share": False,
                       "asr_model": "openai/whisper-large-v3-turbo", "ollama_model": "qwen2.5vl:7b", "mm_ckpt": None,
                       "models_dir": None}, seen[1]


def test_web_main_hands_models_dir_to_the_studio():
    """web.app.main(models_dir=...) -> Studio(models_dir=...) -> BackendConfig (test_pipeline checks the last step);
    None keeps the default of BackendConfig."""
    import inspect
    from bs3.web import app
    assert inspect.signature(app.main).parameters["models_dir"].default is None
    assert "models_dir=models_dir" in inspect.getsource(app.main)
    assert inspect.signature(app.Studio).parameters["models_dir"].default is None


def test_infer_and_explain_options():
    a = cli.parse_args(["infer", "a.mp4", "b.mp4"])
    assert a.backend == bs3.DEFAULT_MODEL == "mm" and a.video == ["a.mp4", "b.mp4"] and a.segment == 20.0
    assert {"mm_ckpt", "ollama_model", "asr_model", "models_dir", "verbose", "out", "no_asr", "transcript"} <= set(vars(a))
    assert cli.parse_args(["infer", "a.mp4", "--backend", "oceanai"]).backend == "oceanai"
    for opt in (["--backend", "ensemble"], ["--backend", "sslmepr"], ["--lang", "ru"], ["--lang", "en"],
                ["--corpus", "fi"], ["--ensemble-members", "oceanai,mm"], ["--primary", "mm"],
                ["--sslmepr-ckpt", "x"], ["--sslmepr-modalities", "scene"]):
        assert _exit_code(["infer", "a.mp4", *opt]) == 2, opt
    a = cli.parse_args(["explain", "a.mp4", "--out", "d", "--mm-ckpt", "c.pt", "--behavior", "b.txt", "--top-k", "3"])
    assert (a.video, a.out, a.mm_ckpt, a.behavior, a.top_k) == ("a.mp4", "d", "c.pt", "b.txt", 3)
    assert "backend" not in vars(a) and "models_dir" not in vars(a)         # always AMLAI 1.0
    for opt in (["--backend", "oceanai"], ["--lang", "ru"], ["--models-dir", "m"]):
        assert _exit_code(["explain", "a.mp4", "--out", "d", *opt]) == 2, opt


def test_backend_builds_one_model_with_russian_speech():
    """`_backend` builds MMBackend or OceanAIBackend from the options, with lang = bs3.LANG; --models-dir only when
    given. The backend modules are fakes that record their configs."""
    built = []

    def module(name, backend, config):
        m = types.ModuleType(name)

        class Cfg:
            def __init__(self, **kw):
                self.kw = kw

        class Backend:
            def __init__(self, cfg):
                self.cfg, self.kind = cfg, backend

            def load(self):
                built.append(self)
                return self
        setattr(m, backend, Backend)
        setattr(m, config, Cfg)
        return m

    with _modules(**{"bs3.backend_mm": module("bs3.backend_mm", "MMBackend", "MMConfig"),
                     "bs3.backend_oceanai": module("bs3.backend_oceanai", "OceanAIBackend", "BackendConfig")}):
        be = cli._backend(cli.parse_args(["infer", "a.mp4"]))
        assert be.kind == "MMBackend" and be.cfg.kw == {"lang": "ru", "asr_model": "openai/whisper-large-v3-turbo",
                                                        "ollama_model": "qwen2.5vl:7b"}, be.cfg.kw
        be = cli._backend(cli.parse_args(["infer", "a.mp4", "--mm-ckpt", "c.pt", "--ollama-model", "q"]))
        assert be.cfg.kw["checkpoint"] == "c.pt" and be.cfg.kw["ollama_model"] == "q"
        be = cli._backend(cli.parse_args(["infer", "a.mp4", "--backend", "oceanai"]))
        assert be.kind == "OceanAIBackend" and be.cfg.kw == {"lang": "ru", "corpus": None,
                                                             "asr_model": "openai/whisper-large-v3-turbo"}, be.cfg.kw
        be = cli._backend(cli.parse_args(["infer", "a.mp4", "--backend", "oceanai", "--models-dir", "/m"]))
        assert be.cfg.kw["models_dir"] == "/m"
        a = cli.parse_args(["explain", "a.mp4", "--out", "d"])
        a.backend = "mm"                                  # what cmd_explain sets before building the backend
        assert cli._backend(a).kind == "MMBackend"
    assert len(built) == 5


def test_building_a_backend_imports_no_ssl_mepr():
    """In a fresh interpreter: building either model imports neither SSL-MEPR (deleted) nor the web wrapper."""
    code = r"""
import sys, types, importlib.util
for name, cls, cfg in (("bs3.backend_oceanai", "OceanAIBackend", "BackendConfig"), ("bs3.backend_mm", "MMBackend", "MMConfig")):
    m = types.ModuleType(name)
    setattr(m, cfg, lambda **kw: kw)
    setattr(m, cls, type(cls, (), {"__init__": lambda self, c: None, "load": lambda self: self}))
    sys.modules[name] = m
from bs3 import cli
for b in ("oceanai", "mm"):
    cli._backend(cli.parse_args(["infer", "a.mp4", "--backend", b]))
print(sorted(n for n in sys.modules if n in ("bs3.backend_sslmepr", "bs3.backend_ensemble", "bs3.web.app")))
print(importlib.util.find_spec("bs3.backend_sslmepr") is None)
"""
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.split() == ["[]", "True"], r.stdout
    assert not (ROOT / "bs3" / "backend_sslmepr.py").exists()


def test_cmd_infer_report_shape():
    """A short clip without segmenting: the JSON report of the chosen model, Russian speech, the transcript_en of
    AMLAI 1.0 kept (the own model reads the English translation), no member list of an ensemble."""
    scores = {k: 0.3 + 0.05 * i for i, k in enumerate(TRAIT_KEYS)}
    scores["interview"] = 0.61
    be = types.SimpleNamespace(cfg=types.SimpleNamespace(corpus="own checkpoints", lang="ru"), load_seconds=2.04,
                               predict_video=lambda v, asr=True, transcript=None: {
                                   "scores": dict(scores), "transcript": "привет", "transcript_en": "hello",
                                   "seconds": 1.5})
    saved, saved_cu = cli._backend, media.check_upload
    cli._backend = lambda a, **kw: be
    media.check_upload = lambda *a, **k: None                  # the byte stub is not a real video; the shape is the point
    try:
        with tempfile.TemporaryDirectory(prefix="bs3_cli_test_") as d:
            clip = Path(d) / "clip.mp4"
            clip.write_bytes(b"")
            out = Path(d) / "out.json"
            a = cli.parse_args(["infer", str(clip), "--segment", "0", "--no-asr", "--out", str(out)])
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                a.fn(a)                                   # main() without its logging set-up
            rep = json.loads(out.read_text(encoding="utf-8"))
    finally:
        cli._backend, media.check_upload = saved, saved_cu
    assert "clip.mp4: openn=0.300" in printed.getvalue() and "interview=0.610" in printed.getvalue()
    assert rep["model"]["backend"] == "mm" and rep["model"]["lang"] == "ru" and rep["model"]["asr_model"] is None
    assert rep["model"]["primary"] is None and rep["model"]["corpus"] == "own checkpoints"
    assert rep["modalities_used"] == ["face", "audio", "text", "behavior"]
    assert rep["transcript"] == "привет" and rep["transcript_en"] == "hello"
    assert rep["traits"]["openness"]["score"] == 0.3 and rep["interview"]["score"] == 0.61
    assert rep["timings_sec"] == {"total": 1.5, "model_load": 2.0}
    assert "variant_scores" not in rep


def test_missing_input_is_a_calm_error_before_any_model():
    """A missing input is refused before a model is built (bs3.errors, stage 20): one calm Russian line on stderr, exit
    code 1, and _backend is never called. With -v the original exception is re-raised for a full traceback."""
    calls = []
    saved = cli._backend
    cli._backend = lambda a, **kw: calls.append(a)
    missing = "/no/such/файл_нет.mp4"
    try:
        with contextlib.redirect_stderr(io.StringIO()) as err:
            code = cli.main(["infer", missing, "--backend", "oceanai"])
        text = err.getvalue()
        lines = [ln for ln in text.splitlines() if ln.strip()]
        assert code == 1 and len(lines) == 1, (code, lines)
        assert re.search(r"[А-Яа-яЁё]", lines[0]) and "Файл не найден" in lines[0] and "Traceback" not in text
        assert calls == []                                          # the file check runs before _backend
        raised = None
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                cli.main(["infer", missing, "--backend", "oceanai", "-v"])
            except UserFacingError as e:
                raised = e
        assert raised is not None and "Файл не найден" in str(raised) and calls == []
        # explain refuses a missing input too
        with contextlib.redirect_stderr(io.StringIO()):
            assert cli.main(["explain", missing, "--out", "d"]) == 1
        assert calls == []
    finally:
        cli._backend = saved


def test_a_bad_upload_is_refused_before_any_model():
    """Stage 22 (owner amendment): the CLI runs media.check_upload on every input before building a model, so `bs3
    infer`/`explain` refuse an unreadable file or a clip shorter than two seconds with the same calm Russian line the
    web page shows — one stderr line, exit 1, and _backend is never reached."""
    calls = []
    saved_be, saved_cu = cli._backend, media.check_upload
    cli._backend = lambda a, **kw: calls.append(a)

    def refuse(path, work_dir):
        raise UserFacingError("Ролик слишком короткий (0,3 с): для оценки нужно хотя бы 2 секунды, "
                              "лучше — от 15 секунд речи в кадре.")
    media.check_upload = refuse
    try:
        with tempfile.TemporaryDirectory() as d:
            clip = Path(d) / "clip.mp4"
            clip.write_bytes(b"\x00" * 64)
            with contextlib.redirect_stderr(io.StringIO()) as err:
                code = cli.main(["infer", str(clip), "--backend", "oceanai"])
            assert code == 1 and "слишком коротк" in err.getvalue() and "Traceback" not in err.getvalue()
            assert calls == []                             # check_upload runs before _backend
            with contextlib.redirect_stderr(io.StringIO()) as err:
                code = cli.main(["explain", str(clip), "--out", str(Path(d) / "o")])
            assert code == 1 and "слишком коротк" in err.getvalue() and calls == []
    finally:
        cli._backend, media.check_upload = saved_be, saved_cu
