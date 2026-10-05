"""The research code in bs3-studio/training/ (3.1 moved it out of the runtime package): what it imports from bs3 still
exists, bs3 never imports it, the FIV2 paths resolve where they did inside bs3/mm/, and the FIV2 evaluation (formerly
`bs3 eval-fiv2`) keeps its options and its output. Nothing loads a model: the backends are fakes."""
from __future__ import annotations

import ast
import contextlib
import io
import json
import logging
import py_compile
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import bs3
from bs3.norms import OCEANAI_COLUMNS, TRAIT_KEYS

ROOT = Path(__file__).resolve().parents[1]            # bs3-studio/
REPO = ROOT.parent
TRAINING = ROOT / "training"
MOVED = ("mm_data", "mm_extract", "mm_train", "evaluate", "eval_fiv2")


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _source(module: str) -> Path | None:
    """The file of a module of this working tree by its dotted name (bs3.x.y or training.x), without importing it."""
    base = ROOT.joinpath(*module.split("."))
    for p in (base.with_suffix(".py"), base / "__init__.py"):
        if p.is_file():
            return p
    return None


def _top_names(path: Path) -> set[str]:
    """Names a module binds at its top level (defs, classes, assignments, imports, also inside if/try)."""
    names: set[str] = set()

    def visit(body):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                for t in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                    names.update(n.id for n in ast.walk(t) if isinstance(n, ast.Name))
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names.update((a.asname or a.name).split(".")[0] for a in node.names)
            elif isinstance(node, (ast.If, ast.Try)):
                visit(node.body)
                visit(node.orelse)
                for h in getattr(node, "handlers", []):
                    visit(h.body)
    visit(_tree(path).body)
    return names


def _imports(path: Path) -> list[tuple[str, list[str]]]:
    """(absolute module, imported names) of every import of a training file, at module or function level."""
    out = []
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            out += [(a.name, []) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level:                                # relative: inside the training package
                mod = "training" + ("." + mod if mod else "")
            out.append((mod, [a.name for a in node.names]))
    return out


def test_training_files_compile():
    files = sorted(TRAINING.glob("*.py"))
    assert {f.stem for f in files} >= set(MOVED) | {"__init__"}, files
    with tempfile.TemporaryDirectory() as d:
        for f in files:
            py_compile.compile(str(f), cfile=str(Path(d) / f"{f.stem}.pyc"), doraise=True)
    for gone in ("bs3/mm/data.py", "bs3/mm/extract.py", "bs3/mm/train.py", "bs3/evaluate.py",
                 "bs3/mm/interview_labels.csv"):
        assert not (ROOT / gone).exists(), gone


def test_what_training_imports_from_bs3_exists_without_torch():
    """Every bs3.* and training.* module a training file imports resolves, and every name it takes from them is bound
    there. The resolution itself (importlib.util.find_spec in a fresh interpreter) imports no torch."""
    bad, mods = [], set()
    for f in sorted(TRAINING.glob("*.py")):
        for mod, names in _imports(f):
            if mod.split(".")[0] not in ("bs3", "training"):
                continue
            src = _source(mod)
            if src is None:
                bad.append(f"{f.name}: no module {mod}")
                continue
            mods.add(mod)
            bound = _top_names(src)
            for n in names:
                if n not in bound and _source(f"{mod}.{n}") is None:
                    bad.append(f"{f.name}: {mod} has no {n}")
    assert not bad, "\n".join(bad)
    assert {"bs3.cli", "bs3.norms", "bs3.mm.model", "bs3.mm.extractors", "bs3.mm.faces", "training.mm_data",
            "training.evaluate"} <= mods, sorted(mods)
    code = ("import importlib.util, sys; mods = sys.argv[1:]; "
            "print([m for m in mods if importlib.util.find_spec(m) is None]); print('torch' in sys.modules)")
    r = subprocess.run([sys.executable, "-c", code, *sorted(mods)], cwd=str(ROOT), capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.split() == ["[]", "False"], r.stdout


def test_bs3_never_imports_training():
    bad = []
    for f in sorted((ROOT / "bs3").rglob("*.py")):
        for node in ast.walk(_tree(f)):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                names = [node.module or ""]
            elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
                fn = node.func
                fname = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
                if fname in ("import_module", "__import__") and isinstance(node.args[0].value, str):
                    names = [node.args[0].value]
            bad += [f"{f.relative_to(ROOT)}:{node.lineno}: {n}" for n in names
                    if n == "training" or n.startswith("training.")]
    assert not bad, "\n".join(bad)


def test_fiv2_paths_as_before_the_move():
    """DEFAULT_CSV_DIR is where bs3/mm/data.py pointed (BS/MM-PSYCHE/data/fiv2); the interview labels moved with the
    code and are still read from next to it."""
    from training import mm_data
    assert mm_data.DEFAULT_CSV_DIR == REPO / "MM-PSYCHE" / "data" / "fiv2", mm_data.DEFAULT_CSV_DIR
    assert mm_data.INTERVIEW_CSV == TRAINING / "interview_labels.csv"
    assert mm_data.INTERVIEW_CSV.is_file()
    with mm_data.INTERVIEW_CSV.open(encoding="utf-8") as fh:
        assert fh.readline().strip() == "video_name,split,interview"
    assert mm_data.LABEL_COLUMNS == ["openness", "conscientiousness", "extraversion", "agreeableness",
                                     "non-neuroticism"]


def test_fiv2_columns_left_the_runtime_package():
    from bs3 import norms
    from training import evaluate
    assert not hasattr(norms, "FIV2_COLUMNS")
    assert evaluate.FIV2_COLUMNS == ["openness", "conscientiousness", "extraversion", "agreeableness",
                                     "non-neuroticism"]
    backend_mm = (ROOT / "bs3" / "backend_mm.py").read_text(encoding="utf-8")
    assert "def predict_dir" not in backend_mm and "import pandas" not in backend_mm
    oceanai = (ROOT / "bs3" / "backend_oceanai.py").read_text(encoding="utf-8")
    assert "def predict_dir" in oceanai and "self.predict_dir(" in oceanai    # its predict_video scores through it


def _exit_code(parse, argv: list[str]) -> int | None:
    try:
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            parse(argv)
    except SystemExit as e:
        return e.code
    return None


def test_eval_fiv2_keeps_its_options():
    from training import eval_fiv2
    parse = eval_fiv2.build_parser().parse_args
    a = parse(["--dir", "d", "--out", "o.json", "--backend", "oceanai", "--lang", "en", "--corpus", "fi",
               "--models-dir", "m", "--limit", "5", "--asr"])
    assert (a.backend, a.lang, a.corpus, a.models_dir, a.limit, a.asr) == ("oceanai", "en", "fi", "m", 5, True)
    a = parse(["--dir", "d", "--out", "o"])
    assert (a.backend, a.lang, a.corpus, a.labels, a.limit, a.asr) == (bs3.DEFAULT_MODEL, bs3.LANG, None, None, 0,
                                                                        False)
    assert {"mm_ckpt", "ollama_model", "asr_model", "verbose"} <= set(vars(a))
    assert _exit_code(parse, ["--dir", "d", "--out", "o", "--backend", "ensemble"]) == 2
    assert _exit_code(parse, ["--out", "o"]) == 2
    r = subprocess.run([sys.executable, "-m", "training.eval_fiv2", "--help"], cwd=str(ROOT), capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "python -m training.eval_fiv2" in r.stdout and "--lang" in r.stdout


class _FakeMM:
    """AMLAI 1.0 without models: scores from the clip name; a clip named bad* fails like a clip without a face."""

    def __init__(self):
        self.cfg = types.SimpleNamespace(corpus="own checkpoints", lang="en")
        self.calls, self.loaded = [], 0

    def load(self):
        self.loaded += 1
        return self

    def predict_video(self, video, asr=True, transcript=None, behavior=None):
        self.calls.append((Path(video).name, asr, transcript, behavior))
        if Path(video).name.startswith("bad"):
            raise RuntimeError("no face")
        base = {"c1": 0.4, "c2": 0.55}.get(Path(video).stem, 0.5)      # 0.05 off the labels of the eval test
        return {"scores": {k: base + 0.01 * i for i, k in enumerate(TRAIT_KEYS)}}


class _Records(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@contextlib.contextmanager
def _captured(logger: logging.Logger):
    """The messages of `logger`, kept off stderr."""
    rec, propagate = _Records(), logger.propagate
    logger.addHandler(rec)
    logger.propagate = False
    try:
        yield rec.messages
    finally:
        logger.removeHandler(rec)
        logger.propagate = propagate


def test_predict_dir_of_amlai():
    from training import eval_fiv2
    be = _FakeMM()
    with tempfile.TemporaryDirectory(prefix="bs3_train_test_") as d, _captured(eval_fiv2.log) as logged:
        d = Path(d)
        for n in ("c1.mp4", "c2.MP4", "bad.mp4", "notes.md"):
            (d / n).write_bytes(b"")
        (d / "c1.txt").write_text("hello there", encoding="utf-8")
        (d / "c1.behavior.txt").write_text("calm", encoding="utf-8")
        df = eval_fiv2.predict_dir(be, d, asr=False)
        assert logged == ["bad.mp4 failed: no face"], logged
        assert be.loaded == 1
        assert be.calls == [("bad.mp4", False, None, None), ("c1.mp4", False, "hello there", "calm"),
                            ("c2.MP4", False, None, None)], be.calls
        assert list(df.columns) == ["Path"] + OCEANAI_COLUMNS
        assert df["Path"].tolist() == ["c1.mp4", "c2.MP4"]
        assert df["Openness"].tolist() == [0.4, 0.55]
        be.calls.clear()
        eval_fiv2.predict_dir(be, d, asr=True)            # with ASR the .txt transcript is not read
        assert [c[2] for c in be.calls] == [None, None, None]


def test_eval_fiv2_scores_and_writes_the_report():
    """cmd_eval with a fake AMLAI 1.0: the report and the predictions land next to --out, mACC/CCC from
    training.evaluate, the model's corpus and language, and a --limit subset through links."""
    from training import eval_fiv2
    be = _FakeMM()
    saved = eval_fiv2._backend, eval_fiv2.tempfile
    built = []
    eval_fiv2._backend = lambda a, **kw: built.append(kw) or be
    try:
        with tempfile.TemporaryDirectory(prefix="bs3_train_test_") as d:
            d = Path(d)
            # the folder of links for --limit goes into this temporary folder too
            eval_fiv2.tempfile = types.SimpleNamespace(mkdtemp=lambda prefix: tempfile.mkdtemp(prefix=prefix, dir=d))
            clips = d / "clips"
            clips.mkdir()
            rows = ["video_name,openness,conscientiousness,extraversion,agreeableness,non-neuroticism"]
            for i, stem in enumerate(("c1", "c2", "c3")):
                (clips / f"{stem}.mp4").write_bytes(b"")
                (clips / f"{stem}.txt").write_text("words", encoding="utf-8")
                rows.append(f"{stem}," + ",".join(f"{0.45 + 0.05 * i + 0.01 * j:.2f}" for j in range(5)))
            (clips / "labels.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
            out = d / "eval.json"
            a = eval_fiv2.build_parser().parse_args(["--dir", str(clips), "--out", str(out), "--lang", "en"])
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                eval_fiv2.cmd_eval(a)
            res = json.loads(out.read_text(encoding="utf-8"))
            pred = (d / "eval.pred.csv").read_text(encoding="utf-8").splitlines()
            a = eval_fiv2.build_parser().parse_args(["--dir", str(clips), "--out", str(d / "two.json"),
                                                     "--limit", "2"])
            with contextlib.redirect_stdout(io.StringIO()):
                eval_fiv2.cmd_eval(a)
            two = json.loads((d / "two.json").read_text(encoding="utf-8"))
    finally:
        eval_fiv2._backend, eval_fiv2.tempfile = saved
    assert built == [{"lang": "en", "corpus": None}, {"lang": "ru", "corpus": None}], built
    assert res["n"] == 3 and set(res["per_trait"]) == set(TRAIT_KEYS)
    assert res["per_trait"]["openness"]["acc"] == 0.95 and res["mACC"] == 0.95
    assert (res["backend"], res["corpus"], res["lang"], res["asr"]) == ("mm", "own checkpoints", "en", False)
    assert res["failed_files"] == [] and res["bs_version"] == bs3.__version__
    assert pred[0] == "Path," + ",".join(OCEANAI_COLUMNS) and len(pred) == 4
    assert "[ok] wrote" in printed.getvalue()
    assert two["n"] == 2
