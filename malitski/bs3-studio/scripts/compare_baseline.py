"""Behaviour baseline of BS Profiler 3.1 for the refactoring: what every finished job shows, captured once and compared
with a new rendering after every change.

    ~/bs/venv/bin/python bs3-studio/scripts/compare_baseline.py capture [--jobs ID ...] [--only BLOCK ...]
    ~/bs/venv/bin/python bs3-studio/scripts/compare_baseline.py compare [--jobs ID ...] [--only BLOCK ...]

capture renders every finished job of ~/bs3_data/web_jobs (a folder with result.json) and writes one file per block
to ~/bs3_data/refactor_baseline/<job>/<block>.txt; the page configuration goes to ~/bs3_data/refactor_baseline/_app/.
compare renders the same things again in memory and prints «OK <job>» or, per job and block, a unified diff against
the snapshot; exit code 1 on any difference, on a block present on one side only and on a job of the baseline that
is gone from the work dir. A job without a baseline (a new analysis) is listed as NEW and does not fail the run.

Blocks of a job:
  page.<name>           the 27 values of web.page.page_outputs, in their order (PAGE_BLOCKS)
  journal.result        the journal entry of a finished analysis (journal.result written into a temporary journal)
  characterization      characterization.build(view, mbti).plain()
  mbti                  mbti.get_mbti(rep, view): the saved section, or the one computed on display
  key_facts             the type card (mbti.fact_card) and facts.key_facts(view): (label, value, note, state)
  frame_captions.page   frame_captions.build as the page calls it while page_outputs runs (recorded, in order)
  frame_captions.pdf    the same while export_pdf runs
  pdf.text              the PDF of bs3.pdf.export_pdf as `pdftotext -layout` (pypdf when poppler is missing)
  pdf.meta              the file name and the page count of that PDF
  pdf.charts            the chart PNGs export_pdf draws for the PDF: file name, size and a hash of the pixels
  render.notes          what the rendering asked for and did not get here (translation, Ollama); empty normally
Blocks of _app: app.tree (the Blocks tree in order: types, labels, choices, default values, tab titles, events),
app.config (the Gradio config of the page with the element ids renumbered in page order), app.journal (the other
journal entries) and app.status (the progress line in its states, the error texts of a failed analysis).

Isolation. Every job is rendered from a temporary copy (result.json and explain/ copied, input.* linked for the media
probe of the PDF), so the page, the words list and the PDF export write nothing into the job folder, and the folder
is checked unchanged afterwards (names, sizes, mtimes). The journal goes to a temporary file. bs3.translate and
bs3.ollama are replaced by stubs: the page never waits for Ollama or loads the translation model, and a request for
either is recorded in render.notes (the page then shows what it shows without the translation). CUDA is hidden. The
results of a run go only into its temporary folder and the baseline folder; no job folder and no journal of the
service change.

Masked, the same way in capture and compare: the job folder and its temporary copy («<JOB>»), other temporary paths
(«<TMP>»), the date of the PDF («создан YYYY-MM-DD HH:MM»), the timestamp of a journal entry, `computed_at` of an
MBTI section computed on display, element ids and the random app id of the Gradio config. Images embedded as data
URIs (the key frames) become their size and a hash of the decoded pixels.

Options: --jobs ID … (job folder names; «_app» for the page configuration; without --jobs: all jobs and _app),
--only BLOCK … (a block or a prefix: «page», «pdf», «page.radar»), --work-dir, --baseline-dir, --max-lines N (diff
lines shown per block, 0 = all). A deliberate change of behaviour updates the snapshots of what it changes, in the
same commit: `capture --only <blocks>` (the other snapshots stay as they are).
"""
from __future__ import annotations

import argparse
import base64
import difflib
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]                  # bs3-studio/: `import bs3` resolves to this working tree
WORK_DIR = Path.home() / "bs3_data" / "web_jobs"
BASELINE_DIR = Path.home() / "bs3_data" / "refactor_baseline"
APP = "_app"
MANIFEST = "_manifest.json"
JSON_TAG = "#json\n"          # first line of a snapshot whose value is not a string (serialised as sorted JSON)

# the values of web.page.page_outputs by position (the output blocks of the page after the status line, without the
# PDF button); index 20 is «Сохранено в», 21 the hidden job folder the PDF button reads
PAGE_BLOCKS = ("radar", "bars", "key_facts_html", "characterization_html", "traits_timeline", "emotions_timeline",
               "voice_timeline", "speech_timeline", "emotion_bars", "segments_table", "speech_cards", "transcript",
               "face_cards", "face_chart", "key_frames_html", "contrib", "words", "behavior_description",
               "model_and_time", "result_json", "saved_to", "job_state", "method", "emo_intro", "mbti_types",
               "mbti_strip", "mbti_read")
JOB_BLOCKS = (tuple(f"page.{b}" for b in PAGE_BLOCKS)
              + ("journal.result", "characterization", "mbti", "key_facts", "frame_captions.page", "frame_captions.pdf",
                 "pdf.text", "pdf.meta", "pdf.charts", "render.notes"))
APP_BLOCKS = ("app.tree", "app.config", "app.journal", "app.status")

WALL_SEC = 83.0               # the processing time the journal entry of a finished job is written with
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 "
              "Safari/537.36")
DATA_URI = re.compile(r"data:image/([a-z+.-]+);base64,([A-Za-z0-9+/=]+)")
PDF_DATE = re.compile(r"(создан\s+)\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}")
JOURNAL_TIME = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", re.M)


class Request:
    """What journal.py reads from a gr.Request: the headers and the session hash."""

    def __init__(self, headers: dict, session_hash: str = "baseline"):
        self.headers, self.session_hash = headers, session_hash


# ----------------------------------------------------------------------------------------------- isolation ----
class Env:
    """The process-wide setup of a run: temporary root, hidden CUDA, temporary journal, the translation and Ollama
    stubs."""

    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.calls: list[str] = []
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["BS3_JOURNAL"] = str(tmp / "journal.txt")
        os.environ["GRADIO_TEMP_DIR"] = str(tmp / "gradio")
        os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
        os.environ["MPLBACKEND"] = "Agg"
        (tmp / "gradio").mkdir()
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        sys.modules["bs3.translate"] = self._translate_stub()
        sys.modules["bs3.ollama"] = self._ollama_stub()
        import logging
        logging.basicConfig(level=logging.ERROR)

    def _translate_stub(self) -> types.ModuleType:
        """bs3.translate without Marian and Ollama: Ollama is «unreachable», every translation fails (and is noted)."""
        calls = self.calls
        stub = types.ModuleType("bs3.translate")

        def ollama_available(*_a, **_k) -> bool:
            calls.append("ollama_available (answered: unreachable)")
            return False

        def blocked(name: str):
            def fn(*_a, **_k):
                calls.append(f"{name} (blocked)")
                raise RuntimeError(f"compare_baseline: {name} is not available here (no translation model, no Ollama)")
            return fn

        def __getattr__(name: str):
            if name.startswith("__"):
                raise AttributeError(name)
            return blocked(name)

        stub.ollama_available = ollama_available
        stub.__getattr__ = __getattr__
        return stub

    def _ollama_stub(self) -> types.ModuleType:
        """bs3.ollama without a server: it never answers, a call or an address lookup fails (and is noted). Setting
        the model of the process (pipeline.Studio) needs no server and is kept."""
        calls = self.calls
        stub = types.ModuleType("bs3.ollama")
        current = {"model": None}

        def available(*_a, **_k) -> bool:
            calls.append("ollama.available (answered: unreachable)")
            return False

        def post_json(path="", *_a, **_k):
            calls.append(f"ollama.post_json {path} (blocked)")
            raise RuntimeError(f"compare_baseline: Ollama is not available here ({path})")

        def configure(model=None) -> None:
            current["model"] = model or None

        def model():
            return current["model"]

        def __getattr__(name: str):
            if name.startswith("__"):
                raise AttributeError(name)

            def blocked(*_a, **_k):
                calls.append(f"ollama.{name} (blocked)")
                raise RuntimeError(f"compare_baseline: ollama.{name} is not available here")
            return blocked

        stub.available, stub.post_json, stub.configure, stub.model = available, post_json, configure, model
        stub.mark_down = lambda: None
        stub.__getattr__ = __getattr__
        return stub


def _copy_job(job: Path, dest: Path) -> None:
    """What rendering reads: result.json and explain/ as files, input.* as a link (read only, for the media probe)."""
    dest.mkdir(parents=True)
    shutil.copy2(job / "result.json", dest / "result.json")
    if (job / "explain").is_dir():
        shutil.copytree(job / "explain", dest / "explain", symlinks=False)
    for p in sorted(job.glob("input.*")):
        (dest / p.name).symlink_to(p)


def _folder_state(job: Path) -> dict:
    """Names, sizes and mtimes of everything in a job folder except the segment videos: rendering must not change it."""
    out = {".": job.stat().st_mtime_ns}
    for p in sorted(job.rglob("*")):
        rel = p.relative_to(job)
        if rel.parts[0] == "segments":
            continue
        st = p.lstat()
        out[rel.as_posix()] = (st.st_size, st.st_mtime_ns)
    return out


# ----------------------------------------------------------------------------------------------- masking ------
def _pixels(data: bytes) -> str:
    from PIL import Image
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        return f"{im.width}x{im.height} {im.mode} pixels sha256:{hashlib.sha256(im.tobytes()).hexdigest()[:16]}"


def _uri(m: re.Match) -> str:
    try:
        return f"data:image/{m.group(1)};base64,[{_pixels(base64.b64decode(m.group(2)))}]"
    except Exception as e:  # noqa: BLE001
        return f"data:image/{m.group(1)};base64,[undecodable {type(e).__name__}, bytes sha256:" \
               f"{hashlib.sha256(m.group(2).encode()).hexdigest()[:16]}]"


class Mask:
    """Replaces the volatile parts of a rendered value (see the module docstring)."""

    def __init__(self, pairs: list[tuple[str, str]]):
        self.pairs = sorted(((p, t) for p, t in pairs if p), key=lambda pt: -len(pt[0]))

    def __call__(self, value):
        if isinstance(value, str):
            for p, token in self.pairs:
                value = value.replace(p, token)
            return DATA_URI.sub(_uri, value)
        if isinstance(value, dict):
            return {self(k): self(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self(v) for v in value]
        return value


def serialise(value) -> str:
    if isinstance(value, str):
        return value
    return JSON_TAG + json.dumps(value, ensure_ascii=False, indent=1, sort_keys=True, default=repr) + "\n"


# ----------------------------------------------------------------------------------------------- one job ------
def _error(e: BaseException) -> str:
    return f"ERROR {type(e).__name__}: {e}"


def _pdf_text(pdf: Path) -> tuple[str, int | None, str]:
    """(text, pages, tool). pdftotext -layout of poppler-utils; pypdf when poppler is missing."""
    if shutil.which("pdftotext"):
        r = subprocess.run(["pdftotext", "-layout", "-enc", "UTF-8", str(pdf), "-"], capture_output=True, check=True)
        text, tool = r.stdout.decode("utf-8"), "pdftotext -layout"
        pages = None
        if shutil.which("pdfinfo"):
            info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, check=True).stdout.decode("utf-8", "replace")
            m = re.search(r"^Pages:\s+(\d+)", info, re.M)
            pages = int(m.group(1)) if m else None
        if pages is None:
            pages = text.count("\f")
    else:
        try:
            import pypdf
        except ImportError:
            return "not captured: neither pdftotext (poppler-utils) nor pypdf is installed", None, "none"
        reader = pypdf.PdfReader(str(pdf))
        text = "\f".join(p.extract_text(extraction_mode="layout") or "" for p in reader.pages)
        pages, tool = len(reader.pages), "pypdf layout"
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return PDF_DATE.sub(r"\1YYYY-MM-DD HH:MM", text), pages, tool


def render_job(env: Env, job: Path, want) -> dict:
    """{block: value} of one job, masked; only the blocks `want(block)` accepts (the page is rendered anyway when
    the journal is wanted: the app writes the journal entry after the page, from the same result)."""
    from bs3 import characterization, facts, frame_captions, journal, mbti
    from bs3.pdf import export_pdf
    from bs3.scores import clean_view
    from bs3.web import page

    copy_dir = env.tmp / "jobs" / job.name
    _copy_job(job, copy_dir)
    mask = Mask([(str(copy_dir), "<JOB>"), (str(job), "<JOB>"), (str(job.resolve()), "<JOB>"),
                 (str(env.tmp), "<TMP>")])
    before = _folder_state(job)
    env.calls.clear()
    out: dict = {}

    def load() -> dict:
        rep = json.loads((copy_dir / "result.json").read_text(encoding="utf-8"))
        rep["job_dir"] = str(copy_dir)            # as the app does for a finished job (preview) and run_analysis
        return rep

    # frame_captions.build is recorded while the page and the PDF are rendered: the captions they print
    recorded: list = []
    real_build = frame_captions.build

    def build(*a, **k):
        res = real_build(*a, **k)
        recorded.append(res)
        return res

    frame_captions.build = build
    try:
        if any(want(b) for b in JOB_BLOCKS if b.startswith(("page.", "journal.", "frame_captions.page"))):
            rep = load()
            recorded.clear()
            try:
                outs = page.page_outputs(rep)
                if len(outs) != len(PAGE_BLOCKS):
                    raise RuntimeError(f"page_outputs gave {len(outs)} values, the baseline knows {len(PAGE_BLOCKS)}")
                for name, v in zip(PAGE_BLOCKS, outs):
                    out[f"page.{name}"] = v
            except Exception as e:  # noqa: BLE001
                for name in PAGE_BLOCKS:
                    out[f"page.{name}"] = _error(e)
            out["frame_captions.page"] = list(recorded) if recorded else "not called (no key frames shown on the page)"
            try:
                journal.PATH = env.tmp / f"journal_{job.name}.txt"
                journal.result(Request({"user-agent": USER_AGENT}), rep, WALL_SEC)
                out["journal.result"] = JOURNAL_TIME.sub("YYYY-MM-DD HH:MM:SS",
                                                         journal.PATH.read_text(encoding="utf-8"))
            except Exception as e:  # noqa: BLE001
                out["journal.result"] = _error(e)

        rep = load()
        try:
            view = clean_view(rep)
            mb = mbti.get_mbti(rep, view)
            shown = dict(mb) if isinstance(mb, dict) else mb
            if isinstance(shown, dict) and shown.get("computed_on_render"):
                shown["computed_at"] = "<computed on display>"
            out["mbti"] = shown
            out["characterization"] = characterization.build(view, mb).plain()
            out["key_facts"] = {"type_card": mbti.fact_card(mb), "facts": facts.key_facts(view)}
        except Exception as e:  # noqa: BLE001
            for block in ("mbti", "characterization", "key_facts"):
                out.setdefault(block, _error(e))

        if any(want(b) for b in ("pdf.text", "pdf.meta", "pdf.charts", "frame_captions.pdf")):
            recorded.clear()
            try:
                pdf = Path(export_pdf(copy_dir))
                text, pages, tool = _pdf_text(pdf)
                out["pdf.text"] = text
                out["pdf.meta"] = {"file": pdf.name, "folder": str(pdf.parent), "pages": pages, "text_tool": tool}
                charts = {}
                for p in sorted((copy_dir / "charts").glob("*")) if (copy_dir / "charts").is_dir() else []:
                    charts[p.name] = _pixels(p.read_bytes()) if p.suffix.lower() == ".png" else \
                        f"bytes sha256:{hashlib.sha256(p.read_bytes()).hexdigest()[:16]}"
                out["pdf.charts"] = charts
            except Exception as e:  # noqa: BLE001
                out["pdf.text"] = out["pdf.meta"] = out["pdf.charts"] = _error(e)
            out["frame_captions.pdf"] = list(recorded) if recorded else "not called (no key frames in the PDF)"
    finally:
        frame_captions.build = real_build

    out["render.notes"] = sorted(set(env.calls))
    after = _folder_state(job)
    if after != before:
        changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        raise SystemExit(f"STOP: rendering changed the job folder {job}: {changed}")
    shutil.rmtree(copy_dir, ignore_errors=True)
    return {k: serialise(mask(v)) for k, v in out.items() if want(k)}


# ----------------------------------------------------------------------------------------------- the page -----
_TREE_PROPS = ("label", "value", "choices", "variant", "interactive", "visible", "show_label", "container", "scale",
               "min_width", "height", "lines", "max_lines", "autoscroll", "sources", "language", "elem_classes",
               "elem_id", "equal_height", "open", "selected", "size", "info", "placeholder", "show_copy_button")


def _short(v) -> str:
    s = json.dumps(v, ensure_ascii=False)
    if len(s) > 120:
        return f"{s[:80]}… ({len(s)} chars, sha256:{hashlib.sha256(s.encode()).hexdigest()[:12]})"
    return s


def _normalised_config(demo) -> dict:
    """demo.get_config_file() with every element id replaced by its position in the page («#0» is the Blocks root,
    then the layout in reading order, then the components outside the layout), the random app id and the memory
    addresses in the repr of Python callables (gr.State's delete_callback) masked."""
    text = json.dumps(demo.get_config_file(), default=repr)
    cfg = json.loads(re.sub(r" at 0x[0-9a-fA-F]+>", " at 0x…>", text))
    ids: dict = {}

    def visit(node):
        ids.setdefault(node["id"], len(ids))
        for ch in node.get("children") or []:
            visit(ch)

    visit(cfg["layout"])
    for c in cfg.get("components") or []:
        ids.setdefault(c["id"], len(ids))

    def rid(i):
        return f"#{ids[i]}" if i in ids else i

    def relayout(node):
        return {**node, "id": rid(node["id"]), "children": [relayout(ch) for ch in node.get("children") or []]} \
            if "children" in node else {**node, "id": rid(node["id"])}

    cfg["layout"] = relayout(cfg["layout"])
    for c in cfg.get("components") or []:
        c["id"] = rid(c["id"])
    for d in cfg.get("dependencies") or []:
        for key in ("inputs", "outputs"):
            if isinstance(d.get(key), list):
                d[key] = [rid(i) for i in d[key]]
        if isinstance(d.get("targets"), list):
            d["targets"] = [[rid(t[0]), *t[1:]] if isinstance(t, list) and t else t for t in d["targets"]]
    for key in ("app_id",):
        if key in cfg:
            cfg[key] = "<masked>"
    return cfg


def _tree(cfg: dict) -> str:
    comps = {c["id"]: c for c in cfg.get("components") or []}
    lines = [f"page title: {cfg.get('title')!r}", f"gradio {cfg.get('version')}, theme {cfg.get('theme')!r}"]

    def name(i) -> str:
        c = comps.get(i) or {}
        lab = (c.get("props") or {}).get("label")
        return f"{c.get('type', 'blocks')}{i}" + (f" «{lab}»" if lab else "")

    def walk(node, depth):
        c = comps.get(node["id"]) or {}
        props = c.get("props") or {}
        bits = [f"{c.get('type', 'blocks')} {node['id']}"]
        for k in _TREE_PROPS:
            if k in props and props[k] is not None and props[k] != []:
                bits.append(f"{k}={_short(props[k])}")
        lines.append("  " * depth + " ".join(bits))
        for ch in node.get("children") or []:
            walk(ch, depth + 1)

    walk(cfg["layout"], 0)
    lines.append("events:")
    for n, d in enumerate(cfg.get("dependencies") or []):
        targets = ", ".join(f"{name(t[0])}.{t[1]}" for t in d.get("targets") or [] if isinstance(t, list))
        lines.append(f"  [{n}] {targets}")
        lines.append(f"      inputs: {', '.join(name(i) for i in d.get('inputs') or []) or '—'}")
        lines.append(f"      outputs: {', '.join(name(i) for i in d.get('outputs') or []) or '—'}")
        extra = {k: d.get(k) for k in ("queue", "show_progress", "cancels", "api_name", "js", "trigger_mode")
                 if k in d}
        lines.append(f"      {json.dumps(extra, ensure_ascii=False, sort_keys=True)}")
    return "\n".join(lines) + "\n"


def render_app(env: Env, want) -> dict:
    from bs3 import errors, journal
    from bs3.pipeline import Studio
    from bs3.web import app as webapp

    mask = Mask([(str(env.tmp), "<TMP>")])
    out: dict = {}
    if want("app.tree") or want("app.config"):
        try:
            demo = webapp.build_app(Studio(), env.tmp / "app_jobs")
            cfg = _normalised_config(demo)
            out["app.tree"], out["app.config"] = _tree(cfg), cfg
        except Exception as e:  # noqa: BLE001
            out["app.tree"] = out["app.config"] = _error(e)
    if want("app.journal"):
        journal.PATH = env.tmp / "journal_app.txt"
        local = Request({"user-agent": USER_AGENT})
        proxied = Request({"x-forwarded-for": "203.0.113.7, 10.0.0.1", "x-remote-user": "tester",
                           "user-agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 "
                                         "(KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1"}, "0123456789")
        video = env.tmp / "upload" / "Интервью.mp4"
        video.parent.mkdir(exist_ok=True)
        video.write_bytes(b"\0" * 3_000_000)
        journal.visit(local)
        journal.visit(proxied)
        for member in ("mm", "oceanai"):
            journal.start(proxied, str(video), member)
        journal.start(local, str(env.tmp / "missing.mov"), "mm")
        journal.failed(proxied, str(video), "остановлено пользователем", stopped=True)
        journal.failed(local, str(video), errors.user_message(RuntimeError("CUDA out of memory")))
        out["app.journal"] = JOURNAL_TIME.sub("YYYY-MM-DD HH:MM:SS", journal.PATH.read_text(encoding="utf-8"))
    if want("app.status"):
        try:
            status = [webapp._status_html(0.0, "[0 с] запуск"),
                      webapp._status_html(0.34, "[1 мин 05 с] Отрезок 3/29 (0:40–1:00)"),
                      webapp._status_html(1.0, "обработано за 5 мин 12 с", state="done"),
                      webapp._status_html(1.0, "предпросмотр готового результата", state="done"),
                      webapp._status_html(0.5, "по запросу пользователя", state="stopped"),
                      webapp._status_html(0.0, "текущий отрезок дорабатывается, затем обработка прерывается (до ~20 с)",
                                          state="stopped"),
                      webapp._status_html(0.2, "Не хватило памяти видеокарты.", state="error")]
            messages = {repr(e): errors.user_message(e) for e in (
                RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"),
                RuntimeError("No segment could be analysed"), RuntimeError("no predictions for any file"),
                RuntimeError("no frames decoded"), RuntimeError("All ensemble members failed: oceanai"),
                subprocess.CalledProcessError(1, ["ffprobe", "x.mp4"]), RuntimeError("Неизвестная модель «x». Выберите …"),
                ValueError("unexpected value 3"), KeyError("traits"))}
            out["app.status"] = {"status_line": status, "analysis_errors": messages}
        except Exception as e:  # noqa: BLE001
            out["app.status"] = _error(e)
    return {k: serialise(mask(v)) for k, v in out.items() if want(k)}


# ----------------------------------------------------------------------------------------------- diff ---------
def _pretty(text: str) -> list[str]:
    """Lines for a readable diff: tags on lines of their own, the chart spec of an iframe srcdoc unescaped and cut at
    its keys, long lines cut into pieces. For display only; the comparison itself is exact."""
    t = text.replace("&quot;", '"').replace("&amp;", "&")
    t = re.sub(r">\s*<", ">\n<", t)
    lines = []
    for line in t.split("\n"):
        if len(line) > 160:
            line = re.sub(r'(,"|\},|\],|;)', "\\1\n", line)
        for piece in line.split("\n"):
            lines.extend(piece[i:i + 200] for i in range(0, max(len(piece), 1), 200))
    return lines


def diff_text(old: str, new: str, label: str, max_lines: int) -> list[str]:
    d = list(difflib.unified_diff(_pretty(old), _pretty(new), f"baseline/{label}", f"now/{label}", lineterm="", n=2))
    if not d:                                        # a difference the display lines hide (escaping, whitespace)
        i = next((k for k, (a, b) in enumerate(zip(old, new)) if a != b), min(len(old), len(new)))
        d = [f"first difference at character {i} (lengths {len(old)} / {len(new)}):",
             f"  baseline: {old[max(0, i - 60):i + 60]!r}", f"  now:      {new[max(0, i - 60):i + 60]!r}"]
    if max_lines and len(d) > max_lines:
        d = d[:max_lines] + [f"… {len(d) - max_lines} more diff lines"]
    return d


# ----------------------------------------------------------------------------------------------- driver -------
def _read(p: Path) -> str:
    with open(p, encoding="utf-8", newline="") as f:
        return f.read()


def _write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def _matcher(only: list[str] | None):
    if not only:
        return lambda block: True
    return lambda block: any(block == o or block.startswith(o.rstrip(".") + ".") for o in only)


def _git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "?"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=("capture", "compare"))
    ap.add_argument("--jobs", nargs="+", help=f"job folder names (and/or {APP}); default: every job and {APP}")
    ap.add_argument("--only", nargs="+", help="blocks or prefixes: page, page.radar, pdf, pdf.text, app.tree …")
    ap.add_argument("--work-dir", default=str(WORK_DIR))
    ap.add_argument("--baseline-dir", default=str(BASELINE_DIR))
    ap.add_argument("--max-lines", type=int, default=80, help="diff lines shown per block (0: all)")
    a = ap.parse_args(argv)
    work, base = Path(a.work_dir).expanduser(), Path(a.baseline_dir).expanduser()
    want = _matcher(a.only)
    t_start = time.time()
    if a.command == "compare" and not (base / MANIFEST).is_file():
        print(f"no baseline in {base}: run `capture` first")
        return 2

    jobs_now = sorted(p.name for p in work.iterdir() if (p / "result.json").is_file()) if work.is_dir() else []
    jobs_base = sorted(p.name for p in base.iterdir() if p.is_dir() and p.name != APP) if base.is_dir() else []
    names = sorted(set(jobs_now) | (set(jobs_base) if a.command == "compare" else set()))
    do_app = not a.jobs or APP in a.jobs
    if a.jobs:
        unknown = [j for j in a.jobs if j != APP and j not in names]
        if unknown:
            print(f"unknown job(s): {', '.join(unknown)}")
            return 2
        names = [j for j in names if j in a.jobs]
    do_app = do_app and any(want(b) for b in APP_BLOCKS)
    if not any(want(b) for b in JOB_BLOCKS):
        names = []

    failed, report = 0, []
    with tempfile.TemporaryDirectory(prefix="bs3_baseline_") as tmp:
        env = Env(Path(tmp))
        from bs3.web.page import PAGE_BLOCKS as _WEB_PAGE_BLOCKS
        if _WEB_PAGE_BLOCKS != PAGE_BLOCKS:                       # the snapshot contract kept here must match the page
            raise SystemExit(f"compare_baseline: bs3.web.page.PAGE_BLOCKS drifted from the snapshot contract "
                             f"({_WEB_PAGE_BLOCKS} vs {PAGE_BLOCKS})")
        targets = ([APP] if do_app else []) + names
        for name in targets:
            t0 = time.time()
            snap_dir = base / name
            if name != APP and name not in jobs_now:
                print(f"MISSING {name}: in the baseline, not in {work} (or it lost its result.json)")
                failed += 1
                continue
            if a.command == "compare" and not snap_dir.is_dir():
                print(f"NEW {name}: no baseline yet (capture --jobs {name})")
                continue
            blocks = render_app(env, want) if name == APP else render_job(env, work / name, want)
            secs = time.time() - t0
            if a.command == "capture":
                if not a.only and snap_dir.is_dir():
                    shutil.rmtree(snap_dir)              # a full capture of a job leaves no stale block behind
                for block, text in blocks.items():
                    _write(snap_dir / f"{block}.txt", text)
                print(f"captured {name}: {len(blocks)} blocks, {secs:.1f} s")
                report.append((name, sorted(blocks)))
                continue
            stored = {p.name[:-4]: p for p in snap_dir.glob("*.txt") if want(p.name[:-4])}
            bad = []
            for block in sorted(set(stored) | set(blocks)):
                if block not in blocks:
                    bad.append((block, ["block in the baseline, not rendered now"]))
                elif block not in stored:
                    bad.append((block, ["block rendered now, not in the baseline (capture --only it if intended)"]))
                else:
                    old = _read(stored[block])
                    if old != blocks[block]:
                        bad.append((block, diff_text(old, blocks[block], f"{name}/{block}", a.max_lines)))
            if not bad:
                print(f"OK {name} ({len(blocks)} blocks, {secs:.1f} s)")
                continue
            failed += 1
            print(f"DIFF {name}: {len(bad)} of {len(set(stored) | set(blocks))} blocks differ ({secs:.1f} s)")
            for block, lines in bad:
                print(f"--- {name} / {block}")
                for line in lines:
                    print("    " + line)

    total = time.time() - t_start
    if a.command == "capture":
        man_path = base / MANIFEST
        man = json.loads(_read(man_path)) if man_path.is_file() else {}
        import gradio
        man.update({"captured_at": time.strftime("%Y-%m-%d %H:%M:%S"), "git_head": _git_head(),
                    "gradio": gradio.__version__, "python": sys.version.split()[0]})
        man.setdefault("targets", {})
        for name, blocks in report:
            prev = set(man["targets"].get(name) or []) if a.only else set()
            man["targets"][name] = sorted(prev | set(blocks))
        _write(man_path, json.dumps(man, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
        print(f"capture: {len(report)} target(s) into {base} in {total:.1f} s")
        return 0 if not failed else 1
    print(f"compare: {len(targets)} target(s), {failed} with differences, {total:.1f} s"
          + ("" if failed else " — OK"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
