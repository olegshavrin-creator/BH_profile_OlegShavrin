"""Failure paths of the web page (refactoring plan of 3.1, stage 20), driven headless without a GPU: the handlers found
in demo.fns are called with a fake run_analysis, and the journal goes to a temporary file. A failed analysis ends with a
calm Russian bar, a gr.Error with that message and a journal ОШИБКА line — the model's English exception never reaches
the page. A cancelled run is «Остановлено». A finished analysis whose page cannot be built is calm too (FP5). The status
bar and the two PDF-button refusals are pinned. Stage 21 adds the stop button and one-analysis-at-a-time: «Остановить»
stops only the caller's own run, the outcome line is written from the worker thread (so it survives the tab closing),
and run_lock keeps a new run from overlapping a stopped one."""
from __future__ import annotations

import contextlib
import logging
import re
import tempfile
import threading
import time
from pathlib import Path

from samples import rep

from bs3 import jobfiles, jobview, journal
from bs3.errors import AnalysisCancelled
from bs3.norms import TRAIT_KEYS
from bs3.web import app, page

# an explanation.json that needs no translation (readable words already Russian, marked ollama): opening the job for a
# preview writes nothing back and calls no model
EXPL = {"modalities": {"input_x_gradient": {k: {"face": {"share": 0.6}, "audio": {"share": 0.38},
                                                "text": {"share": 0.01}, "behavior": {"share": 0.01}}
                                            for k in TRAIT_KEYS}},
        "readable_words": {"transcript_words": {k: {"up": [{"word": "work", "ru": "работа", "signed": 0.01}],
                                                    "down": []} for k in TRAIT_KEYS}},
        "readable_words_by": "ollama"}


class _Req:
    headers = {"user-agent": "Mozilla/5.0 (Windows NT 10.0) Chrome/120"}
    session_hash = "abcdef123"


@contextlib.contextmanager
def _patched(module, **names):
    old = {k: getattr(module, k) for k in names}
    for k, v in names.items():
        setattr(module, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(module, k, v)


@contextlib.contextmanager
def _quiet(name: str = "bs3.web"):
    """Keep the intentional log.error/log.exception of a failure off stderr (a handler stops the last-resort one)."""
    lg = logging.getLogger(name)
    h = logging.NullHandler()
    lg.addHandler(h)
    try:
        yield
    finally:
        lg.removeHandler(h)


def _build(work_dir: Path):
    from bs3.pipeline import Studio
    return app.build_app(Studio(), work_dir)


def _handlers(demo) -> dict:
    """The registered handlers by function name (analyze, stop, make_pdf, on_visit)."""
    out = {}
    for f in demo.fns.values():
        fn = getattr(f, "fn", None)
        if fn is not None:
            out.setdefault(fn.__name__, fn)
    return out


def _drive(gen):
    """Iterate a handler generator to the end; return (yields, gr_error_or_None)."""
    import gradio as gr
    outs = []
    try:
        for o in gen:
            outs.append(o)
    except gr.Error as e:
        return outs, e
    return outs, None


def _strings(outs) -> list[str]:
    """Every string among the yielded tuples (the status bar is index 0; the rest are gr.update())."""
    return [x for row in outs for x in row if isinstance(x, str)]


def _mm_job(base: Path) -> Path:
    """A finished AMLAI 1.0 (3.1) job folder that a preview can open (result.json + explanation.json on disk)."""
    r = rep("B")
    r["model"].update({"selected": "mm", "primary": "mm", "selected_title": "AMLAI 1.0", "backend": "mm",
                       "version": "3.1.0a1"})
    own = {k: r["variant_scores"]["mm"][k] for k in TRAIT_KEYS}
    r["variant_scores"] = {"mm": r["variant_scores"]["mm"]}
    for t in r["timeline"]:
        t.update({"members_used": ["mm"], "primary_used": "mm", "variants": {"mm": dict(own)}, "scores": dict(own)})
    job = base / "20000101_000000_0f3a9c1e"
    job.mkdir(parents=True)
    r["job_dir"] = str(job)
    jobfiles.write_json(job / jobfiles.RESULT, r)
    (job / jobfiles.EXPLAIN_DIR).mkdir()
    jobfiles.write_json(jobfiles.explanation_path(job), EXPL)
    return job


def _req(sess: str):
    """A request of its own browser session (the stop button matches the caller's session_hash)."""
    r = _Req()
    r.session_hash = sess
    return r


def _kinds(text: str) -> list[str]:
    """The kind word (ВХОД / СТАРТ / РЕЗУЛЬТАТ / ОШИБКА / ОСТАНОВЛЕНО) of each journal head line, in order."""
    out = []
    for line in text.splitlines():
        m = re.match(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\s+(\S+)", line)
        if m:
            out.append(m.group(1))
    return out


class _Gate:
    """A fake run_analysis the test drives by hand: it registers as active (so an overlap of two runs would show up in
    `max`), reports when it is running under the lock (`started`), then loops until it is told to finish (`release`) or
    is stopped through `should_stop` (then it raises AnalysisCancelled, like the real segment loop)."""

    def __init__(self, result_rep=None):
        self.result_rep = result_rep
        self.active = self.max = 0
        self._lock = threading.Lock()
        self.started = threading.Semaphore(0)
        self.release = threading.Event()

    def run(self, studio, work_dir, video, *, member, explain, progress=None, should_stop=None):
        with self._lock:
            self.active += 1
            self.max = max(self.max, self.active)
        self.started.release()
        try:
            while not self.release.is_set():
                if should_stop and should_stop():
                    raise AnalysisCancelled("остановлено пользователем")
                time.sleep(0.005)
            return self.result_rep
        finally:
            with self._lock:
                self.active -= 1


def test_analysis_failure_is_calm():
    """A model exception (English) becomes the calm Russian bar and dialog; the journal has СТАРТ and ОШИБКА; nothing
    of the exception text is yielded to the page."""
    import gradio as gr
    with tempfile.TemporaryDirectory() as d, _patched(journal, PATH=Path(d) / "journal.txt"):
        demo = _build(Path(d) / "jobs")
        analyze = _handlers(demo)["analyze"]

        def boom(*a, **k):
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

        with _patched(app, run_analysis=boom), _quiet():
            outs, err = _drive(analyze("/uploads/video.mp4", "mm", _Req()))
        text = journal.PATH.read_text(encoding="utf-8")
    assert isinstance(err, gr.Error)
    msg = "Не хватило памяти видеокарты. Подождите минуту и запустите анализ заново."
    assert err.message == msg and err.title == app.ERROR_TITLE
    last = outs[-1][0]
    assert msg in last and app.STATUS_LABELS["error"] in last                      # the red bar carries the message
    for s in _strings(outs):
        for leak in ("out of memory", "CUDA", "GiB", "Traceback", "RuntimeError"):
            assert leak not in s, (leak, s)
    assert "СТАРТ" in text and "ОШИБКА" in text and "video.mp4" in text
    assert "out of memory" not in text and "CUDA" not in text                     # only the Russian message is journaled


def test_cancelled_analysis_is_stopped():
    """AnalysisCancelled gives the «Остановлено» bar and dialog and an ОСТАНОВЛЕНО journal line. Stage 21: the worker
    thread writes that line itself (it survives the tab closing); the generator only shows the bar and the dialog."""
    import gradio as gr
    with tempfile.TemporaryDirectory() as d, _patched(journal, PATH=Path(d) / "journal.txt"):
        demo = _build(Path(d) / "jobs")
        analyze = _handlers(demo)["analyze"]

        def cancel(*a, **k):
            raise AnalysisCancelled("остановлено пользователем")

        with _patched(app, run_analysis=cancel):
            outs, err = _drive(analyze("/uploads/video.mp4", "mm", _Req()))
        text = journal.PATH.read_text(encoding="utf-8")
    assert isinstance(err, gr.Error)
    assert err.message == "Обработка остановлена. Запустите анализ заново." and err.title == "Остановлено"
    assert "по запросу пользователя" in outs[-1][0]
    assert "ОСТАНОВЛЕНО" in text and "ОШИБКА" not in text


def test_render_failure_after_a_finished_analysis_is_calm():
    """FP5: the analysis finished and was saved, but the page could not be built — a calm message, no traceback, an
    ОШИБКА line."""
    import gradio as gr
    with tempfile.TemporaryDirectory() as d, _patched(journal, PATH=Path(d) / "journal.txt"):
        demo = _build(Path(d) / "jobs")
        analyze = _handlers(demo)["analyze"]

        def ok(*a, **k):
            return {"job_dir": "irrelevant"}

        def broken(rep):
            raise KeyError("traits")

        with _patched(app, run_analysis=ok), _patched(jobview, for_page=broken), _quiet():
            outs, err = _drive(analyze("/uploads/video.mp4", "mm", _Req()))
        text = journal.PATH.read_text(encoding="utf-8")
    msg = ("Анализ завершён и сохранён, но страницу с результатом построить не удалось. "
           "Подробности записаны в журнал сервера.")
    assert isinstance(err, gr.Error) and err.message == msg and err.title == app.ERROR_TITLE
    assert msg in outs[-1][0]
    assert "СТАРТ" in text and "ОШИБКА" in text and "Traceback" not in text and "KeyError" not in text


def test_no_video_asks_for_one_and_writes_no_journal_line():
    import gradio as gr
    with tempfile.TemporaryDirectory() as d, _patched(journal, PATH=Path(d) / "journal.txt"):
        demo = _build(Path(d) / "jobs")
        analyze = _handlers(demo)["analyze"]
        outs, err = _drive(analyze("", "mm", _Req()))
        wrote = journal.PATH.exists()
    assert isinstance(err, gr.Error) and err.message == "Загрузите видео" and err.title == app.ERROR_TITLE
    assert outs == [] and not wrote                       # nothing runs, nothing is journaled


def test_make_pdf_refusals():
    import gradio as gr
    with tempfile.TemporaryDirectory() as d:
        demo = _build(Path(d) / "jobs")
        make_pdf = _handlers(demo)["make_pdf"]
        # no analysis yet
        try:
            make_pdf("")
        except gr.Error as e:
            assert (e.message, e.title) == ("Сначала проанализируйте видео", app.ERROR_TITLE)
        else:
            raise AssertionError("no error without a job")
        # a folder without result.json: export_pdf fails, the dialog stays calm
        empty = Path(d) / "jobs" / "20000101_000000_deadbeef"
        empty.mkdir(parents=True)
        try:
            with _quiet():
                make_pdf(str(empty))
        except gr.Error as e:
            assert (e.message, e.title) == ("Не удалось собрать PDF. Подробности записаны в журнал сервера.",
                                            app.ERROR_TITLE)
        else:
            raise AssertionError("no error for a folder without result.json")


def test_status_html_states():
    """The progress line: the real percentage in the text with a 2 % minimum bar width; the other states fill the track
    in their colour; stopped and error carry no aria-valuenow; an unknown state renders as running."""
    start = app._status_html(0.004, "запуск")
    assert "width:2%" in start and "0% · запуск" in start and "aria-valuenow='0'" in start
    running = app._status_html(0.5, "идёт")
    assert "width:50%" in running and "50% · идёт" in running
    for state in ("done", "stopped", "error"):
        html = app._status_html(1.0, "текст", state=state)
        assert "width:100%" in html
        if state in ("stopped", "error"):
            assert "aria-valuenow" not in html, state
        else:
            assert "aria-valuenow='100'" in html, state
    # an unknown state is drawn exactly like running
    assert app._status_html(0.5, "x", state="zzz") == app._status_html(0.5, "x", state="running")


def test_preview_fills_the_page_and_shows_the_model():
    import gradio as gr
    from bs3.pipeline import Studio
    with tempfile.TemporaryDirectory() as d:
        job = _mm_job(Path(d))
        demo = app.build_app(Studio(), Path(d), preview_job=str(job))
        page_outs = page.page_outputs(jobfiles.load_job(job)[0])
    comps = list(demo.blocks.values())
    char = [c for c in comps if isinstance(c, gr.HTML) and getattr(c, "label", None) == "Характеристика личности"]
    assert len(char) == 1 and char[0].value == page_outs[3] and char[0].value       # the page arrives filled
    assert [c.value for c in comps if isinstance(c, gr.Radio)] == ["mm"]             # the model of the previewed job
    btn = next(c for c in comps if isinstance(c, gr.DownloadButton))
    assert btn.interactive is True
    # the job's server path is kept out of the page config (a preview binds to 0.0.0.0); the PDF button still reads the
    # folder because make_pdf falls back to the preview job
    assert [c.value for c in comps if isinstance(c, gr.State)] == [""]
    make_pdf = _handlers(demo)["make_pdf"]
    with _patched(app, pdf_for_download=lambda job_dir: str(job_dir)):
        assert make_pdf("") == str(job)


def test_stop_stops_only_own_session():
    """«Остановить» from another session leaves this analysis running; the caller's own stop ends it with exactly one
    ОСТАНОВЛЕНО line, and a stop with no own run just says «обработка отменена» (stage 21, FP1 / T7(3))."""
    from bs3.pipeline import Studio
    g = _Gate()
    with tempfile.TemporaryDirectory() as d, _patched(journal, PATH=Path(d) / "journal.txt"):
        studio = Studio()
        demo = app.build_app(studio, Path(d) / "jobs")
        analyze, stop = _handlers(demo)["analyze"], _handlers(demo)["stop"]
        with _patched(app, run_analysis=g.run):
            gA = analyze("/uploads/a.mp4", "mm", _req("sessA"))
            next(gA)                                  # A's worker acquires the lock and starts running
            assert g.started.acquire(timeout=10)      # A is now inside run_analysis (holds the lock)
            bar_b = stop(_req("sessB"))               # B has no run of its own: it must not touch A
            assert "обработка отменена" in bar_b and app.STATUS_LABELS["stopped"] in bar_b
            assert g.active == 1                       # A is still running
            bar_a = stop(_req("sessA"))               # A stops its own run
            assert "дорабатывается" in bar_a and "обработка отменена" not in bar_a
            gA.close()                                # the tab goes away; A's worker still finishes and journals
            assert studio.run_lock.acquire(timeout=10)  # wait until A's worker journaled ОСТАНОВЛЕНО and let go
            studio.run_lock.release()
        text = journal.PATH.read_text(encoding="utf-8")
    assert g.max == 1
    assert _kinds(text) == ["СТАРТ", "ОСТАНОВЛЕНО"]   # one start, one stop — only A; B journaled nothing
    assert "a.mp4" in text and "b.mp4" not in text


def test_cancelled_event_still_journals_and_never_overlaps():
    """Closing the tab of a stopped run does not lose its ОСТАНОВЛЕНО line (the worker writes it), and a run started
    afterwards waits on run_lock so at most one analysis is ever active — the journal reads СТАРТ / ОСТАНОВЛЕНО /
    СТАРТ / outcome (stage 21, FP1)."""
    from bs3.pipeline import Studio
    with tempfile.TemporaryDirectory() as d, _patched(journal, PATH=Path(d) / "journal.txt"):
        job = _mm_job(Path(d))
        g = _Gate(result_rep=jobfiles.load_job(job)[0])   # the second run finishes with a job that renders
        studio = Studio()
        demo = app.build_app(studio, Path(d) / "jobs")
        analyze, stop = _handlers(demo)["analyze"], _handlers(demo)["stop"]
        with _patched(app, run_analysis=g.run):
            g1 = analyze("/uploads/one.mp4", "mm", _req("s1"))
            next(g1)                                  # g1's worker runs under the lock
            assert g.started.acquire(timeout=10)
            stop(_req("s1"))                          # ask g1 to stop
            g1.close()                                # the tab closes: the generator ends, the worker keeps going
            assert studio.run_lock.acquire(timeout=10)  # wait until g1's worker journaled ОСТАНОВЛЕНО and released
            studio.run_lock.release()
            g.release.set()                           # let a non-stopped run finish
            with _quiet():
                outs2, err2 = _drive(analyze("/uploads/two.mp4", "mm", _req("s2")))
        text = journal.PATH.read_text(encoding="utf-8")
    assert g.max == 1                                 # run_lock kept the two runs from overlapping
    assert err2 is None, getattr(err2, "message", err2)
    assert _kinds(text) == ["СТАРТ", "ОСТАНОВЛЕНО", "СТАРТ", "РЕЗУЛЬТАТ"]
    assert "one.mp4" in text and "two.mp4" in text


def test_queued_notice():
    """The second «Анализировать» listener (queue=False, stage 21a / FP7): while another analysis holds studio.run_lock
    it returns the «В очереди» status line; with the lock free it returns a no-op update, so it can never overwrite a
    running analysis' progress."""
    import gradio as gr
    from bs3.pipeline import Studio
    with tempfile.TemporaryDirectory() as d:
        studio = Studio()
        demo = app.build_app(studio, Path(d) / "jobs")
        queued_notice = _handlers(demo)["queued_notice"]
        assert queued_notice() == gr.update()               # server free: no change
        studio.run_lock.acquire()
        try:
            busy = queued_notice()                          # server busy: the queue notice
        finally:
            studio.run_lock.release()
        assert queued_notice() == gr.update()               # released again: back to a no-op update
    assert isinstance(busy, str) and "В очереди" in busy
    assert busy == app._status_html(                        # the real helper and its real signature, exact wording
        0, "сервер занят обработкой другого ролика — анализ начнётся автоматически", label="В очереди")
