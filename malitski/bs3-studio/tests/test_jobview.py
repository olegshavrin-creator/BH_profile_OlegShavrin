"""One load of a job for the page, the PDF and the journal (bs3/jobview.py, refactoring plan of 3.1, stage 13).

The clean view, the MBTI section and the characterization are built once per JobView, whether the job is opened from
its folder (export_pdf, the preview of build_app), rendered from a result in memory (page_outputs; the app hands the
same JobView to the journal entry of the analysis) or given as a report (the fallback of build_pdf,
journal.result_lines). The words of the tab «Объяснения» are only read: webparts._words_text writes nothing.
scores.shown_model is the one choice of the model the page, the PDF and the journal name; the six expressions it
replaced are kept below and compared with it on a job of 2.0, of 3.0 and of both models of 3.1."""
from __future__ import annotations

import contextlib
import copy
import inspect
import re
import tempfile
from pathlib import Path

from samples import english, rep

import bs3
from bs3 import characterization, jobfiles, jobview, journal, mbti, ru_texts, scores
from bs3.web import app, charts, page, parts as webparts
from bs3.norms import TRAIT_KEYS
from bs3.pdf import appendix as pdf_appendix
from bs3.pdf import build as pdf_build
from bs3.pdf import charts as pdf_charts          # the print charts; `charts` is the web charts module
from bs3.pdf import frames as pdf_frames
from bs3.pdf import sections as pdf_sections

# explanation.json of an AMLAI 1.0 job: the modality shares and the readable words, already translated by Ollama, so
# opening the job translates nothing and writes nothing back
EXPL = {"modalities": {"input_x_gradient": {k: {"face": {"share": 0.6}, "audio": {"share": 0.38},
                                                "text": {"share": 0.01}, "behavior": {"share": 0.01}}
                                            for k in TRAIT_KEYS}},
        "readable_words": {"transcript_words": {k: {"up": [{"word": "work", "ru": "работа", "signed": 0.01}],
                                                    "down": []} for k in TRAIT_KEYS}},
        "readable_words_by": "ollama"}
BUILT_ONCE = {"clean_view": 1, "get_mbti": 1, "build": 1}
NOT_BUILT = {"clean_view": 0, "get_mbti": 0, "build": 0}


def _v30() -> dict:
    """Sample B as a job of 3.0: two members, OCEAN-AI the primary one, per segment the member that scored it."""
    r = rep("B")
    r["model"]["version"] = "3.0.0a1"
    for t in r["timeline"]:
        t["primary_used"] = "oceanai" if "oceanai" in t["members_used"] else None
        t["variants"] = {m: dict(t["scores"]) for m in t["members_used"]}
    return r


def _v31(member: str) -> dict:
    """Sample B as a job of 3.1 of one model (`member`): model.selected, one member, the segments carry its scores."""
    r = rep("B")
    r["model"].update({"version": "3.1.0a1", "selected": member, "primary": member, "backend": member,
                       "selected_title": bs3.MODEL_TITLES[member]})
    own = {k: r["variant_scores"][member][k] for k in TRAIT_KEYS}
    r["variant_scores"] = {member: r["variant_scores"][member]}
    for t in r["timeline"]:
        t.update({"members_used": [member], "primary_used": member, "variants": {member: dict(own)},
                  "scores": dict(own)})
    return r


# a job of each generation: an imported 2.0 job (sample B as it is: model.primary, no model.selected), a job of 3.0 and
# the two models of 3.1
JOBS = {"2.0": lambda: rep("B"), "3.0": _v30, "3.1 oceanai": lambda: _v31("oceanai"), "3.1 mm": lambda: _v31("mm")}


@contextlib.contextmanager
def _patched(module, **names):
    """Module globals replaced while the block runs, the old values back afterwards."""
    old = {k: getattr(module, k) for k in names}
    for k, v in names.items():
        setattr(module, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(module, k, v)


@contextlib.contextmanager
def _counted():
    """The calls of scores.clean_view, mbti.get_mbti and characterization.build while the block runs (each still
    does its work)."""
    calls = dict(NOT_BUILT)
    real = {"clean_view": (scores, scores.clean_view), "get_mbti": (mbti, mbti.get_mbti),
            "build": (characterization, characterization.build)}

    def counting(name, fn):
        def call(*a, **k):
            calls[name] += 1
            return fn(*a, **k)
        return call

    for name, (module, fn) in real.items():
        setattr(module, name, counting(name, fn))
    try:
        yield calls
    finally:
        for name, (module, fn) in real.items():
            setattr(module, name, fn)


def _job(base: Path, name: str, r: dict, expl: dict | None = None) -> Path:
    """A finished job folder; its result.json names the folder the job was made in, which is not this one."""
    job = base / name
    job.mkdir(parents=True)
    r["job_dir"] = f"/elsewhere/web_jobs/{name}"
    jobfiles.write_json(job / jobfiles.RESULT, r)
    if expl is not None:
        (job / jobfiles.EXPLAIN_DIR).mkdir()
        jobfiles.write_json(jobfiles.explanation_path(job), expl)
    return job


def _jobs(base: Path) -> dict[str, Path]:
    return {name: _job(base, f"2000010{i}_000000_0f3a9c1e", make(), EXPL if name == "3.1 mm" else None)
            for i, (name, make) in enumerate(JOBS.items(), 1)}


def _files(job: Path) -> dict:
    return {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in sorted(job.rglob("*")) if p.is_file()}


class _Req:
    headers = {"user-agent": "Mozilla/5.0 (Windows NT 10.0) Chrome/120"}
    session_hash = "abcdef123"


def _entry(r: dict, jv=None) -> str:
    """The journal entry of a finished analysis (journal.result), written into a temporary journal, time masked."""
    with tempfile.TemporaryDirectory() as d, _patched(journal, PATH=Path(d) / "journal.txt"):
        journal.result(_Req(), r, 65.0, jv=jv)
        text = journal.PATH.read_text(encoding="utf-8")
    return re.sub(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d", "<TIME>", text)


# ---------------------------------------------------------------- one load, each part built once
def test_load_job_builds_each_part_once():
    """jobview.load_job: the files of the folder being opened (jobfiles.load_job), the Russian texts, then the clean
    view, the MBTI section and the characterization, each built once; nothing to translate, so nothing is written."""
    with tempfile.TemporaryDirectory() as d:
        for name, job in _jobs(Path(d)).items():
            before = _files(job)
            with _counted() as calls:
                jv = jobview.load_job(job)
            assert calls == BUILT_ONCE, (name, calls)
            assert jv.job == job and jv.rep["job_dir"] == str(job), name
            assert jv.expl_path == jobfiles.explanation_path(job)
            assert jv.expl == (EXPL if name == "3.1 mm" else None), name
            view = scores.clean_view(jv.rep)
            assert jv.view == view and jv.mb == mbti.get_mbti(jv.rep, view) and jv.mb, name
            assert jv.character == characterization.build(view, jv.mb), name
            assert _files(job) == before, name


def test_load_job_fills_the_russian_texts_when_asked():
    """ensure_ru (the default) fills the missing Russian texts of the job and stores them in its folder
    (ru_texts.ensure_russian_job); ensure_ru=False reads the job as it is."""
    seen = []
    with tempfile.TemporaryDirectory() as d:
        job = _jobs(Path(d))["3.1 mm"]
        with _patched(ru_texts, ensure_russian_job=lambda j, r, e: seen.append((j, r["job_dir"], e))):
            jobview.load_job(job)
            jobview.load_job(str(job), ensure_ru=False)
    assert seen == [(job, str(job), EXPL)]


def test_the_app_builds_the_page_and_the_journal_entry_from_one_jobview():
    """A finished analysis in the app: the page (page_values) and the journal entry (journal.result(jv=…)) come from one
    JobView (jobview.for_page), so the view, the type and the characterization are built once for both, and they are
    the page and the entry that page_outputs and journal.result give on their own. With the JobView the journal builds
    nothing (no second get_mbti); without it it builds its own (the control)."""
    with tempfile.TemporaryDirectory() as d:
        for name, job in _jobs(Path(d)).items():
            r = jobfiles.load_job(job)[0]
            page_outs, entry = page.page_outputs(copy.deepcopy(r)), _entry(copy.deepcopy(r))
            with _counted() as calls:
                jv = jobview.for_page(r)
                outs = page.page_values(jv)
            assert calls == BUILT_ONCE, (name, calls)
            with _counted() as calls:
                assert _entry(r, jv) == entry, name
            assert calls == NOT_BUILT, (name, calls)
            with _counted() as calls:
                assert _entry(r) == entry, name
            assert calls == BUILT_ONCE, (name, calls)
            assert outs == page_outs, name
            assert jv.rep is r and outs[20] == job.name and outs[21] == str(job)
            assert "Характеристика (коротко): " in entry and f"    Папка: {job}\n" in entry


def test_for_page_fills_the_russian_texts_where_the_job_lies():
    """jobview.for_page: a result with `job_dir` gets its explanation from that folder and its missing Russian texts
    stored there (ru_texts.ensure_russian_job); a result without one gets no explanation and no folder, and its Russian
    texts in memory only (ru_texts.ensure_russian)."""
    seen = []
    with _patched(ru_texts, ensure_russian_job=lambda j, r, e: seen.append(("job", j, e)),
                  ensure_russian=lambda r, e: seen.append(("memory", e)) or (False, False)):
        with tempfile.TemporaryDirectory() as d:
            job = _jobs(Path(d))["3.1 mm"]
            r = jobfiles.load_job(job)[0]
            jv = jobview.for_page(r)
        assert seen == [("job", job, EXPL)] and jv.job == job and jv.expl == EXPL
        seen.clear()
        r = _v31("mm")
        jv = jobview.for_page(r)
    assert seen == [("memory", None)] and jv.job is None and jv.expl is None and jv.expl_path is None
    assert jv.view == scores.clean_view(r)


def test_export_pdf_opens_the_job_once():
    """export_pdf (jobview.load_job): the PDF gets the type and the characterization the load built, so build_pdf
    builds neither again; the charts and the PDF are drawn from the same clean view."""
    from bs3 import media
    got = {}

    def build(view, out, **kw):
        got.update(kw, view=view)
        return str(out)

    def draw(view, out_dir, expl):
        got.update(chart_view=view, expl=expl)
        return {}

    with tempfile.TemporaryDirectory() as d:
        job = _jobs(Path(d))["3.1 mm"]
        with _patched(pdf_build, build_pdf=build), _patched(pdf_charts, save_pdf_charts=draw), \
                _patched(media, probe_media=lambda p: {"probed": Path(p).name}), _counted() as calls:
            pdf_build.export_pdf(job)
        view = scores.clean_view(jobfiles.load_job(job)[0])
    assert calls == BUILT_ONCE, calls
    assert got["chart_view"] is got["view"] and got["expl"] == EXPL == got["explanation"]
    assert got["mbti"] == mbti.get_mbti(view, view) and got["character"] == characterization.build(view, got["mbti"])


def test_build_pdf_builds_only_what_the_caller_does_not_pass():
    """The fallback of build_pdf (jobview.from_report): a raw result.json is cleaned once, the type and the
    characterization are built once when the caller passes neither, the characterization alone when the caller passes
    its own type, nothing when it passes the characterization."""
    drawn = []

    class _Pdf:
        pages_count = 1

        def output(self, path):
            Path(path).write_bytes(b"%PDF-1.4\n")

    def render(report, explanation, media, frames, charts_, fname, total, mb, character):
        drawn.append((report, mb, character))
        return _Pdf()

    raw = _v31("mm")
    view = scores.clean_view(raw)
    mb = mbti.get_mbti(raw, view)
    ch = characterization.build(view, mb)
    with tempfile.TemporaryDirectory() as d, _patched(pdf_build, _render=render):
        out = Path(d) / "r.pdf"
        for case, given, kw, want in (("raw", raw, {}, BUILT_ONCE),
                                      ("view", view, {}, {"clean_view": 0, "get_mbti": 1, "build": 1}),
                                      ("view + type", view, {"mbti": mb}, {"clean_view": 0, "get_mbti": 0, "build": 1}),
                                      ("raw + characterization", raw, {"character": ch},
                                       {"clean_view": 1, "get_mbti": 0, "build": 0})):
            drawn.clear()
            with _counted() as calls:
                assert pdf_build.build_pdf(copy.deepcopy(given), out, **kw) == str(out)
            assert calls == want, (case, calls)
            assert len(drawn) == 2 and drawn[0] == drawn[1], case         # the two layout passes
            report, got_mb, got_ch = drawn[1]
            assert report == view, case
            assert got_mb == (None if case == "raw + characterization" else mb), case
            assert got_ch == ch, case


def test_the_preview_opens_the_job_once():
    """build_app(preview_job=…) (scripts/ui_preview.py): the page arrives filled from one load of the folder
    (jobview.load_job): explanation.json is read once and each part built once; the page shows what page_outputs gives
    for the job, the radio the model of the job, and the PDF button reads the folder."""
    import gradio as gr
    from bs3.pipeline import Studio
    reads = []
    read_json = jobfiles.read_json

    def reading(path):
        reads.append(Path(path))
        return read_json(path)

    with tempfile.TemporaryDirectory() as d:
        job = _jobs(Path(d))["3.1 mm"]
        with _patched(jobfiles, read_json=reading), _counted() as calls:
            demo = app.build_app(Studio(), Path(d), preview_job=str(job))
        assert reads.count(jobfiles.explanation_path(job)) == 1, reads
        assert calls == BUILT_ONCE, calls
        page_outs = page.page_outputs(jobfiles.load_job(job)[0])
    comps = list(demo.blocks.values())
    char = [c for c in comps if isinstance(c, gr.HTML) and getattr(c, "label", None) == "Характеристика личности"]
    assert len(char) == 1 and char[0].value == page_outs[3]
    assert [c.value for c in comps if isinstance(c, gr.Radio)] == ["mm"]
    # the job's server path is kept out of the page config (a preview binds to 0.0.0.0); the PDF button still reads the
    # folder because make_pdf falls back to the preview job
    assert [c.value for c in comps if isinstance(c, gr.State)] == [""]
    mk = next(f.fn for f in demo.fns.values() if getattr(getattr(f, "fn", None), "__name__", "") == "make_pdf")
    with _patched(app, pdf_for_download=lambda job_dir: str(job_dir)):
        assert mk("") == str(job)


# ---------------------------------------------------------------- the words of «Объяснения» are only read
def test_words_text_only_reads():
    """_words_text takes the explanation alone and shows the lists it keeps (narrative.words_summary): it makes no list
    and writes no file. The lists of an older job are made and stored before, with its other Russian texts
    (ru_texts.ensure_russian_job in jobview)."""
    from bs3.narrative import words_summary
    assert list(inspect.signature(webparts._words_text).parameters) == ["expl"]
    made = []
    with tempfile.TemporaryDirectory() as d:
        # lists that ensure_words would make again: the attributions are there, the stored words have no translation
        stale = copy.deepcopy(EXPL)
        stale["transcript_words"] = {}
        for d_ in stale["readable_words"]["transcript_words"].values():
            d_["up"][0].pop("ru")
        stale["readable_words"]["transcript_words"]["openness"]["up"].append(
            {"word": "team", "ru": "команда", "signed": 0.02})
        job = _job(Path(d), "20000105_000000_0f3a9c1e", _v31("mm"), stale)
        before = _files(job)
        expl = jobfiles.read_json(jobfiles.explanation_path(job))
        with _patched(ru_texts, ensure_words=lambda *a: made.append(a) or True, write_json=lambda *a: made.append(a)):
            text = webparts._words_text(expl)
        assert _files(job) == before
    assert made == [] and expl == stale
    assert text == "\n\n".join(words_summary(stale["readable_words"], stale, webparts.TRAIT_TITLES))
    assert "«команда»" in text and "работа" not in text


# ---------------------------------------------------------------- the model a job is shown as
def _main(r: dict):
    return (r.get("view_meta") or {}).get("main_system")


# the six places that chose the model before (stage 13), as they were written; scores.shown_model replaces them
OLD = {
    "page._frames_html": lambda r: _main(r) or (r.get("model") or {}).get("selected"),
    "page.page_outputs": lambda r: _main(r),
    "charts.fig_radar": lambda r: _main(r) or (r.get("model") or {}).get("selected"),
    "pdf_charts._radar_chart": lambda r: _main(r) or (r.get("model") or {}).get("selected"),
    "pdf_report._main_model": lambda r: _main(r) or scores.recorded_model(r),
    "journal.result_lines": lambda r: (_main(r) or (r.get("model") or {}).get("selected")
                                       or (r.get("model") or {}).get("primary")),
}


def test_shown_model_is_the_model_each_place_chose():
    """On the clean view, which is what the page, the PDF and the journal are built from, shown_model is the model
    each of the six old expressions chose: for a job of 2.0, 3.0 and both models of 3.1, and for an English 2.0 job
    that names no primary model. On a raw result.json it is the recorded model (scores.recorded_model), as the PDF chose
    it and the journal did. Three old expressions read only model.selected, which 3.1 added, and gave None for a raw
    2.0 or 3.0 job; the page's gave None for any raw one. None of them was given a raw result.json: page_outputs and
    export_pdf hand them the clean view."""
    shown = {"2.0": ("oceanai", "oceanai"), "3.0": ("oceanai", "oceanai"), "3.1 oceanai": ("oceanai", "oceanai"),
             "3.1 mm": ("mm", "mm"), "2.0 English": ("oceanai", None)}
    for name, make in {**JOBS, "2.0 English": english}.items():
        raw = make()
        view = scores.clean_view(raw)
        assert (scores.shown_model(view), scores.shown_model(raw)) == shown[name], name
        for place, old in OLD.items():
            assert old(view) == scores.shown_model(view), (name, place)
        assert scores.shown_model(raw) == scores.recorded_model(raw) == OLD["pdf_report._main_model"](raw) \
            == OLD["journal.result_lines"](raw), name
        selected = raw["model"].get("selected")
        for place in ("page._frames_html", "charts.fig_radar", "pdf_charts._radar_chart"):
            assert OLD[place](raw) == selected, (name, place)
            assert selected in (None, scores.shown_model(raw)), (name, place)
        assert OLD["page.page_outputs"](raw) is None
        assert scores.has_explanations(view) is (shown[name][0] == "mm"), name
        assert scores.has_explanations(raw) is (shown[name][1] == "mm"), name


def test_the_places_ask_shown_model():
    """The page, the charts, the PDF and the journal take the model from scores (shown_model, has_explanations); no
    module keeps an expression of its own, and pdf_report._main_model is gone."""
    assert charts.shown_model is pdf_charts.shown_model is pdf_appendix.shown_model is scores.shown_model
    assert pdf_sections.shown_model is scores.shown_model
    assert page.has_explanations is pdf_build.has_explanations is pdf_sections.has_explanations \
        is scores.has_explanations
    assert not hasattr(pdf_build, "_main_model") and not hasattr(pdf_appendix, "_main_model")
    src = inspect.getsource
    assert "model_title(shown_model(rep))" in src(charts.fig_radar)
    assert "model_title(shown_model(rep))" in src(pdf_charts._radar_chart)
    assert "if not has_explanations(rep):" in src(page._frames_html)
    assert "own = has_explanations(view)" in src(page.page_values)
    assert "model_title(shown_model(view))" in src(journal)
    # has_explanations: _plan twice (pdf/build.py), _no_explain_note (pdf/sections.py); shown_model: _analysis_rows and
    # _segments_table (pdf/appendix.py), _passport (pdf/sections.py)
    parts = [src(m) for m in (pdf_build, pdf_appendix, pdf_sections, pdf_frames)]
    assert [s.count("has_explanations(report)") for s in parts] == [2, 0, 1, 0]
    assert [s.count("shown_model(report)") for s in parts] == [0, 2, 1, 0]
    for p in sorted(Path(bs3.__file__).parent.rglob("*.py")):
        text = p.read_text(encoding="utf-8")
        assert not re.search(r"""get\("main_system"\)\s*or\b""", text) or p.name == "scores.py", p
        assert not re.search(r"""get\("selected"\)\s*or\s*\(?\w+\.get\("primary"\)""", text) or p.name == "scores.py", p
