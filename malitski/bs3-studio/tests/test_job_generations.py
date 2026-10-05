"""The page and the PDF of BS Profiler 3.1 on every generation of job they must still read (tests/jobs.py): an imported
2.0 job, a 3.0 job, a 3.1 job of OCEAN-AI and of AMLAI 1.0 (with the key frames in the old and in the new format) and a
short single clip with no timeline. Cross-cutting cover of web/page.py, the PDF package, journal and the charts (task
T6, T11(3)); no GPU, no model — the fixtures already carry the Russian texts, so nothing is translated on render.
"""
from __future__ import annotations

import contextlib
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import jobs

from bs3 import jobfiles
from bs3.narrative import NO_EXPLAIN_RU
from bs3.pdf import build as pdf_build
from bs3.web import page

OCEANAI, AMLAI = "OCEAN-AI, веса MuPTA", "AMLAI 1.0"
MODEL_TITLE = {"imported20": OCEANAI, "v30": OCEANAI, "oceanai31": OCEANAI, "short_clip": OCEANAI,
               "mm31_old_frames": AMLAI, "mm31_captions": AMLAI}
AMLAI_KINDS = ("mm31_old_frames", "mm31_captions")
# the PDF page count, pinned per generation (the OCEAN-AI report has no section «Что повлияло …»; the AMLAI 1.0 one adds
# the key frames), with one page of slack for small layout shifts
PDF_PAGES = {"imported20": (5, 7), "v30": (5, 7), "oceanai31": (5, 7), "short_clip": (5, 7),
             "mm31_old_frames": (6, 8), "mm31_captions": (6, 8)}


def _text(path: Path) -> str:
    if not shutil.which("pdftotext"):
        raise unittest.SkipTest("pdftotext (poppler-utils) not installed")
    r = subprocess.run(["pdftotext", "-enc", "UTF-8", str(path), "-"], capture_output=True, text=True, check=True)
    return re.sub(r"\s+", " ", r.stdout)


def _pages(text: str) -> int:
    m = re.search(r"стр\. \d+ из (\d+)", text)
    assert m, "the PDF text has no «стр. N из M» footer"
    return int(m.group(1))


@contextlib.contextmanager
def _no_translation():
    """Every public function of bs3.translate raises while the block runs: a render that reaches one is a bug, because
    the fixtures already carry the Russian texts. The originals (the model guard of tests/run.py) are put back after."""
    from bs3 import translate

    def boom(*a, **k):
        raise AssertionError("no translation may run when the Russian texts are already stored")

    names = ("translate_description", "translate_words", "translate_sentences", "translate_text", "llm_translate",
             "valid_translation", "fix_marian_ru", "ollama_available")
    old = {n: getattr(translate, n) for n in names if hasattr(translate, n)}
    for n in old:
        setattr(translate, n, boom)
    try:
        yield
    finally:
        for n, v in old.items():
            setattr(translate, n, v)


def test_page_outputs_every_generation():
    """page_outputs gives the N_PAGE values for every generation: the recorded model is shown, the segments table has
    one row per per_segment entry (with «нет речи» for the segment whose transcript is empty), and the speech and face
    blocks name the tempo and the dominant expression. Key frames belong to AMLAI 1.0; an OCEAN-AI job shows a note."""
    with tempfile.TemporaryDirectory() as d:
        for kind in jobs.KINDS:
            job = jobs.make_job(kind, Path(d))
            rep, _ = jobfiles.load_job(job)
            outs = page.page_outputs(rep)
            assert len(outs) == page.N_PAGE == 27, kind
            # the recorded model is shown in «Модель и время обработки»
            assert outs[page.page_index("model_and_time")].startswith(f"Модель {MODEL_TITLE[kind]}"), \
                (kind, outs[page.page_index("model_and_time")][:60])
            # the segments table: one row per per_segment entry (three) plus the sticky header row
            table = outs[page.page_index("segments_table")]
            assert table and table.count("<tr") == 4, (kind, table.count("<tr"))
            assert "нет речи" in table, kind                 # the segment with 0 words / an empty transcript
            # the speech block names the tempo, the face block the dominant expression
            assert "Темп речи" in outs[page.page_index("speech_cards")], kind
            assert "Выражение лица чаще всего" in outs[page.page_index("face_cards")], kind
            frames = outs[page.page_index("key_frames_html")]
            if kind in AMLAI_KINDS:
                assert "data:image/jpeg;base64," in frames, kind
            else:
                assert page.NO_FRAMES_OCEANAI in frames, kind


def test_export_pdf_every_generation():
    """export_pdf builds the report in the job folder, with the print charts, for every generation. The page count is
    in the range pinned for the kind; the text names the model. An OCEAN-AI job has the one-line no-explanations note
    and no section «Что повлияло …»; an AMLAI 1.0 job has that section with the key frames and the word list."""
    with tempfile.TemporaryDirectory() as d:
        built = {}
        for kind in jobs.KINDS:
            job = jobs.make_job(kind, Path(d))
            pdf = Path(pdf_build.export_pdf(job))
            assert pdf.exists() and pdf.parent == job and pdf.stat().st_size > 10_000, kind
            assert pdf.name.startswith("BS_Profiler_3_report_") and pdf.suffix == ".pdf", kind
            assert list((job / jobfiles.CHARTS_DIR).glob("*.png")), kind
            built[kind] = pdf
        for kind, pdf in built.items():
            text = _text(pdf)                                # skips the text checks without pdftotext
            lo, hi = PDF_PAGES[kind]
            assert lo <= _pages(text) <= hi, (kind, _pages(text))
            assert f"Модель {MODEL_TITLE[kind]}" in text, kind
            if kind in AMLAI_KINDS:
                assert "Что повлияло на оценку модели AMLAI 1.0" in text, kind
                assert "Ключевые кадры" in text and "под кадром —" in text, kind        # the caption lines
                assert "Слова, на которые откликнулась модель" in text, kind
                assert NO_EXPLAIN_RU[:50] not in text, kind
            else:
                assert NO_EXPLAIN_RU[:50] in text and "Что повлияло" not in text, kind


def test_render_writes_nothing_back_when_texts_exist():
    """The fixtures carry the Russian texts, so the page and the PDF translate nothing and store nothing back: with
    every bs3.translate function stubbed to raise, result.json is byte-for-byte the same, with the same mtime, after
    page_outputs and export_pdf."""
    with tempfile.TemporaryDirectory() as d, _no_translation():
        for kind in jobs.KINDS:
            job = jobs.make_job(kind, Path(d))
            result = job / jobfiles.RESULT
            before, before_mtime = result.read_bytes(), os.stat(result).st_mtime_ns
            rep, _ = jobfiles.load_job(job)
            page.page_outputs(rep)
            pdf_build.export_pdf(job)
            assert result.read_bytes() == before, kind
            assert os.stat(result).st_mtime_ns == before_mtime, kind


def test_short_clip_without_timeline_renders():
    """A short single clip has no timeline at all: the page still gives its N_PAGE values (the segments table from the
    per_segment rows) and the PDF still builds, with the note that a short clip is scored whole and has no timeline
    charts."""
    with tempfile.TemporaryDirectory() as d:
        job = jobs.make_job("short_clip", Path(d))
        rep, _ = jobfiles.load_job(job)
        assert rep.get("timeline") == []
        outs = page.page_outputs(rep)
        assert len(outs) == page.N_PAGE
        assert outs[page.page_index("segments_table")].count("<tr") == 4          # per_segment rows without a timeline
        pdf = Path(pdf_build.export_pdf(job))
        assert pdf.exists() and pdf.stat().st_size > 10_000
        text = _text(pdf)
        assert "графиков по ходу ролика нет" in text
