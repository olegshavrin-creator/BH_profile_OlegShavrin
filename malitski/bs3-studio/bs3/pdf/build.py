"""Planning and building the PDF report of BS Profiler 3.1 (design 10.7; 3.1: one model).

_plan decides which sections and appendices are printed and numbers them (pdf.plan, pdf.appx). _render lays the report
out top down — a passport (pdf/sections._passport), «Характеристика личности» (pdf/mbti_section.characterization_block),
the key facts, the Big Five profile, the MBTI section (pdf/mbti_section.mbti_section), one numbered section per topic
(pdf/sections.py), «Что повлияло …» with the key frames for AMLAI 1.0 (pdf/frames.py), «Как читать результаты» and the
appendices (pdf/appendix.py). build_pdf runs the layout twice (the second pass knows the page count) and writes the file
from the clean view (scores.clean_view). export_pdf opens a job folder (jobview.load_job, jobfiles), draws the print
charts (pdf/charts.py) and hands the file the media fallback on input.* before building it. The page itself (class
Report) is pdf/document.py, its widgets pdf/widgets.py, the text helpers pdf/fmt.py; the per-segment data come from
segments.py. bs3/pdf/__init__.py re-exports build_pdf and export_pdf.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from .. import PRODUCT, PRODUCT_SLUG, jobfiles, jobview, segments, settings
from ..facts import FACTS_LEGEND, fact_cards
from ..ru_texts import transcript_shown
from ..scores import has_explanations
from ..segments import segment_rows
from .appendix import _appendices
from .charts import save_modalities_chart
from .document import Report
from .frames import _explain_section
from .mbti_section import characterization_block, mbti_section
from .sections import (_emotions_section, _how_to_read, _no_explain_note, _passport, _profile_section,
                       _timeline_section, _voice_speech_section)


# ---------------------------------------------------------------- plan: which sections and appendices are printed
def _plan(pdf: Report, report: dict, explanation, frames, charts: dict, mb: dict | None = None) -> None:
    an = report.get("analyses") or {}
    te, fa = an.get("emotions_text") or {}, an.get("face") or {}
    has = {"profile": True,
           "mbti": bool(mb),             # right after the Big Five section, when at least the main type is computed
           "timeline": bool(charts.get("traits")) and len(segments.scored(report)) >= 2,
           "emotions": bool(te.get("mean") or fa.get("mean") or charts.get("emotions")),
           "voice_speech": bool(an.get("voice") or an.get("speech") or charts.get("voice") or charts.get("speech")),
           # explanations exist for AMLAI 1.0 only (3.1): an OCEAN-AI job gets one line under section 4 instead
           "explain": bool(explanation or frames) and has_explanations(report),
           "how_to_read": True}          # numbered like the rest: between the sections and the lettered appendices
    n = 0
    for key, ok in has.items():
        if ok:
            n += 1
            pdf.plan[key] = n
    note, transcript = transcript_shown(report)
    # the behaviour description is written for AMLAI 1.0 (its video-language model); an older OCEAN-AI job that
    # carries the description of the second model of 3.0 does not print it: one model in the report
    appx = {"file": True, "segments": len(segment_rows(report)) >= 2,
            "behavior": bool(report.get("behavior_description_ru")) and has_explanations(report),
            "transcript": bool(note or transcript)}
    letters = iter("АБВГДЕ")
    for key, ok in appx.items():
        if ok:
            pdf.appx[key] = next(letters)


# ---------------------------------------------------------------- the report
FACT_COLS = {7: 4, 8: 4}            # the type card makes 7 key facts: 4 + 3 cards in two rows instead of 3 + 3 + 1


def _render(report: dict, explanation, media, frames: list, charts: dict, fname: str, total: int | None,
            mb: dict | None, ch) -> Report:
    pdf = Report(file_label=fname, total_pages=total)
    _plan(pdf, report, explanation, frames, charts, mb)
    has_expl = "explain" in pdf.plan
    pdf.add_page()
    pdf.h1(f"{PRODUCT} — отчёт по видео: характеристика личности, Big Five, MBTI, эмоции, голос, речь")
    _passport(pdf, report, media, fname)
    if ch is not None:
        characterization_block(pdf, ch)
    # the key facts of the page (facts.fact_cards); «Речь в цифрах» of section 4 prints the pauses and the fillers
    speech_follows = "voice_speech" in pdf.plan and bool((report.get("analyses") or {}).get("speech"))
    facts, legend = fact_cards(report, mb, speech_cards_follow=speech_follows)
    if facts:
        cols = FACT_COLS.get(len(facts), 3)
        # the legend runs to two lines at this width, so the heading keeps the grid and both of them together
        pdf.h3("Ключевые факты", keep_mm=pdf.cards_height(facts, cols, value_first=True) + (8 if legend else 0))
        pdf.cards(facts, cols, value_first=True)
        if legend:
            pdf.caption(FACTS_LEGEND)
    if "timeline" not in pdf.plan:
        dur = float(report.get("duration_sec") or 0)
        pdf.para(("Ролик короче 30 с оценивается целиком" if 0 < dur <= settings.SINGLE_CLIP_MAX_SEC
                  else "Ролик оценён целиком, одним отрезком")
                 + ", поэтому графиков по ходу ролика нет.", 8)
    _profile_section(pdf, report, explanation, charts)
    mbti_section(pdf, report, mb)
    _timeline_section(pdf, report, charts, has_expl)
    _emotions_section(pdf, report, charts)
    _voice_speech_section(pdf, report, charts)
    _no_explain_note(pdf, report)
    _explain_section(pdf, report, explanation, frames, charts, media)
    _how_to_read(pdf, report)
    _appendices(pdf, report, media, has_expl, mb)
    return pdf


def build_pdf(report: dict, out_path: str | Path, explanation: dict | None = None, media: dict | None = None,
              key_frames: list[str] | None = None, mbti: dict | None = None, character=None) -> str:
    """The PDF of a clean view (scores.clean_view; a raw result.json is cleaned here). `mbti` — the section of
    mbti.get_mbti, `character` — characterization.build; both are computed here when the caller does not pass them
    (jobview.from_report: the view, its section and its characterization, each built once)."""
    if mbti is None and character is None:
        jv = jobview.from_report(report)
        report, mbti, character = jv.view, jv.mb, jv.character
    elif not report.get("view_meta"):
        from ..scores import clean_view
        report = clean_view(report)
    if character is None:                           # the characterization of the caller's own section
        from .. import characterization
        character = characterization.build(report, mbti)
    fname = report.get("original_file_name") or (media or {}).get("file_name") or Path(report.get("input", "")).name
    frames = [p for p in (key_frames or []) if os.path.exists(p)]
    charts = dict(report.get("chart_files") or {})
    if explanation and not charts.get("modalities") and charts:
        # save_pdf_charts called without the explanation (an older caller): the modality chart is drawn here
        try:
            p = save_modalities_chart(explanation, Path(next(iter(charts.values()))).parent)
            if p:
                charts["modalities"] = p
        except Exception:  # noqa: BLE001
            pass
    # the page count of the footer («стр. 3 из 7») comes from a first layout pass; the second one is written
    total = _render(report, explanation, media, frames, charts, fname, None, mbti, character).pages_count
    pdf = _render(report, explanation, media, frames, charts, fname, total, mbti, character)
    out_path = Path(out_path)
    pdf.output(str(out_path))
    return str(out_path)


def export_pdf(job_dir: str | Path) -> str:
    from .charts import save_pdf_charts
    from ..media import probe_media
    job = Path(job_dir)
    # the files of this folder, whatever paths result.json stores; jobs processed before the Russian texts are
    # translated once and stored back; the same clean numbers (design 6.1), the saved MBTI section or one computed now
    # and never written (design 7.2) and the same characterization as the page
    jv = jobview.load_job(job)
    view = jv.view
    view["chart_files"] = save_pdf_charts(view, job / jobfiles.CHARTS_DIR, jv.expl)
    frames = [str(p) for p in jobfiles.key_frame_paths(job, jv.rep)]
    media = view.get("media")
    if not media or "error" in media:
        inp = jobfiles.input_file(job)
        media = probe_media(inp) if inp else None
    stem = re.sub(r"[^A-Za-z0-9А-Яа-яЁё._-]+", "_", Path(view.get("original_file_name") or "video").stem)[:60]
    return build_pdf(view, job / f"{PRODUCT_SLUG}_report_{stem}.pdf", explanation=jv.expl, media=media,
                     key_frames=frames, mbti=jv.mb, character=jv.character)
