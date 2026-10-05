"""One finished job as the page, the PDF and the journal show it, built once (render level, on top of jobfiles).

A JobView holds what all three need: result.json with the Russian texts filled in (ru_texts), explanation.json, the
clean view (scores.clean_view), the MBTI section (mbti.get_mbti: the saved one, or computed now and never written)
and the characterization (characterization.build). Each of the three is built exactly once per JobView:

  load_job(job_dir)  a job folder on disk (pdf.export_pdf, the preview of build_app)
  for_page(rep)      a result in memory (web.page.page_outputs; the app hands the same JobView to the journal entry of
                     the analysis, journal.result(jv=…))
  from_report(rep)   a report in memory as it is: no file read or written and no Russian text added (the fallback of
                     pdf.build.build_pdf, journal.result_lines)

The files are found from the folder being opened (jobfiles), never from the paths result.json stores.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import characterization, jobfiles, mbti, ru_texts, scores


@dataclass
class JobView:
    job: Path | None                  # the job folder (`job_dir`); None for a report that names none
    rep: dict                         # result.json as loaded, with the Russian texts filled in
    expl: dict | None                 # explanation.json (AMLAI 1.0); None without one
    expl_path: Path | None            # where explanation.json lies in the job folder (it may not exist)
    view: dict                        # scores.clean_view(rep)
    mb: dict | None                   # mbti.get_mbti(rep, view); None without Big Five
    character: characterization.Character      # characterization.build(view, mb)


def _make(rep: dict, expl: dict | None, view: dict) -> JobView:
    job = Path(rep["job_dir"]) if rep.get("job_dir") else None
    mb = mbti.get_mbti(rep, view)
    return JobView(job=job, rep=rep, expl=expl, expl_path=jobfiles.explanation_path(job) if job else None,
                   view=view, mb=mb, character=characterization.build(view, mb))


def load_job(job_dir: str | Path, ensure_ru: bool = True) -> JobView:
    """The job folder `job_dir`: its result.json and explanation.json (jobfiles.load_job, `job_dir` pointed at this
    folder). ensure_ru: a job processed before the Russian texts existed gets them now, stored back into the folder
    (ru_texts.ensure_russian_job). Raises as jobfiles.load_job does."""
    job = Path(job_dir)
    rep, expl = jobfiles.load_job(job)
    if ensure_ru:
        ru_texts.ensure_russian_job(job, rep, expl)
    return _make(rep, expl, scores.clean_view(rep))


def for_page(rep: dict) -> JobView:
    """What the page shows for `rep`, a result in memory (a finished analysis, a script's copy of a job):
    explanation.json is read from the folder `rep["job_dir"]`, and the missing Russian texts are filled into `rep` and
    stored back into that folder; a `rep` without a folder gets no explanation and its Russian texts in memory only."""
    job = Path(rep["job_dir"]) if rep.get("job_dir") else None
    expl = jobfiles.read_json(jobfiles.explanation_path(job)) if job else None
    if job:
        ru_texts.ensure_russian_job(job, rep, expl)
    else:
        ru_texts.ensure_russian(rep, expl)
    return _make(rep, expl, scores.clean_view(rep))


def from_report(rep: dict, expl: dict | None = None) -> JobView:
    """The JobView of a report in memory as it is: a clean view (it has `view_meta`) is the view itself, a raw
    result.json is cleaned (scores.clean_view is idempotent, so both give the same view); no file is read or written
    and no Russian text is added."""
    return _make(rep, expl, rep if rep.get("view_meta") else scores.clean_view(rep))
