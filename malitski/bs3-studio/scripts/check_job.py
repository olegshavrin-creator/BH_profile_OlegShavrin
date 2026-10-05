"""Check finished jobs of BS Profiler 3.1 without a new analysis (change request 3.1, section 6: «result.json of a
real run has one member only», the page and the PDF of one model).

    ~/bs/venv/bin/python bs3-studio/scripts/check_job.py ~/bs3_data/web_jobs/<id> [<id> ...]

For every job folder (it must hold result.json):
1. result.json of a 3.1 job names ONE model: model.selected in (oceanai, mm), model.primary and model.selected_title
   agree with it, variant_scores holds that member only, traits are its scores, the speech is Russian without
   percentiles, the mbti section is schema 3 with `model` = that member and no `second` / `agreement`. A job without
   model.selected (3.0 or imported 2.0) skips this item: it is only required to open (items 2-3).
2. The page builds (web.page.page_outputs, 27 values); none of its blocks says «второе мнение», names or roles of 3.0
   (rerender_samples.OLD_NAMES), «MM-PSYCHE» outside the recipe clause, «Краткие выводы» or «сегмент» (the
   transcript, the behaviour description and the raw result.json are the person's data and the file itself and are
   not read); «Модель и время обработки» starts with the model line; the MBTI panel names the model; the tab
   «Объяснения» of an OCEAN-AI job carries the one note (narrative.NO_EXPLAIN_RU), «Ключевые кадры» its own note and
   the words and description boxes are empty; for AMLAI 1.0 the note is absent and, when explanation.json exists, the
   modality table and the words are there, and the key frames are shown when the job has them.
3. The PDF builds (bs3.pdf.export_pdf) and its text (pdftotext) has the passport line «Модель …», the headings of
   rerender_samples.PDF_MUST and nothing of PDF_MUST_NOT; an OCEAN-AI job has the one-line note and no section «Что
   повлияло …», an AMLAI 1.0 job with explanation.json has that section and no note.
Exit code 0 when every check of every job passed. Nothing in the job is changed except what the page itself writes
(Russian texts of older jobs, the PDF file and the chart PNGs for it). No job ids are kept in this file.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rerender_samples import OLD_NAMES, PDF_MUST, PDF_MUST_NOT, SECOND_OPINION, Checks, pdf_text  # noqa: E402

from bs3 import MODALITIES, MODEL_TITLES, jobfiles  # noqa: E402
from bs3.labels import model_title  # noqa: E402
from bs3.narrative import NO_EXPLAIN_RU  # noqa: E402
from bs3.norms import TRAIT_KEYS  # noqa: E402
from bs3.pdf import export_pdf  # noqa: E402
from bs3.web.page import N_PAGE, NO_FRAMES_OCEANAI, page_index, page_outputs  # noqa: E402

MMP = re.compile(r"(?<!по рецепту )MM-PSYCHE")
# indices of page_outputs (web.page.page_outputs): the blocks that hold the person's own words or the file itself
I_TRANSCRIPT, I_FRAMES, I_CONTRIB, I_WORDS, I_DESC, I_MODEL, I_JSON, I_PATH, I_JOB, I_MBTI = (
    page_index("transcript"), page_index("key_frames_html"), page_index("contrib"), page_index("words"),
    page_index("behavior_description"), page_index("model_and_time"), page_index("result_json"),
    page_index("saved_to"), page_index("job_state"), page_index("mbti_types"))
SKIP_TEXT = {I_TRANSCRIPT, I_DESC, I_JSON, I_PATH, I_JOB}


def _text(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<style>.*?</style>", " ", re.sub(r"<[^>]+>", " ", html), flags=re.S)).strip()


def check_result(c: Checks, rep: dict) -> str | None:
    """Item 1. Returns the selected member of a 3.1 job, or None for an older job (only required to open)."""
    m = rep.get("model") or {}
    sel = m.get("selected")
    if sel is None:
        print("  (no model.selected: a job of an earlier version, the page and the PDF only)")
        return None
    c.ok(sel in MODEL_TITLES, f"model.selected is a known model ({sel!r})")
    if sel not in MODEL_TITLES:
        return None
    c.ok(m.get("primary") == sel and m.get("selected_title") == MODEL_TITLES[sel], "model.primary / selected_title agree")
    c.ok(m.get("lang") == "ru", "speech language ru")
    var = rep.get("variant_scores") or {}
    c.ok(set(var) == {sel}, f"variant_scores holds the one member ({sorted(var)})")
    traits = rep.get("traits") or {}
    # report.build_report rounds the traits to 4 decimals; variant_scores keep the raw means
    c.ok(all(abs(float(traits[k]["score"]) - float(var.get(sel, {}).get(k, -9))) < 5e-5 for k in TRAIT_KEYS if k in traits),
         "traits are the scores of that member")
    c.ok(not any("percentile" in (traits.get(k) or {}) for k in TRAIT_KEYS), "no percentiles (Russian speech)")
    c.ok(set(rep.get("modalities_used") or []) == set(MODALITIES[sel]), "modalities_used names what the model looked at")
    c.ok("narrative" not in rep, "no stored 2.0 summary (nothing for the tab «Данные» to leave out)")
    for t in rep.get("timeline") or []:
        if isinstance(t.get("variants"), dict) and set(t["variants"]) - {sel}:
            c.ok(False, f"segment {t.get('segment')} carries another member")
            break
    mb = rep.get("mbti") or {}
    c.ok(mb.get("schema_version") == 3 and mb.get("model") == sel and "second" not in mb and "agreement" not in mb,
         "mbti section of schema 3 for that member, no second opinion")
    c.ok(("interview" in rep) == (sel == "mm"), "interview score for AMLAI 1.0 only")
    c.ok(not any(k in (rep.get("interview") or {}) for k in ("percentile", "percentile_vs_fiv2", "percentile_ref")),
         "no percentile on the interview label (Russian speech)")
    return sel


def check_page(c: Checks, rep: dict, sel: str | None) -> str | None:
    """Item 2. Returns the model the page shows (from the model line)."""
    outs = page_outputs(rep)
    c.ok(len(outs) == N_PAGE, f"page_outputs gives {N_PAGE} values ({len(outs)})")
    texts = {i: _text(o) for i, o in enumerate(outs) if isinstance(o, str) and i not in SKIP_TEXT}
    for what, rx in (("«второе мнение»", SECOND_OPINION), ("3.0 names and roles", OLD_NAMES),
                     ("«MM-PSYCHE» outside the recipe clause", MMP), ("«Краткие выводы»", re.compile("Краткие выводы")),
                     ("«сегмент»", re.compile("сегмент", re.I))):
        hit = next(((i, t, rx.search(t)) for i, t in texts.items() if rx.search(t)), None)
        c.ok(hit is None, f"page block {hit[0]}: {what}: …{hit[1][max(0, hit[2].start() - 50):hit[2].end() + 30]}…"
             if hit else f"page: {what}")
    shown = None
    for key, title in MODEL_TITLES.items():
        if texts.get(I_MODEL, "").startswith(f"Модель {model_title(key)}"):
            shown = key
    c.ok(shown is not None, f"«Модель и время обработки» starts with the model line ({texts.get(I_MODEL, '')[:60]!r})")
    if sel is not None:
        c.ok(shown == sel, f"the page shows the selected model ({shown} vs {sel})")
    if shown:
        c.ok(model_title(shown) in texts.get(I_MBTI, ""), "the MBTI panel names the model")
    if shown == "mm":
        c.ok(NO_EXPLAIN_RU not in texts.get(I_CONTRIB, ""), "no OCEAN-AI note in the tab «Объяснения» of AMLAI 1.0")
        job = Path(rep["job_dir"])
        if jobfiles.explanation_path(job).exists():
            c.ok("Вклад" in texts.get(I_CONTRIB, "") or "модальност" in texts.get(I_CONTRIB, ""), "modality table shown")
            c.ok(bool(texts.get(I_WORDS)), "words block filled")
        if jobfiles.key_frame_paths(job, rep):
            c.ok("<img" in outs[I_FRAMES], "key frames shown")
    elif shown == "oceanai":
        c.ok(texts.get(I_CONTRIB, "") == NO_EXPLAIN_RU, "tab «Объяснения»: the one note and nothing else")
        c.ok(texts.get(I_FRAMES, "") == NO_FRAMES_OCEANAI, "«Ключевые кадры»: the one note")
        c.ok(not texts.get(I_WORDS) and not (outs[I_DESC] or "").strip(), "words and description boxes empty")
        c.ok(not any("собеседовани" in t.lower() for i, t in texts.items() if i != I_MODEL),
             "no label «собеседование» of the other model on the page")
    return shown


def check_pdf(c: Checks, job: Path, shown: str | None) -> None:
    """Item 3."""
    pdf = Path(export_pdf(job))
    c.ok(pdf.exists(), "PDF built")
    if not pdf.exists():
        return
    text = pdf_text(pdf)
    for must in PDF_MUST:
        c.ok(must in text, f"PDF has «{must}»")
    body = text.split("Транскрипт речи")[0]          # the transcript is the person's own speech
    for what, rx in PDF_MUST_NOT:
        m = rx.search(body)
        c.ok(m is None, f"PDF: {what}: …{body[max(0, (m.start() if m else 0) - 50):(m.end() if m else 0) + 30]}…" if m else "")
    if shown:
        c.ok(f"Модель {model_title(shown)}" in text, "PDF passport names the model")
    note, sec5 = "не строит объяснений" in text, "Что повлияло на оценку модели AMLAI 1.0" in text
    if shown == "oceanai":
        c.ok(note and not sec5, "PDF of OCEAN-AI: the one-line note, no section «Что повлияло …»")
        c.ok("собеседовани" not in body.lower(), "PDF of OCEAN-AI: no label «собеседование» (and no C2 about it)")
    elif shown == "mm":
        c.ok(not note, "PDF of AMLAI 1.0: no OCEAN-AI note")
        if jobfiles.explanation_path(job).exists():
            c.ok(sec5, "PDF of AMLAI 1.0: section «Что повлияло …»")
    print(f"  PDF {pdf.name}")


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if argv else 2
    bad = 0
    for arg in argv:
        job = Path(arg).expanduser().resolve()
        c = Checks(job.name)
        print(f"== {job.name}")
        if not (job / jobfiles.RESULT).exists():
            print("  FAIL no result.json")
            bad += 1
            continue
        rep, _ = jobfiles.load_job(job)
        sel = check_result(c, rep)
        shown = check_page(c, rep, sel)
        check_pdf(c, job, shown)
        print(f"  {c.n - len(c.failed)} of {c.n} checks passed" + (f", {len(c.failed)} FAILED" if c.failed else ""))
        bad += bool(c.failed)
    print("OK" if not bad else f"{bad} job(s) with failures")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
