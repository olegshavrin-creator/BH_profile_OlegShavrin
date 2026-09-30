"""«Что повлияло на оценку модели AMLAI 1.0» in the PDF report of BS Profiler 3.1: the key frames in rows (portrait
frames up to five in a row, landscape or square ones three) with the captions of the page under each frame
(frame_captions) and the note under them, the chart of the modality contributions and the words the model responded
to. The explanations exist for AMLAI 1.0 only; an OCEAN-AI job gets one line instead (pdf/sections._no_explain_note).
"""
from __future__ import annotations

from .. import frame_captions
from ..norms import RU_TITLES
from ..segments import representative
from .document import Report
from .fmt import _seg


def _frame_rows(pdf: Report, frames: list):
    """Rows of key frames: portrait frames all in one row (5 × 34.9 mm), landscape or square ones 3 per row."""
    from PIL import Image
    sizes = []
    for p in frames:
        with Image.open(p) as im:
            sizes.append(im.size)
    portrait = all(h > w for w, h in sizes)
    per_row = min(len(frames), 5) if portrait else 3
    gap, max_h = 4.0, 62.0
    cell_w = (pdf.epw - gap * (per_row - 1)) / per_row
    rows = []
    for i in range(0, len(frames), per_row):
        row = []
        for p, (iw, ih) in zip(frames[i:i + per_row], sizes[i:i + per_row]):
            scale = min(cell_w / iw, max_h / ih)
            row.append((p, iw * scale, ih * scale))
        rows.append(row)
    return rows, cell_w, gap


# caption under a key frame: 0.8 mm of air, two lines of 7 pt (3 mm each) and 1.7 mm before the next row
FRAME_CAP_SIZE = 7
FRAME_CAP_SIZE_NARROW = 6          # five frames in a row (34.9 mm): 7 pt leaves one word of the phrase
FRAME_CAP_NARROW = 40.0            # mm, below which the caption drops to FRAME_CAP_SIZE_NARROW
FRAME_CAP_LINE = 3.0
FRAME_CAP_H = 8.5


def _clip_words(pdf: Report, text: str, width: float, size: float) -> str:
    """`text` shortened by whole words until it fits `width`, with «…» where it was cut; "" when even the first
    word plus the ellipsis is too wide."""
    if not text:
        return ""
    pdf.set_font("ui", "", size)
    if pdf.get_string_width(text) <= width:
        return text
    words = text.split(" ")
    while len(words) > 1:
        words.pop()
        s = " ".join(words).rstrip(" ·,") + "…"
        if s != "…" and pdf.get_string_width(s) <= width:
            return s
    return ""


def _frame_second(pdf: Report, entry: dict | None, width: float) -> str:
    """The second line this cell really gets: the first candidate of frame_captions.pdf_second_line that fits its
    width, or "" when none of them does. The note under the frames is written from what this returns, so it never
    announces an expression the narrow cell had to drop."""
    if not entry:
        return ""
    pdf.set_font("ui", "", FRAME_CAP_SIZE if width >= FRAME_CAP_NARROW else FRAME_CAP_SIZE_NARROW)
    return next((c for c in frame_captions.pdf_second_line(entry) if pdf.get_string_width(c) <= width), "")


def _frame_caption(pdf: Report, entry: dict | None, x: float, y: float, width: float) -> None:
    """Two centred lines under one key frame: «2:14 · улыбается, смотрит в камеру» and «радость 62% · повысил
    экстраверсию». The second line falls back to its shorter forms («повысил экстраверсию», «повысил эм. стаб.»,
    «радость 62%») and is dropped when none of them fits; the first line, which carries the moment, is cut by
    whole words with «…» instead. Five frames in a row leave only 34.9 mm, where 7 pt keeps one word of the
    phrase: such a cell gets 6 pt, which buys about a sixth more characters on both lines."""
    if not entry:
        return
    size = FRAME_CAP_SIZE if width >= FRAME_CAP_NARROW else FRAME_CAP_SIZE_NARROW
    pdf.set_font("ui", "", size)
    first = _clip_words(pdf, entry.get("caption") or "", width, size) or (entry.get("label") or "")
    pdf.set_xy(x, y)
    pdf.cell(width, FRAME_CAP_LINE, first, align="C")
    second = _frame_second(pdf, entry, width)
    if second:
        pdf.set_xy(x, y + FRAME_CAP_LINE)
        pdf.cell(width, FRAME_CAP_LINE, second, align="C")


def _explain_section(pdf: Report, report: dict, explanation, frames: list, charts: dict, media: dict | None) -> None:
    """5. What drove the score of AMLAI 1.0: key frames, modality contributions, words."""
    if "explain" not in pdf.plan:
        return
    seg = representative(report, min_scored=2)
    tl_all = report.get("timeline") or []
    if seg and "timeline" in pdf.plan:
        intro = (f"Объяснения построены для модели AMLAI 1.0 по отрезку {_seg(report, seg['start'], seg['end'])}, "
                 "ближайшему к среднему профилю (★ на графике «Big Five по ходу ролика»).")
    else:
        intro = "Объяснения построены для модели AMLAI 1.0 по всему ролику."
    rows, cell_w, gap = _frame_rows(pdf, frames) if frames else ([], 0.0, 0.0)
    first_h = (max(h for _, _, h in rows[0]) + FRAME_CAP_H + 6) if rows else 20
    pdf.section("Что повлияло на оценку модели AMLAI 1.0", "explain", keep_mm=pdf.para_height(intro, 8.5) + first_h)
    pdf.para(intro, 8.5)
    if rows:
        # frames come from the clip the explanations were computed on: the representative segment of a long video,
        # otherwise the whole video; file names carry the frame index inside that clip (key_<i>_frame<N>.jpg).
        # The captions are the ones of the page (frame_captions): the moment and, in a few words, what is visible;
        # the second line names the expression and what the frame did to the score. The cell is narrow (34.9 mm
        # with five frames in a row), so there the caption is set in 6 pt, the first line is clipped by whole
        # words and the second falls back to shorter forms — the last of them, «повысил эм. стаб.», still carries
        # the direction — and is dropped only when even that does not fit.
        entries = {e["path"]: e for e in frame_captions.build(report, frames, explanation, media)}
        tenths = frame_captions.has_tenths(report, frames, media, explanation)
        # a job made before the captions has neither a phrase nor the expressions: the note promises only the
        # parts that are actually printed, and it names the second line, which the PDF cannot show on hover
        what = frame_captions.note_what(entries.values(), tenths, "pdf")
        # what the second line ended up carrying in this layout, not what the captions could have offered
        chosen = [(e, _frame_second(pdf, e, cell_w)) for e in entries.values()]
        has_expr = any(s and e["expr_line"] and s.startswith(e["expr_line"]) for e, s in chosen)
        has_eff = any(s and s != e["expr_line"] for e, s in chosen)
        second = [x for x, ok in (("выражение лица", has_expr), ("то, как кадр сдвинул оценку", has_eff)) if ok]
        note = ("Кадры, сильнее всего повлиявшие на оценку модели AMLAI 1.0. Рамкой отмечено найденное лицо; "
                f"под кадром — {what}"
                + (f"; во второй строке — {' и '.join(second)}." if second else "."))
        pdf.h3("Ключевые кадры", keep_mm=max(h for _, _, h in rows[0]) + FRAME_CAP_H)
        per_row = len(rows[0])
        for row in rows:
            row_h = max(h for _, _, h in row)
            if pdf.get_y() + row_h + FRAME_CAP_H > pdf.page_break_trigger:
                pdf.add_page()
            y0 = pdf.get_y()
            # a last row shorter than the others is centred: an empty cell at the right edge reads as a missing frame
            x_row = pdf.l_margin + (per_row - len(row)) * (cell_w + gap) / 2
            for i, (q, iw, ih) in enumerate(row):
                x = x_row + i * (cell_w + gap) + (cell_w - iw) / 2
                try:
                    pdf.image(q, x=x, y=y0 + (row_h - ih), w=iw, h=ih)
                except Exception:  # noqa: BLE001
                    continue
                _frame_caption(pdf, entries.get(q), x_row + i * (cell_w + gap), y0 + row_h + 0.8, cell_w)
            pdf.set_xy(pdf.l_margin, y0 + row_h + FRAME_CAP_H)
        pdf.caption(note, 8)
    if charts.get("modalities"):
        cap = ("Какая доля оценки модели AMLAI 1.0 пришлась на каждую модальность (по градиенту оценки: насколько "
               "признаки модальности сдвигают результат); в каждой строке — 100%. «<1%» — модальность почти не влияет "
               "на оценку этого ролика: модель, обученная на First Impressions V2, опирается в основном на лицо и голос.")
        pdf.chart_block(charts["modalities"], cap)
    rw_all = (explanation or {}).get("readable_words") or {}
    if rw_all:
        try:
            from ..narrative import words_summary
            paras = words_summary(rw_all, explanation, RU_TITLES)
        except Exception as e:  # noqa: BLE001  (never let the words block break the whole PDF)
            paras = [f"Список слов недоступен: {str(e)[:120]}"]
        if paras:
            pdf.ln(1)
            # the heading needs its first lines beside it, not the whole block: these paragraphs may split, and
            # holding them together would leave the page before the appendix nearly empty
            pdf.h3("Слова, на которые откликнулась модель", keep_mm=min(14, sum(pdf.para_height(p, 8) for p in paras)))
            for para in paras:
                pdf.para(para, 8)
    # lists without Russian words (translation failed) are not printed: the raw English tokens stay in the JSON
