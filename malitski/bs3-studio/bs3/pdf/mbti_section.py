"""PDF parts of BS Profiler 3.1 (design 10.7, 5.7; task T21; 3.1: one model).

- `characterization_block(pdf, ch)` — «Характеристика личности» on page 1: a header line (letters of the type 20 pt bold,
  the borderline ones grey with a dashed underline; the name 12 pt bold; the labels 8.5 pt grey joined by « · »), then
  the paragraphs at 9 pt with bold leads (fpdf2 markdown); a paragraph that does not fit on the rest of the page starts
  the next one, so no paragraph is split between pages. «Границы вывода» (only caveats) is set at 8.5 pt.
- `mbti_section(pdf, view, mb)` — section «Тип MBTI (перевод шкал Big Five)»: the type line of the model that ran
  («OCEAN-AI, веса MuPTA» or «AMLAI 1.0»; one model since 3.1), the axis table with neuroticism, «Тип по ходу ролика»
  (letter strip, summary, C8, C19) and the caveats C3, C4, C5, C6, C7, C9, C16 at 7.5 pt.
- `letter_strip(pdf, lanes)` — the letter strip drawn with cells: letters are text, not a picture; bold — clear,
  normal — moderate, grey in a dashed frame — on the border (the letter of the strict split), «—» on light grey — no
  score. Long videos are wrapped into blocks of at most 30 segments.

No new colours: text black or NOTE_GREY (#555555, 7.46:1 on white), frames grey 130 (as #808080 on the page), the
«no score» fill a light grey — all of it stays readable when printed in greyscale.
"""
from __future__ import annotations

import math

from .. import caveats
from ..labels import mbti_model_title
from ..mbti import AXES, AXIS_LABEL, TABLE_NOTE, TABLE_ROWS, corr_cell, load_config, source_title, summary_line
from ..textfmt import clock
from .document import NOTE_GREY, TEXT_W_MM, Report

FRAME_GREY = 130                 # dashed frames and rules, as the #808080 outline of the page
NO_SCORE_FILL = 232              # light grey behind «—» (no score in the segment)
STRIP_LABEL_MM = 18              # column of the axis labels «E–I» …
STRIP_MAX_CELL_MM = 6.0
STRIP_ROW_MM = 4.8
STRIP_BLOCK = 30                 # at most this many segments in one row of the strip
STRIP_LETTER_PT = 6.5
HEADER_H = 10.0                  # height of the header line of the characterization
STRIP_LEGEND = ("Жирная буква — ось выражена отчётливо, обычная — умеренно; серая буква в пунктирной рамке — ось на "
                "границе (показана буква строгого деления); «—» на сером — нет оценки. Над столбцами — начало отрезка "
                "(мин:с).")


def _dashed_underline(pdf: Report, x0: float, x1: float, y: float) -> None:
    pdf.set_draw_color(FRAME_GREY); pdf.set_line_width(0.3); pdf.set_dash_pattern(dash=0.8, gap=0.6)
    pdf.line(x0, y, x1, y)
    pdf.set_dash_pattern(); pdf.set_draw_color(0); pdf.set_line_width(0.2)


# ------------------------------------------------------------------------------------------ characterization ---

def _char_lh(size: float) -> float:
    """Leading of the characterization: 1.3 of the font size (4.1 mm at 9 pt) — the text is the longest on page 1,
    and the 4.5 mm of the other paragraphs of the report made it half a page longer."""
    return round(size * 0.3528 * 1.27, 2)


def _md_height(pdf: Report, text: str, size: float) -> float:
    pdf.set_font("ui", "", size)
    lines = pdf.multi_cell(pdf.epw, _char_lh(size), text, markdown=True, dry_run=True, output="LINES")
    return len(lines) * _char_lh(size)


def _header_line(pdf: Report, hdr: dict) -> None:
    """One line: letters of the type, the name, the labels (wrapped under the letters when they do not fit beside)."""
    y0 = pdf.get_y()
    c_margin = pdf.c_margin
    pdf.c_margin = 0                                   # letters are placed by their own widths
    x = pdf.l_margin
    letters = hdr.get("letters") or []
    name = hdr.get("name")
    if letters:
        pdf.set_font("ui", "B", 20)
        base = y0 + HEADER_H / 2 + 0.3 * pdf.font_size  # baseline of a cell of height HEADER_H
        for ch, border in letters:
            w = pdf.get_string_width(ch)
            pdf.set_text_color(NOTE_GREY if border else 0)
            pdf.set_xy(x, y0)
            pdf.cell(w, HEADER_H, ch)
            if border:
                _dashed_underline(pdf, x, x + w, base + 1.2)
            x += w + 0.6
        pdf.set_text_color(0)
        x += 3.4
    if name:
        pdf.set_font("ui", "B", 12)
        t = f"«{name}»"
        pdf.set_xy(x, y0)
        pdf.cell(pdf.get_string_width(t), HEADER_H, t)
        x += pdf.get_string_width(t) + 4
    bottom = y0 + (HEADER_H if (letters or name) else 0)
    labels = " · ".join(hdr.get("labels") or [])
    if labels:
        pdf.set_font("ui", "", 8.5)
        pdf.set_text_color(NOTE_GREY)
        room = pdf.l_margin + pdf.epw - x
        lines = pdf.multi_cell(room, 4, labels, dry_run=True, output="LINES") if room >= 40 else []
        if (letters or name) and lines and len(lines) <= 2:
            pdf.set_xy(x, y0 + (HEADER_H - 4 * len(lines)) / 2 + 0.6)
            pdf.multi_cell(room, 4, labels, align="L", new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.set_xy(pdf.l_margin, bottom + 0.5)
            pdf.multi_cell(0, 4, labels, align="L", new_x="LMARGIN", new_y="NEXT")
        bottom = max(bottom, pdf.get_y())
        pdf.set_text_color(0)
    pdf.c_margin = c_margin
    pdf.set_draw_color(FRAME_GREY); pdf.set_line_width(0.2)
    pdf.line(pdf.l_margin, bottom + 1.2, pdf.l_margin + pdf.epw, bottom + 1.2)
    pdf.set_draw_color(0)
    pdf.set_xy(pdf.l_margin, bottom + 3.2)


def characterization_block(pdf: Report, ch) -> None:
    """«Характеристика личности» (design 10.7): unnumbered heading, the header line, the paragraphs."""
    paras = ch.pdf_paragraphs()
    first = _md_height(pdf, paras[0]["markdown"], 9) if paras else 0.0
    pdf.h3("Характеристика личности", keep_mm=HEADER_H + 4 + first, size=10.5)
    _header_line(pdf, ch.pdf_header())
    for p in paras:
        size = 8.5 if p["key"] == "limits" else 9
        h = _md_height(pdf, p["markdown"], size)
        if pdf.get_y() + h > pdf.page_break_trigger and h < pdf.page_break_trigger - pdf.t_margin:
            pdf.add_page()                             # a paragraph is never split between pages
        pdf.set_x(pdf.l_margin)
        pdf.set_font("ui", "", size)
        pdf.multi_cell(0, _char_lh(size), p["markdown"], markdown=True, align="L", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(0.9)


# ---------------------------------------------------------------------------------------------- MBTI section ---

def _type_line(pdf: Report, title: str, item: dict, with_alternatives: bool) -> None:
    """«OCEAN-AI, веса MuPTA: ESFJ «Попечитель», с учётом границ EXFJ, возможен ENFJ» (10 pt, letters 14 pt)."""
    lh = 6.5
    x_count = int(item.get("x_count") or 0)
    axes = item.get("axes") or {}
    pdf.set_x(pdf.l_margin)
    pdf.set_font("ui", "", 10)
    pdf.write(lh, title + ": ")
    if x_count <= 2:
        word = item.get("type_strict") or ""
        letters = [(ch, bool((axes.get(ax) or {}).get("borderline"))) for ch, ax in zip(word, AXES)]
    else:
        letters = [(ch, False) for ch in item.get("type") or ""]
    pdf.set_font("ui", "B", 14)
    y = pdf.get_y()
    base = y + lh / 2 + 0.3 * pdf.font_size
    for ch, border in letters:
        x0 = pdf.get_x()
        pdf.set_text_color(NOTE_GREY if border else 0)
        pdf.write(lh, ch)
        if border:
            _dashed_underline(pdf, x0, pdf.get_x(), base + 0.9)
    pdf.set_text_color(0)
    pdf.set_font("ui", "", 10)
    name = item.get("type_name") if x_count <= 2 else None
    rest = f" «{name}»" if name else ""
    if x_count == 0:
        rest += ", пограничных осей нет"
    elif x_count <= 2:
        rest += f", с учётом границ {item.get('type')}"
        alts = item.get("alternatives") or []
        if with_alternatives and len(alts) == 1:
            rest += f", возможен {alts[0]}"
    else:
        rest += f" — тип не выражен: {x_count} оси из 4 на границе"
    pdf.write(lh, rest)
    pdf.ln(lh)


def _axis_cell(a: dict | None) -> str:
    if not a or a.get("missing"):
        return "нет данных"
    if a.get("borderline"):
        return f"X (на границе, ближе к {a.get('letter')})"
    s = f"{a.get('letter')}, {a.get('word')}"
    return s + f" ({float(a.get('confidence') or 0):.2f})"


def _axis_table(pdf: Report, mb: dict, cfg: dict) -> None:
    """The axis table of the model that ran: letter, word and confidence per axis, neuroticism, the correspondence of
    the scales (no agreement column: one model since 3.1)."""
    corr = cfg.get("correspondence") or {}
    header = ["Ось", "Шкала Big Five", source_title(mb), "Соответствие шкал"]
    rows = []
    for ax, scale, _direction in TABLE_ROWS:
        c = corr.get(ax) or {}
        rows.append([AXIS_LABEL[ax], scale, _axis_cell((mb.get("axes") or {}).get(ax)),
                     corr_cell(c) if c.get("r") is not None else "—"])
    rows.append(["—", "Нейротизм", (mb.get("neuroticism") or {}).get("level") or "—", "в MBTI не выражается"])
    # column widths from the content (cells do not wrap); one step smaller while the table is wider than the page
    c_margin = pdf.c_margin
    pdf.c_margin = 1.0
    size = 8.5
    for size in (8.5, 8.0, 7.5, 7.0):
        widths = []
        for j, h in enumerate(header):
            pdf.set_font("ui", "B", size)
            w = pdf.get_string_width(h)
            pdf.set_font("ui", "", size)
            w = max([w] + [pdf.get_string_width(str(r[j])) for r in rows])
            widths.append(w + 2 * pdf.c_margin + 1.0)
        if sum(widths) <= pdf.epw:
            break
    pdf.table(header, rows, widths, size=size, zebra=True, first_left=True, row_h=5.4, min_rows=len(rows))
    pdf.c_margin = c_margin
    pdf.caption("Число в скобках после буквы — уверенность по оси (0 — на границе 0.5, 1 — у края шкалы), а не сама "
                "оценка. Нейротизм = 1 − эмоциональная стабильность. Соответствие шкал: " + TABLE_NOTE[0].lower()
                + TABLE_NOTE[1:], 7.5)


def _lane_rows(pdf: Report, title: str, chunk: list[dict], cell: float, row_h: float) -> None:
    """One lane of the strip: its title, the start times above every third column, four rows of letters (a cell is
    `cell` mm wide and `row_h` mm high)."""
    x0 = pdf.l_margin
    pdf.set_x(x0)
    pdf.set_font("ui", "B", 7.5)
    pdf.cell(0, 4, title, new_x="LMARGIN", new_y="NEXT")
    y = pdf.get_y()
    pdf.set_font("ui", "", 6)
    pdf.set_text_color(NOTE_GREY)
    for j in range(0, len(chunk), 3):
        pdf.set_xy(x0 + STRIP_LABEL_MM + j * cell, y)
        pdf.cell(3 * cell, 3, clock(chunk[j].get("start")), align="L")
    pdf.set_text_color(0)
    y += 3.3
    inset = 0.35
    for i, ax in enumerate(AXES):
        pdf.set_xy(x0, y)
        pdf.set_font("ui", "B", 7.5)
        pdf.cell(STRIP_LABEL_MM, row_h, AXIS_LABEL[ax])
        for j, e in enumerate(chunk):
            x = x0 + STRIP_LABEL_MM + j * cell
            ts = e.get("type_strict")
            word = (e.get("words") or {}).get(ax)
            letter = ts[i] if ts else None
            if not letter or letter == "X" or word in (None, "нет данных"):
                pdf.set_fill_color(NO_SCORE_FILL)
                pdf.rect(x + inset, y + inset, cell - 2 * inset, row_h - 2 * inset, style="F")
                pdf.set_font("ui", "", STRIP_LETTER_PT)
                pdf.set_text_color(NOTE_GREY)
                pdf.set_xy(x, y)
                pdf.cell(cell, row_h, "—", align="C")
                pdf.set_text_color(0)
                continue
            if word == "на границе":
                pdf.set_draw_color(FRAME_GREY); pdf.set_line_width(0.2); pdf.set_dash_pattern(dash=0.6, gap=0.6)
                pdf.rect(x + inset, y + inset, cell - 2 * inset, row_h - 2 * inset)
                pdf.set_dash_pattern(); pdf.set_draw_color(0)
                pdf.set_text_color(NOTE_GREY)
                pdf.set_font("ui", "", STRIP_LETTER_PT)
            else:
                pdf.set_font("ui", "B" if word == "отчётливо" else "", STRIP_LETTER_PT)
            pdf.set_xy(x, y)
            pdf.cell(cell, row_h, letter, align="C")
            pdf.set_text_color(0)
        y += row_h
    pdf.set_xy(x0, y + 1.5)


def _strip_geometry(n: int) -> tuple[int, int, float, float]:
    """(blocks, segments per block, cell width, row height in mm): blocks of at most STRIP_BLOCK segments, split
    evenly (33 segments -> 17 + 16, not 30 + 3); a row is at most STRIP_ROW_MM high (6.5 pt letters need no more)."""
    blocks = max(1, math.ceil(n / STRIP_BLOCK))
    per = max(1, math.ceil(n / blocks))
    cell = min(STRIP_MAX_CELL_MM, (TEXT_W_MM - STRIP_LABEL_MM) / per)
    return blocks, per, cell, min(cell, STRIP_ROW_MM)


def _lane_height(row_h: float) -> float:
    return 4 + 3.3 + 4 * row_h + 1.5


def letter_strip(pdf: Report, lanes: list[tuple[str, list[dict]]]) -> None:
    """lanes: [(title, timeline entries of mbti.segment_types)] — one lane since 3.1 (the model that ran)."""
    n = max((len(e) for _, e in lanes), default=0)
    if not n:
        return
    blocks, per, cell, row_h = _strip_geometry(n)
    for b in range(blocks):
        for title, entries in lanes:
            chunk = entries[b * per:(b + 1) * per]
            if not chunk:
                continue
            if pdf.get_y() + _lane_height(row_h) > pdf.page_break_trigger:
                pdf.add_page()
            t = title
            if blocks > 1:
                first, last = b * per + 1, b * per + len(chunk)
                t += f" · отрезки {first}–{last} из {n}"
            _lane_rows(pdf, t, chunk, cell, row_h)


def _strip_lanes(mb: dict) -> tuple[list, list[str]]:
    """(lanes, summary lines) of the strip: the one model that ran (3.1), titled as its panel on the page."""
    lanes = [(mbti_model_title(mb), mb.get("timeline") or [])]
    return lanes, [x for x in [summary_line(mb)] if x]


def mbti_section(pdf: Report, view: dict, mb: dict | None) -> None:
    """Section «Тип MBTI (перевод шкал Big Five)» right after the Big Five section (design 10.7, item 2): one model,
    one type line, the axis table without an agreement column, one strip, the caveats caveats.MBTI_READ (3.1)."""
    if "mbti" not in pdf.plan or not mb:
        return
    cfg = load_config()
    pdf.section("Тип MBTI (перевод шкал Big Five)", "mbti", keep_mm=52)
    _type_line(pdf, mbti_model_title(mb), mb, with_alternatives=True)
    if mb.get("source") == "own_model" and mb.get("primary_missing"):
        pdf.caption(caveats.text("C20"), 7.5)
    pdf.ln(1.5)
    _axis_table(pdf, mb, cfg)
    # «Тип по ходу ролика»
    total = int(mb.get("segments_total") or len(mb.get("timeline") or []) or 1)
    typed = sum(1 for e in (mb.get("timeline") or []) if e.get("type_strict"))
    if total <= 1:
        pdf.h3("Тип по ходу ролика", keep_mm=8)
        pdf.caption(caveats.text("C19"), 7.5)
    else:
        lanes, lines = _strip_lanes(mb)
        first_h = _lane_height(_strip_geometry(max(len(e) for _, e in lanes))[3])
        pdf.h3("Тип по ходу ролика", keep_mm=first_h)
        if typed >= 2:
            letter_strip(pdf, lanes)
            pdf.caption(STRIP_LEGEND, 7)
        if lines:
            pdf.para(" ".join(lines), 9)
        pdf.caption(caveats.text("C8"), 7.5)
    # how to read the type
    texts = [caveats.text(c) for c in caveats.MBTI_READ]
    pdf.h3("Как читать тип MBTI", keep_mm=min(40, sum(pdf.para_height(t, 7.5) for t in texts[:2])))
    for t in texts:
        pdf.caption(t, 7.5)


def segment_types_by_start(mb: dict | None) -> dict:
    """{rounded start of a segment: type with X or None} of the model that ran (appendix «Значения по отрезкам»)."""
    out = {}
    for e in (mb or {}).get("timeline") or []:
        try:
            out[int(round(float(e.get("start"))))] = e.get("type")
        except (TypeError, ValueError):
            continue
    return out
