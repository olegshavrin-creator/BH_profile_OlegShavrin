"""The two drawn widgets of the PDF page, as mixins of class Report (pdf/document.py): the card grid of «Ключевые
факты» and «Речь в цифрах» (CardsMixin) and the score bars of «Оценки по чертам» with their legend and note
(ScoreBarsMixin). They draw with the methods of fpdf.FPDF, which Report brings in after them.
"""
from __future__ import annotations

from ..facts import card_item, fact_label
from ..norms import RU_TITLES, TRAIT_KEYS
from ..palette import CARD_PDF, FACT_VALUE_PDF, SCORE_BAR_PDF, TRAIT_BAR_PDF
from ..textfmt import fiv2_ref_ru, pct_phrase
from .fmt import _group_name
from .layout import NOTE_GREY


def _rgb(hexc: str) -> tuple:
    return tuple(int(hexc.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))


class CardsMixin:
    """The card grid of «Ключевые факты» and «Речь в цифрах»."""

    def _card_layout(self, items, cols: int, gap: float, value_first: bool = False):
        w = (self.epw - gap * (cols - 1)) / cols
        pad, inner = 1.8, (self.epw - gap * (cols - 1)) / cols - 3.6
        heights = []
        for lab, val, note, state in (card_item(x) for x in items):
            self.set_font("ui", "", 7.5)
            # the label carries the state word in the same cases as in cards(), or the card would be measured short
            n_lab = len(self.multi_cell(inner, 3.4, fact_label(lab, state) if value_first else str(lab),
                                        dry_run=True, output="LINES"))
            self.set_font("ui", "B", 11)
            n_val = len(self.multi_cell(inner, 5.0, str(val if val not in (None, "") else "—"), dry_run=True, output="LINES"))
            n_note = 0
            if note:
                self.set_font("ui", "", 7.5)
                n_note = len(self.multi_cell(inner, 3.4, str(note), dry_run=True, output="LINES"))
            heights.append(pad + n_lab * 3.4 + 0.6 + n_val * 5.0 + (0.3 + n_note * 3.4 if n_note else 0) + pad - 0.4)
        rows = [list(range(i, min(i + cols, len(items)))) for i in range(0, len(items), cols)]
        row_h = [max(heights[i] for i in r) for r in rows]
        return w, pad, inner, rows, row_h

    def cards_height(self, items, cols: int = 3, gap: float = 4.0, value_first: bool = False) -> float:
        if not items:
            return 0.0
        _, _, _, _, row_h = self._card_layout(items, cols, gap, value_first)
        return sum(row_h) + gap * 0.75 * (len(row_h) - 1) + 1.5

    def cards(self, items, cols: int = 3, gap: float = 4.0, value_first: bool = False):
        """(label, value, note[, state]) -> a grid of outlined cards, as on the web page: label and note small and
        grey, the value large and bold. `value_first`: the card reads value, then label, then note; a value with a
        state is printed in the colour of that state (palette.FACT_VALUE_PDF) and its label ends with the word of
        that state (facts.fact_label), so the card also reads on a black-and-white printer — «Ключевые факты»
        of 3.1. The grid is never split between pages."""
        if not items:
            return
        w, pad, inner, rows, row_h = self._card_layout(items, cols, gap, value_first)
        if self.get_y() + self.cards_height(items, cols, gap, value_first) > self.page_break_trigger:
            self.add_page()
        y = self.get_y()
        for r, rh in zip(rows, row_h):
            for j, i in enumerate(r):
                lab, val, note, state = card_item(items[i])
                x = self.l_margin + j * (w + gap)
                self.set_fill_color(CARD_PDF["fill"]); self.set_draw_color(CARD_PDF["outline"]); self.set_line_width(0.2)
                self.rect(x, y, w, rh, style="DF", round_corners=True, corner_radius=1.2)
                self.set_xy(x + pad, y + pad)

                def small(text, lead=0.3):
                    if lead:
                        self.set_y(self.get_y() + lead); self.set_x(x + pad)
                    self.set_font("ui", "", 7.5); self.set_text_color(NOTE_GREY)
                    self.multi_cell(inner, 3.4, str(text), align="L", new_x="LEFT", new_y="NEXT")

                def big(lead=0.6):
                    if lead:
                        self.set_y(self.get_y() + lead); self.set_x(x + pad)
                    self.set_font("ui", "B", 11)
                    ink = _rgb(FACT_VALUE_PDF[state]) if (value_first and state) else (0, 0, 0)
                    self.set_text_color(*ink)
                    self.multi_cell(inner, 5.0, str(val if val not in (None, "") else "—"), align="L",
                                    new_x="LEFT", new_y="NEXT")
                # the same two gaps either way (0.6 + 0.3), so cards_height does not depend on the order
                if value_first:
                    big(lead=0)
                    small(fact_label(lab, state), lead=0.6)
                else:
                    small(lab, lead=0)
                    big()
                if note:
                    small(note)
            y += rh + gap * 0.75
        self.set_text_color(0); self.set_draw_color(0)
        self.set_xy(self.l_margin, y - gap * 0.75 + 1.5)


class ScoreBarsMixin:
    """The score bars of the one model, their legend and the note under them."""
    BAR_W, BAR_H = 52, 3.6

    def _bar_rows(self, traits: dict, interview: dict | None):
        return [(k, traits[k]) for k in TRAIT_KEYS] + ([("interview", interview)] if interview else [])

    def _bars_legend(self):
        """Legend items (kind, text) under the score bars."""
        # «как на графиках» with one reservation: the extraversion fill is a step darker than its line on the charts,
        # because the chart colour gives only 2.9:1 on the light track of the bar
        return [("traits", "оценка черты 0…1 (цвет — как на графиках, экстраверсия чуть темнее)"),
                ("tick", "середина шкалы")]

    def _legend_lines(self, items) -> list:
        """Legend items wrapped into lines of the text width: [[(kind, text, x_offset), …], …]."""
        self.set_font("ui", "", 8)
        lines, cur, x = [], [], 0.0
        for kind, text in items:
            w = 8 + self.get_string_width(text) + 1
            if cur and x + w > self.epw:
                lines.append(cur); cur, x = [], 0.0
            cur.append((kind, text, x)); x += w + 3
        if cur:
            lines.append(cur)
        return lines

    def _bars_note(self, traits: dict, interview: dict | None) -> str:
        groups: dict[str, list[str]] = {}          # FIV2 reference in words -> item keys (Russian speech: none)
        for k, t in self._bar_rows(traits, interview):
            pct = t.get("percentile", t.get("percentile_vs_fiv2"))
            ref = t.get("percentile_ref", "train First Impressions V2 (6000 клипов)" if pct is not None else "")
            if pct is not None and pct_phrase(pct, ref)[0]:
                groups.setdefault(f"относительно {fiv2_ref_ru(ref)}", []).append(k)
        parts = ["Длина полоски — оценка модели от 0 до 1; уровни черт и буквы MBTI считаются по этой же шкале, "
                 "середина — 0.5."]
        interview_named = False
        if groups:
            if len(groups) == 1:
                parts.append(f"Рядом с полоской — процентиль {next(iter(groups))}.")
            else:
                parts += [(f"«Собеседование» (коричневая полоска, модель AMLAI 1.0) — процентиль {ref}."
                           if keys == ["interview"] else f"{_group_name(keys).capitalize()} — процентиль {ref}.")
                          for ref, keys in groups.items()]
                interview_named = any(keys == ["interview"] for keys in groups.values())
        if interview and not interview_named:
            parts.append("Коричневая полоска — впечатление «собеседование» (метка модели AMLAI 1.0, шкала 0…1).")
        return " ".join(parts)

    def score_bars_height(self, traits: dict, interview: dict | None) -> float:
        n = len(TRAIT_KEYS)
        h = n * 6.0 + (8.0 if interview else 0.0) + 4.2
        h += len(self._legend_lines(self._bars_legend())) * 4.4 + 1
        self.set_font("ui", "", 8)
        h += len(self.multi_cell(self.epw, 4.0, self._bars_note(traits, interview), dry_run=True, output="LINES")) * 4.0
        return h + 1

    def score_bars(self, traits: dict, interview: dict | None):
        """One row per trait: the bar is the score itself (0…1, what a reader expects to see filled), the text gives the
        score and, for a FIV2 percentile (an older English job), its position in words. One model (3.1): no second
        opinion under the bars."""
        items = self._bar_rows(traits, interview)
        bar_w, bar_h = self.BAR_W, self.BAR_H
        c = SCORE_BAR_PDF
        # the label column is as wide as the longest title; the phrase after the bar must end at the right margin
        self.set_font("ui", "", 8.5)
        label_w = max(self.get_string_width(RU_TITLES[k]) for k, _ in items) + 3
        x = self.l_margin + label_w
        row_h = 6.0
        for k, t in items:
            pct = t.get("percentile", t.get("percentile_vs_fiv2")); score = float(t["score"])
            ref = t.get("percentile_ref", "train First Impressions V2 (6000 клипов)" if pct is not None else "")
            if k == "interview":        # a separate label of the own model: set off from the five traits (dashed rule)
                yl = self.get_y() + 1.0
                self.set_draw_color(c["outline"]); self.set_line_width(0.2); self.set_dash_pattern(dash=0.8, gap=0.8)
                self.line(self.l_margin, yl, self.l_margin + self.epw, yl)
                self.set_dash_pattern(); self.set_draw_color(0)
                self.set_y(yl + 1.0)
            self.set_x(self.l_margin)
            self.set_font("ui", "", 8.5)
            self.cell(label_w, row_h, RU_TITLES[k])
            x, y = self.get_x(), self.get_y() + (row_h - bar_h) / 2
            # track: light fill with a grey outline, so the full 0…1 length is visible (outline 3.84:1 on white)
            self.set_fill_color(c["track"]); self.set_draw_color(c["outline"]); self.set_line_width(0.2)
            self.rect(x, y, bar_w, bar_h, style="DF")
            # each bar in the colour of its line on the charts (the interview bar stays brown)
            hexc = TRAIT_BAR_PDF.get(k)
            self.set_fill_color(*(_rgb(hexc) if hexc else c["fill"]))
            self.rect(x, y, bar_w * max(0.01, min(1.0, score)), bar_h, style="F")
            # 0.5 reference: grey stubs outside the bar; inside it a white segment where the fill covers the middle,
            # otherwise a grey one on the light track (#555555 on #f2f2f2, 6.7:1), so the mark crosses every bar
            xm = x + bar_w / 2
            self.set_draw_color(c["mid_tick"])
            self.line(xm, y - 1.0, xm, y); self.line(xm, y + bar_h, xm, y + bar_h + 1.0)
            if score > 0.5:
                self.set_draw_color(255)
            self.line(xm, y, xm, y + bar_h)
            self.set_draw_color(0)
            self.set_x(x + bar_w + 2)
            text = f"{score:.2f}   {pct_phrase(pct, ref)[0]}".rstrip()
            # cell() insets the text by c_margin on both sides: the room for it is that much smaller
            room = self.l_margin + self.epw - self.get_x() - 2 * self.c_margin
            for pt in (8, 7.5, 7.2, 7.0):
                self.set_font("ui", "", pt)
                if self.get_string_width(text) <= room:
                    break
            self.cell(0, row_h, text, new_x="LMARGIN", new_y="NEXT")
        # scale under the bars: 0, 0.5, 1, each label centred on its point of the bar
        self.set_font("ui", "", 7.5); self.set_text_color(NOTE_GREY)
        y = self.get_y()
        for val, pos in (("0", x), ("0.5", x + bar_w / 2), ("1", x + bar_w)):
            self.set_xy(pos - 5, y); self.cell(10, 3.6, val, align="C")
        self.set_text_color(0)
        self.set_xy(self.l_margin, y + 4.2)
        self._draw_bars_legend(self._legend_lines(self._bars_legend()))
        self.set_font("ui", "", 8); self.set_text_color(NOTE_GREY)
        self.multi_cell(0, 4.0, self._bars_note(traits, interview), new_x="LMARGIN", new_y="NEXT", align="L")
        self.set_text_color(0)
        self.ln(1)

    def _draw_bars_legend(self, lines):
        c = SCORE_BAR_PDF
        for line in lines:
            y = self.get_y()
            for kind, text, dx in line:
                x = self.l_margin + dx
                if kind == "traits":             # the five trait colours side by side
                    for i, k in enumerate(TRAIT_KEYS):
                        self.set_fill_color(*_rgb(TRAIT_BAR_PDF[k])); self.rect(x + i * 1.3, y + 1.0, 1.3, 2.6, style="F")
                else:
                    self.set_draw_color(c["mid_tick"]); self.set_line_width(0.3)
                    self.line(x + 3, y + 0.4, x + 3, y + 4.0); self.set_draw_color(0); self.set_line_width(0.2)
                self.set_xy(x + 8, y)
                self.set_font("ui", "", 8)
                self.cell(self.get_string_width(text) + 1, 4.4, text)
            self.set_xy(self.l_margin, y + 4.4)
        self.ln(0.8)
