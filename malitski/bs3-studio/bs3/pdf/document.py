"""The page of the PDF report of BS Profiler 3.1 (fpdf2): class Report — A4 with the margins and the font of the
report, the footer with the file name and «стр. 3 из 7», headings, paragraphs, charts with their captions, key-value
tables, the card grid of «Ключевые факты» and «Речь в цифрах», the score bars and the tables whose header repeats on
every page. The card grid and the score bars are the mixins of pdf/widgets.py. The sections that fill the page are in
pdf/sections.py, pdf/frames.py, pdf/mbti_section.py and pdf/appendix.py (the appendices); pdf/charts.py draws the charts.

The layout constants live in pdf/layout.py and are re-exported here: `from bs3.pdf.document import Report, TEXT_W_MM,
NOTE_GREY` works.
"""
from __future__ import annotations

import os

from fpdf import FPDF

from .. import PRODUCT
from .layout import FONT_CANDIDATES, MARGIN_MM, NOTE_GREY, RADAR_W_MM, ROW_GAP_MM, TEXT_W_MM
from .widgets import CardsMixin, ScoreBarsMixin, _rgb

__all__ = ["Report", "FONT_CANDIDATES", "MARGIN_MM", "NOTE_GREY", "RADAR_W_MM", "ROW_GAP_MM", "TEXT_W_MM"]


class Report(CardsMixin, ScoreBarsMixin, FPDF):
    def __init__(self, file_label: str = "", total_pages: int | None = None):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_margins(MARGIN_MM, MARGIN_MM, MARGIN_MM)
        self.set_auto_page_break(auto=True, margin=15)
        for reg, bold in FONT_CANDIDATES:
            if os.path.exists(reg):
                self.add_font("ui", "", reg)
                self.add_font("ui", "B", bold if os.path.exists(bold) else reg)
                break
        else:
            raise RuntimeError("no Unicode TTF font found for the PDF (DejaVu or Arial)")
        self.set_font("ui", "", 10)
        self.file_label = file_label             # original file name of the video, in the footer of every page
        self.total_pages = total_pages           # «стр. 3 из 7»: known from a first layout pass (build_pdf)
        self.plan: dict = {}                     # section key -> number, fixed before printing (references forward)
        self.appx: dict = {}                     # appendix key -> letter

    def footer(self):
        self.set_y(-10)
        self.set_font("ui", "", 8)
        self.set_text_color(NOTE_GREY)
        right = f"{PRODUCT} · стр. {self.page_no()}" + (f" из {self.total_pages}" if self.total_pages else "")
        y = self.get_y()
        if self.file_label:
            self.set_x(self.l_margin)
            self.cell(110, 5, self._middle_cut(self.file_label, 110))
        self.set_xy(self.l_margin, y)
        self.cell(0, 5, right, align="R")
        self.set_text_color(0)

    def _middle_cut(self, text: str, max_w: float) -> str:
        """A long file name loses characters in the middle («начало…конец.mp4»): both ends tell files apart."""
        if self.get_string_width(text) <= max_w:
            return text
        for keep in range(len(text) - 1, 1, -1):
            s = text[:keep // 2] + "…" + text[len(text) - (keep - keep // 2):]
            if self.get_string_width(s) <= max_w:
                return s
        return text[:1] + "…"

    def h1(self, text):
        # a title longer than the line wraps instead of running past the right edge of the page
        self.set_font("ui", "B", 16); self.multi_cell(0, 8, text, new_x="LMARGIN", new_y="NEXT", align="L"); self.ln(2)

    def h2(self, text, keep_mm: float = 25):
        """Section heading kept on the same page as at least `keep_mm` of what follows (its intro and first block)."""
        if self.get_y() + 10 + keep_mm > self.page_break_trigger:
            self.add_page()
        self.ln(2); self.set_font("ui", "B", 12); self.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
        self.set_font("ui", "", 10)

    def section(self, title: str, key: str, keep_mm: float = 25) -> int:
        """Numbered section heading «2. Big Five по ходу ролика»; numbers follow the sections actually printed and are
        fixed in self.plan before printing, so a caption can refer to a later section."""
        n = self.plan[key]
        self.h2(f"{n}. {title}", keep_mm)
        return n

    def h3(self, text, keep_mm: float = 20, size: float = 9):
        """Bold sub-heading kept on the same page as at least `keep_mm` of what follows it."""
        if self.get_y() + 6 + keep_mm > self.page_break_trigger:
            self.add_page()
        self.set_x(self.l_margin)
        self.set_font("ui", "B", size); self.cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")
        self.set_font("ui", "", 10)

    @staticmethod
    def _lh(size: float) -> float:
        return max(3.6, size * 0.5)          # leading follows the font size (small notes stay close)

    @staticmethod
    def _breakable(text) -> str:
        # break words longer than the line (hashes, URLs) so fpdf can wrap them
        return " ".join(w if len(w) < 60 else " ".join(w[i:i + 60] for i in range(0, len(w), 60)) for w in str(text).split(" "))

    def para_height(self, text, size=9, w: float = 0) -> float:
        """Height in mm that para(text, size) takes, without printing it."""
        if not text:
            return 0.0
        self.set_font("ui", "", size)
        lines = self.multi_cell(w or self.epw, self._lh(size), self._breakable(text), align="L", dry_run=True,
                                output="LINES")
        return len(lines) * self._lh(size) + 1

    def para(self, text, size=9, color: int = 0):
        if not text:
            return
        self.set_x(self.l_margin)
        self.set_font("ui", "", size)
        self.set_text_color(color)
        # left-aligned: justified Russian lines get wide gaps
        self.multi_cell(0, self._lh(size), self._breakable(text), align="L")
        self.set_text_color(0)
        self.ln(1)

    def caption(self, text, size=7.5):
        """Caption under a chart: small, dark grey, close to the chart."""
        self.ln(0.5)
        self.para(text, size, color=51)

    def chart_height(self, path, w: float | None = None) -> float:
        from PIL import Image
        with Image.open(path) as im:
            w_px, h_px = im.size
        return (w or self.epw) * h_px / w_px

    def chart(self, path, keep_mm: float = 0, w: float | None = None, x: float | None = None) -> float:
        """Chart PNG at the size pdf/charts drew it (1 pt of the figure = 1 pt on paper), across the text width unless
        `w` says otherwise. A chart that does not fit on the rest of the page, together with `keep_mm` of the caption
        under it, starts a new page: charts are never cut and a caption never ends up on the page after its chart.
        Returns the printed height."""
        w = w or self.epw
        h = self.chart_height(path, w)
        if self.get_y() + h + keep_mm > self.page_break_trigger:
            self.add_page()
        y = self.get_y()
        self.image(path, x=self.l_margin if x is None else x, y=y, w=w, h=h)
        self.set_xy(self.l_margin, y + h)
        return h

    def chart_block_height(self, path, cap: str = "", cap_size: float = 7.5) -> float:
        """Height of chart_block(path, cap): the gap above, the chart and its caption."""
        return 1 + self.chart_height(path) + (0.5 + self.para_height(cap, cap_size) if cap else 0)

    def chart_block(self, path, cap: str = "", cap_size: float = 7.5) -> None:
        """Full-width chart with its caption; the two are never separated by a page break. A chart that opens a page
        carries its own title, so the section heading is not repeated above it: the six millimetres of that repeat
        pushed the next chart off the page and left a hole of 66 mm where it had stood."""
        self.ln(1)
        self.chart(path, keep_mm=(0.5 + self.para_height(cap, cap_size)) if cap else 0)
        if cap:
            self.caption(cap, cap_size)

    @staticmethod
    def _kv_value(v) -> str:
        v = str(v)
        if len(v) > 48 and " " not in v:                # e.g. sha256: split so it wraps
            v = " ".join(v[i:i + 32] for i in range(0, len(v), 32))
        return v

    def kv_table(self, rows, w1=55, size=8.5, lh: float | None = None):
        lh = lh or size * 0.61
        for k, v in rows:
            self.set_x(self.l_margin)
            self.set_font("ui", "B", size); self.cell(w1, lh, str(k))
            self.set_font("ui", "", size)
            self.multi_cell(0, lh, self._kv_value(v), new_x="LMARGIN", new_y="NEXT", align="L")

    def kv_table_height(self, rows, w1=55, size=8.5, lh: float | None = None) -> float:
        """Height in mm that kv_table(rows, w1, size, lh) takes, without printing it."""
        lh = lh or size * 0.61
        self.set_font("ui", "", size)
        return sum(len(self.multi_cell(self.epw - w1, lh, self._kv_value(v), dry_run=True, output="LINES")) * lh
                   for _, v in rows)

    # ---------------------------------------------------------------- tables
    def _header_lines(self, header, widths, size) -> list:
        """How many lines each header title takes in its own column: a title is wrapped by its '\\n' and, when the
        column came out narrower than the title, by the column width itself. Measured at the printed widths, so a
        header that wraps one line more than it was written still gets a tall enough row."""
        self.set_font("ui", "B", size)
        return [len(self.multi_cell(w, size * 0.5, str(h), dry_run=True, output="LINES")) for h, w in zip(header, widths)]

    def _header_height(self, header, widths, size, chips=None, groups=None) -> float:
        chip = (2.2 + 1.0) if chips and any(chips) else 0.0
        return max(self._header_lines(header, widths, size)) * size * 0.5 + 1.5 + chip \
            + ((size * 0.5 + 1.5) if groups else 0.0)

    def _table_header(self, header, widths, size, chips=None, groups=None):
        """Header cells may contain '\\n' (multi-line titles); all cells get the same height.
        chips: optional colour per column ('#rrggbb' or None), drawn as a small outlined strip above the title.
        groups: optional [(title, n_columns)] row above the header, spanning columns."""
        x0, y0 = self.l_margin, self.get_y()
        if groups:
            self.set_font("ui", "B", size)
            gh = size * 0.5 + 1.5
            x, i = x0, 0
            for title, span in groups:
                w = sum(widths[i:i + span])
                self.rect(x, y0, w, gh)
                self.set_xy(x, y0 + 0.75)
                self.cell(w, size * 0.5, str(title), align="C")
                x += w; i += span
            y0 += gh
        n_lines = self._header_lines(header, widths, size)
        self.set_font("ui", "B", size)
        lh = size * 0.5                      # line height in mm for this font size
        chip_h = 2.2 if chips and any(chips) else 0.0
        top = chip_h + 1.0 if chip_h else 0.0
        hh = max(n_lines) * lh + 1.5 + top
        x = x0
        for i, (h, w) in enumerate(zip(header, widths)):
            self.rect(x, y0, w, hh)
            chip = chips[i] if chip_h and i < len(chips) else None
            if chip:
                self.set_fill_color(*_rgb(chip)); self.set_draw_color(51)       # outline #333333
                self.rect(x + w * 0.18, y0 + 1.0, w * 0.64, chip_h, style="DF")
                self.set_draw_color(0)
            self.set_xy(x, y0 + top + (hh - top - n_lines[i] * lh) / 2)
            self.multi_cell(w, lh, str(h), border=0, align="C")
            x += w
        self.set_xy(x0, y0 + hh)
        self.set_font("ui", "", size)

    def table(self, header, rows, widths, size=8, chips=None, zebra=True, first_left=False, groups=None,
              row_h: float = 5.2, min_rows: int = 3, cont_title: str = ""):
        """zebra: every second row on a very light grey (#f5f5f5, decorative) so long rows are easy to follow.
        first_left: left-align the first column (row names) even when it is narrow. The header (with the group row)
        repeats on every page, and a table starts on the next page unless its header and `min_rows` rows fit here.
        cont_title: a quiet line above the repeated header, so a page of numbers says which table it continues."""
        # every table spans the text width, as the charts and the rows of key frames do (and never runs past the margin)
        k = self.epw / sum(widths)
        widths = [w * k for w in widths]
        hh = self._header_height(header, widths, size, chips, groups)
        if self.get_y() + hh + min(min_rows, len(rows)) * row_h > self.page_break_trigger:
            self.add_page()
        self._table_header(header, widths, size, chips, groups)
        for i, r in enumerate(rows):
            if self.get_y() + row_h > self.page_break_trigger:
                self.add_page()
                if cont_title:
                    self.set_font("ui", "B", 8); self.set_text_color(NOTE_GREY)
                    self.cell(0, 5, cont_title, new_x="LMARGIN", new_y="NEXT")
                    self.set_text_color(0)
                self._table_header(header, widths, size, chips, groups)
            shade = zebra and i % 2 == 1
            if shade:
                self.set_fill_color(245)
            self.set_x(self.l_margin)
            for j, (c, w) in enumerate(zip(r, widths)):
                align = "L" if (j == 0 and first_left) or w >= 40 else "C"
                self.cell(w, row_h, str(c), border=1, align=align, fill=shade)
            self.ln()
