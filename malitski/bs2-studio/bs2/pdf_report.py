"""PDF report for one analysed video (fpdf2).

The main part reads top down: a passport of the file and the analysis, the key facts, the Big Five profile (radar beside
the plain-language explanation, score bars with the second opinion), then one numbered section per topic — Big Five over
time, emotions and facial expression, voice and speech, what drove the own model's score — and «Как читать
результаты». The appendices start on a new page: file and analysis parameters, the values of every segment, behaviour
descriptions of the notable segments, the transcript. Every chart of the web page has a print version (pdf_charts.py).
The original file name of the video is printed exactly as it is, also in the footer of every page, so pages of two
reports cannot be mixed up.
"""
from __future__ import annotations

import datetime as _dt
import os
import re
from pathlib import Path

from fpdf import FPDF

from .narrative2 import analyses_parts, fix_counts, key_facts, plural_ru
from .norms import RU_SHORT, TRAIT_KEYS, percentile
from .palette import CARD_PDF, SCORE_BAR_PDF, SECOND_BAR_PDF, TRAIT_BAR_PDF
from .report import _SEC_LABEL, DISCLAIMER_RU, INTERVIEW_DISCLAIMER_RU, fmt_secs, seg_label
from .ru_texts import transcript_shown, vocabulary_shown

TITLES = {
    "openness": "Открытость опыту", "conscientiousness": "Добросовестность", "extraversion": "Экстраверсия",
    "agreeableness": "Доброжелательность", "emotional_stability": "Эмоциональная стабильность",
    "interview": "Впечатление «собеседование»",
}
MEMBERS = {"oceanai": "OCEAN-AI", "mm": "Своя модель (MM-PSYCHE)", "scene": "SSL-MEPR (сцена)"}
# two-line column headers: long Russian words do not fit narrow table columns
TITLES_2L = {"openness": "Открытость\nопыту", "conscientiousness": "Добросовест-\nность", "extraversion": "Экстра-\nверсия",
             "agreeableness": "Доброжела-\nтельность", "emotional_stability": "Эмоц.\nстабильность",
             "interview": "Собесе-\nдование"}
# «Значения по отрезкам» has 14 columns: the short trait names are broken over two lines where one line is wider than
# its column of numbers; the legend under the table joins the halves back («Добро-жел.» -> «Доброжел.»)
SEG_HEAD = {**RU_SHORT, "agreeableness": "Добро-\nжел.", "emotional_stability": "Эм.\nстаб.", "interview": "Собе-\nсед."}
SYSTEM_TITLES = {"oceanai": "OCEAN-AI", "mm": "своя модель (MM-PSYCHE)", "scene": "SSL-MEPR (сцена)",
                 "ensemble": "ансамбль", "sslmepr": "SSL-MEPR"}
MODALITY_TITLES = {**SYSTEM_TITLES, "audio": "голос", "video": "видео", "text": "речь", "face": "лицо",
                   "behavior": "описание поведения"}
# container tags from media.probe_media (ffprobe names) -> row titles of the «Файл» table
MEDIA_TAGS = {"creation_time": "Записан (метка в файле)", "encoder": "Программа записи",
              "com.apple.quicktime.make": "Производитель камеры", "com.apple.quicktime.model": "Модель камеры",
              "title": "Название"}
LANG_RU = {"ru": "русский", "en": "английский"}
EMO_NAMES = {"joy": "радость", "surprise": "удивление", "neutral": "нейтрально", "sadness": "грусть", "fear": "страх",
             "anger": "злость", "disgust": "отвращение"}
HOW_TO_READ_EXTRA = ("Эмоции, голос и мимика — сигналы моделей, обученных на англоязычных корпусах и фотографиях; это "
                     "наблюдения о поведении на видео, а не диагноз.")
NOTE_GREY = 85                      # #555555, 7.46:1 on white: notes, card labels, the footer
BEHAVIOR_MAX = 6                    # appendix «Описание поведения»: at most this many notable segments


def _when(v) -> str:
    """'2026-08-13T15:37:12.000000Z' -> '2026-08-13 15:37:12 по всемирному времени'; anything that is not an ISO time
    stays as it is."""
    s = str(v or "")
    m = re.match(r"^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})(?:\.\d+)?(Z|[+-]\d{2}:?\d{2})?$", s)
    if not m:
        return s
    tz = m.group(3)
    return f"{m.group(1)} {m.group(2)}" + (" по всемирному времени" if tz == "Z" else (f" (часовой пояс {tz})" if tz else ""))


# ffprobe codec names -> the usual names of the formats
CODEC_NAMES = {"h264": "H.264", "hevc": "H.265 (HEVC)", "h265": "H.265", "vp8": "VP8", "vp9": "VP9", "av1": "AV1",
               "mpeg4": "MPEG-4", "mjpeg": "Motion JPEG", "prores": "ProRes", "aac": "AAC", "mp3": "MP3", "opus": "Opus",
               "vorbis": "Vorbis", "flac": "FLAC", "ac3": "AC-3", "eac3": "E-AC-3", "alac": "ALAC"}


def _codec(v) -> str:
    s = str(v or "")
    return CODEC_NAMES.get(s.lower(), "PCM" if s.lower().startswith("pcm_") else s or "—")


def _encoder(v) -> str:
    """'Lavf60.16.100' (the container writer of FFmpeg) -> 'FFmpeg (libavformat 60.16.100)'."""
    s = str(v or "")
    m = re.match(r"^Lav([fc])(\d[\d.]*)$", s)
    return f"FFmpeg (libav{'format' if m.group(1) == 'f' else 'codec'} {m.group(2)})" if m else s


SMALL_POOL = 20      # below this many processed videos a percentage only looks precise (same rule as the web page)


def _small_pool_n(ref: str):
    """Size of the pool when it is below SMALL_POOL (the position is then given in words), otherwise None."""
    m = re.search(r"N\s*=\s*(\d+)", ref or "")
    return int(m.group(1)) if "пула" in (ref or "") and m and int(m.group(1)) < SMALL_POOL else None


def _trained_on(m: dict, members: list) -> str:
    """model.trained_on names one corpus for the whole ensemble; say which member learned from what."""
    oc = {"ru": "MuPTA (русская речь)", "en": "First Impressions V2"}.get(m.get("lang"))
    if m.get("backend") == "ensemble" and members:
        parts = [f"OCEAN-AI — {oc}" if x == "oceanai" and oc else "своя модель — First Impressions V2" if x == "mm"
                 else None for x in members]
        if all(parts):
            return "; ".join(parts)
    return str(m.get("trained_on") or "—")


def pct_phrase(pct, ref: str = "") -> str:
    """'выше, чем у 83% русских роликов' / 'ниже, чем у 95% людей в FIV2' / 'примерно посередине среди …'.
    The wording is the one of the score bars of the web page: webparts writes them, so the same number is never
    worded two ways (the language of the pool — русских / английских / обработанных — comes from the reference)."""
    from .webparts import _pct_phrase
    return _pct_phrase(pct, ref)[0]


def _ref_ru(ref: str) -> str:
    """percentile_ref from result.json (genitive, reads after «относительно») without technical English words."""
    r = re.sub(r",\s*своя модель\s*$", "", ref or "")
    r = re.sub(r"\(N\s*=\s*(\d+)\)", r"(сейчас их \1)", r)          # pool size, as on the web page
    return r.replace("train First Impressions V2", "обучающей выборки First Impressions V2").replace(
        "train FIV2", "обучающей выборки FIV2")


def _where_ru(ref: str) -> str:
    """The reference group as on the web page: «среди обработанных русских роликов (сейчас их 5)» for the pool of
    processed videos, «относительно обучающей выборки First Impressions V2 (6000 клипов)» otherwise."""
    r = _ref_ru(ref)
    m = re.match(r"^пула\s+(обработанных\s.+)$", r)
    return f"среди {m.group(1)}" if m else f"относительно {r}"


def _asr_ru(name) -> str:
    """'openai/whisper-large-v3-turbo' -> 'Whisper large-v3-turbo': the model name without the hub prefix."""
    m = re.match(r"^(?:[\w.-]+/)?whisper-(.+)$", str(name or ""), re.I)
    return f"Whisper {m.group(1)}" if m else str(name or "")


def _version_ru(v) -> str:
    """'2.0.0a1' -> '2.0.0, альфа-версия 1' (PEP 440 pre-release suffixes a/b/rc in words)."""
    m = re.match(r"^(\d+(?:\.\d+)*)(a|b|rc)(\d+)$", str(v or ""))
    if not m:
        return str(v or "—")
    return f"{m.group(1)}, {({'a': 'альфа', 'b': 'бета', 'rc': 'предрелизная'})[m.group(2)]}-версия {m.group(3)}"


def _group_name(keys: list[str]) -> str:
    """Which score rows a reference group applies to, for the note under the score bars."""
    if set(keys) == set(TRAIT_KEYS):
        return "пять черт"
    return ", ".join("«собеседование»" if k == "interview" else TITLES[k].lower() for k in keys)


def _rgb(hexc: str) -> tuple:
    return tuple(int(hexc.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))


def _mmss(t: float) -> str:
    s = int(t)
    return f"{s // 60}:{s % 60:02d}"


def _seg(report: dict, start, end) -> str:
    """«1:20–1:40» as report.seg_label, but h:mm:ss from an hour on, like the time axis of the charts and the
    «Отрезок» column of the web table: one report never prints «1:05:00» on a chart and «65:00» in a table."""
    if float(report.get("duration_sec") or 0) < 3600:
        return seg_label(start, end)

    def f(t) -> str:
        t = int(round(float(t)))
        return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}"
    return f"{f(start)}–{f(end)}"


def _hms_text(text: str, dur: float) -> str:
    """From an hour on the report writes times as h:mm:ss (as the charts do); the narrative is shared with the web
    page and writes «отрезок 37:20–39:40», so its segment ranges are rewritten for the PDF."""
    if float(dur or 0) < 3600:
        return text

    def one(m) -> str:
        def f(mm: str, ss: str) -> str:
            return f"{int(mm) // 60}:{int(mm) % 60:02d}:{ss}"
        return f"{f(m.group(1), m.group(2))}–{f(m.group(3), m.group(4))}"
    return re.sub(r"\b(\d{1,3}):(\d\d)–(\d{1,3}):(\d\d)\b", one, text)


def _dash(text) -> str:
    """Russian punctuation in a description written by a model: a hyphen between spaces is an em dash."""
    return re.sub(r"(?<=\s)-(?=\s)", "—", str(text or ""))


def _one_line(head: str) -> str:
    """A two-line column header as one word for the legend: «Добро-\\nжел.» -> «Доброжел.», «Эм.\\nстаб.» -> «Эм. стаб.»."""
    return str(head).replace("-\n", "").replace("\n", " ")


MARGIN_MM = 10                      # left, right and top page margin (the fpdf default, made explicit)
TEXT_W_MM = 210 - 2 * MARGIN_MM     # A4 text width: every chart (pdf_charts draws them this wide), table and row of frames
RADAR_W_MM, ROW_GAP_MM = 80, 4      # the radar beside the explanation: 80 mm + 4 mm gap + 106 mm of text

FONT_CANDIDATES = [
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/mnt/c/Windows/Fonts/arial.ttf", "/mnt/c/Windows/Fonts/arialbd.ttf"),
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
]


class Report(FPDF):
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
        right = f"BS 2.0 · стр. {self.page_no()}" + (f" из {self.total_pages}" if self.total_pages else "")
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
        """Chart PNG at the size pdf_charts drew it (1 pt of the figure = 1 pt on paper), across the text width unless
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

    def kv_table(self, rows, w1=55, size=8.5, lh: float | None = None):
        lh = lh or size * 0.61
        for k, v in rows:
            self.set_x(self.l_margin)
            self.set_font("ui", "B", size); self.cell(w1, lh, str(k))
            self.set_font("ui", "", size)
            v = str(v)
            if len(v) > 48 and " " not in v:            # e.g. sha256: split so it wraps
                v = " ".join(v[i:i + 32] for i in range(0, len(v), 32))
            self.multi_cell(0, lh, v, new_x="LMARGIN", new_y="NEXT", align="L")

    # ---------------------------------------------------------------- cards (key facts, speech in numbers)
    def _card_layout(self, items, cols: int, gap: float):
        w = (self.epw - gap * (cols - 1)) / cols
        pad, inner = 1.8, (self.epw - gap * (cols - 1)) / cols - 3.6
        heights = []
        for lab, val, note in items:
            self.set_font("ui", "", 7.5)
            n_lab = len(self.multi_cell(inner, 3.4, str(lab), dry_run=True, output="LINES"))
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

    def cards_height(self, items, cols: int = 3, gap: float = 4.0) -> float:
        if not items:
            return 0.0
        _, _, _, _, row_h = self._card_layout(items, cols, gap)
        return sum(row_h) + gap * 0.75 * (len(row_h) - 1) + 1.5

    def cards(self, items, cols: int = 3, gap: float = 4.0):
        """(label, value, note) -> a grid of outlined cards, as on the web page: label and note small and grey, the
        value large and bold. The grid is never split between pages."""
        if not items:
            return
        w, pad, inner, rows, row_h = self._card_layout(items, cols, gap)
        if self.get_y() + self.cards_height(items, cols, gap) > self.page_break_trigger:
            self.add_page()
        y = self.get_y()
        for r, rh in zip(rows, row_h):
            for j, i in enumerate(r):
                lab, val, note = items[i]
                x = self.l_margin + j * (w + gap)
                self.set_fill_color(CARD_PDF["fill"]); self.set_draw_color(CARD_PDF["outline"]); self.set_line_width(0.2)
                self.rect(x, y, w, rh, style="DF", round_corners=True, corner_radius=1.2)
                self.set_xy(x + pad, y + pad)
                self.set_font("ui", "", 7.5); self.set_text_color(NOTE_GREY)
                self.multi_cell(inner, 3.4, str(lab), align="L", new_x="LEFT", new_y="NEXT")
                self.set_y(self.get_y() + 0.6); self.set_x(x + pad)
                self.set_font("ui", "B", 11); self.set_text_color(0)
                self.multi_cell(inner, 5.0, str(val if val not in (None, "") else "—"), align="L", new_x="LEFT", new_y="NEXT")
                if note:
                    self.set_y(self.get_y() + 0.3); self.set_x(x + pad)
                    self.set_font("ui", "", 7.5); self.set_text_color(NOTE_GREY)
                    self.multi_cell(inner, 3.4, str(note), align="L", new_x="LEFT", new_y="NEXT")
            y += rh + gap * 0.75
        self.set_text_color(0); self.set_draw_color(0)
        self.set_xy(self.l_margin, y - gap * 0.75 + 1.5)

    # ---------------------------------------------------------------- score bars
    BAR_W, BAR_H = 52, 3.6

    def _bar_rows(self, traits: dict, interview: dict | None):
        return [(k, traits[k]) for k in TRAIT_KEYS] + ([("interview", interview)] if interview else [])

    def _bars_legend(self, second: bool):
        """Legend items (kind, text) under the score bars."""
        # «как на графиках» with one reservation: the extraversion fill is a step darker than its line on the charts,
        # because the chart colour gives only 2.9:1 on the light track of the bar
        items = [("traits", "оценка черты 0…1 (цвет — как на графиках, экстраверсия чуть темнее)")]
        if second:
            items.append(("second", "второе мнение: своя модель, FIV2"))
        items.append(("tick", "середина шкалы"))
        return items

    def _legend_lines(self, items) -> list:
        """Legend items wrapped into lines of the text width: [[(kind, text, x_offset), …], …]."""
        self.set_font("ui", "", 8)
        lines, cur, x = [], [], 0.0
        for kind, text in items:
            w = 8 + self.get_string_width(text) + 1
            if cur and x + w > self.epw:
                lines.append(cur); cur, x = [], 0.0
            cur.append((kind, text, x)); x += w + 5
        if cur:
            lines.append(cur)
        return lines

    def _bars_note(self, traits: dict, interview: dict | None, second: bool) -> str:
        groups: dict[str, list[str]] = {}          # reference group in words («среди …», «относительно …») -> item keys
        for k, t in self._bar_rows(traits, interview):
            groups.setdefault(_where_ru(t.get("percentile_ref", "train First Impressions V2 (6000 клипов)")), []).append(k)
        if len(groups) == 1:
            where = next(iter(groups))
        else:
            where = "; ".join(f"{_group_name(keys)} — {ref}" for ref, keys in groups.items())
        note = f"Рядом с полоской — положение: {where}."
        n = _small_pool_n(traits[TRAIT_KEYS[0]].get("percentile_ref", ""))
        if n is not None:
            note += f" Роликов в сравнении пока {n}, поэтому положение черт описано словами, а не в процентах."
        if second:
            note += (" Второе мнение — на другой шкале: своя модель обучена на англоязычных влогерах, поэтому на русских "
                     "роликах её значения ниже; сравнивайте положение и порядок черт, а не сами числа.")
        if interview:
            note += " Коричневая полоска — впечатление «собеседование» (своя модель, шкала FIV2)."
        return note

    def score_bars_height(self, traits: dict, interview: dict | None, second: dict | None = None) -> float:
        n = len(TRAIT_KEYS)
        h = n * (8.6 if second else 6.0) + (8.0 if interview else 0.0) + 4.2
        h += len(self._legend_lines(self._bars_legend(bool(second)))) * 4.4 + 1
        self.set_font("ui", "", 8)
        h += len(self.multi_cell(self.epw, 4.0, self._bars_note(traits, interview, bool(second)), dry_run=True,
                                 output="LINES")) * 4.0
        return h + 1

    def score_bars(self, traits: dict, interview: dict | None, second: dict | None = None):
        """One row per trait: the bar is the score itself (0…1, what a reader expects to see filled), the text gives the
        score and the position relative to the reference group in words. second: {trait: score} of the second opinion
        (own model, FIV2 scale), drawn as a thin slate bar under each trait with its FIV2 position."""
        items = self._bar_rows(traits, interview)
        bar_w, bar_h = self.BAR_W, self.BAR_H
        c = SCORE_BAR_PDF
        # the label column is as wide as the longest title; the phrase after the bar must end at the right margin
        self.set_font("ui", "", 8.5)
        label_w = max(self.get_string_width(TITLES[k]) for k, _ in items) + 3
        x = self.l_margin + label_w
        row_h = 5.4 if second else 6.0      # with the second opinion every trait takes 5.4 + 3.2 mm
        for k, t in items:
            pct = t.get("percentile", t.get("percentile_vs_fiv2")); score = float(t["score"])
            ref = t.get("percentile_ref", "train First Impressions V2 (6000 клипов)")
            if k == "interview":        # a separate label of the own model: set off from the five traits (dashed rule)
                yl = self.get_y() + 1.0
                self.set_draw_color(c["outline"]); self.set_line_width(0.2); self.set_dash_pattern(dash=0.8, gap=0.8)
                self.line(self.l_margin, yl, self.l_margin + self.epw, yl)
                self.set_dash_pattern(); self.set_draw_color(0)
                self.set_y(yl + 1.0)
            self.set_x(self.l_margin)
            self.set_font("ui", "", 8.5)
            self.cell(label_w, row_h, TITLES[k])
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
            text = f"{score:.2f}   {pct_phrase(pct, ref)}"
            # cell() insets the text by c_margin on both sides: the room for it is that much smaller
            room = self.l_margin + self.epw - self.get_x() - 2 * self.c_margin
            for pt in (8, 7.5, 7.2, 7.0):
                self.set_font("ui", "", pt)
                if self.get_string_width(text) <= room:
                    break
            self.cell(0, row_h, text, new_x="LMARGIN", new_y="NEXT")
            if second and k in second:
                s = float(second[k])
                ys = self.get_y() + 0.3
                self.set_fill_color(c["track"]); self.rect(x, ys, bar_w, 1.4, style="F")
                self.set_fill_color(*_rgb(SECOND_BAR_PDF))
                self.rect(x, ys, bar_w * max(0.01, min(1.0, s)), 1.4, style="F")
                self.set_xy(x + bar_w + 2, self.get_y() - 0.4)
                self.set_text_color(*_rgb(SECOND_BAR_PDF))
                text = f"второе мнение {s:.2f} · {pct_phrase(percentile(k, s), 'train FIV2')}"
                room = self.l_margin + self.epw - self.get_x() - 2 * self.c_margin
                for pt in (7.5, 7.2, 7.0):          # «примерно посередине …» is the longest phrase: it must not
                    self.set_font("ui", "", pt)     # run past the right margin
                    if self.get_string_width(text) <= room:
                        break
                else:                               # the longest phrase at the smallest size: one word less
                    text = text.replace("примерно посередине", "посередине")
                self.cell(0, 3.2, text, new_x="LMARGIN", new_y="NEXT")
                self.set_text_color(0)
        # scale under the bars: 0, 0.5, 1, each label centred on its point of the bar
        self.set_font("ui", "", 7.5); self.set_text_color(NOTE_GREY)
        y = self.get_y()
        for val, pos in (("0", x), ("0.5", x + bar_w / 2), ("1", x + bar_w)):
            self.set_xy(pos - 5, y); self.cell(10, 3.6, val, align="C")
        self.set_text_color(0)
        self.set_xy(self.l_margin, y + 4.2)
        self._draw_bars_legend(self._legend_lines(self._bars_legend(bool(second))))
        self.set_font("ui", "", 8); self.set_text_color(NOTE_GREY)
        self.multi_cell(0, 4.0, self._bars_note(traits, interview, bool(second)), new_x="LMARGIN", new_y="NEXT", align="L")
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
                elif kind == "second":
                    self.set_fill_color(*_rgb(SECOND_BAR_PDF)); self.rect(x, y + 1.6, 6.5, 1.4, style="F")
                else:
                    self.set_draw_color(c["mid_tick"]); self.set_line_width(0.3)
                    self.line(x + 3, y + 0.4, x + 3, y + 4.0); self.set_draw_color(0); self.set_line_width(0.2)
                self.set_xy(x + 8, y)
                self.set_font("ui", "", 8)
                self.cell(self.get_string_width(text) + 1, 4.4, text)
            self.set_xy(self.l_margin, y + 4.4)
        self.ln(0.8)

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


def _empty_text(r: dict) -> bool:
    """The segment's own transcript is empty: the text-emotion model then answers "neutral 100%", which is no data."""
    return "text_en" in r and not str(r.get("text_en") or "").strip()


def _seg_words(r: dict) -> int:
    try:
        return int((r.get("speech") or {}).get("words") or 0)
    except (TypeError, ValueError):
        return 0


def _scored(report: dict) -> list:
    return [t for t in (report.get("timeline") or []) if t.get("scores")]


def _rep_segment(report: dict):
    """The representative segment (closest to the mean profile; explanations and key frames are built on it) of a
    video with at least two scored segments, otherwise None."""
    segs = _scored(report)
    if len(segs) < 2:
        return None
    return next((t for t in segs if t.get("segment") == report.get("representative_segment")), None)


def behavior_by_segment(report: dict) -> list:
    """[(start, end, text)] of behavior_description_ru split on its «[0–20 с]» labels; [] when it has none."""
    text = str(report.get("behavior_description_ru") or "")
    ms = list(_SEC_LABEL.finditer(text))
    out = []
    for i, m in enumerate(ms):
        s, e = float(m.group(1).replace(",", ".")), float(m.group(2).replace(",", "."))
        body = text[m.end(): ms[i + 1].start() if i + 1 < len(ms) else len(text)].strip()
        out.append((s, e, body))
    return out


# ---------------------------------------------------------------- plan: which sections and appendices are printed
def _plan(pdf: Report, report: dict, explanation, frames, charts: dict) -> None:
    an = report.get("analyses") or {}
    te, fa = an.get("emotions_text") or {}, an.get("face") or {}
    has = {"profile": True,
           "timeline": bool(charts.get("traits")) and len(_scored(report)) >= 2,
           "emotions": bool(te.get("mean") or fa.get("mean") or charts.get("emotions")),
           "voice_speech": bool(an.get("voice") or an.get("speech") or charts.get("voice") or charts.get("speech")),
           "explain": bool(explanation or frames),
           "how_to_read": True}          # numbered like the rest: between the sections and the lettered appendices
    n = 0
    for key, ok in has.items():
        if ok:
            n += 1
            pdf.plan[key] = n
    note, transcript = transcript_shown(report)
    appx = {"file": True, "segments": len(_segment_rows(report)) >= 2,
            "behavior": bool(report.get("behavior_description_ru")), "transcript": bool(note or transcript)}
    letters = iter("АБВГДЕ")
    for key, ok in appx.items():
        if ok:
            pdf.appx[key] = next(letters)


# ---------------------------------------------------------------- main part
def _passport(pdf: Report, report: dict, media: dict | None, fname: str) -> None:
    """Four short lines instead of the file and analysis tables (those are in appendix А)."""
    m = report.get("model") or {}
    md = media or report.get("media") or {}
    parts = [fname]
    dur = md.get("duration_sec") or report.get("duration_sec")
    if dur:
        parts.append(fmt_secs(dur))
    if md.get("width") and md.get("height"):
        w, h = md["width"], md["height"]
        if str(md.get("rotation") or "0").lstrip("-") in ("90", "270"):
            w, h = h, w
        size = f"{w}×{h}"
        try:
            size += f", {float(md['fps']):g} кадр/с" if md.get("fps") else ""
        except (TypeError, ValueError):
            pass
        parts.append(size)
    if md.get("size_mb") is not None:
        parts.append(f"{md['size_mb']} МБ")
    created = str(report.get("created_at") or "")
    mt = re.match(r"^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})", created)
    analysis = [f"{mt.group(1)} {mt.group(2)}" if mt else created or None,
                f"язык речи — {LANG_RU.get(m.get('lang'), m.get('lang') or '—')}"]
    n_seg = int(report.get("segments") or 0)
    analysis.append(f"{n_seg} {plural_ru(n_seg, 'отрезок', 'отрезка', 'отрезков')} по ~20 с" if n_seg > 1
                    else "ролик оценён целиком")
    t = report.get("timings_sec") or {}
    if t.get("total_wall", t.get("total")) is not None:
        analysis.append(f"время обработки {fmt_secs(t.get('total_wall', t.get('total')))}")
    members = [x for x in (report.get("variant_scores") or {})] or \
        [x for x in report.get("modalities_used", []) if x in SYSTEM_TITLES]
    primary = m.get("primary")
    if primary == "oceanai" and m.get("lang") == "ru":
        score = ("основная — OCEAN-AI (веса MuPTA, русская речь); второе мнение — своя модель MM-PSYCHE (шкала FIV2)"
                 if "mm" in members else "OCEAN-AI (веса MuPTA, русская речь)")
    elif primary:
        score = f"основная — {SYSTEM_TITLES.get(primary, primary)}"
    elif set(members) >= {"oceanai", "mm"}:
        score = "среднее OCEAN-AI и своей модели MM-PSYCHE (шкала First Impressions V2)"
    else:
        score = SYSTEM_TITLES.get(m.get("backend"), str(m.get("backend") or "—"))
    rows = [("Файл", " · ".join(parts)), ("Анализ", " · ".join(p for p in analysis if p)), ("Оценка", score),
            ("Отчёт", f"создан {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}; технические сведения о файле и анализе — "
                      f"в приложении {pdf.appx.get('file', 'А')}")]
    pdf.set_font("ui", "B", 8.5)
    w1 = max(pdf.get_string_width(k) for k, _ in rows) + 3
    pdf.kv_table(rows, w1=w1, size=8.5, lh=4.6)
    pdf.ln(2)


def _speech_cards(sp: dict) -> list:
    """The 9 cards of «Речь в цифрах», with the labels, values and notes of the web tab «Речь»."""
    def whole(v):
        return "—" if v is None else f"{float(v):.0f}"

    def per_100(v) -> str:
        """«0 на 100 слов» under a value of 3 contradicts itself: a rate that rounds to zero is said in words."""
        x = float(v or 0)
        return "меньше 1 на 100 слов" if 0 < x < 0.95 else f"{x:.0f} на 100 слов"
    fillers = sp.get("fillers")
    return [("Слов всего", whole(sp.get("words")), ""),
            ("Разных слов", whole(sp.get("unique_words")), "без повторов"),
            ("Темп речи, слов в минуту", whole(sp.get("words_per_min_speech")), "только время, когда человек говорит"),
            ("Темп с учётом пауз, слов в минуту", whole(sp.get("words_per_min_wall")), "по всей длине ролика"),
            ("Доля пауз", f"{sp.get('pause_share', 0):.0%}", "паузы от 0.5 с, доля времени ролика"),
            ("Длинных пауз", whole(sp.get("long_pauses")), "дольше 2 секунд"),
            ("Слов-заполнителей", whole(fillers),
             per_100(sp.get("fillers_per_100")) if fillers is not None else ""),
            ("Слов во фразе", whole(sp.get("mean_sentence")), "в среднем"),
            ("Разнообразие словаря", f"{float(sp['ttr']):.0%}" if sp.get("ttr") is not None else "—",
             "доля разных слов среди всех; зависит от длины текста")]


def _pdf_facts(report: dict, speech_cards: bool) -> list:
    """The key facts of the web overview. Not everything of the page belongs in the report twice: when «Речь в
    цифрах» follows later, the note of «Темп речи» no longer repeats the pauses and the fillers printed there in
    full, it says what the tempo itself is counted on."""
    facts = key_facts(report)
    if not speech_cards:
        return facts
    return [(lab, val, ("только время, когда человек говорит" if "минуту" in str(val) else "")
                       if lab == "Темп речи" else note) for lab, val, note in facts]


def _profile_section(pdf: Report, report: dict, explanation, charts: dict) -> None:
    """1. Radar beside the plain-language explanation (its overflow continues under the row), then the score bars."""
    try:
        from .narrative import build_narrative
        narrative = _hms_text(fix_counts(build_narrative(report, explanation)), report.get("duration_sec"))
    except Exception:  # noqa: BLE001
        narrative = ""
    radar = charts.get("profile")
    text_x = pdf.l_margin + RADAR_W_MM + ROW_GAP_MM
    text_w = pdf.l_margin + pdf.epw - text_x
    radar_h = pdf.chart_height(radar, RADAR_W_MM) if radar else 0.0
    m = report.get("model") or {}
    primary = m.get("primary")
    var = report.get("variant_scores") or {}
    second = next((v for k, v in var.items() if primary and k != primary and all(t in v for t in TRAIT_KEYS)), None)
    bars_h = pdf.score_bars_height(report["traits"], report.get("interview"), second)

    def layout(size: float):
        lh = size * 0.5
        pdf.set_font("ui", "", size)
        lines = pdf.multi_cell(text_w, lh, narrative, dry_run=True, output="LINES") if (narrative and radar) else []
        beside = int(max(0.0, radar_h - 7) // lh)
        row_h = max(radar_h, 7 + min(len(lines), beside) * lh)
        rest = " ".join(s.strip() for s in lines[beside:])
        return size, lh, lines, beside, row_h, rest

    def fits(v) -> bool:
        """The whole score-bar block still starts on this page with the explanation laid out this way."""
        return pdf.get_y() + 12 + v[4] + 1 + (pdf.para_height(v[5], v[0]) if v[5] else 0) + 6 + bars_h \
            <= pdf.page_break_trigger

    # the explanation is 8.5 pt; 8 pt when that keeps the whole score-bar block on the first page, or when it ends
    # the explanation beside the radar instead of leaving a tail of it across the page under the row
    cur = layout(8.5)
    alt = layout(8.0) if (radar and narrative) else None
    if alt and fits(alt) and (not fits(cur) or (cur[5] and not alt[5])):
        cur = alt
    size, lh, lines, beside, row_h, rest = cur
    pdf.section("Big Five: профиль и оценки", "profile", keep_mm=row_h if radar else 40)
    if radar:
        y0 = pdf.get_y() + 1
        pdf.image(radar, x=pdf.l_margin, y=y0, w=RADAR_W_MM, h=radar_h)
        if narrative:
            pdf.set_xy(text_x, y0)
            pdf.set_font("ui", "B", 9); pdf.cell(text_w, 6, "Краткие выводы", new_x="LEFT", new_y="NEXT")
            pdf.set_xy(text_x, y0 + 7)
            pdf.set_font("ui", "", size)
            pdf.multi_cell(text_w, lh, "\n".join(lines[:beside]), align="L", new_x="LEFT", new_y="NEXT")
        pdf.set_xy(pdf.l_margin, y0 + row_h + 1)
        if rest:
            pdf.para(rest, size)
    elif narrative:
        pdf.h3("Краткие выводы")
        pdf.para(narrative, size)
    # ---- score bars with the second opinion (Russian speech) or the members (English speech: their mean)
    title = "Оценки по чертам и второе мнение" if second else "Оценки по чертам"
    pdf.h3(title, keep_mm=bars_h)
    pdf.score_bars(report["traits"], report.get("interview"), second)
    if var and not primary:
        pdf.h3("Участники ансамбля: итоговая оценка — их среднее", keep_mm=22)
        rows = [[MEMBERS.get(n, n)] + [f"{float(v.get(k, float('nan'))):.2f}" for k in TRAIT_KEYS] for n, v in var.items()]
        pdf.table(["Модель"] + [TITLES_2L[k] for k in TRAIT_KEYS], rows, [50, 28, 28, 28, 28, 28], size=8, first_left=True)
        who = "Обе модели" if len(var) == 2 else "Все модели"
        pdf.caption(f"{who} на шкале First Impressions V2, оценки от 0 до 1; итог в полосках выше — их среднее.", 8)


def _timeline_section(pdf: Report, report: dict, charts: dict, has_expl: bool) -> None:
    if "timeline" not in pdf.plan:
        return
    std = report.get("scores_std_across_segments") or {}
    cap = ""
    if std:
        cap = "Разброс между отрезками: " + ", ".join(f"{RU_SHORT[k].lower()} ±{std.get(k, 0):.2f}" for k in TRAIT_KEYS) + "."
    seg = _rep_segment(report)
    if seg:
        cap += (f" Рамка и ★ — отрезок {_seg(report, seg['start'], seg['end'])}, ближайший к среднему профилю"
                + (f": по нему построены объяснения (раздел {pdf.plan['explain']})." if has_expl and "explain" in pdf.plan
                   else "."))
    if "segments" in pdf.appx:
        cap += f" Значения по каждому отрезку — в приложении {pdf.appx['segments']}."
    cap = cap.strip()
    # the heading does not repeat the title inside the chart («Big Five по ходу ролика»): the two stood 5 mm apart
    pdf.section("Как менялись оценки по ходу ролика", "timeline",
                keep_mm=pdf.chart_block_height(charts["traits"], cap))
    pdf.chart_block(charts["traits"], cap)


def _emotions_section(pdf: Report, report: dict, charts: dict) -> None:
    """3. Averages first (speech vs face, then the facial expression), then emotions over time."""
    if "emotions" not in pdf.plan:
        return
    an = report.get("analyses") or {}
    parts = analyses_parts(report)
    intro = " ".join(p for p in (parts["text_emotion"], parts["face"]) if p)
    per = an.get("per_segment") or []
    fa = an.get("face") or {}
    text_src = "по транскрипту" if (report.get("model") or {}).get("lang") == "en" else "по переводу транскрипта на английский"
    blocks = []                          # (chart key, caption, message when the chart is missing)
    prof_cap = (f"Средняя доля каждой эмоции за ролик. Речь — модель эмоций текста {text_src}; лицо — модель выражений "
                "по кадрам.")
    # an отрезок whose own transcript came out empty is «нет текста» everywhere else in the report, but the stored
    # average counts it as «нейтрально 100%»: say so, so the two views do not read as contradicting each other
    no_text = [r for r in per if _empty_text(r)]
    if no_text and (an.get("emotions_text") or {}).get("mean"):
        n = len(no_text)
        prof_cap += (" Один отрезок без распознанного текста учтён в средней доле по речи как нейтральный." if n == 1
                     else f" {n} {plural_ru(n, 'отрезок', 'отрезка', 'отрезков')} без распознанного текста учтены "
                          "в средней доле по речи как нейтральные.")
    blocks.append(("emotion_profile", prof_cap,
                   "График среднего профиля эмоций не построен: данных об эмоциях нет."))
    if fa:
        hm = fa.get("head_motion")
        frames = sum(int((r.get("face") or {}).get("frames") or 0) for r in per)
        cap = []
        if hm is not None:
            cap.append(f"Движение головы {'слабое' if hm < 0.05 else ('умеренное' if hm < 0.15 else 'активное')}: смещение "
                       f"между кадрами — {hm:.0%} ширины лица.")
        found = f"Лицо найдено в {fa['face_share']:.0%} кадров" if fa.get("face_share") is not None else ""
        if frames:
            n = len(per)
            found += ("; разобрано " if found else "Разобрано ") + (f"{frames} {plural_ru(frames, 'кадр', 'кадра', 'кадров')} "
                                                                   f"из {n} {plural_ru(n, 'отрезка', 'отрезков', 'отрезков')}")
        if found:
            cap.append(found + ".")
        cap.append("Модель выражений обучена на фотографиях FER-2013 и склонна видеть «грусть» и «страх» в спокойном лице: "
                   "смотрите на изменения по ходу ролика, а не на абсолютные доли.")
        blocks.append(("face_expr", " ".join(cap), "График выражения лица не построен: лицо в кадре не найдено."))
    if len(per) >= 2:                    # one segment = the whole video: the averages above already say everything
        # hatched gaps. A segment with an empty transcript is not always silent: its own recognition may return nothing
        # while the whole-video transcript still has words in that window (tempo and pauses come from there), so
        # «нет речи» is said only when there are no words at all
        text_gaps = [r for r in per if _empty_text(r) or not r.get("emotions_text")]
        gaps = []
        if text_gaps:
            gaps.append("по речи — " + ("в отрезке нет речи" if all(_seg_words(r) == 0 for r in text_gaps)
                                        else "для отрезка нет распознанного текста"))
        if any(not (r.get("face") or {}).get("expressions") for r in per):
            gaps.append("по лицу — лицо не найдено")
        cap = ("Пустая светлая клетка — доля меньше 5 %; числа — доля эмоции в отрезке в процентах (от 15 %, если "
               "помещаются), жирное число — преобладающая эмоция отрезка"
               + (f", как в приложении {pdf.appx['segments']}" if "segments" in pdf.appx else "")
               + ". Полоска у названия строки — цвет этой эмоции."
               + (f" Штриховка — нет данных ({'; '.join(gaps)})." if gaps else ""))
        blocks.append(("emotions", cap, "График эмоций по ходу ролика не построен."))
    first = next((b for b in blocks if charts.get(b[0])), None)
    keep = pdf.para_height(intro, 9) + (pdf.chart_block_height(charts[first[0]], first[1]) if first else 10)
    pdf.section("Эмоции и мимика", "emotions", keep_mm=keep)
    pdf.para(intro, 9)
    for key, cap, missing in blocks:
        if charts.get(key):
            pdf.chart_block(charts[key], cap)
        else:
            pdf.para(missing, 8)


def _voice_speech_section(pdf: Report, report: dict, charts: dict) -> None:
    """4. Voice and speech together, like the web row: intro, the two time charts, «Речь в цифрах», frequent words."""
    if "voice_speech" not in pdf.plan:
        return
    from .charts import VOICE_RU
    an = report.get("analyses") or {}
    parts = analyses_parts(report)
    intro = " ".join(p for p in (parts["voice"], parts["speech"]) if p)
    vo, sp = an.get("voice") or {}, an.get("speech") or {}
    voice_cap = ("Средние за ролик: " + ", ".join(
        f"{VOICE_RU[d]} {vo['mean'].get(d, 0):.2f} (±{(vo.get('std') or {}).get(d, 0):.2f})" for d in VOICE_RU)
        + "; после ± — разброс между отрезками." if vo.get("mean") else "")
    speech_cap = ("Столбики — темп внутри речи, слов в минуту (левая шкала); линия — доля времени отрезка без речи, паузы "
                  "от 0.5 с (правая шкала).")
    first = "voice" if charts.get("voice") else ("speech" if charts.get("speech") else None)
    keep = pdf.para_height(intro, 9) + (pdf.chart_block_height(charts[first], voice_cap if first == "voice" else speech_cap)
                                        if first else 30)
    pdf.section("Голос и речь", "voice_speech", keep_mm=keep)
    pdf.para(intro, 9)
    # a chart without any data is not drawn; when the other one is there, one line says why this one is missing
    if charts.get("voice"):
        pdf.chart_block(charts["voice"], voice_cap)
    elif charts.get("speech"):
        pdf.para("График голоса не построен: данных о голосе нет.", 8)
    if charts.get("speech"):
        pdf.chart_block(charts["speech"], speech_cap)
    elif charts.get("voice"):
        pdf.para("График речи не построен: данных о темпе и паузах нет.", 8)
    if sp:
        items = _speech_cards(sp)
        pdf.ln(1)
        pdf.h3("Речь в цифрах", keep_mm=pdf.cards_height(items))
        pdf.cards(items)
        vocab = vocabulary_shown(report)          # Russian words; for English speech their translations
        if vocab:
            pdf.para("Частые слова (в скобках — сколько раз): " + ", ".join(f"{w} ({n})" for w, n in vocab[:15]), 8)


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


def _explain_section(pdf: Report, report: dict, explanation, frames: list, charts: dict, media: dict | None) -> None:
    """5. What drove the own model's score: key frames, modality contributions, words."""
    if "explain" not in pdf.plan:
        return
    seg = _rep_segment(report)
    tl_all = report.get("timeline") or []
    if seg and "timeline" in pdf.plan:
        intro = (f"Объяснения построены для своей модели MM-PSYCHE по отрезку {_seg(report, seg['start'], seg['end'])}, "
                 "ближайшему к среднему профилю (★ на графике «Big Five по ходу ролика»).")
    else:
        intro = "Объяснения построены для своей модели MM-PSYCHE по всему ролику."
    rows, cell_w, gap = _frame_rows(pdf, frames) if frames else ([], 0.0, 0.0)
    first_h = (max(h for _, _, h in rows[0]) + 6 + 6) if rows else 20
    pdf.section("Что повлияло на оценку своей модели", "explain", keep_mm=pdf.para_height(intro, 8.5) + first_h)
    pdf.para(intro, 8.5)
    if rows:
        # frames come from the clip the explanations were computed on: the representative segment of a long video,
        # otherwise the whole video; file names carry the frame index inside that clip (key_<i>_frame<N>.jpg)
        seg_any = next((s for s in tl_all if s.get("segment") == report.get("representative_segment")), None) if tl_all else None
        fps = float((media or {}).get("fps") or (report.get("media") or {}).get("fps") or 0)
        seg_start = float(seg_any["start"]) if seg_any else 0.0

        def moment(path: str):
            m = re.search(r"_frame(\d+)", Path(path).stem)
            return seg_start + int(m.group(1)) / fps if (m and fps > 0 and (seg_any or not tl_all)) else None

        # frames a fraction of a second apart would share «0:37»: then every caption shows tenths («0:37,2»)
        secs = [int(t) for t in map(moment, frames) if t is not None]
        tenths = len(set(secs)) < len(secs)
        timed = bool(secs)

        def caption(n: int, path: str) -> str:
            t = moment(path)
            if t is None:
                return f"кадр {n}"
            if tenths:
                d = int(t * 10)
                return f"кадр {n} · {d // 600}:{d // 10 % 60:02d},{d % 10}"
            return f"кадр {n} · {_mmss(t)}"

        note = ("Кадры, сильнее всего повлиявшие на оценку своей модели. Рамкой отмечено найденное лицо; "
                + (("под кадром — его номер и момент ролика (мин:с, после запятой — десятые доли секунды)." if tenths
                    else "под кадром — его номер и момент ролика (мин:с).") if timed else "под кадром — его номер."))
        pdf.h3("Ключевые кадры", keep_mm=max(h for _, _, h in rows[0]) + 6)
        n_done, per_row = 0, len(rows[0])
        for row in rows:
            row_h = max(h for _, _, h in row)
            if pdf.get_y() + row_h + 6 > pdf.page_break_trigger:
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
                pdf.set_xy(x_row + i * (cell_w + gap), y0 + row_h + 0.8)
                pdf.set_font("ui", "", 8); pdf.cell(cell_w, 4, caption(n_done + i + 1, q), align="C")
            n_done += len(row)
            pdf.set_xy(pdf.l_margin, y0 + row_h + 5.5)
        pdf.caption(note, 8)
    if charts.get("modalities"):
        cap = ("Какая доля оценки своей модели пришлась на каждую модальность (по градиенту оценки: насколько признаки "
               "модальности сдвигают результат); в каждой строке — 100%. «<1%» — модальность почти не влияет на оценку "
               "этого ролика: модель, обученная на FIV2, опирается в основном на лицо и голос.")
        pdf.chart_block(charts["modalities"], cap)
    rw_all = (explanation or {}).get("readable_words") or {}
    lang = (report.get("model") or {}).get("lang", "en")
    if rw_all:
        try:
            from .narrative import words_summary
            paras = words_summary(rw_all, explanation, TITLES, lang)
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


def _how_to_read(pdf: Report) -> None:
    texts = [DISCLAIMER_RU, INTERVIEW_DISCLAIMER_RU, HOW_TO_READ_EXTRA]
    pdf.section("Как читать результаты", "how_to_read", keep_mm=sum(pdf.para_height(t, 8) for t in texts))
    for t in texts:
        pdf.para(t, 8)


# ---------------------------------------------------------------- appendices
def _file_rows(report: dict, media: dict | None) -> list:
    rows = []
    if media:
        ch = media.get("channels")
        rows += [("Имя файла", media.get("file_name") or report.get("original_file_name")),
                 ("Размер", f"{media.get('size_mb')} МБ ({media.get('size_bytes')} байт)"),
                 ("Длительность", fmt_secs(media.get("duration_sec"))),
                 ("Контейнер", media.get("container")),
                 ("Видео", f"кодек {_codec(media.get('video_codec'))}, {media.get('width')}×{media.get('height')}, "
                           f"{media.get('fps')} кадра/с" + (f", поворот {media.get('rotation')}°" if media.get("rotation") else "")),
                 ("Аудио", f"кодек {_codec(media.get('audio_codec'))}, {media.get('sample_rate')} Гц, "
                           + ("моно" if ch == 1 else "стерео" if ch == 2 else f"каналов: {ch}")),
                 ("Битрейт", f"{media.get('bitrate_kbps')} кбит/с"), ("Файл изменён", _when(media.get("modified")))]
        for k, v in media.items():
            if k.startswith("tag_"):
                rows.append((MEDIA_TAGS.get(k[4:], k[4:]), _when(v) if k == "tag_creation_time" else
                             _encoder(v) if k == "tag_encoder" else v))
        if media.get("sha256"):
            rows.append(("Контрольная сумма SHA-256", media["sha256"]))
    else:
        rows.append(("Имя файла", report.get("original_file_name") or Path(report.get("input", "")).name))
    return rows


def _analysis_rows(report: dict) -> list:
    m = report.get("model", {})
    members = [x for x in report.get("modalities_used", []) if x in SYSTEM_TITLES]
    if m.get("backend") == "ensemble" and members:
        system = "ансамбль: " + " + ".join(SYSTEM_TITLES[x] for x in members)
        if m.get("primary"):
            system += f"; основная оценка — {SYSTEM_TITLES.get(m['primary'], m['primary'])}"
            if m.get("scale"):
                sc = str(m["scale"])
                sc = "MuPTA для русской речи" if sc.startswith("MuPTA") else (
                    "First Impressions V2" if sc.startswith("FIV2") else sc)
                system += f", шкала {sc}"
    else:
        system = f"{SYSTEM_TITLES.get(m.get('backend'), m.get('backend'))} ({m.get('corpus')})"
    rows = [("Система", system), ("Язык речи", LANG_RU.get(m.get("lang"), m.get("lang"))),
            ("Распознавание речи", _asr_ru(m.get("asr_model")) if m.get("asr_model") else "готовый транскрипт"),
            ("Обучающие данные", _trained_on(m, members)),
            ("Модальности", ", ".join(MODALITY_TITLES.get(x, x) for x in report.get("modalities_used", []))),
            ("Версия", _version_ru(m.get("version")))]
    if report.get("segments"):
        n_seg = int(report["segments"])
        rows.append(("Отрезки", f"{n_seg} по ~20 с; итог — среднее с весом по длительности" if n_seg > 1
                     else "один отрезок (весь ролик)"))
    t = report.get("timings_sec", {})
    if t:
        rows.append(("Время обработки", fmt_secs(t.get("total_wall", t.get("total")))))
    return rows


def _segment_rows(report: dict) -> list:
    """[(start, end, timeline entry or None, per_segment entry or None)] in time order, matched by the start."""
    by: dict = {}                       # rounded start -> [timeline entry, per_segment entry, start, end]
    for t in report.get("timeline") or []:
        by.setdefault(int(round(float(t["start"]))), [None, None, float(t["start"]), float(t["end"])])[0] = t
    for r in (report.get("analyses") or {}).get("per_segment") or []:
        by.setdefault(int(round(float(r["start"]))), [None, None, float(r["start"]), float(r["end"])])[1] = r
    return [(s, e, t, r) for _, (t, r, s, e) in sorted(by.items())]


def _dominant_text(r: dict | None, source: str) -> str:
    """«нейтрально 99%» as on the web; «нет речи» / «нет текста» for an empty transcript, «—» without data."""
    from .pdf_charts import dominant_emotion
    if not r:
        return "—"
    if source == "text" and _empty_text(r):
        return "нет речи" if _seg_words(r) == 0 else "нет текста"
    d = dominant_emotion(r, source)
    return f"{EMO_NAMES.get(d[0], d[0])} {d[1]:.0%}" if d else "—"


def _segments_table(pdf: Report, report: dict) -> None:
    rows_in = _segment_rows(report)
    tl_scored = _scored(report)
    keys = TRAIT_KEYS + (["interview"] if tl_scored and all("interview" in t["scores"] for t in tl_scored) else [])
    seg = _rep_segment(report)
    star = bool(seg)
    # the three voice columns hold «0.46» but were titled «Возбуж-/дение»: their headers alone took 43 of the 190 mm
    # and pushed the whole table below 7.5 pt in the longer videos. They are shortened like the trait columns and
    # spelled out in the legend under the table
    header = ["Отрезок"] + [SEG_HEAD[k] for k in keys] + ["по речи", "по лицу", "Возб.", "Увер.", "Позит.",
                                                          "Темп", "Паузы"]
    rows, any_no_text, any_skipped = [], False, False
    for s, e, t, r in rows_in:
        label = _seg(report, s, e) + (" ★" if seg and t is seg else "")
        if t and t.get("scores"):
            scores = [f"{float(t['scores'][k]):.2f}" if k in t["scores"] else "—" for k in keys]
        elif t:
            scores = ["пропущен"] + [""] * (len(keys) - 1); any_skipped = True
        else:
            scores = ["—"] * len(keys)
        v = (r or {}).get("voice") or {}
        sp = (r or {}).get("speech") or {}
        wpm = sp.get("words_per_min_speech")
        speech = _dominant_text(r, "text")
        any_no_text = any_no_text or speech == "нет текста"
        rows.append([label] + scores + [speech, _dominant_text(r, "face")]
                    + [f"{float(v[d]):.2f}" if v.get(d) is not None else "—" for d in ("arousal", "dominance", "valence")]
                    + [f"{float(wpm):.0f}" if wpm is not None else "—",
                       f"{float(sp.get('pause_share') or 0):.0%}" if sp else "—"])
    # column widths from the content: the widest header line (bold) or cell (regular) of each column plus the margins;
    # the cell margins are narrower than elsewhere (0.6 mm). When 14 columns still do not fit the text width, the
    # whole table goes one step smaller instead of being squeezed: squeezing wraps the titles one line further, and a
    # cell wider than its column runs over the rule next to it
    c_margin = pdf.c_margin
    pdf.c_margin = 0.6

    def col_widths(size: float) -> list:
        ws = []
        for j, h in enumerate(header):
            pdf.set_font("ui", "B", size)
            w = max(pdf.get_string_width(line) for line in str(h).split("\n"))
            pdf.set_font("ui", "", size)
            w = max([w] + [pdf.get_string_width(str(r[j])) for r in rows if str(r[j]) != "пропущен"])
            ws.append(w + 2 * pdf.c_margin + 0.4)
        return ws

    for size in (7.5, 7.2, 7.0):
        widths = col_widths(size)
        if sum(widths) <= pdf.epw:
            break
    groups = [("", 1), (("Big Five и «собеседование», 0…1" if "interview" in keys else "Big Five, 0…1"), len(keys)),
              ("Преобладающая эмоция", 2), ("Голос, 0…1", 3), ("Речь", 2)]
    pdf.table(header, rows, widths, size=size, groups=groups, row_h=4.8,
              cont_title=f"Приложение {pdf.appx.get('segments', 'Б')}. Значения по отрезкам (продолжение)")
    pdf.c_margin = c_margin
    legend = ", ".join(f"{_one_line(SEG_HEAD[k])} — {TITLES[k].lower() if k != 'interview' else 'впечатление «собеседование»'}"
                       for k in keys) + ". "
    legend += ("Возб. — возбуждение, Увер. — уверенность, Позит. — позитивность. "
               "Эмоция — преобладающая в отрезке и её доля. Темп — слов в минуту речи; «—» — речи в отрезке меньше 3 с. "
               "Паузы — доля времени отрезка, занятая паузами от 0.5 с.")
    if any_no_text:
        legend += (" «нет текста» — для отрезка не распознан текст, поэтому эмоция речи не оценена; темп и паузы берутся "
                   "из транскрипта всего ролика.")
    if any_skipped:
        legend += " «пропущен» — отрезок не оценён."
    if star:
        legend += " ★ — отрезок для объяснений."
    pdf.caption(legend, 7)


def _notable(report: dict, has_expl: bool) -> dict:
    """{start of a segment: [reasons]} of the notable segments, at most BEHAVIOR_MAX, chosen in this order: the segment
    for the explanations, segments with unusual scores (largest deviation first), segments whose dominant speech
    emotion or facial expression differs from the one of the whole video (largest share first)."""
    from .charts import EMO_RU
    from .narrative import odd_segments
    from .pdf_charts import dominant_emotion
    from .palette import EMO_ALIAS
    an = report.get("analyses") or {}
    per = an.get("per_segment") or []
    cands: list = []                    # (start, reason) in priority order
    seg = _rep_segment(report)
    if seg and has_expl:
        cands.append((float(seg["start"]), "отрезок для объяснений ★"))
    for t, z in sorted(odd_segments(report), key=lambda tz: -abs(tz[1])):
        cands.append((float(t["start"]), f"оценки {'выше' if z > 0 else 'ниже'} остального ролика"))
    te_dom = (an.get("emotions_text") or {}).get("dominant")
    speech = []
    for r in per:
        d = None if _empty_text(r) else dominant_emotion(r, "text")
        if d and te_dom and d[0] != te_dom:
            speech.append((d[1], float(r["start"]), "в речи нейтральный тон" if d[0] == "neutral" else
                           f"в речи преобладает {EMO_RU.get(d[0], d[0])}"))
    face_dom = EMO_ALIAS.get((an.get("face") or {}).get("dominant") or "", (an.get("face") or {}).get("dominant"))
    face = []
    for r in per:
        d = dominant_emotion(r, "face")
        if d and face_dom and d[0] != face_dom:
            face.append((d[1], float(r["start"]), "лицо нейтральное" if d[0] == "neutral" else
                         f"на лице преобладает {EMO_RU.get(d[0], d[0])}"))
    for group in (speech, face):
        cands += [(s, f"{why} ({p:.0%})") for p, s, why in sorted(group, key=lambda g: -g[0])]
    chosen: dict = {}
    for s, _ in cands:
        if s not in chosen and len(chosen) < BEHAVIOR_MAX:
            chosen[s] = []
    for s, why in cands:
        if s in chosen and why not in chosen[s]:
            chosen[s].append(why)
    return chosen


def _behavior_appendix(pdf: Report, report: dict, has_expl: bool) -> None:
    entries = behavior_by_segment(report)
    letter = pdf.appx["behavior"]
    n_all = max(len(entries), len(_segment_rows(report)))
    if not entries:                      # a description without segment labels is printed as it is
        pdf.h2(f"Приложение {letter}. Описание поведения", keep_mm=20)
        pdf.caption("Описание строит видеоязыковая модель по кадрам ролика.", 7.5)
        pdf.para(_dash(report.get("behavior_description_ru")), 8.5)
        return
    reasons = _notable(report, has_expl)
    if n_all <= BEHAVIOR_MAX:
        shown = entries
        title = f"Приложение {letter}. Описание поведения по отрезкам"
        note = "Описание строит видеоязыковая модель по кадрам каждого отрезка."
    else:
        starts = sorted(reasons)
        shown = [en for en in entries if any(abs(en[0] - s) < 1.5 for s in starts)]
        title = f"Приложение {letter}. Описание поведения в заметных отрезках"
        note = ("Описание строит видеоязыковая модель по кадрам каждого отрезка. Здесь — отрезок для объяснений, отрезки "
                "с необычными оценками и отрезки, где эмоция речи или выражение лица отличаются от преобладающих. "
                f"Описания всех {n_all} {plural_ru(n_all, 'отрезка', 'отрезков', 'отрезков')} — на веб-странице "
                "(вкладка «Объяснения») и в result.json.")
    first = shown[0] if shown else None
    pdf.h2(title, keep_mm=pdf.para_height(note, 7.5) + (pdf.para_height(first[2], 8.5) + 6 if first else 0))
    pdf.caption(note, 7.5)
    for s, e, body in shown:
        why = next((w for st, w in reasons.items() if abs(st - s) < 1.5), [])
        head = f"[{_seg(report, s, e)}]" + (" " + "; ".join(why) if why else "")
        if pdf.get_y() + 5 + min(pdf.para_height(body, 8.5), 13) > pdf.page_break_trigger:
            pdf.add_page()
        pdf.set_x(pdf.l_margin)
        pdf.set_font("ui", "B", 8.5); pdf.multi_cell(0, 4.4, head, new_x="LMARGIN", new_y="NEXT", align="L")
        pdf.para(_dash(body), 8.5)
        pdf.ln(0.5)


TRANSCRIPT_WIDOW_MM = 30            # below this much on the last page the transcript is set one step smaller


def _transcript_size(pdf: Report, text: str) -> float:
    """8.5 pt, or 8 pt when that keeps the tail of the transcript off a page of its own. The transcript is the last
    thing in the report and flows freely, so a few of its lines can land alone on a sheet that is then 90% white."""
    def widow(size: float) -> float:
        """mm of the text that would end up on a page with nothing else on it (0 = it ends on a shared page)."""
        h = pdf.para_height(text, size)
        left = pdf.page_break_trigger - pdf.get_y()
        if h <= left:
            return 0.0
        return (h - left) % (pdf.page_break_trigger - pdf.t_margin)
    w85 = widow(8.5)
    if not 0 < w85 < TRANSCRIPT_WIDOW_MM:
        return 8.5
    w80 = widow(8.0)
    return 8.0 if w80 == 0 or w80 > w85 else 8.5


def _appendices(pdf: Report, report: dict, media: dict | None, has_expl: bool) -> None:
    # the appendices do not start a page of their own: a forced break left the page before them three quarters empty
    # in every report. They begin here when the title and the first rows of appendix А still fit (8 mm for the title,
    # 10 mm for the heading of the appendix, 40 mm of its table), otherwise on the next page
    pdf.ln(3)
    if pdf.get_y() + 8 + 10 + 40 > pdf.page_break_trigger:
        pdf.add_page()
    pdf.set_font("ui", "B", 14); pdf.cell(0, 8, "Приложения", new_x="LMARGIN", new_y="NEXT")
    pdf.h2(f"Приложение {pdf.appx['file']}. Файл и параметры анализа", keep_mm=40)
    pdf.h3("Файл")
    pdf.kv_table(_file_rows(report, media))
    pdf.ln(1)
    pdf.h3("Параметры анализа")
    pdf.kv_table(_analysis_rows(report))
    if "segments" in pdf.appx:
        pdf.h2(f"Приложение {pdf.appx['segments']}. Значения по отрезкам", keep_mm=30)
        _segments_table(pdf, report)
    if "behavior" in pdf.appx:
        _behavior_appendix(pdf, report, has_expl)
    if "transcript" in pdf.appx:
        note, transcript = transcript_shown(report)
        # the recogniser can stop in the middle of a sentence; an ellipsis and one line say that the text really ends
        # there, so the last page of the report does not read as a fault of the export
        t = str(transcript or "").rstrip()
        cut = bool(t) and t[-1] not in ".!?…»)"
        if cut:
            t += "…"
        pdf.h2(f"Приложение {pdf.appx['transcript']}. Транскрипт речи",
               keep_mm=pdf.para_height(note, 7.5) + min(20, pdf.para_height(t, 8.5)))
        if note:
            pdf.caption(note, 7.5)
        if t:
            pdf.para(t, _transcript_size(pdf, t))
            if cut:
                pdf.caption("Распознавание речи оборвалось на этом месте.", 7.5)


# ---------------------------------------------------------------- the report
def _render(report: dict, explanation, media, frames: list, charts: dict, fname: str, total: int | None) -> Report:
    pdf = Report(file_label=fname, total_pages=total)
    _plan(pdf, report, explanation, frames, charts)
    has_expl = "explain" in pdf.plan
    pdf.add_page()
    pdf.h1("BS 2.0 — отчёт по видео: Big Five, эмоции, голос, речь")
    _passport(pdf, report, media, fname)
    facts = _pdf_facts(report, "voice_speech" in pdf.plan and bool((report.get("analyses") or {}).get("speech")))
    if facts:
        pdf.h3("Ключевые факты", keep_mm=pdf.cards_height(facts))
        pdf.cards(facts)
    if "timeline" not in pdf.plan:
        dur = float(report.get("duration_sec") or 0)
        pdf.para(("Ролик короче 30 с оценивается целиком" if 0 < dur <= 30 else "Ролик оценён целиком, одним отрезком")
                 + ", поэтому графиков по ходу ролика нет.", 8)
    _profile_section(pdf, report, explanation, charts)
    _timeline_section(pdf, report, charts, has_expl)
    _emotions_section(pdf, report, charts)
    _voice_speech_section(pdf, report, charts)
    _explain_section(pdf, report, explanation, frames, charts, media)
    _how_to_read(pdf)
    _appendices(pdf, report, media, has_expl)
    return pdf


def build_pdf(report: dict, out_path: str | Path, explanation: dict | None = None, media: dict | None = None,
              key_frames: list[str] | None = None) -> str:
    fname = report.get("original_file_name") or (media or {}).get("file_name") or Path(report.get("input", "")).name
    frames = [p for p in (key_frames or []) if os.path.exists(p)]
    charts = dict(report.get("chart_files") or {})
    if explanation and not charts.get("modalities") and charts:
        # save_pdf_charts called without the explanation (an older caller): the modality chart is drawn here
        try:
            from .pdf_charts import save_modalities_chart
            p = save_modalities_chart(explanation, Path(next(iter(charts.values()))).parent)
            if p:
                charts["modalities"] = p
        except Exception:  # noqa: BLE001
            pass
    # the page count of the footer («стр. 3 из 7») comes from a first layout pass; the second one is written
    total = _render(report, explanation, media, frames, charts, fname, None).pages_count
    pdf = _render(report, explanation, media, frames, charts, fname, total)
    out_path = Path(out_path)
    pdf.output(str(out_path))
    return str(out_path)
