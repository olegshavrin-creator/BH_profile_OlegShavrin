"""Text of the PDF report of BS Profiler 3.1 that is not drawn: segment ranges and times as the report prints them
(h:mm:ss from an hour on), the file and analysis parameters of appendix А in words (ISO times, codecs, the FFmpeg
writer, the Whisper model, a pre-release version), the column titles of «Значения по отрезкам», the score rows a
FIV2 percentile applies to, and the Russian dash in a description written by a model. No fpdf here.
"""
from __future__ import annotations

import re

from ..norms import RU_SHORT, RU_TITLES, TRAIT_KEYS
from ..textfmt import clock, seg_label


# «Значения по отрезкам» has 14 columns: the short trait names are broken over two lines where one line is wider than
# its column of numbers; the legend under the table joins the halves back («Добро-жел.» -> «Доброжел.»)
SEG_HEAD = {**RU_SHORT, "agreeableness": "Добро-\nжел.", "emotional_stability": "Эм.\nстаб.", "interview": "Собе-\nсед."}
MODALITY_TITLES = {"audio": "голос", "video": "видео", "text": "речь", "face": "лицо", "behavior": "описание поведения"}
# container tags from media.probe_media (ffprobe names) -> row titles of the «Файл» table
MEDIA_TAGS = {"creation_time": "Записан (метка в файле)", "encoder": "Программа записи",
              "com.apple.quicktime.make": "Производитель камеры", "com.apple.quicktime.model": "Модель камеры",
              "title": "Название"}
# training data of each model, for appendix А
TRAINED_ON = {"oceanai": "MuPTA (русская речь)", "mm": "First Impressions V2"}


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


def _asr_ru(name) -> str:
    """'openai/whisper-large-v3-turbo' -> 'Whisper large-v3-turbo': the model name without the hub prefix."""
    m = re.match(r"^(?:[\w.-]+/)?whisper-(.+)$", str(name or ""), re.I)
    return f"Whisper {m.group(1)}" if m else str(name or "")


def _version_ru(v) -> str:
    """'3.0.0a1' -> '3.0.0, альфа-версия 1' (PEP 440 pre-release suffixes a/b/rc in words)."""
    m = re.match(r"^(\d+(?:\.\d+)*)(a|b|rc)(\d+)$", str(v or ""))
    if not m:
        return str(v or "—")
    return f"{m.group(1)}, {({'a': 'альфа', 'b': 'бета', 'rc': 'предрелизная'})[m.group(2)]}-версия {m.group(3)}"


def _seg(report: dict, start, end) -> str:
    """«1:20–1:40» as textfmt.seg_label, but h:mm:ss from an hour on, like the time axis of the charts and the
    «Отрезок» column of the web table: one report never prints «1:05:00» on a chart and «65:00» in a table."""
    if float(report.get("duration_sec") or 0) < 3600:
        return seg_label(start, end)
    return f"{clock(start, hours=True)}–{clock(end, hours=True)}"


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


def _group_name(keys: list[str]) -> str:
    """Which score rows a FIV2 percentile applies to, for the note under the score bars."""
    if set(keys) == set(TRAIT_KEYS):
        return "пять черт"
    return ", ".join("«собеседование»" if k == "interview" else RU_TITLES[k].lower() for k in keys)
