"""The lettered appendices of the PDF report of BS Profiler 3.1 (design 10.7): «Файл и параметры анализа», «Значения по
отрезкам» (with the MBTI type of every segment), «Описание поведения» of the notable segments, and «Транскрипт речи».
The per-segment data come from segments.py; the cells «по речи … Паузы» are the ones the page shows (facts.segment_cells).
The plan (which appendices are printed and their letters) and the build are pdf/build.py; the page and its widgets are
pdf/document.py and pdf/widgets.py, the text helpers pdf/fmt.py.
"""
from __future__ import annotations

from pathlib import Path

from .. import MODALITIES, MODEL_TITLES, segments
from ..facts import segment_cells
from ..labels import EMO_RU, model_title
from ..norms import RU_TITLES, TRAIT_KEYS
from ..ru_texts import transcript_shown
from ..scores import shown_model
from ..segments import behavior_by_segment, dominant_emotion, empty_text, odd_segments, representative, segment_rows
from ..textfmt import fmt_secs, plural_ru
from .document import Report
from .fmt import (MEDIA_TAGS, MODALITY_TITLES, SEG_HEAD, TRAINED_ON, _asr_ru, _codec, _dash, _encoder, _one_line,
                  _seg, _version_ru, _when)
from .mbti_section import segment_types_by_start

BEHAVIOR_MAX = 6                    # appendix «Описание поведения»: at most this many notable segments


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
    """Appendix А, «Параметры анализа»: the one model of the report (3.1), speech recognition, training data, the
    modalities the model looked at (a job of 3.1 records them; a job of 3.0 recorded the member names instead, and
    those are replaced by the modalities of the shown model, bs3.MODALITIES), version, segments, time. The speech is
    always Russian, so no language row."""
    m = report.get("model") or {}
    main = shown_model(report)
    mods = [x for x in report.get("modalities_used") or [] if x not in MODEL_TITLES] or list(MODALITIES.get(main, ()))
    rows = [("Модель", model_title(main)),
            ("Распознавание речи", _asr_ru(m.get("asr_model")) if m.get("asr_model") else "готовый транскрипт"),
            ("Обучающие данные", TRAINED_ON.get(main) or str(m.get("trained_on") or "—")),
            ("Модальности", ", ".join(MODALITY_TITLES.get(x, x) for x in mods) or "—"),
            ("Версия", _version_ru(m.get("version")))]
    if report.get("segments"):
        n_seg = int(report["segments"])
        rows.append(("Отрезки", f"{n_seg} по ~20 с; итог — среднее с весом по длительности" if n_seg > 1
                     else "один отрезок (весь ролик)"))
    t = report.get("timings_sec") or {}
    if t:
        rows.append(("Время обработки", fmt_secs(t.get("total_wall", t.get("total")))))
    return rows


def _segments_table(pdf: Report, report: dict, mb: dict | None = None) -> None:
    rows_in = segment_rows(report)
    tl_scored = segments.scored(report)
    keys = TRAIT_KEYS + (["interview"] if tl_scored and all("interview" in t["scores"] for t in tl_scored) else [])
    seg = representative(report, min_scored=2)
    star = bool(seg)
    # the MBTI type of the main system on every segment, with X on the borderline axes (design 10.7, task T27)
    seg_types = segment_types_by_start(mb)
    mbti_col = bool(seg_types) and any(seg_types.values())
    # the three voice columns hold «0.46» but were titled «Возбуж-/дение»: their headers alone took 43 of the 190 mm
    # and pushed the whole table below 7.5 pt in the longer videos. They are shortened like the trait columns and
    # spelled out in the legend under the table
    header = (["Отрезок"] + [SEG_HEAD[k] for k in keys] + (["MBTI"] if mbti_col else [])
              + ["по речи", "по лицу", "Возб.", "Увер.", "Позит.", "Темп", "Паузы"])
    rows, any_no_text, any_skipped, any_no_primary = [], False, False, False
    for s, e, t, r in rows_in:
        label = _seg(report, s, e) + (" ★" if seg and t is seg else "")
        if t and t.get("scores"):
            scores = [f"{float(t['scores'][k]):.2f}" if k in t["scores"] else "—" for k in keys]
        elif t:
            # a segment without scores of the main system (clean view) or not scored at all: a dash in every column
            # (the word «пропущен» was wider than its column and ran over the next one)
            scores = ["—"] * len(keys)
            if t.get("no_primary"):
                any_no_primary = True
            else:
                any_skipped = True
        else:
            scores = ["—"] * len(keys)
        if mbti_col:
            scores.append(seg_types.get(int(round(float(s)))) or "—")
        cells = segment_cells(r)                   # the cells of the page's table (facts.segment_cells)
        any_no_text = any_no_text or cells[0] == "нет текста"
        rows.append([label] + scores + cells)
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
            w = max([w] + [pdf.get_string_width(str(r[j])) for r in rows])
            ws.append(w + 2 * pdf.c_margin + 0.4)
        return ws

    for size in (7.5, 7.2, 7.0):
        widths = col_widths(size)
        if sum(widths) <= pdf.epw:
            break
    groups = [("", 1), (("Big Five и «собеседование», 0…1" if "interview" in keys else "Big Five, 0…1"), len(keys))]
    groups += ([("Тип", 1)] if mbti_col else []) + [("Преобладающая эмоция", 2), ("Голос, 0…1", 3), ("Речь", 2)]
    pdf.table(header, rows, widths, size=size, groups=groups, row_h=4.8,
              cont_title=f"Приложение {pdf.appx.get('segments', 'Б')}. Значения по отрезкам (продолжение)")
    pdf.c_margin = c_margin
    legend = ", ".join(f"{_one_line(SEG_HEAD[k])} — {RU_TITLES[k].lower() if k != 'interview' else 'впечатление «собеседование»'}"
                       for k in keys) + ". "
    if mbti_col:
        legend += ("MBTI — тип отрезка, X — ось на границе (подробнее — в разделе "
                   f"{pdf.plan['mbti']}). " if "mbti" in pdf.plan else "MBTI — тип отрезка, X — ось на границе. ")
    if any_no_primary:
        legend += ("«—» в столбцах Big Five" + (" и MBTI" if mbti_col else "")
                   + f" — модель {MODEL_TITLES.get(shown_model(report), 'OCEAN-AI')} не дала оценки отрезка, он не "
                   "вошёл в основные оценки. ")
    legend += ("Возб. — возбуждение, Увер. — уверенность, Позит. — позитивность. "
               "Эмоция — преобладающая в отрезке и её доля. Темп — слов в минуту речи; «—» — речи в отрезке меньше 3 с. "
               "Паузы — доля времени отрезка, занятая паузами от 0.5 с.")
    if any_no_text:
        legend += (" «нет текста» — для отрезка не распознан текст, поэтому эмоция речи не оценена; темп и паузы берутся "
                   "из транскрипта всего ролика.")
    if any_skipped:
        legend += " «—» во всех столбцах Big Five — отрезок не оценён."
    if star:
        legend += " ★ — отрезок для объяснений."
    pdf.caption(legend, 7)


def _notable(report: dict, has_expl: bool) -> dict:
    """{start of a segment: [reasons]} of the notable segments, at most BEHAVIOR_MAX, chosen in this order: the segment
    for the explanations, segments with unusual scores (largest deviation first), segments whose dominant speech
    emotion or facial expression differs from the one of the whole video (largest share first)."""
    from ..palette import EMO_ALIAS
    an = report.get("analyses") or {}
    per = an.get("per_segment") or []
    cands: list = []                    # (start, reason) in priority order
    seg = representative(report, min_scored=2)
    if seg and has_expl:
        cands.append((float(seg["start"]), "отрезок для объяснений ★"))
    for t, z in sorted(odd_segments(report), key=lambda tz: -abs(tz[1])):
        cands.append((float(t["start"]), f"оценки {'выше' if z > 0 else 'ниже'} остального ролика"))
    te_dom = (an.get("emotions_text") or {}).get("dominant")
    speech = []
    for r in per:
        d = None if empty_text(r) else dominant_emotion(r, "text")
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
    n_all = max(len(entries), len(segment_rows(report)))
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


def _transcript_size(pdf: Report, text: str, extra_mm: float = 0.0) -> float:
    """8.5 pt, or 8 pt when that keeps the tail of the transcript off a page of its own. The transcript is the last
    thing in the report and flows freely, so a few of its lines can land alone on a sheet that is then 90% white.
    extra_mm: what follows the text (the line «Распознавание речи оборвалось…»), which must not be left alone either."""
    def widow(size: float) -> float:
        """mm of the text that would end up on a page with nothing else on it (0 = it ends on a shared page)."""
        h = pdf.para_height(text, size) + extra_mm
        left = pdf.page_break_trigger - pdf.get_y()
        if h <= left:
            return 0.0
        return (h - left) % (pdf.page_break_trigger - pdf.t_margin)
    w85 = widow(8.5)
    if not 0 < w85 < TRANSCRIPT_WIDOW_MM:
        return 8.5
    w80 = widow(8.0)
    return 8.0 if w80 == 0 or w80 > w85 else 8.5


CUT_NOTE = "Распознавание речи оборвалось на этом месте."
KV_LH = 8.5 * 0.61                  # line height of the two tables of appendix А (kv_table default at 8.5 pt)
KV_LH_TIGHT = 4.6                   # … set tighter so that a short transcript stays on the page of appendix А
SHORT_APPENDIX_MM = 30              # a transcript appendix below this is «short»: it never opens a page of its own


def _transcript_parts(report: dict) -> tuple[str, str, bool]:
    """(note, text, cut) of the transcript appendix. The recogniser can stop in the middle of a sentence; an ellipsis
    and one line (CUT_NOTE) say that the text really ends there, so the last page does not read as a fault."""
    note, transcript = transcript_shown(report)
    t = str(transcript or "").rstrip()
    cut = bool(t) and t[-1] not in ".!?…»)"
    return note or "", t + ("…" if cut else ""), cut


def _appendix_layout(pdf: Report, report: dict, media: dict | None, gap_top: float = 3.0) -> dict:
    """How appendix А and a transcript appendix that follows it directly are set, from the current position.

    Normal: the gaps and table lines of the report. A one-line transcript used to open a page of its own when
    appendix А ended a few millimetres too low (the heading rule of h2), leaving an A4 page with two lines on it. So
    when the transcript appendix is short (under SHORT_APPENDIX_MM) and does not fit under appendix А as it is,
    the two are measured tightened — smaller gaps above and between the tables, tighter table lines
    (KV_LH_TIGHT), the transcript at 8 pt — and set that way when that keeps them on one page (`tight`); when even
    that is not enough, the appendices start on the next page together (`new_page`), so the short appendix never
    stands alone. Returns {"tight", "new_page", "gap_top", "gap_mid", "lh", "size"}."""
    normal = dict(tight=False, new_page=False, gap_top=gap_top, gap_mid=1.0, lh=KV_LH, size=8.5)
    if "transcript" not in pdf.appx or "segments" in pdf.appx or "behavior" in pdf.appx:
        return normal
    note, t, cut = _transcript_parts(report)
    cut_h = (0.5 + pdf.para_height(CUT_NOTE, 7.5)) if cut else 0.0
    file_rows, analysis_rows = _file_rows(report, media), _analysis_rows(report)

    def heights(lay: dict) -> tuple[float, float]:
        """(appendix А with the title «Приложения», the transcript appendix) as _appendices prints them."""
        a = (lay["gap_top"] + 8 + 10 + 6 + pdf.kv_table_height(file_rows, lh=lay["lh"]) + lay["gap_mid"] + 6
             + pdf.kv_table_height(analysis_rows, lh=lay["lh"]))
        tr = 10 + ((0.5 + pdf.para_height(note, 7.5)) if note else 0.0) + pdf.para_height(t, lay["size"]) + cut_h
        return a, tr

    a, tr = heights(normal)
    left = pdf.page_break_trigger - pdf.get_y()
    if a + tr <= left or tr > SHORT_APPENDIX_MM:
        return normal
    tight = dict(tight=True, new_page=False, gap_top=min(gap_top, 1.0), gap_mid=0.0, lh=KV_LH_TIGHT, size=8.0)
    a2, tr2 = heights(tight)
    if a2 + tr2 <= left:
        return tight
    return {**normal, "new_page": True, "gap_top": 0.0}


def _appendices(pdf: Report, report: dict, media: dict | None, has_expl: bool, mb: dict | None = None) -> None:
    # the appendices do not start a page of their own: a forced break left the page before them three quarters empty
    # in every report. They begin here when the title and the first rows of appendix А still fit (8 mm for the title,
    # 10 mm for the heading of the appendix, 40 mm of its table), otherwise on the next page
    gap_top = 3.0
    if pdf.get_y() + gap_top + 8 + 10 + 40 > pdf.page_break_trigger:
        pdf.add_page()
        gap_top = 0.0
    lay = _appendix_layout(pdf, report, media, gap_top)
    if lay["new_page"]:                 # a short transcript that would stand alone on the last page otherwise
        pdf.add_page()
    pdf.ln(lay["gap_top"])
    pdf.set_font("ui", "B", 14); pdf.cell(0, 8, "Приложения", new_x="LMARGIN", new_y="NEXT")
    pdf.h2(f"Приложение {pdf.appx['file']}. Файл и параметры анализа", keep_mm=40)
    pdf.h3("Файл")
    pdf.kv_table(_file_rows(report, media), lh=lay["lh"])
    pdf.ln(lay["gap_mid"])
    pdf.h3("Параметры анализа")
    pdf.kv_table(_analysis_rows(report), lh=lay["lh"])
    if "segments" in pdf.appx:
        pdf.h2(f"Приложение {pdf.appx['segments']}. Значения по отрезкам", keep_mm=30)
        _segments_table(pdf, report, mb)
    if "behavior" in pdf.appx:
        _behavior_appendix(pdf, report, has_expl)
    if "transcript" in pdf.appx:
        note, t, cut = _transcript_parts(report)
        size = lay["size"] if lay["tight"] else 8.5
        pdf.h2(f"Приложение {pdf.appx['transcript']}. Транскрипт речи",
               keep_mm=pdf.para_height(note, 7.5) + min(20, pdf.para_height(t, size)))
        if note:
            pdf.caption(note, 7.5)
        if t:
            if not lay["tight"]:
                size = _transcript_size(pdf, t, (0.5 + pdf.para_height(CUT_NOTE, 7.5)) if cut else 0.0)
            pdf.para(t, size)
            if cut:
                pdf.caption(CUT_NOTE, 7.5)
