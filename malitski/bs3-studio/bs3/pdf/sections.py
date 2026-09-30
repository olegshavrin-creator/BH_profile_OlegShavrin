"""The main part of the PDF report of BS Profiler 3.1, section by section (design 10.7; 3.1: one model): the passport
of the file and the analysis, the Big Five profile (the radar beside «Как получены оценки», then the score bars), the
Big Five over time, emotions and facial expression, voice and speech with «Речь в цифрах», the one line that stands in
for the explanations of an OCEAN-AI job, and «Как читать результаты». The section numbers, and whether the optional
sections are printed at all, come from pdf/build.py (_plan, pdf.plan). The MBTI section is pdf/mbti_section.py; the
explanations with the key frames are pdf/frames.py.
"""
from __future__ import annotations

import datetime as _dt
import re

from .. import caveats
from ..analyses_text import analyses_parts
from ..facts import FER_NOTE, head_motion_word, speech_cards
from ..labels import VOICE_RU, model_title
from ..narrative import NO_EXPLAIN_RU
from ..norms import RU_SHORT, TRAIT_KEYS
from ..ru_texts import vocabulary_shown
from ..scores import has_explanations, scored, shown_model
from ..segments import empty_text, representative, seg_words
from ..textfmt import fix_counts, fmt_secs, plural_ru
from .document import Report
from .fmt import _hms_text, _seg
from .layout import RADAR_W_MM, ROW_GAP_MM


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
    analysis = [f"{mt.group(1)} {mt.group(2)}" if mt else created or None]
    n_seg = int(report.get("segments") or 0)
    analysis.append(f"{n_seg} {plural_ru(n_seg, 'отрезок', 'отрезка', 'отрезков')} по ~20 с" if n_seg > 1
                    else "ролик оценён целиком")
    t = report.get("timings_sec") or {}
    if t.get("total_wall", t.get("total")) is not None:
        analysis.append(f"время обработки {fmt_secs(t.get('total_wall', t.get('total')))}")
    # one model per analysis (3.1): the model of the view, «OCEAN-AI, веса MuPTA» or «AMLAI 1.0»
    rows = [("Файл", " · ".join(parts)), ("Анализ", " · ".join(p for p in analysis if p)),
            ("Модель", model_title(shown_model(report))),
            ("Отчёт", f"создан {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}; технические сведения о файле и анализе — "
                      f"в приложении {pdf.appx.get('file', 'А')}")]
    pdf.set_font("ui", "B", 8.5)
    w1 = max(pdf.get_string_width(k) for k, _ in rows) + 3
    pdf.kv_table(rows, w1=w1, size=8.5, lh=4.6)
    pdf.ln(2)


def _profile_section(pdf: Report, report: dict, explanation, charts: dict) -> None:
    """1. Radar beside «Как получены оценки» (narrative.method_notes on the clean view, design 9; its overflow
    continues under the row), then the score bars of the one model that ran (3.1)."""
    try:
        from ..narrative import method_notes
        narrative = _hms_text(fix_counts(method_notes(report)), report.get("duration_sec"))
    except Exception:  # noqa: BLE001
        narrative = ""
    radar = charts.get("profile")
    text_x = pdf.l_margin + RADAR_W_MM + ROW_GAP_MM
    text_w = pdf.l_margin + pdf.epw - text_x
    radar_h = pdf.chart_height(radar, RADAR_W_MM) if radar else 0.0
    interview = scored(report.get("interview"))       # an entry without a numeric score is not shown
    bars_h = pdf.score_bars_height(report["traits"], interview)

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
            pdf.set_font("ui", "B", 9); pdf.cell(text_w, 6, "Как получены оценки", new_x="LEFT", new_y="NEXT")
            pdf.set_xy(text_x, y0 + 7)
            pdf.set_font("ui", "", size)
            pdf.multi_cell(text_w, lh, "\n".join(lines[:beside]), align="L", new_x="LEFT", new_y="NEXT")
        pdf.set_xy(pdf.l_margin, y0 + row_h + 1)
        if rest:
            pdf.para(rest, size)
    elif narrative:
        pdf.h3("Как получены оценки")
        pdf.para(narrative, size)
    # ---- the score bars of the one model (3.1: no second opinion, no table of averaged members)
    pdf.h3("Оценки по чертам", keep_mm=bars_h)
    pdf.score_bars(report["traits"], interview)


def _timeline_section(pdf: Report, report: dict, charts: dict, has_expl: bool) -> None:
    if "timeline" not in pdf.plan:
        return
    std = report.get("scores_std_across_segments") or {}
    cap = ""
    if std:
        cap = "Разброс между отрезками: " + ", ".join(f"{RU_SHORT[k].lower()} ±{std.get(k, 0):.2f}" for k in TRAIT_KEYS) + "."
    seg = representative(report, min_scored=2)
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
    blocks = []                          # (chart key, caption, message when the chart is missing)
    prof_cap = ("Средняя доля каждой эмоции за ролик. Речь — модель эмоций текста по переводу транскрипта на английский; "
                "лицо — модель выражений по кадрам.")
    # an отрезок whose own transcript came out empty is «нет текста» everywhere else in the report, but the stored
    # average counts it as «нейтрально 100%»: say so, so the two views do not read as contradicting each other
    no_text = [r for r in per if empty_text(r)]
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
            cap.append(f"Движение головы {head_motion_word(hm, 'motion')}: смещение между кадрами — {hm:.0%} ширины "
                       "лица.")
        found = f"Лицо найдено в {fa['face_share']:.0%} кадров" if fa.get("face_share") is not None else ""
        if frames:
            n = len(per)
            found += ("; разобрано " if found else "Разобрано ") + (f"{frames} {plural_ru(frames, 'кадр', 'кадра', 'кадров')} "
                                                                   f"из {n} {plural_ru(n, 'отрезка', 'отрезков', 'отрезков')}")
        if found:
            cap.append(found + ".")
        cap.append(FER_NOTE.format(where=""))
        blocks.append(("face_expr", " ".join(cap), "График выражения лица не построен: лицо в кадре не найдено."))
    if len(per) >= 2:                    # one segment = the whole video: the averages above already say everything
        # hatched gaps. A segment with an empty transcript is not always silent: its own recognition may return nothing
        # while the whole-video transcript still has words in that window (tempo and pauses come from there), so
        # «нет речи» is said only when there are no words at all
        text_gaps = [r for r in per if empty_text(r) or not r.get("emotions_text")]
        gaps = []
        if text_gaps:
            gaps.append("по речи — " + ("в отрезке нет речи" if all(seg_words(r) == 0 for r in text_gaps)
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
        items = speech_cards(sp, small_rate_words=True)     # the cards of the page, «меньше 1 на 100 слов» in words
        pdf.ln(1)
        pdf.h3("Речь в цифрах", keep_mm=pdf.cards_height(items))
        pdf.cards(items)
        vocab = vocabulary_shown(report)          # Russian words; for English speech their translations
        if vocab:
            pdf.para("Частые слова (в скобках — сколько раз): " + ", ".join(f"{w} ({n})" for w, n in vocab[:15]), 8)


def _no_explain_note(pdf: Report, report: dict) -> None:
    """An OCEAN-AI job (3.1): one line under section 4 in place of section 5 — no empty section, no heading."""
    if "explain" in pdf.plan or has_explanations(report):
        return
    pdf.ln(1)
    pdf.caption(NO_EXPLAIN_RU, 8)


def _how_to_read(pdf: Report, report: dict) -> None:
    """«Как читать результаты»: the caveats caveats.PDF_HOW_TO_READ, word for word from caveats.py. C2 explains the
    label «собеседование» and is printed only when the report carries that label (a job of AMLAI 1.0)."""
    texts = [caveats.text(c) for c in caveats.PDF_HOW_TO_READ if c != "C2" or scored(report.get("interview"))]
    # 7.5 pt like the caveats of the MBTI section: six caveats instead of the three of 2.0
    pdf.section("Как читать результаты", "how_to_read", keep_mm=sum(pdf.para_height(t, 7.5) for t in texts))
    for t in texts:
        pdf.para(t, 7.5)
