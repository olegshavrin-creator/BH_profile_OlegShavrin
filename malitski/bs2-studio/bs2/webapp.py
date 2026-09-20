"""BS 2.0 web UI (Gradio): Big Five + emotions, voice, face and speech analytics with interactive charts.

Run:  bs2 web [--port 7870]      (inside WSL; open http://localhost:7870 on Windows). Independent of bs 1.0 (:7860).

Readability rules for the HTML blocks (both Gradio themes, see palette.py): text colours are inherited from the theme,
secondary text is the same colour at opacity .75 and at least 13 px, marks and outlines come from palette.HTML, and
every block shows its title (show_label=True, container=True).
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path

from .charts import (EMO_RU, VOICE_RU, fig_emotion_bars, fig_emotions_timeline, fig_face_expr, fig_radar,
                     fig_speech_timeline, fig_traits_timeline, fig_voice_timeline, plot_html as _plot_html)
from .narrative2 import analyses_sentences, fix_counts, key_facts, plural_ru
from .norms import TRAIT_KEYS
from .palette import (ACCENT, BUTTON_PRIMARY, BUTTON_PRIMARY_HOVER, BUTTON_STOP, BUTTON_STOP_HOVER, CARD_TINT,
                      HTML as PAL, PAGE_NOTE_OPACITY, SUBDUED_TEXT_LIGHT)
from .pipeline import Studio, run_analysis
from .report import DISCLAIMER_RU, INTERVIEW_DISCLAIMER_RU, fmt_secs, mmss_labels, seg_label
from .ru_texts import ensure_russian_job, transcript_shown, vocabulary_shown
from .webparts import (MEMBER_TITLES, NOTE, TRAIT_TITLES, _bar_html, _contrib_html, _members_html, _words_text,
                       table_html, th_text)

log = logging.getLogger("bs2.web")
# which OCEAN-AI weights were used, for the «Участники ансамбля» box
CORPUS_RU = {"mupta": "веса OCEAN-AI MuPTA для русской речи", "fi": "веса OCEAN-AI First Impressions V2 для английской речи"}
# metric cards: 1 px outline 3:1 on every background, light tint (palette.CARD_TINT, the background check_palette.py
# measures the card text and the outline on) so label, value and note read as one card
CARDS = "display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:8px"
CARD = (f"padding:10px 14px;border:1px solid {PAL['card_border']};background:rgba(128,128,128,{CARD_TINT});"
        "border-radius:8px;min-width:0")


def _cards(items) -> str:
    """(label, value, note) -> a grid of cards; note may be empty."""
    html = "".join(f"<div style='{CARD}'><div style='{NOTE}'>{lab}</div>"
                   f"<div style='font-size:20px;font-weight:600;line-height:1.25;margin:3px 0;"
                   f"font-variant-numeric:tabular-nums'>{val if val not in (None, '') else '—'}</div>"
                   + (f"<div style='{NOTE}'>{note}</div>" if note else "") + "</div>" for lab, val, note in items)
    return f"<div style='{CARDS}'>{html}</div>" if html else ""


def _facts_html(rep: dict) -> str:
    return _cards(key_facts(rep))


def _dominant(dist: dict) -> str:
    """«нейтрально 91%»: the dominant label with its share, so a weak and a clear dominance read differently."""
    if not dist:
        return "—"
    k, v = max(dist.items(), key=lambda kv: kv[1])
    return f"{EMO_RU.get(k, k)} {float(v):.0%}"


def _clock(sec: float, hours: bool) -> str:
    s = int(sec)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if hours else f"{s // 60}:{s % 60:02d}"


def _segments_table(rep: dict) -> str:
    per = (rep.get("analyses") or {}).get("per_segment") or []
    if not per:
        return ""
    t_max = max(float(r["end"]) for r in per)
    # one time format for the whole column («0:00–0:20 … 10:00–10:12»), the same as on the chart time axes
    if t_max >= 60:
        hours = t_max >= 3600
        seg_head = th_text("Отрезок", "ч:мин:с" if hours else "мин:с")
        when = [f"{_clock(r['start'], hours)}–{_clock(r['end'], hours)}" for r in per]
    else:
        seg_head = th_text("Отрезок", "мин:с")
        when = [seg_label(r["start"], r["end"]) for r in per]
    head = [seg_head, th_text("Эмоция", "по тексту речи"), th_text("Выражение", "лица"),
            th_text("Возбуждение", "голос, 0…1"), th_text("Уверенность", "голос, 0…1"), th_text("Позитивность", "голос, 0…1"),
            th_text("Темп", "слов в минуту"), th_text("Доля пауз", "в отрезке")]
    rows = []
    for r, w in zip(per, when):
        vo = r.get("voice") or {}
        sp = r.get("speech") or {}
        rows.append([w, _dominant(r.get("emotions_text") or {}),
                     _dominant((r.get("face") or {}).get("expressions") or {}),
                     *(f"{vo[d]:.2f}" if vo.get(d) is not None else "—" for d in ("arousal", "dominance", "valence")),
                     f"{sp.get('words_per_min_speech') or 0:.0f}" if sp else "—",
                     f"{sp.get('pause_share', 0):.0%}" if sp else "—"])
    return (f"<div style='{NOTE};margin-bottom:8px'>Для каждого отрезка: преобладающая эмоция по тексту речи и по лицу "
            "(с долей), три характеристики голоса от 0 до 1, темп речи и доля пауз. Шапка таблицы остаётся на месте "
            "при прокрутке.</div>" + table_html(head, rows, max_height=480))


def _speech_html(rep: dict) -> str:
    sp = (rep.get("analyses") or {}).get("speech") or {}
    if not sp:
        return ""

    def whole(v):
        return "—" if v is None else f"{float(v):.0f}"

    fillers = sp.get("fillers")
    items = [("Слов всего", whole(sp.get("words")), ""),
             ("Разных слов", whole(sp.get("unique_words")), "без повторов"),
             ("Темп речи, слов в минуту", whole(sp.get("words_per_min_speech")), "только время, когда человек говорит"),
             ("Темп с учётом пауз, слов в минуту", whole(sp.get("words_per_min_wall")), "по всей длине ролика"),
             ("Доля пауз", f"{sp.get('pause_share', 0):.0%}", "паузы от 0.5 с, доля времени ролика"),
             ("Длинных пауз", whole(sp.get("long_pauses")), "дольше 2 секунд"),
             ("Слов-заполнителей", whole(fillers),
              f"{float(sp.get('fillers_per_100') or 0):.0f} на 100 слов" if fillers is not None else ""),
             ("Слов во фразе", whole(sp.get("mean_sentence")), "в среднем"),
             ("Разнообразие словаря", f"{float(sp['ttr']):.0%}" if sp.get("ttr") is not None else "—",
              "доля разных слов среди всех; зависит от длины текста")]
    vocab = ", ".join(f"{w} ({n})" for w, n in vocabulary_shown(rep)[:15])       # in Russian for any speech language
    return (_cards(items) + "<p style='font-size:15px;line-height:1.5;margin:12px 0 6px'>"
            f"{fix_counts(sp.get('description', ''))}</p>"
            + (f"<p style='font-size:14px;line-height:1.5;margin:0'><b>Частые слова</b> "
               f"<span style='opacity:.75'>(в скобках — сколько раз)</span>: {vocab}</p>" if vocab else ""))


def _face_html(rep: dict) -> str:
    """Cards with the face metrics; the distribution itself is drawn as a chart next to them."""
    fa = (rep.get("analyses") or {}).get("face") or {}
    if not fa:
        return ""
    m = fa.get("mean") or {}
    per = (rep.get("analyses") or {}).get("per_segment") or []
    frames = sum((r.get("face") or {}).get("frames", 0) for r in per)
    cards = []
    if m:
        k, v = max(m.items(), key=lambda kv: kv[1])
        cards.append(("Выражение лица чаще всего", EMO_RU.get(k, k), f"{v:.0%} кадров"))
    hm = fa.get("head_motion")
    if hm is not None:
        cards.append(("Движение головы", "слабое" if hm < 0.05 else ("умеренное" if hm < 0.15 else "активное"),
                      f"смещение между кадрами — {hm:.0%} ширины лица"))
    if fa.get("face_share") is not None:
        cards.append(("Лицо найдено", f"{fa['face_share']:.0%}", "доля разобранных кадров"))
    if frames:
        n = len(per)
        # «взяты из 31 отрезка», not «372 / из 31 отрезка», which reads like a fraction
        cards.append(("Кадров разобрано", f"{frames}", f"взяты из {n} {plural_ru(n, 'отрезка', 'отрезков', 'отрезков')}"))
    return (_cards(cards) + f"<p style='{NOTE};margin-top:10px'>Модель выражений обучена на фотографиях FER-2013 и "
            "склонна видеть «грусть» и «страх» в спокойном лице: смотрите на изменения по ходу ролика (вкладка «Таймлайн»), "
            "а не на абсолютные доли.</p>")


# key frames: the figure toggles .bs2-kf-big; enlarged, the image fills the window and the caption (moment of the
# video) stays readable on a dark plate at the bottom, so it is always clear which moment is shown
FRAMES_CSS = (
    "<style>"
    ".bs2-kf figure{margin:0!important;cursor:zoom-in}"
    # thumbnails keep the frame's own proportions (no empty letterbox bands); tall portrait frames stop at 320 px
    ".bs2-kf img{display:block;width:100%;height:auto;max-height:320px;object-fit:contain;border-radius:8px;"
    "background:rgba(128,128,128,.12);outline:1px solid " + PAL["card_border"] + ";outline-offset:-1px}"
    ".bs2-kf figcaption{text-align:center;font-size:14px;font-weight:600;margin-top:6px;font-variant-numeric:tabular-nums}"
    ".bs2-kf figure.bs2-kf-big{cursor:zoom-out}"
    ".bs2-kf figure.bs2-kf-big img{position:fixed;inset:4vh 4vw;width:92vw;height:92vh;max-height:none;z-index:9999;"
    "border-radius:8px;background:rgba(0,0,0,.92);outline:0;box-shadow:0 0 0 100vmax rgba(0,0,0,.85)}"
    ".bs2-kf figure.bs2-kf-big figcaption{position:fixed;left:50%;bottom:calc(4vh + 14px);transform:translateX(-50%);"
    "z-index:10000;margin:0;padding:6px 14px;border-radius:8px;background:rgba(0,0,0,.8);color:#fff!important;"
    "font-size:16px;white-space:nowrap;pointer-events:none}"
    ".bs2-kf figure:not(.bs2-kf-big) .bs2-kf-more{display:none}"
    "</style>")


def _frames_html(rep: dict, max_side: int = 640) -> str:
    """Key frames embedded as data-URI JPEGs. gr.Gallery depends on Gradio serving files from the job folder, which
    proved unreliable in this setup (images arrive broken); inline images always render. Click enlarges a frame."""
    import base64
    import io
    from PIL import Image

    paths = [p for p in rep.get("key_frames") or [] if Path(p).exists()]
    if not paths:
        return "<p style='font-size:14px'>Ключевые кадры не построены (объяснения отключены или лицо не найдено).</p>"
    # frames come from the representative segment of a long video, otherwise from the whole video (no timeline): the
    # moment is then counted from 0, the same rule as in the PDF
    tl_all = rep.get("timeline") or []
    seg = next((t for t in tl_all if t.get("segment") == rep.get("representative_segment")), None) if tl_all else None
    fps = float((rep.get("media") or {}).get("fps") or 0) or None
    seg_start = float(seg["start"]) if seg else 0.0
    cells = []
    for p in paths:
        try:
            im = Image.open(p).convert("RGB")
            im.thumbnail((max_side, max_side))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=82)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:  # noqa: BLE001
            continue
        m = re.search(r"_frame(\d+)", Path(p).stem)
        t = seg_start + int(m.group(1)) / fps if (m and fps and (seg or not tl_all)) else None
        cells.append((b64, t))
    # frames a fraction of a second apart would get the same «0:37» twice: then show tenths («0:37,2»)
    whole = [f"{int(t) // 60}:{int(t) % 60:02d}" for _, t in cells if t is not None]
    tenths = len(set(whole)) < len(whole)

    def _moment(t):
        if not tenths:
            return f"{int(t) // 60}:{int(t) % 60:02d}"
        d = int(t * 10)
        return f"{d // 600}:{d // 10 % 60:02d},{d % 10}"

    total = len(cells)
    figs = []
    for i, (b64, t) in enumerate(cells, 1):
        # with a moment: «0:37» under the frame; without one (no frame rate in the file): the frame's number in the row
        if t is not None:
            caption, more, tail, alt = _moment(t), f"Кадр {i} из {total} · момент ", "", f"Ключевой кадр, момент {_moment(t)}"
        else:
            caption, more, tail, alt = f"кадр {i}", "", f" из {total}", f"Ключевой кадр {i} из {total}"
        figs.append(
            f"<figure role='button' tabindex='0' title='Щёлкните, чтобы увеличить' onclick=\"this.classList.toggle('bs2-kf-big')\" "
            "onkeydown=\"if(event.key==='Enter'||event.key===' '){event.preventDefault();this.classList.toggle('bs2-kf-big')}"
            "else if(event.key==='Escape'){this.classList.remove('bs2-kf-big')}\">"
            f"<img src='data:image/jpeg;base64,{b64}' alt='{alt}'>"
            f"<figcaption><span class='bs2-kf-more'>{more}</span>{caption}"
            f"<span class='bs2-kf-more'>{tail} · щелчок закрывает</span></figcaption></figure>")
    timed = any(t is not None for _, t in cells)
    where = f" (отрезок {seg_label(seg['start'], seg['end'])})" if seg else ""
    return (FRAMES_CSS + "<div class='bs2-kf' style='display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));"
            f"gap:12px'>{''.join(figs)}</div>"
            f"<p style='{NOTE};margin-top:10px'>Кадры, сильнее всего повлиявшие на оценку своей модели{where}. "
            + ("Рамкой на кадре отмечено найденное лицо, подпись под кадром — момент ролика (минуты:секунды"
               + (", после запятой — десятые доли секунды" if tenths else "") + "). " if timed else
               "Рамкой на кадре отмечено найденное лицо, подпись под кадром — его номер. ")
            + "Щелчок по кадру увеличивает его, повторный щелчок закрывает.</p>")


def page_outputs(rep: dict) -> tuple:
    """Everything the result page shows for a finished job, in the order of the output blocks after the status line
    (without the PDF button). Jobs processed before the Russian texts existed get them here, stored back."""
    job = Path(rep["job_dir"])
    expl_path = job / "explain" / "explanation.json"
    expl = json.loads(expl_path.read_text(encoding="utf-8")) if expl_path.exists() else None
    ensure_russian_job(job, rep, expl)
    lang = (rep.get("model") or {}).get("lang", "ru")
    narrative = fix_counts(rep.get("narrative") or "") + " " + analyses_sentences(rep)
    members = rep.get("variant_scores") or {}
    primary = (rep.get("model") or {}).get("primary")
    role = lambda m: " — основная оценка" if m == primary else (" — второе мнение" if primary else "")
    member_txt = "\n".join(f"{MEMBER_TITLES.get(m, m)}{role(m)}: "
                           + ", ".join(f"{TRAIT_TITLES[k].lower()} {v[k]:.2f}" for k in TRAIT_KEYS if k in v) + "."
                           for m, v in members.items())
    # model.corpus of an ensemble is a technical descriptor; the OCEAN-AI weights follow the language
    weights = CORPUS_RU["mupta" if (rep.get("model") or {}).get("lang") == "ru" else "fi"] if "oceanai" in members else ""
    member_txt += (f"\nОбработка заняла {fmt_secs(rep['timings_sec'].get('total_wall', 0))}"
                   + (f"; {weights}." if weights else "."))
    # English speech: the Russian translation with a one-line note (the original stays in result.json)
    note, transcript = transcript_shown(rep)
    # fill=True: charts in a row of two windows grow to the height of the window next to them (see APP_CSS)
    return (_plot_html(fig_radar, rep, fill=True), _bar_html(rep["traits"], rep.get("interview")) + _members_html(rep),
            _facts_html(rep), narrative.strip(), _plot_html(fig_traits_timeline, rep),
            _plot_html(fig_emotions_timeline, rep), _plot_html(fig_voice_timeline, rep, fill=True),
            _plot_html(fig_speech_timeline, rep, fill=True), _plot_html(fig_emotion_bars, rep),
            _segments_table(rep), _speech_html(rep),
            "\n\n".join(t for t in (note, transcript) if t), _face_html(rep), _plot_html(fig_face_expr, rep),
            _frames_html(rep), _contrib_html(expl),
            _words_text(expl, rep, lang, expl_path) if expl else "",
            mmss_labels(rep.get("behavior_description_ru") or ""), member_txt,
            json.dumps(rep, ensure_ascii=False, indent=2), str(job / "result.json"), str(job))


def export_pdf(job_dir: str | Path) -> str:
    from .pdf_charts import save_pdf_charts
    from .media import probe_media
    from .pdf_report import build_pdf
    job = Path(job_dir)
    rep = json.loads((job / "result.json").read_text(encoding="utf-8"))
    expl_path = job / "explain" / "explanation.json"
    expl = json.loads(expl_path.read_text(encoding="utf-8")) if expl_path.exists() else None
    ensure_russian_job(job, rep, expl)          # jobs processed before the Russian texts: translate once, store back
    rep["chart_files"] = save_pdf_charts(rep, job / "charts", expl)
    frames = sorted(str(p) for p in (job / "explain").glob("key_*.jpg")) if (job / "explain").exists() else []
    media = rep.get("media")
    if not media or "error" in media:
        inp = next(job.glob("input.*"), None)
        media = probe_media(inp) if inp else None
    stem = re.sub(r"[^A-Za-z0-9А-Яа-яЁё._-]+", "_", Path(rep.get("original_file_name") or "video").stem)[:60]
    return build_pdf(rep, job / f"BS2_report_{stem}.pdf", explanation=expl, media=media, key_frames=frames)


def analysis_error_ru(e: BaseException) -> str:
    """A failed analysis in words for the error dialog: the exceptions of the models are English (and show_error=True
    would print them as they are); the original goes to the server log."""
    import subprocess
    msg = str(e)
    low = msg.lower()
    if "out of memory" in low:
        return "Не хватило памяти видеокарты. Подождите минуту и запустите анализ заново."
    if "no segment could be analysed" in low or "no predictions for any file" in low or "no frames decoded" in low:
        return "В ролике не найдено ни лица, ни речи, поэтому оценить его нельзя. Проверьте файл и выбранный язык речи."
    if "all ensemble members failed" in low:
        return ("Ни одна из систем оценки не смогла обработать ролик: чаще всего в кадре не найдено лицо или не слышна "
                "речь. Проверьте файл и выбранный язык речи.")
    if isinstance(e, subprocess.CalledProcessError):
        return "Не удалось прочитать видеофайл: возможно, он повреждён или записан в неподдерживаемом формате."
    if re.search(r"[А-Яа-яЁё]", msg) and not re.search(r"[A-Za-z]", msg):
        return msg                                  # messages of this package are already Russian
    return "Не удалось обработать ролик из-за внутренней ошибки. Подробности записаны в журнал сервера."


STATUS_LABELS = {"running": "Идёт обработка", "done": "Готово", "stopped": "Остановлено", "error": "Ошибка"}


def _live_desc(state: dict) -> str:
    """«[1 мин 05 с] Отрезок 3/29 (0:40–1:00)»: the time since the start, counted anew on every refresh, so it keeps
    running during a long stage such as model loading (stage messages carry the time at which the stage began)."""
    e = time.time() - state["t0"]
    when = f"{int(e)} с" if e < 60 else fmt_secs(e)
    return f"[{when}] " + re.sub(r"^\[[^\]]*\]\s*", "", str(state["desc"]))


def _status_html(frac: float, desc: str, state: str = "running", label: str | None = None) -> str:
    """Progress line above the page. state: running (orange, «34% · …»), done (green), stopped or error (red).
    The text shows the real percentage (0% at the start); the bar keeps a 2% minimum width so it is visible.
    The other states fill the whole outlined track in their colour; the label and a colour dot name the state."""
    state = state if state in STATUS_LABELS else "running"
    pct = max(0, min(100, int(round(float(frac) * 100))))
    color = PAL[f"status_{state}"]
    head = label or STATUS_LABELS[state]
    if state == "running":
        width, txt = max(2, pct), f"{pct}% · {desc}"
    else:
        width, txt = 100, desc
    return (f"<div role='status' aria-live='polite' style='margin:4px 0 8px'>"
            "<div style='display:flex;flex-wrap:wrap;justify-content:space-between;align-items:baseline;gap:2px 12px;"
            "font-size:14px;margin-bottom:5px'>"
            f"<b style='white-space:nowrap'><span aria-hidden='true' style='display:inline-block;width:10px;height:10px;"
            f"border-radius:50%;background:{color};margin-right:7px'></span>{head}</b> "
            f"<span style='text-align:right;font-variant-numeric:tabular-nums'>{txt}</span></div>"
            "<div role='progressbar' aria-valuemin='0' aria-valuemax='100' "
            # stopped and error fill the track without having got that far: no aria-valuenow, the text says what happened
            + (f"aria-valuenow='{pct}' " if state not in ("stopped", "error") else "") + f"aria-label='{head}' "
            f"style='height:10px;border-radius:6px;background:{PAL['track']};box-shadow:inset 0 0 0 1px {PAL['track_outline']}'>"
            f"<div style='width:{width}%;height:10px;border-radius:6px;background:{color};transition:width .5s'></div>"
            "</div></div>")


# the Code icon Gradio puts into every gr.HTML label chip reads as «code»: hide it on the result blocks.
# Chart blocks: charts.plot_html puts the chart title (bold 15 px div) and its subtitle above the iframe; the block chip
# already shows the same title, so the bold title line is hidden there and the subtitle (units, scale) stays.
APP_CSS = (".bs2-block > label[data-testid='block-label'] > span{display:none}"
           ".prose.bs2-chart > div:first-child[style*='font-weight:600']{display:none}")
COPYRIGHT = "© 2026 AMLAI"
# the copyright line under the title: 12 px, theme text colour dimmed (8.8:1 on the dark page, 5.6:1 on white), pulled
# up to the title so it reads as part of the heading, with the usual gap before the description line
APP_CSS += (f".gradio-container .prose p.bs2-copy{{font-size:12px;line-height:16px;letter-spacing:.03em;"
            f"opacity:{PAGE_NOTE_OPACITY};margin:-8px 0 12px}}")
# Gradio paints several marks with the theme accent, orange-500: the name and the underline of the selected tab (and
# the «…» button of the tabs that do not fit on a narrow screen), the ring of the upload progress and the icons of the
# video player. On white orange-500 is 2.8:1, so the whole page takes the accent of palette.ACCENT instead (orange-700
# in the light theme, as the checked boxes in _theme). The theme sets the variable on <html> and on body.dark, so a
# rule on the container wins for everything inside it in both themes.
APP_CSS += (f".gradio-container{{--color-accent:{ACCENT['light']}}}"
            f".dark .gradio-container{{--color-accent:{ACCENT['dark']}}}")
# result.json: the code viewer colours its tokens with fixed pale hues (2.0-2.8:1 on white), so the file is shown in the
# theme text colour. Both themes, although the dark hues do reach 4.5:1: one grey page, and the same block either way
APP_CSS += ".bs2-json .cm-content span{color:inherit!important}"
VIDEO_H = 300          # height of the video window before it is stretched (gr.Video height)
# Two framed windows side by side (gr.Row with class bs2-pair): both columns get the height of the taller one, and in
# each column the window marked bs2-grow takes the extra height, so the two frames start and end on one line with no
# page background under the shorter one. A text box stretches its text area (the text fills the frame instead of
# scrolling in a small box), a chart block stretches its iframe (charts.plot_html fill=True redraws the figure at that
# height), the video window stretches its drop zone or player, a plain HTML block extends its frame under the content.
# Gradio's own equal_height grows every block of a column; here only the marked one grows. On a narrow screen the
# columns wrap onto separate lines, and a line is as tall as its only column, so nothing is stretched there.
APP_CSS += (
    ".gradio-container .row.bs2-pair{align-items:stretch}"
    ".row.bs2-pair>.column>.bs2-grow,.row.bs2-pair>.column>.form>.bs2-grow{flex-grow:1}"
    # a text box sits in a .form wrapper whose inline style (flex-grow:0, from scale) only !important overrides
    ".row.bs2-pair>.column>.form:has(>.bs2-grow){flex-grow:1!important;flex-wrap:nowrap}"
    # text box: block > label.container > title chip + .input-container > textarea
    ".row.bs2-pair .bs2-grow.block:has(textarea),.row.bs2-pair .bs2-grow>label.container,"
    ".row.bs2-pair .bs2-grow .input-container{display:flex;flex-direction:column;flex-grow:1}"
    ".row.bs2-pair .bs2-grow>label.container>[data-testid='block-info']{align-self:flex-start}"
    ".row.bs2-pair .bs2-grow textarea{flex-grow:1}"
    # chart: block > title chip + .html-container > .prose > subtitle + iframe (flex:1 0 auto from plot_html); a chart
    # that stops growing (the radar, max-height from the iframe script) is centred together with its subtitle
    ".row.bs2-pair>.column>.bs2-grow.bs2-chart,.row.bs2-pair .bs2-grow.bs2-chart>.html-container,"
    ".row.bs2-pair .prose.bs2-grow.bs2-chart{display:flex;flex-direction:column;flex-grow:1}"
    ".row.bs2-pair .prose.bs2-grow.bs2-chart{justify-content:center}"
    ".row.bs2-pair>.column>.bs2-grow.bs2-chart>label{align-self:flex-start}"
    # video: Gradio fixes the block height inline (height=VIDEO_H)
    f".row.bs2-pair>.column>.bs2-grow:has(.video-container){{height:auto!important;min-height:{VIDEO_H}px;"
    "display:flex;flex-direction:column}"
    ".row.bs2-pair .bs2-grow .video-container{flex-grow:1}")


def _theme():
    """The BS 1.0 look: Gradio Default theme (zinc neutrals, Source Sans Pro, block titles in the block corner) with
    readable buttons and hints: white labels on orange-700 / red-700 (the Default orange-500 gives 2.8:1, red-500 3.8:1),
    zinc-600 hint text on white (zinc-400 gives 2.6:1), and checked boxes and radios in orange-700 on white, where the
    Default orange-500 is 2.8:1 (the dark theme keeps orange-500). The selected tab underline follows in APP_CSS."""
    import gradio as gr
    return gr.themes.Default().set(
        button_primary_background_fill=BUTTON_PRIMARY, button_primary_background_fill_dark=BUTTON_PRIMARY,
        button_primary_background_fill_hover=BUTTON_PRIMARY_HOVER, button_primary_background_fill_hover_dark=BUTTON_PRIMARY_HOVER,
        button_cancel_background_fill=BUTTON_STOP, button_cancel_background_fill_dark=BUTTON_STOP,
        button_cancel_background_fill_hover=BUTTON_STOP_HOVER, button_cancel_background_fill_hover_dark=BUTTON_STOP_HOVER,
        body_text_color_subdued=SUBDUED_TEXT_LIGHT, body_text_color_subdued_dark="*neutral_400",
        # placeholders have their own variable (light zinc-400 2.6:1 on white, dark zinc-500 3.1:1 on the block)
        input_placeholder_color=SUBDUED_TEXT_LIGHT, input_placeholder_color_dark="*neutral_400",
        checkbox_background_color_selected=ACCENT["light"], checkbox_background_color_selected_dark=ACCENT["dark"],
        checkbox_border_color_selected=ACCENT["light"], checkbox_border_color_selected_dark=ACCENT["dark"],
        checkbox_border_color_focus=ACCENT["light"], checkbox_border_color_focus_dark=ACCENT["dark"],
    )


ERROR_TITLE = "Ошибка"          # title of Gradio's error dialog (its default is the English "Error")
# Gradio's own texts (upload area «Перетащите видео сюда», buttons, footer) come from its translations and follow the
# browser language, so an English browser showed them in English. The page reports a Russian browser before the
# Gradio bundle reads navigator.language; this has to run before that bundle, which the `head` of gr.Blocks does not
# (Gradio inserts it later, after its translations are set up).
RU_LOCALE_JS = ("<script>try{['language','languages'].forEach(function(k){Object.defineProperty(Navigator.prototype,k,"
                "{configurable:true,get:function(){return k==='language'?'ru-RU':['ru-RU','ru'];}});});}catch(e){}</script>")


def force_russian_gradio() -> None:
    """Serve Gradio's index page with RU_LOCALE_JS at the top of <head> and lang="ru" (Gradio 5 renders the page
    from a Jinja template; its loader is wrapped once per process)."""
    import jinja2
    from gradio import routes
    env = routes.templates.env
    if getattr(env.loader, "bs2_russian", False):
        return
    base = env.loader

    class RussianIndexLoader(jinja2.BaseLoader):
        bs2_russian = True

        def get_source(self, environment, template):
            source, filename, uptodate = base.get_source(environment, template)
            if template.endswith("index.html") and RU_LOCALE_JS not in source:
                source = source.replace("<head>", "<head>" + RU_LOCALE_JS, 1).replace('lang="en"', 'lang="ru"', 1)
            return source, filename, uptodate

    env.loader = RussianIndexLoader()


def build_app(studio: Studio, work_dir: Path, preview_job: str | None = None):
    """preview_job: a finished job folder rendered on page load (UI testing without re-running the analysis)."""
    import gradio as gr
    from .longvideo import AnalysisCancelled

    N_REST = 23

    def render(rep: dict) -> tuple:
        return page_outputs(rep) + (gr.update(interactive=True),)

    def analyze(video, lang, explain):
        if not video:
            raise gr.Error("Загрузите видео", title=ERROR_TITLE)
        state = {"frac": 0.0, "desc": "запуск", "t0": time.time()}

        def cb(frac, desc=None, **kw):
            state["frac"] = float(frac)
            state["desc"] = str(desc if desc is not None else kw.get("desc", "")) or state["desc"]

        result: dict = {}

        def work():
            try:
                result["r"] = run_analysis(studio, work_dir, video, lang, explain, progress=cb)
            except BaseException as e:  # noqa: BLE001
                result["e"] = e

        th = threading.Thread(target=work, daemon=True)
        th.start()
        while th.is_alive():
            th.join(1.0)
            yield (_status_html(state["frac"], _live_desc(state)),) + (gr.update(),) * N_REST
        if "e" in result:
            e = result["e"]
            # the bar must not stay on the orange «Идёт обработка» while the error dialog is shown (as in BS 1.0):
            # the outcome goes to the bar first, and the dialog on top of it explains what to do
            if isinstance(e, AnalysisCancelled):
                yield (_status_html(state["frac"], "по запросу пользователя", state="stopped"),) + (gr.update(),) * N_REST
                raise gr.Error("Обработка остановлена. Проверьте язык речи и запустите заново.", title="Остановлено")
            log.error("analysis failed", exc_info=(type(e), e, e.__traceback__))
            msg = analysis_error_ru(e)
            yield (_status_html(state["frac"], msg, state="error"),) + (gr.update(),) * N_REST
            raise gr.Error(msg, title=ERROR_TITLE)
        rep = result["r"]
        yield (_status_html(1.0, f"обработано за {fmt_secs(time.time() - state['t0'])}", state="done"),) + render(rep)

    def stop():
        studio.stop_event.set()
        return _status_html(0.0, "текущий отрезок дорабатывается, затем обработка прерывается (до ~20 с)", state="stopped")

    def make_pdf(job_dir):
        if not job_dir:
            raise gr.Error("Сначала проанализируйте видео", title=ERROR_TITLE)
        try:
            return export_pdf(job_dir)
        except Exception:  # noqa: BLE001
            log.exception("PDF export failed for %s", job_dir)
            raise gr.Error("Не удалось собрать PDF. Подробности записаны в журнал сервера.", title=ERROR_TITLE)

    def block(label: str, chart: bool = False, grow: bool = False):
        """Result block with a visible title in the block corner (gr.HTML hides its label and frame by default). Chart
        blocks (chart=True) use the chart's own title as the label; units and scales are in the chart subtitle under it.
        grow=True: the window of a bs2-pair row that takes the extra height (APP_CSS)."""
        return gr.HTML(label=label, show_label=True, container=True,
                       elem_classes=["bs2-block"] + (["bs2-chart"] if chart else []) + (["bs2-grow"] if grow else []))

    force_russian_gradio()
    with gr.Blocks(title="BS 2.0 — Big Five, эмоции, голос, речь", theme=_theme(), css=APP_CSS) as demo:
        job_state = gr.State("")
        with gr.Row():
            with gr.Column(scale=4):
                # the title stays the first line; the copyright sits right under it in small dimmed type (APP_CSS)
                gr.Markdown(f"# BS 2.0 — анализ человека по видео\n\n<p class='bs2-copy'>{COPYRIGHT}</p>\n\n"
                            "Big Five (первое впечатление), эмоции по речи и по лицу, характеристики голоса, манера речи, "
                            "объяснения. Всё считается локально.")
            with gr.Column(scale=1, min_width=220):
                pdf_btn = gr.DownloadButton("Экспорт в PDF", variant="primary", interactive=False)
        status = gr.HTML(value="")
        with gr.Row(elem_classes=["bs2-pair"]):
            with gr.Column(scale=1, min_width=320):
                video = gr.Video(label="Видео", sources=["upload"], height=VIDEO_H, elem_classes=["bs2-grow"])
                lang = gr.Radio(choices=[("русский", "ru"), ("английский", "en")], value="ru", label="Язык речи",
                                info="Русский: основную оценку даёт OCEAN-AI (веса MuPTA), своя модель — второе мнение. "
                                     "Английский: среднее двух систем.")
                explain = gr.Checkbox(value=True, label="Объяснения (ключевые кадры, вклад модальностей, слова)")
                with gr.Row():
                    btn = gr.Button("Анализировать", variant="primary")
                    stop_btn = gr.Button("Остановить обработку", variant="stop")
            with gr.Column(scale=2, min_width=480):
                facts = block("Ключевые факты")
                narrative = gr.Textbox(label="Краткие выводы", lines=9, max_lines=20, autoscroll=False,
                                       elem_classes=["bs2-grow"])
        with gr.Tabs():
            with gr.Tab("Обзор"):
                # the radar window takes the height of the bars and the second opinion: the circle grows as far as the
                # column width allows and is centred with its subtitle
                with gr.Row(elem_classes=["bs2-pair"]):
                    with gr.Column(scale=1, min_width=360):
                        radar = block("Профиль Big Five", chart=True, grow=True)
                    with gr.Column(scale=1, min_width=360):
                        bars = block("Оценки по чертам и второе мнение", grow=True)
            with gr.Tab("Таймлайн"):
                traits_plot = block("Big Five по ходу ролика", chart=True)
                emo_plot = block("Эмоции по ходу ролика", chart=True)
                with gr.Row(elem_classes=["bs2-pair"]):
                    with gr.Column(scale=1, min_width=360):
                        voice_plot = block("Голос по ходу ролика", chart=True, grow=True)
                    with gr.Column(scale=1, min_width=360):
                        speech_plot = block("Речь по ходу ролика", chart=True, grow=True)
            with gr.Tab("Эмоции и голос"):
                emo_bars = block("Средний профиль эмоций за ролик", chart=True)
                seg_table = block("Эмоции, голос и темп по отрезкам")
            with gr.Tab("Речь"):
                speech_html = block("Речь в цифрах")
                transcript = gr.Textbox(label="Транскрипт речи", lines=10, max_lines=14, autoscroll=False)
            with gr.Tab("Мимика и кадры"):
                face_html = block("Лицо: итоги по ролику")
                face_plot = block("Выражение лица за ролик", chart=True)
                gallery = block("Ключевые кадры")
            with gr.Tab("Объяснения"):
                with gr.Row(elem_classes=["bs2-pair"]):
                    with gr.Column(scale=1, min_width=360):
                        contrib = block("Вклад модальностей в оценку своей модели", grow=True)
                    with gr.Column(scale=1, min_width=360):
                        words_detail = gr.Textbox(label="Слова, на которые откликнулась модель", lines=8, max_lines=16,
                                                  autoscroll=False, elem_classes=["bs2-grow"])
                desc = gr.Textbox(label="Описание поведения по отрезкам", lines=8, max_lines=12, autoscroll=False)
            with gr.Tab("Данные"):
                members = gr.Textbox(label="Участники ансамбля и время обработки", lines=4, max_lines=8, autoscroll=False)
                raw = gr.Code(label="result.json", language="json", lines=24, elem_classes=["bs2-json"])
                path = gr.Textbox(label="Сохранено в", interactive=False)
        # the caveats are the most important small print on the page: 13 px (gr.Markdown <small> gave 11 px)
        gr.HTML(f"<div style='font-size:13px;line-height:1.5;margin-top:6px;padding-top:10px;"
                f"border-top:1px solid {PAL['card_border']}'><b>Как читать результаты.</b> {DISCLAIMER_RU}<br>"
                f"{INTERVIEW_DISCLAIMER_RU}<br>Эмоции, голос и мимика — сигналы моделей, обученных на англоязычных корпусах "
                "и фотографиях; это наблюдения о поведении на видео, а не диагноз.</div>")
        outputs = [status, radar, bars, facts, narrative, traits_plot, emo_plot, voice_plot, speech_plot, emo_bars, seg_table,
                   speech_html, transcript, face_html, face_plot, gallery, contrib, words_detail, desc, members, raw, path, job_state, pdf_btn]
        assert len(outputs) == N_REST + 1
        run_ev = btn.click(analyze, inputs=[video, lang, explain], outputs=outputs, show_progress="hidden", api_name=False)
        stop_btn.click(stop, inputs=None, outputs=[status], cancels=[run_ev], show_progress="hidden", api_name=False)
        pdf_btn.click(make_pdf, inputs=[job_state], outputs=[pdf_btn], api_name=False)
        if preview_job:
            # the finished job becomes the initial value of every output (set before the page config is built), so
            # the page arrives filled; a demo.load event did not always reach the browser
            rep = json.loads((Path(preview_job) / "result.json").read_text(encoding="utf-8"))
            rep["job_dir"] = str(preview_job)
            filled = (_status_html(1.0, "предпросмотр готового результата", state="done"),) + page_outputs(rep)
            for comp, value in zip(outputs, filled):
                comp.value = value
            pdf_btn.interactive = True
    return demo


def main(port: int = 7870, members: str = "oceanai,mm", work_dir: str | None = None, share: bool = False,
         asr_model: str = "openai/whisper-large-v3-turbo", ollama_model: str = "qwen2.5vl:7b", mm_ckpt: str | None = None,
         host: str = "0.0.0.0"):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    os.environ.setdefault("no_proxy", "localhost,127.0.0.1,0.0.0.0")
    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,0.0.0.0")
    wd = Path(work_dir or os.path.expanduser("~/bs2_data/web_jobs"))
    wd.mkdir(parents=True, exist_ok=True)
    studio = Studio(members=tuple(m.strip() for m in members.split(",") if m.strip()), asr_model=asr_model,
                    ollama_model=ollama_model, mm_ckpt=mm_ckpt)
    demo = build_app(studio, wd)
    # allowed_paths: key-frame JPEGs live in the job folder, Gradio 5 refuses to serve files outside it
    demo.queue(default_concurrency_limit=1).launch(server_name=host, server_port=port, share=share, show_api=False,
                                                    show_error=True, quiet=False, allowed_paths=[str(wd)])
