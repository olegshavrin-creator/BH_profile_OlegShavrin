"""The result page of BS Profiler 3.1 (Gradio): `page_outputs` — the values of the result blocks in the order they sit
on the page (`PAGE_BLOCKS`, without the status line and the PDF button) — and the HTML builders behind them.

Readability rules for the HTML blocks (both Gradio themes, see palette.py): text colours are inherited from the theme,
secondary text is the same colour at opacity .75 and at least 13 px, marks and outlines come from palette.HTML, and
every block shows its title (the app builds each block with show_label=True, container=True).
"""
from __future__ import annotations

import json
from pathlib import PurePosixPath

from .. import caveats, jobfiles, jobview
from ..analyses_text import speech_description
from ..facts import (FACTS_LEGEND, FER_NOTE, card_item, fact_cards, fact_label, head_motion_word, segment_cells,
                     speech_cards)
from ..labels import EMO_RU
from ..narrative import NO_EXPLAIN_RU, method_notes
from ..palette import CARD_TINT, FACT_VALUE, HTML as PAL
from ..ru_texts import transcript_shown, vocabulary_shown
from ..scores import FACT_STATES, data_json, has_explanations
from ..segments import representative
from ..textfmt import clock, fmt_secs, mmss_labels, plural_ru, seg_label
from . import mbti_html
from .charts import (fig_emotion_bars, fig_emotions_timeline, fig_face_expr, fig_radar, fig_speech_timeline,
                     fig_traits_timeline, fig_voice_timeline)
from .parts import NOTE, _bar_html, _contrib_html, _words_text, model_line, table_html, th_text
from .plotframe import plot_html as _plot_html

# the result blocks after the status line, in their order on the page and without the PDF button; these are the names
# the byte-level baseline stores the values under (scripts/compare_baseline.PAGE_BLOCKS, asserted equal at start-up):
# 22 blocks kept in the places of the 2.0 page, then 5 added by 3.0. Index 20 is «Сохранено в», 21 the hidden job
# folder the PDF button reads.
PAGE_BLOCKS = ("radar", "bars", "key_facts_html", "characterization_html", "traits_timeline", "emotions_timeline",
               "voice_timeline", "speech_timeline", "emotion_bars", "segments_table", "speech_cards", "transcript",
               "face_cards", "face_chart", "key_frames_html", "contrib", "words", "behavior_description",
               "model_and_time", "result_json", "saved_to", "job_state", "method", "emo_intro", "mbti_types",
               "mbti_strip", "mbti_read")
N_PAGE = len(PAGE_BLOCKS)


def page_index(name: str) -> int:
    """The position of a block in page_outputs by its PAGE_BLOCKS name."""
    return PAGE_BLOCKS.index(name)


# the note of the tab «Данные» when scores.data_json left something out of a Russian job: the percentiles and the
# stored 2.0 `narrative` of older jobs (imported 2.0, 3.0 and early 3.1 jobs), which the page does not use
DATA_TRIMMED = ("В result.json ниже не показаны поля прежних версий, которые 3.1 не использует (процентили, сводка "
                "версии 2.0); файл не изменён.")
# metric cards: 1 px outline 3:1 on every background, light tint (palette.CARD_TINT, the background check_palette.py
# measures the card text and the outline on) so label, value and note read as one card
CARDS = "display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:8px"
CARD = (f"padding:10px 14px;border:1px solid {PAL['card_border']};background:rgba(128,128,128,{CARD_TINT});"
        "border-radius:8px;min-width:0")
# A value is one line of 20 px text in a card 150 px wide at its narrowest, and a single long word («возбуждение
# 0.30») is wider than that: without overflow-wrap it paints over the card border and into the next card. Chrome's
# `hyphens: auto` was tried instead and dropped — it depends on hyphenation patterns the browser may not have, so the
# same card would break differently on two machines.
CARD_VALUE = ("font-size:20px;font-weight:600;line-height:1.25;font-variant-numeric:tabular-nums;"
              "overflow-wrap:anywhere")
# «Ключевые факты»: one class per state, so the value takes the colour of the theme the page is showing (the block
# HTML is built once for both themes, Gradio puts `dark` on an ancestor of the container). The rules travel with the
# block, so the same HTML also reads right outside the app (scripts/rerender_samples.py --html-dir).
# Gradio 5.8 gives everything inside an HTML block the body text colour with `.gradio-container-5-8-0 .prose *`,
# specificity (0,2,0). A rule of one class lost to it, and in the light theme every value was the plain text colour;
# the dark rule, two classes, won only because this <style> comes after Gradio's. So both rules sit under the class of
# the card grid (_cards): the light one, `div.bs3-facts .bs3-fact-…`, is (0,2,1) and beats Gradio's rule whatever the
# order of the style sheets, and the dark one, one class more, beats the light one (tests/test_key_facts.py).
FACTS_SCOPE = "bs3-facts"
FACTS_CSS = "<style>" + "".join(f"div.{FACTS_SCOPE} .bs3-fact-{s}{{color:{FACT_VALUE['light'][s]}}}"
                                f".dark div.{FACTS_SCOPE} .bs3-fact-{s}{{color:{FACT_VALUE['dark'][s]}}}"
                                for s in FACT_STATES) + "</style>"


def _cards(items, min_px: int = 180, value_first: bool = False) -> str:
    """(label, value, note[, state]) -> a grid of cards; note may be empty. `value_first`: the card reads value,
    then label, then note; a value with a state is painted by it (FACTS_CSS, under the grid's class FACTS_SCOPE) and
    its label line ends with the word of that state (facts.fact_label), so the card also reads without colour —
    «Ключевые факты» of 3.1."""
    html = ""
    for item in items:
        lab, val, note, state = card_item(item)
        cls = f" class='bs3-fact-{state}'" if value_first and state else ""
        value = (f"<div{cls} style='{CARD_VALUE};margin:"
                 + ("0 0 2px" if value_first else "3px 0") + f"'>{val if val not in (None, '') else '—'}</div>")
        label = f"<div style='{NOTE}'>{fact_label(lab, state) if value_first else lab}</div>"
        rest = f"<div style='{NOTE}'>{note}</div>" if note else ""
        html += f"<div style='{CARD}'>" + (value + label if value_first else label + value) + rest + "</div>"
    grid = CARDS.replace("minmax(180px", f"minmax({int(min_px)}px")
    scope = f" class='{FACTS_SCOPE}'" if value_first else ""
    return f"<div{scope} style='{grid}'>{html}</div>" if html else ""


def _facts_html(view: dict, mb: dict | None = None) -> str:
    """«Ключевые факты» in the left column (design 10.2): the MBTI type card first, then the cards of 2.0 from the clean
    view; 150 px minimum, two cards in a row in the 320 px column. 3.1: every card reads value, label, explanation; the
    value of a measured card is coloured by where it sits and its label says the same in a word; one line under the
    grid says what the colour and the word mean (facts.fact_cards, the same cards as the PDF)."""
    items, show_legend = fact_cards(view, mb)
    grid = _cards(items, min_px=150, value_first=True)
    if not grid:
        return ""
    legend = f"<p style='{NOTE};margin:8px 0 0'>{FACTS_LEGEND}</p>" if show_legend else ""
    return FACTS_CSS + grid + legend


def _segments_table(rep: dict) -> str:
    per = (rep.get("analyses") or {}).get("per_segment") or []
    if not per:
        return ""
    # one time format for the whole column («0:00–0:20 … 10:00–10:12»), the same as on the chart time axes; the
    # seconds are rounded as in the PDF and in the chart hover (640.0–651.8 is «10:40–10:52»)
    hours = max(float(r["end"]) for r in per) >= 3600
    head = [th_text("Отрезок", "ч:мин:с" if hours else "мин:с"), th_text("Эмоция", "по тексту речи"),
            th_text("Выражение", "лица"), th_text("Возбуждение", "голос, 0…1"), th_text("Уверенность", "голос, 0…1"),
            th_text("Позитивность", "голос, 0…1"), th_text("Темп", "слов в минуту"), th_text("Доля пауз", "в отрезке")]
    # the cells of the PDF appendix «Значения по отрезкам» (facts.segment_cells): «нет речи» / «нет текста» for a
    # segment with an empty transcript, «—» for one without a tempo
    rows = [[f"{clock(r['start'], hours)}–{clock(r['end'], hours)}", *segment_cells(r)] for r in per]
    return (f"<div style='{NOTE};margin-bottom:8px'>Для каждого отрезка: преобладающая эмоция по тексту речи и по лицу "
            "(с долей), три характеристики голоса от 0 до 1, темп речи и доля пауз. Шапка таблицы остаётся на месте "
            "при прокрутке.</div>" + table_html(head, rows, max_height=480))


def _speech_html(rep: dict) -> str:
    sp = (rep.get("analyses") or {}).get("speech") or {}
    if not sp:
        return ""
    items = speech_cards(sp, small_rate_words=False)             # the cards of the PDF; the page says «0 на 100 слов»
    vocab = ", ".join(f"{w} ({n})" for w, n in vocabulary_shown(rep)[:15])       # in Russian for any speech language
    return (_cards(items) + "<p style='font-size:15px;line-height:1.5;margin:12px 0 6px'>"
            f"{speech_description(sp)}</p>"
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
        cards.append(("Движение головы", head_motion_word(hm, "motion"),
                      f"смещение между кадрами — {hm:.0%} ширины лица"))
    if fa.get("face_share") is not None:
        cards.append(("Лицо найдено", f"{fa['face_share']:.0%}", "доля разобранных кадров"))
    if frames:
        n = len(per)
        # «взяты из 31 отрезка», not «372 / из 31 отрезка», which reads like a fraction
        cards.append(("Кадров разобрано", f"{frames}", f"взяты из {n} {plural_ru(n, 'отрезка', 'отрезков', 'отрезков')}"))
    return (_cards(cards) + f"<p style='{NOTE};margin-top:10px'>{FER_NOTE.format(where=' (вкладка «Таймлайн»)')}</p>")


# key frames: the figure toggles .bs3-kf-big; enlarged, the image fills the window and the caption (moment of the
# video) stays readable on a dark plate at the bottom, so it is always clear which moment is shown
FRAMES_CSS = (
    "<style>"
    ".bs3-kf figure{margin:0!important;cursor:zoom-in}"
    # thumbnails keep the frame's own proportions (no empty letterbox bands); tall portrait frames stop at 320 px
    ".bs3-kf img{display:block;width:100%;height:auto;max-height:320px;object-fit:contain;border-radius:8px;"
    "background:rgba(128,128,128,.12);outline:1px solid " + PAL["card_border"] + ";outline-offset:-1px}"
    # one short line under the frame: the moment and what is visible; it never grows past two lines, the rest of
    # the story (expressions, what the frame did to the score) opens on hover as the figure's title
    ".bs3-kf figcaption{text-align:center;font-size:13px;line-height:1.35;margin-top:6px;opacity:.75;"
    "font-variant-numeric:tabular-nums;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;"
    "overflow:hidden}"
    ".bs3-kf figcaption b{font-weight:600}"
    ".bs3-kf figure.bs3-kf-big{cursor:zoom-out}"
    ".bs3-kf figure.bs3-kf-big img{position:fixed;inset:4vh 4vw;width:92vw;height:92vh;max-height:none;z-index:9999;"
    "border-radius:8px;background:rgba(0,0,0,.92);outline:0;box-shadow:0 0 0 100vmax rgba(0,0,0,.85)}"
    ".bs3-kf figure.bs3-kf-big figcaption{position:fixed;left:50%;bottom:calc(4vh + 14px);transform:translateX(-50%);"
    "z-index:10000;margin:0;padding:6px 14px;border-radius:8px;background:rgba(0,0,0,.8);color:#fff!important;"
    "font-size:16px;display:block;opacity:1;max-width:84vw;pointer-events:none}"
    ".bs3-kf figure:not(.bs3-kf-big) .bs3-kf-more{display:none}"
    "</style>")


NO_FRAMES_OCEANAI = ("Ключевых кадров нет: модель OCEAN-AI не строит объяснений, ключевые кадры есть только для модели "
                     "AMLAI 1.0.")
NO_FRAMES_MM = "Ключевые кадры не построены: лицо в кадре не найдено или объяснения не удалось посчитать."


def _frames_html(rep: dict, expl: dict | None = None, max_side: int = 640) -> str:
    """Key frames embedded as data-URI JPEGs. gr.Gallery depends on Gradio serving files from the job folder, which
    proved unreliable in this setup (images arrive broken); inline images always render. Click enlarges a frame.
    Key frames belong to AMLAI 1.0: an OCEAN-AI job shows one line instead (also an older job that carries the frames
    of the second model of 3.0, since the page shows one model), a job of AMLAI 1.0 without frames says why.

    Under every frame one short line — the moment of the video and, in a few words, what is visible there
    (frame_captions.build). Hovering the frame opens the rest in the figure's tooltip: the two strongest facial
    expressions of that frame and what the frame did to the score, with a direction. The enlarged frame shows both
    lines on its dark plate, so nothing is hidden from someone who never hovers."""
    import base64
    import io
    from html import escape
    from PIL import Image

    from .. import frame_captions

    if not has_explanations(rep):
        return f"<p style='font-size:14px'>{NO_FRAMES_OCEANAI}</p>"
    # the frames of the job folder being shown, found by name (jobfiles); without a folder there are none
    paths = [str(p) for p in jobfiles.key_frame_paths(rep["job_dir"], rep)] if rep.get("job_dir") else []
    if not paths:
        return f"<p style='font-size:14px'>{NO_FRAMES_MM}</p>"
    shown, images = [], []
    for p in paths:
        try:
            im = Image.open(p).convert("RGB")
            im.thumbnail((max_side, max_side))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=82)
        except Exception:  # noqa: BLE001
            continue
        shown.append(p)
        images.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    if not shown:
        return f"<p style='font-size:14px'>{NO_FRAMES_MM}</p>"
    # frames come from the representative segment of a long video, otherwise from the whole video (no timeline): the
    # moment is then counted from 0, the same rule as in the PDF
    seg = representative(rep, among="all")
    entries = frame_captions.build(rep, shown, expl)
    tenths = frame_captions.has_tenths(rep, shown, None, expl)
    figs = []
    for b64, e in zip(images, entries):
        tip = e["tooltip"] or "Щёлкните, чтобы увеличить"
        more = f" · {escape(e['tooltip'])}" if e["tooltip"] else ""
        figs.append(
            f"<figure role='button' tabindex='0' title='{escape(tip)}' onclick=\"this.classList.toggle('bs3-kf-big')\" "
            "onkeydown=\"if(event.key==='Enter'||event.key===' '){event.preventDefault();this.classList.toggle('bs3-kf-big')}"
            "else if(event.key==='Escape'){this.classList.remove('bs3-kf-big')}\">"
            f"<img src='data:image/jpeg;base64,{b64}' alt='{escape(e['alt'])}'>"
            f"<figcaption><b>{escape(e['label'])}</b>"
            + (f" · {escape(e['tail'])}" if e["tail"] else "")
            + f"<span class='bs3-kf-more'>{more} · щелчок закрывает</span></figcaption></figure>")
    where = f" (отрезок {seg_label(seg['start'], seg['end'])})" if seg else ""
    # a job made before the captions has neither a phrase nor the expressions: the note promises only what the
    # page really shows, otherwise it sends the reader hunting for a description that is not there
    _, has_expr, has_eff = frame_captions.note_flags(entries)
    what = frame_captions.note_what(entries, tenths, "page")
    hover = [x for x, ok in (("выражение лица", has_expr), ("то, как кадр сдвинул оценку", has_eff)) if ok]
    return (FRAMES_CSS + "<div class='bs3-kf' style='display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));"
            f"gap:12px'>{''.join(figs)}</div>"
            f"<p style='{NOTE};margin-top:10px'>Кадры, сильнее всего повлиявшие на оценку модели AMLAI 1.0{where}. "
            + f"Рамкой на кадре отмечено найденное лицо, под кадром — {what}. "
            + (f"Наведите мышь на кадр — {'покажутся' if len(hover) > 1 else 'покажется'} {' и '.join(hover)}. "
               if hover else "")
            + "Щелчок по кадру увеличивает его, повторный щелчок закрывает.</p>")


def model_text(view: dict, rep: dict, mb: dict | None) -> str:
    """«Модель и время обработки» (tab «Данные», 3.1): one line with the model that ran and its scores, one with the
    processing time, then the notes of the render (C22 for a type computed on display, DATA_TRIMMED)."""
    lines = [model_line(view), f"Обработка заняла {fmt_secs((rep.get('timings_sec') or {}).get('total_wall', 0))}."]
    if mb and mb.get("computed_on_render"):
        lines.append(caveats.text("C22"))
    if data_json(rep)[1]:
        lines.append(DATA_TRIMMED)
    return "\n".join(lines)


def _without_server_paths(data: dict) -> dict:
    """The result.json of the tab «Данные» without the server paths the file keeps: `job_dir`, `input`, every
    `key_frames` entry and every `timeline[].file` become the file or folder name alone. Works in place on the copy
    scores.data_json made; the file on disk keeps its paths."""
    def name(v):
        return PurePosixPath(v).name if isinstance(v, str) else v

    for key in ("job_dir", "input"):
        if key in data:
            data[key] = name(data[key])
    if isinstance(data.get("key_frames"), list):
        data["key_frames"] = [name(v) for v in data["key_frames"]]
    for t in data.get("timeline") if isinstance(data.get("timeline"), list) else []:
        if isinstance(t, dict) and "file" in t:
            t["file"] = name(t["file"])
    return data


def page_outputs(rep: dict) -> tuple:
    """Everything the result page shows for a finished job, in the order of the output blocks after the status line
    (without the PDF button). Jobs processed before the Russian texts existed get them here, stored back.

    Design 10.5: the numbers come from the clean view (scores.clean_view), the MBTI section from mbti.get_mbti (the
    saved one, or computed now and never written), the characterization from characterization.build, each built once
    (jobview.for_page; the app renders that JobView with page_values and hands it to the journal). The first 22
    values keep their places (index 2 — the key facts with the type card first, index 3 — the characterization), then
    five new ones: «Как получены оценки», «Эмоции и голос: коротко», the MBTI panel, the letter strip and «Как читать
    тип MBTI». One model everywhere (3.1): the bars, the panel and the strip are the model recorded in the job; the
    tab «Объяснения» of an OCEAN-AI job shows one note (narrative.NO_EXPLAIN_RU) in its first block. The «Данные» tab
    shows result.json as it lies on disk, except that for Russian speech the 2.0 fields that 3.x does not use
    (percentiles, the stored 2.0 summary) are left out, with a note (scores.data_json), and that the paths on the server
    are shown as file and folder names (_without_server_paths); «Сохранено в» is the name of the job folder. The last
    of the first 22 values, the job folder the PDF button reads, stays the full path: it goes into a gr.State, which
    Gradio keeps on the server and does not send to the browser with the results of an analysis.

    The files of the job (explanation.json, the key frames) are read from the folder `rep["job_dir"]` (jobfiles). A
    `rep` without it renders without them: no explanation, no key frames, and «Сохранено в» and the job folder of the
    PDF button stay empty."""
    return page_values(jobview.for_page(rep))


def page_values(jv: jobview.JobView) -> tuple:
    """The values of page_outputs for a JobView (jobview.for_page): the app renders a finished analysis from it and
    hands the same JobView to the journal entry of that analysis (journal.result)."""
    rep, view, mb, expl, job = jv.rep, jv.view, jv.mb, jv.expl, jv.job
    # explanations exist for AMLAI 1.0 only: an OCEAN-AI job says so once, in the first block of the tab, and the
    # other blocks stay empty (an older job may carry the explanation and the behaviour description of the second
    # model of 3.0; the page shows one model)
    own = has_explanations(view)
    contrib = _contrib_html(expl) if own else f"<p style='font-size:14px;margin:0'>{NO_EXPLAIN_RU}</p>"
    data, _ = data_json(rep)
    # the transcript block; an older job processed as English shows its Russian translation with a one-line note
    note, transcript = transcript_shown(rep)
    # fill=True: charts in a row of two windows grow to the height of the window next to them (see APP_CSS)
    out = (_plot_html(fig_radar, view, fill=True),
           _bar_html(view["traits"], view.get("interview")),
           _facts_html(view, mb), jv.character.html(), _plot_html(fig_traits_timeline, view),
           _plot_html(fig_emotions_timeline, view), _plot_html(fig_voice_timeline, view, fill=True),
           _plot_html(fig_speech_timeline, view, fill=True), _plot_html(fig_emotion_bars, view),
           _segments_table(view), _speech_html(view),
           "\n\n".join(t for t in (note, transcript) if t), _face_html(view), _plot_html(fig_face_expr, view),
           _frames_html(view, expl if own else None), contrib,
           _words_text(expl) if (expl and own) else "",
           mmss_labels(rep.get("behavior_description_ru") or "") if own else "", model_text(view, rep, mb),
           json.dumps(_without_server_paths(data), ensure_ascii=False, indent=2), job.name if job else "",
           str(job) if job else "",
           # the five blocks after the first 22 (design 10.3, 10.5)
           mbti_html.method_html(method_notes(view)), mbti_html.emo_intro_html(view),
           mbti_html.types_html(mb), mbti_html.strip_html(mb), mbti_html.read_html(mb))
    assert len(out) == N_PAGE
    return out
