"""HTML of the tab «Тип MBTI» and of the short emotion paragraph (design 10.3, 5.5–5.7; task T18; 3.1: one model).

- `types_html(mb)` — the one panel of the model that ran, titled «OCEAN-AI, веса MuPTA» or «AMLAI 1.0»: letters of the
  type, its name, the type with the borderline axes, four axis tracks with the score on 0…1 (the borderline zone
  0.35–0.65 shaded) and neuroticism on its own track;
- `strip_html(mb)` — the letter strip «Тип по ходу ролика» of that model: one column per segment, letters as text
  (bold — clear, normal — moderate, dashed frame — on the border, «—» on hatching — no score), the summary line and
  C8 (C19);
- `read_html(mb)` — «Как читать тип MBTI»: the correspondence table 5.5 and the caveats C3, C4, C5, C6, C7, C9, C16
  (caveats.MBTI_READ; C7 says the type is that of the one chosen model);
- `emo_intro_html(view)` — «Эмоции и голос: коротко»: the text-emotion and voice sentences in one paragraph.

The table 5.5 (mbti.TABLE_ROWS, corr_cell, TABLE_NOTE) and the summary line of the strip (mbti.summary_line) are
printed by section 2 of the PDF too, so they live in mbti. No new colours (design 13.4): outlines are palette.HTML
track_outline, the marker main_fill. Text colours come from the theme.
"""
from __future__ import annotations

import html as _html

from .. import caveats
from ..labels import mbti_model_title
from ..mbti import AXES, AXIS_LABEL, TABLE_NOTE, TABLE_ROWS, corr_cell, load_config, source_title, summary_line
from ..norms import RU_NAMES
from ..palette import HTML as PAL
from ..scores import LEVELS_RU, level_phrase, score_text
from ..textfmt import clock, plural_ru
from .parts import NOTE, table_html, th_text

OUTLINE = PAL["track_outline"]
HATCH = "repeating-linear-gradient(45deg,rgba(128,128,128,.25) 0 3px,transparent 3px 6px)"
TEXT14 = "font-size:14px;line-height:1.5"


def _e(s) -> str:
    return _html.escape(str(s), quote=True)


def _letters(item: dict, size: int) -> str:
    """The letters as in the header of the characterization (8.3): up to two X — the strict type with the borderline
    letters dimmed and underlined with a dashed line; three or four X — the type with X."""
    x = int(item.get("x_count") or 0)
    axes = item.get("axes") or {}
    if x <= 2:
        word = item.get("type_strict") or ""
        spans = []
        for ch, ax in zip(word, AXES):
            if (axes.get(ax) or {}).get("borderline"):
                spans.append(f"<span style='opacity:.55;text-decoration:underline 2px dashed {OUTLINE};"
                             f"text-underline-offset:6px'>{_e(ch)}</span>")
            else:
                spans.append(_e(ch))
        border = [AXIS_LABEL[ax] for ax in AXES if (axes.get(ax) or {}).get("borderline")]
        aria = f"Тип {word}" + "".join(f", ось {a} на границе" for a in border)
    else:
        word = item.get("type") or ""
        spans = [_e(ch) for ch in word]
        aria = f"Тип {word}, не выражен"
    return (f"<span aria-label='{_e(aria)}' style='font-size:{size}px;font-weight:700;letter-spacing:.06em;"
            f"line-height:1.1'>{''.join(spans)}</span>")


def _loose_line(item: dict, names: dict) -> str:
    """«С учётом границ: EXFJ · возможен ENFJ («Наставник»)»."""
    x = int(item.get("x_count") or 0)
    loose = item.get("type") or ""
    if x == 0:
        return f"С учётом границ: {loose} (пограничных осей нет)"
    if x >= 3:
        return f"С учётом границ: {loose} · тип не выражен: {x} оси из 4 на границе"
    alts = item.get("alternatives") or []
    shown = ", ".join(t + (f" («{names[t]}»)" if names.get(t) else "") for t in alts)
    return f"С учётом границ: {loose}" + (f" · {'возможен' if len(alts) == 1 else 'возможны'} {shown}" if alts else "")


def _track(p, marker: str, aria: str) -> str:
    """12 px track with the borderline zone 0.35–0.65 (dashed edges), the middle mark 0.5 and a 14 px marker at the
    score p."""
    mark = ""
    if p is not None:
        left = max(0.0, min(100.0, float(p) * 100))
        mark = (f"<div style='position:absolute;left:calc({left:.1f}% - 7px);top:-1px;width:14px;height:14px;"
                f"border-radius:50%;{marker};box-shadow:0 0 0 2px var(--block-background-fill)'></div>")
    return (f"<div role='img' aria-label='{_e(aria)}' style='position:relative;height:12px;border-radius:6px;"
            f"box-shadow:inset 0 0 0 1px {OUTLINE};margin:4px 7px'>"
            f"<div style='position:absolute;left:35%;width:30%;top:0;bottom:0;background:rgba(128,128,128,.16);"
            f"border-left:1px dashed {OUTLINE};border-right:1px dashed {OUTLINE}'></div>"
            "<div style='position:absolute;left:50%;top:0;bottom:0;width:1px;background:currentColor;opacity:.6'></div>"
            f"{mark}</div>")


def _poles(left: str, right: str) -> str:
    return (f"<div style='display:flex;justify-content:space-between;gap:8px;{TEXT14}'>"
            f"<span>{_e(left)}</span><span style='text-align:right'>{_e(right)}</span></div>")


def _axis_row(ax: str, a: dict, cfg: dict, marker: str) -> str:
    trait, high, low = cfg["axes"][ax]
    poles = cfg.get("pole_names_ru") or {}
    corr = (cfg.get("correspondence") or {}).get(ax) or {}
    p = a.get("value")
    if a.get("missing"):
        head, word, lvl = "нет данных", "", ""
    else:
        word = a.get("word") or ""
        if word in ("отчётливо", "умеренно"):
            word += f" (уверенность {float(a.get('confidence') or 0):.2f})"
        if a.get("borderline"):
            head = f"X (ближе к {a.get('letter')})"
        else:
            head = str(a.get("letter"))
        # the borderline zone is always «средний уровень» (also for a section saved with unrounded decisions)
        lp = LEVELS_RU["mid"] if a.get("borderline") else level_phrase(p)
        lvl = f"{RU_NAMES[trait]} {score_text(p)} — {lp}" if lp else ""
    parts = [head, word, lvl] + ([f"соответствие шкал r ≈ {corr['r']}"] if corr.get("r") is not None else [])
    sub = " · ".join(x for x in parts if x)
    aria = f"{AXIS_LABEL[ax]}: " + (head if a.get("missing") else f"{head}, {a.get('word')}")
    return (f"<div style='margin:10px 0 0'>{_poles(f'{low} · {poles.get(low, low)}', f'{poles.get(high, high)} · {high}')}"
            f"{_track(p, marker, aria)}<div style='font-size:13px;opacity:.75;line-height:1.4'>{_e(sub)}</div></div>")


def _neuro_row(item: dict, marker: str) -> str:
    n = item.get("neuroticism")
    if not n or not n.get("level"):
        return ""
    return (f"<div style='margin-top:12px;padding-top:6px;border-top:1px dashed {OUTLINE}'>"
            f"{_poles('низкий', 'высокий')}{_track(n.get('value'), marker, 'Нейротизм: ' + n['level'])}"
            f"<div style='{TEXT14}'>Нейротизм — {_e(n['level'])}. В MBTI этой шкалы нет, поэтому он приводится "
            "отдельно.</div></div>")


def _panel(item: dict, cfg: dict) -> str:
    names = cfg.get("type_names_ru") or {}
    marker = f"background:{PAL['main_fill']}"
    x = int(item.get("x_count") or 0)
    name = item.get("type_name") if x <= 2 else None
    rows = "".join(_axis_row(ax, (item.get("axes") or {}).get(ax) or {"missing": True}, cfg, marker) for ax in AXES)
    return (f"<div style='border:1px solid {OUTLINE};border-radius:8px;padding:12px;min-width:0'>"
            f"<div style='font-size:15px;font-weight:600;line-height:1.4;margin-bottom:8px'>"
            f"{_e(mbti_model_title(item))}</div>"
            f"<div style='display:flex;flex-wrap:wrap;gap:4px 14px;align-items:baseline'>{_letters(item, 40)}"
            + (f"<span style='font-size:18px;font-weight:600'>«{_e(name)}»</span>" if name else "") + "</div>"
            f"<div style='{TEXT14};margin-top:6px'>{_e(_loose_line(item, names))}</div>"
            f"{rows}{_neuro_row(item, marker)}</div>")


def types_html(mb: dict | None) -> str:
    """«Тип MBTI»: the one panel of the model that ran (5.6; 3.1), C20 under it when the recorded model gave no scores
    and the job's other model is shown; C21 without Big Five."""
    if not mb:
        return f"<p style='{TEXT14};margin:0'>{_e(caveats.text('C21'))}</p>"
    cfg = load_config()
    tail = ""
    if mb.get("source") == "own_model" and mb.get("primary_missing"):
        tail = f"<p style='{NOTE};margin:8px 0 0'>{_e(caveats.text('C20'))}</p>"
    # the grid keeps the panel's width rule on a phone (min(320px, 100%)) and leaves room for nothing else: one model
    return ("<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(min(320px,100%),1fr));gap:12px'>"
            + _panel(mb, cfg) + "</div>" + tail)


# --------------------------------------------------------------------------------------------- letter strip ---

def _cell(e: dict, i: int, word: str | None, who: str) -> str:
    base = ("width:26px;height:26px;box-sizing:border-box;display:flex;align-items:center;justify-content:center;"
            "border-radius:3px;line-height:1")
    when = f"{clock(e.get('start'))}–{clock(e.get('end'))}"
    ts = e.get("type_strict")
    if not ts:
        return (f"<div title='{_e(when + ' · нет оценки ' + who)}' style='{base};background:{HATCH};opacity:.9'>"
                "—</div>")
    ax = AXES[i]
    letter = ts[i]
    if letter == "X" or word in (None, "нет данных"):
        return (f"<div title='{_e(when + ' · ' + AXIS_LABEL[ax] + ': нет данных')}' style='{base};"
                f"background:{HATCH}'>—</div>")
    style = base
    if word == "отчётливо":
        style += ";font-weight:700"
    elif word == "на границе":
        style += f";border:1px dashed {OUTLINE}"
    return f"<div title='{_e(f'{when} · {AXIS_LABEL[ax]}: {letter}, {word}')}' style='{style}'>{_e(letter)}</div>"


def _lane(entries: list[dict], title: str, who: str) -> str:
    n = len(entries)
    cols = f"56px repeat({n},26px)"
    cells = ["<div></div>"]
    for j in range(0, n, 3):                          # the start time above every third column
        span = min(3, n - j)
        cells.append(f"<div style='grid-column:span {span};font-size:12px;opacity:.75;white-space:nowrap;"
                     f"font-variant-numeric:tabular-nums;align-self:end'>{clock(entries[j].get('start'))}</div>")
    for i, ax in enumerate(AXES):
        cells.append(f"<div style='font-weight:600;align-self:center'>{AXIS_LABEL[ax]}</div>")
        for e in entries:
            cells.append(_cell(e, i, (e.get("words") or {}).get(ax), who))
    typed = sum(1 for e in entries if e.get("type_strict"))
    aria = f"{title}: буквы типа по {n} {plural_ru(n, 'отрезку', 'отрезкам', 'отрезкам')}, с оценкой {typed}"
    return (f"<div style='font-size:14px;font-weight:600;margin:0 0 4px'>{_e(title)}</div>"
            f"<div style='overflow-x:auto;padding-bottom:4px'><div role='img' aria-label='{_e(aria)}' "
            f"style='display:grid;grid-template-columns:{cols};gap:2px;font-size:14px;width:max-content'>"
            + "".join(cells) + "</div></div>")


LEGEND = ("Жирная буква — ось выражена отчётливо, обычная — умеренно, в пунктирной рамке — на границе (показана буква "
          "строгого деления), «—» — нет оценки. Подсказка при наведении на клетку — время отрезка и буква.")


def strip_html(mb: dict | None) -> str:
    """«Тип по ходу ролика» (5.7): the strip of the model that ran (one model since 3.1), the summary line, C8, and
    C19 for a video analysed as one segment."""
    if not mb:
        return ""
    entries = mb.get("timeline") or []
    total = int(mb.get("segments_total") or len(entries) or 1)
    if total <= 1 or not entries:
        return f"<p style='{TEXT14};margin:0'>{_e(caveats.text('C19'))}</p>"
    body = _lane(entries, mbti_model_title(mb), source_title(mb))
    summary = summary_line(mb)
    return (body + f"<div style='{NOTE};margin-top:6px'>{_e(LEGEND)}</div>"
            + (f"<p style='{TEXT14};margin:10px 0 0'>{_e(summary)}</p>" if summary else "")
            + f"<p style='{NOTE};margin:8px 0 0'>{_e(caveats.text('C8'))}</p>")


# ------------------------------------------------------------------------------------------- how to read ---

def read_html(mb: dict | None = None) -> str:
    """«Как читать тип MBTI»: the correspondence table (5.5) and the caveats caveats.MBTI_READ; the same for any job
    (`mb` is accepted and not read: the page passes the section to every MBTI block)."""
    cfg = load_config()
    corr = cfg.get("correspondence") or {}
    rows = []
    for ax, scale, direction in TABLE_ROWS:
        c = corr.get(ax) or {}
        rows.append([AXIS_LABEL[ax], scale, direction, corr_cell(c)])
    rows.append(["—", "Нейротизм (= 1 − эмоциональная стабильность)", "—", "в MBTI не выражается"])
    head = [th_text("Ось MBTI"), th_text("Шкала Big Five"), th_text("Направление"), th_text("Соответствие шкал")]
    texts = [caveats.text(c) for c in caveats.MBTI_READ]
    return (table_html(head, rows, wrap_first=True)
            + f"<p style='{NOTE};margin:8px 0 12px'>{_e(TABLE_NOTE)}</p>"
            + "".join(f"<p style='{TEXT14};margin:0 0 8px'>{_e(t)}</p>" for t in texts))


# --------------------------------------------------------------------------------------- emotions and voice ---

def emo_intro_html(view: dict) -> str:
    """«Эмоции и голос: коротко»: the text-emotion and voice sentences (the same as the PDF intros) in one paragraph."""
    from ..analyses_text import analyses_parts
    parts = analyses_parts(view)
    text = " ".join(p for p in (parts.get("text_emotion"), parts.get("voice")) if p)
    return f"<p style='{TEXT14};margin:0'>{_e(text)}</p>" if text else ""


def method_html(text: str) -> str:
    """«Как получены оценки» (narrative.method_notes) as a 14 px paragraph."""
    return f"<p style='{TEXT14};margin:0'>{_e(text)}</p>" if text else ""
