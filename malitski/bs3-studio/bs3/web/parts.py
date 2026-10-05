"""HTML helpers for the BS Profiler 3.1 web UI: score bars of the one model that ran, the modality table of AMLAI 1.0
and the shared table style used by the page (bs3/web/page.py). Forked from bs 1.0 (the unused 1.0 page, engine and
run_analysis were removed; the page is bs3/web/app.py); the second-opinion block of 3.0 is gone with the second
model (3.1).

Colours: bars, outlines and rules come from palette.HTML (>= 3:1 on the dark and the light Gradio theme). Text colours
are never hard-coded: Gradio's `.prose *` rule gives the body text colour of the current theme, secondary text is the
same colour at opacity .75 (>= 6:1 on every background). The HTML lives in the page DOM, not in an iframe, so Gradio
CSS variables such as --block-background-fill switch with the theme by themselves (used for the sticky table header).
"""
from __future__ import annotations

import logging

from .. import MODEL_TITLES
from ..labels import ROW_TITLES, model_title
from ..norms import TRAIT_KEYS
from ..palette import HTML as PAL
from ..scores import scored
from ..textfmt import fiv2_ref_ru, pct_phrase

log = logging.getLogger("bs3.web")

TRAIT_TITLES = {
    "openness": "Открытость опыту",
    "conscientiousness": "Добросовестность",
    "extraversion": "Экстраверсия",
    "agreeableness": "Доброжелательность",
    "emotional_stability": "Эмоциональная стабильность",
    "interview": "Впечатление «пригласить на собеседование»",
}
MEMBER_TITLES = {**MODEL_TITLES,
                 "face": "лицо", "audio": "голос (CLAP)", "audio_whisper": "голос (Whisper)", "audio_xlsr": "голос (XLS-R)",
                 "audio_w2v_emo": "голос (wav2vec2)", "text": "речь", "behavior": "описание поведения"}
# modality columns of the contribution table: (column title, second line)
MODALITY_HEADS = {"face": ("Лицо", "кадры"), "audio": ("Голос", "CLAP"), "audio_whisper": ("Голос", "Whisper"),
                  "audio_xlsr": ("Голос", "XLS-R"), "audio_w2v_emo": ("Голос", "wav2vec2"), "text": ("Речь", "текст"),
                  "behavior": ("Поведение", "описание")}
NOTE = "font-size:13px;opacity:.75;line-height:1.45"           # footnotes and card notes (13 px minimum)
SUB = "display:block;font-size:13px;font-weight:400;opacity:.75"  # second line of a table header (units, model)


# ---------------------------------------------------------------- tables
def th_text(title: str, sub: str | None = None) -> str:
    """Header cell content: title plus an optional smaller second line with the unit or the model."""
    return title + (f"<span style='{SUB}'>{sub}</span>" if sub else "")


def table_html(head: list[str], rows: list[list[str]], *, max_height: int | None = None, wrap_first: bool = False) -> str:
    """Table on a gr.HTML block (container=True).

    Opaque sticky header (theme block colour under a grey tint, so scrolled rows never show through) with a 2 px rule;
    1 px rules between rows; first column = row name, left-aligned and bold; numbers in tabular figures. Rows are told
    apart by rules only, without zebra stripes: the #808080 rule is 3.8:1 on the dark block and 3.9:1 on white, and a
    tint under it would take that down.
    `border-collapse:separate` keeps the header rule attached to the sticky header while scrolling; inline `border:0`
    overrides Gradio's prose grid (a full 1 px grid in the text colour)."""
    rule = PAL["table_rule"]
    th = ("border:0;border-bottom:2px solid " + rule + ";background:linear-gradient(rgba(128,128,128,.18),"
          "rgba(128,128,128,.18)),var(--block-background-fill);font-size:14px;font-weight:600;line-height:1.2;"
          "padding:6px 8px;position:sticky;top:0;z-index:1;vertical-align:bottom;")
    td = ("border:0;border-bottom:1px solid " + rule + ";padding:5px 8px;font-variant-numeric:tabular-nums;"
          "vertical-align:middle;")
    first_td = td + "text-align:left;font-weight:600;" + ("min-width:120px" if wrap_first else "white-space:nowrap")
    other_td = td + "text-align:center;white-space:nowrap"
    head_html = "".join(f"<th style='{th}text-align:{'left' if i == 0 else 'center'}'>{h}</th>" for i, h in enumerate(head))
    body = "".join("<tr style='border:0'>" + "".join(f"<td style='{first_td if i == 0 else other_td}'>{c}</td>"
                                                     for i, c in enumerate(r)) + "</tr>" for r in rows)
    wrap = "overflow:auto" + (f";max-height:{max_height}px" if max_height else "")
    return (f"<div style='{wrap}'><table style='border-collapse:separate;border-spacing:0;border:0;margin:0;font-size:14px'>"
            f"<thead><tr style='border:0'>{head_html}</tr></thead><tbody>{body}</tbody></table></div>")


# ---------------------------------------------------------------- score bars
def _track(value: float, fill: str, height: int, radius: int, tick_pct=None, fill_extra: str = "") -> str:
    """Outlined track (transparent inside, 1 px palette.HTML track_outline) with the fill = score 0…1 and an optional tick
    (3 px, text colour, sticks out 4 px above and below) at the percentile position 0…100%."""
    width = max(0.0, min(100.0, float(value) * 100))
    tick = ""
    if tick_pct is not None:
        p = max(0.0, min(100.0, float(tick_pct)))
        tick = (f"<div style='position:absolute;left:calc({p:.1f}% - 1.5px);top:-4px;bottom:-4px;width:3px;"
                "border-radius:1px;background:currentColor'></div>")
    return (f"<div style='position:relative;height:{height}px;border-radius:{radius}px;background:{PAL['track']};"
            f"box-shadow:inset 0 0 0 1px {PAL['track_outline']}'>"
            f"<div style='width:{width:.1f}%;min-width:3px;height:{height}px;border-radius:{radius}px;background:{fill};"
            f"{fill_extra}'></div>{tick}</div>")


def _score_row(title: str, value: float, text: str, fill: str, tick_pct=None, *, height: int = 14, radius: int = 6,
               bold: bool = True, fill_extra: str = "") -> str:
    # when the name and the text do not fit on one line (long names, narrow screens) the text moves to its own line,
    # right-aligned, instead of being squeezed into a narrow column of 4-5 short lines
    name = f"<b>{title}</b>" if bold else f"<span>{title}</span>"
    return (f"<div style='margin:10px 0 12px'><div style='display:flex;flex-wrap:wrap;justify-content:space-between;"
            f"align-items:baseline;gap:2px 12px;font-size:14px;margin-bottom:6px'>{name} <span style='margin-left:auto;"
            f"text-align:right;font-variant-numeric:tabular-nums'>{text}</span></div>"
            f"{_track(value, fill, height, radius, tick_pct, fill_extra)}</div>")


def _scale_row() -> str:
    """0 / 0.5 / 1 under a group of bars (same width as the tracks), so bar lengths can be read and compared."""
    tick = "position:absolute;top:0;width:1px;height:5px;background:currentColor"
    lab = "position:absolute;top:6px"
    return ("<div aria-hidden='true' style='position:relative;height:24px;margin-top:6px;font-size:13px;opacity:.75;"
            "font-variant-numeric:tabular-nums'>"
            f"<span style='{tick};left:0'></span><span style='{tick};left:50%'></span><span style='{tick};right:0'></span>"
            f"<span style='{lab};left:0'>0</span><span style='{lab};left:50%;transform:translateX(-50%)'>0.5</span>"
            f"<span style='{lab};right:0'>1</span></div>")


def _swatch(fill: str, extra: str = "") -> str:
    return (f"<span aria-hidden='true' style='display:inline-block;width:18px;height:10px;border-radius:3px;"
            f"vertical-align:middle;margin-right:6px;background:{fill};box-shadow:inset 0 0 0 1px {PAL['track_outline']};"
            f"{extra}'></span>")


def _tick_swatch() -> str:
    return ("<span aria-hidden='true' style='display:inline-block;width:3px;height:16px;border-radius:1px;"
            "vertical-align:middle;margin:0 7px 0 7px;background:currentColor'></span>")


def _legend(items: list[str]) -> str:
    return ("<div style='display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13px;line-height:1.4;margin-top:2px'>"
            + "".join(f"<span>{it}</span>" for it in items) + "</div>")


TICK_NOTE = ("Риска на полоске — процентиль в First Impressions V2: у какой доли людей этого датасета оценка ниже; "
             "риска посередине — медиана датасета.")
SCALE_NOTE = "Длина полоски — оценка системы от 0 до 1; уровни черт и буквы MBTI считаются по этой же шкале, середина — 0.5."


def _bar_html(traits: dict, interview: dict | None) -> str:
    interview = scored(interview)              # an entry without a numeric score is not shown
    items = [(k, traits[k]) for k in TRAIT_KEYS] + ([("interview", interview)] if interview else [])
    groups: dict[str, list[str]] = {}          # FIV2 reference (display text) -> item keys, for the footnote
    any_tick = False
    rows = []
    for k, t in items:
        pct = t.get("percentile", t.get("percentile_vs_fiv2"))
        ref = t.get("percentile_ref", "train FIV2" if pct is not None else "")
        phrase, tick_ok = pct_phrase(pct, ref)
        if phrase:
            groups.setdefault(fiv2_ref_ru(ref), []).append(k)
        any_tick = any_tick or tick_ok
        score = float(t["score"])
        fill = PAL["interview_fill"] if k == "interview" else PAL["main_fill"]
        text = f"{score:.2f}" + (f" · {phrase}" if phrase else "")
        row = _score_row(TRAIT_TITLES[k], score, text, fill, pct if tick_ok else None)
        if k == "interview":           # a separate label of the own model: set off from the five traits
            row = (f"<div style='margin-top:14px;padding-top:2px;border-top:1px dashed {PAL['track_outline']}'>{row}</div>")
        rows.append(row)
    legend = [_swatch(PAL["main_fill"]) + "черта, оценка 0…1"]
    if interview:
        legend.append(_swatch(PAL["interview_fill"]) + "«собеседование», оценка 0…1")
    if any_tick:
        legend.append(_tick_swatch() + "процентиль в First Impressions V2")
    notes = [SCALE_NOTE]
    for ref, keys in groups.items():
        if len(keys) == len(items) and len(groups) == 1:
            subject = "Все процентили"
        elif set(keys) == set(TRAIT_KEYS):
            subject = "Процентили пяти черт"
        elif keys == ["interview"]:
            subject = "Процентиль «собеседования»"
        else:
            subject = "Процентили: " + ", ".join(ROW_TITLES[k] for k in keys)
        notes.append(f"{subject} — относительно {ref}.")
    if any_tick:
        notes.append(TICK_NOTE)
    if interview:                      # C2 explains the label where it is shown (the page footer holds only what
        from .. import caveats          # is true for both models)
        notes.append(caveats.text("C2"))
    return ("<div style='max-width:640px'>" + "".join(rows) + _scale_row() + _legend(legend) +
            f"<div style='{NOTE};margin-top:8px'>{' '.join(notes)}</div></div>")


def model_line(view: dict) -> str:
    """One line for «Модель и время обработки»: «Модель OCEAN-AI, веса MuPTA: открытость опыту 0.71, …» — the clean
    scores of the one model the view shows (scores.clean_view)."""
    main = (view.get("view_meta") or {}).get("main_system")
    traits = view.get("traits") or {}
    scores = ", ".join(f"{TRAIT_TITLES[k].lower()} {float(traits[k]['score']):.2f}" for k in TRAIT_KEYS
                       if isinstance(traits.get(k), dict) and traits[k].get("score") is not None)
    return f"Модель {model_title(main)}" + (f": {scores}." if scores else ".")


def _words_text(expl: dict) -> str:
    """Readable word attributions (content words in Russian for any speech language, grouped by direction), from the
    lists explanation.json keeps under "readable_words", which the PDF prints too. Reads only: the lists of an older
    job are made and stored before the page is built, with its other Russian texts (jobview,
    ru_texts.ensure_russian_job)."""
    from ..narrative import words_summary
    return "\n\n".join(words_summary(expl.get("readable_words") or {}, expl, TRAIT_TITLES))


# ---------------------------------------------------------------- modality contributions
def _pct(share: float) -> str:
    v = float(share) * 100
    return "&lt;1%" if v < 0.95 else f"{v:.0f}%"


def _share_cell(share: float, top: bool) -> str:
    """Share as text (bold for the largest in the row) with a small outlined bar under it."""
    w = max(0.0, min(100.0, float(share) * 100))
    return (f"<div style='font-weight:{700 if top else 400}'>{_pct(share)}</div>"
            f"<div style='width:56px;height:6px;margin:3px auto 0;border-radius:3px;"
            f"box-shadow:inset 0 0 0 1px {PAL['track_outline']}'><div style='width:{w:.1f}%;height:6px;border-radius:3px;"
            f"background:{PAL['share_fill']};opacity:.55'></div></div>")


def _contrib_html(expl: dict | None) -> str:
    if not expl:
        return ""
    ixg = (expl.get("modalities") or {}).get("input_x_gradient")
    if not ixg:
        return ""
    mods = list(next(iter(ixg.values())).keys())
    head = ["Черта"]
    for m in mods:
        title, sub = MODALITY_HEADS.get(m, (MEMBER_TITLES.get(m, m)[:1].upper() + MEMBER_TITLES.get(m, m)[1:], None))
        head.append(th_text(title, sub))
    rows = []
    for k, row in ixg.items():
        shares = [float(row[m]["share"]) for m in mods]
        top = max(shares) if shares else 0.0
        rows.append([ROW_TITLES.get(k, k)] + [_share_cell(s, s == top) for s in shares])
    return (f"<div style='{NOTE};margin-bottom:8px'>Какая доля оценки модели AMLAI 1.0 пришлась на каждую модальность "
            "(по градиенту оценки: насколько признаки каждой модальности сдвигают результат). В каждой строке доли в "
            "сумме дают 100%; самая большая выделена жирным.</div>"
            + table_html(head, rows, wrap_first=True) +
            f"<div style='{NOTE};margin-top:8px'>«&lt;1%» — модальность почти не влияет на оценку этого ролика: модель, "
            "обученная на First Impressions V2, опирается в основном на лицо и голос; речь и описание поведения слабо "
            "меняют результат.</div>")
