"""Big Five in MBTI notation (design 5, 7; changes of 2026-09-26; one model since 3.1).

Exactly the customer's formula on the model's own score v in [0, 1] (config `raw_thresholds`, 0.5 on every axis):
the letter by v >= 0.5, BORDERLINE = 0.15, confidence = min(|v − 0.5| / 0.5, 1), X on a borderline axis in `type`,
four letters in `type_strict`. The same for either model (OCEAN-AI or AMLAI 1.0); no reference group anywhere. The
section describes the one model the analysis ran (`source`, `model`): there is no second opinion and no agreement of
two systems any more (schema 3).

Details: the confidence denominator for a threshold other than 0.5 (thr below it, 1 − thr above it, so both ends reach
1); the difference is rounded to 9 digits so that 0.65 − 0.5 is not 0.15000000000000002; a missing trait gives the
letter X with `missing: true` in both `type` and `type_strict`; a value outside [0, 1] is clipped with `clipped: true`.

The formula is applied to the score as the report prints it (scores.shown: rounded once to two decimals), so the
letter, the borderline flag, the level word and the printed number of an axis always agree (a printed 0.65 is never
X on one page and a confident letter on another); the axis `value` is that printed value.

Nothing here writes to disk: `get_mbti(rep, view)` returns the section saved by the pipeline (schema 3) or computes it
on the fly; the caller never stores the computed one.
"""
from __future__ import annotations

import copy
import datetime as _dt
import itertools
import json
from collections import Counter
from importlib import resources

from . import MODEL_TITLES, PRODUCT, __version__
from .labels import SOURCE_OF, source_title
from .norms import TRAIT_KEYS
from .scores import level_phrase, num, shown
from .textfmt import plural_ru

AXES = ("EI", "SN", "TF", "JP")
AXIS_LABEL = {"EI": "E–I", "SN": "S–N", "TF": "T–F", "JP": "J–P"}
WORD_CLEAR, WORD_MODERATE, WORD_BORDER, WORD_MISSING = "отчётливо", "умеренно", "на границе", "нет данных"
CLEAR_CONFIDENCE = 0.7         # from this confidence on an axis is «отчётливо», below it «умеренно» (word_for)
RELIABILITY_BASIS = ("соответствие шкал MBTI и NEO-PI в самоотчётах (McCrae, Costa, 1989); "
                     "не точность оценки по видео")
NEURO_NOTE = "шкала не имеет соответствия в MBTI, приводится отдельно"
# 1: letters by the position in a reference group of processed videos (before 2026-09-26); 2: the absolute scale with
# a second opinion and the agreement of two systems (3.0); 3: one model, no `second`, no `agreement` (3.1)
SCHEMA_VERSION = 3
METHOD = "raw"

_cfg: dict | None = None


def load_config() -> dict:
    """config/mbti.json (a copy: callers may change it for an experiment without touching the cache)."""
    global _cfg
    if _cfg is None:
        with resources.files("bs3").joinpath("config/mbti.json").open("r", encoding="utf-8") as f:
            _cfg = json.load(f)
    return copy.deepcopy(_cfg)


def word_for(axis: dict) -> str:
    """Confidence of an axis in words: «отчётливо» (>= CLEAR_CONFIDENCE), «умеренно», «на границе» (X), «нет
    данных»."""
    if axis.get("missing"):
        return WORD_MISSING
    if axis.get("borderline"):
        return WORD_BORDER
    return WORD_CLEAR if round(axis.get("confidence") or 0.0, 9) >= CLEAR_CONFIDENCE else WORD_MODERATE


# --------------------------------------------------------------------------------------------------------- core ---

def alternatives(type_strict: str, axes: dict, cfg: dict | None = None) -> list[str]:
    """All strict types obtained by flipping the borderline axes (missing ones are not flipped), except type_strict
    itself, in the axis order EI, SN, TF, JP: 1 X -> 1 type, 2 -> 3, 3 -> 7, 4 -> 15."""
    ax_cfg = (cfg or load_config())["axes"]
    options = []
    for i, ax in enumerate(AXES):
        a = axes.get(ax) or {}
        cur = type_strict[i]
        if a.get("borderline") and not a.get("missing"):
            _, high, low = ax_cfg[ax]
            options.append([cur, low if cur == high else high])
        else:
            options.append([cur])
    out = ["".join(t) for t in itertools.product(*options)]
    return [t for t in out if t != type_strict]


def bigfive_to_mbti(scores: dict, thresholds: dict | None = None, *, borderline: float = 0.15,
                    cfg: dict | None = None) -> dict:
    """The customer's formula on scores in [0, 1].

    Returns {"type", "type_strict", "type_name", "alternatives", "x_count", "axes": {axis: {"trait", "value",
    "threshold", "letter", "confidence", "borderline"[, "missing"][, "clipped"]}}, "neuroticism": {"value"} | None}.
    """
    cfg = cfg or load_config()
    scores = scores if isinstance(scores, dict) else {}
    thresholds = thresholds or {}
    axes: dict = {}
    loose, strict = [], []
    for ax in AXES:
        trait, high, low = cfg["axes"][ax]
        thr = float(thresholds.get(ax, 0.5))
        v = num(scores.get(trait))
        if v is None:
            axes[ax] = {"trait": trait, "value": None, "threshold": thr, "letter": None, "missing": True,
                        "confidence": 0.0, "borderline": True}
            loose.append("X")
            strict.append("X")
            continue
        a = {"trait": trait}
        if v < 0.0 or v > 1.0:
            v = min(max(v, 0.0), 1.0)
            a["clipped"] = True
        letter = high if v >= thr else low
        d = round(abs(v - thr), 9)
        denom = thr if v < thr else 1.0 - thr
        conf = min(d / denom, 1.0) if denom > 0 else 1.0
        a.update({"value": v, "threshold": thr, "letter": letter, "confidence": round(conf, 9),
                  "borderline": d < borderline})
        axes[ax] = a
        loose.append("X" if a["borderline"] else letter)
        strict.append(letter)
    type_loose, type_strict = "".join(loose), "".join(strict)
    es = num(scores.get("emotional_stability"))
    neuro = None if es is None else {"value": round(1.0 - min(max(es, 0.0), 1.0), 9)}
    return {"type": type_loose, "type_strict": type_strict,
            "type_name": None if "X" in type_strict else cfg["type_names_ru"].get(type_strict),
            "alternatives": alternatives(type_strict, axes, cfg), "x_count": type_loose.count("X"),
            "axes": axes, "neuroticism": neuro}


def _thresholds(cfg: dict) -> dict:
    """The threshold of every axis (config `raw_thresholds`, 0.5 by the customer's document)."""
    th = cfg.get("raw_thresholds") or {}
    return {ax: float(th.get(ax, 0.5)) for ax in AXES}


def _r(x, nd):
    return None if x is None else round(x, nd)


def mbti_for(system: str, raw_scores: dict, cfg: dict | None = None) -> dict:
    """Type of one model ('oceanai' | 'mm') from its own Big Five scores, in the format of result.json. The formula
    does not depend on the speech language and works on the printed scores (scores.shown, two decimals)."""
    cfg = cfg or load_config()
    raw_scores = raw_scores if isinstance(raw_scores, dict) else {}
    scores = {k: shown(raw_scores.get(k)) for k in TRAIT_KEYS}
    core = bigfive_to_mbti(scores, _thresholds(cfg), borderline=cfg.get("borderline", 0.15), cfg=cfg)
    axes = {}
    for ax in AXES:
        a = dict(core["axes"][ax])
        a["value"] = _r(a["value"], 2)
        a["confidence"] = round(a["confidence"], 2)
        a["word"] = word_for(core["axes"][ax])
        axes[ax] = {k: a[k] for k in ("trait", "value", "threshold", "letter", "confidence", "borderline", "word",
                                      "missing", "clipped") if k in a}
    corr = cfg.get("correspondence") or {}
    neuro, neuro_note = None, None
    if core["neuroticism"] is not None:
        nv = core["neuroticism"]["value"]
        lv = level_phrase(nv)
        neuro = {"value": round(nv, 2), "level": lv, "note": NEURO_NOTE}
        neuro_note = f"Шкала нейротизма ({lv}) в MBTI не выражается, приводится отдельно"
    return {
        "source": SOURCE_OF.get(system, system),
        "type": core["type"], "type_strict": core["type_strict"], "type_name": core["type_name"],
        "alternatives": core["alternatives"], "x_count": core["x_count"],
        "axes": axes,
        "reliability": {ax: f"{corr[ax]['label']} (r≈{corr[ax]['r']})" for ax in AXES if ax in corr},
        "reliability_r": {ax: corr[ax]["r"] for ax in AXES if ax in corr},
        "reliability_basis": RELIABILITY_BASIS,
        "neuroticism": neuro, "neuroticism_note": neuro_note,
    }


# ------------------------------------------------------------------------------------------- segments, section ---

def _segment_scores(t: dict, system: str, main: str) -> dict | None:
    """Scores of `system` on one segment of a clean view (design 5.7): variants of 3.x jobs first; otherwise the
    segment's own scores, but only for the shown model and only where it scored."""
    variants = t.get("variants")
    if isinstance(variants, dict) and system in variants:
        v = variants.get(system)
        return v if isinstance(v, dict) else None
    if system == main and isinstance(t.get("scores"), dict) and not t.get("no_primary"):
        return t["scores"]
    return None


def segment_types(view: dict, system: str, cfg: dict | None = None) -> list[dict]:
    """Type of `system` on every segment, with the same thresholds as for the whole video."""
    cfg = cfg or load_config()
    main = (view.get("view_meta") or {}).get("main_system")
    out = []
    for t in view.get("timeline") or []:
        if not isinstance(t, dict):
            continue
        e = {"segment": t.get("segment"), "start": t.get("start"), "end": t.get("end")}
        sc = _segment_scores(t, system, main)
        if sc is None:
            e.update({"type": None, "reason": "no_primary" if system == main else "no_data"})
        else:
            m = mbti_for(system, sc, cfg)
            e.update({"type": m["type"], "type_strict": m["type_strict"],
                      "words": {ax: m["axes"][ax]["word"] for ax in AXES}})
        out.append(e)
    return out


def stability(entries: list[dict], type_strict: str, cfg: dict | None = None) -> tuple[dict | None, list]:
    """({axis: {"same": k, "of": n}} or None when fewer than `min_segments_for_stability` segments were typed,
    modal types [[type, count], ...] — up to three most frequent strict types)."""
    cfg = cfg or load_config()
    typed = [e for e in entries if e.get("type_strict")]
    counts = Counter(e["type_strict"] for e in typed)
    first = {}
    for i, e in enumerate(typed):
        first.setdefault(e["type_strict"], i)
    modal = [[t, c] for t, c in sorted(counts.items(), key=lambda kv: (-kv[1], first[kv[0]]))[:3]]
    n = len(typed)
    if n < int(cfg.get("min_segments_for_stability", 4)) or not type_strict:
        return None, modal
    st = {ax: {"same": sum(e["type_strict"][i] == type_strict[i] for e in typed), "of": n}
          for i, ax in enumerate(AXES)}
    return st, modal


def border_counts(entries: list[dict]) -> tuple[dict, int]:
    """({axis: number of typed segments where that axis is on the border (X in `type`)}, number of typed segments).
    Taken from the saved segment types, so it works for any stored section."""
    typed = [e for e in entries if e.get("type_strict")]
    counts = {ax: sum(1 for e in typed if (e.get("type") or "")[i:i + 1] == "X" and e["type_strict"][i] != "X")
              for i, ax in enumerate(AXES)}
    return counts, len(typed)


def border_text(entries: list[dict]) -> str:
    """«ось E–I на границе во всех 17 отрезках, S–N — в 8, T–F — в 6, J–P — в 8»; '' when no axis of any typed
    segment was on the border."""
    counts, n = border_counts(entries)
    parts = []
    for ax in (a for a in AXES if counts[a]):
        k = counts[ax]
        if not parts:
            where = (f"во всех {n} {plural_ru(n, 'отрезке', 'отрезках', 'отрезках')}" if k == n
                     else f"в {k} из {n} {plural_ru(n, 'отрезка', 'отрезков', 'отрезков')}")
            parts.append(f"ось {AXIS_LABEL[ax]} на границе {where}")
        else:
            parts.append(f"{AXIS_LABEL[ax]} — " + ("во всех" if k == n else f"в {k}"))
    return ", ".join(parts)


def _scores_of(view: dict, system: str) -> dict | None:
    if system == (view.get("view_meta") or {}).get("main_system"):
        tr = view.get("traits") or {}
        sc = {k: (tr.get(k) or {}).get("score") for k in TRAIT_KEYS}
    else:
        sc = (view.get("variant_scores") or {}).get(system)
    if not isinstance(sc, dict) or all(num(sc.get(k)) is None for k in TRAIT_KEYS):
        return None
    return {k: sc.get(k) for k in TRAIT_KEYS}


def build_section(view: dict, *, computed_at: str | None = None, cfg: dict | None = None) -> dict | None:
    """The `mbti` section of result.json (design 7.1; schema 3) from a clean view (scores.clean_view): the type of the
    one model the view shows (`source` "ocean_ai" | "own_model", `model` "oceanai" | "mm", `model_title`), its axes,
    the type per segment with the stability; None without Big Five. No second opinion, no agreement (3.1)."""
    cfg = cfg or load_config()
    meta = view.get("view_meta")
    if not meta:
        from .scores import clean_view
        view = clean_view(view)
        meta = view["view_meta"]
    main = meta["main_system"]
    main_scores = _scores_of(view, main)
    if main_scores is None:
        return None
    m = mbti_for(main, main_scores, cfg)
    entries = segment_types(view, main, cfg)
    st, modal = stability(entries, m["type_strict"], cfg)
    timeline = view.get("timeline") or []
    total = len(timeline) or int(view.get("segments") or 1)
    used = sum(1 for e in entries if e.get("type_strict")) if timeline else 1
    sec = {
        "schema_version": SCHEMA_VERSION,
        "computed_by": f"{PRODUCT} {__version__}",
        "computed_at": computed_at or _dt.datetime.now().isoformat(timespec="seconds"),
        "method": METHOD,
        "borderline": cfg.get("borderline", 0.15),
        "source": m["source"], "model": main, "model_title": MODEL_TITLES.get(main, main), "role": "main",
        # True when the recorded model gave no scores and the view fell back to the other member (C20)
        "primary_missing": bool(meta.get("primary_missing")),
    }
    sec.update({k: v for k, v in m.items() if k != "source"})
    sec.update({"segments_total": total, "segments_used": used, "stability": st, "modal_types": modal,
                "timeline": entries, "llm": None})
    return sec


def get_mbti(rep: dict, view: dict | None = None) -> dict | None:
    """The section to show: the one saved in result.json (schema 3) as it is, otherwise computed now from the clean
    view with `computed_on_render: True` (a saved section of schema 1 placed the scores in a reference group, one of
    schema 2 carries a second opinion and an agreement; neither is shown). Never changes `rep` and never writes
    anything."""
    saved = rep.get("mbti") if isinstance(rep, dict) else None
    if isinstance(saved, dict) and saved.get("schema_version") == SCHEMA_VERSION:
        return copy.deepcopy(saved)
    if view is None:
        from .scores import clean_view
        view = clean_view(rep)
    sec = build_section(view)
    return None if sec is None else {**sec, "computed_on_render": True}


# -------------------------------------------------------------------------------------------- short texts ---

def _name(t: str | None, cfg: dict | None = None) -> str | None:
    if not t:
        return None
    return (cfg or load_config())["type_names_ru"].get(t)


def type_title(mb: dict) -> str:
    """«Тип MBTI · OCEAN-AI» / «Тип MBTI · AMLAI 1.0»."""
    return f"Тип MBTI · {source_title(mb)}"


def fact_card(mb: dict | None) -> tuple[str, str, str] | None:
    """The first card of «Ключевые факты»: (label, value, note); None when there is no type."""
    if not mb:
        return None
    x = mb.get("x_count", 0)
    strict, name = mb.get("type_strict"), mb.get("type_name")
    notes = []
    if x == 0:
        value = strict
        if name:
            notes.append(f"«{name}»")
    elif x <= 2:
        value = mb.get("type")
        notes.append(f"ближайший {strict}" + (f" «{name}»" if name else ""))
    else:
        value = mb.get("type")
        notes.append(f"тип не выражен: {x} оси из 4 на границе")
    return type_title(mb), value, " · ".join(notes)


def _type_words(m: dict) -> str:
    """«EXFJ, ближайший ESFJ «Попечитель», возможен ENFJ» / «ESFJ «Попечитель»» / «XXXJ, тип не выражен …»."""
    x, strict, name, loose = m.get("x_count", 0), m.get("type_strict"), m.get("type_name"), m.get("type")
    q = f" «{name}»" if name else ""
    if x == 0:
        return f"{strict}{q}"
    if x >= 3:
        return f"{loose}, тип не выражен (формально ближайший {strict})"
    alts = m.get("alternatives") or []
    tail = (", возможен " if len(alts) == 1 else ", возможны ") + ", ".join(alts) if alts else ""
    return f"{loose}, ближайший {strict}{q}{tail}"


def journal_lines(mb: dict | None) -> list[str]:
    """The line about the type for the journal (design 10.6): «Тип MBTI (OCEAN-AI): ENFJ «Наставник»; нейротизм —
    средний уровень» — one model, no second opinion (3.1)."""
    if not mb:
        return ["Тип MBTI: не рассчитан (нет оценок Big Five)"]
    line = f"Тип MBTI ({source_title(mb)}): {_type_words(mb)}"
    if mb.get("neuroticism"):
        line += f"; нейротизм — {mb['neuroticism']['level']}"
    return [line]


# ------------------------------------------------------------------ texts of the tab «Тип MBTI» and of the PDF ---
# both the page (mbti_html) and section 2 of the PDF (pdf/mbti_section) print them, so they live here and not in either

# 5.5: axis, Big Five scale, direction, correspondence of the scales (r from config/mbti.json)
TABLE_ROWS = (("EI", "Экстраверсия", "выше → E"), ("SN", "Открытость опыту", "выше → N"),
              ("TF", "Доброжелательность", "выше → F"), ("JP", "Добросовестность", "выше → J"))
# the config keeps the labels of `reliability` (design 4.6, 7.1: «высокая (r≈0.74)», agreeing with «надёжность»); the
# reader's table 5.5 has the column «Соответствие шкал», so there the words agree with «соответствие»
CORR_WORD = {"высокая": "высокое", "средняя": "среднее", "низкая": "низкое"}
TABLE_NOTE = ("Корреляции шкал MBTI и NEO-PI в самоотчётах (McCrae, Costa, 1989; воспроизведено Furnham, 1996, и "
              "Furnham и соавт., 2003). Это соответствие шкал, а не точность оценки по видео.")


def corr_cell(c: dict) -> str:
    """«высокое, r ≈ 0.74»: one cell of the column «Соответствие шкал» (design 5.5)."""
    label = str(c.get("label", ""))
    return f"{CORR_WORD.get(label, label)}, r ≈ {c.get('r')}"


def _seg_word(n: int) -> str:
    """Genitive after «из N»: «из 21 отрезка», «из 26 отрезков»."""
    return plural_ru(n, "отрезка", "отрезков", "отрезков")


def summary_line(item: dict, who: str | None = None) -> str:
    """«OCEAN-AI: ESFJ в 16 из 26 отрезков с оценкой, ENFJ — в 10; ось S–N совпадает с итогом в 16 из 26 отрезков,
    остальные оси — во всех.» `who` defaults to the title of the model the section describes. The types and the
    agreement with the whole video are those of the strict letters; when an axis was on the border in some segments
    the line says «строгий тип» / «строгие буквы» and adds in how many («ось E–I на границе во всех 17 отрезках,
    S–N — в 8»)."""
    who = who or source_title(item)
    modal = item.get("modal_types") or []
    entries = item.get("timeline") or []
    n = sum(1 for e in entries if e.get("type_strict"))
    if not modal or not n:
        return ""
    t1, c1 = modal[0]
    border = border_text(entries)
    strict = "строгий тип " if border else ""
    if c1 == n:
        head = f"{who}: {strict}{t1} во всех {n} {plural_ru(n, 'отрезке', 'отрезках', 'отрезках')} с оценкой"
    else:
        head = f"{who}: {strict}{t1} в {c1} из {n} {_seg_word(n)} с оценкой"
        head += "".join(f", {t} — в {c}" for t, c in modal[1:])
    tail_border = f"; {border}" if border else ""
    st = item.get("stability")
    if not st:
        return head + tail_border + "."
    shaky = [ax for ax in AXES if (st.get(ax) or {}).get("same") != (st.get(ax) or {}).get("of")]
    if not shaky:
        what = "строгие буквы всех четырёх осей совпадают" if border else "все четыре оси совпадают"
        return head + f"; {what} с итогом во всех отрезках" + tail_border + "."
    parts = [f"ось {AXIS_LABEL[ax]} совпадает с итогом в {st[ax]['same']} из {st[ax]['of']} {_seg_word(st[ax]['of'])}"
             for ax in shaky]
    rest = len(AXES) - len(shaky)
    tail = "" if rest == 0 else (", остальные оси — во всех" if rest > 1 else ", остальная ось — во всех")
    return head + "; " + ", ".join(parts) + tail + tail_border + "."
