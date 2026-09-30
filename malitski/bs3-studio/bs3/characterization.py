"""«Характеристика личности» — the main text of BS Profiler 3.1 (design 8).

Deterministic templates over the clean numbers (scores.clean_view) and the MBTI section (mbti.get_mbti); every phrase
comes from config/lexicon_ru.json (the trait texts of 8.5, the short phrases of 8.6 and the paragraph templates of
8.4), every caveat from caveats.py. No language model. Levels are the bands of the model's own score on 0…1
(scores.level; the score itself is printed with two decimals in the parentheses of a trait line); nothing is compared
with a group of processed videos, and «Основа описания» names the one model that ran (OCEAN-AI or AMLAI 1.0, 3.1).
The text is not stored in result.json; it is built on every display.

Structure: a header (letters of the type, its name, labels) and up to eight paragraphs, each with a bold lead —
«Коротко», «Основа описания», four traits (O, C, E, A in the order of |v − 0.5|), emotional stability with
neuroticism, «Типология MBTI», «Что видно в поведении на записи» (only when there are observations), «Границы
вывода».

`build(view, mb) -> Character`; `Character.html()`, `.plain()`, `.short_plain()`, `.pdf_paragraphs()`,
`.pdf_header()`.
"""
from __future__ import annotations

import copy
import html as _html
import json
import re
from dataclasses import dataclass, field
from importlib import resources

from . import caveats
from .bands import PAUSE_WORDS
from .mbti import AXES, AXIS_LABEL
from .norms import TRAIT_KEYS
from .palette import HTML
from .scores import ABOVE, BELOW, NEUTRAL, level, level_phrase, num, score_text, shown, tempo_state
from .textfmt import plural_ru

AXIS_OF = {"extraversion": "EI", "openness": "SN", "agreeableness": "TF", "conscientiousness": "JP"}
MBTI_TRAITS = tuple(k for k in TRAIT_KEYS if k in AXIS_OF)          # O, C, E, A in the order of TRAIT_KEYS
OUTLINE = HTML["track_outline"]                                      # #808080, 3:1 on both themes
# the word of the tempo (lexicon behavior.tempo_words) for its state: one tempo band for the colour of the card
# «Темп речи» and for the words (scores.tempo_state, TEMPO_BAND 100…160; owner, 2026-09-27)
TEMPO_WORD = {BELOW: "slow", NEUTRAL: "calm", ABOVE: "fast"}

_lex: dict | None = None


def load_lexicon() -> dict:
    """config/lexicon_ru.json (a copy)."""
    global _lex
    if _lex is None:
        with resources.files("bs3").joinpath("config/lexicon_ru.json").open("r", encoding="utf-8") as f:
            _lex = json.load(f)
    return copy.deepcopy(_lex)


def _q(name: str | None) -> str:
    return f" («{name}»)" if name else ""


def words(text: str) -> int:
    """Number of words (tokens with at least one letter or digit)."""
    return sum(1 for t in text.split() if re.search(r"\w", t))


@dataclass
class Character:
    """The characterization: header and paragraphs [{"key", "lead", "text"}]."""
    header: dict
    paragraphs: list[dict] = field(default_factory=list)

    # ---------------------------------------------------------------------------------------------- plain text
    def header_plain(self) -> str:
        h = self.header
        head = h["letters_text"] + (f" «{h['name']}»" if h.get("name") else "")
        return " · ".join([head] + [t for t, _ in h["labels"]]) if head else " · ".join(t for t, _ in h["labels"])

    def plain(self) -> str:
        parts = [self.header_plain()]
        for p in self.paragraphs:
            parts.append((p["lead"] + " " + p["text"]).strip())
        return "\n\n".join(parts) + "\n"

    def short_plain(self) -> str:
        """The paragraph «Коротко» as plain text, without its lead (for the journal)."""
        return next((p["text"] for p in self.paragraphs if p["key"] == "short"), "")

    def paragraph(self, key: str) -> dict | None:
        return next((p for p in self.paragraphs if p["key"] == key), None)

    def word_count(self) -> int:
        return words(self.plain())

    # ---------------------------------------------------------------------------------------------- HTML
    def html(self) -> str:
        e = lambda s: _html.escape(s, quote=False)
        h = self.header
        letters = ""
        if h["letters"]:
            spans = []
            for ch, border in h["letters"]:
                if border:
                    spans.append(f"<span style='opacity:.55;text-decoration:underline 2px dashed {OUTLINE};"
                                 f"text-underline-offset:6px'>{e(ch)}</span>")
                else:
                    spans.append(e(ch))
            letters = (f"<span aria-label='{_html.escape(h['aria'])}' style='font-size:36px;font-weight:700;"
                       f"letter-spacing:.06em;line-height:1.1'>{''.join(spans)}</span>")
        name = (f"<span style='font-size:18px;font-weight:600'>«{e(h['name'])}»</span>" if h.get("name") else "")
        pill = ("display:inline-block;font-size:13px;line-height:1.35;border:1px {style} " + OUTLINE +
                ";border-radius:999px;padding:2px 10px")
        labels = "".join(f"<span style='{pill.format(style='dashed' if dashed else 'solid')}'>{e(t)}</span>"
                         for t, dashed in h["labels"])
        head = (f"<div style='display:flex;flex-wrap:wrap;gap:8px 16px;align-items:baseline;"
                f"border-bottom:1px solid {OUTLINE};padding-bottom:10px;margin-bottom:12px'>{letters}{name}"
                f"<span style='display:flex;flex-wrap:wrap;gap:6px 8px;align-items:baseline'>{labels}</span></div>")
        body = "".join(f"<p style='margin:0 0 10px'><b>{e(p['lead'])}</b> {e(p['text'])}</p>" for p in self.paragraphs)
        return (f"<div class='bs3-char-text' style='font-size:15px;line-height:1.55;max-width:75ch'>{head}{body}</div>")

    # ---------------------------------------------------------------------------------------------- PDF
    def pdf_header(self) -> dict:
        """{"letters": [(letter, borderline)], "name", "labels": [text]} for one header line of the PDF."""
        return {"letters": list(self.header["letters"]), "name": self.header.get("name"),
                "labels": [t for t, _ in self.header["labels"]]}

    def pdf_paragraphs(self) -> list[dict]:
        """[{"key", "lead", "text", "markdown"}]: `markdown` is «**lead** text» for fpdf2 multi_cell(markdown=True)."""
        out = []
        for p in self.paragraphs:
            out.append({**p, "markdown": f"**{p['lead']}** {p['text']}"})
        return out


def placeholder_html() -> str:
    """What the window shows before an analysis."""
    t = load_lexicon()["templates"]["placeholder"]
    return f"<div class='bs3-char-text' style='font-size:15px;line-height:1.55;opacity:.75'>{_html.escape(t)}</div>"


# ------------------------------------------------------------------------------------------------------ build ---

def _scores(view: dict) -> dict:
    """The main scores of the clean view on 0…1 as the text prints them and decides on them (scores.shown, two
    decimals; None for a missing trait), so neuroticism = 1 − the printed emotional stability."""
    tr = view.get("traits") or {}
    return {k: shown((tr.get(k) or {}).get("score")) for k in TRAIT_KEYS}


def _by_distance(keys, ps: dict, view: dict) -> list:
    """The traits in the order of |v − 0.5| of the printed score; equal printed distances keep the order of the
    unrounded scores."""
    tr = view.get("traits") or {}
    raw = {k: num((tr.get(k) or {}).get("score")) for k in keys}
    raw = {k: ps[k] if v is None else v for k, v in raw.items()}
    return sorted(keys, key=lambda k: (-round(abs(ps[k] - 0.5), 9), -abs(raw[k] - 0.5)))


def _header(mb: dict | None, T: dict, ps: dict) -> dict:
    H = T["header"]
    labels: list[tuple[str, bool]] = []
    if not mb:
        labels.append((H["no_type"], False))
        es = ps.get("emotional_stability")
        if es is not None:
            labels.append((H["neuroticism"].format(level=level_phrase(1.0 - es)), False))
        return {"letters": [], "letters_text": "", "name": None, "labels": labels, "aria": ""}
    x = int(mb.get("x_count") or 0)
    axes = mb.get("axes") or {}
    if x <= 2:
        word = mb.get("type_strict") or ""
        letters = [(ch, bool((axes.get(ax) or {}).get("borderline"))) for ch, ax in zip(word, AXES)]
        name = mb.get("type_name")
    else:
        word = mb.get("type") or ""
        letters = [(ch, False) for ch in word]
        name = None
    labels.append((H["source"].get(mb.get("source"), H["source"]["ocean_ai"]), False))
    border_axes = [ax for ax in AXES if (axes.get(ax) or {}).get("borderline")]
    if x >= 3:
        labels.append((H["not_expressed"].format(x=x), False))
    else:
        labels += [(H["axis_border"].format(axis=AXIS_LABEL[ax]), True) for ax in border_axes]
    neuro = mb.get("neuroticism")
    if neuro and neuro.get("level"):
        labels.append((H["neuroticism"].format(level=neuro["level"]), False))
    aria = f"Тип {word}" + ("".join(f", ось {AXIS_LABEL[ax]} на границе" for ax in border_axes) if x <= 2 else "")
    return {"letters": letters, "letters_text": word, "name": name, "labels": labels, "aria": aria}


def _p_short(mb, T, lex, ps, view) -> str:
    S = T["short"]
    out = []
    ranked = _by_distance([k for k in MBTI_TRAITS if ps.get(k) is not None and level(ps[k]) != "mid"], ps, view)[:2]
    phrases = []
    for k in ranked:
        lv = level(ps[k])
        # «скорее» + the phrase of the level itself (above / below), so «Коротко» agrees with the trait paragraph
        pole = lv if lv in lex["short"][k] else ("high" if lv in ("high", "above") else "low")
        phrases.append((lex["short_adverbs"][lv], lex["short"][k][pole]))
    if len(phrases) == 2:
        out.append(S["standout_two"].format(a1=phrases[0][0], p1=phrases[0][1], a2=phrases[1][0], p2=phrases[1][1]))
    elif phrases:
        out.append(S["standout_one"].format(a1=phrases[0][0], p1=phrases[0][1]))
    else:
        out.append(S["standout_none"])
    if mb:
        x = int(mb.get("x_count") or 0)
        strict, loose, name = mb.get("type_strict"), mb.get("type"), _q(mb.get("type_name"))
        border = [AXIS_LABEL[ax] for ax in AXES if ((mb.get("axes") or {}).get(ax) or {}).get("borderline")]
        if x == 0:
            out.append(S["mbti_x0"].format(S=strict, name=name))
        elif x == 1:
            out.append(S["mbti_x1"].format(S=strict, name=name, axis=border[0] if border else ""))
        elif x == 2:
            out.append(S["mbti_x2"].format(S=strict, name=name, T=loose))
        else:
            out.append(S["mbti_x3"].format(T=loose))
        neuro = mb.get("neuroticism")
        if neuro and neuro.get("level"):
            out.append(S["neuroticism"].format(level=neuro["level"]))
    elif ps.get("emotional_stability") is not None:
        out.append(S["neuroticism"].format(level=level_phrase(1.0 - ps["emotional_stability"])))
    return " ".join(out)


def _p_basis(view, mb, T) -> str:
    B = T["basis"]
    meta = view.get("view_meta") or {}
    total = int(meta.get("segments_total") or 0)
    n = int(meta.get("segments_used") or 0)
    if total <= 1 or n <= 0:
        where = B["where_whole"]
    else:
        where = B["where_segments"].format(n=n, segments=plural_ru(n, "отрезку", "отрезкам", "отрезкам"))
    # one model per analysis (3.1): the basis names the model the view shows, for any speech language of an old job
    system = B["system"].get(meta.get("main_system"), B["system"]["oceanai"])
    return B["ru"].format(system=system, where=where) + " " + B["scale"]


def _p_trait(k, view, mb, T, lex, ps) -> dict | None:
    p = ps.get(k)
    if p is None:
        return None
    Tr = T["trait"]
    lv = level(p)
    lead = Tr["lead"].format(title=lex["names"][k]["title"], level=level_phrase(p))
    paren = [score_text(p)]
    if mb:
        a = (mb.get("axes") or {}).get(AXIS_OF[k]) or {}
        if a.get("borderline") or a.get("missing"):
            paren.append(Tr["border"].format(axis=AXIS_LABEL[AXIS_OF[k]]))
        else:
            paren.append(Tr["letter"].format(letter=a.get("letter"), word=a.get("word")))
    entry = lex["levels"][k][lv]
    body = entry[:1] if lv == "mid" else entry
    text = f"({'; '.join(paren)}). " + " ".join(body)
    return {"key": f"trait:{k}", "lead": lead, "text": text}


def _p_stability(view, mb, T, lex, ps) -> dict | None:
    p = ps.get("emotional_stability")
    if p is None:
        return None
    St = T["stability"]
    # the score belongs to emotional stability, so it stands right after its level, not after neuroticism
    lead = St["lead"].format(es=level_phrase(p), score=score_text(p), n=level_phrase(1.0 - p))
    text = (" ".join(lex["levels"]["emotional_stability"][level(p)]) + " "
            + St["separate"] + " " + caveats.text("C11"))
    return {"key": "stability", "lead": lead, "text": text}


def _p_mbti(mb, T, type_names) -> str:
    M = T["mbti"]
    x = int(mb.get("x_count") or 0)
    strict, loose, name = mb.get("type_strict"), mb.get("type"), _q(mb.get("type_name"))
    border = [AXIS_LABEL[ax] for ax in AXES if ((mb.get("axes") or {}).get(ax) or {}).get("borderline")]
    out = []
    if x == 0:
        out.append(M["x0"].format(S=strict, name=name))
    elif x == 1:
        alts = mb.get("alternatives") or []
        s = M["x1"].format(S=strict, name=name, axis=border[0] if border else "", T=loose,
                           A=alts[0] if alts else "", name_a=_q(type_names.get(alts[0])) if alts else "")
        if not alts:                                   # a missing trait: no flipped type to offer
            s = s.split(":")[0] + "."
        out.append(s)
    elif x == 2:
        out.append(M["x2"].format(S=strict, name=name, axis1=border[0] if border else "",
                                  axis2=border[1] if len(border) > 1 else "", T=loose))
    else:
        out.append(M["x3"].format(count=M["count"]["4" if x >= 4 else "3"], T=loose, S=strict))
    out.append(M["basis"])
    st = mb.get("stability")
    if st:
        from .mbti import border_text
        border = border_text(mb.get("timeline") or [])
        n = max(int((st.get(ax) or {}).get("of") or 0) for ax in AXES)
        unstable = [ax for ax in AXES if st.get(ax) and st[ax]["same"] < st[ax]["of"]]
        if not unstable:
            out.append(M["stable_strict" if border else "stable_all"].format(
                n=n, segments=plural_ru(n, "отрезке", "отрезках", "отрезках")))
        else:
            first = unstable[0]
            s = M["part_first"].format(axis=AXIS_LABEL[first], k=st[first]["same"], n=st[first]["of"],
                                       segments=plural_ru(st[first]["of"], "отрезка", "отрезков", "отрезков"))
            for ax in unstable[1:]:
                s += M["part_more"].format(axis=AXIS_LABEL[ax], k=st[ax]["same"])
            if len(unstable) < len([ax for ax in AXES if st.get(ax)]):
                s += M["part_rest"]
            out.append(s + ".")
        if border:
            out.append(M["border"].format(border=border))
    return " ".join(out)


def _p_behavior(view, T, ps) -> str:
    Bh = T["behavior"]
    an = view.get("analyses") or {}
    out = []
    sp = an.get("speech") or {}
    wpm = num(sp.get("words_per_min_speech"))
    tempo = tempo_state(wpm) if wpm else None
    if wpm:
        n = int(round(wpm))
        pause = num(sp.get("pause_share"))
        few, many = (100 * x for x in PAUSE_WORDS)          # in percent, as the share below: 10 and 25
        pauses = "" if pause is None else Bh["pauses"]["few" if pause * 100 < few else ("some" if pause * 100 < many
                                                                                          else "many")]
        out.append(Bh["tempo"].format(tempo=Bh["tempo_words"][TEMPO_WORD[tempo]], wpm=n,
                                      words=plural_ru(n, "слово", "слова", "слов"), pauses=pauses))
    vo = (an.get("voice") or {}).get("mean") or {}
    arousal = num(vo.get("arousal"))
    if arousal is not None:
        v = "low" if arousal < 0.4 else ("high" if arousal > 0.6 else "mid")
        out.append(Bh["voice"].format(voice=Bh["voice_words"][v]))
    fa = (an.get("face") or {}).get("mean") or {}
    fa = {k: num(v) for k, v in fa.items() if num(v) is not None}
    if fa:
        k, v = max(fa.items(), key=lambda kv: kv[1])
        caution = Bh["face_caution"] if k in ("sad", "fear", "sadness") else ""
        out.append(Bh["face"].format(expression=Bh["face_names"].get(k, k), share=int(round(100 * v)),
                                     caution=caution))
    te = (an.get("emotions_text") or {}).get("mean") or {}
    te = {k: num(v) for k, v in te.items() if num(v) is not None}
    if te:
        if te.get("neutral", 0.0) >= 0.6:
            out.append(Bh["tone_neutral"])
        else:
            k, v = max(te.items(), key=lambda kv: kv[1])
            out.append(Bh["tone_top"].format(emotion=Bh["tone_names"].get(k, k), share=int(round(100 * v))))
    lv = level(ps.get("extraversion"))
    if lv and arousal is not None and wpm:
        if lv in ("high", "above") and arousal < 0.4 and tempo == BELOW:            # «неторопливая речь»
            out.append(Bh["check_high"])
        elif lv in ("low", "below") and arousal > 0.6 and tempo == ABOVE:         # «быстрая речь»
            out.append(Bh["check_low"])
    if not out:
        return ""
    return " ".join(out + [Bh["closing"]])


def _p_limits(view) -> str:
    meta = view.get("view_meta") or {}
    parts = [caveats.text("C14"), caveats.text("C15")]
    dropped = meta.get("segments_without_primary") or []
    if dropped:
        parts.append(caveats.c13(len(dropped), int(meta.get("segments_total") or len(dropped)), meta.get("main_system")))
    if int(meta.get("segments_total") or 0) <= 1:
        parts.append(caveats.text("C19"))
    if meta.get("primary_missing"):
        parts.append(caveats.text("C20"))
    parts.append(caveats.text("C12"))
    return " ".join(parts)


def build(view: dict, mb: dict | None) -> Character:
    """The characterization of a clean view (scores.clean_view) with its MBTI section (mbti.get_mbti; None when the
    job has no Big Five)."""
    lex = load_lexicon()
    T = lex["templates"]
    L = T["leads"]
    ps = _scores(view)
    header = _header(mb, T, ps)
    if all(v is None for v in ps.values()):
        paragraphs = [{"key": "no_scores", "lead": L["no_scores"], "text": caveats.text("C21")},
                      {"key": "limits", "lead": L["limits"], "text": _p_limits(view)}]
        return Character(header=header, paragraphs=paragraphs)
    from .mbti import load_config
    type_names = load_config().get("type_names_ru") or {}

    paragraphs = [{"key": "short", "lead": L["short"], "text": _p_short(mb, T, lex, ps, view)},
                  {"key": "basis", "lead": L["basis"], "text": _p_basis(view, mb, T)}]
    order = _by_distance([k for k in MBTI_TRAITS if ps.get(k) is not None], ps, view)
    for k in order:
        para = _p_trait(k, view, mb, T, lex, ps)
        if para:
            paragraphs.append(para)
    st = _p_stability(view, mb, T, lex, ps)
    if st:
        paragraphs.append(st)
    if mb:
        paragraphs.append({"key": "mbti", "lead": L["mbti"], "text": _p_mbti(mb, T, type_names)})
    beh = _p_behavior(view, T, ps)
    if beh:
        paragraphs.append({"key": "behavior", "lead": L["behavior"], "text": beh})
    paragraphs.append({"key": "limits", "lead": L["limits"], "text": _p_limits(view)})
    return Character(header=header, paragraphs=paragraphs)
