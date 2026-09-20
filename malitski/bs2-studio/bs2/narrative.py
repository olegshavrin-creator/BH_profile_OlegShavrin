"""Plain-language explanation of a result, built deterministically from the numbers (no LLM): what the scores are
based on, which traits stand out, how stable the video is, what the own model looked at, which words mattered."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .norms import RU_TITLES, TRAIT_KEYS
from .report import seg_label

MOD_RU = {"face": "лицо", "audio": "голос", "audio_whisper": "голос", "audio_xlsr": "голос", "audio_w2v_emo": "голос",
          "text": "содержание речи", "behavior": "описание поведения"}


SMALL_POOL = 20          # below this many processed videos a percentage only looks precise: say it in words
SYSTEM_RU = {"oceanai": "OCEAN-AI", "mm": "своя модель (MM-PSYCHE)", "scene": "SSL-MEPR (сцена)"}
# grammatical gender of the modality names, for «почти не повлиял / повлияло»
MOD_GENDER = {"лицо": "n", "голос": "m", "содержание речи": "n", "описание поведения": "n"}


def plural_ru(n: int, forms: tuple) -> str:
    """plural_ru(21, ("отрезок", "отрезка", "отрезков")) -> "отрезок"."""
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        return forms[0]
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return forms[1]
    return forms[2]


def _pool_size(ref: str):
    import re
    m = re.search(r"N\s*=\s*(\d+)", ref or "")
    return int(m.group(1)) if m else None


def _pct_phrase(t: dict, what: str = "роликов") -> str:
    """Same wording as the score bars: a percentage for large reference groups, words for a small pool."""
    pct = t.get("percentile")
    if pct is None:
        return ""
    ref = t.get("percentile_ref", "")
    if "пула" in ref:
        group = "русских роликов" if "русских" in ref else ("английских роликов" if "английских" in ref else "обработанных роликов")
    else:
        group = "людей в First Impressions V2"
    p = max(0.0, min(100.0, float(pct)))
    n = _pool_size(ref)
    if "пула" in ref and n is not None and n < SMALL_POOL:
        if p > 60:
            return f"выше, чем у большинства из {n} {group}"
        if p < 40:
            return f"ниже, чем у большинства из {n} {group}"
        return f"примерно посередине среди {n} {group}"
    if 45 <= p <= 55:
        return f"примерно посередине среди {group}"
    return f"выше, чем у {p:.0f}% {group}" if p > 50 else f"ниже, чем у {100 - p:.0f}% {group}"


def _name(k: str) -> str:
    return RU_TITLES[k].lower()


def word_effects(rw: Dict[str, dict]) -> tuple:
    """({word: largest |effect|} of the words that raised a score, {…} of those that lowered one) over all traits of
    a readable_words list; Russian words only (an item without a translation is left out)."""
    from .words import shown_word
    ups: Dict[str, float] = {}
    downs: Dict[str, float] = {}
    for d in rw.values():
        for items, acc in ((d["up"], ups), (d["down"], downs)):
            for i in items:
                w = shown_word(i)
                if w:
                    acc[w] = max(acc.get(w, 0.0), abs(i["signed"]))
    return ups, downs


def top_directions(ups: Dict[str, float], downs: Dict[str, float], k: int = 4) -> tuple:
    """The k strongest words of each direction. A word that raised some scores and lowered others names no direction
    and is left out of both lists («в сторону повышения — «продукт»; понижения — «продукт»» is no information)."""
    both = set(ups) & set(downs)
    top_up = [w for w, _ in sorted(ups.items(), key=lambda kv: -kv[1]) if w not in both][:k]
    top_down = [w for w, _ in sorted(downs.items(), key=lambda kv: -kv[1]) if w not in both][:k]
    return top_up, top_down


def odd_segments(rep: dict) -> List[tuple]:
    """[(timeline entry, z)] of the segments whose mean of the five scores lies more than 2 standard deviations from
    the other segments (at least 4 scored segments), in time order; z > 0 — the scores are higher."""
    tl = [t for t in (rep.get("timeline") or []) if t.get("scores")]
    means = np.array([np.mean([t["scores"][k] for k in TRAIT_KEYS]) for t in tl])
    if len(means) < 4 or means.std() <= 0:
        return []
    z = (means - means.mean()) / means.std()
    return [(t, float(z_)) for t, z_ in zip(tl, z) if abs(z_) > 2.0]


def build_narrative(rep: dict, expl: dict | None = None) -> str:
    traits = rep["traits"]
    model = rep.get("model") or {}
    lang, primary = model.get("lang", "en"), model.get("primary")
    parts: List[str] = []

    # 1. what the scores come from
    if primary == "oceanai":
        parts.append("Основные оценки дала система OCEAN-AI на весах MuPTA, обученных на русскоязычных участниках; своя модель "
                     "показана как второе мнение.")
    elif primary:
        parts.append(f"Основные оценки дала система {SYSTEM_RU.get(primary, primary)}.")
    else:
        parts.append("Оценки — среднее двух систем (OCEAN-AI и своей модели), обе на шкале First Impressions V2.")

    # 2. traits that stand out
    order = sorted(TRAIT_KEYS, key=lambda k: -traits[k]["score"])
    hi1, hi2, lo = order[0], order[1], order[-1]

    def tr(k):
        p = _pct_phrase(traits[k])
        return f"{_name(k)} ({traits[k]['score']:.2f}" + (f", {p}" if p else "") + ")"
    parts.append(f"Сильнее всего выражены {tr(hi1)} и {tr(hi2)}; слабее всего — {tr(lo)}.")

    # 3. interview impression
    iv = rep.get("interview")
    if iv:
        p = _pct_phrase(iv)
        parts.append(f"Впечатление «пригласить на собеседование» по своей модели: {iv['score']:.2f}"
                     + (f", {p}" if p else "") + ".")

    # 4. stability across segments
    tl = [t for t in (rep.get("timeline") or []) if t.get("scores")]
    std = rep.get("scores_std_across_segments") or rep.get("scores_std") or {}
    if tl and std:
        worst = max(TRAIT_KEYS, key=lambda k: std.get(k, 0))
        if std.get(worst, 0) <= 0.05:
            parts.append(f"По ходу ролика ({len(tl)} {plural_ru(len(tl), ('отрезок', 'отрезка', 'отрезков'))}) оценки устойчивы: "
                         f"разброс не больше ±{std[worst]:.2f}.")
        else:
            parts.append(f"По ходу ролика ({len(tl)} {plural_ru(len(tl), ('отрезок', 'отрезка', 'отрезков'))}) оценки в целом "
                         f"устойчивы, сильнее всего колеблется "
                         f"{_name(worst)} (±{std[worst]:.2f}).")
        for t, z_ in odd_segments(rep)[:2]:
            parts.append(f"Заметно отличается отрезок {seg_label(t['start'], t['end'])}: оценки "
                         f"{'выше' if z_ > 0 else 'ниже'} остального ролика.")

    # 5. what the own model looked at
    if expl and expl.get("modalities", {}).get("input_x_gradient"):
        ixg = expl["modalities"]["input_x_gradient"]
        mods = list(next(iter(ixg.values())).keys())
        share = {m: float(np.mean([row[m]["share"] for row in ixg.values()])) for m in mods}
        main = [m for m in sorted(mods, key=lambda m: -share[m]) if share[m] >= 0.01]
        weak = [m for m in mods if share[m] < 0.01]
        named = [f"{MOD_RU.get(m, m)} ({share[m] * 100:.0f}%)" for m in main]
        # «на лицо (57%), голос (37%) и описание поведения (4%)», not «… и … и …»
        s = "Своя модель опиралась в основном на " + (", ".join(named[:-1]) + " и " + named[-1] if len(named) > 1
                                                      else "".join(named))
        if weak:
            names = list(dict.fromkeys(MOD_RU.get(m, m) for m in weak))
            verb = "почти не повлияли" if len(names) > 1 else (
                "почти не повлиял" if MOD_GENDER.get(names[0]) == "m" else "почти не повлияло")
            s += "; " + " и ".join(names) + f" {verb} (меньше 1%)"
        parts.append(s + ".")

    # 6. words (union over traits, largest effects), Russian only whatever the speech language; the same lists as the
    # block «Слова, на которые откликнулась модель» (words_summary)
    rw = (expl or {}).get("readable_words", {}).get("transcript_words")
    if rw:
        ups, downs = top_directions(*word_effects(rw))
        if ups or downs:
            s = "Отдельные слова речи сдвигали оценки лишь незначительно"
            if ups:
                s += "; в сторону повышения — " + ", ".join(f"«{w}»" for w in ups)
            if downs:
                s += "; в сторону понижения — " + ", ".join(f"«{w}»" for w in downs)
            parts.append(s + ".")

    # 7. second opinion on another scale
    var = rep.get("variant_scores") or rep.get("variants") or {}
    if primary == "oceanai" and "mm" in var:
        diff = float(np.mean([var["oceanai"][k] - var["mm"][k] for k in TRAIT_KEYS]))
        parts.append(f"Своя модель, обученная на англоязычных влогерах, оценивает те же черты в среднем на {diff:.2f} ниже: "
                     "это разница шкал двух систем, а не противоречие в выводах.")
    return " ".join(parts)


def words_sentences(rw_all: Dict[str, dict], titles: Dict[str, str], lang: str) -> List[str]:
    """Per-trait sentences: 'Открытость опыту: повышали слова «простой»; понижали — «явление», «работе».'"""
    from .words import shown_word
    out = []
    for key, what in (("transcript_words", "речи"), ("behavior_words", "описания поведения")):
        rw = rw_all.get(key)
        if not rw:
            continue
        out.append(f"Слова {what}:")
        for trait in list(TRAIT_KEYS) + [k for k in rw if k not in TRAIT_KEYS]:
            if trait not in rw:
                continue

            ups = [f"«{w}»" for w in map(shown_word, rw[trait]["up"]) if w]
            downs = [f"«{w}»" for w in map(shown_word, rw[trait]["down"]) if w]
            if not ups and not downs:
                continue
            s = f"{titles.get(trait, trait)}: "
            if ups:
                s += "повышали " + ", ".join(ups)
            if downs:
                s += ("; понижали " if ups else "понижали ") + ", ".join(downs)
            out.append(s + ".")
        out.append("")
    return out


def words_summary(rw_all: Dict[str, dict], expl: dict | None, titles: Dict[str, str], lang: str) -> List[str]:
    """Human paragraphs about the attributed words. When the text nodes barely matter (the usual case for the
    FIV2-trained model) one paragraph per source says so and names the few words the model reacted to; per-trait
    lists are shown only when traits really react to different words."""
    shares: Dict[str, float] = {}
    ixg = ((expl or {}).get("modalities") or {}).get("input_x_gradient") or {}
    if ixg:
        mods = next(iter(ixg.values())).keys()
        for m in mods:
            shares[m] = float(np.mean([row[m]["share"] for row in ixg.values()]))

    from .words import shown_word

    def q(ws):
        return ", ".join(f"«{w}»" for w in ws)

    out: List[str] = []
    weak_all = True
    for key, head, node in (("transcript_words", "Речь", "text"), ("behavior_words", "Описание поведения", "behavior")):
        rw = rw_all.get(key)
        if not rw:
            continue
        share = shares.get(node)
        per_trait = {}
        for trait, d in rw.items():
            # Russian words only: an item without a translation is left out
            u = [w for w in map(shown_word, d["up"]) if w]
            dn = [w for w in map(shown_word, d["down"]) if w]
            per_trait[trait] = (u[:3], dn[:3])
        top_up, top_down = top_directions(*word_effects(rw))       # the same lists as the plain-language paragraph
        agree = [1.0 if (set(u) <= set(top_up) and set(dn) <= set(top_down)) else 0.0 for u, dn in per_trait.values()]
        uniform = (sum(agree) / max(1, len(agree))) >= 0.7
        weak = share is not None and share < 0.02
        weak_all = weak_all and weak
        if share is None:
            s = f"{head}."
        elif weak:
            s = (f"{head}. " + ("Содержание речи почти не повлияло" if node == "text" else "Текст описания почти не повлиял")
                 + f" на оценки: вклад меньше {'1' if share < 0.01 else '2'}%.")
        else:
            s = f"{head}. Вклад в оценки около {share * 100:.0f}%."
        if uniform or weak:
            if len(top_up) == 1 and not top_down:
                s += f" Единственное слово, на которое модель заметно отреагировала, — {q(top_up)}: оно немного подняло все оценки."
            else:
                if top_up:
                    s += f" Немного поднимали все оценки слова {q(top_up)}"
                    s += f", немного снижали — {q(top_down)}." if top_down else "."
                elif top_down:
                    s += f" Немного снижали все оценки слова {q(top_down)}."
            if top_up and top_down:
                s += " Направление одинаково для всех черт: модель откликается на общий тон текста, а не на отдельные черты."
        else:
            lines = []
            for trait, (u, dn) in per_trait.items():
                if not u and not dn:
                    continue
                part = f"{titles.get(trait, trait)}: "
                if u:
                    part += f"поднимали {q(u)}"
                if dn:
                    part += (", снижали " if u else "снижали ") + q(dn)
                lines.append(part + ".")
            s += " Разные черты реагируют на разные слова. " + " ".join(lines)
        out.append(s)
    if out:
        out.append("Что это значит: своя модель судит в основном по лицу и голосу, а слова показывают, на какие формулировки она "
                   "откликается; итоговые оценки от слов почти не зависят." if weak_all else
                   "«Поднимали» — слово сдвигало оценку черты вверх, «снижали» — вниз; это реакция своей модели, а не смысл слов сам по себе.")
    return out
