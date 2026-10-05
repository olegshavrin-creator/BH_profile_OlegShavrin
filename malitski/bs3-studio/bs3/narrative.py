"""Plain-language texts about a result, built deterministically from the numbers (no LLM): «Как получены оценки»
(method_notes: which model gave the scores, the scale, how stable they are over the video) and the paragraphs about
the words the own model reacted to (words_summary)."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from . import MODEL_TITLES
from .norms import RU_TITLES, TRAIT_KEYS
from .segments import odd_segments, scored
from .textfmt import plural_ru, seg_label

# where the scores come from, by the model that ran (one model per analysis since 3.1)
SOURCE_RU = {
    "oceanai": "Оценки дала система OCEAN-AI на весах MuPTA, обученных на русскоязычных участниках.",
    "mm": ("Оценки дала модель AMLAI 1.0, построенная по рецепту MM-PSYCHE и обученная на First Impressions V2; "
           "транскрипт русской речи для неё переведён на английский."),
}
SCALE_RU = "Уровни черт и буквы MBTI считаются по самой оценке модели на шкале от 0 до 1 с серединой 0.5."
# the one note of the tab «Объяснения» and of the PDF (under section 4) for an OCEAN-AI job (change request 3.1,
# section 3, the owner's final decision): explanations exist for AMLAI 1.0 only
NO_EXPLAIN_RU = ("Модель OCEAN-AI не строит объяснений: ключевые кадры, вклад модальностей и слова, повлиявшие на "
                 "оценку, есть только для модели AMLAI 1.0.")


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


def method_notes(view: dict) -> str:
    """«Как получены оценки» (design 9; one model since 3.1): which model gave the scores, the scale of levels and
    letters, how stable the scores are over the video and the segments without a score (C13). Built on the clean view
    (scores.clean_view); replaces the Big Five part of the 2.0 summary."""
    from . import caveats
    meta = view.get("view_meta") or {}
    main = meta.get("main_system")
    parts: List[str] = []
    if meta.get("primary_missing"):
        parts.append(caveats.text("C20"))
    else:
        parts.append(SOURCE_RU.get(main) or f"Оценки дала система {MODEL_TITLES.get(main, main)}.")
    parts.append(SCALE_RU)

    # stability over the segments the model scored
    tl = scored(view)
    std = view.get("scores_std_across_segments") or view.get("scores_std") or {}
    if tl and std:
        n = len(tl)
        count = f"{n} {plural_ru(n, 'отрезок', 'отрезка', 'отрезков')} с оценкой {MODEL_TITLES.get(main, main)}"
        worst = max(TRAIT_KEYS, key=lambda k: std.get(k, 0))
        if std.get(worst, 0) <= 0.05:
            parts.append(f"По ходу ролика ({count}) оценки устойчивы: разброс не больше ±{std[worst]:.2f}.")
        else:
            parts.append(f"По ходу ролика ({count}) оценки в целом устойчивы, сильнее всего колеблется "
                         f"{_name(worst)} (±{std[worst]:.2f}).")
        for t, z_ in odd_segments(view)[:2]:
            parts.append(f"Заметно отличается отрезок {seg_label(t['start'], t['end'])}: оценки "
                         f"{'выше' if z_ > 0 else 'ниже'} остального ролика.")

    dropped = meta.get("segments_without_primary") or []
    if dropped:
        parts.append(caveats.c13(len(dropped), int(meta.get("segments_total") or len(dropped)), main))
    return " ".join(parts)


def words_summary(rw_all: Dict[str, dict], expl: dict | None, titles: Dict[str, str]) -> List[str]:
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
        top_up, top_down = top_directions(*word_effects(rw))
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
        out.append("Что это значит: модель AMLAI 1.0 судит в основном по лицу и голосу, а слова показывают, на какие "
                   "формулировки она откликается; итоговые оценки от слов почти не зависят." if weak_all else
                   "«Поднимали» — слово сдвигало оценку черты вверх, «снижали» — вниз; это реакция модели AMLAI 1.0, а не "
                   "смысл слов сам по себе.")
    return out
