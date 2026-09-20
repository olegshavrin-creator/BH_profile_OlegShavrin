"""Human-readable version of the word attributions.

The text branch works on the English translation, so the raw attribution is a list of English tokens dominated by
function words ('to', 'your', 'on'). For the interface we keep content words only, translate them into Russian as
dictionary entries (context-aware) whatever the speech language, and for a Russian transcript map each one back to the
word actually spoken (prefix match on the stem), grouped by the direction of the effect. The page and the PDF show
Russian words only; a word without a translation is left out."""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List

from .norms import TRAIT_KEYS
from .report import clean_word

STOP_EN = set("""
a an the and or but if so as of to in on at by for from with without into onto over under about above below between
through during before after again further then once here there when where why how all any both each few more most
other some such no nor not only own same than too very can will just don should now is are was were be been being
have has had having do does did doing i me my myself we our ours ourselves you your yours yourself yourselves he him
his himself she her hers herself it its itself they them their theirs themselves what which who whom this that these
those am would could might must shall may also however therefore thus yes yeah okay ok well like um uh oh hmm
because while although though even still already yet ever never always often sometimes maybe perhaps quite rather
really actually basically literally something anything nothing everything someone anyone everyone one ones thing
things way lot lots much many little less least going get got gets getting go goes went come comes came make makes
made say says said tell told know knew think thought want wanted need let us gonna wanna kind sort bit
almost seem seems seemed seeming appear appears appeared appearing show shows showed shown showing inaudible
""".split())
_RU_WORD = re.compile(r"[А-Яа-яЁё]{3,}")


def _content_word(w: str) -> bool:
    w = clean_word(w).lower()
    return len(w) >= 3 and w.isalpha() and w not in STOP_EN


def _source_word(ru_lemma: str, transcript_ru: str, counts: Counter) -> str | None:
    """The word as spoken: the most frequent transcript word sharing the lemma's stem (first 5 letters, 4 for
    short lemmas). 'сложный' -> 'сложная', 'ответ' -> 'ответить'."""
    lemma = clean_word(ru_lemma).lower().split()[0] if ru_lemma else ""
    if len(lemma) < 3:
        return None
    stem = lemma[: 5 if len(lemma) >= 6 else max(3, len(lemma) - 1)]
    best, best_n = None, 0
    for w, n in counts.items():
        if w.startswith(stem) and n > best_n:
            best, best_n = w, n
    return best


def readable_words(expl: dict, key: str, lang: str, transcript_ru: str | None = None, context_en: str | None = None,
                   top_k: int = 5, info: dict | None = None) -> Dict[str, dict]:
    """{trait: {"up": [{"ru","en","source"}...], "down": [...]}} for expl[key]; content words with a Russian
    translation only (for English speech too: the page is in Russian). `transcript_ru` is the Russian transcript as
    spoken (lang=ru), used to show the word in the form it was said; a word that matches nothing in it is left out
    (its dictionary translation was never said). `info` receives the translator ("by", see translate_words)."""
    per = expl.get(key, {}).get("per_output", {})
    vocab = sorted({clean_word(w["word"]) for d in per.values() for w in d["top_words"] if _content_word(w["word"])})
    ru_map: Dict[str, str] = {}
    info = info if info is not None else {}
    info.setdefault("by", "ollama")          # nothing to translate counts as done
    if vocab:
        try:
            from .translate import translate_words
            ru_map = translate_words(vocab, "en", "ru", context=context_en, info=info)
        except Exception:  # noqa: BLE001
            ru_map, info["by"] = {}, "marian"
    counts = Counter(m.group(0).lower() for m in _RU_WORD.finditer(transcript_ru or ""))
    out: Dict[str, dict] = {}
    for trait, d in per.items():
        best: Dict[str, dict] = {}          # displayed word -> item with the largest |effect| (several English
        for w in d["top_words"]:            # tokens may map to one Russian word: man / person -> человек)
            en = clean_word(w["word"])
            if not _content_word(en):
                continue
            ru = ru_map.get(en)
            if not ru:                      # an English word is never shown: no translation, no entry
                continue
            src = _source_word(ru, transcript_ru, counts) if transcript_ru else None
            if transcript_ru and not src:   # Russian speech: only words that were actually said
                continue
            shown = (src or ru).lower()
            item = {"en": en, "ru": ru, "source": src, "signed": float(w.get("signed", 0.0))}
            if shown not in best or abs(item["signed"]) > abs(best[shown]["signed"]):
                best[shown] = item
        items = sorted(best.values(), key=lambda i: -abs(i["signed"]))
        up = [i for i in items if i["signed"] >= 0][:top_k]
        down = [i for i in items if i["signed"] < 0][:top_k]
        out[trait] = {"up": up, "down": down}
    return out


def shown_word(item: dict) -> str | None:
    """The Russian word for the page and the PDF: as spoken (Russian speech) or the dictionary translation; None
    when there is no translation (lists made before translations were stored for English speech)."""
    word = item.get("source") or item.get("ru")
    return word if word and re.search(r"[A-Za-z]", word) is None else None


def _label(item: dict, lang: str, show_en: bool) -> str | None:
    word = shown_word(item)
    return f"{word} ({item['en']})" if (word and show_en) else word


def format_words(rw: Dict[str, dict], titles: Dict[str, str], lang: str, show_en: bool = False) -> List[str]:
    """One line per trait: 'Открытость опыту — выше: сложная, ответ; ниже: подход'."""
    lines = []
    for trait in list(TRAIT_KEYS) + [k for k in rw if k not in TRAIT_KEYS]:
        if trait not in rw:
            continue
        parts = []
        up = [w for w in (_label(i, lang, show_en) for i in rw[trait]["up"]) if w]
        down = [w for w in (_label(i, lang, show_en) for i in rw[trait]["down"]) if w]
        if up:
            parts.append("выше: " + ", ".join(up))
        if down:
            parts.append("ниже: " + ", ".join(down))
        lines.append(f"{titles.get(trait, trait)} — " + ("; ".join(parts) if parts else "значимых слов нет"))
    return lines


WORDS_NOTE = ("Слова из речи (или из описания поведения), сильнее всего сдвинувшие оценку своей модели: «выше» — в сторону "
              "большего значения черты, «ниже» — меньшего. Служебные слова отброшены; для русской речи показано слово, как "
              "оно прозвучало.")
