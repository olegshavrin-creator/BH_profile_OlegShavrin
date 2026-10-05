"""Human-readable version of the word attributions.

The text branch works on the English translation, so the raw attribution is a list of English tokens dominated by
function words ('to', 'your', 'on'). For the interface we keep content words only, translate them into Russian as
dictionary entries (context-aware) whatever the speech language, and for a Russian transcript map each one back to the
word actually spoken (prefix match on the stem), grouped by the direction of the effect. The page and the PDF show
Russian words only; a word without a translation is left out.

The words of the speech itself are counted here too (spoken_words, vocabulary): the speech analytics of the pipeline
(analyses/speech_stats) and the frequent words of the «Речь» tab and the PDF (ru_texts.vocabulary_shown) split a
transcript into words the same way."""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Tuple

from .textfmt import clean_word

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


def readable_words(expl: dict, key: str, transcript_ru: str | None = None, context_en: str | None = None,
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


# ---------------------------------------------------------------- the words of the speech
_WORD = re.compile(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё'\-]*")


def spoken_words(text: str) -> List[str]:
    """The words of a transcript, lower-cased: letters of either alphabet, with an inner apostrophe or hyphen."""
    return [w.lower() for w in _WORD.findall(text or "")]


def vocabulary(text: str, top: int = 12) -> List[Tuple[str, int]]:
    """Most frequent content words of Russian speech (short/function words and fillers removed)."""
    stop = set("""и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было вот от меня
    еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас нибудь опять уж вам ведь там потом себя
    ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз тоже себе под будет ж тогда
    кто этот того потому этого какой совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех никогда
    можно при наконец два об другой хоть после над больше тот через эти нас про всего них какая много разве три эту моя
    впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой им более всегда конечно всю между это которые который
    очень the a an and or of to in on at for with is are was were be it this that i you he she we they my your
    самый самая самое самые самого самой самому самым самом самых самыми свой своя своё свое свои своего своей своему
    своим своём своем своих своими числе включая также которая которое которого которой которому котором которым
    которых которыми которую этот этих этим этими этому такая такое такие такого таких таким такими какие каких какое
    каким какую всех всем всеми весь вся всё просто именно лишь нужно будут буду будем была были было могу может могут
    можем хотя тоже ещё чтобы между вообще сейчас потому поэтому где-то что-то кто-то наш наша наше наши нашего нашей
    наших нашим мои моих моей моего ваш ваша ваши него неё нему ними""".split())
    words = [w for w in spoken_words(text) if len(w) > 3 and w not in stop]
    return Counter(words).most_common(top)
