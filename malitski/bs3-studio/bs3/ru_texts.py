"""Russian texts for the web page and the PDF.

The models work in English (behaviour descriptions from the video-language model, word attributions of the text
branch), but every text a person reads is Russian. result.json keeps the English originals and gets the Russian
versions next to them:

  behavior_description_ru          translation of behavior_description
  explanation.json readable_words  attributed content words with their Russian translation (words.py)

Each of them has a "<field>_by" next to it: "ollama" (the local LLM translated it), "ollama+marian" (the LLM was
reachable but rejected a piece, which Marian translated) or "marian" (Ollama was unreachable). A "marian" text, or one
without the mark (made before the LLM translated these texts), is translated again as soon as Ollama answers.

The pipeline fills them once. A job processed before these fields existed gets them on the first render of the page or
the first PDF export; they are stored back into the job folder, so later renders and exports do not translate again.

The speech is Russian (3.1): a Russian transcript is shown as is. Older jobs processed as English speech carry two more
Russian fields, which are read-only compatibility: they are shown as stored and never made or translated again.

  transcript_ru                    translation of the English transcript (transcript_shown)
  analyses.speech.vocabulary_ru    frequent words of the English speech in Russian: [word, count, [English words]]
                                   (vocabulary_shown)
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import List, Tuple

from . import jobfiles

log = logging.getLogger("bs3.ru_texts")

# one short note above a translated transcript (page tab «Речь» and PDF section «Транскрипт речи»); the job only says
# which language was chosen, not which one is spoken
TRANSCRIPT_NOTE = "Ролик обработан как англоязычный: ниже — автоматический перевод транскрипта на русский."
WRONG_LANGUAGE_NOTE = ("В транскрипте есть русские слова: похоже, человек говорит по-русски, а ролик был обработан как "
                       "англоязычный (задание старой версии). Запустите анализ заново — оценки и транскрипт будут точнее.")
NO_TRANSCRIPT_TRANSLATION = "Перевод транскрипта на русский сейчас недоступен. Откройте результат ещё раз чуть позже."
VOCABULARY_TOP = 15
MIN_FREQUENT = 2          # «частые слова» / «чаще всего звучат»: a word said once is not frequent


def _lang(rep: dict) -> str:
    return (rep.get("model") or {}).get("lang", "ru")


def _retry(by) -> bool:
    """A stored translation is made again: Marian's (Ollama was unreachable then) or unmarked, and Ollama answers now."""
    if by in ("ollama", "ollama+marian"):
        return False
    from .translate import ollama_available
    return ollama_available()


def write_json(path: str | Path, data: dict) -> None:
    """Stores a job file the page added Russian texts to (jobfiles.write_json, atomic: a render and a PDF export may
    read it at the same time). Never raises: a read-only job folder must not break the page, the failure is logged."""
    try:
        jobfiles.write_json(path, data)
    except Exception as e:  # noqa: BLE001
        log.warning("could not store %s: %s", Path(path).name, str(e)[:120])


# ---------------------------------------------------------------- fill the missing Russian texts
def ensure_behavior(rep: dict) -> bool:
    desc = rep.get("behavior_description")
    had = bool(rep.get("behavior_description_ru"))
    if not desc or (had and not _retry(rep.get("behavior_description_ru_by"))):
        return False
    try:
        from .translate import translate_description
        ru, by = translate_description(desc)
    except Exception as e:  # noqa: BLE001
        log.warning("description translation failed: %s", str(e).splitlines()[0][:120])
        return False
    if had and by == "marian":          # the retry fell back to Marian again: nothing better to store
        return False
    rep["behavior_description_ru"], rep["behavior_description_ru_by"] = ru, by
    return True


def ensure_words(rep: dict, expl: dict | None) -> bool:
    """explanation.json readable_words with Russian translations. Recomputed when a list is missing, was made without
    translations (English speech before translations were stored) or by Marian while Ollama was unreachable."""
    if not expl:
        return False
    from .words import readable_words
    lang = _lang(rep)
    rw_all = dict(expl.get("readable_words") or {})
    retry = any(k in expl for k in ("transcript_words", "behavior_words")) and _retry(expl.get("readable_words_by"))
    changed, by = False, "ollama"
    for key in ("transcript_words", "behavior_words"):
        if key not in expl:
            continue
        items = [i for d in (rw_all.get(key) or {}).values() for i in (d.get("up") or []) + (d.get("down") or [])]
        if key in rw_all and (not items or any(i.get("ru") for i in items)) and not retry:
            continue
        if key == "transcript_words":
            # the sense of a word is taken from the English text it comes from; the Russian transcript (lang=ru) gives
            # the form in which the word was spoken. Russian speech has no English context: no analyzer returns a
            # `transcript_en` into result.json, so the context read from there was always None
            ctx = None if lang != "en" else (expl.get("transcript") or rep.get("transcript"))
            spoken = rep.get("transcript") if lang != "en" else None
        else:
            ctx, spoken = rep.get("behavior_description"), None
        info: dict = {}
        try:
            new = readable_words(expl, key, transcript_ru=spoken, context_en=ctx, info=info)
        except Exception as e:  # noqa: BLE001
            log.warning("readable words failed for %s: %s", key, str(e).splitlines()[0][:120])
            continue
        if key in rw_all and retry and info.get("by") != "ollama":
            continue                    # the retry fell back to Marian again: keep the stored list
        rw_all[key] = new
        changed = True
        by = by if info.get("by") == "ollama" else "marian"
    if changed:
        expl["readable_words"] = rw_all
        expl["readable_words_by"] = by
    return changed


def ensure_russian(rep: dict, expl: dict | None = None) -> Tuple[bool, bool]:
    """Fills every missing Russian text in place. Returns (result.json changed, explanation.json changed)."""
    t0 = time.time()
    rep_changed = ensure_behavior(rep)
    expl_changed = ensure_words(rep, expl)
    if rep_changed or expl_changed:
        log.info("Russian texts added in %.1f s", time.time() - t0)
    return rep_changed, expl_changed


def ensure_russian_job(job: str | Path, rep: dict, expl: dict | None = None) -> None:
    """ensure_russian for a finished job; whatever was added is stored back into result.json / explanation.json."""
    job = Path(job)
    rep_changed, expl_changed = ensure_russian(rep, expl)
    if rep_changed and (job / jobfiles.RESULT).exists():
        write_json(job / jobfiles.RESULT, rep)
    if expl_changed and (job / jobfiles.EXPLAIN_DIR).is_dir():
        write_json(jobfiles.explanation_path(job), expl)


# ---------------------------------------------------------------- what the page and the PDF show
def transcript_shown(rep: dict) -> Tuple[str, str]:
    """(note, text) for the transcript block: a job processed as English -> the translation with a one-line note, plus
    a warning when the recognised text has Russian words. Whisper forced to English translates Russian speech and
    leaves some Russian words in (5% of the words of the Russian interview processed as English); English speech
    gives none."""
    text = rep.get("transcript") or ""
    if _lang(rep) != "en":
        return "", text
    words = re.findall(r"[A-Za-zА-Яа-яЁё]+", text)
    n_ru = sum(1 for w in words if re.search(r"[А-Яа-яЁё]", w))
    warn = (" " + WRONG_LANGUAGE_NOTE) if (n_ru >= 5 and n_ru >= 0.03 * len(words)) else ""
    if rep.get("transcript_ru"):
        return TRANSCRIPT_NOTE + warn, rep["transcript_ru"]
    return (NO_TRANSCRIPT_TRANSLATION + warn, "") if text.strip() else ("", "")


def vocabulary_shown(rep: dict) -> List[Tuple[str, int]]:
    """Frequent spoken words in Russian: as spoken for Russian speech, translated and merged for English speech. Only
    words said at least MIN_FREQUENT times: in a short clip every word is said once and none of them is «частое»."""
    sp = (rep.get("analyses") or {}).get("speech") or {}
    if _lang(rep) != "en":
        # counted again from the transcript (no model, instant), so older jobs follow the current list of function words
        from .words import vocabulary
        text = rep.get("transcript") or ""
        items = vocabulary(text, top=VOCABULARY_TOP) if (sp and text.strip()) else \
            [(w, n) for w, n in sp.get("vocabulary") or []]
    else:
        items = [(it[0], it[1]) for it in sp.get("vocabulary_ru") or []]
    return [(w, n) for w, n in items if int(n) >= MIN_FREQUENT]
