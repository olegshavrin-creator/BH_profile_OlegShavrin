"""Russian texts for the web page and the PDF, whatever language is spoken in the video.

The models work in English (behaviour descriptions from the video-language model, word attributions of the text
branch, Whisper run with lang=en), but every text a person reads is Russian. result.json keeps the English originals
and gets the Russian versions next to them:

  behavior_description_ru          translation of behavior_description (every speech language)
  transcript_ru                    translation of the English transcript (lang=en; a Russian transcript is shown as is)
  analyses.speech.vocabulary_ru    frequent words of English speech in Russian: [word, count, [English words]]
  explanation.json readable_words  attributed content words with their Russian translation (words.py)

Each of them has a "<field>_by" next to it: "ollama" (the local LLM translated it), "ollama+marian" (the LLM was
reachable but rejected a piece, which Marian translated) or "marian" (Ollama was unreachable). A "marian" text, or one
without the mark (made before the LLM translated these texts), is translated again as soon as Ollama answers.

The pipeline fills them once. A job processed before these fields existed gets them on the first render of the page or
the first PDF export; they are stored back into the job folder, so later renders and exports do not translate again.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import List, Tuple

log = logging.getLogger("bs2.ru_texts")

# one short note above a translated transcript (page tab «Речь» and PDF section «Транскрипт речи»); the job only says
# which language was chosen, not which one is spoken
TRANSCRIPT_NOTE = "Ролик обработан как англоязычный: ниже — автоматический перевод транскрипта на русский."
WRONG_LANGUAGE_NOTE = ("В транскрипте есть русские слова: похоже, человек говорит по-русски, а при запуске был выбран "
                       "английский язык речи. Выберите «русский» и запустите анализ заново — оценки и транскрипт будут точнее.")
NO_TRANSCRIPT_TRANSLATION = "Перевод транскрипта на русский сейчас недоступен. Откройте результат ещё раз чуть позже."
VOCABULARY_TOP = 15
MIN_FREQUENT = 2          # «частые слова» / «чаще всего звучат»: a word said once is not frequent
# English weekdays and months are capitalised in any sentence; their Russian names are not
_NOT_NAMES = set("monday tuesday wednesday thursday friday saturday sunday january february march april may june july "
                 "august september october november december".split())
_TITLE_GLUE = set("a an the of and or in on at to for with my your our his her its their".split())


def _lang(rep: dict) -> str:
    return (rep.get("model") or {}).get("lang", "ru")


def _retry(by) -> bool:
    """A stored translation is made again: Marian's (Ollama was unreachable then) or unmarked, and Ollama answers now."""
    if by in ("ollama", "ollama+marian"):
        return False
    from .translate import ollama_available
    return ollama_available()


def _speaker_note(rep: dict) -> str:
    """The speaker's grammatical gender for the transcript translation, from the behaviour description of the video
    ('The woman appears …'); '' when it is not clear."""
    desc = str(rep.get("behavior_description") or "").lower()
    fem = len(re.findall(r"\b(?:woman|she|her|hers|herself|girl|lady)\b", desc))
    masc = len(re.findall(r"\b(?:man|he|his|him|himself|boy|guy)\b", desc))
    if fem >= 2 and fem >= 3 * masc:
        return "The speaker is a woman: when she speaks about herself use feminine forms («я была», «я пришла»)."
    if masc >= 2 and masc >= 3 * fem:
        return "The speaker is a man: when he speaks about himself use masculine forms («я был», «я пришёл»)."
    return ""


def _title_words(text: str) -> set:
    """Lower-cased words that appear inside a run of capitalised words looking like a title ('Draw My Life',
    'A Fistful of Dollars'): three or more words, or a short function word inside. 'Clint Eastwood' is not a title."""
    out = set()
    for sent in re.split(r"(?<=[.!?])\s+", text or ""):
        toks = re.findall(r"[A-Za-z][A-Za-z'\-]*", sent)
        run: List[str] = []
        for i, t in enumerate(toks + [""]):
            cap = i > 0 and t[:1].isupper() and t != "I"
            glue = bool(run) and t.lower() in _TITLE_GLUE and i + 1 < len(toks) and toks[i + 1][:1].isupper()
            if cap or glue:
                run.append(t)
                continue
            while run and run[-1].lower() in _TITLE_GLUE and not run[-1][:1].isupper():
                run.pop()
            if len(run) >= 3 or (len(run) >= 2 and any(w.lower() in _TITLE_GLUE for w in run)):
                out.update(w.lower() for w in run)
            run = []
    return out


def write_json(path: str | Path, data: dict) -> None:
    """Atomic rewrite of a job file (a render and a PDF export may read it at the same time)."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except Exception as e:  # noqa: BLE001  (a read-only job folder must not break the page)
        log.warning("could not store %s: %s", path.name, str(e)[:120])


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


def ensure_transcript(rep: dict) -> bool:
    text = rep.get("transcript") or ""
    had = bool(rep.get("transcript_ru"))
    if _lang(rep) != "en" or not text.strip() or (had and not _retry(rep.get("transcript_ru_by"))):
        return False
    try:
        from .translate import translate_transcript
        ru, by = translate_transcript(text, note=_speaker_note(rep))
    except Exception as e:  # noqa: BLE001
        log.warning("transcript translation failed: %s", str(e).splitlines()[0][:120])
        return False
    if not ru or (had and by == "marian"):
        return False
    rep["transcript_ru"], rep["transcript_ru_by"] = ru, by
    return True


def ensure_vocabulary(rep: dict) -> bool:
    """English speech: content words of the whole transcript translated as dictionary entries; words with the same
    translation are merged (work / working -> «работа»), so the counts are counts of the Russian word."""
    sp = (rep.get("analyses") or {}).get("speech")
    text = rep.get("transcript") or ""
    if _lang(rep) != "en" or not sp or ("vocabulary_ru" in sp and not _retry(sp.get("vocabulary_ru_by"))):
        return False
    from .analyses.speech_stats import vocabulary
    cands = vocabulary(text, top=3 * VOCABULARY_TOP, lang="en")
    if not cands:
        sp["vocabulary_ru"], sp["vocabulary_ru_by"] = [], "ollama"
        return True
    info: dict = {}
    try:
        from .translate import translate_words
        tr = translate_words([w for w, _ in cands], "en", "ru", context=text, info=info)
    except Exception as e:  # noqa: BLE001
        log.warning("vocabulary translation failed: %s", str(e).splitlines()[0][:120])
        return False
    if "vocabulary_ru" in sp and info.get("by") != "ollama":
        return False
    # two different words with one translation are merged only when they share a stem (work / working); otherwise the
    # later one is checked with Marian on its own, which keeps 'aunt' from joining 'uncle' as «дядя»
    first: dict = {}
    for w, _ in cands:
        ru = (tr.get(w) or "").strip().lower()
        if not ru:
            continue
        other = first.setdefault(ru, w)
        if other != w and w[:4] != other[:4]:
            try:
                from .translate import translate_sentences, valid_translation
                alone = translate_sentences([w], "en", "ru")[0].strip().strip(".").strip()
                if valid_translation(alone, "ru") and alone.lower() != ru and len(alone.split()) <= 3:
                    tr[w] = alone
            except Exception:  # noqa: BLE001
                pass
    merged: dict = {}
    titles = _title_words(text)
    for w, n in cands:
        ru = (tr.get(w) or "").strip()
        if not ru:
            continue
        # a name keeps its capital letter: capitalised inside a sentence and never in lower case, not a weekday or a
        # month, not a word of a title ('Draw My Life'), and marked as a name by the dictionary when it answered
        lower = re.search(rf"(?<![A-Za-z']){re.escape(w)}(?![A-Za-z'])", text)
        name = re.search(rf"[A-Za-z,;]\s+{re.escape(w.capitalize())}(?![A-Za-z'])", text)
        name = (name and not lower and w not in _NOT_NAMES and w not in titles
                and (info.get("by") != "ollama" or w in info.get("names", ())))
        ru = ru[:1].upper() + ru[1:] if name else ru.lower()
        item = merged.setdefault(ru.lower(), [ru, 0, []])
        item[1] += int(n)
        item[2].append(w)
    sp["vocabulary_ru"] = sorted(merged.values(), key=lambda it: -it[1])[:VOCABULARY_TOP]
    sp["vocabulary_ru_by"] = info.get("by", "marian")
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
            # the form in which the word was spoken
            ctx = rep.get("transcript_en") if lang != "en" else (expl.get("transcript") or rep.get("transcript"))
            spoken = rep.get("transcript") if lang != "en" else None
        else:
            ctx, spoken = rep.get("behavior_description"), None
        info: dict = {}
        try:
            new = readable_words(expl, key, lang, transcript_ru=spoken, context_en=ctx, info=info)
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
    rep_changed = False
    for fill in (ensure_behavior, ensure_transcript, ensure_vocabulary):
        rep_changed = fill(rep) or rep_changed
    expl_changed = ensure_words(rep, expl)
    if expl_changed and rep.get("narrative"):
        # the stored plain-language paragraph names the attributed words: rebuild it with the Russian ones
        from .narrative import build_narrative
        rep["narrative"] = build_narrative(rep, expl)
        rep_changed = True
    if rep_changed or expl_changed:
        log.info("Russian texts added in %.1f s", time.time() - t0)
    return rep_changed, expl_changed


def ensure_russian_job(job: str | Path, rep: dict, expl: dict | None = None) -> None:
    """ensure_russian for a finished job; whatever was added is stored back into result.json / explanation.json."""
    job = Path(job)
    rep_changed, expl_changed = ensure_russian(rep, expl)
    if rep_changed and (job / "result.json").exists():
        write_json(job / "result.json", rep)
    if expl_changed and (job / "explain").is_dir():
        write_json(job / "explain" / "explanation.json", expl)


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
        from .analyses.speech_stats import vocabulary
        text = rep.get("transcript") or ""
        items = vocabulary(text, top=VOCABULARY_TOP, lang="ru") if (sp and text.strip()) else \
            [(w, n) for w, n in sp.get("vocabulary") or []]
    else:
        items = [(it[0], it[1]) for it in sp.get("vocabulary_ru") or []]
    return [(w, n) for w, n in items if int(n) >= MIN_FREQUENT]
