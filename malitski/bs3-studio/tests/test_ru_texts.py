"""Russian texts of older English jobs (3.1): the speech is Russian, so nothing of English speech is made or translated
any more. The Russian texts an English job carries are only read: ru_texts.ensure_russian leaves them alone and calls
no translator, transcript_shown and vocabulary_shown return them as stored."""
from __future__ import annotations

import contextlib
import copy
import inspect
import json
import tempfile
from pathlib import Path

from samples import english

import bs3.translate
from bs3 import jobfiles, ru_texts
from bs3.analyses import speech_stats

TRANSCRIPT = "I work at a small company. My work is about new tools, and I like my work. The company is small."
TRANSCRIPT_RU = ("Я работаю в небольшой компании. Моя работа связана с новыми инструментами, и мне нравится моя работа. "
                 "Компания небольшая.")
VOCABULARY_RU = [["работа", 3, ["work"]], ["компания", 2, ["company"]], ["инструмент", 1, ["tools"]]]


def _english_job(by: str = "ollama") -> tuple:
    """(result.json, explanation.json) of an older English job with every Russian text stored; `by` marks the
    translated transcript and the vocabulary, the other texts are marked "ollama"."""
    r = english()
    r["transcript"] = TRANSCRIPT
    r["transcript_ru"], r["transcript_ru_by"] = TRANSCRIPT_RU, by
    r["behavior_description"] = "[0–20 с] The person speaks calmly and keeps eye contact."
    r["behavior_description_ru"] = "[0–20 с] Человек говорит спокойно и поддерживает зрительный контакт."
    r["behavior_description_ru_by"] = "ollama"
    r["analyses"] = {"speech": {"words": 20, "vocabulary": [["work", 3], ["company", 2]],
                                "vocabulary_ru": copy.deepcopy(VOCABULARY_RU), "vocabulary_ru_by": by}}
    per = {"per_output": {"extraversion": {"top_words": [{"word": "work", "signed": 0.2}]}}}
    shown = {"extraversion": {"up": [{"en": "work", "ru": "работа", "source": None, "signed": 0.2}], "down": []}}
    expl = {"transcript": TRANSCRIPT, "transcript_words": per, "behavior_words": copy.deepcopy(per),
            "readable_words": {"transcript_words": shown, "behavior_words": copy.deepcopy(shown)},
            "readable_words_by": "ollama"}
    return r, expl


def _ensure_without_translator(rep: dict, expl: dict) -> tuple:
    """ensure_russian with every public function of bs3.translate replaced by one that fails when called."""
    saved, called = {}, []
    for name, fn in list(vars(bs3.translate).items()):
        if name.startswith("_") or not inspect.isfunction(fn):
            continue

        def stub(*_a, _name=name, **_k):
            called.append(_name)
            raise AssertionError(f"bs3.translate.{_name} was called")
        saved[name] = fn
        setattr(bs3.translate, name, stub)
    assert {"ollama_available", "translate_description", "translate_words", "translate_sentences"} <= set(saved)
    try:
        changed = ru_texts.ensure_russian(rep, expl)
    finally:
        for name, fn in saved.items():
            setattr(bs3.translate, name, fn)
    return changed, called


def test_ensure_russian_translates_nothing_of_an_english_job():
    # "marian": made by Marian while Ollama was unreachable. Before 3.1 such a transcript and vocabulary were
    # translated again as soon as Ollama answered; now they are kept as they are, without asking Ollama
    for by in ("ollama", "marian"):
        rep, expl = _english_job(by)
        rep0, expl0 = copy.deepcopy(rep), copy.deepcopy(expl)
        changed, called = _ensure_without_translator(rep, expl)
        assert changed == (False, False) and called == [], (by, called)
        assert rep == rep0 and expl == expl0, by


def test_english_job_without_translation_stays_so():
    rep, expl = _english_job()
    del rep["transcript_ru"], rep["transcript_ru_by"], rep["analyses"]["speech"]["vocabulary_ru"]
    del rep["analyses"]["speech"]["vocabulary_ru_by"]
    changed, called = _ensure_without_translator(rep, expl)
    assert changed == (False, False) and called == []
    assert "transcript_ru" not in rep and "vocabulary_ru" not in rep["analyses"]["speech"]
    assert ru_texts.transcript_shown(rep) == (ru_texts.NO_TRANSCRIPT_TRANSLATION, "")
    assert ru_texts.vocabulary_shown(rep) == []


def test_shown_texts_are_the_stored_ones():
    rep, _ = _english_job()
    assert ru_texts.transcript_shown(rep) == (ru_texts.TRANSCRIPT_NOTE, TRANSCRIPT_RU)
    # words said at least MIN_FREQUENT times, as stored (the Russian word and its count)
    assert ru_texts.vocabulary_shown(rep) == [("работа", 3), ("компания", 2)]


def test_english_speech_generation_is_gone():
    for name in ("ensure_transcript", "ensure_vocabulary", "_speaker_note", "_title_words", "_NOT_NAMES", "_TITLE_GLUE"):
        assert not hasattr(ru_texts, name), name
    for name in ("translate_transcript", "translate_long", "_script_runs", "_LAT", "_REPEAT", "_patch_latin_runs"):
        assert not hasattr(bs3.translate, name), name
    assert list(inspect.signature(bs3.translate.llm_translate).parameters) == ["texts"]
    assert list(inspect.signature(bs3.translate._llm_answer_ok).parameters) == ["en", "ru"]
    for name in ("_en_content", "STOP_EN_SPEECH"):
        assert not hasattr(speech_stats, name), name
    assert "STOP_EN" not in inspect.getsource(speech_stats)
    assert "lang" not in inspect.signature(speech_stats.vocabulary).parameters
    assert "en" in speech_stats.FILLERS                # English fillers still count in Russian speech


# ---------------------------------------------------------------- retry rule and writing only on a change

@contextlib.contextmanager
def _translate(*, up: bool = False, description=None):
    """bs3.translate for a page render: ollama_available says up/down, and translate_description records its calls and
    returns `description` (raising if it is called without one being given). The run harness' «no models» stubs are put
    back afterwards. _retry imports ollama_available from this module by name, and ensure_behavior imports
    translate_description from it, so replacing the attributes is enough."""
    calls: list = []
    saved = {n: getattr(bs3.translate, n) for n in ("ollama_available", "translate_description")}

    def _describe(text):
        calls.append(text)
        if description is None:
            raise AssertionError("bs3.translate.translate_description was called unexpectedly")
        return description

    bs3.translate.ollama_available = lambda *_a, **_k: up
    bs3.translate.translate_description = _describe
    try:
        yield calls
    finally:
        for name, fn in saved.items():
            setattr(bs3.translate, name, fn)


def test_a_stored_translation_is_remade_only_when_something_better_is_possible():
    # Marian's text (Ollama was down then) or an unmarked one is remade once Ollama answers; a text Ollama already
    # produced ("ollama"/"ollama+marian") is never remade — asking again would give the same result
    with _translate(up=True):
        assert ru_texts._retry("marian") is True
        assert ru_texts._retry(None) is True
        assert ru_texts._retry("ollama") is False
        assert ru_texts._retry("ollama+marian") is False
    with _translate(up=False):                       # Ollama unreachable: nothing better can be made now
        assert ru_texts._retry("marian") is False
        assert ru_texts._retry("ollama+marian") is False


def test_marian_behavior_is_retried_but_ollama_plus_marian_is_not():
    for by, up, retried in [("marian", True, True), ("marian", False, False),
                            ("ollama+marian", True, False), ("ollama", True, False)]:
        rep = {"behavior_description": "[0-20 s] The person is calm.",
               "behavior_description_ru": "[0–20 с] Человек спокоен.", "behavior_description_ru_by": by}
        with _translate(up=up, description=("[0–20 с] Человек говорит спокойно.", "ollama")) as calls:
            changed = ru_texts.ensure_behavior(rep)
        assert bool(calls) == retried and changed == retried, (by, up)
        if retried:
            assert rep["behavior_description_ru"] == "[0–20 с] Человек говорит спокойно."
            assert rep["behavior_description_ru_by"] == "ollama"


_EXPL = {"transcript_words": {"per_output": {"extraversion": {"top_words": [{"word": "work", "signed": 0.2}]}}},
         "readable_words": {"transcript_words": {"extraversion":
                            {"up": [{"en": "work", "ru": "работа", "source": None, "signed": 0.2}], "down": []}}},
         "readable_words_by": "ollama"}


def _ru_job(job: Path, by: str = "ollama") -> tuple:
    """A finished Russian AMLAI 1.0 job on disk: a behaviour translation in result.json and translated attributed words
    in explain/explanation.json, both already made (marked «ollama»/`by`)."""
    (job / jobfiles.EXPLAIN_DIR).mkdir(parents=True)
    rep = {"model": {"lang": "ru", "selected": "mm"},
           "behavior_description": "[0-20 s] The person is calm.",
           "behavior_description_ru": "[0–20 с] Человек спокоен.", "behavior_description_ru_by": by,
           "job_dir": str(job)}
    expl = copy.deepcopy(_EXPL)
    jobfiles.write_json(job / jobfiles.RESULT, rep)
    jobfiles.write_json(jobfiles.explanation_path(job), expl)
    return rep, expl


def test_ensure_russian_job_writes_only_on_a_change():
    with tempfile.TemporaryDirectory() as d:
        # nothing to translate (everything already «ollama»): result.json and explanation.json are left byte for byte
        job = Path(d) / "20000107_000000_deadbeef"
        rep, expl = _ru_job(job)
        rp, ep = job / jobfiles.RESULT, jobfiles.explanation_path(job)
        before = (rp.read_bytes(), ep.read_bytes(), rp.stat().st_mtime_ns, ep.stat().st_mtime_ns)
        with _translate(up=False):
            ru_texts.ensure_russian_job(job, rep, expl)
        assert (rp.read_bytes(), ep.read_bytes(), rp.stat().st_mtime_ns, ep.stat().st_mtime_ns) == before

        # a real change: a Marian behaviour is remade once Ollama answers -> result.json is rewritten, explanation is not
        job2 = Path(d) / "20000108_000000_feedface"
        rep2, expl2 = _ru_job(job2, by="marian")
        rp2, ep2 = job2 / jobfiles.RESULT, jobfiles.explanation_path(job2)
        expl_before = ep2.read_bytes()
        with _translate(up=True, description=("[0–20 с] Человек говорит спокойно.", "ollama")):
            ru_texts.ensure_russian_job(job2, rep2, expl2)
        stored = json.loads(rp2.read_bytes())
        assert stored["behavior_description_ru"] == "[0–20 с] Человек говорит спокойно."
        assert stored["behavior_description_ru_by"] == "ollama"
        assert ep2.read_bytes() == expl_before      # the attributed words were already translated: no rewrite


# ---------------------------------------------------------------- what the page and the PDF show

def test_transcript_shown_for_russian_and_english_jobs():
    # a Russian job: the recognised transcript as it is, no note
    assert ru_texts.transcript_shown({"model": {"lang": "ru"}, "transcript": "Привет, как дела?"}) \
        == ("", "Привет, как дела?")
    assert ru_texts.transcript_shown({"model": {"lang": "ru"}}) == ("", "")
    # an older English job whose recognised text kept Russian words: the translation note plus the «wrong language» warning
    note, text = ru_texts.transcript_shown({"model": {"lang": "en"},
                                            "transcript": "привет как дела хорошо спасибо", "transcript_ru": "…"})
    assert note.startswith(ru_texts.TRANSCRIPT_NOTE) and ru_texts.WRONG_LANGUAGE_NOTE in note and text == "…"


def test_vocabulary_shown_minimum_frequency():
    assert ru_texts.MIN_FREQUENT == 2
    # a Russian job recounts frequent words from the transcript; a word said once is not «частое»
    recount = {"model": {"lang": "ru"}, "transcript": "Собака бежит по дороге. Собака видит кошку.",
               "analyses": {"speech": {"words": 8}}}
    assert ru_texts.vocabulary_shown(recount) == [("собака", 2)]
    # the same threshold on a stored list (no transcript to recount from)
    stored = {"model": {"lang": "ru"}, "transcript": "",
              "analyses": {"speech": {"vocabulary": [["работа", 3], ["пауза", 1]]}}}
    assert ru_texts.vocabulary_shown(stored) == [("работа", 3)]
