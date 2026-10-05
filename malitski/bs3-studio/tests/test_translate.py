"""Pure helpers of bs3.translate (T8(2, 3)): no model is loaded and no request is made — these functions only reshape
strings. fix_marian_ru repairs the recurring Marian mistakes that scripts/check_behavior_translation.py hunts for;
_llm_answer_ok and valid_translation decide whether an answer is usable; _fix_llm_ru fixes small morphology slips of the
local model."""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

from bs3 import translate as tr

# the mistranslation patterns the regression script searches every behaviour description for; fix_marian_ru is what
# repairs the Marian-specific ones, so the two are checked together
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_behavior_translation.py"
_spec = importlib.util.spec_from_file_location("check_behavior_translation", _SCRIPT)
_cbt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cbt)
PATTERNS = set(_cbt.PATTERNS)

# (a Marian output carrying the mistake, the check_behavior_translation pattern it triggers, a fragment the fix inserts)
FIX_CASES = [
    ("человек держится спокойной и спокойной.", r"спокойн\w* и спокойн", "собранн"),
    ("в беседе поведение состоятельное.", r"состоятельн", "собранн"),
    ("в кадре поведение скомпрометированное.", r"скомпрометир", "собранн"),
    ("в кадре созидательное поведение.", r"созидательн", "собранн"),
    ("выглядит помолвленной в разговоре.", r"помолвлен", "вовлеч"),
    ("высокий балл по извращению.", r"извращени", "экстраверси"),
    ("у человека разбитые губы.", r"разбит\w* губ", "приоткры"),
    ("у человека разорванные губы.", r"разорван", "приоткры"),
    ("у человека губы слегка разорваны.", r"разорван", "приоткрыт"),
    ("в кадре непристойное поведение.", r"непристойн", "непринужд"),
    ("приемлемая личность в общении.", r"приемлем\w* личност", "доброжелательн"),
    ("ощущение утешения на лице.", r"утешени", "непринуждённост"),
]


def test_fix_marian_ru_repairs_the_known_marian_mistakes():
    for inp, pat, frag in FIX_CASES:
        assert pat in PATTERNS, pat                              # a pattern check_behavior_translation looks for
        assert re.search(pat, inp, re.I), (inp, pat)            # the input really carries the mistake
        out = tr.fix_marian_ru(inp)
        assert not re.search(pat, out, re.I), (pat, out)        # the fix removed it
        assert frag in out, (frag, out)                         # replaced by the glossary term
    # a clean Russian text is left untouched
    clean = "Человек говорит спокойно и собранно, поддерживает зрительный контакт."
    assert tr.fix_marian_ru(clean) == clean


def test_llm_answer_ok_rejects_latin_other_scripts_and_empty():
    en = "The person is calm and composed."
    assert tr._llm_answer_ok(en, "Человек спокоен и собран.") is True
    assert tr._llm_answer_ok(en, "Человек calm и собран.") is False        # an English word left in
    assert tr._llm_answer_ok(en, "Человек 平和 и собран.") is False          # another script (CJK)
    assert tr._llm_answer_ok(en, "Человек שלום и собран.") is False         # another script (Hebrew)
    for empty in ("", "   ", None, 5):
        assert tr._llm_answer_ok(en, empty) is False, empty
    # length far from the original: a stub answer or a runaway one
    assert tr._llm_answer_ok("A fairly long English sentence about the behaviour of the person.", "Да.") is False
    assert tr._llm_answer_ok("Calm.", "Очень очень очень длинный русский ответ, совсем не по делу и слишком многословный.") \
        is False


def test_valid_translation():
    assert tr.valid_translation("спокойный", "ru") is True
    assert tr.valid_translation("а" * 41, "ru") is False           # too long for a display word
    assert tr.valid_translation("", "ru") is False
    assert tr.valid_translation("   ", "ru") is False
    assert tr.valid_translation("Ero", "ru") is False              # Latin only ('His' left untranslated)
    assert tr.valid_translation("спокойcalm", "ru") is False       # mixed Latin
    assert tr.valid_translation(None, "ru") is False
    assert tr.valid_translation(5, "ru") is False
    # a non-Russian target only needs a short non-empty string, with no Cyrillic/Latin rule
    assert tr.valid_translation("hello", "de") is True
    assert tr.valid_translation("x" * 50, "de") is False


def test_fix_llm_ru():
    # «-ыя» is not a modern ending
    assert tr._fix_llm_ru("человек собранныя") == "человек собранная"
    # masculine adjectives before the neuter «поведение»
    assert tr._fix_llm_ru("спокойный и положительный поведение") == "спокойное и положительное поведение"
    assert tr._fix_llm_ru("уверенный поведение") == "уверенное поведение"
    # nothing to fix
    assert tr._fix_llm_ru("человек спокоен и собран") == "человек спокоен и собран"
