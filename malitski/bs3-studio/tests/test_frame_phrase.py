"""The phrase under a key frame in a leaf, and texts that load no torch (refactoring plan of 3.1, stage 11).

bs3/frame_phrase.py holds the prompt of the phrase, its limits and word lists, its validation and the names of the saved
key frames, on the standard library only. The explanation code (backend_mm.describe_frame, mm/explain.key_frame_info)
asks for the phrase through it without importing frame_captions, and the captions of the page and the PDF read it back
from the same place. The prompt and the lists are pinned: an analysis asks exactly what it asked before the move.

The text modules load no torch: translate imports it only when Marian really translates, and Marian still translates
without gradients; ru_texts counts the frequent words of the speech with words.vocabulary, not through analyses/.
"""
from __future__ import annotations

import ast
import hashlib
import json
import logging
import subprocess
import sys
import types
from pathlib import Path

from bs3 import frame_captions, frame_phrase, settings, translate, words
from bs3.analyses import speech_stats

ROOT = Path(__file__).resolve().parents[1]
# sha256 of the prompt and of json.dumps(list or dict, ensure_ascii=False) of each word list, recorded on the tree
# before the move (frame_captions of commit bb5cbf9)
PINNED = {
    "FRAME_PROMPT": "dd000e09621c98b8998dd76d5fef703f57900faaeb95fcc84828cdbc4e7ac610",
    "PHRASE_VERDICTS": "346b9212ba307cb4553b86c11f86ed651a08482b70b690ea2825806cf767d7d5",
    "PHRASE_BANNED": "2366dedbe0766d5a7dff012b4e4a8a76f8af875aa370e3677886e2d2ea83b9ef",
    "PHRASE_YO": "1ac85ab7e7f126e92d8c822ff09c8b617e02897c8cdff2df64b0bb810d5047ae",
}
MOVED = ("FRAME_PROMPT", "PHRASE_NUM_PREDICT", "PHRASE_VERDICTS", "PHRASE_BANNED", "PHRASE_WORDS", "PHRASE_YO", "_yo",
         "_RE_YA", "_RE_DIGIT", "_RE_WORD")


def _sha(obj) -> str:
    s = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _imports(rel: str) -> set:
    """Every module a file of bs3-studio imports, at module level and inside functions, relative imports resolved
    («from .. import settings» in bs3/mm/explain.py is bs3.settings)."""
    path = ROOT / rel
    pkg = list(path.relative_to(ROOT).with_suffix("").parts[:-1])
    out = set()
    for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(n, ast.Import):
            out |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module != "__future__":
            base = ".".join((pkg[:len(pkg) - n.level + 1] if n.level else []) + ([n.module] if n.module else []))
            out |= {base} if n.module else {f"{base}.{a.name}" for a in n.names}
    return out


# ---------------------------------------------------------------- the phrase
def test_the_prompt_and_the_word_lists_are_pinned():
    got = {"FRAME_PROMPT": _sha(frame_phrase.FRAME_PROMPT), "PHRASE_VERDICTS": _sha(list(frame_phrase.PHRASE_VERDICTS)),
           "PHRASE_BANNED": _sha(list(frame_phrase.PHRASE_BANNED)), "PHRASE_YO": _sha(frame_phrase.PHRASE_YO)}
    assert got == PINNED, {k: v for k, v in got.items() if v != PINNED[k]}
    assert (frame_phrase.PHRASE_MAX, frame_phrase.PHRASE_NUM_PREDICT, frame_phrase.PHRASE_TRIES,
            frame_phrase.PHRASE_WORDS) == (40, 48, 2, (3, 6))


def test_frame_captions_reads_the_phrase_from_frame_phrase():
    for name in ("clean_phrase", "PHRASE_MAX", "PHRASE_TRIES", "_frame_index", "_decoded_index"):
        assert getattr(frame_captions, name) is getattr(frame_phrase, name), name
    assert not [name for name in MOVED if hasattr(frame_captions, name)]


def test_frame_phrase_is_a_leaf_and_the_explanation_code_asks_it():
    assert _imports("bs3/frame_phrase.py") <= {"re", "pathlib", "typing"}, _imports("bs3/frame_phrase.py")
    for rel in ("bs3/backend_mm.py", "bs3/mm/explain.py", "bs3/frame_captions.py"):
        got = _imports(rel)
        assert "bs3.frame_phrase" in got and "bs3.frame_captions" not in got, (rel, sorted(got))


def test_names_of_the_saved_key_frames():
    cases = {"key_18_frame420.jpg": (18, 420), "/jobs/x/explain/key_04_frame120.jpg": (4, 120),
             "x/y/z_12_frame34.webp": (12, 34), "key_1_2_frame3.jpg": (2, 3), "key_1_frame2_frame3.jpg": (None, 3),
             "foo.jpg": (None, None), "k_1_frame.jpg": (None, None)}
    got = {p: (frame_phrase._frame_index(p), frame_phrase._decoded_index(p)) for p in cases}
    assert got == cases


def test_the_phrase_budget_is_the_setting():
    """key_frame_info stops asking for phrases once settings.PHRASE_BUDGET seconds have passed, read when it runs:
    with a clock that moves 12 s per reading the 30 s budget leaves a phrase to the first two of four frames, a larger
    budget to all four."""
    from bs3.mm import explain
    paths = [f"/j/key_0{i}_frame{i}0.jpg" for i in range(1, 5)]
    raw = {Path(p).name: f"image {i}" for i, p in enumerate(paths)}

    warnings = []

    class Keep(logging.Handler):
        def emit(self, record):
            warnings.append(record.getMessage())

    def run(budget: float) -> list:
        clock = [0.0]

        def tick() -> float:
            clock[0] += 12.0
            return clock[0]
        saved = explain.time, settings.PHRASE_BUDGET
        explain.time, settings.PHRASE_BUDGET = types.SimpleNamespace(time=tick), budget
        keep = Keep(logging.WARNING)
        explain.log.addHandler(keep)
        try:
            info = explain.key_frame_info(paths, None, raw, phrase_fn=lambda _b64: "смотрит в камеру прямо")
        finally:
            explain.time, settings.PHRASE_BUDGET = saved
            explain.log.removeHandler(keep)
        return [r.get("phrase") == "смотрит в камеру прямо" for r in info]

    assert settings.PHRASE_BUDGET == 30
    assert run(settings.PHRASE_BUDGET) == [True, True, False, False]
    assert warnings == ["key-frame phrases: out of the 30s budget, the remaining frames keep none"]
    assert run(1000) == [True] * 4 and len(warnings) == 1


# ---------------------------------------------------------------- texts without torch
def test_texts_load_no_torch():
    """Importing the text modules, and using them the way the page and the PDF do, loads neither torch, OpenCV and
    transformers nor any analysis module."""
    code = ("import sys\n"
            "import bs3.translate, bs3.frame_captions, bs3.ru_texts\n"
            "from bs3 import frame_captions, ru_texts, translate\n"
            "rep = {'model': {'lang': 'ru'}, 'transcript': 'Работа, работа и ещё раз работа.',"
            " 'analyses': {'speech': {'words': 6}}}\n"
            "assert ru_texts.vocabulary_shown(rep) == [('работа', 3)], ru_texts.vocabulary_shown(rep)\n"
            "assert frame_captions.clean_phrase('смотрит вперед, голова прямо') == 'смотрит вперёд, голова прямо'\n"
            "entry = frame_captions.build({'media': {'fps': 30.0}}, ['/j/key_04_frame120.jpg'], None)[0]\n"
            "assert entry['label'] == '0:04', entry\n"
            "assert translate.fix_marian_ru('помолвленный') == 'вовлечённый'\n"
            "bad = ('torch', 'cv2', 'transformers', 'bs3.backend_mm', 'bs3.mm', 'bs3.analyses')\n"
            "print(sorted(m for m in bad if m in sys.modules))\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.split() == ["[]"], r.stdout
    assert not hasattr(translate, "torch")                  # imported inside the functions that run Marian


def test_marian_translates_without_gradients():
    """translate_sentences imports torch itself and runs the model under torch.no_grad(), as the decorator did."""
    import torch
    seen = {}

    class Enc(dict):
        def to(self, device):
            seen["enc_device"] = device
            return self

    class Tok:
        def __call__(self, s, add_special_tokens=True, return_tensors=None, **_k):
            return {"input_ids": s.split()} if return_tensors is None else Enc(batch=list(s))

        def batch_decode(self, gen, skip_special_tokens=True):
            return [f"ru:{x}" for x in gen]

    class Mdl:
        device = "cpu"

        def generate(self, batch, max_new_tokens, num_beams):
            seen.setdefault("grad", []).append(torch.is_grad_enabled())
            return batch

    def fake_get(src, tgt, device):
        seen["get"] = (src, tgt, device, torch.is_grad_enabled())
        return Tok(), Mdl()

    saved = translate._get
    translate._get = fake_get
    try:
        assert translate.translate_sentences([]) == []
        assert "get" not in seen
        out = translate.translate_sentences(["Hello there.", "Bye."], device="cpu", batch=1)
    finally:
        translate._get = saved
    assert out == ["ru:Hello there.", "ru:Bye."]
    assert seen == {"get": ("en", "ru", "cpu", False), "enc_device": "cpu", "grad": [False, False]}
    assert torch.is_grad_enabled()
    assert translate._device("cpu") == "cpu" and translate._device(None) in ("cuda", "cpu")


def test_the_frequent_words_are_counted_in_words():
    """vocabulary and the word splitter moved from analyses/speech_stats to words: the speech numbers of the pipeline
    and the frequent words of the page count the same words, and ru_texts no longer imports analyses/."""
    assert speech_stats.vocabulary is words.vocabulary and speech_stats.spoken_words is words.spoken_words
    assert not [n for n in ("_words", "_WORD") if hasattr(speech_stats, n)]
    assert not [m for m in _imports("bs3/ru_texts.py") if m.startswith("bs3.analyses")]
    # the values of speech_stats before the move (commit bb5cbf9); a word of three letters («код») is never frequent
    text = ("Работа, работа и ещё раз работа. Компания большая — компания хорошая, код и код. Инструменты! Don't "
            "stop-it, ёлка 5 раз")
    assert words.spoken_words(text) == ["работа", "работа", "и", "ещё", "раз", "работа", "компания", "большая",
                                        "компания", "хорошая", "код", "и", "код", "инструменты", "don't", "stop-it",
                                        "ёлка", "раз"]
    assert words.vocabulary(text) == [("работа", 3), ("компания", 2), ("большая", 1), ("хорошая", 1),
                                      ("инструменты", 1), ("don't", 1), ("stop-it", 1), ("ёлка", 1)]
    assert words.vocabulary(text, top=2) == [("работа", 3), ("компания", 2)]
    chunks = [(0.0, 4.0, "Ну вот, работа — это важно."), (4.6, 9.0, "Компания большая, типа, и как бы хорошая!"),
              (12.0, 13.5, "Да.")]
    assert speech_stats.stats_for(chunks, 3.0, 10.0, "ru") == {
        "words": 12, "speech_sec": 5.4, "words_per_min_speech": 91.7, "words_per_min_wall": 70.7, "pause_share": 0.229,
        "long_pauses": 0, "fillers": 4, "fillers_per_100": 33.3, "mean_sentence": 6.0, "ttr": 1.0, "unique_words": 12}
