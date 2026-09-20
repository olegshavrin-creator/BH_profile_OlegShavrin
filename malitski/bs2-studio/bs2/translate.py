"""Translation helpers for showing results in the interface language: Marian (Helsinki-NLP opus-mt) on the GPU for
sentences, the local Ollama model for whole texts and dictionary entries (Marian is their fallback). The models compute
in English (text branch, behaviour descriptions from the VLM, word attributions); the web page and the PDF show
Russian translations next to / instead of the English originals, which stay in result.json."""
from __future__ import annotations

import logging
import re
from typing import Dict, List

import torch

log = logging.getLogger("bs.translate")
_models: dict = {}
_SENT = re.compile(r"(?<=[.!?])\s+")
_PREFIX = re.compile(r"^(\[[^\]]*\]\s*)")          # "[0–20 с] " prefixes of segment descriptions


def _get(src: str, tgt: str, device: str):
    key = (src, tgt)
    if key not in _models:
        from transformers import MarianMTModel, MarianTokenizer
        name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
        tok = MarianTokenizer.from_pretrained(name)
        mdl = MarianMTModel.from_pretrained(name).to(device).eval()
        _models[key] = (tok, mdl)
        log.info("loaded %s on %s", name, device)
    return _models[key]


def _device(device: str | None) -> str:
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


MAX_PIECE_TOKENS = 200     # Marian has 512 positions for the input and the output: long pieces are split before that
_CLAUSE = re.compile(r"(?<=[,;:])\s+|\s+(?=[—–-]\s)")


def _pieces(sentence: str, tok, limit: int = MAX_PIECE_TOKENS) -> List[str]:
    """A sentence longer than `limit` tokens (Whisper often returns a whole minute without a full stop) as several
    pieces: clauses packed up to the limit, a clause that is still too long cut by words."""
    def n_tok(s):
        return len(tok(s, add_special_tokens=False)["input_ids"])
    if n_tok(sentence) <= limit:
        return [sentence]
    out, cur = [], ""
    for clause in (c for c in _CLAUSE.split(sentence) if c.strip()):
        if n_tok(clause) > limit:                 # no commas either: fixed windows of words
            words = clause.split()
            step = max(1, len(words) * limit // max(1, n_tok(clause)))
            parts = [" ".join(words[i:i + step]) for i in range(0, len(words), step)]
        else:
            parts = [clause]
        for p in parts:
            if cur and n_tok(cur + " " + p) > limit:
                out.append(cur); cur = p
            else:
                cur = f"{cur} {p}".strip()
    return out + ([cur] if cur else [])


@torch.no_grad()
def translate_sentences(sentences: List[str], src="en", tgt="ru", device=None, batch: int = 16) -> List[str]:
    """One translation per input sentence; over-long sentences are translated in pieces and joined back."""
    if not sentences:
        return []
    tok, mdl = _get(src, tgt, _device(device))
    flat, owner = [], []
    for i, s in enumerate(sentences):
        for p in _pieces(s, tok):
            flat.append(p); owner.append(i)
    tr = []
    for i in range(0, len(flat), batch):
        enc = tok(flat[i:i + batch], return_tensors="pt", padding=True, truncation=True, max_length=400).to(mdl.device)
        gen = mdl.generate(**enc, max_new_tokens=400, num_beams=2)
        tr += tok.batch_decode(gen, skip_special_tokens=True)
    out = [""] * len(sentences)
    for i, t in zip(owner, tr):
        out[i] = f"{out[i]} {t}".strip()
    return out


_CYR = re.compile(r"[А-Яа-яЁё]")
_LAT = re.compile(r"[A-Za-z]")
_REPEAT = re.compile(r"([^\w\s]{1,3})(?:\s*\1){2,}")        # «♪ ♪ ♪ ♪»: a Marian loop on input it cannot read


def _script_runs(sentence: str, keep_min: int = 3) -> List[list]:
    """[[text, keep]]: runs of at least `keep_min` Russian words are kept as spoken (Whisper forced to English still
    writes whole Russian phrases, which Marian en-ru turns into garbage), everything else is translated."""
    runs: List[list] = []                          # [script, tokens, Russian words]
    for t in sentence.split():
        script = "cyr" if (_CYR.search(t) and not _LAT.search(t)) else ("lat" if _LAT.search(t) else None)
        if runs and script in (None, runs[-1][0]):
            runs[-1][1].append(t); runs[-1][2] += script == "cyr"
        else:
            runs.append([script or "lat", [t], int(script == "cyr")])
    out: List[list] = []
    for script, toks, n_ru in runs:
        keep = script == "cyr" and n_ru >= keep_min
        if out and out[-1][1] == keep:
            out[-1][0] += " " + " ".join(toks)
        else:
            out.append([" ".join(toks), keep])
    return out


def translate_long(text: str, src="en", tgt="ru", device=None) -> str:
    """A whole transcript: split into sentences (and over-long sentences into pieces), translated in batches. Into
    Russian, phrases already spoken in Russian stay as they are."""
    sents = [s for s in _SENT.split((text or "").strip()) if s.strip()]
    plan = [_script_runs(s) if tgt == "ru" else [[s, False]] for s in sents]
    tr = iter(translate_sentences([t for runs in plan for t, keep in runs if not keep], src, tgt, device))
    parts: List[str] = []
    for runs in plan:
        for t, keep in runs:
            t = t if keep else next(tr)
            # a phrase kept as spoken that opens a sentence starts with a capital letter, as the translated ones do
            if keep and (not parts or parts[-1].rstrip()[-1:] in ".!?…"):
                t = t[:1].upper() + t[1:]
            parts.append(t)
    return _REPEAT.sub(r"\1", " ".join(parts)).strip()


def translate_text(text: str, src="en", tgt="ru", device=None) -> str:
    """Line by line (keeps "[0–20 с]" prefixes and paragraph structure), sentence-batched."""
    lines = text.split("\n")
    pieces, plan = [], []              # plan: (prefix, start, count) per line
    for line in lines:
        m = _PREFIX.match(line)
        prefix = m.group(1) if m else ""
        body = line[len(prefix):].strip()
        sents = [s for s in _SENT.split(body) if s] if body else []
        plan.append((prefix, len(pieces), len(sents)))
        pieces += sents
    tr = translate_sentences(pieces, src, tgt, device)
    return "\n".join(prefix + " ".join(tr[a:a + n]) for prefix, a, n in plan)


# ---------------------------------------------------------------- whole texts through the local LLM
# Marian garbles the stock phrases of the behaviour descriptions («спокойной и спокойной» for 'calm and composed',
# «помолвленной» for 'engaged', «извращению» for 'extraversion', «разбитые губы» for 'parted lips') and loses the thread
# of a spoken monologue (the speaker changes gender from one sentence to the next). Whole texts go to the local Ollama
# model with a glossary; Marian is the fallback when Ollama is unreachable, its known mistakes repaired by
# fix_marian_ru, and the caller stores which translator was used so that a Marian text is translated again later.
OLLAMA_MODEL = "qwen2.5vl:7b"         # the model that writes the behaviour descriptions (already loaded by the pipeline)
GLOSSARY_RU = """calm and composed -> спокойный и собранный; composed -> собранный; composure -> самообладание;
engaged -> вовлечённый; engagement -> вовлечённость; attentive -> внимательный; attentiveness -> внимательность;
demeanor -> поведение; laid-back -> непринуждённый; at ease, comfortable -> непринуждённо, непринуждённый;
comfort -> непринуждённость; approachable -> располагающий к общению; approachability -> открытость к общению;
the individual -> человек; facial expressions -> мимика, выражение лица; body language -> язык тела;
parted lips -> приоткрытые губы; upright posture -> прямая осанка; eye contact -> зрительный контакт;
hand gestures -> жесты рук; occasional, occasionally -> время от времени; neutral to slightly positive -> от
нейтрального до слегка положительного; ambivalence -> двойственность; hesitation, hesitancy -> нерешительность;
thoughtful -> вдумчивый; reflective -> задумчивый; openness -> открытость опыту; conscientious -> добросовестный;
conscientiousness -> добросовестность; extraversion -> экстраверсия; extraverted -> экстравертированный;
agreeable -> доброжелательный; agreeableness -> доброжелательность; neuroticism -> нейротизм;
emotional stability -> эмоциональная стабильность; frames -> кадры"""
_OTHER_SCRIPT = re.compile("[%s]" % "".join(f"{chr(a)}-{chr(b)}" for a, b in (      # Hebrew, Arabic, Thai, CJK, Hangul
    (0x0590, 0x08FF), (0x0E00, 0x0EFF), (0x3040, 0x30FF), (0x3400, 0x9FFF), (0xAC00, 0xD7AF))))
_OLLAMA_STATE = {"t": 0.0, "ok": False, "url": ""}


def ollama_available(ttl: float = 60.0) -> bool:
    """The Ollama server answers. Checked at most once a minute, so a page render does not wait for a dead server
    again and again."""
    import time
    import urllib.request
    now = time.time()
    if now - _OLLAMA_STATE["t"] < ttl:
        return _OLLAMA_STATE["ok"]
    ok = False
    try:
        from .backend_mm import default_ollama_url
        url = default_ollama_url()
        with urllib.request.urlopen(f"{url}/api/tags", timeout=3) as r:
            ok = r.status == 200
        _OLLAMA_STATE["url"] = url
    except Exception:  # noqa: BLE001
        ok = False
    _OLLAMA_STATE.update(t=now, ok=ok)
    return ok


def _ollama_json(prompt: str, num_predict: int, model: str = OLLAMA_MODEL, timeout: int = 300,
                 temperature: float = 0.0, seed: int = 0) -> dict:
    import json
    import urllib.error
    import urllib.request

    from .backend_mm import default_ollama_url
    url = _OLLAMA_STATE["url"] or default_ollama_url()
    # no num_ctx here: a request with its own context size makes Ollama reload the model, and the behaviour-description
    # requests of the pipeline (frames, default context) would reload it back
    payload = {"model": model, "prompt": prompt, "stream": False, "format": "json", "think": False, "keep_alive": "30m",
               "options": {"temperature": temperature, "seed": seed, "num_predict": num_predict}}
    req = urllib.request.Request(f"{url}/api/generate", data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
        _OLLAMA_STATE.update(ok=False)            # the next ollama_available() in the next minute says no at once
        raise
    return json.loads(data.get("response") or "{}")


def _llm_answer_ok(en: str, ru, kind: str) -> bool:
    """A usable translation: Russian text of a plausible length, no other scripts; English words only where the source
    has them and they look like names (a capital letter: 'YouTube', 'Draw My Life'), none at all in a behaviour
    description. An untranslated 'two days' is rejected."""
    if not isinstance(ru, str) or not _CYR.search(ru) or _OTHER_SCRIPT.search(ru):
        return False
    ratio = len(ru.strip()) / max(1, len(en.strip()))
    if not 0.45 <= ratio <= 2.6:
        return False
    latin = re.findall(r"[A-Za-z][A-Za-z'\-]*", ru)
    if kind == "behavior":
        return not latin
    return all(w[:1].isupper() and re.search(rf"(?<![A-Za-z]){re.escape(w)}(?![A-Za-z])", en) for w in latin)


def llm_translate(texts: List[str], kind: str = "behavior", note: str = "") -> List[str | None]:
    """Russian translations of English texts from the local Ollama model; None where it failed or the answer does not
    look like a translation (the caller falls back to Marian). kind: "behavior" (a description of nonverbal behaviour)
    or "speech" (a piece of a spoken transcript, possibly with phrases already in Russian). note: one more instruction
    (the speaker's gender)."""
    import json
    out: List[str | None] = [None] * len(texts)
    todo = [i for i, t in enumerate(texts) if t and t.strip()]
    if not todo or not ollama_available():
        return out
    what = ("descriptions of a person's nonverbal behaviour and personality impression" if kind == "behavior" else
            "pieces of an automatic speech transcript of one speaker (recognition errors are possible; words or "
            "phrases already written in Russian stay as they are)")
    rules = (f"Use these terms:\n{GLOSSARY_RU}\nAdjectives agree with their noun in gender and case. Write only Russian "
             "words in Cyrillic, no English words.") if kind == "behavior" else (
             "Keep names of people, places, brands and titles in their usual Russian spelling (or in Latin letters "
             "exactly as in the source when there is no Russian spelling). Keep the speaker's grammatical gender the "
             "same in every piece.")

    src = {i: _prepare_behavior_en(texts[i]) if kind == "behavior" else texts[i].strip() for i in todo}

    def ask(i: int) -> None:
        # one text per request: with several texts in one answer the model mixed them up (the gender of one segment
        # leaked into the next). A repeated request samples a little (temperature, seed): at 0 it would return the
        # same broken answer again
        prompt = (f"You are a professional English-Russian translator. Translate the text below ({what}) into natural, "
                  f"grammatical Russian. Keep the full meaning: do not add, drop or summarise anything. {rules} {note}\n"
                  "Answer with a JSON object {\"ru\": \"<the Russian translation>\"}.\n"
                  f"Text: {json.dumps(src[i], ensure_ascii=False)}")
        hint = ""
        for attempt in range(3):
            if not ollama_available():
                return
            try:
                obj = _ollama_json(prompt + hint, num_predict=min(8000, 600 + 2 * len(src[i])),
                                   temperature=0.4 if attempt else 0.0, seed=attempt)
            except Exception as e:  # noqa: BLE001
                log.warning("Ollama translation failed (%s)", str(e).splitlines()[0][:80])
                continue
            ru = obj.get("ru") if isinstance(obj, dict) else None
            if kind == "speech" and isinstance(ru, str) and not _llm_answer_ok(src[i], ru, kind):
                ru = _patch_latin_runs(src[i], ru) or ru
            if _llm_answer_ok(src[i], ru, kind):
                out[i] = _fix_llm_ru(" ".join(ru.split()))
                return
            left = sorted({w for w in re.findall(r"[A-Za-z][A-Za-z'\-]*", ru) if kind == "behavior" or not (
                w[:1].isupper() and re.search(rf"(?<![A-Za-z]){re.escape(w)}(?![A-Za-z])", src[i]))}) if isinstance(ru, str) else []
            # the next try names the English words the answer kept («первые два two days»)
            hint = (f"\nA previous translation left these English words untranslated: {', '.join(left[:12])}. "
                    "Translate them into Russian too.") if left else ""

    # a few requests at a time (Ollama serves parallel requests of one loaded model when OLLAMA_NUM_PARALLEL allows)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=LLM_PARALLEL) as ex:
        list(ex.map(ask, todo))
    return out


LLM_PARALLEL = 2          # more requests only queue up in Ollama and hold back other users of the model


def _patch_latin_runs(en: str, ru: str) -> str | None:
    """A transcript answer that is fine except for a few short English fragments the model copied along with the
    Russian words around them («наша идея первые два two days»): the fragments translated by Marian. None when there
    are more or longer fragments (the answer is then asked for again)."""
    runs = list(re.finditer(r"[A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*)*", ru))
    bad = [m for m in runs if not all(w[:1].isupper() and re.search(rf"(?<![A-Za-z]){re.escape(w)}(?![A-Za-z])", en)
                                      for w in m.group(0).split())]
    if not bad or len(bad) > 4 or any(len(m.group(0).split()) > 3 for m in bad):
        return None
    tr = translate_sentences([m.group(0) for m in bad], "en", "ru")
    out = ru
    for m, t in sorted(zip(bad, tr), key=lambda mt: -mt[0].start()):
        t = t.strip().strip(".").strip()
        out = out[:m.start()] + (t[:1].lower() + t[1:] if m.group(0)[:1].islower() else t) + out[m.end():]
    return out if _llm_answer_ok(en, out, "speech") else None


_NEUTER = {"ый": "ое", "ой": "ое", "ий": "ее"}


def _fix_llm_ru(text: str) -> str:
    """Small morphology slips of the 7B model: «собранныя» -> «собранная» (-ыя is not a modern Russian ending), and
    masculine adjectives in front of «поведение» («спокойный и положительный поведение» -> «спокойное и положительное
    поведение»)."""
    text = re.sub(r"(?<=[а-яё])ныя\b", "ная", text)
    adj = r"([а-яё]{3,})(ый|ой|ий)"
    text = re.sub(adj + r"(\s+и\s+)" + adj + r"(\s+поведение)\b",
                  lambda m: m.group(1) + _NEUTER[m.group(2)] + m.group(3) + m.group(4) + _NEUTER[m.group(5)] + m.group(6), text)
    return re.sub(adj + r"(\s+поведение)\b", lambda m: m.group(1) + _NEUTER[m.group(2)] + m.group(3), text)


# words the model tends to copy untranslated into the Russian text («occasionalными»), and English pronoun forms that
# make it write «они» / «их» about one person of unknown gender
_BEHAVIOR_EN_SUBS = [(r"\boccasionally\b", "from time to time"), (r"\boccasional\b", "periodic"),
                     (r"\bbewilderment\b", "confusion"), (r"\bthe individual\b", "the person"),
                     (r"\bThe individual\b", "The person"), (r"\ban individual\b", "a person")]
_THEY_SUBS = [(r"\bthey are\b", "he is"), (r"\bthey were\b", "he was"), (r"\bthey have\b", "he has"),
              (r"\bthey're\b", "he's"), (r"\bthemselves\b", "himself"), (r"\btheir\b", "his"), (r"\bthem\b", "him"),
              (r"\bthey\b", "he"), (r"\bThey are\b", "He is"), (r"\bThey\b", "He"), (r"\bTheir\b", "His")]


def _prepare_behavior_en(text: str) -> str:
    """The English description as sent to the LLM: a few words it leaves untranslated replaced by plain synonyms, and
    'they' of a single person of unknown gender made 'he' (Russian «человек» is masculine)."""
    t = (text or "").strip()
    for pat, rep in _BEHAVIOR_EN_SUBS:
        t = re.sub(pat, rep, t)
    if not re.search(r"\b(?:she|her|hers|herself|woman|girl|he|his|him|himself|man|boy)\b", t, re.I):
        for pat, rep in _THEY_SUBS:
            t = re.sub(pat, rep, t)
    return t


# Marian's recurring mistakes in behaviour descriptions (see above): stem replacements keep the grammatical ending
_MARIAN_FIXES = [
    (r"\b(спокойн)(ой|ый|ым|ая|ое|ого|ую|ыми|ые|ых)( и )спокойн(?:ой|ый|ым|ая|ое|ого|ую|ыми|ые|ых)\b", r"\1\2\3собранн\2"),
    (r"\b(?:состоятельн|скомпрометированн|созидательн|составн)(ой|ый|ым|ая|ое|ого|ую|ыми|ые|ых|ом)\b", r"собранн\1"),
    (r"\bпомолвленн(ой|ый|ым|ая|ое|ого|ую|ыми|ые|ых|ом)\b", r"вовлечённ\1"),
    (r"\bизвращени(е|я|ю|ем|и)\b", lambda m: "экстраверси" + {"е": "я", "я": "и", "ю": "и", "ем": "ей", "и": "и"}[m.group(1)]),
    (r"\bразби(т|тые|тыми|тых|ты)(\s+губ)", r"приоткры\1\2"),
    (r"\bразорван(ные|ными|ных)(\s+губ)", r"приоткрыт\1\2"),
    (r"\b(губы\s+(?:\w+\s+)?)разорваны\b", r"\1приоткрыты"),
    (r"\bнепристойн(ое|ого|ым|ом)(\s+поведени)", r"непринуждённ\1\2"),
    (r"\bприемлем(ую|ой|ая|ый|ое|ым)(\s+личност)", r"доброжелательн\1\2"),
    (r"\bутешени(е|я|ю|ем)\b", lambda m: "непринуждённост" + {"е": "ь", "я": "и", "ю": "и", "ем": "ью"}[m.group(1)]),
]


def fix_marian_ru(text: str) -> str:
    """Marian's known mistakes in behaviour descriptions replaced by the glossary terms (fallback path only)."""
    for pat, rep in _MARIAN_FIXES:
        text = re.sub(pat, rep, text)
    return text


def translate_description(text: str) -> tuple:
    """A behaviour description ("[0–20 с] …" line per segment) in Russian: (text, "ollama" | "marian")."""
    lines = (text or "").split("\n")
    heads, bodies = [], []
    for line in lines:
        m = _PREFIX.match(line)
        heads.append(m.group(1) if m else "")
        bodies.append(line[len(heads[-1]):].strip())
    ru = llm_translate(bodies, "behavior")
    fallback = False
    for i, body in enumerate(bodies):
        if body and ru[i] is None:
            ru[i] = fix_marian_ru(translate_text(body, "en", "ru"))
            fallback = True
    return "\n".join(h + (r or "") for h, r in zip(heads, ru)), _translator(fallback)


def _translator(fallback: bool) -> str:
    """"ollama"; "ollama+marian" when Ollama answered but a piece was rejected (asking again later would not help);
    "marian" when Ollama was unreachable (the text is translated again once it answers)."""
    return "ollama" if not fallback else ("ollama+marian" if ollama_available() else "marian")


def translate_transcript(text: str, note: str = "") -> tuple:
    """A whole English transcript in Russian: (text, "ollama" | "marian"). Pieces of a few sentences go to the LLM,
    so every piece keeps its context; a piece the LLM could not translate goes through Marian."""
    sents = [s for s in _SENT.split((text or "").strip()) if s.strip()]
    chunks: List[str] = []
    for s in sents:
        if chunks and len(chunks[-1]) + len(s) < 700:
            chunks[-1] += " " + s
        else:
            chunks.append(s)
    ru = llm_translate(chunks, "speech", note)
    fallback = False
    for i, c in enumerate(chunks):
        if ru[i] is None:
            ru[i] = translate_long(c, "en", "ru")
            fallback = True
    out = _REPEAT.sub(r"\1", " ".join(r for r in ru if r)).strip()
    # a recording often starts mid-sentence ('a point in my video …'); the text shown still starts with a capital
    return out[:1].upper() + out[1:], _translator(fallback)


def _context_sentence(word: str, text: str) -> str:
    """First sentence of `text` containing the word (for sense disambiguation), or ''."""
    if not text:
        return ""
    core = re.escape(word.strip(" .,;:!?\"'()[]{}«»—-"))
    if not core:
        return ""
    for sent in _SENT.split(text):
        if re.search(rf"(?i)(?<![A-Za-z]){core}(?![A-Za-z])", sent):
            return sent.strip()[:200]
    return ""


def _ollama_dictionary(words: List[str], tgt: str, model: str = OLLAMA_MODEL, context: str | None = None,
                       names: set | None = None) -> Dict[str, str]:
    """Dictionary-form translations from the local Ollama model (a sentence MT model turns 'calm' into
    'успокойся'; an LLM asked for lemmas gives 'спокойный'). A context sentence per word disambiguates the sense.
    `names` (optional) receives the words the model marks as proper names in their context."""
    import json

    lang_name = {"ru": "Russian"}.get(tgt, tgt)
    items = [{"word": w, "context": _context_sentence(w, context or "")} for w in words]
    examples = {"ru": '{"hesitation": {"tr": "нерешительность", "name": false}, "calm": {"tr": "спокойный", "name": false}, '
                      '"gestures": {"tr": "жесты", "name": false}, "imagine": {"tr": "представлять", "name": false}, '
                      '"work": {"tr": "работа", "name": false}, "oxford": {"tr": "Оксфорд", "name": true}}'}.get(tgt, "{}")
    prompt = (f"You are a bilingual English-{lang_name} dictionary. For each English word below give its {lang_name} "
              f"dictionary translation in base form (nouns: nominative singular; verbs: infinitive; adjectives: "
              f"masculine singular), choosing the sense used in the context sentence when one is given. Translate "
              f"the meaning of the word itself, never a different word from the sentence. When the word is part of a "
              f"set expression in the context, give the meaning it has there (in 'tipping point' the word 'point' is "
              f"«момент»). One to three words, no explanations. \"name\" is true only for a proper name of a person, "
              f"place, company or brand; a common word, a weekday, a month or a word of a film, book or video title is "
              f"not a name. Example of the expected output: {examples}\nAnswer with a JSON object mapping each "
              f"English word exactly as written to its entry.\nItems: {json.dumps(items, ensure_ascii=False)}")
    obj = _ollama_json(prompt, num_predict=2500, model=model, timeout=180)
    out = {}
    for w in words:
        t = obj.get(w) or obj.get(w.lower()) or obj.get(w.capitalize())
        is_name = False
        if isinstance(t, dict):
            is_name = t.get("name") is True
            t = t.get("tr") or t.get("translation")
        if isinstance(t, str) and valid_translation(t, tgt):
            out[w] = t.strip().strip(".")
            if is_name and names is not None:
                names.add(w)
    return out


def valid_translation(t, tgt: str) -> bool:
    """A usable display translation: short, and for Russian written in Cyrillic without stray Latin letters
    (small models sometimes answer 'Ero' for 'His')."""
    if not isinstance(t, str) or not (0 < len(t.strip()) <= 40):
        return False
    if tgt == "ru":
        return re.search(r"[А-Яа-яЁё]", t) is not None and re.search(r"[A-Za-z]", t) is None
    return True


def translate_words(words: List[str], src="en", tgt="ru", device=None, context: str | None = None,
                    info: dict | None = None) -> Dict[str, str]:
    """Single words/short phrases -> {word: translation}. Ollama dictionary first (base forms, sense taken from
    `context` = the text the words come from), Marian for the rest. `info` (optional) receives "by": "ollama" when the
    dictionary answered, "marian" when every word went through Marian (worth translating again later), and "names":
    the words the dictionary marked as proper names."""
    uniq = sorted({w.strip() for w in words if w and w.strip()})
    info = info if info is not None else {}
    info.update(by="marian", names=set())
    if not uniq:
        return {}
    out: Dict[str, str] = {}
    try:
        if not ollama_available():
            raise ConnectionError("Ollama is not reachable")
        out = _ollama_dictionary(uniq, tgt, context=context, names=info["names"])
        info["by"] = "ollama"
    except Exception as e:  # noqa: BLE001
        log.warning("Ollama dictionary translation failed (%s); using Marian", str(e).splitlines()[0][:80])
    missing = [w for w in uniq if w not in out]
    if missing:
        tr = translate_sentences(missing, src, tgt, device)
        for w, t in zip(missing, tr):
            t = t.strip().strip(".").strip()
            if valid_translation(t, tgt) and len(t) <= 4 * max(3, len(w)):   # a runaway translation is worse than none
                out[w] = t
    # a lower-case word stays lower-case ('pause' -> «пауза», not «Пауза»); names and abbreviations keep their capitals
    for w, t in out.items():
        if w[:1].islower() and t[:1].isupper() and not t[1:2].isupper():
            out[w] = t[:1].lower() + t[1:]
    return out
