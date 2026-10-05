"""Every runtime setting of BS Profiler 3.1 in one place: the data folders, the models, the Ollama server, the web
server and the segments of long clips. Standard library only, and importing it changes nothing: the values are read
once, from the environment where a variable is named below, and no folder is created. `apply_process_env()` is the
one function that changes something (the environment of the web server process).

Environment variables: BS3_DATA_DIR (the data folder, default ~/bs3_data), BS3_JOURNAL (the journal file, default
<data folder>/logs/journal.txt), BS3_OLLAMA_URL (the Ollama server, e.g. http://127.0.0.1:11434; default: found
by bs3.ollama.url on localhost, then on the WSL default gateway), BS3_KEEP_FAILED_JOBS (keep the folder of a run
that failed, for debugging; normally a failed run leaves nothing behind) and BS3_KEEP_SEGMENTS (keep the segment
clips of a finished job; normally they are removed once result.json is written).

Domain constants stay next to their logic, not here: config/mbti.json, the score bands, the palette, the caveats, the
lexicon, the prompt limits of the frame phrase (PHRASE_* in frame_phrase), the frame counts of MMConfig, the
windowing of BackendConfig and the model ids pinned in analyses/ and mm/ (the models the features were trained with).
"""
from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str, default) -> Path:
    return Path(os.environ.get(name) or default).expanduser()


# ---------------------------------------------------------------- folders
DATA_DIR = _env_path("BS3_DATA_DIR", "~/bs3_data")
JOBS_DIR = DATA_DIR / "web_jobs"                  # one folder per analysis of the page (the default --work-dir)
LOGS_DIR = DATA_DIR / "logs"
JOURNAL_PATH = _env_path("BS3_JOURNAL", LOGS_DIR / "journal.txt")
GRADIO_TMP = DATA_DIR / "gradio_tmp"              # Gradio's own temp folder (the shared /tmp/gradio is cleaned by another
                                                  # service)

# ---------------------------------------------------------------- models
MM_CHECKPOINTS = os.path.expanduser("~/bs/mm_runs_seeds/seed*/best.pt")   # AMLAI 1.0: 5 seeds, averaged (path, list or
                                                                          # glob, as MMConfig.checkpoint takes it)
OCEANAI_MODELS_DIR = os.path.expanduser("~/bs/models")                    # the OCEAN-AI weights cache
ASR_MODEL = "openai/whisper-large-v3-turbo"                               # Whisper for the transcript (Hugging Face id)

# ---------------------------------------------------------------- Ollama (bs3.ollama)
OLLAMA_MODEL = "qwen2.5vl:7b"     # the vision model of the behaviour descriptions and frame phrases, and the translator
                                  # of the descriptions; same accuracy as qwen3-vl:30b on FIV2 (0.914 vs 0.914 mACC),
                                  # 3x lighter. `bs3 web --ollama-model` replaces it for the whole process
OLLAMA_URL = os.environ.get("BS3_OLLAMA_URL", "").strip().rstrip("/")    # "" -> found automatically
OLLAMA_PORT = 11434               # where the automatic search looks
OLLAMA_KEEP_ALIVE = "30m"         # how long Ollama keeps the model loaded after a request
OLLAMA_DESCRIBE_TIMEOUT = 900     # seconds per behaviour-description request (backend_mm) ...
OLLAMA_DESCRIBE_ATTEMPTS = 3      # ... and the attempts while Ollama (re)loads a model
OLLAMA_TRANSLATE_TIMEOUT = 300    # seconds per translation of a description (translate)
OLLAMA_DICTIONARY_TIMEOUT = 180   # seconds per word-dictionary request (translate)
OLLAMA_PROBE_TTL = 60.0           # the answer of «is Ollama there» is reused this many seconds
OLLAMA_PROBE_TIMEOUT = 3          # seconds a probe of /api/tags waits
PHRASE_TIMEOUT = 45               # seconds per frame-phrase request: a caption never holds the job for minutes
PHRASE_BUDGET = 30                # seconds for the phrases of all five key frames; what is left over keeps no phrase
LLM_PARALLEL = 2                  # translation requests at a time: more only queue up in Ollama and hold back other
                                  # users of the model

# ---------------------------------------------------------------- web server
PORT = 7880                       # `bs3 web`
PREVIEW_PORT = 7882               # scripts/ui_preview.py
HOST = "0.0.0.0"                  # reachable from Windows through localhost
QUEUE_CONCURRENCY = 1             # one analysis at a time: one GPU
NO_PROXY = "localhost,127.0.0.1,0.0.0.0"

# ---------------------------------------------------------------- segments of long clips (longvideo)
SEGMENT_SEC = 20.0                # the models were trained on 15-second clips; the texts say «по ~20 с»
SINGLE_CLIP_MAX_SEC = 30.0        # a clip up to this long is analysed whole, as one segment («Ролик короче 30 с»)
MIN_TAIL_SEC = 6.0                # a shorter last segment is merged into the one before it

# ---------------------------------------------------------------- job folders (pipeline.run_analysis)
KEEP_FAILED_JOBS = os.environ.get("BS3_KEEP_FAILED_JOBS") == "1"   # keep the folder of a run that failed (for
                                                                  # debugging); normally a failed run leaves nothing
KEEP_SEGMENTS = os.environ.get("BS3_KEEP_SEGMENTS") == "1"        # keep the segment clips of a finished job (for
                                                                 # debugging); normally they are removed once
                                                                 # result.json is written (scripts/clean_jobs.py cleans
                                                                 # up jobs that already exist)


def apply_process_env() -> None:
    """The environment of the web server process, without overriding what is already set: Gradio's temp folder and no
    Gradio analytics (Gradio reads both, so this runs before gradio is imported), and no proxy for the local servers."""
    os.environ.setdefault("GRADIO_TEMP_DIR", str(GRADIO_TMP))
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    os.environ.setdefault("no_proxy", NO_PROXY)
    os.environ.setdefault("NO_PROXY", NO_PROXY)


def shown(path) -> str:
    """A path as a help text names it: the home folder as «~»."""
    s, home = str(path), str(Path.home())
    return "~" + s[len(home):] if s == home or s.startswith(home + os.sep) else s
