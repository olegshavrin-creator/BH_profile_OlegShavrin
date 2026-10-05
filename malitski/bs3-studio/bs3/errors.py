"""Analysis failures in one place (refactoring plan of 3.1, stage 20): the exception types the pipeline and the web app
raise, whether a failure is fatal to the server, and one table that turns any exception into a calm Russian sentence for
the error dialog and the command line. No traceback, English text or server path ever reaches the user; the original
goes to the server log.

Leaf module: it imports only the standard library and never torch — a CUDA fault is recognised by its message text — so
building the page, which imports this module for AnalysisCancelled and user_message, never pulls in the GPU stack.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


class UserFacingError(RuntimeError):
    """Its message is already meant for the user and is shown as it is (it may contain Latin names such as «Ollama»)."""


class OllamaUnavailable(RuntimeError):
    """Ollama — the local model that describes behaviour for AMLAI 1.0 — did not answer."""


class NoSegmentsAnalysed(RuntimeError):
    """No segment of a long video could be scored. `errors` holds the per-segment error lines, so user_message can tell
    a run that failed on the video card or on Ollama from one that simply found no face or speech."""

    def __init__(self, message: str = "", errors=None):
        super().__init__(message)
        self.errors = list(errors or [])


class AnalysisCancelled(RuntimeError):
    """Raised between steps when the caller asked to stop (LongVideoAnalyzer.analyze / run_analysis `should_stop`)."""


# --- the calm Russian sentences; the baseline pins the ones that were shown before this stage, word for word ---
CANCELLED = "Обработка остановлена. Запустите анализ заново."
OOM = "Не хватило памяти видеокарты. Подождите минуту и запустите анализ заново."
OLLAMA = ("Не отвечает Ollama — локальная модель, которая описывает поведение человека для AMLAI 1.0. "
          "Запустите Ollama и повторите анализ или выберите модель OCEAN-AI.")
NO_SIGNAL = "В ролике не найдено ни лица, ни речи, поэтому оценить его нельзя. Проверьте файл."
ENSEMBLE = ("Модель не смогла обработать ролик: чаще всего в кадре не найдено лицо или не слышна речь. "
            "Проверьте файл.")
BAD_VIDEO = "Не удалось прочитать видеофайл: возможно, он повреждён или записан в неподдерживаемом формате."
CUDA = "Сбой видеокарты. Сервер нужно перезапустить; подробности записаны в журнал сервера."
DISK = "На сервере закончилось место на диске. Освободите место и запустите анализ заново."
NO_WRITE = "Папка для результатов на сервере недоступна для записи. Подробности записаны в журнал сервера."
GENERIC = "Не удалось обработать ролик из-за внутренней ошибки. Подробности записаны в журнал сервера."

_DISK_FREE_MIN = 500 * 1024 * 1024        # under 500 MB free counts a disk/ffmpeg failure as «no space left»


def _is_oom(text: str) -> bool:
    return "out of memory" in text.lower()


def _is_cuda_fault(text: str) -> bool:
    low = text.lower()
    return "cuda error" in low or "device-side assert" in low


def is_fatal(e: BaseException) -> bool:
    """Whether the whole run must stop at once instead of trying the remaining segments or members: the failure will not
    clear within this run, and the user's message asks them to restart the server (a CUDA fault) or Ollama. True for a
    broken Ollama connection (connection refused or HTTP 404) and for a CUDA fault other than out of memory. A timeout
    is not fatal."""
    if isinstance(e, OllamaUnavailable):
        blob = f"{e} {e.__cause__ or ''}".lower()
        if "timed out" in blob or "timeout" in blob:
            return False                  # a slow answer is not fatal
        return ("refused" in blob or "404" in blob
                or isinstance(e.__cause__, ConnectionRefusedError)
                or getattr(e.__cause__, "code", None) == 404)
    text = str(e)
    return _is_cuda_fault(text) and not _is_oom(text)


def _no_space(e: BaseException, work_dir) -> bool:
    """A disk-full failure: an explicit ENOSPC, or any OSError / ffmpeg (CalledProcessError) failure while the work disk
    has under 500 MB left."""
    if isinstance(e, OSError) and e.errno == 28:
        return True
    if work_dir is not None and isinstance(e, (OSError, subprocess.CalledProcessError)):
        try:
            return shutil.disk_usage(str(work_dir)).free < _DISK_FREE_MIN
        except OSError:
            return False
    return False


def user_message(e: BaseException, work_dir: str | Path | None = None) -> str:
    """Any exception of an analysis in one calm Russian sentence. The order below is checked in tests/test_failures.py;
    the exceptions of the models are English, so they are mapped, never shown. `work_dir` lets the disk-full case look at
    the free space of the folder the results are written to."""
    if isinstance(e, AnalysisCancelled):
        return CANCELLED
    if isinstance(e, UserFacingError):
        return str(e)                     # already meant for the user (may name «Ollama», «AMLAI 1.0», …)
    if isinstance(e, OllamaUnavailable):
        return OLLAMA
    msg = str(e)
    low = msg.lower()
    if _is_oom(msg):
        return OOM
    if isinstance(e, NoSegmentsAnalysed):
        segs = [str(x) for x in e.errors]
        if segs and all(_is_oom(s) for s in segs):
            return OOM
        if segs and all("ollama" in s.lower() for s in segs):
            return OLLAMA
        return NO_SIGNAL
    if _is_cuda_fault(msg):
        return CUDA
    if _no_space(e, work_dir):
        return DISK
    if isinstance(e, OSError) and e.errno in (13, 30):
        return NO_WRITE
    if "no segment could be analysed" in low or "no predictions for any file" in low or "no frames decoded" in low:
        return NO_SIGNAL
    if "all ensemble members failed" in low:
        return ENSEMBLE
    if isinstance(e, subprocess.CalledProcessError):
        return BAD_VIDEO
    if re.search(r"[А-Яа-яЁё]", msg) and not re.search(r"[A-Za-z]", msg):
        return msg                        # messages of this package are already Russian
    return GENERIC
