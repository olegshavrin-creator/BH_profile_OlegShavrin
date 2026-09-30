"""bs3.errors (refactoring plan of 3.1, stage 20): one table that turns any exception of an analysis into a calm Russian
sentence, and is_fatal, which says whether the server must be restarted. Every message is Russian, without a traceback,
the word «Error» or a server path; the models' English exceptions are mapped, never shown. The order of the table is
pinned here, because a wrong order would give the wrong advice (e.g. «нет места» instead of «повреждённое видео»)."""
from __future__ import annotations

import contextlib
import logging
import os
import re
import shutil
import subprocess
import tempfile
import types
import unittest
import urllib.error
from pathlib import Path

from bs3 import errors, media, ollama, pipeline, settings
from bs3.errors import (AnalysisCancelled, NoSegmentsAnalysed, OllamaUnavailable, UserFacingError, is_fatal,
                        user_message)
from bs3.norms import TRAIT_KEYS


class _Usage:
    def __init__(self, free: int):
        self.free = free


@contextlib.contextmanager
def _disk(free_mb: float):
    """shutil.disk_usage(...).free fixed at `free_mb` megabytes while the block runs."""
    real = errors.shutil.disk_usage
    errors.shutil.disk_usage = lambda p: _Usage(int(free_mb * 1024 * 1024))
    try:
        yield
    finally:
        errors.shutil.disk_usage = real


CANONICAL = (errors.CANCELLED, errors.OOM, errors.OLLAMA, errors.NO_SIGNAL, errors.ENSEMBLE, errors.BAD_VIDEO,
             errors.CUDA, errors.DISK, errors.NO_WRITE, errors.GENERIC)


def test_the_canonical_messages_are_clean_russian_sentences():
    """Each mapped sentence is Russian, ends with a full stop, and carries no traceback, «Error» or path (Latin product
    names such as «Ollama» / «AMLAI 1.0» are allowed)."""
    for m in CANONICAL:
        assert re.search(r"[А-Яа-яЁё]", m), m
        assert m.strip().endswith(".") and "\n" not in m, m
        for bad in ("Error", "Traceback", "Exception", "/", "\\"):
            assert bad not in m, (bad, m)


def test_user_message_table():
    low_disk = errors._DISK_FREE_MIN / (1024 * 1024) - 1        # just under the threshold, in MB
    # (exception, work_dir, expected message)
    rows = [
        (AnalysisCancelled("остановлено пользователем"), None, errors.CANCELLED),
        # a UserFacingError is already meant for the user and passes through, Latin names and all
        (UserFacingError("Файл не найден: /home/u/x.mp4"), None, "Файл не найден: /home/u/x.mp4"),
        (UserFacingError("Не отвечает Ollama — модель для AMLAI 1.0."), None,
         "Не отвечает Ollama — модель для AMLAI 1.0."),
        (OllamaUnavailable("connection refused"), None, errors.OLLAMA),
        (RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"), None, errors.OOM),
        # NoSegmentsAnalysed: the shared reason of the skipped segments decides the advice
        (NoSegmentsAnalysed("no segment", errors=["CUDA out of memory", "seg2: cuda out of memory"]), None, errors.OOM),
        (NoSegmentsAnalysed("no segment", errors=["Ollama timed out", "ollama connection refused"]), None,
         errors.OLLAMA),
        (NoSegmentsAnalysed("no segment", errors=["no face found", "no speech found"]), None, errors.NO_SIGNAL),
        (NoSegmentsAnalysed("no segment", errors=[]), None, errors.NO_SIGNAL),
        (RuntimeError("CUDA error: device-side assert triggered"), None, errors.CUDA),
        (OSError(28, "No space left on device"), None, errors.DISK),
        (subprocess.CalledProcessError(1, ["ffmpeg", "in.mp4"]), "/work", errors.DISK),      # ffmpeg + low disk
        (OSError(5, "I/O error"), "/work", errors.DISK),                                     # any OSError + low disk
        (OSError(13, "Permission denied"), None, errors.NO_WRITE),
        (OSError(30, "Read-only file system"), None, errors.NO_WRITE),
        (RuntimeError("No segment could be analysed"), None, errors.NO_SIGNAL),
        (RuntimeError("no predictions for any file"), None, errors.NO_SIGNAL),
        (RuntimeError("no frames decoded"), None, errors.NO_SIGNAL),
        (RuntimeError("All ensemble members failed: oceanai"), None, errors.ENSEMBLE),
        (subprocess.CalledProcessError(1, ["ffprobe", "x.mp4"]), None, errors.BAD_VIDEO),    # no work dir: not disk
        (RuntimeError("Неизвестная модель. Выберите другую."), None, "Неизвестная модель. Выберите другую."),
        (ValueError("unexpected value 3"), None, errors.GENERIC),
        (KeyError("traits"), None, errors.GENERIC),
    ]
    with _disk(low_disk):
        for e, wd, want in rows:
            assert user_message(e, wd) == want, (repr(e), user_message(e, wd))
    # with plenty of free space an ffmpeg failure is a broken video, not a full disk
    with _disk(errors._DISK_FREE_MIN / (1024 * 1024) + 1000):
        assert user_message(subprocess.CalledProcessError(1, ["ffmpeg", "in.mp4"]), "/work") == errors.BAD_VIDEO


def test_the_old_texts_are_word_for_word():
    """The four texts and the generic fallback that the page showed before this stage are unchanged (baseline)."""
    assert user_message(RuntimeError("CUDA out of memory")) == \
        "Не хватило памяти видеокарты. Подождите минуту и запустите анализ заново."
    assert user_message(RuntimeError("no frames decoded")) == \
        "В ролике не найдено ни лица, ни речи, поэтому оценить его нельзя. Проверьте файл."
    assert user_message(RuntimeError("All ensemble members failed: oceanai")) == \
        "Модель не смогла обработать ролик: чаще всего в кадре не найдено лицо или не слышна речь. Проверьте файл."
    assert user_message(subprocess.CalledProcessError(1, ["ffprobe"])) == \
        "Не удалось прочитать видеофайл: возможно, он повреждён или записан в неподдерживаемом формате."
    assert user_message(ValueError("boom")) == \
        "Не удалось обработать ролик из-за внутренней ошибки. Подробности записаны в журнал сервера."


def test_a_user_facing_error_passes_through_with_its_latin_names():
    e = UserFacingError("Не отвечает Ollama — модель для AMLAI 1.0. Запустите Ollama.")
    assert user_message(e) is str(e) or user_message(e) == str(e)
    assert "Ollama" in user_message(e) and "AMLAI 1.0" in user_message(e)


def test_is_fatal():
    """Fatal (the server must be restarted): a broken Ollama connection and a CUDA fault other than out of memory.
    Not fatal: a timeout, out of memory, and ordinary failures."""
    refused = OllamaUnavailable("ollama down")
    refused.__cause__ = ConnectionRefusedError(111, "Connection refused")
    assert is_fatal(refused) is True
    assert is_fatal(OllamaUnavailable("HTTP 404 model not found")) is True
    assert is_fatal(OllamaUnavailable("connection refused by the daemon")) is True
    assert is_fatal(OllamaUnavailable("read timed out")) is False
    assert is_fatal(OllamaUnavailable("something odd")) is False
    assert is_fatal(RuntimeError("CUDA error: device-side assert triggered")) is True
    assert is_fatal(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")) is False
    assert is_fatal(RuntimeError("a plain error")) is False
    assert is_fatal(subprocess.CalledProcessError(1, ["ffprobe"])) is False


# ============================================================ stage 22: early refusals, fatal errors, no leftover jobs

@contextlib.contextmanager
def _swap(module, **names):
    """Swap module globals for the block and put them back afterwards (patch a function or a setting)."""
    old = {k: getattr(module, k) for k in names}
    for k, v in names.items():
        setattr(module, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(module, k, v)


@contextlib.contextmanager
def _no_preflight():
    """Skip run_analysis's upload check and Ollama preflight for a test that drives the rest of the pipeline: the byte
    stub is not a real video and there is no Ollama on the test machine (both checks are tested on their own below)."""
    with _swap(media, check_upload=lambda *a, **k: None), _swap(ollama, status=lambda *a, **k: "ok"):
        yield


@contextlib.contextmanager
def _quiet(name: str):
    """Keep an intentional log.warning of a failure off stderr."""
    lg = logging.getLogger(name)
    lvl = lg.level
    lg.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        lg.setLevel(lvl)


def _raise(msg):
    def fn(*a, **k):
        raise RuntimeError(msg)
    return fn


class _FakeAnalyzer:
    """LongVideoAnalyzer stand-in: one short segment scored by the member of its backend."""

    def __init__(self, member: str):
        self.member = member
        self.calls = []

    def analyze(self, video, work_dir, progress=None, should_stop=None):
        self.calls.append(str(video))
        scores = {k: 0.2 + 0.1 * i for i, k in enumerate(TRAIT_KEYS)}
        if self.member == "mm":
            scores["interview"] = 0.44
        return {"scores": dict(scores), "seconds": 1.0, "variants": {self.member: {k: scores[k] for k in TRAIT_KEYS}},
                "members_used": [self.member], "members_failed": {}, "primary": self.member,
                "primary_used": self.member, "transcript": "", "duration_sec": 15.0, "segments": 1, "timeline": []}


class _FakeStudio:
    """A Studio whose models never load: records the member asked for; the per-segment analyses fail harmlessly (the
    pipeline logs and goes on); explanations are asked for from AMLAI 1.0 only."""
    asr_model = "fake-asr"

    def __init__(self):
        self.asked = []
        self.explained = []
        self._an = {}

    def backend(self, member):
        self.asked.append(member)
        return types.SimpleNamespace(
            cfg=types.SimpleNamespace(corpus=f"ensemble({member}), main={member}", members=(member,)),
            backends={member: types.SimpleNamespace(
                cfg=types.SimpleNamespace(corpus="mupta" if member == "oceanai" else "own checkpoints"))})

    def analyzer(self, member):
        return self._an.setdefault(member, _FakeAnalyzer(member))

    def mm_backend(self, member):
        if member != "mm":
            return None
        studio = self

        class _MM:
            def explain_video(self, *a, **kw):
                studio.explained.append(a[0])
                raise RuntimeError("no GPU in the test: the pipeline logs and goes on")
        return _MM()

    @property
    def text_emotion(self):
        raise RuntimeError("no model in the test")

    voice_emotion = face_expression = text_emotion


def _clip(d) -> Path:
    v = Path(d) / "clip.mp4"
    v.write_bytes(b"\x00" * 64)
    return v


# --- media.check_upload: refuse a bad upload before any folder or model ---

def test_check_upload_refuses_a_bad_upload_before_any_folder_or_model():
    """Each unreadable / silent / too-short shape gives its own calm Russian sentence, no job folder is made and no
    model is built (studio.asked stays empty). The probe is mocked; the real-file version is below."""
    shapes = [
        ({"ffprobe_error": "moov atom not found"}, media.NOT_VIDEO),
        ({"audio_codec": "aac", "duration_sec": 12.0}, media.NOT_VIDEO),           # no video stream
        ({"video_codec": "h264", "duration_sec": 12.0}, media.NO_AUDIO),           # no sound
        ({"video_codec": "h264", "audio_codec": "aac", "duration_sec": 0.3}, "слишком коротк"),
    ]
    for meta, want in shapes:
        studio = _FakeStudio()
        with tempfile.TemporaryDirectory() as d:
            work = Path(d) / "jobs"
            with _swap(media, probe_media=lambda p, **k: dict(meta)):
                try:
                    pipeline.run_analysis(studio, work, str(_clip(d)), member="oceanai", explain=False)
                except UserFacingError as e:
                    got = str(e)
                else:
                    raise AssertionError((meta, "was not refused"))
            assert want in got, (meta, got)
            assert studio.asked == [], meta                    # no model was built
            assert not work.exists(), meta                     # not even the job folder was made


def test_the_two_second_floor():
    """The owner's minimum of two seconds: exactly 2 s is accepted, anything shorter is refused with the duration in the
    message (one decimal, decimal comma)."""
    base = {"video_codec": "h264", "audio_codec": "aac"}
    with _swap(media, probe_media=lambda p, **k: {**base, "duration_sec": 2.0}):
        media.check_upload(Path("x.mp4"), None)               # 2.0 s is long enough (work_dir None: skip the disk check)
    for dur, shown in ((1.9, "1,9"), (0.3, "0,3"), (1.4, "1,4")):
        with _swap(media, probe_media=lambda p, _d=dur, **k: {**base, "duration_sec": _d}):
            try:
                media.check_upload(Path("x.mp4"), None)
            except UserFacingError as e:
                assert "слишком коротк" in str(e) and f"{shown} с" in str(e), (dur, str(e))
            else:
                raise AssertionError(dur)


def _has_ffmpeg() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def _make(path: Path, *args) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args, str(path)], check=True)


def test_check_upload_on_real_files():
    """The same refusals on files ffmpeg makes: a clip without sound, random bytes, an audio-only file and a real clip
    shorter than two seconds are refused; a two-second clip with picture and sound is accepted."""
    if not _has_ffmpeg():
        raise unittest.SkipTest("ffmpeg/ffprobe not installed")

    def refused(path, want):
        try:
            media.check_upload(path, path.parent)
        except UserFacingError as e:
            assert want in str(e), (path.name, str(e))
        else:
            raise AssertionError(f"{path.name} was not refused")

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _make(d / "noaudio.mp4", "-f", "lavfi", "-i", "testsrc=duration=3:size=160x120:rate=10", "-c:v", "libx264",
              "-pix_fmt", "yuv420p")
        refused(d / "noaudio.mp4", media.NO_AUDIO)
        (d / "junk.mp4").write_bytes(os.urandom(4096))
        refused(d / "junk.mp4", media.NOT_VIDEO)
        _make(d / "audio.mp4", "-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-c:a", "aac")
        refused(d / "audio.mp4", media.NOT_VIDEO)
        _make(d / "short.mp4", "-f", "lavfi", "-i", "testsrc=duration=0.3:size=160x120:rate=30", "-f", "lavfi",
              "-i", "sine=frequency=440:duration=0.3", "-c:v", "libx264", "-c:a", "aac", "-pix_fmt", "yuv420p",
              "-shortest")
        refused(d / "short.mp4", "слишком коротк")
        _make(d / "ok.mp4", "-f", "lavfi", "-i", "testsrc=duration=2:size=160x120:rate=15", "-f", "lavfi",
              "-i", "sine=frequency=440:duration=2", "-c:v", "libx264", "-c:a", "aac", "-pix_fmt", "yuv420p",
              "-shortest")
        media.check_upload(d / "ok.mp4", d)                   # a two-second clip with picture and sound passes


def test_a_missing_upload_is_refused_with_no_folder():
    studio = _FakeStudio()
    with tempfile.TemporaryDirectory() as d:
        work = Path(d) / "jobs"
        try:
            pipeline.run_analysis(studio, work, str(Path(d) / "does_not_exist.mp4"), member="oceanai", explain=False)
        except RuntimeError as e:
            assert "не найден" in str(e).lower()
        else:
            raise AssertionError("a missing upload was not refused")
        assert studio.asked == [] and not work.exists()


# --- the Ollama preflight (AMLAI 1.0 only) ---

def test_ollama_preflight_refuses_the_own_model_before_loading():
    """AMLAI 1.0 needs the Ollama vision model: a down or missing model is refused before the model loads (no backend()
    call, no job folder). OCEAN-AI does not use Ollama and runs even when Ollama is down."""
    for st in ("down", "no_model"):
        studio = _FakeStudio()
        with tempfile.TemporaryDirectory() as d:
            work = Path(d) / "jobs"
            with _swap(media, check_upload=lambda *a, **k: None), _swap(ollama, status=lambda *a, **k: st):
                try:
                    pipeline.run_analysis(studio, work, str(_clip(d)), member="mm", explain=True)
                except UserFacingError as e:
                    got = str(e)
                else:
                    raise AssertionError(st)
            assert (got == errors.OLLAMA) if st == "down" else ("нет модели" in got and "OCEAN-AI" in got), got
            assert studio.asked == [] and not work.exists()
    studio = _FakeStudio()
    with tempfile.TemporaryDirectory() as d:
        work = Path(d) / "jobs"
        with _swap(media, check_upload=lambda *a, **k: None), _swap(ollama, status=lambda *a, **k: "down"), \
                _quiet("bs3.pipeline"):
            rep = pipeline.run_analysis(studio, work, str(_clip(d)), member="oceanai", explain=False)
        assert studio.asked == ["oceanai"] and rep["model"]["selected"] == "oceanai"


def test_a_broken_ollama_becomes_a_typed_error_after_the_retries():
    """backend_mm._ollama tries three times (sleeps of 10 and 20 s) and then raises OllamaUnavailable, which is_fatal
    reads as a connection failure — so the run stops at once instead of retrying every segment."""
    from bs3 import backend_mm
    sleeps, calls = [], []

    def boom(*a, **k):
        calls.append(1)
        raise urllib.error.URLError("Connection refused")

    be = backend_mm.MMBackend(backend_mm.MMConfig(ollama_url="http://127.0.0.1:9"))
    with _swap(ollama, post_json=boom), _swap(backend_mm, time=types.SimpleNamespace(sleep=sleeps.append)), \
            _quiet("bs.mm"):
        try:
            be._ollama("prompt", [], 10)
        except OllamaUnavailable as e:
            err = e
        else:
            raise AssertionError("a broken Ollama did not become OllamaUnavailable")
    assert sleeps == [10, 20] and len(calls) == 3
    assert is_fatal(err) is True


# --- fatal errors fail the run at once instead of trying every member / segment ---

def test_a_fatal_error_stops_the_ensemble_at_once():
    from bs3.backend_ensemble import EnsembleBackend, EnsembleConfig
    calls = []

    class Fatal:
        def predict_video(self, video, asr=True, transcript=None, **kw):
            calls.append(video)
            e = OllamaUnavailable("Ollama request failed: Connection refused")
            e.__cause__ = ConnectionRefusedError(111, "Connection refused")
            raise e

    be = EnsembleBackend.__new__(EnsembleBackend)
    be.cfg, be.backends = EnsembleConfig(members=("mm",)), {"mm": Fatal()}
    try:
        be.predict_video("clip.mp4")
    except OllamaUnavailable:
        pass
    else:
        raise AssertionError("a fatal error was turned into «all ensemble members failed»")
    assert len(calls) == 1

    class Blank:
        def predict_video(self, video, asr=True, transcript=None, **kw):
            raise RuntimeError("no face found")

    be2 = EnsembleBackend.__new__(EnsembleBackend)
    be2.cfg, be2.backends = EnsembleConfig(members=("oceanai",)), {"oceanai": Blank()}
    with _quiet("bs.ensemble"):
        try:
            be2.predict_video("clip.mp4")
        except RuntimeError as e:
            assert "all ensemble members failed" in str(e)               # a plain failure is still a member failure
        else:
            raise AssertionError("a non-fatal member failure should still be reported")


def test_a_fatal_error_stops_the_segment_loop_but_a_timeout_is_skipped():
    from bs3 import longvideo

    def cut(v, s, e, out):
        Path(out).write_bytes(b"x")

    calls = []

    class Fatal:
        def predict_video(self, seg, asr=True, transcript=None, **kw):
            calls.append(seg)
            e = OllamaUnavailable("Ollama request failed: Connection refused")
            e.__cause__ = ConnectionRefusedError(111, "refused")
            raise e

    an = longvideo.LongVideoAnalyzer(Fatal(), lang="ru")
    an.transcribe = lambda video, tmp: ("текст", [(0.0, 60.0, "текст")])
    with _swap(longvideo, video_duration=lambda v: 60.0, cut_segment=cut), _quiet("bs.long"):
        with tempfile.TemporaryDirectory() as d:
            try:
                an.analyze("v.mp4", Path(d) / "segs")
            except OllamaUnavailable:
                pass
            else:
                raise AssertionError("a fatal error was swallowed by the segment loop")
    assert len(calls) == 1                                                # stopped after one segment, not all three

    class Flaky:
        def __init__(self):
            self.n = 0

        def predict_video(self, seg, asr=True, transcript=None, **kw):
            self.n += 1
            if self.n == 1:
                raise TimeoutError("read timed out")
            return {"scores": {k: 0.5 for k in TRAIT_KEYS}}

    flaky = Flaky()
    an2 = longvideo.LongVideoAnalyzer(flaky, lang="ru")
    an2.transcribe = lambda video, tmp: ("текст", [(0.0, 60.0, "текст")])
    with _swap(longvideo, video_duration=lambda v: 60.0, cut_segment=cut), _quiet("bs.long"):
        with tempfile.TemporaryDirectory() as d:
            out = an2.analyze("v.mp4", Path(d) / "segs")
    assert flaky.n == 3 and out["segments_ok"] == 2                       # segment 1 timed out (skipped), 2 and 3 scored
    assert out["timeline"][0]["error"] and out["timeline"][0]["scores"] is None


# --- a run that does not finish leaves no folder behind ---

def test_a_failed_run_leaves_no_folder_unless_kept():
    class _FailStudio(_FakeStudio):
        def backend(self, member):
            self.asked.append(member)
            raise RuntimeError("boom while loading the model")

    with tempfile.TemporaryDirectory() as d:
        work = Path(d) / "jobs"
        video = _clip(d)
        with _no_preflight(), _quiet("bs3.pipeline"):
            try:
                pipeline.run_analysis(_FailStudio(), work, str(video), member="oceanai", explain=False)
            except RuntimeError:
                pass
            else:
                raise AssertionError("the failing run did not raise")
            assert work.exists() and list(work.iterdir()) == []          # the job folder was removed, the work dir kept
            with _swap(settings, KEEP_FAILED_JOBS=True):
                try:
                    pipeline.run_analysis(_FailStudio(), work, str(video), member="oceanai", explain=False)
                except RuntimeError:
                    pass
        kept = [p for p in work.iterdir() if p.is_dir()]
        assert len(kept) == 1 and not (kept[0] / "result.json").exists()  # BS3_KEEP_FAILED_JOBS keeps the failed folder


# --- stage 22a: a finished job keeps input.* and the result; its segment clips are removed ---

class _SegAnalyzer(_FakeAnalyzer):
    """Like _FakeAnalyzer, but writes the segment clips and timeline.json into the segments dir it is given, as the
    real LongVideoAnalyzer does, so a test can see whether run_analysis removes them afterwards."""

    def analyze(self, video, work_dir, progress=None, should_stop=None):
        seg = Path(work_dir)
        seg.mkdir(parents=True, exist_ok=True)
        (seg / "seg01_0-20s.mp4").write_bytes(b"segment clip")
        (seg / "timeline.json").write_text("[]", encoding="utf-8")
        return super().analyze(video, work_dir, progress=progress, should_stop=should_stop)


class _SegStudio(_FakeStudio):
    def analyzer(self, member):
        return self._an.setdefault(member, _SegAnalyzer(member))


def test_a_finished_job_keeps_input_but_loses_its_segments():
    """Owner decision of 2026-09-27: a finished analysis keeps its source video (input.*) and result.json, and the
    segment clips (segments/, cut from input.*) are removed once result.json is written. BS3_KEEP_SEGMENTS=1 keeps
    them for debugging."""
    with tempfile.TemporaryDirectory() as d:
        work = Path(d) / "jobs"
        with _no_preflight(), _quiet("bs3.pipeline"):
            rep = pipeline.run_analysis(_SegStudio(), work, str(_clip(d)), member="oceanai", explain=False)
        job = Path(rep["job_dir"])
        assert (job / "result.json").exists() and (job / "input.mp4").is_file()   # the result and the source stay
        assert not (job / "segments").exists()                                    # the segment clips are gone
    with tempfile.TemporaryDirectory() as d:
        work = Path(d) / "jobs"
        with _no_preflight(), _quiet("bs3.pipeline"), _swap(settings, KEEP_SEGMENTS=True):
            rep = pipeline.run_analysis(_SegStudio(), work, str(_clip(d)), member="oceanai", explain=False)
        job = Path(rep["job_dir"])
        assert (job / "segments" / "seg01_0-20s.mp4").is_file()                   # kept for debugging
        assert (job / "input.mp4").is_file()


# --- a failure of one optional step does not kill an otherwise finished analysis ---

def test_a_translation_failure_keeps_the_rest_of_the_run():
    """FP6b: the Marian translation is inside the text-emotion try, so a translation failure only drops that segment's
    text emotions; the run finishes and the analyses that do not need it stay."""
    import bs3.translate as tr
    res = {"duration_sec": 10.0, "input": "x.mp4", "transcript": "Привет мир.",
           "timeline": [], "chunks": [[0.0, 10.0, "Привет мир."]]}
    with _swap(tr, translate_sentences=_raise("marian down")), _swap(pipeline, _wav16k=_raise("no wav")), \
            _quiet("bs3.pipeline"):
        out = pipeline.run_extra_analyses(_FakeStudio(), res, "ru", progress=None)
    assert "emotions_text" not in out and out["per_segment"] and "text_en" not in out["per_segment"][0]
    assert "speech" in out


def test_a_broken_mbti_section_does_not_fail_the_job():
    from bs3 import mbti
    studio = _FakeStudio()
    with tempfile.TemporaryDirectory() as d:
        work = Path(d) / "jobs"
        with _no_preflight(), _swap(mbti, build_section=_raise("mbti boom")), _quiet("bs3.pipeline"):
            rep = pipeline.run_analysis(studio, work, str(_clip(d)), member="oceanai", explain=False)
        assert "mbti" not in rep and (Path(rep["job_dir"]) / "result.json").exists()


def test_a_media_probe_failure_is_recorded_not_fatal():
    studio = _FakeStudio()
    with tempfile.TemporaryDirectory() as d:
        work = Path(d) / "jobs"
        with _no_preflight(), _swap(media, probe_media=_raise("ffprobe exploded")), _quiet("bs3.pipeline"):
            rep = pipeline.run_analysis(studio, work, str(_clip(d)), member="oceanai", explain=False)
        assert isinstance(rep["media"], dict) and rep["media"].get("error")
        assert (Path(rep["job_dir"]) / "result.json").exists()


def test_partial_segment_rows_average_only_the_good_ones():
    """A segment whose text-emotion model failed carries no row into the mean; the weighted mean uses the good rows
    only, and every failed voice/face row is dropped."""
    from bs3.analyses.emotions_text import EMOTION_ORDER

    class S:
        asr_model = "x"

        def __init__(self):
            self.n = 0

        @property
        def text_emotion(self):
            def fn(text):
                self.n += 1
                if self.n == 1:
                    return {e: (1.0 if e == EMOTION_ORDER[0] else 0.0) for e in EMOTION_ORDER}
                raise RuntimeError("second segment: text-emotion model failed")
            return fn

        @property
        def voice_emotion(self):
            raise RuntimeError("no voice model")

        face_expression = voice_emotion

    res = {"duration_sec": 30.0, "input": "x.mp4", "transcript": "речь", "chunks": [[0.0, 30.0, "речь"]],
           "timeline": [{"segment": 1, "start": 0.0, "end": 10.0, "file": "seg1.mp4", "scores": {"o": 0.5},
                         "transcript": ""},
                        {"segment": 2, "start": 10.0, "end": 30.0, "file": "seg2.mp4", "scores": {"o": 0.5},
                         "transcript": ""}]}
    with _swap(pipeline, _wav16k=_raise("no wav")), _quiet("bs3.pipeline"):
        out = pipeline.run_extra_analyses(S(), res, "ru", progress=None)
    et = out["emotions_text"]
    assert et["dominant"] == EMOTION_ORDER[0]                             # only segment 1 had emotions
    assert et["dominant_per_segment"] == [EMOTION_ORDER[0], None]         # segment 2 failed -> no row
    assert "voice" not in out and "face" not in out                       # every voice/face row failed
