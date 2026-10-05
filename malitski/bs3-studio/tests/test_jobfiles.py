"""The files of a job (bs3/jobfiles.py): they are found from the folder being opened, never from the absolute paths
result.json stores, so a job folder moved to another data root still shows its explanation and key frames; writes are
atomic and keep the byte format result.json always had; the page and the PDF do not fail on the null or missing
fields of older or damaged files (a model, modalities or timings of null, the label «собеседование» without a score,
a report without a job folder)."""
from __future__ import annotations

import copy
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from samples import rep

from bs3 import caveats, jobfiles, ru_texts
from bs3.narrative import NO_EXPLAIN_RU
from bs3.norms import TRAIT_KEYS
from bs3.pdf import build as pdf_build
from bs3.web import page

JOB = "20000101_000000_0f3a9c1e"
FRAMES = ("key_01_frame10.jpg", "key_02_frame25.jpg", "key_03_frame40.jpg")
# explanation.json as the page reads it: the modality shares per trait and the readable words with their translation
EXPL = {"modalities": {"input_x_gradient": {k: {"face": {"share": 0.6}, "audio": {"share": 0.38},
                                                "text": {"share": 0.01}, "behavior": {"share": 0.01}}
                                            for k in TRAIT_KEYS}},
        "readable_words": {"transcript_words": {k: {"up": [{"word": "work", "ru": "работа", "signed": 0.01}],
                                                    "down": []} for k in TRAIT_KEYS}},
        "readable_words_by": "ollama"}
# the page values by index (web.page.page_outputs)
I_BARS, I_FACTS, I_FRAMES, I_CONTRIB, I_WORDS, I_SAVED, I_JOB = 1, 2, 14, 15, 16, 20, 21


def _own(name: str = "B") -> dict:
    """The numbers of a sample as a 3.1 job of AMLAI 1.0 (one member, the segments carry its scores)."""
    r = rep(name)
    r["model"].update({"selected": "mm", "primary": "mm", "selected_title": "AMLAI 1.0", "backend": "mm"})
    mm = {k: r["variant_scores"]["mm"][k] for k in TRAIT_KEYS}
    r["variant_scores"] = {"mm": r["variant_scores"]["mm"]}
    for t in r["timeline"]:
        t.update({"members_used": ["mm"], "primary_used": "mm", "variants": {"mm": dict(mm)}, "scores": dict(mm)})
    r["representative_segment"] = 3
    r["media"] = {"fps": 30.0}
    return r


def _frames(ex: Path) -> None:
    from PIL import Image
    ex.mkdir(parents=True, exist_ok=True)
    for f in FRAMES:
        Image.new("RGB", (64, 48), (90, 90, 90)).save(ex / f, "JPEG")


def _pdf_text(path: Path) -> str:
    if not shutil.which("pdftotext"):
        raise unittest.SkipTest("pdftotext (poppler-utils) not installed")
    r = subprocess.run(["pdftotext", "-enc", "UTF-8", str(path), "-"], capture_output=True, text=True, check=True)
    return re.sub(r"\s+", " ", r.stdout)


# ---------------------------------------------------------------- the folder being opened
def test_moved_data_root():
    """A job folder moved to another data root: its result.json still names the old folder in `job_dir` and
    `key_frames`. load_job points the job at the folder being opened, and the page finds the explanation and the key
    frames there; the PDF gets the same frames and the uploaded video of that folder."""
    with tempfile.TemporaryDirectory() as d:
        old = Path(d) / "old_root" / "web_jobs" / JOB             # where the job was made; it is not there any more
        job = Path(d) / "new_root" / "web_jobs" / JOB
        _frames(job / "explain")
        (job / "input.mp4").write_bytes(b"\x00" * 16)
        r = _own()
        r["job_dir"] = str(old)
        r["key_frames"] = [str(old / "explain" / f) for f in FRAMES]
        r["media"] = {"error": "probe failed"}                   # the PDF then probes the uploaded video itself
        (job / "result.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        (job / "explain" / "explanation.json").write_text(json.dumps(EXPL, ensure_ascii=False), encoding="utf-8")
        stored = (job / "result.json").read_bytes()
        loaded, expl = jobfiles.load_job(job)
        assert loaded["job_dir"] == str(job) and expl == EXPL
        assert jobfiles.key_frame_paths(job, loaded) == [job / "explain" / f for f in FRAMES]
        outs = page.page_outputs(loaded)
        # export_pdf reads the same folder: the key frames and the video handed to the PDF are the ones in it
        from bs3 import media
        from bs3.pdf import charts as pdf_charts
        got = {}

        def build(view, out, **kw):
            got.update(kw, out=out)
            return str(out)
        saved = pdf_build.build_pdf, pdf_charts.save_pdf_charts, media.probe_media
        pdf_build.build_pdf, pdf_charts.save_pdf_charts = build, lambda *a, **k: {}
        media.probe_media = lambda p: {"probed": Path(p).name}
        try:
            pdf_build.export_pdf(job)
        finally:
            pdf_build.build_pdf, pdf_charts.save_pdf_charts, media.probe_media = saved
        assert (job / "result.json").read_bytes() == stored                 # nothing was written back
    frames, contrib, words = outs[I_FRAMES], outs[I_CONTRIB], outs[I_WORDS]
    assert frames.count("<img src='data:image/jpeg;base64,") == len(FRAMES), frames[:200]
    assert page.NO_FRAMES_MM not in frames
    assert "Какая доля оценки модели AMLAI 1.0" in contrib and "<table" in contrib
    assert "«работа»" in words
    assert outs[I_SAVED] == JOB and outs[I_JOB] == str(job)
    assert got["key_frames"] == [str(job / "explain" / f) for f in FRAMES]
    assert got["explanation"] == EXPL and got["media"] == {"probed": "input.mp4"} and Path(got["out"]).parent == job


def test_key_frame_paths():
    """The frames in the order result.json lists them, found by name in <job>/explain; the sorted frames on disk when
    the file has no list (the order the pipeline writes the list in, see the audit of the 11 jobs)."""
    with tempfile.TemporaryDirectory() as d:
        job = Path(d) / JOB
        ex = job / "explain"
        ex.mkdir(parents=True)
        for f in reversed(FRAMES):
            (ex / f).write_bytes(b"jpeg")
        (ex / "explanation.json").write_text("{}", encoding="utf-8")
        on_disk = sorted(ex.glob("key_*.jpg"))
        assert [p.name for p in on_disk] == list(FRAMES)
        stale = "/elsewhere/bs3_data/web_jobs/20000101_000000/explain/"
        assert jobfiles.key_frame_paths(job, {"key_frames": [stale + f for f in FRAMES]}) == on_disk
        assert jobfiles.key_frame_paths(str(job), {"key_frames": [str(p) for p in on_disk]}) == on_disk
        for no_list in ({}, {"key_frames": None}):
            assert jobfiles.key_frame_paths(job, no_list) == on_disk, no_list
        assert jobfiles.key_frame_paths(job, {"key_frames": []}) == []          # the analysis saved none
        listed = [stale + FRAMES[2], None, stale + "key_09_frame99.jpg", "", FRAMES[0]]
        assert jobfiles.key_frame_paths(job, {"key_frames": listed}) == [ex / FRAMES[2], ex / FRAMES[0]]
        assert jobfiles.key_frame_paths(Path(d) / "missing", {}) == []
        # the uploaded video of the folder, if any
        assert jobfiles.input_file(job) is None
        (job / "input.mov").write_bytes(b"\x00")
        assert jobfiles.input_file(job) == job / "input.mov"


# ---------------------------------------------------------------- reading and writing
def test_read_json_missing_or_invalid():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d)
        (f / "cut.json").write_text('{"traits": ', encoding="utf-8")
        (f / "list.json").write_text("[1, 2]", encoding="utf-8")
        (f / "bytes.json").write_bytes(b"\xff\xfe\x00{")
        (f / "ok.json").write_text('{"a": 1, "b": "ё"}', encoding="utf-8")
        for name in ("missing.json", "cut.json", "list.json", "bytes.json"):
            assert jobfiles.read_json(f / name) is None, name
        assert jobfiles.read_json(f) is None                                  # a folder
        assert jobfiles.read_json(f / "ok.json") == {"a": 1, "b": "ё"}
        # a job needs its result.json
        job = f / JOB
        job.mkdir()
        for content, error in ((None, FileNotFoundError), ('{"a": ', ValueError), ("[]", ValueError)):
            if content is not None:
                (job / "result.json").write_text(content, encoding="utf-8")
            try:
                jobfiles.load_job(job)
            except error:
                pass
            else:
                raise AssertionError(f"load_job did not raise {error.__name__} for {content!r}")
        (job / "result.json").write_text('{"job_dir": "/elsewhere/x"}', encoding="utf-8")
        assert jobfiles.load_job(job) == ({"job_dir": str(job)}, None)


def test_write_json_keeps_the_byte_format():
    """result.json and explanation.json are written as the pipeline and ru_texts always wrote them: indent 2, UTF-8
    without escapes, NaN allowed; nothing else is left in the folder."""
    data = {"job_dir": "/home/u/bs3_data/web_jobs/x",
            "traits": {"openness": {"score": 0.5123, "name_ru": "Открытость"}},
            "text": "ёлка «в кавычках»\nвторая строка", "nan": float("nan"), "inf": float("inf"),
            "list": [1, 2.5, None, True, []], "empty": {}}
    with tempfile.TemporaryDirectory() as d:
        new, old = Path(d) / "result.json", Path(d) / "old.json"
        jobfiles.write_json(new, data)
        old.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")      # the call of 3.1 before
        assert new.read_bytes() == old.read_bytes()
        assert sorted(p.name for p in Path(d).iterdir()) == ["old.json", "result.json"]
        back = jobfiles.read_json(new)
        assert math.isnan(back["nan"]) and back["text"] == data["text"] and back["traits"] == data["traits"]
        jobfiles.write_json(str(new), {"a": 1})                              # replaces the file in place
        assert json.loads(new.read_text(encoding="utf-8")) == {"a": 1}
        assert sorted(p.name for p in Path(d).iterdir()) == ["old.json", "result.json"]


def test_write_json_failure_leaves_no_file():
    """A full disk (os.replace fails with errno 28) or data that is not JSON: write_json raises and leaves neither the
    file nor its temporary copy; a file that was there stays as it was. ru_texts.write_json, which stores the Russian
    texts a page adds, logs the failure and lets the page go on."""
    real = os.replace

    def full(*_a, **_k):
        raise OSError(28, "No space left on device")

    log = logging.getLogger("bs3.ru_texts")
    level = log.level
    with tempfile.TemporaryDirectory() as d:
        folder = Path(d)
        path = folder / "result.json"

        def untouched() -> bool:
            return [p.name for p in folder.iterdir()] == ["result.json"] and path.read_text(encoding="utf-8") == "{}"

        os.replace = full
        log.setLevel(logging.CRITICAL)
        try:
            try:
                jobfiles.write_json(path, {"a": 1})
            except OSError as e:
                assert e.errno == 28
            else:
                raise AssertionError("write_json did not raise")
            assert list(folder.iterdir()) == []
            path.write_text("{}", encoding="utf-8")
            try:
                jobfiles.write_json(path, {"new": 2})
            except OSError:
                pass
            else:
                raise AssertionError("write_json did not raise")
            assert untouched()
            ru_texts.write_json(path, {"new": 2})                             # no exception: the page goes on
            assert untouched()
        finally:
            os.replace = real
            log.setLevel(level)
        try:
            jobfiles.write_json(path, {"not json": object()})
        except TypeError:
            pass
        else:
            raise AssertionError("write_json did not raise for data that is not JSON")
        assert untouched()


# ---------------------------------------------------------------- null and missing fields
def _null_cases() -> dict:
    """{case: report} of older or damaged files that the page and the PDF must still show."""
    cases = {}
    for field in ("model", "modalities_used", "timings_sec"):
        r = _own()
        r[field] = None
        cases[f"{field} null"] = r
    r = _own()
    r["interview"] = {"name_ru": "впечатление «пригласить на собеседование»"}
    cases["interview without score"] = r
    r = _own()
    r["interview"] = {"score": None, "name_ru": "впечатление «пригласить на собеседование»"}
    cases["interview score null"] = r
    cases["no job folder, AMLAI 1.0"] = _own()
    cases["no job folder, OCEAN-AI"] = rep("B")
    return cases


def test_page_and_pdf_on_null_fields():
    with tempfile.TemporaryDirectory() as d:
        for i, (case, r) in enumerate(_null_cases().items()):
            if not case.startswith("no job folder"):
                job = Path(d) / f"20000101_00000{i}_0f3a9c1e"
                job.mkdir()
                (job / "result.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
                r, _ = jobfiles.load_job(job)
            else:
                assert "job_dir" not in r
            outs = page.page_outputs(copy.deepcopy(r))
            assert len(outs) == page.N_PAGE, case
            bars, facts = outs[I_BARS], outs[I_FACTS]
            assert bars.count("<b>Экстраверсия</b>") == 1, case
            if case.startswith("interview"):
                # the label without a score is absent: no bar, no card, no C2 under the bars, nothing in the PDF
                assert "собеседовани" not in bars.lower() and caveats.text("C2") not in bars, case
                assert "собеседовани" not in facts.lower(), case
            if case.startswith("no job folder"):
                # rendered without the files of a job: no explanation, no key frames, no folder for the PDF button
                assert outs[I_SAVED] == "" and outs[I_JOB] == "", case
                assert outs[I_WORDS] == "" and "<img" not in outs[I_FRAMES], case
                if case.endswith("AMLAI 1.0"):
                    assert outs[I_CONTRIB] == "" and page.NO_FRAMES_MM in outs[I_FRAMES], case
                else:
                    assert NO_EXPLAIN_RU in outs[I_CONTRIB] and page.NO_FRAMES_OCEANAI in outs[I_FRAMES], case
            out = Path(d) / f"report_{i}.pdf"
            assert pdf_build.build_pdf(copy.deepcopy(r), out) == str(out), case
            assert out.stat().st_size > 10_000, case
            if case.startswith("interview"):
                text = _pdf_text(out)
                assert "Коричневая полоска" not in text and caveats.text("C2")[:40] not in text, case
