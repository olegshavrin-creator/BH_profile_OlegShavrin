"""The page of BS Profiler 3.1 (change request 3.1, sections 2–4) built headless, without a GPU: the «Модель» radio
(OCEAN-AI / AMLAI 1.0, no info text) and no «Объяснения» checkbox; the block labels of one model («Оценки по чертам»,
«Тип MBTI», «Модель и время обработки», the AMLAI 1.0 modality block); the compact «Характеристика личности» window
(APP_CSS: a flex item with a zero basis and a minimum height, its html-container scrolling) with the key facts as a
row under the top pair; page_outputs of a synthetic OCEAN-AI job (the note of the tab «Объяснения», the model line)
and of a job of AMLAI 1.0. Gradio is imported here (a few seconds); the Studio loads nothing until an analysis.
What Gradio serves (in-process through TestClient, no port): no file of a job folder by /gradio_api/file=, the PDF of
the button from a copy in Gradio's own temp folder, which Gradio deletes with its other temp files. The page shows no
server path: «Сохранено в» holds the job name, the result.json of the tab «Данные» file and folder names."""
from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
import time
from datetime import timedelta
from pathlib import Path

from samples import own, rep

import bs3
from bs3.narrative import NO_EXPLAIN_RU
from bs3.web import app, page, style


def _job(r: dict, base: Path, name: str) -> dict:
    job = base / name
    job.mkdir(parents=True)
    r["job_dir"] = str(job)
    (job / "result.json").write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
    return json.loads((job / "result.json").read_text(encoding="utf-8"))


def _strip(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def _server_paths(r: dict, job: Path, upload: Path) -> None:
    """The paths a result.json keeps on the server: the upload, the key frames and the segment files."""
    r["input"] = str(upload)
    r["key_frames"] = [str(job / "explain" / "key_01_frame10.jpg")]
    for i, t in enumerate(r["timeline"], 1):
        t["file"] = str(job / "segments" / f"seg{i:02d}_0-20s.mp4")


def _no_server_paths(outs: tuple, tmp: str, job: Path) -> None:
    """The page shows the job name in «Сохранено в» and names in the result.json of the tab «Данные»; only the hidden
    job folder of the PDF button (index 21, a gr.State kept on the server) holds the path."""
    assert outs[20] == job.name and outs[21] == str(job)
    assert f'"job_dir": "{job.name}"' in outs[19]
    shown = json.loads(outs[19])
    assert shown["input"] == "input.mp4" and shown["key_frames"] == ["key_01_frame10.jpg"]
    files = [t["file"] for t in shown["timeline"]]
    assert files == [f"seg{i:02d}_0-20s.mp4" for i in range(1, len(files) + 1)] and files
    leaks = [i for i, o in enumerate(outs) if isinstance(o, str) and i != 21 and tmp in o]
    assert leaks == [], leaks


def test_result_json_without_server_paths():
    """The paths become names; missing, null and odd entries pass as they are (older jobs)."""
    data = {"job_dir": "/home/u/bs3_data/web_jobs/20000101_000000_0f3a9c1e", "input": "/home/u/source/j/input.mp4",
            "key_frames": ["/home/u/j/explain/key_01_frame10.jpg", None],
            "timeline": [{"file": "/home/u/j/segments/seg01_0-20s.mp4", "segment": 1}, {"file": None}, {}, "odd"],
            "media": {"file_name": "clip.mp4"}, "model": {"asr_model": "openai/whisper-large-v3-turbo"}}
    out = page._without_server_paths(data)
    assert out is data and out["job_dir"] == "20000101_000000_0f3a9c1e" and out["input"] == "input.mp4"
    assert out["key_frames"] == ["key_01_frame10.jpg", None]
    assert out["timeline"] == [{"file": "seg01_0-20s.mp4", "segment": 1}, {"file": None}, {}, "odd"]
    assert out["model"]["asr_model"] == "openai/whisper-large-v3-turbo" and out["media"] == {"file_name": "clip.mp4"}
    for odd in ({"key_frames": None, "timeline": None, "input": None}, {}):
        assert page._without_server_paths(dict(odd)) == odd


def test_page_blocks_is_the_contract():
    """PAGE_BLOCKS names the result blocks in order; it is the snapshot contract of scripts/compare_baseline.py (which
    asserts the two tuples equal at start-up). N_PAGE follows it and page_index round-trips."""
    assert page.PAGE_BLOCKS == (
        "radar", "bars", "key_facts_html", "characterization_html", "traits_timeline", "emotions_timeline",
        "voice_timeline", "speech_timeline", "emotion_bars", "segments_table", "speech_cards", "transcript",
        "face_cards", "face_chart", "key_frames_html", "contrib", "words", "behavior_description",
        "model_and_time", "result_json", "saved_to", "job_state", "method", "emo_intro", "mbti_types",
        "mbti_strip", "mbti_read")
    assert page.N_PAGE == len(page.PAGE_BLOCKS) == 27
    assert page.page_index("transcript") == 11 and page.page_index("mbti_read") == 26


def test_page_builds_with_the_model_radio_and_no_checkbox():
    import gradio as gr
    from bs3.pipeline import Studio
    with tempfile.TemporaryDirectory() as d:
        demo = app.build_app(Studio(), Path(d))
    comps = list(demo.blocks.values())
    radios = [c for c in comps if isinstance(c, gr.Radio)]
    assert len(radios) == 1
    radio = radios[0]
    # AMLAI 1.0 is the left choice and is already selected; OCEAN-AI is there for whoever wants it
    assert radio.label == "Модель" and radio.value == bs3.DEFAULT_MODEL == "mm"
    assert [c[0] for c in radio.choices] == ["AMLAI 1.0", "OCEAN-AI"] and [c[1] for c in radio.choices] == ["mm", "oceanai"]
    assert not getattr(radio, "info", None)
    assert not [c for c in comps if isinstance(c, gr.Checkbox)]                  # explanations follow the model
    labels = {getattr(c, "label", None) for c in comps}
    for lab in ("Оценки по чертам", "Тип MBTI", "Тип по ходу ролика", "Как читать тип MBTI",
                "Модель и время обработки", "Вклад модальностей в оценку модели AMLAI 1.0", "Ключевые факты",
                "Характеристика личности", "Видео"):
        assert lab in labels, lab
    for lab in ("Оценки по чертам и второе мнение", "Тип MBTI по двум системам", "Участники ансамбля и время обработки",
                "Вклад модальностей в оценку своей модели", "Язык речи",
                "Объяснения (ключевые кадры, вклад модальностей, слова)"):
        assert lab not in labels, lab
    # the analyze handler takes (video, model) only, and writes the status line, the page blocks and the PDF button
    fns = [f for f in demo.fns.values() if getattr(f, "fn", None) is not None and f.fn.__name__ == "analyze"]
    assert len(fns) == 1 and len(fns[0].inputs) == 2
    assert [c.label for c in fns[0].inputs] == ["Видео", "Модель"]
    assert len(fns[0].outputs) == page.N_PAGE + 2
    # the compact window: the block class, the flex rules and the scrolling container in the page CSS
    char = [c for c in comps if isinstance(c, gr.HTML) and getattr(c, "label", None) == "Характеристика личности"]
    assert len(char) == 1 and "bs3-char" in char[0].elem_classes
    css = style.APP_CSS
    assert f".row.bs3-pair>.column>.bs3-char{{display:flex;flex-direction:column;flex:1 1 0;min-height:{style.CHAR_MIN_PX}px}}" in css
    assert ".bs3-char>.html-container{flex:1 1 0;min-height:0;overflow-y:auto;overflow-x:hidden}" in css
    assert "max-height:none!important" not in css and 300 <= style.CHAR_MIN_PX <= 360
    # the key facts block is not inside the top pair (a row of its own under it)
    facts = [c for c in comps if isinstance(c, gr.HTML) and getattr(c, "label", None) == "Ключевые факты"][0]
    pairs = [c for c in comps if isinstance(c, gr.Row) and "bs3-pair" in (c.elem_classes or [])]
    inside = {id(x) for row in pairs for col in row.children for x in getattr(col, "children", [])}
    assert id(facts) not in inside and id(char[0]) in inside
    # the footer caveats: what holds for both models (C2 stands under the bars of a job that shows the label)
    assert bs3.caveats.PAGE_FOOTER == ("C1", "C10", "C3") and not hasattr(app, "FOOTER_CAVEATS")


def test_page_outputs_oceanai_job():
    with tempfile.TemporaryDirectory() as d:
        r = rep("B")
        r["behavior_description_ru"] = "[0–20 с] Человек говорит спокойно."      # of the second model of 3.0
        r["interview"] = {"score": 0.4011, "name_ru": "впечатление «пригласить на собеседование»"}   # of the same
        r["narrative"] = "Оценки дала система OCEAN-AI …"                          # the stored summary of 2.0
        for t in r["timeline"]:
            if isinstance(t.get("scores"), dict):
                t["scores"]["interview"] = 0.4
        # a job of the older naming (no random suffix) still opens; its result.json keeps the server paths, and the
        # upload of an imported 2.0 job lies outside the job folder
        job = Path(d) / "20000101_000000"
        _server_paths(r, job, Path(d) / "source" / "input.mp4")
        r = _job(r, Path(d), job.name)
        outs = page.page_outputs(r)
        saved = json.loads((job / "result.json").read_text(encoding="utf-8"))
    assert saved["input"] == str(Path(d) / "source" / "input.mp4") and saved["job_dir"] == str(job)   # file unchanged
    _no_server_paths(outs, d, job)
    assert len(outs) == page.N_PAGE == 27
    bars, facts, contrib, words, desc, members = outs[1], outs[2], outs[15], outs[16], outs[17], outs[18]
    assert _strip(contrib) == NO_EXPLAIN_RU and words == "" and desc == ""     # the one note of the tab «Объяснения»
    # the bars of one model: five traits once, no framed block («второе мнение» of 3.0) after the scale row, and no
    # label of the other model anywhere on the overview (bars, key facts, timeline chart)
    assert bars.count("<b>Экстраверсия</b>") == 1 and "второе мнение" not in bars.lower()
    assert "border-radius:8px'><div style='font-weight:600" not in bars
    for h in (bars, facts, outs[4]):
        assert "собеседовани" not in h.lower() and "AMLAI" not in h, h[:80]
    lines = members.split("\n")
    assert lines[0].startswith("Модель OCEAN-AI, веса MuPTA: открытость опыту 0.71")
    assert lines[1].startswith("Обработка заняла ")
    assert page.DATA_TRIMMED in lines                                          # an older job: percentiles left out
    assert "Ключевых кадров нет: модель OCEAN-AI не строит объяснений" in outs[14]
    page_text = " ".join(_strip(o) for o in outs if isinstance(o, str))
    for bad in ("второе мнение", "Второе мнение", "своя модель", "Своя модель", "своей модели", "MM-PSYCHE",
                "Участники ансамбля", "основная оценка", "Основная система", "среднее двух систем", "английской речи",
                "для русской речи", "язык речи"):
        assert bad not in page_text, bad
    assert "OCEAN-AI, веса MuPTA" in _strip(outs[24]) and "OCEAN-AI: ENFJ во всех 26 отрезках" in _strip(outs[25])


def test_page_outputs_own_model_job():
    from bs3 import caveats
    with tempfile.TemporaryDirectory() as d:
        r = own("B")
        r["behavior_description_ru"] = "[0–20 с] Человек говорит спокойно."
        r["interview"] = {"score": 0.4011, "name_ru": "впечатление «пригласить на собеседование»"}
        job = Path(d) / "20000102_000000_0f3a9c1e"                     # a job of 3.1: time stamp and random suffix
        _server_paths(r, job, job / "input.mp4")
        r = _job(r, Path(d), job.name)
        outs = page.page_outputs(r)
    _no_server_paths(outs, d, job)
    contrib, members, frames = outs[15], outs[18], outs[14]
    assert contrib == ""                                                        # no explanation on disk: empty, no note
    # the label of AMLAI 1.0 with its bar and C2 under the bars; a fresh job leaves nothing out of the tab «Данные»
    assert "Впечатление «пригласить на собеседование»" in outs[1] and caveats.text("C2") in outs[1]
    assert "Впечатление «собеседование»" in outs[2] and page.DATA_TRIMMED not in members
    assert outs[17] == "[0:00–0:20] Человек говорит спокойно."                    # the description of AMLAI 1.0 stays
    assert members.startswith("Модель AMLAI 1.0: открытость опыту 0.") and "MuPTA" not in members
    assert "Ключевые кадры не построены: лицо в кадре не найдено" in frames
    # the panel, the strip and the bars name AMLAI 1.0 only (C7 of the reading guide names both models on purpose)
    for i in (1, 24, 25):
        assert "OCEAN-AI" not in outs[i] and "MuPTA" not in outs[i], i
    assert ">AMLAI 1.0</div>" in outs[24] and "AMLAI 1.0: строгий тип ISTP во всех 33 отрезках" in _strip(outs[25])
    assert "Оценки дала модель AMLAI 1.0" in _strip(outs[22]) and "MBTI по AMLAI 1.0" in outs[3]


@contextlib.contextmanager
def _gradio_temp(base: Path):
    """GRADIO_TEMP_DIR -> base/gradio while the test runs (Gradio reads it when the page and the app are built), the
    old value back afterwards."""
    old = os.environ.get("GRADIO_TEMP_DIR")
    up = base / "gradio"
    up.mkdir()
    os.environ["GRADIO_TEMP_DIR"] = str(up)
    try:
        yield up
    finally:
        if old is None:
            os.environ.pop("GRADIO_TEMP_DIR", None)
        else:
            os.environ["GRADIO_TEMP_DIR"] = old


JOB_FILES = ("input.mp4", "segments/seg01_0-20s.mp4", "segments/audio16k.wav", "segments/timeline.json", "result.json",
             "explain/explanation.json", "explain/key_01_frame10.jpg", "charts/chart_profile.png",
             f"{bs3.PRODUCT_SLUG}_report_video.pdf")


def _job_folder(work_dir: Path) -> Path:
    """A finished job folder with one file of every kind a real one holds (made-up content)."""
    job = work_dir / "20000101_000000"
    for name in JOB_FILES:
        p = job / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"%PDF-1.4\n%%EOF\n" if name.endswith(".pdf") else name.encode())
    return job


def _client(demo):
    """The app as launch() without allowed_paths serves it, in-process (no port)."""
    from fastapi.testclient import TestClient
    from gradio.routes import App
    demo.allowed_paths, demo.blocked_paths = [], []
    return TestClient(App.create_app(demo))


def test_file_route_serves_no_job_file():
    """/gradio_api/file=<path> gives 403 for every file of a job folder: the uploaded video, the segments, the
    transcript in result.json, the key frames, the charts and the PDF. The page needs none of them (key frames are
    data URIs, charts are srcdoc); the old launch(allowed_paths=[work dir]) served them to anyone with the path."""
    from bs3.pipeline import Studio
    with tempfile.TemporaryDirectory() as d, _gradio_temp(Path(d)):
        wd = Path(d) / "web_jobs"
        job = _job_folder(wd)
        demo = app.build_app(Studio(), wd)
        client = _client(demo)
        files = sorted(p for p in job.rglob("*") if p.is_file())
        assert len(files) == len(JOB_FILES)
        for f in files:
            r = client.get(f"/gradio_api/file={f}")
            assert r.status_code == 403, (str(f.relative_to(job)), r.status_code)
        # the control: the same request with the old allowed_paths gets the video, so the 403 above is the route's
        demo.allowed_paths = [str(wd)]
        assert client.get(f"/gradio_api/file={job / 'input.mp4'}").status_code == 200


def test_pdf_for_download():
    """The PDF button hands Gradio a copy of the stored PDF in Gradio's upload folder, in a new random folder per
    click, under the same file name; the stored PDF stays in the job folder. Gradio serves the copy and takes it as
    the button's output from any start directory, which the stored PDF itself does not pass without allowed_paths;
    Gradio's hourly pass (delete_cache) deletes the copy with its other temp files after 22 hours."""
    import gradio as gr
    from gradio import processing_utils, route_utils
    from gradio.context import LocalContext
    from gradio.exceptions import InvalidPathError
    from gradio.utils import get_upload_folder
    from bs3.pipeline import Studio
    with tempfile.TemporaryDirectory() as d, _gradio_temp(Path(d)) as up:
        base = Path(d)
        job = _job_folder(base / "web_jobs")
        pdf = job / JOB_FILES[-1]
        calls = []
        export_pdf = app.export_pdf
        app.export_pdf = lambda job_dir: calls.append(job_dir) or str(pdf)
        try:
            a, b = Path(app.pdf_for_download(job)), Path(app.pdf_for_download(str(job)))
        finally:
            app.export_pdf = export_pdf
        assert calls == [job, str(job)]
        for p in (a, b):
            assert p.parent.parent == Path(get_upload_folder()) == up and re.fullmatch(r"[0-9a-f]{32}", p.parent.name)
            assert p.name == pdf.name and p.read_bytes() == pdf.read_bytes() and job not in p.parents
        assert a.parent != b.parent and pdf.exists()

        demo = app.build_app(Studio(), base / "web_jobs")
        client = _client(demo)
        r = client.get(f"/gradio_api/file={a}")
        # Gradio 5.8 sends a file of its own folder that is not an image, audio, video, text or json as an attachment
        # of type application/octet-stream, the PDF too (it did the same with its cache copy of the stored PDF)
        assert r.status_code == 200 and r.content == pdf.read_bytes()
        disposition = r.headers["content-disposition"]
        assert disposition.startswith("attachment") and pdf.name in disposition

        # the button's output through Gradio's own check, as the launched app runs it after make_pdf, started from a
        # directory that holds neither the job folder nor the system temp folder
        btn = next(c for c in demo.blocks.values() if isinstance(c, gr.DownloadButton))
        elsewhere = base / "elsewhere"
        elsewhere.mkdir()
        cwd, tmpdir = os.getcwd(), tempfile.tempdir
        token = LocalContext.blocks.set(demo)
        demo.has_launched = True
        os.chdir(elsewhere)
        tempfile.tempdir = str(elsewhere)
        try:
            out = processing_utils.move_files_to_cache(btn.postprocess(str(a)), btn, postprocess=True)
            refused = False
            try:
                processing_utils.move_files_to_cache(btn.postprocess(str(pdf)), btn, postprocess=True)
            except InvalidPathError:
                refused = True
        finally:
            os.chdir(cwd)
            tempfile.tempdir = tmpdir
            demo.has_launched = False
            LocalContext.blocks.reset(token)
        assert refused, "the stored PDF passed without allowed_paths: the check above proves nothing"
        assert out["path"] == str(a) and out["orig_name"] == pdf.name
        assert client.get(out["url"]).status_code == 200

        # delete_cache: the hourly pass of Gradio deletes the copy once it is older than the age, and keeps it an hour
        # after the click. Gradio 5.8 compares timedelta.seconds (the part under a day), so the age stays under a day
        # with room for two passes; a full day (86400) would never delete anything
        frequency, age = demo.delete_cache
        assert (frequency, age) == (3600, 79200) and age <= 86400 - 2 * frequency
        real_datetime = route_utils.datetime
        shift = timedelta(0)

        class Later(real_datetime):
            @classmethod
            def now(cls, tz=None):
                return real_datetime.now(tz) + shift

        route_utils.datetime = Later
        try:
            shift = timedelta(hours=1)
            route_utils.delete_files_created_by_app(demo, age)
            assert a.exists()
            shift = timedelta(seconds=age + frequency)
            route_utils.delete_files_created_by_app(demo, age)
            assert not a.exists()
        finally:
            route_utils.datetime = real_datetime
        assert pdf.exists()                                                     # the job folder is not Gradio's


def test_sweep_empty_downloads_removes_only_old_empty_token_folders():
    """_sweep_empty_downloads clears the empty folders pdf_for_download leaves once Gradio's delete_cache has removed
    the copied PDF, and only those: it takes a folder only when the name is 32 hex characters, the folder is empty, and
    it has not been touched for over an hour. So a fresh folder a click may still be filling, a folder that still holds
    its PDF, and any folder not named like ours are all kept. test_pdf_for_download only ever exercises the no-op path
    (the folder of the first click still holds its fresh PDF on the second), so without this the removal itself, the
    one-hour cutoff and the name filter are unverified: dropping the cutoff would sweep an in-flight download's folder,
    and turning the rmdir into a recursive delete would take a folder that still holds its PDF."""
    with tempfile.TemporaryDirectory() as d:
        folder = Path(d)
        old = time.time() - 2 * 3600                          # comfortably past the one-hour cutoff
        old_empty = folder / ("a" * 32)                       # 32-hex, empty, old: the only one to remove
        fresh_empty = folder / ("b" * 32)                     # 32-hex, empty, but too new — a click may be filling it
        old_full = folder / ("c" * 32)                        # 32-hex, old, but still holds its PDF
        old_nonhex = folder / "not_a_download_folder"         # old and empty, but not one of ours by name
        for p in (old_empty, fresh_empty, old_full, old_nonhex):
            p.mkdir()
        (old_full / "report.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
        for p in (old_empty, old_full, old_nonhex):           # set the mtime AFTER writing into old_full
            os.utime(p, (old, old))
        app._sweep_empty_downloads(folder)
        survivors = sorted(p.name for p in folder.iterdir())
        assert not old_empty.exists()                         # empty + old + a token name -> removed
        assert survivors == sorted([fresh_empty.name, old_full.name, old_nonhex.name])
        assert (old_full / "report.pdf").read_bytes() == b"%PDF-1.4\n%%EOF\n"    # its PDF is untouched


@contextlib.contextmanager
def _patched(module, **names):
    """Module globals replaced while the block runs, the old values back afterwards."""
    old = {k: getattr(module, k) for k in names}
    for k, v in names.items():
        setattr(module, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(module, k, v)


def test_pdf_button_click():
    """The click of «Экспорт в PDF» as the launched app runs it (Blocks.process_api: the job folder from the session's
    gr.State, the registered handler, the button's postprocess and Gradio's path check), started from a directory that
    holds neither the job folder nor the system temp folder: the button gets the copy in Gradio's upload folder
    (pdf_for_download). The control: a handler that hands over the stored PDF of export_pdf is refused there, so the
    download would fail. No job yet and a failed export end in the Russian error dialog, the failure also in the log."""
    import asyncio
    import logging
    import gradio as gr
    from gradio.exceptions import InvalidPathError
    from gradio.state_holder import SessionState
    from gradio.utils import get_upload_folder
    from bs3.pipeline import Studio
    with tempfile.TemporaryDirectory() as d, _gradio_temp(Path(d)) as up:
        base = Path(d)
        job = _job_folder(base / "web_jobs")
        pdf = job / JOB_FILES[-1]
        demo = app.build_app(Studio(), base / "web_jobs")
        btn = next(c for c in demo.blocks.values() if isinstance(c, gr.DownloadButton))
        fns = [f for f in demo.fns.values() if [c._id for c in f.outputs] == [btn._id]]
        assert len(fns) == 1 and fns[0].fn.__name__ == "make_pdf", [f.name for f in fns]
        click = fns[0]
        assert len(click.inputs) == 1 and isinstance(click.inputs[0], gr.State)
        state = SessionState(demo)
        state[click.inputs[0]._id] = str(job)                               # what analyze left in job_state

        def run():
            return asyncio.run(demo.process_api(block_fn=click, inputs=[None], state=state))["data"][0]

        elsewhere = base / "elsewhere"
        elsewhere.mkdir()
        cwd, tmpdir = os.getcwd(), tempfile.tempdir
        demo.has_launched = True                                            # Gradio checks paths only once launched
        os.chdir(elsewhere)
        tempfile.tempdir = str(elsewhere)
        try:
            with _patched(app, export_pdf=lambda job_dir: str(pdf)):
                out = run()
                refused = False
                with _patched(app, pdf_for_download=lambda job_dir: app.export_pdf(job_dir)):
                    try:
                        run()
                    except InvalidPathError:
                        refused = True
        finally:
            os.chdir(cwd)
            tempfile.tempdir = tmpdir
            demo.has_launched = False
        assert refused, "the stored PDF passed without allowed_paths: the click above proves nothing"
        got = Path(out["path"])
        assert got.parent.parent == Path(get_upload_folder()) == up and re.fullmatch(r"[0-9a-f]{32}", got.parent.name)
        assert out["orig_name"] == got.name == pdf.name and job not in got.parents
        assert got.read_bytes() == pdf.read_bytes() and pdf.exists()
        assert _client(demo).get(out["url"]).status_code == 200

        # no job yet: the dialog asks for an analysis; a failed export: the dialog in Russian, the details in the log
        for job_dir in ("", None):
            try:
                click.fn(job_dir)
            except gr.Error as e:
                assert (e.message, e.title) == ("Сначала проанализируйте видео", app.ERROR_TITLE)
            else:
                raise AssertionError("no error without a job")
        records = []
        handler = logging.Handler()
        handler.emit = records.append
        web_log = logging.getLogger("bs3.web")
        web_log.addHandler(handler)

        def broken(job_dir):
            raise RuntimeError("poppler is missing")

        try:
            with _patched(app, export_pdf=broken):
                click.fn(str(job))
        except gr.Error as e:
            assert (e.message, e.title) == ("Не удалось собрать PDF. Подробности записаны в журнал сервера.",
                                            app.ERROR_TITLE)
        else:
            raise AssertionError("no error for a failed export")
        finally:
            web_log.removeHandler(handler)
        assert [r.getMessage() for r in records] == [f"PDF export failed for {job}"] and records[0].exc_info


def test_launch_without_allowed_paths():
    """The web app and the preview start Gradio without allowed_paths (what the two tests above rely on)."""
    root = Path(app.__file__).resolve().parents[2]
    for f in (root / "bs3" / "web" / "app.py", root / "scripts" / "ui_preview.py"):
        src = f.read_text(encoding="utf-8")
        assert ".launch(" in src and "allowed_paths=" not in src, f.name
