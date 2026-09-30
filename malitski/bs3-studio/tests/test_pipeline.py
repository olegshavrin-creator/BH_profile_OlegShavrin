"""BS Profiler 3.1 pipeline without a GPU (change request 3.1, sections 1, 2, 6): the Studio builds the backend of the
chosen member only, run_analysis(member=...) runs one model, fixes the speech language to Russian, records the model
in result.json (model.selected / selected_title / primary, one member in variant_scores, an mbti section of schema 3
with one source), and asks for explanations only from AMLAI 1.0. The heavy backend modules are replaced by fakes in
sys.modules for the Studio test; run_analysis gets a fake Studio and a canned analyzer result."""
from __future__ import annotations

import contextlib
import inspect
import json
import logging
import re
import sys
import tempfile
import types
from pathlib import Path

import bs3
from bs3 import pipeline
from bs3.norms import TRAIT_KEYS

MEMBERS = ("oceanai", "mm")


@contextlib.contextmanager
def _no_preflight():
    """Stage 22: run_analysis refuses an unreadable upload (media.check_upload) and preflights Ollama for AMLAI 1.0
    before any folder or model. These tests feed byte stubs and a fake Studio, so both preflights are stubbed out here;
    they have their own coverage in test_failures.py."""
    from bs3 import media, ollama
    saved = media.check_upload, ollama.status
    media.check_upload = lambda *a, **k: None
    ollama.status = lambda *a, **k: "ok"
    try:
        yield
    finally:
        media.check_upload, ollama.status = saved


# ------------------------------------------------------------------------------------------------- fakes ---

class _FakeEnsemble:
    """Stands in for backend_ensemble.EnsembleBackend: records the config, builds one inner backend per member."""
    built: list = []

    def __init__(self, cfg):
        self.cfg = cfg
        self.backends = {m: types.SimpleNamespace(cfg=types.SimpleNamespace(corpus=f"corpus-of-{m}")) for m in cfg.members}
        self.loaded = 0
        _FakeEnsemble.built.append(self)

    def load(self):
        self.loaded += 1
        return self


def _fake_modules():
    """bs3.backend_mm and bs3.backend_oceanai without torch / OCEAN-AI: config dataclasses that record their kwargs;
    the real EnsembleConfig (its primary rule is what is tested) with the fake EnsembleBackend."""
    from bs3.backend_ensemble import EnsembleConfig

    def cfg_class(name):
        def __init__(self, **kw):
            self.__dict__.update(kw)
        return type(name, (), {"__init__": __init__})

    mm = types.ModuleType("bs3.backend_mm")
    mm.MMConfig = cfg_class("MMConfig")
    oa = types.ModuleType("bs3.backend_oceanai")
    oa.BackendConfig = cfg_class("BackendConfig")
    ens = types.ModuleType("bs3.backend_ensemble")
    ens.EnsembleConfig, ens.EnsembleBackend = EnsembleConfig, _FakeEnsemble
    return {"bs3.backend_mm": mm, "bs3.backend_oceanai": oa, "bs3.backend_ensemble": ens}


class _Patched:
    def __init__(self, mods: dict):
        self.mods, self.saved = mods, {}

    def __enter__(self):
        for name, mod in self.mods.items():
            self.saved[name] = sys.modules.get(name)
            sys.modules[name] = mod
        return self

    def __exit__(self, *exc):
        for name, old in self.saved.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


class _FakeAnalyzer:
    """LongVideoAnalyzer stand-in: a short clip scored as one segment by the member of its backend."""

    def __init__(self, member: str):
        self.member = member
        self.calls = []

    def analyze(self, video, work_dir, progress=None, should_stop=None):
        self.calls.append(str(video))
        if progress:
            progress(0.5, "отрезок 1/1")
        scores = {k: 0.2 + 0.1 * i for i, k in enumerate(TRAIT_KEYS)}
        if self.member == "mm":
            scores["interview"] = 0.44
        return {"scores": dict(scores), "seconds": 1.0, "variants": {self.member: {k: scores[k] for k in TRAIT_KEYS}},
                "members_used": [self.member], "members_failed": {}, "primary": self.member,
                "primary_used": self.member, "transcript": "", "duration_sec": 15.0, "segments": 1, "timeline": []}


class _FakeStudio:
    """A Studio whose backends never load: remembers which member was asked for, the analyses fail harmlessly (the
    pipeline logs and goes on), explanations are never asked for from OCEAN-AI."""
    asr_model = "fake-asr"

    def __init__(self):
        self.asked = []
        self.explained = []
        self._an = {}

    def backend(self, member):
        self.asked.append(member)
        return types.SimpleNamespace(cfg=types.SimpleNamespace(corpus=f"ensemble({member}), main={member}",
                                                                members=(member,)),
                                     backends={member: types.SimpleNamespace(cfg=types.SimpleNamespace(
                                         corpus="mupta" if member == "oceanai" else "own checkpoints"))})

    def analyzer(self, member):
        return self._an.setdefault(member, _FakeAnalyzer(member))

    def mm_backend(self, member):
        if member != "mm":
            raise AssertionError("explanations were asked for from a model other than AMLAI 1.0")
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


# ------------------------------------------------------------------------------------------------- tests ---

def test_signatures_one_member_russian_only():
    sig = inspect.signature(pipeline.run_analysis)
    assert "member" in sig.parameters and "lang" not in sig.parameters
    assert "should_stop" in sig.parameters                     # stage 21: the stop button passes one Event per session
    assert sig.parameters["member"].default == "mm" == bs3.DEFAULT_MODEL      # a run without a model: AMLAI 1.0
    # everything after the video is passed by keyword (the page and the tests do; a positional call cannot slip a
    # language or a flag into the member)
    params = list(sig.parameters.values())
    first, after = params[:3], params[3:]
    assert [p.name for p in first] == ["studio", "work_dir", "video_path"]
    assert after and all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in after), [(p.name, p.kind) for p in after]
    assert "members" not in inspect.signature(pipeline.Studio.__init__).parameters
    st = pipeline.Studio()
    assert bs3.LANG == "ru" and st.lang == "ru"
    # stage 21: the process-wide stop_event is gone; a per-session Event drives should_stop, and run_lock serialises runs
    assert not hasattr(st, "stop_event") and hasattr(st, "run_lock")
    for bad in ("en", "ensemble", "scene", ""):
        try:
            pipeline.check_member(bad)
        except RuntimeError as e:
            assert "Неизвестная модель" in str(e) and "OCEAN-AI" in str(e) and "AMLAI 1.0" in str(e)
        else:
            raise AssertionError(bad)


def test_studio_builds_one_backend_per_member():
    """One backend of one member at a time: asking for the other member drops the cached one (the GPU holds one
    Big Five model even after the radio is switched in a running server); the same member is not rebuilt."""
    with _Patched(_fake_modules()):
        _FakeEnsemble.built.clear()
        st = pipeline.Studio(asr_model="asr-x", ollama_model="qwen-x", mm_ckpt="/tmp/ckpt.pt")
        mm = st.backend("mm")
        assert mm.cfg.members == ("mm",) and mm.cfg.primary == "mm" and mm.cfg.lang == "ru"
        assert mm.cfg.oceanai_cfg is None and mm.cfg.mm_cfg.lang == "ru"
        assert mm.cfg.mm_cfg.checkpoint == "/tmp/ckpt.pt" and mm.cfg.mm_cfg.ollama_model == "qwen-x"
        assert list(mm.backends) == ["mm"] and mm.loaded == 1
        assert st.mm_backend("mm") is mm.backends["mm"]
        assert st.backend("mm") is mm and list(st._be) == ["mm"]          # cached, not rebuilt
        oa = st.backend("oceanai")
        assert oa is not mm and oa.cfg.members == ("oceanai",) and oa.cfg.primary == "oceanai"
        assert oa.cfg.mm_cfg is None and oa.cfg.oceanai_cfg.lang == "ru" and oa.cfg.oceanai_cfg.asr_model == "asr-x"
        assert list(oa.backends) == ["oceanai"]
        assert list(st._be) == ["oceanai"] and "mm" not in st._an           # the other member is unloaded
        assert st.mm_backend("oceanai") is None                            # explanations exist for AMLAI 1.0 only
        assert st.backend("oceanai") is oa and len(_FakeEnsemble.built) == 2
        mm2 = st.backend("mm")                                             # back again: built anew, OCEAN-AI dropped
        assert mm2 is not mm and list(st._be) == ["mm"] and len(_FakeEnsemble.built) == 3
        assert all(len(b.cfg.members) == 1 for b in _FakeEnsemble.built)
        try:
            st.backend("ensemble")
        except RuntimeError:
            pass
        else:
            raise AssertionError("an unknown member was accepted")


def test_studio_hands_whisper_over_between_analyzers():
    """The segment analyzer of the next member takes the Whisper pipeline of the dropped one instead of loading a
    second copy; at most one analyzer is kept."""
    with _Patched(_fake_modules()):
        st = pipeline.Studio()
        an_mm = st.analyzer("mm")
        assert list(st._an) == ["mm"] and an_mm.backend is st._be["mm"] and an_mm.lang == "ru"
        an_mm._asr = whisper = object()                                     # «loaded» Whisper of this analyzer
        an_oa = st.analyzer("oceanai")
        assert list(st._an) == ["oceanai"] and list(st._be) == ["oceanai"]
        assert an_oa is not an_mm and an_oa._asr is whisper and st._asr_shared is None
        assert st.analyzer("oceanai") is an_oa


def test_studio_passes_models_dir_to_oceanai():
    """`bs3 web --models-dir` reaches the OCEAN-AI config; without it BackendConfig keeps its own default."""
    with _Patched(_fake_modules()):
        oa = pipeline.Studio(models_dir="/tmp/oceanai-models").backend("oceanai")
        assert oa.cfg.oceanai_cfg.models_dir == "/tmp/oceanai-models"
        oa = pipeline.Studio().backend("oceanai")
        assert not hasattr(oa.cfg.oceanai_cfg, "models_dir")


def test_ensemble_config_one_member_only():
    """The wrapper holds exactly one model; its primary is that model (no "auto" rule, no mean of several)."""
    import dataclasses
    from bs3.backend_ensemble import EnsembleConfig
    for m in MEMBERS:
        assert EnsembleConfig(members=(m,)).primary == m
        assert EnsembleConfig(members=(m,), lang="ru", primary=m).primary == m
        assert EnsembleConfig(members=[m]).members == (m,)
    for kw in ({"members": ("mm", "oceanai")}, {"members": ("oceanai", "mm")}, {"members": ()},
               {"members": ("mm",), "primary": "oceanai"}, {"members": ("oceanai",), "primary": "mean"}):
        try:
            EnsembleConfig(**kw)
        except ValueError:
            pass
        else:
            raise AssertionError(kw)
    assert {f.name for f in dataclasses.fields(EnsembleConfig)} == {"members", "oceanai_cfg", "mm_cfg", "lang", "primary"}


def _member_modules(calls: list, fail: str | None = None):
    """bs3.backend_mm / bs3.backend_oceanai with a member that returns canned numbers (numpy floats, as the models
    give them) or raises `fail`."""
    import numpy as np

    class _Member:
        def __init__(self, cfg=None):
            self.cfg, self.loaded = cfg, 0

        def load(self):
            self.loaded += 1
            return self

        def _result(self, video, **kw):
            calls.append((type(self).__name__, str(video), kw))
            if fail:
                raise RuntimeError(fail)
            scores = {k: np.float32(0.1 + 0.123456789 * i) for i, k in enumerate(TRAIT_KEYS)}
            return {"scores": scores, "transcript": "", "seconds": 0.5}

    class MMBackend(_Member):
        def predict_video(self, video, asr=True, transcript=None, behavior=None):
            r = self._result(video, asr=asr, transcript=transcript, behavior=behavior)
            r["scores"]["interview"] = np.float32(0.4321)
            r.update(transcript="речь", behavior_description="smiles", timings={"behavior": 1.5})
            return r

    class OceanAIBackend(_Member):
        def predict_video(self, video, asr=True, transcript=None):
            return self._result(video, asr=asr, transcript=transcript)

    mm = types.ModuleType("bs3.backend_mm")
    mm.MMBackend, mm.MMConfig = MMBackend, object
    oa = types.ModuleType("bs3.backend_oceanai")
    oa.OceanAIBackend, oa.BackendConfig = OceanAIBackend, object
    return {"bs3.backend_mm": mm, "bs3.backend_oceanai": oa}


def test_ensemble_passes_the_one_member_through():
    """The member's scores reach the output unchanged (the same floats, no mean recomputed), with exactly the keys
    the segment analyzer and result.json read; a failure keeps the message the page maps."""
    from bs3.backend_ensemble import EnsembleBackend, EnsembleConfig
    base = {"scores", "seconds", "variants", "members_used", "members_failed", "primary", "primary_used", "transcript"}
    calls: list = []
    with _Patched(_member_modules(calls)):
        for m in MEMBERS:
            be = EnsembleBackend(EnsembleConfig(members=(m,), mm_cfg=object(), oceanai_cfg=object())).load()
            assert list(be.backends) == [m] and be.backends[m].loaded == 1
            out = be.predict_video("clip.mp4", asr=False, transcript="готовый текст", behavior="desc")
            member = out["variants"][m]
            assert set(out) == (base | {"behavior_description", "timings"} if m == "mm" else base), sorted(out)
            for k in TRAIT_KEYS:
                assert type(out["scores"][k]) is float and out["scores"][k] == float(member[k]), k
            assert out["variants"] == {m: member} and out["members_used"] == [m] and out["members_failed"] == {}
            assert out["primary"] == out["primary_used"] == m
            if m == "mm":
                assert out["scores"]["interview"] == member["interview"]
                assert out["transcript"] == "речь" and out["behavior_description"] == "smiles"
                assert out["timings"] == {"mm_behavior": 1.5}
                assert calls[-1][2] == {"asr": False, "transcript": "готовый текст", "behavior": "desc"}
            else:
                assert "interview" not in out["scores"] and "interview" not in member
                assert out["transcript"] == "готовый текст"         # the member heard nothing: the given text
                assert calls[-1][2] == {"asr": False, "transcript": "готовый текст"}
    log = logging.getLogger("bs.ensemble")
    level = log.level
    log.setLevel(logging.ERROR)                           # the expected warning stays out of the test output
    try:
        with _Patched(_member_modules(calls, fail="no face found\nsecond line")):
            be = EnsembleBackend(EnsembleConfig(members=("oceanai",), oceanai_cfg=object()))
            try:
                be.predict_video("clip.mp4")
            except RuntimeError as e:
                assert str(e) == "all ensemble members failed: oceanai: no face found", str(e)
            else:
                raise AssertionError("a failed member did not raise")
    finally:
        log.setLevel(level)


def _run(member: str, explain: bool, tmp: Path):
    studio = _FakeStudio()
    tmp.mkdir(parents=True, exist_ok=True)
    video = tmp / "clip.mp4"
    video.write_bytes(b"\x00" * 2048)
    work = tmp / "jobs" / member
    log = logging.getLogger("bs3.pipeline")
    old_level = log.level
    log.setLevel(logging.CRITICAL)            # the fakes fail on purpose; the pipeline logs and goes on
    try:
        with _no_preflight():
            rep_ = pipeline.run_analysis(studio, work, str(video), member=member, explain=explain,
                                         progress=lambda f, d: None)
    finally:
        log.setLevel(old_level)
    return studio, rep_, work


def test_run_analysis_one_member_each():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        for member in MEMBERS:
            studio, r, work = _run(member, explain=True, tmp=tmp)
            assert set(studio.asked) == {member}, (member, studio.asked)
            m = r["model"]
            assert m["selected"] == member and m["primary"] == member
            assert m["selected_title"] == bs3.MODEL_TITLES[member] and m["lang"] == "ru"
            assert m["backend"] == member and m["product"] == "BS Profiler 3.1" and m["version"] == "3.1.0a1"
            assert r["modalities_used"] == list(bs3.MODALITIES[member])     # what the model looked at, not its name
            assert set(r["variant_scores"]) == {member}
            # nothing of 2.0 that the tab «Данные» would have to leave out: no stored summary, no percentiles
            assert "narrative" not in r
            if member == "mm":
                assert set(r["interview"]) == {"score", "name_ru", "disclaimer"}
            for k in TRAIT_KEYS:
                assert abs(r["traits"][k]["score"] - r["variant_scores"][member][k]) < 1e-9
                assert "percentile" not in r["traits"][k]                   # Russian speech: the score only
            mb = r["mbti"]
            assert mb["schema_version"] == 3 and mb["model"] == member and "second" not in mb and "agreement" not in mb
            assert mb["source"] == {"oceanai": "ocean_ai", "mm": "own_model"}[member]
            assert mb["model_title"] == bs3.MODEL_TITLES[member] and mb["primary_missing"] is False
            assert ("interview" in r) == (member == "mm")
            # explanations: asked for from AMLAI 1.0 only (they failed harmlessly here)
            assert (len(studio.explained) == 1) == (member == "mm")
            # written where the page reads it
            job = Path(r["job_dir"])
            assert job.parent == work and (job / "result.json").exists()
            assert re.fullmatch(r"\d{8}_\d{6}_[0-9a-f]{8}", job.name), job.name     # time stamp + random suffix
            saved = json.loads((job / "result.json").read_text(encoding="utf-8"))
            # written through jobfiles.write_json byte for byte as before (indent 2, UTF-8), no temporary file left
            assert (job / "result.json").read_bytes() == json.dumps(r, ensure_ascii=False, indent=2).encode("utf-8")
            assert not [p.name for p in job.iterdir() if p.name.endswith(".tmp")]
            assert saved["model"]["selected"] == member and set(saved["variant_scores"]) == {member}
            assert saved["mbti"]["schema_version"] == 3
            text = json.dumps(saved, ensure_ascii=False)
            for bad in ("второе мнение", "Второе мнение", "своя модель", "среднее двух систем", "MM-PSYCHE"):
                assert bad not in text.replace("по рецепту MM-PSYCHE", ""), (member, bad)
            # the view of a fresh 3.1 job is the job itself, and the tab «Данные» shows the file whole (no note)
            from bs3.scores import clean_view, data_json
            v = clean_view(saved)
            assert v["view_meta"]["main_system"] == member and v["view_meta"]["primary_missing"] is False
            assert ("interview" in v) == (member == "mm")
            shown, trimmed = data_json(saved)
            assert shown == saved and trimmed is False


def test_run_analysis_same_second_two_folders():
    """Two analyses started in the same second (the time stamp fixed) get two job folders: the random suffix keeps
    them apart, and neither run writes into the folder of the other."""
    real = pipeline.time
    fixed = types.SimpleNamespace(**vars(real))
    fixed.strftime = lambda fmt, *a: "20000101_000000"
    pipeline.time = fixed
    try:
        with tempfile.TemporaryDirectory() as d:
            _, r1, work = _run("oceanai", explain=False, tmp=Path(d))
            _, r2, _ = _run("oceanai", explain=False, tmp=Path(d))
            j1, j2 = Path(r1["job_dir"]), Path(r2["job_dir"])
            assert j1 != j2 and sorted(p.name for p in work.iterdir()) == sorted((j1.name, j2.name))
            for job in (j1, j2):
                assert re.fullmatch(r"20000101_000000_[0-9a-f]{8}", job.name), job.name
                assert (job / "input.mp4").is_file()
                assert json.loads((job / "result.json").read_text(encoding="utf-8"))["job_dir"] == str(job)
    finally:
        pipeline.time = real


def test_run_analysis_oceanai_skips_explanations_even_when_asked():
    with tempfile.TemporaryDirectory() as d:
        studio, r, _ = _run("oceanai", explain=True, tmp=Path(d))
        assert studio.explained == [] and r["key_frames"] == []
        studio, r, _ = _run("mm", explain=False, tmp=Path(d) / "b")
        assert studio.explained == [] and r["key_frames"] == []


def test_should_stop_cancels_the_run():
    """Stage 21: a should_stop that returns True stops the run with AnalysisCancelled before the analysis runs (the
    check after studio.analyzer()). The backend and the analyzer are reached, but an.analyze is never called and no
    explanation is asked for."""
    from bs3.errors import AnalysisCancelled
    with tempfile.TemporaryDirectory() as d:
        studio = _FakeStudio()
        tmp = Path(d)
        video = tmp / "clip.mp4"
        video.write_bytes(b"\x00" * 2048)
        log = logging.getLogger("bs3.pipeline")
        old_level = log.level
        log.setLevel(logging.CRITICAL)
        try:
            try:
                with _no_preflight():
                    pipeline.run_analysis(studio, tmp / "jobs", str(video), member="mm", explain=True,
                                          should_stop=lambda: True)
            except AnalysisCancelled:
                pass
            else:
                raise AssertionError("should_stop=True did not cancel the run")
        finally:
            log.setLevel(old_level)
        assert studio.asked == ["mm"] and studio.explained == []
        assert studio.analyzer("mm").calls == []          # the segment analysis never started


def test_no_pool_of_processed_videos():
    """The pool of processed videos is gone (owner decision of 2026-09-27): it was collected and never read. Neither
    an analysis from the page nor `bs3 infer` registers the scores of a video anywhere besides the job and the report."""
    import importlib.util
    assert importlib.util.find_spec("bs3.pool") is None
    root = Path(pipeline.__file__).resolve().parent
    for p in sorted([*root.rglob("*.py"), *(root.parent / "scripts").glob("*.py")]):
        text = p.read_text(encoding="utf-8")
        for bad in ("BS3_POOL_DIR", "POOL_DIR", "bs3_data/pool", "pool.add", "pool_add", "from .pool", "import pool"):
            assert bad not in text, (str(p.relative_to(root.parent)), bad)
