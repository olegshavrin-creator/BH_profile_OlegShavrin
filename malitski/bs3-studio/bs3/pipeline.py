"""BS Profiler 3.1 pipeline: Big Five by ONE model chosen for the analysis (OCEAN-AI or AMLAI 1.0, segmented) +
per-segment emotion, voice, face and speech analyses, explanations (AMLAI 1.0 only) and plain-language texts. The
speech is always Russian (bs3.LANG). Produces one result.json per job; the web UI and the PDF only render it."""
from __future__ import annotations

import logging
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

from . import DEFAULT_MODEL, LANG, MODALITIES, MODEL_TITLES, jobfiles, ollama, settings
from .report import build_report
from .textfmt import fmt_secs

log = logging.getLogger("bs3.pipeline")


def check_member(member: str) -> str:
    """The internal key of a model the page offers ("oceanai" | "mm"); anything else is a Russian error for the page."""
    if member not in MODEL_TITLES:
        raise RuntimeError(f"Неизвестная модель «{member}». Выберите {' или '.join(MODEL_TITLES.values())}.")
    return member


class Studio:
    """Lazily loaded models shared by all requests (one analysis at a time).

    3.1: one Big Five model per analysis. `backend(member)` builds and caches the backend of that member only
    (`EnsembleConfig(members=(member,))`), and when the page asks for the other member the cached one is dropped
    first (`_evict`), so the GPU holds one Big Five model at a time even after the radio is switched in a running
    server; the other model is never run for a second opinion. The Whisper pipeline of the segment analyzer
    (LongVideoAnalyzer, one per member) is handed over to the analyzer of the next member instead of being loaded
    again. The speech language is fixed to Russian (bs3.LANG). The emotion, voice and face models are shared by both
    members. `models_dir`: the OCEAN-AI weights cache (None = the default of BackendConfig, ~/bs/models).
    `ollama_model` becomes the Ollama model of the whole process (ollama.configure): the translations use it too."""

    def __init__(self, asr_model=settings.ASR_MODEL, ollama_model=settings.OLLAMA_MODEL, mm_ckpt=None,
                 models_dir=None):
        self.asr_model, self.ollama_model, self.mm_ckpt = asr_model, ollama_model, mm_ckpt
        ollama.configure(model=ollama_model)
        self.models_dir = models_dir
        self.lang = LANG
        self._lock = threading.Lock()
        self._be: Dict[str, object] = {}          # member -> loaded backend of that member alone (at most one)
        self._an: Dict[str, object] = {}          # member -> LongVideoAnalyzer over that backend (at most one)
        self._asr_shared = None                   # the Whisper pipeline of a dropped analyzer, reused by the next one
        self._text_emo = self._voice_emo = self._face_expr = None
        # one Big Five analysis at a time: web/app.py holds this while a run works, so a run started after «Остановить»
        # waits for the stopped one to reach its next should_stop check and let go, instead of overlapping with it
        self.run_lock = threading.Lock()

    def _evict(self, keep: str) -> None:
        """Drop the backend and the analyzer of every member other than `keep` and give their GPU memory back."""
        others = [m for m in list(self._be) + list(self._an) if m != keep]
        if not others:
            return
        for m in set(others):
            an = self._an.pop(m, None)
            if an is not None and getattr(an, "_asr", None) is not None:
                self._asr_shared = an._asr
            self._be.pop(m, None)
        log.info("model %s unloaded: the GPU holds one Big Five model (%s)", ", ".join(sorted(set(others))), keep)
        try:
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 — torch is not needed for the bookkeeping itself
            pass

    def backend(self, member: str = DEFAULT_MODEL):
        member = check_member(member)
        with self._lock:
            if member not in self._be:
                self._evict(member)
                from .backend_ensemble import EnsembleBackend, EnsembleConfig
                from .backend_mm import MMConfig
                from .backend_oceanai import BackendConfig
                kw = dict(lang=self.lang, asr_model=self.asr_model, ollama_model=self.ollama_model)
                if self.mm_ckpt:
                    kw["checkpoint"] = self.mm_ckpt
                oa_kw = dict(lang=self.lang, asr_model=self.asr_model)
                if self.models_dir:
                    oa_kw["models_dir"] = self.models_dir
                cfg = EnsembleConfig(members=(member,), lang=self.lang, primary=member,
                                     oceanai_cfg=BackendConfig(**oa_kw) if member == "oceanai" else None,
                                     mm_cfg=MMConfig(**kw) if member == "mm" else None)
                self._be[member] = EnsembleBackend(cfg).load()
            return self._be[member]

    def analyzer(self, member: str = DEFAULT_MODEL):
        from .longvideo import LongVideoAnalyzer
        member = check_member(member)
        be = self.backend(member)                 # evicts the other member's backend and analyzer first
        if member not in self._an:
            an = LongVideoAnalyzer(be, lang=self.lang, asr_model=self.asr_model)
            if self._asr_shared is not None:      # the Whisper of the dropped analyzer, not a second copy
                an._asr, self._asr_shared = self._asr_shared, None
            self._an[member] = an
        return self._an[member]

    def mm_backend(self, member: str = DEFAULT_MODEL):
        """The own model (AMLAI 1.0) when it is the chosen member, else None: explanations exist for it only."""
        return self.backend(member).backends.get("mm") if member == "mm" else None

    @property
    def text_emotion(self):
        if self._text_emo is None:
            from .analyses.emotions_text import TextEmotion
            self._text_emo = TextEmotion()
        return self._text_emo

    @property
    def voice_emotion(self):
        if self._voice_emo is None:
            from .analyses.emotions_voice import VoiceEmotion
            self._voice_emo = VoiceEmotion()
        return self._voice_emo

    @property
    def face_expression(self):
        if self._face_expr is None:
            from .analyses.face_expr import FaceExpression
            self._face_expr = FaceExpression()
        return self._face_expr


def _wav16k(video: str | Path, out: Path) -> Path:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
                    "-acodec", "pcm_s16le", str(out)], check=True)
    return out


def _to_en(text: str, lang: str) -> str:
    if lang == "en" or not text.strip():
        return text
    import re
    from .translate import translate_sentences
    sents = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s] or [text.strip()]
    return " ".join(translate_sentences(sents, lang, "en"))


def _weighted(rows: List[dict], w: List[float], keys: List[str]) -> Dict[str, float]:
    w = np.array(w, dtype=float); w = w / max(w.sum(), 1e-9)
    return {k: round(float(sum(r.get(k, 0.0) * wi for r, wi in zip(rows, w))), 4) for k in keys}


def run_extra_analyses(studio: Studio, res: dict, lang: str, progress: Callable | None = None,
                       should_stop: Callable | None = None) -> dict:
    """Per-segment text emotions, voice dimensions, facial expressions and speech statistics. Uses the segments and
    Whisper chunks already produced by the Big Five pass; single short clips are treated as one segment."""
    from .analyses.emotions_text import EMOTION_ORDER, dominant
    from .analyses.emotions_voice import DIMS
    from .analyses.face_expr import EXPR_ORDER
    from .analyses.speech_stats import describe, stats_for, vocabulary
    from .longvideo import AnalysisCancelled

    tl = [t for t in (res.get("timeline") or []) if t.get("scores") and t.get("file")]
    if not tl:            # short clip analysed as a whole
        tl = [{"segment": 1, "start": 0.0, "end": float(res.get("duration_sec") or 0.0), "file": res.get("input"),
               "transcript": res.get("transcript", "")}]
    chunks = [tuple(c) for c in (res.get("chunks") or [])]
    if not chunks and res.get("transcript"):
        chunks = [(0.0, float(res.get("duration_sec") or 15.0), res["transcript"])]
    per, tmp = [], Path(tempfile.mkdtemp(prefix="bs3_an_"))
    try:
        for i, t in enumerate(tl, 1):
            if should_stop and should_stop():
                raise AnalysisCancelled("остановлено пользователем")
            if progress:
                progress(0.75 + 0.15 * (i - 1) / len(tl), f"Эмоции, голос, мимика: отрезок {i}/{len(tl)}")
            row = {"segment": t["segment"], "start": t["start"], "end": t["end"]}
            try:
                # the translation (Marian) is inside the try too (FP6b): a translation failure only drops this
                # segment's text emotions, it does not kill an analysis that is otherwise finished
                text_en = _to_en(t.get("transcript", ""), lang)
                row["text_en"] = text_en
                row["emotions_text"] = studio.text_emotion(text_en)
            except Exception as e:  # noqa: BLE001
                log.warning("text emotion failed on segment %s: %s", t["segment"], str(e)[:100])
            try:
                row["voice"] = studio.voice_emotion.from_file(str(_wav16k(t["file"], tmp / f"seg{i}.wav")))
            except Exception as e:  # noqa: BLE001
                log.warning("voice emotion failed on segment %s: %s", t["segment"], str(e)[:100])
            try:
                row["face"] = studio.face_expression.from_video(str(t["file"]))
            except Exception as e:  # noqa: BLE001
                log.warning("face expression failed on segment %s: %s", t["segment"], str(e)[:100])
            row["speech"] = stats_for(chunks, t["start"], t["end"], lang=lang)
            per.append(row)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    w = [max(0.1, r["end"] - r["start"]) for r in per]
    out = {"per_segment": per}
    te = [r["emotions_text"] for r in per if r.get("emotions_text")]
    if te:
        mean = _weighted(te, [wi for r, wi in zip(per, w) if r.get("emotions_text")], EMOTION_ORDER)
        out["emotions_text"] = {"mean": mean, "dominant": dominant(mean),
                                "dominant_per_segment": [dominant(r["emotions_text"]) if r.get("emotions_text") else None for r in per]}
    vo = [r["voice"] for r in per if r.get("voice") and not any(np.isnan(v) for v in r["voice"].values())]
    if vo:
        out["voice"] = {"mean": _weighted(vo, [wi for r, wi in zip(per, w) if r.get("voice") and not any(np.isnan(v) for v in r["voice"].values())], DIMS),
                        "std": {d: round(float(np.std([r[d] for r in vo])), 4) for d in DIMS}}
    fa = [r["face"] for r in per if r.get("face") and r["face"].get("expressions")]
    if fa:
        mean = _weighted([f["expressions"] for f in fa], [wi for r, wi in zip(per, w) if r.get("face") and r["face"].get("expressions")], EXPR_ORDER)
        out["face"] = {"mean": mean, "dominant": max(mean.items(), key=lambda kv: kv[1])[0],
                       "face_share": round(float(np.mean([f.get("face_share", 1.0) for f in fa])), 3),
                       "head_motion": round(float(np.mean([f["head_motion"] for f in fa if f.get("head_motion") is not None])), 3)
                       if any(f.get("head_motion") is not None for f in fa) else None}
    if chunks:
        whole = stats_for(chunks, 0.0, float(res.get("duration_sec") or (tl[-1]["end"] if tl else 0.0)), lang=lang)
        out["speech"] = {**whole, "description": describe(whole),
                         "vocabulary": vocabulary(res.get("transcript", ""), top=15)}
    return out


def run_analysis(studio: Studio, work_dir: Path, video_path: str, *, member: str = DEFAULT_MODEL, explain: bool = True,
                 progress: Callable | None = None, should_stop: Callable[[], bool] | None = None) -> dict:
    """Whole request: copy the upload, Big Five by the chosen model only (`member`: "oceanai" | "mm", segmented),
    extra analyses, explanations (AMLAI 1.0 only), plain-language texts; writes <job>/result.json and returns it with
    the job path. The speech language is Russian (bs3.LANG). result.json records the model as `model.selected` /
    `model.selected_title` / `model.primary`; `variant_scores` holds that member only, `modalities_used` what that
    model looks at (bs3.MODALITIES). The plain-language summary of 2.0 (`narrative`) is not written any more: 3.x
    shows «Как получены оценки» (narrative.method_notes), built on display.

    `should_stop`: checked between steps and passed on to the segment loop; when it returns True the run raises
    AnalysisCancelled (the web «Остановить обработку» button, one Event per session)."""
    from . import errors
    from .longvideo import AnalysisCancelled
    from .media import check_upload, probe_media
    from .ru_texts import ensure_russian

    def step(frac, desc=None, **kw):
        if progress is not None:
            progress(frac, f"[{fmt_secs(time.time() - t0)}] {desc if desc is not None else kw.get('desc', '')}")

    member = check_member(member)
    lang = LANG
    title = MODEL_TITLES[member]
    t0 = time.time()
    # refuse before any folder or model is made, so a bad upload leaves nothing behind and the person is told why at once
    src = Path(video_path)
    if not src.is_file():
        raise RuntimeError("Файл загрузки не найден. Загрузите видео заново и дождитесь конца загрузки.")
    check_upload(src, work_dir)          # unreadable / no sound / < 2 s / no space -> UserFacingError
    if member == "mm":                   # AMLAI 1.0 needs the Ollama vision model: check it now, not after loading
        mdl = ollama.model()
        st = ollama.status(mdl)
        if st == "down":
            raise errors.UserFacingError(errors.OLLAMA)
        if st == "no_model":
            raise errors.UserFacingError(f"В Ollama нет модели {mdl}, без которой AMLAI 1.0 не работает. "
                                         "Установите её или выберите модель OCEAN-AI.")
    # the start time and a random suffix: the name of a new job cannot be guessed from the time of the upload, and two
    # analyses started in the same second get two folders (older jobs keep their names without the suffix)
    job = work_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}"
    job.mkdir(parents=True)
    try:
        local = job / ("input" + src.suffix.lower())
        shutil.copy2(src, local)
        step(0.03, f"Загрузка модели {title} (первый запуск до минуты)")
        be = studio.backend(member)
        an = studio.analyzer(member)
        if should_stop and should_stop():
            raise AnalysisCancelled("остановлено пользователем")
        step(0.10, f"Речь, лицо, голос{', описание поведения' if member == 'mm' else ''} — Big Five ({title})")
        res = an.analyze(local, job / jobfiles.SEGMENTS_DIR, progress=lambda f, d: step(0.10 + 0.60 * f, d),
                         should_stop=should_stop)
        # the corpus of the member itself (MuPTA / the own model's checkpoints), not the ensemble wrapper's descriptor
        inner = getattr(be, "backends", {}).get(member)
        corpus = getattr(getattr(inner, "cfg", None), "corpus", None) or be.cfg.corpus
        rep = build_report(local, res, backend=member, corpus=corpus, lang=lang, asr_model=studio.asr_model,
                           modalities=MODALITIES[member], primary=member, selected=member)
        rep["variant_scores"] = {m: v for m, v in (res.get("variants") or {}).items() if m == member}
        for key in ("duration_sec", "segments", "timeline", "representative_segment", "chunks"):
            if res.get(key) is not None:
                rep[key] = res[key]
        rep["scores_std_across_segments"] = res.get("scores_std")

        # ---- BS Profiler 3.x analyses
        rep["analyses"] = run_extra_analyses(studio, {**res, "input": str(local)}, lang, progress=step,
                                             should_stop=should_stop)

        # ---- explanations (AMLAI 1.0 only, representative segment)
        expl, frames = None, []
        if should_stop and should_stop():
            raise AnalysisCancelled("остановлено пользователем")
        if explain and member == "mm" and studio.mm_backend(member) is not None:
            step(0.92, "Объяснения: вклад модальностей, ключевые кадры и подписи к ним, слова")
            mmb = studio.mm_backend(member)
            if res.get("timeline"):
                seg = res["timeline"][res["representative_segment"] - 1]
                x_video, x_text, x_beh = seg["file"], seg["transcript"], seg.get("behavior_description") or None
            else:
                x_video, x_text, x_beh = local, res.get("transcript", ""), res.get("behavior_description") or None
            # the captions under the key frames need the facial-expression model of the report (already loaded by the
            # per-segment analyses above, so no second model goes on the GPU) and one short vision-model request per
            # key frame; both are computed once and stored in explanation.json. A caption never breaks the
            # explanation: when the expression model is not there, the frames keep their phrase alone.
            try:
                expr_fn = studio.face_expression.on_crops
            except Exception as e:  # noqa: BLE001
                log.warning("face expression model unavailable for the key-frame captions: %s", str(e)[:120])
                expr_fn = None
            try:
                expl = mmb.explain_video(x_video, job / jobfiles.EXPLAIN_DIR, asr=False, transcript=x_text,
                                         behavior=x_beh, expression_fn=expr_fn)
                frames = list(expl.get("frames", {}).get("key_frame_files", []))
            except Exception as e:  # noqa: BLE001
                log.warning("explanation failed: %s", str(e).splitlines()[0][:160])
        # ---- texts for people: Russian versions of the English model texts for any speech language (ru_texts.py)
        if should_stop and should_stop():
            raise AnalysisCancelled("остановлено пользователем")
        step(0.97, "Тексты и перевод")
        ensure_russian(rep, expl)
        if expl:
            jobfiles.write_json(jobfiles.explanation_path(job), expl)
        try:
            rep["media"] = probe_media(local)
            rep["media"]["file_name"] = src.name
        except Exception as e:  # noqa: BLE001
            rep["media"] = {"error": str(e)[:200]}
        rep["original_file_name"] = src.name
        rep["key_frames"] = frames
        rep["timings_sec"]["total_wall"] = round(time.time() - t0, 1)
        rep["job_dir"] = str(job)
        # ---- MBTI section (design 7.2; schema 3: one model): computed once from the clean scores; a failure is logged,
        # the job goes on
        try:
            from .mbti import build_section
            from .scores import clean_view
            mb = build_section(clean_view(rep))
            if mb is not None:
                rep["mbti"] = mb
        except Exception as e:  # noqa: BLE001
            log.warning("mbti section failed: %s", str(e).splitlines()[0][:160] if str(e) else type(e).__name__)
        jobfiles.write_json(job / jobfiles.RESULT, rep)
        # a finished job keeps its source video (input.*) and its result; the segment clips were cut from input.* and
        # nothing reads them once result.json exists, so they are removed to keep the folder small (BS3_KEEP_SEGMENTS=1
        # keeps them for debugging). result.json is already written, so a cleanup failure must never fail the analysis:
        # it is logged and swallowed here. The outer except removes the whole folder only when result.json is absent,
        # so it would not clean up after a failure here — hence this inner guard.
        if not settings.KEEP_SEGMENTS:
            seg_dir = job / jobfiles.SEGMENTS_DIR
            try:
                if seg_dir.is_dir():
                    shutil.rmtree(seg_dir)
            except Exception as e:  # noqa: BLE001
                log.warning("could not remove the segments of %s: %s", job.name, str(e)[:120])
        return rep
    except BaseException:
        # a run that did not write result.json leaves no folder behind (unless BS3_KEEP_FAILED_JOBS); only the folder
        # this call just made is touched, never an existing job
        if not (job / jobfiles.RESULT).exists() and not settings.KEEP_FAILED_JOBS:
            log.warning("removing the folder of a failed job: %s", job.name)
            shutil.rmtree(job, ignore_errors=True)
        raise
