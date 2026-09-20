"""BS 2.0 pipeline: Big Five (ensemble, segmented) + per-segment emotion, voice, face and speech analyses,
explanations and plain-language texts. Produces one result.json per job; the web UI and the PDF only render it."""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

from .norms import TRAIT_KEYS
from .report import build_report, fmt_secs

log = logging.getLogger("bs2.pipeline")


class Studio:
    """Lazily loaded models shared by all requests (one analysis at a time)."""

    def __init__(self, members=("oceanai", "mm"), asr_model="openai/whisper-large-v3-turbo", ollama_model="qwen2.5vl:7b",
                 mm_ckpt=None):
        self.members, self.asr_model, self.ollama_model, self.mm_ckpt = tuple(members), asr_model, ollama_model, mm_ckpt
        self._lock = threading.Lock()
        self._be: Dict[str, object] = {}
        self._an: Dict[str, object] = {}
        self._text_emo = self._voice_emo = self._face_expr = None
        self.stop_event = threading.Event()

    def backend(self, lang: str):
        with self._lock:
            if lang not in self._be:
                from .backend_ensemble import EnsembleBackend, EnsembleConfig
                from .backend_mm import MMConfig
                from .backend_oceanai import BackendConfig
                kw = dict(lang=lang, asr_model=self.asr_model, ollama_model=self.ollama_model)
                if self.mm_ckpt:
                    kw["checkpoint"] = self.mm_ckpt
                self._be[lang] = EnsembleBackend(EnsembleConfig(members=self.members, lang=lang,
                                                                oceanai_cfg=BackendConfig(lang=lang, asr_model=self.asr_model),
                                                                mm_cfg=MMConfig(**kw))).load()
            return self._be[lang]

    def analyzer(self, lang: str):
        from .longvideo import LongVideoAnalyzer
        if lang not in self._an:
            self._an[lang] = LongVideoAnalyzer(self.backend(lang), lang=lang, asr_model=self.asr_model)
        return self._an[lang]

    def mm_backend(self, lang: str):
        return self.backend(lang).backends.get("mm")

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


def run_extra_analyses(studio: Studio, res: dict, lang: str, work_dir: Path, progress: Callable | None = None,
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
    per, tmp = [], Path(tempfile.mkdtemp(prefix="bs2_an_"))
    try:
        for i, t in enumerate(tl, 1):
            if should_stop and should_stop():
                raise AnalysisCancelled("остановлено пользователем")
            if progress:
                progress(0.75 + 0.15 * (i - 1) / len(tl), f"Эмоции, голос, мимика: отрезок {i}/{len(tl)}")
            row = {"segment": t["segment"], "start": t["start"], "end": t["end"]}
            text_en = _to_en(t.get("transcript", ""), lang)
            row["text_en"] = text_en
            try:
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
        out["speech"] = {**whole, "description": describe(whole, lang),
                         "vocabulary": vocabulary(res.get("transcript", ""), top=15, lang=lang)}
    return out


def run_analysis(studio: Studio, work_dir: Path, video_path: str, lang: str = "ru", explain: bool = True,
                 progress: Callable | None = None) -> dict:
    """Whole request: copy the upload, Big Five (segmented ensemble), extra analyses, explanations, plain-language
    texts; writes <job>/result.json and returns it with the job path."""
    from .longvideo import AnalysisCancelled
    from .media import probe_media
    from .narrative import build_narrative
    from .ru_texts import ensure_russian

    def step(frac, desc=None, **kw):
        if progress is not None:
            progress(frac, f"[{fmt_secs(time.time() - t0)}] {desc if desc is not None else kw.get('desc', '')}")

    t0 = time.time()
    studio.stop_event.clear()
    job = work_dir / time.strftime("%Y%m%d_%H%M%S")
    job.mkdir(parents=True, exist_ok=True)
    src = Path(video_path)
    if not src.is_file():
        raise RuntimeError("Файл загрузки не найден. Загрузите видео заново и дождитесь конца загрузки.")
    local = job / ("input" + src.suffix.lower())
    shutil.copy2(src, local)
    step(0.03, "Загрузка моделей (первый запуск до минуты)")
    be = studio.backend(lang)
    an = studio.analyzer(lang)
    step(0.10, "Речь, лицо, голос, описание поведения — Big Five")
    res = an.analyze(local, job / "segments", progress=lambda f, d: step(0.10 + 0.60 * f, d), should_stop=studio.stop_event.is_set)
    primary = res.get("primary")
    if primary:
        from . import pool
        pool.add(local, res["scores"], lang, primary, name=src.name)
    rep = build_report(local, res, backend="ensemble", corpus=be.cfg.corpus, lang=lang, asr_model=studio.asr_model,
                       modalities=tuple(getattr(be.cfg, "members", ())), pool_lang=lang if primary else None, primary=primary)
    rep["variant_scores"] = res.get("variants", {})
    for key in ("duration_sec", "segments", "timeline", "representative_segment", "transcript_en", "chunks"):
        if res.get(key) is not None:
            rep[key] = res[key]
    rep["scores_std_across_segments"] = res.get("scores_std")

    # ---- BS 2.0 analyses
    rep["analyses"] = run_extra_analyses(studio, {**res, "input": str(local)}, lang, job, progress=step,
                                         should_stop=studio.stop_event.is_set)

    # ---- explanations (own model, representative segment)
    expl, frames = None, []
    if studio.stop_event.is_set():
        raise AnalysisCancelled("остановлено пользователем")
    if explain and studio.mm_backend(lang) is not None:
        step(0.92, "Объяснения: вклад модальностей, ключевые кадры, слова")
        mmb = studio.mm_backend(lang)
        if res.get("timeline"):
            seg = res["timeline"][res["representative_segment"] - 1]
            x_video, x_text, x_beh = seg["file"], seg["transcript"], seg.get("behavior_description") or None
        else:
            x_video, x_text, x_beh = local, res.get("transcript", ""), res.get("behavior_description") or None
        try:
            expl = mmb.explain_video(x_video, job / "explain", asr=False, transcript=x_text, behavior=x_beh)
            frames = list(expl.get("frames", {}).get("key_frame_files", []))
        except Exception as e:  # noqa: BLE001
            log.warning("explanation failed: %s", str(e).splitlines()[0][:160])
    # ---- texts for people: Russian versions of the English model texts for any speech language (ru_texts.py)
    step(0.97, "Тексты и перевод")
    ensure_russian(rep, expl)
    if expl:
        (job / "explain" / "explanation.json").write_text(json.dumps(expl, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        rep["media"] = probe_media(local)
        rep["media"]["file_name"] = src.name
    except Exception as e:  # noqa: BLE001
        rep["media"] = {"error": str(e)[:200]}
    rep["original_file_name"] = src.name
    rep["key_frames"] = frames
    rep["timings_sec"]["total_wall"] = round(time.time() - t0, 1)
    rep["narrative"] = build_narrative(rep, expl)
    rep["job_dir"] = str(job)
    (job / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    return rep
