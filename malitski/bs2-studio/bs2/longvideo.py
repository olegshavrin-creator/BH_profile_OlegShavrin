"""Long videos: split into ~20-second segments (the models were trained on 15-second clips), analyse every
segment in full (face, voice, its own transcript span from Whisper timestamps, its own behaviour description),
then aggregate: duration-weighted mean, spread across segments and a per-segment timeline.
Nothing is truncated: the transcript is taken from the whole recording, every segment is scored.
"""
from __future__ import annotations

import inspect
import json
import logging
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from .norms import TRAIT_KEYS
from .report import fmt_secs, seg_label


class AnalysisCancelled(RuntimeError):
    """Raised between steps when the caller asked to stop (see LongVideoAnalyzer.analyze(should_stop=...))."""

log = logging.getLogger("bs.long")


def video_duration(path: str | Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def plan_segments(duration: float, seg_len: float = 20.0, min_seg: float = 6.0) -> list[tuple[float, float]]:
    """[(start, end)] covering the whole clip; a short tail is merged into the previous segment."""
    segs, t = [], 0.0
    while t < duration - 1e-3:
        segs.append((t, min(t + seg_len, duration)))
        t += seg_len
    if len(segs) > 1 and segs[-1][1] - segs[-1][0] < min_seg:
        segs[-2] = (segs[-2][0], segs[-1][1])
        segs.pop()
    return segs


def cut_segment(video: Path, start: float, end: float, out: Path):
    """Accurate cut with a fast re-encode (stream copy would snap to keyframes)."""
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(video),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "aac", "-ac", "1", "-ar", "48000",
                    "-movflags", "+faststart", str(out)], check=True)


class LongVideoAnalyzer:
    def __init__(self, backend, lang: str = "en", seg_len: float = 20.0, single_max: float = 30.0,
                 asr_model: str = "openai/whisper-large-v3-turbo", device: str | None = None):
        self.backend, self.lang, self.seg_len, self.single_max = backend, lang, seg_len, single_max
        self.asr_model = asr_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._asr = None

    # ------------------------------------------------------------------ ASR once, with timestamps
    def transcribe(self, video: Path, tmpdir: Path) -> tuple[str, list[tuple[float, float, str]]]:
        if self._asr is None:
            from transformers import pipeline
            dev = 0 if self.device.startswith("cuda") else -1
            self._asr = pipeline("automatic-speech-recognition", model=self.asr_model,
                                 torch_dtype=torch.float16 if dev == 0 else torch.float32, device=dev)
        wav = tmpdir / "audio16k.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
                        "-acodec", "pcm_s16le", str(wav)], check=True)
        res = self._asr(str(wav), chunk_length_s=30, batch_size=8, return_timestamps=True,
                        generate_kwargs={"language": self.lang, "task": "transcribe"})
        chunks = []
        for c in res.get("chunks", []):
            s, e = c.get("timestamp", (None, None))
            if s is None:
                continue
            chunks.append((float(s), float(e) if e is not None else float(s) + 5.0, (c.get("text") or "").strip()))
        return (res.get("text") or "").strip(), chunks

    @staticmethod
    def text_for(chunks, start, end) -> str:
        parts = []
        for s, e, txt in chunks:
            mid = (s + e) / 2
            if start <= mid < end or (s < end and e > start and (min(e, end) - max(s, start)) > 0.5 * (e - s)):
                parts.append(txt)
        # de-duplicate consecutive repeats
        out = []
        for p in parts:
            if not out or out[-1] != p:
                out.append(p)
        return " ".join(out).strip()

    # ------------------------------------------------------------------ main
    def analyze(self, video: str | Path, work_dir: str | Path, progress=None, should_stop=None) -> dict:
        """`should_stop`: callable checked between steps; when it returns True the analysis raises
        AnalysisCancelled (cooperative stop for the web «Остановить обработку» button)."""
        video, work_dir = Path(video), Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        dur = video_duration(video)

        def check():
            if should_stop is not None and should_stop():
                raise AnalysisCancelled("обработка остановлена пользователем")

        check()
        if dur <= self.single_max:
            res = self.backend.predict_video(video, asr=True)
            res.update({"duration_sec": round(dur, 1), "segments": 1, "timeline": []})
            return res
        segs = plan_segments(dur, self.seg_len)
        if progress:
            progress(0.1, f"Распознавание речи (ролик {fmt_secs(dur)})")
        full_text, chunks = self.transcribe(video, work_dir)
        check()
        # re-running the same work dir: reuse the stored behaviour descriptions (Ollama is the slow part)
        old_desc = {}
        old_tl = work_dir / "timeline.json"
        if old_tl.exists() and "behavior" in inspect.signature(self.backend.predict_video).parameters:
            try:
                for t in json.loads(old_tl.read_text(encoding="utf-8")):
                    if t.get("behavior_description"):
                        old_desc[(t["segment"], round(t["start"], 1), round(t["end"], 1))] = t["behavior_description"]
            except Exception:  # noqa: BLE001
                old_desc = {}
        seg_results, timeline, descs = [], [], []
        for i, (s, e) in enumerate(segs, 1):
            check()
            if progress:
                progress(0.15 + 0.7 * (i - 1) / len(segs), f"Отрезок {i}/{len(segs)} ({seg_label(s, e)})")
            seg_path = work_dir / f"seg{i:02d}_{int(s)}-{int(e)}s.mp4"
            if not seg_path.exists():
                cut_segment(video, s, e, seg_path)
            seg_text = self.text_for(chunks, s, e)
            kw = {}
            beh = old_desc.get((i, round(s, 1), round(e, 1)))
            if beh:
                kw["behavior"] = beh
            try:
                r = self.backend.predict_video(seg_path, asr=False, transcript=seg_text, **kw)
            except Exception as ex:  # a segment without a face or speech: skip it, keep the rest
                log.warning("segment %d (%.0f-%.0fs) skipped: %s", i, s, e, str(ex).splitlines()[0][:160])
                timeline.append({"segment": i, "start": round(s, 1), "end": round(e, 1), "scores": None,
                                 "transcript": seg_text, "error": str(ex).splitlines()[0][:160], "file": str(seg_path)})
                continue
            seg_results.append(r)
            timeline.append({"segment": i, "start": round(s, 1), "end": round(e, 1), "scores": r["scores"],
                             "transcript": seg_text, "behavior_description": r.get("behavior_description", ""),
                             "members_used": r.get("members_used"), "file": str(seg_path)})
            if r.get("behavior_description"):
                descs.append(f"[{s:.0f}–{e:.0f} с] {r['behavior_description']}")
            log.info("segment %d/%d %.0f-%.0fs: %s", i, len(segs), s, e,
                     {k[:5]: round(v, 3) for k, v in r["scores"].items()})
        if not seg_results:
            raise RuntimeError("no segment could be analysed (no face or speech found in the whole video)")
        ok = [t for t in timeline if t.get("scores")]
        w = np.array([t["end"] - t["start"] for t in ok], dtype=float)
        w /= w.sum()
        # keys = union over segments: a segment where the own model failed has no "interview" score, so every
        # key is averaged over the segments that actually have it (NaN elsewhere)
        keys = list(TRAIT_KEYS) + sorted({k for r in seg_results for k in r["scores"]} - set(TRAIT_KEYS))
        mat = np.array([[r["scores"].get(k, np.nan) for k in keys] for r in seg_results], dtype=float)   # [S, K]
        have = ~np.isnan(mat)
        wk = np.where(have, w[:, None], 0.0)
        wk = wk / np.maximum(wk.sum(axis=0, keepdims=True), 1e-12)
        mean_vec = np.nansum(np.nan_to_num(mat) * wk, axis=0)
        mean = {k: float(v) for k, v in zip(keys, mean_vec)}
        std = {k: float(np.nanstd(mat[:, i])) if have[:, i].any() else 0.0 for i, k in enumerate(keys)}
        # representative segment = closest to the mean profile on the five traits (used for explanations)
        core = mat[:, : len(TRAIT_KEYS)]
        rep_pos = int(np.argmin(((core - mean_vec[: len(TRAIT_KEYS)]) ** 2).sum(axis=1)))
        rep_idx = ok[rep_pos]["segment"] - 1
        # per-member means over the segments where that member produced a score (a member may skip a segment)
        variants = {}
        for t, r in zip(ok, seg_results):
            for name, sc in (r.get("variants") or {}).items():
                variants.setdefault(name, []).append((t["end"] - t["start"], [sc[k] for k in TRAIT_KEYS]))
        agg = {}
        for name, items in variants.items():
            ww = np.array([d for d, _ in items], dtype=float)
            ww /= ww.sum()
            vals = (np.array([v for _, v in items]) * ww[:, None]).sum(axis=0)
            agg[name] = {k: float(v) for k, v in zip(TRAIT_KEYS, vals)}
            agg[name]["_segments"] = len(items)
        variants = agg
        out = {"scores": mean, "scores_std": std, "timeline": timeline, "segments": len(segs),
               "segments_ok": len(seg_results), "duration_sec": round(dur, 1), "transcript": full_text,
               "behavior_description": "\n".join(descs), "representative_segment": rep_idx + 1,
               "chunks": [[round(s, 2), round(e, 2), txt] for s, e, txt in chunks],       # Whisper timestamps for speech analytics
               "seconds": round(time.time() - t0, 2)}
        if variants:
            out["variants"] = variants
        if seg_results[0].get("primary"):
            out["primary"] = seg_results[0]["primary"]
        (work_dir / "timeline.json").write_text(json.dumps(timeline, ensure_ascii=False, indent=1), encoding="utf-8")
        return out
