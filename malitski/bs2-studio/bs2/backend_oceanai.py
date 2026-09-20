"""OCEAN-AI backend: loads the pretrained audio/video/text/fusion models and scores videos.

OCEAN-AI (aimclub/OCEANAI, BSD-3) ships weights for two corpora:
  fi    - First Impressions V2 (English YouTube clips); text branch expects English
  mupta - MuPTA (Russian speech); text branch translates ru->en internally
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .norms import OCEANAI_COLUMNS, TRAIT_KEYS

log = logging.getLogger("bs.oceanai")

CORPUS_BY_LANG = {"en": "fi", "ru": "mupta"}
MEDIA_EXTS = [".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg"]


@dataclass
class BackendConfig:
    lang: str = "en"                      # language of speech in the videos: en | ru
    corpus: str | None = None             # fi | mupta; default follows lang
    models_dir: str = os.path.expanduser("~/bs/models")
    asr_model: str = "openai/whisper-large-v3-turbo"  # HF id used by the transformers Whisper inside OCEAN-AI
    disk: str = "googledisk"
    force_reload: bool = False
    # OCEAN-AI windowing parameters (documented defaults)
    sr: int = 44100
    window_audio: float = 2.0
    step_audio: float = 1.0
    reduction_fps: int = 5
    window_video: int = 10
    step_video: int = 5

    def __post_init__(self):
        if self.lang not in CORPUS_BY_LANG:
            raise ValueError(f"lang must be one of {list(CORPUS_BY_LANG)}, got {self.lang!r}")
        if self.corpus is None:
            self.corpus = CORPUS_BY_LANG[self.lang]
        if self.corpus == "fi" and self.lang != "en":
            # the library returns no predictions in this combination: FI hand-crafted video features and LIWC text
            # features are English-only. Translate the transcripts yourself and run with lang="en".
            raise ValueError("OCEAN-AI FI weights only run with lang='en'; pass English (translated) transcripts")
        # OCEAN-AI compares corpus names with `is`; interning makes our string identical to its literals.
        self.corpus = sys.intern(self.corpus)
        self.lang = sys.intern(self.lang)


def _guard_translation_generate(b5, max_positions: int = 512):
    """OCEAN-AI translates Russian text with Helsinki-NLP/opus-mt-ru-en (Marian, 512 positions) but calls
    generate(max_new_tokens=4000). On an unpunctuated ASR chunk the decoder can loop and run past position 512,
    which raises a device-side assert (Indexing.cu: srcIndex < srcSelectDimSize) and poisons the CUDA context for
    the whole process. Cap the generation length below the position table and, if the output still looks like a
    repetition loop, re-translate with a no-repeat constraint."""
    import torch

    model = getattr(b5, "_traslate_model", None)
    if model is None or getattr(model.generate, "_bs_guarded", False):
        return
    limit = min(max_positions - 2, int(getattr(model.config, "max_position_embeddings", max_positions) or max_positions) - 2)
    orig = model.generate

    @torch.no_grad()
    def generate(*args, **kw):
        ids = args[0] if args else kw.get("input_ids", kw.get("inputs"))
        n_in = int(ids.shape[-1]) if hasattr(ids, "shape") else limit
        kw["max_new_tokens"] = min(int(kw.get("max_new_tokens") or limit), limit)
        kw.pop("max_length", None)
        out = orig(*args, **kw)
        if out.shape[-1] > max(48, 3 * n_in):           # runaway repetition: translate again, forbid repeats
            log.warning("translation loop (%d in -> %d out tokens); retrying with no_repeat_ngram_size", n_in, out.shape[-1])
            kw.update(max_new_tokens=min(limit, 3 * n_in + 32), no_repeat_ngram_size=3, repetition_penalty=1.2)
            out = orig(*args, **kw)
        return out

    generate._bs_guarded = True
    model.generate = generate
    log.info("translation generate() capped at %d new tokens", limit)


def _patch_torchaudio_load():
    """torchaudio >= 2.9 delegates load() to torchcodec, which is not installed; OCEAN-AI calls
    torchaudio.load(wav) in its ASR path. Replace it with a soundfile-based loader with the same contract."""
    import torch
    import torchaudio

    if getattr(torchaudio.load, "_bs_patched", False):
        return

    def _load(path, frame_offset: int = 0, num_frames: int = -1, normalize: bool = True,
              channels_first: bool = True, **_):
        import soundfile as sf
        data, sr = sf.read(str(path), start=frame_offset, frames=num_frames if num_frames > 0 else -1,
                           dtype="float32" if normalize else "int16", always_2d=True)   # [N, C]
        wav = torch.from_numpy(data)
        return (wav.T.contiguous() if channels_first else wav), int(sr)

    _load._bs_patched = True
    torchaudio.load = _load


class OceanAIBackend:
    """Thin wrapper around oceanai.modules.lab.build.Run."""

    def __init__(self, cfg: BackendConfig | None = None):
        self.cfg = cfg or BackendConfig()
        self.b5 = None
        self.load_seconds = 0.0

    # ------------------------------------------------------------------ loading
    @staticmethod
    def _check(ok: bool, what: str):
        if not ok:
            raise RuntimeError(f"OCEAN-AI step failed: {what}")

    def load(self):
        if self.b5 is not None:
            return self
        t0 = time.time()
        _patch_torchaudio_load()
        from oceanai.modules.lab.build import Run

        cfg = self.cfg
        b5 = Run(lang="en", metadata=False)
        b5.path_to_save_ = cfg.models_dir
        b5.chunk_size_ = 2_000_000
        Path(cfg.models_dir).mkdir(parents=True, exist_ok=True)
        W = b5.weights_for_big5_
        corpus, disk, fr = cfg.corpus, cfg.disk, cfg.force_reload
        q = dict(out=False, runtime=False, run=True)

        log.info("loading OCEAN-AI models: corpus=%s lang=%s models_dir=%s", corpus, cfg.lang, cfg.models_dir)
        # audio
        self._check(b5.load_audio_model_hc(**q), "load_audio_model_hc")
        self._check(b5.load_audio_model_nn(**q), "load_audio_model_nn")
        self._check(b5.load_audio_model_weights_hc(url=W["audio"][corpus]["hc"][disk], force_reload=fr, **q), "audio hc weights")
        self._check(b5.load_audio_model_weights_nn(url=W["audio"][corpus]["nn"][disk], force_reload=fr, **q), "audio nn weights")
        # video
        # the hand-crafted video model differs per corpus (FI: "en", MuPTA: "ru"), not per speech language
        self._check(b5.load_video_model_hc(lang="ru" if corpus == "mupta" else "en", **q), "load_video_model_hc")
        self._check(b5.load_video_model_deep_fe(**q), "load_video_model_deep_fe")
        self._check(b5.load_video_model_nn(**q), "load_video_model_nn")
        self._check(b5.load_video_model_weights_hc(url=W["video"][corpus]["hc"][disk], force_reload=fr, **q), "video hc weights")
        self._check(b5.load_video_model_weights_deep_fe(url=W["video"][corpus]["fe"][disk], force_reload=fr, **q), "video fe weights")
        self._check(b5.load_video_model_weights_nn(url=W["video"][corpus]["nn"][disk], force_reload=fr, **q), "video nn weights")
        # text
        self._check(b5.load_text_features(force_reload=fr, **q), "load_text_features (LIWC dictionary)")
        if cfg.lang == "ru":
            self._check(b5.setup_translation_model(**q), "setup_translation_model")
            _guard_translation_generate(b5)
        self._check(b5.setup_bert_encoder(force_reload=False, **q), "setup_bert_encoder")
        self._check(b5.load_text_model_hc(corpus=corpus, **q), "load_text_model_hc")
        self._check(b5.load_text_model_nn(corpus=corpus, **q), "load_text_model_nn")
        self._check(b5.load_text_model_weights_hc(url=W["text"][corpus]["hc"][disk], force_reload=fr, **q), "text hc weights")
        self._check(b5.load_text_model_weights_nn(url=W["text"][corpus]["nn"][disk], force_reload=fr, **q), "text nn weights")
        # fusion
        self._check(b5.load_avt_model_b5(**q), "load_avt_model_b5")
        self._check(b5.load_avt_model_weights_b5(url=W["avt"][corpus]["b5"][disk], force_reload=fr, **q), "avt weights")
        # ASR model id (OCEAN-AI default is openai/whisper-base)
        b5._path_to_transriber = cfg.asr_model
        self.b5 = b5
        self.load_seconds = time.time() - t0
        log.info("models ready in %.1fs on %s", self.load_seconds, getattr(b5, "_device", "?"))
        return self

    # --------------------------------------------------------------- prediction
    def _run_oceanai(self, files: list[Path], asr: bool, exts: set[str], verbose: bool) -> pd.DataFrame | None:
        """One get_avt_predictions call over `files`. Returns None when OCEAN-AI aborts the batch.

        OCEAN-AI expects dataset_root/<group>/file (depth=1 lists subfolders), so the files are exposed
        through a temporary root with one "clips" subfolder of symlinks. It resolves the symlinks, so a
        <stem>.txt transcript must sit next to the real file when asr=False.
        """
        b5, cfg = self.b5, self.cfg
        with tempfile.TemporaryDirectory(prefix="bs_ds_") as tmp:
            sub = Path(tmp) / "clips"
            sub.mkdir()
            for p in files:
                os.symlink(p, sub / p.name)
            b5.path_to_dataset_ = tmp
            b5.ignore_dirs_ = []
            b5.keys_dataset_ = ["Path"] + OCEANAI_COLUMNS
            b5.ext_ = sorted(exts)
            ok = b5.get_avt_predictions(
                depth=1, recursive=False, sr=cfg.sr,
                window_audio=cfg.window_audio, step_audio=cfg.step_audio,
                reduction_fps=cfg.reduction_fps, window_video=cfg.window_video, step_video=cfg.step_video,
                asr=asr, lang=cfg.lang, accuracy=False, url_accuracy="", logs=False,
                out=verbose, runtime=verbose, run=True,
            )
            df = b5.df_files_.copy()
        if not ok or df.empty:
            return None
        return df.reset_index(drop=True)

    def predict_dir(self, directory: str | Path, asr: bool = True, exts: Iterable[str] = MEDIA_EXTS,
                    verbose: bool = False, chunk_size: int = 25) -> pd.DataFrame:
        """Score every media file in `directory` (non-recursive). Returns DataFrame[Path, 5 traits].

        OCEAN-AI aborts a whole batch when one file fails (no face, no speech, model error), so files are
        processed in chunks and a failing chunk is re-run file by file; failures are logged and listed in
        `df.attrs["failed"]`.
        """
        self.load()
        directory = Path(directory).resolve()
        exts = {e.lower() for e in exts}
        files = sorted(p for p in directory.iterdir() if p.suffix.lower() in exts)
        if not files:
            raise FileNotFoundError(f"no media files {sorted(exts)} in {directory}")
        chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)] if chunk_size else [files]
        frames, failed, done = [], [], 0
        for chunk in chunks:
            df = self._run_oceanai(chunk, asr, exts, verbose)
            if df is not None:
                frames.append(df)
            elif len(chunk) == 1:
                failed.append(chunk[0].name)
                log.warning("oceanai: %s failed (no face/speech or model error)", chunk[0].name)
            else:  # isolate the culprit(s)
                for p in chunk:
                    d1 = self._run_oceanai([p], asr, exts, verbose)
                    if d1 is None:
                        failed.append(p.name)
                        log.warning("oceanai: %s failed (no face/speech or model error)", p.name)
                    else:
                        frames.append(d1)
            done += len(chunk)
            if len(chunks) > 1:
                log.info("oceanai: %d/%d files processed, %d failed", done, len(files), len(failed))
        if not frames:
            raise RuntimeError("OCEAN-AI returned no predictions for any file (no face/speech found, or a model "
                               "step failed); rerun with verbose=True to see the library messages")
        out = pd.concat(frames, ignore_index=True)
        out.attrs["failed"] = failed
        return out

    def last_transcript(self) -> str:
        return str(getattr(self.b5, "_Text__text_pred", "") or "")

    def predict_video(self, video: str | Path, asr: bool = True, transcript: str | None = None) -> dict:
        """Score one video. Returns {"scores": {trait_key: float}, "transcript": str, "seconds": float}."""
        video = Path(video).resolve()
        if not video.is_file():
            raise FileNotFoundError(video)
        t0 = time.time()
        with tempfile.TemporaryDirectory(prefix="bs_") as tmp:
            tmpdir = Path(tmp)
            # OCEAN-AI builds an unquoted ffmpeg shell command from the path and resolves symlinks, so the
            # clip is hard-linked/copied under a shell-safe name and the transcript is written next to it.
            safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", video.stem) or "video"
            local = tmpdir / (safe_stem + video.suffix.lower())
            try:
                os.link(video, local)
            except OSError:
                shutil.copy2(video, local)
            if transcript is not None:
                (tmpdir / (safe_stem + ".txt")).write_text(transcript.strip() or " ", encoding="utf-8")
                asr = False
            df = self.predict_dir(tmpdir, asr=asr, exts=[video.suffix.lower()])
            row = df.iloc[0]
            scores = {k: float(row[c]) for k, c in zip(TRAIT_KEYS, OCEANAI_COLUMNS)}
            text = transcript if transcript is not None else self.last_transcript()
        return {"scores": scores, "transcript": text, "seconds": round(time.time() - t0, 2)}
