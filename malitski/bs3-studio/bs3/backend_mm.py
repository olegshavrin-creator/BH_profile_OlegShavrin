"""Inference backend for the own MM-PSYCHE-style model AMLAI 1.0.

video -> 30 face crops -> CLIP ; wav 48 kHz -> CLAP ; Whisper transcript -> EmoRoBERTa ;
(optional) behaviour description from a local Ollama vision model on sampled frames -> EmoRoBERTa ;
-> PersonalityFusionModel checkpoints (settings.MM_CHECKPOINTS: 5 seeds, averaged) -> 5 scores.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import subprocess
import tempfile
import time
import urllib.error
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from . import ollama, settings
from .errors import OllamaUnavailable
from .mm.extractors import ClapAudioEncoder, ClipFaceEncoder, EmoRobertaTextEncoder, mean_std
from .mm.faces import get_face_crops, select_uniform_frames
from .mm.model import ModelConfig, PersonalityFusionModel
from .norms import TRAIT_KEYS

log = logging.getLogger("bs.mm")

# Full prompt from the MM-PSYCHE paper (pytorch_Qwen3-VL.py / figures/prompts.jpg); the training descriptions in
# data/fiv2/*.csv were generated with it by Qwen3-VL-4B-Instruct on the whole clip. Here the clip is given to a
# local Ollama vision model as uniformly sampled frames.
BEHAVIOR_PROMPT = """You are an expert in visual human behavior analysis. Carefully examine the provided video clip (given as a sequence of frames), which features a person facing the camera. Your task is to describe, in continuous natural language, the person's visible emotional state, personality tendencies, or possible signs of ambivalence and hesitancy as reflected through their nonverbal behavior.

Focus exclusively on observable cues such as facial muscle movements (eyes, eyebrows, mouth, gaze), body posture, gestures, and head motions. Infer emotional tendencies (neutral, anger, disgust, fear, happiness, sadness, surprise), personality traits (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism), or subtle conflicting signals of uncertainty and hesitation when visible.

In your description:
- Comment on the person's appearance, posture, gestures, and expressiveness as indicators of emotional state, personality, or ambivalence.
- Observe and explain facial expressions and body movements as cues, highlighting consistency or discordance across behaviors.
- Avoid assumptions about personal background, spoken content, or context beyond what is visually observable.
- If the state appears mixed or ambiguous, briefly mention this with a short explanation based on visible cues.

Your final response must be a fluent, continuous natural language interpretation of the person's visible behavior in the video, written as a single coherent paragraph without any line breaks, bullet points, special characters, or formatting. The response must express a complete, finished thought and must not exceed 75 tokens in total."""


@dataclass
class MMConfig:
    # one path, a comma-separated list, or a glob: several checkpoints (e.g. seeds) are averaged
    checkpoint: str = settings.MM_CHECKPOINTS
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    lang: str = "en"
    asr_model: str = settings.ASR_MODEL
    ollama_url: str = ""                       # "" -> ollama.url(): BS3_OLLAMA_URL, else localhost or the WSL gateway
    ollama_model: str = settings.OLLAMA_MODEL
    behavior_frames: int = 16
    behavior_max_side: int = 640               # frames sent to the VLM are downscaled to this longest side
    n_frames: int = 30
    corpus: str = "mm-psyche-style (own, FIV2 train)"
    drop_modalities: tuple = ()                # diagnostics: score without these graph nodes (e.g. ("audio",))


class MMBackend:
    def __init__(self, cfg: MMConfig | None = None):
        self.cfg = cfg or MMConfig()
        self.loaded = False
        self.load_seconds = 0.0
        self._asr = None

    def load(self):
        if self.loaded:
            return self
        t0 = time.time()
        cfg = self.cfg
        import glob
        paths = []
        for part in str(cfg.checkpoint).split(","):
            part = os.path.expanduser(part.strip())
            paths += sorted(glob.glob(part)) if any(ch in part for ch in "*?[") else [part]
        if not paths:
            raise FileNotFoundError(f"no checkpoint matches {cfg.checkpoint!r}")
        self.models, self.modalities, epochs = [], None, []
        for p in paths:
            ckpt = torch.load(p, map_location=cfg.device)
            mods = list(ckpt["modalities"])
            if self.modalities is None:
                self.modalities = mods
            elif mods != self.modalities:
                raise ValueError(f"checkpoint {p} has modalities {mods}, expected {self.modalities}")
            m = PersonalityFusionModel(ModelConfig(**ckpt["config"])).to(cfg.device)
            m.load_state_dict(ckpt["state_dict"])
            m.eval()
            self.models.append(m)
            epochs.append(ckpt.get("epoch"))
        self.model = self.models[0]                 # used by explain (attributions of one member)
        self.checkpoints = paths
        self.cfg.corpus = f"mm-psyche-style ({'+'.join(self.modalities)}; {len(paths)} checkpoint(s))"
        self.face = ClipFaceEncoder(cfg.device) if "face" in self.modalities else None
        # voice node(s): CLAP ("audio") or one of the candidate speech encoders (see mm/extractors_audio.py)
        from .mm.extractors_audio import AUDIO_ENCODERS
        self.audio_encoders = {}
        for m in self.modalities:
            if m == "audio":
                self.audio_encoders[m] = ClapAudioEncoder(cfg.device)
            elif m in AUDIO_ENCODERS and m != "audio_egemaps":
                self.audio_encoders[m] = AUDIO_ENCODERS[m](cfg.device)
            elif m.startswith("audio"):
                raise ValueError(f"checkpoint uses voice features {m!r} that have no inference encoder")
        self.audio = self.audio_encoders.get("audio")
        self.text = EmoRobertaTextEncoder(cfg.device) if ("text" in self.modalities or "behavior" in self.modalities) else None
        self.loaded = True
        self.load_seconds = time.time() - t0
        log.info("MM model: %d checkpoint(s) %s (modalities %s, epochs %s) in %.1fs", len(paths),
                 [Path(p).parent.name for p in paths], self.modalities, epochs, self.load_seconds)
        return self

    @torch.no_grad()
    def _score(self, batch: dict) -> np.ndarray:
        """Average of all loaded checkpoints."""
        return np.mean([m(batch).cpu().numpy()[0] for m in self.models], axis=0)

    # ---------------------------------------------------------------- pieces
    def _wav(self, video: Path, tmpdir: Path, sr: int) -> Path:
        wav = tmpdir / f"{video.stem}.{sr}.wav"
        if not wav.exists():
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", str(sr),
                            "-acodec", "pcm_s16le", str(wav)], check=True)
        return wav

    def _transcribe(self, video: Path, tmpdir: Path) -> str:
        if self._asr is None:
            from transformers import pipeline
            dev = 0 if self.cfg.device.startswith("cuda") else -1
            self._asr = pipeline("automatic-speech-recognition", model=self.cfg.asr_model,
                                 torch_dtype=torch.float16 if dev == 0 else torch.float32, device=dev,
                                 return_timestamps=True)
        wav = self._wav(video, tmpdir, 16000)
        res = self._asr(str(wav), generate_kwargs={"language": self.cfg.lang, "task": "transcribe"})
        return (res.get("text") or "").strip()

    _translator = None

    def to_english(self, text: str) -> str:
        """The text branch was trained on English transcripts; other languages are translated (Helsinki-NLP
        opus-mt-<lang>-en, the same model OCEAN-AI uses for Russian). Long texts are translated sentence by sentence."""
        if self.cfg.lang == "en" or not text.strip():
            return text
        if self._translator is None:
            from transformers import MarianMTModel, MarianTokenizer
            name = f"Helsinki-NLP/opus-mt-{self.cfg.lang}-en"
            tok = MarianTokenizer.from_pretrained(name)
            mdl = MarianMTModel.from_pretrained(name).to(self.cfg.device).eval()
            self._translator = (tok, mdl)
        tok, mdl = self._translator
        import re as _re
        sentences = [s for s in _re.split(r"(?<=[.!?])\s+", text.strip()) if s]
        out = []
        with torch.no_grad():
            for i in range(0, len(sentences), 8):
                batch = tok(sentences[i:i + 8], return_tensors="pt", padding=True, truncation=True, max_length=400).to(self.cfg.device)
                gen = mdl.generate(**batch, max_new_tokens=400)
                out += tok.batch_decode(gen, skip_special_tokens=True)
        return " ".join(out)

    def _sample_frames_jpeg(self, video: Path) -> list[str]:
        cap = cv2.VideoCapture(str(video))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        want = set(select_uniform_frames(total, self.cfg.behavior_frames))
        out, i = [], 0
        while True:
            ok, im = cap.read()
            if not ok:
                break
            if i in want:
                h, w = im.shape[:2]
                side = int(self.cfg.behavior_max_side or 640)
                if max(h, w) > side:
                    s = side / max(h, w)
                    im = cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
                ok2, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok2:
                    out.append(base64.b64encode(buf.tobytes()).decode("ascii"))
            i += 1
        cap.release()
        return out

    def _ollama(self, prompt: str, images: list[str], num_predict: int, timeout: int = settings.OLLAMA_DESCRIBE_TIMEOUT,
                attempts: int = settings.OLLAMA_DESCRIBE_ATTEMPTS) -> dict:
        """One /api/generate call to the local vision model, with retries while Ollama (re)loads a model.

        "think": False -> Qwen3 models otherwise spend the whole token budget on hidden reasoning and
        return an empty response."""
        if not self.cfg.ollama_url:
            self.cfg.ollama_url = ollama.url()
        payload = {"model": self.cfg.ollama_model, "prompt": prompt, "images": images, "stream": False,
                   "think": False, "keep_alive": settings.OLLAMA_KEEP_ALIVE,
                   "options": {"num_predict": num_predict, "temperature": 0.2}}
        for attempt in range(1, attempts + 1):
            try:
                return ollama.post_json("/api/generate", payload, timeout, base=self.cfg.ollama_url)
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
                if attempt == attempts:
                    # a broken Ollama is a typed error: errors.is_fatal decides whether to fail the whole run at once
                    # (connection refused / 404), and errors.user_message turns it into the calm Russian «нет Ollama»
                    raise OllamaUnavailable(f"Ollama request failed: {str(e).splitlines()[0][:160]}") from e
                log.warning("Ollama request failed (%s); retry %d/%d in %ds", str(e).splitlines()[0][:80], attempt,
                            attempts, 10 * attempt)
                time.sleep(10 * attempt)
        return {}

    def describe_behavior(self, video: Path) -> str:
        """Behaviour description from a local Ollama vision model over uniformly sampled frames."""
        data = self._ollama(BEHAVIOR_PROMPT, self._sample_frames_jpeg(video), 320)
        text = " ".join(str(data.get("response", "")).split()).strip()
        if not text:
            log.warning("Ollama %s returned an empty description for %s (eval %.0fs, thinking=%s chars)",
                        self.cfg.ollama_model, video.name, data.get("eval_duration", 0) / 1e9,
                        len(str(data.get("thinking", ""))))
        return text

    def describe_frame(self, image_b64: str) -> str:
        """A few words about ONE key frame (the caption under it): the same local vision model, one image, a short
        token budget and a shorter timeout — a caption is never worth holding the job for minutes."""
        from .frame_phrase import FRAME_PROMPT, PHRASE_NUM_PREDICT
        data = self._ollama(FRAME_PROMPT, [image_b64], PHRASE_NUM_PREDICT, timeout=settings.PHRASE_TIMEOUT, attempts=1)
        return " ".join(str(data.get("response", "")).split()).strip()

    # ---------------------------------------------------------------- prediction
    @torch.no_grad()
    def predict_video(self, video: str | Path, asr: bool = True, transcript: str | None = None,
                      behavior: str | None = None) -> dict:
        self.load()
        video = Path(video).resolve()
        t0 = time.time()
        feats, timings, info = {}, {}, {}
        with tempfile.TemporaryDirectory(prefix="bs_mm_") as tmp:
            tmpdir = Path(tmp)
            if self.face is not None:
                t = time.time()
                crops, st = get_face_crops(str(video), n_frames=self.cfg.n_frames)
                if not crops:
                    raise RuntimeError("no frames decoded")
                feats["face"] = self.face(crops)
                timings["face"] = round(time.time() - t, 2)
                info["face_frames_with_face"] = st["frames_with_face"]
            if self.audio_encoders:
                t = time.time()
                for m, enc in self.audio_encoders.items():
                    feats[m] = enc.from_file(str(self._wav(video, tmpdir, enc.sample_rate)))
                timings["audio"] = round(time.time() - t, 2)
            transcript_en = None
            if "text" in self.modalities:
                t = time.time()
                if transcript is None:
                    transcript = self._transcribe(video, tmpdir) if asr else ""
                transcript_en = self.to_english(transcript)
                feats["text"] = self.text(transcript_en)
                timings["asr_text"] = round(time.time() - t, 2)
            if "behavior" in self.modalities:
                t = time.time()
                if behavior is None:
                    behavior = self.describe_behavior(video)
                feats["behavior"] = self.text(behavior)
                timings["behavior"] = round(time.time() - t, 2)
        batch = {m: v.unsqueeze(0).to(self.cfg.device) for m, v in feats.items() if m not in self.cfg.drop_modalities}
        scores_vec = self._score(batch)
        keys = (TRAIT_KEYS + ["interview"])[: len(scores_vec)]
        scores = {k: float(v) for k, v in zip(keys, scores_vec)}
        out = {"scores": scores, "transcript": transcript or "", "behavior_description": behavior or "",
               "seconds": round(time.time() - t0, 2), "timings": timings, "info": info}
        if self.cfg.drop_modalities:
            out["info"]["dropped_modalities"] = list(self.cfg.drop_modalities)
        if transcript_en is not None and transcript_en != (transcript or ""):
            out["transcript_en"] = transcript_en
        return out

    def explain_video(self, video: str | Path, out_dir: str | Path, asr: bool = True, transcript: str | None = None,
                      behavior: str | None = None, top_k: int = 5, expression_fn=None, captions: bool = True) -> dict:
        """Scores plus explanations: modality attribution, key frames (saved as JPEG), word attribution.

        `expression_fn` — the facial-expression model of the report (analyses.face_expr.FaceExpression.on_crops),
        called on the face crops of the key frames so their captions can name the expression; `captions=False`
        skips both the expressions and the one-phrase-per-frame requests to the vision model."""
        from .mm.explain import (clip_fps, frame_attribution, key_frame_info, modality_attribution, save_key_frames,
                                 token_attribution)
        self.load()
        video = Path(video).resolve()
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        feats, face_seq, crops, kept = {}, None, [], None
        with tempfile.TemporaryDirectory(prefix="bs_mmx_") as tmp:
            tmpdir = Path(tmp)
            if self.face is not None:
                crops, face_st = get_face_crops(str(video), n_frames=self.cfg.n_frames)
                kept = face_st.get("frames_kept")           # crops[i] is frame kept[i] of the clip
                face_seq = self.face.sequence(crops)                       # [T, 512]
                feats["face"] = mean_std(face_seq)
            for m, enc in self.audio_encoders.items():
                feats[m] = enc.from_file(str(self._wav(video, tmpdir, enc.sample_rate)))
            transcript_en = ""
            if "text" in self.modalities:
                if transcript is None:
                    transcript = self._transcribe(video, tmpdir) if asr else ""
                transcript_en = self.to_english(transcript)
                feats["text"] = self.text(transcript_en)
            if "behavior" in self.modalities:
                if behavior is None:
                    behavior = self.describe_behavior(video)
                feats["behavior"] = self.text(behavior)
        dev = self.cfg.device
        res = {"input": str(video), "transcript": transcript or "", "behavior_description": behavior or ""}
        fps = clip_fps(str(video))              # the captions turn a frame number of THIS clip into a moment
        if fps:
            res["clip_fps"] = fps
        if transcript_en and transcript_en != transcript:
            res["transcript_en"] = transcript_en
        res["modalities"] = modality_attribution(self.model, feats, dev)
        res["scores"] = res["modalities"].pop("scores")
        if face_seq is not None:
            fa = frame_attribution(self.model, face_seq, feats, dev, top_k=top_k)
            raw: dict[str, str] = {}
            fa["key_frame_files"] = save_key_frames(str(video), fa["top_frames_overall"], self.cfg.n_frames, out_dir,
                                                    raw_jpegs=raw if captions else None, kept_frames=kept)
            fa["key_frame_info"] = key_frame_info(fa["key_frame_files"], crops, raw,
                                                  expression_fn=expression_fn if captions else None,
                                                  phrase_fn=self.describe_frame if captions else None)
            res["frames"] = fa
        if "text" in self.modalities and transcript:
            res["transcript_words"] = token_attribution(self.model, self.text, transcript_en or transcript, "text", feats, dev)
        if "behavior" in self.modalities and behavior:
            res["behavior_words"] = token_attribution(self.model, self.text, behavior, "behavior", feats, dev)
        res["seconds"] = round(time.time() - t0, 2)
        (out_dir / "explanation.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        return res
