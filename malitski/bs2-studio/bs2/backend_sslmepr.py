"""SSL-MEPR backend (benchmark only): the predecessor multimodal cross-domain model on the three
modalities whose unimodal checkpoints are public (scene, audio, text). Face and body checkpoints
were never released, so those branches are disabled in the fusion model.

Needs the `app` branch of LEYA-HSE/SSL-MEPR checked out (BS/SSL-MEPR-app) and the checkpoints
downloaded by scripts/download_ssl_mepr_weights.sh.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .norms import OCEANAI_COLUMNS, TRAIT_KEYS

log = logging.getLogger("bs.sslmepr")

_HERE = Path(__file__).resolve()
DEFAULT_APP_ROOT = _HERE.parents[2] / "SSL-MEPR-app"          # BS/SSL-MEPR-app
DEFAULT_CKPT_ROOT = Path(os.path.expanduser("~/bs/ssl_mepr_ckpt"))


@dataclass
class SSLMEPRConfig:
    app_root: Path = DEFAULT_APP_ROOT
    ckpt_root: Path = DEFAULT_CKPT_ROOT
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    n_frames: int = 30                 # uniform frames for the scene branch (as in the app)
    audio_chunk_sec: float = 20.0      # wav2vec2-large is quadratic in length; chunk long audio
    asr_model: str = "openai/whisper-large-v3-turbo"
    lang: str = "en"                   # ASR language
    modalities: tuple = ("scene", "audio", "text")
    corpus: str = "ssl-mepr(scene+audio+text)"   # label for reports


def _as_logits_if_prob(t: torch.Tensor) -> torch.Tensor:
    if torch.is_floating_point(t):
        mn, mx = float(t.min()), float(t.max())
        if 0.0 <= mn and mx <= 1.0:
            return torch.logit(t.clamp(1e-4, 1 - 1e-4))
    return t


class SSLMEPRBackend:
    def __init__(self, cfg: SSLMEPRConfig | None = None):
        self.cfg = cfg or SSLMEPRConfig()
        self.loaded = False
        self.load_seconds = 0.0
        self._asr = None

    # ------------------------------------------------------------------ loading
    def load(self):
        if self.loaded:
            return self
        t0 = time.time()
        cfg = self.cfg
        root = Path(cfg.app_root)
        if not (root / "core").is_dir():
            raise FileNotFoundError(f"SSL-MEPR app branch not found at {root}")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        # model_loader reads its toml/ckpt paths relative to CWD -> patch to absolute paths
        from core.modalities.video import model_loader as vml
        vml.MODALITY_META["scene"]["toml"] = root / "core/modalities/video/config/inference_config_scene.toml"
        vml.MODALITY_META["scene"]["ckpt"] = str(cfg.ckpt_root / "scene/clip_fusion_transformer_transformer_mamba_best_model_dev.pt")
        from transformers import CLIPModel, CLIPProcessor
        from core.modalities.audio.feature_extractor import PretrainedAudioEmbeddingExtractor
        from core.modalities.text.feature_extractor import PretrainedTextEmbeddingExtractor
        from core.models.models import MultiModalFusionModelWithAblation

        dev = cfg.device
        log.info("loading SSL-MEPR (scene+audio+text) on %s", dev)
        self.clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval()
        self.scene_model = vml.get_fusion_model("scene", dev)
        self.audio = self.text = None
        if "audio" in cfg.modalities:
            a = cfg.ckpt_root / "audio"
            self.audio = PretrainedAudioEmbeddingExtractor(
                device=dev,
                fusion_ckpt=str(a / "best_fusion_overall_mamba.pt"),
                emo_ckpt=str(a / "final_best_model_uni_mamba.pt"),
                per_ckpt=str(a / "best_mamba_regressor.pth"),
            )
        if "text" in cfg.modalities:
            t = cfg.ckpt_root / "text"
            self.text = PretrainedTextEmbeddingExtractor(
                device=dev,
                fusion_ckpt=str(t / "Mamba_Transformer_bge-small_fusion.pt"),
                emo_ckpt=str(t / "Mamba_bge-small_emotion.pt"),
                per_ckpt=str(t / "Transformer_bge-small_personality.pt"),
            )
        disabled = [m for m in ("body", "face", "scene", "audio", "text") if m not in cfg.modalities]
        self.fusion = MultiModalFusionModelWithAblation(
            hidden_dim=256, num_heads=8, dropout=0.2, emo_out_dim=7, pkl_out_dim=5, device=dev,
            ablation_config={"disabled_modalities": disabled, "disable_guide_pkl": True},
        ).to(dev)
        state = torch.load(root / "ssl_mepr_ckpt.pt", map_location=dev)
        missing, unexpected = self.fusion.load_state_dict(state, strict=False)
        if missing or unexpected:
            log.warning("fusion state_dict mismatch: missing=%s unexpected=%s", missing, unexpected)
        self.fusion.eval()
        self.loaded = True
        self.load_seconds = time.time() - t0
        log.info("SSL-MEPR ready in %.1fs (disabled modalities: %s)", self.load_seconds, disabled)
        return self

    # --------------------------------------------------------------- features
    @torch.no_grad()
    def _scene(self, video: Path) -> dict:
        import cv2
        cap = cv2.VideoCapture(str(video))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        want = set(np.linspace(0, max(total - 1, 0), num=min(self.cfg.n_frames, max(total, 1)), dtype=int).tolist())
        frames, i = [], 0
        while True:
            ok, im = cap.read()
            if not ok:
                break
            if i in want:
                frames.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            i += 1
        cap.release()
        if not frames:
            raise RuntimeError(f"no frames decoded from {video}")
        pv = self.clip_proc(images=frames, return_tensors="pt")["pixel_values"].to(self.cfg.device)
        feats = self.clip.get_image_features(pv).unsqueeze(0)              # [1, N, 512]
        out = self.scene_model(emotion_input=feats, personality_input=feats, return_features=True)
        return {k: out[k].cpu() for k in ("emotion_logits", "personality_scores",
                                          "last_emo_encoder_features", "last_per_encoder_features")}

    @torch.no_grad()
    def _audio(self, video: Path, tmpdir: Path) -> dict:
        import librosa
        wav = tmpdir / (video.stem + ".16k.wav")
        if not wav.exists():
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
                            "-acodec", "pcm_s16le", str(wav)], check=True)
        y, sr = librosa.load(str(wav), sr=16000, mono=True)
        n = int(self.cfg.audio_chunk_sec * sr)
        chunks = [y[i:i + n] for i in range(0, len(y), n) if len(y[i:i + n]) >= sr]  # >= 1 s
        if not chunks:
            chunks = [y] if len(y) else [np.zeros(sr, dtype=np.float32)]
        outs = [self.audio.extract(waveform=c) for c in chunks]
        return {
            "emotion_logits": torch.stack([o["emotion_logits"] for o in outs]).mean(0),
            "personality_scores": torch.stack([o["personality_scores"] for o in outs]).mean(0),
            "last_emo_encoder_features": torch.cat([o["last_emo_encoder_features"] for o in outs], dim=1),
            "last_per_encoder_features": torch.cat([o["last_per_encoder_features"] for o in outs], dim=1),
        }

    @torch.no_grad()
    def _text(self, text: str) -> dict:
        return self.text.extract(text.strip() or " ")

    def _transcribe(self, video: Path, tmpdir: Path, lang: str) -> str:
        if self._asr is None:
            from transformers import pipeline
            dev = 0 if self.cfg.device.startswith("cuda") else -1
            self._asr = pipeline("automatic-speech-recognition", model=self.cfg.asr_model,
                                 torch_dtype=torch.float16 if dev == 0 else torch.float32, device=dev,
                                 return_timestamps=True)
        wav = tmpdir / (video.stem + ".16k.wav")
        if not wav.exists():
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
                            "-acodec", "pcm_s16le", str(wav)], check=True)
        res = self._asr(str(wav), generate_kwargs={"language": lang, "task": "transcribe"})
        return (res.get("text") or "").strip()

    # --------------------------------------------------------------- prediction
    @torch.no_grad()
    def predict_video(self, video: str | Path, asr: bool = True, transcript: str | None = None) -> dict:
        self.load()
        video = Path(video).resolve()
        t0 = time.time()
        with tempfile.TemporaryDirectory(prefix="bs_mepr_") as tmp:
            tmpdir = Path(tmp)
            feats, emo, per = {}, {}, {}
            results = {"scene": self._scene(video)}
            if self.audio is not None:
                results["audio"] = self._audio(video, tmpdir)
            if self.text is not None:
                if transcript is None:
                    transcript = self._transcribe(video, tmpdir, self.cfg.lang) if asr else ""
                results["text"] = self._text(transcript)
            else:
                transcript = transcript or ""
            for name, r in results.items():
                feats[name] = torch.cat((r["last_emo_encoder_features"].mean(dim=1),
                                         r["last_per_encoder_features"].mean(dim=1)), dim=1)
                emo[name] = r["emotion_logits"]
                per[name] = _as_logits_if_prob(r["personality_scores"])
            out = self.fusion({"features": feats, "emotion_logits": emo, "personality_scores": per})
            fusion_vec = out["personality_scores"].cpu().numpy()[0]
        unimodal = {name: {k: float(v) for k, v in zip(TRAIT_KEYS, torch.sigmoid(per[name]).numpy()[0])}
                    for name in results}
        # The released MCDM checkpoint was trained with all five modalities; without face/body its output
        # collapses (~0.1-0.2 for every trait). Late fusion = mean of the three unimodal predictions is the
        # honest 3-modality estimate, so it is reported as the main score; the MCDM output is kept as a variant.
        late = {k: float(np.mean([unimodal[m][k] for m in unimodal])) for k in TRAIT_KEYS}
        variants = {"fusion": {k: float(v) for k, v in zip(TRAIT_KEYS, fusion_vec)}, **unimodal}
        return {"scores": late, "transcript": transcript, "seconds": round(time.time() - t0, 2),
                "unimodal": unimodal, "variants": variants}

    def predict_dir(self, directory: str | Path, asr: bool = True,
                    exts=(".mp4", ".mov", ".mkv", ".avi", ".webm")) -> pd.DataFrame:
        """Same contract as the OCEAN-AI backend: DataFrame[Path, 5 traits]. With asr=False reads <stem>.txt."""
        self.load()
        rows = []
        files = sorted(p for p in Path(directory).iterdir() if p.suffix.lower() in exts)
        for i, p in enumerate(files, 1):
            txt = p.with_suffix(".txt")
            transcript = txt.read_text(encoding="utf-8") if (not asr and txt.exists()) else None
            try:
                r = self.predict_video(p, asr=asr, transcript=transcript)
                row = {"Path": p.name, **{c: r["scores"][k] for k, c in zip(TRAIT_KEYS, OCEANAI_COLUMNS)}}
                for vname, vs in r["variants"].items():          # fusion:Openness, scene:Openness, ...
                    row.update({f"{vname}:{c}": vs[k] for k, c in zip(TRAIT_KEYS, OCEANAI_COLUMNS)})
                rows.append(row)
            except Exception as e:  # keep going, report failures
                log.warning("%s failed: %s", p.name, e)
            if i % 20 == 0:
                log.info("scored %d/%d", i, len(files))
        return pd.DataFrame(rows)
