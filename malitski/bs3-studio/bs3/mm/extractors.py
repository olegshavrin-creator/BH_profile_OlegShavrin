"""Frozen encoders used by MM-PSYCHE, sequence outputs aggregated as mean‖std (unbiased=False).

face     : openai/clip-vit-base-patch32 image features, [T, 512]  -> 1024
audio    : laion/clap-htsat-fused audio_model.last_hidden_state, [T, 768] -> 1536
text     : michellejieli/emotion_text_classifier (DistilRoBERTa) last_hidden_state, [L, 768] -> 1536
behavior : same text encoder on the behaviour description
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np
import soundfile as sf
import soxr
import torch

log = logging.getLogger("bs.mm.extractors")

CLIP_ID = "openai/clip-vit-base-patch32"
CLAP_ID = "laion/clap-htsat-fused"
TEXT_ID = "michellejieli/emotion_text_classifier"

FEATURE_DIMS = {"face": 1024, "audio": 1536, "text": 1536, "behavior": 1536}


def mean_std(seq: torch.Tensor) -> torch.Tensor:
    """[T, D] -> [2D] = mean ‖ std (population std, as in MM-PSYCHE _aggregate)."""
    if seq.ndim == 1:
        seq = seq.unsqueeze(0)
    return torch.cat([seq.mean(dim=0), seq.std(dim=0, unbiased=False)], dim=0)


def load_wav_mono(path: str, target_sr: int) -> np.ndarray:
    """Mono float32 waveform at target_sr, peak-normalised (MM-PSYCHE load_wav_mono without torchaudio)."""
    wav, sr = sf.read(path, dtype="float32", always_2d=True)   # [N, C]
    wav = wav.mean(axis=1)
    if wav.size == 0:
        raise ValueError(f"empty audio: {path}")
    if sr != target_sr:
        wav = soxr.resample(wav, sr, target_sr)
    return wav / (np.abs(wav).max() + 1e-8)


class ClipFaceEncoder:
    def __init__(self, device="cuda"):
        from transformers import CLIPModel, CLIPProcessor
        self.device = device
        self.model = CLIPModel.from_pretrained(CLIP_ID).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(CLIP_ID)

    @torch.no_grad()
    def sequence(self, images: List[np.ndarray]) -> torch.Tensor:
        pv = self.proc(images=list(images), return_tensors="pt")["pixel_values"].to(self.device)
        return self.model.get_image_features(pixel_values=pv).float().cpu()          # [T, 512]

    def __call__(self, images) -> torch.Tensor:
        return mean_std(self.sequence(images))


class ClapAudioEncoder:
    def __init__(self, device="cuda"):
        from transformers import ClapModel, ClapProcessor
        self.device = device
        self.model = ClapModel.from_pretrained(CLAP_ID).to(device).eval()
        self.proc = ClapProcessor.from_pretrained(CLAP_ID)
        self.sample_rate = int(getattr(self.model.config.audio_config, "sampling_rate", 48000))

    @torch.no_grad()
    def sequence(self, wav: np.ndarray) -> torch.Tensor:
        inputs = self.proc(audios=[wav], return_tensors="pt", sampling_rate=self.sample_rate)
        inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        args = {k: inputs[k] for k in ("input_features", "is_longer") if k in inputs}
        out = self.model.audio_model(**args, output_hidden_states=True, return_dict=True)
        hidden = out.last_hidden_state
        if hidden.ndim == 4:                          # [B, H, T, F]
            hidden = hidden.mean(dim=-1).transpose(1, 2)
        elif hidden.ndim == 3 and hidden.shape[1] == 768 and hidden.shape[2] != 768:
            hidden = hidden.transpose(1, 2)           # [B, H, T] -> [B, T, H]
        return hidden.squeeze(0).float().cpu()        # [T, 768]

    def from_file(self, wav_path: str) -> torch.Tensor:
        return mean_std(self.sequence(load_wav_mono(wav_path, self.sample_rate)))

    def __call__(self, wav: np.ndarray) -> torch.Tensor:
        return mean_std(self.sequence(wav))


class EmoRobertaTextEncoder:
    def __init__(self, device="cuda"):
        from transformers import AutoModel, AutoTokenizer
        self.device = device
        self.model = AutoModel.from_pretrained(TEXT_ID, torch_dtype=torch.float32).to(device).eval()
        self.tok = AutoTokenizer.from_pretrained(TEXT_ID)

    @torch.no_grad()
    def sequence(self, text: str) -> torch.Tensor:
        inputs = self.tok(text or "", padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs, return_dict=True).last_hidden_state
        valid = int(inputs["attention_mask"].sum(dim=1).item())
        return out[0, :valid].float().cpu()           # [L, 768]

    def __call__(self, text: str) -> torch.Tensor:
        return mean_std(self.sequence(text))
