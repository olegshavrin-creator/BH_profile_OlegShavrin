"""Candidate replacements for the CLAP audio branch (language-independent speech encoders), same contract as
ClapAudioEncoder: .sample_rate, .sequence(wav) -> [T, D], .from_file(path) -> mean‖std [2D].

audio_whisper : openai/whisper-large-v3-turbo encoder (680k h, ~100 languages), frames of the actual audio, [T, 1280] -> 2560
audio_xlsr    : facebook/wav2vec2-xls-r-300m (436k h, 128 languages, self-supervised), mean of the upper half of the
                transformer layers (paralinguistic information sits in the middle/upper layers), [T, 1024] -> 2048
audio_w2v_emo : audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim (English emotion fine-tune), same pooling -> 2048
audio_egemaps : openSMILE eGeMAPSv02 functionals, 88 hand-crafted prosody/voice-quality features (no learning;
                standardised afterwards with train statistics, see scripts/standardize_features.py)
audio_e2v     : emotion2vec+ large, extracted by scripts/extract_e2v.py in its own venv (funasr), [T, 768] -> 1536
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .extractors import load_wav_mono, mean_std

AUDIO_FEATURE_DIMS = {"audio_whisper": 2560, "audio_xlsr": 2048, "audio_w2v_emo": 2048, "audio_egemaps": 88,
                      "audio_e2v": 1536}


class WhisperEncoderAudio:
    def __init__(self, device="cuda", name="openai/whisper-large-v3-turbo"):
        from transformers import WhisperFeatureExtractor, WhisperModel
        self.fe = WhisperFeatureExtractor.from_pretrained(name)
        dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        self.model = WhisperModel.from_pretrained(name, torch_dtype=dtype)
        self.enc = self.model.get_encoder().to(device).eval()
        self.device, self.sample_rate, self.dtype = device, 16000, dtype

    @torch.no_grad()
    def sequence(self, wav: np.ndarray) -> torch.Tensor:
        feats = self.fe(wav, sampling_rate=self.sample_rate, return_tensors="pt").input_features
        out = self.enc(feats.to(self.device, dtype=self.dtype)).last_hidden_state[0]     # [1500, 1280] = 30 s at 50 Hz
        n = min(out.shape[0], max(1, int(math.ceil(len(wav) / self.sample_rate * 50))))
        return out[:n].float().cpu()

    def from_file(self, wav_path: str) -> torch.Tensor:
        return mean_std(self.sequence(load_wav_mono(wav_path, self.sample_rate)))


class Wav2Vec2Audio:
    def __init__(self, device="cuda", name="facebook/wav2vec2-xls-r-300m", max_seconds: float = 40.0):
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(name)
        self.model = Wav2Vec2Model.from_pretrained(name).to(device).eval()      # fp32: wav2vec2 is fragile in fp16
        self.device, self.sample_rate, self.max_len = device, 16000, int(max_seconds * 16000)

    @torch.no_grad()
    def sequence(self, wav: np.ndarray) -> torch.Tensor:
        wav = wav[: self.max_len]
        x = self.fe(wav, sampling_rate=self.sample_rate, return_tensors="pt").input_values.to(self.device)
        hs = self.model(x, output_hidden_states=True).hidden_states          # (L+1) x [1, T, H]
        k = len(hs) // 2
        return torch.stack(hs[k:], 0).mean(0)[0].float().cpu()                # [T, H]

    def from_file(self, wav_path: str) -> torch.Tensor:
        return mean_std(self.sequence(load_wav_mono(wav_path, self.sample_rate)))


class EgemapsAudio:
    def __init__(self, device=None):
        import opensmile
        self.smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                                     feature_level=opensmile.FeatureLevel.Functionals)
        self.sample_rate = 16000

    def from_file(self, wav_path: str) -> torch.Tensor:
        wav = load_wav_mono(wav_path, self.sample_rate)
        df = self.smile.process_signal(wav, self.sample_rate)
        v = torch.tensor(df.to_numpy()[0], dtype=torch.float32)                # [88]
        return torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)


AUDIO_ENCODERS = {
    "audio_whisper": lambda device: WhisperEncoderAudio(device),
    "audio_xlsr": lambda device: Wav2Vec2Audio(device, "facebook/wav2vec2-xls-r-300m"),
    "audio_w2v_emo": lambda device: Wav2Vec2Audio(device, "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"),
    "audio_egemaps": lambda device: EgemapsAudio(),
}
