"""Emotion in the voice, independent of the words: audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim predicts
arousal (calm ... excited), dominance (submissive ... dominant) and valence (negative ... positive) in 0…1 from
16 kHz audio. Trained on English podcasts (MSP-Podcast); the acoustic cues transfer across languages reasonably,
but the numbers are relative, not calibrated for Russian speakers."""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from ..mm.extractors import load_wav_mono

DIMS = ["arousal", "dominance", "valence"]
DIMS_RU = {"arousal": "возбуждение", "dominance": "уверенность (доминантность)", "valence": "позитивность (валентность)"}


class _RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features):
        x = self.dropout(features)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)


def _build(name: str, device: str):
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2PreTrainedModel
    from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model

    class EmotionModel(Wav2Vec2PreTrainedModel):          # architecture from the audeering model card
        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = _RegressionHead(config)
            self.init_weights()

        def forward(self, input_values):
            hidden = self.wav2vec2(input_values)[0]
            pooled = torch.mean(hidden, dim=1)
            return self.classifier(pooled)

    fe = Wav2Vec2FeatureExtractor.from_pretrained(name)
    model = EmotionModel.from_pretrained(name).to(device).eval()
    return fe, model


class VoiceEmotion:
    def __init__(self, device: str = "cuda", name: str = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
                 window_s: float = 10.0):
        self.fe, self.model = _build(name, device)
        self.device, self.sample_rate, self.window = device, 16000, int(window_s * 16000)

    @torch.no_grad()
    def from_wav(self, wav: np.ndarray) -> Dict[str, float]:
        """Mean over ≤10-s windows (the model was trained on short utterances)."""
        if wav.size < 1600:
            return {d: float("nan") for d in DIMS}
        outs = []
        for i in range(0, max(1, wav.size - 1), self.window):
            chunk = wav[i:i + self.window]
            if chunk.size < 8000:                       # < 0.5 s tail: skip
                continue
            x = self.fe(chunk, sampling_rate=self.sample_rate, return_tensors="pt").input_values.to(self.device)
            outs.append(self.model(x)[0].float().cpu().numpy())
        if not outs:
            return {d: float("nan") for d in DIMS}
        m = np.clip(np.mean(outs, axis=0), 0.0, 1.0)
        return {d: round(float(v), 4) for d, v in zip(DIMS, m)}

    def from_file(self, path: str) -> Dict[str, float]:
        return self.from_wav(load_wav_mono(path, self.sample_rate))
