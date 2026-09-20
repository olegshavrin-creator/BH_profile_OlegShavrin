"""Emotions expressed in what is said: michellejieli/emotion_text_classifier (DistilRoBERTa fine-tuned on
emotion-labelled dialogue) gives probabilities for anger, disgust, fear, joy, neutral, sadness, surprise.
The model is English-only, so Russian transcripts are translated first (same Marian model as the text branch)."""
from __future__ import annotations

from typing import Dict, List

import torch

EMOTIONS_RU = {"anger": "злость", "disgust": "отвращение", "fear": "страх", "joy": "радость", "neutral": "нейтрально",
               "sadness": "грусть", "surprise": "удивление"}
EMOTION_ORDER = ["joy", "surprise", "neutral", "sadness", "fear", "anger", "disgust"]


class TextEmotion:
    def __init__(self, device: str = "cuda", name: str = "michellejieli/emotion_text_classifier"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()
        self.device = device
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]

    @torch.no_grad()
    def __call__(self, text_en: str) -> Dict[str, float]:
        """Probabilities per emotion for one text (empty text -> neutral)."""
        if not text_en or not text_en.strip():
            return {e: (1.0 if e == "neutral" else 0.0) for e in self.labels}
        enc = self.tok(text_en, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        p = torch.softmax(self.model(**enc).logits[0], dim=-1).cpu().tolist()
        return {lab: round(float(v), 4) for lab, v in zip(self.labels, p)}

    def batch(self, texts: List[str]) -> List[Dict[str, float]]:
        return [self(t) for t in texts]


def dominant(probs: Dict[str, float]) -> str:
    return max(probs.items(), key=lambda kv: kv[1])[0] if probs else "neutral"
