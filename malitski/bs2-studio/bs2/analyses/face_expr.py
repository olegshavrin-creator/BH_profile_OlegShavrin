"""Facial expressions and face behaviour per segment.

Expressions: trpakov/vit-face-expression (ViT fine-tuned on FER-2013: angry, disgust, fear, happy, neutral, sad,
surprise) applied to MediaPipe face crops of uniformly sampled frames -> distribution over the segment.
Behaviour proxies from the face boxes: share of frames with a face, head-motion index (normalised movement of the
face centre between sampled frames), face size (closeness to the camera)."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from ..mm.faces import get_face_crops

EXPR_RU = {"angry": "злость", "disgust": "отвращение", "fear": "страх", "happy": "радость", "neutral": "нейтрально",
           "sad": "грусть", "surprise": "удивление"}
EXPR_ORDER = ["happy", "surprise", "neutral", "sad", "fear", "angry", "disgust"]


class FaceExpression:
    def __init__(self, device: str = "cuda", name: str = "trpakov/vit-face-expression", n_frames: int = 12):
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        self.proc = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModelForImageClassification.from_pretrained(name).to(device).eval()
        self.device, self.n_frames = device, n_frames
        self.labels = [self.model.config.id2label[i].lower() for i in range(self.model.config.num_labels)]

    @torch.no_grad()
    def on_crops(self, crops: List[np.ndarray]) -> List[Dict[str, float]]:
        out = []
        for i in range(0, len(crops), 16):
            batch = list(crops[i:i + 16])                       # get_face_crops returns RGB crops
            enc = self.proc(images=batch, return_tensors="pt").to(self.device)
            p = torch.softmax(self.model(**enc).logits, dim=-1).cpu().numpy()
            out += [{lab: float(v) for lab, v in zip(self.labels, row)} for row in p]
        return out

    def from_video(self, video: str) -> Dict:
        crops, st = get_face_crops(video, n_frames=self.n_frames)
        if not crops:
            return {"frames": 0, "face_share": 0.0, "expressions": {}, "dominant": None, "head_motion": None}
        per_frame = self.on_crops(crops)
        mean = {lab: round(float(np.mean([f[lab] for f in per_frame])), 4) for lab in self.labels}
        res = {"frames": len(crops), "expressions": mean, "dominant": max(mean.items(), key=lambda kv: kv[1])[0],
               "per_frame_dominant": [max(f.items(), key=lambda kv: kv[1])[0] for f in per_frame],
               "face_share": round(st.get("frames_with_face", len(crops)) / max(1, st.get("frames_used", len(crops))), 3)}
        boxes = st.get("boxes") or []
        if len(boxes) >= 2:
            c = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in boxes], dtype=float)
            size = np.array([max(1e-6, (b[2] - b[0])) for b in boxes], dtype=float)
            # mean displacement of the face centre between sampled frames, in units of face width
            res["head_motion"] = round(float(np.mean(np.linalg.norm(np.diff(c, axis=0), axis=1) / size[:-1])), 3)
            res["face_size"] = round(float(np.mean(size)), 3)          # share of the frame width
        return res
