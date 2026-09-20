"""Explanations for the own fusion model (stage 2 / web UI):

- modality attribution: Input x Gradient per modality vector (as in the SSL-MEPR prototype) and
  leave-one-modality-out deltas of every output;
- key frames: gradient of an output w.r.t. the 30 per-frame CLIP embeddings through the mean‖std pooling,
  top-k frames by |grad * embedding|;
- token attribution: gradient of an output w.r.t. the EmoRoBERTa input token embeddings (transcript and
  behaviour description), aggregated to words.

All functions work on one clip; `explain_clip` runs everything and returns a JSON-serialisable dict.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .extractors import mean_std
from ..norms import TRAIT_KEYS

log = logging.getLogger("bs.mm.explain")
OUTPUT_KEYS = TRAIT_KEYS + ["interview"]


def _keys(n: int) -> List[str]:
    return OUTPUT_KEYS[:n]


def modality_attribution(model, feats: Dict[str, torch.Tensor], device: str) -> dict:
    """feats: {modality: [D] pooled vector}. Returns per-output Input x Gradient share and leave-one-out deltas."""
    model.eval()
    inputs = {m: v.detach().clone().to(device).unsqueeze(0).requires_grad_(True) for m, v in feats.items()}
    out = model(inputs)[0]                                   # [T]
    keys = _keys(out.numel())
    ixg = {k: {} for k in keys}
    for i, k in enumerate(keys):
        grads = torch.autograd.grad(out[i], list(inputs.values()), retain_graph=True, allow_unused=True)
        raw = {}
        for (m, x), g in zip(inputs.items(), grads):
            raw[m] = float((x * g).sum().item()) if g is not None else 0.0
        tot = sum(abs(v) for v in raw.values()) + 1e-12
        ixg[k] = {m: {"signed": round(v, 5), "share": round(abs(v) / tot, 4)} for m, v in raw.items()}
    base = out.detach().cpu().numpy()
    loo = {}
    with torch.no_grad():
        if len(feats) > 1:
            for m in feats:
                sub = {n: v.unsqueeze(0).to(device) for n, v in feats.items() if n != m}
                delta = model(sub)[0].cpu().numpy() - base
                loo[m] = {k: round(float(delta[i]), 4) for i, k in enumerate(keys)}
    return {"scores": {k: round(float(base[i]), 4) for i, k in enumerate(keys)},
            "input_x_gradient": ixg, "leave_one_out_delta": loo}


def frame_attribution(model, face_seq: torch.Tensor, other_feats: Dict[str, torch.Tensor], device: str,
                      top_k: int = 5) -> dict:
    """face_seq: [T, 512] per-frame CLIP embeddings. Returns per-output frame importances and top-k frame ids."""
    model.eval()
    seq = face_seq.detach().clone().to(device).requires_grad_(True)
    inputs = {"face": mean_std(seq).unsqueeze(0)}
    inputs.update({m: v.unsqueeze(0).to(device) for m, v in other_feats.items() if m != "face"})
    out = model(inputs)[0]
    keys = _keys(out.numel())
    res = {}
    for i, k in enumerate(keys):
        g, = torch.autograd.grad(out[i], seq, retain_graph=True)
        imp = (g * seq).sum(dim=1).abs().detach().cpu().numpy()    # [T]
        order = np.argsort(-imp)[:top_k].tolist()
        res[k] = {"importance": [round(float(v), 5) for v in imp], "top_frames": order}
    # frames that matter across all traits
    mean_imp = np.mean([np.array(res[k]["importance"]) for k in keys], axis=0)
    return {"per_output": res, "top_frames_overall": np.argsort(-mean_imp)[:top_k].tolist(),
            "n_frames": int(face_seq.shape[0])}


def token_attribution(model, text_encoder, text: str, modality: str, other_feats: Dict[str, torch.Tensor],
                      device: str, top_k: int = 8) -> dict:
    """Input x Gradient on EmoRoBERTa token embeddings for `modality` in {"text", "behavior"}; words aggregated."""
    model.eval()
    tok, enc = text_encoder.tok, text_encoder.model
    inputs = tok(text or "", padding=True, truncation=True, return_tensors="pt", max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    emb_layer = enc.get_input_embeddings()
    embeds = emb_layer(inputs["input_ids"]).detach().clone().requires_grad_(True)
    hidden = enc(inputs_embeds=embeds, attention_mask=inputs["attention_mask"], return_dict=True).last_hidden_state
    valid = int(inputs["attention_mask"].sum().item())
    pooled = mean_std(hidden[0, :valid]).unsqueeze(0)
    feats = {modality: pooled}
    feats.update({m: v.unsqueeze(0).to(device) for m, v in other_feats.items() if m != modality})
    out = model(feats)[0]
    keys = _keys(out.numel())
    tokens = tok.convert_ids_to_tokens(inputs["input_ids"][0][:valid])
    res = {}
    for i, k in enumerate(keys):
        g, = torch.autograd.grad(out[i], embeds, retain_graph=True)
        tok_imp = (g[0, :valid] * embeds[0, :valid]).sum(dim=1).detach().cpu().numpy()
        # aggregate sub-word tokens (RoBERTa uses 'Ġ' for word starts) into words
        words, vals = [], []
        for t, v in zip(tokens, tok_imp):
            if t in ("<s>", "</s>", "<pad>"):
                continue
            if t.startswith("Ġ") or not words:
                words.append(t.lstrip("Ġ"))
                vals.append(float(v))
            else:
                words[-1] += t
                vals[-1] += float(v)
        order = np.argsort(-np.abs(vals))[:top_k]
        res[k] = {"top_words": [{"word": words[j].strip(" .,;:!?\"'()[]{}«»—-") or words[j], "signed": round(vals[j], 5)}
                                for j in order]}
    return {"per_output": res, "n_tokens": valid}


def save_key_frames(video_path: str, frame_ids: List[int], n_frames: int, out_dir: Path, prefix: str = "key") -> List[str]:
    """Re-decode the clip, take the uniformly sampled frames used for features, and write the requested ones
    (with the detected face box) as JPEG. Returns the written paths."""
    import cv2
    from .faces import detect_faces, select_uniform_frames
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sampled = select_uniform_frames(total, n_frames)
    wanted = {sampled[i]: i for i in frame_ids if i < len(sampled)}
    out_dir.mkdir(parents=True, exist_ok=True)
    paths, t = [], 0
    while wanted:
        ok, im = cap.read()
        if not ok:
            break
        if t in wanted:
            rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            # the frame is shown downscaled (≈640 px on the web, ≈57 mm in the PDF): the line thickness follows the
            # long side of the frame (as in BS 1.0: 5 px core at 1280 px, never below 3 px) so the box survives the
            # downscaling, and a black edge as wide as the core on each side keeps the yellow line visible on light
            # walls and on dark clothes alike
            th = max(3, round(max(im.shape[:2]) / 240))
            for (x1, y1, x2, y2, _) in detect_faces(rgb):
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 0), 3 * th, cv2.LINE_AA)
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 255), th, cv2.LINE_AA)
            p = out_dir / f"{prefix}_{wanted[t]:02d}_frame{t}.jpg"
            cv2.imwrite(str(p), im)
            paths.append(str(p))
            del wanted[t]
        t += 1
    cap.release()
    return paths
