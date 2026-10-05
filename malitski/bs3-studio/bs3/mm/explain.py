"""Explanations for the own fusion model (stage 2 / web UI):

- modality attribution: Input x Gradient per modality vector (as in the SSL-MEPR prototype) and
  leave-one-modality-out deltas of every output;
- key frames: gradient of an output w.r.t. the 30 per-frame CLIP embeddings through the mean‖std pooling,
  top-k frames by |grad * embedding|, with the sign of the contribution kept beside the magnitude, so a caption
  can say whether the frame raised or lowered the score;
- token attribution: gradient of an output w.r.t. the EmoRoBERTa input token embeddings (transcript and
  behaviour description), aggregated to words.

All functions work on one clip; `explain_clip` runs everything and returns a JSON-serialisable dict.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

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
    """face_seq: [T, 512] per-frame CLIP embeddings. Returns per-output frame importances and top-k frame ids.

    `importance` is |grad * embedding| summed over the 512 dimensions, as before (old jobs and the report keep
    reading it). `signed` is the same sum without the absolute value: positive means the frame pushed that output
    up, negative means it pushed it down. `per_frame_effect` names, for every frame, the two of the five traits it
    moved most, with the direction — this is what the caption under the key frame prints."""
    model.eval()
    seq = face_seq.detach().clone().to(device).requires_grad_(True)
    inputs = {"face": mean_std(seq).unsqueeze(0)}
    inputs.update({m: v.unsqueeze(0).to(device) for m, v in other_feats.items() if m != "face"})
    out = model(inputs)[0]
    keys = _keys(out.numel())
    res = {}
    for i, k in enumerate(keys):
        g, = torch.autograd.grad(out[i], seq, retain_graph=True)
        signed = (g * seq).sum(dim=1).detach().cpu().numpy()       # [T]
        imp = np.abs(signed)
        order = np.argsort(-imp)[:top_k].tolist()
        res[k] = {"importance": [round(float(v), 5) for v in imp], "signed": [round(float(v), 5) for v in signed],
                  "top_frames": order}
    # frames that matter across all traits
    mean_imp = np.mean([np.array(res[k]["importance"]) for k in keys], axis=0)
    traits = [k for k in keys if k in TRAIT_KEYS]
    effect = []
    for t in range(int(face_seq.shape[0])):
        ranked = sorted(((k, res[k]["signed"][t]) for k in traits), key=lambda kv: -abs(kv[1]))
        effect.append([{"output": k, "signed": v} for k, v in ranked[:2]])
    return {"per_output": res, "per_frame_effect": effect, "top_frames_overall": np.argsort(-mean_imp)[:top_k].tolist(),
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


def clip_fps(video_path: str) -> Optional[float]:
    """Frames per second of the clip the explanation ran on, or None.

    A key frame is named by its number inside this clip, and the caption turns that number into a moment of the
    video: on a source with a variable frame rate the average rate of the whole file is not that rate, and the
    printed second comes out wrong by one."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    v = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    cap.release()
    return round(v, 3) if v > 0 else None


def save_key_frames(video_path: str, frame_ids: List[int], n_frames: int, out_dir: Path, prefix: str = "key",
                    raw_jpegs: Optional[Dict[str, str]] = None, raw_max_side: int = 640,
                    kept_frames: Optional[Sequence[int]] = None) -> List[str]:
    """Re-decode the clip, take the frames the face features were built from, and write the requested ones
    (with the detected face box) as JPEG. Returns the written paths.

    `frame_ids` are positions in the sequence of face crops, and `kept_frames` (faces.get_face_crops stats
    `frames_kept`) says which frame of the clip every crop came from. Without it the positions are read as
    positions in the uniform sampling, which is the same thing only when no leading frame was dropped for want of
    a face — a clip that opens on an empty chair would otherwise show one frame and describe another.

    `raw_jpegs`, when given, is filled with {file name: base64 JPEG of the same frame WITHOUT the drawn box,
    downscaled to `raw_max_side`}: the caption asks a vision model what is visible on the frame, and a yellow
    rectangle drawn over the face is not part of what was filmed."""
    import base64
    import cv2
    from .faces import detect_faces, select_uniform_frames
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    uniform = select_uniform_frames(total, n_frames)
    sampled = list(kept_frames) if kept_frames else uniform
    if len(sampled) != len(uniform):        # the silent case made visible: the clip opens without a face
        log.info("key frames: %d crops for %d sampled frames", len(sampled), len(uniform))
    wanted = {sampled[i]: i for i in frame_ids if i < len(sampled)}
    out_dir.mkdir(parents=True, exist_ok=True)
    paths, t = [], 0
    while wanted:
        ok, im = cap.read()
        if not ok:
            break
        if t in wanted:
            name = f"{prefix}_{wanted[t]:02d}_frame{t}.jpg"
            if raw_jpegs is not None:
                small = im
                if max(im.shape[:2]) > raw_max_side:
                    s = raw_max_side / max(im.shape[:2])
                    small = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
                ok2, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok2:
                    raw_jpegs[name] = base64.b64encode(buf.tobytes()).decode("ascii")
            rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            # the frame is shown downscaled (≈640 px on the web, ≈57 mm in the PDF): the line thickness follows the
            # long side of the frame (as in BS 1.0: 5 px core at 1280 px, never below 3 px) so the box survives the
            # downscaling, and a black edge as wide as the core on each side keeps the yellow line visible on light
            # walls and on dark clothes alike
            th = max(3, round(max(im.shape[:2]) / 240))
            for (x1, y1, x2, y2, _) in detect_faces(rgb):
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 0), 3 * th, cv2.LINE_AA)
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 255), th, cv2.LINE_AA)
            p = out_dir / name
            cv2.imwrite(str(p), im)
            paths.append(str(p))
            del wanted[t]
        t += 1
    cap.release()
    return paths


def key_frame_info(paths: Sequence[str], crops: Optional[Sequence] = None, raw_jpegs: Optional[Dict[str, str]] = None,
                   expression_fn: Optional[Callable] = None, phrase_fn: Optional[Callable] = None) -> List[dict]:
    """What the caption under every key frame needs, computed once and stored in the job's explanation.json.

    For every saved frame: its index in the sampled sequence, the two strongest facial expressions of THAT frame
    (`expression_fn` — the 7-class model the rest of the report uses, called on the face crop the features were
    built from) and a short phrase about what is visible (`phrase_fn` — the local vision model, one request per
    frame, one retry, then the frame keeps no phrase and the caption falls back to the expression).
    Both helpers are optional; a failure of either never breaks the explanation."""
    from .. import settings
    from ..frame_phrase import PHRASE_TRIES, clean_phrase
    from ..labels import EXPR_RU

    info = [{"file": Path(p).name, "frame": _key_index(p)} for p in paths]
    if expression_fn is not None and crops is not None:
        t0 = time.time()
        pos = [(n, rec["frame"]) for n, rec in enumerate(info) if rec["frame"] is not None and rec["frame"] < len(crops)]
        if pos:
            try:
                dists = expression_fn([crops[i] for _, i in pos])
                for (n, _), d in zip(pos, dists):
                    top = sorted(d.items(), key=lambda kv: -kv[1])[:2]
                    info[n]["expressions"] = [{"label": lab, "ru": EXPR_RU.get(lab, lab), "share": round(float(v), 4)}
                                              for lab, v in top]
            except Exception as e:  # noqa: BLE001
                log.warning("key-frame expressions failed: %s", str(e).splitlines()[0][:120])
        log.info("key-frame expressions in %.1fs", time.time() - t0)
    if phrase_fn is not None and raw_jpegs:
        t0 = time.time()
        for rec in info:
            b64 = raw_jpegs.get(rec["file"])
            if not b64:
                continue
            if time.time() - t0 > settings.PHRASE_BUDGET:   # the captions never hold the job: the rest keep none
                log.warning("key-frame phrases: out of the %.0fs budget, the remaining frames keep none",
                            settings.PHRASE_BUDGET)
                break
            for _ in range(PHRASE_TRIES):
                try:
                    phrase = clean_phrase(phrase_fn(b64))
                except Exception as e:  # noqa: BLE001
                    log.warning("key-frame phrase failed: %s", str(e).splitlines()[0][:120])
                    break
                if phrase:
                    rec["phrase"] = phrase
                    break
        log.info("key-frame phrases in %.1fs (%d of %d kept)", time.time() - t0,
                 sum(1 for r in info if r.get("phrase")), len(info))
    return info


def _key_index(path: str) -> Optional[int]:
    from ..frame_phrase import _frame_index
    return _frame_index(path)
