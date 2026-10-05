"""Face crops from a video, as in MM-PSYCHE src/data_loading/video_preprocessor.py (detector "mp_fd").

Rules reproduced exactly:
- N frames chosen uniformly over the whole clip (np.linspace over frame indices);
- MediaPipe FaceDetection (model_selection=1, min_detection_confidence=0.5), loose boxes;
- faces smaller than relative_threshold * largest face area are dropped;
- several valid faces in one frame are averaged pixel-wise after resizing to the largest crop;
- no face: reuse the last valid crop; before the first valid face use the full frame, and those
  leading full frames are trimmed once a face has been seen.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import cv2
import numpy as np

log = logging.getLogger("bs.mm.faces")

_face_detection = None


def _detector():
    global _face_detection
    if _face_detection is None:
        import mediapipe as mp
        _face_detection = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    return _face_detection


def select_uniform_frames(total: int, n: int) -> List[int]:
    if total <= n:
        return list(range(total))
    return np.linspace(0, total - 1, num=n, dtype=int).tolist()


def detect_faces(frame_rgb: np.ndarray) -> List[Tuple[int, int, int, int, int]]:
    """Returns [(x1, y1, x2, y2, area)] in pixel coordinates."""
    results = _detector().process(frame_rgb)
    faces = []
    if results.detections:
        h, w = frame_rgb.shape[:2]
        for det in results.detections:
            bb = det.location_data.relative_bounding_box
            x1 = max(int(bb.xmin * w), 0)
            y1 = max(int(bb.ymin * h), 0)
            x2 = min(int((bb.xmin + bb.width) * w), w)
            y2 = min(int((bb.ymin + bb.height) * h), h)
            if x2 > x1 and y2 > y1:
                faces.append((x1, y1, x2, y2, (x2 - x1) * (y2 - y1)))
    return faces


def get_face_crops(video_path: str, n_frames: int = 30, relative_threshold: float = 0.3,
                   reuse_last: bool = True, fallback_fullframe: bool = True,
                   average_multi_face: bool = True) -> Tuple[List[np.ndarray], dict]:
    """Returns (list of RGB crops, stats). stats: frames_total, frames_used, frames_with_face, fallback_frames,
    crops, boxes and `frames_kept` — the number in the clip of the frame every crop came from, in the order of the
    crops. Leading frames with no face are dropped from the list, so the position of a crop is not the position of
    the sampled frame; anything that shows a crop (the key frames of an explanation) must go through `frames_kept`
    to name the right frame."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    need = set(select_uniform_frames(total, n_frames))
    crops: List[np.ndarray] = []
    kept: List[int] = []                       # the number in the clip of the frame every crop came from
    boxes: List[tuple] = []
    last_valid = None
    fallback_count = 0
    with_face = 0
    t = 0
    while True:
        ret, im0 = cap.read()
        if not ret:
            break
        if t in need:
            im_rgb = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
            faces = detect_faces(im_rgb)
            done = False
            if faces:
                max_area = max(a for *_, a in faces)
                valid = [(x1, y1, x2, y2) for (x1, y1, x2, y2, a) in faces if a >= relative_threshold * max_area]
                cs = [im_rgb[y1:y2, x1:x2] for (x1, y1, x2, y2) in valid if x2 > x1 and y2 > y1]
                if cs:
                    if average_multi_face and len(cs) > 1:
                        mh = max(c.shape[0] for c in cs)
                        mw = max(c.shape[1] for c in cs)
                        rs = [cv2.resize(c, (mw, mh), interpolation=cv2.INTER_AREA)
                              if (c.shape[0] != mh or c.shape[1] != mw) else c for c in cs]
                        crop = np.clip(np.mean(np.stack(rs), axis=0), 0, 255).astype(np.uint8)
                    else:
                        crop = max(cs, key=lambda c: c.shape[0] * c.shape[1])
                    crops.append(crop)
                    kept.append(t)
                    last_valid = crop
                    with_face += 1
                    done = True
                    bx = max(valid, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                    fh, fw = im_rgb.shape[:2]
                    boxes.append((bx[0] / fw, bx[1] / fh, bx[2] / fw, bx[3] / fh))     # normalised, largest face
            if not done:
                if reuse_last and last_valid is not None:
                    crops.append(last_valid)
                    kept.append(t)
                elif fallback_fullframe:
                    crops.append(im_rgb)
                    kept.append(t)
                    fallback_count += 1
        t += 1
    cap.release()
    if fallback_count and last_valid is not None:
        crops = crops[fallback_count:]
        kept = kept[fallback_count:]           # crops and kept are cut the same way: crops[i] is frame kept[i]
    stats = {"frames_total": total, "frames_used": len(need), "frames_with_face": with_face,
             "fallback_frames": fallback_count, "crops": len(crops), "boxes": boxes, "frames_kept": kept}
    return crops, stats
