"""Media file metadata via ffprobe (+ file system facts) for reports, and the upload check that refuses a file that
cannot be analysed before any job folder or model is made (bs3.pipeline.run_analysis and the CLI)."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from . import errors


def probe_media(path: str | Path, with_hash: bool = True) -> dict:
    p = Path(path)
    meta = {"file_name": p.name, "size_bytes": p.stat().st_size, "size_mb": round(p.stat().st_size / 1e6, 1),
            "modified": _dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")}
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(p)],
                             capture_output=True, text=True, check=True).stdout
        info = json.loads(out)
    except Exception as e:  # noqa: BLE001
        meta["ffprobe_error"] = str(e)[:200]
        return meta
    fmt = info.get("format", {})
    meta["container"] = fmt.get("format_long_name") or fmt.get("format_name")
    meta["duration_sec"] = round(float(fmt.get("duration", 0) or 0), 2)
    meta["bitrate_kbps"] = round(int(fmt.get("bit_rate", 0) or 0) / 1000)
    tags = fmt.get("tags", {}) or {}
    for k in ("creation_time", "encoder", "com.apple.quicktime.make", "com.apple.quicktime.model", "title"):
        if k in tags:
            meta[f"tag_{k}"] = tags[k]
    for s in info.get("streams", []):
        if s.get("codec_type") == "video" and "video_codec" not in meta:
            fr = s.get("avg_frame_rate", "0/1")
            try:
                num, den = fr.split("/")
                fps = round(float(num) / float(den), 2) if float(den) else 0.0
            except Exception:  # noqa: BLE001
                fps = 0.0
            meta.update({"video_codec": s.get("codec_name"), "width": s.get("width"), "height": s.get("height"),
                         "fps": fps, "frames": int(s.get("nb_frames", 0) or 0) or None,
                         "rotation": (s.get("tags", {}) or {}).get("rotate")})
        elif s.get("codec_type") == "audio" and "audio_codec" not in meta:
            meta.update({"audio_codec": s.get("codec_name"), "sample_rate": s.get("sample_rate"),
                         "channels": s.get("channels")})
    if with_hash:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        meta["sha256"] = h.hexdigest()
    return meta


# --- upload check: refuse a file that cannot be analysed, before any folder or model (bs3.pipeline, bs3.cli) ---
NOT_VIDEO = ("Файл не удалось прочитать как видео: возможно, он повреждён или это не видеозапись. "
             "Загрузите ролик в формате MP4, MOV, MKV или WEBM.")
NO_AUDIO = "В ролике нет звуковой дорожки, а для оценки нужна речь. Загрузите запись со звуком."
MIN_UPLOAD_SEC = 2.0             # a clip shorter than this cannot be scored (owner decision of 2026-09-27)
_DISK_HEADROOM = 1024 ** 3       # need twice the file plus this much free space to store the copy and the results


def _existing_dir(path) -> Path | None:
    """The first folder at or above `path` that exists (a new job folder is made under one that does not exist yet), or
    None when `path` is None or nothing above it exists."""
    if path is None:
        return None
    p = Path(path)
    while not p.exists():
        if p == p.parent:
            return None
        p = p.parent
    return p


def check_upload(path, work_dir) -> None:
    """Refuse an upload that cannot be analysed, so a bad file leaves no job folder behind and the person is told at once
    why. Raises errors.UserFacingError with a calm Russian sentence, shown as it is. Refused: not readable as video
    (ffprobe fails or there is no video stream), no audio track (the models need speech), shorter than two seconds, the
    results folder is not writable, or there is less free space than twice the file plus 1 GB. No upper size or duration
    limit. `work_dir` is where the job folder will be made: its disk and write permission are the ones checked."""
    p = Path(path)
    meta = probe_media(p, with_hash=False)
    if meta.get("ffprobe_error") or not meta.get("video_codec"):
        raise errors.UserFacingError(NOT_VIDEO)
    if not meta.get("audio_codec"):
        raise errors.UserFacingError(NO_AUDIO)
    dur = float(meta.get("duration_sec") or 0.0)
    if 0 < dur < MIN_UPLOAD_SEC:
        shown = f"{dur:.1f}".replace(".", ",")
        raise errors.UserFacingError(
            f"Ролик слишком короткий ({shown} с): для оценки нужно хотя бы 2 секунды, "
            "лучше — от 15 секунд речи в кадре.")
    base = _existing_dir(work_dir)
    if base is not None:
        if not os.access(base, os.W_OK):
            raise errors.UserFacingError(errors.NO_WRITE)
        try:
            free = shutil.disk_usage(base).free
        except OSError:
            free = None
        if free is not None and free < 2 * p.stat().st_size + _DISK_HEADROOM:
            raise errors.UserFacingError(errors.DISK)
