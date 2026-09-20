"""Media file metadata via ffprobe (+ file system facts) for reports."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import subprocess
from pathlib import Path


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
