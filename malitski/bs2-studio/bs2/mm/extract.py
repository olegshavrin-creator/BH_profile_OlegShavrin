"""Extract and cache MM-PSYCHE-style features for one FIV2 split.

  python -m bs2.mm.extract --split train --modalities audio,text,behavior
  python -m bs2.mm.extract --split train --modalities face --shard 0/6     # one of 6 parallel shards
  python -m bs2.mm.extract --split train --modalities face --merge         # merge shard files

Shard files: features/<split>/<modality>.shard<i>of<n>.pt ; merged: features/<split>/<modality>.pt
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch

from .data import DEFAULT_ROOT, extract_audio, feature_file, load_split_table, save_features, video_path, audio_path
from .extractors import ClapAudioEncoder, ClipFaceEncoder, EmoRobertaTextEncoder, FEATURE_DIMS
from .extractors_audio import AUDIO_ENCODERS, AUDIO_FEATURE_DIMS
from .faces import get_face_crops

ALL_DIMS = {**FEATURE_DIMS, **AUDIO_FEATURE_DIMS}

log = logging.getLogger("bs.mm.extract")


def shard_file(root: Path, split: str, modality: str, i: int, n: int) -> Path:
    return feature_file(root, split, modality).with_name(f"{modality}.shard{i}of{n}.pt")


def _make_encoder(factory, device: str, attempts: int = 6):
    """Several shards initialising CUDA at the same moment can hit 'CUDA-capable device(s) is/are busy or
    unavailable'; retry with a growing pause instead of losing the shard."""
    for i in range(1, attempts + 1):
        try:
            enc = factory(device)
            if device.startswith("cuda"):
                torch.zeros(1, device=device)  # force context creation now
            return enc
        except Exception as e:  # noqa: BLE001
            if i == attempts or "busy or unavailable" not in str(e):
                raise
            log.warning("CUDA init attempt %d failed (%s); retrying in %ds", i, str(e).splitlines()[0][:80], 5 * i)
            time.sleep(5 * i)


def run_modality(root: Path, split: str, modality: str, names: list, texts: dict, device: str, n_frames: int) -> tuple:
    """Returns (names_done, x [N, D], stats)."""
    t0 = time.time()
    xs, done, stats = [], [], {"failed": [], "no_face_clips": 0, "fallback_frames": 0}
    if modality == "face":
        enc = _make_encoder(ClipFaceEncoder, device)
        for i, n in enumerate(names, 1):
            try:
                crops, st = get_face_crops(str(video_path(root, split, n)), n_frames=n_frames)
                if not crops:
                    raise RuntimeError("no frames")
                if st["frames_with_face"] == 0:
                    stats["no_face_clips"] += 1
                stats["fallback_frames"] += st["fallback_frames"]
                xs.append(enc(crops))
                done.append(n)
            except Exception as e:
                stats["failed"].append(n)
                log.warning("face %s failed: %s", n, e)
            if i % 100 == 0:
                log.info("[%s/face] %d/%d (%.2f s/clip)", split, i, len(names), (time.time() - t0) / i)
    elif modality == "audio":
        enc = _make_encoder(ClapAudioEncoder, device)
        extract_audio(root, split, names, sr=enc.sample_rate)
        for i, n in enumerate(names, 1):
            try:
                xs.append(enc.from_file(str(audio_path(root, split, n))))
                done.append(n)
            except Exception as e:
                stats["failed"].append(n)
                log.warning("audio %s failed: %s", n, e)
            if i % 200 == 0:
                log.info("[%s/audio] %d/%d (%.2f s/clip)", split, i, len(names), (time.time() - t0) / i)
    elif modality in AUDIO_ENCODERS:                    # candidate speech encoders (see extractors_audio.py)
        enc = _make_encoder(AUDIO_ENCODERS[modality], device)
        extract_audio(root, split, names, sr=48000)      # the stored wavs are 48 kHz; encoders resample themselves
        for i, n in enumerate(names, 1):
            try:
                xs.append(enc.from_file(str(audio_path(root, split, n))))
                done.append(n)
            except Exception as e:
                stats["failed"].append(n)
                log.warning("%s %s failed: %s", modality, n, e)
            if i % 200 == 0:
                log.info("[%s/%s] %d/%d (%.2f s/clip)", split, modality, i, len(names), (time.time() - t0) / i)
    elif modality in ("text", "behavior"):
        enc = _make_encoder(EmoRobertaTextEncoder, device)
        col = "text" if modality == "text" else "text_llm"
        for i, n in enumerate(names, 1):
            try:
                xs.append(enc(str(texts[col][n])))
                done.append(n)
            except Exception as e:
                stats["failed"].append(n)
                log.warning("%s %s failed: %s", modality, n, e)
            if i % 500 == 0:
                log.info("[%s/%s] %d/%d", split, modality, i, len(names))
    else:
        raise ValueError(modality)
    x = torch.stack(xs) if xs else torch.zeros(0, ALL_DIMS.get(modality, 1))
    stats.update({"seconds": round(time.time() - t0, 1), "n": len(done)})
    return done, x, stats


def merge_shards(root: Path, split: str, modality: str):
    files = sorted(feature_file(root, split, modality).parent.glob(f"{modality}.shard*of*.pt"))
    if not files:
        raise FileNotFoundError(f"no shard files for {split}/{modality}")
    names, xs, stats = [], [], {"shards": len(files), "failed": [], "no_face_clips": 0, "fallback_frames": 0}
    for f in files:
        d = torch.load(f, map_location="cpu")
        names += d["names"]
        xs.append(d["x"])
        for k in ("failed",):
            stats[k] += d["stats"].get(k, [])
        for k in ("no_face_clips", "fallback_frames"):
            stats[k] += d["stats"].get(k, 0)
    save_features(root, split, modality, names, torch.cat(xs), stats)
    log.info("[%s/%s] merged %d shards -> %d clips, failed %d, no-face clips %d",
             split, modality, len(files), len(names), len(stats["failed"]), stats["no_face_clips"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--split", required=True, choices=["train", "dev", "test"])
    ap.add_argument("--modalities", default="face,audio,text,behavior")
    ap.add_argument("--shard", default=None, help="i/n : process the i-th of n shards (face)")
    ap.add_argument("--merge", action="store_true", help="merge shard files instead of extracting")
    ap.add_argument("--n-frames", type=int, default=30)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads for OpenCV/torch in this process")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import cv2
    cv2.setNumThreads(args.threads)
    torch.set_num_threads(args.threads)

    root = Path(args.root)
    mods = [m for m in args.modalities.split(",") if m]
    if args.merge:
        for m in mods:
            merge_shards(root, args.split, m)
        return
    df = load_split_table(args.split)
    names = df["video_name"].tolist()
    texts = {"text": dict(zip(df["video_name"], df["text"])), "text_llm": dict(zip(df["video_name"], df["text_llm"]))}
    if args.limit:
        names = names[: args.limit]
    i, n = (0, 1)
    if args.shard:
        i, n = (int(v) for v in args.shard.split("/"))
        names = names[i::n]
    for m in mods:
        done, x, stats = run_modality(root, args.split, m, names, texts, args.device, args.n_frames)
        if args.shard:
            p = shard_file(root, args.split, m, i, n)
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"names": done, "x": x, "stats": stats}, p)
            log.info("[%s/%s] shard %d/%d saved: %d clips, %s", args.split, m, i, n, len(done), p)
        else:
            save_features(root, args.split, m, done, x, stats)
        log.info("[%s/%s] stats: %s", args.split, m, {k: v for k, v in stats.items() if k != "failed"})


if __name__ == "__main__":
    main()
