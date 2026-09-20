"""FIV2 index, audio extraction and per-modality feature cache.

Layout (WSL ext4, configurable):
  ROOT/video/{train,dev,test}/<name>.mp4       clips (HF "validation" == "dev")
  ROOT/audio/{train,dev,test}/<name>.wav       48 kHz mono, extracted by ffmpeg
  ROOT/features/{split}/<modality>.pt          {"names": [...], "x": FloatTensor[N, D], "stats": {...}}
Labels, Rev transcripts and Qwen3-VL behaviour descriptions come from MM-PSYCHE data/fiv2/*.csv.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

log = logging.getLogger("bs.mm.data")

SPLITS = ("train", "dev", "test")
LABEL_COLUMNS = ["openness", "conscientiousness", "extraversion", "agreeableness", "non-neuroticism"]
INTERVIEW_COLUMN = "interview"          # ChaLearn job-interview variable, from interview_labels.csv (HF mirror)
TARGET_SETS = {"big5": LABEL_COLUMNS, "big5+interview": LABEL_COLUMNS + [INTERVIEW_COLUMN]}
MODALITIES = ("face", "audio", "text", "behavior")

_HERE = Path(__file__).resolve()
DEFAULT_CSV_DIR = _HERE.parents[3] / "MM-PSYCHE" / "data" / "fiv2"       # BS/MM-PSYCHE/data/fiv2
INTERVIEW_CSV = _HERE.parent / "interview_labels.csv"
DEFAULT_ROOT = Path(os.path.expanduser("~/data/fiv2"))


def load_split_table(split: str, csv_dir: Path = DEFAULT_CSV_DIR, drop_empty_text: bool = True,
                     with_interview: bool = False) -> pd.DataFrame:
    """MM-PSYCHE reads the csv with pandas and drops rows with any NaN (= empty transcript).
    with_interview=True merges the ChaLearn interview label (scripts/fetch_interview_labels.py)."""
    df = pd.read_csv(csv_dir / f"{split}_full_with_description.csv")
    if drop_empty_text:
        df = df.dropna(subset=["text", "text_llm"] + LABEL_COLUMNS)
    if with_interview:
        if not INTERVIEW_CSV.exists():
            raise FileNotFoundError(f"{INTERVIEW_CSV} missing: run scripts/fetch_interview_labels.py")
        iv = pd.read_csv(INTERVIEW_CSV)[["video_name", INTERVIEW_COLUMN]]
        df = df.merge(iv, on="video_name", how="left").dropna(subset=[INTERVIEW_COLUMN])
    return df.reset_index(drop=True)


def video_path(root: Path, split: str, name: str) -> Path:
    return root / "video" / split / f"{name}.mp4"


def audio_path(root: Path, split: str, name: str) -> Path:
    return root / "audio" / split / f"{name}.wav"


def extract_audio(root: Path, split: str, names, sr: int = 48000, workers: int = 8) -> int:
    """ffmpeg -> mono wav at `sr` for every clip that has no wav yet. Returns number of files written."""
    out_dir = root / "audio" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [n for n in names if not audio_path(root, split, n).exists()]

    def one(n):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video_path(root, split, n)), "-vn", "-ac", "1",
               "-ar", str(sr), "-acodec", "pcm_s16le", str(audio_path(root, split, n))]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            log.warning("ffmpeg failed for %s: %s", n, r.stderr.strip()[:200])
            return 0
        return 1

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        done = sum(ex.map(one, todo))
    log.info("[%s] audio: %d written, %d existed, %.0fs", split, done, len(names) - len(todo), time.time() - t0)
    return done


def feature_file(root: Path, split: str, modality: str) -> Path:
    return root / "features" / split / f"{modality}.pt"


def save_features(root: Path, split: str, modality: str, names: list, x: torch.Tensor, stats: dict):
    p = feature_file(root, split, modality)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".pt.tmp")
    torch.save({"names": list(names), "x": x.float().contiguous(), "stats": stats}, tmp)
    os.replace(tmp, p)
    log.info("[%s] saved %s features %s -> %s", split, modality, tuple(x.shape), p)


def load_features(root: Path, split: str, modality: str) -> dict:
    return torch.load(feature_file(root, split, modality), map_location="cpu")


class FeatureTable:
    """Aligned features for one split: names, labels [N, T], dict modality -> [N, D] (present modalities only).
    targets: "big5" (5 traits) or "big5+interview" (6 outputs)."""

    def __init__(self, root: Path, split: str, modalities, csv_dir: Path = DEFAULT_CSV_DIR, targets: str = "big5"):
        self.target_columns = TARGET_SETS[targets]
        df = load_split_table(split, csv_dir, with_interview=(INTERVIEW_COLUMN in self.target_columns))
        feats = {m: load_features(root, split, m) for m in modalities}
        # keep clips present in every requested modality
        keep = set(df["video_name"])
        for m, f in feats.items():
            keep &= set(f["names"])
        df = df[df["video_name"].isin(keep)].reset_index(drop=True)
        self.names = df["video_name"].tolist()
        self.labels = torch.tensor(df[self.target_columns].to_numpy(dtype=np.float32))
        self.x = {}
        for m, f in feats.items():
            idx = {n: i for i, n in enumerate(f["names"])}
            self.x[m] = f["x"][[idx[n] for n in self.names]]
        self.split = split
        log.info("[%s] feature table: %d clips, modalities %s", split, len(self.names),
                 {m: tuple(v.shape) for m, v in self.x.items()})

    def __len__(self):
        return len(self.names)

    def batch(self, idx) -> dict:
        return {"features": {m: v[idx] for m, v in self.x.items()}, "labels": self.labels[idx]}
