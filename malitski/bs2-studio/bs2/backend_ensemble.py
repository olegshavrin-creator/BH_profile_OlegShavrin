"""Ensemble backend: equal-weight average of several systems.

Members (any subset, `--ensemble-members`):
  oceanai  - OCEAN-AI audio+video+text (stage 1)
  mm       - own MM-PSYCHE-style model (stage 2); also supplies the interview score and behaviour description
  scene    - SSL-MEPR scene branch (CLIP over the whole frame)
Full FIV2 test (1997 clips): oceanai+mm -> mACC 0.9270 / mCCC 0.734; oceanai+scene+mm -> 0.9267 / 0.735;
oceanai alone 0.9255 / 0.703 (see STAGE2_REPORT.md).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .norms import OCEANAI_COLUMNS, TRAIT_KEYS

log = logging.getLogger("bs.ensemble")


@dataclass
class EnsembleConfig:
    members: tuple = ("oceanai", "mm")
    oceanai_cfg: object = None
    mm_cfg: object = None
    sslmepr_cfg: object = None
    lang: str = "en"
    # which member gives the main score. "auto": OCEAN-AI for Russian speech (MuPTA weights = Russian reference
    # population; the own model is trained on English FIV2 vloggers and lives on another scale), otherwise the
    # equal-weight mean. None / "mean": always the mean. Other members stay available as second opinions.
    primary: str | None = "auto"

    def __post_init__(self):
        if self.primary == "auto":
            self.primary = "oceanai" if self.lang == "ru" and "oceanai" in self.members else None
        elif self.primary in (None, "", "none", "mean"):
            self.primary = None
        elif self.primary not in self.members:
            raise ValueError(f"primary member {self.primary!r} is not among {self.members}")

    @property
    def corpus(self):
        base = "ensemble(" + "+".join(self.members) + ")"
        return f"{base}, main={self.primary}" if self.primary else base


class EnsembleBackend:
    def __init__(self, cfg: EnsembleConfig):
        self.cfg = cfg
        self.backends = {}
        self.load_seconds = 0.0
        for m in cfg.members:
            if m == "oceanai":
                from .backend_oceanai import BackendConfig, OceanAIBackend
                self.backends[m] = OceanAIBackend(cfg.oceanai_cfg or BackendConfig(lang=cfg.lang))
            elif m == "mm":
                from .backend_mm import MMBackend, MMConfig
                self.backends[m] = MMBackend(cfg.mm_cfg or MMConfig(lang=cfg.lang))
            elif m == "scene":
                from .backend_sslmepr import SSLMEPRBackend, SSLMEPRConfig
                self.backends[m] = SSLMEPRBackend(cfg.sslmepr_cfg or SSLMEPRConfig(lang=cfg.lang, modalities=("scene",)))
            else:
                raise ValueError(f"unknown ensemble member {m!r}")

    def load(self):
        t0 = time.time()
        for b in self.backends.values():
            b.load()
        self.load_seconds = time.time() - t0
        return self

    @staticmethod
    def _member_scores(name: str, res: dict) -> dict:
        if name == "scene":
            return res["variants"]["scene"]
        return res["scores"]

    def predict_video(self, video, asr: bool = True, transcript: str | None = None,
                      behavior: str | None = None) -> dict:
        """`behavior`: a ready behaviour description for the own model (skips Ollama), e.g. when re-running."""
        t0 = time.time()
        results, variants, failed = {}, {}, {}
        for name, b in self.backends.items():
            try:
                if name == "scene":
                    r = b.predict_video(video, asr=False, transcript="")
                elif name == "mm":
                    r = b.predict_video(video, asr=asr, transcript=transcript, behavior=behavior)
                else:
                    r = b.predict_video(video, asr=asr, transcript=transcript)
            except Exception as e:  # e.g. OCEAN-AI drops a clip without speech; keep the other members
                failed[name] = str(e).splitlines()[0][:160]
                log.warning("ensemble member %s failed on %s: %s", name, Path(video).name, failed[name])
                continue
            results[name] = r
            variants[name] = self._member_scores(name, r)
        if not variants:
            raise RuntimeError("all ensemble members failed: " + "; ".join(f"{k}: {v}" for k, v in failed.items()))
        primary = self.cfg.primary
        if primary and primary in variants:
            scores = {k: float(variants[primary][k]) for k in TRAIT_KEYS}
        else:
            scores = {k: sum(v[k] for v in variants.values()) / len(variants) for k in TRAIT_KEYS}
            if primary:
                log.warning("primary member %s failed on %s; using the mean of %s", primary, Path(video).name, list(variants))
        out = {"scores": scores, "seconds": round(time.time() - t0, 2), "variants": variants,
               "members_used": list(variants), "members_failed": failed, "primary": primary,
               "primary_used": primary if primary in variants else None,
               "transcript": next((r["transcript"] for r in results.values() if r.get("transcript")), "")}
        if transcript is not None and not out["transcript"]:
            out["transcript"] = transcript
        if "mm" in results:
            mm = results["mm"]
            if "interview" in mm["scores"]:
                out["scores"]["interview"] = mm["scores"]["interview"]     # only the own model predicts it
            out["behavior_description"] = mm.get("behavior_description", "")
            out["timings"] = {f"mm_{k}": v for k, v in mm.get("timings", {}).items()}
        return out

    def predict_dir(self, directory, asr: bool = True, exts=None) -> pd.DataFrame:
        self.load()
        kw = {} if exts is None else {"exts": exts}
        frames = {}
        for name, b in self.backends.items():
            df = b.predict_dir(directory, asr=(asr if name != "scene" else False), **kw).set_index("Path")
            if name == "scene":
                df = df[[f"scene:{c}" for c in OCEANAI_COLUMNS]].rename(columns={f"scene:{c}": c for c in OCEANAI_COLUMNS})
            frames[name] = df
        common = sorted(set.intersection(*(set(f.index) for f in frames.values())))
        rows = []
        for path in common:
            row = {"Path": path}
            for c in OCEANAI_COLUMNS:
                vals = [float(frames[n].at[path, c]) for n in frames]
                row[c] = sum(vals) / len(vals)
                for n in frames:
                    row[f"{n}:{c}"] = float(frames[n].at[path, c])
            if "mm" in frames and "Interview" in frames["mm"].columns:
                row["Interview"] = float(frames["mm"].at[path, "Interview"])
            rows.append(row)
        missing = {n: len(f) - len(common) for n, f in frames.items()}
        if any(missing.values()):
            log.warning("ensemble: clips dropped because a member had no prediction: %s", missing)
        return pd.DataFrame(rows)
