"""The one-model wrapper of BS Profiler 3.1: the page runs exactly one Big Five model per analysis, OCEAN-AI ("oceanai")
or AMLAI 1.0 ("mm"), and this backend wraps that model so that the segment analyzer and result.json keep the shape the
earlier versions with several members wrote.

Output of `predict_video`: `scores` (the member's own numbers), `seconds`, `variants` ({member: its scores}),
`members_used` ([member]), `members_failed` ({}), `primary` and `primary_used` (the member), `transcript`, and for
AMLAI 1.0 also the `interview` score (inside `scores`), `behavior_description` and `timings`. A failure of the model is
raised as «all ensemble members failed: <member>: <first line of the error>»; the page maps that message.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from . import LANG
from .errors import is_fatal
from .norms import TRAIT_KEYS

log = logging.getLogger("bs.ensemble")


@dataclass
class EnsembleConfig:
    members: tuple                # exactly one model key: ("oceanai",) or ("mm",)
    oceanai_cfg: object = None
    mm_cfg: object = None
    lang: str = LANG
    primary: str | None = None    # the member itself (None = the member); any other value is refused

    def __post_init__(self):
        self.members = tuple(self.members)
        if len(self.members) != 1:
            raise ValueError(f"one model per analysis: members must hold exactly one model, got {self.members}")
        if self.primary is None:
            self.primary = self.members[0]
        elif self.primary != self.members[0]:
            raise ValueError(f"primary member {self.primary!r} is not the model {self.members[0]!r}")

    @property
    def corpus(self):
        base = "ensemble(" + "+".join(self.members) + ")"
        return f"{base}, main={self.primary}" if self.primary else base


class EnsembleBackend:
    def __init__(self, cfg: EnsembleConfig):
        self.cfg = cfg
        self.backends = {}
        self.load_seconds = 0.0
        m = cfg.members[0]
        if m == "oceanai":
            from .backend_oceanai import BackendConfig, OceanAIBackend
            self.backends[m] = OceanAIBackend(cfg.oceanai_cfg or BackendConfig(lang=cfg.lang))
        elif m == "mm":
            from .backend_mm import MMBackend, MMConfig
            self.backends[m] = MMBackend(cfg.mm_cfg or MMConfig(lang=cfg.lang))
        else:
            raise ValueError(f"unknown ensemble member {m!r}")

    def load(self):
        t0 = time.time()
        for b in self.backends.values():
            b.load()
        self.load_seconds = time.time() - t0
        return self

    def predict_video(self, video, asr: bool = True, transcript: str | None = None,
                      behavior: str | None = None) -> dict:
        """`behavior`: a ready behaviour description for the own model (skips Ollama), e.g. when re-running."""
        t0 = time.time()
        name, b = next(iter(self.backends.items()))
        failed = {}
        try:
            if name == "mm":
                r = b.predict_video(video, asr=asr, transcript=transcript, behavior=behavior)
            else:
                r = b.predict_video(video, asr=asr, transcript=transcript)
        except Exception as e:  # e.g. OCEAN-AI drops a clip without speech
            if is_fatal(e):     # a broken Ollama or GPU: fail the whole run now, do not report it as «member failed»
                raise
            failed[name] = str(e).splitlines()[0][:160]
            log.warning("ensemble member %s failed on %s: %s", name, Path(video).name, failed[name])
        if failed:
            raise RuntimeError("all ensemble members failed: " + "; ".join(f"{k}: {v}" for k, v in failed.items()))
        primary = self.cfg.primary
        # the member's numbers as they are (float() only), no mean over members
        scores = {k: float(r["scores"][k]) for k in TRAIT_KEYS}
        out = {"scores": scores, "seconds": round(time.time() - t0, 2), "variants": {name: r["scores"]},
               "members_used": [name], "members_failed": failed, "primary": primary,
               "primary_used": primary if primary == name else None,
               "transcript": r.get("transcript") or ""}
        if transcript is not None and not out["transcript"]:
            out["transcript"] = transcript
        if name == "mm":
            if "interview" in r["scores"]:
                out["scores"]["interview"] = r["scores"]["interview"]     # only the own model predicts it
            out["behavior_description"] = r.get("behavior_description", "")
            out["timings"] = {f"mm_{k}": v for k, v in r.get("timings", {}).items()}
        return out
