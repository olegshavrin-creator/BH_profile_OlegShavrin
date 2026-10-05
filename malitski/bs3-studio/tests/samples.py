"""Truncated synthetic copies of the two sample jobs (tests/fixtures/samples.json): only numbers — model language and
primary system, trait scores, the means of each system, the spread and the segments (time, members used, scores
rounded to 4 digits). No names, paths, job ids, transcripts or fingerprints; nothing is read from the 2.0 work dir.

Sample A: 18 segments, OCEAN-AI missing on segment 16. Sample B: 33 segments, OCEAN-AI missing on 10, 11, 12, 14, 15,
26, 33 (the design's golden values, section 13.2).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from bs3.norms import TRAIT_KEYS

_DATA = json.loads((Path(__file__).resolve().parent / "fixtures" / "samples.json").read_text(encoding="utf-8"))


def rep(name: str) -> dict:
    """A result.json-like dict of sample 'A' or 'B' (a fresh copy on every call)."""
    return copy.deepcopy(_DATA[name])


def english(name: str = "B") -> dict:
    """The same numbers dressed as an English job: no primary, main score = mean of the two systems."""
    r = rep(name)
    r["model"]["lang"] = "en"
    r["model"]["primary"] = None
    return r


def own(name: str = "B") -> dict:
    """The numbers of a sample as a 3.1 job of AMLAI 1.0 (one member; every segment carries its scores)."""
    r = rep(name)
    r["model"].update({"selected": "mm", "primary": "mm", "selected_title": "AMLAI 1.0", "backend": "mm"})
    r["modalities_used"] = ["mm"]
    mm = {k: r["variant_scores"]["mm"][k] for k in TRAIT_KEYS}
    r["variant_scores"] = {"mm": r["variant_scores"]["mm"]}
    for t in r["timeline"]:
        t.update({"members_used": ["mm"], "primary_used": "mm", "variants": {"mm": dict(mm)}, "scores": dict(mm)})
    return r


def oceanai31(name: str = "B") -> dict:
    """The numbers of a sample as a 3.1 job of OCEAN-AI (one member; every segment carries its scores)."""
    r = rep(name)
    r["model"].update({"selected": "oceanai", "primary": "oceanai", "selected_title": "OCEAN-AI, веса MuPTA",
                       "backend": "oceanai"})
    r["modalities_used"] = ["audio", "video", "text"]
    oa = {k: r["variant_scores"]["oceanai"][k] for k in TRAIT_KEYS}
    r["variant_scores"] = {"oceanai": r["variant_scores"]["oceanai"]}
    for t in r["timeline"]:
        t.update({"members_used": ["oceanai"], "primary_used": "oceanai", "variants": {"oceanai": dict(oa)},
                  "scores": dict(oa)})
    return r
