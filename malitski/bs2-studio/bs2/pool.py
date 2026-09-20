"""Pool of processed videos per language: empirical percentiles for a scale that has no labelled reference
population at hand (OCEAN-AI on MuPTA weights for Russian speech). Every analysed video is registered once
(fingerprint of the file), and a score is reported as its rank among the pool. Small pools give coarse
percentiles; below MIN_POOL no percentile is reported."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from pathlib import Path

from .norms import TRAIT_KEYS

POOL_DIR = Path(os.environ.get("BS2_POOL_DIR", "~/bs2_data/pool")).expanduser()
MIN_POOL = 3


def fingerprint(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.md5()
    with open(p, "rb") as f:
        h.update(f.read(1 << 20))
    return f"{p.stat().st_size}_{h.hexdigest()}"


def _dir(lang: str) -> Path:
    d = POOL_DIR / lang
    d.mkdir(parents=True, exist_ok=True)
    return d


def add(video: str | Path, scores: dict, lang: str, primary: str | None, name: str | None = None) -> int:
    """Register (or refresh) a video's main scores in the pool of `lang`. Returns the pool size."""
    d = _dir(lang)
    entry = {"name": name or Path(video).name, "lang": lang, "primary": primary,
             "scores": {k: round(float(scores[k]), 4) for k in TRAIT_KEYS if k in scores},
             "created_at": _dt.datetime.now().isoformat(timespec="seconds")}
    (d / f"{fingerprint(video)}.json").write_text(json.dumps(entry, ensure_ascii=False, indent=1), encoding="utf-8")
    return len(list(d.glob("*.json")))


def entries(lang: str) -> list[dict]:
    out = []
    for p in sorted(_dir(lang).glob("*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            continue
    return out


def percentile(trait: str, score: float, lang: str) -> tuple[float | None, int]:
    """Rank of `score` among the pool (0..100, mid-rank for ties). None when the pool is smaller than MIN_POOL."""
    vals = [e["scores"][trait] for e in entries(lang) if trait in e.get("scores", {})]
    n = len(vals)
    if n < MIN_POOL:
        return None, n
    below = sum(1 for v in vals if v < score - 1e-9)
    equal = sum(1 for v in vals if abs(v - score) <= 1e-9)
    # the video itself counts as a pool member: if its exact score is not stored (e.g. re-analysed with slightly
    # different numbers), rank it as an extra member, so the top video reads 95%, never 100%
    if equal == 0:
        n_eff, equal = n + 1, 1
    else:
        n_eff = n
    return round(100.0 * (below + 0.5 * equal) / n_eff, 1), n


def label(lang: str, n: int) -> str:
    names = {"ru": "русских", "en": "английских"}
    return f"пула обработанных {names.get(lang, lang)} роликов (N={n})"      # reads after «относительно …»
