"""Percentile norms of the five traits on the First Impressions V2 training split (6000 clips)."""
from __future__ import annotations
import bisect, json
from importlib import resources

TRAIT_KEYS = ["openness", "conscientiousness", "extraversion", "agreeableness", "emotional_stability"]
# column names in OCEAN-AI output / FIV2 csv
OCEANAI_COLUMNS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Non-Neuroticism"]
FIV2_COLUMNS = ["openness", "conscientiousness", "extraversion", "agreeableness", "non-neuroticism"]
RU_NAMES = {
    "openness": "открытость опыту",
    "conscientiousness": "добросовестность",
    "extraversion": "экстраверсия",
    "agreeableness": "доброжелательность",
    "emotional_stability": "эмоциональная стабильность (non-neuroticism)",
}

RU_SHORT = {"openness": "Откр.", "conscientiousness": "Добр.", "extraversion": "Экстр.", "agreeableness": "Доброж.",
            "emotional_stability": "Эм. стаб.", "interview": "Собесед."}
RU_TITLES = {"openness": "Открытость опыту", "conscientiousness": "Добросовестность", "extraversion": "Экстраверсия",
             "agreeableness": "Доброжелательность", "emotional_stability": "Эмоциональная стабильность",
             "interview": "Впечатление «собеседование»"}

_norms = None


def load_norms() -> dict:
    global _norms
    if _norms is None:
        with resources.files("bs2").joinpath("fiv2_norms.json").open("r", encoding="utf-8") as f:
            _norms = json.load(f)
    return _norms


def percentile(trait: str, score: float) -> float:
    """Percentile (0..100) of `score` among FIV2 train labels for `trait` (linear interpolation on 101 quantiles)."""
    q = load_norms()["quantiles"][trait]  # q[p] = value at percentile p, p = 0..100
    if score <= q[0]:
        return 0.0
    if score >= q[-1]:
        return 100.0
    i = bisect.bisect_right(q, score)  # q[i-1] <= score < q[i]
    lo, hi = q[i - 1], q[i]
    frac = 0.0 if hi == lo else (score - lo) / (hi - lo)
    return round(i - 1 + frac, 1)
