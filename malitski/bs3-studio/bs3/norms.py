"""Trait keys and their Russian names."""
from __future__ import annotations

TRAIT_KEYS = ["openness", "conscientiousness", "extraversion", "agreeableness", "emotional_stability"]
# column names in OCEAN-AI output
OCEANAI_COLUMNS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Non-Neuroticism"]
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
