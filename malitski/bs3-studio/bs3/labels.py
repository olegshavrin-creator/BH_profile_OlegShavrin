"""The names BS Profiler 3.1 shows for emotions, facial expressions, voice dimensions, traits and models, in one place:
the page, the PDF, the journal and the texts take them from here and keep no copies of their own.

A leaf of the import layers (tests/test_layers.py): it imports only the package itself (MODEL_TITLES) and norms, so
every module may use it, and the charts no longer load the analysis models (torch, cv2) for two lists of labels.
"""
from __future__ import annotations

from . import MODEL_TITLES
from .norms import RU_TITLES

# emotions of the text model and the face labels that mean the same (palette.EMO_ALIAS maps happy -> joy …)
EMO_RU = {"joy": "радость", "surprise": "удивление", "neutral": "нейтрально", "sadness": "грусть", "fear": "страх",
          "anger": "злость", "disgust": "отвращение",
          "happy": "радость", "sad": "грусть", "angry": "злость"}
VOICE_RU = {"arousal": "возбуждение", "dominance": "уверенность", "valence": "позитивность"}
# the labels of the text-emotion model in the order of the charts, the bars and the tables (analyses/emotions_text)
EMOTION_ORDER = ["joy", "surprise", "neutral", "sadness", "fear", "anger", "disgust"]
# the same seven in the rows of the PDF heatmap «Эмоции по ходу ролика», «нейтрально» last and apart; the emotion
# shares of a segment (segments.emotion_shares) are counted in this order
HEAT_ROWS = ["joy", "surprise", "sadness", "fear", "anger", "disgust", "neutral"]
# the labels of the facial-expression model: their Russian names (stored with the key frames) and the order of the
# charts (analyses/face_expr)
EXPR_RU = {"angry": "злость", "disgust": "отвращение", "fear": "страх", "happy": "радость", "neutral": "нейтрально",
           "sad": "грусть", "surprise": "удивление"}
EXPR_ORDER = ["happy", "surprise", "neutral", "sad", "fear", "angry", "disgust"]
# radar axis labels: long names are split so they are not cut off by a narrow column (the PDF breaks at <br> too)
RADAR_LABEL = {"openness": "Открытость<br>опыту", "conscientiousness": "Добросовест-<br>ность",
               "extraversion": "Экстра-<br>версия", "agreeableness": "Доброжела-<br>тельность",
               "emotional_stability": "Эмоциональная<br>стабильность"}
# short row names for tables and chart rows (the interview title is too long for a first column)
ROW_TITLES = {**RU_TITLES, "interview": "«Собеседование»"}
# the model key -> the `source` of its MBTI section and of view_meta.main_source
SOURCE_OF = {"oceanai": "ocean_ai", "mm": "own_model"}
SOURCE_RU = {SOURCE_OF[k]: v for k, v in MODEL_TITLES.items()}       # "ocean_ai": OCEAN-AI, "own_model": AMLAI 1.0


def model_title(main: str | None) -> str:
    """«OCEAN-AI, веса MuPTA» / «AMLAI 1.0»: the model of a clean view with its weights line (change request 3.1,
    section 2)."""
    title = MODEL_TITLES.get(main, str(main or "—"))
    return title + (", веса MuPTA" if main == "oceanai" else "")


def source_title(mb: dict | None) -> str:
    """The title of the model a section describes: «OCEAN-AI» / «AMLAI 1.0»."""
    src = (mb or {}).get("source")
    return SOURCE_RU.get(src, MODEL_TITLES.get((mb or {}).get("model"), str(src)))


def mbti_model_title(mb: dict) -> str:
    """«OCEAN-AI, веса MuPTA» / «AMLAI 1.0»: the model whose type the panel and the strip show (one model, 3.1)."""
    title = source_title(mb)
    return title + (", веса MuPTA" if mb.get("source") == "ocean_ai" else "")
