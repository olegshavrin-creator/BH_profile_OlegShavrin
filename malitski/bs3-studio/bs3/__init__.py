"""BS Profiler 3.1: personality characterization (Big Five with MBTI notation), emotion, voice, face and speech analytics
from a video of Russian speech, by one model chosen for the analysis (OCEAN-AI or AMLAI 1.0)."""
__version__ = "3.1.0a1"
PRODUCT = "BS Profiler 3.1"
PRODUCT_SLUG = "BS_Profiler_3"
# the models the page offers, by their internal key, and the names the user sees (change request 3.1, section 2);
# exactly one of them is loaded and run per analysis. The order is the order of the radio on the page: AMLAI 1.0 is
# the default model and stands on the left, already chosen; OCEAN-AI is there for whoever wants it (owner, 2026-09-26)
MODEL_TITLES = {"mm": "AMLAI 1.0", "oceanai": "OCEAN-AI"}
DEFAULT_MODEL = "mm"
# what each model looks at (result.json `modalities_used`, appendix А «Модальности» of the PDF)
MODALITIES = {"oceanai": ("audio", "video", "text"), "mm": ("face", "audio", "text", "behavior")}
LANG = "ru"          # the speech language of every analysis (3.1: Russian only, section 1 of the change request)
