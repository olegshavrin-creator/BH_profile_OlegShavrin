"""bs3/labels.py: the names of emotions, facial expressions, voice dimensions, traits and models in one place
(refactoring plan of 3.1, stage 10). The titles and orders equal what the page, the PDF and the journal showed before,
the copies are gone, and the web charts no longer load the analysis models for two lists of labels.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from bs3 import (MODEL_TITLES, analyses_text, characterization, facts, journal, labels, mbti, scores,
                 segments)
from bs3.web import charts, mbti_html, page, parts as webparts
from bs3.pdf import appendix as pdf_appendix
from bs3.pdf import build as pdf_build
from bs3.pdf import charts as pdf_charts
from bs3.pdf import document as pdf_document
from bs3.pdf import frames as pdf_frames
from bs3.pdf import mbti_section as pdf_mbti
from bs3.pdf import sections as pdf_sections
from bs3.pdf import widgets as pdf_widgets
from bs3.palette import HTML

ROOT = Path(__file__).resolve().parents[1]
# the orders the charts used before this stage (charts._EMO_BAR_ORDER = analyses/emotions_text.EMOTION_ORDER, and
# analyses/face_expr.EXPR_ORDER); the keys of pdf_report.EMO_NAMES were the same seven text emotions in this order
OLD_EMOTIONS = ["joy", "surprise", "neutral", "sadness", "fear", "anger", "disgust"]
OLD_EXPRESSIONS = ["happy", "surprise", "neutral", "sad", "fear", "angry", "disgust"]
OLD_EMO_NAMES = {"joy": "радость", "surprise": "удивление", "neutral": "нейтрально", "sadness": "грусть",
                 "fear": "страх", "anger": "злость", "disgust": "отвращение"}


def test_model_titles():
    assert labels.model_title("oceanai") == "OCEAN-AI, веса MuPTA"
    assert labels.model_title("mm") == "AMLAI 1.0"
    assert [labels.model_title(k) for k in ("sslmepr", None, "")] == ["sslmepr", "—", "—"]


def test_mbti_model_title_on_schema_1_and_3_sections():
    # schema 1 (3.0 jobs): `source` only, no `model`; its second opinion was the own model
    assert labels.mbti_model_title({"schema_version": 1, "source": "ocean_ai", "model": None}) == "OCEAN-AI, веса MuPTA"
    assert labels.mbti_model_title({"schema_version": 1, "source": "own_model", "model": None}) == "AMLAI 1.0"
    # schema 3 (3.1)
    assert labels.mbti_model_title({"schema_version": 3, "source": "ocean_ai", "model": "oceanai"}) == \
        "OCEAN-AI, веса MuPTA"
    assert labels.mbti_model_title({"schema_version": 3, "source": "own_model", "model": "mm"}) == "AMLAI 1.0"
    # without a `source` the model names the title, and the weights line goes with the source only (as before)
    assert labels.mbti_model_title({"model": "mm"}) == "AMLAI 1.0"
    assert labels.mbti_model_title({"model": "oceanai"}) == "OCEAN-AI"
    assert labels.mbti_model_title({}) == "None"
    assert labels.source_title(None) == "None" and labels.source_title({"source": "ocean_ai"}) == "OCEAN-AI"
    assert labels.SOURCE_RU == {"ocean_ai": "OCEAN-AI", "own_model": "AMLAI 1.0"}
    assert labels.SOURCE_OF == {"oceanai": "ocean_ai", "mm": "own_model"} and set(labels.SOURCE_OF) == set(MODEL_TITLES)


def test_the_journal_names_the_model_like_the_page():
    """journal.result_lines builds «Итог (…)» with labels.model_title (it had its own copy of the title)."""
    from samples import rep
    r = rep("B")
    assert journal.result_lines(r)[0].startswith("Итог (OCEAN-AI, веса MuPTA): открытость 0.71")
    r["model"].update({"selected": "mm", "primary": "mm"})
    r["variant_scores"] = {"mm": r["variant_scores"]["mm"]}
    assert journal.result_lines(r)[0].startswith("Итог (AMLAI 1.0): ")


def test_orders_equal_the_orders_the_charts_used():
    assert labels.EMOTION_ORDER == OLD_EMOTIONS and labels.EXPR_ORDER == OLD_EXPRESSIONS
    assert labels.EXPR_RU == {"angry": "злость", "disgust": "отвращение", "fear": "страх", "happy": "радость",
                              "neutral": "нейтрально", "sad": "грусть", "surprise": "удивление"}
    assert {k: labels.EMO_RU[k] for k in OLD_EMOTIONS} == OLD_EMO_NAMES
    assert labels.ROW_TITLES == {"openness": "Открытость опыту", "conscientiousness": "Добросовестность",
                                 "extraversion": "Экстраверсия", "agreeableness": "Доброжелательность",
                                 "emotional_stability": "Эмоциональная стабильность", "interview": "«Собеседование»"}
    # the stacked bands of «Эмоции по ходу ролика»: every key has its own value, so the traces show the key order
    text = {k: round(0.01 * (i + 1), 2) for i, k in enumerate(OLD_EMOTIONS)}
    face = {k: round(0.11 + 0.01 * i, 2) for i, k in enumerate(OLD_EXPRESSIONS)}
    rep = {"analyses": {"per_segment": [{"start": 0.0, "end": 20.0, "emotions_text": text,
                                         "face": {"expressions": face}}],
                        "emotions_text": {"mean": text}, "face": {"mean": face}}}
    fig = charts.fig_emotions_timeline(rep, "light")
    bands = [(t.stackgroup, t.y[0]) for t in fig.data if t.stackgroup]
    assert bands == [("t", text[k]) for k in OLD_EMOTIONS] + [("f", face[k]) for k in OLD_EXPRESSIONS]
    legend = [t.name for t in fig.data if t.showlegend is not False and not t.stackgroup]
    assert legend == [OLD_EMO_NAMES[k] for k in OLD_EMOTIONS]
    bars = charts.fig_emotion_bars(rep, "light")
    assert list(bars.data[0].x) == [OLD_EMO_NAMES[k] for k in OLD_EMOTIONS]
    assert list(bars.data[0].y) == [text[k] for k in OLD_EMOTIONS]
    face_of = {"joy": "happy", "sadness": "sad", "anger": "angry"}
    assert list(bars.data[1].y) == [face[face_of.get(k, k)] for k in OLD_EMOTIONS]


def test_the_pdf_names_only_the_seven_text_emotions():
    """pdf_report._dominant_text named the dominant emotion by EMO_NAMES (the seven text emotions): a face label comes
    aliased to them, any other label stays as it is. Since stage 14a it is facts.segment_emotion, the cell of the page
    and of the PDF."""
    seg = {"text_en": "we spoke", "emotions_text": {"neutral": 0.2, "joy": 0.7, "fear": 0.1},
           "face": {"expressions": {"happy": 0.3, "sad": 0.6, "neutral": 0.1}}}
    assert facts.segment_emotion(seg, "text") == "радость 70%"
    assert facts.segment_emotion(seg, "face") == "грусть 60%"
    for k, ru in OLD_EMO_NAMES.items():
        assert facts.segment_emotion({"emotions_text": {k: 0.9}}, "text") == f"{ru} 90%"
    assert facts.segment_emotion({"face": {"expressions": {"contempt": 0.8, "neutral": 0.2}}}, "face") == \
        "contempt 80%"
    assert facts.segment_emotion(None, "text") == "—"


def test_the_copies_are_gone():
    """Each module either takes the name from labels (or mbti, scores, palette) or does not have it at all."""
    shared = {"EMO_RU": labels.EMO_RU, "VOICE_RU": labels.VOICE_RU, "EMOTION_ORDER": labels.EMOTION_ORDER,
              "EXPR_ORDER": labels.EXPR_ORDER, "EXPR_RU": labels.EXPR_RU, "RADAR_LABEL": labels.RADAR_LABEL,
              "ROW_TITLES": labels.ROW_TITLES, "model_title": labels.model_title,
              "mbti_model_title": labels.mbti_model_title, "SOURCE_OF": labels.SOURCE_OF,
              "SOURCE_RU": labels.SOURCE_RU, "source_title": labels.source_title,
              "AXES": mbti.AXES, "AXIS_LABEL": mbti.AXIS_LABEL, "num": scores.num, "HEAT_ROWS": labels.HEAT_ROWS}
    for mod in (analyses_text, characterization, charts, facts, journal, mbti, mbti_html, pdf_charts, pdf_document,
                pdf_frames, pdf_mbti, pdf_build, pdf_appendix, pdf_sections, pdf_widgets, scores, segments, page,
                webparts):
        for name, obj in shared.items():
            assert getattr(mod, name, obj) is obj, f"{mod.__name__}.{name} is a copy"
    gone = {charts: ("_RADAR_LABEL", "_EMO_BAR_ORDER"), pdf_charts: ("MOD_ROWS",),
            pdf_build: ("TITLES", "EMO_NAMES"), pdf_appendix: ("TITLES", "EMO_NAMES"),
            webparts: ("_pct_phrase", "_is_fiv2"),
            mbti_html: ("panel_title",), journal: ("MEMBERS",), characterization: ("_num",), mbti: ("_num",),
            scores: ("_num",)}
    for mod, names in gone.items():
        for name in names:
            assert not hasattr(mod, name), f"{mod.__name__}.{name}"
    assert pdf_appendix.model_title is labels.model_title and "num" in scores.__all__
    assert pdf_sections.model_title is labels.model_title and pdf_sections.VOICE_RU is labels.VOICE_RU
    assert characterization.OUTLINE == HTML["track_outline"] == "#808080"
    # the analysis modules take the orders from labels and define none of the names themselves
    for rel, names in (("analyses/emotions_text.py", ("EMOTION_ORDER",)),
                       ("analyses/face_expr.py", ("EXPR_ORDER", "EXPR_RU"))):
        tree = ast.parse((ROOT / "bs3" / rel).read_text(encoding="utf-8"))
        defined = {t.id for n in ast.walk(tree) if isinstance(n, ast.Assign) for t in n.targets
                   if isinstance(t, ast.Name)}
        assert not defined & set(names), (rel, defined & set(names))
        froms = {(n.module, a.name) for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
        assert ("labels", names[0]) in froms, (rel, froms)


def test_web_charts_load_no_analysis_model():
    """The emotion chart used to import analyses/emotions_text and face_expr (torch, transformers, cv2) for two lists;
    the radar imported webparts for the model title. Built in a fresh interpreter, none of them is loaded."""
    code = ("import sys\n"
            "from bs3.web import charts\n"
            "rep = {'traits': {k: {'score': 0.5} for k in ('openness', 'conscientiousness', 'extraversion',"
            " 'agreeableness', 'emotional_stability')}, 'model': {'selected': 'mm'},"
            " 'analyses': {'per_segment': [{'start': 0, 'end': 20, 'emotions_text': {'joy': 1.0},"
            " 'face': {'expressions': {'happy': 1.0}}}], 'emotions_text': {'mean': {'joy': 1.0}},"
            " 'face': {'mean': {'happy': 1.0}}}}\n"
            "for f in (charts.fig_emotions_timeline, charts.fig_emotion_bars, charts.fig_radar):\n"
            "    f(rep, 'light')\n"
            "bad = ('torch', 'cv2', 'transformers', 'librosa', 'bs3.analyses.emotions_text', 'bs3.analyses.face_expr',"
            " 'bs3.web.parts')\n"
            "print(sorted(m for m in bad if m in sys.modules))\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.split() == ["[]"], r.stdout
