"""Clean scores (design 6.1, 5.2, 13.1 test_view): scores.clean_view and the level bands of the absolute scale
(changes of 2026-09-26: no reference group, Russian speech shows the scores only; 3.1: one model in the view)."""
from __future__ import annotations

import copy
import json

from samples import english, rep

from bs3 import scores
from bs3.norms import TRAIT_KEYS


def test_clean_view_sample_b_main_scores_are_oceanai_only():
    r = rep("B")
    assert r["traits"]["extraversion"]["score"] == 0.629            # the mixed 2.0 number
    v = scores.clean_view(r)
    assert round(v["traits"]["extraversion"]["score"], 3) == 0.730
    for k in TRAIT_KEYS:
        assert v["traits"][k]["score"] == r["variant_scores"]["oceanai"][k]


def test_clean_view_sample_b_gaps():
    v = scores.clean_view(rep("B"))
    gaps = [t["segment"] for t in v["timeline"] if t["scores"] is None]
    assert gaps == [10, 11, 12, 14, 15, 26, 33]
    assert all(t.get("no_primary") for t in v["timeline"] if t["scores"] is None)
    meta = v["view_meta"]
    assert meta["main_system"] == "oceanai" and meta["main_source"] == "ocean_ai" and meta["lang"] == "ru"
    assert meta["selected"] == "oceanai" and meta["selected_title"] == "OCEAN-AI"
    assert meta["segments_total"] == 33 and meta["segments_used"] == 26
    assert meta["segments_without_primary"] == gaps and meta["primary_missing"] is False
    assert "reference" not in meta


def test_clean_view_spread_recomputed():
    r = rep("B")
    assert r["scores_std_across_segments"]["extraversion"] > 0.2        # ±0.205 with the mixed segments
    v = scores.clean_view(r)
    std = v["scores_std_across_segments"]
    assert abs(std["extraversion"] - 0.023) < 0.0015, std["extraversion"]
    assert max(std[k] for k in TRAIT_KEYS) <= 0.0235
    a = scores.clean_view(rep("A"))["scores_std_across_segments"]
    assert max(a[k] for k in TRAIT_KEYS) <= 0.0375


def test_clean_view_ru_has_no_percentiles():
    """Older Russian jobs carry percentiles against the pool of processed videos: the view drops them (and any
    position); nothing on the page may show them."""
    r = rep("B")
    for k in TRAIT_KEYS:
        r["traits"][k].update({"percentile": 62.5, "percentile_ref": "пула обработанных русских роликов (N=5)",
                               "position": 0.9})
    r["interview"] = {"score": 0.41, "percentile": 30.0, "percentile_vs_fiv2": 30.0,
                      "percentile_ref": "train First Impressions V2 (6000 клипов), своя модель"}
    v = scores.clean_view(r)
    for k in TRAIT_KEYS:
        assert set(v["traits"][k]) & set(scores.RELATIVE_KEYS) == set(), k
    assert "interview" not in v                                               # the label of the other model
    assert r["traits"]["extraversion"]["percentile"] == 62.5                 # the original is not changed
    r["model"].update({"selected": "mm", "primary": "mm"})                    # a view of AMLAI 1.0 keeps the label
    v = scores.clean_view(r)
    assert v["interview"] == {"score": 0.41}


def test_clean_view_oceanai_drops_the_interview_label():
    """The label «собеседование» exists only for AMLAI 1.0: an imported 2.0 / 3.0 job shown as OCEAN-AI loses it
    whole video and per segment (scores and variants), so no card, bar, chart series or appendix column names the
    other model; a view of AMLAI 1.0 keeps it everywhere."""
    r = rep("B")
    r["interview"] = {"score": 0.4011, "name_ru": "впечатление «пригласить на собеседование»"}
    for t in r["timeline"]:
        if isinstance(t.get("scores"), dict):
            t["scores"]["interview"] = 0.4
        t["variants"] = {"mm": {**{k: 0.3 for k in TRAIT_KEYS}, "interview": 0.4}}
        if "oceanai" in t["members_used"]:
            t["variants"]["oceanai"] = {**{k: t["scores"][k] for k in TRAIT_KEYS}, "interview": 0.4}
    v = scores.clean_view(r)
    assert "interview" not in v
    for t in v["timeline"]:
        assert not (isinstance(t.get("scores"), dict) and "interview" in t["scores"]), t["segment"]
        assert all("interview" not in x for x in t["variants"].values()), t["segment"]
    assert r["interview"]["score"] == 0.4011 and "interview" in r["timeline"][0]["scores"]    # the original is kept
    own = rep("B")
    own["model"].update({"selected": "mm", "primary": "mm"})
    own["interview"] = {"score": 0.4011}
    for t in own["timeline"]:
        t["members_used"] = ["mm"]
        t["scores"] = {**{k: 0.3 for k in TRAIT_KEYS}, "interview": 0.4}
    v = scores.clean_view(own)
    assert v["interview"] == {"score": 0.4011}
    assert all(t["scores"]["interview"] == 0.4 for t in v["timeline"])


def test_clean_view_does_not_change_rep():
    r = rep("B")
    before = copy.deepcopy(r)
    scores.clean_view(r)
    assert r == before


def test_clean_view_idempotent():
    for name in ("A", "B"):
        v1 = scores.clean_view(rep(name))
        v2 = scores.clean_view(v1)
        assert json.dumps(v1, sort_keys=True) == json.dumps(v2, sort_keys=True)


def test_clean_view_one_model():
    """3.1: the view holds the recorded model only — whole video and segments; an imported 2.0 job shows OCEAN-AI."""
    r = rep("B")
    for t in r["timeline"]:
        t["variants"] = {"mm": dict(r["variant_scores"]["mm"])}
        if "oceanai" in t["members_used"]:
            t["variants"]["oceanai"] = dict(t["scores"])
    v = scores.clean_view(r)
    meta = v["view_meta"]
    assert meta["selected"] == "oceanai" and meta["selected_title"] == "OCEAN-AI" and meta["main_system"] == "oceanai"
    assert set(v["variant_scores"]) == {"oceanai"}
    assert all(set(t["variants"]) <= {"oceanai"} for t in v["timeline"])
    assert [t["segment"] for t in v["timeline"] if t["scores"] is None] == [10, 11, 12, 14, 15, 26, 33]
    assert set(r["variant_scores"]) == {"mm", "oceanai"}                        # the original is not changed
    # a 3.1 job of AMLAI 1.0: model.selected wins
    r = rep("B")
    r["model"].update({"selected": "mm", "selected_title": "AMLAI 1.0", "primary": "mm"})
    r["variant_scores"] = {"mm": r["variant_scores"]["mm"]}
    for t in r["timeline"]:
        t["members_used"] = ["mm"]
    v = scores.clean_view(r)
    assert v["view_meta"]["main_system"] == "mm" and v["view_meta"]["selected_title"] == "AMLAI 1.0"
    assert v["view_meta"]["primary_missing"] is False and set(v["variant_scores"]) == {"mm"}
    assert round(v["traits"]["extraversion"]["score"], 4) == round(r["variant_scores"]["mm"]["extraversion"], 4)
    assert all(t["scores"] is not None for t in v["timeline"])


def test_a_report_that_names_no_model_is_read_as_ocean_ai():
    """The reading fallback does not follow the default model of a new analysis (bs3.DEFAULT_MODEL is AMLAI 1.0
    since 3.1): a file that names no model was made before that, and labelling it «AMLAI 1.0» would also give it
    the modalities of a model that never ran."""
    import bs3
    assert scores.READ_FALLBACK == "oceanai" != bs3.DEFAULT_MODEL
    assert scores.main_system({}) == ("oceanai", False)
    assert scores.main_system({"model": {"backend": "ensemble"}}) == ("oceanai", False)
    from bs3.narrative import method_notes
    r = {"model": {"lang": "ru"}, "traits": {k: {"score": 0.4} for k in TRAIT_KEYS}}
    t = method_notes(scores.clean_view(r))
    assert t.startswith("Оценки дала система OCEAN-AI") and "AMLAI" not in t


def test_recorded_model():
    assert scores.recorded_model(rep("A")) == "oceanai"
    assert scores.recorded_model({"model": {"selected": "mm", "primary": "oceanai"}}) == "mm"
    assert scores.recorded_model({"model": {"backend": "mm"}}) == "mm"
    assert scores.recorded_model({"model": {"backend": "ensemble"}}) is None
    assert scores.recorded_model({}) is None
    # a single-model CLI report without per-member scores keeps its traits and is read as its backend
    r = {"model": {"lang": "ru", "backend": "mm"}, "traits": {k: {"score": 0.4} for k in TRAIT_KEYS},
         "timeline": [{"segment": 1, "start": 0, "end": 20, "scores": {k: 0.4 for k in TRAIT_KEYS}}]}
    v = scores.clean_view(r)
    assert v["view_meta"]["main_system"] == "mm" and v["view_meta"]["primary_missing"] is False
    assert v["traits"]["openness"]["score"] == 0.4 and v["timeline"][0]["scores"] is not None


def test_clean_view_old_english_job_shows_one_model():
    """An older English job (no primary, two members): no mean of two systems any more, OCEAN-AI is shown; the FIV2
    percentile of English speech stays."""
    r = english("B")
    v = scores.clean_view(r)
    assert v["view_meta"]["main_system"] == "oceanai" and v["view_meta"]["primary_missing"] is False
    assert "reference" not in v["view_meta"] and "mean" not in v["view_meta"].values()
    for k in TRAIT_KEYS:
        assert v["traits"][k]["score"] == r["variant_scores"]["oceanai"][k]
    r["traits"]["extraversion"].update({"percentile": 72.0, "percentile_ref": "train First Impressions V2 (6000 клипов)"})
    e = scores.clean_view(r)["traits"]["extraversion"]
    assert e["percentile"] == 72.0 and e["percentile_ref"].startswith("train First Impressions V2")   # FIV2 stays


def test_clean_view_primary_missing_falls_back_to_own_model():
    r = rep("B")
    del r["variant_scores"]["oceanai"]
    for t in r["timeline"]:
        t["members_used"] = ["mm"]
    v = scores.clean_view(r)
    assert v["view_meta"]["main_system"] == "mm" and v["view_meta"]["primary_missing"] is True
    assert v["traits"]["extraversion"]["score"] == r["variant_scores"]["mm"]["extraversion"]
    assert all(t["scores"] is not None for t in v["timeline"])


def test_segment_ok_formats():
    r = rep("B")
    assert scores.segment_ok(r, {"members_used": ["oceanai", "mm"]})
    assert not scores.segment_ok(r, {"members_used": ["mm"]})
    assert scores.segment_ok(r, {"members_used": ["mm"], "primary_used": "oceanai"})
    assert not scores.segment_ok(r, {"members_used": ["oceanai", "mm"], "primary_used": None})


def test_levels_and_bands():
    """The five bands of the absolute score (edges in scores.py), decided on the printed value (two decimals):
    high >= 0.80, above 0.65-0.80, mid 0.35-0.65 (exactly the MBTI borderline zone), below 0.20-0.35, low <= 0.20."""
    L = scores.level
    assert L(0.80) == "high" and L(0.79) == "above" and L(0.65) == "above" and L(0.64) == "mid"
    assert L(0.6449) == "mid" and L(0.6451) == "above" and L(0.7951) == "high"      # 0.64 / 0.65 / 0.80 printed
    assert L(0.5) == "mid" and L(0.36) == "mid" and L(0.35) == "below" and L(0.21) == "below"
    assert L(0.3551) == "mid" and L(0.3549) == "below" and L(0.2049) == "low"       # 0.36 / 0.35 / 0.20 printed
    assert L(0.20) == "low" and L(0.0) == "low" and L(1.0) == "high" and L(None) is None
    assert L(float("nan")) is None and L("abc") is None and L(1.3) == "high" and L(-0.2) == "low"
    assert scores.level_phrase(0.9) == "высокий уровень" and scores.level_phrase(0.73) == "выше среднего"
    assert scores.level_phrase(0.5) == "средний уровень" and scores.level_phrase(0.3) == "ниже среднего"
    assert scores.level_phrase(0.1) == "низкий уровень"
    assert set(scores.LEVELS_RU) == {"high", "above", "mid", "below", "low"}
    assert scores.score_text(0.7298) == "0.73" and scores.score_text(None) is None


def test_levels_agree_with_the_letters():
    """A value with a confident MBTI letter is never «средний уровень»; one in the borderline zone always is."""
    from bs3 import mbti
    for i in range(10001):
        v = i / 10000
        a = mbti.mbti_for("oceanai", {"extraversion": v})["axes"]["EI"]
        assert (scores.level(v) == "mid") == a["borderline"], v
        assert a["value"] == scores.shown(v) and f"{a['value']:.2f}" == scores.score_text(v), v
        if not a["borderline"]:
            assert scores.level(v) in (("high", "above") if a["letter"] == "E" else ("below", "low")), v
    for v in (0.35, 0.65, 0.2, 0.8, 0.15, 0.85):
        a = mbti.bigfive_to_mbti({"extraversion": v})["axes"]["EI"]
        assert (scores.level(v) == "mid") == a["borderline"], v


def test_rounded_once():
    """A score is rounded once, to the printed two decimals, and the letter, the borderline flag, the level word and
    the printed number agree everywhere (0.6497 and 0.35049 print as 0.65 and 0.35 and are not on the border;
    0.75499 prints 0.75, not 0.76 through 0.7550; 0.47531 prints 0.48, not 0.47 through 0.475)."""
    from bs3 import mbti
    from bs3.web import mbti_html
    sc = {"openness": 0.6497, "conscientiousness": 0.35049, "extraversion": 0.47531, "agreeableness": 0.75499,
          "emotional_stability": 0.5}
    m = mbti.mbti_for("mm", sc)
    ax = m["axes"]
    assert (ax["SN"]["value"], ax["SN"]["letter"], ax["SN"]["borderline"]) == (0.65, "N", False)
    assert (ax["JP"]["value"], ax["JP"]["letter"], ax["JP"]["borderline"]) == (0.35, "P", False)
    assert (ax["EI"]["value"], ax["EI"]["borderline"]) == (0.48, True) and ax["TF"]["value"] == 0.75
    assert [scores.score_text(v) for v in (0.75499, 0.47531, 0.6497, 0.35049)] == ["0.75", "0.48", "0.65", "0.35"]
    h = mbti_html.types_html(m)
    for s in ("открытость опыту 0.65 — выше среднего", "добросовестность 0.35 — ниже среднего",
              "экстраверсия 0.48 — средний уровень", "доброжелательность 0.75 — выше среднего"):
        assert s in h, s
    assert "умеренно (уверенность 0.30)" in h                          # the confidence is labelled, not a score
    # the view keeps the main scores unrounded; they are rounded at display
    r = rep("B")
    r["variant_scores"]["oceanai"]["conscientiousness"] = 0.75499
    v = scores.clean_view(r)
    assert v["traits"]["conscientiousness"]["score"] == 0.75499
    assert scores.score_text(v["traits"]["conscientiousness"]["score"]) == "0.75"


def test_data_json():
    """Tab «Данные» of a Russian job: no percentiles, no stored 2.0 summary; the input is not changed."""
    r = rep("A")
    r["traits"]["openness"]["percentile"] = 21.4
    r["traits"]["openness"]["percentile_ref"] = "пула обработанных русских роликов (N=6)"
    r["narrative"] = "… ниже, чем у большинства из 6 русских роликов …"
    before = copy.deepcopy(r)
    d, trimmed = scores.data_json(r)
    assert trimmed and "narrative" not in d and "percentile" not in d["traits"]["openness"]
    assert r == before
    assert set(d["variant_scores"]) == {"mm", "oceanai"}         # the tab shows the file as it lies on disk
    e = english("B")
    d2, trimmed2 = scores.data_json(e)
    assert d2 == e and not trimmed2
