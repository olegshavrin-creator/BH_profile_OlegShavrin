"""Captions under the key frames of AMLAI 1.0 (change request «подписи под ключевыми кадрами», 2026-09-26):

- the line on the page with a phrase / without a phrase / without expressions, and the hover line;
- the rules of the signed attribution: one trait, two traits of the same sign, two traits of different signs, a
  frame whose effect is tiny next to the strongest key frame, and an old job that has no signs at all;
- the validation of the phrase the vision model returns (too long, digits, verdict words, a line break, empty)
  and the one retry;
- an explanation of an older job (no `signed`, no phrases, no expressions) still renders on the page and in the PDF.
"""
from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

from samples import rep

from bs3 import frame_captions as fc
from bs3.norms import TRAIT_KEYS

# key_<index in the sampled sequence>_frame<number in the clip>.jpg; 30 fps and a segment starting at 2:00 put
# frame 420 at 2:14 of the whole video
FILES = ["key_04_frame120.jpg", "key_11_frame300.jpg", "key_18_frame420.jpg", "key_22_frame540.jpg",
         "key_27_frame690.jpg"]
IDX = [4, 11, 18, 22, 27]
REPORT = {"representative_segment": 3, "media": {"fps": 30.0},
          "timeline": [{"segment": 1, "start": 0.0, "end": 60.0}, {"segment": 2, "start": 60.0, "end": 120.0},
                       {"segment": 3, "start": 120.0, "end": 180.0}]}


def _paths(base: str = "/jobs/x/explain") -> list:
    return [f"{base}/{f}" for f in FILES]


def _expl(signed: dict | None = None, phrases: dict | None = None, expressions: dict | None = None,
          legacy: bool = False) -> dict:
    """An explanation.json-shaped dict: `signed` maps a frame index to {trait: value}."""
    n = 30
    per = {k: {"importance": [0.0] * n, "top_frames": IDX[:5]} for k in TRAIT_KEYS}
    if not legacy:
        for k in TRAIT_KEYS:
            per[k]["signed"] = [0.0] * n
    for i, vals in (signed or {}).items():
        for k, v in vals.items():
            per[k]["importance"][i] = abs(v)
            if not legacy:
                per[k]["signed"][i] = v
    fr = {"per_output": per, "n_frames": n, "key_frame_files": _paths()}
    if not legacy:
        eff = []
        for i in range(n):
            ranked = sorted(((k, per[k]["signed"][i]) for k in TRAIT_KEYS), key=lambda kv: -abs(kv[1]))
            eff.append([{"output": k, "signed": v} for k, v in ranked[:2]])
        fr["per_frame_effect"] = eff
    info = []
    for f, i in zip(FILES, IDX):
        rec = {"file": f, "frame": i}
        if expressions and i in expressions:
            rec["expressions"] = [{"label": lab, "ru": ru, "share": s} for lab, ru, s in expressions[i]]
        if phrases and i in phrases:
            rec["phrase"] = phrases[i]
        info.append(rec)
    if info and (phrases or expressions):
        fr["key_frame_info"] = info
    return {"frames": fr}


HAPPY = [("happy", "радость", 0.6234), ("neutral", "нейтрально", 0.2071)]


# ---------------------------------------------------------------- the line under the frame
def test_caption_with_phrase_expression_and_effect():
    expl = _expl(signed={18: {"extraversion": 0.9, "agreeableness": 0.62}},
                 phrases={18: "улыбается, смотрит в камеру"}, expressions={18: HAPPY})
    e = fc.build(REPORT, _paths(), expl)[2]
    assert e["label"] == "2:14"
    assert e["caption"] == "2:14 · улыбается, смотрит в камеру"
    assert e["tooltip"] == "радость 62%, нейтрально 21% · повысил оценку экстраверсии и доброжелательности"
    assert e["expr_line"] == "радость 62%" and e["effect_short"] == "повысил экстраверсию"


def test_caption_without_phrase_falls_back_to_the_expression():
    expl = _expl(signed={18: {"extraversion": 0.9}}, expressions={18: HAPPY})
    e = fc.build(REPORT, _paths(), expl)[2]
    assert e["caption"] == "2:14 · радость"
    assert e["tooltip"] == "радость 62%, нейтрально 21% · повысил оценку экстраверсии"


def test_caption_without_expressions_is_the_moment_alone():
    expl = _expl(signed={18: {"extraversion": 0.9}})
    e = fc.build(REPORT, _paths(), expl)[2]
    assert e["caption"] == "2:14" and e["expr_line"] == ""
    assert e["tooltip"] == "повысил оценку экстраверсии"


def test_caption_without_a_frame_rate_numbers_the_frames():
    r = {**REPORT, "media": {}}
    e = fc.build(r, _paths(), _expl(signed={18: {"extraversion": 0.9}}, expressions={18: HAPPY}))[2]
    assert e["caption"] == "кадр 3 · радость" and e["moment"] is None
    assert not fc.any_moment(fc.build(r, _paths(), _expl()))


def test_two_key_frames_in_one_second_get_tenths():
    files = ["key_18_frame420.jpg", "key_19_frame424.jpg"]
    paths = [f"/j/{f}" for f in files]
    expl = {"frames": {"per_output": {}, "key_frame_files": paths}}
    labels = [e["label"] for e in fc.build(REPORT, paths, expl)]
    assert labels == ["2:14,0", "2:14,1"] and fc.has_tenths(REPORT, paths)


# ---------------------------------------------------------------- the sign rules
def _effect(vals: dict, others: dict | None = None) -> str:
    signed = {18: vals}
    signed.update(others or {})
    return fc.build(REPORT, _paths(), _expl(signed=signed))[2]["tooltip"]


def test_sign_rule_single_trait():
    assert _effect({"extraversion": 0.9, "openness": 0.2}) == "повысил оценку экстраверсии"
    assert _effect({"extraversion": -0.9, "openness": 0.2}) == "понизил оценку экстраверсии"


def test_sign_rule_second_trait_only_when_at_least_half():
    assert _effect({"extraversion": 0.9, "agreeableness": 0.45}) == "повысил оценку экстраверсии и доброжелательности"
    assert _effect({"extraversion": 0.9, "agreeableness": 0.44}) == "повысил оценку экстраверсии"


def test_sign_rule_mixed_signs():
    assert _effect({"extraversion": 0.9, "agreeableness": -0.8}) == \
        "повысил оценку экстраверсии, понизил — доброжелательности"
    assert _effect({"extraversion": -0.9, "openness": 0.8}) == "понизил оценку экстраверсии, повысил — открытости опыту"


def test_sign_rule_tiny_effect_has_no_direction():
    # the strongest key frame moves 1.0, this one 0.19 -> below a fifth: no «повысил», and the PDF line follows
    e = fc.build(REPORT, _paths(), _expl(signed={18: {"extraversion": 0.19}, 4: {"extraversion": 1.0}}))[2]
    assert e["tooltip"] == "заметно повлиял на оценку экстраверсии"
    assert e["effect_short"] == "повлиял на экстраверсию"
    big = fc.build(REPORT, _paths(), _expl(signed={18: {"extraversion": 0.21}, 4: {"extraversion": 1.0}}))[2]
    assert big["tooltip"] == "повысил оценку экстраверсии"


def test_old_job_without_signs_keeps_the_old_wording():
    expl = _expl(signed={18: {"extraversion": 0.9, "agreeableness": 0.62}}, legacy=True)
    e = fc.build(REPORT, _paths(), expl)[2]
    assert e["tooltip"] == "сильнее всего повлиял на оценку экстраверсии и доброжелательности"
    assert e["effect_short"] == "повлиял на экстраверсию"
    assert e["caption"] == "2:14"


def test_no_attribution_at_all_leaves_the_effect_out():
    e = fc.build(REPORT, _paths(), {"frames": {"key_frame_files": _paths()}})[2]
    assert e["tooltip"] == "" and e["caption"] == "2:14"
    assert fc.build(REPORT, _paths(), None)[0]["caption"] == "2:04"


# ---------------------------------------------------------------- the phrase from the vision model
def test_phrase_validation():
    assert fc.clean_phrase("«Улыбается, смотрит в камеру.»") == "улыбается, смотрит в камеру"
    assert fc.clean_phrase("улыбается, смотрит в камеру") == "улыбается, смотрит в камеру"
    assert fc.clean_phrase("") is None and fc.clean_phrase(None) is None and fc.clean_phrase("   ") is None
    assert fc.clean_phrase("наклон головы\nруки сложены") is None                      # a line break
    assert fc.clean_phrase("поднял 2 пальца") is None                                  # a digit
    assert fc.clean_phrase("уверенный взгляд") is None                                 # a verdict
    assert fc.clean_phrase("дружелюбная улыбка") is None
    assert fc.clean_phrase("нервничает, теребит рукав") is None
    assert fc.clean_phrase("кажется, отводит взгляд") is None
    assert fc.clean_phrase("выглядит как участник интервью") is None
    assert fc.clean_phrase("я вижу улыбку") is None                                    # the pronoun, not «стоя »
    assert fc.clean_phrase("стоя прямо, плечи развёрнуты") == "стоя прямо, плечи развёрнуты"
    # too long: trimmed at the last comma first, dropped when even that does not fit
    assert fc.clean_phrase("смотрит в камеру, чуть наклонил голову, руки лежат на столе") == \
        "смотрит в камеру, чуть наклонил голову"
    assert fc.clean_phrase("человек сидит перед камерой и подробно объясняет что-то важное") is None
    assert len(fc.clean_phrase("смотрит в камеру, чуть наклонил голову")) <= fc.PHRASE_MAX


def test_phrase_is_asked_twice_and_then_given_up():
    from bs3.mm import explain
    calls = []

    def bad(_b64):
        calls.append("bad")
        return "уверенный взгляд"

    def second_try(_b64):
        calls.append("t")
        return "поднял 3 пальца" if len(calls) == 1 else "смотрит в сторону"

    raw = {FILES[0]: "AAAA"}
    info = explain.key_frame_info(_paths()[:1], None, raw, phrase_fn=bad)
    assert "phrase" not in info[0] and len(calls) == fc.PHRASE_TRIES == 2
    calls.clear()
    info = explain.key_frame_info(_paths()[:1], None, raw, phrase_fn=second_try)
    assert info[0]["phrase"] == "смотрит в сторону" and len(calls) == 2
    assert info[0]["file"] == FILES[0] and info[0]["frame"] == 4


def test_key_frame_info_keeps_the_two_strongest_expressions():
    from bs3.mm import explain
    crops = [object()] * 30

    def expr(items):
        assert len(items) == 2
        return [{"happy": 0.62, "neutral": 0.21, "sad": 0.1, "angry": 0.07},
                {"sad": 0.5, "neutral": 0.4, "happy": 0.1}]

    info = explain.key_frame_info(_paths()[:2], crops, {}, expression_fn=expr)
    assert [e["ru"] for e in info[0]["expressions"]] == ["радость", "нейтрально"]
    assert info[0]["expressions"][0]["share"] == 0.62
    assert [e["ru"] for e in info[1]["expressions"]] == ["грусть", "нейтрально"]


# ---------------------------------------------------------------- the page and the PDF
def _job_with_frames(d: Path, expl: dict, size: tuple = (200, 120)) -> tuple:
    """A synthetic AMLAI 1.0 job with five saved key frames. Landscape frames go three per row in the PDF
    (60.7 mm), portrait ones five per row (34.8 mm) — the narrow case the captions must survive."""
    from PIL import Image
    ex = d / "explain"
    ex.mkdir(parents=True)
    paths = []
    for f in FILES:
        p = ex / f
        Image.new("RGB", size, (90, 90, 90)).save(p, "JPEG")
        paths.append(str(p))
    r = rep("B")
    r["model"].update({"selected": "mm", "primary": "mm", "selected_title": "AMLAI 1.0", "backend": "mm"})
    mm = {k: r["variant_scores"]["mm"][k] for k in TRAIT_KEYS}
    r["variant_scores"] = {"mm": r["variant_scores"]["mm"]}
    for t in r["timeline"]:
        t.update({"members_used": ["mm"], "primary_used": "mm", "variants": {"mm": dict(mm)}, "scores": dict(mm)})
    r["key_frames"] = paths
    r["job_dir"] = str(d)                         # the page finds the frames in the folder of the job (jobfiles)
    r["media"] = {"fps": 30.0}
    r["representative_segment"] = 3
    expl["frames"]["key_frame_files"] = paths
    for rec in expl["frames"].get("key_frame_info") or []:
        rec["file"] = Path(rec["file"]).name
    return r, paths, expl


def test_page_block_shows_the_caption_and_the_tooltip():
    from bs3.web import page
    expl = _expl(signed={18: {"extraversion": 0.9, "agreeableness": 0.62}},
                 phrases={18: "улыбается, смотрит в камеру"}, expressions={18: HAPPY})
    with tempfile.TemporaryDirectory() as d:
        r, paths, expl = _job_with_frames(Path(d), expl)
        html = page._frames_html(r, expl)
    start = float(r["timeline"][2]["start"])
    label = f"{int(start + 14) // 60}:{int(start + 14) % 60:02d}"
    assert f"<b>{label}</b> · улыбается, смотрит в камеру" in html
    assert "title='радость 62%, нейтрально 21% · повысил оценку экстраверсии и доброжелательности'" in html
    assert "font-size:13px" in html and "-webkit-line-clamp:2" in html
    assert "Наведите мышь на кадр" in html


def test_old_explanation_without_signs_and_phrases_still_renders():
    from bs3 import characterization, mbti, scores
    from bs3.pdf import build as pdf_build
    from bs3.web import page
    expl = _expl(signed={18: {"extraversion": 0.9}}, legacy=True)
    with tempfile.TemporaryDirectory() as d:
        r, paths, expl = _job_with_frames(Path(d), expl)
        html = page._frames_html(r, expl)
        assert "сильнее всего повлиял на оценку экстраверсии" in html
        assert "улыб" not in html and "радость" not in html
        # the note promises only what such a job really has: no short description, no expression on hover
        assert "коротко то, что на нём видно" not in html and "выражение лица" not in html
        assert "Наведите мышь на кадр — покажется то, как кадр сдвинул оценку." in html
        view = scores.clean_view(r)
        mb = mbti.get_mbti(r, view)
        out = Path(d) / "report.pdf"
        pdf_build.build_pdf(view, out, explanation=expl, media={"fps": 30.0}, key_frames=paths, mbti=mb,
                            character=characterization.build(view, mb))
        assert out.stat().st_size > 1000


def _pdf_text(expl: dict, size: tuple) -> str:
    import shutil
    import subprocess

    from bs3 import characterization, mbti, scores
    from bs3.pdf import build as pdf_build
    if not shutil.which("pdftotext"):
        raise unittest.SkipTest("pdftotext (poppler-utils) not installed")
    with tempfile.TemporaryDirectory() as d:
        r, paths, expl = _job_with_frames(Path(d), expl, size)
        view = scores.clean_view(r)
        mb = mbti.get_mbti(r, view)
        out = Path(d) / "report.pdf"
        pdf_build.build_pdf(view, out, explanation=expl, media={"fps": 30.0}, key_frames=paths, mbti=mb,
                            character=characterization.build(view, mb))
        return re.sub(r"\s+", " ", subprocess.run(["pdftotext", "-enc", "UTF-8", str(out), "-"],
                                                  capture_output=True, text=True, check=True).stdout)


def _caption_expl() -> dict:
    return _expl(signed={18: {"extraversion": 0.9, "agreeableness": 0.62}},
                 phrases={18: "улыбается, смотрит в камеру"}, expressions={18: HAPPY})


def test_pdf_prints_the_two_caption_lines():
    txt = _pdf_text(_caption_expl(), (200, 120))            # three frames in a row: both lines fit whole
    assert "· улыбается, смотрит в камеру" in txt
    assert "радость 62% · повысил экстраверсию" in txt
    assert "и коротко то, что на нём видно" in txt


def test_pdf_caption_in_a_narrow_cell_keeps_the_moment_and_the_direction():
    txt = _pdf_text(_caption_expl(), (120, 200))            # five portrait frames in a row: 34.8 mm per caption
    assert "· улыбается, смотрит в…" in txt                 # cut by words at 6 pt, never overflowing the frame
    assert "повысил экстраверсию" in txt                    # the shorter second line, not a cut «радость 62% · …»
    assert "радость 62% · повысил" not in txt


def test_phrase_must_be_three_to_six_words():
    """The prompt asks for three to six words; a one-word answer is not a description of the frame."""
    assert fc.clean_phrase("улыбается") is None
    assert fc.clean_phrase("кивает") is None
    assert fc.clean_phrase("голова прямо") is None
    assert fc.clean_phrase("смотрит в камеру") == "смотрит в камеру"
    assert fc.clean_phrase("голова прямо взгляд вниз руки вниз плечи") is None


def test_phrase_is_written_with_yo():
    """The report spells «на нём видно» with ё; a caption right under it must not print «вперед»."""
    assert fc.clean_phrase("смотрит вперед, голова прямо") == "смотрит вперёд, голова прямо"
    assert fc.clean_phrase("плечи развернуты, легкая улыбка") == "плечи развёрнуты, лёгкая улыбка"
    assert fc.clean_phrase("голова наклонена, взгляд опущен") == "голова наклонена, взгляд опущен"


def test_tooltip_never_names_a_class_with_no_percent():
    """«грусть 0%» reads as a mistake: a second class under half a percent is not named at all."""
    assert fc.expr_text([("нейтрально", 0.996), ("грусть", 0.0021)]) == "нейтрально 100%"
    assert fc.expr_text([("нейтрально", 0.9915), ("грусть", 0.0061)]) == "нейтрально 99%, грусть 1%"
    assert fc.expr_text([("радость", 0.6234), ("нейтрально", 0.2071)]) == "радость 62%, нейтрально 21%"


def test_moment_uses_the_frame_rate_of_the_clip():
    """The frame number counts frames of the segment clip, so the clip’s own rate turns it into a second; a job
    made before the rate was stored keeps the average rate of the whole video it was rendered with."""
    r = {"representative_segment": 1, "media": {"fps": 29.63},
         "timeline": [{"segment": 1, "start": 60.0, "end": 80.0}]}
    fs = ["/jobs/x/explain/key_26_frame538.jpg"]
    old, _ = fc.moments(r, fs)
    new, _ = fc.moments(r, fs, None, {"clip_fps": 30.0})
    assert fc.moment_text(old[0], False) == "1:18"
    assert fc.moment_text(new[0], False) == "1:17"


def test_narrow_pdf_caption_keeps_the_direction_of_a_long_trait_name():
    """«повысил эмоциональную стабильность» is 51 mm and never fits a 34.8 mm cell: the sign, which the
    owner asked for, must survive as «повысил эм. стаб.» rather than be dropped in favour of the expression."""
    expl = _expl(signed={18: {"emotional_stability": 0.9}}, phrases={18: "улыбается, смотрит в камеру"},
                 expressions={18: HAPPY})
    e = fc.build(REPORT, _paths(), expl)[2]
    assert e["effect_short"] == "повысил эмоциональную стабильность" and e["effect_mini"] == "повысил эм. стаб."
    assert fc.pdf_second_line(e).index("повысил эм. стаб.") < fc.pdf_second_line(e).index("радость 62%")
    txt = _pdf_text(expl, (120, 200))
    assert "повысил эм. стаб." in txt


def test_key_frames_follow_the_frames_that_produced_a_crop():
    """A clip that opens without a face: those leading frames are dropped from the crops, so the position of a
    crop is not the position of the uniform sampling and the saved picture must follow the crops."""
    try:
        import cv2
        import numpy as np
    except Exception:                                       # noqa: BLE001
        return
    from bs3.mm import explain, faces
    with tempfile.TemporaryDirectory() as d:
        vid = Path(d) / "clip.avi"
        w = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 64))
        if not w.isOpened():
            return
        for t in range(20):
            w.write(np.full((64, 64, 3), (t * 10, 0, 0), dtype=np.uint8))   # BGR: the blue channel is the number
        w.release()
        sampled = faces.select_uniform_frames(20, 10)
        orig, seen = faces.detect_faces, {"n": 0}

        def no_face_at_first(_im):
            seen["n"] += 1
            return [] if seen["n"] <= 3 else [(16, 16, 48, 48, 1024)]

        faces.detect_faces = no_face_at_first
        try:
            crops, st = faces.get_face_crops(str(vid), n_frames=10)
        finally:
            faces.detect_faces = orig
        assert st["fallback_frames"] == 3 and len(crops) == len(sampled) - 3
        assert st["frames_kept"] == sampled[3:]             # crops[i] is frame frames_kept[i] of the clip
        faces.detect_faces = lambda _im: []
        try:
            paths = explain.save_key_frames(str(vid), [0, 1], 10, Path(d) / "out", kept_frames=st["frames_kept"])
        finally:
            faces.detect_faces = orig
        assert [Path(p).name for p in paths] == [f"key_00_frame{sampled[3]}.jpg", f"key_01_frame{sampled[4]}.jpg"]
        assert round(int(cv2.imread(paths[0])[0, 0, 0]) / 10) == sampled[3]
