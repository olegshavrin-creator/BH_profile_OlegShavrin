"""Recompute WCAG contrast for every colour in bs2/palette.py. Exit code 1 if anything fails.
Usage: check_palette.py        (runs with plain python, no package install needed)
"""
import importlib.util
import re
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("palette", Path(__file__).resolve().parents[1] / "bs2" / "palette.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


def rgb(c):
    c = c.strip()
    m = re.match(r"rgba?\(([^)]+)\)", c)
    if m:
        parts = [float(x) for x in m.group(1).split(",")]
        return tuple(parts[:3]), (parts[3] if len(parts) > 3 else 1.0)
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4)), 1.0


def blend(fg, bg, op=1.0):
    """Colour actually seen: fg (with its own alpha and an extra CSS opacity) over an opaque bg."""
    (f, a), (b, _) = rgb(fg), rgb(bg)
    a *= op
    return tuple(a * fi + (1 - a) * bi for fi, bi in zip(f, b))


def lum(t):
    def ch(v):
        v /= 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = t
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def cr(fg, bg, op=1.0):
    a, b = lum(blend(fg, bg, op)), lum(rgb(bg)[0])
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


fails = []


def check(name, color, backgrounds, need, op=1.0):
    shown = color if op == 1 else f"{color}@{op}"
    for bg in dict.fromkeys(backgrounds):          # a theme with one colour for block and page is checked once
        r = cr(color, bg, op)
        flag = "OK " if r >= need else "FAIL"
        if r < need:
            fails.append(f"{name} {shown} on {bg}: {r:.2f} < {need}")
        print(f"  {flag} {name:34s} {shown:24s} on {bg}: {r:5.2f} (need {need})")


for theme, bgs in P.WEB_BACKGROUNDS.items():
    print(f"== web {theme} (block {bgs[0]}, page {bgs[1]})")
    T = P.THEME[theme]
    block, page = bgs
    ink = P.WEB_TEXT[theme]
    # page texts in the Gradio theme colour: notes .75, chart subtitles .85, the copyright line on the page background
    check("page text", ink, bgs, 4.5)
    check("note text (.75)", ink, bgs, 4.5, 0.75)
    check("chart subtitle (.85)", ink, (block,), 4.5, 0.85)
    check("copyright line", ink, (page,), 4.5, P.PAGE_NOTE_OPACITY)
    check("second-opinion bar (.55)", ink, (block,), 3.0, 0.55)
    # metric cards: a grey tint over the block (palette.CARD_TINT), so the value, the label and the 1 px outline of a
    # card are measured on the tinted colour and not on the bare block
    card = P.tint(block)
    check("card value on card", ink, (card,), 4.5)
    check("card label (.75) on card", ink, (card,), 4.5, 0.75)
    check("HTML.card_border on card", P.HTML["card_border"], (card,), 3.0)
    # selected tab name and underline (on the page), checked boxes and radios (on the block)
    check("accent (selected tab, checkbox)", P.ACCENT[theme], bgs, 4.5)
    for k in ("text", "muted", "hover_text"):
        check(f"text {k}", T[k], bgs if k != "hover_text" else (T["hover_bg"],), 4.5)
    check("axis line", T["axis"], bgs, 3.0)
    # chart gridlines are deliberately faint (as in BS 1.0): every series and every tick label carries the meaning, the
    # grid only helps to read a value off the axis. The floor keeps the value from drifting away unnoticed.
    check("grid line (decorative)", T["grid"], (block,), 1.3)
    check("hover label border", T["hover_border"], (T["hover_bg"],), 3.0)
    check("band border", T["band_border"], bgs, 3.0)
    check("no-data border", T["nodata_border"], bgs, 3.0)
    for group in ("TRAIT_WEB", "VOICE_WEB", "EMO_WEB", "BARS_WEB"):
        for k, c in getattr(P, group)[theme].items():
            check(f"{group}.{k}", c, bgs, 3.0)
    for k in ("main", "second"):
        check(f"RADAR_WEB.{k}", P.RADAR_WEB[theme][k], bgs, 3.0)
    for k, c in P.SPEECH_WEB[theme].items():
        check(f"SPEECH_WEB.{k}", c, bgs, 3.0)

print("== HTML (both themes)")
all_bgs = P.WEB_BACKGROUNDS["dark"] + P.WEB_BACKGROUNDS["light"]
for k in ("main_fill", "interview_fill", "track_outline", "card_border", "table_rule", "status_running", "status_done",
          "status_stopped", "status_error"):
    check(f"HTML.{k}", P.HTML[k], all_bgs, 3.0)

print("== Gradio theme overrides")
for k in ("BUTTON_PRIMARY", "BUTTON_PRIMARY_HOVER", "BUTTON_STOP", "BUTTON_STOP_HOVER"):
    check(f"white label on {k}", "#ffffff", (getattr(P, k),), 4.5)
check("hint text on white", P.SUBDUED_TEXT_LIGHT, P.WEB_BACKGROUNDS["light"], 4.5)

print("== PDF (white)")
for group in ("TRAIT_PDF", "VOICE_PDF", "EMO_PDF"):
    for k, c in getattr(P, group).items():
        check(f"{group}.{k}", c, (P.PDF_BACKGROUND,), 3.0)
for k, c in P.SPEECH_PDF.items():
    check(f"SPEECH_PDF.{k}", c, (P.PDF_BACKGROUND,), 3.0)
f = P.SCORE_BAR_PDF["fill"]
check("SCORE_BAR_PDF.fill", "#%02x%02x%02x" % f, ("#%02x%02x%02x" % ((P.SCORE_BAR_PDF["track"],) * 3), "#ffffff"), 3.0)
for k, c in P.RADAR_PDF.items():
    check(f"RADAR_PDF.{k}", c, (P.PDF_BACKGROUND,), 3.0)
check("SECOND_BAR_PDF (bar and its text)", P.SECOND_BAR_PDF, (P.PDF_BACKGROUND,), 4.5)
for k, c in P.BARS_PDF.items():
    check(f"BARS_PDF.{k}", c, (P.PDF_BACKGROUND,), 3.0)
for k, c in P.MODALITY_PDF.items():
    check(f"white label on MODALITY_PDF.{k}", "#ffffff", (c,), 4.5)
# emotions heatmap: step 1 must stand out from the empty cell, and every number printed on a cell (shares from 15%,
# steps 2-4) must read in the colour emo_heat_text_pdf picks
empty = P.EMO_HEAT_PDF["empty"]
for k in P.EMO_PDF:
    if k != "neutral":
        check(f"heatmap {k} step 1 vs empty cell", P.emo_heat_pdf(k, 1), (empty,), 1.8)
    for step in (2, 3, 4):
        cell = P.emo_heat_pdf(k, step)
        check(f"heatmap number on {k} step {step}", P.emo_heat_text_pdf(cell), (cell,), 4.5)

print("\nFAILURES:" if fails else "\nall checks passed")
for x in fails:
    print("  " + x)
sys.exit(1 if fails else 0)
