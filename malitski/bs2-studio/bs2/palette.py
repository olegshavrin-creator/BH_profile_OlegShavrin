"""Single source of colours for BS 2.0 (web charts, HTML infographics, PDF charts).

The page uses the Gradio Default theme with zinc neutrals and orange buttons, the same look as BS 1.0. Its backgrounds:
dark #27272a block / #0f0f11 page, light #ffffff block and page (the Gradio theme text is #f4f4f5 / #27272a).
Why two web palettes: plotly charts are transparent iframes, so they sit on the block background. No single text
colour reaches 4.5:1 on both themes, and several series colours cannot reach 3:1 on both, so every web figure is built
twice (theme='dark' / 'light') and the iframe picks one from the Gradio `dark` class on the page body.
PDF charts are on white paper and use their own, darker palette (PDF_*). Do not "unify" the palettes: a colour that
works on white fails on the dark block and vice versa. scripts/check_palette.py recomputes every contrast ratio.

Rules checked by scripts/check_palette.py (WCAG 2.1):
  text >= 4.5:1, graphical marks (lines, bars, markers, swatches, bar tracks) >= 3:1 on every background of the theme.
"""
from __future__ import annotations

WEB_BACKGROUNDS = {"dark": ("#27272a", "#0f0f11"), "light": ("#ffffff", "#ffffff")}     # (block, page)
WEB_TEXT = {"dark": "#f4f4f5", "light": "#27272a"}       # Gradio theme text: inherited by the HTML blocks
PDF_BACKGROUND = "#ffffff"

# ---------------------------------------------------------------- Gradio theme overrides (webapp: gr.themes.Default().set)
# white button labels need a darker fill than the Default orange-500 (2.8:1) and red-500 (3.8:1 in the light theme)
BUTTON_PRIMARY, BUTTON_PRIMARY_HOVER = "#c2410c", "#9a3412"     # orange-700 / 800: 5.2 / 7.3 with white
BUTTON_STOP, BUTTON_STOP_HOVER = "#b91c1c", "#991b1b"           # red-700 / 800: 6.5 / 8.3 with white
SUBDUED_TEXT_LIGHT = "#52525b"   # zinc-600: Radio info line and placeholders on white (the Default zinc-400 is 2.6:1)
# selected tab underline, checked boxes and radios (the theme accent): orange-500 is 2.8:1 on white, so the light
# theme takes orange-700 like the buttons; the dark theme keeps orange-500 (6.8:1 on the page, 5.3:1 on the block)
ACCENT = {"dark": "#f97316", "light": "#c2410c"}
# small notes on the page background (the «© AMLAI» line under the title): theme text colour at this opacity
PAGE_NOTE_OPACITY = 0.7

# ---------------------------------------------------------------- web chart chrome (text, axes, grid, hover)
# The light text, muted, axis and band colours are also the chrome of the PDF charts (pdf_charts.py reads them).
THEME = {
    "dark": dict(text="#e4e4e7", muted="#d4d4d8", grid="#3f3f46", axis="#808080", hover_bg="#18181b",
                 hover_border="#808080", hover_text="#f4f4f5", sep="#18181b", band="rgba(228,228,231,0.10)",
                 band_border="#a1a1aa", nodata="rgba(248,113,113,0.14)", nodata_border="#f87171"),
    "light": dict(text="#1f2937", muted="#374151", grid="#d4d4d8", axis="#6b7280", hover_bg="#ffffff",
                  hover_border="#6b7280", hover_text="#18181b", sep="#ffffff", band="rgba(31,41,55,0.08)",
                  band_border="#4b5563", nodata="rgba(185,28,28,0.10)", nodata_border="#b91c1c"),
}
# the Gradio Default theme font (Google Fonts, has Cyrillic); the chart iframes load it themselves (charts.FONT_CSS)
FONT_FAMILY = "'Source Sans Pro', ui-sans-serif, system-ui, sans-serif"

# ---------------------------------------------------------------- web series colours (>= 3:1 on both backgrounds of a theme)
TRAIT_WEB = {
    "dark": {"openness": "#60a5fa", "conscientiousness": "#34d399", "extraversion": "#fb923c",
             "agreeableness": "#c084fc", "emotional_stability": "#f87171", "interview": "#fbbf24"},
    "light": {"openness": "#1d4ed8", "conscientiousness": "#047857", "extraversion": "#c2410c",
              "agreeableness": "#7e22ce", "emotional_stability": "#be123c", "interview": "#a16207"},
}
TRAIT_SYMBOL = {"openness": "circle", "conscientiousness": "square", "extraversion": "triangle-up",
                "agreeableness": "diamond", "emotional_stability": "triangle-down", "interview": "star"}

# voice dimensions use hues that are NOT used by the traits (same «Таймлайн» tab)
VOICE_WEB = {
    "dark": {"arousal": "#f472b6", "dominance": "#2dd4bf", "valence": "#a3e635"},
    "light": {"arousal": "#be185d", "dominance": "#0f766e", "valence": "#4d7c0f"},
}
VOICE_SYMBOL = {"arousal": "circle", "dominance": "square", "valence": "diamond"}

# emotions: stacked bands are separated by THEME[t]["sep"] lines, so each band only needs >= 3:1 to the background
EMO_WEB = {
    "dark": {"joy": "#facc15", "surprise": "#fb923c", "neutral": "#a1a1aa", "sadness": "#60a5fa", "fear": "#a78bfa",
             "anger": "#f87171", "disgust": "#4ade80"},
    "light": {"joy": "#a16207", "surprise": "#c2410c", "neutral": "#71717a", "sadness": "#1d4ed8", "fear": "#6d28d9",
              "anger": "#b91c1c", "disgust": "#15803d"},
}
EMO_ALIAS = {"happy": "joy", "sad": "sadness", "angry": "anger"}     # face-expression labels -> text-emotion keys

# radar and score bars: the main score is blue, the second opinion is a neutral grey (another model, another scale)
RADAR_WEB = {
    "dark": dict(main="#60a5fa", main_fill="rgba(96,165,250,0.22)", second="#d4d4d8"),
    "light": dict(main="#1d4ed8", main_fill="rgba(29,78,216,0.16)", second="#52525b"),
}
SPEECH_WEB = {
    "dark": dict(bars="#808080", pauses="#f4f4f5"),
    "light": dict(bars="#71717a", pauses="#18181b"),
}
BARS_WEB = {"dark": dict(speech="#60a5fa", face="#fb923c"), "light": dict(speech="#1d4ed8", face="#c2410c")}

# ---------------------------------------------------------------- HTML infographics (inline styles, must work on BOTH themes)
HTML = dict(
    main_fill="#3b82f6",                 # score bars: 4.05 on #27272a, 3.68 on white
    interview_fill="#b7791f",            # 4.09 / 3.64
    second_fill="currentColor",          # second opinion: theme text colour at opacity .55 (5.2 dark / 3.5 light)
    # outlines, frames and rules: the neutral grey of BS 1.0, 3.8 on #27272a, 4.8 on #0f0f11, 3.9 on white
    track="transparent", track_outline="#808080",
    card_border="#808080", table_rule="#808080",
    # status bar on the page background, the BS 1.0 colours: orange while running, sea green done, red stopped
    # by the user (status_stopped) or broken off by a failure (status_error)
    status_running="#ea580c", status_done="#2e8b57", status_stopped="#dc2626", status_error="#dc2626",
)
# the metric cards (webapp.CARD) are this grey tint over the block, so their text and their 1 px border sit on the
# tinted colour and not on the bare block: tint() below composites it, scripts/check_palette.py measures them there
CARD_TINT = 0.06

# ---------------------------------------------------------------- PDF (white paper)
TRAIT_PDF = {"openness": "#1d4ed8", "conscientiousness": "#0f766e", "extraversion": "#d97706",
             "agreeableness": "#7e22ce", "emotional_stability": "#b91c1c", "interview": "#8a6d3b"}
TRAIT_MARKER_PDF = {"openness": "o", "conscientiousness": "s", "extraversion": "^", "agreeableness": "D",
                    "emotional_stability": "v", "interview": "o"}
VOICE_PDF = {"arousal": "#be185d", "dominance": "#0e7490", "valence": "#4d7c0f"}
VOICE_MARKER_PDF = {"arousal": "o", "dominance": "s", "valence": "^"}
EMO_PDF = {"joy": "#a16207", "surprise": "#db2777", "neutral": "#64748b", "sadness": "#1f5a99", "fear": "#6d28d9",
           "anger": "#b91c1c", "disgust": "#15803d"}
SPEECH_PDF = dict(bars="#64748b", pauses="#111827")
SCORE_BAR_PDF = dict(track=242, outline=130, fill=(29, 78, 216), interview=(138, 109, 59), mid_tick=85)
# score bars in the PDF: the chart colour of each trait; extraversion one step darker (#d97706 gives 2.9:1 on the
# #f2f2f2 track, #b45309 gives 4.5:1), every other fill already has >= 4:1
TRAIT_BAR_PDF = {**TRAIT_PDF, "extraversion": "#b45309"}
# radar and the second-opinion bars of the PDF, as on the web: the main score blue, the second opinion slate (another
# model, another scale); the slate is also the colour of the second-opinion text under each bar (7.58:1)
RADAR_PDF = dict(main="#1d4ed8", second="#475569")
SECOND_BAR_PDF = "#475569"
# average emotion profile: speech blue, face orange with white hatching (a second cue besides colour), as on the web
BARS_PDF = dict(speech="#1d4ed8", face="#c2410c")
# modality contributions: white percentages inside the segments (>= 4.5:1); face orange and speech blue mean the same
# as in BARS_PDF
MODALITY_PDF = {"face": "#c2410c", "audio": "#0e7490", "text": "#1d4ed8", "behavior": "#64748b", "scene": "#6d28d9"}
# key-fact and speech cards: outline grey 130 (as the score-bar track, decorative) on a #f7f7f7 fill
CARD_PDF = dict(outline=130, fill=247)

# «Эмоции по ходу ролика» in the PDF is a heatmap: a cell has the EMO_PDF hue of its emotion at one of five steps of
# the share (EMO_HEAT_STEPS). Every hue reaches the same lightness at a step (CIE L*, EMO_HEAT_L), so «30–49%» looks
# equally strong in amber and in blue; hues lighter than the target (amber, pink, green at the top step) are darkened
# past their EMO_PDF colour. Step 1 keeps >= 1.8:1 to the empty cell, so it survives a laser printer. Every hue,
# «нейтрально» included, follows the same ramp: the legend promises that a darker cell is a larger share, and a
# «нейтрально 98%» painted lighter than a «страх 35%» would turn the chart upside down for the reader
EMO_HEAT_PDF = dict(empty="#f1f3f6")
EMO_HEAT_STEPS = (0.05, 0.15, 0.30, 0.50)          # lower bounds of steps 1-4, rounded %; below 5% the cell is empty
EMO_HEAT_L = (74.0, 58.0, 45.0, 33.0)              # CIE L* of steps 1-4 (the empty cell is L* 96)


def emo(theme: str, key: str) -> str:
    return EMO_WEB[theme].get(EMO_ALIAS.get(key, key), "#888888")


def emo_pdf(key: str) -> str:
    return EMO_PDF.get(EMO_ALIAS.get(key, key), "#888888")


def _hex_rgb(c: str) -> tuple:
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) / 255 for i in (0, 2, 4))


def tint(bg: str, alpha: float = CARD_TINT) -> str:
    """The card colour actually seen: grey 128 at `alpha` over the background (dark block -> #2c2c2f, white -> #f7f7f7)."""
    return "#%02x%02x%02x" % tuple(round(255 * (alpha * 128 / 255 + (1 - alpha) * c)) for c in _hex_rgb(bg))


def luminance(c: str) -> float:
    """WCAG relative luminance of '#rrggbb'."""
    lin = [v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4 for v in _hex_rgb(c)]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def lightness_to_luminance(l_star: float) -> float:
    """CIE L* -> relative luminance Y."""
    f = (l_star + 16) / 116
    return f ** 3 if f ** 3 > 0.008856 else (l_star / 903.3)


def emo_heat_step(share: float) -> int:
    """Step 0-4 of a share; by the rounded percentage, the number printed in the cell («30» is never in step 2)."""
    pct = round(max(0.0, min(1.0, float(share))) * 100)
    return sum(pct >= round(e * 100) for e in EMO_HEAT_STEPS)


def emo_heat_pdf(key: str, step: int) -> str:
    """Cell colour of an emotion at a step: the empty cell for step 0; otherwise the colour on the path empty cell ->
    EMO_PDF colour -> the same hue darker, at the luminance of EMO_HEAT_L[step - 1]."""
    e = _hex_rgb(EMO_HEAT_PDF["empty"])
    key = EMO_ALIAS.get(key, key)
    if step <= 0:
        return EMO_HEAT_PDF["empty"]
    c = _hex_rgb(emo_pdf(key))
    target = lightness_to_luminance(EMO_HEAT_L[step - 1])

    def at(t: float) -> str:          # t 0…1: empty -> colour; 1…2: colour -> 45% darker, luminance falls all the way
        rgb = [a + (b - a) * t for a, b in zip(e, c)] if t <= 1 else [b * (1 - 0.45 * (t - 1)) for b in c]
        return "#%02x%02x%02x" % tuple(round(255 * max(0.0, min(1.0, v))) for v in rgb)
    lo, hi = 0.0, 2.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if luminance(at(mid)) > target else (lo, mid)
    return at(hi)


def emo_heat_text_pdf(cell: str) -> str:
    """Number colour on a heatmap cell: white or the chart ink, whichever contrasts more, and black where neither
    reaches 4.5:1 (mid-light cells)."""
    ink = THEME["light"]["text"]
    y = luminance(cell)
    cr = lambda a, b: (max(a, b) + 0.05) / (min(a, b) + 0.05)
    c_white, c_ink = cr(1.0, y), cr(luminance(ink), y)
    if max(c_white, c_ink) >= 4.5:
        return "#ffffff" if c_white >= c_ink else ink
    return "#ffffff" if c_white >= cr(0.0, y) else "#000000"
