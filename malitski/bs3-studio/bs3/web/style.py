"""Styling and locale of the BS Profiler 3.1 page (Gradio): the app CSS (APP_CSS) and the sizes it uses (VIDEO_H,
CHAR_MIN_PX), the copyright line, the theme (_theme), and forcing the Gradio shell into Russian (RU_LOCALE_JS,
force_russian_gradio)."""
from __future__ import annotations

from ..palette import (ACCENT, BUTTON_PRIMARY, BUTTON_PRIMARY_HOVER, BUTTON_STOP, BUTTON_STOP_HOVER, PAGE_NOTE_OPACITY,
                       SUBDUED_TEXT_LIGHT)


# the Code icon Gradio puts into every gr.HTML label chip reads as «code»: hide it on the result blocks.
# Chart blocks: charts.plot_html puts the chart title (bold 15 px div) and its subtitle above the iframe; the block chip
# already shows the same title, so the bold title line is hidden there and the subtitle (units, scale) stays.
APP_CSS = (".bs3-block > label[data-testid='block-label'] > span{display:none}"
           ".prose.bs3-chart > div:first-child[style*='font-weight:600']{display:none}")
COPYRIGHT = "© 2026 AMLAI"
# the copyright line under the title: 12 px, theme text colour dimmed (8.8:1 on the dark page, 5.6:1 on white), pulled
# up to the title so it reads as part of the heading, with the usual gap before the description line
APP_CSS += (f".gradio-container .prose p.bs3-copy{{font-size:12px;line-height:16px;letter-spacing:.03em;"
            f"opacity:{PAGE_NOTE_OPACITY};margin:-8px 0 12px}}")
# Gradio paints several marks with the theme accent, orange-500: the name and the underline of the selected tab (and
# the «…» button of the tabs that do not fit on a narrow screen), the ring of the upload progress and the icons of the
# video player. On white orange-500 is 2.8:1, so the whole page takes the accent of palette.ACCENT instead (orange-700
# in the light theme, as the checked boxes in _theme). The theme sets the variable on <html> and on body.dark, so a
# rule on the container wins for everything inside it in both themes.
APP_CSS += (f".gradio-container{{--color-accent:{ACCENT['light']}}}"
            f".dark .gradio-container{{--color-accent:{ACCENT['dark']}}}")
# result.json: the code viewer colours its tokens with fixed pale hues (2.0-2.8:1 on white), so the file is shown in the
# theme text colour. Both themes, although the dark hues do reach 4.5:1: one grey page, and the same block either way
APP_CSS += ".bs3-json .cm-content span{color:inherit!important}"
# height of the video window (gr.Video height). 3.1: the left column (video, model radio, buttons) sets the height of
# the compact «Характеристика личности» window next to it; at 240 px the column is about 420 px high and the tabs
# stay on the first screen of a 1080p display (with 300 px the pair alone was 480 px)
VIDEO_H = 240
# Two framed windows side by side (gr.Row with class bs3-pair): both columns get the height of the taller one, and in
# each column the window marked bs3-grow takes the extra height, so the two frames start and end on one line with no
# page background under the shorter one. A text box stretches its text area (the text fills the frame instead of
# scrolling in a small box), a chart block stretches its iframe (charts.plot_html fill=True redraws the figure at that
# height), the video window stretches its drop zone or player, a plain HTML block extends its frame under the content.
# Gradio's own equal_height grows every block of a column; here only the marked one grows. On a narrow screen the
# columns wrap onto separate lines, and a line is as tall as its only column, so nothing is stretched there.
APP_CSS += (
    ".gradio-container .row.bs3-pair{align-items:stretch}"
    ".row.bs3-pair>.column>.bs3-grow,.row.bs3-pair>.column>.form>.bs3-grow{flex-grow:1}"
    # a text box sits in a .form wrapper whose inline style (flex-grow:0, from scale) only !important overrides
    ".row.bs3-pair>.column>.form:has(>.bs3-grow){flex-grow:1!important;flex-wrap:nowrap}"
    # text box: block > label.container > title chip + .input-container > textarea
    ".row.bs3-pair .bs3-grow.block:has(textarea),.row.bs3-pair .bs3-grow>label.container,"
    ".row.bs3-pair .bs3-grow .input-container{display:flex;flex-direction:column;flex-grow:1}"
    ".row.bs3-pair .bs3-grow>label.container>[data-testid='block-info']{align-self:flex-start}"
    ".row.bs3-pair .bs3-grow textarea{flex-grow:1}"
    # chart: block > title chip + .html-container > .prose > subtitle + iframe (flex:1 0 auto from plot_html); a chart
    # that stops growing (the radar, max-height from the iframe script) is centred together with its subtitle
    ".row.bs3-pair>.column>.bs3-grow.bs3-chart,.row.bs3-pair .bs3-grow.bs3-chart>.html-container,"
    ".row.bs3-pair .prose.bs3-grow.bs3-chart{display:flex;flex-direction:column;flex-grow:1}"
    ".row.bs3-pair .prose.bs3-grow.bs3-chart{justify-content:center}"
    ".row.bs3-pair>.column>.bs3-grow.bs3-chart>label{align-self:flex-start}"
    # video: Gradio fixes the block height inline (height=VIDEO_H)
    f".row.bs3-pair>.column>.bs3-grow:has(.video-container){{height:auto!important;min-height:{VIDEO_H}px;"
    "display:flex;flex-direction:column}"
    ".row.bs3-pair .bs3-grow .video-container{flex-grow:1}")
# «Характеристика личности» (change request 3.1, section 4): a compact window like the 2.0 «Краткие выводы» box.
# The block is a flex item of its column with a zero flex basis, so the row's height comes from the left column
# (video, model radio, buttons, about 300 px of video plus the controls); align-items:stretch gives the right column
# that height and the block fills it. Inside, the label keeps its size and the html-container takes the rest and
# scrolls (min-height:0 lets it shrink below its content). CHAR_MIN_PX keeps the window readable when the columns
# wrap onto separate lines on a narrow screen: there the block is exactly this high and the text scrolls inside.
# 15 px text, line height 1.55, at most 75 characters per line, paragraphs 10 px apart with bold leads; the HTML of
# characterization.html() carries the same values inline, these rules keep Gradio's prose styles from overriding them.
CHAR_MIN_PX = 320
APP_CSS += (f".row.bs3-pair>.column>.bs3-char{{display:flex;flex-direction:column;flex:1 1 0;min-height:{CHAR_MIN_PX}px}}"
            ".bs3-char>.html-container{flex:1 1 0;min-height:0;overflow-y:auto;overflow-x:hidden}"
            ".bs3-char .bs3-char-text{font-size:15px;line-height:1.55;max-width:75ch}"
            ".bs3-char .bs3-char-text p{margin:0 0 10px}.bs3-char .bs3-char-text p b{font-weight:700}")


def _theme():
    """The BS 1.0 look: Gradio Default theme (zinc neutrals, Source Sans Pro, block titles in the block corner) with
    readable buttons and hints: white labels on orange-700 / red-700 (the Default orange-500 gives 2.8:1, red-500 3.8:1),
    zinc-600 hint text on white (zinc-400 gives 2.6:1), and checked boxes and radios in orange-700 on white, where the
    Default orange-500 is 2.8:1 (the dark theme keeps orange-500). The selected tab underline follows in APP_CSS."""
    import gradio as gr
    return gr.themes.Default().set(
        button_primary_background_fill=BUTTON_PRIMARY, button_primary_background_fill_dark=BUTTON_PRIMARY,
        button_primary_background_fill_hover=BUTTON_PRIMARY_HOVER, button_primary_background_fill_hover_dark=BUTTON_PRIMARY_HOVER,
        button_cancel_background_fill=BUTTON_STOP, button_cancel_background_fill_dark=BUTTON_STOP,
        button_cancel_background_fill_hover=BUTTON_STOP_HOVER, button_cancel_background_fill_hover_dark=BUTTON_STOP_HOVER,
        body_text_color_subdued=SUBDUED_TEXT_LIGHT, body_text_color_subdued_dark="*neutral_400",
        # placeholders have their own variable (light zinc-400 2.6:1 on white, dark zinc-500 3.1:1 on the block)
        input_placeholder_color=SUBDUED_TEXT_LIGHT, input_placeholder_color_dark="*neutral_400",
        checkbox_background_color_selected=ACCENT["light"], checkbox_background_color_selected_dark=ACCENT["dark"],
        checkbox_border_color_selected=ACCENT["light"], checkbox_border_color_selected_dark=ACCENT["dark"],
        checkbox_border_color_focus=ACCENT["light"], checkbox_border_color_focus_dark=ACCENT["dark"],
    )


# Gradio's own texts (upload area «Перетащите видео сюда», buttons, footer) come from its translations and follow the
# browser language, so an English browser showed them in English. The page reports a Russian browser before the
# Gradio bundle reads navigator.language; this has to run before that bundle, which the `head` of gr.Blocks does not
# (Gradio inserts it later, after its translations are set up).
RU_LOCALE_JS = ("<script>try{['language','languages'].forEach(function(k){Object.defineProperty(Navigator.prototype,k,"
                "{configurable:true,get:function(){return k==='language'?'ru-RU':['ru-RU','ru'];}});});}catch(e){}</script>")


def force_russian_gradio() -> None:
    """Serve Gradio's index page with RU_LOCALE_JS at the top of <head> and lang="ru" (Gradio 5 renders the page
    from a Jinja template; its loader is wrapped once per process)."""
    import jinja2
    from gradio import routes
    env = routes.templates.env
    if getattr(env.loader, "bs3_russian", False):
        return
    base = env.loader

    class RussianIndexLoader(jinja2.BaseLoader):
        bs3_russian = True

        def get_source(self, environment, template):
            source, filename, uptodate = base.get_source(environment, template)
            if template.endswith("index.html") and RU_LOCALE_JS not in source:
                source = source.replace("<head>", "<head>" + RU_LOCALE_JS, 1).replace('lang="en"', 'lang="ru"', 1)
            return source, filename, uptodate

    env.loader = RussianIndexLoader()
