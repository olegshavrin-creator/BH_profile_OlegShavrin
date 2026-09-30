"""The web UI of BS Profiler 3.1 (refactoring plan of 3.1, stages 18–19).

The pieces: app.py (the Gradio app — build_app, main, the status bar, pdf_for_download and the error wording), page.py
(page_outputs and the HTML block builders of the result page), style.py (the app CSS, the theme and the Russian
locale), charts.py (the interactive plotly figures of the page), plotframe.py (embeds a figure as an <iframe srcdoc>
with the theme switch and the page font), parts.py (the HTML helpers — tables, score bars, the modality table) and
mbti_html.py (the tab «Тип MBTI» and the short emotion paragraph).
"""
