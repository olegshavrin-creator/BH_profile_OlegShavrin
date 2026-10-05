"""The PDF report of BS Profiler 3.1 (refactoring plan of 3.1, stages 12–17).

The pieces: layout.py (the page geometry, the grey of the notes, the fonts), document.py (the page itself, class Report),
widgets.py (its card grid and score bars), fmt.py (the text helpers of the report), charts.py (the print charts),
mbti_section.py (the characterization block and the MBTI section), sections.py (the passport and the other sections of
the main part), frames.py (what drove the score of AMLAI 1.0: the key frames, the modality chart, the words),
appendix.py (the lettered appendices) and build.py (the plan, the render and build_pdf/export_pdf).
"""
from .build import build_pdf, export_pdf

__all__ = ["build_pdf", "export_pdf"]
