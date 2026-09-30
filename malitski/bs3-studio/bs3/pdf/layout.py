"""Page geometry, the grey of the notes and the fonts of the PDF report: the numbers the page (pdf/document.py), the
sections and the print charts (pdf/charts draws every chart at its printed width) share. Constants only."""
from __future__ import annotations

MARGIN_MM = 10                      # left, right and top page margin (the fpdf default, made explicit)
TEXT_W_MM = 210 - 2 * MARGIN_MM     # A4 text width: every chart (pdf/charts draws them this wide), table and row of frames
RADAR_W_MM, ROW_GAP_MM = 80, 4      # the radar beside the explanation: 80 mm + 4 mm gap + 106 mm of text
NOTE_GREY = 85                      # #555555, 7.46:1 on white: notes, card labels, the footer

FONT_CANDIDATES = [
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/mnt/c/Windows/Fonts/arial.ttf", "/mnt/c/Windows/Fonts/arialbd.ttf"),
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
]
