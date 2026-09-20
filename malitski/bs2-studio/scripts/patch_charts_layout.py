"""One-off patch: make every plotly chart fill the container width (autosize), move legends below the plot so they
never overlap the title, and use a text colour readable on both Gradio themes."""
import re
import sys
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "bs2" / "charts.py"
s = p.read_text(encoding="utf-8")
orig = s

s = s.replace('VOICE_COLORS = {"arousal": "#e8731a", "dominance": "#2e8b57", "valence": "#4c8bf5"}',
              'VOICE_COLORS = {"arousal": "#e8731a", "dominance": "#2e8b57", "valence": "#4c8bf5"}\n'
              'FONT_COLOR = "#a9adb3"          # readable on both the light and the dark Gradio theme')

s = s.replace(
    '    fig.update_layout(title=dict(text=title, x=0.01, font=dict(size=15)), height=height, margin=dict(l=40, r=20, t=48, b=40),\n'
    '                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0), hovermode="x unified",\n'
    '                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(size=12))',
    '    fig.update_layout(title=dict(text=title, x=0.01, y=0.98, font=dict(size=15)), height=height, autosize=True,\n'
    '                      margin=dict(l=50, r=20, t=48, b=80), legend=dict(orientation="h", yanchor="top", y=-0.22, x=0),\n'
    '                      hovermode="x unified", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",\n'
    '                      font=dict(size=12, color=FONT_COLOR))')

s = s.replace(
    '                      height=360, margin=dict(l=30, r=30, t=40, b=30), legend=dict(orientation="h", y=-0.1),\n'
    '                      paper_bgcolor="rgba(0,0,0,0)", title=dict(text="Профиль Big Five", x=0.01, font=dict(size=15)))',
    '                      height=380, autosize=True, margin=dict(l=30, r=30, t=40, b=60), legend=dict(orientation="h", y=-0.12),\n'
    '                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color=FONT_COLOR),\n'
    '                      title=dict(text="Профиль Big Five", x=0.01, y=0.98, font=dict(size=15)))')

s = s.replace(
    '    fig.update_layout(height=560, margin=dict(l=40, r=20, t=60, b=40), hovermode="x unified",\n'
    '                      legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0),\n'
    '                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",\n'
    '                      title=dict(text="Эмоции по ходу ролика (доли, сумма = 100%)", x=0.01, font=dict(size=15)))',
    '    fig.update_layout(height=600, autosize=True, margin=dict(l=50, r=20, t=70, b=80), hovermode="x unified",\n'
    '                      legend=dict(orientation="h", yanchor="top", y=-0.12, x=0), font=dict(color=FONT_COLOR),\n'
    '                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",\n'
    '                      title=dict(text="Эмоции по ходу ролика (доли, сумма = 100%)", x=0.01, y=0.99, font=dict(size=15)))')

s = s.replace(
    '    fig.update_layout(title=dict(text="Речь по ходу ролика", x=0.01, font=dict(size=15)), height=340,\n'
    '                      margin=dict(l=40, r=40, t=48, b=40), legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),\n'
    '                      hovermode="x unified", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", bargap=0.35)',
    '    fig.update_layout(title=dict(text="Речь по ходу ролика", x=0.01, y=0.98, font=dict(size=15)), height=340, autosize=True,\n'
    '                      margin=dict(l=50, r=50, t=48, b=80), legend=dict(orientation="h", yanchor="top", y=-0.22, x=0),\n'
    '                      hovermode="x unified", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", bargap=0.35,\n'
    '                      font=dict(color=FONT_COLOR))')

s = s.replace(
    '    fig.update_layout(barmode="group", height=320, margin=dict(l=40, r=20, t=48, b=40),\n'
    '                      title=dict(text="Средний профиль эмоций за ролик", x=0.01, font=dict(size=15)),\n'
    '                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0), plot_bgcolor="rgba(0,0,0,0)",\n'
    '                      paper_bgcolor="rgba(0,0,0,0)")',
    '    fig.update_layout(barmode="group", height=340, autosize=True, margin=dict(l=50, r=20, t=48, b=70),\n'
    '                      title=dict(text="Средний профиль эмоций за ролик", x=0.01, y=0.98, font=dict(size=15)),\n'
    '                      legend=dict(orientation="h", yanchor="top", y=-0.18, x=0), plot_bgcolor="rgba(0,0,0,0)",\n'
    '                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color=FONT_COLOR))')

if s == orig:
    sys.exit("nothing patched")
p.write_text(s, encoding="utf-8")
print("autosize:", s.count("autosize=True"), "| FONT_COLOR uses:", s.count("FONT_COLOR"))
