"""Web charts for the BS 2.0 interface (plotly, interactive). All functions take the result.json dict; they never compute
anything new. PDF charts live in pdf_charts.py.

Every figure builder is fig_xxx(rep, theme='dark'|'light') and takes all colours from palette.py. plot_html() builds
both variants and embeds them in one iframe; a small script picks the variant that matches the Gradio theme, waits
until the iframe is actually visible before drawing (charts in hidden tabs must not be measured at width 0), and
re-draws when the theme changes. The charts use the page font (Source Sans Pro of the Gradio Default theme): the
iframe loads it itself and draws once it is there (plotly measures legend and label widths with the font it has)."""
from __future__ import annotations

import html as _html
import json
from typing import List

from .norms import RU_TITLES, TRAIT_KEYS
from .palette import (BARS_WEB, EMO_ALIAS, EMO_WEB, FONT_FAMILY, RADAR_WEB, SPEECH_WEB, THEME, TRAIT_SYMBOL, TRAIT_WEB,
                      VOICE_SYMBOL, VOICE_WEB, emo)
from .report import seg_label

EMO_RU = {"joy": "радость", "surprise": "удивление", "neutral": "нейтрально", "sadness": "грусть", "fear": "страх",
          "anger": "злость", "disgust": "отвращение",
          "happy": "радость", "sad": "грусть", "angry": "злость"}
VOICE_RU = {"arousal": "возбуждение", "dominance": "уверенность", "valence": "позитивность"}

# flat {key: colour} dictionaries kept for older importers (pdf_charts.py); they are the light web palettes
TRAIT_COLORS = dict(TRAIT_WEB["light"])
EMO_COLORS = {**EMO_WEB["light"], **{alias: EMO_WEB["light"][k] for alias, k in EMO_ALIAS.items()}}
VOICE_COLORS = dict(VOICE_WEB["light"])

# legend names that differ from the shared RU_TITLES (the dotted line style of the interview series is named explicitly)
_TRAIT_NAME = {**RU_TITLES, "interview": "Пригласить на собеседование (пунктир)"}
# radar axis labels: long names are split so they are not cut off by a narrow column
_RADAR_LABEL = {"openness": "Открытость<br>опыту", "conscientiousness": "Добросовест-<br>ность",
                "extraversion": "Экстра-<br>версия", "agreeableness": "Доброжела-<br>тельность",
                "emotional_stability": "Эмоциональная<br>стабильность"}
_EMO_BAR_ORDER = ["joy", "surprise", "neutral", "sadness", "fear", "anger", "disgust"]
# the page font for the chart iframes (the same Google Fonts file as the Gradio theme, so it comes from the cache)
FONT_CSS = "https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600&display=swap"


def _segments(rep: dict) -> List[dict]:
    return [t for t in (rep.get("timeline") or []) if t.get("scores")]


def _x(segs: List[dict]):
    return [(t["start"] + t["end"]) / 2 for t in segs], [seg_label(t["start"], t["end"]) for t in segs]


def _theme(theme) -> str:
    return "light" if str(theme).lower() == "light" else "dark"


# ---------------------------------------------------------------- shared layout pieces
def _base(fig, theme: str, title: str, subtitle: str = "", height: int = 360, plot_h: int | None = None,
          hovermode="x unified", margin: dict | None = None, meta: dict | None = None):
    """Theme-aware chrome for every web figure: no plotly template, transparent background, legend ABOVE the plot."""
    T = THEME[theme]
    m = dict(meta or {})
    if plot_h:
        m["plot_h"] = plot_h
    fig.update_layout(
        template="none", height=height, autosize=True, margin=margin or dict(l=8, r=12, t=8, b=8),
        title=dict(text=title, subtitle=dict(text=subtitle, font=dict(size=12, color=T["muted"])), x=0.01, xanchor="left",
                   font=dict(size=15, color=T["text"])),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY, size=13, color=T["text"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(family=FONT_FAMILY, size=13, color=T["text"]), itemsizing="constant"),
        hovermode=hovermode,
        hoverlabel=dict(bgcolor=T["hover_bg"], bordercolor=T["hover_border"], align="left", namelength=-1,
                        font=dict(family=FONT_FAMILY, size=13, color=T["hover_text"])),
        modebar=dict(bgcolor="rgba(0,0,0,0)", color=T["muted"], activecolor=T["text"]),
        meta=m or None,
    )
    return fig


def _axes(fig, theme: str, **selector):
    """Axis lines, ticks, grid and titles for cartesian axes. template='none' drops automargin and brings back #444 zero
    lines, so both are set explicitly."""
    T = THEME[theme]
    common = dict(showgrid=True, gridcolor=T["grid"], gridwidth=1, zeroline=False, showline=True, linecolor=T["axis"],
                  linewidth=1, ticks="outside", tickcolor=T["axis"], ticklen=4, automargin=True,
                  tickfont=dict(family=FONT_FAMILY, size=12, color=T["muted"]),
                  title=dict(font=dict(family=FONT_FAMILY, size=13, color=T["text"]), standoff=10))
    fig.update_xaxes(**common, spikecolor=T["muted"], spikethickness=-2, spikedash="dot", **selector)
    fig.update_yaxes(**common, **selector)


def _clock(sec: float) -> str:
    s = int(round(sec))
    h, rest = divmod(s, 3600)
    m, ss = divmod(rest, 60)
    return f"{h}:{m:02d}:{ss:02d}" if h else f"{m}:{ss:02d}"


def _time_axis(fig, theme: str, t_max: float, axes=("xaxis",), with_title: bool = True, **selector) -> dict:
    """x axis in m:ss. Ticks every 60 s (30 s / 10 s / 5 s for short videos); the iframe script thins them out when the
    plot is too narrow. Returns the meta entry the script needs."""
    t_max = max(float(t_max or 0.0), 1.0)
    base = 60 if t_max > 240 else 30 if t_max > 90 else 10 if t_max > 20 else 5
    thin = {60: [1, 2, 5, 10, 15, 30, 60], 30: [1, 2, 4, 10, 20, 60], 10: [1, 3, 6, 12], 5: [1, 2, 4]}[base]
    vals = list(range(0, int(t_max) + 1, base))
    text = [_clock(v) for v in vals]
    kw = dict(range=[0, t_max], tickmode="array", tickvals=vals, ticktext=text, hoverformat=".0f",
              unifiedhovertitle=dict(text="отрезок около %{x:.0f} с"))
    fig.update_xaxes(**kw, **selector)
    if with_title:
        fig.update_xaxes(title_text="время, ч:мин:с" if t_max >= 3600 else "время, мин:с", **selector)
    return dict(vals=vals, text=text, axes=list(axes), steps=thin, min_px=int(7.5 * max(len(t) for t in text) + 18))


def _empty(fig, theme: str, title: str, subtitle: str, note: str, height: int = 150):
    T = THEME[theme]
    _base(fig, theme, title, subtitle, height=height, hovermode=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.add_annotation(text=note, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(family=FONT_FAMILY, size=14, color=T["muted"]))
    return fig


def _legend_proxy(fig, name: str, fill: str, border: str):
    """Legend-only entry for a shaded band (vrect shapes have no legend item of their own)."""
    import plotly.graph_objects as go
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=name, hoverinfo="skip",
                             marker=dict(symbol="square", size=14, color=fill, line=dict(color=border, width=1.5))))


# ---------------------------------------------------------------- plotly (web)
def fig_traits_timeline(rep: dict, theme: str = "dark"):
    import plotly.graph_objects as go
    theme = _theme(theme)
    T = THEME[theme]
    title, subtitle = "Big Five по ходу ролика", "оценка каждого отрезка по шкале от 0 до 1"
    fig = go.Figure()
    segs = _segments(rep)
    if not segs:
        return _empty(fig, theme, title, subtitle, "Оценок по отрезкам нет: ролик слишком короткий")
    timeline = rep.get("timeline") or segs
    x, labels = _x(timeline)
    colors = TRAIT_WEB[theme]
    keys = list(TRAIT_KEYS) + (["interview"] if all("interview" in t["scores"] for t in segs) else [])
    for k in keys:
        name = _TRAIT_NAME[k]
        fig.add_trace(go.Scatter(
            x=x, y=[(t.get("scores") or {}).get(k) for t in timeline], mode="lines+markers", name=name,
            connectgaps=False, customdata=labels, hovertemplate=name + ": %{y:.2f}<extra></extra>",
            line=dict(color=colors[k], width=2.5, dash="dot" if k == "interview" else "solid"),
            marker=dict(symbol=TRAIT_SYMBOL[k], size=8, color=colors[k], line=dict(width=0))))

    band_font = dict(family=FONT_FAMILY, size=12, color=T["hover_text"])
    t_max = max(t["end"] for t in timeline)
    # a band label grows away from the nearer plot edge, so it is never cut off at the right border
    side = lambda x0, x1: "right" if (x0 + x1) / 2 > 0.6 * t_max else "left"
    rep_i = rep.get("representative_segment")
    t_rep = next((t for t in segs if t["segment"] == rep_i), None) if rep_i else None
    if t_rep:
        fig.add_vrect(x0=t_rep["start"], x1=t_rep["end"], fillcolor=T["band"], layer="below",
                      line=dict(color=T["band_border"], width=1, dash="dash"), annotation_text="отрезок для объяснений",
                      annotation_position="top " + side(t_rep["start"], t_rep["end"]), annotation_font=band_font,
                      annotation_bgcolor=T["hover_bg"], annotation_bordercolor=T["hover_border"], annotation_borderpad=2)
        _legend_proxy(fig, "отрезок для объяснений (ключевые кадры)", T["band"], T["band_border"])
    # consecutive segments without scores are merged into one red band
    gaps: List[list] = []
    for t in timeline:
        if not t.get("scores"):
            if gaps and abs(gaps[-1][1] - t["start"]) < 1e-6:
                gaps[-1][1] = t["end"]
            else:
                gaps.append([t["start"], t["end"]])
    for g0, g1 in gaps:
        fig.add_vrect(x0=g0, x1=g1, fillcolor=T["nodata"], layer="below",
                      line=dict(color=T["nodata_border"], width=1, dash="dashdot"),
                      annotation_text="нет оценки", annotation_position="bottom " + side(g0, g1), annotation_font=band_font,
                      annotation_bgcolor=T["hover_bg"], annotation_bordercolor=T["nodata_border"], annotation_borderpad=2)
    if gaps:
        _legend_proxy(fig, "отрезок без оценки", T["nodata"], T["nodata_border"])

    _base(fig, theme, title, subtitle, height=390, plot_h=250)
    _axes(fig, theme)
    ticks = _time_axis(fig, theme, t_max)
    fig.update_yaxes(title_text="оценка (0 — низкая, 1 — высокая)", range=[-0.03, 1.03], tickmode="linear", tick0=0,
                     dtick=0.2, tickformat=".1f")
    fig.update_layout(meta={**fig.layout.meta, "time_ticks": ticks, "seg_hover": True})
    return fig


def fig_radar(rep: dict, theme: str = "dark"):
    import plotly.graph_objects as go
    theme = _theme(theme)
    T, R = THEME[theme], RADAR_WEB[theme]
    theta = [_RADAR_LABEL[k] for k in TRAIT_KEYS]
    full = [RU_TITLES[k] for k in TRAIT_KEYS]
    main = [rep["traits"][k]["score"] for k in TRAIT_KEYS]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=main + main[:1], theta=theta + theta[:1], customdata=full + full[:1], mode="lines+markers", fill="toself",
        fillcolor=R["main_fill"], name="Основная оценка", line=dict(color=R["main"], width=2.5),
        marker=dict(symbol="circle", size=7, color=R["main"]),
        hovertemplate="Основная оценка<br>%{customdata}: %{r:.2f}<extra></extra>"))
    var = rep.get("variant_scores") or {}
    primary = (rep.get("model") or {}).get("primary")
    for m, v in var.items():
        if m != primary and primary:
            vals = [v[k] for k in TRAIT_KEYS]
            fig.add_trace(go.Scatterpolar(
                r=vals + vals[:1], theta=theta + theta[:1], customdata=full + full[:1], mode="lines+markers",
                name="Второе мнение (своя модель, шкала FIV2)", line=dict(color=R["second"], width=2, dash="dash"),
                marker=dict(symbol="diamond", size=8, color=R["second"]),
                hovertemplate="Второе мнение<br>%{customdata}: %{r:.2f}<extra></extra>"))
    _base(fig, theme, "Профиль Big Five", "итоговые оценки по шкале от 0 до 1", height=450, plot_h=330,
          hovermode="closest", margin=dict(l=80, r=80, t=30, b=40))
    # spokes run counterclockwise from 0° (Открытость) every 72°; the scale sits at 180°, halfway between
    # Экстраверсия (144°) and Доброжелательность (216°), so its labels are horizontal and cross no spoke.
    # The top 10% of the plot area stays free for the two-line label of the 72° spoke (it must not touch the legend).
    fig.update_layout(polar=dict(
        bgcolor="rgba(0,0,0,0)", domain=dict(x=[0, 1], y=[0, 0.9]),
        radialaxis=dict(range=[0, 1], angle=180, showline=False, tickmode="array", tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                        ticktext=["0.2", "0.4", "0.6", "0.8", "1"], ticks="", tickangle=180,
                        tickfont=dict(family=FONT_FAMILY, size=11, color=T["text"]), gridcolor=T["grid"]),
        # invisible 8 px ticks push the trait names off the outer ring (without them «Экстра-версия» touches it)
        angularaxis=dict(gridcolor=T["grid"], linecolor=T["axis"], ticks="outside", ticklen=8,
                         tickcolor="rgba(0,0,0,0)", tickfont=dict(family=FONT_FAMILY, size=12, color=T["text"]))))
    return fig


def _stacked(fig, theme: str, x, labels, per_rows: List[dict], order: List[str], row=None, col=None, showlegend=True,
             group="e"):
    import plotly.graph_objects as go
    T = THEME[theme]
    for k in order:
        ys = [(r or {}).get(k, 0.0) for r in per_rows]
        kw = dict(row=row, col=col) if row else {}
        name = EMO_RU.get(k, k)
        fig.add_trace(go.Scatter(x=x, y=ys, mode="lines", stackgroup=group, name=name, fillcolor=emo(theme, k),
                                 line=dict(width=1, color=T["sep"]), showlegend=showlegend,
                                 legendgroup=EMO_ALIAS.get(k, k), customdata=labels,
                                 hovertemplate=name + ": %{y:.0%}<extra></extra>"), **kw)


def fig_emotions_timeline(rep: dict, theme: str = "dark"):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from .analyses.emotions_text import EMOTION_ORDER
    from .analyses.face_expr import EXPR_ORDER
    theme = _theme(theme)
    T = THEME[theme]
    title, subtitle = "Эмоции по ходу ролика", "доли эмоций в каждом отрезке, сумма = 100%"
    per = (rep.get("analyses") or {}).get("per_segment") or []
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.13,
                        subplot_titles=("Эмоции по речи (что говорит)", "Выражение лица (как выглядит)"))
    fig.update_annotations(font=dict(family=FONT_FAMILY, color=T["text"], size=14), xanchor="left", x=0)
    if not per:
        fig.layout.annotations = ()
        return _empty(fig, theme, title, subtitle, "Нет данных об эмоциях")
    x = [(r["start"] + r["end"]) / 2 for r in per]
    labels = [seg_label(r["start"], r["end"]) for r in per]
    _stacked(fig, theme, x, labels, [r.get("emotions_text") for r in per], EMOTION_ORDER, row=1, col=1,
             showlegend=False, group="t")
    _stacked(fig, theme, x, labels, [(r.get("face") or {}).get("expressions") for r in per], EXPR_ORDER, row=2, col=1,
             showlegend=False, group="f")
    # legend: solid squares in the band colour. The band traces themselves would show a 30x6 px swatch mostly covered
    # by the 5 px separator line (plotly draws legend lines at constant width). Same legendgroup, so a click on a
    # square still hides the emotion in both panels.
    for k in EMOTION_ORDER:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=EMO_RU.get(k, k),
                                 legendgroup=EMO_ALIAS.get(k, k), hoverinfo="skip",
                                 marker=dict(symbol="square", size=14, color=emo(theme, k), line=dict(width=0))),
                      row=1, col=1)
    plot_h = 470
    _base(fig, theme, title, subtitle, height=620, plot_h=plot_h)
    # the legend sits above the first panel title (26 px above the plot area)
    fig.update_layout(legend=dict(y=1 + 28 / plot_h, traceorder="normal",
                                  title=dict(text="Эмоция (цвета одинаковы на обеих панелях):", side="top",
                                             font=dict(family=FONT_FAMILY, size=13, color=T["text"]))))
    _axes(fig, theme)
    ticks = _time_axis(fig, theme, max(r["end"] for r in per), axes=("xaxis", "xaxis2"), with_title=False)
    fig.update_xaxes(title_text="время, ч:мин:с" if ticks["vals"][-1] >= 3600 else "время, мин:с", row=2, col=1)
    fig.update_yaxes(range=[0, 1], tickmode="linear", tick0=0, dtick=0.25, tickformat=".0%", title_text="доля, %")
    fig.update_layout(meta={**fig.layout.meta, "time_ticks": ticks, "seg_hover": True})
    return fig


def fig_voice_timeline(rep: dict, theme: str = "dark"):
    import plotly.graph_objects as go
    theme = _theme(theme)
    title, subtitle = "Голос по ходу ролика", "модель эмоций в речи, уровень от 0 до 1"
    per = (rep.get("analyses") or {}).get("per_segment") or []
    fig = go.Figure()
    rows = [r for r in per if r.get("voice")]
    if not rows:
        return _empty(fig, theme, title, subtitle, "Нет данных о голосе")
    x = [(r["start"] + r["end"]) / 2 for r in rows]
    labels = [seg_label(r["start"], r["end"]) for r in rows]
    colors = VOICE_WEB[theme]
    for d, ru in VOICE_RU.items():
        name = ru
        fig.add_trace(go.Scatter(x=x, y=[r["voice"].get(d) for r in rows], mode="lines+markers", name=name,
                                 line=dict(color=colors[d], width=2.5),
                                 marker=dict(symbol=VOICE_SYMBOL[d], size=8, color=colors[d], line=dict(width=0)),
                                 customdata=labels, hovertemplate=name + ": %{y:.2f}<extra></extra>"))
    _base(fig, theme, title, subtitle, height=380, plot_h=250)
    _axes(fig, theme)
    ticks = _time_axis(fig, theme, max(r["end"] for r in rows))
    fig.update_yaxes(title_text="уровень (0 — низкий, 1 — высокий)", range=[-0.03, 1.03], tickmode="linear", tick0=0,
                     dtick=0.2, tickformat=".1f")
    fig.update_layout(meta={**fig.layout.meta, "time_ticks": ticks, "seg_hover": True})
    return fig


def fig_speech_timeline(rep: dict, theme: str = "dark"):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    theme = _theme(theme)
    T, S = THEME[theme], SPEECH_WEB[theme]
    title, subtitle = "Речь по ходу ролика", "столбики — темп речи (левая ось), линия — доля пауз (правая ось)"
    per = (rep.get("analyses") or {}).get("per_segment") or []
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    rows = [r for r in per if r.get("speech")]
    if not rows:
        return _empty(fig, theme, title, subtitle, "Нет данных о речи")
    x = [(r["start"] + r["end"]) / 2 for r in rows]
    labels = [seg_label(r["start"], r["end"]) for r in rows]
    wpm = [r["speech"].get("words_per_min_speech") or 0 for r in rows]
    fillers = [r["speech"].get("fillers_per_100") or 0.0 for r in rows]
    fig.add_trace(go.Bar(
        x=x, y=wpm, width=[0.7 * (r["end"] - r["start"]) for r in rows], name="темп, слов/мин (левая ось)",
        marker=dict(color=S["bars"], line=dict(width=0)), opacity=1, customdata=[[lab, f] for lab, f in zip(labels, fillers)],
        hovertemplate="темп: %{y:.0f} слов/мин (слова-заполнители: %{customdata[1]:.1f} на 100 слов)<extra></extra>"),
        secondary_y=False)
    fig.add_trace(go.Scatter(
        x=x, y=[r["speech"].get("pause_share", 0) for r in rows], name="паузы, % времени (правая ось)",
        mode="lines+markers", line=dict(color=S["pauses"], width=2.5, dash="dash"),
        marker=dict(symbol="circle", size=7, color=S["pauses"], line=dict(color=T["sep"], width=1)),
        customdata=labels, hovertemplate="паузы: %{y:.0%} времени<extra></extra>"), secondary_y=True)
    _base(fig, theme, title, subtitle, height=380, plot_h=250)
    fig.update_layout(bargap=0.3)
    _axes(fig, theme)
    ticks = _time_axis(fig, theme, max(r["end"] for r in rows))
    # left axis: 5 equal steps, so its grid lines coincide with the 0/20/.../100% ticks of the right axis
    step = next((s for s in (5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 150, 160, 180, 200, 250, 300,
                             400, 500) if 5 * s >= max(wpm + [1]) * 1.05), 1000)
    fig.update_yaxes(title_text="темп, слов/мин (столбики)", range=[0, 5 * step], tickmode="linear", tick0=0, dtick=step,
                     secondary_y=False)
    fig.update_yaxes(title_text="паузы, % времени (линия)", range=[0, 1], tickmode="linear", tick0=0, dtick=0.2,
                     tickformat=".0%", showgrid=False, secondary_y=True)
    fig.update_layout(meta={**fig.layout.meta, "time_ticks": ticks, "seg_hover": True})
    return fig


def fig_emotion_bars(rep: dict, theme: str = "dark"):
    import plotly.graph_objects as go
    theme = _theme(theme)
    T, B = THEME[theme], BARS_WEB[theme]
    title, subtitle = "Средний профиль эмоций за ролик", "средняя доля каждой эмоции: по речи и по выражению лица"
    an = rep.get("analyses") or {}
    fig = go.Figure()
    text_mean = (an.get("emotions_text") or {}).get("mean") or {}
    face_mean = (an.get("face") or {}).get("mean") or {}
    if not text_mean and not face_mean:
        return _empty(fig, theme, title, subtitle, "Нет данных об эмоциях")
    face_map = {v: k for k, v in EMO_ALIAS.items()}           # joy -> happy, sadness -> sad, anger -> angry
    names = [EMO_RU[k] for k in _EMO_BAR_ORDER]
    value_labels = dict(texttemplate="%{y:.0%}", textposition="outside", cliponaxis=False,
                        textfont=dict(family=FONT_FAMILY, size=12, color=T["text"]))
    if text_mean:
        fig.add_trace(go.Bar(x=names, y=[text_mean.get(k, 0) for k in _EMO_BAR_ORDER], name="по речи (текст)",
                             marker=dict(color=B["speech"], line=dict(width=0)), **value_labels,
                             hovertemplate="по речи (текст): %{y:.0%}<extra></extra>"))
    if face_mean:
        # hatching is a second cue besides colour; in 'replace' mode the pattern background must be set explicitly
        fig.add_trace(go.Bar(x=names, y=[face_mean.get(face_map.get(k, k), 0) for k in _EMO_BAR_ORDER],
                             name="по лицу (кадры)", **value_labels,
                             marker=dict(color=B["face"], line=dict(width=0),
                                         pattern=dict(shape="/", fillmode="replace", bgcolor=B["face"],
                                                      fgcolor=T["sep"], size=8, solidity=0.25)),
                             hovertemplate="по лицу (кадры): %{y:.0%}<extra></extra>"))
    _base(fig, theme, title, subtitle, height=360, plot_h=240)
    # value labels shrink to the bar width; below 11 px they are hidden instead (hover still shows the value)
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08, barcornerradius=3,
                      uniformtext=dict(minsize=11, mode="hide"))
    _axes(fig, theme)
    fig.update_xaxes(showgrid=False, tickfont=dict(family=FONT_FAMILY, size=13, color=T["text"]),
                     autotickangles=[0, 45, 90])
    fig.update_yaxes(title_text="средняя доля за ролик", range=[0, 1], tickmode="linear", tick0=0, dtick=0.2,
                     tickformat=".0%")
    return fig


def fig_face_expr(rep: dict, theme: str = "dark"):
    """Horizontal bars of the facial-expression distribution over the whole video."""
    import plotly.graph_objects as go
    theme = _theme(theme)
    T = THEME[theme]
    title, subtitle = "Выражение лица за ролик", "доля проанализированных кадров с каждым выражением"
    m = ((rep.get("analyses") or {}).get("face") or {}).get("mean") or {}
    fig = go.Figure()
    if not m:
        return _empty(fig, theme, title, subtitle, "Лицо в кадре не найдено")
    items = sorted(m.items(), key=lambda kv: kv[1])
    fig.add_trace(go.Bar(
        x=[v for _, v in items], y=[EMO_RU.get(k, k) for k, _ in items], orientation="h", name="выражение лица",
        marker=dict(color=[emo(theme, k) for k, _ in items], line=dict(width=0)),
        text=[f"{v:.0%}" for _, v in items], textposition="outside", cliponaxis=False,
        textfont=dict(family=FONT_FAMILY, color=T["text"], size=13),
        hovertemplate="%{y}: %{x:.0%} кадров<extra></extra>"))
    _base(fig, theme, title, subtitle, height=330, plot_h=250, hovermode="closest", margin=dict(l=8, r=56, t=8, b=8))
    fig.update_layout(showlegend=False, bargap=0.3, barcornerradius=3)
    _axes(fig, theme)
    fig.update_xaxes(range=[0, min(1.0, max(m.values()) * 1.25) or 1.0], tickformat=".0%",
                     title_text="доля проанализированных кадров")
    fig.update_yaxes(showgrid=False, ticks="", tickfont=dict(family=FONT_FAMILY, size=13, color=T["text"]))
    return fig


# ---------------------------------------------------------------- iframe embedding
_FRAME_JS = r"""
(function(){
var SPEC=__SPEC__, EXTRA=__EXTRA__, FILL=__FILL__, FONT=__FONT__, INK=__INK__;
var gd=document.getElementById('g'), cur=null, curK=0, fitTimer=null, fitRuns=0, lastW=-1, fontReady=false;
var lastH=-1, natH=null, maxH=null;         // FILL only: last viewport height, natural and largest figure height
var CFG={responsive:true, displaylogo:false, locale:'ru',
  modeBarButtonsToRemove:['select2d','lasso2d','autoScale2d','zoomIn2d','zoomOut2d','toggleSpikelines',
    'hoverClosestCartesian','hoverCompareCartesian','toImage']};
// plotly's own texts (toolbar tooltips, the zoom hint) in Russian; no number format here, so 0.25 stays 0.25
if(typeof Plotly!=='undefined'){ try{ Plotly.register({moduleType:'locale', name:'ru', dictionary:{
  'Zoom':'Увеличить область', 'Pan':'Сдвигать', 'Reset axes':'Вернуть исходный масштаб', 'Reset views':'Вернуть исходный вид',
  'Reset view':'Вернуть исходный вид', 'Reset':'Сбросить', 'Autoscale':'Масштаб по данным', 'Zoom in':'Приблизить',
  'Zoom out':'Отдалить', 'Box Select':'Выделить прямоугольником', 'Lasso Select':'Выделить лассо',
  'Toggle Spike Lines':'Линии к осям', 'Show closest data on hover':'Подсказка по ближайшей точке',
  'Compare data on hover':'Подсказка по всем линиям', 'Toggle show closest data on hover':'Подсказка по ближайшей точке',
  'Download plot as a PNG':'Сохранить как картинку', 'Download plot':'Сохранить график',
  'Double-click to zoom back out':'Двойной щелчок — вернуть исходный масштаб',
  'Double-click on legend to isolate one trace':'Двойной щелчок по легенде — оставить только эту линию',
  'Taking snapshot - this may take a few seconds':'Сохраняю картинку, это может занять несколько секунд',
  'Snapshot succeeded':'Картинка сохранена', 'Sorry, there was a problem downloading your image!':'Не удалось сохранить картинку',
  'Produced with Plotly.js':'Построено библиотекой графиков', 'Share chart...':'Поделиться графиком'}}); }catch(e){} }
function isDark(){
  var readable=false;
  try{ for(var n=window.frameElement; n; n=n.parentElement){ readable=true; if(n.classList && n.classList.contains('dark')) return true; } }catch(e){}
  try{ var d=window.parent.document; if(d && d!==document){ readable=true;
    if((d.body && d.body.classList.contains('dark')) || d.documentElement.classList.contains('dark')) return true; } }catch(e){}
  if(readable) return false;
  try{ return window.matchMedia('(prefers-color-scheme: dark)').matches; }catch(e){ return true; }
}
function copy(o){ return JSON.parse(JSON.stringify(o)); }
// Visible = the iframe element itself has a box in the parent page. The iframe's own clientWidth is not enough: once a
// Gradio tab has been shown and hidden again, the hidden iframe keeps its old non-zero width.
function visible(){
  try{ var fe=window.frameElement;
    if(fe && (fe.offsetWidth<=0 || fe.getClientRects().length===0)) return false; }catch(e){}
  return document.documentElement.clientWidth>0;
}
var pollTimer=null;
function waitVisible(){
  if(pollTimer) return;
  pollTimer=setInterval(function(){ if(visible()){ clearInterval(pollTimer); pollTimer=null; draw(); } }, 300);
}
// FILL: the iframe height is the chart's natural height (its flex basis), never the drawn one, so a window stretched
// to its neighbour's height can shrink back when the neighbour gets shorter. maxH stops the growth (radar), the page
// CSS then centres the chart with its subtitle in the window.
function sizeFrame(){
  try{ var fe=window.frameElement, fl=gd._fullLayout; if(!fe || !fl) return;
    var h=Math.ceil((FILL && natH ? natH : fl.height)+EXTRA)+'px', mh=FILL && maxH ? Math.ceil(maxH+EXTRA)+'px' : '';
    if(fe.style.height===h && fe.style.maxHeight===mh) return;
    fe.style.height=h; fe.style.maxHeight=mh;
    // FILL: the figure follows the new iframe height; a hidden browser tab fires no resize event until it is shown
    if(FILL) scheduleFit(); }catch(e){}
}
function scheduleFit(){ clearTimeout(fitTimer); fitTimer=setTimeout(fit, 220); }
// A legend entry wider than the chart is cut off (plotly never wraps it): break such names before their "(…)" part.
function legendWrap(W){
  var over=false;
  gd.querySelectorAll('.legend .legendtext').forEach(function(t){ if(t.getBoundingClientRect().right>W-1) over=true; });
  if(!over) return null;
  var names=[], idx=[];
  (gd.data||[]).forEach(function(tr, i){ var n=tr.name||'';
    if(tr.showlegend!==false && n.indexOf('<br>')<0 && n.indexOf(' (')>0){ names.push(n.replace(' (', '<br>(')); idx.push(i); } });
  return idx.length ? {names:names, idx:idx} : null;
}
function fit(){
  var fl=gd._fullLayout, L=gd.layout; if(!fl || !fl._size || !L) return;
  if(!visible()) return;
  // Plotly pins layout.width as soon as the height has been relaid and then ignores container resizes
  // (Plots.resize does nothing when both are set), so the width follows the iframe here.
  var W=document.documentElement.clientWidth;
  if(W>0 && Math.abs(fl.width-W)>1){ Plotly.relayout(gd, {width:W}).then(function(){ fit(); }); return; }
  var wrap=legendWrap(W);
  if(wrap){ Plotly.restyle(gd, {name:wrap.names}, wrap.idx).then(function(){ fit(); }); return; }
  var m=L.meta||{}, upd={}, tt=m.time_ticks;
  if(L.polar && L.polar.radialaxis){
    // radar: the scale labels sit 0.2·r apart; on a small circle only every other label is kept
    var r=Math.min(fl._size.w, 0.9*fl._size.h)/2;
    var pt=r*0.2>=24 ? ['0.2','0.4','0.6','0.8','1'] : ['0.2','','0.6','','1'];
    if(pt.join()!==(L.polar.radialaxis.ticktext||[]).join()) upd['polar.radialaxis.ticktext']=pt;
  }
  if(tt && tt.vals && tt.vals.length>1){
    var need=tt.min_px*tt.vals.length/Math.max(40, fl._size.w), k=tt.steps[tt.steps.length-1];
    for(var i=0;i<tt.steps.length;i++){ if(tt.steps[i]>=need){ k=tt.steps[i]; break; } }
    if(k!==curK){ var v=[], t=[]; for(var j=0;j<tt.vals.length;j+=k){ v.push(tt.vals[j]); t.push(tt.text[j]); }
      tt.axes.forEach(function(a){ upd[a+'.tickvals']=v; upd[a+'.ticktext']=t; }); curK=k; }
  }
  if(m.plot_h && fitRuns<6){
    // radar in a narrow column: the circle is limited by the width, so do not keep empty space above and below it
    var target=L.polar ? Math.max(200, Math.min(m.plot_h, Math.round(fl._size.w)+60)) : m.plot_h;
    var diff=target-fl._size.h;
    if(FILL){
      // natural height as above; in a window stretched by its row the figure grows with the iframe. The radar grows
      // only while its circle can grow (plot height up to the width + 60, as above), so no empty band opens between
      // the legend and the circle
      natH=Math.round(Math.max(160, fl.height+diff));
      maxH=L.polar ? Math.max(natH, Math.round(fl.height-fl._size.h+fl._size.w+60)) : null;
      var want=Math.max(natH, Math.min(window.innerHeight-EXTRA, maxH || Infinity));
      if(Math.abs(want-fl.height)>2){ upd.height=want; fitRuns++; }
    } else if(Math.abs(diff)>2){ upd.height=Math.round(Math.max(160, fl.height+diff)); fitRuns++; } }
  if(Object.keys(upd).length){ Plotly.relayout(gd, upd).then(function(){ sizeFrame(); if('height' in upd) scheduleFit(); }); }
  else sizeFrame();
}
function draw(){
  if(!fontReady) return;                // the first draw comes from the font loader at the end of this script
  if(typeof Plotly==='undefined'){
    gd.textContent='График не загрузился: библиотека графиков загружается из интернета, а доступа к нему нет';
    gd.style.cssText='font:14px '+FONT+';padding:8px;color:'+INK[isDark()?'dark':'light']; return; }
  var mode=isDark()?'dark':'light';
  // hidden tab: plotly would measure text as 0x0 (overlapping legend items, lost bar labels), so wait until shown
  if(!visible()){ if(mode!==cur) waitVisible(); return; }
  var w=document.documentElement.clientWidth, h=window.innerHeight;
  if(w!==lastW || (FILL && h!==lastH)){ lastW=w; lastH=h; fitRuns=0; }
  if(mode===cur){ scheduleFit(); return; }
  cur=mode; curK=0; fitRuns=0;
  document.documentElement.setAttribute('data-theme', mode);
  var F=SPEC[mode];
  Plotly.newPlot(gd, copy(F.data), copy(F.layout), CFG).then(function(){ fit(); });
}
// unified hover: show the segment ("отрезок 5:00–5:20") instead of the raw midpoint in seconds
new MutationObserver(function(){
  var L=gd.layout; if(!L || !L.meta || !L.meta.seg_hover) return;
  var t=gd.querySelector('.hoverlayer .legendtitletext'); if(!t) return;
  var hd=gd._hoverdata||[], lab=null;
  for(var i=0;i<hd.length && lab===null;i++){ var cd=hd[i].customdata; if(Array.isArray(cd)) cd=cd[0]; if(typeof cd==='string') lab=cd; }
  if(lab===null) return;
  var txt='отрезок '+lab; if(t.textContent===txt) return;
  t.textContent=txt;
  try{ var box=t.closest('.legend'), bg=box && box.querySelector('rect.bg');
    if(bg){ var over=t.getBoundingClientRect().right+8-bg.getBoundingClientRect().right;
      if(over>0) bg.setAttribute('width', parseFloat(bg.getAttribute('width'))+over); } }catch(e){}
}).observe(gd, {childList:true, subtree:true, characterData:true});
try{ new ResizeObserver(draw).observe(document.documentElement); }catch(e){ window.addEventListener('resize', draw); }
// the observer sees only width changes (the document is as tall as the chart); stretching the iframe changes its height
if(FILL) window.addEventListener('resize', draw);
try{ var pd=window.parent.document; if(pd && pd!==document){
  var mo=new MutationObserver(draw);
  mo.observe(pd.body, {attributes:true, attributeFilter:['class']});
  mo.observe(pd.documentElement, {attributes:true, attributeFilter:['class']}); } }catch(e){}
try{ var mq=window.matchMedia('(prefers-color-scheme: dark)');
  if(mq.addEventListener) mq.addEventListener('change', draw); else mq.addListener(draw); }catch(e){}
// draw once the page font is loaded (Cyrillic and Latin faces, regular and semibold), but never wait long: without a
// connection plotly draws with the fallback font
var ready=Promise.resolve();
try{ if(document.fonts && document.fonts.load){
  var loads=['400 13px "Source Sans Pro"','600 13px "Source Sans Pro"'].map(function(f){ return document.fonts.load(f, 'Жж Aa 0'); });
  ready=Promise.race([Promise.all(loads), new Promise(function(res){ setTimeout(res, 1500); })]); } }catch(e){}
function start(){ fontReady=true; draw(); }
ready.then(start, start);
})();
"""


def plot_html(builder, rep: dict | None = None, extra_height: int = 24, fill: bool = False) -> str:
    """builder(rep, theme) -> figure. Both theme variants go into one <iframe srcdoc>; the chart title is rendered as
    HTML above the iframe (gr.HTML shows no label), so it uses the page text colour of the current Gradio theme.
    fill=True: for a chart in one of two windows side by side. The iframe may grow (flex, see webapp.APP_CSS) to the
    height of the neighbouring window and the figure grows with it instead of leaving an empty band under the chart;
    a radar grows only while its circle can grow, then the page centres it."""
    import plotly.io as pio
    from plotly.offline import get_plotlyjs_version

    if callable(builder):
        figs = {"dark": builder(rep, "dark"), "light": builder(rep, "light")}
    else:                                            # a ready figure: the same spec for both themes
        figs = {"dark": builder, "light": builder}
    lt = figs["dark"].layout.title
    title = (lt.text or "").strip()
    subtitle = ((lt.subtitle.text if lt.subtitle else "") or "").strip()
    height = int(figs["dark"].layout.height or 360)
    specs = []
    for name in ("dark", "light"):
        fig = figs[name]
        fig.update_layout(title=None)
        specs.append(f'"{name}":' + pio.to_json(fig, validate=False, remove_uids=True))
    spec = ("{" + ",".join(specs) + "}").replace("</", "<\\/").replace("<!--", "<\\!--")
    js = (_FRAME_JS.replace("__SPEC__", spec).replace("__EXTRA__", str(int(extra_height)))
          .replace("__FILL__", "true" if fill else "false").replace("__FONT__", json.dumps(FONT_FAMILY))
          .replace("__INK__", json.dumps({t: THEME[t]["text"] for t in ("dark", "light")})))
    cdn = f"https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"
    # radar scale labels (0.2 … 1) sit on top of the data lines: a background-coloured outline keeps them legible
    halo = ".radial-axis text{paint-order:stroke;stroke-width:3px;stroke-linejoin:round}" + "".join(
        f"html[data-theme={t}] .radial-axis text{{stroke:{THEME[t]['sep']}}}" for t in ("dark", "light"))
    doc = ("<!doctype html><html><head><meta charset='utf-8'>"
           f"<link rel='stylesheet' href='{FONT_CSS}'>"
           f"<style>html,body{{margin:0;padding:0;background:transparent;overflow:hidden}}#g{{width:100%}}{halo}</style>"
           f"<script src='{cdn}' charset='utf-8'></script></head><body><div id='g'></div><script>{js}</script>"
           "</body></html>")
    srcdoc = doc.replace("&", "&amp;").replace('"', "&quot;")
    head = ""
    if title:
        head = f"<div style='font-size:15px;font-weight:600;line-height:1.35;margin:10px 0 2px'>{_html.escape(title)}</div>"
        if subtitle:
            head += f"<div style='font-size:13px;line-height:1.35;opacity:.85;margin:0 0 4px'>{_html.escape(subtitle)}</div>"
    grow = "flex:1 0 auto;" if fill else ""
    return (head + f"<iframe style='width:100%;height:{height + int(extra_height)}px;{grow}border:0;display:block' "
            f"scrolling='no' srcdoc=\"{srcdoc}\"></iframe>")
