"""Web charts for the BS Profiler 3.1 interface (plotly, interactive). All functions take the result.json dict; they never compute
anything new. PDF charts live in pdf/charts.py; the iframe embedding of these figures lives in web/plotframe.py.

Every figure builder is fig_xxx(rep, theme='dark'|'light') and takes all colours from palette.py. The figures use the
page font (Source Sans Pro of the Gradio Default theme, FONT_FAMILY); web/plotframe.plot_html loads that font in the
iframe so plotly measures legend and label widths with the font it has."""
from __future__ import annotations

from typing import List

from ..labels import EMO_RU, EMOTION_ORDER, EXPR_ORDER, RADAR_LABEL, VOICE_RU, model_title
from ..norms import RU_TITLES, TRAIT_KEYS
from ..palette import (BARS_WEB, EMO_ALIAS, FONT_FAMILY, RADAR_WEB, SPEECH_WEB, THEME, TRAIT_SYMBOL, TRAIT_WEB,
                       VOICE_SYMBOL, VOICE_WEB, emo)
from ..scores import shown_model
from ..segments import representative, scored
from ..textfmt import clock, seg_label

# legend names that differ from the shared RU_TITLES (the dotted line style of the interview series is named explicitly)
_TRAIT_NAME = {**RU_TITLES, "interview": "Пригласить на собеседование (пунктир)"}


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


def _time_axis(fig, theme: str, t_max: float, axes=("xaxis",), with_title: bool = True, **selector) -> dict:
    """x axis in m:ss. Ticks every 60 s (30 s / 10 s / 5 s for short videos); the iframe script thins them out when the
    plot is too narrow. Returns the meta entry the script needs."""
    t_max = max(float(t_max or 0.0), 1.0)
    base = 60 if t_max > 240 else 30 if t_max > 90 else 10 if t_max > 20 else 5
    thin = {60: [1, 2, 5, 10, 15, 30, 60], 30: [1, 2, 4, 10, 20, 60], 10: [1, 3, 6, 12], 5: [1, 2, 4]}[base]
    vals = list(range(0, int(t_max) + 1, base))
    text = [clock(v) for v in vals]
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
    segs = scored(rep)
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
    t_rep = representative(rep)             # among the scored segments, however many there are
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
    theta = [RADAR_LABEL[k] for k in TRAIT_KEYS]
    full = [RU_TITLES[k] for k in TRAIT_KEYS]
    main = [rep["traits"][k]["score"] for k in TRAIT_KEYS]
    # one model per analysis (3.1): the trace is named after it («OCEAN-AI, веса MuPTA» / «AMLAI 1.0»)
    name = model_title(shown_model(rep))
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=main + main[:1], theta=theta + theta[:1], customdata=full + full[:1], mode="lines+markers", fill="toself",
        fillcolor=R["main_fill"], name=name, line=dict(color=R["main"], width=2.5),
        marker=dict(symbol="circle", size=7, color=R["main"]),
        hovertemplate=name + "<br>%{customdata}: %{r:.2f}<extra></extra>"))
    _base(fig, theme, "Профиль Big Five", "оценки модели по шкале от 0 до 1", height=450, plot_h=330,
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
    names = [EMO_RU[k] for k in EMOTION_ORDER]
    value_labels = dict(texttemplate="%{y:.0%}", textposition="outside", cliponaxis=False,
                        textfont=dict(family=FONT_FAMILY, size=12, color=T["text"]))
    if text_mean:
        fig.add_trace(go.Bar(x=names, y=[text_mean.get(k, 0) for k in EMOTION_ORDER], name="по речи (текст)",
                             marker=dict(color=B["speech"], line=dict(width=0)), **value_labels,
                             hovertemplate="по речи (текст): %{y:.0%}<extra></extra>"))
    if face_mean:
        # hatching is a second cue besides colour; in 'replace' mode the pattern background must be set explicitly
        fig.add_trace(go.Bar(x=names, y=[face_mean.get(face_map.get(k, k), 0) for k in EMOTION_ORDER],
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
