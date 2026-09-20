"""PDF charts for BS 2.0: matplotlib PNGs placed on the A4 pages by pdf_report.py, one print version for every chart
of the web page.

Colours come from palette.py (the *_PDF dictionaries, contrast-checked on white by scripts/check_palette.py), never
from the web palettes: a colour that reads on the dark Gradio block fails on paper and vice versa.

Sizes: every figure is drawn at its printed size (TEXT_W_MM wide — the page width minus the margins of
pdf_report.Report — except the radar, which stands beside the plain-language explanation), and pdf_report places it at
exactly that width, so 1 pt in matplotlib is 1 pt on paper: tick labels, legends and value labels 8 pt, axis titles
9 pt, chart titles 9.5-10 pt. PNGs are rendered at 300 dpi. Every legend sits OUTSIDE the axes, under them, so the
plot area keeps the full width and the text never covers data. The time charts share one plot area (PLOT_LEFT_MM /
PLOT_RIGHT_MM), so 0:00 and the last tick are at the same place on all of them.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .charts import _EMO_BAR_ORDER, _RADAR_LABEL, EMO_RU, VOICE_RU, _segments
from .norms import RU_TITLES, TRAIT_KEYS
from .palette import (BARS_PDF, EMO_ALIAS, EMO_HEAT_L, EMO_HEAT_PDF, MODALITY_PDF, RADAR_PDF, SPEECH_PDF, THEME,
                      TRAIT_MARKER_PDF, TRAIT_PDF, VOICE_MARKER_PDF, VOICE_PDF, emo_heat_pdf, emo_heat_step,
                      emo_heat_text_pdf, emo_pdf)
from .pdf_report import TEXT_W_MM, _empty_text, _seg_words

log = logging.getLogger("bs2.pdf")

NAN = float("nan")
_L = THEME["light"]                 # white paper = the light web theme's chrome colours
INK = _L["text"]                    # titles, axis labels (14.7:1 on white)
MUTED = _L["muted"]                 # tick labels (10.3:1)
AXIS = _L["axis"]                   # spines (4.83:1); also the quiet «речь» / «лицо» row labels of the heatmap
GRID = "#d0d0d0"                    # supplementary grid lines, deliberately quiet (1.54:1)
OUTLINE = "#333333"                 # outline of legend swatches, so light bands still show a clear edge (12.6:1)
NODATA_HATCH = AXIS                 # grey hatching = "no data" in every PDF chart
BARS_TEXT = "#475569"               # tick labels / axis title of the tempo scale: slate like the bars, 7.58:1 as text

RC = {
    "font.size": 8.5, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlecolor": INK, "axes.labelsize": 9, "axes.labelcolor": INK, "axes.edgecolor": AXIS,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "legend.fontsize": 8, "legend.title_fontsize": 8, "legend.frameon": False,
    "figure.titlesize": 10, "figure.titleweight": "bold", "figure.dpi": 300, "savefig.dpi": 300,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "hatch.linewidth": 0.6,
}
# printed heights, mm (the width is TEXT_W_MM unless WIDTH_MM says otherwise); the modality chart grows with its
# rows (20 mm + 6 mm per row); the rows of the emotions heatmap are ~3.7 mm
HEIGHT_MM = {"traits": 70, "profile": 86, "emotion_profile": 58, "face_expr": 46, "emotions": 98, "voice": 56,
             "speech": 56}
WIDTH_MM = {"profile": 80}          # the radar stands beside the explanation (80 mm + 4 mm gap + 106 mm of text)
TIME_TICKS = 13                     # at most this many labels on a full-width time axis (6 min -> every 30 s)
SERIES_LW, SERIES_MS = 1.6, 3.5     # line width / marker size of the series, pt on paper
# horizontal extent of the plot area of every time chart, mm from the left and the right edge of the image: the left
# margin fits the row names of the emotions heatmap («отвращение» at 8.5 pt with its colour bar), the right one the
# pause scale of the speech chart. Charts stacked on a page then share one time axis (0:30 at the same x on all of
# them); a chart whose labels need more room (a four-digit tempo scale) takes it and is the only one off by that much
PLOT_LEFT_MM, PLOT_RIGHT_MM = 23.0, 11.5


def _figure(plt, name: str, nrows: int = 1, height_mm: float | None = None, **kw):
    """Figure at its printed size: WIDTH_MM[name] (TEXT_W_MM by default) × HEIGHT_MM[name]."""
    w = WIDTH_MM.get(name, TEXT_W_MM)
    h = height_mm if height_mm is not None else HEIGHT_MM[name]
    return plt.subplots(nrows, 1, figsize=(w / 25.4, h / 25.4), layout="constrained", **kw)


def _align_plot(fig) -> None:
    """Replace the horizontal extent of every axes (twin axes too) by the common PLOT_LEFT_MM / PLOT_RIGHT_MM. Constrained
    layout still places titles, the legend and the vertical margins; it is switched off afterwards, so the common x
    extent is not recomputed per figure."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px_mm = fig.bbox.width / TEXT_W_MM
    axes = fig.axes
    x0 = min(ax.get_position().x0 for ax in axes) * fig.bbox.width
    x1 = max(ax.get_position().x1 for ax in axes) * fig.bbox.width
    # what needs room beside the plot: the two axes (their tick labels and titles) and the marks placed next to the
    # rows. The chart title is deliberately NOT measured: it starts flush with the plot and may be wider than it, and
    # measuring it would pull one chart's plot inwards and break the time axis shared by the charts of a page
    boxes = [b for ax in axes for b in (ax.xaxis.get_tightbbox(renderer), ax.yaxis.get_tightbbox(renderer)) if b]
    boxes += [a.get_window_extent(renderer) for ax in axes for a in ax.texts
              if a.get_visible() and str(a.get_text()).strip()]
    need_l = (x0 - min(b.x0 for b in boxes)) / px_mm + 0.8
    need_r = (max(b.x1 for b in boxes) - x1) / px_mm + 0.8
    left, right = max(PLOT_LEFT_MM, need_l), max(PLOT_RIGHT_MM, need_r)
    fig.set_layout_engine("none")
    for ax in axes:
        p = ax.get_position()
        ax.set_position([left / TEXT_W_MM, p.y0, 1 - (left + right) / TEXT_W_MM, p.height])


def _rgba(css: str) -> Tuple[float, float, float, float]:
    """'rgba(31,41,55,0.08)' or '#1f2937' -> matplotlib RGBA tuple."""
    s = css.strip()
    if s.startswith("rgba"):
        r, g, b, a = [float(v) for v in s[s.index("(") + 1:s.index(")")].split(",")]
        return r / 255, g / 255, b / 255, a
    s = s.lstrip("#")
    return int(s[0:2], 16) / 255, int(s[2:4], 16) / 255, int(s[4:6], 16) / 255, 1.0


def _num(v) -> float:
    try:
        return NAN if v is None else float(v)
    except (TypeError, ValueError):
        return NAN


def _pct(v: float) -> str:
    """0.834 -> '83%'; a share that rounds to 0 but is not 0 is '<1%', as in the web tables."""
    p = float(v) * 100
    return "<1%" if 0 < p < 0.95 else f"{p:.0f}%"


def _duration(rep: dict, rows: List[dict]) -> float:
    ends = [float(r.get("end") or 0) for r in rows]
    return max([float(rep.get("duration_sec") or 0)] + ends + [1.0])


def _time_axis(ax, dur: float, max_ticks: int = TIME_TICKS, label: bool = True) -> None:
    """x axis from 0 to the end of the video with round steps; m:ss labels for any length («0:05» in a 15-second
    clip), as in the tables and on the web charts; h:mm:ss from an hour on."""
    from matplotlib.ticker import FuncFormatter, MultipleLocator
    ax.set_xlim(0, dur)
    step = next((s for s in (5, 10, 15, 20, 30, 60, 120, 180, 300, 600, 900, 1200, 1800, 3600) if dur / s <= max_ticks), 7200)
    ax.xaxis.set_major_locator(MultipleLocator(step))
    if dur >= 3600:
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda v, _: f"{int(round(v)) // 3600}:{int(round(v)) % 3600 // 60:02d}:{int(round(v)) % 60:02d}"))
        text = "время ролика, ч:мин:с"
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v)) // 60}:{int(round(v)) % 60:02d}"))
        text = "время ролика, мин:с"
    if label:
        ax.set_xlabel(text)


def _fig_legend(fig, handles, labels, **kw):
    """Legend outside the axes, under the whole chart: the fewest rows that fit the figure width, then the fewest
    columns for that many rows (7 entries on two rows are 4 + 3, not 6 + 1), read row by row like text. A legend
    above the axes would collide with a suptitle: constrained layout reserves the top margin for only one of them."""
    renderer = fig.canvas.get_renderer()
    n = len(handles)
    if not n:
        return None
    for rows in range(1, n + 1):
        ncols = math.ceil(n / rows)
        # matplotlib fills a legend column by column, the first columns getting the extra entries; listing the
        # entries column by column out of a row-major grid makes the rows read left to right
        order = [r * ncols + c for c in range(ncols) for r in range(math.ceil(n / ncols)) if r * ncols + c < n]
        args = ([handles[i] for i in order], [labels[i] for i in order])
        if ncols == 1:
            break
        probe = fig.legend(*args, ncols=ncols, **kw)        # measured and removed again
        fits = probe.get_window_extent(renderer).width <= 0.99 * fig.bbox.width
        probe.remove()
        if fits:
            break
    return fig.legend(*args, loc="outside lower center", ncols=ncols, **kw)


def _nodata_patch(label: str):
    from matplotlib.patches import Patch
    return Patch(facecolor="white", edgecolor=NODATA_HATCH, hatch="///", lw=0.6, label=label)


def _nodata_span(ax, start: float, end: float) -> None:
    ax.axvspan(start, end, facecolor="white", edgecolor=NODATA_HATCH, hatch="///", lw=0, zorder=0.6)


def _hide_spines(ax, *sides: str) -> None:
    for s in sides:
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------- Big Five profile (radar)
def _radar_chart(plt, rep: dict, out_dir: Path) -> Optional[str]:
    """Print version of charts.fig_radar: the main score filled blue, the second opinion slate and dashed on top; the
    spokes run counterclockwise from 0° (Открытость on the right) every 72°, the scale sits at 180°."""
    import matplotlib.patheffects as pe
    traits = rep.get("traits") or {}
    if not all(k in traits for k in TRAIT_KEYS):
        return None
    model = rep.get("model") or {}
    primary = model.get("primary")
    var = rep.get("variant_scores") or {}
    second = next((v for m, v in var.items() if primary and m != primary and all(k in v for k in TRAIT_KEYS)), None)
    fig = plt.figure(figsize=(WIDTH_MM["profile"] / 25.4, HEIGHT_MM["profile"] / 25.4), layout="constrained")
    ax = fig.add_subplot(projection="polar")
    ang = [2 * math.pi * i / len(TRAIT_KEYS) for i in range(len(TRAIT_KEYS))]
    closed = ang + ang[:1]
    ax.set_theta_offset(0); ax.set_theta_direction(1)
    ax.set_ylim(0, 1)
    # every 0.2 gets a ring, but only every second one a number: five numbers along one spoke run into each other and
    # into the data polygon they cross
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "", "0.6", "", "1"], fontsize=8, color=MUTED)
    ax.set_rlabel_position(180)
    for t in ax.get_yticklabels():            # the scale labels cross the data lines: a white halo keeps them legible
        t.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])
    ax.set_xticks(ang, [_RADAR_LABEL[k].replace("<br>", "\n") for k in TRAIT_KEYS], fontsize=8, color=INK)
    ax.tick_params(axis="x", pad=5)
    ax.grid(color=GRID, lw=0.5)
    ax.spines["polar"].set_color(AXIS); ax.spines["polar"].set_linewidth(0.8)
    main = [float(traits[k]["score"]) for k in TRAIT_KEYS]
    if primary == "oceanai" and model.get("lang") == "ru":
        main_label = "Основная оценка (OCEAN-AI, шкала MuPTA)"
    elif primary:
        main_label = "Основная оценка"
    else:
        main_label = "Итоговая оценка (среднее двух систем)" if len(var) == 2 else "Итоговая оценка"
    ax.plot(closed, main + main[:1], color=RADAR_PDF["main"], lw=SERIES_LW, marker="o", ms=SERIES_MS, label=main_label,
            zorder=3, clip_on=False)
    ax.fill(closed, main + main[:1], color=RADAR_PDF["main"], alpha=0.14, lw=0, zorder=2)
    if second:
        vals = [float(second[k]) for k in TRAIT_KEYS]
        ax.plot(closed, vals + vals[:1], color=RADAR_PDF["second"], lw=1.3, dashes=(4, 2), marker="D", ms=3.2,
                label="Второе мнение (своя модель, шкала FIV2)", zorder=4, clip_on=False)
    fig.suptitle("Профиль Big Five", x=0.02, ha="left", fontsize=9.5)
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncols=1, handlelength=2.2)
    p = out_dir / "chart_profile.png"
    fig.savefig(p); plt.close(fig)
    return str(p)


# ---------------------------------------------------------------- Big Five timeline
def _traits_chart(plt, rep: dict, out_dir: Path) -> Optional[str]:
    from matplotlib.patches import Patch
    from matplotlib.transforms import blended_transform_factory
    segs = _segments(rep)
    if not segs:
        return None
    timeline = rep.get("timeline") or segs        # full timeline: skipped segments break the lines (NaN) instead of
    dur = _duration(rep, timeline)                # being silently interpolated across
    fig, ax = _figure(plt, "traits")
    keys = list(TRAIT_KEYS) + (["interview"] if all("interview" in t["scores"] for t in segs) else [])
    x = [(t["start"] + t["end"]) / 2 for t in timeline]
    for k in keys:
        ax.plot(x, [_num((t.get("scores") or {}).get(k)) if t.get("scores") else NAN for t in timeline],
                color=TRAIT_PDF[k], marker=TRAIT_MARKER_PDF[k], ms=SERIES_MS, lw=SERIES_LW, ls=":" if k == "interview" else "-",
                label=RU_TITLES[k], clip_on=False, zorder=3)
    handles, labels = ax.get_legend_handles_labels()
    rep_i = rep.get("representative_segment")
    t_rep = next((t for t in segs if t["segment"] == rep_i), None) if rep_i and len(segs) > 1 else None
    if t_rep:
        band, border = _rgba(_L["band"]), _L["band_border"]
        ax.axvspan(t_rep["start"], t_rep["end"], facecolor=band, edgecolor=border, lw=0.8, ls="--", zorder=1)
        ax.text((t_rep["start"] + t_rep["end"]) / 2, 0.02, "★", transform=blended_transform_factory(ax.transData, ax.transAxes),
                ha="center", va="bottom", fontsize=10, color=border, zorder=4)
        handles.append(Patch(facecolor=band, edgecolor=border, lw=0.8, ls="--"))
        labels.append("★ отрезок для объяснений")
    skipped = [t for t in (rep.get("timeline") or []) if not t.get("scores")]
    for t in skipped:
        _nodata_span(ax, t["start"], t["end"])
    if skipped:
        handles.append(_nodata_patch("")); labels.append("отрезок пропущен (нет оценки)")
    ax.set_ylim(0, 1); ax.set_ylabel("оценка 0…1"); ax.set_title("Big Five по ходу ролика")
    _time_axis(ax, dur)
    ax.set_axisbelow(True); ax.grid(color=GRID, lw=0.5)
    _fig_legend(fig, handles, labels, handlelength=2.2, columnspacing=1.4)
    _align_plot(fig)
    p = out_dir / "chart_traits.png"
    fig.savefig(p); plt.close(fig)
    return str(p)


# ---------------------------------------------------------------- average emotion profile (speech vs face)
def _emotion_profile_chart(plt, rep: dict, out_dir: Path) -> Optional[str]:
    """Print version of charts.fig_emotion_bars: grouped bars per emotion, speech blue, face orange with white
    hatching. The chart carries one legend, its own: the emotion colours belong to «Эмоции по ходу ролика», where
    every row already has its colour bar, and a second row of chips under these bars only read as a broken legend."""
    from matplotlib.ticker import PercentFormatter
    an = rep.get("analyses") or {}
    text_mean = (an.get("emotions_text") or {}).get("mean") or {}
    face_mean: Dict[str, float] = {}
    for k, v in ((an.get("face") or {}).get("mean") or {}).items():
        face_mean[EMO_ALIAS.get(k, k)] = face_mean.get(EMO_ALIAS.get(k, k), 0.0) + float(v or 0)
    series = [(m, lab, col, hatch) for m, lab, col, hatch in (
        (text_mean, "по речи (текст)", BARS_PDF["speech"], None), (face_mean, "по лицу (кадры)", BARS_PDF["face"], "///"))
        if m]
    if not series:
        return None
    fig, ax = _figure(plt, "emotion_profile")
    w = 0.38 if len(series) == 2 else 0.6
    for i, (m, lab, col, hatch) in enumerate(series):
        off = (i - (len(series) - 1) / 2) * w
        ys = [max(0.0, _num(m.get(k)) if not math.isnan(_num(m.get(k))) else 0.0) for k in _EMO_BAR_ORDER]
        # white hatching on the face bars: the hatch takes the edge colour, the edge itself is not drawn (lw=0)
        bars = ax.bar([j + off for j in range(len(ys))], ys, width=w, color=col, label=lab, zorder=2,
                      **(dict(hatch=hatch, edgecolor="white", linewidth=0) if hatch else dict(linewidth=0)))
        for b, v in zip(bars, ys):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.0%}", ha="center", va="bottom", fontsize=8,
                    color=INK, zorder=3)
    ax.set_ylim(0, 1.1)                        # room for the label of a bar at 100%
    ax.set_yticks([0, .2, .4, .6, .8, 1]); ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("средняя доля за ролик")
    ax.set_xticks(range(len(_EMO_BAR_ORDER)), [EMO_RU[k] for k in _EMO_BAR_ORDER], fontsize=8.5, color=INK)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.set_xlim(-0.6, len(_EMO_BAR_ORDER) - 0.4)
    ax.set_axisbelow(True); ax.grid(axis="y", color=GRID, lw=0.5)
    _hide_spines(ax, "top", "right")
    ax.set_title("Средний профиль эмоций за ролик")
    h, lab = ax.get_legend_handles_labels()
    _fig_legend(fig, h, lab, handlelength=1.8, columnspacing=1.6)
    p = out_dir / "chart_emotion_profile.png"
    fig.savefig(p); plt.close(fig)
    return str(p)


# ---------------------------------------------------------------- facial expression over the whole video
def _face_expr_chart(plt, rep: dict, out_dir: Path) -> Optional[str]:
    """Print version of charts.fig_face_expr: horizontal bars, the most frequent expression on top."""
    from matplotlib.ticker import PercentFormatter
    m = ((rep.get("analyses") or {}).get("face") or {}).get("mean") or {}
    items = [(k, float(v or 0)) for k, v in m.items()]
    if not items:
        return None
    items.sort(key=lambda kv: kv[1])            # barh draws from the bottom: the largest ends on top
    fig, ax = _figure(plt, "face_expr")
    top = max(v for _, v in items)
    xmax = min(1.0, top * 1.25) or 1.0
    ys = list(range(len(items)))
    ax.barh(ys, [v for _, v in items], height=0.6, color=[emo_pdf(k) for k, _ in items], zorder=2)
    for y, (_, v) in zip(ys, items):
        ax.text(v + xmax * 0.008, y, f"{v:.0%}", va="center", ha="left", fontsize=8, color=INK, zorder=3)
    ax.set_yticks(ys, [EMO_RU.get(k, k) for k, _ in items], fontsize=8.5, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, xmax); ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("доля разобранных кадров")
    ax.set_axisbelow(True); ax.grid(axis="x", color=GRID, lw=0.5)
    _hide_spines(ax, "top", "right")
    ax.set_title("Выражение лица за ролик")
    p = out_dir / "chart_face_expr.png"
    fig.savefig(p); plt.close(fig)
    return str(p)


# ---------------------------------------------------------------- emotions over time: heatmap, speech and face paired
HEAT_ROWS = ["joy", "surprise", "sadness", "fear", "anger", "disgust", "neutral"]     # «нейтрально» last and apart
HEAT_SOURCES = {"text": "речь", "face": "лицо"}
HEAT_MIN_LABEL = 15                 # numbers from this share (%), where they fit into the cell
HEAT_STEP_LABELS = ["меньше 5 % (пустая клетка)", "5–14 %", "15–29 %", "30–49 %", "50 % и больше"]
HEAT_DECOR_MM = 34                  # title, subtitle, time axis and legend of the heatmap; the rows get the rest
# the strength steps of the legend are shown in three of the hues of the chart at once (amber, blue, slate), side by
# side in one swatch: a single hue would look like the legend of that one row, and a grey ramp like the legend of
# «нейтрально»; three hues at the same lightness say what the step really means — the same share in any colour
HEAT_LEGEND_HUES = ("joy", "sadness", "neutral")


def emotion_shares(r: dict, source: str) -> Optional[Dict[str, float]]:
    """{emotion: share} of one segment rescaled to sum 1 (rounded model outputs leave 0.99…), or None = no data. Face
    labels are mapped to the text keys (happy -> joy …); an empty transcript is no data, not «neutral 100%»."""
    d = ((r.get("face") or {}).get("expressions") or {}) if source == "face" else ({} if _empty_text(r) else
                                                                                   (r.get("emotions_text") or {}))
    vals = {k: 0.0 for k in HEAT_ROWS}
    for k, v in d.items():
        k, x = EMO_ALIAS.get(k, k), _num(v)
        if k in vals and not math.isnan(x):
            vals[k] += max(0.0, x)
    total = sum(vals.values())
    if not d or total <= 1e-6:
        return None
    return {k: v / total for k, v in vals.items()}


def dominant_emotion(r: dict, source: str) -> Optional[Tuple[str, float]]:
    """(emotion key, raw share) of the largest raw value, the way the per-segment table picks it; None = no data."""
    d = ((r.get("face") or {}).get("expressions") or {}) if source == "face" else ({} if _empty_text(r) else
                                                                                   (r.get("emotions_text") or {}))
    items = [(k, _num(v)) for k, v in d.items() if not math.isnan(_num(v))]
    if not items:
        return None
    k, v = max(items, key=lambda kv: kv[1])
    return EMO_ALIAS.get(k, k), v


def _emotions_chart(plt, rep: dict, per: List[dict], out_dir: Path) -> str:
    """One block per emotion with two rows, speech (what is said) above face (how it looks), so agreement is read one
    row down; the columns are the segments on the true time axis. A cell has the colour of its emotion (the bar at the
    row name), saturated by the share in five steps of equal lightness for every hue (palette.emo_heat_pdf); below 5%
    it stays empty, from 15% it carries the share in %, bold for the dominant emotion of the segment. «нейтрально»,
    80-99% in a calm video, stands at the bottom behind an extra gap — apart from the other emotions, but on the same
    scale as they are, so the darkest band of the chart is the largest share and not the other way round."""
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.transforms import blended_transform_factory
    dur = _duration(rep, per)
    # rows: y grows downwards, one row = 1 unit; the neutral block is set apart by an extra gap
    rows, blocks, y = [], [], 0.0
    for e in HEAT_ROWS:
        if e == "neutral":
            y += 0.55
        rows += [(y, e, "text"), (y + 1, e, "face")]
        blocks.append((y, e))
        y += 2.45
    y_max = y - 0.45
    fig, ax = _figure(plt, "emotions")
    fig.suptitle("Эмоции по ходу ролика: по речи и по лицу", x=0.01, ha="left")
    sub = ax.set_title("У каждой эмоции две строки: сверху — речь (что говорит), снизу — лицо (как выглядит); "
                       "числа — доля в отрезке, %", fontsize=8, fontweight="normal", color=MUTED, loc="left", pad=4)
    data = {src: [emotion_shares(r, src) for r in per] for src in HEAT_SOURCES}
    dom = {src: [dominant_emotion(r, src) for r in per] for src in HEAT_SOURCES}
    # time no segment covers (a segment missing from per_segment) is hatched on every row, never left looking like 0%
    holes, t = [], 0.0
    for s, e in sorted((float(r["start"]), float(r["end"])) for r in per):
        if s - t > 0.5:
            holes.append((t, s))
        t = max(t, e)
    if dur - t > 0.5:
        holes.append((t, dur))
    # white gaps between the cells are geometry, not edge lines (an edge ate a third of a 1-mm cell of an hour-long
    # video): rows 0.5 mm apart, columns 0.35 mm, at most 12% of the cell width
    s_per_mm = dur / (TEXT_W_MM - PLOT_LEFT_MM - PLOT_RIGHT_MM)
    gy = 0.5 / ((HEIGHT_MM["emotions"] - HEAT_DECOR_MM) / y_max)
    cells = []                          # (x0, x1, y, share, colour, bold) for the numbers
    gap_src = set()
    for yr, emo, src in rows:
        for i, (r, sh) in enumerate(zip(per, data[src])):
            x0, x1 = float(r["start"]), float(r["end"])
            gx = min(0.35 * s_per_mm, 0.12 * (x1 - x0))
            box = ((x0 + gx / 2, yr + gy / 2), x1 - x0 - gx, 1 - gy)
            if sh is None:               # hatched, in the same cell geometry as its neighbours
                ax.add_patch(Rectangle(*box, facecolor="white", edgecolor=NODATA_HATCH, hatch="///", lw=0, zorder=1))
                gap_src.add(src)
                continue
            c = emo_heat_pdf(emo, emo_heat_step(sh[emo]))
            ax.add_patch(Rectangle(*box, facecolor=c, lw=0, zorder=2))
            d = dom[src][i]
            cells.append((x0, x1, yr, sh[emo], c, bool(d and d[0] == emo)))
        for h0, h1 in holes:
            ax.add_patch(Rectangle((h0, yr + gy / 2), h1 - h0, 1 - gy, facecolor="white", edgecolor=NODATA_HATCH,
                                   hatch="///", lw=0, zorder=1))
    ax.set_ylim(y_max, 0)
    ax.set_yticks([yb + 1 for yb, _ in blocks], [EMO_RU[e] for _, e in blocks], color=INK, fontsize=8.5)
    ax.tick_params(axis="y", length=0, pad=7.5)          # the pad leaves room for the colour bar of the row
    _hide_spines(ax, "top", "right", "left")
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    for yr, _, src in rows:
        ax.text(1.004, yr + 0.5, HEAT_SOURCES[src], transform=tr, ha="left", va="center", fontsize=8, color=AXIS)
    _time_axis(ax, dur)
    # legend: every strength step as one swatch split between three hues of the chart; «нет данных» only when hatched
    handles = [Patch(facecolor=EMO_HEAT_PDF["empty"], edgecolor=OUTLINE, lw=0.4)]
    handles += [tuple(Patch(facecolor=emo_heat_pdf(e, step), edgecolor=OUTLINE, lw=0.4) for e in HEAT_LEGEND_HUES)
                for step in range(1, len(EMO_HEAT_L) + 1)]
    labels = list(HEAT_STEP_LABELS)
    if gap_src or holes:
        why = []
        if "text" in gap_src:
            gaps = [r for r, sh in zip(per, data["text"]) if sh is None]
            why.append("в отрезке нет речи" if all(_seg_words(r) == 0 for r in gaps) else "нет текста речи")
        if "face" in gap_src:
            why.append("лицо не найдено")
        if holes:
            why.append("время вне отрезков")
        handles.append(_nodata_patch("")); labels.append(f"нет данных ({'; '.join(why)})")
    _fig_legend(fig, handles, labels, handlelength=2.2, handletextpad=0.5, columnspacing=1.3, alignment="left",
                handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
                title="Доля эмоции в отрезке: чем насыщеннее клетка, тем больше — одинаково в любом цвете "
                      "(цвет клетки — сама эмоция)")
    _align_plot(fig)
    # after the final geometry: the subtitle starts where the title does, the colour bars and the numbers are placed
    pos = ax.get_position()
    ax_w_mm = pos.width * TEXT_W_MM
    sub.set_x((0.01 * TEXT_W_MM - pos.x0 * TEXT_W_MM) / ax_w_mm)
    bar_x0, bar_w = -2.3 / ax_w_mm, 1.2 / ax_w_mm
    for yb, e in blocks:
        ax.add_patch(Rectangle((bar_x0, yb + gy / 2), bar_w, 2 - gy, transform=tr, facecolor=emo_pdf(e), lw=0,
                               clip_on=False, zorder=3))
    for x0, x1, yr, sh, c, bold in cells:
        p = round(sh * 100)
        if p < HEAT_MIN_LABEL:
            continue
        s = str(p)
        if (x1 - x0) / dur * ax_w_mm < len(s) * (2.0 if bold else 1.85) + 0.9:    # 8 pt digits are ~1.8 mm wide
            continue
        ax.text((x0 + x1) / 2, yr + 0.5, s, ha="center", va="center_baseline", fontsize=8, color=emo_heat_text_pdf(c),
                fontweight="bold" if bold else "normal", zorder=4)
    p = out_dir / "chart_emotions.png"
    fig.savefig(p); plt.close(fig)
    return str(p)


# ---------------------------------------------------------------- voice dimensions
def _voice_chart(plt, rep: dict, per: List[dict], out_dir: Path) -> str:
    dur = _duration(rep, per)
    x = [(r["start"] + r["end"]) / 2 for r in per]
    fig, ax = _figure(plt, "voice")
    ax.set_title("Голос по ходу ролика")
    for d, name in VOICE_RU.items():
        ax.plot(x, [_num((r.get("voice") or {}).get(d)) for r in per], color=VOICE_PDF[d], marker=VOICE_MARKER_PDF[d],
                ms=SERIES_MS, lw=SERIES_LW, label=name, clip_on=False, zorder=3)
    ax.set_ylim(0, 1); ax.set_ylabel("уровень (0 — низкий,\n1 — высокий)", fontsize=8.5)
    ax.set_axisbelow(True); ax.grid(color=GRID, lw=0.5)
    h, lab = ax.get_legend_handles_labels()
    no_voice = [r for r in per if not r.get("voice")]          # the lines break there: say why
    for r in no_voice:
        _nodata_span(ax, r["start"], r["end"])
    if no_voice:
        h.append(_nodata_patch("")); lab.append("нет данных")
    _fig_legend(fig, h, lab, handlelength=1.8, columnspacing=1.4)
    _time_axis(ax, dur)
    _align_plot(fig)
    p = out_dir / "chart_voice.png"
    fig.savefig(p); plt.close(fig)
    return str(p)


# ---------------------------------------------------------------- speech tempo and pauses
def _speech_chart(plt, rep: dict, per: List[dict], out_dir: Path) -> str:
    import matplotlib.patheffects as pe
    from matplotlib.ticker import PercentFormatter
    dur = _duration(rep, per)
    x = [(r["start"] + r["end"]) / 2 for r in per]
    fig, ax = _figure(plt, "speech")
    ax.set_title("Речь по ходу ролика: темп и паузы")
    bars_c, pause_c = SPEECH_PDF["bars"], SPEECH_PDF["pauses"]
    with_wpm = [r for r in per if (r.get("speech") or {}).get("words_per_min_speech") is not None]
    no_wpm = [r for r in per if r.get("speech") and r["speech"].get("words_per_min_speech") is None]
    no_speech = [r for r in per if not r.get("speech")]         # no speech analytics at all for the segment
    values = [float(r["speech"]["words_per_min_speech"]) for r in with_wpm]
    ax.bar([(r["start"] + r["end"]) / 2 for r in with_wpm], values,
           width=[max(4.0, 0.8 * (r["end"] - r["start"])) for r in with_wpm], color=bars_c, alpha=1.0, zorder=2)
    for r in no_wpm + no_speech:
        _nodata_span(ax, r["start"], r["end"])
    # five equal steps, like the 0-100% pause scale on the right, so both grids line up; the step is a round
    # number (120 → 0/120/…/600, 300 → 0/300/…/1500) instead of 20·ceil(max/100), which gave 260 or 340
    top = max(values) if values else 0
    step = next((s for s in (20, 40, 60, 80, 100, 120, 160, 200, 240, 300, 400, 500, 600, 800, 1000) if 5 * s >= top),
                200 * math.ceil(top / 1000))
    ax.set_ylim(0, 5 * step); ax.set_yticks(range(0, 5 * step + 1, step))
    # left scale text: a darker slate of the bar colour (7.58:1 on white; the bar colour itself is only 4.76:1)
    ax.set_ylabel("слов в минуту", color=BARS_TEXT); ax.tick_params(axis="y", colors=BARS_TEXT)
    ax.set_axisbelow(True); ax.grid(color=GRID, lw=0.5)
    ax2 = ax.twinx()
    ax2.plot(x, [_num((r.get("speech") or {}).get("pause_share")) for r in per], color=pause_c, lw=SERIES_LW, marker="s",
             ms=SERIES_MS, mfc="white", mec=pause_c, clip_on=False, zorder=3,
             path_effects=[pe.withStroke(linewidth=3.0, foreground="white")])
    ax2.set_ylim(0, 1); ax2.set_yticks([0, .2, .4, .6, .8, 1]); ax2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    # the right scale has no axis title: the legend names it («доля пауз, % (правая шкала)») and its ticks are in the
    # line colour; a title there would widen the common right margin of every chart (PLOT_RIGHT_MM) by 5 mm
    ax2.tick_params(axis="y", colors=pause_c)
    ax2.spines["left"].set_visible(False)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=bars_c, edgecolor=bars_c),
               Line2D([], [], color=pause_c, lw=SERIES_LW, marker="s", ms=SERIES_MS, mfc="white", mec=pause_c)]
    labels = ["слов в минуту (левая шкала)", "доля пауз, % (правая шкала)"]
    if no_wpm or no_speech:          # one hatch style = one legend entry
        handles.append(_nodata_patch(""))
        labels.append("темп не посчитан (речи меньше 3 с)" if not no_speech else
                      "нет данных о речи" if not no_wpm else "нет данных о темпе")
    _fig_legend(fig, handles, labels, handlelength=1.8, columnspacing=1.4)
    _time_axis(ax, dur)
    _align_plot(fig)
    p = out_dir / "chart_speech.png"
    fig.savefig(p); plt.close(fig)
    return str(p)


# ---------------------------------------------------------------- modality contributions of the own model
MOD_LEGEND = {"face": "лицо (кадры)", "audio": "голос (CLAP)", "audio_whisper": "голос (Whisper)",
              "audio_xlsr": "голос (XLS-R)", "audio_w2v_emo": "голос (wav2vec2)", "text": "речь (текст)",
              "behavior": "описание поведения", "scene": "сцена (SSL-MEPR)"}
MOD_SHORT = {"face": "лицо", "text": "речь", "behavior": "поведение", "scene": "сцена"}
MOD_ROWS = {**RU_TITLES, "interview": "«Собеседование»"}


def _mod_colour(m: str) -> str:
    return MODALITY_PDF.get("audio" if m.startswith("audio") else m, "#52525b")


def _modalities_chart(plt, expl: dict | None, out_dir: Path) -> Optional[str]:
    """Print version of the web table «Вклад модальностей»: one 100% bar per trait, a segment per modality with its
    share inside; shares too small for a label are listed right of the bar."""
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter
    from matplotlib.transforms import blended_transform_factory
    ixg = ((expl or {}).get("modalities") or {}).get("input_x_gradient") or {}
    if not ixg:
        return None
    keys = [k for k in TRAIT_KEYS if k in ixg] + [k for k in ixg if k not in TRAIT_KEYS]
    mods = list(next(iter(ixg.values())).keys())
    fig, ax = _figure(plt, "modalities", height_mm=20 + 6 * len(keys))
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    for i, k in enumerate(keys):
        shares = [max(0.0, _num((ixg[k].get(m) or {}).get("share")) if not math.isnan(_num((ixg[k].get(m) or {}).get("share")))
                      else 0.0) for m in mods]
        total = sum(shares) or 1.0
        left, small = 0.0, []
        for m, s in zip(mods, shares):
            s /= total
            ax.barh(i, s, left=left, height=0.62, color=_mod_colour(m), edgecolor="white", lw=0.6, zorder=2)
            if s >= 0.08:
                ax.text(left + s / 2, i, _pct(s), ha="center", va="center", color="white", fontsize=8, fontweight="bold",
                        zorder=3)
            else:
                small.append(f"{MOD_SHORT.get(m, 'голос' if m.startswith('audio') else m)} {_pct(s)}")
            left += s
        if small:
            ax.text(1.012, i, " · ".join(small), transform=tr, ha="left", va="center", fontsize=8, color=MUTED)
    ax.set_yticks(range(len(keys)), [MOD_ROWS.get(k, k) for k in keys], fontsize=8.5, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(keys) - 0.5, -0.5)
    ax.set_xlim(0, 1); ax.set_xticks([0, .2, .4, .6, .8, 1]); ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    _hide_spines(ax, "top", "right", "left")
    ax.set_title("Вклад модальностей в оценку своей модели")
    handles = [Patch(facecolor=_mod_colour(m), edgecolor=OUTLINE, lw=0.4) for m in mods]
    _fig_legend(fig, handles, [MOD_LEGEND.get(m, m) for m in mods], handlelength=1.6, columnspacing=1.6)
    p = out_dir / "chart_modalities.png"
    fig.savefig(p); plt.close(fig)
    return str(p)


def save_modalities_chart(expl: dict | None, out_dir: str | Path) -> Optional[str]:
    """The modality chart alone (pdf_report draws it when save_pdf_charts was called without the explanation)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context(RC):
        return _modalities_chart(plt, expl, Path(out_dir))


def save_pdf_charts(rep: dict, out_dir: str | Path, expl: dict | None = None) -> Dict[str, str]:
    """PNG files for the PDF: profile (radar), traits timeline, average emotion profile, facial expression, emotions
    over time, voice, speech and — with the explanation — the modality contributions. Returns {name: path}; a chart
    without data is left out (pdf_report says so in one line), and so is one that fails: the PDF is still built."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "chart_voice_speech.png").unlink(missing_ok=True)     # older exports drew voice and speech side by side
    files: Dict[str, str] = {}
    per = (rep.get("analyses") or {}).get("per_segment") or []
    jobs = [("profile", lambda: _radar_chart(plt, rep, out_dir)),
            ("traits", lambda: _traits_chart(plt, rep, out_dir)),
            ("emotion_profile", lambda: _emotion_profile_chart(plt, rep, out_dir)),
            ("face_expr", lambda: _face_expr_chart(plt, rep, out_dir))]
    if len(per) >= 2:           # a video scored as one segment has nothing to show over time: the averages say it all
        jobs.append(("emotions", lambda: _emotions_chart(plt, rep, per, out_dir)))
        # a chart without any data is not drawn: pdf_report says so in one line instead of a blank full-width frame
        if any(r.get("voice") for r in per):
            jobs.append(("voice", lambda: _voice_chart(plt, rep, per, out_dir)))
        if any(r.get("speech") for r in per):
            jobs.append(("speech", lambda: _speech_chart(plt, rep, per, out_dir)))
    if expl:
        jobs.append(("modalities", lambda: _modalities_chart(plt, expl, out_dir)))
    with plt.rc_context(RC):            # wraps figure creation AND savefig, so sizes and dpi apply to every PNG
        for name, draw in jobs:
            try:
                p = draw()
            except Exception:  # noqa: BLE001  (one broken chart must not cost the whole report)
                log.exception("PDF chart %s failed", name)
                plt.close("all")
                continue
            if p:
                files[name] = p
    return files
