
# MyBasePlots/multipleCurvesAndHist_v4.py
# Rich, paper-ready multi-curve plot with optional side histograms (Y on the right, X at the bottom),
# consistent main panel geometry, selectable palettes (cb-safe, grayscale, strict B/W, mono, or custom),
# and robust tick/legend/cdf/statistics handling.
#
# Design goals:
# - Main panel physical size identical across figures (W_MAIN x H_MAIN in inches).
# - Figure grows only to accommodate optional decorations (histograms/legend/title).
# - All style (fonts/line widths/tick widths) inherited from the user's .mplstyle via utils.paper_style().
# - No rcParams mutation at module import; everything is decided inside the function.
# - No hard-coded "magic" font sizes or line widths: read from rcParams.
#
# API notes:
# - palette: "cb_safe" | "grayscale" | "bw" | "mono" | sequence of colors | cycler
# - yhist_ticks: "auto" | "right" | "none" (DEFAULT: "auto" → none if redundant with main)
# - xhist_ticks: "auto" | "bottom" | "top" | "none" (DEFAULT: "auto" → none if redundant with main)
# - yhist_xticks: "top" | "bottom" | "none" for the short axis (counts) of the Y-hist panels (DEFAULT: "top").
# - yhist_top_labels: str | list[str] | None to put a label above each Y-hist panel (split mode).
#
# Return:
#   (fig, ax_main, meta) where meta is a dict reporting the geometry/resolution choices.
#
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Iterable
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import (
    ScalarFormatter, LogLocator, LogFormatterSciNotation,
    NullFormatter, NullLocator, FixedLocator
)
from matplotlib.font_manager import FontProperties

from cycler import cycler, Cycler

# external style context (no-op if missing per the project utils contract)
try:
    from utils import paper_style
except Exception:
    from contextlib import contextmanager
    @contextmanager
    def paper_style(*args, **kwargs):
        yield

ArrayLike = Union[np.ndarray, Sequence[float], List[float]]


def multipleCurvesAndHist(
    # --- identifiers and data ---
    name: str,
    title: str,
    x_list: Union[ArrayLike, Sequence[ArrayLike], np.ndarray],
    x_label: str,
    y_list: Union[ArrayLike, Sequence[ArrayLike], np.ndarray],
    y_label: str,

    *,
    # --- curve selection/labels ---
    nameForCurve: str = 'traj',
    curvesIndeces: Optional[Sequence[int]] = None,  # keep legacy spelling used in the project
    namesForCurves: Optional[Sequence[str]] = None,

    # --- histogram toggles & layout ---
    show_yhist: bool = True,
    show_xhist: bool = False,
    yhist_overlay: bool = True,   # overlay extras onto a single Y-hist panel
    xhist_overlay: bool = True,   # overlay extras onto a single X-hist panel

    # tick policies for axes potentially redundant with the main plot
    yhist_ticks: str = "auto",    # "auto" | "right" | "none"
    xhist_ticks: str = "auto",    # "auto" | "bottom" | "top" | "none"
    yhist_xticks: str = "bottom",    # short axis (counts) of Y-hist: "top" | "bottom" | "none"

    # extras (arrays or dicts {'data','label','color','alpha'})
    extraHistsY: Optional[Sequence[Union[ArrayLike, Dict[str, Any]]]] = None,
    extraHistsX: Optional[Sequence[Union[ArrayLike, Dict[str, Any]]]] = None,

    # binning
    y_bins: Union[str, int, Sequence[float]] = 'auto',  # supports 'auto', 'fd', 'scott', 'sturges', 'sqrt', 'max', int, or explicit edges
    x_bins: Union[str, int, Sequence[float]] = 'auto',
    bins_within_visible_window: bool = True,
    use_max_bins: bool = False,

    # density/log
    density: bool = True,
    y_log_density: bool = False,
    x_log_density: bool = False,

    # CDF (independent per axis)
    y_cdf: bool = False,
    x_cdf: bool = False,
    # control CDF ticks visibility (for cleanliness)
    y_cdf_ticks: str = "top",    # "auto" | "none"
    x_cdf_ticks: str = "auto",

    # statistics overlays in hist: tuple subset of ('mu','sigma')
    showYStats: Sequence[str] = ('mu', 'sigma'),
    showXStats: Sequence[str] = ('mu', 'sigma'),

    # grid
    grid_main: bool = True,
    grid_hist: bool = True,

    # guide lines in main: (value,) | (value,label) | (value,label,color)
    xGuides: Optional[Sequence[Tuple[Any, ...]]] = None,
    yGuides: Optional[Sequence[Tuple[Any, ...]]] = None,

    # legend
    legend: bool = True,

    # palette: high-level selector; has precedence over internal defaults
    palette: Union[str, Sequence[str], cycler] = "cb_safe",
) -> Tuple[Figure, Axes, Dict[str, Any]]:
    """
    Draw multiple curves with optional side histograms in a geometry that keeps the main panel
    physically constant across figures (independent of optional decorations).
    """

    # =========================
    # = INTERNAL DEFINITIONS =
    # =========================
    with paper_style():
        rc = matplotlib.rcParams  # snapshot (read-only mapping)

        # ---- GEOMETRY (in inches) ----
        # Main panel size identical to the "reference" plot in this project
        W_MAIN = 4.2
        H_MAIN = 3.2

        # Margins left/right used in the reference plotWithDifferentColorbars
        LEFT_FRAC  = 0.16
        RIGHT_FRAC = 0.985

        # Title pad (inch): base + per-extra-line increment
        TOP_PAD_BASE   = 0.52
        TOP_PAD_PER_EX = 0.30

        # Fixed gaps (inch) between blocks
        GAP_W = 0.10   # between main and Y-hist columns
        GAP_H = 0.10   # between main and X-hist rows

        # Histogram module sizes (inch)
        W_HCOL = 0.90  # width of a single Y-hist column
        H_HROW = 0.90  # height of a single X-hist row

        # Legend padding column (inch) to avoid tick collisions
        LEG_PAD_W = 0.12  # base pad
        LEG_PAD_W_CDF = 0.16  # used if y_cdf + legend

        # Legend width estimation limits (inch)
        LEG_W_MIN = 1.0
        LEG_W_MAX = 2.2

        # ---- STYLE ACCESSORS ----
        # Never hard-code; always read from rcParams
        def _lw_base() -> float:
            try:
                return float(rc.get("lines.linewidth", 1.5))
            except Exception:
                return 1.5

        def _axes_lw() -> float:
            try:
                return float(rc.get("axes.linewidth", 1.0))
            except Exception:
                return 1.0

        def _tick_w_major() -> float:
            try:
                return float(rc.get("xtick.major.width", 0.8))
            except Exception:
                return 0.8

        def _tick_w_minor() -> float:
            try:
                return float(rc.get("xtick.minor.width", 0.6))
            except Exception:
                return 0.6

        def _to_points(val, default_pt=10.0) -> float:
            try:
                # numeric already (pt)
                return float(val)
            except Exception:
                try:
                    return float(FontProperties(size=val).get_size_in_points())
                except Exception:
                    return float(default_pt)

        def _fontsizes() -> Dict[str, float]:
            base = matplotlib.rcParams.get('font.size', 10.0)
            return {
                'label':  _to_points(rc.get('axes.labelsize',  base), base),
                'tick':   _to_points(rc.get('xtick.labelsize', base), base),
                'legend': _to_points(rc.get('legend.fontsize', base), base),
                'title':  _to_points(rc.get('axes.titlesize',  base), base),
            }

        # Auxiliary linewidth policies (relative to lines.linewidth)
        def _lw_mu()   -> float: return 1.05 * _lw_base()
        def _lw_sigma()-> float: return 0.95 * _lw_base()
        def _lw_guide()-> float: return 1.00 * _lw_base()
        def _lw_cdf()  -> float: return 1.15 * _lw_base()

        # ---- SIMPLE DATA HELPERS ----
        def _as_arrays(ls: Union[ArrayLike, Sequence[ArrayLike], np.ndarray]) -> List[np.ndarray]:
            arr = np.asarray(ls, dtype=object)
            if arr.ndim == 1 and len(arr) > 0 and not isinstance(arr[0], (list, tuple, np.ndarray)):
                # single array → wrap
                return [np.asarray(arr, dtype=float)]
            # assume sequence of arrays
            return [np.asarray(a, dtype=float) for a in ls]

        def _finite(x: np.ndarray) -> np.ndarray:
            m = np.isfinite(x)
            return x[m]

        def _stack_finite(seq: Sequence[np.ndarray]) -> np.ndarray:
            if not seq:
                return np.array([], dtype=float)
            return np.concatenate([_finite(np.asarray(a, dtype=float)) for a in seq]) if len(seq) else np.array([], dtype=float)

        def _mu_sigma(data: np.ndarray) -> Tuple[float, float]:
            data = _finite(np.asarray(data, dtype=float))
            if data.size == 0:
                return 0.0, 0.0
            return float(np.mean(data)), float(np.std(data, ddof=0))

        # ---- BINNING ----
        def _bins_from_arg(data: np.ndarray,
                           arg: Union[str, int, Sequence[float]],
                           within_limits: Optional[Tuple[float, float]] = None) -> np.ndarray:
            """Return bin edges given a 'numpy.histogram_bin_edges' compatible arg plus 'max' and explicit edges."""
            data = _finite(np.asarray(data, dtype=float))
            if isinstance(arg, (list, tuple, np.ndarray)) and len(np.asarray(arg).shape) == 1:
                edges = np.asarray(arg, dtype=float)
            elif isinstance(arg, int):
                edges = np.histogram_bin_edges(data, bins=arg)
            elif isinstance(arg, str):
                a = arg.lower()
                if a == 'max':
                    if data.size == 0:
                        edges = np.array([0.0, 1.0], dtype=float)
                    else:
                        vals = np.sort(np.unique(data))
                        if vals.size == 1:
                            edges = np.array([vals[0]-0.5, vals[0]+0.5])
                        else:
                            # edges halfway between unique values
                            mids = (vals[1:] + vals[:-1]) * 0.5
                            edges = np.concatenate([[vals[0] - (mids[0]-vals[0])], mids, [vals[-1] + (vals[-1]-mids[-1])]])
                elif a in ('auto', 'fd', 'scott', 'sturges', 'sqrt'):
                    edges = np.histogram_bin_edges(data, bins=(a if a != 'auto' else 'auto'))
                else:
                    raise ValueError(f"Unsupported bins spec: {arg!r}")
            else:
                # fallback to numpy behavior
                edges = np.histogram_bin_edges(data, bins=arg)

            if within_limits is not None:
                lo, hi = map(float, within_limits)
                edges = np.asarray(edges, dtype=float)
                edges = edges[(edges >= lo) & (edges <= hi)]
                if edges.size < 2:
                    edges = np.array([lo, hi], dtype=float)
                if edges[0] > lo:
                    edges = np.concatenate([[lo], edges])
                if edges[-1] < hi:
                    edges = np.concatenate([edges, [hi]])
            return edges

        # ---- CDF FROM HIST ----
        def _cdf_from_hist(data: np.ndarray, bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            data = _finite(np.asarray(data, dtype=float))
            if data.size == 0:
                return np.array([0, 1], dtype=float), np.array([0, 1], dtype=float)
            counts, edges = np.histogram(data, bins=bins, density=True)
            cdf = np.cumsum(counts) * np.diff(edges)
            cdf = np.concatenate([[0.0], cdf])
            cdf = np.clip(cdf, 0.0, 1.0)
            centers = np.asarray(edges, dtype=float)
            return centers, cdf

        # ---- PALETTES ----
        def _cb_safe_colors() -> List[str]:
            # Okabe–Ito (8 colors)
            return ['#377eb8', '#ff7f00', '#4daf4a', '#e41a1c', '#984ea3',
                    '#a65628', '#f781bf', '#999999']

        def _grayscale_colors(n: int) -> List[str]:
            if n <= 0:
                return []
            # perceptually spaced grays (avoid pure black/white extremes)
            vals = np.linspace(0.15, 0.85, n)
            return [str(float(v)) for v in vals]

        def _bw_linestyles(n: int) -> List[Tuple[str, str, str]]:
            """
            Strict B/W: (color, linestyle, marker) triplets.
            Use only black/gray colors with varied linestyles/markers for differentiation.
            """
            colors = ['0.0', '0.3', '0.6', '0.0', '0.45', '0.75', '0.15', '0.55']
            linestyles = ['-', '--', '-.', ':', (0, (1, 1)), (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 2))]
            markers = ['o', 's', 'D', '^', 'v', 'P', 'X', '']
            out: List[Tuple[str, str, str]] = []
            for i in range(n):
                c = colors[i % len(colors)]
                ls = linestyles[i % len(linestyles)]
                mk = markers[i % len(markers)]
                out.append((c, ls, mk))
            return out

        def _resolve_palette(n_curves: int) -> Tuple[List[str], Optional[List[Tuple[str, str, str]]]]:
            """
            Return either (colors, None) for normal color cycles, or ([], triplets) for strict B/W mode.
            """
            # user-specified
            if isinstance(palette, Cycler):
                cols = list(palette)
                # extract color field if present; else rely on Matplotlib to unpack
                return [d.get('color', d) for d in cols][:n_curves], None
            if isinstance(palette, (list, tuple, np.ndarray)):
                return [str(c) for c in palette][:n_curves], None
            p = str(palette).lower() if isinstance(palette, str) else "cb_safe"
            if p == "cb_safe":
                cols = _cb_safe_colors()
                if len(cols) < n_curves:
                    times = int(np.ceil(n_curves / len(cols)))
                    cols = (cols * times)[:n_curves]
                return cols[:n_curves], None
            if p == "grayscale":
                return _grayscale_colors(n_curves), None
            if p == "mono":
                base = rc.get("axes.prop_cycle", cycler(color=['C0']))
                try:
                    first = list(base)[0].get('color', 'C0')
                except Exception:
                    first = 'C0'
                return [first for _ in range(n_curves)], None
            if p == "bw":
                # special, no regular color cycle; we return triplets
                return [], _bw_linestyles(n_curves)
            # fallback
            cols = _cb_safe_colors()
            if len(cols) < n_curves:
                times = int(np.ceil(n_curves / len(cols)))
                cols = (cols * times)[:n_curves]
            return cols[:n_curves], None

        # ---- LEGEND WIDTH ESTIMATION ----
        def _estimate_legend_width_in(labels: Sequence[str]) -> float:
            """
            Estimate required legend width (inches) using a simple character-width heuristic.
            """
            if not legend or not labels:
                return 0.0
            # base approximations
            fs = _fontsizes()["legend"]
            # typical monospace-ish width: ~0.6 * fontsize px; convert to inches via figure DPI later
            # here we approximate in inches by assuming 72 px/in and 0.6 factor: 0.6*fs/72 ≈ 0.0083*fs
            char_w_in = 0.0085 * fs
            max_label_len = max(len(str(s)) for s in labels)
            handle_w_in = 0.40   # space for handle glyph
            pad_in       = 0.35  # text padding
            approx = handle_w_in + pad_in + char_w_in*max_label_len
            return float(np.clip(approx, LEG_W_MIN, LEG_W_MAX))

        # ---- TICK/FORMATTER HELPERS ----
        def _apply_scalar_formatter(ax: Axes, axis: str = 'x') -> None:
            fmt = ScalarFormatter(useMathText=True)
            fmt.set_powerlimits((-2, 2))
            fmt.set_useOffset(False)
            if axis == 'x':
                ax.xaxis.set_major_formatter(fmt)
            else:
                ax.yaxis.set_major_formatter(fmt)

        def _apply_log_formatter(ax: Axes, axis: str = 'x') -> None:
            locator = LogLocator(base=10.0, numticks=8)
            formatter = LogFormatterSciNotation(base=10.0)
            if axis == 'x':
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(formatter)
            else:
                ax.yaxis.set_major_locator(locator)
                ax.yaxis.set_major_formatter(formatter)

        def _hide_all_ticks(ax: Axes, axis: str) -> None:
            if axis == 'x':
                ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labeltop=False)
            else:
                ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)

        def _ticks_top(ax: Axes) -> None:
            # position x ticks on top; no bottom labels
            ax.xaxis.set_ticks_position('top')
            ax.tick_params(axis='x', which='both', top=True, labeltop=True, bottom=False, labelbottom=False)

        def _ticks_bottom(ax: Axes) -> None:
            ax.xaxis.set_ticks_position('bottom')
            ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True, top=False, labeltop=False)

        def _ticks_right(ax: Axes) -> None:
            ax.yaxis.set_ticks_position('right')
            ax.tick_params(axis='y', which='both', right=True, labelright=True, left=False, labelleft=False)
            # hide left spine, show right spine
            if 'left' in ax.spines:  ax.spines['left'].set_visible(False)
            if 'right' in ax.spines: ax.spines['right'].set_visible(True)

        def _ticks_none_y(ax: Axes) -> None:
            ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)

        # ---- STATS DRAW ----
        def _draw_stats_y(ax: Axes, mu: float, sigma: float, color_mu: str='0.1', color_sigma: str='0.35') -> None:
            ax.axhline(mu,           color=color_mu,   ls='--', lw=_lw_mu(),    zorder=3)
            if 'sigma' in showYStats and sigma > 0:
                ax.axhline(mu - sigma, color=color_sigma, ls='--', lw=_lw_sigma(), zorder=3)
                ax.axhline(mu + sigma, color=color_sigma, ls='--', lw=_lw_sigma(), zorder=3)

        def _draw_stats_x(ax: Axes, mu: float, sigma: float, color_mu: str='0.1', color_sigma: str='0.35') -> None:
            ax.axvline(mu,           color=color_mu,   ls='--', lw=_lw_mu(),    zorder=3)
            if 'sigma' in showXStats and sigma > 0:
                ax.axvline(mu - sigma, color=color_sigma, ls='--', lw=_lw_sigma(), zorder=3)
                ax.axvline(mu + sigma, color=color_sigma, ls='--', lw=_lw_sigma(), zorder=3)

        # ---- NORMALIZE EXTRAS ----
        def _normalize_extras(extras: Optional[Sequence[Union[ArrayLike, Dict[str, Any]]]]) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            if not extras:
                return out
            for e in extras:
                if isinstance(e, dict):
                    d = dict(e)
                    d.setdefault('data', [])
                    d.setdefault('label', '')
                    d.setdefault('color', None)
                    d.setdefault('alpha', 0.45)
                    out.append(d)
                else:
                    out.append({'data': np.asarray(e), 'label': '', 'color': None, 'alpha': 0.45})
            return out

        # =======================
        # = DATA PREPARATIONS  =
        # =======================
        Xs = _as_arrays(x_list)
        Ys = _as_arrays(y_list)

        if curvesIndeces is None and (len(Xs) != 1 or len(Ys) != 1) and namesForCurves is None:
            # no disambiguation provided on multi-array input
            pass  # keep going; we will infer

        # How many curves?
        n_curves = max(len(Xs), len(Ys))
        if len(Xs) == 1 and n_curves > 1:
            Xs = [Xs[0] for _ in range(n_curves)]
        if len(Ys) == 1 and n_curves > 1:
            Ys = [Ys[0] for _ in range(n_curves)]

        # Labels for curves
        if namesForCurves is not None:
            lab_raw = list(namesForCurves)
            curve_labels = lab_raw[:n_curves] if len(lab_raw) >= n_curves else \
                           lab_raw + [f"{nameForCurve} {i}" for i in range(len(lab_raw), n_curves)]
        else:
            if curvesIndeces is not None and len(curvesIndeces) >= n_curves:
                curve_labels = [f"{nameForCurve} {curvesIndeces[i]}" for i in range(n_curves)]
            else:
                curve_labels = [f"{nameForCurve} {i}" for i in range(n_curves)]

        # Palette resolution
        colors, bw_triplets = _resolve_palette(n_curves)

        # Compact all data for hist decisions
        y_all = _stack_finite(Ys)
        x_all = _stack_finite(Xs)

        # Normalize extras
        extraY = _normalize_extras(extraHistsY)
        extraX = _normalize_extras(extraHistsX)

        # Visible windows for bins='max'
        x_visible: Tuple[float, float] = (float(np.min(x_all)) if x_all.size else 0.0,
                                          float(np.max(x_all)) if x_all.size else 1.0)
        y_visible: Tuple[float, float] = (float(np.min(y_all)) if y_all.size else 0.0,
                                          float(np.max(y_all)) if y_all.size else 1.0)

        _x_bins = 'max' if use_max_bins else x_bins
        _y_bins = 'max' if use_max_bins else y_bins
        x_bins_final = _bins_from_arg(x_all, _x_bins, within_limits=(x_visible if bins_within_visible_window else None))
        y_bins_final = _bins_from_arg(y_all, _y_bins, within_limits=(y_visible if bins_within_visible_window else None))

        # =========================
        # = FIGURE / GRID LAYOUT =
        # =========================
        # legend label candidates (preliminary: curve labels + hist proxies)
        prelim_leg_labels = list(curve_labels) + (['Y data', 'μ', 'μ ± σ'] if show_yhist else [])
        LEG_W = _estimate_legend_width_in(prelim_leg_labels) if legend else 0.0

        # Y-hist column count: 1 if overlay; else 1 + n_extras
        n_cols_histY = (1 if yhist_overlay else (1 + len(extraY))) if show_yhist else 0
        # X-hist rows: 1 if overlay; else 1 + n_extras
        n_rows_histX = (1 if xhist_overlay else (1 + len(extraX))) if show_xhist else 0

        # Title pad (lines)
        n_title_lines = int(max(1, title.count('\n') + (1 if title else 0)))
        TOP_PAD = TOP_PAD_BASE + (max(0, n_title_lines - 1)) * TOP_PAD_PER_EX

        # Figure size in inches
        fig_w = W_MAIN \
              + ((GAP_W + n_cols_histY*W_HCOL) if n_cols_histY > 0 else 0.0) \
              + (( (LEG_PAD_W_CDF if (legend and (LEG_W > 0) and y_cdf) else LEG_PAD_W) + LEG_W) if (legend and LEG_W > 0) else 0.0)
        fig_h = TOP_PAD \
              + H_MAIN \
              + ((GAP_H + n_rows_histX*H_HROW) if n_rows_histX > 0 else 0.0)

        fig: Figure = plt.figure(name, figsize=(fig_w, fig_h), constrained_layout=False, clear=True)

        # GridSpec: columns = [main] + [gap] + [yhist cols] + [legend pad] + [legend]
        width_ratios  = [W_MAIN] \
                      + ([GAP_W] if n_cols_histY > 0 else []) \
                      + ([W_HCOL] * n_cols_histY) \
                      + ([ (LEG_PAD_W_CDF if (legend and y_cdf) else LEG_PAD_W) , LEG_W] if (legend and LEG_W > 0) else [])
        height_ratios = [H_MAIN] \
                      + ([GAP_H] if n_rows_histX > 0 else []) \
                      + ([H_HROW] * n_rows_histX)

        gs: GridSpec = fig.add_gridspec(
            nrows=len(height_ratios),
            ncols=len(width_ratios),
            width_ratios=width_ratios,
            height_ratios=height_ratios,
            wspace=0.0, hspace=0.0
        )

        # ---- MAIN AXES ----
        ax_main: Axes = fig.add_subplot(gs[0, 0])
        if grid_main:
            ax_main.grid(True, which='both', alpha=0.25)

        # enforce border thickness from style
        for side in ('left', 'right', 'bottom', 'top'):
            if side in ax_main.spines:
                ax_main.spines[side].set_linewidth(_axes_lw())

        # margins & title
        if title:
            # place suptitle with consistent vertical pad fraction derived from TOP_PAD
            fig.suptitle(title, y=1.0 - (TOP_PAD/fig_h)*0.25)
        fig.subplots_adjust(top=1.0 - (TOP_PAD/fig_h), bottom=0.18, left=LEFT_FRAC, right=RIGHT_FRAC)

        # formatters
        _apply_scalar_formatter(ax_main, 'x')
        _apply_scalar_formatter(ax_main, 'y')
        # enforce main tick label size/width from style
        ax_main.tick_params(axis='both', which='both', labelsize=_fontsizes()['tick'], width=_tick_w_major())
        ax_main.tick_params(axis='both', which='minor', width=_tick_w_minor())

        # labels
        ax_main.set_xlabel(x_label)
        ax_main.set_ylabel(y_label)

        
        # ---- DRAW CURVES ----
        lines_main: List[Line2D] = []

        # determine which indices to plot
        max_len = max(len(Xs), len(Ys))
        if curvesIndeces is not None and len(curvesIndeces) > 0:
            idx_list = [i for i in curvesIndeces if 0 <= i < max_len]
        else:
            if len(Ys) > 1 and len(Xs) == 1:
                idx_list = list(range(len(Ys)))              # common X, multiple Y
            elif len(Xs) > 1 and len(Ys) == 1:
                idx_list = list(range(len(Xs)))              # multiple X, common Y
            else:
                idx_list = list(range(min(len(Xs), len(Ys))))  # pairwise zip

        # restrict labels/palette to plotted indices
        plotted_labels = []
        if namesForCurves is not None:
            # take provided labels at requested indices
            for i in idx_list:
                if 0 <= i < len(namesForCurves):
                    plotted_labels.append(str(namesForCurves[i]))
                else:
                    plotted_labels.append(f"{nameForCurve} {i}")
        else:
            # default labels using provided indices or sequential
            for i in idx_list:
                plotted_labels.append(f"{nameForCurve} {i}")

        n_plot = len(idx_list)
        cols, bw_trip = _resolve_palette(n_plot)

        for k, i in enumerate(idx_list):
            xarr = Xs[i] if i < len(Xs) else Xs[0]
            yarr = Ys[i] if i < len(Ys) else Ys[0]
            if bw_trip is None:
                c = cols[k % len(cols)]
                ln, = ax_main.plot(xarr, yarr, color=c, zorder=2)
            else:
                c, ls, mk = bw_trip[k]
                ln, = ax_main.plot(xarr, yarr, color=c, ls=ls, marker=mk, zorder=2)
            lines_main.append(ln)
# ---- MAIN GUIDE LINES ----
        def _unpack_guide(t: Tuple[Any, ...]) -> Tuple[float, Optional[str], str]:
            if len(t) == 1:
                return float(t[0]), None, '0.25'
            if len(t) == 2:
                return float(t[0]), str(t[1]), '0.25'
            return float(t[0]), str(t[1]), str(t[2])

        if xGuides:
            for g in xGuides:
                xv, lab, col = _unpack_guide(g)
                ax_main.axvline(xv, ls='--', lw=_lw_guide(), color=col, zorder=1)
                if lab:
                    ax_main.text(xv, 1.0, lab, transform=ax_main.get_xaxis_transform(),
                                 ha='center', va='bottom')
        if yGuides:
            for g in yGuides:
                yv, lab, col = _unpack_guide(g)
                ax_main.axhline(yv, ls='--', lw=_lw_guide(), color=col, zorder=1)
                if lab:
                    ax_main.text(1.0, yv, lab, transform=ax_main.get_yaxis_transform(),
                                 ha='left', va='center')

        # ===========================
        # = Y-HISTOGRAMS (RIGHT)   =
        # ===========================
        yhist_axes: List[Axes] = []
        yhist_twin: List[Axes] = []
        if show_yhist and n_cols_histY > 0:
            # column index for first y-hist
            col0 = 2 if (n_cols_histY > 0) else 1  # [0=main, 1=gap, 2=first-hist]
            if yhist_overlay:
                axy = fig.add_subplot(gs[0, col0])
                yhist_axes.append(axy)
            else:
                for j in range(n_cols_histY):
                    axy = fig.add_subplot(gs[0, col0 + j])
                    yhist_axes.append(axy)

            # grid & spines
            for axy in yhist_axes:
                if grid_hist:
                    axy.grid(True, which='major', alpha=0.25)
                # force spine width from style
                for side in ('left', 'right', 'bottom', 'top'):
                    if side in axy.spines:
                        axy.spines[side].set_linewidth(_axes_lw())

            # unify tick label size on hist axes (slightly smaller than main)
            for axy in yhist_axes:
                axy.tick_params(axis='both', which='both', labelsize=max(6.0, 0.9*_fontsizes()['tick']))

            # build data series for hist panels
            # first column: "data" from all curves
            def _hist_y(ax: Axes, dat: np.ndarray, color: str, alpha: float=0.45) -> None:
                ax.hist(dat, bins=y_bins_final, orientation='horizontal',
                        density=density, histtype='stepfilled',
                        color=color, alpha=alpha, zorder=1)

            # base color for data panel
            if bw_triplets is None:
                data_color = (colors[0] if palette not in ("mono", "grayscale") else '0.4')
            else:
                data_color = '0.6'  # neutral gray in strict B/W

            # "data" panel
            _hist_y(yhist_axes[0], y_all, color=('0.5' if str(palette).lower() in ('grayscale','bw','mono') else data_color), alpha=(0.55 if yhist_overlay else 0.45))

            # extras
            if not yhist_overlay and len(yhist_axes) > 1:
                for k, h in enumerate(extraY[:len(yhist_axes)-1], start=1):
                    dat = _finite(np.asarray(h.get('data', [])))
                    if dat.size == 0:
                        continue
                    col = h.get('color', None)
                    if col is None:
                        # adopt palette consistently; for bw we draw outlines
                        if bw_triplets is None:
                            # cycle through remaining colors
                            idx = (k-1) % max(1, (len(colors)-1))
                            col = colors[1 + idx] if len(colors) > 1 else '0.3'
                        else:
                            col = '0.25'
                    if str(palette).lower() == 'bw':
                        # outline only with linestyle variations
                        a = fig.add_subplot(axy.get_subplotspec())  # ensure consistent Z? not needed.
                        yhist_axes[k].hist(dat, bins=y_bins_final, orientation='horizontal',
                                           density=density, histtype='step',
                                           color='0.0', alpha=1.0, lw=_lw_guide())
                    else:
                        _hist_y(yhist_axes[k], dat, color=col, alpha=float(h.get('alpha', 0.45)))

            # set y-limits consistent with main (visual alignment)
            y_min, y_max = ax_main.get_ylim()
            for axy in yhist_axes:
                axy.set_ylim(y_min, y_max)

            # density/log scaling on the short axis (counts)
            if y_log_density:
                for axy in yhist_axes:
                    axy.set_xscale('log', base=10.0)
                    _apply_log_formatter(axy, 'x')
            else:
                for axy in yhist_axes:
                    _apply_scalar_formatter(axy, 'x')

            # replicate guide lines on Y-hist
            if yGuides:
                for yv, _, col in map(_unpack_guide, yGuides):
                    for axy in yhist_axes:
                        axy.axhline(yv, ls='--', lw=_lw_guide(), color=col, zorder=3)

            # stats on data panel
            if 'mu' in showYStats or 'sigma' in showYStats:
                mu, sig = _mu_sigma(y_all)
                _draw_stats_y(yhist_axes[0], mu, sig)

            # ticks policy on long axis (y)
            _tick_policy = (yhist_ticks or 'auto').lower()
            if _tick_policy not in ('auto', 'right', 'none'):
                _tick_policy = 'auto'
            if _tick_policy == 'auto':
                # redundant with main → none
                for axy in yhist_axes:
                    _ticks_none_y(axy)
            elif _tick_policy == 'right':
                for axy in yhist_axes:
                    _ticks_right(axy)
            else:  # none
                for axy in yhist_axes:
                    _ticks_none_y(axy)

            # ticks on short axis (x of hist) to avoid collision with main's x ticks
            short_opt = (yhist_xticks or 'bottom').lower()
            for axy in yhist_axes:
                if short_opt == 'bottom':
                    _ticks_bottom(axy)
                    axy.tick_params(axis='x', which='both', top=False)
                    axy.tick_params(axis='x', which='major', pad=4.0)
                elif short_opt == 'top':
                    _ticks_top(axy)
                    # add a little extra pad to separate from legend area if present
                    axy.tick_params(axis='x', which='both', bottom=False)
                    axy.tick_params(axis='x', which='major', pad=4.0)
                elif short_opt == 'bottom':
                    _ticks_bottom(axy)
                else:
                    axy.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labeltop=False)

            
            
            # CDF (twin on short axis)
            if y_cdf:
                for axy in yhist_axes:
                    twin = axy.twiny()
                    # place CDF ticks on TOP by default
                    tick_mode = (y_cdf_ticks or 'top').lower()
                    if tick_mode not in ('top','bottom','none'):
                        tick_mode = 'top'
                    # clear both to start
                    twin.tick_params(axis='x', which='both', top=False, bottom=False, labeltop=False, labelbottom=False)
                    if tick_mode == 'top':
                        twin.xaxis.set_ticks_position('top')
                        twin.tick_params(axis='x', which='both', top=True, labeltop=True)
                    elif tick_mode == 'bottom':
                        twin.xaxis.set_ticks_position('bottom')
                        twin.tick_params(axis='x', which='both', bottom=True, labelbottom=True)
                    # spine position (aligned with hist axis start)
                    twin.spines['top' if tick_mode=='top' else 'bottom'].set_position(('axes', 1.0 if tick_mode=='top' else 0.0))
                    twin.spines['top' if tick_mode=='top' else 'bottom'].set_linewidth(_axes_lw())
                    # compute data
                    if axy is yhist_axes[0]:
                        dat = y_all
                    else:
                        idx = yhist_axes.index(axy) - 1
                        dat = _finite(np.asarray(extraY[idx].get('data', []))) if 0 <= idx < len(extraY) else np.array([])
                    centers, cdf = _cdf_from_hist(dat, y_bins_final)
                    # plot with x=cdf (0..1), y=centers
                    twin.set_xlim(0.0, 1.0)
                    twin.plot(cdf, centers, color='0.25', lw=_lw_cdf(), zorder=4)
                    # ticks for CDF axis
                    if tick_mode != 'none':
                        from matplotlib.ticker import FixedLocator
                        twin.xaxis.set_major_locator(FixedLocator([0.0, 0.5, 1.0]))
                        twin.tick_params(axis='x', which='major', pad=3.0)
# ===========================
        # = X-HISTOGRAMS (BOTTOM)  =
        # ===========================
        xhist_axes: List[Axes] = []
        if show_xhist and n_rows_histX > 0:
            row0 = 2 if (n_rows_histX > 0) else 1  # [0=main, 1=gap, 2=first-xhist]
            if xhist_overlay:
                axx = fig.add_subplot(gs[row0, 0])
                xhist_axes.append(axx)
            else:
                for j in range(n_rows_histX):
                    axx = fig.add_subplot(gs[row0 + j, 0])
                    xhist_axes.append(axx)

            for axx in xhist_axes:
                axx.tick_params(axis='both', which='both', labelsize=max(6.0, 0.9*_fontsizes()['tick']))
                if grid_hist:
                    axx.grid(True, which='major', alpha=0.25)
                for side in ('left', 'right', 'bottom', 'top'):
                    if side in axx.spines:
                        axx.spines[side].set_linewidth(_axes_lw())

            def _hist_x(ax: Axes, dat: np.ndarray, color: str, alpha: float=0.45) -> None:
                ax.hist(dat, bins=x_bins_final, orientation='vertical',
                        density=density, histtype='stepfilled',
                        color=color, alpha=alpha, zorder=1)

            if bw_triplets is None:
                data_color_x = (colors[0] if palette not in ("mono", "grayscale") else '0.4')
            else:
                data_color_x = '0.6'

            _hist_x(xhist_axes[0], x_all, color=('0.5' if str(palette).lower() in ('grayscale','bw','mono') else data_color_x), alpha=(0.55 if xhist_overlay else 0.45))

            if not xhist_overlay and len(xhist_axes) > 1:
                for k, h in enumerate(extraX[:len(xhist_axes)-1], start=1):
                    dat = _finite(np.asarray(h.get('data', [])))
                    if dat.size == 0:
                        continue
                    col = h.get('color', None)
                    if col is None:
                        if bw_triplets is None:
                            idx = (k-1) % max(1, (len(colors)-1))
                            col = colors[1 + idx] if len(colors) > 1 else '0.3'
                        else:
                            col = '0.25'
                    if str(palette).lower() == 'bw':
                        xhist_axes[k].hist(dat, bins=x_bins_final, orientation='vertical',
                                           density=density, histtype='step',
                                           color='0.0', alpha=1.0, lw=_lw_guide())
                    else:
                        _hist_x(xhist_axes[k], dat, color=col, alpha=float(h.get('alpha', 0.45)))

            # align x-limits with main
            x_min, x_max = ax_main.get_xlim()
            for axx in xhist_axes:
                axx.tick_params(axis='both', which='both', labelsize=max(6.0, 0.9*_fontsizes()['tick']))
                axx.set_xlim(x_min, x_max)

            if x_log_density:
                for axx in xhist_axes:
                    axx.tick_params(axis='both', which='both', labelsize=max(6.0, 0.9*_fontsizes()['tick']))
                    axx.set_yscale('log', base=10.0)
                    _apply_log_formatter(axx, 'y')
            else:
                for axx in xhist_axes:
                    axx.tick_params(axis='both', which='both', labelsize=max(6.0, 0.9*_fontsizes()['tick']))
                    _apply_scalar_formatter(axx, 'y')

            if xGuides:
                for xv, _, col in map(_unpack_guide, xGuides):
                    for axx in xhist_axes:
                        axx.tick_params(axis='both', which='both', labelsize=max(6.0, 0.9*_fontsizes()['tick']))
                        axx.axvline(xv, ls='--', lw=_lw_guide(), color=col, zorder=3)

            if 'mu' in showXStats or 'sigma' in showXStats:
                mu, sig = _mu_sigma(x_all)
                _draw_stats_x(xhist_axes[0], mu, sig)

            # ticks on long axis (x) possibly redundant with main
            _x_tick_pol = (xhist_ticks or 'auto').lower()
            if _x_tick_pol not in ('auto', 'bottom', 'top', 'none'):
                _x_tick_pol = 'auto'
            if _x_tick_pol == 'auto':
                for axx in xhist_axes:
                    axx.tick_params(axis='both', which='both', labelsize=max(6.0, 0.9*_fontsizes()['tick']))
                    _hide_all_ticks(axx, 'x')  # redundant with main
            elif _x_tick_pol == 'bottom':
                for axx in xhist_axes:
                    axx.tick_params(axis='both', which='both', labelsize=max(6.0, 0.9*_fontsizes()['tick']))
                    _ticks_bottom(axx)
            elif _x_tick_pol == 'top':
                for axx in xhist_axes:
                    axx.tick_params(axis='both', which='both', labelsize=max(6.0, 0.9*_fontsizes()['tick']))
                    _ticks_top(axx)
            else:
                for axx in xhist_axes:
                    axx.tick_params(axis='both', which='both', labelsize=max(6.0, 0.9*_fontsizes()['tick']))
                    _hide_all_ticks(axx, 'x')

            # CDF on X if requested
            if x_cdf:
                for axx in xhist_axes:
                    axx.tick_params(axis='both', which='both', labelsize=max(6.0, 0.9*_fontsizes()['tick']))
                    twin = axx.twinx()
                    twin.yaxis.set_ticks_position('right')
                    if 'left' in twin.spines:
                        twin.spines['left'].set_visible(False)
                    if 'right' in twin.spines:
                        twin.spines['right'].set_linewidth(_axes_lw())
                    twin.set_ylim(0.0, 1.0)
                    if axx is xhist_axes[0]:
                        dat = x_all
                    else:
                        idx = xhist_axes.index(axx) - 1
                        dat = _finite(np.asarray(extraX[idx].get('data', []))) if 0 <= idx < len(extraX) else np.array([])
                    centers, cdf = _cdf_from_hist(dat, x_bins_final)
                    twin.plot(centers, cdf, color='0.25', lw=_lw_cdf(), zorder=4)
                    if (x_cdf_ticks or 'auto').lower() == 'none':
                        twin.tick_params(axis='y', which='both', right=False, labelright=False)
                    else:
                        twin.set_yticks([0.0, 0.5, 1.0])
                        twin.set_ylim(0.0, 1.0)

        # ===========================
        # = LEGEND (RIGHT COLUMN)  =
        # ===========================
        if legend and LEG_W > 0:
            ax_leg: Axes = fig.add_subplot(gs[:, -1])
            ax_leg.axis('off')
            handles: List[Any] = []
            labels: List[str] = []

            # Curves
            for ln, lab in zip(lines_main, plotted_labels):
                handles.append(ln); labels.append(lab)

            # Hist proxies
            if show_yhist:
                # 'Y data' rectangle proxy (gray fill)
                rect = Patch(facecolor='0.7', edgecolor='0.3', alpha=0.55, label='Y data')
                handles.append(rect); labels.append('Y data')
                # mu and mu±sigma
                lmu = Line2D([0], [0], color='0.1', lw=_lw_mu(), ls='--')
                lsig = Line2D([0], [0], color='0.35', lw=_lw_sigma(), ls='--')
                handles.extend([lmu, lsig]); labels.extend([r'$\mu$', r'$\mu \pm \sigma$'])

            # Fit legend width: we already reserved the column; just draw
            ax_leg.legend(handles, labels, loc='center left', frameon=False, borderaxespad=0.0)

        # ===========================
        # = META INFORMATION       =
        # ===========================
        meta: Dict[str, Any] = {
            "figsize_in": (fig_w, fig_h),
            "main_panel_in": (W_MAIN, H_MAIN),
            "n_curves": n_curves,
            "n_cols_histY": n_cols_histY,
            "n_rows_histX": n_rows_histX,
            "legend_width_in": LEG_W,
            "yhist_ticks_policy": (yhist_ticks or 'auto'),
            "xhist_ticks_policy": (xhist_ticks or 'auto'),
            "yhist_xticks_short_axis": (yhist_xticks or 'top'),
            "palette": palette,
        }

        return fig, ax_main, meta




def report_layout(fig: Figure, meta: Optional[Dict[str, Any]] = None) -> None:
    """
    Stampa un report numerico sulle dimensioni effettive (in pollici) del main e della figura.
    Robusta a 'meta' non-dict o oggetti strani.
    """
    try:
        W, H = fig.get_size_inches()
    except Exception:
        W = H = float('nan')

    # Predefiniti
    ax = fig.axes[0] if fig.axes else None
    w_main_in = h_main_in = None
    fig_w_in = W
    fig_h_in = H
    legend_pad_in = None
    pads = (None, None, None, None)

    # Se 'meta' è un dict, estrai info; altrimenti ignora senza crash
    if isinstance(meta, dict):
        ax = meta.get('axes_main', ax)
        w_main_in = meta.get('w_main_in', None)
        h_main_in = meta.get('h_main_in', None)
        fig_w_in  = meta.get('fig_w_in', fig_w_in)
        fig_h_in  = meta.get('fig_h_in', fig_h_in)
        legend_pad_in = meta.get('legend_pad_in', None)
        pads = (meta.get('pad_left_in', None), meta.get('pad_right_in', None),
                meta.get('pad_top_in', None), meta.get('pad_bottom_in', None))

    # Bounding box effettivo del main
    if ax is not None:
        bb = ax.get_position()  # in figure coordinates
        main_w_eff = (bb.x1 - bb.x0) * W
        main_h_eff = (bb.y1 - bb.y0) * H
    else:
        main_w_eff = main_h_eff = float('nan')

    print("----- Layout report -----")
    print(f"Figure size (in): {W:.3f} × {H:.3f}  | meta fig (in): {fig_w_in:.3f} × {fig_h_in:.3f}")
    if legend_pad_in is not None:
        print(f"Legend pad (in): {legend_pad_in:.3f}")
    if all(v is not None for v in pads):
        pl, pr, pt, pb = pads
        print(f"PADS (in): left={pl:.3f} right={pr:.3f} top={pt:.3f} bottom={pb:.3f}")
    if (w_main_in is not None) and (h_main_in is not None):
        print(f"Main design (in): {w_main_in:.3f} × {h_main_in:.3f}")
    print(f"Main effective bbox (in): {main_w_eff:.3f} × {main_h_eff:.3f}")
    if (w_main_in is not None) and (h_main_in is not None) and np.isfinite(main_w_eff) and np.isfinite(main_h_eff):
        dw = main_w_eff - w_main_in
        dh = main_h_eff - h_main_in
        print(f"Δ main (eff - design): dW={dw:+.3e} in, dH={dh:+.3e} in")
    print("-------------------------")
