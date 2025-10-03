
# MyBasePlots/multipleCurvesAndHist_v21.py
# Multi-curve plot with optional side histograms (Y right, X top/bottom) keeping the MAIN panel physically fixed.
# Style is sourced from paper_style() and propagated from the MAIN axis to all auxiliary axes (no rcParams lookups for sizes).
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Mapping, Callable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter, LogLocator, LogFormatterSciNotation, FixedLocator, MaxNLocator
from cycler import cycler, Cycler



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
    curvesIndeces: Optional[Sequence[int]] = None,
    curvesIndices: Optional[Sequence[int]] = None,  # legacy spelling kept
    namesForCurves: Optional[Sequence[str]] = None,

    # --- histogram toggles & layout ---
    show_yhist: bool = True,
    show_xhist: bool = False,
    yhist_overlay: bool = True,   # overlay extras onto a single Y-hist panel
    xhist_overlay: bool = True,   # overlay extras onto a single X-hist panel
    xhist_position: str = 'top',  # 'top' | 'bottom'

    # tick policies for axes potentially redundant with the main plot
    yhist_ticks: str = "auto",    # "auto" | "right" | "none"  (long axis of Y-hist)
    xhist_ticks: str = "auto",    # "auto" | "bottom" | "top" | "none" (long axis of X-hist)
    yhist_xticks: str = "top",    # short axis (counts) of Y-hist: "bottom" | "top" | "none"

    # extras (arrays or dicts {'data','label','color','alpha'})
    extraHistsY: Optional[Sequence[Union[ArrayLike, Dict[str, Any]]]] = None,
    extraHistsX: Optional[Sequence[Union[ArrayLike, Dict[str, Any]]]] = None,

    # binning
    y_bins: Union[str, int, Sequence[float]] = 'auto',  # supports 'auto', 'fd', 'scott', 'sturges', 'sqrt', 'max', int, explicit edges
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
    y_cdf_ticks: str = "top",     # "top" | "bottom" | "none"
    x_cdf_ticks: str = "right",   # "right" | "left" | "none"

    # statistics overlays in hist: subset of ('mu','sigma')
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
    reserve_legend_space: bool = True,
    manage_style: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    guide_labels: str = 'legend',   # 'legend' | 'axes' | 'none'

    # palette
    palette: Union[str, Sequence[str], Cycler] = "cb_safe",

    # --- NEW: fine control without touching global rc ---
    hist_tick_scale: float = 0.88,   # histogram & CDF-twin tick label size = scale * main tick size

    # --- CDF reference lines ---
    cdf_ref_probs_y: Sequence[float] = (),
    cdf_ref_probs_x: Sequence[float] = (),
    cdf_ref_apply_y: str = 'first',  # 'first' | 'all'
    cdf_ref_apply_x: str = 'first',  # 'first' | 'all'
    cdf_ref_style: Optional[Mapping[str, Any]] = None,
    cdf_ref_labels: bool = False,
    cdf_ref_fmt: Optional[Callable[[float], str]] = None,
    # NEW color controls (optional)
    cdf_ref_colors_y: Optional[Sequence[str]] = None,
    cdf_ref_colors_x: Optional[Sequence[str]] = None,
) -> Tuple[Figure, Axes, Dict[str, Any]]:
    """Draw multiple curves with optional side histograms keeping the main panel physically constant.
    The main axis is the single source of style; all auxiliary axes copy linewidths and font sizes from it.
    """
    if True:
        # --------------------
        # CONSTANT GEOMETRY
        # --------------------
        W_MAIN = 4.2
        H_MAIN = 3.2
        # Outer pads (inches) for CDF ticklabels (top/right)
        TOP_OUTER_PAD = 0.16
        RIGHT_OUTER_PAD = 0.16
        LEFT_FRAC  = 0.16
        RIGHT_FRAC = 0.985
        TOP_PAD_BASE   = 0.52
        TOP_PAD_PER_EX = 0.30
        GAP_W = 0.10
        GAP_H = 0.10
        W_HCOL = 0.90
        H_HROW = 0.90
        LEG_PAD_W = 0.12
        LEG_PAD_W_CDF = 0.16
        LEG_W_MIN = 1.0
        LEG_W_MAX = 2.2

        # --------------------
        # HELPERS (no rc lookups for sizes; we copy from main)
        # --------------------
        def _finite(x: np.ndarray) -> np.ndarray:
            m = np.isfinite(x)
            return x[m]

        def _as_arrays(ls: Union[ArrayLike, Sequence[ArrayLike], np.ndarray]) -> List[np.ndarray]:
            if isinstance(ls, np.ndarray) and ls.ndim > 1:
                return [np.asarray(a, dtype=float) for a in ls]
            arr = np.asarray(ls, dtype=object)
            if arr.ndim == 1 and len(arr) > 0 and not isinstance(arr[0], (list, tuple, np.ndarray)):
                return [np.asarray(arr, dtype=float)]
            return [np.asarray(a, dtype=float) for a in ls]

        def _stack_finite(seq: Sequence[np.ndarray]) -> np.ndarray:
            if not seq:
                return np.array([], dtype=float)
            return np.concatenate([_finite(np.asarray(a, dtype=float)) for a in seq]) if len(seq) else np.array([], dtype=float)

        def _bins_from_arg(data: np.ndarray, arg: Union[str, int, Sequence[float]], within_limits: Optional[Tuple[float, float]] = None) -> np.ndarray:
            data = _finite(np.asarray(data, dtype=float))
            if isinstance(arg, (list, tuple, np.ndarray)) and np.asarray(arg).ndim == 1:
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
                            mids = (vals[1:] + vals[:-1]) * 0.5
                            edges = np.concatenate([[vals[0] - (mids[0]-vals[0])], mids, [vals[-1] + (vals[-1]-mids[-1])]])
                elif a in ('auto','fd','scott','sturges','sqrt'):
                    edges = np.histogram_bin_edges(data, bins=a)
                else:
                    raise ValueError(f"Unsupported bins spec: {arg!r}")
            else:
                edges = np.histogram_bin_edges(data, bins=arg)
            if within_limits is not None:
                lo, hi = map(float, within_limits)
                edges = np.asarray(edges, dtype=float)
                edges = edges[(edges >= lo) & (edges <= hi)]
                if edges.size < 2:
                    edges = np.array([lo, hi], dtype=float)
                if edges[0] > lo: edges = np.concatenate([[lo], edges])
                if edges[-1] < hi: edges = np.concatenate([edges, [hi]])
            # ensure unique & increasing
            edges = np.unique(np.asarray(edges, dtype=float))
            if edges.size < 2:
                lo = float(np.min(data)) if data.size else 0.0
                hi = float(np.max(data)) if data.size else 1.0
                edges = np.array([lo, hi], dtype=float)
            return edges

        def _cdf_from_hist(data: np.ndarray, bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            data = _finite(np.asarray(data, dtype=float))
            if data.size == 0:
                return np.array([0., 1.], dtype=float), np.array([0., 1.], dtype=float)
            counts, edges = np.histogram(data, bins=bins, density=True)
            incr = counts * np.diff(edges)
            cdf = np.concatenate([[0.0], np.cumsum(incr)])
            cdf = np.maximum.accumulate(cdf)
            if not np.isclose(cdf[-1], 1.0) and cdf[-1] > 0:
                cdf = cdf / cdf[-1]
            cdf = np.clip(cdf, 0.0, 1.0)
            return edges.astype(float), cdf.astype(float)

        def _apply_scalar_formatter(ax: Axes, axis: str = 'x') -> None:
            fmt = ScalarFormatter(useMathText=True)
            fmt.set_powerlimits((-2, 2))
            fmt.set_useOffset(False)
            if axis == 'x': ax.xaxis.set_major_formatter(fmt)
            else:           ax.yaxis.set_major_formatter(fmt)

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
            ax.xaxis.set_ticks_position('top')
            ax.tick_params(axis='x', which='both', top=True, labeltop=True, bottom=False, labelbottom=False)

        def _ticks_bottom(ax: Axes) -> None:
            ax.xaxis.set_ticks_position('bottom')
            ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True, top=False, labeltop=False)

        def _ticks_right(ax: Axes) -> None:
            ax.yaxis.set_ticks_position('right')
            ax.tick_params(axis='y', which='both', right=True, labelright=True, left=False, labelleft=False)
            if 'left' in ax.spines:  ax.spines['left'].set_visible(False)
            if 'right' in ax.spines: ax.spines['right'].set_linewidth(_LW_AXES)

        def _unpack_guide(t: Tuple[Any, ...]) -> Tuple[float, Optional[str], str]:
            if len(t) == 1: return float(t[0]), None, '0.25'
            if len(t) == 2: return float(t[0]), str(t[1]), '0.25'
            return float(t[0]), str(t[1]), str(t[2])

        # --------------------
        # DATA
        # --------------------
        Xs = _as_arrays(x_list)
        Ys = _as_arrays(y_list)
        n_raw = max(len(Xs), len(Ys))
        if len(Xs) == 1 and n_raw > 1: Xs = [Xs[0] for _ in range(n_raw)]
        if len(Ys) == 1 and n_raw > 1: Ys = [Ys[0] for _ in range(n_raw)]

        if (curvesIndices is not None) and (curvesIndeces is None):
            curvesIndeces = curvesIndices
        if curvesIndeces is not None and len(curvesIndeces) > 0:
            idx_list = [i for i in curvesIndeces if 0 <= i < n_raw]
        else:
            if len(Ys) > 1 and len(Xs) == 1: idx_list = list(range(len(Ys)))
            elif len(Xs) > 1 and len(Ys) == 1: idx_list = list(range(len(Xs)))
            else: idx_list = list(range(min(len(Xs), len(Ys))))

        if namesForCurves is not None:
            raw_labels = list(namesForCurves)
            plotted_labels = [raw_labels[i] if i < len(raw_labels) else f"{nameForCurve} {i}" for i in idx_list]
        else:
            plotted_labels = [f"{nameForCurve} {i}" for i in idx_list]

        n_plot = len(idx_list)

        # --------------------
        # PALETTES
        # --------------------
        def _okabe_ito() -> List[str]:
            # Okabe–Ito (CB-safe). Reordered for line contrast on white; yellow placed last.
            # We will drop yellow for small N to maintain contrast.
            return ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#56B4E9', '#000000', '#E69F00', '#F0E442']

        def _tableau_cb10() -> List[str]:
            return ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']

        def _bw_triplets(n: int) -> List[Tuple[str, Any, Any]]:
            colors = ['0.0', '0.3', '0.6', '0.0', '0.45', '0.75', '0.15', '0.55']
            linestyles = ['-', '--', '-.', ':', (0, (1,1)), (0, (3,1,1,1)), (0, (5,2)), (0, (1,2))]
            markers = ['o', 's', 'D', '^', 'v', 'P', 'X', '']
            out: List[Tuple[str, Any, Any]] = []
            for i in range(n):
                out.append((colors[i % len(colors)], linestyles[i % len(linestyles)], markers[i % len(markers)]))
            return out

        def _resolve_palette(n: int) -> Tuple[List[str], Optional[List[Tuple[str, Any, Any]]]]:
            if isinstance(palette, Cycler):
                cols = list(palette)
                return [d.get('color', d) for d in cols][:n], None
            if isinstance(palette, (list, tuple, np.ndarray)):
                return [str(c) for c in palette][:n], None
            p = str(palette).lower()
            if p in ('cb_safe','okabeito'):
                cols = _okabe_ito()
                # drop the last entry (yellow) if small N to avoid low-contrast lines
                base = cols[:-1] if n <= 7 else cols
                if len(base) < n:
                    base = (base * int(np.ceil(n/len(base))))[:n]
                return base[:n], None
            if p in ('tableau_cb10','tab10'):
                cols = _tableau_cb10()
                if len(cols) < n: cols = (cols * int(np.ceil(n/len(cols))))[:n]
                return cols[:n], None
            if p in ('grayscale','greys','gray','grey'):
                vals = np.linspace(0.15, 0.85, max(1,n))
                return [str(float(v)) for v in vals[:n]], None
            if p == 'mono':
                return ['C0' for _ in range(n)], None
            if p == 'bw':
                return [], _bw_triplets(n)
            # default
            cols = _okabe_ito()
            if len(cols) < n: cols = (cols * int(np.ceil(n/len(cols))))[:n]
            return cols[:n], None

        colors, bw_triplets = _resolve_palette(n_plot)

        # Compact arrays for hist
        y_all = _stack_finite(Ys)
        x_all = _stack_finite(Xs)

        # Extras normalized
        def _normalize_extras(extras: Optional[Sequence[Union[ArrayLike, Dict[str, Any]]]]) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            if not extras: return out
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

        extraY = _normalize_extras(extraHistsY)
        extraX = _normalize_extras(extraHistsX)

        # Visible windows for bins='max'
        x_visible = (float(np.min(x_all)) if x_all.size else 0.0, float(np.max(x_all)) if x_all.size else 1.0)
        y_visible = (float(np.min(y_all)) if y_all.size else 0.0, float(np.max(y_all)) if y_all.size else 1.0)

        _x_bins = 'max' if use_max_bins else x_bins
        _y_bins = 'max' if use_max_bins else y_bins
        x_bins_final = _bins_from_arg(x_all, _x_bins, within_limits=(x_visible if bins_within_visible_window else None))
        y_bins_final = _bins_from_arg(y_all, _y_bins, within_limits=(y_visible if bins_within_visible_window else None))

        # --------------------
        # FIGURE / GRID
        # --------------------
        n_cols_histY = (1 if yhist_overlay else (1 + len(extraY))) if show_yhist else 0
        n_rows_histX = (1 if xhist_overlay else (1 + len(extraX))) if show_xhist else 0

        def _estimate_legend_width_in(labels: Sequence[str]) -> float:
            if not legend or not labels: return 0.0
            # rough estimate: handle + 0.35" + 0.0085*fs*max_len, clamp
            fs = 10.0  # will be scaled later to legend font; keep placeholder here
            char_w_in = 0.0085 * fs
            max_len = max(len(str(s)) for s in labels)
            handle_w_in = 0.40
            pad_in = 0.35
            return float(np.clip(handle_w_in + pad_in + char_w_in*max_len, LEG_W_MIN, LEG_W_MAX))

        prelim_leg_labels = list(plotted_labels) + (['Y data', 'μ', 'μ ± σ'] if show_yhist else [])
        if legend and guide_labels == 'legend':
            if xGuides:
                for g in xGuides:
                    if len(g) >= 2 and g[1] is not None:
                        prelim_leg_labels.append(str(g[1]))
            if yGuides:
                for g in yGuides:
                    if len(g) >= 2 and g[1] is not None:
                        prelim_leg_labels.append(str(g[1]))
        LEG_W = _estimate_legend_width_in(prelim_leg_labels) if legend else 0.0

        # Reserve legend space even if legend=False
        if reserve_legend_space and LEG_W <= 0.0:
            LEG_W = max(LEG_W_MIN, LEG_W)


        n_title_lines = int(max(1, title.count('\n') + (1 if title else 0)))
        TOP_PAD = TOP_PAD_BASE + (max(0, n_title_lines - 1)) * TOP_PAD_PER_EX

        fig_w = W_MAIN \
              + ((GAP_W + n_cols_histY*W_HCOL) if n_cols_histY > 0 else 0.0) \
              + (((LEG_PAD_W_CDF if (legend and (LEG_W > 0) and y_cdf) else LEG_PAD_W) + LEG_W) if (legend and LEG_W > 0) else 0.0)
        fig_h = TOP_PAD + H_MAIN + ((GAP_H + n_rows_histX*H_HROW) if n_rows_histX > 0 else 0.0)
        if figsize is not None:
            fig_w, fig_h = float(figsize[0]), float(figsize[1])


        # apply outer pads to overall figure size
        fig_w += RIGHT_OUTER_PAD
        fig_h += TOP_OUTER_PAD

        fig: Figure = plt.figure(name, figsize=(fig_w, fig_h), constrained_layout=False, clear=True)

        width_ratios  = [W_MAIN] \
                      + ([GAP_W] if n_cols_histY > 0 else []) \
                      + ([W_HCOL] * n_cols_histY) \
                      + ([(LEG_PAD_W_CDF if ((legend or reserve_legend_space) and y_cdf) else LEG_PAD_W), LEG_W] if ((legend and LEG_W > 0) or reserve_legend_space) else [])+ [RIGHT_OUTER_PAD]
        if n_rows_histX > 0 and (xhist_position or 'top').lower() == 'top':
            height_ratios = [TOP_OUTER_PAD] + ([H_HROW] * n_rows_histX) + [GAP_H] + [H_MAIN]
        else:
            height_ratios = [TOP_OUTER_PAD] + [H_MAIN] + ([GAP_H] if n_rows_histX > 0 else []) + ([H_HROW] * n_rows_histX)

        gs: GridSpec = fig.add_gridspec(
            nrows=len(height_ratios), ncols=len(width_ratios),
            width_ratios=width_ratios, height_ratios=height_ratios,
            wspace=0.0, hspace=0.0
        )

        # --------------------
        # MAIN AXES
        # --------------------
        _main_row = (len(height_ratios) - 1) if (n_rows_histX > 0 and (xhist_position or 'top').lower() == 'top') else 1
        ax_main: Axes = fig.add_subplot(gs[_main_row, 0])
        if grid_main:
            ax_main.grid(True, which='both', alpha=0.25)

        # label/ticks basic (style will be later propagated)
        ax_main.set_xlabel(x_label)
        ax_main.set_ylabel(y_label)
        _apply_scalar_formatter(ax_main, 'x')
        _apply_scalar_formatter(ax_main, 'y')

        if title:
            fig.suptitle(title, y=1.0 - (TOP_PAD/fig_h)*0.25)
        fig.subplots_adjust(top=1.0 - (TOP_PAD/fig_h), bottom=0.18, left=LEFT_FRAC, right=RIGHT_FRAC)

        # --------------------
        # DRAW CURVES
        # --------------------
        # palettes (color or BW triplets)
        lines_main: List[Line2D] = []
        for k, i in enumerate(idx_list):
            xarr = Xs[i] if i < len(Xs) else Xs[0]
            yarr = Ys[i] if i < len(Ys) else Ys[0]
            if bw_triplets is None:
                c = colors[k % len(colors)]
                ln, = ax_main.plot(xarr, yarr, color=c, zorder=2)
            else:
                c, ls, mk = bw_triplets[k]
                ln, = ax_main.plot(xarr, yarr, color=c, ls=ls, zorder=2)
            lines_main.append(ln)

        # Guides
        def _add_guides(ax: Axes) -> List[Line2D]:
            out: List[Line2D] = []
            if xGuides:
                for g in xGuides:
                    xv, lab, col = _unpack_guide(g)
                    ln = ax.axvline(xv, ls='--', color=col, zorder=1)
                    out.append(ln)
                    if guide_labels == 'axes' and lab:
                        ax.text(xv, 1.0, lab, transform=ax.get_xaxis_transform(),
                                ha='center', va='bottom', clip_on=True, zorder=5)
            if yGuides:
                for g in yGuides:
                    yv, lab, col = _unpack_guide(g)
                    ln = ax.axhline(yv, ls='--', color=col, zorder=1)
                    out.append(ln)
                    if guide_labels == 'axes' and lab:
                        ax.text(1.0, yv, lab, transform=ax.get_yaxis_transform(),
                                ha='left', va='center', clip_on=True, zorder=5)
            return out

        guide_lines = _add_guides(ax_main)

        
        
        # --------------------
        # COPY-FROM-MAIN STYLE: sample sizes/widths from main (no rcParams lookups)
        # --------------------
        try:
            fig.canvas.draw()
        except Exception:
            pass

        # sample linewidths
        _LW_AXES = 1.0
        if ax_main.spines:
            try:
                _LW_AXES = float(np.median([sp.get_linewidth() for sp in ax_main.spines.values() if sp is not None]))
            except Exception:
                _LW_AXES = list(ax_main.spines.values())[0].get_linewidth()
        _LW_LINE = float(lines_main[0].get_linewidth()) if lines_main else float(_LW_AXES*1.2)
        _LW_MU   = 1.05 * _LW_LINE
        _LW_SIG  = 0.95 * _LW_LINE
        _LW_GUIDE= 1.00 * _LW_LINE
        _LW_CDF  = max(1.15 * _LW_LINE, 1.2)

        # sample tick widths
        def _sample_tick_w(ax: Axes, minor: bool=False) -> float:
            lines = ax.xaxis.get_ticklines(minor=minor) + ax.yaxis.get_ticklines(minor=minor)
            vals = [float(tl.get_linewidth()) for tl in lines if hasattr(tl, 'get_linewidth')]
            if vals: return float(np.median(vals))
            return float(_LW_AXES*(0.6 if minor else 0.8))

        _W_TICK_MAJOR = _sample_tick_w(ax_main, minor=False)
        _W_TICK_MINOR = _sample_tick_w(ax_main, minor=True)

        # sample font sizes
        _FS_LABEL = float(ax_main.xaxis.label.get_size() or 10.0)
        xt = [t for t in ax_main.get_xticklabels() if hasattr(t, 'get_size')]
        yt = [t for t in ax_main.get_yticklabels() if hasattr(t, 'get_size')]
        _FS_TICK = None
        for t in (xt+yt):
            try:
                _FS_TICK = float(t.get_size())
                if _FS_TICK: break
            except Exception:
                pass
        if not _FS_TICK:
            _FS_TICK = _FS_LABEL
        _FS_LEG = _FS_TICK


        # adjust MAIN tick size: enlarge by 1.18x with absolute floor 12 pt
        _FS_TICK_MAIN = max(float(_FS_TICK)*1.26, 13.0)
        # histogram tick font size: slightly smaller than main (0.82x) with floor 9 pt
        _FS_TICK_HIST = max(float(_FS_TICK*hist_tick_scale), 9.0)
        
        def _apply_axes_style(ax: Axes, tick_size: float) -> None:
            for side in ('left','right','bottom','top'):
                sp = ax.spines.get(side, None)
                if sp is not None: sp.set_linewidth(_LW_AXES)
            ax.tick_params(axis='both', which='both', labelsize=tick_size, width=_W_TICK_MAJOR)
            ax.tick_params(axis='both', which='minor', width=_W_TICK_MINOR)

        # apply sampled style to main
        _FS_LABEL = max(1.35*_FS_LABEL, _FS_TICK_MAIN + 4.0, 13.5)
        _apply_axes_style(ax_main, _FS_TICK_MAIN)
        # Thicken main spines and tick widths (main only)
        for _sp in ax_main.spines.values():
            try:
                _sp.set_linewidth(max(_sp.get_linewidth(), 1.6))
            except Exception:
                pass
        try:
            ax_main.tick_params(which='major', width=1.2)
            ax_main.tick_params(which='minor', width=1.0)
        except Exception:
            pass
        ax_main.xaxis.label.set_size(_FS_LABEL)
        ax_main.yaxis.label.set_size(_FS_LABEL)
        for ln in lines_main: ln.set_linewidth(_LW_LINE)
        for ln in guide_lines: ln.set_linewidth(_LW_GUIDE)
# --------------------
        # HISTOGRAMS (Y RIGHT)
        # --------------------
        yhist_axes: List[Axes] = []
        if show_yhist and ((1 if yhist_overlay else (1 + len(extraY))) > 0):
            col0 = 2  # [0=main, 1=gap, 2=first-hist]
            n_cols_histY_eff = (1 if yhist_overlay else (1 + len(extraY)))
            if yhist_overlay:
                axy = fig.add_subplot(gs[_main_row, col0])
                yhist_axes.append(axy)
            else:
                for j in range(n_cols_histY_eff):
                    axy = fig.add_subplot(gs[_main_row, col0 + j])
                    yhist_axes.append(axy)

            for axy in yhist_axes:
                if grid_hist: axy.grid(True, which='major', alpha=0.25)
                _apply_axes_style(axy, tick_size=_FS_TICK_HIST)

            def _hist_y(ax: Axes, dat: np.ndarray, color: str='0.6', alpha: float=0.55) -> None:
                ax.hist(dat, bins=y_bins_final, orientation='horizontal',
                        density=density, histtype='stepfilled',
                        color=color, alpha=alpha, zorder=1)

            _hist_y(yhist_axes[0], y_all, color='0.6', alpha=(0.55 if yhist_overlay else 0.45))

            if not yhist_overlay and len(yhist_axes) > 1:
                for k, h in enumerate(extraY[:len(yhist_axes)-1], start=1):
                    dat = _finite(np.asarray(h.get('data', [])))
                    if dat.size == 0: continue
                    col = h.get('color', '0.6')
                    yhist_axes[k].hist(dat, bins=y_bins_final, orientation='horizontal',
                                       density=density, histtype='stepfilled',
                                       color=col, alpha=float(h.get('alpha', 0.45)), zorder=1)

            y_min, y_max = ax_main.get_ylim()
            for axy in yhist_axes:
                axy.set_ylim(y_min, y_max)

            if y_log_density:
                for axy in yhist_axes:
                    axy.set_xscale('log', base=10.0)
                    _apply_log_formatter(axy, 'x')
            else:
                for axy in yhist_axes:
                    _apply_scalar_formatter(axy, 'x')

            if yGuides:
                for yv, _, col in map(lambda t: _unpack_guide(t), yGuides):
                    for axy in yhist_axes:
                        ln = axy.axhline(yv, ls='--', color=col, zorder=3)
                        ln.set_linewidth(_LW_GUIDE)

            # stats μ/σ
            if 'mu' in showYStats or 'sigma' in showYStats:
                if y_all.size > 0:
                    mu = float(np.mean(y_all))
                    sig = float(np.std(y_all, ddof=0))
                else:
                    mu = 0.0; sig = 0.0
                ln = yhist_axes[0].axhline(mu, color='0.1',  ls='--', zorder=3); ln.set_linewidth(_LW_MU)
                if 'sigma' in showYStats and sig > 0:
                    ln = yhist_axes[0].axhline(mu - sig, color='0.35', ls='--', zorder=3); ln.set_linewidth(_LW_SIG)
                    ln = yhist_axes[0].axhline(mu + sig, color='0.35', ls='--', zorder=3); ln.set_linewidth(_LW_SIG)

            # long axis (y) ticks policy
            _tick_policy = (yhist_ticks or 'auto').lower()
            if _tick_policy not in ('auto','right','none'): _tick_policy = 'auto'
            if _tick_policy == 'auto':
                for axy in yhist_axes: _hide_all_ticks(axy, 'y')
            elif _tick_policy == 'right':
                for axy in yhist_axes: _ticks_right(axy)
            else:
                for axy in yhist_axes: _hide_all_ticks(axy, 'y')

            # short axis (x) ticks position to avoid collision with main xticks
            short_opt = (yhist_xticks or 'top').lower()
            _cdf_side_y = (y_cdf_ticks or 'top').lower()
            if _cdf_side_y not in ('top','bottom','none'): _cdf_side_y = 'top'
            if _cdf_side_y != 'none' and short_opt == _cdf_side_y:
                short_opt = 'bottom' if short_opt == 'top' else 'top'
            for axy in yhist_axes:
                # disable both then enable only chosen side with consistent size
                axy.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labeltop=False, width=_W_TICK_MAJOR)
                if short_opt == 'bottom':
                    _ticks_bottom(axy)
                    axy.tick_params(axis='x', which='both', labelsize=_FS_TICK_HIST, width=_W_TICK_MAJOR)
                else:
                    _ticks_top(axy)
                    axy.tick_params(axis='x', which='both', labelsize=_FS_TICK_HIST, width=_W_TICK_MAJOR)

            # CDF twin on short axis (x)
            if y_cdf:
                for axy in yhist_axes:
                    twin = axy.twiny()
                    # store for reuse and always linear for CDF
                    try:
                        setattr(axy, '_cdf_twin_x', twin)
                    except Exception:
                        pass
                    twin.set_xscale('linear'); twin.set_yscale('linear')
                    twin.tick_params(axis='x', which='both', labelsize=_FS_TICK_HIST, width=_W_TICK_MAJOR)
                    twin.tick_params(axis='x', which='minor', width=_W_TICK_MINOR)
                    tick_mode = _cdf_side_y
                    twin.tick_params(axis='x', which='both', top=False, bottom=False, labeltop=False, labelbottom=False, width=_W_TICK_MAJOR)
                    if tick_mode == 'top':
                        twin.xaxis.set_ticks_position('top')
                        twin.tick_params(axis='x', which='both', top=True, labeltop=True)
                        if 'top' in twin.spines: twin.spines['top'].set_linewidth(_LW_AXES)
                    elif tick_mode == 'bottom':
                        twin.xaxis.set_ticks_position('bottom')
                        twin.tick_params(axis='x', which='both', bottom=True, labelbottom=True)
                        if 'bottom' in twin.spines: twin.spines['bottom'].set_linewidth(_LW_AXES)

                    edges, cdf = _cdf_from_hist(y_all if axy is yhist_axes[0] else _finite(np.asarray(extraY[yhist_axes.index(axy)-1].get('data', []))), y_bins_final)
                    twin.set_xlim(0.0, 1.0)
                    twin.set_ylim(axy.get_ylim())
                    ln, = twin.plot(cdf, edges, color='0.25', zorder=4); ln.set_linewidth(_LW_CDF)

                    # reference probabilities (verticals at x=p)
                    if cdf_ref_probs_y:
                        cols = list(cdf_ref_colors_y) if cdf_ref_colors_y is not None else []
                        for j, p in enumerate(cdf_ref_probs_y):
                            col = (cols[j % len(cols)] if cols else '0.35')
                            lref = twin.axvline(float(p), ls=(0,(2,2)), color=col, zorder=4)
                            lref.set_linewidth(_LW_GUIDE)
                            if cdf_ref_labels and cdf_ref_fmt is not None:
                                twin.text(float(p), axy.get_ylim()[1], str(cdf_ref_fmt(float(p))), ha='center', va='bottom')

                # CDF ticks
                if (y_cdf_ticks or 'top').lower() != 'none':
                    for axy in yhist_axes:
                        twin = getattr(axy, '_cdf_twin_x', None) or axy.twiny()
                        try:
                            setattr(axy, '_cdf_twin_x', twin)
                        except Exception:
                            pass
                        twin.xaxis.set_major_locator(FixedLocator([0.0, 0.5, 1.0]))

        # --------------------
        # HISTOGRAMS (X TOP/BOTTOM)
        # --------------------
        xhist_axes: List[Axes] = []
        if show_xhist and ((1 if xhist_overlay else (1 + len(extraX))) > 0):
            if (xhist_position or 'top').lower() == 'top':
                row0 = 1
            else:
                row0 = 3
            n_rows_histX_eff = (1 if xhist_overlay else (1 + len(extraX)))
            if xhist_overlay:
                axx = fig.add_subplot(gs[row0, 0])
                xhist_axes.append(axx)
            else:
                for j in range(n_rows_histX_eff):
                    axx = fig.add_subplot(gs[row0 + j, 0])
                    xhist_axes.append(axx)

            for axx in xhist_axes:
                if grid_hist: axx.grid(True, which='major', alpha=0.25)
                _apply_axes_style(axx, tick_size=_FS_TICK_HIST)

            def _hist_x(ax: Axes, dat: np.ndarray, color: str='0.6', alpha: float=0.55) -> None:
                ax.hist(dat, bins=x_bins_final, orientation='vertical',
                        density=density, histtype='stepfilled',
                        color=color, alpha=alpha, zorder=1)

            _hist_x(xhist_axes[0], x_all, color='0.6', alpha=(0.55 if xhist_overlay else 0.45))

            if not xhist_overlay and len(xhist_axes) > 1:
                for k, h in enumerate(extraX[:len(xhist_axes)-1], start=1):
                    dat = _finite(np.asarray(h.get('data', [])))
                    if dat.size == 0: continue
                    col = h.get('color', '0.6')
                    xhist_axes[k].hist(dat, bins=x_bins_final, orientation='vertical',
                                       density=density, histtype='stepfilled',
                                       color=col, alpha=float(h.get('alpha', 0.45)), zorder=1)

            x_min, x_max = ax_main.get_xlim()
            for axx in xhist_axes:
                axx.set_xlim(x_min, x_max)

            if x_log_density:
                for axx in xhist_axes:
                    axx.set_yscale('log', base=10.0)
                    _apply_log_formatter(axx, 'y')
            else:
                for axx in xhist_axes:
                    _apply_scalar_formatter(axx, 'y')

            if xGuides:
                for xv, _, col in map(lambda t: _unpack_guide(t), xGuides):
                    for axx in xhist_axes:
                        ln = axx.axvline(xv, ls='--', color=col, zorder=3); ln.set_linewidth(_LW_GUIDE)

            if 'mu' in showXStats or 'sigma' in showXStats:
                if x_all.size > 0:
                    mu = float(np.mean(x_all))
                    sig = float(np.std(x_all, ddof=0))
                else:
                    mu = 0.0; sig = 0.0
                ln = xhist_axes[0].axvline(mu,           color='0.1',  ls='--', zorder=3); ln.set_linewidth(_LW_MU)
                if 'sigma' in showXStats and sig > 0:
                    ln = xhist_axes[0].axvline(mu - sig, color='0.35', ls='--', zorder=3); ln.set_linewidth(_LW_SIG)
                    ln = xhist_axes[0].axvline(mu + sig, color='0.35', ls='--', zorder=3); ln.set_linewidth(_LW_SIG)

            _x_tick_pol = (xhist_ticks or 'auto').lower()
            if _x_tick_pol not in ('auto','bottom','top','none'): _x_tick_pol = 'auto'
            if _x_tick_pol == 'auto':
                for axx in xhist_axes: _hide_all_ticks(axx, 'x')
            elif _x_tick_pol == 'bottom':
                for axx in xhist_axes: _ticks_bottom(axx)
            elif _x_tick_pol == 'top':
                for axx in xhist_axes: _ticks_top(axx)
            else:
                for axx in xhist_axes: _hide_all_ticks(axx, 'x')

            if x_cdf:
                for axx in xhist_axes:
                    twin = axx.twinx()
                    # store for reuse
                    try:
                        setattr(axx, '_cdf_twin_y', twin)
                    except Exception:
                        pass
                    twin.set_xscale('linear'); twin.set_yscale('linear')
                    twin.tick_params(axis='y', which='both', labelsize=_FS_TICK_HIST, width=_W_TICK_MAJOR)
                    twin.tick_params(axis='y', which='minor', width=_W_TICK_MINOR)
                    side = (x_cdf_ticks or 'right').lower()
                    # assign density ticks to the opposite side to avoid duplication
                    if side == 'left':
                        axx.yaxis.set_ticks_position('right')
                        axx.tick_params(axis='y', which='both', right=True, labelright=True, left=False, labelleft=False, width=_W_TICK_MAJOR, labelsize=_FS_TICK_HIST)
                    else:
                        axx.yaxis.set_ticks_position('left')
                        axx.tick_params(axis='y', which='both', left=True, labelleft=True, right=False, labelright=False, width=_W_TICK_MAJOR, labelsize=_FS_TICK_HIST)
                    if side == 'right':
                        twin.yaxis.set_ticks_position('right')
                        twin.tick_params(axis='y', which='both', right=True, labelright=True)
                        if 'right' in twin.spines: twin.spines['right'].set_linewidth(_LW_AXES)
                    elif side == 'left':
                        twin.yaxis.set_ticks_position('left')
                        twin.tick_params(axis='y', which='both', left=True, labelleft=True)
                        if 'left' in twin.spines: twin.spines['left'].set_linewidth(_LW_AXES)

                    twin.set_ylim(0.0, 1.0)
                    dat = x_all if axx is xhist_axes[0] else _finite(np.asarray(extraX[xhist_axes.index(axx)-1].get('data', [])))
                    edges, cdf = _cdf_from_hist(dat, x_bins_final)
                    ln, = twin.plot(edges, cdf, color='0.25', zorder=4); ln.set_linewidth(_LW_CDF)

                    # reference probabilities (horizontals at y=p)
                    if cdf_ref_probs_x:
                        cols = list(cdf_ref_colors_x) if cdf_ref_colors_x is not None else []
                        for j, p in enumerate(cdf_ref_probs_x):
                            col = (cols[j % len(cols)] if cols else '0.35')
                            lref = twin.axhline(float(p), ls=(0,(2,2)), color=col, zorder=4)
                            lref.set_linewidth(_LW_GUIDE)
                            if cdf_ref_labels and cdf_ref_fmt is not None:
                                twin.text(axx.get_xlim()[1], float(p), str(cdf_ref_fmt(float(p))), ha='left', va='center')

                # CDF ticks
                if (x_cdf_ticks or 'right').lower() != 'none':
                    for axx in xhist_axes:
                        twin = getattr(axx, '_cdf_twin_y', None) or axx.twinx()
                        try:
                            setattr(axx, '_cdf_twin_y', twin)
                        except Exception:
                            pass
                        twin.yaxis.set_major_locator(FixedLocator([0.0, 0.5, 1.0]))

        # --------------------
        # Y TICK policy on main (sparse in [-1,1])
        # --------------------
        try:
            ymin, ymax = ax_main.get_ylim()
            if np.isclose([ymin, ymax], [-1.0, 1.0], atol=1e-6).all():
                ax_main.yaxis.set_major_locator(FixedLocator([-1.0, -0.5, 0.0, 0.5, 1.0]))
            else:
                ax_main.yaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))
        except Exception:
            pass

        # --------------------
        # LEGEND (RIGHT COLUMN)
        # --------------------
        if legend and LEG_W > 0:
            ax_leg: Axes = fig.add_subplot(gs[:, -2])
            ax_leg.axis('off')
            handles: List[Any] = []
            labels: List[str] = []

            for ln, lab in zip(lines_main, plotted_labels):
                handles.append(ln); labels.append(lab)

            if guide_labels == 'legend':
                if xGuides:
                    for g in xGuides:
                        xv, lab, col = _unpack_guide(g)
                        if lab:
                            h = Line2D([0],[0], color=col, ls='--'); h.set_linewidth(_LW_GUIDE)
                            handles.append(h); labels.append(str(lab))
                if yGuides:
                    for g in yGuides:
                        yv, lab, col = _unpack_guide(g)
                        if lab:
                            h = Line2D([0],[0], color=col, ls='--'); h.set_linewidth(_LW_GUIDE)
                            handles.append(h); labels.append(str(lab))

            if show_yhist:
                rect = Patch(facecolor='0.7', edgecolor='0.3', alpha=0.55, label='Y data')
                handles.append(rect); labels.append('Y data')
                hmu  = Line2D([0],[0], color='0.1',  ls='--'); hmu.set_linewidth(_LW_MU)
                hsig = Line2D([0],[0], color='0.35', ls='--'); hsig.set_linewidth(_LW_SIG)
                handles.extend([hmu, hsig]); labels.extend([r'$\mu$', r'$\mu \pm \sigma$'])

            # legend font from main tick size (close enough), since we avoid rc lookups
            from matplotlib.font_manager import FontProperties
            leg = ax_leg.legend(handles, labels, loc='center left', frameon=False, borderaxespad=0.0,
                          prop=FontProperties(size=_FS_LEG))
            try:
                leg.get_frame().set_linewidth(_LW_AXES)
                leg.get_frame().set_edgecolor('0.0')
                leg.get_frame().set_alpha(1.0)
                leg.get_frame().set_facecolor('white')
            except Exception:
                pass

        meta: Dict[str, Any] = {
            "figsize_in": (fig_w, fig_h),
            "main_panel_in": (W_MAIN, H_MAIN),
            "n_curves_plotted": n_plot,
            "plotted_indices": idx_list,
            "n_cols_histY": n_cols_histY,
            "n_rows_histX": n_rows_histX,
            "legend_width_in": LEG_W,
            "yhist_ticks_policy": (yhist_ticks or 'auto'),
            "xhist_ticks_policy": (xhist_ticks or 'auto'),
            "yhist_xticks_short_axis": (yhist_xticks or 'top'),
            "y_cdf_ticks": y_cdf_ticks,
            "x_cdf_ticks": x_cdf_ticks,
            "palette": palette,
            "guide_labels": guide_labels,
            "hist_tick_scale": hist_tick_scale,
        }
        return fig, ax_main, meta


def report_layout(fig: Figure, meta: Optional[Dict[str, Any]] = None) -> None:
    """Print a numeric report of effective sizes (in) of the main and the whole figure."""
    try:
        W, H = fig.get_size_inches()
    except Exception:
        W = H = float('nan')

    ax = fig.axes[0] if fig.axes else None
    w_main_in = h_main_in = None
    fig_w_in = W
    fig_h_in = H

    if isinstance(meta, dict):
        ax = meta.get('axes_main', ax)
        w_main_in = meta.get('w_main_in', None)
        h_main_in = meta.get('h_main_in', None)
        fig_w_in  = meta.get('fig_w_in', fig_w_in)
        fig_h_in  = meta.get('fig_h_in', fig_h_in)

    if ax is not None:
        bb = ax.get_position()
        main_w_eff = (bb.x1 - bb.x0) * W
        main_h_eff = (bb.y1 - bb.y0) * H
    else:
        main_w_eff = main_h_eff = float('nan')

    print("----- Layout report -----")
    print(f"Figure size (in): {W:.3f} × {H:.3f}  | meta fig (in): {fig_w_in:.3f} × {fig_h_in:.3f}")
    if (w_main_in is not None) and (h_main_in is not None) and np.isfinite(main_w_eff) and np.isfinite(main_h_eff):
        dw = main_w_eff - w_main_in
        dh = main_h_eff - h_main_in
        print(f"Main design (in): {w_main_in:.3f} × {h_main_in:.3f}")
        print(f"Δ main (eff - design): dW={dw:+.3e} in, dH={dh:+.3e} in")
    print("-------------------------")