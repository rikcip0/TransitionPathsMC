from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import (
    FormatStrFormatter, ScalarFormatter, FixedLocator,
    LogLocator, LogFormatterSciNotation
)

ArrayLike = Union[np.ndarray, Sequence[float], List[float]]

def multipleCurvesAndHist_clean(
    # --- dati principali ---
    name: str,
    title: str,
    x_list: Union[ArrayLike, Sequence[ArrayLike], np.ndarray],
    x_label: str,
    y_list: Union[ArrayLike, Sequence[ArrayLike], np.ndarray],
    y_label: str,

    # --- selezione curve/etichette ---
    nameForCurve: str = 'traj',
    curvesIndeces: Optional[Sequence[int]] = None,   # spelling storico
    curvesIndices: Optional[Sequence[int]] = None,   # alias
    namesForCurves: Optional[Sequence[str]] = None,

    # --- istogrammi & layout logico ---
    show_yhist: bool = True,
    show_xhist: bool = False,
    yhist_overlay: bool = True,   # un solo pannello per Y-hist (overlay extras)
    xhist_overlay: bool = True,   # come sopra per X-hist
    xhist_position: str = 'top',  # 'top' | 'bottom'

    # policy tick (solo posizionamento, NESSUN resize font)
    yhist_ticks: str = "auto",    # long axis di Y-hist: 'auto' | 'right' | 'none'
    xhist_ticks: str = "auto",    # long axis di X-hist: 'auto' | 'bottom' | 'top' | 'none'
    yhist_xticks: str = "top",    # short axis di Y-hist (pdf): 'bottom' | 'top' | 'none'

    # curve extra per istogrammi (array o dict {'data','label','color','alpha'})
    extraHistsY: Optional[Sequence[Union[ArrayLike, Dict[str, Any]]]] = None
    ,
    extraHistsX: Optional[Sequence[Union[ArrayLike, Dict[str, Any]]]] = None,

    # binning
    y_bins: Union[str, int, Sequence[float]] = 'auto',  # 'auto'/'fd'/'scott'/'sturges'/'sqrt'/'max'/int/edges
    x_bins: Union[str, int, Sequence[float]] = 'auto',
    bins_within_visible_window: bool = True,
    use_max_bins: bool = False,

    # densità/log
    density: bool = True,
    y_log_density: bool = False,
    x_log_density: bool = False,

    # CDF (indipendenti)
    y_cdf: bool = False,
    x_cdf: bool = False,
    y_cdf_ticks: str = "top",     # 'top' | 'bottom' | 'none'
    x_cdf_ticks: str = "right",   # 'right' | 'left' | 'none'

    # overlays statistici sugli istogrammi
    showYStats: Sequence[str] = ('mu', 'sigma'),
    showXStats: Sequence[str] = ('mu', 'sigma'),

    # griglia
    grid_main: bool = True,
    grid_hist: bool = True,

    # guide nel main: tuple (value,) | (value,label) | (value,label,color)
    xGuides: Optional[Sequence[Tuple[Any, ...]]] = None,
    yGuides: Optional[Sequence[Tuple[Any, ...]]] = None,
    guide_labels: str = 'legend',    # 'legend' | 'axes' | 'none'

    # legenda (colonna dedicata opzionale)
    legend: bool = True,
    legend_width_in: float = 0.0,    # 0 => niente colonna
    legend_pad_in: float = 0.12,

    # palette
    palette: Union[str, Sequence[str]] = "cb_safe",

    # --- GEOMETRIA ESPLICITA (pollici) ---
    main_w_in: float = 1.604,
    main_h_in: float = 1.060,
    yhist_w_in: float = 0.90,        # larghezza colonna Y-hist (altezza = main_h_in)
    xhist_h_in: float = 0.90,        # altezza riga X-hist (larghezza = main_w_in)
    gap_w_in: float = 0.10,
    gap_h_in: float = 0.10,
    left_in: float = 0.44,
    right_in: float = 0.08,
    bottom_in: float = 0.34,
    top_in: float = 0.18,
) -> Tuple[Figure, Axes, Dict[str, Any]]:
    """
    Versione 'clean': NESSUN auto-styling (no font/tick/linewidth scaling).
    Geometria esplicita in input. Restituisce (fig, ax_main, meta) + hook utili in meta.
    """
    # -------------------- helper (no style) --------------------
    def _finite(x: np.ndarray) -> np.ndarray:
        m = np.isfinite(x); return x[m]

    def _as_arrays(ls: Union[ArrayLike, Sequence[ArrayLike], np.ndarray]) -> List[np.ndarray]:
        if isinstance(ls, np.ndarray) and ls.ndim > 1:
            return [np.asarray(a, dtype=float) for a in ls]
        arr = np.asarray(ls, dtype=object)
        if arr.ndim == 1 and len(arr) > 0 and not isinstance(arr[0], (list, tuple, np.ndarray)):
            return [np.asarray(arr, dtype=float)]
        return [np.asarray(a, dtype=float) for a in ls]

    def _stack_finite(seq: Sequence[np.ndarray]) -> np.ndarray:
        if not seq: return np.array([], dtype=float)
        return np.concatenate([_finite(np.asarray(a, dtype=float)) for a in seq]) if len(seq) else np.array([], dtype=float)

    def _bin_edges_magnetization_complete(data: np.ndarray) -> np.ndarray:
        arr = np.asarray(data, dtype=float); arr = arr[np.isfinite(arr)]
        vals = np.sort(np.unique(arr))
        if vals.size <= 1:
            return np.array([vals[0]-0.5, vals[0]+0.5]) if vals.size == 1 else np.array([0.0, 1.0])
        diffs = np.diff(vals); pos = diffs[diffs > 0]
        step = float(np.min(pos)) if pos.size else float(diffs[0])
        lo = float(vals[0]) - 0.5*step; hi = float(vals[-1]) + 0.5*step
        eps = 1e-12 * max(1.0, abs(hi - lo))
        return np.arange(lo - eps, hi + step + eps, step, dtype=float)

    def _bins_from_arg_strict(data: np.ndarray,
                              arg: Union[str, int, Sequence[float]],
                              within_limits: Optional[Tuple[float,float]] = None) -> np.ndarray:
        data = np.asarray(data, dtype=float); data = data[np.isfinite(data)]
        if isinstance(arg, (list, tuple, np.ndarray)) and np.asarray(arg).ndim == 1:
            e = np.array(sorted(np.unique(np.asarray(arg, dtype=float))), dtype=float)
            if e.size < 2:
                lo = float(np.min(data)) if data.size else 0.0
                hi = float(np.max(data)) if data.size else 1.0
                e = np.array([lo, hi], dtype=float)
            return e
        if isinstance(arg, str):
            a = arg.lower()
            if a in ("magnetizationcomplete", "magnetization_complete", "m_complete"):
                return _bin_edges_magnetization_complete(data)
            if a in ("auto", "fd", "scott", "sturges", "sqrt", "max"):
                if a == "max":
                    if data.size == 0:
                        e = np.array([0.0, 1.0], dtype=float)
                    else:
                        vals = np.sort(np.unique(data))
                        if vals.size == 1:
                            e = np.array([vals[0]-0.5, vals[0]+0.5], dtype=float)
                        else:
                            mids = (vals[1:]+vals[:-1])*0.5
                            e = np.concatenate([[vals[0]-(mids[0]-vals[0])], mids,
                                                [vals[-1]+(vals[-1]-mids[-1])]]).astype(float)
                else:
                    e = np.histogram_bin_edges(data, bins=a).astype(float)
            else:
                raise ValueError(f"Unsupported bins spec: {arg!r}")
        elif isinstance(arg, int):
            e = np.histogram_bin_edges(data, bins=arg).astype(float)
        else:
            e = np.histogram_bin_edges(data, bins=arg).astype(float)

        if within_limits is not None:
            lo, hi = map(float, within_limits)
            ee = np.asarray(e, dtype=float)
            ee = ee[(ee >= lo) & (ee <= hi)]
            if ee.size < 2:
                ee = np.array([lo, hi], dtype=float)
            if ee[0] > lo: ee = np.concatenate([[lo], ee])
            if ee[-1] < hi: ee = np.concatenate([ee, [hi]])
            e = ee

        e = np.array(sorted(np.unique(e)), dtype=float)
        if e.size < 2:
            lo = float(np.min(data)) if data.size else 0.0
            hi = float(np.max(data)) if data.size else 1.0
            e = np.array([lo, hi], dtype=float)
        return e

    def _apply_scalar_formatter(ax: Axes, axis: str='x') -> None:
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((-2, 2))
        fmt.set_useOffset(False)
        (ax.xaxis if axis=='x' else ax.yaxis).set_major_formatter(fmt)

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

    def _unpack_guide(t: Tuple[Any, ...]) -> Tuple[float, Optional[str], str]:
        if len(t) == 1: return float(t[0]), None, '0.25'
        if len(t) == 2: return float(t[0]), str(t[1]), '0.25'
        return float(t[0]), str(t[1]), str(t[2])

    # -------------------- normalizzazione input --------------------
    Xs = _as_arrays(x_list)
    Ys = _as_arrays(y_list)
    n_raw = max(len(Xs), len(Ys))
    if len(Xs) == 1 and n_raw > 1: Xs = [Xs[0] for _ in range(n_raw)]
    if len(Ys) == 1 and n_raw > 1: Ys = [Ys[0] for _ in range(n_raw)]

    if curvesIndices is not None and curvesIndeces is None:
        curvesIndeces = curvesIndices
    if curvesIndeces: idx_list = [i for i in curvesIndeces if 0 <= i < n_raw]
    else:
        if len(Ys) > 1 and len(Xs) == 1: idx_list = list(range(len(Ys)))
        elif len(Xs) > 1 and len(Ys) == 1: idx_list = list(range(len(Xs)))
        else: idx_list = list(range(min(len(Xs), len(Ys))))

    if namesForCurves is not None:
        raw_labels = list(namesForCurves)
        plotted_labels = [raw_labels[i] if i < len(raw_labels) else f"{nameForCurve} {i}" for i in idx_list]
    else:
        plotted_labels = [f"{nameForCurve} {i}" for i in idx_list]

    colors = (['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#56B4E9', '#000000', '#E69F00', '#F0E442']
              if isinstance(palette, str) else [str(c) for c in palette])
    if len(colors) < len(idx_list):
        k = int(np.ceil(len(idx_list)/len(colors)))
        colors = (colors * k)[:len(idx_list)]

    x_all = _stack_finite(Xs)
    y_all = _stack_finite(Ys)

    # bins (con finestra visibile opzionale)
    _x_bins = 'max' if use_max_bins else x_bins
    _y_bins = 'max' if use_max_bins else y_bins
    def _is_edges(v): return isinstance(v, (list, tuple, np.ndarray)) and np.asarray(v).ndim == 1
    x_vis = (float(np.min(x_all)) if x_all.size else 0.0, float(np.max(x_all)) if x_all.size else 1.0)
    y_vis = (float(np.min(y_all)) if y_all.size else 0.0, float(np.max(y_all)) if y_all.size else 1.0)
    x_within = x_vis if (bins_within_visible_window and not (_is_edges(_x_bins) or (isinstance(_x_bins,str) and _x_bins.lower() in ('magnetizationcomplete','magnetization_complete','m_complete')))) else None
    y_within = y_vis if (bins_within_visible_window and not (_is_edges(_y_bins) or (isinstance(_y_bins,str) and _y_bins.lower() in ('magnetizationcomplete','magnetization_complete','m_complete')))) else None
    x_bins_final = _bins_from_arg_strict(x_all, _x_bins, within_limits=x_within)
    y_bins_final = _bins_from_arg_strict(y_all, _y_bins, within_limits=y_within)

    # -------------------- figura / griglia (solo geometria) --------------------
    n_cols_histY = (1 if yhist_overlay else (1 + len(extraHistsY or []))) if show_yhist else 0
    n_rows_histX = (1 if xhist_overlay else (1 + len(extraHistsX or []))) if show_xhist else 0

    width_ratios  = [left_in, main_w_in]
    if n_cols_histY > 0: width_ratios += [gap_w_in] + [yhist_w_in]*n_cols_histY
    if legend and legend_width_in > 0.0: width_ratios += [legend_pad_in, legend_width_in]
    width_ratios += [right_in]

    if n_rows_histX > 0 and (xhist_position or 'top').lower() == 'top':
        height_ratios = [top_in] + [xhist_h_in]*n_rows_histX + [gap_h_in] + [main_h_in] + [bottom_in]
        main_row = 1 + n_rows_histX + 1
    else:
        height_ratios = [top_in] + [main_h_in] + ([gap_h_in] + [xhist_h_in]*n_rows_histX if n_rows_histX>0 else []) + [bottom_in]
        main_row = 1

    fig_w = float(sum(width_ratios)); fig_h = float(sum(height_ratios))
    fig: Figure = plt.figure(name, figsize=(fig_w, fig_h), constrained_layout=False, clear=True)
    gs: GridSpec = fig.add_gridspec(
        nrows=len(height_ratios), ncols=len(width_ratios),
        width_ratios=width_ratios, height_ratios=height_ratios,
        wspace=0.0, hspace=0.0
    )

    # -------------------- main axes --------------------
    ax_main: Axes = fig.add_subplot(gs[main_row, 1])
    if grid_main: ax_main.grid(True, which='both', alpha=0.25)
    ax_main.set_xlabel(x_label); ax_main.set_ylabel(y_label)
    if title: fig.suptitle(title)
    _apply_scalar_formatter(ax_main, 'x'); _apply_scalar_formatter(ax_main, 'y')

    # curve
    lines_main: List[Line2D] = []
    for k, i in enumerate(idx_list):
        xarr = Xs[i] if i < len(Xs) else Xs[0]
        yarr = Ys[i] if i < len(Ys) else Ys[0]
        ln, = ax_main.plot(xarr, yarr, color=colors[k%len(colors)], zorder=2, label=plotted_labels[k] if k < len(plotted_labels) else None)
        lines_main.append(ln)

    # guide lines (con label/gid per styling esterno)
    def _add_guides(ax: Axes) -> List[Line2D]:
        out: List[Line2D] = []
        if xGuides:
            for g in xGuides:
                xv, lab, col = _unpack_guide(g)
                ln = ax.axvline(xv, ls='--', color=col, zorder=1, label=(lab or 'x-guide'))
                try: ln.set_gid('x-guideline')
                except Exception: pass
                out.append(ln)
                if guide_labels == 'axes' and lab:
                    ax.text(xv, 1.0, lab, transform=ax.get_xaxis_transform(), ha='center', va='bottom', clip_on=True, zorder=5)
        if yGuides:
            for g in yGuides:
                yv, lab, col = _unpack_guide(g)
                ln = ax.axhline(yv, ls='--', color=col, zorder=1, label=(lab or 'y-guide'))
                try: ln.set_gid('y-guideline')
                except Exception: pass
                out.append(ln)
                if guide_labels == 'axes' and lab:
                    ax.text(1.0, yv, lab, transform=ax.get_yaxis_transform(), ha='left', va='center', clip_on=True, zorder=5)
        return out

    guide_lines = _add_guides(ax_main)

    # -------------------- Y-hist a destra --------------------
    yhist_axes: List[Axes] = []
    if n_cols_histY > 0:
        col0 = 3  # 0:left, 1:main, 2:gap, 3:first yhist
        if yhist_overlay:
            axy = fig.add_subplot(gs[main_row, col0]); yhist_axes.append(axy)
        else:
            for j in range(n_cols_histY):
                axy = fig.add_subplot(gs[main_row, col0 + j]); yhist_axes.append(axy)
        for axy in yhist_axes:
            if grid_hist: axy.grid(True, which='major', alpha=0.25)

        def _hist_y(ax: Axes, dat: np.ndarray, color: str='0.6', alpha: float=0.55) -> None:
            ax.hist(dat, bins=y_bins_final, orientation='horizontal',
                    density=density, histtype='stepfilled',
                    color=color, alpha=alpha, zorder=1)

        _hist_y(yhist_axes[0], y_all, color='0.6', alpha=(0.55 if yhist_overlay else 0.45))
        if not yhist_overlay and len(yhist_axes) > 1:
            for k, h in enumerate((extraHistsY or [])[:len(yhist_axes)-1], start=1):
                dat = _finite(np.asarray(h.get('data', []))) if isinstance(h, dict) else _finite(np.asarray(h))
                if dat.size == 0: continue
                col = (h.get('color', '0.6') if isinstance(h, dict) else '0.6')
                a = float(h.get('alpha', 0.45)) if isinstance(h, dict) else 0.45
                yhist_axes[k].hist(dat, bins=y_bins_final, orientation='horizontal',
                                   density=density, histtype='stepfilled',
                                   color=col, alpha=a, zorder=1)

        # allinea range y con main
        y_min, y_max = ax_main.get_ylim()
        for axy in yhist_axes: axy.set_ylim(y_min, y_max)

        # formatter densità
        if y_log_density:
            for axy in yhist_axes:
                axy.set_xscale('log', base=10.0)
                axy.xaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
                axy.xaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))
        else:
            for axy in yhist_axes: _apply_scalar_formatter(axy, 'x')

        # μ/σ (con label/gid per styling)
        y_stats_lines: List[Line2D] = []
        if ('mu' in showYStats or 'sigma' in showYStats) and y_all.size > 0:
            mu = float(np.mean(y_all)); sig = float(np.std(y_all, ddof=0))
            axy = yhist_axes[0]
            ymu = axy.axhline(mu, color='0.1', ls='--', zorder=3, label='y-mu');  ymu.set_gid('y-mu');  y_stats_lines.append(ymu)
            if 'sigma' in showYStats and sig > 0:
                ylo = axy.axhline(mu - sig, color='0.35', ls='--', zorder=3, label='y-mu-sigma');    ylo.set_gid('y-mu-sigma-lo'); y_stats_lines.append(ylo)
                yhi = axy.axhline(mu + sig, color='0.35', ls='--', zorder=3, label='y-mu+sigma');     yhi.set_gid('y-mu-sigma-hi'); y_stats_lines.append(yhi)

        # policy tick long axis (y)
        _tp = (yhist_ticks or 'auto').lower()
        if _tp not in ('auto','right','none'): _tp = 'auto'
        if _tp == 'right':
            for axy in yhist_axes: _ticks_right(axy)
        elif _tp == 'none':
            for axy in yhist_axes: _hide_all_ticks(axy, 'y')
        else:
            for axy in yhist_axes: _hide_all_ticks(axy, 'y')

        # short axis vs CDF side
        short_opt = (yhist_xticks or 'top').lower()
        _cdf_side_y = (y_cdf_ticks or 'top').lower()
        if _cdf_side_y not in ('top','bottom','none'): _cdf_side_y = 'top'
        if _cdf_side_y != 'none' and short_opt == _cdf_side_y:
            short_opt = 'bottom' if short_opt == 'top' else 'top'
        for axy in yhist_axes:
            axy.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labeltop=False)
            (_ticks_bottom if short_opt=='bottom' else _ticks_top)(axy)

        # CDF (twiny) + line handle + tick locator default 0, 0.5, 1
        y_cdf_twins: List[Axes] = []
        y_cdf_lines: List[Line2D] = []
        if y_cdf:
            for axy in yhist_axes:
                twin = axy.twiny()
                y_cdf_twins.append(twin); axy._cdf_twin_x = twin
                twin.set_xscale('linear'); twin.set_yscale('linear')
                side = _cdf_side_y
                twin.tick_params(axis='x', which='both', top=False, bottom=False, labeltop=False, labelbottom=False)
                if side == 'top':
                    twin.xaxis.set_ticks_position('top'); twin.tick_params(axis='x', which='both', top=True,  labeltop=True)
                elif side == 'bottom':
                    twin.xaxis.set_ticks_position('bottom'); twin.tick_params(axis='x', which='both', bottom=True, labelbottom=True)

                counts, edges = np.histogram(y_all, bins=y_bins_final, density=True)
                incr = counts * np.diff(edges)
                cdf = np.concatenate([[0.0], np.cumsum(incr)])
                if cdf[-1] > 0: cdf = cdf / cdf[-1]
                cdf = np.clip(cdf, 0.0, 1.0)
                twin.set_xlim(0.0, 1.0); twin.set_ylim(axy.get_ylim())
                ln, = twin.plot(cdf, edges, color='0.25', zorder=4, label='y-cdf'); ln.set_gid('y-cdf')
                y_cdf_lines.append(ln)
                twin.xaxis.set_major_locator(FixedLocator([0.0, 0.5, 1.0]))

    # -------------------- X-hist (top/bottom) --------------------
    xhist_axes: List[Axes] = []
    x_cdf_twins: List[Axes] = []
    x_cdf_lines: List[Line2D] = []
    x_stats_lines: List[Line2D] = []
    if n_rows_histX > 0:
        row0 = (1 if (xhist_position or 'top').lower() == 'top' else (main_row + 2))
        if xhist_overlay:
            axx = fig.add_subplot(gs[row0, 1]); xhist_axes.append(axx)
        else:
            for j in range(n_rows_histX):
                axx = fig.add_subplot(gs[row0 + j, 1]); xhist_axes.append(axx)
        for axx in xhist_axes:
            if grid_hist: axx.grid(True, which='major', alpha=0.25)

        def _hist_x(ax: Axes, dat: np.ndarray, color: str='0.6', alpha: float=0.55) -> None:
            ax.hist(dat, bins=x_bins_final, orientation='vertical',
                    density=density, histtype='stepfilled',
                    color=color, alpha=alpha, zorder=1)

        _hist_x(xhist_axes[0], x_all, color='0.6', alpha=(0.55 if xhist_overlay else 0.45))
        if not xhist_overlay and len(xhist_axes) > 1:
            for k, h in enumerate((extraHistsX or [])[:len(xhist_axes)-1], start=1):
                dat = _finite(np.asarray(h.get('data', []))) if isinstance(h, dict) else _finite(np.asarray(h))
                if dat.size == 0: continue
                col = (h.get('color', '0.6') if isinstance(h, dict) else '0.6')
                a = float(h.get('alpha', 0.45)) if isinstance(h, dict) else 0.45
                xhist_axes[k].hist(dat, bins=x_bins_final, orientation='vertical',
                                   density=density, histtype='stepfilled',
                                   color=col, alpha=a, zorder=1)

        # allinea range x con main
        x_min, x_max = ax_main.get_xlim()
        for axx in xhist_axes: axx.set_xlim(x_min, x_max)

        # formatter densità
        if x_log_density:
            for axx in xhist_axes:
                axx.set_yscale('log', base=10.0)
                axx.yaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
                axx.yaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))
        else:
            for axx in xhist_axes: _apply_scalar_formatter(axx, 'y')

        # μ/σ
        if ('mu' in showXStats or 'sigma' in showXStats) and x_all.size > 0:
            mu = float(np.mean(x_all)); sig = float(np.std(x_all, ddof=0))
            axx = xhist_axes[0]
            xmu = axx.axvline(mu, color='0.1', ls='--', zorder=3, label='x-mu'); xmu.set_gid('x-mu'); x_stats_lines.append(xmu)
            if 'sigma' in showXStats and sig > 0:
                xlo = axx.axvline(mu - sig, color='0.35', ls='--', zorder=3, label='x-mu-sigma'); xlo.set_gid('x-mu-sigma-lo'); x_stats_lines.append(xlo)
                xhi = axx.axvline(mu + sig, color='0.35', ls='--', zorder=3, label='x-mu+sigma');  xhi.set_gid('x-mu-sigma-hi'); x_stats_lines.append(xhi)

        # long axis (x) tick policy
        _xp = (xhist_ticks or 'auto').lower()
        if _xp == 'bottom':
            for axx in xhist_axes: _ticks_bottom(axx)
        elif _xp == 'top':
            for axx in xhist_axes: _ticks_top(axx)
        elif _xp == 'none':
            for axx in xhist_axes: _hide_all_ticks(axx, 'x')
        else:
            for axx in xhist_axes: _hide_all_ticks(axx, 'x')

        # CDF twiny/ytwin (y) con handle e tick locator 0,0.5,1
        if x_cdf:
            for axx in xhist_axes:
                twin = axx.twinx()
                x_cdf_twins.append(twin); axx._cdf_twin_y = twin
                side = (x_cdf_ticks or 'right').lower()
                twin.set_xscale('linear'); twin.set_yscale('linear')
                if side == 'right':
                    twin.yaxis.set_ticks_position('right'); twin.tick_params(axis='y', which='both', right=True, labelright=True)
                    axx.yaxis.set_ticks_position('left');  axx.tick_params(axis='y', which='both', left=True, labelleft=True, right=False, labelright=False)
                elif side == 'left':
                    twin.yaxis.set_ticks_position('left');  twin.tick_params(axis='y', which='both', left=True, labelleft=True)
                    axx.yaxis.set_ticks_position('right'); axx.tick_params(axis='y', which='both', right=True, labelright=True, left=False, labelleft=False)

                counts, edges = np.histogram(x_all, bins=x_bins_final, density=True)
                incr = counts * np.diff(edges)
                cdf = np.concatenate([[0.0], np.cumsum(incr)])
                if cdf[-1] > 0: cdf = cdf / cdf[-1]
                cdf = np.clip(cdf, 0.0, 1.0)
                twin.set_ylim(0.0, 1.0); twin.set_xlim(axx.get_xlim())
                ln, = twin.plot(edges, cdf, color='0.25', zorder=4, label='x-cdf'); ln.set_gid('x-cdf')
                x_cdf_lines.append(ln)
                twin.yaxis.set_major_locator(FixedLocator([0.0, 0.5, 1.0]))

    # -------------------- legenda opzionale (colonna dedicata) --------------------
    if legend and legend_width_in > 0.0:
        leg_col = len(width_ratios) - 2   # penultima
        ax_leg: Axes = fig.add_subplot(gs[:, leg_col]); ax_leg.axis('off')
        handles: List[Any] = []; labels: List[str] = []
        for ln, lab in zip(lines_main, plotted_labels): handles.append(ln); labels.append(lab)
        if guide_labels == 'legend':
            if xGuides:
                for g in xGuides:
                    xv, lab, col = _unpack_guide(g)
                    if lab: handles.append(Line2D([0],[0], color=col, ls='--')); labels.append(str(lab))
            if yGuides:
                for g in yGuides:
                    yv, lab, col = _unpack_guide(g)
                    if lab: handles.append(Line2D([0],[0], color=col, ls='--')); labels.append(str(lab))
        if show_yhist:
            rect = Patch(facecolor='0.7', edgecolor='0.3', alpha=0.55, label='Y data')
            handles.append(rect); labels.append('Y data')
            handles.extend([Line2D([0],[0], color='0.1', ls='--'),
                            Line2D([0],[0], color='0.35', ls='--')])
            labels.extend([r'$\mu$', r'$\mu \pm \sigma$'])
        ax_leg.legend(handles, labels, loc='center left', frameon=False, borderaxespad=0.0)

    # -------------------- meta (hook per styling esterno) --------------------
    meta: Dict[str, Any] = {
        "figsize_in": (fig_w, fig_h),
        "ax_main": ax_main,
        "yhist_axes": yhist_axes,
        "xhist_axes": xhist_axes,
        "y_cdf_twins": [getattr(ax, "_cdf_twin_x", None) for ax in yhist_axes],
        "x_cdf_twins": [getattr(ax, "_cdf_twin_y", None) for ax in xhist_axes],
        "guide_lines": guide_lines,
        "n_curves_plotted": len(idx_list),
        "plotted_indices": idx_list,
        "y_bins": y_bins_final,
        "x_bins": x_bins_final,
    }
    # per convenienza: salva le linee CDF/stat in meta se esistono
    if 'y_cdf_lines' in locals(): meta["y_cdf_lines"] = y_cdf_lines
    if 'x_cdf_lines' in locals(): meta["x_cdf_lines"] = x_cdf_lines
    if 'y_stats_lines' in locals(): meta["y_stats_lines"] = y_stats_lines
    if 'x_stats_lines' in locals(): meta["x_stats_lines"] = x_stats_lines

    return fig, ax_main, meta
