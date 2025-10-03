
"""
utils_plot_v7.py
- Keep v6 features (no-clip canvas growth, right-strip geometry).
- Backward-compat: ensure_right_strip_no_clip(fig, layout=None).
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple, Optional, Sequence, Dict, Any, Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, MaxNLocator
import matplotlib.transforms as mtrans

from . import plot_cfg as cfg

@dataclass
class GridLayout:
    fig_w_in: float
    fig_h_in: float
    nrows: int
    ncols: int
    data_rects: List[List[float]]
    cbar_rect: Optional[List[float]] = None
    legend_rect: Optional[List[float]] = None

def _renderer(fig: plt.Figure):
    fig.canvas.draw()
    return fig.canvas.get_renderer()

def _preserve_panel_widths(fig: plt.Figure, new_fw_in: float):
    fw0, fh0 = fig.get_size_inches()
    if new_fw_in <= fw0 + 1e-10:
        return
    pre = {ax: ax.get_position().bounds for ax in fig.axes}
    widths_in = {ax: b[2] * fw0 for ax, b in pre.items()}
    fig.set_size_inches(new_fw_in, fh0, forward=True)
    fw1, _ = fig.get_size_inches()
    for ax, (L, B, W, H) in pre.items():
        ax.set_position([L, B, widths_in[ax] / fw1, H])
    fig.canvas.draw()

def _preserve_panel_heights(fig: plt.Figure, new_fh_in: float, align: str = "bottom"):
    fw0, fh0 = fig.get_size_inches()
    if new_fh_in <= fh0 + 1e-10:
        return
    pre = {ax: ax.get_position().bounds for ax in fig.axes}
    heights_in = {ax: b[3] * fh0 for ax, b in pre.items()}
    anchors = {ax: (b[1] if align=="bottom" else (b[1] + b[3])) for ax, b in pre.items()}
    fig.set_size_inches(fw0, new_fh_in, forward=True)
    _, fh1 = fig.get_size_inches()
    for ax, (L, B, W, H) in pre.items():
        h_in = heights_in[ax]; Hf = h_in / fh1
        y0f = anchors[ax] if align=="bottom" else (anchors[ax] - Hf)
        ax.set_position([L, y0f, W, Hf])
    fig.canvas.draw()

# -------- figure builders --------
def figure_single(data_height_in: Optional[float] = None):
    fig_w_in, fig_h_in = cfg.panel_box_figsize_single(height_in=data_height_in)
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))
    rect = cfg.axes_rect_from_cfg(height_in=data_height_in, fig_w_in=fig_w_in, fig_h_in=fig_h_in)
    ax = fig.add_axes(rect)
    layout = GridLayout(fig_w_in, fig_h_in, 1, 1, [rect])
    return fig, ax, layout

def figure_grid(nrows: int, ncols: int,
                data_w_in: Optional[float] = None,
                data_h_in: Optional[float] = None,
                reserve_cbar_right: bool = False,
                reserve_legend_right: bool = False,
                gaps: Tuple[float, float] = (cfg.GAPS.W_IN, cfg.GAPS.H_IN)):
    fig_w_in, fig_h_in = cfg.grid_figsize(nrows, ncols,
                                          data_w_in=data_w_in, data_h_in=data_h_in,
                                          gaps=gaps,
                                          reserve_cbar_right=reserve_cbar_right,
                                          reserve_legend_right=reserve_legend_right)
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))
    rects = cfg.grid_axes_rects(nrows, ncols,
                                fig_w_in=fig_w_in, fig_h_in=fig_h_in,
                                data_w_in=data_w_in, data_h_in=data_h_in,
                                gaps=gaps,
                                reserve_cbar_right=reserve_cbar_right,
                                reserve_legend_right=reserve_legend_right)
    axes = [fig.add_axes(r) for r in rects]
    layout = GridLayout(fig_w_in, fig_h_in, nrows, ncols, rects)

    if reserve_cbar_right:
        total_data_h = (data_h_in if data_h_in is not None else cfg.PANEL_DATA.HEIGHT_IN) * nrows \
                       + cfg.GAPS.H_IN * (nrows - 1)
        cbar_rect = cfg.right_cbar_rect(fig_w_in, fig_h_in, nrows, ncols,
                                        data_h_in=data_h_in, total_data_h_in=total_data_h)
        layout.cbar_rect = cbar_rect

    if reserve_legend_right:
        usable_h_in = fig_h_in - cfg.PANEL_MARGINS_IN.TOP - cfg.PANEL_MARGINS_IN.BOTTOM
        w = cfg.LEGEND_SLOT.WIDTH_IN
        pad_between = cfg.LEGEND_SLOT.PAD_IN
        x_right = fig_w_in - cfg.PANEL_MARGINS_IN.RIGHT
        shift = 0.0
        if reserve_cbar_right and layout.cbar_rect is not None:
            shift += cfg.CBAR.WIDTH_IN + cfg.CBAR.PAD_IN + pad_between
        x0_in = x_right - shift - w
        y0_in = cfg.PANEL_MARGINS_IN.BOTTOM
        layout.legend_rect = [x0_in/fig_w_in, y0_in/fig_h_in, w/fig_w_in, usable_h_in/fig_h_in]

    return fig, axes, layout

# -------- convenience --------
def pick_main_axes(axes, index: int = 0):
    if not axes:
        raise ValueError("axes is empty")
    return axes[index % len(axes)]

def get_legend_handles_labels(ax: mpl.axes.Axes, include: Optional[Callable[[Any, str], bool]] = None):
    handles, labels = ax.get_legend_handles_labels()
    if include is None:
        return handles, labels
    out_h, out_l = [], []
    for h, l in zip(handles, labels):
        try:
            if include(h, l):
                out_h.append(h); out_l.append(l)
        except Exception:
            out_h.append(h); out_l.append(l)
    return out_h, out_l

legend_filtered = get_legend_handles_labels

# -------- colorbar / legend --------
def add_colorbar_right(fig: plt.Figure,
                       mappable: mpl.cm.ScalarMappable,
                       layout: GridLayout,
                       orientation: str = "vertical",
                       **cbar_kw):
    if layout.cbar_rect is None:
        raise ValueError("No colorbar strip reserved. Call figure_grid(..., reserve_cbar_right=True) first.")
    ax_cbar = fig.add_axes(layout.cbar_rect)
    cb = fig.colorbar(mappable, cax=ax_cbar, orientation=orientation, **cbar_kw)
    return cb

def _first_mappable_from_axes(ax: mpl.axes.Axes):
    if ax.images:
        return ax.images[0]
    for coll in ax.collections:
        if hasattr(coll, "get_array"):
            return coll
    for coll in ax.collections:
        if coll.__class__.__name__ == "QuadMesh":
            return coll
    return None

def add_colorbar_right_auto(fig: plt.Figure,
                            ax: mpl.axes.Axes,
                            layout: GridLayout,
                            orientation: str = "vertical",
                            **cbar_kw):
    m = _first_mappable_from_axes(ax)
    if m is None:
        raise ValueError("No mappable found on the provided axes for automatic colorbar.")
    return add_colorbar_right(fig, m, layout, orientation=orientation, **cbar_kw)

def add_legend_outside_right(fig: plt.Figure,
                             ax_ref: mpl.axes.Axes,
                             layout: GridLayout,
                             handles: Optional[Sequence[Any]] = None,
                             labels: Optional[Sequence[str]] = None,
                             ncol: int = 1,
                             title: Optional[str] = None,
                             frameon: bool = True,
                             fontsize: Optional[float] = None,
                             **legend_kw):
    if layout.legend_rect is None:
        raise ValueError("No legend strip reserved. Call figure_grid(..., reserve_legend_right=True) first.")
    if handles is None or labels is None:
        handles, labels = ax_ref.get_legend_handles_labels()
    ax_leg = fig.add_axes(layout.legend_rect)
    ax_leg.set_facecolor('none')
    ax_leg.axis("off")
    lg = ax_leg.legend(handles, labels, ncol=ncol, title=title, frameon=frameon,
                       loc="center", fontsize=fontsize, **legend_kw)
    return lg

def attach_legend_right(fig: plt.Figure,
                        ax_ref: mpl.axes.Axes,
                        layout: GridLayout,
                        include: Optional[Callable[[Any, str], bool]] = None,
                        ncol: int = 1,
                        title: Optional[str] = None,
                        frameon: bool = True,
                        fontsize: Optional[float] = None,
                        **legend_kw):
    handles, labels = get_legend_handles_labels(ax_ref, include=include)
    return add_legend_outside_right(fig, ax_ref, layout, handles=handles, labels=labels,
                                    ncol=ncol, title=title, frameon=frameon, fontsize=fontsize, **legend_kw)

# -------- no-clip --------


def figure_single_fixed(data_w_in: float=3.20, data_h_in: float=2.20,
                        left_in: float=0.60, right_in: float=0.25,
                        bottom_in: float=0.55, top_in: float=0.18):
    """
    Crea fig+ax con area dati ESATTAMENTE (data_w_in x data_h_in) inch.
    I margini sono assoluti in inches. NO tight.
    """
    fig_w = data_w_in + left_in + right_in
    fig_h = data_h_in + bottom_in + top_in
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([
        left_in/fig_w,
        bottom_in/fig_h,
        data_w_in/fig_w,
        data_h_in/fig_h
    ])
    meta = SimpleNamespace(
        fig_w_in=fig_w, fig_h_in=fig_h,
        data_w_in=data_w_in, data_h_in=data_h_in
    )
    return fig, ax, meta

def _axes_margins_in(fig: mpl.figure.Figure, ax: mpl.axes.Axes):
    """Return (L,R,B,T,data_w,data_h) in inches from current figure/axes geometry."""
    W, H = fig.get_size_inches()
    pos  = ax.get_position()
    data_w = pos.width  * W
    data_h = pos.height * H
    L = pos.x0 * W
    B = pos.y0 * H
    R = (1.0 - (pos.x0 + pos.width))  * W
    T = (1.0 - (pos.y0 + pos.height)) * H
    return L, R, B, T, data_w, data_h

def auto_expand_margins_to_fit(fig: mpl.figure.Figure,
                               ax: mpl.axes.Axes,
                               min_left_in: float=0.45, min_right_in: float=0.32,
                               min_bottom_in: float=0.45, min_top_in: float=0.16,
                               pad_in: float=0.04,
                               max_iter: int=3,
                               tol_in: float=0.01) -> None:
    """
    Iteratively enlarge ONLY the margins (inches) needed to include all tick/labels,
    keeping the data area size constant. No tight layout.
    - min_*_in: hard lower bounds in inches
    - pad_in: safety padding added when overflow is detected
    - max_iter: at most this many geometry updates
    - tol_in: stop when all overflows < tol_in
    """
    for _ in range(max_iter):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        tb = ax.get_tightbbox(renderer)  # pixels
        bb = ax.bbox                      # pixels
        px2in = 1.0 / fig.dpi

        left_extra   = max(0.0, (bb.x0 - tb.x0) * px2in)
        right_extra  = max(0.0, (tb.x1 - bb.x1) * px2in)
        bottom_extra = max(0.0, (bb.y0 - tb.y0) * px2in)
        top_extra    = max(0.0, (tb.y1 - bb.y1) * px2in)

        if (left_extra  < tol_in and right_extra < tol_in and
            bottom_extra < tol_in and top_extra   < tol_in):
            break

        L, R, B, T, data_w, data_h = _axes_margins_in(fig, ax)

        new_L = max(L, min_left_in,  L + (left_extra  + pad_in if left_extra  >= tol_in else 0.0))
        new_R = max(R, min_right_in, R + (right_extra + pad_in if right_extra >= tol_in else 0.0))
        new_B = max(B, min_bottom_in,B + (bottom_extra+ pad_in if bottom_extra>= tol_in else 0.0))
        new_T = max(T, min_top_in,   T + (top_extra   + pad_in if top_extra   >= tol_in else 0.0))

        new_W = data_w + new_L + new_R
        new_H = data_h + new_B + new_T
        fig.set_size_inches(new_W, new_H, forward=True)

        ax.set_position([
            new_L / new_W,
            new_B / new_H,
            data_w / new_W,
            data_h / new_H
        ])

        # loop continues in case expanding one side causes different text wrapping elsewhere


def standardize_ticks(ax: mpl.axes.Axes,
                      xbins: int=5, ybins: int=5,
                      integer_x: bool=False, integer_y: bool=False,
                      prune: Optional[str]='both') -> None:
    """4–6 major tick/asse sono l’ottimo: qui default 5."""
    ax.xaxis.set_major_locator(MaxNLocator(nbins=xbins, integer=integer_x, prune=prune))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=ybins, integer=integer_y, prune=prune))
    ax.minorticks_off()

def align_twinx_linear(ax_left: mpl.axes.Axes, ax_right: mpl.axes.Axes,
                       scale: float, set_locator: bool=True) -> None:
    """Allinea l’asse destro: y_r = scale * y_l (limiti + ticks)."""
    yl = ax_left.get_ylim()
    ax_right.set_ylim(scale*yl[0], scale*yl[1])
    if set_locator:
        ticks_left = ax_left.get_yticks()
        ax_right.yaxis.set_major_locator(FixedLocator(scale * ticks_left))

def ensure_outer_text_no_clip(fig: plt.Figure, safety_pt: float = 2.0) -> None:
    rr = _renderer(fig)
    fbb = fig.bbox
    add_left_px = add_right_px = add_top_px = add_bot_px = 0.0
    texts_bb = []
    for ax in fig.axes:
        if ax.title and ax.title.get_text():
            texts_bb.append(ax.title.get_window_extent(renderer=rr))
        if ax.xaxis.label and ax.xaxis.label.get_text():
            texts_bb.append(ax.xaxis.label.get_window_extent(renderer=rr))
        if ax.yaxis.label and ax.yaxis.label.get_text():
            texts_bb.append(ax.yaxis.label.get_window_extent(renderer=rr))
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            if t.get_visible() and t.get_text():
                texts_bb.append(t.get_window_extent(renderer=rr))
    if not texts_bb:
        return
    allbb = mtrans.Bbox.union(texts_bb)
    margin = max(1.0, safety_pt)
    if allbb.x0 < fbb.x0 + margin: add_left_px  = (fbb.x0 + margin - allbb.x0)
    if allbb.x1 > fbb.x1 - margin: add_right_px = (allbb.x1 - (fbb.x1 - margin))
    if allbb.y0 < fbb.y0 + margin: add_bot_px   = (fbb.y0 + margin - allbb.y0)
    if allbb.y1 > fbb.y1 - margin: add_top_px   = (allbb.y1 - (fbb.y1 - margin))
    if max(add_left_px, add_right_px, add_top_px, add_bot_px) < 0.5:
        return
    fw, fh = fig.get_size_inches()
    if add_left_px > 0 or add_right_px > 0:
        _preserve_panel_widths(fig, fw + (add_left_px + add_right_px)/fig.dpi)
    if add_top_px > 0 or add_bot_px > 0:
        _preserve_panel_heights(fig, fh + (add_top_px + add_bot_px)/fig.dpi, align="bottom")
    fig.canvas.draw()

def ensure_right_strip_no_clip(fig: plt.Figure, layout: Optional[GridLayout]=None) -> None:
    """Accept optional layout for backward-compat; it is not used."""
    rr = _renderer(fig)
    fbb = fig.bbox
    px_right = 0.0
    for a in fig.axes:
        if a.has_data():
            continue
        try:
            L = a.get_legend()
            bb = (L.get_window_extent(renderer=rr) if L is not None else a.get_window_extent(renderer=rr))
            px_right = max(px_right, bb.x1)
        except Exception:
            pass
    if px_right > fbb.x1 - 1.0:
        extra_in = (px_right - fbb.x1)/fig.dpi + 0.02
        fw0, _ = fig.get_size_inches()
        _preserve_panel_widths(fig, fw0 + extra_in)
        fig.canvas.draw()

# -------- export --------
def export_figure(fig: plt.Figure,
                  filepath_no_ext: str,
                  formats: Sequence[str] = ("pdf", "png"),
                  dpi: int = cfg.EXPORT.DPI_RASTER,
                  metadata: Optional[Dict[str, str]] = None):
    ensure_outer_text_no_clip(fig, safety_pt=2.0)
    ensure_right_strip_no_clip(fig, None)
    written: List[str] = []
    md = metadata or {}
    for ext in formats:
        path = f"{filepath_no_ext}.{ext.lower()}"
        if ext.lower() in {"png", "tif", "tiff", "jpg", "jpeg"}:
            fig.savefig(path, dpi=dpi, metadata=md, bbox_inches=None)
        else:
            fig.savefig(path, metadata=md, bbox_inches=None)
        written.append(path)
    return written


def _axes_margins_in(fig: mpl.figure.Figure, ax: mpl.axes.Axes):
    W, H = fig.get_size_inches()
    pos  = ax.get_position()
    data_w = pos.width  * W
    data_h = pos.height * H
    L = pos.x0 * W
    B = pos.y0 * H
    R = (1.0 - (pos.x0 + pos.width))  * W
    T = (1.0 - (pos.y0 + pos.height)) * H
    return L, R, B, T, data_w, data_h

def _slug_filename(name: str) -> str:
    """Make a safe filename from the figure label/name."""
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in (name or ""))
    return safe.strip("_") or "figure"

def exportAllOpenFigures(folder: str, to_pdf: bool = False, dpi: int = 300, *, close: bool = True) -> List[Tuple[int, str]]:
    outdir = Path(folder); outdir.mkdir(parents=True, exist_ok=True)
    formats = ("png","pdf") if to_pdf else ("png",)
    saved: List[Tuple[int,str]] = []
    used = set()
    fig_nums = list(plt.get_fignums())  # snapshot, così non interferiamo con export che crea figure

    for num in fig_nums:
        fig = plt.figure(num)
        base = fig.get_label() or f"figure_{num}"
        name = _slug_filename(base)
        cand, k = name, 2
        while cand in used:
            cand = f"{name}_{k}"; k += 1
        used.add(cand)

        out_base = str(outdir / cand)
        export_figure_strict(fig, out_base, formats=formats, dpi=dpi)
        saved.append((num, out_base))
        if close:
            plt.close(fig)

    return saved

def expand_margins_for_axes(fig: mpl.figure.Figure,
                            axes: Sequence[mpl.axes.Axes],
                            min_left_in: float=0.35, min_right_in: float=0.25,
                            min_bottom_in: float=0.35, min_top_in: float=0.12,
                            pad_in: float=0.03,
                            max_iter: int=4,
                            tol_in: float=0.01) -> None:
    """
    Iteratively enlarge ONLY margins (in inches) so that the union of all texts
    (labels/ticks) from `axes` fits within the figure. The data area of `axes[0]`
    is preserved exactly; twin axes (twinx/twiny) are supported by passing both.
    No bbox_inches='tight' is used.
    """
    if not axes:
        return
    base_ax = axes[0]
    for _ in range(max_iter):
        fig.canvas.draw()
        rr = fig.canvas.get_renderer()
        # union of tight bboxes for all provided axes
        tight_bboxes = []
        for a in axes:
            tb = a.get_tightbbox(rr)
            if tb is not None:
                tight_bboxes.append(tb)
            # include legend of that axes, if any
            L = a.get_legend()
            if L is not None:
                try:
                    tight_bboxes.append(L.get_window_extent(rr))
                except Exception:
                    pass
        if not tight_bboxes:
            break
        union_tb = mtrans.Bbox.union(tight_bboxes)
        bb = base_ax.bbox  # axes frame bbox
        px2in = 1.0 / fig.dpi

        left_extra   = max(0.0, (bb.x0 - union_tb.x0) * px2in)
        right_extra  = max(0.0, (union_tb.x1 - bb.x1) * px2in)
        bottom_extra = max(0.0, (bb.y0 - union_tb.y0) * px2in)
        top_extra    = max(0.0, (union_tb.y1 - bb.y1) * px2in)

        if (left_extra  < tol_in and right_extra < tol_in and
            bottom_extra < tol_in and top_extra   < tol_in):
            break

        L, R, B, T, data_w, data_h = _axes_margins_in(fig, base_ax)
        new_L = max(L, min_left_in,  L + (left_extra  + pad_in if left_extra  >= tol_in else 0.0))
        new_R = max(R, min_right_in, R + (right_extra + pad_in if right_extra >= tol_in else 0.0))
        new_B = max(B, min_bottom_in,B + (bottom_extra+ pad_in if bottom_extra>= tol_in else 0.0))
        new_T = max(T, min_top_in,   T + (top_extra   + pad_in if top_extra   >= tol_in else 0.0))

        new_W = data_w + new_L + new_R
        new_H = data_h + new_B + new_T
        fig.set_size_inches(new_W, new_H, forward=True)

        base_ax.set_position([
            new_L / new_W,
            new_B / new_H,
            data_w / new_W,
            data_h / new_H
        ])
        # twins share position with base_ax automatically

        # loop again in case new layout reveals other overflows


def _axes_margins_in(fig: mpl.figure.Figure, ax: mpl.axes.Axes):
    W, H = fig.get_size_inches()
    pos  = ax.get_position()
    data_w = pos.width  * W
    data_h = pos.height * H
    L = pos.x0 * W
    B = pos.y0 * H
    R = (1.0 - (pos.x0 + pos.width))  * W
    T = (1.0 - (pos.y0 + pos.height)) * H
    return L, R, B, T, data_w, data_h

def layout_fit_text(fig: mpl.figure.Figure,
                    axes: Sequence[mpl.axes.Axes],
                    base_ax: Optional[mpl.axes.Axes] = None,
                    min_left_in: float=0.35, min_right_in: float=0.28,
                    min_bottom_in: float=0.35, min_top_in: float=0.12,
                    pad_in: float=0.02,
                    max_iter: int=3,
                    tol_in: float=0.01) -> None:
    """
    Deterministic layout fitter:
    - Keeps base_ax data area in inches constant.
    - Computes union tightbbox of all provided axes (labels/ticks/legends).
    - Sets figure size and base_ax position to the minimal margins (>=min_*_in) + pad.
    - Iterates a few times to converge.
    """
    if not axes:
        return
    if base_ax is None:
        base_ax = axes[0]

    for _ in range(max_iter):
        fig.canvas.draw()
        rr = fig.canvas.get_renderer()
        # Union of all text bboxes
        tbb = []
        for a in axes:
            tb = a.get_tightbbox(rr)
            if tb is not None:
                tbb.append(tb)
            L = a.get_legend()
            if L is not None:
                try: tbb.append(L.get_window_extent(rr))
                except Exception: pass
        if not tbb:
            break
        union_tb = mtrans.Bbox.union(tbb)
        bb = base_ax.bbox  # axes frame
        px2in = 1.0 / fig.dpi

        # Overflow (inches) relative to axes frame
        left_need   = max(0.0, (bb.x0 - union_tb.x0) * px2in)
        right_need  = max(0.0, (union_tb.x1 - bb.x1) * px2in)
        bottom_need = max(0.0, (bb.y0 - union_tb.y0) * px2in)
        top_need    = max(0.0, (union_tb.y1 - bb.y1) * px2in)

        L0, R0, B0, T0, data_w, data_h = _axes_margins_in(fig, base_ax)

        # New margins: at least min_*, and at least current, and cover needs + pad
        L = max(L0, min_left_in,  left_need  + pad_in if left_need  >= tol_in else L0)
        R = max(R0, min_right_in, right_need + pad_in if right_need >= tol_in else R0)
        B = max(B0, min_bottom_in,bottom_need+ pad_in if bottom_need>= tol_in else B0)
        T = max(T0, min_top_in,   top_need   + pad_in if top_need   >= tol_in else T0)

        # Update figure size and base position to preserve data area
        new_W = data_w + L + R
        new_H = data_h + B + T
        fig.set_size_inches(new_W, new_H, forward=True)
        base_ax.set_position([L/new_W, B/new_H, data_w/new_W, data_h/new_H])
        # Make all companion axes share the same rect (twinx/twiny)
        rect = base_ax.get_position().bounds
        for a in axes:
            if a is base_ax: continue
            a.set_position(rect)

        # Re-evaluate; stop if no significant change needed
        fig.canvas.draw()
        rr = fig.canvas.get_renderer()
        union_tb2 = mtrans.Bbox.union([a.get_tightbbox(rr) for a in axes if a.get_tightbbox(rr) is not None])
        bb2 = base_ax.bbox
        left_need2   = max(0.0, (bb2.x0 - union_tb2.x0) * px2in)
        right_need2  = max(0.0, (union_tb2.x1 - bb2.x1) * px2in)
        bottom_need2 = max(0.0, (bb2.y0 - union_tb2.y0) * px2in)
        top_need2    = max(0.0, (union_tb2.y1 - bb2.y1) * px2in)
        if max(left_need2, right_need2, bottom_need2, top_need2) < tol_in:
            break

# Wrapper exporter that can skip ensure_* (to avoid double-expansion)
def export_figure_strict(fig: mpl.figure.Figure,
                         filepath_no_ext: str,
                         formats: Sequence[str] = ("pdf","png"),
                         dpi: int = 300,
                         metadata: Optional[Dict[str,str]] = None):
    md = metadata or {}
    written = []
    for ext in formats:
        path = f"{filepath_no_ext}.{ext.lower()}"
        if ext.lower() in {"png","tif","tiff","jpg","jpeg"}:
            fig.savefig(path, dpi=dpi, metadata=md, bbox_inches=None)
        else:
            fig.savefig(path, metadata=md, bbox_inches=None)
        written.append(path)
    return written

def align_twinx_symlog(ax, axr, scale: float=-1.0, base: float=10.0, linthresh: float=1e-3, linscale: float=1.0):
    """
    Align right y-axis to left y-axis for SYMLOG scaling:
      y_right = scale * y_left  (element-wise for limits).
    Assumes left axis scale is already set ('symlog'). Sets the same for axr.
    """
    import matplotlib.pyplot as plt  # noqa
    if ax.get_yscale() != "symlog":
        raise ValueError("Left axis is not 'symlog'; call ax.set_yscale('symlog', ...) first.")
    axr.set_yscale('symlog', linthresh=linthresh, linscale=linscale, base=base)
    yl = ax.get_ylim()
    axr.set_ylim(scale*yl[0], scale*yl[1])
    return axr