"""
utils_plot_v6.py
================
Back-compat + strumenti *gold* per multipannello, SENZA toccare la geometria
(che resta in cfg/template). Qui misuri/decidi e, se necessario, applichi
policy di etichettatura "outer-only".

Novità vs v5:
- recommend_vgap_between(ax_top, ax_bottom, min_gap_in=None)  -> gap verticale consigliato [in]
- outerize_two_columns(ax_left, ax_right, right_ticks_out=True)
- outerize_two_rows(ax_top, ax_bottom, top_ticks_out=True)
- outerize_grid(axes, nrows, ncols)  -> mostra label solo ai bordi esterni
- standard helpers restano: export_figure, export_all_open_figures, standardize_ticks,
  place_legend_outside, axes_bbox_inches, data_bbox_inches, recommend_gap_between
"""
from __future__ import annotations
from pathlib import Path
from typing import Sequence, Tuple
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox

# cfg opzionale (per default sensati)
try:
    import plot_cfg as cfg
except Exception:
    cfg = None

# ---------- Export ----------
def _ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def export_figure(fig, basename: str, outdir, to_pdf: bool = False, dpi: int = 300):
    outdir = Path(outdir); _ensure_dir(outdir)
    ext = "pdf" if to_pdf else "png"
    fname = outdir / f"{basename}.{ext}"
    fig.savefig(fname, dpi=dpi)
    return str(fname)

def export_all_open_figures(outdir, to_pdf: bool = False, dpi: int = 300):
    outdir = Path(outdir); _ensure_dir(outdir)
    paths = []
    for num in plt.get_fignums():
        fig = plt.figure(num)
        name = fig.get_label() or f"Figure_{num}"
        paths.append(export_figure(fig, name, outdir, to_pdf=to_pdf, dpi=dpi))
    plt.close("all")
    return paths

# ---------- Legend Outside ----------
def place_legend_outside(fig, ax, right_frac: float | None = None,
                         pad_inches: float | None = None, loc: str = "upper left"):
    handles, labels = ax.get_legend_handles_labels()
    if not handles: return None
    if right_frac is None: right_frac = ax.get_position().x1
    if pad_inches is None:
        pad_inches = getattr(getattr(cfg, "LEGEND", object()), "PAD_W", 0.12)
    fig_w = fig.get_size_inches()[0]
    x = right_frac + pad_inches/fig_w
    return fig.legend(handles, labels, loc=loc, bbox_to_anchor=(x, 1.0), frameon=False)

# ---------- Ticks ----------
def standardize_ticks(ax, xbins=None, ybins=None, prune=None, minor=None):
    if cfg is not None:
        if xbins is None: xbins = getattr(getattr(cfg, "TICKS", object()), "X_MAJ", 5)
        if ybins is None: ybins = getattr(getattr(cfg, "TICKS", object()), "Y_MAJ", 5)
        if prune is None: prune = getattr(getattr(cfg, "TICKS", object()), "PRUNE", "both")
        if minor is None: minor = getattr(getattr(cfg, "TICKS", object()), "MINOR", False)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=xbins or 5, prune=prune))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=ybins or 5, prune=prune))
    ax.minorticks_on() if (minor if minor is not None else False) else ax.minorticks_off()

# ---------- Misure pannello ----------
def axes_bbox_inches(fig, ax):
    l, b, w, h = ax.get_position().bounds
    fw, fh = fig.get_size_inches()
    return (w*fw, h*fh)

def data_bbox_inches(fig, ax):
    fig.canvas.draw()
    bb = ax.patch.get_window_extent(renderer=fig.canvas.get_renderer())
    return (bb.width/fig.dpi, bb.height/fig.dpi)

# ---------- Ticklabel gutters (robusti) ----------
def _labels_bbox(ax, which: str):
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    if which in ("left", "right"):
        labs = ax.yaxis.get_ticklabels(which="both")
    elif which in ("top", "bottom"):
        labs = ax.xaxis.get_ticklabels(which="both")
    else:
        raise ValueError("which must be 'left'|'right'|'top'|'bottom'")
    labs = [lab for lab in labs if lab.get_text()]
    if not labs: return None
    bboxes = [lab.get_window_extent(renderer=rend) for lab in labs]
    return Bbox.union(bboxes)

def ticklabel_gutter_in(ax, side: str) -> float:
    bb = _labels_bbox(ax, side)
    if bb is None: return 0.0
    fig = ax.figure
    return (bb.width/fig.dpi) if side in ("left","right") else (bb.height/fig.dpi)

def recommend_gap_between(ax_left, ax_right, min_gap_in: float | None = None) -> float:
    if min_gap_in is None:
        min_gap_in = getattr(getattr(cfg, "GAPS", object()), "GAP_W", 0.12)
    g_left  = ticklabel_gutter_in(ax_left,  "right")
    g_right = ticklabel_gutter_in(ax_right, "left")
    extra = 2.0/72.0  # ~2pt
    return max(min_gap_in, g_left + g_right + extra)

def recommend_vgap_between(ax_top, ax_bottom, min_gap_in: float | None = None) -> float:
    if min_gap_in is None:
        min_gap_in = getattr(getattr(cfg, "GAPS", object()), "GAP_H", 0.10 if cfg is None else cfg.GAPS.GAP_H)
    g_top    = ticklabel_gutter_in(ax_top,    "bottom")
    g_bottom = ticklabel_gutter_in(ax_bottom, "top")
    extra = 2.0/72.0
    return max(min_gap_in, g_top + g_bottom + extra)

# ---------- Policy "outer labels" ----------
def outerize_two_columns(ax_left, ax_right, *, right_ticks_out: bool = True):
    """Mostra solo le etichette Y a sinistra sul pannello sinistro e a destra sul pannello destro."""
    ax_left.tick_params(axis='y', labelright=False)
    ax_right.tick_params(axis='y', labelleft=False)
    if right_ticks_out:
        ax_right.yaxis.tick_right()
        ax_right.yaxis.set_label_position('right')

def outerize_two_rows(ax_top, ax_bottom, *, top_ticks_out: bool = True):
    """Mostra solo etichette X in alto sul pannello alto e in basso sul pannello basso."""
    ax_top.tick_params(axis='x', labelbottom=False)
    ax_bottom.tick_params(axis='x', labeltop=False)
    if top_ticks_out:
        ax_top.xaxis.tick_top()
        ax_top.xaxis.set_label_position('top')

def outerize_grid(axes: Sequence, nrows: int, ncols: int):
    """Mostra ticklabels solo ai bordi esterni di una griglia nrows×ncols (sequenza ravel() di axes)."""
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r*ncols + c]
            left  = (c == 0)
            right = (c == ncols-1)
            bottom = (r == nrows-1)
            top = (r == 0)
            ax.tick_params(labelleft=left, labelright=right, labelbottom=bottom, labeltop=top)
            if right: ax.yaxis.tick_right(); ax.yaxis.set_label_position('right')
            if top:   ax.xaxis.tick_top();   ax.xaxis.set_label_position('top')
