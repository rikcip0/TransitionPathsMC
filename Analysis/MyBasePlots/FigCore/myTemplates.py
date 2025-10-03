
"""
myTemplates_v3.py
Template factories that orchestrate plot_cfg, utils_plot, utils_style, and myEncodings.
- Fixed data/panel geometry.
- Optional right strip for legend/colorbar without resizing data panels.
- Minimal finalize/export helpers.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, List, Optional, Sequence, Union

import matplotlib as mpl
import matplotlib.pyplot as plt

from . import plot_cfg as cfg
from . import utils_plot as uplot
from . import utils_style as ustyle
# myEncodings can be used by calling code; not required here.

# ------------------------------ metadata ------------------------------

def _init_meta(fig: plt.Figure,
               axes: Sequence[mpl.axes.Axes],
               layout: uplot.GridLayout) -> Dict[str, Any]:
    """Create a metadata dict describing roles and layout."""
    meta: Dict[str, Any] = {
        "fig": fig,
        "axes": list(axes),
        "layout": layout,
        "roles": {
            "panels": list(axes),
            "legend_ax": None,
            "cbar_ax": None,
            "legend": None,
            "colorbar": None,
            "title": None,
        },
    }
    return meta

def main_ax(meta: Dict[str, Any], index: int = 0) -> mpl.axes.Axes:
    """Return a reference panel axes (default: first)."""
    return meta["axes"][index % len(meta["axes"])]

# ------------------------------ factories ------------------------------

def new_single_panel(data_height_in: Optional[float] = None) -> Tuple[plt.Figure, mpl.axes.Axes, Dict[str, Any]]:
    """Single panel figure using cfg geometry."""
    fig, ax, layout = uplot.figure_single(data_height_in=data_height_in)
    meta = _init_meta(fig, [ax], layout)
    return fig, ax, meta

def new_two_panels_h(data_h_in: Optional[float] = None,
                     reserve_cbar_right: bool = False,
                     reserve_legend_right: bool = False,
                     gaps: Tuple[float, float] = (cfg.GAPS.W_IN, cfg.GAPS.H_IN)
                     ) -> Tuple[plt.Figure, List[mpl.axes.Axes], Dict[str, Any]]:
    """Two panels in a row; optional right strip for colorbar and/or legend."""
    fig, axes, layout = uplot.figure_grid(1, 2,
                                          data_h_in=data_h_in,
                                          reserve_cbar_right=reserve_cbar_right,
                                          reserve_legend_right=reserve_legend_right,
                                          gaps=gaps)
    meta = _init_meta(fig, axes, layout)
    return fig, axes, meta

def new_grid(nrows: int, ncols: int,
             data_w_in: Optional[float] = None,
             data_h_in: Optional[float] = None,
             reserve_cbar_right: bool = False,
             reserve_legend_right: bool = False,
             gaps: Tuple[float, float] = (cfg.GAPS.W_IN, cfg.GAPS.H_IN)
             ) -> Tuple[plt.Figure, List[mpl.axes.Axes], Dict[str, Any]]:
    """General grid template."""
    fig, axes, layout = uplot.figure_grid(nrows, ncols,
                                          data_w_in=data_w_in,
                                          data_h_in=data_h_in,
                                          reserve_cbar_right=reserve_cbar_right,
                                          reserve_legend_right=reserve_legend_right,
                                          gaps=gaps)
    meta = _init_meta(fig, axes, layout)
    return fig, axes, meta

# ------------------------------ legend / colorbar ------------------------------

def attach_colorbar_right(fig: plt.Figure,
                          target: Union[mpl.cm.ScalarMappable, mpl.axes.Axes],
                          meta: Dict[str, Any],
                          orientation: str = "vertical",
                          **cbar_kw) -> mpl.colorbar.Colorbar:
    """
    Add a colorbar in the reserved right strip.
    'target' can be a mappable or an Axes; if Axes, the mappable is inferred.
    """
    if isinstance(target, mpl.axes.Axes):
        cb = uplot.add_colorbar_right_auto(fig, target, meta["layout"], orientation=orientation, **cbar_kw)
    else:
        cb = uplot.add_colorbar_right(fig, target, meta["layout"], orientation=orientation, **cbar_kw)
    meta["roles"]["colorbar"] = cb
    # store cbar axes for metadata export if needed
    try:
        meta["roles"]["cbar_ax"] = cb.ax
    except Exception:
        pass
    return cb

def attach_legend_right(fig: plt.Figure,
                        ax_ref: mpl.axes.Axes,
                        meta: Dict[str, Any],
                        include=None,
                        ncol: int = 1,
                        title: Optional[str] = None,
                        frameon: bool = True,
                        fontsize: Optional[float] = None,
                        **legend_kw) -> mpl.legend.Legend:
    """Filter and place a legend in the right strip."""
    lg = uplot.attach_legend_right(fig, ax_ref, meta["layout"],
                                   include=include, ncol=ncol, title=title,
                                   frameon=frameon, fontsize=fontsize, **legend_kw)
    meta["roles"]["legend"] = lg
    # legend lives on its own Axes we created; save it
    try:
        meta["roles"]["legend_ax"] = lg.axes
    except Exception:
        pass
    return lg

# ------------------------------ labels / finalize / export ------------------------------

def apply_panel_labels(meta: Dict[str, Any], labels: Optional[Sequence[str]] = None) -> None:
    """Apply panel labels via utils_style; enforces minimum readable size."""
    ustyle.apply_panel_labels(meta["fig"], meta["axes"], labels=labels)

def finalize(meta: Dict[str, Any]) -> None:
    """Placeholder for future validation hooks (font/linewidth checks, overlap warnings)."""
    # Examples for future: font size >= cfg.FONT_MIN_PRINT_PT; linewidth >= cfg.LINE_MIN_PT
    return


def finalize_and_export(meta: Dict[str, Any],
                        filepath_no_ext: str,
                        formats: Sequence[str] = ("pdf","png"),
                        dpi: Optional[int] = None,
                        metadata: Optional[Dict[str,str]] = None) -> List[str]:
    """Finalize then save, using deterministic layout fitting (no tight, no double expand)."""
    fig: plt.Figure = meta["fig"]
    axes = meta.get("axes", list(fig.axes))
    base_ax = axes[0] if axes else fig.axes[0]

    # Fit margins to include all texts from all axes (incl. twin/legend/cbar)
    uplot.layout_fit_text(fig, axes, base_ax=base_ax,
                          min_left_in=0.35, min_right_in=0.28,
                          min_bottom_in=0.35, min_top_in=0.12,
                          pad_in=0.02, max_iter=3, tol_in=0.01)

    dpi_eff = dpi if dpi is not None else 300
    return uplot.export_figure_strict(fig, filepath_no_ext, formats=formats, dpi=dpi_eff, metadata=metadata or {})

