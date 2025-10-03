
"""
report_config_v4.py
Diagnostics and QA reports for Matplotlib figures and rcParams.
- JSON writing now converts NumPy scalars/bools to native Python types.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, Sequence, List
import json
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from . import plot_cfg as cfg

# ----------------------------- helpers (bbox in inches) -----------------------------

def _renderer(fig: plt.Figure):
    fig.canvas.draw()
    return fig.canvas.get_renderer()

def _ax_bbox_in(fig: plt.Figure, ax: mpl.axes.Axes) -> Tuple[float,float,float,float]:
    rr = _renderer(fig)
    bb = ax.get_window_extent(renderer=rr)
    dpi = fig.dpi
    return (bb.x0/dpi, bb.y0/dpi, bb.width/dpi, bb.height/dpi)

def _data_bbox_in(fig: plt.Figure, ax: mpl.axes.Axes) -> Tuple[float,float,float,float]:
    rr = _renderer(fig)
    patch = getattr(ax, "patch", None)
    if patch is None:
        return _ax_bbox_in(fig, ax)
    bb = patch.get_window_extent(renderer=rr)
    dpi = fig.dpi
    return (bb.x0/dpi, bb.y0/dpi, bb.width/dpi, bb.height/dpi)

def _min_ticklabel_size(ax: mpl.axes.Axes) -> float:
    sizes = []
    for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        try:
            sizes.append(float(t.get_size()))
        except Exception:
            pass
    return min(sizes) if sizes else float('inf')

def _min_axislabel_size(ax: mpl.axes.Axes) -> float:
    sizes = []
    if ax.get_xlabel():
        sizes.append(float(ax.xaxis.get_label().get_size()))
    if ax.get_ylabel():
        sizes.append(float(ax.yaxis.get_label().get_size()))
    return min(sizes) if sizes else float('inf')

def _all_linewidths(fig: plt.Figure) -> List[float]:
    lws: List[float] = []
    for ax in fig.axes:
        for sp in ax.spines.values():
            try:
                lws.append(float(sp.get_linewidth()))
            except Exception:
                pass
        for line in ax.lines:
            try:
                lws.append(float(line.get_linewidth()))
            except Exception:
                pass
        for coll in ax.collections:
            try:
                lw = getattr(coll, "get_linewidth", None) or getattr(coll, "get_linewidths", None)
                if lw is None:
                    continue
                val = lw()
                if isinstance(val, (list, tuple, np.ndarray)):
                    lws.extend([float(v) for v in val if v is not None])
                else:
                    lws.append(float(val))
            except Exception:
                pass
    return lws

# ----------------------------- rc snapshot -----------------------------

_RCKEYS = ["text.usetex", "figure.autolayout", "font.size", "axes.linewidth",
           "lines.linewidth", "xtick.major.size", "ytick.major.size",
           "pdf.fonttype", "ps.fonttype", "svg.fonttype"]

def _py(v):
    """Best-effort convert to plain Python types for JSON safety."""
    if isinstance(v, (np.generic, )):
        return v.item()
    return v

def report_rc(details: bool=True, printout: bool=True) -> Dict[str, Any]:
    out = {k: _py(mpl.rcParams.get(k)) for k in _RCKEYS}
    if details:
        try:
            import MyBasePlots.FigCore.utils_style as _ustyle  # within package
            out["latex_available_cache"] = _py(getattr(_ustyle, "_LATEX_AVAILABLE", None))
        except Exception:
            out["latex_available_cache"] = None
        out["MYBP_DISABLE_LATEX"] = os.environ.get("MYBP_DISABLE_LATEX", "0")
    if printout:
        print("[rc]", out)
    return out

# ----------------------------- figure QA -----------------------------

def report(fig: plt.Figure, *, check_panel_size: bool=True, tol_in: float=0.02) -> Dict[str, Any]:
    res: Dict[str, Any] = {"Figure": {"width_in": float(fig.get_size_inches()[0]),
                                      "height_in": float(fig.get_size_inches()[1]),
                                      "dpi": float(fig.dpi)},
                           "rc": report_rc(details=True, printout=False),
                           "axes": []}

    font_min_target = float(getattr(cfg, "FONT_MIN_PRINT_PT", 6.0))
    line_min = float(getattr(cfg, "LINE_MIN_PT", 0.5))
    tick_len_min = float(getattr(cfg, "TICK_MAJOR_LEN_PT", 2.0))

    data_w_target = float(getattr(cfg.PANEL_DATA, "WIDTH_IN", 0.0))
    data_h_target = float(getattr(cfg.PANEL_DATA, "HEIGHT_IN", 0.0))

    for ax in fig.axes:
        ax_min_tick = float(_min_ticklabel_size(ax))
        ax_min_label = float(_min_axislabel_size(ax))
        x,y,w,h = _data_bbox_in(fig, ax)
        res["axes"].append({
            "min_tick_pt": ax_min_tick,
            "min_axislabel_pt": ax_min_label,
            "data_box_in": {"w": float(w), "h": float(h)},
        })

    lws = _all_linewidths(fig)
    res["global"] = {
        "min_linewidth_pt": (float(min(lws)) if lws else float('inf')),
        "xtick_major_size_pt": float(mpl.rcParams.get("xtick.major.size")),
        "ytick_major_size_pt": float(mpl.rcParams.get("ytick.major.size")),
    }

    ok_fonts = all(float(min(d["min_tick_pt"], d["min_axislabel_pt"])) >= font_min_target for d in res["axes"]) if res["axes"] else True
    ok_lines = float(res["global"]["min_linewidth_pt"]) >= line_min
    ok_ticks = (float(res["global"]["xtick_major_size_pt"]) >= tick_len_min and
                float(res["global"]["ytick_major_size_pt"]) >= tick_len_min)

    res["checks"] = {
        "font_pt_min_ok": bool(ok_fonts),
        "line_width_min_ok": bool(ok_lines),
        "tick_length_min_ok": bool(ok_ticks),
    }

    if check_panel_size and res["axes"]:
        sizes_ok = True
        sizes_detail: List[Dict[str, Any]] = []
        for d in res["axes"]:
            w, h = float(d["data_box_in"]["w"]), float(d["data_box_in"]["h"])
            dw = abs(w - data_w_target); dh = abs(h - data_h_target)
            ok = (dw <= tol_in) and (dh <= tol_in)
            sizes_ok = sizes_ok and ok
            sizes_detail.append({"w_in": w, "h_in": h, "dw_in": float(dw), "dh_in": float(dh), "ok": bool(ok)})
        res["checks"]["panel_size_ok"] = bool(sizes_ok)
        res["panel_size_details"] = sizes_detail

    return res

# ----------------------------- io -----------------------------

def _json_default(o):
    if isinstance(o, (np.generic, )):
        return o.item()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

def write_json(rep: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, default=_json_default)
