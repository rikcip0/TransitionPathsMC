
"""
figure_metadata_v3.py
Metadata utilities for roles (panel, legend, colorbar, strip) and crop/export helpers.
- Measurements in inches.
- No style or layout mutations.
- Backward-compatible with previous API (detect/gather/attach/write).

Public API
----------
detect_roles(fig) -> dict[str, list[Axes]]
gather_metadata(fig, *, roles=None) -> dict
attach_roles_to_meta(meta, roles) -> dict
write_json(meta, path) -> None

Additions
---------
register_role(ax, role: str) -> None
roles_from_meta(meta, fig=None) -> dict[str, list[Axes]]
bbox_union_in(fig, axes: list[Axes]) -> (x,y,w,h)
crop_bbox_in(fig, bbox_in: tuple, pad_in: float=0.0) -> matplotlib.transforms.Bbox
export_cropped(fig, path: str, bbox_in: tuple, *, dpi=None, metadata=None) -> None
export_roles(meta, roles: list[str], path_no_ext: str, *, pad_in=0.0, formats=('pdf','png'), dpi=None, metadata=None) -> list[str]
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional, Sequence
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

# ---------- helpers (bbox in inches) ----------

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

# ---------- roles (tagging) ----------

def register_role(ax: mpl.axes.Axes, role: str) -> None:
    """Attach a simple attribute tag on an axes to help role detection."""
    role = (role or "").lower()
    if role == "panel":
        setattr(ax, "_is_panel", True)
    elif role == "legend":
        setattr(ax, "_is_legend", True)
    elif role == "colorbar":
        setattr(ax, "_is_colorbar", True)
    elif role == "strip":
        setattr(ax, "_is_strip", True)

# ---------- roles detection ----------

def detect_roles(fig: plt.Figure) -> Dict[str, List[plt.Axes]]:
    """Heuristic + tag-based role detection."""
    roles = {"panel": [], "colorbar": [], "legend": [], "strip": []}

    # Prefer user tags
    panels = [ax for ax in fig.axes if getattr(ax, "_is_panel", False)]
    if not panels:
        panels = [ax for ax in fig.axes if ax.has_data()]
    # sort by x0
    fig.canvas.draw()
    panels = sorted(panels, key=lambda a: a.get_position().x0)
    roles["panel"] = panels

    # Colorbars: explicit tag or mpl Colorbar on ax
    for ax in fig.axes:
        if getattr(ax, "_is_colorbar", False):
            roles["colorbar"].append(ax)
            continue
        cb = getattr(ax, "colorbar", None)
        if cb is not None or ax.get_label().lower().startswith("colorbar"):
            roles["colorbar"].append(ax)

    # Legends: explicit tag or axes with a Legend and no data
    for ax in fig.axes:
        if getattr(ax, "_is_legend", False):
            roles["legend"].append(ax)
            continue
        if ax.has_data():
            continue
        if ax.get_legend() is not None:
            roles["legend"].append(ax)

    # Strip: explicit tag or any non-data axes right of rightmost panel
    if panels:
        rightmost = max(panels, key=lambda a: a.get_position().x1)
        x_anchor = rightmost.get_position().x1
        strip_axes = []
        for ax in fig.axes:
            if getattr(ax, "_is_strip", False):
                strip_axes.append(ax); continue
            if (not ax.has_data()) and (ax.get_position().x0 >= x_anchor):
                strip_axes.append(ax)
        strip_axes = sorted(strip_axes, key=lambda a: a.get_position().x0)
        roles["strip"] = strip_axes

    return roles

# ---------- metadata assembly ----------

def gather_metadata(fig: plt.Figure, *, roles: Optional[Dict[str, List[plt.Axes]]] = None) -> Dict[str, Any]:
    """Collect metadata in inches + role indices."""
    if roles is None:
        roles = detect_roles(fig)

    fw, fh = fig.get_size_inches()
    dpi = fig.dpi

    ax_meta = []
    for i, ax in enumerate(fig.axes):
        x, y, w, h = _ax_bbox_in(fig, ax)
        dx, dy, dw, dh = _data_bbox_in(fig, ax)
        ax_meta.append({
            "index": i,
            "is_panel": bool(getattr(ax, "_is_panel", False)),
            "bbox_axes_in": {"x": x, "y": y, "w": w, "h": h},
            "bbox_data_in": {"x": dx, "y": dy, "w": dw, "h": dh},
            "tick_params": {
                "xmaj": len(ax.get_xticklabels()), "ymaj": len(ax.get_yticklabels())
            },
            "has_data": bool(ax.has_data()),
        })

    meta = {
        "Figure": {"width_in": fw, "height_in": fh, "dpi": dpi},
        "Matplotlib": {"version": mpl.__version__},
        "Axes": ax_meta,
        "Roles": {k: [fig.axes.index(a) for a in v] for k, v in roles.items()},
    }
    return meta

def attach_roles_to_meta(meta: Dict[str, Any], roles: Dict[str, List[plt.Axes]]) -> Dict[str, Any]:
    """Attach/override roles in a previously created meta dict (stores indices)."""
    fig = plt.gcf()
    idx_map = {ax: i for i, ax in enumerate(fig.axes)}
    meta["Roles"] = {k: [idx_map.get(a, -1) for a in v] for k, v in roles.items()}
    return meta

def roles_from_meta(meta: Dict[str, Any], fig: Optional[plt.Figure]=None) -> Dict[str, List[plt.Axes]]:
    """Resolve role indices from a saved meta back to Axes objects on a figure."""
    f = fig if fig is not None else plt.gcf()
    axes = f.axes
    out: Dict[str, List[plt.Axes]] = {}
    for k, idxs in meta.get("Roles", {}).items():
        out[k] = [axes[i] for i in idxs if 0 <= i < len(axes)]
    return out

def write_json(meta: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

# ---------- bbox union / crop ----------

def bbox_union_in(fig: plt.Figure, axes: Sequence[mpl.axes.Axes]) -> Tuple[float,float,float,float]:
    """Union of axes bboxes in inches: returns (x,y,w,h)."""
    if not axes:
        raise ValueError("axes is empty")
    boxes = [_ax_bbox_in(fig, ax) for ax in axes]
    xs = [b[0] for b in boxes]; ys = [b[1] for b in boxes]
    ws = [b[2] for b in boxes]; hs = [b[3] for b in boxes]
    x0 = min(xs); y0 = min(ys)
    x1 = max(x+ w for x, w in zip(xs, ws))
    y1 = max(y+ h for y, h in zip(ys, hs))
    return (x0, y0, x1-x0, y1-y0)

def crop_bbox_in(fig: plt.Figure, bbox_in: Tuple[float,float,float,float], pad_in: float=0.0) -> Bbox:
    """Create a Matplotlib Bbox (display coords) from an inches bbox with optional padding."""
    x, y, w, h = bbox_in
    x -= pad_in; y -= pad_in; w += 2*pad_in; h += 2*pad_in
    dpi = fig.dpi
    return Bbox.from_bounds(x*dpi, y*dpi, w*dpi, h*dpi)

def export_cropped(fig: plt.Figure, path: str,
                   bbox_in: Tuple[float,float,float,float],
                   *, dpi: Optional[int]=None, metadata: Optional[Dict[str,str]]=None) -> None:
    """
    Export a cropped view of the figure to a given bbox (in inches).
    Note: this uses bbox_inches=Bbox(...), NOT 'tight'.
    """
    bb = crop_bbox_in(fig, bbox_in, pad_in=0.0)
    if dpi is None:
        fig.savefig(path, bbox_inches=bb, metadata=(metadata or {}))
    else:
        fig.savefig(path, dpi=dpi, bbox_inches=bb, metadata=(metadata or {}))

def export_roles(meta: Dict[str, Any],
                 roles: Sequence[str],
                 path_no_ext: str,
                 *, pad_in: float=0.0,
                 formats: Sequence[str]=('pdf','png'),
                 dpi: Optional[int]=None,
                 metadata: Optional[Dict[str,str]]=None) -> List[str]:
    """
    Export the union bbox of the requested roles as cropped files.
    Example: export_roles(meta, ['panel'], 'fig1_panel')
    """
    fig = meta.get("fig", None)
    if fig is None:
        fig = plt.gcf()
    role_map = roles_from_meta(meta, fig)
    selected_axes = []
    for r in roles:
        selected_axes.extend(role_map.get(r, []))
    if not selected_axes:
        return []
    bbox_in = bbox_union_in(fig, selected_axes)
    # apply padding in inches if requested
    x,y,w,h = bbox_in
    bbox_in = (x-pad_in, y-pad_in, w+2*pad_in, h+2*pad_in)

    written = []
    for ext in formats:
        path = f"{path_no_ext}.{ext.lower()}"
        export_cropped(fig, path, bbox_in, dpi=dpi, metadata=metadata)
        written.append(path)
    return written
