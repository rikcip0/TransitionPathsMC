#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch plots: β_M vs β_G scatter per subset (color=N), with optional edge color by trajInit.
- Reads TI tables from Data/MultiPathsMC/<model>/v1/ti
- Saves figures + metadata under Analysis/FigScripts/__figs/<MODEL>/betaM_vs_betaG/<subsetLabel__id8>/
- Configuration is list-driven: see DEFAULTS and PLOTS below.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ======================= Defaults & Plot List =======================

DEFAULTS: Dict[str, Any] = {
    "base": "../../Data",     # Data root (relative to this script's folder)
    "ref_stat": "mean",
    "include_unused": False,
    "edge_by_init": False,
    "edge_palette": "auto",   # 'auto' => DEFAULT_EDGE_PALETTE; or JSON dict str
    "edge_lw": 0.6,
    "cmap": "viridis",
    "vmin": None,
    "vmax": None,
    "dpi": 160,
    "figsize": (5, 4),        # inches
}

# Default edge palette for trajInit (as requested)
DEFAULT_EDGE_PALETTE = {
    "740": "red",
    "74": "orange",
    "73": "orange",
    "72": "purple",
    "71": "black",
    "70": "lightgreen",
    "0": "none",
    "-2": "black",
}

# --- LISTA PLOT ---
# Un unico elemento: equivalente a chiamare
#   python3 test2.py --model ER --subset-id c2435bc67a3825b9 --include-unused --edge-by-init
PLOTS = [
    {
        "id": "ER_Ngt30_edgeInit_unused",
        "model": "ER",
        "subset_id": "c2435bc67a3825b9",
        "include_unused": True,
        "edge_by_init": True,
    },
]


# ======================= Helpers =======================

def _sanitize_segment(s: str) -> str:
    """Permette solo [A-Za-z0-9._-]; il resto diventa '_'."""
    if s is None:
        return "none"
    s = str(s).replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)


def _subset_folder_name(label: str, subset_id: str) -> str:
    safe_lab = _sanitize_segment(label or "subset")
    return f"{safe_lab}__{subset_id[:8]}"


def _fmt_float(x):
    try:
        if x is None:
            return None
        xf = float(x)
        if np.isnan(xf) or np.isinf(xf):
            return None
        return xf
    except Exception:
        return None


def _utcnow_iso():
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_edge_palette(spec: str | dict | None) -> dict:
    """Ritorna dict palette. 'auto' => DEFAULT_EDGE_PALETTE; dict => normalizzato."""
    if spec is None or (isinstance(spec, str) and spec.strip().lower() == "auto"):
        return {str(k): ("none" if str(v).lower() == "none" else v)
                for k, v in DEFAULT_EDGE_PALETTE.items()}
    if isinstance(spec, dict):
        return {str(k): ("none" if str(v).lower() == "none" else v) for k, v in spec.items()}
    # try parse json
    obj = json.loads(spec)
    if not isinstance(obj, dict):
        raise ValueError("edge_palette non è un dict JSON")
    return {str(k): ("none" if str(v).lower() == "none" else v) for k, v in obj.items()}


def _edge_colors_from_trajInit(series: pd.Series, palette: dict) -> list[str]:
    """Restituisce lista di colori per edge, in base a trajInit, usando la palette fornita."""
    vals = series.to_numpy()
    out = []
    for v in vals:
        col = palette.get(str(v))
        if col is None:
            try:
                col = palette.get(str(int(v)))
            except Exception:
                col = None
        out.append(col if col is not None else "k")
    return out


def _ti_dir(base: Path, model: str) -> Path:
    return base / "MultiPathsMC" / model / "v1" / "ti"


def _read_parquet(p: Path, tag: str) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(f"[{tag}] non trovato: {p}")
    return pd.read_parquet(p)


def _resolve_subset_id(subsets: pd.DataFrame, subset_id: str | None, subset_label: str | None) -> tuple[str, str]:
    """Ritorna (subset_id, subset_label_resolved)."""
    if subset_id:
        sid = str(subset_id)
        row = subsets.loc[subsets["subset_id"] == sid].head(1)
        lab = row["subset_label"].iloc[0] if not row.empty else sid
        return sid, lab
    if subset_label:
        ids = subsets.loc[subsets["subset_label"] == str(subset_label), "subset_id"].drop_duplicates().astype(str).tolist()
        if len(ids) == 0:
            raise ValueError(f"[subset-label] '{subset_label}' non trovato")
        if len(ids) > 1:
            raise ValueError(f"[subset-label] '{subset_label}' mappa a più subset_id: {ids}. Specifica subset_id.")
        sid = ids[0]
        return sid, str(subset_label)
    raise ValueError("Serve 'subset_id' oppure 'subset_label'")


# ======================= Core plotting =======================

def make_one_plot(cfg: Dict[str, Any], defaults: Dict[str, Any], figs_root: Path):
    # Merge defaults
    opt = dict(defaults)
    opt.update(cfg)

    # Required
    model = opt.get("model")
    if not model:
        raise ValueError("cfg['model'] mancante")
    base = Path(opt["base"]).resolve()

    # Locate TI parquet directory
    ti_dir = _ti_dir(base, model)

    # Parquet paths
    p_curves = ti_dir / "ti_curves.parquet"
    p_members = ti_dir / "ti_subset_members.parquet"
    p_subsets = ti_dir / "ti_family_subsets.parquet"
    p_refs = ti_dir / "ti_subset_refs.parquet"
    p_points = ti_dir / "ti_points.parquet"  # for trajInit if needed

    curves = _read_parquet(p_curves, "ti_curves")
    members = _read_parquet(p_members, "ti_subset_members")
    subsets = _read_parquet(p_subsets, "ti_family_subsets")
    refs = _read_parquet(p_refs, "ti_subset_refs")

    # Normalize types
    for df in (members, subsets, refs, curves):
        if "subset_id" in df.columns:
            df["subset_id"] = df["subset_id"].astype(str)

    # Resolve subset_id & label
    subset_id, subset_label = _resolve_subset_id(subsets, opt.get("subset_id"), opt.get("subset_label"))

    # Members
    need_mem = ["TIcurve_id", "subset_id", "family_id", "N", "is_used"]
    for c in need_mem:
        if c not in members.columns:
            raise ValueError(f"[ti_subset_members] manca colonna '{c}'")
    mem = members.loc[members["subset_id"] == subset_id, need_mem].drop_duplicates()
    if mem.empty:
        raise ValueError(f"[subset_id={subset_id}] nessun membro trovato in ti_subset_members")

    # Curves join
    need_curves = ["TIcurve_id", "betaM", "betaG"]
    has_traj_curves = "trajInit" in curves.columns
    if has_traj_curves:
        need_curves.append("trajInit")
    for c in need_curves:
        if c not in curves.columns:
            raise ValueError(f"[ti_curves] manca colonna '{c}'")
    df = mem.merge(curves[need_curves], on="TIcurve_id", how="inner")
    if df.empty:
        raise ValueError(f"[subset_id={subset_id}] dopo join con ti_curves nessuna riga")

    # edge-by-init → ensure trajInit
    palette = None
    if bool(opt.get("edge_by_init", False)):
        if "trajInit" not in df.columns:
            pts = _read_parquet(p_points, "ti_points")
            if "trajInit" not in pts.columns or "TIcurve_id" not in pts.columns:
                raise ValueError("[edge-by-init] 'trajInit' non presente né in ti_curves né in ti_points")
            map_init = pts[["TIcurve_id", "trajInit"]].dropna().drop_duplicates(subset=["TIcurve_id"])
            df = df.merge(map_init, on="TIcurve_id", how="left")
            if df["trajInit"].isna().all():
                raise ValueError("[edge-by-init] impossibile ricavare 'trajInit' per le curve selezionate")
        palette = _parse_edge_palette(opt.get("edge_palette"))

    # Refs (M/G)
    need_refs = ["subset_id", "ref_type", "ref_stat"]
    for c in need_refs:
        if c not in refs.columns:
            raise ValueError(f"[ti_subset_refs] manca colonna '{c}'")
    rsub = refs[(refs["subset_id"] == subset_id) & (refs["ref_stat"] == opt["ref_stat"])]
    def _pick_beta_ref(ref_type: str):
        r = rsub[rsub["ref_type"] == ref_type]
        if r.empty:
            return None
        col = "beta_ref" if "beta_ref" in r.columns else ("beta_curve" if "beta_curve" in r.columns else None)
        if col is None:
            raise ValueError("[ti_subset_refs] attesa colonna 'beta_ref' o 'beta_curve' per i riferimenti")
        return _fmt_float(r.iloc[0][col])
    betaM_ref = _pick_beta_ref("M")
    betaG_ref = _pick_beta_ref("G")

    # Split used/unused
    used = df[df["is_used"] == True].copy()
    unused = df[df["is_used"] != True].copy() if bool(opt.get("include_unused", False)) else df.iloc[0:0].copy()
    if used.empty and unused.empty:
        raise ValueError("[plot] nessun punto da mostrare (tutti unused e include_unused=False?)")

    # Figure
    figsize = tuple(opt["figsize"])
    fig, ax = plt.subplots(figsize=figsize, dpi=int(opt["dpi"]))

    def scatter_block(dfin: pd.DataFrame, size, alpha, edge_default):
        if dfin.empty:
            return None
        edge_kw = {}
        if palette is not None:
            edge_kw["edgecolors"] = _edge_colors_from_trajInit(dfin["trajInit"], palette)
            edge_kw["linewidths"] = float(opt["edge_lw"])
        else:
            edge_kw["edgecolors"] = edge_default
            if edge_default != "none":
                edge_kw["linewidths"] = 0.4
        return ax.scatter(dfin["betaM"], dfin["betaG"], c=dfin["N"], cmap=opt["cmap"],
                          vmin=opt["vmin"], vmax=opt["vmax"], s=size, alpha=alpha, **edge_kw)

    if not unused.empty:
        scatter_block(unused, size=18, alpha=0.25, edge_default="none")
    sc = scatter_block(used, size=28, alpha=0.9, edge_default="k")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("N")

    if betaM_ref is not None:
        ax.axvline(betaM_ref, color="C3", linestyle="--", linewidth=1.2)
    if betaG_ref is not None:
        ax.axhline(betaG_ref, color="C4", linestyle="--", linewidth=1.2)

    ax.set_xlabel(r"$\beta_M$ (originale)")
    ax.set_ylabel(r"$\beta_G$ (originale)")
    ax.grid(True, alpha=0.3)

    # Output path: Analysis/FigScripts/__figs/<MODEL>/betaM_vs_betaG/<subset_folder>/
    model_folder = _sanitize_segment(model)
    out_dir = figs_root / model_folder / "betaM_vs_betaG" / _subset_folder_name(subset_label, subset_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_id = _sanitize_segment(opt.get("id") or f"{model_folder}__{subset_id[:8]}")
    out_base = out_dir / f"{base_id}__{_sanitize_segment(opt['ref_stat'])}"

    fig.savefig(str(out_base) + ".png", bbox_inches="tight", dpi=int(opt["dpi"]))
    fig.savefig(str(out_base) + ".pdf", bbox_inches="tight", dpi=int(opt["dpi"]))
    print(out_base)
    # Metadata JSON
    meta = {
        "kind": "betaM_vs_betaG_scatter",
        "computed_at": _utcnow_iso(),
        "model": model,
        "subset_id": subset_id,
        "subset_label": subset_label,
        "ref_stat": opt["ref_stat"],
        "include_unused": bool(opt.get("include_unused", False)),
        "edge_by_init": bool(opt.get("edge_by_init", False)),
        "edge_lw": float(opt.get("edge_lw", DEFAULTS["edge_lw"])),
        "edge_palette": (palette if palette is not None else None),
        "plot": {
            "dpi": int(opt["dpi"]),
            "figsize": [float(figsize[0]), float(figsize[1])],
            "cmap": opt["cmap"],
            "vmin": _fmt_float(opt["vmin"]),
            "vmax": _fmt_float(opt["vmax"]),
            "lines": {
                "betaM_ref": _fmt_float(betaM_ref),
                "betaG_ref": _fmt_float(betaG_ref),
            }
        },
        "counts": {
            "total_join": int(len(df)),
            "used": int(len(used)),
            "unused_shown": int(len(unused)),
        },
        "sources": {
            "ti_dir": str(ti_dir),
            "parquets": {
                "curves": str(p_curves),
                "members": str(p_members),
                "subsets": str(p_subsets),
                "refs": str(p_refs),
                "points": str(p_points) if bool(opt.get("edge_by_init", False)) else None,
            }
        },
        "paths": {
            "png": str(out_base) + ".png",
            "pdf": str(out_base) + ".pdf",
        }
    }
    with open(str(out_base) + "_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    plt.close(fig)


# ======================= Runner =======================

def main():
    # Fig output root (fixed under Analysis/FigScripts/__figs)
    script_dir = Path(__file__).resolve().parent
    figs_root = script_dir / "__figs"

    for cfg in PLOTS:
        make_one_plot(cfg, DEFAULTS, figs_root)


if __name__ == "__main__":
    main()
