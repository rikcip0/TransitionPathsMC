#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Drop-in “paper-ready” per β_M vs β_G scatter (identico formato dell'altro drop-in).
⟶ Mantiene *identica* data-area (W×H) con/ senza colorbar. Colorbar in gutter dedicato (cax separato).
⟶ Nessun tight/ constrained. Nessun bbox_inches="tight".
⟶ LaTeX overlay via MyBasePlots (percorsi assoluti agli .mplstyle).
⟶ Spessori/ticks coerenti (spines ~0.9 pt; ticks out 3.2 pt/0.8 pt; Y MaxNLocator 3–4; offset ridotto).
⟶ Auto-espansione in larghezza per tick/label della colorbar, data-area invariata.
⟶ Decimali colorbar esattamente N (di default 2). Basename export personalizzabile.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.colors import Normalize
from matplotlib.colorbar import Colorbar
from matplotlib.cm import get_cmap

# === USER TUNABLE (assoluti, in pollici) ===
FIG_SCALE   = 1.40  # moltiplica larghezza+altezza complessiva (2 colonne: es. 1.40-1.50)
DATA_W_IN   = 1.60   # larghezza data-area (identica in entrambi i casi)
DATA_H_IN   = 1.10   # altezza   data-area (identica in entrambi i casi)

LEFT_IN     = 0.40   # margine sinistro (base, prima del "bump")
RIGHT_FRAME = 0.08   # margine destro “di cornice” (senza colorbar)
BOTTOM_IN   = 0.34   # margine inferiore (base, prima del "bump")
TOP_IN      = 0.16   # margine superiore

CB_PAD_IN       = 0.06  # spazio tra data-area e colorbar
CB_W_IN         = 0.10  # larghezza *barra* colorbar (snella)
CB_RIGHT_PAD_IN = 0.16  # spazio a destra della colorbar per ticklabels/label

# Micro-margini extra per etichette senza toccare la data-area
LEFT_BUMP_IN   = 0.02  # extra a sinistra
BOTTOM_BUMP_IN = 0.02  # extra in basso
TOP_BUMP_IN    = 0.00
RIGHT_BUMP_IN  = 0.02  # extra a destra SOLO senza colorbar

# Modalità stile/LaTeX
USE_TEX_MODE   = "latex"  # "latex", "latex_text", "pgf"
STYLE_BASE     = "paper_base.mplstyle"
STYLE_OVERLAY  = "overlay_latex.mplstyle"

# === Formatting / export ===
CB_TICK_NBINS    = 4   # numero target di tick nella colorbar
CB_TICK_DECIMALS = 2   # esattamente N decimali sui tick della colorbar
OUT_NAME_OVERRIDE = None  # se non None, forza il basename (senza estensione)

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
    # figsize è ignorata dal layout deterministico — usiamo i pollici sopra.
}

# Default edge palette per trajInit
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
PLOTS = [
    {
        "id": "ER_Ngt30_edgeInit_unused",
        "model": "ER",
        "subset_id": "c2435bc67a3825b9",
        "include_unused": True,
        "edge_by_init": True,
        # "out_name": "nome_corto_opzionale",
    },
]

# ------------------- MyBasePlots imports -------------------
import sys
sys.path.append('../')
from MyBasePlots.FigCore import utils_style as ustyle
from MyBasePlots.FigCore import utils_plot as uplot

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

def _auto_expand_for_cbar(fig: plt.Figure, ax: plt.Axes, cbar: Colorbar,
                          L: float, B: float, T: float, R_frame: float,
                          fig_w: float, fig_h: float,
                          DW: float, DH: float, CB_PAD: float, CB_W: float) -> float:
    """Allarga la figura se tick/label della colorbar escono a destra, mantenendo invariata la data-area."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W0, H0 = fig.get_size_inches()
    right_inches = 0.0
    # bbox del cax e di tutti gli elementi testuali
    b = cbar.ax.get_window_extent(renderer=r).transformed(fig.transFigure.inverted())
    right_inches = max(right_inches, b.x1 * W0)
    for artist in list(cbar.ax.get_yticklabels()) + [cbar.ax.yaxis.get_label()]:
        try:
            bb = artist.get_window_extent(renderer=r).transformed(fig.transFigure.inverted())
            right_inches = max(right_inches, bb.x1 * W0)
        except Exception:
            continue
    desired_right_edge = right_inches + 0.02  # piccolo margine
    extra = max(0.0, desired_right_edge + R_frame - W0)
    if extra > 1e-3:
        new_w = W0 + extra
        fig.set_size_inches(new_w, H0, forward=False)
        ax.set_position([L/new_w, B/H0, DW/new_w, DH/H0])
        cax = cbar.ax
        cax.set_position([(L + DW + CB_PAD)/new_w, B/H0, CB_W/new_w, DH/H0])
        fig.canvas.draw()
        print(f"[W EXPAND] +{extra:.3f} in to fit colorbar labels/ticks")
        return extra
    return 0.0

def _auto_expand_for_axes_labels(fig: plt.Figure, ax: plt.Axes,
                                 L: float, B: float, T: float, R_frame: float,
                                 fig_w: float, fig_h: float,
                                 DW: float, DH: float) -> tuple[float, float, float, float]:
    """Se xlabel/ylabel o i loro tick vengono tagliati a SINISTRA o in BASSO,
    aumenta la larghezza/altezza figura e (se necessario) i margini L/B, mantenendo fissa la data-area."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W0, H0 = fig.get_size_inches()

    # Trova il minimo x delle etichette a sinistra (tick + ylabel)
    min_x = 1e9
    for artist in list(ax.get_yticklabels()) + [ax.yaxis.get_label()]:
        try:
            bb = artist.get_window_extent(renderer=r).transformed(fig.transFigure.inverted())
            min_x = min(min_x, bb.x0)
        except Exception:
            pass
    extra_left = 0.0
    if min_x < 0.0:
        extra_left = (-min_x) * W0 + 0.02  # + un pelino

    # Trova il minimo y delle etichette in basso (tick + xlabel)
    min_y = 1e9
    for artist in list(ax.get_xticklabels()) + [ax.xaxis.get_label()]:
        try:
            bb = artist.get_window_extent(renderer=r).transformed(fig.transFigure.inverted())
            min_y = min(min_y, bb.y0)
        except Exception:
            pass
    extra_bottom = 0.0
    if min_y < 0.0:
        extra_bottom = (-min_y) * H0 + 0.02  # + un pelino

    new_L, new_B, new_W, new_H = L, B, W0, H0
    if extra_left > 1e-3:
        new_W = W0 + extra_left
        new_L = L + extra_left
    if extra_bottom > 1e-3:
        new_H = H0 + extra_bottom
        new_B = B + extra_bottom

    if (extra_left > 1e-3) or (extra_bottom > 1e-3):
        fig.set_size_inches(new_W, new_H, forward=False)
        ax.set_position([new_L/new_W, new_B/new_H, DW/new_W, DH/new_H])
        fig.canvas.draw()

    return new_L, new_B, new_W, new_H

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

    # ====================== GEOMETRIA DETERMINISTICA ======================
    # Sempre con colorbar per N
    cb_enabled = True

    # Bump base margins
    L0 = LEFT_IN + LEFT_BUMP_IN
    B0 = BOTTOM_IN + BOTTOM_BUMP_IN
    T0 = TOP_IN + TOP_BUMP_IN
    R0 = RIGHT_FRAME + (RIGHT_BUMP_IN if not cb_enabled else 0.0)

    # Apply FIG_SCALE to all inch quantities
    L = L0 * FIG_SCALE; B = B0 * FIG_SCALE; T = T0 * FIG_SCALE; R_frame = R0 * FIG_SCALE
    DW = DATA_W_IN * FIG_SCALE; DH = DATA_H_IN * FIG_SCALE
    CB_PAD = CB_PAD_IN * FIG_SCALE; CB_W = CB_W_IN * FIG_SCALE; CB_RP = CB_RIGHT_PAD_IN * FIG_SCALE

    # Figure dimensions (scaled)
    fig_w = L + DW + (CB_PAD + CB_W + CB_RP if cb_enabled else 0.0) + R_frame
    fig_h = B + DH + T

    # Create figure/axes with exact data-area via framework
    _styles_root = Path(ustyle.__file__).resolve().parent / "styles"
    with ustyle.auto_style(mode=USE_TEX_MODE,
                           base=str(_styles_root / STYLE_BASE),
                           overlay=str(_styles_root / STYLE_OVERLAY)):
        fig, ax, _meta = uplot.figure_single_fixed(
            data_w_in=DW, data_h_in=DH,
            left_in=L, right_in=(R_frame + (CB_PAD + CB_W + CB_RP if cb_enabled else 0.0)),
            bottom_in=B, top_in=T
        )
        fig.set_constrained_layout(False)
        try:
            fig.tight_layout = lambda *a, **k: None
        except Exception:
            pass

        # Spessori/ ticks coerenti
        for sp in ax.spines.values():
            sp.set_linewidth(0.9)
        ax.tick_params(direction='out', length=3.2, width=0.8, pad=2.6)

        # cax per colorbar (stessa altezza della data-area; NON riduce l'axes)
        cax = fig.add_axes([(L + DW + CB_PAD)/fig_w, B/fig_h, CB_W/fig_w, DH/fig_h])

        # ----- DATA LOGIC (immutata) -----
        cmap = get_cmap(str(opt.get("cmap", "viridis")))

        # vmin/vmax: se non forniti, calcola da used+unused per coerenza
        if opt.get("vmin") is not None and opt.get("vmax") is not None:
            vmin = float(opt["vmin"]); vmax = float(opt["vmax"])
        else:
            cvals = []
            if not used.empty: cvals.append(used["N"].astype(float).to_numpy())
            if not unused.empty: cvals.append(unused["N"].astype(float).to_numpy())
            vv = np.concatenate(cvals) if cvals else np.array([], dtype=float)
            vv = vv[np.isfinite(vv)]
            if vv.size == 0:
                vmin = 0.0; vmax = 1.0
            else:
                vmin = float(np.min(vv)); vmax = float(np.max(vv))

        def scatter_block(dfin: pd.DataFrame, size, alpha, edge_default):
            if dfin.empty:
                return None
            edge_kw = {}
            if bool(opt.get("edge_by_init", False)):
                # palette è già pronto se edge_by_init True
                if "trajInit" not in dfin.columns:
                    # in pratica qui dovremmo avere già merge fatto sopra
                    pass
                edge_kw["edgecolors"] = _edge_colors_from_trajInit(dfin["trajInit"], palette)
                edge_kw["linewidths"] = float(opt.get("edge_lw", DEFAULTS["edge_lw"]))
            else:
                edge_kw["edgecolors"] = edge_default
                if edge_default != "none":
                    edge_kw["linewidths"] = 0.4
            return ax.scatter(dfin["betaM"], dfin["betaG"],
                              c=dfin["N"], cmap=cmap, vmin=vmin, vmax=vmax,
                              s=size, alpha=alpha, **edge_kw)

        if not unused.empty:
            scatter_block(unused, size=18, alpha=0.25, edge_default="none")
        sc = scatter_block(used, size=28, alpha=0.9, edge_default="k")

        # --- colorbar decoupled: mappable dedicato ---
        sm = plt.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
        sm.set_array([])
        cbar = Colorbar(cax, mappable=sm, orientation='vertical')
        cbar.ax.tick_params(pad=2.0, length=3.0, width=0.8, direction='out')
        cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=CB_TICK_NBINS, min_n_ticks=3))
        cbar.set_label(r"$N$")

        # draw + *ripristino* posizione dell’axes principale (area-dati invariata)
        fig.canvas.draw()
        ax.set_position([L/fig_w, B/fig_h, DW/fig_w, DH/fig_h])
        _exp = _auto_expand_for_cbar(fig, ax, cbar, L, B, T, R_frame, fig_w, fig_h, DW, DH, CB_PAD, CB_W)
        if _exp > 0:
            fig_w += _exp
        # Auto-espansione per evitare tagli a sinistra/in basso delle etichette
        L, B, fig_w, fig_h = _auto_expand_for_axes_labels(fig, ax, L, B, T, R_frame, fig_w, fig_h, DW, DH)

        # Riferimenti M/G
        if _fmt_float(betaM_ref) is not None:
            ax.axvline(betaM_ref, color="black", linestyle="--", linewidth=1.2, zorder=0)
        if _fmt_float(betaG_ref) is not None:
            ax.axhline(betaG_ref, color="black", linestyle="--", linewidth=1.2, zorder=0)

        ax.set_xlabel(r"$\beta_M$")
        ax.set_ylabel(r"$\beta_G$")
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        try:
            off = ax.yaxis.get_offset_text(); off.set_fontsize(off.get_fontsize()*0.72)
        except Exception:
            pass

        # Output path: Analysis/FigScripts/_figs/<MODEL>/betaM_vs_betaG/<subset_folder>/
        model_folder = _sanitize_segment(model)
        out_dir = figs_root / "betaM_vs_betaG" /model_folder /  _subset_folder_name(subset_label, subset_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        base_id = _sanitize_segment(opt.get("id") or f"{model_folder}__{subset_id[:8]}")
        default_base = f"{base_id}__{_sanitize_segment(opt['ref_stat'])}"
        _user_basename = cfg.get("out_name") or OUT_NAME_OVERRIDE
        out_base = out_dir / (_sanitize_segment(_user_basename) if _user_basename else default_base)

        # === LOG GEOMETRIA PRIMA DEL SALVATAGGIO ===
        W, H = fig.get_size_inches()
        print(f"[FIG SIZE] betaM_vs_betaG: {W:.3f} × {H:.3f} in")
        bbox = ax.get_position(); data_w = bbox.width * W; data_h = bbox.height * H
        right_total = (R_frame + (CB_PAD + CB_W + CB_RP))
        print(f"[AX BOX]   data: {data_w:.3f} × {data_h:.3f} in; margins L/R/B/T={L:.2f}/{right_total:.2f}/{B:.2f}/{T:.2f}")
        print(f"[SCALE]     FIG_SCALE={FIG_SCALE:.2f} ⇒ DW×DH={DW:.2f}×{DH:.2f} in")

        # Export rigoroso (PNG+PDF) senza tight
        uplot.export_figure_strict(fig, str(out_base), formats=('png','pdf'), dpi=int(opt["dpi"]))

        # Metadata JSON (coerente con l'originale)
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
            "edge_palette": (_parse_edge_palette(opt.get("edge_palette")) if bool(opt.get("edge_by_init", False)) else None),
            "plot": {
                "dpi": int(opt["dpi"]),
                "figsize": [float(W), float(H)],
                "cmap": str(opt.get("cmap", "viridis")),
                "vmin": _fmt_float(opt.get("vmin")),
                "vmax": _fmt_float(opt.get("vmax")),
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
    # Fig output root (fixed under Analysis/FigScripts/_figs)
    script_dir = Path(__file__).resolve().parent
    figs_root = script_dir / "_figs"

    for cfg in PLOTS:
        make_one_plot(cfg, DEFAULTS, figs_root)


if __name__ == "__main__":
    main()
