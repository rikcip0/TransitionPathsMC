#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Wrapper PAPER — STRICT + GUIDES + NORMALIZATION + pass-through CDF refs
# FIX: niente pre-slicing delle traiettorie -> passiamo i 2D array completi
#      (n_traj, n_t) a multipleCurvesAndHist + curvesIndices non consecutivi OK.
#
# - Nessun filtro kwargs (STRICT): se MH non supporta un argomento -> TypeError (voluto).
# - Richiede che MH accetti 'figsize' e lo applichi davvero (fail-fast).
# - Normalizza M/energy dividendo per N (OBBLIGATORIO quando y_key in {'M','energy'}).
# - Passa yGuides/xGuides e cdf_ref_probs_y/x direttamente a MH.
# - Replica guide nei pannelli istogrammi (solo le guide, non le CDF refs).
# - Path Windows/realGraphs normalizzati, stile FigCore esterno, title="" (niente spazio).

from __future__ import annotations
from typing import List, Dict, Any, Optional, Sequence, Tuple, Union
import os, sys, re, json, inspect
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ======================
# LISTA PLOT (EDITA QUI)
# ======================
PLOTS: List[Dict[str, Any]] = [
    # Esempio
    {
       "name": "Fig01_M_vs_t/ZKC40_0p9",
       "run_path": r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\40_0.9_0_inf_20_inf_run6952",
       "y_key": "M",
       "N": 34,                              # obbligatorio con M/energy
       "curvesIndices": [0, 14, 132], # NON consecutivi OK
       "x_label": "t", "y_label": "m",
       "show_yhist": True, "y_cdf": True,
       "yGuides": [[20./34., "m*", "red"]],
       "cdf_ref_probs_y": [1/3, 2/3],
       "y_log_density":True,
       "legend": False, "reserve_legend_space": True,
       "figsize": [6.0, 4.2],
       "outfile": "__figs/Fig01_m_vs_t/ZKC40_0p9",
     },
    {
       "name": "Fig01_M_vs_t/ZKC40_0p9b",
       "run_path": r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\40_0.9_0_inf_20_inf_run2620",
       "y_key": "M",
       "N": 34,                              # obbligatorio con M/energy
       "curvesIndices": [0, 3, 10, 25, 130], # NON consecutivi OK
       "x_label": "t", "y_label": "m",
       "show_yhist": True, "y_cdf": True,
       "yGuides": [[20./34., "m*", "red"]],
       "cdf_ref_probs_y": [1/3, 2/3],
       "y_log_density":False,
       "legend": False, "reserve_legend_space": True,
       "figsize": [6.0, 4.2],
       "outfile": "__figs/Fig01_m_vs_t/ZKC40_0p9b",
     },
    {
       "name": "Fig01_M_vs_t/ZKC40_0p8",
       "run_path": r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\40_0.8_0_inf_20_inf_run2624",
       "y_key": "M",
       "N": 34,                              # obbligatorio con M/energy
       "curvesIndices": [0, 3, 10, 25, 130], # NON consecutivi OK
       "x_label": "t", "y_label": "m",
       "show_yhist": True, "y_cdf": True,
       "yGuides": [[20./34., "m*", "red"]],
       "cdf_ref_probs_y": [1/3, 2/3],
       "legend": False, "reserve_legend_space": True,
       "figsize": [6.0, 4.2],
       "outfile": "__figs/Fig01_m_vs_t/ZKC40_0p8",
     },
]

# ===============
# CONFIG AMBIENTE
# ===============
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.append('../')

from MyBasePlots.FigCore import utils_style as ustyle
from MyBasePlots.FigCore import utils_plot  as uplot
from MyBasePlots.multipleCurvesAndHist import multipleCurvesAndHist

ArrayLike = Union[np.ndarray, Sequence[float], List[float]]

_CANDIDATE_BASES = [
    Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent,
    REPO_ROOT, REPO_ROOT / "Data", REPO_ROOT / "Data" / "MultiPathsMC",
    REPO_ROOT / "Data" / "Graphs",
]

def _windows_to_wsl(p: str) -> str:
    m = re.match(r"^([a-zA-Z]):[\\/](.*)$", p)
    if m:
        drive = m.group(1).lower()
        rest  = m.group(2).replace("\\\\","/").replace("\\","/")
        return f"/mnt/{drive}/{rest}"
    return p

def _maybe_graphs_prefix(s: str) -> Optional[Path]:
    s_clean = s.replace("\\\\","/").replace("\\","/").lstrip("/")
    if s_clean.lower().startswith("realgraphs/"):
        return (REPO_ROOT / "Data" / "Graphs" / s_clean).resolve()
    return None

def _normalize_run_path(raw: str) -> Path:
    s = os.path.expanduser(str(raw)).strip()
    if not s: raise ValueError("run_path vuoto")
    cand = _maybe_graphs_prefix(s)
    if cand and cand.exists(): return cand
    s = _windows_to_wsl(s)
    cand = Path(s)
    if cand.is_absolute() and cand.exists(): return cand
    s2 = s.lstrip("/")
    for base in _CANDIDATE_BASES:
        cand = (base / s2).resolve()
        if cand.exists(): return cand
    return Path(s if s.startswith("/") else s2)

def get_file_with_prefix(parent_dir: str, prefix: str) -> Optional[str]:
    if not os.path.isdir(parent_dir): return None
    all_files = [f for f in os.listdir(parent_dir) if os.path.isfile(os.path.join(parent_dir, f))]
    for file in all_files:
        if file.startswith(prefix): return os.path.join(parent_dir, file)
    return None

def arraysFromBlockFile(file_path: str) -> np.ndarray:
    with open(file_path, 'r') as file:
        lines = file.readlines()
    blocks, current = [], []
    for line in lines:
        line = line.strip()
        if not line:
            if current: blocks.append(current); current = []
        else:
            cols = [float(val) for val in line.split()]
            current.append(cols)
    if current: blocks.append(current)
    arr = np.asarray([np.array(block) for block in blocks])  # (n_blocks, n_rows, n_cols)
    arr = np.transpose(arr, (2, 0, 1))  # (n_cols, n_blocks, n_rows)
    return arr

_YKEY_TO_INDEX = {"qin": 1, "qout": 2, "m": 3, "energy": 4}

def _slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s).strip("_")

def _find_hist_axes(fig, ax_main):
    ax_xhist = None; ax_yhist = None
    try:
        for ax in fig.axes:
            if ax is ax_main: continue
            if ax.get_shared_x_axes().joined(ax, ax_main): ax_xhist = ax
            if ax.get_shared_y_axes().joined(ax, ax_main): ax_yhist = ax
    except Exception: pass
    return ax_xhist, ax_yhist

def _assert_figsize_supported() -> None:
    sig = inspect.signature(multipleCurvesAndHist)
    if "figsize" not in sig.parameters:
        raise RuntimeError("La tua multipleCurvesAndHist NON accetta 'figsize'. Aggiorna il file per supportarlo.")

def _normalize_series(Y2D, key, spec):
    key_l = key.lower()
    if key_l in ("m", "energy"):
        if "N" not in spec or spec["N"] in (None, 0):
            raise ValueError(f"Per y_key='{key}' devi specificare N (>0) per la normalizzazione.")
        N = float(spec["N"])
        return np.asarray(Y2D, dtype=float) / N
    return Y2D

def run_from_specs(specs: List[Dict[str, Any]]) -> None:
    _assert_figsize_supported()  # fail-fast su figsize

    with ustyle.auto_style(mode="latex",
                           base=str(Path(ustyle.__file__).resolve().parent / "styles" / "paper_base.mplstyle"),
                           overlay=str(Path(ustyle.__file__).resolve().parent / "styles" / "overlay_latex.mplstyle")):
        for spec in specs:
            name     = spec.get("name", "Figure")
            raw_path = spec["run_path"]
            run_path = _normalize_run_path(raw_path)

            if not run_path.exists():
                print(run_path)
                bases = "\\n  - " + "\\n  - ".join(str(b) for b in _CANDIDATE_BASES)
                raise FileNotFoundError(f"Cartella run non trovata:\\n  {raw_path}\\nNormalizzato a:\\n  {run_path}\\nCercati anche i prefissi:\\n{bases}")

            y_key    = str(spec.get("y_key","M")).lower()
            if y_key not in _YKEY_TO_INDEX:
                raise ValueError(f"y_key non valido: {y_key} (ammessi: {list(_YKEY_TO_INDEX.keys())})")

            story_path = get_file_with_prefix(str(run_path), "story")
            if story_path is None:
                raise FileNotFoundError(f"Nessun file 'story*' in {run_path}")

            arrays = arraysFromBlockFile(story_path)
            times_2d  = arrays[0, :, :]                    # (n_traj, n_t)
            yfull_2d  = arrays[_YKEY_TO_INDEX[y_key], :, :]# (n_traj, n_t)

            nTrajs = times_2d.shape[0]
            if nTrajs == 0:
                raise RuntimeError(f"Nessuna traiettoria in {story_path}")

            indices = spec.get("curvesIndices") or spec.get("curvesIndeces")
            if indices is not None and len(indices) == 0:
                indices = None  # evita edge case di lista vuota

            # Normalizzazione M/N o energy/N sull'intero 2D
            yfull_2d = _normalize_series(yfull_2d, y_key, spec)

            fig_size = tuple(spec.get("figsize", (6.0, 4.2)))
            yGuides = spec.get("yGuides")
            xGuides = spec.get("xGuides")

            # --- chiamata DIRETTA a MH con array 2D COMPLETI + indici ---
            fig, ax_main, meta = multipleCurvesAndHist(
                name=name,
                title="",
                x_list=times_2d,                # 2D (n_traj, n_t)
                x_label=spec.get("x_label","t"),
                y_list=yfull_2d,                # 2D (n_traj, n_t) normalizzato
                y_label=spec.get("y_label", y_key),
                curvesIndices=indices,          # <— indici non consecutivi OK (nessun pre-slicing)
                namesForCurves=spec.get("namesForCurves"),
                show_xhist=bool(spec.get("show_xhist", False)),
                show_yhist=bool(spec.get("show_yhist", True)),
                x_cdf=bool(spec.get("x_cdf", False)),
                y_cdf=bool(spec.get("y_cdf", True if spec.get("show_yhist", True) else False)),
                density=bool(spec.get("density", True)),
                y_log_density=bool(spec.get("y_log_density", False)),
                x_log_density=bool(spec.get("x_log_density", False)),
                y_bins='magnetizationComplete',
                x_bins=spec.get("x_bins", "auto"),
                legend=bool(spec.get("legend", False)),
                reserve_legend_space=bool(spec.get("reserve_legend_space", True)),
                manage_style=False,
                figsize=fig_size,
                palette=spec.get("palette", "cb_safe"),
                yGuides=yGuides,
                xGuides=xGuides,
                cdf_ref_probs_y=spec.get("cdf_ref_probs_y"),
                cdf_ref_probs_x=spec.get("cdf_ref_probs_x"),
            )

            # Verifica 'figsize'
            if fig_size:
                w, h = fig.get_size_inches()
                if abs(w - fig_size[0]) > 1e-6 or abs(h - fig_size[1]) > 1e-6:
                    print(f"'figsize' non applicato: atteso {fig_size}, ottenuto {(w,h)}")

            # Replica guide anche nei pannelli istogrammi (non le refs delle CDF: le fa MH)
            ax_xhist, ax_yhist = _find_hist_axes(fig, ax_main)
            if yGuides and ax_yhist is not None:
                for val, lab, col in yGuides:
                    ax_yhist.axhline(val, ls='--', lw=1.0, color=col)
            if xGuides and ax_xhist is not None:
                for val, lab, col in xGuides:
                    ax_xhist.axvline(val, ls='--', lw=1.0, color=col)

            # scale assi
            ax_main.set_xscale(spec.get("xscale", "linear"))
            ax_main.set_yscale(spec.get("yscale", "linear"))

            out = spec.get("outfile") or _slug(name)
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            uplot.export_figure_strict(fig, out, formats=tuple(spec.get("formats", ["pdf","png"])), dpi=int(spec.get("dpi", 300)))
            plt.close(fig)
            print(f"Saved figure '{out}'.")

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        specs = json.loads(Path(sys.argv[1]).read_text())
    else:
        specs = PLOTS
    run_from_specs(specs)
    print("Done.")
