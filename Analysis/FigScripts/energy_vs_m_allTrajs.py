#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# energy_vs_m_allTrajs — PAPER READY (drop-in from user's version, v8)
# Focus: N=34, labels non tagliati, >=3 ticks per asse, larghezze identiche.

from __future__ import annotations
from typing import List, Dict, Any, Optional
import os, sys, re, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from pathlib import Path

# ======================
# LISTA PLOT (N=34)
# ======================
PLOTS: List[Dict[str, Any]] = [
    {
      "name": "EnergyVsM_allTrajs/ZKC40_0p55",
      "run_path": r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\40_0.55_0_inf_20_inf_run2634",
      "N": 34,
      "x_label": r"$m$", "y_label": r"$e$",
      "xGuides": [[0.5, "m*", "red"]],  # solo linea tratteggiata (testo ignorato)
      "outfile": "_figs/EnergyVsM_allTrajs/ZKC40_0p55",
      "formats": ["pdf","png"],
      "dpi": 300,
      # Geometria: mid/right un filo stretti (113.00156 pt). Se vuoi .30\linewidth usa 153.00156.
      "layout_width_pt": 113.00156,
      "data_h_in": 1.12,
      "left_in": 0.46, "right_in": 0.10, "bottom_in": 0.37, "top_in": 0.09,
      # Tipografia locale (nessun rc globale)
      "font_scale": 1,
      # Format Y to 1 decimal to keep labels compact
      "y_fmt_1dec": True
    },
    {
      "name": "EnergyVsM_allTrajs/ZKC40_0p35",
      "run_path": r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\40_0.35_0_inf_20_inf_run2642",
      "N": 34,
      "x_label": r"$m$", "y_label": r"$e$",
      "xGuides": [[0.5, "m*", "red"]],
      "outfile": "_figs/EnergyVsM_allTrajs/ZKC40_0p35",
      "formats": ["pdf","png"],
      "dpi": 300,
      "layout_width_pt": 113.00156,
      "data_h_in": 1.12,
      "left_in": 0.46, "right_in": 0.10, "bottom_in": 0.37, "top_in": 0.09,
      "font_scale": 1,
      "y_fmt_1dec": True
    },
    {
      "name": "EnergyVsM_allTrajs/ZKC40_0p25",
      "run_path": r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\40_0.25_0_inf_20_inf_run6871",
      "N": 34,
      "x_label": r"$m$", "y_label": r"$e$",
      "xGuides": [[0.5, "m*", "red"]],
      "outfile": "_figs/EnergyVsM_allTrajs/ZKC40_0p25",
      "formats": ["pdf","png"],
      "dpi": 300,
      "layout_width_pt": 113.00156,
      "data_h_in": 1.12,
      "left_in": 0.46, "right_in": 0.10, "bottom_in": 0.37, "top_in": 0.09,
      "font_scale": 1,
      "y_fmt_1dec": True
    },
    {
      "name": "EnergyVsM_allTrajs/ZKC40_0p9",
      "run_path": r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\40_0.9_0_inf_20_inf_run2620",
      "N": 34,
      "x_label": r"$m$", "y_label": r"$e$",
      "xGuides": [[0.5, "m*", "red"]],
      "outfile": "_figs/EnergyVsM_allTrajs/ZKC40_0p9",
      "formats": ["pdf","png"],
      "dpi": 300,
      "layout_width_pt": 113.00156,
      "data_h_in": 1.12,
      "left_in": 0.46, "right_in": 0.10, "bottom_in": 0.37, "top_in": 0.09,
      "font_scale": 1,
      "y_fmt_1dec": True
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

_CANDIDATE_BASES = [
    Path.cwd(),
    Path.cwd().parent,
    Path.cwd().parent.parent,
    REPO_ROOT,
    REPO_ROOT / "Data",
    REPO_ROOT / "Data" / "MultiPathsMC",
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
    if not s:
        raise ValueError("run_path is empty")
    cand = _maybe_graphs_prefix(s)
    if cand and cand.exists():
        return cand
    s = _windows_to_wsl(s)
    cand = Path(s)
    if cand.is_absolute() and cand.exists():
        return cand
    s2 = s.lstrip("/")
    for base in _CANDIDATE_BASES:
        cand = (base / s2).resolve()
        if cand.exists():
            return cand
    return Path(s if s.startswith("/") else s2)

def get_file_with_prefix(parent_dir: str, prefix: str) -> Optional[str]:
    if not os.path.isdir(parent_dir):
        return None
    all_files = [f for f in os.listdir(parent_dir) if os.path.isfile(os.path.join(parent_dir, f))]
    for file in all_files:
        if file.startswith(prefix):
            return os.path.join(parent_dir, file)
    return None

def arraysFromBlockFile(file_path: str) -> np.ndarray:
    with open(file_path, 'r') as file:
        lines = file.readlines()

    blocks, current = [], []
    for line in lines:
        line = line.strip()
        if not line:
            if current:
                blocks.append(current); current = []
        else:
            cols = [float(val) for val in line.split()]
            current.append(cols)
    if current:
        blocks.append(current)

    arr = np.asarray([np.array(block) for block in blocks])  # (n_blocks, n_rows, n_cols)
    return np.transpose(arr, (2, 0, 1))  # -> (n_cols, n_blocks, n_rows)

_YKEY_TO_INDEX = {"qin": 1, "qout": 2, "m": 3, "energy": 4}

def _slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s).strip("_")

# ====== statistiche parametriche ======
def meanAndSigmaForParametricPlot(toBecomeX: np.ndarray, toBecomeY: np.ndarray):
    x_unique_values = np.unique(toBecomeX)
    y_mean_values = np.asarray([np.mean(toBecomeY[toBecomeX == x_value]) for x_value in x_unique_values])
    y_var_values  = np.asarray([np.var (toBecomeY[toBecomeX == x_value])**0.5 for x_value in x_unique_values])
    y_median_vals = np.asarray([np.median(toBecomeY[toBecomeX == x_value]) for x_value in x_unique_values])
    return x_unique_values, y_mean_values, y_var_values, y_median_vals

def _apply_font_scale(ax, scale: float):
    ax.xaxis.label.set_fontsize(ax.xaxis.label.get_size() * scale)
    ax.yaxis.label.set_fontsize(ax.yaxis.label.get_size() * scale)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontsize(lab.get_size() * scale)

def _ensure_min_ticks(ax, axis: str, min_major: int = 3):
    """Fallback: if the axis currently has < min_major ticks, enforce LinearLocator(min_major)."""
    if axis == 'x':
        if len(ax.get_xticks()) < min_major:
            ax.xaxis.set_major_locator(LinearLocator(min_major))
    else:
        if len(ax.get_yticks()) < min_major:
            ax.yaxis.set_major_locator(LinearLocator(min_major))

def run_from_specs(specs: List[Dict[str, Any]]) -> None:
    # Motore LaTeX (PGF heavy) come nel tuo file
    with ustyle.auto_style(mode="latex",
                           base=str(Path(ustyle.__file__).resolve().parent / "styles" / "paper_base.mplstyle"),
                           overlay=str(Path(ustyle.__file__).resolve().parent / "styles" / "overlay_latex.mplstyle")):
        for spec in specs:
            name     = spec.get("name", "EnergyVsM_allTrajs")
            raw_path = spec["run_path"]
            run_path = _normalize_run_path(raw_path)

            if not run_path.exists():
                bases = "\\n  - " + "\\n  - ".join(str(b) for b in _CANDIDATE_BASES)
                raise FileNotFoundError(f"Run folder not found:\\n  {raw_path}\\nNormalized to:\\n  {run_path}\\nTried bases:\\n{bases}")

            N = spec.get("N", None)
            if N in (None, 0):
                raise ValueError("N must be specified (>0) to normalize M and energy.")

            story_path = get_file_with_prefix(str(run_path), "story")
            if story_path is None:
                raise FileNotFoundError(f"No 'story*' file in {run_path}")

            arrays = arraysFromBlockFile(story_path)
            M       = arrays[_YKEY_TO_INDEX["m"], :, :]
            energy  = arrays[_YKEY_TO_INDEX["energy"], :, :]

            # normalize by N (float) — corretto N=34
            N = float(N)
            M      = M / N
            energy = energy / N

            # drop initialization traj (index 0), se presente
            if M.shape[0] > 0:
                M      = M[1:, :]
                energy = energy[1:, :]

            # statistiche parametriche
            xvals, ymean, ysig, ymed = meanAndSigmaForParametricPlot(M, energy)

            # ---------- GEOMETRIA FISSA ----------
            TEX_PT_PER_INCH = 72.27
            data_w_in = float(spec.get("data_w_in", spec.get("layout_width_pt", 153.00156))) / TEX_PT_PER_INCH
            data_h_in = float(spec.get("data_h_in", 1.12))
            left_in   = float(spec.get("left_in",   0.46))
            right_in  = float(spec.get("right_in",  0.10))
            bottom_in = float(spec.get("bottom_in", 0.34))
            top_in    = float(spec.get("top_in",    0.12))

            fig, ax, _meta = uplot.figure_single_fixed(
                data_w_in=data_w_in, data_h_in=data_h_in,
                left_in=left_in, right_in=right_in,
                bottom_in=bottom_in, top_in=top_in
            )

            # Labels
            ax.set_xlabel(spec.get("x_label", r"$m$"))
            ax.set_ylabel(spec.get("y_label", r"$e$"))

            # mean + sigma
            ax.errorbar(xvals, ymean, ysig, fmt='-',
                        linewidth=1.4, elinewidth=0.8, capsize=0, color='C0', zorder=10)

            # mediana
            ax.scatter(xvals, ymed, color='darkorange', s=8, zorder=50)

            # guida rossa (solo linea tratteggiata)
            for val, *rest in (spec.get("xGuides") or []):
                col = (rest[1] if len(rest) >= 2 else 'red')
                ax.axvline(val, color=col, linestyle='dashed', linewidth=1.6)

            # tick naturali, ma garantisci almeno 3 per asse
            _ensure_min_ticks(ax, 'x', min_major=3)
            _ensure_min_ticks(ax, 'y', min_major=3)

            # Y formatter (opzionale) per etichette strette e uniformi
            if bool(spec.get("y_fmt_1dec", True)):
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

            # Font scaling locale
            _apply_font_scale(ax, float(spec.get("font_scale", 1)))
            ax.xaxis.label.set_fontsize(ax.xaxis.label.get_size()*0.90)
            ax.yaxis.label.set_fontsize(ax.yaxis.label.get_size()*0.90)
            # Export (NO layout_fit_text, NO tight)
            out = spec.get("outfile") or _slug(name)
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            fmts = tuple(spec.get("formats", ["pdf","png"]))
            dpi  = int(spec.get("dpi", 300))

            w_in, h_in = fig.get_size_inches()
            print(f"[FIG SIZE] {name}: {w_in:.3f}×{h_in:.3f} in  ->  {int(round(w_in*dpi))}×{int(round(h_in*dpi))} px @ {dpi}dpi")

            uplot.export_figure_strict(fig, out, formats=fmts, dpi=dpi)
            plt.close(fig)
            print(f"Saved figure '{out}'.")

if __name__ == "__main__":
    run_from_specs(PLOTS)
