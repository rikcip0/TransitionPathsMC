#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# energy_vs_m_allTrajs — PAPER READY
# Replicates the "allTrajs → energyVsM" branch from singleRunAnalysis,
# with a simple PLOTS list at the top.
#
# - Reads story* from a simulation run folder (path Windows/realGraphs handled).
# - Normalizes M and energy by N (N is REQUIRED).
# - Drops the initialization traj (index 0).
# - Computes mean/σ/median of energy parametrized by M (as in meanAndSigmaForParametricPlot).
# - Plots errorbar (mean±σ) and, by default, median points (can be disabled per-plot).
# - Optional vertical guide on m* via xGuides.
# - Uses FigCore style (no fallback). title is empty -> no extra top margin.
# - Exports and closes the figure.
#
# Dependencies expected in your repo (as in your other scripts):
#   from MyBasePlots.FigCore import utils_style as ustyle
#   from MyBasePlots.FigCore import utils_plot  as uplot

from __future__ import annotations
from typing import List, Dict, Any, Optional, Sequence, Tuple
import os, sys, re, json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

show_median_default = True

# ======================
# LISTA PLOT (EDITA QUI)
# ======================
PLOTS: List[Dict[str, Any]] = [
    {
      "name": "EnergyVsM_allTrajs/ZKC40_0p9",
      "run_path": r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\40_0.9_0_inf_20_inf_run2620",
      "N": 40,                               # REQUIRED (normalization M/N, E/N)
      "figsize": [6.0, 4.2],
      "x_label": "m", "y_label": "e",       # labels for paper
      "xGuides": [[0.5, "m*", "red"]],       # optional vertical guide(s)
      "outfile": "fig/EnergyVsM_allTrajs/ZKC40_0p9",
      "formats": ["pdf","png"],
      "dpi": 300
    },
]

# ===============
# CONFIG AMBIENTE
# ===============
REPO_ROOT = Path(__file__).resolve().parents[2]  # adjust if needed
sys.path.insert(0, str(REPO_ROOT))
sys.path.append('../')  # match your other scripts

from MyBasePlots.FigCore import utils_style as ustyle
from MyBasePlots.FigCore import utils_plot  as uplot

# ===== util path =====
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

# ====== IO story (same logic as your singleRunAnalysis) ======
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

# ====== as in your singleRunAnalysis ======
def meanAndSigmaForParametricPlot(toBecomeX: np.ndarray, toBecomeY: np.ndarray):
    x_unique_values = np.unique(toBecomeX)
    y_mean_values = np.asarray([np.mean(toBecomeY[toBecomeX == x_value]) for x_value in x_unique_values])
    y_var_values = np.asarray([np.var (toBecomeY[toBecomeX == x_value])**0.5 for x_value in x_unique_values])
    y_median_values = np.asarray([np.median(toBecomeY[toBecomeX == x_value]) for x_value in x_unique_values])
    return x_unique_values, y_mean_values, y_var_values, y_median_values

def run_from_specs(specs: List[Dict[str, Any]]) -> None:
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
            times   = arrays[0, :, :]
            q_in    = arrays[_YKEY_TO_INDEX["qin"], :, :]
            q_out   = arrays[_YKEY_TO_INDEX["qout"], :, :]
            M       = arrays[_YKEY_TO_INDEX["m"], :, :]
            energy  = arrays[_YKEY_TO_INDEX["energy"], :, :]

            # normalize by N (float) as in your code
            N = float(N)
            M      = M / N
            energy = energy / N

            # drop initialization traj (index 0)
            M      = M[1:, :]
            energy = energy[1:, :]

            # Compute parametric stats
            a = meanAndSigmaForParametricPlot(M, energy)  # returns x, mean, sigma, median

            # Figure
            figsize = tuple(spec.get("figsize", (6.0, 4.2)))
            fig = plt.figure(name, figsize=figsize)
            ax  = fig.add_subplot(111)
            # Title empty (paper): no top margin reserved
            ax.set_title("")

            # Labels
            ax.set_xlabel(spec.get("x_label", "m"))
            ax.set_ylabel(spec.get("y_label", "e"))

            # Plot mean ± sigma
            ax.errorbar(a[0], a[1], a[2], label='mean')

            # Median points (default True, overridable by spec)
            show_median = bool(spec.get("show_median", show_median_default))
            if show_median:
                ax.scatter(a[0], a[3], color='darkorange', s=18, label='median', zorder=50)

            # Optional vertical guides (e.g., m*)
            xGuides = spec.get("xGuides")
            if xGuides:
                for val, *rest in xGuides:
                    color = rest[1] if len(rest) >= 2 else 'red'
                    ax.axvline(val, color=color, linestyle='dashed', linewidth=1)

            # Legend optional (default off for paper)
            if bool(spec.get("legend", False)):
                ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

            # Export
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
