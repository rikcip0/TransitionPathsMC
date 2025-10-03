#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Chi vs t (allTrajs) — EXACT progressive fit (copiato dal tuo singleRunAnalysis)

from __future__ import annotations
import os, sys, json, math
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt

# ======================
# LISTA PLOT (EDITA QUI)
# ======================
PLOTS: List[Dict[str, Any]] = [
    {
      "name": "ChiVsT_allTrajs/ZKC40_0p9",
      "run_path": r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\40_0.9_0_inf_20_inf_run2620",
      "Z_table": "",                 
      "Z_ref": None,                 
      "figsize": [6.0, 4.2],
      "legend": False,
      "outfile": "fig/ChiVsT_allTrajs/ZKC40_0p9",
      "formats": ["pdf","png"],
      "dpi": 300
    },
]

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2] if len(HERE.parents) >= 3 else HERE.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.append('../')

from MyBasePlots.FigCore import utils_style as ustyle
from MyBasePlots.FigCore import utils_plot  as uplot

def _norm(p: str|Path) -> Path:
    s = str(p)
    if os.name == "posix" and len(s) >= 2 and s[1] == ":" and s[0].isalpha():
        return Path("/mnt/{}/{}".format(s[0].lower(), s[3:].replace('\\','/')))
    return Path(p)

def get_file_with_prefix(parent_dir: str|Path, prefix: str) -> str|None:
    parent_dir = _norm(parent_dir)
    if not parent_dir.is_dir():
        return None
    for f in os.listdir(parent_dir):
        fp = parent_dir / f
        if fp.is_file() and f.startswith(prefix):
            return str(fp)
    return None

def _load_av_series(run_path: Path) -> Tuple[np.ndarray,np.ndarray]:
    av_path = get_file_with_prefix(run_path, "av")
    if av_path is None:
        raise FileNotFoundError("Manca av.dat nella cartella run: {}".format(run_path))
    rows = []
    with open(av_path, 'r', encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith(('#','%','//')): 
                continue
            parts = s.split()
            try:
                vals = [float(v) for v in parts]
                rows.append(vals)
            except ValueError:
                continue
    arr = np.asarray(rows, float)
    if arr.ndim != 2 or arr.shape[1] < 5:
        raise RuntimeError("Formato av.dat inatteso (cols<5) in: {}".format(av_path))
    t   = arr[:,0]
    chi = arr[:,4]   # avChi
    return t, chi

def _chi_err_from_runData(run_path: Path, chi: np.ndarray) -> np.ndarray:
    rj_path = run_path / "Results" / "runData.json"
    if not rj_path.exists():
        raise FileNotFoundError("runData.json non trovato in {}/Results".format(run_path))
    rj = json.loads(rj_path.read_text(encoding='utf-8'))
    lastMeasureMc = float(rj.get("lastMeasureMC", 0))
    mcMeas        = float(rj.get("mcMeas", 1))
    muAvEnergy    = float(rj.get("muAvEnergy", 0))
    effectiveMuToUse = muAvEnergy if muAvEnergy >= 1500 else 1500.0
    denom = (lastMeasureMc/mcMeas) / (1.0 + 2.0*effectiveMuToUse/mcMeas)
    if denom <= 0:
        raise RuntimeError("Parametri runData non validi per l'errore")
    return np.sqrt(np.clip(chi*(1.0-chi), 0.0, None) / denom)

# ==============
# EXACT FUNCTION COPIED
# ==============
def progressiveLinearFit(x, y, yerr, threshold_chi_square=0.5, onlyEnd=False):

    par_values = []
    minimumShifting = np.maximum(len(x)//150, 5)
    minimumLength = 3*minimumShifting

    def linear(t, a, b):
        return t*a+b

    iStartIndex = np.maximum(np.argmax(y>0.001), minimumShifting)

    if iStartIndex + minimumShifting >= len(x)-1:
        return None
    largestIndexOfTimeLargerThanTminus2 = np.where(x<x[-1]-0.3)[0][-1]

    for i in range(iStartIndex, len(x)-2*minimumShifting, minimumShifting):
        jMin = i+minimumShifting
        if onlyEnd:
            jMin = np.maximum(largestIndexOfTimeLargerThanTminus2, i+minimumLength)
        for j in range(jMin, len(x)-1, minimumShifting):
            
            try:
                popt, pcov = curve_fit(
                    linear, x[i:j], y[i:j],
                sigma=yerr[i:j],
                method='lm',
                p0=[1/x[-1], -0.6],
                maxfev=5000
                )
            except RuntimeError as e:
                import warnings
                warnings.warn(
                f"[progressiveLinearFit] fit fallito per i={i}, j={j}: {e}",
                RuntimeWarning
                )
                continue
            
            slope = popt[0]
            intercept = popt[1]
            chi_r_value = np.nansum(((y[i:j]-(linear(x[i:j],*popt)))/yerr[i:j])**2.)/(j-i)
            if chi_r_value < threshold_chi_square and intercept+slope*x[-1]>0.02:
                par_values.append((chi_r_value, i, j, slope, intercept))

    if len(par_values)==0:
        return None
    best_segment = min(par_values, key=lambda x: x[0]/(x[2]-x[1])**5.)
    best_Chi = best_segment[0]
    terminalParameters= ["m","b"]
    tauIndex= best_segment[1]
    if not(onlyEnd) or (best_segment[4]+best_segment[3]*x[-1])>0.88:
        return terminalParameters, [best_segment[3], best_segment[4]], np.sqrt(pcov[0,0]), [best_segment[1], best_segment[2]], best_Chi, linear
    else:
        return None

nameOfFoldersContainingGraphs = ["fPosJ"
                               ]

_TEXT_EXT = {".dat",".txt",".csv",".tsv"}
def _load_Z_table(fp: Path):
    rows = []
    with open(fp, 'r', encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            s=line.strip()
            if not s or s.startswith(('#','%','//')): continue
            parts = s.split()
            if len(parts) < 2: continue
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    if not rows:
        raise RuntimeError("Z_table vuoto o illeggibile: {}".format(fp))
    arr = np.asarray(rows, float)
    idx = np.argsort(arr[:,0])
    return arr[idx,0], arr[idx,1]

def _auto_find_Z_table(run_path: Path):
    candidates = []
    spots = [
        run_path,
        run_path / "Results",
        run_path / "Results" / "Tables",
        run_path / "Results" / "tables",
        run_path / "Results" / "AveragedData",
        run_path.parent / "Results",
        run_path.parent / "Results" / "Tables",
    ]
    keys = ("z", "thermo", "integr", "ti")
    for sp in spots:
        if not sp.exists() or not sp.is_dir(): 
            continue
        for f in sp.rglob("*"):
            if f.is_file() and f.suffix.lower() in _TEXT_EXT:
                name = f.name.lower()
                if any(k in name for k in keys):
                    candidates.append(f)
    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda p: (len(p.parts), 0 if 'z' in p.name.lower() else 1))
    return candidates[0]

def _nearest(x: np.ndarray, y: np.ndarray, x0: float) -> float:
    k = int(np.argmin(np.abs(x - x0)))
    return float(y[k])


def run_from_specs(specs: List[Dict[str, Any]]) -> None:
    with ustyle.auto_style(mode="latex",
                           base=str(Path(ustyle.__file__).resolve().parent / "styles" / "paper_base.mplstyle"),
                           overlay=str(Path(ustyle.__file__).resolve().parent / "styles" / "overlay_latex.mplstyle")):
        for spec in specs:
            name     = spec.get("name", "ChiVsT_allTrajs")
            run_path = _norm(spec["run_path"])

            t, chi = _load_av_series(run_path)
            chi_err = _chi_err_from_runData(run_path, chi)

            res = progressiveLinearFit(t, chi, chi_err,
                                       threshold_chi_square=float(spec.get("threshold_chi_square", 0.5)),
                                       onlyEnd=bool(spec.get("onlyEnd", False)))
            if res is None:
                raise RuntimeError("progressiveLinearFit ha restituito None")
            terminalParameters, best_fit_params, m_err, fit_idx, chi2r, linear = res
            i_low, i_up = fit_idx[0], fit_idx[1]
            t_low = t[i_low]
            t_end = t[i_up-1] if i_up>i_low else t[i_up]

            Z_ref = spec.get("Z_ref", None)
            Z_source = None
            if Z_ref is None:
                ztab = spec.get("Z_table", "") or ""
                zpath = Path(ztab) if ztab else None
                if not zpath or not zpath.exists():
                    zpath = _auto_find_Z_table(run_path)
                if zpath and zpath.exists():
                    tZ, Z = _load_Z_table(zpath)
                    Z_ref = _nearest(tZ, Z, t_end)
                    Z_source = str(zpath)
                else:
                    Z_ref = 1.0
                    Z_source = "fallback=1.0 (no Z table found)"
            Z_ref = float(Z_ref)

            chi_Z     = chi * Z_ref
            chi_err_Z = chi_err * Z_ref

            figsize = tuple(spec.get("figsize", (6.0, 4.2)))
            fig = plt.figure(name, figsize=figsize)
            ax  = fig.add_subplot(111)
            ax.set_title("")
            ax.errorbar(t, chi_Z, chi_err_Z, elinewidth=0.4, lw=0.0, color='black', alpha=0.9)
            ax.plot(t, chi_Z, lw=1.0, color='black')

            xfit = np.linspace(t.min(), t.max(), 200)
            yfit = (best_fit_params[0]*xfit + best_fit_params[1]) * Z_ref
            ax.plot(xfit, yfit, lw=1.1, color="#1f77b4",
                    label="fit: m={:.3g}, b={:.3g}, chi2r={:.3g}".format(best_fit_params[0], best_fit_params[1], chi2r))

            ax.axvline(t_low, lw=0.8, ls='--', color='#777777')
            ax.axvline(t_end, lw=0.8, ls='--', color='#777777')

            ax.set_xlabel(spec.get("x_label", "t"))
            ax.set_ylabel(spec.get("y_label", r"$\chi \cdot Z$"))

            if bool(spec.get("legend", False)):
                ax.legend(frameon=False, loc='best')

            out = spec.get("outfile") or name.replace(" ","_")
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            uplot.export_figure_strict(fig, out, formats=tuple(spec.get("formats", ["pdf","png"])), dpi=int(spec.get("dpi", 300)))
            plt.close(fig)
            print("Saved figure '{}'.  Z_ref={:.6g} at t≈{:.6g}  [{}]".format(out, Z_ref, t_end, Z_source if Z_source else "spec.Z_ref"))

if __name__ == "__main__":
    specs = PLOTS
    run_from_specs(specs)
