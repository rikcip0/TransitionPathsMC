#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==============================
# LISTA PLOT (in testa al file)
# ==============================
specs = [
    {
        "run_path": r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\RRG\p2C3\N120\structure194235\fPosJ1.00\graph4245\DataForPathsMC\PathsMCs\70_1_0_inf_72_inf_run5798",
        "outfile": "__figs/ChiVsT_allTrajs/RRG_ref",
        "only_end": True,
        "figsize": (4.0, 3.0),
        "dpi": 300,
        "formats": ["pdf","png"],
        "x_label": "t",
        "y_label": r"$\chi$",
        "grid": False,
        "use_Z": True,
        "model": "RRG"
    },
    {
        "run_path": r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\ER\p2C3\N160\structure998785\fPosJ1.00\graph8797\DataForPathsMC\PathsMCs\45_1_0_inf_96_inf_run8927",
        "outfile": "__figs/ChiVsT_allTrajs/ER_ref",
        "only_end": True,
        "figsize": (4.0, 3.0),
        "dpi": 300,
        "formats": ["pdf","png"],
        "x_label": "t",
        "y_label": r"$\chi$",
        "grid": False,
        "use_Z": True,
        "model": "ER"
    },
    {
        "run_path": r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\ER\p2C3\N140\structure655712\fPosJ1.00\graph5735\DataForPathsMC\PathsMCs\70_1_0_inf_84_inf_run6751",
        "outfile": "__figs/ChiVsT_allTrajs/ER_ref2",
        "only_end": True,
        "figsize": (4.0, 3.0),
        "dpi": 300,
        "formats": ["pdf","png"],
        "x_label": "t",
        "y_label": r"$\chi$",
        "grid": False,
        "use_Z": True,
        "model": "ER"
    },
]

# ==============================
# TUNABLES (facili da editare)
# ==============================
FIT_EXTEND_LEFT  = 0.10
FIT_EXTEND_RIGHT = 0.10

import os, sys, json, hashlib
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# FigCore style (se manca, deve fallire: voluto)
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2] if len(HERE.parents) >= 3 else HERE.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.append('../')
from MyBasePlots.FigCore import utils_style as ustyle
from MyBasePlots.FigCore import utils_plot  as uplot

# --- helpers path utils (robusto Windows->WSL) ---
def _norm_path(p: str) -> str:
    if p is None:
        return p
    # normalizza backslash in slash e compatta eventuale doppio slash iniziale
    q = p.replace("\\\\", "\\").replace("\\", "/")
    # su WSL: se inizia con "C:/" -> mappa a /mnt/c/...
    if len(q) >= 3 and q[1:3] == ":/":
        drive = q[0].lower()
        q = f"/mnt/{drive}/{q[3:]}"
    return q

def _get_file_with_prefix(parent_dir: str, prefix: str) -> Optional[str]:
    if not os.path.isdir(parent_dir):
        return None
    for f in os.listdir(parent_dir):
        fp = os.path.join(parent_dir, f)
        if os.path.isfile(fp) and f.startswith(prefix):
            return fp
    return None

def _find_graphs_root_from_run(run_path: str) -> Path:
    p = Path(run_path).resolve()
    for anc in [p] + list(p.parents):
        cur = anc
        for _ in range(8):
            dg = cur / "Data" / "Graphs"
            if dg.exists() and dg.is_dir():
                return dg
            cur = cur.parent
    raise RuntimeError("Graphs root (Data/Graphs) non trovata risalendo dal run_path.")

def _default_outdir_for(graphs_root: Path) -> Path:
    return graphs_root.parent / "MultiPathsMC"

def _load_av_table(run_path: str):
    av_file = _get_file_with_prefix(run_path, "av")
    if av_file is None:
        raise FileNotFoundError(f"av.dat non trovato in {run_path}")
    with open(av_file, "r", encoding="utf-8", errors="ignore") as fh:
        lines = fh.readlines()
    data_lines = [ln for ln in lines if not ln.lstrip().startswith("#")]
    data = np.genfromtxt(data_lines, delimiter=" ")
    t = data[:, 0]
    avChi = data[:, 4]
    return t, avChi

# --- progressiveLinearFit: IDENTICO al tuo ---
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

def _load_run_json_bits(run_path: str):
    run_json = os.path.join(run_path, "Results", "runData.json")
    if not os.path.isfile(run_json):
        raise FileNotFoundError(f"runData.json non trovato in {run_json}")
    with open(run_json, "r", encoding="utf-8") as fh:
        J = json.load(fh)
    last_measure = int(J.get("lastMeasureMC"))
    mc_meas = int(J["configuration"]["mcParameters"]["MCmeas"])
    mu = None
    try:
        mu = J["results"]["thermalization"]["avEnergy"]["mu"]
    except Exception:
        try:
            mu = J["results"]["thermalization"]["H"]["mu"]
        except Exception:
            pass
    if mu is None or (isinstance(mu, str) and mu == "nan"):
        raise ValueError("muAvEnergy non trovato in runData.json (results.thermalization.avEnergy.mu).")
    mu = float(mu)
    return last_measure, mc_meas, mu

def _compute_avchi_err(avChi, lastMeasureMc, mcMeas, muAvEnergy):
    effectiveMu = muAvEnergy if muAvEnergy >= 1500 else 1500
    denom = (lastMeasureMc/mcMeas)/(1.0 + 2.0*effectiveMu/mcMeas)
    return np.sqrt(avChi*(1.0-avChi)/denom)

# --------- Z deterministica via run_uid + ti_points.parquet (ZFromTIBeta) ---------
def _make_run_uid(run_dir: Path, graphs_root: Path) -> str:
    try:
        rel = str(run_dir.resolve().relative_to(graphs_root.resolve()))
    except Exception:
        rel = str(run_dir.resolve())
    return hashlib.sha1(rel.encode("utf-8")).hexdigest()[:16]

def _get_Z_ref_for_run(run_path: str, model: str) -> tuple[float, str]:
    graphs_root = _find_graphs_root_from_run(run_path)
    outdir = _default_outdir_for(graphs_root)
    ti_points = outdir / model / "v1" / "ti" / "ti_points.parquet"
    if not ti_points.exists():
        return 1.0, "ti_points.parquet not found"
    run_uid = _make_run_uid(Path(run_path), graphs_root)
    import pandas as pd, numpy as np
    df = pd.read_parquet(ti_points)
    if "run_uid" not in df.columns or "ZFromTIBeta" not in df.columns:
        return 1.0, "ti_points missing run_uid or ZFromTIBeta"
    sub = df.loc[df["run_uid"].astype(str) == run_uid]
    if sub.empty:
        return 1.0, "run_uid not found in ti_points"
    # usa la riga a beta massimo per quella run (deterministico)
    sub = sub.sort_values(by=["beta"], ascending=True, na_position="last")
    z = float(sub["ZFromTIBeta"].iloc[-1])
    if not np.isfinite(z) or z <= 0:
        return 1.0, "invalid ZFromTIBeta"
    return z, f"ti_points.parquet:run_uid={run_uid}:beta=max"

def run_from_specs(_specs):
    with ustyle.auto_style(mode="latex",
                           base=str(Path(ustyle.__file__).resolve().parent / "styles" / "paper_base.mplstyle"),
                           overlay=str(Path(ustyle.__file__).resolve().parent / "styles" / "overlay_latex.mplstyle")):
        for s in _specs:
            raw_path = s["run_path"]
            run_path = _norm_path(raw_path)
            if not os.path.isdir(run_path):
                raise FileNotFoundError(f"Cartella run non trovata:\n  {raw_path}\nNormalizzato a:\n  {run_path}")

            t, avChi = _load_av_table(run_path)

            lastMeasureMc, mcMeas, muAvEnergy = _load_run_json_bits(run_path)
            yerr = _compute_avchi_err(avChi, lastMeasureMc, mcMeas, muAvEnergy)

            only_end = bool(s.get("only_end", True))
            fit = progressiveLinearFit(t, avChi, yerr, threshold_chi_square=0.5, onlyEnd=only_end)

            Z_ref = 1.0
            Z_source = "none"
            if bool(s.get("use_Z", False)):
                Z_ref, Z_source = _get_Z_ref_for_run(run_path, s.get("model",""))

            figsize = tuple(s.get("figsize", (4.0,3.0)))
            dpi = int(s.get("dpi", 300))
            fig = plt.figure(figsize=figsize, dpi=dpi)
            ax = fig.add_subplot(111)

            ax.plot(t, avChi*Z_ref, color="black", linewidth=1.0)
            ax.errorbar(t, avChi*Z_ref, yerr=yerr*Z_ref, fmt=" ", elinewidth=0.4, alpha=0.3, color="black")

            meta = {}

            if fit is not None:
                names, params, m_err, (i_low, i_high), chi_r, f_linear = fit
                m, b = params
                tL = float(t[i_low]); tR = float(t[i_high-1])
                Lw = max(1e-12, tR - tL)

                xlim_before = ax.get_xlim()
                extra_L = max(0.0, float(FIT_EXTEND_LEFT)) * Lw
                extra_R = max(0.0, float(FIT_EXTEND_RIGHT)) * Lw
                x0 = max(float(t.min()), tL - extra_L)
                x1 = min(float(t.max()), tR + extra_R)

                xx = np.linspace(x0, x1, 256)
                yy = (m*xx + b) * Z_ref
                ax.plot(xx, yy, "--", linewidth=0.9, color="#666666")

                ax.axvline(tL, linestyle="--", linewidth=0.9, color="#999999")
                ax.axvline(tR, linestyle="--", linewidth=0.9, color="#999999")
                ax.set_xlim(xlim_before)

                meta.update({
                    "fit_interval_idx": [int(i_low), int(i_high)],
                    "fit_interval_t": [tL, tR],
                    "m": float(m), "b": float(b),
                    "m_err": float(m_err) if m_err is not None else None,
                    "chi2_r": float(chi_r),
                })

            xl = s.get("x_label", "")
            yl = s.get("y_label", "")
            if xl: ax.set_xlabel(xl)
            if yl: ax.set_ylabel(yl)
            if bool(s.get("grid", False)):
                ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)

            out_base = s.get("outfile", "__figs/ChiVsT_allTrajs/output")
            out_base = Path(out_base)
            out_base.parent.mkdir(parents=True, exist_ok=True)

            formats = list(s.get("formats", ["pdf","png"]))
            for ext in formats:
                uplot.export_figure_strict(fig, str(out_base), formats=(ext,), dpi=dpi)

            plt.close(fig)

            meta.update({
                "run_path": raw_path,
                "only_end": only_end,
                "lastMeasureMc": int(lastMeasureMc),
                "mcMeas": int(mcMeas),
                "muAvEnergy": float(muAvEnergy),
                "figsize": list(figsize),
                "dpi": dpi,
                "outfile_base": str(out_base),
                "saved_formats": formats,
                "x_label": xl,
                "y_label": yl,
                "grid": bool(s.get("grid", False)),
                "use_Z": bool(s.get("use_Z", False)),
                "Z_ref": float(Z_ref),
                "Z_source": Z_source
            })
            meta_path = str(out_base) + "_meta.json"
            Path(meta_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(f"[ok] saved: {out_base}.[{','.join(formats)}]  (Z_ref={Z_ref} from {Z_source})")
            print(f"[ok] meta:  {meta_path}")

if __name__ == "__main__":
    run_from_specs(specs)
