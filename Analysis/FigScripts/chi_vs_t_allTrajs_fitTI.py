#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
chi_vs_t_allTrajs_fitTI_paper_dropin.py — drop-in “paper-ready” per χ(t) su tutte le traiettorie con fit TI.
⟶ Nessuna colorbar. Data-area fissa in pollici, margini assoluti: identico layout ai tuoi altri drop-in.
⟶ Nessun tight/constrained; nessun bbox_inches="tight".
⟶ LaTeX overlay con i tuoi .mplstyle via percorsi assoluti.
⟶ Spessori/ticks coerenti (spines ~0.9 pt; ticks out 3.2 pt/0.8 pt; Y 3–4 tick con MaxNLocator; offset ridotto).
⟶ Log [FIG SIZE]/[AX BOX]/[SCALE] e export via uplot.export_figure_strict (PNG+PDF).
"""

# ==============================
# LISTA PLOT (copiata dall'originale)
# ==============================
specs = [
        {
        "run_path": r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\ER\p2C3\N200\structure388845\fPosJ1.00\graph8865\DataForPathsMC\PathsMCs\80_0.525_0_inf_120_inf_run7201",
        "outfile": "_figs/ChiVsT_allTrajs/ER_HT",
        "only_end": False,
        "figsize": (4.0, 3.0),
        "dpi": 300,
        "formats": ["pdf","png"],
        "x_label": r"$t$",
        "y_label": r"$\chi$",
        "grid": False,
        "use_Z": False,
        "model": "ER",
        "xticks":[0,40,80],
        "yticks":[0,0.2, 0.4,0.6,0.8,1.0],
    },
    {
        "run_path": r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\RRG\p2C3\N160\structure140322\fPosJ1.00\graph340\DataForPathsMC\PathsMCs\80_1_0_inf_96_inf_run9157",
        "outfile": "_figs/ChiVsT_allTrajs/RRG_refnew",
        "only_end": True,
        "figsize": (4.0, 3.0),
        "dpi": 300,
        "formats": ["pdf","png"],
        "x_label": r"$t$",
        "y_label": r"$Z$",
        "grid": False,
        "use_Z": True,
        "model": "RRG",
        "xticks":[0,40,80]
    },
    {
        "run_path": r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\ER\p2C3\N140\structure655712\fPosJ1.00\graph5735\DataForPathsMC\PathsMCs\70_1_0_inf_84_inf_run6751",
        "outfile": "_figs/ChiVsT_allTrajs/ER_refnew",
        "only_end": True,
        "figsize": (4.0, 3.0),
        "dpi": 300,
        "formats": ["pdf","png"],
        "x_label": r"$t$",
        "y_label":  r"$Z$",
        "grid": False,
        "use_Z": True,
        "model": "ER",
        "xticks":[0,35,70]
    },

]

# ==============================
# USER TUNABLE (assoluti, in pollici)
# ==============================
FIG_SCALE   = 1.50  # per impaginazione a 2 colonne: 1.40–1.50
DATA_W_IN   = 1.60  # larghezza data-area
DATA_H_IN   = 1.10  # altezza   data-area

LEFT_IN     = 0.40  # margine sinistro (base)
RIGHT_FRAME = 0.08  # margine destro “cornice”
BOTTOM_IN   = 0.34  # margine inferiore
TOP_IN      = 0.16  # margine superiore

# micro-“bump” per sicurezza (non altera la data-area)
LEFT_BUMP_IN   = 0.02
BOTTOM_BUMP_IN = 0.02
TOP_BUMP_IN    = 0.00
RIGHT_BUMP_IN  = 0.02

# Stile/LaTeX
USE_TEX_MODE   = "latex"  # "latex", "latex_text", "pgf"
STYLE_BASE     = "paper_base.mplstyle"
STYLE_OVERLAY  = "overlay_latex.mplstyle"

# Fit extenders (preservati)
FIT_EXTEND_LEFT  = 0.1
FIT_EXTEND_RIGHT = 15

# ==============================
# Import e setup
# ==============================
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
from matplotlib.ticker import MaxNLocator

# --- helpers path utils (robusto Windows->WSL) ---
def _norm_path(p: str) -> str:
    if p is None: return p
    q = p.replace("\\\\", "\\").replace("\\", "/")
    if len(q) >= 3 and q[1:3] == ":/":
        drive = q[0].lower()
        q = f"/mnt/{drive}/{q[3:]}"
    return q

def _get_file_with_prefix(parent_dir: str, prefix: str) -> Optional[str]:
    if not os.path.isdir(parent_dir): return None
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

# --- progressiveLinearFit: IDENTICO all'originale ---
def progressiveLinearFit(x, y, yerr, threshold_chi_square=0.5, onlyEnd=False):
    par_values = []
    minimumShifting = np.maximum(len(x)//150, 5)
    minimumLength = 3*minimumShifting
    def linear(t, a, b): return t*a+b
    iStartIndex = np.maximum(np.argmax(y>0.001), minimumShifting)
    if iStartIndex + minimumShifting >= len(x)-1: return None
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
                warnings.warn(f"[progressiveLinearFit] fit fallito per i={i}, j={j}: {e}", RuntimeWarning)
                continue
            slope = popt[0]; intercept = popt[1]
            chi_r_value = np.nansum(((y[i:j]-(linear(x[i:j],*popt)))/yerr[i:j])**2.)/(j-i)
            if chi_r_value < threshold_chi_square and intercept+slope*x[-1]>0.02:
                par_values.append((chi_r_value, i, j, slope, intercept))
    if len(par_values)==0: return None
    best_segment = min(par_values, key=lambda x: x[0]/(x[2]-x[1])**5.)
    best_Chi = best_segment[0]
    terminalParameters= ["m","b"]
    tauIndex= best_segment[1]
    if not(onlyEnd) or (best_segment[4]+best_segment[3]*x[-1])>0.88:
        return terminalParameters, [best_segment[3], best_segment[4]], np.sqrt(pcov[0,0]), [best_segment[1], best_segment[2]], best_Chi, linear
    else:
        return None

nameOfFoldersContainingGraphs = ["fPosJ"]

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
from hashlib import sha1
def _make_run_uid(run_dir: Path, graphs_root: Path) -> str:
    try:
        rel = str(run_dir.resolve().relative_to(graphs_root.resolve()))
    except Exception:
        rel = str(run_dir.resolve())
    return sha1(rel.encode("utf-8")).hexdigest()[:16]

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
    sub = sub.sort_values(by=["beta"], ascending=True, na_position="last")
    z = float(sub["ZFromTIBeta"].iloc[-1])
    if not np.isfinite(z) or z <= 0:
        return 1.0, "invalid ZFromTIBeta"
    return z, f"ti_points.parquet:run_uid={run_uid}:beta=max"

# ==============================
# Runner con layout deterministico
# ==============================
def run_from_specs(_specs):
    styles_root = Path(ustyle.__file__).resolve().parent / "styles"
    with ustyle.auto_style(mode=USE_TEX_MODE,
                           base=str(styles_root / STYLE_BASE),
                           overlay=str(styles_root / STYLE_OVERLAY)):
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

            # ====================== GEOMETRIA DETERMINISTICA (NO COLORBAR) ======================
            L0 = LEFT_IN + LEFT_BUMP_IN
            B0 = BOTTOM_IN + BOTTOM_BUMP_IN
            T0 = TOP_IN + TOP_BUMP_IN
            R0 = RIGHT_FRAME + RIGHT_BUMP_IN

            L = L0 * FIG_SCALE; B = B0 * FIG_SCALE; T = T0 * FIG_SCALE; R_frame = R0 * FIG_SCALE
            DW = DATA_W_IN * FIG_SCALE; DH = DATA_H_IN * FIG_SCALE

            fig_w = L + DW + R_frame
            fig_h = B + DH + T

            dpi = int(s.get("dpi", 300))

            # Figura con area dati ESATTA
            fig, ax, _meta = uplot.figure_single_fixed(
                data_w_in=DW, data_h_in=DH,
                left_in=L, right_in=R_frame,
                bottom_in=B, top_in=T
            )
            fig.set_constrained_layout(False)
            try: fig.tight_layout = lambda *a, **k: None
            except Exception: pass

            # Spessori / ticks coerenti e labelpad contenuti
            for sp in ax.spines.values():
                sp.set_linewidth(0.9)
            ax.tick_params(direction='out', length=3.2, width=0.8, pad=2.0)
            ax.xaxis.labelpad = 1.2
            ax.yaxis.labelpad = 1.2

            # === DRAW ===
            ax.plot(t, avChi*Z_ref, color="black", linewidth=1.2, zorder=2)
            ax.errorbar(t, avChi*Z_ref, yerr=yerr*Z_ref, fmt=" ", elinewidth=0.4, alpha=0.1, color="tab:blue", zorder=1)

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
                ax.plot(xx, yy, "--", linewidth=1.2, color="#E90000", zorder=3)

                ax.axvline(tL, linestyle="--", linewidth=0.9, color="#7D7D7D", zorder=2)
                ax.axvline(tR, linestyle="--", linewidth=0.9, color="#7D7D7D", zorder=2)
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

            # Y ticks/offset coerenti con gli altri pannelli
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
            try:
                ax.yaxis.get_offset_text().set_fontsize(ax.yaxis.get_offset_text().get_fontsize()*0.72)
            except Exception:
                pass
            xTicks=s.get("xticks",None)
            if xTicks is not None:
                ax.xaxis.set_ticks(xTicks)
                ax.xaxis.set_ticklabels([str(x) for x in xTicks])
            yTicks=s.get("yticks",None)
            if yTicks is not None:
                ax.yaxis.set_ticks(yTicks)
                ax.yaxis.set_ticklabels([str(y) for y in yTicks])

            out_base = Path(s.get("outfile", "_figs/ChiVsT_allTrajs/output"))
            out_base.parent.mkdir(parents=True, exist_ok=True)

            # === LOG GEOMETRIA PRIMA DEL SALVATAGGIO ===
            W, H = fig.get_size_inches()
            print(f"[FIG SIZE] chi_vs_t_allTrajs: {W:.3f} × {H:.3f} in")
            bbox = ax.get_position(); data_w = bbox.width * W; data_h = bbox.height * H
            right_total = R_frame
            print(f"[AX BOX]   data: {data_w:.3f} × {data_h:.3f} in; margins L/R/B/T={L:.2f}/{right_total:.2f}/{B:.2f}/{T:.2f}")
            print(f"[SCALE]     FIG_SCALE={FIG_SCALE:.2f} ⇒ DW×DH={DW:.2f}×{DH:.2f} in")

            # Export (senza tight)
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
                "figsize": [W,H],
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
