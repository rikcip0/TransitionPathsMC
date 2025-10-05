#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================
# LISTA PLOT (IN TESTA)
# =====================
specs = [
    {
        "model": "realGraphs/ZKC",
        "runs": [
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\40_0.9_0_inf_20_inf_run2620",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\35_0.9_0_inf_20_inf_run6950",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\30_0.9_0_inf_20_inf_run7081",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\25_0.9_0_inf_20_inf_run7079",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\20_0.9_0_inf_20_inf_run7077",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\15_0.9_0_inf_20_inf_run7083",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\10_0.9_0_inf_20_inf_run7085",
        ],
        "outfile": "_figs/Z_withChiCurves/ZKC_ZandChi_bet0p9_3States",
        "overlay_chi": "tmax",              # "tmax" | "all" | "none"
        "fit_model": "cinematic_chi",            # "linear_Z" | "saturation_Z" | "quad_chi" | "cinematic_chi" | "none"
        "chi_scale": "Z",                   # "Z" | "none"
        "figsize": (4.2, 3.2),
        "dpi": 300,
        "formats": ["pdf", "png"],
        "grid": False,
        "x_label": r"$t$  (punti a $T$)",
        "y_label": r"$Z(T)$  e  $Z\cdot\chi(t)$",
        "plot_tau_vertical": True,          # NEW: disegna la linea verticale a tau*
    },
    {
        "model": "realGraphs/ZKC",
        "runs": [
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\40_0.55_0_inf_20_inf_run2634",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\35_0.55_0_inf_20_inf_run8308",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\30_0.55_0_inf_20_inf_run9935",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\25_0.55_0_inf_20_inf_run42",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\20_0.55_0_inf_20_inf_run9742",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\15_0.55_0_inf_20_inf_run8772",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\10_0.55_0_inf_20_inf_run4146",
        ],
        "outfile": "_figs/Z_withChiCurves/ZKC_ZandChi_bet0p55_Quadratic",
        "overlay_chi": "tmax",
        "fit_model": "quad_chi",
        "chi_scale": "Z",
        "figsize": (4.2, 3.2),
        "dpi": 300,
        "formats": ["pdf", "png"],
        "grid": False,
        "x_label": r"$t$  (punti a $T$)",
        "y_label": r"$Z(T)$  e  $Z\cdot\chi(t)$",
        "plot_tau_vertical": True,          # NEW
    },
    {
        "model": "realGraphs/ZKC",
        "runs": [
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\40_0.9_0_inf_20_inf_run2620",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\35_0.9_0_inf_20_inf_run6950",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\30_0.9_0_inf_20_inf_run7081",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\25_0.9_0_inf_20_inf_run7079",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\20_0.9_0_inf_20_inf_run7077",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\15_0.9_0_inf_20_inf_run7083",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\10_0.9_0_inf_20_inf_run7085",
        ],
        "outfile": "_figs/Z_withChiCurves/ZKC_ZandChi_bet0p9",
        "overlay_chi": "tmax",              # "tmax" | "all" | "none"
        "fit_model": "quad_chi",            # "linear_Z" | "saturation_Z" | "quad_chi" | "cinematic_chi" | "none"
        "chi_scale": "Z",                   # "Z" | "none"
        "figsize": (4.2, 3.2),
        "dpi": 300,
        "formats": ["pdf", "png"],
        "grid": False,
        "x_label": r"$t$  (punti a $T$)",
        "y_label": r"$Z(T)$  e  $Z\cdot\chi(t)$",
        "plot_tau_vertical": True,          # NEW: disegna la linea verticale a tau*
    },
    {
        "model": "realGraphs/ZKC",
        "runs": [
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\40_0.55_0_inf_20_inf_run2634",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\35_0.55_0_inf_20_inf_run8308",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\30_0.55_0_inf_20_inf_run9935",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\25_0.55_0_inf_20_inf_run42",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\20_0.55_0_inf_20_inf_run9742",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\15_0.55_0_inf_20_inf_run8772",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\10_0.55_0_inf_20_inf_run4146",
        ],
        "outfile": "_figs/Z_withChiCurves/ZKC_ZandChi_bet0p55",
        "overlay_chi": "tmax",
        "fit_model": "cinematic_chi",
        "chi_scale": "Z",
        "figsize": (4.2, 3.2),
        "dpi": 300,
        "formats": ["pdf", "png"],
        "grid": False,
        "x_label": r"$t$  (punti a $T$)",
        "y_label": r"$Z(T)$  e  $Z\cdot\chi(t)$",
        "plot_tau_vertical": True,          # NEW
    },
    {
        "model": "realGraphs/ZKC",
        "runs": [
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\40_0.35_0_inf_20_inf_run2642",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\35_0.35_0_inf_20_inf_run5103",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\30_0.35_0_inf_20_inf_run9927",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\25_0.35_0_inf_20_inf_run34",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\20_0.35_0_inf_20_inf_run3494",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\15_0.35_0_inf_20_inf_run8764",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\10_0.35_0_inf_20_inf_run4138",
        ],
        "outfile": "_figs/Z_withChiCurves/ZKC_ZandChi_bet0p35",
        "overlay_chi": "tmax",
        "fit_model": "linear_Z",
        "chi_scale": "Z",
        "figsize": (4.2, 3.2),
        "dpi": 300,
        "formats": ["pdf", "png"],
        "grid": False,
        "x_label": r"$t$  (punti a $T$)",
        "y_label": r"$Z(T)$  e  $Z\cdot\chi(t)$",
        "plot_tau_vertical": True,          # NEW
    },
    {
        "model": "realGraphs/ZKC",
        "runs": [
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\40_0.25_0_inf_20_inf_run6871",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\35_0.25_0_inf_20_inf_run3840",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\30_0.25_0_inf_20_inf_run6865",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\25_0.25_0_inf_20_inf_run6859",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\20_0.25_0_inf_20_inf_run9958",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\15_0.25_0_inf_20_inf_run8758",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\10_0.25_0_inf_20_inf_run4134",
        ],
        "outfile": "_figs/Z_withChiCurves/ZKC_ZandChi_bet0p25",
        "overlay_chi": "tmax",
        "fit_model": "saturation_Z",
        "chi_scale": "Z",
        "figsize": (4.2, 3.2),
        "dpi": 300,
        "formats": ["pdf", "png"],
        "grid": False,
        "x_label": r"$t$  (punti a $T$)",
        "y_label": r"$Z(T)$  e  $Z\cdot\chi(t)$",
        "plot_tau_vertical": True,          # NEW
    },
]

import os, sys, json, hashlib, numpy as np
from pathlib import Path
from typing import List, Dict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import least_squares

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2] if len(HERE.parents) >= 3 else HERE.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.append('../')
from MyBasePlots.FigCore import utils_style as ustyle
from MyBasePlots.FigCore import utils_plot  as uplot

# === colonne χ in av*.dat
CHI_COL = 4
CHI_ERR_COL = 5
EPS = 1e-12
TMIN_CHI = 0.0
# Se True, TMIN_CHI funge ANCHE da bound inferiore per tau nei fit su χ(t)
ENFORCE_TMIN_CHI_ON_TAU = True

# === politica per sigma(Z): 'from_table' | 'binomial' | 'uniform'
Z_SIGMA_POLICY = "from_table"
Z_SIGMA_COLS_ORDER = ["ZFromTIBetaStd","ZFromTIBeta_sigma","Z_sigma","sigma_Z","err_Z","ZFromTIBetaErr"]

# === colori
COL_Z_POINTS = "black"
COL_CHI_CURVE = "blue"
COL_FIT = "red"
COL_TAU = "0.6"

# === forza shift prima del primo T (per includere tutti i punti in Z-fit)
FORCE_SHIFT_BEFORE_FIRST = True

# ----------------- helpers path -----------------
def _norm_path(p: str) -> str:
    if p is None: return p
    q = p.replace('\\\\', '\\').replace('\\', '/')
    if len(q) >= 3 and q[1:3] == ":/":
        drive = q[0].lower(); q = f"/mnt/{drive}/{q[3:]}"
    return q

def _find_graphs_root_from_run(run_path: str, override: str = None) -> Path:
    if override:
        gr = Path(_norm_path(override)).resolve()
        if not gr.exists(): raise RuntimeError(f"graphs_root override non esiste: {gr}")
        return gr
    p = Path(_norm_path(run_path)).resolve()
    for anc in [p] + list(p.parents):
        cur = anc
        for _ in range(20):
            dg = cur / "Data" / "Graphs"
            if dg.exists() and dg.is_dir(): return dg
            cur = cur.parent
    raise RuntimeError("Graphs root (Data/Graphs) non trovata. Usa 'graphs_root' in specs.")

def _tables_root(graphs_root: Path, model: str) -> Path:
    return (graphs_root.parent / "MultiPathsMC" / model / "v1").resolve()

def _make_run_uid(run_dir: Path, graphs_root: Path) -> str:
    try: rel = str(run_dir.resolve().relative_to(graphs_root.resolve()))
    except Exception: rel = str(run_dir.resolve())
    return hashlib.sha1(rel.encode("utf-8")).hexdigest()[:16]

# ----------------- carico tabelle -----------------
def _load_ti_points_for_runs(model: str, runs: List[str], graphs_root_override: str = None):
    if not runs: raise ValueError("Lista 'runs' vuota.")
    graphs_root = _find_graphs_root_from_run(_norm_path(runs[0]), graphs_root_override)
    base = _tables_root(graphs_root, model)
    tip_path = base / "ti" / "ti_points.parquet"
    if not tip_path.exists(): raise FileNotFoundError(f"ti_points.parquet non trovato: {tip_path}")
    df = pd.read_parquet(tip_path)
    want = {_make_run_uid(Path(_norm_path(r)), graphs_root) for r in runs}
    df = df[df["run_uid"].astype(str).isin(want)].copy()
    if df.empty: raise RuntimeError("Nessuna riga in ti_points per i run forniti.")
    return df, tip_path, graphs_root

def _load_runs_params_T(model: str, graphs_root: Path, run_uids: List[str]):
    base = _tables_root(graphs_root, model)
    rp_path = base / "runs_params" / "runs_params.parquet"
    if not rp_path.exists(): raise FileNotFoundError(f"runs_params.parquet non trovato: {rp_path}")
    df = pd.read_parquet(rp_path)
    if "run_uid" not in df.columns: raise RuntimeError("runs_params.parquet: manca 'run_uid'.")
    pref = ["T","totalTime","total_time","T_end","t_end","last_t"]
    T_cols = [c for c in pref if c in df.columns]
    if not T_cols: raise RuntimeError(f"runs_params.parquet: nessuna colonna T tra {pref}.")
    src_col = T_cols[0]
    dfT = df[["run_uid", src_col]].copy()
    dfT["T"] = dfT[src_col].astype(float)
    dfT = dfT[["run_uid","T"]]
    dfT = dfT[dfT["run_uid"].astype(str).isin([str(u) for u in run_uids])].copy()
    if dfT.empty or dfT["T"].isna().any() or (~np.isfinite(dfT["T"])).any():
        raise RuntimeError("runs_params: T mancante/non finito per uno o più run richiesti.")
    return dfT, rp_path, src_col

def _Z_at_beta_max_and_sigma(df_ti: pd.DataFrame, uid: str):
    sub = df_ti[df_ti["run_uid"].astype(str) == uid]
    if sub.empty: return np.nan, np.nan, None
    row = sub.loc[sub["beta"].idxmax()]
    z = float(row["ZFromTIBeta"])
    sigma_used = None
    if Z_SIGMA_POLICY == "from_table":
        for col in Z_SIGMA_COLS_ORDER:
            if col in sub.columns:
                sigma_used = col
                s = float(sub.loc[row.name, col])
                return z, s, sigma_used
    if Z_SIGMA_POLICY == "binomial":
        p = np.clip(z, EPS, 1.0-EPS); return z, float(np.sqrt(p*(1.0-p))), "binomial"
    return z, 1.0, "uniform"

def _load_av_series_with_err(run_path: str):
    parent = Path(_norm_path(run_path))
    av_file = None
    for f in parent.iterdir():
        if f.is_file() and f.name.startswith("av"):
            av_file = f; break
    if av_file is None: raise FileNotFoundError(f"av*.dat non trovato in {parent}")
    rows = []
    with open(av_file, "r", encoding="utf-8", errors="ignore") as fh:
        for ln in fh:
            s = ln.strip()
            if not s or s.startswith(("#","%","//")): continue
            parts = s.split()
            try: rows.append([float(v) for v in parts])
            except ValueError: continue
    arr = np.asarray(rows, float)
    if arr.ndim != 2 or arr.shape[1] <= CHI_COL: raise RuntimeError(f"Formato inatteso per {av_file}")
    t = arr[:,0].astype(float)
    chi = arr[:,CHI_COL].astype(float)
    chi_err = arr[:,CHI_ERR_COL].astype(float) if arr.shape[1] > CHI_ERR_COL else None
    good = np.isfinite(t) & np.isfinite(chi)
    t, chi = t[good], chi[good]
    if chi_err is not None:
        chi_err = chi_err[good]
        if not np.any(np.isfinite(chi_err)): chi_err = None
    return t, chi, chi_err, str(av_file)

def _sigma_from_prob_or_given(y, yerr):
    return np.clip(
        np.sqrt(np.clip(y, EPS, 1.0 - EPS) * (1.0 - np.clip(y, EPS, 1.0 - EPS))),
        1e-9,
        np.inf,
    ) if yerr is None else np.clip(
        np.where(
            np.isfinite(yerr) & (yerr > 0),
            yerr,
            np.sqrt(np.clip(y, EPS, 1.0 - EPS) * (1.0 - np.clip(y, EPS, 1.0 - EPS)))
        ),
        1e-9,
        np.inf,
    )

# ============================== MODELLI (una sola volta) ==============================
def model_linear_Z(t, k, tau, C):
    t = np.asarray(t, float); return k * (t - tau) + C

def model_saturation_Z(t, c, m, s):
    t = np.asarray(t, float)
    out = np.zeros_like(t)
    tt = t - s
    mask = tt > 0
    out[mask] = c * (1.0 - np.exp(-m * tt[mask]))
    return out

def model_quad_chi(t, a, tau, C):
    t = np.asarray(t, float); tt = t - tau
    return a * tt**2 + C

def model_cinematic_chi(t, r, k, tau, C):
    t = np.asarray(t, float); tt = t - tau
    return (r/(2.0*r+1.0)) - 0.5*np.exp(-k*tt) + (1.0/(2.0*(2.0*r+1.0)))*np.exp(-k*(2.0*r+1.0)*tt) + C

# ============================== UTIL: p0 robusti ==============================
def _wls_slope_intercept(x, y, sigma):
    w = 1.0 / np.clip(sigma, 1e-9, np.inf)**2
    X = np.vstack([x, np.ones_like(x)]).T
    WX = X * w[:,None]
    Wy = y * w
    beta, *_ = np.linalg.lstsq(WX, Wy, rcond=None)
    slope, intercept = float(beta[0]), float(beta[1])
    return slope, intercept

def _guess_tau_from_rise(t, y):
    # trova il primo t in cui y ha superato il 2% della dinamica
    y0, y1 = float(np.min(y)), float(np.max(y))
    span = max(y1 - y0, 1e-12)
    thr = y0 + 0.02 * span
    idx = np.argmax(y >= thr)
    tau0 = float(np.clip(t[idx], t.min(), t.max()))
    return tau0

def _guess_k_from_tail_decay(t, y, plateau):
    # stima k su coda: log(plateau - y) ~ -k*t + const
    resid = plateau - y
    mask = (resid > 1e-12)
    print(t)
    if np.sum(mask) < 5: 
        return 1.0 / max(t[-1]-t[0], 1.0)
    xt = t[mask]
    yt = np.log(resid[mask])
    # usa ultimi 30% dei punti se possibile
    m2 = np.arange(xt.size) >= int(0.7 * xt.size)
    if np.sum(m2) >= 5:
        xt, yt = xt[m2], yt[m2]
    slope, _ = _wls_slope_intercept(xt, yt, np.ones_like(yt))
    k = max(-slope, 1e-12)
    return k

# ============================== FITS (logica invariata; p0 migliorati) ==============================
def _stderr_from_jac(res, dof):
    try:
        J = res.jac
        JTJ = J.T @ J
        s2 = 2.0 * res.cost / max(dof, 1)
        cov = np.linalg.pinv(JTJ) * s2
        se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
        return se, cov
    except Exception:
        return None, None

def do_fit_quadratic_chi(time_fit, avChi_fit, avChiErr_fit):
    w_all = 1.0 / avChiErr_fit**2
    # tau0 dal "punto di risalita"
    tau0 = max(_guess_tau_from_rise(time_fit, avChi_fit), TMIN_CHI if ENFORCE_TMIN_CHI_ON_TAU else -np.inf)
    # a0 dalla dinamica complessiva
    span_t = max(time_fit.max() - tau0, 1e-6)
    a0 = max((avChi_fit.max() - avChi_fit.min()) / (span_t**2), 1e-12)
    log_a0 = np.log(a0)
    p0_q   = np.array([log_a0, tau0])
    # bounds
    tau_min = time_fit.min() - (time_fit.max() - time_fit.min())
    if ENFORCE_TMIN_CHI_ON_TAU:
        tau_min = max(tau_min, TMIN_CHI)
    tau_max = time_fit.max()
    lower_q = np.array([-40.0, tau_min])
    upper_q = np.array([ 40.0, tau_max])
    def residuals(p):
        log_a, tau = p
        a = np.exp(log_a)
        pred = model_quad_chi(time_fit, a, tau, 0.0)
        tt = time_fit - tau
        m  = tt > 0
        if m.sum() < 2: return np.full(time_fit.size, 1e6)
        base_m = pred[m]
        C = np.sum(w_all[m] * (avChi_fit[m] - base_m)) / np.sum(w_all[m])
        res = (pred + C - avChi_fit) / avChiErr_fit
        res[~m] = 0.0
        return res
    res = least_squares(residuals, p0_q, bounds=(lower_q, upper_q), max_nfev=3_000_000)
    log_a_fit, tau_fit = res.x
    a_fit = float(np.exp(log_a_fit))
    tt = time_fit - tau_fit; m = tt > 0
    base = model_quad_chi(time_fit, a_fit, tau_fit, 0.0)
    C_fit = float(np.sum(w_all[m] * (avChi_fit[m] - base[m])) / np.sum(w_all[m]))
    dof = max(1, m.sum() - 2)
    chi2r = float(2*res.cost / dof)
    se_vec, cov = _stderr_from_jac(res, dof)
    stderr_map = {}
    if se_vec is not None:
        se_log_a, se_tau = float(se_vec[0]), float(se_vec[1])
        stderr_map = {
            "a": abs(a_fit)*se_log_a,
            "tau": se_tau,
            "C": (1.0/np.sqrt(np.sum(w_all[m]))) if np.isfinite(np.sum(w_all[m])) and np.sum(w_all[m])>0 else None,
        }
    return {"model":"quad_chi","tau": float(tau_fit),"a":a_fit,"C":C_fit,"chi2r": chi2r,
            "stderr_params": stderr_map, "cov": cov.tolist() if cov is not None else None}

def do_fit_cinematic_chi(time_fit, avChi_fit, avChiErr_fit):
    w_all = 1.0 / avChiErr_fit**2
    # plateau ~ media degli ultimi punti
    y_tail = np.mean(avChi_fit[-min(20, len(avChi_fit)):])
    r0  = max(y_tail / max(1 - 2 * y_tail, 1e-6), 2.0)
    # k0 dalla coda
    k0  = _guess_k_from_tail_decay(time_fit, avChi_fit, plateau=y_tail)
    tau0 = max(_guess_tau_from_rise(time_fit, avChi_fit), TMIN_CHI if ENFORCE_TMIN_CHI_ON_TAU else -np.inf)
    tau_min = time_fit.min() - (time_fit.max() - time_fit.min())
    if ENFORCE_TMIN_CHI_ON_TAU:
        tau_min = max(tau_min, TMIN_CHI)
    tau_max = time_fit.max()
    def residuals(p):
        log_r, log_k, tau = p
        r = np.exp(log_r); k = np.exp(log_k)
        pred = model_cinematic_chi(time_fit, r, k, tau, 0.0)
        tt = time_fit - tau
        m  = tt > 0
        if m.sum() < 3: return np.full(time_fit.size, 1e6)
        base_m = pred[m]
        C = np.sum(w_all[m] * (avChi_fit[m] - base_m)) / np.sum(w_all[m])
        res = (pred + C - avChi_fit) / avChiErr_fit
        res[~m] = 0.0
        return res
    log_r_min, log_r_max = 0.8, 80.0
    log_k_min, log_k_max = np.log(1e-12), np.log(1e-3)
    log_r0 = np.clip(np.log(r0), log_r_min+1e-12, log_r_max-1e-12)
    log_k0 = np.clip(np.log(max(k0, 1e-12)), log_k_min+1e-12, log_k_max-1e-12)
    p0    = np.array([log_r0, log_k0, tau0])
    lower = np.array([log_r_min, log_k_min, tau_min])
    upper = np.array([log_r_max, log_k_max, tau_max])
    res = least_squares(residuals, p0, bounds=(lower, upper), max_nfev=3_000_000)
    log_r_fit, log_k_fit, tau_fit = res.x
    r_fit, k_fit = float(np.exp(log_r_fit)), float(np.exp(log_k_fit))
    tt = time_fit - tau_fit; m = tt > 0
    base = model_cinematic_chi(time_fit, r_fit, k_fit, tau_fit, 0.0)
    C_fit = float(np.sum(w_all[m] * (avChi_fit[m] - base[m])) / np.sum(w_all[m]))
    dof = max(1, m.sum() - 3)
    chi2r = float(2*res.cost / dof)
    se_vec, cov = _stderr_from_jac(res, dof)
    stderr_map = {}
    if se_vec is not None:
        se_log_r, se_log_k, se_tau = float(se_vec[0]), float(se_vec[1]), float(se_vec[2])
        stderr_map = {
            "r": abs(r_fit)*se_log_r,
            "k": abs(k_fit)*se_log_k,
            "tau": se_tau,
            "C": (1.0/np.sqrt(np.sum(w_all[m]))) if np.isfinite(np.sum(w_all[m])) and np.sum(w_all[m])>0 else None,
        }
    return {"model":"cinematic_chi","tau": float(tau_fit),"r":r_fit,"k":k_fit,"C":C_fit,"chi2r": chi2r,
            "stderr_params": stderr_map, "cov": cov.tolist() if cov is not None else None}

def do_fit_linear_Z(time_ti, Z_ti, sigma_ti):
    chiErr_ti = np.array(sigma_ti, float)
    chiErr_ti[~np.isfinite(chiErr_ti) | (chiErr_ti<=0)] = 1.0
    # p0: slope da WLS, tau0 appena prima del primo T
    k_wls, b_wls = _wls_slope_intercept(time_ti, Z_ti, chiErr_ti)
    k0 = max(abs(k_wls), 1e-12)  # positivo; il segno è assorbito da C durante il fit con maschera
    tau0 = time_ti.min() - max(1e-3, 0.02 * (time_ti.ptp()))
    def residuals(p):
        log_k, tau = p
        k  = np.exp(log_k)
        pred = model_linear_Z(time_ti, k, tau, 0.0)
        tt = time_ti - tau
        m  = tt > 0
        if m.sum() < 2: return np.full(time_ti.size, 1e6)
        C = np.sum((Z_ti[m] - pred[m]) / (chiErr_ti[m]**2)) / np.sum(1.0/(chiErr_ti[m]**2))
        res = (pred + C - Z_ti) / chiErr_ti
        res[~m] = 0.0
        return res
    lower = np.array([np.log(1e-12), time_ti.min()-time_ti.ptp()])
    upper = np.array([np.log(1e-1),  time_ti.max()])
    if FORCE_SHIFT_BEFORE_FIRST:
        upper[1] = min(upper[1], time_ti.min()-1e-9)
        tau0 = min(tau0, time_ti.min()-1e-3)
    p0    = np.array([np.log(k0), tau0])
    res = least_squares(residuals, p0, bounds=(lower, upper), max_nfev=3_000_000)
    log_k_fit, tau_fit = res.x
    k_fit = float(np.exp(log_k_fit))
    tt = time_ti - tau_fit; m = tt > 0
    base = model_linear_Z(time_ti, k_fit, tau_fit, 0.0)
    C_fit = float(np.sum((Z_ti[m] - base[m]) / (chiErr_ti[m]**2)) / np.sum(1.0/(chiErr_ti[m]**2)))
    dof = max(1, m.sum() - 2)
    chi2r = float(2*res.cost / dof)
    se_vec, cov = _stderr_from_jac(res, dof)
    stderr_map = {}
    if se_vec is not None:
        se_log_k, se_tau = float(se_vec[0]), float(se_vec[1])
        stderr_map = {
            "k": abs(k_fit)*se_log_k,
            "tau": se_tau,
            "C": (1.0/np.sqrt(np.sum(1.0/(chiErr_ti[m]**2)))) if np.isfinite(np.sum(1.0/(chiErr_ti[m]**2))) and np.sum(1.0/(chiErr_ti[m]**2))>0 else None,
        }
    return {"model":"linear_Z","tau": float(tau_fit),"k":k_fit,"C":C_fit,"chi2r": chi2r,
            "stderr_params": stderr_map, "cov": cov.tolist() if cov is not None else None}

def do_fit_saturation_Z(time_ti, Z_ti, sigma_ti):
    chiErr_ti = np.array(sigma_ti, float)
    chiErr_ti[~np.isfinite(chiErr_ti) | (chiErr_ti<=0)] = 1.0
    # c0 ~ max(Z), m0 ~ slope iniziale / c0, s0 appena prima del primo T
    c0 = max(np.max(Z_ti), 1e-6)
    # slope iniziale WLS su primi ~30% punti
    n0 = max(2, int(0.3*len(time_ti)))
    k_init, _ = _wls_slope_intercept(time_ti[:n0], Z_ti[:n0], chiErr_ti[:n0])
    m0 = max(abs(k_init)/max(c0, 1e-6), 1e-6)
    s0 = time_ti.min() - max(1e-3, 0.02 * (time_ti.ptp()))
    def residuals(p):
        log_c, log_m, s = p
        c, m = np.exp(log_c), np.exp(log_m)
        pred = model_saturation_Z(time_ti, c, m, s)
        return (pred - Z_ti) / chiErr_ti
    p0   = np.array([np.log(c0), np.log(m0), s0])
    lower = np.array([np.log(1e-6), np.log(1e-6), time_ti.min()-time_ti.ptp()])
    upper = np.array([np.log(1.0),  np.log(1e-1), time_ti.max()])
    if FORCE_SHIFT_BEFORE_FIRST:
        upper[2] = min(upper[2], time_ti.min()-1e-9)
        p0[2] = min(p0[2], time_ti.min()-1e-3)
    res = least_squares(residuals, p0, bounds=(lower, upper), max_nfev=3_000_000)
    log_c_fit, log_m_fit, s_fit = res.x
    c_fit, m_fit = float(np.exp(log_c_fit)), float(np.exp(log_m_fit))
    dof = max(1, time_ti.size - 3)
    chi2r = float(2*res.cost / dof)
    se_vec, cov = _stderr_from_jac(res, dof)
    stderr_map = {}
    if se_vec is not None:
        se_log_c, se_log_m, se_s = float(se_vec[0]), float(se_vec[1]), float(se_vec[2])
        stderr_map = {
            "c": abs(c_fit)*se_log_c,
            "m": abs(m_fit)*se_log_m,
            "s": se_s
        }
    return {"model":"saturation_Z","c":c_fit,"m":m_fit,"s": float(s_fit),"chi2r": chi2r,
            "stderr_params": stderr_map, "cov": cov.tolist() if cov is not None else None}

# ----------------- Plotter (usa i modelli) -----------------
def _plot_same_axes_with_fit(model, runs, df_ti, graphs_root, outfile, overlay_chi, chi_scale, fit_model,
                             figsize=(4.2,3.2), dpi=300, grid=False, x_label="", y_label="",
                             plot_tau_vertical=True):
    run_norm = [_norm_path(r) for r in runs]
    uids = [_make_run_uid(Path(r), graphs_root) for r in run_norm]

    dfT, rp_path, T_col = _load_runs_params_T(model, graphs_root, uids)
    T_map = dict(zip(dfT["run_uid"].astype(str), dfT["T"].astype(float)))
    T_arr = np.array([T_map[str(uid)] for uid in uids], float)

    Zs, Zsig = [], []
    for uid in uids:
        z, zs, _ = _Z_at_beta_max_and_sigma(df_ti, uid)
        if Z_SIGMA_POLICY == "binomial":
            p = np.clip(z, EPS, 1.0-EPS); zs = float(np.sqrt(p*(1.0-p)))
        elif Z_SIGMA_POLICY == "uniform":
            zs = 1.0
        Zs.append(z); Zsig.append(zs)
    Z_arr = np.array(Zs, float); Z_sig = np.array(Zsig, float)

    order = np.argsort(T_arr)
    T_arr, Z_arr, Z_sig = T_arr[order], Z_arr[order], Z_sig[order]
    run_ord = [run_norm[i] for i in order]

    chi_t = chi_y = chi_s = None
    idx_tmax = int(np.argmax(T_arr))
    if overlay_chi in ("tmax","all") or fit_model in ("quad_chi","cinematic_chi"):
        t_series, chi_series, chi_err_series, _ = _load_av_series_with_err(run_ord[idx_tmax])
        T_max = float(T_arr[idx_tmax])
        mask = t_series <= T_max + 1e-12
        t_series, chi_series = t_series[mask], chi_series[mask]
        chi_sigma = _sigma_from_prob_or_given(chi_series, chi_err_series[mask] if chi_err_series is not None else None)
        if chi_scale.lower() == "z":
            chi_series = chi_series * float(Z_arr[idx_tmax])
            chi_sigma  = chi_sigma  * float(Z_arr[idx_tmax])
        avTime   = np.asarray(t_series)
        avChi    = np.asarray(chi_series)
        avChiErr = np.asarray(chi_sigma)
        core_mask = (np.isfinite(avTime) & np.isfinite(avChi) & np.isfinite(avChiErr) & (avChiErr>0) & (avTime>TMIN_CHI))
        chi_t, chi_y, chi_s = avTime[core_mask], avChi[core_mask], avChiErr[core_mask]

    with ustyle.auto_style(mode="latex",
        base=str(Path(ustyle.__file__).resolve().parent / "styles" / "paper_base.mplstyle"),
        overlay=str(Path(ustyle.__file__).resolve().parent / "styles" / "overlay_latex.mplstyle")):
        fig = plt.figure(figsize=figsize, dpi=dpi); ax = fig.add_subplot(111)
        ax.scatter(T_arr, Z_arr, s=12, color=COL_Z_POINTS)
        if overlay_chi == "tmax" and (chi_t is not None):
            t_full, chi_full, chi_err_full, _ = _load_av_series_with_err(run_ord[idx_tmax])
            if chi_scale.lower() == "z": chi_full = chi_full * float(Z_arr[idx_tmax])
            ax.plot(t_full, chi_full, color=COL_CHI_CURVE, lw=1.1)
        elif overlay_chi == "all":
            for i, rpath in enumerate(run_ord):
                t, chi, chi_err, _ = _load_av_series_with_err(rpath)
                if chi_scale.lower() == "z": chi = chi * float(Z_arr[i])
                ax.plot(t, chi, color=COL_CHI_CURVE, lw=0.9, alpha=0.55)

        fit_res = None
        tau_char = None
        if fit_model == "linear_Z":
            fit_res = do_fit_linear_Z(T_arr, Z_arr, Z_sig)
            tau,k,C = fit_res["tau"], fit_res["k"], fit_res["C"]
            tx = np.linspace(max(tau, T_arr.min()), T_arr.max(), 600)
            ax.plot(tx, model_linear_Z(tx, k, tau, C), color=COL_FIT, lw=1.2)
            tau_char = (tau - C/k) if (k != 0) else None
        elif fit_model == "saturation_Z":
            fit_res = do_fit_saturation_Z(T_arr, Z_arr, Z_sig)
            c,m,s = fit_res["c"], fit_res["m"], fit_res["s"]
            tx = np.linspace(max(s, T_arr.min()), T_arr.max(), 600)
            ax.plot(tx, model_saturation_Z(tx, c, m, s), color=COL_FIT, lw=1.2)
            tau_char = s
        elif fit_model == "quad_chi":
            if chi_t is None or chi_y is None or chi_s is None: raise RuntimeError("Serve χ(t) per 'quad_chi'.")
            fit_res = do_fit_quadratic_chi(chi_t, chi_y, chi_s)
            tau,a,C = fit_res["tau"], fit_res["a"], fit_res["C"]
            tx = np.linspace(max(tau, chi_t.min()), chi_t.max(), 600)
            ax.plot(tx, model_quad_chi(tx, a, tau, C), color=COL_FIT, lw=1.2)
            tau_char = tau
        elif fit_model == "cinematic_chi":
            if chi_t is None or chi_y is None or chi_s is None: raise RuntimeError("Serve χ(t) per 'cinematic_chi'.")
            fit_res = do_fit_cinematic_chi(chi_t, chi_y, chi_s)
            tau,r,k,C = fit_res["tau"], fit_res["r"], fit_res["k"], fit_res["C"]
            tx = np.linspace(max(tau, chi_t.min()), chi_t.max(), 600)
            ax.plot(tx, model_cinematic_chi(tx, r, k, tau, C), color=COL_FIT, lw=1.2)
            tau_char = tau

        if plot_tau_vertical and (tau_char is not None) and np.isfinite(tau_char):
            ax.axvline(tau_char, linestyle="--", linewidth=0.9, color=COL_TAU)

        if x_label: ax.set_xlabel(x_label)
        if y_label: ax.set_ylabel(y_label)
        if grid: ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)

        out_base = Path(outfile); out_base.parent.mkdir(parents=True, exist_ok=True)
        fmts = [f.lower() for f in specs[0].get("formats", ["pdf","png"])]
        for ext in ("pdf","png"):
            if ext in fmts: uplot.export_figure_strict(fig, str(out_base), formats=(ext,), dpi=dpi)
        plt.close(fig)

    meta = {
        "model": model, "runs_sorted": run_ord, "graphs_root": str(graphs_root),
        "tables": {
            "ti_points": str((_tables_root(graphs_root, model) / "ti" / "ti_points.parquet").resolve()),
            "runs_params": str((_tables_root(graphs_root, model) / "runs_params" / "runs_params.parquet").resolve()),
        },
        "T": T_arr.tolist(), "Z_at_beta_max": Z_arr.tolist(), "Z_sigma": Z_sig.tolist(),
        "overlay_chi": overlay_chi, "chi_scale": chi_scale, "fit_model": fit_model,
        "fit_result": fit_res if fit_res is not None else {},
        "Z_sigma_policy": Z_SIGMA_POLICY,
        "figsize": list(figsize), "dpi": dpi, "outfile_base": str(Path(outfile)),
        "plot_tau_vertical": bool(plot_tau_vertical)
    }
    Path(str(Path(outfile)) + "_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

def run_from_specs(_specs: List[Dict]):
    for s in _specs:
        model = s["model"]; runs  = s["runs"]
        graphs_override = s.get("graphs_root", None)
        df_ti, _, graphs_root = _load_ti_points_for_runs(model, runs, graphs_override)
        _plot_same_axes_with_fit(
            model=model, runs=runs, df_ti=df_ti, graphs_root=graphs_root,
            outfile=s.get("outfile", "fig/Z_withChiCurves_sameAxes/output"),
            overlay_chi=s.get("overlay_chi", "tmax"), chi_scale=s.get("chi_scale", "Z"),
            fit_model=s.get("fit_model", "linear_Z"),
            figsize=tuple(s.get("figsize", (4.2,3.2))), dpi=int(s.get("dpi", 300)),
            grid=bool(s.get("grid", False)), x_label=s.get("x_label", r"$t$"),
            y_label=s.get("y_label", r"$Z$ e $Z\cdot\chi$"),
            plot_tau_vertical=bool(s.get("plot_tau_vertical", True))
        )

if __name__ == "__main__":
    run_from_specs(specs)
