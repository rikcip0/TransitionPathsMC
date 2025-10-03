#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==============================
# LISTA PLOT (in testa al file)
# ==============================
specs = [
    {
        "model": "realGraphs/ZKC",
        "runs": [
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\40_0.9_0_inf_20_inf_run6952",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\35_0.9_0_inf_20_inf_run6950",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\30_0.9_0_inf_20_inf_run7081",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\25_0.9_0_inf_20_inf_run7079",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\20_0.9_0_inf_20_inf_run7077",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\15_0.9_0_inf_20_inf_run7083",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\10_0.9_0_inf_20_inf_run7085",
        ],
        "outfile": "_figs/Z_withChiCurves/ZKC_ZandChi_bet0p9",
        # "tmax" = solo la run con T massimo; "all" = tutte; "none" = nessuna
        "overlay_chi": "tmax",
        # scala Chi(t) per Z(@beta max) della stessa run: "Z" oppure "none"
        "chi_scale": "Z",
        # opzionale se autodetect grafs root fallisse
        # "graphs_root": r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs",
        # stile/export
        "figsize": (4.2, 3.2),
        "dpi": 300,
        "formats": ["pdf", "png"],
        "grid": False,
        "x_label": r"$t$  (punti a $T$)",
        "y_label": r"$Z(T)$  e  $Z\cdot\chi(t)$",
    },
    {
        "model": "realGraphs/ZKC",
        "runs": [
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\40_0.55_0_inf_20_inf_run9959",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\35_0.55_0_inf_20_inf_run8308",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\30_0.55_0_inf_20_inf_run9935",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\25_0.55_0_inf_20_inf_run42",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\20_0.55_0_inf_20_inf_run9742",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\15_0.55_0_inf_20_inf_run8772",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\10_0.55_0_inf_20_inf_run4146",
        ],
        "outfile": "_figs/Z_withChiCurves/ZKC_ZandChi_bet0p55",
        # "tmax" = solo la run con T massimo; "all" = tutte; "none" = nessuna
        "overlay_chi": "tmax",
        # scala Chi(t) per Z(@beta max) della stessa run: "Z" oppure "none"
        "chi_scale": "Z",
        # opzionale se autodetect grafs root fallisse
        # "graphs_root": r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs",
        # stile/export
        "figsize": (4.2, 3.2),
        "dpi": 300,
        "formats": ["pdf", "png"],
        "grid": False,
        "x_label": r"$t$  (punti a $T$)",
        "y_label": r"$Z(T)$  e  $Z\cdot\chi(t)$",
    },
    {
        "model": "realGraphs/ZKC",
        "runs": [
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\40_0.35_0_inf_20_inf_run9951",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\35_0.35_0_inf_20_inf_run5103",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\30_0.35_0_inf_20_inf_run9927",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\25_0.35_0_inf_20_inf_run34",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\20_0.35_0_inf_20_inf_run3494",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\15_0.35_0_inf_20_inf_run8764",
            r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\10_0.35_0_inf_20_inf_run4138",
        ],
        "outfile": "_figs/Z_withChiCurves/ZKC_ZandChi_bet0p35",
        # "tmax" = solo la run con T massimo; "all" = tutte; "none" = nessuna
        "overlay_chi": "tmax",
        # scala Chi(t) per Z(@beta max) della stessa run: "Z" oppure "none"
        "chi_scale": "Z",
        # opzionale se autodetect grafs root fallisse
        # "graphs_root": r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs",
        # stile/export
        "figsize": (4.2, 3.2),
        "dpi": 300,
        "formats": ["pdf", "png"],
        "grid": False,
        "x_label": r"$t$  (punti a $T$)",
        "y_label": r"$Z(T)$  e  $Z\cdot\chi(t)$",
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
        # "tmax" = solo la run con T massimo; "all" = tutte; "none" = nessuna
        "overlay_chi": "tmax",
        # scala Chi(t) per Z(@beta max) della stessa run: "Z" oppure "none"
        "chi_scale": "Z",
        # opzionale se autodetect grafs root fallisse
        # "graphs_root": r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs",
        # stile/export
        "figsize": (4.2, 3.2),
        "dpi": 300,
        "formats": ["pdf", "png"],
        "grid": False,
        "x_label": r"$t$  (punti a $T$)",
        "y_label": r"$Z(T)$  e  $Z\cdot\chi(t)$",
    },
]

import os, sys, json, hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# FigCore (se manca è giusto che fallisca)
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2] if len(HERE.parents) >= 3 else HERE.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.append('../')
from MyBasePlots.FigCore import utils_style as ustyle
from MyBasePlots.FigCore import utils_plot  as uplot

# ====== CONFIG: colonna di av*.dat per Chi (0-based) ======
CHI_COL = 4  # stessa che usi nel fit Chi vs t

# --- helpers path utils (Windows->WSL robusto) ---
def _norm_path(p: str) -> str:
    if p is None:
        return p
    q = p.replace("\\\\", "\\").replace("\\", "/")
    if len(q) >= 3 and q[1:3] == ":/":
        drive = q[0].lower()
        q = f"/mnt/{drive}/{q[3:]}"
    return q

def _find_graphs_root_from_run(run_path: str, override: str = None) -> Path:
    if override:
        gr = Path(_norm_path(override)).resolve()
        if not gr.exists():
            raise RuntimeError(f"graphs_root override non esiste: {gr}")
        return gr
    p = Path(run_path).resolve()
    for anc in [p] + list(p.parents):
        cur = anc
        for _ in range(20):
            dg = cur / "Data" / "Graphs"
            if dg.exists() and dg.is_dir():
                return dg
            cur = cur.parent
    raise RuntimeError("Graphs root (Data/Graphs) non trovata risalendo dal run_path. Usa 'graphs_root' in specs per override.")

def _tables_root(graphs_root: Path, model: str) -> Path:
    return (graphs_root.parent / "MultiPathsMC" / model / "v1").resolve()

def _make_run_uid(run_dir: Path, graphs_root: Path) -> str:
    try:
        rel = str(run_dir.resolve().relative_to(graphs_root.resolve()))
    except Exception:
        rel = str(run_dir.resolve())
    return hashlib.sha1(rel.encode("utf-8")).hexdigest()[:16]

def _load_ti_points_for_runs(model: str, runs: List[str], graphs_root_override: str = None) -> Tuple[pd.DataFrame, Path, Path]:
    if not runs:
        raise ValueError("Lista 'runs' vuota.")
    graphs_root = _find_graphs_root_from_run(_norm_path(runs[0]), graphs_root_override)
    base = _tables_root(graphs_root, model)
    tip_path = base / "ti" / "ti_points.parquet"
    if not tip_path.exists():
        raise FileNotFoundError(f"ti_points.parquet non trovato in {tip_path}")
    df = pd.read_parquet(tip_path)
    want = {_make_run_uid(Path(_norm_path(r)), graphs_root) for r in runs}
    df = df[df["run_uid"].astype(str).isin(want)].copy()
    if df.empty:
        raise RuntimeError("Nessuna riga in ti_points per i run forniti.")
    return df, tip_path, graphs_root

def _load_runs_params_T(model: str, graphs_root: Path, run_uids: List[str]):
    base = _tables_root(graphs_root, model)
    rp_path = base / "runs_params" / "runs_params.parquet"
    if not rp_path.exists():
        raise FileNotFoundError(f"runs_params.parquet non trovato in {rp_path}")
    df = pd.read_parquet(rp_path)
    if "run_uid" not in df.columns:
        raise RuntimeError("runs_params.parquet: manca colonna 'run_uid'.")
    T_cols = [c for c in ["T","totalTime","total_time","T_end","t_end","last_t"] if c in df.columns]
    if not T_cols:
        raise RuntimeError("runs_params.parquet: nessuna colonna T trovata tra ['T','totalTime','total_time','T_end','t_end','last_t'].")
    src_col = T_cols[0]
    dfT = df[["run_uid", src_col]].copy()
    dfT["T"] = dfT[src_col].astype(float)
    dfT = dfT[["run_uid","T"]]
    dfT = dfT[dfT["run_uid"].astype(str).isin([str(u) for u in run_uids])].copy()
    if dfT.empty or dfT["T"].isna().any() or (~np.isfinite(dfT["T"])).any():
        raise RuntimeError("runs_params: T mancante/non finito per uno o più run richiesti (niente fallback).")
    return dfT, rp_path, src_col

def _Z_at_beta_max_for_uid(df_ti: pd.DataFrame, uid: str) -> float:
    sub = df_ti[df_ti["run_uid"].astype(str) == uid]
    if sub.empty:
        return np.nan
    row = sub.loc[sub["beta"].idxmax()]
    return float(row["ZFromTIBeta"])

def _load_av_series(run_path: str) -> tuple:
    parent = Path(_norm_path(run_path))
    av_file = None
    for f in parent.iterdir():
        if f.is_file() and f.name.startswith("av"):
            av_file = f
            break
    if av_file is None:
        raise FileNotFoundError(f"av*.dat non trovato in {parent}")
    rows = []
    with open(av_file, "r", encoding="utf-8", errors="ignore") as fh:
        for ln in fh:
            s = ln.strip()
            if not s or s.startswith(("#","%","//")):
                continue
            parts = s.split()
            try:
                vals = [float(v) for v in parts]
                rows.append(vals)
            except ValueError:
                continue
    arr = np.asarray(rows, float)
    if arr.ndim != 2 or arr.shape[1] <= CHI_COL:
        raise RuntimeError(f"Formato inatteso per {av_file} (cols <= {CHI_COL})")
    t = arr[:,0].astype(float)
    chi = arr[:,CHI_COL].astype(float)  # stessa colonna del fit Chi vs t
    good = np.isfinite(t) & np.isfinite(chi)
    return t[good], chi[good], str(av_file)

def _plot_same_axes(model: str,
                    runs: List[str],
                    df_ti: pd.DataFrame,
                    graphs_root: Path,
                    outfile: str,
                    overlay_chi: str,
                    chi_scale: str,
                    figsize=(4.2,3.2), dpi=300,
                    grid=False, x_label="", y_label=""):
    run_norm = [_norm_path(r) for r in runs]
    uids = [_make_run_uid(Path(r), graphs_root) for r in run_norm]

    # T da runs_params
    dfT, rp_path, T_col = _load_runs_params_T(model, graphs_root, uids)
    T_map = dict(zip(dfT["run_uid"].astype(str), dfT["T"].astype(float)))
    
    T_arr = np.array([T_map[str(uid)] for uid in uids], float)

    # Z(@beta max) da ti_points
    Z_arr = np.array([_Z_at_beta_max_for_uid(df_ti, uid) for uid in uids], float)

    # ordina per T crescente
    order = np.argsort(T_arr)
    T_arr = T_arr[order]
    Z_arr = Z_arr[order]
    run_ord = [run_norm[i] for i in order]

    # quali curve Chi(t) disegnare
    chi_indices = []
    if overlay_chi == "tmax":
        chi_indices = [int(np.argmax(T_arr))]
    elif overlay_chi == "all":
        chi_indices = list(range(len(run_ord)))
    else:
        chi_indices = []

    # Plot: stessi assi
    with ustyle.auto_style(mode="latex",
                           base=str(Path(ustyle.__file__).resolve().parent / "styles" / "paper_base.mplstyle"),
                           overlay=str(Path(ustyle.__file__).resolve().parent / "styles" / "overlay_latex.mplstyle")):
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111)

        # Punti Z(T) NON collegati
        ax.scatter(T_arr, Z_arr, s=10, color='blue')

        # Curve Chi(t) (eventualmente scalate) sovrapposte
        for idx in chi_indices:
            rpath = run_ord[idx]
            t, chi, avf = _load_av_series(rpath)
            scale = float(Z_arr[idx]) if chi_scale.lower() == "z" else 1.0
            ax.plot(t, chi*scale, linewidth=1.0, color='black')

        if x_label: ax.set_xlabel(x_label)
        if y_label: ax.set_ylabel(y_label)
        if grid: ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)

        out_base = Path(outfile); out_base.parent.mkdir(parents=True, exist_ok=True)
        fmts = [f.lower() for f in specs[0].get("formats", ["pdf","png"])]
        for ext in ("pdf","png"):
            if ext in fmts:
                uplot.export_figure_strict(fig, str(out_base), formats=(ext,), dpi=dpi)
        plt.close(fig)

    # Meta
    meta = {
        "model": model,
        "runs_sorted": run_ord,
        "graphs_root": str(graphs_root),
        "tables": {
            "ti_points": str((_tables_root(graphs_root, model) / "ti" / "ti_points.parquet").resolve()),
            "runs_params": str((_tables_root(graphs_root, model) / "runs_params" / "runs_params.parquet").resolve()),
        },
        "chi_column_index": CHI_COL,
        "T": T_arr.tolist(),
        "Z_at_beta_max": Z_arr.tolist(),
        "overlay_chi": overlay_chi,
        "chi_scale": chi_scale,
        "figsize": list(figsize),
        "dpi": dpi,
        "outfile_base": str(out_base),
    }
    Path(str(out_base) + "_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

def run_from_specs(_specs: List[Dict]):
    for s in _specs:
        model = s["model"]
        runs  = s["runs"]
        graphs_override = s.get("graphs_root", None)
        df_ti, _, graphs_root = _load_ti_points_for_runs(model, runs, graphs_override)
        print(runs)
        _plot_same_axes(
            model=model,
            runs=runs,
            df_ti=df_ti,
            graphs_root=graphs_root,
            outfile=s.get("outfile", "_figs/Z_withChiCurves/output"),
            overlay_chi=s.get("overlay_chi", "tmax"),
            chi_scale=s.get("chi_scale", "Z"),
            figsize=tuple(s.get("figsize", (4.2,3.2))),
            dpi=int(s.get("dpi", 300)),
            grid=bool(s.get("grid", False)),
            x_label=s.get("x_label", r"$t$  (punti a $T$)"),
            y_label=s.get("y_label", r"$Z(T)$  e  $Z\cdot\chi(t)$"),
        )

if __name__ == "__main__":
    run_from_specs(specs)
