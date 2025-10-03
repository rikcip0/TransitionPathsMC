
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
        "outfile": "fig/Z_vs_beta_withChiOverlay/example_group",
        "overlay_chi": "max",   # "max" | "all" | "none"
        "figsize": (4.2, 3.2),
        "dpi": 300,
        "formats": ["pdf", "png"],
        "grid": False,
        "x_label": r"$\beta$",
        "y_label": r"$Z(\beta)$  and  $Z(\beta)\cdot\chi_{\mathrm{ref}}$",
    },
]

FIT_EXTEND_LEFT  = 0.00
FIT_EXTEND_RIGHT = 0.00

import os, sys, json, hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2] if len(HERE.parents) >= 3 else HERE.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.append('../')
from MyBasePlots.FigCore import utils_style as ustyle
from MyBasePlots.FigCore import utils_plot  as uplot

def _norm_path(p: str) -> str:
    if p is None:
        return p
    q = p.replace("\\\\", "\\").replace("\\", "/")
    if len(q) >= 3 and q[1:3] == ":/":
        drive = q[0].lower()
        q = f"/mnt/{drive}/{q[3:]}"
    return q

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

def _tables_root(graphs_root: Path, model: str) -> Path:
    return (graphs_root.parent / "MultiPathsMC" / model / "v1").resolve()

def _make_run_uid(run_dir: Path, graphs_root: Path) -> str:
    try:
        rel = str(run_dir.resolve().relative_to(graphs_root.resolve()))
    except Exception:
        rel = str(run_dir.resolve())
    import hashlib
    return hashlib.sha1(rel.encode("utf-8")).hexdigest()[:16]

def _load_ti_points_for_runs(model: str, runs: List[str]):
    if not runs:
        raise ValueError("Lista 'runs' vuota.")
    graphs_root = _find_graphs_root_from_run(_norm_path(runs[0]))
    base = _tables_root(graphs_root, model)
    tip_path = base / "ti" / "ti_points.parquet"
    if not tip_path.exists():
        raise FileNotFoundError(f"ti_points.parquet non trovato in {tip_path}")
    df = pd.read_parquet(tip_path)
    want = {_make_run_uid(Path(_norm_path(r)), graphs_root) for r in runs}
    df = df[df["run_uid"].astype(str).isin(want)].copy()
    if df.empty:
        raise RuntimeError("Nessuna riga in ti_points per i run forniti.")
    return df, tip_path

def _load_av_chi_mean(run_path: str, tail_frac: float = 0.10):
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
    if arr.ndim != 2 or arr.shape[1] < 5:
        raise RuntimeError(f"Formato inatteso per {av_file} (cols<5)")
    t = arr[:,0]
    chi = arr[:,4]
    n = len(t)
    i0 = int(max(0, n - max(5, int(n*tail_frac))))
    mean_chi = float(np.nanmean(chi[i0:]))
    return mean_chi, n - i0, str(av_file)

def _choose_ref_run_by_maxT(df_ti: pd.DataFrame) -> str:
    g = df_ti.groupby("run_uid")["beta"].min()
    uid_ref = g.idxmin()
    return str(uid_ref)

def _lineplot_Z_and_overlays(df_ti: pd.DataFrame,
                             run_paths: List[str],
                             model: str,
                             outfile: str,
                             overlay_chi: str = "max",
                             figsize=(4.2,3.2), dpi=300,
                             grid=False, x_label="", y_label=""):
    run_paths_norm = [_norm_path(p) for p in run_paths]
    graphs_root = _find_graphs_root_from_run(run_paths_norm[0])
    uid_order = [_make_run_uid(Path(p), graphs_root) for p in run_paths_norm]
    series = {}
    for uid in uid_order:
        sub = df_ti[df_ti["run_uid"].astype(str) == uid].copy()
        sub = sub.sort_values("beta")
        series[uid] = (sub["beta"].to_numpy(), sub["ZFromTIBeta"].to_numpy())

    chi_info_list = []
    if overlay_chi in ("max","all"):
        uid2path = dict(zip(uid_order, run_paths_norm))
        if overlay_chi == "max":
            uid_ref = _choose_ref_run_by_maxT(df_ti)
            chi_ref, n_tail, av_file = _load_av_chi_mean(uid2path[uid_ref])
            chi_info_list.append((uid_ref, chi_ref, n_tail, av_file))
        else:
            for uid in uid_order:
                chi_ref, n_tail, av_file = _load_av_chi_mean(uid2path[uid])
                chi_info_list.append((uid, chi_ref, n_tail, av_file))

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    # Z(β)
    for uid in uid_order:
        beta, Z = series[uid]
        ax.plot(beta, Z, linewidth=1.0)

    # Z·chi overlays
    if overlay_chi == "max":
        uid_ref, chi_ref, _, _ = chi_info_list[0]
        beta, Z = series[uid_ref]
        ax.plot(beta, Z*chi_ref, linestyle="--", linewidth=1.0)
    elif overlay_chi == "all":
        for uid, chi_ref, _, _ in chi_info_list:
            beta, Z = series[uid]
            ax.plot(beta, Z*chi_ref, linestyle="--", linewidth=1.0)

    if x_label: ax.set_xlabel(x_label)
    if y_label: ax.set_ylabel(y_label)
    if grid: ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)

    out_base = Path(outfile); out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf","png"]:
        if ext in set(fmt.lower() for fmt in specs[0].get("formats", ["pdf","png"])):
            uplot.export_figure_strict(fig, str(out_base), formats=(ext,), dpi=dpi)
    plt.close(fig)

    meta = {
        "model": model,
        "runs": run_paths_norm,
        "graphs_root": str(graphs_root),
        "ti_points_parquet": str((_tables_root(graphs_root, model) / "ti" / "ti_points.parquet").resolve()),
        "overlay_chi": overlay_chi,
        "chi_info": [
            {"run_uid": uid, "chi_tail_mean": chi, "tail_count": n, "av_file": avf}
            for (uid, chi, n, avf) in chi_info_list
        ],
        "figsize": list(figsize), "dpi": dpi,
        "outfile_base": str(out_base),
    }
    Path(str(out_base) + "_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

def run_from_specs(_specs: List[Dict]):
    with ustyle.auto_style(mode="latex",
                           base=str(Path(ustyle.__file__).resolve().parent / "styles" / "paper_base.mplstyle"),
                           overlay=str(Path(ustyle.__file__).resolve().parent / "styles" / "overlay_latex.mplstyle")):
        for s in _specs:
            model = s["model"]
            runs  = s["runs"]
            df_ti, _ = _load_ti_points_for_runs(model, runs)
            _lineplot_Z_and_overlays(
                df_ti=df_ti,
                run_paths=runs,
                model=model,
                outfile=s.get("outfile", "fig/Z_vs_beta_withChiOverlay/output"),
                overlay_chi=s.get("overlay_chi", "max"),
                figsize=tuple(s.get("figsize", (4.2,3.2))),
                dpi=int(s.get("dpi", 300)),
                grid=bool(s.get("grid", False)),
                x_label=s.get("x_label", ""),
                y_label=s.get("y_label", ""),
            )

if __name__ == "__main__":
    run_from_specs(specs)
