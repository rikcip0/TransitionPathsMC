#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_fig_fit_scatter_paper.py — plot “paper” dei fit -ln k vs N (UN plot per voce in PLOTS).

- Lista PLOTS in testa (una voce => un plot), con beta fissata (raw o rescaled).
- Import MyBasePlots con sys.path.append('../') e senza fallback (REQUIRE_MYBASEPLOTS=True).
- NESSUNA legenda/titolo nel grafico (solo assi).
- Opzione colorbar per rescaled: mostra $\tilde{\beta}$ pre-binning (cioè PRIMA dell'arrotondamento/binning).
- Opzione edge-by-init: bordo punto colorato in funzione di 'trajInit' (se presente in ti_points).
- PNG+PDF e JSON metadati accanto alle figure.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
edge_by_init_default = True
FIG_ID    = "F3"
SLUG      = "fit_scatter"
FIGSIZE   = (3.0, 2.2)
DPI       = 300
REQUIRE_MYBASEPLOTS = True
DATA_ROOT: Optional[str] = None   # se None: deduce ../../Data dalla posizione del file
VERBOSE = True

# Soglie qualità (coerenti con i fit): chi2 sempre; scale2 solo per *_Scaled
CHI2_THRESHOLD_DEFAULT   = 0.43
SCALE2_THRESHOLD_DEFAULT = 0.33
YMODE = "minuslnk"  # 'minuslnk' | 'logk' | 'k'

PLOTS: List[Dict[str, Any]] = [
    # Esempi — sostituisci con i tuoi
     {"id":"raw_ER_Ngt30_beta0p65", "model":"ER","subset_id":"07018d22622d32a8","subset_label":"N>30", "beta_kind":"raw", "kcol":"kFromChi", "beta":0.65,
      "include_unused": True, "edge_by_init": True},
     {"id":"raw_ER_Ngt30_beta0p65", "model":"ER","subset_id":"07018d22622d32a8","subset_label":"N>30", "beta_kind":"rescaled", "anchor":"M", "kcol":"kFromChi", "beta_rescaled_bin":0.65,
      "include_unused": True, "colorbar": True, "edge_by_init": True},
     {"id":"raw_ER_Ngt30_beta1", "model":"ER","subset_id":"07018d22622d32a8","subset_label":"N>30", "beta_kind":"raw", "kcol":"kFromChi", "beta":1.,
      "include_unused": True},
     {"id":"raw_ER_Ngt30_beta1", "model":"ER","subset_id":"07018d22622d32a8","subset_label":"N>30", "beta_kind":"rescaled", "anchor":"M", "kcol":"kFromChi", "beta_rescaled_bin":1.,
      "include_unused": True, "colorbar": True},
    {"id":"raw_RRG_all_beta1", "model":"RRG", "subset_id":"2354bcad23a43145", "beta_kind":"raw", "kcol":"kFromChi", "beta":1.},
    {"id":"raw_RRG_all_beta07", "model":"RRG", "subset_id":"2354bcad23a43145", "beta_kind":"raw", "kcol":"kFromChi", "beta":0.7},
    {"id":"raw_RRG_all_beta1", "model":"RRG", "subset_id":"2354bcad23a43145", "beta_kind":"raw", "kcol":"kFromChi_InBetween", "beta":1.},
    {"id":"raw_RRG_all_beta07", "model":"RRG", "subset_id":"2354bcad23a43145", "beta_kind":"raw", "kcol":"kFromChi_InBetween", "beta":0.7},
]

# MyBasePlots import (no fallback)
sys.path.append('../')
try:
    from MyBasePlots.FigCore import utils_style as ustyle
except Exception as e:
    if REQUIRE_MYBASEPLOTS:
        raise RuntimeError(f"MyBasePlots non disponibile: {e}. Interrompo.") from e
    else:
        print(f"[warn] MyBasePlots non disponibile ({e}).", file=sys.stderr)
        ustyle = None

def _data_base_dir(data_root: Optional[str]) -> Path:
    if data_root is not None:
        return Path(data_root)
    # ../../Data rispetto a questo file
    return Path(__file__).resolve().parents[2] / "Data"

def _ti_dir(model: str, data_root: Optional[str]) -> Path:
    return _data_base_dir(data_root) / "MultiPathsMC" / model / "v1" / "ti"

def _read_parquet(p: Path, tag: str) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(f"[{tag}] non trovato: {p}")
    return pd.read_parquet(p)

def _sanitize(s: str) -> str:
    s = "" if s is None else str(s)
    for a,b in [('>','gt'),('<','lt'),('=','eq'),(' ','_'),('/','-'),(';','__'),(':',''),(',','_')]:
        s = s.replace(a,b)
    return s[:120] if s else "unknown"

def _fmt_val(v) -> str:
    try:
        if v is None or (isinstance(v,float) and (np.isnan(v) or np.isinf(v))): return str(v)
        vf = float(v)
        return str(int(round(vf))) if abs(vf-round(vf))<1e-9 else f"{vf:.6g}"
    except Exception:
        return str(v)

def _subset_folder_name(label: str, subset_id: str) -> str:
    safe_lab = (label or "subset").replace('>','gt').replace('<','lt').replace('=','eq').replace(' ','_').replace('/','-')
    return f"{safe_lab}__{subset_id[:8]}"

def _families_for_subset(members: pd.DataFrame, subset_id: str) -> List[str]:
    fams: List[str] = []
    if "family_id" in members.columns:
        fams = [f for f in members.loc[members["subset_id"]==subset_id,"family_id"].dropna().unique().tolist() if isinstance(f,str)]
    return fams

def _phys_signature_parts(families_df: pd.DataFrame, members: pd.DataFrame, subset_id: str, model: str) -> List[str]:
    fams = _families_for_subset(members, subset_id)
    if len(fams)!=1 or families_df is None or families_df.empty:
        return [model, f"multiFamilies__n{len(fams)}" if len(fams)>1 else "noFamily"]
    fid = fams[0]
    row = families_df.loc[families_df["family_id"]==fid]
    if row.empty: return [model, f"family__{fid[:8]}"]
    r = row.iloc[0]
    parts = [model]
    parts.append(_sanitize(f"C_{_fmt_val(r.get('C'))}__fPosJ_{_fmt_val(r.get('fPosJ'))}"))
    parts.append(_sanitize(str(r.get("fieldType"))))
    parts.append(_sanitize(f"fieldSigma_{_fmt_val(r.get('fieldSigma'))}"))
    parts.append(_sanitize(f"Hext_{_fmt_val(r.get('Hext'))}"))
    nq = None
    for qname in ["normalizedQstar","nQstar","normalized_qstar"]:
        if qname in r.index: nq = r.get(qname); break
    parts.append(_sanitize(f"Hin_{_fmt_val(r.get('Hin'))}__Hout_{_fmt_val(r.get('Hout'))}__nQstar_{_fmt_val(nq)}__{fid[:8]}"))
    return parts

def _fig_dir(model: str, anchor_tag: str, phys_parts: List[str], subset_folder: str, spec_id: str) -> Path:
    base = Path(__file__).resolve().parent / "_figs" / "TI" / f"{FIG_ID}_{SLUG}" / model / anchor_tag
    for p in phys_parts: base = base / p
    base = base / subset_folder / spec_id
    base.mkdir(parents=True, exist_ok=True)
    return base

def _y_from_k(arr_k: np.ndarray, mode: str) -> np.ndarray:
    if mode == "minuslnk":
        return -np.log(arr_k)
    elif mode == "logk":
        return np.log(arr_k)
    else:
        return arr_k

def _to_jsonable(o):
    import numpy as _np
    from pathlib import Path as _Path
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(x) for x in o]
    if isinstance(o, dict):
        return {str(k): _to_jsonable(v) for k,v in o.items()}
    if isinstance(o, _Path):
        return str(o)
    if isinstance(o, (_np.integer,)):
        return int(o)
    if isinstance(o, (_np.floating,)):
        return float(o)
    if isinstance(o, (_np.ndarray,)):
        return [_to_jsonable(x) for x in o.tolist()]
    return str(o)

def _plot_one(spec: Dict[str, Any]) -> Optional[Path]:
    # stile (LaTeX via MyBasePlots)
    try:
        ustyle.auto_style(mode="latex", base="paper_base.mplstyle", overlay="overlay_latex.mplstyle")
    except Exception as e:
        raise RuntimeError(f"Errore nel setup stile MyBasePlots: {e}") from e

    model     = str(spec.get("model","ER"))
    subset_id = spec.get("subset_id")
    subset_lb = spec.get("subset_label")
    beta_kind = str(spec.get("beta_kind","raw"))
    kcol      = str(spec.get("kcol","kFromChi"))
    spec_id   = str(spec.get("id","plot"))

    beta_raw  = spec.get("beta", None)
    anchor    = spec.get("anchor", None)
    ref_stat  = str(spec.get("ref_stat","mean"))
    beta_star = spec.get("beta_rescaled_bin", None)
    step      = float(spec.get("rescaled_step", 0.025))

    chi2_thr  = float(spec.get("chi2_threshold", CHI2_THRESHOLD_DEFAULT))
    scale_thr = float(spec.get("scale2_threshold", SCALE2_THRESHOLD_DEFAULT))
    filters   = spec.get("filters", None)
    xlim      = spec.get("xlim")
    ylim      = spec.get("ylim")
    include_unused         = bool(spec.get("include_unused", False))
    include_family_outside = bool(spec.get("include_family_outside", False))

    # Paths
    ti_dir = _ti_dir(model, DATA_ROOT)
    fits_path     = ti_dir / "ti_linear_fits.parquet"
    points_path   = ti_dir / "ti_points.parquet"
    resc_points_p = ti_dir / "ti_subset_points_rescaled.parquet"
    members_path  = ti_dir / "ti_subset_members.parquet"
    subsets_path  = ti_dir / "ti_family_subsets.parquet"
    families_path = ti_dir / "ti_families.parquet"

    fits  = _read_parquet(fits_path, "ti_linear_fits")
    pts   = _read_parquet(points_path, "ti_points")
    resc  = _read_parquet(resc_points_p, "ti_subset_points_rescaled")
    mem   = _read_parquet(members_path, "ti_subset_members")
    subs  = _read_parquet(subsets_path, "ti_family_subsets")
    fams  = _read_parquet(families_path, "ti_families") if families_path.exists() else pd.DataFrame()

    for df in (fits, pts, resc, mem, subs, fams):
        if "subset_id" in df.columns: df["subset_id"] = df["subset_id"].astype(str)

    # resolve subset_label -> subset_id (se necessario)
    if subset_id is None and subset_lb is not None:
        ids = subs.loc[subs["subset_label"]==str(subset_lb), "subset_id"].drop_duplicates().astype(str).tolist()
        if len(ids)==0:
            raise ValueError(f"[{spec_id}] subset_label='{subset_lb}' non trovato in {model}")
        if len(ids)>1:
            raise ValueError(f"[{spec_id}] subset_label='{subset_lb}' mappa a più subset_id {ids}. Specifica subset_id.")
        subset_id = ids[0]
    if subset_id is None:
        raise ValueError(f"[{spec_id}] va specificato subset_id o subset_label")

    # Fila del fit selezionata per la retta
    if beta_kind == "raw":
        if beta_raw is None:
            raise ValueError(f"[{spec_id}] beta_kind='raw' richiede 'beta'")
        mask_fit = (fits["beta_kind"]=="raw") & (fits["subset_id"]==subset_id) & (fits["kcol"]==kcol) & np.isclose(fits["beta"], float(beta_raw), atol=5e-4)
    elif beta_kind == "rescaled":
        if anchor not in ("M","G","L","G2","G3","G2b","G2c"):
            raise ValueError(f"[{spec_id}] anchor non valido/assente per rescaled")
        if beta_star is None:
            raise ValueError(f"[{spec_id}] beta_kind='rescaled' richiede 'beta_rescaled_bin'")
        mask_fit = (fits["beta_kind"]=="rescaled") & (fits["subset_id"]==subset_id) & (fits["kcol"]==kcol) & \
                   (fits["anchor"]==anchor) & (fits["ref_stat"]==ref_stat) & np.isclose(fits["beta_rescaled_bin"], float(beta_star), atol=1e-12+1e-9*abs(float(beta_star)))
    else:
        raise ValueError(f"[{spec_id}] beta_kind deve essere 'raw' o 'rescaled'")

    frows = fits.loc[mask_fit].copy()
    if frows.empty:
        raise ValueError(f"[{spec_id}] nessun fit trovato per la specifica")
    if len(frows) > 1 and VERBOSE:
        print(f"[warn] {spec_id}: trovati {len(frows)} fit; uso il primo.")
    fit_row = frows.iloc[0]
    slope = float(fit_row["slope"])
    intercept = float(fit_row["intercept"])

    # subset meta
    srec = subs.loc[subs["subset_id"]==subset_id].head(1).to_dict("records")
    srec = srec[0] if srec else {"subset_label": subset_id, "subset_spec": None}
    subset_label = srec.get("subset_label","subset")
    subset_folder = _subset_folder_name(subset_label, subset_id)

    # base = members ⨝ points
    need_pts = ["TIcurve_id","run_uid","beta",kcol,"chi_chi2"]
    if "trajInit" in pts.columns:
        need_pts.append("trajInit")
    for c in need_pts:
        if c not in pts.columns:
            raise ValueError(f"[ti_points] manca colonna: {c}")
    need_mem = ["TIcurve_id","subset_id","family_id","N","is_used"]
    for c in need_mem:
        if c not in mem.columns:
            raise ValueError(f"[ti_subset_members] manca colonna: {c}")

    mem_sub = mem.loc[mem["subset_id"]==subset_id, need_mem].drop_duplicates()
    base = mem_sub.merge(pts[need_pts], on="TIcurve_id", how="inner")

    # selezione per x (raw/rescaled bin)
    atol = 5e-4
    if beta_kind == "raw":
        sel_x = np.isfinite(base["beta"].astype(float).to_numpy()) & (np.abs(base["beta"].astype(float).to_numpy() - float(beta_raw)) <= atol)
        x_value = float(beta_raw)
        anchor_tag = "raw"
    else:
        # join coi rescaled per prendere beta_rescaled (pre-bin) e poi fare il bin
        need_resc = ["TIcurve_id","run_uid","beta","subset_id","ref_type","ref_stat","beta_rescaled"]
        for c in need_resc:
            if c not in resc.columns:
                raise ValueError(f"[ti_subset_points_rescaled] manca colonna: {c}")
        rsub = resc[(resc["subset_id"]==subset_id) & (resc["ref_type"]==anchor) & (resc["ref_stat"]==ref_stat)][["TIcurve_id","run_uid","beta","beta_rescaled"]].drop_duplicates()
        joined = base.merge(rsub, on=["TIcurve_id","run_uid","beta"], how="inner")
        if joined.empty:
            raise ValueError(f"[{spec_id}] nessun punto rescaled corrispondente")
        # binning
        binned = np.round(joined["beta_rescaled"].astype(float).to_numpy() / step) * step
        sel_x = np.isfinite(binned) & (np.abs(binned - float(beta_star)) <= 1e-12 + 1e-9*abs(float(beta_star)))
        base = joined
        x_value = float(beta_star)
        anchor_tag = f"rescaled/{anchor}"

    # qualità: chi2 sempre; scale2 solo per *_Scaled
    k = base[kcol].astype(float).to_numpy()
    N = base["N"].astype(float).to_numpy()
    chi = base["chi_chi2"].astype(float).to_numpy()
    finite = np.isfinite(k) & np.isfinite(N) & np.isfinite(chi) & (k > 0)
    ok = finite & (chi <= chi2_thr)

    if kcol.endswith("_Scaled"):
        # Se scale2/valid non ci sono, NON mostro punti (conservativo).
        if "scale2" in base.columns and "scale2_valid" in base.columns:
            s2 = base["scale2"].astype(float).to_numpy()
            s2v = base["scale2_valid"].astype(bool).to_numpy()
            ok = ok & s2v & np.isfinite(s2) & (s2 >= scale_thr)
        else:
            ok = np.zeros_like(ok, dtype=bool)

    ok = ok & sel_x
    used_mask   = ok & (base["is_used"]==True)
    unused_mask = ok & (base["is_used"]!=True)

    used   = base.loc[used_mask].copy()
    unused = base.loc[unused_mask].copy() if include_unused else base.iloc[0:0].copy()

    # stessa family fuori subset (opzionale)
    fam_out_used = base.iloc[0:0].copy()
    fam_out_unused = base.iloc[0:0].copy()
    fam_ids = mem_sub["family_id"].dropna().unique().tolist()
    family_id = fam_ids[0] if len(fam_ids)==1 else None
    if include_family_outside and (family_id is not None):
        mem_out = mem[(mem["family_id"]==family_id) & (mem["subset_id"]!=subset_id)][need_mem].drop_duplicates()
        if not mem_out.empty:
            base_out = mem_out.merge(pts[need_pts], on="TIcurve_id", how="inner")
            if beta_kind == "raw":
                sel_out = np.isfinite(base_out["beta"].astype(float).to_numpy()) & (np.abs(base_out["beta"].astype(float).to_numpy() - x_value) <= atol)
            else:
                r_out = resc[(resc["ref_type"]==anchor) & (resc["ref_stat"]==ref_stat) & resc["subset_id"].isin(mem_out["subset_id"].unique())][["TIcurve_id","run_uid","beta","subset_id","beta_rescaled"]].drop_duplicates()
                j2 = base_out.merge(r_out, on=["TIcurve_id","run_uid","beta"], how="inner")
                if not j2.empty:
                    b2 = np.round(j2["beta_rescaled"].astype(float).to_numpy() / step) * step
                    sel_out = np.isfinite(b2) & (np.abs(b2 - x_value) <= 1e-12 + 1e-9*abs(x_value))
                    base_out = j2
                else:
                    sel_out = np.zeros(len(base_out), dtype=bool)

            k2 = base_out[kcol].astype(float).to_numpy() if kcol in base_out.columns else np.full(len(base_out), np.nan)
            N2 = base_out["N"].astype(float).to_numpy() if "N" in base_out.columns else np.full(len(base_out), np.nan)
            chi2 = base_out["chi_chi2"].astype(float).to_numpy() if "chi_chi2" in base_out.columns else np.full(len(base_out), np.nan)
            fin2 = np.isfinite(k2) & np.isfinite(N2) & np.isfinite(chi2) & (k2 > 0)
            ok2 = fin2 & (chi2 <= chi2_thr)
            if kcol.endswith("_Scaled"):
                if "scale2" in base_out.columns and "scale2_valid" in base_out.columns:
                    s22 = base_out["scale2"].astype(float).to_numpy()
                    s2v2 = base_out["scale2_valid"].astype(bool).to_numpy()
                    ok2 = ok2 & s2v2 & np.isfinite(s22) & (s22 >= scale_thr)
                else:
                    ok2 = np.zeros_like(ok2, dtype=bool)
            ok2 = ok2 & sel_out
            fam_out_used   = base_out.loc[ok2 & (base_out["is_used"]==True)].copy()
            fam_out_unused = base_out.loc[ok2 & (base_out["is_used"]!=True)].copy() if include_unused else base_out.iloc[0:0].copy()

    if used.empty and unused.empty and fam_out_used.empty and fam_out_unused.empty:
        raise ValueError(f"[{spec_id}] nessun punto selezionato per il plot")

    # Prepara dataframe "pronto al plot"
    def prep_df(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return df
        out = pd.DataFrame({
            "N": df["N"].astype(float).to_numpy(),
            "k": df[kcol].astype(float).to_numpy()
        })
        if "trajInit" in df.columns:
            out["trajInit"] = df["trajInit"].to_numpy()
        out["y"] = _y_from_k(out["k"].to_numpy(), YMODE)
        # pre-bin beta rescaled, se presente
        if "beta_rescaled" in df.columns:
            out["beta_prebin"] = df["beta_rescaled"].astype(float).to_numpy()
        return out

    used_p = prep_df(used)
    un_p   = prep_df(unused)
    fu_p   = prep_df(fam_out_used)
    fuu_p  = prep_df(fam_out_unused)

    # filtri semplici su x/y (solo sui dati mostrati)
    filters = filters or {}
    def f_apply(d: pd.DataFrame) -> pd.DataFrame:
        if d.empty: return d
        m = np.ones(len(d), dtype=bool)
        if "xmin" in filters: m &= d["N"].to_numpy() >= float(filters["xmin"])
        if "xmax" in filters: m &= d["N"].to_numpy() <= float(filters["xmax"])
        if "ymin" in filters: m &= d["y"].to_numpy() >= float(filters["ymin"])
        if "ymax" in filters: m &= d["y"].to_numpy() <= float(filters["ymax"])
        return d.loc[m]

    used_p = f_apply(used_p)
    un_p   = f_apply(un_p)
    fu_p   = f_apply(fu_p)
    fuu_p  = f_apply(fuu_p)

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    # Colorbar opzionale per rescaled: usa i valori PRE-binning (beta_prebin) dei punti mostrati
    cb_enabled = bool(spec.get("colorbar", False)) and (beta_kind == "rescaled")
    sc_main = None
    vmin = vmax = None
    if cb_enabled:
        vals = np.concatenate([
            used_p["beta_prebin"].to_numpy() if ("beta_prebin" in used_p.columns and not used_p.empty) else np.array([], dtype=float),
            un_p["beta_prebin"].to_numpy()   if ("beta_prebin" in un_p.columns and not un_p.empty) else np.array([], dtype=float),
            fu_p["beta_prebin"].to_numpy()   if ("beta_prebin" in fu_p.columns and not fu_p.empty) else np.array([], dtype=float),
            fuu_p["beta_prebin"].to_numpy()  if ("beta_prebin" in fuu_p.columns and not fuu_p.empty) else np.array([], dtype=float),
        ], dtype=float) if any([
            ("beta_prebin" in df.columns and not df.empty) for df in (used_p,un_p,fu_p,fuu_p)
        ]) else np.array([], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            vmin, vmax = float(vals.min()), float(vals.max())
        else:
            cb_enabled = False

    # mapping edge color per init (optional)
    edge_by_init = bool(spec.get("edge_by_init", edge_by_init_default))
    edge_lw = float(spec.get("edge_lw", 0.6))
    edge_palette = spec.get("edge_palette", {740:"red", 74:"orange", 73:"orange", 72:"purple", 71:"black", 70:"lightgreen", 0:"None", -2:"black"})  # e.g., {740:"red", 74:"orange", 73:"orange", 72:"purple", 71:"black", 70:"lightgreen", 0:"None", -2:"black"}

    def _edge_colors(df):
        if (not edge_by_init) or df.empty:
            return None
        if "trajInit" not in df.columns:
            return None
        vals = df["trajInit"].to_numpy()
        if edge_palette and isinstance(edge_palette, dict):
            def map_color(v):
                # accetta chiavi int o str
                return edge_palette.get(int(v), edge_palette.get(str(v), "k"))
            return [map_color(v) for v in vals]
        # default: mapping deterministico su palette base
        u = pd.unique(vals)
        base_colors = ["C3","C1","C4","C5","C6","C7","C8","C9","C0","k"]
        lut = {u[i]: base_colors[i % len(base_colors)] for i in range(len(u))}
        return [lut[v] for v in vals]

    import matplotlib.cm as cm
    cmap = cm.get_cmap('viridis')

    def sc(ax, df, size, alpha, marker, color=None):
        if df.empty: return None
        edges = _edge_colors(df)
        kw = dict(s=size, alpha=alpha, marker=marker)
        if edges is not None:
            kw["edgecolors"] = edges
            kw["linewidths"] = edge_lw
        if cb_enabled and ("beta_prebin" in df.columns):
            return ax.scatter(df["N"], df["y"], c=df["beta_prebin"].to_numpy(), vmin=vmin, vmax=vmax, cmap=cmap, **kw)
        else:
            if color is not None and edges is None:
                kw["color"] = color
            return ax.scatter(df["N"], df["y"], **kw)

    # draw
    if not un_p.empty:
        s = sc(ax, un_p, 14, 0.25, "o")
        sc_main = sc_main or s
    if not used_p.empty:
        s = sc(ax, used_p, 16, 0.9, "o")
        sc_main = sc_main or s
        xs = used_p["N"].to_numpy()
        x_line = np.linspace(float(np.min(xs)), float(np.max(xs)), 100)
        if YMODE in ("minuslnk","logk"):
            y_line = slope * x_line + intercept
        else:
            y_line = np.exp(-(slope * x_line + intercept))
        ax.plot(x_line, y_line, linestyle="-", linewidth=1.6)
    if not fuu_p.empty:
        s = sc(ax, fuu_p, 14, 0.35, "^", color=None if cb_enabled else "C2")
        sc_main = sc_main or s
    if not fu_p.empty:
        s = sc(ax, fu_p, 16, 0.9, "^", color=None if cb_enabled else "C2")
        sc_main = sc_main or s

    if cb_enabled and (sc_main is not None):
        cbar = plt.colorbar(sc_main, ax=ax)
        cbar.set_label(r"$\tilde{\beta}$ (pre-bin)")

    # assi
    ax.set_xlabel("N")
    if YMODE == "minuslnk":
        ax.set_ylabel(r"$-\ln k$")
    elif YMODE == "logk":
        ax.set_ylabel(r"$\ln k$")
    else:
        ax.set_ylabel("k")
    ax.grid(True, alpha=0.3)
    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)

    # Firma fisica per path
    phys_parts = _phys_signature_parts(fams, mem, subset_id, model)
    subset_folder = _subset_folder_name(subset_label, subset_id)
    out_dir  = _fig_dir(model, "raw" if beta_kind=="raw" else f"rescaled/{anchor}", phys_parts, subset_folder, spec_id)
    out_base = out_dir / f"{FIG_ID}_{SLUG}__{model}__{_sanitize(subset_label)}__{spec_id}__{kcol}"

    fig.savefig(str(out_base)+".png", dpi=DPI, bbox_inches="tight")
    fig.savefig(str(out_base)+".pdf", dpi=DPI, bbox_inches="tight")

    # metadati
    meta = {
        "fig_id": FIG_ID,
        "slug": SLUG,
        "model": model,
        "subset_id": str(subset_id),
        "subset_label": str(subset_label),
        "beta_kind": beta_kind,
        "kcol": kcol,
        "beta": (float(beta_raw) if beta_kind=="raw" else None),
        "beta_rescaled_bin": (float(beta_star) if beta_kind=="rescaled" else None),
        "rescaled_step": (float(step) if beta_kind=="rescaled" else None),
        "anchor": (anchor if beta_kind=="rescaled" else None),
        "ref_stat": (ref_stat if beta_kind=="rescaled" else None),
        "filters": (filters or {}),
        "ymode": YMODE,
        "fit": {
            "slope": float(slope),
            "intercept": float(intercept),
            "r2": float(fit_row.get("r2")) if "r2" in fit_row.index else None,
            "n_points": int(fit_row.get("n_points")) if "n_points" in fit_row.index else None,
            "n_unique_N": int(fit_row.get("n_unique_N")) if "n_unique_N" in fit_row.index else None,
            "N_min": float(fit_row.get("N_min")) if "N_min" in fit_row.index else None,
            "N_max": float(fit_row.get("N_max")) if "N_max" in fit_row.index else None
        },
        "edge_by_init": bool(spec.get("edge_by_init", edge_by_init)),
        "edge_lw": float(spec.get("edge_lw", 0.6)),
        "phys_signature_parts": phys_parts,
        "paths": {
            "out_png": str(out_base)+".png",
            "out_pdf": str(out_base)+".pdf"
        },
        "sources": {
            "ti_dir": str(ti_dir),
            "parquets": {
                "fits": str(fits_path),
                "points": str(points_path),
                "rescaled_points": str(resc_points_p),
                "members": str(members_path),
                "subsets": str(subsets_path),
                "families": str(families_path)
            }
        }
    }
    with open(str(out_base) + "_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    plt.close(fig)
    if VERBOSE:
        print("[out]", out_base)
    return out_base

def main():
    for spec in PLOTS:
        _plot_one(spec)

if __name__ == "__main__":
    main()
