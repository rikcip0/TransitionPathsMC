#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
make_fig_fit_vs_beta_paper.py — script “paper” per tracciare slope vs β (un plot per voce).

Requisiti implementati
- Ogni voce in PLOTS produce un SOLO plot (una sola kcol).
- Selezione esplicita per (model, subset_id, beta_kind[, anchor/ref_stat], kcol).
- Filtri semplici: applicati ESATTAMENTE alle colonne mostrate nel plot (x_col, slope).
- Assi: x="$\beta J$", y="$\beta\,\delta f$" (niente titolo).
- Output sotto Analysis/FigScripts/_figs/TI/F2_fit_vs_beta/… (cartelle per model/anchor/firma_fisica/subset/id).
- Nessun fallback: se MyBasePlots non è importabile → errore; se manca un file richiesto → errore esplicito.
- Overlay teorico opzionale per-plot con `theory_txt` (2 colonne: β, β·δf), tracciato sullo stesso asse y.
- Salvataggio metadati: JSON con stesso basename delle figure + `_meta.json`.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

FIG_ID    = "F2"
SLUG      = "fit_vs_beta"
FIGSIZE   = (3.0, 2.2)
DPI       = 300
REQUIRE_MYBASEPLOTS = True
DATA_ROOT: Optional[str] = None
VERBOSE = True

PLOTS: List[Dict[str, Any]] = [
    {"id":"raw_ER_all_07018d", "model":"ER",  "subset_id":"07018d22622d32a8", "beta_kind":"raw",      "kcol":"kFromChi"},
    {"id":"M_ER_all_07018d",   "model":"ER",  "subset_id":"07018d22622d32a8", "beta_kind":"rescaled", "anchor":"M", "kcol":"kFromChi_InBetween",  "filters":{"xmin":0.5,"xmax":1.0}},
     {"id":"raw_RRG_all_xxx", "model":"RRG", "subset_id":"2354bcad23a43145", "beta_kind":"raw", "kcol":"kFromChi_InBetween_Scaled",
      "theory_txt":"./data.txt", "filters":{"xmin":0.5,"xmax":1.0}},
]

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
    return Path(__file__).resolve().parents[2] / "Data"

def _ti_dir(model: str, data_root: Optional[str]) -> Path:
    return _data_base_dir(data_root) / "MultiPathsMC" / model / "v1" / "ti"

def _fig_dir(model: str, anchor_tag: str, phys_parts: List[str], subset_folder: str, spec_id: str) -> Path:
    base = Path(__file__).resolve().parent / "_figs" / "TI" / f"{FIG_ID}_{SLUG}" / model / anchor_tag
    for p in phys_parts: base = base / p
    base = base / subset_folder / spec_id
    base.mkdir(parents=True, exist_ok=True)
    return base

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

def _families_for_subset(df_plot: pd.DataFrame, members: pd.DataFrame, subset_id: str) -> List[str]:
    fams: List[str] = []
    if "family_id" in df_plot.columns:
        fams = [f for f in df_plot["family_id"].dropna().unique().tolist() if isinstance(f,str)]
    if not fams and "family_id" in members.columns:
        fams = [f for f in members.loc[members["subset_id"]==subset_id,"family_id"].dropna().unique().tolist() if isinstance(f,str)]
    return fams

def _phys_signature_parts(df_plot: pd.DataFrame, families_df: pd.DataFrame, members: pd.DataFrame, subset_id: str) -> List[str]:
    fams = _families_for_subset(df_plot, members, subset_id)
    if len(fams)!=1 or families_df is None or families_df.empty:
        return [f"multiFamilies__n{len(fams)}" if len(fams)>1 else "noFamily"]
    fid = fams[0]
    row = families_df.loc[families_df["family_id"]==fid]
    if row.empty: return [f"family__{fid[:8]}"]
    r = row.iloc[0]
    parts = []
    parts.append(_sanitize(r.get("model_type","model")))
    parts.append(_sanitize(f"C_{_fmt_val(r.get('C'))}__fPosJ_{_fmt_val(r.get('fPosJ'))}"))
    parts.append(_sanitize(str(r.get("fieldType"))))
    parts.append(_sanitize(f"fieldSigma_{_fmt_val(r.get('fieldSigma'))}"))
    parts.append(_sanitize(f"Hext_{_fmt_val(r.get('Hext'))}"))
    nq = None
    for qname in ["normalizedQstar","nQstar","normalized_qstar"]:
        if qname in r.index: nq = r.get(qname); break
    parts.append(_sanitize(f"Hin_{_fmt_val(r.get('Hin'))}__Hout_{_fmt_val(r.get('Hout'))}__nQstar_{_fmt_val(nq)}__{fid[:8]}"))
    return parts

def _apply_filters(df: pd.DataFrame, x_col: str, y_col: str, flt: Optional[Dict[str,float]]) -> pd.DataFrame:
    if not flt: return df
    m = np.ones(len(df), dtype=bool)
    if "xmin" in flt: m &= df[x_col].to_numpy() >= float(flt["xmin"])
    if "xmax" in flt: m &= df[x_col].to_numpy() <= float(flt["xmax"])
    if "ymin" in flt: m &= df[y_col].to_numpy() >= float(flt["ymin"])
    if "ymax" in flt: m &= df[y_col].to_numpy() <= float(flt["ymax"])
    return df.loc[m]

def _load_theory_txt(txt_path: Path) -> np.ndarray:
    if not txt_path.exists():
        raise FileNotFoundError(f"[theory] file non trovato: {txt_path}")
    arr = np.loadtxt(txt_path)
    if arr.ndim==1:
        if arr.size<2: raise ValueError("[theory] file troppo corto (attese >=2 colonne)")
        arr = arr.reshape((-1,2))
    if arr.shape[1] < 2:
        raise ValueError(f"[theory] attese 2 colonne (beta, beta_delta_f); trovate {arr.shape[1]}")
    return arr

def main():
    try:
        ustyle.auto_style(mode="latex", base="paper_base.mplstyle", overlay="overlay_latex.mplstyle")
    except Exception as e:
        raise RuntimeError(f"Errore nel setup stile MyBasePlots: {e}") from e

    for spec in PLOTS:
        model     = str(spec.get("model","ER"))
        subset_id = spec.get("subset_id")
        subset_lb = spec.get("subset_label")
        beta_kind = str(spec.get("beta_kind","raw"))
        kcol      = str(spec.get("kcol","kFromChi"))
        ref_stat  = str(spec.get("ref_stat","mean"))
        anchor    = spec.get("anchor")
        spec_id   = str(spec.get("id","plot"))
        xlim      = spec.get("xlim")
        ylim      = spec.get("ylim")
        filters   = spec.get("filters")

        ti_dir = _ti_dir(model, DATA_ROOT)
        fits_path     = ti_dir / "ti_linear_fits.parquet"
        subsets_path  = ti_dir / "ti_family_subsets.parquet"
        members_path  = ti_dir / "ti_subset_members.parquet"
        families_path = ti_dir / "ti_families.parquet"

        fits     = _read_parquet(fits_path, "ti_linear_fits")
        subsets  = _read_parquet(subsets_path, "ti_family_subsets")
        members  = _read_parquet(members_path, "ti_subset_members")
        families = _read_parquet(families_path, "ti_families") if families_path.exists() else pd.DataFrame()

        for df in (fits, subsets, members, families):
            if "subset_id" in df.columns: df["subset_id"] = df["subset_id"].astype(str)

        if VERBOSE:
            print(f"[ti_dir:{model}]", ti_dir)

        if subset_id is None and subset_lb is not None:
            ids = subsets.loc[subsets["subset_label"]==str(subset_lb), "subset_id"].drop_duplicates().astype(str).tolist()
            if len(ids)==0:
                raise ValueError(f"[{spec_id}] subset_label='{subset_lb}' non trovato in {model}")
            if len(ids)>1:
                raise ValueError(f"[{spec_id}] subset_label='{subset_lb}' mappa a più subset_id {ids} in {model}. Specifica subset_id.")
            subset_id = ids[0]

        if subset_id is None:
            raise ValueError(f"[{spec_id}] va specificato subset_id o subset_label")

        if kcol not in fits["kcol"].unique().tolist():
            raise ValueError(f"[{spec_id}] kcol='{kcol}' non presente in ti_linear_fits ({model})")

        if beta_kind == "raw":
            x_col = "beta"
            sel = (fits["beta_kind"]=="raw") & (fits["subset_id"]==subset_id) & (fits["kcol"]==kcol)
            cols = ["family_id","subset_id","beta","kcol","slope","slope_stderr"] if "slope_stderr" in fits.columns else ["family_id","subset_id","beta","kcol","slope"]
            df = fits.loc[sel, cols].dropna(subset=[x_col,"slope"]).copy()
            anchor_tag = "raw"
        elif beta_kind == "rescaled":
            if anchor not in ("M","G"):
                raise ValueError(f"[{spec_id}] beta_kind='rescaled' ma anchor non valido/assente")
            need = ["beta_rescaled_bin","anchor","ref_stat"]
            if not set(need).issubset(fits.columns):
                raise ValueError(f"[{spec_id}] ti_linear_fits manca colonne {need}")
            x_col = "beta_rescaled_bin"
            sel = (fits["beta_kind"]=="rescaled") & (fits["subset_id"]==subset_id) & (fits["kcol"]==kcol) & (fits["anchor"]==anchor) & (fits["ref_stat"]==ref_stat)
            cols = ["family_id","subset_id","beta_rescaled_bin","kcol","slope","slope_stderr","anchor","ref_stat"] if "slope_stderr" in fits.columns else ["family_id","subset_id","beta_rescaled_bin","kcol","slope","anchor","ref_stat"]
            df = fits.loc[sel, cols].dropna(subset=[x_col,"slope"]).copy()
            anchor_tag = f"rescaled/{anchor}"
        else:
            raise ValueError(f"[{spec_id}] beta_kind deve essere 'raw' o 'rescaled'")

        if df.empty:
            raise ValueError(f"[{spec_id}] nessun dato selezionato per (model={model}, subset_id={subset_id}, beta_kind={beta_kind}, kcol={kcol})")

        df = _apply_filters(df, x_col, "slope", filters)

        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
        x = df[x_col].astype(float).to_numpy()
        y = df["slope"].astype(float).to_numpy()
        order = np.argsort(x); x,y = x[order], y[order]
        if "slope_stderr" in df.columns:
            yerr = df["slope_stderr"].astype(float).to_numpy()[order]
            ax.errorbar(x, y, yerr=yerr, fmt="-o", capsize=2)
        else:
            ax.plot(x, y, "-o")

        if "theory_txt" in spec and spec["theory_txt"]:
            tpath = Path(spec["theory_txt"])
            if not tpath.is_absolute():
                tpath = (Path(__file__).resolve().parent / tpath).resolve()
            arr = _load_theory_txt(tpath)
            xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
            m = np.isfinite(arr[:,0]) & np.isfinite(arr[:,1]) & (arr[:,0]>=xmin) & (arr[:,0]<=xmax)
            if not np.any(m):
                raise ValueError(f"[{spec_id}] theory_txt non ha punti nel range x dei dati [{xmin},{xmax}]")
            bx, by = arr[m,0], arr[m,1]
            ord2 = np.argsort(bx)
            ax.plot(bx[ord2], by[ord2], "--", lw=1.2, label="theory")
            ax.legend(frameon=False, loc="best")

        ax.set_xlabel(r"$\beta J$")
        ax.set_ylabel(r"$\beta\,\delta f$")
        ax.grid(True, alpha=0.3)
        if xlim: ax.set_xlim(*xlim)
        if ylim: ax.set_ylim(*ylim)

        try:
            families = _read_parquet(_ti_dir(model, DATA_ROOT) / "ti_families.parquet", "ti_families")
        except Exception:
            families = pd.DataFrame()
        try:
            members  = _read_parquet(_ti_dir(model, DATA_ROOT) / "ti_subset_members.parquet", "ti_subset_members")
        except Exception:
            members = pd.DataFrame()

        def _phys_signature_parts_safe(df_plot: pd.DataFrame, families_df: pd.DataFrame, members: pd.DataFrame, subset_id: str) -> List[str]:
            fams: List[str] = []
            if "family_id" in df_plot.columns:
                fams = [f for f in df_plot["family_id"].dropna().unique().tolist() if isinstance(f,str)]
            if not fams and "family_id" in members.columns:
                fams = [f for f in members.loc[members["subset_id"]==subset_id,"family_id"].dropna().unique().tolist() if isinstance(f,str)]
            if len(fams)!=1 or families_df is None or families_df.empty:
                return [f"multiFamilies__n{len(fams)}" if len(fams)>1 else "noFamily"]
            fid = fams[0]
            row = families_df.loc[families_df["family_id"]==fid]
            if row.empty: return [f"family__{fid[:8]}"]
            r = row.iloc[0]
            parts = []
            parts.append(_sanitize(r.get("model_type","model")))
            parts.append(_sanitize(f"C_{_fmt_val(r.get('C'))}__fPosJ_{_fmt_val(r.get('fPosJ'))}"))
            parts.append(_sanitize(str(r.get("fieldType"))))
            parts.append(_sanitize(f"fieldSigma_{_fmt_val(r.get('fieldSigma'))}"))
            parts.append(_sanitize(f"Hext_{_fmt_val(r.get('Hext'))}"))
            nq = None
            for qname in ["normalizedQstar","nQstar","normalized_qstar"]:
                if qname in r.index: nq = r.get(qname); break
            parts.append(_sanitize(f"Hin_{_fmt_val(r.get('Hin'))}__Hout_{_fmt_val(r.get('Hout'))}__nQstar_{_fmt_val(nq)}__{fid[:8]}"))
            return parts

        try:
            phys_parts = _phys_signature_parts_safe(df, families, members, subset_id)
        except Exception:
            phys_parts = ["noFamily"]

        try:
            subsets = _read_parquet(_ti_dir(model, DATA_ROOT) / "ti_family_subsets.parquet", "ti_family_subsets")
            row = subsets.loc[subsets["subset_id"]==str(subset_id)]
            sub_label = row.iloc[0]["subset_label"] if not row.empty else "subset"
        except Exception:
            sub_label = "subset"
        subset_folder = _subset_folder_name(str(sub_label), str(subset_id))

        anchor_tag = "raw" if beta_kind=="raw" else f"rescaled/{anchor}"
        out_dir  = _fig_dir(model, anchor_tag, phys_parts, subset_folder, spec_id)
        out_base = out_dir / f"{FIG_ID}_{SLUG}__{model}__{_sanitize(str(sub_label))}__{spec_id}__{kcol}"
        fig.savefig(str(out_base)+".png", dpi=DPI, bbox_inches="tight")
        fig.savefig(str(out_base)+".pdf", dpi=DPI, bbox_inches="tight")

        # --- write metadata JSON ---
        meta = {
            "fig_id": FIG_ID,
            "slug": SLUG,
            "model": model,
            "subset_id": str(subset_id),
            "subset_label": str(sub_label),
            "beta_kind": beta_kind,
            "kcol": kcol,
            "anchor": (anchor if beta_kind=="rescaled" else None),
            "ref_stat": (ref_stat if beta_kind=="rescaled" else None),
            "filters": (filters or {}),
            "x_col": x_col,
            "data": {
                "n_points": int(len(df)),
                "x_min": (float(np.nanmin(x)) if len(df) else None),
                "x_max": (float(np.nanmax(x)) if len(df) else None),
                "y_min": (float(np.nanmin(y)) if len(df) else None),
                "y_max": (float(np.nanmax(y)) if len(df) else None)
            },
            "axes": {
                "xlim": list(ax.get_xlim()),
                "ylim": list(ax.get_ylim())
            },
            "phys_signature_parts": phys_parts,
            "paths": {
                "out_png": str(out_base)+".png",
                "out_pdf": str(out_base)+".pdf"
            },
            "sources": {
                "ti_dir": str(ti_dir),
                "parquets": {
                    "fits": str(fits_path),
                    "subsets": str(subsets_path),
                    "members": str(members_path),
                    "families": str(families_path)
                }
            }
        }
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
        with open(str(out_base) + "_meta.json", "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(meta), f, ensure_ascii=False, indent=2)

        plt.close(fig)
        if VERBOSE:
            print("[out]", out_base)

if __name__ == "__main__":
    main()
