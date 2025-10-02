#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fit_rates_vs_N.py  (v7 – CLI pulita, niente legacy)
---------------------------------------------------
- Fit lineari di -ln(k) vs N su punti RAW e RESCALED.
- Filtri qualità OBBLIGATORI:
    RAW:      chi_chi2 <= chi2_threshold
    RESCALED: chi_chi2 <= chi2_threshold, scale2_valid == True, scale2 >= scale_threshold
- Default:
    * k = [kFromChi, kFromChi_InBetween, kFromChi_InBetween_Scaled]
    * anchors (RESCALED) = [M, G]
    * ref_stat (RESCALED) = mean
    * Δ beta* (binning) = 0.025

Esempi:
  python3 fit_rates_vs_N.py --model ER --subset-label 'N>30' -v
  python3 fit_rates_vs_N.py --model ER --subset-label 'N>30' --anchors M,G --ref-stats mean -v
  python3 fit_rates_vs_N.py --model ER --subset-id 3d770592476e771c --kcols kFromChi -v
"""
import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import sys

# ------------------------------
# Utils
# ------------------------------

def _path_base(model: str, data_root: Path|None) -> Path:
    root = Path("../../Data") if data_root is None else Path(data_root)
    return root / "MultiPathsMC" / model / "v1" / "ti"

def _load_parquet(p: Path, what: str) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(f"[{what}] non trovato: {p}")
    return pd.read_parquet(p)

def _ensure_cols(df: pd.DataFrame, cols: list[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] colonne mancanti: {missing}")

def _finite_mask(*arrs):
    m = np.ones(len(arrs[0]), dtype=bool)
    for a in arrs:
        m &= np.isfinite(a)
    return m

def _linear_fit(x: np.ndarray, y: np.ndarray):
    n = len(x)
    if n < 2:
        return (np.nan, np.nan, np.nan, np.nan, n)
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    resid = y - yhat
    sse = np.sum(resid**2)
    sst = np.sum((y - np.mean(y))**2)
    r2 = 1.0 - (sse / sst) if sst > 0 else np.nan
    if n > 2:
        s2 = sse / (n - 2)
        sx2 = np.sum((x - np.mean(x))**2)
        slope_stderr = (s2 / sx2) ** 0.5 if sx2 > 0 else np.nan
    else:
        slope_stderr = np.nan
    return (slope, intercept, slope_stderr, r2, n)

def _bin_rescaled(beta_rescaled: pd.Series, step: float) -> pd.Series:
    return (np.round(beta_rescaled.values / step) * step).astype(float)

def _anchor_list(arg: str|None) -> list[str]:
    all_anchors = ["M","L","G","G2","G3","G2b","G2c"]
    if arg is None or arg.strip() == "":
        return ["M","G"]
    toks = [t.strip() for t in arg.split(",") if t.strip()]
    for t in toks:
        if t not in all_anchors:
            raise ValueError(f"Anchor sconosciuto: {t}. Validi: {all_anchors}")
    return toks

def _refstats_list(arg: str|None) -> list[str]:
    all_stats = ["mean","median"]
    if arg is None or arg.strip() == "":
        return ["mean"]
    toks = [t.strip() for t in arg.split(",") if t.strip()]
    for t in toks:
        if t not in all_stats:
            raise ValueError(f"ref_stat sconosciuta: {t}. Valide: {all_stats}")
    return toks

def _kcols_list(arg: str|None) -> list[str]:
    default = ["kFromChi","kFromChi_InBetween","kFromChi_InBetween_Scaled"]
    if arg is None or arg.strip() == "":
        return default
    return [t.strip() for t in arg.split(",") if t.strip()]

# ------------------------------
# Core (RAW)
# ------------------------------

def fit_raw(points: pd.DataFrame,
            members: pd.DataFrame,
            subset_ids: list[str],
            family_ids: list[str]|None,
            kcol: str,
            include_unused: bool,
            min_unique_N: int,
            chi2_thr: float,
            verbose: bool) -> list[dict]:

    _ensure_cols(points, ["TIcurve_id","run_uid","beta",kcol,"chi_chi2"], "ti_points")  # chi_chi2 OBBLIGATORIO
    need_m = ["TIcurve_id","subset_id","family_id","N","is_used"]
    _ensure_cols(members, need_m, "ti_subset_members")

    mkey = members["subset_id"].isin(subset_ids)
    if family_ids:
        mkey &= members["family_id"].isin(family_ids)
    if not include_unused:
        mkey &= (members["is_used"]==True)
    mem = members.loc[mkey, ["TIcurve_id","subset_id","family_id","N","is_used"]].drop_duplicates()

    use_cols = ["TIcurve_id","run_uid","beta",kcol,"chi_chi2"]
    df = points[use_cols].merge(mem, on="TIcurve_id", how="inner")

    # filtri
    k = df[kcol].to_numpy(dtype=float)
    N = df["N"].to_numpy(dtype=float)
    b = df["beta"].to_numpy(dtype=float)
    chi = df["chi_chi2"].to_numpy(dtype=float)
    good = (k > 0) & _finite_mask(k, N, b, chi) & (chi <= chi2_thr)

    df = df.loc[good].copy()

    fits = []
    group_keys = ["subset_id"]
    if family_ids:
        group_keys.append("family_id")
    for keys, gsub in df.groupby(group_keys):
        if isinstance(keys, tuple):
            subset_id = keys[0]
            fam = keys[1] if len(keys)>1 else None
        else:
            subset_id = keys
            fam = None
        for b_val, g in gsub.groupby("beta"):
            Ns = g["N"].to_numpy(float)
            if len(np.unique(Ns)) < min_unique_N:
                continue
            y = -np.log(g[kcol].to_numpy(float))
            slope, inter, slope_se, r2, n = _linear_fit(Ns, y)
            if verbose:
                print(f"[RAW] subset={subset_id} beta={b_val:.6g} k={kcol} n={n} slope={slope:.6g} r2={r2:.3f}")
            fits.append({
                "beta_kind": "raw",
                "subset_id": subset_id,
                "family_id": fam,
                "anchor": None,
                "ref_stat": None,
                "beta": float(b_val),
                "beta_rescaled_bin": None,
                "kcol": kcol,
                "slope": float(slope),
                "slope_stderr": float(slope_se) if np.isfinite(slope_se) else np.nan,
                "intercept": float(inter),
                "r2": float(r2) if np.isfinite(r2) else np.nan,
                "n_points": int(n),
                "n_unique_N": int(len(np.unique(Ns))),
                "N_min": float(np.min(Ns)),
                "N_max": float(np.max(Ns)),
            })
    return fits

# ------------------------------
# Core (RESCALED)
# ------------------------------

def fit_rescaled(points: pd.DataFrame,
                 rescaled_points: pd.DataFrame,
                 members: pd.DataFrame,
                 subset_ids: list[str],
                 family_ids: list[str]|None,
                 kcol: str,
                 anchors: list[str],
                 ref_stats: list[str],
                 step: float,
                 include_unused: bool,
                 min_unique_N: int,
                 chi2_thr: float,
                 scale_thr: float,
                 verbose: bool) -> list[dict]:

    _ensure_cols(points, ["TIcurve_id","run_uid","beta",kcol,"chi_chi2","scale2","scale2_valid"], "ti_points")
    need_r = ["TIcurve_id","run_uid","beta","subset_id","ref_type","ref_stat","beta_rescaled"]
    _ensure_cols(rescaled_points, need_r, "ti_subset_points_rescaled")
    need_m = ["TIcurve_id","subset_id","family_id","N","is_used"]
    _ensure_cols(members, need_m, "ti_subset_members")

    mkey = members["subset_id"].isin(subset_ids)
    if family_ids:
        mkey &= members["family_id"].isin(family_ids)
    if not include_unused:
        mkey &= (members["is_used"]==True)
    mem = members.loc[mkey, ["TIcurve_id","subset_id","family_id","N","is_used"]].drop_duplicates()

    rp = rescaled_points.copy()
    rkey = rp["subset_id"].isin(subset_ids) & rp["ref_type"].isin(anchors) & rp["ref_stat"].isin(ref_stats)
    if family_ids and ("family_id" in rp.columns):
        rkey &= rp["family_id"].isin(family_ids)
    rp = rp.loc[rkey]

    join_keys = ["TIcurve_id","run_uid","beta"]
    df = rp.merge(points[join_keys + [kcol,"chi_chi2","scale2","scale2_valid"]], on=join_keys, how="inner")
    df = df.merge(mem, on=["TIcurve_id","subset_id"], how="inner")

    k = df[kcol].to_numpy(dtype=float)
    N = df["N"].to_numpy(dtype=float)
    br = df["beta_rescaled"].to_numpy(dtype=float)
    chi = df["chi_chi2"].to_numpy(dtype=float)
    s2 = df["scale2"].to_numpy(dtype=float)
    s2v = df["scale2_valid"].astype(bool).to_numpy()
    good = (k > 0) & _finite_mask(k, N, br, chi, s2) & (chi <= chi2_thr) & s2v & (s2 >= scale_thr)

    df = df.loc[good].copy()
    df["beta_rescaled_bin"] = _bin_rescaled(df["beta_rescaled"], step)

    fits = []
    group_keys = ["subset_id","ref_type","ref_stat"]
    if family_ids:
        group_keys.append("family_id")
    for keys, gsub in df.groupby(group_keys):
        if isinstance(keys, tuple):
            subset_id, ref_type, ref_stat = keys[:3]
            fam = keys[3] if len(keys)>3 else None
        else:
            subset_id, ref_type, ref_stat = keys, None, None
            fam = None
        for bbin, g in gsub.groupby("beta_rescaled_bin"):
            Ns = g["N"].to_numpy(float)
            if len(np.unique(Ns)) < min_unique_N:
                continue
            y = -np.log(g[kcol].to_numpy(float))
            slope, inter, slope_se, r2, n = _linear_fit(Ns, y)
            if verbose:
                print(f"[RESC] subset={subset_id} anchor={ref_type}/{ref_stat} beta*={bbin:.6g} k={kcol} n={n} slope={slope:.6g} r2={r2:.3f}")
            fits.append({
                "beta_kind": "rescaled",
                "subset_id": subset_id,
                "family_id": fam,
                "anchor": ref_type,
                "ref_stat": ref_stat,
                "beta": None,
                "beta_rescaled_bin": float(bbin),
                "kcol": kcol,
                "slope": float(slope),
                "slope_stderr": float(slope_se) if np.isfinite(slope_se) else np.nan,
                "intercept": float(inter),
                "r2": float(r2) if np.isfinite(r2) else np.nan,
                "n_points": int(n),
                "n_unique_N": int(len(np.unique(Ns))),
                "N_min": float(np.min(Ns)),
                "N_max": float(np.max(Ns)),
            })
    return fits

# ------------------------------
# CLI
# ------------------------------

def _parser() -> argparse.ArgumentParser:
    ep = (
"Esempi:\n"
"  python3 fit_rates_vs_N.py --model ER --subset-label 'N>30' -v\n"
"  python3 fit_rates_vs_N.py --model ER --subset-label 'N>30' --anchors M,G --ref-stats mean -v\n"
"  python3 fit_rates_vs_N.py --model ER --subset-id <subset_id> --kcols kFromChi -v\n"
)
    ap = argparse.ArgumentParser(
        description="Fit -ln(k) vs N (RAW + RESCALED). Niente flag legacy; usare i parametri esatti.",
        epilog=ep,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", required=True, help="Modello (es. ER, RRG, realGraphs/ZKC, ...)")
    ap.add_argument("--data-root", default=None, help="Root Data (default: ../../Data)")
    ap.add_argument("--subset-label", default="all", help="Label del subset (es. all, 'N>30')")
    ap.add_argument("--subset-id", default=None, help="Subset ID (sovrascrive label)")
    ap.add_argument("--family-id", default=None, help="Family ID (opzionale)")
    ap.add_argument("--kcols", default=None, help="Lista colonne k (comma). Default: tre standard")
    ap.add_argument("--anchors", default=None, help="Anchors per RESCALED (default: M,G). Usa stringa vuota per disattivare RESCALED")
    ap.add_argument("--ref-stats", default=None, help="Statistiche per anchor (default: mean; usare 'mean,median' per entrambe)")
    ap.add_argument("--rescaled-step", type=float, default=0.025, help="Δ per binning di beta_rescaled")
    ap.add_argument("--min-unique-N", type=int, default=2, help="Min # di N distinti nel fit")
    ap.add_argument("--include-unused", action="store_true", help="Includi anche TIcurve con is_used=False")
    ap.add_argument("--chi2-threshold", type=float, default=0.43, help="Soglia su chi_chi2")
    ap.add_argument("--scale-threshold", type=float, default=0.33, help="Soglia su scale2 (solo RESCALED)")
    ap.add_argument("--outname", default="ti_linear_fits.parquet", help="Nome file output parquet")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def main():
    ap = _parser()
    ns = ap.parse_args()

    base = _path_base(ns.model, ns.data_root)
    points_path   = base / "ti_points.parquet"
    members_path  = base / "ti_subset_members.parquet"
    rescaled_path = base / "ti_subset_points_rescaled.parquet"
    subsets_path  = base / "ti_family_subsets.parquet"

    print("[in ]", points_path)
    print("[in ]", members_path)
    print("[in ]", rescaled_path)
    print("[in ]", subsets_path)

    points   = _load_parquet(points_path, "ti_points")
    members  = _load_parquet(members_path, "ti_subset_members")
    rescaled = _load_parquet(rescaled_path, "ti_subset_points_rescaled")
    subsets  = _load_parquet(subsets_path, "ti_family_subsets")

    # Enforce qualità richieste
    _ensure_cols(points, ["chi_chi2"], "ti_points (RAW quality)")

    anchors = _anchor_list(ns.anchors)
    do_rescaled = len(anchors) > 0
    if do_rescaled:
        _ensure_cols(points, ["scale2","scale2_valid"], "ti_points (RESCALED quality)")

    # subset target
    if ns.subset_id:
        target_subset_ids = [ns.subset_id]
    else:
        _ensure_cols(subsets, ["subset_id","subset_label"], "ti_family_subsets")
        target_subset_ids = subsets.loc[subsets["subset_label"]==ns.subset_label, "subset_id"].drop_duplicates().tolist()
        print(f"[subset] label='{ns.subset_label}' -> {len(target_subset_ids)} subset_id")

    fam_ids = [ns.family_id] if ns.family_id else None
    kcols = _kcols_list(ns.kcols)
    ref_stats = _refstats_list(ns.ref_stats)

    if ns.verbose:
        print(f"[conf] kcols={kcols} anchors={anchors} ref_stats={ref_stats} step={ns.rescaled_step} include_unused={ns.include_unused}")
        print(f"[thr ] chi2<= {ns.chi2_threshold} ; scale2>= {ns.scale_threshold} (solo RESCALED)")
        print(f"[sizes] points={len(points)} members={len(members)} rescaled={len(rescaled)}")

    fits_all = []
    # RAW
    for kcol in kcols:
        if kcol not in points.columns:
            print(f"[skip] colonna k assente in ti_points: {kcol}")
            continue
        fits_all.extend(
            fit_raw(points, members, target_subset_ids, fam_ids, kcol, ns.include_unused, ns.min_unique_N, ns.chi2_threshold, ns.verbose)
        )

    # RESCALED
    if do_rescaled:
        for kcol in kcols:
            if kcol not in points.columns:
                continue
            fits_all.extend(
                fit_rescaled(points, rescaled, members, target_subset_ids, fam_ids, kcol, anchors, ref_stats,
                             ns.rescaled_step, ns.include_unused, ns.min_unique_N, ns.chi2_threshold, ns.scale_threshold, ns.verbose)
            )

    out_df = pd.DataFrame(fits_all)
    out_path = base / ns.outname
    if len(out_df) == 0:
        print("[warn] Nessun fit calcolato: controlla subset/anchors/kcols o filtri troppo stringenti.")
        return

    out_df["computed_at"] = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    out_df["analysis_rev"] = "unversioned"
    out_df.to_parquet(out_path, index=False)

    print(f"[done] {out_path} (rows={len(out_df)})")

if __name__ == "__main__":
    main()
