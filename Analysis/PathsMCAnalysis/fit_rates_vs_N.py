#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fit_rates_vs_N.py  (v11 – membership + filtro scale2 anche per RAW quando k == Scaled)
-------------------------------------------------------------------------------------
Cosa fa:
- Esegue fit lineari di Y(N) dove Y ≡ -ln(k) (o equivalenti, a livello di fit fissiamo k-colonna).
- RAW: asse x = beta (non riscalata). RESCALED: asse x = beta* binnata.
- Regole filtri per i punti del fit (sia per RAW che per RESCALED):
  * Validità numerica: k>0 e finitezza di (k, N, beta, chi[, beta*])
  * χ: kFromChi -> chi_chi ; InBetween/Scaled -> chi_chi2 ; soglia --chi2-threshold
  * scale2: SOLO per kFromChi_InBetween_Scaled, preso da runs_results; soglia --scale-threshold
            (vale anche in RAW, come concordato).

Output:
  - ti_linear_fits.parquet                     (una riga per (subset, kcol, x-bin))
  - ti_linear_fit_members.parquet              (una riga per OGNI punto candidato al singolo fit)
    colonne chiave:
      beta_kind (raw|rescaled), subset_id, family_id (se presente), kcol,
      [raw] beta  |  [rescaled] anchor, ref_stat, beta_rescaled_bin
    flags: used_by_fit, excluded_chi, excluded_scale, is_used_subset
    numeriche: k_value, chi_value, scale2, beta_point, beta_rescaled, N
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

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
    sse = float(np.sum(resid**2))
    sst = float(np.sum((y - float(np.mean(y)))**2))
    r2 = 1.0 - (sse / sst) if sst > 0 else np.nan
    if n > 2:
        s2 = sse / (n - 2)
        sx2 = float(np.sum((x - float(np.mean(x)))**2))
        slope_stderr = (s2 / sx2) ** 0.5 if sx2 > 0 else np.nan
    else:
        slope_stderr = np.nan
    return (float(slope), float(intercept), float(slope_stderr), float(r2), int(n))

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
# Core
# ------------------------------

def fit_and_members_raw(points: pd.DataFrame,
                        runs_results: pd.DataFrame,
                        members_all: pd.DataFrame,
                        subset_ids: list[str],
                        family_ids: list[str]|None,
                        kcol: str,
                        include_unused: bool,
                        min_unique_N: int,
                        chi_col: str,
                        chi2_thr: float,
                        scale_thr: float,
                        verbose: bool):
    """Ritorna (fits_rows, members_rows) per RAW, con filtro scale2 se kcol == Scaled."""
    _ensure_cols(points, ["TIcurve_id","run_uid","beta",kcol,chi_col], "ti_points")
    need_m = ["TIcurve_id","subset_id","family_id","N","is_used"]
    _ensure_cols(members_all, need_m, "ti_subset_members")

    base = members_all[need_m].merge(points[["TIcurve_id","run_uid","beta",kcol,chi_col]], on="TIcurve_id", how="inner")

    sel = base["subset_id"].isin(subset_ids)
    if family_ids:
        sel &= base["family_id"].isin(family_ids)
    base = base.loc[sel].copy()

    # merge scale2 per RAW se kcol è Scaled
    need_scale = (kcol == "kFromChi_InBetween_Scaled")
    if need_scale:
        _ensure_cols(runs_results, ["run_uid","scale2"], "runs_results")
        base = base.merge(runs_results[["run_uid","scale2"]].drop_duplicates(), on="run_uid", how="left")
    else:
        base["scale2"] = np.nan

    k = base[kcol].to_numpy(dtype=float)
    N = base["N"].to_numpy(dtype=float)
    b = base["beta"].to_numpy(dtype=float)
    chi = base[chi_col].to_numpy(dtype=float)
    ok_basic = (k > 0) & _finite_mask(k, N, b, chi)
    base = base.loc[ok_basic].copy()

    fits, members = [], []
    group_keys = ["subset_id"]
    if family_ids: group_keys.append("family_id")

    for keys, gsub in base.groupby(group_keys):
        if isinstance(keys, tuple):
            subset_id = keys[0]; fam = keys[1] if len(keys)>1 else None
        else:
            subset_id = keys; fam = None

        for b_val, g in gsub.groupby("beta"):
            is_used_subset = g["is_used"].astype(bool).to_numpy()
            considered = is_used_subset | include_unused
            chi_ok = (g[chi_col].to_numpy(float) <= chi2_thr)

            if need_scale:
                s2 = g["scale2"].to_numpy(float)
                scale_ok = np.isfinite(s2) & (s2 >= scale_thr)
            else:
                scale_ok = np.ones(len(g), dtype=bool)

            used_by_fit = considered & chi_ok & scale_ok

            if len(np.unique(g.loc[used_by_fit, "N"].to_numpy(float))) < min_unique_N:
                continue

            y = -np.log(g.loc[used_by_fit, kcol].to_numpy(float))
            Ns = g.loc[used_by_fit, "N"].to_numpy(float)
            slope, inter, slope_se, r2, n = _linear_fit(Ns, y)
            if verbose:
                print(f"[RAW] subset={subset_id} beta={b_val:.6g} k={kcol} n={n} slope={slope:.6g} r2={r2:.3f}")
            fits.append({
                "beta_kind": "raw", "subset_id": subset_id, "family_id": fam, "anchor": None, "ref_stat": None,
                "beta": float(b_val), "beta_rescaled_bin": None, "kcol": kcol,
                "slope": float(slope), "slope_stderr": float(slope_se) if np.isfinite(slope_se) else np.nan,
                "intercept": float(inter), "r2": float(r2) if np.isfinite(r2) else np.nan,
                "n_points": int(n), "n_unique_N": int(len(np.unique(Ns))),
                "N_min": float(np.min(Ns)), "N_max": float(np.max(Ns)),
            })

            excl_chi = considered & (~chi_ok)
            excl_scale = considered & (~scale_ok) if need_scale else np.zeros(len(g), dtype=bool)
            rows = pd.DataFrame({
                "beta_kind": "raw", "subset_id": subset_id, "family_id": fam, "anchor": None, "ref_stat": None, "kcol": kcol,
                "beta": float(b_val), "beta_rescaled_bin": np.nan,
                "TIcurve_id": g["TIcurve_id"].astype(str).values, "run_uid": g["run_uid"].astype(str).values,
                "N": g["N"].astype(float).values, "beta_point": g["beta"].astype(float).values,
                "beta_rescaled": np.nan, "k_value": g[kcol].astype(float).values, "chi_value": g[chi_col].astype(float).values,
                "scale2": (g["scale2"].astype(float).values if need_scale else np.full(len(g), np.nan)),
                "is_used_subset": is_used_subset, "used_by_fit": used_by_fit,
                "excluded_chi": excl_chi, "excluded_scale": excl_scale,
                "chi2_threshold": float(chi2_thr), "scale_threshold": (float(scale_thr) if need_scale else np.nan),
                "rescaled_step": np.nan,
            })
            members.append(rows)

    return fits, (pd.concat(members, ignore_index=True) if members else pd.DataFrame())

def fit_and_members_rescaled(points: pd.DataFrame,
                             rescaled_points: pd.DataFrame,
                             members_all: pd.DataFrame,
                             runs_results: pd.DataFrame,
                             subset_ids: list[str],
                             family_ids: list[str]|None,
                             kcol: str,
                             anchors: list[str],
                             ref_stats: list[str],
                             step: float,
                             include_unused: bool,
                             min_unique_N: int,
                             chi_col: str,
                             chi2_thr: float,
                             scale_thr: float,
                             verbose: bool):
    """Ritorna (fits_rows, members_rows) per RESCALED (anchor/ref_stat)."""
    _ensure_cols(points, ["TIcurve_id","run_uid","beta",kcol,chi_col], "ti_points")
    need_r = ["TIcurve_id","run_uid","beta","subset_id","ref_type","ref_stat","beta_rescaled"]
    _ensure_cols(rescaled_points, need_r, "ti_subset_points_rescaled")
    need_m = ["TIcurve_id","subset_id","family_id","N","is_used"]
    _ensure_cols(members_all, need_m, "ti_subset_members")
    if kcol == "kFromChi_InBetween_Scaled":
        _ensure_cols(runs_results, ["run_uid","scale2"], "runs_results")

    base_m = members_all[need_m].copy()
    rp = rescaled_points.copy()
    rkey = rp["subset_id"].isin(subset_ids) & rp["ref_type"].isin(anchors) & rp["ref_stat"].isin(ref_stats)
    if family_ids and ("family_id" in rp.columns):
        rkey &= rp["family_id"].isin(family_ids)
    rp = rp.loc[rkey, need_r].drop_duplicates()

    join_keys = ["TIcurve_id","run_uid","beta"]
    base = rp.merge(points[join_keys + [kcol,chi_col]], on=join_keys, how="inner")
    base = base.merge(base_m, on=["TIcurve_id","subset_id"], how="inner")

    k = base[kcol].to_numpy(dtype=float)
    N = base["N"].to_numpy(dtype=float)
    br = base["beta_rescaled"].to_numpy(dtype=float)
    chi = base[chi_col].to_numpy(dtype=float)
    ok_basic = (k > 0) & _finite_mask(k, N, br, chi)
    base = base.loc[ok_basic].copy()

    need_scale = (kcol == "kFromChi_InBetween_Scaled")
    if need_scale:
        base = base.merge(runs_results[["run_uid","scale2"]].drop_duplicates(), on="run_uid", how="left")

    base["beta_rescaled_bin"] = _bin_rescaled(base["beta_rescaled"], step)

    fits, members = [], []
    group_keys = ["subset_id","ref_type","ref_stat"]
    if family_ids: group_keys.append("family_id")

    for keys, gsub in base.groupby(group_keys):
        if isinstance(keys, tuple):
            subset_id, ref_type, ref_stat = keys[:3]; fam = keys[3] if len(keys)>3 else None
        else:
            subset_id, ref_type, ref_stat = keys; fam = None

        for bbin, g in gsub.groupby("beta_rescaled_bin"):
            is_used_subset = g["is_used"].astype(bool).to_numpy()
            considered = is_used_subset | include_unused
            chi_ok = (g[chi_col].to_numpy(float) <= chi2_thr)
            if need_scale:
                s2 = g["scale2"].to_numpy(float)
                scale_ok = np.isfinite(s2) & (s2 >= scale_thr)
            else:
                scale_ok = np.ones(len(g), dtype=bool)

            used_by_fit = considered & scale_ok & chi_ok

            if len(np.unique(g.loc[used_by_fit, "N"].to_numpy(float))) < min_unique_N:
                continue

            y = -np.log(g.loc[used_by_fit, kcol].to_numpy(float))
            Ns = g.loc[used_by_fit, "N"].to_numpy(float)
            slope, inter, slope_se, r2, n = _linear_fit(Ns, y)
            if verbose:
                print(f"[RESC] subset={subset_id} anchor={ref_type}/{ref_stat} beta*={bbin:.6g} k={kcol} n={n} slope={slope:.6g} r2={r2:.3f}")
            fits.append({
                "beta_kind": "rescaled", "subset_id": subset_id, "family_id": fam,
                "anchor": ref_type, "ref_stat": ref_stat, "beta": None, "beta_rescaled_bin": float(bbin),
                "kcol": kcol, "slope": float(slope), "slope_stderr": float(slope_se) if np.isfinite(slope_se) else np.nan,
                "intercept": float(inter), "r2": float(r2) if np.isfinite(r2) else np.nan,
                "n_points": int(n), "n_unique_N": int(len(np.unique(Ns))), "N_min": float(np.min(Ns)), "N_max": float(np.max(Ns)),
            })

            excl_scale = considered & (~scale_ok)
            excl_chi   = considered & scale_ok & (~chi_ok)
            rows = pd.DataFrame({
                "beta_kind": "rescaled", "subset_id": subset_id, "family_id": fam,
                "anchor": ref_type, "ref_stat": ref_stat, "kcol": kcol,
                "beta": np.nan, "beta_rescaled_bin": float(bbin),
                "TIcurve_id": g["TIcurve_id"].astype(str).values, "run_uid": g["run_uid"].astype(str).values,
                "N": g["N"].astype(float).values, "beta_point": g["beta"].astype(float).values,
                "beta_rescaled": g["beta_rescaled"].astype(float).values, "k_value": g[kcol].astype(float).values,
                "chi_value": g[chi_col].astype(float).values,
                "scale2": (g["scale2"].astype(float).values if need_scale else np.full(len(g), np.nan)),
                "is_used_subset": is_used_subset, "used_by_fit": used_by_fit,
                "excluded_chi": excl_chi, "excluded_scale": (excl_scale if need_scale else np.zeros(len(g), dtype=bool)),
                "chi2_threshold": float(chi2_thr), "scale_threshold": (float(scale_thr) if need_scale else np.nan),
                "rescaled_step": float(step),
            })
            members.append(rows)

    return fits, (pd.concat(members, ignore_index=True) if members else pd.DataFrame())

# ------------------------------
# CLI
# ------------------------------

def _parser() -> argparse.ArgumentParser:
    ep = (
"Examples:\n"
"  python3 fit_rates_vs_N.py --model ER --subset-label 'N>30' -v\n"
"  python3 fit_rates_vs_N.py --model ER --subset-label 'N>30' --anchors M,G --ref-stats mean -v\n"
"  python3 fit_rates_vs_N.py --model ER --subset-id <subset_id> --kcols kFromChi -v\n"
)
    ap = argparse.ArgumentParser(
        description="Fit -ln(k) vs N (RAW + RESCALED) + membership dei fit (punti usati/esclusi).",
        epilog=ep,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", required=True, help="Modello (es. ER, RRG, realGraphs/ZKC, ...)")
    ap.add_argument("--data-root", default=None, help="Root Data (default: ../../Data)")
    ap.add_argument("--subset-label", default="all", help="Label del subset (es. all, 'N>30')")
    ap.add_argument("--subset-id", default=None, help="Subset ID (sovrascrive label)")
    ap.add_argument("--family-id", default=None, help="Family ID (opzionale)")
    ap.add_argument("--kcols", default=None, help="Lista colonne k (comma). Default: tre standard")
    ap.add_argument("--anchors", default=None, help="Anchors per RESCALED (default: M,G). Vuota per disattivare")
    ap.add_argument("--ref-stats", default=None, help="Statistiche per anchor (default: mean; usare 'mean,median')")
    ap.add_argument("--rescaled-step", type=float, default=0.025, help="Delta per binning di beta_rescaled")
    ap.add_argument("--min-unique-N", type=int, default=5, help="Min # di N distinti nel fit (default 4)")
    ap.add_argument("--include-unused", action="store_true", help="Includi anche TIcurve con is_used=False (appariranno come unused_subset=True)")
    ap.add_argument("--chi2-threshold", type=float, default=0.45, help="Soglia su χ")
    ap.add_argument("--scale-threshold", type=float, default=0.4, help="Soglia su scale2 (solo per kFromChi_InBetween_Scaled)")
    ap.add_argument("--outname", default="ti_linear_fits.parquet", help="Nome file output parquet")
    ap.add_argument("--out-members", default="ti_linear_fit_members.parquet", help="Nome file membership output parquet")
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
    runs_path     = base.parent / "runs_results/runs_results.parquet"

    print("[in ]", points_path)
    print("[in ]", members_path)
    print("[in ]", rescaled_path)
    print("[in ]", subsets_path)
    print("[in ]", runs_path, " (per scale2)")

    points   = _load_parquet(points_path, "ti_points")
    members  = _load_parquet(members_path, "ti_subset_members")
    rescaled = _load_parquet(rescaled_path, "ti_subset_points_rescaled")
    subsets  = _load_parquet(subsets_path, "ti_family_subsets")
    runs_res = _load_parquet(runs_path, "runs_results")

    chi_map = {"kFromChi":"chi_chi","kFromChi_InBetween":"chi_chi2","kFromChi_InBetween_Scaled":"chi_chi2"}

    if ns.subset_id:
        target_subset_ids = [ns.subset_id]
    else:
        _ensure_cols(subsets, ["subset_id","subset_label"], "ti_family_subsets")
        target_subset_ids = subsets.loc[subsets["subset_label"]==ns.subset_label, "subset_id"].drop_duplicates().tolist()
        print(f"[subset] label='{ns.subset_label}' -> {len(target_subset_ids)} subset_id")

    fam_ids = [ns.family_id] if ns.family_id else None
    kcols = _kcols_list(ns.kcols)
    anchors = _anchor_list(ns.anchors)
    ref_stats = _refstats_list(ns.ref_stats)

    print(f"[conf] kcols={kcols} anchors={anchors} ref_stats={ref_stats} step={ns.rescaled_step} include_unused={ns.include_unused}")
    print(f"[thr ] chi<= {ns.chi2_threshold} ; scale2>= {ns.scale_threshold} (solo per kFromChi_InBetween_Scaled, sia RAW che RESCALED)")
    print(f"[sizes] points={len(points)} members={len(members)} rescaled={len(rescaled)} runs_results={len(runs_res)}")

    fits_all, members_all = [], []

    # RAW (ora con scale2 se kcol==Scaled)
    for kcol in kcols:
        if kcol not in points.columns:
            print(f"[skip] colonna k assente in ti_points: {kcol}")
            continue
        chi_col = chi_map.get(kcol, "chi_chi2")
        if chi_col not in points.columns:
            raise SystemExit(f"[preflight] ti_points manca la colonna richiesta per chi ({chi_col}) per kcol={kcol}")
        fits, mems = fit_and_members_raw(points, runs_res, members, target_subset_ids, fam_ids, kcol,
                                         ns.include_unused, ns.min_unique_N, chi_col, ns.chi2_threshold, ns.scale_threshold, ns.verbose)
        fits_all.extend(fits)
        if len(mems) > 0:
            members_all.append(mems)

    # RESCALED (immutato: scale2 solo se Scaled)
    if len(anchors) > 0:
        for kcol in kcols:
            if kcol not in points.columns:
                continue
            chi_col = chi_map.get(kcol, "chi_chi2")
            if chi_col not in points.columns:
                raise SystemExit(f"[preflight] ti_points manca la colonna richiesta per chi ({chi_col}) per kcol={kcol}")
            fits, mems = fit_and_members_rescaled(points, rescaled, members, runs_res, target_subset_ids, fam_ids,
                                                  kcol, anchors, ref_stats, ns.rescaled_step, ns.include_unused,
                                                  ns.min_unique_N, chi_col, ns.chi2_threshold, ns.scale_threshold, ns.verbose)
            fits_all.extend(fits)
            if len(mems) > 0:
                members_all.append(mems)

    # --- scrittura output ---
    out_fits = pd.DataFrame(fits_all)
    out_path = base / ns.outname
    if len(out_fits) == 0:
        print("[warn] Nessun fit calcolato: controlla subset/anchors/kcols o filtri troppo stringenti.")
        return
    out_fits["computed_at"] = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    out_fits["analysis_rev"] = "v11"
    out_fits["chi2_threshold"] = float(ns.chi2_threshold)
    out_fits["scale_threshold"] = float(ns.scale_threshold)
    out_fits["rescaled_step"] = float(ns.rescaled_step)
    out_fits.to_parquet(out_path, index=False)

    if members_all:
        out_members = pd.concat(members_all, ignore_index=True)
        out_members["computed_at"] = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        out_members["analysis_rev"] = "v11"
        out_members.to_parquet(base / ns.out_members, index=False)
        print(f"[done] {base / ns.out_members} (rows={len(out_members)})")
    else:
        print("[warn] Nessuna membership prodotta.")

    print(f"[done] {out_path} (rows={len(out_fits)})")

if __name__ == "__main__":
    main()
