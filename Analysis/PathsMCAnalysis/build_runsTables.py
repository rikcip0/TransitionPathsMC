#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Drop-in builder for PathsMC analysis tables.

- Legge JSON annidati come l'analisi originale (configuration.parameters, results.*).
- Scansiona: <graphs_root>/<model>/DataForPathsMC/{PathsMCs,stdMCs}
- Per default **richiede** un blocco "results" non vuoto; override con --allow-missing-results.
- Usa run_uid (SHA1 del path della run relativo a graphs_root) come chiave **unica**.
- Scrive Parquet in: <outdir>/<model>/v1/{runs_params,runs_results,stdmcs}/*.parquet
- Supporta: --model (subpath o 'all'), --include filters, --verbose, --dry-run
- Auto-scopre graphs_root/outdir se non passati.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from hashlib import sha1
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# -------------------------- Auto-discovery --------------------------

def _exists_graphs_root(path: Path) -> bool:
    if not path or not path.exists():
        return False
    try:
        sub = [p.name for p in path.iterdir() if p.is_dir()]
    except Exception:
        return False
    return any(name in sub for name in ("ER", "RRG", "realGraphs"))

def discover_graphs_root(cli_value: Optional[Path]) -> Path:
    if cli_value and _exists_graphs_root(cli_value):
        return cli_value.resolve()
    # walk up from script dir and cwd
    candidates: List[Path] = []
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()
    for base in {script_dir, cwd}:
        cur = base
        for _ in range(8):
            candidates.append(cur / "Data" / "Graphs")
            if cur.name == "TransitionPathsMC":
                candidates.append(cur / "Data" / "Graphs")
            cur = cur.parent
    seen = set()
    ordered = []
    for c in candidates:
        rc = c.resolve() if c.exists() else c
        if rc not in seen:
            seen.add(rc)
            ordered.append(c)
    for cand in ordered:
        if _exists_graphs_root(cand):
            return cand.resolve()
    raise FileNotFoundError("Could not auto-discover Data/Graphs. Pass --graphs-root.")

def default_outdir_for(graphs_root: Path) -> Path:
    return graphs_root.parent / "MultiPathsMC"


# -------------------------- Utils --------------------------

def eprint(*args, **kwargs):
    print(*args, **kwargs)

def make_run_uid(run_dir: Path, graphs_root: Path) -> str:
    try:
        rel = str(run_dir.resolve().relative_to(graphs_root.resolve()))
    except Exception:
        rel = str(run_dir.resolve())
    return sha1(rel.encode("utf-8")).hexdigest()[:16]

def load_run_json(run_dir: Path) -> Optional[Dict]:
    j = run_dir / "Results" / "runData.json"
    if not j.exists():
        return None
    try:
        return json.loads(j.read_text(encoding="utf-8"))
    except Exception as exc:
        eprint(f"[WARN] Failed to read {j}: {exc}")
        return None

def _get_in(d: Dict, path: Iterable[str]):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur

def _pick_any_in(root: Dict, *paths: Tuple[str, ...], default=None):
    for path in paths:
        val = _get_in(root, path)
        if val is not None:
            return val
    return default

def _safe_float(x):
    if x is None:
        return np.nan
    if isinstance(x, str):
        xl = x.strip().lower()
        if xl in ("nan", "none", ""):
            return np.nan
        if xl in ("inf","+inf"):
            return np.inf
        if xl == "-inf":
            return -np.inf
    try:
        return float(x)
    except Exception:
        return np.nan


# -------------------------- Scanning --------------------------

def scan_model_root(model_root: Path, includes: List[str], verbose: bool=False) -> Tuple[List[Path], List[Path]]:
    """
    FAST: trova le directory 'DataForPathsMC' solo in:
      - model_root/DataForPathsMC
      - model_root/*/DataForPathsMC
    Se non trova nulla (caso raro), fallback a una scansione ricorsiva PRUNED:
      - cammina l’albero ma, appena vede 'DataForPathsMC', la aggiunge e NON scende dentro.
    Poi, per ogni DataForPathsMC trovata, legge SOLTANTO le sottodir immediate
    in {PathsMCs,stdMCs} e verifica la presenza di Results/runData.json.
    """
    dfpmc_roots: List[Path] = []

    # 1) posizioni "note" (O(1))
    direct = model_root / "DataForPathsMC"
    if direct.is_dir():
        dfpmc_roots.append(direct)

    # figli immediati (O(#subdirs))
    try:
        for child in model_root.iterdir():
            if not child.is_dir():
                continue
            d = child / "DataForPathsMC"
            if d.is_dir():
                dfpmc_roots.append(d)
    except FileNotFoundError:
        pass

    # 2) fallback ricorsivo PRUNED (solo se non abbiamo trovato nulla)
    if not dfpmc_roots:
        for root, dirs, files in os.walk(model_root, topdown=True):
            if "DataForPathsMC" in dirs:
                dfpmc_roots.append(Path(root) / "DataForPathsMC")
                # PRUNE: non scendere in DataForPathsMC, evita di esplorare run
                dirs.remove("DataForPathsMC")

    # de-dup (preserva ordine)
    seen = set(); uniq: List[Path] = []
    for d in dfpmc_roots:
        try:
            key = d.resolve()
        except Exception:
            key = d
        if key not in seen:
            seen.add(key); uniq.append(d)

    def list_runs(one_root: Path, sub: str) -> List[Path]:
        base = one_root / sub
        if not base.is_dir():
            return []
        out: List[Path] = []
        try:
            with os.scandir(base) as it:
                for de in it:
                    if not de.is_dir():
                        continue
                    run_dir = Path(de.path)
                    print(run_dir)
                    j = run_dir / "Results" / "runData.json"
                    if not j.exists():
                        continue
                    s = de.path
                    if includes and not any(tok in s for tok in includes):
                        continue
                    out.append(run_dir)
        except FileNotFoundError:
            pass
        return sorted(out)

    paths_runs: List[Path] = []
    std_runs: List[Path] = []
    for dfpmc in uniq:
        paths_runs.extend(list_runs(dfpmc, "PathsMCs"))
        std_runs.extend(list_runs(dfpmc, "stdMCs"))

    if verbose:
        eprint(f"[scan] {model_root}: DataForPathsMC dirs={len(uniq)} | PathsMCs={len(paths_runs)} stdMCs={len(std_runs)}")
        for ex in (paths_runs[:3] + std_runs[:3]):
            eprint("  -", ex)

    return sorted(paths_runs), sorted(std_runs)


# -------------------------- Extraction --------------------------

def extract_runs_params_row(data: Dict, run_dir: Path, model_type: str) -> Dict:
    cfg = data.get("configuration", {}) or {}
    MCpars = cfg.get("mcParameters", {}) or {}
    par = cfg.get("parameters", {}) or {}
    ref = cfg.get("referenceConfigurationsInfo", {}) or {}
    mcp = cfg.get("mcParameters", {}) or {}

    row = dict(
        ID = cfg.get("ID"),
        model_type = model_type,
        runPath = str(run_dir),

        graphID = _pick_any_in({"cfg":cfg,"par":par,"mcp":mcp},
                               ("par","graphID"), ("cfg","graphID"), ("mcp","graphID")),
        simulationType = _pick_any_in({"cfg":cfg}, ("cfg","simulationTypeId"), ("cfg","simulationType")),
        N    = _pick_any_in({"cfg":cfg,"par":par,"mcp":mcp}, ("par","N"), ("cfg","N"), ("mcp","N")),
        T    = _pick_any_in({"cfg":cfg,"par":par,"mcp":mcp}, ("par","T"), ("cfg","T"), ("mcp","T")),
        beta = _pick_any_in({"cfg":cfg,"par":par,"mcp":mcp}, ("par","beta"), ("cfg","beta"), ("mcp","beta")),

        h_ext = _pick_any_in({"cfg":cfg,"par":par,"mcp":mcp},
                             ("par","hext"), ("par","Hext"), ("cfg","Hext"), ("cfg","h_ext"),
                             ("mcp","Hext"), ("mcp","h_ext")),
        h_in  = _pick_any_in({"cfg":cfg,"par":par,"mcp":mcp},
                             ("par","h_in"), ("cfg","Hinit"), ("par","Hinit"), ("mcp","Hinit")),
        h_out = _pick_any_in({"cfg":cfg,"par":par,"mcp":mcp},
                             ("par","h_out"), ("cfg","Hout"), ("par","Hout"), ("mcp","Hout")),

        Qstar = _pick_any_in({"cfg":cfg,"par":par,"mcp":mcp}, ("par","Qstar"), ("cfg","Qstar"), ("mcp","Qstar")),
        C     = _pick_any_in({"cfg":cfg,"par":par,"mcp":mcp}, ("par","C"), ("cfg","C"), ("mcp","C")),
        fPosJ = _pick_any_in({"cfg":cfg,"par":par,"mcp":mcp}, ("par","fPosJ"), ("cfg","fPosJ"), ("mcp","fPosJ")),

        fieldType        = _pick_any_in({"cfg":cfg,"par":par}, ("par","fieldType"), ("cfg","fieldType")),
        fieldMean        = _pick_any_in({"cfg":cfg,"par":par}, ("par","fieldMean"), ("cfg","fieldMean")),
        fieldSigma       = _pick_any_in({"cfg":cfg,"par":par}, ("par","fieldSigma"), ("cfg","fieldSigma")),
        fieldRealization = _pick_any_in({"cfg":cfg,"par":par}, ("par","fieldRealization"), ("cfg","fieldRealization")),

        betaOfExtraction = _pick_any_in({"cfg":cfg,"par":par,"ref":ref},
                                        ("ref","betaOfExtraction"), ("par","betaOfExtraction"), ("cfg","betaOfExtraction")),
        firstConfigurationIndex  = _pick_any_in({"cfg":cfg,"ref":ref},
                                                ("ref","firstConfigurationIndex"), ("cfg","firstConfigurationIndex")),
        secondConfigurationIndex = _pick_any_in({"cfg":cfg,"ref":ref},
                                                ("ref","secondConfigurationIndex"), ("cfg","secondConfigurationIndex")),
        refConfMutualQ = _pick_any_in({"cfg":cfg,"ref":ref},
                                      ("ref","mutualOverlap"), ("cfg","refConfMutualQ")),
        refConfInitID  = _pick_any_in({"cfg":cfg,"ref":ref},
                                      ("ref","ID"), ("cfg","refConfInitID")),
        trajsExtremesInitID = _pick_any_in({"cfg":cfg}, ("cfg","trajs_Initialization","ID"), ("cfg","trajsExtremesInitID")),
        trajsJumpsInitID    = _pick_any_in({"cfg":cfg}, ("cfg","trajs_jumpsInitialization","ID"), ("cfg","trajsJumpsInitID")),
        MCprint      = _pick_any_in({"cfg":cfg,"mcp":mcp}, ("mcp","MCprint"), ("cfg","MCprint")),
        lastMeasureMC= _pick_any_in(data, ("lastMeasureMC",)),
    )
    return row

def extract_runs_results_row(data: Dict, model_type: str, analysis_rev: str) -> Dict:
    res = data.get("results", {}) or {}
    cfg = data.get("configuration", {}) or {}
    MCpars = cfg.get("mcParameters", {}) or {}
    par = cfg.get("parameters", {}) or {}

    chi1 = res.get("chiLinearFit") or {}
    chi2 = res.get("chiLinearFit_InBetween") or {}
    chi_tau  = _safe_float(chi1.get("tau"))
    chi_m    = _safe_float(chi1.get("m"))
    chi_c    = _safe_float(chi1.get("c"))
    chi_chi  = _safe_float(chi1.get("Chi"))
    chi_tau2 = _safe_float(chi2.get("tau"))
    chi_m2   = _safe_float(chi2.get("m"))
    chi_c2   = _safe_float(chi2.get("c"))
    chi_chi2 = _safe_float(chi2.get("Chi"))

    rt = res.get("realTime") or {}
    realTime    = _safe_float(rt.get("mean"))
    realTimeErr = _safe_float(rt.get("sigma"))
    meanBarrier   = _safe_float(res.get("meanBarrier"))
    stdDevBarrier = _safe_float(res.get("stdDevBarrier"))

    th   = res.get("thermalization") or {}
    avE  = th.get("avEnergy") or {}
    nJ   = th.get("nJumps") or {}
    dNJ  = th.get("deltaNJumps") or {}
    qD   = th.get("qDist") or {}
    muAvEnergy     = _safe_float(avE.get("mu"))
    avEnergy       = _safe_float(avE.get("mean"))
    avEnergyStdErr = _safe_float(avE.get("stdErr"))
    nJumps         = _safe_float(nJ.get("mean"))
    nJumpsStdErr   = _safe_float(nJ.get("stdErr"))
    deltaNJumps    = _safe_float(dNJ.get("mean"))
    deltaNJumpsStdErr = _safe_float(dNJ.get("stdErr"))
    qDist          = _safe_float(qD.get("mean"))
    qDistStdErr    = _safe_float(qD.get("stdErr"))

    TI     = res.get("TI") or {}
    TIbeta = _safe_float(TI.get("beta"))
    TIhout = _safe_float(TI.get("hout"))
    TIQstar= _safe_float(TI.get("Qstar"))

    Tval = _safe_float(par.get("T") if par.get("T") is not None else cfg.get("T"))
    effectiveFlipRate      = (nJumps / Tval) if (np.isfinite(nJumps) and np.isfinite(Tval) and Tval != 0) else np.nan
    effectiveFlipRateError = (nJumpsStdErr / Tval) if (np.isfinite(nJumpsStdErr) and np.isfinite(Tval) and Tval != 0) else np.nan

    row = dict(
        ID = cfg.get("ID"),
        model_type = model_type,
        computed_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z"),
        analysis_rev = analysis_rev,
        chi_tau=chi_tau, chi_m=chi_m, chi_c=chi_c, chi_chi=chi_chi,
        chi_tau2=chi_tau2, chi_m2=chi_m2, chi_c2=chi_c2, chi_chi2=chi_chi2,
        realTime=realTime, realTimeErr=realTimeErr,
        meanBarrier=meanBarrier, stdDevBarrier=stdDevBarrier,
        muAvEnergy=muAvEnergy, avEnergy=avEnergy, avEnergyStdErr=avEnergyStdErr,
        nJumps=nJumps, nJumpsStdErr=nJumpsStdErr,
        deltaNJumps=deltaNJumps, deltaNJumpsStdErr=deltaNJumpsStdErr,
        qDist=qDist, qDistStdErr=qDistStdErr,
        effectiveFlipRate=effectiveFlipRate, effectiveFlipRateError=effectiveFlipRateError,
        TIbeta=TIbeta, TIhout=TIhout, TIQstar=TIQstar,
    )
    return row

def extract_stdmcs_row(data: Dict, run_dir: Path) -> Optional[Dict]:
    cfg = data.get("configuration", {}) or {}
    MCpars = cfg.get("mcParameters", {}) or {}
    par = cfg.get("parameters", {}) or {}
    ref = cfg.get("referenceConfigurationsInfo", {}) or {}
    TI  = (data.get("results", {}) or {}).get("TI", {}) or {}

    # Usa il nome cartella come fonte primaria; fallback a simulationTypeId==15
    is_std = "stdMCs" in str(run_dir)
    if not is_std:
        stid = cfg.get("simulationTypeId")
        is_std = (str(stid) == "15")
    if not is_std:
        return None

    row = dict(
        ID = cfg.get("ID"),
        runPath = str(run_dir),
        stMC_N      = par.get("N"),
        stMC_beta   = par.get("beta"),
        stMC_MC   = MCpars.get("MC"),
        stMC_Hext   = par.get("hext") if par.get("hext") is not None else par.get("Hext"),
        stMC_Hout   = par.get("h_out") if par.get("h_out") is not None else par.get("Hout"),
        stMC_Qstar  = par.get("Qstar"),
        stMC_graphID= par.get("graphID"),
        stMC_betaOfExtraction   = ref.get("betaOfExtraction"),
        stMC_configurationIndex = ref.get("configurationIndex"),
        stMC_fieldType          = par.get("fieldType"),
        stMC_fieldMean          = par.get("fieldMean"),
        stMC_fieldSigma         = par.get("fieldSigma"),
        stMC_fieldRealization   = par.get("fieldRealization"),
        stMC_TIbeta = TI.get("beta"),
    )
    return row


# -------------------------- Dtypes & Upsert --------------------------

RUNS_PARAMS_DTYPES = {
    "run_uid": "string",
    "ID": "string",
    "model_type": "string",
    "runPath": "string",
    "graphID": "string",
    "simulationType": "string",
    "N": "Int64",
    "T": "float64",
    "beta": "float64",
    "h_ext": "float64",
    "h_in": "float64",
    "h_out": "float64",
    "Qstar": "float64",
    "C": "float64",
    "fPosJ": "float64",
    "fieldType": "string",
    "fieldMean": "float64",
    "fieldSigma": "float64",
    "fieldRealization": "string",
    "betaOfExtraction": "float64",
    "firstConfigurationIndex": "Int64",
    "secondConfigurationIndex": "Int64",
    "refConfMutualQ": "float64",
    "refConfInitID": "string",
    "trajsExtremesInitID": "string",
    "trajsJumpsInitID": "string",
    "MCprint": "Int64",
    "lastMeasureMC": "Int64",
}

RUNS_RESULTS_DTYPES = {
    "run_uid": "string",
    "ID": "string",
    "model_type": "string",
    "computed_at": "string",
    "analysis_rev": "string",
    "chi_tau": "float64",
    "chi_m": "float64",
    "chi_c": "float64",
    "chi_chi": "float64",
    "chi_tau2": "float64",
    "chi_m2": "float64",
    "chi_c2": "float64",
    "chi_chi2": "float64",
    "realTime": "float64",
    "realTimeErr": "float64",
    "meanBarrier": "float64",
    "stdDevBarrier": "float64",
    "muAvEnergy": "float64",
    "avEnergy": "float64",
    "avEnergyStdErr": "float64",
    "nJumps": "float64",
    "nJumpsStdErr": "float64",
    "deltaNJumps": "float64",
    "deltaNJumpsStdErr": "float64",
    "qDist": "float64",
    "qDistStdErr": "float64",
    "effectiveFlipRate": "float64",
    "effectiveFlipRateError": "float64",
    "TIbeta": "float64",
    "TIhout": "float64",
    "TIQstar": "float64",
}

STDMCS_DTYPES = {
    "run_uid": "string",
    "ID": "string",
    "runPath": "string",
    "stMC_N": "Int64",
    "stMC_beta": "float64",
    "stMC_Hext": "float64",
    "stMC_Hout": "float64",
    "stMC_Qstar": "float64",
    "stMC_graphID": "string",
    "stMC_betaOfExtraction": "float64",
    "stMC_configurationIndex": "Int64",
    "stMC_fieldType": "string",
    "stMC_fieldMean": "float64",
    "stMC_fieldSigma": "float64",
    "stMC_fieldRealization": "string",
    "stMC_MC": "Int64",
    "stMC_TIbeta": "float64",
}

def enforce_dtypes(df: pd.DataFrame, dtypes_map: Dict[str, str]) -> pd.DataFrame:
    if df.empty:
        cols = {k: pd.Series(dtype=v) for k, v in dtypes_map.items()}
        return pd.DataFrame(cols)[list(dtypes_map.keys())]
    for col in dtypes_map.keys():
        if col not in df.columns:
            df[col] = pd.Series(dtype=dtypes_map[col])
    for col, dt in dtypes_map.items():
        try:
            if dt == "Int64":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif dt == "float64":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
            elif dt == "string":
                df[col] = df[col].astype("string")
            else:
                df[col] = df[col].astype(dt)
        except Exception:
            pass
    return df[list(dtypes_map.keys())]

def upsert_parquet(df_new: pd.DataFrame, out_path: Path, key_col: str = "run_uid") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Overwrite entirely: no merge with existing parquet
    if key_col in df_new.columns:
        df_new = df_new.drop_duplicates(subset=[key_col], keep="last")
    df_new.to_parquet(out_path, index=False)


# -------------------------- Build per model --------------------------

def scan_and_build(model: str, includes: List[str], outdir: Path, graphs_root: Path,
                   analysis_rev: str, verbose: bool=False, dry_run: bool=False,
                   allow_missing_results: bool=False) -> Dict[str, int]:
    model_root = graphs_root / model
    paths_runs, std_runs = scan_model_root(model_root, includes, verbose=verbose)

    eprint(f"[build] model={model} | PathsMCs={len(paths_runs)} stdMCs={len(std_runs)}")
    if dry_run:
        sample = (paths_runs[:3] if paths_runs else []) + (std_runs[:3] if std_runs else [])
        for p in sample:
            eprint("  -", p)
        return dict(n_paths=len(paths_runs), n_std=len(std_runs), n_params=0, n_results=0, n_std_rows=0, skipped_no_results=0)

    model_dir = outdir / model / "v1"
    params_path  = model_dir / "runs_params"  / "runs_params.parquet"
    results_path = model_dir / "runs_results" / "runs_results.parquet"
    std_path     = model_dir / "stdmcs"       / "stdmcs.parquet"

    params_rows: List[Dict] = []
    results_rows: List[Dict] = []
    std_rows: List[Dict] = []
    seen_uids: set = set()
    skipped_no_results = 0

    # PathsMCs
    for i, run in enumerate(paths_runs, 1):
        data = load_run_json(run)
        if not data:
            continue
        # Require non-empty results unless explicitly allowed
        if not allow_missing_results:
            res_block = data.get("results", None)
            if not isinstance(res_block, dict) or len(res_block) == 0:
                skipped_no_results += 1
                if verbose:
                    eprint(f"  [skip] no results: {run}")
                continue

        uid = make_run_uid(run, graphs_root)
        if uid in seen_uids:
            continue
        seen_uids.add(uid)

        row_p = extract_runs_params_row(data, run, model_type=model); row_p["run_uid"] = uid
        row_r = extract_runs_results_row(data, model_type=model, analysis_rev=analysis_rev); row_r["run_uid"] = uid
        params_rows.append(row_p); results_rows.append(row_r)

        if verbose and (i % 200 == 0 or i == len(paths_runs)):
            eprint(f"  [paths] processed {i}/{len(paths_runs)}")

    # stdMCs
    for i, run in enumerate(std_runs, 1):
        data = load_run_json(run)
        if not data:
            continue
        if not allow_missing_results:
            res_block = data.get("results", None)
            if not isinstance(res_block, dict) or len(res_block) == 0:
                skipped_no_results += 1
                if verbose:
                    eprint(f"  [skip-std] no results: {run}")
                continue

        uid = make_run_uid(run, graphs_root)
        row_s = extract_stdmcs_row(data, run)
        if row_s is None:
            continue
        row_s["run_uid"] = uid
        std_rows.append(row_s)

        if verbose and (i % 200 == 0 or i == len(std_runs)):
            eprint(f"  [std] processed {i}/{len(std_runs)}")

    df_params  = enforce_dtypes(pd.DataFrame(params_rows), RUNS_PARAMS_DTYPES)
    df_results = enforce_dtypes(pd.DataFrame(results_rows), RUNS_RESULTS_DTYPES)
    df_std     = enforce_dtypes(pd.DataFrame(std_rows), STDMCS_DTYPES)

    upsert_parquet(df_params,  params_path,  key_col="run_uid")
    upsert_parquet(df_results, results_path, key_col="run_uid")
    upsert_parquet(df_std,     std_path,     key_col="run_uid")

    return dict(
        n_paths=len(paths_runs),
        n_std=len(std_runs),
        n_params=len(df_params),
        n_results=len(df_results),
        n_std_rows=len(df_std),
        skipped_no_results=skipped_no_results,
    )


# -------------------------- CLI & Main --------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build Parquet tables for PathsMC runs (params/results/stdMCs).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--graphs-root", type=Path, default=None,
                   help="Root of Data/Graphs. Auto-discovered if omitted.")
    p.add_argument("--outdir", type=Path, default=None,
                   help="Output root (Data/MultiPathsMC). Defaults next to graphs-root.")
    p.add_argument("--model", type=str, required=True,
                   help="Model subpath under graphs-root (e.g. ER, RRG, realGraphs, realGraphs/ZKC) or 'all'.")
    p.add_argument("--include", nargs="*", default=[],
                   help="Filter tokens that the run path must contain (any of).")
    p.add_argument("--analysis-rev", type=str, default="unversioned",
                   help="Optional analysis revision string written to runs_results.")
    p.add_argument("--allow-missing-results", action="store_true",
                   help="Include runs even if they have no results block (default: skip them).")
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose progress logs.")
    p.add_argument("--dry-run", action="store_true", help="Scan only; do not write parquet.")
    return p.parse_args()


def main():
    ns = parse_args()
    graphs_root = discover_graphs_root(ns.graphs_root)
    outdir = (default_outdir_for(graphs_root) if ns.outdir is None else ns.outdir.resolve())

    print(f"[roots] graphs_root={graphs_root}")
    print(f"[roots] outdir     ={outdir}")

    # Resolve models
    if ns.model == "all":
        candidates = ["ER", "RRG", "realGraphs"]
        models = [m for m in candidates if (graphs_root / m).exists()]
        if not models:
            print("[WARN] No default models found under graphs_root.")
            return
    else:
        models = [ns.model]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for model in models:
        stats = scan_and_build(
            model=model, includes=ns.include, outdir=outdir, graphs_root=graphs_root,
            analysis_rev=ns.analysis_rev, verbose=ns.verbose, dry_run=ns.dry_run,
            allow_missing_results=ns.allow_missing_results
        )

        # Manifest
        manifest_root = outdir / model / "v1" / "manifests"
        manifest_root.mkdir(parents=True, exist_ok=True)
        mf = manifest_root / f"build_runs_tables_{stamp}.md"
        block = f"""# build_runs_tables manifest

- graphs_root: {graphs_root}
- outdir:      {outdir}
- model arg:   {model}
- includes:    {ns.include}
- analysis_rev:{ns.analysis_rev}

## Per-model stats
- {model}: paths={stats['n_paths']} std={stats['n_std']} -> params={stats['n_params']} results={stats['n_results']} std_rows={stats['n_std_rows']} skipped_no_results={stats['skipped_no_results']}
"""
        if mf.exists():
            with mf.open("a", encoding="utf-8") as f:
                f.write("\n" + block)
        else:
            mf.write_text(block, encoding="utf-8")
        print(block.strip())


if __name__ == "__main__":
    main()
