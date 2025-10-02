#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostics for MultiPathsMC Parquet tables produced by build_runsTables.py.

Features
- Auto-discovers Data/Graphs and default Data/MultiPathsMC (or pass paths explicitly).
- Loads {runs_params, runs_results, stdmcs} for a given --model (subpath like realGraphs/ZKC).
- Validates:
  * Unique run_uid in each table
  * Equality of run_uid sets between params and results
  * Coverage (non-null fraction) per column
  * Optional model-aware "core columns" coverage (ER/RRG/realGraphs/ZKC)
  * Optional recomputation of run_uid from runPath and comparison
- Exports optional CSVs: coverage tables, duplicates, set diffs.
- Shows sample joined rows (--show-sample N).
- Optional JSON spot-check (--json-spot-check K): shows which nested keys exist for K runs.

Usage
  python diagnose_runsTables.py --model realGraphs/ZKC -v --show-sample 5 --export-prefix report
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from hashlib import sha1


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


# -------------------------- Helpers --------------------------

def make_run_uid_from_path(run_path: Path, graphs_root: Path) -> str:
    try:
        rel = str(Path(run_path).resolve().relative_to(graphs_root.resolve()))
    except Exception:
        rel = str(Path(run_path).resolve())
    return sha1(rel.encode("utf-8")).hexdigest()[:16]

def load_json(run_dir: Path) -> Optional[Dict]:
    j = run_dir / "Results" / "runData.json"
    if not j.exists():
        return None
    try:
        return json.loads(j.read_text(encoding="utf-8"))
    except Exception:
        return None

def coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame({"column": [], "coverage": []})
    cov = (1 - df.isna().mean()).sort_values(ascending=False)
    return cov.reset_index().rename(columns={"index": "column", 0: "coverage"})

def model_core_columns(model: str) -> Dict[str, List[str]]:
    # Minimal expectations per family; tune as needed
    if model.startswith("ER"):
        return {"params": ["N","beta","Qstar"], "results": ["meanBarrier","stdDevBarrier"]}
    if model.startswith("RRG"):
        return {"params": ["N","beta","Qstar","C","fPosJ"], "results": ["meanBarrier","stdDevBarrier"]}
    if model.startswith("realGraphs/ZKC") or model.startswith("realGraphs"):
        return {"params": ["N","T","beta","h_in","h_out","Qstar"], "results": ["meanBarrier","stdDevBarrier"]}
    # default
    return {"params": ["N","beta"], "results": ["meanBarrier"]}


# -------------------------- CLI --------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Diagnose MultiPathsMC Parquet tables for a given model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--graphs-root", type=Path, default=None,
                   help="Root of Data/Graphs. Auto-discovered if omitted.")
    p.add_argument("--outdir", type=Path, default=None,
                   help="Output root (Data/MultiPathsMC). Defaults next to graphs-root.")
    p.add_argument("--model", type=str, required=True,
                   help="Model subpath under graphs-root (e.g. ER, RRG, realGraphs, realGraphs/ZKC).")
    p.add_argument("--show-sample", type=int, default=0,
                   help="Print N sample joined rows (params ⟂ results).")
    p.add_argument("--json-spot-check", type=int, default=0,
                   help="Open K random JSONs and show which nested keys exist (debug mapping).")
    p.add_argument("--export-prefix", type=str, default="",
                   help="If set, export CSVs with coverage/duplicates/diffs using this prefix in the manifest dir.")
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose logs.")
    return p.parse_args()


# -------------------------- Main --------------------------

def main():
    ns = parse_args()
    graphs_root = discover_graphs_root(ns.graphs_root)
    outdir = (default_outdir_for(graphs_root) if ns.outdir is None else ns.outdir.resolve())

    print(f"[roots] graphs_root={graphs_root}")
    print(f"[roots] outdir     ={outdir}")

    model = ns.model
    base = outdir / model / "v1"
    p_path = base / "runs_params" / "runs_params.parquet"
    r_path = base / "runs_results" / "runs_results.parquet"
    s_path = base / "stdmcs" / "stdmcs.parquet"

    if not p_path.exists() or not r_path.exists():
        print(f"[FATAL] Missing parquet files under {base}.")
        print("        Expected:", p_path, "and", r_path)
        return

    par = pd.read_parquet(p_path)
    res = pd.read_parquet(r_path)
    std = pd.read_parquet(s_path) if s_path.exists() else pd.DataFrame()

    print(f"[load] params={par.shape} results={res.shape} stdmcs={std.shape if not std.empty else (0,0)}")

    # 1) Uniqueness checks
    def uniq(df, name):
        if df.empty:
            return True, 0
        u = df["run_uid"].is_unique
        dup = (df["run_uid"].duplicated(keep=False)).sum()
        print(f"[uniq] {name}: run_uid unique? {u} | duplicate rows={dup}")
        return u, dup

    u1, d1 = uniq(par, "runs_params")
    u2, d2 = uniq(res, "runs_results")

    # 2) Set equality
    sp = set(par["run_uid"]) if not par.empty else set()
    sr = set(res["run_uid"]) if not res.empty else set()
    only_par = sp - sr
    only_res = sr - sp
    print(f"[sets] only in params: {len(only_par)} | only in results: {len(only_res)}")

    # 3) Coverage
    cov_p = coverage_table(par)
    cov_r = coverage_table(res)
    print("\n[coverage] runs_params (top 10):\n", cov_p.head(10).to_string(index=False))
    print("\n[coverage] runs_results (top 10):\n", cov_r.head(10).to_string(index=False))

    # 4) Model-aware core columns
    cores = model_core_columns(model)
    miss_p = par[cores["params"]].isna().any(axis=1) if par.shape[0] else pd.Series([], dtype=bool)
    miss_r = res[cores["results"]].isna().any(axis=1) if res.shape[0] else pd.Series([], dtype=bool)
    print(f"\n[core] model={model} | params core={cores['params']} | missing rows: {int(miss_p.sum())}/{len(par)}")
    print(f"[core] model={model} | results core={cores['results']} | missing rows: {int(miss_r.sum())}/{len(res)}")

    # 5) Recompute run_uid from runPath and compare
    bad_uid = []
    if "runPath" in par.columns:
        for rp, uid in par[["runPath","run_uid"]].head(10000).itertuples(index=False):
            uid2 = make_run_uid_from_path(Path(str(rp)), graphs_root)
            if uid2 != uid:
                bad_uid.append((uid, uid2, rp))
        print(f"[uid] mismatch run_uid vs hashed(runPath) for first 10k rows: {len(bad_uid)}")
    else:
        print("[uid] runPath not present in runs_params; cannot recompute run_uid check.")

    # 6) Exports (optional)
    if ns.export_prefix:
        mdir = base / "manifests"
        mdir.mkdir(parents=True, exist_ok=True)

        cov_p.to_csv(mdir / f"{ns.export_prefix}_coverage_runs_params.csv", index=False)
        cov_r.to_csv(mdir / f"{ns.export_prefix}_coverage_runs_results.csv", index=False)
        if only_par:
            pd.Series(sorted(list(only_par))).to_csv(mdir / f"{ns.export_prefix}_only_in_params.csv", index=False, header=["run_uid"])
        if only_res:
            pd.Series(sorted(list(only_res))).to_csv(mdir / f"{ns.export_prefix}_only_in_results.csv", index=False, header=["run_uid"])
        if d1 or d2:
            dups_p = par[par["run_uid"].duplicated(keep=False)] if d1 else pd.DataFrame(columns=par.columns)
            dups_r = res[res["run_uid"].duplicated(keep=False)] if d2 else pd.DataFrame(columns=res.columns)
            if d1:
                dups_p.to_csv(mdir / f"{ns.export_prefix}_dups_params.csv", index=False)
            if d2:
                dups_r.to_csv(mdir / f"{ns.export_prefix}_dups_results.csv", index=False)
        if bad_uid:
            pd.DataFrame(bad_uid, columns=["run_uid","recomputed","runPath"]).to_csv(mdir / f"{ns.export_prefix}_bad_run_uid.csv", index=False)
        print(f"[export] Wrote CSVs to {mdir} with prefix '{ns.export_prefix}'")

    # 7) Sample join
    if ns.show_sample > 0:
        cols_r = ["run_uid","meanBarrier","avEnergy","TIbeta","TIhout","TIQstar"]
        cols_p = ["run_uid","runPath","N","T","beta","h_in","h_out","Qstar","graphID"]
        for c in cols_r:
            if c not in res.columns: res[c] = np.nan
        for c in cols_p:
            if c not in par.columns: par[c] = np.nan
        merged = (res[cols_r].merge(par[cols_p], on="run_uid", how="left").head(ns.show_sample))
        print("\n[sample join]\n", merged.to_string(index=False))

    # 8) JSON spot-check
    if ns.json_spot_check > 0:
        print("\n[json spot-check] Checking nested keys for a few runs...")
        sample = par.sample(min(ns.json_spot_check, len(par)), random_state=42) if len(par) else par
        for _, row in sample.iterrows():
            rp = Path(str(row["runPath"])) if "runPath" in row else None
            if not rp:
                continue
            data = load_json(rp)
            if not data:
                print(" -", rp, "-> missing JSON")
                continue
            cfg = data.get("configuration", {}) or {}
            parj = cfg.get("parameters", {}) or {}
            resj = data.get("results", {}) or {}
            print(" -", rp)
            print("   cfg.keys:", sorted(list(cfg.keys()))[:20])
            print("   parameters.keys:", sorted(list(parj.keys()))[:20])
            print("   results keys:", sorted(list(resj.keys()))[:20])

    print("\n[done] diagnostics complete.")


if __name__ == "__main__":
    main()
