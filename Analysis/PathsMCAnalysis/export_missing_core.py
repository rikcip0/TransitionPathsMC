#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_missing_core.py
Export a CSV of runs where specified columns are missing (NaN) in runs_results (or runs_params).

Usage examples:
  # Missing barrier metrics in ZKC
  python export_missing_core.py --model realGraphs/ZKC --cols meanBarrier stdDevBarrier \
    --export zkc_missing_barriers.csv -v --with-json-keys --limit 20

  # Missing core params in ER
  python export_missing_core.py --model ER --table params --cols N beta Qstar --export er_missing_params.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from hashlib import sha1


# ---------- Auto-discovery (same logic as diagnose/build) ----------

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


# ---------- Helpers ----------

def load_json(run_dir: Path) -> Optional[Dict]:
    j = run_dir / "Results" / "runData.json"
    if not j.exists():
        return None
    try:
        return json.loads(j.read_text(encoding="utf-8"))
    except Exception:
        return None


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export runs missing specified columns from Parquet tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--graphs-root", type=Path, default=None,
                   help="Root of Data/Graphs. Auto-discovered if omitted.")
    p.add_argument("--outdir", type=Path, default=None,
                   help="Output root (Data/MultiPathsMC). Defaults next to graphs-root.")
    p.add_argument("--model", type=str, required=True,
                   help="Model subpath under graphs-root (e.g. ER, RRG, realGraphs, realGraphs/ZKC).")
    p.add_argument("--table", choices=["results","params"], default="results",
                   help="Which table to inspect for missing columns.")
    p.add_argument("--cols", nargs="+", required=True,
                   help="Columns to check for NaN (missing if any is NaN).")
    p.add_argument("--export", type=str, default="",
                   help="Export CSV filename (default: <model>_missing_<cols>.csv in manifests).")
    p.add_argument("--with-json-keys", action="store_true",
                   help="Add a column with top-level results/parameters keys from JSON (slow).")
    p.add_argument("--limit", type=int, default=50,
                   help="Limit number of JSONs to open when --with-json-keys is set.")
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose logs.")
    return p.parse_args()


# ---------- Main ----------

def main():
    ns = parse_args()
    graphs_root = discover_graphs_root(ns.graphs_root)
    outdir = (default_outdir_for(graphs_root) if ns.outdir is None else ns.outdir.resolve())

    base = outdir / ns.model / "v1"
    p_path = base / "runs_params" / "runs_params.parquet"
    r_path = base / "runs_results" / "runs_results.parquet"
    if ns.table == "results":
        if not r_path.exists():
            print("[FATAL] Missing", r_path); return
        df = pd.read_parquet(r_path)
    else:
        if not p_path.exists():
            print("[FATAL] Missing", p_path); return
        df = pd.read_parquet(p_path)

    missing_mask = df[ns.cols].isna().any(axis=1) if len(df) else pd.Series([], dtype=bool)
    missing = df.loc[missing_mask].copy()
    print(f"[info] table={ns.table} rows={len(df)} | missing rows for {ns.cols}: {len(missing)}")

    # Attach runPath for context (from params if necessary)
    runpath = None
    if "runPath" in df.columns:
        runpath = df[["run_uid","runPath"]]
    else:
        # Load params to get runPath
        if p_path.exists():
            par = pd.read_parquet(p_path)[["run_uid","runPath"]]
            runpath = par
    if runpath is not None:
        missing = missing.merge(runpath, on="run_uid", how="left")

    # Add JSON keys summary (optional, limited)
    if ns.with_json_keys and len(missing) > 0:
        keys_col = []
        for i, row in enumerate(missing.head(ns.limit).itertuples(index=False)):
            try:
                rp = Path(str(getattr(row, "runPath")))
            except Exception:
                rp = None
            if not rp or not rp.exists():
                keys_col.append("")
                continue
            data = load_json(rp) or {}
            if ns.table == "results":
                keys = sorted(list((data.get("results", {}) or {}).keys()))
            else:
                cfg = (data.get("configuration", {}) or {})
                parj = (cfg.get("parameters", {}) or {})
                keys = [f"parameters:{k}" for k in sorted(list(parj.keys()))]
            keys_col.append(";".join(keys))
        # pad remaining with empty
        keys_col += [""] * (len(missing) - len(keys_col))
        missing["json_keys"] = keys_col

    # Export path
    mdir = base / "manifests"
    mdir.mkdir(parents=True, exist_ok=True)
    if ns.export:
        out_csv = Path(ns.export)
        if not out_csv.is_absolute():
            out_csv = mdir / ns.export
    else:
        cols_tag = "_".join(ns.cols)
        out_csv = mdir / f"{ns.model.replace('/','_')}_missing_{ns.table}_{cols_tag}.csv"

    # Reorder columns for readability
    ordered_cols = ["run_uid", "runPath"] + [c for c in ns.cols if c in missing.columns] + \
                   [c for c in missing.columns if c not in {"run_uid","runPath", *ns.cols}]
    missing = missing[[c for c in ordered_cols if c in missing.columns]]

    missing.to_csv(out_csv, index=False)
    print(f"[export] wrote {out_csv}")

if __name__ == "__main__":
    main()
