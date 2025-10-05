#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
peek_tables.py
Quickly inspect model tables under MultiPathsMC:
- runs_params
- runs_results
- stdmcs
- ti_curves
- ti_points
- ti_membership
- ti_families                 (NEW)
- ti_family_subsets           (NEW)
- ti_subset_members           (NEW)
- ti_subset_refs              (NEW)
- ti_subset_points_rescaled   (NEW)
- ti_linear_fits   (NEW)

Usage examples:
  python3 peek_tables.py --model ER
  python3 peek_tables.py --model realGraphs/ZKC --tables ti_curves ti_points ti_subset_members
  python3 peek_tables.py --model ER --head 30
  python3 peek_tables.py --model ER --list
"""
import argparse
from pathlib import Path
from typing import Optional, List
import pandas as pd

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 200)

DEFAULT_TABLES: List[str] = ["runs_params", "runs_results", "stdmcs"]
ALL_TABLES: List[str] = DEFAULT_TABLES + [
    "ti_curves",
    "ti_points",
    # NEW familying-first artifacts
    "ti_membership",
    "ti_families",
    "ti_family_subsets",
    "ti_subset_members",
    "ti_subset_refs",
    "ti_subset_points_rescaled",
    "ti_linear_fits",
]

def _exists_graphs_root(path: Path) -> bool:
    if not path or not path.exists():
        return False
    try:
        subs = [p.name for p in path.iterdir() if p.is_dir()]
    except Exception:
        return False
    return any(n in subs for n in ("ER","RRG","realGraphs"))

def discover_graphs_root(cli_value: Optional[Path]) -> Path:
    if cli_value and _exists_graphs_root(cli_value):
        return cli_value.resolve()
    # fallback: risali la gerarchia a caccia di Data/Graphs
    cands = []
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()
    for base in {script_dir, cwd}:
        cur = base
        for _ in range(8):
            cands.append(cur / "Data" / "Graphs")
            if cur.name == "TransitionPathsMC":
                cands.append(cur / "Data" / "Graphs")
            cur = cur.parent
    seen, ordered = set(), []
    for c in cands:
        rc = c.resolve() if c.exists() else c
        if rc not in seen:
            seen.add(rc); ordered.append(c)
    for cand in ordered:
        if _exists_graphs_root(cand):
            return cand.resolve()
    raise FileNotFoundError("Could not auto-discover Data/Graphs. Pass --graphs-root.")

def default_outdir_for(graphs_root: Path) -> Path:
    return graphs_root.parent / "MultiPathsMC"

def _path_for_table(base_v1: Path, table: str) -> Path:
    if table == "runs_params":
        return base_v1 / "runs_params" / "runs_params.parquet"
    if table == "runs_results":
        return base_v1 / "runs_results" / "runs_results.parquet"
    if table == "stdmcs":
        return base_v1 / "stdmcs" / "stdmcs.parquet"
    if table == "ti_curves":
        return base_v1 / "ti" / "ti_curves.parquet"
    if table == "ti_points":
        return base_v1 / "ti" / "ti_points.parquet"
    # NEW tables
    if table == "ti_membership":
        return base_v1 / "ti" / "ti_membership.parquet"
    if table == "ti_families":
        return base_v1 / "ti" / "ti_families.parquet"
    if table == "ti_family_subsets":
        return base_v1 / "ti" / "ti_family_subsets.parquet"
    if table == "ti_subset_members":
        return base_v1 / "ti" / "ti_subset_members.parquet"
    if table == "ti_subset_refs":
        return base_v1 / "ti" / "ti_subset_refs.parquet"
    if table == "ti_subset_points_rescaled":
        return base_v1 / "ti" / "ti_subset_points_rescaled.parquet"
    if table == "ti_linear_fits":
        return base_v1 / "ti" / "ti_linear_fits.parquet"
    raise ValueError(f"Unknown table: {table}")

def _load_parquet(p: Path, name: str, head_n: int, verbose: bool):
    if not p.exists():
        print(f"[{name}] NOT FOUND: {p}")
        return None
    df = pd.read_parquet(p)
    print(f"\n[{name}] path={p}")
    print(f"[{name}] shape={df.shape}")
    print(f"[{name}] columns= {list(df.columns)}")
    if verbose:
        dtypes = df.dtypes.astype(str).to_dict()
        print(f"[{name}] dtypes (first 20): {dict(list(dtypes.items())[:20])}")
    head = df.head(head_n)
    print(f"\n[{name}] head({head_n}):\n{head.to_string(index=False)}")
    return df

def parse_args():
    p = argparse.ArgumentParser(description="Inspect MultiPathsMC tables for a model.")
    p.add_argument("--graphs-root", type=Path, default=None, help="Root Data/Graphs (auto-discovery if omitted)." )
    p.add_argument("--outdir", type=Path, default=None, help="Root MultiPathsMC (defaults next to Graphs)." )
    p.add_argument("--model", type=str, required=True, help="Model name, e.g. ER, RRG, realGraphs/ZKC." )
    p.add_argument("--tables", nargs="*", choices=ALL_TABLES, help="Which tables to print. Default: runs_params runs_results stdmcs." )
    p.add_argument("--head", type=int, default=100, help="Number of rows to print from the head (default: 100)." )
    p.add_argument("--list", action="store_true", help="List available tables on disk and exit." )
    p.add_argument("--verbose","-v", action="store_true")
    return p

def main():
    ns = parse_args().parse_args()
    graphs_root = discover_graphs_root(ns.graphs_root)
    outdir = default_outdir_for(graphs_root) if ns.outdir is None else ns.outdir.resolve()
    base_v1 = outdir / ns.model / "v1"
    print(f"[roots] graphs_root={graphs_root}")
    print(f"[roots] outdir     ={outdir}")
    print(f"[model] base       ={base_v1}")

    mapping = {t: _path_for_table(base_v1, t) for t in ALL_TABLES}
    if ns.list:
        print("\n[available tables on disk]")
        for t, p in mapping.items():
            print(f"- {t:22s}: {'FOUND' if p.exists() else 'missing'} -> {p}")
        return

    to_show = ns.tables if ns.tables else DEFAULT_TABLES
    for t in to_show:
        _load_parquet(mapping[t], t, ns.head, ns.verbose)

if __name__ == "__main__":
    main()
