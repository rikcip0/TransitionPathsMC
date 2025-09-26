#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rescale_from_ti.py  — produce refs, members e punti rescalati (schema "long").

Output (per <model>): Data/MultiPathsMC/<model>/v1/ti/
  - ti_rescale_refs.parquet      (una riga per (mode, ref_group_id, anchor))
  - ti_rescale_members.parquet   (una riga per (mode, ref_group_id, TIcurve_id))
  - ti_rescaled_points.parquet   (una riga per (punto, anchor, mode))

Note importanti
- Schema dichiarativo del grouping per robustezza futura:
  * mode ∈ {fixed_family, fixed_family_N}
  * atFixed  = CSV ordinato delle dimensioni fissate
  * atValues = JSON canonico {dim:val} per quelle dimensioni
  * atLabel  = etichetta leggibile
  * ref_group_id = sha1( mode + "|" + atFixed + "|" + canonical(atValues) )[:16]

- Selezione "is_used":
  Qui conservativamente NON applichiamo una selezione "best" su (T, trajInit).
  Includiamo T e trajInit tra le dimensioni fissate; tutti i membri sono "used" (=True).
  Quando servirà, toglieremo T/trajInit da atFixed e popoleremo is_used secondo i criteri espliciti.

Requisiti (nessun fallback):
- ti_curves deve contenere le colonne
  ['TIcurve_id','graphID','fieldRealization','model_type','N','fieldType','fieldSigma',
   'Hext','Hout','Hin','normalizeQstar','T','trajInit','p','C','fPosJ',
   'betaM','betaL','betaG','betaG2','betaG3','betaG2b','betaG2c']
  (Se 'normalizeQstar' manca, viene derivata come Qstar/N, ma solo se entrambe presenti.)
- ti_points deve contenere ['TIcurve_id','run_uid','beta','T','trajInit']
"""
from __future__ import annotations

import argparse, json, hashlib, os
from datetime import datetime as _dt
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from pathlib import Path

def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    search = list(here.parents) + [Path.cwd().resolve()] + list(Path.cwd().resolve().parents)
    for d in search:
        if (d / "Data" / "MultiPathsMC").exists():
            return d
    return here.parents[2] if len(here.parents) >= 3 else here.parent

from datetime import datetime, timezone
def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _require_columns(df: pd.DataFrame, cols: List[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] colonne mancanti: {missing}")

def _paths_for_model(model: str) -> Dict[str, str]:
    root = _find_project_root()
    base_out = root / "Data" / "MultiPathsMC" / model / "v1"
    ti_dir   = base_out / "ti"
    return {
        "ti_curves": (ti_dir / "ti_curves.parquet").as_posix(),
        "ti_points": (ti_dir / "ti_points.parquet").as_posix(),
        "refs":      (ti_dir / "ti_rescale_refs.parquet").as_posix(),
        "members":   (ti_dir / "ti_rescale_members.parquet").as_posix(),
        "points":    (ti_dir / "ti_rescaled_points.parquet").as_posix(),
        "root":      root.as_posix(),
    }

def _nanmean_safe(x) -> float:
    a = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(a)
    return float(a[m].mean()) if np.any(m) else np.nan

def _nanmedian_safe(x) -> float:
    a = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(a)
    return float(np.median(a[m])) if np.any(m) else np.nan

def _fmt_number_human(x: float) -> str:
    if x is None:
        return "nan"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if not np.isfinite(xf):
        if np.isnan(xf):
            return "nan"
        return "inf" if xf > 0 else "-inf"
    if abs(xf - round(xf)) < 1e-9:
        return str(int(round(xf)))
    return f"{xf:.12g}"

def _canonical_json_from_kv(keys: List[str], row: pd.Series) -> str:
    obj = {}
    for k in keys:
        v = row[k]
        if isinstance(v, (str,)):
            obj[k] = v
        else:
            try:
                vf = float(v)
                if not np.isfinite(vf):
                    obj[k] = "nan" if np.isnan(vf) else ("inf" if vf > 0 else "-inf")
                else:
                    if abs(vf - round(vf)) < 1e-9:
                        obj[k] = int(round(vf))
                    else:
                        obj[k] = float(f"{vf:.12g}")
            except Exception:
                obj[k] = str(v)
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)

def _label_from_kv(mode: str, keys: List[str], row: pd.Series) -> str:
    parts = [f"{k}={_fmt_number_human(row[k])}" for k in keys]
    return f"{mode}: " + ", ".join(parts)

def _group_id(mode: str, atFixed: str, atValues: str) -> str:
    h = hashlib.sha1((mode + "|" + atFixed + "|" + atValues).encode("utf-8")).hexdigest()
    return h[:16]

_BETA_COLS = ["betaM","betaL","betaG","betaG2","betaG3","betaG2b","betaG2c"]
_GROUP_KEYS_FAMILY = [
    "model_type", "fieldType", "fieldSigma",
    "Hext", "Hout", "Hin",
    "normalizeQstar",
    "C", "fPosJ",
]
_GROUP_KEYS_FAMILY_N = _GROUP_KEYS_FAMILY + ["N"]

def _build_refs(curves: pd.DataFrame, keys: List[str], stat: str, verbose: bool, mode: str):
    aggfn = _nanmean_safe if stat == "mean" else _nanmedian_safe
    realkey = curves["graphID"].astype("string").fillna("nan") + "|" + curves["fieldRealization"].astype("string").fillna("nan")
    curves = curves.copy()
    curves["__realkey__"] = realkey
    dedup = curves.drop_duplicates(subset=keys + ["__realkey__"]).copy()
    n_per_group = dedup.groupby(keys, dropna=False)["__realkey__"].nunique().rename("n_curves").reset_index()
    agg_dict = {c: aggfn for c in _BETA_COLS}
    betas = dedup.groupby(keys, dropna=False).agg(agg_dict).reset_index()
    betas = betas.merge(n_per_group, on=keys, how="left", validate="one_to_one")
    betas["mode"] = mode
    atFixed = ",".join(keys)
    betas["atFixed"] = atFixed
    betas["atValues"] = betas.apply(lambda r: _canonical_json_from_kv(keys, r), axis=1)
    betas["atLabel"]  = betas.apply(lambda r: _label_from_kv(mode, keys, r), axis=1)
    betas["ref_group_id"] = betas.apply(lambda r: _group_id(mode, atFixed, r["atValues"]), axis=1)
    betas["ref_stat"] = stat
    betas["computed_at"] = _now_iso()
    betas["analysis_rev"] = "unversioned"
    refs_rows = []
    for _, r in betas.iterrows():
        for c in _BETA_COLS:
            refs_rows.append({
                "mode": r["mode"],
                "atFixed": r["atFixed"],
                "atValues": r["atValues"],
                "atLabel": r["atLabel"],
                "ref_group_id": r["ref_group_id"],
                "anchor": c[4:],
                "beta_ref": r[c],
                "ref_stat": r["ref_stat"],
                "n_curves": int(r["n_curves"]) if pd.notna(r["n_curves"]) else 0,
                "computed_at": r["computed_at"],
                "analysis_rev": r["analysis_rev"],
            })
    refs_long = pd.DataFrame(refs_rows)
    key_df = betas[keys + ["ref_group_id","atFixed","atValues","atLabel","mode"]].copy()
    members = curves.merge(key_df, on=keys, how="inner", validate="many_to_one")
    members = members[["mode","ref_group_id","atFixed","atValues","atLabel","TIcurve_id","graphID","fieldRealization","N"]].copy()
    members["is_used"] = True
    members["computed_at"] = _now_iso()
    members["analysis_rev"] = "unversioned"
    if verbose:
        print(f"[{mode}] gruppi={refs_long['ref_group_id'].nunique()}  members={len(members)}")
    return refs_long, members

def _build_rescaled_points(points: pd.DataFrame, curves: pd.DataFrame, refs_fam: pd.DataFrame, refs_famN: pd.DataFrame, verbose: bool, stat: str) -> pd.DataFrame:
    def _make_map(refs: pd.DataFrame):
        groups = {}
        for gid, g in refs.groupby("ref_group_id", sort=False):
            atFixed = g["atFixed"].iloc[0]
            keys = atFixed.split(",")
            atvals = json.loads(g["atValues"].iloc[0])
            key_tuple = tuple(atvals[k] for k in keys)
            beta_ref_by_anchor = {row["anchor"]: row["beta_ref"] for _, row in g.iterrows()}
            groups[key_tuple] = (gid, keys, beta_ref_by_anchor, atvals)
        return groups
    map_fam  = _make_map(refs_fam)
    map_famN = _make_map(refs_famN)


    req_cols = list(set(_GROUP_KEYS_FAMILY_N + ["TIcurve_id"] + _BETA_COLS))
    cur_min= curves[req_cols].drop_duplicates("TIcurve_id")
    pts = points.merge(cur_min, on="TIcurve_id", how="left", validate="many_to_one", suffixes=("", "_curve"))
    rows = []
    for _, r in pts.iterrows():
        beta_raw = float(r["beta"])
        for mode, mp in (("fixed_family", map_fam), ("fixed_family_N", map_famN)):
            if not mp:
                continue
            some_item = next(iter(mp.values()))
            keys = some_item[1]
            def _val_for(k):
                v = r[k]
                if isinstance(v, (str,)):
                    return v
                try:
                    vf = float(v)
                    if not np.isfinite(vf):
                        return "nan" if np.isnan(vf) else ("inf" if vf > 0 else "-inf")
                    if abs(vf - round(vf)) < 1e-9:
                        return int(round(vf))
                    return float(f"{vf:.12g}")
                except Exception:
                    return str(v)
            key_tuple = tuple(_val_for(k) for k in keys)
            if key_tuple not in mp:
                continue
            ref_group_id, _, ref_betas, atvals = mp[key_tuple]
            for c in _BETA_COLS:
                anchor = c[4:]
                beta_curve = float(r[c]) if pd.notna(r[c]) else np.nan
                beta_ref = ref_betas.get(anchor, np.nan)
                if np.isfinite(beta_raw) and np.isfinite(beta_curve) and beta_curve > 0.0 and np.isfinite(beta_ref):
                    beta_rescaled = beta_raw * (beta_ref / beta_curve)
                else:
                    beta_rescaled = np.nan
                rows.append({
                    "TIcurve_id": r["TIcurve_id"],
                    "run_uid": r["run_uid"],
                    "beta": beta_raw,
                    "T": r["T"],
                    "trajInit": r["trajInit"],
                    "anchor": anchor,
                    "mode": mode,
                    "beta_rescaled": beta_rescaled,
                    "scale_factor": (beta_rescaled / beta_raw) if np.isfinite(beta_rescaled) and beta_raw != 0 else np.nan,
                    "ref_group_id": ref_group_id,
                    "ref_stat": stat,
                    "computed_at": _now_iso(),
                    "analysis_rev": "unversioned",
                })
    return pd.DataFrame(rows)

def main() -> None:
    ap = argparse.ArgumentParser(description="Rescaling β dai riferimenti TI in formato long + refs/members.")
    ap.add_argument("--model", required=True, help="ER, RRG, realGraphs/ZKC, ...")
    ap.add_argument("--ref", choices=["mean","median"], default="mean", help="Statistica per beta_ref (default: mean)")
    ap.add_argument("-v", "--verbose", action="store_true", help="Verbose")
    ns = ap.parse_args()

    paths = _paths_for_model(ns.model)
    curves_path = paths["ti_curves"]
    points_path = paths["ti_points"]
    refs_path   = paths["refs"]
    memb_path   = paths["members"]
    out_points  = paths["points"]

    if ns.verbose:
        print(f"[root] {paths['root']}")
        print(f"[model] {ns.model}")
        print(f"[in]  ti_curves: {curves_path}")
        print(f"[in]  ti_points: {points_path}")
        print(f"[out] refs    : {refs_path}")
        print(f"[out] members : {memb_path}")
        print(f"[out] points  : {out_points}")

    if not os.path.exists(curves_path):
        raise FileNotFoundError(f"File non trovato: {curves_path}")
    if not os.path.exists(points_path):
        raise FileNotFoundError(f"File non trovato: {points_path}")

    curves = pd.read_parquet(curves_path)
    points = pd.read_parquet(points_path)

    if "normalizeQstar" not in curves.columns:
        if "Qstar" not in curves.columns or "N" not in curves.columns:
            raise ValueError("[ti_curves] manca 'normalizeQstar' e non posso derivarla: servono 'Qstar' e 'N'.")
        with np.errstate(divide='ignore', invalid='ignore'):
            curves["normalizeQstar"] = pd.to_numeric(curves["Qstar"], errors="coerce") / pd.to_numeric(curves["N"], errors="coerce")

    req_curves = [
        "TIcurve_id","graphID","fieldRealization","model_type","N","fieldType","fieldSigma",
        "Hext","Hout","Hin","normalizeQstar","T","trajInit","C","fPosJ",
        "betaM","betaL","betaG","betaG2","betaG3","betaG2b","betaG2c",
    ]
    _require_columns(curves, req_curves, "ti_curves")
    req_points = ["TIcurve_id","run_uid","beta","T","trajInit"]
    _require_columns(points, req_points, "ti_points")

    refs_fam, memb_fam   = _build_refs(curves, _GROUP_KEYS_FAMILY, ns.ref, ns.verbose, mode="fixed_family")
    refs_famN, memb_famN = _build_refs(curves, _GROUP_KEYS_FAMILY_N, ns.ref, ns.verbose, mode="fixed_family_N")

    refs_all = pd.concat([refs_fam, refs_famN], ignore_index=True)
    memb_all = pd.concat([memb_fam, memb_famN], ignore_index=True)
    points_long = _build_rescaled_points(points, curves, refs_fam, refs_famN, ns.verbose, ns.ref)

    for p in (refs_path, memb_path, out_points):
        os.makedirs(os.path.dirname(p), exist_ok=True)
    refs_all.to_parquet(refs_path, index=False)
    memb_all.to_parquet(memb_path, index=False)
    points_long.to_parquet(out_points, index=False)

    if ns.verbose:
        print(f"[done] refs    : {refs_path}  (rows={len(refs_all)})")
        print(f"[done] members : {memb_path}  (rows={len(memb_all)})")
        print(f"[done] points  : {out_points} (rows={len(points_long)})")

if __name__ == "__main__":
    main()
