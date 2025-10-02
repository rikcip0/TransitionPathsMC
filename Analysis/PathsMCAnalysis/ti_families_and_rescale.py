#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ti_families_and_rescale.py  (normQ v6)
-------------------------------------
- Firma di famiglia con 'normalizedQstar' (non 'Qstar').
- Se 'normalizedQstar' manca: **DERIVA** come Qstar / N (con np.errstate).
- Nessun riferimento a fixed_family / fixed_family_N.
- Scrive SOLO le nuove tabelle (families / family_subsets / subset_members / subset_refs / subset_points_rescaled).
- Base path configurabile con --data-root (default ../../Data).
"""

import os
import argparse
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

# ----------------------- Helpers -----------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

def _canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',',':'))

_ANCHOR_MAP = {
    'betaM':'M','betaL':'L','betaG':'G','betaG2':'G2','betaG3':'G3','betaG2b':'G2b','betaG2c':'G2c',
}
_CANON_FAMILY_KEYS = ['model_type','fieldType','fieldSigma','Hext','Hout','Hin','normalizedQstar','C','fPosJ','p']

_TRAJINIT_PRIORITY = [740, 74, 73, 72, 71, 70]

def _trajinit_rank(x):
    try:
        x = int(x)
    except Exception:
        return 999
    try:
        return _TRAJINIT_PRIORITY.index(x)
    except ValueError:
        return 998

def _available_family_keys(columns):
    return [k for k in _CANON_FAMILY_KEYS if k in columns]

def _physics_signature_from_row(row: 'pd.Series', fam_keys) -> dict:
    sig = {}
    for k in fam_keys:
        v = row.get(k)
        if (k in ('fieldSigma','normalizedQstar')) and (v is None or pd.isna(v)):
            v = 0.0  # mai NaN nelle chiavi
        sig[k] = v
    return sig

def _make_family_id_from_row(row: 'pd.Series', fam_keys) -> str:
    sig = _physics_signature_from_row(row, fam_keys)
    s = _canonical_json(sig)
    return hashlib.sha1(s.encode('utf-8')).hexdigest()[:16]

def _subset_spec_from_label(label: str) -> dict:
    if label == 'all':
        return {'and':[]}
    if label == 'N>30':
        return {'and':[{'col':'N','op':'>','val':30}]}
    return {'label':label}

def _anchors_available_for_curve(curve_row: 'pd.Series'):
    labs = []
    for col, lab in _ANCHOR_MAP.items():
        if col in curve_row.index:
            val = curve_row.get(col)
            if pd.notna(val) and np.isfinite(val):
                labs.append(lab)
    return labs

def _at_strings_from_signature(sig: dict):
    items = sorted(sig.items(), key=lambda kv: kv[0])
    fixed = [k for k,_ in items]
    def fmt(v):
        if isinstance(v, float):
            return f'{v:.6g}'
        return str(v)
    values = [fmt(v) for _,v in items]
    atFixed = ','.join(fixed)
    atValues = ','.join(values)
    atLabel  = '; '.join([f'{k}={fmt(v)}' for k,v in items])
    return atFixed, atValues, atLabel

def _make_subset_id(family_id: str, subset_spec: dict) -> str:
    payload = {'family_id': family_id, 'spec': subset_spec}
    return hashlib.sha1(_canonical_json(payload).encode('utf-8')).hexdigest()[:16]

def _make_tip_id(TIcurve_id: str, run_uid: str, beta: float) -> str:
    return hashlib.sha1(f'{TIcurve_id}|{run_uid}|{float(beta):.12g}'.encode('utf-8')).hexdigest()[:16]

def _make_ref_id(subset_id: str, ref_type: str, ref_stat: str) -> str:
    return hashlib.sha1(f'{subset_id}|{ref_type}|{ref_stat}'.encode('utf-8')).hexdigest()[:16]

# ----------------------- Core -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='Model tag (ER, RRG, ...) used to locate <data-root>/MultiPathsMC/<model>/v1/ti/')
    ap.add_argument('--data-root', default='../../Data', help='Root of Data folder (default: ../../Data)')
    ap.add_argument('-v','--verbose', action='store_true')
    ns = ap.parse_args()

    base_dir = Path(ns.data_root) / 'MultiPathsMC' / ns.model / 'v1' / 'ti'
    paths = {
        'ti_dir': str(base_dir),
        'ti_curves': str(base_dir / 'ti_curves.parquet'),
        'ti_points': str(base_dir / 'ti_points.parquet'),
    }

    curves = pd.read_parquet(paths['ti_curves'])
    points = pd.read_parquet(paths['ti_points'])

    if ns.verbose:
        print(f'[in]  ti_curves: {paths["ti_curves"]}')
        print(f'[in]  ti_points: {paths["ti_points"]}')
        print('[curves] rows=', len(curves), 'unique TIcurve_id=', curves["TIcurve_id"].nunique())

    # ---- DERIVE normalizedQstar := Qstar / N if missing ----
    if 'normalizedQstar' not in curves.columns:
        if 'Qstar' in curves.columns and 'N' in curves.columns:
            with np.errstate(divide='ignore', invalid='ignore'):
                curves['normalizedQstar'] = curves['Qstar'].astype(float) / curves['N'].astype(float)
            if ns.verbose:
                print('[derive] normalizedQstar := Qstar / N')
        else:
            if ns.verbose:
                print('[warn] normalizedQstar missing and cannot derive (need Qstar and N)')

    # Families (physics signature)
    fam_keys = _available_family_keys(curves.columns)
    if ns.verbose:
        print(f'[family] keys used for signature: {fam_keys}')
    fam_id_series = curves.apply(lambda r: _make_family_id_from_row(r, fam_keys), axis=1)
    curves = curves.assign(family_id=fam_id_series)

    fam_rows = []
    for _, r in curves.drop_duplicates('family_id').iterrows():
        sig = _physics_signature_from_row(r, fam_keys)
        atFixed, atValues, atLabel = _at_strings_from_signature(sig)
        fam_rows.append({
            'family_id': r['family_id'], **sig, 'family_kind':'physics_signature',
            'atFixed': atFixed, 'atValues': atValues, 'atLabel': atLabel,
            'computed_at': _now_iso(), 'analysis_rev':'unversioned'
        })
    families_df = pd.DataFrame(fam_rows)
    if ns.verbose:
        print(f'[family] count={len(families_df)}')
        for _, row in families_df.iterrows():
            print('  -', row['family_id'], '::', row['atLabel'])

    # Subsets (defaults)
    subset_labels = ['all','N>30']
    subset_rows = []
    for fam in families_df['family_id'].unique():
        for lab in subset_labels:
            spec = _subset_spec_from_label(lab)
            subset_id = _make_subset_id(fam, spec)
            grouping_mode = 'family' if lab=='all' else 'family+condition'
            subset_rows.append({
                'subset_id': subset_id, 'family_id': fam,
                'grouping_mode': grouping_mode, 'subset_label': lab, 'subset_spec': _canonical_json(spec),
                'computed_at': _now_iso(), 'analysis_rev':'unversioned'
            })
    subsets_df = pd.DataFrame(subset_rows)

    # Members
    required_cols = {'graphID','fieldRealization','TIcurve_id'}
    if not required_cols.issubset(curves.columns):
        missing = required_cols - set(curves.columns)
        raise RuntimeError(f'Missing columns in ti_curves: {missing}')
    curves = curves.copy()
    curves['realization_id'] = curves['graphID'].astype(str) + '|' + curves['fieldRealization'].astype(str)
    curves['anchors_available'] = curves.apply(_anchors_available_for_curve, axis=1)

    def subset_filter(df, lab):
        if lab == 'all':
            return df
        if lab == 'N>30':
            if 'N' not in df.columns:
                return df.iloc[0:0]
            return df[df['N'] > 30]
        return df

    member_rows = []
    for fam in families_df['family_id'].unique():
        fam_curves = curves[curves['family_id'] == fam].copy()
        sig = _physics_signature_from_row(fam_curves.iloc[0], fam_keys)
        atFixed, atValues, atLabel = _at_strings_from_signature(sig)
        for _, sub in subsets_df[subsets_df['family_id'] == fam].iterrows():
            lab = sub['subset_label']
            subset_id = sub['subset_id']
            grouping_mode = sub['grouping_mode']
            sub_curves = subset_filter(fam_curves, lab).copy()
            if sub_curves.empty:
                continue
            sub_curves['_rank_traj'] = sub_curves.get('trajInit', pd.Series([None]*len(sub_curves))).map(_trajinit_rank)
            sub_curves['_rank_T'] = -sub_curves.get('T', pd.Series([np.nan]*len(sub_curves))).astype(float).fillna(-np.inf)
            sub_curves = sub_curves.sort_values(by=['realization_id','_rank_T','_rank_traj','TIcurve_id'],
                                                ascending=[True, True, True, True])
            first_idx = sub_curves.groupby('realization_id', as_index=False).head(1).index
            for idx, row in sub_curves.iterrows():
                member_rows.append({
                    'family_id': fam, 'grouping_mode': grouping_mode,
                    'subset_id': subset_id, 'subset_label': lab, 'subset_spec': sub['subset_spec'],
                    'TIcurve_id': row['TIcurve_id'],
                    'graphID': row.get('graphID'), 'fieldRealization': row.get('fieldRealization'),
                    'realization_id': row['realization_id'],
                    'N': row.get('N'), 'T': row.get('T'), 'trajInit': row.get('trajInit'),
                    'is_used': bool(idx in first_idx),
                    'anchors_available': row['anchors_available'],
                    'atFixed': atFixed, 'atValues': atValues, 'atLabel': atLabel,
                    'computed_at': _now_iso(), 'analysis_rev':'unversioned'
                })

    members_df = pd.DataFrame(member_rows)
    if ns.verbose:
        by = members_df.groupby(['subset_label'])['TIcurve_id'].nunique()
        print('[members] curves per subset:', dict(by))

    # Refs
    ref_rows = []
    for subset_id, g in members_df[members_df['is_used']].groupby('subset_id'):
        fam = g['family_id'].iloc[0]; grouping_mode = g['grouping_mode'].iloc[0]
        used_curves = curves[curves['TIcurve_id'].isin(g['TIcurve_id'])].copy()
        for col, ref_type in _ANCHOR_MAP.items():
            if col not in used_curves.columns: continue
            vals = used_curves[col].astype(float); vals = vals[np.isfinite(vals)]
            if len(vals) == 0: continue
            for stat in ['mean','median']:
                beta_ref = float(np.nanmean(vals) if stat=='mean' else np.nanmedian(vals))
                ref_rows.append({
                    'family_id': fam, 'grouping_mode': grouping_mode, 'subset_id': subset_id,
                    'ref_type': ref_type, 'beta_ref': beta_ref, 'ref_stat': stat,
                    'n_used': int(len(vals)), 'computed_at': _now_iso(), 'analysis_rev':'unversioned'
                })
    refs_df = pd.DataFrame(ref_rows)

    # Rescaled points
    anchor_cols = [k for k in _ANCHOR_MAP.keys() if k in curves.columns]
    anchor_values = curves[['TIcurve_id'] + anchor_cols].set_index('TIcurve_id')
    req_pts = {'TIcurve_id','run_uid','beta'}
    if not req_pts.issubset(points.columns):
        missing = req_pts - set(points.columns)
        raise RuntimeError(f'ti_points missing required columns: {missing}')
    pts_join = points.merge(members_df[['subset_id','family_id','grouping_mode','TIcurve_id','is_used']], on='TIcurve_id', how='left')
    pts_join = pts_join[pts_join['is_used'] == True].copy()

    pts_rows = []
    if not pts_join.empty and not refs_df.empty:
        ref_lookup = {(r['subset_id'], r['ref_type'], r['ref_stat']): r['beta_ref'] for _, r in refs_df.iterrows()}
        for _, row in pts_join.iterrows():
            TIcurve_id = row['TIcurve_id']
            curv_av = anchor_values.loc[TIcurve_id] if TIcurve_id in anchor_values.index else None
            subset_id = row['subset_id']
            available_refs = refs_df.loc[refs_df['subset_id']==subset_id, 'ref_type'].unique().tolist()
            for ref_type in available_refs:
                col = next(k for k,v in _ANCHOR_MAP.items() if v==ref_type)
                beta_curve = float(curv_av[col]) if (curv_av is not None and col in curv_av.index and pd.notna(curv_av[col])) else np.nan
                if not np.isfinite(beta_curve) or beta_curve <= 0.0:
                    continue
                for ref_stat in ['mean','median']:
                    beta_ref = ref_lookup.get((subset_id, ref_type, ref_stat), np.nan)
                    if not np.isfinite(beta_ref) or beta_ref <= 0.0:
                        continue
                    scale = float(beta_ref / beta_curve)
                    beta_resc = float(row['beta']) * scale
                    pts_rows.append({
                        'tip_id': _make_tip_id(TIcurve_id, str(row['run_uid']), float(row['beta'])),
                        'ref_id': _make_ref_id(subset_id, ref_type, ref_stat),
                        'family_id': row['family_id'], 'grouping_mode': row['grouping_mode'],
                        'subset_id': subset_id, 'ref_type': ref_type, 'ref_stat': ref_stat,
                        'TIcurve_id': TIcurve_id, 'run_uid': row['run_uid'],
                        'beta': float(row['beta']), 'beta_rescaled': beta_resc, 'scale_factor': scale,
                        'is_used': True, 'computed_at': _now_iso(), 'analysis_rev':'unversioned'
                    })
    pts_rescaled_df = pd.DataFrame(pts_rows, columns=[
        'tip_id','ref_id','family_id','grouping_mode','subset_id','ref_type','ref_stat',
        'TIcurve_id','run_uid','beta','beta_rescaled','scale_factor','is_used','computed_at','analysis_rev'
    ])

    # Write outputs
    base = paths['ti_dir']
    out_families = os.path.join(base, 'ti_families.parquet')
    out_subsets  = os.path.join(base, 'ti_family_subsets.parquet')
    out_members  = os.path.join(base, 'ti_subset_members.parquet')
    out_refs     = os.path.join(base, 'ti_subset_refs.parquet')
    out_points   = os.path.join(base, 'ti_subset_points_rescaled.parquet')

    families_df.to_parquet(out_families, index=False)
    subsets_df.to_parquet(out_subsets, index=False)
    members_df.to_parquet(out_members, index=False)
    refs_df.to_parquet(out_refs, index=False)
    pts_rescaled_df.to_parquet(out_points, index=False)

    print(f'[done:new] families : {out_families} (rows={len(families_df)})')
    print(f'[done:new] subsets  : {out_subsets} (rows={len(subsets_df)})')
    print(f'[done:new] members  : {out_members} (rows={len(members_df)})')
    print(f'[done:new] refs     : {out_refs} (rows={len(refs_df)})')
    print(f'[done:new] points   : {out_points} (rows={len(pts_rescaled_df)})')

if __name__ == '__main__':
    main()
