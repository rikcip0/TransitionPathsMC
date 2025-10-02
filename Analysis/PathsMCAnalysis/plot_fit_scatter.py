#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_fit_scatter.py - Visualizza i fit con punti used/unused (+ opz.: stessa family fuori subset)

- Per ogni riga di ti_linear_fits selezionata:
  * ricostruisce i punti USED del fit (RAW/RESCALED) applicando gli stessi filtri qualita,
  * opzionale: mostra anche gli UNUSED (stesso subset/x) e i punti della STESSA FAMILY ma FUORI SUBSET
    (stesso asse-x): questi hanno colore aggiuntivo e marker '^' senza cambiare i colori esistenti,
  * disegna la retta del fit in unita di -ln(k) vs N.

Output (di default separato per anchor):
  ../../Data/MultiPathsMC/<model>/v1/figures/ti/fits_scatter/<anchor_tag>/<subsetLabel__id8>/...

Opzione per non sovrascrivere tra anchor e tenere tutto insieme:
  --single-folder-per-subset  =>  .../fits_scatter/_combined/<subsetLabel__id8>/... (il filename include l'anchor).
"""
import argparse
from pathlib import Path
import sys
import json
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _base_dir(model: str, data_root: Optional[str]) -> Path:
    root = Path('../../Data') if data_root is None else Path(data_root)
    return root / 'MultiPathsMC' / model / 'v1'

def _ti_dir(model: str, data_root: Optional[str]) -> Path:
    return _base_dir(model, data_root) / 'ti'

def _fig_dir(model: str, data_root: Optional[str], anchor_tag: str, subset_folder: str, combined: bool) -> Path:
    if combined:
        d = _base_dir(model, data_root) / 'figures' / 'ti' / 'fits_scatter' / '_combined' / subset_folder
    else:
        d = _base_dir(model, data_root) / 'figures' / 'ti' / 'fits_scatter' / anchor_tag / subset_folder
    d.mkdir(parents=True, exist_ok=True)
    return d

def _read_parquet(p: Path, tag: str) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(f'[{tag}] non trovato: {p}')
    return pd.read_parquet(p)

def _subset_folder_name(label: str, subset_id: str) -> str:
    safe_lab = (label or '').replace('>','gt').replace('<','lt').replace('=','eq').replace(' ','_').replace('/','-')
    short_id = subset_id[:8] if isinstance(subset_id, str) else 'unknown'
    return f'{safe_lab}__{short_id}'

def _subset_desc(rec: dict) -> str:
    label = str(rec.get('subset_label', ''))
    js = rec.get('subset_selector_json')
    data = None
    if isinstance(js, str) and js.strip():
        try:
            data = json.loads(js)
        except Exception:
            data = None
    elif isinstance(js, dict):
        data = js
    if isinstance(data, dict) and data:
        parts = []
        for k, v in data.items():
            parts.append(f'{k}={v}')
        cond = '; '.join(parts)
        return f'{label} | {cond}' if label else cond
    return label or '(subset)'

def _plot_one_fit(model: str,
                  data_root: Optional[str],
                  fit_row: pd.Series,
                  points: pd.DataFrame,
                  rescaled_points: pd.DataFrame,
                  members: pd.DataFrame,
                  subsets: pd.DataFrame,
                  chi2_thr: float,
                  scale_thr: float,
                  step: float,
                  include_unused_in_plot: bool,
                  include_family_outside: bool,
                  ymode: str,
                  combined_folder: bool,
                  verbose: bool):
    subset_id = fit_row['subset_id']
    kcol = fit_row['kcol']
    beta_kind = fit_row['beta_kind']
    slope = float(fit_row['slope'])
    intercept = float(fit_row['intercept'])
    family_id = fit_row.get('family_id', None)
    anchor = fit_row.get('anchor', None)
    ref_stat = fit_row.get('ref_stat', None)

    # subset meta
    srec = subsets.loc[subsets['subset_id']==subset_id].head(1).to_dict('records')
    srec = srec[0] if srec else {'subset_label': subset_id, 'subset_selector_json': None}
    subset_folder = _subset_folder_name(srec.get('subset_label',''), subset_id)
    subset_desc = _subset_desc(srec)

    # required columns
    need_points_cols = ['TIcurve_id','run_uid','beta',kcol,'chi_chi2']
    if beta_kind == 'rescaled':
        need_points_cols += ['scale2','scale2_valid']
    miss = [c for c in need_points_cols if c not in points.columns]
    if miss:
        raise ValueError(f"[ti_points] colonne mancanti: {miss}")

    need_members_cols = ['TIcurve_id','subset_id','family_id','N','is_used']
    miss = [c for c in need_members_cols if c not in members.columns]
    if miss:
        raise ValueError(f"[ti_subset_members] colonne mancanti: {miss}")

    # membership for subset (+family if present)
    mem = members[members['subset_id']==subset_id].copy()
    if family_id is not None and pd.notna(family_id):
        mem = mem[mem['family_id']==family_id]
    mem = mem[['TIcurve_id','subset_id','family_id','N','is_used']].drop_duplicates()

    # Join with points
    base_cols = ['TIcurve_id','run_uid','beta',kcol,'chi_chi2']
    if beta_kind == 'rescaled':
        base_cols += ['scale2','scale2_valid']
    base = mem.merge(points[base_cols], on='TIcurve_id', how='inner')

    # x selection
    atol = 5e-4
    anchor_tag = 'raw'
    if beta_kind == 'raw':
        beta = float(fit_row['beta'])
        sel = (np.isfinite(base['beta'].to_numpy(float)) &
               (np.abs(base['beta'].to_numpy(float) - beta) <= atol))
        x_value = beta
    else:
        need_r_cols = ['TIcurve_id','run_uid','beta','subset_id','ref_type','ref_stat','beta_rescaled']
        miss = [c for c in need_r_cols if c not in rescaled_points.columns]
        if miss:
            raise ValueError(f"[ti_subset_points_rescaled] colonne mancanti: {miss}")
        rp = rescaled_points[(rescaled_points['subset_id']==subset_id) &
                             (rescaled_points['ref_type']==anchor) &
                             (rescaled_points['ref_stat']==ref_stat)][
                ['TIcurve_id','run_uid','beta','beta_rescaled']
             ].drop_duplicates()
        joined = base.merge(rp, on=['TIcurve_id','run_uid','beta'], how='inner')
        if joined.empty:
            if verbose:
                print(f"[skip] nessun punto rescaled per subset={subset_id}, anchor={anchor}/{ref_stat}")
            return None
        binned = (np.round(joined['beta_rescaled'].to_numpy(float) / step) * step)
        target = float(fit_row['beta_rescaled_bin'])
        sel = np.isfinite(binned) & (np.abs(binned - target) <= 1e-12 + 1e-9*abs(target))
        base = joined
        anchor_tag = str(anchor)
        x_value = target

    # quality masks
    k = base[kcol].to_numpy(float)
    N = base['N'].to_numpy(float)
    chi = base['chi_chi2'].to_numpy(float)
    finite = np.isfinite(k) & np.isfinite(N) & np.isfinite(chi) & (k > 0)
    ok = finite & (chi <= chi2_thr)
    if beta_kind == 'rescaled':
        s2 = base['scale2'].to_numpy(float)
        s2v = base['scale2_valid'].astype(bool).to_numpy()
        ok = ok & s2v & np.isfinite(s2) & (s2 >= scale_thr)

    ok = ok & sel
    used_mask = ok & (base['is_used']==True)

    used = base.loc[used_mask].copy()
    unused = base.loc[(sel) & (~used_mask)].copy() if include_unused_in_plot else pd.DataFrame(columns=base.columns)

    # same-family fuori subset
    fam_out_used = pd.DataFrame(columns=base.columns)
    fam_out_unused = pd.DataFrame(columns=base.columns)
    if include_family_outside and family_id is not None and pd.notna(family_id):
        mem_out = members[(members['family_id']==family_id) & (members['subset_id']!=subset_id)][['TIcurve_id','subset_id','family_id','N','is_used']].drop_duplicates()
        if not mem_out.empty:
            base_out = mem_out.merge(points[base_cols], on='TIcurve_id', how='inner')
            if beta_kind == 'raw':
                sel_out = (np.isfinite(base_out['beta'].to_numpy(float)) &
                           (np.abs(base_out['beta'].to_numpy(float) - x_value) <= atol))
            else:
                rp_out = rescaled_points[(rescaled_points['ref_type']==anchor) &
                                         (rescaled_points['ref_stat']==ref_stat) &
                                         (rescaled_points['subset_id'].isin(mem_out['subset_id'].unique()))][
                             ['TIcurve_id','run_uid','beta','subset_id','beta_rescaled']
                         ].drop_duplicates()
                j2 = base_out.merge(rp_out, on=['TIcurve_id','run_uid','beta'], how='inner')
                if not j2.empty:
                    b2 = (np.round(j2['beta_rescaled'].to_numpy(float) / step) * step)
                    sel_out = np.isfinite(b2) & (np.abs(b2 - x_value) <= 1e-12 + 1e-9*abs(x_value))
                    base_out = j2
                else:
                    sel_out = np.zeros(len(base_out), dtype=bool)

            k2 = base_out[kcol].to_numpy(float) if kcol in base_out.columns else np.full(len(base_out), np.nan)
            N2 = base_out['N'].to_numpy(float) if 'N' in base_out.columns else np.full(len(base_out), np.nan)
            chi2 = base_out['chi_chi2'].to_numpy(float) if 'chi_chi2' in base_out.columns else np.full(len(base_out), np.nan)
            fin2 = np.isfinite(k2) & np.isfinite(N2) & np.isfinite(chi2) & (k2 > 0)
            ok2 = fin2 & (chi2 <= chi2_thr)
            if beta_kind == 'rescaled':
                if 'scale2' in base_out.columns and 'scale2_valid' in base_out.columns:
                    s22 = base_out['scale2'].to_numpy(float)
                    s2v2 = base_out['scale2_valid'].astype(bool).to_numpy()
                    ok2 = ok2 & s2v2 & np.isfinite(s22) & (s22 >= scale_thr)
                else:
                    ok2 = np.zeros_like(ok2, dtype=bool)
            ok2 = ok2 & sel_out
            fam_out_used = base_out.loc[ok2 & (base_out['is_used']==True)].copy()
            fam_out_unused = base_out.loc[ok2 & (~(base_out['is_used']==True))].copy() if include_unused_in_plot else pd.DataFrame(columns=base_out.columns)

    if used.empty and unused.empty and fam_out_used.empty and fam_out_unused.empty:
        if verbose:
            print(f"[skip] nessun punto per fit (subset={subset_id}, anchor={anchor_tag}, k={kcol})")
        return None

    # y mapping
    def yval(df: pd.DataFrame) -> np.ndarray:
        kk = df[kcol].to_numpy(float)
        if ymode == 'minuslnk':
            return -np.log(kk)
        elif ymode == 'logk':
            return np.log(kk)
        else:
            return kk

    # Plot
    fig, ax = plt.subplots(figsize=(7,5), dpi=140)

    if not unused.empty:
        ax.scatter(unused['N'].to_numpy(float), yval(unused), s=22, alpha=0.25, marker='o', label='unused (subset)')
    if not used.empty:
        ax.scatter(used['N'].to_numpy(float), yval(used), s=28, alpha=0.9, marker='o', label='used (subset)')
        xs = used['N'].to_numpy(float)
        x_min, x_max = float(np.min(xs)), float(np.max(xs))
        x_line = np.linspace(x_min, x_max, 100)
        if ymode == 'minuslnk':
            y_line = slope * x_line + intercept
        elif ymode == 'logk':
            y_line = slope * x_line + intercept
        else:
            y_line = np.exp(-(slope * x_line + intercept))
        ax.plot(x_line, y_line, linestyle='-', linewidth=2, label=f'fit subset: slope={slope:.3g}')

    # same-family fuori subset (COLORE AGGIUNTIVO fissato)
    if not fam_out_unused.empty:
        ax.scatter(fam_out_unused['N'].to_numpy(float), yval(fam_out_unused), s=22, alpha=0.35, marker='^', color='C2', label='unused (same-family,out)')
    if not fam_out_used.empty:
        ax.scatter(fam_out_used['N'].to_numpy(float), yval(fam_out_used), s=28, alpha=0.9, marker='^', color='C2', label='used (same-family,out)')

    ax.set_xlabel('N')
    if ymode == 'minuslnk':
        ax.set_ylabel(r'$-\ln k$')
    elif ymode == 'logk':
        ax.set_ylabel(r'$\ln k$')
    else:
        ax.set_ylabel('k')

    if beta_kind == 'raw':
        xtext = rf'$\beta={x_value:.3g}$'
        anchor_descr = 'raw'
    else:
        xtext = rf'$\tilde{{\beta}}={x_value:.3g}$'
        anchor_descr = f'anchor={anchor}/{ref_stat}'
    ax.set_title(f"{subset_desc}\n{kcol} • {anchor_descr} • {xtext}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    out_dir = _fig_dir(model, data_root, (anchor_tag if not combined_folder else '_combined'), subset_folder, combined=combined_folder)
    if beta_kind == 'raw':
        fname = f"fit_scatter__{subset_folder}__beta_{x_value:.3g}__{kcol}"
    else:
        fname = f"fit_scatter__{subset_folder}__betastar_{x_value:.3g}__{anchor}_{ref_stat}__{kcol}"
    if combined_folder:
        if beta_kind == 'raw':
            fname += "__raw"
        else:
            fname += f"__{anchor}_{ref_stat}"
    out_path = out_dir / (fname + ".png")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print("[out]", out_path)
    return out_path

def main():
    ap = argparse.ArgumentParser(description='Visualizza i fit con punti used/unused (-ln k vs N).')
    ap.add_argument('--model', required=True)
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--subset-label', default='all')
    ap.add_argument('--subset-id', nargs='*', default=None)
    ap.add_argument('--kcols', default=None, help='Comma-separated, default: 3 standard')
    ap.add_argument('--anchors', default='raw,M,G', help='Quali x-axis: raw e/o anchors M,L,G,G2,G3,G2b,G2c')
    ap.add_argument('--ref-stats', default='mean', help='Per RESCALED: ref_stat (es. mean o mean,median)')
    ap.add_argument('--rescaled-step', type=float, default=0.025, help='Delta per il binning di beta_rescaled')
    ap.add_argument('--chi2-threshold', type=float, default=0.43)
    ap.add_argument('--scale-threshold', type=float, default=0.33)
    ap.add_argument('--include-unused', action='store_true', help='Mostra anche i punti unused nello stesso subset/bin')
    ap.add_argument('--include-family-outside', action='store_true', help='Mostra punti della stessa family ma fuori dal subset (stesso x)')
    ap.add_argument('--single-folder-per-subset', action='store_true', help='Salva tutte le anchor in una sola cartella per subset; anchor nel filename')
    ap.add_argument('--ymode', choices=['minuslnk','logk','k'], default='minuslnk')
    ap.add_argument('-v','--verbose', action='store_true')
    ns = ap.parse_args()

    ti = _ti_dir(ns.model, ns.data_root)
    fits_path = ti / 'ti_linear_fits.parquet'
    points_path = ti / 'ti_points.parquet'
    rescaled_points_path = ti / 'ti_subset_points_rescaled.parquet'
    members_path = ti / 'ti_subset_members.parquet'
    subsets_path = ti / 'ti_family_subsets.parquet'

    print('[in ]', fits_path)
    print('[in ]', points_path)
    print('[in ]', rescaled_points_path)
    print('[in ]', members_path)
    print('[in ]', subsets_path)

    fits = _read_parquet(fits_path, 'ti_linear_fits')
    points = _read_parquet(points_path, 'ti_points')
    rescaled_points = _read_parquet(rescaled_points_path, 'ti_subset_points_rescaled')
    members = _read_parquet(members_path, 'ti_subset_members')
    subsets = _read_parquet(subsets_path, 'ti_family_subsets')

    if ns.subset_id:
        subset_ids = ns.subset_id
    else:
        if 'subset_label' not in subsets.columns or 'subset_id' not in subsets.columns:
            print('[err] ti_family_subsets deve avere subset_label e subset_id')
            sys.exit(1)
        subset_ids = subsets.loc[subsets['subset_label']==ns.subset_label, 'subset_id'].drop_duplicates().tolist()
    print(f"[subset] scelti {len(subset_ids)} subset_id: {subset_ids[:5]}{'...' if len(subset_ids)>5 else ''}")

    all_anchors = ['M','L','G','G2','G3','G2b','G2c']
    anchors_req = [s.strip() for s in ns.anchors.split(',') if s.strip()]
    anchors_clean = []
    for a in anchors_req:
        if a == 'raw':
            anchors_clean.append('raw')
        elif a in all_anchors:
            anchors_clean.append(a)
        else:
            print(f"[warn] anchor ignorato: {a}")
    if len(anchors_clean) == 0:
        anchors_clean = ['raw','M','G']

    ref_stats = [s.strip() for s in ns.ref_stats.split(',') if s.strip()] if ns.ref_stats else ['mean']
    kcols = [s.strip() for s in ns.kcols.split(',')] if ns.kcols else ['kFromChi','kFromChi_InBetween','kFromChi_InBetween_Scaled']

    count = 0
    for sid in subset_ids:
        for kcol in kcols:
            if 'raw' in anchors_clean:
                ff = fits[(fits['beta_kind']=='raw') & (fits['subset_id']==sid) & (fits['kcol']==kcol)].copy()
                for _, row in ff.iterrows():
                    if _plot_one_fit(ns.model, ns.data_root, row, points, rescaled_points, members, subsets,
                                     chi2_thr=ns.chi2_threshold, scale_thr=ns.scale_threshold,
                                     step=ns.rescaled_step, include_unused_in_plot=ns.include_unused,
                                     include_family_outside=ns.include_family_outside,
                                     ymode=ns.ymode, combined_folder=ns.single_folder_per_subset,
                                     verbose=ns.verbose):
                        count += 1
            for anc in [a for a in anchors_clean if a!='raw']:
                ff = fits[(fits['beta_kind']=='rescaled') & (fits['subset_id']==sid) & (fits['kcol']==kcol) &
                          (fits['anchor']==anc) & (fits['ref_stat'].isin(ref_stats))].copy()
                for _, row in ff.iterrows():
                    if _plot_one_fit(ns.model, ns.data_root, row, points, rescaled_points, members, subsets,
                                     chi2_thr=ns.chi2_threshold, scale_thr=ns.scale_threshold,
                                     step=ns.rescaled_step, include_unused_in_plot=ns.include_unused,
                                     include_family_outside=ns.include_family_outside,
                                     ymode=ns.ymode, combined_folder=ns.single_folder_per_subset,
                                     verbose=ns.verbose):
                        count += 1

    if count == 0:
        print('[warn] Nessuna figura prodotta (controlla i filtri/anchors/kcols/subset).')
    else:
        print(f'[done] figure create: {count}')

if __name__ == '__main__':
    main()
