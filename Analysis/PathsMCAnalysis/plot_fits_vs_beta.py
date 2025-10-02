#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_fits_vs_beta.py  (fix mathtext + separazione per subset)
-------------------------------------------------------------
• Plotta slope vs beta (raw) e vs beta* (rescaled) SENZA mescolare subset.
• Sottocartelle output: figures/ti/{raw|M|G}/{<subset_label>__<subset_id8>}/...
• Legenda: prima riga con descrizione subset (dal JSON se presente) + curve per famiglie (top-3).
• ref_stat default = mean (coerente con fit).

Uso:
  python3 plot_fits_vs_beta.py --model ER --subset-label 'N>30' -v
"""
import argparse
from pathlib import Path
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def _base_dir(model: str, data_root: Path|None) -> Path:
    root = Path('../../Data') if data_root is None else Path(data_root)
    return root / 'MultiPathsMC' / model / 'v1'

def _ti_dir(model: str, data_root: Path|None) -> Path:
    return _base_dir(model, data_root) / 'ti'

def _fig_dir(model: str, data_root: Path|None, anchor_tag: str, subset_folder: str) -> Path:
    d = _base_dir(model, data_root) / 'figures' / 'ti' / anchor_tag / subset_folder
    d.mkdir(parents=True, exist_ok=True)
    return d

def _load_parquet(p: Path, tag: str) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(f'[{tag}] non trovato: {p}')
    return pd.read_parquet(p)

def _pick_top_families(df: pd.DataFrame, top: int) -> list[str]:
    vc = df['family_id'].dropna().value_counts()
    return vc.index[:top].tolist()

def _legend_label_map(families_df: pd.DataFrame|None) -> dict:
    lab = {}
    if families_df is None or families_df.empty:
        return lab
    for _, r in families_df.iterrows():
        fid = r.get('family_id')
        at = r.get('atLabel')
        lab[fid] = at if isinstance(at, str) and len(at) > 0 else fid
    return lab

def _subset_desc(rec: dict) -> str:
    label = str(rec.get('subset_label', ''))
    js = rec.get('subset_selector_json', None)
    data = None
    if isinstance(js, (dict, list)):
        data = js
    elif isinstance(js, str) and js.strip():
        try:
            data = json.loads(js)
        except Exception:
            data = None
    if isinstance(data, dict) and data:
        parts = []
        for k, v in data.items():
            if isinstance(v, (int,float,str)):
                parts.append(f"{k}={v}")
            else:
                parts.append(f"{k}={str(v)}")
        cond = '; '.join(parts)
        if label and cond:
            return f"{label} | {cond}"
        return cond or label or '(subset)'
    return label or '(subset)'

def _subset_folder_name(label: str, subset_id: str) -> str:
    safe_lab = label.replace('>','gt').replace('<','lt').replace('=','eq').replace(' ','_').replace('/','-')
    short_id = subset_id[:8] if isinstance(subset_id, str) else 'unknown'
    return f"{safe_lab}__{short_id}"

def _plot_families(ax, df_plot: pd.DataFrame, x_col: str, y_col: str, fam_ids: list[str], fam_label_map: dict):
    if fam_ids == ['__ALL__']:
        x = df_plot[x_col].astype(float).to_numpy()
        y = df_plot[y_col].astype(float).to_numpy()
        order = np.argsort(x)
        ax.plot(x[order], y[order], marker='o', linestyle='-', label='ALL')
        return
    for fid in fam_ids:
        d = df_plot[df_plot['family_id'] == fid]
        if d.empty:
            continue
        x = d[x_col].astype(float).to_numpy()
        y = d[y_col].astype(float).to_numpy()
        order = np.argsort(x)
        x = x[order]; y = y[order]
        lbl = fam_label_map.get(fid, fid)
        ax.plot(x, y, marker='o', linestyle='-', label=lbl)

def parse_args():
    ap = argparse.ArgumentParser(description='Plot slope vs beta (raw e rescaled), separato per subset.')
    ap.add_argument('--model', required=True, help='ER, RRG, realGraphs/ZKC, ...')
    ap.add_argument('--data-root', default=None, help='Root Data (default: ../../Data)')
    ap.add_argument('--subset-label', default='all', help="Label del subset (es. 'all', 'N>30')")
    ap.add_argument('--subset-id', nargs='*', default=None, help='Uno o più subset_id (sovrascrive subset-label se presenti)')
    ap.add_argument('--family-id', nargs='*', default=None, help='Una o più family_id (opzionale)')
    ap.add_argument('--top-families', type=int, default=3, help='Quante famiglie plottare se non specificate (default: 3)')
    ap.add_argument('--kcols', default=None,
                   help='Colonne k separate da virgola (default: kFromChi,kFromChi_InBetween,kFromChi_InBetween_Scaled)')
    ap.add_argument('--anchors', default='raw,M,G',
                   help="Quali 'assi x' plottare: 'raw' e/o anchors tra M,L,G,G2,G3,G2b,G2c (default: raw,M,G)")
    ap.add_argument('--ref-stats', default='mean', help="Per RESCALED: ref_stat da usare (default: 'mean'; es. 'median' o 'mean,median')")
    ap.add_argument('--value', default='slope', help='Valore da plottare (default: slope)')
    ap.add_argument('-v','--verbose', action='store_true')
    return ap.parse_args()

def main():
    ns = parse_args()

    ti_dir = _ti_dir(ns.model, ns.data_root)
    fits_path      = ti_dir / 'ti_linear_fits.parquet'
    subsets_path   = ti_dir / 'ti_family_subsets.parquet'
    families_path  = ti_dir / 'ti_families.parquet'

    print('[in ]', fits_path)
    print('[in ]', subsets_path)
    print('[in ]', families_path)

    fits = _load_parquet(fits_path, 'ti_linear_fits')
    subsets = _load_parquet(subsets_path, 'ti_family_subsets')
    families = None
    if families_path.exists():
        families = _load_parquet(families_path, 'ti_families')

    needed = ['beta_kind','subset_id','kcol', ns.value]
    missing = [c for c in needed if c not in fits.columns]
    if missing:
        print('[dbg] colonne presenti:', list(fits.columns))
        print('[err] Parquet dei fit non ha colonne essenziali:', missing)
        sys.exit(1)

    # subset selection
    if ns.subset_id:
        subset_ids = ns.subset_id
    else:
        if 'subset_label' not in subsets.columns or 'subset_id' not in subsets.columns:
            print('[err] ti_family_subsets deve avere subset_label e subset_id')
            sys.exit(1)
        subset_ids = subsets.loc[subsets['subset_label']==ns.subset_label, 'subset_id'].drop_duplicates().tolist()
    if ns.verbose:
        print(f"[subset] selezionati {len(subset_ids)} subset_id: {subset_ids}")

    # subset info
    subset_info = {}
    for _, r in subsets.iterrows():
        sid = r.get('subset_id')
        if sid in subset_ids:
            rec = r.to_dict()
            js = rec.get('subset_selector_json')
            if isinstance(js, str):
                try:
                    rec['subset_selector_json'] = json.loads(js)
                except Exception:
                    pass
            subset_info[sid] = {
                'label': r.get('subset_label',''),
                'desc': _subset_desc(rec),
                'folder': _subset_folder_name(r.get('subset_label',''), sid)
            }

    # k columns
    if ns.kcols:
        kcols = [s.strip() for s in ns.kcols.split(',') if s.strip()]
    else:
        kcols = ['kFromChi','kFromChi_InBetween','kFromChi_InBetween_Scaled']

    # anchors
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

    # ref_stats
    ref_stats = [s.strip() for s in ns.ref_stats.split(',') if s.strip()] if ns.ref_stats else ['mean']

    fam_label_map = _legend_label_map(families)

    saved = []
    for sid in subset_ids:
        sinfo = subset_info.get(sid, {'label': sid, 'desc': str(sid), 'folder': f'subset__{str(sid)[:8]}'})

        for kcol in kcols:
            for av in anchors_clean:
                if av == 'raw':
                    df = fits[(fits['beta_kind']=='raw') & (fits['subset_id']==sid) & (fits['kcol']==kcol)].copy()
                    x_col = 'beta'
                    x_label = '$\\beta$'  # FIX: mathtext con dollari
                    anchor_tag = 'raw'
                    if x_col not in df.columns:
                        print("[warn] Nessuna colonna 'beta' nel parquet (RAW). Salto plot RAW.")
                        continue
                else:
                    need_cols = ['anchor','beta_rescaled_bin','ref_stat']
                    if not set(need_cols).issubset(fits.columns):
                        print('[warn] Mancano colonne per plot rescaled:', need_cols)
                        continue
                    df = fits[(fits['beta_kind']=='rescaled') & (fits['subset_id']==sid) &
                              (fits['kcol']==kcol) & (fits['anchor']==av) & (fits['ref_stat'].isin(ref_stats))].copy()
                    x_col = 'beta_rescaled_bin'
                    x_label = '$\\tilde{\\beta}$'  # FIX: mathtext con dollari
                    anchor_tag = av

                if df.empty:
                    if ns.verbose:
                        print(f"[skip] subset={sid} k={kcol} anchor={anchor_tag}: nessuna riga")
                    continue

                fam_ids = _pick_top_families(df, ns.top_families)
                if len(fam_ids) == 0:
                    fam_ids = ['__ALL__']

                df_plot = df.copy() if fam_ids == ['__ALL__'] else df[df['family_id'].isin(fam_ids)].copy()
                if df_plot.empty:
                    if ns.verbose:
                        print(f"[skip] subset={sid} k={kcol} anchor={anchor_tag}: nessuna family dopo filtro")
                    continue

                fig, ax = plt.subplots(figsize=(7,5), dpi=140)
                ax.plot([], [], ' ', label=f"subset: {sinfo['desc']}")
                _plot_families(ax, df_plot=df_plot, x_col=x_col, y_col=ns.value, fam_ids=fam_ids, fam_label_map=fam_label_map)
                ax.set_xlabel(x_label)
                ax.set_ylabel(ns.value)
                title = f"{ns.model} • subset={sinfo['label']} • k={kcol} • {('raw' if av=='raw' else f'anchor={av}') }"
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)

                out_dir = _fig_dir(ns.model, ns.data_root, anchor_tag, sinfo['folder'])
                safe_subset = sinfo['label'].replace('>','gt').replace(' ','_')
                ref_suffix = '' if anchor_tag=='raw' or 'ref_stat' not in df.columns or df['ref_stat'].nunique()==1 else f"__{df['ref_stat'].iloc[0]}"
                out_name = f"fits_vs_beta__{ns.model}__{safe_subset}__{kcol}__{ns.value}{ref_suffix}.png"
                out_path = out_dir / out_name
                fig.tight_layout()
                fig.savefig(out_path)
                plt.close(fig)
                print('[out]', out_path)
                saved.append(out_path)

    if len(saved) == 0:
        print('[warn] Nessuna figura prodotta (controlla filtri e dati).')

if __name__ == '__main__':
    main()
