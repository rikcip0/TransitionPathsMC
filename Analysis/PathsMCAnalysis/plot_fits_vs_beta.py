#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_fits_vs_beta.py  (subset-separato, logica precedente ripristinata + physical signature spezzata)
-----------------------------------------------------------------------------------------------
• Identico comportamento di plotting al file precedente che funzionava, con in più
  la gerarchia di cartelle per la signature fisica spezzata: model / C&fPosJ / fieldType / fieldSigma / Hext / (Hin,Hout,nQstar).
• Verbose migliorato: stampa il numero di righe per ogni (subset,kcol,anchor).
"""
import argparse
from pathlib import Path
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Paths & IO ----------------
def _base_dir(model: str, data_root: str|None) -> Path:
    root = Path('../../Data') if data_root is None else Path(data_root)
    return root / 'MultiPathsMC' / model / 'v1'

def _ti_dir(model: str, data_root: str|None) -> Path:
    return _base_dir(model, data_root) / 'ti'

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _fig_dir(model: str, data_root: str|None, anchor_tag: str, phys_sig_parts: list[str], subset_folder: str) -> Path:
    d = _base_dir(model, data_root) / 'figures' / 'ti' / anchor_tag
    for part in phys_sig_parts:
        d = d / part
    d = d / subset_folder
    return _ensure_dir(d)

def _read_parquet(p: Path, tag: str) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(f'[{tag}] non trovato: {p}')
    return pd.read_parquet(p)

# ---------------- Formatting helpers ----------------
def _sanitize(s: str) -> str:
    s = '' if s is None else str(s)
    for a,b in [('>','gt'),('<','lt'),('=','eq'),(' ','_'),('/','-'),(';','__'),(':',''),(',','_')]:
        s = s.replace(a,b)
    return s[:160] if s else 'unknown'

def _fmt_val(v):
    try:
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            return str(v)
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        vf = float(v)
        if abs(vf - round(vf)) < 1e-9:
            return str(int(round(vf)))
        return f"{vf:.6g}"
    except Exception:
        return str(v)

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
        parts = [f"{k}={v}" for k, v in data.items()]
        cond = '; '.join(parts)
        return f"{label} | {cond}" if label else cond
    return label or '(subset)'

def _subset_folder_name(label: str, subset_id: str) -> str:
    safe_lab = label.replace('>','gt').replace('<','lt').replace('=','eq').replace(' ','_').replace('/','-')
    short_id = subset_id[:8] if isinstance(subset_id, str) else 'unknown'
    return f"{safe_lab}__{short_id}"

# ---------------- Physical signature (split folders) ----------------
def _families_for_subset(df_plot: pd.DataFrame, members: pd.DataFrame, subset_id: str) -> list[str]:
    fams = []
    if 'family_id' in df_plot.columns:
        fams = [f for f in df_plot['family_id'].dropna().unique().tolist() if isinstance(f, str)]
    if not fams and 'family_id' in members.columns:
        fams = [f for f in members.loc[members['subset_id']==subset_id, 'family_id'].dropna().unique().tolist() if isinstance(f, str)]
    return fams

def _phys_signature_parts(df_plot: pd.DataFrame, families_df: pd.DataFrame, members: pd.DataFrame, subset_id: str) -> list[str]:
    fams = _families_for_subset(df_plot, members, subset_id)
    if len(fams) != 1 or families_df is None or families_df.empty:
        return [f"multiFamilies__n{len(fams)}" if len(fams)>1 else 'noFamily']
    fid = fams[0]
    row = families_df.loc[families_df['family_id']==fid]
    if row.empty:
        return [f"family__{str(fid)[:8]}"]
    r = row.iloc[0]
    parts = []
    parts.append(_sanitize(r.get('model_type','model')))
    parts.append(_sanitize(f"C_{_fmt_val(r.get('C'))}__fPosJ_{_fmt_val(r.get('fPosJ'))}"))
    parts.append(_sanitize(str(r.get('fieldType'))))
    parts.append(_sanitize(f"fieldSigma_{_fmt_val(r.get('fieldSigma'))}"))
    parts.append(_sanitize(f"Hext_{_fmt_val(r.get('Hext'))}"))
    nq = None
    for qname in ['normalizedQstar','nQstar','normalized_qstar']:
        if qname in r.index:
            nq = r.get(qname)
            break
    parts.append(_sanitize(f"Hin_{_fmt_val(r.get('Hin'))}__Hout_{_fmt_val(r.get('Hout'))}__nQstar_{_fmt_val(nq)}__{str(fid)[:8]}"))
    return parts

# ---------------- Plot helpers ----------------
def _pick_top_families(df: pd.DataFrame, top: int) -> list[str]:
    vc = df['family_id'].dropna().value_counts()
    return vc.index[:top].tolist()

def _legend_label_map(families_df: pd.DataFrame|None) -> dict:
    lab = {}
    if families_df is None or families_df.empty:
        return lab
    for _, r in families_df.iterrows():
        fid = r.get('family_id')
        parts = []
        for key in ['C','fPosJ','fieldType','fieldSigma','Hin','Hout']:
            if key in r.index:
                parts.append(f"{key}={_fmt_val(r[key])}")
        for qname in ['normalizedQstar','nQstar','normalized_qstar']:
            if qname in r.index:
                parts.append(f"nQstar={_fmt_val(r[qname])}")
                break
        lab[fid] = ', '.join(parts) if parts else str(fid)
    return lab

# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser(description='Plot slope vs beta (raw e rescaled), separato per subset.')
    ap.add_argument('--model', required=True)
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--subset-label', default='all')
    ap.add_argument('--subset-id', nargs='*', default=None)
    ap.add_argument('--family-id', nargs='*', default=None)
    ap.add_argument('--top-families', type=int, default=3)
    ap.add_argument('--kcols', default=None)
    ap.add_argument('--anchors', default='raw,M,G')
    ap.add_argument('--ref-stats', default='mean')
    ap.add_argument('--value', default='slope')
    ap.add_argument('-v','--verbose', action='store_true')
    return ap.parse_args()

# ---------------- Main ----------------
def main():
    ns = parse_args()

    ti = _ti_dir(ns.model, ns.data_root)
    fits_path     = ti / 'ti_linear_fits.parquet'
    subsets_path  = ti / 'ti_family_subsets.parquet'
    members_path  = ti / 'ti_subset_members.parquet'
    families_path = ti / 'ti_families.parquet'

    print('[in ]', fits_path)
    print('[in ]', subsets_path)
    print('[in ]', members_path)
    print('[in ]', families_path)

    fits     = _read_parquet(fits_path, 'ti_linear_fits')
    subsets  = _read_parquet(subsets_path, 'ti_family_subsets')
    members  = _read_parquet(members_path, 'ti_subset_members')
    families = _read_parquet(families_path, 'ti_families') if families_path.exists() else pd.DataFrame()

    need = ['beta_kind','subset_id','kcol', ns.value]
    miss = [c for c in need if c not in fits.columns]
    if miss:
        print('[dbg] colonne disponibili:', list(fits.columns))
        print('[err] mancano colonne in ti_linear_fits:', miss)
        sys.exit(1)

    # subset selection (no mixing)
    if ns.subset_id:
        subset_ids = ns.subset_id
    else:
        if 'subset_label' not in subsets.columns or 'subset_id' not in subsets.columns:
            print('[err] ti_family_subsets deve avere subset_label e subset_id')
            sys.exit(1)
        subset_ids = subsets.loc[subsets['subset_label']==ns.subset_label, 'subset_id'].drop_duplicates().tolist()
    if ns.verbose:
        print(f"[subset] selezionati {len(subset_ids)} subset_id: {subset_ids}")

    # subset meta
    subset_meta = {}
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
            subset_meta[sid] = {
                'label': r.get('subset_label',''),
                'desc': _subset_desc(rec),
                'folder': _subset_folder_name(r.get('subset_label',''), sid)
            }

    # k columns & anchors & ref_stats
    kcols = [s.strip() for s in ns.kcols.split(',')] if ns.kcols else ['kFromChi','kFromChi_InBetween','kFromChi_InBetween_Scaled']
    all_anchors = ['M','L','G','G2','G3','G2b','G2c']
    anchors_req = [s.strip() for s in ns.anchors.split(',') if s.strip()]
    anchors = []
    for a in anchors_req:
        if a == 'raw' or a in all_anchors:
            anchors.append(a)
    if not anchors:
        anchors = ['raw','M','G']
    ref_stats = [s.strip() for s in ns.ref_stats.split(',') if s.strip()] if ns.ref_stats else ['mean']

    fam_label_map = _legend_label_map(families)

    saved = []
    for sid in subset_ids:
        meta = subset_meta.get(sid, {'label': sid, 'desc': str(sid), 'folder': f'subset__{str(sid)[:8]}'})

        for kcol in kcols:
            for anc in anchors:
                if anc == 'raw':
                    sel = (fits['beta_kind']=='raw') & (fits['subset_id']==sid) & (fits['kcol']==kcol)
                    df = fits.loc[sel, ['family_id','subset_id','beta','kcol',ns.value]].dropna(subset=['beta', ns.value])
                    x_col = 'beta'
                    x_label = '$\\beta$'
                    anchor_tag = 'raw'
                else:
                    need_cols = ['anchor','beta_rescaled_bin','ref_stat']
                    if not set(need_cols).issubset(fits.columns):
                        if ns.verbose:
                            print('[skip] manca una delle colonne richieste per RESCALED:', need_cols)
                        continue
                    sel = (fits['beta_kind']=='rescaled') & (fits['subset_id']==sid) & (fits['kcol']==kcol) &                           (fits['anchor']==anc) & (fits['ref_stat'].isin(ref_stats))
                    df = fits.loc[sel, ['family_id','subset_id','beta_rescaled_bin','ref_stat','kcol',ns.value]].dropna(subset=['beta_rescaled_bin', ns.value])
                    x_col = 'beta_rescaled_bin'
                    x_label = '$\\tilde{\\beta}$'
                    anchor_tag = f'rescaled/{anc}'

                if ns.verbose:
                    print(f"[sel] subset={sid[:8]} k={kcol} anchor={anc} -> rows={len(df)}")
                if df.empty:
                    continue

                # family filter (optional)
                if ns.family_id:
                    df = df[df['family_id'].isin(ns.family_id)]
                    if ns.verbose:
                        print(f"[filter] family_id -> rows={len(df)}")
                    if df.empty:
                        continue

                # physical signature folders
                phys_sig_parts = _phys_signature_parts(df, families, members, sid)

                # choose families to draw (like the earlier file)
                fam_ids = df['family_id'].dropna().unique().tolist()
                if ns.family_id:
                    fam_ids = [f for f in fam_ids if f in ns.family_id]
                if not fam_ids:
                    fam_ids = ['__ALL__']
                else:
                    fam_ids = fam_ids[:ns.top_families] if ns.top_families>0 else fam_ids

                # Plot
                fig, ax = plt.subplots(figsize=(7,5), dpi=140)
                ax.plot([], [], ' ', label=f"subset: {meta['desc']}")
                if fam_ids == ['__ALL__']:
                    x = df[x_col].astype(float).to_numpy()
                    y = df[ns.value].astype(float).to_numpy()
                    order = np.argsort(x)
                    ax.plot(x[order], y[order], marker='o', linestyle='-', label='ALL')
                else:
                    for fid in fam_ids:
                        d = df[df['family_id']==fid]
                        if d.empty:
                            continue
                        x = d[x_col].astype(float).to_numpy()
                        y = d[ns.value].astype(float).to_numpy()
                        order = np.argsort(x)
                        lbl = fam_label_map.get(fid, str(fid)[:8])
                        ax.plot(x[order], y[order], marker='o', linestyle='-', label=lbl)

                ax.set_xlabel(x_label)
                ax.set_ylabel(ns.value)
                ax.set_title(f"{ns.model} • subset={meta['label']} • k={kcol} • {('raw' if anc=='raw' else f'anchor={anc}')}")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)

                out_dir = _fig_dir(ns.model, ns.data_root, anchor_tag, phys_sig_parts, meta['folder'])
                safe_subset = meta['label'].replace('>','gt').replace(' ','_')
                ref_suffix = ''
                if anc != 'raw' and 'ref_stat' in df.columns and df['ref_stat'].nunique() == 1:
                    ref_suffix = f"__{df['ref_stat'].iloc[0]}"
                out_name = f"fits_vs_beta__{ns.model}__{safe_subset}__{kcol}__{ns.value}{ref_suffix}.png"
                out_path = out_dir / out_name
                fig.tight_layout()
                fig.savefig(out_path)
                plt.close(fig)
                print('[out]', out_path)
                saved.append(out_path)

    if not saved:
        print('[warn] Nessuna figura prodotta (controlla filtri/dati).')

if __name__ == '__main__':
    main()
