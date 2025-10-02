#!/usr/bin/env python3
import matplotlib as mpl
import argparse, os, pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DEFAULT_FAMILY_IDS = {
    "ER": [
        "44ba08ad6ffa4873",  # subset=all
        "c1003a22dbe9ba73",  # subset=N>30
    ],
    # "ZKC": ["..."],
}


def main():
    ap = argparse.ArgumentParser(description="Scatter β_M vs β_G ORIGINALI per un family-subset")
    ap.add_argument("--model", required=True, help="es. ER, realGraphs/ZKC, ...")
    ap.add_argument("--mode", default="fixed_family", help="es. fixed_family, fixed_family_N")
    ap.add_argument("--family-id", nargs="*", default=None,
                    help="Uno o più family_subset_id. Se omesso, usa DEFAULT_FAMILY_IDS[--model].")
    ap.add_argument("--subset", default="auto",
                    help="Subset (es. all, 'N>30'). Se 'auto' (default), deduce dal family-id nei refs.")

    ap.add_argument("--base", default=None, help="base dir (default: ../../Data/MultiPathsMC/<model>/v1)")
    ap.add_argument("--out", default=None, help="PNG di output")
    ap.add_argument("-v", "--verbose", action="store_true")
    ns = ap.parse_args()

    # default base: esegui da Analysis/PathsMCAnalysis
    base = ns.base or os.path.join("..","..","Data","MultiPathsMC", ns.model, "v1")
    p_refs   = os.path.join(base, "ti", "ti_rescale_refs.parquet")
    p_memb   = os.path.join(base, "ti", "ti_rescale_members.parquet")
    p_curves = os.path.join(base, "ti", "ti_curves.parquet")

    if ns.verbose:
        print("[in]", p_refs, p_memb, p_curves, sep="\n    ")

    refs   = pd.read_parquet(p_refs)
    memb   = pd.read_parquet(p_memb)
    curves = pd.read_parquet(p_curves)

        
    family_ids = ns.family_id or DEFAULT_FAMILY_IDS.get(ns.model, [])
    if isinstance(family_ids, str):
        family_ids = [family_ids]
    if not family_ids:
        raise SystemExit(f"Nessun --family-id e nessun DEFAULT_FAMILY_IDS per il model '{ns.model}'.")

    for fid in family_ids:
        subset = ns.subset
        if not subset or str(subset).lower() == "auto":
            subs = refs[
                (refs["mode"] == ns.mode) &
                (refs["family_subset_id"] == fid)
            ]["subset"].dropna().unique().tolist()

            if not subs:
                raise SystemExit(f"family-id '{fid}' non trovato in refs per mode='{ns.mode}'.")
            if len(subs) > 1:
                # Non dovrebbe succedere con gli ID attuali; se succede, meglio fermarsi.
                raise SystemExit(f"family-id '{fid}' presente in più subset: {subs}. Specifica --subset.")
            subset = subs[0]
        # filtra membership per (mode, subset, family_subset_id)
        mk = (
            (memb["mode"] == ns.mode) &
            (memb["subset"] == subset) &
            (memb["family_subset_id"] == fid)
        )
        mem = memb.loc[mk, ["TIcurve_id", "is_used", "atLabel"]].drop_duplicates()
        if mem.empty:
            print(f"[skip] Nessuna curva per mode={ns.mode}, subset={subset}, family={fid}")
            continue

        # unisci le betas ORIGINALI dalle curves
        cols_need = ["TIcurve_id","betaM","betaG","T","trajInit","N","graphID","fieldRealization"]
        cur = curves[cols_need].copy()
        df = mem.merge(cur, on="TIcurve_id", how="left").dropna(subset=["betaM","betaG"])

        rk = (
            (refs["mode"]==ns.mode) &
            (refs["subset"]==subset) &
            (refs["family_subset_id"]==fid) &
            (refs["anchor"].isin(["M","G"]))
        )
        rsub = refs.loc[rk, ["anchor","beta_ref","atLabel"]].drop_duplicates()
        if rsub.empty:
            print(f"[warn] Riferimenti non trovati per mode={ns.mode}, subset={subset}, family={fid}")
            betaM_ref = betaG_ref = None
            at_label = "(nessun riferimento)"
        else:
            betaM_ref = float(rsub.loc[rsub["anchor"]=="M","beta_ref"].values[0])
            betaG_ref = float(rsub.loc[rsub["anchor"]=="G","beta_ref"].values[0])
            at_label = rsub["atLabel"].iloc[0]

        # --- plot come prima (invariato), ma usa 'subset' e 'fid' al posto di ns.subset/ns.family_id ---
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6,6))
        vmin, vmax = float(df["N"].min()), float(df["N"].max())
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = mpl.cm.get_cmap("cividis")

        used   = df[df["is_used"] == True]
        unused = df[df["is_used"] == False]

        ax.scatter(used["betaM"], used["betaG"], c=used["N"], cmap=cmap, norm=norm,
                s=30, alpha=0.95, edgecolor="k", linewidth=0.3, label="used")
        ax.scatter(unused["betaM"], unused["betaG"], c=unused["N"], cmap=cmap, norm=norm,
                s=30, alpha=0.25, edgecolor="none", label="unused")

        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        cb = plt.colorbar(sm, ax=ax); cb.set_label("N")
        uniqN = sorted(df["N"].dropna().unique())
        if len(uniqN) <= 10:
            cb.set_ticks(uniqN); cb.set_ticklabels([str(int(n)) for n in uniqN])

        if betaM_ref is not None: ax.axvline(betaM_ref, linestyle="--", linewidth=1)
        if betaG_ref is not None: ax.axhline(betaG_ref, linestyle="--", linewidth=1)

        ax.set_xlabel(r"$\beta_M$ (originale)")
        ax.set_ylabel(r"$\beta_G$ (originale)")
        ax.set_title(f"{ns.mode} · {subset}\n{at_label}\nfamily_subset_id={fid}")
        ax.legend(); ax.grid(True)
        # output: se --out è passato e ci sono più famiglie, evita overwrite
        if ns.out:
            root, ext = os.path.splitext(ns.out)
            out_path = f"{root}_{fid}_{ns.mode}_{subset}{ext or '.png'}"
        else:
            out_path = os.path.join(base, "ti", "figs", f"{fid}_{ns.mode}_{subset}_betaM_vs_betaG_ORIG.png")

        from pathlib import Path
        Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(out_path, dpi=160); plt.close(fig)
        print("[saved]", out_path)

if __name__ == "__main__":
    main()
