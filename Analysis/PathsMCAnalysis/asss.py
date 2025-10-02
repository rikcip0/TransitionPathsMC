# verify_rescale_consistency.py
import argparse, os
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/c/Users/ricca/Desktop/College/Codici/TransitionPathsMC/Data/MultiPathsMC")
    ap.add_argument("--model", required=True)
    ap.add_argument("--rev", default="v1")
    ap.add_argument("--subdir", default="ti")
    args = ap.parse_args()

    base = os.path.join(args.root, args.model, args.rev, args.subdir)
    refs_path     = os.path.join(base, "ti_rescale_refs.parquet")
    members_path  = os.path.join(base, "ti_rescale_members.parquet")
    points_path   = os.path.join(base, "ti_rescaled_points.parquet")

    refs    = pd.read_parquet(refs_path)
    members = pd.read_parquet(members_path)
    points  = pd.read_parquet(points_path)

    print(f"[loaded] refs={len(refs)} members={len(members)} points={len(points)}")

    # 1) Una sola curva 'used' per (mode, subset, family_subset_id, realizzazione)
    #    NB: la "realizzazione" la identifichiamo con (graphID, fieldRealization).
    key_real = ["mode", "subset", "family_subset_id", "graphID", "fieldRealization"]
    if not all(k in members.columns for k in key_real + ["is_used"]):
        missing = [k for k in key_real + ["is_used"] if k not in members.columns]
        raise RuntimeError(f"[members] colonne mancanti per la verifica: {missing}")

    g = members.groupby(key_real, dropna=False)["is_used"].sum().reset_index(name="used_sum")
    bad = g[(g["used_sum"] != 1)]
    if bad.empty:
        print("[OK] per ogni (mode, subset, family_subset_id, realizzazione) c'è esattamente una curva is_used=True")
    else:
        print("[FAIL] gruppi con used_sum != 1:")
        print(bad.sort_values(key_real).to_string(index=False))

    # 2) Integrità referenziale: tutti i family_subset_id nei members/points esistono in refs
    id_in_refs = set(refs["family_subset_id"])
    missing_m  = set(members["family_subset_id"]) - id_in_refs
    missing_p  = set(points["family_subset_id"])  - id_in_refs
    if not missing_m and not missing_p:
        print("[OK] tutti i family_subset_id usati da members/points esistono in refs")
    else:
        if missing_m:
            print(f"[FAIL] members con family_subset_id non presente in refs: {len(missing_m)}")
        if missing_p:
            print(f"[FAIL] points con family_subset_id non presente in refs: {len(missing_p)}")

    # 3) Nessuna collisione di ID tra subset diversi per la stessa famiglia logica
    #    (stessa combinazione (mode, atFixed, atValues) ma subset diversi => devono avere ID diversi)
    core_cols = ["mode", "atFixed", "atValues", "subset", "family_subset_id"]
    if not all(c in refs.columns for c in core_cols):
        miss = [c for c in core_cols if c not in refs.columns]
        print(f"[WARN] non posso fare check collisioni subset: mancano {miss}")
    else:
        # prendo coppie di subset diversi sulla stessa (mode, atFixed, atValues)
        cols_key = ["mode", "atFixed", "atValues"]
        df = refs[cols_key + ["subset", "family_subset_id"]]
        merged = df.merge(df, on=cols_key, suffixes=("_a","_b"))
        merged = merged[merged["subset_a"] < merged["subset_b"]]  # coppie ordinate
        collisions = merged[merged["family_subset_id_a"] == merged["family_subset_id_b"]]
        if collisions.empty:
            print("[OK] nessuna collisione di family_subset_id fra subset diversi della stessa famiglia")
        else:
            print("[FAIL] collisioni di ID fra subset diversi:")
            print(collisions[cols_key + ["subset_a","subset_b","family_subset_id_a"]].to_string(index=False))

    # 4) (Facoltativo) Allineamento points -> members: ogni riga points deve trovare la curva usata
    #    per la stessa (mode, subset, family_subset_id, TIcurve_id)
    #    Qui non forziamo che la curva dei points sia quella 'usata' (potresti voler plottare anche non-used),
    #    ma controlliamo che l'unione sia consistente.
    join_cols = ["mode", "subset", "family_subset_id", "TIcurve_id"]
    if all(c in points.columns for c in join_cols) and all(c in members.columns for c in join_cols):
        chk = points[join_cols].merge(members[join_cols + ["is_used"]].drop_duplicates(),
                                      on=join_cols, how="left")
        missing = chk["is_used"].isna().sum()
        if missing == 0:
            print("[OK] points si uniscono a members su (mode, subset, family_subset_id, TIcurve_id)")
        else:
            print(f"[WARN] {missing} righe di points non trovano match in members su {join_cols}")
    else:
        print("[INFO] salto il check points->members: colonne non presenti.")

if __name__ == "__main__":
    main()
