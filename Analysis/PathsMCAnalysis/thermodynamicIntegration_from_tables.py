#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
thermodynamicIntegration_from_tables.py (patched)

- Legge i parquet in Data/MultiPathsMC/<model>/v1: runs_params, runs_results, stdmcs
- Ricostruisce gli array globali attesi dalla tua implementazione storica
- Supporta colonne stdMC con o senza prefisso 'stMC_' (es. N / stMC_N, beta / stMC_beta, ...)
- Mantiene i token stringa ('nan') per i key di matching (graphID, fieldType, fieldRealization, betaOfExtraction, configurationIndex, trajs*)
- Esegue la tua thermodynamicIntegration senza alterazioni concettuali
- Scrive v1/ti/ti_curves.parquet + un manifest con diagnostica

Uso:
  python3 thermodynamicIntegration_from_tables.py --model realGraphs/ZKC -v
  python3 thermodynamicIntegration_from_tables.py --model ER -v
"""

from __future__ import annotations

import argparse
import json
from scipy.special import comb
import sys, time, os
import numpy as np
import pandas as pd
from pathlib import Path

from scipy import interpolate
from scipy.integrate import quad
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
sys.path.append('../')
from MyBasePlots.plotWithDifferentColorbars import plotWithDifferentColorbars

fieldTypeDictionary ={"2":"gauss", "1":"bernoulli", "nan":"noField"}
fieldTypePathDictionary ={"gauss":"stdGaussian", "bernoulli":"stdBernoulli"}
trajInitShortDescription_Dict= {0: "stdMC", 70: "Random", '71': "Ref 12", 72: "Ref 21", 73: "Annealing", 74: "Annealing", '740': "AnnealingF", -2:"Fit"}
edgeColorPerInitType_Dic={'0': "None", 70: "lightGreen", '71': "black", 72: "purple", 73: "orange", 74: "orange", '740': "red", -2:"black"}
preferred_trajInit = [740, 74, 73, 72, 71, 70]
nameOfFoldersContainingGraphs = ["fPosJ","realGraphs"
                               ]
# ---------- Helper functions (dal tuo originale) ----------

def getUniqueXAndYZAccordingToZ(x, y, criterion, preliminaryFilter=None):
    print("xprima",x)
    if preliminaryFilter is None:
        preliminaryFilter =  ~np.isnan(y)
    else:
        preliminaryFilter = np.logical_and(~np.isnan(y), preliminaryFilter)
    best_indices = {}
    filterToReturn = np.full(len(x), False)
    # Iteriamo su ogni valore unico in filteredStdMCsBetas
    for value in np.sort(np.unique(x[preliminaryFilter])):
        print("value",value)
        # Trova gli indici corrispondenti a questo valore unico
        indices = np.where(np.logical_and(x == value, preliminaryFilter))[0]

        if len(indices) > 1:
            print(len(indices))
            print(criterion[indices])
            # Se ci sono più di un indice, scegli quello con il valore massimo in lastMeasureMC
            best_index = indices[np.nanargmax(criterion[indices])]
        else:
            # Se c'è solo un indice, lo prendiamo direttamente
            best_index = indices[0]          
        # Memorizza l'indice migliore
        best_indices[value] = best_index
        filterToReturn[best_index]=True
    filtered_x = np.asarray(x[list(best_indices.values())])
    filtered_y = np.asarray(y[list(best_indices.values())])
    filtered_criterion = np.asarray(criterion[list(best_indices.values())])
    return filtered_x, filtered_y, filtered_criterion, filterToReturn

def addLevelOnNestedDictionary( structure, param_tuples, levelToAdd):
    current_level = structure
    for param_tuple in param_tuples:
        if param_tuple not in current_level:
            current_level[param_tuple] = {}
        current_level = current_level[param_tuple]
        
    for key, value in levelToAdd.items():
        current_level[key] = value
        
    return current_level

def getLevelFromNestedStructureAndKeyName(structure, param_tuples, keyName):
    current_level = structure
    for param_tuple in param_tuples:
        if param_tuple in current_level:
            current_level = current_level[param_tuple]
        else:
            return None
    if keyName in current_level:
        return current_level
    else:
        return None

def P_t(n, q, p_up):
    """
    print(n,q, p_up)
    # Calcolo del logaritmo della funzione per maggiore precisione
    log_comb = np.log(comb(n, (n+q)/2))
    log_term1 = (n + q) / 2 * np.log(p_up)
    log_term2 = (n - q) / 2 * np.log(1. - p_up)
    # Somma dei logaritmi e poi esponenziale per ottenere il risultato finale
    log_result = log_comb + log_term1 + log_term2
    return np.exp(log_result)
    """
    return comb(n, (n+q)/2)*((p_up)**((n+q)/2))*((1-p_up)**((n-q)/2))



def findFoldersWithString(parent_dir, target_strings):
    result = []
    # Funzione ricorsiva per cercare le cartelle
    def search_in_subfolders(directory, livello=1):
        if livello > 10:
            return
        for root, dirs, _ in os.walk(directory):
            rightLevelReachedFor=[]
            for dir_name in dirs:
                full_path = os.path.join(root, dir_name)
                # Controlla se il nome della cartella corrente è "stdMCs" o "PathsMCs"
                if any(folder in dir_name for folder in nameOfFoldersContainingGraphs):
                    # Cerca le cartelle che contengono "_run" nel loro nome
                    rightLevelReachedFor.append(dir_name)
                    for subdir in os.listdir(full_path):
                        if all(string in os.path.join(full_path, subdir) for string in target_strings):
                            result.append(os.path.join(full_path, subdir))
                
            for r in rightLevelReachedFor:
                dirs.remove(r)
                
            # Se non troviamo "stdMCs" o "PathsMCs", passiamo al livello successivo
            for dir_name in dirs:
                search_in_subfolders(os.path.join(root, dir_name), livello+1)
            break  # Si processa solo il primo livello di cartelle per evitare ricorsione non necessaria
    
    # Inizia la ricerca dalla directory di base
    search_in_subfolders(parent_dir)
    return result

def delete_files_in_folder(folder_path):
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            # Ricorsivamente elimina i file all'interno della cartella
            delete_files_in_folder(item_path)
            # Dopo aver eliminato i file, elimina la cartella stessa
            os.rmdir(item_path)

# ---------- Helpers di canonizzazione ----------
def _to_str_with_nan(a, length):
    import numpy as _np, pandas as _pd
    if a is None:
        return _np.array(["nan"] * length, dtype=object)
    arr = _np.asarray(a, dtype=object)
    out = _np.empty(len(arr), dtype=object)
    for i, v in enumerate(arr):
        if _pd.isna(v) or v is None or (isinstance(v, float) and _np.isnan(v)) or (isinstance(v, str) and v.strip()==""):
            out[i] = "nan"
        else:
            out[i] = str(v)
    return out

_to_str_with_nan_std = _to_str_with_nan

# ---------- Utility per scoprire root ----------
def _exists_graphs_root(path: Path) -> bool:
    if not path or not path.exists():
        return False
    try:
        subs = [p.name for p in path.iterdir() if p.is_dir()]
    except Exception:
        return False
    return any(n in subs for n in ("ER","RRG","realGraphs"))

def discover_graphs_root(cli_value: Path|None) -> Path:
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

def _list_available_models(outdir: Path) -> list[str]:
    models = []
    for d in outdir.glob("*"):
        if (d / "v1" / "runs_params" / "runs_params.parquet").exists():
            models.append(d.name)
        for d2 in d.glob("*"):
            if (d2 / "v1" / "runs_params" / "runs_params.parquet").exists():
                models.append("{}/{}".format(d.name, d2.name))
    return sorted(set(models))

# ---------- Globals attesi dalla TI ----------
# params
N=T=beta=h_out=h_in=h_ext=Qstar=graphID=C=fPosJ=None
fieldType=fieldSigma=fieldRealization=None
betaOfExtraction=firstConfigurationIndex=secondConfigurationIndex=refConfInitID=refConfMutualQ=None
lastMeasureMC=MCprint=None
trajsJumpsInitID=trajsExtremesInitID=None
runPath=simulationType=ID=None

# results
TIbeta=TIhout=TIQstar=None
realTime=realTimeErr=None
chi_tau=chi_m=chi_c=None
chi_tau2=chi_m2=chi_c2=chi_chi2=None
meanBarrier=stdDevBarrier=None

# stdMC
stMC_Hext=stMC_fieldType=stMC_fieldSigma=stMC_fieldRealization=None
stMC_N=stMC_beta=None
stMC_Hout=stMC_Qstar=stMC_nQstar=stMC_graphID=None
stMC_betaOfExtraction=stMC_configurationIndex=stMC_fieldMean=None
stMC_MC=stMC_TIbeta=None

# derived
normalizedQstar=None

def _series_or_default(df: pd.DataFrame, col: str, default=np.nan):
    if col in df.columns:
        return df[col].to_numpy()
    return np.full(len(df), default)

def _col_any(df: pd.DataFrame, names, default=np.nan, as_str=False):
    """Ritorna df[first existing name] tra `names`; altrimenti array di default."""
    n = len(df)
    for name in names:
        if name in df.columns:
            arr = df[name].to_numpy()
            if as_str:
                return _to_str_with_nan_std(arr, n)
            return arr
    return np.full(n, default)

def _ensure_tables(outdir: Path, model: str):
    base = outdir / model / "v1"
    p_path = base / "runs_params" / "runs_params.parquet"
    r_path = base / "runs_results" / "runs_results.parquet"
    if not p_path.exists() or not r_path.exists():
        avail = _list_available_models(outdir)
        msg = ["Missing parquet under: {}".format(base),
               "Expected files:",
               "  - {}".format(p_path),
               "  - {}".format(r_path)]
        if avail:
            msg.append("Available models under {}:".format(outdir))
            for m in avail:
                msg.append("  - {}".format(m))
        raise FileNotFoundError("\n".join(msg))
    return base

# ---------- Loader ----------
def load_tables_as_arrays(model: str, graphs_root: Path, outdir: Path, includes: list[str]|None=None, verbose: bool=False) -> int:
    global N,T,beta,h_out,h_in,h_ext,Qstar,graphID,C,fPosJ,p,model_type
    global fieldType,fieldSigma,fieldRealization
    global betaOfExtraction,firstConfigurationIndex,secondConfigurationIndex,refConfInitID,refConfMutualQ
    global lastMeasureMC,MCprint,trajsJumpsInitID,trajsExtremesInitID,runPath,simulationType,ID
    global TIbeta,TIhout,TIQstar,realTime,realTimeErr,chi_tau,chi_m,chi_c,chi_tau2,chi_m2,chi_c2,chi_chi2,meanBarrier,stdDevBarrier
    global stMC_Hext,stMC_fieldType,stMC_fieldSigma,stMC_fieldRealization,stMC_N,stMC_beta
    global stMC_Hout,stMC_Qstar,stMC_nQstar,stMC_graphID,stMC_betaOfExtraction,stMC_configurationIndex,stMC_fieldMean,stMC_MC,stMC_TIbeta
    global normalizedQstar

    base = _ensure_tables(outdir, model)
    p_path = base / "runs_params" / "runs_params.parquet"
    r_path = base / "runs_results" / "runs_results.parquet"
    s_path = base / "stdmcs" / "stdmcs.parquet"

    par = pd.read_parquet(p_path)
    res = pd.read_parquet(r_path)
    df = res.merge(par, on="run_uid", how="left", suffixes=("", "_p"))

    globals()['run_uid'] = df['run_uid'].to_numpy()
    globals()['model_type'] = df.get('model_type', pd.Series(['unknown']*len(df))).to_numpy()

    if includes:
        mask = np.zeros(len(df), dtype=bool)
        rp = df.get("runPath", pd.Series([""]*len(df)))
        for tok in includes:
            mask |= rp.astype(str).str.contains(str(tok), na=False)
        df = df.loc[mask].copy()

    if verbose:
        print("[load] rows in runs: {}".format(len(df)))

    # results
    TIbeta = _series_or_default(df, 'TIbeta')
    TIhout = _series_or_default(df, 'TIhout')
    TIQstar= _series_or_default(df, 'TIQstar')
    realTime    = _series_or_default(df, 'realTime')
    realTimeErr = _series_or_default(df, 'realTimeErr')
    chi_tau = _series_or_default(df, 'chi_tau')
    chi_m   = _series_or_default(df, 'chi_m')
    chi_c   = _series_or_default(df, 'chi_c')
    chi_tau2= _series_or_default(df, 'chi_tau2')
    chi_m2  = _series_or_default(df, 'chi_m2')
    chi_c2  = _series_or_default(df, 'chi_c2')
    chi_chi2= _series_or_default(df, 'chi_chi2')
    meanBarrier   = _series_or_default(df, 'meanBarrier')
    stdDevBarrier = _series_or_default(df, 'stdDevBarrier')

    # ---- derived 'scale' and 'scale2' as in original TI ----
    try:
        _T = np.asarray(T, dtype=float)
    except Exception:
        _T = T
    try:
        _chi_m2 = np.asarray(chi_m2, dtype=float)
        _chi_c2 = np.asarray(chi_c2, dtype=float)
        _chi_chi2 = np.asarray(chi_chi2, dtype=float)
    except Exception:
        _chi_m2, _chi_c2, _chi_chi2 = chi_m2, chi_c2, chi_chi2
    scale = _chi_m2 * _T + _chi_c2
    scale2 = scale.copy()
    # Mask like your historical logic: scale2 < 0.33 or chi_chi2 > 0.43 -> NaN
    with np.errstate(invalid='ignore'):
        bad = (scale2 < 0.33) | (_chi_chi2 > 0.43)
    scale2[bad] = np.nan
    globals().update(dict(scale=scale, scale2=scale2))

    # params
    N    = _series_or_default(df,"N")
    N = N.astype(int)
    T    = _series_or_default(df,"T")
    beta = _series_or_default(df,"beta")
    h_out= _series_or_default(df,"h_out")
    h_in = _series_or_default(df,"h_in")
    h_ext= _series_or_default(df,"h_ext")
    Qstar= _series_or_default(df,"Qstar")
    graphID = _to_str_with_nan(df.get("graphID", pd.Series(["nan"]*len(df))).to_numpy(), len(df))
    C    = _series_or_default(df,"C")
    fPosJ= _series_or_default(df,"fPosJ")
    p    = _series_or_default(df,"p")

    fieldType  = _to_str_with_nan(df.get("fieldType", pd.Series(["nan"]*len(df))).to_numpy(), len(df))
    fieldType =  np.asarray([fieldTypeDictionary[value] for value in fieldType])
    fieldSigma = _series_or_default(df,"fieldSigma")
    fieldRealization = _to_str_with_nan(df.get("fieldRealization", pd.Series(["nan"]*len(df))).to_numpy(), len(df))

    betaOfExtraction      = _to_str_with_nan(_series_or_default(df,"betaOfExtraction"), len(df))
    firstConfigurationIndex  = _to_str_with_nan(_series_or_default(df,"firstConfigurationIndex"), len(df))
    secondConfigurationIndex = _to_str_with_nan(_series_or_default(df,"secondConfigurationIndex"), len(df))
    refConfInitID         = _to_str_with_nan(df.get("refConfInitID", pd.Series(["nan"]*len(df))).to_numpy(), len(df))
    refConfMutualQ        = _series_or_default(df,"refConfMutualQ")

    lastMeasureMC = _series_or_default(df,"lastMeasureMC")
    MCprint       = _series_or_default(df,"MCprint")
    trajsJumpsInitID    = _to_str_with_nan(df.get("trajsJumpsInitID", pd.Series(["nan"]*len(df))).to_numpy(), len(df))
    trajsExtremesInitID = _to_str_with_nan(df.get("trajsExtremesInitID", pd.Series(["nan"]*len(df))).to_numpy(), len(df))
    runPath       = _to_str_with_nan(df.get("runPath", pd.Series(["nan"]*len(df))).to_numpy(), len(df))
    simulationType= _series_or_default(df,"simulationType")
    ID            = _to_str_with_nan(df.get("ID", pd.Series(["nan"]*len(df))).to_numpy(), len(df))

    # stdMCs (supporta bare e 'stMC_' prefix)
    if s_path.exists():
        std = pd.read_parquet(s_path)
        stMC_Hext           = _col_any(std, ["Hext","stMC_Hext"])   
        stMC_fieldType      = _col_any(std, ["fieldType","stMC_fieldType"], as_str=True)
        stMC_fieldType =  np.asarray([fieldTypeDictionary[value] for value in stMC_fieldType])
        stMC_fieldSigma     = _col_any(std, ["fieldSigma","stMC_fieldSigma"]) 
        stMC_fieldRealization = _col_any(std, ["fieldRealization","stMC_fieldRealization"], as_str=True)
        stMC_N              = _col_any(std, ["N","stMC_N"])        
        stMC_beta           = _col_any(std, ["beta","stMC_beta"]) 

        # Hout could be named h_out or Hout
        stMC_Hout           = _col_any(std, ["Hout","h_out","stMC_Hout","stMC_h_out"]) 
        stMC_Qstar          = _col_any(std, ["Qstar","stMC_Qstar"]) 
        if ("Qstar" not in std.columns and "stMC_Qstar" not in std.columns) and ("nQstar" in std.columns or "stMC_nQstar" in std.columns):
            try:
                ncol = "nQstar" if "nQstar" in std.columns else "stMC_nQstar"
                Ncol = "N" if "N" in std.columns else ("stMC_N" if "stMC_N" in std.columns else None)
                if Ncol is not None:
                    stMC_Qstar = std[ncol].to_numpy(dtype=float) * std[Ncol].to_numpy(dtype=float)
            except Exception:
                pass

        stMC_graphID        = _col_any(std, ["graphID","stMC_graphID"], as_str=True) 
        stMC_betaOfExtraction   = _col_any(std, ["betaOfExtraction","stMC_betaOfExtraction"], as_str=True) 
        stMC_configurationIndex = _col_any(std, ["configurationIndex","stMC_configurationIndex"], as_str=True)
        stMC_fieldMean      = _col_any(std, ["fieldMean","stMC_fieldMean"]) 
        stMC_MC             = _col_any(std, ["MC","stMC_MC"]) 

        # Fallback per beta: alcuni stdMC hanno solo TIbeta
        if (np.asarray(stMC_beta).size == 0) or (np.all(pd.isna(stMC_beta)) and (("TIbeta" in std.columns) or ("stMC_TIbeta" in std.columns))):
            stMC_beta = _col_any(std, ["TIbeta","stMC_TIbeta"]) 

        stMC_TIbeta         = _col_any(std, ["TIbeta","stMC_TIbeta"]) 

        try:
            stMC_nQstar = stMC_Qstar / stMC_N
        except Exception:
            stMC_nQstar = np.full(len(std), np.nan)

        if verbose:
            print("[load] rows in stdmcs: {}".format(len(std)))
            # Diagnostics
            def _nn(a):
                try:
                    return int(np.sum(~pd.isna(a)))
                except Exception:
                    return -1
            print("[std] non-NaN counts:",
                  "N=", _nn(stMC_N),
                  "beta=", _nn(stMC_beta),
                  "TIbeta=", _nn(stMC_TIbeta),
                  "Hext=", _nn(stMC_Hext),
                  "Hout=", _nn(stMC_Hout),
                  "Qstar=", _nn(stMC_Qstar),
                  "graphID=", _nn(stMC_graphID),
                  "fieldType=", _nn(stMC_fieldType),
                  "fieldSigma=", _nn(stMC_fieldSigma),
                  "fieldRealization=", _nn(stMC_fieldRealization),
                  )
    else:
        stMC_Hext = stMC_fieldType = stMC_fieldSigma = stMC_fieldRealization = np.array([])
        stMC_N = stMC_beta = np.array([])
        stMC_Hout = stMC_Qstar = stMC_nQstar = np.array([])
        stMC_graphID = np.array([], dtype=object)
        stMC_betaOfExtraction = stMC_configurationIndex = stMC_fieldMean = np.array([])
        stMC_MC = stMC_TIbeta = np.array([])
        if verbose:
            print("[load] stdmcs not found; proceeding without them.")

    # derived
    try:
        normalizedQstar = np.where(N>0, Qstar / N, np.nan)
    except Exception:
        normalizedQstar = np.full(len(df), np.nan)

    # refConfMutualQ fallback: -N quando mancante
    L = len(df)
    try:
        qf = np.asarray(refConfMutualQ, dtype=float)
    except Exception:
        qf = np.full(L, np.nan)
    Nf = np.asarray(N, dtype=float) if N is not None else np.full(L, np.nan)
    N = N.astype(int)
    qf = np.where(np.isnan(qf), -Nf, qf)
    try:
        refConfMutualQ = np.nan_to_num(qf, nan=0.0).astype(int)
    except Exception:
        refConfMutualQ = np.full(L, 0, dtype=int)

    # working arrays richiesti dalla TI
    _n = len(df)
    ZFromTIBeta = np.full(_n, np.nan, dtype=np.float64)
    rescaledBetas_M  = np.full(_n, np.nan, dtype=np.float64)
    rescaledBetas_L  = np.full(_n, np.nan, dtype=np.float64)
    rescaledBetas_G  = np.full(_n, np.nan, dtype=np.float64)
    rescaledBetas_G2 = np.full(_n, np.nan, dtype=np.float64)
    rescaledBetas_G3 = np.full(_n, np.nan, dtype=np.float64)
    rescaledBetas_G2b= np.full(_n, np.nan, dtype=np.float64)
    rescaledBetas_G2c= np.full(_n, np.nan, dtype=np.float64)

    minusLnKFromChi         = np.full(_n, np.nan, dtype=np.float64)
    minusLnKFromChi_2       = np.full(_n, np.nan, dtype=np.float64)
    minusLnKFromChi_2_scaled= np.full(_n, np.nan, dtype=np.float64)

    discretizedRescaledBetas_M  = np.full(_n, np.nan, dtype=np.float64)
    discretizedRescaledBetas_G  = np.full(_n, np.nan, dtype=np.float64)
    discretizedRescaledBetas_G2 = np.full(_n, np.nan, dtype=np.float64)
    discretizedRescaledBetas_G3 = np.full(_n, np.nan, dtype=np.float64)
    discretizedRescaledBetas_G2b= np.full(_n, np.nan, dtype=np.float64)
    discretizedRescaledBetas_G2c= np.full(_n, np.nan, dtype=np.float64)

    betaMax         = np.full(_n, np.nan, dtype=np.float64)
    averageBetaMax  = np.full(_n, np.nan, dtype=np.float64)

    kFromChi                   = np.full(_n, np.nan, dtype=np.float64)
    kFromChi_InBetween         = np.full(_n, np.nan, dtype=np.float64)
    kFromChi_InBetween_Scaled  = np.full(_n, np.nan, dtype=np.float64)

    tentativeBarrier   = np.full(_n, np.nan, dtype=np.float64)
    tentativeBarrier_2 = np.full(_n, np.nan, dtype=np.float64)
    tentativeBarrier_3 = np.full(_n, np.nan, dtype=np.float64)

    globals().update(dict(
        ZFromTIBeta=ZFromTIBeta,
        rescaledBetas_M=rescaledBetas_M, rescaledBetas_L=rescaledBetas_L,
        rescaledBetas_G=rescaledBetas_G, rescaledBetas_G2=rescaledBetas_G2,
        rescaledBetas_G3=rescaledBetas_G3, rescaledBetas_G2b=rescaledBetas_G2b,
        rescaledBetas_G2c=rescaledBetas_G2c,
        minusLnKFromChi=minusLnKFromChi, minusLnKFromChi_2=minusLnKFromChi_2,
        minusLnKFromChi_2_scaled=minusLnKFromChi_2_scaled,
        discretizedRescaledBetas_M=discretizedRescaledBetas_M,
        discretizedRescaledBetas_G=discretizedRescaledBetas_G,
        discretizedRescaledBetas_G2=discretizedRescaledBetas_G2,
        discretizedRescaledBetas_G3=discretizedRescaledBetas_G3,
        discretizedRescaledBetas_G2b=discretizedRescaledBetas_G2b,
        discretizedRescaledBetas_G2c=discretizedRescaledBetas_G2c,
        betaMax=betaMax, averageBetaMax=averageBetaMax,
        kFromChi=kFromChi, kFromChi_InBetween=kFromChi_InBetween,
        kFromChi_InBetween_Scaled=kFromChi_InBetween_Scaled,
        tentativeBarrier=tentativeBarrier, tentativeBarrier_2=tentativeBarrier_2,
        tentativeBarrier_3=tentativeBarrier_3
    ))

    return len(N)

# ---------- Sanitizzazione ----------
def sanitize_globals(verbose: bool=False):
    # Lasciamo intatti i token stringa ('nan') per i key di matching
    import numpy as _np

    # Converte in float solo i campi numerici, non i key stringa
    def _as_float_array(a):
        import pandas as pd, numpy as _np
        if a is None:
            return np.array([], dtype=float)
        aa = np.asarray(a, dtype=object)
        out = []
        for x in aa:
            if x is None or (x is pd.NA):
                out.append(np.nan)
            else:
                try:
                    xv = float(x)
                    out.append(xv if not _np.isnan(xv) else np.nan)
                except Exception:
                    out.append(np.nan)
        return np.asarray(out, dtype=float)

    num_names = [
        "N","T","beta","h_out","h_in","h_ext","Qstar","C","fPosJ",
        "fieldSigma","refConfMutualQ","lastMeasureMC","MCprint",
        "simulationType",
        "stMC_Hext","stMC_fieldSigma","stMC_N","stMC_beta",
        "stMC_Hout","stMC_Qstar","stMC_nQstar","stMC_MC","stMC_TIbeta",
        "stMC_fieldMean"
    ]
    for name in num_names:
        try:
            globals()[name] = _as_float_array(globals().get(name, None))
        except Exception:
            pass

    # Riempie NaN di alcuni numerici specifici (matching robusto)
    for _name in ("fieldSigma","stMC_fieldSigma","h_ext","stMC_Hext"):
        if _name in globals() and globals()[_name] is not None:
            arr = globals()[_name]
            try:
                mask = _np.isnan(arr)
                if mask.any():
                    arr = arr.copy()
                    arr[mask] = 0.0
                    globals()[_name] = arr
                    if verbose:
                        print("[sanitize] replaced NaN -> 0.0 in", _name, "count=", int(mask.sum()))
            except Exception:
                pass

# ---------- La tua thermodynamicIntegration (copiata) ----------

# ---- TIcurve helpers ----
import hashlib as _hashlib
from datetime import datetime as _dt

def _canon_token(x):
    import numpy as _np, pandas as _pd
    if x is None or (isinstance(x, float) and _np.isnan(x)) or (isinstance(x, str) and x.strip()=="") or (x is _pd.NA):
        return "nan"
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x).strip()

def make_TIcurve_id(model_type, N, graphID, fieldType, fieldSigma, fieldRealization,
                    Hext, Hout, Hin, Qstar, T, trajInit, betaOfExtraction, firstConfigurationIndex, secondConfigurationIndex):
    parts = [
        _canon_token(model_type), _canon_token(N), _canon_token(graphID), _canon_token(fieldType),
        _canon_token(fieldSigma), _canon_token(fieldRealization), _canon_token(Hext),
        _canon_token(Hout), _canon_token(Hin), _canon_token(Qstar),
        _canon_token(T), _canon_token(trajInit), _canon_token(betaOfExtraction),
        _canon_token(firstConfigurationIndex), _canon_token(secondConfigurationIndex)
    ]
    key = "|".join(parts).encode("utf-8")
    return _hashlib.blake2b(key, digest_size=8).hexdigest()

def write_ti_curves_points(outdir: Path, model: str, curves_rows: list, points_rows: list, verbose: bool=False):
    base = outdir / model / "v1" / "ti"
    base.mkdir(parents=True, exist_ok=True)
    import pandas as _pd
    dfc = _pd.DataFrame(curves_rows)
    dfp = _pd.DataFrame(points_rows)
    (base / "ti_curves.parquet").unlink(missing_ok=True)
    (base / "ti_points.parquet").unlink(missing_ok=True)
    dfc.to_parquet(base / "ti_curves.parquet", index=False)
    dfp.to_parquet(base / "ti_points.parquet", index=False)
    if verbose:
        print(f"[write] {base/'ti_curves.parquet'} rows={len(dfc)}")
        print(f"[write] {base/'ti_points.parquet'} rows={len(dfp)}")
    return base/'ti_curves.parquet', base/'ti_points.parquet'
def thermodynamicIntegration(filt, analysis_path):
    # -- diagnostics placeholders to avoid NameError --
    max_value = np.nan
    max_value2 = np.nan
    max_value3 = np.nan
    curves_rows = []
    points_rows = []

    # Accumulatore per Z (persistente lungo la scansione)
    Zdict = {}
    cleanFilt = np.zeros_like(filt, dtype=bool)
    TDN=[]
    TDTrajInit=[]
    TDT=[]
    TDBetOfEx=[]
    TDFirstConfIndex=[]
    TDSecondConfIndex=[]
    TDGraphId=[]
    TDFieldType=[]
    TDFieldReali=[]
    TDFieldSigma=[]
    TDHext=[]
    TDHout=[]
    TDHin=[]
    TDnQstar=[]
    TDBetaM=[]
    TDBetaG=[]
    TDBetaG2=[]
    TDBetaG3=[]
    TDbetaG2b=[]
    TDbetaG2c=[]
    TDBetaL=[]
    TDZmax=[]

    TIFolder= os.path.join(analysis_path, 'TI')

    if not os.path.exists(TIFolder):
        os.makedirs(TIFolder, exist_ok=True)
    else:
        delete_files_in_folder(TIFolder)


    #specifying graph in 2 cycle
    for sim_C, sim_Hext in set(zip(C[filt],h_ext[filt])):
        TIFilt = np.logical_and.reduce([ C==sim_C, h_ext==sim_Hext, filt])
        st_TIFilt = np.logical_and.reduce([stMC_Hext==sim_Hext])
        for sim_fieldType, sim_fieldSigma in set(zip(fieldType[TIFilt], fieldSigma[TIFilt])):
            TIFilt1 = np.logical_and(TIFilt, np.logical_and.reduce([fieldType==sim_fieldType, fieldSigma==sim_fieldSigma]))
            st_TIFilt1 = np.logical_and(st_TIFilt, np.logical_and.reduce([stMC_fieldType==sim_fieldType, stMC_fieldSigma==sim_fieldSigma]))
            #specifying configurations
            for sim_betOfEx, sim_firstConfIndex, sim_secondConfIndex, sim_Qif in set(zip(betaOfExtraction[TIFilt1],firstConfigurationIndex[TIFilt1], secondConfigurationIndex[TIFilt1], refConfMutualQ[TIFilt1])):
                TIFilt2 = np.logical_and(TIFilt1, np.logical_and.reduce([betaOfExtraction==sim_betOfEx, firstConfigurationIndex==sim_firstConfIndex, secondConfigurationIndex==sim_secondConfIndex, refConfMutualQ==sim_Qif]))
                st_TIFilt2 = np.logical_and(st_TIFilt1, np.logical_and.reduce([stMC_betaOfExtraction==sim_betOfEx, stMC_configurationIndex==sim_secondConfIndex]))
                #specifying stochastic measure parameters
                for sim_Hin, sim_Hout, sim_nQstar in set(zip(h_in[TIFilt2], h_out[TIFilt2], normalizedQstar[TIFilt2])):
                    TIFilt3 = np.logical_and(TIFilt2, np.logical_and.reduce([h_in==sim_Hin, h_out==sim_Hout, normalizedQstar==sim_nQstar]))
                    st_TIFilt3 = np.logical_and(st_TIFilt2, np.logical_and.reduce([stMC_Hout==sim_Hout, stMC_nQstar==sim_nQstar]))

                    realizationsId=[]
                    betaMaxOverRealizations=[]
                    betaLOverRealizations=[]
                    betaGOverRealizations=[]
                    betaG2OverRealizations=[]
                    betaG3OverRealizations=[]
                    betaG2bOverRealizations=[]
                    betaG2cOverRealizations=[]
                    for sim_N, sim_graphID, sim_fieldRealization in set(zip(N[TIFilt3],graphID[TIFilt3], fieldRealization[TIFilt3])):
                        sim_Qstar=(int)(sim_N*sim_nQstar)
                        TIFilt4 = np.logical_and(TIFilt3, np.logical_and.reduce([N==sim_N, graphID==sim_graphID, fieldRealization==sim_fieldRealization]))
                        st_TIFilt4 = np.logical_and(st_TIFilt3, np.logical_and.reduce([stMC_N==sim_N , stMC_graphID==sim_graphID, stMC_fieldRealization==sim_fieldRealization]))
                        print(sim_N, sim_graphID, sim_fieldRealization, sim_Hin, sim_Hout, sim_nQstar)
                        print(stMC_MC)
                        stdMCsBetas_forThisTDSetting, stdMCsTIBetas_forThisTDSetting, stdMCsMC_forThisTDSetting, filterForStdMCs_forThisTDSetting = getUniqueXAndYZAccordingToZ(stMC_beta, stMC_TIbeta, stMC_MC, preliminaryFilter=st_TIFilt4)

                        #specifico il tempo
                        betaLForThisRealization=0.
                        betaGForThisRealization=0.
                        betaG2ForThisRealization=0.
                        betaG3ForThisRealization=0.
                        betaG2bForThisRealization=0.
                        betaG2cForThisRealization=0.
                        betaMaxForThisRealization=0.
                        betaMaxForThisRealizationCounter=0
                        betaMax2ForThisRealization=0.
                        betaMax2ForThisRealizationCounter=0

                        TIPlotsFolder = os.path.join(TIFolder, f'N{sim_N}', f'h{sim_Hext}_f{sim_fieldType}{sim_fieldSigma}' if sim_fieldSigma!=0. else f'h{sim_Hext}_noField', f'g{sim_graphID}_fr{sim_fieldRealization}' if sim_fieldSigma!=0. else f'g{sim_graphID}',
                                                     f'bExt{sim_betOfEx}_cs{sim_firstConfIndex}_{sim_secondConfIndex}_{sim_Qif}' if (sim_firstConfIndex!="nan" and sim_firstConfIndex is not None) else 'FM',
                                                     f'meas_{(str)(sim_Hin)}_{(str)(sim_Hout)}_{(sim_nQstar):.3f}' if sim_Hin is not np.inf else f'meas_inf_inf_{(sim_nQstar):.3f}')
                        # ---- Inject head: p{p}[C{C}]/fPosJ{fPosJ:.2f} (C only ER/RRG; p fallback=2) ----
                        try:
                            _mask_for_path = TIFilt3 if 'TIFilt3' in locals() else (TIFilt2 if 'TIFilt2' in locals() else (filt if 'filt' in locals() else slice(None)))
                        except Exception:
                            _mask_for_path = slice(None)
                        try:
                            sim_fPosJ = float(np.nanmean(fPosJ[_mask_for_path]))
                        except Exception:
                            sim_fPosJ = np.nan
                        try:
                            sim_C = float(np.nanmean(C[_mask_for_path])) if 'C' in globals() else np.nan
                        except Exception:
                            sim_C = np.nan
                        try:
                            sim_p = float(np.nanmean(p[_mask_for_path])) if 'p' in globals() else np.nan
                        except Exception:
                            sim_p = np.nan
                        try:
                            # Prefer model id from runPath to decide presence of C
                            if 'runPath' in globals():
                                if hasattr(_mask_for_path, 'dtype'):
                                    idxs = np.where(_mask_for_path)[0]
                                    _idx = int(idxs[0]) if idxs.size>0 else 0
                                else:
                                    _idx = 0
                                _model_from_path = str(runPath[_idx])
                            else:
                                _model_from_path = ''
                        except Exception:
                            _model_from_path = ''
                        include_C = (('ER' in _model_from_path) or ('RRG' in _model_from_path)) and np.isfinite(sim_C)
                        sim_p_fmt = int(sim_p) if np.isfinite(sim_p) else 2
                        head_segment = f"p{sim_p_fmt}"
                        if include_C:
                            head_segment = head_segment + f"C{sim_C:.3f}"
                        fposj_segment = f"fPosJ{sim_fPosJ:.2f}" if np.isfinite(sim_fPosJ) else "fPosJnan"
                        _rel = os.path.relpath(TIPlotsFolder, TIFolder)
                        TIPlotsFolder = os.path.join(TIFolder, head_segment, fposj_segment, _rel)

                        thisTransitionData_T=[]
                        thisTransitionData_trajInit=[]
                        thisTransitionData_betaMax=[]
                        thisTransitionData_betaL=[]
                        thisTransitionData_betaG=[]
                        thisTransitionData_betaG2=[]
                        thisTransitionData_betaG2b=[]
                        thisTransitionData_betaG2c=[]
                        thisTransitionData_betaG3=[]

                        for sim_T, sim_trajInit in set(zip(T[TIFilt4], trajsExtremesInitID[TIFilt4])):
                            pathMCFilt_forThisTAndInit = np.logical_and.reduce([TIFilt4, T==sim_T, trajsExtremesInitID==sim_trajInit])


                            if (0. not in stdMCsBetas_forThisTDSetting and 0. not in beta[pathMCFilt_forThisTAndInit]):
                                continue
                            maxPathsMCsBeta = np.nanmax(beta[pathMCFilt_forThisTAndInit])
                            stdMC_filtForThisTAndInit = np.logical_and(filterForStdMCs_forThisTDSetting, stMC_beta<maxPathsMCsBeta)
                            stdMC_filtForThisTAndInit = np.logical_and(filterForStdMCs_forThisTDSetting, stMC_beta<maxPathsMCsBeta)


                            pathMCBetas_forThisTAndInit, pathMCTIs_forThisTAndInit, pathMCMCs_forThisTAndInit, pathMCFilter_forThisTAndInit = getUniqueXAndYZAccordingToZ(beta, TIbeta, lastMeasureMC, preliminaryFilter=pathMCFilt_forThisTAndInit)

                            smallestPathsMcBetaToConsider_i = np.nanargmin(pathMCBetas_forThisTAndInit)
                            smallestPathsMcBetaToConsider= pathMCBetas_forThisTAndInit[smallestPathsMcBetaToConsider_i]
                            smallestPathsMcTIToConsider = pathMCTIs_forThisTAndInit[smallestPathsMcBetaToConsider_i]

                            if smallestPathsMcBetaToConsider<=np.nanmin(stdMCsBetas_forThisTDSetting):
                                print("EECCCCOOCCC,", sim_T, smallestPathsMcBetaToConsider,stdMCsBetas_forThisTDSetting)
                                largestStdMcBetaToConsider_i=np.nanargmin(stdMCsBetas_forThisTDSetting)
                                largestStdMcBetaToConsider=stdMCsBetas_forThisTDSetting[largestStdMcBetaToConsider_i]
                                largestStdMcTIToConsider = stdMCsTIBetas_forThisTDSetting[largestStdMcBetaToConsider_i]
                            else:  
                                largestStdMcBetaToConsider_i = np.nanargmax(stdMCsBetas_forThisTDSetting*(stdMCsBetas_forThisTDSetting<smallestPathsMcBetaToConsider))
                                largestStdMcBetaToConsider = stdMCsBetas_forThisTDSetting[largestStdMcBetaToConsider_i]
                                largestStdMcTIToConsider = stdMCsTIBetas_forThisTDSetting[largestStdMcBetaToConsider_i]
                            #print("Z", smallestPathsMcBetaToConsider)
                            #print("A", largestStdMcBetaToConsider)
                            TIDifferenceMax=np.nanmax(stMC_TIbeta[stdMC_filtForThisTAndInit])-np.nanmin(pathMCTIs_forThisTAndInit)
                            """
                            for i, stdMcBeta in enumerate(stMC_beta[stdMC_filtForThisTAndInit]):
                                stdMcTIbeta = stMC_TIbeta[stdMC_filtForThisTAndInit][i]
                                if stdMcBeta< largestStdMcBetaToConsider:
                                    continue
                                pathsMcsToConsider = pathMCBetas_forThisTAndInit>=stdMcBeta
                                temp=pathMCBetas_forThisTAndInit[pathsMcsToConsider]
                                if(len(temp)==0):
                                    continue
                                smallestLEqPathMCBeta_index=np.nanargmin(pathMCBetas_forThisTAndInit[pathsMcsToConsider])
                                PathsMcTIToCompare=pathMCTIs_forThisTAndInit[pathsMcsToConsider][smallestLEqPathMCBeta_index]
                                if abs((stdMcTIbeta-PathsMcTIToCompare)/TIDifferenceMax)<0.:#0.003:
                                    largestStdMcBetaToConsider = stdMcBeta
                                    largestStdMcTIToConsider = stdMcTIbeta
                                    smallestLargerPathMCBeta_index=np.nanargmin(pathMCBetas_forThisTAndInit[pathMCBetas_forThisTAndInit>stdMcBeta])
                                    smallestPathsMcBetaToConsider = pathMCBetas_forThisTAndInit[pathMCBetas_forThisTAndInit>stdMcBeta][smallestLargerPathMCBeta_index]
                                    smallestPathsMcTIToConsider = pathMCTIs_forThisTAndInit[pathMCBetas_forThisTAndInit>stdMcBeta][smallestLargerPathMCBeta_index]
                            """

                            #print("B", largestStdMcBetaToConsider)
                            #print("C", smallestPathsMcBetaToConsider)
                            stdMC_filtForThisTAndInit_used = np.logical_and(stdMC_filtForThisTAndInit, stMC_beta <= largestStdMcBetaToConsider)
                            pathsMC_filtForThisTAndInit_used = np.logical_and(pathMCFilter_forThisTAndInit, beta >= smallestPathsMcBetaToConsider)

                            temp = np.sort(np.concatenate([stMC_beta[stdMC_filtForThisTAndInit_used], beta[pathsMC_filtForThisTAndInit_used]]))
                            maxBetaNotTooSpaced = np.nanmax([temp[i] for i in range(len(temp)-1,0,-1) if temp[i] - temp[i-1] <= 0.101])

                            stdMC_filtForThisTAndInit_used = np.logical_and(stdMC_filtForThisTAndInit_used, stMC_beta <= maxBetaNotTooSpaced)
                            pathsMC_filtForThisTAndInit_used = np.logical_and(pathsMC_filtForThisTAndInit_used, beta <= maxBetaNotTooSpaced+0.1)
                            stdMC_filtForThisTAndInit_unused = np.logical_and(stdMC_filtForThisTAndInit, ~stdMC_filtForThisTAndInit_used)
                            pathsMC_filtForThisTAndInit_unused = np.logical_and(pathMCFilter_forThisTAndInit, ~pathsMC_filtForThisTAndInit_used)

                            stdMCBetas_forThisTAndInit_used = stMC_beta[stdMC_filtForThisTAndInit_used]
                            stdMCTIBetas_forThisTAndInit_used = stMC_TIbeta[stdMC_filtForThisTAndInit_used]
                            stdMCsMC_forThisTAndInit_used = stMC_MC[stdMC_filtForThisTAndInit_used]

                            pathMCBetas_forThisTAndInit_used = beta[pathsMC_filtForThisTAndInit_used]
                            pathMCTIs_forThisTAndInit_used = TIbeta[pathsMC_filtForThisTAndInit_used]
                            pathMCMCs_forThisTAndInit_used = lastMeasureMC[pathsMC_filtForThisTAndInit_used]

                            #print(filteredStdMCsBetasForThisT, filteredBetas)
                            if len(pathMCBetas_forThisTAndInit_used)<4:
                                continue
                            stdMCBetas_forThisTAndInit_used_sort = np.argsort(stdMCBetas_forThisTAndInit_used)
                            pathMCBetas_forThisTAndInit_used_sort = np.argsort(pathMCBetas_forThisTAndInit_used)
                            print(stdMCBetas_forThisTAndInit_used[stdMCBetas_forThisTAndInit_used_sort])
                            TIx=np.concatenate([stdMCBetas_forThisTAndInit_used[stdMCBetas_forThisTAndInit_used_sort],
                                                pathMCBetas_forThisTAndInit_used[pathMCBetas_forThisTAndInit_used_sort]])

                            TIy=np.concatenate([stdMCTIBetas_forThisTAndInit_used[stdMCBetas_forThisTAndInit_used_sort],
                                                pathMCTIs_forThisTAndInit_used[pathMCBetas_forThisTAndInit_used_sort]])
                            f_interp = None
                            TIfunction= None
                            Zfunction= None
                            betaMax= np.nan
                            #print(largestStdMcTIToConsider, smallestPathsMcTIToConsider, TIDifferenceMax)
                            if np.fabs(largestStdMcTIToConsider-smallestPathsMcTIToConsider)<TIDifferenceMax/15.:
                                print("doing g ", sim_graphID)
                                #print(TIx)
                                #print("BU",TIy)
                                print(TIx, TIy)
                                f_interp = interpolate.InterpolatedUnivariateSpline(TIx, TIy, k=3)

                                p_up_0 = (sim_N*(1.+sim_Qif))/(2.*sim_N)
                                p_up_t = 0.5*(1.+(2.*p_up_0-1.)*np.exp(-2.*sim_T))

                                ZAtBet0 =0.
                                sim_Qstar = (int)(sim_Qstar)
                                sim_N = (int)(sim_N)
                                for this_q_star in range(int(sim_Qstar), int(sim_N)+1, 2):
                                    if (this_q_star%2)!=(sim_N%2):
                                        this_q_star=this_q_star+1
                                    ZAtBet0+=P_t(sim_N, this_q_star, p_up_t)
                                def integral_to_x(x_point, aoF=f_interp, maxValue = maxBetaNotTooSpaced):
                                    if np.isscalar(x_point):  # Check if it's a scalar
                                        if x_point < 0 or x_point > maxValue:
                                            return np.nan
                                        integral, _ = quad(aoF, 0., x_point, limit=150)
                                        return integral
                                    else:
                                        return np.array([integral_to_x(x, aoF, maxValue) for x in x_point]) 

                                def exp_integral_to_x(x_point, aoF=f_interp, factor=ZAtBet0,  maxValue = maxBetaNotTooSpaced):
                                    return factor* np.exp(integral_to_x(x_point, aoF, maxValue))
                                print(f"Therm. int. complete for N={sim_N} g={sim_graphID} fieldType={sim_fieldType} sigma={sim_fieldSigma} T={sim_T} init={trajInitShortDescription_Dict[sim_trajInit]}")

                                TIfunction= integral_to_x
                                Zfunction= exp_integral_to_x
                                betaMax = minimize_scalar(lambda z:-Zfunction(z), bounds=(np.nanmin(TIx), np.nanmax(TIx))).x
                            #else:
                            #    print(largestStdMcTIToConsider, smallestPathsMcTIToConsider ,TIDifferenceMax)
                            whereToFindBetaCs= findFoldersWithString('../../Data/Graphs', [f'graph{sim_graphID}' if sim_C!=0 else f'{sim_graphID}'])

                            if len(whereToFindBetaCs)>1:
                                print("Errore, piu di un grafo trovato")
                                print(whereToFindBetaCs)
                                return None
                            betaL=np.nan
                            betaG=np.nan
                            betaG2=np.nan
                            betaG3=np.nan
                            betaG2b=np.nan
                            betaG2c=np.nan
                            if len(whereToFindBetaCs)!=0:
                                whereToFindBetaCs = whereToFindBetaCs[0]
                                #print(sim_fieldType)
                                if sim_fieldType != "noField" and sim_fieldType != "nan" and sim_fieldType is not None:
                                    whereToFindBetaCs = os.path.join(whereToFindBetaCs, 'randomFieldStructures',
                                                                    fieldTypePathDictionary[sim_fieldType])
                                    whereToFindBetaCs = os.path.join(whereToFindBetaCs, f'realization{sim_fieldRealization}')

                                whereToFindBetaCs= os.path.join(whereToFindBetaCs,'graphAnalysis/graphData.json')
                                if os.path.exists(whereToFindBetaCs):
                                    with open(whereToFindBetaCs, 'r') as file:
                                        graphData = json.load(file)
                                        betaL = graphData['beta_c'].get('localApproach', np.nan)
                                        betaG = graphData['beta_c'].get('globalApproach', np.nan)
                                        betaG2 = graphData['beta_c'].get('globalApproach_2', np.nan)
                                        betaG3 = graphData['beta_c'].get('globalApproach_3', np.nan)
                                        betaG2b = graphData['beta_c'].get('globalApproach_4', np.nan)
                                        betaG2c = graphData['beta_c'].get('globalApproach_5', np.nan)

                            else:
                                print("Info su beta di ", whereToFindBetaCs," non disponibili.")

                            os.makedirs(TIPlotsFolder, exist_ok=True)

                            vLines = []
                            if not np.isnan(betaG):
                                vLines.append([betaG, r"$\beta_{G}$", "blue"])
                            if not np.isnan(betaG2):
                                vLines.append([betaG2, r"$\beta_{G,2}$", "green"])
                            if not np.isnan(betaG3):
                                vLines.append([betaG3, r"$\beta_{G,3}$", "purple"])
                            if not np.isnan(betaG2b):
                                vLines.append([betaG2b, r"$\beta_{G,2b}$", "hotpink"])
                            if not np.isnan(betaG2c):
                                vLines.append([betaG2c, r"$\beta_{G,2c}$", "darkslategray"])

                            if not np.isnan(betaMax):
                                vLines.append([betaMax, r"$\beta_{M}$", "red"])

                            f=None
                            if f_interp is not None:
                                f=[[f_interp], [None]]
                            plotWithDifferentColorbars(f'T{sim_T}_{trajInitShortDescription_Dict[sim_trajInit]}_U',
                                                       np.concatenate([stdMCBetas_forThisTAndInit_used, pathMCBetas_forThisTAndInit_used]),r'$\beta$',  np.concatenate([stdMCTIBetas_forThisTAndInit_used, pathMCTIs_forThisTAndInit_used]),'U',
                                                       'Best data for TI\nand integration curve', np.asarray(np.concatenate([np.full(len(stdMCBetas_forThisTAndInit_used), 0), trajsExtremesInitID[pathsMC_filtForThisTAndInit_used]])), trajInitShortDescription_Dict, edgeColorPerInitType_Dic,
                                                       np.concatenate([np.full(len(stdMCBetas_forThisTAndInit_used), "inf"), T[pathsMC_filtForThisTAndInit_used]]), ["T"], np.concatenate([np.full(len(stdMCTIBetas_forThisTAndInit_used), -1), refConfMutualQ[pathsMC_filtForThisTAndInit_used]]),
                                                       additionalMarkerTypes_Unused=[[stMC_beta[stdMC_filtForThisTAndInit_unused], stMC_TIbeta[stdMC_filtForThisTAndInit_unused], [stMC_betaOfExtraction[stdMC_filtForThisTAndInit_unused], np.full(len(stMC_configurationIndex[stdMC_filtForThisTAndInit_unused]),sim_Qif)], np.full(len(stMC_beta[stdMC_filtForThisTAndInit_unused]),f"inf")]],
                                                       functionsToPlotContinuously=f, linesAtXValueAndName=vLines)
                            f=None
                            if Zfunction is not None:
                                f=[[Zfunction], [None]]
                                plotWithDifferentColorbars(f'T{sim_T}_{trajInitShortDescription_Dict[sim_trajInit]}_Z',
                                                       np.concatenate([stdMCBetas_forThisTAndInit_used, pathMCBetas_forThisTAndInit_used]),r'$\beta$',  Zfunction(np.concatenate([stdMCBetas_forThisTAndInit_used, pathMCBetas_forThisTAndInit_used])),'Z',
                                                       'Z from best data for TI\nand corresponding points', np.asarray(np.concatenate([np.full(len(stdMCBetas_forThisTAndInit_used), 0), trajsExtremesInitID[pathsMC_filtForThisTAndInit_used]])), trajInitShortDescription_Dict, edgeColorPerInitType_Dic,
                                                       np.concatenate([np.full(len(stdMCBetas_forThisTAndInit_used), "inf"), T[pathsMC_filtForThisTAndInit_used]]), ["T"], np.concatenate([np.full(len(stdMCTIBetas_forThisTAndInit_used), -1), refConfMutualQ[pathsMC_filtForThisTAndInit_used]]),
                                                       functionsToPlotContinuously=f, linesAtXValueAndName=vLines)

                                plotWithDifferentColorbars(f'T{sim_T}_{trajInitShortDescription_Dict[sim_trajInit]}_Zlog',
                                                       np.concatenate([stdMCBetas_forThisTAndInit_used, pathMCBetas_forThisTAndInit_used]),r'$\beta$',  Zfunction(np.concatenate([stdMCBetas_forThisTAndInit_used, pathMCBetas_forThisTAndInit_used])),'Z',
                                                       'Z from best data for TI\nand corresponding points', np.asarray(np.concatenate([np.full(len(stdMCBetas_forThisTAndInit_used), 0), trajsExtremesInitID[pathsMC_filtForThisTAndInit_used]])), trajInitShortDescription_Dict, edgeColorPerInitType_Dic,
                                                       np.concatenate([np.full(len(stdMCBetas_forThisTAndInit_used), "inf"), T[pathsMC_filtForThisTAndInit_used]]), ["T"], np.concatenate([np.full(len(stdMCTIBetas_forThisTAndInit_used), -1), refConfMutualQ[pathsMC_filtForThisTAndInit_used]]),
                                                       functionsToPlotContinuously=f, yscale='log', linesAtXValueAndName=vLines)

                            figs = plt.get_figlabels()  # Ottieni i nomi di tutte le figure create
                            for fig_name in figs:
                                fig = plt.figure(fig_name)
                                filename = os.path.join(TIPlotsFolder, f'{fig_name}.png')
                                print(filename)
                                fig.savefig(filename, dpi=300, bbox_inches='tight')
                            plt.close('all')    

                            
                            # ---- Collect TI curve and points ----
                            try:
                                _model_type = "unknown"
                                try:
                                    _model_type = str(model_type[pathsMC_filtForThisTAndInit_used][np.where(pathsMC_filtForThisTAndInit_used)[0][0]])
                                except Exception:
                                    _model_type = "unknown"
                                TIcurve_id = make_TIcurve_id(_model_type, sim_N, sim_graphID, sim_fieldType, sim_fieldSigma,
                                                             sim_fieldRealization, sim_Hext, sim_Hout, sim_Hin, sim_Qstar,
                                                             sim_T, sim_trajInit, sim_betOfEx, sim_firstConfIndex, sim_secondConfIndex)
                                # knots
                                Z_knots_beta = np.concatenate([stdMCBetas_forThisTAndInit_used, pathMCBetas_forThisTAndInit_used]).tolist()
                                Z_knots = (Zfunction(np.concatenate([stdMCBetas_forThisTAndInit_used, pathMCBetas_forThisTAndInit_used]))).tolist()
                                TI_U_knots_beta = TIx.tolist()
                                TI_U_knots = TIy.tolist()
                                n_paths_used = int(np.sum(pathsMC_filtForThisTAndInit_used))
                                n_stdmc_used = int(np.sum(stdMC_filtForThisTAndInit_used))
                                curves_rows.append(dict(
                                    TIcurve_id=TIcurve_id, model_type=_model_type,
                                    N=int(sim_N), graphID=str(sim_graphID), fieldType=str(sim_fieldType),
                                    fieldSigma=float(sim_fieldSigma) if not pd.isna(sim_fieldSigma) else np.nan,
                                    fieldRealization=str(sim_fieldRealization),
                                    Hext=float(sim_Hext), Hout=float(sim_Hout), Hin=float(sim_Hin), Qstar=float(sim_Qstar),
                                    T=float(sim_T), trajInit=str(sim_trajInit),
                                    betaOfExtraction=str(sim_betOfEx), firstConfigurationIndex=str(sim_firstConfIndex),
                                    secondConfigurationIndex=str(sim_secondConfIndex),
                                    betaM=float(betaMax) if isinstance(betaMax,(int,float)) else np.nan,
                                    betaG=float(betaG) if isinstance(betaG,(int,float)) else np.nan,
                                    betaG2=float(betaG2) if isinstance(betaG2,(int,float)) else np.nan,
                                    betaG3=float(betaG3) if isinstance(betaG3,(int,float)) else np.nan,
                                    betaG2b=float(betaG2b) if isinstance(betaG2b,(int,float)) else np.nan,
                                    betaG2c=float(betaG2c) if isinstance(betaG2c,(int,float)) else np.nan,
                                    betaL=float(betaL) if isinstance(betaL,(int,float)) else np.nan,
                                    Zmax=float(Zfunction(betaMax)) if isinstance(betaMax,(int,float)) else np.nan,
                                    Z_knots_beta=Z_knots_beta, Z_knots=Z_knots,
                                    TI_U_knots_beta=TI_U_knots_beta, TI_U_knots=TI_U_knots,
                                    n_paths_used=n_paths_used, n_stdmc_used=n_stdmc_used,
                                    computed_at=_dt.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                                    analysis_rev="unversioned"
                                ))
                            except Exception as _e:
                                print("[warn] failed collecting TIcurve rows:", _e)
                            if Zfunction is None:
                                continue
                            mask = pathsMC_filtForThisTAndInit_used
                            with np.errstate(divide='ignore', invalid='ignore'):
                                ZFromTIBeta[mask] = Zfunction(beta[mask])
                                kFromChi[mask] = ZFromTIBeta[mask] * chi_m[mask]
                                kFromChi_InBetween[mask] = ZFromTIBeta[mask] * chi_m2[mask]
                                kFromChi_InBetween_Scaled[mask] = kFromChi_InBetween[mask] / scale2[mask]
                                minusLnKFromChi[mask] = -np.log(kFromChi[mask])
                                minusLnKFromChi_2[mask] = -np.log(kFromChi_InBetween[mask])
                                minusLnKFromChi_2_scaled[mask] = -np.log(kFromChi_InBetween_Scaled[mask])
                                tentativeBarrier[mask] = -np.log(kFromChi[mask]) / N[mask]
                                used_idx = np.where(pathsMC_filtForThisTAndInit_used)[0]
                                for i_idx in used_idx:
                                    points_rows.append(dict(
                                        TIcurve_id=TIcurve_id,
                                        run_uid=str(run_uid[i_idx]) if i_idx < len(run_uid) else None,
                                        beta=float(beta[i_idx]),
                                        ZFromTIBeta=float(ZFromTIBeta[i_idx]) if not np.isnan(ZFromTIBeta[i_idx]) else np.nan,
                                        kFromChi=float(kFromChi[i_idx]) if not np.isnan(kFromChi[i_idx]) else np.nan,
                                        kFromChi_InBetween=float(kFromChi_InBetween[i_idx]) if not np.isnan(kFromChi_InBetween[i_idx]) else np.nan,
                                        kFromChi_InBetween_Scaled=float(kFromChi_InBetween_Scaled[i_idx]) if not np.isnan(kFromChi_InBetween_Scaled[i_idx]) else np.nan,
                                        minusLnKFromChi=float(minusLnKFromChi[i_idx]) if not np.isnan(minusLnKFromChi[i_idx]) else np.nan,
                                        minusLnKFromChi_2=float(minusLnKFromChi_2[i_idx]) if not np.isnan(minusLnKFromChi_2[i_idx]) else np.nan,
                                        minusLnKFromChi_2_scaled=float(minusLnKFromChi_2_scaled[i_idx]) if not np.isnan(minusLnKFromChi_2_scaled[i_idx]) else np.nan,
                                        tentativeBarrier=float(tentativeBarrier[i_idx]) if not np.isnan(tentativeBarrier[i_idx]) else np.nan,
                                        tentativeBarrier_2=float(tentativeBarrier_2[i_idx]) if not np.isnan(tentativeBarrier_2[i_idx]) else np.nan,
                                        tentativeBarrier_3=float(tentativeBarrier_3[i_idx]) if not np.isnan(tentativeBarrier_3[i_idx]) else np.nan,
                                        rescaledBetas_M=float(rescaledBetas_M[i_idx]) if not np.isnan(rescaledBetas_M[i_idx]) else np.nan,
                                        rescaledBetas_G=float(rescaledBetas_G[i_idx]) if not np.isnan(rescaledBetas_G[i_idx]) else np.nan,
                                        rescaledBetas_G2=float(rescaledBetas_G2[i_idx]) if not np.isnan(rescaledBetas_G2[i_idx]) else np.nan,
                                        rescaledBetas_G3=float(rescaledBetas_G3[i_idx]) if not np.isnan(rescaledBetas_G3[i_idx]) else np.nan,
                                        rescaledBetas_G2b=float(rescaledBetas_G2b[i_idx]) if not np.isnan(rescaledBetas_G2b[i_idx]) else np.nan,
                                        rescaledBetas_G2c=float(rescaledBetas_G2c[i_idx]) if not np.isnan(rescaledBetas_G2c[i_idx]) else np.nan,
                                        T=float(T[i_idx]) if not np.isnan(T[i_idx]) else np.nan,
                                        trajInit=str(trajsExtremesInitID[i_idx]),
                                        computed_at=_dt.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                                        analysis_rev="unversioned"
                                    ))
                                tentativeBarrier_2[mask] = -np.log(kFromChi_InBetween[mask]) / N[mask]
                                tentativeBarrier_3[mask] = -np.log(kFromChi_InBetween_Scaled[mask]) / N[mask]
                            TDFirstConfIndex.append(sim_firstConfIndex)
                            TDSecondConfIndex.append(sim_secondConfIndex)
                            TDGraphId.append(sim_graphID)
                            TDFieldType.append(sim_fieldType)
                            TDFieldReali.append(sim_fieldRealization)
                            TDFieldSigma.append(sim_fieldSigma)
                            TDHext.append(sim_Hext)
                            TDHout.append(sim_Hout)
                            TDHin.append(sim_Hin)
                            TDnQstar.append(sim_nQstar)
                            TDBetaM.append(betaMax)

                            TDBetaG.append(betaG)
                            TDBetaG2.append(betaG2)
                            TDBetaG3.append(betaG3)
                            TDbetaG2b.append(betaG2b)
                            TDbetaG2c.append(betaG2c)
                            TDBetaL.append(betaL)
                            TDZmax.append(Zfunction(betaMax))

                            thisTransitionData_T.append(sim_T)
                            thisTransitionData_trajInit.append(sim_trajInit)
                            thisTransitionData_betaMax.append(betaMax)
                            thisTransitionData_betaL.append(betaLForThisRealization)
                            thisTransitionData_betaG.append(betaGForThisRealization)
                            thisTransitionData_betaG2.append(betaG2ForThisRealization)
                            thisTransitionData_betaG3.append(betaG3ForThisRealization)
                            thisTransitionData_betaG2b.append(betaG2bForThisRealization)
                            thisTransitionData_betaG2c.append(betaG2cForThisRealization)

                        thisTransitionData_T=np.array(thisTransitionData_T)
                        thisTransitionData_trajInit=np.array(thisTransitionData_trajInit)
                        thisTransitionData_betaMax=np.array(thisTransitionData_betaMax)
                        thisTransitionData_betaL=np.array(thisTransitionData_betaL)
                        thisTransitionData_betaG=np.array(thisTransitionData_betaG)
                        thisTransitionData_betaG2=np.array(thisTransitionData_betaG2)
                        thisTransitionData_betaG3=np.array(thisTransitionData_betaG3)
                        thisTransitionData_betaG2b=np.array(thisTransitionData_betaG2b)
                        thisTransitionData_betaG2c=np.array(thisTransitionData_betaG2c)

                        integratedData_Filt = np.logical_and(TIFilt4,~np.isnan(ZFromTIBeta))

                        T_vals = T[integratedData_Filt]
                        if T_vals.size == 0:
                            continue

                        best_T = np.nanmax(T_vals)
                        mask_Tmax = np.logical_and(integratedData_Filt, T == best_T)
                        best_trajInitId = None

                        selected_mask = np.zeros_like(cleanFilt, dtype=bool)
                        for tid in preferred_trajInit:
                            candidate_mask = np.logical_and.reduce([mask_Tmax, trajsExtremesInitID == tid])
                            if np.any(candidate_mask):
                                best_trajInitId=tid
                                selected_mask = candidate_mask
                                break

                        if not np.any(selected_mask):
                            continue


                        #Questo voglio spostarlo dentro l'analisi come filtro aggiuntivo dove levo dai fit dati scartati attrav. una funzione PreliminaryFilt
                        if sim_N>30:
                            indices = np.nonzero(selected_mask)[0]
                            for idx in indices:
                                cleanFilt[idx] = True

                        bestIntegrationData_Filt = np.logical_and(thisTransitionData_T==best_T,thisTransitionData_trajInit==best_trajInitId)
                        if np.sum(bestIntegrationData_Filt)!=1:
                            print(f"Error, found {np.sum(bestIntegrationData_Filt)} when trying to extract best information.")


                        realizationsId.append(thisTransitionData_betaMax[bestIntegrationData_Filt][0])
                        betaMaxOverRealizations.append(thisTransitionData_betaMax[bestIntegrationData_Filt][0])
                        betaLOverRealizations.append(thisTransitionData_betaL[bestIntegrationData_Filt][0])
                        betaGOverRealizations.append(thisTransitionData_betaG[bestIntegrationData_Filt][0])
                        betaG2OverRealizations.append(thisTransitionData_betaG2[bestIntegrationData_Filt][0])
                        betaG2bOverRealizations.append(thisTransitionData_betaG2b[bestIntegrationData_Filt][0])
                        betaG2cOverRealizations.append(thisTransitionData_betaG2c[bestIntegrationData_Filt][0])
                        betaG3OverRealizations.append(thisTransitionData_betaG3[bestIntegrationData_Filt][0])


                    realizationsId=np.array(realizationsId)
                    betaMaxOverRealizations = np.array(betaMaxOverRealizations, dtype=float)
                    betaLOverRealizations = np.array(betaLOverRealizations, dtype=float)
                    betaGOverRealizations = np.array(betaGOverRealizations, dtype=float)
                    betaG2OverRealizations = np.array(betaG2OverRealizations, dtype=float)
                    betaG2bOverRealizations = np.array(betaG2bOverRealizations, dtype=float)
                    betaG2cOverRealizations = np.array(betaG2cOverRealizations, dtype=float)
                    betaG3OverRealizations = np.array(betaG3OverRealizations, dtype=float)

                    uniqueValues, unique_indices = np.unique(realizationsId, axis=0, return_index=True)
                    if len(realizationsId)>0:
                        betaMaxOverRealizationsV = np.nanmean(betaMaxOverRealizations[unique_indices])
                        betaLOverRealizationsV = np.nanmean(betaLOverRealizations[unique_indices])
                        betaGOverRealizationsV = np.nanmean(betaGOverRealizations[unique_indices])
                        betaG2OverRealizationsV = np.nanmean(betaG2OverRealizations[unique_indices])
                        betaG3OverRealizationsV = np.nanmean(betaG3OverRealizations[unique_indices])
                        betaG2bOverRealizationsV = np.nanmean(betaG2bOverRealizations[unique_indices])
                        betaG2cOverRealizationsV = np.nanmean(betaG2cOverRealizations[unique_indices])

                    for sim_N, sim_graphID, sim_fieldRealization in set(zip(N[TIFilt3], graphID[TIFilt3], fieldRealization[TIFilt3])):
                        TIFilt4 = np.logical_and(TIFilt3, np.logical_and.reduce([N==sim_N, graphID==sim_graphID, fieldRealization==sim_fieldRealization]))
                        for sim_T, sim_trajInit in set(zip(T[TIFilt4],trajsExtremesInitID[TIFilt4])):
                            TIFilt_forThisTAndInit = np.logical_and.reduce([TIFilt4, T==sim_T, trajsExtremesInitID==sim_trajInit])
                            level = getLevelFromNestedStructureAndKeyName(Zdict, [(sim_N, sim_graphID, sim_Hext),
                                                                                  (sim_fieldType, sim_fieldRealization, sim_fieldSigma ),
                                                                                  (sim_betOfEx, sim_firstConfIndex, sim_secondConfIndex),
                                                                                  (sim_Hin, sim_Hout, sim_nQstar), (sim_T, sim_trajInit)],
                                                                                    'Zfunction')
                            if level is not None:
                                originalZfunction = level['Zfunction']

                                thisCurveBetaMax = level['betaMax']
                                thisCurveBetaL = level['beta_l']
                                thisCurveBetaG = level['beta_g']
                                thisCurveBetaG2 = level['beta_g2']
                                thisCurveBetaG3 = level['beta_g3']
                                thisCurvebetaG2b = level['beta_g2b']
                                thisCurvebetaG2c = level['beta_g2c']

                                if isinstance(thisCurveBetaMax, (int, float)):
                                    #print(betaGOverRealizations, betaMaxOverRealizations, thisCurveBetaMax, thisCurveBetaG)
                                    rescaledBetas_M[TIFilt_forThisTAndInit] = (beta[TIFilt_forThisTAndInit])/(thisCurveBetaMax)*betaMaxOverRealizationsV
                                    #minusLnKFromChi[TIFilt_forThisTAndInit] *= thisCurveBetaMax/betaMaxOverRealizations
                                    #minusLnKFromChi_2[TIFilt_forThisTAndInit] *= thisCurveBetaMax/betaMaxOverRealizations
                                    #minusLnKFromChi_2_scaled[TIFilt_forThisTAndInit] *= thisCurveBetaMax/betaMaxOverRealizations
                                    def rescaledZfunction_M(bet, numBet=betaMaxOverRealizations, denBet=thisCurveBetaMax, function=originalZfunction):
                                        return function(bet*denBet/numBet)
                                    level['rescaledZfunction_Max']=rescaledZfunction_M

                                if isinstance(thisCurveBetaL, (int, float)):
                                    rescaledBetas_L[TIFilt_forThisTAndInit] = beta[TIFilt_forThisTAndInit]/thisCurveBetaL*betaLOverRealizationsV
                                    def rescaledZfunction_L(bet, numBet=1., denBet=thisCurveBetaL, function=originalZfunction):
                                        return function(bet*denBet/numBet)
                                    level['rescaledZfunction_l']=rescaledZfunction_L

                                if isinstance(thisCurveBetaG, (int, float)):
                                    rescaledBetas_G[TIFilt_forThisTAndInit] = beta[TIFilt_forThisTAndInit]/thisCurveBetaG*betaGOverRealizationsV
                                    def rescaledZfunction_G(bet, numBet=1., denBet=thisCurveBetaG, function=originalZfunction):
                                        return function(bet*denBet/numBet)
                                    level['rescaledZfunction_g']=rescaledZfunction_G
                                if isinstance(thisCurveBetaG2, (int, float)):
                                    rescaledBetas_G2[TIFilt_forThisTAndInit] = beta[TIFilt_forThisTAndInit]/thisCurveBetaG2*betaG2OverRealizationsV
                                    def rescaledZfunction_G2(bet, numBet=1., denBet=thisCurveBetaG2, function=originalZfunction):
                                        return function(bet*denBet/numBet)
                                    level['rescaledZfunction_g2']=rescaledZfunction_G2
                                if isinstance(thisCurveBetaG3, (int, float)):
                                    rescaledBetas_G3[TIFilt_forThisTAndInit] = beta[TIFilt_forThisTAndInit]/thisCurveBetaG3*betaG3OverRealizationsV
                                    def rescaledZfunction_G3(bet, numBet=1., denBet=thisCurveBetaG3, function=originalZfunction):
                                        return function(bet*denBet/numBet)
                                    level['rescaledZfunction_g3']=rescaledZfunction_G3

                                if isinstance(thisCurvebetaG2b, (int, float)):
                                    rescaledBetas_G2b[TIFilt_forThisTAndInit] = beta[TIFilt_forThisTAndInit]/thisCurvebetaG2b*betaG2bOverRealizationsV
                                    def rescaledZfunction_G(bet, numBet=1., denBet=thisCurvebetaG2b, function=originalZfunction):
                                        return function(bet*denBet/numBet)
                                    level['rescaledZfunction_g']=rescaledZfunction_G

                                if isinstance(thisCurvebetaG2c, (int, float)):
                                    rescaledBetas_G2c[TIFilt_forThisTAndInit] = beta[TIFilt_forThisTAndInit]/thisCurvebetaG2c*betaG2cOverRealizationsV
                                    def rescaledZfunction_G(bet, numBet=1., denBet=thisCurvebetaG2c, function=originalZfunction):
                                        return function(bet*denBet/numBet)
                                    level['rescaledZfunction_g']=rescaledZfunction_G
                            else:
                                print("level non trovato per ",(sim_N, sim_graphID, sim_Hext), (sim_fieldType, sim_fieldSigma, sim_fieldRealization), (sim_betOfEx, sim_firstConfIndex, sim_secondConfIndex), (sim_Hin, sim_Hout, sim_nQstar), (sim_T, sim_trajInit))

                for sim_Hin, sim_Hout, sim_nQstar in set(zip(h_in[TIFilt2], h_out[TIFilt2], normalizedQstar[TIFilt2])):
                    TIFilt3 = np.logical_and(TIFilt2, np.logical_and.reduce([h_in==sim_Hin, h_out==sim_Hout, normalizedQstar==sim_nQstar]))
                    __vals = rescaledBetas_M[TIFilt3]
                    __vals = __vals[np.isfinite(__vals)] if hasattr(__vals, 'shape') else np.array([])
                    max_value = np.nanmax(__vals) if getattr(__vals, 'size', 0) > 0 else np.nan
                    if np.isnan(max_value):
                        continue
                    for discRescBeta in np.round(np.arange(0., max_value+discBetaStep, discBetaStep, dtype=float),decimals=4):
                        discretizableFilt=np.logical_and(TIFilt3, np.fabs(discRescBeta-rescaledBetas_M)<discBetaStep/2.+0.00001)
                        if len(np.unique(N[discretizableFilt]))>=3:
                            for g,ft,fr,fs, t in set(zip(graphID[discretizableFilt], fieldType[discretizableFilt],fieldRealization[discretizableFilt],fieldSigma[discretizableFilt], T[discretizableFilt])):
                                discretizableFilt2=np.logical_and.reduce([
                                    discretizableFilt, fieldType==ft,fieldRealization==fr, fieldSigma==fs, graphID==g, T==t])
                                diff = np.fabs(rescaledBetas_M[discretizableFilt2]-discRescBeta)
                                closestBet = np.nanmin(diff)
                                diff = np.fabs(rescaledBetas_M-discRescBeta)
                                discretizableFilt2=np.logical_and.reduce([discretizableFilt2, diff==closestBet])

                                discretizedRescaledBetas_M[discretizableFilt2] =discRescBeta

                for sim_Hin, sim_Hout, sim_nQstar in set(zip(h_in[TIFilt2], h_out[TIFilt2], normalizedQstar[TIFilt2])):
                    TIFilt3 = np.logical_and(TIFilt2, np.logical_and.reduce([h_in==sim_Hin, h_out==sim_Hout, normalizedQstar==sim_nQstar]))
                    __vals = rescaledBetas_G[TIFilt3]
                    __vals = __vals[np.isfinite(__vals)] if hasattr(__vals, 'shape') else np.array([])
                    max_value = np.nanmax(__vals) if getattr(__vals, 'size', 0) > 0 else np.nan
                    __vals = rescaledBetas_G2[TIFilt3]
                    __vals = __vals[np.isfinite(__vals)] if hasattr(__vals, 'shape') else np.array([])
                    max_value2 = np.nanmax(__vals) if getattr(__vals, 'size', 0) > 0 else np.nan
                    __vals = rescaledBetas_G3[TIFilt3]
                    __vals = __vals[np.isfinite(__vals)] if hasattr(__vals, 'shape') else np.array([])
                    max_value3 = np.nanmax(__vals) if getattr(__vals, 'size', 0) > 0 else np.nan
                    if np.isnan(max_value):
                        continue
                    for discRescBeta in np.round(np.arange(0., max_value+discBetaStep, discBetaStep, dtype=float),decimals=4):
                        discretizableFilt=np.logical_and(TIFilt3, np.fabs(discRescBeta-rescaledBetas_G)<discBetaStep/2.+0.000000000000001)
                        if len(np.unique(N[discretizableFilt]))>=3:
                            for g,ft,fr,fs in set(zip(graphID[discretizableFilt], fieldType[discretizableFilt],fieldRealization[discretizableFilt],fieldSigma[discretizableFilt])):
                                discretizableFiltb=np.logical_and.reduce([
                                    discretizableFilt, fieldType==ft,fieldRealization==fr, fieldSigma==fs, graphID==g])
                                diff = np.fabs(rescaledBetas_G[discretizableFiltb]-discRescBeta)
                                closestBet = np.nanmin(diff)
                                diff = np.fabs(rescaledBetas_G-discRescBeta)
                                discretizableFiltC=np.logical_and.reduce([discretizableFiltb, diff==closestBet])
                                discretizedRescaledBetas_G[discretizableFiltC] =discRescBeta

                        discretizableFilt2=np.logical_and(TIFilt3, np.fabs(discRescBeta-rescaledBetas_G2)<discBetaStep/2.+0.000000000000001)
                        if len(np.unique(N[discretizableFilt2]))>=3:
                            for g,ft,fr,fs in set(zip(graphID[discretizableFilt2], fieldType[discretizableFilt2],fieldRealization[discretizableFilt2],fieldSigma[discretizableFilt2])):
                                discretizableFiltb2=np.logical_and.reduce([
                                    discretizableFilt2, fieldType==ft,fieldRealization==fr, fieldSigma==fs, graphID==g])
                                diff = np.fabs(rescaledBetas_G2[discretizableFiltb2]-discRescBeta)
                                closestBet = np.nanmin(diff)
                                diff = np.fabs(rescaledBetas_G2-discRescBeta)
                                discretizableFiltC2=np.logical_and.reduce([discretizableFiltb2, diff==closestBet])
                                discretizedRescaledBetas_G2[discretizableFiltC2] =discRescBeta

                        discretizableFilt3=np.logical_and(TIFilt3, np.fabs(discRescBeta-rescaledBetas_G3)<discBetaStep/2.+0.000000000000001)
                        if len(np.unique(N[discretizableFilt3]))>=3:
                            for g,ft,fr,fs in set(zip(graphID[discretizableFilt3], fieldType[discretizableFilt3],fieldRealization[discretizableFilt3],fieldSigma[discretizableFilt3])):
                                discretizableFiltb3=np.logical_and.reduce([
                                    discretizableFilt3, fieldType==ft,fieldRealization==fr, fieldSigma==fs, graphID==g])
                                diff = np.fabs(rescaledBetas_G3[discretizableFiltb3]-discRescBeta)
                                closestBet = np.nanmin(diff)
                                diff = np.fabs(rescaledBetas_G3-discRescBeta)
                                discretizableFiltC3=np.logical_and.reduce([discretizableFiltb3, diff==closestBet])
                                discretizedRescaledBetas_G3[discretizableFiltC3] =discRescBeta

                        discretizableFilt3=np.logical_and(TIFilt3, np.fabs(discRescBeta-rescaledBetas_G2b)<discBetaStep/2.+0.000000000000001)
                        if len(np.unique(N[discretizableFilt3]))>=3:
                            for g,ft,fr,fs in set(zip(graphID[discretizableFilt3], fieldType[discretizableFilt3],fieldRealization[discretizableFilt3],fieldSigma[discretizableFilt3])):
                                discretizableFiltb3=np.logical_and.reduce([
                                    discretizableFilt3, fieldType==ft,fieldRealization==fr, fieldSigma==fs, graphID==g])
                                diff = np.fabs(rescaledBetas_G2b[discretizableFiltb3]-discRescBeta)
                                closestBet = np.nanmin(diff)
                                diff = np.fabs(rescaledBetas_G2b-discRescBeta)
                                discretizableFiltC3=np.logical_and.reduce([discretizableFiltb3, diff==closestBet])
                                discretizedRescaledBetas_G2b[discretizableFiltC3] =discRescBeta

                        discretizableFilt3=np.logical_and(TIFilt3, np.fabs(discRescBeta-rescaledBetas_G2c)<discBetaStep/2.+0.000000000000001)
                        if len(np.unique(N[discretizableFilt3]))>=3:
                            for g,ft,fr,fs in set(zip(graphID[discretizableFilt3], fieldType[discretizableFilt3],fieldRealization[discretizableFilt3],fieldSigma[discretizableFilt3])):
                                discretizableFiltb3=np.logical_and.reduce([
                                    discretizableFilt3, fieldType==ft,fieldRealization==fr, fieldSigma==fs, graphID==g])
                                diff = np.fabs(rescaledBetas_G2c[discretizableFiltb3]-discRescBeta)
                                closestBet = np.nanmin(diff)
                                diff = np.fabs(rescaledBetas_G2c-discRescBeta)
                                discretizableFiltC3=np.logical_and.reduce([discretizableFiltb3, diff==closestBet])
                                discretizedRescaledBetas_G2c[discretizableFiltC3] =discRescBeta

    TDN=np.array(TDN)
    TDTrajInit=np.array(TDTrajInit)
    TDT=np.array(TDT)
    TDBetOfEx=np.array(TDBetOfEx)
    TDFirstConfIndex=np.array(TDFirstConfIndex)
    TDSecondConfIndex=np.array(TDSecondConfIndex)
    TDGraphId=np.array(TDGraphId)
    TDFieldType=np.array(TDFieldType)
    TDFieldReali=np.array(TDFieldReali)
    TDFieldSigma=np.array(TDFieldSigma)
    TDHext=np.array(TDHext)
    TDHout=np.array(TDHout)
    TDHin=np.array(TDHin)
    TDnQstar=np.array(TDnQstar,dtype=np.float64)
    TDBetaM=np.array(TDBetaM,dtype=np.float64)
    TDBetaG=np.array(TDBetaG,dtype=np.float64)
    TDBetaG2=np.array(TDBetaG2,dtype=np.float64)
    TDBetaG3=np.array(TDBetaG3,dtype=np.float64)
    TDbetaG2b=np.array(TDbetaG2b,dtype=np.float64)
    TDbetaG2c=np.array(TDbetaG2c,dtype=np.float64)
    TDBetaL=np.array(TDBetaL,dtype=np.float64)
    TDZmax=np.array(TDZmax,dtype=np.float64)
    globals()['curves_rows'] = curves_rows
    globals()['points_rows'] = points_rows


    return (TDN, TDTrajInit, TDT, TDBetOfEx, TDFirstConfIndex, TDSecondConfIndex, TDGraphId,
            TDFieldType, TDFieldReali, TDFieldSigma, TDHext,
            TDHout, TDHin, TDnQstar, TDBetaM, TDBetaG, TDBetaG2,TDBetaG3, TDbetaG2b,TDbetaG2c, TDBetaL, TDZmax, cleanFilt)   

# ---------- Writer & diagnostica ----------
def write_ti(outdir: Path, model: str, data: dict, verbose: bool=False) -> Path:
    base = outdir / model / "v1" / "ti"
    base.mkdir(parents=True, exist_ok=True)
    cols = [
        "N","TrajInit","T","betaOfExtraction","firstConfIndex","secondConfIndex","graphID",
        "fieldType","fieldRealization","fieldSigma","Hext","Hout","Hin","Qstar",
        "betaM","betaG","betaG2","betaG3","betaG2b","betaG2c","betaL","Zmax"
    ]
    lens = [len(v) for k,v in data.items() if hasattr(v, "__len__")]
    n = max(lens) if lens else 0
    payload = {}
    for c in cols:
        v = data.get(c, None)
        if v is None:
            payload[c] = [np.nan]*n
        else:
            arr = np.atleast_1d(v)
            payload[c] = (arr if len(arr)==n else list(arr) + [np.nan]*(n-len(arr)))
    df = pd.DataFrame(payload)
    out = base / "ti_curves.parquet"
    df.to_parquet(out, index=False)
    if verbose:
        print("[write] {} rows={}".format(out, len(df)))
    return out

def _unique_combos_report():
    if N is None:
        return {"rows": 0, "groups": 0, "top": []}
    n = len(N)
    C_ = C if C is not None else np.full(n, np.nan)
    H_ = h_ext if h_ext is not None else np.full(n, 0.0)
    FT = fieldType if fieldType is not None else np.full(n, "", dtype=object)
    FS = fieldSigma if fieldSigma is not None else np.full(n, 0.0)
    keys = [(float(C_[i]) if C_[i] is not None else np.nan,
             float(H_[i]) if H_[i] is not None else 0.0,
             str(FT[i]),
             float(FS[i]) if FS[i] is not None else 0.0) for i in range(n)]
    from collections import Counter
    cnt = Counter(keys)
    top = cnt.most_common(10)
    return {"rows": n, "groups": len(cnt), "top": top}

def _write_manifest(manifest_path: Path, stats: dict, ti_rows: int, elapsed_s: float, out_parquet):
    lines = []
    lines.append("# thermodynamicIntegration run manifest\n")
    lines.append("- rows loaded: {}\n".format(stats.get("rows", 0)))
    lines.append("- unique groups (C,Hext,fieldType,fieldSigma): {}\n".format(stats.get("groups", 0)))
    lines.append("- top groups (count):\n")
    for (Cval, HextVal, ftype, fsig), c in stats.get("top", []):
        lines.append("  - C={} Hext={} fieldType='{}' fieldSigma={} -> {}\n".format(Cval, HextVal, ftype, fsig, c))
    lines.append("- TI produced rows: {}\n".format(ti_rows))
    lines.append("- elapsed seconds: {:.3f}\n".format(elapsed_s))
    (
        (
        lines.append("- output parquet: {}\n".format(out_parquet))
    ) if not isinstance(out_parquet, (list, tuple)) else (
        lines.append(f"- output parquet (curves): {out_parquet[0]}\n"),
        lines.append(f"- output parquet (points): {out_parquet[1]}\n")
    )
    ) if not isinstance(out_parquet, (list, tuple)) else (
        lines.append(f"- output parquet (curves): {out_parquet[0]}\n"),
        lines.append(f"- output parquet (points): {out_parquet[1]}\n")
    )
    manifest_path.write_text("".join(lines), encoding="utf-8")

# ---------- CLI ----------
def parse_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Esegue la thermodynamicIntegration partendo dalle tabelle Parquet (con diagnostica)." ,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--graphs-root", type=Path, default=None, help="Root Data/Graphs (auto)." )
    p.add_argument("--outdir", type=Path, default=None, help="Root MultiPathsMC (auto)." )
    p.add_argument("--model", type=str, required=True, help="Modello (ER, RRG, realGraphs/ZKC…)." )
    p.add_argument("--include", nargs="*", default=[], help="Filtri substring su runPath." )
    p.add_argument("--analysis-path", type=Path, default=None,
                   help="Cartella per output/plot TI (default: Data/MultiRun/<model>)." )
    p.add_argument("--verbose","-v", action="store_true")
    return p

def main():
    ns = parse_args().parse_args()
    graphs_root = discover_graphs_root(ns.graphs_root)
    outdir = default_outdir_for(graphs_root) if ns.outdir is None else ns.outdir.resolve()
    analysis_path = (graphs_root.parent / "MultiRun" / ns.model) if ns.analysis_path is None else ns.analysis_path

    print("[roots] graphs_root={}".format(graphs_root))
    print("[roots] outdir     ={}".format(outdir))
    print("[paths] analysis   ={}".format(analysis_path))

    try:
        nrows = load_tables_as_arrays(ns.model, graphs_root, outdir, ns.include, verbose=ns.verbose)
    except FileNotFoundError as e:
        print("\n[ERROR] {}".format(e), file=sys.stderr)
        print("\nSuggerimento: controlla l'argomento --model. Esempi validi che ho trovato:", file=sys.stderr)
        for m in _list_available_models(outdir):
            print("  - {}".format(m), file=sys.stderr)
        sys.exit(2)

    sanitize_globals(verbose=ns.verbose)

    filt = np.ones(nrows, dtype=bool)

    t0 = time.perf_counter()
    (TDN, TDTrajInit, TDT, TDBetOfEx, TDFirstConfIndex, TDSecondConfIndex, TDGraphId,
     TDFieldType, TDFieldReali, TDFieldSigma, TDHext,
     TDHout, TDHin, TDnQstar, TDBetaM, TDBetaG, TDBetaG2, TDBetaG3, TDbetaG2b, TDbetaG2c, TDBetaL, TDZmax,
     PathsMCsUsedForTI_cleanFilt) = thermodynamicIntegration(filt, str(analysis_path))
    elapsed = time.perf_counter() - t0

    payload = dict(
        N=TDN, TrajInit=TDTrajInit, T=TDT, betaOfExtraction=TDBetOfEx,
        firstConfIndex=TDFirstConfIndex, secondConfIndex=TDSecondConfIndex, graphID=TDGraphId,
        fieldType=TDFieldType, fieldRealization=TDFieldReali, fieldSigma=TDFieldSigma,
        Hext=TDHext, Hout=TDHout, Hin=TDHin, Qstar=TDnQstar,
        betaM=TDBetaM, betaG=TDBetaG, betaG2=TDBetaG2, betaG3=TDBetaG3,
        betaG2b=TDbetaG2b, betaG2c=TDbetaG2c, betaL=TDBetaL, Zmax=TDZmax
    )
    out_curves, out_points = write_ti_curves_points(outdir, ns.model, curves_rows, points_rows, verbose=ns.verbose)

    stats = _unique_combos_report()
    manifest_dir = (outdir / ns.model / "v1" / "ti"); manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_dir / "ti_manifest.md"
    _write_manifest(manifest, stats, len(curves_rows), elapsed, (out_curves, out_points))

    print("[done] ti: curves={} points={}".format(out_curves, out_points))
    if ns.verbose:
        print("[diag] groups={} rows={} elapsed_s={}".format(stats.get("groups"), stats.get("rows"), round(elapsed,3)))
        print("[diag] manifest: {}".format(manifest))

if __name__ == "__main__":
    main()