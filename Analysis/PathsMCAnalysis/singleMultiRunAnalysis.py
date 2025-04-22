import os
import sys

from scipy import interpolate
sys.path.append('../')
import glob
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from itertools import cycle
import json
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from scipy.integrate import quad
from scipy.special import comb
from scipy.optimize import minimize_scalar

from MyBasePlots.plotWithDifferentColorbars import plotWithDifferentColorbars
minNumberOfSingleRunsToDoAnAnalysis=4
discBetaStep=0.025
nNsToConsiderForSubFit=6 #For extracting free energy barriers, there is also a fit
                                #involving only this number of largest Ns

analyBet= None
analyBarr= None
analyBetBarr= None
                        
fieldTypeDictionary ={"2":"gauss", "1":"bernoulli", "nan":"noField"}
fieldTypePathDictionary ={"gauss":"stdGaussian", "bernoulli":"stdBernoulli"}
trajInitShortDescription_Dict= {0: "stdMC", 70: "Random", 71: "Ref 12", 72: "Ref 21", 73: "Annealing", 74: "Annealing", 740: "AnnealingF", -2:"Fit"}
edgeColorPerInitType_Dic={0: "None", 70: "lightGreen", 71: "black", 72: "purple", 73: "orange", 74: "orange", 740: "red", -2:"black"}
preferred_trajInit = [740, 74, 73, 72, 71, 70]

typeOfFitInN_Dict= {0: r"fixed $\beta$", 1: r"fixed $\beta_{r}$"}
edgeColorPerTypeOfFitInIn_Dic={0: "black", 1: "red"}

def ensure_directories_exist(path):
    try:
        os.makedirs(path, exist_ok=True)
        print(f"Directory '{path}' creata con successo o già esistente.")
    except Exception as e:
        print(f"Errore nella creazione della directory '{path}': {e}")

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

def getUniqueXAndYZAccordingToZ(x, y, criterion, preliminaryFilter=None):
    if preliminaryFilter is None:
        preliminaryFilter =  ~np.isnan(y)
    else:
        preliminaryFilter = np.logical_and(~np.isnan(y), preliminaryFilter)
    best_indices = {}
    filterToReturn = np.full(len(x), False)
    # Iteriamo su ogni valore unico in filteredStdMCsBetas
    for value in np.sort(np.unique(x[preliminaryFilter])):
        # Trova gli indici corrispondenti a questo valore unico
        indices = np.where(np.logical_and(x == value, preliminaryFilter))[0]

        if len(indices) > 1:
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

def analyticDataF(x, xName, filt,fieldSigma,beta):

    if (analyBet is None) or np.any(fieldSigma[filt]!=0.) or (("beta" not in xName) and len(np.unique(beta[filt]))>1):

        return None, None, None
    arrayToConsider=beta
    if ("beta" in xName):

        arrayToConsider=x
    betaRange= np.nanmin(arrayToConsider[filt]), np.nanmax(arrayToConsider[filt]) 
    betaRangeCondition = np.logical_and(analyBet>=betaRange[0],analyBet<=betaRange[1]) 
    betaRangeCondition = np.logical_and(betaRangeCondition, analyBarr>0) 
    if np.all(betaRangeCondition==False):

        return None, None, None
    theoreticalX=analyBet[betaRangeCondition]
    theoreticalY0=analyBarr[betaRangeCondition]
    theoreticalY1=analyBetBarr[betaRangeCondition]

    if ("beta" not in xName):

        theoreticalX = np.linspace(np.nanmin(x[filt]), np.nanmax(x[filt]),100)
        theoreticalY0 = np.full(100, theoreticalY0[0])
        theoreticalY1 = np.full(100, theoreticalY1[0])
    return theoreticalX, theoreticalY0, theoreticalY1

def analyticDataF2(x, xType, fieldSigma,filt):
    global analyBet, analyBarr, analyBetBarr

    fieldSigma = np.array(fieldSigma,dtype=np.float64)
    if len(x)==0 or np.any(fieldSigma[filt]!=0.):
        return None, None, None
    x=np.array(x,dtype=np.float64)
    if (analyBet is None) or (("beta" != xType)):

        return None, None, None
    arrayToConsider=x
    betaRange= np.nanmin(arrayToConsider), np.nanmax(arrayToConsider) 
    betaRangeCondition = np.logical_and(analyBet>=betaRange[0]-0.1,analyBet<=betaRange[1]) 
    betaRangeCondition = np.logical_and(betaRangeCondition, analyBarr>0) 
    if np.all(betaRangeCondition==False):
        return None, None, None
    theoreticalX=analyBet[betaRangeCondition]
    theoreticalY0=analyBarr[betaRangeCondition]
    theoreticalY1=analyBetBarr[betaRangeCondition]
    if ("beta"!= xType):
        theoreticalX = np.linspace(np.nanmin(arrayToConsider), np.nanmax(arrayToConsider),100)
        theoreticalY0 = np.full(100, theoreticalY0[0])
        theoreticalY1 = np.full(100, theoreticalY1[0])
    return theoreticalX, theoreticalY0, theoreticalY1

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

nameOfFoldersContainingGraphs = ["fPosJ","realGraphs"
                               ]

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


def singleMultiRunAnalysis(runsData, parentAnalysis_path, symType):

    plt.rcParams["axes.grid"]= True
    plt.rcParams['lines.marker'] = 'o'
    plt.rcParams['lines.linestyle'] = '-'
    markers = ['.', '^', 'o', 'v','p', 'h']

    ID = []
    simulationType =[]
    fPosJ = []
    C = []
    graphID = []

    refConfInitID =[]
    betaOfExtraction =[]
    firstConfigurationIndex = []
    secondConfigurationIndex = []
    refConfMutualQ =[]

    trajsExtremesInitID =[]
    trajsJumpsInitID =[]

    N = []
    beta = []
    T = []
    h_ext = []
    h_in = []
    h_out = []
    Qstar = []
    chi_tau = []
    chi_c = []
    chi_m = []
    chi_chi = []
    chi_tau2 = []
    chi_m2 = []
    chi_c2 = []
    chi_chi2 = []
    TIbeta = []
    TIhout = []
    TIQstar = []

    lastMeasureMC = []
    MCprint = []


    meanBarrier = []
    stdDevBarrier = []
    muAvEnergy = []

    avEnergy = []
    avEnergyStdErr = []
    nJumps = []
    nJumpsStdErr = []
    deltaNJumps = []
    deltaNJumpsStdErr = []
    qDist = []
    qDistStdErr = []
    realTime = []
    realTimeErr = []
    fieldType = []
    fieldMean = []
    fieldSigma = []
    fieldRealization = []

    stMC_N = []
    stMC_beta = []
    stMC_Hext = []
    stMC_Hout = []
    stMC_Qstar = []
    stMC_TIbeta= []
    stMC_MC= []
    stMC_graphID = []
    stMC_betaOfExtraction = []
    stMC_configurationIndex = []
    stMC_fieldType = []
    stMC_fieldMean = []
    stMC_fieldSigma = []
    stMC_fieldRealization = []

    for item in runsData:
        if "results" not in item.keys():
            continue

        if item['configuration']['simulationTypeId']==15:
                simulationType.append(15)
                n=(int)(item['configuration']['parameters']['N'])
                stMC_N.append(n)
                stMC_beta.append(item['configuration']['parameters']['beta'])
                stMC_Hext.append(item['configuration']['parameters']['hext'])
                stMC_Hout.append(item['configuration']['parameters']['h_out'])
                stMC_Qstar.append(item['configuration']['parameters']['Qstar'])
                stMC_graphID.append(item['configuration']['parameters']['graphID'])
                if item['configuration']['referenceConfigurationsInfo']['ID'] == 55:
                    stMC_betaOfExtraction.append(item['configuration']['referenceConfigurationsInfo']['betaOfExtraction'])
                    stMC_configurationIndex.append(item['configuration']['referenceConfigurationsInfo']['configurationIndex'])
                else:
                    stMC_betaOfExtraction.append("nan")
                    stMC_configurationIndex.append("nan")
                if item['configuration']['parameters']['ID'] == 220:
                    stMC_fieldType.append(item['configuration']['parameters']['fieldType'])
                    stMC_fieldMean.append(item['configuration']['parameters']['fieldMean'])
                    stMC_fieldSigma.append(item['configuration']['parameters']['fieldSigma'])
                    stMC_fieldRealization.append(item['configuration']['parameters']['fieldRealization'])
                else:
                    stMC_fieldType.append("nan")
                    stMC_fieldMean.append("nan")
                    stMC_fieldSigma.append("nan")
                    stMC_fieldRealization.append("nan")

                stMC_MC.append(item['configuration']['mcParameters']['MC'])
                stMC_TIbeta.append(item["results"]['TI']['beta'])
                continue

        #print("Extracting data from", item['configuration']['ID'])  #decommentare per controllare quando c'è un intoppo
        ID.append(item['configuration']['ID']) #ID only contains new IDs. will be used to check what analysis to repeat
        refConfInitID.append(item['configuration']['referenceConfigurationsInfo']['ID'])

        
        matplotlib.rcParams.update({
            # Font
            'font.size': 20,              # Dimensione testo generale
            'axes.titlesize': 22,         # Titolo asse
            'axes.labelsize': 20,         # Etichette assi
            'xtick.labelsize': 18,
            'ytick.labelsize': 18,
            'legend.fontsize': 18,
            'figure.titlesize': 24,

            # Linee
            'lines.linewidth': 2.2,         # Spessore linee principali
            'lines.markersize': 8,        # Dimensione marker

            # Assi
            'axes.linewidth': 2,          # Spessore contorno assi
            'xtick.major.width': 2,       # Spessore ticks
            'ytick.major.width': 2,
            'xtick.major.size': 7,        # Lunghezza ticks
            'ytick.major.size': 7,
            
            # Griglia (opzionale)
            'axes.grid': True,           # Imposta a True se vuoi griglia leggera
            'grid.linewidth': 1,
            'grid.alpha': 0.5,

            # Layout
            'figure.dpi': 300,            # Risoluzione per export PNG
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',      # Rimuove margini inutili
            'pdf.fonttype': 42,           # Testo selezionabile nei PDF
            'ps.fonttype': 42,
            
            # Font family (opzionale)
            'font.family': 'sans-serif',
        })


        #trajsJumpsInitID.append(item['configuration']['trajs_jumpsInitialization']['ID'])

        n=(int)(item['configuration']['parameters']['N'])
        N.append(n)
        C.append(item['configuration']['parameters']['C']) 
        fPosJ.append(item['configuration']['parameters']['fPosJ']) 
        graphID.append(item['configuration']['parameters']['graphID'])
        
        h_ext.append(item['configuration']['parameters']['hext'])
        beta.append(item['configuration']['parameters']['beta'])
        T.append(item['configuration']['parameters']['T'])
        
        h_in.append(item['configuration']['parameters']['h_in'])
        h_out.append(item['configuration']['parameters']['h_out'])
        Qstar.append(item['configuration']['parameters']['Qstar'])
        
        if item['configuration']['parameters']['ID']==210:
            fieldType.append(item['configuration']['parameters']['fieldType'])
            fieldMean.append(item['configuration']['parameters']['fieldMean'])
            fieldSigma.append(item['configuration']['parameters']['fieldSigma'])
            fieldRealization.append(item['configuration']['parameters']['fieldRealization'])
        else:
            fieldType.append("nan")
            fieldMean.append("nan")
            fieldSigma.append("nan")
            fieldRealization.append("nan")


        if item['configuration']['referenceConfigurationsInfo']['ID']==50 or item['configuration']['referenceConfigurationsInfo']['ID']==56:
            betaOfExtraction.append("nan")
            firstConfigurationIndex.append("nan")
            secondConfigurationIndex.append("nan")
            refConfMutualQ.append(-n)
        else:
            betaOfExtraction.append(item['configuration']['referenceConfigurationsInfo']['betaOfExtraction'])
            firstConfigurationIndex.append(item['configuration']['referenceConfigurationsInfo']['firstConfigurationIndex'])
            secondConfigurationIndex.append(item['configuration']['referenceConfigurationsInfo']['secondConfigurationIndex'])
            refConfMutualQ.append(item['configuration']['referenceConfigurationsInfo']['mutualOverlap'])

        trajsExtremesInitID.append(item['configuration']['trajs_Initialization']['ID'])

        MCprint.append(item['configuration']['mcParameters']['MCprint'])
        lastMeasureMC.append(item['lastMeasureMC'])
        
        if 'chiLinearFit' not in item["results"].keys():
            chi_tau.append("nan")
            chi_m.append("nan")
            chi_c.append("nan")
            chi_chi.append("nan")
        else:
            chi_tau.append(item["results"]['chiLinearFit']['tau'])
            chi_m.append(item["results"]['chiLinearFit']['m'])
            chi_c.append(item["results"]['chiLinearFit']['c'])
            #chi_chi.append(item["results"]["chiLinearFit"]['Chi'])

        if "chiLinearFit_InBetween" not in item["results"].keys():
            chi_tau2.append("nan")
            chi_m2.append("nan")
            chi_c2.append("nan")
            chi_chi2.append("nan")
        else:
            chi_tau2.append(item["results"]["chiLinearFit_InBetween"]['tau'])
            chi_m2.append(item["results"]["chiLinearFit_InBetween"]['m'])
            chi_c2.append(item["results"]["chiLinearFit_InBetween"]['c'])
            chi_chi2.append(item["results"]["chiLinearFit_InBetween"]['Chi'])
        
        realTime.append(item["results"]["realTime"]['mean'])
        realTimeErr.append(item["results"]["realTime"]['sigma'])
        meanBarrier.append(item["results"]["meanBarrier"])
        stdDevBarrier.append(item["results"]["stdDevBarrier"])
        muAvEnergy.append(item["results"]["thermalization"]["avEnergy"]["mu"])
        avEnergy.append(item["results"]["thermalization"]["avEnergy"]["mean"])
        avEnergyStdErr.append(item["results"]["thermalization"]["avEnergy"]["stdErr"])
        nJumps.append(item["results"]["thermalization"]["nJumps"]["mean"])
        nJumpsStdErr.append(item["results"]["thermalization"]["nJumps"]["stdErr"])
        deltaNJumps.append(item["results"]["thermalization"]["deltaNJumps"]["mean"])
        deltaNJumpsStdErr.append(item["results"]["thermalization"]["deltaNJumps"]["stdErr"])
        qDist.append(item["results"]["thermalization"]["qDist"]["mean"])
        qDistStdErr.append(item["results"]["thermalization"]["qDist"]["stdErr"])
        TIbeta.append(item["results"]['TI']['beta'])
        TIhout.append(item["results"]['TI']['hout'])
        TIQstar.append(item["results"]['TI']['Qstar'])



    for typeOfSim in np.unique(simulationType):
        # Print the keyword, parameters, and their values
        print(f"Found {np.count_nonzero(simulationType == typeOfSim)} groups of type {typeOfSim}.\n")

    ID = np.array(ID)
    simulationType = np.array(simulationType)
    graphID = np.array(graphID)

    N = np.array(N, dtype=np.int16)
    T = np.array(T, dtype=np.float64)
    beta = np.array(beta, dtype=np.float64)
    h_in = np.array(h_in, dtype=np.float64)
    h_out = np.array(h_out, dtype=np.float64)
    C = np.array(C, dtype=np.int16)
    fPosJ = np.array(fPosJ, dtype=np.float64)
    Qstar= np.array(Qstar, dtype=np.int16)
    
    for i in range(len(Qstar)):
        n=N[i]
        q=Qstar[i]
        if n % 20 == 0:
            if (q-1)%(n/20)==0:
                Qstar[i]=q-1
        elif n % 10 == 0:
            if (q-1)%(n/10)==0:
                Qstar[i]=q-1
        elif n % 50 == 0:
            if (q-1)%(n/50)==0:
                Qstar[i]=q-1
    
            
    normalizedQstar=Qstar/N
    h_ext = np.array(h_ext, dtype=np.float64)

    
    fieldType =  [fieldTypeDictionary[value] for value in fieldType]
    fieldType = np.array(fieldType)
    fieldMean = [float(value) if (value != "infty" and value!="nan") else ( 0.) for value in fieldMean]
    fieldMean = np.array(fieldMean, dtype=np.float64)
    fieldSigma = [float(value) if (value != "infty" and value!="nan") else (0.) for value in fieldSigma]
    fieldSigma = np.array(fieldSigma, dtype=np.float64)
    fieldRealization = [value if (value != "infty" and value!="nan") else (0) for value in fieldRealization]
    fieldRealization = np.array(fieldRealization, dtype=int)
    
    lastMeasureMC = np.array(lastMeasureMC, dtype=np.int64)
    MCprint = np.array(MCprint, dtype=np.int64)

    nMeasures = lastMeasureMC//MCprint #not completely correcty

    refConfInitID = np.array(refConfInitID, dtype=np.int16)
    refConfMutualQ = np.array(refConfMutualQ, dtype=np.int16)
    betaOfExtraction = [float(value) if (value != "infty" and value!="nan") else ( np.inf if value!="nan" else np.nan)for value in betaOfExtraction]
    betaOfExtraction = np.array(betaOfExtraction, dtype=np.float64)
    firstConfigurationIndex = [int(value) if value != "nan" else np.nan for value in firstConfigurationIndex]
    firstConfigurationIndex = np.array(firstConfigurationIndex)
    secondConfigurationIndex = [int(value) if value != "nan" else np.nan for value in secondConfigurationIndex]
    secondConfigurationIndex = np.array(secondConfigurationIndex)
    betaOfExtraction = betaOfExtraction.astype(str)
    firstConfigurationIndex = firstConfigurationIndex.astype(str)
    secondConfigurationIndex = secondConfigurationIndex.astype(str)

    trajsExtremesInitID = np.array(trajsExtremesInitID, dtype=np.int16)
    trajsJumpsInitID = np.array(trajsJumpsInitID, dtype=np.int16)

    realTime = np.array(realTime, dtype=np.float64)
    realTimeErr = [float(value) if value != "sigma" else "nan" for value in realTimeErr]
    realTimeErr = np.array(realTimeErr, dtype=np.float64)
    realTimeErr/= nMeasures**0.5
    TIbeta = np.array(TIbeta, dtype=np.float64)
    TIhout = np.array(TIhout, dtype=np.float64)
    TIQstar = np.array(TIQstar, dtype=np.float64)

    chi_tau = np.array(chi_tau, dtype=np.float64)
    chi_m = np.array(chi_m, dtype=np.float64)
    chi_c = np.array(chi_c, dtype=np.float64)

    chi_tau2 = np.array(chi_tau2, dtype=np.float64)
    chi_m2 = np.array(chi_m2, dtype=np.float64)
    chi_c2 = np.array(chi_c2, dtype=np.float64)
    chi_chi2 = np.array(chi_chi2, dtype=np.float64)
    
    scale2 = chi_m2*T+chi_c2
    scale2[scale2<0.75] = np.nan
    scale2[chi_chi2>0.3] = np.nan
    
    ZFromTIBeta = np.full_like(N, np.nan, dtype=np.float64) # l'esistenza di questo array è una sconfitta
    rescaledBetas_M = np.full_like(N, np.nan, dtype=np.float64) # l'esistenza di questo array è una sconfitta
    rescaledBetas_L = np.full_like(N, np.nan, dtype=np.float64) # l'esistenza di questo array è una sconfitta
    rescaledBetas_G = np.full_like(N, np.nan, dtype=np.float64) # l'esistenza di questo array è una sconfitta
    rescaledBetas_G2 = np.full_like(N, np.nan, dtype=np.float64) # l'esistenza di questo array è una sconfitta
    rescaledBetas_G3 = np.full_like(N, np.nan, dtype=np.float64) # l'esistenza di questo array è una sconfitta
    rescaledBetas_G2b = np.full_like(N, np.nan, dtype=np.float64) # l'esistenza di questo array è una sconfitta
    rescaledBetas_G2c = np.full_like(N, np.nan, dtype=np.float64) # l'esistenza di questo array è una sconfitta
    minusLnKFromChi = np.full_like(N, np.nan, dtype=np.float64) # l'esistenza di questo array è una sconfitta
    minusLnKFromChi_2 = np.full_like(N, np.nan, dtype=np.float64) # l'esistenza di questo array è una sconfitta
    minusLnKFromChi_2_scaled = np.full_like(N, np.nan, dtype=np.float64) # l'esistenza di questo array è una sconfitta
    discretizedRescaledBetas_M = np.full_like(N, np.nan, dtype=np.float64) # l'esistenza di questo array è una sconfitta
    discretizedRescaledBetas_G = np.full_like(N, np.nan, dtype=np.float64) # l'esistenza di questo array è una sconfitta
    discretizedRescaledBetas_G2 = np.full_like(N, np.nan, dtype=np.float64) # l'esistenza di questo array è una sconfitta
    discretizedRescaledBetas_G3 = np.full_like(N, np.nan, dtype=np.float64) # l'esistenza di questo array è una sconfitta
    discretizedRescaledBetas_G2b = np.full_like(N, np.nan, dtype=np.float64) # l'esistenza di questo array è una sconfitta
    discretizedRescaledBetas_G2c = np.full_like(N, np.nan, dtype=np.float64) # l'esistenza di questo array è una sconfitta
    betaMax = np.full_like(N, np.nan, dtype=np.float64) # l'esistenza di questo array è una sconfitta
    averageBetaMax = np.full_like(N, np.nan, dtype=np.float64) # l'esistenza di questo array è una sconfitta
    kFromChi = np.full_like(N, np.nan, dtype=np.float64)
    kFromChi_InBetween = np.full_like(N, np.nan, dtype=np.float64)
    kFromChi_InBetween_Scaled = np.full_like(N, np.nan, dtype=np.float64)
    tentativeBarrier = np.full_like(N, np.nan, dtype=np.float64)
    tentativeBarrier_2 = np.full_like(N, np.nan, dtype=np.float64)
    tentativeBarrier_3 = np.full_like(N, np.nan, dtype=np.float64)
    
    meanBarrier = np.array(meanBarrier, dtype=np.float64)
    stdDevBarrier = np.array(stdDevBarrier, dtype=np.float64)
    
    avEnergy = np.array(avEnergy, dtype=np.float64)
    avEnergyStdErr = np.array(avEnergyStdErr, dtype=np.float64)
    
    avEnergy/=N
    avEnergyStdErr/=N
    muAvEnergy = np.array(muAvEnergy, dtype=np.float64)
    

    nJumps =  np.array(nJumps, dtype=np.float64)
    nJumpsStdErr =  np.array(nJumpsStdErr, dtype=np.float64)
    
    effectiveFlipRate = nJumps/T
    effectiveFlipRateError = nJumpsStdErr/T
    
    deltaNJumps =  np.array(deltaNJumps, dtype=np.float64)
    deltaNJumpsStdErr =  np.array(deltaNJumpsStdErr, dtype=np.float64)

    deltaN2JumpsOverNJumps = deltaNJumps**2/nJumps
    deltaN2JumpsOverNJumpsStdErr = deltaN2JumpsOverNJumps*np.sqrt((nJumpsStdErr/nJumps)**2.+(deltaNJumpsStdErr/deltaNJumps)**2.)
    
    qDist = np.array(qDist, dtype=np.float64)
    qDistStdErr = np.array(qDistStdErr, dtype=np.float64)
    
    stMC_N = np.array(stMC_N, dtype=np.int16)
    stMC_beta  = np.array(stMC_beta, dtype=np.float64)
    stMC_TIbeta  = np.array(stMC_TIbeta, dtype=np.float64)
    stMC_MC  = np.array(stMC_MC, dtype=np.int64)
    stMC_Hext  = np.array(stMC_Hext, dtype=np.float64)
    stMC_Hout = np.array(stMC_Hout, dtype=np.float64)
    stMC_Qstar = np.array(stMC_Qstar, dtype=np.int16)
    
    for i in range(len(stMC_N)):
        n=stMC_N[i]
        q=stMC_Qstar[i]
        if n % 20 == 0:
            if (q-1)%(n/20)==0:
                stMC_Qstar[i]=stMC_Qstar[i]-1
        elif n % 10 == 0:
            if (q-1)%(n/10)==0:
                stMC_Qstar[i]=stMC_Qstar[i]-1
        elif n % 50 == 0:
            if (q-1)%(n/50)==0:
                stMC_Qstar[i]=stMC_Qstar[i]-1
    stMC_nQstar=stMC_Qstar/stMC_N
    stMC_graphID  = np.array(stMC_graphID)
    stMC_betaOfExtraction = [float(value) if value != "infty" else np.inf for value in stMC_betaOfExtraction]
    stMC_betaOfExtraction = np.array(stMC_betaOfExtraction, dtype=str)
    stMC_configurationIndex  = np.array(stMC_configurationIndex)

    stMC_fieldType =  [fieldTypeDictionary[value] for value in stMC_fieldType]
    stMC_fieldType = np.array(stMC_fieldType)
    stMC_fieldMean = [float(value) if (value != "infty" and value!="nan") else ( 0)for value in stMC_fieldMean]
    stMC_fieldMean = np.array(stMC_fieldMean, dtype=np.float64)
    stMC_fieldSigma = [float(value) if (value != "infty" and value!="nan") else (0)for value in stMC_fieldSigma]
    stMC_fieldSigma = np.array(stMC_fieldSigma, dtype=np.float64)
    stMC_fieldRealization = [value if (value != "infty" and value!="nan") else (0)for value in stMC_fieldRealization]
    stMC_fieldRealization = np.array(stMC_fieldRealization, dtype=int)
    
    for g, t, r in set(zip(graphID, fieldType, fieldRealization)):
        if r==0:
            continue
        whereToFindFieldInfo= findFoldersWithString('../../Data/Graphs', [f'graph{g}'])
                                
        if len(whereToFindFieldInfo)>1:
            print("Errore, piu di un grafo trovato")
            print(whereToFindFieldInfo)
            return None
        whereToFindFieldInfo = whereToFindFieldInfo[0]
        whereToFindFieldInfo = os.path.join(whereToFindFieldInfo, 'randomFieldStructures',
        fieldTypePathDictionary[t])
        whereToFindFieldInfo = os.path.join(whereToFindFieldInfo, f'realization{r}',
                                            'graphFieldAnalysis','graphFieldData.json')
        thisFieldPathsMCs=np.logical_and.reduce([graphID==g,fieldType==t,fieldRealization==r])
        thisFieldStdMCs=np.logical_and.reduce([stMC_graphID==g, stMC_fieldType==t, stMC_fieldRealization==r])
        fieldActualMean=np.nan
        if os.path.exists(whereToFindFieldInfo):
            with open(whereToFindFieldInfo, 'r') as file:
                graphData = json.load(file)
                fieldActualMean=graphData['mean']
        else:
            print("Info su true mean di ", whereToFindFieldInfo," non disponibili.")
        
        if fieldActualMean<0.:
            fieldSigma[thisFieldPathsMCs]=-fieldSigma[thisFieldPathsMCs]
            stMC_fieldSigma[thisFieldStdMCs]=-stMC_fieldSigma[thisFieldStdMCs]
    
    fieldRealization[fieldSigma<0.]*=-1
    stMC_fieldRealization[stMC_fieldSigma<0.]*=-1
    refConfMutualQ = refConfMutualQ/N

    # Removed initialization here as it is already initialized within the function and returned
    Zdict = {}

    def thermodynamicIntegration(filt, analysis_path):
        
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
                                if len(stdMCsBetas_forThisTDSetting*(stdMCsBetas_forThisTDSetting<smallestPathsMcBetaToConsider))==0:
                                    continue
                                largestStdMcBetaToConsider_i = np.nanargmax(stdMCsBetas_forThisTDSetting*(stdMCsBetas_forThisTDSetting<smallestPathsMcBetaToConsider))
                                largestStdMcBetaToConsider = stdMCsBetas_forThisTDSetting[largestStdMcBetaToConsider_i]
                                largestStdMcTIToConsider = stdMCsTIBetas_forThisTDSetting[largestStdMcBetaToConsider_i]
                                #print("Z", smallestPathsMcBetaToConsider)
                                #print("A", largestStdMcBetaToConsider)
                                TIDifferenceMax=np.nanmax(stMC_TIbeta[stdMC_filtForThisTAndInit])-np.nanmin(pathMCTIs_forThisTAndInit)
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
                                    if abs((stdMcTIbeta-PathsMcTIToCompare)/TIDifferenceMax)<0:#0.003:
                                        largestStdMcBetaToConsider = stdMcBeta
                                        largestStdMcTIToConsider = stdMcTIbeta
                                        smallestLargerPathMCBeta_index=np.nanargmin(pathMCBetas_forThisTAndInit[pathMCBetas_forThisTAndInit>stdMcBeta])
                                        smallestPathsMcBetaToConsider = pathMCBetas_forThisTAndInit[pathMCBetas_forThisTAndInit>stdMcBeta][smallestLargerPathMCBeta_index]
                                        smallestPathsMcTIToConsider = pathMCTIs_forThisTAndInit[pathMCBetas_forThisTAndInit>stdMcBeta][smallestLargerPathMCBeta_index]

                                #print("B", largestStdMcBetaToConsider)
                                #print("C", smallestPathsMcBetaToConsider)
                                stdMC_filtForThisTAndInit_used = np.logical_and(stdMC_filtForThisTAndInit, stMC_beta <= largestStdMcBetaToConsider)
                                pathsMC_filtForThisTAndInit_used = np.logical_and(pathMCFilter_forThisTAndInit, beta >= smallestPathsMcBetaToConsider)
                                temp = np.sort(np.concatenate([stMC_beta[stdMC_filtForThisTAndInit_used], beta[pathsMC_filtForThisTAndInit_used]]))
                                maxBetaNotTooSpaced = np.nanmax([temp[i] for i in range(1, len(temp)) if temp[i] - temp[i-1] <= 0.1])
                                stdMC_filtForThisTAndInit_used = np.logical_and(stdMC_filtForThisTAndInit_used, stMC_beta <= maxBetaNotTooSpaced)
                                pathsMC_filtForThisTAndInit_used = np.logical_and(pathsMC_filtForThisTAndInit_used, beta <= maxBetaNotTooSpaced)
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
                                    f_interp = interpolate.InterpolatedUnivariateSpline(TIx, TIy, k=4)

                                    p_up_0 = (sim_N*(1.+sim_Qif))/(2.*sim_N)
                                    p_up_t = 0.5*(1.+(2.*p_up_0-1.)*np.exp(-2.*sim_T))

                                    ZAtBet0 =0.
                                    for this_q_star in range(sim_Qstar, sim_N+1, 2):
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
                                    if sim_fieldType != "noField":
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
                                                           additionalMarkerTypes_Unused=[[stMC_beta[stdMC_filtForThisTAndInit_unused], stMC_TIbeta[stdMC_filtForThisTAndInit_unused], np.full((len(stdMC_filtForThisTAndInit_unused), 2), ["nan", -1]), f"inf"]],
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
                                    #print(filename)
                                    fig.savefig(filename, dpi=300, bbox_inches='tight')
                                plt.close('all')    
                                
                                if Zfunction is None:
                                    continue
                                
                                indices = np.where(pathsMC_filtForThisTAndInit_used)[0]
                                for index in indices:
                                    ZFromTIBeta[index] = Zfunction(beta[index])
                                    kFromChi[index] = ZFromTIBeta[index] * chi_m[index]
                                    kFromChi_InBetween[index] = ZFromTIBeta[index] * chi_m2[index]
                                    kFromChi_InBetween_Scaled[index] = kFromChi_InBetween[index]/scale2[index]
                                    minusLnKFromChi[index]=-np.log(kFromChi[index] )
                                    minusLnKFromChi_2[index]=-np.log(kFromChi_InBetween[index] )
                                    minusLnKFromChi_2_scaled[index]=-np.log(kFromChi_InBetween_Scaled[index] )
                                    tentativeBarrier[index] = -np.log(kFromChi[index])/(N[index])
                                    tentativeBarrier_2[index] = -np.log(kFromChi_InBetween[index])/(N[index])
                                    tentativeBarrier_3[index] = -np.log(kFromChi_InBetween_Scaled[index])/(N[index])
                                
                                levelToAdd = {}
                                levelToAdd['TIfunction'] = TIfunction
                                levelToAdd['Zfunction'] = Zfunction
                                levelToAdd['betaMax'] = betaMax

                                    
                                levelToAdd['beta_l'] = betaL
                                levelToAdd['beta_g'] = betaG
                                levelToAdd['beta_g2'] = betaG2
                                levelToAdd['beta_g3'] = betaG3
                                levelToAdd['beta_g2b'] = betaG2b
                                levelToAdd['beta_g2c'] = betaG2c
                                
                                TIdata={}
                                TIdata['usedStMCsFilter'] = stdMC_filtForThisTAndInit_used
                                TIdata['usedPathsMCsFilter'] = pathsMC_filtForThisTAndInit_used
                                TIdata['unusedStMCsFilter'] = stdMC_filtForThisTAndInit_unused
                                TIdata['unusedPathsMCsFilter'] = pathsMC_filtForThisTAndInit_unused
                                levelToAdd['TIdata'] = TIdata
                                addedLevel = addLevelOnNestedDictionary(Zdict, [(sim_N, sim_graphID, sim_Hext), (sim_fieldType, sim_fieldRealization, sim_fieldSigma), (sim_betOfEx, sim_firstConfIndex, sim_secondConfIndex), (sim_Hin, sim_Hout, sim_nQstar), (sim_T, sim_trajInit)],
                                                levelToAdd)
                                                                
                                betaGForThisRealization = betaG    
                                betaG2ForThisRealization = betaG2
                                betaG3ForThisRealization = betaG3    
                                betaG2bForThisRealization = betaG2b    
                                betaG2cForThisRealization = betaG2c    
                                betaLForThisRealization = betaL
                                betaMaxForThisRealizationCounter+=1
                                betaMaxForThisRealization+=betaMax
                           
                                TDN.append(sim_N)
                                TDTrajInit.append(sim_trajInit)
                                TDT.append(sim_T)
                                TDBetOfEx.append(sim_betOfEx)
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
                                candidate_mask = np.logical_and(mask_Tmax, trajsExtremesInitID == tid)
                                if np.any(candidate_mask):
                                    best_trajInitId=tid
                                    selected_mask = candidate_mask
                                    break

                            if not np.any(selected_mask):
                                continue
                            indices = np.nonzero(selected_mask)[0]
                            for idx in indices:
                                cleanFilt[idx] = True

                            print(sim_N, sim_graphID, np.nansum(cleanFilt), np.unique(N[cleanFilt]))
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
                                        rescaledBetas_M[TIFilt_forThisTAndInit] = (beta[TIFilt_forThisTAndInit])-(thisCurveBetaMax)+betaMaxOverRealizationsV
                                        #minusLnKFromChi[TIFilt_forThisTAndInit] *= thisCurveBetaMax/betaMaxOverRealizations
                                        #minusLnKFromChi_2[TIFilt_forThisTAndInit] *= thisCurveBetaMax/betaMaxOverRealizations
                                        #minusLnKFromChi_2_scaled[TIFilt_forThisTAndInit] *= thisCurveBetaMax/betaMaxOverRealizations
                                        def rescaledZfunction_M(bet, numBet=betaMaxOverRealizations, denBet=thisCurveBetaMax, function=originalZfunction):
                                            return function(bet*denBet/numBet)
                                        level['rescaledZfunction_Max']=rescaledZfunction_M
                                        
                                    if isinstance(thisCurveBetaL, (int, float)):
                                        rescaledBetas_L[TIFilt_forThisTAndInit] = beta[TIFilt_forThisTAndInit]-thisCurveBetaL+betaLOverRealizationsV
                                        def rescaledZfunction_L(bet, numBet=1., denBet=thisCurveBetaL, function=originalZfunction):
                                            return function(bet*denBet/numBet)
                                        level['rescaledZfunction_l']=rescaledZfunction_L
                                    
                                    if isinstance(thisCurveBetaG, (int, float)):
                                        rescaledBetas_G[TIFilt_forThisTAndInit] = beta[TIFilt_forThisTAndInit]-thisCurveBetaG+betaGOverRealizationsV
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
                        max_value=np.nanmax(rescaledBetas_M[TIFilt3])
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
                        max_value=np.nanmax(rescaledBetas_G[TIFilt3])
                        max_value2=np.nanmax(rescaledBetas_G2[TIFilt3])
                        max_value3=np.nanmax(rescaledBetas_G3[TIFilt3])
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
        print(np.unique(N[cleanFilt]))
        print(trajsExtremesInitID)
        print(len(N[cleanFilt]))
                                
        
        print("A LUNGHEZZA DI", np.nansum(cleanFilt))

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
        print("[DBG] id fine TI:",  cleanFilt.__array_interface__['data'][0], "sum:", np.sum(cleanFilt))

        return (TDN, TDTrajInit, TDT, TDBetOfEx, TDFirstConfIndex, TDSecondConfIndex, TDGraphId,
                TDFieldType, TDFieldReali, TDFieldSigma, TDHext,
                TDHout, TDHin, TDnQstar, TDBetaM, TDBetaG, TDBetaG2,TDBetaG3, TDbetaG2b,TDbetaG2c, TDBetaL, TDZmax, cleanFilt)   
                            
                   
                   
    def myTDStudy( x, xName, subfolderingVariable, subfolderingVariableNames, markerShapeVariables, markerShapeVariablesNames, arrayForColorCoordinate, colorMapSpecifier=None):
        thisStudyFolder= os.path.join(analysis_path, "TDStudy")
        os.makedirs(thisStudyFolder, exist_ok=True)

        thisStudyFolder= os.path.join(thisStudyFolder, xName)
        if colorMapSpecifier is None:
            colorMapSpecifier = np.full(len(x), 1)
        if not os.path.exists(thisStudyFolder):
            os.makedirs(thisStudyFolder, exist_ok=True)
        else:
            delete_files_in_folder(thisStudyFolder)
        
        for v in np.unique(subfolderingVariable, axis=0):
            oldV=v
            
            print(v)
            subFolderingFilter = []
            if subfolderingVariable.ndim==1:
                subFolderingFilter = (subfolderingVariable==v)
                if np.isnan(v) or v=='nan':
                    continue
            else: 
                subFolderingFilter= np.all(subfolderingVariable == v, axis=1)
                if any(x=='nan' for x in v):
                    print(f"Salto il ciclo esterno a causa di {v}")
                    continue 
            filt = subFolderingFilter

            v_iter = iter(v)  # Trasforma v in un iteratore
            v = [[next(v_iter) for _ in sublist] for sublist in subfolderingVariableNames] if isinstance(subfolderingVariableNames[0], list) else [next(v_iter) for _ in subfolderingVariableNames]

            if len(np.unique(x[filt]))<3:
                continue
            
            theseFiguresFolder = os.path.join(
                thisStudyFolder,
                "/".join(
                    [
                        "{}_{}".format(
                            ''.join([name.replace('\\', '').replace('$', '').capitalize() for name in level]),
                            '_'.join([f"{float(item):.4g}" if isinstance(item, float) or (isinstance(item, str) and item.replace('.', '', 1).isdigit()) 
                                      else str(item) for item in v[nLevel]])
                            
                        )
                        for nLevel, level in enumerate(subfolderingVariableNames)
                    ]
                )
            )
            
                
            specificationLine = "at "+ '\n'.join(', '.join([f"{k}={float(item):.4g}" if isinstance(item, float) or (isinstance(item, str) and item.replace('.', '', 1).isdigit())
                                                            else str(item) for k, item in zip(subfolderingVariableNames[nLevel], v[nLevel])]) for nLevel in range(len(subfolderingVariableNames))).replace('star', '*')

            if not os.path.exists(theseFiguresFolder):
                os.makedirs(theseFiguresFolder, exist_ok=True)
            else:
                delete_files_in_folder(theseFiguresFolder)
            
            if xName=="N":
                mainPlot, _ = plotWithDifferentColorbars(f"betaG", x[filt], xName, TDBetaG[filt], r"$\beta_{G}$", r"$\beta_{G}$ vs "+ xName+"\n"+specificationLine,
                            TDTrajInit[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic, markerShapeVariables[filt], [markerShapeVariablesNames[0]],
                            np.full(np.sum(filt),1),
                            nGraphs=len(np.unique(TDGraphId[filt])))
                mainPlot, _ = plotWithDifferentColorbars(f"betaG2", x[filt], xName, TDBetaG2[filt], r"$\beta_{G}$", r"$\beta_{G}$ vs "+ xName+"\n"+specificationLine,
                            TDTrajInit[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic, markerShapeVariables[filt], [markerShapeVariablesNames[0]],
                            np.full(np.sum(filt),1),
                            nGraphs=len(np.unique(TDGraphId[filt])))
                mainPlot, _ = plotWithDifferentColorbars(f"betaG3", x[filt], xName, TDBetaG3[filt], r"$\beta_{G}$", r"$\beta_{G}$ vs "+ xName+"\n"+specificationLine,
                            TDTrajInit[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic, markerShapeVariables[filt], [markerShapeVariablesNames[0]],
                            np.full(np.sum(filt),1),
                            nGraphs=len(np.unique(TDGraphId[filt])))
                mainPlot, _ = plotWithDifferentColorbars(f"betaG2b", x[filt], xName, TDbetaG2b[filt], r"$\beta_{G}$", r"$\beta_{G}$ vs "+ xName+"\n"+specificationLine,
                            TDTrajInit[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic, markerShapeVariables[filt], [markerShapeVariablesNames[0]],
                            np.full(np.sum(filt),1),
                            nGraphs=len(np.unique(TDGraphId[filt])))
                mainPlot, _ = plotWithDifferentColorbars(f"betaG2c", x[filt], xName, TDbetaG2c[filt], r"$\beta_{G}$", r"$\beta_{G}$ vs "+ xName+"\n"+specificationLine,
                            TDTrajInit[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic, markerShapeVariables[filt], [markerShapeVariablesNames[0]],
                            np.full(np.sum(filt),1),
                            nGraphs=len(np.unique(TDGraphId[filt])))
                mainPlot, _ = plotWithDifferentColorbars(f"betaL", x[filt], xName, TDBetaL[filt], r"$\beta_{L}$", r"$\beta_{L}$ vs "+ xName+"\n"+specificationLine,
                            TDTrajInit[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic, markerShapeVariables[filt], [markerShapeVariablesNames[0]],
                            np.full(np.sum(filt),1),
                            nGraphs=len(np.unique(TDGraphId[filt])))
                mainPlot, _ = plotWithDifferentColorbars(f"betaMOvL", x[filt], xName, (TDBetaM/TDBetaL)[filt], r"$\beta_{M}/\beta_{L}$", r"$\beta_{M}/\beta_{L}$ vs "+ xName+"\n"+specificationLine,
                            TDTrajInit[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic, markerShapeVariables[filt], markerShapeVariablesNames,
                            arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                            nGraphs=len(np.unique(TDGraphId[filt])))
                mainPlot, _ = plotWithDifferentColorbars(f"betaMOvG", x[filt], xName, (TDBetaM/TDBetaG)[filt], r"$\beta_{M}/\beta_{G}$", r"$\beta_{M}/\beta_{G}$ vs "+ xName+"\n"+specificationLine,
                            TDTrajInit[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic, markerShapeVariables[filt], markerShapeVariablesNames,
                            arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                            nGraphs=len(np.unique(TDGraphId[filt])))
                mainPlot, _ = plotWithDifferentColorbars(f"betaMOvG2", x[filt], xName, (TDBetaM/TDBetaG2)[filt], r"$\beta_{M}/\beta_{G}$", r"$\beta_{M}/\beta_{G}$ vs "+ xName+"\n"+specificationLine,
                            TDTrajInit[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic, markerShapeVariables[filt], markerShapeVariablesNames,
                            arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                            nGraphs=len(np.unique(TDGraphId[filt])))
                mainPlot, _ = plotWithDifferentColorbars(f"betaMOvG3", x[filt], xName, (TDBetaM/TDBetaG3)[filt], r"$\beta_{M}/\beta_{G}$", r"$\beta_{M}/\beta_{G}$ vs "+ xName+"\n"+specificationLine,
                            TDTrajInit[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic, markerShapeVariables[filt], markerShapeVariablesNames,
                            arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                            nGraphs=len(np.unique(TDGraphId[filt])))
                mainPlot, _ = plotWithDifferentColorbars(f"betaMOvG2b", x[filt], xName, (TDBetaM/TDbetaG2b)[filt], r"$\beta_{M}/\beta_{G}$", r"$\beta_{M}/\beta_{G}$ vs "+ xName+"\n"+specificationLine,
                            TDTrajInit[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic, markerShapeVariables[filt], markerShapeVariablesNames,
                            arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                            nGraphs=len(np.unique(TDGraphId[filt])))
                mainPlot, _ = plotWithDifferentColorbars(f"betaMOvG2c", x[filt], xName, (TDBetaM/TDbetaG2c)[filt], r"$\beta_{M}/\beta_{G}$", r"$\beta_{M}/\beta_{G}$ vs "+ xName+"\n"+specificationLine,
                            TDTrajInit[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic, markerShapeVariables[filt], markerShapeVariablesNames,
                            arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                            nGraphs=len(np.unique(TDGraphId[filt])))
            else:
                mainPlot, _ = plotWithDifferentColorbars(f"betaMOvL", x[filt], xName, (TDBetaM/TDBetaL)[filt], r"$\beta_{M}/\beta_{L}$", r"$\beta_{M}/\beta_{L}$ vs "+ xName+"\n"+specificationLine,
                            TDTrajInit[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic, markerShapeVariables[filt], markerShapeVariablesNames,
                            arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                            nGraphs=len(np.unique(TDGraphId[filt])))
                mainPlot, _ = plotWithDifferentColorbars(f"betaMOvG", x[filt], xName, (TDBetaM/TDBetaG)[filt], r"$\beta_{M}/\beta_{G}$", r"$\beta_{M}/\beta_{G}$ vs "+ xName+"\n"+specificationLine,
                            TDTrajInit[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic, markerShapeVariables[filt], markerShapeVariablesNames,
                            arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                            nGraphs=len(np.unique(TDGraphId[filt])))
                mainPlot, _ = plotWithDifferentColorbars(f"betaMOvG2", x[filt], xName, (TDBetaM/TDBetaG2)[filt], r"$\beta_{M}/\beta_{G}$", r"$\beta_{M}/\beta_{G}$ vs "+ xName+"\n"+specificationLine,
                            TDTrajInit[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic, markerShapeVariables[filt], markerShapeVariablesNames,
                            arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                            nGraphs=len(np.unique(TDGraphId[filt])))
                mainPlot, _ = plotWithDifferentColorbars(f"betaMOvG3", x[filt], xName, (TDBetaM/TDBetaG3)[filt], r"$\beta_{M}/\beta_{G}$", r"$\beta_{M}/\beta_{G}$ vs "+ xName+"\n"+specificationLine,
                            TDTrajInit[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic, markerShapeVariables[filt], markerShapeVariablesNames,
                            arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                            nGraphs=len(np.unique(TDGraphId[filt])))
                mainPlot, _ = plotWithDifferentColorbars(f"betaMOvG2b", x[filt], xName, (TDBetaM/TDbetaG2b)[filt], r"$\beta_{M}/\beta_{G}$", r"$\beta_{M}/\beta_{G}$ vs "+ xName+"\n"+specificationLine,
                            TDTrajInit[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic, markerShapeVariables[filt], markerShapeVariablesNames,
                            arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                            nGraphs=len(np.unique(TDGraphId[filt])))
                mainPlot, _ = plotWithDifferentColorbars(f"betaMOvG2c", x[filt], xName, (TDBetaM/TDbetaG2c)[filt], r"$\beta_{M}/\beta_{G}$", r"$\beta_{M}/\beta_{G}$ vs "+ xName+"\n"+specificationLine,
                            TDTrajInit[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic, markerShapeVariables[filt], markerShapeVariablesNames,
                            arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                            nGraphs=len(np.unique(TDGraphId[filt])))
            mainPlot, _ = plotWithDifferentColorbars(f"betaM", x[filt], xName, TDBetaM[filt], r"$\beta_{M}$", r"$\beta_{M}$ vs " +xName+"\n"+specificationLine,
                        TDTrajInit[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic, markerShapeVariables[filt], markerShapeVariablesNames,
                        arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                        nGraphs=len(np.unique(TDGraphId[filt])))
            mainPlot, _ = plotWithDifferentColorbars(f"Zmax", x[filt], xName, TDZmax[filt], r"$Z_{max}$", r"$Z_{max}$ vs "+ xName+"\n"+specificationLine,
                        TDTrajInit[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic, markerShapeVariables[filt], markerShapeVariablesNames,
                        arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                        nGraphs=len(np.unique(TDGraphId[filt])))
            figs = plt.get_figlabels()  # Ottieni i nomi di tutte le figure create
            for fig_name in figs:
                fig = plt.figure(fig_name)
                filename = os.path.join(theseFiguresFolder, f'{fig_name}.png')
                print(filename)
                fig.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close('all')
                                                       
    def myMultiRunStudy(filter, studyName, x, xName, subfolderingVariable, subfolderingVariableNames,
                        markerShapeVariables, markerShapeVariablesNames,
                        arrayForColorCoordinate=refConfMutualQ,
                        colorMapSpecifier=betaOfExtraction, colorMapSpecifierName=r'$\beta_{extr}$'):

        print("C LUNGHEZZA DI", np.nansum(PathsMCsUsedForTI_cleanFilt))

        thisStudyFolder= os.path.join(analysis_path, studyName)
        nSubLists=len(subfolderingVariableNames)
        subListsLengths=np.array([len(subfolderingVariableNames[n]) for n in range(nSubLists)])

        if not os.path.exists(thisStudyFolder):
            os.makedirs(thisStudyFolder, exist_ok=True)
        else:
            delete_files_in_folder(thisStudyFolder)
            
        subFolderingValues=[]
        nLastSubfVal=[]
        #print(np.unique(subfolderingVariable[:,-subListsLengths[-1]:],axis=0))
        for unique_values in np.unique(subfolderingVariable[filter,:-subListsLengths[-1]],axis=0):
            if any([(x=='nan') for x in unique_values]):
                continue
            v = subfolderingVariable[np.all(subfolderingVariable[:, :-subListsLengths[-1]] == unique_values, axis=1), -subListsLengths[-1]:]
            v = v[~np.any(v == 'nan', axis=1)]
            v = np.unique(v,axis=0)
            if(len(v))>0:
                for a in v:
                    subFolderingValues.append([*unique_values, *a])
                nLastSubfVal.append(len(subFolderingValues))
            
        nLastSubfVal=np.array(nLastSubfVal,dtype=int)
        theseFiguresFolder=thisStudyFolder
        for i,v in enumerate(subFolderingValues):
            if (i==0) or (i in nLastSubfVal):
                eqAnBet =[]
                eqAnBetAn = []
                eqAnBetAnErr = []
                
                barrierBet1 =[]
                barrierBetBarr1 = []
                barrierBetBarrErr1 = []
                barrierBet2 =[]
                barrierBetBarr2 = []
                barrierBetBarrErr2 = []
                barrierBet3 =[]
                barrierBetBarr3 = []
                barrierBetBarrErr3 = []
                barrierBet1lN =[]
                barrierBetBarr1lN = []
                barrierBetBarrErr1lN = []
                barrierBet2lN =[]
                barrierBetBarr2lN = []
                barrierBetBarrErr2lN = []
                barrierBet3lN =[]
                barrierBetBarr3lN = []
                barrierBetBarrErr3lN = []
                
            def considerToPlotN(theseFiguresFolder):        
                folderToPlot=theseFiguresFolder[:theseFiguresFolder.rfind("/")]
                barrierBetaAEdgeColors= np.full(len(barrierBetBarrErr1),0)
                
                theoreticalX, theoreticalY0, theoreticalY1=analyticDataF2(barrierBet1, "beta",fieldSigma,filt)
                mainPlot, _ = plotWithDifferentColorbars(f"betTentativeBarrier_FromFit1", barrierBet1, r"$\beta$", barrierBetBarr1,  r"$\beta \delta$f", '',#"Tentative free energy barrier (1) "+ r"(-ln(k)/ N)"+", vs "+ xName +"\n",
                    barrierBetaAEdgeColors, typeOfFitInN_Dict, edgeColorPerTypeOfFitInIn_Dic,
                    np.full(len(barrierBet1), "fit in N"), ["from"],
                    np.full(len(barrierBet1), 1), yerr= barrierBetBarrErr1,
                    theoreticalX=theoreticalX, theoreticalY=theoreticalY1)
                
                
                barrierBetaAEdgeColors= np.full(len(barrierBetBarrErr2),0)                
                theoreticalX, theoreticalY0, theoreticalY1=analyticDataF2(barrierBet2, "beta",fieldSigma,filt)
                
                mainPlot, _ = plotWithDifferentColorbars(f"betTentativeBarrier_FromFit2", barrierBet2, r"$\beta$", barrierBetBarr2,  r"$\beta\delta$f", "Tentative free energy barrier (2) "+ r"(-ln(k)/ N)"+", vs "+ xName +"\n",
                    barrierBetaAEdgeColors, typeOfFitInN_Dict, edgeColorPerTypeOfFitInIn_Dic,
                    np.full(len(barrierBet2), "fit in N"), ["from"],
                    np.full(len(barrierBet2), 1), yerr= barrierBetBarrErr2,
                    theoreticalX=theoreticalX, theoreticalY=theoreticalY1)
                
                barrierBetaAEdgeColors= np.full(len(barrierBetBarrErr3),0)                
                theoreticalX, theoreticalY0, theoreticalY1=analyticDataF2(barrierBet3, "beta",fieldSigma,filt)
                
                mainPlot, _ = plotWithDifferentColorbars(f"betTentativeBarrier_FromFit3", barrierBet3, r"$\beta$", barrierBetBarr3,  r"$\beta\delta$f", "Tentative free energy barrier (3) "+ r"(-ln(k)/ N)"+", vs "+ xName +"\n",
                    barrierBetaAEdgeColors, typeOfFitInN_Dict, edgeColorPerTypeOfFitInIn_Dic,
                    np.full(len(barrierBet3), "fit in N"), ["from"],
                    np.full(len(barrierBet3), 1), yerr= barrierBetBarrErr3,
                    theoreticalX=theoreticalX, theoreticalY=theoreticalY1)
                
                barrierBetaAEdgeColors= np.full(len(barrierBetBarrErr1lN),0)                
                theoreticalX, theoreticalY0, theoreticalY1=analyticDataF2(barrierBet1lN, "beta",fieldSigma,filt)
                
                mainPlot, _ = plotWithDifferentColorbars(f"betTentativeBarrier_FromFit1lNs", barrierBet1lN, r"$\beta$", barrierBetBarr1lN,  r"$\beta\delta$f", "Tentative free energy barrier (1 largeNs) "+ r"(-ln(k)/ N)"+", vs "+ xName +"\n",
                    barrierBetaAEdgeColors, typeOfFitInN_Dict, edgeColorPerTypeOfFitInIn_Dic,
                    np.full(len(barrierBetBarrErr1lN), "fit in N"), ["from"],
                    np.full(len(barrierBetBarrErr1lN), 1), yerr= barrierBetBarrErr1lN,
                    theoreticalX=theoreticalX, theoreticalY=theoreticalY1)
                
                barrierBetaAEdgeColors= np.full(len(barrierBetBarrErr2lN),0)
                theoreticalX, theoreticalY0, theoreticalY1=analyticDataF2(barrierBet2lN, "beta",fieldSigma,filt)
                
                mainPlot, _ = plotWithDifferentColorbars(f"betTentativeBarrier_FromFit2lNs", barrierBet2lN, r"$\beta$", barrierBetBarr2lN,  r"$\beta\delta$f", "Tentative free energy barrier (2 largeNs) "+ r"(-ln(k)/ N)"+", vs "+ xName +"\n",
                    barrierBetaAEdgeColors, typeOfFitInN_Dict, edgeColorPerTypeOfFitInIn_Dic,
                    np.full(len(barrierBet2lN), "fit in N"), ["from"],
                    np.full(len(barrierBet2lN), 1), yerr= barrierBetBarrErr2lN,
                    theoreticalX=theoreticalX, theoreticalY=theoreticalY1)
                
                barrierBetaAEdgeColors= np.full(len(barrierBetBarrErr3lN),0)
                theoreticalX, theoreticalY0, theoreticalY1=analyticDataF2(barrierBet3lN, "beta",fieldSigma,filt)
                
                mainPlot, _ = plotWithDifferentColorbars(f"betTentativeBarrier_FromFit3lNs", barrierBet3lN, r"$\beta$", barrierBetBarr3lN,  r"$\beta\delta$f", "Tentative free energy barrier (3 largeNs) "+ r"(-ln(k)/ N)"+", vs "+ xName +"\n",
                    barrierBetaAEdgeColors, typeOfFitInN_Dict, edgeColorPerTypeOfFitInIn_Dic,
                    np.full(len(barrierBet3lN), "fit in N"), ["from"],
                    np.full(len(barrierBet3lN), 1), yerr= barrierBetBarrErr3lN,
                    theoreticalX=theoreticalX, theoreticalY=theoreticalY1)
                
                figs = plt.get_figlabels()  # Ottieni i nomi di tutte le figure create
                for fig_name in figs:
                    fig = plt.figure(fig_name)
                    filename = os.path.join(folderToPlot, f'{fig_name}.png')
                    print(filename)
                    fig.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close('all')

            def considerToPlotT(theseFiguresFolder):    
                folderToPlot=theseFiguresFolder[:theseFiguresFolder.rfind("/")]    
                barrierBetaAEdgeColors= np.full(len(eqAnBet),0)
                mainPlot, _ = plotWithDifferentColorbars(f"eqAn", eqAnBet, r"$\beta$", eqAnBetAn,"slope", "Fit for linear jump anisotropy"+", vs "+ xName +"\n",
                    barrierBetaAEdgeColors, typeOfFitInN_Dict, edgeColorPerTypeOfFitInIn_Dic,
                    np.full(len(eqAnBet), "fit in T"), ["from"],
                    np.full(len(eqAnBet), 1), yerr= eqAnBetAnErr)
                
                figs = plt.get_figlabels()  # Ottieni i nomi di tutte le figure create
                for fig_name in figs:
                    fig = plt.figure(fig_name)
                    filename = os.path.join(folderToPlot, f'{fig_name}.png')
                    print(filename)
                    fig.savefig(filename,dpi=300, bbox_inches='tight')
                plt.close('all')

            

            subFolderingFilter = []
            if subfolderingVariable.ndim==1:
                subFolderingFilter = (subfolderingVariable==v)
                if np.isnan(v) or v=='nan':
                    continue
            else: 
                subFolderingFilter= np.all(subfolderingVariable == v, axis=1)
                if any(x=='nan' for x in v):
                    print(f"Salto il ciclo esterno a causa di {v}")
                    continue
                
            filt = np.logical_and.reduce([filter,
                                        subFolderingFilter
                                        ])

            v_iter = iter(v)  # Trasforma v in un iteratore
            v = [[next(v_iter) for _ in sublist] for sublist in subfolderingVariableNames] if isinstance(subfolderingVariableNames[0], list) else [next(v_iter) for _ in subfolderingVariableNames]

            if len(np.unique(x[filt]))<3:
                if (xName=="N") and (i in (nLastSubfVal-1)):
                    considerToPlotN(theseFiguresFolder)
                elif (xName=="T") and (i in (nLastSubfVal-1)):     
                    considerToPlotT(theseFiguresFolder)
                continue
            
            theseFiguresFolder = os.path.join(
                thisStudyFolder,
                "/".join(
                    [
                        "{}_{}".format(
                            ''.join([name.replace('\\', '').replace('$', '').capitalize() for name in level]),
                            '_'.join([f"{float(item):.4g}" if isinstance(item, float) or (isinstance(item, str) and item.replace('.', '', 1).isdigit())
                                      else str(item) for item in v[nLevel]])
                            
                        )
                        for nLevel, level in enumerate(subfolderingVariableNames)
                    ]
                )
            )
            
                
            specificationLine = "at "+ '\n'.join(', '.join([f"{k}={float(item):.4g}" if isinstance(item, float) or (isinstance(item, str) and item.replace('.', '', 1).isdigit())
                                                            else str(item) for k, item in zip(subfolderingVariableNames[nLevel], v[nLevel])]) for nLevel in range(len(subfolderingVariableNames))).replace('star', '*')

            
            if not os.path.exists(theseFiguresFolder):
                os.makedirs(theseFiguresFolder, exist_ok=True)
            else:
                delete_files_in_folder(theseFiguresFolder)

            additional= []
            
            tempFilt=filt
            if r"$\beta$" in xName:  #da riscalare nel caso /beta_max
                stMC_corrBetaAndQif = np.empty((len(stMC_beta), 2), dtype=object)
                for sim_N, sim_graphID, sim_FieldType, sim_FieldRealization, sim_FieldSigma, sim_betOfEx, sim_SecConfInd, sim_Qif, simQstar in set(zip(N[filt],graphID[filt], fieldType[filt], fieldRealization[filt], fieldSigma[filt], betaOfExtraction[filt], secondConfigurationIndex[filt], refConfMutualQ[filt], Qstar[filt])):
                    stMCFilt= np.logical_and.reduce([stMC_N == sim_N,
                                                     stMC_graphID==sim_graphID,
                                                     stMC_betaOfExtraction==sim_betOfEx,
                                                     stMC_configurationIndex==sim_SecConfInd,
                                                     stMC_Qstar==simQstar,
                                                     stMC_fieldType==sim_FieldType,
                                                     stMC_fieldRealization==sim_FieldRealization,
                                                     stMC_fieldSigma==sim_FieldSigma,
                                                    ])
                    stMC_corrBetaAndQif[stMCFilt]=[sim_betOfEx, sim_Qif]
                    if stMCFilt.sum()>1:
                        additional.append(list([stMC_beta[stMCFilt], stMC_TIbeta[stMCFilt], stMC_corrBetaAndQif[stMCFilt], f"inf, {sim_graphID}"]))
                if len(additional)==0:
                    additional=None
                

            filt=tempFilt
            TIbetamainPlot, _ = plotWithDifferentColorbars(f"TIbeta", x[filt], xName, TIbeta[filt], "U", "Quantity for thermodynamic integration vs "+ xName +"\n"+specificationLine,
            trajsExtremesInitID[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic,
            markerShapeVariables[filt], markerShapeVariablesNames,
            arrayForColorCoordinate[filt], colorMapSpecifier=colorMapSpecifier[filt],
            nGraphs=len(np.unique(graphID[filt])), colorMapSpecifierName=colorMapSpecifierName, 
            additionalMarkerTypes=additional
            )

            mainPlot, _ = plotWithDifferentColorbars(f"lastMeasureMC", x[filt], xName, lastMeasureMC[filt], "lastMeasureMC", "lastMeasureMC vs "+ xName+"\n"+specificationLine,
                        trajsExtremesInitID[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic, markerShapeVariables[filt], markerShapeVariablesNames,
                        arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                        nGraphs=len(np.unique(graphID[filt])), colorMapSpecifierName=colorMapSpecifierName
                        )
            
            mainPlot, _ = plotWithDifferentColorbars(f"meanBarrier", x[filt], xName, meanBarrier[filt], "barrier", "mean barrier vs "+ xName+"\n"+specificationLine,
                        trajsExtremesInitID[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic, markerShapeVariables[filt], markerShapeVariablesNames,
                        arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                        yerr=stdDevBarrier[filt], nGraphs=len(np.unique(graphID[filt])), colorMapSpecifierName=colorMapSpecifierName
                        )
            
            mainPlot, _ = plotWithDifferentColorbars(f"avEnergy", x[filt], xName, avEnergy[filt], "energy", "Mean average energy over trajectory vs "+ xName +"\n"+specificationLine,
                        trajsExtremesInitID[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic, markerShapeVariables[filt], markerShapeVariablesNames,
                        arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                        yerr=avEnergyStdErr[filt], nGraphs=len(np.unique(graphID[filt])), colorMapSpecifierName=colorMapSpecifierName
                        )
            
            mainPlot, _ = plotWithDifferentColorbars(f"muAvEnergy", x[filt], xName, muAvEnergy[filt], r"$\mu$", r"$\mu$"+" vs "+ xName+"\n"+specificationLine,
                            trajsExtremesInitID[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic, markerShapeVariables[filt], markerShapeVariablesNames,
                             arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                            yerr=stdDevBarrier[filt], fitTypes='', xscale='', 
                            yscale='log', 
                            nGraphs=len(np.unique(graphID[filt])), colorMapSpecifierName=colorMapSpecifierName
                            )

            mainPlot, _ = plotWithDifferentColorbars(f"nJumps", x[filt], xName, nJumps[filt], "# jumps", "Mean number of jumps per spin over trajectory vs "+ xName +"\n"+specificationLine,
                            trajsExtremesInitID[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic, markerShapeVariables[filt], markerShapeVariablesNames,
                            arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                            yerr=nJumpsStdErr[filt], nGraphs=len(np.unique(graphID[filt])), colorMapSpecifierName=colorMapSpecifierName
                            )

            mainPlot, _ = plotWithDifferentColorbars(f"effFlipRate", x[filt], xName, effectiveFlipRate[filt], "r", "Effective flip rate over trajectory (#jumps/T) vs "+ xName +"\n"+specificationLine,
                    trajsExtremesInitID[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic, markerShapeVariables[filt], markerShapeVariablesNames,
                     arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                    yerr=effectiveFlipRateError[filt], nGraphs=len(np.unique(graphID[filt])), colorMapSpecifierName=colorMapSpecifierName
                    )

            mainPlot, _ = plotWithDifferentColorbars(f"deltaNJumps", x[filt], xName, deltaNJumps[filt], r"$\delta$", "Spins number of jumps over trajectory stdDev (over sites) vs "+ xName +"\n"+specificationLine,
                    trajsExtremesInitID[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic, markerShapeVariables[filt], markerShapeVariablesNames,
                     arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                    yerr=deltaNJumpsStdErr[filt], nGraphs=len(np.unique(graphID[filt])), colorMapSpecifierName=colorMapSpecifierName
                    )
            
            toFit = []
            if xName=="T":
                toFit=['linear']
            mainPlot, fitData  = plotWithDifferentColorbars(f"deltaNOverAvJumps", x[filt], xName, deltaN2JumpsOverNJumps[filt], "ratio", r"($\delta$"+"#jumps)^2/(#jumps)" +" vs "+ xName +"\n"+specificationLine,
                    trajsExtremesInitID[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic, markerShapeVariables[filt], markerShapeVariablesNames,
                     arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                    yerr=deltaN2JumpsOverNJumpsStdErr[filt], nGraphs=len(np.unique(graphID[filt])), colorMapSpecifierName=colorMapSpecifierName,
                    fitTypes=toFit)
            if fitData is not None:
                eqAnBet.append((float)(v[-1][0]))
                eqAnBetAn.append(fitData[0])
                eqAnBetAnErr.append(fitData[2])
                
            toFit = []
                
            mainPlot, _ = plotWithDifferentColorbars(f"deltaNOverAvJumpsOvT", x[filt], xName, deltaN2JumpsOverNJumps[filt]/T[filt], "ratio", r"($\delta$"+"#jumps)^2/(#jumps)/T" +" vs "+ xName +"\n"+specificationLine,
                    trajsExtremesInitID[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic, markerShapeVariables[filt], markerShapeVariablesNames,
                     arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                    yerr=deltaN2JumpsOverNJumpsStdErr[filt]/T[filt], nGraphs=len(np.unique(graphID[filt])), colorMapSpecifierName=colorMapSpecifierName
                    )
                
            mainPlot, _ = plotWithDifferentColorbars(f"qDist", x[filt], xName, qDist[filt], "distance", "Average distance from stfwd path between reference configurations over trajectory vs "+xName +"\n"+specificationLine,
                    trajsExtremesInitID[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic, markerShapeVariables[filt], markerShapeVariablesNames,
                     arrayForColorCoordinate[filt], colorMapSpecifier=colorMapSpecifier[filt],
                    yerr=qDistStdErr[filt], nGraphs=len(np.unique(graphID[filt])), colorMapSpecifierName=colorMapSpecifierName)

            mainPlot, _ = plotWithDifferentColorbars(f"tau", x[filt], xName, chi_tau[filt], r"$\tau_{trans}$", r"transient time $\tau_{trans}$ vs "+ xName+"\n"+specificationLine,
                        trajsExtremesInitID[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic, markerShapeVariables[filt], markerShapeVariablesNames,
                         arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                        yerr=stdDevBarrier[filt], nGraphs=len(np.unique(graphID[filt])), colorMapSpecifierName=colorMapSpecifierName)
                
            mainPlot, _ = plotWithDifferentColorbars(f"tau_InBetween", x[filt], xName, chi_tau2[filt], r"$\tau_{trans}$", r"transient time (in between fit) $\tau_{trans}$ vs "+ xName+"\n"+specificationLine,
                        trajsExtremesInitID[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic, markerShapeVariables[filt], markerShapeVariablesNames,
                         arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                        yerr=stdDevBarrier[filt], nGraphs=len(np.unique(graphID[filt])), colorMapSpecifierName=colorMapSpecifierName)
                
            mainPlot, _ = plotWithDifferentColorbars(f"realTime", x[filt], xName, realTime[filt], "computer time (seconds)", "Seconds required to perform 10^5 mc sweeps vs "+ xName+"\n"+specificationLine,
                        trajsExtremesInitID[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic, markerShapeVariables[filt], markerShapeVariablesNames,
                         arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                        yerr=realTimeErr[filt], nGraphs=len(np.unique(graphID[filt])), colorMapSpecifierName=colorMapSpecifierName)

            mainPlot, _ = plotWithDifferentColorbars(f"TIhout", x[filt], xName, TIhout[filt], "L", "Quantity for thermodynamic integration vs "+ xName +"\n"+specificationLine,
                    trajsExtremesInitID[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic,
                    markerShapeVariables[filt], markerShapeVariablesNames,
                     arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                    nGraphs=len(np.unique(graphID[filt])), colorMapSpecifierName=colorMapSpecifierName)
                
            mainPlot, _ = plotWithDifferentColorbars(f"TIQstar", x[filt], xName, TIQstar[filt], "L", "Quantity for thermodynamic integration vs "+ xName +"\n"+specificationLine,
                    trajsExtremesInitID[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic,
                    markerShapeVariables[filt], markerShapeVariablesNames,
                     arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                    nGraphs=len(np.unique(graphID[filt])), colorMapSpecifierName=colorMapSpecifierName)
 
 
            figs = plt.get_figlabels()  # Ottieni i nomi di tutte le figure create
            for fig_name in figs:
                fig = plt.figure(fig_name)
                if studyName == "StudyInBetaOverBetaMax":
                    for val in np.unique(averageBetaMax[filt][~np.isnan(averageBetaMax[filt])]):
                        plt.axvline(val, 0, 1, color='black', linestyle='--', linewidth=3)
                    for val in np.unique(betaMax[filt][~np.isnan(betaMax[filt])]):
                        plt.axvline(val, 0, 1, color='red', linewidth=3,  linestyle='--')
                    fig.canvas.draw() 
                
                filename = os.path.join(theseFiguresFolder, f'{fig_name}.png')
                #print(filename)
                fig.savefig(filename,dpi=300, bbox_inches='tight')

            plt.close('all')
            
            toFit = [None]
            if xName=="N":
                toFit=['linear']
                
            toCompare = np.logical_and(filt, PathsMCsUsedForTI_cleanFilt)
            if "N"==xName and len(np.unique(N[toCompare]))>=nNsToConsiderForSubFit:
                tempFilt=filt
                filt=np.logical_and.reduce([filt, toCompare, N>=np.sort(np.unique(N[toCompare]))[-nNsToConsiderForSubFit]])
                
                mainPlot, fitData = plotWithDifferentColorbars(f"k_log_{nNsToConsiderForSubFit}LargerNs", x[filt], xName, minusLnKFromChi[filt], "-ln k", "Transition rate computed from single TI and "+r"$\chi$ vs "+ xName+"\n"+specificationLine,
                    trajsExtremesInitID[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic,
                    markerShapeVariables[filt], markerShapeVariablesNames,
                     arrayForColorCoordinate[filt], colorMapSpecifier=colorMapSpecifier[filt],
                    nGraphs=len(np.unique(graphID[filt])), fitTypes=toFit, colorMapSpecifierName=colorMapSpecifierName)
                
                if fitData is not None:
                    barrierBet1lN.append((float)(v[-1][0]))
                    barrierBetBarr1lN.append(fitData[0])
                    barrierBetBarrErr1lN.append(fitData[2])
               
                mainPlot, fitData = plotWithDifferentColorbars(f"k2_log_{nNsToConsiderForSubFit}LargerNs", x[filt], xName, minusLnKFromChi_2[filt], "-ln k", "Transition rate (2) computed from single TI and "+r"$\chi$ vs "+ xName+"\n"+specificationLine,
                        trajsExtremesInitID[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic,
                        markerShapeVariables[filt], markerShapeVariablesNames,
                        arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                        nGraphs=len(np.unique(graphID[filt])), fitTypes=toFit, colorMapSpecifierName=colorMapSpecifierName)
                if fitData is not None:
                        barrierBet2lN.append((float)(v[-1][0]))
                        barrierBetBarr2lN.append(fitData[0])
                        barrierBetBarrErr2lN.append(fitData[2])
                
                    
                mainPlot, fitData = plotWithDifferentColorbars(f"k2_scaled_log_{nNsToConsiderForSubFit}LargerNs", x[filt], xName, minusLnKFromChi_2_scaled[filt], "-ln k", "Transition rate (3) computed from single TI and "+r"$\chi$ vs "+ xName +"\n"+specificationLine,
                        trajsExtremesInitID[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic,
                        markerShapeVariables[filt], markerShapeVariablesNames,
                        arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                        nGraphs=len(np.unique(graphID[filt])), fitTypes=toFit, colorMapSpecifierName=colorMapSpecifierName)
                if fitData is not None:
                    barrierBet3lN.append((float)(v[-1][0]))
                    barrierBetBarr3lN.append(fitData[0])
                    barrierBetBarrErr3lN.append(fitData[2])
                
                filt = tempFilt #Remove previous filter that was selecting only large N-values
                
                

            tempFilt=filt
            filt= np.logical_and(filt,PathsMCsUsedForTI_cleanFilt)

            mainPlot, fitData = plotWithDifferentColorbars(f"k_log", x[filt], xName, minusLnKFromChi[filt], "-ln(k)", "Transition rate computed from single TI and "+r"$\chi$ vs "+ xName+"\n"+specificationLine,
                    trajsExtremesInitID[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic,
                    markerShapeVariables[filt], markerShapeVariablesNames,
                     arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                    #nGraphs=len(np.unique(graphID[filt])),
                    fitTypes=toFit, colorMapSpecifierName=colorMapSpecifierName)
            if fitData is not None:
                    barrierBet1.append((float)(v[-1][0]))
                    barrierBetBarr1.append(fitData[0])
                    barrierBetBarrErr1.append(fitData[2])
                
                
            mainPlot, fitData = plotWithDifferentColorbars(f"k2_log", x[filt], xName, minusLnKFromChi_2[filt], "-ln k",  "Transition rate (1) computed from single TI and "+r"$\chi$ vs "+ xName+"\n"+specificationLine,
                    trajsExtremesInitID[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic,
                    markerShapeVariables[filt], markerShapeVariablesNames,
                     arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                    nGraphs=len(np.unique(graphID[filt])), fitTypes=toFit, colorMapSpecifierName=colorMapSpecifierName)
            if fitData is not None:
                barrierBet2.append((float)(v[-1][0]))
                barrierBetBarr2.append(fitData[0])
                barrierBetBarrErr2.append(fitData[2])
                
                
            mainPlot, fitData = plotWithDifferentColorbars(f"k2_scaled_log", x[filt], xName, minusLnKFromChi_2_scaled[filt], "-ln k",  "Transition rate (3) computed from single TI and "+ xName +"\n"+specificationLine,
                    trajsExtremesInitID[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic,
                    markerShapeVariables[filt], markerShapeVariablesNames,
                     arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                    nGraphs=len(np.unique(graphID[filt])), fitTypes=toFit, colorMapSpecifierName=colorMapSpecifierName)
            if fitData is not None:
                barrierBet3.append((float)(v[-1][0]))
                barrierBetBarr3.append(fitData[0])
                barrierBetBarrErr3.append(fitData[2])
                
            filt=tempFilt
            
            mainPlot, fitData = plotWithDifferentColorbars(f"k_log_rep", x[filt], xName, minusLnKFromChi[filt], "-ln(k)", "Transition rate computed from single TI and "+r"$\chi$ vs "+ xName+"\n"+specificationLine,
                    trajsExtremesInitID[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic,
                    markerShapeVariables[filt], markerShapeVariablesNames,
                     arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                    nGraphs=len(np.unique(graphID[filt])), fitTypes=toFit, colorMapSpecifierName=colorMapSpecifierName)
                
            mainPlot, fitData = plotWithDifferentColorbars(f"k2_log_rep", x[filt], xName, minusLnKFromChi_2[filt], "-ln(k)",  "Transition rate (1) computed from single TI and "+r"$\chi$ vs "+ xName+"\n"+specificationLine,
                    trajsExtremesInitID[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic,
                    markerShapeVariables[filt], markerShapeVariablesNames,
                     arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                    nGraphs=len(np.unique(graphID[filt])), fitTypes=toFit, colorMapSpecifierName=colorMapSpecifierName)
            
            mainPlot, fitData = plotWithDifferentColorbars(f"k2_scaled_log_rep", x[filt], xName, minusLnKFromChi_2_scaled[filt], "-ln(k)",  "Transition rate (3) computed from single TI and "+ xName +"\n"+specificationLine,
                    trajsExtremesInitID[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic,
                    markerShapeVariables[filt], markerShapeVariablesNames,
                     arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                    nGraphs=len(np.unique(graphID[filt])), fitTypes=toFit, colorMapSpecifierName=colorMapSpecifierName)

                
            mainPlot, _ = plotWithDifferentColorbars(f"k", x[filt], xName, kFromChi[filt], "k", "Transition rate computed from single TI and "+r"$\chi$ vs "+ xName+"\n"+specificationLine,
                    trajsExtremesInitID[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic,
                    markerShapeVariables[filt], markerShapeVariablesNames,
                     arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                    nGraphs=len(np.unique(graphID[filt])), colorMapSpecifierName=colorMapSpecifierName)
            
            mainPlot, _ = plotWithDifferentColorbars(f"k2", x[filt], xName, kFromChi_InBetween[filt], "k", "Generalized transition rate computed from single TI and "+r"$\chi$ vs "+ xName+"\n"+specificationLine,
                    trajsExtremesInitID[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic,
                    markerShapeVariables[filt], markerShapeVariablesNames,
                     arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                    nGraphs=len(np.unique(graphID[filt])), colorMapSpecifierName=colorMapSpecifierName)
            
            mainPlot, _ = plotWithDifferentColorbars(f"k2_scaled", x[filt], xName, kFromChi_InBetween_Scaled[filt], "L", "Quantity for thermodynamic integration vs "+ xName +"\n"+specificationLine,
                    trajsExtremesInitID[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic,
                    markerShapeVariables[filt], markerShapeVariablesNames,
                     arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                    nGraphs=len(np.unique(graphID[filt])), colorMapSpecifierName=colorMapSpecifierName)
            
            mainPlot, _ = plotWithDifferentColorbars(f"scale", x[filt], xName, scale2[filt], "f", "Projected probability of being in final cone based on linearity, vs "+ xName +"\n"+specificationLine,
                    trajsExtremesInitID[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic,
                    markerShapeVariables[filt], markerShapeVariablesNames,
                     arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                    nGraphs=len(np.unique(graphID[filt])), colorMapSpecifierName=colorMapSpecifierName)
            
            theoreticalX, theoreticalY0, theoreticalY1 = analyticDataF(x, xName, filt, fieldSigma,beta)


            mainPlot, _ = plotWithDifferentColorbars(f"tentativeBarrier", x[filt], xName, tentativeBarrier[filt]/beta[filt], "Tentative "+ r"$\delta$f", "Tentative free energy barrier (1) "+ r"(-ln(k)/($\beta$N))"+", vs "+ xName +"\n"+specificationLine,
                    trajsExtremesInitID[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic,
                    markerShapeVariables[filt], markerShapeVariablesNames,
                     arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                    nGraphs=len(np.unique(graphID[filt])), colorMapSpecifierName=colorMapSpecifierName,
                    theoreticalX=theoreticalX, theoreticalY=theoreticalY0)

            mainPlot, _ = plotWithDifferentColorbars(f"tentativeBarrier2", x[filt], xName, tentativeBarrier_2[filt]/beta[filt], "Tentative "+ r"$\delta$f", "Tentative free energy barrier (2) "+ r"(-ln(k)/ ($\beta$N))"+", vs "+ xName +"\n"+specificationLine,
                    trajsExtremesInitID[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic,
                    markerShapeVariables[filt], markerShapeVariablesNames,
                     arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                    nGraphs=len(np.unique(graphID[filt])), colorMapSpecifierName=colorMapSpecifierName,
                    theoreticalX=theoreticalX, theoreticalY=theoreticalY0)
                    
            mainPlot, _ = plotWithDifferentColorbars(f"tentativeBarrier3", x[filt], xName, tentativeBarrier_3[filt]/beta[filt], "Tentative "+ r"$\delta$f", "Tentative free energy barrier (3) "+ r"(-ln(k)/ ($\beta$N))"+", vs "+ xName +"\n"+specificationLine,
                    trajsExtremesInitID[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic,
                    markerShapeVariables[filt], markerShapeVariablesNames,
                     arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                    nGraphs=len(np.unique(graphID[filt])), colorMapSpecifierName=colorMapSpecifierName,
                    theoreticalX=theoreticalX, theoreticalY=theoreticalY0)
            
            mainPlot, _ = plotWithDifferentColorbars(f"betTentativeBarrier", x[filt], xName, tentativeBarrier[filt], "Tentative "+ r"$\beta\delta$f", "Tentative free energy barrier (1) "+ r"(-ln(k)/N)"+", vs "+ xName +"\n"+specificationLine,
                    trajsExtremesInitID[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic,
                    markerShapeVariables[filt], markerShapeVariablesNames,
                     arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                    nGraphs=len(np.unique(graphID[filt])), colorMapSpecifierName=colorMapSpecifierName,
                    theoreticalX=theoreticalX, theoreticalY=theoreticalY1)

            mainPlot, _ = plotWithDifferentColorbars(f"betTentativeBarrier2", x[filt], xName, tentativeBarrier_2[filt], "Tentative "+ r"$\beta\delta$f", "Tentative free energy barrier (2) "+ r"(-ln(k)/ N)"+", vs "+ xName +"\n"+specificationLine,
                    trajsExtremesInitID[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic,
                    markerShapeVariables[filt], markerShapeVariablesNames,
                     arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                    nGraphs=len(np.unique(graphID[filt])), colorMapSpecifierName=colorMapSpecifierName,
                    theoreticalX=theoreticalX, theoreticalY=theoreticalY1)
            
            mainPlot, _ = plotWithDifferentColorbars(f"betTentativeBarrier3", x[filt], xName, tentativeBarrier_3[filt], "Tentative "+ r"$\beta\delta$f", "Tentative free energy barrier (3) "+ r"(-ln(k)/ N)"+", vs "+ xName +"\n"+specificationLine,
                    trajsExtremesInitID[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic,
                    markerShapeVariables[filt], markerShapeVariablesNames,
                     arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                    nGraphs=len(np.unique(graphID[filt])), colorMapSpecifierName=colorMapSpecifierName,
                    theoreticalX=theoreticalX, theoreticalY=theoreticalY1)
            
            
            filters = []
            functions = []
            rescaledFunctions = []

            functionNameToGet =None
            if studyName=="StudyInBeta" or studyName=="StudyInBeta_AllNs":
                functionNameToGet = 'Zfunction'
            elif "StudyInBetaOverBetaMax" in studyName:
                functionNameToGet = 'rescaledZfunction_Max'
            elif studyName=="StudyInBetaOverBetaG":
                functionNameToGet = 'rescaledZfunction_g'
            elif studyName=="StudyInBetaOverBetaL":
                functionNameToGet = 'rescaledZfunction_l'
                
            if functionNameToGet is not None:
                for sim_N, sim_graphID, sim_Hext in set(zip(N[filt], graphID[filt], h_ext[filt])):
                    if (sim_N, sim_graphID, sim_Hext) not in Zdict.keys():
                        continue
                    subdict1 = Zdict[(sim_N, sim_graphID, sim_Hext)]
                    for sim_fieldType, sim_fieldSigma, sim_fieldRealization in set(zip(fieldType[filt], fieldSigma[filt], fieldRealization[filt])):
                        if (sim_fieldType, sim_fieldRealization, sim_fieldSigma) not in subdict1.keys():
                            continue
                        subdict2 = subdict1[(sim_fieldType, sim_fieldRealization, sim_fieldSigma)]
                        #specifying configurations
                        for sim_betOfEx, simfirst_ConfIndex, sim_secondConfIndex in set(zip(betaOfExtraction[filt],firstConfigurationIndex[filt], secondConfigurationIndex[filt])):
                            if (sim_betOfEx, simfirst_ConfIndex, sim_secondConfIndex) not in subdict2.keys():
                                continue
                            subdict3 = subdict2[(sim_betOfEx, simfirst_ConfIndex, sim_secondConfIndex)]
                            #specifying stochastic measure parameters
                            for sim_Hin, sim_Hout, sim_nQstar in set(zip(h_in[filt], h_out[filt], normalizedQstar[filt])):
                                if (sim_Hin, sim_Hout, sim_nQstar) not in subdict3.keys():
                                    continue
                                subdict4 = subdict3[(sim_Hin, sim_Hout, sim_nQstar)]
                                #specifico il tempo
                                for sim_T, sim_trajInit in set(zip(T[filt],trajsExtremesInitID[filt])):
                                    if (sim_T, sim_trajInit) not in subdict4.keys():
                                        continue
                                    folderingVariableFilt = np.logical_and.reduce([
                                        sim_N==N[filt], sim_graphID==graphID[filt], sim_Hext==h_ext[filt],
                                        sim_fieldType==fieldType[filt], sim_fieldSigma==fieldSigma[filt], sim_fieldRealization==fieldRealization[filt],
                                        sim_betOfEx==betaOfExtraction[filt], simfirst_ConfIndex==firstConfigurationIndex[filt], sim_secondConfIndex==secondConfigurationIndex[filt],
                                        sim_Hin==h_in[filt], sim_Hout==h_out[filt], sim_nQstar==Qstar[filt],
                                        sim_T==T[filt], sim_trajInit==trajsExtremesInitID[filt]
                                    ])
                                    f= subdict4[(sim_T, sim_trajInit)]['Zfunction']
                                    f= subdict4[(sim_T, sim_trajInit)][functionNameToGet]
                                    filters.append(folderingVariableFilt)
                                    functions.append(f)
            #print(functions,filters)
            
            mainPlot, _ = plotWithDifferentColorbars(f"ZfunctionAndCurve_log", x[filt], xName, ZFromTIBeta[filt], "Z", "Probability of having Q(s(T), "+r"$s_{out}$) $\geq$"+"Q* vs "+ xName +"\n"+specificationLine,
                    trajsExtremesInitID[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic,
                    markerShapeVariables[filt], markerShapeVariablesNames,
                     arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                    nGraphs=len(np.unique(graphID[filt])), yscale='log', functionsToPlotContinuously=[functions, filters])
                
            mainPlot, _ = plotWithDifferentColorbars(f"ZfunctionAndCurve_log_log", x[filt], xName, ZFromTIBeta[filt], "Z", "Probability of having Q(s(T), "+r"$s_{out}$) $\geq$"+"Q* vs "+ xName +"\n"+specificationLine,
                    trajsExtremesInitID[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic,
                    markerShapeVariables[filt], markerShapeVariablesNames,
                     arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                    nGraphs=len(np.unique(graphID[filt])),xscale='log', yscale='log', functionsToPlotContinuously=[functions, filters])
                
            if "StudyInT" in studyName:
                toFit=['linear']
            if np.all(N[filt]==34) and np.all(beta[filt]<0.3):
                toFit=['expo']
            if np.all(N[filt]==34) and np.all(beta[filt]>0.5):
                toFit=['quadratic']
            mainPlot, _ = plotWithDifferentColorbars(f"ZfunctionAndCurve", x[filt], xName, ZFromTIBeta[filt], "Z", '', #"Probability of having Q(s(T), "+r"$s_{out}$) $\geq$"+"Q* vs "+ xName +"\n"+specificationLine,
                    trajsExtremesInitID[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic,
                    markerShapeVariables[filt], markerShapeVariablesNames,
                     arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],  fitTypes=toFit,
                    #nGraphs=len(np.unique(graphID[filt])),
                    functionsToPlotContinuously=[functions, filters])
            
            figs = plt.get_figlabels()  # Ottieni i nomi di tutte le figure create
            for fig_name in figs:
                fig = plt.figure(fig_name)
                if studyName == "StudyInBetaOverBetaMax":
                    for val in np.unique(averageBetaMax[filt][~np.isnan(averageBetaMax[filt])]):
                        plt.axvline(val, 0, 1, color='black', linestyle='--', linewidth=3)
                    for val in np.unique(betaMax[filt][~np.isnan(betaMax[filt])]):
                        plt.axvline(val, 0, 1, color='red', linewidth=3,  linestyle='--')
                    fig.canvas.draw() 
                
                filename = os.path.join(theseFiguresFolder, f'{fig_name}.png')
                #print(filename)
                fig.savefig(filename, dpi=300, bbox_inches='tight')

            plt.close('all')       
            
            if (xName=="N") and (i in (nLastSubfVal-1)):
                considerToPlotN(theseFiguresFolder)
            elif (xName=="T") and (i in (nLastSubfVal-1)):     
                    considerToPlotT(theseFiguresFolder)

                    


    selectedRunGroups_FiltersAndNames = [
        [
            np.logical_and(C==c, fPosJ==f),
            f"p2C{c}/fPosJ{f:.2f}" if c!=0 else f"fPosJ{f:.2f}"
        ]
        for c,f in set(zip(C,fPosJ))
        ]

    selectedRunGroups_FiltersAndNames= [runGroup for runGroup in selectedRunGroups_FiltersAndNames if runGroup[0].sum()>minNumberOfSingleRunsToDoAnAnalysis]

    for runGroupFilter, runGroupName in selectedRunGroups_FiltersAndNames:
        
        analysis_path = os.path.join(parentAnalysis_path, runGroupName)
        #runGroupFilter=np.logical_and.reduce([runGroupFilter,fieldSigma==0.,normalizedQstar==0.6])
        #runGroupFilter=np.logical_and.reduce([runGroupFilter,fieldSigma==0.,normalizedQstar==0.6])
        #print(np.nancumsum(realTime*nMeasures," seconds"))
        #return
        global analyBet, analyBarr, analyBetBarr
        
        analyticalData_path = "data.txt"
        analyBet= None
        analyBarr= None
        analyBetBarr= None

        if symType=="RRG" and np.all(C[runGroupFilter]==3) and np.all(fPosJ[runGroupFilter]==1.0):
            # Carica i dati dal file di testo
            data = np.loadtxt(analyticalData_path)
            # Estrai le colonne x e y dai dati
            analyBet = data[:, 0]  # Seconda colonna
            analyBetBarr = data[:, 1]  # Seconda colonna
            analyBet=np.asarray(analyBet)
            analyBetBarr=np.asarray(analyBetBarr)
            analyBet=analyBet[analyBetBarr>0]
            analyBetBarr=analyBetBarr[analyBetBarr>0]
            analyBarr=[analyBetBarr[i]/analyBet[i] for i in range(len(analyBetBarr))]
            analyBarr=np.asarray(analyBarr)
        else:
            analyBet=None
            analyBetBarr=None
            analyBarr=None
            
        
        (TDN, TDTrajInit, TDT, TDBetOfEx, TDFirstConfIndex, TDSecondConfIndex, TDGraphId,
        TDFieldType, TDFieldReali, TDFieldSigma, TDHext,
        TDHout, TDHin, TDnQstar, TDBetaM, TDBetaG, TDBetaG2, TDBetaG3,TDbetaG2b,TDbetaG2c, TDBetaL, TDZmax,
        PathsMCsUsedForTI_cleanFilt) = thermodynamicIntegration(runGroupFilter, analysis_path)
        
        print("B LUNGHEZZA DI", np.nansum(PathsMCsUsedForTI_cleanFilt))

        myTDStudy(TDN, "N", np.asarray(list(zip(TDnQstar, TDFieldType, TDFieldSigma))),
                            [["Qstar"], ["fieldType", r"$\sigma$"]],
                            np.array(list(zip( TDGraphId))),
                            [r"graphID"],
                            arrayForColorCoordinate=TDT
                            )
        myTDStudy(TDT, "T", np.asarray(list(zip(TDnQstar, TDFieldType, TDFieldSigma))),
                            [["Qstar", "fieldType", r"$\sigma$"]],
                            np.array(list(zip( TDGraphId))),
                            [r"graphID"],
                            arrayForColorCoordinate=TDN
                            )
        
        if len(np.unique(N[runGroupFilter]))>2: 
            myMultiRunStudy(runGroupFilter, "StudyInN", N, "N",
                            np.asarray(list(zip(h_ext, fieldType, fieldSigma,
                                                normalizedQstar, h_in, h_out,
                                                beta))),
                            [["Hext","fieldType","sigma"],
                             ["Qstar","Hin","Hout"],
                             [r"beta"]],
                            np.array(list(zip( graphID, fieldRealization))),
                            ["graph", "r"],
                            colorMapSpecifier=np.full(len(normalizedQstar),"nan"),arrayForColorCoordinate=refConfMutualQ)
        
        myMultiRunStudy(runGroupFilter, "StudyInNProvaM", N, "N",
                            np.asarray(list(zip(h_ext, fieldType, fieldSigma,
                                                normalizedQstar, h_in, h_out,
                                                discretizedRescaledBetas_M))),
                            [["Hext","fieldType","sigma"], ["Qstar","Hin","Hout"],["rMbeta"]],
                            np.array(list(zip( beta, graphID, fieldRealization))),
                            [r"beta","graphID",r"r"],
                            colorMapSpecifier=np.full(len(normalizedQstar),"nan"),
                            arrayForColorCoordinate=rescaledBetas_M)
     
        if len(np.unique(T[runGroupFilter]))>2:
            myMultiRunStudy(runGroupFilter,"StudyInT", T,  "T",
                            np.array(list(zip(N,h_ext, normalizedQstar, h_in, h_out,beta))), [["N","Hext"],["Qstar","Hin","Hout"],[r"beta"]],
                            np.array(list(zip( graphID))), [ "graphID"])
        if len(np.unique(T[runGroupFilter]))>2:
            myMultiRunStudy(runGroupFilter,"StudyInT_allNs_withRescBetas", T,  "T",
                            np.array(list(zip(h_ext, normalizedQstar, h_in, h_out,discretizedRescaledBetas_M))), [["Hext"],["Qstar","Hin","Hout"],[r"rbeta"]],
                            np.array(list(zip(N, graphID))), [ "N","graphID"])
        if len(np.unique(T[runGroupFilter]))>2:
            myMultiRunStudy(runGroupFilter,"StudyInT_allNs", T,  "T",
                            np.array(list(zip(h_ext, normalizedQstar, h_in, h_out,beta))), [["Hext"],["Qstar","Hin","Hout"],[r"beta"]],
                            np.array(list(zip(N, graphID))), [ "N","graphID"])
        #""" 
        if sum((np.sum(fieldSigma == v) >= 8 and np.sum(fieldSigma == -v) >= 8) for v in np.unique(fieldSigma[fieldSigma > 0])) >= 1:
            myMultiRunStudy(runGroupFilter, "StudyInNProvaM_sigmaAbs", N, "N",
                                np.asarray(list(zip(h_ext, fieldType, np.fabs(fieldSigma),
                                                    normalizedQstar, h_in, h_out,
                                                    discretizedRescaledBetas_M))),
                                [["Hext","fieldType","sigma"], ["Qstar","Hin","Hout"],["rMbeta"]],
                                np.array(list(zip( beta, graphID, fieldRealization))),
                                [r"beta","graphID",r"r"],
                                colorMapSpecifier=np.full(len(normalizedQstar),"nan"),
                                arrayForColorCoordinate=rescaledBetas_M)
                                                    normalizedQstar, h_in, h_out,
                                                    discretizedRescaledBetas_M))),
                                [["Hext","fieldType","sigma"], ["Qstar","Hin","Hout"],["rMbeta"]],
                                np.array(list(zip( beta, graphID, fieldRealization))),
                                [r"beta","graphID",r"r"],
                                colorMapSpecifier=np.full(len(normalizedQstar),"nan"),
                                arrayForColorCoordinate=rescaledBetas_M)
        
            
        if len(np.unique(rescaledBetas_G[runGroupFilter]/beta[runGroupFilter]))>2:
            myMultiRunStudy(runGroupFilter,"StudyInBetaOverBetaG_AllNs", rescaledBetas_G, r"$\beta$  $\frac{\langle \beta_{max} \rangle_g}{\beta_{max}}$",
                            np.array(list(zip( normalizedQstar, fieldType, fieldSigma))),
                            [["Qstar", "fieldType", r"$\sigma$"]],
                            np.array(list(zip(graphID, fieldRealization, T))),
                            ["graphID", "r", "T"])
            
        if len(np.unique(rescaledBetas_G3[runGroupFilter]/beta[runGroupFilter]))>2:
            myMultiRunStudy(runGroupFilter,"StudyInBetaOverBetaG3_AllNs", rescaledBetas_G3, r"$\beta$  $\frac{\langle \beta_{max} \rangle_g}{\beta_{max}}$",
                            np.array(list(zip( normalizedQstar, fieldType, fieldSigma))),
                            [["Qstar", "fieldType", r"$\sigma$"]],
                            np.array(list(zip(graphID, fieldRealization, T))),
                            ["graphID", "r", "T"],
                            colorMapSpecifier=N,
                            arrayForColorCoordinate=T)
        if len(np.unique(rescaledBetas_G2b[runGroupFilter]/beta[runGroupFilter]))>2:
            myMultiRunStudy(runGroupFilter,"StudyInBetaOverbetaG2b_AllNs", rescaledBetas_G2b, r"$\beta$  $\frac{\langle \beta_{max} \rangle_g}{\beta_{max}}$",
                            np.array(list(zip( normalizedQstar, fieldType, fieldSigma))),
                            [["Qstar", "fieldType", r"$\sigma$"]],
                            np.array(list(zip(graphID, fieldRealization, T))),
                            ["graphID", "r", "T"],
                            colorMapSpecifier=N,
                            arrayForColorCoordinate=T)
        if len(np.unique(rescaledBetas_G2c[runGroupFilter]/beta[runGroupFilter]))>2:
            myMultiRunStudy(runGroupFilter,"StudyInBetaOverbetaG2c_AllNs", rescaledBetas_G2c, r"$\beta$  $\frac{\langle \beta_{max} \rangle_g}{\beta_{max}}$",
                            np.array(list(zip( normalizedQstar, fieldType, fieldSigma))),
                            [["Qstar", "fieldType", r"$\sigma$"]],
                            np.array(list(zip(graphID, fieldRealization, T))),
                            ["graphID", "r", "T"],
                            colorMapSpecifier=N,
                            arrayForColorCoordinate=T)

            
        if len(np.unique(rescaledBetas_G2[runGroupFilter]/beta[runGroupFilter]))>2:
            myMultiRunStudy(runGroupFilter,"StudyInBetaOverBetaG2", rescaledBetas_G2, r"$\beta$  $\frac{\langle \beta_{max} \rangle_g}{\beta_{max}}$",
                            np.array(list(zip(N, normalizedQstar, fieldType, fieldSigma))),
                            [["N", "Qstar", "fieldType", r"$\sigma$"]],
                            np.array(list(zip(graphID, fieldRealization, T))),
                            ["graphID", "r", "T"])
            
        myMultiRunStudy(runGroupFilter, "StudyInNProvaG", N, "N",
                            np.asarray(list(zip(h_ext, fieldType, fieldSigma,
                                                normalizedQstar, h_in, h_out,
                                                discretizedRescaledBetas_G))),
                            [["Hext","fieldType","sigma"], ["Qstar","Hin","Hout"],["rGbeta"]],
                            np.array(list(zip( beta, graphID))),
                            [r"beta","graphID"],
                            colorMapSpecifier=np.full(len(normalizedQstar),"nan"),
                            arrayForColorCoordinate=rescaledBetas_G)
        
        myMultiRunStudy(runGroupFilter, "StudyInNProvaG2", N, "N",
                            np.asarray(list(zip(h_ext, fieldType, fieldSigma,
                                                normalizedQstar, h_in, h_out,
                                                discretizedRescaledBetas_G2))),
                            [["Hext","fieldType","sigma"], ["Qstar","Hin","Hout"],["rGbeta"]],
                            np.array(list(zip( beta, graphID))),
                            [r"beta","graphID"],
                            colorMapSpecifier=np.full(len(normalizedQstar),"nan"),
                            arrayForColorCoordinate=rescaledBetas_G2)
        
        myMultiRunStudy(runGroupFilter, "StudyInNProvaG3", N, "N",
                            np.asarray(list(zip(h_ext, fieldType, fieldSigma,
                                                normalizedQstar, h_in, h_out,
                                                discretizedRescaledBetas_G3))),
                            [["Hext","fieldType","sigma"], ["Qstar","Hin","Hout"],["rGbeta"]],
                            np.array(list(zip( beta, graphID))),
                            [r"beta","graphID"],
                            colorMapSpecifier=np.full(len(normalizedQstar),"nan"),
                            arrayForColorCoordinate=rescaledBetas_G3)
        myMultiRunStudy(runGroupFilter, "StudyInNProvaG2b", N, "N",
                            np.asarray(list(zip(h_ext, fieldType, fieldSigma,
                                                normalizedQstar, h_in, h_out,
                                                discretizedRescaledBetas_G3))),
                            [["Hext","fieldType","sigma"], ["Qstar","Hin","Hout"],["rGbeta"]],
                            np.array(list(zip( beta, graphID))),
                            [r"beta","graphID"],
                            colorMapSpecifier=np.full(len(normalizedQstar),"nan"),
                            arrayForColorCoordinate=rescaledBetas_G2b)
        myMultiRunStudy(runGroupFilter, "StudyInNProvaG2c", N, "N",
                            np.asarray(list(zip(h_ext, fieldType, fieldSigma,
                                                normalizedQstar, h_in, h_out,
                                                discretizedRescaledBetas_G2c))),
                            [["Hext","fieldType","sigma"], ["Qstar","Hin","Hout"],["rGbeta"]],
                            np.array(list(zip( beta, graphID))),
                            [r"beta","graphID"],
                            colorMapSpecifier=np.full(len(normalizedQstar),"nan"),
                            arrayForColorCoordinate=rescaledBetas_G2c)

        myMultiRunStudy(runGroupFilter, "StudyInNabs", N, "N",
                            np.asarray(list(zip(h_ext, fieldType, np.fabs(fieldSigma),
                                                normalizedQstar, h_in, h_out,
                                                beta))),
                            [["Hext","absFieldType","sigma"],
                             ["Qstar","Hin","Hout"],
                             [r"beta"]],
                            np.array(list(zip( graphID, fieldRealization))),
                            ["graph", "r"],
                            colorMapSpecifier=np.full(len(normalizedQstar),"nan"),arrayForColorCoordinate=refConfMutualQ)
        
        


        if len(np.unique(N[runGroupFilter]))>1:
            if len(np.unique(rescaledBetas_M[runGroupFilter]))>2:
                myMultiRunStudy(runGroupFilter,"StudyInBetaOverBetaMax_allNs", rescaledBetas_M,
                                r"$\beta$  $\frac{\langle \beta_{max} \rangle_g}{\beta_{max}}$",
                                np.array(list(zip( normalizedQstar, fieldType, fieldSigma))),
                                [[ "Qstar", "fieldType", r"$\sigma$"]],
                                np.array(list(zip(graphID, fieldRealization))),
                                ["graphID", "r"],
                                colorMapSpecifier=N,
                                arrayForColorCoordinate=T)

                          
                
            if len(np.unique(beta[runGroupFilter]))>2:
                myMultiRunStudy(runGroupFilter,"StudyInBeta_allNs", beta, r"$\beta$",
                                np.array(list(zip(h_ext, fieldType, fieldSigma, normalizedQstar, h_in, h_out))),
                                [["Hext"], [ "fieldType", r"$\sigma$"], ["Qstar","Hin","Hout"]],
                                np.array(list(zip(graphID, fieldRealization))),
                                ["graphID", "r"],
                                colorMapSpecifier=N,
                                arrayForColorCoordinate=T)
        
        
            
        if len(np.unique(beta[runGroupFilter]))>2:
            myMultiRunStudy(runGroupFilter,"StudyInBeta", beta, r"$\beta$",
                            np.array(list(zip(N, h_ext, fieldType, fieldSigma, normalizedQstar, h_in, h_out))),
                            [["N"], ["Hext"], [ "fieldType", r"$\sigma$"], ["Qstar","Hin","Hout"]],
                            np.array(list(zip(graphID, T))),
                            ["graphID", "T"])
        
        
        if len(np.unique(rescaledBetas_M[runGroupFilter]))>2:
            myMultiRunStudy(runGroupFilter,"StudyInBetaOverBetaMax", rescaledBetas_M, r"$\beta$  $\frac{\langle \beta_{max} \rangle_g}{\beta_{max}}$",
                            np.array(list(zip(N, normalizedQstar, fieldType, fieldSigma))),
                            [["N", "Qstar", "fieldType", r"$\sigma$"]],
                            np.array(list(zip(graphID, fieldRealization, T))), ["graphID", "r", "T"])
    
        if len(np.unique(rescaledBetas_L[runGroupFilter]/beta[runGroupFilter]))>2:
            myMultiRunStudy(runGroupFilter,"StudyInBetaOverBetaL_AllNs", rescaledBetas_L, r"$\beta$  $\frac{\langle \beta_{max} \rangle_g}{\beta_{max}}$",
                            np.array(list(zip(normalizedQstar, fieldType, fieldSigma))),
                            [[ "Qstar", "fieldType", r"$\sigma$"]],
                            np.array(list(zip(graphID, fieldRealization, T))),
                            ["graphID", "r", "T"])
        

        if len(np.unique(fieldSigma[runGroupFilter]))>1:
                myMultiRunStudy(runGroupFilter,"StudyInFieldSigma", fieldSigma, r"fieldSigma",
                                np.array(list(zip(N, T))), [["N", "T"]],
                                np.array(list(zip(beta, graphID, fieldRealization))), ["field"])
     
    