import os
import sys

from scipy import interpolate
sys.path.append('../')
import glob
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
discBetaStep=0.02
nNsToConsiderForSubFit=4 #For extracting free energy barriers, there is also a fit
                                #involving only this number of largest Ns

analyBet= None
analyBarr= None
analyBetBarr= None
                        
fieldTypeDictionary ={"2":"gauss", "1":"bernoulli", "nan":"noField"}
fieldTypePathDictionary ={"gauss":"stdGaussian", "bernoulli":"stdBernoulli"}
trajInitShortDescription_Dict= {0: "stdMC", 70: "Random", 71: "Ref 12", 72: "Ref 21", 73: "Annealing", 74: "Annealing", 740: "AnnealingF", -2:"Fit"}
edgeColorPerInitType_Dic={0: "None", 70: "lightGreen", 71: "black", 72: "purple", 73: "orange", 74: "orange", 740: "red", -2:"black"}

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
        preliminaryFilter = np.full(len(x), True)
    best_indices = {}
    filterToReturn = np.full(len(x), False)
    # Iteriamo su ogni valore unico in filteredStdMCsBetas
    for value in np.sort(np.unique(x[preliminaryFilter])):
        # Trova gli indici corrispondenti a questo valore unico
        indices = np.where(np.logical_and(x == value, preliminaryFilter))[0]

        if len(indices) > 1:
            # Se ci sono più di un indice, scegli quello con il valore massimo in lastMeasureMC
            best_index = indices[np.argmax(criterion[indices])]
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

def analyticDataF2(x, xName):
    x=np.array(x,dtype=np.float64)
    if (analyBet is None) or ("beta" not in xName):
        return None, None, None
    arrayToConsider=x
    betaRange= np.nanmin(arrayToConsider), np.nanmax(arrayToConsider) 
    betaRangeCondition = np.logical_and(analyBet>=betaRange[0],analyBet<=betaRange[1]) 
    betaRangeCondition = np.logical_and(betaRangeCondition, analyBarr>0) 
    if np.all(betaRangeCondition==False):
        return None, None, None
    theoreticalX=analyBet[betaRangeCondition]
    theoreticalY0=analyBarr[betaRangeCondition]
    theoreticalY1=analyBetBarr[betaRangeCondition]
    if ("beta" not in xName):
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
    plt.rcParams['lines.markersize'] = 10
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
    scale2[scale2<0.34] = np.nan
    scale2[chi_chi2>0.3] = np.nan
    
    ZFromTIBeta = np.full_like(N, np.nan, dtype=np.float64) # l'esistenza di questo array è una sconfitta
    rescaledBetas_M = np.full_like(N, np.nan, dtype=np.float64) # l'esistenza di questo array è una sconfitta
    rescaledBetas_L = np.full_like(N, np.nan, dtype=np.float64) # l'esistenza di questo array è una sconfitta
    rescaledBetas_G = np.full_like(N, np.nan, dtype=np.float64) # l'esistenza di questo array è una sconfitta
    minusLnKFromChi = np.full_like(N, np.nan, dtype=np.float64) # l'esistenza di questo array è una sconfitta
    minusLnKFromChi_2 = np.full_like(N, np.nan, dtype=np.float64) # l'esistenza di questo array è una sconfitta
    minusLnKFromChi_2_scaled = np.full_like(N, np.nan, dtype=np.float64) # l'esistenza di questo array è una sconfitta
    discretizedRescaledBetas_M = np.full_like(N, np.nan, dtype=np.float64) # l'esistenza di questo array è una sconfitta
    discretizedRescaledBetas_G = np.full_like(N, np.nan, dtype=np.float64) # l'esistenza di questo array è una sconfitta
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
            print("flippo", np.cumsum(thisFieldPathsMCs), " e ", np.cumsum(thisFieldStdMCs))
            fieldSigma[thisFieldPathsMCs]=-fieldSigma[thisFieldPathsMCs]
            stMC_fieldSigma[thisFieldStdMCs]=-stMC_fieldSigma[thisFieldStdMCs]
    
    refConfMutualQ = refConfMutualQ/N

    
    Zdict = {}

    def thermodynamicIntegration(filt, analysis_path):
        
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
        TDBetaL=[]
        TDZmax=[]
        
        TIFolder= os.path.join(analysis_path, 'TI')

        if not os.path.exists(TIFolder):
            os.makedirs(TIFolder, exist_ok=True)
        else:
            delete_files_in_folder(TIFolder)
            
            
        #specifying graph in 2 cycle
        for sim_Hext in set(h_ext[filt]):
            TIFilt = np.logical_and.reduce([ h_ext==sim_Hext, filt])
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

                        betaMaxOverRealizations=0.
                        betaMaxOverRealizationsCounter=0
                        betaLOverRealizations=[]
                        betaLOverRealizationsId=[]
                        betaGOverRealizations=[]
                        betaGOverRealizationsId=[]
                        for sim_N, sim_C, sim_graphID, sim_fieldRealization in set(zip(N[TIFilt3], C[TIFilt3], graphID[TIFilt3], fieldRealization[TIFilt3])):
                            sim_Qstar=(int)(sim_N*sim_nQstar)
                            TIFilt4 = np.logical_and(TIFilt3, np.logical_and.reduce([N==sim_N, graphID==sim_graphID, fieldRealization==sim_fieldRealization]))
                            st_TIFilt4 = np.logical_and(st_TIFilt3, np.logical_and.reduce([stMC_N==sim_N , stMC_graphID==sim_graphID, stMC_fieldRealization==sim_fieldRealization]))

                            stdMCsBetas_forThisTDSetting, stdMCsTIBetas_forThisTDSetting, stdMCsMC_forThisTDSetting, filterForStdMCs_forThisTDSetting = getUniqueXAndYZAccordingToZ(stMC_beta, stMC_TIbeta, stMC_MC, preliminaryFilter=st_TIFilt4)
                            
                            #specifico il tempo
                            betaLForThisRealization=0.
                            betaGForThisRealization=0.
                            betaMaxForThisRealization=0.
                            betaMaxForThisRealizationCounter=0
                            betaMax2ForThisRealization=0.
                            betaMax2ForThisRealizationCounter=0
                            
                            TIPlotsFolder = os.path.join(TIFolder, f'N{sim_N}', f'h{sim_Hext}_f{sim_fieldType}{sim_fieldSigma}' if sim_fieldSigma!=0. else f'h{sim_Hext}_noField', f'g{sim_graphID}_fr{sim_fieldRealization}' if sim_fieldSigma!=0. else f'g{sim_graphID}',
                                                         f'bExt{sim_betOfEx}_cs{sim_firstConfIndex}_{sim_secondConfIndex}_{sim_Qif}' if (sim_firstConfIndex!="nan" and sim_firstConfIndex is not None) else 'FM',
                                                         f'meas_{(str)(sim_Hin)}_{(str)(sim_Hout)}_{(sim_nQstar):.3f}' if sim_Hin is not np.inf else f'meas_inf_inf_{(sim_nQstar):.3f}')

                            for sim_T, sim_trajInit in set(zip(T[TIFilt4], trajsExtremesInitID[TIFilt4])):
                                pathMCFilt_forThisTAndInit = np.logical_and.reduce([TIFilt4, T==sim_T, trajsExtremesInitID==sim_trajInit])
                            
                                
                                if (0. not in stdMCsBetas_forThisTDSetting and 0. not in beta[pathMCFilt_forThisTAndInit]):
                                    continue
                                maxPathsMCsBeta = np.nanmax(beta[pathMCFilt_forThisTAndInit])
                                stdMC_filtForThisTAndInit = np.logical_and(filterForStdMCs_forThisTDSetting, stMC_beta<maxPathsMCsBeta)
                                
                                
                                pathMCBetas_forThisTAndInit, pathMCTIs_forThisTAndInit, pathMCMCs_forThisTAndInit, pathMCFilter_forThisTAndInit = getUniqueXAndYZAccordingToZ(beta, TIbeta, lastMeasureMC, preliminaryFilter=pathMCFilt_forThisTAndInit)
                                
                                smallestPathsMcBetaToConsider_i = np.argmin(pathMCBetas_forThisTAndInit)
                                smallestPathsMcBetaToConsider= pathMCBetas_forThisTAndInit[smallestPathsMcBetaToConsider_i]
                                smallestPathsMcTIToConsider = pathMCTIs_forThisTAndInit[smallestPathsMcBetaToConsider_i]
                                if len(stdMCsBetas_forThisTDSetting*(stdMCsBetas_forThisTDSetting<smallestPathsMcBetaToConsider))==0:
                                    continue
                                largestStdMcBetaToConsider_i = np.argmax(stdMCsBetas_forThisTDSetting*(stdMCsBetas_forThisTDSetting<smallestPathsMcBetaToConsider))
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
                                    smallestLEqPathMCBeta_index=np.argmin(pathMCBetas_forThisTAndInit[pathsMcsToConsider])
                                    PathsMcTIToCompare=pathMCTIs_forThisTAndInit[pathsMcsToConsider][smallestLEqPathMCBeta_index]
                                    if abs((stdMcTIbeta-PathsMcTIToCompare)/TIDifferenceMax)<0:#0.003:
                                        largestStdMcBetaToConsider = stdMcBeta
                                        largestStdMcTIToConsider = stdMcTIbeta
                                        smallestLargerPathMCBeta_index=np.argmin(pathMCBetas_forThisTAndInit[pathMCBetas_forThisTAndInit>stdMcBeta])
                                        smallestPathsMcBetaToConsider = pathMCBetas_forThisTAndInit[pathMCBetas_forThisTAndInit>stdMcBeta][smallestLargerPathMCBeta_index]
                                        smallestPathsMcTIToConsider = pathMCTIs_forThisTAndInit[pathMCBetas_forThisTAndInit>stdMcBeta][smallestLargerPathMCBeta_index]

                                #print("B", largestStdMcBetaToConsider)
                                #print("C", smallestPathsMcBetaToConsider)
                                stdMC_filtForThisTAndInit_used = np.logical_and(stdMC_filtForThisTAndInit, stMC_beta <= largestStdMcBetaToConsider)
                                pathsMC_filtForThisTAndInit_used = np.logical_and(pathMCFilter_forThisTAndInit, beta >= smallestPathsMcBetaToConsider)
                                temp = np.sort(np.concatenate([stMC_beta[stdMC_filtForThisTAndInit_used], beta[pathsMC_filtForThisTAndInit_used]]))
                                maxBetaNotTooSpaced = np.max([temp[i] for i in range(1, len(temp)) if temp[i] - temp[i-1] <= 0.1])
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
                                betaMax= None
                                if np.fabs(largestStdMcTIToConsider-smallestPathsMcTIToConsider)<TIDifferenceMax/15.:
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
                                    print("Term int fatta")
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
                                betaL="unknown"
                                betaG="unknown"
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
                                            betaL=graphData['beta_c']['localApproach']
                                            betaG=graphData['beta_c']['globalApproach']

                                else:
                                    print("Info su beta di ", whereToFindBetaCs," non disponibili.")
                                
                                os.makedirs(TIPlotsFolder, exist_ok=True)
                                
                                vLines = []
                                if betaG!="unknown":
                                    vLines.append([betaG, r"$\beta_{G}$", "blue"])
                                    
                                if betaMax is not None:
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
                                    fig.savefig(filename, bbox_inches='tight')
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
                                
                                TIdata={}
                                TIdata['usedStMCsFilter'] = stdMC_filtForThisTAndInit_used
                                TIdata['usedPathsMCsFilter'] = pathsMC_filtForThisTAndInit_used
                                TIdata['unusedStMCsFilter'] = stdMC_filtForThisTAndInit_unused
                                TIdata['unusedPathsMCsFilter'] = pathsMC_filtForThisTAndInit_unused
                                levelToAdd['TIdata'] = TIdata
                                addedLevel = addLevelOnNestedDictionary(Zdict, [(sim_N, sim_graphID, sim_Hext), (sim_fieldType, sim_fieldRealization, sim_fieldSigma), (sim_betOfEx, sim_firstConfIndex, sim_secondConfIndex), (sim_Hin, sim_Hout, sim_nQstar), (sim_T, sim_trajInit)],
                                                levelToAdd)
                                                                
                                betaGForThisRealization = betaG    
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
                                if betaL=="unknown":
                                    betaL=np.nan
                                if betaG=="unknown":
                                    betaG=np.nan
                                TDBetaG.append(betaG)
                                TDBetaL.append(betaL)
                                TDZmax.append(Zfunction(betaMax))
                                

                            if betaMaxForThisRealizationCounter>0:
                                betaMaxOverRealizations += betaMaxForThisRealization/betaMaxForThisRealizationCounter
                                betaMaxOverRealizationsCounter+=1
                            
                            if isinstance(betaLForThisRealization, (int, float)):      
                                betaLOverRealizations.append(betaLForThisRealization)
                                betaLOverRealizationsId.append(sim_graphID)
                            if isinstance(betaGForThisRealization, (int, float)):      
                                betaGOverRealizations.append(betaGForThisRealization)
                                betaGOverRealizationsId.append(sim_graphID)
      
                        if betaMaxOverRealizationsCounter>0:
                            betaMaxOverRealizations /= betaMaxOverRealizationsCounter
                        if len(betaLOverRealizationsId)>0:
                            betaLOverRealizations = np.array(betaLOverRealizations)
                            betaLOverRealizationsId = np.array(betaLOverRealizationsId)
                            uniqueValues, unique_indices = np.unique(betaLOverRealizationsId, return_index=True)
                            betaLOverRealizations = np.sum(betaLOverRealizations[unique_indices])/len(uniqueValues)
                        if len(betaGOverRealizationsId)>0:
                            betaGOverRealizations = np.array(betaGOverRealizations)
                            betaGOverRealizationsId = np.array(betaGOverRealizationsId)
                            uniqueValues, unique_indices = np.unique(betaGOverRealizationsId, return_index=True)
                            betaGOverRealizations = np.sum(betaGOverRealizations[unique_indices])/len(uniqueValues)
                        
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
                                    if isinstance(thisCurveBetaMax, (int, float)):
                                        #print(betaGOverRealizations, betaMaxOverRealizations, thisCurveBetaMax, thisCurveBetaG)
                                        rescaledBetas_M[TIFilt_forThisTAndInit] = (beta[TIFilt_forThisTAndInit])/(thisCurveBetaMax)*(betaMaxOverRealizations)
                                        #minusLnKFromChi[TIFilt_forThisTAndInit] *= thisCurveBetaMax/betaMaxOverRealizations
                                        #minusLnKFromChi_2[TIFilt_forThisTAndInit] *= thisCurveBetaMax/betaMaxOverRealizations
                                        #minusLnKFromChi_2_scaled[TIFilt_forThisTAndInit] *= thisCurveBetaMax/betaMaxOverRealizations
                                        def rescaledZfunction_M(bet, numBet=betaMaxOverRealizations, denBet=thisCurveBetaMax, function=originalZfunction):
                                            return function(bet*denBet/numBet)
                                        level['rescaledZfunction_Max']=rescaledZfunction_M
                                        
                                    if isinstance(thisCurveBetaL, (int, float)):
                                        rescaledBetas_L[TIFilt_forThisTAndInit] = beta[TIFilt_forThisTAndInit]/thisCurveBetaL*betaLOverRealizations
                                        def rescaledZfunction_L(bet, numBet=1., denBet=thisCurveBetaL, function=originalZfunction):
                                            return function(bet*denBet/numBet)
                                        level['rescaledZfunction_l']=rescaledZfunction_L
                                    
                                    if isinstance(thisCurveBetaG, (int, float)):
                                        rescaledBetas_G[TIFilt_forThisTAndInit] = beta[TIFilt_forThisTAndInit]/thisCurveBetaG*betaGOverRealizations
                                        def rescaledZfunction_G(bet, numBet=1., denBet=thisCurveBetaG, function=originalZfunction):
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
                        if np.isnan(max_value):
                            continue
                        for discRescBeta in np.round(np.arange(0., max_value+discBetaStep, discBetaStep, dtype=float),decimals=4):
                            discretizableFilt=np.logical_and(TIFilt3, np.fabs(discRescBeta-rescaledBetas_G)<discBetaStep/2.+0.00001)
                            if len(np.unique(N[discretizableFilt]))>=3:
                                for g,ft,fr,fs in set(zip(graphID[discretizableFilt], fieldType[discretizableFilt],fieldRealization[discretizableFilt],fieldSigma[discretizableFilt])):
                                    discretizableFilt2=np.logical_and.reduce([
                                        discretizableFilt, fieldType==ft,fieldRealization==fr, fieldSigma==fs, graphID==g])
                                    diff = np.fabs(rescaledBetas_G[discretizableFilt2]-discRescBeta)
                                    closestBet = np.nanmin(diff)
                                    diff = np.fabs(rescaledBetas_G-discRescBeta)
                                    discretizableFilt2=np.logical_and.reduce([discretizableFilt2, diff==closestBet])
                                
                                    discretizedRescaledBetas_G[discretizableFilt2] =discRescBeta
    
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
        TDBetaL=np.array(TDBetaL,dtype=np.float64)
        TDZmax=np.array(TDZmax,dtype=np.float64)
        return (TDN, TDTrajInit, TDT, TDBetOfEx, TDFirstConfIndex, TDSecondConfIndex, TDGraphId,
                TDFieldType, TDFieldReali, TDFieldSigma, TDHext,
                TDHout, TDHin, TDnQstar, TDBetaM, TDBetaG, TDBetaL, TDZmax)   
                            
                   
                   
    def myTDStudy( x, xName, subfolderingVariable, subfolderingVariableNames, markerShapeVariables, markerShapeVariablesNames, arrayForColorCoordinate, colorMapSpecifier):
        thisStudyFolder= os.path.join(analysis_path, "TDStudy")
        os.makedirs(thisStudyFolder, exist_ok=True)

        thisStudyFolder= os.path.join(thisStudyFolder, xName)

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
            
            if xName!="T":
                mainPlot, _ = plotWithDifferentColorbars(f"betaG", x[filt], xName, TDBetaG[filt], r"$\beta_{G}$", r"$\beta_{G}$ vs "+ xName+"\n"+specificationLine,
                            TDTrajInit[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic, markerShapeVariables[filt], markerShapeVariablesNames,
                            arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                            nGraphs=len(np.unique(TDGraphId[filt])))
                mainPlot, _ = plotWithDifferentColorbars(f"betaL", x[filt], xName, TDBetaL[filt], r"$\beta_{L}$", r"$\beta_{L}$ vs "+ xName+"\n"+specificationLine,
                            TDTrajInit[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic, markerShapeVariables[filt], markerShapeVariablesNames,
                            arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                            nGraphs=len(np.unique(TDGraphId[filt])))
                mainPlot, _ = plotWithDifferentColorbars(f"betaMOvL", x[filt], xName, (TDBetaM/TDBetaL)[filt], r"$\beta_{M}/\beta_{L}$", r"$\beta_{M}/\beta_{L}$ vs "+ xName+"\n"+specificationLine,
                            TDTrajInit[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic, markerShapeVariables[filt], markerShapeVariablesNames,
                            arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                            nGraphs=len(np.unique(TDGraphId[filt])))
            mainPlot, _ = plotWithDifferentColorbars(f"betaMOvG", x[filt], xName, (TDBetaM/TDBetaG)[filt], r"$\beta_{M}/\beta_{G}$", r"$\beta_{M}/\beta_{G}$ vs "+ xName+"\n"+specificationLine,
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
                #print(filename)
                fig.savefig(filename, bbox_inches='tight')
            plt.close('all')
                                                       
    def myMultiRunStudy(filter, studyName, x, xName, subfolderingVariable, subfolderingVariableNames,
                        markerShapeVariables, markerShapeVariablesNames,
                        arrayForColorCoordinate=refConfMutualQ,
                        colorMapSpecifier=betaOfExtraction, colorMapSpecifierName=r'$\beta_{extr}$'):
        
        thisStudyFolder= os.path.join(analysis_path, studyName)
        nSubLists=len(subfolderingVariableNames)
        subListsLengths=np.array([len(subfolderingVariableNames[n]) for n in range(nSubLists)])

        if not os.path.exists(thisStudyFolder):
            os.makedirs(thisStudyFolder, exist_ok=True)
        else:
            delete_files_in_folder(thisStudyFolder)
            
        ao=[]
        ao2=[]
        #print(np.unique(subfolderingVariable[:,-subListsLengths[-1]:],axis=0))
        for unique_values in np.unique(subfolderingVariable[:,:-subListsLengths[-1]],axis=0):
            v = subfolderingVariable[np.all(subfolderingVariable[:, :-subListsLengths[-1]] == unique_values, axis=1), -subListsLengths[-1]:]
            v=np.unique(v,axis=0)
            for a in v:
                ao.append([*unique_values, *a])
            ao2.append(len(v))
        ao2=np.array(ao2,dtype=int)
        ao2=np.cumsum(ao2)
        ao2=ao2-1
        theseFiguresFolder=thisStudyFolder
        for i,v in enumerate(ao):
            print("I VALE",i)
            print("ao2",ao2)
            if i in [0,*((ao2+1)[:-1])]:
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
                eqAnBet.append(np.unique(beta[filt])[0])
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
                fig.savefig(filename, bbox_inches='tight')

            plt.close('all')
            
            toFit = [None]
            if "N"==xName:
                toFit=['linear']
                
            toCompare = np.logical_and(filt, np.array([True if x is not None  else False for x in kFromChi_InBetween_Scaled ]))
            if "N" in xName and len(np.unique(N[toCompare]))>=nNsToConsiderForSubFit:
                tempFilt=filt
                filt=np.logical_and.reduce([filt, N>=np.sort(np.unique(N[toCompare]))[-nNsToConsiderForSubFit]])
                
                mainPlot, fitData = plotWithDifferentColorbars(f"k_log_{nNsToConsiderForSubFit}LargerNs", x[filt], xName, minusLnKFromChi[filt], "-ln(k)", "Transition rate computed from single TI and "+r"$\chi$ vs "+ xName+"\n"+specificationLine,
                    trajsExtremesInitID[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic,
                    markerShapeVariables[filt], markerShapeVariablesNames,
                     arrayForColorCoordinate[filt], colorMapSpecifier=colorMapSpecifier[filt],
                    nGraphs=len(np.unique(graphID[filt])), fitTypes=toFit, colorMapSpecifierName=colorMapSpecifierName)
                
                if fitData is not None:
                        barrierBet1lN.append(np.unique(beta[filt])[0])
                        barrierBetBarr1lN.append(fitData[0])
                        barrierBetBarrErr1lN.append(fitData[2])
               
                mainPlot, fitData = plotWithDifferentColorbars(f"k2_log_{nNsToConsiderForSubFit}LargerNs", x[filt], xName, minusLnKFromChi_2[filt], "-ln(k)", "Transition rate (2) computed from single TI and "+r"$\chi$ vs "+ xName+"\n"+specificationLine,
                        trajsExtremesInitID[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic,
                        markerShapeVariables[filt], markerShapeVariablesNames,
                        arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                        nGraphs=len(np.unique(graphID[filt])), fitTypes=toFit, colorMapSpecifierName=colorMapSpecifierName)
                if fitData is not None:
                        barrierBet2lN.append(np.unique(beta[filt])[0])
                        barrierBetBarr2lN.append(fitData[0])
                        barrierBetBarrErr2lN.append(fitData[2])
                
                    
                mainPlot, fitData = plotWithDifferentColorbars(f"k2_scaled_log_{nNsToConsiderForSubFit}LargerNs", x[filt], xName, minusLnKFromChi_2_scaled[filt], "-ln(k)", "Transition rate (3) computed from single TI and "+r"$\chi$ vs "+ xName +"\n"+specificationLine,
                        trajsExtremesInitID[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic,
                        markerShapeVariables[filt], markerShapeVariablesNames,
                        arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                        nGraphs=len(np.unique(graphID[filt])), fitTypes=toFit, colorMapSpecifierName=colorMapSpecifierName)
                if fitData is not None:
                        barrierBet3lN.append(np.unique(beta[filt])[0])
                        barrierBetBarr3lN.append(fitData[0])
                        barrierBetBarrErr3lN.append(fitData[2])
                
                
                filt=tempFilt
                
            mainPlot, fitData = plotWithDifferentColorbars(f"k_log", x[filt], xName, minusLnKFromChi[filt], "-ln(k)", "Transition rate computed from single TI and "+r"$\chi$ vs "+ xName+"\n"+specificationLine,
                    trajsExtremesInitID[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic,
                    markerShapeVariables[filt], markerShapeVariablesNames,
                     arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                    nGraphs=len(np.unique(graphID[filt])), fitTypes=toFit, colorMapSpecifierName=colorMapSpecifierName)
            if fitData is not None:
                    barrierBet1.append(np.unique(beta[filt])[0])
                    barrierBetBarr1.append(fitData[0])
                    barrierBetBarrErr1.append(fitData[2])
                
                
            mainPlot, fitData = plotWithDifferentColorbars(f"k2_log", x[filt], xName, minusLnKFromChi_2[filt], "-ln(k)",  "Transition rate (1) computed from single TI and "+r"$\chi$ vs "+ xName+"\n"+specificationLine,
                    trajsExtremesInitID[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic,
                    markerShapeVariables[filt], markerShapeVariablesNames,
                     arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                    nGraphs=len(np.unique(graphID[filt])), fitTypes=toFit, colorMapSpecifierName=colorMapSpecifierName)
            if fitData is not None:
                    barrierBet2.append(np.unique(beta[filt])[0])
                    barrierBetBarr2.append(fitData[0])
                    barrierBetBarrErr2.append(fitData[2])
                
                
            mainPlot, fitData = plotWithDifferentColorbars(f"k2_scaled_log", x[filt], xName, minusLnKFromChi_2_scaled[filt], "-ln(k)",  "Transition rate (3) computed from single TI and "+ xName +"\n"+specificationLine,
                    trajsExtremesInitID[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic,
                    markerShapeVariables[filt], markerShapeVariablesNames,
                     arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],
                    nGraphs=len(np.unique(graphID[filt])), fitTypes=toFit, colorMapSpecifierName=colorMapSpecifierName)
            if fitData is not None:
                    barrierBet3.append(np.unique(beta[filt])[0])
                    barrierBetBarr3.append(fitData[0])
                    barrierBetBarrErr3.append(fitData[2])
                
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
                
            if studyName=="StudyInT":
                toFit=['linear','quadratic','mix']
            mainPlot, _ = plotWithDifferentColorbars(f"ZfunctionAndCurve", x[filt], xName, ZFromTIBeta[filt], "Z", "Probability of having Q(s(T), "+r"$s_{out}$) $\geq$"+"Q* vs "+ xName +"\n"+specificationLine,
                    trajsExtremesInitID[filt], trajInitShortDescription_Dict, edgeColorPerInitType_Dic,
                    markerShapeVariables[filt], markerShapeVariablesNames,
                     arrayForColorCoordinate[filt],colorMapSpecifier=colorMapSpecifier[filt],  fitTypes=toFit,
                    nGraphs=len(np.unique(graphID[filt])), functionsToPlotContinuously=[functions, filters])
            
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
                fig.savefig(filename, bbox_inches='tight')

            plt.close('all')
            
            if xName=="N" and (i in ao2):       
                print("SO ENTRATOOO")         
                theseFiguresFolder=theseFiguresFolder[:theseFiguresFolder.rfind("/")]
                barrierBetaAEdgeColors= np.full(len(barrierBetBarrErr1),0)
                
                
                theoreticalX, theoreticalY0, theoreticalY1=analyticDataF2(barrierBet1, r"$\beta$")
                mainPlot, _ = plotWithDifferentColorbars(f"betTentativeBarrier_FromFit1", barrierBet1, r"$\beta$", barrierBetBarr1, "Tentative "+ r"$\beta\delta$f", "Tentative free energy barrier (1) "+ r"(-ln(k)/ N)"+", vs "+ xName +"\n",
                    barrierBetaAEdgeColors, typeOfFitInN_Dict, edgeColorPerTypeOfFitInIn_Dic,
                    np.full(len(barrierBet1), "fit in N"), ["from"],
                    np.full(len(barrierBet1), 1), yerr= barrierBetBarrErr1,
                    theoreticalX=theoreticalX, theoreticalY=theoreticalY1)
                
                barrierBetaAEdgeColors= np.full(len(barrierBetBarrErr2),0)                
                theoreticalX, theoreticalY0, theoreticalY1=analyticDataF2(barrierBet2, r"$\beta$")
                
                mainPlot, _ = plotWithDifferentColorbars(f"betTentativeBarrier_FromFit2", barrierBet2, r"$\beta$", barrierBetBarr2, "Tentative "+ r"$\beta\delta$f", "Tentative free energy barrier (2) "+ r"(-ln(k)/ N)"+", vs "+ xName +"\n",
                    barrierBetaAEdgeColors, typeOfFitInN_Dict, edgeColorPerTypeOfFitInIn_Dic,
                    np.full(len(barrierBet2), "fit in N"), ["from"],
                    np.full(len(barrierBet2), 1), yerr= barrierBetBarrErr2,
                    theoreticalX=theoreticalX, theoreticalY=theoreticalY1)
                
                barrierBetaAEdgeColors= np.full(len(barrierBetBarrErr3),0)                
                theoreticalX, theoreticalY0, theoreticalY1=analyticDataF2(barrierBet3, r"$\beta$")
                
                mainPlot, _ = plotWithDifferentColorbars(f"betTentativeBarrier_FromFit3", barrierBet3, r"$\beta$", barrierBetBarr3, "Tentative "+ r"$\beta\delta$f", "Tentative free energy barrier (3) "+ r"(-ln(k)/ N)"+", vs "+ xName +"\n",
                    barrierBetaAEdgeColors, typeOfFitInN_Dict, edgeColorPerTypeOfFitInIn_Dic,
                    np.full(len(barrierBet3), "fit in N"), ["from"],
                    np.full(len(barrierBet3), 1), yerr= barrierBetBarrErr3,
                    theoreticalX=theoreticalX, theoreticalY=theoreticalY1)
                
                barrierBetaAEdgeColors= np.full(len(barrierBetBarrErr1lN),0)                
                theoreticalX, theoreticalY0, theoreticalY1=analyticDataF2(barrierBet1lN, r"$\beta$")
                
                mainPlot, _ = plotWithDifferentColorbars(f"betTentativeBarrier_FromFit1lNs", barrierBet1lN, r"$\beta$", barrierBetBarr1lN, "Tentative "+ r"$\beta\delta$f", "Tentative free energy barrier (1 largeNs) "+ r"(-ln(k)/ N)"+", vs "+ xName +"\n",
                    barrierBetaAEdgeColors, typeOfFitInN_Dict, edgeColorPerTypeOfFitInIn_Dic,
                    np.full(len(barrierBetBarrErr1lN), "fit in N"), ["from"],
                    np.full(len(barrierBetBarrErr1lN), 1), yerr= barrierBetBarrErr1lN,
                    theoreticalX=theoreticalX, theoreticalY=theoreticalY1)
                
                barrierBetaAEdgeColors= np.full(len(barrierBetBarrErr2lN),0)
                theoreticalX, theoreticalY0, theoreticalY1=analyticDataF2(barrierBet2lN, r"$\beta$")
                
                mainPlot, _ = plotWithDifferentColorbars(f"betTentativeBarrier_FromFit2lNs", barrierBet2lN, r"$\beta$", barrierBetBarr2lN, "Tentative "+ r"$\beta\delta$f", "Tentative free energy barrier (2 largeNs) "+ r"(-ln(k)/ N)"+", vs "+ xName +"\n",
                    barrierBetaAEdgeColors, typeOfFitInN_Dict, edgeColorPerTypeOfFitInIn_Dic,
                    np.full(len(barrierBet2lN), "fit in N"), ["from"],
                    np.full(len(barrierBet2lN), 1), yerr= barrierBetBarrErr2lN,
                    theoreticalX=theoreticalX, theoreticalY=theoreticalY1)
                
                barrierBetaAEdgeColors= np.full(len(barrierBetBarrErr3lN),0)
                theoreticalX, theoreticalY0, theoreticalY1=analyticDataF2(barrierBet3lN, r"$\beta$")
                
                mainPlot, _ = plotWithDifferentColorbars(f"betTentativeBarrier_FromFit3lNs", barrierBet3lN, r"$\beta$", barrierBetBarr3lN, "Tentative "+ r"$\beta\delta$f", "Tentative free energy barrier (3 largeNs) "+ r"(-ln(k)/ N)"+", vs "+ xName +"\n",
                    barrierBetaAEdgeColors, typeOfFitInN_Dict, edgeColorPerTypeOfFitInIn_Dic,
                    np.full(len(barrierBet3lN), "fit in N"), ["from"],
                    np.full(len(barrierBet3lN), 1), yerr= barrierBetBarrErr3lN,
                    theoreticalX=theoreticalX, theoreticalY=theoreticalY1)


            if xName=="T" and i in ao2:            
                theseFiguresFolder=theseFiguresFolder[:theseFiguresFolder.rfind("/")]    
                barrierBetaAEdgeColors= np.full(len(eqAnBet),0)
                mainPlot, _ = plotWithDifferentColorbars(f"eqAn", eqAnBet, r"$\beta$", eqAnBetAn,"Tentative "+ r"$\beta\delta$f", "Tentative free energy barrier (1) "+ r"(-ln(k)/ N)"+", vs "+ xName +"\n",
                    barrierBetaAEdgeColors, typeOfFitInN_Dict, edgeColorPerTypeOfFitInIn_Dic,
                    np.full(len(eqAnBet), "fit in T"), ["from"],
                    np.full(len(eqAnBet), 1), yerr= eqAnBetAnErr)

    
            figs = plt.get_figlabels()  # Ottieni i nomi di tutte le figure create
            for fig_name in figs:
                fig = plt.figure(fig_name)
                filename = os.path.join(theseFiguresFolder, f'{fig_name}.png')
                print(filename)
                fig.savefig(filename, bbox_inches='tight')
            plt.close('all')
                    


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
        #runGroupFilter=np.logical_and.reduce([runGroupFilter,fieldSigma==0.,normalizedQstar==0.6,N>80])
        
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
            #print(analyBarr)
        else:
            analyBet=None
            analyBetBarr=None
            analyBarr=None
            
        print("E ORA",analyBarr)
        
        #"""
        (TDN, TDTrajInit, TDT, TDBetOfEx, TDFirstConfIndex, TDSecondConfIndex, TDGraphId,
        TDFieldType, TDFieldReali, TDFieldSigma, TDHext,
        TDHout, TDHin, TDnQstar, TDBetaM, TDBetaG, TDBetaL, TDZmax) = thermodynamicIntegration(runGroupFilter, analysis_path)
        

        myTDStudy(TDN, "N", np.asarray(list(zip(TDnQstar, TDFieldType, TDFieldSigma))),
                            [["Qstar"], ["fieldType", r"$\sigma$"]],
                            np.array(list(zip( TDGraphId, TDT))),
                            [r"graphID","T"],
                            colorMapSpecifier=TDN,
                            arrayForColorCoordinate=TDT
                            )
        myTDStudy(TDT, "T", np.asarray(list(zip(TDnQstar, TDFieldType, TDFieldSigma))),
                            [["Qstar", "fieldType", r"$\sigma$"]],
                            np.array(list(zip( TDGraphId))),
                            [r"graphID"],
                            colorMapSpecifier=TDN,
                            arrayForColorCoordinate=TDT
                            )
        #""" 
        
    
        if len(np.unique(N[runGroupFilter]))>2: 
            myMultiRunStudy(runGroupFilter, "StudyInNProva", N, "N",
                            np.asarray(list(zip(h_ext, fieldType, fieldSigma,
                                                normalizedQstar, h_in, h_out,
                                                discretizedRescaledBetas_M))),
                            [["Hext","fieldType","sigma"], ["Qstar","Hin","Hout"],["r2beta"]],
                            np.array(list(zip( beta, graphID, rescaledBetas_M))),
                            [r"beta","graphID",r"r1beta"],
                            colorMapSpecifier=np.full(len(normalizedQstar),"nan"),
                            arrayForColorCoordinate=refConfMutualQ)
            

        
        if len(np.unique(N[runGroupFilter]))>2: 
            myMultiRunStudy(runGroupFilter, "StudyInN", N, "N",
                            np.asarray(list(zip(h_ext, fieldType, fieldSigma,
                                                normalizedQstar, h_in, h_out,
                                                beta))),
                            [["Hext","fieldType","sigma"],
                             ["Qstar","Hin","Hout"],
                             ["beta"]],
                            np.array(list(zip( graphID, fieldRealization,T))),
                            ["graph", "r","T"],)

                                
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
        
        if len(np.unique(T[runGroupFilter]))>2:
            myMultiRunStudy(runGroupFilter,"StudyInT", T,  "T",
                            np.array(list(zip(N,h_ext, normalizedQstar, h_in, h_out,beta))), [["N","Hext"],["Qstar","Hin","Hout"],["beta"]],
                            np.array(list(zip( graphID))), [ "graphID"])
            
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
    
        if len(np.unique(rescaledBetas_G[runGroupFilter]/beta[runGroupFilter]))>2:
            myMultiRunStudy(runGroupFilter,"StudyInBetaOverBetaG", rescaledBetas_G, r"$\beta$  $\frac{\langle \beta_{max} \rangle_g}{\beta_{max}}$",
                            np.array(list(zip(N, normalizedQstar, fieldType, fieldSigma))),
                            [["N", "Qstar", "fieldType", r"$\sigma$"]],
                            np.array(list(zip(graphID, fieldRealization, T))),
                            ["graphID", "r", "T"])
                          
        if len(np.unique(rescaledBetas_L[runGroupFilter]/beta[runGroupFilter]))>2:
            myMultiRunStudy(runGroupFilter,"StudyInBetaOverBetaL", rescaledBetas_L, r"$\beta$  $\frac{\langle \beta_{max} \rangle_g}{\beta_{max}}$",
                            np.array(list(zip(N, normalizedQstar, fieldType, fieldSigma))),
                            [["N", "Qstar", "fieldType", r"$\sigma$"]],
                            np.array(list(zip(graphID, fieldRealization, T))),
                            ["graphID", "r", "T"])
        

        if len(np.unique(fieldSigma[runGroupFilter]))>1:
                myMultiRunStudy(runGroupFilter,"StudyInFieldSigma", fieldSigma, r"fieldSigma",
                                np.array(list(zip(N, T))), [["N", "T"]],
                                np.array(list(zip(beta, graphID, fieldRealization))), ["field"])
     
    