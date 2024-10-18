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
        preliminaryFilter = np.full_like(x, True)
    best_indices = {}
    filterToReturn = np.full_like(x, False)
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

nameOfFoldersContainingGraphs = ["fPosJ"
                               ]

def findFoldersWithString(parent_dir, target_strings):
    result = []
    # Funzione ricorsiva per cercare le cartelle
    def search_in_subfolders(directory, livello=1):
        if livello > 10:
            return
        for root, dirs, _ in os.walk(directory):
            rightLevelReached=False
            for dir_name in dirs:
                full_path = os.path.join(root, dir_name)
                # Controlla se il nome della cartella corrente è "stdMCs" o "PathsMCs"
                if any(folder in dir_name for folder in nameOfFoldersContainingGraphs):
                    # Cerca le cartelle che contengono "_run" nel loro nome
                    rightLevelReached=True
                    for subdir in os.listdir(full_path):
                        if all(string in os.path.join(full_path, subdir) for string in target_strings):
                            result.append(os.path.join(full_path, subdir))
                
            if rightLevelReached:
                return  # Evita di cercare ancora più in profondità
                
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

        print("analyzing", item['configuration']['ID'])  #decommentare per controllare quando c'è un intoppo
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
    h_ext = np.array(h_ext, dtype=np.float64)

    fieldTypeDictionary ={"2":"gauss", "1":"bernoulli", "nan":"noField"}
    fieldType =  [fieldTypeDictionary[value] for value in fieldType]
    fieldType = np.array(fieldType)
    fieldMean = [float(value) if (value != "infty" and value!="nan") else ( 0.) for value in fieldMean]
    fieldMean = np.array(fieldMean, dtype=np.float64)
    fieldSigma = [float(value) if (value != "infty" and value!="nan") else (0.) for value in fieldSigma]
    fieldSigma = np.array(fieldSigma, dtype=np.float64)
    fieldRealization = [value if (value != "infty" and value!="nan") else (0) for value in fieldRealization]
    fieldRealization = np.array(fieldRealization)

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
    rescaledBetas = np.full_like(N, np.nan, dtype=np.float64) # l'esistenza di questo array è una sconfitta
    rescaledBetas2 = np.full_like(N, np.nan, dtype=np.float64) # l'esistenza di questo array è una sconfitta
    rescaledBetas3 = np.full_like(N, np.nan, dtype=np.float64) # l'esistenza di questo array è una sconfitta
    rescaledBetas4 = np.full_like(N, np.nan, dtype=np.float64) # l'esistenza di questo array è una sconfitta
    betaMax = np.full_like(N, np.nan, dtype=np.float64) # l'esistenza di questo array è una sconfitta
    averageBetaMax = np.full_like(N, np.nan, dtype=np.float64) # l'esistenza di questo array è una sconfitta
    kFromChi = np.full_like(N, np.nan, dtype=np.float64)
    kFromChi_InBetween = np.full_like(N, np.nan, dtype=np.float64)
    kFromChi_InBetween_Scaled = np.full_like(N, np.nan, dtype=np.float64)
    tentativeBarrier = np.full_like(N, np.nan, dtype=np.float64)
    tentativeBarrier_2 = np.full_like(N, np.nan, dtype=np.float64)
    
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

    qDist = np.array(qDist, dtype=np.float64)
    qDistStdErr = np.array(qDistStdErr, dtype=np.float64)
    
    stMC_N = np.array(stMC_N, dtype=np.int16)
    stMC_beta  = np.array(stMC_beta, dtype=np.float64)
    stMC_TIbeta  = np.array(stMC_TIbeta, dtype=np.float64)
    stMC_MC  = np.array(stMC_MC, dtype=np.int64)
    stMC_Hext  = np.array(stMC_Hext, dtype=np.float64)
    stMC_Hout = np.array(stMC_Hout, dtype=np.float64)
    stMC_Qstar = np.array(stMC_Qstar, dtype=np.int16)
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
    stMC_fieldRealization = np.array(stMC_fieldRealization)
    
    normalizedRefConfMutualQ = refConfMutualQ/N

    annealingShortDescription_Dic= {70: "Random", 71: "Ref 12", 72: "Ref 21", 73: "Annealing", 74: "Annealing", 740: "AnnealingF"}
    edgeColorPerInitType_Dic={ 70: "lightGreen", 71: "black", 72: "purple", 73: "orange", 74: "orange", 740: "red"}

    Zdict = {}

    def thermodynamicIntegration(filt):

        #specifying graph in 2 cycle
        for sim_N, sim_Hext in set(zip(N[filt], h_ext[filt])):
            TIFilt = np.logical_and.reduce([N==sim_N, h_ext==sim_Hext, filt])
            st_TIFilt = np.logical_and.reduce([stMC_N==sim_N, stMC_Hext==sim_Hext])
            for sim_fieldType, sim_fieldSigma in set(zip(fieldType[TIFilt], fieldSigma[TIFilt])):
                TIFilt1 = np.logical_and(TIFilt, np.logical_and.reduce([fieldType==sim_fieldType, fieldSigma==sim_fieldSigma]))
                st_TIFilt1 = np.logical_and(st_TIFilt, np.logical_and.reduce([stMC_fieldType==sim_fieldType, stMC_fieldSigma==sim_fieldSigma]))
                #specifying configurations
                for sim_betOfEx, sim_firstConfIndex, sim_secondConfIndex, sim_Qif in set(zip(betaOfExtraction[TIFilt1],firstConfigurationIndex[TIFilt1], secondConfigurationIndex[TIFilt1], refConfMutualQ[TIFilt1])):
                    TIFilt2 = np.logical_and(TIFilt1, np.logical_and.reduce([betaOfExtraction==sim_betOfEx, firstConfigurationIndex==sim_firstConfIndex, secondConfigurationIndex==sim_secondConfIndex, refConfMutualQ==sim_Qif]))
                    st_TIFilt2 = np.logical_and(st_TIFilt1, np.logical_and.reduce([stMC_betaOfExtraction==sim_betOfEx, stMC_configurationIndex==sim_secondConfIndex]))
                    #specifying stochastic measure parameters
                    for sim_Hin, sim_Hout, sim_Qstar in set(zip(h_in[TIFilt2], h_out[TIFilt2], Qstar[TIFilt2])):
                        TIFilt3 = np.logical_and(TIFilt2, np.logical_and.reduce([h_in==sim_Hin, h_out==sim_Hout, Qstar==sim_Qstar]))
                        st_TIFilt3 = np.logical_and(st_TIFilt2, np.logical_and.reduce([stMC_Hout==sim_Hout, stMC_Qstar==sim_Qstar]))

                        betaMaxOverRealizations=0.
                        betaMaxOverRealizationsCounter=0
                        betaMax2OverRealizations=0.
                        betaMax2OverRealizationsCounter=0
                        betaLOverRealizations=0.
                        betaLOverRealizationsCounter=0
                        betaGOverRealizations=0.
                        betaGOverRealizationsCounter=0
                        for sim_graphID, sim_fieldRealization in set(zip(graphID[TIFilt3], fieldRealization[TIFilt3])):
                            TIFilt4 = np.logical_and(TIFilt3, np.logical_and.reduce([graphID==sim_graphID, fieldRealization==sim_fieldRealization]))
                            st_TIFilt4 = np.logical_and(st_TIFilt3, np.logical_and.reduce([stMC_graphID==sim_graphID, stMC_fieldRealization==sim_fieldRealization]))

                            stdMCsBetas_forThisTDSetting, stdMCsTIBetas_forThisTDSetting, stdMCsMC_forThisTDSetting, filterForStdMCs_forThisTDSetting = getUniqueXAndYZAccordingToZ(stMC_beta, stMC_TIbeta, stMC_MC, preliminaryFilter=st_TIFilt4)
                            
                            #specifico il tempo
                            betaLForThisRealization=0.
                            betaGForThisRealization=0.
                            betaMaxForThisRealization=0.
                            betaMaxForThisRealizationCounter=0
                            betaMax2ForThisRealization=0.
                            betaMax2ForThisRealizationCounter=0
                            for sim_T, sim_trajInit in set(zip(T[TIFilt4], trajsExtremesInitID[TIFilt4])):
                                pathMCFilt_forThisTAndInit = np.logical_and.reduce([TIFilt4, T==sim_T, trajsExtremesInitID==sim_trajInit])
                            
                                
                                if (0. not in stdMCsBetas_forThisTDSetting and 0. not in beta[pathMCFilt_forThisTAndInit]):
                                    continue
                                
                                maxPathsMCsBeta = np.nanmax(beta[pathMCFilt_forThisTAndInit])
                                stdMC_filtForThisTAndInit = np.logical_and(filterForStdMCs_forThisTDSetting, stMC_beta<maxPathsMCsBeta)
                                
                                pathMCBetas_forThisTAndInit, pathMCTIs_forThisTAndInit, pathMCMCs_forThisTAndInit, pathMCFilter_forThisTAndInit = getUniqueXAndYZAccordingToZ(beta, TIbeta, lastMeasureMC, preliminaryFilter=pathMCFilt_forThisTAndInit)

                                smallestPathsMCBetaToConsider = np.nanmin(pathMCBetas_forThisTAndInit)
                                for i, stMCbeta in enumerate(stMC_beta[stdMC_filtForThisTAndInit]):
                                    if stMCbeta in pathMCBetas_forThisTAndInit:
                                        if abs(stMC_beta[stdMC_filtForThisTAndInit][i]-pathMCTIs_forThisTAndInit[pathMCBetas_forThisTAndInit==stMCbeta])<=0:#filteredStdMCsTIBetasForThisT[0]/40.:
                                            smallestPathsMCBetaToConsider = stMCbeta
                                            break
                                        
                                stdMC_filtForThisTAndInit_used = np.logical_and(stdMC_filtForThisTAndInit, stMC_beta < smallestPathsMCBetaToConsider)
                                pathsMC_filtForThisTAndInit_used = np.logical_and(pathMCFilter_forThisTAndInit, beta >= smallestPathsMCBetaToConsider)
                                temp = np.concatenate([stMC_beta[stdMC_filtForThisTAndInit_used], beta[pathsMC_filtForThisTAndInit_used]])

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
                                
                                f_interp = interpolate.InterpolatedUnivariateSpline(TIx, TIy, k=4)

                                p_up_0 = (sim_N+sim_Qif)/(2.*sim_N)
                                p_up_t = 0.5*(1.+(2.*p_up_0-1.)*np.exp(-2.*sim_T))

                                ZAtBet0 =0.
                                for this_q_star in range(sim_Qstar, sim_N+1, 2):
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
                                    
                                TIfunction= integral_to_x
                                Zfunction= exp_integral_to_x
                                betaMax = minimize_scalar(lambda z:-Zfunction(z), bounds=(np.nanmin(TIx), np.nanmax(TIx))).x
                                #beta_l = 
                                
                                levelToAdd = {}
                                levelToAdd['TIfunction'] = TIfunction
                                levelToAdd['Zfunction'] = Zfunction
                                levelToAdd['betaMax'] = betaMax

                                graphPath=findFoldersWithString('../../Data', [f'/graph{sim_graphID}'])
                                
                                if len(graphPath)>1:
                                    print("Errore, piu di un grafo trovato")
                                    print(graphPath)
                                    return None
                                graphPath = graphPath[0]

                                graphAnalysisJsonPath= os.path.join(graphPath,'graphAnalysis/graphData.json')
                                if os.path.exists(graphAnalysisJsonPath):
                                    with open(graphAnalysisJsonPath, 'r') as file:
                                        graphData = json.load(file)
                                    print(graphData)
                                else:
                                    print("nun ce staaa")
                                betaL=graphData['beta_c']['localApproach']
                                levelToAdd['beta_l'] = betaL
                                betaG=graphData['beta_c']['globalApproach']
                                levelToAdd['beta_g'] = betaG
                                
                                TIdata={}
                                TIdata['usedStMCsFilter'] = stdMC_filtForThisTAndInit_used
                                TIdata['usedPathsMCsFilter'] = pathsMC_filtForThisTAndInit_used
                                TIdata['unusedStMCsFilter'] = stdMC_filtForThisTAndInit_unused
                                TIdata['unusedPathsMCsFilter'] = pathsMC_filtForThisTAndInit_unused
                                levelToAdd['TIdata'] = TIdata
                                
                                
                                #addedLevel = addLevelOnNestedDictionary(filtDict, [(sim_N, sim_graphID, sim_Hext), (sim_fieldType, sim_fieldSigma, sim_fieldRealization), (sim_betOfEx, sim_firstConfIndex, sim_secondConfIndex), (sim_Hin, sim_Hout, sim_Qstar), (sim_T, sim_trajInit)],
                                #                levelToAdd)
                                
                                indices = np.where(pathsMC_filtForThisTAndInit_used)[0]
                                for index in indices:
                                    ZFromTIBeta[index] = Zfunction(beta[index])
                                    kFromChi[index] = ZFromTIBeta[index] * chi_m[index]
                                    kFromChi_InBetween[index] = ZFromTIBeta[index] * chi_m2[index]
                                    kFromChi_InBetween_Scaled[index] = kFromChi_InBetween[index]/scale2[index]
                                    tentativeBarrier[index] = -np.log(kFromChi[index])/(beta[index]*N[index])
                                    tentativeBarrier_2[index] = -np.log(kFromChi_InBetween_Scaled[index])/(beta[index]*N[index])
                                betaLForThisRealization = betaL
                                betaGForThisRealization = betaG
                                tentativeBarrier[indices]-=np.nanmin(tentativeBarrier[indices])
                                tentativeBarrier_2[indices]-=np.nanmin(tentativeBarrier_2[indices])
                                #f_interp =
                                
                                #levelToAdd['betaMax2'] = 
                                addedLevel = addLevelOnNestedDictionary(Zdict, [(sim_N, sim_graphID, sim_Hext), (sim_fieldType, sim_fieldSigma, sim_fieldRealization), (sim_betOfEx, sim_firstConfIndex, sim_secondConfIndex), (sim_Hin, sim_Hout, sim_Qstar), (sim_T, sim_trajInit)],
                                                levelToAdd)
                                
                                sort = np.argsort(beta[pathsMC_filtForThisTAndInit_used])
                                
                                #f_interp = interpolate.InterpolatedUnivariateSpline(beta[pathsMC_filtForThisTAndInit_used][sort], tentativeBarrier_2[pathsMC_filtForThisTAndInit_used][sort], k=3)
                                #tentativeBarrier_2[indices]-= f_interp(betaG)
                                
                                
                                nonNanEntries= ~np.isnan(tentativeBarrier[pathsMC_filtForThisTAndInit_used][sort])
                                if np.sum(nonNanEntries)<2:
                                    print("troppi pochi punti in grafo ", sim_graphID)
                                else:
                                    print("abbastanza punti in grafo ", sim_graphID)
                                    f_interp = interpolate.InterpolatedUnivariateSpline(beta[pathsMC_filtForThisTAndInit_used][sort][nonNanEntries], tentativeBarrier[pathsMC_filtForThisTAndInit_used][sort][nonNanEntries], k=2)
                                    #tentativeBarrier[indices]-= f_interp(betaL)
                                                                
                                betaMaxForThisRealizationCounter+=1
                                betaMaxForThisRealization+=betaMax
                                    

                            if betaMaxForThisRealizationCounter>0:
                                betaMaxOverRealizations += betaMaxForThisRealization/betaMaxForThisRealizationCounter
                                betaMaxOverRealizationsCounter+=1
                                
                            betaLOverRealizations += betaLForThisRealization
                            betaLOverRealizationsCounter+=1
                            betaGOverRealizations += betaGForThisRealization
                            betaGOverRealizationsCounter+=1


      
                        if betaMaxOverRealizationsCounter>0:
                            betaMaxOverRealizations /= betaMaxOverRealizationsCounter
                        if betaLOverRealizationsCounter>0:
                            betaLOverRealizations /= betaLOverRealizationsCounter
                        if betaGOverRealizationsCounter>0:
                            betaGOverRealizations /= betaGOverRealizationsCounter
                        
                        for sim_graphID, sim_fieldRealization in set(zip(graphID[TIFilt3], fieldRealization[TIFilt3])):
                            TIFilt4 = np.logical_and(TIFilt3, np.logical_and.reduce([graphID==sim_graphID, fieldRealization==sim_fieldRealization]))
                            for sim_T, sim_trajInit in set(zip(T[TIFilt4],trajsExtremesInitID[TIFilt4])):
                                TIFilt_forThisTAndInit = np.logical_and.reduce([TIFilt4, T==sim_T, trajsExtremesInitID==sim_trajInit])
                                level = getLevelFromNestedStructureAndKeyName(Zdict, [(sim_N, sim_graphID, sim_Hext), (sim_fieldType, sim_fieldSigma, sim_fieldRealization), (sim_betOfEx, sim_firstConfIndex, sim_secondConfIndex), (sim_Hin, sim_Hout, sim_Qstar), (sim_T, sim_trajInit)], 'Zfunction')
                                if level is not None:
                                    originalZfunction = level['Zfunction']
                                    thisCurveBetaMax = level['betaMax']
                                    thisCurveBetaL = level['beta_l']
                                    thisCurveBetaG = level['beta_g']
                                    rescaledBetas[TIFilt_forThisTAndInit] = beta[TIFilt_forThisTAndInit]*betaMaxOverRealizations/thisCurveBetaMax
                                    rescaledBetas2[TIFilt_forThisTAndInit] = beta[TIFilt_forThisTAndInit]/thisCurveBetaL
                                    rescaledBetas3[TIFilt_forThisTAndInit] = beta[TIFilt_forThisTAndInit]/thisCurveBetaG
                                    def rescaledZfunction(bet, numBet=betaMaxOverRealizations, denBet=thisCurveBetaMax, function=originalZfunction):
                                        return function(bet*denBet/numBet)
                                    level['rescaledZfunction_Max']=rescaledZfunction
                                    def rescaledZfunction(bet, numBet=1., denBet=thisCurveBetaG, function=originalZfunction):
                                        return function(bet*denBet/numBet)
                                    level['rescaledZfunction_g']=rescaledZfunction
                                    def rescaledZfunction(bet, numBet=1., denBet=thisCurveBetaL, function=originalZfunction):
                                        return function(bet*denBet/numBet)
                                    level['rescaledZfunction_l']=rescaledZfunction
                            
                        
    def myMultiRunStudy(filter, studyName, x, xName, subfolderingVariable, subfolderingVariableNames, markerShapeVariables, markerShapeVariablesNames):
        
        file_path = "data.txt"

        # Carica i dati dal file di testo
        data = np.loadtxt(file_path)

        # Estrai le colonne x e y dai dati
        myX = data[:, 0]  # Prima colonna
        myY = data[:, 1]  # Seconda colonna
        myX=np.asarray(myX)
        myY=np.asarray(myY)
        filt=np.logical_and(myX<1.01, myY>0)
        myX=myX[filt]
        myY=myY[filt]
        
        thisStudyFolder= os.path.join(analysis_path, studyName)

        if not os.path.exists(thisStudyFolder):
            os.makedirs(thisStudyFolder, exist_ok=True)
        else:
            delete_files_in_folder(thisStudyFolder)
        
        for v in np.unique(subfolderingVariable, axis=0):
            subFolderingFilter = []
            if subfolderingVariable.ndim==1:
                subFolderingFilter = (subfolderingVariable==v)
            else: 
                subFolderingFilter= np.all(subfolderingVariable == v, axis=1)
            filt = np.logical_and.reduce([filter,
                                        subFolderingFilter
                                        ])

            if len(np.unique(x[filt]))<2:
                continue

            theseFiguresFolder = os.path.join(
                thisStudyFolder, 
                "{}_{}".format(
                    ''.join([name.replace('\\', '').replace('$', '').capitalize() for name in subfolderingVariableNames]),
                    '_'.join([str(item) for item in v])
                )
            )

            specificationLine = "at "+ ', '.join([f"{k}={v}" for k, v in zip(subfolderingVariableNames, v)]).replace('star', '*')
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
            TIbetaMainPlot = plotWithDifferentColorbars(f"TIbeta", x[filt], xName, TIbeta[filt], "U", "Quantity for thermodynamic integration vs "+ xName +"\n"+specificationLine,
            betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
            trajsExtremesInitID[filt], annealingShortDescription_Dic, edgeColorPerInitType_Dic,
            markerShapeVariables[filt], markerShapeVariablesNames,
            nGraphs=len(np.unique(graphID[filt])), 
            additionalMarkerTypes=additional
            )

            mainPlot = plotWithDifferentColorbars(f"meanBarrier", x[filt], xName, meanBarrier[filt], "barrier", "mean barrier vs "+ xName+"\n"+specificationLine,
                        betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                        trajsExtremesInitID[filt], annealingShortDescription_Dic, edgeColorPerInitType_Dic, markerShapeVariables[filt], markerShapeVariablesNames,
                        yerr=stdDevBarrier[filt], nGraphs=len(np.unique(graphID[filt])))
            
            mainPlot = plotWithDifferentColorbars(f"avEnergy", x[filt], xName, avEnergy[filt], "energy", "Mean average energy over trajectory vs "+ xName +"\n"+specificationLine,
                    betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                    trajsExtremesInitID[filt], annealingShortDescription_Dic, edgeColorPerInitType_Dic, markerShapeVariables[filt], markerShapeVariablesNames,
                    yerr=avEnergyStdErr[filt], nGraphs=len(np.unique(graphID[filt])))
            
            mainPlot = plotWithDifferentColorbars(f"muAvEnergy", x[filt], xName, muAvEnergy[filt], r"$\mu$", r"$\mu$"+" vs "+ xName+"\n"+specificationLine,
                            betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                            trajsExtremesInitID[filt], annealingShortDescription_Dic, edgeColorPerInitType_Dic, markerShapeVariables[filt], markerShapeVariablesNames,
                            yerr=stdDevBarrier[filt], fitType='', xscale='', 
                            yscale='log', 
                            nGraphs=len(np.unique(graphID[filt])))

            mainPlot = plotWithDifferentColorbars(f"nJumps", x[filt], xName, nJumps[filt], "# jumps", "Mean number of jumps per spin over trajectory vs "+ xName +"\n"+specificationLine,
                    betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                    trajsExtremesInitID[filt], annealingShortDescription_Dic, edgeColorPerInitType_Dic, markerShapeVariables[filt], markerShapeVariablesNames,
                    yerr=nJumpsStdErr[filt], nGraphs=len(np.unique(graphID[filt])))

            mainPlot = plotWithDifferentColorbars(f"effFlipRate", x[filt], xName, effectiveFlipRate[filt], "r", "Effective flip rate over trajectory (#jumps/T) vs "+ xName +"\n"+specificationLine,
                    betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                    trajsExtremesInitID[filt], annealingShortDescription_Dic, edgeColorPerInitType_Dic, markerShapeVariables[filt], markerShapeVariablesNames,
                    yerr=effectiveFlipRateError[filt], nGraphs=len(np.unique(graphID[filt])))

            mainPlot = plotWithDifferentColorbars(f"deltaNJumps", x[filt], xName, deltaNJumps[filt], r"$\delta$", "Spins number of jumps over trajectory stdDev (over sites) vs "+ xName +"\n"+specificationLine,
                    betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                    trajsExtremesInitID[filt], annealingShortDescription_Dic, edgeColorPerInitType_Dic, markerShapeVariables[filt], markerShapeVariablesNames,
                    yerr=deltaNJumpsStdErr[filt], nGraphs=len(np.unique(graphID[filt])))
                
            mainPlot = plotWithDifferentColorbars(f"deltaNOverAvJumps", x[filt], xName, deltaNJumps[filt]**2/nJumps[filt], "ratio", r"($\delta$"+"#jumps)^2/(#jumps)" +" vs "+ xName +"\n"+specificationLine,
                    betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                    trajsExtremesInitID[filt], annealingShortDescription_Dic, edgeColorPerInitType_Dic, markerShapeVariables[filt], markerShapeVariablesNames,
                    yerr=deltaNJumpsStdErr[filt], nGraphs=len(np.unique(graphID[filt])))
                
            mainPlot = plotWithDifferentColorbars(f"qDist", x[filt], xName, qDist[filt], "distance", "Average distance from stfwd path between reference configurations over trajectory vs "+xName +"\n"+specificationLine,
                    betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                    trajsExtremesInitID[filt], annealingShortDescription_Dic, edgeColorPerInitType_Dic, markerShapeVariables[filt], markerShapeVariablesNames,
                    yerr=qDistStdErr[filt], nGraphs=len(np.unique(graphID[filt])))

            mainPlot = plotWithDifferentColorbars(f"tau", x[filt], xName, chi_tau[filt], r"$\tau_{trans}$", r"transient time $\tau_{trans}$ vs "+ xName+"\n"+specificationLine,
                        betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                        trajsExtremesInitID[filt], annealingShortDescription_Dic, edgeColorPerInitType_Dic, markerShapeVariables[filt], markerShapeVariablesNames,
                        yerr=stdDevBarrier[filt], nGraphs=len(np.unique(graphID[filt])))
                
            mainPlot = plotWithDifferentColorbars(f"realTime", x[filt], xName, realTime[filt], "computer time (seconds)", "Seconds required to perform 10^5 mc sweeps vs "+ xName+"\n"+specificationLine,
                        betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                        trajsExtremesInitID[filt], annealingShortDescription_Dic, edgeColorPerInitType_Dic, markerShapeVariables[filt], markerShapeVariablesNames,
                        yerr=realTimeErr[filt], nGraphs=len(np.unique(graphID[filt])))

            mainPlot = plotWithDifferentColorbars(f"TIhout", x[filt], xName, TIhout[filt], "L", "Quantity for thermodynamic integration vs "+ xName +"\n"+specificationLine,
                    betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                    trajsExtremesInitID[filt], annealingShortDescription_Dic, edgeColorPerInitType_Dic,
                    markerShapeVariables[filt], markerShapeVariablesNames,
                    nGraphs=len(np.unique(graphID[filt])))
                
            mainPlot = plotWithDifferentColorbars(f"TIQstar", x[filt], xName, TIQstar[filt], "L", "Quantity for thermodynamic integration vs "+ xName +"\n"+specificationLine,
                    betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                    trajsExtremesInitID[filt], annealingShortDescription_Dic, edgeColorPerInitType_Dic,
                    markerShapeVariables[filt], markerShapeVariablesNames,
                    nGraphs=len(np.unique(graphID[filt])))
 
 
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
                print(filename)
                fig.savefig(filename, bbox_inches='tight')

            plt.close('all')
            
            
            mainPlot = plotWithDifferentColorbars(f"k", x[filt], xName, kFromChi[filt], "k", "Transition rate computed from single TI and "+r"$\chi$ vs "+ xName+"\n"+specificationLine,
                    betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                    trajsExtremesInitID[filt], annealingShortDescription_Dic, edgeColorPerInitType_Dic,
                    markerShapeVariables[filt], markerShapeVariablesNames,
                    nGraphs=len(np.unique(graphID[filt])), yscale='log')
            
            mainPlot = plotWithDifferentColorbars(f"k2", x[filt], xName, kFromChi_InBetween[filt], "k", "Generalized transition rate computed from single TI and "+r"$\chi$ vs "+ xName+"\n"+specificationLine,
                    betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                    trajsExtremesInitID[filt], annealingShortDescription_Dic, edgeColorPerInitType_Dic,
                    markerShapeVariables[filt], markerShapeVariablesNames,
                    nGraphs=len(np.unique(graphID[filt])), yscale='log')
            
            mainPlot = plotWithDifferentColorbars(f"k2_scaled", x[filt], xName, kFromChi_InBetween_Scaled[filt], "L", "Quantity for thermodynamic integration vs "+ xName +"\n"+specificationLine,
                    betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                    trajsExtremesInitID[filt], annealingShortDescription_Dic, edgeColorPerInitType_Dic,
                    markerShapeVariables[filt], markerShapeVariablesNames,
                    nGraphs=len(np.unique(graphID[filt])), yscale='log')
            
            mainPlot = plotWithDifferentColorbars(f"scale", x[filt], xName, scale2[filt], "f", "Projected probability of being in final cone based on linearity, vs "+ xName +"\n"+specificationLine,
                    betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                    trajsExtremesInitID[filt], annealingShortDescription_Dic, edgeColorPerInitType_Dic,
                    markerShapeVariables[filt], markerShapeVariablesNames,
                    nGraphs=len(np.unique(graphID[filt])), yscale='log')
            
            mainPlot = plotWithDifferentColorbars(f"tentativeBarrier", x[filt], xName, tentativeBarrier[filt], "Tentative "+ r"$\delta$f", "Tentative free energy barrier "+ r"(-ln(k)/N)"+", vs "+ xName +"\n"+specificationLine,
                    betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                    trajsExtremesInitID[filt], annealingShortDescription_Dic, edgeColorPerInitType_Dic,
                    markerShapeVariables[filt], markerShapeVariablesNames,
                    nGraphs=len(np.unique(graphID[filt])),
                    theoreticalX=myX, theoreticalY=myY)

            mainPlot = plotWithDifferentColorbars(f"tentativeBarrier2", x[filt], xName, tentativeBarrier_2[filt], "Tentative "+ r"$\delta$f", "Tentative free energy barrier "+ r"(-ln(k)/ N)"+", vs "+ xName +"\n"+specificationLine,
                    betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                    trajsExtremesInitID[filt], annealingShortDescription_Dic, edgeColorPerInitType_Dic,
                    markerShapeVariables[filt], markerShapeVariablesNames,
                    nGraphs=len(np.unique(graphID[filt])),
                    theoreticalX=myX, theoreticalY=myY
                    
                    )
            
            filters = []
            functions = []
            rescaledFunctions = []


            if studyName=="StudyInBeta":
                functionNameToGet = 'Zfunction'
            elif studyName=="StudyInBetaOverBetaMax":
                functionNameToGet = 'rescaledZfunction_Max'
            elif studyName=="StudyInBetaOverBetaG":
                functionNameToGet = 'rescaledZfunction_g'
            elif studyName=="StudyInBetaOverBetaL":
                functionNameToGet = 'rescaledZfunction_l'
                
                
            for sim_N, sim_graphID, sim_Hext in set(zip(N[filt], graphID[filt], h_ext[filt])):
                if (sim_N, sim_graphID, sim_Hext) not in Zdict.keys():
                    continue
                subdict1 = Zdict[(sim_N, sim_graphID, sim_Hext)]
                for sim_fieldType, sim_fieldSigma, sim_fieldRealization in set(zip(fieldType[filt], fieldSigma[filt], fieldRealization[filt])):
                    if (sim_fieldType, sim_fieldSigma, sim_fieldRealization) not in subdict1.keys():
                        continue
                    subdict2 = subdict1[(sim_fieldType, sim_fieldSigma, sim_fieldRealization)]
                    #specifying configurations
                    for sim_betOfEx, simfirst_ConfIndex, sim_secondConfIndex in set(zip(betaOfExtraction[filt],firstConfigurationIndex[filt], secondConfigurationIndex[filt])):
                        if (sim_betOfEx, simfirst_ConfIndex, sim_secondConfIndex) not in subdict2.keys():
                            continue
                        subdict3 = subdict2[(sim_betOfEx, simfirst_ConfIndex, sim_secondConfIndex)]
                        #specifying stochastic measure parameters
                        for sim_Hin, sim_Hout, sim_Qstar in set(zip(h_in[filt], h_out[filt], Qstar[filt])):
                            if (sim_Hin, sim_Hout, sim_Qstar) not in subdict3.keys():
                                continue
                            subdict4 = subdict3[(sim_Hin, sim_Hout, sim_Qstar)]
                            #specifico il tempo
                            for sim_T, sim_trajInit in set(zip(T[filt],trajsExtremesInitID[filt])):
                                if (sim_T, sim_trajInit) not in subdict4.keys():
                                    continue
                                folderingVariableFilt = np.logical_and.reduce([
                                    sim_N==N[filt], sim_graphID==graphID[filt], sim_Hext==h_ext[filt],
                                    sim_fieldType==fieldType[filt], sim_fieldSigma==fieldSigma[filt], sim_fieldRealization==fieldRealization[filt],
                                    sim_betOfEx==betaOfExtraction[filt], simfirst_ConfIndex==firstConfigurationIndex[filt], sim_secondConfIndex==secondConfigurationIndex[filt],
                                    sim_Hin==h_in[filt], sim_Hout==h_out[filt], sim_Qstar==Qstar[filt],
                                    sim_T==T[filt], sim_trajInit==trajsExtremesInitID[filt]
                                ])
                                f= subdict4[(sim_T, sim_trajInit)]['Zfunction']
                                f= subdict4[(sim_T, sim_trajInit)][functionNameToGet]
                                filters.append(folderingVariableFilt)
                                functions.append(f)

                
            mainPlot = plotWithDifferentColorbars(f"ZfunctionAndCurve", x[filt], xName, ZFromTIBeta[filt], "Z", "Probability of having Q(s(T), "+r"$s_{out}$) $\geq$"+"Q* vs "+ xName +"\n"+specificationLine,
                    betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                    trajsExtremesInitID[filt], annealingShortDescription_Dic, edgeColorPerInitType_Dic,
                    markerShapeVariables[filt], markerShapeVariablesNames,
                    nGraphs=len(np.unique(graphID[filt])), functionsToPlotContinuously=[functions, filters])
            
            mainPlot = plotWithDifferentColorbars(f"ZfunctionAndCurve_log", x[filt], xName, ZFromTIBeta[filt], "Z", "Probability of having Q(s(T), "+r"$s_{out}$) $\geq$"+"Q* vs "+ xName +"\n"+specificationLine,
                    betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                    trajsExtremesInitID[filt], annealingShortDescription_Dic, edgeColorPerInitType_Dic,
                    markerShapeVariables[filt], markerShapeVariablesNames,
                    nGraphs=len(np.unique(graphID[filt])), yscale='log', functionsToPlotContinuously=[functions, filters])
                
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
                print(filename)
                fig.savefig(filename, bbox_inches='tight')

            plt.close('all')



    selectedRunGroups_FiltersAndNames = [
        [
            np.logical_and(C==c, fPosJ==f),
            f"p2C{c}/fPosJ{f:.2f}"
        ]
        for c,f in set(zip(C,fPosJ))
        ]

    selectedRunGroups_FiltersAndNames= [runGroup for runGroup in selectedRunGroups_FiltersAndNames if runGroup[0].sum()>minNumberOfSingleRunsToDoAnAnalysis]

    for runGroupFilter, runGroupName in selectedRunGroups_FiltersAndNames:

        analysis_path = os.path.join(parentAnalysis_path, runGroupName)
        
        
        thermodynamicIntegration(runGroupFilter)
        
        if len(np.unique(beta[runGroupFilter]))>2:
            #myMultiRunStudy(runGroupFilter,"StudyInBeta", beta, r"$\beta$", np.array(list(zip(N, Qstar, fieldType, fieldSigma))), ["N", "Qstar", "fieldType", r"$\sigma$"], np.array(list(zip(graphID, fieldRealization, T))), ["graphID", "r", "T"])
            myMultiRunStudy(runGroupFilter,"StudyInBeta", beta, r"$\beta$", np.array(list(zip(N, Qstar, fieldType, fieldSigma))), ["N", "Qstar", "fieldType", r"$\sigma$"], np.array(list(zip(graphID, T))), ["graphID", "T"])
        
        if len(np.unique(rescaledBetas3[runGroupFilter]))>2:
            myMultiRunStudy(runGroupFilter,"StudyInBetaOverBetaG", rescaledBetas3, r"$\beta$  $\frac{\langle \beta_{max} \rangle_g}{\beta_{max}}$", np.array(list(zip(N, Qstar, fieldType, fieldSigma))), ["N", "Qstar", "fieldType", r"$\sigma$"], np.array(list(zip(graphID, fieldRealization, T))), ["graphID", "r", "T"])
                          
        if len(np.unique(rescaledBetas2[runGroupFilter]))>2:
            myMultiRunStudy(runGroupFilter,"StudyInBetaOverBetaL", rescaledBetas2, r"$\beta$  $\frac{\langle \beta_{max} \rangle_g}{\beta_{max}}$", np.array(list(zip(N, Qstar, fieldType, fieldSigma))), ["N", "Qstar", "fieldType", r"$\sigma$"], np.array(list(zip(graphID, fieldRealization, T))), ["graphID", "r", "T"])
                          
        if len(np.unique(rescaledBetas[runGroupFilter]))>2:
            myMultiRunStudy(runGroupFilter,"StudyInBetaOverBetaMax", rescaledBetas, r"$\beta$  $\frac{\langle \beta_{max} \rangle_g}{\beta_{max}}$", np.array(list(zip(N, Qstar, fieldType, fieldSigma))), ["N", "Qstar", "fieldType", r"$\sigma$"], np.array(list(zip(graphID, fieldRealization, T))), ["graphID", "r", "T"])


        if len(np.unique(T[runGroupFilter]))>2:
            myMultiRunStudy(runGroupFilter,"StudyInT", T,  "T", np.array(list(zip(N))), ["N"], np.array(list(zip(beta, graphID))), ["beta", "graphID"])
            
        if len(np.unique(N[runGroupFilter]))>2: 
            Qstar #così, slice-ando rispetto a N unisco i casi con q_star uguale
            normalizedQstar = Qstar/N
            #myMultiRunStudy(runGroupFilter, "StudyInN", N, "N", np.asarray(list(zip(normalizedQstar))), ["Qstar"] , np.array(list(zip(graphID, T, rescaledBetas))), ["graphID", "T",r"$\beta'$"])
            myMultiRunStudy(runGroupFilter, "StudyInN", N, "N", np.asarray(list(zip(normalizedQstar))), ["Qstar"] , np.array(list(zip( rescaledBetas))), [r"$\beta'$"])
        return

        if len(np.unique(fieldSigma[runGroupFilter]))>1:
                myMultiRunStudy(runGroupFilter,"StudyInFieldSigma", fieldSigma, r"fieldSigma", np.array(list(zip(N, T))), "N, T", np.array(list(zip(beta, graphID, fieldRealization))), "field")

        if len(np.unique((T*N)[runGroupFilter]))>2:
            myMultiRunStudy(runGroupFilter,"StudyInNT", N*T, "NT",N, "N", beta, r"beta")