from math import comb
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
from scipy.optimize import curve_fit
from uncertainties import ufloat

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

def getUniqueXAndYZAccordingToZ(x, y, criterion):
    best_indices = {}
    # Iteriamo su ogni valore unico in filteredStdMCsBetas
    for value in np.sort(np.unique(x)):
        # Trova gli indici corrispondenti a questo valore unico
        indices = np.where(x == value)[0]

        if len(indices) > 1:
            # Se ci sono più di un indice, scegli quello con il valore massimo in lastMeasureMC
            best_index = indices[np.argmax(criterion[indices])]
        else:
            # Se c'è solo un indice, lo prendiamo direttamente
            best_index = indices[0]          
        # Memorizza l'indice migliore
        best_indices[value] = best_index
    filtered_x = np.asarray(x[list(best_indices.values())])
    filtered_y = np.asarray(y[list(best_indices.values())])
    filtered_criterion = np.asarray(criterion[list(best_indices.values())])
    return filtered_x, filtered_y, filtered_criterion

def aggiungi_funzione(structure, param_tuples, x, y, TIfun, Zfun):
    current_level = structure
    for param_tuple in param_tuples:
        if param_tuple not in current_level:
            current_level[param_tuple] = {}
        current_level = current_level[param_tuple]
    # Alla fine, salviamo la funzione nel livello corrente
    current_level['TIfunction'] = TIfun
    current_level['Zfunction'] = Zfun
    current_level['TIx'] = x
    current_level['TIy'] = y

def ottieni_funzione(structure, param_tuples):
    current_level = structure
    for param_tuple in param_tuples:
        if param_tuple in current_level:
            current_level = current_level[param_tuple]
        else:
            return np.nan
    if 'Zfunction' in current_level:
        return current_level
    else:
        return np.nan

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
    chi_m = []
    chi_c = []
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
            print(item.keys())
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

        if "chiLinearFit" not in item["results"]:
            chi_tau.append("nan")
            chi_m.append("nan")
            chi_c.append("nan")
        else:
            chi_tau.append(item["results"]["chiLinearFit"]['tau'])
            chi_m.append(item["results"]["chiLinearFit"]['m'])
            chi_c.append(item["results"]["chiLinearFit"]['c'])
        
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
        print(f"Found {np.count_nonzero(simulationType == typeOfSim)} eGroups of type {typeOfSim}.\n")

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

    ZFromTIBeta = np.full_like(N, np.nan, dtype=np.float64) # l'esistenza di questo array è una sconfitta
    kFromChi = np.full_like(N, np.nan, dtype=np.float64)
    
    meanBarrier = np.array(meanBarrier, dtype=np.float64)
    stdDevBarrier = np.array(stdDevBarrier, dtype=np.float64)
    
    avEnergy = np.array(avEnergy, dtype=np.float64)
    avEnergyStdErr = np.array(avEnergyStdErr, dtype=np.float64)
    
    avEnergy/=N
    avEnergyStdErr/=N
    muAvEnergy = np.array(muAvEnergy, dtype=np.float64)
    

    nJumps =  np.array(nJumps, dtype=np.float64)
    nJumpsStdErr =  np.array(nJumpsStdErr, dtype=np.float64)
    
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
    def thermodynamicIntegrationAndPlots(filt):

        #specifying graph in 2 cycle
        for sim_N, sim_graphID, sim_Hext in set(zip(N[filt], graphID[filt], h_ext[filt])):
            TIFilt = np.logical_and.reduce([N==sim_N, graphID==sim_graphID, h_ext==sim_Hext, filt])
            st_TIFilt = np.logical_and.reduce([stMC_N==sim_N, stMC_graphID==sim_graphID, stMC_Hext==sim_Hext])
            for sim_fieldType, sim_fieldSigma, sim_fieldRealization in set(zip(fieldType[TIFilt], fieldSigma[TIFilt], fieldRealization[TIFilt])):
                TIFilt1 = np.logical_and(TIFilt, np.logical_and.reduce([fieldType==sim_fieldType, fieldSigma==sim_fieldSigma, fieldRealization==sim_fieldRealization]))
                st_TIFilt1 = np.logical_and(st_TIFilt, np.logical_and.reduce([stMC_fieldType==sim_fieldType, stMC_fieldSigma==sim_fieldSigma, stMC_fieldRealization==sim_fieldRealization]))
                #specifying configurations
                for sim_betOfEx, sim_firstConfIndex, sim_secondConfIndex, sim_Qif in set(zip(betaOfExtraction[TIFilt1],firstConfigurationIndex[TIFilt1], secondConfigurationIndex[TIFilt1], refConfMutualQ[TIFilt1])):
                    TIFilt2 = np.logical_and(TIFilt1, np.logical_and.reduce([betaOfExtraction==sim_betOfEx, firstConfigurationIndex==sim_firstConfIndex, secondConfigurationIndex==sim_secondConfIndex, refConfMutualQ==sim_Qif]))
                    st_TIFilt2 = np.logical_and(st_TIFilt1, np.logical_and.reduce([stMC_betaOfExtraction==sim_betOfEx, stMC_configurationIndex==sim_secondConfIndex]))
                    #specifying stochastic measure parameters
                    for sim_Hin, sim_Hout, sim_Qstar in set(zip(h_in[TIFilt2], h_out[TIFilt2], Qstar[TIFilt2])):
                        TIFilt3 = np.logical_and(TIFilt2, np.logical_and.reduce([h_in==sim_Hin, h_out==sim_Hout, Qstar==sim_Qstar]))
                        st_TIFilt3 = np.logical_and(st_TIFilt2, np.logical_and.reduce([stMC_Hout==sim_Hout, stMC_Qstar==sim_Qstar]))

                        filteredStdMCsBetas = stMC_beta[st_TIFilt3]
                        filteredStdMCsTIBetas = stMC_TIbeta[st_TIFilt3]
                        filteredStdMCsMC = stMC_MC[st_TIFilt3]

                        filteredStdMCsBetas, filteredStdMCsTIBetas, filteredStdMCsMC = getUniqueXAndYZAccordingToZ(filteredStdMCsBetas, filteredStdMCsTIBetas, filteredStdMCsMC)
                        #specifico il tempo
                        for sim_T, sim_trajInit in set(zip(T[TIFilt3],trajsExtremesInitID[TIFilt3])):
                            TIFilt_atThisT = np.logical_and.reduce([TIFilt3, T==sim_T, trajsExtremesInitID==sim_trajInit])
                        
                            filteredBetas = beta[TIFilt_atThisT]
                            if 0. not in filteredStdMCsBetas and 0. not in filteredBetas:
                                continue
                            filteredTIBetas = TIbeta[TIFilt_atThisT]
                            filteredLastMeasureMC = lastMeasureMC[TIFilt_atThisT]

                            minPathsMCsBeta = np.min(filteredBetas)
                            stMC_FiltForThisT = (filteredStdMCsBetas<minPathsMCsBeta)
                            filteredStdMCsBetasForThisT = filteredStdMCsBetas[stMC_FiltForThisT]
                            filteredStdMCsTIBetasForThisT = filteredStdMCsTIBetas[stMC_FiltForThisT]
                            filteredStdMCsMCForThisT = filteredStdMCsMC[stMC_FiltForThisT]
                            filteredBetas, filteredTIBetas, filteredLastMeasureMC = getUniqueXAndYZAccordingToZ(filteredBetas, filteredTIBetas, filteredLastMeasureMC)

                            if len(filteredBetas)>4:
                                TIx=np.concatenate([filteredStdMCsBetasForThisT, filteredBetas])
                                TIy=np.concatenate([filteredStdMCsTIBetasForThisT, filteredTIBetas])

                                #aggiungi controllo che stdTI del piu grande stdBeta e TI del piu piccolo Beta siano circa uguali
                                #aggiungi controllo che le beta non siano troppo spaziate
                                f_interp = interpolate.InterpolatedUnivariateSpline(TIx, TIy, k=3)

                                def ZAtBeta0(n, q_if, q_star, t):
                                    p_up_0 = (n+q_if)/(2.*n)
                                    p_up_t = 0.5*(1.+(2.*p_up_0-1.)*np.exp(-2.*t))
                                    def P_t(q):
                                        return comb(n, q)*p_up_t**((n+q)/2) * ((1.-2*p_up_t)**((n-q)/2))
                                    ZAtBet0 =0.
                                    for this_q_star in range(q_star,n+1):
                                        ZAtBet0+=P_t(this_q_star)
                                    return ZAtBet0

                                def integral_to_x(x_point, aoF=f_interp, aoTIx=TIx):
                                        if x_point<0 or x_point>aoTIx[-1]:
                                            return np.nan
                                        integral, _ = quad(aoF, aoTIx[0], x_point)
                                        return integral
                                def exp_integral_to_x(x_point, aoF=f_interp, aoTIx=TIx):
                                    return ZAtBeta0(sim_N, sim_Qif, sim_Qstar, sim_T) * np.exp(integral_to_x(x_point, aoF, aoTIx))
                                
                                TIfunction= integral_to_x
                                Zfunction= exp_integral_to_x
                                aggiungi_funzione(Zdict, [(sim_N, sim_graphID, sim_Hext), (sim_betOfEx, sim_firstConfIndex, sim_secondConfIndex), (sim_Hin, sim_Hout, sim_Qstar), (sim_T, sim_trajInit)],
                                                  TIx, TIy, TIfunction, Zfunction)
                                
                                indices = np.where(TIFilt_atThisT)[0]
                                for index in indices:
                                    ZFromTIBeta[index] = Zfunction(beta[index])
                                    kFromChi[index] = ZFromTIBeta[index] * chi_m[index]
        plt.figure("text")
        x_continuous = np.linspace(0, 1., 1000) 
        for key1 in Zdict.keys():
            for key2 in Zdict[key1]:
                for key3 in Zdict[key1][key2]:
                    for k in Zdict[key1][key2][key3]:
                        print(Zdict[key1][key2][key3][k])
                        f=Zdict[key1][key2][key3][k]['Zfunction']
                        exp_integrals_continuous = [f(xp) for xp in x_continuous]
                        plt.plot(x_continuous, exp_integrals_continuous, label=f"{key1} {key2} {key3} {k}")
        plt.legend()

    def myMultiRunStudy(filter, studyName, x, xName, subfolderingVariable, subfolderingVariableName, markerShapeVariable, markerShapeVariableName):

        if len(np.unique(x[filter]))<2:
            return
        
        thisStudyFolder= os.path.join(plotsFolder, studyName)

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

            theseFiguresFolder= os.path.join(thisStudyFolder, f"{subfolderingVariableName}{v}")
            if not os.path.exists(theseFiguresFolder):
                os.makedirs(theseFiguresFolder, exist_ok=True)
            else:
                delete_files_in_folder(theseFiguresFolder)

            specificationLine =f"at {subfolderingVariableName} = {v}"

            additional= []
            tempFilt=filt
            if xName==r"$\beta$":
                stMC_corrBetaAndQif = np.empty((len(stMC_beta), 2), dtype=object)
                for sim_N, sim_graphID, sim_betOfEx, sim_SecConfInd, sim_Qif, simQstar in set(zip(N[filt],graphID[filt], betaOfExtraction[filt], secondConfigurationIndex[filt], refConfMutualQ[filt], Qstar[filt])):
                    stMCFilt= np.logical_and.reduce([stMC_N == sim_N,
                                                     stMC_graphID==sim_graphID,
                                                     stMC_betaOfExtraction==sim_betOfEx,
                                                     stMC_configurationIndex==sim_SecConfInd,
                                                     stMC_Qstar==simQstar
                                                    ])
                    stMC_corrBetaAndQif[stMCFilt]=[sim_betOfEx, sim_Qif]
                    if stMCFilt.sum()>1:
                        additional.append(list([stMC_beta[stMCFilt], stMC_TIbeta[stMCFilt], stMC_corrBetaAndQif[stMCFilt], f"inf, {sim_graphID}"]))
                if len(additional)==0:
                    additional=None
                
                thermodynamicIntegrationAndPlots(filt)

            filt=tempFilt
            TIbetaMainPlot = plotWithDifferentColorbars(f"TIbeta", x[filt], xName, TIbeta[filt], "L", "Quantity for thermodynamic integration vs "+ xName +"\n"+specificationLine,
            betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
            trajsExtremesInitID[filt], annealingShortDescription_Dic, edgeColorPerInitType_Dic,
            markerShapeVariable[filt], markerShapeVariableName,
            nGraphs=len(np.unique(graphID[filt])), 
            additionalMarkerTypes=additional
            )

            mainPlot = plotWithDifferentColorbars(f"meanBarrier", x[filt], xName, meanBarrier[filt], "barrier", "mean barrier vs "+ xName+"\n"+specificationLine,
                        betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                        trajsExtremesInitID[filt], annealingShortDescription_Dic, edgeColorPerInitType_Dic, markerShapeVariable[filt], markerShapeVariableName,
                        yerr=stdDevBarrier[filt], nGraphs=len(np.unique(graphID[filt])))
            
            mainPlot = plotWithDifferentColorbars(f"avEnergy", x[filt], xName, avEnergy[filt], "energy", "Mean average energy over trajectory vs "+ xName +"\n"+specificationLine,
                    betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                    trajsExtremesInitID[filt], annealingShortDescription_Dic, edgeColorPerInitType_Dic, markerShapeVariable[filt], markerShapeVariableName,
                    yerr=avEnergyStdErr[filt], nGraphs=len(np.unique(graphID[filt])))
            
            mainPlot = plotWithDifferentColorbars(f"muAvEnergy", x[filt], xName, muAvEnergy[filt], r"$\mu$", r"$\mu$"+" vs "+ xName+"\n"+specificationLine,
                            betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                            trajsExtremesInitID[filt], annealingShortDescription_Dic, edgeColorPerInitType_Dic, markerShapeVariable[filt], markerShapeVariableName,
                            yerr=stdDevBarrier[filt], fitType='', xscale='', 
                            #yscale='log', 
                            nGraphs=len(np.unique(graphID[filt])))
            tempFilt=filt
            for t in np.unique(markerShapeVariable[filt]):
                filt = np.logical_and(np.all(markerShapeVariable == t, axis=markerShapeVariable.ndim-1), tempFilt)
                if len(ID[filt])>0:
                    mainPlot = plotWithDifferentColorbars(f"muAvEnergy_{markerShapeVariableName}{t}", x[filt], xName, muAvEnergy[filt], r"$\mu$", r"$\mu$"+" vs "+ xName+"\n"+specificationLine,
                            betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                            trajsExtremesInitID[filt], annealingShortDescription_Dic, edgeColorPerInitType_Dic, markerShapeVariable[filt], markerShapeVariableName,
                            yerr=stdDevBarrier[filt], fitType='', xscale='', yscale='', nGraphs=len(np.unique(graphID[filt])))
            filt=tempFilt

            mainPlot = plotWithDifferentColorbars(f"nJumps", x[filt], xName, nJumps[filt], "# jumps", "Mean number of jumps per spin over trajectory vs "+ xName +"\n"+specificationLine,
                    betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                    trajsExtremesInitID[filt], annealingShortDescription_Dic, edgeColorPerInitType_Dic, markerShapeVariable[filt], markerShapeVariableName,
                    yerr=nJumpsStdErr[filt], nGraphs=len(np.unique(graphID[filt])))

            mainPlot = plotWithDifferentColorbars(f"deltaNJumps", x[filt], xName, deltaNJumps[filt], r"$\delta$", "Spins number of jumps over trajectory stdDev (over sistes) vs "+ xName +"\n"+specificationLine,
                    betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                    trajsExtremesInitID[filt], annealingShortDescription_Dic, edgeColorPerInitType_Dic, markerShapeVariable[filt], markerShapeVariableName,
                    yerr=deltaNJumpsStdErr[filt], nGraphs=len(np.unique(graphID[filt])))
                
            mainPlot = plotWithDifferentColorbars(f"deltaNOverAvJumps", x[filt], xName, deltaNJumps[filt]**2/nJumps[filt], "ratio", r"($\delta$"+"#jumps)^2/(#jumps)" +" vs "+ xName +"\n"+specificationLine,
                    betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                    trajsExtremesInitID[filt], annealingShortDescription_Dic, edgeColorPerInitType_Dic, markerShapeVariable[filt], markerShapeVariableName,
                    yerr=deltaNJumpsStdErr[filt], nGraphs=len(np.unique(graphID[filt])))
                
            mainPlot = plotWithDifferentColorbars(f"qDist", x[filt], xName, qDist[filt], "distance", "Average distance from stfwd path between reference configurations over trajectory vs "+xName +"\n"+specificationLine,
                    betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                    trajsExtremesInitID[filt], annealingShortDescription_Dic, edgeColorPerInitType_Dic, markerShapeVariable[filt], markerShapeVariableName,
                    yerr=qDistStdErr[filt], nGraphs=len(np.unique(graphID[filt])))

            mainPlot = plotWithDifferentColorbars(f"tau", x[filt], xName, chi_tau[filt], r"$\tau_{trans}$", r"transient time $\tau_{trans}$ vs "+ xName+"\n"+specificationLine,
                        betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                        trajsExtremesInitID[filt], annealingShortDescription_Dic, edgeColorPerInitType_Dic, markerShapeVariable[filt], markerShapeVariableName,
                        yerr=stdDevBarrier[filt], nGraphs=len(np.unique(graphID[filt])))
                
            mainPlot = plotWithDifferentColorbars(f"realTime", x[filt], xName, realTime[filt], "computer time (seconds)", "Seconds required to perform 10^5 mc sweeps vs "+ xName+"\n"+specificationLine,
                        betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                        trajsExtremesInitID[filt], annealingShortDescription_Dic, edgeColorPerInitType_Dic, markerShapeVariable[filt], markerShapeVariableName,
                        yerr=realTimeErr[filt], nGraphs=len(np.unique(graphID[filt])))

            mainPlot = plotWithDifferentColorbars(f"TIhout", x[filt], xName, TIhout[filt], "L", "Quantity for thermodynamic integration vs "+ xName +"\n"+specificationLine,
                    betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                    trajsExtremesInitID[filt], annealingShortDescription_Dic, edgeColorPerInitType_Dic,
                    markerShapeVariable[filt], markerShapeVariableName,
                    nGraphs=len(np.unique(graphID[filt])))
                
            mainPlot = plotWithDifferentColorbars(f"TIQstar", x[filt], xName, TIQstar[filt], "L", "Quantity for thermodynamic integration vs "+ xName +"\n"+specificationLine,
                    betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                    trajsExtremesInitID[filt], annealingShortDescription_Dic, edgeColorPerInitType_Dic,
                    markerShapeVariable[filt], markerShapeVariableName,
                    nGraphs=len(np.unique(graphID[filt])))
                
            mainPlot = plotWithDifferentColorbars(f"Z", x[filt], xName, ZFromTIBeta[filt], "L", "Quantity for thermodynamic integration vs "+ xName +"\n"+specificationLine,
                    betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                    trajsExtremesInitID[filt], annealingShortDescription_Dic, edgeColorPerInitType_Dic,
                    markerShapeVariable[filt], markerShapeVariableName,
                    nGraphs=len(np.unique(graphID[filt])))

            mainPlot = plotWithDifferentColorbars(f"Zlog", x[filt], xName, ZFromTIBeta[filt], "L", "Quantity for thermodynamic integration vs "+ xName +"\n"+specificationLine,
                    betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                    trajsExtremesInitID[filt], annealingShortDescription_Dic, edgeColorPerInitType_Dic,
                    markerShapeVariable[filt], markerShapeVariableName,
                    nGraphs=len(np.unique(graphID[filt])), yscale='log')
            
            mainPlot = plotWithDifferentColorbars(f"k", x[filt], xName, kFromChi[filt], "L", "Quantity for thermodynamic integration vs "+ xName +"\n"+specificationLine,
                    betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                    trajsExtremesInitID[filt], annealingShortDescription_Dic, edgeColorPerInitType_Dic,
                    markerShapeVariable[filt], markerShapeVariableName,
                    nGraphs=len(np.unique(graphID[filt])), yscale='log')

            figs = plt.get_figlabels()  # Ottieni i nomi di tutte le figure create
            for fig_name in figs:
                fig = plt.figure(fig_name)
                filename = os.path.join(theseFiguresFolder, f'{fig_name}.png')
                #print(filename)
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

        analysis_path = parentAnalysis_path+"/"+runGroupName
        plotsFolder = os.path.join(analysis_path, "Plots")
        os.makedirs(plotsFolder, exist_ok=True)

        if len(np.unique(beta[runGroupFilter]))>2:
            folderingArray= [ [t,g] for t, g in list(zip(T, graphID))]
            folderingArray = np.asarray(folderingArray)
            myMultiRunStudy(runGroupFilter,"StudyInBeta", beta, r"$\beta$", np.array(list(zip(N, Qstar, fieldType, fieldSigma))), "N, Qstar, fieldType"+r"$\sigma$", folderingArray, "T, graphID")
        
        if len(np.unique(T[runGroupFilter]))>2:
            myMultiRunStudy(filter=runGroupFilter,studyName="StudyInT", x=T,  xName="T", subfolderingVariable=N, subfolderingVariableName="N", markerShapeVariable=np.array(list(zip(beta, graphID))), markerShapeVariableName="beta, graphID")
            
        if len(np.unique(N[runGroupFilter]))>2: 
            print(np.unique(N[runGroupFilter]))
            tempQstar= Qstar #così, slice-ando rispetto a N unisco i casi con q_star uguale
            Qstar*=0
            b=np.array(list(zip(h_in, h_out, Qstar)))
            myMultiRunStudy(runGroupFilter, "StudyInN", N, "N", np.asarray(list(zip(Qstar, beta))), "h_in, h_out, Qstar, beta" , np.array(list(zip(T, graphID))), "T, graphID")
            Qstar=tempQstar
        return

        if len(np.unique(fieldSigma[runGroupFilter]))>1:
                myMultiRunStudy(runGroupFilter,"StudyInFieldSigma", fieldSigma, r"fieldSigma", np.array(list(zip(N, T))), "N, T", np.array(list(zip(beta, graphID, fieldRealization))), "field")

        if len(np.unique((T*N)[runGroupFilter]))>2:
            myMultiRunStudy(runGroupFilter,"StudyInNT", N*T, "NT",N, "N", beta, r"beta")

        """
        # Define the path of destination for the plots
        Results_path = os.path.join(analysis_path,"../..","OverallResults")
        if not os.path.exists(Results_path):
            os.makedirs(Results_path)

        TIbetaFolder= os.path.join(Results_path, "TIbeta")
        if not os.path.exists(TIbetaFolder):
            os.makedirs(TIbetaFolder)

        ChiFitFolder = os.path.join(Results_path, "ChiData")

        for simHin, simHout, simQstar, simGraphID, simT, simBetaOfExtraction, simFirstConfigurationIndex, simSecondConfigurationIndex, simRefConfMutualQ, simFieldType, simFieldMean, simFieldSigma, simFieldRealization in set(zip(h_in, h_out, Qstar, graphID, T, betaOfExtraction, firstConfigurationIndex, secondConfigurationIndex, refConfMutualQ,fieldType, fieldMean, fieldSigma, fieldRealization)):
            #dovrei anche controllare che le configurazioni di riferimento sono le stesse. Per ora non è un problema
            filt = np.logical_and.reduce([runGroupFilter,
                                          h_in == simHin, h_out == simHout, Qstar == simQstar, graphID == simGraphID, T == simT, betaOfExtraction==simBetaOfExtraction,
                                        firstConfigurationIndex==simFirstConfigurationIndex, secondConfigurationIndex== simSecondConfigurationIndex,
                                        fieldType==simFieldType, fieldMean==simFieldMean, fieldSigma==simFieldSigma, fieldRealization==simFieldRealization])
            if len(np.unique(beta[filt]))<=4:
                continue
            
            n=np.unique(N[filt])
            if len(n)>1:
                print("There seems to be more N values for the specified set of parameters.")
                continue

            #file_path = f"TIbeta_N{n}C3T{simT}f1.00g{simGraphID}Hin{simHin}Hout{simHout}Qstar{simQstar}.txt"

            if simT == np.floor(simT):
                fileName = f"{symType}_{simGraphID}_{simRefConfMutualQ}_{simHin}_{simHout}_{simQstar}_{int(simT)}_{simFieldType}_{simFieldMean}_{simFieldSigma}_{simFieldRealization}.txt"
            else:
                fileName = f"{symType}_{simGraphID}_{simRefConfMutualQ}_{simHin}_{simHout}_{simQstar}_{simT:.1f}_{simFieldType}_{simFieldMean}_{simFieldSigma}_{simFieldRealization}.txt"

            file_path = os.path.join(TIbetaFolder, fileName)
            with open(file_path, 'w') as f:
                f.write("#graphType graphID [refConfsInfo] Hin Hout Qstar T fieldType fieldMean fieldSigma fieldRealization\n")
                f.write(f"#{symType} {simGraphID} {simRefConfMutualQ} {simHin} {simHout} {simQstar} {simT} {simFieldType} {simFieldMean} {simFieldSigma} {simFieldRealization}")
                for valore1, valore2 in zip(beta[filt], TIbeta[filt]):
                    f.write('\n{} {}'.format(valore1, valore2))

            filt = np.logical_and(filt, fPosJ==1.0)
            file_path = os.path.join(ChiFitFolder, fileName)
            if len(beta[filt])>0:
                with open(file_path, 'w') as f:
                    f.write("#graphType graphID [refConfsInfo] Hin Hout Qstar T fieldType fieldMean fieldSigma fieldRealization\n")
                    f.write(f"#{symType} {simGraphID} {simRefConfMutualQ} {simHin} {simHout} {simQstar} {simT} {simFieldType} {simFieldMean} {simFieldSigma} {simFieldRealization}")
                    for bet, m, tau in zip(beta[filt], chi_m[filt], chi_tau[filt]):
                        f.write('\n{} {} {}'.format(bet, m, tau))
        
        stMC_betaOfExtraction = stMC_betaOfExtraction.astype(str)
        stMC_configurationIndex = stMC_configurationIndex.astype(str)

        for simHout, simQstar, simGraphID, simBetaOfExtraction, simConfigurationIndex, simFieldType, simFieldMean, simFieldSigma, simFieldRealization in set(zip(stMC_Hout, stMC_Qstar, stMC_graphID, stMC_betaOfExtraction, stMC_configurationIndex, stMC_fieldType, stMC_fieldMean, stMC_fieldSigma, stMC_fieldRealization)):
            #dovrei anche controllare che le configurazioni di riferimento sono le stesse. Per ora non è un problema
            thisQif = np.unique(refConfMutualQ[np.logical_and.reduce([h_out==simHout, Qstar==simQstar, graphID==simGraphID, betaOfExtraction==simBetaOfExtraction, secondConfigurationIndex==simConfigurationIndex,
                                                                    fieldType==simFieldType, fieldMean==simFieldMean, fieldSigma==simFieldSigma, fieldRealization==simFieldRealization ])])
            
            if len(thisQif)==0:
                continue
            thisQif = thisQif[0]
            filt = np.logical_and.reduce([stMC_Hout == simHout, stMC_Qstar == simQstar, stMC_graphID == simGraphID,  stMC_betaOfExtraction==simBetaOfExtraction, stMC_configurationIndex==simConfigurationIndex,
                                        stMC_fieldType==simFieldType, stMC_fieldMean==simFieldMean, stMC_fieldSigma==simFieldSigma, stMC_fieldRealization==simFieldRealization])
            if len(np.unique(stMC_beta[filt]))<=4:
                continue

            if len(np.unique(stMC_N[filt]))>1:
                print("There seems to be more N values for the specified set of parameters.")
                continue

            #file_path = f"TIbeta_N{n}C3T{simT}f1.00g{simGraphID}Hin{simHin}Hout{simHout}Qstar{simQstar}.txt"
            file_path = os.path.join(TIbetaFolder,f"{symType}_{simGraphID}_{simHout}_{simQstar}_{thisQif}_inf_{simFieldType}_{simFieldMean}_{simFieldSigma}_{simFieldRealization}.txt")
            with open(file_path, 'w') as f:
                f.write("#graphType graphID [refConfsInfo] Hin Hout Qstar T\n")
                f.write(f"#{symType} {simGraphID} {simBetaOfExtraction} {simHout} {simQstar}")
                for valore1, valore2 in zip(stMC_beta[filt], stMC_TIbeta[filt]):
                    f.write('\n{} {}'.format(valore1, valore2))
        """