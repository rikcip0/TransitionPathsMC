import os
import sys
sys.path.append('../')
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from itertools import cycle
import json
from matplotlib.colors import LinearSegmentedColormap, to_rgba

from scipy.optimize import curve_fit
from uncertainties import ufloat

from MyBasePlots.plotWithDifferentColorbars import plotWithDifferentColorbars


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

def singleMultiRunAnalysis(runsData, analysis_path, symType):

    plt.rcParams["axes.grid"]= True
    plt.rcParams['lines.marker'] = 'o'
    plt.rcParams['lines.linestyle'] = '-'
    plt.rcParams['lines.markersize'] = 10
    markers = ['.', '^', 'o', 'v','p', 'h']



    plotsFolder = os.path.join(analysis_path, "Plots")
    os.makedirs(plotsFolder, exist_ok=True)
    # Define the path of destination for the plots

    ID = []
    simulationType =[]
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

    stMC_N = []
    stMC_beta = []
    stMC_Hext = []
    stMC_Hout = []
    stMC_Qstar = []
    stMC_TIbeta= []
    stMC_graphID = []
    stMC_betaOfExtraction = []
    stMC_configurationIndex = []

    for item in runsData:
        if "results" not in item.keys():
            print("non c è")
            continue

        if 'configuration' not in item.keys() or 'parameters' not in item['configuration'] or 'h_in' not in item['configuration']['parameters']:
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

                stMC_TIbeta.append(item["results"]['TI']['beta'])
            continue

        #print("analyzing", item['configuration']['ID'])  #decommentare per controllare quando c'è un intoppo
        ID.append(item['configuration']['ID']) #ID only contains new IDs. will be used to check what analysis to repeat
        refConfInitID.append(item['configuration']['referenceConfigurationsInfo']['ID'])
        graphID.append(item['configuration']['parameters']['graphID'])


        #trajsJumpsInitID.append(item['configuration']['trajs_jumpsInitialization']['ID'])

        n=(int)(item['configuration']['parameters']['N'])
        N.append(n)
        beta.append(item['configuration']['parameters']['beta'])
        T.append(item['configuration']['parameters']['T'])
        h_ext.append(item['configuration']['parameters']['hext'])
        h_in.append(item['configuration']['parameters']['h_in'])
        h_out.append(item['configuration']['parameters']['h_out'])
        Qstar.append(item['configuration']['parameters']['Qstar'])
            
        if item['configuration']['referenceConfigurationsInfo']['ID']==50:
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
    Qstar= np.array(Qstar, dtype=np.int16)
    h_ext = np.array(h_ext, dtype=np.float64)

    lastMeasureMC = np.array(lastMeasureMC, dtype=np.int16)
    MCprint = np.array(MCprint, dtype=np.int16)

    nMeasures = lastMeasureMC//MCprint #not completely correcty

    refConfInitID = np.array(refConfInitID, dtype=np.int16)
    refConfMutualQ = np.array(refConfMutualQ, dtype=np.int16)
    betaOfExtraction = [float(value) if (value != "infty" and value!="nan") else ( np.inf if value!="nan" else np.nan)for value in betaOfExtraction]
    betaOfExtraction = np.array(betaOfExtraction, dtype=np.float64)
    firstConfigurationIndex = [int(value) if value != "nan" else np.nan for value in firstConfigurationIndex]
    firstConfigurationIndex = np.array(firstConfigurationIndex)
    secondConfigurationIndex = [int(value) if value != "nan" else np.nan for value in secondConfigurationIndex]
    secondConfigurationIndex = np.array(secondConfigurationIndex)


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
    
    meanBarrier = np.array(meanBarrier, dtype=np.float64)
    stdDevBarrier = np.array(stdDevBarrier, dtype=np.float64)
    
    avEnergy = np.array(avEnergy, dtype=np.float64)
    avEnergyStdErr = np.array(avEnergyStdErr, dtype=np.float64)
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
    stMC_Hext  = np.array(stMC_Hext, dtype=np.float64)
    stMC_Hout = np.array(stMC_Hout, dtype=np.float64)
    stMC_Qstar = np.array(stMC_Qstar, dtype=np.int16)
    stMC_graphID  = np.array(stMC_graphID)
    stMC_betaOfExtraction = [float(value) if value != "infty" else np.inf for value in stMC_betaOfExtraction]
    stMC_betaOfExtraction = np.array(stMC_betaOfExtraction)
    stMC_configurationIndex  = np.array(stMC_configurationIndex)


    
    normalizedRefConfMutualQ = refConfMutualQ/N

    shortDescription= {70: "Random", 71: "Ref 12", 72: "Ref 21", 73: "Annealing", 74: "Annealing"}
    edgeColorPerInitType={ 70: "lightGreen", 71: "black", 72: "purple", 73: "red", 74: "red"}

    def myMultiRunStudy(studyName, x, xName, subfolderingVariable, subfolderingVariableName, markerShapeVariable, markerShapeVariableName):
        thisStudyFolder= os.path.join(plotsFolder, studyName)
        if not os.path.exists(thisStudyFolder):
            os.makedirs(thisStudyFolder)
        else:
            delete_files_in_folder(thisStudyFolder)

        for v, sim_Hin, sim_Hout, sim_Qstar in set(zip(subfolderingVariable, h_in, h_out, Qstar)):
            filt = np.logical_and.reduce([subfolderingVariable==v, h_in==sim_Hin, h_out==sim_Hout, Qstar==sim_Qstar])

            if len(np.unique(x[filt]))<3:
                continue

            theseFiguresFolder= os.path.join(thisStudyFolder, f"{subfolderingVariableName}{v}", f"{sim_Hin}_{sim_Hout}_{sim_Qstar}")
            if not os.path.exists(theseFiguresFolder):
                os.makedirs(theseFiguresFolder)
            else:
                delete_files_in_folder(theseFiguresFolder)

            specificationLine =f"at {subfolderingVariableName} = {v} h_in = {sim_Hin} h_out = {sim_Hout} Qstar = {sim_Qstar}"

            additional=None
            if xName=="beta":
                stMC_corrBetaAndQif = np.empty((len(stMC_beta), 2), dtype=object)
                for sim_Bet, sim_Qif in set(zip(betaOfExtraction[filt], refConfMutualQ[filt])):
                    secondConfIndeces = np.unique(secondConfigurationIndex[np.logical_and(filt, refConfMutualQ==sim_Qif)])
                    for secondConfIndex in secondConfIndeces:
                        tempFilt=filt
                        filt= np.logical_and.reduce([stMC_N == v, stMC_Hout == sim_Hout, stMC_Qstar == sim_Qstar, stMC_configurationIndex==secondConfIndex])
                        stMC_corrBetaAndQif[filt]=[sim_Bet,sim_Qif]
                        filt=tempFilt
                additional = [stMC_beta, stMC_TIbeta, stMC_corrBetaAndQif, "inf"]

            TIbetaMainPlot = plotWithDifferentColorbars(f"TIbeta", x[filt], xName, TIbeta[filt], "L", "Quantity for thermodynamic integration vs "+ xName +"\n"+specificationLine,
            betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
            trajsExtremesInitID[filt], shortDescription, edgeColorPerInitType,
            markerShapeVariable[filt], markerShapeVariableName,
            nGraphs=len(np.unique(graphID[filt])), additionalMarkerTypes=additional)
            



            mainPlot = plotWithDifferentColorbars(f"TIhout", x[filt], xName, TIhout[filt], "L", "Quantity for thermodynamic integration vs "+ xName +"\n"+specificationLine,
                    betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                    trajsExtremesInitID[filt], shortDescription, edgeColorPerInitType,
                    markerShapeVariable[filt], markerShapeVariableName,
                    nGraphs=len(np.unique(graphID[filt])))
                
            mainPlot = plotWithDifferentColorbars(f"TIQstar", x[filt], xName, TIQstar[filt], "L", "Quantity for thermodynamic integration vs "+ xName +"\n"+specificationLine,
                    betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                    trajsExtremesInitID[filt], shortDescription, edgeColorPerInitType,
                    markerShapeVariable[filt], markerShapeVariableName,
                    nGraphs=len(np.unique(graphID[filt])))
                
            mainPlot = plotWithDifferentColorbars(f"avEnergy", x[filt], xName, avEnergy[filt], "energy", "Mean average energy over trajectory vs "+ xName +"\n"+specificationLine,
                    betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                    trajsExtremesInitID[filt], shortDescription, edgeColorPerInitType, markerShapeVariable[filt], markerShapeVariableName,
                    yerr=avEnergyStdErr[filt], nGraphs=len(np.unique(graphID[filt])))

            mainPlot = plotWithDifferentColorbars(f"nJumps", x[filt], xName, nJumps[filt], "# jumps", "Mean average energy over trajectory vs "+ xName +"\n"+specificationLine,
                    betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                    trajsExtremesInitID[filt], shortDescription, edgeColorPerInitType, markerShapeVariable[filt], markerShapeVariableName,
                    yerr=nJumpsStdErr[filt], nGraphs=len(np.unique(graphID[filt])))

            mainPlot = plotWithDifferentColorbars(f"deltaNJumps", x[filt], xName, deltaNJumps[filt], "# jumps", "Mean average energy over trajectory vs "+ xName +"\n"+specificationLine,
                    betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                    trajsExtremesInitID[filt], shortDescription, edgeColorPerInitType, markerShapeVariable[filt], markerShapeVariableName,
                    yerr=deltaNJumpsStdErr[filt], nGraphs=len(np.unique(graphID[filt])))
                
            mainPlot = plotWithDifferentColorbars(f"deltaNOverAvJumps", x[filt], xName, deltaNJumps[filt]**2/nJumps[filt], "# jumps", "Mean average energy over trajectory vs "+ xName +"\n"+specificationLine,
                    betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                    trajsExtremesInitID[filt], shortDescription, edgeColorPerInitType, markerShapeVariable[filt], markerShapeVariableName,
                    yerr=deltaNJumpsStdErr[filt], nGraphs=len(np.unique(graphID[filt])))
                
            mainPlot = plotWithDifferentColorbars(f"qDist", x[filt], xName, qDist[filt], "distance", "Mean average energy over trajectory vs "+xName +"\n"+specificationLine,
                    betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                    trajsExtremesInitID[filt], shortDescription, edgeColorPerInitType, markerShapeVariable[filt], markerShapeVariableName,
                    yerr=qDistStdErr[filt], nGraphs=len(np.unique(graphID[filt])))

            mainPlot = plotWithDifferentColorbars(f"meanBarrier", x[filt], xName, meanBarrier[filt], "barrier", "mean barrier vs "+ xName+"\n"+specificationLine,
                        betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                        trajsExtremesInitID[filt], shortDescription, edgeColorPerInitType, markerShapeVariable[filt], markerShapeVariableName,
                        yerr=stdDevBarrier[filt], nGraphs=len(np.unique(graphID[filt])))

            mainPlot = plotWithDifferentColorbars(f"tau", x[filt], xName, chi_tau[filt], r"$\tau_{trans}$", r"transient time $\tau_{trans}$ vs "+ xName+"\n"+specificationLine,
                        betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                        trajsExtremesInitID[filt], shortDescription, edgeColorPerInitType, markerShapeVariable[filt], markerShapeVariableName,
                        yerr=stdDevBarrier[filt], nGraphs=len(np.unique(graphID[filt])))

                
            mainPlot = plotWithDifferentColorbars(f"realTime", x[filt], xName, realTime[filt], "computer time (seconds)", "Seconds required to perform 10^5 mc sweeps vs "+ xName+"\n"+specificationLine,
                        betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                        trajsExtremesInitID[filt], shortDescription, edgeColorPerInitType, markerShapeVariable[filt], markerShapeVariableName,
                        yerr=realTimeErr[filt], nGraphs=len(np.unique(graphID[filt])))

            mainPlot = plotWithDifferentColorbars(f"muAvEnergy", x[filt], xName, muAvEnergy[filt], r"$\mu$", r"$\mu$"+" vs "+ xName+"\n"+specificationLine,
                            betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                            trajsExtremesInitID[filt], shortDescription, edgeColorPerInitType, markerShapeVariable[filt], markerShapeVariableName,
                            yerr=stdDevBarrier[filt], fitType='', xscale='', yscale='log', nGraphs=len(np.unique(graphID[filt])))
            
            tempFilt=filt
            for t in np.unique(markerShapeVariable[filt]):
                filt = np.logical_and(markerShapeVariable==t, tempFilt)
                if len(ID[filt])>0:
                    mainPlot = plotWithDifferentColorbars(f"muAvEnergy_{markerShapeVariableName}{t}", x[filt], xName, muAvEnergy[filt], r"$\mu$", r"$\mu$"+" vs "+ xName+"\n"+specificationLine,
                            betaOfExtraction[filt], normalizedRefConfMutualQ[filt],
                            trajsExtremesInitID[filt], shortDescription, edgeColorPerInitType, markerShapeVariable[filt], markerShapeVariableName,
                            yerr=stdDevBarrier[filt], fitType='', xscale='', yscale='', nGraphs=len(np.unique(graphID[filt])))
            filt=tempFilt


            figs = plt.get_figlabels()  # Ottieni i nomi di tutte le figure create
            for fig_name in figs:
                fig = plt.figure(fig_name)
                filename = os.path.join(theseFiguresFolder, f'{fig_name}.png')
                fig.savefig(filename, bbox_inches='tight')
            plt.close('all')


    Results_path = os.path.join(analysis_path,"../..","OverallResults")
    if not os.path.exists(Results_path):
        os.makedirs(Results_path)

    TIbetaFolder= os.path.join(Results_path, "TIbeta")
    if not os.path.exists(TIbetaFolder):
        os.makedirs(TIbetaFolder)


    betaOfExtraction = betaOfExtraction.astype(str)
    firstConfigurationIndex = firstConfigurationIndex.astype(str)
    secondConfigurationIndex = secondConfigurationIndex.astype(str)

    for simHin, simHout, simQstar, simGraphID, simT, simBetaOfExtraction, simFirstConfigurationIndex, simSecondConfigurationIndex, simRefConfMutualQ in set(zip(h_in, h_out, Qstar, graphID, T, betaOfExtraction, firstConfigurationIndex, secondConfigurationIndex, refConfMutualQ)):
        #dovrei anche controllare che le configurazioni di riferimento sono le stesse. Per ora non è un problema
        filt = np.logical_and.reduce([h_in == simHin, h_out == simHout, Qstar == simQstar, graphID == simGraphID, T == simT, betaOfExtraction==simBetaOfExtraction,
                                      firstConfigurationIndex==simFirstConfigurationIndex, secondConfigurationIndex== simSecondConfigurationIndex])
        if len(np.unique(beta[filt]))<=4:
            continue
        
        n=np.unique(N[filt])
        if len(n)>1:
            print("There seems to be more N values for the specified set of parameters.")
            continue

        #file_path = f"TIbeta_N{n}C3T{simT}f1.00g{simGraphID}Hin{simHin}Hout{simHout}Qstar{simQstar}.txt"
        file_path = os.path.join(TIbetaFolder, f"{symType}_{simGraphID}_{simRefConfMutualQ}_{simHin}_{simHout}_{simQstar}_{(int (simT))}.txt")
        with open(file_path, 'w') as f:
            f.write("#graphType graphID [refConfsInfo] Hin Hout Qstar T\n")
            f.write(f"#{symType} {simGraphID} {simRefConfMutualQ} {simHin} {simHout} {simQstar} {simT}")
            for valore1, valore2 in zip(beta[filt], TIbeta[filt]):
                f.write('\n{} {}'.format(valore1, valore2))

    stMC_betaOfExtraction = stMC_betaOfExtraction.astype(str)
    stMC_configurationIndex = stMC_configurationIndex.astype(str)

    for simHout, simQstar, simGraphID, simBetaOfExtraction, simConfigurationIndex in set(zip(stMC_Hout, stMC_Qstar, stMC_graphID, stMC_betaOfExtraction, stMC_configurationIndex)):
        #dovrei anche controllare che le configurazioni di riferimento sono le stesse. Per ora non è un problema
        thisQif = np.unique(refConfMutualQ[np.logical_and.reduce([h_out==simHout, Qstar==simQstar, graphID==simGraphID, betaOfExtraction==simBetaOfExtraction, secondConfigurationIndex==simConfigurationIndex ])])
        
        if len(thisQif)==0:
            continue
        thisQif = thisQif[0]
        filt = np.logical_and.reduce([stMC_Hout == simHout, stMC_Qstar == simQstar, stMC_graphID == simGraphID,  stMC_betaOfExtraction==simBetaOfExtraction, stMC_configurationIndex==simConfigurationIndex])
        if len(np.unique(stMC_beta[filt]))<=4:
            continue

        if len(np.unique(stMC_N[filt]))>1:
            print("There seems to be more N values for the specified set of parameters.")
            continue

        #file_path = f"TIbeta_N{n}C3T{simT}f1.00g{simGraphID}Hin{simHin}Hout{simHout}Qstar{simQstar}.txt"
        file_path = os.path.join(TIbetaFolder,f"{symType}_{simGraphID}_{simHout}_{simQstar}_{thisQif}_inf.txt")
        with open(file_path, 'w') as f:
            f.write("#graphType graphID [refConfsInfo] Hin Hout Qstar T\n")
            f.write(f"#{symType} {simGraphID} {simBetaOfExtraction} {simHout} {simQstar}")
            for valore1, valore2 in zip(stMC_beta[filt], stMC_TIbeta[filt]):
                f.write('\n{} {}'.format(valore1, valore2))

    if len(np.unique(beta))>2:
        myMultiRunStudy("StudyInBeta", beta, r"beta", N, "N", T, "T")
    if len(np.unique(N))>2:
        myMultiRunStudy("StudyInN", N, "N", beta, "beta", T, "T")

    Qstar= Qstar/N #così, slice-ando rispetto a N unisco i casi con q_star uguale
    if len(np.unique(T))>2:
        myMultiRunStudy("StudyInT", T,  "T", beta, r"beta", N, "N")
    if len(np.unique(T*N))>2:
        myMultiRunStudy("StudyInNT", N*T, "NT", beta, r"beta", N, "N")