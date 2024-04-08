import random
import json
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import os
import sys
from scipy.optimize import curve_fit
import scipy.stats as stats
from scipy.stats import linregress


currentAnalysisVersion = "singleRunV01"
preset_Path="../../Data/Graphs/RRG/p2C3/N100/structure199200/fPosJ1.00/graph8960/1.2e+02_1_0_inf_100_inf_run1" #for a quick single run analysis

matplotlib.use('Agg') 
sys.path.append('../')
from MyBasePlots.hist import myHist
from MyBasePlots.autocorrelation import autocorrelationWithExpDecayAndMu

def writeJsonResult(data, filePath):
    json_data = json.dumps(data, indent=4)
    with open(filePath, 'w') as output_file:
        output_file.write(json_data)

def meanAndStdErrForParametricPlot(toBecomeX, toBecomeY):
    x_unique_values = np.unique(toBecomeX)
    y_mean_values = np.asarray([np.mean(toBecomeY[toBecomeX == x_value]) for x_value in x_unique_values])
    y_var_values = np.asarray([np.var(toBecomeY[toBecomeX == x_value])**0.5 for x_value in x_unique_values])
    return x_unique_values, y_mean_values, y_var_values

def progressiveLinearFit(x, y, type, threshold_chi_square=10.):

    par_values = []
    minimumShifting = np.maximum(len(x)//100, 5)

    def linear(t, a, b):
        return t*a+b
    
    iStartIndex=np.maximum(np.argmax(y>0.001),minimumShifting)

    if iStartIndex+minimumShifting>=len(x)-1:
        return None
    

    for i in range(iStartIndex, len(x)-2*minimumShifting, minimumShifting):
        for j in range(i+minimumShifting, len(x)-1, minimumShifting):
                    popt, pcov = curve_fit(linear, x[i:j], y[i:j], method='lm', p0=[1/x[-1],0.6])
                    slope = popt[0]
                    intercept = popt[1]
                    chi_value = np.nansum(((y[i:j]-(linear(x[i:j],*popt))))**2./y[i:j])/(j-i)
                    if chi_value < 3*0.001 and slope*x[-1]>0.1:
                        par_values.append((chi_value, i, j, slope, intercept))

    if len(par_values)==0:
        return None
    best_segment = min(par_values, key=lambda x: x[0]/(x[2]-x[1])**7.5)
    best_Chi = best_segment[0]

    if best_Chi<threshold_chi_square and best_segment[3]>0.00001: #and best_segment[3]*x[-1]+best_segment[4]>0.15
        return [best_segment[3], best_segment[4]], [best_segment[1], best_segment[2]], best_Chi, linear
    else:
        return None
    
def get_file_with_prefix(parent_dir, prefix):
    # Get a list of all files in the parent directory
    all_files = [f for f in os.listdir(parent_dir) if os.path.isfile(os.path.join(parent_dir, f))]
    
    # Iterate through each file
    for file in all_files:
        # Check if the file starts with the specified prefix
        if file.startswith(prefix):
            # If it does, return the path to this file
            return os.path.join(parent_dir, file)
    
    # If no file with the specified prefix is found, return None
    return None

def arraysFromBlockFile(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Inizializza le strutture dati
    blocks = []
    current_block = []

    # Processa le linee del file
    for line in lines:
        line = line.strip()

        if not line:  # Rilevato un block vuoto, salva il block corrente
            if current_block:
                blocks.append(current_block)
                current_block = []
        else:
            # Divide la riga in columns e converte in float
            columns = [float(val) for val in line.split()]
            current_block.append(columns)

    # Aggiungi l'ultimo block se non è vuoto
    if current_block:
        blocks.append(current_block)

    # Trasponi i blocks in array NumPy
    array_per_column = [np.array(block) for block in blocks]

    return np.transpose(np.asarray(array_per_column), (2,0,1))

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

def txtToInfo(file_path, mappa):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    plt.rcParams['axes.grid'] = True
    # Leggi la prima riga e ottieni simulationType e versione
    firstLine_tokens = lines[0].strip().split()
    if len(firstLine_tokens) < 2:
        print("Error in " +file_path+". First line should include at least simulationType and version.")
        return None

    simulation_type = firstLine_tokens[0]
    version = firstLine_tokens[1]
    # Trova le informazioni sulla simulazione nella mappa
    simulationType_info = mappa["simulationTypes"].get(simulation_type, {})
    simulationTypeVersion_info = mappa["simulationTypes"].get(simulation_type, {}).get("versions", {}).get(version, None)
    if (simulationType_info is None) or simulationTypeVersion_info is None:
        print(f"Tipo di simulazione o versione non trovato nella mappa per '{simulation_type} - Versione {version}'.")
        return None

    data = {
        "name": simulationType_info["name"],
        "simulationTypeId": simulationType_info["ID"],
        "shortName": simulationType_info["shortName"],
        "versionId": simulationTypeVersion_info["ID"],
    }

    if len(firstLine_tokens)>2:
        # Assegna i valori dei parametri aggiuntivi dalla prima riga
        for nParameter, paramName in enumerate(simulationTypeVersion_info["additionalParameters"]):
            if nParameter + 2 < len(firstLine_tokens):  # Considera solo se ci sono abbastanza token nella riga
                data[paramName] = firstLine_tokens[nParameter + 2]
    
    if ("machine" in data) and ("seed" in data):
        data["ID"]= data["machine"]+data["seed"]
    else:
        data["ID"] = (str)(random.randint(0,1000000))

    if(len((simulationTypeVersion_info["linesMap"])) != simulationTypeVersion_info["nAdditionalLines"]):
        print("Error in the map construction")
        return None

    for nLine, lineType in enumerate(simulationTypeVersion_info["linesMap"]):
        #print(nLine, lineType) #useful for debug
        data[lineType]={}
        line_info = mappa["lines"].get(lineType, None)
        if line_info is not None:
            parameters = lines[nLine+1].strip().split()
            line_structure = line_info[parameters[0]] 
            data[lineType]=line_structure
            if "nAdditionalParameters" in line_structure.keys():
                if line_structure["nAdditionalParameters"] !=0:
                    for n, parameter in enumerate(line_structure["additionalParameters"]):
                        data[lineType][parameter] = parameters[n+1]
            data[lineType].pop("nAdditionalParameters", None)
            data[lineType].pop("additionalParameters", None)
    #print(data)
    return data

def singlePathMCAnalysis(run_Path, configurationsInfo, goFast=False):

    simData = {}
    simData["configuration"]= configurationsInfo
    #getting run infos: END

    #setting results folders and plots default lines: START
    resultsFolder= os.path.join(run_Path, "Results")

    jsonResultsPath = os.path.join(resultsFolder,"runData.json")
    """
    if os.path.exists(jsonResultsPath):
        return
        with open(jsonResultsPath, 'r') as map_file:
            oldResults = json.load(map_file)
        if "lastAnalysisLastMCprint" in oldResults.keys():
            if lastAnalysisLastMCprint==CurrentAnalysisLastMcPrint and lastAnalysisAnalysisVersio==currentAnalysisVersion:
                return
    """
    
    os.makedirs(resultsFolder, exist_ok=True)
    delete_files_in_folder(resultsFolder)
    plotsFolder= os.path.join(resultsFolder, "Plots")
    os.makedirs(plotsFolder, exist_ok=True)
    

    otherInfoKeys = ['ID', 'description', 'shortDescription']
    graphInfoKeys = [ 'graphID', 'fPosJ', 'p', 'C', 'd']
    parametersInfo_Line = " ".join([str(parameter) + "=" + str(int(value) if value.isdigit() else float(value) if value.replace('-','',1).replace('.', '', 1).isdigit() else value) for parameter, value in simData['configuration']['parameters'].items() if parameter not in graphInfoKeys+otherInfoKeys])
    graphInfo_Line = " ".join([str(parameter) + "=" + str(int(value) if value.isdigit() else float(value) if value.replace('.', '', 1).isdigit() else value) for parameter, value in simData['configuration']['parameters'].items() if parameter in graphInfoKeys])
    
    
    if simData['configuration']['simulationTypeId']== 10:
        graphInfo_Line = "FM "+graphInfo_Line

    fracPosJ =simData['configuration']['parameters']['fPosJ']
    mcEq = simData['configuration']['mcParameters']['MCeq']
    mcPrint = simData['configuration']['mcParameters']['MCprint']
    mcMeas = simData['configuration']['mcParameters']['MCmeas']
    measuresInfo_Line = r"MC$_{eq}$="+f"{mcEq}"+" "+ r"MC$_{pr}$="+f"{mcPrint}"+" "+r"MC$_{meas}$="+f"{mcMeas}"

    settingInfo_Line = parametersInfo_Line+"\n"+ graphInfo_Line+"\n"+measuresInfo_Line

    if simData['configuration']['referenceConfigurationsInfo']['ID']== 54:
        refConInfo_Line = "refConf:"+(simData['configuration']['referenceConfigurationsInfo']['shortDescription'])+"Q"+simData['configuration']['referenceConfigurationsInfo']['mutualOverlap']
    elif simData['configuration']['referenceConfigurationsInfo']['ID']== 53:
        refConInfo_Line = "refConf:"+(simData['configuration']['referenceConfigurationsInfo']['shortDescription'])+r" $\beta$"+f"{(float)(simData['configuration']['referenceConfigurationsInfo']['betaOfExtraction']):.2g}"+" Q"+simData['configuration']['referenceConfigurationsInfo']['mutualOverlapBeforeQuenching']+"->"+simData['configuration']['referenceConfigurationsInfo']['mutualOverlap']
    elif simData['configuration']['referenceConfigurationsInfo']['ID']== 51:
        refConInfo_Line = "refConf:"+(simData['configuration']['referenceConfigurationsInfo']['shortDescription'])+" Q"+simData['configuration']['referenceConfigurationsInfo']['mutualOverlap']
    else:
        refConInfo_Line = "refConf:"+(simData['configuration']['referenceConfigurationsInfo']['shortDescription'])
    
    if simData['configuration']['simulationTypeId']<100:   #so, no annealing
        trajsInitInfo_Line = "ext_init:"+(simData['configuration']['trajs_Initialization']['shortDescription'])+"\n"
        trajsInitInfo_Line += "j_init:"+ simData['configuration']['trajs_jumpsInitialization']['shortDescription']
    else:
        trajsInitInfo_Line = "traj_init: Annealing" 
        trajsInitInfo_Line += "("+simData['configuration']['trajs_Initialization']['shortDescription']+")\n"
        if simData['configuration']['trajs_Initialization']['ID']==73:
            trajsInitInfo_Line += fr"$\beta_{{start}}$"+simData['configuration']['trajs_Initialization']['startingBeta']
            trajsInitInfo_Line += fr"$\Delta\beta$"+simData['configuration']['trajs_Initialization']['deltaBeta']
            trajsInitInfo_Line += fr"mc/$\beta$"+simData['configuration']['trajs_Initialization']['sweepsPerBeta']
        elif simData['configuration']['trajs_Initialization']['ID']==74:
            trajsInitInfo_Line += fr"$\beta_{{start}}$"+f"{float(simData['configuration']['trajs_Initialization']['startingBeta']):.2f} "
            trajsInitInfo_Line += fr"$\Delta\beta$"+f"{float(simData['configuration']['trajs_Initialization']['deltaBeta']):.2f} "
            trajsInitInfo_Line += fr"MC$_{{start}}$"+simData['configuration']['trajs_Initialization']['startingMC']+" "
            trajsInitInfo_Line += fr"MC$_{{end}}$"+simData['configuration']['trajs_Initialization']['finalMC']

    initInfo_Line = refConInfo_Line + "\n" + trajsInitInfo_Line

    def addInfoLines():       #useful for following plots
        xlabel_position = plt.gca().xaxis.label.get_position()
        plt.text(0, xlabel_position[1] - 0.18, settingInfo_Line, fontsize=7, ha='left', va='center', transform=plt.gca().transAxes)
        plt.text(1, xlabel_position[1] - 0.18, initInfo_Line, fontsize=7, ha='right', va='center', transform=plt.gca().transAxes)
    #setting results folders and plots default lines: END

    #ANALYSIS OF TI FILES: START
    results = {}
    results["TI"] ={"beta":[], "hout":[], "Qstar":[]}

    TIFile = get_file_with_prefix(run_Path, "TI_beta")
    if TIFile is not None:
        with open(TIFile, 'r') as file:
            numbers = [float(num) for num in file.readline().replace(',', '.').split() if any(c.isdigit() or c == '.' for c in num)]
        results["TI"]["beta"] = numbers[-1]
    else:
        results["TI"]["beta"] = "nan"

    TIFile = get_file_with_prefix(run_Path, "TI_hout")
    if TIFile is not None:
        with open(TIFile, 'r') as file:
            numbers = [float(num) for num in file.readline().replace(',', '.').split() if any(c.isdigit() or c == '.' for c in num)]
        results["TI"]["hout"] = numbers[-1]
    else:
        results["TI"]["hout"] = "nan"

    TIFile = get_file_with_prefix(run_Path, "TI_Qstar")
    if TIFile is not None:
        with open(TIFile, 'r') as file:
            numbers = [float(num) for num in file.readline().replace(',', '.').split() if any(c.isdigit() or c == '.' for c in num)]
        results["TI"]["Qstar"] = numbers[-1 ]
    else:
        results["TI"]["Qstar"] = "nan"
    #ANALYSIS OF TI FILES: END


    #ANALYSIS OF THERMALIZATION DATA: START
    thermCheck_filePath = get_file_with_prefix(run_Path, "thermCheck")
    if thermCheck_filePath is None:
        print("No thermCheck file found")
        return None

    theseFiguresFolder= os.path.join(plotsFolder, "thermalization")
    if not os.path.exists(theseFiguresFolder):
        os.makedirs(theseFiguresFolder)
    else:
        delete_files_in_folder(theseFiguresFolder)

    titleSpecification = "as from thermalization data"


    therm_mcMeasures = []
    therm_avEnergies = []
    therm_barriers = []
    therm_distFromStrfwdPath = []
    therm_meanNJumps = []
    therm_deltaNJumps = []
    therm_minNJumps = []
    therm_maxNJumps = []

    mcForThermCheck= int(simData['configuration']["thermalizationCheckParameters"]["mcForThermCheck"])
    mcEq= int(simData['configuration']["mcParameters"]["MCeq"])
    firstIndexOfMeasuresAtEq = (int)(np.ceil(mcEq/mcForThermCheck))

    measuresToSkipInOOEPlot = 1      #The first measure may be very different even wrt non-eq ones, as it contains the initialization trajectory
    
    with open(thermCheck_filePath, 'r') as file:
        #print("analizzando ", nome_file)
        lines = file.readlines()
        dataLines = filter(lambda x: not x.startswith('#'), lines)
        data = np.genfromtxt(dataLines, delimiter=' ')
        therm_mcMeasures=data[:,0]
        therm_avEnergies=data[:,1]
        therm_barriers=data[:,2]
        therm_distFromStrfwdPath=data[:,3]
        therm_meanNJumps=data[:,4]
        therm_deltaNJumps =data[:,5]
        therm_minNJumps =data[:,6]
        therm_maxNJumps =data[:,7]

    therm_mcMeasures = np.asarray(therm_mcMeasures, dtype=np.float64)
    therm_avEnergies = np.asarray(therm_avEnergies, dtype=np.float64)
    therm_barriers = np.asarray(therm_barriers, dtype=np.float64)
    therm_distFromStrfwdPath = np.asarray(therm_distFromStrfwdPath, dtype=np.float64)
    therm_meanNJumps = np.asarray(therm_meanNJumps, dtype=np.float64)
    therm_deltaNJumps = np.asarray(therm_deltaNJumps, dtype=np.float64)
    therm_minNJumps = np.asarray(therm_minNJumps, dtype=np.float64)
    therm_maxNJumps = np.asarray(therm_maxNJumps, dtype=np.float64)

    if len(therm_mcMeasures)==1:
        print("Not enough measures")
        return None

    results["thermalization"] = {}
    nMusToConsider = 40
    titleForAutocorrelations = "autocorrelation over mcSweeps"
    results["thermalization"]["maxNJumps"] = {}

    #defining a function to plot quantities evolution over mc iterations and the respective autocorrelation
    def mcEvolutionAndAutocorrelation(mcSweeps, quantity, firstIndexForEquilibrium,
                                      quantityShortName, quantityFullName, quantityLabelName, nMus):
        
        results["thermalization"][quantityShortName] = {}

        plt.figure(quantityShortName)
        plt.title(quantityFullName+" vs MC\n"+titleSpecification)
        plt.plot(mcSweeps, quantity)
        plt.xlabel("MC sweep")
        plt.ylabel(quantityLabelName)

        #plt.axvline(x=mcEq, color='red', linestyle='--')
        #plt.text(mcEq, plt.ylim()[1], 'MCeq', color='red', verticalalignment='bottom', horizontalalignment='right', fontsize=7)

        addInfoLines()

        results["thermalization"][quantityShortName]["mean"] = np.mean(quantity[firstIndexForEquilibrium:])
        results["thermalization"][quantityShortName]["stdErr"] = stats.sem(quantity[firstIndexForEquilibrium:])

        if (len(np.unique(quantity[firstIndexForEquilibrium:])) > 1): #i.e., if it s not a constant
            mu, muErr, rChi2, dof  = autocorrelationWithExpDecayAndMu(quantityShortName+"Autocorrelation", quantityFullName+" "+titleForAutocorrelations,
                                        mcSweeps[firstIndexForEquilibrium:], "mc", quantity[firstIndexForEquilibrium:], quantityLabelName,
                                        nMus)
            addInfoLines()
            results["thermalization"][quantityShortName]["mu"] = mu
            results["thermalization"][quantityShortName]["muErr"] = muErr
            results["thermalization"][quantityShortName]["rChi2"] = rChi2
            results["thermalization"][quantityShortName]["dof"] = dof
        else:
            results["thermalization"][quantityShortName]["mu"] = "nan"
            results["thermalization"][quantityShortName]["muErr"] = "nan"
            results["thermalization"][quantityShortName]["rChi2"] = "nan"
            results["thermalization"][quantityShortName]["dof"] = "nan"

    mcEvolutionAndAutocorrelation(therm_mcMeasures[:len(therm_mcMeasures)], therm_avEnergies[:len(therm_mcMeasures)], firstIndexOfMeasuresAtEq,
                                      "avEnergy", "trajectory average energy", "energy", nMusToConsider)
    
    mcEvolutionAndAutocorrelation(therm_mcMeasures, therm_barriers, firstIndexOfMeasuresAtEq,
                                      "maxEnergy", "trajectory max energy", "energy", nMusToConsider)
    
    mcEvolutionAndAutocorrelation(therm_mcMeasures, therm_distFromStrfwdPath, firstIndexOfMeasuresAtEq,
                                      "qDist", "Average trajectory distance from the straightforward", "energy", nMusToConsider)
    
    mcEvolutionAndAutocorrelation(therm_mcMeasures, therm_meanNJumps, firstIndexOfMeasuresAtEq,
                                      "nJumps", "Mean number of jumps per spin", "energy", nMusToConsider)
    
    mcEvolutionAndAutocorrelation(therm_mcMeasures, therm_deltaNJumps, firstIndexOfMeasuresAtEq,
                                      "deltaNJumps", r"Trajectory $\Delta$(#jumps) per spin ", "energy", nMusToConsider)
    
    mcEvolutionAndAutocorrelation(therm_mcMeasures, therm_minNJumps, firstIndexOfMeasuresAtEq,
                                      "minNJumps", r"Trajectory min(#jumps) per spin ", "energy", nMusToConsider)
    
    mcEvolutionAndAutocorrelation(therm_mcMeasures, therm_maxNJumps, firstIndexOfMeasuresAtEq,
                                      "maxNJumps", r"Trajectory max(#jumps) per spin ", "energy", nMusToConsider)
    
    figs = plt.get_figlabels()  # Ottieni i nomi di tutte le figure create
    for fig_name in figs:
        fig = plt.figure(fig_name)
        filename = os.path.join(theseFiguresFolder, f'{fig_name}.png')
        fig.savefig(filename, bbox_inches='tight')
    plt.close('all')
    #ANALYSIS OF THERMALIZATION DATA: END


    #ANALYSIS OF LOG: START
    file_path = get_file_with_prefix(run_Path, "log")
    if file_path is None:
        print("log file not found")
        return None
    
    with open(file_path, 'r') as file:
        #print("analizzando ", nome_file)
        lines = file.readlines()
        if len(lines)<3:
            plt.close('all')
            writeJsonResult(simData, os.path.join(resultsFolder,"runData.json"))
            return None
        for i in range(len(lines)):
            lines[i]=lines[i].replace('\n', '')
            lines[i] = ' '.join(lines[i].split())
        dataLines = filter(lambda x: not x.startswith('#'), lines)
        data = np.genfromtxt(dataLines, delimiter=' ')
        realTime = data[:,0]
        mcSteps = data[:,1]

    lastMeasureMc = (int)(np.max(mcSteps))
    simData["lastMeasureMC"] = lastMeasureMc
    results["realTime"]={}
    theseFiguresFolder= os.path.join(plotsFolder, "runRealTime")
    if not os.path.exists(theseFiguresFolder):
        os.makedirs(theseFiguresFolder)
    else:
        delete_files_in_folder(theseFiguresFolder)

    measuresInfo_Line += r" MC$_{lastPrin}$="+f"{lastMeasureMc:.3g}"
    settingInfo_Line = parametersInfo_Line+"\n"+ graphInfo_Line+"\n"+measuresInfo_Line

    realTime=np.asarray(realTime)
    realTime=np.diff(realTime)
    mcSteps = np.asarray(mcSteps, dtype=np.int32)

    plt.figure('realTimeVsMCSweeps')
    plt.xlabel('mc')
    plt.ylabel('Computer time')
    plt.scatter(mcSteps[1:], realTime)   #This 1 is different from that of firstMeasureToConsider (eventually, if it s =1). It is so because first entry is 0 by definition
    addInfoLines()

    firstMeasureToConsider = 1
    if len(realTime[firstMeasureToConsider:])>0:
        title = f"Histogram of computer time needed to perform {mcSteps[1]} steps\n partitioning total MC time. First {firstMeasureToConsider} measure(s) skipped"
        mean, sigma =myHist("realTimeHist", title, realTime[firstMeasureToConsider:], 'computer time')
        addInfoLines()
        results["realTime"]["mean"]=mean
        results["realTime"]["sigma"]=sigma
    else:
        results["realTime"]["mean"]="nan"
        results["realTime"]["sigma"]="nan"

    figs = plt.get_figlabels()  # Ottieni i nomi di tutte le figure create
    for fig_name in figs:
        fig = plt.figure(fig_name)
        filename = os.path.join(theseFiguresFolder, f'{fig_name}.png')
        fig.savefig(filename, bbox_inches='tight')
    plt.close('all')
    #ANALYSIS OF LOG: END


    #START OF ANALYSIS OF SAMPLED TRAJS 
    file_path = get_file_with_prefix(run_Path, "story")
    if file_path is None:
        print("No story file found")
        return None
    
    arrays = arraysFromBlockFile(file_path)

    # Esempio di accesso agli array per la prima column del primo block
    times = arrays[0,:, :]
    qStart = arrays[1,:, :]
    qEnd = arrays[2,:, :]
    M = arrays[3,:, :]
    energy = arrays[4,:, :]

    nTrajs=times.shape[0]

    if nTrajs == 0:
        plt.close('all')
        simData["results"] = results
        writeJsonResult(simData, os.path.join(resultsFolder,"runData.json"))
        return None
    
    theseFiguresFolder= os.path.join(plotsFolder, "sampledTrajs")
    if not os.path.exists(theseFiguresFolder):
        os.makedirs(theseFiguresFolder)
    else:
        delete_files_in_folder(theseFiguresFolder)

    titleSpecification = 'for initializing trajectory'

    #Plots of initialization trajectory: START
    theseFiguresSubFolder= os.path.join(theseFiguresFolder, "initializingTraj")
    if not os.path.exists(theseFiguresSubFolder):
        os.makedirs(theseFiguresSubFolder)
    else:
        delete_files_in_folder(theseFiguresSubFolder)

    plt.figure("energy")
    plt.plot(times[0], energy[0])
    plt.title(f'Energy vs time\n'+ titleSpecification)
    plt.xlabel('time')
    plt.ylabel('energy')
    addInfoLines()

    plt.figure("Qin")
    plt.plot(times[0], qStart[0])
    plt.title(r'$Q_{in}$ vs time'+'\n'+ titleSpecification)
    plt.xlabel('time')
    plt.ylabel(r'$Q_{in}$')
    addInfoLines()

    plt.figure("Qout")
    plt.plot(times[0], qEnd[0])
    plt.title(r'$Q_{out}$ vs time'+'\n'+ titleSpecification)
    plt.xlabel('time')
    plt.ylabel(r'$Q_{out}$')
    addInfoLines()

    plt.figure("M")
    plt.plot(times[0], M[0])
    plt.title(f'M vs time'+'\n'+ titleSpecification)
    plt.xlabel('time')
    plt.ylabel("M")
    addInfoLines()

    plt.figure("QoutVsQin")
    plt.plot(qStart[0], qEnd[0])
    plt.title(r'$Q_{out}$ vs $Q_{in}$' +'\n'+ titleSpecification)
    plt.xlabel(r'$Q_{in}$')
    plt.ylabel(r'$Q_{out}$')
    addInfoLines()

    plt.figure("MVsQin")
    plt.plot(qStart[0], M[0])
    plt.title(r'M vs $Q_{in}$'+'\n'+ titleSpecification)
    plt.xlabel(r'$Q_{in}$')
    plt.ylabel(r'M')
    addInfoLines()

    plt.figure("MVsQout")
    plt.plot(qEnd[0], M[0])
    plt.title(f'M vs' +r"Q_{out}"+'\n'+ titleSpecification)
    plt.xlabel(r'$Q_{out}$')
    plt.ylabel(r'M')
    addInfoLines()

    plt.figure("EnergyVsQin")
    plt.plot(qStart[0], energy[0])
    plt.title(f'energy vs ' +r"$Q_{in}$"+'\n'+ titleSpecification)
    plt.xlabel(r'$Q_{in}$')
    plt.ylabel(r'M')
    addInfoLines()

    plt.figure("EnergyVsQout")
    plt.plot(qEnd[0], energy[0])
    plt.title(f'energy vs ' +r"$Q_{out}$"+'\n'+ titleSpecification)
    plt.xlabel(r'$Q_{out}$')
    plt.ylabel(r'energy')
    addInfoLines()

    plt.figure("EnergyVsM")
    plt.plot(M[0], energy[0])
    plt.title(f'energy vs M\n'+ titleSpecification)
    plt.xlabel('M')
    plt.ylabel('Energy')
    addInfoLines()

    figs = plt.get_figlabels()  # Ottieni i nomi di tutte le figure create
    for fig_name in figs:
        fig = plt.figure(fig_name)
        filename = os.path.join(theseFiguresSubFolder, f'{fig_name}.png')
        fig.savefig(filename, bbox_inches='tight')
    plt.close('all')
    #Plots of initialization trajectory: END

    #Plots considering over all trajs: START
    times = times[1:,:]
    qStart = qStart[1:,:]
    qEnd = qEnd[1:,:]
    M = M[1:,:]
    energy = energy[1:,:]
    theseFiguresSubFolder= os.path.join(theseFiguresFolder, "allTrajs")
    if not os.path.exists(theseFiguresSubFolder):
        os.makedirs(theseFiguresSubFolder)
    else:
        delete_files_in_folder(theseFiguresSubFolder)

    titleSpecification = 'over all sampled trajectories'

    plt.figure("energyMean")
    plt.title(f'Mean of energy vs time\n'+ titleSpecification)
    plt.xlabel(r'time')
    plt.ylabel(r'$mean energy$')
    plt.plot(times.mean(0), energy.mean(0))
    addInfoLines()

    plt.figure("energyStdDev")
    plt.title(r'$\Delta$'+f' energy vs time\n'+ titleSpecification)
    plt.xlabel(r'time')
    plt.ylabel(r'$\Delta$E')
    plt.plot(times.mean(0), np.sqrt(energy.var(0)))
    addInfoLines()

    valori_unici_M = np.unique(M)
    medie_di_y_corrispondenti = [np.mean(energy[M == valore_x]) for valore_x in valori_unici_M]

    plt.figure("energyVsM")
    plt.title(f'Mean of energy vs M\n'+ titleSpecification)
    plt.xlabel(r'M')
    plt.ylabel(r'$energy$')
    plt.errorbar(*meanAndStdErrForParametricPlot(M, energy))
    addInfoLines()

    valori_unici_M = np.unique(qStart)
    medie_di_y_corrispondenti = [np.mean(energy[qStart == valore_x]) for valore_x in valori_unici_M]
    plt.figure("energyVsQin")
    plt.title(f'Mean of energy vs '+r'$Q_{in}$'+'\n'+ titleSpecification)
    plt.xlabel(r'$Q_{in}$')
    plt.ylabel(r'$energy$')
    plt.errorbar(*meanAndStdErrForParametricPlot(qStart, energy))
    addInfoLines()

    plt.figure("energyVsQout")
    plt.title(f'Mean of energy vs '+r'$Q_{out}$'+'\n'+ titleSpecification)
    plt.xlabel(r'$Q_{out}$')
    plt.ylabel(r'$energy$')
    plt.errorbar(*meanAndStdErrForParametricPlot(qEnd, energy))
    addInfoLines()

    barrier = energy - np.mean(energy[:, [0, -1]], axis=1, keepdims=True)
    barrier = np.max(barrier,1)

    myHist("barriersHistogram", 'Histogram of energy barriers\n'+ titleSpecification, barrier, 'barrier')
    addInfoLines()

    plt.figure("barriersEvolution")
    plt.plot(barrier)
    plt.title(f'Energy barriers over sampled trajectories\n'+ titleSpecification)
    plt.xlabel('#(sampled trajectory)')
    plt.ylabel('barrier')
    addInfoLines()

    results["meanBarrier"]= barrier.mean()
    results["deltaBarrier"]= barrier.var()**0.5
    results["stdDevBarrier"]= (barrier.var()/barrier.size)**0.5

    figs = plt.get_figlabels()  # Ottieni i nomi di tutte le figure create
    for fig_name in figs:
        fig = plt.figure(fig_name)
        filename = os.path.join(theseFiguresSubFolder, f'{fig_name}.png')
        fig.savefig(filename, bbox_inches='tight')
    plt.close('all')
    #Plots considering over all trajs: END

    

    #Plots considering only some trajs: START
    theseFiguresSubFolder= os.path.join(theseFiguresFolder, "someTrajs")
    if not os.path.exists(theseFiguresSubFolder):
        os.makedirs(theseFiguresSubFolder)
    else:
        delete_files_in_folder(theseFiguresSubFolder)

    nRandomTrajs= np.min([5, nTrajs-1])-1
    someTrajs= np.array([0])
    if nRandomTrajs>0:
        someTrajs = np.append(someTrajs, np.asarray([nTrajs-2]))
        someTrajs = np.sort(np.append(someTrajs, np.random.choice(np.arange(1, nTrajs-2), nRandomTrajs-1, replace=False)))
    
    titleSpecification = 'considering some sampled trajectories'

    plt.figure("energy")
    [plt.plot(times[t], energy[t], label=f"traj {t}") for t in someTrajs ]
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.title(f'Energy vs time\n'+ titleSpecification)
    plt.xlabel('time')
    plt.ylabel('energy')
    addInfoLines()

    plt.figure("Qin")
    [plt.plot(times[t], qStart[t], label=f"traj {t}") for t in someTrajs ]
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.title(r'$Q_{in}$ vs time'+'\n'+ titleSpecification)
    plt.xlabel('time')
    plt.ylabel(r'$Q_{in}$')
    addInfoLines()

    plt.figure("Qout")
    [plt.plot(times[t], qEnd[t], label=f"traj {t}") for t in someTrajs ]
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.title(r'$Q_{out}$ vs time'+'\n'+ titleSpecification)
    plt.xlabel('time')
    plt.ylabel(r'$Q_{out}$')
    addInfoLines()

    plt.figure("M")
    [plt.plot(times[t], M[t], label=f"traj {t}") for t in someTrajs ]
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.title(f'M vs time'+'\n'+ titleSpecification)
    plt.xlabel('time')
    plt.ylabel("M")
    addInfoLines()

    plt.figure("QoutVsQin")
    [plt.plot(qStart[t], qEnd[t], label=f"traj {t}") for t in someTrajs ]
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.title(r'$Q_{out}$ vs $Q_{in}$' +'\n'+ titleSpecification)
    plt.xlabel(r'$Q_{in}$')
    plt.ylabel(r'$Q_{out}$')
    addInfoLines()

    plt.figure("MVsQin")
    [plt.plot(qStart[t], M[t], label=f"traj {t}") for t in someTrajs ]
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.title(r'M vs $Q_{in}$'+'\n'+ titleSpecification)
    plt.xlabel(r'$Q_{in}$')
    plt.ylabel(r'M')
    addInfoLines()

    plt.figure("MVsQout")
    [plt.plot(qEnd[t], M[t], label=f"traj {t}") for t in someTrajs ]
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.title(f'M vs ' +r"$Q_{out}$"+'\n'+ titleSpecification)
    plt.xlabel(r'$Q_{out}$')
    plt.ylabel(r'M')
    addInfoLines()

    plt.figure("EnergyVsQin")
    [plt.plot(qStart[t], energy[t], label=f"traj {t}") for t in someTrajs ]
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.title(f'energy vs ' +r"$Q_{in}$"+'\n'+ titleSpecification)
    plt.xlabel(r'$Q_{in}$')
    plt.ylabel(r'energy')
    addInfoLines()

    plt.figure("EnergyVsQout")
    [plt.plot(qEnd[t], energy[t], label=f"traj {t}") for t in someTrajs ]
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.title(f'energy vs ' +r"$Q_{out}$"+'\n'+ titleSpecification)
    plt.xlabel(r'$Q_{out}$')
    plt.ylabel(r'energy')
    addInfoLines()

    plt.figure("EnergyVsM")
    [plt.plot(M[t], energy[t], label=f"traj {t}") for t in someTrajs ]
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.title(f'energy vs M\n'+ titleSpecification)
    plt.xlabel('M')
    plt.ylabel('Energy')
    addInfoLines()

    figs = plt.get_figlabels()  # Ottieni i nomi di tutte le figure create
    for fig_name in figs:
        fig = plt.figure(fig_name)
        filename = os.path.join(theseFiguresSubFolder, f'{fig_name}.png')
        fig.savefig(filename, bbox_inches='tight')
    plt.close('all')
    #Plots considering only some trajs: END
    #ANALYSIS OF SAMPLED TRAJS: END 
    
    #ANALYSIS OF AV: START 
    theseFiguresFolder= os.path.join(plotsFolder, "averagedData")
    titleSpecification = "averaged over measured trajs"
    if not os.path.exists(theseFiguresFolder):
        os.makedirs(theseFiguresFolder)
    else:
        delete_files_in_folder(theseFiguresFolder)

    file_path = get_file_with_prefix(run_Path, "av")
    if file_path is None:
        plt.close('all')
        simData["results"]=results
        writeJsonResult(simData, os.path.join(resultsFolder,"runData.json"))
        return None
    
    time = []
    avQin = []
    avQout = []
    avM = []
    avChi = []

    with open(file_path, 'r') as file:
        #print("analizzando ", nome_file)
        lines = file.readlines()
        dataLines = filter(lambda x: not x.startswith('#'), lines)
        data = np.genfromtxt(dataLines, delimiter=' ')
        time=data[:,0]
        avQin=data[:,1]
        avQout=data[:,2]
        avM=data[:,3]
        avChi=data[:,4]

    plt.figure("Chi")

    plt.plot(time, avChi)

    if fracPosJ==1.:
        fitOutput = progressiveLinearFit(time, avChi, "linear")
    else:
        fitOutput = progressiveLinearFit(time, avChi, "boh2")

    linearFitResults = {}
    if fitOutput is not None:
        best_fit_params, [linearity_lowerIndex, linearity_upperIndex], chi, funcToPlot = fitOutput

        x_limits = plt.xlim()
        y_limits = plt.ylim()
        if linearity_lowerIndex is not None:
            plt.axvline(time[linearity_lowerIndex], linestyle='dashed', linewidth=1, color='red', label=r"$\tau_{trans}=$"+f"{time[linearity_lowerIndex]:.2f}")
        if linearity_upperIndex is not None:
            plt.axvline(time[linearity_upperIndex], linestyle='dashed', linewidth=1, color='green', label=r"$\tau_{lin. end}=$"+f"{time[linearity_upperIndex]:.2f}")
        plt.plot(time,funcToPlot(time, *best_fit_params), '--', label=r'k'+f"={best_fit_params[0]:.3g}\n"+r"c"+f"={best_fit_params[1]:.3g}\n"+r'$\chi^2_{r}$'+f'={chi:.3g}', linewidth= 0.8) 
        #plt.plot(time,funcToPlot(time, best_fit_params[0]), '--', label=r'k'+f"={best_fit_params[0]:.3g}\n"+r'$\chi^2_{r}$'+f'={chi:.3g}', linewidth= 0.8) 
        plt.xlim(x_limits)
        plt.ylim(y_limits)
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        addInfoLines()
        linearFitResults["tau"] = time[linearity_lowerIndex]
        linearFitResults["m"] = best_fit_params[0]
        linearFitResults["c"] = best_fit_params[1]
    else:
        linearFitResults["tau"] = "nan"
        linearFitResults["m"] = "nan"
        linearFitResults["c"] = "nan"

    results["chiLinearFit"] = linearFitResults

    plt.title(f'$\chi$ vs time\n'+titleSpecification)
    plt.xlabel('time')
    plt.ylabel(r'$\chi$')
    addInfoLines()

    plt.figure("ChiVsQout")
    plt.plot(avQout, avChi)
    plt.title(f'$\chi$ vs '+r"$Q_{out}$"+"\n"+titleSpecification)
    plt.xlabel(r'$Q_{out}$')
    plt.ylabel(r'$\chi$')
    addInfoLines()

    plt.figure("ChiVsQin")
    plt.plot(avQin, avChi)
    plt.title(f'$\chi$ vs '+r"$Q_{in}$"+"\n"+titleSpecification)
    plt.xlabel(r'$Q_{in}$')
    plt.ylabel(r'$\chi$')
    addInfoLines()

    plt.figure("ChiVsM")
    plt.plot(avM, avChi)
    plt.title(f'$\chi$ vs '+r"M"+"\n"+titleSpecification)
    plt.xlabel(r'$M$')
    plt.ylabel(r'$\chi$')
    addInfoLines()

    plt.figure("Qs")
    plt.plot(time, avQin, label=r"$Q_{in}$")
    plt.plot(time, avQout, label=r"$Q_{out}$")
    #plt.plot(time, np.tanh(2*(time-time[-1]/2.)/time[-1]), label=r"tanh()")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.title(f'Qs vs time\n'+titleSpecification)
    plt.xlabel('time')
    plt.ylabel(r'$Q$')
    addInfoLines()

    plt.figure("M")
    plt.plot(time, avM)
    plt.title(f'Magnetization conf. vs time\n'+titleSpecification)
    plt.xlabel('time')
    plt.ylabel('M')
    addInfoLines()

    plt.figure("QoutVsQin")
    plt.plot(avQin, avQout)
    plt.title(r"$Q_{out}$"+" vs "+ r"$Q_{out}$"+'\n'+titleSpecification)
    plt.xlabel(r'$Q_{in}$')
    plt.ylabel(r'$Q_{out}$')
    addInfoLines()

    plt.figure("QsVsM")
    plt.plot(avM, avQin, label=r"$Q_{in}$")
    plt.plot(avM, avQout, label=r"$Q_{out}$")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.title(f'Qs vs M.\n'+titleSpecification)
    plt.xlabel('M')
    plt.ylabel(r'$Q$')
    addInfoLines()


    figs = plt.get_figlabels()  # Ottieni i nomi di tutte le figure create
    for fig_name in figs:
        fig = plt.figure(fig_name)
        filename = os.path.join(theseFiguresFolder, f'{fig_name}.png')
        fig.savefig(filename, bbox_inches='tight')
    plt.close('all')

    simData["results"]=results
    writeJsonResult(simData, os.path.join(resultsFolder,"runData.json"))

def singleStandardMCAnalysis(run_Path, configurationInfo, goFast=False):
    print("è un normale MC")
    simData = {}
    simData["configuration"]= configurationInfo
    #getting run infos: END

    #setting results folders and plots default lines: START
    resultsFolder= os.path.join(run_Path, "Results")
    os.makedirs(resultsFolder, exist_ok=True)
    delete_files_in_folder(resultsFolder)
    plotsFolder= os.path.join(resultsFolder, "Plots")
    os.makedirs(plotsFolder, exist_ok=True)
    

    otherInfoKeys = ['ID', 'description', 'shortDescription']
    graphInfoKeys = [ 'graphID', 'fPosJ', 'p', 'C', 'd']
    parametersInfo_Line = " ".join([str(parameter) + "=" + str(int(value) if value.isdigit() else float(value) if value.replace('-','',1).replace('.', '', 1).isdigit() else value) for parameter, value in simData['configuration']['parameters'].items() if parameter not in graphInfoKeys+otherInfoKeys])
    graphInfo_Line = " ".join([str(parameter) + "=" + str(int(value) if value.isdigit() else float(value) if value.replace('.', '', 1).isdigit() else value) for parameter, value in simData['configuration']['parameters'].items() if parameter in graphInfoKeys])
    
    
    if simData['configuration']['simulationTypeId']== 10:
        graphInfo_Line = "FM "+graphInfo_Line

    mcEq = simData['configuration']['mcParameters']['MCeq']
    mcMeas = simData['configuration']['mcParameters']['MCmeas']
    measuresInfo_Line = r"MC$_{eq}$="+f"{mcEq}"+" "+r"MC$_{meas}$="+f"{mcMeas}"

    settingInfo_Line = parametersInfo_Line+"\n"+ graphInfo_Line+"\n"+measuresInfo_Line
    refConInfo_Line = "refConf:"+(simData['configuration']['referenceConfigurationsInfo']['shortDescription'])
    initInfo_Line = refConInfo_Line

    def addInfoLines():       #useful for following plots
        xlabel_position = plt.gca().xaxis.label.get_position()
        plt.text(0, xlabel_position[1] - 0.18, settingInfo_Line, fontsize=7, ha='left', va='center', transform=plt.gca().transAxes)
        plt.text(1, xlabel_position[1] - 0.18, initInfo_Line, fontsize=7, ha='right', va='center', transform=plt.gca().transAxes)
    #setting results folders and plots default lines: END

    #ANALYSIS OF TI FILES: START
    results = {}
    results["TI"] ={"beta":[]}

    TIFile = get_file_with_prefix(run_Path, "TIbeta")
    if TIFile is not None:
        with open(TIFile, 'r') as file:
            numbers = [float(num) for num in file.readline().replace(',', '.').split() if any(c.isdigit() or c == '.' for c in num)]
        results["TI"]["beta"] = numbers[-1]
    else:
        results["TI"]["beta"] = "nan"

    #ANALYSIS OF THERMALIZATION DATA: START
    thermCheck_filePath = get_file_with_prefix(run_Path, "thermCheck")
    if thermCheck_filePath is None:
        print("No thermCheck file found")
        return None

    theseFiguresFolder= os.path.join(plotsFolder, "thermalization")
    if not os.path.exists(theseFiguresFolder):
        os.makedirs(theseFiguresFolder)
    else:
        delete_files_in_folder(theseFiguresFolder)

    titleSpecification = "as from thermalization data"


    therm_H = []
    therm_HB = []


    mcMeas= int(simData['configuration']["mcParameters"]["MCmeas"])
    mcEq= int(simData['configuration']["mcParameters"]["MCeq"])
    firstIndexOfMeasuresAtEq = (int)(np.ceil(mcEq/mcMeas))

    measuresToSkipInOOEPlot = 1      #The first measure may be very different even wrt non-eq ones, as it contains the initialization trajectory
    
    with open(thermCheck_filePath, 'r') as file:
        #print("analizzando ", nome_file)
        lines = file.readlines()
        dataLines = filter(lambda x: not x.startswith('#'), lines)
        data = np.genfromtxt(dataLines, delimiter=' ')
        therm_mcMeasures=data[:,0]
        therm_H=data[:,1]
        therm_HB=data[:,2]

    therm_mcMeasures = np.asarray(therm_mcMeasures, dtype=np.float64)
    therm_H = np.asarray(therm_H, dtype=np.float64)
    therm_HB = np.asarray(therm_HB, dtype=np.float64)


    if len(therm_mcMeasures)==1:
        print("Not enough measures")
        return None

    results["thermalization"] = {}
    nMusToConsider = 40
    titleForAutocorrelations = "autocorrelation over mcSweeps"

        #defining a function to plot quantities evolution over mc iterations and the respective autocorrelation
    def mcEvolutionAndAutocorrelation(mcSweeps, quantity, firstIndexForEquilibrium,
                                      quantityShortName, quantityFullName, quantityLabelName, nMus):
        
        results["thermalization"][quantityShortName] = {}

        plt.figure(quantityShortName)
        plt.title(quantityFullName+" vs MC\n"+titleSpecification)
        plt.plot(mcSweeps, quantity)
        plt.xlabel("MC sweep")
        plt.ylabel(quantityLabelName)

        #plt.axvline(x=mcEq, color='red', linestyle='--')
        #plt.text(mcEq, plt.ylim()[1], 'MCeq', color='red', verticalalignment='bottom', horizontalalignment='right', fontsize=7)

        addInfoLines()

        results["thermalization"][quantityShortName]["mean"] = np.mean(quantity[firstIndexForEquilibrium:])
        results["thermalization"][quantityShortName]["stdErr"] = stats.sem(quantity[firstIndexForEquilibrium:])

        if (len(np.unique(quantity[firstIndexForEquilibrium:])) > 1): #i.e., if it s not a constant
            mu, muErr, rChi2, dof  = autocorrelationWithExpDecayAndMu(quantityShortName+"Autocorrelation", quantityFullName+" "+titleForAutocorrelations,
                                        mcSweeps[firstIndexForEquilibrium:], "mc", quantity[firstIndexForEquilibrium:], quantityLabelName,
                                        nMus)
            addInfoLines()
            results["thermalization"][quantityShortName]["mu"] = mu
            results["thermalization"][quantityShortName]["muErr"] = muErr
            results["thermalization"][quantityShortName]["rChi2"] = rChi2
            results["thermalization"][quantityShortName]["dof"] = dof
        else:
            results["thermalization"][quantityShortName]["mu"] = "nan"
            results["thermalization"][quantityShortName]["muErr"] = "nan"
            results["thermalization"][quantityShortName]["rChi2"] = "nan"
            results["thermalization"][quantityShortName]["dof"] = "nan"

    mcEvolutionAndAutocorrelation(therm_mcMeasures[:len(therm_mcMeasures)], therm_H[:len(therm_mcMeasures)], firstIndexOfMeasuresAtEq,
                                      "H", "trajectory average energy", "energy", nMusToConsider)
    
    mcEvolutionAndAutocorrelation(therm_mcMeasures, therm_HB, firstIndexOfMeasuresAtEq,
                                      "HB", "trajectory max energy", "energy", nMusToConsider)

    figs = plt.get_figlabels()  # Ottieni i nomi di tutte le figure create
    for fig_name in figs:
        fig = plt.figure(fig_name)
        filename = os.path.join(theseFiguresFolder, f'{fig_name}.png')
        fig.savefig(filename, bbox_inches='tight')
    plt.close('all')

    simData["results"]=results
    writeJsonResult(simData, os.path.join(resultsFolder,"runData.json"))

def singleRunAnalysis(run_Path="", goFast=False):

    standardMCSimIDs = [15]
    pathMCSimIDs = [10,100,11,110]
    if run_Path=="":
        run_Path=preset_Path
    print("Analysis of "+run_Path+"\n\n")

    #getting run infos: START
    map_file_path = "../../Data/infoMap.json"  
    file_path = get_file_with_prefix(run_Path, "info.")

    
    if file_path is None:   #this is to do the analysis only if there are info about the simulation
        file_path = get_file_with_prefix(run_Path, "info_")
        if file_path is None:
            file_path = get_file_with_prefix(run_Path, "details_")
            if file_path is None:
                print("No info on simulation found.")
                return None

    with open(map_file_path, 'r') as map_file:
        mappa = json.load(map_file)

    configurationInfo= txtToInfo(file_path, mappa)
    simTypeID = configurationInfo['simulationTypeId']
    if simTypeID in pathMCSimIDs:
        singlePathMCAnalysis(run_Path=run_Path, configurationsInfo=configurationInfo)
    elif simTypeID in standardMCSimIDs:
        singleStandardMCAnalysis(run_Path=run_Path, configurationInfo=configurationInfo)
