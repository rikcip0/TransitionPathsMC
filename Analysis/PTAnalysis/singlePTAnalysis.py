#Analysis of parallel tempering
import sys
sys.path.append('../')
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import os
import json

from MyBasePlots.hist import myHist
from MyBasePlots.hist import myHistForOverlaps2
from MyBasePlots.hist import myHistForOverlaps_notLog

matplotlib.use('Agg') 

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

def writeJsonResult(data, filePath):
    json_data = json.dumps(data, indent=4)
    with open(filePath, 'w') as output_file:
        output_file.write(json_data)

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
    global simulationCode_version
    simulationCode_version = firstLine_tokens[1]
    # Trova le informazioni sulla simulazione nella mappa
    simulationType_info = mappa["simulationTypes"].get(simulation_type, {})
    simulationTypeVersion_info = mappa["simulationTypes"].get(simulation_type, {}).get("versions", {}).get(simulationCode_version, None)
    if (simulationType_info is None) or simulationTypeVersion_info is None:
        print(f"Tipo di simulazione o versione non trovato nella mappa per '{simulation_type} - Versione {simulationCode_version}'.")
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
    simulationCode_version = (int) (simulationCode_version)
    return data

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

def find_files_with_string(parent_dir, target_string):
    matching_files = []
    # Traverse through all subdirectories recursively
    for root, dirs, files in os.walk(parent_dir):
        for file in files:
            # Check if the directory name contains the target string
            if target_string in file:
                matching_files.append(os.path.join(root, file))
    
    return matching_files

def oldSinglePTAnalysis( folder=""):
    
    print(folder)
    resultsFolder = os.path.join(folder, "Analysis")
    os.makedirs(resultsFolder, exist_ok=True)
    delete_files_in_folder(resultsFolder)


    aConfsFile_Path = find_files_with_string(folder, "confs")[0]

    with open(aConfsFile_Path, 'r') as file:
        next(file)
        # Read the second line
        second_line = file.readline().strip().split()

    N=len(second_line)

    #swap rate plot: start
    PTInfo_path = os.path.join(folder,"PTInfo.txt")
    if not (os.path.exists(PTInfo_path)):
        return None
    temperatures = []
    rates = []

    nHeaderLines = 1  #lines to skip before receing actual data

    with open(PTInfo_path, 'r') as file:
            #print("analizzando ", nome_file)
            lines = file.readlines()
            lines = lines[nHeaderLines:]
            dataLines = filter(lambda x: not x.startswith('#'), lines)
            data = np.genfromtxt(dataLines, delimiter=' ')
            temperatures = data[:,0]
            rates=data[:,1]

    temperatures=np.asarray(temperatures)
    rates=np.asarray(rates)
    meanPoint = (temperatures[:-1] + temperatures[1:]) / 2  # Calcolo dei punti medi degli intervalli

    plt.figure("swapRates")
    plt.title("Swap rates for the parallel tempering")
    plt.xlabel(r"$\beta$")
    plt.ylabel("swap rate")
    plt.grid(True)
    plt.scatter(meanPoint, rates[:-1])
    plt.axhline(0.23, color='red', linestyle='dashed', linewidth=1, label=r"0.23")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    #selected_xticks = temperatures[[0, 2, 4, -1]]  # Selective xtick values to display labels
    #plt.xticks(temperatures, [str(x)[:4] if x in selected_xticks else '' for x in temperatures])

    plt.figure("chosenBetas")
    plt.title("Betas chosen for the parallel tempering")
    plt.xlabel(r"#Histogram")
    plt.ylabel(r"$\beta$")
    plt.grid(True)
    plt.plot(temperatures, marker='s')

    #swap rate plot: end

    qsPath = find_files_with_string(folder, "Qs")

    overlaps = []

    for qPath in qsPath:

        overlapsAtBeta = []
        overlapsAtBetaMinusAutoOv = []

        with open(qPath, 'r') as file:
                #print("analizzando ", nome_file)
                lines = file.readlines()
                beta = (float)(lines[0])
                lines = lines[1:]
                dataLines = filter(lambda x: not x.startswith('#'), lines)
                data = np.genfromtxt(dataLines, delimiter=' ')
                for i in range(data.shape[1]):
                    overlapsAtBetaOfConf = np.asarray(data[:,i])
                    overlapsAtBetaOfConfMinusAutoOv = np.delete(overlapsAtBetaOfConf, i)
                    overlapsAtBeta.append(overlapsAtBetaOfConf)
                    overlapsAtBetaMinusAutoOv.append(overlapsAtBetaOfConfMinusAutoOv)
        
        overlaps.append(overlapsAtBeta)

        overlapsAtBetaMinusAutoOv=np.asarray(overlapsAtBetaMinusAutoOv)


        myHistForOverlaps2(f"QsNormalized_HistogramBeta{beta:.2f}",
                           "Fraction of configurations extracted at Q wrt all configurations at Q,\n,and between a certain configuration and the others, at"+r"$\beta$="+f"{beta:.3g}",
               overlapsAtBetaMinusAutoOv, "Q", N)

        myHistForOverlaps_notLog(f"QsHistogramBeta{beta:.2f}",
                           "Histogram of overlaps among extracted configurations,\n and between a certain configuration and the others, at "+r"$\beta$="+f"{beta:.3g}",
               overlapsAtBetaMinusAutoOv, "Q", N)

    figs = plt.get_figlabels()  # Ottieni i nomi di tutte le figure create
    for fig_name in figs:
        fig = plt.figure(fig_name)
        filename = os.path.join(resultsFolder, f'{fig_name}.png')
        fig.savefig(filename, bbox_inches='tight')
    plt.close('all')

    file_path= os.path.join(folder, "PTEnergies.txt")

    betaHist = []

    with open(file_path, 'r') as file:
            #print("analizzando ", nome_file)
            lines = file.readlines()
            dataLines = filter(lambda x: not x.startswith('#'), lines)
            data = np.genfromtxt(dataLines, delimiter=' ')
            for i in range(2, 2+temperatures.size):
                betaHist.append(data[:,i])


    betaHist = np.asarray(betaHist)
    bins=np.linspace(np.min(betaHist.flatten()),np.max(betaHist.flatten())+1, (int)(13*np.sqrt(temperatures.size)))
    plt.figure("EnergyHistograms")
    plt.title("histograms of energies of extracted configurations")
    for i in range(0, temperatures.size):
        plt.hist(betaHist[i], alpha=0.23, label = r"$\beta$="+f"{temperatures[i]}", bins=bins, edgecolor='black')
    
    plt.xlabel("energy")
    plt.yscale('log')
    plt.ylabel("n. of occurrences")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

    nOfHighestBetasToConsider = 5
    bins=np.linspace(np.min(betaHist[-nOfHighestBetasToConsider:].flatten()),np.max(betaHist[-nOfHighestBetasToConsider:].flatten())+1, (int)(3*nOfHighestBetasToConsider))
    plt.figure("EnergyHistograms_zoomOnHighBetas")
    plt.title(f"histograms of energies of extracted configurations\n zoom on {nOfHighestBetasToConsider} highest betas")
    for i in range(1, nOfHighestBetasToConsider+1):
        plt.hist(betaHist[-1], alpha=0.5, label = r"$\beta$="+f"{temperatures[-1]}", bins=bins)
    plt.xlabel("energy")
    plt.yscale('log')
    plt.ylabel("n. of occurrences")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
#    plt.xlim(np.min(betaHist[-1])-0.5, np.max(betaHist[-nOfHighestBetasToConsider])+0.5)

    figs = plt.get_figlabels()  # Ottieni i nomi di tutte le figure create
    for fig_name in figs:
        fig = plt.figure(fig_name)
        filename = os.path.join(resultsFolder, f'{fig_name}.png')
        fig.savefig(filename, bbox_inches='tight')
    plt.close('all')

def newSinglePTAnalysis( folder=""):

    print(folder)
    resultsFolder = os.path.join(folder, "Analysis")
    os.makedirs(resultsFolder, exist_ok=True)
    delete_files_in_folder(resultsFolder)

    #getting run infos: START
    map_file_path = "../../Data/PTInfoMap.json"  
    file_path = get_file_with_prefix(folder, "info.dat")



    with open(map_file_path, 'r') as map_file:
        mappa = json.load(map_file)

    configurationInfo= txtToInfo(file_path, mappa)

    aConfsFile_Path = find_files_with_string(folder, "confs")[0]

    with open(aConfsFile_Path, 'r') as file:
        next(file)
        # Read the second line
        second_line = file.readline().strip().split()

    N=len(second_line)

    #swap rate plot: start
    PTInfo_path = os.path.join(folder,"energies.txt")
    if not (os.path.exists(PTInfo_path)):
        return None
    temperatures = []
    rates = []

    nHeaderLines = 1  #lines to skip before receing actual data

    with open(PTInfo_path, 'r') as file:
            #print("analizzando ", nome_file)
            lines = file.readlines()
            lines = lines[nHeaderLines:]
            dataLines = filter(lambda x: not x.startswith('#'), lines)
            data = np.genfromtxt(dataLines, delimiter=' ')
            temperatures = data[:,0]
            rates=data[:,1]

    temperatures=np.asarray(temperatures)
    rates=np.asarray(rates)
    meanPoint = (temperatures[:-1] + temperatures[1:]) / 2  # Calcolo dei punti medi degli intervalli

    plt.figure("swapRates")
    plt.title("Swap rates for the parallel tempering")
    plt.xlabel(r"$\beta$")
    plt.ylabel("swap rate")
    plt.grid(True)
    plt.scatter(meanPoint, rates[:-1])
    plt.axhline(0.23, color='red', linestyle='dashed', linewidth=1, label=r"0.23")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    #selected_xticks = temperatures[[0, 2, 4, -1]]  # Selective xtick values to display labels
    #plt.xticks(temperatures, [str(x)[:4] if x in selected_xticks else '' for x in temperatures])

    plt.figure("chosenBetas")
    plt.title("Betas chosen for the parallel tempering")
    plt.xlabel(r"#Histogram")
    plt.ylabel(r"$\beta$")
    plt.grid(True)
    plt.plot(temperatures, marker='s')

    #swap rate plot: end

    qsPath = find_files_with_string(folder, "Qs")

    overlaps = []

    for qPath in qsPath:

        overlapsAtBeta = []
        overlapsAtBetaMinusAutoOv = []

        with open(qPath, 'r') as file:
                #print("analizzando ", nome_file)
                lines = file.readlines()
                beta = (float)(lines[0])
                lines = lines[1:]
                dataLines = filter(lambda x: not x.startswith('#'), lines)
                data = np.genfromtxt(dataLines, delimiter=' ')
                for i in range(data.shape[1]):
                    overlapsAtBetaOfConf = np.asarray(data[:,i])
                    overlapsAtBetaOfConfMinusAutoOv = np.delete(overlapsAtBetaOfConf, i)
                    overlapsAtBeta.append(overlapsAtBetaOfConf)
                    overlapsAtBetaMinusAutoOv.append(overlapsAtBetaOfConfMinusAutoOv)
        
        overlaps.append(overlapsAtBeta)

        overlapsAtBetaMinusAutoOv=np.asarray(overlapsAtBetaMinusAutoOv)


        myHistForOverlaps2(f"QsNormalized_HistogramBeta{beta:.2f}",
                           "Fraction of configurations extracted at Q wrt all configurations at Q,\n,and between a certain configuration and the others, at"+r"$\beta$="+f"{beta:.3g}",
               overlapsAtBetaMinusAutoOv, "Q", N)

        myHistForOverlaps_notLog(f"QsHistogramBeta{beta:.2f}",
                           "Histogram of overlaps among extracted configurations,\n and between a certain configuration and the others, at "+r"$\beta$="+f"{beta:.3g}",
               overlapsAtBetaMinusAutoOv, "Q", N)

    figs = plt.get_figlabels()  # Ottieni i nomi di tutte le figure create
    for fig_name in figs:
        fig = plt.figure(fig_name)
        filename = os.path.join(resultsFolder, f'{fig_name}.png')
        fig.savefig(filename, bbox_inches='tight')
    plt.close('all')

    file_path= os.path.join(folder, "PTEnergies.txt")

    betaHist = []

    with open(file_path, 'r') as file:
            #print("analizzando ", nome_file)
            lines = file.readlines()
            dataLines = filter(lambda x: not x.startswith('#'), lines)
            data = np.genfromtxt(dataLines, delimiter=' ')
            for i in range(2, 2+temperatures.size):
                betaHist.append(data[:,i])


    betaHist = np.asarray(betaHist)
    bins=np.linspace(np.min(betaHist.flatten()),np.max(betaHist.flatten())+1, (int)(13*np.sqrt(temperatures.size)))
    plt.figure("EnergyHistograms")
    plt.title("histograms of energies of extracted configurations")
    for i in range(0, temperatures.size):
        plt.hist(betaHist[i], alpha=0.23, label = r"$\beta$="+f"{temperatures[i]}", bins=bins, edgecolor='black')
    
    plt.xlabel("energy")
    plt.yscale('log')
    plt.ylabel("n. of occurrences")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

    nOfHighestBetasToConsider = 5
    bins=np.linspace(np.min(betaHist[-nOfHighestBetasToConsider:].flatten()),np.max(betaHist[-nOfHighestBetasToConsider:].flatten())+1, (int)(3*nOfHighestBetasToConsider))
    plt.figure("EnergyHistograms_zoomOnHighBetas")
    plt.title(f"histograms of energies of extracted configurations\n zoom on {nOfHighestBetasToConsider} highest betas")
    for i in range(1, nOfHighestBetasToConsider+1):
        plt.hist(betaHist[-1], alpha=0.5, label = r"$\beta$="+f"{temperatures[-1]}", bins=bins)
    plt.xlabel("energy")
    plt.yscale('log')
    plt.ylabel("n. of occurrences")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
#    plt.xlim(np.min(betaHist[-1])-0.5, np.max(betaHist[-nOfHighestBetasToConsider])+0.5)

    figs = plt.get_figlabels()  # Ottieni i nomi di tutte le figure create
    for fig_name in figs:
        fig = plt.figure(fig_name)
        filename = os.path.join(resultsFolder, f'{fig_name}.png')
        fig.savefig(filename, bbox_inches='tight')
    plt.close('all')

def singlePTAnalysis( folder=""):
    PTInfo_path = os.path.join(folder,"info.dat")
    if (os.path.exists(PTInfo_path)):
        newSinglePTAnalysis(folder)
        print("doinf new")
    else:
        PTInfo_path = os.path.join(folder,"PTInfo.txt")
        if not(os.path.exists(PTInfo_path)):
            print("doinf no")
            return None
        print("doinf old")
        oldSinglePTAnalysis(folder)