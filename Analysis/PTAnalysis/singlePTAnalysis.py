#Analysis of parallel tempering
import sys
sys.path.append('../')
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import os

from MyBasePlots.hist import myHist
matplotlib.use('Agg') 

preset_folder = "../../Data/Graphs/RRG/p2C4/N80/structure936212/fPosJ0.50/graph3991/configurations6"

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

def singlePTAnalysis(folder=""):
    
    if folder =="":
        folder= preset_folder
    print(folder)
    plotsFolder = os.path.join(folder, "Analysis")
    os.makedirs(plotsFolder, exist_ok=True)
    delete_files_in_folder(plotsFolder)


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
    selected_xticks = temperatures[[0, 2, 4, -1]]  # Selective xtick values to display labels
    plt.xticks(temperatures, [str(x)[:4] if x in selected_xticks else '' for x in temperatures])

    plt.figure("chosenBetas")
    plt.title("Betas chosen for the parallel tempering")
    plt.xlabel(r"#Histogram")
    plt.ylabel(r"$\beta$")
    plt.grid(True)
    plt.plot(temperatures, marker='')
    #swap rate plot: end

    qsPath = find_files_with_string(folder, "Qs")
    for qPath in qsPath:
        overlaps = []

        with open(qPath, 'r') as file:
                #print("analizzando ", nome_file)
                lines = file.readlines()
                beta = (float)(lines[0])
                lines = lines[1:]
                dataLines = filter(lambda x: not x.startswith('#'), lines)
                data = np.genfromtxt(dataLines, delimiter=' ')
                for i in range(data.shape[1]):
                    overlapsMinusAutoOv = np.asarray(data[:,i])
                    overlapsMinusAutoOv = np.delete(overlapsMinusAutoOv, i)
                    overlaps.append(overlapsMinusAutoOv)

        overlaps=np.asarray(overlaps)
        plt.figure(f"QsHistogramBeta{beta:.2f}")
        myHist(f"QsHistogramBeta{beta:.2f}", "Histogram of overlaps among extracted configurations\n"+r"$\beta$="+f"{beta:.3g}",
               overlaps.flatten(), "Q", nbins=len(np.unique(overlaps.flatten())))

    figs = plt.get_figlabels()  # Ottieni i nomi di tutte le figure create
    for fig_name in figs:
        fig = plt.figure(fig_name)
        filename = os.path.join(plotsFolder, f'{fig_name}.png')
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


    plt.figure("EnergyHistograms")
    plt.title("histograms of energies of extracted configurations")
    for i in range(0, temperatures.size):
        plt.hist(betaHist[i], alpha=0.5, label = r"$\beta$="+f"{temperatures[i]}", bins=len(np.unique(betaHist[i].flatten())))
    plt.xlabel("energy")
    plt.yscale('log')
    plt.ylabel("n. of occurrences")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

    plt.figure("EnergyHistograms_zoomOnHighBetas")
    plt.title("histograms of energies of extracted configurations\n zoom on 4 higher betas")
    for i in range(0, temperatures.size):
        plt.hist(betaHist[i], alpha=0.5, label = r"$\beta$="+f"{temperatures[i]}")
    plt.xlabel("energy")
    plt.yscale('log')
    plt.ylabel("n. of occurrences")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.xlim(np.min(betaHist[-1])-0.5, np.max(betaHist[-4])+0.5)

    figs = plt.get_figlabels()  # Ottieni i nomi di tutte le figure create
    for fig_name in figs:
        fig = plt.figure(fig_name)
        filename = os.path.join(plotsFolder, f'{fig_name}.png')
        fig.savefig(filename, bbox_inches='tight')
    plt.close('all')
