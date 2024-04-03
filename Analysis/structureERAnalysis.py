#Analysis of parallel tempering
import sys
sys.path.append('../')
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import os

from MyBasePlots.hist import myHist
matplotlib.use('Agg') 

N=100
C=3
preset_folder = "../Data/ER/p2C3/N100/structure559248"

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

def singleDegreeAnalysis(folder=""):
    
    if folder =="":
        folder= preset_folder
    print(folder)
    plotsFolder = os.path.join(folder, "Analysis")
    os.makedirs(plotsFolder, exist_ok=True)
    delete_files_in_folder(plotsFolder)


    #swap rate plot: start
    degreeDist_path = os.path.join(folder,"structure_degreeDistribution.txt")
    if not (os.path.exists(degreeDist_path)):
        return None
    degrees = []

    nHeaderLines = 0  #lines to skip before receing actual data

    with open(degreeDist_path, 'r') as file:
            #print("analizzando ", nome_file)
            lines = file.readlines()
            lines = lines[nHeaderLines:]
            dataLines = filter(lambda x: not x.startswith('#'), lines)
            data = np.genfromtxt(dataLines, delimiter=' ')
            degrees = data[:]

    degrees=np.asarray(degrees)

    plt.figure("degrees")
    plt.title("histograms of energies of extracted configurations")
    plt.xlabel("energy")

    poiss=np.random.poisson(C,N)
    binom=np.random.binomial(N-1,C/(N-1),N)
    plt.hist(poiss, label="Poiss",bins=(int)(poiss.max()-poiss.min()+1))
    plt.hist(binom,label="Binom",bins=(int)(binom.max()-binom.min()+1))
    plt.hist(degrees,bins=(int)(degrees.max()-degrees.min()+1), label="data")
    plt.ylabel("n. of occurrences")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')



    figs = plt.get_figlabels()  # Ottieni i nomi di tutte le figure create
    for fig_name in figs:
        fig = plt.figure(fig_name)
        filename = os.path.join(plotsFolder, f'{fig_name}.png')
        fig.savefig(filename, bbox_inches='tight')
    plt.close('all')

singleDegreeAnalysis("")