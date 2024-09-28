import glob
import os
import re
import sys

from singlePTAnalysis import singlePTAnalysis

nameOfFoldersContainingRuns=["graph8400","graph8412", "graph8415","graph8417"]#,"graph1149","graph5492","graph9450","graph5508","graph3991","graph2473", "graph9351", "graph9366", "graph5657"]

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
                if dir_name in nameOfFoldersContainingRuns:
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

def get_substring_up_to_first_slash(input_string):
    index = input_string.find('/')
    if index != -1:
        return input_string[:index]
    else:
        return input_string

if len(sys.argv) > 1:
    analysisVsSimTypesDict = {"DPRRG":"DPRRG", "RRG":"RRG", "ER":"ER", "SqLatt":"SqLatt"}

    analysisType = get_substring_up_to_first_slash(sys.argv[1])
    additional_strings= sys.argv[2:]

    if analysisType in analysisVsSimTypesDict:
        simType = analysisVsSimTypesDict[analysisType]
        archive_path = f"../../Data/Graphs"
        if analysisType!="all":
            archive_path+="/"+sys.argv[1]
    else:
        print(f"Analysis of type {analysisType} not implemented.\n")
        print("Implemented analysis types include:")
        for implementedAnalysis in analysisVsSimTypesDict:
            print(implementedAnalysis)
        print("\n")
        exit()

        
    selected_PTs = findFoldersWithString(archive_path, additional_strings)

    if not selected_PTs:
        raise FileNotFoundError(f"No files of type  found in the specified path.")
    print(f"Analyzing all PTs of type {simType}. {len(selected_PTs)} PTs found.")
    i=0
    for parTemp in sorted(selected_PTs, reverse=True):
        i+=1
        print(f"Analyzing simulation #{i} out of {len(selected_PTs)}\n")
        singlePTAnalysis(parTemp)
    # Get the stories names in the folder
    print("Analysis completed.\n")
else:
    singlePTAnalysis("")