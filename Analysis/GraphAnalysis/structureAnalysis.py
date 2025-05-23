import glob
import os
import re
import sys

from singleStructureAnalysis import singleStructureAnalysis
from singleGraphFieldAnalysis import singleGraphFieldAnalysis

nameOfFoldersContainingRequested=None
archive_path = f"../../Data/Graphs/"
stringToFindGraph='graph'
nameOfFoldersContainingGraphs = ["fPosJ"
                               ]
nameOfFoldersContainingGraphFields = ["stdGaussian","stdBernoulli",
                               ]

presetPath = "somePath"

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
                if any(folder in dir_name for folder in nameOfFoldersContainingRequested):
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


if len(sys.argv) > 1:
    analysisVsSimTypesDict = {"graph":"graphs","field":"fields"}
    ao = sys.argv[1]
    analysisType = sys.argv[2]
    additional_strings= sys.argv[3:]

    if analysisType=="all":
        simType=["fields","graphs"]
    elif analysisType in analysisVsSimTypesDict:            
        simType = [analysisVsSimTypesDict[analysisType]]
    elif os.path.exists(archive_path+"/"+analysisType):
        simType= [analysisType]
        archive_path+="/"+analysisType
    else:
        print(f"Analysis of type {analysisType} not implemented.")
        print(f"Also, "+archive_path+"/"+analysisType+" folder does not exist.\n")
        print("Implemented analysis types include:")
        for implementedAnalysis in analysisVsSimTypesDict:
            print(implementedAnalysis)
        print("and all of them together.\n")
        print("\n")
        exit()
    for s in simType:
        if s=="fields":
            nameOfFoldersContainingRequested=nameOfFoldersContainingGraphFields
            stringToFind="realization1"
            analysisToDo=singleGraphFieldAnalysis
        elif s=="graphs":
            nameOfFoldersContainingRequested=nameOfFoldersContainingGraphs
            stringToFind= stringToFindGraph
            analysisToDo=singleStructureAnalysis
        print("AA")
        selected_runs = findFoldersWithString(archive_path, [stringToFindGraph, *additional_strings])
        print("AA2")
        if not selected_runs:
            raise FileNotFoundError(f"No files of type  found in the specified path.")
        
        print(f"Analyzing all runs of type {simType}. {len(selected_runs)} runs found.")

        for i, run in enumerate(sorted(selected_runs, reverse=True)):
            print(f"Analyzing graph #{i+1} out of {len(selected_runs)}\n")
            analysisToDo(run)
        # Get the stories names in the folder
        print("Analysis completed.\n")
    
else:
    print("AO")