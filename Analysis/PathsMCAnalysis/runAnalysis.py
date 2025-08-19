import glob
import os
import re
import sys

from singleRunAnalysis import singleRunAnalysis

nameOfFoldersContainingRuns = ["stdMCs", 
    "PathsMCs"
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


if len(sys.argv) > 1:
    archive_path = f"../../Data/Graphs"
    analysisVsSimTypesDict = {"all": "any"}
    redo=False
    toPdf=False
    analysisType = sys.argv[1]
    additional_strings= sys.argv[2:]
    threeFitS = None  # il valore finale; il nome non può iniziare con un numero

    for s in additional_strings[:]:      # copia per poter rimuovere
        if s.startswith("threeFitS="):
            val_str = s.split("=", 1)[1] # parte dopo '='
            threeFitS = val_str
            try:
                threeFitS = float(val_str)
            except ValueError:
                pass                     # resta stringa se non è convertibile
            additional_strings.remove(s) # lo tolgo dalla lista
            break
    if "redo" in additional_strings:
        additional_strings.remove("redo")
        redo=True
    if "toPdf" in additional_strings:
        additional_strings.remove("toPdf")
        toPdf=True

    if analysisType in analysisVsSimTypesDict:
        simType = analysisVsSimTypesDict[analysisType]
        if simType!="any":
            archive_path+="/"+analysisType
    elif os.path.exists(archive_path+"/"+analysisType):
        simType=analysisType
        archive_path+="/"+analysisType
    else:
        print(f"Analysis of type {analysisType} not implemented.")
        print(f"Also, "+archive_path+"/"+analysisType+" folder does not exist.\n")
        print("Implemented analysis types include:")
        for implementedAnalysis in analysisVsSimTypesDict:
            print(implementedAnalysis)
        print("\n")
        exit()

    selected_runs = findFoldersWithString(archive_path, ['_run', *additional_strings])
    if not selected_runs:
        raise FileNotFoundError(f"No files of type  found in the specified path.")
    
    print(f"Analyzing all runs of type {simType}. {len(selected_runs)} runs found.")

    for i, run in enumerate(sorted(selected_runs, reverse=True)):
        print(f"Analyzing simulation #{i+1} out of {len(selected_runs)}\n")
        singleRunAnalysis(run,redoIfDone=redo, threeFitS=threeFitS,toPdf=toPdf)
    # Get the stories names in the folder
    print("Analysis completed.\n")
    
else:
    singleRunAnalysis(presetPath)