import glob
import json
import os
import re
import sys

import numpy as np

from singleMultiRunAnalysis import singleMultiRunAnalysis

archive_path = f"../../Data/Graphs"
nameOfFoldersContainingRuns = ["stdMCs", "PathsMCs"
                                ]

def findFoldersWithString(parent_dir, target_strings):
    result = []
    
    # Funzione ricorsiva per cercare le cartelle
    def search_in_subfolders(directory, livello=1):
        if livello > 10:
            return
        rightLevelReached = False
        for root, dirs, _ in os.walk(directory):
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

if len(sys.argv) != 1:
    analysisVsSimTypesDict = {"all":"any","ER":"ER", "RRG":"RRG"}
    
    requestedAnalysis = sys.argv[1]
    additional_strings= sys.argv[2:]

    noUpdate = ("noUpdate" in additional_strings)
    if noUpdate:
        additional_strings.remove("noUpdate")

    if requestedAnalysis in analysisVsSimTypesDict:
        if requestedAnalysis!="all":
            simType = analysisVsSimTypesDict[requestedAnalysis]
        else:
            simTypes = ["ER", "RRG"]
    if os.path.exists(archive_path+"/"+requestedAnalysis):
        archive_path+="/"+requestedAnalysis
        simTypes= [requestedAnalysis.split('/')[0].split('\\')[0]]
    else:
        print(f"Analysis of type {requestedAnalysis} not implemented.")
        print(f"Also, "+archive_path+"/"+requestedAnalysis+" folder does not exist.\n")
        print("Implemented analysis types include:")
        for implementedAnalysis in analysisVsSimTypesDict:
            print(implementedAnalysis)
        print("\n")
        exit()

    for i, simType in enumerate(simTypes):

        print("analyzing ", simType)

        archivedSingleRuns = findFoldersWithString(archive_path, ['_run',*additional_strings])
        parentFolderForResults = f"../../Data/MultiRun/"+requestedAnalysis
        os.makedirs(parentFolderForResults, exist_ok=True)
        presentRunsData = []
        """
        previousRuns_JsonFile =parentFolderForResults+"/runsData.json"
        if os.path.exists(previousRuns_JsonFile):
            with open(previousRuns_JsonFile, 'r') as file:
                presentRunsData = json.load(file)
        """

        oldDataIDs=[]
        for run in presentRunsData:
            oldDataIDs.append(run['configuration']['ID'])
        oldDataIDs=np.array(oldDataIDs)

        newRunsData = []
        for run in sorted(archivedSingleRuns):
        #EXTRACTING INPUT PARAMETERS
            # Open the text file with infos on the input in read mode
            if os.path.exists(run+ "/Results/runData.json"):
                    try:
                        with open(run+ "/Results/runData.json", 'r') as file:
                            #print("analyzing"+run)
                            runData = json.load(file)
                            id_run = runData['configuration']['ID']
                            presente = any(item['configuration']['ID'] == id_run for item in presentRunsData)
                            if not presente:
                                newRunsData.append(runData)
                    except FileNotFoundError:
                            print("Errore nella lettura di una simulazione\n")
        presentRunsData.extend(newRunsData)

        #with open(previousRuns_JsonFile, "w") as json_file:
                #json.dump(presentRunsData, json_file, indent=4) 

        if not presentRunsData:
            raise FileNotFoundError(f"No files of type  found in the specified path.")
        print(f"Analyzing all run groups of type {simType}. {len(simTypes)} groups found.")

        print(f"Analyzing run group #{i} ({simTypes[i]}) out of {len(simTypes)}\n")
        singleMultiRunAnalysis(presentRunsData, parentFolderForResults, simType)
        # Get the stories names in the folder
        print("Analysis completed.\n")
else:
    print("Analysis type not specified.")
