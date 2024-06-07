import glob
import json
import os
import re
import sys

import numpy as np

from singleMultiRunAnalysis import singleMultiRunAnalysis

def find_directories_with_strings(parent_dir, target_strings):
    matching_dirs = []
    # Traverse through all subdirectories recursively
    for root, dirs, files in os.walk(parent_dir):
        for directory in dirs:
            # Check if the directory name contains the first target string
            if target_strings[0] in directory:
                # Check if any of the eventual other strings are in any part of the full name
                if all(string in os.path.join(root,directory) for string in target_strings[1:]):
                    matching_dirs.append(os.path.join(root, directory))
    
    return matching_dirs


def find_directories_with_string(parent_dir, target_string):
    matching_dirs = []
    # Traverse through all subdirectories recursively
    for root, dirs, files in os.walk(parent_dir):
        for directory in dirs:
            # Check if the directory name contains the target string
            if target_string in directory:
                matching_dirs.append(os.path.join(root, directory))
    return matching_dirs

if len(sys.argv) > 1:
    analysisVsSimTypesDict = {"all":"all","ER":"ER", "RRG":"RRG"}
    analysisType = sys.argv[1]
    additional_strings= sys.argv[2:]

    if analysisType not in analysisVsSimTypesDict:
        print(f"Analysis of type {analysisType} not implemented.\n")
        print("Implemented analysis types include:")
        for implementedAnalysis in analysisVsSimTypesDict:
            print(implementedAnalysis)
        print("\n")
        exit()
    
    if analysisType == "all":
        simTypes = ["ER", "RRG"]
    else:
        simTypes = [analysisVsSimTypesDict[analysisType]]

    for simType in simTypes:

        print("analyzing ", simType)
        singleRunsArchive_path = f"../../Data/Graphs/"+simType

        archivedSingleRuns = find_directories_with_string(singleRunsArchive_path, 'run')


        parentFolder = f"../../Data/MultiRun/"+simType
        os.makedirs(parentFolder, exist_ok=True)
        previousRuns_JsonFile =parentFolder+"/runsData.json"
        if os.path.exists(previousRuns_JsonFile):
            with open(previousRuns_JsonFile, 'r') as file:
                presentRunsData = json.load(file)
        else:
            presentRunsData = []

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

        selected_runGroups = [
        [
            [
                item for item in presentRunsData if (float(item["configuration"]["parameters"]["fPosJ"]) == x[0] and float(item["configuration"]["parameters"]["C"]) == x[1])
            ],
            f"p2C{x[1]}fPosJ{x[0]:.2f}"
        ]
        for x in [[0.5,4],[0.5,3], [1.0,3], [1.0,4]]
        ]

        selected_runGroups= [runGroup for runGroup in selected_runGroups if len(runGroup[0])>1]

        if not selected_runGroups:
            raise FileNotFoundError(f"No files of type  found in the specified path.")
        print(f"Analyzing all run groups of type {simType}. {len(selected_runGroups)} groups found.")

        i=0
        for i, runGroup in enumerate(selected_runGroups):

            analysis_path = os.path.join(parentFolder, runGroup[1])
            print(f"Analyzing run group #{i} ({runGroup[1]}) out of {len(selected_runGroups)}\n")
            singleMultiRunAnalysis(runGroup[0], analysis_path,simType)
        # Get the stories names in the folder
        print("Analysis completed.\n")
else:
    singleMultiRunAnalysis("")
