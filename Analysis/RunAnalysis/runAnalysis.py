import glob
import os
import re
import sys

from singleRunAnalysis import singleRunAnalysis

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

if len(sys.argv) > 1:
    analysisVsSimTypesDict = {"all": "any", "RRG":"RRG", "ER":"ER"}
    analysisType = sys.argv[1]
    additional_strings= sys.argv[2:]

    if analysisType in analysisVsSimTypesDict:
        simType = analysisVsSimTypesDict[analysisType]
        archive_path = f"../../Data/Graphs"
        if analysisType!="all":
            archive_path+="/"+analysisType
    else:
        print(f"Analysis of type {analysisType} not implemented.\n")
        print("Implemented analysis types include:")
        for implementedAnalysis in analysisVsSimTypesDict:
            print(implementedAnalysis)
        print("\n")
        exit()

    selected_runs = find_directories_with_strings(archive_path, ['_run', *additional_strings])
    if not selected_runs:
        raise FileNotFoundError(f"No files of type  found in the specified path.")
    
    print(f"Analyzing all runs of type {simType}. {len(selected_runs)} runs found.")

    i=0
    for run in sorted(selected_runs, reverse=True):
        i+=1 
        print(f"Analyzing simulation #{i} out of {len(selected_runs)}\n")
        singleRunAnalysis(run)
    # Get the stories names in the folder
    print("Analysis completed.\n")
    
else:
    singleRunAnalysis("")