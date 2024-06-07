import glob
import os
import re
import sys

from singlePTAnalysis import singlePTAnalysis


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

        
    selected_PTs = find_directories_with_strings(archive_path, ['configurations', *additional_strings])

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