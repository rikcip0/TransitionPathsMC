#Analysis of parallel tempering
import random
import sys
import networkx as nx




from matplotlib.colors import LinearSegmentedColormap
sys.path.append('../')
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import os
import json
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix
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

    global simulation_type
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
        print(len((simulationTypeVersion_info["linesMap"])))
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
    simulation_type = (int) (simulation_type)
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

def singleStructureAnalysis( folder=""):
    resultsFolder= os.path.join(folder, 'graphAnalysis')
    os.makedirs(resultsFolder, exist_ok=True)
    jsonResultsPath = os.path.join(resultsFolder,'graphData.json')

    """
    #getting run infos: START
    map_file_path = "../../Data/PTInfoMap.json"  
    file_path = get_file_with_prefix(folder, "info.dat")



    with open(map_file_path, 'r') as map_file:
        mappa = json.load(map_file)

    configurationInfo= txtToInfo(file_path, mappa)

    """
    graphFile_path = find_files_with_string(folder, "graph.txt")[0]
    print(graphFile_path)
    N=0
    list=[]
    orientedEdges=[]
    G = nx.Graph()
    with open(graphFile_path, 'r') as file:
        #print("analizzando ", nome_file)
        lines = file.readlines()
        firstLineData =np.genfromtxt(lines[0].split(), delimiter=' ')
        N=(int)(firstLineData[0])
        G.add_nodes_from(np.arange(0, N))
        lines = lines[1:]
        dataLines = filter(lambda x: not x.startswith('#'), lines)
        data = np.genfromtxt(dataLines, delimiter=' ')
        for i in range(0,2,1):
            list.append(data[:,i])
        for i in range(np.shape(data)[0]):
            color='black'
            if data[i,2]==-1:
                color='red'
            G.add_edge(*data[i,0:2], color=color)
            orientedEdges.append((tuple)(data[i,0:2]))
            orientedEdges.append((data[i,1], data[i,0]))
            
    colors = nx.get_edge_attributes(G,'color').values()
    
    fig=plt.figure(figsize=(10, 7))  # Dimensione della figura
     # Se non è un reticolo, usiamo un layout automatico
    pos = nx.spring_layout(G)

    nx.draw(G, pos,
        edge_color=colors, 
        with_labels=True,
        width=2.)
    fig.savefig(os.path.join(resultsFolder,'graph.png'))  
    fig=plt.figure(figsize=(10, 9))  # Dimensione della figura
    G = nx.grid_2d_graph(8, 8)

    # Crea le posizioni dei nodi come una griglia regolare
    pos = {(i, j): (j, 7-i) for i, j in G.nodes()}
    mapping = {node: i for i, node in enumerate(G.nodes())}
    #mapping2 = [i for i, node in enumerate(G.nodes())]
    #label
    #G = nx.relabel_nodes(G, mapping)
    print(mapping)
    # Plotta il grafo con posizioni specifiche
    nx.draw(G, pos, labels=mapping, with_labels=True, width=2)
    fig.savefig(os.path.join(resultsFolder,'64SqLatt.png'))  
 
    
    NBT=[]
    
    for edge1 in orientedEdges:
        row =[]
        for edge2 in orientedEdges:
            if edge1[1]==edge2[0] and edge1[0]!=edge2[1]: 
                row.append(1)
            else:
                row.append(0)
        NBT.append(row)
    #for i,r in enumerate(NBT):
        #print("r ",i,np.sum(r))
    NBT = np.asarray(NBT, dtype=np.float64)  # Use float64 for higher precision
    sparse_matrix = csr_matrix(NBT, dtype=np.float64)
    values, vectors = eigs(sparse_matrix, k=1, which='LM', tol=1e-10) 
    

        
    # The largest eigenvalue
    largest_eigenvalue = values[0].real
    #print("val", values)
    #print("vec", vectors)
    #print("lar", largest_eigenvalue)
    #print("bet", np.arctanh(1/largest_eigenvalue))
    list=np.asarray(list).flatten()
    degrees = []
    for i in range(0, N):
        degrees.append(np.sum(list==i))
    degrees = np.asarray(degrees)
    firstMoment = np.mean(degrees)
    secondMoment = np.mean(pow(degrees,2))
    quantity = firstMoment/(secondMoment-firstMoment)
    simData = {}
    simData['beta_c']= {}
    simData['beta_c']['localApproach'] = np.arctanh(quantity)
    simData['beta_c']['globalApproach'] = np.arctanh(1/largest_eigenvalue)

    writeJsonResult(simData, jsonResultsPath)