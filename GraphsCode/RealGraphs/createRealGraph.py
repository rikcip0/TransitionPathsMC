import os

from matplotlib import pyplot as plt
import numpy as np
import networkx as nx

m=25
d= 1.6 #numero giorni
maxTime=86400*d #3 giorni
nToConsider= m*3*d #media di m minuti al giorno

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


#ANALYSIS OF LOG: START
file_path = get_file_with_prefix('Data', 's.txt')
if file_path is None:
    print('log file not found')

with open(file_path, 'r') as file:
    #print('analizzando ', nome_file)
    lines = file.readlines()
    for i in range(len(lines)):
        lines[i]=lines[i].replace('\n', '')
        lines[i] = ' '.join(lines[i].split())
    dataLines = filter(lambda x: not x.startswith('#'), lines)
    data = np.genfromtxt(dataLines, delimiter=' ')
    timeOfMeet = data[:,0]
    first = data[:,1]
    second = data[:,2]
    
mask=timeOfMeet<maxTime
timeOfMeet = timeOfMeet[mask]
first = first[mask]
second =second[mask]

# Concatenazione degli array
concatenated = np.concatenate([first, second])

# Ordinare i valori
unique_values = np.unique(concatenated)

# Creare un dizionario che mappa ogni valore al suo indice
value_to_index = {value: index for index, value in enumerate(unique_values)}

meet={}

for i in range(len(timeOfMeet)):
    f=value_to_index[first[i]]
    s=value_to_index[second[i]]
    if f>s:
        t=f
        f=s
        s=t
    if (f,s) in meet.keys():
        meet[(f,s)]=meet[(f,s)]+1
    else:
        meet[(f,s)]=1
meet2={}
h=[]
G = nx.Graph()

for i, a in enumerate(meet.keys()):
    if meet[a]>=nToConsider:
        v=meet[a]/nToConsider
        meet2[a]=v
        color='black'
        if v<0.:
            color='red'
        G.add_edge(a[0],a[1], color=color, width=v)
# Trova le componenti connesse
largest_cc = max(nx.connected_components(G), key=len)
# Crea un nuovo grafo con solo la componente più grande
G_largest = G.subgraph(largest_cc).copy()
pos = nx.spring_layout(G_largest)
colors = nx.get_edge_attributes(G_largest,'color').values()
widths = np.array(list(nx.get_edge_attributes(G_largest,'width').values()))

node_to_index = {node: idx for idx, node in enumerate(sorted(largest_cc))}
N=len(largest_cc)
print("N=",N)
M=np.nanmean(widths)
minVal=np.nanmin(widths)
maxVal=np.nanmax(widths)


widths = np.array(list(nx.get_edge_attributes(G_largest,'width').values()))
print(len(widths), "link")

plt.figure("before")
plt.hist(widths)
plt.yscale('log')
plt.savefig('before.png')

def scale(v):
    return np.log(v/minVal)/np.log(maxVal/minVal)*(1.5)+0.5
    return 1

newG = nx.Graph()
for u, v, attrs in G_largest.edges(data=True):
    width = scale(attrs.get("width", 1.0)) 
    color='black'
    if width<0.:
        color='red'
    newG.add_edge(node_to_index[u],node_to_index[v], color=color, width=width)
filename = "output_file.txt"
# Apri il file in modalità scrittura
with open(filename, "w") as file:
    # Scrivi l'intestazione (opzionale)
    file.write(f"{N} {len(widths)}\n")
    
    # Ciclo per generare i dati
    for u, v, attrs in newG.edges(data=True):
        width = attrs.get("width", 1.0)  # Modifica il range secondo necessità
        # Scrivi i dati nel file separati da tabulazioni o spazi
        file.write(f"{u} {v} {width:.4f}\n")
        
fig=plt.figure("graph_cc", figsize=(10, 7))  # Dimensione della figura
pos = nx.spring_layout(newG)
colors = nx.get_edge_attributes(newG,'color').values()
widths = np.array(list(nx.get_edge_attributes(newG,'width').values()))
nx.draw(newG, pos,
        edge_color=colors, 
        with_labels=True,
        width=widths)
        
widths = np.array(list(nx.get_edge_attributes(newG,'width').values()))
print(len(widths), "link")

plt.figure("after")
plt.hist(widths)
plt.yscale('log')
figs = plt.get_figlabels()  # Ottieni i nomi di tutte le figure create
for fig_name in figs:
        fig = plt.figure(fig_name)
        filename = f'{fig_name}.png'
        print(filename)
        fig.savefig(filename, bbox_inches='tight')
plt.close('all')
    
