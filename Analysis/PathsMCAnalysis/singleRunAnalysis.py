import random
import json
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import os
import sys
from scipy.optimize import curve_fit
import scipy.stats as stats
from scipy.stats import linregress
matplotlib.use('Agg') 
sys.path.append('../')
from MyBasePlots.multipleCurvesAndHist import multipleCurvesAndHist
from MyBasePlots.hist import myHist
from MyBasePlots.autocorrelation import autocorrelationWithExpDecayAndMu
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
import networkx as nx

doAdditional2dHist=False
simulationCode_version = None
currentAnalysisVersion = 'singleRunAnalysisV0003'
fieldTypesDict = {'1': "Bernoulli", '2': "Gaussian"}
nMaxTrajsToPlot = 5
nMeasuresToDoBootstrap=5

def meanAndSigmaForParametricPlot(toBecomeX, toBecomeY):
    x_unique_values = np.unique(toBecomeX)
    y_mean_values = np.asarray([np.mean(toBecomeY[toBecomeX == x_value]) for x_value in x_unique_values])
    y_var_values = np.asarray([np.var(toBecomeY[toBecomeX == x_value])**0.5 for x_value in x_unique_values])
    y_median_values = np.asarray([np.median(toBecomeY[toBecomeX == x_value]) for x_value in x_unique_values])
    return x_unique_values, y_mean_values, y_var_values, y_median_values


def aDiscreteDerivative(xArray, yArray):
    if len(xArray) < 4 or len(yArray) < 4:
        raise ValueError("Not enough points to calculate the derivative.")
    if len(xArray)!= len(yArray):
        raise ValueError("Something wrong: x and y datasets for calculating the derivative should have the same length.")
    derivative = (((yArray[1:]-yArray[:-1])/(xArray[1:]-xArray[:-1]))[:-2]+2.*((yArray[2:]-yArray[:-2])/(xArray[2:]-xArray[:-2]))[:-1]+((yArray[3:]-yArray[:-3])/(xArray[3:]-xArray[:-3])))/4.
    return xArray[:-3], derivative


def progressiveLinearFit(x, y, yerr, threshold_chi_square=0.5, onlyEnd=False):

    par_values = []
    minimumShifting = np.maximum(len(x)//150, 5)
    minimumLength = 3*minimumShifting

    def linear(t, a, b):
        return t*a+b

    iStartIndex = np.maximum(np.argmax(y>0.001), minimumShifting)

    if iStartIndex + minimumShifting >= len(x)-1:
        return None
    largestIndexOfTimeLargerThanTminus2 = np.where(x<x[-1]-0.3)[0][-1]

    for i in range(iStartIndex, len(x)-2*minimumShifting, minimumShifting):
        jMin = i+minimumShifting
        if onlyEnd:
            jMin = np.maximum(largestIndexOfTimeLargerThanTminus2, i+minimumLength)
        for j in range(jMin, len(x)-1, minimumShifting):
            popt, pcov = curve_fit(linear, x[i:j], y[i:j], sigma=yerr[i:j], method='lm', p0=[1/x[-1],0.6])
            slope = popt[0]
            intercept = popt[1]
            chi_r_value = np.nansum(((y[i:j]-(linear(x[i:j],*popt)))/yerr[i:j])**2.)/(j-i)
            if chi_r_value < threshold_chi_square and intercept+slope*x[-1]>0.02:
                par_values.append((chi_r_value, i, j, slope, intercept))

    if len(par_values)==0:
        return None
    best_segment = min(par_values, key=lambda x: x[0]/(x[2]-x[1])**5.)
    best_Chi = best_segment[0]
    terminalParameters= ["m","b"]
    tauIndex= best_segment[1]
    if not(onlyEnd) or (best_segment[4]+best_segment[3]*x[-1])>0.88:
        return terminalParameters, [best_segment[3], best_segment[4]], np.sqrt(pcov[0,0]), [best_segment[1], best_segment[2]], best_Chi, linear
    else:
        return None

nameOfFoldersContainingGraphs = ["fPosJ"
                               ]
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
                if any(folder in dir_name for folder in nameOfFoldersContainingGraphs):
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

def provaEmailExp(x, y, yerr, threshold_chi_square=10.):

    par_values = []
    minimumShifting = np.maximum(len(x)//100, 5)
    
    iStartIndex=np.maximum(np.argmax(y>0.001),minimumShifting)

    if iStartIndex+minimumShifting>=len(x)-1:
        return None
    
    def provExp(t, b, c, d, iStart):
            tStart = x[iStart]
            a= y[iStart]
            return a+(b*t+c*(np.exp(np.clip(d * t, None, np.log(np.finfo(float).max)))-1-d*t-(d*t)**2./2.))-(b*tStart+c*(np.exp(np.clip(d * tStart, None, np.log(np.finfo(float).max)))-1-d*tStart-(d*tStart**2./2.)))
    
    j=len(x)-1
    bounds = ([0, 0, 0], [np.inf, np.inf, (np.log(2)+80.)/x[-1]])
    for i in range(iStartIndex, len(x)-2*minimumShifting, minimumShifting):
        def provExpWrapper(t, b, c,d):
            return provExp(t, b, c, d, iStart=i)

        popt, pcov = curve_fit(provExpWrapper, x[i:j], y[i:j], bounds=bounds, method="dogbox", maxfev=5000000, p0=[0.015/x[-1], 10**(-15.), np.log(1.5*10**(15.))/x[-1]])
        b = popt[0]
        c = popt[1]
        d = popt[2]
        chi_value = np.nansum(((y[i:j]-(provExpWrapper(x[i:j],*popt))))**2./y[i:j])/(j-i)
        if chi_value < 3*0.001:# and a*x[-1]>0.0001:
            par_values.append((chi_value, i, j, b, c, d))

    if len(par_values)==0:
        return None
    best_segment = min(par_values, key=lambda x: x[0]/(x[2]-x[1])**6.5)
    best_Chi = best_segment[0]
    print(best_Chi, threshold_chi_square)
    if best_Chi<threshold_chi_square: #and best_segment[3]*x[-1]+best_segment[4]>0.15
        return [ "b",r"$\epsilon$ (1-$\delta$)/N", "Nk"],[best_segment[3], best_segment[4], best_segment[5]], [best_segment[1], best_segment[2]], best_Chi, lambda t, b, c, d: provExp(t, b, c, d, iStart= best_segment[1])
    else:
        return None

def provaEmail23(x, y, yerr, threshold_chi_square=10.):

    par_values = []
    minimumShifting = np.maximum(len(x)//100, 5)
    
    iStartIndex=np.maximum(np.argmax(y>0.001),minimumShifting)

    if iStartIndex+minimumShifting>=len(x)-1:
        return None
    
    def prov(t,b, c,d, a, iStart):
            tStart = x[iStart]
            return a+(t)*b+(t)**2*c/2.+d*(t)**3/3.-(tStart*b+tStart**2*c/2.+d*tStart**3/3.)
    
    j=len(x)-1
    finalTime=x[-1]
    for i in range(iStartIndex, len(x)-2*minimumShifting, minimumShifting):


        
        def provWrapper(t, b, c,d, a):
            return prov(t, b, c, d, a, iStart=i)

        popt, pcov = curve_fit(provWrapper, x[i:j], y[i:j], method='lm', maxfev=50000)
        b = popt[0]
        c = popt[1]
        d = popt[2]
        a = popt[3]
        chi_value = np.nansum(((y[i:j]-(provWrapper(x[i:j],*popt))))**2./y[i:j])/(j-i)
        if chi_value < 3*0.001:# and a*x[-1]>0.0001:
            par_values.append((chi_value, i, j, a, b,c,d))

    if len(par_values)==0:
        return None
    best_segment = min(par_values, key=lambda x: x[0]/(x[2]-x[1])**7.4)
    best_Chi = best_segment[0]
    print(best_Chi, threshold_chi_square)
    if best_Chi<threshold_chi_square: #and best_segment[3]*x[-1]+best_segment[4]>0.15
        return ["b","c", "d","a"],[best_segment[4], best_segment[5], best_segment[6], best_segment[3]], [best_segment[1], best_segment[2]], best_Chi,lambda t, b, c, d, a: prov(t,b,c,d,a, iStart= best_segment[1])
    else:
        return None
    
def provaEmail234(x, y, yerr, threshold_chi_square=10.):

    par_values = []
    minimumShifting = np.maximum(len(x)//100, 5)
    
    iStartIndex=np.maximum(np.argmax(y>0.001),minimumShifting)

    if iStartIndex+minimumShifting>=len(x)-1:
        return None
    
    def prov(t,b, c,d, a, iStart):
            tStart = x[iStart]
            return a+(t)*b+(t)**2*c/2.+d*(t)**3/6.+((d**2)/c)*(t)**4/24.-(tStart*b+tStart**2*c/2.+d*tStart**3/3.+((d**2)/c)*(tStart)**4/24.)
    
    j=len(x)-1
    finalTime=x[-1]
    for i in range(iStartIndex, len(x)-2*minimumShifting, minimumShifting):
        
        def provWrapper(t, b, c,d, a):
            return prov(t, b, c, d, a, iStart=i)

        popt, pcov = curve_fit(provWrapper, x[i:j], y[i:j], sigma=yerr[i:j] , method='lm', maxfev=50000)
        b = popt[0]
        c = popt[1]
        d = popt[2]
        a = popt[3]
        chi_value = np.nansum(((y[i:j]-(provWrapper(x[i:j],*popt))))**2./y[i:j])/(j-i)
        if chi_value < 3*0.001:# and a*x[-1]>0.0001:
            par_values.append((chi_value, i, j, a, b,c,d))

    if len(par_values)==0:
        return None
    best_segment = min(par_values, key=lambda x: x[0]/(x[2]-x[1])**7.5)
    best_Chi = best_segment[0]
    print(best_Chi, threshold_chi_square)
    if best_Chi<threshold_chi_square: #and best_segment[3]*x[-1]+best_segment[4]>0.15
        return ["b","c", "d","a"],[best_segment[4], best_segment[5], best_segment[6], best_segment[3]], [best_segment[1], best_segment[2]], best_Chi,lambda t, b, c, d, a: prov(t,b,c,d,a, iStart= best_segment[1])
    else:
        return None
    
def provaEmail2(x, y, yerr, threshold_chi_square=10.):

    par_values = []
    minimumShifting = np.maximum(len(x)//100, 5)
    
    iStartIndex=np.maximum(np.argmax(y>0.001),minimumShifting)

    if iStartIndex+minimumShifting>=len(x)-1:
        return None
    
    def prov(t,b, d, a, iStart):
            tStart = x[iStart]
            return a+(t)*b+d*(t)**2/2.-(tStart*b+d*tStart**2/2.)
    
    j=len(x)-1
    finalTime=x[-1]
    for i in range(iStartIndex, len(x)-2*minimumShifting, minimumShifting):


        
        def provWrapper(t, b, d):
            a = y[i]
            iStart = i
            return prov(t, b, d, a=a, iStart=iStart)

        popt, pcov = curve_fit(provWrapper, x[i:j], y[i:j], method='lm', maxfev=50000)
        a = y[i]
        b = popt[0]
        d = popt[1]
        chi_value = np.nansum(((y[i:j]-(provWrapper(x[i:j],*popt))))**2./y[i:j])/(j-i)
        if chi_value < 3*0.001:# and a*x[-1]>0.0001:
            par_values.append((chi_value, i, j, a, b, d))

    if len(par_values)==0:
        return None
    best_segment = min(par_values, key=lambda x: x[0]/(x[2]-x[1])**7.5)
    best_Chi = best_segment[0]

    if best_Chi<threshold_chi_square: #and best_segment[3]*x[-1]+best_segment[4]>0.15
        return ["b","c","a"],[best_segment[4], best_segment[5], best_segment[3]], [best_segment[1], best_segment[2]], best_Chi,lambda t, b, d, a: prov(t,b,d,a, iStart= best_segment[1])
    else:
        return None
    
def provaEmail3(x, y, yerr, threshold_chi_square=10.):

    par_values = []
    minimumShifting = np.maximum(len(x)//100, 5)
    
    iStartIndex=np.maximum(np.argmax(y>0.001),minimumShifting)

    if iStartIndex+minimumShifting>=len(x)-1:
        return None
    
    def prov(t,b, d, a, iStart):
            tStart = x[iStart]
            return a+(t)*b+d*(t)**3/3.-(tStart*b+d*tStart**3/3.)
    
    j=len(x)-1
    finalTime=x[-1]
    for i in range(iStartIndex, len(x)-2*minimumShifting, minimumShifting):

        def provWrapper(t, b, d):
            a = y[i]
            iStart = i
            return prov(t, b, d, a=a, iStart=iStart)

        popt, pcov = curve_fit(provWrapper, x[i:j], y[i:j], sigma=yerr[i:j] , method='lm', maxfev=50000)
        a = y[i]
        b = popt[0]
        d = popt[1]
        chi_value = np.nansum(((y[i:j]-(provWrapper(x[i:j],*popt))))**2./y[i:j])/(j-i)
        if chi_value < 3*0.001:# and a*x[-1]>0.0001:
            par_values.append((chi_value, i, j, a, b, d))

    if len(par_values)==0:
        return None
    best_segment = min(par_values, key=lambda x: x[0]/(x[2]-x[1])**7.5)
    best_Chi = best_segment[0]

    if best_Chi<threshold_chi_square: #and best_segment[3]*x[-1]+best_segment[4]>0.15
        return ["b","d","a"],[best_segment[4], best_segment[5], best_segment[3]], [best_segment[1], best_segment[2]], best_Chi,lambda t, b, d, a: prov(t,b,d,a, iStart= best_segment[1])
    else:
        return None
    
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

def arraysFromBlockFile(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Inizializza le strutture dati
    blocks = []
    current_block = []

    # Processa le linee del file
    for line in lines:
        line = line.strip()

        if not line:  # Rilevato un block vuoto, salva il block corrente
            if current_block:
                blocks.append(current_block)
                current_block = []
        else:
            # Divide la riga in columns e converte in float
            columns = [float(val) for val in line.split()]
            current_block.append(columns)

    # Aggiungi l'ultimo block se non è vuoto
    if current_block:
        blocks.append(current_block)

    # Trasponi i blocks in array NumPy
    array_per_column = [np.array(block) for block in blocks]

    return np.transpose(np.asarray(array_per_column), (2,0,1))

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
        print('Error in ' +file_path+'. First line should include at least simulationType and version.')
        return None

    simulation_type = firstLine_tokens[0]
    global simulationCode_version
    simulationCode_version = firstLine_tokens[1]
    # Trova le informazioni sulla simulazione nella mappa
    simulationType_info = mappa['simulationTypes'].get(simulation_type, {})
    simulationTypeVersion_info = mappa['simulationTypes'].get(simulation_type, {}).get('versions', {}).get(simulationCode_version, None)
    if (simulationType_info is None) or simulationTypeVersion_info is None:
        print(f"Tipo di simulazione o versione non trovato nella mappa per '{simulation_type} - Versione {simulationCode_version}'.")
        return None

    data = {
        'name': simulationType_info['name'],
        'simulationTypeId': simulationType_info['ID'],
        'shortName': simulationType_info['shortName'],
        'versionId': simulationTypeVersion_info['ID'],
    }

    if len(firstLine_tokens)>2:
        # Assegna i valori dei parametri aggiuntivi dalla prima riga
        for nParameter, paramName in enumerate(simulationTypeVersion_info['additionalParameters']):
            if nParameter + 2 < len(firstLine_tokens):  # Considera solo se ci sono abbastanza token nella riga
                data[paramName] = firstLine_tokens[nParameter + 2]
    
    if ('machine' in data) and ('seed' in data):
        data['ID']= data['machine']+data['seed']
    else:
        data['ID'] = (str)(random.randint(0,1000000))

    if(len((simulationTypeVersion_info['linesMap'])) != simulationTypeVersion_info['nAdditionalLines']):
        print('Error in the map construction')
        return None

    for nLine, lineType in enumerate(simulationTypeVersion_info['linesMap']):
        #print(nLine, lineType) #useful for debug
        data[lineType]={}
        line_info = mappa['lines'].get(lineType, None)
        if line_info is not None:
            parameters = lines[nLine+1].strip().split()
            line_structure = line_info[parameters[0]] 
            data[lineType]=line_structure
            if 'nAdditionalParameters' in line_structure.keys():
                if line_structure['nAdditionalParameters'] !=0:
                    for n, parameter in enumerate(line_structure['additionalParameters']):
                        data[lineType][parameter] = parameters[n+1]
            data[lineType].pop('nAdditionalParameters', None)
            data[lineType].pop('additionalParameters', None)
    #print(data)
    simulationCode_version = (int) (simulationCode_version)
    return data

def singlePathMCAnalysis(run_Path, configurationsInfo, goFast=False, redoIfDone=False):

    simData = {}
    simData['configuration']= configurationsInfo
    #getting run infos: END

    #setting results folders and plots default lines: START
    resultsFolder= os.path.join(run_Path, 'Results')

    jsonResultsPath = os.path.join(resultsFolder,'runData.json')

    analysisVersionOfLastAnalysis = None
    lastMeasureMcOfLastAnalysis = None
    if os.path.exists(jsonResultsPath):
        with open(jsonResultsPath, 'r') as map_file:
            oldResults = json.load(map_file)
        if 'lastMeasureMC' in oldResults.keys():
            lastMeasureMcOfLastAnalysis = oldResults['lastMeasureMC']
        
        if 'analysisVersion' in oldResults.keys():
            analysisVersionOfLastAnalysis = oldResults['analysisVersion']

    
    
    os.makedirs(resultsFolder, exist_ok=True)
    plotsFolder= os.path.join(resultsFolder, 'Plots')
    os.makedirs(plotsFolder, exist_ok=True)
    
    simTypeId = simData['configuration']['simulationTypeId']
    simVersionID = simData['configuration']['versionId']
    seed = simData['configuration']['seed']

    fracPosJ = (float)(simData['configuration']['parameters']['fPosJ'])
    graphID = simData['configuration']['parameters']['graphID']
    N = (int) (simData['configuration']['parameters']['N'])
    p = (int) (simData['configuration']['parameters']['p'])
    C = (int) (simData['configuration']['parameters']['C'])
    T = (float)(simData['configuration']['parameters']['T'])
    beta = (float)(simData['configuration']['parameters']['T'])
    hext =  (float)(simData['configuration']['parameters']['hext'])
    h_in = simData['configuration']['parameters']['h_in']
    h_out = simData['configuration']['parameters']['h_out']
    Qstar = (int)(simData['configuration']['parameters']['Qstar'])
    Qstar/=N
    
    matplotlib.rcParams.update({
        # Font
        'font.size': 20,              # Dimensione testo generale
        'axes.titlesize': 22,         # Titolo asse
        'axes.labelsize': 20,         # Etichette assi
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18,
        'figure.titlesize': 24,

        # Linee
        'lines.linewidth': 2.2,         # Spessore linee principali
        'lines.markersize': 8,        # Dimensione marker

        # Assi
        'axes.linewidth': 2,          # Spessore contorno assi
        'xtick.major.width': 2,       # Spessore ticks
        'ytick.major.width': 2,
        'xtick.major.size': 7,        # Lunghezza ticks
        'ytick.major.size': 7,
        
        # Griglia (opzionale)
        'axes.grid': True,           # Imposta a True se vuoi griglia leggera
        'grid.linewidth': 1,
        'grid.alpha': 0.5,

        # Layout
        'figure.dpi': 300,            # Risoluzione per export PNG
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',      # Rimuove margini inutili
        'pdf.fonttype': 42,           # Testo selezionabile nei PDF
        'ps.fonttype': 42,
        
        # Font family (opzionale)
        'font.family': 'sans-serif',
    })
    
    matplotlib.rcParams.update({
    # …il tuo blocco esistente…
    'font.family':       'sans-serif',            # rimane il family per testo, tick, legende…
    'font.sans-serif':   ['Arial','DejaVu Sans'], # scegli il sans-serif che ti piace per il testo
    'mathtext.fontset':  'cm',                    # o 'stix' se preferisci STIX
    })

    totalMC = (int)(simData['configuration']['mcParameters']['MC'])
    mcEq = (int)(simData['configuration']['mcParameters']['MCeq'])
    mcPrint = (int)(simData['configuration']['mcParameters']['MCprint'])
    mcMeas = (int)(simData['configuration']['mcParameters']['MCmeas'])

    otherInfoKeys = ['ID', 'description', 'shortDescription','fieldType','fieldRealization']
    graphInfoKeys = [ 'graphID', 'fPosJ', 'p', 'C', 'd']
    fieldInfoKeys = ['fieldMean', 'fieldSigma']
    parametersInfo_Line = ' '.join([str(parameter) + '=' + str(int(value) if value.isdigit() else float(value) if value.replace('-','',1).replace('.', '', 1).isdigit() else value) for parameter, value in simData['configuration']['parameters'].items() if parameter not in graphInfoKeys+otherInfoKeys+fieldInfoKeys])
    graphInfo_Line = ' '.join([str(parameter) + '=' + str(int(value) if value.isdigit() else float(value) if value.replace('.', '', 1).isdigit() else value) for parameter, value in simData['configuration']['parameters'].items() if parameter in graphInfoKeys and(parameter!='C' or ((int)(value))!=0)])
    
    
    M_RedLine=None
    Qin_RedLine=None
    Qout_RedLine=None
    
    parametersSettingID = simData['configuration']['parameters']['ID']
    areConfigurationsFM=False
    refConfSettingID = simData['configuration']['referenceConfigurationsInfo']['ID']
    if refConfSettingID== 50 or refConfSettingID== 56:
        Qif = -1
        graphInfo_Line = 'FM '+graphInfo_Line
        areConfigurationsFM=True
        #M_RedLine = [Qstar, 'm*']
        Qout_RedLine = [Qstar, 'q*']
    else:
        Qif = (int)(simData['configuration']['referenceConfigurationsInfo']['mutualOverlap'])
        Qif/=N
        Qout_RedLine = [Qstar, 'q*']
        

    if parametersSettingID== 210:
        fieldInfoKeysDict = {'fieldType': 'type', 'fieldMean':r'$\mu$', 'fieldSigma': r'$\sigma$'}
        fieldInfo_Line = ' '.join([str(fieldInfoKeysDict[parameter]) + '=' + str(int(value) if value.isdigit() else float(value) if value.replace('.', '', 1).replace('-','',1).isdigit() else value) for parameter, value in simData['configuration']['parameters'].items() if parameter in fieldInfoKeys])
        graphInfo_Line += ", w/ " + fieldTypesDict[simData['configuration']['parameters']['fieldType']]+ " field "+fieldInfo_Line+" (r"+ simData['configuration']['parameters']['fieldRealization'] +")"
        configurationsInfo['ID']+= ''.join([str(int(value) if value.isdigit() else float(value) if value.replace('.', '', 1).isdigit() else value) for parameter, value in simData['configuration']['parameters'].items() if parameter in fieldInfoKeys])
    configurationsInfo['ID']+= str(parametersSettingID)+str(refConfSettingID)+''.join([str(int(value) if value.isdigit() else float(value) if value.replace('.', '', 1).isdigit() else value) for parameter, value in simData['configuration']['parameters'].items() if parameter in graphInfoKeys]) 
    configurationsInfo['ID']+=str(parametersSettingID) + str(simData['configuration']['trajs_Initialization']['ID'])
    
    measuresInfo_Line = r'MC$_{eq}$='+f'{mcEq:.2g}'+' '+ r'MC$_{pr}$='+f'{mcPrint:.2g}'+' '+r'MC$_{meas}$='+f'{mcMeas:.2g}'
    settingInfo_Line = parametersInfo_Line+'\n'+ graphInfo_Line+'\n'+measuresInfo_Line

    if refConfSettingID==54:
        refConInfo_Line = 'refConf:'+(simData['configuration']['referenceConfigurationsInfo']['shortDescription'])+r' $\beta$'+f"{(float)(simData['configuration']['referenceConfigurationsInfo']['betaOfExtraction']):.2g}"+'Q'+simData['configuration']['referenceConfigurationsInfo']['mutualOverlap']
    elif refConfSettingID== 53:
        refConInfo_Line = 'refConf:'+(simData['configuration']['referenceConfigurationsInfo']['shortDescription'])+r' $\beta$'+f"{(float)(simData['configuration']['referenceConfigurationsInfo']['betaOfExtraction']):.2g}"+' Q'+simData['configuration']['referenceConfigurationsInfo']['mutualOverlapBeforeQuenching']+'->'+simData['configuration']['referenceConfigurationsInfo']['mutualOverlap']
    elif refConfSettingID== 51:
        refConInfo_Line = 'refConf:'+(simData['configuration']['referenceConfigurationsInfo']['shortDescription'])+' Q'+simData['configuration']['referenceConfigurationsInfo']['mutualOverlap']
    else:
        refConInfo_Line = 'refConf:'+(simData['configuration']['referenceConfigurationsInfo']['shortDescription'])
    
    Qin_AdditionalHists = []
    Qout_AdditionalHists = []
    M_AdditionalHists = []
    
    Hist2D_Data = []
    
    if refConfSettingID in [51,53,54]:
        absolute_path2 = os.path.abspath(os.path.join(run_Path, simData['configuration']['referenceConfigurationsInfo']['fileOfExtractionName']))
        absolute_path = os.path.abspath(os.path.join("..", simData['configuration']['referenceConfigurationsInfo']['fileOfExtractionName']))
        absolute_path = absolute_path.replace('confs','Qs')
        # Now you can use the absolute_path to access the file
        #print(absolute_path)
        if os.path.exists(absolute_path):
            areConfsFromPT=True

            with open(absolute_path, 'r') as file:
                lines = file.readlines()
                lines=lines[1:]  #first line is header
                firstConfigurationIndex = (int) (simData['configuration']['referenceConfigurationsInfo']["firstConfigurationIndex"])
                secondConfigurationIndex = (int) (simData['configuration']['referenceConfigurationsInfo']["secondConfigurationIndex"])

            # Check if the line exists in the file
            try:
                if len(lines) >= np.min([firstConfigurationIndex, secondConfigurationIndex]):
                    firstConfigurationQ = lines[firstConfigurationIndex].strip()
                    firstConfigurationQ = firstConfigurationQ.split()  
                    secondConfigurationQ = lines[secondConfigurationIndex].strip()
                    secondConfigurationQ = secondConfigurationQ.split() 
                    
                    firstConfigurationQs = [int(entry) for entry in firstConfigurationQ]
                    firstConfigurationQs = np.asarray(firstConfigurationQs)/N
                    secondConfigurationQs = [int(entry) for entry in secondConfigurationQ]
                    secondConfigurationQs = np.asarray(secondConfigurationQs)/N
                    
                    Qin_AdditionalHists = [(firstConfigurationQs, "PT")]
                    Qout_AdditionalHists = [(secondConfigurationQs, "PT")]
                    Hist2D_Data = [[(firstConfigurationQs, secondConfigurationQs), "PT"]]
            except Exception:
                pass
                
                
                
                
        else:
            print(f"The file does not have enough lines.")

    
    if simData['configuration']['simulationTypeId']<100:   #so, no annealing
        trajsInitInfo_Line = 'ext_init:'+(simData['configuration']['trajs_Initialization']['shortDescription'])+'\n'
        trajsInitInfo_Line += 'j_init:'+ simData['configuration']['trajs_jumpsInitialization']['shortDescription']
    else:
        trajsInitInfo_Line = 'traj_init: Annealing' 
        trajsInitInfo_Line += '('+simData['configuration']['trajs_Initialization']['shortDescription']+')\n'
        if simData['configuration']['trajs_Initialization']['ID']==73:
            trajsInitInfo_Line += fr'$\beta_{{start}}$'+simData['configuration']['trajs_Initialization']['startingBeta']
            trajsInitInfo_Line += fr'$\Delta\beta$'+simData['configuration']['trajs_Initialization']['deltaBeta']
            trajsInitInfo_Line += fr'mc/$\beta$'+simData['configuration']['trajs_Initialization']['sweepsPerBeta']
        elif simData['configuration']['trajs_Initialization']['ID']==74 or simData['configuration']['trajs_Initialization']['ID']==740:
            trajsInitInfo_Line += fr"$\beta_{{start}}$" +f"{float(simData['configuration']['trajs_Initialization']['startingBeta']):.2f} "
            trajsInitInfo_Line += fr"$\Delta\beta$" +f"{float(simData['configuration']['trajs_Initialization']['deltaBeta']):.2f} "
            trajsInitInfo_Line += fr"MC$_{{start}}$" + simData['configuration']['trajs_Initialization']['startingMC']+' '
            trajsInitInfo_Line += fr"MC$_{{end}}$"+simData['configuration']['trajs_Initialization']['finalMC']



    initInfo_Line = refConInfo_Line + '\n' + trajsInitInfo_Line

    def addInfoLines(whereToAddLines=None):
        # Get the figure object
        fig = plt.gcf() if whereToAddLines is None else whereToAddLines.figure

        # Get all the axes in the figure
        axes = fig.get_axes()

        # Find the leftmost and rightmost positions of all axes
        leftmost = min(ax.get_position().x0 for ax in axes)  # Left boundary of the main plot
        rightmost = max(ax.get_position().x1 for ax in axes)  # Right boundary of the last subplot (histogram)

        # Find the bottom of the axes (to avoid overlap with x-axis labels)
        lowest_position = min(ax.get_position().y0 for ax in axes)
        
        # Adjust the vertical position to be slightly below the x-axis labels
        vertical_position = lowest_position - 0.145  # Adjust this value to control the distance

        # Add text at the bottom of the combined plot, spanning from leftmost to rightmost
        fig.text(leftmost, vertical_position, settingInfo_Line, fontsize=7, ha='left', va='center', transform=fig.transFigure)
        fig.text(rightmost, vertical_position, initInfo_Line, fontsize=7, ha='right', va='center', transform=fig.transFigure)
    
    def addInfoLinesForVideo(whereToAddLines):
        # Get the figure object
        fig = whereToAddLines

        # Get all the axes in the figure
        axes = fig.get_axes()
        # Find the leftmost and rightmost positions of all axes
        leftmost = min(ax.get_position().x0 for ax in axes)  # Left boundary of the main plot
        rightmost = max(ax.get_position().x1 for ax in axes)  # Right boundary of the last subplot (histogram)

        # Find the bottom of the axes (to avoid overlap with x-axis labels)
        lowest_position = min(ax.get_position().y0 for ax in axes)
        
        # Adjust the vertical position to be slightly below the x-axis labels
        vertical_position = lowest_position - 0.145  # Adjust this value to control the distance

        # Add text at the bottom of the combined plot, spanning from leftmost to rightmost
        fig.text(leftmost, vertical_position, settingInfo_Line, fontsize=7, ha='left', va='center', transform=fig.transFigure)
        fig.text(rightmost, vertical_position, initInfo_Line, fontsize=7, ha='right', va='center', transform=fig.transFigure)
    

    simData['analysisVersion'] = currentAnalysisVersion
    results = {}
    #setting results folders and plots default lines: END


    #ANALYSIS OF LOG: START
    file_path = get_file_with_prefix(run_Path, 'log')
    if file_path is None:
        print('log file not found')
        return None
    
    with open(file_path, 'r') as file:
        #print('analizzando ', nome_file)
        lines = file.readlines()
        if len(lines)<3:
            plt.close('all')
            writeJsonResult(simData, os.path.join(resultsFolder,'runData.json'))
            return None
        for i in range(len(lines)):
            lines[i]=lines[i].replace('\n', '')
            lines[i] = ' '.join(lines[i].split())
        dataLines = filter(lambda x: not x.startswith('#'), lines)
        data = np.genfromtxt(dataLines, delimiter=' ')
        realTime = data[:,0]
        mcSteps = data[:,1]

    lastMeasureMc = (int)(np.max(mcSteps))
    simData['lastMeasureMC'] = lastMeasureMc
    

    if redoIfDone is False and currentAnalysisVersion==analysisVersionOfLastAnalysis and lastMeasureMc==lastMeasureMcOfLastAnalysis:
        print('Nor the analysis or the data changed from last analysis.\n\n')
        return None

    
    results['realTime']={}
    theseFiguresFolder= os.path.join(plotsFolder, 'runRealTime')
    if not os.path.exists(theseFiguresFolder):
        os.makedirs(theseFiguresFolder)
    else:
        delete_files_in_folder(theseFiguresFolder)

    measuresInfo_Line += r' MC$_{lastPrin}$='+f'{lastMeasureMc:.2g}'
    settingInfo_Line = parametersInfo_Line+'\n'+ graphInfo_Line+'\n'+measuresInfo_Line

    realTime=np.asarray(realTime)
    realTime=np.diff(realTime)
    mcSteps = np.asarray(mcSteps, dtype=np.int32)

    plt.figure('realTimeVsMCSweeps')
    plt.xlabel('mc')
    plt.ylabel('Computer time')
    plt.scatter(mcSteps[1:], realTime)   #This 1 is different from that of firstMeasureToConsider (eventually, if it s =1). It is so because first entry is 0 by definition
    addInfoLines()

    firstMeasureToConsider = 1
    if len(realTime[firstMeasureToConsider:])>0:
        title = f'Histogram of computer time needed to perform {mcSteps[1]} steps\n partitioning total MC time. First {firstMeasureToConsider} measure(s) skipped'
        mean, sigma =myHist('realTimeHist', title, realTime[firstMeasureToConsider:], 'computer time')
        addInfoLines()
        results['realTime']['mean']=mean
        results['realTime']['sigma']=sigma
    else:
        results['realTime']['mean']='nan'
        results['realTime']['sigma']='nan'

    figs = plt.get_figlabels()  # Ottieni i nomi di tutte le figure create
    for fig_name in figs:
        fig = plt.figure(fig_name)
        filename = os.path.join(theseFiguresFolder, f'{fig_name}.png')
        fig.savefig(filename, bbox_inches='tight')
    plt.close('all')
    #ANALYSIS OF LOG: END
    
    
    confsFolder=os.path.join(run_Path, 'conf')
    if os.path.exists(confsFolder) and bool(os.listdir(confsFolder)) and False:
            
        graphFile_path= findFoldersWithString('../../Data/Graphs', [f'{graphID}'])
        print(graphFile_path)
        if len(graphFile_path)!=0:
            graphFile_path = graphFile_path[0]
            graphFile_path=os.path.join(graphFile_path,'graph.txt')
            print(graphFile_path)
        else:
            print("AO")
        print(graphFile_path, graphID)
        G = nx.Graph()
        G.add_nodes_from(np.arange(0, N))
        with open(graphFile_path, 'r') as file:
            #print("analizzando ", nome_file)
            lines = file.readlines()
            firstLineData =np.genfromtxt(lines[0].split(), delimiter=' ')
            lines = lines[1:]
            dataLines = filter(lambda x: not x.startswith('#'), lines)
            data = np.genfromtxt(dataLines, delimiter=' ')
            for i in range(np.shape(data)[0]):
                color='blue'
                if data[i,2]<0.:
                    color='red'
                G.add_edge(*data[i,0:2], color=color, width=data[i,2])
                
        pos = nx.spring_layout(G)
        theseFiguresFolder= os.path.join(plotsFolder, "someTrajsVideo")
        if not os.path.exists(theseFiguresFolder):
            os.makedirs(theseFiguresFolder)
        else:
            delete_files_in_folder(theseFiguresFolder)
            
        for tj in [0,200000, 400000]:
            confs0=os.path.join(confsFolder, f'mc{tj}')
            node_colors_over_time = {}
            if not os.path.exists(confs0):
                continue
            with open(confs0, 'r') as file:
                #print("analizzando ", nome_file)
                lines = file.readlines()
                dataLines = filter(lambda x: not x.startswith('#'), lines)
                data = np.genfromtxt(dataLines, delimiter=' ')
                for i in range(np.shape(data)[0]):
                    key=data[i,0]
                    colors=['red' if n > 0 else 'blue' for n in data[i, 1:]]
                    node_colors_over_time[key]=colors
                if T not in node_colors_over_time:
                    node_colors_over_time[T]=colors
            times = sorted(node_colors_over_time.keys())

            # Funzione per trovare i colori in base al tempo
            def get_node_colors(t):
                # Usa i colori del tempo precedente rispetto al valore di t
                for i in range(len(times) - 1, -1, -1):
                    if t >= times[i]:
                        return node_colors_over_time[times[i]]
                return node_colors_over_time[times[0]]

            def make_frame(t):
                fig, (ax_main, ax_info) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [9, 1]}, figsize=(8, 6))

                # Rimuovi spazi extra tra i subplot
                fig.subplots_adjust(hspace=0)

                # Disegna il grafo sul primo subplot
                ax_main.clear()
                node_colors = get_node_colors(t)
                edge_widths = nx.get_edge_attributes(G, "width")
                widths = [edge_widths[edge] if edge in edge_widths else 1.0 for edge in G.edges()]
                nx.draw(
                    G, pos, ax=ax_main, with_labels=True,
                    node_color=node_colors, edge_color="black", width=widths,
                    node_size=60
                )
                ax_main.text(
                    0.02, 1.02, f"traj{tj} Time: {t:.2f}",
                    transform=ax_main.transAxes, fontsize=12, color="black",
                    verticalalignment="bottom", horizontalalignment="left"
                )

                # Rimuovi gli assi dal subplot delle InfoLines
                ax_info.axis('off')

                # Aggiungi le InfoLines sul secondo subplot
                ax_info.text(
                    0.02, 0.5, settingInfo_Line, fontsize=7, ha='left', va='center', color='black'
                )
                ax_info.text(
                    0.98, 0.5, initInfo_Line, fontsize=7, ha='right', va='center', color='black'
                )

                # Converti la figura in immagine
                return mplfig_to_npimage(fig)

            # Creazione del video
            duration = max(times)  # Durata del video (tempo massimo + margine)
            fps = 10  # Frame al secondo (aggiornamento fluido del tempo)

            animation = VideoClip(make_frame, duration=duration)

            videoPath= os.path.join(theseFiguresFolder, f"tj{tj}.mp4")
            animation.write_videofile(videoPath, fps=fps)
            plt.close('all')

    #ANALYSIS OF TI FILES: START
    results['TI'] ={'beta':[], 'hout':[], 'Qstar':[]}

    plotTIUAutocorrelation=False
    TIFile = get_file_with_prefix(run_Path, 'TI_beta')
    if TIFile is not None:
        with open(TIFile, 'r') as file:
            lines = file.readlines()
            if len(lines)<1:
                plt.close('all')
                writeJsonResult(simData, os.path.join(resultsFolder,'runData.json'))
                return None
            for i in range(len(lines)):
                lines[i]=lines[i].replace('\n', '')
                lines[i] = ' '.join(lines[i].split())
            dataLines = filter(lambda x: not x.startswith('#'), lines)
            data = np.genfromtxt(dataLines, delimiter=' ')
            if len(lines)>1:
                mcTimes = data[:,0]
                cumulativeUs = data[:,-1]
                results['TI']['beta'] = cumulativeUs[-1]
                cumulativeUs=np.array(cumulativeUs,dtype=np.float128)
            else:
                results['TI']['beta'] = data[-1]
        if len(lines)>2:
            plotTIUAutocorrelation=True
        if len(lines)>1:
            measuresCounter = np.arange(1, len(mcTimes)+1,dtype=np.float128)
            previousContribution=np.roll(cumulativeUs*measuresCounter,1)
            previousContribution[0]=0
            singleUs = cumulativeUs*measuresCounter-previousContribution
            singleUs =np.array(singleUs,dtype=np.float64)
            cumulativeUs =np.array(cumulativeUs,dtype=np.float64)
            measuresCounter =np.array(measuresCounter,dtype=np.int64)
            nMeasuresToDoBootstrap2=np.min([nMeasuresToDoBootstrap, len(singleUs)])
            chunk_size = len(singleUs) // nMeasuresToDoBootstrap
            means = [np.mean(np.random.choice(singleUs, size=chunk_size, replace=True)) 
                    for _ in range(nMeasuresToDoBootstrap2)]
            results['TI']['betaForBS'] = means

            measuresCounter*=mcPrint
            measuresCounter+=mcEq

            titleSpecification = 'computed over sampled trajectories'
            plt.figure('TI_beta_U')
            plt.title(r'Quantity for thermodynamic integration (U) vs mc sweeps'+'\n'+ titleSpecification)
            plt.xlabel('mc sweep')
            plt.ylabel('U')
            plt.plot(measuresCounter, cumulativeUs) 
            plt.scatter(measuresCounter, cumulativeUs, label= "cumulative average")
            #m2=[measuresCounter[i] for i in range(len(measuresCounter//4))]
            #s2=[np.mean(singleUs[i*4:(i+1)*4]) for i in range(len(measuresCounter//4))]
            #plt.scatter(m2, s2, label= "single measure") 
            plt.scatter(measuresCounter, singleUs, label= "single measure") 
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            addInfoLines()
    else:
        results['TI']['beta'] = 'nan'


    TIFile = get_file_with_prefix(run_Path, 'TI_hout')
    if TIFile is not None:
        with open(TIFile, 'r') as file:
            numbers = [float(num) for num in file.readline().replace(',', '.').split() if any(c.isdigit() or c == '.' for c in num)]
        results['TI']['hout'] = numbers[-1]
    else:
        results['TI']['hout'] = 'nan'

    TIFile = get_file_with_prefix(run_Path, 'TI_Qstar')
    if TIFile is not None:
        with open(TIFile, 'r') as file:
            numbers = [float(num) for num in file.readline().replace(',', '.').split() if any(c.isdigit() or c == '.' for c in num)]
        results['TI']['Qstar'] = numbers[-1 ]
    else:
        results['TI']['Qstar'] = 'nan'
    #ANALYSIS OF TI FILES: END


    #ANALYSIS OF THERMALIZATION DATA: START
    thermCheck_filePath = get_file_with_prefix(run_Path, 'thermCheck')
    if thermCheck_filePath is None:
        print('No thermCheck file found')
        return None

    theseFiguresFolder= os.path.join(plotsFolder, 'thermalization')
    if not os.path.exists(theseFiguresFolder):
        os.makedirs(theseFiguresFolder)
    else:
        delete_files_in_folder(theseFiguresFolder)

    titleSpecification = 'as from thermalization data'


    therm_mcMeasures = []
    therm_avEnergies = []
    therm_barriers = []
    therm_distFromStrfwdPath = []
    therm_meanNJumps = []
    therm_deltaNJumps = []
    therm_minNJumps = []
    therm_maxNJumps = []

    mcForThermCheck= int(simData['configuration']['thermalizationCheckParameters']['mcForThermCheck'])
    firstIndexOfMeasuresAtEq = (int)(np.ceil(mcEq/mcForThermCheck))

    measuresToSkipInOOEPlot = 1      #The first measure may be very different even wrt non-eq ones, as it contains the initialization trajectory
    
    with open(thermCheck_filePath, 'r') as file:
        #print('analizzando ', nome_file)
        lines = file.readlines()
        dataLines = filter(lambda x: not x.startswith('#'), lines)
        data = np.genfromtxt(dataLines, delimiter=' ')
        therm_mcMeasures=data[:,0]
        therm_avEnergies=data[:,1]
        therm_barriers=data[:,2]
        therm_distFromStrfwdPath=data[:,3]
        therm_meanNJumps=data[:,4]
        therm_deltaNJumps =data[:,5]
        therm_minNJumps =data[:,6]
        therm_maxNJumps =data[:,7]

    therm_mcMeasures = np.asarray(therm_mcMeasures, dtype=np.float64)
    therm_avEnergies = np.asarray(therm_avEnergies, dtype=np.float64)
    therm_barriers = np.asarray(therm_barriers, dtype=np.float64)
    therm_distFromStrfwdPath = np.asarray(therm_distFromStrfwdPath, dtype=np.float64)
    therm_meanNJumps = np.asarray(therm_meanNJumps, dtype=np.float64)
    therm_deltaNJumps = np.asarray(therm_deltaNJumps, dtype=np.float64)
    therm_minNJumps = np.asarray(therm_minNJumps, dtype=np.float64)
    therm_maxNJumps = np.asarray(therm_maxNJumps, dtype=np.float64)

    if len(therm_mcMeasures)==1:
        print('Not enough measures')
        return None

    results['thermalization'] = {}
    nMusToConsider = 40
    titleForAutocorrelations = 'autocorrelation over mcSweeps'
    results['thermalization']['maxNJumps'] = {}

    #defining a function to plot quantities evolution over mc iterations and the respective autocorrelation
    def mcEvolutionAndAutocorrelation(mcSweeps, quantity, firstIndexForEquilibrium,
                                      quantityShortName, quantityFullName, quantityLabelName, nMus,
                                      plotBoth=True):
        results['thermalization'][quantityShortName] = {}

        # 3) Usi β come plain‑text (Unicode) e J in math‑mode con \mathrm
        if plotBoth:
            plt.figure(quantityShortName)
            plt.title(quantityFullName+' vs MC\n'+titleSpecification)
            plt.plot(mcSweeps, quantity)
            plt.xlabel('MC sweep')
            plt.xlabel(r"$\beta\,J$")
            
            plt.ylabel(quantityLabelName)
            addInfoLines()

        #plt.axvline(x=mcEq, color='red', linestyle='--')
        #plt.text(mcEq, plt.ylim()[1], 'MCeq', color='red', verticalalignment='bottom', horizontalalignment='right', fontsize=7)


        results['thermalization'][quantityShortName]['mean'] = np.mean(quantity[firstIndexForEquilibrium:])
        if((int)(therm_mcMeasures[-1])==mcPrint):
            return None        
        results['thermalization'][quantityShortName]['stdErr'] = stats.sem(quantity[firstIndexForEquilibrium:])

        if (len(np.unique(quantity[firstIndexForEquilibrium:])) > 1): #i.e., if it s not a constant
            mu, muErr, rChi2, dof  = autocorrelationWithExpDecayAndMu(quantityShortName+'Autocorrelation', quantityFullName+' '+titleForAutocorrelations,
                                        mcSweeps[firstIndexForEquilibrium:], 'mc', quantity[firstIndexForEquilibrium:], quantityLabelName,
                                        nMus)
            addInfoLines()
            results['thermalization'][quantityShortName]['mu'] = mu
            results['thermalization'][quantityShortName]['muErr'] = muErr
            results['thermalization'][quantityShortName]['rChi2'] = rChi2
            results['thermalization'][quantityShortName]['dof'] = dof
            return mu, muErr, rChi2, dof
        else:
            results['thermalization'][quantityShortName]['mu'] = 'nan'
            results['thermalization'][quantityShortName]['muErr'] = 'nan'
            results['thermalization'][quantityShortName]['rChi2'] = 'nan'
            results['thermalization'][quantityShortName]['dof'] = 'nan'

    if((int)(therm_mcMeasures[-1])==mcPrint):
        simData['results'] = results
        return None     
           
    muAvEnergy, _, _, _ = mcEvolutionAndAutocorrelation(therm_mcMeasures[:len(therm_mcMeasures)], therm_avEnergies[:len(therm_mcMeasures)], firstIndexOfMeasuresAtEq,
                                      'avEnergy', 'trajectory average energy', 'energy', nMusToConsider)
    
    mcEvolutionAndAutocorrelation(therm_mcMeasures, therm_barriers, firstIndexOfMeasuresAtEq,
                                      'maxEnergy', 'trajectory max energy', 'energy', nMusToConsider)
    
    mcEvolutionAndAutocorrelation(therm_mcMeasures, therm_distFromStrfwdPath, firstIndexOfMeasuresAtEq,
                                      'qDist', 'Average trajectory distance from the straightforward', 'energy', nMusToConsider)
    
    mcEvolutionAndAutocorrelation(therm_mcMeasures, therm_meanNJumps, firstIndexOfMeasuresAtEq,
                                      'nJumps', 'Mean number of jumps per spin', 'energy', nMusToConsider)
    
    mcEvolutionAndAutocorrelation(therm_mcMeasures, therm_deltaNJumps, firstIndexOfMeasuresAtEq,
                                      'deltaNJumps', r'Trajectory $\Delta$(#jumps) per spin ', 'energy', nMusToConsider)
    
    mcEvolutionAndAutocorrelation(therm_mcMeasures, therm_minNJumps, firstIndexOfMeasuresAtEq,
                                      'minNJumps', r'Trajectory min(#jumps) per spin ', 'energy', nMusToConsider)
    
    mcEvolutionAndAutocorrelation(therm_mcMeasures, therm_maxNJumps, firstIndexOfMeasuresAtEq,
                                      'maxNJumps', r'Trajectory max(#jumps) per spin ', 'energy', nMusToConsider)
    if plotTIUAutocorrelation:
        mcEvolutionAndAutocorrelation(measuresCounter, singleUs, 0,
                                        'TI_beta_U', r'Quantity for thermodynamic integration in $\beta$', 'U', nMusToConsider,
                                        plotBoth=False)
    figs = plt.get_figlabels()  # Ottieni i nomi di tutte le figure create
    for fig_name in figs:
        fig = plt.figure(fig_name)
        filename = os.path.join(theseFiguresFolder, f'{fig_name}.png')
        fig.savefig(filename, bbox_inches='tight')
    plt.close('all')
    #ANALYSIS OF THERMALIZATION DATA: END


    #START OF ANALYSIS OF SAMPLED TRAJS 
    file_path = get_file_with_prefix(run_Path, 'story')
    if file_path is None:
        print('No story file found')
        return None
    
    arrays = arraysFromBlockFile(file_path)

    # Esempio di accesso agli array per la prima column del primo block
    times = arrays[0,:, :]
    q_in = arrays[1,:, :]
    q_out = arrays[2,:, :]
    M = arrays[3,:, :]
    energy = arrays[4,:, :]

    q_in/=N
    q_out/=N
    M/=N
    energy/=N

    nTrajs=times.shape[0]

    if nTrajs == 0:
        plt.close('all')
        simData['results'] = results
        writeJsonResult(simData, os.path.join(resultsFolder,'runData.json'))
        return None
    
    if len(Hist2D_Data) and doAdditional2dHist:
        theseFiguresFolder= os.path.join(plotsFolder, 'additionalRelevantData')
        if not os.path.exists(theseFiguresFolder):
            os.makedirs(theseFiguresFolder)
        else:
            delete_files_in_folder(theseFiguresFolder)
            
        plt.figure('referenceConfigurations_OverlapWithPT')
        plt.title('Overlap between reference configurations and the others at same temperature from PT')
        auto_bins_x = np.histogram_bin_edges(Hist2D_Data[0][0][0], bins='auto')
        auto_bins_y = np.histogram_bin_edges(Hist2D_Data[0][0][1], bins='auto')

        # Imposta i nuovi bin raddoppiando quelli automatici
        bins_x = (int)(len(auto_bins_x) * 1.1)
        bins_y = (int)(len(auto_bins_y) * 1.1)
        plt.hist2d(Hist2D_Data[0][0][0], Hist2D_Data[0][0][1], bins=[bins_x, bins_y], cmap='Blues', density=True)
        plt.xlabel('q_in')
        plt.ylabel('q_out')
        # Add a colorbar for reference
        plt.colorbar()
        figs = plt.get_figlabels()  # Ottieni i nomi di tutte le figure create
        for fig_name in figs:
            fig = plt.figure(fig_name)
            filename = os.path.join(theseFiguresFolder, f'{fig_name}.png')
            fig.savefig(filename, bbox_inches='tight')
        plt.close('all')

    theseFiguresFolder= os.path.join(plotsFolder, 'sampledTrajs')
    if not os.path.exists(theseFiguresFolder):
        os.makedirs(theseFiguresFolder)
    else:
        delete_files_in_folder(theseFiguresFolder)

    titleSpecification = 'for initializing trajectory'

    #Plots of initialization trajectory: START
    theseFiguresSubFolder= os.path.join(theseFiguresFolder, 'initializingTraj')
    if not os.path.exists(theseFiguresSubFolder):
        os.makedirs(theseFiguresSubFolder)
    else:
        delete_files_in_folder(theseFiguresSubFolder)

    histScale = ''
    if areConfigurationsFM:
        histScale='log'
    
    figure, mainPlot = multipleCurvesAndHist('energy', r'Energy vs time'+'\n'+ titleSpecification,
                                times[0], 't',  energy[0], r'$energy$', N, histScale=histScale)
    addInfoLines(figure)

    figure, mainPlot = multipleCurvesAndHist('M', r'm vs time'+'\n'+ titleSpecification,
                                times[0], 't',  M[0], 'm', N, redLineAtYValueAndName=M_RedLine, histScale=histScale)
    addInfoLines(figure)
    
    figure, mainPlot = multipleCurvesAndHist('EnergyVsM', r'Energy vs m'+'\n'+ titleSpecification,
                                    M[0], 'm',  energy[0], 'energy', N, redLineAtXValueAndName=M_RedLine, histScale=histScale)
    addInfoLines(figure)
    
    if not areConfigurationsFM:
        figure, mainPlot = multipleCurvesAndHist('Qin', r'$q_{in}$ vs time'+'\n'+ titleSpecification,
                                    times[0], 't',  q_in[0], r'$q_{in}$', N, 
                                    additionalYHistogramsArraysAndLabels=Qin_AdditionalHists, redLineAtYValueAndName=Qin_RedLine, histScale=histScale)
        addInfoLines(figure)

        figure, mainPlot = multipleCurvesAndHist('Qout', r'$q_{out}$ vs time'+'\n'+ titleSpecification,
                                    times[0], 't',  q_out[0],r'$q_{out}$', N, 
                                    additionalYHistogramsArraysAndLabels=Qout_AdditionalHists, redLineAtYValueAndName=Qout_RedLine, histScale=histScale)
        addInfoLines(figure)
        
        
        
        plt.figure('QoutVsQin')
        plt.plot(q_in[0], q_out[0])
        plt.title(r'$q_{out}$ vs $q_{in}$' +'\n'+ titleSpecification)
        plt.xlabel(r'$q_{in}$')
        plt.ylabel(r'$q_{out}$')
        plt.axhline(Qstar, color='red', linestyle='dashed', linewidth=1, label=r'Q*')
        plt.scatter([1.],[Qif], marker= '*', s=45, color='black')
        if(np.max(q_out)>=0.85):
            plt.scatter([Qif], [1.], marker= '*', s=45, color='black')
        addInfoLines()

        figure, mainPlot = multipleCurvesAndHist('MVsQin', r'M vs $q_{in}$'+'\n'+ titleSpecification,
                                    q_in[0], r'$q_{in}$',  M[0],'m', N, 
                                    additionalYHistogramsArraysAndLabels=M_AdditionalHists, redLineAtXValueAndName=Qin_RedLine, redLineAtYValueAndName=M_RedLine, histScale=histScale)
        addInfoLines(figure)
        
        figure, mainPlot = multipleCurvesAndHist('MVsQout', r'M vs $q_{out}$'+'\n'+ titleSpecification,
                                    q_out[0], r'$q_{out}$',  M[0],'m', N, 
                                    additionalYHistogramsArraysAndLabels=M_AdditionalHists, redLineAtXValueAndName=Qout_RedLine, redLineAtYValueAndName=M_RedLine, histScale=histScale)
        addInfoLines(figure)
        
        figure, mainPlot = multipleCurvesAndHist('EnergyVsQin', 'energy vs ' +r'$q_{in}$'+'\n'+ titleSpecification,
                                    q_in[0], r'$q_{in}$',  energy[0], 'energy', N, redLineAtXValueAndName=Qin_RedLine)
        addInfoLines(figure)
        
        figure, mainPlot = multipleCurvesAndHist('EnergyVsQout', 'energy vs ' +r'$q_{out}$'+'\n'+ titleSpecification,
                                    q_out[0], r'$q_{out}$',  energy[0], 'energy', N, redLineAtXValueAndName=Qout_RedLine)
        addInfoLines(figure)

    figs = plt.get_figlabels()  # Ottieni i nomi di tutte le figure create
    for fig_name in figs:
        fig = plt.figure(fig_name)
        filename = os.path.join(theseFiguresSubFolder, f'{fig_name}.png')
        fig.savefig(filename, bbox_inches='tight')
    plt.close('all')
    #Plots of initialization trajectory: END

    #Plots considering over all trajs: START
    times = times[1:,:]
    q_in = q_in[1:,:]
    q_out = q_out[1:,:]
    M = M[1:,:]
    energy = energy[1:,:]
    theseFiguresSubFolder= os.path.join(theseFiguresFolder, 'allTrajs')
    if not os.path.exists(theseFiguresSubFolder):
        os.makedirs(theseFiguresSubFolder)
    else:
        delete_files_in_folder(theseFiguresSubFolder)

    titleSpecification = 'over all sampled trajectories'

    plt.figure('energyMean')
    plt.title(f'Mean of energy vs time\n'+ titleSpecification)
    plt.xlabel(r't')
    plt.ylabel(r'$mean energy$')
    plt.plot(times.mean(0), energy.mean(0))
    addInfoLines()

    plt.figure('energyStdDev')
    plt.title(r'$\Delta$'+f' energy vs time\n'+ titleSpecification)
    plt.xlabel(r't')
    plt.ylabel(r'$\Delta$E')
    plt.plot(times.mean(0), np.sqrt(energy.var(0)))
    addInfoLines()

    plt.figure('energyVsM')
    #plt.title(f'Mean of energy vs M\n'+ titleSpecification)
    plt.xlabel(r'm')
    plt.ylabel(r'$energy$')
    a = meanAndSigmaForParametricPlot(M, energy)
    plt.errorbar(a[0],a[1],a[2], label='mean')
    plt.scatter(a[0],a[3], color='darkorange', s=25, label='median')
    if M_RedLine is not None:
        plt.axvline(M_RedLine[0], color='red', linestyle='dashed', linewidth=1, label=M_RedLine[1])
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    #addInfoLines()
    
    if not areConfigurationsFM:
        plt.figure('energyVsQin')
        plt.title(f'Mean of energy vs '+r'$q_{in}$'+'\n'+ titleSpecification)
        plt.xlabel(r'$q_{in}$')
        plt.ylabel(r'$energy$')
        a = meanAndSigmaForParametricPlot(q_in, energy)
        plt.errorbar(a[0],a[1],a[2], label='mean')
        plt.scatter(a[0],a[3], color='darkorange', s=25, label='median')
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        addInfoLines()

        plt.figure('energyVsQout')
        plt.title(f'Mean of energy vs '+r'$q_{out}$'+'\n'+ titleSpecification)
        plt.xlabel(r'$q_{out}$')
        plt.ylabel(r'$energy$')
        a = meanAndSigmaForParametricPlot(q_out, energy)
        plt.errorbar(a[0],a[1],a[2], label='mean')
        plt.scatter(a[0],a[3], color='darkorange', s=25, label='median')
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.axvline(Qstar, color='red', linestyle='dashed', linewidth=1, label=r'Q*')
        addInfoLines()
        
        plt.figure('trajectories_OverlapWithRefConfs')
        plt.title('Overlap between reference configurations and the others at same temperature from PT')
        auto_bins_x = np.histogram_bin_edges(q_in, bins='auto')
        auto_bins_y = np.histogram_bin_edges( q_out.flatten(), bins='auto')

        # Imposta i nuovi bin raddoppiando quelli automatici
        bins_x = (int)(len(auto_bins_x) * 1.1)
        bins_y = (int)(len(auto_bins_y) * 1.1)
        plt.hist2d(q_in.flatten(), q_out.flatten(), bins=[bins_x, bins_y], cmap='Blues', density=True)
        plt.xlabel('q_in')
        plt.ylabel('q_out')
        # Add a colorbar for reference
        plt.colorbar()
        addInfoLines()

    barrier = energy - np.mean(energy[:, [0]], axis=1, keepdims=True)
    barrierIndices = np.argmax(barrier,1)
    barrier = np.max(barrier,1)
    MOfBarrier = np.asarray([M[i,index] for i,index in enumerate(barrierIndices)])
    QinOfBarrier = np.asarray([q_in[i,index] for i,index in enumerate(barrierIndices)])
    QoutOfBarrier = np.asarray([q_out[i,index] for i,index in enumerate(barrierIndices)])

    myHist('barriersHistogram', 'Histogram of energy barriers\n'+ titleSpecification, barrier, 'barrier')
    addInfoLines()

    myHist('barriersM', 'Histogram of M of energy barriers\n'+ titleSpecification, MOfBarrier, 'barrier')
    addInfoLines()

    if not areConfigurationsFM:
        myHist('barriersQin', 'Histogram of '+r'$q_{in}$' + 'of energy barriers\n'+ titleSpecification, QinOfBarrier, 'barrier')
        addInfoLines()

        myHist('barriersQout', 'Histogram of '+r'$q_{out}$' + 'of energy barriers\n'+ titleSpecification, QoutOfBarrier, 'barrier')
        addInfoLines()

    
    plt.figure('barriersEvolution')
    plt.plot(barrier)
    plt.title(f'Energy barriers over sampled trajectories\n'+ titleSpecification)
    plt.xlabel('#(sampled trajectory)')
    plt.ylabel('barrier')
    addInfoLines()

    results['meanBarrier']= barrier.mean()
    results['deltaBarrier']= barrier.var()**0.5
    results['stdDevBarrier']= (barrier.var()/barrier.size)**0.5
    
    results['mOfBarrier']= MOfBarrier.mean()
    results['deltaMOfBarrier']= MOfBarrier.var()**0.5
    results['q_inOfBarrier']= QinOfBarrier.mean()
    results['deltaQ_inOfBarrier']= QinOfBarrier.var()**0.5
    results['q_outOfBarrier']= QoutOfBarrier.mean()
    results['deltaQ_outOfBarrier']= QoutOfBarrier.var()**0.5

    figs = plt.get_figlabels()  # Ottieni i nomi di tutte le figure create
    for fig_name in figs:
        fig = plt.figure(fig_name)
        filename = os.path.join(theseFiguresSubFolder, f'{fig_name}.png')
        fig.savefig(filename, bbox_inches='tight')
    plt.close('all')
    #Plots considering over all trajs: END

    

    #Plots considering only some trajs: START
    theseFiguresSubFolder= os.path.join(theseFiguresFolder, 'someTrajs')
    if not os.path.exists(theseFiguresSubFolder):
        os.makedirs(theseFiguresSubFolder)
    else:
        delete_files_in_folder(theseFiguresSubFolder)

    nRandomTrajs= np.min([nMaxTrajsToPlot, nTrajs-1])-1
    someTrajs= np.array([0]) #non è quella dell'inizializzazione, ma quella subito dopo
    if nRandomTrajs>0:
        someTrajs = np.append(someTrajs, np.asarray([nTrajs-4]))
        someTrajs = np.sort(np.append(someTrajs, np.random.choice(np.arange(1, nTrajs-4), nRandomTrajs-1, replace=False)))
    
    someTrajs_MC=[tj*mcPrint for tj in someTrajs]
    titleSpecification = 'considering some sampled trajectories'
    
    figure, mainPlot = multipleCurvesAndHist('energy','',# r'Energy vs time'+'\n'+ titleSpecification,
                                times, 't', energy, r'energy', N, nameForCurve= 'traj', curvesIndeces=someTrajs,
                                isYToHist=True, histScale=histScale)
    #addInfoLines(figure)
    
    
    figure, mainPlot = multipleCurvesAndHist('M', #'M vs time'+'\n'+ titleSpecification,
                                             '',
                                times, 't', M, 'm', N, nameForCurve= 'traj', curvesIndeces=someTrajs,
                                isYToHist=True, histScale=histScale,  redLineAtYValueAndName=M_RedLine)
    #addInfoLines(figure)
    
    figure, mainPlot = multipleCurvesAndHist('EnergyVsM', f'energy vs m'+'\n'+ titleSpecification,
                                M, r'm', energy, 'energy', N, nameForCurve= 'traj', curvesIndeces=someTrajs,
                                isYToHist=True, histScale=histScale, redLineAtXValueAndName=M_RedLine)
    addInfoLines(figure)
    
    if not areConfigurationsFM:
        figure, mainPlot = multipleCurvesAndHist('Qin', r'$q_{in}$ vs time'+'\n'+ titleSpecification,
                                    times, 't', q_in, r'$q_{out}$', N, nameForCurve= 'traj', curvesIndeces=someTrajs,
                                    isYToHist=True, additionalYHistogramsArraysAndLabels=Qin_AdditionalHists, redLineAtYValueAndName=Qin_RedLine, histScale=histScale)
        addInfoLines(figure)
        
        
        figure, mainPlot = multipleCurvesAndHist('Qout', r'$q_{out}$ vs time'+'\n'+ titleSpecification,
                                    times, 't', q_out, r'$q_{out}$', N, nameForCurve= 'traj', curvesIndeces=someTrajs,
                                    isYToHist=True, additionalYHistogramsArraysAndLabels=Qout_AdditionalHists, redLineAtYValueAndName=Qout_RedLine, histScale=histScale)
        addInfoLines(figure)

        figure, mainPlot = multipleCurvesAndHist('QoutVsQinProva', r'$q_{out}$ vs $q_{in}$'+'\n'+ titleSpecification,
                                    q_in, r'$q_{in}$', q_out, r'$q_{out}$', N, nameForCurve= 'traj', curvesIndeces=someTrajs,
                                    isXToHist=True, additionalXHistogramsArraysAndLabels=Qin_AdditionalHists,
                                    isYToHist=True, additionalYHistogramsArraysAndLabels=Qout_AdditionalHists,
                                    redLineAtXValueAndName=Qin_RedLine, redLineAtYValueAndName=Qout_RedLine,
                                    histScale=histScale)
        addInfoLines(figure)


        plt.figure('QoutVsQin')
        [plt.plot(q_in[t], q_out[t], label=f'traj {t}') for t in someTrajs ]
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.title(r'$q_{out}$ vs $q_{in}$' +'\n'+ titleSpecification)
        plt.xlabel(r'$q_{in}$')
        plt.ylabel(r'$q_{out}$')
        plt.axhline(Qstar, color='red', linestyle='dashed', linewidth=1, label=r'Q*')
        plt.scatter([1.],[Qif], marker= '*', s=45, color='black')
        if(np.max(q_out)>=0.85):
            plt.scatter([Qif], [1.], marker= '*', s=45, color='black')
        addInfoLines()

        figure, mainPlot = multipleCurvesAndHist('MVsQin', r'M vs $q_{in}$'+'\n'+ titleSpecification,
                                    q_in, r'$q_{in}$', M, 'm', N, nameForCurve= 'traj', curvesIndeces=someTrajs,
                                    isYToHist=True)
        addInfoLines(figure)
        
        figure, mainPlot = multipleCurvesAndHist('MVsQout', f'M vs ' +r'$q_{out}$'+'\n'+ titleSpecification,
                                    q_out, r'$q_{out}$', M, 'm', N, nameForCurve= 'traj', curvesIndeces=someTrajs,
                                    isYToHist=True, histScale=histScale)
        addInfoLines(figure)

        figure, mainPlot = multipleCurvesAndHist('EnergyVsQin', f'energy vs ' +r'$q_{in}$'+'\n'+ titleSpecification,
                                    q_in, r'$q_{in}$', energy, 'ebergt', N, nameForCurve= 'traj', curvesIndeces=someTrajs,
                                    isYToHist=True, histScale=histScale)
        addInfoLines(figure)

        figure, mainPlot = multipleCurvesAndHist('EnergyVsQout', f'energy vs ' +r'$q_{out}$'+'\n'+ titleSpecification,
                                    q_out, r'$q_{out}$', energy, 'energy', N, nameForCurve= 'traj', curvesIndeces=someTrajs,
                                    isYToHist=True, histScale=histScale)
        addInfoLines(figure)


    figs = plt.get_figlabels()  # Ottieni i nomi di tutte le figure create
    for fig_name in figs:
        fig = plt.figure(fig_name)
        filename = os.path.join(theseFiguresSubFolder, f'{fig_name}.png')
        fig.savefig(filename, bbox_inches='tight')
    plt.close('all')
    #Plots considering only some trajs: END
    #ANALYSIS OF SAMPLED TRAJS: END 
    
    #ANALYSIS OF AV: START 

    
    theseFiguresFolder= os.path.join(plotsFolder, 'averagedData')
    titleSpecification = 'averaged over measured trajs'
    if not os.path.exists(theseFiguresFolder):
        os.makedirs(theseFiguresFolder)
    else:
        delete_files_in_folder(theseFiguresFolder)

    file_path = get_file_with_prefix(run_Path, 'av')
    if file_path is None:
        plt.close('all')
        simData['results']=results
        writeJsonResult(simData, os.path.join(resultsFolder,'runData.json'))
        return None
    
    time = []
    avQin = []
    avQout = []
    avM = []
    avChi = []

    if simulationCode_version>=2:
        firstConfData = []
        secondConfData = []
        mileStones = [Qif, 0., Qstar/2., Qstar]
        nMileStones = len(mileStones)

    with open(file_path, 'r') as file:
        #print('analizzando ', nome_file)
        lines = file.readlines()
        dataLines = filter(lambda x: not x.startswith('#'), lines)
        data = np.genfromtxt(dataLines, delimiter=' ')
        try:
            time = data[:, 0]  # Attempt to access the first column of data
            time=data[:,0]
            avQin=data[:,1]
            avQout=data[:,2]
            avM=data[:,3]
            avChi=data[:,4]
            if simulationCode_version>=2:  #to change in check on program version, >2:
                firstConfData = data[:, 5:5+nMileStones ]
                secondConfData = data[:, 5+nMileStones:5+2*nMileStones ]
        except IndexError:
            plt.close('all')
            return None
            

    effectiveMuToUse = muAvEnergy
    if muAvEnergy<1500:
        effectiveMuToUse = 1500
    avChiErr = np.sqrt(avChi*(1.-avChi)/((lastMeasureMc/mcMeas)/(1.+2.*effectiveMuToUse/mcMeas) ))

    linearFitResults = {}
    linearFitResults2 = {}

    linearFitResults['tau'] = 'nan'
    linearFitResults['m'] = 'nan'
    linearFitResults['m_err'] = 'nan'
    linearFitResults['c'] = 'nan'
    linearFitResults['Chi'] = 'nan'

    linearFitResults2['tau'] = 'nan'
    linearFitResults2['m'] = 'nan'
    linearFitResults2['m_err'] = 'nan'
    linearFitResults2['c'] = 'nan'
    linearFitResults2['Chi'] = 'nan'

    for fitTypeAndName in [["", progressiveLinearFit, True], ["InBetween", progressiveLinearFit, False]]:#, ["2", provaEmail2,False], ["3", provaEmail3, False], ["2e3", provaEmail23, False], ["exp", provaEmailExp, False]]:#, ["2e3_4", provaEmail234]]: #:
        plt.figure('Chi'+fitTypeAndName[0])
        plt.plot(time, avChi, color='black', label=r"$\chi$")
        plt.errorbar(time, avChi, yerr=avChiErr, color='blue', fmt= ' ', marker='', elinewidth=0.4, alpha=0.3, label='err')

        fitOutput = progressiveLinearFit(time, avChi, avChiErr, onlyEnd=fitTypeAndName[2])

        plt.legend()

        if fitOutput is not None:
            paramsNames, best_fit_params, m_err, [linearity_lowerIndex, linearity_upperIndex], chi, funcToPlot = fitOutput

            x_limits = plt.xlim()
            y_limits = plt.ylim()
            if linearity_lowerIndex is not None:
                plt.axvline(time[linearity_lowerIndex], linestyle='dashed', linewidth=1, color='blue', label=r'$\tau_{trans}=$'+f'{time[linearity_lowerIndex]:.2f}')
            if linearity_upperIndex is not None:
                plt.axvline(time[linearity_upperIndex], linestyle='dashed', linewidth=1, color='blue', label=r'$\tau_{lin. end}=$'+f'{time[linearity_upperIndex]:.2f}')
            paramsLine = '\n'.join([f'{paramsNames[i]}={best_fit_params[i]:.3g}' for i in range(len(best_fit_params))])+'\n'
            plt.plot(time,funcToPlot(time, *best_fit_params), '--', label=paramsLine+r'$\chi^2_{r}$'+f'={chi:.3g}', linewidth= 0.8) 
            #plt.plot(time,funcToPlot(time, best_fit_params[0]), '--', label=r'k'+f'={best_fit_params[0]:.3g}\n'+r'$\chi^2_{r}$'+f'={chi:.3g}', linewidth= 0.8) 
            plt.xlim(x_limits)
            plt.ylim(y_limits)
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            if fitTypeAndName[0]=="":
                linearFitResults['tau'] = time[linearity_lowerIndex]
                linearFitResults['m'] = best_fit_params[0]
                linearFitResults['m_err'] =m_err
                linearFitResults['c'] = best_fit_params[1]
                linearFitResults['Chi'] = chi
            if fitTypeAndName[0]=="InBetween":
                linearFitResults2['tau'] = time[linearity_lowerIndex]
                linearFitResults2['m'] = best_fit_params[0]
                linearFitResults2['m_err'] = m_err
                linearFitResults2['c'] = best_fit_params[1]
                linearFitResults2['Chi'] = chi


        #addInfoLines()
        #plt.title(f'$\chi$ vs time\n'+titleSpecification)
        plt.xlabel('t')
        plt.ylabel(r'$\chi$')

    results['chiLinearFit'] = linearFitResults
    results['chiLinearFit_InBetween'] = linearFitResults2

    xForDerivative, yDerivative= aDiscreteDerivative(time, avChi)
    plt.figure('ChiDeriv')
    plt.plot(xForDerivative, yDerivative)
    plt.title(f'Derivative of $\chi$ vs '+r't'+'\n'+titleSpecification)
    plt.xlabel(r't')
    plt.ylabel(r'$\chi$'+'\'')
    addInfoLines()

    xForDerivative2, yDerivative2= aDiscreteDerivative(xForDerivative, yDerivative)
    plt.figure('ChiDeriv2')
    plt.plot(xForDerivative2, yDerivative2)
    plt.title(f'Second derivative of $\chi$ vs '+r't'+'\n'+titleSpecification)
    plt.xlabel(r't')
    plt.ylabel(r'$\chi$'+'\'\'')
    addInfoLines()

    plt.figure('ChiVsQout')
    plt.plot(avQout, avChi)
    plt.title(f'$\chi$ vs '+r'$q_{out}$'+'\n'+titleSpecification)
    plt.xlabel(r'$q_{out}$')
    plt.ylabel(r'$\chi$')
    plt.axvline(Qstar, color='red', linestyle='dashed', linewidth=1, label=r'Q*')
    addInfoLines()

    if simulationCode_version>=2:
        firstConfData = np.asarray(firstConfData)
        secondConfData = np.asarray(secondConfData)
        firstConfData= firstConfData.T
        secondConfData= secondConfData.T

        plt.figure('milestones_in')
        for i, milestone in enumerate(mileStones):
            specialMileStoneString = ''
            if milestone==Qstar:
                specialMileStoneString += '= q*'
            if milestone==Qif:
                specialMileStoneString += r'= $q_{if}$'
            plt.plot(time, firstConfData[i], label=r'$\tilde{q}$'+f'={milestone:.2g}'+specialMileStoneString)
        plt.title(f'Fraction of trajectories with overlap with final configuration '+r'$q_i>\tilde{q}$'+'\n'+titleSpecification)
        plt.xlabel(r'$t$')
        plt.ylabel(r'$fraction$')
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        addInfoLines()

        plt.figure('milestones_out')
        for i, milestone in enumerate(mileStones):
            specialMileStoneString = ''
            if milestone==Qstar:
                specialMileStoneString += ' = q*'
            if milestone==Qif:
                specialMileStoneString += r' = $q_{if}$'
            plt.plot(time, secondConfData[i], label=r'$\tilde{q}$'+f' = {milestone:.2g}'+specialMileStoneString)
        plt.title(f'Fraction of trajectories with overlap with final configuration '+r'$q_f>\tilde{q}$'+'\n'+titleSpecification)
        plt.xlabel(r'$t$')
        plt.ylabel(r'$fraction$')
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        addInfoLines()

    plt.figure('ChiVsQin')
    plt.plot(avQin, avChi)
    plt.title(f'$\chi$ vs '+r'$q_{in}$'+'\n'+titleSpecification)
    plt.xlabel(r'$q_{in}$')
    plt.ylabel(r'$\chi$')
    addInfoLines()

    plt.figure('ChiVsM')
    plt.plot(avM, avChi)
    plt.title(f'$\chi$ vs '+r'm'+'\n'+titleSpecification)
    plt.xlabel(r'$M$')
    plt.ylabel(r'$\chi$')
    addInfoLines()

    plt.figure('Qs')
    plt.plot(time, avQin, label=r'$q_{in}$')
    plt.plot(time, avQout, label=r'$q_{out}$')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.title(f'Qs vs time\n'+titleSpecification)
    plt.xlabel('t')
    plt.ylabel(r'$Q$')
    plt.axhline(Qstar, color='red', linestyle='dashed', linewidth=1, label=r'Q*')
    addInfoLines()

    plt.figure('M')
    plt.plot(time, avM)
    plt.title(f'Magnetization conf. vs time\n'+titleSpecification)
    plt.xlabel('t')
    plt.ylabel('m')
    addInfoLines()

    plt.figure('QoutVsQin')
    plt.plot(avQin, avQout)
    plt.title(r'$q_{out}$'+' vs '+ r'$q_{in}$'+'\n'+titleSpecification)
    plt.xlabel(r'$q_{in}$')
    plt.ylabel(r'$q_{out}$')
    plt.axhline(Qstar, color='red', linestyle='dashed', linewidth=1, label=r'Q*')
    plt.scatter([1],[Qif], marker= '*', s=45, color='black')
    if(np.max(q_out)>=0.85):
        plt.scatter([Qif], [1.], marker= '*', s=45, color='black')
    addInfoLines()

    plt.figure('QsVsM')
    plt.plot(avM, avQin, label=r'$q_{in}$')
    plt.plot(avM, avQout, label=r'$q_{out}$')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.title(f'Qs vs M.\n'+titleSpecification)
    plt.xlabel('m')
    plt.ylabel(r'$Q$')
    plt.axhline(Qstar, color='red', linestyle='dashed', linewidth=1, label=r'Q*')
    addInfoLines()


    figs = plt.get_figlabels()  # Ottieni i nomi di tutte le figure create
    for fig_name in figs:
        fig = plt.figure(fig_name)
        filename = os.path.join(theseFiguresFolder, f'{fig_name}.png')
        fig.savefig(filename, bbox_inches='tight')
    plt.close('all')

    simData['results']=results
    writeJsonResult(simData, os.path.join(resultsFolder,'runData.json'))
    

    
    

def singleStandardMCAnalysis(run_Path, configurationInfo, goFast=False):
    print('It is a MC over configurations to do thermodynamic integration.')
    simData = {}
    simData['configuration']= configurationInfo
    #getting run infos: END

    #setting results folders and plots default lines: START
    resultsFolder= os.path.join(run_Path, 'Results')
    os.makedirs(resultsFolder, exist_ok=True)
    delete_files_in_folder(resultsFolder)
    plotsFolder= os.path.join(resultsFolder, 'Plots')
    os.makedirs(plotsFolder, exist_ok=True)
    

    otherInfoKeys = ['ID', 'description', 'shortDescription']
    graphInfoKeys = [ 'graphID', 'fPosJ', 'p', 'C', 'd']
    parametersInfo_Line = ' '.join([str(parameter) + '=' + str(int(value) if value.isdigit() else float(value) if value.replace('-','',1).replace('.', '', 1).isdigit() else value) for parameter, value in simData['configuration']['parameters'].items() if parameter not in graphInfoKeys+otherInfoKeys])
    graphInfo_Line = ' '.join([str(parameter) + '=' + str(int(value) if value.isdigit() else float(value) if value.replace('.', '', 1).isdigit() else value) for parameter, value in simData['configuration']['parameters'].items() if parameter in graphInfoKeys])
    
    
    if simData['configuration']['simulationTypeId']== 10:
        graphInfo_Line = 'FM '+graphInfo_Line

    mcEq = (int)(simData['configuration']['mcParameters']['MCeq'])
    mcMeas = (int)(simData['configuration']['mcParameters']['MCmeas'])
    measuresInfo_Line = r'MC$_{eq}$='+f'{mcEq:.2g}'+' '+r'MC$_{meas}$='+f'{mcMeas:.2g}'

    settingInfo_Line = parametersInfo_Line+'\n'+ graphInfo_Line+'\n'+measuresInfo_Line
    refConInfo_Line = 'refConf:'+(simData['configuration']['referenceConfigurationsInfo']['shortDescription'])
    initInfo_Line = refConInfo_Line

    def addInfoLines():       #useful for following plots
        xlabel_position = plt.gca().xaxis.label.get_position()
        plt.text(0, xlabel_position[1] - 0.18, settingInfo_Line, fontsize=7, ha='left', va='center', transform=plt.gca().transAxes)
        plt.text(1, xlabel_position[1] - 0.18, initInfo_Line, fontsize=7, ha='right', va='center', transform=plt.gca().transAxes)
    #setting results folders and plots default lines: END


    #ANALYSIS OF TI FILES: START
    results = {}
    results['TI'] ={'beta':[]}

    TIFile = get_file_with_prefix(run_Path, 'TIbeta')
    if TIFile is not None:
        with open(TIFile, 'r') as file:
            numbers = [(np.float64)(num) for num in file.readline().replace(',', '.').split() if any(c.isdigit() or c == '.' for c in num)]
        results['TI']['beta'] = numbers[-1]
    else:
        results['TI']['beta'] = 'nan'

    #ANALYSIS OF THERMALIZATION DATA: START
    thermCheck_filePath = get_file_with_prefix(run_Path, 'thermCheck')
    if thermCheck_filePath is None:
        print('No thermCheck file found')
        return None

    theseFiguresFolder= os.path.join(plotsFolder, 'thermalization')
    if not os.path.exists(theseFiguresFolder):
        os.makedirs(theseFiguresFolder)
    else:
        delete_files_in_folder(theseFiguresFolder)

    titleSpecification = 'as from thermalization data'


    therm_H = []
    therm_HB = []


    mcMeas= (int)(simData['configuration']['mcParameters']['MCmeas'])
    mcEq= (int)(simData['configuration']['mcParameters']['MCeq'])
    firstIndexOfMeasuresAtEq = 0
    measuresToSkipInOOEPlot = 1      #The first measure may be very different even wrt non-eq ones, as it contains the initialization trajectory
    
    with open(thermCheck_filePath, 'r') as file:
        #print('analizzando ', nome_file)
        lines = file.readlines()
        dataLines = filter(lambda x: not x.startswith('#'), lines)
        data = np.genfromtxt(dataLines, delimiter=' ')
        therm_mcMeasures=data[:,0]
        therm_H=data[:,1]
        therm_HB=data[:,2]

    therm_mcMeasures = np.asarray(therm_mcMeasures, dtype=np.float64)
    therm_H = np.asarray(therm_H, dtype=np.float64)
    therm_HB = np.asarray(therm_HB, dtype=np.float64)


    if len(therm_mcMeasures)==1:
        print('Not enough measures')
        return None

    results['thermalization'] = {}
    nMusToConsider = 40
    titleForAutocorrelations = 'autocorrelation over mcSweeps'

        #defining a function to plot quantities evolution over mc iterations and the respective autocorrelation
    def mcEvolutionAndAutocorrelation(mcSweeps, quantity, firstIndexForEquilibrium,
                                      quantityShortName, quantityFullName, quantityLabelName, nMus):
        
        results['thermalization'][quantityShortName] = {}

        plt.figure(quantityShortName)
        plt.title(quantityFullName+' vs MC\n'+titleSpecification)
        plt.plot(mcSweeps, quantity)
        plt.xlabel('MC sweep')
        plt.ylabel(quantityLabelName)

        #plt.axvline(x=mcEq, color='red', linestyle='--')
        #plt.text(mcEq, plt.ylim()[1], 'MCeq', color='red', verticalalignment='bottom', horizontalalignment='right', fontsize=7)

        addInfoLines()

        results['thermalization'][quantityShortName]['mean'] = np.mean(quantity[firstIndexForEquilibrium:])
        results['thermalization'][quantityShortName]['stdErr'] = stats.sem(quantity[firstIndexForEquilibrium:])

        if (len(np.unique(quantity[firstIndexForEquilibrium:])) > 1): #i.e., if it s not a constant
            mu, muErr, rChi2, dof  = autocorrelationWithExpDecayAndMu(quantityShortName+'Autocorrelation', quantityFullName+' '+titleForAutocorrelations,
                                        mcSweeps[firstIndexForEquilibrium:], 'mc', quantity[firstIndexForEquilibrium:], quantityLabelName,
                                        nMus)
            addInfoLines()
            results['thermalization'][quantityShortName]['mu'] = mu
            results['thermalization'][quantityShortName]['muErr'] = muErr
            results['thermalization'][quantityShortName]['rChi2'] = rChi2
            results['thermalization'][quantityShortName]['dof'] = dof
        else:
            results['thermalization'][quantityShortName]['mu'] = 'nan'
            results['thermalization'][quantityShortName]['muErr'] = 'nan'
            results['thermalization'][quantityShortName]['rChi2'] = 'nan'
            results['thermalization'][quantityShortName]['dof'] = 'nan'

    mcEvolutionAndAutocorrelation(therm_mcMeasures[:len(therm_mcMeasures)], therm_H[:len(therm_mcMeasures)], firstIndexOfMeasuresAtEq,
                                      'H', 'trajectory average energy', 'energy', nMusToConsider)
    
    mcEvolutionAndAutocorrelation(therm_mcMeasures, therm_HB, firstIndexOfMeasuresAtEq,
                                      'HB', 'trajectory max energy', 'energy', nMusToConsider)

    figs = plt.get_figlabels()  # Ottieni i nomi di tutte le figure create
    for fig_name in figs:
        fig = plt.figure(fig_name)
        filename = os.path.join(theseFiguresFolder, f'{fig_name}.png')
        fig.savefig(filename, bbox_inches='tight')
    plt.close('all')

    simData['results']=results
    writeJsonResult(simData, os.path.join(resultsFolder,'runData.json'))

def singleRunAnalysis(run_Path,redoIfDone=False):

    standardMCSimIDs = [15]
    pathMCSimIDs = [10,100,11,110]

    print('Analysis of '+run_Path+'\n\n')

    #getting run infos: START
    map_file_path = '../../Data/pathsMCsInfoMap.json'  
    file_path = get_file_with_prefix(run_Path, 'info.')

    
    if file_path is None:   #this is to do the analysis only if there are info about the simulation
        file_path = get_file_with_prefix(run_Path, 'info_')
        if file_path is None:
            file_path = get_file_with_prefix(run_Path, 'details_')
            if file_path is None:
                print('No info on simulation found.')
                return None

    with open(map_file_path, 'r') as map_file:
        mappa = json.load(map_file)

    configurationInfo= txtToInfo(file_path, mappa)
    simTypeID = configurationInfo['simulationTypeId']
    if simTypeID in pathMCSimIDs:
        singlePathMCAnalysis(run_Path=run_Path, configurationsInfo=configurationInfo, redoIfDone=redoIfDone)
    elif simTypeID in standardMCSimIDs:
        singleStandardMCAnalysis(run_Path=run_Path, configurationInfo=configurationInfo)
