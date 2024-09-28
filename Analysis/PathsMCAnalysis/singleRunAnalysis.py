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
from MyBasePlots.hist import myHist
from MyBasePlots.autocorrelation import autocorrelationWithExpDecayAndMu

simulationCode_version = None
currentAnalysisVersion = 'singleRunAnalysisV0002new'
fieldTypesDict = {'1': "Bernoulli", '2': "Gaussian"}


def meanAndStdErrForParametricPlot(toBecomeX, toBecomeY):
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

def singlePathMCAnalysis(run_Path, configurationsInfo, goFast=False):

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
    
    totalMC = (int)(simData['configuration']['mcParameters']['MC'])
    mcEq = (int)(simData['configuration']['mcParameters']['MCeq'])
    mcPrint = (int)(simData['configuration']['mcParameters']['MCprint'])
    mcMeas = (int)(simData['configuration']['mcParameters']['MCmeas'])

    otherInfoKeys = ['ID', 'description', 'shortDescription','fieldType','fieldRealization']
    graphInfoKeys = [ 'graphID', 'fPosJ', 'p', 'C', 'd']
    fieldInfoKeys = ['fieldMean', 'fieldSigma']
    parametersInfo_Line = ' '.join([str(parameter) + '=' + str(int(value) if value.isdigit() else float(value) if value.replace('-','',1).replace('.', '', 1).isdigit() else value) for parameter, value in simData['configuration']['parameters'].items() if parameter not in graphInfoKeys+otherInfoKeys+fieldInfoKeys])
    graphInfo_Line = ' '.join([str(parameter) + '=' + str(int(value) if value.isdigit() else float(value) if value.replace('.', '', 1).isdigit() else value) for parameter, value in simData['configuration']['parameters'].items() if parameter in graphInfoKeys])
    
    
    parametersSettingID = simData['configuration']['parameters']['ID']
    refConfSettingID = simData['configuration']['referenceConfigurationsInfo']['ID']
    if refConfSettingID== 50 or refConfSettingID== 56:
        Qif = -N
        graphInfo_Line = 'FM '+graphInfo_Line
    else:
        Qif = (int)(simData['configuration']['referenceConfigurationsInfo']['mutualOverlap'])


    if parametersSettingID== 210:
        fieldInfoKeysDict = {'fieldType': 'type', 'fieldMean':r'$\mu$', 'fieldSigma': r'$\sigma$'}
        fieldInfo_Line = ' '.join([str(fieldInfoKeysDict[parameter]) + '=' + str(int(value) if value.isdigit() else float(value) if value.replace('.', '', 1).replace('-','',1).isdigit() else value) for parameter, value in simData['configuration']['parameters'].items() if parameter in fieldInfoKeys])
        graphInfo_Line += ", w/ " + fieldTypesDict[simData['configuration']['parameters']['fieldType']]+ " field "+fieldInfo_Line+" (r"+ simData['configuration']['parameters']['fieldRealization'] +")"
        configurationsInfo['ID']+= ''.join([str(int(value) if value.isdigit() else float(value) if value.replace('.', '', 1).isdigit() else value) for parameter, value in simData['configuration']['parameters'].items() if parameter in fieldInfoKeys])
    configurationsInfo['ID']+= str(parametersSettingID)+str(refConfSettingID)+''.join([str(int(value) if value.isdigit() else float(value) if value.replace('.', '', 1).isdigit() else value) for parameter, value in simData['configuration']['parameters'].items() if parameter in graphInfoKeys]) 
    configurationsInfo['ID']+=str(parametersSettingID) + str(simData['configuration']['trajs_Initialization']['ID'])
    
    measuresInfo_Line = r'MC$_{eq}$='+f'{mcEq:.2g}'+' '+ r'MC$_{pr}$='+f'{mcPrint:.2g}'+' '+r'MC$_{meas}$='+f'{mcMeas:.2g}'
    settingInfo_Line = parametersInfo_Line+'\n'+ graphInfo_Line+'\n'+measuresInfo_Line

    if refConfSettingID in [53, 54]:
        refConInfo_Line = 'refConf:'+(simData['configuration']['referenceConfigurationsInfo']['shortDescription'])+r' $\beta$'+f"{(float)(simData['configuration']['referenceConfigurationsInfo']['betaOfExtraction']):.2g}"+'Q'+simData['configuration']['referenceConfigurationsInfo']['mutualOverlap']
    elif refConfSettingID== 53:
        refConInfo_Line = 'refConf:'+(simData['configuration']['referenceConfigurationsInfo']['shortDescription'])+r' $\beta$'+f"{(float)(simData['configuration']['referenceConfigurationsInfo']['betaOfExtraction']):.2g}"+' Q'+simData['configuration']['referenceConfigurationsInfo']['mutualOverlapBeforeQuenching']+'->'+simData['configuration']['referenceConfigurationsInfo']['mutualOverlap']
    elif refConfSettingID== 51:
        refConInfo_Line = 'refConf:'+(simData['configuration']['referenceConfigurationsInfo']['shortDescription'])+' Q'+simData['configuration']['referenceConfigurationsInfo']['mutualOverlap']
    else:
        refConInfo_Line = 'refConf:'+(simData['configuration']['referenceConfigurationsInfo']['shortDescription'])
    
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

    def addInfoLines(whereToAddLines=None):       #useful for following plots
        if whereToAddLines is None:
            xlabel_position = plt.gca().xaxis.label.get_position()
            plt.text(0, xlabel_position[1] - 0.18, settingInfo_Line, fontsize=7, ha='left', va='center', transform=plt.gca().transAxes)
            plt.text(1, xlabel_position[1] - 0.18, initInfo_Line, fontsize=7, ha='right', va='center', transform=plt.gca().transAxes)
        else:
            if isinstance(whereToAddLines, plt.Axes):  # Check if whereToAddLines is an Axes object
                xlabel_position = whereToAddLines.xaxis.label.get_position()
                whereToAddLines.text(0, xlabel_position[1] - 0.20, settingInfo_Line, fontsize=6, ha='left', va='center', transform=whereToAddLines.transAxes)
                whereToAddLines.text(1, xlabel_position[1] - 0.20, initInfo_Line, fontsize=6, ha='right', va='center', transform=whereToAddLines.transAxes)
            else:
                xlabel_position = whereToAddLines.gca().xaxis.label.get_position()
                whereToAddLines.text(0, xlabel_position[1] - 0.20, settingInfo_Line, fontsize=6, ha='left', va='center', transform=whereToAddLines.gca().transAxes)
                whereToAddLines.text(1, xlabel_position[1] - 0.20, initInfo_Line, fontsize=6, ha='right', va='center', transform=whereToAddLines.gca().transAxes)

    

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
    
    if currentAnalysisVersion==analysisVersionOfLastAnalysis and lastMeasureMc==lastMeasureMcOfLastAnalysis:
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

    #ANALYSIS OF TI FILES: START
    results['TI'] ={'beta':[], 'hout':[], 'Qstar':[]}

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
            else:
                results['TI']['beta'] = data[-1]
        
        if len(lines)>1:
            measuresCounter = np.arange(1, len(mcTimes)+1)
            previousContribution=np.roll(cumulativeUs*measuresCounter,1)
            previousContribution[0]=0
            singleUs = cumulativeUs*measuresCounter-previousContribution
            measuresCounter = np.arange(1, len(mcTimes)+1)

            measuresCounter*=mcPrint
            measuresCounter+=mcEq

            titleSpecification = 'computed over sampled trajectories'
            plt.figure('TI_beta_U')
            plt.title(r'Quantity for thermodynamic integration (U) vs mc sweeps'+'\n'+ titleSpecification)
            plt.xlabel('mc sweep')
            plt.ylabel('U')
            plt.plot(measuresCounter, cumulativeUs) 
            plt.scatter(measuresCounter, cumulativeUs, label= "cumulative average")
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
            return mu, muErr, rChi2, dof
        else:
            results['thermalization'][quantityShortName]['mu'] = 'nan'
            results['thermalization'][quantityShortName]['muErr'] = 'nan'
            results['thermalization'][quantityShortName]['rChi2'] = 'nan'
            results['thermalization'][quantityShortName]['dof'] = 'nan'

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
    Qif/=N
    Qstar/=N
    energy/=N

    nTrajs=times.shape[0]

    if nTrajs == 0:
        plt.close('all')
        simData['results'] = results
        writeJsonResult(simData, os.path.join(resultsFolder,'runData.json'))
        return None
    
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

    plt.figure('energy')
    plt.plot(times[0], energy[0])
    plt.title(f'Energy vs time\n'+ titleSpecification)
    plt.xlabel('time')
    plt.ylabel('energy')
    addInfoLines()

    plt.figure('Qin')
    plt.plot(times[0], q_in[0])
    plt.title(r'$Q_{in}$ vs time'+'\n'+ titleSpecification)
    plt.xlabel('time')
    plt.ylabel(r'$Q_{in}$')
    addInfoLines()

    plt.figure('Qout')
    plt.plot(times[0], q_out[0])
    plt.title(r'$Q_{out}$ vs time'+'\n'+ titleSpecification)
    plt.xlabel('time')
    plt.ylabel(r'$Q_{out}$')
    plt.axhline(Qstar, color='red', linestyle='dashed', linewidth=1, label=r'Q*')
    addInfoLines()

    plt.figure('M')
    plt.plot(times[0], M[0])
    plt.title(f'M vs time'+'\n'+ titleSpecification)
    plt.xlabel('time')
    plt.ylabel('M')
    addInfoLines()

    plt.figure('QoutVsQin')
    plt.plot(q_in[0], q_out[0])
    plt.title(r'$Q_{out}$ vs $Q_{in}$' +'\n'+ titleSpecification)
    plt.xlabel(r'$Q_{in}$')
    plt.ylabel(r'$Q_{out}$')
    plt.axhline(Qstar, color='red', linestyle='dashed', linewidth=1, label=r'Q*')
    plt.scatter([1.],[Qif], marker= '*', s=45, color='black')
    if(np.max(q_out)>=0.85):
        plt.scatter([Qif], [1.], marker= '*', s=45, color='black')
    addInfoLines()

    plt.figure('MVsQin')
    plt.plot(q_in[0], M[0])
    plt.title(r'M vs $Q_{in}$'+'\n'+ titleSpecification)
    plt.xlabel(r'$Q_{in}$')
    plt.ylabel(r'M')
    addInfoLines()

    plt.figure('MVsQout')
    plt.plot(q_out[0], M[0])
    plt.title(f'M vs' +r'Q_{out}'+'\n'+ titleSpecification)
    plt.xlabel(r'$Q_{out}$')
    plt.ylabel(r'M')
    plt.axvline(Qstar, color='red', linestyle='dashed', linewidth=1, label=r'Q*')
    addInfoLines()

    plt.figure('EnergyVsQin')
    plt.plot(q_in[0], energy[0])
    plt.title(f'energy vs ' +r'$Q_{in}$'+'\n'+ titleSpecification)
    plt.xlabel(r'$Q_{in}$')
    plt.ylabel(r'M')
    addInfoLines()

    plt.figure('EnergyVsQout')
    plt.plot(q_out[0], energy[0])
    plt.title(f'energy vs ' +r'$Q_{out}$'+'\n'+ titleSpecification)
    plt.xlabel(r'$Q_{out}$')
    plt.ylabel(r'energy')
    plt.axvline(Qstar, color='red', linestyle='dashed', linewidth=1, label=r'Q*')
    addInfoLines()

    plt.figure('EnergyVsM')
    plt.plot(M[0], energy[0])
    plt.title(f'energy vs M\n'+ titleSpecification)
    plt.xlabel('M')
    plt.ylabel('Energy')
    addInfoLines()

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
    plt.xlabel(r'time')
    plt.ylabel(r'$mean energy$')
    plt.plot(times.mean(0), energy.mean(0))
    addInfoLines()

    plt.figure('energyStdDev')
    plt.title(r'$\Delta$'+f' energy vs time\n'+ titleSpecification)
    plt.xlabel(r'time')
    plt.ylabel(r'$\Delta$E')
    plt.plot(times.mean(0), np.sqrt(energy.var(0)))
    addInfoLines()

    valori_unici_M = np.unique(M)
    medie_di_y_corrispondenti = [np.mean(energy[M == valore_x]) for valore_x in valori_unici_M]

    plt.figure('energyVsM')
    plt.title(f'Mean of energy vs M\n'+ titleSpecification)
    plt.xlabel(r'M')
    plt.ylabel(r'$energy$')
    a = meanAndStdErrForParametricPlot(M, energy)
    plt.errorbar(a[0],a[1],a[2], label='mean')
    plt.scatter(a[0],a[3], color='darkorange', s=25, label='median')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    addInfoLines()

    valori_unici_M = np.unique(q_in)
    medie_di_y_corrispondenti = [np.mean(energy[q_in == valore_x]) for valore_x in valori_unici_M]
    plt.figure('energyVsQin')
    plt.title(f'Mean of energy vs '+r'$Q_{in}$'+'\n'+ titleSpecification)
    plt.xlabel(r'$Q_{in}$')
    plt.ylabel(r'$energy$')
    a = meanAndStdErrForParametricPlot(q_in, energy)
    plt.errorbar(a[0],a[1],a[2], label='mean')
    plt.scatter(a[0],a[3], color='darkorange', s=25, label='median')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    addInfoLines()

    plt.figure('energyVsQout')
    plt.title(f'Mean of energy vs '+r'$Q_{out}$'+'\n'+ titleSpecification)
    plt.xlabel(r'$Q_{out}$')
    plt.ylabel(r'$energy$')
    a = meanAndStdErrForParametricPlot(q_out, energy)
    plt.errorbar(a[0],a[1],a[2], label='mean')
    plt.scatter(a[0],a[3], color='darkorange', s=25, label='median')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.axvline(Qstar, color='red', linestyle='dashed', linewidth=1, label=r'Q*')
    addInfoLines()

    barrier = energy - np.mean(energy[:, [0]], axis=1, keepdims=True)
    print(energy.shape)
    barrierIndices = np.argmax(barrier,1)
    barrier = np.max(barrier,1)
    MOfBarrier = np.asarray([M[i,index] for i,index in enumerate(barrierIndices)])
    QinOfBarrier = np.asarray([q_in[i,index] for i,index in enumerate(barrierIndices)])
    QoutOfBarrier = np.asarray([q_out[i,index] for i,index in enumerate(barrierIndices)])

    myHist('barriersHistogram', 'Histogram of energy barriers\n'+ titleSpecification, barrier, 'barrier')
    addInfoLines()

    myHist('barriersM', 'Histogram of M of energy barriers\n'+ titleSpecification, MOfBarrier, 'barrier')
    addInfoLines()

    myHist('barriersQin', 'Histogram of '+r'$Q_{in}$' + 'of energy barriers\n'+ titleSpecification, QinOfBarrier, 'barrier')
    addInfoLines()

    myHist('barriersQout', 'Histogram of '+r'$Q_{out}$' + 'of energy barriers\n'+ titleSpecification, QoutOfBarrier, 'barrier')
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

    nRandomTrajs= np.min([5, nTrajs-1])-1
    someTrajs= np.array([0])
    if nRandomTrajs>0:
        someTrajs = np.append(someTrajs, np.asarray([nTrajs-2]))
        someTrajs = np.sort(np.append(someTrajs, np.random.choice(np.arange(1, nTrajs-2), nRandomTrajs-1, replace=False)))
    
    titleSpecification = 'considering some sampled trajectories'
    
    fig = plt.figure('energy')
    gs = fig.add_gridspec(18, 100)
    mainPlot = fig.add_subplot(gs[3:, 0:85])
    [mainPlot.plot(times[t], energy[t], label=f'traj {t}') for t in someTrajs ]
    mainPlot.axhline(Qstar, color='red', linestyle='dashed', linewidth=1)
    mainPlot.set_title(r'Energy vs time'+'\n'+ titleSpecification)
    mainPlot.set_xlabel('time')
    mainPlot.set_ylabel(r'energy')
    ax_y = fig.add_subplot(gs[3:, 85:100])
    energy_min, energy_max = np.min(energy), np.max(energy)
    energy_mean, energy_VarSq = np.mean(energy), np.var(energy)**0.5
    ax_y.hist(energy.flatten(), orientation='horizontal', color='gray', alpha=0.7)
    ax_y.yaxis.tick_right()
    if fracPosJ==1.0:
        ax_y.set_xscale('log')
    axYExtremes = ax_y.get_ylim()
    ax_y.axhline(energy_mean, color='black', linestyle='solid', linewidth=2)
    ax_y.text(1.9, 1., f'Mean '+ r"$Q_{out}$"+ f': {energy_mean: .3f}  $\sigma$: {energy_VarSq: .3f}   total occurrences: {len(energy.flatten())}', color='black', ha='left', fontsize=7, rotation=270, va='top',transform=ax_y.transAxes)
    if (energy_mean-energy_VarSq>axYExtremes[0]):
        ax_y.axhline(energy_mean-energy_VarSq, color='green', linestyle='dashed', linewidth=1)
    if (energy_mean+energy_VarSq<axYExtremes[1]):
        ax_y.axhline(energy_mean+energy_VarSq, color='green', linestyle='dashed', linewidth=1)
    mainPlot.set_ylim(axYExtremes)
    plt.subplots_adjust(hspace=0.7, wspace=0.7)
    handles, labels = mainPlot.get_legend_handles_labels()
    # Create figure-level legend using handles and labels from mainPlot
    fig.legend(handles, labels, bbox_to_anchor=(-0.13, 0.6), loc='center left')
    addInfoLines(mainPlot)
    

    fig = plt.figure('Qin')
    gs = fig.add_gridspec(18, 100)
    mainPlot = fig.add_subplot(gs[3:, 0:85])
    [mainPlot.plot(times[t], q_in[t], label=f'traj {t}') for t in someTrajs ]
    mainPlot.set_title(r'$Q_{in}$ vs time'+'\n'+ titleSpecification)
    mainPlot.set_xlabel('time')
    mainPlot.set_ylabel(r'$Q_{in}$')
    ax_y = fig.add_subplot(gs[3:, 85:100])
    q_in_min, q_in_max = np.min(q_in), np.max(q_in)
    q_in_mean, q_in_VarSq = np.mean(q_in), np.var(q_in)**0.5
    bins = np.arange(q_in_min-1./N, q_in_max + 2./N, 2./N)
    ax_y.hist(q_in.flatten(), bins= bins, orientation='horizontal', color='gray', alpha=0.7)
    ax_y.yaxis.tick_right()
    if fracPosJ==1.0:
        ax_y.set_xscale('log')
    axYExtremes = ax_y.get_ylim()
    ax_y.axhline(q_in_mean, color='black', linestyle='solid', linewidth=2)
    ax_y.text(1.9, 1., f'Mean '+ r"$Q_{in}$"+ f': {q_in_mean: .3f}  $\sigma$: {q_in_VarSq: .3f}   total occurrences: {len(q_in.flatten())}', color='black', ha='left', fontsize=7, rotation=270, va='top',transform=ax_y.transAxes)
    if (q_in_mean-q_in_VarSq>axYExtremes[0]):
        ax_y.axhline(q_in_mean-q_in_VarSq, color='green', linestyle='dashed', linewidth=1)
    if (q_in_mean+q_in_VarSq<axYExtremes[1]):
        ax_y.axhline(q_in_mean+q_in_VarSq, color='green', linestyle='dashed', linewidth=1)
    mainPlot.set_ylim(axYExtremes)
    plt.subplots_adjust(hspace=0.7, wspace=0.7)
    handles, labels = mainPlot.get_legend_handles_labels()
    # Create figure-level legend using handles and labels from mainPlot
    fig.legend(handles, labels, bbox_to_anchor=(-0.13, 0.6), loc='center left')
    addInfoLines(mainPlot)

    fig = plt.figure('Qout')
    gs = fig.add_gridspec(18, 100)
    mainPlot = fig.add_subplot(gs[3:, 0:85])
    [mainPlot.plot(times[t], q_out[t], label=f'traj {t}') for t in someTrajs ]
    mainPlot.axhline(Qstar, color='red', linestyle='dashed', linewidth=1)
    mainPlot.set_title(r'$Q_{out}$ vs time'+'\n'+ titleSpecification)
    mainPlot.set_xlabel('time')
    mainPlot.set_ylabel(r'$Q_{out}$')
    ax_y = fig.add_subplot(gs[3:, 85:100])
    q_out_min, q_out_max = np.min(q_out), np.max(q_out)
    q_out_mean, q_out_VarSq = np.mean(q_out), np.var(q_out)**0.5
    bins = np.arange(q_out_min-1./N, q_out_max+2./N, 2./N)
    ax_y.hist(q_out.flatten(), bins= bins, orientation='horizontal', color='gray', alpha=0.7)
    ax_y.yaxis.tick_right()
    if fracPosJ==1.0:
        ax_y.set_xscale('log')
    axYExtremes = ax_y.get_ylim()
    ax_y.axhline(q_out_mean, color='black', linestyle='solid', linewidth=2)
    ax_y.text(1.9, 1., f'Mean '+ r"$Q_{out}$"+ f': {q_out_mean: .3f}  $\sigma$: {q_out_VarSq: .3f}   total occurrences: {len(q_out.flatten())}', color='black', ha='left', fontsize=7, rotation=270, va='top',transform=ax_y.transAxes)
    ax_y.axhline(Qstar, color='red', linestyle='dashed', linewidth=1)
    if (q_out_mean-q_out_VarSq>axYExtremes[0]):
        ax_y.axhline(q_out_mean-q_out_VarSq, color='green', linestyle='dashed', linewidth=1)
    if (q_out_mean+q_out_VarSq<axYExtremes[1]):
        ax_y.axhline(q_out_mean+q_out_VarSq, color='green', linestyle='dashed', linewidth=1)
    mainPlot.set_ylim(axYExtremes)
    plt.subplots_adjust(hspace=0.7, wspace=0.7)
    handles, labels = mainPlot.get_legend_handles_labels()
    # Create figure-level legend using handles and labels from mainPlot
    fig.legend(handles, labels, bbox_to_anchor=(-0.13, 0.6), loc='center left')
    addInfoLines(mainPlot)

    fig = plt.figure('M')
    gs = fig.add_gridspec(18, 100)
    mainPlot = fig.add_subplot(gs[3:, 0:85])
    [mainPlot.plot(times[t], M[t], label=f'traj {t}') for t in someTrajs ]
    if np.array_equal(M,q_out):
        mainPlot.axhline(Qstar, color='red', linestyle='dashed', linewidth=1)
    mainPlot.set_title(r'M vs time'+'\n'+ titleSpecification)
    mainPlot.set_xlabel('time')
    mainPlot.set_ylabel(r'M')
    ax_y = fig.add_subplot(gs[3:, 85:100])
    M_min, M_max = np.min(M), np.max(M)
    M_mean, M_VarSq = np.mean(M), np.var(M)**0.5
    bins = np.arange(M_min-1./N, M_max + 2./N, 2./N)
    ax_y.hist(M.flatten(), bins= bins, range=[M_min, M_max], orientation='horizontal', color='gray', alpha=0.7)
    ax_y.yaxis.tick_right()
    if fracPosJ==1.0:
        ax_y.set_xscale('log')
    axYExtremes = ax_y.get_ylim()
    ax_y.axhline(q_out_mean, color='black', linestyle='solid', linewidth=2)
    ax_y.text(1.9, 1., f'Mean M: {M_mean: .3f}  $\sigma$: {M_VarSq: .3f}   total occurrences: {len(M.flatten())}', color='black', ha='left', fontsize=7, rotation=270, va='top',transform=ax_y.transAxes)
    if np.array_equal(M,q_out):
        ax_y.axhline(Qstar, color='red', linestyle='dashed', linewidth=1)
    if (M_mean-M_VarSq>axYExtremes[0]):
        ax_y.axhline(M_mean-M_VarSq, color='green', linestyle='dashed', linewidth=1)
    if (M_mean+M_VarSq<axYExtremes[1]):
        ax_y.axhline(M_mean+M_VarSq, color='green', linestyle='dashed', linewidth=1)
    mainPlot.set_ylim(axYExtremes)
    plt.subplots_adjust(hspace=0.7, wspace=0.7)
    handles, labels = mainPlot.get_legend_handles_labels()
    # Create figure-level legend using handles and labels from mainPlot
    fig.legend(handles, labels, bbox_to_anchor=(-0.13, 0.6), loc='center left')
    addInfoLines(mainPlot)


    plt.figure('QoutVsQin')
    [plt.plot(q_in[t], q_out[t], label=f'traj {t}') for t in someTrajs ]
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.title(r'$Q_{out}$ vs $Q_{in}$' +'\n'+ titleSpecification)
    plt.xlabel(r'$Q_{in}$')
    plt.ylabel(r'$Q_{out}$')
    plt.axhline(Qstar, color='red', linestyle='dashed', linewidth=1, label=r'Q*')
    plt.scatter([1.],[Qif], marker= '*', s=45, color='black')
    if(np.max(q_out)>=0.85):
        plt.scatter([Qif], [1.], marker= '*', s=45, color='black')
    addInfoLines()

    plt.figure('MVsQin')
    [plt.plot(q_in[t], M[t], label=f'traj {t}') for t in someTrajs ]
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.title(r'M vs $Q_{in}$'+'\n'+ titleSpecification)
    plt.xlabel(r'$Q_{in}$')
    plt.ylabel(r'M')
    addInfoLines()

    plt.figure('MVsQout')
    [plt.plot(q_out[t], M[t], label=f'traj {t}') for t in someTrajs ]
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.title(f'M vs ' +r'$Q_{out}$'+'\n'+ titleSpecification)
    plt.xlabel(r'$Q_{out}$')
    plt.ylabel(r'M')
    plt.axvline(Qstar, color='red', linestyle='dashed', linewidth=1, label=r'Q*')
    addInfoLines()

    plt.figure('EnergyVsQin')
    [plt.plot(q_in[t], energy[t], label=f'traj {t}') for t in someTrajs ]
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.title(f'energy vs ' +r'$Q_{in}$'+'\n'+ titleSpecification)
    plt.xlabel(r'$Q_{in}$')
    plt.ylabel(r'energy')
    addInfoLines()

    plt.figure('EnergyVsQout')
    [plt.plot(q_out[t], energy[t], label=f'traj {t}') for t in someTrajs ]
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.title(f'energy vs ' +r'$Q_{out}$'+'\n'+ titleSpecification)
    plt.xlabel(r'$Q_{out}$')
    plt.ylabel(r'energy')
    plt.axvline(Qstar, color='red', linestyle='dashed', linewidth=1, label=r'Q*')
    addInfoLines()

    plt.figure('EnergyVsM')
    [plt.plot(M[t], energy[t], label=f'traj {t}') for t in someTrajs ]
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.title(f'energy vs M\n'+ titleSpecification)
    plt.xlabel('M')
    plt.ylabel('Energy')
    addInfoLines()

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
        time=data[:,0]
        avQin=data[:,1]
        avQout=data[:,2]
        avM=data[:,3]
        avChi=data[:,4]
        if simulationCode_version>=2:  #to change in check on program version, >2:
            firstConfData = data[:, 5:5+nMileStones ]
            secondConfData = data[:, 5+nMileStones:5+2*nMileStones ]

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
                plt.axvline(time[linearity_lowerIndex], linestyle='dashed', linewidth=1, color='red', label=r'$\tau_{trans}=$'+f'{time[linearity_lowerIndex]:.2f}')
            if linearity_upperIndex is not None:
                plt.axvline(time[linearity_upperIndex], linestyle='dashed', linewidth=1, color='green', label=r'$\tau_{lin. end}=$'+f'{time[linearity_upperIndex]:.2f}')
            paramsLine = '\n'.join([f'{paramsNames[i]}={best_fit_params[i]:.3g}' for i in range(len(best_fit_params))])+'\n'
            plt.plot(time,funcToPlot(time, *best_fit_params), '--', label=paramsLine+r'$\chi^2_{r}$'+f'={chi:.3g}', linewidth= 0.8) 
            #plt.plot(time,funcToPlot(time, best_fit_params[0]), '--', label=r'k'+f'={best_fit_params[0]:.3g}\n'+r'$\chi^2_{r}$'+f'={chi:.3g}', linewidth= 0.8) 
            plt.xlim(x_limits)
            plt.ylim(y_limits)
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            print("CHI", chi)
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


        addInfoLines()
        plt.title(f'$\chi$ vs time\n'+titleSpecification)
        plt.xlabel('time')
        plt.ylabel(r'$\chi$')

    results['chiLinearFit'] = linearFitResults
    results['chiLinearFit_InBetween'] = linearFitResults2

    xForDerivative, yDerivative= aDiscreteDerivative(time, avChi)
    plt.figure('ChiDeriv')
    plt.plot(xForDerivative, yDerivative)
    plt.title(f'Derivative of $\chi$ vs '+r'time'+'\n'+titleSpecification)
    plt.xlabel(r'time')
    plt.ylabel(r'$\chi$'+'\'')
    addInfoLines()

    xForDerivative2, yDerivative2= aDiscreteDerivative(xForDerivative, yDerivative)
    plt.figure('ChiDeriv2')
    plt.plot(xForDerivative2, yDerivative2)
    plt.title(f'Second derivative of $\chi$ vs '+r'time'+'\n'+titleSpecification)
    plt.xlabel(r'time')
    plt.ylabel(r'$\chi$'+'\'\'')
    addInfoLines()

    plt.figure('ChiVsQout')
    plt.plot(avQout, avChi)
    plt.title(f'$\chi$ vs '+r'$Q_{out}$'+'\n'+titleSpecification)
    plt.xlabel(r'$Q_{out}$')
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
            plt.plot(time, firstConfData[i], label=r'$\tilde{Q}$'+f'={milestone:.2g}')
        plt.title(f'Fraction of trajectories with overlap with final configuration '+r'$Q_i>\tilde{Q}$'+'\n'+titleSpecification)
        plt.xlabel(r'$t$')
        plt.ylabel(r'$fraction$')
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        addInfoLines()

        plt.figure('milestones_out')
        for i, milestone in enumerate(mileStones):
            plt.plot(time, secondConfData[i], label=r'$\tilde{Q}$'+f'={milestone}')
        plt.title(f'Fraction of trajectories with overlap with final configuration '+r'$Q_f>\tilde{Q}$'+'\n'+titleSpecification)
        plt.xlabel(r'$t$')
        plt.ylabel(r'$fraction$')
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        addInfoLines()

    plt.figure('ChiVsQin')
    plt.plot(avQin, avChi)
    plt.title(f'$\chi$ vs '+r'$Q_{in}$'+'\n'+titleSpecification)
    plt.xlabel(r'$Q_{in}$')
    plt.ylabel(r'$\chi$')
    addInfoLines()

    plt.figure('ChiVsM')
    plt.plot(avM, avChi)
    plt.title(f'$\chi$ vs '+r'M'+'\n'+titleSpecification)
    plt.xlabel(r'$M$')
    plt.ylabel(r'$\chi$')
    addInfoLines()

    plt.figure('Qs')
    plt.plot(time, avQin, label=r'$Q_{in}$')
    plt.plot(time, avQout, label=r'$Q_{out}$')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.title(f'Qs vs time\n'+titleSpecification)
    plt.xlabel('time')
    plt.ylabel(r'$Q$')
    plt.axhline(Qstar, color='red', linestyle='dashed', linewidth=1, label=r'Q*')
    addInfoLines()

    plt.figure('M')
    plt.plot(time, avM)
    plt.title(f'Magnetization conf. vs time\n'+titleSpecification)
    plt.xlabel('time')
    plt.ylabel('M')
    addInfoLines()

    plt.figure('QoutVsQin')
    plt.plot(avQin, avQout)
    plt.title(r'$Q_{out}$'+' vs '+ r'$Q_{in}$'+'\n'+titleSpecification)
    plt.xlabel(r'$Q_{in}$')
    plt.ylabel(r'$Q_{out}$')
    plt.axhline(Qstar, color='red', linestyle='dashed', linewidth=1, label=r'Q*')
    plt.scatter([1],[Qif], marker= '*', s=45, color='black')
    if(np.max(q_out)>=0.85):
        plt.scatter([Qif], [1.], marker= '*', s=45, color='black')
    addInfoLines()

    plt.figure('QsVsM')
    plt.plot(avM, avQin, label=r'$Q_{in}$')
    plt.plot(avM, avQout, label=r'$Q_{out}$')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.title(f'Qs vs M.\n'+titleSpecification)
    plt.xlabel('M')
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
            numbers = [float(num) for num in file.readline().replace(',', '.').split() if any(c.isdigit() or c == '.' for c in num)]
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

def singleRunAnalysis(run_Path):

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
        singlePathMCAnalysis(run_Path=run_Path, configurationsInfo=configurationInfo)
    elif simTypeID in standardMCSimIDs:
        singleStandardMCAnalysis(run_Path=run_Path, configurationInfo=configurationInfo)
