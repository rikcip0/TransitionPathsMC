from copy import copy, deepcopy
from matplotlib import cm
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import curve_fit
import numpy as np
from matplotlib.cm import ScalarMappable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize, hsv_to_rgb, rgb_to_hsv

markers = ['s', '^', 'o', 'D', 'v', 'p', 'h', 's', '^', 'o', 'D', 'v', 'p', 'h','s', '^', 'o', 'D', 'v', 'p', 'h', 's', '^', 'o', 'D', 'v', 'p', 'h']

specialAssociation_markerShapeVariable ={"T": {"inf": [".", 40]}}

def plotWithDifferentColorbars(name, x, xName, y, yName, title,
                                markerEdgeVariable, edge_shortDescription, edgeColorPerVar,
                                markerShapeVariable, markerShapeVariableNames,
                                arrayForColorCoordinate, colorMapSpecifier=None,
                                edgeColorVariableName='Initialization',
                                colorCoordinateVariableName='', colorMapSpecifierName='',
                                dynamicalTicksForColorbars=False,
                                additionalMarkerTypes=None,
                                additionalMarkerTypes_Unused=None,
                                yerr=None, fitTypes= [], xscale='', yscale ='', fittingOverDifferentEdges=True, nGraphs=None,
                                functionsToPlotContinuously = None, theoreticalX=None, theoreticalY=None,
                                linesAtXValueAndName=None, linesAtYValueAndName=None,):
    plt.rcParams.update({
    'font.size': 14,           # dimensione font globale
    'axes.titlesize': 16,      # dimensione del titolo degli assi
    'axes.labelsize': 14,      # dimensione delle etichette degli assi
    'xtick.labelsize': 12,     # dimensione dei tick sull'asse x
    'ytick.labelsize': 12,     # dimensione dei tick sull'asse y
    'legend.fontsize': 13,     # dimensione del testo della legenda
    'figure.titlesize': 18,    # dimensione del titolo della figura (se usato)
    })
    
    x=np.array(x)
    y=np.array(y)
    fittingOverDifferentShapes=True
    #checking if enough non-nan values
    valid_indices = np.isfinite(x) & np.isfinite(y)
    if "betTentative" in name:
        valid_indices = np.logical_and(valid_indices, x<=1)
    x=x[valid_indices]
    y=y[valid_indices]

    if(len(y)<3):
         return None, None
    markerEdgeVariable = markerEdgeVariable[valid_indices]
    markerShapeVariable = markerShapeVariable[valid_indices]
    arrayForColorCoordinate = arrayForColorCoordinate[valid_indices]
    keyIsNan=False
    if colorMapSpecifier is not None:
        colorMapSpecifier = colorMapSpecifier[valid_indices]
        uniqueColorMapsSpecifiers = np.sort(np.unique(colorMapSpecifier))
        nColorbars = uniqueColorMapsSpecifiers.size
    else:
        colorMapSpecifier=np.full(len(x),"nan",dtype=np.float64)
        nColorbars=1
        keyIsNan=True
        uniqueColorMapsSpecifiers=np.array(["nan"])
    uniqueColorMapsSpecifiers = uniqueColorMapsSpecifiers.astype(str)
    colorMapSpecifier = np.asarray(colorMapSpecifier, dtype=str)
        
    if yerr is not None:
        yerr=np.array(yerr)
        yerr = yerr[valid_indices]
        
    cmaps = {}
    # Creare la colormap 'cool'
    gnuplot_map = plt.cm.gnuplot

    # Numero di colorbar
    num_colorbars = len(uniqueColorMapsSpecifiers)

    # Estrarre i colori dalla colormap 'cool'
    gnuplot_colors = gnuplot_map(np.linspace(0.12, 0.97, num_colorbars))
    gnuplot_colors = np.flip(gnuplot_colors, axis=0)

    # Calcolare la saturazione, che varia da 0.6 (medio) a 1 (satura)
    saturations = np.linspace(0.4, 1., 256)

    # Luminosità che varia da 1.0 (luminoso) a 0.3 (più scuro)
    values = np.linspace(1., 0.9, 256)

    if nColorbars == 1:
        print("ASSERTO", uniqueColorMapsSpecifiers[0])
        cmaps[uniqueColorMapsSpecifiers[0]] = plt.cm.gnuplot
    else:
        # Creazione delle colorbar
        for i, (val, color) in enumerate(zip(uniqueColorMapsSpecifiers, gnuplot_colors)):
            # Converti il colore in spazio HSV per manipolare saturazione e luminosità
            hsv_color = rgb_to_hsv(color[:3])  # Ignoriamo la componente alpha (trasparenza)
            
            # Creare la mappa di colori variando la saturazione e la luminosità
            # Manteniamo la hue (tonalità) fissa, ma variamo saturazione e luminosità
            hues = np.full_like(saturations, hsv_color[0])  # Usa la hue originale
            hsv_colors = np.stack([hues, saturations, values], axis=1)
            rgb_colors = hsv_to_rgb(hsv_colors)  # Converti di nuovo in RGB
            print("ASSEWRGNO", val)
            # Creare la colormap per la colorbar corrente
            cmaps[val] = ListedColormap(rgb_colors)

    #defining useful arrays
    uniqueEdgeVars = np.unique([x for x in markerEdgeVariable if x is not None])
    uniqueMarkerShapeVariable = np.unique(markerShapeVariable,axis=0)
    arrayForColorCoordinate=np.array(arrayForColorCoordinate,dtype=np.float64)
    uniqueColorCoordinates = np.unique(arrayForColorCoordinate)
    
    #sorting all variables according to x
    xSort = np.argsort(x)
    x = x[xSort]
    y = y[xSort]
    markerEdgeVariable = markerEdgeVariable[xSort]
    markerShapeVariable = markerShapeVariable[xSort]
    arrayForColorCoordinate = arrayForColorCoordinate[xSort]
    colorMapSpecifier = colorMapSpecifier[xSort]
    
    if yerr is not None:
        yerr = yerr[xSort]
        
    if functionsToPlotContinuously is not None:
        if deepcopy(functionsToPlotContinuously[1]) is not None:
            filters = deepcopy(functionsToPlotContinuously[1])
            for i, filter in enumerate(filters):
                if filters[i] is None:
                    filters[i]=np.full(len(xSort),True)
                else:
                    filters[i] = filters[i][xSort]
    
    fitResult=None
    
    # Parametri per il layout
    base_height = 9.5       # Altezza del plot principale
    colorbar_height = 0.25   # Altezza per ogni colorbar
    spacing_height = 0.25  # Spazio extra (ad esempio tra il plot e ogni colorbar)

    # Calcola il numero totale di righe: 
    # 1 riga per il plot principale + 2 righe per ogni colorbar (una per lo spazio e una per la colorbar)
    total_rows = 1 + 2 * nColorbars

    # Definisci le height_ratios: 
    # - La prima riga ha peso base_height.
    # - Per ogni colorbar, una riga di spazio (peso spacing_height) e una riga per la colorbar (peso colorbar_height).
    height_ratios = [base_height] + [spacing_height, colorbar_height] * nColorbars

    # Calcola l'altezza totale della figura (in pollici)
    total_fig_height = base_height + nColorbars * (colorbar_height + spacing_height)
    total_fig_length= 1.2*total_fig_height
    # Crea la figura con constrained_layout per un migliore posizionamento automatico degli elementi
    fig = plt.figure(name, figsize=(total_fig_length, total_fig_height))#, constrained_layout=True)

    # Crea la griglia (gs) con le righe definite
    gs = fig.add_gridspec(total_rows, 1, height_ratios=height_ratios)

    # Crea l'asse per il plot principale (occupando la prima riga)
    ax1 = fig.add_subplot(gs[0, 0])

    # Per ogni colorbar, supponiamo di voler utilizzare la seconda riga di ciascun gruppo (ossia le righe 2, 4, 6, ...)
    colorbar_axes = []
    ax1.set_title(title)
    ax1.set_xlabel(xName)
    ax1.set_ylabel(yName)
    edgeColorMap = {}

    #setting legend of main plot
    ax1.scatter([],[], label= edgeColorVariableName+':', color="None")
    for key, value in edgeColorPerVar.items():
        for key2, value2 in edge_shortDescription.items():
            if key == key2 and (key in uniqueEdgeVars):
                edgeColorMap[value2] = value

    for key, value in edgeColorMap.items():
        ax1.scatter([],[], label=f"{key}", color="grey", edgecolors=value)

    ax1.scatter([],[], label=f"        ", color="None")
    ax1.scatter([],[], label=', '.join(markerShapeVariableNames)+":", color="None")
    for i, t in enumerate(uniqueMarkerShapeVariable):
        if i>=len(markers):
            continue
        marker= markers[i]
        if markerShapeVariableNames==["T"]:
            if t=='inf':
                marker="."

        if t.ndim==0:
            if yerr is None:
                ax1.scatter([],[], label=f"{t}", color="grey", marker=marker)
            else:
                ax1.errorbar([],[], label=f"{t}", color="grey", marker=marker)
        else:
            if yerr is None:
                ax1.scatter([],[], label=f"{', '.join(map(str, t ))}", color="grey", marker=marker)
            else:
                ax1.errorbar([],[], label=f"{', '.join(map(str, t))}", color="grey", marker=marker)
    if additionalMarkerTypes is not None:
            [ax1.errorbar([],[], label=f"{additionalMarkerType[3]}", color="grey", marker=".") for additionalMarkerType in additionalMarkerTypes ]
    plottedYs=[]
    if fitTypes!=[]:
        ax1.scatter([],[], label="", color="None")
        
        
        
    markerEdgeVariable_forComparison = np.array([x if x is not None else -1 for x in markerEdgeVariable])    
    for i, t in enumerate(uniqueMarkerShapeVariable):
        
        if i>=len(markers):
            continue
        
        s=80
        marker = markers[i]
        if markerShapeVariableNames==["T"]:
            if t=='inf':
                s=40
                marker="."
        
        for q in uniqueColorCoordinates:
            for ext in uniqueEdgeVars:
                outCondition = np.logical_and.reduce([markerEdgeVariable_forComparison == ext,
                                                    [np.array_equal(t,variable) for variable in markerShapeVariable],
                                                    arrayForColorCoordinate == q])
                for selectedColorMap in np.unique(colorMapSpecifier[outCondition]):
                    fToPlot=None
                    condition = np.logical_and(outCondition, colorMapSpecifier==selectedColorMap)
                    if len(x[condition]) == 0:
                        continue
                    if len(uniqueColorCoordinates)>1:
                        print("QUI", np.min(uniqueColorCoordinates), np.max(uniqueColorCoordinates))
                        norm = Normalize(vmin=np.min(uniqueColorCoordinates), vmax=np.max(uniqueColorCoordinates))
                    else:
                        print("QUA")
                        norm = lambda x: 0.5
                    color = cmaps[selectedColorMap](norm(q))

                    if functionsToPlotContinuously is not None:
                        for i, filter in enumerate(filters):
                            if np.all(filter[condition]):
                                fToPlot = functionsToPlotContinuously[0][i]
                                x_continous = np.linspace(0., np.nanmax(x[condition]), 2000)
                                y_continous = [fToPlot(x) for x in x_continous]
                                ax1.plot(x_continous, y_continous, color=edgeColorPerVar[ext], marker=" ", linewidth=1.1)
                                ax1.plot(x_continous, y_continous, color=color,  marker=" ", linewidth=0.9)
                    else:
                        ax1.plot(x[condition], y[condition], color=color, marker=" ", linewidth=0.4)
 
                    if yerr is None:
                        ax1.scatter(x[condition], y[condition], color=color, marker=marker, edgecolor=edgeColorPerVar[ext], linewidths=2)
                        plottedYs.extend(y[condition])
                    else:
                        ax1.scatter(x[condition], y[condition], color=color, marker=marker, edgecolor=edgeColorPerVar[ext], linewidths=2, s=s)
                        ax1.errorbar(x[condition], y[condition], yerr=yerr[condition], color=color, fmt= ' ', marker='')
                        plottedYs.extend(y[condition])
            
            if fittingOverDifferentEdges is False:
                fitCondition =  [np.array_equal(t,variable) for variable in markerShapeVariable]
                xToPlot=np.linspace(np.nanmin(x[fitCondition]), np.nanmax(x[fitCondition]), 100)
                if len(np.unique(x[fitCondition]))>2:
                    if 'linear' in fitTypes:
                        popt, pcov = curve_fit(lambda x, c, m:  c+(x*m), x[fitCondition], y[fitCondition])
                        c = popt[0]
                        m= popt[1]
                        mErr = np.sqrt(pcov[1,1])
                        plt.plot(xToPlot, m*xToPlot+c, linestyle='--', marker='', color=color)
                        plt.plot([], [], label=f'c+mT', linestyle='--', marker=marker, color=color)
                        plt.plot([], [], label=f'c={c:.3g} ' + r'm'+f'={m:.3g}'+'±'+f'{mErr:.3g}', linestyle='--', marker=marker, color=color)
                        plt.plot([], [], label=f' ', linestyle=' ', marker=' ')
                        fitResult= m, c, mErr
                        
                    if 'quadratic' in fitTypes:
                        popt, pcov = curve_fit(lambda x, c, a:  c+(x*x*a), x[fitCondition], y[fitCondition])
                        c = popt[0]
                        a= popt[1]
                        mErr =np.sqrt(pcov[1,1])
                        plt.plot([], [], label=r'c+aT^{2}', linestyle='--', marker=marker, color=color)
                        plt.plot(xToPlot, a*xToPlot**2.+c, linestyle='--', marker='', color=color)
                        plt.plot([], [], label=f'c={c:.3g} ' + r'm'+f'={a:.3g}'+'±'+f'{mErr:.3g}', linestyle='--', marker=marker, color=color)
                        fitResult= m, c, mErr
                        
        if fittingOverDifferentEdges is True and fittingOverDifferentShapes is False:
            fitCondition =  [np.array_equal(t,variable) for variable in markerShapeVariable]
            xToPlot=np.linspace(np.nanmin(x[fitCondition]), np.nanmax(x[fitCondition]), 100)
            if len(np.unique(x[fitCondition]))>2:
                if 'linear' in fitTypes:
                    popt, pcov = curve_fit(lambda x, c, m:  c+(x*m), x[fitCondition], y[fitCondition])
                    c = popt[0]
                    m= popt[1]
                    mErr = np.sqrt(pcov[1,1])
                    plt.plot(xToPlot, m*xToPlot+c, linestyle='--', marker='', color='grey')
                    plt.plot([], [], label=f'c+mT', linestyle='--', marker=marker, color='grey')
                    plt.plot([], [], label=f'c={c:.3g} ' + r'm'+f'={m:.3g}'+'±'+f'{mErr:.3g}', linestyle='--', marker=marker, color='grey')
                    plt.plot([], [], label=f' ', linestyle=' ', marker=' ')
                    fitResult= m, c, mErr
                if 'quadratic' in fitTypes:
                    popt, pcov = curve_fit(lambda x, c, a:  c+(x*x*a), x[fitCondition], y[fitCondition])
                    c = popt[0]
                    a= popt[1]
                    mErr = np.sqrt(pcov[1,1])
                    plt.plot([], [], label=r'c+aT^{2}', linestyle='--', marker=marker, color='grey')
                    plt.plot(xToPlot, a*xToPlot**2.+c, linestyle='--', marker='', color='grey')
                    plt.plot([], [], label=f'c={c:.3g} ' + r'm'+f'={a:.3g}'+'±'+f'{mErr:.3g}', linestyle='--', marker=marker, color='grey')
                    fitResult= m, c, mErr

                    
    if fittingOverDifferentShapes is True:
        xToPlot=np.linspace(np.nanmin(x), np.nanmax(x), 100)
        if len(np.unique(x))>1:
            if 'expo' in fitTypes:
                    popt, pcov = curve_fit(lambda x, c, m,s:  c*(1.-np.exp(-(x-s)*m)), x, y,p0=[y[-1],(y[1]-y[0])/(x[1]-x[0])/y[-1],0], maxfev=100000)
                    c = popt[0]
                    m= popt[1] 
                    s= popt[2] 
                    mErr = np.sqrt(pcov[1,1])
                    #s =popt[2]
                    plt.plot(xToPlot, c*(1.-np.exp(-(xToPlot-s)*m)), linestyle='--', marker='', color='grey')
                    plt.plot([], [], label=f'exp0', linestyle='--', marker=marker, color='grey')
                    plt.plot([], [], label=f's={s:.3g} ' +f'c={c:.3g} ' + r'm'+f'={m:.3g}'+'±'+f'{mErr:.3g}', linestyle='--', marker=marker, color='grey')
                    plt.plot([], [], label=f' ', linestyle=' ', marker=' ')
                    fitResult= m, c, mErr
            if 'linear' in fitTypes:
                    popt, pcov = curve_fit(lambda x, c, m:  c+(x*m), x, y)
                    c = popt[0]
                    m= popt[1]
                    mErr = np.sqrt(pcov[1,1])
                    plt.plot(xToPlot, m*xToPlot+c, linestyle='--', marker='', color='grey')
                    plt.plot([], [], label=f'c+mT', linestyle='--', marker=marker, color='grey')
                    plt.plot([], [], label=f'c={c:.3g} ' + r'm'+f'={m:.3g}'+'±'+f'{mErr:.3g}', linestyle='--', marker=marker, color='grey')
                    plt.plot([], [], label=f' ', linestyle=' ', marker=' ')
                    fitResult= m, c, mErr
            if 'expo2' in fitTypes:
                def model(x, ty, delta,s):
                    xt = ty / delta**2
                    return xt/(xt - ty) + (xt*ty)/(ty - xt)*(x-s) + (xt/(ty - xt))*np.exp(-(x-s)*ty)
                def model2(x, a, b,c,delta):
                    d = c / delta**2
                    return a+b(xToPlot)+c*np.exp(-(xToPlot)*d)
                p0=[(y[-1]-y[-2])/(x[-1]-x[-2]),(y[-1]-y[-2])/(x[-1]-x[-2]),(y[-1]-y[-2])/(x[-1]-x[-2]),5]
                popt, pcov = curve_fit(lambda x, a, b,c,delta:  model2(x,a,b,c,delta), x, y,p0=p0, maxfev=5000000, bounds=([0.,1.01,0],[np.inf,np.inf,10]))
                a = popt[0]
                ty = popt[0]
                b= popt[1] 
                c=popt[2]
                delta=popt[3]
                d= c/(delta*delta) 
                mErr = np.sqrt(pcov[1,1])
                plt.plot(xToPlot, model2(xToPlot,a,b,c,delta), linestyle='--', marker='x', color='grey')
                plt.plot([], [], label=f'exp02', linestyle='--', marker=marker, color='grey')
                plt.plot([], [], label=f's={s:.3g} ' +f'ty={c:.3g} ' + r'xt'+f'={xt:.3g}'+'±'+f'{mErr:.3g}', linestyle='--', marker=marker, color='grey')
                plt.plot([], [], label=f' ', linestyle=' ', marker=' ')
                fitResult= m, c, mErr
            if 'quadratic' in fitTypes:
                    popt, pcov = curve_fit(lambda x, c,a:  c+(a*x**2.), x, y)
                    c = popt[0]
                    a= popt[1]
                    mErr = np.sqrt(pcov[1,1])
                    plt.plot([], [], label=r'$c+aT^{2}$', linestyle='--', marker=marker, color='grey')
                    plt.plot(xToPlot, a*xToPlot**2.+c, linestyle='--', marker='', color='grey')
                    plt.plot([], [], label=f'c={c:.3g} ' + r'm'+f'={a:.3g}'+'±'+f'{mErr:.3g}', linestyle='--', marker=marker, color='grey')
                    fitResult= a, c, mErr
            if 'mix' in fitTypes:
                def mix(x, k1p,k2p):
                    return (k1p/(k2p-k1p))*(np.exp(-k2p*x)-1.+k2p*x)
                k1Test=(y[-1]-y[-2])/(x[-1]-x[-2])
                popt, pcov = curve_fit(mix, x, y,p0=[k1Test,k1Test*100.],method='trf' , max_nfev=10000)
                c=0.
                k1= popt[0]
                k2= popt[1]
                k1Err = np.sqrt(pcov[1,1])
                plt.plot(xToPlot, mix(xToPlot,k1,k2), linestyle='--', marker='', color='grey')
                plt.plot([], [], label=f'c={c:.3g} '+f'k1={k1:.3g} ' + r'k2'+f'={k2:.3g}', linestyle='--', marker=marker, color='grey')
                fitResult= m, c, mErr
 
    if additionalMarkerTypes_Unused is not None:
        for additionalMarkerType in additionalMarkerTypes_Unused:
            additional_X = np.asarray(additionalMarkerType[0])
            additional_Y = np.asarray(additionalMarkerType[1])
            print(additionalMarkerType)
            print("EECC", additionalMarkerType[2])
            additional_correspBetaOfExAndQif = np.transpose(np.asarray(additionalMarkerType[2]))
            additional_correspBetaOfExAndQif[1]=additional_correspBetaOfExAndQif[1].astype(np.float64)
            additionalXSort = np.argsort(additional_X)

            additional_X = additional_X[additionalXSort]
            additional_Y = additional_Y[additionalXSort]
            additional_correspBetaOfExAndQif =additional_correspBetaOfExAndQif[additionalXSort]

            marker = "."
            for BetaOfExAndQif in additional_correspBetaOfExAndQif:
                    if BetaOfExAndQif[0] is None:
                        continue
                    print(BetaOfExAndQif[0])
                    BetaOfExAndQif[0] = str(BetaOfExAndQif[0])
                    condition = np.all(additional_correspBetaOfExAndQif == BetaOfExAndQif, axis=1)
                    if len(additional_X[condition]) == 0:
                        continue
                    print(BetaOfExAndQif)
                    if keyIsNan:
                        BetaOfExAndQif[0]='nan'
                    print(cmaps)
                    print(str(BetaOfExAndQif[0]))
                    print(norm(BetaOfExAndQif[1].astype(np.float64)))
                    color = cmaps[str(BetaOfExAndQif[0])](norm(BetaOfExAndQif[1].astype(np.float64)))
                    ax1.scatter(additional_X[condition], additional_Y[condition], color=color, marker=marker, s=40, alpha=0.01)

    plottedYs = np.asarray(plottedYs)
    if yscale!='' and len(plottedYs[plottedYs>0])>0:
        #print(name, len(plottedYs[plottedYs>0]))
        plt.yscale(yscale)
    if xscale!='' and len(x[x>0])>0:
         plt.xscale(xscale)

    if theoreticalX is not None:
        plt.plot([],[], label=f"          ", marker=None, color="None")
        if "betTentative" in name:
            f= theoreticalX<=1
            theoreticalX = theoreticalX[f]
            theoreticalY = theoreticalY[f] 
        plt.plot(theoreticalX, theoreticalY, linestyle='--', marker=' ', label='cavity')
    
    if linesAtXValueAndName is not None:
        plt.plot([],[], label=f"          ", marker=None, color="None")
        for l in linesAtXValueAndName :
            l_Value, l_Name, l_color = l
            plt.axvline(l_Value, color=l_color, linestyle='dashed', marker=' ', linewidth=1)
            plt.plot([],[], label=f"{l_Name}", linestyle='dashed', marker=' ', color=l_color)
    
    if linesAtYValueAndName is not None:
        plt.plot([],[], label=f"          ", marker=None, color="None")
        for l in linesAtYValueAndName :
            l_Value, l_Name, l_color = l
            plt.axhline(l_Value, color=l_color, linestyle='dashed', marker=' ', linewidth=1)
            plt.plot([],[], label=f"{l_Name}", linestyle='dashed', marker=' ', color=l_color)
        
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels,
            bbox_to_anchor=(1.05, 1.0),
            loc='upper left',
            borderaxespad=0.,
            bbox_transform=ax1.transAxes)
    
    sm = ['']*nColorbars

    for i, thisColorMapSpecifier in enumerate(uniqueColorMapsSpecifiers):
        # Crea l'asse per la colorbar (non serve l'asse extra ax_cb, se non lo usi per altro)
        ax_colorbar = fig.add_subplot(gs[2 + 2 * i, 0])
        
        # Seleziona la mappa per questo gruppo
        subset = arrayForColorCoordinate[colorMapSpecifier == thisColorMapSpecifier]
        norm = Normalize(vmin=np.min(subset), vmax=np.max(subset))
        sm[i] = ScalarMappable(cmap=cmaps[thisColorMapSpecifier], norm=norm)
        sm[i].set_array(subset)
        
        # Crea la colorbar orizzontale sull'asse dedicato
        cbar = plt.colorbar(sm[i], orientation='horizontal', cax=ax_colorbar, pad=0.0)

        if dynamicalTicksForColorbars:
            currentTicks = np.sort(np.unique(subset))
            desired_ticks = np.array([float(f'{tick}') for j, tick in enumerate(currentTicks) if j == 0 or np.abs(tick - currentTicks[j-1]) > np.mean(np.diff(currentTicks))/6.])
            cbar.set_ticks(desired_ticks)
            cbar.ax.xaxis.set_major_formatter(ScalarFormatter())
        # Imposta le etichette dei tick (solo per l'asse che gestiremo)
        #cbar.ax.set_xticklabels([f"{tick}" for tick in desired_ticks])
        
        # Forza la posizione dei tick: solo in alto
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.tick_params(axis='x', which='both', bottom=False, top=True,
                            labelbottom=False, labeltop=True)
        
        # Disattiva ogni tick e ticklabel sugli assi verticali (sinistra/destra)
        cbar.ax.tick_params(axis='y', which='both', left=False, right=False,
                            labelleft=False, labelright=False)
        
        # Se per caso appaiono ancora ticklabel "extra" (ad esempio sul fondo),
        # forzali invisibili:
        for label in cbar.ax.get_xticklabels():
            # Se il centro del label (y) è al di sotto di 0.5 (la metà dell'asse normalizzato),
            # lo nascondiamo (questo è un hack; di solito labelbottom=False è sufficiente)
            if label.get_position()[1] < 0.5:
                label.set_visible(False)
        
        # Imposta il label della colorbar
        if i+1 != nColorbars:
            cbar.set_label(colorMapSpecifierName + '=' + f"{thisColorMapSpecifier}", labelpad=6)
        else:
            if nColorbars == 1 and (str(thisColorMapSpecifier) == 'nan' or thisColorMapSpecifier is None):
                cbar.set_label(colorCoordinateVariableName, labelpad=6)
            else:
                cbar.set_label(colorCoordinateVariableName + ', ' + colorMapSpecifierName + '=' + f"{thisColorMapSpecifier}", labelpad=6)
        
        # Posiziona manualmente il label in basso (modifica il valore di y se serve)
        cbar.ax.xaxis.set_label_coords(0.5, -1.)

    if nGraphs is not None:
        x_min, x_max = ax1.get_xlim()
        y_min, y_max = ax1.get_ylim()
        text_x = x_max + 0.00 * (x_max - x_min)
        if yscale =='log' and (y_min >= 0 and y_max >= 0):
            text_y = y_max * 10 ** (0.04 * (np.log10(y_max) - np.log10(y_min)))
        else:
            text_y = y_max + 0.04 * (y_max - y_min)  
        ax1.text(text_x, text_y,  f"Different graphs: {nGraphs}", fontsize=11, color='black', ha='right', va='top')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.1, hspace=0.3)
    return ax1, fitResult