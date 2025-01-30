from copy import copy, deepcopy
from matplotlib import cm
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
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
                                edgeColorVariableName='Traj init',
                                colorCoordinateVariableName=r'$Q_{i,f}$', colorMapSpecifierName=r'$\beta_{extr}$',
                                additionalMarkerTypes=None,
                                additionalMarkerTypes_Unused=None,
                                yerr=None, fitTypes= [], xscale='', yscale ='', fittingOverDifferentEdges=True, nGraphs=None,
                                functionsToPlotContinuously = None, theoreticalX=None, theoreticalY=None,
                                linesAtXValueAndName=None, linesAtYValueAndName=None,):
    x=np.array(x)
    y=np.array(y)
    fittingOverDifferentShapes=True
    #checking if enough non-nan values
    valid_indices = np.isfinite(x) & np.isfinite(y)
    x=x[valid_indices]
    y=y[valid_indices]
    if(len(y)<3):
         return None, None
    markerEdgeVariable = markerEdgeVariable[valid_indices]
    markerShapeVariable = markerShapeVariable[valid_indices]
    arrayForColorCoordinate = arrayForColorCoordinate[valid_indices]
    if colorMapSpecifier is not None:
        colorMapSpecifier = colorMapSpecifier[valid_indices]
    if yerr is not None:
        yerr=np.array(yerr)
        yerr = yerr[valid_indices]

    #setting colorBars internal options
    if colorMapSpecifier is None:
        colorMapSpecifier = np.full(len(x), "nan")
    uniqueColorMapsSpecifiers = np.sort(np.unique(colorMapSpecifier))
    nColorbars = uniqueColorMapsSpecifiers.size
    cmaps = {}
    # Define a base color (redder for lower values, bluer for higher values)
    colorMapSpecifier_float=colorMapSpecifier
    uniqueColorMapsSpecifiers = uniqueColorMapsSpecifiers.astype(str)
    colorMapSpecifier = np.asarray(colorMapSpecifier, dtype=str)

    # Creare la colormap 'cool'
    gnuplot_map = plt.cm.gnuplot

    # Numero di colorbar
    num_colorbars = len(uniqueColorMapsSpecifiers)

    # Estrarre i colori dalla colormap 'cool'
    gnuplot_colors = gnuplot_map(np.linspace(0.15, 0.97, num_colorbars))
    gnuplot_colors = np.flip(gnuplot_colors, axis=0)

    # Calcolare la saturazione, che varia da 0.6 (medio) a 1 (satura)
    saturations = np.linspace(0.4, 1., 256)

    # Luminosità che varia da 1.0 (luminoso) a 0.3 (più scuro)
    values = np.linspace(1., 0.9, 256)

    if len(uniqueColorMapsSpecifiers) == 1:
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
            
            # Creare la colormap per la colorbar corrente
            cmaps[val] = ListedColormap(rgb_colors)

    #defining useful arrays
    uniqueEdgeVars = np.unique([x for x in markerEdgeVariable if x is not None])
    uniqueMarkerShapeVariable = np.unique(markerShapeVariable,axis=0)
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
    
    #setting format of image            
    plotToBarRatio = 55

    figHeight = 10 + (2.2+(1.9*(nColorbars-1)))*10./plotToBarRatio
    fig = plt.figure(name, figsize=(10, figHeight))  # Adjust the figsize as needed
    gs = fig.add_gridspec(1+2*nColorbars, 1, height_ratios=[plotToBarRatio] +  [1.2,1] +[0.9,1] * (nColorbars-1))

    # Plot the main scatter plot
    ax1 = fig.add_subplot(gs[0])
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
                        norm = Normalize(vmin=np.min(uniqueColorCoordinates), vmax=np.max(uniqueColorCoordinates))
                    else:
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
                        mErr =pcov[1,1]
                        plt.plot(xToPlot, m*xToPlot+c, linestyle='--', marker='', color=color)
                        plt.plot([], [], label=f'c+mT', linestyle='--', marker=marker, color=color)
                        plt.plot([], [], label=f'c={c:.3g} ' + r'm'+f'={m:.3g}'+'±'+f'{mErr:.3g}', linestyle='--', marker=marker, color=color)
                        plt.plot([], [], label=f' ', linestyle=' ', marker=' ')
                        fitResult= m, c, mErr
                    if 'quadratic' in fitTypes:
                        popt, pcov = curve_fit(lambda x, c, a:  c+(x*x*a), x[fitCondition], y[fitCondition])
                        c = popt[0]
                        a= popt[1]
                        mErr =pcov[1,1]
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
                    mErr =pcov[1,1]
                    plt.plot(xToPlot, m*xToPlot+c, linestyle='--', marker='', color=color)
                    plt.plot([], [], label=f'c+mT', linestyle='--', marker=marker, color=color)
                    plt.plot([], [], label=f'c={c:.3g} ' + r'm'+f'={m:.3g}'+'±'+f'{mErr:.3g}', linestyle='--', marker=marker, color=color)
                    plt.plot([], [], label=f' ', linestyle=' ', marker=' ')
                    fitResult= m, c, mErr
                if 'quadratic' in fitTypes:
                    popt, pcov = curve_fit(lambda x, c, a:  c+(x*x*a), x[fitCondition], y[fitCondition])
                    c = popt[0]
                    a= popt[1]
                    mErr =pcov[1,1]
                    plt.plot([], [], label=r'c+aT^{2}', linestyle='--', marker=marker, color=color)
                    plt.plot(xToPlot, a*xToPlot**2.+c, linestyle='--', marker='', color=color)
                    plt.plot([], [], label=f'c={c:.3g} ' + r'm'+f'={a:.3g}'+'±'+f'{mErr:.3g}', linestyle='--', marker=marker, color=color)
                    fitResult= m, c, mErr

                    
    if fittingOverDifferentShapes is True:
        xToPlot=np.linspace(np.nanmin(x), np.nanmax(x), 100)
        if len(np.unique(x))>1:
            if 'expo' in fitTypes:
                    popt, pcov = curve_fit(lambda x, c, m,s:  c*(1.-np.exp(-(x-s)*m)), x, y,p0=[y[-1],(y[1]-y[0])/(x[1]-x[0])/y[-1],0.], maxfev=10000)
                    c = popt[0]
                    m= popt[1] 
                    s= popt[2] 
                    mErr =pcov[1,1]
                    #s =popt[2]
                    plt.plot(xToPlot, c*(1.-np.exp(-(xToPlot-s)*m)), linestyle='--', marker='', color=color)
                    plt.plot([], [], label=f'exp0', linestyle='--', marker=marker, color=color)
                    plt.plot([], [], label=f'c={c:.3g} ' + r'm'+f'={m:.3g}'+'±'+f'{mErr:.3g}', linestyle='--', marker=marker, color=color)
                    plt.plot([], [], label=f' ', linestyle=' ', marker=' ')
                    fitResult= m, c, mErr
            if 'linear' in fitTypes:
                    popt, pcov = curve_fit(lambda x, c, m:  c+(x*m), x, y)
                    c = popt[0]
                    m= popt[1]
                    mErr =pcov[1,1]
                    plt.plot(xToPlot, m*xToPlot+c, linestyle='--', marker='', color=color)
                    plt.plot([], [], label=f'c+mT', linestyle='--', marker=marker, color=color)
                    plt.plot([], [], label=f'c={c:.3g} ' + r'm'+f'={m:.3g}'+'±'+f'{mErr:.3g}', linestyle='--', marker=marker, color=color)
                    plt.plot([], [], label=f' ', linestyle=' ', marker=' ')
                    fitResult= m, c, mErr
            if 'quadratic' in fitTypes:
                    popt, pcov = curve_fit(lambda x, c,a:  c+(a*x**2.), x, y)
                    c = popt[0]
                    a= popt[1]
                    mErr =pcov[1,1]
                    plt.plot([], [], label=r'$c+aT^{2}$', linestyle='--', marker=marker, color=color)
                    plt.plot(xToPlot, a*xToPlot**2.+c, linestyle='--', marker='', color=color)
                    plt.plot([], [], label=f'c={c:.3g} ' + r'm'+f'={a:.3g}'+'±'+f'{mErr:.3g}', linestyle='--', marker=marker, color=color)
                    fitResult= m, c, mErr
            if 'mix' in fitTypes:
                print(x)
                print("Y",y)
                def mix(x, k1p,k2p):
                    return (k1p/(k2p-k1p))*(np.exp(-k2p*x)-1.+k2p*x)
                k1Test=(y[-1]-y[-2])/(x[-1]-x[-2])
                popt, pcov = curve_fit(mix, x, y,p0=[k1Test,k1Test*100.],method='trf' , max_nfev=10000)
                c=0.
                k1= popt[0]
                k2= popt[1]
                k1Err =pcov[1,1]
                plt.plot(xToPlot, mix(xToPlot,k1,k2), linestyle='--', marker='', color='black')
                plt.plot([], [], label=f'c={c:.3g} '+f'k1={k1:.3g} ' + r'k2'+f'={k2:.3g}', linestyle='--', marker=marker, color=color)
                fitResult= m, c, mErr
 
    if additionalMarkerTypes_Unused is not None:
        for additionalMarkerType in additionalMarkerTypes_Unused:
            additional_X = np.asarray(additionalMarkerType[0])
            additional_Y = np.asarray(additionalMarkerType[1])
            additional_correspBetaOfExAndQif = np.asarray(additionalMarkerType[2])
            additionalXSort = np.argsort(additional_X)

            additional_X = additional_X[additionalXSort]
            additional_Y = additional_Y[additionalXSort]
            additional_correspBetaOfExAndQif =additional_correspBetaOfExAndQif[additionalXSort]

            marker = "."
            for BetaOfExAndQif in additional_correspBetaOfExAndQif:
                    if BetaOfExAndQif[0] is None:
                        continue
                    BetaOfExAndQif[0] = str(BetaOfExAndQif[0])
                    condition = np.all(additional_correspBetaOfExAndQif == BetaOfExAndQif, axis=1)
                    if len(additional_X[condition]) == 0:
                        continue
                    color = cmaps[BetaOfExAndQif[0]](norm(BetaOfExAndQif[1]))
                    ax1.scatter(additional_X[condition], additional_Y[condition], color=color, marker=marker, s=40, alpha=0.01)

    plottedYs = np.asarray(plottedYs)
    if yscale!='' and len(plottedYs[plottedYs>0])>0:
        #print(name, len(plottedYs[plottedYs>0]))
        plt.yscale(yscale)
    if xscale!='' and len(x[x>0])>0:
         plt.xscale(xscale)

    if theoreticalX is not None:
        plt.plot([],[], label=f"          ", marker=None, color="None")
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
        
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    
    totalTicks = []
    sm = ['']*nColorbars
    ax = ['']*nColorbars
    totalTicks = np.unique(arrayForColorCoordinate)
    for i in range(nColorbars):
        thisColorMapSpecifier = uniqueColorMapsSpecifiers[i]
        sm[i] = ScalarMappable(cmap=cmaps[thisColorMapSpecifier])
        sm[i].set_array(np.unique(arrayForColorCoordinate))
        ax[i] = fig.add_subplot(gs[2+2*i])
        cbar = plt.colorbar(sm[i], orientation='horizontal', cax=ax[i])
        cbar.set_ticks(np.unique(arrayForColorCoordinate[colorMapSpecifier==thisColorMapSpecifier]))
        currentTicks =cbar.ax.get_xticks()
        desired_ticks = np.array([float(f'{tick:.2f}') for j, tick in enumerate(currentTicks) if j == 0 or np.abs(tick - currentTicks[j-1]) > np.max(np.diff(totalTicks))/12.])
        cbar.set_ticks(desired_ticks)
        # Set tick colors individually based on their values
        if i!=nColorbars-1:
            if xName == "N":
                cbar.set_label(colorMapSpecifierName+'='+fr'={thisColorMapSpecifier} N={",".join(map(str, np.unique(x[colorMapSpecifier==thisColorMapSpecifier])))}', labelpad=6)
            else:
                cbar.set_label(colorMapSpecifierName+'=' + f"{thisColorMapSpecifier}", labelpad=6)
                 
            for tick in currentTicks:
                color = 'black'
                cbar.ax.vlines(tick, 0, 1, color=color, linewidth=4)
        else:
            if xName == "N":
                cbar.set_label(colorCoordinateVariableName+', '+ colorMapSpecifierName +f'={thisColorMapSpecifier} N={",".join(map(str, np.unique(x[colorMapSpecifier==thisColorMapSpecifier])))}', labelpad=6)
            else:
                cbar.set_label(colorCoordinateVariableName+', '+ colorMapSpecifierName + f"{thisColorMapSpecifier}", labelpad=6)
            for tick in totalTicks:
                if tick in currentTicks:
                    color = 'black'
                else:
                    color= 'grey'
                cbar.ax.vlines(tick, 0, 1, color=color, linewidth=4)
        cbar.ax.xaxis.tick_top()
    # Adjust layout to leave space for the colorbar
    plt.subplots_adjust(hspace=0.5)

    if nGraphs is not None:
        x_min, x_max = ax1.get_xlim()
        y_min, y_max = ax1.get_ylim()
        text_x = x_max + 0.00 * (x_max - x_min)
        if yscale =='log' and (y_min >= 0 and y_max >= 0):
            text_y = y_max * 10 ** (0.04 * (np.log10(y_max) - np.log10(y_min)))
        else:
            text_y = y_max + 0.04 * (y_max - y_min)  
        ax1.text(text_x, text_y,  f"Different graphs: {nGraphs}", fontsize=9, color='black', ha='right', va='top')
    
    return ax1, fitResult