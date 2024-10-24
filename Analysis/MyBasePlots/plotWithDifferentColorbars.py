from copy import copy, deepcopy
from matplotlib import cm
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize


def plotWithDifferentColorbars(name, x, xName, y, yName, title,
                                trajsExtremesInitID, shortDescription, edgeColorPerInitType,
                                markerShapeVariable, markerShapeVariableNames,
                                Qif, betaOfExt=None,
                                additionalMarkerTypes=None,
                                additionalMarkerTypes_Unused=None,
                                yerr=None, fitType= '', xscale='', yscale ='', fittingOverDifferentEdges=True, nGraphs=None,
                                functionsToPlotContinuously = None, theoreticalX=None, theoreticalY=None):
    
    markers = ['s', '^', 'o', 'D', 'v', 'p', 'h', 's', '^', 'o', 'D', 'v', 'p', 'h','s', '^', 'o', 'D', 'v', 'p', 'h', 's', '^', 'o', 'D', 'v', 'p', 'h']
    

    if(len(y[y!="nan"])<=1):
         return None

    xSort = np.argsort(x)
    
    if betaOfExt is None:
        betaOfExt = np.full(len(x), "nan")
    if 1==1:
        betas = np.unique(betaOfExt)
        nColorbars = betas.size
        cmaps = {}
        myMap = cm.get_cmap('cool', 256)
        # Define a base color (redder for lower values, bluer for higher values)

        fBetaOfExt=betaOfExt
        betas = betas.astype(str)
        betaOfExt = np.asarray(betaOfExt, dtype=str)
        for val in betas:
            if len(betas)==1 or val =="nan":
                normalized= 1.
            else:
                normalized = (((float) (val) - np.min(fBetaOfExt.astype(float))) / (np.max(fBetaOfExt.astype(float)) - np.min(fBetaOfExt.astype(float))))
            newcolors = myMap(np.linspace(0, 1, 256))
            newcolors[:, 3] = 1
            newcolors[:, 1] = (newcolors[:, 1]*normalized + (1-normalized))
            cmaps[val] = ListedColormap(newcolors)
        betaOfExt = betaOfExt[xSort]

        


    x = x[xSort]
    y = y[xSort]
    trajsExtremesInitID = trajsExtremesInitID[xSort]
    markerShapeVariable = markerShapeVariable[xSort]
    
    Qif = Qif[xSort]
    if yerr is not None:
        yerr = yerr[xSort]
    if functionsToPlotContinuously is not None:
        filters = deepcopy(functionsToPlotContinuously[1])
        for i, filter in enumerate(filters):
            filters[i] = filters[i][xSort]
    plotToBarRatio = 55

    figHeight = 10 + (2.2+(1.9*(nColorbars-1)))*10./plotToBarRatio
    fig = plt.figure(name, figsize=(10, figHeight))  # Adjust the figsize as needed
    gs = fig.add_gridspec(1+2*nColorbars, 1, height_ratios=[plotToBarRatio] +  [1.2,1] +[0.9,1] * (nColorbars-1))

    np.asarray(markerShapeVariable)
    curveTypes = np.unique(markerShapeVariable,axis=0)
    trajsExtInitIDs = np.unique(trajsExtremesInitID)
    Qifs = np.unique(Qif)
    # Plot the main scatter plot
    ax1 = fig.add_subplot(gs[0])
    ax1.set_title(title)
    ax1.set_xlabel(xName)
    ax1.set_ylabel(yName)
    edgeColorMap = {}

    ax1.scatter([],[], label=f"traj init:", color="None")
    for key, value in edgeColorPerInitType.items():
        for key2, value2 in shortDescription.items():
            if key == key2 and (key in trajsExtInitIDs):
                edgeColorMap[value2] = value

    for key, value in edgeColorMap.items():
        ax1.scatter([],[], label=f"{key}", color="grey", edgecolors=value)

    ax1.scatter([],[], label=f"        ", color="None")
    ax1.scatter([],[], label=', '.join(markerShapeVariableNames)+":", color="None")
    for i, t in enumerate(curveTypes):
        if i>=len(markers):
            continue
        marker= markers[i]
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
    #print(additionalMarkerTypes)
    if additionalMarkerTypes is not None:
            [ax1.errorbar([],[], label=f"{additionalMarkerType[3]}", color="grey", marker=".") for additionalMarkerType in additionalMarkerTypes ]
    plottedYs=[]
    if fitType!='':
        ax1.scatter([],[], label="", color="None")
    for i, t in enumerate(curveTypes):
        if i>=len(markers):
            continue
        
        marker = markers[i]
        for ext in trajsExtInitIDs:
            for q in Qifs:
                #[print(t,variable) for variable in markerShapeVariable]
                outCondition = np.logical_and.reduce([trajsExtremesInitID == ext,
                                                    [np.array_equal(t,variable) for variable in markerShapeVariable],
                                                    Qif == q])
                for betOfEx in np.unique(betaOfExt[outCondition]):
                    fToPlot=None
                    condition = np.logical_and(outCondition, betaOfExt==betOfEx)
                    if len(x[condition]) == 0:
                        continue
                    if len(Qifs)>1:
                        norm = Normalize(vmin=np.min(Qifs), vmax=np.max(Qifs))
                    else:
                        norm = lambda x: 0.5
                    color = cmaps[betOfEx](norm(q))

                    if functionsToPlotContinuously is not None:
                        for i, filter in enumerate(filters):
                            if np.all(filter[condition]):
                                fToPlot = functionsToPlotContinuously[0][i]
                                x_continous = np.linspace(0., np.nanmax(x[condition]), 2000)
                                y_continous = [fToPlot(x) for x in x_continous]
                                ax1.plot(x_continous, y_continous, color=edgeColorPerInitType[ext], marker=" ", linewidth=1.1)
                                ax1.plot(x_continous, y_continous, color=color,  marker=" ", linewidth=0.9)
                    else:
                        ax1.plot(x[condition], y[condition], color=color, marker=" ", linewidth=0.4)
                                
                    if yerr is None:
                        ax1.scatter(x[condition], y[condition], color=color, marker=marker, edgecolor=edgeColorPerInitType[ext], linewidths=2)
                        plottedYs.extend(y[condition])
                    else:
                        ax1.scatter(x[condition], y[condition], color=color, marker=marker, edgecolor=edgeColorPerInitType[ext], linewidths=2)
                        ax1.errorbar(x[condition], y[condition], yerr=yerr[condition], color=color, fmt= ' ', marker='')
                        plottedYs.extend(y[condition])
            
            if fittingOverDifferentEdges is False:
                if fitType=='powerLaw':
                    fitCondition =  np.logical_and(markerShapeVariable == t, trajsExtInitIDs==ext)
                    if len(np.unique(x[fitCondition]))>1:
                        popt, pcov = curve_fit(lambda x, alfa, c:  c*(x**alfa), x[fitCondition], y[fitCondition])
                        alpha = popt[0]
                        c = popt[1]
                        plt.plot(x[fitCondition], c*x[fitCondition]**alpha, linestyle='--', marker='', color=color)
                        plt.plot([], [], label=f'c={c:.3g} ' + r'$\alpha$'+f'={alpha:.3g}', linestyle='--', marker=marker, color=color)

        if fittingOverDifferentEdges is True:
            if fitType=='powerLaw':
                fitCondition =  markerShapeVariable == t
                if len(np.unique(x[fitCondition]))>1:
                    popt, pcov = curve_fit(lambda x, alfa, c:  c*(x**alfa), x[fitCondition], y[fitCondition])
                    alpha = popt[0]
                    c = popt[1]
                    plt.plot(x[fitCondition], c*x[fitCondition]**alpha, linestyle='--', marker='', color=color)

                    plt.plot([], [], label=f'c={c:.3g} ' + r'$\alpha$'+f'={alpha:.3g}', linestyle='--', marker=marker, color=color)
 
    if additionalMarkerTypes is not None:
        for additionalMarkerType in additionalMarkerTypes:
            additional_X = np.asarray(additionalMarkerType[0])
            additional_Y = np.asarray(additionalMarkerType[1])
            additional_correspBetaOfExAndQif = np.asarray(additionalMarkerType[2])
            print("add X è",additional_X)
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
                    ax1.scatter(additional_X[condition], additional_Y[condition], color=color, marker=marker, s=40)
                    ax1.plot(additional_X[condition], additional_Y[condition], color=color, marker=" ", linewidth=0.4)
                    
    if additionalMarkerTypes_Unused is not None:
        for additionalMarkerType in additionalMarkerTypes_Unused:
            additional_X = np.asarray(additionalMarkerType[0])
            additional_Y = np.asarray(additionalMarkerType[1])
            additional_correspBetaOfExAndQif = np.asarray(additionalMarkerType[2])
            print("add X è",additional_X)
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
                    ax1.scatter(additional_X[condition], additional_Y[condition], color=color, marker=marker, s=40, alpha=0.1)

    plottedYs = np.asarray(plottedYs)
    if yscale!='' and len(plottedYs[plottedYs>0])>0:
        #print(name, len(plottedYs[plottedYs>0]))
        plt.yscale(yscale)
    if xscale!='' and len(x[x>0])>0:
         plt.xscale(xscale)

    if theoreticalX is not None:
        plt.plot([],[], label=f"          ", marker=None, color="None")
        plt.plot(theoreticalX, theoreticalY, linestyle='--', marker=' ', label='cavity')
        
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    
    totalTicks = []
    sm = ['']*nColorbars
    ax = ['']*nColorbars
    totalTicks = np.unique(Qif)
    for i in range(nColorbars):
        thisBeta = betas[i]
        sm[i] = ScalarMappable(cmap=cmaps[thisBeta])
        sm[i].set_array(np.unique(Qif))
        ax[i] = fig.add_subplot(gs[2+2*i])
        cbar = plt.colorbar(sm[i], orientation='horizontal', cax=ax[i])
        cbar.set_ticks(np.unique(Qif[betaOfExt==thisBeta]))
        currentTicks =cbar.ax.get_xticks()
        desired_ticks = np.array([float(f'{tick:.2f}') for j, tick in enumerate(currentTicks) if j == 0 or np.abs(tick - currentTicks[j-1]) > np.max(np.diff(totalTicks))/12.])
        cbar.set_ticks(desired_ticks)
        # Set tick colors individually based on their values
        if i!=nColorbars-1:
            if xName == "N":
                cbar.set_label(fr'$\beta_{{extr}}$={thisBeta} N={",".join(map(str, np.unique(x[betaOfExt==thisBeta])))}', labelpad=6)
            else:
                cbar.set_label(r'$\beta_{extr}$=' + f"{thisBeta}", labelpad=6)
                 
            for tick in currentTicks:
                color = 'black'
                cbar.ax.vlines(tick, 0, 1, color=color, linewidth=4)
        else:
            if xName == "N":
                cbar.set_label(fr'$Q_{{i,f}}$, $\beta_{{extr}}$={thisBeta} N={",".join(map(str, np.unique(x[betaOfExt==thisBeta])))}', labelpad=6)
            else:
                cbar.set_label(r'$Q_{i,f}$, $\beta_{extr}$=' + f"{thisBeta}", labelpad=6)
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
    return ax1