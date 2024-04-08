from matplotlib import cm
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize


def plotWithDifferentColorbars(name, x, xName, y, yName, title,
                                betaOfExt, Qif,
                                trajsExtremesInitID, shortDescription, edgeColorPerInitType,
                                markerShapeVariable, markerShapeVariableName,
                                additionalMarkerTypes=None,
            yerr=None, fitType= '', xscale='', yscale ='', fittingOverDifferentEdges=True, nGraphs=None):
    
    markers = ['s', '^', 'o', 'D', 'v', 'p', 'h', 's', '^', 'o', 'D', 'v', 'p', 'h','s', '^', 'o', 'D', 'v', 'p', 'h', 's', '^', 'o', 'D', 'v', 'p', 'h']
    
    betas = np.unique(betaOfExt)
    nColorbars = betas.size
    cmaps = {}

    if(len(y[y!="nan"])<=1):
         return None

    myMap = cm.get_cmap('cool', 256)
    # Define a base color (redder for lower values, bluer for higher values)

    fBetaOfExt=betaOfExt
    betas = betas.astype(str)
    betaOfExt = betaOfExt.astype(str)

    for val in betas:
        if len(betas)==1 or val =="nan":
            normalized= 1.
        else:
            normalized = (((float) (val) - np.min(fBetaOfExt.astype(float))) / (np.max(fBetaOfExt.astype(float)) - np.min(fBetaOfExt.astype(float))))
        newcolors = myMap(np.linspace(0, 1, 256))
        newcolors[:, 3] = 1
        newcolors[:, 1] = (newcolors[:, 1]*normalized + (1-normalized))
        cmaps[val] = ListedColormap(newcolors)

    xSort = np.argsort(x)

    x = x[xSort]
    y = y[xSort]

    trajsExtremesInitID = trajsExtremesInitID[xSort]
    markerShapeVariable = markerShapeVariable[xSort]
    betaOfExt = betaOfExt[xSort]
    Qif = Qif[xSort]
    if yerr is not None:
        yerr = yerr[xSort]

    plotToBarRatio = 55

    figHeight = 10 + (2.2+(1.9*(nColorbars-1)))*10./plotToBarRatio
    fig = plt.figure(name, figsize=(10, figHeight))  # Adjust the figsize as needed
    gs = fig.add_gridspec(1+2*nColorbars, 1, height_ratios=[plotToBarRatio] +  [1.2,1] +[0.9,1] * (nColorbars-1))

    Ts = np.unique(markerShapeVariable)
    trajsExtInitIDs = np.unique(trajsExtremesInitID)
    Qifs = np.unique(Qif)
    # Plot the main scatter plot
    ax1 = fig.add_subplot(gs[0])
    ax1.set_title(title)
    ax1.set_xlabel(xName)
    ax1.set_ylabel(yName)

    edgeColorMap = {}

    for key, value in edgeColorPerInitType.items():
        for key2, value2 in shortDescription.items():
            if key == key2 and (key in trajsExtInitIDs):
                edgeColorMap[value2] = value

    for key, value in edgeColorMap.items():
        ax1.scatter([],[], label=f"{key}", color="grey", edgecolors=value)

    ax1.scatter([],[], label=f"        ", color="None")
    ax1.scatter([],[], label=markerShapeVariableName+":", color="None")

    for i, t in enumerate(Ts):
        marker= markers[i]
        if yerr is None:
            ax1.scatter([],[], label=f"{t}", color="grey", marker=marker)
        else:
            ax1.errorbar([],[], label=f"{t}", color="grey", marker=marker)
    if additionalMarkerTypes is not None:
            ax1.errorbar([],[], label=f"{additionalMarkerTypes[3]}", color="grey", marker=".")
    
    if fitType!='':
        ax1.scatter([],[], label="", color="None")

    for i, t in enumerate(Ts):
        marker = markers[i]
        for ext in trajsExtInitIDs:
            for q in Qifs:
                outCondition = np.logical_and(np.logical_and(trajsExtremesInitID == ext, markerShapeVariable == t),
                                            Qif == q)
                for betOfEx in np.unique(betaOfExt[outCondition]):
                    condition = np.logical_and(outCondition, betaOfExt==betOfEx)
                    if len(x[condition]) == 0:
                        continue
                    
                    if len(Qifs)>1:
                        norm = Normalize(vmin=np.min(Qifs), vmax=np.max(Qifs))
                    else:
                        norm = lambda x: 0.5
                    color = cmaps[betOfEx](norm(q))

                    if yerr is None:
                            ax1.scatter(x[condition], y[condition], color=color, marker=marker, edgecolor=edgeColorPerInitType[ext], linewidths=2)
                            ax1.plot(x[condition], y[condition], color=color, marker=" ", linewidth=0.4)
                    else:
                            ax1.scatter(x[condition], y[condition], color=color, marker=marker, edgecolor=edgeColorPerInitType[ext], linewidths=2)
                            ax1.plot(x[condition], y[condition], color=color, marker=" ", linewidth=0.4)
                            ax1.errorbar(x[condition], y[condition], yerr=yerr[condition], color=color, fmt= ' ', marker='')
            
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
        additional_X = additionalMarkerTypes[0]
        additional_Y = additionalMarkerTypes[1]
        additional_correspBetaOfExAndQif = additionalMarkerTypes[2]

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

    if yscale!='':
         plt.yscale(yscale)
    if xscale!='':
         plt.xscale(xscale)
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
        if yscale =='log':
            text_y = y_max + 0.09 * (y_max - y_min)  
        else:
            text_y = y_max + 0.04 * (y_max - y_min)  
        ax1.text(text_x, text_y,  f"Different graphs: {nGraphs}", fontsize=9, color='black', ha='right', va='top')
    return ax1