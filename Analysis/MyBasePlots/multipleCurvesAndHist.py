from copy import copy, deepcopy
from matplotlib import cm
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize


def multipleCurvesAndHist(name, xArray, xName, yArray, yName, title, curvesIndeces,
                        nameForCurve, isYToHist, additionalHistogramsArrays, N,
                        yArrayErr=None,  histScale =''):
    
    
    fig = plt.figure(name)
    
    nTotalHistograms=0
    
    if isYToHist:
        nTotalHistograms+=1
    
    if additionalHistogramsArrays is not None:
        nTotalHistograms+=len(additionalHistogramsArrays)
    
    gs = fig.add_gridspec(18, 85+(15*nTotalHistograms))

    mainPlot = fig.add_subplot(gs[3:, 0:85])
    [mainPlot.plot(xArray[t], yArray[t], label=f'{nameForCurve} {t}') for t in curvesIndeces ]
    mainPlot.set_title(title)
    mainPlot.set_xlabel(xName)
    mainPlot.set_ylabel(yName)
    
    if isYToHist:
        ax_y = fig.add_subplot(gs[3:, 85:100])
        y_min, y_max = np.min(yArray), np.max(yArray)
        y_mean, y_VarSq = np.mean(yArray), np.var(yArray)**0.5
        bins = np.arange(y_min-1./N, y_max + 2./N, 2./N)
        ax_y.hist(yArray.flatten(), bins= bins, orientation='horizontal', color='gray', alpha=0.7)
        ax_y.yaxis.tick_right()
        if histScale!='':
            ax_y.set_xscale(histScale)
        axYExtremes = ax_y.get_ylim()
        ax_y.axhline(y_mean, color='black', linestyle='solid', linewidth=2)
        ax_y.text(1.9, 1., f'Mean '+ r"$Q_{in}$"+ f': {y_mean: .3f}  $\sigma$: {y_VarSq: .3f}   total occurrences: {len(yArray.flatten())}', color='black', ha='left', fontsize=7, rotation=270, va='top',transform=ax_y.transAxes)
        if (y_mean-y_VarSq>axYExtremes[0]):
            ax_y.axhline(y_mean-y_VarSq, color='green', linestyle='dashed', linewidth=1)
        if (y_mean+y_VarSq<axYExtremes[1]):
            ax_y.axhline(y_mean+y_VarSq, color='green', linestyle='dashed', linewidth=1)
    
    mainPlot.set_ylim(axYExtremes)
    for i, histArray in enumerate(additionalHistogramsArrays):
        ax_y2 = fig.add_subplot(gs[3:, 100+i*15:100+(i+1)*15])
        q_in_min, q_in_max = np.min(histArray), np.max(histArray)
        q_in_mean, q_in_VarSq = np.mean(histArray), np.var(histArray)**0.5
        bins = np.arange(q_in_min-1./N, q_in_max + 2./N, 2./N)
        ax_y2.hist(histArray.flatten(), bins= bins, orientation='horizontal', color='gray', alpha=0.7)
        ax_y2.yaxis.tick_right()
        if histScale!='':
            ax_y.set_xscale(histScale)
        axYExtremes = ax_y2.get_ylim()
        ax_y2.axhline(q_in_mean, color='black', linestyle='solid', linewidth=2)
        ax_y2.text(1.9, 1., f'Mean '+ r"$Q_{in}$"+ f': {q_in_mean: .3f}  $\sigma$: {q_in_VarSq: .3f}   total occurrences: {len(histArray.flatten())}', color='black', ha='left', fontsize=7, rotation=270, va='top',transform=ax_y2.transAxes)
        if (q_in_mean-q_in_VarSq>axYExtremes[0]):
            ax_y2.axhline(q_in_mean-q_in_VarSq, color='green', linestyle='dashed', linewidth=1)
        if (q_in_mean+q_in_VarSq<axYExtremes[1]):
            ax_y2.axhline(q_in_mean+q_in_VarSq, color='green', linestyle='dashed', linewidth=1)
        mainPlot.set_ylim(axYExtremes)
        ax_y.set_ylim(axYExtremes)
        
    plt.subplots_adjust(hspace=0.7, wspace=0.7)
    handles, labels = mainPlot.get_legend_handles_labels()
    # Create figure-level legend using handles and labels from mainPlot
    fig.legend(handles, labels, bbox_to_anchor=(-0.13, 0.6), loc='center left')
    return fig, mainPlot