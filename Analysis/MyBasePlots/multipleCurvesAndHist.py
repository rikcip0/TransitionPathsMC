from copy import copy, deepcopy
from matplotlib import cm
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize

topPad= 3
mainPlotGSLen=114
mainPlotGSHeight=72
histogramGSLen= 18
histogramGSPad=1

def multipleCurvesAndHist(name, title, xArray, xName, yArray, yName, N, yArrayErr=None,
                           curvesIndeces=None, nameForCurve=None, 
                           isYToHist=False, additionalYHistogramsArraysAndLabels=[], redLineAtYValueAndName=None,
                           isXToHist=False, additionalXHistogramsArraysAndLabels=[], redLineAtXValueAndName=None, histScale=''):
    
    if curvesIndeces is None and np.ndim(xArray)>1:
        print("Provided array is more than 1d but no specification about what to plot was given.")
        return
    
    
    nTotalXHistograms = 0
    if isXToHist:
        additionalXHistogramsArraysAndLabels = [ [xArray, ''], *additionalXHistogramsArraysAndLabels]
    nTotalXHistograms = len(additionalXHistogramsArraysAndLabels)
        
    nTotalYHistograms = 0
    if isYToHist:
        additionalYHistogramsArraysAndLabels = [ [yArray, ''], *additionalYHistogramsArraysAndLabels]
    
    nTotalYHistograms = len(additionalYHistogramsArraysAndLabels)
    
    fig = plt.figure(name)
    gs = fig.add_gridspec(topPad+ mainPlotGSHeight+ (nTotalXHistograms) * (histogramGSLen+histogramGSPad), mainPlotGSLen+ (nTotalYHistograms) * (histogramGSLen+histogramGSPad))

    mainPlot = fig.add_subplot(gs[topPad:topPad+mainPlotGSHeight, 0:mainPlotGSLen])
    mainPlot.set_title(title)
    mainPlot.set_xlabel(xName)
    mainPlot.set_ylabel(yName)
    
    
    # Keep track of legend handles and labels for curves
    curve_handles = []
    curve_labels = []

    histogram_handles = []
    histogram_labels = [r'$\mu$', r'$\mu \pm \sigma$']
    
    # Plot each curve and store handles and labels
    if curvesIndeces is not None:
        for t in curvesIndeces:
            line, = mainPlot.plot(xArray[t], yArray[t], label=f'{nameForCurve} {t}')
            curve_handles.append(line)
            curve_labels.append(f'{nameForCurve} {t}')
    else:
        mainPlot.plot(xArray, yArray)
        
    if redLineAtXValueAndName is not None:
        redLineAtXValue_Value, redLineAtXValue_Name = redLineAtXValueAndName
        redLine = mainPlot.axvline(redLineAtXValue_Value, color='red', linestyle='dashed', linewidth=1)
        curve_handles.append(redLine)
        curve_labels.extend([redLineAtXValue_Name])
    
    if redLineAtYValueAndName is not None:
        redLineAtYValue_Value, redLineAtYValue_Name = redLineAtYValueAndName
        redLine = mainPlot.axhline(redLineAtYValue_Value, color='red', linestyle='dashed', linewidth=1)
        histogram_labels.extend([redLineAtYValue_Name])
        
        
    nPlottedXHistograms=0
    nPlottedYHistograms=0
    
    # Keep track of legend handles and labels for histograms
    if additionalXHistogramsArraysAndLabels is not None:
        for i, histArrayAndLabel in enumerate(additionalXHistogramsArraysAndLabels):
            histArray, label = histArrayAndLabel
            if isXToHist:
                ax_x = fig.add_subplot(gs[topPad+mainPlotGSHeight+ (nPlottedXHistograms+1)*(histogramGSPad+histogramGSLen)-histogramGSLen:topPad+mainPlotGSHeight+ (nPlottedXHistograms+1)*(histogramGSPad+histogramGSLen), 0:mainPlotGSLen ])                
                x_min, x_max = np.min(histArray), np.max(histArray)
                x_mean, x_VarSq = np.mean(histArray), np.var(histArray)**0.5
                bins = np.arange(x_min - 1. / N, x_max + 2. / N, 2. / N)
                ax_x.hist(histArray.flatten(), bins=bins, color='gray', alpha=0.7, density=True)
                if histScale != '':
                    ax_x.set_xscale(histScale)
                    
                nPlottedXHistograms+=1
                ax_x.xaxis.tick_bottom()
                if nPlottedXHistograms<nTotalYHistograms:
                    ax_x.tick_params(color='gray')

                axXExtremes = ax_x.get_xlim()
                
                mean_line = ax_x.axvline(x_mean, color='black', linestyle='solid', linewidth=2)
                sigma_lower = ax_x.axvline(x_mean - x_VarSq, color='green', linestyle='dashed', linewidth=1)
                sigma_upper = ax_x.axvline(x_mean + x_VarSq, color='green', linestyle='dashed', linewidth=1)
                if redLineAtXValueAndName is not None:
                    ax_x.axvline(redLineAtXValue_Value, color='red', linestyle='dashed', linewidth=1)
                mainPlot.set_xlim(axXExtremes)
        
    if additionalYHistogramsArraysAndLabels is not None:
        for i, histArrayAndLabel in enumerate(additionalYHistogramsArraysAndLabels):
            histArray, label = histArrayAndLabel
            
            ax_y2 = fig.add_subplot(gs[topPad:topPad+ mainPlotGSHeight, mainPlotGSLen+nPlottedYHistograms*(histogramGSPad+histogramGSLen)+histogramGSPad:mainPlotGSLen+(nPlottedYHistograms+1)*(histogramGSPad+histogramGSLen)])
            
            q_in_min, q_in_max = np.min(histArray), np.max(histArray)
            q_in_mean, q_in_VarSq = np.mean(histArray), np.var(histArray)**0.5
            bins = np.arange(q_in_min - 1. / N, q_in_max + 2. / N, 2. / N)
            ax_y2.hist(histArray.flatten(), bins=bins, orientation='horizontal', color='gray', alpha=0.7, density=True)
            
            if histScale != '':
                ax_y2.set_xscale(histScale)
                
            nPlottedYHistograms+=1
            ax_y2.yaxis.tick_right()
            if nPlottedYHistograms<=nTotalYHistograms:
                ax_y2.tick_params(color='gray')

            mean_line = ax_y2.axhline(q_in_mean, color='black', linestyle='solid', linewidth=2)
            sigma_lower = ax_y2.axhline(q_in_mean - q_in_VarSq, color='green', linestyle='dashed', linewidth=1)
            sigma_upper = ax_y2.axhline(q_in_mean + q_in_VarSq, color='green', linestyle='dashed', linewidth=1)
            if redLineAtYValueAndName is not None:
                ax_y2.axhline(redLineAtYValue_Value, color='red', linestyle='dashed', linewidth=1)

            ax_y2.text(0.5, 1.02, label, color='black', ha='center', va='bottom', 
                fontsize=9, transform=ax_y2.transAxes) 
            
            # Add histogram legend handles

            axYExtremes = ax_y2.get_ylim()
            mainPlot.set_ylim(axYExtremes)
            ax_y2.set_ylim(axYExtremes)
            
    if nTotalYHistograms>0:
        histogram_handles = [mean_line, sigma_upper]
        
    if redLineAtYValueAndName is not None:
        histogram_handles.extend([redLine])
    
    plt.subplots_adjust(hspace=0.7, wspace=0.7)

    # Create curve legend on the left
    verticalPosition=0.775
    if len(curve_labels)>0:
        fig.legend(curve_handles, curve_labels, loc='upper left', bbox_to_anchor=(-0.13, verticalPosition), frameon=True)

    if len(histogram_labels)>0:
        # Create histogram legend on the right
        fig.legend(histogram_handles, histogram_labels, loc='upper right', bbox_to_anchor=(1.10, verticalPosition), frameon=True, handlelength=0.75)

    return fig, mainPlot