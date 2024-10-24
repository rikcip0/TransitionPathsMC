import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat
from scipy.optimize import curve_fit

def autocorr(x):
    """
    Compute the autocorrelation function of a 1D array.

    Parameters:
    - x: 1D array

    Returns:
    - acorr: Autocorrelation function
    """
    mean = np.mean(x)
    var = np.var(x)
    l = len(x)
    acorr = np.correlate(x - mean, x - mean, mode='full') / (var * l)
    return acorr[l-1:]

def autocorrelationWithExpDecayAndMu(name, title, xArray, xName, toComputeCorrelation, yName, nMusToConsider, p0=0):
    plt.rcParams['axes.grid'] = True
    plt.figure(name)
    plt.xlabel(xName)
    plt.ylabel(yName+ " autocorrelation")
    auto = autocorr(toComputeCorrelation)
    deltaX= xArray - xArray[0]

    if p0==0:
        if auto[1]<=0:
            p0 = 5*deltaX[1]   #so if it s 
        else:
            p0 = -deltaX[1]/np.log(auto[1])
    if len(xArray)<6:
        return xArray[-1]/2., None, None, len(xArray)
    popt, pcov = curve_fit(lambda x, mu:  np.exp(-x/mu), deltaX, auto, p0=p0)
    plt.scatter(deltaX, auto)
    mu = popt[0]
    muErr = pcov[0, 0]**0.5
    if 1==1:
        maxX = np.minimum(np.maximum(nMusToConsider*mu, deltaX[4] if len(deltaX)>=5 else deltaX[-1]), deltaX[-1] )
    else:
        maxX = deltaX[-1]
    maxXIndex = np.where(deltaX >= maxX)[0][0]
    # Calculate R-squared (coefficient of determination)
    residuals = np.exp(-deltaX[:maxXIndex] / mu) - auto[:maxXIndex]
    rChi2 = np.sum((residuals)**2/np.abs(auto[:maxXIndex]))/maxXIndex

    ls = np.linspace(0, maxX, 2000)
    plt.plot(ls, np.exp(-ls/ mu), color='red', linestyle='--', label=r'$\mu$ ' + f'={ufloat(mu, muErr)}     ' + r'$\chi^{{2,(r)}}_{{{}}}$'.format(maxXIndex) + f'={rChi2:.3g}')
    plt.xlim(0, maxX)
    plt.legend()
    ax = plt.gca()

    # Get the data limits
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # Set text position
    text_x = x_max + 0.00 * (x_max - x_min)  # Adjust the multiplier to change the distance from the right edge
    text_y = y_max + 0.04 * (y_max - y_min)  # Adjust the multiplier to change the distance from the top edge


    # Add text over the plot
    plt.text(text_x, text_y, f"Total length of sample is {deltaX[-1]:.3g}", fontsize=9, color='black', ha='right', va='top')

    plt.title(title+"\n")

    if mu-muErr <0 and mu< deltaX[1]/3:  #so, if mu is compatible with zero returns 0
        mu=0
        muErr = "nan"
    return mu, muErr, rChi2, int(maxXIndex)