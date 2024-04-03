import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat

def myHist(name, title, toHist, toHistName, nbins=False):
    plt.figure(name)
    if nbins is False:
        plt.hist(toHist)
    else:
        plt.hist(toHist, bins=nbins)
    mean = np.nanmean(toHist)
    sigma = (np.nanvar(toHist))**0.5
    plt.axvline(mean, color='black', linestyle='solid', linewidth=2)
    plt.grid(True)
    plt.axvline(mean+sigma, color='red', linestyle='dashed', linewidth=1)
    plt.axvline(mean-sigma, color='red', linestyle='dashed', linewidth=1)
    plt.title(f'{title}\n mean value = {ufloat(mean, sigma)}')
    plt.xlabel(toHistName)
    plt.ylabel(f'Occurrences')
    return mean, sigma