import matplotlib.pyplot as plt
import numpy as np
import math
from uncertainties import ufloat

def find_rows_similarity(matrix):
    matrix = np.sort(matrix, axis=1)[:, ::-1]
    flattened_matrix = matrix.flatten()
    hist, bins = np.histogram(flattened_matrix, bins=50, density=True)
    cum_hist = np.cumsum(hist)

    row_similarity = []
    for row in matrix:
        row_hist, _ = np.histogram(row, bins=bins, density=True)
        row_cum_hist = np.cumsum(row_hist)
        #difference = np.sum(np.abs(cum_hist - row_cum_hist))
        difference = np.sum(np.abs(hist - row_hist))
        row_similarity.append(difference)

    sorted_indices = np.argsort(row_similarity)

    # Rows most similar and most different from cumulative histogram
    most_different_rows = matrix[sorted_indices[-3:]]
    most_similar_rows = matrix[sorted_indices[:3]]

    # Rows at the middle of the range of similarity
    middle_index = len(sorted_indices) // 2
    middle_rows = matrix[sorted_indices[middle_index-1:middle_index+2]]
    return [[most_different_rows.flatten(), f"different {sorted_indices[-3:]}"], [most_similar_rows.flatten(), f"similar {[sorted_indices[:3]]}"], [middle_rows.flatten(), f"medium {sorted_indices[middle_index-1:middle_index+2]}"]]

def myHist(name, title, toHist, toHistName, nbins=False):
    plt.figure(name)
    min_val = np.min(toHist)
    max_val = np.max(toHist)

    toHist=toHist.flatten()
    totalOccurrences= len(toHist)
    if nbins is False:
        hist, bins = np.histogram(toHist, density=True)
    else:
        hist, bins = np.histogram(toHist, bins=nbins, density=True)
    
    mean = np.nanmean(toHist)
    sigma = (np.nanvar(toHist))**0.5

    plt.bar(bins[:-1], hist, width=np.diff(bins), alpha=0.5)
    plt.axvline(mean, color='black', linestyle='solid', linewidth=2)
    plt.grid(True)
    plt.axvline(mean+sigma, color='red', linestyle='dashed', linewidth=1)
    plt.axvline(mean-sigma, color='red', linestyle='dashed', linewidth=1)
    plt.title(f'{title}\n mean value = {ufloat(mean, sigma)}')
    plt.xlabel(toHistName)
    plt.ylabel(f'Density')
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    text_x = x_max + 0.00 * (x_max - x_min)
    text_y = y_max + 0.04 * (y_max - y_min)  
    plt.text(text_x, text_y,  f"Total occurrences: {totalOccurrences}", fontsize=7, color='black', ha='right', va='top')
    return mean, sigma




def myHistForOverlaps2(name, title, toHist, toHistName, N):
    plt.figure(name)
    min_val = np.min(toHist.flatten())
    max_val = np.max(toHist.flatten())

    def overlapsOccurrenceOverNumberOfConfigus(histogram, function_values):
        normalized_histogram = [histogram[i] / function_values[i] for i in range(len(histogram))]
        normalized_histogram /= np.sum(normalized_histogram)
        return normalized_histogram
    
    hist, bins = np.histogram(toHist, bins=np.arange(min_val, max_val + 2, 2))

    function_values = []
    [function_values.append(math.comb(N,(N+int(value))//2)) for value in bins]

    hist = overlapsOccurrenceOverNumberOfConfigus(hist, function_values)
    
    mean = np.nanmean(hist.flatten())
    sigma = (np.nanvar(hist.flatten()))**0.5


    plt.bar(bins[:-1], hist, width=np.diff(bins), alpha=1.)
    plt.axvline(mean, color='black', linestyle='solid', linewidth=2)
    plt.grid(True)
    plt.axvline(mean+sigma, color='red', linestyle='dashed', linewidth=1)
    plt.axvline(mean-sigma, color='red', linestyle='dashed', linewidth=1)
    plt.title(f'{title}\n mean value = {ufloat(mean, sigma)}')
    plt.xlabel(toHistName)
    plt.ylabel(f'Occurrences')

    rows = find_rows_similarity(toHist)

    for row, rowLabel in rows:
        min_val = np.min(row)
        max_val = np.max(row)
        hist, bins  = np.histogram(row, density=True,bins=np.arange(min_val, max_val + 2, 2))
        hist, bins = np.histogram(row, bins=np.arange(min_val, max_val + 2, 2),density=True)

        function_values = []
        [function_values.append(math.comb(N,(N+int(value))//2)) for value in bins]

        hist = overlapsOccurrenceOverNumberOfConfigus(hist, function_values)
        plt.bar(bins[:-1], hist, width=np.diff(bins), alpha=0.23, edgecolor='black',linewidth=0.5,label=rowLabel)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.yscale('log')
    return mean, sigma

def myHistForOverlaps_notLog(name, title, toHist, toHistName, N):
    plt.figure(name)
    min_val = np.min(toHist.flatten())
    max_val = np.max(toHist.flatten())


    hist, bins = np.histogram(toHist, bins=np.arange(min_val, max_val + 2, 2),density=True)
    
    # Normalize histogram
    mean = np.nanmean(toHist.flatten())
    sigma = (np.nanvar(toHist.flatten()))**0.5


    plt.bar(bins[:-1], hist, width=np.diff(bins), alpha=1.)
    plt.axvline(mean, color='black', linestyle='solid', linewidth=2)
    plt.grid(True)
    plt.axvline(mean+sigma, color='red', linestyle='dashed', linewidth=1)
    plt.axvline(mean-sigma, color='red', linestyle='dashed', linewidth=1)
    plt.title(f'{title}\n mean value = {ufloat(mean, sigma)}')
    plt.xlabel(toHistName)
    plt.ylabel(f'Occurrences')

    rows = find_rows_similarity(toHist)

    for row, rowLabel in rows:
        min_val = np.min(row)
        max_val = np.max(row)
        hist, bins  = np.histogram(row, density=True,bins=np.arange(min_val, max_val + 2, 2))

        plt.bar(bins[:-1], hist, width=np.diff(bins), alpha=0.23, edgecolor='black',linewidth=0.5,label=rowLabel)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')


    return mean, sigma