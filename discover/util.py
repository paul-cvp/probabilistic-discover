import os
import pm4py
import csv
import pandas as pd
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter

import numpy as np
import matplotlib.pyplot as plt

from scipy import stats


def load_log(log_path):
    log = None
    if os.path.exists(log_path) and os.path.isfile(log_path):
        if str.endswith(log_path, '.csv'):
            # convert from csv to xes
            log_csv = pd.read_csv(log_path, sep=';', engine='c', quoting=csv.QUOTE_NONNUMERIC)
            log_csv = dataframe_utils.convert_timestamp_columns_in_df(log_csv)
            log_csv = log_csv.sort_values('time:timestamp')
            log = log_converter.apply(log_csv)

        elif str.endswith(log_path, '.xes'):
            # load using pm4py
            log = pm4py.read_xes(log_path)

        return log
    else:
        raise Exception(f'[x] Error, path not found! {log_path}')

def element_wise_difference_matrix(list1,list2):
    '''
    difference bound matrix (DBM)
    https://en.wikipedia.org/wiki/Difference_bound_matrix#DBMs
    :param list1:
    :param list2:
    :return:
    '''
    # using list comprehension to perform task in one line
    return [[ele2 - ele1 for ele1, ele2 in zip(sub1, sub2)]
           for sub1, sub2 in zip(list1, list2)]

def create_timing_box_plot(title,data,file,xmin=0,xmax=None,save=False):
    range = None
    if xmax:
        range = (xmin,xmax)
    if save:
        fig, ax = plt.subplots(figsize=(16, 2))
        ax.boxplot(data,vert=False,flierprops={'marker':'x', 'markerfacecolor':'r', 'markersize':1})
        ax.set_title(title)
        ax.set_xlabel('Duration (Days)')
        ax.set(xlim=range)
        plt.tight_layout()
        plt.savefig(file)
        plt.close()
    return get_boxplot_values(data)

def create_timing_box_plots(timings,folder,xmin=0,xmax=None):
    res = {}
    for (rule,e1,e2),data in timings.items():
        if len(data)>5: # too low statistics, needs at least 5 data points (or another limit).
            file_name = os.path.join(folder,f'{rule}-{e1}-{e2}_boxplot.jpg')
            #print(f'[i] Boxplot for: {rule}-{e1}-{e2} Data: {data}')
            res[(rule,e1,e2)] = create_timing_box_plot(f'{rule}, {e1} --> {e2}',data,file_name,xmin,xmax)
        else:
            res[(rule, e1, e2)] = None
    return res

def get_mean_values(timings):
    res = {}
    for (rule,e1,e2),data in timings.items():
        if len(data)>5: # too low statistics, needs at least 5 data points (or another limit).
            res[(rule,e1,e2)] = np.mean(data)
        else:
            res[(rule, e1, e2)] = None
    return res

def create_histogram_plot(title,data,file,xmin=0,xmax=None,save=True):
    Nbins, _ = freedman_diaconis_rule(data)
    range = None
    if xmax:
        range = (xmin,xmax)
    counts, bin_edges = np.histogram(data,bins=Nbins,range=range)
    # take only non empty bins, that's why counts>0
    x = (bin_edges[1:][counts>0] + bin_edges[:-1][counts>0])/2
    y = counts[counts>0]
    sy = np.sqrt(counts[counts>0])   # NOTE: We (naturally) assume that the bin count is Poisson distributed.
    if save:
        fig, ax = plt.subplots(figsize=(16,2))
        ax.hist(data, bins=Nbins, range=range, histtype='step', density=False, alpha=1, color='g',
                label='Binned Duration Data')
        ax.errorbar(x, y, yerr=sy, xerr=0.0, label='Data, with Poisson errors', fmt='.k',  ecolor='k', elinewidth=1, capsize=1, capthick=1)
        ax.set_xlabel('Duration (Days)')
        ax.set_ylabel('Binned count')
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        plt.savefig(file)
        plt.close()
    return (counts, bin_edges)

def create_histograms(timings,folder,xmin=0,xmax=None):
    res = {}
    for (rule,e1,e2),data in timings.items():
        if len(data)>5: # too low statistics, needs at least 5 data points (or another limit).
            file_name = os.path.join(folder,f'{rule}-{e1}-{e2}_hist.jpg')
            res[(rule,e1,e2)] = create_histogram_plot(f'{rule}, {e1} --> {e2}',data,file_name,xmin,xmax,save=True)
        else:
            res[(rule, e1, e2)] = None
    return res

def get_boxplot_values(data):
    median = np.median(data)
    upper_quartile = np.percentile(data, 75)
    lower_quartile = np.percentile(data, 25)

    iqr = upper_quartile - lower_quartile
    upper_whisker = data[data<=upper_quartile+1.5*iqr].max()
    lower_whisker = data[data>=lower_quartile-1.5*iqr].min()
    min = data.min()
    max = data.max()
    mean = np.mean(data)
    return (lower_whisker,lower_quartile,median,upper_quartile,upper_whisker,iqr,min,max,mean)

def freedman_diaconis_rule(data):
    """rule to find the bin width and number of bins from data"""
    if (stats.iqr(data)>0):
        bin_width = 2*stats.iqr(data) / len(data)**(1/3)
        Nbins = int(np.ceil((data.max()-data.min())/bin_width))
        return Nbins, bin_width
    else:
        return 100, 0