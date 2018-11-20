from scipy.signal import welch
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from scipy.signal import welch,spectrogram,lfilter,butter
import matplotlib.colors as colors
from tqdm import tqdm
import scipy
from scipy.stats import signaltonoise as calcSNR
from moran_lab.plotter import our_ts_plot

def spike_triggered_LFP(ax, spike_train, LFP_data, FS=300, start_time_in_secs=None, stop_time_in_secs=None, LFP_start=-0.5, LFP_stop=0.1, num_of_stds=3):

    if start_time_in_secs is None:
        start_time_in_secs = 0
        if LFP_start < 0:
            start_time_in_secs += np.abs(LFP_start)
    if stop_time_in_secs is None:
        stop_time_in_secs = spike_train[-1]
        if LFP_stop > 0:
            stop_time_in_secs -= LFP_stop

    left_border = np.searchsorted(spike_train, start_time_in_secs)
    right_border = np.searchsorted(spike_train, stop_time_in_secs)
#     print(len(spike_train))
    spike_train = spike_train[left_border:right_border]
#     print(len(spike_train))
    rows = len(spike_train)
    columns = int(FS*(LFP_stop-LFP_start))
    res_mat = np.zeros((rows,columns))

    start_index_fix = int(LFP_start*FS)
    stop_index_fix = int(LFP_stop*FS)
#     print(res_mat.shape,res_mat)

    for i, spike_time in enumerate(spike_train):
        j = int(spike_time*FS)

        res_mat[i,:] = LFP_data[j+start_index_fix:j+stop_index_fix]

#     print(res_mat.shape,res_mat)
    ax,y2,x,y1 = our_ts_plot(ax,res_mat,num_of_stds)
#     stds = np.std(res_mat,axis=0)/np.sqrt(res_mat.shape[0])
#     means = np.mean(res_mat,axis=0)
#     y1 = means+10*stds
#     y2 = means-10*stds
#     ax.plot(means,color='b')
#     ax.fill_between(np.arange(len(y1)),y1, y2, color='blue', alpha='0.2')
    ax.vlines(len(y1)-stop_index_fix,min(y2),max(y1),color='black',linestyle='dashed',lw=2)
    amount_of_ticks = len(ax.xaxis.get_major_ticks())
    my_ticks = np.linspace(LFP_start, LFP_stop, amount_of_ticks)
    ax.set_xticklabels(["{0:.2f}".format(i) for i in my_ticks])
#     ax.plot(means+stds*3,'r')
#     ax.plot(means-stds*3,'r')

    return ax,res_mat,y2,x,y1