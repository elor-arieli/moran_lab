__author__ = 'elor'

from band_pass_filters import savitzky_golay
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import welch,spectrogram
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
import matplotlib.colors as colors
# import cv2
from matplotlib.mlab import PCA
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d


def plot_psth(spike_train,event_dic,bin_width=0.05,start_time=-1,end_time=4,overlap=0,normalize='Hz'):
    """
    spike_train is an array
    event dic is a dictionary of event times gotten from the get_all_events_from_directory function.
    bin width, start time and end time are integers in seconds.

    plots a smooth curve PSTH for every taste based on event times and correlation with spikes that happened within the start to end times.
    """

    assert end_time > start_time, 'start time cannot be bigger or equal to end time'
    assert normalize == 'Hz' or normalize == 'Average', 'normalize parameter has to be Average or Hz'
    # assert isinstance(spike_train,(list,np.array,np.ndarray)), 'spike train has to be a list of spike times.'
    assert isinstance(event_dic,dict), 'event dic needs to be a dictionary'

    bin_amount = (end_time-start_time)//bin_width
    plt.figure(1)
    for taste in event_dic:
        print ('collecting spike times for event of taste: %s') % taste
        spikes = [spike_train[i] - event for i in range(len(spike_train)) for event in event_dic[taste] if start_time < spike_train[i] - event < end_time]
        hist1,bin_edges = np.histogram(spikes,bin_amount,(start_time,end_time))
        average_spikes_in_bin = hist1 / float(len(event_dic[taste]))
        if normalize == 'Hz':
            spikes_in_bin = average_spikes_in_bin / bin_width
        norm_curve = savitzky_golay(spikes_in_bin,9,3)
        plt.plot(bin_edges[:-1],norm_curve,label=taste,color=get_color_for_taste(taste))
    plt.xlabel('time')
    if normalize == 'Hz':
        plt.ylabel('Fire rate - spikes / s (Hz)')
    else:
        plt.ylabel('spikes')
    plt.legend()
    plt.axvline(0, linestyle='--', color='k') # vertical lines
    plt.show()
    return

def plot_psth_with_rasters_with_without_laser(electrode,cluster,spike_train,event_dic_with_laser_data,laser_start,laser_stop,taste_list,bin_width=0.05,start_time=-2,end_time=5,overlap=0,normalize='Hz'):
    """
    plots a figure with 4 subplots, the top will be a raster with all tastes given in the list. the bottom will be a PSTH with the tastes given in the list.
    spike train is a list of spike times.
    event_dic is a dictionary with keys as tastes and values that are lists of taste times.
    start and end time are the boundaries in secs of the PSTH and raster.
    """
    assert isinstance(taste_list,list), 'tastes parameter need to be a list of the taste event file names'
    assert end_time > start_time, 'start time cannot be bigger or equal to end time'
    assert normalize == 'Hz' or normalize == 'Average', 'normalize parameter has to be Average or Hz'
    # assert isinstance(spike_train,(list,np.array,np.ndarray)), 'spike train has to be a list of spike times.'
    assert isinstance(event_dic_with_laser_data,dict), 'event dic needs to be a dictionary'

    ############ Rasters ##############

    fig = plt.figure(figsize=(20, 15))
    fig.suptitle('Electrode: {}, Cluster: {}'.format(electrode,cluster), fontsize=20)
    ax1 = fig.add_subplot(221)
    j = 0
    for taste in taste_list:
        # print 'collecting spike times for event of taste: %s' % taste
        event_times_list = []
        for event in event_dic_with_laser_data['without laser'][taste]:
            # get the spike times that are in the range of start-stop from each event.
            event_times_list.append([spike_train[i] - event for i in range(len(spike_train)) if start_time < spike_train[i] - event < end_time])
        for ith, trial in enumerate(event_times_list):
            ax1.vlines(trial, j + ith + .5, j + ith + 1.5, color=get_color_for_taste(taste))
        j += len(event_times_list)
    ax1.set_ylim(.5, len(event_times_list)*len(taste_list) + .5)
    ax1.set_xlabel('time')
    ax1.set_ylabel('trial')
    ax1.set_title('Normal')
    ax1.axvline(0, linestyle='--', color='k') # vertical lines
    ax1.axvline(laser_start, linestyle='--', color='r') # vertical lines
    ax1.axvline(laser_stop, linestyle='--', color='r') # vertical lines

    ax2 = fig.add_subplot(222)
    j = 0
    for taste in taste_list:
        # print 'collecting spike times for event of taste: %s' % taste
        event_times_list = []
        for event in event_dic_with_laser_data['with laser'][taste]:
            # get the spike times that are in the range of start-stop from each event.
            event_times_list.append([spike_train[i] - event for i in range(len(spike_train)) if start_time < spike_train[i] - event < end_time])
        for ith, trial in enumerate(event_times_list):
            ax2.vlines(trial, j + ith + .5, j + ith + 1.5, color=get_color_for_taste(taste))
        j += len(event_times_list)
    ax2.set_ylim(.5, len(event_times_list) * len(taste_list) + .5)
    ax2.set_xlabel('time')
    ax2.set_ylabel('trial')
    ax2.set_title('BLAx')
    ax2.axvline(0, linestyle='--', color='k')  # vertical lines
    ax2.axvline(laser_start, linestyle='--', color='r')  # vertical lines
    ax2.axvline(laser_stop, linestyle='--', color='r')  # vertical lines

    ############ PSTHs ##############

    ax3 = fig.add_subplot(223)
    bin_amount = (end_time-start_time)//bin_width
    for taste in taste_list:
        # get the spike times that are in the range of start-stop from each event.
        spikes = [spike_train[i] - event for i in range(len(spike_train)) for event in event_dic_with_laser_data['without laser'][taste] if start_time < spike_train[i] - event < end_time]
        hist1,bin_edges = np.histogram(spikes,bin_amount,(start_time,end_time))
        average_spikes_in_bin = hist1 / float(len(event_dic_with_laser_data['without laser'][taste]))
        if normalize == 'Hz':
            spikes_in_bin = average_spikes_in_bin / bin_width
        norm_curve = savitzky_golay(spikes_in_bin,9,3)
        ax3.plot(bin_edges[:-1],norm_curve,label=taste,color=get_color_for_taste(taste))
    ax3.set_xlabel('time')
    if normalize == 'Hz':
        ax3.set_ylabel('Fire rate - spikes / s (Hz)')
    else:
        ax3.set_ylabel('spikes')
    ax3.set_label('Normal')
    ax3.axvline(0, linestyle='--', color='k') # vertical lines
    ax3.axvline(laser_start, linestyle='--', color='r')  # vertical lines
    ax3.axvline(laser_stop, linestyle='--', color='r')  # vertical lines

    ax4 = fig.add_subplot(224)
    bin_amount = (end_time - start_time) // bin_width
    for taste in taste_list:
        # get the spike times that are in the range of start-stop from each event.
        spikes = [spike_train[i] - event for i in range(len(spike_train)) for event in
                  event_dic_with_laser_data['with laser'][taste] if start_time < spike_train[i] - event < end_time]
        hist1, bin_edges = np.histogram(spikes, bin_amount, (start_time, end_time))
        average_spikes_in_bin = hist1 / float(len(event_dic_with_laser_data['with laser'][taste]))
        if normalize == 'Hz':
            spikes_in_bin = average_spikes_in_bin / bin_width
        norm_curve = savitzky_golay(spikes_in_bin, 9, 3)
        ax4.plot(bin_edges[:-1], norm_curve, label=taste, color=get_color_for_taste(taste))
    ax4.set_xlabel('time')
    if normalize == 'Hz':
        ax4.set_ylabel('Fire rate - spikes / s (Hz)')
    else:
        ax4.set_ylabel('spikes')
    ax4.set_label('Blax')
    ax4.axvline(0, linestyle='--', color='k')  # vertical lines
    ax4.axvline(laser_start, linestyle='--', color='r')  # vertical lines
    ax4.axvline(laser_stop, linestyle='--', color='r')  # vertical lines
    ax4.legend()
    adjust_ylim(ax3,ax4)
    # plt.show()
    return fig


def raster(event_times,spike_train,start_time=-1,end_time=4,color='k'):
    """
    Creates a raster plot
    Parameters
    ----------
    event_times_list : iterable
                       a list of event time iterables
    color : string
            color of vlines
    Returns
    -------
    ax : an axis containing the raster plot
    """

    assert end_time > start_time, 'start time cannot be bigger or equal to end time'
    assert isinstance(spike_train,(list,np.array,np.ndarray)), 'spike train has to be a list of spike times.'
    assert isinstance(event_times,(list,np.array,np.ndarray)), "event times has to be a list of taste event times."

    event_times_list = []
    for event in event_times:
        print ('collecting spike times for events')
        event_times_list.append([spike_train[i] - event for i in range(len(spike_train)) if start_time < spike_train[i] - event < end_time])
    ax = plt.gca()
    for ith, trial in enumerate(event_times_list):
        plt.vlines(trial, ith + .5, ith + 1.5, color=color)
    plt.ylim(.5, len(event_times_list) + .5)
    plt.axvline(0, linestyle='--', color='k') # vertical lines
    return ax

def plot_psth_with_rasters(electrode, cluster,spike_train,event_dic,taste_list,bin_width=0.05,start_time=-1,end_time=4,overlap=0,normalize='Hz'):
    """
    plots a figure with 2 subplots, the top will be a raster with all tastes given in the list. the bottom will be a PSTH with the tastes given in the list.
    spike train is a list of spike times.
    event_dic is a dictionary with keys as tastes and values that are lists of taste times.
    start and end time are the boundaries in secs of the PSTH and raster.
    """

    assert isinstance(taste_list,list), 'tastes parameter need to be a list of the taste event file names'
    assert end_time > start_time, 'start time cannot be bigger or equal to end time'
    assert normalize == 'Hz' or normalize == 'Average', 'normalize parameter has to be Average or Hz'
    # assert isinstance(spike_train,(list,np.array,np.ndarray)), 'spike train has to be a list of spike times.'
    assert isinstance(event_dic,dict), 'event dic needs to be a dictionary'

    ############ Raster ##############

    fig = plt.figure(figsize=(20, 15))
    fig.suptitle('Electrode: {}, Cluster: {}'.format(electrode, cluster), fontsize=20)
    ax1 = fig.add_subplot(211)
    j = 0
    for taste in taste_list:
        print ('collecting spike times for events of taste: {}'.format(taste))
        event_times_list = []
        for event in event_dic[taste]:
            # get the spike times that are in the range of start-stop from each event.
            event_times_list.append([spike_train[i] - event for i in range(len(spike_train)) if start_time < spike_train[i] - event < end_time])
        for ith, trial in enumerate(event_times_list):
            ax1.vlines(trial, j + ith + .5, j + ith + 1.5, color=get_color_for_taste(taste))
        j += len(event_times_list)
    ax1.set_ylim(.5, len(event_times_list)*len(taste_list) + .5)
    ax1.set_xlabel('time')
    ax1.set_ylabel('trial')
    ax1.axvline(0, linestyle='--', color='k') # vertical lines

    ############ PSTH ##############

    ax2 = fig.add_subplot(212)
    bin_amount = (end_time-start_time)//bin_width
    for taste in taste_list:
        # get the spike times that are in the range of start-stop from each event.
        spikes = [spike_train[i] - event for i in range(len(spike_train)) for event in event_dic[taste] if start_time < spike_train[i] - event < end_time]
        hist1,bin_edges = np.histogram(spikes,bin_amount,(start_time,end_time))
        average_spikes_in_bin = hist1 / float(len(event_dic[taste]))
        if normalize == 'Hz':
            spikes_in_bin = average_spikes_in_bin / bin_width
        norm_curve = savitzky_golay(spikes_in_bin,9,3)
        ax2.plot(bin_edges[:-1],norm_curve,label=taste,color=get_color_for_taste(taste))
    ax2.set_xlabel('time')
    if normalize == 'Hz':
        ax2.set_ylabel('Fire rate - spikes / s (Hz)')
    else:
        ax2.set_ylabel('spikes')
    ax2.axvline(0, linestyle='--', color='k') # vertical lines
    ax2.legend()
    # plt.show()
    return fig

def plot_psth_with_raster_divide_by_trial(electrode, cluster, spike_train, trial_change, event_dic,taste_list,bin_width=0.05,start_time=-1,end_time=4,overlap=0,normalize='Hz'):
    """
    plots a figure with 2 subplots, the top will be a raster with all tastes given in the list. the bottom will be a PSTH with the tastes given in the list.
    spike train is a list of spike times.
    event_dic is a dictionary with keys as tastes and values that are lists of taste times.
    start and end time are the boundaries in secs of the PSTH and raster.
    """

    assert isinstance(taste_list,list), 'tastes parameter need to be a list of the taste event file names'
    assert end_time > start_time, 'start time cannot be bigger or equal to end time'
    assert normalize == 'Hz' or normalize == 'Average', 'normalize parameter has to be Average or Hz'
    # assert isinstance(spike_train,(list,np.array,np.ndarray)), 'spike train has to be a list of spike times.'
    assert isinstance(event_dic,dict), 'event dic needs to be a dictionary'

    ############ Raster ##############

    fig = plt.figure(1,(20,20))
    fig.suptitle('Electrode: {}, Cluster: {}'.format(electrode, cluster), fontsize=20)
    ax1 = fig.add_subplot(211)
    j = 0
    for taste in taste_list:
        print ('collecting spike times for events of taste: {}'.format(taste))
        event_times_list = []
        for event in event_dic[taste]:
            # get the spike times that are in the range of start-stop from each event.
            event_times_list.append([spike_train[i] - event for i in range(len(spike_train)) if start_time < spike_train[i] - event < end_time])
        for ith, trial in enumerate(event_times_list):
            ax1.vlines(trial, j + ith + .5, j + ith + 1.5, color=get_color_for_taste(taste))
        j += len(event_times_list)
    ax1.set_ylim(.5, len(event_times_list)*len(taste_list) + .5)
    ax1.set_xlabel('time')
    ax1.set_ylabel('trial')
    ax1.axvline(0, linestyle='--', color='k') # vertical lines

    ############ PSTH ##############

    ax2 = fig.add_subplot(212)
    bin_amount = (end_time-start_time)//bin_width
    for taste in taste_list:
        # get the spike times that are in the range of start-stop from each event.
        spikes = [spike_train[i] - event for i in range(len(spike_train)) for event in event_dic[taste][:trial_change] if start_time < spike_train[i] - event < end_time]
        hist1,bin_edges = np.histogram(spikes,bin_amount,(start_time,end_time))
        average_spikes_in_bin = hist1 / float(trial_change)
        if normalize == 'Hz':
            spikes_in_bin = average_spikes_in_bin / bin_width
        norm_curve = savitzky_golay(spikes_in_bin,9,3)
        ax2.plot(bin_edges[:-1],norm_curve,label=taste+' - first trials',color=get_color_for_taste(taste),linestyle='solid')

        spikes = [spike_train[i] - event for i in range(len(spike_train)) for event in event_dic[taste][trial_change:] if
                  start_time < spike_train[i] - event < end_time]
        hist1, bin_edges = np.histogram(spikes, bin_amount, (start_time, end_time))
        average_spikes_in_bin = hist1 / float(len(event_dic[taste])-trial_change)
        if normalize == 'Hz':
            spikes_in_bin = average_spikes_in_bin / bin_width
        norm_curve = savitzky_golay(spikes_in_bin, 9, 3)
        ax2.plot(bin_edges[:-1], norm_curve, label=taste+' - last trials', color=get_color_for_taste(taste),linestyle='dashed')
    ax2.set_xlabel('time')
    if normalize == 'Hz':
        ax2.set_ylabel('Fire rate - spikes / s (Hz)')
    else:
        ax2.set_ylabel('spikes')
    ax2.axvline(0, linestyle='--', color='k') # vertical lines
    ax2.legend()
    # plt.show()
    return fig

def plot_clustergram(fig, psth_response_matrix_dic):
    fig.suptitle('Clustergram of population responses to taste', fontsize=20)
    # Generate random features and distance matrix.
    # data_array_for_clustergram = pca_3D_data_array # this line for event based clustergram
    tastes_used = [i for i in psth_response_matrix_dic.keys()]
    data_array_for_clustergram = np.array([psth_response_matrix_dic[taste] for taste in
                                           tastes_used])  # this line for taste based clustergram
    num_of_events, num_of_neurons, amount_of_time_points = data_array_for_clustergram.shape
    D = np.zeros([num_of_events, num_of_events])
    for i in range(num_of_events):
        for j in range(num_of_events):
            distances = 0
            for k in range(num_of_neurons):
                distances += sum(
                    abs(np.array(data_array_for_clustergram[i, k, :]) - np.array(data_array_for_clustergram[j, k, :])))
            D[i, j] = distances
    max_val = D.max()
    D = D / max_val


    # taste_event_labels = np.array(['water']*25 + ['sugar']*27 + ['nacl']*27 + ['CA']*27) # this line for event based clustergram
    taste_event_labels = np.array(tastes_used)  # this line for taste based clustergram

    # Compute and plot first dendrogram.
    ax1 = fig.add_axes([0.76, 0.1, 0.2, 0.8])
    Y = sch.linkage(squareform(D), method='single')
    Z1 = sch.dendrogram(Y, orientation='right')
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Plot distance matrix.
    axmatrix = fig.add_axes([0.15, 0.1, 0.6, 0.8])
    idx1 = Z1['leaves'][::-1]
    D = D[idx1, :]
    taste_event_labels_y = taste_event_labels[idx1]
    im = axmatrix.matshow(D, aspect='auto', origin='upper', cmap='jet')
    axmatrix.set_xticks(range(num_of_events))
    axmatrix.set_xticklabels(taste_event_labels, minor=False)
    axmatrix.xaxis.set_label_position('bottom')
    axmatrix.xaxis.tick_bottom()

    # plt.xticks(rotation=-90, fontsize=8) # this line for event based clustergram
    # plt.yticks(fontsize=8) # this line for event based clustergram

    axmatrix.set_yticks(range(num_of_events))
    axmatrix.set_yticklabels(taste_event_labels_y, minor=False)
    axmatrix.yaxis.set_label_position('left')
    axmatrix.yaxis.tick_left()

    # Plot colorbar.
    axcolor = fig.add_axes([0.05, 0.1, 0.02, 0.8])
    axcolor.set_title('Distance')
    plt.colorbar(im, cax=axcolor)
    return fig

def get_color_for_taste(taste):
    color_dic = {'water':'blue', 'nacl':'orange', 'sugar':'green', 'CA':'red', 'quinine':'black', '0.1M nacl':'orange', '1M nacl': 'purple'}
    return color_dic[taste]

def adjust_ylim(ax1,ax2):
    minylim = min(ax1.get_ylim()[0],ax2.get_ylim()[0])
    if minylim < 0:
        minylim = 0
    maxylim = max(ax1.get_ylim()[1],ax2.get_ylim()[1])
    ax1.set_ylim(minylim,maxylim)
    ax2.set_ylim(minylim,maxylim)
    return