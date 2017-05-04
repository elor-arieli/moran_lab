from spike_sort import get_data_from_mat_file
import numpy as np
import matplotlib.pyplot as plt
from plotter import plot_psth,plot_psth_with_rasters
from plotter import get_color_for_taste
from band_pass_filters import savitzky_golay
import os
import csv
from scipy import mean

def extract_data_from_mat_file(dir,file_name,laser_start,laser_stop,save_figure=False):
    if save_figure:
        os.chdir(dir)
        os.system('md ' + '"' + file_name[:-4] + '"')
    file = dir + '\\' + file_name
    data_matrix = get_data_from_mat_file(file)
    neurons = get_neurons(data_matrix)
    taste_events = get_taste_events(data_matrix)
    laser_times = get_laser_times(data_matrix)
    taste_events_with_laser_data = seperate_tastes_with_without_laser(taste_events,laser_times,laser_start)

    i = 1
    for neuron in neurons:
        print 'neuron %s out of %s' % (i,len(neurons))
        fig = plot_psth_with_rasters_with_without_laser_all_tastes_together(neuron[1],taste_events_with_laser_data,laser_start,laser_stop,['sugar','0.1M nacl','CA','water'])
        fig.suptitle(neuron[0], fontsize=22)
        if save_figure:
            filename = dir + '\\' + file_name[:-4] + '\\' + neuron[0] + '.eps'
            plt.savefig(filename, format='eps', dpi=1000)
        else:
            plt.show()
        i += 1

def plot_psth_with_rasters_with_without_laser(spike_train,event_dic_with_laser_data,laser_start,laser_stop,taste_list,bin_width=0.05,start_time=-2,end_time=5,overlap=0,normalize='Hz'):
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

    fig = plt.figure(figsize=(20, 15), dpi=1000)
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

    ax3 = fig.add_subplot(221)
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

    ax4 = fig.add_subplot(222)
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

def plot_psth_with_rasters_with_without_laser_all_tastes_together(spike_train,event_dic_with_laser_data,laser_start,laser_stop,taste_list,bin_width=0.05,start_time=-1,end_time=0,overlap=0,normalize='Hz'):
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

    fig = plt.figure(figsize=(20, 15), dpi=1000)
    ############ PSTHs ##############

    ax3 = fig.add_subplot(111)
    bin_amount = (end_time-start_time)//bin_width
    spikes = []
    total_events_laser = 0
    total_events_no_laser = 0
    for taste in taste_list:
        # get the spike times that are in the range of start-stop from each event.
        spikes += [spike_train[i] - event for i in range(len(spike_train)) for event in event_dic_with_laser_data['without laser'][taste] if start_time < spike_train[i] - event < end_time]
        total_events_no_laser += len(event_dic_with_laser_data['without laser'][taste])
    hist1,bin_edges = np.histogram(spikes,bin_amount,(start_time,end_time))
    average_spikes_in_bin = hist1 / float(total_events_no_laser)
    if normalize == 'Hz':
        spikes_in_bin = average_spikes_in_bin / bin_width
    norm_curve = savitzky_golay(spikes_in_bin,9,3)
    ax3.plot(bin_edges[:-1],norm_curve,label='Normal',color='k')
    ax3.set_xlabel('time')
    if normalize == 'Hz':
        ax3.set_ylabel('Fire rate - spikes / s (Hz)')
    else:
        ax3.set_ylabel('spikes')
    # ax3.set_label('Normal')
    # ax3.axvline(0, linestyle='--', color='k') # vertical lines
    # ax3.axvline(laser_start, linestyle='--', color='r')  # vertical lines
    # ax3.axvline(laser_stop, linestyle='--', color='r')  # vertical lines

    # ax4 = fig.add_subplot(222)
    # bin_amount = (end_time - start_time) // bin_width
    spikes2 = []
    for taste in taste_list:
        # get the spike times that are in the range of start-stop from each event.
        spikes2 += [spike_train[i] - event for i in range(len(spike_train)) for event in event_dic_with_laser_data['with laser'][taste] if start_time < spike_train[i] - event < end_time]
        total_events_laser += len(event_dic_with_laser_data['with laser'][taste])
    hist1, bin_edges = np.histogram(spikes2, bin_amount, (start_time, end_time))
    average_spikes_in_bin = hist1 / float(total_events_laser)
    if normalize == 'Hz':
        spikes_in_bin = average_spikes_in_bin / bin_width
    norm_curve = savitzky_golay(spikes_in_bin, 9, 3)
    ax3.plot(bin_edges[:-1], norm_curve, label='BLAx', color='r')
    # ax4.set_xlabel('time')
    # if normalize == 'Hz':
    #     ax4.set_ylabel('Fire rate - spikes / s (Hz)')
    # else:
    #     ax4.set_ylabel('spikes')
    # ax4.set_label('Blax')
    # ax4.axvline(0, linestyle='--', color='k')  # vertical lines
    # ax4.axvline(laser_start, linestyle='--', color='r')  # vertical lines
    # ax4.axvline(laser_stop, linestyle='--', color='r')  # vertical lines
    ax3.legend()
    # adjust_ylim(ax3,ax4)
    # plt.show()
    return fig

def get_taste_events(matrix):
    taste_events = {}
    taste_events['water'] = np.array(matrix['Event3'])
    taste_events['0.1M nacl'] = np.array(matrix['Event4'])
    # taste_events['1M nacl'] = np.array(matrix['Event7'])
    taste_events['CA'] = np.array(matrix['Event6'])
    taste_events['sugar'] = np.array(matrix['Event5'])
    # for key in taste_events:
    #     taste_events[key] = np.array([i for i in taste_events[key] if i > 1274])
    return taste_events

def get_laser_times(matrix):
    laser_times = {}
    laser_times['on'] = np.array(matrix['Event11'][:,0])
    laser_times['off'] = np.array(matrix['Event12'][:,0])
    laser_times['off'] = np.array([find_nearest(laser_times['off'],i) for i in laser_times['on']])
    return laser_times

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

def get_neurons(matrix):
    neurons = []
    for key in matrix.keys():
        if 'sig' in key:
            neurons.append((key,matrix[key][:,0]))
    return neurons

def seperate_tastes_with_without_laser(taste_events,laser_times,laser_start):
    tastes_with_without_laser = {}
    tastes_with_without_laser['with laser'] = {}
    tastes_with_without_laser['without laser'] = {}
    for taste in taste_events.keys():
        tastes_with_without_laser['with laser'][taste] = []
        tastes_with_without_laser['without laser'][taste] = []
        for event in taste_events[taste][:,0]:
            if laser_start - 1 < find_nearest(laser_times['on'],event) - event < laser_start + 1:
                tastes_with_without_laser['with laser'][taste].append(event)
            else:
                tastes_with_without_laser['without laser'][taste].append(event)
    # for key1 in tastes_with_without_laser.keys():
    #     print key1
    #     for key2 in tastes_with_without_laser[key1].keys():
    #         print '%s: %s' % (key2, len(tastes_with_without_laser[key1][key2]))
    return tastes_with_without_laser

def adjust_ylim(ax1,ax2):
    minylim = min(ax1.get_ylim()[0],ax2.get_ylim()[0])
    if minylim < 0:
        minylim = 0
    maxylim = max(ax1.get_ylim()[1],ax2.get_ylim()[1])
    ax1.set_ylim(minylim,maxylim)
    ax2.set_ylim(minylim,maxylim)
    return

extract_data_from_mat_file("D:\\data\\anan post data","AM196 D2 -1000 4000-sorted.mat",-1,4,True)
# a = get_data_from_mat_file("D:\\data\\anan post data\\optrode\\AM150 D1-sorted.mat")
# for key in a.keys():
#     if 'Event' in key:
#         print key,len(a[key])
# lasers = get_laser_times(a)
# print lasers['on']
# print lasers['off']
# print lasers['off'] - lasers['on']
# print get_taste_events(a)['0.1M nacl']

def check_base_line(dir):
    os.chdir(dir)
    pre = ['pre']
    post = ['post']
    ext = ['ext']
    for file in os.listdir('.'):
        if os.path.isfile(file):
            data_matrix = get_data_from_mat_file(file)
            neurons = get_neurons(data_matrix)
            taste_dic = get_taste_events(data_matrix)
            taste_events = []
            for key in taste_dic.keys():
                for i in taste_dic[key]:
                    taste_events.append(i)
            all_neuron_event_rates = []
            for neuron in neurons:
                all_event_rates = []
                for event in taste_events:
                    all_event_rates.append(len([i for i in neuron[1] if 0.0 < event - i <= 5.0])/5.0)
                    all_event_rates.append(len([i for i in neuron[1] if 0.0 < event - i <= 5.0])/5.0)
                all_neuron_event_rates.append(mean(all_event_rates))
            if 'pre' in file:
                pre.extend(all_neuron_event_rates)
            elif 'post' in file:
                post.extend(all_neuron_event_rates)
            elif 'ext' in file:
                ext.extend(all_neuron_event_rates)
    rows = zip(pre,post,ext)
    with open(file[:4]+'.csv', 'wb') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)


# check_base_line("D:\\data\\anan post baseline\\am18")
# check_base_line("D:\\data\\anan post baseline\\am36")
# check_base_line("D:\\data\\anan post baseline\\am42")
# check_base_line("D:\\data\\anan post baseline\\am47")
# check_base_line("D:\\data\\anan post baseline\\am49")