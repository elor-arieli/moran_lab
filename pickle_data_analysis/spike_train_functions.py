import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.mlab import PCA
from scipy.spatial import distance
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
import seaborn as sns
from scipy.signal import correlate,correlate2d
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from sklearn import manifold
from sklearn.metrics import euclidean_distances
# from sklearn.decomposition import PCA
from scipy.stats import zscore
from moran_lab.band_pass_filters import savitzky_golay

def calc_batch_times(taste_events):
    diffs = taste_events[1:] - taste_events[:-1]
    change_locs = np.array(np.where(diffs > 300)) + 1
    return taste_events[change_locs[0]]

def split_event_times_by_batches(event_times, batch_times):
    mat = []
    previous = 0
    for current in batch_times:
        this_batch_times = [i for i in event_times if previous < i < current]
        mat.append(this_batch_times)
        previous = current
    last_batch = [i for i in event_times if previous < i]
    mat.append(last_batch)
    return mat

def get_spikes_in_time_frame(ST,start,stop):
    return np.array([i for i in ST if start<i<stop])

def get_neuron_spikes_from_dic(neuron_dic,start_time=0,stop_time=43200,binned=True,binsize=1,session_start=False):
    if not binned:
        new_mat = []
        for neuron in neuron_dic:
            ST = neuron[2]
            if session_start:
                ST-=session_start
            new_mat.append(get_spikes_in_time_frame(ST,start_time,stop_time))
        return np.array(new_mat)

    else:
        bin_amount = int((stop_time-start_time)/binsize)
        new_mat = np.zeros((len(neuron_dic),bin_amount))
        for i,neuron in enumerate(neuron_dic):
            ST = neuron[2]
            if session_start:
                ST-=session_start
            true_ST = get_spikes_in_time_frame(ST,start_time,stop_time)
            new_mat[i,:] = np.histogram(true_ST,bin_amount,range=(start_time,stop_time))[0]
        return np.array(new_mat)

def calculate_Corr_over_time_of_BL_all_neurons(neuron_list,large_binsize=60, small_binsize=1,sig_start=0,sig_stop=43200):
    sig_len = sig_stop-sig_start
    neuron_mat = get_neuron_spikes_from_dic(neuron_list,sig_start,sig_stop,True,small_binsize)
    new_neuron_mat = neuron_mat.reshape(len(neuron_list),-1,large_binsize)
    num_neurons, num_times, _ = new_neuron_mat.shape
    full_corr_mat = []
    for time in tqdm(range(num_times)):
        specific_corr_vec = []
        for N1 in range(num_neurons):
            for N2 in range(N1+1, num_neurons):
                specific_corr_vec.append(np.corrcoef(new_neuron_mat[N1,time,:],new_neuron_mat[N2,time,:])[0,1])
        full_corr_mat.append(specific_corr_vec)
    return np.array(full_corr_mat).T

def CC_norm_by_shuffled_trials(N1,N2,ST_len_in_ms=3000,ms_to_take_each_way=500,times_to_shuffle=20):
    num_of_bins = len(N1[0])
    bin_len = ST_len_in_ms/num_of_bins
    bin_to_take_each_way = ms_to_take_each_way/bin_len

    corrs = []
    corrs_shuffled_trials = []
    for trial in range(len(N1)):
        corrs.append(np.correlate(N1[trial],N2[trial],mode='Full'))
    corrs = np.array(corrs)
    for i in range(times_to_shuffle):
        np.random.shuffle(N1)
        np.random.shuffle(N2)
        for trial in range(len(N1)):
            corrs_shuffled_trials.append(np.correlate(N1[trial],N2[trial],mode='Full'))
    corrs_shuffled_trials = np.array(corrs_shuffled_trials)

    mean_corrs = corrs.mean(axis=0)
    mean_shuffled_corrs = corrs_shuffled_trials.mean(axis=0)
#     print(num_of_bins-bin_to_take_each_way)
#     print(num_of_bins+bin_to_take_each_way)
    X = np.arange(bin_to_take_each_way*2)*bin_len-(bin_to_take_each_way*bin_len)
    return X,(mean_corrs/mean_shuffled_corrs)[int(num_of_bins-bin_to_take_each_way):int(num_of_bins+bin_to_take_each_way)]

def CC_norm_by_shuffled_trials_zscore(N1,N2,ST_len_in_ms=3000,ms_to_take_each_way=500,times_to_shuffle=20):
    num_of_bins = len(N1[0])
    bin_len = ST_len_in_ms/num_of_bins
    bin_to_take_each_way = ms_to_take_each_way/bin_len

    corrs = []
    corrs_shuffled_trials = []
    for trial in range(len(N1)):
        corrs.append(np.correlate(N1[trial],N2[trial],mode='Full'))
    corrs = np.array(corrs)
    for i in range(times_to_shuffle):
        np.random.shuffle(N1)
        np.random.shuffle(N2)
        for trial in range(len(N1)):
            corrs_shuffled_trials.append(np.correlate(N1[trial],N2[trial],mode='Full'))
    corrs_shuffled_trials = np.array(corrs_shuffled_trials)

    mean_corrs = corrs.mean(axis=0)
    mean_shuffled_corrs = corrs_shuffled_trials.mean(axis=0)
    std_shuffled_corrs = corrs_shuffled_trials.mean(axis=0)
#     print(num_of_bins-bin_to_take_each_way)
#     print(num_of_bins+bin_to_take_each_way)
    X = np.arange(bin_to_take_each_way*2)*bin_len-(bin_to_take_each_way*bin_len)
    return X,((mean_corrs-mean_shuffled_corrs)/std_shuffled_corrs)[int(num_of_bins-bin_to_take_each_way):int(num_of_bins+bin_to_take_each_way)]

def CC_norm_by_book(N1,N2,ST_len_in_ms=3000,ms_to_take_each_way=500):
    num_of_bins = len(N1[0])
    bin_len = ST_len_in_ms/num_of_bins
    bin_to_take_each_way = ms_to_take_each_way/bin_len

    corrs = []
#     corrs2 = []
    N1_psth = N1.mean(axis=0)
    N2_psth = N2.mean(axis=0)
    corr_psths = np.correlate(N1_psth,N2_psth,mode='Full')
    for trial in range(len(N1)):
        sum1 = sum(N1[trial])
        sum2 = sum(N2[trial])
        up = (sum1*sum2)/num_of_bins
        down = up**0.5
        if sum1 != 0 and sum2 != 0:
            corrs.append((np.correlate(N1[trial],N2[trial],mode='Full')-up)/down)
#         corrs2.append((np.correlate(N2[trial],N1[trial],mode='Full')-up)/down)
    corrs = np.array(corrs)
#     corrs2 = np.array(corrs2)

    mean_corrs = corrs.mean(axis=0)
#     mean_corrs2 = corrs2.mean(axis=0)
#     print(num_of_bins-bin_to_take_each_way)
#     print(num_of_bins+bin_to_take_each_way)
    X = np.arange(bin_to_take_each_way*2+1)*bin_len-(bin_to_take_each_way*bin_len)
    return X,(mean_corrs)[int(num_of_bins-bin_to_take_each_way-1):int(num_of_bins+bin_to_take_each_way)]

def CC_norm_by_psth(N1,N2,ST_len_in_ms=3000,ms_to_take_each_way=500):
    num_of_bins = len(N1[0])
    bin_len = ST_len_in_ms/num_of_bins
    bin_to_take_each_way = ms_to_take_each_way/bin_len

    corrs = []
#     corrs2 = []
    N1_psth = N1.mean(axis=0)
    N2_psth = N2.mean(axis=0)
    corr_psths = np.correlate(N1_psth,N2_psth,mode='Full')
    for trial in range(len(N1)):
        sum1 = sum(N1[trial])
        sum2 = sum(N2[trial])
        up = (sum1*sum2)/num_of_bins
        down = up**0.5
        if sum1 != 0 and sum2 != 0:
            corrs.append(np.correlate(N1[trial],N2[trial],mode='Full'))
#         corrs2.append((np.correlate(N2[trial],N1[trial],mode='Full')-up)/down)
    corrs = np.array(corrs)
#     corrs2 = np.array(corrs2)

    mean_corrs = corrs.mean(axis=0)
#     mean_corrs2 = corrs2.mean(axis=0)
#     print(num_of_bins-bin_to_take_each_way)
#     print(num_of_bins+bin_to_take_each_way)
    X = np.arange(bin_to_take_each_way*2+1)*bin_len-(bin_to_take_each_way*bin_len)
    return X,(mean_corrs-corr_psths)[int(num_of_bins-bin_to_take_each_way-1):int(num_of_bins+bin_to_take_each_way)]

def CC_norm_by_book_for_fast_spiking(N1,N2,ST_len_in_ms=3000,ms_to_take_each_way=500):
    num_of_bins = len(N1[0])
    bin_len = ST_len_in_ms/num_of_bins
    bin_to_take_each_way = ms_to_take_each_way/bin_len

    corrs = []
#     corrs2 = []
    N1_psth = N1.mean(axis=0)
    N2_psth = N2.mean(axis=0)
    corr_psths = np.correlate(N1_psth,N2_psth,mode='Full')
    for trial in range(len(N1)):
        sum1 = sum(N1[trial])
        sum2 = sum(N2[trial])
        up = (sum1*sum2)/num_of_bins
        down = ((sum1-sum1**2/num_of_bins)*(sum2-sum2**2/num_of_bins))**0.5
        if sum1 != 0 and sum2 != 0:
            corrs.append((np.correlate(N1[trial],N2[trial],mode='Full')-up)/down)
#         corrs2.append((np.correlate(N2[trial],N1[trial],mode='Full')-up)/down)
    corrs = np.array(corrs)
#     corrs2 = np.array(corrs2)

    mean_corrs = corrs.mean(axis=0)
#     mean_corrs2 = corrs2.mean(axis=0)
#     print(num_of_bins-bin_to_take_each_way)
#     print(num_of_bins+bin_to_take_each_way)
    X = np.arange(bin_to_take_each_way*2+1)*bin_len-(bin_to_take_each_way*bin_len)
    return X,(mean_corrs)[int(num_of_bins-bin_to_take_each_way-1):int(num_of_bins+bin_to_take_each_way)]

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]


def get_all_responses_in_time_frame(spike_train, event_list, choice_start_time, choice_end_time, psth_start_time, psth_stop_time, use_zscore=False):
    """
    :param spike_times: list of spike times
    :param event_list: list of taste event time
    :param choice_start_time: time in seconds from which to start looking at the events
    :param choice_end_time: time in seconds until which to look at the events
    :param psth_start_time: left edge of psth in seconds
    :param psth_stop_time: right edge of psth in seconds
    :param zscore: True for zscore normalization
    :return: a trial by time matrix
    """

    response_mat = []
    timed_event_list = [i for i in event_list if choice_start_time < i < choice_end_time]
    for event in timed_event_list:
        # get the spike times that are in the range of start-stop from each event.
        left_border = np.searchsorted(spike_train - event, psth_start_time)
        right_border = np.searchsorted(spike_train - event, psth_stop_time)
        spikes = spike_train[left_border:right_border] - event
        hist1, bin_edges = np.histogram(spikes, 100, (psth_start_time, psth_stop_time))
        spikes_in_bin = hist1 / 0.05
        norm_curve = savitzky_golay(spikes_in_bin, 9, 3)
        if use_zscore:
            response_mat.append(zscore(norm_curve))
        else:
            response_mat.append(norm_curve/np.median(norm_curve[:20]))
    return np.array(response_mat)


def create_psth_matrix_for_pca(all_neurons_spike_times, event_dic,bad_elec_list=[]):
    matrix_dic = []
    # taste_event_amount = {'water': 0, 'sugar': 0, 'nacl': 0, 'CA': 0}
    bin_amount = 5//0.05
    for taste in ('water','sugar','nacl','CA'):
        for event in event_dic[taste]:
            all_neural_responses_for_event = []
            # taste_event_amount[taste] += 1
            for neural_spike_times in all_neurons_spike_times:
                if neural_spike_times[0] not in bad_elec_list:
                    spikes = [neural_spike_times[2][i] - event for i in range(len(neural_spike_times[2])) if -1 < neural_spike_times[2][i] - event < 4]
                    hist1, bin_edges = np.histogram(spikes, int(bin_amount), (-1, 4))
                    spikes_in_bin = hist1 / 0.05
                    all_neural_responses_for_event.append(spikes_in_bin)
            matrix_dic.append(all_neural_responses_for_event)
    matrix_dic = np.array(matrix_dic)
    # print(taste_event_amount)
    return matrix_dic

def seperate_tastes_with_without_laser(taste_events, laser_times, laser_start):
    tastes_with_without_laser = {}
    tastes_with_without_laser['with laser'] = {}
    tastes_with_without_laser['without laser'] = {}
    for taste in taste_events.keys():
        print('taste:',taste)
        tastes_with_without_laser['with laser'][taste] = []
        tastes_with_without_laser['without laser'][taste] = []
        for event in taste_events[taste]:
            print(taste_events[taste])
            print('event time:',event,'nearest laser time:',find_nearest(laser_times, event), 'difference is:',(find_nearest(laser_times, event) - event))
            if laser_start - 5 < find_nearest(laser_times, event) - event < laser_start + 5:
                tastes_with_without_laser['with laser'][taste].append(event)
                # print('classified as with laser')
            else:
                tastes_with_without_laser['without laser'][taste].append(event)
                # print('classified as without laser')
    # for key1 in tastes_with_without_laser.keys():
    #     print key1
    #     for key2 in tastes_with_without_laser[key1].keys():
    #         print '%s: %s' % (key2, len(tastes_with_without_laser[key1][key2]))
    return tastes_with_without_laser