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

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

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