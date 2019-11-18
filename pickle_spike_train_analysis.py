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

class pickle_dic_loader(object):
    def __init__(self,filename):
        if type(filename) in (list,tuple):
            print("you have chosen to load and merge several pickle files into 1")
            file_list = []
            for file in filename:
                with (open(file, "rb")) as openfile:
                    while True:
                        try:
                            # objects.append(pickle.load(openfile))
                            a = pickle.load(openfile)
                        except EOFError:
                            break
                    file_list.append(a)
            self.original_data_mat = np.concatenate(file_list,axis=1)

        else:
            print("you have chosen to load a single pickle file")
            with (open(filename, "rb")) as openfile:
                while True:
                    try:
                        # objects.append(pickle.load(openfile))
                        self.original_data_mat = pickle.load(openfile)
                    except EOFError:
                        break
        self.neurons = self.original_data_mat['neurons']
        self.event_times = self.original_data_mat['event_times']
        self.psth_matrix = create_psth_matrix_for_pca(self.neurons, self.event_times, bad_elec_list=[])
        print("created matrix has {} neurons".format(self.psth_matrix.shape[1]))

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

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]