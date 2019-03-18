import yaml
import pickle
import numpy as np
import scipy as scp
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.mlab import PCA
from scipy.spatial import distance
from moran_lab.plotter import savitzky_golay
import os
from scipy.io import loadmat
from scipy.stats import ttest_ind as Ttest
from scipy.signal import butter, lfilter
from math import factorial
from moran_lab.plotter import adjust_ylim
from tqdm import tqdm
from scipy.stats import zscore

plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams["axes.edgecolor"] = "0.15"
plt.rcParams["axes.linewidth"]  = 1.25
plt.rcParams["savefig.edgecolor"] = "0.15"


""" **********************************************  """

class multi_lfp_with_events(object):
    def __init__(self,yaml_file):
        with open(yaml_file) as stream:
            self.data_dic = yaml.load(stream)

    def create_lfp_mat(self, fs=300, normalize=True):
        slice_len = fs * 3600 * self.data_dic['running_settings']['length_of_slice_to_take_in_hours']
        amount_of_files = len(self.data_dic['filenames'].keys())
        mat = np.zeros((amount_of_files, slice_len))

        for i, filename in enumerate(yaml_dic['filenames'].keys()):
            data = np.fromfile(filename, dtype=np.int16)
            start_index = self.data_dic['filenames'][filename]['session_start'][0] * 3600 * fs + \
                          self.data_dic['filenames'][filename]['session_start'][1] * 60 * fs
            print(start_index, slice_len)
            data = data[start_index:start_index + slice_len]
            if normalize:
                data = data / np.median(data[:1200])
            mat[i, :] = data
        return mat


def get_event_times_from_pickle(filename):
    with (open(filename, "rb")) as openfile:
        while True:
            try:
                data = pickle.load(openfile)
            except EOFError:
                break
    return data['event_times']['water'], data['event_times']['sugar']