__author__ = 'elor'

import numpy as np
from matplotlib import pyplot as plt
from klusta.kwik.model import KwikModel
import os
from band_pass_filters import butter_bandpass_filter
from moran_lab.plotter import plot_psth,raster,get_color_for_taste,plot_psth_with_rasters,plot_psth_with_rasters_with_without_laser,plot_psth_with_raster_divide_by_trial,plot_clustergram
from scipy.io import loadmat
import pickle
from scipy.stats import ttest_ind as Ttest
print('site-packages')
# from multi_klusta import get_params_from_file
# from neural_models import run_HMM


class model_sorter:
    """
    recieves a kwik file.
    builds the model from the kwik file.
    with fields for the spike times, spike times by cluster, all cluster and good clusters.
    """

    def __init__(self,file_location):
        self.file = file_location
        self.model = KwikModel(self.file)
        self.spike_times = self.model.spike_times
        self.spike_cluster = self.model.spike_clusters
        self.cluster_groups = self.model.cluster_groups
        self.good_clusters = self.get_good_clusters()
        self.spike_times_by_cluster = self.sort_by_cluster()
        self.waveform_mat_per_cluster = self.get_waveform()

    def get_waveform(self):
        waveform_per_cluster_dic = {}
        with open(self.file[:-5] + ".dat", "rb") as dat_file:
            for cluster in self.good_clusters:
                dat_file.seek(0)
                lst_uint = []
                spike_times = self.spike_times_by_cluster[cluster][::100]
                spike_times_in_indexs = (np.array(spike_times) * 60000).astype(np.int64)
                for i in spike_times_in_indexs:
                    dat_file.seek(i - 40)
                    bys = dat_file.read(80)
                    ints = np.frombuffer(bys, dtype=np.uint16)
                    #         ints = int.from_bytes(bys, byteorder='little')
                    new_ints = (ints - 32768) * 0.195
                    new_ints -= new_ints.mean()
                    lst_uint.append(new_ints)
                waveform_per_cluster_dic[cluster] = np.array(lst_uint)
        return waveform_per_cluster_dic


    def get_good_clusters(self):
        """
        returns a list of the good cluster from the model based on the manual selection in kwik-gui
        """
        good_clusters = []
        for key in self.cluster_groups:
            if self.cluster_groups[key] == 'good':
                good_clusters.append(key)
        return good_clusters

    def sort_by_cluster(self):
        """
        return a dictionary of the spikes times for every good cluster
        """
        spike_trains_by_cluster = {}
        for cluster in self.good_clusters:
            spike_trains_by_cluster[cluster] = [self.spike_times[i] for i in range(len(self.spike_times)) if self.spike_cluster[i] == cluster]
        return spike_trains_by_cluster

class All_electrodes():
    """
    creates one object that holds a field called electrodes which is a dictionary of electrode numbers, each one is the model for that electrode.

    for example All_electrodes.electrodes[1] will hold the model for electrode 1
    if you want to access the spike times of cluster 2 in electrode 7 the way to that is the following:
        All_electrodes.electrodes[7].spike_times_by_cluster[2]

    if you set get_events to True you should set event_list to be list of strings which correspond to taste files.
    what you will get is another field called event_times that holds the dictionary of event_times

    for example All_electrodes.event_times['water'] will hold an array of water event times.

    finally lets look at the final example - creating a PSTH:

    let's say you want to psth the spike_train of cluster 12 in electrode 21:
    plot_psth(All_electrodes.electrodes[21].spike_times_by_cluster[12],All_electrodes.event_times)
    """

    def __init__(self,mother_directory=None,base_file_name='amp-A-',start_val=0,stop_val=32,get_events=False,get_laser=False,event_list=['water','sugar','nacl','CA'],fs=30000,use_memmap=True):
        self.base_file_name = base_file_name
        if mother_directory == None:
            mother_directory = os.getcwd()
        self.mother_directory = mother_directory + '\\'
        self.electrodes = self.get_electrode_models_in_range(start_val,stop_val)
        if get_laser:
            self.laser_times = get_event_time(mother_directory + '\\laser.dat')
        else:
            self.laser_times = False
        self.event_times = False
        if get_events:
            if isinstance(event_list,(list,np.array,np.ndarray)):
                self.event_times = get_all_events_from_directory(self.mother_directory,event_list,fs=fs,use_memmap=use_memmap)
            else:
                print ('event_list should be a list of strings cooresponding to taste file names')
        print('model created, these are the the good cluster and their corresponding electrodes: {}'.format(self.show_all_good_clusters()))
        if self.event_times:
            print('you have chosen to get taste event times, plotting functions are available')
        else:
            print('you have chosen not to get taste event times, so plotting functions will not be available')

    def get_electrode_models_in_range(self,start_val,stop_val):
        """
        returns a dictionary where the key is the electrode number and the value is the kwik model.
        """

        if isinstance(start_val,list):
            file_list = [self.mother_directory + self.base_file_name + "{0:03}".format(
                i) + '\\' + self.base_file_name + "{0:03}".format(i) + '.kwik' for i in start_val]
        else:
            file_list = [self.mother_directory + self.base_file_name + "{0:03}".format(i) + '\\' + self.base_file_name + "{0:03}".format(i) + '.kwik' for i in range(start_val,stop_val)]
        d = {}
        print ('loading electrode data')
        for i in range(start_val,stop_val):
            if os.path.exists(file_list[i-start_val]) and os.path.exists(file_list[i-start_val][:-5] + '.dat'):
                d[i] = model_sorter(file_list[i-start_val])
                print('got data from electrode {}'.format(i))
            else:
                print('no file: {}'.format(file_list[i-start_val]))
        return d

    def PSTH(self,electrode,cluster,event_times='default',bin_width=0.05,start_time=-1,end_time=4,overlap=0,normalize='Hz'):
        """
        an envelope function that calls the PSTH function from the plotter module.
        """

        if event_times == 'default':
            assert self.event_times, 'no taste events were added to model, make sure that when you build the model you enter get_events as true.'
            event_times = self.event_times
        plot_psth(self.electrodes[electrode].spike_times_by_cluster[cluster],event_times,bin_width,start_time,end_time,overlap,normalize)
        return

    def save_as_pickle_dic(self,filename,only_taste_responsive=False):
        save_file = self.mother_directory + filename + '.pkl'
        full_dic = {}
        full_dic['format'] = 'event times are either false if no data, or a dictionary with tastes as keys and taste ' \
                             'times array as the value\n' \
                             'laser times are either false if no date, or a list with laser times\n' \
                             'neurons is an array of arrays where each array[0] is the electrode number array[1] ' \
                             'is the cluster, array[2] is an array of spike times and array[3] is the waveform mat with 60' \
                             'samples centered around each spike time'
        full_dic['neurons'] = []
        full_dic['event_times'] = self.event_times
        full_dic['laser_times'] = self.laser_times
        if only_taste_responsive:
            good_clusters = self.show_all_taste_responsive_clusters()
        else:
            good_clusters = self.show_all_good_clusters()
        for elec in good_clusters.keys():
            for cluster in good_clusters[elec]:
                lst = []
                lst.append(elec)
                lst.append(cluster)
                spike_times = np.array(self.electrodes[elec].spike_times_by_cluster[cluster])
                lst.append(spike_times)
                lst.append(self.electrodes[elec].waveform_mat_per_cluster[cluster])
                full_dic['neurons'].append(lst)
        with open(save_file, 'wb') as f:
            pickle.dump(full_dic, f, pickle.HIGHEST_PROTOCOL)

    def psth_with_raster_divide_by_trial(self, electrode, cluster, trial_change, tastes=['water','sugar','nacl','CA'],bin_width=0.05,start_time=-1,end_time=4,overlap=0,normalize='Hz'):
        assert self.event_times, 'no taste events were added to model, make sure that when you build the model you enter get_events as true.'
        assert self.electrodes[electrode].spike_times_by_cluster[
            cluster], 'make sure to enter correct electrode and cluster numbers, the good clusters are: %s' % self.show_all_good_clusters()
        fig = plot_psth_with_raster_divide_by_trial(electrode, cluster, self.electrodes[electrode].spike_times_by_cluster[cluster],trial_change,
                                     self.event_times, tastes, bin_width, start_time, end_time, overlap, normalize)
        return fig


    def raster(self,taste,electrode,cluster,start_time=-1,end_time=4):
        """
        an envelope function that calls the raster function from the plotter module.
        """

        assert self.event_times, 'no taste events were added to model, make sure that when you build the model you enter get_events as true.'
        assert self.electrodes[electrode].spike_times_by_cluster[cluster], 'make sure to enter correct electrode and cluster numbers, the good clusters are: %s' % self.show_all_good_clusters()

        event_times = self.event_times[taste]
        spike_train = self.electrodes[electrode].spike_times_by_cluster[cluster]
        plt.figure(1)
        color = get_color_for_taste(taste)
        ax = raster(event_times,spike_train,start_time,end_time,color)
        plt.xlabel('time')
        plt.ylabel('trial')
        plt.show()
        return

    def PSTH_with_rasters(self,electrode,cluster,tastes=['water','sugar','nacl','CA'],bin_width=0.05,start_time=-1,end_time=4,overlap=0,normalize='Hz'):
        """
        an envelope function that calls the PSTH with raster function from the plotter module.
        """

        assert self.event_times, 'no taste events were added to model, make sure that when you build the model you enter get_events as true.'
        assert self.electrodes[electrode].spike_times_by_cluster[cluster], 'make sure to enter correct electrode and cluster numbers, the good clusters are: %s' % self.show_all_good_clusters()
        fig = plot_psth_with_rasters(electrode,cluster,self.electrodes[electrode].spike_times_by_cluster[cluster],self.event_times,tastes,bin_width,start_time,end_time,overlap,normalize)
        return fig

    def show_all_good_clusters(self):
        """
        collects and return a dictionary with all the electrodes that have clusters marked as good as keys and a list of the good clusters as values.
        """
        G_cluster_dic = {}
        for key in self.electrodes:
            if self.electrodes[key].good_clusters != []:
                G_cluster_dic[key] = self.electrodes[key].good_clusters
        return G_cluster_dic

    def show_all_taste_responsive_clusters(self,threshold=0.05):
        responsive_cluster_dic = {}
        G_clus = self.show_all_good_clusters()
        for elec in G_clus.keys():
            for cluster in G_clus[elec]:
                print('elec: {}, cluster: {}'.format(elec,cluster))
                if check_taste_responsiveness_for_all_tastes(self.electrodes[elec].spike_times_by_cluster[cluster],self.event_times, threshold)[0]:
                    if elec in responsive_cluster_dic.keys():
                        responsive_cluster_dic[elec].append(cluster)
                    else:
                        responsive_cluster_dic[elec] = [cluster]
        return responsive_cluster_dic

    def get_trace(self,electrode,start_time,stop_time,high_pass=None,sampling_rate=30000.,plot=False):
        """
        returns a smoothed (with butter filter) trace between the times given.
        """
        start_index,stop_index = int(start_time*sampling_rate),int(stop_time*sampling_rate)
        file_name = self.mother_directory + self.base_file_name + "{0:03}".format(electrode) + '\\' + self.base_file_name + "{0:03}".format(electrode) + ".dat"
        full_trace = np.fromfile(file_name,dtype='int16')
        trace = full_trace[start_index:stop_index]
        if high_pass is not None:
            trace = butter_bandpass_filter(trace, float(high_pass), 8000., sampling_rate, order=6)
        if plot:
            plt.figure(1)
            plt.plot(trace)
            plt.show()
        return trace

    def HMM(self,taste,start_time, end_time, bin_width=0.1):
        """
        creates a matrix A such that A[x][y][z]:
            x = trial
            y = neuron
            y = neuron
            z = bin
        """
        bin_amount = (end_time - start_time) // bin_width
        cluster_dic = self.show_all_good_clusters()
        events = self.event_times[taste]
        matrix = []
        for event in events:
            neurons = []
            for key in cluster_dic:
                spike_train = self.electrodes[key].spike_times_by_cluster[cluster_dic[key]]
                spikes = [spike_train[i] - event for i in range(len(spike_train)) if start_time < spike_train[i] - event < end_time]
                hist1, bin_edges = np.histogram(spikes, bin_amount, (start_time, end_time))
                neurons.append(hist1)
            matrix.append(neurons)
        matrix = np.array(matrix)
        run_HMM(directory,index,matrix)

    def PSTH_raster_with_laser_data(self,electrode,cluster,taste_list,laser_start=0, laser_stop=0, bin_width=0.05,start_time=-1,end_time=4,overlap=0,normalize='Hz',save=False):
        if self.laser_times is not False and self.event_times is not False:
            tastes_with_without_laser = seperate_tastes_with_without_laser(self.event_times, self.laser_times,
                                                                           laser_start)
            fig = plot_psth_with_rasters_with_without_laser(electrode, cluster, self.electrodes[electrode].spike_times_by_cluster[cluster],
                                                            tastes_with_without_laser, laser_start, laser_stop,
                                                            taste_list, bin_width, start_time, end_time,
                                                            overlap,
                                                            normalize)
            if not save:
                plt.show()
            else:
                fig.savefig(self.mother_directory + 'psth with laser {}-{}.jpg'.format(electrode,cluster))
            return fig

    def show_all_psth(self,tastes=['water','sugar','nacl','CA'],bin_width=0.05,start_time=-1,end_time=4,overlap=0,normalize='Hz',save=False):
        G_clus = self.show_all_good_clusters()
        for elec in G_clus.keys():
            for cluster in G_clus[elec]:
                fig = self.PSTH_with_rasters(elec,cluster,tastes,bin_width,start_time,end_time,overlap,normalize)
                if save:
                    fig.savefig(self.mother_directory + 'psth {}_{}'.format(elec,cluster))
                else:
                    plt.show()

    def plot_cluster_gram(self, start_time=-1, stop_time=4, save_name=False):
        PSTH_matrix = create_psth_matrix(self.electrodes, self.event_times, start_time, stop_time)
        fig = plt.figure(figsize=(20, 15))
        fig = plot_clustergram(fig, PSTH_matrix)
        if save_name == False:
            plt.show()
        else:
            fig.savefig(self.mother_directory + save_name + '.jpg')

def create_psth_matrix(all_neurons_spike_times, event_dic, start_time=-1, stop_time=4):
    matrix_dic = {}
    bin_amount = (stop_time-start_time)//0.05
    for taste in event_dic.keys():
        matrix_dic[taste] = []
        for elec in all_neurons_spike_times.keys():
            for cluster in all_neurons_spike_times[elec].spike_times_by_cluster.keys():
                spikes = [all_neurons_spike_times[elec].spike_times_by_cluster[cluster][i] - event for i in range(len(all_neurons_spike_times[elec].spike_times_by_cluster[cluster])) for event in event_dic[taste] if start_time < all_neurons_spike_times[elec].spike_times_by_cluster[cluster][i] - event < stop_time]
                hist1, bin_edges = np.histogram(spikes, int(bin_amount), (start_time, stop_time))
                average_spikes_in_bin = hist1 / len(event_dic[taste])
                spikes_in_bin = average_spikes_in_bin / 0.05
                matrix_dic[taste].append(spikes_in_bin)
        matrix_dic[taste] = np.asmatrix(matrix_dic[taste])
    return matrix_dic

def seperate_tastes_with_without_laser(taste_events, laser_times, laser_start):
    tastes_with_without_laser = {}
    tastes_with_without_laser['with laser'] = {}
    tastes_with_without_laser['without laser'] = {}
    for taste in taste_events.keys():
        # print('taste:',taste)
        tastes_with_without_laser['with laser'][taste] = []
        tastes_with_without_laser['without laser'][taste] = []
        for event in taste_events[taste]:
            # print('event time:',event,'nearest laser time:',find_nearest(laser_times, event), 'difference is:',(find_nearest(laser_times, event) - event))
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

def get_event_time(event_file,sampling_rate=30000,use_memmap=True):
    """
    extracts event times from a .dat file of a digital input signal
    returns an array of event times in seconds
    """

    if use_memmap:
        event_array = np.memmap(event_file,dtype=np.uint16,mode='r')
    else:
        event_array = np.fromfile(event_file, dtype=np.uint16)
    event_indexs = np.where(event_array[1:]-event_array[:-1] == 1)
    event_times = event_indexs[0] / float(sampling_rate)
    return event_times

def get_all_events_from_directory(directory_path=None,event_list=['water','sugar','nacl','CA'],fs=30000,use_memmap=True):
    """
    gets all events times from the list given.
    returns a dictionary with the tastes as keys and array of event times in seconds as value
    e.x. {'water': [1,2,3], 'sugar: [3.45,17.687]}
    """

    if directory_path == None:
        directory_path = os.getcwd()
    directory_path = directory_path + '\\'
    event_dic = {}
    print (event_list,directory_path)
    for taste in event_list:
        print (taste)
        taste_file = directory_path + taste + ".dat"
        event_dic[taste] = get_event_time(taste_file,sampling_rate=fs,use_memmap=use_memmap)
    return event_dic

def get_data_from_mat_file(file):
    matrix = loadmat(file)
    return matrix

def check_taste_responsiveness_for_taste(spike_train, event_times, threshold=0.05):
    base_line_FR = []
    event_FR = []
    for event in event_times:
        base_line_FR.append(len([spike_train[i] - event for i in range(len(spike_train)) if 0 < spike_train[i] - event < 2.5]))
        event_FR.append(len([spike_train[i] - event for i in range(len(spike_train)) if -2.5 < spike_train[i] - event < 0]))
    TV, PV = Ttest(base_line_FR,event_FR)
    print(PV)
    return (threshold > PV)

def check_taste_responsiveness_for_all_tastes(spike_train, event_times_dic, threshold=0.05):
    threshold = threshold/len(event_times_dic.keys())
    responsiveness_dic = {}
    for taste in event_times_dic.keys():
        print(taste)
        responsiveness_dic[taste] = check_taste_responsiveness_for_taste(spike_train,event_times_dic[taste],threshold)
    for taste in event_times_dic.keys():
        if responsiveness_dic[taste] == True:
            return (True,responsiveness_dic)
    return (False,responsiveness_dic)

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]