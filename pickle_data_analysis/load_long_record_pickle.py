import pickle
import numpy as np
import scipy as scp
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.mlab import PCA
from scipy.spatial import distance
import os
from moran_lab.band_pass_filters import butter_bandpass
from scipy.io import loadmat
from scipy.stats import ttest_ind as Ttest
from scipy.signal import butter, lfilter, filtfilt, decimate
from math import factorial
from moran_lab.plotter import adjust_ylim,plot_psth_with_rasters_for_axes,our_ts_plot
from moran_lab.pickle_data_analysis.lfp_functions import spike_triggered_LFP,plot_average_spectogram_in_time_slice,average_response_spectogram
from moran_lab.pickle_data_analysis.lfp_functions import spectrum_power_over_time as single_band_PSD_over_time
import seaborn as sns

plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams["axes.edgecolor"] = "0.15"
plt.rcParams["axes.linewidth"]  = 1.25
plt.rcParams["savefig.edgecolor"] = "0.15"

class pickle_loader(object):
    """
    session start time needs to be in (hours,minute) format from the start of recording
    """
    def __init__(self, filename, session_start_time, lfp_data_file=False, lfp_fs=300):
        with (open(filename, "rb")) as openfile:
            while True:
                try:
                    self.data = pickle.load(openfile)
                except EOFError:
                    break
        self.lfp_data = False
        if lfp_data_file:
            self.lfp_data = np.fromfile(lfp_data_file,dtype=np.int16)
        self.get_all_event_and_batch_times(session_start_time)
        self.lfp_FS = lfp_fs
        self.PCA_data_array = create_psth_matrix_for_long_rec(self.data["neurons"],self.data["event_times"])
        self.event_times_by_batchs = {'water': split_event_times_by_batches(self.data['event_times']['water'],
                                                                            self.event_times_in_secs['water batch times']),
                                      'sacc': split_event_times_by_batches(self.data['event_times']['sugar'],
                                                                           self.event_times_in_secs['sacc batch times'])}


    # def compare_average_spectogram_in_time_slice(ax, lfp_data, event_times, start_time, stop_time, fs=300, filtered=True, sigma=3):

    def plot_LFP_corr_heatmaps_for_critical_times(self, average_every_x_minutes = 1, smooth = True,
                                                  normalize = True, use_zscore = True, save=True, top_title=''):
        if type(self.lfp_data) not in (np.ndarray,np.array,list):
            print("no LFP data file was loaded, cannot slice")
            return

        fig = plt.figure(figsize=(20, 15), dpi=1000)
        fig.clf()
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        names = ['correlation mat during drinking', 'correlation mat drinking => CTA', 'correlation mat CTA +0h => +3h',
                 'correlation mat CTA +3h => +6h']

        start_times = [0,20,80,260]
        stop_times = [20,80,260,440]

        for i, ax in enumerate([ax1, ax2, ax3, ax4]):
            ax = self.plot_corr_heatmap_for_slice(ax=ax, start_time=start_times[i], stop_time=stop_times[i],
                                                  count_from_drinking_session=True, time_units="minutes",
                                                  average_every_x_minutes=average_every_x_minutes, smooth=smooth,
                                                  normalize=normalize, use_zscore=use_zscore, title=names[i])

        fig.suptitle(top_title,fontsize=22)

        if save:
            stringy = "correlation between bands across stages of learning"
            if top_title != '':
                stringy = top_title + " " + stringy
            fig.savefig(stringy+'.jpeg', format='jpeg')
            fig.savefig(stringy+'.svg', format='svg')
        else:
            plt.show()
        return

    def plot_corr_heatmap_for_slice(self, ax, start_time=0, stop_time=440, count_from_drinking_session=True,
                                    time_units="minutes", average_every_x_minutes = 1, smooth = True,
                                    normalize = True, use_zscore = True, title=""):
        if type(self.lfp_data) not in (np.ndarray, np.array, list):
            print("no LFP data file was loaded, cannot slice")
            return

        data_mat,band_list = self.get_correlations_for_lfp_slice(start_time=start_time, stop_time=stop_time,
                                                       count_from_drinking_session=count_from_drinking_session,
                                                       time_units=time_units, average_every_x_minutes = average_every_x_minutes,
                                                       smooth = smooth, normalize = normalize, use_zscore = use_zscore)


        sns.heatmap(data_mat, annot=True, ax=ax, annot_kws={"size": 12})

        # bla = sns.heatmap(corr_coef_during_drinking, annot=True)
        # cbar = plt.colorbar(bla
        ax.set_xticks(np.arange(5) + 0.5)
        ax.set_xticklabels(['delta', 'theta', 'alpha', 'beta', 'gamma'], fontsize=14)
        ax.set_yticks(np.arange(5) + 0.5)
        ax.set_yticklabels(['delta', 'theta', 'alpha', 'beta', 'gamma'][::-1], rotation=0, fontsize=14)
        #     ax.xaxis.xticks(np.arange(5)+0.5,['delta','theta','alpha','beta','gamma'])
        #     ax.xaxis.yticks(np.arange(5)+0.5,['delta','theta','alpha','beta','gamma'][::-1],rotation=0)
        ax.set_title(title, fontsize=18)

        return ax

    def get_correlations_for_lfp_slice(self, start_time=0, stop_time=440, count_from_drinking_session=True,
                                       time_units="minutes", average_every_x_minutes = 1, smooth = True,
                                       normalize = True, use_zscore = True):

        if type(self.lfp_data) not in (np.ndarray, np.array, list):
            print("no LFP data file was loaded, cannot slice")
            return

        LFP_slice = self.get_lfp_slice(start_time=start_time, stop_time=stop_time,
                                       count_from_drinking_session=count_from_drinking_session,time_units=time_units)

        Delta = single_band_PSD_over_time(LFP_slice, fs = self.lfp_FS, average_every_x_minutes = average_every_x_minutes,
                                          band = 'Delta', smooth = smooth, normalize = normalize, use_zscore = use_zscore)

        Theta = single_band_PSD_over_time(LFP_slice, fs = self.lfp_FS, average_every_x_minutes = average_every_x_minutes,
                                          band = 'Theta', smooth = smooth, normalize = normalize, use_zscore = use_zscore)

        Alpha = single_band_PSD_over_time(LFP_slice, fs = self.lfp_FS, average_every_x_minutes = average_every_x_minutes,
                                          band = 'Alpha', smooth = smooth, normalize = normalize, use_zscore = use_zscore)

        Beta = single_band_PSD_over_time(LFP_slice, fs = self.lfp_FS, average_every_x_minutes = average_every_x_minutes,
                                         band = 'Beta', smooth = smooth, normalize = normalize, use_zscore = use_zscore)

        Gamma = single_band_PSD_over_time(LFP_slice, fs = self.lfp_FS, average_every_x_minutes = average_every_x_minutes,
                                          band = 'Gamma', smooth = smooth, normalize = normalize, use_zscore = use_zscore)

        return np.corrcoef([Delta,Theta,Alpha,Beta,Gamma]),["Delta","Theta","Alpha","Beta","Gamma"]


    def merge_neurons(self, elec, cluster1, cluster2):
        neuron1 = get_neuron_num_from_dic(self.data, elec, cluster1)
        neuron2 = get_neuron_num_from_dic(self.data, elec, cluster2)
        if neuron1 is not None and neuron2 is not None:
            self.data["neurons"][neuron1][2] = np.array(
                sorted(np.concatenate(self.data["neurons"][neuron1][2], self.data["neurons"][neuron2][2])))
            if len(self.data["neurons"][neuron1]) > 3:
                self.data["neurons"][neuron1][3] = np.array(
                    sorted(np.concatenate(self.data["neurons"][neuron1][3], self.data["neurons"][neuron2][3])))
            del self.data["neurons"][neuron2]
        return

    def get_lfp_slice(self,start_time=0,stop_time=440,count_from_drinking_session=True,time_units="minutes"):
        if type(self.lfp_data) not in (np.ndarray, np.array, list):
            print("no LFP data file was loaded, cannot slice")
            return

        units = {"minutes": 60, "hours": 3600, "seconds": 1}
        start_index = start_time * self.lfp_FS * units[time_units]
        stop_index = stop_time * self.lfp_FS * units[time_units]

        if count_from_drinking_session:
            added_indexs = self.event_times_in_secs['session start']*self.lfp_FS
            start_index += added_indexs
            stop_index += added_indexs

        return self.lfp_data[int(start_index):int(stop_index)]

    def remove_neurons(self,tuple_list):
        assert isinstance(tuple_list, list), "must recieve a list of (elec,cluster) tuples"
        for elec,clus in tuple_list:
            num = get_neuron_num_from_dic(self.data,elec,clus)
            if num is not None:
                del self.data["neurons"][num]
        return

    def check_and_display_all_neurons(self, average_every_x_minutes=5, save_figs=False):
        for neuron in self.data["neurons"]:
            elec, cluster = neuron[0], neuron[1]
            self.display_neuron(elec, cluster, average_every_x_minutes=average_every_x_minutes, save_fig=save_figs)
        return

    def display_neuron(self, elec, cluster, average_every_x_minutes=5, save_fig=False):

        # create plots
        print("creating plots")
        fig = plt.figure(num=1, figsize=(20, 12), dpi=1000)
        fig.clf()
        ax_upL = fig.add_subplot(231)
        ax_upM = fig.add_subplot(232)
        ax_upR = fig.add_subplot(233)
        ax_downL = fig.add_subplot(234)
        ax_downM = fig.add_subplot(235)
        ax_downR = fig.add_subplot(236)

        # get and set params
        print("getting and setting params")
        hours_per_point = average_every_x_minutes / 60.0
        bin_size_in_secs = 60 * average_every_x_minutes
        session_start = self.event_times_in_secs['session start'] / 3600
        batch_times_in_hours = self.event_times_in_secs['sacc batch times'] / 3600
        neuron_num = get_neuron_num_from_dic(self.data, elec, cluster)
        N0FR = self.data['neurons'][neuron_num][2]
        if len(self.data['neurons'][neuron_num]) > 3:
            waveforms = self.data['neurons'][neuron_num][3]
            amount_of_waveform = waveforms.shape[0]
            middle_waveform = int(waveforms.shape[0] / 2)

        CV_over_time = self.calc_CV_over_time_for_neuron(elec, cluster, average_ever_x_minutes=average_every_x_minutes)
        FF_over_time = self.calc_FF_over_time_for_neuron(elec, cluster, average_ever_x_minutes=average_every_x_minutes)
        smooth_CV = savitzky_golay(CV_over_time, 9, 3)
        smooth_FF = savitzky_golay(FF_over_time, 9, 3)

        # plot
        print("plotting PSTH and Raster")
        ax_upL, ax_downL = plot_psth_with_rasters_for_axes(ax_upL, ax_downL, N0FR, self.data["event_times"],
                                                           ['sugar', 'water'])

        print("plotting CV anf FF")
        ax_downM.plot(hours_per_point * np.arange(len(smooth_CV)), smooth_CV, label="CV",color='b')
        ax_downM.vlines(batch_times_in_hours, 0, 1, 'g', linestyles='dashed', linewidth=2, label='batch times')
        ax_downM.plot(hours_per_point * np.arange(len(smooth_FF)), smooth_FF, label="FF",color='r')

        if self.lfp_data is not False:
            ax_downR = spike_triggered_LFP(ax_downR, N0FR, self.lfp_data, FS=self.lfp_FS,
                                                   LFP_start=-0.5, LFP_stop=0.1, num_of_stds=5)[0]

        print("plotting BL firing rate and waveforms")
        if len(self.data['neurons'][neuron_num]) > 3:
            ax_upR = sns.tsplot(waveforms[:200], ax=ax_upR, color='b', lw=2)
            ax_upR = sns.tsplot(waveforms[middle_waveform - 100:middle_waveform + 100], ax=ax_upR, color='g', lw=2)
            ax_upR = sns.tsplot(waveforms[-200:], ax=ax_upR, color='r', lw=2)

        ax_upM = plot_BL_FR_analysis(ax_upM, N0FR, batch_times_in_hours=batch_times_in_hours,
                                     bin_size_in_secs=bin_size_in_secs, plot_batch_times=True,
                                     session_start=session_start)

        # set titles and stuff
        fig.suptitle(
            "neuron #{}, elec: {}, cluster: {}\n   amount of spikes: {}".format(neuron_num, elec, cluster, len(N0FR)),
            fontsize=18)
        ax_upL.set_title("Raster", fontsize=14)
        ax_upM.set_title("BL Firing Rate", fontsize=14)
        ax_upR.set_title("Waveforms", fontsize=14)
        ax_downL.set_title("PSTH", fontsize=14)
        ax_downM.set_title("CV and FF", fontsize=14)
        ax_downR.set_title("Spike Triggered LFP", fontsize=14)

        for ax in (ax_upM, ax_downM, ax_downR):
            ax.legend()
        ax_upR.legend(["first 200", "middle 200", "last 200"])

        if save_fig:
            fig.savefig('full showing of neuron {}-{}.jpeg'.format(elec, cluster), format='jpeg')
            fig.savefig('full showing of neuron {}-{}.svg'.format(elec, cluster), format='svg')
        else:
            plt.show()

    def show_spike_triggered_LFP_sig(self,elec,cluster,start_time_in_secs = None, stop_time_in_secs = None, LFP_start = -0.5, LFP_stop = 0.1,show=False,save_fig=False,num_of_stds=3):
        """
        :param elec: elec num of neuron
        :param cluster: cluster num of neuron
        :param start_time_in_secs: start of spiking window to look at, example: 22h and 15m = 22*3600+15*60
        :param stop_time_in_secs: end of spiking window to look at, example: 22h and 15m = 22*3600+15*60
        :param LFP_start: how many secs to look at LFP before spike time (should probably be -something, like -0.5 secs)
        :param LFP_stop: how many secs to look at LFP after spike time
        :param show: whether to show plot
        :param save_fig: whwther to save the fig
        :return: matrix of lfp data (rows = trials, coloumn = LFP data), bottom edge, mean plot, top edge.
        """
        if not self.lfp_data:
            print("cannot calculate, no LFP data present in model")
            return None

        fig = plt.figure(figsize=(20,15))
        ax = fig.add_subplot(1,1,1)

        num = get_neuron_num_from_dic(self.data, elec, cluster)
        spike_train = self.data["neurons"][num][2]

        ax, res_mat, y2, x, y1 = spike_triggered_LFP(ax, spike_train, self.lfp_data, FS = self.lfp_FS,
                                                     start_time_in_secs = start_time_in_secs,
                                                     stop_time_in_secs = stop_time_in_secs,
                                                     LFP_start = LFP_start, LFP_stop = LFP_stop, num_of_stds=num_of_stds)
        if save_fig:
            fig.savefig('spike triggered LFP for {}-{}.jpeg'.format(elec,cluster), format='jpeg')
            fig.savefig('spike triggered LFP for {}-{}.svg'.format(elec,cluster), format='svg')

        if show:
            plt.show()

        return res_mat,y2, x, y1

    def resave_as_new_pickle_dic(self, filename):
        save_file = self.mother_directory + filename + '.pkl'
        full_dic = self.data
        with open(save_file, 'wb') as f:
            pickle.dump(full_dic, f, pickle.HIGHEST_PROTOCOL)

    def get_all_event_and_batch_times(self,session_start_time):
        self.event_times_in_secs = {
            'session start': session_start_time[0]*3600+session_start_time[1]*60,
            'sacc batch times': calc_batch_times(self.data,'sugar'),
            'water batch times': calc_batch_times(self.data, 'water'),
        }

    def calc_CV_over_time_for_all_neurons(self, average_ever_x_minutes=5, pre_title=None, single_fig=False):
        all_CV_times = []
        fig = plt.figure()
        fig.clf()
        ax = fig.add_subplot(111)

        if single_fig:
            for neural_spike_times in self.data['neurons']:
                CV_over_time = []
                total_time = neural_spike_times[2][-1] - neural_spike_times[2][0]
                num_of_pieces = int(total_time / (average_ever_x_minutes * 60))
                pieces = np.linspace(neural_spike_times[2][0], neural_spike_times[2][-1], num_of_pieces)

                for i in range(len(pieces) - 1):
                    left_border = np.searchsorted(neural_spike_times[2], pieces[i])
                    right_border = np.searchsorted(neural_spike_times[2], pieces[i + 1])
                    spikes = neural_spike_times[2][left_border:right_border]
                    CV_over_time.append(calcCV(spikes))
                ax.plot(CV_over_time,label='{}-{}'.format(neural_spike_times[0],neural_spike_times[1]))
                all_CV_times.append(CV_over_time)
            ax.legend()
            if pre_title is not None:
                fig.savefig('{} CV over time for all neurons.jpeg'.format(pre_title), format='jpeg')
                fig.savefig('{} CV over time for all neurons.svg'.format(pre_title), format='svg')
            else:
                fig.savefig('CV over time for all neurons.jpeg', format='jpeg')
                fig.savefig('CV over time for all neurons.svg', format='svg')

        else:
            for neural_spike_times in self.data['neurons']:
                fig.clf()
                CV_over_time = []
                total_time = neural_spike_times[2][-1] - neural_spike_times[2][0]
                num_of_pieces = int(total_time / (average_ever_x_minutes * 60))
                pieces = np.linspace(neural_spike_times[2][0], neural_spike_times[2][-1], num_of_pieces)

                for i in range(len(pieces) - 1):
                    left_border = np.searchsorted(neural_spike_times[2], pieces[i])
                    right_border = np.searchsorted(neural_spike_times[2], pieces[i + 1])
                    spikes = neural_spike_times[2][left_border:right_border]
                    CV_over_time.append(calcCV(spikes))

                ax.plot(CV_over_time, label='{}-{}'.format(neural_spike_times[0], neural_spike_times[1]))
                all_CV_times.append(CV_over_time)
                ax.legend()

                if pre_title is not None:
                    fig.savefig('{} CV over time for neuron {}-{}.jpeg'.format(pre_title, neural_spike_times[0], neural_spike_times[1]), format='jpeg')
                    fig.savefig('{} CV over time for neuron {}-{}.svg'.format(pre_title, neural_spike_times[0], neural_spike_times[1]), format='svg')
                else:
                    fig.savefig('CV over time for neuron {}-{}.jpeg'.format(neural_spike_times[0], neural_spike_times[1]), format='jpeg')
                    fig.savefig('CV over time for neuron {}-{}.svg'.format(neural_spike_times[0], neural_spike_times[1]), format='svg')

        return all_CV_times

    def calc_CV_over_time_for_neuron(self, elec, cluster, average_ever_x_minutes=5,save=False, show=False, pre_title=None):
        CV_over_time = []
        for neural_spike_times in self.data['neurons']:
            if neural_spike_times[0] == elec and neural_spike_times[1] == cluster:
                total_time = neural_spike_times[2][-1] - neural_spike_times[2][0]
                num_of_pieces = int(total_time/(average_ever_x_minutes*60))
                pieces = np.linspace(neural_spike_times[2][0],neural_spike_times[2][-1],num_of_pieces)
                for i in range(len(pieces)-1):
                    left_border = np.searchsorted(neural_spike_times[2], pieces[i])
                    right_border = np.searchsorted(neural_spike_times[2], pieces[i+1])
                    spikes = neural_spike_times[2][left_border:right_border]
                    CV_over_time.append(calcCV(spikes))

        fig = plt.figure()
        fig.clf()
        ax = fig.add_subplot(111)
        ax.plot(CV_over_time)
        if save:
            if pre_title is not None:
                fig.savefig('{} CV over time for neuron {}-{}.jpeg'.format(pre_title,elec,cluster),format='jpeg')
                fig.savefig('{} CV over time for neuron {}-{}.svg'.format(pre_title,elec,cluster),format='svg')
            else:
                fig.savefig('CV over time for neuron {}-{}.jpeg'.format(elec,cluster),format='jpeg')
                fig.savefig('CV over time for neuron {}-{}.svg'.format(elec,cluster),format='svg')
        if show:
            plt.show()
        return CV_over_time

    def calc_FF_over_time_for_all_neurons(self, average_ever_x_minutes=5, pre_title=None, single_fig=False):
        all_FF_times = []
        fig = plt.figure()
        fig.clf()
        ax = fig.add_subplot(111)

        if single_fig:
            for neural_spike_times in self.data['neurons']:
                FF_over_time = []
                total_time = neural_spike_times[2][-1] - neural_spike_times[2][0]
                num_of_pieces = int(total_time / (average_ever_x_minutes * 60))
                pieces = np.linspace(neural_spike_times[2][0], neural_spike_times[2][-1], num_of_pieces)

                for i in range(len(pieces) - 1):
                    left_border = np.searchsorted(neural_spike_times[2], pieces[i])
                    right_border = np.searchsorted(neural_spike_times[2], pieces[i + 1])
                    spikes = neural_spike_times[2][left_border:right_border]
                    FF_over_time.append(calcFF(spikes,average_ever_x_minutes*12))
                ax.plot(FF_over_time, label='{}-{}'.format(neural_spike_times[0], neural_spike_times[1]))
                all_FF_times.append(FF_over_time)
            ax.legend()
            if pre_title is not None:
                fig.savefig('{} FF over time for all neurons.jpeg'.format(pre_title), format='jpeg')
                fig.savefig('{} FF over time for all neurons.svg'.format(pre_title), format='svg')
            else:
                fig.savefig('FF over time for all neurons.jpeg', format='jpeg')
                fig.savefig('FF over time for all neurons.svg', format='svg')

        else:
            for neural_spike_times in self.data['neurons']:
                fig.clf()
                FF_over_time = []
                total_time = neural_spike_times[2][-1] - neural_spike_times[2][0]
                num_of_pieces = int(total_time / (average_ever_x_minutes * 60))
                pieces = np.linspace(neural_spike_times[2][0], neural_spike_times[2][-1], num_of_pieces)

                for i in range(len(pieces) - 1):
                    left_border = np.searchsorted(neural_spike_times[2], pieces[i])
                    right_border = np.searchsorted(neural_spike_times[2], pieces[i + 1])
                    spikes = neural_spike_times[2][left_border:right_border]
                    FF_over_time.append(calcFF(spikes,average_ever_x_minutes*12))

                ax.plot(FF_over_time, label='{}-{}'.format(neural_spike_times[0], neural_spike_times[1]))
                all_FF_times.append(FF_over_time)
                ax.legend()

                if pre_title is not None:
                    fig.savefig('{} FF over time for neuron {}-{}.jpeg'.format(pre_title, neural_spike_times[0],
                                                                               neural_spike_times[1]), format='jpeg')
                    fig.savefig('{} FF over time for neuron {}-{}.svg'.format(pre_title, neural_spike_times[0],
                                                                              neural_spike_times[1]), format='svg')
                else:
                    fig.savefig(
                        'FF over time for neuron {}-{}.jpeg'.format(neural_spike_times[0], neural_spike_times[1]),
                        format='jpeg')
                    fig.savefig(
                        'FF over time for neuron {}-{}.svg'.format(neural_spike_times[0], neural_spike_times[1]),
                        format='svg')

        return all_FF_times

    def calc_FF_over_time_for_neuron(self, elec, cluster, average_ever_x_minutes=5, save=False, show=False, pre_title=None):
        FF_over_time = []
        for neural_spike_times in self.data['neurons']:
            if neural_spike_times[0] == elec and neural_spike_times[1] == cluster:
                total_time = neural_spike_times[2][-1] - neural_spike_times[2][0]
                num_of_pieces = int(total_time / (average_ever_x_minutes * 60))
                pieces = np.linspace(neural_spike_times[2][0], neural_spike_times[2][-1], num_of_pieces)
                for i in range(len(pieces) - 1):
                    left_border = np.searchsorted(neural_spike_times[2], pieces[i])
                    right_border = np.searchsorted(neural_spike_times[2], pieces[i + 1])
                    spikes = neural_spike_times[2][left_border:right_border]
                    FF_over_time.append(calcFF(spikes,average_ever_x_minutes*12))

        fig = plt.figure()
        fig.clf()
        ax = fig.add_subplot(111)
        ax.plot(FF_over_time)
        if save:
            if pre_title is not None:
                fig.savefig('{} FF over time for neuron {}-{}.jpeg'.format(pre_title, elec, cluster), format='jpeg')
                fig.savefig('{} FF over time for neuron {}-{}.svg'.format(pre_title, elec, cluster), format='svg')
            else:
                fig.savefig('FF over time for neuron {}-{}.jpeg'.format(elec, cluster), format='jpeg')
                fig.savefig('FF over time for neuron {}-{}.svg'.format(elec, cluster), format='svg')
        if show:
            plt.show()
        return FF_over_time

    def plot_lfp_power_over_time(self,average_every_x_minutes=5, bands_to_plot=['Delta','Theta','Alpha','Beta','Gamma','Fast Gamma'],
                                 smooth=True,save_fig=False):
        if self.lfp_data is False:
            print('no LFP data file')
            return
        fig = plt.figure()
        fig.clf()
        ax = fig.add_subplot(111)
        drinking_session_start = self.event_times_in_secs['session start']/3600
        batch_times_in_hours = self.event_times_in_secs['sacc batch times']/3600
        ax,power_over_time = spectrum_power_over_time(ax, self.lfp_data, batch_times_in_hours, fs=self.lfp_FS,
                                                      average_every_x_minutes=average_every_x_minutes, bands_to_plot=bands_to_plot,
                                                      smooth=smooth, drinking_session_start=drinking_session_start)
        if save_fig:
            fig.savefig('Power spectrum over time for bands: {}.jpeg'.format(str(bands_to_plot)[1:-1]), format='jpeg')
            fig.savefig('Power spectrum over time for bands: {}.svg'.format(str(bands_to_plot)[1:-1]), format='svg')
        else:
            plt.show()
        return power_over_time

    def plot_PCA_2D_for_batch_responses_dual_taste(self, elec_cluster, taste='both', title='',
                                                   normalize=False, save_fig=False):
        neuron_num = get_neuron_num_from_dic(self.data, elec_cluster[0], elec_cluster[1])
        fig = plt.figure()
        fig.clf()
        ax = fig.add_subplot(111)
        ax, distances, distances_2d = PCA_2D_for_batch_responses_dual_taste(ax, self.PCA_data_array, neuron_num,
                                                                                 taste=taste, title=title,normalize=normalize)
        if save_fig:
            fig.savefig('PCA over time for {} .jpeg'.format(taste), format='jpeg')
            fig.savefig('PCA over time for {} .svg'.format(taste), format='svg')
        else:
            plt.show()
        return distances, distances_2d

    def plot_and_save_BL_FR_for_all_neurons(self, bin_size_in_secs=300, plot_batch_times=True):
        for data in self.data['neurons']:
            fig = plt.figure(figsize=(20, 15), dpi=1000)
            fig.clf()
            ax = fig.add_subplot(111)
            N0FR = data[2]
            session_start = self.event_times_in_secs['session start'] / 3600

            batch_times_in_hours = self.event_times_in_secs['sacc batch times'] / 3600
            ax = plot_BL_FR_analysis(ax, N0FR, batch_times_in_hours, bin_size_in_secs=bin_size_in_secs,
                                     plot_batch_times=plot_batch_times, session_start=session_start)

            fig.savefig('BaseLine FR for {}-{}.jpeg'.format(data[0], data[1]), format='jpeg')
            fig.savefig('BaseLine FR for {}-{}.svg'.format(data[0], data[1]), format='svg')
        return

    def plot_PCA_2D_for_population_batch_responses(self, elec_cluster_list, taste='sacc', title='',normalize=False, save_fig=False):
        """ elec_cluster_list is a list of elec-cluster tuples"""
        neuron_num_list = []
        for elec_cluster in elec_cluster_list:
            neuron_num_list.append(get_neuron_num_from_dic(self.data, elec_cluster[0], elec_cluster[1]))
        fig = plt.figure()
        fig.clf()
        ax = fig.add_subplot(111)
        ax = PCA_2D_for_population_batch_responses(self, ax, self.PCA_data_array, neuron_num_list, taste=taste,
                                                        title=title, normalize=normalize)

        if save_fig:
            fig.savefig('population response PCA over batchs for {}.jpeg'.format(taste), format='jpeg')
            fig.savefig('population response PCA over batchs for {}.svg'.format(taste), format='svg')
        else:
            plt.show()
        return

    def psth_over_time_for_neuron(self, elec_cluster, taste='sugar', start_batch=0, stop_batch=30, save_fig=False):
        fig = plt.figure()
        fig.clf()
        ax = fig.add_subplot(111)
        neuron_num = get_neuron_num_from_dic(self.data, elec_cluster[0], elec_cluster[1])
        ax = plot_psth_over_time_for_neuron(ax, self.PCA_data_array, neuron_num, taste=taste,
                                            start_batch=start_batch,stop_batch=stop_batch)

        if save_fig:
            fig.savefig('PSTH over batchs for taste {}.jpeg'.format(taste), format='jpeg')
            fig.savefig('PSTH over batchs for taste {}.svg'.format(taste), format='svg')
        else:
            plt.show()
        return

    def BL_FR_analysis(self, elec_cluster, bin_size_in_secs=300, plot_batch_times=True, save_fig=False):
        fig = plt.figure()
        fig.clf()
        ax = fig.add_subplot(111)
        neuron_num = get_neuron_num_from_dic(self.data, elec_cluster[0], elec_cluster[1])
        N0FR = self.data['neurons'][neuron_num][2]
        session_start = self.event_times_in_secs['session start']/3600

        batch_times_in_hours = self.event_times_in_secs['sacc batch times']/3600
        ax = plot_BL_FR_analysis(ax, N0FR, batch_times_in_hours, bin_size_in_secs=bin_size_in_secs,
                                 plot_batch_times=plot_batch_times, session_start=session_start)

        if save_fig:
            fig.savefig('BaseLine FR for {}-{}.jpeg'.format(elec_cluster[0],elec_cluster[1]), format='jpeg')
            fig.savefig('BaseLine FR for {}-{}.svg'.format(elec_cluster[0],elec_cluster[1]), format='svg')
        else:
            plt.show()
        return

    def average_response_FR_for_batch_responses_dual_taste(self, elec_cluster, style='bars', title='',
                                                                normalize=False,save_fig=False):
        fig = plt.figure()
        fig.clf()
        ax = fig.add_subplot(111)
        neuron_num = get_neuron_num_from_dic(self.data, elec_cluster[0], elec_cluster[1])
        ax = plot_average_response_FR_for_batch_responses_dual_taste(ax, self.PCA_data_array, neuron_num, style=style,
                                                                     title=title, normalize=normalize)

        if save_fig:
            fig.savefig('Response FR for {}-{}.jpeg'.format(elec_cluster[0], elec_cluster[1]), format='jpeg')
            fig.savefig('Response FR for {}-{}.svg'.format(elec_cluster[0], elec_cluster[1]), format='svg')
        else:
            plt.show()
        return

    def plot_response_power_spectrum(self, taste='sacc', fs=300, band_to_plot='Gamma', len_in_secs=3, save_fig=False, debug=False):
        fig = plt.figure()
        fig.clf()
        ax = fig.add_subplot(111)
        if self.lfp_data is False:
            print('no LFP data file')
            return

        ax, band_power_per_batch, band_std_per_batch = spectrum_power_for_response(ax, self.lfp_data,
                                                                                   self.event_times_by_batchs[taste],
                                                                                   fs=fs, band_to_plot=band_to_plot,
                                                                                   len_in_secs=len_in_secs,debug=debug)

        if save_fig:
            fig.savefig('Response power for {} band.jpeg'.format(band_to_plot), format='jpeg')
            fig.savefig('Response power for {} band.svg'.format(band_to_plot), format='svg')
        else:
            plt.show()
        return band_power_per_batch, band_std_per_batch

    def plot_response_power_spectrum_dual_taste(self, fs=300, band_to_plot='Gamma',
                                                len_in_secs=3, save_fig=False, debug=False):
        fig = plt.figure(figsize=(16,10))
        fig.clf()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.set_title('Sacc')
        ax2.set_title('Water')

        if self.lfp_data is False:
            print('no LFP data file')
            return

        dic = {}

        ax1, dic['sacc power'], dic['sacc std'] = spectrum_power_for_response(ax1, self.lfp_data,
                                                                                   self.event_times_by_batchs['sacc'],
                                                                                   fs=fs, band_to_plot=band_to_plot,
                                                                                   len_in_secs=len_in_secs, debug=debug)

        ax2, dic['water power'], dic['water std'] = spectrum_power_for_response(ax2, self.lfp_data,
                                                                                   self.event_times_by_batchs['water'],
                                                                                   fs=fs, band_to_plot=band_to_plot,
                                                                                   len_in_secs=len_in_secs, debug=debug)

        adjust_ylim(ax1,ax2)

        if save_fig:
            fig.savefig('Response power for {} band.jpeg'.format(band_to_plot), format='jpeg')
            fig.savefig('Response power for {} band.svg'.format(band_to_plot), format='svg')
        else:
            plt.show()
        return dic

    def plot_and_save_multiple_lfp_bands_BL_and_response(self, average_every_x_minutes=5,
                                                         bands_to_plot=['Delta','Theta','Alpha','Beta','Gamma','Fast Gamma'],
                                                         smooth=True, taste='sacc', len_in_secs=3):
        for band in bands_to_plot:
            self.plot_lfp_power_over_time(average_every_x_minutes=average_every_x_minutes,
                                          bands_to_plot=[band], smooth=smooth, save_fig=True)

            self.plot_response_power_spectrum(taste=taste, fs=self.lfp_FS, band_to_plot=band,
                                              len_in_secs=len_in_secs, save_fig=True)
        return

    def plot_and_save_BL_and_response_FR_for_all_neurons(self, bin_size_in_secs=300, plot_batch_times=True, style='bars'):
        for neuron in self.data['neurons']:
            elec_cluster = (neuron[0],neuron[1])

            self.BL_FR_analysis(elec_cluster=elec_cluster, bin_size_in_secs=bin_size_in_secs,
                                plot_batch_times=plot_batch_times, save_fig=True)

            self.average_response_FR_for_batch_responses_dual_taste(elec_cluster=elec_cluster,
                                                                    style=style, save_fig=True)
        return

def calcCV(spike_train):
    """
    :param spike_train: spike times
    :return: the coefficient of variation of the spike train
    """
    if len(spike_train) < 1:
        return 0
#     print(spike_train)
    if isinstance(spike_train[0],list):
        means = []
        stds = []
        for train in spike_train:
            nptrain = np.array(train)
            spike_intervals = (nptrain[1:] - nptrain[:-1])*1000
            if len(spike_intervals) > 1:
                means.append(np.mean(spike_intervals))
                stds.append(np.std(spike_intervals))
        mean = np.mean([i for i in means if np.isnan(i)==False])
        std = np.mean([i for i in stds if np.isnan(i)==False])
    else:
        spike_train = np.array(spike_train)
        spike_intervals = (spike_train[1:] - spike_train[:-1])*1000
        mean = np.mean(spike_intervals)
        std = np.std(spike_intervals)
    CV = mean/std
    return CV

def calcFF(spike_train,pieces):
    """
    :param spike_train: spike times
    :return: the Fano Factor of the spike train
    """
    spikes_in_pieces = []
    if len(spike_train) < 1:
        return 0
    if isinstance(spike_train[0],list):
        for train in spike_train:
            if len(train) > 1:
                spikes_in_pieces.append(np.count_nonzero(train)/(train[-1]-train[0]))
    else:
        borders = np.linspace(spike_train[0],spike_train[-1],pieces)
        for i in range(1,len(borders)):
            spikes_in_pieces.append(len([spike for spike in spike_train if borders[i-1] < spike < borders[i]]))
    mean = np.mean(spikes_in_pieces)
    var = np.var(spikes_in_pieces)
    FF = mean/var
    return FF


def spectrum_power_for_response(ax, lfp_data, event_times_by_batch, fs=300,
                             band_to_plot='Gamma',len_in_secs=3, debug=False):
    """
    drinking session start needs to be in hours
    """

    lfp_response_mat = []
    samples_per_event = fs * len_in_secs

    for batch in event_times_by_batch:
        batch_responses = []
        for event_time in batch:
            start_index = int(event_time*fs)
            stop_index = int(start_index+samples_per_event)
            batch_responses.append(lfp_data[start_index:stop_index])
        lfp_response_mat.append(batch_responses)

    if debug:
        print('ax')
        print(ax)
        print('lfp response mat')
        print(lfp_response_mat)

    power_over_time = {'Delta': [],
                       'Theta': [],
                       'Alpha': [],
                       'Beta': [],
                       'Gamma': [],
                       'Fast Gamma': []}

    eeg_bands = {'Delta': (1, 4),
                 'Theta': (4, 8),
                 'Alpha': (8, 12),
                 'Beta': (12, 30),
                 'Gamma': (30, 45),
                 'Fast Gamma': (45, 120)}

    # Get real amplitudes of FFT (only in postive frequencies)
    band_power_per_batch = []
    band_std_per_batch = []

    for batch in lfp_response_mat:
        powers = []
        for event_response in batch:
            fft_vals = np.absolute(np.fft.rfft(event_response))

            # Get frequencies for amplitudes in Hz
            fft_freq = np.fft.rfftfreq(len(event_response), 1.0 / fs)

            # Take the mean of the fft amplitude for each EEG band
                #             print('addint to band {}'.format(band))
            freq_ix = np.where((fft_freq >= eeg_bands[band_to_plot][0]) &
                               (fft_freq <= eeg_bands[band_to_plot][1]))[0]
            powers.append(np.mean(fft_vals[freq_ix]))
        band_power_per_batch.append(np.mean(powers))
        band_std_per_batch.append(np.std(powers)/np.sqrt(len(powers)))

    left_edges_BL = np.arange(len(band_power_per_batch)) * 0.45 + 0.1

    bar_width = 0.35
    opacity = 0.8

    error_config = {'ecolor': '0.3'}

    rects1 = ax.bar(left_edges_BL, band_power_per_batch, bar_width,
                     alpha=opacity, yerr=band_std_per_batch,
                     error_kw=error_config, label='{} response'.format(band_to_plot))

    ax.set_xticks(left_edges_BL + bar_width / 2.0)
    ax.set_xticklabels([str(i) for i in range(len(band_power_per_batch))])

    ax.set_xlabel("Batch", fontsize=18)
    ax.set_ylabel("Mean band Amplitude", fontsize=18)
    #     ax.legend()
    legend = ax.legend(fontsize=18, loc='upper right', shadow=True)

    # Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('#ffffff')
    return ax, band_power_per_batch,band_std_per_batch

def plot_average_response_FR_for_batch_responses_dual_taste(ax1, pca_data_array, neuron_num, style='bars',title='', normalize=False):
    """
    creating and showing a dynamic PCA over time of population responses to different tastes.
    """
#     print('creating and showing dynamic PCA over time of population responses to different tastes')

    batch_lengths = {'sacc': [], 'water': []}

    for batch_num,batch in enumerate(pca_data_array[1]):
        batch_lengths['sacc'].append(len(batch))
        if batch_num == 0:
            batch_mat = np.array(batch)
        else:
#             print(batch_mat.shape)
#             print(np.array(batch).shape)
            batch_mat = np.concatenate((batch_mat,np.array(batch)),axis=0)

    for batch_num,batch in enumerate(pca_data_array[0]):
        batch_lengths['water'].append(len(batch))
        batch_mat = np.concatenate((batch_mat,np.array(batch)),axis=0)

    batch_mat = batch_mat[:,neuron_num,:]

#     pca = PCA(pca_3D_data_array[:,:,i], standardize=False)

    batchs_sacc = []
    batchs_water = []
    batch_means_sacc = []
    batch_means_water = []

    i = 0
#     print(pca.Y.shape)

    for j in batch_lengths['sacc']:
        batchs_sacc.append(batch_mat[i:i+j, :])
        batch_means_sacc.append(batch_mat[i:i+j, :].mean(axis=0))
        i += j

    for j in batch_lengths['water']:
        batchs_water.append(batch_mat[i:i+j, :])
        batch_means_water.append(batch_mat[i:i+j, :].mean(axis=0))
        i += j

    batchs_sacc = np.array(batchs_sacc)
    batchs_water = np.array(batchs_water)
    batch_means_sacc = np.array(batch_means_sacc)
    batch_means_water = np.array(batch_means_water)



    sacc_early = batch_means_sacc[:,20:40].mean(axis=1)
    sacc_early_std = batch_means_sacc[:,20:40].std(axis=1)
    sacc_late = batch_means_sacc[:,40:70].mean(axis=1)
    sacc_late_std = batch_means_sacc[:,40:70].std(axis=1)
    water_early = batch_means_water[:,20:40].mean(axis=1)
    water_early_std = batch_means_water[:,20:40].std(axis=1)
    water_late = batch_means_water[:,40:70].mean(axis=1)
    water_late_std = batch_means_water[:,40:70].std(axis=1)

    if len(sacc_early) > len(water_early):
        diff = len(sacc_early) - len(water_early)
        adding = np.array([0 for i in range(diff)])
#         print('sacc larger',diff,adding,adding.shape,water_early,water_early.shape)
        water_early = np.concatenate((adding,water_early))
        water_early_std = np.concatenate((adding,water_early_std))
        water_late = np.concatenate((adding,water_late))
        water_late_std = np.concatenate((adding,water_late_std))
        BL_response = batch_means_sacc[:,:20].mean(axis=1)

    elif len(sacc_early) < len(water_early):
        diff = len(water_early) - len(sacc_early)
        adding = np.array([0 for i in range(diff)])
#         print('water larger',diff,adding,adding.shape)
        sacc_early = np.concatenate((adding,sacc_early))
        sacc_early_std = np.concatenate((adding,sacc_early_std))
        sacc_late = np.concatenate((adding,sacc_late))
        sacc_late_std = np.concatenate((adding,sacc_late_std))
        BL_response = batch_means_water[:,:20].mean(axis=1)

    if style == 'bars':
        left_edges_BL = np.arange(len(sacc_early))*1.9+0.15

        bar_width = 0.35
        opacity = 0.8

        error_config = {'ecolor': '0.3'}

        rects1 = ax1.bar(left_edges_BL, BL_response, bar_width,
                        alpha=opacity, color='grey',
                        error_kw=error_config,
                        label='BL response')

        rects2 = ax1.bar(left_edges_BL + bar_width, sacc_early, bar_width,
                        alpha=opacity, color='lightgreen',
                        yerr=sacc_early_std, error_kw=error_config,
                        label='sacc_early')

        rects3 = ax1.bar(left_edges_BL + bar_width*2, sacc_late, bar_width,
                        alpha=opacity, color='darkgreen',
                        yerr=sacc_late_std, error_kw=error_config,
                        label='sacc_late')

        rects4 = ax1.bar(left_edges_BL + bar_width*3, water_early, bar_width,
                        alpha=opacity, color='lightblue',
                        yerr=water_early_std, error_kw=error_config,
                        label='water_early')

        rects5 = ax1.bar(left_edges_BL + bar_width*4, water_late, bar_width,
                        alpha=opacity, color='darkblue',
                        yerr=water_late_std, error_kw=error_config,
                        label='water_late')

        ax1.set_xticks(left_edges_BL + bar_width * 2)
        ax1.set_xticklabels([str(i) for i in range(len(batch_means_sacc))])

    else:
        if len(sacc_early) >= len(water_early):
            ax1.plot(BL_response,color='grey',label='BL response')
            ax1.plot(sacc_early,color='lightgreen',label='sacc_early')
            ax1.plot(sacc_late,color='darkgreen',label='sacc_late')
            ax1.plot(np.arange(diff,len(sacc_early)),water_early[diff:],color='lightblue',label='water_early')
            ax1.plot(np.arange(diff,len(sacc_early)),water_late[diff:],color='darkblue',label='water_late')
        else:
            ax1.plot(BL_response,color='grey',label='BL response')
            ax1.plot(np.arange(diff,len(water_early)),sacc_early[diff:],color='lightgreen',label='sacc_early')
            ax1.plot(np.arange(diff,len(water_early)),sacc_late[diff:],color='darkgreen',label='sacc_late')
            ax1.plot(water_early,color='lightblue',label='water_early')
            ax1.plot(water_late,color='darkblue',label='water_late')

#         ax1 = sns.tsplot(batchs_sacc[:,:,0:20].mean(axis=2))
#         batchs_water

    ax1.set_xlabel('Batch #')
    ax1.set_ylabel('Fire rate (Hz)')
    ax1.set_title('Response FR across times')
    ax1.legend(loc='upper right')

    return ax1

def plot_BL_FR_analysis(ax, N0FR, batch_times_in_hours=[], bin_size_in_secs=300, plot_batch_times=True, session_start=0):

    bin_amount = int(N0FR[-1])/bin_size_in_secs
    N0FR_over_time,bin_edges = np.histogram(N0FR,int(bin_amount),(0,N0FR[-1]))
#     N0FR_over_time_normalized = N0FR_over_time-np.mean(N0FR_over_time)
    # norm_curve = savitzky_golay(spikes_in_bin,5,3)
    smoothed_curve = savitzky_golay(N0FR_over_time/bin_size_in_secs,9,3)
    ax.plot(bin_edges[:-1]/3600,smoothed_curve,color="black")
    top_of_line = max(smoothed_curve)

    if plot_batch_times:
        ax.vlines(session_start,0,top_of_line*1.1,'r',linestyles='dashed',linewidth=2)
        ax.vlines(session_start+1.0/3,0,top_of_line*1.1,'r',linestyles='dashed',linewidth=2,label='drinking session')
        ax.vlines(batch_times_in_hours,0,top_of_line*1.1,'g',linestyles='dashed',linewidth=2,label='batch times')
        ax.vlines(session_start+1+1.0/6,0,top_of_line*1.1,'b',linestyles='dashed',linewidth=2,label='CTA injection')

    ax.legend()
    return ax

def plot_psth_over_time_for_neuron(ax, psth_mat, neuron_num, taste='sugar', start_batch=0, stop_batch=30):
    """
    plots a figure with 2 subplots, the top will be a raster with all tastes given in the list. the bottom will be a PSTH with the tastes given in the list.
    spike train is a list of spike times.
    event_dic is a dictionary with keys as tastes and values that are lists of taste times.
    start and end time are the boundaries in secs of the PSTH and raster.
    """
    ax.set_title("{} response".format(taste),fontsize=18)
    bin_amount = 5//0.05
    if taste == 'sacc':
        taste = 'sugar'

    switch = 0
    if taste == 'sugar':
        switch = 1

    for batch_num,batch in enumerate(psth_mat[switch][start_batch:stop_batch]):
        batch_mat = np.array(batch)
        spikes_in_bin = batch_mat[:,neuron_num,:].mean(axis=0)
        norm_curve = savitzky_golay(spikes_in_bin,15,3)
        ax.plot(np.linspace(-1,4,99),norm_curve,label="batch {}".format(batch_num),linewidth=3)
    ax.set_xlabel('Peri-stimulus time (Sec)',fontsize=18)
    ax.set_ylabel('Firing rate (Spikes/Sec)',fontsize=18)
    ax.axvline(0, linestyle='--', color='k') # vertical lines
    ax.legend(fontsize=18)
    ax.xaxis.set_ticks(np.arange(-1, 4+0.5, 0.5))
    ax.tick_params(axis='both', which='major', labelsize=18)

    return ax

def PCA_2D_for_batch_responses_dual_taste(ax1, pca_data_array, neuron_num, taste='both', title='', normalize=False):
    """
    creating and showing a dynamic PCA over time of population responses to different tastes.
    """
    print('creating and showing dynamic PCA over time of population responses to different tastes')
    batch_lengths = {'sacc': [], 'water': []}

    for batch_num, batch in enumerate(pca_data_array[1]):
        batch_lengths['sacc'].append(len(batch))
        if batch_num == 0:
            batch_mat = np.array(batch)
        else:
            #             print(batch_mat.shape)
            #             print(np.array(batch).shape)
            batch_mat = np.concatenate((batch_mat, np.array(batch)), axis=0)

    for batch_num, batch in enumerate(pca_data_array[0]):
        batch_lengths['water'].append(len(batch))
        batch_mat = np.concatenate((batch_mat, np.array(batch)), axis=0)

    batch_mat = batch_mat[:, neuron_num, 30:70]

    #     pca = PCA(pca_3D_data_array[:,:,i], standardize=False)

    pca = PCA(batch_mat, standardize=normalize)

    PCA_batchs_sacc = []
    PCA_batchs_water = []
    PCA_batch_means_sacc = []
    PCA_batch_means_water = []

    distances = []
    distances_2d = []

    i = 0
    #     print(pca.Y.shape)

    for j in batch_lengths['sacc']:
        #         print (i,j,pca.Y[i:j, :2])
        PCA_batchs_sacc.append(pca.Y[i:i + j, :])
        PCA_batch_means_sacc.append(pca.Y[i:i + j, :].mean(axis=0))
        i += j

    for j in batch_lengths['water']:
        #         print (i,j,pca.Y[i:j, :2])
        PCA_batchs_water.append(pca.Y[i:i + j, :])
        PCA_batch_means_water.append(pca.Y[i:i + j, :].mean(axis=0))
        i += j

    PCA_batchs_sacc = np.array(PCA_batchs_sacc)
    PCA_batchs_water = np.array(PCA_batchs_water)
    PCA_batch_means_sacc = np.array(PCA_batch_means_sacc)
    PCA_batch_means_water = np.array(PCA_batch_means_water)

    if taste == 'both' or taste == 'sacc':
        ax1.plot(PCA_batch_means_sacc[:, 0], PCA_batch_means_sacc[:, 1], color='green',
                 label='sacc progress through time')
    if taste == 'both' or taste == 'water':
        ax1.plot(PCA_batch_means_water[:, 0], PCA_batch_means_water[:, 1], color='blue',
                 label='water progress through time')

    #     color_scheme = ['black','C1','C2','C3','c4','c5','c6','c7','c8','c9','c10','c11','c12','c13','c14','c15']
    cmap = plt.get_cmap('gist_rainbow')
    color_scheme = [cmap(i) for i in np.linspace(0, 1, len(PCA_batchs_sacc) + len(PCA_batchs_water))]

    #     print(PCA_batchs)
    if taste == 'both' or taste == 'sacc':
        for batch_num, batch in enumerate(PCA_batchs_sacc):
            batch = np.array(batch)
            #         print(batch.shape)
            ax1.scatter(batch[:, 0], batch[:, 1], label='sacc batch {}'.format(batch_num),
                        color=color_scheme[batch_num], s=15)

            ax1.scatter(PCA_batch_means_sacc[batch_num, 0], PCA_batch_means_sacc[batch_num, 1],
                        color=color_scheme[batch_num], s=30)
            ax1.text(PCA_batch_means_sacc[batch_num, 0], PCA_batch_means_sacc[batch_num, 1], '{}'.format(batch_num),
                     fontsize=12)

    if taste == 'both' or taste == 'water':
        for batch_num, batch in enumerate(PCA_batchs_water):
            batch = np.array(batch)
            #         print(batch.shape)
            ax1.scatter(batch[:, 0], batch[:, 1], label='water batch {}'.format(batch_num + 3),
                        color=color_scheme[batch_num + len(PCA_batchs_sacc)], s=15)

            ax1.scatter(PCA_batch_means_water[batch_num, 0], PCA_batch_means_water[batch_num, 1],
                        color=color_scheme[batch_num + len(PCA_batchs_sacc)], s=30)
            ax1.text(PCA_batch_means_water[batch_num, 0], PCA_batch_means_water[batch_num, 1],
                     '{}'.format(batch_num + 3), fontsize=12)

    for i in range(len(PCA_batch_means_water)):
        distances.append(distance.euclidean(PCA_batch_means_water[i, :], PCA_batch_means_sacc[i + 3, :]))
        distances_2d.append(distance.euclidean(PCA_batch_means_water[i, :2], PCA_batch_means_sacc[i + 3, :2]))

    ax1.set_title(title + '\nTotal Var Exp = {}%'.format(sum(pca.fracs[:2]) * 100))
    # ax1.set_xlim(-2,2)
    # ax1.set_ylim(-1, 1)
    # ax1.set_zlim(-1, 1)
    ax1.set_xlabel('PC1 - V.E = {0:02f}%'.format(pca.fracs[0] * 100))
    ax1.set_ylabel('PC2 - V.E = {0:02f}%'.format(pca.fracs[1] * 100))
    ax1.legend(loc='upper right')
    return ax1, distances, distances_2d

def PCA_2D_for_population_batch_responses(ax1, pca_data_array, neuron_num_list, taste='sacc', title='', normalize=False):
    """
    creating and showing a dynamic PCA over time of population responses to different tastes.
    """
    print('creating and showing dynamic PCA over time of population responses to different tastes')
    batch_lengths = []
    taste_to_num = {'sacc': 1, 'water': 0}

    for batch_num, batch in enumerate(pca_data_array[taste_to_num[taste]]):
        batch_lengths.append(len(batch))
        if batch_num == 0:
            batch_mat = np.array(batch)
        else:
            #             print(batch_mat.shape)
            #             print(np.array(batch).shape)
            batch_mat = np.concatenate((batch_mat, np.array(batch)), axis=0)

    batch_mat = batch_mat[:, neuron_num_list, 30:70]
    averaged_neural_response_batch_mat = []
    total_batch_len = 0
    for current_batch_len in batch_lengths:
        batch_responses = []
        for neuron_num in neuron_num_list:
            average_neuron_FR_for_batch = batch_mat[total_batch_len : total_batch_len + current_batch_len, neuron_num, :].mean(axis=0)
            batch_responses.append(average_neuron_FR_for_batch)
        averaged_neural_response_batch_mat.append(batch_responses)

    """ averaged_neural_response_batch_mat dimensions: 0 - batch, 1 - neuron average response, 2 - time """
    averaged_neural_response_batch_mat = np.swapaxes(averaged_neural_response_batch_mat,0,1)
    """ averaged_neural_response_batch_mat dimensions: 0 - neuron average response, 1 - batch, 2 - time """
    d1,d2,d3 = averaged_neural_response_batch_mat.shape
    flat_averaged_neural_response_batch_mat = np.transpose(np.reshape(averaged_neural_response_batch_mat,(d1,d2+d3)))
    """ averaged_neural_response_batch_mat dimensions: 1 - neurons, 0 - batch_times so every 40 bins is
    1 average batch response ex: dim 0 is neurons, and dim 1 is built like this: 0-40 is bins of batch 1, 41-80 is bins of batch 2 etc"""
    #     pca = PCA(pca_3D_data_array[:,:,i], standardize=False)

    pca = PCA(flat_averaged_neural_response_batch_mat, standardize=normalize)
    reshaped_pca = np.reshape(np.transpose(pca.Y),(d1,d2,d3))
    # reshaped_pca = np.swapaxes(reshaped_pca, 0, 1)
    # reshaped_pca = np.swapaxes(reshaped_pca, 1, 2)

    PCA_batchs_x = []
    PCA_batchs_y = []

    for i in range(d2):
        PCA_batchs_x.append(reshaped_pca[0,i,:])
        PCA_batchs_y.append(reshaped_pca[1,i,:])


    for i in range(len(PCA_batchs_x)):
        ax1.plot(PCA_batchs_x[i], PCA_batchs_y[i],label='population response for batch {}'.format(i+1))

    ax1.set_title(title + '\nTotal Var Exp = {}%'.format(sum(pca.fracs[:2]) * 100))
    # ax1.set_xlim(-2,2)
    # ax1.set_ylim(-1, 1)
    # ax1.set_zlim(-1, 1)
    ax1.set_xlabel('PC1 - V.E = {0:02f}%'.format(pca.fracs[0] * 100))
    ax1.set_ylabel('PC2 - V.E = {0:02f}%'.format(pca.fracs[1] * 100))
    ax1.legend(loc='upper right')
    return ax1

def calc_batch_times(dic,taste):
    diffs = dic['event_times'][taste][1:] - dic['event_times'][taste][:-1]
    change_locs = np.array(np.where(diffs > 300)) + 1
    return dic['event_times'][taste][change_locs[0]]

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

def create_psth_matrix_for_long_rec(all_neurons_spike_times, event_dic, taste_list=('water','sugar'),bad_elec_list=[]):
    matrix_dic = []
    # taste_event_amount = {'water': 0, 'sugar': 0, 'nacl': 0, 'CA': 0}
    bin_amount = 5//0.05
    for taste in taste_list:
        print("working on taste: {}".format(taste))
        taste_dic = []
        diffs = event_dic[taste][1:]-event_dic[taste][:-1]
        change_locs = np.array(np.where(diffs>300))+1
        change_locs = np.concatenate((np.array([0]),change_locs[0],np.array([len(event_dic[taste])])))
        for i in range(len(change_locs)-1):
            print("working on batch {} out of {}".format(i,len(change_locs)))
            batch_dic = []
            batch_events = event_dic[taste][change_locs[i]:change_locs[i+1]]
            for event in batch_events:
                all_neural_responses_for_event = []
                # taste_event_amount[taste] += 1
                for neural_spike_times in all_neurons_spike_times:
                    if neural_spike_times[0] not in bad_elec_list:
                        left_border = np.searchsorted(neural_spike_times[2] - event, -1)
                        right_border = np.searchsorted(neural_spike_times[2] - event, 4)
                        spikes = neural_spike_times[2][left_border:right_border] - event
                        hist1, bin_edges = np.histogram(spikes, int(bin_amount), (-1, 4))
                        spikes_in_bin = hist1 / 0.05
                        all_neural_responses_for_event.append(spikes_in_bin)
                batch_dic.append(all_neural_responses_for_event)
            taste_dic.append(batch_dic)
        matrix_dic.append(taste_dic)
    matrix_dic = np.array(matrix_dic)
    # print(taste_event_amount)
    return matrix_dic

def spectrum_power_over_time(ax, lfp_data, batch_times_in_hours, fs=300, average_every_x_minutes=5,
                             bands_to_plot=['Delta','Theta','Alpha','Beta','Gamma','Fast Gamma'], smooth=True,
                             drinking_session_start=None):
    """
    drinking session start needs to be in hours
    """
    samples_per_batch = fs*60*average_every_x_minutes
    amount_of_chunks = int(len(lfp_data)/samples_per_batch)
    batchs = np.array_split(lfp_data, amount_of_chunks)
    batchs = np.array([x for x in batchs if x.size > 0])
    power_over_time = {'Delta': [],
                     'Theta': [],
                     'Alpha': [],
                     'Beta': [],
                     'Gamma': [],
                     'Fast Gamma': []}

    eeg_bands = {'Delta': (1, 4),
                 'Theta': (4, 8),
                 'Alpha': (8, 12),
                 'Beta': (12, 30),
                 'Gamma': (30, 45),
                 'Fast Gamma': (45, 120)}

    # Get real amplitudes of FFT (only in postive frequencies)
    counter = 1
    for batch in batchs:
#         print('working on batch {} out of {}'.format(counter,len(batchs)))
        counter+=1

        fft_vals = np.absolute(np.fft.rfft(batch))
#         print('calculated fft vals')

    # Get frequencies for amplitudes in Hz
        fft_freq = np.fft.rfftfreq(len(batch), 1.0/fs)
#         print('calculated fft freqs')

    # Take the mean of the fft amplitude for each EEG band
        for band in eeg_bands:
#             print('addint to band {}'.format(band))
            freq_ix = np.where((fft_freq >= eeg_bands[band][0]) &
                               (fft_freq <= eeg_bands[band][1]))[0]
            power_over_time[band].append(np.mean(fft_vals[freq_ix]))
    # Plot the data (using pandas here cause it's easy)
    xaxis_values = np.arange(len(power_over_time['Gamma']))*average_every_x_minutes/60
    for band in bands_to_plot:
        if smooth:
            power_over_time[band] = savitzky_golay(power_over_time[band],9,3)
        ax.plot(xaxis_values,power_over_time[band],label=band)

    ymin, ymax = ax.get_ylim()

    if drinking_session_start is not None:
        ax.vlines(drinking_session_start,ymin,ymax,'r',linestyles='dashed',linewidth=2)
        ax.vlines(drinking_session_start+1.0/3,ymin,ymax,'r',linestyles='dashed',linewidth=2,label='drinking session')
        ax.vlines(batch_times_in_hours,ymin,ymax,'g',linestyles='dashed',linewidth=2,label='batch times')
        ax.vlines(drinking_session_start+1+1.0/6,ymin,ymax,'b',linestyles='dashed',linewidth=2,label='CTA injection')
        ax.vlines(drinking_session_start+4+1.0/6,ymin,ymax,'purple',linestyles='dashed',linewidth=2,label='3h post CTA')


    ax.set_xlabel("Time",fontsize=18)
    ax.set_ylabel("Mean band Amplitude",fontsize=18)
#     ax.legend()
    legend = ax.legend(fontsize=18,loc='upper right', shadow=True)

    # Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('#ffffff')
    return ax,power_over_time

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def get_spike_data_from_dic(dic,elec,cluster):
    for arr in dic['neurons']:
        if arr[0] == elec and arr[1] == cluster:
            return arr[2]
    print('no such neuron found, returning None')
    return None

def get_neuron_num_from_dic(dic,elec,cluster):
    counter = -1
    for arr in dic['neurons']:
        counter += 1
        if arr[0] == elec and arr[1] == cluster:
            return counter
    print('no neuron found with elec = {}, cluster = {}, returning None'.format(elec,cluster))
    return None

def undersample(in_file,out_file,undersample_factor=[10,10]):
    assert isinstance(undersample_factor,(list,tuple)),"undersample factor must be a list or tuple."
    with open(in_file,'rb') as in_f:
        with open(out_file,'wb') as out_f:
            data = np.fromfile(in_f,dtype=np.int16,count=3000000)
            while len(data)>0:
                if len(undersample_factor) == 2:
                    filtered_once = decimate(data,undersample_factor[0],zero_phase=True)
                    filtered_twice = decimate(filtered_once,undersample_factor[1],zero_phase=True)
                else:
                    filtered_twice = decimate(data,undersample_factor[0],zero_phase=True)
                filtered_twice.astype('int16').tofile(out_f)
                data = np.fromfile(in_f,dtype=np.int16,count=3000000)

def undersample_file(in_file,out_file,current_FS=30000, new_FS=300, filter=True, low_pass=500):
    print("please do not use this function - use multi_klusta.undersample instead")
    return
    open_read_file = open(in_file + '.dat', 'r+b')
    open_write_file = open(out_file + '.dat', 'wb')

    # get first bytes to initialize while loop
    open_read_file.seek(0)
    by = open_read_file.read(2)
    byte_arr = []
    # iterate over 2 bytes each time and write them minus the average to new file.
    i = 0
    write_every_x = int(current_FS/new_FS)
    if filter:
        while len(by) > 1 and i<=300000:
            if i == 300000:
                i = 0
                ints = [int.from_bytes(by1, 'little', signed=True) for by1 in byte_arr]
                b, a = butter_bandpass(0,low_pass,current_FS,5)
                filtered_ints = filtfilt(b,a,ints)
                bytes_to_write = [int.to_bytes(int(num), 2, 'little', signed=True) for num in filtered_ints[::write_every_x]]
                for write_byte in bytes_to_write:
                    open_write_file.write(write_byte)
                byte_arr = []

            else:
                i+=1
                byte_arr.append(by)
            by = open_read_file.read(2)
    else:
        while len(by) > 1:
            i+=1
            by = open_read_file.read(2)
            if i % write_every_x == 0:
                open_write_file.write(by)
    open_read_file.close()
    open_write_file.close()

def get_event_times_from_pickle(filename):
    with (open(filename, "rb")) as openfile:
        while True:
            try:
                data = pickle.load(openfile)
            except EOFError:
                break
    return data['event_times']['water'], data['event_times']['sugar']