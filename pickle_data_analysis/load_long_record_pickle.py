import pickle
import numpy as np
import scipy as scp
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.mlab import PCA
from scipy.spatial import distance
import os
from band_pass_filters import butter_bandpass_filter
from scipy.io import loadmat
from scipy.stats import ttest_ind as Ttest
from scipy.signal import butter, lfilter
from math import factorial

plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams["axes.edgecolor"] = "0.15"
plt.rcParams["axes.linewidth"]  = 1.25
plt.rcParams["savefig.edgecolor"] = "0.15"

class pickle_loader(object):
    """
    session start time needs to be in (hours,minute) format from the start of recording
    """
    def __init__(self,filename,session_start_time,lfp_data_file=False,lfp_fs=300):
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

    def get_all_event_and_batch_times(self,session_start_time):
        self.event_times_in_secs = {
            'session start': session_start_time[0]*3600+session_start_time[1]*60,
            'sacc batch times': calc_batch_times(self.data,'sugar'),
            'water batch times': calc_batch_times(self.data, 'water'),
        }

    def plot_lfp_power_over_time(self,average_every_x_minutes=5, bands_to_plot=['Delta','Theta','Alpha','Beta','Gamma','Fast Gamma'],
                                 smooth=True,save_fig=False):
        if not self.lfp_data:
            print('no LFP data file')
            return
        fig = plt.Figure()
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
        fig = plt.Figure()
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

    def psth_over_time_for_neuron(self, elec_cluster, taste='sugar', start_batch=0, stop_batch=30, save_fig=False):
        fig = plt.Figure()
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
        fig = plt.Figure()
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
        fig = plt.Figure()
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

    def plot_response_power_spectrum(self, taste='sacc', fs=300, band_to_plot='Gamma', len_in_secs=3, save_fig=False):
        fig = plt.Figure()
        fig.clf()
        ax = fig.add_subplot(111)
        ax, band_power_per_batch, band_std_per_batch = spectrum_power_for_response(ax, self.lfp_data,
                                                                                   self.event_times_by_batchs[taste],
                                                                                   fs=fs, band_to_plot=band_to_plot,
                                                                                   len_in_secs=len_in_secs)
        if save_fig:
            fig.savefig('Response power for {} band.jpeg'.format(band_to_plot), format='jpeg')
            fig.savefig('Response power for {} band.svg'.format(band_to_plot), format='svg')
        else:
            plt.show()
        return band_power_per_batch, band_std_per_batch

    def plot_and_save_multiple_lfp_bands_BL_and_response(self, average_every_x_minutes=5,
                                                         bands_to_plot=['Delta','Theta','Alpha','Beta','Gamma','Fast Gamma'],
                                                         smooth=True, taste='sacc', len_in_secs=3):
        for band in bands_to_plot:
            self.plot_lfp_power_over_time(average_every_x_minutes=average_every_x_minutes,
                                          bands_to_plot=[band], smooth=smooth, save_fig=True)

            self.plot_response_power_spectrum(taste=taste, fs=self.lfp_FS, bands_to_plot=band,
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




def spectrum_power_for_response(ax, lfp_data, event_times_by_batch, fs=300,
                             band_to_plot='Gamma',len_in_secs=3):
    """
    drinking session start needs to be in hours
    """

    lfp_response_mat = []
    samples_per_event = fs * len_in_secs

    for batch in event_times_by_batch:
        batch_responses = []
        for event_time in batch:
            start_index = event_time*fs
            stop_index = start_index+samples_per_event
            batch_responses.append(lfp_data[start_index:stop_index])
        lfp_response_mat.append(batch_responses)

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
                        spikes = [neural_spike_times[2][i] - event for i in range(len(neural_spike_times[2])) if -1 < neural_spike_times[2][i] - event < 4]
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
    print('no such neuron found, returning None')
    return None

def undersample_file(in_file,out_file,current_FS=30000, new_FS=300):
    open_read_file = open(in_file + '.dat', 'r+b')
    open_write_file = open(out_file + '.dat', 'wb')

    # get first bytes to initialize while loop
    by = open_read_file.read(2)
    open_read_file.seek(0)

    # iterate over 2 bytes each time and write them minus the average to new file.
    i = 0
    write_every_x = int(current_FS/new_FS)
    while len(by) > 1:
        i+=1
        by = open_read_file.read(2)
        if i % write_every_x == 0:
            open_write_file.write(by)
    open_read_file.close()
    open_write_file.close()