from scipy.signal import welch
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from scipy.signal import welch,spectrogram,lfilter,butter
import matplotlib.colors as colors
from matplotlib.mlab import PCA
from tqdm import tqdm
import scipy
from scipy.stats import signaltonoise as calcSNR
from scipy.stats import zscore
from moran_lab.plotter import our_ts_plot
from moran_lab.band_pass_filters import savitzky_golay, butter_bandpass_filter
from moran_lab.pickle_data_analysis.extra_functions import get_events_between_times
from scipy.ndimage.filters import gaussian_filter

def average_response_spectogram(lfp_data, event_times, fs=300,filtered=True,sigma=3):
    mats = []
    for t in event_times:
        start_index = int(t*300-600)
        stop_index = int(t*300+1500)
        freq_ax, time_ax, Sxx = spectrogram(lfp_data[start_index:stop_index], 300, nperseg=80, noverlap=60, nfft=300)
        if filtered:
            mats.append(gaussian_filter(zscore(Sxx[:,:]), [sigma,sigma], mode='constant'))
        else:
            mats.append(zscore(Sxx[:,:]))
    return time_ax[13:-13]-2.0, freq_ax[:51], np.array(mats)[:,:51,13:-13]

def plot_average_spectogram_in_time_slice(ax, lfp_data, event_times, start_time, stop_time, fs=300, filtered=True, sigma=3):
    real_event_times = get_events_between_times(event_times, start_time, stop_time)
    time_ax, freq_ax, Sxx_mat = average_response_spectogram(lfp_data, real_event_times, fs=fs,
                                                            filtered=filtered, sigma=sigma)
    im = ax.pcolormesh(time_ax, freq_ax, Sxx_mat.mean(axis=0), cmap='jet')
    divider = make_axes_locatable(axes[2])
    cax = divider.append_axes("right", size="5%", pad=0.05, fontsize=14)
    ax.colorbar(im, cax=cax)
    ax.set_xlabel("Time (sec)",fontsize=14)
    ax.set_ylabel("Frequency (Hz)",fontsize=14)
    return ax

def spike_triggered_LFP(ax, spike_train, LFP_data, FS=300, start_time_in_secs=None, stop_time_in_secs=None, LFP_start=-0.5, LFP_stop=0.1, num_of_stds=3, band=False):

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
        if band is False:
            res_mat[i,:] = LFP_data[j+start_index_fix:j+stop_index_fix]
        else:
            res_mat[i,:] = butter_bandpass_filter(LFP_data[j+start_index_fix:j+stop_index_fix],fs=FS,lowcut=band[0],highcut=band[1])

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


def power_spec_on_mat(mat, fs=300, average_every_x_minutes=5,band='Gamma', smooth=True, normalize=True,z_score=True):
    new_mat = []
    for line in range(mat.shape[0]):
        new_mat.append(spectrum_power_over_time(mat[line,:], fs=fs, average_every_x_minutes=average_every_x_minutes,band=band, smooth=smooth))
    new_mat = np.array(new_mat)
    if normalize:
        if z_score:
            for line in range(new_mat.shape[0]):
                new_mat[line,:] = zscore(new_mat[line,:])
        else:
            for line in range(new_mat.shape[0]):
                new_mat[line,:] /= np.mean(new_mat[line,:int(20/average_every_x_minutes)])
    return new_mat

def PCA_for_spec_power(mat,fs=300, average_every_x_minutes=5, smooth=True, normalize=True,z_score=True):
    new_mat = []
    for i,band in enumerate(['Alpha','Beta','Gamma','Delta','Theta']):
        new_mat.append(power_spec_on_mat(mat, fs=fs, average_every_x_minutes=average_every_x_minutes,
                                         band=band, smooth=smooth, normalize=normalize,z_score=z_score).mean(axis=0))
    new_mat = np.array(new_mat)
    print(new_mat.shape)
    pca = PCA(new_mat.T)
    fig = plt.figure(figsize=(20,15),dpi=1000)
    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(pca.Y[:,0],pca.Y[:,1])
    for i in range(pca.Y.shape[0]):
        ax.text(pca.Y[i, 0], pca.Y[i, 1], '{}'.format(i))
    plt.show()


def plot_spectrum_power_over_time(ax, lfp_data, batch_times_in_hours, fs=300, average_every_x_minutes=5,
                             bands_to_plot=['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma', 'Fast Gamma'], smooth=True,
                             drinking_session_start=None):


    """
    drinking session start needs to be in hours
    """
    samples_per_batch = fs * 60 * average_every_x_minutes
    amount_of_chunks = int(len(lfp_data) / samples_per_batch)
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
        counter += 1

        fft_vals = np.absolute(np.fft.rfft(batch))
        #         print('calculated fft vals')

        # Get frequencies for amplitudes in Hz
        fft_freq = np.fft.rfftfreq(len(batch), 1.0 / fs)
        #         print('calculated fft freqs')

        # Take the mean of the fft amplitude for each EEG band
        for band in eeg_bands:
            #             print('addint to band {}'.format(band))
            freq_ix = np.where((fft_freq >= eeg_bands[band][0]) &
                               (fft_freq <= eeg_bands[band][1]))[0]
            power_over_time[band].append(np.mean(fft_vals[freq_ix]))
    # Plot the data (using pandas here cause it's easy)
    xaxis_values = np.arange(len(power_over_time['Gamma'])) * average_every_x_minutes / 60
    for band in bands_to_plot:
        if smooth:
            power_over_time[band] = savitzky_golay(power_over_time[band], 9, 3)
        ax.plot(xaxis_values, power_over_time[band], label=band)

    ymin, ymax = ax.get_ylim()

    if drinking_session_start is not None:
        ax.vlines(drinking_session_start, ymin, ymax, 'r', linestyles='dashed', linewidth=2)
        ax.vlines(drinking_session_start + 1.0 / 3, ymin, ymax, 'r', linestyles='dashed', linewidth=2,
                  label='drinking session')
        ax.vlines(batch_times_in_hours, ymin, ymax, 'g', linestyles='dashed', linewidth=2, label='batch times')
        ax.vlines(drinking_session_start + 1 + 1.0 / 6, ymin, ymax, 'b', linestyles='dashed', linewidth=2,
                  label='CTA injection')
        ax.vlines(drinking_session_start + 4 + 1.0 / 6, ymin, ymax, 'purple', linestyles='dashed', linewidth=2,
                  label='3h post CTA')

    ax.set_xlabel("Time", fontsize=18)
    ax.set_ylabel("Mean band Amplitude", fontsize=18)
    #     ax.legend()
    legend = ax.legend(fontsize=18, loc='upper right', shadow=True)

    # Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('#ffffff')
    return ax, power_over_time

def spectrum_power_over_time(lfp_data, fs=300, average_every_x_minutes=5,band='Gamma', smooth=True,normalize=True,use_zscore=True):
    """
    drinking session start needs to be in hours
    """
    samples_per_batch = fs*60*average_every_x_minutes
    amount_of_chunks = int(len(lfp_data)/samples_per_batch)
    batchs = np.array_split(lfp_data, amount_of_chunks)
    batchs = np.array([x for x in batchs if x.size > 0])

    power_over_time = []

    eeg_bands = {'Delta': (1, 4),
                 'Theta': (4, 8),
                 'Alpha': (8, 12),
                 'Beta': (12, 30),
                 'Gamma': (30, 45),
                 'Fast Gamma': (45, 120)}

    # Get real amplitudes of FFT (only in postive frequencies)
    counter = 1
    print('running FFT')
    for batch in tqdm(batchs):
#         print('working on batch {} out of {}'.format(counter,len(batchs)))
        counter+=1

        fft_vals = np.absolute(np.fft.rfft(batch))
#         print('calculated fft vals')

    # Get frequencies for amplitudes in Hz
        fft_freq = np.fft.rfftfreq(len(batch), 1.0/fs)
#         print('calculated fft freqs')

    # Take the mean of the fft amplitude for each EEG band
#             print('addint to band {}'.format(band))
        freq_ix = np.where((fft_freq >= eeg_bands[band][0]) &
                           (fft_freq <= eeg_bands[band][1]))[0]

        power_over_time.append(np.mean(fft_vals[freq_ix]))
    # Plot the data (using pandas here cause it's easy)
    if smooth:
        power_over_time = savitzky_golay(power_over_time,9,3)

    if normalize:
        if use_zscore:
            power_over_time = zscore(power_over_time)
        else:
            power_over_time /= np.mean(power_over_time[:int(20 / average_every_x_minutes)])
    return power_over_time

def corr_coef_mat_in_window(lfp_data,start_time_in_minutes,stop_time_in_minutes,average_every_x_minutes=1,ax=None):
    pass

# def plot_corrcoef_on_ax(ax,lfp_data,start_time_in_minutes,stop_time_in_minutes,average_every_x_minutes=1)