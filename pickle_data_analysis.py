from scipy.signal import welch
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from scipy.signal import welch,spectrogram,lfilter,butter
import matplotlib.colors as colors
from tqdm import tqdm
import scipy
from scipy.stats import signaltonoise as calcSNR



class lfpFunctions(object):
    def __init__(self):
        pass

    def get_average_trace_for_events(self, trace, event_list, start=-1, stop=4, Fs=30000.):
        multi_traces = []
        for event in event_list:
            low_edge = (event + start) * Fs
            high_edge = (event + stop) * Fs
            new_trace = trace[int(low_edge):int(high_edge)]
            over_hang = len(new_trace) - (stop - start) * Fs
            if over_hang > 0:
                new_trace = new_trace[:-int(over_hang)]
            multi_traces.append(new_trace)
        multi_traces = np.array(multi_traces)
        return multi_traces.mean(axis=0)

    def get_trace(self, file_name, high_pass=None, sampling_rate=30000.):
        """
        returns a smoothed (with butter filter) trace between the times given.
        """
        trace = np.fromfile(file_name, dtype='int16')
        if high_pass is not None:
            smooth_trace = self.butter_bandpass_filter(trace, float(high_pass), 2000., sampling_rate, order=6)
            return (trace, smooth_trace)
        return trace

    def middle_pass(self, file_name, low_range, high_range, sampling_rate=30000.):
        """
            returns a smoothed (with butter filter) trace between the times given.
            """
        trace = np.fromfile(file_name, dtype='int16')
        # new_trace = trace[::100]
        smooth_trace = self.butter_lowpass_filter(trace, float(high_range), sampling_rate, order=3)
        return smooth_trace

    def get_power_spectrum_for_taste(self, lfp_data, event_dic, taste, start, stop, Fs=30000.):
        if isinstance(taste, str):
            event_list = event_dic[taste]
        else:
            event_list = []
            for key in taste:
                event_list += list(event_dic[key])
        average_taste_trace = self.get_average_trace_for_events(lfp_data, event_list, start, stop, Fs)
        f, Pxx = welch(average_taste_trace, Fs, nperseg=len(average_taste_trace))
        return (f, Pxx)

    def get_trace_for_event(self, trace, event_time, length, Fs):
        low = int(event_time * Fs)
        high = low + int(length * Fs)
        return trace[low:high]

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def butter_lowpass(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(self, data, cutoff, fs, order=5):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y