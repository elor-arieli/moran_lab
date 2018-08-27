import pickle
import numpy as np
import scipy as scp
from matplotlib import pyplot as plt
import os
from band_pass_filters import butter_bandpass_filter
from scipy.io import loadmat
from scipy.stats import ttest_ind as Ttest

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
        if lfp_data_file:
            self.lfp_data = np.fromfile(lfp_data_file,dtype=np.int16)

    def get_all_event_and_batch_times(self):
        self.event_times_in_secs = {
            'session start' = session_start_time[0]
        }