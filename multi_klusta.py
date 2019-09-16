__author__ = 'elor'

import os
import numpy as np
import sys
import fileinput
import glob
import tqdm
import cv2
import load_intan_rhd_format as load_rhd
# from moran_lab.band_pass_filters import savitzky_golay
from scipy.signal import decimate
from moran_lab.band_pass_filters import butter_bandpass_filter

# import yaml

def turn_amplifier_arr_to_bytes(arr):
    return [int.to_bytes(int(num), 2, 'little', signed=True) for num in arr]

def get_amp_din_data(file):
    a = load_rhd.read_data(file)
    full_amp_data = a['amplifier_data']
    full_din_data = a['board_dig_in_data']
    return full_amp_data,full_din_data

def get_amp_names(file):
    a = load_rhd.read_data(file)
    amps = [a['amplifier_channels'][i]['native_channel_name'] for i in range(len(a['amplifier_channels']))]
    dins = [a['board_dig_in_channels'][i]['native_channel_name'] for i in range(len(a['board_dig_in_channels']))]
    return sorted(amps),sorted(dins)

def merge_directories(first_dir,second_dir,out_dir,amp_letter="A"):
    """
    :param first_dir: the directory name with the first half of the recording
    :param second_dir: the directory name with the second half of the recording
    :param out_dir: the directory name where the full ouput will be saved
    :param amp_letter:
    :return:
    """
    file_list = ["amp-"+amp_letter+"-{0:03}.dat".format(i) for i in range(32)]
    file_list.append("water.dat")
    file_list.append("sugar.dat")
    file_list.append("nacl.dat")
    file_list.append("CA.dat")
    file_list.append("board-DIN-00.dat")
    file_list.append("board-DIN-01.dat")
    file_list.append("board-DIN-02.dat")
    file_list.append("board-DIN-03.dat")

    for file_name in (file_list):
        if os.path.exists(first_dir + "//" + file_name):
            print("working on file {}".format(file_name))
            with open(out_dir + "//" + file_name, "wb") as out_file:

                with open(first_dir + "//" + file_name, "rb") as in_file_first:
                    i = 0
                    print("working getting data from first file")
                    piece = in_file_first.read(4096)
                    in_file_first.seek(0)

                    while len(piece) > 4095:
                        i+=1
                        piece = in_file_first.read(4096)

                        if piece == "":
                            break # end of file

                        out_file.write(piece)
                        if i % 50000 == 0:
                            print("wrote {} Gbs".format(i/262144))

                with open(second_dir + "//" + file_name, "rb") as in_file_second:
                    print("working getting data from second file")
                    i = 0
                    piece = in_file_second.read(4096)
                    in_file_second.seek(0)

                    while len(piece) > 4095:
                        i+=1
                        piece = in_file_second.read(4096)

                        if piece == "":
                            break # end of file

                        out_file.write(piece)
                        if i % 50000 == 0:
                            print("wrote {} Gbs".format(i/262144))


def turn_rhd_to_dat_full_directory(directory=None):
    """
    turns a full folder of rhd files into dat files
    :param directory: if None will go over the files in the same directory, if given it will use the given dir
    :return:
    """
    # create new directory to write to

    if directory is not None:
        os.chdir(directory)
    else:
        directory = os.getcwd()
    os.system('md dat_files')

    # get files
    files = glob.glob('*.rhd')
    files.sort(key = lambda x: x[-17:-4])

    # get amp names to write files
    amp_names,din_names = get_amp_names(files[0])

    # open files to write to
    write_files = [open('dat_files//amp-{}.dat'.format(amp), 'wb') for amp in amp_names]
    write_files_din = [open('dat_files//{}.dat'.format(din), 'wb') for din in din_names]

    print("found {} RHD files".format(len(files)))
    print("found the following AMP data files: {}".format(amp_names))
    print("found the following DIN data files: {}".format(din_names))

    # go over rhd files and write to dats
    for file in tqdm.tqdm(files):
        full_amp_data,full_din_data = get_amp_din_data(file)

        # go over amp data and write to relevant file
        for j in range(len(amp_names)):
            data_in_bytes = turn_amplifier_arr_to_bytes(full_amp_data[j])

            # write bytes to the file
            for by in data_in_bytes:
                write_files[j].write(by)

        # go over DIN data and write to relevant file
        for j in range(len(din_names)):
            data_in_bytes = turn_amplifier_arr_to_bytes(full_din_data[j])

            # write bytes to the file
            for by in data_in_bytes:
                write_files_din[j].write(by)

    for file in write_files:
        file.close()
    for file in write_files_din:
        file.close()

def run_klusta(base_file_name='amp-A-',start_val=0,stop_val=32,move_files=False,CAR=False):

    assert isinstance(base_file_name,str), "base file name has to be a string"
    if isinstance(start_val, list):
        prm_file_list = [base_file_name + "{0:03}".format(i) + '.prm' for i in start_val]
    else:
        prm_file_list = [base_file_name + "{0:03}".format(i) + '.prm' for i in range(start_val,stop_val)]
    mother_folder_path = os.getcwd()

    if CAR and len(prm_file_list)>2:
        # create and open the files for read / write
        OFL = [open(f[:-4] + '.dat', 'r+b') for f in prm_file_list]
        write_files = [open(f[:-4] + '_new.dat', 'wb') for f in prm_file_list]

        # get first bytes to initialize while loop
        bytes = [f.read(2) for f in OFL]
        for f in OFL:
            f.seek(0)

        # iterate over 2 bytes each time and write them minus the average to new file.
        while len(bytes[0]) > 1:
            bytes = [f.read(2) for f in OFL]
            ints = [int.from_bytes(by, 'little', signed=True) for by in bytes]
            mean_val = int(np.mean(ints))
            ints_to_write = [num - mean_val for num in ints]
            bytes_to_write = [int.to_bytes(num, 2, 'little', signed=True) for num in ints_to_write]
            for i, f in enumerate(write_files):
                f.write(bytes_to_write[i])

        # close the files
        for file in OFL:
            file.close()
        for file in write_files:
            file.close()

        # change prm files to new format
        for file in prm_file_list:
            for i, line in enumerate(fileinput.input(file, inplace=1)):
                exp_name = file[:-4]
                sys.stdout.write(line.replace(exp_name, exp_name + '_new'))

        # move the files and activate klusta
        for file in prm_file_list:
            dir_name = file[:-4]
            os.system('md ' + dir_name)
            os.system('move ' + file + ' ' + dir_name)
            os.system('move ' + file[:-4] + '_new.dat ' + dir_name)
            os.system('xcopy 1chan28.prb ' + dir_name)
            os.chdir(dir_name)
            os.system('klusta ' + file)
            os.chdir('..')

    else:
        if CAR:
            raw_input = str(input(
                'not enough electrodes to use C.A.R - would you like to continue with Non-CAR sorting? (y/n)\nNote: make sure to input as string - with "" on both sides.\n')).lower()
            while raw_input not in ('yes' or 'no'):
                raw_input = str(input('Invalid input!\nnot enough electrodes to use C.A.R - would you like to continue with Non-CAR sorting? (y/n)\nNote: make sure to input as string - with "" on both sides.\n')).lower()
            if raw_input in 'no':
                print('Exiting')
                return
        print('starting sorting')
        for file in prm_file_list:
            dir_name = file[:-4]
            os.system('md ' + dir_name)
            os.system('move ' + file + ' ' + dir_name)
            os.system('move ' + file[:-4] + '.dat ' + dir_name)
            os.system('xcopy 1chan28.prb ' + dir_name)
            os.chdir(dir_name)
            os.system('klusta ' + file)
            os.chdir('..')
        if move_files:
            move_kwiks(base_file_name,start_val,stop_val)

    return

def move_kwiks(base_file_name='amp-A-',start_val=0,stop_val=32):
    if isinstance(start_val, list):
        file_list = [base_file_name + "{0:03}".format(i) for i in start_val]
    else:
        file_list = [base_file_name + "{0:03}".format(i) for i in range(start_val, stop_val)]
    file_formats = ['.dat','.kwik','.kwx']
    os.system('md kwiks')
    for name in file_list:
        os.chdir(name)
        for file_format in file_formats:
            os.system('move ' + name + file_format + ' ..\kwiks')
        os.chdir('..')
    return

def undersample(in_file,out_file,undersample_factor=[10,10],lowpass=None,FS=30000):
    """
    :param in_file: the file to be undersampled
    :param out_file: the output file name
    :param undersample_factor: if you want to undersample by a factor of 10 or less use an integer. if you want to
    undersample by a factor between 11-100 use a list or tuple of 2 ints, example: use [5,6] to undersample by 30.
    :return:
    """
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

                if lowpass is not None:
                    filtered_twice = butter_bandpass_filter(lowcut=5,highcut=lowpass,data=filtered_twice,fs=FS)

                filtered_twice.astype('int16').tofile(out_f)
                data = np.fromfile(in_f,dtype=np.int16,count=3000000)


def run_kwik_gui(base_file_name='amp-A-',start_val=0,stop_val=32):
    """
    :param base_file_name: 'amp-X-' where X is the relevant letter from the intan recording station.
    :param start_val: electrode to start from if its an integer, if its a list the function goes through the list and
    no stop is needed.
    :param stop_val: if start is not a list stop-1 will be the the electrode to stop at.
    :return: nothing
    """
    if isinstance(start_val,list):
        file_list = [base_file_name + "{0:03}".format(i) for i in start_val]
    else:
        file_list = [base_file_name + "{0:03}".format(i) for i in range(start_val, stop_val)]
    for name in file_list:
        os.system('phy kwik-gui ' + name + '\\' + name + '.kwik')
    return


def analyze_movement_from_vid(vid, output_file=None):
    """
    :param vid: input file for analysis
    :param output_file: output file name, if None will have same name as video.
    :return: does not return anything, but saves a .npy file with the result
    """
    if output_file is None:
        output_file = vid[:-4]

    arr = []
    vidcap = cv2.VideoCapture(vid)
    success, last_frame = vidcap.read()
    last_frame = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
    counter = 0
    while success:
        counter += 1
        if counter % 10000 == 0:
            hours = counter // 108000
            minutes = (counter % 108000) // 1800
            secs = ((counter % 108000) % 1800) // 30
            print("{}:{}:{} finished".format(hours, minutes, secs))
        success, new_frame = vidcap.read()
        if success:
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(last_frame, new_frame)
            arr.append(np.mean(diff))
            last_frame = new_frame
    np.save(output_file, arr)

# def get_params_from_file(file):
#     with open(file, 'r') as param_file:
#         params = yaml.safe_load(param_file)
#     return params