__author__ = 'elor'

import os
# import yaml

def run_klusta(base_file_name='amp-A-',start_val=0,stop_val=32,move_files=False):

    assert isinstance(base_file_name,str), "base file name has to be a string"
    if isinstance(start_val, list):
        prm_file_list = [base_file_name + "{0:03}".format(i) for i in start_val]
    else:
        prm_file_list = [base_file_name + "{0:03}".format(i) + '.prm' for i in range(start_val,stop_val)]
    mother_folder_path = os.getcwd()
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

def run_kwik_gui(base_file_name='amp-A-',start_val=0,stop_val=32):
    if isinstance(start_val,list):
        file_list = [base_file_name + "{0:03}".format(i) for i in start_val]
    else:
        file_list = [base_file_name + "{0:03}".format(i) for i in range(start_val, stop_val)]
    for name in file_list:
        os.system('phy kwik-gui ' + name + '\\' + name + '.kwik')
    return

# def get_params_from_file(file):
#     with open(file, 'r') as param_file:
#         params = yaml.safe_load(param_file)
#     return params