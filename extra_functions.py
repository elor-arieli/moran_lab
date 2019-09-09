import numpy as np
import pandas as pd
import statsmodels.stats.anova as anova
from scipy.stats import f_oneway,ttest_ind
import statsmodels
import cv2
import matplotlib.pyplot as plt
from moran_lab.band_pass_filters import savitzky_golay
from tqdm import tqdm
from scipy.io import loadmat
import yaml

def analyze_movement_from_vid(vid, output_file=None):
    if output_file is None:
        output_file = vid[:-4]

    arr = []
    vidcap = cv2.VideoCapture(vid)
    success,last_frame = vidcap.read()
    last_frame = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
    counter = 0
    while success:
        counter+=1
        if counter%10000 == 0:
            hours = counter//108000
            minutes = (counter%108000)//1800
            secs = ((counter%108000)%1800)//30
            print("{}:{}:{} finished".format(hours, minutes,secs))
        success,new_frame = vidcap.read()
        if success:
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(last_frame, new_frame)
            arr.append(np.mean(diff))
            last_frame = new_frame
    np.save(output_file,arr)

def two_way_anova(data, independent_var1, independent_var2, dependent_var,dtype='df'):
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    if dtype not in ['df','csv']:
        print('data type must be either df (if given a pandas dataframe) or csv (if given a csv file)')
        return None
    if dtype=='df':
        df = data.dropna()
    else:
        with open(data,'r') as F:
            df = pd.read_csv(F)
            df = df.dropna()

    if ' ' in independent_var1:
            new_var = independent_var1.replace(' ', '_')
            df = df.rename(columns={independent_var1: new_var})
            independent_var1 = new_var
    if ' ' in independent_var2:
            new_var = independent_var2.replace(' ', '_')
            df = df.rename(columns={independent_var2: new_var})
            independent_var2 = new_var
    if ' ' in dependent_var:
            new_var = dependent_var.replace(' ', '_')
            df = df.rename(columns={dependent_var: new_var})
            dependent_var = new_var


    moore_lm = ols('{} ~ C({}, Sum)*C({}, Sum)'.format(dependent_var,independent_var1, independent_var2),data=df,eval_env=-1).fit()
    table = sm.stats.anova_lm(moore_lm, typ=2)
    print(table)

def two_way_repeated_measures_anova(data, subject_id_var, independent_var1, independent_var2, dependent_var, dtype='df'):
    if dtype not in ['df', 'csv']:
        print('data type must be either df (if given a pandas dataframe) or csv (if given a csv file)')
        return None
    if dtype == 'df':
        df = data.dropna()
    else:
        with open(data, 'r') as F:
            df = pd.read_csv(F)
            df = df.dropna()

    if ' ' in independent_var1:
        new_var = independent_var1.replace(' ', '_')
        df = df.rename(columns={independent_var1: new_var})
        independent_var1 = new_var
    if ' ' in subject_id_var:
        new_var = subject_id_var.replace(' ', '_')
        df = df.rename(columns={subject_id_var: new_var})
        subject_id_var = new_var
    if ' ' in independent_var2:
        new_var = independent_var2.replace(' ', '_')
        df = df.rename(columns={independent_var2: new_var})
        independent_var2 = new_var
    if ' ' in dependent_var:
        new_var = dependent_var.replace(' ', '_')
        df = df.rename(columns={dependent_var: new_var})
        dependent_var = new_var

    aovrm2way = anova.AnovaRM(df, dependent_var, subject_id_var, within=[independent_var1, independent_var2])
    res2way = aovrm2way.fit()

    print(res2way)

def one_way_anova(*args):
    return f_oneway(*args)

def ttest(a,b):
    return ttest_ind(a,b)