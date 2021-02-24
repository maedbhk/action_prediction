import numpy as np
import pandas as pd
import os

from action_prediction import constants as const

class DataSet:
    def __init__(task='social_prediction'):
        self.task = task

def load_eye():
    eye_fname = 'group_eyetracking.csv'
    df = pd.read_csv(os.path.join(const.EYE_DIR, eye_fname))

    return filter_eye(dataframe=df)

def load_behav():
    behav_fname = 'group_behavior.csv'
    df = pd.read_csv(os.path.join(const.BEHAVE_DIR, behav_fname))

    return filter_behav(dataframe=df)  

def filter_behav(dataframe):
    dataframe = dataframe.query(f'task=="{task}" and balance_exp=="behavioral"').dropna(axis=1)
    
    dataframe['sess'] = dataframe['sess'].div(2).astype(int)
    return dataframe

def filter_eye(dataframe):
    # recode some variables (easier for visualization)
    dataframe['sess'] = dataframe['sess'].str.extract('(\d+)').astype(int)
    dataframe['run_num'] = dataframe['run'] + 1
    dataframe.loc[dataframe.sess==2, 'run_num'] = dataframe.loc[dataframe.sess==2, 'run_num']+7

    # filter fixation, saccade, blinks
    fix = _process_fixations(dataframe=dataframe)
    sacc = _process_saccades(dataframe=dataframe)
    blink = dataframe.query('type=="blink"')
    dataframe = pd.concat([fix, sacc, blink])
    
    # filter data to only include specific task
    dataframe = dataframe.query(f'task=="{task}" and event_type!="instructions"')
    return dataframe

def merge_behav_eye(dataframe1, dataframe2):
    dataframe = dataframe1.merge(dataframe2, on=['subj', 'run_num', 'sess', 'task'])
    cols_to_keep = [col for col in dataframe.columns if 'Unnamed' not in col]
    return dataframe[cols_to_keep]

def _distance(row, center):
    return np.sqrt((row['mean_gx']-center[0])**2+(row['mean_gy']-center[1])**2)

def _process_fixations(dataframe):
    df = dataframe[dataframe['type']=='fixations']
    df = df.query('mean_gx>=0 and mean_gx<=1 and mean_gy>=0 and mean_gy<=1')
    duration = np.mean(df['duration'])
    center = (np.mean(df['mean_gx']), np.mean(df['mean_gy']))
    df['dispersion'] = df.apply(_distance, args=[center], axis=1)
    return df

def _process_saccades(dataframe):
    return dataframe.query(f'type=="saccade" and amplitude<1')
    
def _rescale_fixations(dataframe, dispsize):
    x = dataframe['mean_gx']
    dataframe['mean_gx'] = np.interp(x, (x.min(), x.max()), (0, dispsize[0]))

    y = dataframe['mean_gy']
    dataframe['mean_gy'] = np.interp(y, (y.min(), y.max()), (0, dispsize[1]))

    return dataframe