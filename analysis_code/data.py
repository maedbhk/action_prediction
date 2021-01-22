import numpy as np
import pandas as pd
import os

from analysis_code.constants import Dirs, Defaults

def load_behavior():
    # initialize dirs
    dirs = Dirs()

    return pd.read_csv(os.path.join(dirs.BEHAVE_DIR, 'behavior-all-sessions.csv'))

def _filter_data(dataframe):
    fix = _process_fixations(dataframe=dataframe)
    sacc = _process_saccades(dataframe=dataframe)
    blink = dataframe.query('type=="blink"')
    return pd.concat([fix, sacc, blink])

def _process_saccades(dataframe):
    return dataframe.query(f'type=="saccade" and amplitude<1')

def _process_fixations(dataframe):
    df = dataframe[dataframe['type']=='fixations']
    df = df.query('mean_gx>=0 and mean_gx<=1 and mean_gy>=0 and mean_gy<=1')
    duration = np.mean(df['duration'])
    center = (np.mean(df['mean_gx']), np.mean(df['mean_gy']))
    df['dispersion'] = df.apply(_distance, args=[center], axis=1)
    return df

def _distance(row, center):
    return np.sqrt((row['mean_gx']-center[0])**2+(row['mean_gy']-center[1])**2)

def load_eyetracking():
    # initialise defaults
    defaults = Defaults() 
    
    dirs = Dirs()  

    # load data
    dataframe = pd.read_csv(os.path.join(dirs.EYE_DIR, 'eyetracking-all-sessions.csv'))

    # recode some variables (easier for visualization)
    dataframe['subj'] = dataframe['subj'].apply(lambda x: defaults.subj_id[x]).str.extract('(\d+)')
    dataframe['run'] = dataframe['run'] + 1
    dataframe['sess'] = dataframe['sess'].str.extract('(\d+)') 

    # filter events
    dataframe = _filter_data(dataframe)

    return dataframe 