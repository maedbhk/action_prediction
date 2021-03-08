import numpy as np
import pandas as pd
import os
import glob 
import click
from pathlib import Path

from eye_tracking.preprocessing.functions.et_preprocess import preprocess_et
from eye_tracking.preprocessing.functions.detect_events import make_fixations, make_blinks, make_saccades

import warnings
warnings.filterwarnings("ignore")

def preprocess_eye(subj_fpath):
    """ preprocess eye tracking data using code from https://github.com/teresa-canasbajo/bdd-driveratt/tree/master/eye_tracking/preprocessing
        saves out preprocessed data for events, saccades, fixations
            Args: 
                subj_fpath (str): full path to top-level directory of eye-tracking
    """
    # get all sessions
    ses_dirs = glob.glob(os.path.join(subj_fpath, '*ses*'))

    # get subj name
    subj = Path(subj_fpath).name

    # loop over sessions
    for ses_dir in ses_dirs:

        # get sess name
        sess = Path(ses_dir).name

        # get all runs
        run_dirs = glob.glob(os.path.join(ses_dir, '*'))

        # loop over runs
        for run_dir in run_dirs:
            
            # get run name
            run = Path(run_dir).name

            # get preprocess dir
            preprocess_dir = os.path.join(run_dir, 'preprocessed')

            # check if data have already been preprocessed
            if not os.path.isdir(preprocess_dir):  
                try: 
                    data = preprocess_et(subject='', datapath=run_dir, surfaceMap=False, eventfunctions=(make_fixations, make_blinks, make_saccades))
                    
                    # modify the msgs and save to disk
                    msgs = _modify_msgs(dataframe=data[1])
                    msgs.to_csv(os.path.join(preprocess_dir, f'{subj}_{sess}_{run}_pl_msgs.csv'))

                    # merge msgs to events and save to disk
                    events_msgs = _merge_msgs_events(events=data[2], msgs=msgs)
                    events_msgs.to_csv(os.path.join(preprocess_dir, f'{subj}_{sess}_{run}_pl_msgs_events.csv'))
                    print('Preprocessing complete!')
                except: 
                    print('something went wrong with preprocessing ...')
            else:
                print('These data have already been preprocessed ...')

def group_data(data_dir):
    # navigate to top-level subj dir
    os.chdir(data_dir)

    # get all preprocessed files
    fpaths = set(glob.glob('**/**/*msgs_events*', recursive=True))

    # loop over files
    df_all = pd.DataFrame()
    for fpath in list(fpaths):

        # read in dataframe
        df = pd.read_csv(fpath)

        # add subj, sess, run
        df['subj'] = Path(fpath).name.split('_')[0]
        df['sess'] = Path(fpath).name.split('_')[1]
        df['run'] = Path(fpath).name.split('_')[2]

        # load group dataframe if it exists and add results
        group_fname = os.path.join(data_dir, 'group_eyetracking.csv')

        # concat to group dataframe
        df_all = pd.concat([df_all, df])
        print(f'saving {fpath} to group dataframe')

    # save out group dataframe
    df_all.to_csv(group_fname, index=False)

def _extract_eventtype(x):
    if 'starting instructions' in x:
            return 'instructions'
    elif 'starting task' in x:
        return 'task_start'
    elif 'ending task' in x:
        return 'task_end'
    else:
        return None

def _merge_msgs_events(events, msgs):
    cols = ['start_time', 'duration', 'end_time']
    for col in cols:
        events[col] = events[col].astype(int)

    events_msgs = events.merge(msgs, left_on='start_time', right_on='timestamp')

    return events_msgs

def _modify_msgs(dataframe):
    dataframe['onset_sec'] = dataframe['exp_event'].str.extract('onset\:(\d+)').astype(int) + 1
    dataframe['block'] = dataframe['exp_event'].str.extract('block\:(\d+)')
    dataframe['task'] = dataframe['exp_event'].str.extract('for\s(\w*)')
    dataframe['event_type'] = dataframe['exp_event'].apply(lambda x: _extract_eventtype(x))
    dataframe['run'] = dataframe['task'].tail(1).str.extract('(\d+)').values.astype(int)[0][0]
    dataframe[dataframe['task']=='run_01']['task'] = None
    dataframe = dataframe.drop('trial', axis=1)

    # filter dataframe, remove task end
    dataframe = dataframe.query('event_type!="task_end"')

    # get missing timestamps
    dataframe['timestamp'] = dataframe['timestamp'].astype(int)
    dataframe['time_diff'] = dataframe['timestamp'].diff(periods=-1)*-1
    dataframe = dataframe.dropna()
    dataframe['time_diff'] = dataframe['time_diff'].astype(int)
    idx = dataframe['time_diff'].tail(1).index
    dataframe['time_diff'].loc[idx] = dataframe['time_diff'].tail(1)

    # make new dataframe
    dataframe_expand = pd.DataFrame()
    for col in dataframe:
        dataframe_expand[col] = dataframe[col].repeat(np.array(dataframe['time_diff'].values)) #dataframe[col].repeat(list(dataframe['time_diff'].values))

    # expand timestamp and onset sec
    dataframe_expand['timestamp'] = np.arange(dataframe['timestamp'].head(1).values, dataframe['timestamp'].tail(1).values + dataframe['time_diff'].tail(1).values)
    dataframe_expand['onset_sec'] = np.arange(dataframe['onset_sec'].head(1).values, dataframe['onset_sec'].tail(1).values + dataframe['time_diff'].tail(1).values)

    # drop time diff
    dataframe_expand = dataframe_expand.drop('time_diff', axis=1)
    return dataframe_expand

@click.command()
@click.option("--data_dir")

def run(data_dir):
    # subj dirs
    subj_dirs = glob.glob(os.path.join(data_dir, '*'))

    for subj_fpath in subj_dirs:
        if Path(subj_fpath).name not in ['sON', 'sEA', 'sAA1', 's01']:
            preprocess_eye(subj_fpath=subj_fpath)

    group_data(data_dir)

if __name__ == '__main__':
    run()