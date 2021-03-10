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
                    msgs_df = pd.read_csv(os.path.join(preprocess_dir, 'pl_msgs.csv'))
                    msgs = _modify_msgs(dataframe=msgs_df)
                    msgs.to_csv(os.path.join(preprocess_dir, f'{subj}_{sess}_{run}_pl_msgs.csv'))

                    # merge msgs to events and save to disk
                    events_df = pd.read_csv(os.path.join(preprocess_dir, 'pl_events.csv'))
                    events_msgs = _merge_msgs_events(events=events_df, msgs=msgs)
                    events_msgs.to_csv(os.path.join(preprocess_dir, f'{subj}_{sess}_{run}_pl_msgs_events.csv'))

                    # merge msgs to samples and save to disk
                    samples_df = pd.read_csv(os.path.join(preprocess_dir, 'pl_samples.csv'))
                    samples_msgs = _merge_msgs_samples(samples=samples_df, msgs=msgs)
                    samples_msgs.to_csv(os.path.join(preprocess_dir, f'{subj}_{sess}_{run}_pl_msgs_samples.csv'))

                    print('Preprocessing complete!')
                except: 
                    print('something went wrong with preprocessing ...')
            else:
                print('These data have already been preprocessed ...')

def concat_runs(subj_fpath):
    # get all sessions
    ses_dirs = glob.glob(os.path.join(subj_fpath, '*ses*'))

    # get subj name
    subj = Path(subj_fpath).name

    df_events_all = pd.DataFrame()
    df_samples_all = pd.DataFrame()
    # loop over sessions
    for ses_dir in np.sort(ses_dirs):

        # get sess name
        sess = Path(ses_dir).name

        # get all runs
        run_dirs = glob.glob(os.path.join(ses_dir, '*'))

        # loop over runs
        for run_dir in np.sort(run_dirs):

            # get run name
            run = Path(run_dir).name

            # load preprocessed data for subj/sess
            try: 
                df_events = pd.read_csv(os.path.join(subj_fpath, sess, run, 'preprocessed', f'{subj}_{sess}_{run}_pl_msgs_events.csv'))
                df_events['subj'] = subj
                df_events['sess'] = sess

                df_samples = pd.read_csv(os.path.join(subj_fpath, sess, run, 'preprocessed', f'{subj}_{sess}_{run}_pl_msgs_samples.csv'))
                df_samples['subj'] = subj
                df_samples['sess'] = sess

                # clean up
                df_events = _clean_up(dataframe=df_events)
                df_samples = _clean_up(dataframe=df_samples)

                # concat to dataframe
                df_events_all = pd.concat([df_events_all, df_events])
                df_samples_all = pd.concat([df_samples_all, df_samples])
            except: 
                print(f'no preprocessed data for {subj}_{sess}_{run}')

    # clean up
    df_events_all.to_csv(os.path.join(subj_fpath, f'eyetracking_events_{subj}.csv'), index=False)
    df_samples_all.to_csv(os.path.join(subj_fpath, f'eyetracking_samples_{subj}.csv'), index=False)

def group_data(data_dir):
    # get all subjs
    subj_dirs = glob.glob(os.path.join(data_dir, '*s*'))

    # loop over subjs
    df_events_all = pd.DataFrame()
    df_samples_all = pd.DataFrame()
    for subj_dir in subj_dirs:
        if os.path.isdir(subj_dir):

            # get subj name
            subj = Path(subj_dir).name

            # load preprocessed data for subj/sess
            try: 
                df_events = pd.read_csv(os.path.join(subj_dir, f'eyetracking_events_{subj}.csv'))
                df_samples = pd.read_csv(os.path.join(subj_dir, f'eyetracking_samples_{subj}.csv'))

                # concat to dataframe
                df_events_all = pd.concat([df_events_all, df_events])
                df_samples_all = pd.concat([df_samples_all, df_samples])
            except: 
                print(f'no preprocessed data for {subj}')

        print(f'saving {subj} to group')

    df_events_all.to_csv(os.path.join(data_dir, f'group_eyetracking_events.csv'), index=False)
    df_samples_all.to_csv(os.path.join(data_dir, f'group_eyetracking_samples.csv'), index=False)
    print("done!")

def _clean_up(dataframe):
    dataframe['sess'] = dataframe['sess'].str.extract('(\d+)').astype(int)
    dataframe.rename({'block': 'block_iter', 'run': 'run_num'}, axis=1, inplace=True)
    
    return dataframe

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

    events.rename({'start_time': 'timestamp'}, axis=1, inplace=True)
    events_msgs = events.merge(msgs, on='timestamp')

    events_msgs['timestamp'] = events_msgs['timestamp'] - events_msgs['timestamp'].head(1).values

    # figure out onset sec of tasks
    cols_to_group = ['task', 'run', 'block', 'event_type']
    onsets = events_msgs.groupby(cols_to_group).apply(lambda x: x['timestamp'].iloc[0]).reset_index().rename({0:'subtract'}, axis=1)
    events_msgs = events_msgs.merge(onsets, on=['task' ,'run', 'block', 'event_type'])
    events_msgs['onset_sec'] = events_msgs['timestamp'] - events_msgs['subtract']

    return events_msgs

def _merge_msgs_samples(samples, msgs):
    samples['smpl_time'] = samples['smpl_time'].astype(int)

    samples.rename({'smpl_time': 'timestamp'}, axis=1, inplace=True)
    samples_msgs = samples.merge(msgs, on='timestamp')

    samples_msgs['timestamp'] = samples_msgs['timestamp'] - samples_msgs['timestamp'].head(1).values

    # figure out onset sec of tasks
    cols_to_group = ['task', 'run', 'block', 'event_type']
    onsets = samples_msgs.groupby(cols_to_group).apply(lambda x: x['timestamp'].iloc[0]).reset_index().rename({0:'subtract'}, axis=1)
    samples_msgs = samples_msgs.merge(onsets, on=['task' ,'run', 'block', 'event_type'])
    samples_msgs['onset_sec'] = samples_msgs['timestamp'] - samples_msgs['subtract']

    return samples_msgs

def _modify_msgs(dataframe):

    # normalize timestamp
    dataframe['timestamp'] = dataframe['timestamp'].astype(int)

    # dataframe['onset_sec'] = dataframe['exp_event'].str.extract('onset\:(\d+)').astype(int) + 1
    dataframe['block'] = dataframe['exp_event'].str.extract('block\:(\d+)')
    dataframe['task'] = dataframe['exp_event'].str.extract('for\s(\w*)')
    dataframe['event_type'] = dataframe['exp_event'].apply(lambda x: _extract_eventtype(x))
    dataframe['run'] = dataframe['task'].tail(1).str.extract('(\d+)').values.astype(int)[0][0]
    dataframe[dataframe['task']=='run_01']['task'] = None
    dataframe = dataframe.drop('trial', axis=1)

    # filter dataframe, remove task end
    dataframe = dataframe.query('event_type!="task_end"')

    # get missing timestamps
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

    # drop time diff
    dataframe_expand = dataframe_expand.drop({'time_diff'}, axis=1)
    return dataframe_expand

@click.command()
@click.option("--data_dir")

def run(data_dir):
    # subj dirs
    subj_dirs = glob.glob(os.path.join(data_dir, '*'))

    # loop over subjects
    for subj_fpath in subj_dirs:
        if Path(subj_fpath).name not in ['sON', 'sEA', 'sAA1', 's01']:
            if os.path.isdir(subj_fpath):
                preprocess_eye(subj_fpath=subj_fpath)
                concat_runs(subj_fpath=subj_fpath)

    # group subject data
    group_data(data_dir=data_dir)

if __name__ == '__main__':
    run()