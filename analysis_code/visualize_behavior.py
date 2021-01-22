import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from analysis_code.constants import Dirs, Defaults

import warnings
warnings.filterwarnings("ignore")

def _plotting_style():
    # fig = plt.figure(num=2, figsize=[20,8])
    plt.style.use('seaborn-poster') # ggplot
    plt.rc('font', family='sans-serif') 
    plt.rc('font', serif='Helvetica Neue') 
    plt.rc('text', usetex='false') 
    plt.rcParams['lines.linewidth'] = 2
    plt.rc('xtick', labelsize=14)     
    plt.rc('ytick', labelsize=14)
    plt.rcParams.update({'font.size': 14})
    plt.rcParams["axes.labelweight"] = "regular"
    plt.rcParams["font.weight"] = "regular"
    plt.rcParams["savefig.format"] = 'svg'
    plt.rc("axes.spines", top=False, right=False) # removes certain axes

def _rename_sess(x):
    defaults = Defaults()

    for key in defaults.behavioral_sessions:
        if x in defaults.behavioral_sessions[key]:
            return int(key.split('-')[1])

def _rename_run(dataframe):
    defaults = Defaults()

    num_sess = len(defaults.behavioral_sessions)

    sess_dict = {1: 0}
    for sess in np.arange(1,num_sess):
        sess_dict.update({sess+1: np.unique(dataframe[dataframe['session_num']==sess]['run_num']).max()})
    #     sess_idx.append(np.unique(dataframe[dataframe['session_num']==sess]['run_num']).max())

    df_all = pd.DataFrame()
    grouped = dataframe.groupby('session_num')
    for name, group in grouped:
        group['run_num_new'] = group['run_num'] + sess_dict[name]
        df_all = pd.concat([df_all, group])

    return df_all

def _load_task_dataframe(task='social_prediction', sessions='behavioral'):
    # initialise defaults
    defaults = Defaults()

    dirs = Dirs()

    # load tasks for each subj, concat in one dataframe
    df_all = pd.DataFrame()
    for subj in defaults.subj_id:
        
        fpath = os.path.join(dirs.BEHAVE_DIR, subj, f'behavioral_{subj}_{task}.csv')
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            df['subj_id'] = subj
            df['session_num'] = df['run_num'].apply(lambda x: _rename_sess(x))
            df_all = pd.concat([df_all, df], sort=False)

    return df_all

def _load_run_dataframe():
    # initialise defaults
    defaults = Defaults()

    # initialise dir class
    dirs = Dirs()

    # load run file for each subj, concat in one dataframe
    df_all = pd.DataFrame()
    for subj in defaults.subj_id:
        fpath = os.path.join(dirs.BEHAVE_DIR, subj, f'fmri_{subj}.csv')
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            df['subj_id'] = subj

            df_all = pd.concat([df_all, df], sort=False)

    return df_all

def plot_behavior_acc(dataframe, x='run_num', y='corr_resp', hue='condition_name']):

    # initialise defaults
    defaults = Defaults()

    # plot data
    sns.lineplot(x=x, y=y, hue=hue, data=dataframe)

    plt.legend(fontsize=15)
    plt.ylabel(y, fontsize=20)
    plt.xlabel('Run', fontsize=20);

    plt.show()

def plot_behavior_rt(dataframe, x='run_num', y='rt', hue='condition_name'):

    # initialise defaults
    defaults = Defaults()

    # plot data
    sns.lineplot(x=x, y=y, hue=hue, data=dataframe[dataframe['corr_resp']==1])

    plt.legend(fontsize=15)
    plt.ylabel(y, fontsize=20)
    plt.xlabel('Run', fontsize=20)

    plt.show()

def check_ttl():

    # initialise defaults
    defaults = Defaults()

    df = _load_run_dataframe()

    print(df[['block_name','real_start_time', 'real_end_time', 'ttl_counter', 'ttl_time', 'run_num']])

    return df[['block_name','real_start_time', 'real_end_time', 'ttl_counter', 'ttl_time', 'run_num']]



 