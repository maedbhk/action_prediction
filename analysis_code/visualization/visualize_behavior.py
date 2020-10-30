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

def _rename_sess(x, sess_type):
    defaults = Defaults()

    if sess_type=='fmri':
        remap = np.arange(3)
        for i, key in enumerate(defaults.fmri_sessions):
            if x in defaults.fmri_sessions[key]:
                return int(key.split('-')[1]) + remap[i]
    elif sess_type=='behavioral':
        for key in defaults.behavioral_sessions:
            if x in defaults.behavioral_sessions[key]:
                return int(key.split('-')[1])*2

def _rename_run(dataframe):
    defaults = Defaults()

    num_sess = len(defaults.behavioral_sessions) + len(defaults.fmri_sessions)

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

def _load_task_dataframe(task='social_prediction', sessions=['behavioral']):
    # initialise defaults
    defaults = Defaults()

    # load tasks for each subj, concat in one dataframe
    df_all = pd.DataFrame()
    for subj in defaults.subj_id:
        for sess in sessions:
            # initialise dir class
            dirs = Dirs(session=sess)
            fpath = os.path.join(dirs.BEHAVE_DIR, subj, f'{sess}_{subj}_{task}.csv')
            if os.path.exists(fpath):
                df = pd.read_csv(fpath)
                df['session_type'] = sess
                df['subj_id'] = subj
                df['session_num'] = df['run_num'].apply(lambda x: _rename_sess(x, sess_type=sess))
                df_all = pd.concat([df_all, df], sort=False)
    
    # rename runs
    df_all = _rename_run(dataframe=df_all)

    return df_all

def _load_run_dataframe():
    # initialise defaults
    defaults = Defaults()

    # initialise dir class
    dirs = Dirs()

    # initialise defaults
    defaults = Defaults()

    # load run file for each subj, concat in one dataframe
    df_all = pd.DataFrame()
    for subj in defaults.subj_id:
        fpath = os.path.join(dirs.BEHAVE_DIR, subj, f'fmri_{subj}.csv')
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            df['subj_id'] = subj

            df_all = pd.concat([df_all, df], sort=False)

    return df_all

def plot_behavior_acc(x='run_num_new', y='corr_resp', hue='condition_name', hue_order=['easy', 'hard']):

    # set up plot
    plt.clf()
    _plotting_style()

    x_pos = -0.2
    y_pos = 1.02

    nrows = 2
    ncols = 3

    fig = plt.figure(figsize=(10,10));
    gs = GridSpec(nrows, ncols, figure=fig);

    # initialise defaults
    defaults = Defaults()

    idx = 0
    for row in range(nrows):
        for col in range(ncols):

            # define subplot
            ax = fig.add_subplot(gs[row, col])

            df = _load_task_dataframe(task=defaults.tasks[idx])

            # plot data
            sns.lineplot(x=x, y=y, hue=hue, data=df, hue_order=hue_order, ax=ax)

            # plot session info if x axis is run_num
            if x=='run_num_new':
                sess_run = df.groupby('session_num').apply(lambda x: x['run_num_new'].min())
                if len(sess_run)==5:
                    linestyles = ['solid', 'dashed', 'solid', 'dashed', 'solid']
                else:
                    linestyles = np.tile('dashed', len(sess_run))
                for i, s in enumerate(sess_run):
                    plt.axvline(s, 0, 1, color='k', linestyle=linestyles[i])

            ax.text(x_pos + 0.3, 1.1, defaults.tasks[idx], transform=ax.transAxes, fontsize=15,
            verticalalignment='top')
            plt.legend(fontsize=15)
            plt.ylabel(y, fontsize=20)
            plt.xlabel('Run', fontsize=20)

            idx = idx + 1
            if idx==len(defaults.tasks):
                break
    fig.tight_layout(pad=2.0)
    plt.show()

def plot_behavior_rt(x='run_num_new', y='rt', hue='condition_name', hue_order=['easy', 'hard']):

    # set up plot
    plt.clf()
    _plotting_style()

    x_pos = -0.2
    y_pos = 1.02

    nrows = 2
    ncols = 3

    fig = plt.figure(figsize=(10,10));
    gs = GridSpec(nrows, ncols, figure=fig);

    # initialise defaults
    defaults = Defaults()

    idx = 0
    for row in range(nrows):
        for col in range(ncols):

            # define subplot
            ax = fig.add_subplot(gs[row, col])

            df = _load_task_dataframe(task=defaults.tasks[idx])
            
            # get only correct trials for rt
            df = df[df['corr_resp']==1]

            # plot data
            sns.lineplot(x=x, y=y, hue=hue, data=df, hue_order=hue_order, ax=ax)

            # plot session info if x axis is run_num
            if x=='run_num_new':
                sess_run = df.groupby('session_num').apply(lambda x: x['run_num_new'].min())
                if len(sess_run)==5:
                    linestyles = ['solid', 'dashed', 'solid', 'dashed', 'solid']
                else:
                    linestyles = np.tile('dashed', len(sess_run))
                for i, s in enumerate(sess_run):
                    plt.axvline(s, 0, 1, color='k', linestyle=linestyles[i])

            ax.text(x_pos + 0.3, 1.1, defaults.tasks[idx], transform=ax.transAxes, fontsize=15,
            verticalalignment='top')
            plt.legend(fontsize=15)
            plt.ylabel(y, fontsize=20)
            plt.xlabel('Run', fontsize=20)

            idx = idx + 1
            if idx==len(defaults.tasks):
                break
    fig.tight_layout(pad=2.0)
    plt.show()

def check_ttl():

    # initialise defaults
    defaults = Defaults()

    df = _load_run_dataframe()

    print(df[['block_name','real_start_time', 'real_end_time', 'ttl_counter', 'ttl_time', 'run_num']])

    return df[['block_name','real_start_time', 'real_end_time', 'ttl_counter', 'ttl_time', 'run_num']]



 