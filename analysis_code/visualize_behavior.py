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

def plot_behavior_acc(dataframe, x='run_num', y='corr_resp', hue='condition_name'):

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


 