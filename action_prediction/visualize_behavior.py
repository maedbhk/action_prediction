import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from matplotlib.gridspec import GridSpec

from action_prediction import constants as const

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

def plot_acc():
    pass
    