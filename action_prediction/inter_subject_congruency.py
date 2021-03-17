import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy import stats

from action_prediction import constants as const

def grid_fixations(fixations):
    """
    Converts dataframe of fixations at x,y positions to datatable of number of fixations at grid positions 

    Args: 
        fixations = a table with columns x,y 
    Returns:
        table with columns grid_x, grid_y, num
    """
    grid_size=(const.image_size[0]//10, const.image_size[1]//10)
    fixations['grid_x'] = np.floor(fixations['x']*grid_size[0])
    fixations['grid_y'] = np.floor(fixations['y']*grid_size[1])
    fixations = fixations.groupby(['grid_x', 'grid_y']).size().reset_index(level=['grid_x', 'grid_y'], name='num')
    
    blank_data = []
    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            blank_data.append([x,y,0])
    blank = pd.DataFrame(blank_data, columns = ['grid_x', 'grid_y','num']) 
    
    fixations = fixations.append(blank, ignore_index=True)
    return fixations.groupby(['grid_x', 'grid_y']).sum().reset_index(level=['grid_x', 'grid_y'])

def grid_all_subjects(fixations):
    """grid fixations across subjects

    Args: 
        fixations (pd dataframe): must contain 'x', 'y', 'subj'
    Returns: 
        all_gridded_fixations (pd dataframe) for all subjs
    """
    grid_size=(const.image_size[0]//10, const.image_size[1]//10)
    #create grid table 
    grid_positions = []
    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            grid_positions.append([x,y])
    all_gridded_fixations = pd.DataFrame(grid_positions, columns = ['x', 'y']) 

    #fill in table with column for each subject
    for subj in fixations['subj'].unique():
        gridded_fixations = grid_fixations(fixations[fixations['subj']==subj])
        all_gridded_fixations[subj] = gridded_fixations['num']

    #add total column with the total number of fixations at that grid position
    col_list = list(all_gridded_fixations)
    col_list.remove('x')
    col_list.remove('y')
    all_gridded_fixations['total'] = all_gridded_fixations[col_list].sum(axis=1)
    
    return all_gridded_fixations

def ioc(gridded_data, thresholds):
    """Calculate inter-observer congruency

    Args: 
        gridded_data (pd dataframe): must contain subj cols, 'x', 'y', 'total'
        thresholds (list): example: [5, 10, 15, 20]
    Returns: 
        ioc_rates (pd dataframe): hit rate for each subj for each threshold
    """
    rates = pd.DataFrame([])
    subjs = [col for col in gridded_data.columns if 's' in col]
    for subj in subjs:
        for thresh in thresholds: 
            temp = pd.DataFrame()
            temp['subject'] = gridded_data[subj]
            temp['others_sum'] = gridded_data['total'] - gridded_data[subj]
            temp['binarized'] = temp['others_sum'] > thresh
            rate = sum(temp[temp['binarized']]['subject'])/sum(temp['subject'])
            rates = rates.append([[subj, rate, thresh]], ignore_index=True)
        
    ioc_rates = rates.rename(columns={0: "subject", 1: "hit_rate", 2:"threshold"})
    return ioc_rates

def fisher(r):
    return 0.5 * (np.log(1+r) - np.log(1-r))

def get_subj_fixations(dataframe, data_type):
    """get list of fixations for subjects
    
    Args: 
        dataframe (pd dataframe): must contain 'x', 'y' cols for all subjects
        data_type (str): 'events' or 'samples'
    """
    if data_type=="events":
        cols= {"mean_gx":"x", "mean_gy":"y"}
        filtered=dataframe[dataframe['type']=="fixations"].rename(columns=cols)
        fix = filtered[['x', 'y', 'duration', 'subj']]
    elif data_type=="samples":
        cols={"gx":"x", "gy":"y"}
        filtered =dataframe[dataframe['type']=="fixations"]
        filtered = filtered[filtered['confidence']>=0.8].rename(columns=cols).query('x>=0 and x<=1 and y>=0 and y<=1')
        fix = filtered[['x', 'y', 'subj']]
    return fix

def pairwise(gridded_data):
    scores = pd.DataFrame([])
    subjs = [col for col in gridded_data.columns if 's' in col]
    for subj in subjs:
        z_scores = []
        for other in subjs:
            if subj != other: 
                corr = np.corrcoef(gridded_data[subj], gridded_data[other])[0][1]
                z = fisher(corr)
                z_scores.append(z)
        avg_z = np.mean(z_scores)
        scores = scores.append([[subj, avg_z,stats.norm.cdf(avg_z)]])
    return scores.rename(columns={0:"observer", 1:"avrg_zscore", 2:"probability"})