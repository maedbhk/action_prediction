from pathlib import Path
import os

"""
Constants and defaults for running and visualizing connectivity models for the cerebellum learning project

@author: Maedbh King
"""

subj_id =  {'sON':'sub-01', 'sEA':'sub-02', 
            'sAA1':'sub-03', 'sHU':'sub-04', 
            'sEH': 'sub-05', 'sSE': 'sub-06', 
            'sEO': 'sub-07', 'sAA': 'sub-08', 
            'sEE': 'sub-09', 'sML': 'sub-10',
            'sIU1': 'sub-11', 'sIU': 'sub-12',
            'sRH': 'sub-13', 'sAA2': 'sub-14',
            'sAE': 'sub-15', 'sNH': 'sub-16',
            'sAO1': 'sub-17', 'sAO': 'sub-18',
            'sOA': 'sub-19', 'sLG': 'sub-20',
            'sRO': 'sub-21', 'sEI': 'sub-22'}
behavioral_sessions = {'ses-1':[1,2,3,4,5,6,7], 'ses-2':[8,9,10,11,12,13,14]}
tasks = ['semantic_prediction', 'visual_search', 'action_observation', 'social_prediction', 'n_back']

# 
image_size = (1280,720)

BASE_DIR = Path(__file__).absolute().parent.parent / 'data'
BEHAVE_DIR = BASE_DIR / 'behavior'
EYE_DIR = BASE_DIR / 'eyetracking'
FIG_DIR = Path(__file__).absolute().parent.parent / 'figures'

# create folders if they don't already exist
fpaths = [FIG_DIR]
for fpath in fpaths:
    if not os.path.exists(fpath):
        print(f'creating {fpath}')
        os.makedirs(fpath)

