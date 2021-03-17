from pathlib import Path
import os

"""
Constants and defaults for running and visualizing connectivity models for the cerebellum learning project

@author: Maedbh King
"""

subj_id = {'sON':'sub-01', 'sEA':'sub-02', 
                'sAA1':'sub-03', 'sHU':'sub-04', 
                'sEH': 'sub-05', 'sSE': 'sub-06', 
                'sEO': 'sub-07', 'sAA': 'sub-08', 
                'sEE': 'sub-09', 'sML': 'sub-10',
                'sIU1': 'sub-11', 'sIU': 'sub-12'}
behavioral_sessions = {'ses-1':[1,2,3,4,5,6,7], 'ses-2':[8,9,10,11,12,13,14]}
tasks = ['semantic_prediction', 'visual_search', 'action_observation', 'social_prediction', 'n_back']

# 
image_size = (1280,720)

BASE_DIR = Path(__file__).absolute().parent.parent / 'data'
BEHAVE_DIR = BASE_DIR / 'behavior'
EYE_DIR = BASE_DIR / 'eyetracking'

