from pathlib import Path
import os

"""
Created on Oct 10 18:41:50
Constants and defaults for running and visualizing connectivity models for the cerebellum learning project

@author: Maedbh King
"""

class Defaults:

    def __init__(self):
        self.subj_id = {'sON':1, 'sEA':2, 'sAA1':3, 'sHU':4}
        self.behavioral_sessions = {'ses-1':[1,2,3,4,5,6,7], 'ses-2':[8,9,10,11,12,13,14]}
        self.tasks = ['semantic_prediction', 'visual_search', 'action_observation', 'social_prediction', 'n_back']

class Dirs: 

    def __init__(self, session='behavioral'):
        self.BASE_DIR = Path(__file__).absolute().parent.parent / 'data'
        self.BEHAVE_DIR = self.BASE_DIR / f'{session}_sessions' / 'behavior'
        self.EYE_DIR = self.BASE_DIR /'behavioral_sessions' / 'eyetracking'

        # create folders if they don't already exist
        # fpaths = [self.BETA_REG_DIR, self.CONN_TRAIN_DIR, self.CONN_EVAL_DIR, self.ATLAS]
        # for fpath in fpaths:
        #     if not os.path.exists(fpath):
        #         print(f'creating {fpath} although this dir should already exist, check your folder transfer!')
        #         os.makedirs(fpath)