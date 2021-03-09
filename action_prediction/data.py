import numpy as np
import pandas as pd
import os

from action_prediction import constants as const

class DataSet:

    def __init__(self, task='social_prediction'):
        self.task = task

    def load_eye(self):
        eye_fname = 'group_eyetracking.csv'
        df = pd.read_csv(os.path.join(const.EYE_DIR, eye_fname))

        return self.filter_eye(dataframe=df)

    def load_behav(self):
        behav_fname = 'group_behavior.csv'
        df = pd.read_csv(os.path.join(const.BEHAVE_DIR, behav_fname))

        return self.filter_behav(dataframe=df)  

    def filter_behav(self, dataframe):
        dataframe = dataframe.query(f'task=="{self.task}" and balance_exp=="behavioral"').dropna(axis=1)
        dataframe['sess'] = dataframe['sess'].div(2).astype(int)

        # drop some cols
        dataframe.drop(['start_time', 'timestamp', 'timestamp_sec', 'run_num_new', 'run_name'], axis=1, inplace=True)
        # dataframe = dataframe.sort_values(by=['subj', 'run_num', 'block_iter'])
        return dataframe.reset_index(drop=True)

    def filter_eye(self, dataframe):
        # recode some variables (easier for visualization)
        # dataframe['sess'] = dataframe['sess'].str.extract('(\d+)').astype(int)
        # dataframe['run_num'] = dataframe['run'] + 1
        # dataframe.loc[dataframe.sess==2, 'run_num'] = dataframe.loc[dataframe.sess==2, 'run_num']+7
        
        # # drop some cols
        # dataframe.drop(['start_time', 'timestamp', 'run'], axis=1, inplace=True)

        # # do some cleaning
        # dataframe.rename({'block':'block_iter'}, axis=1, inplace=True)

        # filter fixation, saccade, blinks
        fix = self._process_fixations(dataframe=dataframe)
        sacc = self._process_saccades(dataframe=dataframe)
        blink = dataframe.query('type=="blink"')
        dataframe = pd.concat([fix, sacc, blink])

        # rescale fixations
        dataframe = self._rescale_fixations(dataframe=dataframe, dispsize=(1280, 720))
        
        # filter data to only include specific task
        dataframe = dataframe.query(f'task=="{self.task}"')

        # cols_to_group = ['run_num', 'block_iter', 'subj']
        # dataframe['onset_sec_corr'] = dataframe.groupby(cols_to_group).apply(
        #     lambda x: x['onset_sec']-x['onset_sec'].head(1).values[0]
        #     ).reset_index(drop=True).values
        # dataframe = dataframe.sort_values(by=['subj', 'run_num', 'block_iter'])
        return dataframe.reset_index(drop=True)

    def merge_behav_eye(self, dataframe_behav, dataframe_eye):
        """merges behavioral and eyetracking dataframes on common keys

            Args: 
                dataframe_behav (pd dataframe):
                datarame_eye (pd dataframe): 
            Returns: 
                pd dataframe
        """
        # add new start time for eyetracking
        dataframe_eye['real_start_time'] = self._find_nearest(dataframe_behav['real_start_time'], dataframe_eye['onset_sec_corr'])

        # merge behav with eyetracking data
        # dataframe = dataframe_eye.merge(dataframe_behav, on=['subj', 'run_num', 'block_iter', 'real_start_time'])
        dataframe = dataframe_eye
        cols_to_keep = [col for col in dataframe.columns if 'Unnamed' not in col]
        return dataframe[cols_to_keep]

    def _find_nearest(self, arr1, arr2):
        """ find value in arr1 that is closest 
        to given value from arr2
        
        Args: 
            arr1 (1d np array): ref array
            arr2 (1d np array): matching array
        Returns: 
            1d np array of nearest values from arr1 that match
            values from arr2
        """
        arr1 = np.asarray(arr1)
        arr2 = np.asarray(arr2)
        indices = []
        for val in arr2:
            indices.append((np.abs(arr1 - val)).argmin())
        return arr1[indices]
   
    def _distance(self, row, center):
        return np.sqrt((row['mean_gx']-center[0])**2+(row['mean_gy']-center[1])**2)

    def _process_fixations(self, dataframe):
        df = dataframe[dataframe['type']=='fixations']
        df = df.query('mean_gx>=0 and mean_gx<=1 and mean_gy>=0 and mean_gy<=1')
        duration = np.mean(df['duration'])
        center = (np.mean(df['mean_gx']), np.mean(df['mean_gy']))
        df['dispersion'] = df.apply(self._distance, args=[center], axis=1)
        return df

    def _process_saccades(self, dataframe):
        return dataframe.query(f'type=="saccade" and amplitude<1')
        
    def _rescale_fixations(self, dataframe, dispsize):
        x = dataframe['mean_gx']
        dataframe['mean_gx'] = np.interp(x, (x.min(), x.max()), (0, dispsize[0]))

        y = dataframe['mean_gy']
        dataframe['mean_gy'] = np.interp(y, (y.min(), y.max()), (0, dispsize[1]))

        return dataframe