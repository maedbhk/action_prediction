import numpy as np
import pandas as pd
import os

from action_prediction import constants as const

class DataSet:

    def __init__(self, task='social_prediction'):
        self.task = task
        self.dispsize = (1280, 720)

    def load_eye(self, data_type='events'):
        """loads preprocessed eyetracking data 
            Args:   
                data_type (str): 'events' or 'samples'
            Returns: 
            filtered eyetracking data
        """
        eye_fname = f'group_eyetracking_{data_type}.csv'
        df = pd.read_csv(os.path.join(const.EYE_DIR, eye_fname))
        df_filtered = self.filter_eye(dataframe=df)
        return df_filtered

    def load_behav(self):
        """
        loads preprocessed behavioral data
        """
        behav_fname = 'group_behavior.csv'
        df = pd.read_csv(os.path.join(const.BEHAVE_DIR, behav_fname))
        df_filtered = self.filter_behav(dataframe=df)  
        return df_filtered  

    def filter_behav(self, dataframe):
        dataframe = dataframe.query(f'task=="{self.task}" and balance_exp=="behavioral"').dropna(axis=1)
        dataframe['sess'] = dataframe['sess'].div(2).astype(int)

        # drop some cols
        dataframe.drop(['timestamp', 'timestamp_sec', 'run_num_new', 'run_name'], axis=1, inplace=True)
        return dataframe

    def filter_eye(self, dataframe):
        """filter eyetracking dataframe
            Args: 
                dataframe (pd dataframe): preprocessed eyetracking data
        """
        dataframe.loc[dataframe.sess==2, 'run_num'] = dataframe.loc[dataframe.sess==2, 'run_num']+7

        # filter fixation, saccade, blinks
        fix = self._process_fixations(dataframe=dataframe)
        sacc = self._process_saccades(dataframe=dataframe)
        blink = dataframe.query('type=="blink"')
        dataframe = pd.concat([fix, sacc, blink])

        # # rescale fixations
        dataframe = self._rescale_fixations(dataframe=dataframe)
        
        # filter data to only include specific task
        dataframe = dataframe.query(f'task=="{self.task}" and event_type!="instructions"')
        return dataframe

    def merge_behav_eye(self, dataframe_behav, dataframe_eye):
        """merges behavioral and eyetracking dataframes on common keys

            Args: 
                dataframe_behav (pd dataframe):
                datarame_eye (pd dataframe): 
            Returns: 
                pd dataframe
        """
        trial_start_times = dataframe_behav['start_time'].unique()
        dataframe_eye['start_time'] = self._find_nearest(arr1=trial_start_times, arr2=dataframe_eye['onset_sec'])
        dataframe = dataframe_eye.merge(dataframe_behav, on=['task', 'subj', 'run_num', 'sess', 'block_iter', 'start_time'], how='inner').dropna(axis=1)
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
        
    def _rescale_fixations(self, dataframe):
        x = dataframe['mean_gx']
        dataframe['mean_gx'] = np.interp(x, (x.min(), x.max()), (0, self.dispsize[0]))

        y = dataframe['mean_gy']
        dataframe['mean_gy'] = np.interp(y, (y.min(), y.max()), (0, self.dispsize[1]))

        return dataframe