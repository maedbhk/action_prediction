import numpy as np
import pandas as pd
import os

from action_prediction import constants as const

class DataSet:

    def __init__(self, task='social_prediction'):
        self.task = task
        self.dispsize = (1280, 720)

    def load_eye(self, data_type='events'):
        """ loads eyetracking data, either samples or events
        Args: 
            data_type (str): 'events' or 'samples'
        """
        fpath = os.path.join(const.EYE_DIR, f'group_eyetracking_{data_type}_{self.task}.csv')
        if os.path.isfile(fpath):
            df = pd.read_csv(fpath)
            return df
        else:
            return self.preprocess_eye(data_type=data_type, save=True)

    def load_behav(self):
        fpath = os.path.join(const.EYE_DIR, f'group_behavior_{self.task}.csv')
        if os.path.isfile(fpath):
            df = pd.read_csv(fpath)
            return df
        else:
            return self.preprocess_behav(save=True) 
    
    def preprocess_eye(self, data_type='events', save=False):
        """loads preprocessed eyetracking data 
            Args:   
                data_type (list): 'events' and/or 'samples'
            Returns: 
            filtered eyetracking data
        """
        eye_fname = f'group_eyetracking_{data_type}.csv'
        df = pd.read_csv(os.path.join(const.EYE_DIR, eye_fname))
        df_filtered = self._filter_eye(dataframe=df, data_type=data_type)

        df_filtered['corr_resp'] = df_filtered['corr_resp'].astype(float)

        # option to save to disk
        if save:
            fname = f'group_eyetracking_{data_type}_{self.task}.csv'
            df_filtered.to_csv(os.path.join(const.EYE_DIR, fname), index=False)
        return df

    def preprocess_behav(self, save=False):
        """
        loads preprocessed behavioral data
        """
        behav_fname = 'group_behavior.csv'
        df = pd.read_csv(os.path.join(const.BEHAVE_DIR, behav_fname))
        df_filtered = self._filter_behav(dataframe=df)  

        #optionally save to disk
        if save:
            df_filtered.to_csv(os.path.join(const.BEHAVE_DIR, f'group_behavior_{self.task}.csv'))
        return df_filtered  

    def merge_events_samples(self, dataframe1, dataframe2):
        df = dataframe1.merge(dataframe2, on=['type', 'block_iter', 'exp_event', 'task', 'event_type', 'run_num', 'onset_sec', 'subj', 'sess'])
        cols_to_keep = [col for col in df.columns if '_x' not in col]
        cols_to_keep = [col for col in cols_to_keep if '_y' not in col]
        return df[cols_to_keep]
    
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
        dataframe = dataframe_eye.merge(dataframe_behav, on=['task', 'subj', 'run_num', 'sess', 'block_iter', 'start_time'], how='outer')
        cols_to_keep = [col for col in dataframe.columns if 'Unnamed' not in col]
        cols_to_keep = [col for col in cols_to_keep if '_x' not in col]
        cols_to_keep = [col for col in cols_to_keep if '_y' not in col]
        return dataframe[cols_to_keep]
    
    def _filter_behav(self, dataframe):
        dataframe = dataframe.query(f'task=="{self.task}" and session_type=="behavioral"').dropna(axis=1)
        dataframe['sess'] = dataframe['sess'].div(2).astype(int)

        # drop some cols
        dataframe.drop(['run_num_new', 'run_name'], axis=1, inplace=True)
        dataframe['block_iter_corr'] = 'run' + dataframe['run_num'].astype(str).str.zfill(2) + '_block' + dataframe['block_iter'].astype(str)
        return dataframe

    def _filter_eye(self, dataframe, data_type):
        """filter eyetracking dataframe
            Args: 
                dataframe (pd dataframe): preprocessed eyetracking data
                data_type (str): 'samples' or 'events'
        """
        # filter fixation, saccade, blinks
        if data_type=="events":
            x_name = 'mean_gx'; y_name = 'mean_gy'
            sacc = dataframe.query(f'type=="saccade" and amplitude<1')
            fix = self._process_fixations(dataframe=dataframe, x_name=x_name, y_name=y_name)
            blink = dataframe.query('type=="blink"')
            dataframe = pd.concat([fix, sacc, blink])
        elif data_type=="samples":
            dataframe = dataframe.query('confidence>=0.8')
        
        # filter data to only include specific task
        dataframe = dataframe.query(f'task=="{self.task}" and event_type!="instructions"')
        return dataframe

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
   
    def _distance(self, row, center, x_name, y_name):
        return np.sqrt((row[x_name]-center[0])**2+(row[y_name]-center[1])**2)

    def _process_fixations(self, dataframe, x_name, y_name):
        df = dataframe[dataframe['type']=='fixations']
        df = df.query(f'{x_name}>=0 and {y_name}<=1 and {y_name}>=0 and {y_name}<=1')
        center = (np.mean(df[x_name]), np.mean(df[y_name]))
        df['dispersion'] = df.apply(self._distance, args=[center, x_name, y_name], axis=1)
        return df
        
    def _rescale_fixations(self, dataframe, x_name, y_name):
        x = dataframe[x_name]
        dataframe[x_name] = np.interp(x, (x.min(), x.max()), (0, self.dispsize[0]))

        y = dataframe[y_name]
        dataframe[y_name] = np.interp(y, (y.min(), y.max()), (0, self.dispsize[1]))

        return dataframe