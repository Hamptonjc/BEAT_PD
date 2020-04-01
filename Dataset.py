# Imports
import os
import pandas as pd
import numpy as np
import warnings
import random
import matplotlib.pyplot as plt
import sklearn
from tqdm import tqdm
import librosa
import librosa.display
from scipy.fftpack import fft
from scipy.signal import get_window
from numba import jit

class Dataset:
    '''
    dataset_name: 'CIS' or 'REAL'

    sort_by: Keys for dict. 'subject_id', 'on_off', 'dyskinesia', or 'tremor'.

    label_class: what class to make labels from. 'on_off', 'dyskinesia', or 'tremor'.

    train_ts_dir: Directory to training time series.
    
    train_label_dir: Path to training labels.
    
    ancil_ts_dir: Directory to ancillary time series.
    
    ancil_label_dir: Path to ancillary labels.

    test_ts_dir(Optional): Directory to test time series

    test_data_id_dir(Optional): Path to test data id CSV
    
    combine_ancil = True: Combine training and ancillary data.
    '''
    def __init__(self, dataset_name, sort_by, label_class,
        train_ts_dir, train_label_dir,ancil_ts_dir, ancil_label_dir,
        test_ts_dir = None, test_data_id_dir = None, combine_ancil=True):

        self.dataset_name = dataset_name
        self.sort_by = sort_by
        self.label_class = label_class
        self.train_ts_dir = train_ts_dir
        self.ancil_ts_dir = ancil_ts_dir
        self.combine_ancil = combine_ancil
        self.train_labels = pd.read_csv(train_label_dir).replace(np.nan, 'nan', regex=True)
        self.ancil_labels = pd.read_csv(ancil_label_dir).replace(np.nan, 'nan', regex=True)
        self.test_ts_dir = test_ts_dir
        self.test_data_id_dir = test_data_id_dir
        self.data_list = []
        self.test_data_list = []
        self.issue_measurements = []
        if self.combine_ancil:
            self.train_labels = pd.concat([self.train_labels, 
                                           self.ancil_labels]).reset_index(drop=True)
        self.train_labels = self.train_labels[self.train_labels[label_class] != 'nan']
        self.classes = self.train_labels[self.sort_by].unique()
        if self.test_data_id_dir:
            self.test_data_ids = pd.read_csv(self.test_data_id_dir)
        self.create_dictionary()
        
    def CIS_dictionary(self): # Create CIS dictionary
        self.data_dict = {k : [] for k in self.classes}
        for index, row in tqdm(self.train_labels.iterrows(), 'Creating Dictionary',
                               total=self.train_labels.shape[0],position=0, leave=True):
            try: # look for measurement in training
                self.data_dict[row[self.sort_by]].append([pd.read_csv(
                    self.train_ts_dir + '/' + row.measurement_id +
                    '.csv').drop(columns='Timestamp'),row.measurement_id,row.subject_id,
                                                                   row.on_off,row.dyskinesia,
                                                                  row.tremor])
            except: # look for measurement in ancillary
                self.data_dict[row[self.sort_by]].append([pd.read_csv(
                    self.ancil_ts_dir + '/' + row.measurement_id +
                    '.csv').drop(columns='Timestamp'),row.measurement_id,row.subject_id,
                                                                   row.on_off,row.dyskinesia,
                                                                  row.tremor])
        return self.data_dict
    
    def REAL_dictionary(self):
        self.data_dict = {k : [] for k in self.classes}
        for index, row in tqdm(self.train_labels.iterrows(), 'Creating Dictionary',
                               total=self.train_labels.shape[0],position=0, leave=True):
            try: # Look in train data
                try: # smartphone_accelerometer
                    self.data_dict[row[self.sort_by]].append([pd.read_csv(
                        self.train_ts_dir + 'smartphone_accelerometer/' + row.measurement_id +
                        '.csv').drop(columns='t'),row.measurement_id,row.subject_id,
                                                                   row.on_off,row.dyskinesia,
                                                                  row.tremor])
                except:
                    pass
                try: #smartwatch_accelerometer
                    try:#attach gyroscope data
                        self.data_dict[row[self.sort_by]].append([pd.read_csv(
                            self.train_ts_dir + 'smartwatch_accelerometer/' + row.measurement_id +
                            '.csv').drop(columns='t'),row.measurement_id,row.subject_id,
                                                                       row.on_off,
                                                                       row.dyskinesia,row.tremor,
                                                                       pd.read_csv(
                                                                           self.train_ts_dir+
                                                                           'smartwatch_gyroscope/'+
                                                                           row.measurement_id+
                                                                           '.csv').drop(columns='t')])
                    except:#no gyroscope data
                        self.data_dict[row[self.sort_by]].append([pd.read_csv(
                            self.train_ts_dir + 'smartwatch_accelerometer/' + row.measurement_id +
                            '.csv').drop(columns='t'),row.measurement_id,row.subject_id,
                                                                       row.on_off,
                                                                       row.dyskinesia,
                                                                       row.tremor])
                except:
                    raise

            except: # look in ancil data
                try:# smartphone_accelerometer
                    self.data_dict[row[self.sort_by]].append([pd.read_csv(
                        self.ancil_ts_dir + 'smartphone_accelerometer/' + row.measurement_id +
                        '.csv').drop(columns='t'),row.measurement_id,row.subject_id,
                                                                   row.on_off,row.dyskinesia,
                                                                  row.tremor])
                except:
                    pass
                
                try: #smartwatch_accelerometer
                    try:#attach gyroscope data
                        self.data_dict[row[self.sort_by]].append([pd.read_csv(
                            self.ancil_ts_dir + 'smartwatch_accelerometer/' + row.measurement_id +
                            '.csv').drop(columns='t'),row.measurement_id,row.subject_id,
                                                                       row.on_off,
                                                                       row.dyskinesia,row.tremor,
                                                                       pd.read_csv(
                                                                           self.train_ts_dir+
                                                                           'smartwatch_gyroscope/'+
                                                                           row.measurement_id+
                                                                           '.csv').drop(columns='t')])
                    except:#no gyroscope data
                        self.data_dict[row[self.sort_by]].append([pd.read_csv(
                            self.ancil_ts_dir + 'smartwatch_accelerometer/' + row.measurement_id +
                            '.csv').drop(columns='t'),row.measurement_id,row.subject_id,
                                                                       row.on_off,
                                                                       row.dyskinesia,
                                                                       row.tremor])
                except:
                    self.issue_measurements.append(row.measurement_id)
        return self.data_dict

    def CIS_test_dictionary(self):
        self.test_subjects = self.test_data_ids.subject_id.unique()
        self.test_data_dict = {k : [] for k in self.test_subjects}
        for index, row in tqdm(self.test_data_ids.iterrows(), 'Creating Test Dictionary',
            total=self.test_data_ids.shape[0],position=0, leave=True):
            self.test_data_dict[row.subject_id].append([pd.read_csv(
            self.test_ts_dir + '/' + row.measurement_id +
                    '.csv').drop(columns='Timestamp'),row.measurement_id,row.subject_id])

    def REAL_test_dictionary(self):
        self.test_subjects = self.test_data_ids.subject_id.unique()
        self.test_data_dict = {k : [] for k in self.test_subjects}
        for index, row in tqdm(self.test_data_ids.iterrows(), 'Creating Test Dictionary',
                               total=self.test_data_ids.shape[0],position=0, leave=True):
                try: # smartphone_accelerometer
                    self.test_data_dict[row.subject_id].append([pd.read_csv(
                        self.test_ts_dir + 'smartphone_accelerometer/' + row.measurement_id +
                        '.csv').drop(columns='t'),row.measurement_id,row.subject_id])
                except:#smartwatch_accelerometer
                    try:#attach gyroscope data
                        self.test_data_dict[row.subject_id].append([pd.read_csv(
                            self.test_ts_dir + 'smartwatch_accelerometer/' + row.measurement_id +
                            '.csv').drop(columns='t'),row.measurement_id,row.subject_id,
                        pd.read_csv(self.train_ts_dir+'smartwatch_gyroscope/'+ row.measurement_id+'.csv').drop(columns='t')])
                    except:#no gyroscope data
                        self.test_data_dict[row.subject_id].append([pd.read_csv(
                            self.test_ts_dir + 'smartwatch_accelerometer/' + row.measurement_id +
                            '.csv').drop(columns='t'),row.measurement_id,row.subject_id])


    def pad_ts(self,ts):
        if ts.index[-1] < 60000:
            pad_size = 60000 - ts.index[-1]
            pre_pad_size = pad_size//2
            post_pad_size = pad_size//2
            pre_pad = pd.DataFrame(np.zeros((pre_pad_size, 3)),columns=['X', 'Y', 'Z'])
            post_pad = pd.DataFrame(np.zeros((post_pad_size, 3)),columns=['X', 'Y', 'Z'])
            measurement = pd.concat([ts, post_pad]).reset_index(drop=True)
            measurement = pd.concat([pre_pad, ts]).reset_index(drop=True)
        elif ts.index[-1] > 60000:
            ts = ts[:60001]
        return ts
    
    def center_n(self,ts, n=10):
        n_rows = int(n/(.02/60))
        center = ts.index[-1]//2
        return ts[center-(n_rows//2):center+(n_rows//2)]

    def magnitude(self,ts):
        if self.dataset_name == 'CIS':
            for _ in ts:
                ts['R'] = np.sqrt(ts['X']**2 + ts['Y']**2 + ts['Z']**2)
        else:
            for _ in ts:
                ts['R'] = np.sqrt(ts['x']**2 + ts['y']**2 + ts['z']**2)  
        return ts

    def zero_center_R(self,ts):
        R = ts.R.to_numpy()
        @jit(nopython=True)
        def R_zero_center(R):
            mean = np.mean(R)
            for item in R:
                item = item - mean
            return R
        ts.R = R
        return ts

    def spec_array(self,ts, n_mels = 128, hop_length = 256, sample_rate = 0.02,
                 n_fft = 2048, fmin = 0, fmax = 0.01, power=1.0):
        fig, ax = plt.subplots(figsize=(12, 5))
        mel_spec = librosa.feature.melspectrogram(np.array(ts.R), n_fft=n_fft, hop_length=hop_length,
                                              n_mels=n_mels, sr=sample_rate, power=power,
                                              fmin=fmin, fmax=fmax)
        mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
        librosa.display.specshow(mel_spec_db, x_axis='time',  y_axis='mel', 
                             sr=sample_rate, hop_length=hop_length,fmin=fmin, fmax=fmax, ax=ax)
        fig.patch.set_visible(False)
        ax.axis('off')
        fig.canvas.draw()
        X = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close()
        return X
    
    def spec_prep(self,spec_arr, cropx=670, cropy=272):
        y,x,c = spec_arr.shape
        startx = x//2 - cropx//2
        starty = y//2 - cropy//2    
        spec_arr = spec_arr[starty:starty+cropy, startx:startx+cropx, :]
        spec_arr = spec_arr[:,:,:3]
        return spec_arr
    
    def standard_preprocessing(self,ts):
        ts = self.pad_ts(ts)
        ts = self.center_n(ts)
        ts = self.magnitude(ts)
        ts = self.zero_center_R(ts)
        arr = self.spec_array(ts)
        arr = self.spec_prep(arr)
        return arr
        
    def create_dictionary(self):
        '''Create CIS or REAL dictionary. Dictionary keys are classes specified to sort by.
        Elements are lists of measurements.
        Items in lists are lists of: [time series, measurement_id, subject_id, on_off,
        dyskinesia, tremor, gyroscope_df (when available)].
        These nested lists were chosen over tuples since a mutable data type
        was needed to preprocess the data further.'''
        if self.dataset_name == 'CIS':
            self.CIS_dictionary()
            if self.test_ts_dir and self.test_data_id_dir:
                self.CIS_test_dictionary()
        elif self.dataset_name == 'REAL':
            self.REAL_dictionary()
            if self.test_ts_dir and self.test_data_id_dir:
                self.REAL_test_dictionary()
        else:
            raise NameError('Specify dataset name as REAL or CIS')
        print(f'Issue with {len(self.issue_measurements)} measurements',
              'enter self.issue_measuements for a list of them.')
        
    
    def run_preprocessing(self, specific_key_dataset=None):
        '''
        specific_key_dataset (OPTIONAL): Pick a specific key to preprocess and create a torch dataset with.

        Use: For example, data dictionary is sorted by subject id. Select a specific subject to create a dataset to train with.
        '''
        if specific_key_dataset:
            self.preprocessed_dict = {specific_key_dataset: self.data_dict[specific_key_dataset]}
        else:   
            self.preprocessed_dict = self.data_dict
        warnings.filterwarnings("ignore")
        for key, tup_list in tqdm(self.preprocessed_dict.items(),'Preprocessing Data',
                                 total=len(self.preprocessed_dict),position=0, leave=True):
            for i, tup in enumerate(tup_list):
                tup_list[i][0] = self.standard_preprocessing(tup[0])
                tup_list[i] = tuple(tup)
        for key, tup_list in self.preprocessed_dict.items():
            for i, tup in enumerate(tup_list):
                self.data_list.append(tup)
        warnings.filterwarnings("default")

    def run_test_preprocessing(self, specific_key_dataset=None):
        if specific_key_dataset:
            self.preprocessed_test_dict = {specific_key_dataset: self.test_data_dict[specific_key_dataset]}
        else:   
            self.preprocessed_test_dict = self.test_data_dict
        warnings.filterwarnings("ignore")
        for key, tup_list in tqdm(self.preprocessed_test_dict.items(),'Preprocessing Test Data',
                                 total=len(self.preprocessed_test_dict),position=0, leave=True):
            for i, tup in enumerate(tup_list):
                tup_list[i][0] = self.standard_preprocessing(tup[0])
                tup_list[i] = tuple(tup)
        for key, tup_list in self.preprocessed_test_dict.items():
            for i, tup in enumerate(tup_list):
                self.test_data_list.append(tup)
        warnings.filterwarnings("default")

    def train_validation_split(self, val_proportion=0.2):
        random.shuffle(self.data_list)
        split_index = int(len(self.data_list) * val_proportion)
        self.train_list = self.data_list[split_index:]
        self.val_list = self.data_list[:split_index]

