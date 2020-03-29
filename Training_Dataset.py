# Imports
import os
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import sklearn
from tqdm import tqdm
import librosa
import librosa.display
from scipy.fftpack import fft
from scipy.signal import get_window
from numba import jit


class Training_Dataset:
    
    '''train_ts_dir: Directory to training time series.
    
    train_label_dir: Directory to training labels.
    
    ancil_ts_dir: Directory to ancillary time series.
    
    ancil_label_dir: Directory to ancillary labels.
    
    sort_by: subject_id, on_off, dyskinesia, tremor.
    
    combine_ancil = True: Combine training and ancillary data.
    
    dataset_name: 'CIS' or 'REAL'
    
    '''
    
    def __init__(self, train_ts_dir, train_label_dir,
                 ancil_ts_dir, ancil_label_dir, sort_by,dataset_name,
                 combine_ancil=True):
        self.train_ts_dir = train_ts_dir
        self.ancil_ts_dir = ancil_ts_dir
        self.sort_by = sort_by
        self.dataset_name = dataset_name
        self.train_labels = pd.read_csv(train_label_dir).replace(np.nan, 'nan', regex=True)
        self.ancil_labels = pd.read_csv(ancil_label_dir).replace(np.nan, 'nan', regex=True)
        self.classes = self.train_labels[f'{self.sort_by}'].unique()
        self.combine_ancil = combine_ancil
        self.spec_arrays = []
        self.spec_labels = []
        self.issue_measurements = []
        if self.combine_ancil:
            self.train_labels = pd.concat([self.train_labels, 
                                           self.ancil_labels]).reset_index(drop=True)
        
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
            self.data_dict = self.CIS_dictionary()
        elif self.dataset_name == 'REAL':
            self.data_dict = self.REAL_dictionary()
        else:
            raise NameError('Specify dataset name as REAL or CIS')
        print(f'Issue with {len(self.issue_measurements)} measurements',
              'enter self.issue_measuements for a list of them.')
        
    
    def run_preprocessing(self):
        warnings.filterwarnings("ignore")
        for key, tup_list in tqdm(self.data_dict.items(),'Preprocessing Data',
                                 total=len(self.data_dict),position=0, leave=True):
            for i, tup in enumerate(tup_list):
                tup_list[i][0] = self.standard_preprocessing(tup[0])
                tup_list[i] = tuple(tup)
        for key, tup_list in tqdm(self.data_dict.items(), 'Forming Data Lists'):
            for i, tup in enumerate(tup_list):
                self.spec_arrays.append(tup[0])
                self.spec_labels.extend([key])
        warnings.filterwarnings("default")
                
    # def torch_transform(self, spec, label):
    #     spec = TF.to_tensor(spec)
    #     label = torch.tensor(label)
    #     return spec, label
        
    # def __getitem__(self, index):
    #     spec = self.spec_arrays[index]
    #     label = self.spec_labels[index]
    #     spec, label = self.torch_transform(spec, label)
    #     return spec, label
    
    # def __len__(self):
    #     return len(self.spec_arrays)