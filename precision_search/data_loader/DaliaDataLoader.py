#*----------------------------------------------------------------------------*
#* Copyright (C) 2021 Politecnico di Torino, Italy                            *
#* SPDX-License-Identifier: Apache-2.0                                        *
#*                                                                            *
#* Licensed under the Apache License, Version 2.0 (the "License");            *
#* you may not use this file except in compliance with the License.           *
#* You may obtain a copy of the License at                                    *
#*                                                                            *
#* http://www.apache.org/licenses/LICENSE-2.0                                 *
#*                                                                            *
#* Unless required by applicable law or agreed to in writing, software        *
#* distributed under the License is distributed on an "AS IS" BASIS,          *
#* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
#* See the License for the specific language governing permissions and        *
#* limitations under the License.                                             *
#*                                                                            *
#* Author:  Alessio Burrello                                                  *
#*----------------------------------------------------------------------------*

import torch
from base import BaseDataLoader
from torch.utils.data import Dataset
from pathlib import Path
import pickle
import numpy as np
from skimage.util.shape import view_as_windows
import random
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
import os

class DaliaDataLoader(BaseDataLoader):
    """
    PPG-Dalia dataset loading and pre-processing
    """
    def __init__(self, data_dir, batch_size, shuffle=True, kfold_it=None, set_='train', validation_split=0.0, num_workers=0):
        
        self.dataset = DaliaDataset(data_dir, kfold_it, set_)        
        
        if set_ == 'test':
            # The test set does not need to be batched, thus the batch size is the whole dataset length
            super(DaliaDataLoader, self).__init__(self.dataset, self.dataset.__len__(), shuffle, validation_split, num_workers)
        else:
            super(DaliaDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
   
class DaliaDataset(Dataset):
    """
    PPG-Dalia dataset : https://archive.ics.uci.edu/ml/datasets/PPG-DaLiA

    :param data_dir: Path of the directory with the data
    :param kfold_it: Current iteration of the cross-validation scheme
    :param set_: Type of data to be parsed, i.e., train or validation or test
    """
    def __init__(self, data_dir, kfold_it=None, set_='train', transform=None):
        super(DaliaDataset, self).__init__()
        self.data_dir = Path(data_dir)
        if not os.path.exists(self.data_dir / 'slimmed_dalia.pkl'):
            self.dataset = self._collect_data(self.data_dir)
            self._X, self._y, self._groups = self._preprocess_data(self.dataset)
        else:
            with open(self.data_dir / 'slimmed_dalia.pkl', 'rb') as f:
                self.dataset = pickle.load(f, encoding='latin1')
            self._groups = self.dataset['groups']
            self._X = self.dataset['X'].astype('float32')
            self._y = self.dataset['y'].astype('float32')
        self.transform = transform
        
        self.kfold_it = kfold_it
        self.set_ = set_
        self.test_subj = 0
        
        if kfold_it == None:
            self.X, self.y = self._X, self._y
        else:
            self.X, self.y = self._kfold_split(self.kfold_it, self.set_)
            
    def get_test_subj(self):
        return self.test_subj

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = {
                'data': self.X[idx],
                'target': self.y[idx]
                }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _collect_data(self, data_dir):
        random.seed(42)
     
        dataset = dict()
        num = list(range(1, 15+1))
        session_list = random.sample(num, len(num))
        for subj in session_list:
            # data_dir = '/space/risso/PPG_Dalia/PPG_FieldStudy'
            with open(data_dir / ('S'+str(subj)) / ('S'+str(subj)+'.pkl'), 'rb') as f:
                subject = pickle.load(f, encoding='latin1')
            ppg = subject['signal']['wrist']['BVP'][::2].astype('float32')
            acc = subject['signal']['wrist']['ACC'].astype('float32')
            target = subject['label'].astype('float32')
            dataset[subj] = {
                    'ppg': ppg,
                    'acc': acc,
                    'target': target
                    }
        return dataset

    def _preprocess_data(self, dataset):
        """
        Process data with a sliding window of size 'time_window' and overlap 'overlap'
        """
        fs = 32
        time_window = 8
        overlap = 2

        groups = list()
        signals = list()
        targets = list()
        
        for k in dataset:
            sig = np.concatenate(
                        (dataset[k]['ppg'], dataset[k]['acc']), 
                        axis=1
                        )
            sig = np.moveaxis(
                        view_as_windows(
                            sig,
                            (fs*time_window, 4),
                            fs*overlap 
                            )[:,0,:,:],
                        1,
                        2
                        )
            ''' 
            # Normalization
            scalers = {}
            for i in range(sig.shape[1]):
                scalers[i] = StandardScaler()
                sig[:, i, :] = scalers[i].fit_transform(sig[:, i, :])
            '''
            groups.append(np.full(sig.shape[0], k))
            signals.append(sig)
            targets.append(np.reshape(
                dataset[k]['target'],
                (dataset[k]['target'].shape[0], 1)
                ))

        groups = np.hstack(groups)
        X = np.vstack(signals)
        y = np.reshape(
                np.vstack(targets),
                (-1, 1)
                )
        
        dataset = {
                'X': X,
                'y': y,
                'groups': groups
                }
        with open(self.data_dir / 'slimmed_dalia.pkl', 'wb') as f:
            pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

        return X, y, groups

    def _kfold_split(self, kfold_it, set_):
        """
        Return the training set for the actual fold of the cross-validation scheme
        """
        n = 4
        subjects = 15
        indices, _ = self._rndgroup_kfold(self._groups, n)
        
        fold = kfold_it  // n

        if set_ == 'train':
            train_index, _ = indices[fold]
            return self._X[train_index], self._y[train_index]
        elif set_ == 'validation' or set_ == 'test':
            _, test_val_index = indices[fold]
            logo = LeaveOneGroupOut()
            j = 0
            X_val_test = self._X[test_val_index]
            y_val_test = self._y[test_val_index]
            for validate_index, test_index in logo.split(X_val_test, y_val_test, self._groups[test_val_index]):
                
                self.test_subj = self._groups[test_val_index][test_index][0]
                
                if j == kfold_it % n:
                    if set_ == 'validation':
                        return X_val_test[validate_index], y_val_test[validate_index]
                    if set_ == 'test':
                        return X_val_test[test_index], y_val_test[test_index]
                j += 1
        else:
            raise ValueError


    def _rndgroup_kfold(self, groups, n, seed=35):
        """
        Random analogous of sklearn.model_selection.GroupKFold.split.

        :return: list of (train, test) indices
        """
        groups = pd.Series(groups)
        ix = np.arange(len(groups))
        unique = np.unique(groups)
        np.random.RandomState(seed).shuffle(unique)
        indices = list()
        split_dict = dict()
        i = 0
        for split in np.array_split(unique, n):
            split_dict[i] = split 
            i += 1
            mask = groups.isin(split)
            train, test = ix[~mask], ix[mask]
            indices.append((train, test))

        return indices, split_dict

    
