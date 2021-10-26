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
#* Author:  Matteo Risso                                                      *
#*----------------------------------------------------------------------------*

import pickle
import matplotlib.pyplot as plt
import sys
import os
import json
import numpy as np
import copy

from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.utils import shuffle

from scipy import signal

def obtain_MAE(dataset, fine=False):
    pred = '_pred' if not fine else '_pred_fine'
    label = '_label' if not fine else '_label_fine'
    MAE = []
    for pat in np.arange(1,16):
        dataset['P' +str(pat) +pred] = dataset['P' +str(pat) +pred]
        dataset['P' +str(pat) +label] = dataset['P' +str(pat) +label]
        MAE_pat = np.mean(np.abs(dataset['P' +str(pat) +label]-dataset['P' +str(pat) +pred]))
        MAE.append(MAE_pat)
    MAE = np.asarray(MAE)
    print('Mean-Pre: {}'.format(np.mean(MAE)))
    print('Median-Pre: {}'.format(np.median(MAE)))
    return MAE

def post_processing(dataset, fine=False):
    n = 10
    f_h = 10
    f_l = 10
    pred = '_pred' if not fine else '_pred_fine'
    label = '_label' if not fine else '_label_fine'
    MAE_postprocessing = []
    for pat in np.arange(1,16):
    	old_value = dataset['P' +str(pat) +pred][0]
    	for i in np.arange(n,len(dataset['P' +str(pat) +label])):
    		if np.mean(dataset['P' +str(pat) +pred][i]) > np.mean(dataset['P' +str(pat) +pred][(i-n):i])*(100+f_h)/100.0:
    		    dataset['P' +str(pat) +pred][i] = np.mean(dataset['P' +str(pat) +pred][(i-n):i])*(100+f_h)/100
    		if np.mean(dataset['P' +str(pat) +pred][i]) < np.mean(dataset['P' +str(pat) +pred][(i-n):i])*(100-f_l)/100.0:
    		    dataset['P' +str(pat) +pred][i] = np.mean(dataset['P' +str(pat) +pred][(i-n):i])*(100-f_l)/100
    	MAE_pat = np.mean(np.abs(dataset['P' +str(pat) +label]-dataset['P' +str(pat) +pred]))
    	MAE_postprocessing.append(MAE_pat)
    MAE_postprocessing = np.asarray(MAE_postprocessing)
    print('Mean-Post: {}'.format(np.mean(MAE_postprocessing)))
    print('Median-Post: {}'.format(np.median(MAE_postprocessing)))
    return MAE_postprocessing

path = '5.0e-05_0_data.pickle'
with open(path, 'rb') as f:
    dataset = pickle.load(f)

MAE = obtain_MAE(dataset, fine=False)
MAE_post = post_processing(dataset, fine=False)
