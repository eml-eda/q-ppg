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
import numpy as np
from skimage.util.shape import view_as_windows
from scipy.io import loadmat
import random
import os

def preprocessing(dataset, cf):
    # Sampling frequency of both ppg and acceleration data in IEEE_Training dataset
    fs_IEEE_Training = 125
    # Sampling frequency of acceleration data in PPG_Dalia dataset
    # The sampling frequency of ppg data in PPG_Dalia dataset is fs_PPG_Dalia*2
    fs_PPG_Dalia = 32
    
    fs_activity = 4
    
    Sessioni = dict()
    S = dict()
    acc = dict()
    ppg = dict()
    activity = dict()
    
    random.seed(20)
     
    ground_truth = dict()
    
    val = dataset
    
    if not os.path.exists(cf.path_PPG_Dalia+'slimmed_dalia.pkl'):
        numbers= list(range(1,16))
        session_list=random.sample(numbers,len(numbers))
        for j in session_list:
            paz = j
            
            with open(cf.path_PPG_Dalia + 'PPG_FieldStudy/S' + str(j) +'/S' + str(j) +'.pkl', 'rb') as f:
                S[paz] = pickle.load(f, encoding='latin1')
            ppg[paz] = S[paz]['signal']['wrist']['BVP'][::2]
            acc[paz] = S[paz]['signal']['wrist']['ACC']
            activity[paz] = S[paz]['activity']
            ground_truth[paz] = S[paz]['label']
            
        sig = dict()
        act_list = []
        groups= []
        sig_list = []
        ground_truth_list = []
        
        # Loop on keys of dictionary ground_truth
        for k in ground_truth:
            # Remeber to set the desired time window
            activity[k] = np.moveaxis(view_as_windows(activity[k], (4*cf.time_window,1),4*2)[:,0,:,:],1,2)
            activity[k] = activity[k][:,:,0]
            sig[k] = np.concatenate((ppg[k],acc[k]),axis=1)
            sig[k]= np.moveaxis(view_as_windows(sig[k], (fs_PPG_Dalia*cf.time_window,4),fs_PPG_Dalia*2)[:,0,:,:],1,2)
            groups.append(np.full(sig[k].shape[0],k))
            sig_list.append(sig[k])
            act_list.append(activity[k])
            ground_truth[k] = np.reshape(ground_truth[k], (ground_truth[k].shape[0],1))
            ground_truth_list.append(ground_truth[k])
    
        #print("gruppo",groups)
        groups = np.hstack(groups)
        X = np.vstack(sig_list)
        y = np.reshape(np.vstack(ground_truth_list),(-1,1))
        
        act = np.vstack(act_list)
        
        data = dict()
        data['X'] = X
        data['y'] = y
        data['groups'] = groups
        data['act'] = act
        
        with open(cf.path_PPG_Dalia+'slimmed_dalia.pkl', 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        
    else:
        with open(cf.path_PPG_Dalia+'slimmed_dalia.pkl', 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            
            X = data['X']
            y = data['y']
            groups = data['groups']
            act = data['act']
   
    print("dimensione train",X.shape, "dimesione test", y.shape,"dimensione gruppi",groups.shape)
    
    return X[:y.shape[0]], y, groups[:y.shape[0]], act[:y.shape[0]]
