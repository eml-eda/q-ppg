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


#
# BEFORE RUNNING TAKE CARE THAT ARE THE CORRECT ONES
# CHECK DATASET AND SAVING PATHs
#

class Config:
    def __init__(self, search_type, root='./'):
        self.dataset = 'PPG_Dalia'
        self.root = root
        
        self.search_type = search_type
        
        # Data preprocessing parameters. Needs to be left unchanged
        self.time_window = 8
        self.input_shape = 32 * self.time_window
    
        # Training Parameters
        self.batch_size = 128
        self.lr = 0.001
        self.epochs = 500
        self.a = 35
        
        
        self.path_PPG_Dalia = self.root
        
        # warmup_epochs determines the number of training epochs without regularization
        # it could be an integer number or the string 'max' to indicate that we fully train the 
        # network
        self.warmup = 20
        # reg_strength determines how agressive lasso-reg is
        self.reg_strength = 1e-6
        # Amount of l2 regularization to be applied. Usually 0.
        self.l2 = 0.
        # threshold value is the value at which a weight is treated as 0. 
        self.threshold = 0.5
        
        self.hyst = 0
        
        # Where data are saved
        self.saving_path = self.root+'saved_models_'+self.search_type+'/'
        
        # parameters MorphNet training
        self.epochs_MN = 350
        self.batch_size_MN = 128
