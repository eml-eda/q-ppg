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

#import config as cf
import numpy as np
import tensorflow as tf
import utils

import re
import sys
import os

from morph_net.tools import structure_exporter

# aliases
val_mae = 'val_mean_absolute_error'
mae = 'mean_absolute_error'

class export_structure(tf.keras.callbacks.Callback):
    def __init__(self, cf):
        self.cf = cf
        super(export_structure, self).__init__()
    
    def set_model(self, model):
        self.model = model

    def on_train_begin(self, logs=None):
        # Initialize the best as infinity.
        if self.cf.dataset == 'PPG_Dalia':
           self.best = np.Inf
        elif self.cf.dataset == 'Nottingham' or self.cf.dataset == 'JSB_Chorales':
           self.best = np.Inf    
        elif self.cf.dataset == 'SeqMNIST' or self.cf.dataset == 'PerMNIST':
           self.best = 0
        else:
           print("{} is not supported".format(self.cf.dataset))
           sys.exit()
        self.gamma = dict()
        self.i = 0

    def on_epoch_end(self, epoch, logs=None):
        #get current validation mae
        if self.cf.dataset == 'PPG_Dalia':
            current = logs.get(val_mae)
            l = 1
            h = 0
            wait = 0
        else:
            print("{} is not supported".format(self.cf.dataset))
            sys.exit()
	
        if self.i > wait:
            # compare with previous best one
            if bool(np.less(current, self.best) * l) ^ \
                bool((current > self.best) * h):
                self.best = current

                # Record the best model if current results is better.
                names = [weight.name for layer in self.model.layers for weight in layer.weights]
                weights = self.model.get_weights()
                for name, weight in zip(names, weights):
                    if re.search('learned_conv2d.+_?[0-9]/gamma', name):
                        self.gamma[name] = weight
                        self.gamma[name] = np.array(self.gamma[name] > self.cf.threshold, dtype=bool)
                        self.gamma[name] = utils.dil_fact(self.gamma[name], op='mul')
                    elif re.search('weight_norm.+_?[0-9]/gamma', name):
                        self.gamma[name] = weight
                        self.gamma[name] = np.array(self.gamma[name] > self.cf.threshold, dtype=bool)
                        self.gamma[name] = utils.dil_fact(self.gamma[name], op='mul')
                print("New best model, update file. \n")
                print(self.gamma)
                if self.cf.dataset == 'PPG_Dalia':
                    utils.save_dil_fact(self.cf.saving_path, self.gamma, self.cf)
        else:
            self.i += 1

class SaveGamma(tf.keras.callbacks.Callback):
    
    
    def set_model(self, model):
        self.model = model

    def on_epoch_end(self, epoch, logs):

        names = [weight.name for layer in self.model.layers for weight in layer.weights]
        weights = self.model.get_weights()

        gamma = dict()
        i = 0
        for name, weight in zip(names, weights):
            if re.search('learned_conv2d.+_?[0-9]/gamma', name):
                print('gamma: ', weight.tolist()[0])
                gamma[i] = weight.tolist()[0]
                i += 1
            elif re.search('weight_norm.+_?[0-9]/gamma', name):
                print('gamma: ', weight.tolist()[0])
                gamma[i] = weight.tolist()[0]
                i += 1
        #gamma_history.append(gamma)

class export_structure_MN(tf.keras.callbacks.Callback):
    '''  
    Custom callback that saves the best structure of the network by following the next steps:
    It creates a StructureExporter object from the network_regularizer we defined before.
    It then creates a dictionary containing all of the tensors in the regularizer and evaluates them.
    It populates the tensors with the evaluated values and saves the current status in a file. 
    Here it saves the structure at ./saved_models/MN/learned_structure/ 
    with regularization_strength_gamma_threshold.json as the file name.
    
    Moreover an earlystopping mechanism has been implemented .
    
    Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
      '''

    def __init__(self, cf, network_regularizer, patience=0):
        super(export_structure_MN, self).__init__()
        self.cf = cf
        self.network_regularizer = network_regularizer
        self.patience = patience
        # exporter to store best architecture.
        self.exporter = None    

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf
    
    def on_epoch_end(self, epoch, logs=None):
        
        #get current validation mae
        current = logs.get("val_mean_absolute_error")
        
        #compare with previous best one
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best model if current results is better (less).
            self.exporter = structure_exporter.StructureExporter(self.network_regularizer.op_regularizer_manager)
        
            values = {}
            for key, item in self.exporter.tensors.items():
              values[key] = tf.keras.backend.eval(item)
            
            self.exporter.populate_tensor_values(values)
            
            self.exporter.create_file_and_save_alive_counts(
                self.cf.saving_path, 
                'learned_channels_{:.1e}'.format(self.cf.reg_strength)+
                '_'+
                '{:.1e}'.format(self.cf.threshold)+'.json')
            # rename file because the exporter.create_file_and_save_alive_counts() methods automatically
            # add an unwanted prefix
            os.replace(
                self.cf.saving_path+'learned_structure/alive_learned_channels_'+
                '{:.1e}'.format(self.cf.reg_strength)+
                '_'+'{:.1e}'.format(self.cf.threshold)+'.json',
                self.cf.saving_path+'learned_structure/learned_channels_'+
                '{:.1e}'.format(self.cf.reg_strength)+
                '_'+'{:.1e}'.format(self.cf.threshold)+'.json')
            path = self.cf.saving_path+'learned_channels_'+'{:.1e}'.format(self.cf.reg_strength)+'_'+'{:.1e}'.format(self.cf.threshold)+'.json'
            print('\nSaving model at:', path)
        else:
            self.wait += 1
            print('\nval_mae did not improve from {}'.format(self.best))
            print('Keep going on for at least {} epochs'.format(self.patience-self.wait))
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("\n Epoch %05d: early stopping" % (self.stopped_epoch + 1))