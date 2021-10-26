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

import numpy as np
import tensorflow as tf
import argparse
import json
from config import Config
import sys
import pdb

import math

# aliases
val_mae = 'val_mean_absolute_error'
mae = 'mean_absolute_error'

from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from morph_net.network_regularizers import flop_regularizer, model_size_regularizer
from morph_net.tools import structure_exporter

from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.utils import shuffle

from scipy.io import loadmat

from custom_callbacks import SaveGamma, export_structure, export_structure_MN

from preprocessing import preprocessing_Dalia as pp

from trainer import train_TEMPONet

from models import build_TEMPONet

import utils
import eval_flops

import pickle

# MorphNet is compatible only with tf1.x
if tf.__version__ != '1.14.0':
    import tensorflow.compat.v1 as tf
    tf.compat.v1.disable_eager_execution()

limit = 1024 * 2
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=limit)])
    except RuntimeError as e:
        print(e)

# PARSER
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--root', help='Insert the root path where dataset is stored \
                    and where data will be saved')
parser.add_argument('--NAS', help='PIT | PIT-Retrain | MN-Size | MN-Flops | Retrain | Fine-Tune')
parser.add_argument(
    '--learned_ch', nargs='*', type=int, default=None
)
parser.add_argument('--strength', help='Regularization Strength')
parser.add_argument('--threshold', help='Pruning Threshold', default=0.5)
parser.add_argument('--warmup', help='Number of warmup epochs', default=0)
args = parser.parse_args()

# Setup config
cf = Config(args.NAS, args.root)
cf.search_type = args.NAS
cf.reg_strength = float(args.strength)
cf.threshold = float(args.threshold)
try:
    cf.warmup = int(args.warmup)
except:
    if args.warmup == 'max':
        cf.warmup = args.warmup
    else:
        raise ValueError

#######
# PIT #
#######
if args.NAS == 'PIT':
    # callbacks
    save_gamma = SaveGamma()
    exp_str = export_structure(cf)
    early_stop = EarlyStopping(monitor=val_mae, min_delta=0.01, patience=35, mode='min', verbose=1)
    
    # Load data
    X, y, groups, activity = pp.preprocessing(cf.dataset, cf)
    
    # organize data
    group_kfold = GroupKFold(n_splits=4)
    group_kfold.get_n_splits(X, y, groups)
    
    if args.learned_ch is not None:
        ofmap = args.learned_ch

    model = build_TEMPONet.TEMPONet_pit(1, cf.input_shape, cf, ofmap=ofmap)
    del model
    model = build_TEMPONet.TEMPONet_pit(1, cf.input_shape, cf, trainable=False,
            ofmap=ofmap)
    
    # save model and weights
    checkpoint = ModelCheckpoint(
        cf.saving_path+
        'weights_strength{}_warmup{}'.format(cf.reg_strength, cf.warmup)+'.h5', 
        monitor=val_mae, verbose=1, 
        save_best_only=True, save_weights_only=True, mode='min', period=1)
    #configure  model
    adam = Adam(lr=cf.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='logcosh', optimizer=adam, metrics=[mae])
    
    X_sh, y_sh = shuffle(X, y)
    
    ##########
    # Warmup #
    ##########
    if cf.warmup != 0:
        print('Train model for {} epochs'.format(cf.warmup))
        strg = cf.reg_strength
        cf.reg_strength = 0
    
        if cf.warmup == 'max':
            epochs_num = cf.epochs
        else:
            epochs_num = cf.warmup
    
        warmup_hist = train_TEMPONet.warmup(model, epochs_num, X_sh, y_sh, 
                              early_stop, checkpoint, cf)
        cf.reg_strength = strg
    
    del model
    model = build_TEMPONet.TEMPONet_pit(1, cf.input_shape, 
                                       cf, trainable=True, ofmap=ofmap)
    model.compile(loss='logcosh', optimizer=adam, metrics=[mae])
    
    if cf.warmup != 0:
        tmp_model = build_TEMPONet.TEMPONet_pit(1, cf.input_shape, cf, trainable=False, ofmap=ofmap)
        # load weights in temp model
        tmp_model.load_weights(cf.saving_path+
                               'weights_strength{}_warmup{}'.format(cf.reg_strength, cf.warmup)+
                               '.h5')
        utils.copy_weights(model, tmp_model, cf)

    ################
    # Train gammas #
    ################
    print('Train on Gammas')
    print('Reg strength : {}'.format(cf.reg_strength))
    pit_hist = train_TEMPONet.train_gammas(model, X_sh, y_sh, early_stop, save_gamma, exp_str, cf)
    
    # Save hist
    try:
        with open('warmup_hist.pickle', 'wb') as f:
            pickle.dump(warmup_hist.history, f, protocol=pickle.HIGHEST_PROTOCOL)
    except:
        print('Something goes wrong')

    with open('pit_hist.pickle', 'wb') as f:
        pickle.dump(pit_hist.history, f, protocol=pickle.HIGHEST_PROTOCOL)

    ##############################
    # Retrain and cross-validate #
    ##############################
    tr_model, MAE = train_TEMPONet.retrain_dil(groups, X, y, activity, checkpoint, early_stop, cf, ofmap=ofmap)
    print(MAE)
    # Evaluate average MAE
    avg = 0
    for _, val in MAE.items():
        avg += val
        print("Average MAE : %f", avg/len(MAE))
       
    ####################### 
    # Create summary file #
    #######################
    f=open(
        cf.saving_path+
        "summary_strength{}_warmup{}.txt".format(cf.reg_strength, cf.warmup), 
        "a+")
    f.write("regularization strength : {reg_str} \t warmup : {wu} \t MAE : {mae} \t Model size : {size} \t FLOPS : {flops} \n".format(
           reg_str = cf.reg_strength,
           wu = cf.warmup,
           mae = avg/len(MAE),
           size = tr_model.count_params(),
           flops = eval_flops.get_flops(tr_model)
           ))
    f.close()

elif args.NAS == 'PIT-Retrain':
    cf.saving_path = cf.root+'saved_models_PIT/'
    # callbacks
    save_gamma = SaveGamma()
    exp_str = export_structure(cf)
    early_stop = EarlyStopping(monitor=val_mae, min_delta=0.01, patience=35, mode='min', verbose=1)
    # save model and weights
    checkpoint = ModelCheckpoint(
        cf.saving_path+
        'weights_strength{}_warmup{}'.format(cf.reg_strength, cf.warmup)+'.h5', 
        monitor=val_mae, verbose=1, 
        save_best_only=True, save_weights_only=True, mode='min', period=1)
    
    # Load data
    X, y, groups, activity = pp.preprocessing(cf.dataset, cf)
    
    # organize data
    group_kfold = GroupKFold(n_splits=4)
    group_kfold.get_n_splits(X, y, groups)
    
    if args.learned_ch is not None:
        ofmap = args.learned_ch

    ##############################
    # Retrain and cross-validate #
    ##############################
    tr_model, MAE = train_TEMPONet.retrain_dil(groups, X, y, activity, checkpoint, early_stop, cf, ofmap=ofmap)
    print(MAE)
    # Evaluate average MAE
    avg = 0
    for _, val in MAE.items():
        avg += val
        print("Average MAE : %f", avg/len(MAE))
       
    ####################### 
    # Create summary file #
    #######################
    f=open(
        cf.saving_path+
        "summary_strength{}_warmup{}.txt".format(cf.reg_strength, cf.warmup), 
        "a+")
    f.write("regularization strength : {reg_str} \t warmup : {wu} \t MAE : {mae} \t Model size : {size} \t FLOPS : {flops} \n".format(
           reg_str = cf.reg_strength,
           wu = cf.warmup,
           mae = avg/len(MAE),
           size = tr_model.count_params(),
           flops = eval_flops.get_flops(tr_model)
           ))
    f.close()   

######
# MN #
######
elif args.NAS == 'MN-Size' or args.NAS == 'MN-Flops':
    
    # Load data
    X, y, groups, activity = pp.preprocessing(cf.dataset, cf)
    
    # Learn channels
    model = build_TEMPONet.TEMPONet_mn(1, cf.input_shape, 
                                       dil_ht=False, 
                                       dil_list=[], ofmap=[])
    del model
    model = build_TEMPONet.TEMPONet_mn(1, cf.input_shape, 
                                       dil_ht=False, 
                                       dil_list=[], ofmap=[])
    
    # MorphNet definition
    if args.NAS == 'MN-Size':
        regularizer_fn = model_size_regularizer.GroupLassoModelSizeRegularizer
    elif args.NAS == 'MN-Flops':
        regularizer_fn = flop_regularizer.GroupLassoFlopsRegularizer
    network_regularizer = regularizer_fn( 
                        output_boundary=[model.output.op], 
                        input_boundary=[model.input.op], 
                        threshold=cf.threshold)
                        #gamma_threshold=cf.gamma_threshold)
                        
    morph_net_loss = network_regularizer.get_regularization_term()*cf.reg_strength
    
    cost = network_regularizer.get_cost()
    
    # add the new loss to the model
    model.add_loss(lambda: morph_net_loss)
    
    # add the cost and the new loss as metrics so we can keep track of them
    model.add_metric(cost, name='cost', aggregation='mean')
    model.add_metric(morph_net_loss, name='morphnet_loss', aggregation='mean')
    
    #configure  model
    adam = Adam(lr=5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='logcosh', optimizer=adam, metrics=[mae])
    
    X_sh, y_sh = shuffle(X, y)
    
    # Callbacks
    callback_list = [export_structure_MN(cf, network_regularizer, patience=20)]

    
    ###################
    # Search Channels #
    ###################
    print('Search Channels')
    print('Reg strength : {}'.format(cf.reg_strength))
    mn_hist = train_TEMPONet.morphnet_search(model, X_sh, y_sh, callback_list, cf)

    with open('mn_hist.pickle', 'wb') as f:
        pickle.dump(mn_hist.history, f, protocol=pickle.HIGHEST_PROTOCOL)

    
    ##############################
    # Retrain and cross-validate #
    ##############################
    # save model and weights
    early_stop = EarlyStopping(monitor=val_mae, min_delta=0.01, 
                 patience=20, mode='min', verbose=1)
    
    tr_model, MAE = train_TEMPONet.retrain_ch(
        groups, X, y, activity, early_stop, cf, ofmap=[])
    print(MAE)
    # Evaluate average MAE
    avg = 0
    for _, val in MAE.items():
        avg += val
        print("Average MAE : %f", avg/len(MAE))
       
    ####################### 
    # Create summary file #
    #######################
    f=open(
        cf.saving_path+
        "summary_strength{}_threshold{}.txt".format(cf.reg_strength, cf.threshold), 
        "a+")
    f.write("regularization strength : {reg_str} \t threshold : {th} \t MAE : {mae} \t Model size : {size} \t FLOPS : {flops} \n".format(
           reg_str = cf.reg_strength,
           th = cf.threshold,
           mae = avg/len(MAE),
           size = tr_model.count_params(),
           flops = eval_flops.get_flops(tr_model)
           ))
    f.close()

elif args.NAS == 'Retrain':
    cf.saving_path = cf.root+'saved_models/'
    # callbacks
    save_gamma = SaveGamma()
    exp_str = export_structure(cf)
    early_stop = EarlyStopping(monitor=val_mae, min_delta=0.01, patience=35, mode='min', verbose=1)
    # save model and weights
    checkpoint = ModelCheckpoint(
        cf.saving_path+
        'weights_strength{}_warmup{}'.format(cf.reg_strength, cf.warmup)+'.h5', 
        monitor=val_mae, verbose=1, 
        save_best_only=True, save_weights_only=True, mode='min', period=1)
    
    # Load data
    X, y, groups, activity = pp.preprocessing(cf.dataset, cf)
    
    # organize data
    group_kfold = GroupKFold(n_splits=4)
    group_kfold.get_n_splits(X, y, groups)
    
    # OFMAP
    # Could be 'small' or 'medium' or 'large' or 'largest' or 'other'
    ofmap_type = 'other'
    if ofmap_type == 'small':
        ofmap = [
            1, 1, 16,
            1, 1, 128,
            1, 4, 2,
            14, 74, 1
        ]
    elif ofmap_type == 'medium':
        ofmap = [
            3, 9, 1,
            36, 8, 20,
            2, 5, 25,
            49, 85, 1
        ]
    elif ofmap_type == 'large':
        ofmap = [
            27, 26, 60,
            58, 64, 80,
            27, 29, 38,
            44, 57, 1
        ]
    elif ofmap_type == 'largest':
        ofmap = [
            32, 32, 63,
            62, 64, 128,
            89, 45, 38, 
            50, 61, 1
        ]
    else:
        # BestMAE
        ofmap = [
            32, 32, 63,
            62, 64, 128,
            89, 45, 38,
            50, 61, 1
        ]
        dil = [
            1, 1, 2,
            2, 1,
            2, 2
        ]

        # BestSize
        #ofmap = [
        #    1, 1, 16,
        #    1, 1, 128,
        #    1, 4, 2,
        #    14, 74, 1
        #]
        #dil = [
        #    2, 2, 4,
        #    1, 1,
        #    16, 1
        #]

    ##############################
    # Retrain and cross-validate #
    ##############################
    # input_setup:
    # 'normal': 4 channels, 1 PPG + 3 ACC
    # 'ppg_only_1': 1 channel, 1 PPG
    # 'ppg_only_2': 2 channels, 2 PPG
    # 'all': 5 channels, 2 PPG + 3 ACC
    input_setup = 'normal'
    tr_model, MAE = train_TEMPONet.retrain(groups, X, y, activity, checkpoint, early_stop, 
        cf, ofmap=ofmap, dil=dil, input_setup = input_setup, test_all_subj = True)
    print(MAE)
    # Evaluate average MAE
    avg = 0
    for _, val in MAE.items():
        avg += val
        print("Average MAE : %f", avg/len(MAE))
       
    ####################### 
    # Create summary file #
    #######################
    f=open(
        cf.saving_path+
        "summary_strength{}_warmup{}.txt".format(cf.reg_strength, cf.warmup), 
        "a+")
    f.write("regularization strength : {reg_str} \t warmup : {wu} \t MAE : {mae} \t Model size : {size} \t FLOPS : {flops} \n".format(
           reg_str = cf.reg_strength,
           wu = cf.warmup,
           mae = avg/len(MAE),
           size = tr_model.count_params(),
           flops = eval_flops.get_flops(tr_model)
           ))
    f.close()
    
elif args.NAS == 'Fine-Tune':
    cf.saving_path = cf.root+'saved_models/'
    early_stop = EarlyStopping(monitor=val_mae, min_delta=0.01, patience=35, mode='min', verbose=1)
    # save model and weights
    checkpoint = ModelCheckpoint(
        cf.saving_path+
        'weights_strength{}_warmup{}'.format(cf.reg_strength, cf.warmup)+'.h5', 
        monitor=val_mae, verbose=1, 
        save_best_only=True, save_weights_only=True, mode='min', period=1)
    
    # Load data
    X, y, groups, activity = pp.preprocessing(cf.dataset, cf)
    
    # organize data
    group_kfold = GroupKFold(n_splits=4)
    group_kfold.get_n_splits(X, y, groups)
    
    # OFMAP
    # Could be 'small' or 'medium' or 'large' or 'largest' or 'other'
    ofmap_type = 'other'
    if ofmap_type == 'small':
        ofmap = [
            1, 1, 16,
            1, 1, 128,
            1, 4, 2,
            14, 74, 1
        ]
    elif ofmap_type == 'medium':
        ofmap = [
            3, 9, 1,
            36, 8, 20,
            2, 5, 25,
            49, 85, 1
        ]
    elif ofmap_type == 'large':
        ofmap = [
            27, 26, 60,
            58, 64, 80,
            27, 29, 38,
            44, 57, 1
        ]
    elif ofmap_type == 'largest':
        ofmap = [
            32, 32, 63,
            62, 64, 128,
            89, 45, 38, 
            50, 61, 1
        ]
    else:
        ofmap = [
            32, 32, 63,
            62, 64, 128,
            89, 45, 38,
            50, 61, 1
        ]
        dil = [
            1, 1, 2,
            2, 1,
            2, 2
        ]

    ##############################
    # Retrain and cross-validate #
    ##############################
    tr_model, MAE = train_TEMPONet.fine_tune(groups, X, y, activity, checkpoint, early_stop, cf, ofmap=ofmap, dil=dil)
    print(MAE)
    # Evaluate average MAE
    avg = 0
    for _, val in MAE.items():
        avg += val
        print("Average MAE : %f", avg/len(MAE))
       
    ####################### 
    # Create summary file #
    #######################
    f=open(
        cf.saving_path+
        "summary_strength{}_warmup{}_threshold{}.txt".format(cf.reg_strength, cf.warmup, cf.threshold), 
        "a+")
    f.write("regularization strength : {reg_str} \t warmup : {wu} \t MAE : {mae} \t Model size : {size} \t FLOPS : {flops} \n".format(
           reg_str = cf.reg_strength,
           wu = cf.warmup,
           mae = avg/len(MAE),
           size = tr_model.count_params(),
           flops = eval_flops.get_flops(tr_model)
           ))
    f.close()