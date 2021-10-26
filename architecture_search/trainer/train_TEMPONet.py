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
#import RandomGroupkfold as rgkf
from RandomGroupkfold import RandomGroupKFold_split

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.utils import shuffle
import pickle
import json
from models import build_TEMPONet

def warmup(model, epochs_num, X_sh, y_sh, early_stop, checkpoint, cf):
    hist = model.fit(x=np.transpose(X_sh.reshape(X_sh.shape[0], 4, cf.input_shape, 1), (0, 3, 2, 1)), \
                        y=y_sh, shuffle=True, \
                        validation_split=0.1, verbose=1, \
                        batch_size= cf.batch_size, epochs=epochs_num,
                        callbacks = [early_stop, checkpoint])
    return hist

def train_gammas(model, X_sh, y_sh, early_stop, save_gamma, exp_str, cf): 
    hist = model.fit(x=np.transpose(X_sh.reshape(X_sh.shape[0], 4, cf.input_shape, 1), (0, 3, 2, 1)), \
                        y=y_sh, shuffle=True, \
                        validation_split=0.1, verbose=1, \
                        batch_size= cf.batch_size, epochs=cf.epochs,
                        callbacks = [early_stop, save_gamma, exp_str])
    return hist

def morphnet_search(model, X_sh, y_sh, callback_list, cf): 
    hist = model.fit(
        x=np.transpose(X_sh.reshape(X_sh.shape[0], 4, cf.input_shape, 1), (0, 3, 2, 1)), \
        y=y_sh, shuffle=True, \
        validation_split=0.1, verbose=1, \
        batch_size= cf.batch_size_MN, epochs=cf.epochs_MN, 
        callbacks = callback_list)
    
    return hist

def retrain_dil(groups, X, y, activity, checkpoint, early_stop, cf, ofmap):
    
    predictions = dict()
    MAE = dict()
    
    dataset = dict()
    
    # organize data
    group_kfold = GroupKFold(n_splits=4)
    group_kfold.get_n_splits(X, y, groups)

    # retrain and cross-validate
    #result = rgkf.RandomGroupKFold_split(groups,4,cf.a)
    result = RandomGroupKFold_split(groups,4,cf.a)
    for train_index, test_val_index in result:
        X_train, X_val_test = X[train_index], X[test_val_index]
        y_train, y_val_test = y[train_index], y[test_val_index]
        activity_train, activity_val_test = activity[train_index], activity[test_val_index]

        logo = LeaveOneGroupOut()
        logo.get_n_splits(groups=groups[test_val_index])  # 'groups' is always required
        for validate_index, test_index in logo.split(X_val_test, y_val_test, groups[test_val_index]):
            X_validate, X_test = X_val_test[validate_index], X_val_test[test_index]
            y_validate, y_test = y_val_test[validate_index], y_val_test[test_index]
            activity_validate, activity_test = activity_val_test[validate_index], activity_val_test[test_index]
            groups_val=groups[test_val_index]
            k=groups_val[test_index][0]
        
            # init
            try:
               del model
            except:
               pass

            # obtain conv #output filters from learned json structure
            with open(cf.saving_path+'/learned_dil_'+
                      '{:.1e}'.format(cf.reg_strength)+
                      '_'+'{}'.format(cf.warmup)+'.json', 'r') as f:
                dil_list = [val for _,val in json.loads(f.read()).items()]

            model = build_TEMPONet.TEMPONet_learned(1, cf.input_shape, 
                                                    dil_ht=False,
                                                    dil_list=dil_list, ofmap=ofmap)

            # save model and weights
            val_mae = 'val_mean_absolute_error'
            mae = 'mean_absolute_error'
            checkpoint = ModelCheckpoint(cf.saving_path+'test_reg'+str(k)+'.h5', monitor=val_mae, verbose=1,\
            save_best_only=True, save_weights_only=False, mode='min', period=1)
            #configure  model
            adam = Adam(lr=cf.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
            model.compile(loss='logcosh', optimizer=adam, metrics=[mae])


            X_train, y_train = shuffle(X_train, y_train)
            print(X_train.shape)
            print(X_validate.shape)
            print(X_test.shape)

            # Training
            hist = model.fit(x=np.transpose(X_train.reshape(X_train.shape[0], 4, cf.input_shape, 1), (0, 3, 2, 1)), y=y_train, epochs=cf.epochs, batch_size=cf.batch_size, \
                             validation_data=(np.transpose(X_validate.reshape(X_validate.shape[0], 4, cf.input_shape, 1), (0, 3, 2, 1)), y_validate), verbose=1,\
                                 callbacks=[checkpoint, early_stop])

            #evaluate
            predictions[k] = model.predict(np.transpose(X_test.reshape(X_test.shape[0], 4, cf.input_shape, 1), (0, 3, 2, 1)))
            MAE[k] = np.linalg.norm(y_test-predictions[k], ord=1)/y_test.shape[0]

            print(MAE)
            
            dataset['P'+str(k)+'_label'] = y_test
            dataset['P'+str(k)+'_pred'] = predictions[k]
            dataset['P'+str(k)+'_activity'] = activity_test
            
    # save predictions and real values
    with open(cf.saving_path+
              '{:.1e}'.format(cf.reg_strength)+
              '_'+
              '{}'.format(cf.warmup)+
              '_data.pickle', 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return model, MAE

def retrain_ch(groups, X, y, activity, early_stop, cf, ofmap):
    
    predictions = dict()
    MAE = dict()
    dataset = dict()
    
    # organize data
    group_kfold = GroupKFold(n_splits=4)
    group_kfold.get_n_splits(X, y, groups)

    # retrain and cross-validate
    #result = rgkf.RandomGroupKFold_split(groups,4,cf.a)
    result = RandomGroupKFold_split(groups,4,cf.a)
    for train_index, test_val_index in result:
        X_train, X_val_test = X[train_index], X[test_val_index]
        y_train, y_val_test = y[train_index], y[test_val_index]
        activity_train, activity_val_test = activity[train_index], activity[test_val_index]

        logo = LeaveOneGroupOut()
        logo.get_n_splits(groups=groups[test_val_index])  # 'groups' is always required
        for validate_index, test_index in logo.split(X_val_test, y_val_test, groups[test_val_index]):
            X_validate, X_test = X_val_test[validate_index], X_val_test[test_index]
            y_validate, y_test = y_val_test[validate_index], y_val_test[test_index]
            activity_validate, activity_test = activity_val_test[validate_index], activity_val_test[test_index]
            groups_val=groups[test_val_index]
            k=groups_val[test_index][0]
        
            # init
            try:
               del model
            except:
               pass

            # obtain conv #output filters from learned json structure
            with open(cf.saving_path+
                      'learned_structure/learned_channels_'+
                      '{:.1e}'.format(cf.reg_strength)+
                      '_'+
                      '{:.1e}'.format(cf.threshold)+'.json',
                      'r') as f:
                conv_list = [val for _,val in json.loads(f.read()).items()]
            #conv_list=conv_list[3:]+conv_list[:3]
            
            model = build_TEMPONet.TEMPONet_learned(1, 
                                                    cf.input_shape, 
                                                    dil_ht=False,
                                                    dil_list=[], 
                                                    ofmap=conv_list)

            # save model and weights
            val_mae = 'val_mean_absolute_error'
            mae = 'mean_absolute_error'
            #configure  model
            adam = Adam(lr=cf.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
            model.compile(loss='logcosh', optimizer=adam, metrics=[mae])


            X_train, y_train = shuffle(X_train, y_train)
            print(X_train.shape)
            print(X_validate.shape)
            print(X_test.shape)

            # Training
            hist = model.fit(
                x=np.transpose(X_train.reshape(X_train.shape[0], 4, cf.input_shape, 1), (0, 3, 2, 1)), 
                y=y_train, epochs=cf.epochs, batch_size=cf.batch_size, \
                validation_data=(np.transpose(X_validate.reshape(X_validate.shape[0], 4, cf.input_shape, 1), (0, 3, 2, 1)), y_validate), verbose=1,\
                callbacks=[early_stop])

            #evaluate
            predictions[k] = model.predict(np.transpose(X_test.reshape(X_test.shape[0], 4, cf.input_shape, 1), (0, 3, 2, 1)))
            MAE[k] = np.linalg.norm(y_test-predictions[k], ord=1)/y_test.shape[0]

            print(MAE)
            
            dataset['P'+str(k)+'_label'] = y_test
            dataset['P'+str(k)+'_pred'] = predictions[k]
            dataset['P'+str(k)+'_activity'] = activity_test
            
    # save predictions and real values
    with open(
            cf.saving_path+
            '{:.1e}'.format(cf.reg_strength)+
            '_'+
            '{:.1e}'.format(cf.threshold)+
            '_data.pickle',
            'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return model, MAE

def retrain(groups, X, y, activity, checkpoint, early_stop, cf, ofmap, dil, input_setup='normal', test_all_subj=False):
    
    predictions = dict()
    MAE = dict()
    
    dataset = dict()

    all_subjects_perf = dict()
    
    # organize data
    group_kfold = GroupKFold(n_splits=4)
    group_kfold.get_n_splits(X, y, groups)

    # retrain and cross-validate
    #result = rgkf.RandomGroupKFold_split(groups,4,cf.a)
    result = RandomGroupKFold_split(groups,4,cf.a)
    for train_index, test_val_index in result:
        X_train, X_val_test = X[train_index], X[test_val_index]
        y_train, y_val_test = y[train_index], y[test_val_index]
        activity_train, activity_val_test = activity[train_index], activity[test_val_index]

        logo = LeaveOneGroupOut()
        logo.get_n_splits(groups=groups[test_val_index])  # 'groups' is always required
        for validate_index, test_index in logo.split(X_val_test, y_val_test, groups[test_val_index]):
            X_validate, X_test = X_val_test[validate_index], X_val_test[test_index]
            y_validate, y_test = y_val_test[validate_index], y_val_test[test_index]
            activity_validate, activity_test = activity_val_test[validate_index], activity_val_test[test_index]
            groups_val=groups[test_val_index]
            k=groups_val[test_index][0]

            
            # init
            try:
               del model
            except:
               pass

            dil_list = dil

            # [PPG_1, PPG_2, ACC_x, ACC_y, ACC_z]
            if input_setup == 'normal':
                n_ch = 4
                #X_train = X_train[:, [0, 2, 3, 4], :]
                #X_validate = X_validate[:, [0, 2, 3, 4], :]
                #X_test = X_test[:, [0, 2, 3, 4], :]
            elif input_setup == 'ppg_only_1':
                n_ch = 1
                X_train = X_train[:, [0], :]
                X_validate = X_validate[:, [0], :]
                X_test = X_test[:, [0], :]
            elif input_setup == 'ppg_only_2':
                n_ch = 2
                X_train = X_train[:, [0, 1], :]
                X_validate = X_validate[:, [0, 1], :]
                X_test = X_test[:, [0, 1], :]
            elif input_setup == 'all':
                n_ch = 5
            else:
                raise ValueError()

            model = build_TEMPONet.TEMPONet_learned(1, cf.input_shape, 
                                                    dil_ht=False,
                                                    dil_list=dil_list, ofmap=ofmap,
                                                    n_ch = n_ch)

            # save model and weights
            val_mae = 'val_mean_absolute_error'
            mae = 'mean_absolute_error'
            checkpoint = ModelCheckpoint(cf.saving_path+'test_reg'+str(k)+'.h5', monitor=val_mae, verbose=1,\
            save_best_only=True, save_weights_only=False, mode='min', period=1)
            #configure  model
            adam = Adam(lr=cf.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
            model.compile(loss='logcosh', optimizer=adam, metrics=[mae])


            X_train, y_train = shuffle(X_train, y_train)
            print(X_train.shape)
            print(X_validate.shape)
            print(X_test.shape)

            #import pdb; pdb.set_trace()

            # Training
            hist = model.fit(
                x = np.transpose(X_train.reshape(X_train.shape[0], n_ch, cf.input_shape, 1), (0, 3, 2, 1)), 
                y = y_train, epochs=cf.epochs, batch_size=cf.batch_size,
                validation_data=(np.transpose(X_validate.reshape(X_validate.shape[0], n_ch, cf.input_shape, 1), (0, 3, 2, 1)), y_validate), 
                verbose=1, callbacks=[checkpoint, early_stop])

            with open('retrain_hist.pickle', 'wb') as f:
                pickle.dump(hist.history, f, protocol=pickle.HIGHEST_PROTOCOL)

            #evaluate
            predictions[k] = model.predict(
                np.transpose(X_test.reshape(X_test.shape[0], n_ch, cf.input_shape, 1), (0, 3, 2, 1))
                )
            MAE[k] = np.linalg.norm(y_test-predictions[k], ord=1)/y_test.shape[0]

            print(MAE)

            if test_all_subj:
                train_subj = np.unique(groups[train_index])
                val_subj = np.unique(groups[test_val_index][validate_index])
                test_subj = np.unique(groups[test_val_index][test_index])

                all_subjects_perf['Subj_'+str(test_subj)] = dict()

                for i in train_subj:
                    train_data_X = X_train[groups[train_index] == i]
                    train_data_y = y_train[groups[train_index] == i]
                    predictions_curr = model.predict(
                        np.transpose(
                            train_data_X.reshape(train_data_X.shape[0], n_ch, cf.input_shape, 1), 
                            (0, 3, 2, 1)
                            )
                        )
                    MAE_curr = np.linalg.norm(train_data_y-predictions_curr, ord=1)/train_data_y.shape[0]
                    all_subjects_perf['Subj_'+str(test_subj)]['Train_subj_'+str(i)] = dict()
                    all_subjects_perf['Subj_'+str(test_subj)]['Train_subj_'+str(i)]['P'+str(i)+'_label'] = train_data_y
                    all_subjects_perf['Subj_'+str(test_subj)]['Train_subj_'+str(i)]['P'+str(i)+'_pred'] = predictions_curr
                    all_subjects_perf['Subj_'+str(test_subj)]['Train_subj_'+str(i)]['P'+str(i)+'_MAE'] = MAE_curr

                for i in val_subj:
                    val_data_X = X_validate[groups[test_val_index][validate_index] == i]
                    val_data_y = y_validate[groups[test_val_index][validate_index] == i]
                    predictions_curr = model.predict(
                        np.transpose(
                            val_data_X.reshape(val_data_X.shape[0], n_ch, cf.input_shape, 1), 
                            (0, 3, 2, 1)
                            )
                        )
                    MAE_curr = np.linalg.norm(val_data_y-predictions_curr, ord=1)/val_data_y.shape[0]
                    all_subjects_perf['Subj_'+str(test_subj)]['Val_subj_'+str(i)] = dict()
                    all_subjects_perf['Subj_'+str(test_subj)]['Val_subj_'+str(i)]['P'+str(i)+'_label'] = val_data_y
                    all_subjects_perf['Subj_'+str(test_subj)]['Val_subj_'+str(i)]['P'+str(i)+'_pred'] = predictions_curr
                    all_subjects_perf['Subj_'+str(test_subj)]['Val_subj_'+str(i)]['P'+str(i)+'_MAE'] = MAE_curr

                all_subjects_perf['Subj_'+str(test_subj)]['Test_subj_'+str(test_subj)] = dict()
                predictions_curr = model.predict(
                        np.transpose(
                            X_test.reshape(X_test.shape[0], n_ch, cf.input_shape, 1), 
                            (0, 3, 2, 1)
                            )
                        )
                MAE_curr = np.linalg.norm(y_test-predictions_curr, ord=1)/y_test.shape[0]
                all_subjects_perf['Subj_'+str(test_subj)]['Test_subj_'+str(test_subj)]['P'+str(k)+'_label'] = y_test
                all_subjects_perf['Subj_'+str(test_subj)]['Test_subj_'+str(test_subj)]['P'+str(k)+'_pred'] = predictions_curr
                all_subjects_perf['Subj_'+str(test_subj)]['Test_subj_'+str(test_subj)]['P'+str(k)+'_MAE'] = MAE_curr

                print(all_subjects_perf)

            dataset['P'+str(k)+'_label'] = y_test
            dataset['P'+str(k)+'_pred'] = predictions[k]
            dataset['P'+str(k)+'_activity'] = activity_test

    if test_all_subj:
        with open('all_subj_data.pickle', 'wb') as handle:
            pickle.dump(all_subjects_perf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # save predictions and real values
    with open(cf.saving_path+
              '{:.1e}'.format(cf.reg_strength)+
              '_'+
              '{:.1e}'.format(cf.threshold)+
              '{}'.format(cf.warmup)+
              '_data.pickle', 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return model, MAE

def fine_tune(groups, X, y, activity, checkpoint, early_stop, cf, ofmap, dil):
    
    predictions = dict()
    MAE = dict()
    predictions_fine = dict()
    MAE_fine = dict()
    
    dataset = dict()
    
    # organize data
    group_kfold = GroupKFold(n_splits=4)
    group_kfold.get_n_splits(X, y, groups)

    # retrain and cross-validate
    #result = rgkf.RandomGroupKFold_split(groups,4,cf.a)
    result = RandomGroupKFold_split(groups,4,cf.a)
    for train_index, test_val_index in result:
        X_train, X_val_test = X[train_index], X[test_val_index]
        y_train, y_val_test = y[train_index], y[test_val_index]
        activity_train, activity_val_test = activity[train_index], activity[test_val_index]

        logo = LeaveOneGroupOut()
        logo.get_n_splits(groups=groups[test_val_index])  # 'groups' is always required
        for validate_index, test_index in logo.split(X_val_test, y_val_test, groups[test_val_index]):
            X_validate, X_test = X_val_test[validate_index], X_val_test[test_index]
            y_validate, y_test = y_val_test[validate_index], y_val_test[test_index]
            activity_validate, activity_test = activity_val_test[validate_index], activity_val_test[test_index]
            groups_val=groups[test_val_index]
            k=groups_val[test_index][0]
        
            # init
            try:
               del model
            except:
               pass

            dil_list = dil
            model = build_TEMPONet.TEMPONet_learned(1, cf.input_shape, 
                                                    dil_ht=False,
                                                    dil_list=dil_list, ofmap=ofmap)

            # save model and weights
            val_mae = 'val_mean_absolute_error'
            mae = 'mean_absolute_error'
            checkpoint = ModelCheckpoint(cf.saving_path+'model'+str(k)+'.h5', monitor=val_mae, verbose=1,\
            save_best_only=True, save_weights_only=False, mode='min', period=1)
            #configure  model
            adam = Adam(lr=cf.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
            model.compile(loss='logcosh', optimizer=adam, metrics=[mae])


            X_train, y_train = shuffle(X_train, y_train)
            print(X_train.shape)
            print(X_validate.shape)
            print(X_test.shape)

            # Training
            hist = model.fit(x=np.transpose(X_train.reshape(X_train.shape[0], 4, cf.input_shape, 1), (0, 3, 2, 1)), y=y_train, epochs=cf.epochs, batch_size=cf.batch_size, \
                             validation_data=(np.transpose(X_validate.reshape(X_validate.shape[0], 4, cf.input_shape, 1), (0, 3, 2, 1)), y_validate), verbose=1,\
                                 callbacks=[checkpoint, early_stop])

            #evaluate
            predictions[k] = model.predict(np.transpose(X_test.reshape(X_test.shape[0], 4, cf.input_shape, 1), (0, 3, 2, 1)))
            MAE[k] = np.linalg.norm(y_test-predictions[k], ord=1)/y_test.shape[0]

            print('MAE Pre Fine Tuning: {}'.format(MAE))
            
            dataset['P'+str(k)+'_label'] = y_test
            dataset['P'+str(k)+'_pred'] = predictions[k]
            
            ### fine tuning ###
            frac = X_test.shape[0]*25//100
            
            X_fine_train = X_test[:frac]
            y_fine_train = y_test[:frac]
            activity_fine_train = activity_test[frac:]
                
            X_fine_test = X_test[frac:]
            y_fine_test = y_test[frac:]
            activity_fine_test = activity_test[frac:]
            
            try:
                del model
            except:
                pass
            
            model = load_model(cf.saving_path+'model'+str(k)+'.h5')
            adam = Adam(lr=cf.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
            
            j = 0
            for i in range(len(model.layers)):
                #if re.search('batch_normalization.+', layer.get_config()['name']):
                    #layer.trainable = False
                if j < 11:
                    #print(model.layers[i].trainable)
                    model.layers[i].trainable = False
                    #print(model.layers[i].trainable)
                
                j += 1
                
            model.compile(loss='logcosh', optimizer=adam, metrics=['mean_absolute_error'])
        
            model.summary()
            
            hist = model.fit(x=np.transpose(X_fine_train.reshape(X_fine_train.shape[0], 4, cf.input_shape, 1), (0, 3, 2, 1)), y=y_fine_train, epochs=100, batch_size=256, \
                         verbose=1)
                
            #evaluate
            predictions_fine[k] = model.predict(np.transpose(X_fine_test.reshape(X_fine_test.shape[0], 4, cf.input_shape, 1), (0, 3, 2, 1)))
            MAE_fine[k] = np.linalg.norm(y_fine_test-predictions_fine[k], ord=1)/y_fine_test.shape[0]
            print('MAE post fine tuning: {}'.format(MAE_fine))
            
            dataset['P'+str(k)+'_label_fine'] = y_fine_test
            dataset['P'+str(k)+'_pred_fine'] = predictions_fine[k]
            
    # save predictions and real values
    with open(cf.saving_path+
              '{:.1e}'.format(cf.reg_strength)+
              '_'+
              '{}'.format(cf.warmup)+
              '_data_finetune.pickle', 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return model, MAE