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

from tensorflow.keras import Sequential, layers
from custom_layers.auto_layers import LearnedConv2D
import math

def TEMPONet_pit(width_mult, in_shape, cf, trainable=True, ofmap=[]):
    
    input_channel = width_mult * 32
    output_channel = input_channel * 2
    
    if not ofmap:
        ofmap = [
                32, 32, 64,
                64, 64, 128,
                128, 128, 128,
                256, 128, 1
                ]

    model = Sequential()
    model.add(LearnedConv2D(
        cf=cf,
        gamma_trainable=trainable, 
        filters=ofmap[0], kernel_size=(1,5), padding='same', 
        dilation_rate=(1,1), input_shape = (1, in_shape, 4)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
     
    model.add(LearnedConv2D(
        cf=cf,
        gamma_trainable=trainable, 
        filters=ofmap[1], kernel_size=(1,5), padding='same', 
        dilation_rate=(1,1), input_shape = (1, in_shape, 4)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    
    model.add(LearnedConv2D(
        cf=cf,
        gamma_trainable=trainable,
        filters=ofmap[2], kernel_size=(1,5), padding='same',
        dilation_rate=(1,1), input_shape = (1, in_shape, 4)))
    model.add(layers.AveragePooling2D(pool_size=(1,2), strides=2, padding='valid'))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    
    input_channel = width_mult * 64
    output_channel = input_channel*2
    
    model.add(LearnedConv2D(
        cf=cf,
        gamma_trainable=trainable,
        filters=ofmap[3], kernel_size=(1,9), padding='same', 
        dilation_rate=(1,1), input_shape = (1, in_shape//2, 4)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    
    model.add(LearnedConv2D(
        cf=cf,
        gamma_trainable=trainable,
        filters=ofmap[4], kernel_size=(1,9), padding='same', 
        dilation_rate=(1,1), input_shape = (1, in_shape//2, 4)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Conv2D(
        filters=ofmap[5], 
        kernel_size=(1,5), 
        padding='same', 
        strides=2))
    model.add(layers.AveragePooling2D(pool_size=(1,2), strides=2, padding='valid'))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    
    input_channel = width_mult * 128
    output_channel = input_channel*2
    
    model.add(LearnedConv2D(
        cf=cf,
        gamma_trainable=trainable,
        filters=ofmap[6], kernel_size=(1,17), padding='same', 
        dilation_rate=(1,1), input_shape = (1, in_shape//4, 4)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    
    model.add(LearnedConv2D(
        cf=cf,
        gamma_trainable=trainable,
        filters=ofmap[7], kernel_size=(1,17), padding='same', 
        dilation_rate=(1,1), input_shape = (1, in_shape//4, 4)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Conv2D(filters=ofmap[8], 
        kernel_size=(1,5), 
        padding='valid', 
        strides=4))
    model.add(layers.AveragePooling2D(pool_size=(1,2), strides=2, padding='valid'))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Flatten())
    model.add(layers.Dense(ofmap[9]))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Dense(ofmap[10]))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Dense(ofmap[11]))

    model.summary()
    
    return model

def TEMPONet_mn(width_mult, in_shape, dil_ht, dil_list=[], ofmap=[]):
    
    rf = [5, 9, 17]
    
    if dil_ht:
        dil_list = [
                    2, 2, 1,
                    4, 4, 
                    8, 8
                    ]
    else:
        if not dil_list:
            dil_list = [
                    1, 1, 1,
                    1, 1, 
                    1, 1
                    ]
    
    if not ofmap:
        ofmap = [
                32, 32, 64,
                64, 64, 128,
                128, 128, 128,
                256, 128, 1
                ]

    input_channel = width_mult * 32
    output_channel = input_channel * 2

    model = Sequential()
    
    model.add(layers.Conv2D(
        filters=ofmap[0], 
        kernel_size=(1,math.ceil(rf[0]/dil_list[0])), 
        padding='same', dilation_rate=(1,dil_list[0]), 
        input_shape = (1, in_shape, 4)))
    model.add(layers.Activation('relu'))
    #model.add(layers.BatchNormalization())
    
    model.add(layers.Conv2D(
        filters=ofmap[1], 
        kernel_size=(1,math.ceil(rf[0]/dil_list[1])), 
        padding='same', dilation_rate=(1,dil_list[1]), 
        input_shape = (1, in_shape, 32)))
    model.add(layers.Activation('relu'))
    #model.add(layers.BatchNormalization())
     
    model.add(layers.ZeroPadding2D(padding=((0, 0), (4, 0)))) 
    model.add(layers.Conv2D(
        filters=ofmap[2], 
        kernel_size=(1,math.ceil(rf[0]/dil_list[2])), 
        padding='valid', dilation_rate=(1,dil_list[2]), 
        input_shape = (1, in_shape+4, 32))) 
    model.add(layers.AveragePooling2D(pool_size=(1,2), strides=2, padding='valid'))
    model.add(layers.Activation('relu'))
    #model.add(layers.BatchNormalization())
    
    input_channel = width_mult * 64
    output_channel = input_channel*2
    
    model.add(layers.Conv2D(
        filters=ofmap[3], 
        kernel_size=(1,math.ceil(rf[1]/dil_list[3])), 
        padding='same', dilation_rate=(1,dil_list[3]), 
        input_shape = (1, in_shape/2 + 8, 64)))
    model.add(layers.Activation('relu'))
    #model.add(layers.BatchNormalization())
    
    model.add(layers.Conv2D(
        filters=ofmap[4], 
        kernel_size=(1,math.ceil(rf[1]/dil_list[4])), 
        padding='same', dilation_rate=(1,dil_list[4]), 
        input_shape = (1, in_shape/2 + 8, 64)))
    model.add(layers.Activation('relu'))
    #model.add(layers.BatchNormalization())
    
    model.add(layers.ZeroPadding2D(padding=((0, 0), (4, 0))))
    model.add(layers.Conv2D(
        filters=ofmap[5], 
        kernel_size=(1,5), padding='valid', 
        strides=2, input_shape = (1, in_shape/2 + 4, 64)))
    model.add(layers.AveragePooling2D(pool_size=(1,2), strides=2, padding='valid'))
    model.add(layers.Activation('relu'))
    #model.add(layers.BatchNormalization())
    
    input_channel = width_mult * 128
    output_channel = input_channel*2
    
    model.add(layers.Conv2D(
        filters=ofmap[6], 
        kernel_size=(1,math.ceil(rf[2]/dil_list[5])), 
        padding='same', dilation_rate=(1,dil_list[5]), 
        input_shape = (1, in_shape/8 + 16, 128)))
    model.add(layers.Activation('relu'))
    #model.add(layers.BatchNormalization())
    
    model.add(layers.Conv2D(
        filters=ofmap[7], 
        kernel_size=(1,math.ceil(rf[2]/dil_list[6])), 
        padding='same', dilation_rate=(1,dil_list[6]), 
        input_shape = (1, in_shape/8 + 16, 128)))
    model.add(layers.Activation('relu'))
    #model.add(layers.BatchNormalization())
    
    model.add(layers.ZeroPadding2D(padding=((0, 0), (5, 0))))
    model.add(layers.Conv2D(
        filters=ofmap[8], 
        kernel_size=(1,5), padding='valid',
        strides=4, input_shape = (1, in_shape/8 + 5, 128)))
    model.add(layers.AveragePooling2D(pool_size=(1,2), strides=2, padding='valid'))
    model.add(layers.Activation('relu'))
    #model.add(layers.BatchNormalization())
    
    # Conv2D <==> Dense(256)
    model.add(layers.Conv2D(filters=ofmap[9], kernel_size=(1,4), padding='valid', strides=1, input_shape = (1, in_shape/64, 128)))
    model.add(layers.Activation('relu'))

    # Conv2D <==> Dense(128)
    model.add(layers.Conv2D(filters=ofmap[10], kernel_size=(1,1), padding='valid', strides=1, input_shape = (1, in_shape/256, 256)))
    model.add(layers.Activation('relu'))
    
    # Conv2D <==> Dense(1)
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(filters=1, kernel_size=(1,1), padding='valid', strides=1, input_shape = (1, in_shape/256, 128)))
    
    model.add(layers.GlobalAveragePooling2D())
    
    model.summary()
    
    return model

def TEMPONet_learned(width_mult, in_shape, dil_ht, dil_list=[], ofmap=[], n_ch=4):
    
    rf = [5, 9, 17]
    
    if not dil_list and dil_ht:
        dil_list = [
                    2, 2, 1,
                    4, 4, 
                    8, 8
                    ]
    elif not dil_list:
        dil_list = [
                    1, 1, 1,
                    1, 1, 
                    1, 1
                    ]
        
    
    if not ofmap:
        ofmap = [
                32, 32, 64,
                64, 64, 128,
                128, 128, 128,
                256, 128, 1
                ]
    else:
        for idx, i in enumerate(ofmap):
            if i == 0:
                ofmap[idx] = 1
                

    input_channel = width_mult * 32
    output_channel = input_channel * 2

    model = Sequential()
    
    model.add(layers.Conv2D(
        filters=ofmap[0], 
        kernel_size=(1,math.ceil(rf[0]/dil_list[0])), 
        padding='same', dilation_rate=(1,dil_list[0]), 
        input_shape = (1, in_shape, n_ch)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Conv2D(
        filters=ofmap[1], 
        kernel_size=(1,math.ceil(rf[0]/dil_list[1])), 
        padding='same', dilation_rate=(1,dil_list[1]), 
        input_shape = (1, in_shape, 32)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
     
    model.add(layers.ZeroPadding2D(padding=((0, 0), (4, 0)))) 
    model.add(layers.Conv2D(
        filters=ofmap[2], 
        kernel_size=(1,math.ceil(rf[0]/dil_list[2])), 
        padding='valid', dilation_rate=(1,dil_list[2]), 
        input_shape = (1, in_shape+4, 32))) 
    model.add(layers.AveragePooling2D(pool_size=(1,2), strides=2, padding='valid'))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    
    input_channel = width_mult * 64
    output_channel = input_channel*2
    
    model.add(layers.Conv2D(
        filters=ofmap[3], 
        kernel_size=(1,math.ceil(rf[1]/dil_list[3])), 
        padding='same', dilation_rate=(1,dil_list[3]), 
        input_shape = (1, in_shape/2 + 8, 64)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Conv2D(
        filters=ofmap[4], 
        kernel_size=(1,math.ceil(rf[1]/dil_list[4])), 
        padding='same', dilation_rate=(1,dil_list[4]), 
        input_shape = (1, in_shape/2 + 8, 64)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    
    model.add(layers.ZeroPadding2D(padding=((0, 0), (4, 0))))
    model.add(layers.Conv2D(
        filters=ofmap[5], 
        kernel_size=(1,5), padding='valid', 
        strides=2, input_shape = (1, in_shape/2 + 4, 64)))
    model.add(layers.AveragePooling2D(pool_size=(1,2), strides=2, padding='valid'))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    
    input_channel = width_mult * 128
    output_channel = input_channel*2
    
    model.add(layers.Conv2D(
        filters=ofmap[6], 
        kernel_size=(1,math.ceil(rf[2]/dil_list[5])), 
        padding='same', dilation_rate=(1,dil_list[5]), 
        input_shape = (1, in_shape/8 + 16, 128)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Conv2D(
        filters=ofmap[7], 
        kernel_size=(1,math.ceil(rf[2]/dil_list[6])), 
        padding='same', dilation_rate=(1,dil_list[6]), 
        input_shape = (1, in_shape/8 + 16, 128)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    
    model.add(layers.ZeroPadding2D(padding=((0, 0), (5, 0))))
    model.add(layers.Conv2D(
        filters=ofmap[8], 
        kernel_size=(1,5), padding='valid',
        strides=4, input_shape = (1, in_shape/8 + 5, 128)))
    model.add(layers.AveragePooling2D(pool_size=(1,2), strides=2, padding='valid'))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Flatten())
    model.add(layers.Dense(ofmap[9]))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Dense(ofmap[10]))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
     
    model.add(layers.Dense(ofmap[11]))
   
    model.summary()
    
    return model
