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
from tensorflow.python.framework import ops
import math
import copy
import re
import json
#import config as cf

if tf.__version__ == '1.14.0':
    def binarize(x, th):

        g = tf.get_default_graph()
    
        with ops.name_scope("Binarized") as name:
            with g.gradient_override_map({"Sign" : "Identity", "Round" : "Identity"}):
                return tf.round((tf.sign(x-th) + 1) / 2)
else:
    @tf.custom_gradient
    def binarize(x, th):
        bin_x = tf.round((tf.sign(x-th) + 1) / 2)
        def grad(dy):
            return tf.identity(x)*dy, None
        
        return bin_x, grad

def gamma_mul(dil_fact, it=0, line=[]):
    
    # entry point
    if it == 0:
        line = list()
        line.extend([[1]])
        it += 1
    
    # exit point
    elif it == int(math.log(dil_fact, 2)):
        return line
    
    else:
        #it += 1 
        for pos in range(len(line)):
            line[pos].append(0)
        
        line.extend([[0]*(it) + [1]])
            
        line.extend(copy.deepcopy(line[:(2**it-1)]))
        
        it += 1
        
    return gamma_mul(dil_fact, it, line)
    
def prune_mul(kernel, gamma):
    eps = 1e-6
    kernel_size = kernel.get_shape().as_list()[1]
    n_max = math.floor(math.log(kernel_size - eps,2))
    dil_fact_max = 2 ** n_max
    
    # gamma_mul matrix gen
    matrix_list = list()
    sum_list = list()

    i = 0
    while i < kernel_size:
        vector_list = list()
          
        # first element and multiples of dil_fact_max are always not pruned
        if i % dil_fact_max == 0:
            vector_list.extend([0] * n_max)
            matrix_list.append(vector_list)
            sum_list.append(1)
            i += 1
        else:
            for line in gamma_mul(dil_fact_max):
                matrix_list.append(line) 
                sum_list.append(0)
                i += 1
    
    # Truncate not necessary rows in matrix_list. 
    # i.e., from kernel_size to end
    # if len(matrix_list) == kernel_size, matrix_list[:-0] = [] !!! 
    if len(matrix_list) != kernel_size:
        matrix_list = matrix_list[:-(len(matrix_list)-kernel_size)]
        # Same for sum_list
        sum_list = sum_list[:-(len(sum_list)-kernel_size)]

    mask_mul = tf.transpose(tf.constant(matrix_list,shape=[kernel_size,n_max], dtype='float32'))    
    mask_sum = tf.constant(sum_list,shape=[kernel_size,1], dtype='float32') 
    
    m_1 = tf.constant(
        np.flip(
            np.triu(
                np.ones((n_max, n_max))),
            1
            ),
        dtype='float32'
        )
    
    m_2 = tf.constant(
        np.flip(
            np.tril(
                np.ones((n_max, n_max)),
                -1
                ),
            1
            ),
        dtype='float32'
        )
    
    Gamma = tf.add(
        tf.math.reduce_prod(
            tf.matmul(
                tf.add(
                    tf.multiply(
                        tf.matmul(
                            tf.cast(tf.reshape(gamma, [tf.shape(gamma)[1], 1]), dtype='float32'),
                            tf.constant(1, shape=(1,gamma.get_shape().as_list()[1]), dtype='float32')
                            ),
                            m_1
                        ), 
                    m_2
                    ),
                mask_mul
                ),
            axis=0
            ),
        tf.reshape(mask_sum, [mask_sum.get_shape().as_list()[0], ])   
        )

    Gamma = tf.reshape(Gamma, [Gamma.get_shape().as_list()[0], 1])
    
    return tf.transpose(tf.multiply(tf.transpose(Gamma, [0, 1]), 
                       tf.transpose(kernel, [2, 3, 1, 0])
                       ), [3, 2, 0, 1])
                        
def dil_fact(arr, op='sum'):
    if op == 'sum':
        dil = 0
        for i in arr.flatten():
            if i == 0:
                dil +=1
            else:
                break
        return 2 ** dil
    else:
        dil = 0
        for i in reversed(arr.flatten()):
            if i == 0:
                dil +=1
            else:
                break
        return 2 ** dil

def save_dil_fact(saving_path, dil, cf):
    f = open(saving_path+'learned_dil_'+
             '{:.1e}'.format(cf.reg_strength)+
             '_'+'{}'.format(cf.warmup)+'.json','w')
    f.write(format_structure(dil))
    f.close()
    
def format_structure(dil):
  return json.dumps(dil, indent=2, sort_keys=False, default=str)

def g_weights(w, c_in, c_out, r_f):
    kernel_size = w.get_shape().as_list()[1]
    
    g_w_list = []
    for i in range(kernel_size):
        g_w_list.append(
                c_in * c_out * math.ceil(((r_f - 1) / 2**(max_dil(r_f) - i)) - 0.5)
                )    
    return tf.constant(g_w_list, shape=[1,kernel_size], dtype='float32')

def effective_size(model, cf):
    actual_gamma = dict()
    
    delta_params = 0
    i = 0
    names = [weight.name for layer in model.layers for weight in layer.weights]
    weights = model.get_weights()
    for name, weight in zip(names, weights):
        if re.search('learned_conv2d.+_?[0-9]/gamma', name):
            actual_gamma[i] = weight
            actual_gamma[i] = np.array(actual_gamma[i] > cf.threshold, dtype=bool)
            actual_gamma[i] = dil_fact(actual_gamma[i], op='mul')
            i += 1
    
    i = 0
    for layer in model.layers:
        if re.search('learned_conv2d.+_?[0-9]', layer.name):
            # weights organized in the returned list as a gamma | kernel | bias
            kernels = layer.get_weights()[1]
            delta_params += kernels.shape[3] * kernels.shape[2] * (
                    kernels.shape[1] - math.ceil(kernels.shape[1] / actual_gamma[i]))
            i += 1

    return model.count_params() - delta_params

def copy_weights(model, tmp_model, cf):
    # copy weights from tmp_model to model
    # this tedious step is necessary because keras save in last positions non-trainable
    # weights, thus passing from non-trainable to trainable generates a mismatch error
    # between shapes of array of weights
    weight_list = tmp_model.get_weights()
    for i, layer in enumerate(tmp_model.layers):
        if re.search('learned_conv2d.+_?[0-9]', layer.name): 
            if cf.hyst == 0:
                order = [2, 0, 1]
            elif cf.hyst == 1:
                order = [2, 0, 1, 3]
            ordered_w = [layer.get_weights()[i] for i in order]
            model.layers[i].set_weights(ordered_w)
        elif re.search('weight_norm.+_?[0-9]', layer.name):
            if cf.hyst == 0:
                order = [0, 1, 4, 2, 3]
            elif cf.hyst == 1:
                order = [0, 1, 4, 2, 3, 5]
            ordered_w = [layer.get_weights()[i] for i in order]
            model.layers[i].set_weights(ordered_w)
        else:
            model.layers[i].set_weights(layer.get_weights())
    return

def max_dil(kernel_dim):
    eps = 1e-6
    return math.floor(math.log(kernel_dim - eps,2))
