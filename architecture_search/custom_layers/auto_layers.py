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

import tensorflow as tf


from tensorflow.python.framework import tensor_shape
from tensorflow.python.eager import context
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import ops
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.util.tf_export import tf_export

from tensorflow.keras.layers import Conv2D, Layer

from utils import max_dil, prune_mul, binarize, g_weights 

import sys

class Dilation_Reg(tf.keras.regularizers.Regularizer):
    
    def __init__(self, reg_strength, c_in, c_out, r_f, l2=0.05):
        self.reg_strength = reg_strength
        self.c_in = c_in
        self.c_out = c_out
        self.r_f = r_f
        self.l2 = l2

    def __call__(self, w):
        gamma_weights = g_weights(w, self.c_in, self.c_out, self.r_f)
        return self.reg_strength * tf.reduce_sum(
            tf.multiply(
                    gamma_weights,
                    tf.abs(w), 
                    )
            ) + self.l2 * tf.reduce_sum(
            tf.square(w)
            )

    # Necessary to support serialization
    def get_config(self):
        return {'regularization_strength': self.reg_strength,
                'channel_in' : self.c_in,
                'channel_out' : self.c_out,
                'receptive field' : self.r_f,
                'l2_strength' : self.l2
                }

class clip_0_1(tf.keras.constraints.Constraint):
    """Constrains weight tensors to be between 0 and 1."""
    def __init__(self):
        pass

    def __call__(self, w):
        return tf.clip_by_value(w, 0, 1)

    def get_config(self):
        pass

class LearnedConv2D(Conv2D):
    
    def __init__(self, cf=None, gamma_trainable=True, hyst=0, **kwargs):
        self.cf = cf
        self.gamma_trainable = gamma_trainable
        self.hyst = hyst
        super(LearnedConv2D, self).__init__(**kwargs)

    def _assign_new_value(self, variable, value):
        with K.name_scope('AssignNewValue') as scope:
            with ops.colocate_with(variable):
                return state_ops.assign(variable, value, name=scope)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if self.data_format == 'channels_first':
          channel_axis = 1
        else:
          channel_axis = -1
        if input_shape.dims[channel_axis].value is None:
          raise ValueError('The channel dimension of the inputs '
                           'should be defined. Found `None`.')
        
        input_dim = int(input_shape[channel_axis])
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        
        # Trainable parameters that identifies the learned amount of dilation
        self.gamma = self.add_weight(
            name='gamma',
            shape=(1, max_dil(self.kernel_size[-1])),
            #constraint=tf.keras.constraints.NonNeg(),
            constraint=clip_0_1(),
            regularizer=Dilation_Reg(self.cf.reg_strength, input_dim, 
                self.filters, self.kernel_size[-1], l2=self.cf.l2),
            initializer=tf.keras.initializers.RandomUniform(1,1),
            trainable=self.gamma_trainable,
            dtype=self.dtype)

        if self.hyst == 1:
            self.alpha = self.add_weight(
                name='alpha',
                shape=(1, max_dil(self.kernel_size[-1])),
                constraint=tf.keras.constraints.NonNeg(),
                initializer=tf.keras.initializers.RandomUniform(1.,1.),
                synchronization=tf_variables.VariableSynchronization.ON_READ,
                trainable=False,
                dtype=self.dtype)
        
        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
        
        if self.use_bias:
          self.bias = self.add_weight(
              name='bias',
              shape=(self.filters,),
              initializer=self.bias_initializer,
              regularizer=self.bias_regularizer,
              constraint=self.bias_constraint,
              trainable=True,
              dtype=self.dtype)
        else:
          self.bias = None
          
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        
        if self.padding == 'causal':
          op_padding = 'valid'
        else:
          op_padding = self.padding
        if not isinstance(op_padding, (list, tuple)):
          op_padding = op_padding.upper()
    
        self._convolution_op = nn_ops.Convolution(
            input_shape,
            filter_shape=self.kernel.shape,
            dilation_rate=self.dilation_rate,
            strides=self.strides,
            padding=op_padding,
            data_format=conv_utils.convert_data_format(self.data_format,
                                                       self.rank + 2))
        
        self.built = True
        
    def call(self, inputs):
        
        if self.hyst == 0:
            bin_gamma = binarize(self.gamma, self.cf.threshold)
            pruned_kernel = prune_mul(self.kernel, bin_gamma)
        elif self.hyst == 1:
            bin_alpha_a = binarize(self.gamma, self.cf.threshold)
            bin_alpha_b = binarize(self.gamma, self.cf.threshold+self.cf.epsilon)
            bin_alpha = tf.add(
                tf.multiply(
                    self.alpha, 
                    bin_alpha_a
                    ),
                tf.multiply(
                    tf.constant(1.0, shape=[1, max_dil(self.kernel_size[-1])]) - self.alpha,
                    bin_alpha_b
                    )
                )
            
            #self.add_update((self.alpha, bin_alpha), inputs)
            self._assign_new_value(self.alpha, bin_alpha)

            pruned_kernel = prune_mul(self.kernel, bin_alpha)

        outputs = self._convolution_op(inputs, pruned_kernel)
    
        if self.use_bias:
          if self.data_format == 'channels_first':
            if self.rank == 1:
              # nn.bias_add does not accept a 1D input tensor.
              bias = array_ops.reshape(self.bias, (1, self.filters, 1))
              outputs += bias
            else:
              outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
          else:
            outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')
    
        if self.activation is not None:
          return self.activation(outputs)
        return outputs

class WeightNormConv2D(Conv2D):
    
    def __init__(self, **kwargs):
        super(WeightNormConv2D, self).__init__(**kwargs)
        

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if self.data_format == 'channels_first':
          channel_axis = 1
        else:
          channel_axis = -1
        if input_shape.dims[channel_axis].value is None:
          raise ValueError('The channel dimension of the inputs '
                           'should be defined. Found `None`.')
        
        input_dim = int(input_shape[channel_axis])
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        
        self.wn_g = self.add_weight(
            name='wn_g',
            shape=(self.filters,),
            initializer=tf.keras.initializers.RandomUniform(1,1),
            trainable=True,
            dtype=self.dtype)
        
        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
        
        if self.use_bias:
          self.bias = self.add_weight(
              name='bias',
              shape=(self.filters,),
              initializer=self.bias_initializer,
              regularizer=self.bias_regularizer,
              constraint=self.bias_constraint,
              trainable=True,
              dtype=self.dtype)
        else:
          self.bias = None
          
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        
        if self.padding == 'causal':
          op_padding = 'valid'
        else:
          op_padding = self.padding
        if not isinstance(op_padding, (list, tuple)):
          op_padding = op_padding.upper()
    
        self._convolution_op = nn_ops.Convolution(
            input_shape,
            filter_shape=self.kernel.shape,
            dilation_rate=self.dilation_rate,
            strides=self.strides,
            padding=op_padding,
            data_format=conv_utils.convert_data_format(self.data_format,
                                                       self.rank + 2))
        
        self.built = True
        
    def call(self, inputs):
        
        
        norm_w = tf.sqrt(tf.reduce_sum(
        tf.square(self.kernel), [0, 1, 2], keepdims=False))
        norm_v = tf.rsqrt(tf.reduce_sum(
        tf.square(self.wn_g)))
        norm_kernel = self.kernel * self.wn_g * (norm_v * norm_w)
      
        outputs = self._convolution_op(inputs, norm_kernel)
    
        if self.use_bias:
          if self.data_format == 'channels_first':
            if self.rank == 1:
              # nn.bias_add does not accept a 1D input tensor.
              bias = array_ops.reshape(self.bias, (1, self.filters, 1))
              outputs += bias
            else:
              outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
          else:
            outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')
    
        if self.activation is not None:
          return self.activation(outputs)
        return outputs           

class DenseTied(Layer):
    
    """Just your regular densely-connected NN layer.
    `Dense` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).
    Note: if the input to the layer has a rank greater than 2, then
    it is flattened prior to the initial dot product with `kernel`.
    Example:
    ```python
        # as first layer in a sequential model:
        model = Sequential()
        model.add(Dense(32, input_shape=(16,)))
        # now the model will take as input arrays of shape (*, 16)
        # and output arrays of shape (*, 32)
        # after the first layer, you don't need to specify
        # the size of the input anymore:
        model.add(Dense(32))
    ```
    Arguments:
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation")..
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        tied_to: tf layer name or layer variable to tie
    Input shape:
        nD tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.
    Output shape:
        nD tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 # kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 # kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 # kernel_constraint=None,
                 bias_constraint=None,
                 tied_to=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(DenseTied, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer), **kwargs)
        self.tied_to = tied_to
        self.units = int(units)
        self.activation = activations.get(activation)

        """transposed weights are variables and don't use any regularizators or initizlizators"""
        # self.kernel_initializer = None
        # self.kernel_constraint = None  
        # self.kernel_regularizer = None 

        """biases are still initialized and regularized"""
        self.use_bias = use_bias
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = True
        self.input_spec = InputSpec(min_ndim=2)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if input_shape[-1] is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')
        self.input_spec = InputSpec(min_ndim=2,
                                    axes={-1: input_shape[-1]})

        """Get and transpose tied weights 
        Caution: <weights> method returns array of arrays with kernels and biases and use only kernels here"""

        if isinstance(self.tied_to, str):
            # if <tied_to> is str i.e. tf layer name
            try:
                weights = model.get_layer("{}".format(self.tied_to)).weights[0] 
            except:
                weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "{}".format(self.tied_to))[0]
            self.transposed_weights = tf.transpose(weights, name='{}_kernel_transpose'.format(self.tied_to))

        else:
            # if <tied_to> is layer variable
            weights = self.tied_to.weights[0]
            #weights = self.tied_to.kernel
            self.transposed_weights = tf.transpose(weights, name='{}_kernel_transpose'.format(self.tied_to.name))


        if self.use_bias:
            self.bias = self.add_weight(
                'bias',
                shape=[self.units, ],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        rank = len(inputs.shape)
        if rank > 2:
          # Broadcasting is required for the inputs.
          outputs = standard_ops.tensordot(inputs, self.transposed_weights, [[rank - 1], [0]])
          # Reshape the output back to the original ndim of the input.
          if not context.executing_eagerly():
            shape = inputs.shape.as_list()
            output_shape = shape[:-1] + [self.units]
            outputs.set_shape(output_shape)
        else:
          inputs = math_ops.cast(inputs, self._compute_dtype)
          if K.is_sparse(inputs):
            outputs = sparse_ops.sparse_tensor_dense_matmul(inputs, self.transposed_weights)
          else:
            outputs = gen_math_ops.mat_mul(inputs, self.transposed_weights)
        if self.use_bias:
          outputs = nn.bias_add(outputs, self.bias)
        if self.activation is not None:
          return self.activation(outputs)  # pylint: disable=not-callable
        return outputs
        '''
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        rank = common_shapes.rank(inputs)
        if rank > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(inputs, self.transposed_weights, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                shape = inputs.get_shape().as_list()
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            outputs = gen_math_ops.mat_mul(inputs, self.transposed_weights)
        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs
        '''

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if input_shape[-1].value is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)
        return input_shape[:-1].concatenate(self.units)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            # 'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            # 'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            # 'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(DenseTied, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))        
        
        
        
        
