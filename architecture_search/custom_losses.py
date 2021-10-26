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
import tensorflow.keras.backend as K


def NLL(y_true, y_pred):

    return -tf.linalg.trace(
        tf.matmul(
            tf.cast(y_true, dtype='float32'),
            tf.transpose(tf.cast(tf.math.log(tf.clip_by_value(y_pred, 1e-8, 1.0)), dtype='float32'), [0, 1, 3, 2])
            ) +
        tf.matmul(
            tf.cast((1 - y_true), dtype='float32'),
            tf.transpose(tf.cast(tf.math.log(tf.clip_by_value(1 - y_pred, 1e-8, 1.0)), dtype='float32'), [0, 1, 3, 2])
            )
        )

def accuracy(y_true, y_pred):
            # reshape in case it's in shape (num_samples, 1) instead of (num_samples,)
            if K.ndim(y_true) == K.ndim(y_pred):
                y_true = K.squeeze(y_true, -1)
            # convert dense predictions to labels
            y_pred_labels = K.argmax(y_pred, axis=-1)
            y_pred_labels = K.cast(y_pred_labels, K.floatx())
            return K.cast(K.equal(y_true, y_pred_labels), K.floatx())
