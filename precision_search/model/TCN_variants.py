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
#* Author:  Alessio Burrello                                                  *
#*----------------------------------------------------------------------------*

import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from math import ceil
import sys
sys.path.append("..")
from models import quant_module_1d as qm
import json
import torch
import pandas as pd

def TCN_network(**kwargs):
    if kwargs['quantization'] == 'False':
        return TCN_network_float(dilations = kwargs['dilations'], channels = kwargs['channels'])
    elif kwargs['quantization'] == 'mix':
        dfs = pd.read_excel('ppg-mixed-precision.xlsx', sheet_name='mix-quantizations')
        dataset = dfs[dfs['Name'] == kwargs['sheet_name']][dfs['cd'] == kwargs['cd']]
        return TCN_network_quantized_mix(qm.QuantizedChanConv1d, wbits=dataset.values[0][2:14], abits=dataset.values[0][14:26], dilations = kwargs['dilations'], channels = kwargs['channels'], share_weight=True)
    elif kwargs['quantization'] == 'mix-search':
        return TCN_network_quantized_mix_search(qm.MixActivChanConv1d, wbits=[2, 4, 8], abits=[2, 4, 8], dilations = kwargs['dilations'], channels = kwargs['channels'], share_weight=True)
    else:
        return TCN_network_quantized(qm.QuantizedChanConv1d, abits = kwargs['quantization'], wbits = kwargs['quantization'], dilations = kwargs['dilations'], channels = kwargs['channels'])

class TCN_network_quantized_mix(BaseModel):
    """
    TEMPONet architecture:
    Three repeated instances of TemporalConvBlock and ConvBlock organized as follows:
    - TemporalConvBlock
    - ConvBlock
    Two instances of Regressor followed by a final Linear layer with a single neuron.
    """

    def __init__(self, conv, wbits, abits, dilations, channels, share_weight = True, dataset_name='PPG_Dalia', dataset_args={}):
        super(TCN_network_quantized_mix, self).__init__()
        self.conv_func = conv
        self.dil = dilations
        self.rf = [5, 5, 5, 9, 9,17, 17]
        self.ch = channels

        # 1st instance of two TempConvBlocks and ConvBlock
        k_tcb00 = ceil(self.rf[0] / self.dil[0])
        self.tcb00 = TempConvBlock(conv,
                                   ch_in=4,
                                   ch_out=self.ch[0],
                                   k_size=k_tcb00,
                                   dil=self.dil[0],
                                   pad=((k_tcb00 - 1) * self.dil[0] + 1) // 2,
                                   wbits=wbits[0],
                                   abits=abits[0],
                                   share_weight=share_weight,
                                   first_layer = True
                                   )
        k_tcb01 = ceil(self.rf[1] / self.dil[1])
        self.tcb01 = TempConvBlock(conv,
                                   ch_in=self.ch[0],
                                   ch_out=self.ch[1],
                                   k_size=k_tcb01,
                                   dil=self.dil[1],
                                   pad=((k_tcb01 - 1) * self.dil[1] + 1) // 2,
                                   wbits=wbits[1],
                                   abits=abits[1],
                                   share_weight=share_weight
                                   )
        k_cb0 = ceil(self.rf[2] / self.dil[2])
        self.cb0 = ConvBlock(conv,
                             ch_in=self.ch[1],
                             ch_out=self.ch[2],
                             k_size=k_cb0,
                             strd=1,
                             pad=((k_cb0 - 1) * self.dil[2] + 1) // 2,
                             dilation=self.dil[2],
                             wbits=wbits[2],
                             abits=abits[2],
                             share_weight=share_weight
                             )

        # 2nd instance of two TempConvBlocks and ConvBlock
        k_tcb10 = ceil(self.rf[3] / self.dil[3])
        self.tcb10 = TempConvBlock(conv,
                                   ch_in=self.ch[2],
                                   ch_out=self.ch[3],
                                   k_size=k_tcb10,
                                   dil=self.dil[3],
                                   pad=((k_tcb10 - 1) * self.dil[3] + 1) // 2,
                                   wbits=wbits[3],
                                   abits=abits[3],
                                   share_weight=share_weight
                                   )
        k_tcb11 = ceil(self.rf[4] / self.dil[4])
        self.tcb11 = TempConvBlock(conv,
                                   ch_in=self.ch[3],
                                   ch_out=self.ch[4],
                                   k_size=k_tcb11,
                                   dil=self.dil[4],
                                   pad=((k_tcb11 - 1) * self.dil[4] + 1) // 2,
                                   wbits=wbits[4],
                                   abits=abits[4],
                                   share_weight=share_weight
                                   )
        self.cb1 = ConvBlock(conv,
                             ch_in=self.ch[4],
                             ch_out=self.ch[5],
                             k_size=5,
                             strd=2,
                             pad=2,
                             wbits=wbits[5],
                             abits=abits[5],
                              dilation=self.dil[5],
                             share_weight=share_weight
                             )

        # 3td instance of TempConvBlock and ConvBlock
        k_tcb20 = ceil(self.rf[5] / self.dil[6])
        self.tcb20 = TempConvBlock(conv,
                                   ch_in=self.ch[5],
                                   ch_out=self.ch[6],
                                   k_size=k_tcb20,
                                   dil=self.dil[6],
                                   pad=((k_tcb20 - 1) * self.dil[6] + 1) // 2,
                                   wbits=wbits[6],
                                   abits=abits[6],
                                   share_weight=share_weight
                                   )
        k_tcb21 = ceil(self.rf[6] / self.dil[7])
        self.tcb21 = TempConvBlock(conv,
                                   ch_in=self.ch[6],
                                   ch_out=self.ch[7],
                                   k_size=k_tcb21,
                                   dil=self.dil[7],
                                   pad=((k_tcb21 - 1) * self.dil[7] + 1) // 2,
                                   wbits=wbits[7],
                                   abits=abits[7],
                                   share_weight=share_weight
                                   )
        self.cb2 = ConvBlock(conv,
                             ch_in=self.ch[7],
                             ch_out=self.ch[8],
                             k_size=5,
                             strd=4,
                             pad=4,
                             wbits=wbits[8],
                             abits=abits[8],
                             dilation=self.dil[8],
                             share_weight=share_weight
                             )

        # 1st instance of regressor
        self.regr0 = Regressor(
            ft_in=self.ch[8] * 4,
            ft_out=self.ch[9],
            wbits=wbits[9],
            abits=abits[9]
        )

        # 2nd instance of regressor
        self.regr1 = Regressor(
            ft_in=self.ch[9],
            ft_out=self.ch[10],
            wbits=wbits[10],
            abits=abits[10]
        )

        self.out_neuron = qm.QuantizedLinear(
            inplane=self.ch[10],
            outplane=1,
            wbits=wbits[11],
            abits=abits[11]
        )

    def forward(self, x):
        x = self.cb0(self.tcb01(self.tcb00(x)))
        x = self.cb1(self.tcb11(self.tcb10(x)))
        x = self.cb2(self.tcb21(self.tcb20(x)))

        x = x.flatten(1)
        x = self.regr0(x)
        x = self.regr1(x)

        x = self.out_neuron(x)
        return x


class TCN_network_quantized(BaseModel):
    """
    TEMPONet architecture:
    Three repeated instances of TemporalConvBlock and ConvBlock organized as follows:
    - TemporalConvBlock
    - ConvBlock
    Two instances of Regressor followed by a final Linear layer with a single neuron.
    """

    def __init__(self, conv, wbits, abits, dilations, channels, share_weight = True, dataset_name='PPG_Dalia', dataset_args={}):
        super(TCN_network_quantized, self).__init__()
        self.conv_func = conv
        self.dil = dilations
        self.rf = [5, 5, 5, 9, 9,17, 17]
        self.ch = channels

        # 1st instance of two TempConvBlocks and ConvBlock
        k_tcb00 = ceil(self.rf[0] / self.dil[0])
        self.tcb00 = TempConvBlock(conv,
                                   ch_in=4,
                                   ch_out=self.ch[0],
                                   k_size=k_tcb00,
                                   dil=self.dil[0],
                                   pad=((k_tcb00 - 1) * self.dil[0] + 1) // 2,
                                   wbits=wbits,
                                   abits=abits,
                                   share_weight=share_weight,
                                   first_layer = True
                                   )
        k_tcb01 = ceil(self.rf[1] / self.dil[1])
        self.tcb01 = TempConvBlock(conv,
                                   ch_in=self.ch[0],
                                   ch_out=self.ch[1],
                                   k_size=k_tcb01,
                                   dil=self.dil[1],
                                   pad=((k_tcb01 - 1) * self.dil[1] + 1) // 2,
                                   wbits=wbits,
                                   abits=abits,
                                   share_weight=share_weight
                                   )
        k_cb0 = ceil(self.rf[2] / self.dil[2])
        self.cb0 = ConvBlock(conv,
                             ch_in=self.ch[1],
                             ch_out=self.ch[2],
                             k_size=k_cb0,
                             strd=1,
                             pad=((k_cb0 - 1) * self.dil[2] + 1) // 2,
                             dilation=self.dil[2],
                             wbits=wbits,
                             abits=abits,
                             share_weight=share_weight
                             )

        # 2nd instance of two TempConvBlocks and ConvBlock
        k_tcb10 = ceil(self.rf[3] / self.dil[3])
        self.tcb10 = TempConvBlock(conv,
                                   ch_in=self.ch[2],
                                   ch_out=self.ch[3],
                                   k_size=k_tcb10,
                                   dil=self.dil[3],
                                   pad=((k_tcb10 - 1) * self.dil[3] + 1) // 2,
                                   wbits=wbits,
                                   abits=abits,
                                   share_weight=share_weight
                                   )
        k_tcb11 = ceil(self.rf[4] / self.dil[4])
        self.tcb11 = TempConvBlock(conv,
                                   ch_in=self.ch[3],
                                   ch_out=self.ch[4],
                                   k_size=k_tcb11,
                                   dil=self.dil[4],
                                   pad=((k_tcb11 - 1) * self.dil[4] + 1) // 2,
                                   wbits=wbits,
                                   abits=abits,
                                   share_weight=share_weight
                                   )
        self.cb1 = ConvBlock(conv,
                             ch_in=self.ch[4],
                             ch_out=self.ch[5],
                             k_size=5,
                             strd=2,
                             pad=2,
                             wbits=wbits,
                             abits=abits,
                              dilation=self.dil[5],
                             share_weight=share_weight
                             )

        # 3td instance of TempConvBlock and ConvBlock
        k_tcb20 = ceil(self.rf[5] / self.dil[6])
        self.tcb20 = TempConvBlock(conv,
                                   ch_in=self.ch[5],
                                   ch_out=self.ch[6],
                                   k_size=k_tcb20,
                                   dil=self.dil[6],
                                   pad=((k_tcb20 - 1) * self.dil[6] + 1) // 2,
                                   wbits=wbits,
                                   abits=abits,
                                   share_weight=share_weight
                                   )
        k_tcb21 = ceil(self.rf[6] / self.dil[7])
        self.tcb21 = TempConvBlock(conv,
                                   ch_in=self.ch[6],
                                   ch_out=self.ch[7],
                                   k_size=k_tcb21,
                                   dil=self.dil[7],
                                   pad=((k_tcb21 - 1) * self.dil[7] + 1) // 2,
                                   wbits=wbits,
                                   abits=abits,
                                   share_weight=share_weight
                                   )
        self.cb2 = ConvBlock(conv,
                             ch_in=self.ch[7],
                             ch_out=self.ch[8],
                             k_size=5,
                             strd=4,
                             pad=4,
                             wbits=wbits,
                             abits=abits,
                             dilation=self.dil[8],
                             share_weight=share_weight
                             )

        # 1st instance of regressor
        self.regr0 = Regressor(
            ft_in=self.ch[8] * 4,
            ft_out=self.ch[9],
            wbits=wbits,
            abits=abits
        )

        # 2nd instance of regressor
        self.regr1 = Regressor(
            ft_in=self.ch[9],
            ft_out=self.ch[10],
            wbits=wbits,
            abits=abits
        )

        self.out_neuron = qm.QuantizedLinear(
            inplane=self.ch[10],
            outplane=1,
            wbits=wbits,
            abits=abits
        )

    def forward(self, x):
        x = self.cb0(self.tcb01(self.tcb00(x)))
        x = self.cb1(self.tcb11(self.tcb10(x)))
        x = self.cb2(self.tcb21(self.tcb20(x)))

        x = x.flatten(1)
        x = self.regr0(x)
        x = self.regr1(x)

        x = self.out_neuron(x)
        return x


class TCN_network_float(BaseModel):
    """
    TEMPONet architecture:
    Three repeated instances of TemporalConvBlock and ConvBlock organized as follows:
    - TemporalConvBlock
    - ConvBlock
    Two instances of Regressor followed by a final Linear layer with a single neuron.
    """

    def __init__(self, dilations, channels, dataset_name='PPG_Dalia', dataset_args={}):
        super(TCN_network_float, self).__init__()

        self.dil = dilations
        self.rf = [5, 5, 5, 9, 9,17, 17]
        self.ch = channels


        # 1st instance of two TempConvBlocks and ConvBlock
        k_tcb00 = ceil(self.rf[0] / self.dil[0])
        self.tcb00 = TempConvBlock_float(
            ch_in=4,
            ch_out=self.ch[0],
            k_size=k_tcb00,
            dil=self.dil[0],
            pad=((k_tcb00 - 1) * self.dil[0] + 1) // 2
        )
        k_tcb01 = ceil(self.rf[1] / self.dil[1])
        self.tcb01 = TempConvBlock_float(
            ch_in=self.ch[0],
            ch_out=self.ch[1],
            k_size=k_tcb01,
            dil=self.dil[1],
            pad=((k_tcb01 - 1) * self.dil[1] + 1) // 2
        )
        k_cb0 = ceil(self.rf[2] / self.dil[2])
        self.cb0 = ConvBlock_float(
            ch_in=self.ch[1],
            ch_out=self.ch[2],
            k_size=k_cb0,
            strd=1,
            pad=((k_cb0 - 1) * self.dil[2] + 1) // 2,
            dilation=self.dil[2]
        )

        # 2nd instance of two TempConvBlocks and ConvBlock
        k_tcb10 = ceil(self.rf[3] / self.dil[3])
        self.tcb10 = TempConvBlock_float(
            ch_in=self.ch[2],
            ch_out=self.ch[3],
            k_size=k_tcb10,
            dil=self.dil[3],
            pad=((k_tcb10 - 1) * self.dil[3] + 1) // 2
        )
        k_tcb11 = ceil(self.rf[4] / self.dil[4])
        self.tcb11 = TempConvBlock_float(
            ch_in=self.ch[3],
            ch_out=self.ch[4],
            k_size=k_tcb11,
            dil=self.dil[4],
            pad=((k_tcb11 - 1) * self.dil[4] + 1) // 2
        )
        self.cb1 = ConvBlock_float(
            ch_in=self.ch[4],
            ch_out=self.ch[5],
            k_size=5,
            strd=2,
            pad=2
        )

        # 3td instance of TempConvBlock and ConvBlock
        k_tcb20 = ceil(self.rf[5] / self.dil[6])
        self.tcb20 = TempConvBlock_float(
            ch_in=self.ch[5],
            ch_out=self.ch[6],
            k_size=k_tcb20,
            dil=self.dil[6],
            pad=((k_tcb20 - 1) * self.dil[6] + 1) // 2
        )
        k_tcb21 = ceil(self.rf[6] / self.dil[7])
        self.tcb21 = TempConvBlock_float(
            ch_in=self.ch[6],
            ch_out=self.ch[7],
            k_size=k_tcb21,
            dil=self.dil[7],
            pad=((k_tcb21 - 1) * self.dil[7] + 1) // 2
        )
        self.cb2 = ConvBlock_float(
            ch_in=self.ch[7],
            ch_out=self.ch[8],
            k_size=5,
            strd=4,
            pad=4
        )

        # 1st instance of regressor
        self.regr0 = Regressor_float(
            ft_in=self.ch[8] * 4,
            ft_out=self.ch[9]
        )

        # 2nd instance of regressor
        self.regr1 = Regressor_float(
            ft_in=self.ch[9],
            ft_out=self.ch[10]
        )

        self.out_neuron = nn.Linear(
            in_features=self.ch[10],
            out_features=1
        )

    def forward(self, x):
        x = self.cb0(self.tcb01(self.tcb00(x)))
        x = self.cb1(self.tcb11(self.tcb10(x)))
        x = self.cb2(self.tcb21(self.tcb20(x)))

        x = x.flatten(1)
        x = self.regr0(x)
        x = self.regr1(x)

        x = self.out_neuron(x)
        return x


class TempConvBlock_float(BaseModel):
    """
    Temporal Convolutional Block composed of one temporal convolutional layers.
    The block is composed of :
    - Conv1d layer
    - Chomp1d layer
    - ReLU layer
    - BatchNorm1d layer

    :param ch_in: Number of input channels
    :param ch_out: Number of output channels
    :param k_size: Kernel size
    :param dil: Amount of dilation
    :param pad: Amount of padding
    """

    def __init__(self, ch_in, ch_out, k_size, dil, pad):
        super(TempConvBlock_float, self).__init__()

        self.tcn0 = nn.Conv1d(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=k_size,
            dilation=dil,
            bias = False,
            padding=pad
        )
        self.relu0 = nn.ReLU6()
        self.bn0 = nn.BatchNorm1d(
            num_features=ch_out
        )

    def forward(self, x):
        x = self.relu0(self.bn0(self.tcn0(x)))
        return x


class ConvBlock_float(BaseModel):
    """
    Convolutional Block composed of:
    - Conv1d layer
    - AvgPool1d layer
    - ReLU layer
    - BatchNorm1d layer

    :param ch_in: Number of input channels
    :param ch_out: Number of output channels
    :param k_size: Kernel size
    :param strd: Amount of stride
    :param pad: Amount of padding
    """

    def __init__(self, ch_in, ch_out, k_size, strd, pad, dilation=1):
        super(ConvBlock_float, self).__init__()

        self.conv0 = nn.Conv1d(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=k_size,
            stride=strd,
            dilation=dilation,
            bias = False,
            padding=pad
        )
        self.pool0 = nn.AvgPool1d(
            kernel_size=2,
            stride=2,
            padding=0
        )
        self.relu0 = nn.ReLU6()
        self.bn0 = nn.BatchNorm1d(ch_out)

    def forward(self, x):
        x = self.relu0(self.bn0(self.pool0(self.conv0(x))))
        return x


class Regressor_float(BaseModel):
    """
    Regressor block  composed of :
    - Linear layer
    - ReLU layer
    - BatchNorm1d layer

    :param ft_in: Number of input channels
    :param ft_out: Number of output channels
    """

    def __init__(self, ft_in, ft_out):
        super(Regressor_float, self).__init__()
        self.ft_in = ft_in
        self.ft_out = ft_out

        self.fc0 = nn.Linear(
            in_features=ft_in,
            out_features=ft_out,
            bias = False
        )

        self.relu0 = nn.ReLU6()
        self.bn0 = nn.BatchNorm1d(
            num_features=ft_out
        )

    def forward(self, x):
        x = self.relu0(self.bn0(self.fc0(x)))
        return x


class TempConvBlock(BaseModel):
    """
    Temporal Convolutional Block composed of one temporal convolutional layers.
    The block is composed of :
    - Conv1d layer
    - Chomp1d layer
    - ReLU layer
    - BatchNorm1d layer

    :param ch_in: Number of input channels
    :param ch_out: Number of output channels
    :param k_size: Kernel size
    :param dil: Amount of dilation
    :param pad: Amount of padding
    """
    def __init__(self, conv, ch_in, ch_out, k_size, dil, pad, wbits, abits, share_weight, first_layer = False):
        super(TempConvBlock, self).__init__()

        self.tcn0 = conv(
                ch_in,
                ch_out,
                kernel_size = k_size,
                dilation = dil,
                padding = pad,
                groups = 1,
                bias = False,
                abits = abits,
                wbits = wbits,
                share_weight = share_weight,
                first_layer = first_layer
                )

        self.bn0 = nn.BatchNorm1d(num_features = ch_out)
       
    def forward(self, x):
        x = self.bn0(self.tcn0(x))
        return x

class ConvBlock(BaseModel):
    """
    Convolutional Block composed of:
    - Conv1d layer
    - AvgPool1d layer
    - ReLU layer
    - BatchNorm1d layer

    :param ch_in: Number of input channels
    :param ch_out: Number of output channels
    :param k_size: Kernel size
    :param strd: Amount of stride
    :param pad: Amount of padding
    """
    def __init__(self, conv, ch_in, ch_out, k_size, strd, pad, wbits, abits, share_weight, dilation=1):
        super(ConvBlock, self).__init__()
        
        self.conv0 = conv(
                ch_in,
                ch_out,
                kernel_size = k_size,
                stride = strd,
                dilation = dilation,
                padding = pad,
                groups = 1,
                bias = False,
                abits = abits,
                wbits = wbits,
                share_weight = share_weight,
                first_layer = False
                )
        self.pool0 = nn.AvgPool1d(
                kernel_size = 2,
                stride = 2,
                padding = 0
                )
        self.bn0 = nn.BatchNorm1d(ch_out)

    def forward(self, x):
        x = self.bn0(
                    self.pool0(
                            self.conv0(
                                x
                                )
                        )
                )
        return x

class Regressor(BaseModel):
    """
    Regressor block  composed of :
    - Linear layer
    - ReLU layer
    - BatchNorm1d layer

    :param ft_in: Number of input channels
    :param ft_out: Number of output channels
    """
    def __init__(self, ft_in, ft_out, wbits, abits):
        super(Regressor, self).__init__()
        self.ft_in = ft_in
        self.ft_out = ft_out
            
        self.fc0 = qm.QuantizedLinear(
                inplane = ft_in,
                outplane = ft_out,
                wbits=wbits,
                abits=abits
            )

        self.bn0 = nn.BatchNorm1d(
                num_features = ft_out
            )

    def forward(self, x):
        x = self.bn0(
                    self.fc0(
                            x
                        )
                )
        return x

class Chomp1d(BaseModel):
    """
    Module that perform a chomping operation on the input tensor.
    It is used to chomp the amount of zero-padding added on the right of the input tensor, this operation is necessary to compute causal convolutions.
    :param chomp_size: amount of padding 0s to be removed
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TCN_network_quantized_mix_search(BaseModel):
    """
    TEMPONet architecture:
    Three repeated instances of TemporalConvBlock and ConvBlock organized as follows:
    - TemporalConvBlock
    - ConvBlock
    Two instances of Regressor followed by a final Linear layer with a single neuron.
    """

    def __init__(self, conv, wbits, abits, dilations, channels, share_weight = True, dataset_name='PPG_Dalia', dataset_args={}):
        super(TCN_network_quantized_mix_search, self).__init__()
        self.conv_func = conv
        self.dil = dilations
        self.rf = [5, 5, 5, 9, 9,17, 17]
        self.ch = channels
        # 1st instance of two TempConvBlocks and ConvBlock
        k_tcb00 = ceil(self.rf[0] / self.dil[0])
        self.tcb00 = TempConvBlock(conv,
                                   ch_in=4,
                                   ch_out=self.ch[0],
                                   k_size=k_tcb00,
                                   dil=self.dil[0],
                                   pad=((k_tcb00 - 1) * self.dil[0] + 1) // 2,
                                   wbits=wbits,
                                   abits=abits,
                                   share_weight=share_weight,
                                   first_layer=True
                                   )
        k_tcb01 = ceil(self.rf[1] / self.dil[1])
        self.tcb01 = TempConvBlock(conv,
                                   ch_in=self.ch[0],
                                   ch_out=self.ch[1],
                                   k_size=k_tcb01,
                                   dil=self.dil[1],
                                   pad=((k_tcb01 - 1) * self.dil[1] + 1) // 2,
                                   wbits=wbits,
                                   abits=abits,
                                   share_weight=share_weight
                                   )
        k_cb0 = ceil(self.rf[2] / self.dil[2])
        self.cb0 = ConvBlock(conv,
                             ch_in=self.ch[1],
                             ch_out=self.ch[2],
                             k_size=k_cb0,
                             strd=1,
                             pad=((k_cb0 - 1) * self.dil[2] + 1) // 2,
                             dilation=self.dil[2],
                             wbits=wbits,
                             abits=abits,
                             share_weight=share_weight
                             )

        # 2nd instance of two TempConvBlocks and ConvBlock
        k_tcb10 = ceil(self.rf[3] / self.dil[3])
        self.tcb10 = TempConvBlock(conv,
                                   ch_in=self.ch[2],
                                   ch_out=self.ch[3],
                                   k_size=k_tcb10,
                                   dil=self.dil[3],
                                   pad=((k_tcb10 - 1) * self.dil[3] + 1) // 2,
                                   wbits=wbits,
                                   abits=abits,
                                   share_weight=share_weight
                                   )
        k_tcb11 = ceil(self.rf[4] / self.dil[4])
        self.tcb11 = TempConvBlock(conv,
                                   ch_in=self.ch[3],
                                   ch_out=self.ch[4],
                                   k_size=k_tcb11,
                                   dil=self.dil[4],
                                   pad=((k_tcb11 - 1) * self.dil[4] + 1) // 2,
                                   wbits=wbits,
                                   abits=abits,
                                   share_weight=share_weight
                                   )
        self.cb1 = ConvBlock(conv,
                             ch_in=self.ch[4],
                             ch_out=self.ch[5],
                             k_size=5,
                             strd=2,
                             pad=2,
                             wbits=wbits,
                             abits=abits,
                             dilation=self.dil[5],
                             share_weight=share_weight
                             )

        # 3td instance of TempConvBlock and ConvBlock
        k_tcb20 = ceil(self.rf[5] / self.dil[6])
        self.tcb20 = TempConvBlock(conv,
                                   ch_in=self.ch[5],
                                   ch_out=self.ch[6],
                                   k_size=k_tcb20,
                                   dil=self.dil[6],
                                   pad=((k_tcb20 - 1) * self.dil[6] + 1) // 2,
                                   wbits=wbits,
                                   abits=abits,
                                   share_weight=share_weight
                                   )
        k_tcb21 = ceil(self.rf[6] / self.dil[7])
        self.tcb21 = TempConvBlock(conv,
                                   ch_in=self.ch[6],
                                   ch_out=self.ch[7],
                                   k_size=k_tcb21,
                                   dil=self.dil[7],
                                   pad=((k_tcb21 - 1) * self.dil[7] + 1) // 2,
                                   wbits=wbits,
                                   abits=abits,
                                   share_weight=share_weight
                                   )
        self.cb2 = ConvBlock(conv,
                             ch_in=self.ch[7],
                             ch_out=self.ch[8],
                             k_size=5,
                             strd=4,
                             pad=4,
                             wbits=wbits,
                             abits=abits,
                             dilation=self.dil[8],
                             share_weight=share_weight
                             )
        # 1st instance of regressor
        self.regr0 = Regressor(
            ft_in=self.ch[8] * 4,
            ft_out=self.ch[9],
            wbits=8,
            abits=8
        )

        # 2nd instance of regressor
        self.regr1 = Regressor(
            ft_in=self.ch[9],
            ft_out=self.ch[10],
            wbits=8,
            abits=8
        )

        self.out_neuron = nn.Linear(
            in_features=self.ch[10],
            out_features=1
        )

    def forward(self, x):
        x = self.cb0(self.tcb01(self.tcb00(x)))
        x = self.cb1(self.tcb11(self.tcb10(x)))
        x = self.cb2(self.tcb21(self.tcb20(x)))

        x = x.flatten(1)
        x = self.regr0(x)
        x = self.regr1(x)

        x = self.out_neuron(x)
        return x

    def complexity_loss(self):
        size_product = []
        loss = 0
        for m in self.modules():
            if isinstance(m, self.conv_func):
                loss += m.complexity_loss()
                size_product += [m.size_product]
        normalizer = size_product[0].item()
        loss /= normalizer
        return loss

    def fetch_best_arch(self):
        sum_bitops, sum_bita, sum_bitw = 0, 0, 0
        sum_mixbitops, sum_mixbita, sum_mixbitw = 0, 0, 0
        layer_idx = 0
        best_arch = None
        for m in self.modules():
            if isinstance(m, self.conv_func):
                layer_arch, bitops, bita, bitw, mixbitops, mixbita, mixbitw = m.fetch_best_arch(layer_idx)
                if best_arch is None:
                    best_arch = layer_arch
                else:
                    for key in layer_arch.keys():
                        if key not in best_arch:
                            best_arch[key] = layer_arch[key]
                        else:
                            best_arch[key].append(layer_arch[key][0])
                sum_bitops += bitops
                sum_bita += bita
                sum_bitw += bitw
                sum_mixbitops += mixbitops
                sum_mixbita += mixbita
                sum_mixbitw += mixbitw
                layer_idx += 1
        return best_arch, sum_bitops, sum_bita, sum_bitw, sum_mixbitops, sum_mixbita, sum_mixbitw
