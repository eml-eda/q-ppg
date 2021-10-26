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

__all__ = ['TempoNetfloat']


def TempoNetfloat(**kwargs):
    return TEMPONet(**kwargs)


class TEMPONet(BaseModel):
    """
    TEMPONet architecture:
    Three repeated instances of TemporalConvBlock and ConvBlock organized as follows:
    - TemporalConvBlock
    - ConvBlock
    Two instances of Regressor followed by a final Linear layer with a single neuron.
    """

    def __init__(self, dataset_name='PPG_Dalia', dataset_args={}):
        super(TEMPONet, self).__init__()

        self.dil = [
            2, 2, 1,
            4, 4,
            8, 8
        ]
        self.rf = [
            5, 5, 5,
            9, 9,
            17, 17
        ]
        self.ch = [
            32, 32, 64,
            64, 64, 128,
            128, 128, 128,
            256, 128
        ]

        # 1st instance of two TempConvBlocks and ConvBlock
        k_tcb00 = ceil(self.rf[0] / self.dil[0])
        self.tcb00 = TempConvBlock(
            ch_in=4,
            ch_out=self.ch[0],
            k_size=k_tcb00,
            dil=self.dil[0],
            pad=((k_tcb00 - 1) * self.dil[0] + 1) // 2
        )
        k_tcb01 = ceil(self.rf[1] / self.dil[1])
        self.tcb01 = TempConvBlock(
            ch_in=self.ch[0],
            ch_out=self.ch[1],
            k_size=k_tcb01,
            dil=self.dil[1],
            pad=((k_tcb01 - 1) * self.dil[1] + 1) // 2
        )
        k_cb0 = ceil(self.rf[2] / self.dil[2])
        self.cb0 = ConvBlock(
            ch_in=self.ch[1],
            ch_out=self.ch[2],
            k_size=k_cb0,
            strd=1,
            pad=((k_cb0 - 1) * self.dil[2] + 1) // 2,
            dilation=self.dil[2]
        )

        # 2nd instance of two TempConvBlocks and ConvBlock
        k_tcb10 = ceil(self.rf[3] / self.dil[3])
        self.tcb10 = TempConvBlock(
            ch_in=self.ch[2],
            ch_out=self.ch[3],
            k_size=k_tcb10,
            dil=self.dil[3],
            pad=((k_tcb10 - 1) * self.dil[3] + 1) // 2
        )
        k_tcb11 = ceil(self.rf[4] / self.dil[4])
        self.tcb11 = TempConvBlock(
            ch_in=self.ch[3],
            ch_out=self.ch[4],
            k_size=k_tcb11,
            dil=self.dil[4],
            pad=((k_tcb11 - 1) * self.dil[4] + 1) // 2
        )
        self.cb1 = ConvBlock(
            ch_in=self.ch[4],
            ch_out=self.ch[5],
            k_size=5,
            strd=2,
            pad=2
        )

        # 3td instance of TempConvBlock and ConvBlock
        k_tcb20 = ceil(self.rf[5] / self.dil[5])
        self.tcb20 = TempConvBlock(
            ch_in=self.ch[5],
            ch_out=self.ch[6],
            k_size=k_tcb20,
            dil=self.dil[5],
            pad=((k_tcb20 - 1) * self.dil[5] + 1) // 2
        )
        k_tcb21 = ceil(self.rf[6] / self.dil[6])
        self.tcb21 = TempConvBlock(
            ch_in=self.ch[6],
            ch_out=self.ch[7],
            k_size=k_tcb21,
            dil=self.dil[6],
            pad=((k_tcb21 - 1) * self.dil[6] + 1) // 2
        )
        self.cb2 = ConvBlock(
            ch_in=self.ch[7],
            ch_out=self.ch[8],
            k_size=5,
            strd=4,
            pad=4
        )

        # 1st instance of regressor
        self.regr0 = Regressor(
            ft_in=self.ch[8] * 4,
            ft_out=self.ch[9]
        )

        # 2nd instance of regressor
        self.regr1 = Regressor(
            ft_in=self.ch[9],
            ft_out=self.ch[10]
        )

        self.out_neuron = nn.Linear(
            in_features=self.ch[10],
            out_features=1
        )

    def forward(self, x):
        x = self.cb0(
            self.tcb01(
                self.tcb00(
                    x
                )
            )
        )
        x = self.cb1(
            self.tcb11(
                self.tcb10(
                    x
                )
            )
        )
        x = self.cb2(
            self.tcb21(
                self.tcb20(
                    x
                )
            )
        )

        x = x.flatten(1)
        x = self.regr0(
            x
        )
        x = self.regr1(
            x
        )

        x = self.out_neuron(
            x
        )
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

    def __init__(self, ch_in, ch_out, k_size, dil, pad):
        super(TempConvBlock, self).__init__()

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

    def __init__(self, ch_in, ch_out, k_size, strd, pad, dilation=1):
        super(ConvBlock, self).__init__()

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


class Regressor(BaseModel):
    """
    Regressor block  composed of :
    - Linear layer
    - ReLU layer
    - BatchNorm1d layer

    :param ft_in: Number of input channels
    :param ft_out: Number of output channels
    """

    def __init__(self, ft_in, ft_out):
        super(Regressor, self).__init__()
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

