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

import torch.nn.functional as F
import torch
import torch.nn as nn
import pdb

def nll_loss(output, target, *args):
    return F.nll_loss(output, target)

def logcosh(output, target, *args):
    x = output - target
    return torch.mean(x + torch.nn.Softplus()(-2*x) - torch.log(torch.tensor(2.)))

def MAE(output, target, *args):
    return F.l1_loss(output, target)

def MSE(output, target, *args):
    return F.mse_loss(output, target)

def crossentropy_loss(output, target, *args):
    return nn.CrossEntropyLoss()(output, target)

def bce_loss(output, target, *args):
    return nn.BCELoss()(output, target.float())

def trace_nll_loss(output, target, *args):
    target = target.squeeze()
    output = output.squeeze()
    return -torch.trace(
                torch.matmul(target, torch.log(output+1e-10).float().t()) +
                torch.matmul((1-target), torch.log(1-output+1e-10).float().t())
            ) / output.size(0)

def WordPTB_crossentropy_loss(output, target, *args):
    final_output = output[:, 40:].contiguous().view(-1, 10000)
    final_target = target[:, 40:].contiguous().view(-1) 
    return nn.CrossEntropyLoss()(final_output, final_target)

def weighted_L1_loss(output, target, *args):
    num_batches = args[0]
    eps = (.001 / num_batches)
    target = target.float()
    return (F.l1_loss(output, target) * (target + target.mean() + eps)).mean()
