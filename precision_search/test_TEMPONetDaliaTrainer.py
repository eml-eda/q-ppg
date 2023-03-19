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

from parse_config import ConfigParser
import argparse
import collections
import torch
from trainer.TEMPONetDaliaTrainer import TEMPONetDaliaTrainer
from data_loader.DaliaDataLoader import DaliaDataLoader
import model
import model.loss as module_loss
import model.metric as module_metric
from utils import prepare_device
import pandas as pd
import numpy as np

model_names = sorted(name for name in model.__dict__
    if not name.startswith("__")
    and callable(model.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='TempoNetfloat',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: TEMPONetfloat)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lra', '--learning-rate-alpha', default=0.01, type=float,
                    metavar='LR', help='initial alpha learning rate')
parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--ad', '--alpha-decay', default=1e-4, type=float,
                    metavar='A', help='alpha decay (default: 1e-4)',
                    dest='alpha_decay')
parser.add_argument('--complexity-decay', '--cd', default=0.0001, type=float,
                    metavar='W', help='complexity decay (default: 1e-4)', dest='complexity_decay')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--cross-validation', default='True', help='Cross validation with fixed topology.', dest = 'cross_val')
parser.add_argument('--finetuning', default='False', help='False or True', dest = 'ft')
parser.add_argument('--sheet', default='False', help='False or MN-dilht or PIT', dest = 'sheet')
parser.add_argument('--net_number', default='1', help='1 to N', dest = 'net_number')
parser.add_argument('--quantization', default='8', help='2, 4, 8, mix, mix-search', dest = 'quantization')



args_parse = parser.parse_args()
print(args_parse)

args_tuple = collections.namedtuple('args', 'config resume device')

# Pars CLI argument and config file
args = args_tuple("config_Dalia.json", None, None)
config = ConfigParser.from_args(args)
# Setup logger
logger = config.get_logger('train')
if args_parse.sheet == 'False':
    print("=> creating model '{}'".format(args_parse.arch))
    model_principal = model.__dict__[args_parse.arch]()

    if 'mix' not in args_parse.arch and args_parse.cross_val == 'False':
        print('Searching for architecture topology without NAS nodes. Retry...')
        exit(0)

    if 'mix' in args_parse.arch and args_parse.cross_val == 'True':
        print('Validating NAS nodes without fixed bitwidth. Retry...')
        exit(0)
else:
    print("=> creating model from sheet {}, number {}'".format(args_parse.sheet, args_parse.net_number))
    dfs = pd.read_excel('ppg-mixed-precision.xlsx', sheet_name=args_parse.sheet)
    dilations = dfs.iloc[int(args_parse.net_number)-1].values[8:26][::2]
    channels = np.concatenate((dfs.iloc[int(args_parse.net_number)-1].values[8:26][1::2],dfs.iloc[int(args_parse.net_number)-1].values[26:28]),axis=0)
    if '2' in args_parse.quantization or '4' in args_parse.quantization or '8' in args_parse.quantization:
        quantization = int(args_parse.quantization)
    else:
        quantization = (args_parse.quantization)
    model_principal = model.TCN_network(
        dilations = dilations,
        channels = channels,
        quantization = quantization,
        sheet_name=args_parse.sheet,
        cd=args_parse.complexity_decay)

if '2' not in args_parse.quantization and '4' not in args_parse.quantization and '8' not in args_parse.quantization and 'mix'!=args_parse.quantization:
    args_parse.ft = 'False'

if args_parse.ft == 'True':
    if args_parse.sheet == 'False':
        if args_parse.arch.split('_')[0] == 'TempoNet':
            name = args_parse.arch.split('_')[0]+'float'
        else:
            name = args_parse.arch.split('_')[0]+'float_'+ args_parse.arch.split('_')[1]
        model_float = model.__dict__[name]()
    else:
        model_float = model.TCN_network(
            quantization = 'False',
            dilations = dilations,
            channels = channels)

else:
    model_float = False
logger.info(model_principal)
# Prepare for (multi-device) GPU training
device, device_ids = prepare_device(config['n_gpu'])
dev = args_parse.gpu
model_principal = model_principal.cuda(dev)
if args_parse.ft == 'True':
    model_float = model_float.cuda(dev)
if len(device_ids) > 1:
    model_principal = torch.nn.DataParallel(model_principal, device_ids=device_ids)
    if args_parse.ft == 'True':
        model_float = torch.nn.DataParallel(model_float, device_ids=device_ids)

# Get function handles of loss and metrics
criterion = getattr(module_loss, config['loss'])
metrics = [getattr(module_metric, met) for met in config['metrics']]
# group model/architecture parameters
params, alpha_params = [], []
for name, param in model_principal.named_parameters():
    if 'alpha' in name:
        alpha_params += [param]
    else:
        params += [param]
# Build optimizer
optimizer = torch.optim.Adam(params, lr = args_parse.lr, amsgrad='False',
                                 weight_decay=args_parse.weight_decay)

if args_parse.ft == 'True':
    # group model/architecture parameters
    params_float = []
    for name, param in model_float.named_parameters():
        params_float += [param]
    # Build optimizer
    optimizer_float = torch.optim.Adam(params_float, lr = args_parse.lr, amsgrad='False',
                                     weight_decay=args_parse.weight_decay)
else:
    optimizer_float = False
if args_parse.cross_val == 'True':
    arch_optimizer = False
else:
    arch_optimizer = torch.optim.SGD(alpha_params, lr = args_parse.lra, momentum=0.9,
                                    weight_decay=args_parse.alpha_decay)

# Trainer
trainer = TEMPONetDaliaTrainer(
                model = model_principal,
                model_float = model_float,
                criterion = criterion,
                metric_ftns = metrics,
                optimizer = optimizer,
                optimizer_float = optimizer_float,
                arch_optimizer = arch_optimizer,
                complexity_decay = args_parse.complexity_decay,
                config = config,
                device = dev,
                data_loader = DaliaDataLoader,
                data_dir = config['data_loader']['args']['data_dir'],
                batch_size = config['data_loader']['args']['batch_size'],
                finetuning = args_parse.ft,
                args_input = args_parse
                )

# Perform cross-validation
if args_parse.cross_val == 'True':
    print("Do cross-validation")
    
    MAE = trainer.cross_val(config['trainer']['cross_validation']['folds'])
    avg = sum(MAE.values()) / len(MAE)
    print("Average MAE : {}".format(avg))
else:
    # In order to train on the whole dataset instead of performing cross-val simply run:
    trainer.train_on_whole_dataset()
