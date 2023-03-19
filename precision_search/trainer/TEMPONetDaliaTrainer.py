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

import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import copy
import pickle

data_to_save = {}
class TEMPONetDaliaTrainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, model_float, criterion, metric_ftns, optimizer, optimizer_float, arch_optimizer, complexity_decay, config, device, data_loader, data_dir, batch_size, finetuning, args_input, do_validation=True, lr_scheduler=None, len_epoch=None):
        super(TEMPONetDaliaTrainer, self).__init__(model, model_float, criterion, metric_ftns, optimizer, optimizer_float, arch_optimizer, finetuning, config, args_input)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.len_epoch = len_epoch
        self.do_validation = do_validation
        self.lr_scheduler = lr_scheduler
        self.cd = complexity_decay
        self.finetuning = finetuning
        self.log_step = int(np.sqrt(batch_size))
        self.args_input = args_input
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.test_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def cross_val(self, n_folds):
        MAE = dict()
        init_state_model = copy.deepcopy(self.model.state_dict())
        init_state_optimizer = copy.deepcopy(self.optimizer.state_dict())
        init_state_model_float = copy.deepcopy(self.model_float.state_dict())
        init_state_optimizer_float = copy.deepcopy(self.optimizer_float.state_dict())
        for i in range(n_folds):
            print('Iteration {}/{}'.format(i+1, n_folds))
            # Reload initial model state
            self.model.load_state_dict(init_state_model)
            self.model_float.load_state_dict(init_state_model_float)
            
            # Reload initial optimizer state
            self.optimizer.load_state_dict(init_state_optimizer)
            self.optimizer_float.load_state_dict(init_state_optimizer_float)

            # Build data loaders for the current fold
            self.train_data_loader = self.data_loader(
                    data_dir = self.data_dir,
                    batch_size = self.batch_size,
                    kfold_it = i,
                    set_ = 'train'
                    )
            self.valid_data_loader = self.data_loader(
                    data_dir = self.data_dir,
                    batch_size = self.batch_size,
                    kfold_it = i,
                    set_ = 'validation'
                    )
            self.test_data_loader = self.data_loader(
                    data_dir = self.data_dir,
                    batch_size = self.batch_size,
                    shuffle = False,
                    kfold_it = i,
                    set_ = 'test'
                    )
            
            self.len_epoch = len(self.train_data_loader)
            
            self.train()
            
            subj = self.test_data_loader.dataset.get_test_subj()
            res = self.test()
            MAE[subj] = res['MAE'].cpu()
            print("Subj {} : {}".format(subj, MAE[subj]))
            print("MAE : {}".format(MAE))
        file_to_write = open("net_results/net_results_{}_net_{}_quantization_{}.pickle".format(self.args_input.sheet, self.args_input.net_number, self.args_input.quantization), "wb")
        pickle.dump(data_to_save, file_to_write)
        return MAE
    
    def test(self):
        self.model.eval()
        self.test_metrics.reset()
        subj = self.test_data_loader.dataset.get_test_subj()
        target_list = []
        output_list = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_data_loader):
                data, target = batch['data'].cuda(self.device), batch['target'].cuda(self.device)
                output = self.model(data)
                target_list.append(target.cpu().numpy())
                output_list.append(output.cpu().numpy())
                loss = self.criterion(output, target)
                self.test_metrics.update('loss', loss.item())
                for metr in self.metric_ftns:
                    self.test_metrics.update(metr.__name__, metr(output, target))
        data_to_save['P' + str(subj) + '_label'] = np.asarray(target_list).flatten()
        data_to_save['P' + str(subj) + '_pred'] = np.asarray(output_list).flatten()
        return self.test_metrics.result()

    def train_on_whole_dataset(self):
        self.data_loader_args = self.config['data_loader']['args']
        self.validation_split = self.config['data_loader']['args'].get('validation_split', None)
        self.train_data_loader = self.data_loader(
                data_dir = self.data_loader_args['data_dir'],
                batch_size = self.data_loader_args['batch_size'],
                validation_split = self.validation_split,
		        num_workers = self.data_loader_args['num_workers']
                )
        if self.validation_split is not None:
            self.valid_data_loader = self.train_data_loader.split_validation()
        
        self.len_epoch = len(self.train_data_loader)
        self.train()

    def _train_epoch(self, epoch, prec_float):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return : A log that contains average loss and metric in this epoch
        """
        if prec_float == 'True':
            self.model_float.train()
        else:
            self.model.train()
        self.train_metrics.reset()
        for batch_idx, batch in enumerate(self.train_data_loader):
            data, target = batch['data'].cuda(self.device), batch['target'].cuda(self.device)
            if self.arch_optimizer != False:
                self.arch_optimizer.zero_grad()
            if prec_float == 'True':
                self.optimizer_float.zero_grad()
            else:
                self.optimizer.zero_grad()
            if prec_float == 'True':
                output = self.model_float(data)
            else:
                output = self.model(data)
            
            loss = self.criterion(output, target)
            if self.arch_optimizer != False:
                if hasattr(self.model, 'module'):
                    loss_complexity = self.cd * self.model.module.complexity_loss()
                else:
                    loss_complexity = self.cd * self.model.complexity_loss()
                loss = loss + loss_complexity

            loss.backward()
            if self.arch_optimizer != False:
                self.arch_optimizer.step()
            if prec_float == 'True':
                self.optimizer_float.step()
            else:
                self.optimizer.step()

            self.writer.set_step((epoch - 1)* self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for metr in self.metric_ftns:
                self.train_metrics.update(metr.__name__, metr(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(epoch, self._progress(batch_idx), loss.item()))
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            
            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch, prec_float)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch, prec_float):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation.
        """
        if prec_float == 'True':
            self.model_float.eval()
        else:
            self.model.eval()

        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.valid_data_loader):
                data, target = batch['data'].cuda(self.device), batch['target'].cuda(self.device)

                if prec_float == 'True':
                    output = self.model_float(data)
                else:
                    output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for metr in self.metric_ftns:
                    self.valid_metrics.update(metr.__name__, metr(output, target))
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
        # Add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.train_data_loader, 'n_samples'):
            current = batch_idx * self.train_data_loader.batch_size
            total = self.train_data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
