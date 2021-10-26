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

import torch
from abc import abstractmethod
from numpy import inf 
from logger import TensorboardWriter
import pdb

class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, model_float, criterion, metric_ftns, optimizer, optimizer_float, arch_optimizer, finetuning, config, args_input):
        self.config = config
        self.args = args_input
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
        
        self.model = model
        self.model_float = model_float
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.optimizer_float = optimizer_float
        self.arch_optimizer = arch_optimizer
        self.finetuning = finetuning

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch, prec_float):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        if self.finetuning == 'True':
            print("Float Training")
            not_improved_count = 0
            for epoch in range(self.start_epoch, self.epochs + 1):
                result = self._train_epoch(epoch, self.finetuning)

                # save logged informations into log dicts
                log = {'epoch': epoch}
                log.update(result)
                # print logged information to the screen
                for key, value in log.items():
                    self.logger.info('{:15s}: {}'.format(str(key), value))

                # evaluate model performance according to configured metric, save best checkpoint as model_best
                best = False
                if self.mnt_mode != 'off':
                    try:
                        # check whether the model performance improved or not, according to specified metric (mnt_metric)
                        improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or (
                                    self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                    except KeyError:
                        self.logger.warning("Warning: Metric '{}' is not found."
                                            "Model performance monitoring is disabled".format(self.mnt_metric))
                        self.mnt_mode = 'off'
                        improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1
                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs."
                                     "Training Stops.".format(self.early_stop))
                    break
            #for name_float, param_float, name, param in zip(self.model_float.named_parameters(), self.model.named_parameters()):
            state_dict = self.model.state_dict()
            for name_float, param_float in self.model_float.named_parameters():
                for name, param in self.model.named_parameters():
                    name_float_s = name_float.split('.')
                    if name == name_float or name == ''.join(name_float_s[:-1])+'.linear.'+name_float_s[-1] or name == ''.join(name_float_s[:-1])+'.quantized_weight.'+name_float_s[-1]:
                        state_dict[name] = param_float
            self.model.load_state_dict(state_dict)

        not_improved_count = 0
        print("Quantized Training")
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch, 'False')

            if self.arch_optimizer != False:
                print('========= architecture =========')
                if hasattr(self.model, 'module'):
                    best_arch, bitops, bita, bitw, mixbitops, mixbita, mixbitw = self.model.module.fetch_best_arch()
                else:
                    best_arch, bitops, bita, bitw, mixbitops, mixbita, mixbitw = self.model.fetch_best_arch()
                print('best model with bitops: {:.3f}M, bita: {:.3f}K, bitw: {:.3f}M'.format(
                    bitops, bita, bitw))
                print('expected model with bitops: {:.3f}M, bita: {:.3f}K, bitw: {:.3f}M'.format(
                    mixbitops, mixbita, mixbitw))
                for key, value in best_arch.items():
                    print('{}: {}'.format(key, value))

            # save logged informations into log dicts
            log = {'epoch' : epoch}
            log.update(result)

            # print logged information to the screen
            for key, value in log.items():
                self.logger.info('{:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether the model performance improved or not, according to specified metric (mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found."
                                        "Model performance monitoring is disabled".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

            if improved:
                self.mnt_best = log[self.mnt_metric]
                not_improved_count = 0
                best = True
            else:
                not_improved_count += 1
            if not_improved_count > self.early_stop:
                self.logger.info("Validation performance didn\'t improve for {} epochs."
                                 "Training Stops.".format(self.early_stop))
                if self.arch_optimizer != False:
                    import json
                    complexity = str(int(self.args.complexity_decay*1000000))
                    a_file = open("mix_archs/architecture_"+self.args.arch+complexity+".json", "w")
                    best_arch['best_weight'] = [array_weights.astype('int').tolist() for array_weights in best_arch['best_weight']]
                    import numpy as np
                    best_arch['best_activ'] = np.asarray(best_arch['best_activ']).astype('int').tolist()
                    json.dump(best_arch, a_file)
                break

            if epoch % self.save_period == 0 or best==True:
                self._save_checkpoint(epoch, save_best=best)
    
    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        if self.arch_optimizer != False:
            state = {
                    'arch': arch,
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'arch_optimizer': self.arch_optimizer.state_dict(),
                    'monitor_best': self.mnt_best,
                    'config': self.config
                    }
        else:
            state = {
                    'arch': arch,
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'monitor_best': self.mnt_best,
                    'config': self.config
                    }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")
    
    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.arch_optimizer.load_state_dict(checkpoint['arch_optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

