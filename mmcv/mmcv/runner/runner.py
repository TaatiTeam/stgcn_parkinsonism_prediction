# Copyright (c) Open-MMLab. All rights reserved.
import logging
import os.path as osp
import time
import shutil

from icecream import ic

import math
from torch import nn

import torch
import csv
import matplotlib.pyplot as plt

import mmcv
from .checkpoint import load_checkpoint, save_checkpoint
from .dist_utils import get_dist_info
from .hooks import HOOKS, Hook, IterTimerHook
from .log_buffer import LogBuffer
from .priority import get_priority
from .utils import get_host_info, get_time_str, obj_from_dict
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
from .pytorchtools import EarlyStopping
import os
import wandb

def weight_reset(m):
    if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.Linear):
        m.reset_parameters()


class TooManyRetriesException(Exception):
    """Exception raised for errors in the input.

    """
    def __init__(self, message="Too many retries"):
        self.message = message
        super().__init__(self.message)



class Runner(object):
    """A training helper for PyTorch.

    Args:
        model (:obj:`torch.nn.Module`): The model to be run.
        batch_processor (callable): A callable method that process a data
            batch. The interface of this method should be
            `batch_processor(model, data, train_mode) -> dict`
        optimizer (dict or :obj:`torch.optim.Optimizer`): If it is a dict,
            runner will construct an optimizer according to it.
        work_dir (str, optional): The working directory to save checkpoints
            and logs.
        log_level (int): Logging level.
        logger (:obj:`logging.Logger`): Custom logger. If `None`, use the
            default logger.
        meta (dict | None): A dict records some import information such as
            environment info and seed, which will be logged in logger hook.
    """

    def __init__(self,
                 model,
                 batch_processor,
                 optimizer=None,
                 work_dir=None,
                 log_level=logging.INFO,
                 logger=None,
                 meta=None, 
                 things_to_log=None,
                 early_stopping=False,
                 force_run_all_epochs=True, 
                 es_patience=10, 
                 es_start_up=50, 
                 freeze_encoder=False, 
                 finetuning=False,
                 visualize_preds={'visualize': False}, 
                 num_class=4,
                 log_conf_mat=False,
                 ):

        assert callable(batch_processor)
        self.model = model
        if optimizer is not None:
            self.optimizer = self.init_optimizer(optimizer)
        else:
            self.optimizer = None
        self.batch_processor = batch_processor
        self.things_to_log = things_to_log
        self.early_stopping = early_stopping
        self.force_run_all_epochs = force_run_all_epochs
        self.freeze_encoder = freeze_encoder

        self.es_patience = es_patience
        self.es_start_up = es_start_up

        self.finetuning = finetuning
        self.visualize_preds = visualize_preds
        self.num_class = num_class
        self.log_conf_mat = log_conf_mat
        
        # create work_dir
        if mmcv.is_str(work_dir):
            self.work_dir = osp.abspath(work_dir)
            print("workdir in mmcv runner is: ", self.work_dir)
            mmcv.mkdir_or_exist(self.work_dir)

        elif work_dir is None:
            self.work_dir = None
        else:
            raise TypeError('"work_dir" must be a str or None')

        # get model name from the model class
        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

        self._rank, self._world_size = get_dist_info()
        self.timestamp = get_time_str()



        if logger is None:
            self.logger = self.init_logger(work_dir, log_level)
        else:
            self.logger = logger
        self.log_buffer = LogBuffer()

        if meta is not None:
            assert isinstance(meta, dict), '"meta" must be a dict or None'
        self.meta = meta

        self.mode = None
        self._hooks = []
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0
        self._max_epochs = 0
        self._max_iters = 0

    @property
    def model_name(self):
        """str: Name of the model, usually the module class name."""
        return self._model_name

    @property
    def rank(self):
        """int: Rank of current process. (distributed training)"""
        return self._rank

    @property
    def world_size(self):
        """int: Number of processes participating in the job.
        (distributed training)"""
        return self._world_size

    @property
    def hooks(self):
        """list[:obj:`Hook`]: A list of registered hooks."""
        return self._hooks

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    @property
    def inner_iter(self):
        """int: Iteration in an epoch."""
        return self._inner_iter

    @property
    def max_epochs(self):
        """int: Maximum training epochs."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Maximum training iterations."""
        return self._max_iters

    def init_optimizer(self, optimizer):
        """Init the optimizer.

        Args:
            optimizer (dict or :obj:`~torch.optim.Optimizer`): Either an
                optimizer object or a dict used for constructing the optimizer.

        Returns:
            :obj:`~torch.optim.Optimizer`: An optimizer object.

        Examples:
            >>> optimizer = dict(type='SGD', lr=0.01, momentum=0.9)
            >>> type(runner.init_optimizer(optimizer))
            <class 'torch.optim.sgd.SGD'>
        """
        if isinstance(optimizer, dict):
            optimizer = obj_from_dict(optimizer, torch.optim,
                                      dict(params=self.model.parameters()))
        elif not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                'optimizer must be either an Optimizer object or a dict, '
                f'but got {type(optimizer)}')
        return optimizer

    def _add_file_handler(self,
                          logger,
                          filename=None,
                          mode='w',
                          level=logging.INFO):
        # TODO: move this method out of runner
        file_handler = logging.FileHandler(filename, mode)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
        return logger

    def init_logger(self, log_dir=None, level=logging.INFO):
        """Init the logger.

        Args:
            log_dir(str, optional): Log file directory. If not specified, no
                log file will be used.
            level (int or str): See the built-in python logging module.

        Returns:
            :obj:`~logging.Logger`: Python logger.
        """
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s', level=level)
        logger = logging.getLogger(__name__)
        print("logger in here: ", __name__)
        if log_dir and self.rank == 0:
            filename = f'{self.timestamp}.log'
            log_file = osp.join(log_dir, filename)
            self._add_file_handler(logger, log_file, level=level)
        return logger

    def current_lr(self):
        """Get current learning rates.

        Returns:
            list: Current learning rate of all param groups.
        """
        if self.optimizer is None:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')
        return [group['lr'] for group in self.optimizer.param_groups]

    def current_momentum(self):
        """Get current momentums.

        Returns:
            list: Current momentum of all param groups.
        """
        if self.optimizer is None:
            raise RuntimeError(
                'momentum is not applicable because optimizer does not exist.')
        momentums = []
        for group in self.optimizer.param_groups:
            if 'momentum' in group.keys():
                momentums.append(group['momentum'])
            elif 'betas' in group.keys():
                momentums.append(group['betas'][0])
            else:
                momentums.append(0)
        return momentums

    def register_hook(self, hook, priority='NORMAL'):
        """Register a hook into the hook list.

        Args:
            hook (:obj:`Hook`): The hook to be registered.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority
        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)

    def call_hook(self, fn_name):
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    def load_checkpoint(self, filename, map_location='cpu', strict=False):
        self.logger.info('load checkpoint from %s', filename)
        return load_checkpoint(self.model, filename, map_location, strict,
                               self.logger)

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        else:
            meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            mmcv.symlink(filename, osp.join(out_dir, 'latest.pth'))



    def visualize_preds_func(self, outputs, data_batch, force_vis=False):
        if not self.visualize_preds['visualize']:
            return

        try: 
            if force_vis or self.visualize_preds['epochs_to_visualize'] == 'all' or \
            (self.epoch in self.visualize_preds['epochs_to_visualize'] ) or \
            ('first' in self.visualize_preds['epochs_to_visualize'] and self.epoch == 0 and self.mode == 'train') or \
            ('first' in self.visualize_preds['epochs_to_visualize'] and self.epoch == 1 and self.mode != 'train'):
                outputs = outputs['predicted_joint_positions'].cpu().data.numpy()
                # Save the results
                print('SAVING PREDS: epoch', str(self.epoch), " mode: ", self.mode)
                try:
                    data, data_flipped, label, name, num_ts= data_batch
                except:
                    data, data_flipped, label, name, num_ts, true_future_ts = data_batch

                # Reshape the data so that both the data and labels/preds are in the order:
                # [bs, coords, num_joints, num_ts]
                data = data.permute(0, 1, 3, 2, 4).contiguous()
                data = data.squeeze()


                neighbor_1base = [[12, 10], [11, 9], [10, 8], [9, 7], [8, 7],
                                    [7, 1], [8, 2], [2, 0], [1, 3], [2, 4], 
                                    [4, 6], [0, 2], [0, 1]]

                order_of_keypoints = ['Nose', 
                    'LShoulder', 'RShoulder',
                    'LElbow', 'RElbow', 
                    'LWrist', 'RWrist', 
                    'LHip', 'RHip',
                    'LKnee', 'RKnee',
                    'LAnkle', 'RAnkle',
                ]

                time_axis_scale = 15
                num_frames_btwn_plots = 30

                data_size = data.shape
                data_length = data_size[3]
                # Plot the first part of the walk
                data = data.numpy()
                label = label.numpy()

                for walk_num in range(data_size[0]):
                    plt.close('all')
                    # New figure for each participant
                    fig = plt.figure(figsize=(40,20))
                    ax = fig.gca(projection='3d')


                    for ts in range(0, data_length, num_frames_btwn_plots):
                        time_axis = ts/time_axis_scale

                        xcords = data[walk_num, 0, :, ts].squeeze()

                        ycords = data[walk_num, 1, :, ts].squeeze()


                        for i in range(len(neighbor_1base)):
                            xs = xcords[neighbor_1base[i]]
                            ys = ycords[neighbor_1base[i]]
                            ax.plot(xs, ys, zs=time_axis, zdir='y', color='k')




                    # Plot the true and predicted points at future timesteps
                    future_ts = [arr[0] for arr in true_future_ts['pred_ts']]
                    predicted_joints = [arr[0] for arr in true_future_ts['joints']]

                    data_true = true_future_ts['true_skel']
                    # data_true = data_true.squeeze()
                    data_true = data_true.numpy()
                    for t in range(len(future_ts)):
                        local_ts = future_ts[t]
                        ts = data_length + local_ts

                        time_axis = ts/time_axis_scale

                        # Plot the baseline we should be predicting
                        xcords = data_true[walk_num, 0, :, t].squeeze()
                        ycords = data_true[walk_num, 1, :, t].squeeze()

                        for i in range(len(neighbor_1base)):
                            xs = xcords[neighbor_1base[i]]
                            ys = ycords[neighbor_1base[i]]
                            ax.plot(xs, ys, zs=time_axis, zdir='y', color='g')

                        # Plot the predictions
                        for j in range(len(predicted_joints)):
                            ax.scatter(outputs[walk_num, 0, j, t], outputs[walk_num, 1, j, t], zs=time_axis, zdir='y', s=100, label=order_of_keypoints[predicted_joints[j]]+ '_' + str(local_ts.data.numpy()))


                    x_lim_min = min(np.amin(data[walk_num, 0, :, :]), np.amin(data_true[walk_num, 0, :, :]), np.amin(outputs[walk_num, 0, :, :]))
                    x_lim_max = max(np.amax(data[walk_num, 0, :, :]), np.amax(data_true[walk_num, 0, :, :]), np.max(outputs[walk_num, 0, :, :]))
                    y_lim_min = min(np.amin(data[walk_num, 1, :, :]), np.amin(data_true[walk_num, 1, :, :]), np.amin(outputs[walk_num, 1, :, :]))
                    y_lim_max = max(np.amax(data[walk_num, 1, :, :]), np.amax(data_true[walk_num, 1, :, :]), np.amax(outputs[walk_num, 1, :, :]))

                    mse = np.sum((outputs[walk_num, :, :, :] - label[walk_num, :, :, :])**2)

                    ax.legend(fontsize='xx-large')
                    ax.set_xlim(x_lim_min, x_lim_max)

                    ax.set_ylim(0, (data_length + max(future_ts))/time_axis_scale + 1)
                    ax.set_zlim(y_lim_min, y_lim_max)
                    plt.gca().invert_zaxis()
                    ax.set_ylim(ax.get_ylim()[::-1])
                    ax.set_title('MSE of all predictions: ' + str(mse))

                    ax.set_xlabel('')
                    ax.set_ylabel('')
                    ax.set_zlabel('')

                    output_folder = os.path.join(self.visualize_preds['output_dir'], self.mode)  
                    mmcv.mkdir_or_exist(output_folder)

                    fig_name, _ = os.path.splitext(name[walk_num])
                    _, name_clean = os.path.split(fig_name)
                    fig_save = os.path.join(output_folder, name_clean + '_epoch' + str(self.epoch) + '_.png')
                    # fig.savefig(fig_save)

            else: 
                return
        
        except:
            print("failed to save predictions...")
            logging.exception("Error message =================================================")


    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        true_labels, predicted_labels, pred_raw, raw_labels, non_pseudo_label = [], [], [], [], []
        batch_loss = 0

        self.call_hook('before_train_epoch')
        train_extrema_for_epochs = -1
        # Should we use extrema or all data?
        if 'train_extrema_for_epochs' in kwargs:
            train_extrema_for_epochs = kwargs['train_extrema_for_epochs']

        data_loader.dataset.data_source.sample_extremes = False
        if self._epoch <= train_extrema_for_epochs:
            data_loader.dataset.data_source.sample_extremes = True
        
        try: 
            kwargs['class_weights_dict'][self.mode] = data_loader.dataset.data_source.get_class_dist()
        except:
            pass

        for i, data_batch in enumerate(data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            outputs, raw, overall_loss = self.batch_processor(
                self.model, data_batch, train_mode=True, num_class=self.num_class, **kwargs)

            self.visualize_preds_func(outputs, data_batch)


            try:
                overall_loss_np = overall_loss.cpu().data.numpy()
            except: 
                overall_loss_np = overall_loss
            self.visualize_preds_func(outputs, data_batch)


            if not np.isnan(overall_loss_np):
                batch_loss += overall_loss*len(raw['true'])
            else:
                pass
            true_labels.extend(raw['true'])
            predicted_labels.extend(raw['pred'])
            pred_raw.extend(raw['raw_preds'])

            try:
                raw_labels.extend(raw['raw_labels'])
                non_pseudo_label.extend(raw['non_pseudo_label'])

            except:
                pass

            if not isinstance(outputs, dict):
                raise TypeError('batch_processor() must return a dict')
            if 'log_vars' in outputs and not self.pretrain_mode:
                self.log_buffer.update(outputs['log_vars'],
                                       outputs['num_samples'])

                                  
            self.outputs = outputs
            if not np.isnan(overall_loss_np):
                self.call_hook('after_train_iter') # the backward step is called here
            self._iter += 1
        self._epoch += 1


        batch_loss = batch_loss / len(true_labels)

        if not self.pretrain_mode:
            acc = accuracy_score(true_labels, predicted_labels)
            log_this = {'accuracy': acc}
            self.log_buffer.update(log_this, 1) 

            self.preds = predicted_labels
            self.labels = true_labels
            self.preds_raw = pred_raw
            self.raw_labels = raw_labels
            self.non_pseudo_label = non_pseudo_label
            self.call_hook('after_val_epoch')

            self.call_hook('after_train_epoch')

        else:
            log_this = {'pretrain_loss': batch_loss}
            self.log_buffer.update(log_this, 1) 
            self.call_hook('buffer_log_only')
        # print("end training epoch")
        return true_labels, predicted_labels

    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        true_labels, predicted_labels, pred_raw, raw_labels, non_pseudo_label = [], [], [], [], []
        batch_loss = 0

        for i, data_batch in enumerate(data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            with torch.no_grad():
                outputs, raw, overall_loss = self.batch_processor(
                    self.model, data_batch, train_mode=False, num_class=self.num_class, **kwargs)
                true_labels.extend(raw['true'])
                predicted_labels.extend(raw['pred'])
                pred_raw.extend(raw['raw_preds'])

                try:
                    raw_labels.extend(raw['raw_labels'])
                    non_pseudo_label.extend(raw['non_pseudo_label'])

                except:
                    pass


                try:
                    overall_loss_np = overall_loss.cpu().data.numpy()
                except: 
                    overall_loss_np = overall_loss
                self.visualize_preds_func(outputs, data_batch)

                if not np.isnan(overall_loss_np):
                    batch_loss += overall_loss_np*len(raw['true'])

            if not isinstance(outputs, dict):
                raise TypeError('batch_processor() must return a dict')
            if 'log_vars' in outputs:
                self.log_buffer.update(outputs['log_vars'],
                                       outputs['num_samples'])
            self.outputs = outputs
            self.call_hook('after_val_iter')
        

        batch_loss = batch_loss / len(true_labels)
        # true_labels, predicted_labels = self.remove_non_labelled_data(true_labels, predicted_labels)
        if not self.pretrain_mode:
            acc = accuracy_score(true_labels, predicted_labels)
            log_this = {'accuracy': acc}
            self.log_buffer.update(log_this, 1) 

            self.preds = predicted_labels
            self.labels = true_labels
            self.preds_raw = pred_raw
            self.raw_labels = raw_labels
            self.non_pseudo_label = non_pseudo_label

        self.early_stopping_epoch = self.epoch
        if self.early_stopping and not self.early_stopping_obj.early_stop and self.epoch >= self.es_start_up:
            self.es_before_step = self.early_stopping_obj.early_stop
            self.early_stopping_obj(batch_loss, self.model)

            if self.es_before_step == False and self.early_stopping_obj.early_stop == True:
                self.early_stopping_epoch = self.epoch - self.es_patience

                if not self.pretrain_mode:
                    self.log_buffer.update({'stop_epoch_val': self.early_stopping_epoch}, 1)
                    print("Updated the buffer with the stop epoch: ", self.early_stopping_epoch)
        
        if not self.early_stopping and self.epoch == self._max_epochs: #dont have early stopping
            torch.save(self.model.state_dict(), self.es_checkpoint)


        if self.pretrain_mode:
            log_this = {'pretrain_loss': batch_loss}
            self.log_buffer.update(log_this, 1) 
            self.call_hook('buffer_log_only')        
        
        else:
            self.call_hook('after_val_epoch')

        return true_labels, predicted_labels


    def test(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'test'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        true_labels, predicted_labels, pred_raw, raw_labels, non_pseudo_label = [], [], [], [], []
        demo_data = {}
        batch_loss = 0

        for i, data_batch in enumerate(data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            with torch.no_grad():
                outputs, raw, overall_loss = self.batch_processor(
                    self.model, data_batch, train_mode=False, num_class=self.num_class, **kwargs)
                true_labels.extend(raw['true'])
                predicted_labels.extend(raw['pred'])
                pred_raw.extend(raw['raw_preds'])

                try:
                    raw_labels.extend(raw['raw_labels'])
                    non_pseudo_label.extend(raw['non_pseudo_label'])
                except:
                    pass

                try:
                    demo_data_batch = outputs['demo_data']
                    for k in demo_data_batch:
                        if k in demo_data:
                            demo_data[k].extend(demo_data_batch[k].detach().cpu().tolist())
                        else:
                            demo_data[k] = demo_data_batch[k].detach().cpu().tolist()
                except:
                    pass

                try:
                    overall_loss_np = overall_loss.cpu().data.numpy()
                except: 
                    overall_loss_np = overall_loss
                self.visualize_preds_func(outputs, data_batch)

                if not np.isnan(overall_loss_np):
                    batch_loss += overall_loss*len(raw['true'])

            if not isinstance(outputs, dict):
                raise TypeError('batch_processor() must return a dict')
            if 'log_vars' in outputs:
                self.log_buffer.update(outputs['log_vars'],
                                       outputs['num_samples'])
            self.outputs = outputs
            self.call_hook('after_val_iter')
        


        batch_loss = batch_loss / len(true_labels)

        log_this = demo_data
        log_this = {}
        if not self.pretrain_mode:
            acc = accuracy_score(true_labels, predicted_labels)
            log_this.update({'accuracy': acc})
            self.log_buffer.update(log_this, 1) 

            self.non_pseudo_label = non_pseudo_label
            self.preds = predicted_labels
            self.labels = true_labels
            self.preds_raw = pred_raw
            self.raw_labels = raw_labels
            self.call_hook('after_val_epoch')
    
        else:
            log_this.update({'pretrain_loss': batch_loss})
            self.log_buffer.update(log_this, 1) 
            self.call_hook('buffer_log_only')

        return true_labels, predicted_labels


    def remove_non_labelled_data(self, true_labels, pred_labels):
        true_np = np.asarray(true_labels)
        pred_np = np.asarray(pred_labels)
        keep = np.argwhere(true_np >= 0 ).transpose().squeeze()
        # keep = np.transpose(keep).squeeze()
        pred_labels = pred_np[keep]
        true_labels = true_np[keep]
        return list(true_labels), list(pred_labels)


    def resume(self,
               checkpoint,
               resume_optimizer=True,
               map_location='default'):
        if map_location == 'default':
            device_id = torch.cuda.current_device()
            checkpoint = self.load_checkpoint(
                checkpoint,
                map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)

        self._epoch = checkpoint['meta']['epoch']
        self._iter = checkpoint['meta']['iter']
        if 'optimizer' in checkpoint and resume_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info('resumed epoch %d, iter %d', self.epoch, self.iter)

    def basic_no_log_eval(self, data_loader, **kwargs):
        self.model.eval()
        self.data_loader = data_loader
        true_labels, predicted_labels, pred_raw = [], [], []
        names, num_ts = [], []
        batch_loss = 0
        demo_data = {}
        all_data = {}
        all_data['all_true'] = []
        all_data['all_pred_round'] = []
        all_data['all_pred_raw'] = []
        

        for i, data_batch in enumerate(data_loader):
            self._inner_iter = i
            with torch.no_grad():
                outputs, raw, overall_loss = self.batch_processor(
                    self.model, data_batch, train_mode=False, num_class=self.num_class, **kwargs)
                true_labels.extend(raw['true'])
                try:
                    all_data['all_true'].extend(raw['raw_labels'])
                    all_data['all_pred_round'].extend(raw['round_preds_all'])
                    all_data['all_pred_raw'].extend(raw['raw_preds_all'])
                    predicted_labels.extend(raw['pred'])
                    pred_raw.extend(raw['raw_preds'])
                    names.extend(raw['name'])
                    num_ts.extend(raw['num_ts'])
                    batch_loss += overall_loss*len(raw['true'])
                    self.visualize_preds_func(outputs, data_batch, True)
                except Exception as e:
                    ic(e)
                    pass

                try:
                    demo_data_batch = outputs['demo_data']
                    for k in demo_data_batch:
                        if k in demo_data:
                            demo_data[k].extend(demo_data_batch[k].detach().cpu().tolist())
                        else:
                            demo_data[k] = demo_data_batch[k].detach().cpu().tolist()
                except:
                    pass



        return true_labels, predicted_labels, pred_raw, names, num_ts, demo_data, all_data

    def basic_no_log_eval_pretrain(self, data_loader, **kwargs):
        self.model.eval()
        self.data_loader = data_loader
        true_labels, predicted_labels, pred_raw = [], [], []
        names, num_ts = [], []

        predicted_joint_positions, last_joint_position, true_future_joint_positions = [], [], []
        predicted_joint_positions_ref, last_joint_position_ref, true_future_joint_positions_ref = [], [], []
        batch_loss = 0

        for i, data_batch in enumerate(data_loader):
            self._inner_iter = i
            with torch.no_grad():
                outputs, raw, overall_loss = self.batch_processor(
                    self.model, data_batch, train_mode=False, num_class=self.num_class, **kwargs)
                predicted_joint_positions.extend(raw['predicted_joint_positions'])
                last_joint_position.extend(raw['last_joint_position'])
                true_future_joint_positions.extend(raw['true_future_joint_positions'])
                predicted_joint_positions_ref.extend(raw['predicted_joint_positions_ref'])
                last_joint_position_ref.extend(raw['last_joint_position_ref'])
                true_future_joint_positions_ref.extend(raw['true_future_joint_position_ref'])
                names.extend(raw['names'])


        return predicted_joint_positions, last_joint_position, true_future_joint_positions, predicted_joint_positions_ref, last_joint_position_ref, true_future_joint_positions_ref, names


    def early_stop_eval_pretrain(self, es_checkpoint, workflow, data_loaders, **kwargs):

        self.model.eval()
        for i, flow in enumerate(workflow):  # Should only have test in the workflow
            self.mode, _ = flow

            # mode = "train", "val", "test"
            predicted_joint_positions, last_joint_position, true_future_joint_positions, predicted_joint_positions_ref, last_joint_position_ref, true_future_joint_positions_ref, names = self.basic_no_log_eval_pretrain(data_loaders[i], **kwargs)


        return predicted_joint_positions, last_joint_position, true_future_joint_positions, predicted_joint_positions_ref, last_joint_position_ref, true_future_joint_positions_ref, names


    def eval_pretrain(self, data_loaders, workflow, max_epochs, **kwargs):
        """Evaluate the performance of a pretrained position model.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
            max_epochs (int): Total training epochs.
        """
        not_done = True
        max_retry = 10
        max_retry_counter = 0

        try:
            self.pretrain_mode = kwargs['supcon_pretraining']
        except:
            self.pretrain_mode = False

        kwargs = {k: v for k, v in kwargs.items() if k != 'supcon_pretraining'}
        # Freeze the encoder if needed
        if self.freeze_encoder:
            for param in self.model.module.encoder.parameters():
                param.requires_grad = False

        # Reset the epoch counters
        self.mode = None
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0
        self._max_epochs = 0
        self._max_iters = 0

        es_checkpoint = self.work_dir + '/checkpoint.pt'
        self.es_checkpoint = es_checkpoint
        if self.early_stopping:
            self.early_stopping_obj = EarlyStopping(patience=self.es_patience, verbose=True, path=es_checkpoint)

        self._max_epochs = max_epochs

        predicted_joint_positions, last_joint_position, true_future_joint_positions, predicted_joint_positions_ref, last_joint_position_ref, true_future_joint_positions_ref, names = self.early_stop_eval_pretrain(es_checkpoint, workflow, data_loaders, **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')
        self.early_stopping_epoch = 0
        return predicted_joint_positions, last_joint_position, true_future_joint_positions, predicted_joint_positions_ref, last_joint_position_ref, true_future_joint_positions_ref, names



    def run(self, data_loaders, workflow, max_epochs, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
            max_epochs (int): Total training epochs.
        """
        not_done = True
        max_retry = 10
        max_retry_counter = 0

        try:
            self.pretrain_mode = kwargs['supcon_pretraining']
        except:
            self.pretrain_mode = False

        kwargs = {k: v for k, v in kwargs.items() if k != 'supcon_pretraining'}
        # Freeze the encoder if needed
        if self.freeze_encoder:
            for param in self.model.module.encoder.parameters():
                param.requires_grad = False

        while not_done:
            print('===================starting training...=========================')
            # Reset the epoch counters
            self.mode = None
            self._epoch = 0
            self._iter = 0
            self._inner_iter = 0
            self._max_epochs = 0
            self._max_iters = 0

            try: 
                print("Starting training for ", self.work_dir)
                assert isinstance(data_loaders, list)
                assert mmcv.is_list_of(workflow, tuple)
                assert len(data_loaders) == len(workflow)

                if self.pretrain_mode:
                    es_checkpoint = self.work_dir + '/pretrain_checkpoint.pt'
                else:
                    es_checkpoint = self.work_dir + '/checkpoint.pt'
                self.es_checkpoint = es_checkpoint
                if self.early_stopping:
                    self.early_stopping_obj = EarlyStopping(patience=self.es_patience, verbose=True, path=es_checkpoint)

                self._max_epochs = max_epochs
                for i, flow in enumerate(workflow):
                    mode, epochs = flow

                    if mode == 'train':
                        self._max_iters = self._max_epochs * len(data_loaders[i])
                        break

                work_dir = self.work_dir if self.work_dir is not None else 'NONE'
                self.logger.info('Start running, host: %s, work_dir: %s',
                                get_host_info(), work_dir)
                self.logger.info('workflow: %s, max: %d epochs', workflow, max_epochs)
                self.call_hook('before_run')
                print("work dir: ", self.work_dir)
                train_accs = np.zeros((1, max_epochs)) * np.nan
                val_accs = np.zeros((1, max_epochs)) * np.nan
                
                columns = ['epoch', 'train_acc', 'val_acc']
                df_all = pd.DataFrame(columns=columns)
                print('mmcv - runner.run(), max_epochs')
                while self.epoch < max_epochs:
                    for i, flow in enumerate(workflow):
                        mode, epochs = flow
                        kwargs['workflow_stage'] = mode
                        kwargs['cur_epoch'] = self.epoch

                        if isinstance(mode, str):  # self.train()
                            if not hasattr(self, mode):
                                raise ValueError(
                                    f'runner has no method named "{mode}" to run an '
                                    'epoch')
                            epoch_runner = getattr(self, mode)
                        elif callable(mode):  # custom train()
                            epoch_runner = mode
                        else:
                            raise TypeError('mode in workflow must be a str or '
                                            f'callable function, not {type(mode)}')
                        for _ in range(epochs):
                            if mode == 'train' and self.epoch >= max_epochs:
                                return
                            true_labels, predicted_labels = epoch_runner(data_loaders[i], **kwargs)

                            if not self.pretrain_mode:
                                acc = accuracy_score(true_labels, predicted_labels)

                                if mode == 'train':
                                    df_all.loc[len(df_all)] = [self.epoch-1, acc, val_accs[0, self.epoch - 1]]

                                elif mode == 'val':
                                    val_accs[0, self.epoch-1] = acc
                                    df_all.loc[df_all['epoch'] == self.epoch-1,'val_acc'] = acc
                            else: 
                                # What to do after epoch if we're in pretrain mode
                                pass
                    if not self.pretrain_mode:
                        mmcv.mkdir_or_exist(self.work_dir)
                        df_all.to_csv(self.work_dir + "/results_df.csv")

                    if self.early_stopping:
                        print('checking early stopping:', self.early_stopping_obj.early_stop, self.force_run_all_epochs)
                        if not self.force_run_all_epochs and self.early_stopping_obj.early_stop:
                            print('should STOP now')
                            break

                    # We have successfully finished this participant
                    not_done = False

            except Exception as e: 
                not_done = True
                logging.exception("Error message =================================================")
                max_retry_counter += 1
                # Reset the model parameters
                print("======================================going to retrain again, resetting parameters...")
                print("This is the error we got:", e)
                try:
                    self.model.module.apply(weight_reset)
                    if os.path.isfile(self.es_checkpoint):
                        os.remove(self.es_checkpoint)
                        print('deleted pretrain')
                    print('successfully reset weights')
                except Exception as e: 
                    print("This is the error we got _ 2:", e)


                try:
                    shutil.rmtree(self.work_dir)
                except:
                    print('failed to delete the self.work_dir folder')


                if max_retry_counter >= max_retry:
                    not_done = False
                    raise TooManyRetriesException



        # If we stopped early, evaluate the performance of the saved model on all datasets
        if self.early_stopping:
            try:
                self.log_buffer.update({'early_stop_epoch': self.early_stopping_epoch}, 1) 
                print('stopped at epoch: ', self.early_stopping_epoch)
            except:
                print("didn't meet early stopping criterion so we ran for all epochs")
            print("*****************************now doing eval: ")

            if not self.pretrain_mode:
                self.model.load_state_dict(torch.load(es_checkpoint))
                self.model.eval()
                self.early_stop_eval(workflow, data_loaders, **kwargs)
            else:
                pass



        time.sleep(10)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

        return self.model, self.early_stopping_epoch


    def early_stop_eval(self, workflow, data_loaders, **kwargs):
        self.model.eval()
        print("doing final eval:")
        for i, flow in enumerate(workflow):
            mode, _ = flow
            print('now evaluating: ', mode)
            kwargs['workflow_stage'] = mode

            # mode = "train", "val", "test"
            true_labels, predicted_labels, raw_preds, names, num_ts, demo_data, all_data  = self.basic_no_log_eval(data_loaders[i], **kwargs)
            acc = accuracy_score(true_labels, predicted_labels)
            final_results_base, amb = os.path.split(self.work_dir)
            try:
                final_results_path = os.path.join(final_results_base, 'all_final_eval', str(self.things_to_log['num_reps_pd']))
            except:
                final_results_path = os.path.join(final_results_base, 'all_final_eval', self.things_to_log['wandb_group'])


            if mode == 'test':
                final_results_file = os.path.join(final_results_path,'test.csv')
            if mode == 'val':
                final_results_file = os.path.join(final_results_path,'val.csv')
            if mode == 'train':
                final_results_file = os.path.join(final_results_path,'train.csv')           

            print("saving to ", final_results_file)
            mmcv.mkdir_or_exist(final_results_path)
            header = ['amb', 'walk_name', 'num_ts', 'true_score', 'pred_round', 'pred_raw']

            for k in demo_data:
                header.append(k)

            if not os.path.exists(final_results_file):
                with open (final_results_file,'w') as f:                            
                    writer = csv.writer(f, delimiter=',') 
                    writer.writerow(header)

            with open (final_results_file,'a') as f:                            
                writer = csv.writer(f, delimiter=',') 
                for num in range(len(all_data['all_true'])):
                    core_data = [amb, names[num], num_ts[num], all_data['all_true'][num], all_data['all_pred_round'][num], all_data['all_pred_raw'][num]]
                    for k in demo_data:
                        core_data.append(demo_data[k][num])

                    writer.writerow(core_data)


    def register_lr_hook(self, lr_config):
        if isinstance(lr_config, dict):
            assert 'policy' in lr_config
            policy_type = lr_config.pop('policy')
            # If the type of policy is all in lower case, e.g., 'cyclic',
            # then its first letter will be capitalized, e.g., to be 'Cyclic'.
            # This is for the convenient usage of Lr updater updater.
            # Since this is not applicable for `CosineAnealingLrUpdater`,
            # the string will not be changed if it contains capital letters.
            if policy_type == policy_type.lower():
                policy_type = policy_type.title()
            hook_type = policy_type + 'LrUpdaterHook'
            lr_config['type'] = hook_type
            hook = mmcv.build_from_cfg(lr_config, HOOKS)
        else:
            hook = lr_config
        self.register_hook(hook)

    def register_optimizer_hook(self, optimizer_config):
        if optimizer_config is None:
            return
        if isinstance(optimizer_config, dict):
            optimizer_config.setdefault('type', 'OptimizerHook')
            hook = mmcv.build_from_cfg(optimizer_config, HOOKS)
        else:
            hook = optimizer_config
        self.register_hook(hook)

    def register_checkpoint_hook(self, checkpoint_config):
        if checkpoint_config is None:
            return
        if isinstance(checkpoint_config, dict):
            checkpoint_config.setdefault('type', 'CheckpointHook')
            hook = mmcv.build_from_cfg(checkpoint_config, HOOKS)
        else:
            hook = checkpoint_config
        self.register_hook(hook)

    def register_momentum_hook(self, momentum_config):
        if momentum_config is None:
            return
        if isinstance(momentum_config, dict):
            assert 'policy' in momentum_config
            policy_type = momentum_config.pop('policy')
            # If the type of policy is all in lower case, e.g., 'cyclic',
            # then its first letter will be capitalized, e.g., to be 'Cyclic'.
            # This is for the convenient usage of momentum updater.
            # Since this is not applicable for `CosineAnealingMomentumUpdater`,
            # the string will not be changed if it contains capital letters.
            if policy_type == policy_type.lower():
                policy_type = policy_type.title()
            hook_type = policy_type + 'MomentumUpdaterHook'
            momentum_config['type'] = hook_type
            hook = mmcv.build_from_cfg(momentum_config, HOOKS)
        else:
            hook = momentum_config
        self.register_hook(hook)

    def register_logger_hooks(self, log_config):
        log_interval = log_config['interval']
        for info in log_config['hooks']:
            logger_hook = mmcv.build_from_cfg(
                info, HOOKS, default_args=dict(interval=log_interval, initial_config=self.things_to_log))
            self.register_hook(logger_hook, priority='VERY_LOW')

    def register_training_hooks(self,
                                lr_config,
                                optimizer_config=None,
                                checkpoint_config=None,
                                log_config=None,
                                momentum_config=None):
        """Register default hooks for training.

        Default hooks include:

        - LrUpdaterHook
        - MomentumUpdaterHook
        - OptimizerStepperHook
        - CheckpointSaverHook
        - IterTimerHook
        - LoggerHook(s)
        """
        self.register_lr_hook(lr_config)
        self.register_momentum_hook(momentum_config)
        self.register_optimizer_hook(optimizer_config)
        self.register_checkpoint_hook(checkpoint_config)
        self.register_hook(IterTimerHook())
        self.register_logger_hooks(log_config)
