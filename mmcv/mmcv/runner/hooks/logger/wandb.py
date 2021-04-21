# Copyright (c) Open-MMLab. All rights reserved.
import numbers

from mmcv.runner import master_only
from ..hook import HOOKS
from .base import LoggerHook
import collections
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt

#https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

@HOOKS.register_module()
class WandbLoggerHook(LoggerHook):

    def __init__(self,
                 init_kwargs=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=True, 
                 initial_config=None):
        super(WandbLoggerHook, self).__init__(interval, ignore_last,
                                              reset_flag)
        self.import_wandb()
        self.init_kwargs = init_kwargs
        self.initial_config = flatten(initial_config)
        self.group = None

    def import_wandb(self):
        try:
            import wandb
        except ImportError:
            raise ImportError(
                'Please run "pip install wandb" to install wandb')
        self.wandb = wandb

    @master_only
    def before_run(self, runner):
        # input("before run")
        if self.wandb is None:

            self.import_wandb()


        if self.init_kwargs:

            self.wandb.init(**self.init_kwargs)
        elif self.initial_config:
            print('initializing with: ', self.initial_config)
            run_name = "AMB"+str(self.initial_config['test_AMBID'])

            if runner.finetuning:
                run_name = run_name + "_FT"

            try:
                if self.initial_config['self_train_iteration_count'] is not None:
                    run_name = run_name + '_' + str(self.initial_config['self_train_iteration_count'])
            except: # Don't have the count iteration so dont include it
                pass

            try:
                self.wandb.init(dir = self.initial_config['dir'], project=self.initial_config['wandb_project'], config=self.initial_config, group=self.initial_config['wandb_group'], name=run_name, reinit=True)
            except:
                self.wandb.init(project=self.initial_config['wandb_project'], config=self.initial_config, group=self.initial_config['wandb_group'], name=run_name, reinit=True)


        else:
            self.wandb.init()
        
        print("the run dir is: ", self.wandb.run.dir)

    @master_only
    def log(self, runner):
        metrics = {}
        for var, val in runner.log_buffer.output.items():
            if var in ['time', 'data_time', 'confusion_matrix', 'regression_plot', 'confusion_matrix_normed']:
                continue
            tag = f'{runner.mode}/{var}'
            metrics[tag] = val
        metrics['learning_rate'] = runner.current_lr()[0]
        metrics['momentum'] = runner.current_momentum()[0]
        try:
            if metrics:
                self.wandb.log(metrics, step=runner._epoch)
            print("logging: ", metrics, ", epoch: ", runner._epoch)

            for plot_type in ['confusion_matrix', 'regression_plot', 'confusion_matrix_normed']:
                if plot_type in runner.log_buffer.output:
                    self.wandb.log({plot_type + "/" + runner.log_buffer.output[plot_type][1]: [self.wandb.Image(runner.log_buffer.output[plot_type][0])]}, step=runner._epoch)
        except:
            print("SOMETHING WENT WRONG WITH WANDB LOG")


    @master_only
    def after_run(self, runner):
        self.wandb.join()
