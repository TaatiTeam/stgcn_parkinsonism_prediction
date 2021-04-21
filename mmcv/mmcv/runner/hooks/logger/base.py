# Copyright (c) Open-MMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

from ..hook import Hook
import os, mmcv

import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.utils.multiclass import unique_labels
import csv, copy
from pathlib import Path



class LoggerHook(Hook):
    """Base class for logger hooks.

    Args:
        interval (int): Logging interval (every k iterations).
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than `interval`.
        reset_flag (bool): Whether to clear the output buffer after logging.
    """

    __metaclass__ = ABCMeta

    def __init__(self, interval=10, ignore_last=True, reset_flag=False, initial_config=None):
        self.interval = interval
        self.ignore_last = ignore_last
        self.reset_flag = reset_flag
        self.initial_config = initial_config
    @abstractmethod
    def log(self, runner):
        pass

    def before_run(self, runner):
        for hook in runner.hooks[::-1]:
            if isinstance(hook, LoggerHook):
                hook.reset_flag = True
                break

    def before_epoch(self, runner):
        runner.log_buffer.clear()  # clear logs of last epoch

    def after_train_iter(self, runner):
        if self.every_n_inner_iters(runner, self.interval):
            runner.log_buffer.average(self.interval)
        elif self.end_of_epoch(runner) and not self.ignore_last:
            # not precise but more stable
            runner.log_buffer.average(self.interval)

        if runner.log_buffer.ready:
            self.log(runner)
            if self.reset_flag:
                runner.log_buffer.clear_output()

    def after_train_epoch(self, runner):
        if runner.log_buffer.ready:
            self.log(runner)
            if self.reset_flag:
                runner.log_buffer.clear_output()

    def buffer_log_only(self, runner):
        runner.log_buffer.average()
        self.log(runner)
        if self.reset_flag:
            runner.log_buffer.clear_output()

            
    def after_val_epoch(self, runner):

        log_stats = False
        final_results_base = str(Path(runner.work_dir).parents[0])
        final_results_base, amb = os.path.split(runner.work_dir)

        final_results_path = os.path.join(final_results_base, 'all_test', runner.things_to_log['wandb_group'])
        if runner.mode == 'test':
            final_results_file = os.path.join(final_results_path,'test_' + str(runner._epoch) + '.csv')
            log_stats = False
        if runner.mode == 'val':
            final_results_file = os.path.join(final_results_path,'val_' + str(runner._epoch) + '.csv')
            log_stats = False

        if log_stats:

            mmcv.mkdir_or_exist(final_results_path)
            header = ['amb', 'true_score', 'pred_round', 'pred_raw']

            if not os.path.exists(final_results_file):
                with open (final_results_file,'w') as f:                            
                    writer = csv.writer(f, delimiter=',') 
                    writer.writerow(header)


            with open (final_results_file,'a') as f:                            
                writer = csv.writer(f, delimiter=',') 
                for num in range(len(runner.labels)):
                    writer.writerow([amb, runner.labels[num], runner.preds[num], runner.preds_raw[num]])

        if runner._epoch % 5 == 0 and runner.log_conf_mat:
            # class_names = np.array([str(x) for x in range(10)])
            # print(runner.labels)
            num_class = runner.things_to_log['num_class']
            class_names = [str(i) for i in range(num_class)]
            fig_title = runner.mode.upper() + " Confusion matrix, epoch: " + str(runner._epoch)
            fig = self.plot_confusion_matrix(runner.labels, runner.preds, class_names, False, fig_title)

            figure_name = runner.work_dir +"/" + runner.mode + "_confusion_" + str(runner._epoch)+  ".png"
            # fig.savefig(figure_name)

            runner.log_buffer.logChart(fig, runner.mode + "_" + str(runner._epoch)+  ".png", "confusion_matrix")

            fig = self.plot_confusion_matrix(runner.labels, runner.preds, class_names, True, fig_title)

            figure_name = runner.work_dir +"/" + runner.mode + "_confusion_normed" + str(runner._epoch)+  ".png"
            fig.savefig(figure_name)

            runner.log_buffer.logChart(fig, runner.mode + "_" + str(runner._epoch)+  ".png", "confusion_matrix_normed")


            # regression plots
            fig_title = runner.mode.upper() + " Regression plot, epoch: " + str(runner._epoch)

            try:
                reg_fig = self.regressionPlot(runner.raw_labels, runner.preds_raw, class_names, fig_title, runner.non_pseudo_label)
            except Exception as e:
                print("THis is error", e)
                reg_fig = self.regressionPlot(runner.raw_labels, runner.preds_raw, class_names, fig_title)


            figure_name = runner.work_dir +"/" + runner.mode + "_regression_" + str(runner._epoch)+  ".png"
            # reg_fig.savefig(figure_name)
            runner.log_buffer.logChart(reg_fig, runner.mode + "_" + str(runner._epoch)+  ".png", "regression_plot")


        # print("predictions: ", runner.preds)
        runner.log_buffer.average()
        self.log(runner)
        if self.reset_flag:
            runner.log_buffer.clear_output()
            
    def regressionPlot(self, labels, raw_preds, classes, fig_title, non_pseudo_label=None):
        labels = np.asarray(labels)
        raw_preds = np.asarray(raw_preds)
        true_labels_jitter = labels + np.random.random_sample(labels.shape)/6
        
        fig = plt.figure()

        if non_pseudo_label is None:
            plt.plot(true_labels_jitter, raw_preds, 'bo', markersize=6)

        else:
            # Plot the pseudo labels in red 
            # print('non_pseudo_label', len(non_pseudo_label))
            # print('non_pseudo_label', non_pseudo_label)
            non_pseudo_label = np.asarray(non_pseudo_label)
            one_mask = list(np.argwhere(non_pseudo_label > 0).squeeze())
            zero_mask = list(np.argwhere(non_pseudo_label == 0).squeeze()) 
            # print('one_mask', one_mask)
            # print('zero_mask', len(zero_mask))
            # print("true_labels_jitter", len(true_labels_jitter))
            plt.plot(true_labels_jitter[one_mask], raw_preds[one_mask], 'bo', markersize=6)
            plt.plot(true_labels_jitter[zero_mask], raw_preds[zero_mask], 'ro', markersize=4)



        plt.title(fig_title)

        plt.xlim(-0.5, 4.5)
        plt.ylim(-0.5, 4.5)

        plt.xlabel("True Label")
        plt.ylabel("Regression Value")
        return fig


    def plot_confusion_matrix(self, y_true_raw, y_pred_raw, classes,normalize=False,title=None,cmap=plt.cm.Blues):
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Round the labels if they aren't already
        y_true = copy.deepcopy(y_true_raw)
        y_pred = copy.deepcopy(y_pred_raw)
        y_true = list(np.rint(np.asarray(y_true)))
        y_pred = list(np.rint(np.asarray(y_pred)))


        # How many classes are there? Go from 0 -> max in preds or labels
        max_label = int(max(max(y_true), max(y_pred)) + 1)
        class_names_int = [int(i) for i in range(max_label)]
        classes = [str(i) for i in range(max_label)]
        cm = confusion_matrix(y_true, y_pred, labels=class_names_int)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange( cm.shape[1]),
            yticks=np.arange( cm.shape[0]),
            # ... and label them with the respective list entries
            xticklabels=classes, yticklabels=classes,
            title=title,
            ylabel='True label',
            xlabel='Predicted label')
        
        ax.set_xlim(-0.5, cm.shape[1]-0.5)
        ax.set_ylim(cm.shape[0]-0.5, -0.5)

        # Rotate the tick labels and set their alignment.
        # print(ax.get_xticklabels())
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.3f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return fig


    # def plot_confusion_matrix(self, y_true, y_pred, classes,normalize=False,title=None,cmap=plt.cm.Blues):

    #     if not title:
    #         if normalize:
    #             title = 'Normalized confusion matrix'
    #         else:
    #             title = 'Confusion matrix, without normalization'

    #     cm = confusion_matrix(y_true, y_pred)
    #     if cm.shape[1] is not len(classes):
    #         # print("our CM is not the right size!!")

    #         all_labels = y_true + y_pred
    #         y_all_unique = list(set(all_labels))
    #         y_all_unique.sort()


    #         try:
    #             max_cm_size = len(classes)
    #             print('max_cm_size: ', max_cm_size)
    #             cm_new = np.zeros((max_cm_size, max_cm_size), dtype=np.int64)
    #             for i in range(len(y_all_unique)):
    #                 for j in range(len(y_all_unique)):
    #                     i_global = y_all_unique[i]
    #                     j_global = y_all_unique[j]
                        
    #                     cm_new[i_global, j_global] = cm[i,j]
    #         except:
    #             print('CM failed++++++++++++++++++++++++++++++++++++++')
    #             print('cm_new', cm_new)
    #             print('cm', cm)
    #             print('classes', classes)
    #             print('y_all_unique', y_all_unique)
    #             print('y_true', list(set(y_true)))
    #             print('y_pred', list(set(y_pred)))
    #             print('max_cm_size: ', max_cm_size)
    #             max_cm_size = max([len(classes), y_all_unique[-1]])

    #             cm_new = np.zeros((max_cm_size, max_cm_size), dtype=np.int64)
    #             for i in range(len(y_all_unique)):
    #                 for j in range(len(y_all_unique)):
    #                     i_global = y_all_unique[i]
    #                     j_global = y_all_unique[j]
    #                     try:
    #                         cm_new[i_global, j_global] = cm[i,j]
    #                     except:
    #                         print('CM failed second time++++++++++++++++++++++++++++++++++++++')
    #                         print('cm_new', cm_new)
    #                         print('cm', cm)
    #                         print('classes', classes)
    #                         print('y_all_unique', y_all_unique)
    #                         print('y_true', list(set(y_true)))
    #                         print('y_pred', list(set(y_pred)))
    #                         print('max_cm_size: ', max_cm_size)



    #         cm = cm_new

    #         classes = [i for i in range(max_cm_size)]

    #     # print(cm)
    #     # classes = classes[unique_labels(y_true, y_pred).astype(int)]
    #     if normalize:
    #         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    #         # print("Normalized confusion matrix")
    #     # else:
    #         # print('Confusion matrix, without normalization')
    # # 
    #     #print(cm)

    #     fig, ax = plt.subplots(figsize=(8, 6))
    #     im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    #     ax.figure.colorbar(im, ax=ax)
    #     # We want to show all ticks...
    #     ax.set(xticks=np.arange( cm.shape[1]),
    #         yticks=np.arange( cm.shape[0]),
    #         # ... and label them with the respective list entries
    #         xticklabels=classes, yticklabels=classes,
    #         title=title,
    #         ylabel='True label',
    #         xlabel='Predicted label')
        
    #     ax.set_xlim(-0.5, cm.shape[1]-0.5)
    #     ax.set_ylim(cm.shape[0]-0.5, -0.5)

    #     # Rotate the tick labels and set their alignment.
    #     # print(ax.get_xticklabels())
    #     plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #             rotation_mode="anchor")

    #     # Loop over data dimensions and create text annotations.
    #     fmt = '.3f' if normalize else 'd'
    #     thresh = cm.max() / 2.
    #     for i in range(cm.shape[0]):
    #         for j in range(cm.shape[1]):
    #             ax.text(j, i, format(cm[i, j], fmt),
    #                     ha="center", va="center",
    #                     color="white" if cm[i, j] > thresh else "black")
    #     fig.tight_layout()
    #     return fig
            