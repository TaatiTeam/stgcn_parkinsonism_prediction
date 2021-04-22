from collections import OrderedDict
import torch
import logging
import numpy as np
from mmskeleton.utils import call_obj, import_obj, load_checkpoint
from mmcv.runner import Runner
from mmcv import Config, ProgressBar
from mmcv.parallel import MMDataParallel
import os, re, copy
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix, precision_recall_fscore_support
import wandb
import matplotlib.pyplot as plt

import pandas as pd
import pickle
import math
from torch import nn
import wandb
import shutil
import random
import time

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# Clean names for WANDB logging
def get_model_type(model_cfg):
    model_type = ''

    if model_cfg['type'] == 'models.backbones.ST_GCN_18_ordinal_smaller_2_position_pretrain':
        model_type = "v2"
    elif model_cfg['type'] == 'models.backbones.ST_GCN_18_ordinal_smaller_10_position_pretrain':
        model_type = "v10"
    elif model_cfg['type'] == 'models.backbones.ST_GCN_18_ordinal_smaller_11_position_pretrain':
        model_type = "v11"
    elif model_cfg['type'] == 'models.backbones.ST_GCN_18_ordinal_orig_position_pretrain':
        model_type = "v0"
    elif model_cfg['type'] == 'models.backbones.ST_GCN_18_ordinal_orig_position_pretrain_dynamic_v1':
        model_type = "dynamic_v1"
    elif model_cfg['type'] == 'models.backbones.cnn_custom_1_pretrain':
        model_type = "cnn_v1"
    elif model_cfg['type'] == 'models.backbones.cnn_custom_2_pretrain':
        model_type = "cnn_v2"
    elif model_cfg['type'] == 'models.backbones.cnn_custom_3_pretrain':
        model_type = "cnn_v3"
    elif model_cfg['type'] == 'models.backbones.cnn_custom_4_pretrain':
        model_type = "cnn_v4"

    else: 
        model_type = model_cfg['type']

    return model_type

# Remove random cropping for validation and test sets
def setup_eval_pipeline(pipeline):
    eval_pipeline = []
    for item in pipeline:
        if item['type'] != "datasets.skeleton.scale_walk" and item['type'] != "datasets.skeleton.shear_walk":
            if item['type'] == "datasets.skeleton.random_crop":
                item_local = copy.deepcopy(item)
                item_local['type'] = "datasets.skeleton.crop_middle"
                eval_pipeline.append(item_local)
            else:
                eval_pipeline.append(item)

    return eval_pipeline

# Processing a batch of data for label prediction
# process a batch of data
def batch_processor(model, datas, train_mode, loss, num_class, **kwargs):
    try:
        flip_loss_mult = kwargs['flip_loss_mult']
    except:
        flip_loss_mult = 0

    try:
        balance_classes = kwargs['balance_classes']
    except:
        balance_classes = False

    try:
        class_weights_dict = kwargs['class_weights_dict']
        class_weights_dict = class_weights_dict[kwargs['workflow_stage']]
    except:
        class_weights_dict = {}
    
    mse_loss = torch.nn.MSELoss()
    mae_loss = torch.nn.L1Loss()
    have_flips = 0

    try:
        data, label, name, num_ts, index, non_pseudo_label = datas
    except:
        data, data_flipped, label, name, num_ts, index, non_pseudo_label = datas
        have_flips = 1


    # If we have both data and gait features, we need to remove them from the datastuct and 
    # move them to he GPU separately
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor  

    gait_features = np.empty([1, 9])# default value if we dont have any gait features to load in
    if isinstance(data, dict):
        demo_data = {}
        for k in data.keys():
            if k.startswith('demo_data'):
                demo_data[k] = data[k]

        gait_features = data['gait_feats'].type(dtype)
        data = data['data'].type(dtype)

    data_all = data.cuda()
    gait_features_all = gait_features.cuda()
    label = label.cuda()

    # Remove the -1 labels
    y_true_all = label.data.reshape(-1, 1).float()
    non_pseudo_label = non_pseudo_label.data.reshape(-1, 1)
    condition = y_true_all >= 0.

    row_cond = condition.all(1)
    y_true = y_true_all[row_cond, :]

    non_pseudo_label = non_pseudo_label[row_cond, :]
    data = data_all.data[row_cond, :]
    gait_features = gait_features_all.data[row_cond, :]

    num_valid_samples = data.shape[0]
    # print("data shape is: ", data.shape)
    if have_flips:
        model_2 = copy.deepcopy(model)
        data_all = data_all.data
        data_all_flipped = data_flipped.cuda()
        data_all_flipped = data_all_flipped.data 
        output_all_flipped = model_2(data_all_flipped, gait_features_all)
        torch.clamp(output_all_flipped, min=-1, max=num_class+1)

    # Get predictions from the model
    output_all = model(data_all, gait_features_all)

    if torch.sum(output_all) == 0:        
        raise ValueError("=============================== got all zero output...")
    output = output_all[row_cond]
    loss_flip_tensor = torch.tensor([0.], dtype=torch.float, requires_grad=True) 

    if have_flips:
        loss_flip_tensor = mse_loss(output_all_flipped, output_all)
        if loss_flip_tensor.data > 10:
            pass

    if not flip_loss_mult:
        loss_flip_tensor = torch.tensor([0.], dtype=torch.float, requires_grad=True) 
        loss_flip_tensor = loss_flip_tensor.cuda()
    else:
        loss_flip_tensor = loss_flip_tensor * flip_loss_mult

    
    # Calculate the label loss
    non_pseudo_label = non_pseudo_label.reshape(1,-1).squeeze()
    y_true_orig_shape = y_true.reshape(1,-1).squeeze()
    y_true_all_orig_shape = y_true_all.reshape(1,-1).squeeze()
    losses = loss(output, y_true)


    if balance_classes:
        if type(loss) == type(mse_loss):
            losses = weighted_mse_loss(output, y_true, class_weights_dict)
        if type(loss) == type(mae_loss):
            losses = weighted_mae_loss(output, y_true, class_weights_dict)

    # Convert the output to classes and clip from 0 to number of classes
    y_pred_rounded = output.detach().cpu().numpy()
    y_pred_rounded = np.nan_to_num(y_pred_rounded)
    output = y_pred_rounded
    output_list = output.squeeze().tolist()
    output_list = np.clip(np.asarray(output_list), 0, num_class).tolist()
    y_pred_rounded = y_pred_rounded.reshape(1, -1).squeeze()
    y_pred_rounded = np.round(y_pred_rounded, 0)
    y_pred_rounded = np.clip(y_pred_rounded, 0, num_class-1)
    preds = y_pred_rounded.squeeze().tolist()

    output_list_all = output_all.detach().cpu().numpy()
    output_list_all = np.nan_to_num(output_list_all)
    output_list_all_rounded = np.clip(output_list_all.squeeze(), 0, num_class).tolist()
    output_list_all = output_list_all_rounded
    output_list_all_rounded = np.round(np.asarray(output_list_all_rounded), 0).tolist()

    non_pseudo_label  = non_pseudo_label.data.tolist()
    labels = y_true_orig_shape.data.tolist()
    y_true_all = y_true_all.data.squeeze().tolist()

    num_ts = num_ts.data.tolist()
    # Case when we have a single output
    if type(labels) is not list:
        labels = [labels]
    if type(preds) is not list:
        preds = [preds]
    if type(output_list) is not list:
        output_list = [output_list]    
    if type(num_ts) is not list:
        num_ts = [num_ts]
    if type(output_list_all) is not list:
        output_list_all = [output_list_all]

    if type(y_true_all) is not list:
        y_true_all = [y_true_all]

    if type(non_pseudo_label) is not list:
        non_pseudo_label = [non_pseudo_label]

    raw_labels = copy.deepcopy(labels)

    try:
        # Dealing with NaN and converting to ints
        labels = [0 if x != x else x for x in labels]
        preds = [0 if x != x else x for x in preds]
        labels = [int(round(cl)) for cl in labels]
        preds = [int(round(cl)) for cl in preds]
    except Exception as e:
        print(labels)
        print(preds)
        print("got an error: ", e)
        raise RuntimeError('')


    overall_loss = losses + loss_flip_tensor
    log_vars = dict(loss_label=losses.item(), loss_flip = loss_flip_tensor.item(), loss_all=overall_loss.item())

    try:
        log_vars['mae_raw'] = mean_absolute_error(labels, output)
        log_vars['mae_rounded'] = mean_absolute_error(labels, preds)

    except:
        log_vars['mae_raw'] = math.nan
        log_vars['mae_rounded'] = math.nan


    output_labels = dict(true=labels, raw_labels=y_true_all, non_pseudo_label=non_pseudo_label,\
         pred=preds, raw_preds=output_list, raw_preds_all=output_list_all, round_preds_all=output_list_all_rounded, name=name, num_ts=num_ts)
    outputs = dict(loss=overall_loss, log_vars=log_vars, num_samples=len(labels), demo_data=demo_data)
    return outputs, output_labels, overall_loss


def set_up_results_table(workflow, num_class):
    col_names = []
    for i, flow in enumerate(workflow):
        mode, _ = flow
        class_names_int = [int(i) for i in range(num_class)]

        # Calculate the mean metrics across classes
        average_types = ['macro', 'micro', 'weighted']
        average_metrics_to_log = ['precision', 'recall', 'f1score', 'support']
        prefix_name =  mode + '/'
        for av in average_types:
            for m in average_metrics_to_log:
                col_names.append(prefix_name + m +'_average_' + av)


        # Calculate metrics per class
        for c in range(len(average_metrics_to_log)):
            for s in range(len(class_names_int)):
                col_names.append(prefix_name + str(class_names_int[s]) + '_'+ average_metrics_to_log[c])

        col_names.append(prefix_name + 'mae_rounded')
        col_names.append(prefix_name + 'mae_raw')
        col_names.append(prefix_name + 'accuracy')


    df = pd.DataFrame(columns=col_names)
    return df

def final_stats_per_trial(final_results_local_path, wandb_group, wandb_project, num_class, workflow, num_epochs, results_table):
    try:
        # Compute summary statistics (accuracy and confusion matrices)
        print("getting final results from: ", final_results_local_path)
        log_vars = {'num_epochs': num_epochs}

        # final results +++++++++++++++++++++++++++++++++++++++++
        for i, flow in enumerate(workflow):
            mode, _ = flow

            class_names = [str(i) for i in range(num_class)]
            class_names_int = [int(i) for i in range(num_class)]
            results_file = os.path.join(final_results_local_path, mode+".csv")

            print("loading from: ", results_file)
            df = pd.read_csv(results_file)
            true_labels = df['true_score']
            preds = df['pred_round']
            preds_raw = df['pred_raw']

            # Calculate the mean metrics across classes
            average_types = ['macro', 'micro', 'weighted']
            average_metrics_to_log = ['precision', 'recall', 'f1score', 'support']
            prefix_name = mode + '/'
            for av in average_types:
                results_tuple = precision_recall_fscore_support(true_labels, preds, average=av)
                for m in range(len(average_metrics_to_log)):      
                    log_vars[prefix_name +  average_metrics_to_log[m] +'_average_' + av] = results_tuple[m]


            # Calculate metrics per class
            results_tuple = precision_recall_fscore_support(true_labels, preds, average=None, labels=class_names_int)

            for c in range(len(average_metrics_to_log)):
                cur_metrics = results_tuple[c]
                print(cur_metrics)
                for s in range(len(class_names_int)):
                    log_vars[prefix_name + str(class_names_int[s]) + '_'+ average_metrics_to_log[c]] = cur_metrics[s]


            # Keep the original metrics for backwards compatibility
            log_vars[prefix_name + 'mae_rounded'] = mean_absolute_error(true_labels, preds)
            log_vars[prefix_name + 'mae_raw'] = mean_absolute_error(true_labels, preds_raw)
            log_vars[prefix_name + 'accuracy'] = accuracy_score(true_labels, preds)

        # print(log_vars)
        # print(results_table.columns)
        df = pd.DataFrame(log_vars, index=[0])
        results_table = results_table.append(df)

        return results_table
    except:
        logging.exception("in batch stats after trials: \n")


def final_stats_variance(results_df, wandb_group, wandb_project, total_epochs, num_class, workflow):
    wandb.init(name="ALL_var", project=wandb_project, group=wandb_group, tags=['summary'], reinit=True)
    stdev = results_df.std().to_dict()
    means = results_df.mean().to_dict()
    all_stats = dict()
    for k,v in stdev.items():
        all_stats[k + "_stdev"] = stdev[k]
        all_stats[k + "_mean"] = means[k]


    wandb.log(all_stats)


def final_stats_worker(work_dir, folder_count, wandb_group, wandb_project, total_epochs, num_class, workflow, log_name, wandb_log_local_group, results_table=None):
    max_label = num_class
    # Compute summary statistics (accuracy and confusion matrices)
    final_results_dir = os.path.join(work_dir, 'all_final_eval', str(folder_count))
    final_results_dir2 = os.path.join(work_dir, 'all_test', wandb_group) # we just delete the contents of this

    if wandb_log_local_group is not None:
        wandb.init(dir=wandb_log_local_group, name=log_name, project=wandb_project, group=wandb_group, config = {'wandb_group':wandb_group}, tags=['summary'], reinit=True)
    else:
        wandb.init(name=log_name, project=wandb_project, group=wandb_group, config = {'wandb_group':wandb_group}, tags=['summary'], reinit=True)

    print("getting final results from: ", final_results_dir)
    print("total_epochs: ", total_epochs)

    # final results +++++++++++++++++++++++++++++++++++++++++

    dict_for_table = {}
    for i, flow in enumerate(workflow):
        mode, _ = flow

        class_names = [str(i) for i in range(num_class)]
        class_names_int = [int(i) for i in range(num_class)]

        log_vars = {}
        results_file = os.path.join(final_results_dir, mode+".csv")
        print("loading from: ", results_file)
        df = pd.read_csv(results_file)
        true_labels = df['true_score']
        preds = df['pred_round']
        preds_raw = df['pred_raw']

        # Last check to make sure that the preds are in the correct range
        preds[preds > (num_class - 1)] = num_class - 1

        # Calculate the mean metrics across classes
        average_types = ['macro', 'micro', 'weighted']
        average_metrics_to_log = ['precision', 'recall', 'f1score', 'support']
        average_dict = {}
        prefix_name = 'final/'+ mode + '/'
        for av in average_types:
            results_tuple = precision_recall_fscore_support(true_labels, preds, average=av)
            for m in range(len(average_metrics_to_log)):      
                average_dict[prefix_name + '_'+ average_metrics_to_log[m] +'_average_' + av] = results_tuple[m]

        wandb.log(average_dict)

        # Calculate metrics per class
        results_tuple = precision_recall_fscore_support(true_labels, preds, average=None, labels=class_names_int)

        per_class_stats = {}
        for c in range(len(average_metrics_to_log)):
            cur_metrics = results_tuple[c]
            print(cur_metrics)
            for s in range(len(class_names_int)):
                per_class_stats[prefix_name + str(class_names_int[s]) + '_'+ average_metrics_to_log[c]] = cur_metrics[s]

        wandb.log(per_class_stats)


        # Keep the original metrics for backwards compatibility
        log_vars['early_stop_eval/'+mode+ '/mae_rounded'] = mean_absolute_error(true_labels, preds)
        log_vars['early_stop_eval/'+mode+ '/mae_raw'] = mean_absolute_error(true_labels, preds_raw)
        log_vars['early_stop_eval/'+mode+ '/accuracy'] = accuracy_score(true_labels, preds)
        wandb.log(log_vars)

        fig = plot_confusion_matrix( true_labels,preds, class_names, max_label)
        wandb.log({"confusion_mat_earlystop/" + mode + "_final_confusion_matrix.png": fig})

        try:
            fig_normed = plot_confusion_matrix( true_labels,preds, class_names, max_label, True)
            wandb.log({"confusion_mat_earlystop/" + mode + "_final_normed_confusion_matrix.png": fig_normed})
        except:
            pass
        
        fig_title = "Regression for ALL unseen participants"
        reg_fig = regressionPlot(true_labels, preds_raw, class_names, fig_title)
        try:
            wandb.log({"regression_plot_earlystop/" + mode + "_final_regression_plot.png": [wandb.Image(reg_fig)]})
        except:
            try:
                wandb.log({"regression_plot_earlystop/" + mode + "_final_regression_plot.png": reg_fig})
            except:
                print("failed to log regression plot")

        # Log the final dataframe to wandb for future analysis
        header = ['amb', 'walk_name', 'num_ts', 'true_score', 'pred_round', 'pred_raw']
        try:
            wandb.log({"final_results_csv/"+mode: wandb.Table(data=df.values.tolist(), columns=header)})
        except: 
            logging.exception("Could not save final table =================================================\n")


        dict_for_table.update(log_vars)
        dict_for_table.update(per_class_stats)
        dict_for_table.update(average_dict)
    if results_table is not None:
        df = pd.DataFrame(dict_for_table, index=[0])
        results_table = results_table.append(df)

    try:
        # Remove the files generated so we don't take up this space
        print('removing ', final_results_dir)
        shutil.rmtree(final_results_dir)
        print('removing ', final_results_dir2)
        shutil.rmtree(final_results_dir2)
    except:
        pass
    return results_table


def final_stats(work_dir, wandb_group, wandb_project, total_epochs, num_class, workflow, num_self_train_iter=0, wandb_log_local_group=None):
    ## DEPRECATED
    # input("the group is: " + wandb_group)
    work_dir_back = work_dir
    try:
        if num_self_train_iter == 0:
            work_dir = work_dir_back 
            final_stats_worker(work_dir, num_self_train_iter, wandb_group, wandb_project, total_epochs, num_class, workflow, log_name="ALL", wandb_log_local_group=wandb_log_local_group)
        else:
            for iter_count in range(num_self_train_iter):
                work_dir = work_dir_back + "/" + str(iter_count)
                final_stats_worker(work_dir, num_self_train_iter, wandb_group, wandb_project, total_epochs, num_class, workflow, log_name="ALL_" + str(iter_count), wandb_log_local_group=wandb_log_local_group)
                

    except:
        print('something when wrong in the summary stats')
        logging.exception("Error message =================================================")    


def final_stats_numbered(work_dir, wandb_group, wandb_project, total_epochs, num_class, workflow, num_self_train_iter=1, wandb_log_local_group=None):
    # input("the group is: " + wandb_group)
    results_df = set_up_results_table(workflow, num_class)

    try:
        for iter_count in range(num_self_train_iter):
            print(str(iter_count + 1) + "/" + str(num_self_train_iter))
            folder_count = iter_count + 1
            results_df = final_stats_worker(work_dir, folder_count, wandb_group, wandb_project, total_epochs, num_class, workflow, log_name="ALL_" + str(folder_count), wandb_log_local_group=wandb_log_local_group, results_table = results_df)

        # if we ran multiple trials, compute the summary stats
        if num_self_train_iter is not 1:
            final_stats_variance(results_df, wandb_group, wandb_project, total_epochs, num_class, workflow)

    except:
        print('something when wrong in the summary stats')
        logging.exception("Error message =================================================")    



# From: https://github.com/elliottzheng/AdaptiveWingLoss/blob/master/wing_loss.py
# torch.log  and math.log is e based
class WingLoss(nn.Module):
    def __init__(self, omega=10, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon

    def forward(self, pred, target):
        y = target
        y_hat = pred
        # print(y.shape, y_hat.shape)
        dim = y_hat.shape
        # Make sure we only use the coordinates that the model is predicting
        y = y[:, 0:dim[1], :, :]
        # print(y.shape, y_hat.shape)

        # input('stop')


        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.omega]
        delta_y2 = delta_y[delta_y >= self.omega]
        loss1 = self.omega * torch.log(1 + delta_y1 / self.epsilon)
        C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
        loss2 = delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))



def weights_init_xavier(model):
    set_seed(0)
    classname = model.__class__.__name__
    if classname.find('Conv1d') != -1:
        torch.nn.init.xavier_uniform_(model.weight)
        if model.bias is not None:
            model.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        torch.nn.init.xavier_uniform_(model.weight)
        if model.bias is not None:
            model.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(model.weight)
        if model.bias is not None:
            model.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)


def weights_init(model):
    set_seed(0)
    classname = model.__class__.__name__
    if classname.find('Conv1d') != -1:
        model.weight.data.normal_(0.0, 0.02)
        if model.bias is not None:
            model.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        model.weight.data.normal_(0.0, 0.02)
        if model.bias is not None:
            model.bias.data.fill_(0)
    elif classname.find('Conv3d') != -1:
        model.weight.data.normal_(0.0, 0.02)
        if model.bias is not None:
            model.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)


def weights_init_cnn(m):
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.1)


    if isinstance(m, torch.nn.Conv3d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.1)

    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.1)


#https://discuss.pytorch.org/t/how-to-implement-weighted-mean-square-error/2547
def weighted_mse_loss(input, target, weights):
    # If the targets are not integers, then round them so that they match the 
    # dictionary keys

    # target_local = 

    error_per_sample = (input - target) ** 2
    numerator = 0
    weights.pop(-1, None)
    for key in weights:
        numerator += weights[key]
    try:
        # print("weights: ", weights)

        weights_list = [numerator / weights[int(round((i.data.tolist()[0])))]  for i in target]
    except Exception as e:
        print("target: ", target)
        print("weights: ", weights)
        print("target length: ", len(target))
        print("error here is: ", e)

    # print("target: ", target)
    # print("weights list: ", weights_list)
    # print("weights ", weights)

    # print("numerator: ", numerator)
    weight_tensor = torch.FloatTensor(weights_list)
    weight_tensor = weight_tensor.unsqueeze(1).cuda()

    loss = torch.mul(weight_tensor, error_per_sample).mean()
    return loss

def weighted_mae_loss(input, target, weights):
    error_per_sample = abs(input - target)
    numerator = 0
    
    for key in weights:
        numerator += weights[key]

    weights_list = [numerator / weights[int(i.data.tolist()[0])]  for i in target]
    weight_tensor = torch.FloatTensor(weights_list)
    weight_tensor = weight_tensor.unsqueeze(1).cuda()

    loss = torch.mul(weight_tensor, error_per_sample).mean()
    return loss

#https://discuss.pytorch.org/t/how-to-implement-weighted-mean-square-error/2547
def log_weighted_mse_loss(input, target, weights):
    error_per_sample = (input - target) ** 2
    numerator = 0
    
    for key in weights:
        numerator += weights[key]

    inv_weights_list = [numerator / weights[int(i.data.tolist()[0])]  for i in target]

    weight_tensor = torch.FloatTensor(inv_weights_list)
    weight_tensor = torch.log(weight_tensor) # Take log of the tensor
    weight_tensor = weight_tensor.unsqueeze(1).cuda()

    loss = torch.mul(weight_tensor, error_per_sample).mean()
    return loss



def topk_accuracy(score, label, k=1):
    rank = score.argsort()
    hit_top_k = [l in rank[i, -k:] for i, l in enumerate(label)]
    accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
    return accuracy


def plot_confusion_matrix( y_true, y_pred, classes, max_label, normalize=False,title=None,cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'


    class_names_int = [int(i) for i in range(max_label)]
    classes = [str(i) for i in range(max_label)]
    cm = confusion_matrix(y_true, y_pred, labels=class_names_int)

    if normalize:
        try:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        except:
            pass
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

def regressionPlotByGroup(labels, raw_preds, classes, fig_title, non_pseudo_label=None):
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
        one_mask = list(np.argwhere(non_pseudo_label == 1).squeeze())
        zero_mask = list(np.argwhere(non_pseudo_label == 0).squeeze()) 
        # print('one_mask', one_mask)
        # print('zero_mask', len(zero_mask))
        # print("true_labels_jitter", len(true_labels_jitter))
        plt.plot(true_labels_jitter[one_mask], raw_preds[one_mask], 'bo', markersize=6)
        plt.plot(true_labels_jitter[zero_mask], raw_preds[zero_mask], 'ro', markersize=4)
        print(len(raw_preds[zero_mask]))


    plt.title(fig_title)

    plt.xlim(-0.5, 4.5)
    plt.ylim(-0.5, 4.5)

    plt.xlabel("True Label")
    plt.ylabel("Regression Value")
    return fig


def regressionPlot(labels, raw_preds, classes, fig_title):
    labels = np.asarray(labels)
    true_labels_jitter = labels + np.random.random_sample(labels.shape)/6

    fig = plt.figure()
    plt.plot(true_labels_jitter, raw_preds, 'bo', markersize=6)
    plt.title(fig_title)

    plt.xlim(-0.5, 4.5)
    plt.ylim(-0.5, 4.5)

    plt.xlabel("True Label")
    plt.ylabel("Regression Value")
    return fig


def rmdir(directory):
    try:
        directory.rmdir()
    except:
        print("couldn't delete: ", directory)


def robust_rmtree(path, logger=None, max_retries=3):
    """Robustly tries to delete paths.
    Retries several times (with increasing delays) if an OSError
    occurs.  If the final attempt fails, the Exception is propagated
    to the caller.
    """
    print("removing robustly", path)
    dt = 1
    for i in range(max_retries):
        print("removing robustly: ", i)

        try:
            shutil.rmtree(path)
            return
        except Exception as e:
            rmdir(path)
            time.sleep(dt)
            dt *= 1.5

    # Final attempt, pass any Exceptions up to caller.
    shutil.rmtree(path)


def sync_wandb(wandb_local_id):
    # Sync everything to wandb at the end
    try:
        os.system('wandb sync ' + wandb_local_id)

        # Delete the work_dir if successful sync
        try:
            robust_rmtree(wandb_local_id)
            # shutil.rmtree(work_dir)
        except:
            logging.exception('This: ')
            print('failed to delete the wandb_log_local_group folder: ', wandb_local_id)

    except:
        print('failed to sync wandb')


