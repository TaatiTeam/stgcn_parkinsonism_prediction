from collections import OrderedDict
import torch
import logging
from mmskeleton.utils import call_obj, import_obj, load_checkpoint
from mmcv.runner import Runner
from mmcv import Config, ProgressBar
from mmcv.parallel import MMDataParallel
import os, re, copy
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix, precision_recall_fscore_support
from .utils_summary import *

import pandas as pd
import numpy as np
import math
from torch import nn
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
    os.environ["PYTHONHASHSEED"] = str(seed)


# Clean names for WANDB logging
def get_model_type(model_cfg):
    model_type = ""

    if (
        model_cfg["type"] == "models.backbones.ST_GCN_18_ordinal_smaller_2_position_pretrain"
        or model_cfg["type"] == "models.backbones.ST_GCN.ST_GCN_18_ordinal_smaller_2_position_pretrain"
    ):
        model_type = "v2"
    elif model_cfg["type"] == "models.backbones.ST_GCN_18_ordinal_smaller_10_position_pretrain":
        model_type = "v10"
    elif model_cfg["type"] == "models.backbones.ST_GCN_18_ordinal_smaller_11_position_pretrain":
        model_type = "v11"
    elif model_cfg["type"] == "models.backbones.ST_GCN_18_ordinal_orig_position_pretrain":
        model_type = "v0"
    elif model_cfg["type"] == "models.backbones.ST_GCN_18_ordinal_orig_position_pretrain_dynamic_v1":
        model_type = "dynamic_v1"
    elif model_cfg["type"] == "models.backbones.cnn_custom_1_pretrain":
        model_type = "cnn_v1"
    elif model_cfg["type"] == "models.backbones.cnn_custom_2_pretrain":
        model_type = "cnn_v2"
    elif model_cfg["type"] == "models.backbones.cnn_custom_3_pretrain":
        model_type = "cnn_v3"
    elif model_cfg["type"] == "models.backbones.cnn_custom_4_pretrain":
        model_type = "cnn_v4"

    else:
        model_type = model_cfg["type"]

    return model_type


# Remove random cropping for validation and test sets
def setup_eval_pipeline(pipeline):
    eval_pipeline = []
    for item in pipeline:
        if item["type"] != "datasets.skeleton.scale_walk" and item["type"] != "datasets.skeleton.shear_walk":
            if item["type"] == "datasets.skeleton.random_crop":
                item_local = copy.deepcopy(item)
                item_local["type"] = "datasets.skeleton.crop_middle"
                eval_pipeline.append(item_local)
            else:
                eval_pipeline.append(item)

    return eval_pipeline


# Processing a batch of data for label prediction
# process a batch of data
def batch_processor(model, datas, train_mode, loss, num_class, **kwargs):
    try:
        flip_loss_mult = kwargs["flip_loss_mult"]
    except:
        flip_loss_mult = 0

    try:
        balance_classes = kwargs["balance_classes"]
    except:
        balance_classes = False

    try:
        class_weights_dict = kwargs["class_weights_dict"]
        class_weights_dict = class_weights_dict[kwargs["workflow_stage"]]
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

    gait_features = np.empty([1, 9])  # default value if we dont have any gait features to load in
    if isinstance(data, dict):
        demo_data = {}
        for k in data.keys():
            if k.startswith("demo_data"):
                demo_data[k] = data[k]

        gait_features = data["gait_feats"].type(dtype)
        data = data["data"].type(dtype)

    data_all = data.cuda()
    gait_features_all = gait_features.cuda()
    label = label.cuda()
    non_pseudo_label = non_pseudo_label.cuda()

    # Remove the -1 labels (missing clinical score labels)
    y_true_all = label.data.reshape(-1, 1).float()
    non_pseudo_label = non_pseudo_label.data.reshape(-1, 1)
    condition = y_true_all >= 0.0

    row_cond = condition.all(1)
    y_true = y_true_all[row_cond, :]

    non_pseudo_label = non_pseudo_label[row_cond, :]
    data = data_all.data[row_cond, :]
    gait_features = gait_features_all.data[row_cond, :]

    num_valid_samples = data.shape[0]
    if have_flips:
        model_2 = copy.deepcopy(model)
        data_all = data_all.data
        data_all_flipped = data_flipped.cuda()
        data_all_flipped = data_all_flipped.data
        output_all_flipped = model_2(data_all_flipped, gait_features_all)
        torch.clamp(output_all_flipped, min=-1, max=num_class + 1)

    # Get predictions from the model
    output_all = model(data_all, gait_features_all)

    if torch.sum(output_all) == 0:
        raise ValueError("=============================== got all zero output...")
    output = output_all[row_cond]
    loss_flip_tensor = torch.tensor([0.0], dtype=torch.float, requires_grad=True)

    if have_flips:
        loss_flip_tensor = mse_loss(output_all_flipped, output_all)
        if loss_flip_tensor.data > 10:
            pass

    if not flip_loss_mult:
        loss_flip_tensor = torch.tensor([0.0], dtype=torch.float, requires_grad=True)
        loss_flip_tensor = loss_flip_tensor.cuda()
    else:
        loss_flip_tensor = loss_flip_tensor * flip_loss_mult

    # Calculate the label loss
    non_pseudo_label = non_pseudo_label.reshape(1, -1).squeeze()
    y_true_orig_shape = y_true.reshape(1, -1).squeeze()
    y_true_all_orig_shape = y_true_all.reshape(1, -1).squeeze()
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
    y_pred_rounded = np.clip(y_pred_rounded, 0, num_class - 1)
    preds = y_pred_rounded.squeeze().tolist()

    output_list_all = output_all.detach().cpu().numpy()
    output_list_all = np.nan_to_num(output_list_all)
    output_list_all_rounded = np.clip(output_list_all.squeeze(), 0, num_class).tolist()
    output_list_all = output_list_all_rounded
    output_list_all_rounded = np.round(np.asarray(output_list_all_rounded), 0).tolist()

    non_pseudo_label = non_pseudo_label.data.tolist()
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
    if type(output_list_all_rounded) is not list:
        output_list_all_rounded = [output_list_all_rounded]

    raw_labels = copy.deepcopy(labels)

    try:
        # Dealing with NaN and converting to ints
        labels = [0 if x != x else x for x in labels]
        preds = [0 if x != x else x for x in preds]
        output_list = [0 if x != x else x for x in output_list]
        output_list_all = [0 if x != x else x for x in output_list_all]
        labels = [int(round(cl)) for cl in labels]
        preds = [int(round(cl)) for cl in preds]


    except Exception as e:
        raise RuntimeError("Error converting NaNs to ints")

    overall_loss = losses + loss_flip_tensor
    log_vars = dict(loss_label=losses.item(), loss_flip=loss_flip_tensor.item(), loss_all=overall_loss.item())

    try:
        log_vars["mae_raw"] = mean_absolute_error(labels, output)
        log_vars["mae_rounded"] = mean_absolute_error(labels, preds)

    except:
        log_vars["mae_raw"] = math.nan
        log_vars["mae_rounded"] = math.nan

    output_labels = dict(
        true=labels,
        raw_labels=y_true_all,
        non_pseudo_label=non_pseudo_label,
        pred=preds,
        raw_preds=output_list,
        raw_preds_all=output_list_all,
        round_preds_all=output_list_all_rounded,
        name=name,
        num_ts=num_ts,
    )
    outputs = dict(loss=overall_loss, log_vars=log_vars, num_samples=len(labels), demo_data=demo_data)
    return outputs, output_labels, overall_loss


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
        y = y[:, 0 : dim[1], :, :]
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
    if classname.find("Conv1d") != -1:
        torch.nn.init.xavier_uniform_(model.weight)
        if model.bias is not None:
            model.bias.data.fill_(0)
    elif classname.find("Conv2d") != -1:
        torch.nn.init.xavier_uniform_(model.weight)
        if model.bias is not None:
            model.bias.data.fill_(0)
    elif classname.find("Linear") != -1:
        torch.nn.init.xavier_uniform_(model.weight)
        if model.bias is not None:
            model.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)


def weights_init(model):
    set_seed(0)
    classname = model.__class__.__name__
    if classname.find("Conv1d") != -1:
        model.weight.data.normal_(0.0, 0.02)
        if model.bias is not None:
            model.bias.data.fill_(0)
    elif classname.find("Conv2d") != -1:
        model.weight.data.normal_(0.0, 0.02)
        if model.bias is not None:
            model.bias.data.fill_(0)
    elif classname.find("Conv3d") != -1:
        model.weight.data.normal_(0.0, 0.02)
        if model.bias is not None:
            model.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
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


# https://discuss.pytorch.org/t/how-to-implement-weighted-mean-square-error/2547
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

        weights_list = [numerator / weights[int(round((i.data.tolist()[0])))] for i in target]
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

    weights_list = [numerator / weights[int(i.data.tolist()[0])] for i in target]
    weight_tensor = torch.FloatTensor(weights_list)
    weight_tensor = weight_tensor.unsqueeze(1).cuda()

    loss = torch.mul(weight_tensor, error_per_sample).mean()
    return loss


# https://discuss.pytorch.org/t/how-to-implement-weighted-mean-square-error/2547
def log_weighted_mse_loss(input, target, weights):
    error_per_sample = (input - target) ** 2
    numerator = 0

    for key in weights:
        numerator += weights[key]

    inv_weights_list = [numerator / weights[int(i.data.tolist()[0])] for i in target]

    weight_tensor = torch.FloatTensor(inv_weights_list)
    weight_tensor = torch.log(weight_tensor)  # Take log of the tensor
    weight_tensor = weight_tensor.unsqueeze(1).cuda()

    loss = torch.mul(weight_tensor, error_per_sample).mean()
    return loss


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


def getAllInputFiles(dataset_cfg):
    # If we only have one dataset, use this to create the train/val or test sets (depending on calling function), otherwise,
    # if two datasets are provided, assume the second one is the independent test set (so one will not
    # be split off and created from the first dataset_cfg)

    data_dir_all_data = dataset_cfg[0]["data_source"]["data_dir"]
    first_dataset = [
        os.path.join(data_dir_all_data, f)
        for f in os.listdir(data_dir_all_data)
        if os.path.isfile(os.path.join(data_dir_all_data, f))
    ]
    have_second_dataset = False
    second_dataset = []

    if len(dataset_cfg) == 2:
        have_second_dataset = True
        data_dir_second = dataset_cfg[1]["data_source"]["data_dir"]
        second_dataset = [
            os.path.join(data_dir_second, f)
            for f in os.listdir(data_dir_second)
            if os.path.isfile(os.path.join(data_dir_second, f))
        ]

    first_dataset.sort()
    second_dataset.sort()

    return first_dataset, second_dataset, have_second_dataset


def initModel(model_cfg_local):
    if isinstance(model_cfg_local, list):
        model = [call_obj(**c) for c in model_cfg_local]
        model = torch.nn.Sequential(*model)

    else:
        model = call_obj(**model_cfg_local)

    return model
