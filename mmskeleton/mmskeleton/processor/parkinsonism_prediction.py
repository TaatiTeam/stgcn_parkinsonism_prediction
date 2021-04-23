from collections import OrderedDict
import torch
import logging
import numpy as np
from mmskeleton.utils import call_obj, import_obj, load_checkpoint
from mmcv.runner import Runner, TooManyRetriesException
from mmcv import Config, ProgressBar
from mmcv.parallel import MMDataParallel
import os, re, copy
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix
import wandb
import matplotlib.pyplot as plt

import pandas as pd
import pickle
import shutil
from mmskeleton.processor.utils_recognition import *
from mmskeleton.processor.supcon_loss import *
import time
import scipy


# When developing/testing, we can save time by only loading in a subset of the data
fast_dev = True                    # Should be False to evaluate on entire dataset
num_walks_in_fast = 50


# Global variables
num_class = 4                       # This is overwritten using the info in the YAML config file
balance_classes = False
class_weights_dict = {}
flip_loss_mult = False
turn_off_weight_decay = False       # Keep as False to use the configuration from the YAML file
log_incrementally = True
log_code = False

def eval(
        work_dir,
        model_cfg,
        loss_cfg,
        dataset_cfg,
        optimizer_cfg,
        batch_size,
        total_epochs,
        training_hooks,
        workflow=[('train', 1)],
        gpus=1,
        log_level=0,
        workers=4,
        resume_from=None,
        load_from=None,
        cv=5,
        exclude_cv=False,
        notes=None,
        model_save_root='None',
        flip_loss=0,
        weight_classes=False,
        group_notes='',
        launch_from_local=True,
        wandb_project="mmskel",
        early_stopping=False,
        force_run_all_epochs=True,
        es_patience_1=5,
        es_start_up_1=5,
        es_patience_2=10,
        es_start_up_2=50,
        train_extrema_for_epochs=0,
        head='stgcn',
        freeze_encoder=True,
        do_position_pretrain=True,
        resource_root='.',
):
    # Set seed for reproductibility
    set_seed(0)
    print("==================================")
    print('have cuda: ', torch.cuda.is_available())
    print('using device: ', torch.cuda.get_device_name())
    

    # Update globals
    global flip_loss_mult
    flip_loss_mult = flip_loss

    global balance_classes
    balance_classes = weight_classes

    global num_class
    num_class = model_cfg['num_class']
    
    if turn_off_weight_decay:
        for stage in range(len(optimizer_cfg)):
            optimizer_cfg[stage]['weight_decay'] = 0


    # Format data loaders
    if isinstance(dataset_cfg, dict):
        dataset_cfg = [dataset_cfg]

    assert len(dataset_cfg) == 1, 'Found more than one dataset_cfg. In eval mode, only provide one dataset_cfg.'


    # Set up the name for WANDB logging and local work dir
    outcome_label = dataset_cfg[0]['data_source']['outcome_label']
    eval_pipeline = setup_eval_pipeline(dataset_cfg[0]['pipeline'])

    if log_code:
        os.environ['WANDB_DISABLE_CODE'] = 'false'
    else:
        os.environ['WANDB_DISABLE_CODE'] = 'true'


    # Add the wandb group to work_dir to prevent conflicts if running multiple repetitions of the same configuration
    model_type = get_model_type(model_cfg)
    group_notes = model_type + '_pretrain15' + "_dropout" + str(model_cfg['dropout']) + '_tempkernel' + str(model_cfg['temporal_kernel_size']) + "_batch" + str(batch_size)
    wandb_local_id = wandb.util.generate_id()
    wandb_group = wandb_local_id + "_" + outcome_label + "_" + group_notes
    work_dir = os.path.join(work_dir, wandb_group)

    # Check if we should use gait features
    if 'use_gait_feats' in dataset_cfg[0]['data_source']:
        model_cfg['use_gait_features'] = dataset_cfg[0]['data_source']['use_gait_feats']

    for ds in dataset_cfg:
        ds['data_source']['layout'] = model_cfg['graph_cfg']['layout']

    # Set the paths for input and output
    work_dir = os.path.join(resource_root, work_dir)
    wandb_log_local_group = os.path.join(resource_root, 'wandb', wandb_local_id)
    model_zoo_root = os.path.join(resource_root, 'model_zoo')
    dataloader_temp = os.path.join(resource_root, 'dataloaders')
    local_data_base = os.path.join(resource_root, 'data')
    for i in range(len(dataset_cfg)):
        dataset_cfg[i]['data_source']['data_dir'] = os.path.join(local_data_base, dataset_cfg[i]['data_source']['data_dir'])


    os.makedirs(work_dir)
    os.environ["WANDB_RUN_GROUP"] = wandb_group

    # Load data from provided dataloaders
    all_files_test, _, _ = getAllInputFiles(dataset_cfg)


    try:
        plt.close('all')
        test_walks = all_files_test


        for fold in range(1, cv + 1):
            path_to_pretrained_model = os.path.join(model_zoo_root, model_save_root, model_type, \
                                        str(model_cfg['temporal_kernel_size']), str(model_cfg['dropout']), str(fold))


            load_all = flip_loss != 0
            load_all = False  # Hold over, this is just for location of dataloaders TODO: remove this and reformat save locations
            path_to_saved_dataloaders = os.path.join(dataloader_temp, outcome_label, model_save_root, str(fold), \
                                                    "load_all" + str(load_all), "gait_feats_" + str(model_cfg['use_gait_features']), str(fold))

            
            work_dir_amb = work_dir + "/" + str(fold)



            loss_cfg_stage_2 = copy.deepcopy(loss_cfg[0])
            optimizer_cfg_stage_2 = optimizer_cfg[0]

            datasets = [copy.deepcopy(dataset_cfg[0]) for i in range(len(workflow))]

            
            datasets[0]['data_source']['data_dir'] = all_files_test

            if fast_dev:
                datasets[0]['data_source']['data_dir'] = all_files_test[:num_walks_in_fast]

            things_to_log = {'do_position_pretrain': do_position_pretrain, 'train_extrema_for_epochs': train_extrema_for_epochs, \
                'supcon_head': head, 'freeze_encoder': freeze_encoder, 'es_start_up_2': es_start_up_2, 'es_patience_2': es_patience_2, \
                'force_run_all_epochs': force_run_all_epochs, 'early_stopping': early_stopping, 'weight_classes': weight_classes, \
                'keypoint_layout': model_cfg['graph_cfg']['layout'], 'outcome_label': outcome_label, 'num_class': num_class, 'wandb_project': wandb_project, \
                'wandb_group': wandb_group, 'test_AMBID_num': len(test_walks), 'model_cfg': model_cfg, 'loss_cfg': loss_cfg_stage_2, \
                'optimizer_cfg': optimizer_cfg_stage_2, 'dataset_cfg_data_source': dataset_cfg[0]['data_source'], 'notes': notes, \
                'batch_size': batch_size, 'total_epochs': total_epochs , 'num_reps_pd': fold}

            model_cfg_local = copy.deepcopy(model_cfg)
            pretrained_model = initModel(model_cfg_local)
            continue
            # Evaluate the model on the finetuning task
            evaluate_model(work_dir_amb,
                        pretrained_model,
                        loss_cfg_stage_2,
                        datasets,
                        optimizer_cfg_stage_2,
                        batch_size,
                        total_epochs,
                        training_hooks,
                        workflow,
                        gpus,
                        log_level,
                        workers,
                        resume_from,
                        load_from,
                        things_to_log,
                        early_stopping,
                        force_run_all_epochs,
                        es_patience_2,
                        es_start_up_2, 
                        freeze_encoder, 
                        num_class,
                        train_extrema_for_epochs, 
                        path_to_saved_dataloaders, 
                        path_to_pretrained_model)

    except Exception as e:
        logging.exception(e)
        print(e)

    # Calculate summary metrics
    computeAllSummaryStats(work_dir, wandb_group, wandb_project, total_epochs, num_class, workflow, cv)
    
    # Delete the work_dir
    try:
        shutil.rmtree(work_dir)
    except:
        logging.exception('This: ')
        print('failed to delete the work_dir folder: ', work_dir)


def evaluate_model(
        work_dir,
        model,
        loss_cfg,
        datasets,
        optimizer_cfg,
        batch_size,
        total_epochs,
        training_hooks,
        workflow=[('train', 1)],
        gpus=1,
        log_level=0,
        workers=4,
        resume_from=None,
        load_from=None,
        things_to_log=None,
        early_stopping=False,
        force_run_all_epochs=True,
        es_patience=10,
        es_start_up=50,
        freeze_encoder=False, 
        num_class=4,
        train_extrema_for_epochs=0,
        path_to_saved_dataloaders=None, 
        path_to_pretrained_model=None, 
):
    print("======================================EVALUATING MODEL")

    # Load the model from the saved checkpoint if it exists
    if path_to_pretrained_model is not None:
        checkpoint_file = os.path.join(path_to_pretrained_model, 'checkpoint_pretrain.pt')
        if os.path.isfile(checkpoint_file):
            model.load_state_dict(torch.load(checkpoint_file))
            model = MMDataParallel(model, device_ids=range(gpus)).cuda()

            print('have pretrained model!')
        else:
            raise ValueError('The path to pretrained models does not exists')
    

    set_seed(0)

    data_loaders = [torch.utils.data.DataLoader(dataset=call_obj(**datasets[0]),
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=workers,
                                    drop_last=False)]


    data_loaders[0].dataset.data_source.sample_extremes = True
        

    workflow = [tuple(w) for w in workflow]
    global balance_classes
    global class_weights_dict
    for i in range(len(data_loaders)):
        class_weights_dict[workflow[i][0]] = data_loaders[i].dataset.data_source.get_class_dist()

    # Make sure model is set up for finetuning
    model.module.set_stage_2()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    set_seed(0)

    loss_cfg_local = copy.deepcopy(loss_cfg)
    training_hooks_local = copy.deepcopy(training_hooks)
    optimizer_cfg_local = copy.deepcopy(optimizer_cfg)

    try:
        loss = call_obj(**loss_cfg_local)
    except:
        print(loss)
        raise ValueError("Invalid loss" )


    # Set up the MMCV runner object
    optimizer = call_obj(params=model.parameters(), **optimizer_cfg_local)
    runner = Runner(model, batch_processor, optimizer, work_dir, log_level, num_class=num_class, \
                    things_to_log=things_to_log, early_stopping=early_stopping, force_run_all_epochs=force_run_all_epochs, \
                    es_patience=es_patience, es_start_up=es_start_up, freeze_encoder=freeze_encoder, finetuning=True)
    runner.register_training_hooks(**training_hooks_local)

    # Evaluate the model
    runner.early_stop_eval(workflow, data_loaders, loss=loss, flip_loss_mult=flip_loss_mult, balance_classes=balance_classes, class_weights_dict=class_weights_dict)
    
    try:
        # Wait a bit so the WANDB thread can sync
        time.sleep(15)
        shutil.rmtree(wandb.run.dir)
    except:
        print('failed to delete the wandb folder')    
    




def finetune_model(
        work_dir,
        model,
        loss_cfg,
        datasets,
        optimizer_cfg,
        batch_size,
        total_epochs,
        training_hooks,
        workflow=[('train', 1)],
        gpus=1,
        log_level=0,
        workers=4,
        resume_from=None,
        load_from=None,
        things_to_log=None,
        early_stopping=False,
        force_run_all_epochs=True,
        es_patience=10,
        es_start_up=50,
        freeze_encoder=False, 
        num_class=4,
        train_extrema_for_epochs=0,
        path_to_saved_dataloaders=None, 
        path_to_pretrained_model=None, 
        eval_only=False,
):
    print("=============================================================Starting STAGE 2: Fine-tuning...")

    # Load the model from the saved checkpoint if it exists
    if path_to_pretrained_model is not None:
        checkpoint_file = os.path.join(path_to_pretrained_model, 'checkpoint_pretrain.pt')
        if os.path.isfile(checkpoint_file):
            model.load_state_dict(torch.load(checkpoint_file))
            print('have pretrained model!')
        elif eval_only:
            raise ValueError('The path to pretrained models does not exists')
    

    load_data = True
    base_dl_path = os.path.join(path_to_saved_dataloaders, 'finetuning') 
    full_dl_path = os.path.join(base_dl_path, 'dataloaders_fine.pt')
    print("expecting dataloaders here:", full_dl_path)
    os.makedirs(base_dl_path, exist_ok=True) 

    if os.path.isfile(full_dl_path) and not eval_only:
        try:
            data_loaders = torch.load(full_dl_path)
            load_data = False
        except:
            print(f'failed to load dataloaders from file: {full_dl_path}, loading from individual files')

    if load_data:
        set_seed(0)

        train_dataloader = torch.utils.data.DataLoader(dataset=call_obj(**datasets[0]),
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=workers,
                                        drop_last=False)
    
        input(train_dataloader)
        # Normalize by the train scaler
        for d in datasets[1:]:
            d['data_source']['fit_scaler'] = train_dataloader.dataset.get_fit_scaler()
            d['data_source']['scaler'] = train_dataloader.dataset.get_scaler()

        data_loaders = [
            torch.utils.data.DataLoader(dataset=call_obj(**d),
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=workers,
                                        drop_last=False) for d in datasets[1:]
        ]

        data_loaders.insert(0, train_dataloader) # insert the train dataloader
        data_loaders[0].dataset.data_source.sample_extremes = True

        if not eval_only:
            # Save for next time
            torch.save(data_loaders, full_dl_path)
        

    workflow = [tuple(w) for w in workflow]
    global balance_classes
    global class_weights_dict
    for i in range(len(data_loaders)):
        class_weights_dict[workflow[i][0]] = data_loaders[i].dataset.data_source.get_class_dist()

    # Set up model for finetuning
    set_seed(0)
    model.module.set_classification_head_size(data_loaders[-1].dataset.data_source.get_num_gait_feats())
    model.module.set_stage_2()
    # print(data_loaders[-1].dataset.data_source.get_num_gait_feats())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # input(model)
    set_seed(0)

    loss_cfg_local = copy.deepcopy(loss_cfg)
    training_hooks_local = copy.deepcopy(training_hooks)
    optimizer_cfg_local = copy.deepcopy(optimizer_cfg)

    try:
        loss = call_obj(**loss_cfg_local)
    except:
        print(loss)

    # print('training hooks: ', training_hooks_local)
    # build runner
    optimizer = call_obj(params=model.parameters(), **optimizer_cfg_local)
    runner = Runner(model, batch_processor, optimizer, work_dir, log_level, num_class=num_class, \
                    things_to_log=things_to_log, early_stopping=early_stopping, force_run_all_epochs=force_run_all_epochs, \
                    es_patience=es_patience, es_start_up=es_start_up, freeze_encoder=freeze_encoder, finetuning=True)
    runner.register_training_hooks(**training_hooks_local)

    # run
    final_model, num_epoches_early_stop_finetune = runner.run(data_loaders, workflow, total_epochs, train_extrema_for_epochs=train_extrema_for_epochs, loss=loss, flip_loss_mult=flip_loss_mult, balance_classes=balance_classes, class_weights_dict=class_weights_dict)
    
    try:
        # Wait half a minute so the WANDB thread can sync
        time.sleep(30)
        shutil.rmtree(wandb.run.dir)
    except:
        print('failed to delete the wandb folder')    
    
    if path_to_pretrained_model is not None:
        checkpoint_file = os.path.join(path_to_pretrained_model, 'checkpoint_final.pt')
        torch.save(final_model.module.state_dict(), checkpoint_file)


    return final_model, num_epoches_early_stop_finetune


def pretrain_model(
        work_dir,
        simple_work_dir_amb,
        model_cfg,
        loss_cfg,
        datasets,
        optimizer_cfg,
        batch_size,
        total_epochs,
        training_hooks,
        workflow=[('train', 1)],
        gpus=1,
        log_level=0,
        workers=4,
        resume_from=None,
        load_from=None,
        things_to_log=None,
        early_stopping=False,
        force_run_all_epochs=True,
        es_patience=10,
        es_start_up=50,
        do_position_pretrain=True, 
        path_to_pretrained_model=None, 
        path_to_saved_dataloaders=None):
    print("============= Starting STAGE 1: Pretraining...", "="*50)
    print(path_to_pretrained_model)
    set_seed(0)

    model_cfg_local = copy.deepcopy(model_cfg)
    loss_cfg_local = copy.deepcopy(loss_cfg)
    training_hooks_local = copy.deepcopy(training_hooks)
    optimizer_cfg_local = copy.deepcopy(optimizer_cfg)



    # init model to return
    model = initModel(model_cfg_local)

    # print("the model is: ", model)

    # print("These are the model parameters:")
    # for param in model.parameters():
    #     print(param.data)


    print("path_to_pretrained_model===============================================================", path_to_pretrained_model)
    if not do_position_pretrain:
        print("SKIPPING PRETRAINING-------------------")
        model = MMDataParallel(model, device_ids=range(gpus)).cuda()
        return model

    if path_to_pretrained_model is not None:
        checkpoint_file = os.path.join(path_to_pretrained_model, 'checkpoint_pretrain.pt')
        if os.path.isfile(checkpoint_file):
            print(checkpoint_file)

            # Only copy over the ST-GCN layer from this model
            model_state = model.state_dict()

            pretrained_state = torch.load(checkpoint_file)
            pretrained_state = { k:v for k,v in pretrained_state.items() if k in model_state and v.size() == model_state[k].size() }  


            model_state.update(pretrained_state)
            model.load_state_dict(model_state)
            # input(model.use_gait_features)

            model = MMDataParallel(model, device_ids=range(gpus)).cuda()

            return model

    # Step 1: Initialize the model with random weights, 
    set_seed(0)
    model.apply(weights_init_xavier)
    model = MMDataParallel(model, device_ids=range(gpus)).cuda()


    load_data = True
    base_dl_path = os.path.join(path_to_saved_dataloaders, 'finetuning') 
    full_dl_path = os.path.join(base_dl_path, 'dataloaders_pre.pt')
    os.makedirs(base_dl_path, exist_ok=True)     
    if os.path.isfile(full_dl_path):
        try:
            data_loaders = torch.load(full_dl_path)
            load_data = False
        except:
            print(f'failed to load dataloaders from file: {full_dl_path}, loading from individual files')

    if load_data:
        set_seed(0)
        data_loaders = [
            torch.utils.data.DataLoader(dataset=call_obj(**d),
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=workers,
                                        drop_last=False) for d in datasets
        ]


        # Save for next time
        torch.save(data_loaders, full_dl_path)

    # return model
    global balance_classes
    global class_weights_dict

    # if balance_classes:
    #     dataset_train = call_obj(**datasets[0])
    #     class_weights_dict = dataset_train.data_source.class_dist


    torch.cuda.set_device(0)
    loss = call_obj(**loss_cfg_local)
    loss = WingLoss()

    visualize_preds = {'visualize': False, 'epochs_to_visualize': ['first', 'last'], 'output_dir': os.path.join('.', simple_work_dir_amb)}

    # print('training hooks: ', training_hooks_local)
    # build runner
    # loss = SupConLoss()
    optimizer = call_obj(params=model.parameters(), **optimizer_cfg_local)
    runner = Runner(model, batch_processor_position_pretraining, optimizer, work_dir, log_level, \
                    things_to_log=things_to_log, early_stopping=early_stopping, force_run_all_epochs=force_run_all_epochs, \
                    es_patience=es_patience, es_start_up=es_start_up, visualize_preds=visualize_preds)
    runner.register_training_hooks(**training_hooks_local)

    # run
    workflow = [tuple(w) for w in workflow]
    # [('train', 5), ('val', 1)]
    pretrained_model, _ = runner.run(data_loaders, workflow, total_epochs, loss=loss, supcon_pretraining=True)
    
    try:
        shutil.rmtree(wandb.run.dir)
    except:
        print('failed to delete the wandb folder')

    # print(pretrained_model)
    # input('model')
    if path_to_pretrained_model is not None:
        torch.save(pretrained_model.module.state_dict(), checkpoint_file)

        # input('saved')
    return pretrained_model


# process a batch of data
def batch_processor_position_pretraining(model, datas, train_mode, loss, num_class, **kwargs):
    try:
        data, data_flipped, label, name, num_ts, index, non_pseudo_label = datas
    except:
        data, data_flipped, label, name, num_ts, index, non_pseudo_label, demo_data = datas

    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor  

    # Even if we have flipped data, we only want to use the original in this stage
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
    num_valid_samples = data.shape[0]

    # Predict the future joint positions using all data
    predicted_joint_positions = model(data_all, gait_features_all)

    if torch.sum(predicted_joint_positions) == 0:        
        raise ValueError("=============================== got all zero output...")


    # Calculate the supcon loss for this data
    try:
        batch_loss = loss(predicted_joint_positions, label)
    except Exception as e:
        logging.exception("loss calc message=================================================")

    preds = []
    raw_preds = []

    label_placeholders = [-1 for i in range(num_valid_samples)]

    # Case when we have a single output
    if type(label_placeholders) is not list:
        label_placeholders = [label_placeholders]

    log_vars = dict(loss_pretrain_position=batch_loss.item())
    output_labels = dict(true=label_placeholders, pred=preds, raw_preds=raw_preds, name=name, num_ts=num_ts)
    outputs = dict(predicted_joint_positions=predicted_joint_positions, loss=batch_loss, log_vars=log_vars, num_samples=num_valid_samples, demo_data=demo_data)

    return outputs, output_labels, batch_loss.item()
