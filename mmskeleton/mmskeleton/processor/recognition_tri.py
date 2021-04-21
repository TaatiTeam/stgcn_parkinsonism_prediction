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
# from spacecutter.models import OrdinalLogisticModel
# import spacecutter
import pandas as pd
import pickle
import shutil
from mmskeleton.processor.utils_recognition import *
#os.environ['WANDB_MODE'] = 'dryrun'

num_class = 3
balance_classes = False
class_weights_dict = {}
flip_loss_mult = False

local_data_base = '/home/saboa/data'
cluster_data_base = '/home/asabo/projects/def-btaati/asabo'
local_output_base = '/home/saboa/data/mmskel_out'
local_long_term_base = '/home/saboa/data/mmskel_long_term'

def train(
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
        test_ids=None,
        cv=5,
        exclude_cv=False,
        notes=None,
        flip_loss=0,
        weight_classes=False,
        group_notes='',
        launch_from_local=False,
        wandb_project="mmskel",
        early_stopping=False,
        force_run_all_epochs=True,
        es_patience=10,
        es_start_up=50,
):

    global flip_loss_mult
    flip_loss_mult = flip_loss

    global balance_classes
    balance_classes = weight_classes

    outcome_label = dataset_cfg[0]['data_source']['outcome_label']
    global num_class
    num_class = model_cfg['num_class']
    wandb_group = wandb.util.generate_id() + "_" + outcome_label + "_" + group_notes
    print("ANDREA - TRI-recognition: ", wandb_group)

    id_mapping = {27:25, 33:31, 34:32, 37:35, 39:37,
                  46:44, 47:45, 48:46, 50:48, 52:50, 
                  55:53, 57:55, 59:57, 66:63}


    eval_pipeline = setup_eval_pipeline(dataset_cfg[1]['pipeline'])

    # Add the wandb group to work_dir to prevent conflicts if running multiple repetitions of the same configuration
    work_dir = os.path.join(work_dir, wandb_group)

    # prepare data loaders
    if isinstance(dataset_cfg, dict):
        dataset_cfg = [dataset_cfg]

    print("==================================")
    print('have cuda: ', torch.cuda.is_available())
    print('using device: ', torch.cuda.get_device_name())

    # Correctly set the full data path
    if launch_from_local:
        simple_work_dir = work_dir
        work_dir = os.path.join(local_data_base, work_dir)
        
        for i in range(len(dataset_cfg)):
            dataset_cfg[i]['data_source']['data_dir'] = os.path.join(local_data_base, dataset_cfg[i]['data_source']['data_dir'])
    else:
        for i in range(len(dataset_cfg)):
            dataset_cfg[i]['data_source']['data_dir'] = os.path.join(cluster_data_base, dataset_cfg[i]['data_source']['data_dir'])




    # print(dataset_cfg[0])
    # assert len(dataset_cfg) == 1
    data_dir = dataset_cfg[0]['data_source']['data_dir']
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    workflow_orig = copy.deepcopy(workflow)
    for test_id in test_ids:
        plt.close('all')
        ambid = id_mapping[test_id]

        # These are all of the walks (both labelled and not) of the test participant and cannot be included in training data
        test_subj_walks = [i for i in all_files if re.search('ID_'+str(test_id), i) ]
        non_test_subj_walks = list(set(all_files).symmetric_difference(set(test_subj_walks)))
    
        try:
            test_data_dir = dataset_cfg[1]['data_source']['data_dir']
        except: 
            test_data_dir = data_dir
    
        all_test_files = [os.path.join(test_data_dir, f) for f in os.listdir(test_data_dir)]
        test_walks = [i for i in all_test_files if re.search('ID_'+str(test_id), i) ]

        datasets = [copy.deepcopy(dataset_cfg[0]) for i in range(len(workflow))]
        # datasets = [copy.deepcopy(dataset_cfg[0]), copy.deepcopy(dataset_cfg[0])]
        work_dir_amb = work_dir + "/" + str(ambid)
        for ds in datasets:
            ds['data_source']['layout'] = model_cfg['graph_cfg']['layout']


        if len(test_subj_walks) == 0:
            continue
        
        # Split the non_test walks into train/val
        kf = KFold(n_splits=cv, shuffle=True, random_state=1)
        kf.get_n_splits(non_test_subj_walks)

        try:
            num_reps = 1
            for train_ids, val_ids in kf.split(non_test_subj_walks):
                if num_reps > 1:
                    break
                num_reps += 1
                train_walks = [non_test_subj_walks[i] for i in train_ids]
                val_walks = [non_test_subj_walks[i] for i in val_ids]

                plt.close('all')
                ambid = id_mapping[test_id]

                # test_subj_walks = [i for i in all_files if re.search('ID_'+str(test_id), i) ]
                # non_test_subj_walks = list(set(all_files).symmetric_difference(set(test_subj_walks)))
            
                if exclude_cv: 
                    workflow = [workflow_orig[0], workflow_orig[2]]
                    datasets = [copy.deepcopy(dataset_cfg[0]) for i in range(len(workflow))]
                    datasets[0]['data_source']['data_dir'] = non_test_subj_walks
                    datasets[1]['data_source']['data_dir'] = test_walks
                else:
                    datasets[0]['data_source']['data_dir'] = train_walks
                    datasets[1]['data_source']['data_dir'] = val_walks
                    datasets[2]['data_source']['data_dir'] = test_walks

                    print('size of train set: ', len(datasets[0]['data_source']['data_dir']))
                    print('size of val set: ', len(datasets[1]['data_source']['data_dir']))                
                    print('size of test set: ', len(test_walks))

                work_dir_amb = work_dir + "/" + str(ambid)
                for ds in datasets:
                    ds['data_source']['layout'] = model_cfg['graph_cfg']['layout']
                # x = dataset_cfg[0]['data_source']['outcome_label']
        
                # Don't shear or scale the test or val data
                datasets[1]['pipeline'] = eval_pipeline
                datasets[2]['pipeline'] = eval_pipeline


                print(workflow)
                # print(model_cfg['num_class'])
                things_to_log = {'es_start_up': es_start_up, 'es_patience': es_patience, 'force_run_all_epochs': force_run_all_epochs, 'early_stopping': early_stopping, 'weight_classes': weight_classes, 'keypoint_layout': model_cfg['graph_cfg']['layout'], 'outcome_label': outcome_label, 'num_class': num_class, 'wandb_project': wandb_project, 'wandb_group': wandb_group, 'test_AMBID': ambid, 'test_AMBID_num': len(test_walks), 'model_cfg': model_cfg, 'loss_cfg': loss_cfg, 'optimizer_cfg': optimizer_cfg, 'dataset_cfg_data_source': dataset_cfg[0]['data_source'], 'notes': notes, 'batch_size': batch_size, 'total_epochs': total_epochs }
                print('size of train set: ', len(datasets[0]['data_source']['data_dir']))
                print('size of test set: ', len(test_walks))

                train_model(
                        work_dir_amb,
                        model_cfg,
                        loss_cfg,
                        datasets,
                        optimizer_cfg,
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
                        es_patience,
                        es_start_up,
                        num_class
                        )
        except Exception as e: 
            print("caught error ==========================", e)
            logging.exception('this went wrong')
        # Done with this participant, we can delete the temp foldeer

        try:
            shutil.rmtree(work_dir_amb)
        except:
            print('failed to delete the participant folder')

    # Final stats
    final_stats(work_dir, wandb_group, wandb_project, total_epochs, num_class, workflow)



def train_model(
        work_dir,
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
        num_class=4,
):
    print("==================================")

    data_loaders = [
        torch.utils.data.DataLoader(dataset=call_obj(**d),
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=workers,
                                    drop_last=False) for d in datasets
    ]

    global balance_classes
    global class_weights_dict

    if balance_classes:
        print("loading data to get class weights")
        dataset_train =call_obj(**datasets[0])
        class_weights_dict = dataset_train.data_source.class_dist

    model_cfg_local = copy.deepcopy(model_cfg)
    loss_cfg_local = copy.deepcopy(loss_cfg)
    training_hooks_local = copy.deepcopy(training_hooks)
    optimizer_cfg_local = copy.deepcopy(optimizer_cfg)



    # put model on gpus
    if isinstance(model_cfg, list):
        model = [call_obj(**c) for c in model_cfg_local]
        model = torch.nn.Sequential(*model)

    else:
        model = call_obj(**model_cfg_local)


    if loss_cfg_local['type'] == 'spacecutter.losses.CumulativeLinkLoss':
        pass
        # model = OrdinalLogisticModel(model, model_cfg_local['num_class'])


    model.apply(weights_init)
    
    model = MMDataParallel(model, device_ids=range(gpus)).cuda()
    torch.cuda.set_device(0)
    loss = call_obj(**loss_cfg_local)

    # print('training hooks: ', training_hooks_local)
    # build runner
    optimizer = call_obj(params=model.parameters(), **optimizer_cfg_local)
    runner = Runner(model, batch_processor, optimizer, work_dir, log_level, num_class=num_class, things_to_log=things_to_log, early_stopping=early_stopping, force_run_all_epochs=force_run_all_epochs, es_patience=es_patience, es_start_up=es_start_up)
    runner.register_training_hooks(**training_hooks_local)

    if resume_from:
        runner.resume(resume_from)
    elif load_from:
        runner.load_checkpoint(load_from)

    # run
    workflow = [tuple(w) for w in workflow]
    # [('train', 5), ('val', 1)]
    runner.run(data_loaders, workflow, total_epochs, loss=loss)
    try:
        shutil.rmtree(wandb.run.dir)
    except:
        print('failed to delete the wandb folder')