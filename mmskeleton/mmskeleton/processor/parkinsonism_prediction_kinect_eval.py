import torch
import logging
from mmskeleton.utils import call_obj 
from mmcv.runner import Runner
from mmcv.parallel import MMDataParallel
import os, copy
import wandb
import matplotlib.pyplot as plt

import shutil
from mmskeleton.processor.utils_recognition import *
from mmskeleton.processor.supcon_loss import *
import time


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
os.environ['WANDB_MODE'] = 'offline'


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
        compute_stats=False
):
    # Set seed for reproductibility
    set_seed(0)
    print("==================================")
    print('have cuda: ', torch.cuda.is_available())
    print('using device: ', torch.cuda.get_device_name())
    
    # Update globals
    updateGlobals(flip_loss, weight_classes, model_cfg['num_class'])
    updateWeightDecay(turn_off_weight_decay, optimizer_cfg)


    # Format data loaders
    if isinstance(dataset_cfg, dict):
        dataset_cfg = [dataset_cfg]

    assert len(dataset_cfg) == 1, 'Found more than one dataset_cfg. In eval mode, only provide one dataset_cfg.'


    # Set up the name for WANDB logging and local work dir
    outcome_label = dataset_cfg[0]['data_source']['outcome_label']
    eval_pipeline = setup_eval_pipeline(dataset_cfg[0]['pipeline'])
    dataset_cfg[0]['pipeline'] = eval_pipeline
    
    
    if log_code:
        os.environ['WANDB_DISABLE_CODE'] = 'false'
    else:
        os.environ['WANDB_DISABLE_CODE'] = 'true'

    centre = "no_centring"
    if dataset_cfg[0]['pipeline'][0]['type'] == "datasets.skeleton.kinect3D_centre_hip":
        centre = "(" + ','.join(str(num) for num in dataset_cfg[0]['pipeline'][0]['centre']) + ")"
        
    # Add the wandb group to work_dir to prevent conflicts if running multiple repetitions of the same configuration
    model_type = get_model_type(model_cfg)
    group_notes = model_type + '_pretrain15' + "_dropout" + str(model_cfg['dropout']) + '_tempkernel' + str(model_cfg['temporal_kernel_size']) + "_batch" + str(batch_size)
    wandb_local_id = wandb.util.generate_id()
    wandb_group = wandb_local_id + "_" + outcome_label + "_" + group_notes

    # Check if we should use gait features
    if 'use_gait_feats' in dataset_cfg[0]['data_source']:
        model_cfg['use_gait_features'] = dataset_cfg[0]['data_source']['use_gait_feats']

    # Set the paths for input and output
    work_dir = os.path.join(resource_root, work_dir, wandb_group)
    wandb_log_local_group = os.path.join(resource_root, 'wandb', wandb_local_id)
    model_zoo_root = os.path.join(resource_root, 'model_zoo')
    dataloader_temp = os.path.join(resource_root, 'data_loaders')
    local_data_base = os.path.join(resource_root, 'data')

    for ds in dataset_cfg:
        ds['data_source']['layout'] = model_cfg['graph_cfg']['layout']
        ds['data_source']['data_dir'] = os.path.join(local_data_base, ds['data_source']['data_dir'])


    os.makedirs(work_dir)
    os.environ["WANDB_RUN_GROUP"] = wandb_group

    # Load data from provided dataloaders
    test_walks, _, _ = getAllInputFiles(dataset_cfg)
    print('num test walks: ', len(test_walks) )

    ambid = 'all'
    try:

        for fold in range(1, cv + 1):
            plt.close('all')
            path_to_pretrained_model = os.path.join(model_zoo_root, model_save_root, model_type, \
                                        str(model_cfg['temporal_kernel_size']), str(model_cfg['dropout']), str(fold))

            path_to_pretrained_model = os.path.join(model_zoo_root, model_save_root, model_type, \
                                            str(model_cfg['temporal_kernel_size']), str(model_cfg['dropout']), str(fold), str(ambid), centre, "gait_feats_" + str(model_cfg['use_gait_features']))


            load_all = not fast_dev
            path_to_saved_dataloaders = os.path.join(dataloader_temp, outcome_label, model_save_root, \
                                                        "load_all_" + str(load_all), "gait_feats_" + str(model_cfg['use_gait_features']), str(fold), str(ambid), centre)

            
            work_dir_amb = work_dir + "/" + str(fold)


            loss_cfg_stage_2 = copy.deepcopy(loss_cfg[0])
            optimizer_cfg_stage_2 = optimizer_cfg[0]

            datasets = [copy.deepcopy(dataset_cfg[0]) for i in range(len(workflow))]

            
            datasets[0]['data_source']['data_dir'] = test_walks

            if fast_dev:
                datasets[0]['data_source']['data_dir'] = test_walks[:num_walks_in_fast]

            things_to_log = {'do_position_pretrain': do_position_pretrain, 'train_extrema_for_epochs': train_extrema_for_epochs, \
                'supcon_head': head, 'freeze_encoder': freeze_encoder, 'es_start_up_2': es_start_up_2, 'es_patience_2': es_patience_2, \
                'force_run_all_epochs': force_run_all_epochs, 'early_stopping': early_stopping, 'weight_classes': weight_classes, \
                'keypoint_layout': model_cfg['graph_cfg']['layout'], 'outcome_label': outcome_label, 'num_class': num_class, 'wandb_project': wandb_project, \
                'wandb_group': wandb_group, 'test_AMBID_num': len(test_walks), 'model_cfg': model_cfg, 'loss_cfg': loss_cfg_stage_2, \
                'optimizer_cfg': optimizer_cfg_stage_2, 'dataset_cfg_data_source': dataset_cfg[0]['data_source'], 'notes': notes, \
                'batch_size': batch_size, 'total_epochs': total_epochs , 'num_reps_pd': fold}

            model_cfg_local = copy.deepcopy(model_cfg)
            pretrained_model = initModel(model_cfg_local)
            pretrained_model.set_stage_2()

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
    if compute_stats:
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
        checkpoint_file = os.path.join(path_to_pretrained_model, 'checkpoint_final.pt')
        print("looking for: ", checkpoint_file)
        if os.path.isfile(checkpoint_file):
            print('have pretrained model!')
            print(checkpoint_file)
            model.load_state_dict(torch.load(checkpoint_file))
            model = MMDataParallel(model, device_ids=range(gpus)).cuda()

            print('have pretrained model!')
        else:
            print("checkpoint file: ", checkpoint_file)
            raise ValueError('The path to pretrained models does not exists')
    

    set_seed(0)
    data_loaders = [torch.utils.data.DataLoader(dataset=call_obj(**datasets[0]),
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=workers,
                                    drop_last=False)]


    data_loaders[0].dataset.data_source.sample_extremes = False
        

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
    


def updateGlobals(flip_loss, weight_classes, num_class_local):
    # Update globals
    global flip_loss_mult
    flip_loss_mult = flip_loss

    global balance_classes
    balance_classes = weight_classes

    global num_class
    num_class = num_class_local

def updateWeightDecay(turn_off_weight_decay, optimizer_cfg):
    if turn_off_weight_decay:
        for stage in range(len(optimizer_cfg)):
            optimizer_cfg[stage]['weight_decay'] = 0