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
    all_files_test, _, have_second_dataset = getAllInputFiles(dataset_cfg)


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
                'batch_size': batch_size, 'total_epochs': total_epochs }

            model_cfg_local = copy.deepcopy(model_cfg)
            pretrained_model = initModel(model_cfg_local)
            
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

            input('pause')
    except Exception as e:
        logging.exception(e)
        print(e)

    # Calculate summary metrics
    final_stats_objective2(work_dir, wandb_group, wandb_project, total_epochs, num_class, workflow, cv)
    

    # # Delete the work_dir
    # try:
    #     shutil.rmtree(work_dir)
    # except:
    #     logging.exception('This: ')
    #     print('failed to delete the work_dir folder: ', work_dir)


def final_stats_objective2(work_dir, wandb_group, wandb_project, total_epochs, num_class, workflow, cv):
    # work_dir = "/home/saboa/data/OBJECTIVE_2_ML_DATA/data/./work_dir/recognition/tri_all/dataset_example/v2/UPDRS/120_v0_pred15_ankles_wrists_wing/1t4a3yqu_UPDRS_gait_v0_pretrain15_dropout0.0_tempkernel5_batch100"
    # work_dir = "/home/saboa/data/OBJECTIVE_2_ML_DATA/data/work_dir/recognition/tri_all/dataset_example/v2/UPDRS/120_v0_pred15_ankles_wrists_wing_do_01/176rgyac_UPDRS_gait_v10_pretrain15_dropout0.5_tempkernel5_batch100"
    print("work_dir", work_dir)
    print("wandb_group", wandb_group)
    print("wandb_project", wandb_project)
    print("total_epochs", total_epochs)
    print("num_class", num_class)
    print("workflow", workflow)
    print("cv", cv)

    raw_results_dict = {}

    log_name = "CV_ALL"
    wandb.init(name=log_name, project=wandb_project, group=wandb_group, config = {'wandb_group':wandb_group}, tags=['summary'], reinit=True)
    wandb.Table.MAX_ROWS =100000
    results_table = set_up_results_table(workflow, num_class)

    # Load in all the data from all folds
    for i, flow in enumerate(workflow):
        mode, _ = flow

        root_result_path = os.path.join(work_dir, 'all_final_eval')
        root_result_path_1 = os.path.join(root_result_path, '1', mode+'.csv')
        df_all = pd.read_csv(root_result_path_1)

        for i in range(2, cv + 1):
            root_result_path_temp = os.path.join(root_result_path, str(i), mode+'.csv')
            df_temp = pd.read_csv(root_result_path_temp)
            df_all = df_all.append(df_temp)


        df_all['demo_data_is_flipped'] = df_all.apply(label_flipped, axis=1)
        # df_all['join_id'] = df_all.apply(generate_id_label_DBS, axis=1)
        raw_results_dict[mode] = copy.deepcopy(df_all)


        wandb.log({mode+'_CSV': wandb.Table(dataframe=df_all)})
        reg_fig_DBS, reg_fig_MEDS, con_mat_fig_normed, con_mat_fig = createSummaryPlots(df_all, num_class)

        log_vars = computeSummaryStats(df_all, num_class, mode)
        wandb.log(log_vars)

        wandb.log({"confusion_matrix/" + mode + "_final_confusion_matrix.png": con_mat_fig})
        wandb.log({"confusion_matrix/" + mode + "_final_confusion_matrix_normed.png": con_mat_fig_normed})
        if mode == "test":
            _, results_dict = compute_obj2_stats(df_all)
            wandb.log(results_dict)

            wandb.log({"regression_plot/"+ mode + "_final_regression_DBS.png": [wandb.Image(reg_fig_DBS)]})
            wandb.log({"regression_plot/"+ mode + "_final_regression_MEDS.png": [wandb.Image(reg_fig_MEDS)]})


    # Compute stats for each fold
    for i in range(cv):
        plt.close('all')
        fold_num = i + 1
        log_name="CV_" + str(fold_num) 
        wandb.init(name=log_name, project=wandb_project, group=wandb_group, config = {'wandb_group':wandb_group}, tags=['summary'], reinit=True)

        for _, flow in enumerate(workflow):
            mode, _ = flow
            df_all = raw_results_dict[mode]
            df_test = df_all[df_all['amb'] == fold_num]

            # Compute stats across all folds
            log_vars = computeSummaryStats(df_test, num_class, mode)
            wandb.log(log_vars)
            df = pd.DataFrame(log_vars, index=[0])
            results_table = results_table.append(df)

            reg_fig_DBS, reg_fig_MEDS, con_mat_fig_normed, con_mat_fig = createSummaryPlots(df_test, num_class)
            wandb.log({"confusion_matrix/" + mode + "_final_confusion_matrix.png": con_mat_fig})
            wandb.log({"confusion_matrix/" + mode + "_final_confusion_matrix_normed.png": con_mat_fig_normed})
            if mode == "test":
                _, results_dict = compute_obj2_stats(df_all)
                wandb.log(results_dict)

                wandb.log({"regression_plot/"+ mode + "_final_regression_DBS.png": [wandb.Image(reg_fig_DBS)]})
                wandb.log({"regression_plot/"+ mode + "_final_regression_MEDS.png": [wandb.Image(reg_fig_MEDS)]})


    final_stats_variance(results_table, wandb_group, wandb_project, total_epochs, num_class, workflow)

def set_up_results_table_objective_2():
    col_names = []
    ind = ['MEDS', 'DBS']
    paired = ['unpaired']
    direction = ['allwalks', 'forwardwalks', 'backwardwalks']
    stat = ['tstatistic', 'pvalue', 'pos_num_samples', 'neg_num_samples', 'total_num_samples']

    for i in ind:
        for p in paired:
            for d in direction:
                for s in stat:
                    col_names.append("_".join([i, p, d, s]))
    print(col_names)
    df = pd.DataFrame(columns=col_names)
    return df

def label_flipped(row):
    if 'flipped' in row['walk_name']:
        return 1
    return 0

def generate_id_label_DBS(row):
    data = [row['amb'], row['demo_data_patient_ID'], row['demo_data_patient_ID'], row['demo_data_is_backward'], row['demo_data_is_flipped']] #, row['demo_data_DBS']]
    data = [str(s) for s in data]
    return "_".join(data)

def createSummaryPlots(df_all, num_class):
    true_labels = df_all['true_score']
    preds = df_all['pred_round']
    preds_raw = df_all['pred_raw']
    class_names = [str(i) for i in range(num_class)]
    try:
        dbs_label = df_all['demo_data_DBS']
        meds_label = df_all['demo_data_MEDS']
    except:
        dbs_label = [-1] * len(true_labels)
        meds_label = [-1] * len(true_labels)

    fig_title = "Regression for unseen participants - DBS"
    reg_fig_DBS = regressionPlotByGroup(true_labels, preds_raw, class_names, fig_title, dbs_label)
    fig_title = "Regression for unseen participants - MEDS"
    reg_fig_MEDS = regressionPlotByGroup(true_labels, preds_raw, class_names, fig_title, meds_label)
    con_mat_fig_normed = plot_confusion_matrix( true_labels,preds, class_names, num_class, True)
    con_mat_fig= plot_confusion_matrix( true_labels,preds, class_names, num_class, False)

    return reg_fig_DBS, reg_fig_MEDS, con_mat_fig_normed, con_mat_fig

def computeSummaryStats(df_all, num_class, mode):
    class_names_int = [int(i) for i in range(num_class)]

    # Only use labelled walks to calculate metrics
    true_labels = df_all.loc[df_all['true_score'] >= 0, 'true_score']
    preds = df_all.loc[df_all['true_score'] >= 0, 'pred_round']
    preds_raw = df_all.loc[df_all['true_score'] >= 0, 'pred_raw']

    log_vars = {}
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
        # print(cur_metrics)
        for s in range(len(class_names_int)):
            log_vars[prefix_name + str(class_names_int[s]) + '_'+ average_metrics_to_log[c]] = cur_metrics[s]


    # Keep the original metrics for backwards compatibility
    log_vars[prefix_name + 'mae_rounded'] = mean_absolute_error(true_labels, preds)
    log_vars[prefix_name + 'mae_raw'] = mean_absolute_error(true_labels, preds_raw)
    log_vars[prefix_name + 'accuracy'] = accuracy_score(true_labels, preds)

    return log_vars

def compute_obj2_stats(df_all):
    results_df= {}
    ind = ['MEDS', 'DBS']
    paired = ['unpaired']
    direction = {'allwalks': '', 'forwardwalks': 'forward', 'backwardwalks': 'backward'}
    stat = ['tstatistic', 'pvalue', 'pos_num_samples', 'neg_num_samples', 'total_num_samples']


    from scipy.stats import ttest_ind  
    
    def t_test(x,y,equal_var, alternative='less'):
            double_t, double_p = ttest_ind(x,y, nan_policy='omit', equal_var = equal_var)
            if alternative == 'both-sided':
                pval = double_p
            elif alternative == 'greater':
                if np.mean(x) > np.mean(y):
                    pval = double_p/2.
                else:
                    pval = 1.0 - double_p/2.
            elif alternative == 'less':
                if np.mean(x) < np.mean(y):
                    pval = double_p/2.
                else:
                    pval = 1.0 - double_p/2.
            return double_t, pval

    for i in ind:
        for p in paired:
            for d in direction:

                search_dir = direction[d]
                demo_str = "demo_data_" + i
                # Filter df_all to only extract walks of interest
                test_df = df_all[df_all['walk_name'].str.contains(search_dir)]
                test_df = test_df[test_df[demo_str] >= 0]

                off_condition = test_df[test_df[demo_str] == 0]
                on_condition = test_df[test_df[demo_str] == 1]
                # print(off_condition)
                # print(on_condition)

                # comparison_df = off_condition.merge(
                #     on_condition,
                #     indicator=True,
                #     how='inner', 
                #     on='join_id'
                # )
                # # Set ipython's max row display
                # pd.set_option('display.max_row', 1000)
                # pd.set_option('display.max_colwidth', None)
                # print(comparison_df)
                # ids_unique = comparison_df.join_id.unique()
                # print(len(ids_unique))
                # print(comparison_df[comparison_df['join_id'] == ids_unique[0]]['walk_name_y'])
                # # print(comparison_df[comparison_df['join_id'] == ids_unique[0]])
                # if p is 'paired':
                    
                #     pass


                off_condition_vals = off_condition['pred_raw'].to_list()
                on_condition_vals = on_condition['pred_raw'].to_list()
                import statistics
                
                # print(statistics.mean(off_condition_vals))
                # print(statistics.mean(on_condition_vals))
                # print(df_all)

                tstat_welch, p_val_welch = t_test(on_condition_vals, off_condition_vals,equal_var=False) # welch
                tstat_t, p_val_t = t_test(on_condition_vals, off_condition_vals,equal_var=True) # student's t-test
                tstat_mw, p_val_mw = scipy.stats.mannwhitneyu(on_condition_vals, off_condition_vals, use_continuity=True, alternative='less') # mann whitney test



                # Save values to df 
                stat_base = "_".join([i, p, d])
                results_df[stat_base + "_welch_tstatistic"] = tstat_welch
                results_df[stat_base + "_welch_pvalue"] = p_val_welch
                results_df[stat_base + "_t_tstatistic"] = tstat_t
                results_df[stat_base + "_t_pvalue"] = p_val_t

                results_df[stat_base + "_mannwhitney_tstatistic"] = tstat_mw
                results_df[stat_base + "_mannwhitney_pvalue"] = p_val_mw

                results_df[stat_base + "_pos_mean"] = statistics.mean(on_condition_vals)
                results_df[stat_base + "_neg_mean"] = statistics.mean(off_condition_vals)
                results_df[stat_base + "_pos_stdev"] = statistics.stdev(on_condition_vals)
                results_df[stat_base + "_neg_stdev"] = statistics.stdev(off_condition_vals)
                results_df[stat_base + "_pos_num_samples"] = len(on_condition)
                results_df[stat_base + "_neg_num_samples"] = len(off_condition)
                results_df[stat_base + "_total_num_samples"] = len(off_condition) + len(on_condition)
                results_df[stat_base + "_pos_shapiro_teststat"],  results_df[stat_base + "_pos_shapiro_pval"]= scipy.stats.shapiro(on_condition_vals)
                results_df[stat_base + "_neg_shapiro_teststat"],  results_df[stat_base + "_neg_shapiro_pval"]= scipy.stats.shapiro(off_condition_vals)

    return pd.DataFrame(results_df, index=[0]), results_df


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



    optimizer = call_obj(params=model.parameters(), **optimizer_cfg_local)
    runner = Runner(model, batch_processor, optimizer, work_dir, log_level, num_class=num_class, \
                    things_to_log=things_to_log, early_stopping=early_stopping, force_run_all_epochs=force_run_all_epochs, \
                    es_patience=es_patience, es_start_up=es_start_up, freeze_encoder=freeze_encoder, finetuning=True)
    runner.register_training_hooks(**training_hooks_local)

    # run
    runner.early_stop_eval(workflow, data_loaders, loss=loss, flip_loss_mult=flip_loss_mult, balance_classes=balance_classes, class_weights_dict=class_weights_dict)
    # final_model, num_epoches_early_stop_finetune = runner.run(data_loaders, workflow, total_epochs, train_extrema_for_epochs=train_extrema_for_epochs, loss=loss, flip_loss_mult=flip_loss_mult, balance_classes=balance_classes, class_weights_dict=class_weights_dict)
    
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
