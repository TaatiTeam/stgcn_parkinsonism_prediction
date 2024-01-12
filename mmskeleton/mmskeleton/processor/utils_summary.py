import pandas as pd
import pickle
import math
from torch import nn
import shutil
import random
import time
import statistics
import wandb
import copy, os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix, precision_recall_fscore_support


import pandas as pd
import numpy as np
from icecream import ic

def computeAllSummaryStats(work_dir, wandb_group, wandb_project, total_epochs, num_class, workflow, cv):
    raw_results_dict = {}

    log_name = "CV_ALL"
    wandb.init(name=log_name, project=wandb_project, group=wandb_group, config = {'wandb_group':wandb_group}, tags=['summary'], reinit=True)
    wandb.Table.MAX_ROWS =100000
    results_table = set_up_results_table(workflow, num_class)

    have_demo_data = True  # Do we have demographic data (ie. MEDS/DBS status)?
    print(workflow)
    # Load in all the data from all folds
    for i, flow in enumerate(workflow):
        mode, _ = flow

        root_result_path = os.path.join(work_dir, 'all_final_eval')
        root_result_path_1 = os.path.join(root_result_path, '1', mode+'.csv')
        df_all = pd.read_csv(root_result_path_1)

        for i in range(2, cv + 1):
            root_result_path_temp = os.path.join(root_result_path, str(i), mode+'.csv')
            df_temp = pd.read_csv(root_result_path_temp)
            df_all = df_all._append(df_temp)


        df_all['demo_data_is_flipped'] = df_all.apply(label_flipped, axis=1)
        raw_results_dict[mode] = copy.deepcopy(df_all)


        wandb.log({mode+'_CSV': wandb.Table(dataframe=df_all)})
        reg_fig_DBS, reg_fig_MEDS, con_mat_fig_normed, con_mat_fig = createSummaryPlots(df_all, num_class)

        log_vars = computeSummaryStats(df_all, num_class, mode)
        wandb.log(log_vars)
        wandb.log({"confusion_matrix/" + mode + "_final_confusion_matrix.png": con_mat_fig})
        wandb.log({"confusion_matrix/" + mode + "_final_confusion_matrix_normed.png": con_mat_fig_normed})

        if mode == "test" and have_demo_data:
            try:
                _, results_dict = compute_obj2_stats(df_all)
                wandb.log(results_dict)

                wandb.log({"regression_plot/"+ mode + "_final_regression_DBS.png": [wandb.Image(reg_fig_DBS)]})
                wandb.log({"regression_plot/"+ mode + "_final_regression_MEDS.png": [wandb.Image(reg_fig_MEDS)]})

            except:
                print("don't have demographic data")
                have_demo_data = False
                



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
            results_table = results_table._append(df)

            reg_fig_DBS, reg_fig_MEDS, con_mat_fig_normed, con_mat_fig = createSummaryPlots(df_test, num_class)
            wandb.log({"confusion_matrix/" + mode + "_final_confusion_matrix.png": con_mat_fig})
            wandb.log({"confusion_matrix/" + mode + "_final_confusion_matrix_normed.png": con_mat_fig_normed})
            if mode == "test" and have_demo_data:
                _, results_dict = compute_obj2_stats(df_all)
                wandb.log(results_dict)

                wandb.log({"regression_plot/"+ mode + "_final_regression_DBS.png": [wandb.Image(reg_fig_DBS)]})
                wandb.log({"regression_plot/"+ mode + "_final_regression_MEDS.png": [wandb.Image(reg_fig_MEDS)]})


    final_stats_variance(results_table, wandb_group, wandb_project, total_epochs, num_class, workflow)


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

                off_condition_vals = off_condition['pred_raw'].to_list()
                on_condition_vals = on_condition['pred_raw'].to_list()

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


def final_stats_variance(results_df, wandb_group, wandb_project, total_epochs, num_class, workflow):
    wandb.init(name="ALL_var", project=wandb_project, group=wandb_group, tags=['summary'], reinit=True)
    stdev = results_df.std().to_dict()
    means = results_df.mean().to_dict()
    all_stats = dict()
    for k,v in stdev.items():
        all_stats[k + "_stdev"] = stdev[k]
        all_stats[k + "_mean"] = means[k]


    wandb.log(all_stats)


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


def generate_id_label_DBS(row):
    data = [row['amb'], row['demo_data_patient_ID'], row['demo_data_patient_ID'], row['demo_data_is_backward'], row['demo_data_is_flipped']] #, row['demo_data_DBS']]
    data = [str(s) for s in data]
    return "_".join(data)


def label_flipped(row):
    if 'flipped' in row['walk_name']:
        return 1
    return 0


def sync_wandb(wandb_local_id):
    # Sync everything to wandb at the end
    try:
        os.system('wandb sync ' + wandb_local_id)

        # Delete the work_dir if successful sync
        try:
            robust_rmtree(wandb_local_id)
        except:
            logging.exception('This: ')
            print('failed to delete the wandb_log_local_group folder: ', wandb_local_id)

    except:
        print('failed to sync wandb')


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


def regressionPlotByGroup(labels, raw_preds, classes, fig_title, condition_label=None):
    labels = np.asarray(labels)
    raw_preds = np.asarray(raw_preds)
    true_labels_jitter = labels + np.random.random_sample(labels.shape)/6
    
    fig = plt.figure()

    if condition_label is None:
        plt.plot(true_labels_jitter, raw_preds, 'bo', markersize=6)
    else:
        # Plot the conditions in different colors
        condition_label = np.asarray(condition_label)
        one_mask = list(np.argwhere(condition_label == 1).squeeze())
        zero_mask = list(np.argwhere(condition_label == 0).squeeze()) 

        plt.plot(true_labels_jitter[one_mask], raw_preds[one_mask], 'bo', markersize=6)
        plt.plot(true_labels_jitter[zero_mask], raw_preds[zero_mask], 'ro', markersize=4)


    plt.title(fig_title)

    plt.xlim(-0.5, 4.5)
    plt.ylim(-0.5, 4.5)

    plt.xlabel("True Label")
    plt.ylabel("Regression Value")
    return fig


