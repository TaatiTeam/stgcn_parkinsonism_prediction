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
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix
import wandb
import matplotlib.pyplot as plt
# from spacecutter.models import OrdinalLogisticModel
# import spacecutter
import pandas as pd
import pickle
#os.environ['WANDB_MODE'] = 'dryrun'

num_class = 3
balance_classes = False
class_weights_dict = {}
flip_loss_bool = False

def test(model_cfg, dataset_cfg, checkpoint, batch_size=64, gpus=1, workers=4):
    dataset = call_obj(**dataset_cfg)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=workers)

    # put model on gpus
    if isinstance(model_cfg, list):
        model = [call_obj(**c) for c in model_cfg]
        model = torch.nn.Sequential(*model)
    else:
        model = call_obj(**model_cfg)
    load_checkpoint(model, checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=range(gpus)).cuda()
    model.eval()

    results = []
    labels = []
    prog_bar = ProgressBar(len(dataset))
    for data, label in data_loader:
        with torch.no_grad():
            output = model(data).data.cpu().numpy()
        results.append(output)
        labels.append(label)
        for i in range(len(data)):
            prog_bar.update()
    results = np.concatenate(results)
    labels = np.concatenate(labels)

    print('Top 1: {:.2f}%'.format(100 * topk_accuracy(results, labels, 1)))
    print('Top 5: {:.2f}%'.format(100 * topk_accuracy(results, labels, 5)))


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
        flip_loss=False,
        weight_classes=False,
        group_notes='',
        launch_from_local=False,
        wandb_project="mmskel",
):

    global flip_loss_bool
    flip_loss_bool = flip_loss

    global balance_classes
    balance_classes = weight_classes

    outcome_label = dataset_cfg[0]['data_source']['outcome_label']
    global num_class
    num_class = model_cfg['num_class']
    wandb_group = wandb.util.generate_id() + "_" + outcome_label + "_" + group_notes
    wandb_group = "2goee0h1_SAS_gait_150_all_simplified_3"
    wandb_group = "xnhol0sq_UPDRS_gait_150_all_simplified_3"
    wandb_group = "2rn70zfb_UPDRS_gait_120_all_simplified_3"
#    wandb_group = "38tznb2y_SAS_gait_120_all_simplified_3"

    print("ANDREA - TRI-recognition: ", wandb_group)

    id_mapping = {27:25, 33:31, 34:32, 37:35, 39:37,
                  46:44, 47:45, 48:46, 50:48, 52:50, 
                  55:53, 57:55, 59:57, 66:63}


    # prepare data loaders
    if isinstance(dataset_cfg, dict):
        dataset_cfg = [dataset_cfg]

    print("==================================")
    print('have cuda: ', torch.cuda.is_available())
    print('using device: ', torch.cuda.get_device_name())
    # print(dataset_cfg[0])
    assert len(dataset_cfg) == 1
    data_dir = dataset_cfg[0]['data_source']['data_dir']
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    workflow_orig = copy.deepcopy(workflow)



    # Compute summary statistics (accuracy and confusion matrices)
    final_results_dir = os.path.join(work_dir, 'all_test', wandb_group)
    wandb.init(name="ALL", project=wandb_project, group=wandb_group, tags=['summary'], reinit=True)
    print(final_results_dir)
    for e in range(0, total_epochs):
        log_vars = {}
        results_file = os.path.join(final_results_dir, "test_" + str(e + 1) + ".csv")
        df = pd.read_csv(results_file)
        true_labels = df['true_score']
        preds = df['pred_round']
        preds_raw = df['pred_raw']

        log_vars['val/mae_rounded'] = mean_absolute_error(true_labels, preds)
        log_vars['val/mae_raw'] = mean_absolute_error(true_labels, preds_raw)
        log_vars['val/accuracy'] = accuracy_score(true_labels, preds)
        wandb.log(log_vars, step=e+1)

        if e % 5 == 0:
            class_names = [str(i) for i in range(num_class)]

            fig = plot_confusion_matrix( true_labels,preds, class_names)
            wandb.log({"confusion_matrix/val_"+ str(e)+".png": fig}, step=e+1)
            fig_title = "Regression for ALL unseen participants"
            reg_fig = regressionPlot(true_labels,preds, class_names, fig_title)
            try:
                wandb.log({"regression/val_"+ str(e)+".png": [self.wandb.Image(reg_fig)]}, step=e+1)
            except:
                pass
    
    return





    for test_id in test_ids:
        plt.close('all')
        ambid = id_mapping[test_id]

        test_walks = [i for i in all_files if re.search('ID_'+str(test_id), i) ]
        non_test_walks = list(set(all_files).symmetric_difference(set(test_walks)))
    
        datasets = [copy.deepcopy(dataset_cfg[0]) for i in range(len(workflow))]
        # datasets = [copy.deepcopy(dataset_cfg[0]), copy.deepcopy(dataset_cfg[0])]
        work_dir_amb = work_dir + "/" + str(ambid)
        for ds in datasets:
            ds['data_source']['layout'] = model_cfg['graph_cfg']['layout']


        if len(test_walks) == 0:
            continue
        
        # Split the non_test walks into train/val
        kf = KFold(n_splits=cv, shuffle=True, random_state=1)
        kf.get_n_splits(non_test_walks)


        num_reps = 1
        for train_ids, val_ids in kf.split(non_test_walks):
            if num_reps > 1:
                break
            num_reps += 1
            train_walks = [non_test_walks[i] for i in train_ids]
            val_walks = [non_test_walks[i] for i in val_ids]

            plt.close('all')
            ambid = id_mapping[test_id]

            test_walks = [i for i in all_files if re.search('ID_'+str(test_id), i) ]
            non_test_walks = list(set(all_files).symmetric_difference(set(test_walks)))
        
            if exclude_cv: 
                workflow = [workflow_orig[0], workflow_orig[2]]
                datasets = [copy.deepcopy(dataset_cfg[0]) for i in range(len(workflow))]
                datasets[0]['data_source']['data_dir'] = non_test_walks
                datasets[1]['data_source']['data_dir'] = test_walks
            else:
                datasets[0]['data_source']['data_dir'] = train_walks
                datasets[1]['data_source']['data_dir'] = val_walks
                datasets[2]['data_source']['data_dir'] = test_walks
            work_dir_amb = work_dir + "/" + str(ambid)
            for ds in datasets:
                ds['data_source']['layout'] = model_cfg['graph_cfg']['layout']
            # x = dataset_cfg[0]['data_source']['outcome_label']
    
            print(workflow)
            # print(model_cfg['num_class'])
            things_to_log = {'keypoint_layout': model_cfg['graph_cfg']['layout'], 'outcome_label': outcome_label, 'num_class': num_class, 'wandb_project': wandb_project, 'wandb_group': wandb_group, 'test_AMBID': ambid, 'test_AMBID_num': len(test_walks), 'model_cfg': model_cfg, 'loss_cfg': loss_cfg, 'optimizer_cfg': optimizer_cfg, 'dataset_cfg_data_source': dataset_cfg[0]['data_source'], 'notes': notes, 'batch_size': batch_size, 'total_epochs': total_epochs }
            print('size of train set: ', len(datasets[0]['data_source']['data_dir']))
            print('size of test set: ', len(test_walks))

            if launch_from_local:

                file_path = 'C:/Users/Andrea/andrea/mmskeleton/mmskeleton/processor/recognition_tri_win_train.py'
                pkl_file = os.path.join(work_dir, 'obj.pkl')
                vars_to_save = [work_dir_amb,
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
                    things_to_log]

                with open(pkl_file, 'wb') as f:
                    pickle.dump(vars_to_save, f)


                os_call = f"python {file_path} --pkl_file {pkl_file}"


                print("os call: ", os_call)
                os.system(os_call)

            else: # Launching from linux
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
                        things_to_log)


        


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
        
):
    # print(all_files)
    print("==================================")
    
    # if loss_cfg['type'] == "spacecutter.losses.CumulativeLinkLoss":
    #     print('we have a cumulative loss')
    # else:
    #     print('we DO NOT have a cumulative loss')

    # raise ValueError("stop")


    # print(datasets)

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
    runner = Runner(model, batch_processor, optimizer, work_dir, log_level, things_to_log=things_to_log)
    runner.register_training_hooks(**training_hooks_local)

    if resume_from:
        runner.resume(resume_from)
    elif load_from:
        runner.load_checkpoint(load_from)

    # run
    workflow = [tuple(w) for w in workflow]
    # [('train', 5), ('val', 1)]
    runner.run(data_loaders, workflow, total_epochs, loss=loss)


# process a batch of data
def batch_processor(model, datas, train_mode, loss):
    #torch.cuda.empty_cache()
    #print('have cuda: ', torch.cuda.is_available())
    #print('using device: ', torch.cuda.get_device_name())
    
    mse_loss = torch.nn.MSELoss()
    model_2 = copy.deepcopy(model)
    have_flips = 0
    try:
        data, label , name = datas
    except:
        data, data_flipped, label, name = datas
        have_flips = 1

    data_all = data.cuda()
    label = label.cuda()

    # Remove the -1 labels
    y_true = label.data.reshape(-1, 1).float()
    condition = y_true >= 0.
    row_cond = condition.all(1)
    y_true = y_true[row_cond, :]
    data = data_all.data[row_cond, :]
    num_valid_samples = data.shape[0]


    if have_flips:
        data_all = data_all.data
        data_all_flipped = data_flipped.cuda()
        data_all_flipped = data_all_flipped.data
        # print('input_flipped', torch.sum(torch.isnan(data_all_flipped)))     
        # print('raw', torch.sum(torch.isnan(data_all)))   
        output_all_flipped = model_2(data_all_flipped)


    # Get predictions from the model
    output_all = model(data_all)
    output = output_all[row_cond]
    loss_flip_tensor = torch.tensor([0.], dtype=torch.float, requires_grad=True) 

    if have_flips:
        loss_flip_tensor = mse_loss(output_all_flipped, output_all)
        if loss_flip_tensor.data > 10:
            pass
            # print('output_all_flipped', output_all_flipped, 'output_all', output_all)

    if not flip_loss_bool:
        loss_flip_tensor = torch.tensor([0.], dtype=torch.float, requires_grad=True) 
        loss_flip_tensor = loss_flip_tensor.cuda()

    # if we don't have any valid labels for this batch...
    if num_valid_samples < 1:
        labels = []
        preds = []
        raw_preds = []
        # loss_tensor = torch.tensor([0.], dtype=torch.float, requires_grad=True) 
        # loss_tensor = loss_tensor.cuda()



        log_vars = dict(loss_label=0, loss_flip = loss_flip_tensor.item(), loss_all=loss_flip_tensor.item())
        log_vars['mae_raw'] = 0
        log_vars['mae_rounded'] = 0
        output_labels = dict(true=labels, pred=preds, raw_preds=raw_preds)
        outputs = dict(loss=loss_flip_tensor, log_vars=log_vars, num_samples=0)

        return outputs, output_labels
    


    y_true_orig_shape = y_true.reshape(1,-1).squeeze()
    losses = loss(output, y_true)

    if type(loss) == type(mse_loss):
        if balance_classes:
            losses = weighted_mse_loss(output, y_true, class_weights_dict)
        # Convert the output to classes and clip from 0 to number of classes
        y_pred_rounded = output.detach().cpu().numpy()
        output = y_pred_rounded
        output_list = output.squeeze().tolist()
        y_pred_rounded = y_pred_rounded.reshape(1, -1).squeeze()
        y_pred_rounded = np.round(y_pred_rounded, 0)
        y_pred_rounded = np.clip(y_pred_rounded, 0, 3)
        preds = y_pred_rounded.squeeze().tolist()
    else:    
        rank = output.argsort()
        preds = rank[:,-1].data.tolist()

    labels = y_true_orig_shape.data.tolist()

    # Case when we have a single output
    if type(labels) is not list:
        labels = [labels]
    if type(preds) is not list:
        preds = [preds]
    if type(output_list) is not list:
        output_list = [output_list]

    try:
        labels = [int(cl) for cl in labels]
        preds = [int(cl) for cl in preds]
    except TypeError as e:
        print("got an error: ", e)
        print(labels)
        print(preds)

    overall_loss = losses + loss_flip_tensor
    log_vars = dict(loss_label=losses.item(), loss_flip = loss_flip_tensor.item(), loss_all=overall_loss.item())
    # print('l1', losses, 'l2', loss_flip_tensor)

    try:
        log_vars['mae_raw'] = mean_absolute_error(labels, output)
    except:
        print("labels: ", labels, "output", output)
        print('input', torch.sum(torch.isnan(data_all)))
        print('output_all', output_all, 'output_all_flipped', output_all_flipped)
        raise ValueError('stop')
    log_vars['mae_rounded'] = mean_absolute_error(labels, preds)
    output_labels = dict(true=labels, pred=preds, raw_preds=output_list)
    outputs = dict(loss=overall_loss, log_vars=log_vars, num_samples=len(labels))
    # print(type(labels), type(preds))
    # print('this is what we return: ', output_labels)
    return outputs, output_labels

def my_loss(output, target):
    loss = torch.mean((output - target)**2)
    return loss

#https://discuss.pytorch.org/t/how-to-implement-weighted-mean-square-error/2547
def weighted_mse_loss(input, target, weights):
    error_per_sample = (input - target) ** 2
    numerator = 0
    
    for key in weights:
        numerator += weights[key]

    weights_list = [numerator / weights[int(i.data.tolist()[0])]  for i in target]
    weight_tensor = torch.FloatTensor(weights_list)
    weight_tensor = weight_tensor.unsqueeze(1).cuda()

    loss = torch.mul(weight_tensor, error_per_sample).mean()
    return loss


def topk_accuracy(score, label, k=1):
    rank = score.argsort()
    hit_top_k = [l in rank[i, -k:] for i, l in enumerate(label)]
    accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
    return accuracy


def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv1d') != -1:
        model.weight.data.normal_(0.0, 0.02)
        if model.bias is not None:
            model.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        model.weight.data.normal_(0.0, 0.02)
        if model.bias is not None:
            model.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)


def plot_confusion_matrix( y_true, y_pred, classes,normalize=False,title=None,cmap=plt.cm.Blues):

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape[1] is not len(classes):
        # print("our CM is not the right size!!")
        all_labels = y_true + y_pred
        y_all_unique = list(set(all_labels))
        y_all_unique.sort()

        cm_new = np.zeros((len(classes), len(classes)), dtype=np.int64)
        for i in range(len(y_all_unique)):
            for j in range(len(y_all_unique)):
                i_global = y_all_unique[i]
                j_global = y_all_unique[j]
                cm_new[i_global, j_global] = cm[i,j]
                

        cm = cm_new


    # print(cm)
    # classes = classes[unique_labels(y_true, y_pred).astype(int)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # print("Normalized confusion matrix")
    # else:
        # print('Confusion matrix, without normalization')
# 
    #print(cm)

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
