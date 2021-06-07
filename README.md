# Preamble
This repository includes a minimal demo and pretrained spatial-temporal graph convolutional networks (ST-GCN) models proposed in "Estimating Parkinsonism Severity in Natural Gait Videos of Older Adults with Dementia" by Andrea Sabo, Sina Mehdizadeh, Andrea Iaboni, and Babak Taati (https://arxiv.org/pdf/2105.03464.pdf). The goal of this work is to predict clinical scores of parkinsonism in gait from 2D or 3D skeleton joint trajectories obtained from video. 

This work extends upon work by Yan et al. (https://arxiv.org/abs/1801.07455) on the use of ST-GCNs for action recognition. Modified forks of the `mmcv` and `mmskeleton` repositories developed by `OpenMMLab` (https://openmmlab.com/) are included in this repository. The original GitHub pages for these repositories for these projects are: \
&emsp;    - https://github.com/open-mmlab/mmcv \
&emsp;    - https://github.com/open-mmlab/mmskeleton 


This code was tested on Linux 20.04 with Python 3.7.8, Pytorch 1.5, and Cuda 10.1.

# Installation

 - First install pyenv and the desired Python version (for a guide, see https://amaral.northwestern.edu/resources/guides/pyenv-tutorial)

 - Create and activate virtualenv:
    ```
    pyenv virtualenv 3.7.8 stgcn_parkinsonism
    pyenv activate stgcn_parkinsonism
    ```

 - Install dependencies:
    ```
    pip install Cython numpy ninja
    pip install -r requirements.txt
    ```

 - Install MMCV
    ```
    pip install -e mmcv/
    ```

 - Install MMSkeleton

   ```
    cd mmskeleton/mmskeleton/ops/nms/ 
    python setup_linux.py develop
    cd ../../../
    python setup.py develop --mmdet
   ```

- Login to WANDB
    ```
    wandb login
    ```

# Data Preparation
This library does not provide pose-tracking or data preparation functionality, but can be used to predict parkinsonism scores from 2D or 3D skeleton trajectories. Possible libraries for extracting joint trajectories are OpenPose, Detectron2, AlphaPose, or the Microsoft Kinect. However, as these libraries predict joint positions independently for each frame of the input video, post-processing is necessary to temporally join and smooth the trajectories of the person of interest. Additionally, to use the pretrained models provided in this library, the joint trajectories must be centered at [100, 100, (100 - *z only*)]. This is not strictly necessary if training models from scratch. 
A more detailed description of these data preparation steps is available in our manuscript. 

## Sample data
Some sample input files can be found in `sample_data/data/tri_detectron/stgcn_normalized_100_center_pd_no_norm`. These files should be used as reference for naming conventions. 

## Using your own data format
Note that it is possible to develop and use a different input data format for your dataset. To import your data format for use by this library, you will need to write a new data parser (ie. replacing `SkeletonLoaderTRI` in `loader_tri.py` with your implementation).  


# Evaluating on Pretrained Models
This library relies on YAML files for specifying configurations. YAML configuration files for the best models as proposed in our manuscript are included for each of the five feature sets investigated (OpenPose, Detectron2, ,AlphaPose, Kinect2D, Kinect3D). A sample YAML file for evaluating pretraining models on a new dataset is: `mmskeleton/configs/parkinsonism_prediction/detectron/eval/train_cluster_2_120_pred_15_4_joints_do_0.2.yaml`. 

Modify this file as appropriate to specify the filepath of the dataset to evaluate on. Other parameters may be modified in the YAML file as necessary, but the model settings should not be altered as they correspond to the configurations of the provided pretrained models. 

To run the evaluation workflow provide the appropriate YAML file to `mmskl.py`:
```
    python mmskeleton/mmskl.py mmskeleton/configs/parkinsonism_prediction/detectron/eval/train_cluster_2_120_pred_15_4_joints_do_0.2.yaml
```

# Training
A simple training workflow is provided by this library, allowing for n-fold train/validation on one dataset, and testing on an external dataset. This training workflow requires two input data sources: 
- One data source for the train/validation sets. During n-fold cross-validation, a different train/validation set will be used to train the model. 
- A second data source will be exclusively used for evaluating the trained models. 


A sample YAML file for the training workflow is provided in: `python mmskeleton/mmskl.py mmskeleton/configs/parkinsonism_prediction/detectron/train/train_cluster_2_120_pred_15_4_joints_do_0.2.yaml`

Refer to the comments in this YAML file to appropriately modify the filepaths for these two datasets. 


To run the training workflow provide the appropriate YAML file to `mmskl.py`:
```
    python mmskeleton/mmskl.py mmskeleton/configs/parkinsonism_prediction/detectron/train/train_cluster_2_120_pred_15_4_joints_do_0.2.yaml
```

