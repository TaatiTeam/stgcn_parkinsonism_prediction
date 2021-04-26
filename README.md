## Preamble
This repository includes a minimal demo and pretrained spatial-temporal graph convolutional networks (ST-GCN) models proposed in [TODO - link to paper]. The goal of this work is to predict clinical scores of parkinsonism in gait from 2D or 3D skeleton joint trajectories obtained from video. 

This work extends upon work by Yan et al. (https://arxiv.org/abs/1801.07455) on the use of ST-GCNs for action recognition. Modified forks of the `mmcv` and `mmskeleton` repositories developed by `OpenMMLab` (https://openmmlab.com/) are included in this repository. The original GitHub pages for these repositories for these projects are: \
&emsp;    - https://github.com/open-mmlab/mmcv \
&emsp;    - https://github.com/open-mmlab/mmskeleton 


This code was tested on Linux 20.04 with Python 3.7, Pytorch [TODO], and Cuda [TODO].

## Installation
Notes to self:
 - install pyenv and create virtualenv for repository (for a guide, see https://amaral.northwestern.edu/resources/guides/pyenv-tutorial)
 - Tested with Python 3.7.8
 - pyenv virtualenv 3.7.8 stgcn_parkinsonism

 - pip install Cython numpy ninja
 - pip install -r requirements.txt 
 # Install mmcvn
 - pip install -e mmcv/
 # Install mmskeleton
 - cd mmskeleton/mmskeleton/ops/nms/
 - python setup_linux.py develop
 - cd ../../../
 - python setup.py develop --mmdet

- log into WANDB

## Training

## Evalation
