Notes to self:
 - install pyenv and create virtualenv for repository (https://amaral.northwestern.edu/resources/guides/pyenv-tutorial)
 - Tested with Python 3.7.8
 - pyenv virtualenv 3.7.8 stgcn_parkinsonism

 - pip install Cython numpy ninja
 - pip install -r requirements.txt 
 # Install mmcv
 - pip install -e mmcv/
 # Install mmskeleton
 - cd mmskeleton/mmskeleton/ops/nms/
 - python setup_linux.py develop
 - cd ../../../
 - python setup.py develop --mmdet


- log into WANDB