#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "ERROR! Illegal number of parameters. Usage: bash install.sh conda_install_path environment_name"
    exit 0
fi

conda_install_path=$1
conda_env_name=$2

echo ""
echo ""

source $conda_install_path/etc/profile.d/conda.sh
echo "****************** Creating conda environment ${conda_env_name} python=3.10 ******************"
conda create -y --name $conda_env_name python=3.10

echo ""
echo ""
echo "****************** Activating conda environment ${conda_env_name} ******************"
conda activate $conda_env_name

## 1. install pytorch according to the matchine's cuda version
echo "****************** Installing pytorch with cuda11.3 ******************"
conda install -y  pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# echo "****************** Installing pytorch with cuda10.2 ******************"
# conda install -y pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

echo ""
echo ""
echo "****************** Installing kornia, e2cnn, pandas, opencv, scipy, tqdm, matplotlib, jupyter, tensorboard ******************"

pip install opencv-python opencv-contrib-python tqdm e2cnn mmcv albumentations pandas matplotlib
pip install  kornia==0.4.1
pip install easydict path colorlog pyflann-py3 h5py tensorboardX ## for evaluate baselines
pip install tensorflow ## for LF-Net

