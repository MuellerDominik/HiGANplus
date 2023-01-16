#!/bin/sh

NETZH_USR="taskins"
HOME_PATH="/cluster/home/$NETZH_USR"
REPO_PATH="$HOME_PATH/HiGANplus/HiGAN+"

python3 $REPO_PATH/train.py --config $REPO_PATH/configs/gan_iam_euler.yml
