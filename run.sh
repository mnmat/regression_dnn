#!/bin/bash

learning_rate=$1
batch_size=$2
hidden_layers=$3
alphaL_min=$4
alphaR_min=$5
enable_gpu=True

# Load githup repo

git clone git@github.com:mnmat/regression_dnn.git

python3 grid_search.py \
    --learning_rate $learning_rate \
    --batch_size $batch_size \
    --hidden_layers $hidden_layers \
    --alphaL_min $alphaL_min \
    --alphaR_min $alphaR_min \
    --enable_gpu $enable_gpu

