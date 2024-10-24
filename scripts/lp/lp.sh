#!/bin/bash

model=$1
dataset=$2

python -m src.link_prediction.tune  --model $model --dataset $dataset
python -m src.link_prediction.train --model $model --dataset $dataset --valid 5
python -m src.link_prediction.test  --model $model --dataset $dataset 
python -m src.select_preds --model $model --dataset $dataset
python -m src.select_preds --model $model --dataset $dataset