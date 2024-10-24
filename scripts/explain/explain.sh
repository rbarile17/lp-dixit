#!/bin/bash

method=$1
mode=$2
dataset=$3
model=$4
summarization=$5

python -m src.explain                        --method $method --mode $mode --dataset $dataset --model $model --summarization $summarization
python -m src.evaluation.re-training         --method $method --mode $mode --dataset $dataset --model $model --summarization $summarization
python -m src.evaluation.re-training_metrics --method $method --mode $mode --dataset $dataset --model $model --summarization $summarization