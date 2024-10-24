#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --account=IscrC_DIXTI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --array=1-27

config=./scripts/lp_config.txt

model=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)
dataset=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)

export WANDB_MODE=offline
export WANDB_RESUME=allow
export WANDB_NAME="${dataset}_${model}"
export WANDB_RUN_ID=$(echo -n "${dataset}_${model}" | md5sum | awk '{print $1}')

./scripts/lp.sh $model $dataset > "./logs/lp/${model}_${dataset}.log"  2>&1
