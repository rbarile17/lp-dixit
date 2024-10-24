#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --account=IscrC_DIXTI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --array=1-132

config=./scripts/explain_config.txt

method=$(awk -v ID=$SLURM_ARRAY_TASK_ID '$1==ID {print $2}' $config)
mode=$(awk -v ID=$SLURM_ARRAY_TASK_ID '$1==ID {print $3}' $config)
dataset=$(awk -v ID=$SLURM_ARRAY_TASK_ID '$1==ID {print $4}' $config)
model=$(awk -v ID=$SLURM_ARRAY_TASK_ID '$1==ID {print $5}' $config)
summarization=$(awk -v ID=$SLURM_ARRAY_TASK_ID '$1==ID {print $6}' $config)


export WANDB_MODE=offline
export WANDB_RESUME=allow
export WANDB_NAME="${method}_${mode}_${dataset}_${model}_${summarization}"
export WANDB_RUN_ID=$(echo -n "${method}_${mode}_${dataset}_${model}_${summarization}" | md5sum | awk '{print $1}')

./scripts/explain.sh $method $mode $dataset $model $summarization > "./logs/explain/${method}_${mode}_${dataset}_${model}_${summarization}.log"   2>&1

