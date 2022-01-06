#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=logs/slurm-%j.txt
#SBATCH --open-mode=append
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=t4v1,t4v2,p100
#SBATCH --cpus-per-task=8
#SBATCH --mem=25GB
#SBATCH --exclude=gpu109

dataset=$1
data_dir=$2
env=$3
algorithm=$4
index=$5
output_dir=$6

python3 -m domainbed.scripts.my_train \
       --data_dir=$data_dir \
       --algorithm $algorithm \
       --dataset $dataset \
       --train_env $env \
       --output_dir "${output_dir}/${algorithm}-${dataset}-${env}-${index}" \ 
       --seed $index 