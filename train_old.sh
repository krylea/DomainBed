#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=logs/slurm-%j.txt
#SBATCH --open-mode=append
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=t4v1,t4v2,p100
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=25GB
#SBATCH --exclude=gpu109

python3 -m domainbed.scripts.train \
       --data_dir="./domainbed/data/MNIST/" \
       --algorithm ERM \
       --dataset RotatedMNIST \
       --output_dir "./results/test-mnist-base"