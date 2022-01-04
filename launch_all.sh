#!/bin/bash

dataset="VLCS"
data_dir="domainbed/data"
algorithm="ERM"
num_envs=4

num_runs=3

for (( i = 0 ; i < $num_envs ; i++ ))
do
    for (( j = 0 ; j < $num_runs ; j++ ))
    do
        sbatch train.sh $dataset $data_dir $i $algorithm $j
    done
done


