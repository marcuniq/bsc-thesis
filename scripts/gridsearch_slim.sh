#!/bin/bash
#SBATCH --job-name=slim
#SBATCH -N 1
#SBATCH --cpus-per-task=20
#SBATCH --partition=kraken_slow
#SBATCH --export=ALL
#SBATCH -o gridsearch_slim.out
#SBATCH -e gridsearch_slim.err

script=gridsearch_slim.py
projectDir=~/bsc-thesis/code

/home/user/unternaehrer/anaconda2/bin/python $projectDir/$script
