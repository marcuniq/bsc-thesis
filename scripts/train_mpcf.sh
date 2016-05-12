#!/bin/bash
#SBATCH --job-name=mpcf
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --partition=kraken_fast
#SBATCH --export=ALL
#SBATCH -o train_mpcf.out
#SBATCH -e train_mpcf.err

script=train_mpcf.py
projectDir=~/bsc-thesis/code

/home/user/unternaehrer/anaconda2/bin/python $projectDir/$script

