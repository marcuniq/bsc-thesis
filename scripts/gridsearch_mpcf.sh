#!/bin/bash
#SBATCH --job-name=mpcf
#SBATCH -N 1
#SBATCH --cpus-per-task=20
#SBATCH --partition=kraken_slow
#SBATCH --export=ALL
#SBATCH -o gridsearch_mpcf.out
#SBATCH -e gridsearch_mpcf.err

script=gridsearch_mpcf.py
projectDir=~/bsc-thesis/code

/home/user/unternaehrer/anaconda2/bin/python $projectDir/$script

