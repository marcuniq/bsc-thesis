#!/bin/bash
#SBATCH --job-name=mpcf_si
#SBATCH -N 1
#SBATCH --cpus-per-task=24
#SBATCH --partition=kraken_slow
#SBATCH --export=ALL
#SBATCH -o gridsearch_mpcf_si.out
#SBATCH -e gridsearch_mpcf_si.err

script=gridsearch_mpcf_si.py
projectDir=~/bsc-thesis/code

/home/user/unternaehrer/anaconda2/bin/python $projectDir/$script
