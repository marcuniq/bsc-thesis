#!/bin/bash
#SBATCH --job-name=bprmf
#SBATCH -N 1
#SBATCH --cpus-per-task=24
#SBATCH --partition=kraken_slow
#SBATCH --export=ALL
#SBATCH -o gridsearch_bprmf.out
#SBATCH -e gridsearch_bprmf.err

script=gridsearch_bprmf.py
projectDir=~/bsc-thesis/code

/home/user/unternaehrer/anaconda2/bin/python $projectDir/$script
