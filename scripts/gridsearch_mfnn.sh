#!/bin/bash
#SBATCH --job-name=mfnn
#SBATCH -N 1
#SBATCH --cpus-per-task=24
#SBATCH --partition=kraken_slow
#SBATCH --export=ALL
#SBATCH -o gridsearch_mfnn.out
#SBATCH -e gridsearch_mfnn.err

script=gridsearch_mfnn.py
projectDir=~/bsc-thesis/code
workingDir=/home/slurm/unternaehrer-${SLURM_JOB_ID}

THEANO_FLAGS="floatX=float32,base_compiledir=$workingDir" /home/user/unternaehrer/anaconda2/bin/python $projectDir/$script
