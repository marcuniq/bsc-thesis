#!/bin/bash
#SBATCH --job-name=mpcf_si
#SBATCH -N 1
#SBATCH --cpus-per-task=24
#SBATCH --partition=kraken_slow
#SBATCH --export=ALL
#SBATCH -o out/gridsearch_mpcf_si.out
#SBATCH -e out/gridsearch_mpcf_si.err

script=gridsearch_mpcf_si.py
projectDir=~/bsc-thesis/code
workingDir=/home/slurm/unternaehrer-${SLURM_JOB_ID}

#cp $projectDir/*.py $workingDir
#cp -r $projectDir/data $workingDir/data
#cp -r $projectDir/doc2vec-models $workingDir/doc2vec-models
#cp -r /home/user/unternaehrer/anaconda2/ $workingDir/anaconda2

THEANO_FLAGS="floatX=float32,base_compiledir=$workingDir" /home/user/unternaehrer/anaconda2/bin/python $projectDir/$script

#cp -r $workingDir/models $projectDir/models
#cp -r $workingDir/metrics $projectDir/metrics