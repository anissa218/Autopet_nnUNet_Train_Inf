#!/bin/bash

#$ -P papiez.prjc
#$ -N nn-unet

echo "------------------------------------------------"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
echo "------------------------------------------------"
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/11.7.0

source env1-${MODULE_CPU_TYPE}/bin/activate

python pp-inf.py

#submit with: sbatch -p gpu_long --gres gpu:1 f-219-1500e-1e2lr-inference.sh
