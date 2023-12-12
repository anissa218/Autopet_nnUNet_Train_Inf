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

export nnUNet_raw="/well/papiez/users/hri611/python/nnUnet_raw"
export nnUNet_preprocessed="/well/papiez/users/hri611/python/nnUNet_preprocessed"
export nnUNet_results="/well/papiez/users/hri611/python/nnUNet_results"

module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/11.7.0

source env1-${MODULE_CPU_TYPE}/bin/activate

#nnUNetv2_plan_and_preprocess -d 221 -c 3d_fullres --verify_dataset_integrity

sleep 240

nnUNetv2_train 221 3d_fullres 4 --npz -device cuda --c #0 is which of 5 CV folds is trained



#submit with: sbatch -p gpu_long --gres gpu:1 0-219-2000e-noM-nnunet.sh
