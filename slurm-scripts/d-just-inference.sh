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

# do post-processing and CV once all are finished
#nnUNetv2_find_best_configuration 219 -c 3d_fullres -tr nnUNetTrainer_probabilisticOversampling_033 -f 0 1 2 3 4

# run inference
nnUNetv2_predict -d Dataset219_PETCT -i nnUnet_raw/Dataset220_PETCT/imagesTs -o nnUNet_results/Dataset219_PETCT/nnUNetTrainer__nnUNetPlans__3d_fullres_maxnum_512_patch_192/inferTs -f  0 1 2 -tr nnUNetTrainer -c 3d_fullres_maxnum_512_patch_192 -p nnUNetPlans

# run CV metrics calculatoin
#python f-calc-metrics-cv.py

#submit with: sbatch -p gpu_long --gres gpu:1 f-219-1500e-1e2lr-inference.sh
