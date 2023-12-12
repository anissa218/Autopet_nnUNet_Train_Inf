#!/bin/bash 
 
# This script sets up a task array with a step size of one. 
 
#SBATCH -J PostProcessArray
#SBATCH -p short
#SBATCH--array 5,10,20,40,80 
#SBATCH --requeue 
 
echo `date`: Executing task ${SLURM_ARRAY_TASK_ID} of job ${SLURM_ARRAY_JOB_ID} on `hostname` as user ${USER} 
echo SLURM_ARRAY_TASK_MIN=${SLURM_ARRAY_TASK_MIN}, SLURM_ARRAY_TASK_MAX=${SLURM_ARRAY_TASK_MAX}, SLURM_ARRAY_TASK_STEP=${SLURM_ARRAY_TASK_STEP} 
 
########################################################################################## 
# one-off setup
module load Python/3.10.8-GCCcore-12.2.0


source env1-${MODULE_CPU_TYPE}/bin/activate
########################################################################################## 
 
########################################################################################## 
# Do your per-task processing here 

python post-process-metrics.py ${SLURM_ARRAY_TASK_ID} 

########################################################################################## 
 
echo `date`: task complete 

# submit with sbatch --array 5,10,20,40,80 --cpus-per-task 6 post-process-metrics.sh