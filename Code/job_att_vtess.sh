#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --partition=thin

logpath="log/"
mkdir -p $logpath
logfile="$logpath/${SLURM_ARRAY_TASK_ID}.out"

echo "Writing Python log to ${logfile}"
scontrol show -dd job $SLURM_JOB_ID

srun python attrbute_vtess.py  ${SLURM_ARRAY_TASK_ID} > ${logfile}