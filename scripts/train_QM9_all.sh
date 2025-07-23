#!/bin/bash
#SBATCH --account=lyu
#SBATCH --partition=normal

#SBATCH --array=0-16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G

#SBATCH --time=24:00:00
#SBATCH --job-name=RCD_QM9

#SBATCH --output=logs/log_%A_%a.out
#SBATCH --error=logs/err_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=da720397@ucf.edu

set -euo pipefail

#FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" data/target_list.txt)
#FILENAME=$(basename "$FILE" .dat)

echo "Running target: $FILENAME"
/home/da720397/.conda/envs/descriptor_env/bin/python training/train.py \
  --DESC             descriptors/rcp_qm9.npy \
  --TARGET           "$FILE"                 \
  --OUT              "results/${FILENAME}"   \
  --N                16                      \
  --TRAINFRAC        "0.25"                  \
  --KERNEL           rbf                     \
  --METRIC           MAE                     \
  --STRATA           10                      \
  --KFOLD            5                       \
  --NORMALIZE_DESC   0                       \
  --NORMALIZE_TARGET 0                       \
  --SIGMA_MIN        0                       \
  --SIGMA_MAX        20                      \
  --LAMBDA_MIN       -12                     \
  --LAMBDA_MAX       -1                      \
  --ITER             50                      \
  --PLOT             1                       \
  --VERBOSE          1                       \
  --DEBUG            0
