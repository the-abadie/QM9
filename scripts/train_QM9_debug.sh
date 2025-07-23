#!/bin/bash
#SBATCH --account=lyu
#SBATCH --partition=normal

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G

#SBATCH --time=24:00:00
#SBATCH --job-name=QM9DEBUG
#SBATCH --output=logs/DEBUG_log_%j.out
#SBATCH --error=logs/DEBUG_err_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=da720397@ucf.edu

set -euo pipefail

FILE=$(head -n 1 data/target_list.txt)
FILENAME=$(basename "$FILE" .dat)

echo "Running target: $FILENAME"
/home/da720397/.conda/envs/descriptor_env/bin/python training/train.py \
  --DESC             descriptors/RCD_0.1.npy \
  --TARGET           "$FILE"                 \
  --OUT              "results/${FILENAME}"   \
  --OUTLIER          1                       \
  -N                 16                      \
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
  --DEBUG            1
