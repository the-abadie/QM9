#!/bin/bash
#SBATCH --account=lyu   
#SBATCH --partition=normal

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G

#SBATCH --time=24:00:00
#SBATCH --job-name=descriptors

#SBATCH --output=log.out
#SBATCH --error=err.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=da720397@ucf.edu

echo "generate_descriptors/atomicDensities.py"
/home/da720397/.conda/envs/descriptor_env/bin/python \
generate_descriptors/atomicDensities.py \
    --IN      data/senior_densities_raw \
    --OUT     descriptors \
    --CUTOFF  1000 \
    --RES     200 \
    --Z_SIGMA 0.1 \
    --VERBOSE 1

echo "generate_descriptors/generateCoulomb.py"
/home/da720397/.conda/envs/descriptor_env/bin/python \
generate_descriptors/generateCoulomb.py \
    --IN      data/all_xyz_blocks.xyz \
    --OUT     descriptors \
    --VERBOSE 1

echo "generate_descriptors/generateRCD.py"
/home/da720397/.conda/envs/descriptor_env/bin/python \
generate_descriptors/generateRCD.py \
    --RHOS    data/densities.dat \
    --MOLS    data/all_xyz_blocks.xyz \
    --OUT     descriptors \
    --n       16 \
    --Z_SIGMA 0.1 \
    --RES     0 \
    --VERBOSE 1