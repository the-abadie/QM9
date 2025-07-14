#----------------
# PACKAGE IMPORTS
#----------------

import numpy as np
import os
import re
import argparse

from copy import deepcopy

#-----------------
# DEFINE CONSTANTS
#-----------------

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--IN"     , type=str  , default="data/senior_densities_raw",
                    help="Path to raw density data.")

parser.add_argument("-o", "--OUT"    , type=str  , default="data",
                    help="Directory to save atomic densities file.")

parser.add_argument("-c", "--CUTOFF" , type=int  , default=1000,
                    help="Cutoff for the number of data points.")

parser.add_argument("-r", "--RES   " , type=int  , default=200,
                    help="Resolution of final densities.")

parser.add_argument("-z", "--Z_SIGMA", type=float, default=0.1,
                    help="Standard deviation for Z-smoothing.")
                    
args = parser.parse_args()

input_path  = args.IN
output_path = args.OUT
N_cutoff    = args.CUTOFF
resolution  = args.RESOLUTION
Z_sigma     = args.Z_SIGMA

dR         :float      = 0.01  # Length, Angstrom
n_elements :int        = 92
bin_size   :int        = N_cutoff // resolution
densities  :np.ndarray = np.zeros((n_elements, resolution))

#--------------
# PROGRAM START
#--------------

files = os.listdir(input_path)
files = [f for f in files if f.endswith(".txt")]

files.sort(key=lambda x: int(re.match(r'(\d+)', x).group(1)))

valences = []

i = 0

for fname in files:
    path = os.path.join(input_path, fname)

    data = np.loadtxt(path, skiprows=2)

    valences.append(float(open(path).readline().split(':')[-1]))

    distribution = data[:N_cutoff, 2]

    if len(distribution) < N_cutoff:
        zeros = np.zeros(N_cutoff - len(distribution), dtype=float)
        distribution = np.concatenate((distribution, zeros))

    densities[i] = distribution.reshape(resolution, bin_size).sum(axis=1) / 100.
    i += 1

densities_nuclear = deepcopy(densities)
densities_valence = deepcopy(densities)

for i in range(n_elements):
    densities_nuclear[i] = (i+1)      *(1.0/(Z_sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*(np.arange(0, resolution)*dR/Z_sigma)**2.0) - 100*densities[i]
    densities_valence[i] = valences[i]*(1.0/(Z_sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*(np.arange(0, resolution)*dR/Z_sigma)**2.0) - 100*densities[i]

#--------------
# WRITE TO FILE
#--------------

np.save(file = f"{output_path}/densities.npy" , 
        arr = densities)
np.save(file = f"{output_path}/nuclear_densities.npy", 
        arr = densities_nuclear)
np.save(file = f"{output_path}/valence_densities.npy", 
        arr = densities_valence)