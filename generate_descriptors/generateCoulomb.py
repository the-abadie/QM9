#----------------
# PACKAGE IMPORTS
#----------------

import numpy             as np
import numpy.linalg      as lin
import time
import argparse
import os
from dataclasses import dataclass
from joblib import Parallel, delayed

#------------------
# CLASS DEFINITIONS
#------------------

@dataclass(frozen=True)
class Molecule:
    atom_ids : np.ndarray
    positions: np.ndarray

#----------------------------
# READ COMMAND-LINE ARGUMENTS
#----------------------------

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--IN"     , type=str  , default="data/all_xyz_blocks.xyz",
                    help="Path to .xyz file containing all molecules.")

parser.add_argument("-o", "--OUT"    , type=str  , default="descriptors",
                    help="Path to save Coulomb matrix descriptors.")

parser.add_argument("-v", "--VERBOSE", type=int  , default=1,
                    help="Turn on verbose output.")
args = parser.parse_args()

input_path  = args.IN
output_path = args.OUT
verbose     = args.VERBOSE

#---------------------
# FUNCTION DEFINITIONS
#---------------------

def getZ(label:str) -> int:    
    elements="H   He\
        Li  Be  B   C   N   O   F   Ne\
        Na  Mg  Al  Si  P   S   Cl  Ar\
        K   Ca  Sc  Ti  V   Cr  Mn  Fe  Co  Ni  Cu  Zn  Ga  Ge  As  Se  Br  Kr\
        Rb  Sr  Y   Zr  Nb  Mo  Tc  Ru  Rh  Pd  Ag  Cd  In  Sn  Sb  Te  I   Xe\
        Cs  Ba  La  Ce  Pr  Nd  Pm  Sm  Eu  Gd  Tb  Dy  Ho  Er  Tm  Yb\
        Lu  Hf  Ta  W   Re  Os  Ir  Pt  Au  Hg  Tl  Pb  Bi  Po  At  Rn\
        Fr  Ra  Ac  Th  Pa  U".split()    
    
    return elements.index(label)+1

def importQM7(structure_file:str):
    """
    Return: Z, R, E\n
    Z: list of 1D-arrays containing atomic identities\n
    R: list of 2D-arrays containing atomic positions\n
    E: 1D-array containing atomization energy\n
    """
    structures = open(structure_file,  'r').readlines()

    Z = []
    R = []
    E = []
    n_max = 0

    for line in range(len(structures)):
        x = structures[line].split()

        #Check for start of molecule structure data:
        if len(x) == 1:
            n_atoms = int(x[0])
            if n_atoms > n_max: n_max = n_atoms

            Zs   = np.zeros(n_atoms)
            xyzs = np.zeros((n_atoms, 3))

            #Go through every atom in the molecule:
            atom_index = 0
            for j in range(line+2, line+2+n_atoms):
                Zs  [atom_index] = getZ(structures[j].split()[0])
                xyzs[atom_index] = np.array([float(val) for val in structures[j].split()[1:4]])

                atom_index += 1
            
            Z.append(Zs)
            R.append(xyzs)
    
    return Z, R

def coulomb_matrix(Z, R, n_max):
    n_mols = len(Z)

    #Generate Descriptors, unique values of the Coulomb Matrix M
    n_unique = int(0.5*n_max*(n_max-1))

    coulomb = np.zeros((n_mols, n_unique))

    for k in range(n_mols):

        n_atoms = len(Z[k])

        M = np.zeros((n_atoms, n_atoms))

        for i in range(n_atoms):
            for j in range(n_atoms):
                if i == j:
                    M[i][j] = 0.5 * (Z[k][i])**2.4
                else:
                    M[i][j] = (Z[k][i]*Z[k][j]) / lin.norm(R[k][i] - R[k][j])**2

        unique_entries = M[np.triu_indices_from(M, k=1)]

        #Append 0s to match molecule with largest number of eigenvalues
        if n_atoms == n_max:
            coulomb[k] = unique_entries
        else:
            coulomb[k] = np.concatenate((unique_entries, [0]*(n_unique-len(unique_entries))))

    return coulomb

def normalizeDescriptors(desc: np.ndarray) -> np.ndarray:
    mean = np.mean(desc, axis=0)
    std  = np.std (desc, axis=0)

    std[std == 0] = 1
    return (desc - mean) / std

#--------------
# PROGRAM START
#--------------

Z, R = importQM7(structure_file = input_path)

start_time = time.time()

coulomb_descriptors = coulomb_matrix(Z     = Z,
                                     R     = R,
                                     n_max = len(max(Z, key=len)))

duration = time.time() - start_time 

#--------------
# WRITE TO FILE
#--------------

np.save(file=f"{output_path}/CM.npy", arr = coulomb_descriptors)

if verbose != 0:
    print(
        " ===========================================================\n",
        "COULOMB MATRIX DESCRIPTORS COMPLETE\n",
       f"DESCRIPTORS WRITTEN TO {output_path}/CM.npy ({round(os.path.getsize(f"{output_path}/CM.npy")/10**9, 2)} GB).\n",
       f"{len(Z)} DESCRIPTORS COMPUTED IN {round(duration, 2)} SECONDS ({round(duration/60., 2)} MINUTES).\n",
       f"{round(len(Z)/duration, 2)} DESCRIPTORS/SECOND ({round(3600*len(Z)/duration, 2)} DESCRIPTORS/HOUR)\n",
        "===========================================================\n"
    )