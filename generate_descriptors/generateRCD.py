#----------------
# PACKAGE IMPORTS
#----------------

import numpy as np
import time
import argparse
import os 

from dataclasses import dataclass
from scipy       import interpolate
from joblib      import Parallel, delayed

#----------------------------
# READ COMMAND-LINE ARGUMENTS
#----------------------------

parser = argparse.ArgumentParser()

parser.add_argument("-r", "--RHOS"   , type=str  , default="data/densities.dat",
                    help="Path to atomic density files (.npy).")

parser.add_argument("-m", "--MOLS"   , type=str  , default="data/all_xyz_blocks.xyz",
                    help="Path to .xyz file containing all molecules.")

parser.add_argument("-o", "--OUT"    , type=str  , default="descriptors",
                    help="Path to save RCD descriptors.")

parser.add_argument("-n"             , type=int  , default=16,
                    help="Number of cores for parellization.")

parser.add_argument("-z", "--Z_SIGMA", type=float, default=0.1,
                    help="Standard deviation for Z-smoothing.")

parser.add_argument("-R", "--RES"    , type=int, default=0,
                    help="Base resolution of descriptor. Default is 0, which takes original resolution of input densities.")

parser.add_argument("-v", "--VERBOSE", type=int  , default=1,
                    help="Turn on verbose output.")

args = parser.parse_args()

rho_path   = args.RHOS
mol_path   = args.MOLS
out_path   = args.OUT
n_cores    = args.n
verbose    = args.VERBOSE
Z_sigma    = args.Z_SIGMA
resolution = args.RES

#------------------
# CLASS DEFINITIONS
#------------------

@dataclass(frozen=True)
class Molecule:
    atom_ids : np.ndarray
    positions: np.ndarray

#---------------------
# FUNCTION DEFINITIONS
#---------------------

def getZ(label) -> int:
    assert label != "", "Label must not be empty."
    
    elements="H   He\
        Li  Be  B   C   N   O   F   Ne\
        Na  Mg  Al  Si  P   S   Cl  Ar\
        K   Ca  Sc  Ti  V   Cr  Mn  Fe  Co  Ni  Cu  Zn  Ga  Ge  As  Se  Br  Kr\
        Rb  Sr  Y   Zr  Nb  Mo  Tc  Ru  Rh  Pd  Ag  Cd  In  Sn  Sb  Te  I   Xe\
        Cs  Ba  La  Ce  Pr  Nd  Pm  Sm  Eu  Gd  Tb  Dy  Ho  Er  Tm  Yb\
        Lu  Hf  Ta  W   Re  Os  Ir  Pt  Au  Hg  Tl  Pb  Bi  Po  At  Rn\
        Fr  Ra  Ac  Th  Pa  U".split()    
    
    return elements.index(label)+1

def getValence(Z:int) -> int:
    assert Z > 0

    valences = [1,  2,  1,  2,  3,  4,  5,  6,  7,  8,  1,  2,  3,  4,  5,  6,  7,  8,  7,  8,  3,  4,  5,  6, 
                7,  8,  9, 10, 11, 12,  3,  4,  5,  6,  7,  8,  7, 10, 11, 12, 11,  6,  7,  8,  9, 10, 11, 12, 
                3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,  4, 
                5,  6,  7,  8,  9, 10, 11, 12,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14]
    return valences[Z-1]

def getDensity(Z:int, DATA: np.ndarray) -> np.ndarray:
    return DATA[Z-1]

def getMols(filepath:str) -> list[Molecule]:
    structures = open(filepath,  'r').readlines()

    molecules = []
    n_max = 0

    for line in range(len(structures)):
        x = structures[line].split()

        if len(x) == 1:
            n_atoms = int(x[0])
            if n_atoms > n_max: n_max = n_atoms

            Zs   = np.zeros( n_atoms    , dtype=int)
            xyzs = np.zeros((n_atoms, 3), dtype=float)

            atom_index = 0
            for j in range(line+2, line+2+n_atoms):
                Zs  [atom_index] = getZ(structures[j].split()[0])
                xyzs[atom_index] = np.array([float(val) for val in structures[j].split()[1:4]])

                atom_index += 1
            
            molecules.append(Molecule(atom_ids  = Zs,
                                      positions = xyzs))
            
    return molecules

def generateDescriptor(mol:Molecule, distribution:np.ndarray, derivatives:bool = False, resolution:int = 0) -> np.ndarray:
    Z:np.ndarray = mol.atom_ids
    R:np.ndarray = mol.positions

    nAtoms:int = len(Z)

    i_FP           = np.zeros((2*N)) # 1-body valence overlap fingerprint

    ij_sym_FP      = np.zeros((2*N)) # 2-body valence overlap fingerprint
    ij_antisym_FP  = np.zeros((2*N)) # 2-body valence overlap fingerprint

    ijk_sym_FP     = np.zeros((2*N)) # 3-body valence overlap fingerprint
    ijk_antisym_FP = np.zeros((2*N)) # 3-body valence overlap fingerprint

    for i in range(nAtoms):
        rho_i     = getDensity(Z[i], distribution)

        # Calculate 1-body (self) interaction (i)
        i_FP += rho_i

        for j in range(i+1, nAtoms):
            # print(f"{i}{j}:")

            R_ij = R[i] - R[j]
            d_ij = np.linalg.norm(R_ij)

            # Check if interaction is beyond cutoff.
            if d_ij > 2*cutoff: continue 

            rho_j     = getDensity(Z[j], distribution)

            # Calculate 2-body valence overlap (i~j)
            N_ij = int(d_ij // dR) # Bond distance discretization
            
            ij_overlap = rho_i[N_ij:]*rho_j[:2*N - N_ij]

            ij_sym_FP    [N_ij:] += 0.5*      (ij_overlap + ij_overlap[::-1]) #     Symmetric overlap with self
            ij_antisym_FP[N_ij:] += 0.5*np.abs(ij_overlap - ij_overlap[::-1]) # Antisymmetric overlap with self

            # Calculate 3-body valence overlap (ij~k)
            ij_k_interactions = np.zeros(2*N - N_ij)
            for k in range(nAtoms):
                if k == i or k == j: continue

                d_ik = np.linalg.norm(R[i] - R[k])
                d_jk = np.linalg.norm(R[j] - R[k])

                h_k  = np.linalg.norm(np.cross(R[k] - R[i], R[k] - R[j]))/d_ij
                
                if (d_ik >= 2*cutoff or d_jk >= 2*cutoff or h_k >= cutoff): continue

                rho_k     = getDensity(Z[k], distribution)

                k_interp = interpolate.interp1d(np.arange(N)*dR, rho_k[N:])

                R_k0 = R[i] - R[k] - R_ij*(cutoff/d_ij)

                xs = np.linalg.norm((np.arange(2*N)*dR)[:,None] * R_ij/d_ij + R_k0, axis=1)
                xs = np.clip(xs, 0, (N-2)*dR)

                ij_k_interactions += k_interp(xs[N_ij:])
            
            ijk_overlap = ij_overlap*ij_k_interactions

            ijk_sym_FP    [N_ij:] += 0.5*      (ijk_overlap + ijk_overlap[::-1]) #     Symmetric overlap with self
            ijk_antisym_FP[N_ij:] += 0.5*np.abs(ijk_overlap - ijk_overlap[::-1]) # Antisymmetric overlap with self

    descriptor = [i_FP, 
                  ij_sym_FP, ij_antisym_FP, 
                  ijk_sym_FP, ijk_antisym_FP]
    
    if derivatives:
        D_i_FP           = np.arange(-N, N) * dR * i_FP           
        D_ij_sym_FP      = np.arange(-N, N) * dR * ij_sym_FP     
        D_ij_antisym_FP  = np.arange(-N, N) * dR * ij_antisym_FP 
        D_ijk_sym_FP     = np.arange(-N, N) * dR * ijk_sym_FP     
        D_ijk_antisym_FP = np.arange(-N, N) * dR * ijk_antisym_FP

        descriptor = [i_FP, 
                      ij_sym_FP , ij_antisym_FP, 
                      ijk_sym_FP, ijk_antisym_FP,

                      D_i_FP, 
                      D_ij_sym_FP , D_ij_antisym_FP, 
                      D_ijk_sym_FP, D_ijk_antisym_FP]
    else:
        descriptor = [i_FP, 
                      ij_sym_FP, ij_antisym_FP, 
                      ijk_sym_FP, ijk_antisym_FP]

    if resolution == 0:
        return np.array(descriptor).ravel()
    
    if N % resolution != 0:
        print(f"`resolution` ({resolution}) does not divide `N` ({N}). Descriptor size will not be reduced.")
        return np.array(descriptor).ravel()
    
    new_descriptor = []

    bin_size = 2*N // resolution

    for desc in descriptor:
        new_descriptor.append(desc.reshape(resolution, bin_size).sum(axis=1))

    return np.array(new_descriptor).ravel()

def normalizeDescriptors(desc: np.ndarray) -> np.ndarray:
    mean = np.mean(desc, axis=0)
    std  = np.std (desc, axis=0)

    std[std == 0] = 1
    return (desc - mean) / std

def compute_descriptor(i, mol, distribution):
    desc = generateDescriptor(mol, distribution, derivatives=True, resolution=0)
    return i, desc

#---------------
# PRE-PROCESSING
#---------------

RHOS = np.load(rho_path)
MOLS = getMols(mol_path)

N     :int   = np.shape(RHOS)[1]  
dR    :float = 0.01              
cutoff:float = N*dR              

RHOS_SYM = np.zeros((len(RHOS), N*2))

for i in range(len(RHOS)):
    RHOS_SYM[i] = np.concatenate((RHOS[i][::-1], RHOS[i])) / 2.0

RHOS_NUC = np.zeros(np.shape(RHOS_SYM))
RHOS_VAL = np.zeros(np.shape(RHOS_SYM))

for i in range(len(RHOS)):
    RHOS_NUC[i] = (i+1)           * (1.0/(Z_sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*(np.arange(-N, N)*dR/Z_sigma)**2.0) - RHOS_SYM[i]*100

    RHOS_VAL[i] = getValence(i+1) * (1.0/(Z_sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*(np.arange(-N, N)*dR/Z_sigma)**2.0) - RHOS_SYM[i]*100

#--------------
# PROGRAM START
#--------------

start_time = time.time()

results = Parallel(n_jobs=n_cores, prefer="processes")(
    delayed(compute_descriptor)(i, mol, RHOS_VAL) for i, mol in enumerate(MOLS)
)

duration = time.time() - start_time

descriptors = [None] * len(MOLS)
for i, desc in results:
    descriptors[i] = desc

np.save(file=f"{out_path}/mol_RCD_{Z_sigma}.npy", arr = descriptors)

if verbose != 0:
    print(
        " ===========================================================\n",
        "RCD DESCRIPTORS COMPLETE\n",
       f"DESCRIPTORS WRITTEN TO {out_path}/RCD_{Z_sigma}.npy ({round(os.path.getsize(f"{out_path}/RCD_{Z_sigma}.npy")/10**9, 2)} GB).\n",
       f"{len(MOLS)} DESCRIPTORS COMPUTED IN {round(duration, 2)} SECONDS ({round(duration/60., 2)} MINUTES).\n",
       f"{round(len(MOLS)/duration, 2)} DESCRIPTORS/SECOND ({round(3600*len(MOLS)/duration, 2)} DESCRIPTORS/HOUR)\n",
       f"{round(len(MOLS)/duration/n_cores, 2)} DESCRIPTORS/SECOND/CORE ({round(3600*len(MOLS)/duration/n_cores, 2)} DESCRIPTORS/HOUR/CORE)\n",
        "===========================================================\n"
    )