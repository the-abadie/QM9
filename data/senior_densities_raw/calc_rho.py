import numpy  as np
import sys
from   scipy.interpolate import interp1d, make_interp_spline   
from   scipy.integrate   import simpson, trapezoid
from   scipy.signal import savgol_filter
from   scipy.ndimage import gaussian_filter1d

"""
 @author Liping Yu
 repurposed for ema4915 2024/2025
 python calc_rrho_edited.py CHG output_x
"""

data = open(sys.argv[1],'r').readlines()
fout = open(sys.argv[2],'w')

a = float(data[1].split()[0])
a1=a*(np.sum([float(x)**2 for x in data[2].split()]))**0.5
a2=a*(np.sum([float(x)**2 for x in data[3].split()]))**0.5
a3=a*(np.sum([float(x)**2 for x in data[4].split()]))**0.5
vol = a1*a2*a3

nat      = np.sum([int(x) for x in data[6].split()])
nx,ny,nz = [ int(x) for x in data[nat+9].split() ]
dx = a1/nx
dy = a2/ny
dz = a3/nz
dvol = dx*dy*dz

"""
 read charge data
"""
ncolums    = len(data[nat+11].split())
grid_lines = int(nx*ny*nz/ncolums)
if float(nx*ny*nz)/ncolums > grid_lines :
   grid_lines += 1

nat_lines = int(nat/ncolums)
if float(nat)/ncolums > nat_lines : 
   nat_lines  += 1

data_1D = []
for l in range(nat+10, nat+10+grid_lines) :
    for x in data[l].split() :
        data_1D.append(float(x))
data_3D = np.reshape(np.array(data_1D),(nx,ny,nz),order='F')/vol
  
print("# total charge (all r's) and rho_min: ", np.sum(data_3D)*dvol, np.min(data_3D))


"""
calculate radial charge density rrho, and Rs
"""
N = nx*ny*nz
Rs_all   = np.zeros(N)
rrho_all = np.zeros(N)

xinds = np.arange(N)//(ny*nz) 
yinds = np.arange(N)//nz%ny
zinds = np.arange(N)%nz

xinds = np.where(xinds > nx-xinds, nx-xinds, xinds)**2
yinds = np.where(yinds > ny-yinds, ny-yinds, yinds)**2
zinds = np.where(zinds > nz-zinds, nz-zinds, zinds)**2

Rs_all   = np.dot( np.transpose([xinds,yinds,zinds]), np.array([dx,dy,dz])**2 )**0.5
rrho_all = data_3D.reshape(N)


# sort Rs_all and remove remove residual charge (<1e-5) beyond r_cut
ind      = np.argsort(Rs_all)
Rs_all   = Rs_all[ind]
rrho_all = rrho_all[ind]

eps      = 1e-5    
i_cut    = len(Rs_all[np.cumsum(rrho_all[::-1])*dvol > eps ])

Rs_red   = Rs_all[:i_cut]
rrho_red = rrho_all[:i_cut]
r_cut    = Rs_red[-1]

Qtot_all = np.sum(rrho_all)*dvol
Qtot_red = np.sum(rrho_red)*dvol

print("Residual charge/points skipped/r_cut:", Qtot_all-Qtot_red, N-len(Rs_red), r_cut)


"""
 grouping the grids at the same R's
"""
counter       = np.arange(1, len(Rs_red))
Rs_splitted   = np.split(Rs_red,   counter[Rs_red[1:]!=Rs_red[:-1]])
rrho_splitted = np.split(rrho_red, counter[Rs_red[1:]!=Rs_red[:-1]])

Rs_red    = np.array([ Rs_splitted[i][0]         for i in range(len(Rs_splitted))])
rrho_red  = np.array([ np.mean(rrho_splitted[i]) for i in range(len(Rs_splitted))])
Qtot_red  = trapezoid(rrho_red*4*np.pi*Rs_red**2, Rs_red)


# rrho interpolation over a regular mesh
f = interp1d(Rs_red, rrho_red, kind='slinear')

step  = 1e-2
Rs_reg   = np.arange(0,r_cut,step)
rrho_reg = f(Rs_reg)
Drho_reg = [rrho_reg[i]*np.pi*4.*Rs_reg[i]**2 for i in range(len(Rs_reg))]
Qtot_reg = simpson(Drho_reg, Rs_reg)

"""
Gaussian smoothing 
"""
sigma = (dvol*3/(4*np.pi))**(1/3.)*0.6

rrho_reg_sm = np.zeros(len(Rs_reg))

for i in range(len(Rs_reg)) : 
    delta_x     = Rs_reg - Rs_reg[i] 
    weights     = np.exp(-0.5*((delta_x /sigma)**2))
    rrho_reg_sm[i] = np.sum(weights*rrho_reg)/np.sum(weights)

    if i%100 == 0 : print("...Calculating rrho at regular R point: ", i, Rs_reg[i], rrho_reg[i])


Drho_reg_sm = 4*np.pi*Rs_reg**2 *rrho_reg_sm
Qtot_reg_sm = simpson(Drho_reg_sm, Rs_reg)

"""
normalized to the right total charge
"""
rrho = rrho_reg_sm/Qtot_reg_sm*round(Qtot_all)
Drho = rrho*4.*np.pi*Rs_reg**2 
Qtot_normalized = simpson(Drho, Rs_reg)

print("# total charge (all r's | regular r's | regular smoothed | normalized): ", Qtot_all, Qtot_reg, Qtot_reg_sm, Qtot_normalized)

"""
write data
"""

# write data for ML
zsigma = 0.1
trho =  round(Qtot_all) * np.exp(-0.5*(Rs_reg/zsigma)**2)/zsigma/(0.5*np.pi)**0.5 - Drho

fout.write("# total charge: %f \n" %(simpson(Drho, Rs_reg)))
fout.write("# r(A)     rho (e/A^3)     4*pi*r^2*rho(e/A)  dip  \n")
for i in range(len(Rs_reg)) :
    fout.write("%7.4f  %15.7E  %15.7E  %15.7E\n" %(Rs_reg[i], rrho[i], Drho[i], trho[i]))

fout.close()

# write source data for debugging...
sout = open('ir_rrho.dat','w') 

sout.write("# total charge (all r's | reg r's | normalized): %f %f %f" %(Qtot_all, Qtot_reg, Qtot_normalized))
sout.write("# r(A)     rho (e/A^3)\n")
sout.write("# ..........................@ all r's...................... \n")
for i in range(len(Rs_all)) :
    sout.write("%7.4f  %15.7E\n" %(Rs_all[i], rrho_all[i]))

sout.write("\n")
sout.write("# ..........................intepolatd @ even-spaced r's.............. \n")
for i in range(len(Rs_reg)) :
    sout.write("%7.4f  %15.7E\n" %(Rs_reg[i], rrho_reg[i]))

sout.write("\n")
sout.write("# ..........................smoothed @  even-spaced r's.............. \n")
for i in range(len(Rs_reg)) :
    sout.write("%7.4f  %15.7E\n" %(Rs_reg[i], rrho_reg_sm[i]))

sout.write("\n")
sout.write("# ..........................normalized @  even-spaced r's.............. \n")
for i in range(len(Rs_reg)) :
    sout.write("%7.4f  %15.7E\n" %(Rs_reg[i], rrho[i]))

sout.close()