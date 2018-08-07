'''
auto.py: Ryuichiro Hada, January 2018

This automizes the process running "read.cpp".

'''

import numpy as np
import subprocess, sys, os

phase = int(sys.argv[1])
tag = sys.argv[2]

#******* Flags *********
index_rsd  = "y"          #  to redshift-space

##### simulation #####
planck = "y"          #  for matter or HALO or GAL
GAL    = "y"             #  for galaxie (hod)

##### option #####
SHIFT   = ""             #  to get "true" shift
periodic = "y"           #  To use "periodic" boundary
Random   = "y"           #  To compute "Random"
Uniform  = "y"           #  To compute "Uniform" Random
no_FKP   = "y"           #  Not to use "FKP" as weight
inp_total_num = "y"      #  To input the total number of particles in advance

#******* Options *********
data = np.fromfile('/home/dyt/store/recon_temp/gal_cat-'+tag+'.dat', dtype=np.float32).reshape(-1, 4)
total_par_ori_inp = data.shape[0]  # number of galaxy data

ngridCube = 480
ngrid = [-1, -1, -1]
maxell = 4

sep = 155.0            # Default to max_sep from the file
dsep = 5.00

kmax = 1.00
dk = 0.005
cell = -123.0          # Default to what's implied by the file

z_ave = 0.500
z_ini = 49.0

power_k0 = 10000.0     # [Mpc/h]^3   along with DR12 [Ashley(2016)]
Om_0 = 0.314153        #  matter parameter, planck
Ol_0 = 0.685847        #  dark energy parameter, planck
qperiodic = 1;

#----- Simulation -----
typesim = 0            # the type of Simulation
                       #  0 :  AbacusCosmos  (planck emulator, etc.)
                       #  1 :  Others        (BOSS, etc.)
                       #  2 :  ST            (T.Sunayama provided)
boxsizesim = 1100      # if typesim = 0,  1100 or 720
# phase = 1           # the number of realization

#----- file  -----
#  general
typeobject = 2         # the type of object
                       #  0 :  matter
                       #  1 :  halo
                       #  2 :  galaxy (generated using HOD)

                       # if typeobject = 0,
qshift = 1             # includes the shift field files (_S)
                       #  0 : not,  1 : yes


                       # if typeobject = 1,
M_cut    = 12.60206    # cut-off mass    (= 4*10^{12} solar mass)

                       # if typeobject = 2,

sc       = 0.000       # factor changing the number density
index_sc = 0 if sc == 0.000 else 1 if sc == 0.75 else 2 if sc == -0.75 else 3
                       # (int) 0: default, 1: sc = 0.75, 2: sc = -0.75, 3: sc = 3.000 (fake for "du")             

quniform = 1           # Random distributed "uniformly"
#  final
ratio_RtoD = -1        #  ratio of the number of Random and Data
num_parcentage = 3     #  parcentage of particle used
typeFKP = 0            # the type of estimator
                       #  0 : not FKP,  1 : FKP
def flag_create():
    FLAG = []
    for k, v in globals().items():
        if id(v) == id("y"):
            FLAG.append("-D" + k)
    return FLAG

FLAGS = flag_create()

FLAGS_var = ["-Dmaxcep_ori=" + str(sep), \
             "-Dpower_0=" + str(power_k0), \
             "-Dnum_phase=" + str(phase), \
             "-DM_cut=" + str(M_cut), \
             "-Dindex_sc=" + str(index_sc), \
             "-DNUM_GRID=" + str(ngridCube), \
             "-Dtotal_par_ori_inp=" + str(total_par_ori_inp), \
             "-Dtag=" + str(tag), \
             "-Dredshift=" + str("{0:.3f}".format(z_ave)), \
             "-Dindex_ratio_RtoD=" + str(ratio_RtoD), \
             "-Dindex_num_parcentage=" + str(num_parcentage)
             ]


os.chdir(os.path.dirname(os.path.abspath(__file__)))
cmd_make = ["g++", "read.cpp", "-march=native",  "-fopenmp", "-lgomp", "-fopt-info-vec-missed", \
              "-fopt-info-vec-optimized", "-std=c++11", "-g", "-o", "read_"+tag+".out"] + FLAGS + FLAGS_var
subprocess.call(cmd_make)

cmd_run = ["./read_"+tag+".out"]
subprocess.call(cmd_run)
