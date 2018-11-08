'''
auto.py: Ryuichiro Hada, January 2018

This automizes the process running "read.cpp".

'''

# import numpy as np
import subprocess
# import math, subprocess, inspect, os, re
import matplotlib as mpl
mpl.use('AGG')
# import matplotlib.pyplot as plt

#******* Flags *********
index_rsd  = ""          #  to redshift-space

##### simulation #####
planck = "y"          #  for matter or HALO or GAL
BOSS   = ""           #  for matter or HALO
ST     = "" 
HALO   = ""             #  for halo
GAL    = ""             #  for galaxie (hod)

##### option #####

INITIAL = ""             #  to get initial density field
SHIFT   = "y"             #  to get "true" shift
periodic = "y"           #  To use "periodic" boundary
Random   = "y"           #  To compute "Random"
Uniform  = "y"           #  To compute "Uniform" Random
no_FKP   = "y"           #  Not to use "FKP" as weight

#******* Options *********
#------ run -------
ngridCube = 480
ngrid = [ -1, -1, -1]
maxell = 4
sep = 250.0            # Default to max_sep from the file
dsep = 4.00
kmax = 1.00
dk = 0.005
cell = -123.0          # Default to what's implied by the file

z_ave = 0.500
z_ini = 49.0

power_k0 = 10000.0     # [Mpc/h]^3   along with DR12 [Ashley(2016)]
Om_0 = 0.314153        #  matter parameter, planck
Ol_0 = 0.685847        #  dark energy parameter, planck
#Om_0 = 0.2648          #  matter parameter, ST
#Ol_0 = 0.7352          #  dark energy parameter, ST

# for standard
sig_sm_std = 15.0        # [Mpc/h], proper smoothing scale
# for iteration
sig_sm_ite = 20.0        # [Mpc/h], initial smoothing scale
divi_sm = 1.2          #  by which sig_sm is divided each time
last_sm = 15.0         #  to which the division is performed
C_ani = 1.3            #  the ratio of z-axis and x,y-axis
f_eff = 1.0            #  effective linear growth rate

ite_times = 6          #  how many times do we iterate reconstruction
ite_switch = 0         #  when switching to ite_weight_2
ite_weight_ini = 0.7   #  [ini]   (next) = w1*(estimate) + (1-w1)*(previous)
ite_weight_2 = 0.7     #  [2nd]   (next) = w1*(estimate) + (1-w1)*(previous)

# last_sm =  5.0:  ite_times = 17, ite_weight_ini = ite_weight_2 = 0.3
# last_sm =  7.0:  ite_times = 13, ite_weight_ini = ite_weight_2 = 0.4
# last_sm = 10.0:  ite_times =  9, ite_weight_ini = ite_weight_2 = 0.5
# last_sm = 15.0:  ite_times =  6, ite_weight_ini = ite_weight_2 = 0.7

bias = 1.00            #  for matter field by default
bias_uncer = 0        #  uncertainty of bais 
                       #  -1 :   90 % 
                       #   0 :  100 %      
                       #   1 :  110 % 

qperiodic = 1;

#----- Simulation -----
typesim = 0            # the type of Simulation
                       #  0 :  AbacusCosmos  (planck emulator, etc.)
                       #  1 :  Others        (BOSS, etc.)
                       #  2 :  ST            (T.Sunayama provided)
boxsizesim = 1100      # if typesim = 0,  1100 or 720
whichsim = 0           # the number of realization

typehalo = 0           # if typesim = 0 or 1,
                       #  0 :  FoF
                       #  1 :  rockstar

#----- file  -----
#  general
typeobject = 0         # the type of object
                       #  0 :  matter
                       #  1 :  halo
                       #  2 :  galaxy (generated using HOD)

                       # if typeobject = 0,
qshift = 1             # includes the shift field files (_S)
                       #  0 : not,  1 : yes





                       # if typeobject = 1,
M_cut    = 12.60206    # cut-off mass    (= 4*10^{12} solar mass)
sc       = 1.000       # factor changing the number density
index_sc = 0           # (int) 0: default, 1: sc = 1.052, 2: sc = 0.940  

quniform = 1           # Random distributed "uniformly"
#  final
ratio_RtoD = 1        #  ratio of the number of Random and Data
num_parcentage = 3     #  parcentage of particle used
typeFKP = 0            # the type of estimator
                       #  0 : not FKP,  1 : FKP
#  initial
ratio_RtoD_ini = 0     #  ratio of the number of Random and Data
num_parcentage_ini = 1 #  parcentage of particle used
typeFKP_ini = 0        # the type of estimator
                       #  0 : not FKP,  1 : FKP
#----------------------


#******* Inputs *********

inp_all_sim = input("# All of simulations >>")
inp_make = ""
inp_run = ""
inp_a_f = whichsim 
inp_a_l = whichsim
if(inp_all_sim):
    inp_a_f  = input("#     first >>")
    inp_a_l  = input("#     last  >>")
if not(inp_all_sim):
    inp_make = input("# Make >>")
    inp_run  = input("# Run >>")


def flag_create():
    FLAG = []
    for key, value in globals().items():
        if id(value) == id("y"):
            FLAG.append("-D" + key)
    return FLAG
#******* Main *********

#if(INITIAL):
#  ratio_RtoD = ratio_RtoD_ini
#  num_parcentage = num_parcentage_ini
#  typeFKP = typeFKP_ini

for k in range(int(inp_a_f), int(inp_a_l) + 1):
    whichsim = k
    for j in range(2):
        if(j == 0):
            if not(inp_all_sim):
                continue
            inp_make = "y"
            inp_run = ""
        else:
            print("\n# Now, whichsim = " + str(whichsim) + "\n")
            if (inp_all_sim):
                inp_make = ""
                inp_run = "y"

        
        ##########  Make  ###########

        # creating flags

        FLAGS = flag_create()

        FLAGS_var = ["-Dnum_phase=" + str(whichsim), \
                     "-DM_cut=" + str(M_cut), \
                     "-Dindex_sc=" + str(index_sc), \
                     "-DNUM_GRID=" + str(ngridCube), \
                     "-Dredshift=" + str("{0:.3f}".format(z_ave)), \
                     "-Dindex_ratio_RtoD=" + str(ratio_RtoD), \
                     "-Dindex_num_parcentage=" + str(num_parcentage)
                     ]

        cmd_make = ["g++", "read.cpp", "-march=native",  "-fopenmp", "-lgomp", "-fopt-info-vec-missed", \
                      "-fopt-info-vec-optimized", "-std=c++11", "-g"] + FLAGS + FLAGS_var
        if(inp_make):
            print(cmd_make)
            subprocess.run(cmd_make)
        ##########  Run  ###########

        cmd_run = ["./a.out"]
        if (not inp_make and inp_run):
            subprocess.call(cmd_run)
            print("\007")

