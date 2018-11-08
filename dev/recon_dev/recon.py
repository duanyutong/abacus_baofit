'''
auto.py: Ryuichiro Hada, January 2018

This automizes the process inputting, reconstructing, computing correlations,
and plotting figures.

'''

from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import numpy as np
# import math, subprocess, inspect, os, re
import  subprocess
#import matplotlib as mpl
#mpl.use('AGG')
# import matplotlib.pyplot as plt

#******* Flags (for make) *********
RSD = "y"                 # To take account of RSD in reconstruction
DIFF_SHIFT = ""          # To sum the square of difference between shift_t_X and reconstructed shift
RECONST = "y"             # To reconsturuct (standard by default)
ITERATION = ""           # To do the iterative reconstruction
CHANGE_SM = "y"           # To change smoothing scale adptive smoothing scale for iterative reconstruction
SECOND = ""               # To take account of 2nd order in reconstruction
INITIAL = ""             # To compute the corr for the initial density correlation function

#******* Options *********
#------ run -------
ngridCube = 480
ngrid = [ -1, -1, -1]
maxell = 4
sep = np.sqrt(3)*1100  # 250.0            # Default to max_sep from the file
dsep = 4.00
kmax = 1.00
dk = 0.005
cell = -123.0          # Default to what's implied by the file
z_ave = 0.700
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
ite_weight_ini = 0.7  #  [ini]   (next) = w1*(estimate) + (1-w1)*(previous)
ite_weight_2 = 0.7     #  [2nd]   (next) = w1*(estimate) + (1-w1)*(previous)

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
whichsim = 1           # the number of realization
typehalo = 1           # if typesim = 0 or 1,
                       #  0 :  FoF
                       #  1 :  rockstar

#----- file  -----
#  general
typeobject = 2         # the type of object
                       #  0 :  matter
                       #  1 :  halo
                       #  2 :  galaxy (generated using HOD)

                       # if typeobject = 0,
qshift = 0             # includes the shift field files (_S)
                       #  0 : not,  1 : yes

                       # if typeobject = 1,
M_cut    = 12.60206    # cut-off mass    (= 4*10^{12} solar mass)
sc       = 1.00        # factor changing the number density
index_sc = 0           # (int) 0: default, 1: sc = 1.052, 2: sc = 0.940  

quniform = 1           # Random distributed "uniformly"
#  final
ratio_RtoD = 1         #  ratio of the number of Random and Data
num_parcentage = 3     #  parcentage of particle used
typeFKP = 0            # the type of estimator
                       #  0 : not FKP,  1 : FKP
#  initial
ratio_RtoD_ini = 0     #  ratio of the number of Random and Data
num_parcentage_ini = 1 #  parcentage of particle used
typeFKP_ini = 0        # the type of estimator
                       #  0 : not FKP,  1 : FKP

#******* Inputs *********

#inp_all_sim = input("# All of simulations >>")
inp_make = True
inp_run = True
#inp_a_f = whichsim 
#inp_a_l = whichsim
#if(inp_all_sim):
#    inp_a_f  = input("#     first >>")
#    inp_a_l  = input("#     last  >>")
#if not(inp_all_sim):
#    inp_make = input("# Make >>")
#    inp_run  = input("# Run >>")
#inp_plot = input("# Plot >>")
inp_c = ""
inp_xc_s = ""
inp_xc_d = ""
inp_c_var = ""
inp_all  = ""
inp_ini  = ""
inp_ori  = ""
inp_std  = ""
inp_ite  = ""
inp_c_var_f = 0
inp_c_var_l = 0
#if(inp_plot):
#    inp_c = input("#  Correlation >>")
#    if(inp_c):
#        inp_all = input("#     all >>")
#        if not(inp_all):
#            inp_ini = input("#     initial   >>")
#            inp_ori = input("#     original  >>")
#            inp_std = input("#     standard  >>")
#            inp_ite = input("#     iteration >>")
#    inp_xc_s = input("#  X-corr for shift >>")
#    if(inp_xc_s):
#            inp_std = input("#     standard  >>")
#            inp_ite = input("#     iteration >>")
#    inp_xc_d = input("#  X-corr for density >>")
#    if(inp_xc_d):
#            inp_ori = input("#     original  >>")
#            inp_std = input("#     standard  >>")
#            inp_ite = input("#     iteration >>")
#    inp_c_var = input("#  Average for Correlations >>")
#    if(inp_c_var):
#            inp_c_var_f  = input("#     first >>")
#            inp_c_var_l  = input("#     last  >>")
#            inp_ori = input("#     original  >>")
#            inp_std = input("#     standard  >>")
#            inp_ite = input("#     iteration >>")


#******* Main *********

#for k in range(int(inp_a_f), int(inp_a_l) + 1):
#    whichsim = k
#    for j in range(2):
#        if(j == 0):
#            if not(inp_all_sim):
#                continue
#            inp_make = "y"
#            inp_run = ""
#        else:
#            print("\n# Now, whichsim = " + str(whichsim) + "\n")
#            if (inp_all_sim):
#                inp_make = ""
#                inp_run = "y"

#        #******* Path *********
#
#        typehalo_n = "_rockstar" if typehalo else "_FoF"
#        typeobject_n = "" if typeobject == 0 else "_halo" + "_Mc" + str("{0:.4f}".format(M_cut)) if typeobject == 1 \
#                                    else "_gal" if sc == 1.0 else "_gal_sc" + str("{0:.3f}".format(sc))
#        typeFKP_n = "_F" if typeFKP else "_noF"
#        typeFKP_ini_n = "_F" if typeFKP_ini else "_noF"
#        uni_n = "_uni" if quniform else ""
#        periodic_n = "_p" if qperiodic else ""
#        shift_n = "_S" if qshift and typeobject == 0 else ""
#        rsd_n = "_rsd" if RSD else ""
#
#        if (typesim == 0):    # AbacusCosmos
#            path_to_sim = "/AbacusCosmos" \
#                + "/emulator_" + str(boxsizesim) + "box_planck_products" \
#                + "/emulator_" + str(boxsizesim) + "box_planck_00-" + str(whichsim) + "_products" \
#                + "/emulator_" + str(boxsizesim) + "box_planck_00-" + str(whichsim) + typehalo_n + "_halos"
#            whichsim_n = "_00-"
#        elif (typesim == 1):  # Others
#            path_to_sim = "/Others" \
#                + "/BOSS_1600box_products" \
#                + "/BOSS_1600box_FoF_halos"
#            whichsim_n = ""
#        elif (typesim == 2):  # ST
#            path_to_sim = "/ST"
#            whichsim_n = ""
#        else:
#            path_to_sim = "fake"
#            print("# this typesim is not defined !!! ")
#
#        path_me = "/mnt/store1/rhada"
#        path_z = "/z" + "{:.3f}".format(z_ave)
#        # input
#        path_input = path_me + path_to_sim + path_z
#        path_input_ini = path_me + path_to_sim + "/ini"
#        # output
#        typeobject_d = "/matter" if typeobject == 0 else "/halo_Mc" + str("{0:.4f}".format(M_cut)) if typeobject == 1 \
#                                else "/gal" if sc == 1.0 else "/gal_sc" + str("{0:.3f}".format(sc))
#        sample_f = "fin_" + str(ratio_RtoD) + "_" + str(num_parcentage) + typeFKP_n
#        sample_i = "ini_" + str(ratio_RtoD_ini) + "_" + str(num_parcentage_ini) + typeFKP_ini_n
#        sample_d = "/" + sample_i + "_" + sample_f + "_ng" + str(ngridCube) + uni_n + periodic_n
#        sample_ini_d = "/" + sample_i + "_ng" + str(ngridCube) + uni_n + periodic_n
#        run_d = "/sep" + str(sep) + "_dsep" + str(dsep) + "_kmax" + str(kmax) + "_dk" + str(dk)
#        space_d = "/rsd" if RSD else "/real"
#        path_output = "../graph" + path_to_sim + path_z + typeobject_d + sample_d + run_d + space_d + "/file"
#        path_ini_output = "../graph" + path_to_sim + "/ini" + sample_ini_d + run_d + "/file"
#        path_bias_cr = "../graph" + path_to_sim + path_z
#        path_bias = "../graph" + path_to_sim + path_z + typeobject_d + sample_d + run_d 
#
#
#        #******* filename *********
#
#        typeFKP_spe = "" if typeFKP else "_nF"
#        typeFKP_ini_spe = "" if typeFKP_ini else "_nF"
#
#        sample_in_d = "/" + sample_f + "_ng" + str(ngridCube) + uni_n + periodic_n + shift_n
#        sample_ini_in_d  = sample_ini_d

        # input
infile  = '/home/dyt/analysis_scripts/sample_gal.dat'
infile2 = '/home/dyt/analysis_scripts/sample_gal_random.dat'

#        infile_ini  = path_input_ini + sample_ini_in_d + "/file_D"
#        infile_ini2 = path_input_ini + sample_ini_in_d + "/file_R"
#
#        shift_true_x = infile + "_S0"
#        shift_true_y = infile + "_S1"
#        shift_true_z = infile + "_S2"

        # -------------------------

#        bias_n = "" if bias_uncer == 0 else "_b10m" if bias_uncer == -1 else "_b10p"
#        run_ini = "_ini"
#        run_ori = "_ori"
#        run_std = "_std_sm" + str(sig_sm_std) + bias_n
#        run_ite = "_ite_num" + str(ite_times) + "_smi" + str(sig_sm_ite) + "_div" + str(divi_sm) + "_sml" + str(last_sm) \
#               + "_Ca" + str(C_ani) + "_fe" + str(f_eff) \
#               + "_wi" + str(ite_weight_ini) + "_sn" + str(ite_switch) + "_ws" + str(ite_weight_2) + bias_n
sig_sm = sig_sm_std

#        if(INITIAL):
#            path_output = path_ini_output
#            run_id  = run_ini
#            infile = infile_ini
#            infile2 = infile_ini2
#            # flags not needed
#            RSD = ""
#            XCORR_SHIFT = ""
#            XCORR_DENSITY = ""
#            RECONST = ""
if not(RECONST):
    run_id  = 'testorg'
else:
    if not(ITERATION):
        run_id = 'teststd'
    else:
        run_id  = 'testite'
        sig_sm = sig_sm_ite

# output
outfile = '/home/dyt/analysis_script/recon_output.dat'

outfile_corr =  '/home/dyt/analysis_script/recon_output_corr_1.dat'
outfile_corr2 = '/home/dyt/analysis_script/recon_output_corr_2.dat'

#        outfile_xcorr_sh = path_output + "/xcorr_shift" + run_id + ".dat"
#        outfile_xcorr_de = path_output + "/xcorr_dense" + run_id + ".dat"
#
#        def file_exist(filename, inp):
#            inp_l = inp
#            file_path = os.path.dirname(filename)
#            if not os.path.exists(file_path):
#                os.makedirs(file_path)
#            elif os.path.exists(filename) and inp_l:
#                inp_l = input("# This file has already existed !! \n# Do you really want to run ?>>")
#            return inp_l
#        inp_run = file_exist(outfile, inp_run)
#

        #******* Bias arrangemnet *********

#        bias_cr_f = path_bias_cr + "/bias_cr" + "_kmax" + str(kmax) + "_dk" + str(dk) + ".dat"
#        bias_f = path_bias + "/bias.dat"
#        if not(INITIAL):
#            if inp_run and (not os.path.exists(bias_cr_f)) and (not inp_xc_d):
#                print("\n# Need to compute X-corr(i) for density(ori) to get the criteria of bias")
#                inp_make = inp_run = inp_plot = None
#            if not os.path.exists(bias_f):
#                f = open(bias_f,'w')
#                f.write("1.0\n")
#                f.close()
#            with open(bias_f,'r') as f:
#                for bias_r in f:
#                    bias = float(bias_r)
#        if (bias_uncer == -1):
#            bias *= 0.90
#        elif (bias_uncer == 1):
#            bias *= 1.10

        ##########  Make  ###########
        # creating flags
def flag_create():
    FLAG = ""
    for k, v in globals().items():
        if id(v) == id("y"):
            FLAG += " -D" + k
    return FLAG

OMP = " -DOPENMP -DFFTSLAB -DSLAB"
FLAGS = OMP + flag_create()

cmd_make = ["make", "CXXFLAGS=-O3" + FLAGS]
if(inp_make):
    subprocess.run(["make", "clean"])
    #print(cmd_make)
    subprocess.run(cmd_make)

##########  Run  ###########
OPTIONS = ["-ngrid", str(ngridCube), \
           "-sep", str(sep), \
           "-dsep", str(dsep), \
           "-kmax", str(kmax), \
           "-dk", str(dk), \
           "-zave", str(z_ave), \
           "-powerk0", str(power_k0), \
           "-omegam0", str(Om_0), \
           "-omegal0", str(Ol_0), \
           "-sigmasm", str(sig_sm), \
           "-divism", str(divi_sm), \
           "-lastsm", str(last_sm), \
           "-cani", str(C_ani), \
           "-feff", str(f_eff), \
           "-itetimes", str(ite_times), \
           "-iteswitch", str(ite_switch), \
           "-itewini", str(ite_weight_ini), \
           "-itew2", str(ite_weight_2), \
           "-bias", str(bias), \
           "-p" if qperiodic else "",  \
           "-in", infile, \
           "-in2", infile2, \
           # "-inini", infile_ini, \
           # "-inini2", infile_ini2, \
           # "-inshifttruex", shift_true_x, \
           # "-inshifttruey", shift_true_y, \
           # "-inshifttruez", shift_true_z, \
           "-out", outfile, \
           "-outcorr", outfile_corr, \
           "-outcorr2", outfile_corr2]
           #"-outxcorrsh", outfile_xcorr_sh, \
           #"-outxcorrde", outfile_xcorr_de]

cmd_run = ["./reconst"] + OPTIONS
if (not inp_make and inp_run):
    subprocess.call(cmd_run)
    print("\007")


# ##########  Plot  ###########
    # # general
    # if inp_all_sim:
    #     inp_c_var = ""
    # # correlation
    # if inp_c and inp_all:
    #     inp_ini = inp_ori = inp_std = inp_ite = "y"
    # if inp_c_var:
    #     inp_ini = "y"

    # N_b = 1.00 if bias_uncer == 0 else 0.90 if bias_uncer == -1 else 1.10

    # #******* functions *********

    # def corr_arr(S, X_ar):
    #     X = np.zeros((S.size, 3))
    #     for i in range(3):
    #         X[:,i] = np.array(X_ar[:,3+i])
    #     return X

    # def xcorr_arr(k, XX_ar, iso_num):
    #     XX = np.zeros((k.size, 4))
    #     j = 0
    #     for i in range(4):
    #         XX[:,i] = np.array(XX_ar[:,iso_num + j])
    #         j += 7 if i == 0 else 6
    #     return XX

    # #******* Reading *********

    # for i in range(int(inp_c_var_f), int(inp_c_var_l) + 1):
    #     if(inp_c_var):
    #         path_ini_output = re.sub(whichsim_n + r'[1-9]?[0-9]', whichsim_n + str(i), path_ini_output)
    #         path_output = re.sub(whichsim_n + r'[1-9]?[0-9]', whichsim_n + str(i), path_output)
    #     #  initial
    #     if(inp_ini):
    #         N_ar_ini = np.loadtxt(path_ini_output + "/corr_N" + run_ini + ".dat", dtype = 'float')
    #         R_ar_ini = np.loadtxt(path_ini_output + "/corr_R" + run_ini + ".dat", dtype = 'float')
    #         S_ini = np.array(N_ar_ini[:,1])
    #         N_ini = corr_arr(S_ini, N_ar_ini)
    #         R_ini = corr_arr(S_ini, R_ar_ini)
    #         if(inp_c_var):
    #             if(i == int(inp_c_var_f)):
    #                 N_ini_var = np.zeros((int(inp_c_var_l) + 1, N_ini.shape[0], N_ini.shape[1]))
    #                 R_ini_var = np.zeros((int(inp_c_var_l) + 1, R_ini.shape[0], R_ini.shape[1]))
    #             N_ini_var[i,:,:] = N_ini
    #             R_ini_var[i,:,:] = R_ini
    #     #  original
    #     if(inp_ori):
    #         N_ar_ori = np.loadtxt(path_output + "/corr_N" + run_ori + ".dat", dtype = 'float')
    #         R_ar_ori = np.loadtxt(path_output + "/corr_R" + run_ori + ".dat", dtype = 'float')
    #         S_ori = np.array(N_ar_ori[:,1])
    #         N_ori = corr_arr(S_ori, N_ar_ori)
    #         R_ori = corr_arr(S_ori, R_ar_ori)
    #         if(inp_xc_d):
    #             XD_ar_ori = np.loadtxt(path_output + "/xcorr_dense" + run_ori + ".dat", dtype = 'float')
    #             k_ori = np.array(XD_ar_ori[:,1])
    #             XD_n_ori = xcorr_arr(k_ori, XD_ar_ori, 3)   # numerator:   < fin*ini >
    #             XD_df_ori = xcorr_arr(k_ori, XD_ar_ori, 4)  # denominator: < fin^2 >
    #             XD_di_ori = xcorr_arr(k_ori, XD_ar_ori, 5)  # denominator: < ini^2 >
    #             XD_f_ori = xcorr_arr(k_ori, XD_ar_ori, 6)   # < fin*ini > / < fin^2 >
    #             XD_i_ori = xcorr_arr(k_ori, XD_ar_ori, 7)   # < fin*ini > / < ini^2 >
    #             XD_c_ori = np.array(XD_ar_ori[:,8])         # < fin*ini > / sq(< ini^2 >< fin^2 >)
    #         if(inp_c_var):
    #             if(i == int(inp_c_var_f)):
    #                 N_ori_var = np.zeros((int(inp_c_var_l) + 1, N_ori.shape[0], N_ori.shape[1]))
    #                 R_ori_var = np.zeros((int(inp_c_var_l) + 1, R_ori.shape[0], R_ori.shape[1]))
    #             N_ori_var[i,:,:] = N_ori
    #             R_ori_var[i,:,:] = R_ori
    #     #  standard
    #     if(inp_std):
    #         N_ar_std = np.loadtxt(path_output + "/corr_N" + run_std + ".dat", dtype = 'float')
    #         R_ar_std = np.loadtxt(path_output + "/corr_R" + run_std + ".dat", dtype = 'float')
    #         S_std = np.array(N_ar_std[:,1])
    #         N_std = (N_b**2.)*corr_arr(S_std, N_ar_std)
    #         R_std = corr_arr(S_std, R_ar_std)
    #         if(inp_xc_s):
    #             XS_ar_std = np.loadtxt(path_output + "/xcorr_shift" + run_std + ".dat", dtype = 'float')
    #             k_std = np.array(XS_ar_std[:,1])
    #             XS_n_std = xcorr_arr(k_std, XS_ar_std, 3)   # numerator:   < fin*ini >
    #             XS_df_std = xcorr_arr(k_std, XS_ar_std, 4)  # denominator: < fin^2 >
    #             XS_di_std = xcorr_arr(k_std, XS_ar_std, 5)  # denominator: < ini^2 >
    #             XS_f_std = xcorr_arr(k_std, XS_ar_std, 6)   # < fin*ini > / < fin^2 >
    #             XS_i_std = xcorr_arr(k_std, XS_ar_std, 7)   # < fin*ini > / < ini^2 >
    #             XS_c_std = np.array(XS_ar_std[:,8])         # < fin*ini > / sq(< ini^2 >< fin^2 >)
    #         if(inp_xc_d):
    #             XD_ar_std = np.loadtxt(path_output + "/xcorr_dense" + run_std + ".dat", dtype = 'float')
    #             k_std = np.array(XD_ar_std[:,1])
    #             XD_n_std = (N_b)*xcorr_arr(k_std, XD_ar_std, 3)   # numerator:   < fin*ini >
    #             XD_df_std = (N_b**2.)*xcorr_arr(k_std, XD_ar_std, 4)  # denominator: < fin^2 >
    #             XD_di_std = xcorr_arr(k_std, XD_ar_std, 5)  # denominator: < ini^2 >
    #             XD_f_std = (N_b**(-1.))*xcorr_arr(k_std, XD_ar_std, 6)   # < fin*ini > / < fin^2 >
    #             XD_i_std = (N_b)*xcorr_arr(k_std, XD_ar_std, 7)   # < fin*ini > / < ini^2 >
    #             XD_c_std = np.array(XD_ar_std[:,8])         # < fin*ini > / sq(< ini^2 >< fin^2 >)
    #         if(inp_c_var):
    #             if(i == int(inp_c_var_f)):
    #                 N_std_var = np.zeros((int(inp_c_var_l) + 1, N_std.shape[0], N_std.shape[1]))
    #                 R_std_var = np.zeros((int(inp_c_var_l) + 1, R_std.shape[0], R_std.shape[1]))
    #             N_std_var[i,:,:] = N_std
    #             R_std_var[i,:,:] = R_std
    #     #  iteration
    #     if(inp_ite):
    #         N_ar_ite = np.loadtxt(path_output + "/corr_N" + run_ite + ".dat", dtype = 'float')
    #         R_ar_ite = np.loadtxt(path_output + "/corr_R" + run_ite + ".dat", dtype = 'float')
    #         S_ite = np.array(N_ar_ite[:,1])
    #         N_ite = (N_b**2.)*corr_arr(S_ite, N_ar_ite)
    #         R_ite = corr_arr(S_ite, R_ar_ite)
    #         if(inp_xc_s):
    #             XS_ar_ite = np.loadtxt(path_output + "/xcorr_shift" + run_ite + ".dat", dtype = 'float')
    #             k_ite = np.array(XS_ar_ite[:,1])
    #             XS_n_ite = xcorr_arr(k_ite, XS_ar_ite, 3)   # numerator:   < fin*ini >
    #             XS_df_ite = xcorr_arr(k_ite, XS_ar_ite, 4)  # denominator: < fin^2 >
    #             XS_di_ite = xcorr_arr(k_ite, XS_ar_ite, 5)  # denominator: < ini^2 >
    #             XS_f_ite = xcorr_arr(k_ite, XS_ar_ite, 6)   # < fin*ini > / < fin^2 >
    #             XS_i_ite = xcorr_arr(k_ite, XS_ar_ite, 7)   # < fin*ini > / < ini^2 >
    #             XS_c_ite = np.array(XS_ar_ite[:,8])         # < fin*ini > / sq(< ini^2 >< fin^2 >)
    #         if(inp_xc_d):
    #             XD_ar_ite = np.loadtxt(path_output + "/xcorr_dense" + run_ite + ".dat", dtype = 'float')
    #             k_ite = np.array(XD_ar_ite[:,1])
    #             XD_n_ite = (N_b)*xcorr_arr(k_ite, XD_ar_ite, 3)   # numerator:   < fin*ini >
    #             XD_df_ite = (N_b**2.)*xcorr_arr(k_ite, XD_ar_ite, 4)  # denominator: < fin^2 >
    #             XD_di_ite = xcorr_arr(k_ite, XD_ar_ite, 5)  # denominator: < ini^2 >
    #             XD_f_ite = (N_b**(-1.))*xcorr_arr(k_ite, XD_ar_ite, 6)   # < fin*ini > / < fin^2 >
    #             XD_i_ite = (N_b)*xcorr_arr(k_ite, XD_ar_ite, 7)   # < fin*ini > / < ini^2 >
    #             XD_c_ite = np.array(XD_ar_ite[:,8])         # < fin*ini > / sq(< ini^2 >< fin^2 >)
    #         if(inp_c_var):
    #             if(i == int(inp_c_var_f)):
    #                 N_ite_var = np.zeros((int(inp_c_var_l) + 1, N_ite.shape[0], N_ite.shape[1]))
    #                 R_ite_var = np.zeros((int(inp_c_var_l) + 1, R_ite.shape[0], R_ite.shape[1]))
    #             N_ite_var[i,:,:] = N_ite
    #             R_ite_var[i,:,:] = R_ite
    #     if not(inp_c_var):
    #         break


# #******* Plot *********

    # #  plot N_0/R_0 and N_2/R_0
    # if(inp_c):
    #     x_min_c = 0
    #     x_max_c = 200
    #     if("0.15" in str(z_ave)):
    #         y_min_c = -10
    #         y_max_c = 50
    #     elif("0.5" in str(z_ave)):
    #         y_min_c = -10
    #         y_max_c = 30 if not(inp_c == "2") else 15
    #     else:
    #         y_min_c = -5
    #         y_max_c = 20
    #     if(inp_c == "s"):
    #         x_min_c = 50
    #         x_max_c = 160
    #         y_min_c = -5
    #         y_max_c = 25
    #     plt.figure(figsize=(13, 10), dpi=80)
    #     plt.xlim(xmin = x_min_c, xmax = x_max_c)
    #     plt.ylim(ymin = y_min_c, ymax = y_max_c)
    #     if(inp_ini):
    #         if(inp_c == "0"):
    #           plt.plot(S_ini, (S_ini**2.*(N_ini[:,0])/(R_ini[:,0])),"-.", color = "y", linewidth = 3.0, label = "Initial")
    #         if(inp_c == "2"):
    #           plt.plot(S_ini, (S_ini**2.*(N_ini[:,1])/(R_ini[:,0])),"-.", color = "y", linewidth = 3.0, label = "Initial")
    #         if(inp_c == "s"):
    #           plt.plot(S_ini, (S_ini**2.*(N_ini[:,0])/(R_ini[:,0])),"-.", color = "y", linewidth = 3.0, label = "Initial")
    #     if(inp_ori):
    #         if(inp_c == "0"):
    #           plt.plot(S_ori, (S_ori**2.*(N_ori[:,0])/(R_ori[:,0])),"--",color = "b", linewidth = 3.0, label = "Observed")
    #         if(inp_c == "2"):
    #           plt.plot(S_ori, (S_ori**2.*(N_ori[:,1])/(R_ori[:,0])),"--", color = "b", linewidth = 3.0, label = "Observed")
    #         if(inp_c == "s"):
    #           plt.plot(S_ori, (S_ori**2.*(N_ori[:,0])/(R_ori[:,0])),"--",color = "b", linewidth = 3.0, label = "pre-recon")
    #     if(inp_std):
    #         if(inp_c == "0"):
    #           plt.plot(S_std, (S_std**2.*(N_std[:,0])/(R_std[:,0])),"-",color = "g", linewidth = 3.0, label = "Standard rec")
    #         if(inp_c == "2"):
    #           plt.plot(S_std, (S_std**2.*(N_std[:,1])/(R_std[:,0])),"-", color = "g", linewidth = 3.0, label = "Standard rec")
    #         if(inp_c == "s"):
    #           plt.plot(S_std, (S_std**2.*(N_std[:,0])/(R_std[:,0])),"-",color = "g", linewidth = 3.0, label = "post-recon")
    #     if(inp_ite):
    #         if(inp_c == "0"):
    #           plt.plot(S_ite, (S_ite**2.*(N_ite[:,0])/(R_ite[:,0])),"-",color = "r", linewidth = 3.0, label = "Iterative rec")
    #         if(inp_c == "2"):
    #           plt.plot(S_ite, (S_ite**2.*(N_ite[:,1])/(R_ite[:,0])),"-", color = "r", linewidth = 3.0, label = "Iterative rec")
    #         if(inp_c == "s"):
    #           plt.plot(S_ite, (S_ite**2.*(N_ite[:,0])/(R_ite[:,0])),"-",color = "r", linewidth = 3.0, label = "Iterative rec")
    #     path_fig_c = path_output.replace("/file", "") + "/corr_" + inp_c
    #     path_fig_c += run_ini if inp_ini else ""
    #     path_fig_c += run_ori if inp_ori else ""
    #     path_fig_c += run_std if inp_std else ""
    #     path_fig_c += run_ite if inp_ite else ""
    #     path_fig_c += ".png"
    #     plt.legend(fontsize=30)
    #     plt.tick_params(labelsize=30)
    #     plt.xlabel(r'$S\ [{\rm Mpc}/h]$',fontsize=30)
    #     plt.ylabel(r'$S^2 \xi_{l}\  [{\rm Mpc}/h]^{2}$',fontsize=30)
    #     plt.title(r'$\xi_{l} = \mathcal{N}_{l}/\mathcal{R}_{0}$',fontsize=30)
    #     if(inp_c):
    #         plt.savefig(path_fig_c)

    # #  plot x-corr for shift
    # if(inp_xc_s):
    #     x_min_xc_s = 0.01
    #     x_max_xc_s = 0.5
    #     y_min_xc_s = 0
    #     y_max_xc_s = 1.2
    #     plt.figure(figsize=(13, 10), dpi=80)
    #     plt.xlim(xmin = x_min_xc_s, xmax = x_max_xc_s)
    #     plt.xscale("log")
    #     plt.ylim(ymin = y_min_xc_s, ymax = y_max_xc_s)
    #     if(inp_xc_s == "f"): # < fin*tru > / < fin^2 >
    #         type_xc_s = r'$<S_{\rm rec} S_{\rm tru}>/<S_{\rm rec}^2>$'
    #         if inp_std:
    #             plt.plot(k_std, XS_f_std[:,0],"-", color = "g", linewidth = 3.0, label = "Standard rec")
    #         if inp_ite:
    #             plt.plot(k_ite, XS_f_ite[:,0],"-", color = "r", linewidth = 3.0, label = "Iterative rec")
    #     if(inp_xc_s == "fw"):
    #         type_xc_s = r'$<S_{\rm rec} S_{\rm tru}>/<S_{\rm rec}^2>$'
    #         if inp_std:
    #             plt.plot(k_std, XS_f_std[:,1],".",label = "w1, Standard rec")
    #             plt.plot(k_std, XS_f_std[:,2],".",label = "w2")
    #             plt.plot(k_std, XS_f_std[:,3],".",label = "w3")
    #         if inp_ite:
    #             plt.plot(k_ite, XS_f_ite[:,1],".",label = "w1, Iterative rec")
    #             plt.plot(k_ite, XS_f_ite[:,2],".",label = "w2")
    #             plt.plot(k_ite, XS_f_ite[:,3],".",label = "w3")
    #     if(inp_xc_s == "i"): # < fin*tru > / < tru^2 >
    #         type_xc_s = r'$<S_{\rm rec} S_{\rm tru}>/<S_{\rm tru}^2>$'
    #         if inp_std:
    #             plt.plot(k_std, XS_i_std[:,0],"-", color = "g", linewidth = 3.0, label = "Standard rec")
    #         if inp_ite:
    #             plt.plot(k_ite, XS_i_ite[:,0],"-", color = "r", linewidth = 3.0, label = "Iterative rec")
    #     if(inp_xc_s == "iw"):
    #         type_xc_s = r'$<S_{\rm rec} S_{\rm tru}>/<S_{\rm tru}^2>$'
    #         if inp_std:
    #             plt.plot(k_std, XS_i_std[:,1],".",label = "w1, Standard rec")
    #             plt.plot(k_std, XS_i_std[:,2],".",label = "w2")
    #             plt.plot(k_std, XS_i_std[:,3],".",label = "w3")
    #         if inp_ite:
    #             plt.plot(k_ite, XS_i_ite[:,1],".",label = "w1, Iterative rec")
    #             plt.plot(k_ite, XS_i_ite[:,2],".",label = "w2")
    #             plt.plot(k_ite, XS_i_ite[:,3],".",label = "w3")
    #     if(inp_xc_s == "c"): # < fin*tru > / sq(< tru^2 >< fin^2 >)
    #         type_xc_s = r'$<S_{\rm rec} S_{\rm tru}>/<S_{\rm rec}><S_{\rm tru}>$'
    #         if inp_std:
    #             plt.plot(k_std, XS_c_std,"-", color = "g", linewidth = 3.0, label = "Standard rec")
    #         if inp_ite:
    #             plt.plot(k_ite, XS_c_ite,"-", color = "r", linewidth = 3.0, label = "Iterative rec")
    #     if(inp_xc_s == "cw"):
    #         type_xc_s = r'$<S_{\rm rec} S_{\rm tru}>/<S_{\rm rec}><S_{\rm tru}>$'
    #         if inp_std:
    #             plt.plot(k_std, XS_n_std[:,1]/np.sqrt(XS_df_std[:,1]*XS_di_std[:,1]),".",label = "w1, Standard rec")
    #             plt.plot(k_std, XS_n_std[:,2]/np.sqrt(XS_df_std[:,2]*XS_di_std[:,2]),".",label = "w2")
    #             plt.plot(k_std, XS_n_std[:,3]/np.sqrt(XS_df_std[:,3]*XS_di_std[:,3]),".",label = "w3")
    #         if inp_ite:
    #             plt.plot(k_ite, XS_n_ite[:,1]/np.sqrt(XS_df_ite[:,1]*XS_di_ite[:,1]),".",label = "w1, Iterative rec")
    #             plt.plot(k_ite, XS_n_ite[:,2]/np.sqrt(XS_df_ite[:,2]*XS_di_ite[:,2]),".",label = "w2")
    #             plt.plot(k_ite, XS_n_ite[:,3]/np.sqrt(XS_df_ite[:,3]*XS_di_ite[:,3]),".",label = "w3")
    #     path_fig_xc_s = path_output.replace("/file", "") + "/x-corr_shift_" + inp_xc_s
    #     path_fig_xc_s += run_std if inp_std else ""
    #     path_fig_xc_s += run_ite if inp_ite else ""
    #     path_fig_xc_s += ".png"
    #     plt.grid()
    #     plt.legend(fontsize=30)
    #     plt.tick_params(labelsize=30)
    #     plt.xlabel(r'$k\ [h/{\rm Mpc}]$',fontsize=30)
    #     plt.ylabel(r'$r(k)$',fontsize=30)
    #     plt.title(r'Cross correlation for shift: ' + type_xc_s,fontsize=30)
    #     if(inp_xc_s):
    #         plt.savefig(path_fig_xc_s)

    # #  plot x-corr for density
    # if(inp_xc_d):
    #     x_min_xc_d = 0.01
    #     x_max_xc_d = 1.0
    #     y_min_xc_d = 0.0
    #     y_max_xc_d = 1.2
    #     plt.figure(figsize=(13, 10), dpi=80)
    #     plt.xlim(xmin = x_min_xc_d, xmax = x_max_xc_d)
    #     plt.xscale("log")
    #     plt.ylim(ymin = y_min_xc_d, ymax = y_max_xc_d)
    #     if(inp_xc_d == "f"): # < fin*ini > / < fin^2 >
    #         type_xc_d = r'$<\delta_{\rm rec} \delta_{\rm ini}>/<\delta_{\rm rec}^2>$'
    #         if inp_ori:
    #             plt.plot(k_ori, XD_f_ori[:,0],"-.", color = "b", linewidth = 3.0, label = "Observed")
    #         if inp_std:
    #             plt.plot(k_std, XD_f_std[:,0],"--", color = "g", linewidth = 3.0, label = "Standard rec")
    #         if inp_ite:
    #             plt.plot(k_ite, XD_f_ite[:,0],"-", color = "r", linewidth = 3.0, label = "Iterative rec")
    #     if(inp_xc_d == "fw"):
    #         type_xc_d = r'$<\delta_{\rm rec} \delta_{\rm ini}>/<\delta_{\rm rec}^2>$'
    #         if inp_ori:
    #             plt.plot(k_ori, XD_f_ori[:,1],".",label = "w1, Observed")
    #             plt.plot(k_ori, XD_f_ori[:,2],".",label = "w2")
    #             plt.plot(k_ori, XD_f_ori[:,3],".",label = "w3")
    #         if inp_std:
    #             plt.plot(k_std, XD_f_std[:,1],".",label = "w1, Standard rec")
    #             plt.plot(k_std, XD_f_std[:,2],".",label = "w2")
    #             plt.plot(k_std, XD_f_std[:,3],".",label = "w3")
    #         if inp_ite:
    #             plt.plot(k_ite, XD_f_ite[:,1],".",label = "w1, Iterative rec")
    #             plt.plot(k_ite, XD_f_ite[:,2],".",label = "w2")
    #             plt.plot(k_ite, XD_f_ite[:,3],".",label = "w3")
    #     if(inp_xc_d == "i"): # < fin*ini > / < ini^2 >
    #         type_xc_d = r'$<\delta_{\rm rec} \delta_{\rm ini}>/<\delta_{\rm ini}^2>$'
    #         if inp_ori:
    #             plt.plot(k_ori, XD_i_ori[:,0],"-.", color = "b", linewidth = 3.0, label = "Observed")
    #             # estimate bias
    #             if not(RSD):
    #                 k_ul = 0.0
    #                 i_min = 0
    #                 i_max = 0
    #                 #k_cr = 0.02
    #                 #Dk = 0.01
    #                 k_cr = 0.1
    #                 Dk = 0.05
    #                 while k_ul < k_cr + Dk:
    #                     k_ul = k_ori[i_max]
    #                     if k_ul < k_cr - Dk:
    #                         i_min += 1
    #                     i_max += 1
    #                 if i_min == 0:
    #                     i_min = 1
    #                 B = XD_i_ori[i_min:i_max,0]
    #                 b = np.average(B)
    #                 if(typeobject == 0):  # for matter
    #                     print("# b_cr = " + str(b))
    #                 #    inp_b_cr = input("# Update bias_cr ? >>") if inp_run else ""
    #                 #    if inp_b_cr:
    #                 #        f = open(bias_cr_f,'w')
    #                 #        f.write(str(b) + "\n")
    #                 #        f.close()
    #                 else:  # for halo and galaxy
    #                     with open(bias_cr_f,'r') as f:
    #                         for bias_cr_r in f:
    #                             b_cr = float(bias_cr_r)
    #                     if not "{0:.6f}".format(b) == "{0:.6f}".format(b_cr) :
    #                         print("#\n# b = " + str(b))
    #                         print("# b_cr = " + str(b_cr))
    #                         inp_b = input("# Update bias ? >>") if inp_run else ""
    #                         if inp_b:
    #                             f = open(bias_f,'w')
    #                             f.write(str(bias*b/b_cr) + "\n")
    #                             f.close()
    #                     print("#\n# bias = " + str(bias*b/b_cr))
    #         if inp_std:
    #             plt.plot(k_std, XD_i_std[:,0],"--", color = "g", linewidth = 3.0, label = "Standard rec")
    #         if inp_ite:
    #             plt.plot(k_ite, XD_i_ite[:,0],"-", color = "r", linewidth = 3.0, label = "Iterative rec")
    #     if(inp_xc_d == "iw"):
    #         type_xc_d = r'$<\delta_{\rm rec} \delta_{\rm ini}>/<\delta_{\rm ini}^2>$'
    #         if inp_ori:
    #             plt.plot(k_ori, XD_i_ori[:,1],"-.", linewidth = 3.0, label = "w1, Observed")
    #             plt.plot(k_ori, XD_i_ori[:,2],"-.", linewidth = 3.0, label = "w2")
    #             plt.plot(k_ori, XD_i_ori[:,3],"-.", linewidth = 3.0, label = "w3")
    #         if inp_std:
    #             plt.plot(k_std, XD_i_std[:,1],"--", linewidth = 3.0, label = "w1, Standard rec")
    #             plt.plot(k_std, XD_i_std[:,2],"--", linewidth = 3.0, label = "w2")
    #             plt.plot(k_std, XD_i_std[:,3],"--", linewidth = 3.0, label = "w3")
    #         if inp_ite:
    #             plt.plot(k_ite, XD_i_ite[:,1],"-", linewidth = 3.0, label = r'$k_{z}/k > 2/3$')
    #             plt.plot(k_ite, XD_i_ite[:,2],"-", linewidth = 3.0, label = r'$2/3 > k_{z}/k > 1/3$')
    #             plt.plot(k_ite, XD_i_ite[:,3],"-", linewidth = 3.0, label = r'$1/3 > k_{z}/k$')
    #     if(inp_xc_d == "c"): # < fin*ini > / sq(< ini^2 >< fin^2 >)
    #         type_xc_d = r'$<\delta_{\rm rec} \delta_{\rm ini}>/<\delta_{\rm rec}><\delta_{\rm ini}>$'
    #         if inp_ori:
    #             plt.plot(k_ori, XD_c_ori,"-.", color = "b", linewidth = 3.0, label = "Observed")
    #         if inp_std:
    #             plt.plot(k_std, XD_c_std,"--", color = "g", linewidth = 3.0, label = "Standard rec")
    #         if inp_ite:
    #             plt.plot(k_ite, XD_c_ite,"-", color = "r", linewidth = 3.0, label = "Iterative rec")
    #     if(inp_xc_d == "cw"):
    #         type_xc_d = r'$<\delta_{\rm rec} \delta_{\rm ini}>/<\delta_{\rm rec}><\delta_{\rm ini}>$'
    #         if inp_ori:
    #             plt.plot(k_ori, XD_n_ori[:,1]/np.sqrt(XD_df_ori[:,1]*XD_di_ori[:,1]),"-.", linewidth = 3.0, label = "w1, Observed")
    #             plt.plot(k_ori, XD_n_ori[:,2]/np.sqrt(XD_df_ori[:,2]*XD_di_ori[:,2]),"-.", linewidth = 3.0, label = "w2")
    #             plt.plot(k_ori, XD_n_ori[:,3]/np.sqrt(XD_df_ori[:,3]*XD_di_ori[:,3]),"-.", linewidth = 3.0, label = "w3")
    #         if inp_std:
    #             plt.plot(k_std, XD_n_std[:,1]/np.sqrt(XD_df_std[:,1]*XD_di_std[:,1]),"--", linewidth = 3.0, label = "w1, Standard rec")
    #             plt.plot(k_std, XD_n_std[:,2]/np.sqrt(XD_df_std[:,2]*XD_di_std[:,2]),"--", linewidth = 3.0, label = "w2")
    #             plt.plot(k_std, XD_n_std[:,3]/np.sqrt(XD_df_std[:,3]*XD_di_std[:,3]),"--", linewidth = 3.0, label = "w3")
    #         if inp_ite:
    #             plt.plot(k_ite, XD_n_ite[:,1]/np.sqrt(XD_df_ite[:,1]*XD_di_ite[:,1]),"-", linewidth = 3.0, label = r'$k_{z}/k > 2/3$')
    #             plt.plot(k_ite, XD_n_ite[:,2]/np.sqrt(XD_df_ite[:,2]*XD_di_ite[:,2]),"-", linewidth = 3.0, label = r'$2/3 > k_{z}/k > 1/3$')
    #             plt.plot(k_ite, XD_n_ite[:,3]/np.sqrt(XD_df_ite[:,3]*XD_di_ite[:,3]),"-", linewidth = 3.0, label = r'$1/3 > k_{z}/k$')
    #     path_fig_xc_d = path_output.replace("/file", "") + "/x-corr_dens_" + inp_xc_d
    #     path_fig_xc_d += run_ori if inp_ori else ""
    #     path_fig_xc_d += run_std if inp_std else ""
    #     path_fig_xc_d += run_ite if inp_ite else ""
    #     path_fig_xc_d += ".png"
    #     plt.grid()
    #     plt.legend(loc="lower left", fontsize=30)
    #     plt.tick_params(labelsize=30)
    #     plt.xlabel(r'$k\ [h/{\rm Mpc}]$',fontsize=30)
    #     plt.ylabel(r'$r(k)$',fontsize=30)
    #     plt.title(r'Cross correlation for density: ' + type_xc_d,fontsize=30)
    #     if(inp_xc_d):
    #         plt.savefig(path_fig_xc_d)


    # #  plot variance for N_0/R_0 and N_2/R_0
    # if(inp_c_var):
    #     x_min_c_var = 0
    #     x_max_c_var = 150
    #     if(typeobject == 0):
    #         y_min_c_var = -5
    #         #y_max_c_var = 15
    #         y_max_c_var = 7
    #     else:
    #         y_min_c_var = -5
    #         y_max_c_var = 7
    #     plt.figure(figsize=(13, 18), dpi=80)
    #     plt.subplot(2, 1, 1) # monopole
    #     plt.xlim(xmin = x_min_c_var, xmax = x_max_c_var)
    #     plt.ylim(ymin = y_min_c_var, ymax = y_max_c_var)
    #     if(inp_ori):
    #         var_ori_0_av = 0.
    #         for i in range(int(inp_c_var_f), int(inp_c_var_l) + 1):
    #             var_ori_0_av += (S_ori**2.*(N_ori_var[i,:,0])/(R_ori_var[i,:,0]) - S_ini**2.*(N_ini_var[i,:,0])/(R_ini_var[i,:,0]))
    #         var_ori_0_av /= (int(inp_c_var_l)-int(inp_c_var_f)+1)
    #         var_ori_0_v_sq = 0.
    #         for i in range(int(inp_c_var_f), int(inp_c_var_l) + 1):
    #             var_ori_0_v_sq += ((S_ori**2.*(N_ori_var[i,:,0])/(R_ori_var[i,:,0]) - S_ini**2.*(N_ini_var[i,:,0])/(R_ini_var[i,:,0])) - var_ori_0_av)**2.    
    #         var_ori_0_v = np.sqrt(var_ori_0_v_sq/(int(inp_c_var_l)-int(inp_c_var_f)+1))
    #         if(inp_c_var == "v"):
    #             #plt.errorbar(S_ori, var_ori_0_av, yerr = var_ori_0_v, fmt='bo', ecolor='b')
    #             plt.fill_between(S_ori, var_ori_0_av - var_ori_0_v, var_ori_0_av + var_ori_0_v, facecolor='lightblue', alpha=0.6)
    #             plt.plot(S_ori, var_ori_0_av, "-", linewidth = 3.0, color = "b",label = "Observed")
    #         else:
    #             plt.plot(S_ori, var_ori_0_av, "-", linewidth = 3.0, color = "b",label = "Observed")
    #     if(inp_std):
    #         var_std_0_av = 0.
    #         for i in range(int(inp_c_var_f), int(inp_c_var_l) + 1):
    #             var_std_0_av += (S_std**2.*(N_std_var[i,:,0])/(R_std_var[i,:,0]) - S_ini**2.*(N_ini_var[i,:,0])/(R_ini_var[i,:,0]))
    #         var_std_0_av /= (int(inp_c_var_l)-int(inp_c_var_f)+1)
    #         var_std_0_v_sq = 0.
    #         for i in range(int(inp_c_var_f), int(inp_c_var_l) + 1):
    #             var_std_0_v_sq += ((S_std**2.*(N_std_var[i,:,0])/(R_std_var[i,:,0]) - S_ini**2.*(N_ini_var[i,:,0])/(R_ini_var[i,:,0])) - var_std_0_av)**2.
    #         var_std_0_v = np.sqrt(var_std_0_v_sq/(int(inp_c_var_l)-int(inp_c_var_f)+1))
    #         if(inp_c_var == "v"):
    #             #plt.errorbar(S_std, var_std_0_av, yerr = var_std_0_v, fmt='go', ecolor='g')
    #             plt.fill_between(S_std, var_std_0_av - var_std_0_v, var_std_0_av + var_std_0_v, facecolor='lightgreen', alpha=0.6)
    #             plt.plot(S_std, var_std_0_av, "-", linewidth = 3.0, color = "g",label = "Standard rec")
    #         else:
    #             plt.plot(S_std, var_std_0_av, "-", linewidth = 3.0, color = "g",label = "Standard rec")
    #     if(inp_ite):
    #         var_ite_0_av = 0.
    #         for i in range(int(inp_c_var_f), int(inp_c_var_l) + 1):
    #             var_ite_0_av += (S_ite**2.*(N_ite_var[i,:,0])/(R_ite_var[i,:,0]) - S_ini**2.*(N_ini_var[i,:,0])/(R_ini_var[i,:,0]))
    #         var_ite_0_av /= (int(inp_c_var_l)-int(inp_c_var_f)+1)
    #         var_ite_0_v_sq = 0.
    #         for i in range(int(inp_c_var_f), int(inp_c_var_l) + 1):
    #             var_ite_0_v_sq += ((S_ite**2.*(N_ite_var[i,:,0])/(R_ite_var[i,:,0]) - S_ini**2.*(N_ini_var[i,:,0])/(R_ini_var[i,:,0])) - var_ite_0_av)**2.
    #         var_ite_0_v = np.sqrt(var_ite_0_v_sq/(int(inp_c_var_l)-int(inp_c_var_f)+1))
    #         if(inp_c_var == "v"):
    #             #plt.errorbar(S_ite, var_ite_0_av, yerr = var_ite_0_v, fmt='ro', ecolor='r')
    #             plt.fill_between(S_std, var_ite_0_av - var_ite_0_v, var_ite_0_av + var_ite_0_v, facecolor='lightpink', alpha=0.6)
    #             plt.plot(S_ite, var_ite_0_av, "-", linewidth = 3.0, color = "r",label = "Iterative rec")
    #         else:
    #             plt.plot(S_ite, var_ite_0_av, "-", linewidth = 3.0, color = "r",label = "Iterative rec")
    #     plt.grid() 
    #     plt.legend(fontsize=25)
    #     plt.tick_params(labelsize=30)
    #     plt.ylabel(r'diff. for $l = 0$',fontsize=30)
    #     plt.title(r'Difference with the initial',fontsize=30)
    #     if(typeobject == 0):
    #         y_min_c_var = -10
    #         #y_max_c_var = 30
    #         y_max_c_var = 12
    #     else:
    #         y_min_c_var = -10
    #         y_max_c_var = 12 
    #     plt.subplot(2, 1, 2) # quadrupole
    #     plt.xlim(xmin = x_min_c_var, xmax = x_max_c_var)
    #     plt.ylim(ymin = y_min_c_var, ymax = y_max_c_var)
    #     if(inp_ori):
    #         var_ori_2_av = 0.
    #         for i in range(int(inp_c_var_f), int(inp_c_var_l) + 1):
    #             var_ori_2_av += (S_ori**2.*(N_ori_var[i,:,1])/(R_ori_var[i,:,0]) - S_ini**2.*(N_ini_var[i,:,1])/(R_ini_var[i,:,0]))
    #         var_ori_2_av /= (int(inp_c_var_l)-int(inp_c_var_f)+1)
    #         var_ori_2_v_sq = 0.
    #         for i in range(int(inp_c_var_f), int(inp_c_var_l) + 1):
    #             var_ori_2_v_sq += ((S_ori**2.*(N_ori_var[i,:,1])/(R_ori_var[i,:,0]) - S_ini**2.*(N_ini_var[i,:,1])/(R_ini_var[i,:,0])) - var_ori_2_av)**2.    
    #         var_ori_2_v = np.sqrt(var_ori_2_v_sq/(int(inp_c_var_l)-int(inp_c_var_f)+1))
    #         if(inp_c_var == "v"):
    #             #plt.errorbar(S_ori, var_ori_2_av, yerr = var_ori_2_v, fmt='bo', ecolor='b')
    #             plt.fill_between(S_ori, var_ori_2_av - var_ori_2_v, var_ori_2_av + var_ori_2_v, facecolor='lightblue', alpha=0.5)
    #             plt.plot(S_ori, var_ori_2_av, "-", linewidth = 3.0, color = "b")
    #         else:
    #             plt.plot(S_ori, var_ori_2_av, "-", linewidth = 3.0, color = "b")
    #     if(inp_std):
    #         var_std_2_av = 0.
    #         for i in range(int(inp_c_var_f), int(inp_c_var_l) + 1):
    #             var_std_2_av += (S_std**2.*(N_std_var[i,:,1])/(R_std_var[i,:,0]) - S_ini**2.*(N_ini_var[i,:,1])/(R_ini_var[i,:,0]))
    #         var_std_2_av /= (int(inp_c_var_l)-int(inp_c_var_f)+1)
    #         var_std_2_v_sq = 0.
    #         for i in range(int(inp_c_var_f), int(inp_c_var_l) + 1):
    #             var_std_2_v_sq += ((S_std**2.*(N_std_var[i,:,1])/(R_std_var[i,:,0]) - S_ini**2.*(N_ini_var[i,:,1])/(R_ini_var[i,:,0])) - var_std_2_av)**2.
    #         var_std_2_v = np.sqrt(var_std_2_v_sq/(int(inp_c_var_l)-int(inp_c_var_f)+1))
    #         if(inp_c_var == "v"):
    #             #plt.errorbar(S_std, var_std_2_av, yerr = var_std_2_v, fmt='go', ecolor='g')
    #             plt.fill_between(S_std, var_std_2_av - var_std_2_v, var_std_2_av + var_std_2_v, facecolor='lightgreen', alpha=0.5)
    #             plt.plot(S_std, var_std_2_av, "-", linewidth = 3.0, color = "g")
    #         else:
    #             plt.plot(S_std, var_std_2_av, "-", linewidth = 3.0, color = "g")
    #     if(inp_ite):
    #         var_ite_2_av = 0.
    #         for i in range(int(inp_c_var_f), int(inp_c_var_l) + 1):
    #             var_ite_2_av += (S_ite**2.*(N_ite_var[i,:,1])/(R_ite_var[i,:,0]) - S_ini**2.*(N_ini_var[i,:,1])/(R_ini_var[i,:,0]))
    #         var_ite_2_av /= (int(inp_c_var_l)-int(inp_c_var_f)+1)
    #         var_ite_2_v_sq = 0.
    #         for i in range(int(inp_c_var_f), int(inp_c_var_l) + 1):
    #             var_ite_2_v_sq += ((S_ite**2.*(N_ite_var[i,:,1])/(R_ite_var[i,:,0]) - S_ini**2.*(N_ini_var[i,:,1])/(R_ini_var[i,:,0])) - var_ite_2_av)**2.
    #         var_ite_2_v = np.sqrt(var_ite_2_v_sq/(int(inp_c_var_l)-int(inp_c_var_f)+1))
    #         if(inp_c_var == "v"):
    #             #plt.errorbar(S_ite, var_ite_2_av, yerr = var_ite_2_v, fmt='ro', ecolor='r')
    #             plt.fill_between(S_std, var_ite_2_av - var_ite_2_v, var_ite_2_av + var_ite_2_v, facecolor='lightpink', alpha=0.5)
    #             plt.plot(S_ite, var_ite_2_av, "-", linewidth = 3.0, color = "r")
    #         else:
    #             plt.plot(S_ite, var_ite_2_av, "-", linewidth = 3.0, color = "r")
    #     plt.grid()
    #     plt.legend(fontsize=25)
    #     plt.tick_params(labelsize=30)
    #     plt.xlabel(r'$S\ [{\rm Mpc}/h]$',fontsize=30)
    #     plt.ylabel(r'diff. for $l = 2$',fontsize=30)
    #     path_fig_c_var = path_output.replace("/file", "") + "/var_corr_" + inp_c_var + "_" + inp_c_var_f + "-" + inp_c_var_l
    #     path_fig_c_var += run_ori if inp_ori else ""
    #     path_fig_c_var += run_std if inp_std else ""
    #     path_fig_c_var += run_ite if inp_ite else ""
    #     #path_fig_c_var += "_no_divide"
    #     path_fig_c_var += "_std_v"
    #     path_fig_c_var += ".png"
    #     if(inp_c_var):
    #         plt.savefig(path_fig_c_var)
