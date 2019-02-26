'''
auto.py: Ryuichiro Hada, January 2018

This automizes the process inputting, reconstructing, computing correlations,
and plotting figures.

'''
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
# import numpy as np
import sys
import subprocess, os
# import matplotlib as mpl
# mpl.use('AGG')
# import matplotlib.pyplot as plt

make_recon = False
run_recon = True

# inp_make = ""
# inp_run = "y"

if make_recon:
    #******* Flags (for make) *********
    RECONST = "y"
    ITERATION = "y"
    RSD = ""                 # To take account of RSD in reconstruction
    
    XCORR_SHIFT = ""         # To get X-correlation for shift fields
    FROM_DENSITY = ""        # To get shift_t_X from the density field
    DIFF_SHIFT = ""          # To sum the square of difference between shift_t_X and reconstructed shift
    XCORR_DENSITY = ""       # To get X-correlation for density fields
    
    CHANGE_SM = "y"           # To change smoothing scale
    SECOND = ""               # To take account of 2nd order in reconstruction
    INITIAL = ""             # To compute the corr for the initial density correlation function


if run_recon:
    
    # input arguments for running reconstruction
    reconst = sys.argv[1]
    rsd = sys.argv[2]
    tag = sys.argv[3]
    model_name = sys.argv[4]
    save_dir = sys.argv[5]
    bias = float(sys.argv[6])  # 2.23 for gal, 1.00 for ptcl
    
    #******* Options *********
    #------ run -------
    ngridCube = 480
    ngrid = [ -1, -1, -1]
    maxell = 2
    
    sep = 150.0            # Default to max_sep from the file
    dsep = 5.00
    
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
    
    sig_sm = 15.0
    sig_sm_std = 15.0        # [Mpc/h], proper smoothing scale
    sig_sm_ite = 15.0        # [Mpc/h], initial smoothing scale

    if reconst == '0':
        RECONST = ""
        ITERATION = ""
        recon_tag = 'pre-recon'
    elif reconst == '1':
        RECONST = "y"
        ITERATION = ""
        sig_sm = sig_sm_std
        recon_tag = 'post-recon-std'
    elif reconst == '2':
        RECONST = "y"
        ITERATION = "y"
        sig_sm = sig_sm_ite
        recon_tag = 'post-recon-ite'
    
            
    
    divi_sm = 1.2          #  by which sig_sm is divided each time
    last_sm = 10.0         #  to which the division is performed
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
    
    # bias = 2.23            #  2.23 for galaxies, 1.0 for matter field by default
    bias_uncer = 0         #  uncertainty of bais 
                           #  -1 :   80 % 
                           #   0 :  100 %      
                           #   1 :  120 % 
    
    qperiodic = 1;
    
    #----- Simulation -----
    typesim = 0            # the type of Simulation
                           #  0 :  AbacusCosmos  (planck emulator, etc.)
                           #  1 :  Others        (BOSS, etc.)
                           #  2 :  ST            (T.Sunayama provided)
    boxsizesim = 1100      # if typesim = 0,  1100 or 720
    # phase = 1           # the number of realization
    
    typehalo = 0           # if typesim = 0 or 1,
                           #  0 :  FoF
                           #  1 :  rockstar
    
    #----- file  -----
    #  general
    # if sample == 'ptcl':
    #     typeobject = 0
    # elif sample == 'halo':
    #     typeobject = 1
    # elif sample == 'gal':
    #     typeobject = 2
    # typeobject = 2         # the type of object
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
    ratio_RtoD = -1         #  ratio of the number of Random and Data
    num_parcentage = 3     #  parcentage of particle used
    typeFKP = 0            # the type of estimator
                           #  0 : not FKP,  1 : FKP
    #  initial
    ratio_RtoD_ini = 0     #  ratio of the number of Random and Data
    num_parcentage_ini = 1 #  parcentage of particle used
    typeFKP_ini = 0        # the type of estimator
                           #  0 : not FKP,  1 : FKP
#----------------------

#if ((not typeobject == 0) or (not RECONST)) and (XCORR_SHIFT):
#    print("# XCORR_SHIFT works on reconstructed matter fields")
#    XCORR_SHIFT = ""


#******* Inputs *********

# inp_all_sim = input("# All of simulations >>")
# inp_make = ""
# inp_run = ""
# inp_a_f = phase
# inp_a_l = phase
# if(inp_all_sim):
#     inp_a_f  = input("#     first >>")
#     inp_a_l  = input("#     last  >>")
# if not(inp_all_sim):
#     inp_make_ori = input("# Make >>")
#     inp_run_ori  = input("# Run >>")
# inp_plot = input("# Plot >>")
#inp_c = ""
#inp_xc_s = ""
#inp_xc_d = ""
#inp_c_var = ""
#inp_all  = ""
#inp_ini  = ""
#inp_ori  = ""
#inp_std  = ""
#inp_ite  = ""
#inp_c_var_f = 0
#inp_c_var_l = 0

#******* Main *********

# for k in range(int(inp_a_f), int(inp_a_l) + 1):
# phase = k
# for j in range(2):
#     if(j == 0):
#         if (inp_all_sim or inp_make_ori):
#             inp_make = "y"
#         inp_run = ""
#     else:
#         print("\n# Now, phase = " + str(phase) + "\n")
#         inp_make = ""
#         if (inp_all_sim or inp_run_ori):         
#             

# #******* Path *********

# typehalo_n = "_rockstar" if typehalo else "_FoF"
# typeobject_n = "" if typeobject == 0 else "_halo" + "_Mc" + str("{0:.4f}".format(M_cut)) if typeobject == 1 \
#                             else "_gal" if index_sc == 0 else "_gal_du" if index_sc == 3 else "_gal_sc" + str("{0:.3f}".format(sc)) 
# typeFKP_n = "_F" if typeFKP else "_noF"
# typeFKP_ini_n = "_F" if typeFKP_ini else "_noF"
# uni_n = "_uni" if quniform else ""
# periodic_n = "_p" if qperiodic else ""
# shift_n = "_S" if qshift and typeobject == 0 else ""
# rsd_n = "_rsd" if RSD else ""

# if (typesim == 0):    # AbacusCosmos
#     path_to_sim = "/AbacusCosmos" \
#         + "/emulator_" + str(boxsizesim) + "box_planck_products" \
#         + "/emulator_" + str(boxsizesim) + "box_planck_00-" + str(phase) + "_products" \
#         + "/emulator_" + str(boxsizesim) + "box_planck_00-" + str(phase) + typehalo_n + "_halos"
#     phase_n = "_00-"
# elif (typesim == 1):  # Others
#     path_to_sim = "/Others" \
#         + "/BOSS_1600box_products" \
#         + "/BOSS_1600box_FoF_halos"
#     phase_n = ""
# elif (typesim == 2):  # ST
#     path_to_sim = "/ST"
#     phase_n = ""
# else:
#     path_to_sim = "fake"
#     print("# this typesim is not defined !!! ")

# path_me = "/mnt/store1/rhada"
# path_z = "/z" + "{:.3f}".format(z_ave)
# # input
# path_input = path_me + path_to_sim + path_z
# path_input_ini = path_me + path_to_sim + "/ini"
# # output
# typeobject_d = "/matter" if typeobject == 0 else "/halo_Mc" + str("{0:.4f}".format(M_cut)) if typeobject == 1 \
#                         else "/gal" if index_sc == 0 else "/gal_du" if index_sc == 3 else "/gal_sc" + str("{0:.3f}".format(sc)) 
# sample_f = "fin_" + str(ratio_RtoD) + "_" + str(num_parcentage) + typeFKP_n
# sample_i = "ini_" + str(ratio_RtoD_ini) + "_" + str(num_parcentage_ini) + typeFKP_ini_n
# sample_d = "/" + sample_i + "_" + sample_f + "_ng" + str(ngridCube) + uni_n + periodic_n
# sample_ini_d = "/" + sample_i + "_ng" + str(ngridCube) + uni_n + periodic_n
# run_d = "/sep" + str(sep) + "_dsep" + str(dsep) + "_kmax" + str(kmax) + "_dk" + str(dk)
# space_d = "/rsd" if RSD else "/real"
# path_output = "../graph" + path_to_sim + path_z + typeobject_d + sample_d + run_d + space_d + "/file"
# path_ini_output = "../graph" + path_to_sim + "/ini" + sample_ini_d + run_d + "/file"
# path_bias_cr = "../graph" + path_to_sim + path_z
# path_bias = "../graph" + path_to_sim + path_z + typeobject_d + sample_d + run_d 


# #******* filename *********

# typeFKP_spe = "" if typeFKP else "_nF"
# typeFKP_ini_spe = "" if typeFKP_ini else "_nF"

# sample_in_d = "/" + sample_f + "_ng" + str(ngridCube) + uni_n + periodic_n + shift_n
# sample_ini_in_d  = sample_ini_d

# input
if run_recon:
    infile = '/home/dyt/store/recon/temp/file_D-' + tag # infile  = path_input + typeobject_d + sample_in_d + space_d + "/file_D"
    infile2 = '/home/dyt/store/recon/temp/file_R-' + tag # infile2 = path_input + typeobject_d + sample_in_d + space_d + "/file_R"
    
    infile_ini = infile  # path_input_ini + sample_ini_in_d + "/file_D"
    infile_ini2 = infile2  # path_input_ini + sample_ini_in_d + "/file_R"

    shift_true_x = infile + "_S0"
    shift_true_y = infile + "_S1"
    shift_true_z = infile + "_S2"

# # -------------------------

# bias_n = "" if bias_uncer == 0 else "_b20m" if bias_uncer == -1 else "_b20p"
# run_ini = "_ini"
# run_ori = "_ori"
# run_std = "_std_sm" + str(sig_sm_std) + bias_n
# run_ite = "_ite_num" + str(ite_times) + "_smi" + str(sig_sm_ite) + "_div" + str(divi_sm) + "_sml" + str(last_sm) \
#        + "_Ca" + str(C_ani) + "_fe" + str(f_eff) \
#        + "_wi" + str(ite_weight_ini) + "_sn" + str(ite_switch) + "_ws" + str(ite_weight_2) + bias_n


# if(INITIAL):
#     path_output = path_ini_output
#     run_id  = run_ini
#     infile = infile_ini
#     infile2 = infile_ini2
#     # flags not needed
#     RSD = ""
#     XCORR_SHIFT = ""
#     XCORR_DENSITY = ""
#     RECONST = ""

if run_recon:
    # output
    outfile = os.path.join(save_dir, '{}-auto-fftcorr_result-{}-{}_hmpc.log'
                           .format(model_name, recon_tag, sig_sm))
    if os.path.isfile(outfile):
        os.remove(outfile)
    if not os.path.exists(os.path.dirname(outfile)):
        try:
            os.makedirs(os.path.dirname(outfile))
        except OSError:
            pass
    outfile_corr = os.path.join(save_dir, '{}-auto-fftcorr_N-{}-{}_hmpc.txt'
                                .format(model_name, recon_tag, sig_sm))
    outfile_corr2 = os.path.join(save_dir, '{}-auto-fftcorr_R-{}-{}_hmpc.txt'
                                 .format(model_name, recon_tag, sig_sm))
    
    path_output = '/home/dyt/store/recon/temp'
    outfile_xcorr_sh = path_output + "/xcorr_shift" + tag + ".dat"
    outfile_xcorr_de = path_output + "/xcorr_dense" + tag + ".dat"

# def file_exist(filename, inp):
#     inp_l = inp
#     file_path = os.path.dirname(filename)
#     if not os.path.exists(file_path):
#         os.makedirs(file_path)
#     elif os.path.exists(filename) and inp_l:
#         inp_l = input("# This file has already existed !! \n# Do you really want to run ?>>")
#     return inp_l
# inp_run = file_exist(outfile, inp_run)


# #******* Bias arrangemnet *********

# bias_cr_f = path_bias_cr + "/bias_cr" + "_kmax" + str(kmax) + "_dk" + str(dk) + ".dat"
# bias_f = path_bias + "/bias.dat"
# if not(INITIAL):
#     f = open(bias_f,'w')
#     f.write(str(bias) + "\n")
#     f.close()
#     with open(bias_f,'r') as f:
#         for bias_r in f:
#             bias = float(bias_r)
# if (bias_uncer == -1):
#     bias *= 0.80
# elif (bias_uncer == 1):
#     bias *= 1.20

if make_recon:
    #########  Make  ###########
    
    # creating flags
    
    def flag_create():
        FLAG = ""
        for k, v in globals().items():
            if v == "y":
                FLAG += " -D" + k
        return FLAG
    
    OMP = " -DOPENMP -DFFTSLAB -DSLAB"
    FLAGS = OMP + flag_create()
    print('make flags are: ' + FLAGS)
    
    cmd_make = ["make", "CXXFLAGS=-O3" + FLAGS]
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # if(inp_make):
    subprocess.call(["make", "clean"])
    #print(cmd_make)
    subprocess.call(cmd_make)
    
if run_recon:
    #########  Run  ###########
    
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
               "-inini", infile_ini, \
               "-inini2", infile_ini2, \
               "-inshifttruex", shift_true_x, \
               "-inshifttruey", shift_true_y, \
               "-inshifttruez", shift_true_z, \
               "-out", outfile, \
               "-outcorr", outfile_corr, \
               "-outcorr2", outfile_corr2, \
               "-outxcorrsh", outfile_xcorr_sh, \
               "-outxcorrde", outfile_xcorr_de \
               ]
    # print('outfile path', outfile)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cmd_run = ["./reconst_" + reconst + "_" + rsd] + OPTIONS
    print('Reconstruction command:', ' '.join(cmd_run))
    # if (not inp_make and inp_run):
    subprocess.call(cmd_run)
