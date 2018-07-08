# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 17:30:51 2017

@author: givoltage
"""
from __future__ import absolute_import, division, print_function, unicode_literals

# import Halotools as ht # this is the Halotools for Abacus which imports catalogues in a unified format, not astropy.halotools
import numpy as np
import os
import pickle
from Corrfunc.theory.DDrppi import DDrppi
# http://corrfunc.readthedocs.io/en/master/api/Corrfunc.theory.html#module-Corrfunc.theory.DDrppi

save_dir = '/home/dyt/analysis_data/'
#%% parameters
# sim_name_prefix = 'emulator_720box_planck'
# cosmologies = [0]
# redshifts = [0.100, 0.300]
# products_dir = r'/mnt/gosling2/bigsim_products/emulator_720box_planck_products'
# phases = 'all'
# halo_type = 'Rockstar'

#%%

# cats = ht.make_catalogs(sim_name = sim_name_prefix,
#                         cosmologies = cosmologies, # 'all'
#                         redshifts = redshifts, # 'all'
#                         products_dir = products_dir,
#                         phases = phases, # 'all'
#                         halo_type = halo_type,
#                         load_halo_ptcl_catalog = False, # this loads subsamples, which does not work
#                         load_ptcl_catalog = False, # this loads uniform subsamples, which is not implemented
#                         load_pids = 'auto'
#                         )
#%%

def baofit_data(cats, sim_name_prefix, n_rp_bins = 8, n_pi_bins = 12, num_threads = 10):
    
    ''' use with only one cosmology at a time'''
    
    n_redshifts = len(cats)
    n_phases = len(cats[0][0])
    xi_cats = np.empty((n_rp_bins-1, n_pi_bins-1, n_redshifts, n_phases))
    xi_cov_cats = np.empty(((n_rp_bins-1)*(n_pi_bins-1), (n_rp_bins-1)*(n_pi_bins-1), n_redshifts, n_phases))
    
    l=0
    for m in range(n_phases):

        try:
            sim_name = cats[0][0][m].SimName
        except:
            sim_name = cats[0][0][m].simname
            
        for n in range(len(cats[0])):
            for k in range(n_redshifts):
                l = l + 1
                print('Processing catalogue ' + str(l) + ' of ' + str(n_redshifts*n_phases))
                xi_cats[:, :, k, m], xi_cov_cats[:, :, k, m] = baofit_data_cat(
                        cats[k][n][m], n_rp_bins, n_pi_bins, num_threads = num_threads)
                # save baofit data file for a given phase and redshift3
                redshift = cats[k][n][m].redshift
                baofit_data = xi_cats[:, :, k, m]
                baofit_cov = xi_cov_cats[:, :, k, m]
                baofit_data_save = np.column_stack((np.arange(baofit_data.size), 
                                                    baofit_data.flatten(order = 'C')))
                iu = np.triu_indices(baofit_cov.shape[0])
                baofit_cov_save = np.column_stack(iu, baofit_cov[iu].flatten(order = 'C'))
                path_data = os.path.join(save_dir, sim_name+'-z'+str(redshift)+'.data')
                path_cov = os.path.join(save_dir, sim_name+'-z'+str(redshift)+'.cov')
                print('Saving: ' + path_data)
                np.savetxt(path_data, baofit_data_save, fmt = ['%.0f', '%.30f'], delimiter=' ')
                print('Saving: ' + path_cov)
                np.savetxt(path_cov, baofit_cov_save, fmt = ['%.0f', '%.0f', '%.30f'], delimiter=' ')

    # save all xi data together as pickle
    path_xi = os.path.join(save_dir, sim_name_prefix+'-xi.pkl')
    path_cov = os.path.join(save_dir, sim_name_prefix+'-cov.pkl')
    with open(path_xi, 'wb') as file:
        print('Saving: ' + path_xi)
        pickle.dump(xi_cats, file, protocol = pickle.HIGHEST_PROTOCOL)
    with open(path_cov, 'wb') as file:
        print('Saving: ' + path_cov)
        pickle.dump(xi_cov_cats, file, protocol = pickle.HIGHEST_PROTOCOL)

    return xi_cats, xi_cov_cats

def baofit_data_cat(cat, n_rp_bins, n_pi_bins, num_threads = 10):
#    try:
#        sim_name = cat.SimName
#    except:
#        sim_name = cat.simname
    try:
        L = cat.BoxSize
    except:
        L = cat.Lbox[0]
    # redshift = cat.redshift
    x = cat.halo_table['halo_x']
    y = cat.halo_table['halo_y']
    z = cat.halo_table['halo_z']
    pos = return_xyz_formatted_array(x, y, z, period = L)
    rp_bins = np.logspace(np.log10(80), np.log10(130), n_rp_bins) # perpendicular bins
    pi_bins = np.logspace(np.log10(80), np.log10(130), n_pi_bins) # parallel bins
#    rp_bin_centres = (rp_bins[1:] + rp_bins[:-1])/2
#    pi_bin_centres = (pi_bins[1:] + pi_bins[:-1])/2
    # define randoms
    n_ran = len(cat.halo_table) * 10 # 500 ideal?
    print('Preparing randoms...')
    xran = np.random.uniform(0, L, n_ran)
    yran = np.random.uniform(0, L, n_ran)
    zran = np.random.uniform(0, L, n_ran)
    randoms = return_xyz_formatted_array(xran, yran, zran, period = L)
    print('2pt jackknife with ' + str(num_threads) + ' threads...')
    xi, xi_cov = rp_pi_tpcf_jackknife(pos, randoms, rp_bins, pi_bins, Nsub=3, period=L, num_threads = num_threads)
    # save xi as .data file for baofit
#    baofit_data = np.column_stack((np.arange(xi.size), xi.flatten(order = 'c')))
#    path = os.path.join('/home/dyt/analysis_data/', sim_name+'_'+str(redshift)+'.data')
#    print 'saving: ' + path
#    np.savetxt(path, baofit_data, fmt = ['%.0f', '%.30f'], delimiter=' ')
    return xi, xi_cov

def baofit_cov(xi_cats, sim_name_prefix, bias = True):
    shape = xi_cats.shape
    bin_xi = xi_cats.reshape(shape[0]*shape[1]*shape[2], shape[3])
    cov_matrix = np.cov(bin_xi, bias = bias)
    indices = np.indices(cov_matrix.shape)
    baofit_cov = np.column_stack((indices[0].flatten(order = 'C'),
                                  indices[1].flatten(order = 'C'),
                                  cov_matrix.flatten(order = 'C')))
    path = os.path.join(save_dir, sim_name_prefix+'.cov')
    print('Saving: ' + path)
    np.savetxt(path, baofit_cov, fmt = ['%.0f', '%.0f', '%.30f'], delimiter=' ')
    return cov_matrix