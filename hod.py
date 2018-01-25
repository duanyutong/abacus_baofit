# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 18:30:13 2017

@author: givoltage
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import os
# import pickle
# from halotools.mock_observables import return_xyz_formatted_array, rp_pi_tpcf_jackknife
# from halotools.empirical_models import PrebuiltHodModelFactory
import Halotools as ht # this is the Halotools for Abacus which imports catalogues in a unified format, not astropy.halotools
# from halotools.sim_manager import FakeSim
from halotools.utils import add_halo_hostid
# from Corrfunc.theory.DDrppi import DDrppi
# from Corrfunc.utils import convert_3d_counts_to_cf
from abacus_baofit import ht_rp_pi_2pcf, cov_from_xi_list, xi_to_baofit_input, cov_to_baofit_input, populate_halocat
# import subprocess

# catalogue parameters
sim_name_prefix = 'emulator_1100box_planck'
products_dir = r'/mnt/gosling2/bigsim_products/emulator_1100box_planck_products/'
halo_type = 'Rockstar'
redshifts = [0.700] # one redshift at a time 'all'
cosmologies = [0] # use with only one cosmology at a time # 'all'
phases =  list(range(16)) #[0, 1] # list(range(16)) # 'all'

# HOD models
prebuilt_models = ['zheng07']
n_s_bins = 15
n_mu_bins = 20
num_threads = 10
save_dir = '/home/dyt/analysis_data/hod/'
# ['zheng07', 'leauthaud11', 'tinker13', 'hearin15', 'zu_mandelbaum15', 'zu_mandelbaum16', 'cacciato09']
n_rp_bins = 13# 21
n_pi_bins = 15 # 21
use_corrfunc = False
use_jackknife = False

#%%
def make_halocat(phase):

    halocats = ht.make_catalogs(
        sim_name = sim_name_prefix, products_dir = products_dir, 
        redshifts = redshifts, cosmologies = cosmologies, phases = [phase], 
        halo_type = halo_type,
        load_halo_ptcl_catalog = False, # this loads subsamples, does not work
        load_ptcl_catalog = False, # this loads uniform subsamples, not implemented
        load_pids = 'auto')
    
    return halocats[0]
   
def do_hod():
    xi_dict = {}
    for model_name in prebuilt_models:
        xi_dict[model_name] = [] # initialise list of correlation functions
    
    for redshift in redshifts:
        for phase in phases:
            halocats = ht.make_catalogs(
                sim_name = sim_name_prefix, products_dir = products_dir, 
                redshifts = redshifts, cosmologies = cosmologies, phases = [phase], 
                halo_type = halo_type,
                load_halo_ptcl_catalog = False, # this loads subsamples, does not work
                load_ptcl_catalog = False, # this loads uniform subsamples, not implemented
                load_pids = 'auto')
            
            halocat = halocats[0][0][0]
            # setting halocat properties to be compatibble with halotools
            try:
                sim_name = halocat.SimName
            except: 
                sim_name = halocat.simname
            try:
                L = halocat.BoxSize
            except:
                L = halocat.Lbox[0]
            halocat.halo_table['halo_mvir'] = halocat.halo_table['halo_m']
            halocat.halo_table['halo_rvir'] = halocat.halo_table['halo_r']
            add_halo_hostid(halocat.halo_table, delete_possibly_existing_column = True)
            # setting NFW concentration, klypin_rs is more stable
            # can be improved following https://arxiv.org/abs/1709.07099
            halocat.halo_table['halo_nfw_conc'] = (
                halocat.halo_table['halo_rvir'] / halocat.halo_table['halo_klypin_rs'])
            
            # populate halocat with galaxies
            for model_name in prebuilt_models:
                model = populate_halocat(halocat, model_name)
                print('Calculating correlation functions with {} threads...'.format(num_threads))
                xi = ht_rp_pi_2pcf(model, L, 
                                   n_rp_bins = n_rp_bins, n_pi_bins = n_pi_bins,
                                   num_threads = num_threads)
                xi_dict[model_name].append(xi)
                # save correlation functions to file formatted for baofit
                path_xi = os.path.join(save_dir, sim_name+'-z'+str(redshift)+'-'+model_name+'.data')
                xi_to_baofit_input(xi, path_xi)
    
            #    rp_bin_centres = (rp_bins[1:] + rp_bins[:-1])/2
            #    pi_bin_centres = (pi_bins[1:] + pi_bins[:-1])/2
    
    # after all boxes are done, calculate covariance matrix
    for model_name in prebuilt_models:
        # calculate covariance matrix from 16 boxes of different phases
        cov = cov_from_xi_list(xi_dict[model_name])
        print(cov.shape)
        try:
            np.linalg.cholesky(cov)
            print('Covariance matrix passes cholesky decomposition')
        except:
            print('Covariance matrix fails cholesky decomposition')
        # save covariance matrix to file formatted for baofit, same for all boxes
        path_cov = os.path.join(save_dir, sim_name[:27]+'z'+str(redshift)+'-'+model_name+'.cov')
        cov_to_baofit_input(cov, path_cov)
        
        # call baofit in command line
    #    baofit_process = subprocess.Popen([
    #            'baofit', '-i', '/home/dyt/analysis_scripts/hod_xi_bao.ini'
    #            '--output-prefix', ])
        
if __name__ == "__main__":
    
    for phase in phases:
        for model_name in prebuilt_models:
            # data processing
            halocat = make_halocat(phase)
            hod_model = populate_halocat(halocat)
            do_auto(hod_model)
            do_subscross(hod_model, 3)
            # calculate xi and cov
            do_coadd_xi(hod_model)
            do_cov(hod_model)
            # baofit continues in ipynb