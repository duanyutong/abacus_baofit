# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 17:30:51 2017

@author: givoltage
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import os
import pickle
from halotools.mock_observables import return_xyz_formatted_array, rp_pi_tpcf, rp_pi_tpcf_jackknife
import Halotools as ht # this is the Halotools for Abacus which imports catalogues in a unified format, not astropy.halotools
from halotools.sim_manager import FakeSim
from Corrfunc.theory.DDrppi import DDrppi
from Corrfunc.utils import convert_3d_counts_to_cf

save_dir = '/home/dyt/analysis_data/'
use_corrfunc = False
use_jackknife = False

def produce_baofit_input(cats, sim_name_prefix, np_cut = 100, n_rp_bins = 8, n_pi_bins = 12, num_threads = 10):
    
    ''' use with only one cosmology at a time'''
    
    n_redshifts = len(cats)
    n_phases = len(cats[0][0])
    # xi_cats = np.empty((n_rp_bins-1, n_pi_bins-1, n_redshifts, n_phases))
    # xi_cov_cats = np.empty(((n_rp_bins-1)*(n_pi_bins-1), (n_rp_bins-1)*(n_pi_bins-1), n_redshifts, n_phases))
    
    l=0
    xi_list = []
    for m in range(n_phases):

        try:
            sim_name = cats[0][0][m].SimName
        except:
            sim_name = cats[0][0][m].simname
            
        for n in range(len(cats[0])):
            for k in range(n_redshifts):
                l = l + 1
                print('Processing catalogue ' + str(l) + ' of ' + str(n_redshifts*n_phases))
                cat = cats[k][n][m]
                redshift = cat.redshift
#                # apply particle number cut for halo clustering
#                n0_halos = len(cat.halo_table)
#                cat.halo_table= cat.halo_table[cat.halo_table["halo_num_p"] > np_cut]
#                n_halo = len(cat.halo_table)
#                print('Particle number cut at {} particles: {} / {} = {:.2%} left'.format(np_cut, n_halo, n0_halos, n_halo/n0_halos))
#                
                
                if use_corrfunc:
                    # use corrfunc
                    xi = baofit_data_corrfunc()
                    xi_list.append(xi)
                    return xi
                if use_jackknife:
                    # use halotools jackknife
                    xi, xi_cov = baofit_data_ht_jackknife(cat, n_rp_bins, n_pi_bins, num_threads = num_threads)
                    # save baofit data file for a given phase and redshift
                    # baofit_data = xi_cats[:, :, k, m]
                    # baofit_cov = xi_cov_cats[:, :, k, m]
                    baofit_data_save = np.column_stack((np.arange(xi.size), 
                                                        xi.flatten(order = 'C')))
                    iu = np.triu_indices(xi_cov.shape[0])
                    baofit_cov_save = np.column_stack((iu[0], iu[1], xi_cov[iu].flatten(order = 'C').T))
                    path_data = os.path.join(save_dir, sim_name+'-z'+str(redshift)+'.data')
                    path_cov = os.path.join(save_dir, sim_name+'-z'+str(redshift)+'.cov')
                    print('Saving: ' + path_data)
                    np.savetxt(path_data, baofit_data_save, fmt = ['%.0f', '%.30f'], delimiter=' ')
                    print('Saving: ' + path_cov)
                    np.savetxt(path_cov, baofit_cov_save, fmt = ['%.0f', '%.0f', '%.30f'], delimiter=' ')
                else:
                    pass
    # take xi list and calculate cov

    # save cov and each xi
    # to do 

    # # save all xi data together as pickle
    # path_xi = os.path.join(save_dir, sim_name_prefix+'-xi.pkl')
    # path_cov = os.path.join(save_dir, sim_name_prefix+'-cov.pkl')
    # with open(path_xi, 'wb') as file:
    #     print('Saving: ' + path_xi)
    #     pickle.dump(xi_cats, file, protocol = pickle.HIGHEST_PROTOCOL)
    # with open(path_cov, 'wb') as file:
    #     print('Saving: ' + path_cov)
    #     pickle.dump(xi_cov_cats, file, protocol = pickle.HIGHEST_PROTOCOL)

def ht_rp_pi_2pcf(model, L, n_rp_bins = 12, n_pi_bins = 12, num_threads = 10):
    
    x = model.mock.galaxy_table['x']
    y = model.mock.galaxy_table['y']
    z = model.mock.galaxy_table['z']
    pos = return_xyz_formatted_array(x, y, z, period = L)
    rp_bins = np.linspace(50, 150, n_rp_bins) # perpendicular bins
    pi_bins = np.linspace(50, 150, n_pi_bins) # parallel bins
    xi = rp_pi_tpcf(pos, rp_bins, pi_bins, period=L, num_threads = num_threads)
    return xi

def cov_from_xi_list(xi_list):

    print('Calculating covariance matrix from {} mocks...'.format(str(len(xi_list))))
    xi_stack = np.stack(xi_list).transpose([1,2,0])
    xi_stack = xi_stack.reshape(np.int(xi_stack.size/len(xi_list)), len(xi_list))
    # print(xi_stack.shape)
    return np.cov(xi_stack)

def xi_to_baofit_input(xi, path_xi):
    
    xi_save = np.column_stack((np.arange(xi.size), xi.flatten(order = 'C')))
    print('Saving: ' + path_xi)
    np.savetxt(path_xi, xi_save, fmt = ['%.0f', '%.30f'], delimiter=' ')
    
def cov_to_baofit_input(cov, path_cov):
    
    iu = np.triu_indices(cov.shape[0])
    cov_save = np.column_stack((iu[0], iu[1], cov[iu].flatten(order = 'C').T))
    print('Saving: ' + path_cov)
    np.savetxt(path_cov, cov_save, fmt = ['%.0f', '%.0f', '%1.19e'], delimiter=' ')

def baofit_data_ht_jackknife(cat, n_rp_bins, n_pi_bins, num_threads):
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
    n_ran = len(cat.halo_table) * 50 # 500 ideal?
    print('Preparing randoms...')
    xran = np.random.uniform(0, L, n_ran)
    yran = np.random.uniform(0, L, n_ran)
    zran = np.random.uniform(0, L, n_ran)
    randoms = return_xyz_formatted_array(xran, yran, zran, period = L)
    print('2pt jackknife with ' + str(num_threads) + ' threads...')
    xi, xi_cov = rp_pi_tpcf_jackknife(pos, randoms, rp_bins, pi_bins, Nsub=3, period=L, num_threads = num_threads)
    try:
        np.linalg.cholesky(xi_cov)
        print('Covariance matrix passes cholesky decomposition')
    except:
        print('Covariance matrix fails cholesky decomposition')
        
    # save xi as .data file for baofit
#    baofit_data = np.column_stack((np.arange(xi.size), xi.flatten(order = 'c')))
#    path = os.path.join('/home/dyt/analysis_data/', sim_name+'_'+str(redshift)+'.data')
#    print 'saving: ' + path
#    np.savetxt(path, baofit_data, fmt = ['%.0f', '%.30f'], delimiter=' ')
    return xi, xi_cov

def baofit_data_corrfunc(cat, n_rp_bins, n_threads):
    # Distances along the :math:\pi direction are binned with unit depth. 
    # For instance, if pimax=40, then 40 bins will be created along the pi direction
    try:
        L = cat.BoxSize
    except:
        L = cat.Lbox[0]
    # redshift = cat.redshift
    x = cat.halo_table['halo_x']
    y = cat.halo_table['halo_y']
    z = cat.halo_table['halo_z']
    pimax = 140
    n = len.cat.halo_table
    nr = n * 3
    xr = np.random.uniform(0, L, nr)
    yr = np.random.uniform(0, L, nr)
    zr = np.random.uniform(0, L, nr)
    rp_bins = np.logspace(np.log10(80), np.log10(130), n_rp_bins) # perpendicular bins
    DD = DDrppi(1, n_threads, pimax, rp_bins, x, y, z, periodic=True, verbose=True, boxsize=L, output_rpavg=True, zbin_refine_factor=20)
    RR = DDrppi(1, n_threads, pimax, rp_bins, xr, yr, zr, periodic=True, verbose=True, boxsize=L, output_rpavg=True, zbin_refine_factor=20)
    DR = DDrppi(0, n_threads, pimax, rp_bins, x, y, z, X2 = xr, Y2 = yr, Z2 = zr, periodic=True, verbose=True, boxsize=L, output_rpavg=True, zbin_refine_factor=20)
    xi = convert_3d_counts_to_cf(n, n, nr, nr, DD, DR, DR, RR)
    return xi
    
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

if __name__ == "__main__":
    #    %%time
    # catalogue parameters
    sim_name_prefix = 'emulator_1100box_planck'
    cosmologies = [0] # 'all'
    redshifts = [0.700] #'all'
    products_dir = r'/mnt/gosling2/bigsim_products/emulator_1100box_planck_products/'
    phases =  list(range(16)) # 'all'
    halo_type = 'Rockstar'
    num_threads = 10
    for i, phase in enumerate(phases):
        print('Working on phase {} of {}'.format(i, len(phases)))
        cats = ht.make_catalogs(sim_name = sim_name_prefix, cosmologies = cosmologies, redshifts = redshifts, 
                                products_dir = products_dir, phases = [phase], halo_type = halo_type,
                                load_halo_ptcl_catalog = False, # this loads subsamples, does not work
                                load_ptcl_catalog = False, # this loads uniform subsamples, not implemented
                                load_pids = 'auto')
        # calculate 2pcf, xi as a function of rp (radial bins perpendicular to the LOS) and pi (radial bins parallel to the LOS) in redshift-space
        produce_baofit_input(cats, sim_name_prefix, np_cut = 1000, n_rp_bins = 8, n_pi_bins = 12, num_threads = num_threads)



