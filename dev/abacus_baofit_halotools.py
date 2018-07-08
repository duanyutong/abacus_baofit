# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 17:30:51 2017

@author: Duan Yutong (dyt@physics.bu.edu)

"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import numpy as np
import os
from glob import glob
import itertools
from multiprocessing import Pool
import pickle
from halotools.mock_observables import tpcf_multipole
from halotools.mock_observables.two_point_clustering.s_mu_tpcf import (
        spherical_sector_volume)
from halotools.empirical_models import PrebuiltHodModelFactory
from halotools.utils import add_halo_hostid
import Halotools as ht  # Abacus' "Halotools" for importing Abacus catalogues
from Corrfunc.theory.DDsmu import DDsmu

# %% catalogue parameters
sim_name_prefix = 'emulator_1100box_planck'
save_dir = os.path.join('/home/dyt/analysis_data/', sim_name_prefix+'-mgrav')
redshift = 0.7  # one redshift at a time instead of 'all'
cosmology = 0  # use with only one cosmology at a time instead of 'all'
phases = [0]  # range(16)  # [0, 1] # list(range(16))
n_sub = 3
n_cut = 100  # number particle cut, 100 corresponds to 1e12 Msun
n_reals = 1  # indices for realisations for an HOD, list of integers
prebuilt_models = ['zheng07']
# prebuilt_models = ['zheng07', 'cacciato09', 'leauthaud11', 'tinker13']
# prebuilt_models = ['zheng07', 'leauthaud11', 'tinker13', 'hearin15',
#                   'zu_mandelbaum15', 'zu_mandelbaum16', 'cacciato09']
prod_dir = r'/mnt/gosling2/bigsim_products/emulator_1100box_planck_products/'
halo_type = 'Rockstar'
halo_m_prop = 'halo_mgrav'
n_threads = 10
step_s_bins = 5  # mpc/h, bins for fitting
step_mu_bins = 0.05  # bins for fitting
use_analytic_randoms = True
debug_mode = False
# use_jackknife = False
txtfmt = b'%.30e'

# %% set up bins using given parameters
s_bins = np.arange(0, 150 + step_s_bins, step_s_bins)  # for fitting
s_bins_centre = (s_bins[:-1] + s_bins[1:]) / 2
mu_bins = np.arange(0, 1 + step_mu_bins, step_mu_bins)  # for fitting
s_bins_counts = np.arange(0, 151, 1)  # always count with s bin size 1
mu_max = 1.0  # for counting
n_mu_bins = 100  # for couting, mu bin size 0.01


# %% definitionss of statisical formulae

def auto_analytic_random(ND, s_bins, mu_bins, V):
    '''
    DD and RR for auto-correlations
    assume NR = ND for analytic calculations, an arbitrary choice.
    doesn't matter as long as number densities are the same n_R = n_D
    '''
    NR = ND
    mu_bins_reverse_sorted = np.sort(mu_bins)[::-1]
    dv = spherical_sector_volume(s_bins, mu_bins_reverse_sorted)
    dv = np.diff(dv, axis=1)  # volume of wedges
    dv = np.diff(dv, axis=0)  # volume of wedge sector
    DR = ND*(ND-1)/V*dv  # calculate randoms for sample1 and 2
    RR = NR*(NR-1)/V*dv  # calculate the random-random pairs.

    return DR, RR


def cross_analytic_random(NR1, ND2, s_bins, mu_bins, V1):
    '''
    DR for cross-correlations betewen random_1 and data_2
    NR1 is the random sample for the box, V1 is L^3, and ND2 is the subsample
    '''
    mu_bins_reverse_sorted = np.sort(mu_bins)[::-1]
    dv = spherical_sector_volume(s_bins, mu_bins_reverse_sorted)
    dv = np.diff(dv, axis=1)  # volume of wedges
    dv = np.diff(dv, axis=0)  # volume of wedge sector
    R1D2 = NR1*ND2/V1*dv

    return R1D2


def ph_estimator(ND, NR, DD, RR):

    '''
    DD/RR - 1, a.k.a natural estimator
    for auto-correlation only
    '''
    if DD.shape == RR.shape:
        xi = NR*(NR-1)/ND/(ND-1)*DD/RR - 1
        return xi
    else:
        print('Warning: DD, RR dimensions mismatch')
        return None


def dp_estimator(nD1, nR1, D1D2, R1D2):
    '''
    Davis & Peebles (1983) estimator
    Can be found in https://arxiv.org/abs/1207.0005, equation 15
    for cross-correlation only
    lower case n is the number density
    '''
    xi = nR1/nD1*D1D2/R1D2 - 1
    return xi


def rebin(cts):
    '''
    input counts is the output table of DDSmu, saved as npy file
    output npairs is an n_s_bins by n_mu_bins matrix
    '''

    npairs = np.zeros((s_bins.size-1, mu_bins.size-1), dtype=np.int64)
    bins_included = 0
    for m, n in itertools.product(
            range(npairs.shape[0]), range(npairs.shape[1])):
        # smin and smax fields can be equal to bin edge
        # when they are equal, it's a bin right on the edge to be included
        mask = (
            (s_bins[m] <= cts['smin']) & (cts['smax'] <= s_bins[m+1]) &
            (mu_bins[n] < cts['mu_max']) & (cts['mu_max'] <= mu_bins[n+1]))
        npairs[m, n] = cts['npairs'][mask].sum()
        bins_included = bins_included + cts['npairs'][mask].size
    # print('Total DD fine bins: {}. Total bins included: {}'
    #       .format(cts.size, bins_included))
    # print('Total npairs in fine bins: {}. Total npairs after rebinning: {}'
    #       .format(cts['npairs'].sum(), npairs.sum()))
    indices = np.where(npairs == 0)  # check if all bins are filled
    if len(indices[0]) != 0:  # print problematic bins if any is empty
        for i in range(len(indices[0])):
            print('({}, {}) is empty'.format(indices[0][i], indices[1][i]))

    return npairs


def xi1d_list_to_cov(xi_list):
    '''
    rows are bins and columns are sample values for each bin
    '''

    print('Calculating covariance matrix from {} xi multipole matrices...'
          .format(len(xi_list)))
    return np.cov(np.array(xi_list).transpose())


def coadd_xi_list(xi_s_mu_list, xi_0_list, xi_2_list):

    xi_s_mu_array = np.array(xi_s_mu_list)
    xi_0_array = np.array(xi_0_list)
    xi_2_array = np.array(xi_2_list)
    xi_s_mu_ca = np.mean(xi_s_mu_array, axis=0)
    xi_s_mu_err = np.std(xi_s_mu_array, axis=0)
    xi_0_ca = np.mean(xi_0_array, axis=0)  # from coadding xi_0
    xi_0_err = np.std(xi_0_array, axis=0)
    xi_2_ca = np.mean(xi_2_array, axis=0)  # from coadding xi_2
    xi_2_err = np.std(xi_2_array, axis=0)

    return xi_s_mu_ca, xi_s_mu_err, xi_0_ca, xi_0_err, xi_2_ca, xi_2_err


# %% functions for tasks

def make_halocat(phase):

    print('---\n'
          + 'Importing phase {} halo catelogue from: \n'.format(phase)
          + '{}{}_{:02}-{}/z{}/'
            .format(prod_dir, sim_name_prefix, cosmology, phase, redshift)
          + '\n---')

    halocats = ht.make_catalogs(
        sim_name=sim_name_prefix, products_dir=prod_dir,
        redshifts=[redshift], cosmologies=[cosmology], phases=[phase],
        halo_type=halo_type,
        load_halo_ptcl_catalog=False,  # this loads subsamples, does not work
        load_ptcl_catalog=False,  # this loads uniform subsamples, dnw
        load_pids='auto')

    halocat = halocats[0][0][0]

    # fill in fields required by HOD models
    if 'halo_mgrav' not in halocat.halo_table.keys():
        print('halo_mgrav field missing in halo table')
    if 'halo_mvir' not in halocat.halo_table.keys():
        halocat.halo_table['halo_mvir'] = halocat.halo_table['halo_m']
        print('halo_mvir field missing, assuming halo_mvir = halo_m')
    if 'halo_rvir' not in halocat.halo_table.keys():
        halocat.halo_table['halo_rvir'] = halocat.halo_table['halo_r']
        print('halo_rvir field missing, assuming halo_rvir = halo_r')
    add_halo_hostid(halocat.halo_table, delete_possibly_existing_column=True)
    # setting NFW concentration using scale radius
    # klypin_rs is more stable and better for small halos
    # can be improved following https://arxiv.org/abs/1709.07099
    halocat.halo_table['halo_nfw_conc'] = (
        halocat.halo_table['halo_rvir'] / halocat.halo_table['halo_klypin_rs'])

    return halocat


def initialise_model(redshift, model_name='zheng07'):

    print('Initialising HOD model: {}...'.format(model_name))

    # Default is ‘halo_mvir’
    if model_name == 'zheng07':
        model = PrebuiltHodModelFactory('zheng07',
                                        redshift=redshift,
                                        threshold=-18,
                                        prim_haloprop_key=halo_m_prop)
        model.param_dict['logMmin'] = 13.3
        model.param_dict['sigma_logM'] = 0.8
        model.param_dict['alpha'] = 1
        model.param_dict['logM0'] = 13.3
        model.param_dict['logM1'] = 13.8

    elif model_name == 'leauthaud11':
        model = PrebuiltHodModelFactory('leauthaud11',
                                        redshift=redshift,
                                        threshold=11,
                                        prim_haloprop_key=halo_m_prop)
        model.param_dict['smhm_m0_0'] = 11.5  # 10.72
        model.param_dict['smhm_m0_a'] = 0.59
        model.param_dict['smhm_m1_0'] = 13.4  # 12.35
        model.param_dict['smhm_m1_a'] = 0.3
        model.param_dict['smhm_beta_0'] = 2  # 0.43
        model.param_dict['smhm_beta_a'] = 0.18
        model.param_dict['smhm_delta_0'] = 0.1  # 0.56
        model.param_dict['smhm_delta_a'] = 0.18
        model.param_dict['smhm_gamma_0'] = 1  # 1.54
        model.param_dict['smhm_gamma_a'] = 2.52
        model.param_dict['scatter_model_param1'] = 0.2
        model.param_dict['alphasat'] = 1
        model.param_dict['betasat'] = 1.1  # 0.859
        model.param_dict['bsat'] = 11  # 10.62
        model.param_dict['betacut'] = 6  # -0.13
        model.param_dict['bcut'] = 0.01  # 1.47

    elif model_name == 'tinker13':
        model = PrebuiltHodModelFactory(
                    'tinker13',
                    redshift=redshift,
                    threshold=11,
                    prim_haloprop_key=halo_m_prop,
                    quiescent_fraction_abscissa=[1e12, 1e13, 1e14, 1e15],
                    quiescent_fraction_ordinates=[0.25, 0.5, 0.75, 0.9])

        model.param_dict['smhm_m0_0_active'] = 11
        model.param_dict['smhm_m0_0_quiescent'] = 10.8
        model.param_dict['smhm_m1_0_active'] = 12.2
        model.param_dict['smhm_m1_0_quiescent'] = 11.8
        model.param_dict['smhm_beta_0_active'] = 0.44
        model.param_dict['smhm_beta_0_quiescent'] = 0.32
        # model.param_dict['alphasat_active'] = 1
        # model.param_dict['alphasat_quiescent'] = 1
        # model.param_dict['betacut_active'] = 0.77
        # model.param_dict['betacut_quiescent'] = -0.12
        # model.param_dict['bcut_active'] = 0.2
        # model.param_dict['bcut_quiescent'] = 0.2
        # model.param_dict['betasat_active'] = 1.5
        # model.param_dict['betasat_quiescent'] = 0.62
        model.param_dict['bsat_active'] = 13
        model.param_dict['bsat_quiescent'] = 8

    elif model_name == 'hearin15':
        model = PrebuiltHodModelFactory('hearin15',
                                        redshift=redshift,
                                        threshold=11)
    elif model_name == 'zu_mandelbaum15':
        model = PrebuiltHodModelFactory('zheng07',
                                        redshift=redshift,
                                        threshold=-1)
    elif model_name == 'zu_mandelbaum16':
        model = PrebuiltHodModelFactory('zheng07',
                                        redshift=redshift,
                                        threshold=-1)
    elif model_name == 'cacciato09':
        model = PrebuiltHodModelFactory('cacciato09',
                                        redshift=redshift,
                                        threshold=10,
                                        prim_haloprop_key=halo_m_prop)

        model.param_dict['log_L_0'] = 9.935
        model.param_dict['log_M_1'] = 12.9  # 11.07
        model.param_dict['gamma_1'] = 0.3  # 3.273
        model.param_dict['gamma_2'] = 0.255
        model.param_dict['sigma'] = 0.143
        model.param_dict['a_1'] = 0.501
        model.param_dict['a_2'] = 2.106
        model.param_dict['log_M_2'] = 14.28
        model.param_dict['b_0'] = -0.5  # -0.766
        model.param_dict['b_1'] = 1.008
        model.param_dict['b_2'] = -0.094
        model.param_dict['delta_1'] = 0
        model.param_dict['delta_2'] = 0

    # add useful model properties here
    model.model_name = model_name  # add model_name field

    return model


def populate_halocat(halocat, model, n_cut=100,
                     reuse_old_model=False, save_new_model=True):

    '''
    if save_new_model is True:
        the current monte carlo realisation is saved and used
    '''

    print('Populating {} halos with n_cut = {}...'
          .format(len(halocat.halo_table), n_cut))
    if hasattr(model, 'mock'):
        model.mock.populate()
    else:
        model.populate_mock(halocat, Num_ptcl_requirement=n_cut)
    print('Mock catalogue populated with {} galaxies'
          .format(len(model.mock.galaxy_table)))

    if save_new_model:
        # save this particular monte carlo realisation of the model
        # dumping model instance yields error; dump galaxy table only
        sim_name = model.mock.header['SimName']
        filedir = os.path.join(save_dir, sim_name,
                               'z{}-r{}'.format(redshift, model.r))
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        filepath = os.path.join(filedir,
                                '{}-galaxy_table.pkl'.format(model.model_name))
        print('Pickle dumping galaxy table...')
        with open(filepath, 'wb') as handle:
            pickle.dump(model.mock.galaxy_table, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
        print('Galaxy table saved to: {}'.format(filepath))

    return model


def do_auto_count(model, do_DD=True, do_RR=True):
    # do the pair counting in 1 Mpc bins

    sim_name = model.mock.header['SimName']
    L = model.mock.BoxSize
    Lbox = model.mock.Lbox
    redshift = model.mock.redshift
    model_name = model.model_name
    print('Auto-correlation pair counting for entire volume with {} threads...'
          .format(n_threads))
    x = model.mock.galaxy_table['x']
    y = model.mock.galaxy_table['y']
    z = model.mock.galaxy_table['z']
    filedir = os.path.join(save_dir, sim_name,
                           'z{}-r{}'.format(redshift, model.r))
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    # c_api_timer needs to be set to false to make sure the DD variable
    # and the npy file saved are exactly the same and can be recovered
    # otherwise, with c_api_timer enabled, file is in "object" format,
    # not proper datatypes, and will cause indexing issues
    if do_DD:
        DD = DDsmu(1, n_threads, s_bins_counts, mu_max, n_mu_bins, x, y, z,
                   periodic=True, verbose=False, boxsize=L,
                   c_api_timer=False)
        # save counting results as original structured array in npy format
        np.save(os.path.join(filedir,
                             '{}-auto-paircount-DD.npy'
                             .format(model_name)),
                DD)

    # calculate RR analytically for auto-correlation, in coarse bins
    if do_RR:
        ND = len(model.mock.galaxy_table)
        _, RR = auto_analytic_random(ND, s_bins, mu_bins, Lbox.prod())
        # save counting results as original structured array in npy format
        np.savetxt(os.path.join(filedir,
                                '{}-auto-paircount-RR.txt'
                                .format(model_name)),
                   RR, fmt=txtfmt)
    if not use_analytic_randoms or debug_mode:
        ND = len(model.mock.galaxy_table)
        NR = ND
        xr = np.random.uniform(0, L, NR)
        yr = np.random.uniform(0, L, NR)
        zr = np.random.uniform(0, L, NR)
        RR = DDsmu(1, n_threads, s_bins_counts, mu_max, n_mu_bins,
                   xr, yr, zr, periodic=True, verbose=False,
                   boxsize=L, c_api_timer=False)
        np.save(os.path.join(filedir, '{}-auto-paircount-RR.npy'
                             .format(model_name)),
                RR)


def do_subcross_count(model, n_sub, do_DD=True, do_DR=True):

    sim_name = model.mock.header['SimName']
    model_name = model.model_name
    L = model.mock.BoxSize
    Lbox = model.mock.Lbox
    redshift = model.mock.redshift
    l_sub = L / n_sub  # length of subvolume box
    gt = model.mock.galaxy_table
    x1, y1, z1 = gt['x'], gt['y'], gt['z']
    ND1 = len(gt)  # number of galaxies in entire volume of periodic box
    filedir = os.path.join(save_dir, sim_name,
                           'z{}-r{}'.format(redshift, model.r))

    for i, j, k in itertools.product(range(n_sub), repeat=3):

        linind = i*n_sub**2 + j*n_sub + k  # linearised index of subvolumes
        mask = ((l_sub * i < gt['x']) & (gt['x'] <= l_sub * (i+1)) &
                (l_sub * j < gt['y']) & (gt['y'] <= l_sub * (j+1)) &
                (l_sub * k < gt['z']) & (gt['z'] <= l_sub * (k+1)))
        gt_sub = gt[mask]  # select a subvolume of the galaxy table
        ND2 = len(gt_sub)  # number of galaxies in the subvolume

        if do_DD:
            # print('Subvolume {} of {} mask created, '
            #       '{} of {} galaxies selected'
            #       .format(linind+1, np.power(n_sub, 3),
            #               len(gt_sub), len(gt)))
            # print('Actual min/max x, y, z of galaxy sample:'
            #       '({:3.2f}, {:3.2f}, {:3.2f}),'
            #       '({:3.2f}, {:3.2f}, {:3.2f})'
            #       .format(np.min(gt_sub['x']),
            #               np.min(gt_sub['y']),
            #               np.min(gt_sub['z']),
            #               np.max(gt_sub['x']),
            #               np.max(gt_sub['y']),
            #               np.max(gt_sub['z'])))
            # print('Cross-correlation pair counting for subvolume'
            #       + 'with {} threads...'.format(n_threads))
            # calculate D1D2, where D1 is entire box, and D2 is subvolume
            x2, y2, z2 = gt_sub['x'], gt_sub['y'], gt_sub['z']
            D1D2 = DDsmu(0, n_threads, s_bins_counts, mu_max, n_mu_bins,
                         x1, y1, z1, periodic=True, X2=x2, Y2=y2, Z2=z2,
                         verbose=False, boxsize=L, c_api_timer=False)
            np.save(os.path.join(filedir,
                                 '{}-cross_{}-paircount-D1D2.npy'
                                 .format(model_name, linind)),
                    D1D2)

        if do_DR:
            NR1 = ND1
            # the time trade off is ~5 minutes
            if use_analytic_randoms or debug_mode:
                # calculate cross-correlation DR analytically
                # similar code can be found in halotools:
                # https://goo.gl/W9Njbv
                R1D2 = cross_analytic_random(NR1, ND2,
                                             s_bins, mu_bins, Lbox.prod())
                np.savetxt(os.path.join(filedir,
                                        '{}-cross_{}-paircount-R1D2.txt'
                                        .format(model_name, linind)),
                           R1D2, fmt=txtfmt)

            if not use_analytic_randoms or debug_mode:
                # calculate D2Rbox by brute force sampling, where
                # Rbox sample is homogeneous in volume of box
                print('Pair counting R1D2 with Corrfunc.DDsmu, '
                      '{} threads...'
                      .format(n_threads))
                # force float32 (a.k.a. single precision float in C)
                # to be consistent with galaxy table precision
                # otherwise corrfunc raises an error
                xr = np.random.uniform(0, L, NR1).astype(np.float32)
                yr = np.random.uniform(0, L, NR1).astype(np.float32)
                zr = np.random.uniform(0, L, NR1).astype(np.float32)
                R1D2 = DDsmu(0, n_threads,
                             s_bins_counts, mu_max, n_mu_bins,
                             xr, yr, zr, periodic=True,
                             X2=x2, Y2=y2, Z2=z2,
                             verbose=False, boxsize=L, c_api_timer=False)
                np.save(os.path.join(filedir,
                                     '{}-cross_{}-paircount-R1D2.npy'
                                     .format(model_name, linind)),
                        R1D2)


def do_auto_correlation(model):

    '''
    read in counts, and generate xi_s_mu using bins specified
    using the PH (natural) estimator for periodic sim boxes instead of LS
    '''

    # setting halocat properties to be compatibble with halotools
    sim_name = model.mock.header['SimName']
    redshift = model.mock.redshift
    model_name = model.model_name
    s_bins = np.arange(0, 150 + step_s_bins, step_s_bins)
    s_bins_centre = (s_bins[:-1] + s_bins[1:]) / 2
    mu_bins = np.arange(0, 1 + step_mu_bins, step_mu_bins)
    filedir = os.path.join(save_dir, sim_name,
                           'z{}-r{}'.format(redshift, model.r))

    # read in pair counts file which has fine bins
    filepath1 = os.path.join(filedir,
                             '{}-auto-paircount-DD.npy'.format(model_name))

    DD = np.load(filepath1)
    ND = len(model.mock.galaxy_table)
    NR = ND
    # re-count pairs in new bins, re-bin the counts
    # print('Re-binning auto DD counts into ({}, {})...'
    #       .format(s_bins.size-1, mu_bins.size-1))
    npairs_DD = rebin(DD)
    # save re-binned pair ounts
    np.savetxt(os.path.join(filedir, '{}-auto-paircount-DD-rebinned.txt'
                            .format(model_name)),
               npairs_DD, fmt=txtfmt)
    if use_analytic_randoms:
        filepath2 = os.path.join(filedir,
                                 '{}-auto-paircount-RR.txt'.format(model_name))
        npairs_RR = np.loadtxt(filepath2)
    if not use_analytic_randoms or debug_mode:
        # rebin RR.npy counts
        RR = np.load(os.path.join(filedir, '{}-auto-paircount-RR.npy'
                                  .format(model_name)))

        npairs_RR = rebin(RR)
        # save re-binned pair ounts
        np.savetxt(os.path.join(
                filedir,
                '{}-auto-paircount-RR-rebinned.txt'.format(model_name)),
            npairs_RR, fmt=txtfmt)

    xi_s_mu = ph_estimator(ND, NR, npairs_DD, npairs_RR)
    # print('Calculating first two orders of auto-correlation multipoles...')
    xi_0 = tpcf_multipole(xi_s_mu, mu_bins, order=0)
    xi_2 = tpcf_multipole(xi_s_mu, mu_bins, order=2)
    xi_0_txt = np.vstack([s_bins_centre, xi_0]).transpose()
    xi_2_txt = np.vstack([s_bins_centre, xi_2]).transpose()
    # save to text
    # print('Saving auto-correlation texts to: {}'.format(filedir))
    np.savetxt(os.path.join(filedir, '{}-auto-xi_s_mu.txt'
                            .format(model_name)),
               xi_s_mu, fmt=txtfmt)
    np.savetxt(os.path.join(filedir, '{}-auto-xi_0.txt'
                            .format(model_name)),
               xi_0_txt, fmt=txtfmt)
    np.savetxt(os.path.join(filedir, '{}-auto-xi_2.txt'
                            .format(model_name)),
               xi_2_txt, fmt=txtfmt)

    return xi_s_mu, xi_0_txt, xi_2_txt


def do_subcross_correlation(model, n_sub=3):  # n defines number of subvolums
    '''
    cross-correlation between 1/n_sub^3 of a box to the whole box
    n_sub^3 * 16 results are used for emperically estimating covariance matrix
    '''
    # setting halocat properties to be compatibble with halotools
    sim_name = model.mock.header['SimName']
    Lbox = model.mock.Lbox
    redshift = model.mock.redshift
    model_name = model.model_name
    gt = model.mock.galaxy_table
    ND1 = len(gt)
    s_bins = np.arange(0, 150 + step_s_bins, step_s_bins)
    s_bins_centre = (s_bins[:-1] + s_bins[1:]) / 2
    mu_bins = np.arange(0, 1 + step_mu_bins, step_mu_bins)
    filedir = os.path.join(save_dir, sim_name,
                           'z{}-r{}'.format(redshift, model.r))

    for i, j, k in itertools.product(range(n_sub), repeat=3):
        linind = i*n_sub**2 + j*n_sub + k  # linearised index

        # read in pair counts file which has fine bins
        D1D2 = np.load(os.path.join(filedir,
                                    '{}-cross_{}-paircount-D1D2.npy'
                                    .format(model_name, linind)))
        # print('Re-binning {} of {} cross-correlation into ({}, {})...'
        #       .format(linind+1, n_sub**3, s_bins.size-1, mu_bins.size-1))
        # set up bins using specifications, re-bin the counts
        npairs_D1D2 = rebin(D1D2)
        # save re-binned pair ounts
        np.savetxt(os.path.join(filedir,
                                '{}-cross_{}-paircount-D1D2-rebinned.txt'
                                .format(model_name, linind)),
                   npairs_D1D2, fmt=txtfmt)

        # read R1D2 counts and re-bin
        if use_analytic_randoms:
            npairs_R1D2 = np.loadtxt(
                os.path.join(filedir, '{}-cross_{}-paircount-R1D2.txt'
                             .format(model_name, linind)))
        if not use_analytic_randoms or debug_mode:
            R1D2 = np.load(os.path.join(filedir,
                                        '{}-cross_{}-paircount-R1D2.npy'
                                        .format(model_name, linind)))
            npairs_R1D2 = rebin(R1D2)
            # save re-binned pair ounts
            np.savetxt(os.path.join(
                    filedir,
                    '{}-cross_{}-paircount-R1D2-rebinned.txt'
                    .format(model_name, linind)),
                npairs_R1D2, fmt=txtfmt)

        # calculate cross-correlation from counts using dp estimator
        nD1 = ND1 / Lbox.prod()
        nR1 = nD1
        xi_s_mu = dp_estimator(nD1, nR1, npairs_D1D2, npairs_R1D2)

        # print('Calculating cross-correlation for subvolume {} of {}...'
        #       .format(linind+1, np.power(n_sub, 3)))
        xi_0 = tpcf_multipole(xi_s_mu, mu_bins, order=0)
        xi_2 = tpcf_multipole(xi_s_mu, mu_bins, order=2)
        xi_0_txt = np.vstack([s_bins_centre, xi_0]).transpose()
        xi_2_txt = np.vstack([s_bins_centre, xi_2]).transpose()
        # save to text
        np.savetxt(os.path.join(filedir,
                                '{}-cross_{}-xi_s_mu.txt'
                                .format(model_name, linind)),
                   xi_s_mu, fmt=txtfmt)
        np.savetxt(os.path.join(filedir, '{}-cross_{}-xi_0.txt'
                                .format(model_name, linind)),
                   xi_0_txt, fmt=txtfmt)
        np.savetxt(os.path.join(filedir,
                                '{}-cross_{}-xi_2.txt'
                                .format(model_name, linind)),
                   xi_2_txt, fmt=txtfmt)


def do_coadd_phases(model_name, coadd_phases=range(16)):

    xi_list_phases = []
    xi_0_list_phases = []
    xi_2_list_phases = []

    for phase in coadd_phases:

        sim_name = '{}_{:02}-{}'.format(sim_name_prefix, cosmology, phase)
        path_prefix = os.path.join(
                        save_dir,
                        sim_name,
                        'z{}'.format(redshift),
                        model_name + '-auto-')
        path_xi = path_prefix + 'xi_s_mu.txt'
        path_xi_0 = path_prefix + 'xi_0.txt'
        path_xi_2 = path_prefix + 'xi_2.txt'
        xi_list_phases.append(np.loadtxt(path_xi))
        xi_0_list_phases.append(np.loadtxt(path_xi_0))
        xi_2_list_phases.append(np.loadtxt(path_xi_2))
    # create save dir
    filedir = os.path.join(
        save_dir,
        sim_name_prefix + '_' + str(cosmology).zfill(2) + '-combined',
        'z{}'.format(redshift))
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    # perform coadding for two cases
    # print('Coadding xi for all phases, without dropping any...')
    xi_s_mu_ca, xi_s_mu_err, xi_0_ca, xi_0_err, xi_2_ca, xi_2_err = \
        coadd_xi_list(xi_list_phases, xi_0_list_phases, xi_2_list_phases)
    # print('Saving coadded xi texts...')
    np.savetxt(os.path.join(
            filedir, '{}-auto-xi_s_mu-coadd.txt'.format(model_name)),
        xi_s_mu_ca, fmt=txtfmt)
    np.savetxt(os.path.join(
            filedir, '{}-auto-xi_s_mu-coadd_err.txt'.format(model_name)),
        xi_s_mu_err, fmt=txtfmt)
    np.savetxt(os.path.join(
            filedir, '{}-auto-xi_0-coadd.txt'.format(model_name)),
        xi_0_ca, fmt=txtfmt)
    np.savetxt(os.path.join(
            filedir, '{}-auto-xi_0-coadd_err.txt'.format(model_name)),
        xi_0_err, fmt=txtfmt)
    np.savetxt(os.path.join(
            filedir, '{}-auto-xi_2-coadd.txt'.format(model_name)),
        xi_2_ca, fmt=txtfmt)
    np.savetxt(os.path.join(
            filedir, '{}-auto-xi_2-coadd_err.txt'.format(model_name)),
        xi_2_err, fmt=txtfmt)

    for phase in coadd_phases:
        # print('Jackknife coadding xi, dropping phase {}...'.format(phase))
        xi_list = xi_list_phases[:phase] + xi_list_phases[phase+1:]
        xi_0_list = xi_0_list_phases[:phase] + xi_0_list_phases[phase+1:]
        xi_2_list = xi_2_list_phases[:phase] + xi_2_list_phases[phase+1:]
        # print('For phase {}, # phases selected: {}, {}, {}'
        #      .format(phase, len(xi_list), len(xi_0_list), len(xi_2_list)))
        xi_s_mu_ca, xi_s_mu_err, xi_0_ca, xi_0_err, xi_2_ca, xi_2_err = \
            coadd_xi_list(xi_list, xi_0_list, xi_2_list)
        np.savetxt(os.path.join(
                filedir, '{}-auto-xi_s_mu-coadd_jackknife_{}.txt'
                .format(model_name, phase)),
            xi_s_mu_ca, fmt=txtfmt)
        np.savetxt(os.path.join(
                filedir, '{}-auto-xi_s_mu-coadd_err_jackknife_{}.txt'
                .format(model_name, phase)),
            xi_s_mu_err, fmt=txtfmt)
        np.savetxt(os.path.join(
                filedir, '{}-auto-xi_0-coadd_jackknife_{}.txt'
                .format(model_name, phase)),
            xi_0_ca, fmt=txtfmt)
        np.savetxt(os.path.join(
                filedir, '{}-auto-xi_0-coadd_err_jackknife_{}.txt'
                .format(model_name, phase)),
            xi_0_err, fmt=txtfmt)
        np.savetxt(os.path.join(
                filedir, '{}-auto-xi_2-coadd_jackknife_{}.txt'
                .format(model_name, phase)),
            xi_2_ca, fmt=txtfmt)
        np.savetxt(os.path.join(
                filedir, '{}-auto-xi_2-coadd_err_jackknife_{}.txt'
                .format(model_name, phase)),
            xi_2_err, fmt=txtfmt)


def do_cov(model_name, n_sub=3, cov_phases=list(range(16))):

    # calculate cov combining all phases, 16 * n_sub^3 * n_reals
    for ell in [0, 2]:
        paths = []
        for phase in cov_phases:
            sim_name = '{}_{:02}-{}'.format(sim_name_prefix, cosmology, phase)
            paths = paths + glob(os.path.join(
                        save_dir, sim_name, 'z{}-r*'.format(redshift),
                        '{}-cross_*-xi_{}.txt'.format(model_name, ell)))
        print('Calculating covariance matrix using {} xi samples from all'
              'phases for l = {}...'
              .format(len(paths), ell))
        # read in all xi files for all phases
        xi_list = [np.loadtxt(path)[:, 1] for path in paths]
        cov = xi1d_list_to_cov(xi_list) / np.power(n_sub, 3)/15/n_reals
        # save cov
        filepath = os.path.join(  # save to '-combined' folder for all phases
                save_dir,
                sim_name_prefix + '_' + str(cosmology).zfill(2) + '-combined',
                'z{}'.format(redshift),
                '{}-cross-xi_{}-cov.txt'.format(model_name, ell))
        np.savetxt(filepath, cov, fmt=txtfmt)
        print('l = {} covariance matrix saved to: {}'.format(ell, filepath))

    # calculate monoquad cov for baofit
    print('Calculating combined mono-quad covariance matrix...')
    xi_list = []
    for phase in phases:
        sim_name = '{}_{:02}-{}'.format(sim_name_prefix, cosmology, phase)
        paths0 = sorted(glob(os.path.join(
                        save_dir, sim_name, 'z{}-r*'.format(redshift),
                        '{}-cross_*-xi_0.txt'.format(model_name))))
        paths2 = sorted(glob(os.path.join(
                        save_dir, sim_name, 'z{}-r*'.format(redshift),
                        '{}-cross_*-xi_2.txt'.format(model_name))))
        if len(paths0) != len(paths2):
            print('Cross correlation monopole and quadrupole mismatch')
            return False
        for path0, path2 in zip(paths0, paths2):
            xi_0 = np.loadtxt(path0)[:, 1]
            xi_2 = np.loadtxt(path2)[:, 1]
            xi_list.append(np.hstack((xi_0, xi_2)))
    cov_monoquad = xi1d_list_to_cov(xi_list) / np.power(n_sub, 3)/15/n_reals
    # save cov
    filepath = os.path.join(  # save to '-combined' folder for all phases
            save_dir,
            sim_name_prefix + '_' + str(cosmology).zfill(2) + '-combined',
            'z{}'.format(redshift),
            '{}-cross-xi_monoquad-cov.txt'.format(model_name, ell))
    np.savetxt(filepath, cov_monoquad, fmt=txtfmt)
    print('Monoquad covariance matrix saved to: ', filepath)


def do_realisations(halocat, model_name, n_reals=5):
    '''
    generates n HOD realisations for a given (phase, model)
    then co-add them and get a single set of xi data
    the rest of the programme will work as if there is only 1 realisation
    '''

    model = initialise_model(halocat.redshift, model_name=model_name)
    # create n_real realisations of the given HOD model
    xi_list_reals = []
    xi_0_list_reals = []
    xi_2_list_reals = []
    for r in range(n_reals):
        # add useful model properties here
        model.r = r
        # check if current realisation, r, exists;
        filepath = os.path.join(
                save_dir,
                sim_name_prefix + '_' + str(cosmology).zfill(2),
                'z{}-r{}'.format(redshift, r),
                '{}-galaxy_table.pkl'.format(model_name))
        if os.path.isfile(filepath):
            print('---\n'
                  + '{} of {} realisations for model {} exists.'
                  + '\n---'
                  .format(r+1, n_reals, model_name))
        else:
            print(filepath, 'does not exist')
            print('---\n'
                  + 'Generating {} of {} realisations for model {}...'
                  .format(r+1, n_reals, model_name)
                  + '\n---')
            model = populate_halocat(halocat, model,
                                     n_cut=n_cut, save_new_model=True)
            do_auto_count(model, do_DD=True, do_RR=True)
            do_subcross_count(model, n_sub=n_sub, do_DD=True, do_DR=True)
        # always rebin counts and calculate xi in case bins change
        xi_s_mu, xi_0, xi_2 = do_auto_correlation(model)
        do_subcross_correlation(model, n_sub=n_sub)
        # collect auto-xi to coadd all realisations for the current phase
        xi_list_reals.append(xi_s_mu)
        xi_0_list_reals.append(xi_0)
        xi_2_list_reals.append(xi_2)

    print('Co-adding auto-correlation for {} realisations...'.format(n_reals))
    xi_s_mu_ca, _, xi_0_ca, _, xi_2_ca, _ = \
        coadd_xi_list(xi_list_reals, xi_0_list_reals, xi_2_list_reals)
    filedir = os.path.join(save_dir, halocat.SimName, 'z{}'.format(redshift))
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    np.savetxt(os.path.join(filedir, '{}-auto-xi_s_mu.txt'
                            .format(model_name)),
               xi_s_mu_ca, fmt=txtfmt)
    np.savetxt(os.path.join(filedir, '{}-auto-xi_0.txt'
                            .format(model_name)),
               xi_0_ca, fmt=txtfmt)
    np.savetxt(os.path.join(filedir, '{}-auto-xi_2.txt'
                            .format(model_name)),
               xi_2_ca, fmt=txtfmt)


def run_baofit():

    from baofit import baofit

    filedir = os.path.join(
            save_dir,
            sim_name_prefix + '_' + str(cosmology).zfill(2) + '-combined',
            'z{}'.format(redshift))
    list_of_inputs = []
    for model_name in prebuilt_models:
        for phase in phases:  # construct list of inputs for parallelisation
            path_xi_0 = os.path.join(filedir,
                                     '{}-auto-xi_0-coadd_jackknife_{}.txt'
                                     .format(model_name, phase))
            path_xi_2 = os.path.join(filedir,
                                     '{}-auto-xi_2-coadd_jackknife_{}.txt'
                                     .format(model_name, phase))
            path_cov = os.path.join(filedir, '{}-cross-xi_monoquad-cov.txt'
                                    .format(model_name))
            fout_tag = '{}-{}'.format(model_name, phase)
            list_of_inputs.append([path_xi_0, path_xi_2, path_cov, fout_tag])
    pool = Pool(n_threads)
    pool.map(baofit, list_of_inputs)


if __name__ == "__main__":

    for phase in phases:
        cat = make_halocat(phase)
        for model_name in prebuilt_models:
            do_realisations(cat, model_name, n_reals=n_reals)

    for model_name in prebuilt_models:
        do_coadd_phases(model_name, coadd_phases=phases)
        do_cov(model_name, n_sub=n_sub, cov_phases=phases)

    run_baofit()
