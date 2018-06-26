# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 17:30:51 2017

@author: Duan Yutong (dyt@physics.bu.edu)

"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import os
from glob import glob
import itertools
import functools
from multiprocessing import Pool
import numpy as np
from astropy import table
from halotools.mock_observables import tpcf_multipole
from halotools.mock_observables.two_point_clustering.s_mu_tpcf import (
        spherical_sector_volume)
from halotools.utils import add_halo_hostid
from Corrfunc.theory.DDsmu import DDsmu
from abacus_hod import initialise_model, populate_model

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import Halotools as abacus_ht  # Abacus' "Halotools" for importing Abacus
from tqdm import tqdm

# %% custom settings
sim_name_prefix = 'emulator_1100box_planck'
tagout = 'subsample'
phases = range(16)  # range(16)  # [0, 1] # list(range(16))
cosmology = 0  # one cosmology at a time instead of 'all'
redshift = 0.7  # one redshift at a time instead of 'all'
model_names = ['gen_base3', 'gen_base1', 'gen_base2',
               'gen_ass1', 'gen_ass2', 'gen_ass3']
N_reals = 10  # indices for realisations for an HOD, list of integers
N_cut = 70  # number particle cut, 70 corresponds to 4e12 Msun
N_threads = 16
N_sub = 3  # number of subvolumes per dWimension

# %% flags
debug_mode = False
overwrite_mode = False
save_hod_realisation = True
use_analytic_randoms = True
use_jackknife = True
add_rsd = True

# %% bin settings
step_s_bins = 5  # mpc/h, bins for fitting
step_mu_bins = 0.05  # bins for fitting
s_bins = np.arange(0, 150 + step_s_bins, step_s_bins)  # for fitting
s_bins_centre = (s_bins[:-1] + s_bins[1:]) / 2
mu_bins = np.arange(0, 1 + step_mu_bins, step_mu_bins)  # for fitting
s_bins_counts = np.arange(0, 151, 1)  # always count with s bin size 1
mu_max = 1.0  # for counting
n_mu_bins = 100  # for couting, mu bin size 0.01

# %% fixed catalogue parameters
prod_dir = r'/mnt/gosling2/bigsim_products/emulator_1100box_planck_products/'
save_dir = os.path.join('/home/dyt/store/', sim_name_prefix+'-'+tagout)
halo_type = 'Rockstar'
halo_m_prop = 'halo_mvir'  # mgrav is better but not using sub or small haloes
txtfmt = b'%.30e'


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
    return np.cov(np.array(xi_list).T, bias=0)


def coadd_xi_list(xi_s_mu_list, xi_0_list, xi_2_list):

    xi_s_mu_array = np.array(xi_s_mu_list)
    xi_0_array = np.array(xi_0_list)
    xi_2_array = np.array(xi_2_list)
    xi_s_mu_ca = np.mean(xi_s_mu_array, axis=0)
    xi_s_mu_err = np.std(xi_s_mu_array, axis=0)
    xi_0_ca = np.mean(xi_0_array, axis=0)  # from coadding xi_0
    xi_0_err = np.std(xi_0_array, axis=0)
    xi_0_err[:, 0] = xi_0_ca[:, 0]  # reset r vector, all zeros after std
    xi_2_ca = np.mean(xi_2_array, axis=0)  # from coadding xi_2
    xi_2_err = np.std(xi_2_array, axis=0)
    xi_2_err[:, 0] = xi_2_ca[:, 0]  # reset r vector, all zeros after std

    return xi_s_mu_ca, xi_s_mu_err, xi_0_ca, xi_0_err, xi_2_ca, xi_2_err


# %% functions for tasks

def make_halocat(phase):

    print('---\n'
          + 'Importing phase {} halo catelogue from: \n'.format(phase)
          + '{}{}_{:02}-{}/z{}/'
            .format(prod_dir, sim_name_prefix, cosmology, phase, redshift)
          + '\n---')

    halocats = abacus_ht.make_catalogs(
        sim_name=sim_name_prefix, products_dir=prod_dir,
        redshifts=[redshift], cosmologies=[cosmology], phases=[phase],
        halo_type=halo_type,
        load_halo_ptcl_catalog=True,  # this loads 10% particle subsamples
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


def vrange(starts, lengths):

    '''Create concatenated ranges of integers for multiple start/stop
    Input:
        starts (1-D array_like): starts for each range
        lengths (1-D array_like): lengths for each range (same shape as starts)
    Returns:
        numpy.ndarray: concatenated ranges

    For example:
        >>> starts = [1, 3, 4, 6]
        >>> lengths = [0, 2, 3, 0]
        >>> vrange(starts, lengths)
        array([3, 4, 4, 5, 6])

    '''

    lengths = np.asarray(lengths)
    stops = starts + lengths
    ret = (np.repeat(stops - lengths.cumsum(), lengths)
           + np.arange(lengths.sum()))
    return ret.astype(np.uint64)


def find_repeated_entries(arr):

    '''
    arr = array([1, 2, 3, 1, 1, 3, 4, 3, 2])
    idx_sort_split =
        [array([0, 3, 4]), array([1, 8]), array([2, 5, 7]), array([6])]
    '''

    idx_sort = np.argsort(arr)  # just the argsort
    sorted_arr = arr[idx_sort]
    vals, idx_start = np.unique(sorted_arr, return_index=True,
                                return_counts=False)
    # split as sets of indices
    idx_sort_split = np.split(idx_sort, idx_start[1:])
    # filter w.r.t their size, keeping only items occurring more than once
    # vals = vals[count > 1]
    # ret = filter(lambda x: x.size > 1, res)
    return vals, idx_sort, idx_sort_split


def sum_lengths(i, len_arr):

    return np.sum(len_arr[i])


def process_rockstar_halocat(halocat, N_cut):

    '''
    In Rockstar halo catalogues, the subhalo particles are not included in the
    host halo subsample indices in halo_table. This function applies mass
    cut (70 particles in AbacusCosmos corresponds to 4e12 Msun), gets rid of
    all subhalos, and puts all subhalo particles together with their host halo
    particles in halo_ptcl_table.

    Output halocat has halo_table containing only halos (not subhalos) meeting
    the masscut, and halo_ptcl_table containing all particles belonging to the
    halos selected in contiguous blocks.

    '''

    print('Applying mass cut N={} to halocat and re-organising subsamples...'
          .format(N_cut))
    N0 = len(halocat.halo_table)
    mask_halo = ((halocat.halo_table['halo_upid'] == -1)
                 & (halocat.halo_table['halo_N'] >= N_cut))  # only host halos
    mask_subhalo = ((halocat.halo_table['halo_upid'] != -1)  # child subhalos
                    & np.isin(
                            halocat.halo_table['halo_upid'],
                            halocat.halo_table['halo_id'][mask_halo]))
    htab = halocat.halo_table[mask_halo | mask_subhalo]  # original order
    print('Locating relavent halo host ids in halo table...')
    hostids, hidx, hidx_split = find_repeated_entries(htab['halo_hostid'].data)
    print('Creating subsample particle indices...')
    pidx = vrange(htab['halo_subsamp_start'][hidx],
                  htab['halo_subsamp_len'][hidx])
    print('Re-arranging particle table...')
    ptab = halocat.halo_ptcl_table[pidx]
    # ss_len = [np.sum(htab['halo_subsamp_len'][i]) for i in tqdm(hidx_split)]
    print('Re-writing halo_table subsample fields...')
    pool = Pool()
    ss_len = pool.map(functools.partial(sum_lengths,
                                        len_arr=htab['halo_subsamp_len']),
                      hidx_split)
    htab = htab[htab['halo_upid'] == -1]  # drop subhalos now that ptcl r done
    assert len(htab) == len(hostids)  # sanity check, hostids are unique values
    hidx = np.argsort(htab['halo_hostid'])
    htab['halo_subsamp_len'][hidx] = ss_len  # overwrite halo_subsamp_len field
    assert np.sum(htab['halo_subsamp_len']) == len(ptab)
    htab['halo_subsamp_start'][0] = 0  # reindex halo_subsamp_start
    htab['halo_subsamp_start'][1:] = htab['halo_subsamp_len'].cumsum()[:-1]
    assert (htab['halo_subsamp_start'][-1] + htab['halo_subsamp_len'][-1]
            == len(ptab))
    halocat.halo_table = htab
    halocat.halo_ptcl_table = ptab
    pool.close()
    print('Initial total N halos {}; mass cut mask count {}; '
          'subhalos cut mask count {}; final N halos {}.'
          .format(N0, mask_halo.sum(), mask_subhalo.sum(), len(htab)))

    return halocat


def do_auto_count(model, do_DD=True, do_RR=True):
    # do the pair counting in 1 Mpc bins

    sim_name = model.mock.header['SimName']
    L = model.mock.BoxSize
    Lbox = model.mock.Lbox
    redshift = model.mock.redshift
    model_name = model.model_name
    print('Auto-correlation pair counting for entire volume with {} threads...'
          .format(N_threads))
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
        DD = DDsmu(1, N_threads, s_bins_counts, mu_max, n_mu_bins, x, y, z,
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
        xr = np.random.uniform(0, L, NR).astype(np.float32)
        yr = np.random.uniform(0, L, NR).astype(np.float32)
        zr = np.random.uniform(0, L, NR).astype(np.float32)
        RR = DDsmu(1, N_threads, s_bins_counts, mu_max, n_mu_bins,
                   xr, yr, zr, periodic=True, verbose=False,
                   boxsize=L, c_api_timer=False)
        np.save(os.path.join(filedir, '{}-auto-paircount-RR.npy'
                             .format(model_name)),
                RR)


def do_subcross_count(model, N_sub, do_DD=True, do_DR=True):

    sim_name = model.mock.header['SimName']
    model_name = model.model_name
    L = model.mock.BoxSize
    Lbox = model.mock.Lbox
    redshift = model.mock.redshift
    l_sub = L / N_sub  # length of subvolume box
    gtab = model.mock.galaxy_table
    x1, y1, z1 = gtab['x'], gtab['y'], gtab['z']
    ND1 = len(gtab)  # number of galaxies in entire volume of periodic box
    filedir = os.path.join(save_dir, sim_name,
                           'z{}-r{}'.format(redshift, model.r))

    for i, j, k in itertools.product(range(N_sub), repeat=3):

        linind = i*N_sub**2 + j*N_sub + k  # linearised index of subvolumes
        mask = ((l_sub * i < gtab['x']) & (gtab['x'] <= l_sub * (i+1)) &
                (l_sub * j < gtab['y']) & (gtab['y'] <= l_sub * (j+1)) &
                (l_sub * k < gtab['z']) & (gtab['z'] <= l_sub * (k+1)))
        gt_sub = gtab[mask]  # select a subvolume of the galaxy table
        ND2 = len(gt_sub)  # number of galaxies in the subvolume

        if do_DD:
            # print('Subvolume {} of {} mask created, '
            #       '{} of {} galaxies selected'
            #       .format(linind+1, np.power(N_sub, 3),
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
            #       + 'with {} threads...'.format(N_threads))
            # calculate D1D2, where D1 is entire box, and D2 is subvolume
            x2, y2, z2 = gt_sub['x'], gt_sub['y'], gt_sub['z']
            D1D2 = DDsmu(0, N_threads, s_bins_counts, mu_max, n_mu_bins,
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
                      .format(N_threads))
                # force float32 (a.k.a. single precision float in C)
                # to be consistent with galaxy table precision
                # otherwise corrfunc raises an error
                xr = np.random.uniform(0, L, NR1).astype(np.float32)
                yr = np.random.uniform(0, L, NR1).astype(np.float32)
                zr = np.random.uniform(0, L, NR1).astype(np.float32)
                R1D2 = DDsmu(0, N_threads,
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


def do_subcross_correlation(model, N_sub=3):  # n defines number of subvolums
    '''
    cross-correlation between 1/N_sub^3 of a box to the whole box
    N_sub^3 * 16 results are used for emperically estimating covariance matrix
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

    for i, j, k in itertools.product(range(N_sub), repeat=3):
        linind = i*N_sub**2 + j*N_sub + k  # linearised index

        # read in pair counts file which has fine bins
        D1D2 = np.load(os.path.join(filedir,
                                    '{}-cross_{}-paircount-D1D2.npy'
                                    .format(model_name, linind)))
        # print('Re-binning {} of {} cross-correlation into ({}, {})...'
        #       .format(linind+1, N_sub**3, s_bins.size-1, mu_bins.size-1))
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
        #       .format(linind+1, np.power(N_sub, 3)))
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

    if use_jackknife:
        for phase in coadd_phases:
            # print('Jackknife coadding xi, dropping phase {}...'
            # .format(phase))
            xi_list = xi_list_phases[:phase] + xi_list_phases[phase+1:]
            xi_0_list = xi_0_list_phases[:phase] + xi_0_list_phases[phase+1:]
            xi_2_list = xi_2_list_phases[:phase] + xi_2_list_phases[phase+1:]
            # print('For phase {}, # phases selected: {}, {}, {}'
            #      .format(phase, len(xi_list),
            #              len(xi_0_list), len(xi_2_list)))
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


def do_cov(model_name, N_sub=3, cov_phases=range(16)):

    # calculate cov combining all phases, 16 * N_sub^3 * N_reals
    for ell in [0, 2]:
        paths = []
        for phase in cov_phases:
            sim_name = '{}_{:02}-{}'.format(sim_name_prefix, cosmology, phase)
            paths = paths + glob(os.path.join(
                        save_dir, sim_name, 'z{}-r*'.format(redshift),
                        '{}-cross_*-xi_{}.txt'.format(model_name, ell)))
        print('Calculating covariance matrix using {} xi samples from all'
              ' phases for l = {}...'
              .format(len(paths), ell))
        # read in all xi files for all phases
        xi_list = [np.loadtxt(path)[:, 1] for path in paths]
        # as cov gets smaller, chisq gets bigger, contour gets smaller
        cov = xi1d_list_to_cov(xi_list) / (np.power(N_sub, 3)-1)
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
    cov_monoquad = xi1d_list_to_cov(xi_list) / (np.power(N_sub, 3)-1)
    # save cov
    filepath = os.path.join(  # save to '-combined' folder for all phases
            save_dir,
            sim_name_prefix + '_' + str(cosmology).zfill(2) + '-combined',
            'z{}'.format(redshift),
            '{}-cross-xi_monoquad-cov.txt'.format(model_name, ell))
    np.savetxt(filepath, cov_monoquad, fmt=txtfmt)
    print('Monoquad covariance matrix saved to: ', filepath)


def do_realisations(halocat, model_name, N_reals=16):
    '''
    generates n HOD realisations for a given (phase, model)
    then co-add them and get a single set of xi data
    the rest of the programme will work as if there is only 1 realisation
    '''
    sim_name = halocat.header['SimName']
    phase = halocat.ZD_Seed
    model = initialise_model(halocat.redshift, model_name,
                             halo_m_prop=halo_m_prop)
    model.N_cut = N_cut
    model.c_median_poly = np.poly1d(
        np.loadtxt(os.path.join(save_dir, 'c_median_poly.txt')))
    # create n_real realisations of the given HOD model
    xi_list_reals = []
    xi_0_list_reals = []
    xi_2_list_reals = []
    for r in range(N_reals):
        # add useful model properties here
        model.r = r
        # check if current realisation, r, exists;
        filedir = os.path.join(save_dir, sim_name,
                               'z{}-r{}'.format(redshift, r))
        gt_path = os.path.join(filedir,
                               '{}-galaxy_table.csv'.format(model_name))
        if not overwrite_mode and os.path.isfile(gt_path):
            print('---\n'
                  '{} of {} realisations for model {} exists. \n'
                  '---'
                  .format(r+1, N_reals, model_name))
            xi_s_mu = np.loadtxt(os.path.join(filedir, '{}-auto-xi_s_mu.txt'
                                              .format(model_name)))
            xi_0 = np.loadtxt(os.path.join(filedir, '{}-auto-xi_0.txt'
                                           .format(model_name)))
            xi_2 = np.loadtxt(os.path.join(filedir, '{}-auto-xi_2.txt'
                                           .format(model_name)))
        else:
            print('---\n'
                  'Generating {} of {} realisations for model {} ...\n'
                  '---'
                  .format(r+1, N_reals, model_name))
            # set random seed using phase and realisation index r, model indep
            seed = phase*100 + r
            np.random.seed(seed)
            model = populate_model(halocat, model,
                                   add_rsd=add_rsd, N_threads=N_threads)
            # save galaxy table meta
            N_cen = np.sum(model.mock.galaxy_table['gal_type'] == 'centrals')
            N_sat = np.sum(model.mock.galaxy_table['gal_type'] == 'satellites')
            N_gal = len(model.mock.galaxy_table)
            if os.path.isfile(os.path.join(save_dir, 'galaxy_table_meta.csv')):
                gt_meta = table.Table.read(
                        os.path.join(save_dir, 'galaxy_table_meta.csv'))
            else:
                # create a csv file to store galaxy table metadata
                gt_meta = table.Table(
                        names=('model name', 'phase', 'realisation', 'seed',
                               'N centrals', 'N satellites', 'N galaxies'),
                        dtype=('S30', 'i4', 'i4', 'i4', 'i4', 'i4', 'i4'))
            gt_meta.add_row((model_name, phase, r, seed, N_cen, N_sat, N_gal))
            gt_meta.write(os.path.join(save_dir, 'galaxy_table_meta.csv'),
                          format='ascii.fast_csv', overwrite=True)
            # save galaxy table
            if save_hod_realisation:
                print('Saving galaxy table to: {} ...'.format(gt_path))
                if not os.path.exists(os.path.dirname(gt_path)):
                    os.makedirs(os.path.dirname(gt_path))
                model.mock.galaxy_table.write(gt_path,
                                              format='ascii.fast_csv',
                                              overwrite=overwrite_mode)

            do_auto_count(model, do_DD=True, do_RR=True)
            do_subcross_count(model, N_sub=N_sub, do_DD=True, do_DR=True)
            # always rebin counts and calculate xi in case bins change
            xi_s_mu, xi_0, xi_2 = do_auto_correlation(model)
            do_subcross_correlation(model, N_sub=N_sub)
        # collect auto-xi to coadd all realisations for the current phase
        xi_list_reals.append(xi_s_mu)
        xi_0_list_reals.append(xi_0)
        xi_2_list_reals.append(xi_2)

    print('Co-adding auto-correlation for {} realisations...'.format(N_reals))
    xi_s_mu_ca, _, xi_0_ca, _, xi_2_ca, _ = \
        coadd_xi_list(xi_list_reals, xi_0_list_reals, xi_2_list_reals)
    filedir = os.path.join(save_dir, sim_name, 'z{}'.format(redshift))
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


def fit_c_median(phases=range(16)):

    '''
    median concentration as a function of log halo mass in Msun/h
    c_med(log(m)) = poly(log(m))

    '''
    # check if output file already exists
    if os.path.isfile(os.path.join(save_dir, 'c_median_poly.txt')):
        return None
    # load 16 phases for the given cosmology and redshift
    print('Loading halo catalogues...')
    halocats = abacus_ht.make_catalogs(
        sim_name=sim_name_prefix, products_dir=prod_dir,
        redshifts=[redshift], cosmologies=[cosmology], phases=phases,
        halo_type=halo_type,
        load_halo_ptcl_catalog=False,  # this loads 10% particle subsamples
        load_ptcl_catalog=False,  # this loads uniform subsamples, dnw
        load_pids='auto')[0][0]
    htabs = [halocat.halo_table for halocat in halocats]
    htab = table.vstack(htabs)  # combine all phases into one halo table
    mask = (htab['halo_N'] >= N_cut) & (htab['halo_upid'] == -1)
    htab = htab[mask]   # mass cut
    halo_logm = np.log10(htab['halo_mvir'].data)
    halo_nfw_conc = htab['halo_rvir'].data / htab['halo_klypin_rs'].data
    # set up bin edges, the last edge larger than the most massive halo
    logm_bins = np.linspace(np.min(halo_logm), np.max(halo_logm)+0.001,
                            endpoint=True, num=100)
    logm_bins_centre = (logm_bins[:-1] + logm_bins[1:]) / 2
    N_halos, _ = np.histogram(halo_logm, bins=logm_bins)
    c_median = np.zeros(len(logm_bins_centre))
    c_median_std = np.zeros(len(logm_bins_centre))
    for i in range(len(logm_bins_centre)):
        mask = (logm_bins[i] <= halo_logm) & (halo_logm < logm_bins[i+1])
        c_median[i] = np.median(halo_nfw_conc[mask])
        c_median_std[i] = np.std(halo_nfw_conc[mask])
    # when there is 0 data point in bin, median = std = nan, remove
    # when there is only 1 data point in bin, std = 0 and w = inf
    # when there are few data points, std very small (<1), w large
    # rescale weight by sqrt(N-1)
    mask = c_median_std > 0
    N_halos, logm_bins_centre, c_median, c_median_std = \
        N_halos[mask], logm_bins_centre[mask], \
        c_median[mask], c_median_std[mask]
    weights = 1/np.array(c_median_std)*np.sqrt(N_halos-1)
    coefficients = np.polyfit(logm_bins_centre, c_median, 3, w=weights)
    print('Polynomial found as : {}.'.format(coefficients))
    # save coefficients to txt file
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.savetxt(os.path.join(save_dir, 'c_median_poly.txt'),
               coefficients, fmt=txtfmt)
    c_median_poly = np.poly1d(coefficients)
    # plot fit
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(logm_bins_centre, c_median, yerr=c_median_std,
                capsize=3, capthick=0.5, fmt='.', ms=4, lw=0.2)
    ax.plot(logm_bins_centre, c_median_poly(logm_bins_centre), '-')
    ax.set_title('Median Concentration Polynomial Fit from {} Boxes'
                 .format(len(phases)))
    fig.savefig(os.path.join(save_dir, 'c_median_poly.pdf'))

    return halocats


def run_baofit_parallel(baofit_phases=range(16)):

    from baofit import baofit

    filedir = os.path.join(
            save_dir,
            sim_name_prefix + '_' + str(cosmology).zfill(2) + '-combined',
            'z{}'.format(redshift))
    list_of_inputs = []
    for model_name in model_names:
        for phase in baofit_phases:  # list of inputs for parallelisation
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
    pool = Pool(N_threads)
    pool.map(baofit, list_of_inputs)
    pool.close()


if __name__ == "__main__":

#    halocats = fit_c_median(phases=phases)
    for phase in phases:
        halocat = make_halocat(phase)
        halocat = process_rockstar_halocat(halocat, N_cut)
        for model_name in model_names:
            do_realisations(halocat, model_name, N_reals=N_reals)

    for model_name in model_names:
        do_coadd_phases(model_name, coadd_phases=phases)
        do_cov(model_name, N_sub=N_sub, cov_phases=phases)

    run_baofit_parallel(baofit_phases=phases)
