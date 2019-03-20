# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 17:30:51 2017

@author: Duan Yutong (dyt@physics.bu.edu)

"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
from contextlib import closing
import os
from glob import glob
from itertools import product
from functools import partial
import subprocess
import traceback
import numpy as np
import numpy.lib.recfunctions as rfn
from scipy import optimize
from astropy import table
from halotools.mock_observables import tpcf_multipole
from halotools.mock_observables.two_point_clustering.s_mu_tpcf import \
    spherical_sector_volume
from halotools.mock_observables.two_point_clustering.rp_pi_tpcf import \
    cylinder_volume
# from halotools.utils import add_halo_hostid
from Corrfunc.theory.DDsmu import DDsmu
from Corrfunc.theory.DDrppi import DDrppi
from Corrfunc.utils import convert_rp_pi_counts_to_wp
from abacus_hod import MyPool, fit_c_median, process_rockstar_halocat, \
    initialise_model, populate_model
from tqdm import tqdm
from shutil import copyfile
from Abacus import Halotools as abacus_ht  # Abacus
from baofit import baofit


# %% custom settings

#sim_name_prefix = 'emulator_1100box_planck'
#tagout = '-mmatter'  # 'norsd'  # '# 'matter'  # 'z0.5'
#phases = range(16)  # range(16)  # [0, 1] # list(range(16))
#sim_name_prefix = 'AbacusCosmos_1100box_planck'
#tagout = '-mmatter'
#phases = range(20)
sim_name_prefix = 'joint_1100box_planck'
tagout = 'mmatter'
phases = range(36)

model_names = ['gen_base1', 'gen_base6', 'gen_base7',
               'gen_ass1',  'gen_ass1_n',
               'gen_ass2',  'gen_ass2_n',
               'gen_vel2',  'gen_vel1',
               'gen_s1',    'gen_s1_n',
               'gen_sv1',   'gen_sv1_n',
               'gen_sp1',   'gen_sp1_n']
model_names = ['matter']
reals = range(1)  # range(12)
N_threads = 24  # number of threads for a single realisation
do_create = False
do_coadd = False
do_optimise_fitter = False
do_fit = True

cosmology = 0  # one cosmology at a time instead of 'all'
redshift = 0.5  # one redshift at a time instead of 'all'
L = 1100  # boxsize in 1D
N_cut = 70  # number particle cut, 70 corresponds to 4e12 Msun
N_concurrent_reals = 4
N_sub = 3  # number of subvolumes per dimension
random_multiplier = 5
galaxy_bias = 2.23  # FFTcorr always unbiased, but need to match paircount cov
matter_subsample_fraction = 0.002

# %% flags
save_hod_realisation = True
use_rsd = False

# %% bin settings
step_s_bins = 5  # mpc/h, bins for fitting
step_mu_bins = 0.01
s_bins = np.arange(0, 150 + step_s_bins, step_s_bins)
s_bins_centre = (s_bins[:-1] + s_bins[1:]) / 2
mu_bins = np.arange(0, 1 + step_mu_bins, step_mu_bins)
s_bins_counts = np.arange(0, 151, 1)  # always count with s bin size 1
mu_max = 1.0  # for counting
n_mu_bins = mu_bins.size-1
pi_max = 120  # for counting
pi_bins = np.arange(0, pi_max+1, 1)
rp_bins = pi_bins

# %% fixed catalogue parameters
store_dir = '/home/dyt/store/'
save_dir = os.path.join(store_dir, sim_name_prefix+tagout)
recon_temp_dir = os.path.join(store_dir, 'recon/temp')
txtfmt = b'%.30e'
if sim_name_prefix == 'emulator_1100box_planck':
    prod_dir = \
        '/mnt/gosling2/bigsim_products/emulator_1100box_planck_products/'
elif sim_name_prefix == 'AbacusCosmos_1100box_planck':
    prod_dir = \
        '/mnt/gosling1/bigsim_products/AbacusCosmos_1100box_planck_products/'


# %% definitionss of statisical formulae and convenience functions
def analytic_random(ND1, NR1, ND2, NR2, boxsize, mode='smu',
                    s_bins=None, mu_bins=None, rp_bins=None, pi_bins=None):
    '''
    V is the global volume L^3
    for auto-correlation, we may set ND1 = NR1 = ND2 = NR2
    for cross-correlation, returns D1R2 and R1D2 for LS estimator
    '''
    V = boxsize**3
    if 'smu' in mode and s_bins is not None and mu_bins is not None:
        mu_bins_reverse_sorted = np.sort(mu_bins)[::-1]
        dv = spherical_sector_volume(s_bins, mu_bins_reverse_sorted)
        dv = np.diff(dv, axis=1)  # volume of wedges
        dv = np.diff(dv, axis=0)  # volume of wedge sector
    elif 'rppi' in mode and rp_bins is not None and pi_bins is not None:
        v = cylinder_volume(rp_bins, 2*pi_bins)  # volume of spheres
        dv = np.diff(np.diff(v, axis=0), axis=1)  # volume of annuli
    else:
        print('Input bins cannot be None for mode: {}'.format(mode))
        raise ValueError
    DR = ND1*NR2/V*dv
    RD = NR1*ND2/V*dv
    RR = NR1*NR2/V*dv  # calculate the random-random pairs
    return DR, RD, RR  # return raw pair counts


def ph_estimator(ND, NR, DD, RR):

    '''
    Peebles-Hauser
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


def ls_estimator(DD, DR, RD, RR_numerator, RR_denominator):
    '''
    generalised Landy & Szalay (1993) estimator for cross-correlation;
    reduces to (DD-2DR+RR)RR in case of auto-correlation if we set
    DR = RD
    all input counts should be normalised to sample sizes already
    '''
#    DD = DDraw / ND1 / ND2  # normalise pair-counts by sample sizes
#    DR = DRraw / ND1 / NR2
#    RD = RDraw / NR1 / ND2
#    RR = RRraw / NR1 / NR2
    return (DD - DR - RD + RR_numerator) / RR_denominator


def rebin_smu_counts(cts, npairs_key):

    '''
    input counts is the output table of DDSmu, saved as npy file
    output npairs is a 2D histogram of dimensions
    n_s_bins by n_mu_bins matrix (30 by 20 by default)
    savg is the paircount-weighted average of s in the bin, same dimension
    which may be inaccurate when bins are very fine and cmpty
    '''
    mask = cts['npairs'] != 0  # bins where npairs=0 yields NaN ratio
    ratio = np.around(cts['npairs'][mask] / cts[npairs_key][mask])
    assert np.all(ratio == np.median(ratio))
    npairs = np.zeros((s_bins.size-1, mu_bins.size-1), dtype=np.float64)
    savg = np.zeros((s_bins.size-1, mu_bins.size-1), dtype=np.float64)
    bins_included = 0
    for m, n in product(range(npairs.shape[0]), range(npairs.shape[1])):
        # smin and smax fields can be equal to bin edge
        # when they are equal, it's a bin right on the edge to be included
        mask = (
            (s_bins[m] <= cts['smin']) & (cts['smax'] <= s_bins[m+1]) &
            (mu_bins[n] < cts['mu_max']) & (cts['mu_max'] <= mu_bins[n+1]))
        arr = cts[mask]
        npairs[m, n] = arr['npairs'].sum()
        if npairs[m, n] == 0:
            savg[m, n] = (s_bins[m] + s_bins[m+1]) / 2
        else:
            savg[m, n] = np.sum(arr['savg'] * arr['npairs'] / npairs[m, n])
        bins_included += arr['npairs'].size

    try:
        assert cts.size == bins_included
        assert cts['npairs'].sum() == npairs.sum()
    except AssertionError as E:
        print('Total fine bins: {}. Total bins included: {}'
              .format(cts.size, bins_included))
        print(('Total raw npairs in fine bins: {:0.3f}. '
               'Total npairs after rebinning: {:0.3f}.')
              .format(cts['npairs'].sum(), npairs.sum()))
        raise E
#    indices = np.where(npairs == 0)
#    if indices[0].size != 0:  # and np.any(indices[0] != 0):
#        for i in range(indices[0].size):
#            print('({}, {}) bin is empty'
#                  .format(indices[0][i], indices[1][i]))
    return npairs/np.median(ratio), savg


def subvol_mask(x, y, z, i, j, k, L, N_sub):

    L_sub = L / N_sub  # length of subvolume box
    mask = ((L_sub * i < x) & (x <= L_sub * (i+1)) &
            (L_sub * j < y) & (y <= L_sub * (j+1)) &
            (L_sub * k < z) & (z <= L_sub * (k+1)))
    return x[mask], y[mask], z[mask]


def coadd_correlation(corr_list):

    '''
    input is a list of xi or wp samples
    array in the list is of shape (:, 2), and 1st column is r
    '''
    # print('Number of samples being coadded: ', len(corr_list))
    arr = np.array(corr_list)
    coadd = np.mean(arr, axis=0)
    err = np.std(arr, axis=0)
    if arr.shape[2] == 2:
        err[:, 0] = coadd[:, 0]
    return coadd, err


# %% functions for tasks

def make_halocat(phase, halo_type='Rockstar'):

    if halo_type == 'Rockstar':
        load_halo_ptcl = True
        load_all_ptcl = False
        halo_m_prop = 'halo_mvir'
    elif halo_type == 'FoF':
        load_halo_ptcl = False
        load_all_ptcl = True
        halo_m_prop = 'halo_m547m'
    print('---\n'
          + 'Importing phase {} {} halocat from: \n'.format(phase, halo_type)
          + '{}{}_{:02}-{}_products/z{}/'
            .format(prod_dir, sim_name_prefix, cosmology, phase, redshift)
          + '\n---')
    halocat = abacus_ht.make_catalogs(  # load one phase at a time to save ram
        sim_name=sim_name_prefix, products_dir=prod_dir,
        redshifts=[redshift], cosmologies=[cosmology], phases=[phase],
        halo_type=halo_type, load_pids=False,
        load_halo_ptcl_catalog=load_halo_ptcl, load_ptcl_catalog=load_all_ptcl,
        )[0][0][0]
    halocat.halo_m_prop = halo_m_prop
    # add_halo_hostid(halocat.halo_table, delete_possibly_existing_column=True)
    if halo_type == 'Rockstar':
        # fill in fields required by HOD models
        if 'halo_mvir' not in halocat.halo_table.keys():
            halocat.halo_table['halo_mvir'] = halocat.halo_table['halo_m']
            print('halo_mvir field missing, assuming halo_mvir = halo_m')
        if 'halo_rvir' not in halocat.halo_table.keys():
            halocat.halo_table['halo_rvir'] = halocat.halo_table['halo_r']
            print('halo_rvir field missing, assuming halo_rvir = halo_r')
        # setting NFW concentration using scale radius
        # klypin_rs is more stable and better for small halos
        # can be improved following https://arxiv.org/abs/1709.07099
        halocat.halo_table['halo_nfw_conc'] = (
            halocat.halo_table['halo_rvir']
            / halocat.halo_table['halo_klypin_rs'])
    elif halo_type == 'FoF':
        halocat.halo_table['halo_mvir'] = halocat.halo_table['halo_m547m']
        halocat.halo_table['halo_rvir'] = halocat.halo_table['halo_r547m']
        halocat.halo_table['halo_nfw_conc'] = 0
    return halocat


def do_auto_count(model, mode='smu-pre-recon'):

    '''
    mode is a string that contains 'smu' or 'wp', and can include
    pre/post-recon tag which gets saved as part of filename
    '''
    # do pair counting in 1 Mpc bins
    sim_name = model.mock.header['SimName']
    redshift = model.mock.redshift
    model_name = model.model_name
    ND = model.mock.ND
    if model.model_type == 'matter':
        tb = model.mock.ptcl_table
    else:
        tb = model.mock.galaxy_table
    x, y, z = tb['x'], tb['y'], tb[model.zfield]
    filedir = os.path.join(save_dir, sim_name,
                           'z{}-r{}'.format(redshift, model.r))
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    if 'post-recon' in mode:
        assert hasattr(model.mock, 'shifted_randoms')
        use_shifted_randoms = True
        sr_list = np.array_split(model.mock.shifted_randoms, random_multiplier)
    elif 'pre-recon' in mode:
        use_shifted_randoms = False
    # c_api_timer needs to be set to false to make sure the DD variable
    # and the npy file saved are exactly the same and can be recovered
    # otherwise, with c_api_timer enabled, file is in "object" format,
    # not proper datatypes, and will cause indexing issues
    if 'smu' in mode:
        # save counting results as original structured array in npy format
        DDnpy = DDsmu(1, N_threads, s_bins_counts, mu_max, n_mu_bins,
                      x, y, z, periodic=True, verbose=False, boxsize=L,
                      output_savg=True, c_api_timer=False)
    elif 'rppi' in mode:  # also returns wp in addition to counts
        DDnpy = DDrppi(1, N_threads, pi_max, rp_bins, x, y, z,
                       periodic=True, verbose=False, boxsize=L,
                       output_rpavg=True, c_api_timer=False)
    DDnpy = rfn.append_fields(DDnpy, ['DD'],
                              [DDnpy['npairs'].astype(np.float64)/ND/ND],
                              usemask=False)
    np.save(os.path.join(filedir, '{}-auto-paircount-DD-{}.npy'
                         .format(model_name, mode)),
            DDnpy)
    if use_shifted_randoms:
        sr = model.mock.shifted_randoms  # random shifted array
        xr, yr, zr = sr[:, 0], sr[:, 1], sr[:, 2]
        NR = xr.size
        if 'smu' in mode:
            DRnpy = DDsmu(0, N_threads, s_bins_counts, mu_max, n_mu_bins,
                          x, y, z, X2=xr, Y2=yr, Z2=zr,
                          periodic=True, verbose=False, boxsize=L,
                          output_savg=True, c_api_timer=False)
        elif 'rppi' in mode:
            DRnpy = DDrppi(0, N_threads, pi_max, rp_bins, x, y, z,
                           X2=xr, Y2=yr, Z2=zr, periodic=True, verbose=False,
                           boxsize=L, output_rpavg=True, c_api_timer=False)
        DRnpy = rfn.append_fields(
            DRnpy, ['DR'], [DRnpy['npairs'].astype(np.float64)/ND/NR],
            usemask=False)
        np.save(os.path.join(filedir,
                             '{}-auto-paircount-DR-{}-sr.npy'
                             .format(model_name, mode)),
                DRnpy)
        for n in range(random_multiplier):  # for each split copy of R
            sr = sr_list[n]  # this is one of the N copies from R split
            xr, yr, zr = sr[:, 0], sr[:, 1], sr[:, 2]
            NR = xr.size
            if 'smu' in mode:
                RRnpy = DDsmu(1, N_threads, s_bins_counts, mu_max, n_mu_bins,
                              xr, yr, zr, periodic=True, verbose=False,
                              output_savg=True, boxsize=L, c_api_timer=False)
            elif 'rppi' in mode:
                RRnpy = DDrppi(1, N_threads, pi_max, rp_bins, xr, yr, zr,
                               periodic=True, verbose=False, boxsize=L,
                               output_rpavg=True, c_api_timer=False)
            RRnpy = rfn.append_fields(
                RRnpy, ['RR'],
                [RRnpy['npairs'].astype(np.float64)/NR/NR],
                usemask=False)
            np.save(os.path.join(
                    filedir, '{}-auto_{}-paircount-RR-{}-sr.npy'
                    .format(model_name, n, mode)),
                    RRnpy)


def do_auto_correlation(model_name, phase, r, mode='smu-pre-recon-std',
                        use_shifted_randoms=False):

    if 'pre-recon' in mode:
        # use_shifted_randoms = False
        rand_type = 'ar'
    elif 'post-recon' in mode:
        # use_shifted_randoms = True
        rand_type = 'sr'
    sim_name = '{}_{:02}-{}'.format(sim_name_prefix, cosmology, phase)
    filedir = os.path.join(save_dir, sim_name, 'z{}-r{}'.format(redshift, r))
    DDnpy = np.load(os.path.join(filedir,
                                 '{}-auto-paircount-DD-{}.npy'
                                 .format(model_name, mode)))
    if 'smu' in mode:
        DD, _ = rebin_smu_counts(DDnpy, 'DD')
        np.savetxt(os.path.join(filedir, '{}-auto-paircount-DD-{}-rebinned.txt'
                                .format(model_name, mode)),
                   DD, fmt=txtfmt)
    elif 'rppi' in mode:  # no rebinning necessary
        # even for post-recon, rppi and wp are calculated with analytic randoms
        # counts not normalised becuase we are using corrfunc for wp conversion
        # ND = np.median(np.around(np.sqrt(DDnpy['npairs'] / DDnpy['DD'])))
        DD = DDnpy['DD'].reshape(len(rp_bins)-1, -1)

    DR_ar, _, RR_ar = analytic_random(1, 1, 1, 1, L, mode=mode,
                                      s_bins=s_bins, mu_bins=mu_bins,
                                      rp_bins=rp_bins, pi_bins=pi_bins)
    assert DD.shape == DR_ar.shape == RR_ar.shape
    if use_shifted_randoms:
        DR = np.load(os.path.join(filedir,
                                  '{}-auto-paircount-DR-{}-sr.npy'
                                  .format(model_name, mode)))['DR']
        RR_list = []
        for n in range(random_multiplier):
            RRnpy = np.load(os.path.join(
                    filedir, '{}-auto_{}-paircount-RR-{}-sr.npy'
                    .format(model_name, n, mode)))
            if 'smu' in mode:
                RR, _ = rebin_smu_counts(RRnpy, 'RR')
                np.savetxt(os.path.join(
                        filedir, '{}-auto_{}-paircount-RR-{}-sr_rebinned.txt'
                        .format(model_name, n, mode)),
                    RR, fmt=txtfmt)
            elif 'rppi' in mode:
                RR = RRnpy['RR']
            RR_list.append(RR)
        RR = np.mean(RR_list, axis=0)
    else:
        DR, RR = DR_ar, RR_ar
        np.savetxt(os.path.join(filedir,
                                '{}-auto-paircount-DR-{}-{}.txt'
                                .format(model_name, mode, rand_type)),
                   DR_ar, fmt=txtfmt)
        np.savetxt(os.path.join(filedir,
                                '{}-auto-paircount-RR-{}-{}.txt'
                                .format(model_name, mode, rand_type)),
                   RR_ar, fmt=txtfmt)
    xi = ls_estimator(DD, DR, DR, RR, RR_ar)
    if 'smu' in mode:
        xi_0 = tpcf_multipole(xi, mu_bins, order=0)
        xi_2 = tpcf_multipole(xi, mu_bins, order=2)
        xi_0_txt = np.vstack([s_bins_centre, xi_0]).T
        xi_2_txt = np.vstack([s_bins_centre, xi_2]).T
        for output, tag in zip([xi, xi_0_txt, xi_2_txt],
                               ['xi', 'xi_0', 'xi_2']):
            np.savetxt(os.path.join(filedir, '{}-auto-{}-{}-{}.txt'
                                    .format(model_name, tag, mode, rand_type)),
                       output, fmt=txtfmt)
    elif 'rppi' in mode:
        wp = convert_rp_pi_counts_to_wp(
            1, 1, 1, 1, DD.flatten(), DR.flatten(), DR.flatten(),
            RR.flatten(), len(rp_bins)-1, pi_max, estimator='LS')
        wp_txt = np.vstack([(rp_bins[:-1] + rp_bins[1:])/2, wp]).T
        np.savetxt(os.path.join(filedir, '{}-auto-xi-{}-{}.txt'
                                .format(model_name, mode, rand_type)),
                   xi, fmt=txtfmt)
        np.savetxt(os.path.join(filedir, '{}-auto-wp-{}-{}.txt'
                                .format(model_name, mode, rand_type)),
                   wp_txt, fmt=txtfmt)


def do_subcross_count(model, mode='smu-post-recon-std'):

    '''
    cross-correlation between 1/N_sub^3 of a box and the whole box
    can be pre/post-reconstruction data
    '''
    sim_name = model.mock.header['SimName']
    model_name = model.model_name
    redshift = model.mock.redshift
    if model.model_type == 'matter':
        tb = model.mock.ptcl_table
    else:
        tb = model.mock.galaxy_table
    x1, y1, z1 = tb['x'], tb['y'], tb[model.zfield]
    ND1 = model.mock.ND  # number of galaxies in entire volume of periodic box
    filedir = os.path.join(save_dir, sim_name,
                           'z{}-r{}'.format(redshift, model.r))
    # split the (already shuffled) randoms into N copies, each of size NData
    if 'post-recon' in mode:
        assert hasattr(model.mock, 'shifted_randoms')
        use_shifted_randoms = True
        sr_list = np.array_split(model.mock.shifted_randoms, random_multiplier)
    elif 'pre-recon' in mode:
        use_shifted_randoms = False
    for i, j, k in product(range(N_sub), repeat=3):
        linind = i*N_sub**2 + j*N_sub + k  # linearised index of subvolumes
        print('r = {:2d}, subcross counting subvol {:2d}...'
              .format(model.r, linind))
        x2, y2, z2 = subvol_mask(x1, y1, z1,
                                 i, j, k, L, N_sub)
        ND2 = x2.size  # number of galaxies in the subvolume
        DDnpy = DDsmu(0, N_threads, s_bins_counts, mu_max, n_mu_bins,
                      x1, y1, z1, X2=x2, Y2=y2, Z2=z2,
                      periodic=True, output_savg=True, verbose=False,
                      boxsize=L, c_api_timer=False)
        DDnpy = rfn.append_fields(
            DDnpy, ['DD'], [DDnpy['npairs'].astype(np.float64)/ND1/ND2],
            usemask=False)
        np.save(os.path.join(filedir, '{}-cross_{}-paircount-DD-{}.npy'
                             .format(model_name, linind, mode)),
                DDnpy)
        if use_shifted_randoms:
            # calculate D2Rbox by brute force sampling, where
            # Rbox sample is homogeneous in volume of box
            # force float32 (a.k.a. single precision float in C)
            # to be consistent with galaxy table precision
            # otherwise corrfunc raises an error
            sr = model.mock.shifted_randoms  # random shifted array
            xr1, yr1, zr1 = sr[:, 0], sr[:, 1], sr[:, 2]
            xr2, yr2, zr2 = subvol_mask(xr1, yr1, zr1, i, j, k, L, N_sub)
            NR1, NR2 = xr1.size, xr2.size
            DRnpy = DDsmu(0, N_threads, s_bins_counts, mu_max, n_mu_bins,
                          x1, y1, z1, X2=xr2, Y2=yr2, Z2=zr2,
                          periodic=True, verbose=False, output_savg=True,
                          boxsize=L, c_api_timer=False)
            RDnpy = DDsmu(0, N_threads, s_bins_counts, mu_max, n_mu_bins,
                          xr1, yr1, zr1, X2=x2, Y2=y2, Z2=z2,
                          periodic=True, verbose=False, output_savg=True,
                          boxsize=L, c_api_timer=False)
            DRnpy = rfn.append_fields(
                DRnpy, ['DR'], [DRnpy['npairs'].astype(np.float64)/ND1/NR2],
                usemask=False)
            RDnpy = rfn.append_fields(
                RDnpy, ['RD'], [RDnpy['npairs'].astype(np.float64)/NR1/ND2],
                usemask=False)
            for npy, tag in zip([DRnpy, RDnpy], ['DR', 'RD']):
                np.save(os.path.join(
                            filedir,
                            '{}-cross_{}-paircount-{}-{}-sr.npy'
                            .format(model_name, linind, tag, mode)),
                        npy)
            for n in range(random_multiplier):  # for each split copy of R
                sr = sr_list[n]  # this is one of the N copies from R split
                xr1, yr1, zr1 = sr[:, 0], sr[:, 1], sr[:, 2]
                xr2, yr2, zr2 = subvol_mask(xr1, yr1, zr1, i, j, k, L, N_sub)
                NR1, NR2 = xr1.size, xr2.size
                RRnpy = DDsmu(0, N_threads, s_bins_counts, mu_max, n_mu_bins,
                              xr1, yr1, zr1, X2=xr2, Y2=yr2, Z2=zr2,
                              periodic=True, verbose=False, output_savg=True,
                              boxsize=L, c_api_timer=False)
                RRnpy = rfn.append_fields(
                    RRnpy, ['RR'],
                    [RRnpy['npairs'].astype(np.float64)/NR1/NR2],
                    usemask=False)
                np.save(os.path.join(
                        filedir, '{}-cross_{}_{}-paircount-RR-{}-sr.npy'
                        .format(model_name, linind, n, mode)),
                        RRnpy)
        elif 'pre-recon' in mode:  # use_analytic_randoms
            pass
        else:
            raise ValueError('Post-reconstruction cross-correlation counting'
                             ' requires shifted randoms for mode: {}'
                             .format(mode))


def do_subcross_correlation(linind, phase, r, model_name,
                            mode='smu-post-recon-std'):

    sim_name = '{}_{:02}-{}'.format(sim_name_prefix, cosmology, phase)
    filedir = os.path.join(save_dir, sim_name, 'z{}-r{}'.format(redshift, r))
    # print('r = {:2d}, x-correlation subvolume {}...'.format(r, linind))
    DDnpy = np.load(os.path.join(filedir, '{}-cross_{}-paircount-DD-{}.npy'
                                 .format(model_name, linind, mode)))
    DD, _ = rebin_smu_counts(DDnpy, 'DD')  # re-bin and re-weight
    np.savetxt(os.path.join(filedir,
                            '{}-cross_{}-paircount-DD-{}-rebinned.txt'
                            .format(model_name, linind, mode)),
               DD, fmt=txtfmt)
    if 'pre-recon' in mode:  # pre-recon covariance, use analytic randoms
        DR, RD, RR = analytic_random(1, 1, 1, 1, L, mode='smu',
                                     s_bins=s_bins, mu_bins=mu_bins)
        for output, tag in zip([DR, RD, RR], ['DR', 'RD', 'RR']):
            np.savetxt(os.path.join(
                    filedir, '{}-cross_{}-paircount-{}-{}-ar.txt'
                    .format(model_name, linind, tag, mode)),
                output, fmt=txtfmt)
        xi_smu = ls_estimator(DD, DR, RD, RR, RR)
        rand_type = 'ar'
    elif 'post-recon' in mode:
        DRnpy = np.load(os.path.join(filedir,
                                     '{}-cross_{}-paircount-DR-{}-sr.npy'
                                     .format(model_name, linind, mode)))
        RDnpy = np.load(os.path.join(filedir,
                                     '{}-cross_{}-paircount-RD-{}-sr.npy'
                                     .format(model_name, linind, mode)))
        DR, _ = rebin_smu_counts(DRnpy, 'DR')
        RD, _ = rebin_smu_counts(RDnpy, 'RD')
        for txt, tag in zip([DR, RD], ['DR', 'RD']):
            np.savetxt(os.path.join(
                    filedir, '{}-cross_{}-paircount-{}-{}-sr_rebinned.txt'
                    .format(model_name, linind, tag, mode)),
                txt, fmt=txtfmt)
        RR_list = []
        for n in range(random_multiplier):
            RRnpy = np.load(os.path.join(
                filedir, '{}-cross_{}_{}-paircount-RR-{}-sr.npy'
                .format(model_name, linind, n, mode)))
            RR, _ = rebin_smu_counts(RRnpy, 'RR')
            RR_list.append(RR)
        RR = np.mean(RR_list, axis=0)  # shifted, could be co-adding zeros
        np.savetxt(os.path.join(
                filedir, '{}-cross_{}-paircount-RR-{}-sr_rebinned.txt'
                .format(model_name, linind, mode)),
            RR, fmt=txtfmt)
        # use analytic RR as denominator in LS estimator, numerator is shifted
        _, _, RR_ar = analytic_random(1, 1, 1, 1, L, mode='smu',
                                      s_bins=s_bins, mu_bins=mu_bins)
        np.savetxt(os.path.join(filedir, '{}-cross_{}-paircount-RR-{}-ar.txt'
                                .format(model_name, linind, mode)),
                   RR_ar, fmt=txtfmt)
        assert DD.shape == DR.shape == RD.shape == RR.shape == RR_ar.shape
        xi_smu = ls_estimator(DD, DR, RD, RR, RR_ar)
        rand_type = 'sr'
    else:
        print('Mode error for: {}'.format(mode))
    # calculate cross-correlation from counts using ls estimator
    xi_0 = tpcf_multipole(xi_smu, mu_bins, order=0)
    xi_2 = tpcf_multipole(xi_smu, mu_bins, order=2)
    xi_0_txt = np.vstack([s_bins_centre, xi_0]).T
    xi_2_txt = np.vstack([s_bins_centre, xi_2]).T
    for output, tag in zip([xi_smu, xi_0_txt, xi_2_txt],
                           ['xi', 'xi_0', 'xi_2']):
        np.savetxt(os.path.join(
                filedir, '{}-cross_{}-{}-{}-{}.txt'
                .format(model_name, linind, tag, mode, rand_type)),
            output, fmt=txtfmt)


def do_galaxy_table(r, phase, model_name, overwrite=False):

    try:
        sim_name = '{}_{:02}-{}'.format(sim_name_prefix, cosmology, phase)
        if model_name == 'matter':
            return
        filedir = os.path.join(save_dir, sim_name,
                               'z{}-r{}'.format(redshift, r))
        gt_path = os.path.join(filedir,
                               '{}-galaxy_table.csv'.format(model_name))
        if os.path.isfile(gt_path) and not overwrite:
            return
        # global halocat
        assert halocat.ZD_Seed == phase or halocat.ZD_Seed == 100 + phase
        seed = phase*100 + r
        # all realisations in parallel, each model needs to be instantiated
        model = initialise_model(redshift, model_name,
                                 halo_m_prop=halocat.halo_m_prop)
        model.N_cut = N_cut  # add useful model properties here
        model.c_median_poly = np.poly1d(np.loadtxt(os.path.join(
                store_dir, sim_name_prefix, 'c_median_poly.txt')))
        model.r = r
        # random seed w/ phase and realisation index r, model independent
        print('r = {:2d} for phase {}, model {} starting...'
              .format(r, phase, model_name))
        model = populate_model(
            halocat, model, gt_path=(not overwrite)*gt_path,
            matter_subsample_fraction=matter_subsample_fraction)
        gt = model.mock.galaxy_table
        # save galaxy table if it's not already loaded from disk
        if save_hod_realisation and not model.mock.gt_loaded:
            print('r = {:2d}, saving galaxy table...'.format(r))
            try:
                os.makedirs(os.path.dirname(gt_path))
            except OSError:
                pass
            gt.write(gt_path, format='ascii.fast_csv', overwrite=True)
            # save galaxy table metadata to sim directory, one file each
            # to be combined later to avoid threading conflicts
            N_cen = np.sum(gt['gal_type'] == 'central')
            N_sat = np.sum(gt['gal_type'] == 'satellite')
            assert N_cen + N_sat == model.mock.ND
            gt_meta = table.Table(
                    [[model_name], [phase], [r], [seed],
                     [N_cen], [N_sat], [model.mock.ND]],
                    names=('model name', 'phase', 'realisation', 'seed',
                           'N centrals', 'N satellites', 'N galaxies'),
                    dtype=('S30', 'i4', 'i4', 'i4', 'i4', 'i4', 'i4'))
            if not os.path.exists(os.path.join(save_dir,
                                               'galaxy_table_meta')):
                os.makedirs(os.path.join(save_dir, 'galaxy_table_meta'))
            gt_meta.write(os.path.join(
                    save_dir, 'galaxy_table_meta',
                    '{}-z{}-{}-r{}.csv'
                    .format(model_name, redshift, phase, r)),
                format='ascii.fast_csv', overwrite=True)
    except Exception as E:
        print('Exception caught in worker thread r = {:2d}'.format(r))
        traceback.print_exc()
        raise E


def do_realisation(r, phase, model_name, overwrite=False,
                   do_pre_auto_smu=False, do_pre_auto_rppi=True,
                   do_pre_auto_fft=True, do_pre_cross=True,
                   do_recon_std=True, do_post_auto_rppi=True,
                   do_post_cross=True, do_recon_ite=True,
                   do_pre_auto_corr=True, do_post_auto_corr=True,
                   do_pre_cross_corr=True, do_post_cross_corr=True):

    # global halocat
    try:  # this helps catch exception in a worker thread of multiprocessing
        seed = phase*100 + r
        sim_name = '{}_{:02}-{}'.format(sim_name_prefix, cosmology, phase)
        filedir = os.path.join(save_dir, sim_name,
                               'z{}-r{}'.format(redshift, r))
        if not os.path.exists(filedir):
            try:
                os.makedirs(filedir)
            except OSError:
                assert os.path.exists(filedir)
        gt_path = os.path.join(filedir, '{}-galaxy_table.csv'
                               .format(model_name))
        gt_recon_path = os.path.join(filedir,
                                     '{}-galaxy_table-post-recon-std.csv'
                                     .format(model_name))
        pathD = os.path.join(recon_temp_dir,
                             '{}-file_D-{}_rec.dat'
                             .format(model_name, seed))
        pathR = os.path.join(recon_temp_dir,
                             '{}-file_R-{}_rec.dat'
                             .format(model_name, seed))
        path_sr = os.path.join(filedir,
                               '{}-shifted_randoms-post-recon-std.npy'
                               .format(model_name))
        if os.path.isfile(gt_path) and not overwrite:
            temp = os.path.join(filedir, model_name)
            if len(glob(temp+'-auto-paircount-DD-smu-pre-recon.npy')) == 1:
                do_pre_auto_smu = False
            if len(glob(temp+'-auto-paircount-DD-rppi-pre-recon.npy')) == 1:
                do_pre_auto_rppi = False
            if len(glob(temp+'-auto-fftcorr_N-pre-recon*.txt')) == 1:
                do_pre_auto_fft = False
            if len(glob(temp+'-cross_*-DD-smu-pre-recon*.npy')) == 27:
                do_pre_cross = False
            if len(glob(temp+'-auto-fftcorr_N-post-recon-std*.txt')) == 1 \
                    and os.path.isfile(gt_recon_path) \
                    and os.path.isfile(path_sr):
                do_recon_std = False
            if len(glob(temp+'-auto-*-DD-rppi-post-recon-std*.npy')) == 1:
                do_post_auto_rppi = False
            if len(glob(temp+'-cross_*-DD-smu-post-recon-std*.npy')) == 27:
                do_post_cross = False
            if len(glob(temp+'-auto-fftcorr_N-post-recon-ite*.txt')) == 1:
                do_recon_ite = False
        flags = [do_pre_auto_smu, do_pre_auto_rppi, do_pre_auto_fft,
                 do_pre_cross,
                 do_recon_std, do_post_auto_rppi, do_post_cross, do_recon_ite]
        if not np.any(flags):
            print('r = {:2d}, all counting done,'.format(r))
            do_count = False
        else:
            do_count = True
        if do_count:  # everything within is counting, require loading halocat
            assert halocat.ZD_Seed == phase or halocat.ZD_Seed == 100 + phase
            # all realisations in parallel, each model needs to be instantiated
            model = initialise_model(redshift, model_name,
                                     halo_m_prop=halocat.halo_m_prop)
            model.r = r
            if use_rsd or model_name == 'matter':
                model.zfield = 'z'
            else:
                model.zfield = 'zreal'
            # random seed w/ phase and realisation index r, model independent
            print('r = {:2d} for phase {}, model {} starting...'
                  .format(r, phase, model_name))
            model = populate_model(
                halocat, model, gt_path=gt_path,
                matter_subsample_fraction=matter_subsample_fraction)
            if model_name == 'matter':
                tb = model.mock.ptcl_table
                sample_type = 'ptcl'
                bias = 1
            else:
                tb = model.mock.galaxy_table
                sample_type = 'gal'
                bias = galaxy_bias
            if do_pre_auto_smu:  # pre-recon auto-correlation pair-counting
                print('r = {:2d}, pre-recon smu pair-counting...'.format(r))
                do_auto_count(model, mode='smu-pre-recon')
            if do_pre_auto_rppi:
                print('r = {:2d}, pre-recon rppi pair-counting...'.format(r))
                do_auto_count(model, mode='rppi-pre-recon')
            if do_pre_cross:
                print('r = {:2d}, counting for pre-recon x-corr...'.format(r))
                do_subcross_count(model, mode='smu-pre-recon')
            if do_pre_auto_fft or do_recon_std or do_recon_ite:
                m = np.zeros(len(tb))  # reconstruction prep work
                arr = np.array(
                    [m, tb['x'], tb['y'], tb[model.zfield]]
                    ).T.astype(np.float32)
                path_arr = os.path.join(recon_temp_dir,
                                        'recon_array-{}.dat'.format(seed))
                arr.tofile(path_arr, sep='')
                subprocess.call(['python', './recon/read/read.py',
                                 path_arr, sample_type, str(phase), str(seed)])
            if do_pre_auto_fft:  # pre-recon auto-correlation with FFTcorr
                print('r = {:2d}, calculating pre-recon FFTcorr...'.format(r))
                subprocess.call(
                    ['python', './recon/reconstruct/reconst.py',
                     '0', ('rsd' if use_rsd else 'real'),
                     str(seed), model_name, filedir, str(bias)])
            # check that reconstructed catalogue exists
            if do_recon_std:
                print('r = {:2d}, now applying standard recon...'.format(r))
                subprocess.call(
                    ['python', './recon/reconstruct/reconst.py',
                     '1', ('rsd' if use_rsd else 'real'), str(seed),
                     model_name, filedir, str(bias)])
                # rename shifted D and R to avoid being deleted by recon-ite
                os.rename(os.path.join(recon_temp_dir,
                                       'file_D-{}_rec'.format(seed)), pathD)
                os.rename(os.path.join(recon_temp_dir,
                                       'file_R-{}_rec'.format(seed)), pathR)
                # save shifted D and R
                print('r = {:2d}, reading shifted D, R samples...'.format(r))
                D = np.float32(np.fromfile(pathD, dtype=np.float64)
                               [8:].reshape(-1, 4)[:, :3])
                R = np.float32(np.fromfile(pathR, dtype=np.float64)
                               [8:].reshape(-1, 4)[:, :3])
                D[D < 0] += 1100  # read.cpp wraps data above 550 to negative
                R[R < 0] += 1100
                # randomly choose a 10 x ND subset of shited random sample
                print('r = {:2d}, choosing shifted randoms...'.format(r))
                idx = np.random.choice(np.arange(R.shape[0]),
                                       size=model.mock.ND * random_multiplier,
                                       replace=False)
                model.mock.shifted_randoms = R[idx, :]
                tb['x'], tb['y'], tb['z'] = D[:, 0], D[:, 1], D[:, 2]
                model.mock.reconstructed = True
                model.zfield = 'z'
                tb.write(gt_recon_path, format='ascii.fast_csv',
                         overwrite=True)
                np.save(path_sr, model.mock.shifted_randoms)
                print('r = {:2d}, shifted D and R saved.'.format(r))
            if do_post_auto_rppi or do_post_cross:
                if model.mock.reconstructed:
                    pass  # mock already has shifted D, R samples
                elif os.path.isfile(gt_recon_path) and os.path.isfile(path_sr):
                    model.mock.galaxy_table = tb = table.Table.read(
                        gt_recon_path, format='ascii.fast_csv',
                        fast_reader={'parallel': True,
                                     'use_fast_converter': False})
                    for key in ['x', 'y', 'z']:  # float64 right after loading
                        tb[key] = np.float32(tb[key])
                    model.mock.reconstructed = True
                    model.zfield = 'z'
                    model.mock.shifted_randoms = np.load(path_sr)
                else:
                    raise Exception('Missing reconstructed catalogues.')
            if do_post_auto_rppi:
                print('r = {:2d}, post-recon xi(rp, pi) counting..'.format(r))
                do_auto_count(model, mode='rppi-post-recon-std')
            if do_post_cross:
                print('r = {:2d}, full-subvolume cross counting...'.format(r))
                do_subcross_count(model, mode='smu-post-recon-std')
            # iterative reconstruction, reads original positions from .dat file
            # this removes reconstructed catalogues, so run it after x-corr
            if do_recon_ite:
                print('r = {:2d}, now applying iterative recon...'.format(r))
                subprocess.call(
                    ['python', './recon/reconstruct/reconst.py',
                     '2',  ('rsd' if use_rsd else 'real'), str(seed),
                     model_name, filedir, str(bias)])
        if do_pre_auto_corr:
            do_auto_correlation(model_name, phase, r,
                                mode='smu-pre-recon')
            do_auto_correlation(model_name, phase, r,
                                mode='rppi-pre-recon')
        if do_post_auto_corr:
            do_auto_correlation(model_name, phase, r,
                                mode='rppi-post-recon-std')
        with closing(MyPool(processes=N_threads,
                            maxtasksperchild=1)) as p:
            if do_pre_cross_corr:
                p.map(partial(do_subcross_correlation, model_name=model_name,
                              phase=phase, r=r, mode='smu-pre-recon'),
                      range(N_sub**3))
            if do_post_cross_corr:
                p.map(partial(do_subcross_correlation, model_name=model_name,
                              phase=phase, r=r, mode='smu-post-recon-std'),
                      range(N_sub**3))
        p.close()
        p.join()
        print('r = {:2d} finished.'.format(r))
    except Exception as E:
        print('Exception caught in worker thread r = {:2d}'.format(r))
        traceback.print_exc()
        raise E


recon_types = ['pre-recon', 'post-recon-std']  # 'post-recon'ite'
coadd_filenames = [  # coadd_realisations, change this for iterative recon
#    '-auto-xi-smu-pre-recon-ar',
#    '-auto-xi_0-smu-pre-recon-ar', '-auto-xi_2-smu-pre-recon-ar',
#    '-auto-xi-rppi-pre-recon-ar', '-auto-xi-rppi-post-recon-std-sr',
#    '-auto-wp-rppi-pre-recon-ar', '-auto-wp-rppi-post-recon-std-sr',
]  # exclude iterative reconstruction for fitter test with baseline HOD
for recon_type in recon_types:
    coadd_filenames = coadd_filenames + [
        '-auto-fftcorr_xi_0-{}-15.0_hmpc'.format(recon_type),
        '-auto-fftcorr_xi_2-{}-15.0_hmpc'.format(recon_type)]
fft = ['fftcorr' in fn for fn in coadd_filenames].index(True)


def coadd_realisations(model_name, phase, reals):

    try:
        # Co-add auto-correlation results from all realisations
        sim_name = '{}_{:02}-{}'.format(sim_name_prefix, cosmology, phase)
        filedir = os.path.join(save_dir, sim_name, 'z{}'.format(redshift))
        if not os.path.exists(filedir):
            try:
                os.makedirs(filedir)
            except OSError:
                assert os.path.exists(filedir)
        # convert N, R counts to correlations for each realisation
        print('Converting fftcorr N, R to multipoles for phase {}, model {}...'
              .format(phase, model_name))
        if model_name == 'matter':  # simply copy
            bias = 1
        else:
            bias = galaxy_bias
        for r, i in product(reals, range(len(recon_types))):
            recon_type = recon_types[i]
            realdir = os.path.join(
                save_dir,
                '{}_{:02}-{}'.format(sim_name_prefix, cosmology, phase),
                'z{}-r{}'.format(redshift, r))
            pathN = os.path.join(realdir,
                                 '{}-auto-fftcorr_N-{}-15.0_hmpc.txt'
                                 .format(model_name, recon_type))
            pathR = os.path.join(realdir,
                                 '{}-auto-fftcorr_R-{}-15.0_hmpc.txt'
                                 .format(model_name, recon_type))
            N, R = np.loadtxt(pathN), np.loadtxt(pathR)
            assert np.all(N[:, 1] == R[:, 1])
            r = N[:, 1]
            xi_0 = N[:, 3] / R[:, 3] * np.square(bias)
            xi_2 = N[:, 4] / R[:, 3] * np.square(bias)
            xi_0_txt = np.vstack([r, xi_0]).T
            xi_2_txt = np.vstack([r, xi_2]).T
            tag = recon_type + '-15.0_hmpc'
            for ell, xi_ell in zip([0, 2], [xi_0_txt, xi_2_txt]):
                np.savetxt(os.path.join(realdir,
                                        '{}-auto-fftcorr_xi_{}-{}.txt'
                                        .format(model_name, ell, tag)),
                           xi_ell, fmt=txtfmt)
        if model_name == 'matter':  # simply copy
            fl = glob(os.path.join(realdir,
                                   'matter-auto-fftcorr_xi_[0-9]*.txt'))
            for src in fl:
                dst = os.path.join(filedir,
                                   os.path.basename(src)).replace(
                                           '.txt', '-coadd.txt')
                copyfile(src, dst)
        print('Phase {}, model {}: coadding x-corr for {} realisations...'
              .format(phase, model_name, len(reals)))
        cross_temps = []
        if 'pre-recon' in recon_types:
            cross_temps = cross_temps + [
                '-cross_{}-xi-smu-pre-recon-ar',
                '-cross_{}-xi_0-smu-pre-recon-ar',
                '-cross_{}-xi_2-smu-pre-recon-ar']
        if 'post-recon-std' in recon_types:
            cross_temps = cross_temps + [
                '-cross_{}-xi-smu-post-recon-std-sr',
                '-cross_{}-xi_0-smu-post-recon-std-sr',
                '-cross_{}-xi_2-smu-post-recon-std-sr']
        coadd_fl = []
        for fn in cross_temps:
            for i in range(N_sub**3):
                coadd_fl.append(fn.format(i))
        coadd_fl = coadd_fl + coadd_filenames
        for fn in coadd_fl:
            temp = os.path.join(save_dir, sim_name,
                                'z{}-r*'.format(redshift),
                                model_name+fn+'.txt')
            paths = glob(temp)
            try:
                assert len(paths) == len(reals)
            except AssertionError as E:
                print('Number of files coadded does not equal to N_reals. '
                      'Glob template is: ', temp)
                print('Paths found:', paths)
                raise E
            corr_list = [np.loadtxt(path) for path in paths]
            coadd, error = coadd_correlation(corr_list)
            np.savetxt(os.path.join(
                    filedir, '{}{}-coadd.txt'.format(model_name, fn)),
                coadd, fmt=txtfmt)
            np.savetxt(os.path.join(
                    filedir, '{}{}-error.txt'.format(model_name, fn)),
                error, fmt=txtfmt)
    except Exception as E:
        print('Exception caught in thread {}'.format(model_name))
        traceback.print_exc()
        raise E


def coadd_phases(model_name):

    filedir = os.path.join(save_dir,
                           '{}_{:02}-coadd'.format(sim_name_prefix, cosmology),
                           'z{}'.format(redshift))
    if not os.path.exists(filedir):
        try:
            os.makedirs(filedir)
        except OSError:
            assert os.path.exists(filedir)
    for fn in coadd_filenames:
        print('Coadding {} from {} phases for model {}...'
              .format(fn, len(phases), model_name))
        temp = os.path.join(
            save_dir,
            '{}_{:02}-[0-9]*'.format(sim_name_prefix, cosmology),
            'z{}'.format(redshift), model_name+fn+'*coadd*')
        paths = glob(temp)  # co-add all boxes together
        try:
            assert len(paths) == len(phases)
        except AssertionError as E:
            print('coadd phases, template is:', temp)
            print('paths globbed:', paths)
            raise E
        corr_list = [np.loadtxt(path) for path in paths]
        coadd, error = coadd_correlation(corr_list)
        np.savetxt(os.path.join(
                filedir, '{}{}-coadd.txt'.format(model_name, fn)),
            coadd, fmt=txtfmt)
        np.savetxt(os.path.join(
                filedir, '{}{}-error.txt'.format(model_name, fn)),
            error, fmt=txtfmt)
        print('Producing delete-1 jackknife samples for model {}...'
              .format(model_name))
        for phase in phases:  # jackknife delete-one
            jk_list = corr_list[:phase] + corr_list[phase+1:]
            assert len(jk_list) == len(phases) - 1
            coadd, error = coadd_correlation(jk_list)
            np.savetxt(os.path.join(
                    filedir, '{}{}-jackknife_{}-coadd.txt'
                    .format(model_name, fn, phase)),
                coadd, fmt=txtfmt)
            np.savetxt(os.path.join(
                    filedir, '{}{}-jackknife_{}-error.txt'
                    .format(model_name, fn, phase)),
                error, fmt=txtfmt)


def do_cov(model_name):

    # calculate monoquad cov for baofit from co-added xi for each phase box
    rand_types = []
    if 'pre-recon' in recon_types:
        rand_types = rand_types + ['ar']
    if 'post-recon-std' in recon_types:
        rand_types = rand_types + ['sr']
    for rec, rand in zip(recon_types, rand_types):
        tmp = save_dir + (
            '/{0}_{1:02}-[0-9]*/z{2}/{3}-cross_*-xi_{4}*smu*{5}*{6}-coadd.txt'
            .format(sim_name_prefix, cosmology, redshift, model_name,
                    '{}', rec, rand))
        paths0 = sorted(glob(tmp.format(0)))
        paths2 = sorted(glob(tmp.format(2)))
        try:
            assert len(paths0) == len(phases)*N_sub**3
        except AssertionError as E:
            print('Number of files coadded does not equal to N_reals. '
                  'Glob template is: ', tmp.format(0))
            print('Paths found:', paths0)
            raise E
        assert len(paths0) == len(paths2)
        xi_list = []
        for path0, path2 in zip(paths0, paths2):
            # sorted list of paths within a phase guarantees
            # xi0 and xi2 are from the same realisation
            xi_0 = np.loadtxt(path0)[:, 1]
            xi_2 = np.loadtxt(path2)[:, 1]
            xi_list.append(np.hstack((xi_0, xi_2)))
        print('Calculating covariance matrix from {} multipole samples...'
              .format(len(xi_list)))
        cov_monoquad = np.cov(np.array(xi_list).T, bias=0)
        # correct for volume and jackknife
        cov_monoquad = cov_monoquad / np.power(N_sub, 3) / (len(phases)-1)
        filepath = os.path.join(  # save to coadd folder for all phases
            save_dir, '{}_{:02}-coadd'.format(sim_name_prefix, cosmology),
            'z{}'.format(redshift),
            '{}-cross-xi_monoquad-cov-smu-{}-{}.txt'
            .format(model_name, rec, rand))
        np.savetxt(filepath, cov_monoquad, fmt=txtfmt)
        print('Monoquad covariance matrix saved to: ', filepath)


def combine_galaxy_table_metadata(phases):

    # check if metadata for all models, phases, and realisations exist
    paths = [os.path.join(
                save_dir, 'galaxy_table_meta',
                '{}-z{}-{}-r{}.csv'.format(model_name, redshift, phase, r))
             for model_name in model_names
             for phase in phases
             for r in reals]
    exist = [os.path.exists(path) for path in paths]
    if not np.all(exist):  # at least one path does not exist, print it
        raise NameError('Galaxy table metadata incomplete. Missing:\n{}'
                        .format(np.array(paths)[~np.where(exist)]))
    tables = [table.Table.read(
                path, format='ascii.fast_csv',
                fast_reader={'parallel': True, 'use_fast_converter': False})
              for path in tqdm(paths)]
    gt_meta = table.vstack(tables)
    gt_meta.write(os.path.join(save_dir, 'galaxy_table_meta.csv'),
                  format='ascii.fast_csv', overwrite=True)


class FitterCache:

    def __init__(self, model_name, recon_type, force_isotropy=False,
                 beta=0, rmin=50, r_rescale_factor=1):
        '''
        input parameters are: (beta, r_rescale_factor)
        '''
        self.polydeg = '3'  # optimise parameters for each polynomial degree
        self.rmin = 50
        self.model_name = model_name
        self.recon_type = recon_type
        self.force_isotropy = force_isotropy
        self.results = {}  # initialise lookup dict results[beta][r_fac]
        self.baofit_parallel((beta, 1))  # pass beta prime
        if 'matter' in model_name:
            self.sample_type = 'mat'
        else:
            self.sample_type = 'gal'

    def baofit_parallel(self, parameters):
        # input beta and r_rescale factor (algways ignored)
        beta = np.around(parameters[0], decimals=2)
        if np.isnan(beta):
            print('NaN parameters encountered, returning NaN...')
            return np.nan
        if beta in self.results.keys():
            print('Returning cached result to optimiser: '
                  'beta = {}, error = {:.6f}'
                  .format(beta, self.results[beta]))
            return self.results[beta]
        indir = os.path.join(
            save_dir, '{}_{:02}-coadd'.format(sim_name_prefix, cosmology),
            'z{}'.format(redshift))
        outdir = os.path.join(save_dir,
                              '2Dbaofits_poly{}_rmin_{}_beta_{:.3f}'
                              .format(self.polydeg, self.rmin, beta))
        list_of_inputs = []
        xi_type = self.recon_type + '-15.0_hmpc'
        alpha = {}
        alpha['mat'] = {'r': np.zeros(len(phases)),
                        't': np.zeros(len(phases))}
        for phase in phases:  # list of inputs for parallelisation
            # sim_name = '{}_{:02}-{}'.format(
            #         sim_name_prefix, cosmology, phase)
            path_p_lin = "/mnt/gosling1/bigsim_products/" \
                "AbacusCosmos_1100box_planck_products/" \
                "AbacusCosmos_1100box_planck_00-0_products/" \
                "AbacusCosmos_1100box_planck_00-0_power/" \
                "info/camb_matterpower.dat"
            # path_p_lin = os.path.join(
            #     prod_dir, sim_name+'_products', sim_name+'_power', 'info',
            #     'camb_matterpower.dat')
            path_xi_0 = os.path.join(
                indir,
                '{}-auto-fftcorr_xi_0-{}-jackknife_{}-coadd.txt'
                .format(self.model_name, xi_type, phase))
            path_xi_2 = os.path.join(
                indir,
                '{}-auto-fftcorr_xi_2-{}-jackknife_{}-coadd.txt'
                .format(self.model_name, xi_type, phase))
            if 'pre-recon' in xi_type:
                recon = False
                path_cov = os.path.join(
                    indir,
                    '{}-cross-xi_monoquad-cov-smu-pre-recon-ar.txt'
                    .format(self.model_name))
            elif 'post-recon' in xi_type:
                recon = True
                path_cov = os.path.join(
                    indir,
                    '{}-cross-xi_monoquad-cov-smu-post-recon-std-sr.txt'
                    .format(self.model_name))
            fout_tag = '{}-{}-{}'.format(self.model_name, phase, xi_type)
            path_cg = os.path.join(outdir, fout_tag+'-chisq_grid.txt')
            path_ct = os.path.join(outdir, fout_tag+'-chisq_table.txt')
            list_of_inputs.append(
                [self.sample_type, redshift, recon, self.force_isotropy,
                 self.polydeg, self.rmin, beta, path_p_lin,
                 path_xi_0, path_xi_2, path_cov, path_cg, path_ct])
            # path_matter = os.path.join(
            #     store_dir, sim_name_prefix+'-mmatter',
            #     '2Dbaofits_poly3_rmin_50',
            #     'matter-{}-{}-chisq_table.txt'.format(phase, xi_type))
            # chisq_table = np.loadtxt(path_matter)
            # ar, at, _ = chisq_table[np.argmin(chisq_table[:, 2]), :]
            # alpha['mat']['r'][phase], alpha['mat']['t'][phase] = ar, at
        # ret = [baofit(list_of_inputs[0])]  # debug
        # N_proc = len(phases) if len(phases) <= 20 else int(len(phases)/2)
        with closing(MyPool(processes=len(phases),
                            maxtasksperchild=1)) as p:
            ret = p.map(baofit, list_of_inputs)
        p.close()
        p.join()
        alpha['gal'] = {'r': np.array([item[0] for item in ret]),
                        't': np.array([item[1] for item in ret])}
        # aiso = np.power(alpha['gal']['r'], 1/3) \
        #     * np.power(alpha['gal']['t'], 2/3)
        if beta not in self.results.keys():
            self.results[beta] = {}
        ret = self.results[beta] = \
            + np.linalg.norm(alpha['gal']['r']-1) * 1/3 \
            + np.linalg.norm(alpha['gal']['t']-1) * 2/3
        print('Returning new result to optimiser: beta = {}, error = {:.6f}'
              .format(beta, ret))
        return ret


def optimise_fitter(model_name, recon_type, brute=True):

    cache = FitterCache(model_name, recon_type)
    if brute:
        betas = np.arange(0, 0.45, 0.01)
        errors = np.zeros(betas.size)
        for i, beta in enumerate(betas):
            print('Brute progress currently: {}/{}'.format(i, beta.size))
            errors[i] = cache.baofit_parallel((beta, 1))
        np.savetxt(os.path.join(save_dir, 'brute_beta.txt'),
                   np.hstack([betas, errors]).T, fmt=txtfmt)
    else:
        beta = 0
        bounds = ((0, 0.5), (1, 1))
        ret = optimize.minimize(
            cache.baofit_parallel, (beta, 1), bounds=bounds, method='SLSQP',
            options={'ftol': 1e-5, 'eps': 1e-2, 'maxiter': 4, 'disp': True})
        if ret.success:
            print('Best beta = {}'.format(ret.x))
        else:
            print('fitter optimiser failed', ret.message)
    print('Results:')
    for beta in cache.results.keys():
        print('beta = {:.2f}, error = {:.6f}'
              .format(beta, cache.results[beta]))


if __name__ == "__main__":
    if do_create:
        fit_c_median(sim_name_prefix, prod_dir, store_dir, redshift, cosmology,
                     phases=phases)
        for phase in phases:
            global halocat
            if 'matter' in model_names:
                halocat = make_halocat(phase, halo_type='FoF')
                print('\nSkipping halocat processing, matter field...')
            else:
                halocat = process_rockstar_halocat(halocat, N_cut=N_cut)
            for model_name in model_names:
                print('---\nWorking on {} realisations, phase {}, model {}...'
                      .format(len(reals), phase, model_name))
                with closing(MyPool(processes=int(np.ceil(len(reals)/2)),
                                    maxtasksperchild=1)) as p:
                    p.map(partial(do_galaxy_table,
                                  phase=phase, model_name=model_name,
                                  overwrite=False),
                          reals)
                    p.close()
                    p.join()
                with closing(MyPool(processes=N_concurrent_reals,
                                    maxtasksperchild=1)) as p:
                    p.map(partial(
                            do_realisation, phase=phase, model_name=model_name,
                            overwrite=False,
                            do_pre_auto_smu=False, do_pre_auto_rppi=False,
                            do_pre_auto_fft=True, do_pre_cross=True,
                            do_recon_std=False, do_post_auto_rppi=False,
                            do_post_cross=False, do_recon_ite=False,
                            do_pre_auto_corr=False, do_post_auto_corr=False,
                            do_pre_cross_corr=True, do_post_cross_corr=False),
                          reals)
                    p.close()
                    p.join()
                print('---\nPool closed cleanly for model {}.\n---'
                      .format(model_name))
    if do_coadd:
        for phase in phases:
            with closing(MyPool(processes=len(model_names),
                                maxtasksperchild=1)) as p:
                p.map(partial(coadd_realisations, phase=phase, reals=reals),
                      model_names)
                p.close()
                p.join()
        # combine_galaxy_table_metadata(phases)
        with closing(MyPool(processes=len(model_names),
                            maxtasksperchild=1)) as p:
            p.map(coadd_phases, model_names)
            p.map(do_cov, model_names)
            p.close()
            p.join()
    if do_optimise_fitter:
        optimise_fitter('gen_base1', 'pre-recon', brute=True)
    if do_fit:
        for model_name in model_names:
            # for recon in recon_types:
            '''
            best-fit parameters for 16-box sim post-recon:
            real-space matter: beta = 0.00, r_rescale_factor = 1.00174
            redshift-space ga: beta = 0.00, r_rescale_factor = 1.00232

            best-fit parameters for 20-box sim post-recon:
            real-space matter: beta = 0.00, r_rescale_factor = 0.99843
            redshift-space ga: beta = 0.00, r_rescale_factor = 0.99707

            we expect a 0.2% shift due to nonlinear evolutions when fitting
            with a linear power spectrum which does not account for
            nonlinear effects; no need to fiddle with r-rescaling

            best-fit parameters for 36-box joint set:
            real-space matter pre-/post-recon: beta = 0.00
            redshift-space galaxy pre-recon:   beta = 0.00
            (lower chisq 50 vs 80, a insensitive to beta to 0.2%, )
            redshift-space galaxy post-recon:  beta = 0.00
            '''
            # model_name = 'gen_base1'
            cache = FitterCache(model_name, 'pre-recon',
                                force_isotropy=False, beta=0, rmin=50)
