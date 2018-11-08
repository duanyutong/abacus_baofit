# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 17:30:51 2017

@author: Duan Yutong (dyt@physics.bu.edu)L

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
from astropy import table
from halotools.mock_observables import tpcf_multipole
from halotools.mock_observables.two_point_clustering.s_mu_tpcf import \
        spherical_sector_volume
from halotools.mock_observables.two_point_clustering.rp_pi_tpcf import \
    cylinder_volume
from halotools.utils import add_halo_hostid
from Corrfunc.theory.DDsmu import DDsmu
from Corrfunc.theory.DDrppi import DDrppi
from Corrfunc.utils import convert_rp_pi_counts_to_wp
from abacus_hod import MyPool, fit_c_median, process_rockstar_halocat, \
    initialise_model, populate_model
from tqdm import tqdm
import Halotools as abacus_ht  # Abacus' "Halotools" for importing Abacus


# %% custom settings
sim_name_prefix = 'emulator_1100box_planck'
tagout = 'recon'  # 'z0.5'
phases = range(16)  # range(16)  # [0, 1] # list(range(16))
cosmology = 0  # one cosmology at a time instead of 'all'
redshift = 0.5  # one redshift at a time instead of 'all'
L = 1100  # boxsize in 1D
model_names = ['gen_base1', 'gen_base4', 'gen_base5',
               'gen_ass1',  'gen_ass1_n',
               'gen_ass2',  'gen_ass2_n',
               'gen_ass3',  'gen_ass3_n',
               'gen_s1',    'gen_s1_n',
               'gen_sv1',   'gen_sv1_n',
               'gen_sp1',   'gen_sp1_n',
               'gen_vel1']
N_reals = 12  # number of realisations for an HOD
N_cut = 70  # number particle cut, 70 corresponds to 4e12 Msun
N_concurrent_reals = 4
N_threads = 8  # number of threads for a single realisation
N_sub = 3  # number of subvolumes per dimension
random_multiplier = 10
galaxy_bias = 2.23

# %% flags
reuse_galaxies = True
save_hod_realisation = True
add_rsd = True

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
prod_dir = r'/mnt/gosling2/bigsim_products/emulator_1100box_planck_products/'
store_dir = '/home/dyt/store/'
save_dir = os.path.join(store_dir, sim_name_prefix+'-'+tagout)
recon_temp_dir = os.path.join(store_dir, 'recon/temp/')
halo_type = 'Rockstar'
halo_m_prop = 'halo_mvir'  # mgrav is better but not using sub or small haloes
txtfmt = b'%.30e'


# %% definitionss of statisical formulae and convenience functions
def analytic_random(ND1, NR1, ND2, NR2, boxsize, mode='smu',
                    s_bins=None, mu_bins=None, rp_bins=None, pi_bins=None):
    '''
    V is the global volume L^3
    for auto-correlation, we may set ND1 = NR1 = ND2 = NR2
    for cross-correlation, returns D1R2 and R1D2 for LS estimator
    '''
    V = boxsize**3
    if mode == 'smu' and s_bins is not None and mu_bins is not None:
        mu_bins_reverse_sorted = np.sort(mu_bins)[::-1]
        dv = spherical_sector_volume(s_bins, mu_bins_reverse_sorted)
        dv = np.diff(dv, axis=1)  # volume of wedges
        dv = np.diff(dv, axis=0)  # volume of wedge sector
    elif mode == 'rppi' and rp_bins is not None and pi_bins is not None:
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
    print('Length of items being coadded: ', len(corr_list))
    arr = np.array(corr_list)
    coadd = np.mean(arr, axis=0)
    err = np.std(arr, axis=0)
    if arr.shape[2] == 2:
        err[:, 0] = coadd[:, 0]
    return coadd, err


# %% functions for tasks

def make_halocat(phase):

    print('---\n'
          + 'Importing phase {} halo catelogue from: \n'.format(phase)
          + '{}{}_{:02}-{}_products/z{}/'
            .format(prod_dir, sim_name_prefix, cosmology, phase, redshift)
          + '\n---')
    halocat = abacus_ht.make_catalogs(
        sim_name=sim_name_prefix, products_dir=prod_dir,
        redshifts=[redshift], cosmologies=[cosmology], phases=[phase],
        halo_type=halo_type,
        load_halo_ptcl_catalog=True,  # this loads 10% particle subsamples
        load_ptcl_catalog=False,  # this loads uniform subsamples, dnw
        load_pids=False)[0][0][0]
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


def do_auto_count(model, mode='smu-pre-recon'):

    '''
    mode is a string that contains 'smu' or 'wp', and can include
    pre/post-recon tag which gets saved as part of filename
    '''
    # do pair counting in 1 Mpc bins
    sim_name = model.mock.header['SimName']
    redshift = model.mock.redshift
    model_name = model.model_name
    model.ND = ND = len(model.mock.galaxy_table)
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
    if 'smu' in mode:
        # save counting results as original structured array in npy format
        DDnpy = DDsmu(1, N_threads, s_bins_counts, mu_max, n_mu_bins,
                      x, y, z, periodic=True, verbose=False, boxsize=L,
                      output_savg=True, c_api_timer=False)
    elif 'rppi' in mode:  # also returns wp in addition to counts
        DDnpy = DDrppi(1, N_threads, pi_max, rp_bins, x, y, z,
                       periodic=True, verbose=False, boxsize=L,
                       output_rpavg=True, c_api_timer=False)
    else:
        print('Bad mode: {}'.format(mode))
        raise ValueError
    DDnpy = rfn.append_fields(DDnpy, ['DD'],
                              [DDnpy['npairs'].astype(np.float64)/ND/ND],
                              usemask=False)
    np.save(os.path.join(filedir, '{}-auto-paircount-DD-{}.npy'
                         .format(model_name, mode)),
            DDnpy)


def do_auto_correlation(model_name, phase, r, mode='smu-pre-recon-std'):

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
        DR_ar, _, RR_ar = analytic_random(1, 1, 1, 1, L, mode='smu',
                                          s_bins=s_bins, mu_bins=mu_bins)
        xi_smu = ls_estimator(DD, DR_ar, DR_ar, RR_ar, RR_ar)
        xi_0 = tpcf_multipole(xi_smu, mu_bins, order=0)
        xi_2 = tpcf_multipole(xi_smu, mu_bins, order=2)
        xi_0_txt = np.vstack([s_bins_centre, xi_0]).T
        xi_2_txt = np.vstack([s_bins_centre, xi_2]).T
        for output, tag in zip([xi_smu, xi_0_txt, xi_2_txt],
                               ['xi', 'xi_0', 'xi_2']):
            np.savetxt(os.path.join(filedir, '{}-auto-{}-{}-ar.txt'
                                    .format(model_name, tag, mode)),
                       output, fmt=txtfmt)
    elif 'rppi' in mode:  # no rebinning necessary
        # even for post-recon, rppi and wp are calculated with analytic randoms
        # counts not normalised becuase we are using corrfunc for wp conversion
        ND = np.median(np.around(DD['npairs'] / DD['DD']))
        DR_ar, _, RR_ar = analytic_random(ND, ND, ND, ND, L, mode='rppi',
                                          rp_bins=rp_bins, pi_bins=pi_bins)

        DD = DDnpy['npairs'].reshape(len(rp_bins)-1, -1)
        xi_rppi = ls_estimator(DD, DR_ar, DR_ar, RR_ar, RR_ar)
        wp = convert_rp_pi_counts_to_wp(
            ND, ND, ND, ND, DD.flatten(), DR_ar.flatten(), DR_ar.flatten(),
            RR_ar.flatten(), len(rp_bins)-1, pi_max, estimator='LS')
        wp_txt = np.vstack([(rp_bins[:-1] + rp_bins[1:])/2, wp]).T
        np.savetxt(os.path.join(filedir, '{}-auto-xi-{}-ar.txt'
                                .format(model_name, mode)),
                   xi_rppi, fmt=txtfmt)
        np.savetxt(os.path.join(filedir, '{}-auto-wp-{}.txt'
                                .format(model_name, mode)),
                   wp_txt, fmt=txtfmt)
    assert DD.shape == DR_ar.shape == RR_ar.shape
    np.savetxt(os.path.join(filedir,
                            '{}-auto-paircount-DR-{}-ar.txt'
                            .format(model_name, mode)),
               DR_ar, fmt=txtfmt)
    np.savetxt(os.path.join(filedir,
                            '{}-auto-paircount-RR-{}-ar.txt'
                            .format(model_name, mode)),
               RR_ar, fmt=txtfmt)


def do_subcross_count(model, mode='smu-post-recon-std',
                      use_shifted_randoms=True):

    '''
    cross-correlation between 1/N_sub^3 of a box and the whole box
    can be pre/post-reconstruction data
    '''
    sim_name = model.mock.header['SimName']
    model_name = model.model_name
    redshift = model.mock.redshift
    gt = model.mock.galaxy_table
    x1, y1, z1 = gt['x'], gt['y'], gt['z']  # sample 1 is the full box volume
    ND1 = model.mock.ND  # number of galaxies in entire volume of periodic box
    filedir = os.path.join(save_dir, sim_name,
                           'z{}-r{}'.format(redshift, model.r))
    # split the (already shuffled) randoms into N copies, each of size NData
    if use_shifted_randoms:
        assert hasattr(model.mock, 'shifted_randoms')
        sr_list = np.array_split(model.mock.shifted_randoms, random_multiplier)
    for i, j, k in product(range(N_sub), repeat=3):
        linind = i*N_sub**2 + j*N_sub + k  # linearised index of subvolumes
        print('r = {}, x-corr counting, subvol {}...'.format(model.r, linind))
        x2, y2, z2 = subvol_mask(gt['x'], gt['y'], gt['z'], i, j, k, L, N_sub)
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
        if use_shifted_randoms and 'post-recon' in mode and \
                hasattr(model.mock, 'shifted_randoms'):
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
                            mode='smu-post-recon-std',
                            use_shifted_randoms=True):

    sim_name = '{}_{:02}-{}'.format(sim_name_prefix, cosmology, phase)
    filedir = os.path.join(save_dir, sim_name, 'z{}-r{}'.format(redshift, r))
    print('r = {}, x-correlation subvolume {}...'.format(r, linind))
    DDnpy = np.load(os.path.join(filedir, '{}-cross_{}-paircount-DD-{}.npy'
                                 .format(model_name, linind, mode)))
    DD, _ = rebin_smu_counts(DDnpy, 'DD')  # re-bin and re-weight
    np.savetxt(os.path.join(filedir,
                            '{}-cross_{}-paircount-DD-{}-rebinned.txt'
                            .format(model_name, linind, mode)),
               DD, fmt=txtfmt)
    if 'post-recon-std' in mode and use_shifted_randoms:
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
        np.savetxt(os.path.join(
                filedir,
                '{}-cross_{}-paircount-RR-{}-ar.txt'
                .format(model_name, linind, mode)),
            RR_ar, fmt=txtfmt)
        assert DD.shape == DR.shape == RD.shape == RR.shape == RR_ar.shape
        filename = '{}-cross_{}-{}-{}-sr.txt'
    elif 'pre-recon' in mode:  # pre-recon covariance, use analytic randoms
        DR, RD, RR = analytic_random(1, 1, 1, 1, L, mode='smu',
                                     s_bins=s_bins, mu_bins=mu_bins)
        for output, tag in zip([DR, RD, RR], ['DR', 'RD', 'RR']):
            np.savetxt(os.path.join(
                    filedir, '{}-cross_{}-paircount-{}-{}-ar.txt'
                    .format(model_name, linind, tag, mode)),
                output, fmt=txtfmt)
        filename = '{}-cross_{}-{}-{}-ar.txt'
    else:
        print('Mode error for: {}'.format(mode))
    # calculate cross-correlation from counts using ls estimator
    xi_smu = ls_estimator(DD, DR, RD, RR, RR)
    xi_0 = tpcf_multipole(xi_smu, mu_bins, order=0)
    xi_2 = tpcf_multipole(xi_smu, mu_bins, order=2)
    xi_0_txt = np.vstack([s_bins_centre, xi_0]).T
    xi_2_txt = np.vstack([s_bins_centre, xi_2]).T
    for output, tag in zip([xi_smu, xi_0_txt, xi_2_txt],
                           ['xi', 'xi_0', 'xi_2']):
        np.savetxt(os.path.join(
                filedir,
                filename.format(model_name, linind, tag, mode)),
            output, fmt=txtfmt)


def do_galaxy_table(r, phase, model_name, overwrite=False):

    sim_name = '{}_{:02}-{}'.format(sim_name_prefix, cosmology, phase)
    filedir = os.path.join(save_dir, sim_name,
                           'z{}-r{}'.format(redshift, r))
    gt_path = os.path.join(filedir, '{}-galaxy_table.csv'.format(model_name))
    if os.path.isfile(gt_path):
        return
    global halocat
    try:
        assert halocat.ZD_Seed == phase
    except NameError or AssertionError:
        halocat = make_halocat(phase)
        process_rockstar_halocat(halocat, N_cut=N_cut)
    seed = phase*100 + r
    # all realisations in parallel, each model needs to be instantiated
    model = initialise_model(redshift, model_name,
                             halo_m_prop=halo_m_prop)
    # random seed w/ phase and realisation index r, model independent
    print('r = {} for phase {}, model {} starting...'
          .format(r, phase, model_name))
    model = populate_model(halocat, model, add_rsd=add_rsd,
                           gt_path=reuse_galaxies * gt_path)
    gt = model.mock.galaxy_table
    # save galaxy table if it's not already loaded from disk
    if save_hod_realisation and not model.mock.gt_loaded:
        print('r = {}, saving galaxy table...'.format(r))
        try:
            os.makedirs(os.path.dirname(gt_path))
        except OSError:
            pass
        gt.write(gt_path, format='ascii.fast_csv', overwrite=True)
        # save galaxy table metadata to sim directory, one file each
        # to be combined later to avoid threading conflicts
        N_cen = np.sum(gt['gal_type'] == 'centrals')
        N_sat = np.sum(gt['gal_type'] == 'satellites')
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


def do_realisation(r, phase, model_name, overwrite=False,
                   do_pre_auto_smu=False, do_pre_auto_rppi=True,
                   do_pre_auto_fft=True, do_pre_cross=True,
                   do_recon_std=True, do_post_auto_rppi=True,
                   do_post_cross=True, do_recon_ite=True,
                   do_pre_auto_corr=True, do_post_auto_corr=True,
                   do_pre_cross_corr=True, do_post_cross_corr=True):

    global halocat
    try:  # this helps catch exception in a worker thread of multiprocessing
        sim_name = '{}_{:02}-{}'.format(sim_name_prefix, cosmology, phase)
        filedir = os.path.join(save_dir, sim_name,
                               'z{}-r{}'.format(redshift, r))
        gt_path = os.path.join(filedir, '{}-galaxy_table.csv'
                               .format(model_name))
        gt_recon_path = os.path.join(filedir,
                                     '{}-galaxy_table-post-recon-std.csv'
                                     .format(model_name))
        seed = phase*100 + r
        if os.path.isfile(gt_path) and not overwrite:
            temp = os.path.join(filedir, model_name)
            if len(glob(temp+'-auto-paircount-DD-smu-pre-recon.npy')) == 1:
                do_pre_auto_smu = False
            if len(glob(temp+'-auto-paircount-DD-rppi-pre-recon.npy')) == 1:
                do_pre_auto_rppi = False
            if len(glob(temp+'-auto-fftcorr_N-pre-recon*.txt')) == 1:
                do_pre_auto_fft = False
            if len(glob(temp+'-cross_*-DD-smu-pre-recon.npy')) == 27:
                do_pre_cross = False
            if len(glob(temp+'-auto-fftcorr_N-post-recon-std*.txt')) == 1 \
                    and os.path.isfile(gt_recon_path):
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
            print('r = {}, all counting done,'.format(r))
            do_count = False
        else:
            do_count = True
        if do_count:  # everything within is counting, require loading halocat
            assert halocat.ZD_Seed == phase
            # all realisations in parallel, each model needs to be instantiated
            model = initialise_model(redshift, model_name,
                                     halo_m_prop=halo_m_prop)
            model.N_cut = N_cut  # add useful model properties here
            model.c_median_poly = np.poly1d(np.loadtxt(os.path.join(
                    store_dir, sim_name_prefix, 'c_median_poly.txt')))
            model.r = r
            # random seed w/ phase and realisation index r, model independent
            print('r = {} for phase {}, model {} starting...'
                  .format(r, phase, model_name))
            model = populate_model(halocat, model, add_rsd=add_rsd,
                                   gt_path=reuse_galaxies * gt_path)
            gt = model.mock.galaxy_table
            if do_pre_auto_smu:  # pre-recon auto-correlation pair-counting
                print('r = {}, pre-recon smu pair-counting...'.format(r))
                do_auto_count(model, mode='smu-pre-recon')
            if do_pre_auto_rppi:
                print('r = {}, pre-recon rppi pair-counting...'.format(r))
                do_auto_count(model, mode='rppi-pre-recon')
            if do_pre_cross:
                print('r = {}, now counting for pre-recon x-corr...'.format(r))
                do_subcross_count(model, mode='smu-pre-recon',
                                  use_shifted_randoms=False)
            if do_pre_auto_fft or do_recon_std or do_recon_ite:
                # reconstruction prep work
                m = np.zeros(len(gt))
                arr = np.array(
                        [m, gt['x'], gt['y'], gt['z']]).T.astype(np.float32)
                arr.tofile(os.path.join(recon_temp_dir,
                                        'gal_cat-{}.dat'.format(seed)),
                           sep='')
                subprocess.call(['python', './recon/read/read.py',
                                 str(phase), str(seed)])
            if do_pre_auto_fft:  # pre-recon auto-correlation with FFTcorr
                print('r = {}, now calculating pre-recon FFTcorr...'.format(r))
                subprocess.call(['python', './recon/reconstruct/reconst.py',
                                 '0', str(seed), model_name, filedir])
            # check that reconstructed catalogue exists
            if do_recon_std or ((do_post_cross or do_post_auto_rppi) and
                                not os.path.isfile(os.path.join(
                    recon_temp_dir, 'file_D-{}_rec'.format(seed)))):
                print('r = {}, now applying standard recon...'.format(r))
                subprocess.call(['python', './recon/reconstruct/reconst.py',
                                 '1', str(seed), model_name, filedir])
            if do_post_cross or do_post_auto_rppi:
                if os.path.isfile(gt_recon_path):
                    model.mock.galaxy_table = gt = table.Table.read(
                        gt_path, format='ascii.fast_csv',
                        fast_reader={'parallel': True,
                                     'use_fast_converter': False})
                else:  # load shifted D, R samples
                    print('r = {}, reading shifted D samples...'.format(r))
                    D = np.float32(np.fromfile(  # read shifted data
                            os.path.join(recon_temp_dir,
                                         'file_D-{}_rec'.format(seed)),
                            dtype=np.float64)[8:].reshape(-1, 4))[:, :3]
                    # read.cpp re-wraps data above 550 to negative pos
                    D[D < 0] += 1100  # undo wrapping by read.cpp
                    # write shifted positions to model galaxy table and save it
                    gt['x'], gt['y'], gt['z'] = D[:, 0], D[:, 1], D[:, 2]
                    print('r = {}, saving shifted galaxy table...'.format(r))
                    gt.write(gt_path, format='ascii.fast_csv', overwrite=True)
            if do_post_cross and not os.path.isfile(gt_recon_path):
                print('r = {}, reading shifted R samples...'.format(r))
                R = np.float32(np.fromfile(  # read shifted randoms
                        os.path.join(recon_temp_dir,
                                     'file_R-{}_rec'.format(seed)),
                        dtype=np.float64)[8:].reshape(-1, 4))[:, :3]
                R[R < 0] += 1100
                # randomly choose a 10 x ND subset of shited random sample
                print('r = {}, randomly choosing shifted randoms...'.format(r))
                idx = np.random.choice(np.arange(R.shape[0]),
                                       size=model.mock.ND * random_multiplier,
                                       replace=False)
                model.mock.shifted_randoms = R[idx, :]
                model.mock.reconstructed = True
            if do_post_auto_rppi:
                do_auto_count(model, mode='rppi-post-recon-std')
            if do_post_cross:
                print('r = {}, now calculating x-corr covariance...'.format(r))
                do_subcross_count(model, mode='smu-post-recon-std',
                                  use_shifted_randoms=True)
            # iterative reconstruction, reads original positions from .dat file
            # this removes reconstructed catalogues, so run it after x-corr
            if do_recon_ite:
                print('r = {}, now applying iterative recon...'.format(r))
                subprocess.call(['python', './recon/reconstruct/reconst.py',
                                 '2', str(seed), model_name, filedir])
        if do_pre_auto_corr:
            # debug
            # do_auto_correlation(model_name, phase, r, mode='smu-pre-recon')
            do_auto_correlation(model_name, phase, r, mode='rppi-pre-recon')
        if do_post_auto_corr:
            do_auto_correlation(model_name, phase, r,
                                mode='rppi-post-recon-std')
        # cross correlation calculation after pair-counting
        with closing(MyPool(processes=N_threads,
                            maxtasksperchild=1)) as p:
            if do_pre_cross_corr:
                p.map(partial(do_subcross_correlation, model_name=model_name,
                              phase=phase, r=r, mode='smu-pre-recon',
                              use_shifted_randoms=True),
                      range(N_sub**3))
            if do_post_cross_corr:
                p.map(partial(do_subcross_correlation, model_name=model_name,
                              phase=phase, r=r, mode='smu-post-recon-std',
                              use_shifted_randoms=True),
                      range(N_sub**3))
        p.close()
        p.join()
        print('r = {} finished.'.format(r))
    except Exception as E:
        print('Exception caught in worker thread r = {}'.format(r))
        traceback.print_exc()
        raise E


coadd_filenames = [  # coadd_realisations
    'wp-pre-recon', 'wp-post-recon-std',
    'xi-rppi-pre-recon-ar', 'xi-rppi-post-recon-std-ar',
    'xi-smu-pre-recon-ar',
    'xi_0-smu-pre-recon-ar', 'xi_2-smu-pre-recon-ar',
    'fftcorr_N-pre-recon-15.0_hmpc',
    'fftcorr_R-pre-recon-15.0_hmpc',
    'fftcorr_N-post-recon-std-15.0_hmpc',
    'fftcorr_R-post-recon-std-15.0_hmpc',
    'fftcorr_N-post-recon-ite-15.0_hmpc',
    'fftcorr_R-post-recon-ite-15.0_hmpc']
#       'xi-smu-post-recon-std-ar', 'xi-smu-post-recon-std-nr',
#       'xi_0-smu-post-recon-std-ar', 'xi_0-smu-post-recon-std-nr',
#       'xi_2-smu-post-recon-std-ar', 'xi_2-smu-post-recon-std-nr']


def coadd_realisations(model_name, phase, N_reals):

    # Co-add auto-correlation results from all realisations
    sim_name = '{}_{:02}-{}'.format(sim_name_prefix, cosmology, phase)
    filedir = os.path.join(save_dir, sim_name, 'z{}'.format(redshift))
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    print('Coadding {} realisations for phase {}, model {}...'
          .format(N_reals, phase, model_name))
    for fn in coadd_filenames:
        temp = os.path.join(save_dir, sim_name, 'z{}-r*'.format(redshift),
                            model_name+'-auto*'+fn+'*')
        paths = glob(temp)
        try:
            assert len(paths) == N_reals
        except AssertionError as E:
            print('Number of files to be co-added does not equal to N_reals. '
                  'Glob template is: ', temp)
            print('Paths found:', paths)
            raise E
        corr_list = [np.loadtxt(path) for path in paths]
        coadd, error = coadd_correlation(corr_list)
        np.savetxt(os.path.join(
                filedir, '{}-auto-{}-coadd.txt'.format(model_name, fn)),
            coadd, fmt=txtfmt)
        np.savetxt(os.path.join(
                filedir, '{}-auto-{}-error.txt'.format(model_name, fn)),
            error, fmt=txtfmt)


def coadd_phases(model_name, coadd_phases=range(16)):

    print('Coadding xi from {} phases for model {}...'
          .format(len(coadd_phases), model_name))
    filedir = os.path.join(save_dir,
                           '{}_{:02}-coadd'.format(sim_name_prefix, cosmology),
                           'z{}'.format(redshift))
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    for fn in coadd_filenames:
        temp = os.path.join(
            save_dir,
            '{}_{:02}-[0-9]*'.format(sim_name_prefix, cosmology),
            'z{}'.format(redshift), model_name+'-auto*'+fn+'*coadd*')
        paths = glob(temp)  # co-add all boxes together
        try:
            assert len(paths) == len(coadd_phases)
        except AssertionError as E:
            print('temp is:', temp)
            print('paths are:', paths)
            raise E
        corr_list = [np.loadtxt(path) for path in paths]
        print('dtype is', corr_list[0].dtype)
        coadd, error = coadd_correlation(corr_list)
        np.savetxt(os.path.join(
                filedir, '{}-auto-{}-coadd.txt'.format(model_name, fn)),
            coadd, fmt=txtfmt)
        np.savetxt(os.path.join(
                filedir, '{}-auto-{}-error.txt'.format(model_name, fn)),
            error, fmt=txtfmt)
        print('saved to', os.path.join(
                filedir, '{}-auto-{}-coadd.txt'.format(model_name, fn)))
        for phase in coadd_phases:  # jackknife delete-one
            jk_list = corr_list[:phase] + corr_list[phase+1:]
            assert len(jk_list) == len(coadd_phases) - 1
            coadd, error = coadd_correlation(jk_list)
            np.savetxt(os.path.join(
                    filedir, '{}-auto-{}-jackknife_{}-coadd.txt'
                    .format(model_name, fn, phase)),
                coadd, fmt=txtfmt)
            np.savetxt(os.path.join(
                    filedir, '{}-auto-{}-jackknife_{}-error.txt'
                    .format(model_name, fn, phase)),
                error, fmt=txtfmt)


def convert_recon_correlation(model_name, phases=range(16)):

    '''
    take pre- and post-recon FFTcorr jackknife results from each box,
    the N and R files, and produce fftcorr_xi_0 and xi_2
    with the correct scaling, taking into account b = 2.23


    b = 2.23  # galaxy bias for rescaling xi

    '''

    print('Converting recon N and R to multipoles for model {}...'
          .format(model_name))
    filedir = os.path.join(save_dir,
                           '{}_{:02}-coadd'.format(sim_name_prefix, cosmology),
                           'z{}'.format(redshift))
    for i in range(3):
        fn = coadd_filenames[7+2*i][10:]
        for phase in phases:
            N = np.loadtxt(os.path.join(
                    filedir, '{}-auto-{}-jackknife_{}-coadd.txt'
                    .format(model_name, coadd_filenames[7+2*i], phase)))
            R = np.loadtxt(os.path.join(
                    filedir, '{}-auto-{}-jackknife_{}-coadd.txt'
                    .format(model_name, coadd_filenames[7+2*i], phase)))
            assert np.all(N[:, 1] == R[:, 1])
            r = N[:, 1]
            xi_0 = N[:, 3] / R[:, 3] * np.square(galaxy_bias)
            xi_2 = N[:, 4] / R[:, 3] * np.square(galaxy_bias)
            xi_0_txt = np.vstack([r, xi_0]).T
            xi_2_txt = np.vstack([r, xi_2]).T
            np.savetxt(os.path.join(
                    filedir,
                    '{}-auto-fftcorr_xi_0-{}-jackknife_{}-coadd.txt'
                    .format(model_name, fn, phase)),
                xi_0_txt, fmt=txtfmt)
            np.savetxt(os.path.join(
                    filedir,
                    '{}-auto-fftcorr_xi_2-{}-jackknife_{}-coadd.txt'
                    .format(model_name, fn, phase)),
                xi_2_txt, fmt=txtfmt)


def do_cov(model_name, N_sub=3, cov_phases=range(16)):

    # for ell in [0, 2]:
    #     paths = []
    #     for phase in cov_phases:
    #         sim_name = '{}_{:02}-{}' \
    #                    .format(sim_name_prefix, cosmology, phase)
    #         paths = paths + glob(os.path.join(
    #                     save_dir, sim_name, 'z{}-r*'.format(redshift),
    #                     '{}-cross_*-xi_{}.txt'.format(model_name, ell)))
    #     # read in all xi files for all phases
    #     xi_list = [np.loadtxt(path)[:, 1] for path in paths]

    # calculate monoquad cov for baofit, 4 types of covariance
    for rec, rnd in zip(['smu-pre-recon', 'smu-post-recon-std'],
                        ['ar', 'sr']):
        xi_list = []
        for phase in cov_phases:
            sim_name = '{}_{:02}-{}'.format(sim_name_prefix, cosmology, phase)
            paths0 = sorted(glob(os.path.join(
                            save_dir, sim_name, 'z{}-r*'.format(redshift),
                            '{}-cross_*-xi_0*{}*{}.txt'
                            .format(model_name, rec, rnd))))
            paths2 = sorted(glob(os.path.join(
                            save_dir, sim_name, 'z{}-r*'.format(redshift),
                            '{}-cross_*xi_2*{}*{}.txt'
                            .format(model_name, rec, rnd))))
            assert len(paths0) == len(paths2)
            if len(paths0) == 0:
                break  # no xi_0 in current phase directory
            for path0, path2 in zip(paths0, paths2):
                # sorted list of paths within a phase guarantees
                # xi0 and xi2 are from the same realisation
                xi_0 = np.loadtxt(path0)[:, 1]
                xi_2 = np.loadtxt(path2)[:, 1]
                xi_list.append(np.hstack((xi_0, xi_2)))
        print('Calculating covariance matrix from {} xi samples...'
              .format(len(xi_list)))
        cov_monoquad = np.cov(np.array(xi_list).T, bias=0)
        # correct for volume and jackknife
        cov_monoquad = cov_monoquad / (np.power(N_sub, 3)-1) / 15
        filepath = os.path.join(  # save to coadd folder for all phases
            save_dir, '{}_{:02}-coadd'.format(sim_name_prefix, cosmology),
            'z{}'.format(redshift),
            '{}-cross-xi_monoquad-cov-{}-{}.txt'.format(model_name, rec, rnd))
        np.savetxt(filepath, cov_monoquad, fmt=txtfmt)
        print('Monoquad covariance matrix saved to: ', filepath)


def combine_galaxy_table_metadata():

    # check if metadata for all models, phases, and realisations exist
    paths = [os.path.join(
                save_dir, 'galaxy_table_meta',
                '{}-z{}-{}-r{}.csv'.format(model_name, redshift, phase, r))
             for model_name in model_names
             for phase in phases
             for r in range(N_reals)]
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


def run_baofit_parallel(baofit_phases=range(16)):

    from baofit import baofit
    filedir = os.path.join(save_dir,
                           '{}_{:02}-coadd'.format(sim_name_prefix, cosmology),
                           'z{}'.format(redshift))
    list_of_inputs = []
    for model_name in model_names:
        # for i in [0]:  # cebug
        for i in range(3):
            xi_type = coadd_filenames[7+2*i][10:]
            for phase in baofit_phases:  # list of inputs for parallelisation
                path_xi_0 = os.path.join(
                    filedir,
                    '{}-auto-fftcorr_xi_0-{}-jackknife_{}-coadd.txt'
                    .format(model_name, xi_type, phase))
                path_xi_2 = os.path.join(
                    filedir,
                    '{}-auto-fftcorr_xi_2-{}-jackknife_{}-coadd.txt'
                    .format(model_name, xi_type, phase))
                if 'pre-recon' in xi_type:
                    path_cov = os.path.join(
                        filedir,
                        '{}-cross-xi_monoquad-cov-smu-pre-recon-ar.txt'
                        .format(model_name))
                elif 'post-recon' in xi_type:
                    path_cov = os.path.join(
                        filedir,
                        '{}-cross-xi_monoquad-cov-smu-post-recon-std-sr.txt'
                        .format(model_name))
                fout_tag = '{}-{}-{}'.format(model_name, phase, xi_type)
                list_of_inputs.append(
                    [path_xi_0, path_xi_2, path_cov, fout_tag])
    with closing(MyPool(processes=N_threads*N_concurrent_reals,
                        maxtasksperchild=1)) as p:
        p.map(baofit, list_of_inputs)
    p.close()
    p.join()


if __name__ == "__main__":

    c_median_poly = fit_c_median(sim_name_prefix, prod_dir, store_dir,
                                 redshift, cosmology)
    for phase in phases:
        halocat = make_halocat(phase)
        process_rockstar_halocat(halocat, N_cut=N_cut)
        for model_name in model_names:
            print('---\nWorking on {} realisations of phase {}, model {}...'
                  .format(N_reals, phase, model_name))
            with closing(MyPool(processes=N_concurrent_reals,
                                maxtasksperchild=1)) as p:
#                p.map(partial(do_galaxy_table,
#                              phase=phase, model_name=model_name,
#                              overwrite=False),
#                      range(N_reals))
                p.map(partial(
                        do_realisation, phase=phase, model_name=model_name,
                        overwrite=True,
                        do_pre_auto_smu=False, do_pre_auto_rppi=True,
                        do_pre_auto_fft=False, do_pre_cross=False,
                        do_recon_std=True, do_post_auto_rppi=True,
                        do_post_cross=False, do_recon_ite=False,
                        do_pre_auto_corr=True, do_post_auto_corr=True,
                        do_pre_cross_corr=False, do_post_cross_corr=False),
                      range(N_reals))
            p.close()
            p.join()
            print('---\nPool closed cleanly for model {}.\n---'
                  .format(model_name))
#        with closing(MyPool(processes=len(model_names),
#                            maxtasksperchild=1)) as p:
#            p.map(partial(coadd_realisations, phase=phase, N_reals=N_reals),
#                  model_names)
#        p.close()
#        p.join()
#    combine_galaxy_table_metadata()
    with closing(MyPool(processes=len(model_names), maxtasksperchild=1)) as p:
        p.map(coadd_phases, model_names)
#        p.map(do_cov, model_names)
#        p.map(convert_recon_correlation, model_names)
#    p.close()
#    p.join()
#    run_baofit_parallel(baofit_phases=phases)
