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
from multiprocessing import Pool
import traceback
import numpy as np
from astropy import table
from halotools.mock_observables import tpcf_multipole
from halotools.mock_observables.two_point_clustering.s_mu_tpcf import (
        spherical_sector_volume)
from halotools.utils import add_halo_hostid
from Corrfunc.theory.DDsmu import DDsmu
from Corrfunc.theory.wp import wp
from abacus_hod import initialise_model, populate_model, vrange
import Halotools as abacus_ht  # Abacus' "Halotools" for importing Abacus
import matplotlib.pyplot as plt
from tqdm import tqdm
# plt.switch_backend('Agg')  # switch this on if backend error is reported

# %% custom settings
sim_name_prefix = 'emulator_1100box_planck'
tagout = 'recon'  # 'z0.5'
phases = range(2)  # range(16)  # [0, 1] # list(range(16))
cosmology = 0  # one cosmology at a time instead of 'all'
redshift = 0.5  # one redshift at a time instead of 'all'
model_names = ['gen_base1', 'gen_base4', 'gen_base5',
               'gen_ass1', 'gen_ass2', 'gen_ass3',
               'gen_ass1_n', 'gen_ass2_n', 'gen_ass3_n',
               'gen_s1', 'gen_sv1', 'gen_sp1',
               'gen_s1_n', 'gen_sv1_n', 'gen_sp1_n',
               'gen_vel1', 'gen_allbiases', 'gen_allbiases_n']
model_names = ['gen_base1']
N_reals = 1  # number of realisations for an HOD
N_cut = 70  # number particle cut, 70 corresponds to 4e12 Msun
N_threads = 30  # for a single MP pool thread
N_sub = 3  # number of subvolumes per dimension

# %% flags
reuse_galaxies = True
save_hod_realisation = True
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
pi_max = 150  # for counting
rp_bins = s_bins

# %% fixed catalogue parameters
prod_dir = r'/mnt/gosling2/bigsim_products/emulator_1100box_planck_products/'
save_dir = os.path.join('/home/dyt/store/', sim_name_prefix+'-'+tagout)
halo_type = 'Rockstar'
halo_m_prop = 'halo_mvir'  # mgrav is better but not using sub or small haloes
txtfmt = b'%.30e'

coadd_filenames = ['wp-pre_recon',
                   'xi-smu-pre_recon-ar', 'xi-smu-pre_recon-nr',
                   'xi_0-smu-pre_recon-ar', 'xi_0-smu-pre_recon-nr',
                   'xi_2-smu-pre_recon-ar', 'xi_2-smu-pre_recon-nr',
                   'wp-post_recon',
                   'xi-smu-post_recon-ar', 'xi-smu-post_recon-nr',
                   'xi_0-smu-post_recon-ar', 'xi_0-smu-post_recon-nr',
                   'xi_2-smu-post_recon-ar', 'xi_2-smu-post_recon-nr']

# # %% MP Class
# class NoDaemonProcess(Process):
#     # make 'daemon' attribute always return False
#     def _get_daemon(self):
#         return False
#     def _set_daemon(self, value):
#         pass
#     daemon = property(_get_daemon, _set_daemon)
# # We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# # because the latter is only a wrapper function, not a proper class.
# class Pool(pool.Pool):
#     Process = NoDaemonProcess


# %% definitionss of statisical formulae
def analytic_random(ND1, NR1, ND2, NR2, s_bins, mu_bins, V):
    '''
    V is the global volume L^3
    for auto-correlation, we may set ND1 = NR1 = ND2 = NR2
    for cross-correlation, returns D1R2 and R1D2 for LS estimator
    '''

    mu_bins_reverse_sorted = np.sort(mu_bins)[::-1]
    dv = spherical_sector_volume(s_bins, mu_bins_reverse_sorted)
    dv = np.diff(dv, axis=1)  # volume of wedges
    dv = np.diff(dv, axis=0)  # volume of wedge sector
    DR = ND1*NR2/V*dv
    RD = NR1*ND2/V*dv
    RR = NR1*NR2/V*dv  # calculate the random-random pairs
    return DR, RD, RR


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


def ls_estimator(ND1, NR1, ND2, NR2, DD, DR, RD, RR):

    '''
    generalised Landy & Szalay (1993) estimator for cross-correlation
    reduces to (DD-2DR+RR)RR in case of auto-correlation if we set
    ND1 = ND2, NR1 = NR1, and DR = RD

    ref: http://d-scholarship.pitt.edu/20997/1/DanMatthews_thesis.pdf
    '''
    xi = (DD*(NR1*NR2/ND1/ND2) - DR*(NR1/ND1) - RD*(NR2/ND2) + RR) / RR
    return xi


def rebin_smu_counts(cts):

    '''
    input counts is the output table of DDSmu, saved as npy file
    output npairs is a 2D histogram of dimensions
    n_s_bins by n_mu_bins matrix (30 by 20 by default)
    savg is the paircount-weighted average of s in the bin, same dimension

    '''

    npairs = np.zeros((s_bins.size-1, mu_bins.size-1), dtype=np.int64)
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
        savg[m, n] = np.sum(arr['savg'] * arr['npairs'] / npairs[m, n])
        bins_included = bins_included + cts['npairs'][mask].size
    # print('Total DD fine bins: {}. Total bins included: {}'
    #       .format(cts.size, bins_included))
    # print('Total npairs in fine bins: {}. Total npairs after rebinning: {}'
    #       .format(cts['npairs'].sum(), npairs.sum()))
    indices = np.where(npairs == 0)  # check if all bins are filled
    if len(indices[0]) != 0:  # print problematic bins if any is empty
        for i in range(len(indices[0])):
            print('({}, {}) bin is empty'.format(indices[0][i], indices[1][i]))

    return npairs, savg


def xi1d_list_to_cov(xi_list):
    '''
    rows are bins and columns are sample values for each bin
    '''

    print('Calculating covariance matrix from {} xi multipole matrices...'
          .format(len(xi_list)))
    return np.cov(np.array(xi_list).T, bias=0)


def coadd_correlation(corr_list):

    '''
    input is a list of xi or wp samples

    '''
    arr = np.array(corr_list)
    coadd = np.mean(arr, axis=0)
    err = np.std(arr, axis=0)
    if arr.shape[2] == 2:
        # each input array in list is of shape (:, 2), and 1st column is r
        err[:, 0] = coadd[:, 0]
    return coadd, err


# %% functions for tasks

def make_halocat(phase):

    print('---\n'
          + 'Importing phase {} halo catelogue from: \n'.format(phase)
          + '{}{}_{:02}-{}_products/z{}/'
            .format(prod_dir, sim_name_prefix, cosmology, phase, redshift)
          + '\n---')

    halocats = abacus_ht.make_catalogs(
        sim_name=sim_name_prefix, products_dir=prod_dir,
        redshifts=[redshift], cosmologies=[cosmology], phases=[phase],
        halo_type=halo_type,
        load_halo_ptcl_catalog=True,  # this loads 10% particle subsamples
        load_ptcl_catalog=False,  # this loads uniform subsamples, dnw
        load_pids=False)

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
    halos selected in contiguous blocks. Also we force subsamp_len > 0 so that
    halo contains at least one subsample particle.

    '''

    print('Applying mass cut N = {} to halocat and re-organising subsamples...'
          .format(N_cut))
    N0 = len(halocat.halo_table)
    mask_halo = ((halocat.halo_table['halo_upid'] == -1)  # only host halos
                 & (halocat.halo_table['halo_N'] >= N_cut))
    #             & (halocat.halo_table['halo_subsamp_len'] > 0))
    mask_subhalo = ((halocat.halo_table['halo_upid'] != -1)  # child subhalos
                    & np.isin(
                            halocat.halo_table['halo_upid'],
                            halocat.halo_table['halo_id'][mask_halo]))
    htab = halocat.halo_table[mask_halo | mask_subhalo]  # original order
    print('Locating relevant halo host ids in halo table...')
    hostids, hidx, hidx_split = find_repeated_entries(htab['halo_hostid'].data)
    print('Creating subsample particle indices...')
    pidx = vrange(htab['halo_subsamp_start'][hidx].data,
                  htab['halo_subsamp_len'][hidx].data)
    print('Re-arranging particle table...')
    ptab = halocat.halo_ptcl_table[pidx]
    # ss_len = [np.sum(htab['halo_subsamp_len'][i]) for i in tqdm(hidx_split)]
    print('Rewriting halo_table subsample fields with MP...')
    # with closing(Pool(processes=16, maxtasksperchild=1)) as p:
    with closing(Pool(processes=16)) as p:
        ss_len = p.map(partial(sum_lengths, len_arr=htab['halo_subsamp_len']),
                       hidx_split)
    p.close()
    p.join()
    htab = htab[htab['halo_upid'] == -1]  # drop subhalos now that ptcl r done
    assert len(htab) == len(hostids)  # sanity check, hostids are unique values
    htab = htab[htab['halo_hostid'].data.argsort()]
    htab['halo_subsamp_len'] = ss_len  # overwrite halo_subsamp_len field
    assert np.sum(htab['halo_subsamp_len']) == len(ptab)
    htab['halo_subsamp_start'][0] = 0  # reindex halo_subsamp_start
    htab['halo_subsamp_start'][1:] = htab['halo_subsamp_len'].cumsum()[:-1]
    assert (htab['halo_subsamp_start'][-1] + htab['halo_subsamp_len'][-1]
            == len(ptab))
    halocat.halo_table = htab
    halocat.halo_ptcl_table = ptab

    print('Initial total N halos {}; mass cut mask count {}; '
          'subhalos cut mask count {}; final N halos {}. '
          'Smallest subsamp_len = {}.'
          .format(N0, mask_halo.sum(), mask_subhalo.sum(), len(htab),
                  htab['halo_subsamp_len'].min()))

    return halocat


def fit_c_median(phases=range(16)):

    '''
    median concentration as a function of log halo mass in Msun/h
    c_med(log(m)) = poly(log(m))

    '''
    # check if output file already exists
    if os.path.isfile(os.path.join(
            '/home/dyt/store/', sim_name_prefix, 'c_median_poly.txt')):
        return None
    # load 16 phases for the given cosmology and redshift
    print('Loading halo catalogues...')
    halocats = abacus_ht.make_catalogs(
        sim_name=sim_name_prefix, products_dir=prod_dir,
        redshifts=[redshift], cosmologies=[cosmology], phases=phases,
        halo_type=halo_type,
        load_halo_ptcl_catalog=True,  # this loads 10% particle subsamples
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


def do_auto_correlation(model, mode='smu-pre_recon',
                        use_numerical_randoms=True):

    '''
    mode is a string that contains 'smu' or 'wp', and can include
    pre/post-reconstruction tag which gets saved as part of filename
    '''
    # do pair counting in 1 Mpc bins
    sim_name = model.mock.header['SimName']
    L = model.mock.BoxSize
    Lbox = model.mock.Lbox
    redshift = model.mock.redshift
    model_name = model.model_name
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
        DDnpy = DDsmu(1, N_threads, s_bins_counts, mu_max, n_mu_bins,
                      x, y, z, periodic=True, verbose=False, boxsize=L,
                      output_savg=True, c_api_timer=False)
        DD, savg = rebin_smu_counts(DDnpy)  # re-count pairs in new bins
        # create r vector from weighted average of DD pair counts
        savg_vec = np.sum(savg * DD, axis=1) / np.sum(DD, axis=1)
        # save counting results as original structured array in npy format
        np.save(os.path.join(filedir,
                             '{}-auto-paircount-DD-{}.npy'
                             .format(model_name, mode)),
                DDnpy)
        np.savetxt(os.path.join(filedir, '{}-auto-paircount-DD-{}-rebinned.txt'
                                .format(model_name, mode)),
                   DD, fmt=txtfmt)
        model.ND = ND = len(model.mock.galaxy_table)
        NR = 200 * ND
        DR_ar, _, RR_ar = analytic_random(
                ND, NR, ND, NR, s_bins, mu_bins, Lbox.prod())
        np.savetxt(os.path.join(filedir,
                                '{}-auto-paircount-DR-{}-ar.txt'
                                .format(model_name, mode)),
                   DR_ar, fmt=txtfmt)
        np.savetxt(os.path.join(filedir,
                                '{}-auto-paircount-RR-{}-ar.txt'
                                .format(model_name, mode)),
                   RR_ar, fmt=txtfmt)
        xi_s_mu = ls_estimator(ND, NR, ND, NR, DD, DR_ar, DR_ar, RR_ar)
        xi_0 = tpcf_multipole(xi_s_mu, mu_bins, order=0)
        xi_2 = tpcf_multipole(xi_s_mu, mu_bins, order=2)
        xi_0_txt = np.vstack([savg_vec, xi_0]).T
        xi_2_txt = np.vstack([savg_vec, xi_2]).T
        for output, tag in zip([xi_s_mu, xi_0_txt, xi_2_txt],
                               ['xi', 'xi_0', 'xi_2']):
            np.savetxt(os.path.join(filedir, '{}-auto-{}-{}-ar.txt'
                                    .format(model_name, tag, mode)),
                       output, fmt=txtfmt)
        if use_numerical_randoms:
            if hasattr(model.mock, 'numerical_randoms'):
                model.NR = NR = model.mock.numerical_randoms.shape[0]
                xr = model.mock.numerical_randoms[:, 0]
                yr = model.mock.numerical_randoms[:, 1]
                zr = model.mock.numerical_randoms[:, 2]
            else:
                print('No grid numerical randoms found. Assuming NR=ND.')
                model.NR = NR = model.ND
                xr = np.random.uniform(0, L, NR).astype(np.float32)
                yr = np.random.uniform(0, L, NR).astype(np.float32)
                zr = np.random.uniform(0, L, NR).astype(np.float32)
            DRnpy = DDsmu(0, N_threads, s_bins_counts, mu_max, n_mu_bins,
                          x, y, z, X2=xr, Y2=yr, Z2=zr,
                          periodic=True, verbose=True, output_savg=True,
                          boxsize=L, c_api_timer=False)
            RRnpy = DDsmu(1, N_threads, s_bins_counts, mu_max, n_mu_bins,
                          xr, yr, zr,
                          periodic=True, verbose=True, output_savg=True,
                          boxsize=L, c_api_timer=False)
            DR_nr, _ = rebin_smu_counts(DRnpy)
            RR_nr, _ = rebin_smu_counts(RRnpy)
            np.save(os.path.join(filedir, '{}-auto-paircount-DR-{}-nr.npy'
                                 .format(model_name, mode)),
                    DRnpy)
            np.save(os.path.join(filedir, '{}-auto-paircount-RR-{}-nr.npy'
                                 .format(model_name, mode)),
                    RRnpy)
            np.savetxt(os.path.join(filedir,
                                    '{}-auto-paircount-DR-{}-nr_rebinned.txt'
                                    .format(model_name, mode)),
                       DR_nr, fmt=txtfmt)
            np.savetxt(os.path.join(filedir,
                                    '{}-auto-paircount-RR-{}-nr_rebinned.txt'
                                    .format(model_name, mode)),
                       RR_nr, fmt=txtfmt)
            for output, tag in zip([xi_s_mu, xi_0_txt, xi_2_txt],
                                   ['xi', 'xi_0', 'xi_2']):
                np.savetxt(os.path.join(filedir, '{}-auto-{}-{}-nr.txt'
                                        .format(model_name, tag, mode)),
                           output, fmt=txtfmt)
    elif 'wp' in mode:  # also returns wp in addition to counts
        DD = wp(L, pi_max, N_threads, rp_bins, x, y, z,
                output_rpavg=True, verbose=False, c_api_timer=False)
        wp_txt = np.vstack([DD['rpavg'], DD['wp']]).T
        np.savetxt(os.path.join(filedir, '{}-auto-{}.txt'
                                .format(model_name, mode)),
                   wp_txt, fmt=txtfmt)


def do_subcross_correlation(model, N_sub=3, mode='post_recon',
                            use_numerical_randoms=False):
    '''
    cross-correlation between 1/N_sub^3 of a box and the whole box
    can be pre/post-reconstruction data
    '''
    sim_name = model.mock.header['SimName']
    model_name = model.model_name
    L = model.mock.BoxSize
    L_sub = L / N_sub  # length of subvolume box
    Lbox = model.mock.Lbox
    redshift = model.mock.redshift
    gt = model.mock.galaxy_table
    x1, y1, z1 = gt['x'], gt['y'], gt['z']
    ND1 = len(gt)  # number of galaxies in entire volume of periodic box
    filedir = os.path.join(save_dir, sim_name,
                           'z{}-r{}'.format(redshift, model.r))
    for i, j, k in product(range(N_sub), repeat=3):
        linind = i*N_sub**2 + j*N_sub + k  # linearised index of subvolumes
        mask = ((L_sub * i < gt['x']) & (gt['x'] <= L_sub * (i+1)) &
                (L_sub * j < gt['y']) & (gt['y'] <= L_sub * (j+1)) &
                (L_sub * k < gt['z']) & (gt['z'] <= L_sub * (k+1)))
        ND2 = len(gt[mask])  # number of galaxies in the subvolume
        x2, y2, z2 = gt[mask]['x'], gt[mask]['y'], gt[mask]['z']
        DDnpy = DDsmu(0, N_threads, s_bins_counts, mu_max, n_mu_bins,
                      x1, y1, z1, X2=x2, Y2=y2, Z2=z2,
                      periodic=True, output_savg=True, verbose=False,
                      boxsize=L, c_api_timer=False)
        # re-bin DD counts
        DD, _ = rebin_smu_counts(DDnpy)
        np.save(os.path.join(filedir,
                             '{}-cross_{}-paircount-DD.npy'
                             .format(model_name, linind)),
                DDnpy)
        # use_analytic_randoms:
        NR1, NR2 = 200 * ND1, 200 * ND2
        DR, RD, RR = analytic_random(
                ND1, NR1, ND2, NR2, s_bins, mu_bins, Lbox.prod())
        for output, tag in zip([DR, RD, RR], ['DR', 'RD', 'RR']):
            np.savetxt(os.path.join(filedir,
                                    '{}-cross_{}-paircount-{}-{}-ar.txt'
                                    .format(model_name, linind, tag, mode)),
                       output, fmt=txtfmt)
        # calculate cross-correlation from counts using ls estimator
        xi_s_mu = ls_estimator(ND1, NR1, ND2, NR2, DD, DR, RD, RR)
        xi_0 = tpcf_multipole(xi_s_mu, mu_bins, order=0)
        xi_2 = tpcf_multipole(xi_s_mu, mu_bins, order=2)
        xi_0_txt = np.vstack([s_bins_centre, xi_0]).T
        xi_2_txt = np.vstack([s_bins_centre, xi_2]).T
        for output, tag in zip([xi_s_mu, xi_0_txt, xi_2_txt],
                               ['xi', 'xi_0', 'xi_2']):
            np.savetxt(os.path.join(filedir,
                                    '{}-cross_{}-{}-{}-ar.txt'
                                    .format(model_name, linind, tag, mode)),
                       output, fmt=txtfmt)
        if use_numerical_randoms:
            # calculate D2Rbox by brute force sampling, where
            # Rbox sample is homogeneous in volume of box
            # force float32 (a.k.a. single precision float in C)
            # to be consistent with galaxy table precision
            # otherwise corrfunc raises an error
            if hasattr(model.mock, 'numerical_randoms'):
                nr = model.mock.random_array  # random shifted array
                xr1, yr1, zr1 = nr[:, 0], nr[:, 1], nr[:, 2]
                mask = ((L_sub * i < xr1) & (xr1 <= L_sub * (i+1)) &
                        (L_sub * j < yr1) & (yr1 <= L_sub * (j+1)) &
                        (L_sub * k < zr1) & (zr1 <= L_sub * (k+1)))
                xr2, yr2, zr2 = xr1[mask], yr1[mask], zr1[mask]
                NR1 = xr1.size
                NR2 = xr2.size
            else:
                xr1 = np.random.uniform(0, L, NR1).astype(np.float32)
                yr1 = np.random.uniform(0, L, NR1).astype(np.float32)
                zr1 = np.random.uniform(0, L, NR1).astype(np.float32)
                xr2 = np.random.uniform(0, L_sub, NR2).astype(np.float32)
                yr2 = np.random.uniform(0, L_sub, NR2).astype(np.float32)
                zr2 = np.random.uniform(0, L_sub, NR2).astype(np.float32)
            DRnpy = DDsmu(0, N_threads, s_bins_counts, mu_max, n_mu_bins,
                          x1, y1, z1, X2=xr2, Y2=yr2, Z2=zr2,
                          periodic=True, verbose=False, output_savg=True,
                          boxsize=L, c_api_timer=False)
            RDnpy = DDsmu(0, N_threads, s_bins_counts, mu_max, n_mu_bins,
                          xr1, yr1, zr1, X2=x2, Y2=y2, Z2=z2,
                          periodic=True, verbose=False, output_savg=True,
                          boxsize=L, c_api_timer=False)
            RRnpy = DDsmu(0, N_threads, s_bins_counts, mu_max, n_mu_bins,
                          xr1, yr1, zr1, X2=xr2, Y2=yr2, Z2=zr2,
                          periodic=True, verbose=False, output_savg=True,
                          boxsize=L, c_api_timer=False)
            for npy, txt, tag in zip([DRnpy, RDnpy, RRnpy],
                                     [DR, RD, RR],
                                     ['DR', 'RD', 'RR']):
                np.save(os.path.join(
                        filedir, '{}-cross_{}-paircount-{}-{}-nr.npy'
                        .format(model_name, linind, tag, mode)),
                        npy)
                np.savetxt(os.path.join(
                        filedir, '{}-cross_{}-paircount-{}-{}-nr_rebinned.txt'
                        .format(model_name, linind, tag, mode)),
                    txt, fmt=txtfmt)
            DR, _ = rebin_smu_counts(DRnpy)
            RD, _ = rebin_smu_counts(RDnpy)
            RR, _ = rebin_smu_counts(RRnpy)
            xi_s_mu = ls_estimator(ND1, NR1, ND2, NR2, DD, DR, RD, RR)
            xi_0 = tpcf_multipole(xi_s_mu, mu_bins, order=0)
            xi_2 = tpcf_multipole(xi_s_mu, mu_bins, order=2)
            xi_0_txt = np.vstack([s_bins_centre, xi_0]).T
            xi_2_txt = np.vstack([s_bins_centre, xi_2]).T
            for output, tag in zip([xi_s_mu, xi_0_txt, xi_2_txt],
                                   ['xi', 'xi_0', 'xi_2']):
                np.savetxt(os.path.join(
                        filedir,
                        '{}-cross_{}-{}-{}-nr.txt'
                        .format(model_name, linind, tag, mode)),
                    output, fmt=txtfmt)


def coadd_phases(model_name, coadd_phases=range(16)):

    print('Coadding xi from all phases for model {}...'.format(model_name))
    filedir = os.path.join(save_dir,
                           '{}_{:02}-coadd'.format(sim_name_prefix, cosmology),
                           'z{}'.format(redshift))
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    for fn in coadd_filenames:
        temp = os.path.join(
            save_dir,
            '{}_{:02}-[0-9]*'.format(sim_name_prefix, cosmology),
            'z{}'.format(redshift), '*'+fn+'*')
        paths = glob(temp)
        print('temp is:', temp)
        print('paths are:', paths)
        assert len(paths) == len(coadd_phases)
        corr_list = [np.loadtxt(path) for path in paths]
        coadd, error = coadd_correlation(corr_list)
        np.savetxt(os.path.join(
                filedir, '{}-auto-{}-coadd.txt'.format(model_name, fn)),
            coadd, fmt=txtfmt)
        np.savetxt(os.path.join(
                filedir, '{}-auto-{}-error.txt'.format(model_name, fn)),
            error, fmt=txtfmt)
        for phase in coadd_phases:  # jackknife delete-one
            jk_list = corr_list[:phase] + corr_list[phase+1:]
            coadd, error = coadd_correlation(jk_list)
            np.savetxt(os.path.join(
                    filedir, '{}-auto-{}-jackknife_{}-coadd.txt'
                    .format(model_name, fn, phase)),
                coadd, fmt=txtfmt)
            np.savetxt(os.path.join(
                    filedir, '{}-auto-{}-jackknife_{}-error.txt'
                    .format(model_name, fn, phase)),
                error, fmt=txtfmt)


def do_cov(model_name, N_sub=3, cov_phases=range(16)):

    # # calculate cov combining all phases, 16 * N_sub^3 * N_reals
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
    #     # as cov gets smaller, chisq gets bigger, contour gets smaller
    #     cov = xi1d_list_to_cov(xi_list) / (np.power(N_sub, 3)-1) / 15
    #     # save cov
    #     filepath = os.path.join(  # save to coadd folder for all phases
    #             save_dir,
    #             '{}_{:02}-coadd'.format(sim_name_prefix, cosmology),
    #             'z{}'.format(redshift),
    #             '{}-cross-xi_{}-cov.txt'.format(model_name, ell))
    #     np.savetxt(filepath, cov, fmt=txtfmt)

    # calculate monoquad cov for baofit, 4 types of covariance
    for recon, rand in product(['pre_recon', 'post_recon'], ['ar', 'nr']):
        xi_list = []
        for phase in cov_phases:
            sim_name = '{}_{:02}-{}'.format(sim_name_prefix, cosmology, phase)
            paths0 = sorted(glob(os.path.join(
                            save_dir, sim_name, 'z{}-r*'.format(redshift),
                            '{}-cross_*xi_0*{}*{}.txt'
                            .format(model_name, recon, rand))))
            paths2 = sorted(glob(os.path.join(
                            save_dir, sim_name, 'z{}-r*'.format(redshift),
                            '{}-cross_*xi_2*{}*{}.txt'
                            .format(model_name, recon, rand))))
            assert len(paths0) == len(paths2)
            if len(paths0) == 0:
                break
            for path0, path2 in zip(paths0, paths2):
                # sorted list of paths within a phase guarantees
                # xi0 and xi2 are from the same realisation
                xi_0 = np.loadtxt(path0)[:, 1]
                xi_2 = np.loadtxt(path2)[:, 1]
                xi_list.append(np.hstack((xi_0, xi_2)))
        cov_monoquad = xi1d_list_to_cov(xi_list) / (np.power(N_sub, 3)-1) / 15
        filepath = os.path.join(  # save to coadd folder for all phases
                save_dir,
                '{}_{:02}-coadd'.format(sim_name_prefix, cosmology),
                'z{}'.format(redshift),
                '{}-cross-xi_monoquad-cov-{}-{}.txt'
                .format(model_name, recon, rand))
        np.savetxt(filepath, cov_monoquad, fmt=txtfmt)
        # print('Monoquad covariance matrix saved to: ', filepath)


def do_realisation(r, model_name):

    global halocat, use_analytic_randoms
    try:  # this helps catch exception in a worker thread of multiprocessing
        phase = halocat.ZD_Seed
        sim_name = '{}_{:02}-{}'.format(sim_name_prefix, cosmology, phase)
        recon_dir = os.path.join(save_dir, sim_name,
                                 'z{}-r{}'.format(redshift, r))
        gt_path = os.path.join(recon_dir, '{}-galaxy_table.csv'
                               .format(model_name))
        # all realisations in parallel, each model needs to be instantiated
        model = initialise_model(halocat.redshift, model_name,
                                 halo_m_prop=halo_m_prop)
        model.N_cut = N_cut  # add useful model properties here
        model.c_median_poly = np.poly1d(np.loadtxt(os.path.join(
                '/home/dyt/store/', sim_name_prefix, 'c_median_poly.txt')))
        model.r = r
        # random seed using phase and realisation index r, model independent
        seed = phase*100 + r
        np.random.seed(seed)
        print('Working on r = {} for phase {}, model {}...'
              .format(r, phase, model_name))
        if reuse_galaxies:
            gt_path_input = gt_path
        else:
            gt_path_input = None
        model = populate_model(halocat, model, gt_path=gt_path_input,
                               add_rsd=add_rsd, N_threads=N_threads)
        gt = model.mock.galaxy_table
        model.ND = len(gt)
        # save galaxy table
        if not reuse_galaxies and save_hod_realisation:
            print('Saving galaxy table for r = {} ...'.format(r))
            try:
                os.makedirs(os.path.dirname(gt_path))
            except OSError:
                pass
            gt.write(gt_path, format='ascii.fast_csv', overwrite=True)
        # save galaxy table metadata to sim directory, one file for each
        # to be combined later to avoid threading conflicts
        N_cen = np.sum(gt['gal_type'] == 'centrals')
        N_sat = np.sum(gt['gal_type'] == 'satellites')
        assert N_cen + N_sat == model.ND
        gt_meta = table.Table(
                [[model_name], [phase], [r], [seed],
                 [N_cen], [N_sat], [model.ND]],
                names=('model name', 'phase', 'realisation', 'seed',
                       'N centrals', 'N satellites', 'N galaxies'),
                dtype=('S30', 'i4', 'i4', 'i4', 'i4', 'i4', 'i4'))
        if not os.path.exists(os.path.join(save_dir, 'galaxy_table_meta')):
            os.makedirs(os.path.join(save_dir, 'galaxy_table_meta'))
        gt_meta.write(os.path.join(
            save_dir, 'galaxy_table_meta',
            '{}-z{}-{}-r{}.csv'.format(model_name, redshift, phase, r)),
            format='ascii.fast_csv', overwrite=True)
        # pre-reconstruction auto-correlation with FT
        print('pre-recon, FT')
        m = np.zeros(len(gt))
        arr = np.array([m, gt['x'], gt['y'], gt['z']]).T.astype(np.float32)
        arr.tofile('/home/dyt/store/recon_temp/gal_cat-{}.dat'.format(seed),
                   sep='')
        subprocess.call(['python', './recon/read/read.py',
                         str(phase), str(seed)])
        subprocess.call(['python', './recon/reconstruct/reconst.py',
                         '0', str(phase), str(seed), model_name, recon_dir])
        nr = np.float32(np.fromfile('/home/dyt/store/recon_temp/file_R-{}'
                                    .format(seed),
                                    dtype=np.float64)[8:].reshape(-1, 4))
        nr[nr < 0] += 1100  # read.cpp re-wraps data above 550 to negative pos
        print('nr max, min', nr.max(), nr.min())
        model.mock.numerical_randoms = nr
        # pre-recon auto-corr with pair-counting, analytical & numerical random
        print('pre-recon, pair-counting')
        do_auto_correlation(model, mode='smu-pre_recon',
                            use_numerical_randoms=True)
        do_auto_correlation(model, mode='wp-pre_recon')
        do_subcross_correlation(model, N_sub=N_sub, mode='pre_recon',
                                use_numerical_randoms=False)
        # apply standard reconstruction and calculate covariance matrix
        print('standard recon')
        subprocess.call(['python', './recon/reconstruct/reconst.py',
                         '1', str(phase), str(seed), model_name, recon_dir])
        ds = np.float32(np.fromfile('/home/dyt/store/recon_temp/file_D-{}_rec'
                                    .format(seed),
                                    dtype=np.float64)[8:].reshape(-1, 4))
        nr = np.float32(np.fromfile('/home/dyt/store/recon_temp/file_R-{}_rec'
                                    .format(seed),
                                    dtype=np.float64)[8:].reshape(-1, 4))
        ds[ds < 0] += 1100  # undo wrapping by read.cpp for data_shifted
        nr[nr < 0] += 1100  # undo wrapping by read.cpp for numerical randoms
        gt['x'], gt['y'], gt['z'] = ds[:, 0], ds[:, 1], ds[:, 2]
        model.mock.numerical_randoms = nr
        do_auto_correlation(model, mode='smu-post_recon',
                            use_numerical_randoms=True)
        do_auto_correlation(model, mode='wp-post_recon')
        do_subcross_correlation(model, N_sub=N_sub, mode='post_recon',
                                use_numerical_randoms=False)
        print('Finished r = {}.'.format(r))
    except Exception as E:
        print('Exception caught in worker thread r = {}'.format(r))
        traceback.print_exc()
        raise E


def do_realisations(halocat, model_name, phase, N_reals):

    '''
    generates n HOD realisations for a given (phase, model)
    then co-add them and get a single set of xi data
    the rest of the programme will work as if there is only 1 realisation
    '''
    # sim_name = '{}_{:02}-{}'.format(sim_name_prefix, cosmology, phase)
    sim_name = halocat.header['SimName']
    print('---\nWorking on {} realisations of {}, model {}...\n---\n'
          .format(N_reals, sim_name, model_name))
    # create n_real realisations of the given HOD model
    with closing(Pool(processes=8, maxtasksperchild=1)) as p:
        p.map(partial(do_realisation, model_name=model_name),
              range(N_reals))
    print('---\nPool closed cleanly for {} realisations of model {}.\n---'
          .format(N_reals, model_name))


def coadd_realisations(model_name, phase, N_reals):

    # Co-add auto-correlation results from all realisations
    sim_name = '{}_{:02}-{}'.format(sim_name_prefix, cosmology, phase)
    filedir = os.path.join(save_dir, sim_name, 'z{}'.format(redshift))
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    print('Coadding {} realisations for phase {}...'.format(N_reals, phase))
    for fn in coadd_filenames:
        temp = os.path.join(
            save_dir, sim_name, 'z{}-r*'.format(redshift), '*'+fn+'*')
        paths = glob(temp)
        print('temp is:', temp)
        print('paths are:', paths)
        assert len(paths) == N_reals
        corr_list = [np.loadtxt(path) for path in paths]
        coadd, error = coadd_correlation(corr_list)
        np.savetxt(os.path.join(
                filedir, '{}-auto-{}-coadd.txt'.format(model_name, fn)),
            coadd, fmt=txtfmt)
        np.savetxt(os.path.join(
                filedir, '{}-auto-{}-error.txt'.format(model_name, fn)),
            error, fmt=txtfmt)


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
    tables = [table.Table.read(path) for path in tqdm(paths)]
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
        for phase in baofit_phases:  # list of inputs for parallelisation
            path_xi_0 = os.path.join(filedir,
                                     '{}-auto-xi_0-ca-jackknife_{}.txt'
                                     .format(model_name, phase))
            path_xi_2 = os.path.join(filedir,
                                     '{}-auto-xi_2-ca-jackknife_{}.txt'
                                     .format(model_name, phase))
            path_cov = os.path.join(filedir, '{}-cross-xi_monoquad-cov.txt'
                                    .format(model_name))
            fout_tag = '{}-{}'.format(model_name, phase)
            list_of_inputs.append([path_xi_0, path_xi_2, path_cov, fout_tag])
    with closing(Pool(16)) as p:
        p.map(baofit, list_of_inputs)


if __name__ == "__main__":

    halocats = fit_c_median(phases=phases)
    for phase in phases:
        try:
            halocat = halocats[phase]
        except TypeError:
            halocat = make_halocat(phase)
            halocat = process_rockstar_halocat(halocat, N_cut)
        for model_name in model_names:
            do_realisations(halocat, model_name, phase, N_reals)
            coadd_realisations(model_name, phase, N_reals)

    for model_name in model_names:
        coadd_phases(model_name, coadd_phases=phases)
        do_cov(model_name, N_sub=N_sub, cov_phases=phases)
    combine_galaxy_table_metadata()

    # run_baofit_parallel(baofit_phases=range(16))
