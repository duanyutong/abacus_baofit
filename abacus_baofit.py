# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 17:30:51 2017

@author: Duan Yutong
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import numpy as np
import os
from glob import glob
import itertools
import pickle
from halotools.mock_observables import (
        return_xyz_formatted_array, rp_pi_tpcf, rp_pi_tpcf_jackknife,
        s_mu_tpcf, tpcf_multipole)
from halotools.mock_observables.two_point_clustering.s_mu_tpcf import (
    spherical_sector_volume)
from halotools.empirical_models import PrebuiltHodModelFactory
from halotools.utils import add_halo_hostid
import Halotools as ht  # "Halotools" for importing Abacus catalogues
from Corrfunc.theory.DDsmu import DDsmu
# from Corrfunc.utils import convert_3d_counts_to_cf

# catalogue parameters
sim_name_prefix = 'emulator_1100box_planck'
prod_dir = r'/mnt/gosling2/bigsim_products/emulator_1100box_planck_products/'
halo_type = 'Rockstar'
redshift = 0.700  # one redshift at a time instead of 'all'
cosmology = 0  # use with only one cosmology at a time instead of 'all'
phases = list(range(16))  # [0, 1] # list(range(16)) # 'all'

# HOD models
prebuilt_models = ['zheng07']
# prebuilt_models = ['zheng07', 'leauthaud11', 'tinker13', 'hearin15',
#                   'zu_mandelbaum15', 'zu_mandelbaum16', 'cacciato09']
step_s_bins = 5  # mpc/h
step_mu_bins = 0.05
n_threads = 10
save_dir = os.path.join('/home/dyt/analysis_data/', sim_name_prefix)
# n_s_bins = 13  # not used
# n_mu_bins = 15
use_corrfunc = True
# use_jackknife = False
txtfmt = b'%.30e'


def ph_estimator(ND, NR, DD, RR):

    ''' DD/RR - 1 estimator '''
    if DD.shape == RR.shape:
        xi = NR*(NR-1)/ND/(ND-1)*DD/RR - 1
        return xi
    else:
        print('Warning: DD, RR dimensions mismatch')
        return None


def analytic_random_counts(N1, s_bins, mu_bins, Lbox, N2=0):
    ''' for auto-correlation only '''
    NR = N1
    mu_bins_reverse_sorted = np.sort(mu_bins)[::-1]
    dv = spherical_sector_volume(s_bins, mu_bins_reverse_sorted)
    dv = np.diff(dv, axis=1)  # volume of wedges
    dv = np.diff(dv, axis=0)  # volume of wedge sector
    V = model.mock.Lbox.prod()  # volume of entire simulation box
    # calculate randoms for sample1 and 2
    D1R = N1*(N1-1)/V*dv
    D2R = N2*(N2-1)/V*dv
    # calculate the random-random pairs.
    RR = NR*(NR-1)/V*dv

    return D1R, D2R, RR


def make_halocat(phase):

    print('Importing halo catelogue: {}-{}/z{}'.format(
        prod_dir + sim_name_prefix + cosmology, phase, redshift))
    halocats = ht.make_catalogs(
        sim_name=sim_name_prefix, products_dir=prod_dir,
        redshifts=[redshift], cosmologies=[cosmology], phases=[phase],
        halo_type=halo_type,
        load_halo_ptcl_catalog=False,  # this loads subsamples, does not work
        load_ptcl_catalog=False,  # this loads uniform subsamples, dnw
        load_pids='auto')

    halocat = halocats[0][0][0]

    # fill in fields required by HOD models
    halocat.halo_table['halo_mvir'] = halocat.halo_table['halo_m']
    halocat.halo_table['halo_rvir'] = halocat.halo_table['halo_r']
    add_halo_hostid(halocat.halo_table, delete_possibly_existing_column=True)
    # setting NFW concentration, klypin_rs is more stable
    # can be improved following https://arxiv.org/abs/1709.07099
    halocat.halo_table['halo_nfw_conc'] = (
        halocat.halo_table['halo_rvir'] / halocat.halo_table['halo_klypin_rs'])

    return halocat


def populate_halocat(halocat, model_name, num_cut=1):

    print('Initialising HOD model {}...'.format(model_name))
    if model_name == 'zheng07':
        model = PrebuiltHodModelFactory('zheng07',
                                        redshift=halocat.redshift,
                                        threshold=-18)
        # model_instance.param_dict['logMmin'] = 12.1
        # model_instance.mock.populate()

    if model_name == 'leauthaud11':
        model = PrebuiltHodModelFactory('leauthaud11',
                                        redshift=halocat.redshift,
                                        threshold=10)

    if model_name == 'tinker13':
        model = PrebuiltHodModelFactory(
                    'tinker13',
                    redshift=halocat.redshift,
                    threshold=10,
                    quiescent_fraction_abscissa=[1e12, 1e13, 1e14, 1e15],
                    quiescent_fraction_ordinates=[0.25, 0.5, 0.75, 0.9])

    if model_name == 'hearin15':
        model = PrebuiltHodModelFactory('zheng07',
                                        redshift=halocat.redshift,
                                        threshold=-1)
    if model_name == 'zu_mandelbaum15':
        model = PrebuiltHodModelFactory('zheng07',
                                        redshift=halocat.redshift,
                                        threshold=-1)
    if model_name == 'zu_mandelbaum16':
        model = PrebuiltHodModelFactory('zheng07',
                                        redshift=halocat.redshift,
                                        threshold=-1)
    if model_name == 'cacciato09':
        model = PrebuiltHodModelFactory('zheng07',
                                        redshift=halocat.redshift,
                                        threshold=-1)

    model.model_name = model_name  # add model_name field
    print('HOD model initialised. Populating {} halos with number cut {}...'
          .format(len(halocat.halo_table), num_cut))
    model.populate_mock(halocat, Num_ptcl_requirement=num_cut)
    print('Mock catalogue populated with {} galaxies'
          .format(len(model.mock.galaxy_table)))
    # save this particular monte carlo realisation of the model
    sim_name = model.mock.header['SimName']
    filedir = os.path.join(save_dir, sim_name, 'z'+str(redshift))
    filepath = os.path.join(filedir,
                            '{}-z{}-{}-galaxy_table.pkl'
                            .format(sim_name, redshift, model_name))
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    print('Pickle dumping galaxy table...')
    with open(filepath, 'wb') as handle:
        pickle.dump(model.mock.galaxy_table, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

    return model


def load_model(halocat, model):

    try:
        sim_name = halocat.SimName
    except AttributeError:
        sim_name = halocat.simname

    redshift = halocat.redshift
    filepath = os.path.join(save_dir, sim_name, 'z'+str(redshift),
                            '{}-z{}-{}-galaxy_table.pkl'
                            .format(sim_name, redshift, model_name))

    with open(filepath, 'rb') as handle:
        gt = pickle.load(handle)
    # this just replaces the galaxy table; everything elae is the same anyway
    model.mock.galaxy_table = gt

    return model


def do_auto_count(model):  # perform counting, save counting

    if use_corrfunc:

        '''
        skip DR and RR countings; they can be calculated analytically
        '''

        sim_name = model.mock.header['SimName']
        L = model.mock.BoxSize
        redshift = model.mock.redshift
        model_name = model.model_name
        print('Pair counting auto-xi with Corrfunc.DDsmu, {} threads...'
              .format(n_threads))
        x = model.mock.galaxy_table['x']
        y = model.mock.galaxy_table['y']
        z = model.mock.galaxy_table['z']
        s_bins = np.arange(0, 151, 1)  # count with bin size 1
        mu_max = 1.0
        n_mu_bins = 100
#        N = len(model.mock.galaxy_table)
#        NR = N * 3
#        xr = np.random.uniform(0, L, NR)
#        yr = np.random.uniform(0, L, NR)
#        zr = np.random.uniform(0, L, NR)
        DD = DDsmu(1, n_threads, s_bins, mu_max, n_mu_bins, x, y, z,
                   periodic=True, verbose=True, boxsize=L, c_api_timer=True)
#        DR = DDsmu(0, n_threads, s_bins, mu_max, n_mu_bins, x, y, z,
#                   X2=xr, Y2=yr, Z2=zr,
#                   periodic=True, verbose=True, boxsize=L, c_api_timer=True)
#        RR = DDsmu(1, n_threads, s_bins, mu_max, n_mu_bins, xr, yr, zr,
#                   periodic=True, verbose=True, boxsize=L, c_api_timer=True)
        # save counting results as original structured array in npy format
        filepath = os.path.join(save_dir, sim_name, 'z'+str(redshift),
                                '{}-auto-paircount-DD.npy'.format(model_name))
        np.save(filepath, DD)
        print('Cross-correlation pair counts saved to: {}'
              .format(filepath))
#        np.save(os.path.join(filedir, '{}-auto-paircount-DR.npy'
#                             .format(model_name)),
#                DR)
#        np.save(os.path.join(filedir, '{}-auto-paircount-RR.npy'
#                             .format(model_name)),
#                RR)

    else:  # use halotools, no need to do counting first, skip to xi
        pass


def do_subcross_count(model, n_sub):

    if use_corrfunc:

        sim_name = model.mock.header['SimName']
        model_name = model.model_name
        L = model.mock.BoxSize
        redshift = model.mock.redshift
        l_sub = L / n_sub  # length of subvolume box
        gt = model.mock.galaxy_table
        for i, j, k in itertools.product(range(n_sub), repeat=3):
            linind = i*n_sub**2 + j*n_sub + k  # linearised index of subvolumes
            mask = ((l_sub * i < gt['x']) & (gt['x'] <= l_sub * (i+1)) &
                    (l_sub * j < gt['y']) & (gt['y'] <= l_sub * (j+1)) &
                    (l_sub * k < gt['z']) & (gt['z'] <= l_sub * (k+1)))
            gt_sub = gt[mask]  # select a subvolume of the galaxy table
            print('Subvolume {} of {} mask created, {} of {} galaxies selected'
                  .format(linind+1, n_sub**3, len(gt_sub), len(gt)))
            print('Pair counting cross-xi with Corrfunc.DDsmu, {} threads...'
                  .format(n_threads))
            x1 = gt['x']
            y1 = gt['y']
            z1 = gt['z']
            x2 = gt_sub['x']
            y2 = gt_sub['y']
            z2 = gt_sub['z']
            s_bins = np.arange(0, 151, 1)  # count with s bin size 1
            mu_max = 1.0
            n_mu_bins = 100  # count with mu bin size 0.01
            DD = DDsmu(0, n_threads, s_bins, mu_max, n_mu_bins, x1, y1, z1,
                       periodic=True, X2=x2, Y2=y2, Z2=z2,
                       verbose=True, boxsize=L, c_api_timer=True)
            # save counting results as original structured array in npy format
            filepath = os.path.join(save_dir, sim_name, 'z'+str(redshift),
                                    '{}-cross_{}-paircount-DD.npy'
                                    .format(model_name, linind))
            np.save(filepath, DD)
            print('Cross-correlation pair counts saved to: {}'
                  .format(filepath))

    else:  # use halotools, no need to do counting first, skip to xi
        pass


def do_auto_correlation(model):

    # setting halocat properties to be compatibble with halotools
    sim_name = model.mock.header['SimName']
    L = model.mock.BoxSize
    Lbox = model.mock.Lbox
    redshift = model.mock.redshift
    model_name = model.model_name
    x = model.mock.galaxy_table['x']
    y = model.mock.galaxy_table['y']
    z = model.mock.galaxy_table['z']

    if use_corrfunc:
        '''
        read in counts, and generate xi using bins specified
        using the PH (natural) estimator for periodic sim boxes instead of LS
        '''
        # read in pair counts file which has fine bins
        filepath = os.path.join(save_dir, sim_name, 'z'+str(redshift),
                                '{}-auto-paircount-DD.npy'.format(model_name))
        DD = np.load(filepath)
        print('Re-binning counts...')
        # set up bins using specifications
        s_bins = np.arange(0, 150 + step_s_bins, step_s_bins)
        s_bins_centre = (s_bins[:-1] + s_bins[1:]) / 2
        mu_bins = np.arange(0, 1 + step_mu_bins, step_mu_bins)
        # re-count pairs in new bins
        npairs = np.zeros((s_bins.size-1, mu_bins.size-1), dtype=np.int32)
        for i, j in itertools.product(
                range(npairs.shape[0]), range(npairs.shape[1])):
            mask = (
                (s_bins[i] < DD['smin']) & (DD['smax'] <= s_bins[i+1]) &
                (mu_bins[j] < DD['mu_max']) & (DD['mu_max'] <= mu_bins[j+1]))
            npairs[i, j] = DD[mask]['npairs'].sum()
        indices = np.where(npairs == 0)
        print('{} bins are empty:'.format(len(indices[0])))
        if len(indices[0]) != 0:  # print problematic bins if any is empty
            for i in range(len(indices)):
                print('({}, {})'.format(indices[0][i], indices[1][i]))

        # calculate RR analytically for auto-correlation
        ND = len(model.mock.galaxy_table)
        _, _, RR = analytic_random_counts(ND, s_bins, mu_bins, Lbox)
        xi_s_mu = ph_estimator(ND, ND, npairs, RR)

    else:  # use halotools, skip the counts, and calculate xi_s_mu directly
        print('Calculating auto-xi with halotools, {} threads...'
              .format(n_threads))
        pos = return_xyz_formatted_array(x, y, z, period=L)
        s_bins = np.arange(1, 150 + step_s_bins, step_s_bins)
        s_bins_centre = (s_bins[:-1] + s_bins[1:]) / 2
        mu_bins = np.linspace(0, 1, 1/step_mu_bins+1)
        xi_s_mu = s_mu_tpcf(pos, s_bins, mu_bins,
                            do_auto=True, do_cross=False,
                            period=L, n_threads=n_threads)

    print('Calculating first two orders of multipole decomposition...')
    xi_0 = tpcf_multipole(xi_s_mu, mu_bins, order=0)
    xi_2 = tpcf_multipole(xi_s_mu, mu_bins, order=2)
    # save to text
    filedir = os.path.join(save_dir, sim_name, 'z'+str(redshift))
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    print('Saving auto-correlation texts to: {}'.format(filedir))
    np.savetxt(os.path.join(filedir, '{}-auto-xi_s_mu.txt'
                            .format(model_name)),
               xi_s_mu, fmt=txtfmt)
    np.savetxt(os.path.join(filedir, '{}-auto-xi_0.txt'
                            .format(model_name)),
               np.vstack([s_bins_centre, xi_0]).transpose(), fmt=txtfmt)
    np.savetxt(os.path.join(filedir, '{}-auto-xi_2.txt'
                            .format(model_name)),
               np.vstack([s_bins_centre, xi_2]).transpose(), fmt=txtfmt)


def do_subcross_correlation(model, n_sub=3):  # n defines number of subvolums
    '''
    cross-correlation between 1/n_sub^3 of a box to the whole box
    n_sub^3 * 16 results are used for emperically estimating covariance matrix
    '''

    if use_corrfunc:
        pass
    else:
        sim_name = model.mock.header['SimName']
        model_name = model.model_name
        L = model.mock.BoxSize
        l_sub = L / n_sub  # length of subvolume box
        gt = model.mock.galaxy_table
        for i, j, k in itertools.product(range(n_sub), repeat=3):
            linind = i*n_sub**2 + j*n_sub + k  # linearised index
            mask = ((l_sub * i < gt['x']) & (gt['x'] < l_sub * (i+1)) &
                    (l_sub * j < gt['y']) & (gt['y'] < l_sub * (j+1)) &
                    (l_sub * k < gt['z']) & (gt['z'] < l_sub * (k+1)))
            gt_sub = gt[mask]
            print('Subvolume {} of {} mask created, {} of {} galaxies selected'
                  .format(linind+1, n_sub**3, len(gt_sub), len(gt)))
            print('Calculating cross-xi with halotools, {} threads...'
                  .format(n_threads))
            pos1 = return_xyz_formatted_array(
                    gt_sub['x'], gt_sub['y'], gt_sub['z'], period=L)
            pos2 = return_xyz_formatted_array(gt['x'], gt['y'], gt['z'],
                                              period=L)
            s_bins = np.arange(1, 150 + step_s_bins, step_s_bins)
            s_bins_centre = (s_bins[:-1] + s_bins[1:]) / 2
            mu_bins = np.linspace(0, 1, 1/step_mu_bins+1)
            xi_s_mu = s_mu_tpcf(pos1, s_bins, mu_bins, sample2=pos2,
                                do_auto=False, do_cross=True,
                                period=L, n_threads=n_threads)
            print('Calculating first two orders of multipole decomposition...')
            xi_0 = tpcf_multipole(xi_s_mu, mu_bins, order=0)
            xi_2 = tpcf_multipole(xi_s_mu, mu_bins, order=2)
            # save to text
            filedir = os.path.join(save_dir, sim_name, 'z'+str(redshift))
            if not os.path.exists(filedir):
                os.makedirs(filedir)
            print('Saving cross-correlation texts to: {}'.format(filedir))
            np.savetxt(os.path.join(filedir,
                                    '{}-cross_{}-xi_s_mu.txt'
                                    .format(model_name, linind)),
                       xi_s_mu, fmt=txtfmt)
            np.savetxt(os.path.join(filedir, '{}-cross_{}-xi_0.txt'
                                    .format(model_name, linind)),
                       np.vstack([s_bins_centre, xi_0]).transpose(),
                       fmt=txtfmt)
            np.savetxt(os.path.join(filedir,
                                    '{}-cross_{}-xi_2.txt'
                                    .format(model_name, linind)),
                       np.vstack([s_bins_centre, xi_2]).transpose(),
                       fmt=txtfmt)


def do_coadd_xi(model_name):

    print('Coadding xi for all phases...')
    xi_list = []
    xi_0_list = []
    xi_2_list = []
    for phase in phases:
        print('Reading phase {}'.format(phase))
        sim_name = '{}_{:02}-{}'.format(sim_name_prefix, cosmology, phase)
        path_prefix = os.path.join(
                        save_dir,
                        sim_name,
                        'z{}'.format(redshift),
                        model_name + '-auto-')
        path_xi = path_prefix + 'xi_s_mu.txt'
        path_xi_0 = path_prefix + 'xi_0.txt'
        path_xi_2 = path_prefix + 'xi_2.txt'
        xi_list.append(np.loadtxt(path_xi))
        xi_0_list.append(np.loadtxt(path_xi_0))
        xi_2_list.append(np.loadtxt(path_xi_2))

    xi_array = np.array(xi_list)
    xi_0_array = np.array(xi_0_list)
    xi_2_array = np.array(xi_2_list)
    # the first column, r, is unaffected under coadding, because same for all
    # s_bins = np.arange(1, 150 + step_s_bins, step_s_bins)
    # s_bins_centre = (s_bins[:-1] + s_bins[1:]) / 2
    # mu_bins = np.linspace(0, 1, np.int(1/step_mu_bins+1))
    xi_s_mu_ca = np.mean(xi_array, axis=0)
    xi_s_mu_err = np.std(xi_array, axis=0)
    # two ways of calculating 0, 2 components are the same, linear operation
    # xi_0 = tpcf_multipole(xi_s_mu_ca, mu_bins, order=0)
    xi_0_ca = np.mean(xi_0_array, axis=0)  # from coadding xi_0
    xi_0_err = np.std(xi_0_array, axis=0)
    # xi_2 = tpcf_multipole(xi_s_mu_ca, mu_bins, order=2)
    xi_2_ca = np.mean(xi_2_array, axis=0)  # from coadding xi_2
    xi_2_err = np.std(xi_2_array, axis=0)

    # save all results
    filedir = os.path.join(
                save_dir,
                sim_name_prefix + '_' + str(cosmology).zfill(2) + '-combined',
                str(redshift))  # across phase
    if not os.path.exists(filedir):
        os.makedirs(filedir)

    print('Saving coadded xi texts...')
    np.savetxt(os.path.join(
            filedir, '{}-auto-xi_s_mu-coadd.txt'.format(model_name)),
        xi_s_mu_ca, fmt=txtfmt)
    np.savetxt(os.path.join(
            filedir, '{}-auto-xi_s_mu-coadd_err.txt'.format(model_name)),
        xi_s_mu_err, fmt=txtfmt)
#    np.savetxt(os.path.join(
#            filedir, '{}-auto-xi_0-from_xi_coadd.txt'.format(model_name)),
#        np.vstack([s_bins_centre, xi_0]).transpose(), fmt=txtfmt)
    np.savetxt(os.path.join(
            filedir, '{}-auto-xi_0-coadd.txt'.format(model_name)),
        xi_0_ca, fmt=txtfmt)
    np.savetxt(os.path.join(
            filedir, '{}-auto-xi_0-coadd_err.txt'.format(model_name)),
        xi_0_err, fmt=txtfmt)
#    np.savetxt(os.path.join(
#            filedir, '{}-auto-xi_2-from_xi_coadd.txt'.format(model_name)),
#        np.vstack([s_bins_centre, xi_2]).transpose(), fmt=txtfmt)
    np.savetxt(os.path.join(
            filedir, '{}-auto-xi_2-coadd.txt'.format(model_name)),
        xi_2_ca, fmt=txtfmt)
    np.savetxt(os.path.join(
            filedir, '{}-auto-xi_2-coadd_err.txt'.format(model_name)),
        xi_2_err, fmt=txtfmt)


def do_cov(model_name):

    for phase in phases:
        for ell in [0, 2]:
            # calculate covariance matrix for each phase box using N_sub^3 xi's
            sim_name = '{}_{:02}-{}'.format(sim_name_prefix, cosmology, phase)
            glob_template = os.path.join(
                                save_dir,
                                sim_name,
                                'z{}'.format(redshift),
                                '{}-cross_*-xi_{}.txt'.format(model_name, ell))
            paths = glob(glob_template)
            print('Paths found:\n', paths)  # debug
            print('Calculating covariance matrix for phase {}, l = {}...'
                  .format(phase, ell))
            xi_list = [np.loadtxt(path)[:, 1] for path in paths]
            cov = xi1d_list_to_cov(xi_list)
            filepath = os.path.join(  # save to respective phase/z folder
                    save_dir, sim_name, 'z{}'.format(redshift),
                    '{}-cross-xi_{}-cov.txt'.format(model_name, ell))
            print('Saving covariance matrix for phase {}, l={}...'
                  .format(phase, ell))
            np.savetxt(filepath, cov, fmt=txtfmt)
            print('Covariance matrix saved to: ', filepath)

    # calculate cov combining all phases, 16 * N_sub^3

    for ell in [0, 2]:
        print('Calculating covariance matrix using all phases, l = {}...'
              .format(ell))
        paths = []
        for phase in phases:
            sim_name = '{}_{:02}-{}'.format(sim_name_prefix, cosmology, phase)
            paths = paths + glob(os.path.join(
                        save_dir, sim_name, 'z{}'.format(redshift),
                        '{}-cross_*-xi_{}.txt'.format(model_name, ell)))
        # read in all xi files for all phases
        xi_list = [np.loadtxt(path)[:, 1] for path in paths]
        cov = xi1d_list_to_cov(xi_list)
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
                        save_dir, sim_name, 'z{}'.format(redshift),
                        '{}-cross_*-xi_0.txt'.format(model_name))))
        paths2 = sorted(glob(os.path.join(
                        save_dir, sim_name, 'z{}'.format(redshift),
                        '{}-cross_*-xi_2.txt'.format(model_name))))
        if len(paths0) != len(paths2):
            print('Cross correlation monopole and quadrupole mismatch')
            return False
        for path0, path2 in zip(paths0, paths2):
            xi_0 = np.loadtxt(path0)[:, 1]
            xi_2 = np.loadtxt(path2)[:, 1]
            xi_list.append(np.hstack(xi_0, xi_2))
    cov_monoquad = xi1d_list_to_cov(xi_list)
    # save cov
    filepath = os.path.join(  # save to '-combined' folder for all phases
            save_dir,
            sim_name_prefix + '_' + str(cosmology).zfill(2) + '-combined',
            'z{}'.format(redshift),
            '{}-cross-xi_monoquad-cov.txt'.format(model_name, ell))
    np.savetxt(filepath, cov_monoquad, fmt=txtfmt)
    print('Monoquad covariance matrix saved to: ', filepath)


def xi1d_list_to_cov(xi_list):

    print('Calculating covariance matrix from {} xi multipole matrices...'
          .format(len(xi_list)))
    return np.cov(np.array(xi_list).transpose())


def xi2d_list_to_cov(xi_list):

    print('Calculating covariance matrix from {} xi matrices...'
          .format(len(xi_list)))
    xi_stack = np.stack(xi_list).transpose([1, 2, 0])
    xi_stack = xi_stack.reshape(int(xi_stack.size/len(xi_list)), len(xi_list))
    # print(xi_stack.shape)
    return np.cov(xi_stack)


def ht_rp_pi_2pcf(model, L, n_rp_bins=12, n_pi_bins=12, n_threads=10):

    x = model.mock.galaxy_table['x']
    y = model.mock.galaxy_table['y']
    z = model.mock.galaxy_table['z']
    pos = return_xyz_formatted_array(x, y, z, period=L)
    rp_bins = np.linspace(50, 150, n_rp_bins)  # perpendicular bins
    pi_bins = np.linspace(50, 150, n_pi_bins)  # parallel bins
    xi = rp_pi_tpcf(pos, rp_bins, pi_bins, period=L, n_threads=n_threads)
    return xi


def baofit_data_ht_jackknife(cat, n_rp_bins, n_pi_bins, n_threads):

    try:
        L = cat.BoxSize
    except AttributeError:
        L = cat.Lbox[0]
    # redshift = cat.redshift
    x = cat.halo_table['halo_x']
    y = cat.halo_table['halo_y']
    z = cat.halo_table['halo_z']
    pos = return_xyz_formatted_array(x, y, z, period=L)
    rp_bins = np.logspace(np.log10(80), np.log10(130), n_rp_bins)
    pi_bins = np.logspace(np.log10(80), np.log10(130), n_pi_bins)
#    rp_bin_centres = (rp_bins[1:] + rp_bins[:-1])/2
#    pi_bin_centres = (pi_bins[1:] + pi_bins[:-1])/2
    # define randoms
    n_ran = len(cat.halo_table) * 50  # 500 ideal?
    print('Preparing randoms...')
    xran = np.random.uniform(0, L, n_ran)
    yran = np.random.uniform(0, L, n_ran)
    zran = np.random.uniform(0, L, n_ran)
    randoms = return_xyz_formatted_array(xran, yran, zran, period=L)
    print('2pt jackknife with ' + str(n_threads) + ' threads...')
    xi, xi_cov = rp_pi_tpcf_jackknife(pos, randoms, rp_bins, pi_bins, Nsub=3,
                                      period=L, n_threads=n_threads)
    try:
        np.linalg.cholesky(xi_cov)
        print('Covariance matrix passes cholesky decomposition')
    except np.linalg.LinAlgError:
        print('Covariance matrix fails cholesky decomposition')

    return xi, xi_cov


if __name__ == "__main__":

    for phase in phases:
        for model_name in prebuilt_models:
            # data processing
            cat = make_halocat(phase)
            model0 = populate_halocat(cat, model_name=model_name, num_cut=1)
            model = load_model(cat, model0)
            # print(model0==model)
            do_auto_count(model)  # do the pair counting in 1 Mpc bins
            do_subcross_count(model, n_sub=3)
            do_auto_correlation(model)  # sum into specified bins
            do_subcross_correlation(model)

    for model_name in prebuilt_models:
        do_coadd_xi(model_name)
        do_cov(model_name)  # baofit continues in ipynb
