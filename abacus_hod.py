# -*- coding: utf-8 -*-
"""
Created on Wed May  9 20:34:08 2018

@author: Duan Yutong (dyt@physics.bu.edu)

halo/galaxy/particle tables have the following default units:

    position:   Mpc / h
    velocity:   km / s
    mass:       Msun / h
    radius:     kpc/h

"""

from __future__ import (
        absolute_import, division, print_function, unicode_literals)
# import os
import numpy as np
from scipy.special import erfc
from halotools.empirical_models import PrebuiltHodModelFactory
from astropy import table
# from multiprocessing import Pool
from tqdm import tqdm
# from functools import partial


def N_cen_mean(M, param_dict):

    # M. white 2011 parametrisation of zheng07. mass in Msun
    Mcut = np.power(10, param_dict['logMcut'])
    sigma = param_dict['sigma_lnM']
    return 1/2*erfc(np.log(Mcut/M)/(np.sqrt(2)*sigma))


def N_sat_mean(M, param_dict):

    # M. white 2011 parametrisation of zheng07. mass in Msun
    Mcut = np.power(10, param_dict['logMcut'])
    M1 = np.power(10, param_dict['logM1'])
    kappa = param_dict['kappa']
    alpha = param_dict['alpha']
    ncm = N_cen_mean(M, param_dict)
    mask = (M - kappa * Mcut) / M1 < 0  # if this < 0, should get 0 probability
    nsm = ncm * np.power((M - kappa * Mcut) / M1, alpha)
    # replace NaN with 0, also get rid of even power of negative number
    nsm[mask] = 0

    return nsm


def mean_occupation(model, halo_mass_bins):

    '''
    return N_sat_mean, N_cen_mean, N_tot_mean
    all mass in unit of Msun/h
    '''
    N_halos, _ = np.histogram(model.mock.halo_table['halo_mvir'],
                              bins=halo_mass_bins)
    mask_cen = model.mock.galaxy_table['gal_type'] == 'centrals'
    mask_sat = model.mock.galaxy_table['gal_type'] == 'satellites'
    N_cen, _ = np.histogram(model.mock.galaxy_table['halo_mvir'][mask_cen],
                            bins=halo_mass_bins)
    N_sat, _ = np.histogram(model.mock.galaxy_table['halo_mvir'][mask_sat],
                            bins=halo_mass_bins)
    N_cen_mean = N_cen / N_halos
    N_sat_mean = N_sat / N_halos

    return N_cen_mean, N_sat_mean, N_cen_mean + N_sat_mean


def do_vpec(vz, vrms, alpha_c):

    # draw from normal distribution and add peculiar velocity in LOS direction
    # for central galaxies
    # has width = vrms/sqrt(3) * alpha_c
    vpec = np.random.normal(loc=0, scale=vrms/np.sqrt(3)*alpha_c)
    return vz + vpec


def do_rsd(z, vz, redshift, cosmology, period):
    '''
    Perform Redshift Space Distortion (RSD)
    The z direction position is modified by the z direction velocity

    z' = z + v_z/(aH)

    where units are
    z: Mpc
    vz: km/s
    H: km/s/Mpc
    period: Mpc
    z' also wraps around to the other side of the simulation box
    '''
    H0 = cosmology.H0.value  # cosmology.H0 returns an astropy quantity
    E = cosmology.efunc(redshift)  # returns a float
    z = z + (1 + redshift) / (H0*E) * vz
    return z % period


def process_particle_props(ptab, h, halo_m_prop='halo_mvir',
                           max_iter=30, precision=0.1):

    print('Processing particle table properties...')
    # add host_centric_distance for all particles for bias s
    ptab['x_rel'] = ptab['x'] - ptab['halo_x']  # Mpc/h
    ptab['y_rel'] = ptab['y'] - ptab['halo_y']  # Mpc/h
    ptab['z_rel'] = ptab['z'] - ptab['halo_z']  # Mpc/h
    r_rel = np.array([ptab['x_rel'], ptab['y_rel'], ptab['z_rel']]).T*1e3
    ptab['r_centric'] = np.linalg.norm(r_rel, axis=1)  # kpc/h
    # add peculiar velocity and speed columns for bias s_v
    ptab['vx_pec'] = ptab['vx'] - ptab['halo_vx']
    ptab['vy_pec'] = ptab['vy'] - ptab['halo_vy']
    ptab['vz_pec'] = ptab['vz'] - ptab['halo_vz']
    v_pec = np.array([ptab['vx_pec'], ptab['vy_pec'], ptab['vz_pec']]).T
    # for some particles close to halo centre due to single precision, we find
    # v_pec = v_rad = v_tan = 0, and r_0 = r_min
    ptab['v_pec'] = np.linalg.norm(v_pec, axis=1)  # peculiar speed km/s
    # dimensionless r relative hat
    norm = np.linalg.norm(r_rel, axis=1)
    norm = norm.reshape(len(norm), 1)  # reshape it into a column vector
    r_rel_hat = r_rel / norm
    ptab['v_rad'] = np.sum(np.multiply(v_pec, r_rel_hat), axis=1)  # dot prod
    ptab['v_tan'] = np.sqrt(np.square(ptab['v_pec'])
                            - np.square(ptab['v_rad']))
    # calculate perihelion distance for bias s_p, note h factors
    G_N = 6.674e-11  # G newton in m3, s-2, kg-1
    M_sun = 1.98855e30  # solar mass in kg
    kpc = 3.0857e19  # kpc in meters
    M = ptab[halo_m_prop] / h  # halo mass in Msun
    c = ptab['halo_nfw_conc']  # nfw concentration, dimensionless
    r0 = ptab['r_centric'] / h  # distance to halo centre in kpc
    Rs = ptab['halo_klypin_rs'] / h  # halo scale radius in kpc
    vr2, vt2 = np.square(ptab['v_rad']), np.square(ptab['v_tan'])
    alpha = G_N*M*M_sun/1e6/kpc/(np.log(1+c) - c/(1+c))  # kpc (km/s)^2
    # initial X = 1, set first iteration value by hand
    X2 = np.sqrt(vt2/(vr2 + vt2))  # has NaN when v_pec = 0
    X2[np.isnan(X2)] = 1  # set nan values to 1 to restore r_0 = r_min

    i = 0
    print('Calculating X2 for perihelion ranking...')
    for i in tqdm(range(max_iter)):
        i = i + 1
        Xold = np.sqrt(X2)
        Xold[Xold == 0] = 1  # for cases where X=0, this ensures Xnew = 0
        X2 = vt2 / (vt2 + vr2 +
                    2*alpha/r0*(np.log(1+Xold*r0/Rs)/Xold - np.log(1+r0/Rs)))
        # set if v_pec = v_rad = v_tan = 0, then X2 = nan, but it should be 1
        X2[np.isnan(X2)] = 1  # set to 1 manually for stationary particles
        Xnew = np.sqrt(X2)

        # error analysis
        ferr = np.abs(Xold - Xnew) / Xnew  # for Xnew = 0, this results in inf
        ferr[np.isinf(ferr)] = 0
        str_template = ('Iteration {:2d} fractional error: {:6.5f} '
                        '+/- {:6.5f}, max {:6.5f}.'
                        .format(i, np.mean(ferr), np.std(ferr), np.max(ferr)))
        if np.max(ferr) < precision:
            print(str_template + ' Precision reached.')
            break
        elif i == max_iter:
            print(str_template + ' Maximum iterations reached.')
            break
        else:
            # print(str_template)
            pass

    ptab['r_perihelion'] = np.float32(r0*Xnew * h)  # in kpc/h

    # return ptab


def initialise_model(redshift, model_name, halo_m_prop='halo_mvir'):

    '''
    create an instance of halotools model class

    logMcut : The cut-off mass for the halo to host in a central galaxy. Given
    in solar mass.

    sigma_lnM : Parameter that modulates the shape of the number of
    central galaxies.

    logM1 : The scale mass for the number of satellite galaxies.

    kappa : Parameter that affects the cut-off mass for satellite galaxies.
    A detailed discussion and best-fit values of these parameters can be found
    in Zheng+2009.

    alpha : The power law index of the number of satellite galaxies.

    s : The satellite profile modulation parameter. Modulates how the radial
    distribution of satellite galaxies within halos deviate from the radial
    profile of the halo. Positive value favors satellite galaxies to populate
    the outskirts of the halo whereas negative value favors satellite galaxy
    to live near the center of the halo.

    s_v : float. The satellite velocity bias parameter. Modulates how the
    satellite galaxy peculiar velocity deviates from that of the local dark
    matter particle. Positive value favors high peculiar velocity satellite
    galaxies and vice versa. Note that our implementation preserves the
    Newton's second law of the satellite galaxies.

    alpha_c : float. The central velocity bias parameter. Modulates the
    peculiar velocity of the central galaxy. The larger the absolute value of
    the parameter, the larger the peculiar velocity of the central. The sign
    of the value does not matter.

    s_p : float. The perihelion distance modulation parameter. A positive
    value favors satellite galaxies to have larger distances to the halo
    center upon their closest approach to the center and vice versa. This can
    be regarded as a "fancier" satellite profile modulation parameter.

    A : float. The assembly bias parameter. Introduces the effect of assembly
    bias. A positive value favors higher concentration halos to host galaxies
    whereas a negative value favors lower concentration halos to host galaxies.
    If you are invoking assembly bias decoration, i.e. a non-zero A parameter,
    you need to run gen_medianc.py first. A detailed discussion of these
    parameters can be found in Yuan et al. in prep. To turn off any of the five
    decorations, just set the corresponding parameter to 0.

    '''

    print('Initialising HOD model: {}...'.format(model_name))

    # halotools prebiult models
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

    # generalised HOD models with 5-parameter zheng07 as base model
    else:  # 'gen_base1'
        model = PrebuiltHodModelFactory('zheng07',
                                        redshift=redshift,
                                        threshold=-18,
                                        prim_haloprop_key=halo_m_prop)
        # five baseline parameters
        model.param_dict['logMcut'] = 13.35
        model.param_dict['sigma_lnM'] = 0.85
        model.param_dict['kappa'] = 1
        model.param_dict['logM1'] = 13.8
        model.param_dict['alpha'] = 1
        # decoration parameters
        model.param_dict['s'] = 0
        model.param_dict['s_v'] = 0
        model.param_dict['s_p'] = 0
        model.param_dict['alpha_c'] = 0
        model.param_dict['A_cen'] = 0
        model.param_dict['A_sat'] = 0

        if model_name == 'gen_base2':
            model.param_dict['logM1'] = 13.85
            model.param_dict['alpha'] = 1.377
        elif model_name == 'gen_base3':
            model.param_dict['logM1'] = 13.9
            model.param_dict['alpha'] = 1.663
        elif model_name == 'gen_ass1':
            model.param_dict['A_cen'] = 1
            model.param_dict['A_sat'] = 0
        elif model_name == 'gen_ass2':
            model.param_dict['A_cen'] = 0
            model.param_dict['A_sat'] = 1
        elif model_name == 'gen_ass3':
            model.param_dict['A_cen'] = 1
            model.param_dict['A_sat'] = 1

    #  add useful model properties here
    model.model_name = model_name
    if 'gen' in model_name:
        model.model_type = 'general'
    else:
        model.model_type = 'prebuilt'
    model.halo_m_prop = halo_m_prop

    return model


def populate_model(halocat, model,
                   add_rsd=True, use_subhalos=False, N_threads=10):

    # use halotools HOD
    if model.model_type == 'prebuilt':

        print('Populating {} halos with N_cut = {} for r = {}'
              'using prebuilt model {} ...'
              .format(len(halocat.halo_table), model.N_cut,
                      model.r, model.model_name))

        if hasattr(model, 'mock'):  # already populated once, just re-populate
            model.mock.populate()
        else:  # model was never populated and mock attribute does not exist
            model.populate_mock(halocat, Num_ptcl_requirement=model.N_cut)

    # use particle-based HOD
    elif model.model_type == 'general':

        if hasattr(model, 'mock'):
            # attribute mock present, at least second realisation
            # halo_table already had N_cut correction
            pass
        else:
            # create dummy mock attribute, using a huge N_cut to save time
            print('Instantiating model.mock using base class zheng07...')
            model.populate_mock(halocat, Num_ptcl_requirement=10000)
            # replace the halo_table, now using the correct N_cut
            print('Applying N_cut = {} to table...'.format(model.N_cut))
            N_halos_0 = len(halocat.halo_table)
            # 'halo_num_child_particles', 'halo_num_p', 'halo_N', 'halo_alt_N'
            # are all different fields, only halo_N corresponds to halo_mvir
            mask = halocat.halo_table['halo_N'] >= model.N_cut
            N_halos_mcut = np.sum(mask)
            if use_subhalos:
                N_halos_scut = N_halos_0
                pass  # only apply mass cut as mask
            else:  # drop all subhalos
                print('Removing subhalos...')
                mask_sub = halocat.halo_table['halo_upid'] == -1
                N_halos_scut = np.sum(mask_sub)
                mask = mask & mask_sub
            model.mock.halo_table = halocat.halo_table[mask]
            N_halos = len(model.mock.halo_table)

            print('Total N halos {}; mass cut mask {}; '
                  'subhalos cut mask {}; final N halos {}.'
                  .format(N_halos_0, N_halos_mcut, N_halos_scut, N_halos))

            # particle table reduction
            # either remove unwanted rows or copy wanted rows
            htab = model.mock.halo_table
            htab_del = halocat.halo_table[~mask]  # deleted halo rows in htab
            ind = []
            if len(htab) > len(htab_del):  # go through rows to delete
                print('Collecting particle table indices for removal...')
                ptab = halocat.halo_ptcl_table
                for i in range(len(htab_del)):
                    m = htab['halo_subsamp_start'][i]
                    n = htab['halo_subsamp_len'][i]
                    ind.append(np.uint64(np.arange(m, m+n)))
                ind = np.concatenate(ind)
                print('Initial particle table length {}, removing {} rows...'
                      .format(len(ptab), len(ind)))
                ptab.remove_rows(ind)
            else:  # only copy particles in selected halos
                print('Collecting particle table indices for copying...')
                for i in tqdm(range(len(htab))):
                    m = htab['halo_subsamp_start'][i]
                    n = htab['halo_subsamp_len'][i]
                    ind.append(np.uint64(np.arange(m, m+n)))
                ind = np.concatenate(ind)
                ptab = halocat.halo_ptcl_table[ind]

            # reindex particle meta in halo table
            htab['halo_subsamp_start'][0] = 0
            htab['halo_subsamp_start'][1:] = \
                htab['halo_subsamp_len'].cumsum()[:-1]
            # verify total particles in halos agree with particle table
            assert htab['halo_subsamp_len'].sum() == len(ptab)
            assert (
                (htab['halo_subsamp_start'][-1] + htab['halo_subsamp_len'][-1])
                == len(ptab))
            print('Reduced particle table passes sanity checks.')
            # assign particle table to mock attribute
            model.mock.halo_ptcl_table = ptab

        # generate galaxy catalogue and overwrite model.mock.galaxy_table
        model = make_galaxies(model, add_rsd=add_rsd)

    print('Mock catalogue populated with {} galaxies'
          .format(len(model.mock.galaxy_table)))

    return model


def make_galaxies(model, add_rsd=True, N_threads=10):

    h = (model.mock.cosmology.H0/100).value
    htab = model.mock.halo_table  # make a convenient copy of halo table
    N_halos = len(htab)  # total number of halos available
    # remove existing galaxy table, because modifying it is too painfully slow
    if hasattr(model.mock, 'galaxy_table'):
        del model.mock.galaxy_table

    '''
    centrals

    '''
    # if we add assembly bias, re-rank halos using pseudomass
    halo_m = htab[model.halo_m_prop].data
    c_median = model.c_median_poly(np.log10(halo_m))
    delta_c = htab['halo_nfw_conc'] - c_median
    A_cen = model.param_dict['A_cen']
    if A_cen != 0:
        halo_pseudomass_cen = np.int64(
                halo_m * np.exp(A_cen*(2*(delta_c > 0) - 1)))
        ind_m = halo_m.argsort()
        ind_pm = halo_pseudomass_cen.argsort().argsort()
        halo_m = halo_m[ind_m][ind_pm]
    # calculate N_cen_mean using given model for all halos
    htab['N_cen_model'] = N_cen_mean(halo_m / h, model.param_dict)
    # create N_halos random numbers in half-open interval [0. 1) for centrals
    htab['N_cen_rand'] = np.random.random(N_halos)
    # if random number is less than model probably, halo hosts a central
    mask_cen = htab['N_cen_rand'] < htab['N_cen_model']
    # add halo velocity bias for observer along LOS
    # if alpha_c = 0, no bias is added.
    # this operation should be fast enough, no skip needed
    vz = do_vpec(htab['halo_vz'][mask_cen], htab['halo_vrms'][mask_cen],
                 model.param_dict['alpha_c'])
    # at last, add observational rsd for halos
    if add_rsd:
        z = do_rsd(htab['halo_z'][mask_cen]/h, vz, model.mock.redshift,
                   model.mock.cosmology, model.mock.BoxSize/h) * h  # in Mpc/h
    else:
        z = htab['halo_z'][mask_cen]
    # central calculation done, create centrals table
    # create galaxy_table columns to be inherited from halo_table
    # excludes halo_z and halo_vz due to velocity bias and rsd
    col_cen_inh = ['halo_upid', 'halo_id', 'halo_hostid',
                   'halo_x', 'halo_y',
                   'halo_vx', 'halo_vy',
                   'halo_klypin_rs', 'halo_nfw_conc', model.halo_m_prop,
                   'N_cen_model', 'N_cen_rand']
    gtab_inh = table.Table([htab[col][mask_cen] for col in col_cen_inh],
                           names=col_cen_inh)
    # create new columns containing galaxy properties, with bias & rsd
    # r_centric is the host centric distance between particle and halo centre
    col_cen_new = ['x', 'y', 'z', 'vx', 'vy', 'vz']
    gtab_new = table.Table(
        [htab['halo_x'][mask_cen],  htab['halo_y'][mask_cen],  np.float32(z),
         htab['halo_vx'][mask_cen], htab['halo_vy'][mask_cen], np.float32(vz)],
        names=col_cen_new)
    gtab_new['r_centric'] = np.float32(0)
    # combine inherited fields and new field(s)
    gtab_cen = table.hstack([gtab_inh, gtab_new], join_type='exact')
    gtab_cen['gal_type'] = 'centrals'
    print('{} centrals generated.'.format(len(gtab_cen)))

    '''
    satellites

    '''
    # before decorations, N_sat only depend on halo mass
    halo_m = htab[model.halo_m_prop].data
    A_sat = model.param_dict['A_sat']
    if A_sat != 0:
        halo_pseudomass_sat = np.int64(
                halo_m * np.exp(A_sat*(2*(delta_c > 0) - 1)))
        ind_m = halo_m.argsort()
        ind_pm = halo_pseudomass_sat.argsort().argsort()
        halo_m = halo_m[ind_m][ind_pm]
    htab['N_sat_model'] = N_sat_mean(halo_m / h, model.param_dict) \
        / htab['halo_subsamp_len']

    print('Creating inherited halo properties in particle table...')
    ptab = model.mock.halo_ptcl_table  # 10% subsample of halo DM particles
    N_particles = len(ptab)
    # inherite columns from halo talbe and add to particle table
    # particles belonging to the same halo share the same values
    col_ptc_inh = ['halo_upid', 'halo_id', 'halo_hostid',
                   'halo_x', 'halo_y', 'halo_z',
                   'halo_vx', 'halo_vy', 'halo_vz',
                   'halo_klypin_rs', 'halo_nfw_conc', model.halo_m_prop,
                   'halo_subsamp_start', 'halo_subsamp_len',
                   'N_cen_model', 'N_cen_rand', 'N_sat_model']
    for col in col_ptc_inh:
        ptab[col] = np.repeat(htab[col], htab['halo_subsamp_len'])

    # calculate additional particle quantities for decorations
    process_particle_props(ptab, h, halo_m_prop=model.halo_m_prop)

    # particle table complete. onto satellite generation
    # calculate satellite probablity and generate random numbers
    ptab['N_sat_rand'] = np.random.random(N_particles)
    # add particle ranking using s, s_v, s_p for each halo
    for key in ['rank_s', 'rank_s_v', 'rank_s_p']:
        # initialise column, astropy doesn't support empty columns
        ptab[key] = np.int32(-1)
    print('Calculating rankings within each halo...')
    for i in tqdm(range(N_halos)):
        m = htab['halo_subsamp_start'][i]
        n = htab['halo_subsamp_len'][i]
        if model.param_dict['s'] != 0:
            # furtherst particle has highest rank, closest has 0
            ptab['rank_s'][m:m+n] = ptab['r_centric'][m:m+n] \
                                    .argsort()[::-1].argsort()
        if model.param_dict['s_v'] != 0:
            ptab['rank_s_v'][m:m+n] = ptab['v_pec'][m:m+n] \
                                      .argsort()[::-1].argsort()
        if model.param_dict['s_p'] != 0:
            ptab['rank_s_p'][m:m+n] = ptab['r_perihelion'][m:m+n] \
                                      .argsort()[::-1].argsort()
    # calculate new probability regardless of decoration parameters
    # if any s is zero, the formulae guarantee the random numbers are unchanged
    ptab['N_sat_rand'] = (
        ptab['N_sat_rand'] *
        (1 + model.param_dict['s']
         * (1 - 2*ptab['rank_s']/(ptab['halo_subsamp_len']-1))))
    ptab['N_sat_rand'] = (
        ptab['N_sat_rand'] *
        (1 + model.param_dict['s_v']
         * (1 - 2*ptab['rank_s_v']/(ptab['halo_subsamp_len']-1))))
    ptab['N_sat_rand'] = (
        ptab['N_sat_rand'] *
        (1 + model.param_dict['s_p']
         * (1 - 2*ptab['rank_s_p']/(ptab['halo_subsamp_len']-1))))
    # select particles which host satellites
    mask_sat = ptab['N_sat_rand'] < ptab['N_sat_model']
    # correct for RSD
    if add_rsd:
        z = do_rsd(ptab['z'][mask_sat]/h, ptab['vz'][mask_sat],
                   model.mock.redshift, model.mock.cosmology,
                   model.mock.BoxSize/h) * h  # in Mpc/h
    else:
        z = ptab['z'][mask_sat]
    # satellite calculation done, create satellites table
    # create galaxy_table columns to be inherited from particle table
    # all position and velocity columns except z are inherited from particle
    col_sat_inh = list(set(col_ptc_inh + ['x', 'y', 'vx', 'vy', 'vz',
                                          'v_rad', 'v_tan', 'r_centric',
                                          'N_sat_model',
                                          'rank_s', 'rank_s_v', 'rank_s_p']))
    gtab_inh = table.Table([ptab[col][mask_sat] for col in col_sat_inh],
                           names=col_sat_inh)
    col_sat_new = ['z']
    # save z as float32 as all positions in catalogues are float32
    # this also ensures Corrfunc compatibility
    gtab_new = table.Table([np.float32(z)], names=col_sat_new)
    # combine inherited fields and new field(s)
    gtab_sat = table.hstack([gtab_inh, gtab_new], join_type='exact')
    gtab_sat['gal_type'] = 'satellites'
    print('{} satellites generated.'.format(len(gtab_sat)))
    # combine centrals table and satellites table
    model.mock.galaxy_table = table.vstack([gtab_cen, gtab_sat],
                                           join_type='outer')
    model.mock.halo_ptcl_table = ptab

    return model