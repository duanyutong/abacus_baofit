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
import numpy as np
# from multiprocessing import Pool
from scipy.special import erfc
from halotools.empirical_models import PrebuiltHodModelFactory
from astropy import table


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
    starts = np.asarray(starts).astype(np.int64)
    lengths = np.asarray(lengths).astype(np.int64)
    stops = starts + lengths
    ret = (np.repeat(stops - lengths.cumsum(), lengths)
           + np.arange(lengths.sum()))
    return ret.astype(np.uint64)


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


def mean_occupation(halo_table, galaxy_table, halo_mass_bins):

    '''
    return N_sat_mean, N_cen_mean, N_tot_mean
    all mass in unit of Msun/h
    '''
    N_halos, _ = np.histogram(halo_table['halo_mvir'], bins=halo_mass_bins)
    mask_cen = galaxy_table['gal_type'] == 'centrals'
    mask_sat = galaxy_table['gal_type'] == 'satellites'
    N_cen, _ = np.histogram(galaxy_table['halo_mvir'][mask_cen],
                            bins=halo_mass_bins)
    N_sat, _ = np.histogram(galaxy_table['halo_mvir'][mask_sat],
                            bins=halo_mass_bins)
    N_cen_mean = N_cen / N_halos
    N_sat_mean = N_sat / N_halos

    return N_cen_mean, N_sat_mean, N_cen_mean + N_sat_mean


def do_vpec(vz, vrms, alpha_c):

    # draw from normal distribution and add peculiar velocity in LOS direction
    # for central galaxies
    # has width = vrms/sqrt(3) * alpha_c
    vpec = np.random.normal(loc=0, scale=vrms/np.sqrt(3)*np.abs(alpha_c))
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
    z = z + vz * (1 + redshift) / (H0*E)
    return z % period


def rank_halo_particles(arr):

    return arr.argsort()[::-1].argsort()


# def rank_particles_by_halo(arr_split):
#     '''
#     input split array consists of properties of each particle within a halo
#     grouped/split by halo, e.g.
#     arr_split = [array([], dtype=uint64),
#                  array([0], dtype=uint64),
#                  array([1, 2], dtype=uint64),
#                  array([3, 4, 5], dtype=uint64),
#                  array([6, 7, 8, 9], dtype=uint64)]
#     Properties can be r_centric, v_pec, or r_perihelion to be sorted
#     Returns ranking of particles, concatinated
#     '''
#     with closing(Pool(3)) as p:
#         rank = p.map(rank_halo_particles, list(arr_split))
#     return np.concatenate(rank)


def process_particle_props(pt, h, halo_m_prop='halo_mvir',
                           max_iter=10, precision=0.1):

    # add host_centric_distance for all particles for bias s
    pt['x_rel'] = pt['x'] - pt['halo_x']  # Mpc/h
    pt['y_rel'] = pt['y'] - pt['halo_y']  # Mpc/h
    pt['z_rel'] = pt['z'] - pt['halo_z']  # Mpc/h
    # r in kpc
    r_rel = np.array([pt['x_rel'], pt['y_rel'], pt['z_rel']]).T*1e3
    pt['r_centric'] = np.linalg.norm(r_rel, axis=1)  # kpc/h
    # add peculiar velocity and speed columns for bias s_v
    pt['vx_pec'] = pt['vx'] - pt['halo_vx']
    pt['vy_pec'] = pt['vy'] - pt['halo_vy']
    pt['vz_pec'] = pt['vz'] - pt['halo_vz']
    v_pec = np.array([pt['vx_pec'], pt['vy_pec'], pt['vz_pec']]).T
    # for some particles close to halo centre due to single precision, we find
    # v_pec = v_rad = v_tan = 0, and r_0 = r_min
    pt['v_pec'] = np.linalg.norm(v_pec, axis=1)  # peculiar speed km/s
    # dimensionless r relative hat
    norm = np.linalg.norm(r_rel, axis=1)
    norm = norm.reshape(len(norm), 1)  # reshape it into a column vector
    r_rel_hat = r_rel / norm
    pt['v_rad'] = np.sum(np.multiply(v_pec, r_rel_hat), axis=1)  # dot prod
    pt['v_tan'] = np.sqrt(np.square(pt['v_pec']) - np.square(pt['v_rad']))
    # calculate perihelion distance for bias s_p, note h factors
    G_N = 6.674e-11  # G newton in m3, s-2, kg-1
    M_sun = 1.98855e30  # solar mass in kg
    kpc = 3.0857e19  # kpc in meters
    M = pt[halo_m_prop] / h  # halo mass in Msun
    c = pt['halo_nfw_conc']  # nfw concentration, dimensionless
    r0 = pt['r_centric'] / h  # distance to halo centre in kpc
    Rs = pt['halo_klypin_rs'] / h  # halo scale radius in kpc
    vr2, vt2 = np.square(pt['v_rad']), np.square(pt['v_tan'])
    alpha = G_N*M*M_sun/1e6/kpc/(np.log(1+c) - c/(1+c))  # kpc (km/s)^2
    # initial X = 1, set first iteration value by hand
    X2 = np.sqrt(vt2/(vr2 + vt2))  # has NaN when v_pec = 0
    X2[np.isnan(X2)] = 1  # set nan values to 1 to restore r_0 = r_min

    i = 0
    print('Calculating X2 for perihelion ranking...')
    for i in range(max_iter):
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
            pass  # print(str_template)

    pt['r_perihelion'] = np.float32(r0*Xnew * h)  # in kpc/h


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

    alpha_c : float. The central velocity bias parameter. Modulates the
    peculiar velocity of the central galaxy. The larger the absolute value of
    the parameter, the larger the peculiar velocity of the central. The sign
    of the value does not matter.

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

    # print('Initialising HOD model: {}...'.format(model_name))

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
        model.param_dict['s'] = 0  # sat ranking by halo centric distance
        model.param_dict['s_v'] = 0  # sat ranking by relative speed
        model.param_dict['s_p'] = 0  # sat ranking perihelion distance
        model.param_dict['alpha_c'] = 0  # centrals velocity bias
        model.param_dict['A_cen'] = 0  # centrals assembly bias, pseudomass
        model.param_dict['A_sat'] = 0  # satellites assembly bias, pseudomass
        ''' BOOKKEEPING: DONT MODIFY EXISTING MODELS, CREATE NEW ONES '''
        if model_name == 'gen_base2':
            model.param_dict['logM1'] = 13.85
            model.param_dict['alpha'] = 1.377
        elif model_name == 'gen_base3':
            model.param_dict['logM1'] = 13.9
            model.param_dict['alpha'] = 1.663
        elif model_name == 'gen_base4':
            model.param_dict['logM1'] = 13.796
            model.param_dict['alpha'] = 0.95
        elif model_name == 'gen_base5':
            model.param_dict['logM1'] = 13.805
            model.param_dict['alpha'] = 1.05
        elif model_name == 'gen_ass1':
            model.param_dict['A_cen'] = 1
            model.param_dict['A_sat'] = 0
        elif model_name == 'gen_ass2':
            model.param_dict['A_cen'] = 0
            model.param_dict['A_sat'] = 1
        elif model_name == 'gen_ass3':
            model.param_dict['A_cen'] = 1
            model.param_dict['A_sat'] = 1
        elif model_name == 'gen_ass1_n':
            model.param_dict['A_cen'] = -1
            model.param_dict['A_sat'] = 0
        elif model_name == 'gen_ass2_n':
            model.param_dict['A_cen'] = 0
            model.param_dict['A_sat'] = -1
        elif model_name == 'gen_ass3_n':
            model.param_dict['A_cen'] = -1
            model.param_dict['A_sat'] = -1
        elif model_name == 'gen_vel1':
            model.param_dict['alpha_c'] = 1
        elif model_name == 'gen_s1':
            model.param_dict['s'] = 1
        elif model_name == 'gen_sv1':
            model.param_dict['s_v'] = 1
        elif model_name == 'gen_sp1':
            model.param_dict['s_p'] = 1
        elif model_name == 'gen_s1_n':
            model.param_dict['s'] = -1
        elif model_name == 'gen_sv1_n':
            model.param_dict['s_v'] = -1
        elif model_name == 'gen_sp1_n':
            model.param_dict['s_p'] = -1
        elif model_name == 'gen_allbiases':
            model.param_dict['A_cen'] = 1
            model.param_dict['A_sat'] = 1
            model.param_dict['alpha_c'] = 1
            model.param_dict['s'] = 1
            model.param_dict['s_v'] = 1
            model.param_dict['s_p'] = 1
        elif model_name == 'gen_allbiases_n':
            model.param_dict['A_cen'] = -1
            model.param_dict['A_sat'] = -1
            model.param_dict['alpha_c'] = 1
            model.param_dict['s'] = -1
            model.param_dict['s_v'] = -1
            model.param_dict['s_p'] = -1

    # add useful model properties here
    model.model_name = model_name
    if 'gen' in model_name:
        model.model_type = 'general'
    else:
        model.model_type = 'prebuilt'
    model.halo_m_prop = halo_m_prop

    return model


def populate_model(halocat, model, add_rsd=True, N_threads=10):

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
            pass
        else:
            # create dummy mock attribute, using a huge N_cut to save time
            # print('Instantiating model.mock using base class zheng07...')
            model.populate_mock(halocat, Num_ptcl_requirement=10000)
            # halo_table already had N_cut correction, just copy tables
            model.mock.halo_table = halocat.halo_table
            model.mock.halo_ptcl_table = halocat.halo_ptcl_table
        # generate galaxy catalogue and overwrite model.mock.galaxy_table
        print('Populating {} halos, r = {}...'
              .format(len(model.mock.halo_table), model.r))
        model = make_galaxies(model, add_rsd=add_rsd, N_threads=N_threads)

    print('Mock catalogue populated with {} galaxies'
          .format(len(model.mock.galaxy_table)))

    return model


def make_galaxies(model, add_rsd=True, N_threads=10):

    h = model.mock.cosmology.H0.value/100  # 0.6726000000000001
    ht = model.mock.halo_table
    N_halos = len(ht)  # total number of host halos available
    # remove existing galaxy table, because modifying it is too painfully slow
    if hasattr(model.mock, 'galaxy_table'):
        del model.mock.galaxy_table

    '''
    centrals

    '''
    # if we add assembly bias, re-rank halos using pseudomass
    halo_m = ht[model.halo_m_prop].data
    c_median = model.c_median_poly(np.log10(halo_m))
    delta_c = ht['halo_nfw_conc'] - c_median
    A_cen = model.param_dict['A_cen']
    if A_cen != 0:
        print('Adding assembly bias for centrals...')
        halo_pseudomass_cen = np.int64(
                halo_m * np.exp(A_cen*(2*(delta_c > 0) - 1)))
        ind_m = halo_m.argsort()
        ind_pm = halo_pseudomass_cen.argsort().argsort()
        halo_m = halo_m[ind_m][ind_pm]
    # calculate N_cen_mean using given model for all halos
    ht['N_cen_model'] = N_cen_mean(halo_m / h, model.param_dict)
    # create N_halos random numbers in half-open interval [0. 1) for centrals
    ht['N_cen_rand'] = np.random.random(N_halos)
    # if random number is less than model probably, halo hosts a central
    mask_cen = ht['N_cen_rand'] < ht['N_cen_model']
    # add halo velocity bias for observer along LOS
    # if alpha_c = 0, no bias is added.
    # this operation should be fast enough, no skip needed
    vz = do_vpec(ht['halo_vz'][mask_cen], ht['halo_vrms'][mask_cen],
                 model.param_dict['alpha_c'])
    # add observational rsd for halos
    if add_rsd:
        z = do_rsd(ht['halo_z'][mask_cen]/h, vz, model.mock.redshift,
                   model.mock.cosmology, model.mock.BoxSize/h) * h  # in Mpc/h
    else:
        z = ht['halo_z'][mask_cen]
    # central calculation done, create centrals table
    # create galaxy_table columns to be inherited from halo_table
    # excludes halo_z and halo_vz due to velocity bias and rsd
    col_cen_inh = ['halo_upid', 'halo_id', 'halo_hostid',
                   'halo_x', 'halo_y', 'halo_z',
                   'halo_vx', 'halo_vy', 'halo_vz',
                   'halo_klypin_rs', 'halo_nfw_conc', model.halo_m_prop,
                   'N_cen_model', 'N_cen_rand']
    gt_inh = table.Table([ht[col][mask_cen] for col in col_cen_inh],
                         names=col_cen_inh)
    # create new columns containing galaxy properties, with bias & rsd
    # r_centric is the host centric distance between particle and halo centre
    col_cen_new = ['x', 'y', 'z', 'vx', 'vy', 'vz']
    gt_new = table.Table(
        [ht['halo_x'][mask_cen],  ht['halo_y'][mask_cen],  np.float32(z),
         ht['halo_vx'][mask_cen], ht['halo_vy'][mask_cen], np.float32(vz)],
        names=col_cen_new)
    gt_new['r_centric'] = np.float32(0)
    # combine inherited fields and new field(s)
    gt_cen = table.hstack([gt_inh, gt_new], join_type='exact')
    gt_cen['gal_type'] = 'centrals'
    print('{} centrals generated.'.format(len(gt_cen)))

    '''
    satellites

    '''
    # before decorations, N_sat only depend on halo mass
    halo_m = ht[model.halo_m_prop].data
    A_sat = model.param_dict['A_sat']
    if A_sat != 0:
        halo_pseudomass_sat = np.int64(
                halo_m * np.exp(A_sat*(2*(delta_c > 0) - 1)))
        ind_m = halo_m.argsort()
        ind_pm = halo_pseudomass_sat.argsort().argsort()
        halo_m = halo_m[ind_m][ind_pm]
    # if subsamp_len = 0 for a row, we skip it and keep the undivided prob
    mask = ht['halo_subsamp_len'] != 0
    ht['N_sat_model'] = N_sat_mean(halo_m / h, model.param_dict)
    ht['N_sat_model'][mask] /= ht['halo_subsamp_len'][mask]
    # # fix inf due to dividing by zero
    # ht['N_sat_model'][ht['halo_subsamp_len'] == 0] = 0
    print('Creating inherited halo properties for centrals, r = {}...'
          .format(model.r))
    pt = model.mock.halo_ptcl_table  # 10% subsample of halo DM particles
    N_particles = len(pt)
    # inherite columns from halo talbe and add to particle table
    # particles belonging to the same halo share the same values
    col_cen_inh = ['halo_upid', 'halo_id', 'halo_hostid',
                   'halo_x', 'halo_y', 'halo_z',
                   'halo_vx', 'halo_vy', 'halo_vz',
                   'halo_klypin_rs', 'halo_nfw_conc', model.halo_m_prop,
                   'halo_subsamp_start', 'halo_subsamp_len',
                   'N_cen_model', 'N_cen_rand', 'N_sat_model']
    for col in list(set(col_cen_inh)):
        # numpy bug, cannot cast uint64 to int64 for repeat
        pt[col] = np.repeat(ht[col], ht['halo_subsamp_len'].astype(np.int64))
    # calculate additional particle quantities for decorations
    print('Processing particle table properties for r = {}...'.format(model.r))
    process_particle_props(pt, h, halo_m_prop=model.halo_m_prop)
    # particle table complete. onto satellite generation
    # calculate satellite probablity and generate random numbers
    pt['N_sat_rand'] = np.random.random(N_particles)
    # add particle ranking using s, s_v, s_p for each halo
    for key in ['rank_s', 'rank_s_v', 'rank_s_p']:
        # initialise column, astropy doesn't support empty columns
        pt[key] = np.int32(-1)

    # print('Creating particle indices...')
    # pidx = vrange(ht['halo_subsamp_start'].data,
    #               ht['halo_subsamp_len'].data)
    # splited arrays by halos, becomes iterable for MP
    # if model.param_dict['s'] != 0:
    #     r_centric = pt['r_centric'][pidx].data
    #     print('Splitting r_centric data array by halos...')
    #     r_centric_split = np.split(r_centric,
    #                                ht['halo_subsamp_start'].data[1:])
    #     print('Calculating halo centric distance rankings with MP...')
    #     pt['rank_s'] = rank_particles_by_halo(r_centric_split)
    # if model.param_dict['s_v'] != 0:
    #     v_pec = pt['v_pec'][pidx]
    #     print('Splitting v_pec data array by halos...')
    #     v_pec_split = np.split(v_pec, ht['halo_subsamp_start'].data[1:])
    #     print('Calculating peculiar speed rankings with MP...')
    #     pt['rank_s_v'] = rank_particles_by_halo(v_pec_split)
    # if model.param_dict['s_p'] != 0:
    #     r_perihelion = pt['r_perihelion'][pidx]
    #     print('Splitting r_perihelion data array by halos...')
    #     r_perihelion_split = np.split(r_perihelion,
    #                                   ht['halo_subsamp_start'].data[1:])
    #     print('Calculating perihelion distance rankings with MP...')
    #     pt['rank_s_p'] = rank_particles_by_halo(r_perihelion_split)
    print('Ranking particles within each halo for r = {} ...'.format(model.r))
    for i in range(N_halos):
        m = ht['halo_subsamp_start'][i]
        n = ht['halo_subsamp_len'][i]
        if model.param_dict['s'] != 0:
            # furtherst particle has lowest rank, 0, innermost N-1
            pt['rank_s'][m:m+n] = pt['r_centric'][m:m+n] \
                .argsort()[::-1].argsort()
        if model.param_dict['s_v'] != 0:
            pt['rank_s_v'][m:m+n] = pt['v_pec'][m:m+n] \
                .argsort()[::-1].argsort()
        if model.param_dict['s_p'] != 0:
            pt['rank_s_p'][m:m+n] = pt['r_perihelion'][m:m+n] \
                .argsort()[::-1].argsort()
    # calculate new probability regardless of decoration parameters
    # if any s is zero, the formulae guarantee the random numbers are unchanged
    # only re-rank halos where subsamp_len >= 2
    mask = pt['halo_subsamp_len'] >= 2
    pt['N_sat_model'][mask] *= 1 + model.param_dict['s'] * (
        1 - 2*pt['rank_s'][mask] / (pt['halo_subsamp_len'][mask]-1))
    pt['N_sat_model'][mask] *= 1 + model.param_dict['s_v'] * (
        1 - 2*pt['rank_s_v'][mask] / (pt['halo_subsamp_len'][mask]-1))
    pt['N_sat_model'][mask] *= 1 + model.param_dict['s_p'] * (
        1 - 2*pt['rank_s_p'][mask] / (pt['halo_subsamp_len'][mask]-1))
    # select particles which host satellites
    mask_sat = pt['N_sat_rand'] < pt['N_sat_model']
    # add RSD to mock sample
    if add_rsd:
        z = do_rsd(pt['z'][mask_sat]/h, pt['vz'][mask_sat],
                   model.mock.redshift, model.mock.cosmology,
                   model.mock.BoxSize/h) * h  # in Mpc/h
    else:
        z = pt['z'][mask_sat]
    print('Creating inherited halo properties for satellites, r = {}...'
          .format(model.r))
    # satellite calculation done, create satellites table
    # create galaxy_table columns to be inherited from particle table
    # all position and velocity columns except z are inherited from particle
    col_sat_inh = list(set(col_cen_inh + ['x', 'y', 'vx', 'vy', 'vz',
                                          'x_rel', 'y_rel', 'z_rel',
                                          'vx_pec', 'vy_pec', 'vz_pec',
                                          'v_rad', 'v_tan',
                                          'r_centric', 'r_perihelion',
                                          'N_sat_model', 'N_sat_rand',
                                          'rank_s', 'rank_s_v', 'rank_s_p']))
    for col in col_sat_inh:  # debug
        assert col in pt.keys()
        assert len(pt[col][mask_sat]) == len(pt[mask_sat])
    gt_inh = table.Table([pt[col][mask_sat] for col in col_sat_inh],
                         names=col_sat_inh)
    col_sat_new = ['z']
    # save z as float32 as all positions in catalogues are float32
    # this also ensures Corrfunc compatibility
    gt_new = table.Table([np.float32(z)], names=col_sat_new)
    # combine inherited fields and new field(s)
    gt_sat = table.hstack([gt_inh, gt_new], join_type='exact')
    gt_sat['gal_type'] = 'satellites'
    print('{} satellites generated.'.format(len(gt_sat)))
    # combine centrals table and satellites table
    model.mock.galaxy_table = table.vstack([gt_cen, gt_sat],
                                           join_type='outer')
    model.mock.halo_ptcl_table = pt

    return model
