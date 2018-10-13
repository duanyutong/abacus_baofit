# -*- coding: utf-8 -*-
"""
re-calculate cross correlation from paircounts

Created on Sat Oct 13 12:08:14 2018

@author: givoltage
"""

from itertools import product
import os
import numpy as np
from abacus_baofit import rebin_smu_counts, subvol_mask
from astropy import table

phases = range(16)
N_reals = 12
N_sub = 3
random_multiplier = 10
cosmology = 0
redshift = 0.5
L = 1100
model_names = ['gen_base1', 'gen_base4', 'gen_base5',
               'gen_ass1', 'gen_ass2', 'gen_ass3']
sim_name_prefix = 'emulator_1100box_planck'
tagout = 'recon'  # 'z0.5'
prod_dir = r'/mnt/gosling2/bigsim_products/emulator_1100box_planck_products/'
store_dir = '/home/dyt/store/'
recon_temp_dir = os.path.join(store_dir, 'recon/temp/')
save_dir = os.path.join(store_dir, sim_name_prefix+'-'+tagout)

for phase in phases:
    for model_name in model_names:
        for r in range(N_reals):
            sim_name = '{}_{:02}-{}'.format(sim_name_prefix, cosmology, phase)
            filedir = os.path.join(save_dir, sim_name,
                                   'z{}-r{}'.format(redshift, r))
            gt_path = os.path.join(filedir, '{}-galaxy_table.csv'
                                   .format(model_name))
            gt = table.Table.read(
                gt_path, format='ascii.fast_csv',
                fast_reader={'parallel': True, 'use_fast_converter': False})
            ND1 = len(gt)
            NR1 = ND1 * random_multiplier
            print('r = {}, reading shifted samples...'.format(r))
            seed = 100 * phase + r
            R = np.float32(np.fromfile(  # read shifted randoms
                os.path.join(recon_temp_dir, 'file_R-{}_rec'.format(seed)),
                dtype=np.float64)[8:].reshape(-1, 4))[:, :3]
            nr = R[:NR1]
            N_copies = int(random_multiplier/2)
            nr_list = np.array_split(model.mock.numerical_randoms, N_copies)
            for i, j, k in product(range(N_sub), repeat=3):
                linind = i*N_sub**2 + j*N_sub + k
                x2, _, _ = subvol_mask(gt['x'], gt['y'], gt['z'],
                                       i, j, k, L, N_sub)
                ND2 = len()
                DDnpy = np.load(os.path.join(
                        filedir, '{}-cross_{}-paircount-DD.npy'
                        .format(model_name, linind)))
                DD_npairs, _ = rebin_smu_counts(DDnpy)  # re-bin and re-weight
