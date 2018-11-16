# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:12:25 2018

@author: givoltage
"""

# rename gen_vel2 files

import os
from glob import glob

cosmology = 0
redshift = 0.5
sim_name_prefix = 'emulator_1100box_planck'
tagout = 'recon'  # 'z0.5'
store_dir = '/home/dyt/store/'
save_dir = os.path.join(store_dir, sim_name_prefix+'-'+tagout)

for phase in range(16):
    sim_name = '{}_{:02}-{}'.format(sim_name_prefix, cosmology, phase)
    # filedir = os.path.join(save_dir, sim_name, 'z{}'.format(redshift))
    filedir = os.path.join(save_dir,
                           '{}_{:02}-coadd'.format(sim_name_prefix, cosmology),
                           'z{}'.format(redshift))
    paths = glob(os.path.join(filedir, 'gen_vel2*'))
    for path in paths:
        if 'fft*' in path:
            print('renaming', path)
            os.rename(path, path.replace('fft*', 'fftcorr'))
        elif 'wp*' in path:
            print('renaming', path)
            os.rename(path, path.replace('wp*', 'wp'))
        else:
            pass
