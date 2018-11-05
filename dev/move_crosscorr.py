# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 14:56:32 2018

@author: givoltage
"""

import os
from glob import glob
from shutil import copyfile
from tqdm import tqdm

olddir = '/mnt/store1/dyt/emulator_1100box_planck-z0.5/'
newdir = '/mnt/store1/dyt/emulator_1100box_planck-recon-test/'
sim_name_prefix = 'emulator_1100box_planck'
N_reals = 12
cosmology = 0
redshift = 0.5
filelist = []
for r in range(N_reals):
    temp = os.path.join(olddir, sim_name_prefix+'*',
                        'z{}-r{}'.format(redshift, r),
                        '*-cross_*-*')
    print('temp is:', temp)
    ls = glob(temp)
    print('number of xcorr files founded for r = {} :, {}'.format(r, len(ls)))
    filelist += ls

for p in tqdm(filelist):
    newpath = p.replace('z0.5', 'recon-test', 1)
    if 'xi_s_mu' in newpath:
        newpath = newpath.replace('xi_s_mu', 'xi')
    newpath = newpath[:-4] + '-smu-pre-recon-ar' + newpath[-4:]
    copyfile(p, newpath)
