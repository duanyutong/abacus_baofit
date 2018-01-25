# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 17:53:18 2018

@author: givoltage

one-time use code to add r column before xi column in old xi data files
current abacus_baofit code already takes care of this

"""

import glob
import numpy as np

txtfmt = b'%.30e'
pattern = '/home/dyt/analysis_data/emulator_1100box_planck/*/*/*-xi_[02]*.txt'
paths = glob.glob(pattern)
s_bins = np.arange(1, 150 + 6, 6)
s_bins_centre = (s_bins[:-1] + s_bins[1:]) / 2
for i, path in enumerate(paths):
    print('Processing {} of {}...'.format(i, len(paths)))
    xi_ell = np.loadtxt(path)
    np.savetxt(path,
               np.vstack([s_bins_centre, xi_ell]).transpose(), fmt=txtfmt)
    print('{} rewritten.'.format(path))
