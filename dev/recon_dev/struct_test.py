# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 18:30:21 2018

@author: givoltage
"""

import numpy as np
from astropy.table import Table
import os

os.chdir(r'K:\Google Drive\Harvard\repository\recon_dev')
# check "correct" sample input and output of read.cpp
inp = np.fromfile(r'K:\Google Drive\Harvard\repository\recon\input',
                  dtype=np.float32)
outp = np.fromfile(r'K:\Google Drive\Harvard\repository\recon\output_data',
                   dtype=np.float64)

# produce sample input and output for read.cpp
gt = Table.read(r'K:\Google Drive\Harvard\debug\phase_0_z0.5_r0\gen_base1-galaxy_table.csv')
h = 0.7
max_sep = 250  # np.sqrt(3) * 1100
blank8 = 123.4567890123456789
N_pts = len(gt)
weights = np.ones(N_pts).reshape(-1, 1)
masses = np.zeros(N_pts).reshape(-1, 1)
header = [-550, -550, -550, 550, 550, 550, max_sep, blank8]

# gal input for read.cpp before recon.cpp, 0 to 1100/h in Mpc
pos = np.array([gt['x'], gt['y'], gt['z']]).T
arr = np.hstack([masses, pos/h]).astype(np.float32)
arr.tofile('sample_input_gal_mx3_float32_0-1100_mpc.dat', sep='')

# galaxy sample
pos = np.array([gt['x'], gt['y'], gt['z']]).T
arr = np.array(header + list(np.hstack([pos, weights]).flatten()))
arr.astype(np.float64).tofile(
    'sample_output_gal_x3w_float64_pm550_mpch.dat', sep='')

# randoms
# position range is -550 to +550
pos = np.random.rand(N_pts, 3) * 1100
arr = np.array(header + list(np.hstack([pos, weights]).flatten()))
arr.astype(np.float64).tofile(
    'sample_output_gal_random_x3w_float64_pm550_mpch.dat', sep='')

# read file
arr_shifted = np.fromfile(r'C:\Users\givoltage\Google Drive\Harvard\repository\recon_dev\sample_input_gal_mx3_float32_0-1100_mpch_rsd.dat', dtype=np.float32).reshape(-1,4)
