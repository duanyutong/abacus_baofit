import sys
import numpy as np
from Corrfunc.theory.DDsmu import DDsmu
from abacus_baofit import rebin_smu_counts, analytic_random, ls_estimator
from halotools.mock_observables import tpcf_multipole

tag = sys.argv[1]
s_bins_counts = np.arange(0, 151, 1)
s_bins = np.arange(0, 155, 5)
mu_bins = np.arange(0, 1.05, 0.05)

data = np.fromfile('/home/dyt/store/recon_temp/gal_cat-{}.dat'.format(tag), dtype=np.float32).reshape(-1, 4)
x = data[:, 1]
y = data[:, 2]
z = data[:, 3]
DDnpy = DDsmu(1, 30, s_bins_counts, 1, 100,
              x, y, z, periodic=True, verbose=True, boxsize=1100,
              output_savg=True, c_api_timer=False)
DD, savg = rebin_smu_counts(DDnpy)
savg_vec = np.sum(savg * DD, axis=1) / np.sum(DD, axis=1)
NR = ND = x.size
DR_ar, _, RR_ar = analytic_random(
        ND, NR, ND, NR, s_bins, mu_bins, 1100**3)
xi_s_mu = ls_estimator(ND, NR, ND, NR, DD, DR_ar, DR_ar, RR_ar)
xi_0 = tpcf_multipole(xi_s_mu, mu_bins, order=0)
xi_2 = tpcf_multipole(xi_s_mu, mu_bins, order=2)
xi_0_txt = np.vstack([savg_vec, xi_0]).T
xi_2_txt = np.vstack([savg_vec, xi_2]).T
for output, fn in zip([xi_s_mu, xi_0_txt, xi_2_txt],
                       ['xi', 'xi_0', 'xi_2']):
    np.savetxt('/home/dyt/store/recon_temp/auto-{}-{}-ar.txt'.format(fn, tag),
               output, fmt=b'%.30e')