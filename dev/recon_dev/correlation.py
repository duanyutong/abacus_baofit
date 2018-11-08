import numpy as np
from halotools.mock_observables import tpcf, s_mu_tpcf, tpcf_multipole
import matplotlib.pyplot as plt

h = 0.6726
b = 2.23
pos = np.fromfile('sample_input_gal_mx3_float32_0-1100_mpc.dat',
                  dtype=np.float32).reshape(-1, 4)[:, 1:4]*h
s_bins, mu_bins = np.arange(5, 155, 5), np.arange(0, 1.05, 0.05)
r = (s_bins[:-1] + s_bins[1:])/2
xi = tpcf(pos, s_bins, period=1100)
xi_s_mu = s_mu_tpcf(pos, s_bins, mu_bins, period=1100)
xi_0 = tpcf_multipole(xi_s_mu, mu_bins, order=0)
xi_2 = tpcf_multipole(xi_s_mu, mu_bins, order=2)
fig, ax = plt.subplots()
ax.plot(r, xi*np.square(r), 'o:', label='xi')
ax.plot(r, xi_0*np.square(r), '+:', label='xi_0')
ax.plot(r, xi_2*np.square(r), 'x-.', label='xi_2')
fig.legend()
fig.savefig('correlation.pdf')
