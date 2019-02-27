from __future__ import (
        absolute_import, division, print_function, unicode_literals)
from itertools import product
from functools import reduce
import os
import numpy as np
from scipy.optimize import minimize
import sys
import traceback
from scipy.special import legendre
from scipy.interpolate import InterpolatedUnivariateSpline
from p2xi import PowerTemplate
from tqdm import tqdm

# setup to run the data in the Ross_2016_COMBINEDDR12 folder
# polydeg = '3p'  # 1, 2, 3, 3p
# rmin = 50  # 20, 30, 40, 50
# Hdir = '/home/dyt/store/AbacusCosmos_1100box_planck/'
# ft = 'zheng07' # common prefix of all data files cinlduing last '_'
# zb = ''  # zb = 'zbin3_' # change number to change zbin
# binc = ''  # binc = 0 # change number to change bin center
# bs = 5.0  # the r bin size of the data
# bc = '.txt'  # bc = 'post_recon_bincent'+str(binc)+'.dat'
# fout = ft
n_mu_bins = 1000
chisq_min = 1000
xi_temp_rescale = 2.9
r_avg_increment = 1
txtfmt = b'%.30e'


def matmul(*args):
    return reduce(np.dot, [*args])


class baofit3D:

    def __init__(self, k, P_lin, r, xi_0, xi_2, cov, polydeg, reconstructed,
                 z=0.5, r_min=50, r_max=150, xi_temp_weighted=False):

        power = PowerTemplate(z=z, reconstructed=reconstructed)
        power.P(k, P_lin, n_mu_bins, beta=0.25)
        self.xi_multipole = {}
        for ell in [0, 2, 4]:
            self.xi_multipole[ell] = InterpolatedUnivariateSpline(
                r, power.xi_multipole(r, ell=ell))
        self.polydeg = polydeg
        self.xi_temp_weighted = xi_temp_weighted
        # monoquad r vector, with factor to correct for pairs should have
        # slightly larger average pair distance than the bin center
        r_mask = (r_min < r) & (r < r_max)
        self.r = r[r_mask]
        self.rv = np.hstack([self.r, self.r])  # * 1.000396  # debug
        self.xidata = {'0': xi_0[r_mask], '2': xi_2[r_mask]}
        self.dv = np.hstack([self.xidata['0'], self.xidata['2']])
        print('Shape of data vector:', self.dv.shape)
        ri_mask = np.tile(r_mask, (len(r_mask), 1)).T
        rj_mask = np.tile(r_mask, (len(r_mask), 1))
        quadrant_mask = ri_mask & rj_mask
        covm_mask = np.tile(quadrant_mask, (2, 2))
        self.cov = cov[covm_mask].reshape(self.dv.size, self.dv.size)
        self.icov = np.linalg.pinv(self.cov)
        # print('{} - Length of data vector: {}'.format(fout_tag, dv.size))
        self.n = self.r.size
        self.H = np.zeros((2*self.n, 6))
        if self.polydeg == '1':  # set up auxillary matrix H
            self.H[:self.n,  1] = 1/np.square(self.r)
            self.H[-self.n:, 4] = self.H[:self.n, 0]
        elif self.polydeg == '2':
            self.H[:self.n,  0] = 1/np.square(self.r)
            self.H[:self.n,  1] = 1/self.r
            self.H[-self.n:, 3:5] = self.H[:self.n, :2]
        elif self.polydeg == '3':
            self.H[:self.n,  0] = 1/np.square(self.r)
            self.H[:self.n,  1] = 1/self.r
            self.H[:self.n,  2] = 1
            self.H[-self.n:, 3:] = self.H[:self.n, :3]
        elif self.polydeg == '3p':
            self.H[:self.n,  1] = 1/self.r
            self.H[:self.n,  2] = 1
            self.H[-self.n:, 4:] = self.H[:self.n, :2]
        self.at = 1
        self.ar = 1
        self.B0 = 1  # default prior values are for initial bias fitting
        self.B2 = 1  # default prior values are for initial bias fitting
        self.B0_ln_width = 100
        self.B2_ln_width = 100
        self.A = np.zeros(6)

    def xi_r_mu(self, r, mu, ell_max=6):
        ''' input r may be 2D, return 2D xi_r_mu(r, mu) '''
        orders = np.arange(ell_max//2)*2
        xi = 0
        for order in orders:
            xi_ell = self.xi_multipole[order](r)
            Ln = legendre(order)
            xi += xi_ell * Ln(mu)
        return xi*xi_temp_rescale

    def xi_temp_components(self, r, component='mu2'):
        ''' rescaled xi_0, xi_2, xi_mu_2 as a function of 1D r array '''
        mu_bins = np.linspace(0, 1, num=n_mu_bins+1)
        mu = (mu_bins[1:] + mu_bins[:-1])/2
        dmu = np.diff(mu_bins)
        alpha = np.sqrt(np.square(mu*self.ar)
                        + (1-np.square(mu))*np.square(self.at))  # size n_mu
        rp = np.outer(r, alpha)  # r prime rescaled, 2D dim (n_r, n_mu)
        mup = mu*self.ar/alpha  # mu prime, size n_mu
        # mup = np.tile(mup, (self.n, 1))
        if component in ['0', '2']:  # can add any even order, '2', '4'
            ell = np.int(component)
            Ln = legendre(ell)
            xi = (2*ell + 1) / 2 * np.sum(
                    self.xi_r_mu(rp, mup)*(Ln(-mup)+Ln(mup))*dmu, axis=1)
        elif component == 'mu2':
            xi = np.sum(3/2*np.square(mup)*self.xi_r_mu(rp, mup)*dmu, axis=1)
        assert self.r.size == xi.size
        return xi

    def make_xi_temp(self, weighted=False):
        ''' xi_0, xi_2, xi_mu_2 over r bin, weighted by r^2 integral '''
        if not hasattr(self, 'xitemp'):
            self.xitemp = {}
        n = np.int(np.floor(np.mean(np.diff(self.r))/r_avg_increment)//2*2+1)
        for comp in ['0', '2', 'mu2']:
            if weighted:
                # print('Making weighted xi {} tempplate across bin width...'
                #       .format(comp))
                array = np.zeros((self.r.size, 2, n))
                for i in range(n):
                    r = self.r - (n-1)/2*r_avg_increment + i*r_avg_increment
                    # print('r bin {} of {}, r = {}'.format(i, n, r))
                    array[:, 0, i] = r
                    array[:, 1, i] = self.xi_temp_components(r, component=comp)
                self.xitemp[comp] = np.sum(
                    np.square(array[:, 0, :])*array[:, 1, :],
                    axis=1) / np.sum(np.square(array[:, 0, :]), axis=1)
            else:
                # print('Making unweighted xi {} template ...'.format(comp))
                self.xitemp[comp] = self.xi_temp_components(self.r,
                                                            component=comp)

    def make_model_vector(self, B0, B2):
        ''' calculate model vector from templates using B, A coefficients '''
        self.ximod = {
            '0': B0*self.xitemp['0']
            + self.A[0]/np.square(self.r) + self.A[1]/self.r + self.A[2],
            '2': 5*(B2*self.xitemp['mu2']-B0/2*self.xitemp['0'])
            + self.A[3]/np.square(self.r) + self.A[4]/self.r + self.A[5]}
        self.mv = np.hstack([self.ximod['0'], self.ximod['2']])  # model vector

    def calculate_chisq(self, B0, B2):
        ''' input Bs is a tuple of two B coefficients '''
        if B0 < 0 or B2 < 0:
            return chisq_min
        dxi = self.dv - self.mv  # delta xi vector between data and model
        chisq = matmul(dxi, self.icov, dxi)
        B0_term = np.square(np.log(B0/self.B0)/self.B0_ln_width)
        B2_term = np.square(np.log(B2/self.B2)/self.B2_ln_width)
        # print('chisq components', chisq, B0_term, B2_term)
        factor = (chisq_min-2*self.n) / (chisq_min-1)
        return factor * (chisq + B0_term + B2_term)

    def B0_B2_optimize(self, Bs):

        B0, B2 = Bs
        self.A = np.zeros(6)  # assume zero A before taking diff solving for A
        self.make_model_vector(B0=B0, B2=B2)  # get model vector with 0 A
        self.A = matmul(
                np.linalg.pinv(matmul(self.H.T, self.icov, self.H)),
                self.H.T, self.icov, self.dv-self.mv)  # solve for A hat
        self.make_model_vector(B0=B0, B2=B2)  # now make new mv with A hat
        return self.calculate_chisq(B0, B2)

    def do_fit(self,
               armin=0.910, armax=1.040, arsp=0.005,
               atmin=0.990, atmax=1.040, atsp=0.005):
        '''
        armin=0.975, armax=1.035, arsp=0.0004
        atmin=1.000, atmax=1.040, atsp=0.0004
        '''
        ars = np.arange(armin, armax, arsp)
        ats = np.arange(atmin, atmax, atsp)
        chisq_grid = np.zeros((ars.size, ats.size))
        chisq_list = []
        for i, j in tqdm(product(range(ars.size), range(ats.size))):
            self.ar = ars[i]
            self.at = ats[j]
            self.make_xi_temp(weighted=self.xi_temp_weighted)
            B0, B2 = minimize(self.B0_B2_optimize, (1, 1),
                              bounds=((0.1, 2), (0.1, 2)), method='SLSQP').x
            chisq_min = self.B0_B2_optimize((B0, B2))
            chisq_grid[i, j] = chisq_min
            chisq_list.append([ars[i], ats[j], chisq_min])
            # print('Grid {}/{}, ({}, {}), {}'.format(
            #         i*ats.size+j+1, chisq_grid.size,
            #         ars[i], ats[j], chisq_min))
            # print('B0, B2, A:', B0, B2, self.A)
        return chisq_grid, np.array(chisq_list)


def baofit(argv):
    try:
        z, path_p_lin, path_xi_0, path_xi_2, path_cov, polydeg, rmin, \
            path_chisq_grid, path_chisq_table, recon = argv
        print('Saving chisq to:', os.path.dirname(path_chisq_grid))
        data = np.loadtxt(path_p_lin)
        k = data[:, 0]
        P_lin = data[:, 1]
        data = np.loadtxt(path_xi_0)
        r = data[:, 0] * 1.000396
        xi_0 = data[:, 1]
        xi_2 = np.loadtxt(path_xi_2)[:, 1]
        cov = np.loadtxt(path_cov)  # combined monopole, quadrupole cov matrix
        assert type(recon) is bool
        print('xi samplesa are reconstructed:', recon)
        try:
            assert (r.size == xi_0.size == xi_2.size
                    == cov.shape[0]/2 == cov.shape[1]/2)
            assert not np.any(cov == np.nan)
        except AssertionError:
            print('Assertion error, shapes', r.shape, xi_0.shape, xi_2.shape,
                  cov.shape[0]/2, cov.shape[1]/2)
            print('NaN values in cov at: ', np.where(cov == np.nan))
            raise Exception
        # find best B_0 from bias prior, reduced range
        # print('Initializing bias prior instance')
        fit1 = baofit3D(k, P_lin, r, xi_0, xi_2, cov, polydeg, recon,
                        z=z, r_min=rmin, r_max=80, xi_temp_weighted=True)
        fit1.make_xi_temp()
        B0 = np.arange(0.1, 2, 0.01)  # template & data scale differ by 2 to 3
        chisq = np.zeros(B0.shape)
        for i in range(B0.size):
            fit1.make_model_vector(B0[i], 1)  # A and widths set with init
            chisq[i] = fit1.calculate_chisq(B0[i], 1)
        assert np.any(chisq < chisq_min) is np.True_
        # print('\nBest-fit prior B0 {}, chisq_min {}'
        #       .format(B0[np.argmin(chisq)], np.min(chisq)))
        # update prior and do chisq grid scan, full range
        fit2 = baofit3D(k, P_lin, r, xi_0, xi_2, cov, polydeg, recon,
                        z=z, r_min=rmin, r_max=150, xi_temp_weighted=True)
        fit2.B0 = B0[np.argmin(chisq)]  # new B0 prior for fit2
        fit2.B2 = fit2.B0  # same prior for B2
        fit2.B0_ln_width = 0.4
        fit2.B2_ln_width = 0.4
        chisq_grid, chisq_table = fit2.do_fit(
            armin=0.982, armax=1.011, arsp=0.0004,
            atmin=1.000, atmax=1.020, atsp=0.0004
            )
        if not os.path.exists(os.path.dirname(path_chisq_grid)):
            os.mkdir(os.path.dirname(path_chisq_grid))
        np.savetxt(path_chisq_grid, chisq_grid, fmt=txtfmt)
        np.savetxt(path_chisq_table, chisq_table, fmt=txtfmt)
        ar, at, chisq = chisq_table[np.argmin(chisq_table[:, 2]), :]
        print('\nBest-fit ar, at, chisq: ({}, {}), {}'
              .format(ar, at, chisq))
    except Exception as E:
        print('Exception caught in worker thread {}'.format(path_xi_0))
        traceback.print_exc()
        raise E


if __name__ == '__main__':

    baofit(sys.argv[1:])
