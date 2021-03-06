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
chisq_min = 2000
r_avg_increment = 1
txtfmt = b'%.30e'


def matmul(*args):
    return reduce(np.dot, [*args])


class baofit3D:

    def __init__(self, sample_type, z, k, P_lin, r, xi_0, xi_2, cov,
                 polydeg, beta, reconstructed, force_isotropy=False,
                 r_min=50, r_max=150, xi_temp_weighted=False,
                 xi_temp_rescale=1):

        power = PowerTemplate(z=z)
        Sigma_s = 4  # 4 Mpc/h, streaming scale
        Sigma_r = 15  # 15 Mpc/h, smoothing scale used in reconstruction
        self.reconstructed = reconstructed
        self.sample_type = sample_type
        if self.reconstructed:
            Sigma_perp = 0  # 2.5 Mpc/h, the smaller, the sharper BAO peak
            # Sigma_para = 4  # Mpc/h
        else:  # pre-recon peak is wider
            if self.sample_type == 'mat':
                Sigma_perp = 1.5  # 1.5 Mpc/h for matter, 5 Mpc/h for galaxies
            elif self.sample_type == 'gal':
                Sigma_perp = 5
            # Sigma_para = 10  # Mpc/h
        Sigma_para = Sigma_perp / (1 - beta)  # Mpc/h
        power.P(k, P_lin, n_mu_bins, beta=beta, Sigma_s=Sigma_s,
                Sigma_r=Sigma_r, Sigma_perp=Sigma_perp, Sigma_para=Sigma_para)
        self.xi_multipole = {}
        for ell in [0, 2, 4]:
            self.xi_multipole[ell] = InterpolatedUnivariateSpline(
                r, power.xi_multipole(r, ell=ell))
        self.force_isotropy = force_isotropy
        self.polydeg = polydeg
        self.xi_temp_weighted = xi_temp_weighted
        self.xi_temp_rescale = xi_temp_rescale
        r_mask = (r_min < r) & (r < r_max)
        self.r = r[r_mask]
        self.rfull = np.arange(1, 151)
        self.rv = np.hstack([self.r, self.r])
        self.xidata = {'0': xi_0[r_mask],
                       '2': xi_2[r_mask]}
        self.dv = np.hstack([self.xidata['0'], self.xidata['2']])
        ri_mask = np.tile(r_mask, (len(r_mask), 1)).T
        rj_mask = np.tile(r_mask, (len(r_mask), 1))
        quadrant_mask = ri_mask & rj_mask
        covm_mask = np.tile(quadrant_mask, (2, 2))
        self.cov = cov[covm_mask].reshape(self.dv.size, self.dv.size)
        self.icov = np.linalg.pinv(self.cov)
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
        self.range_ok = False
        print(('\nReconstructed: {}, sample: {}, force isotropy: {}, '
               'xitemp rescale factor: {:.2f}, data vector shape: {}')
              .format(reconstructed, sample_type, force_isotropy,
                      xi_temp_rescale, self.dv.shape))

    def xi_r_mu(self, r, mu, ell_max=6):
        ''' input r may be 2D, return 2D xi_r_mu(r, mu) '''
        orders = np.arange(ell_max//2)*2
        xi = 0
        for order in orders:
            xi_ell = self.xi_multipole[order](r)
            Ln = legendre(order)
            xi += xi_ell * Ln(mu)
        return xi * self.xi_temp_rescale

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
        assert r.size == xi.size
        return xi

    def make_xi_temp(self, r0=None):
        ''' xi_0, xi_2, xi_mu_2 over r bin, weighted by r^2 integral '''
        if r0 is None:
            r0 = self.r
        n = np.int(np.floor(np.mean(np.diff(r0))/r_avg_increment)//2*2+1)
        ret = {}
        for comp in ['0', '2', 'mu2']:
            if self.xi_temp_weighted:
                # print('Making {}-weighted xi {} template across bin width...'
                #       .format(n, comp))
                array = np.zeros((r0.size, 2, n))
                for i in range(n):
                    r = r0 - (n-1)/2*r_avg_increment + i*r_avg_increment
                    # print('r bin {} of {}, r = {}'.format(i, n, r))
                    array[:, 0, i] = r
                    array[:, 1, i] = self.xi_temp_components(r, component=comp)
                ret[comp] = np.sum(
                    np.square(array[:, 0, :])*array[:, 1, :],
                    axis=1) / np.sum(np.square(array[:, 0, :]), axis=1)
            else:
                # print('Making unweighted xi {} template ...'.format(comp))
                ret[comp] = self.xi_temp_components(r0, component=comp)
        return ret

    def make_model_vector(self, B0, B2, r0=None):
        ''' calculate model vector from templates using B, A coefficients '''
        if r0 is None:
            r0 = self.r
        ximod0 = B0*self.xitemp['0'] \
            + self.A[0]/np.square(r0) + self.A[1]/r0 + self.A[2]
        ximod2 = 5*(B2*self.xitemp['mu2']-B0/2*self.xitemp['0']) \
            + self.A[3]/np.square(r0) + self.A[4]/r0 + self.A[5]
        self.mv = np.hstack([ximod0, ximod2])  # model vector
        return ximod0, ximod2

    def calculate_chisq(self, B0, B2):
        if B0 < 0 or B2 < 0:
            return chisq_min
        dxi = self.dv - self.mv  # delta xi vector between data and model
        chisq = matmul(dxi, self.icov, dxi)
        B0_term = np.square(np.log(B0/self.B0)/self.B0_ln_width)
        B2_term = np.square(np.log(B2/self.B2)/self.B2_ln_width)
        # print('chisq components', chisq, B0_term, B2_term)
        factor = (chisq_min-2*self.n) / (chisq_min-1)
        return factor * (chisq + B0_term + B2_term)

    def B0_B2_optimize(self, Bs):  # function to be minimised by varying B0, B2
        B0, B2 = Bs
        self.A = np.zeros(6)  # assume zero A before taking diff solving for A
        self.make_model_vector(B0, B2)  # get model vector with 0 A
        self.A = matmul(
                np.linalg.pinv(matmul(self.H.T, self.icov, self.H)),
                self.H.T, self.icov, self.dv-self.mv)  # solve for A hat
        self.make_model_vector(B0, B2)  # now make new mv with A hat
        return self.calculate_chisq(B0, B2)

    def chisq_scan(self):
        '''
        matter pre-recon:
                ars = np.arange(1.0065, 1.0110, 0.00005)
                ats = np.arange(0.9985, 1.0020, 0.00005)

        matter post-recon:
                ars = np.arange(0.9960, 0.9990, 0.00005)
                ats = np.arange(0.9995, 1.0020, 0.00005)

        galaxy pre-recon:
                ars = np.arange(0.990, 1.016, 0.0001)
                ats = np.arange(0.993, 1.016, 0.0001)

        galaxy post-recon:
                ars = np.arange(0.990, 1.016, 0.0001)
                ats = np.arange(0.993, 1.016, 0.0001)
        '''
        if self.sample_type == 'mat':
            if not self.reconstructed:  # pre-recon
                ars = np.arange(1.0065, 1.0110, 0.00005)
                ats = np.arange(0.9985, 1.0020, 0.00005)
            else:  # post=recon
                ars = np.arange(0.9960, 0.9990, 0.00005)
                ats = np.arange(0.9995, 1.0020, 0.00005)
        elif self.sample_type == 'gal':
            if not self.reconstructed:  # pre-recon
                ars = np.arange(0.990, 1.016, 0.0001)
                ats = np.arange(0.993, 1.016, 0.0001)
            else:  # post-recon
                ars = np.arange(0.990, 1.016, 0.0001)
                ats = np.arange(0.993, 1.016, 0.0001)
        chisq_grid = np.zeros((ars.size, ats.size))  # default value is 0
        B0_grid = np.zeros(chisq_grid.shape)
        B2_grid = np.zeros(chisq_grid.shape)
        chisq_list = []
        print('\nChi-square grid scan size', chisq_grid.size)
        for i, j in tqdm(product(range(ars.size), range(ats.size))):
            if self.force_isotropy and ars[i] != ats[j]:
                continue
            self.ar = ars[i]
            self.at = ats[j]
            self.xitemp = self.make_xi_temp()
            ret = minimize(self.B0_B2_optimize, (1, 1),
                           bounds=((0.1, 2), (0.1, 2)), method='SLSQP')
            # B0, B2 = solution.x
            # chisq_min = self.B0_B2_optimize((B0, B2))
            chisq_grid[i, j] = ret.fun
            B0_grid[i, j], B2_grid[i, j] = ret.x[0], ret.x[1]
            chisq_list.append([ars[i], ats[j], ret.fun])
            # print('B0, B2, A:', B0, B2, self.A)
        valid_idx = np.where(chisq_grid > 0)  # np.where returns a tuple
        i = valid_idx[0][chisq_grid[valid_idx].argmin()]
        j = valid_idx[1][chisq_grid[valid_idx].argmin()]
        chisq_table = np.array(chisq_list)
        model_dump = self.model_dump(ars[i], ats[j], chisq_grid[i, j],
                                     B0_grid[i, j], B2_grid[i, j])
        if ars.min() < ars[i] < ars.max() and ats.min() < ats[j] < ats.max():
            self.range_ok = True  # must be strictly unequal to be within range
        return chisq_grid, chisq_table, model_dump

    def model_dump(self, ar, at, chisq_min, B0, B2):
        self.ar = ar
        self.at = at
        self.xitemp = self.make_xi_temp()
        chisq = self.B0_B2_optimize((B0, B2))  # this sets A hat coefficients
        dxi = self.dv - self.mv
        print('dxi_0:', np.mean(dxi[:self.n]),
              'dxi_2:', np.mean(dxi[self.n:]), 'chisq', chisq)  # debug
        assert chisq == chisq_min
        r_full = np.arange(2.5, 150, 5)  # np.arange(1, 151, 1)  # debug
        self.xitemp = self.make_xi_temp(r0=r_full)
        ximod0, ximod2 = self.make_model_vector(B0, B2, r0=r_full)
        return np.array([r_full, ximod0, ximod2,
                         self.xitemp['0'], self.xitemp['2']]).T


def baofit(argv):
    try:
        sample_type, z, recon, force_isotropy, polydeg, rmin, beta, \
            path_p_lin, path_xi_0, path_xi_2, path_cov, path_cg, path_ct = argv
        print('Saving chisq:', os.path.dirname(path_cg))
        if not os.path.exists(os.path.dirname(path_cg)):
            try:
                os.mkdir(os.path.dirname(path_cg))
            except FileExistsError:
                pass
        print('inputs are:\n', argv)
        data = np.loadtxt(path_p_lin)
        k = data[:, 0]
        P_lin = data[:, 1]
        data = np.loadtxt(path_xi_0)
        # monoquad r vector, with factor to correct for pairs should have
        # slightly larger average pair distance than the bin center
        r = data[:, 0]
        xi_0 = data[:, 1]
        xi_2 = np.loadtxt(path_xi_2)[:, 1]
        cov = np.loadtxt(path_cov)
        assert type(recon) is bool
        assert (r.size == xi_0.size == xi_2.size
                == cov.shape[0]/2 == cov.shape[1]/2), \
            'Assertion error, shapes {}, {}, {}, {}, {}'.format(
            r.shape, xi_0.shape, xi_2.shape, cov.shape[0]/2, cov.shape[1]/2)
        assert not np.any(cov == np.nan), (
            'NaN values in cov at: {}'.format(np.where(cov == np.nan)))
        # find best B_0 from bias prior, reduced range
        fit = baofit3D(sample_type, z, k, P_lin, r, xi_0, xi_2, cov,
                       polydeg, beta, recon, force_isotropy=force_isotropy,
                       r_min=rmin, r_max=150, xi_temp_weighted=False,
                       xi_temp_rescale=1)
        fit.xitemp = fit.make_xi_temp()
        # determine best-fit rescaling factor and target bias using xi_0
        xi_temp_rescale = matmul(fit.xitemp['0'].T, fit.xidata['0'])/matmul(
            fit.xitemp['0'].T, fit.xitemp['0'])
        fit = baofit3D(sample_type, z, k, P_lin, r, xi_0, xi_2, cov,
                       polydeg, beta, recon, force_isotropy=force_isotropy,
                       r_min=rmin, r_max=80, xi_temp_weighted=False,
                       xi_temp_rescale=xi_temp_rescale)
        fit.xitemp = fit.make_xi_temp()
        B0_list = np.arange(0.1, 2, 0.01)
        chisq = np.zeros(B0_list.shape)
        for i, B0 in enumerate(B0_list):
            fit.make_model_vector(B0, 1)  # A and widths set with init
            chisq[i] = fit.calculate_chisq(B0, 1)
            # print('Prior B0, B2, chisq: {}, 1, {}'.format(B0, chisq[i]))
        assert np.any(chisq < chisq_min) is np.True_, \
            ('B0 = {} at lowest chisq = {} > chisq_min,'
             .format(B0_list[np.argmin(chisq)], chisq.min()))
        # update prior and do chisq grid scan, full range
        fit = baofit3D(sample_type, z, k, P_lin, r, xi_0, xi_2, cov,
                       polydeg, beta, recon, force_isotropy=force_isotropy,
                       r_min=rmin, r_max=150, xi_temp_weighted=True,
                       xi_temp_rescale=xi_temp_rescale)
        fit.B0 = fit.B2 = B0_list[np.argmin(chisq)]  # new B0 prior for fit2
        fit.B0_ln_width = 0.4
        fit.B2_ln_width = 0.4
        chisq_grid, chisq_table, model_dump = fit.chisq_scan()
        np.savetxt(path_cg, chisq_grid, fmt=txtfmt)
        np.savetxt(path_ct, chisq_table, fmt=txtfmt)
        np.savetxt(path_cg.replace('chisq_grid.txt', 'xi_0_2_model.txt'),
                   model_dump, fmt=txtfmt)
        ar, at, chisq = chisq_table[np.argmin(chisq_table[:, 2]), :]
        print(('\nBest-fit ar, at, chisq: ({:.4f}, {:.4f}), {:.2f}, '
               'range OK: {}')
              .format(ar, at, chisq, fit.range_ok))
        return ar, at
    except Exception as E:
        print('Exception caught in worker thread {}'.format(path_xi_0))
        traceback.print_exc()
        raise E


if __name__ == '__main__':

    baofit(sys.argv[1:])
