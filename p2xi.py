# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 21:08:49 2019

@author: givoltage

Convert power spectrum template to xi template for BAO fitting
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
from scipy.special import legendre
from scipy.integrate import romberg
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from scipy.special import spherical_jn


# %%
def W(kR):
    '''
    spherical top hat window function in Fourier space
    MBW Eqn 6.32, input argument y = x/r = kr, dimensionless
    '''
    return 3 * (np.sin(kR)-(kR*np.cos(kR))) / np.power(kR, 3)


class PowerTemplate:

    def __init__(self, reconstructed=True, force_isotropy=False, z=0.5,
                 ombhsq=0.02222, omhsq=0.14212,
                 sigma8=0.830, h=0.6726, ns=0.9652, Tcmb0=2.725):
        # print('Power Temp is isotropic: ', isotropic)
        self.z = z
        self.h = h
        self.reconstructed = reconstructed
        self.force_isotropy = force_isotropy
        self.ombhsq = ombhsq  # Omega matter baryon
        self.omhsq = omhsq  # Omega matter (baryon + CDM)
        self.om = omhsq/np.square(h)
        self.omb = ombhsq/np.square(h)
        self.cosmology = FlatLambdaCDM(H0=100*h, Om0=self.om, Tcmb0=Tcmb0,
                                       Ob0=self.omb)
        self.ol = 1 - self.omhsq/np.square(h)  # ignring radiation density
        self.sigma8 = sigma8
        self.ns = ns
        self.Tcmb0 = 2.725
        self.Rscale = 8
        self.P0smooth = 1  # initial value for the coefficient
        self.P0smooth = np.square(sigma8/self.siglogsmooth())  # correction
        # omega=0.3,lamda=0.7,h=0.7,h0=1.,ombhh=0.0224,CMBtemp=2.725,sig8=.8,
        # c=2997,hmode=0,sbao=8.,nindex=.95

    def T0(self, k):
        '''
        Zero-baryon transfer function shape from E&H98, input k in h/Mpc
        '''
        k = k * self.h  # convert unit of k from h/Mpc to 1/Mpc
        #  Eqn 26, approximate sound horizon in Mpc
        s = 44.5*np.log(9.83/self.omhsq) / np.sqrt(
                1+10*np.power(self.ombhsq, 3/4))
        # Eqn 28, both dimensionless
        Theta = self.Tcmb0 / 2.7
        # Eqn 31, alpha_gamma, dimensionless
        agam = 1 - 0.328 * np.log(431*self.omhsq) * self.ombhsq/self.omhsq \
            + 0.38*np.log(22.3*self.omhsq) * np.square(self.ombhsq/self.omhsq)
        # Eqn 30, in unit of h
        gamma_eff = self.omhsq/self.h*(agam+(1-agam)/np.power(1+(0.43*k*s), 4))
        q = k / self.h * np.square(Theta) / gamma_eff

        C0 = 14.2 + 731 / (1+(62.5*q))  # Eqn 29
        L0 = np.log(2*np.e + 1.8*q)
        T0 = L0 / (L0 + C0*np.square(q))
        return T0

    def siglogsmooth(self, tol=1e-10):
        '''rms mass fluctuations (integrated in logspace) for sig8 etc.
        NOTES: formalism is all real-space distance in Mpc/h, all k in
        h/Mpc
        '''
        return np.sqrt(romberg(self.intefunclogsmooth, -10, 10, tol=tol))

    def intefunclogsmooth(self, logk):
        """
        Function to integrate to get rms mass fluctuations (logspace) base 10
        NOTES: Because the k's here are all expressed in h/Mpc the
        value of P0 that arises using this integration is in
        (Mpc**3)/(h**3.) and therefore values of P(k) derived from this
        integration are in (Mpc**3)/(h**3.), i.e. h^-3Mpc^3 and need
        divided by h^3 to get the 'absolute' P(k) in Mpc^3.
        v1.0 Adam D. Myers Jan 2007
        """
        k = np.power(10.0, logk)  # base 10
        kR = k * self.Rscale
        integrand = np.log(10) / (2*np.square(np.pi)) * np.power(k, 3) \
            * self.Psmooth(k) * np.square(W(kR))
        return integrand

    def Psmooth(self, k):
        """ Baryonic Linear power spectrum SHAPE, pass k in h/Mpc """
        om = self.om
        ol = self.ol
        omz = self.cosmology.Om(self.z)  # Omega matter at z
        olz = ol/np.square(self.cosmology.efunc(self.z))  # MBW Eqn 3.77
        g0 = 5/2*om/(np.power(om, 4/7) - ol + ((1+om/2)*(1+ol/70)))  # Eqn 4.76
        gz = 5/2*omz/(np.power(omz, 4/7) - olz + ((1+omz/2)*(1+olz/70)))
        Dlin_ratio = gz / (1+self.z) / g0
        Psmooth = self.P0smooth * np.square(self.T0(k)) * \
            np.power(k, self.ns) * np.square(Dlin_ratio)
        return Psmooth

    def P(self, k_lin, P_lin, n_mu_bins, beta=0.4, Sigma_s=4):
        """
        This is the final power template from P_lin and P_nw
        including the C and exponential damping terms
        input k, P are 1D arrays from CAMB, n_mu_bins is an integer
        defauolt beta = 0.4, dimensionless
        """
        self.k = k_lin  # 1D array
        self.P_lin = P_lin  # 1D array
        self.P_nw = self.Psmooth(k_lin)  # 1D array
        self.mu_bins = np.linspace(0, 1, num=n_mu_bins+1)  # 1D bin edges
        self.mu = (self.mu_bins[1:] + self.mu_bins[:-1])/2  # 1D bin centres
        mu, k = np.meshgrid(self.mu, k_lin)  # 2D meshgrid
        P_lin = np.tile(P_lin, (n_mu_bins, 1)).T  # meshgrid
        P_nw = self.Psmooth(k)  # follows meshgrid shape of k_lin
        try:
            assert P_lin.size == P_nw.size
        except AssertionError:
            print('P shapes are', P_lin.shape, P_nw.shape)
        Sigma_r = 15  # Mpc/h
        if self.reconstructed:
            Sigma_perp = 2.5  # Mpc/h
            Sigma_para = 4  # Mpc/h
        else:
            Sigma_perp = 6  # Mpc/h
            Sigma_para = 10  # Mpc/h
        if self.force_isotropy:
            Sigma_perp = Sigma_para = np.mean([Sigma_perp, Sigma_para])
            beta = 0
            print('Forcing beta = 0')
        sigmavsq = (1 - np.square(mu))*np.square(Sigma_perp)/2 \
            + np.square(mu*Sigma_para)/2
        S = np.exp(-np.square(k*Sigma_r)/2)
        C = (1 + np.square(mu) * beta * (1-S)) / \
            (1 + np.square(k*mu*Sigma_s)/2)
        self.P_k_mu = np.square(C) * (
            (P_lin - P_nw) * np.exp(-np.square(k)*sigmavsq) + P_nw)
        return self.P_k_mu

    def p_multipole(self, ell):

        Ln = legendre(ell)
        P_ell = (2*ell + 1) / 2 * np.sum(
            self.P_k_mu * (Ln(-self.mu) + Ln(self.mu)) * np.diff(self.mu_bins),
            axis=1)
        return P_ell

    def xi_multipole(self, r_input, ell, a=0.34, r_damp=True):
        '''
        integrate in logk space using trapozoidal rule
        a is the exponential damping parameter to suppress high-k oscillations
        usually 0.3 to 1
        input r may be 2D of dim (n_r_bins, n_mu_bins)
        '''
        P_ell = self.p_multipole(ell)
        P_ell = (P_ell[1:] + P_ell[:-1])/2  # interpolate midpoint
        dk = np.diff(self.k)
        lnk_edges = np.log(self.k)
        lnk = (lnk_edges[1:] + lnk_edges[:-1])/2  # lnk value at midpoint
        k = np.exp(lnk)  # k value at midpoint
        # turn into 2D grids
        k = np.tile(k, (r_input.size, 1))
        P_ell = np.tile(P_ell, (r_input.size, 1))
        dk = np.tile(dk, (r_input.size, 1))
        r = np.tile(r_input.flatten(), (k.shape[1], 1)).T
        assert k.shape == P_ell.shape == dk.shape == r.shape
        if r_damp:
            damp = np.exp(-r*np.square(k*a))
        else:
            damp = np.exp(-np.square(k*a))
        xi_ell = np.power(1j, ell) / (2*np.square(np.pi)) * np.sum(
            np.square(k) * P_ell * spherical_jn(ell, k*r) * damp * dk,
            axis=1)
        return np.real(xi_ell.reshape(r_input.shape))


# %% unit test
if __name__ == '__main__':
    r = np.arange(10, 300, 1)
    n_mu_bins = 1000
    path_p_lin = '/mnt/gosling2/bigsim_products/emulator_1100box_planck_products/emulator_1100box_planck_00-0_products/emulator_1100box_planck_00-0_power/info/camb_matterpower.dat'
    data = np.loadtxt(path_p_lin)
    k = data[:, 0]
    P_lin = data[:, 1]
    power = PowerTemplate(z=0.5)
    P = power.P(k, P_lin, n_mu_bins)
    xi_0 = power.xi_multipole(r, ell=0)
    xi_2 = power.xi_multipole(r, ell=2)
    xi_4 = power.xi_multipole(r, ell=4)
