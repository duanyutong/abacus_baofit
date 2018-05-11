from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
from scipy.optimize import minimize
import pickle
import sys

# setup to run the data in the Ross_2016_COMBINEDDR12 folder
rmin = 50
rmax = 150 # the minimum and maximum scales to be used in the fit
rbmax = 80 # the maximum scale to be used to set the bias prior
Hdir = '/home/dyt/analysis_data/emulator_1100box_planck-mgrav/' # this is the save directory; must contain "2Dbaofits" folder
# datadir = '/home/dyt/analysis_data/emulator_1100box_planck/emulator_1100box_planck_00-combined/z0.7/' # where the xi data are
# ft = 'zheng07' # common prefix of all data files cinlduing last '_'
zb = '' # zb = 'zbin3_' # change number to change zbin
binc = '' # binc = 0 # change number to change bin center
bs = 5. # the r bin size of the data
bc = '.txt' # bc = 'post_recon_bincent'+str(binc)+'.dat' # common ending string of data files
# fout = ft

def P2(mu):
    return 0.5*(3.*mu**2.-1.)
    
def P4(mu):
    return 0.125*(35.*mu**4.-30.*mu**2.+3.)

def findPolya(H,ci,d):
    ht = H.transpose()
    onei = np.linalg.pinv(np.dot(np.dot(H,ci),ht))
    comb = np.dot(np.dot(onei,H),ci)
    return np.dot(comb,d)

class baofit3D_ellFull_1cov:
    def __init__(self,dv,ic,mod,rl):
        self.xim = dv
        self.rl = rl
        m2 = 1.
        self.nbin = len(self.rl)
        
        # print('nbin is: ', self.nbin)
        # print('xim is: ', self.xim)
        # print('r list is: ', self.rl)
        self.invt = ic
        if self.nbin != len(self.invt):
            return 'vector matrix mismatch!'

        self.ximodmin = 10.

        self.x0 = [] #empty lists to be filled for model templates
        self.x2 = []
        self.x4 = []
        self.x0sm = []
        self.x2sm = []
        self.x4sm = []
        
        # change current working directory to file directory
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        mf0 = open('BAOtemplates/xi0'+mod).readlines()
        mf2 = open('BAOtemplates/xi2'+mod).readlines()
        mf4 = open('BAOtemplates/xi4'+mod).readlines()
        mf0sm = open('BAOtemplates/xi0sm'+mod).readlines()
        mf2sm = open('BAOtemplates/xi2sm'+mod).readlines()
        mf4sm = open('BAOtemplates/xi4sm'+mod).readlines()
        for i in range(0,len(mf0)):
            ln0 = mf0[i].split()
            self.x0.append(2.1*(float(ln0[1])))
            self.x0sm.append(2.1*float(mf0sm[i].split()[1]))
            ln2 = mf2[i].split()
            m2 = 2.1*(float(ln2[1]))
            self.x2.append(m2)
            self.x2sm.append(2.1*float(mf2sm[i].split()[1]))
            ln4 = mf4[i].split()
            m4 = 2.1*(float(ln4[1]))
            self.x4.append(m4)
            self.x4sm.append(2.1*float(mf4sm[i].split()[1]))
        self.at = 1.
        self.ar = 1.
        self.b0 = 1.
        self.b2 = 1.
        self.b4 = 1.
        self.H = np.zeros((6,self.nbin))
        for i in range(0,self.nbin):
            if i < self.nbin/2:
                self.H[0][i] = 1.
                self.H[1][i] = 1./self.rl[i]
                self.H[2][i] = 1./self.rl[i]**2.
            if i >= self.nbin/2:
                self.H[3][i] = 1.
                self.H[4][i] = 1./self.rl[i]
                self.H[5][i] = 1./self.rl[i]**2.
        
    def wmod(self,r,sp=1.):
        self.sp = sp
        sum1 = 0
        sum2 = 0
        nmu = 100
        dmu = 1./float(nmu)
        for i in range(nmu):
            mu = i*dmu+dmu/2.
            al = np.sqrt(mu**2.*self.ar**2.+(1.-mu**2.)*self.at**2.)
            mup = mu*self.ar/al
            rp = r*al
            ximu = self.lininterp(self.x0,rp)+P2(mup)*self.lininterp(self.x2,rp)+P4(mup)*self.lininterp(self.x4,rp)
            sum1 += ximu
            sum2 += mu**2.*ximu
        return dmu*sum1, 1.5*dmu*sum2 

    def wmodsm(self,r,sp=1.):
        self.sp = sp
        sum1 = 0
        sum2 = 0
        nmu = 100
        dmu = 1./float(nmu)
        for i in range(0,nmu):
            mu = i*dmu+dmu/2.
            al = np.sqrt(mu**2.*self.ar**2.+(1.-mu**2.)*self.at**2.)
            mup = mu*self.ar/al
            rp = r*al
            ximu = self.lininterp(self.x0sm,rp)+P2(mup)*self.lininterp(self.x2sm,rp)+P4(mup)*self.lininterp(self.x4sm,rp)
            sum1 += ximu
            sum2 += mu**2.*ximu
        return dmu*sum1, 1.5*dmu*sum2

    def wmodW(self,r,sp=1.):
        self.sp = sp
        sum1 = 0
        sum2 = 0
        nmu = 100
        dmu = 1./float(nmu)
        nspl = int(nmu*self.Wsp)
        for i in range(0,nspl):
            mu = i*dmu+dmu/2.
            al = np.sqrt(mu**2.*self.ar**2.+(1.-mu**2.)*self.at**2.)
            mup = mu*self.ar/al
            rp = r*al
            ximu = self.lininterp(self.x0,rp)+P2(mup)*self.lininterp(self.x2,rp)+P4(mup)*self.lininterp(self.x4,rp)
            sum1 += ximu
        for i in range(nspl,nmu):
            mu = i*dmu+dmu/2.
            al = np.sqrt(mu**2.*self.ar**2.+(1.-mu**2.)*self.at**2.)
            mup = mu*self.ar/al
            rp = r*al
            ximu = self.lininterp(self.x0,rp)+P2(mup)*self.lininterp(self.x2,rp)+P4(mup)*self.lininterp(self.x4,rp)
            sum2 += ximu
        n0 = 1./float(self.Wsp)
        n2 = 1./(1.-self.Wsp)
        return dmu*sum1*n0, dmu*sum2*n2

    def lininterp(self,f,r):
        indd = int((r-self.ximodmin)/self.sp)
        indu = indd + 1
        fac = (r-self.ximodmin)/self.sp-indd
        if fac > 1.:
            print('ERROR: BAD FAC in wmod')
            return 'ERROR'
        if indu >= len(f)-1:
            return 0
        return f[indu]*fac+(1.-fac)*f[indd] 

    def mkxi(self):
        self.xia = []
        for i in range(0, int(len(self.rl)/2)):
            xi0,xi2 = self.wmod(self.rl[i])
            self.xia.append(xi0)
        for i in range(int(len(self.rl)/2),self.nbin):
            xi0,xi2 = self.wmod(self.rl[i])
            self.xia.append(xi2)
        return True 

    def mkxism(self):
        self.xiasm = []
        for i in range(0,int(len(self.rl)/2)):
            xi0,xi2 = self.wmodsm(self.rl[i])
            self.xiasm.append(xi0)
        for i in range(int(len(self.rl)/2),self.nbin):
            xi0,xi2 = self.wmodsm(self.rl[i])
            self.xiasm.append(xi2)
        return True 

    def mkxiW(self):
        self.xi0a = []
        self.xi2a = []
        for i in range(0,len(self.rl)):
            xi0,xi2 = self.wmodW(self.rl[i])
            self.xi0a.append(xi0)
            self.xi2a.append(xi2)
        return True 
        
    def chi_templ_alphfXX(self, parameter_list ,wo='n',fw='',v='n'):

        BB = parameter_list[0]
        if BB < 0:
            return 1000
        A0 = parameter_list[1]
        A1 = parameter_list[2]
        A2 = parameter_list[3]
        Beta = parameter_list[4]
        if Beta < 0:
            return 1000
        A02 = parameter_list[5]
        A12 = parameter_list[6]
        A22 = parameter_list[7]
        modl = []
        if wo == 'y':
            fo = open('ximod'+fw+'.dat','w')
        for i in range(0, int(self.nbin/2)):
            r = self.rl[i]
            mod0 = BB*self.xia[i]+A0+A1/r+A2/r**2.
            modl.append(mod0)
            if wo == 'y':
                fo.write(str(self.rl[i])+' '+str(mod0)+'\n')
        
        for i in range(int(self.nbin/2),self.nbin):
            r = self.rl[i]
            mod2 = 5.*(Beta*self.xia[i]-BB*0.5*self.xia[int(i-self.nbin/2)])+A02+A12/r+A22/r**2.
            if wo == 'y':
                fo.write(str(self.rl[i])+' '+str(mod2)+'\n')            
            modl.append(mod2)
        if wo == 'y':
            fo.close()      

        dl = []
        for i in range(0,self.nbin):
            dl.append(self.xim[i]-modl[i])
        chit = np.dot(np.dot(dl,self.invt),dl)
        if v == 'y':
            print('dl, chit: ', dl,chit)
        BBfac = (np.log(BB/self.BB)/self.Bp)**2.
        Btfac = (np.log(Beta/self.B0)/self.Bt)**2.
        return chit+BBfac+Btfac

    def chi_templ_alphfXX_an(self, parameter_list, wo='n',fw='',v='n'):

        BB = parameter_list[0]
        if BB < 0:
            return 1000
        Beta = parameter_list[1]
        if Beta < 0:
            return 1000
        modl = []
        if wo == 'y':
            fo = open(Hdir+'2Dbaofits/'+fw+'-ep0-ximod.dat','w')
        pv = []
        for i in range(0, int(self.nbin/2)):
            pv.append(self.xim[i]-BB*self.xia[i])
        for i in range(int(self.nbin/2), self.nbin):
            pv.append(self.xim[i]-(5.*(Beta*self.xia[i]-BB*0.5*self.xia[int(i-self.nbin/2)])))
         
        Al = findPolya(self.H,self.invt,pv)
        A0,A1,A2,A02,A12,A22 = Al[0],Al[1],Al[2],Al[3],Al[4],Al[5]
        for i in range(0, int(self.nbin/2)):
            r = self.rl[i]
            mod0 = BB*self.xia[i]+A0+A1/r+A2/r**2.
            modl.append(mod0)
            if wo == 'y':
                mod0sm = BB*self.xiasm[i]+A0+A1/r+A2/r**2.
                fo.write(str(self.rl[i])+' '+str(mod0)+' '+str(mod0sm)+'\n')
        
        for i in range(int(self.nbin/2),self.nbin):
            r = self.rl[i]
            mod2 = 5.*(Beta*self.xia[i]-BB*0.5*self.xia[int(i-self.nbin/2)])+A02+A12/r+A22/r**2.
            if wo == 'y':
                mod2sm = 5.*(Beta*self.xiasm[i]-BB*0.5*self.xiasm[int(i-self.nbin/2)])+A02+A12/r+A22/r**2.
                fo.write(str(self.rl[i])+' '+str(mod2)+' '+str(mod2sm)+'\n')            
            modl.append(mod2)
        if wo == 'y':
            fo.close()

        dl = []
        for i in range(0,self.nbin):
            dl.append(self.xim[i]-modl[i])  
        chit = np.dot(np.dot(dl,self.invt),dl)
        if v == 'y':
            print('dl, chit: ', dl,chit)
        BBfac = (np.log(BB/self.BB)/self.Bp)**2.
        Btfac = (np.log(Beta/self.B0)/self.Bt)**2.
        return chit + BBfac + Btfac

def sigreg_2dme(file, spar=0.006, spat=0.003, amin=0.8, amax=1.2):
    # find the confidence region from the chi2 grid found in the module below
    f = open(file+'.dat').readlines()
    sumt = 0
    nb1 = int((amax-amin)/spar)   
    nb2 = int((amax-amin)/spat)
    pl1 = []
    pl2 = []
    flar = []
    flat = []
    for i in range(0,nb1):
        pl1.append(0)
    for i in range(0,nb2):  
        pl2.append(0)
    for i in range(0,nb1):
        flar.append(amin+i*spar+spar/2.)
    for i in range(0,nb2):
        flat.append(amin+i*spat+spat/2.)
    pmax = 0
    chimin = 1000
    corr = 0
    for i in range(0,len(f)):
        ln = f[i].split()
        if len(ln) == 3:
            chi = float(ln[2])
            if chi < chimin:
                chimin = chi
            p = np.exp(-0.5*chi)
            a1 = float(ln[0])
            a2 = float(ln[1])
            if p > pmax:
                pmax = p
            corr += a1*a2*p
            ind1 = int((a1-amin)/spar)
            pl1[ind1] += p      
            ind2 = int((a2-amin)/spat)
            pl2[ind2] += p
            sumt += p
    corr = corr/sumt
    sum0 = 0
    sig1 = 0.682
    sig2 = 0.95
    min2 = (1.-sig2)/2.
    min1 = (1.-sig1)/2.
    max1 = min1+sig1
    max2 = min2+sig2
    pmax = 0
    s = 0
    ofn = 0
    sumf = 0
    sumar = 0
    sumat = 0
    for i in range(len(pl1)):
        sumar += pl1[i]
    for i in range(len(pl2)):
        sumat += pl2[i]
    for i in range(len(flar)):
        fn = flar[i]
        od = sum0
        pb = pl1[i]/sumar
        sum0 += pb
        sumf += fn*pb
        if pb > pmax:
            pmax = pb
        d = sum0 
        if d > min2 and s == 0:
            abs1 = abs(od-min2)
            abs2 = abs(d-min2)
            s =1
        if sum0 > min1 and s == 1:
            abs1 = abs(od-min1)
            abs2 = abs(d-min1)
            fn1d = (ofn/abs1+fn/abs2)/(1./abs1+1./abs2)
            s =2
        if sum0 > max1 and s == 2:
            abs1 = abs(od-max1)
            abs2 = abs(d-max1)
            fn1u = (ofn/abs1+fn/abs2)/(1./abs1+1./abs2)
            s =3
        if sum0 > max2 and s == 3:
            abs1 = abs(od-max2)
            abs2 = abs(d-max2)
            s =4
        ofn = fn

    a1b = sumf/sum0
    err1 = (fn1u-fn1d)/2.       

    sum0 = 0
    sig1 = 0.682
    sig2 = 0.95
    min2 = (1.-sig2)/2.
    min1 = (1.-sig1)/2.
    max1 = min1 + sig1
    max2 = min2 + sig2
    pmax = 0
    s = 0
    sumf = 0
    for i in range(0,len(flat)):
        fn = flat[i]
        od = sum0
        pb = pl2[i]/sumat
        sum0 += pb
        sumf += fn*pb
        if pb > pmax:
            pmax = pb
        d = sum0
        if d > min2 and s == 0:
            abs1 = abs(od-min2)
            abs2 = abs(d-min2)
            s = 1
        if sum0 > min1 and s == 1:
            abs1 = abs(od-min1)
            abs2 = abs(d-min1)
            fn1d = (ofn/abs1+fn/abs2)/(1./abs1+1./abs2)
            s = 2
        if sum0 > max1 and s == 2:
            abs1 = abs(od-max1)
            abs2 = abs(d-max1)
            fn1u = (ofn/abs1+fn/abs2)/(1./abs1+1./abs2)
            s = 3
        if sum0 > max2 and s == 3:
            abs1 = abs(od-max2)
            abs2 = abs(d-max2)
            s = 4
        ofn = fn

    a2b = sumf/sum0
    err2 = (fn1u-fn1d)/2.       
    corr = corr - a1b*a2b
    return a1b, err1, a2b, err2, chimin, corr, corr/(err1*err2)

def Xism_arat_1C_an(dv,icov,rl,mod,dvb,icovb,rlb,
                    B0=1., spar=0.006, spat=0.003, 
                    amin=0.8,amax=1.2,nobao='n',Bp=.4,Bt=.4,meth='Nelder-Mead',
                    fout=''):

    # print('try meth = "Nelder-Mead" if does not work or answer is weird')
    bb = baofit3D_ellFull_1cov(dvb,icovb,mod,rlb) #initialize for bias prior
    b = baofit3D_ellFull_1cov(dv,icov,mod,rl) #initialize for fitting
    b.B0 = B0
    b.Bt = Bt
    
    bb.Bp = 100.
    bb.BB = 1.
    bb.B0 = B0
    bb.Bt = 100.
    bb.mkxi()
    b.bins = bs
    
    B = 0.1
    BB = None
    chiBmin = 2000
    while B < 2.:
        chiB = bb.chi_templ_alphfXX((B,0,0,0,1.,0,0,0))
        if chiB < chiBmin:
            chiBmin = chiB
            BB = B
        B += .01
    if BB == None:
        print('ChiB >= ChiBmin for B in [0.1, 2)')
    
#    print('BB, chiBmin:', BB,chiBmin)
    b.BB = BB
    b.B0 = BB       
    b.Bp= Bp
    b.Bt = Bt
    if not os.path.exists(os.path.join(Hdir+'2Dbaofits')):
        print('2Dbaofits DNE. Creating folder')
        os.makedirs(os.path.join(Hdir+'2Dbaofits'))
    fo = open(Hdir+'2Dbaofits/'+fout+'-arat-covchi.dat','w+')
    fg = open(Hdir+'2Dbaofits/'+fout+'-arat-covchigrid.dat','w+')
    chim = 1000
    nar = int((amax-amin)/spar)
    nat = int((amax-amin)/spat)
    grid = np.zeros((nar,nat))
    fac = (998.-float(len(dv)))/999.

    for i in range(nar):      
        b.ar = amin+spar*i+spar/2.
#        print('b.ar: ', b.ar)
        for j in range(nat):
            b.at = amin+spat*j+spat/2.
            b.mkxi()
            inl = (B,B0)
            (B, B0) = minimize(b.chi_templ_alphfXX_an, inl, 
                                method = meth, options = {'disp': False}).x
            chi = b.chi_templ_alphfXX_an((B,B0))*fac
            grid[i][j] = chi
            fo.write(str(b.ar)+' '+str(b.at)+' '+str(chi)+'\n')
            fg.write(str(chi)+' ')
            if chi < chim:
#                print('alpha_r, alpha_t, chisq: ', b.ar,b.at,chi)
                chim = chi
                alrm = b.ar
                altm = b.at
                Bm = B
                Betam = B0
        fg.write('\n')
    b.ar = alrm
    b.at = altm 
    b.mkxi()
    b.mkxism()
    chi = b.chi_templ_alphfXX_an((Bm,Betam),wo='y',fw=fout) # writes out best-fit model
#    print('alpha_r, alpha_t, chisq at minimum: ', alrm,altm,chim) # alphlk,likm
    alph = (alrm*altm**2.)**(1/3.)
    b.ar = alph
    b.at = alph 
    b.mkxi()
    b.mkxism()
    chi = b.chi_templ_alphfXX_an((Bm,Betam),wo='y',fw=fout)
    fo.close()
    fg.close()
#    ans = sigreg_2dme(Hdir+'2Dbaofits/arat-'+fout+'-covchi',spar=spar,spat=spat)
    return alrm, altm, chim


def baofit(inputs):
# def baofit():

    path_xi_0, path_xi_2, path_cov, fout_tag = inputs
    # read in data
    # c = np.loadtxt(datadir+ft+zb+'cross-xi_monoquad-cov'+bc)  # combined monopole, quadrupole cov matrix
    # r_bins_centre = np.loadtxt(datadir+ft+zb+'auto-xi_0-coadd'+bc)[:, 0]
    # d0 = np.loadtxt(datadir+ft+zb+'auto-xi_0-coadd'+bc)[:, 1]
    # d2 = np.loadtxt(datadir+ft+zb+'auto-xi_2-coadd'+bc)[:, 1]

    c = np.loadtxt(path_cov)  # combined monopole, quadrupole cov matrix
    r_bins_centre = np.loadtxt(path_xi_0)[:, 0]
    d0 = np.loadtxt(path_xi_0)[:, 1]
    d2 = np.loadtxt(path_xi_2)[:, 1]

    if len(c) != len(d0)*2: print('MISMATCHED data and cov matrix!')
    if len(d0) != len(d2): print('0 and 2 components of xi mismatch')
    # create data vectors in given range
    r_mask = (rmin < r_bins_centre) & (r_bins_centre < rmax)
    rl = np.hstack([r_bins_centre[r_mask], r_bins_centre[r_mask]]) * 1.000396 #factor to correct for pairs should have slightly larger average pair distance than the bin center
    dv = np.hstack([d0[r_mask], d2[r_mask]])
    # dv_len = int(dv.size/2) # length of selected data vector for ell = 0 and 2
    rb_mask = r_mask & (r_bins_centre < rbmax)
    rlb = np.hstack([r_bins_centre[rb_mask], r_bins_centre[rb_mask]]) * 1.000396
    dvb = np.hstack([d0[rb_mask], d2[rb_mask]])
    # dvb_len = int(dvb.size/2)
    print('{} - length of data vector (0 and 2 components): {}'.format(fout_tag, len(dv)))
    # create cov matrix for data, must keep the cov between mono and quadrupole
    ri_mask = np.tile(r_mask, (len(r_mask), 1)).transpose()
    rj_mask = np.tile(r_mask, (len(r_mask), 1))
    quadrant_mask = ri_mask & rj_mask
    covm_mask = np.tile(quadrant_mask, (2,2))
    covm = c[covm_mask].reshape(dv.size, dv.size)
    invc = np.linalg.pinv(covm) #the inverse covariance matrix to pass to the code
    # covm for bias
    rbi_mask = np.tile(rb_mask, (len(rb_mask), 1)).transpose()
    rbj_mask = np.tile(rb_mask, (len(rb_mask), 1))
    quadrant_mask = rbi_mask & rbj_mask
    covmb_mask = np.tile(quadrant_mask, (2,2))
    covmb = c[covmb_mask].reshape(dvb.size, dvb.size)
    invc = np.linalg.pinv(covm) #the inverse covariance matrix to pass to the code
    invcb = np.linalg.pinv(covmb)
    # define template
    mod = 'Challenge_matterpower0.44.02.54.015.01.0.dat'  #BAO template used     
    alrm, altm, chim = Xism_arat_1C_an(dv, invc, rl, mod, dvb, invcb, rlb, amin=1.005, amax=1.035, spar=0.0004, spat=0.0004, fout = fout_tag)
    print('{} - alpha_r, alpha_t, chisq at minimum: {}, {}, {}'.format(fout_tag, alrm, altm, chim))
#    with open(os.path.join(Hdir, '2Dbaofits', fout_tag + '-confidence_region.pkl'), 'wb') as handle:
#        pickle.dump(ans, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return alrm, altm


if __name__ == '__main__':

    # baofit()
    baofit(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])