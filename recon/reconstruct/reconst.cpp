/* reconst.cpp: Ryuichiro Hada, November 2016

This reads input particles files (filename and filename2) arranged by "read.cpp",
reconstructs mass distribution, and creats "reconstructed" input particles files
(filename and filename2) to be used in "fftcorr.cpp" written by Daniel.

*/

/* ======================= Compile-time user flags ================= */

//#define RSD             // To take account of RSD in reconstruction

//#define XCORR_SHIFT          // To get X-correlation for shift fields
  //#define FROM_DENSITY       // To get shift_t_X from the density field
  //#define DIFF_SHIFT          // To sum the square of difference between shift_t_X and reconstructed shift
//#define XCORR_DENSITY           // To get X-correlation for density fields
//#define RECONST         // To reconsturuct (standard by default)
  //#define ITERATION       // To do the iterative reconstruction
    //#define CHANGE_SM       // To change smoothing scale
    //#define SECOND        // To take account of 2nd order in reconstruction

//#define INITIAL         // To compute the corr for the initial density correlation function

//#define NEAREST_CELL    // To turn-off triangular CIC, e.g., for comparing to python
//#define WAVELET         // To turn on a D12 wavelet density assignment
//#define SLAB            // Handle the arrays by associating threads to x-slabs
//#define FFTSLAB         // Do the FFTs in 2D, then 1D
//#define OPENMP          // Turn on the OPENMP items

//#define DEPRECATED      // not to use merge_sort_omp

/* ======================= Preamble ================= */
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <complex>
#include <vector>
#include <algorithm>
#include <string>
#include <iostream>
#include <sstream>
#include <cstring>
#include <sys/time.h>
#include <fftw3.h>
#include "STimer.cc"

#include <iomanip>
#include <complex>
#include <cmath>

using namespace std;


// For multi-threading:
#ifdef OPENMP
    #include <omp.h>
#else
    // Just fake these for the single-threaded case, to make code easier to read.
    int omp_get_max_threads() { return 1; }
    int omp_get_num_threads() { return 1; }
    int omp_get_thread_num() { return 0; }
#endif

#define PAGE 4096     // To force some memory alignment; unit is bytes.

// In principle, this gives flexibility, but one also would need to adjust the FFTW calls.
typedef double Float;
typedef std::complex<double> Complex;

// We want to allow that the FFT grid could exceed 2^31 cells
typedef unsigned long long int uint64;

STimer IO, Setup, FFTW, Correlate, YlmTime, Total, CIC, Misc,
       FFTonly, Hist, Extract, AtimesB, Init, FFTyz, FFTx,
       Reconstruct, FS, Copy, fqts, mu, Out;


#include "d12.cpp"



//******* Linear factor *********

Float lgf, lgr_f, lgr_beta, shift_2nd;

// Habble(z)^2 / Habble(0)^2
Float H_H0_2(Float Om_0, Float Ol_0, Float z) {
    Float A;
      A = Om_0*pow((z+1.0),3.0) + Ol_0 + (1.0 - Om_0 - Ol_0)*pow((z+1.0),2.0);
    return A;
}

// Omega_m depending on redshift
Float Omf(Float Om_0, Float Ol_0, Float z) {
    Float A;
      A = Om_0/H_H0_2(Om_0, Ol_0, z)*pow((z+1.0),3.0);
    return A;
}

// Omega_l depending on redshift
Float Olf(Float Om_0, Float Ol_0, Float z) {
    Float A;
      A = Ol_0/H_H0_2(Om_0, Ol_0, z);
    return A;
}

//  Liniear growth factor
Float lg_factor(Float Om_0, Float Ol_0, Float z) {
    Float A;
    A = 5.0/2.0*(1.0/(1.0 + z))*Omf(Om_0,Ol_0,z)/(pow(Omf(Om_0,Ol_0,z),(4.0/7.0)) - Olf(Om_0,Ol_0,z) + (1.0 + Omf(Om_0,Ol_0,z)/2.0)*(1.0 + Olf(Om_0,Ol_0,z)/70.0));
    return A;
}

//  Liniear growth rate
Float lg_rate(Float Om_0, Float Ol_0, Float z) {
    Float A;
    if((Om_0 + Ol_0) == 1.0){
      A = pow(Omf(Om_0,Ol_0,z),(4.0/7.0)) + (1.0 - Omf(Om_0,Ol_0,z)/2.0*(1.0 + Omf(Om_0,Ol_0,z)))/70.0;
    }
    else{
      A = pow(Omf(Om_0,Ol_0,z),(4.0/7.0));
    }
    return A;
}



//******* Structure *********

//  structure for header
typedef struct
{
    Float posmin[3];
    Float posmax[3];
    Float maxcep;
    Float blank8;
} struct_header;

//  structure for data output  //
typedef struct
{
    Float x[3];
    Float w;
} struct_out_x3w;



/* ========================================================================= */

class Histogram {
  // This should set up the binning and the space to hold the answers
  public:
    int maxell;
    Float sep;
    int nbins;

    Float *cnt;
    Float *cnt_w1;
    Float *cnt_w2;
    Float *cnt_w3;
    Float *hist;
    Float *hist_w1;
    Float *hist_w2;
    Float *hist_w3;
    Float binsize;
    Float zerolag;    // The value at zero lag

    Histogram(int _maxell, Float _sep, Float _dsep) {
        int err;
        maxell = _maxell;
        sep = _sep;
        binsize = _dsep;
        zerolag = -12345.0;
        nbins = floor(sep/binsize);
        assert(nbins>0&&nbins<1e6);  // Too big probably means data entry error.

        // Allocate cnt[nbins], hist[maxell/2+1, nbins]
        err=posix_memalign((void **) &cnt, PAGE, sizeof(Float)*nbins); assert(err==0);
        err=posix_memalign((void **) &cnt_w1, PAGE, sizeof(Float)*nbins); assert(err==0);
        err=posix_memalign((void **) &cnt_w2, PAGE, sizeof(Float)*nbins); assert(err==0);
        err=posix_memalign((void **) &cnt_w3, PAGE, sizeof(Float)*nbins); assert(err==0);
        err=posix_memalign((void **) &hist, PAGE, sizeof(Float)*nbins*(maxell/2+1)); assert(err==0);
        err=posix_memalign((void **) &hist_w1, PAGE, sizeof(Float)*nbins*(maxell/2+1)); assert(err==0);
        err=posix_memalign((void **) &hist_w2, PAGE, sizeof(Float)*nbins*(maxell/2+1)); assert(err==0);
        err=posix_memalign((void **) &hist_w3, PAGE, sizeof(Float)*nbins*(maxell/2+1)); assert(err==0);
        assert(cnt!=NULL);
        assert(cnt_w1!=NULL);
        assert(cnt_w2!=NULL);
        assert(cnt_w3!=NULL);
        assert(hist!=NULL);
        assert(hist_w1!=NULL);
        assert(hist_w2!=NULL);
        assert(hist_w3!=NULL);
    }
    ~Histogram() {
        // For some reason, these cause a crash!  Weird!
        // free(hist);
        // free(cnt);
    }

    // TODO: Might consider creating more flexible ways to select a binning.
    inline int r2bin(Float r) {
        return floor(r/binsize);
    }

    void histcorr(int ell, int n, Float *rnorm, Float *r_znorm, Float *total) {
        // Histogram into bins by rnorm[n], adding up weighting by total[n].
        // Add to multipole ell.
        if (ell==0) {
            for (int j=0; j<nbins; j++) cnt[j] = 0.0;
            for (int j=0; j<nbins; j++) cnt_w1[j] = 0.0;
            for (int j=0; j<nbins; j++) cnt_w2[j] = 0.0;
            for (int j=0; j<nbins; j++) cnt_w3[j] = 0.0;
            for (int j=0; j<nbins; j++) hist[j] = 0.0;
            for (int j=0; j<nbins; j++) hist_w1[j] = 0.0;
            for (int j=0; j<nbins; j++) hist_w2[j] = 0.0;
            for (int j=0; j<nbins; j++) hist_w3[j] = 0.0;
            for (int j=0; j<n; j++) {
                int b = r2bin(rnorm[j]);
                if (rnorm[j]<binsize*1e-6) {
                    zerolag = total[j];
                }
                if (b>=nbins||b<0) continue;
                cnt[b]++;
                hist[b]+= total[j];
                //fprintf(stderr," %16.9e\n", r_znorm[j]/rnorm[j]);
                if (r_znorm != NULL){
                  if (r_znorm[j]/rnorm[j] > 2.0/3.0){
                    cnt_w1[b]++;
                    hist_w1[b]+= total[j];
                  }
                  else if (r_znorm[j]/rnorm[j] > 1.0/3.0){
                    cnt_w2[b]++;
                    hist_w2[b]+= total[j];
                  }
                  else{
                    cnt_w3[b]++;
                    hist_w3[b]+= total[j];
                  }
                }
            }
        } else {
            // ell>0
            Float *h = hist+ell/2*nbins;
            for (int j=0; j<nbins; j++) h[j] = 0.0;
            for (int j=0; j<n; j++) {
                int b = r2bin(rnorm[j]);
                if (b>=nbins||b<0) continue;
                h[b]+= total[j];
            }
        }
    }

    Float sum() {
         // Add up the histogram values for ell=0
         Float total=0.0;
         for (int j=0; j<nbins; j++) total += hist[j];
         return total;
    }

    void print(FILE *fp, int norm) {
        // Print out the results
        // If norm==1, divide by counts
        Float denom;
        for (int j=0; j<nbins; j++) {
            fprintf(fp,"%1d ", norm);
            if (sep>2)
                fprintf(fp,"%6.2f %8.0f", (j+0.5)*binsize, cnt[j]);
            else
                fprintf(fp,"%7.4f %8.0f", (j+0.5)*binsize, cnt[j]);
            if (cnt[j]!=0&&norm) denom = cnt[j]; else denom = 1.0;
            for (int k=0; k<=maxell/2; k++)
                fprintf(fp," %16.9e", hist[k*nbins+j]/denom);
            fprintf(fp,"\n");
        }
    }
};    // end Histogram

/* ======================================================================= */
// Here are some matrix handling routines, which may need OMP attention


#ifndef OPENMP
    #undef SLAB                // We probably don't want to do this if single threaded
#endif

#ifdef SLAB
    // Try treating the matrices explicitly by x slab;
    // this might allow NUMA memory to be closer to the socket running the thread.
    #define MY_SCHEDULE schedule(static,1)
    #define YLM_SCHEDULE schedule(static,1)
#else
    // Just treat the matrices as one big object
    #define MY_SCHEDULE schedule(dynamic,512)
    #define YLM_SCHEDULE schedule(dynamic,1)
#endif

void initialize_matrix(Float *&m, const uint64 size, const int nx) {
    // Initialize a matrix m and set it to zero.
    // We want to touch the whole matrix, because in NUMA this defines the association
    // of logical memory into the physical banks.
    // nx will be our slab decomposition; it must divide into size evenly
    // Warning: This will only allocate new space if m==NULL.  This allows
    // one to reuse space.  But(!!) there is no check that you've not changed
    // the size of the matrix -- you could overflow the previously allocated
    // space.
    assert (size%nx==0);
    Init.Start();
    if (m==NULL) {
        int err=posix_memalign((void **) &m, PAGE, sizeof(Float)*size+PAGE); assert(err==0);
    }
    assert(m!=NULL);
#ifdef SLAB
    const uint64 nyz = size/nx;
    #pragma omp parallel for MY_SCHEDULE
    for (int x=0; x<nx; x++) {
        Float *mslab = m+x*nyz;
        for (uint64 j=0; j<nyz; j++) mslab[j] = 0.0;
    }
#else
    #pragma omp parallel for MY_SCHEDULE
    for (uint64 j=0; j<size; j++) m[j] = 0.0;
#endif
    Init.Stop();
    return;
}

void initialize_matrix_by_copy(Float *&m, const uint64 size, const int nx, Float *copy) {
    // Initialize a matrix m and set it to copy[size].
    // nx will be our slab decomposition; it must divide into size evenly
    // Warning: This will only allocate new space if m==NULL.  This allows
    // one to reuse space.  But(!!) there is no check that you've not changed
    // the size of the matrix -- you could overflow the previously allocated
    // space.
    assert (size%nx==0);
    Init.Start();
    if (m==NULL) {
        int err=posix_memalign((void **) &m, PAGE, sizeof(Float)*size+PAGE); assert(err==0);
    }
    assert(m!=NULL);
#ifdef SLAB
    const uint64 nyz = size/nx;
    #pragma omp parallel for MY_SCHEDULE
    for (int x=0; x<nx; x++) {
        Float *mslab = m+x*nyz;
        Float *cslab = copy+x*nyz;
        for (uint64 j=0; j<nyz; j++) mslab[j] = cslab[j];
        // memcpy(mslab, cslab, sizeof(Float)*nyz);    // Slower for some reason!
    }
#else
    #pragma omp parallel for MY_SCHEDULE
    for (uint64 j=0; j<size; j++) m[j] = copy[j];
#endif
    Init.Stop();
    return;
}

void set_matrix(Float *a, const Float b, const uint64 size, const int nx) {
    // Set a equal to a scalar b
    // nx will be our slab decomposition; it must divide into size evenly
    assert (size%nx==0);
#ifdef SLAB
    const uint64 nyz = size/nx;
    #pragma omp parallel for MY_SCHEDULE
    for (int x=0; x<nx; x++) {
        Float *aslab = a+x*nyz;
        for (uint64 j=0; j<nyz; j++) aslab[j] = b;
    }
#else
    #pragma omp parallel for MY_SCHEDULE
    for (uint64 j=0; j<size; j++) a[j] = b;
#endif
}

void scale_matrix_const(Float *a, const Float b, const uint64 size, const int nx) {
    // Multiply a by a scalar b
    // nx will be our slab decomposition; it must divide into size evenly
    assert (size%nx==0);
#ifdef SLAB
    const uint64 nyz = size/nx;
    #pragma omp parallel for MY_SCHEDULE
    for (int x=0; x<nx; x++) {
        Float *aslab = a+x*nyz;
        for (uint64 j=0; j<nyz; j++) aslab[j] *= b;
    }
#else
    #pragma omp parallel for MY_SCHEDULE
    for (uint64 j=0; j<size; j++) a[j] *= b;
#endif
}

void scale_matrix(Float *a, Float *b, const uint64 size, const int nx) {
    // Multiply a by b
    // nx will be our slab decomposition; it must divide into size evenly
    assert (size%nx==0);
#ifdef SLAB
    const uint64 nyz = size/nx;
    #pragma omp parallel for MY_SCHEDULE
    for (int x=0; x<nx; x++) {
        Float *aslab = a+x*nyz;
        Float *bslab = b+x*nyz;
        for (uint64 j=0; j<nyz; j++) aslab[j] *= bslab[j];
    }
#else
    #pragma omp parallel for MY_SCHEDULE
    for (uint64 j=0; j<size; j++) a[j] *= b[j];
#endif
}

void devide_matrix(Float *a, Float *b, uint64 &zero_c, const uint64 size, const int nx) {
    // Devide a by b and put it on a
    // nx will be our slab decomposition; it must divide into size evenly
    assert (size%nx==0);
#ifdef SLAB
    const uint64 nyz = size/nx;
    #pragma omp parallel for MY_SCHEDULE reduction(+:zero_c)
    for (int x=0; x<nx; x++) {
        Float *aslab = a+x*nyz;
        Float *bslab = b+x*nyz;
        for (uint64 j=0; j<nyz; j++) {
            if(bslab[j] == 0.0) { bslab[j] = 1.0; zero_c++; }
            aslab[j] = aslab[j]/bslab[j];
        }
    }
#else
    #pragma omp parallel for MY_SCHEDULE reduction(+:zero_c)
    for (uint64 j=0; j<size; j++) {
        if(b[j] == 0.0) { b[j] = 1.0; zero_c++;  }
        a[j] = a[j]/b[j];
    }
#endif
}

void copy_matrix(Float *a, Float *b, const uint64 size, const int nx) {
    // Set a equal to a vector b
    // nx will be our slab decomposition; it must divide into size evenly
    assert (size%nx==0);
#ifdef SLAB
    const uint64 nyz = size/nx;
    #pragma omp parallel for MY_SCHEDULE
    for (int x=0; x<nx; x++) {
        Float *aslab = a+x*nyz;
        Float *bslab = b+x*nyz;
        for (uint64 j=0; j<nyz; j++) aslab[j] = bslab[j];
    }
#else
    #pragma omp parallel for MY_SCHEDULE
    for (uint64 j=0; j<size; j++) a[j] = b[j];
#endif
}

void copy_matrix(Float *a, Float *b, const Float c, const uint64 size, const int nx) {
    // Set a equal to a vector b times a scalar c
    // nx will be our slab decomposition; it must divide into size evenly
    assert (size%nx==0);
#ifdef SLAB
    const uint64 nyz = size/nx;
    #pragma omp parallel for MY_SCHEDULE
    for (int x=0; x<nx; x++) {
        Float *aslab = a+x*nyz;
        Float *bslab = b+x*nyz;
        for (uint64 j=0; j<nyz; j++) aslab[j] = bslab[j]*c;
    }
#else
    #pragma omp parallel for MY_SCHEDULE
    for (uint64 j=0; j<size; j++) a[j] = b[j]*c;
#endif
}

void add_matrix(Float *a, Float *b, const Float c, const uint64 size, const int nx) {
    // Set a equal to a vector a added with a vector b times a scalar c
    // nx will be our slab decomposition; it must divide into size evenly
    assert (size%nx==0);
#ifdef SLAB
    const uint64 nyz = size/nx;
    #pragma omp parallel for MY_SCHEDULE
    for (int x=0; x<nx; x++) {
        Float *aslab = a+x*nyz;
        Float *bslab = b+x*nyz;
        for (uint64 j=0; j<nyz; j++) aslab[j] += bslab[j]*c;
    }
#else
    #pragma omp parallel for MY_SCHEDULE
    for (uint64 j=0; j<size; j++) a[j] += b[j]*c;
#endif
}

void add_matrix_const(Float *a, const Float c, const uint64 size, const int nx) {
    // Set a equal to the vector a added with a scalar c
    // nx will be our slab decomposition; it must divide into size evenly
    assert (size%nx==0);
#ifdef SLAB
    const uint64 nyz = size/nx;
    #pragma omp parallel for MY_SCHEDULE
    for (int x=0; x<nx; x++) {
        Float *aslab = a+x*nyz;
        for (uint64 j=0; j<nyz; j++) aslab[j] += c;
    }
#else
    #pragma omp parallel for MY_SCHEDULE
    for (uint64 j=0; j<size; j++) a[j] += c;
#endif
}

void sub_matrix(Float *a, Float *b, const Float c, const uint64 size, const int nx) {
    // Set a equal to a vector a subtracted by a vector b times a scalar c
    // nx will be our slab decomposition; it must divide into size evenly
    assert (size%nx==0);
#ifdef SLAB
    const uint64 nyz = size/nx;
    #pragma omp parallel for MY_SCHEDULE
    for (int x=0; x<nx; x++) {
        Float *aslab = a+x*nyz;
        Float *bslab = b+x*nyz;
        for (uint64 j=0; j<nyz; j++) aslab[j] -= bslab[j]*c;
    }
#else
    #pragma omp parallel for MY_SCHEDULE
    for (uint64 j=0; j<size; j++) a[j] -= b[j]*c;
#endif
}

void mu_1_matrix(Float *a, Float *b_xx, Float *b_yy, Float *b_zz,
                 const uint64 size, const int nx) {
    // Calculate mu_1
    // nx will be our slab decomposition; it must divide into size evenly
    assert (size%nx==0);
#ifdef SLAB
    const uint64 nyz = size/nx;
    #pragma omp parallel for MY_SCHEDULE
    for (int x=0; x<nx; x++) {
        Float *aslab = a+x*nyz;
        Float *bslab_xx = b_xx+x*nyz;
        Float *bslab_yy = b_yy+x*nyz;
        Float *bslab_zz = b_zz+x*nyz;
        for (uint64 j=0; j<nyz; j++){
           aslab[j] = (bslab_xx[j] + bslab_yy[j] + bslab_zz[j]);
        }
    }
#else
    #pragma omp parallel for MY_SCHEDULE
    for (uint64 j=0; j<size; j++){
        a[j] = (b_xx[j] + b_yy[j] + b_zz[j]);
    }
#endif
}

void mu_2_matrix(Float *a, Float *b_xx, Float *b_xy, Float *b_xz,
                 Float *b_yx, Float *b_yy, Float *b_yz, Float *b_zx, Float *b_zy, Float *b_zz,
                 const uint64 size, const int nx) {
    // Calculate mu_2
    // nx will be our slab decomposition; it must divide into size evenly
    assert (size%nx==0);
#ifdef SLAB
    const uint64 nyz = size/nx;
    #pragma omp parallel for MY_SCHEDULE
    for (int x=0; x<nx; x++) {
        Float *aslab = a+x*nyz;
        Float *bslab_xx = b_xx+x*nyz;
        Float *bslab_xy = b_xy+x*nyz;
        Float *bslab_xz = b_xz+x*nyz;
        Float *bslab_yx = b_yx+x*nyz;
        Float *bslab_yy = b_yy+x*nyz;
        Float *bslab_yz = b_yz+x*nyz;
        Float *bslab_zx = b_zx+x*nyz;
        Float *bslab_zy = b_zy+x*nyz;
        Float *bslab_zz = b_zz+x*nyz;
        for (uint64 j=0; j<nyz; j++){
           aslab[j] = (bslab_xx[j]*bslab_yy[j] + bslab_yy[j]*bslab_zz[j] + bslab_zz[j]*bslab_xx[j]
                - bslab_xy[j]*bslab_yx[j] - bslab_yz[j]*bslab_zy[j] - bslab_zx[j]*bslab_xz[j]);
        }
    }
#else
    #pragma omp parallel for MY_SCHEDULE
    for (uint64 j=0; j<size; j++){
        a[j] = (b_xx[j]*b_yy[j] + b_yy[j]*b_zz[j] + b_zz[j]*b_xx[j]
             - b_xy[j]*b_xy[j] - b_yz[j]*b_yz[j] - b_zx[j]*b_zx[j]);
    }
#endif
}

void mu_3_matrix(Float *a, Float *b_xx, Float *b_xy, Float *b_xz,
                 Float *b_yx, Float *b_yy, Float *b_yz, Float *b_zx, Float *b_zy, Float *b_zz,
                 const uint64 size, const int nx) {
    // Calculate mu_3
    // nx will be our slab decomposition; it must divide into size evenly
    assert (size%nx==0);
#ifdef SLAB
    const uint64 nyz = size/nx;
    #pragma omp parallel for MY_SCHEDULE
    for (int x=0; x<nx; x++) {
        Float *aslab = a+x*nyz;
        Float *bslab_xx = b_xx+x*nyz;
        Float *bslab_xy = b_xy+x*nyz;
        Float *bslab_xz = b_xz+x*nyz;
        Float *bslab_yx = b_yx+x*nyz;
        Float *bslab_yy = b_yy+x*nyz;
        Float *bslab_yz = b_yz+x*nyz;
        Float *bslab_zx = b_zx+x*nyz;
        Float *bslab_zy = b_zy+x*nyz;
        Float *bslab_zz = b_zz+x*nyz;
        for (uint64 j=0; j<nyz; j++){
           aslab[j] = (bslab_xx[j]*bslab_yy[j]*bslab_zz[j] + bslab_xy[j]*bslab_yz[j]*bslab_zx[j]
                 + bslab_xz[j]*bslab_yx[j]*bslab_zy[j] - bslab_xz[j]*bslab_yy[j]*bslab_zx[j]
                 - bslab_xy[j]*bslab_yx[j]*bslab_zz[j] - bslab_xx[j]*bslab_yz[j]*bslab_zy[j]);
        }
    }
#else
    #pragma omp parallel for MY_SCHEDULE
    for (uint64 j=0; j<size; j++){
        a[j] = (b_xx[j]*b_yy[j]*b_zz[j] + b_xy[j]*b_yz[j]*b_zx[j]
              + b_xz[j]*b_yx[j]*b_zy[j] - b_xz[j]*b_yy[j]*b_zx[j]
              - b_xy[j]*b_yx[j]*b_zz[j] - b_xx[j]*b_yz[j]*b_zy[j]);
    }
#endif
}

Float sum_matrix(Float *a, const uint64 size, const int nx) {
    // Sum the elements of the matrix
    // nx will be our slab decomposition; it must divide into size evenly
    assert (size%nx==0);
    Float tot=0.0;
#ifdef SLAB
    const uint64 nyz = size/nx;
    #pragma omp parallel for MY_SCHEDULE reduction(+:tot)
    for (int x=0; x<nx; x++) {
        Float *aslab = a+x*nyz;
        for (uint64 j=0; j<nyz; j++) tot += aslab[j];
    }
#else
    #pragma omp parallel for MY_SCHEDULE reduction(+:tot)
    for (uint64 j=0; j<size; j++) tot += a[j];
#endif
    return tot;
}

Float sumsq_matrix(Float *a, const uint64 size, const int nx) {
    // Sum the square of elements of the matrix
    // nx will be our slab decomposition; it must divide into size evenly
    assert (size%nx==0);
    Float tot=0.0;
#ifdef SLAB
    const uint64 nyz = size/nx;
    #pragma omp parallel for MY_SCHEDULE reduction(+:tot)
    for (int x=0; x<nx; x++) {
        Float *aslab = a+x*nyz;
        for (uint64 j=0; j<nyz; j++) tot += aslab[j]*aslab[j];
    }
#else
    #pragma omp parallel for MY_SCHEDULE reduction(+:tot)
    for (uint64 j=0; j<size; j++) tot += a[j]*a[j];
#endif
    return tot;
}

Float sumsq_dif_matrix(Float *a, Float *b, const uint64 size, const int nx) {
    // Sum the square of difference between elements of the matrix a and b
    // nx will be our slab decomposition; it must divide into size evenly
    assert (size%nx==0);
    Float tot = 0.0;
#ifdef SLAB
    const uint64 nyz = size/nx;
    #pragma omp parallel for MY_SCHEDULE reduction(+:tot)
    for (int x=0; x<nx; x++) {
        Float *aslab = a+x*nyz;
        Float *bslab = b+x*nyz;
        for (uint64 j=0; j<nyz; j++) {
          tot += (aslab[j] - bslab[j])*(aslab[j] - bslab[j]);
        }
    }
#else
    #pragma omp parallel for MY_SCHEDULE reduction(+:tot)
    for (uint64 j=0; j<size; j++) {
      tot += (a[j] - b[j])*(a[j] - b[j]);
    }
#endif
    return tot;
}

uint64 count_matrix(Float *a, Float index_con, const uint64 size, const int nx) {
    // Count the number of elements of the matrix satisfing a condition
    // nx will be our slab decomposition; it must divide into size evenly
    assert (size%nx==0);
    uint64 tot = 0;
#ifdef SLAB
    const uint64 nyz = size/nx;
    #pragma omp parallel for MY_SCHEDULE reduction(+:tot)
    for (int x=0; x<nx; x++) {
        Float *aslab = a+x*nyz;
        for (uint64 j=0; j<nyz; j++) {
          if(aslab[j] > index_con && aslab[j] <= (index_con + 1.0)) tot += 1;
        }
    }
#else
    #pragma omp parallel for MY_SCHEDULE reduction(+:tot)
    for (uint64 j=0; j<size; j++) {
      if(aslab[j] > index_con && aslab[j] <= (index_con + 1.0)) tot += 1;
    }
#endif
    return tot;
}

void multiply_matrix_with_conjugation(Complex *a, Complex *b, const uint64 size, const int nx) {
    // Element-wise multiply a[] by conjugate of b[]
    // Note that size refers to the Complex size; the calling routine
    // is responsible for dividing the Float size by 2.
    // nx will be our slab decomposition; it must divide into size evenly
    assert (size%nx==0);
#ifdef SLAB
    const uint64 nyz = size/nx;
    #pragma omp parallel for MY_SCHEDULE
    for (int x=0; x<nx; x++) {
        Complex *aslab = a+x*nyz;
        Complex *bslab = b+x*nyz;
        for (uint64 j=0; j<nyz; j++) aslab[j] *= std::conj(bslab[j]);
    }
#else
    #pragma omp parallel for MY_SCHEDULE
    for (uint64 j=0; j<size; j++) a[j] *= std::conj(b[j]);
#endif
}

/* ==========================  Submatrix extraction =================== */

void extract_submatrix(Float *total, Float *corr, int csize[3],
                       Float *work, int ngrid[3], const int ngrid2) {
    // Given a large matrix work[ngrid^3],
    // extract out a submatrix of size csize^3, centered on work[0,0,0].
    // Multiply the result by corr[csize^3] and add it onto total[csize^3]
    // Again, zero lag is mapping to corr(csize/2, csize/2, csize/2),
    // but it is at (0,0,0) in the FFT grid.
    Extract.Start();
    int cx = csize[0]/2;    // This is the middle of the submatrix
    int cy = csize[1]/2;    // This is the middle of the submatrix
    int cz = csize[2]/2;    // This is the middle of the submatrix
    #pragma omp parallel for schedule(dynamic,1)
    for (uint64 i=0; i<csize[0]; i++) {
        uint64 ii = (ngrid[0]-cx+i)%ngrid[0];
        for (int j=0; j<csize[1]; j++) {
            uint64 jj = (ngrid[1]-cy+j)%ngrid[1];
            Float *t = total+(i*csize[1]+j)*csize[2];  // This is (i,j,0)
            Float *cc = corr+(i*csize[1]+j)*csize[2];  // This is (i,j,0)
            Float *Y =  work+(ii*ngrid[1]+jj)*ngrid2+ngrid[2]-cz;
                                        // This is (ii,jj,ngrid[2]-c)
            for (int k=0; k<cz; k++) t[k] += cc[k]*Y[k];
            Y =  work+(ii*ngrid[1]+jj)*ngrid2-cz;
                                        // This is (ii,jj,-c)
            for (int k=cz; k<csize[2]; k++) t[k] += cc[k]*Y[k];
        }
    }
    Extract.Stop();
}

void extract_submatrix_C2R(Float *total, Float *corr, int csize[3],
                       Complex *work, int ngrid[3], const int ngrid2) {
    // Given a large matrix work[ngrid^3/2],
    // extract out a submatrix of size csize^3, centered on work[0,0,0].
    // The input matrix is Complex * with the half-domain Fourier convention.
    // We are only summing the real part; the imaginary part always sums to zero.
    // Need to reflect the -z part around the origin, which also means reflecting x & y.
    // ngrid[2] and ngrid2 are given as their Float values, not yet divided by two.
    // Multiply the result by corr[csize^3] and add it onto total[csize^3]
    // Again, zero lag is mapping to corr(csize/2, csize/2, csize/2),
    // but it is at (0,0,0) in the FFT grid.
    Extract.Start();
    int cx = csize[0]/2;    // This is the middle of the submatrix
    int cy = csize[1]/2;    // This is the middle of the submatrix
    int cz = csize[2]/2;    // This is the middle of the submatrix
    #pragma omp parallel for schedule(dynamic,1)
    for (uint64 i=0; i<csize[0]; i++) {
        uint64 ii = (ngrid[0]-cx+i)%ngrid[0];
        uint64 iin = (ngrid[0]-ii)%ngrid[0];   // The reflected coord
        for (int j=0; j<csize[1]; j++) {
            uint64 jj = (ngrid[1]-cy+j)%ngrid[1];
            uint64 jjn = (ngrid[1]-jj)%ngrid[1];   // The reflected coord
            Float *t = total+(i*csize[1]+j)*csize[2];  // This is (i,j,0)
            Float *cc = corr+(i*csize[1]+j)*csize[2];  // This is (i,j,0)
            // The positive half-plane (inclusize)
            Complex *Y =  work+(ii*ngrid[1]+jj)*ngrid2/2-cz;
                                        // This is (ii,jj,-cz)
            for (int k=cz; k<csize[2]; k++) t[k] += cc[k]*std::real(Y[k]);
            // The negative half-plane (inclusize), reflected.
            // k=cz-1 should be +1, k=0 should be +cz
            Y =  work+(iin*ngrid[1]+jjn)*ngrid2/2+cz;
                                        // This is (iin,jjn,+cz)
            for (int k=0; k<cz; k++) t[k] += cc[k]*std::real(Y[-k]);
        }
    }
    Extract.Stop();
}

/* ==========================  For reconstruction =================== */

void densFFT_to_work(Complex *work, Complex *densFFT, int ngrid[3], const int ngrid2,
                         Float cell_size, Float sig_sm, int index_comp, Float bias) {
    // Given a large matrix densFFT[ngrid^3/2], make work[ngrid^3/2] of the same format.
    // The input matrix is Complex * with the half-domain Fourier convention.
    // ngrid[2] and ngrid2 are given as their Float values, not yet divided by two.
    // k =   2*pi/L*i for (i < ngrid[]/2+1)          (L = cell_size*ngrid[], i = index)
    // k = - 2*pi/L*(ngrid[] - i) for (i >= ngrid[]/2+1)
    // the number of z-component is only ngrid2/2 (= ngrid[2]/2+1).
    FS.Start();
    Float k_cep_size[3], k_com[3], k_norm, S;
    k_cep_size[0] = 2.0*M_PI/(cell_size*ngrid[0]);
    k_cep_size[1] = 2.0*M_PI/(cell_size*ngrid[1]);
    k_cep_size[2] = 2.0*M_PI/(cell_size*ngrid[2]);
    Complex pIm(0.0,1.0), coef;

    //#pragma omp parallel for schedule(static,1)
    for (uint64 i=0; i<ngrid[0]; i++) {
        // k_x
        if(i < (ngrid[0]/2+1))  {
          k_com[0] = k_cep_size[0]*(Float)i;
        }
        else{
          k_com[0] = - k_cep_size[0]*(Float)(ngrid[0]-i);
        }
        for (int j=0; j<ngrid[1]; j++) {
            // k_y
            if(j < (ngrid[1]/2+1))  {
              k_com[1] = k_cep_size[1]*(Float)j;
            }
            else{
              k_com[1] = - k_cep_size[1]*(Float)(ngrid[1]-j);
            }
            Complex *w, *Y;
            w = Y = NULL;
            w = work+(i*ngrid[1]+j)*ngrid2/2;        // This is (i,j,0)
            Y = densFFT+(i*ngrid[1]+j)*ngrid2/2;     // This is (i,j,0)
            for (int k=0; k<ngrid2/2; k++) {
              // k_z
              k_com[2] = k_cep_size[0]*(Float)k;
              k_norm = sqrt(k_com[0]*k_com[0]
                          + k_com[1]*k_com[1]
                          + k_com[2]*k_com[2]);   // k norm
              // smoothing function
              S = exp(-0.5*k_norm*k_norm*sig_sm*sig_sm);
              // incorporating the coefficient for RSD and bias into smooothing function
              S *= 1.0/(bias*(1.0 + lgr_beta*(k_com[2]/k_norm)*(k_com[2]/k_norm)));

              if(index_comp == 0){
                //  for shift_x
                if(k_norm == 0.0) coef = 0.0;
                else coef = pIm*k_com[0]/(k_norm*k_norm)*S;
                w[k] = coef*Y[k];
              }
              else if(index_comp == 1){
                //  for shift_y
                if(k_norm == 0.0) coef = 0.0;
                else coef = pIm*k_com[1]/(k_norm*k_norm)*S;
                w[k] = coef*Y[k];
              }
              else{
                //  for shift_z
                if(k_norm == 0.0) coef = 0.0;
                else coef = pIm*k_com[2]/(k_norm*k_norm)*S;
                w[k] = coef*Y[k];
              }
            }
        }
    }
    FS.Stop();
}

void densFFT_to_shift(Complex *work, Complex *densFFT, int ngrid[3], const int ngrid2,
                         Float cell_size, Float sig_sm, int index_comp, Float C_ani) {
    // Given a large matrix densFFT[ngrid^3/2], make work[ngrid^3/2] of the same format.
    // The input matrix is Complex * with the half-domain Fourier convention.
    // ngrid[2] and ngrid2 are given as their Float values, not yet divided by two.
    // k =   2*pi/L*i for (i < ngrid[]/2+1)          (L = cell_size*ngrid[], i = index)
    // k = - 2*pi/L*(ngrid[] - i) for (i >= ngrid[]/2+1)
    // the number of z-component is only ngrid2/2 (= ngrid[2]/2+1).
    FS.Start();
    Float k_cep_size[3], k_com[3], k_norm, S;
    k_cep_size[0] = 2.0*M_PI/(cell_size*ngrid[0]);
    k_cep_size[1] = 2.0*M_PI/(cell_size*ngrid[1]);
    k_cep_size[2] = 2.0*M_PI/(cell_size*ngrid[2]);
    Complex pIm(0.0,1.0), coef;

    //#pragma omp parallel for schedule(static,1)
    for (uint64 i=0; i<ngrid[0]; i++) {
        // k_x
        if(i < (ngrid[0]/2+1))  {
          k_com[0] = k_cep_size[0]*(Float)i;
        }
        else{
          k_com[0] = - k_cep_size[0]*(Float)(ngrid[0]-i);
        }
        for (int j=0; j<ngrid[1]; j++) {
            // k_y
            if(j < (ngrid[1]/2+1))  {
              k_com[1] = k_cep_size[1]*(Float)j;
            }
            else{
              k_com[1] = - k_cep_size[1]*(Float)(ngrid[1]-j);
            }
            Complex *w, *Y;
            w = Y = NULL;
            w = work+(i*ngrid[1]+j)*ngrid2/2;        // This is (i,j,0)
            Y = densFFT+(i*ngrid[1]+j)*ngrid2/2;     // This is (i,j,0)
            for (int k=0; k<ngrid2/2; k++) {
              // k_z
              k_com[2] = k_cep_size[0]*(Float)k;
              k_norm = sqrt(k_com[0]*k_com[0]
                          + k_com[1]*k_com[1]
                          + k_com[2]*k_com[2]);   // k norm
              // smoothing function
              S = exp(-0.5*((k_com[0]*k_com[0] + k_com[1]*k_com[1])*sig_sm*sig_sm
                        + k_com[2]*k_com[2]*sig_sm*sig_sm*(C_ani*C_ani)));
              if(index_comp == 0){
                //  for shift_x
                if(k_norm == 0.0) coef = 0.0;
                else coef = pIm*k_com[0]/(k_norm*k_norm)*S;
                w[k] = coef*Y[k];
              }
              else if(index_comp == 1){
                //  for shift_y
                if(k_norm == 0.0) coef = 0.0;
                else coef = pIm*k_com[1]/(k_norm*k_norm)*S;
                w[k] = coef*Y[k];
              }
              else{
                //  for shift_z
                if(k_norm == 0.0) coef = 0.0;
                else coef = pIm*k_com[2]/(k_norm*k_norm)*S;
                w[k] = coef*Y[k];
              }
            }
        }
    }
    FS.Stop();
}

void densFFT_to_shift_1d(Complex *work, Complex *densFFT, int ngrid[3], const int ngrid2,
                         Float cell_size, Float sig_sm, int index_comp_r, int index_comp_c, Float C_ani) {
    // Given a large matrix densFFT[ngrid^3/2], make work[ngrid^3/2] of the same format.
    // The input matrix is Complex * with the half-domain Fourier convention.
    // ngrid[2] and ngrid2 are given as their Float values, not yet divided by two.
    // k =   2*pi/L*i for (i < ngrid[]/2+1)          (L = cell_size*ngrid[], i = index)
    // k = - 2*pi/L*(ngrid[] - i) for (i >= ngrid[]/2+1)
    // the number of z-component is only ngrid2/2 (= ngrid[2]/2+1).
    FS.Start();
    Float k_cep_size[3], k_com[3], k_norm, S;
    k_cep_size[0] = 2.0*M_PI/(cell_size*ngrid[0]);
    k_cep_size[1] = 2.0*M_PI/(cell_size*ngrid[1]);
    k_cep_size[2] = 2.0*M_PI/(cell_size*ngrid[2]);
    Complex pIm(0.0,1.0), coef;

    //#pragma omp parallel for schedule(static,1)
    for (uint64 i=0; i<ngrid[0]; i++) {
        // k_x
        if(i < (ngrid[0]/2+1))  {
          k_com[0] = k_cep_size[0]*(Float)i;
        }
        else{
          k_com[0] = - k_cep_size[0]*(Float)(ngrid[0]-i);
        }
        for (int j=0; j<ngrid[1]; j++) {
            // k_y
            if(j < (ngrid[1]/2+1))  {
              k_com[1] = k_cep_size[1]*(Float)j;
            }
            else{
              k_com[1] = - k_cep_size[1]*(Float)(ngrid[1]-j);
            }
            Complex *w, *Y;
            w = Y = NULL;
            w = work+(i*ngrid[1]+j)*ngrid2/2;        // This is (i,j,0)
            Y = densFFT+(i*ngrid[1]+j)*ngrid2/2;     // This is (i,j,0)
            for (int k=0; k<ngrid2/2; k++) {
              // k_z
              k_com[2] = k_cep_size[0]*(Float)k;
              k_norm = sqrt(k_com[0]*k_com[0]
                          + k_com[1]*k_com[1]
                          + k_com[2]*k_com[2]);   // k norm
              // smoothing function
              S = exp(-0.5*((k_com[0]*k_com[0] + k_com[1]*k_com[1])*sig_sm*sig_sm
                        + k_com[2]*k_com[2]*sig_sm*sig_sm*(C_ani*C_ani)));
              if(index_comp_r == 0){
                //  for shift_x
                if(k_norm == 0.0) coef = 0.0;
                else{
                  if(index_comp_c == 0) coef = - k_com[0]*k_com[0]/(k_norm*k_norm)*S;
                  else if(index_comp_c == 1) coef = - k_com[0]*k_com[1]/(k_norm*k_norm)*S;
                  else coef = - k_com[0]*k_com[2]/(k_norm*k_norm)*S;
                }
                w[k] = coef*Y[k];
              }
              else if(index_comp_r == 1){
                //  for shift_y
                if(k_norm == 0.0) coef = 0.0;
                else{
                  if(index_comp_c == 0) coef = - k_com[1]*k_com[0]/(k_norm*k_norm)*S;
                  else if(index_comp_c == 1) coef = - k_com[1]*k_com[1]/(k_norm*k_norm)*S;
                  else coef = - k_com[1]*k_com[2]/(k_norm*k_norm)*S;
                }
                w[k] = coef*Y[k];
              }
              else{
                //  for shift_z
                if(k_norm == 0.0) coef = 0.0;
                else{
                  if(index_comp_c == 0) coef = - k_com[2]*k_com[0]/(k_norm*k_norm)*S;
                  else if(index_comp_c == 1) coef = - k_com[2]*k_com[1]/(k_norm*k_norm)*S;
                  else coef = - k_com[2]*k_com[2]/(k_norm*k_norm)*S;
                }
                w[k] = coef*Y[k];
              }
            }
        }
    }
    FS.Stop();
}


/* ===============================  FFTW wrapper routines ===================== */

void setup_FFTW(fftw_plan &fft,  fftw_plan &fftYZ,  fftw_plan &fftX,
                fftw_plan &ifft, fftw_plan &ifftYZ, fftw_plan &ifftX,
                int ngrid[3], const int ngrid2, Float *work) {
    // Setup the FFTW plans, possibly from disk, and save the wisdom
    fprintf(stdout,"# Planning the FFTs..."); fflush(NULL);
    FFTW.Start();
    FILE *fp = NULL;
    #ifdef OPENMP
        #ifndef FFTSLAB
            { int errval = fftw_init_threads(); assert(errval); }
            fftw_plan_with_nthreads(omp_get_max_threads());
        #endif
        #define WISDOMFILE "wisdom_fftw_omp"
    #else
        #define WISDOMFILE "wisdom_fftw"
    #endif
    #ifdef FFTSLAB
        #undef WISDOMFILE
        #define WISDOMFILE "wisdom_fftw"
    #endif
    fp = fopen(WISDOMFILE, "r");
    if (fp!=NULL) {
        fprintf(stdout,"Reading %s...", WISDOMFILE); fflush(NULL);
        fftw_import_wisdom_from_file(fp);
        fclose(fp);
    }

    #ifndef FFTSLAB
        // The following interface should work even if ngrid2 was 'non-minimal',
        // as might be desired by padding.
        int nfft[3], nfftc[3];
        nfft[0] = nfftc[0] = ngrid[0];
        nfft[1] = nfftc[1] = ngrid[1];
        nfft[2] = ngrid2;   // Since ngrid2 is always even, this will trick
        nfftc[2]= nfft[2]/2;
                // FFTW to assume ngrid2/2 Complex numbers in the result, while
                // fulfilling that nfft[2]>=ngrid[2].
        fft = fftw_plan_many_dft_r2c(3, ngrid, 1,
                work, nfft, 1, 0,
                (fftw_complex *)work, nfftc, 1, 0,
                FFTW_MEASURE);
        ifft = fftw_plan_many_dft_c2r(3, ngrid, 1,
                (fftw_complex *)work, nfftc, 1, 0,
                work, nfft, 1, 0,
                FFTW_MEASURE);

    /*        // The original interface, which only works if ngrid2 is tightly packed.
        fft = fftw_plan_dft_r2c_3d(ngrid[0], ngrid[1], ngrid[2],
                        work, (fftw_complex *)work, FFTW_MEASURE);
        ifft = fftw_plan_dft_c2r_3d(ngrid[0], ngrid[1], ngrid[2],
                        (fftw_complex *)work, work, FFTW_MEASURE);
    */

    #else
        // If we wanted to split into 2D and 1D by hand (and therefore handle the OMP
        // aspects ourselves), then we need to have two plans each.
        int nfft2[2], nfft2c[2];
        nfft2[0] = nfft2c[0] = ngrid[1];
        nfft2[1] = ngrid2;   // Since ngrid2 is always even, this will trick
        nfft2c[1]= nfft2[1]/2;
        int ngridYZ[2];
        ngridYZ[0] = ngrid[1];
        ngridYZ[1] = ngrid[2];
        fftYZ = fftw_plan_many_dft_r2c(2, ngridYZ, 1,
                work, nfft2, 1, 0,
                (fftw_complex *)work, nfft2c, 1, 0,
                FFTW_MEASURE);
        ifftYZ = fftw_plan_many_dft_c2r(2, ngridYZ, 1,
                (fftw_complex *)work, nfft2c, 1, 0,
                work, nfft2, 1, 0,
                FFTW_MEASURE);

        // After we've done the 2D r2c FFT, we have to do the 1D c2c transform.
        // We'll plan to parallelize over Y, so that we're doing (ngrid[2]/2+1)
        // 1D FFTs at a time.
        // Elements in the X direction are separated by ngrid[1]*ngrid2/2 complex numbers.
        int ngridX = ngrid[0];
        fftX = fftw_plan_many_dft(1, &ngridX, (ngrid[2]/2+1),
                (fftw_complex *)work, NULL, ngrid[1]*ngrid2/2, 1,
                (fftw_complex *)work, NULL, ngrid[1]*ngrid2/2, 1,
                -1, FFTW_MEASURE);
        ifftX = fftw_plan_many_dft(1, &ngridX, (ngrid[2]/2+1),
                (fftw_complex *)work, NULL, ngrid[1]*ngrid2/2, 1,
                (fftw_complex *)work, NULL, ngrid[1]*ngrid2/2, 1,
                +1, FFTW_MEASURE);
    #endif

    fp = fopen(WISDOMFILE, "w");
    assert(fp!=NULL);
    fftw_export_wisdom_to_file(fp);
    fclose(fp);
    fprintf(stdout,"Done!\n"); fflush(NULL);
    FFTW.Stop();
    return;
}


void free_FFTW(fftw_plan &fft,  fftw_plan &fftYZ,  fftw_plan &fftX,
               fftw_plan &ifft, fftw_plan &ifftYZ, fftw_plan &ifftX) {
    // Call all of the FFTW destroy routines
    #ifndef FFTSLAB
        fftw_destroy_plan(fft);
        fftw_destroy_plan(ifft);
    #else
        fftw_destroy_plan(fftYZ);
        fftw_destroy_plan(fftX);
        fftw_destroy_plan(ifftYZ);
        fftw_destroy_plan(ifftX);
    #endif
    #ifdef OPENMP
        #ifndef FFTSLAB
            fftw_cleanup_threads();
        #endif
    #endif
    return;
}

void FFT_Execute(fftw_plan fft, fftw_plan fftYZ, fftw_plan fftX,
            int ngrid[3], const int ngrid2, Float *work) {
    // Note that if FFTSLAB is not set, then the *work input is ignored!
    // Routine will use the array that was called for setup!
    // TODO: Might fix this behavior, but note alignment issues!
    FFTonly.Start();
    #ifndef FFTSLAB
        fftw_execute(fft);
    #else
        FFTyz.Start();
        // Then need to call this for every slab.  Can OMP these lines
        #pragma omp parallel for MY_SCHEDULE
        for (uint64 x=0; x<ngrid[0]; x++)
            fftw_execute_dft_r2c(fftYZ, work+x*ngrid[1]*ngrid2,
                      (fftw_complex *)work+x*ngrid[1]*ngrid2/2);
        FFTyz.Stop();
        FFTx.Start();
        #pragma omp parallel for schedule(dynamic,1)
        for (uint64 y=0; y<ngrid[1]; y++)
            fftw_execute_dft(fftX, (fftw_complex *)work+y*ngrid2/2,
                                       (fftw_complex *)work+y*ngrid2/2);
        FFTx.Stop();
    #endif
    FFTonly.Stop();
}

void IFFT_Execute(fftw_plan ifft, fftw_plan ifftYZ, fftw_plan ifftX,
            int ngrid[3], const int ngrid2, Float *work) {
    // Note that if FFTSLAB is not set, then the *work input is ignored!
    // Routine will use the array that was called for setup!
    // TODO: Might fix this behavior, but note alignment issues!
    FFTonly.Start();
    #ifndef FFTSLAB
        fftw_execute(ifft);
    #else
        FFTx.Start();
        // Then need to call this for every slab.  Can OMP these lines
        #pragma omp parallel for schedule(dynamic,1)
        for (uint64 y=0; y<ngrid[1]; y++)
            fftw_execute_dft(ifftX, (fftw_complex *)work+y*ngrid2/2,
                                       (fftw_complex *)work+y*ngrid2/2);
        FFTx.Stop();
        FFTyz.Start();
        #pragma omp parallel for MY_SCHEDULE
        for (uint64 x=0; x<ngrid[0]; x++)
            fftw_execute_dft_c2r(ifftYZ,
                      (fftw_complex *)work+x*ngrid[1]*ngrid2/2,
                                  work+x*ngrid[1]*ngrid2);
        FFTyz.Stop();
    #endif
    FFTonly.Stop();
}

/* ======================================================================== */

// A very simple class to contain the input objects
class Galaxy {
  public:
    Float x, y, z, w;
    uint64 index;
    Galaxy(Float a[4], uint64 i) { x = a[0]; y = a[1]; z = a[2]; w = a[3]; index = i; return; }
    ~Galaxy() { }
    // We'll want to be able to sort in 'x' order
    // bool operator < (const Galaxy& str) const { return (x < str.x); }
    // Sort in cell order
    bool operator < (const Galaxy& str) const { return (index < str.index); }
};

#include "merge_sort_omp.cpp"

/* ======================================================================== */

class Grid {
  public:
    // Inputs
    int ngrid[3];     // We might prefer a non-cubic box.  The cells are always cubic!
    Float max_sep;     // How much separation has already been built in.
    Float posmin[3];   // Including the border; we don't support periodic wrapping in CIC
    Float posmax[3];   // Including the border; we don't support periodic wrapping in CIC

    // Items to be computed
    Float posrange[3];    // The range of the padded box
    Float cell_size;      // The size of the cubic cells
    Float origin[3];      // The location of the origin in grid units.
    Float *xcell, *ycell, *zcell;   // The cell centers, relative to the origin

    // Storage for the r-space submatrices
    Float sep;                // The range of separations we'll be histogramming
    int csize[3];        // How many cells we must extract as a submatrix to do the histogramming.
    int csize3;                // The number of submatrix cells
    Float *cx_cell, *cy_cell, *cz_cell;   // The cell centers, relative to zero lag.
    Float *rnorm;        // The radius of each cell, in a flattened submatrix.

    // Storage for the k-space submatrices
    Float k_Nyq;          // The Nyquist frequency for our grid.
    Float kmax;           // The maximum wavenumber we'll use
    int ksize[3];         // How many cells we must extract as a submatrix to do the histogramming.
    int ksize3;           // The number of submatrix cells
    Float *kx_cell, *ky_cell, *kz_cell;    // The cell centers, relative to zero lag.
    Float *knorm;         // The wavenumber of each cell, in a flattened submatrix.
    Float *k_znorm;       // The z-wavenumber of each cell, in a flattened submatrix.
    Float *CICwindow;     // The inverse of the window function for the CIC cell assignment

    // The big grids
    int ngrid2;           // ngrid[2] padded out for the FFT work
    int ite_num;          // The number of iteration
    uint64 ngrid3;        // The total number of FFT grid cells
    uint64 galsize_out;   // The number of galaxies output
    uint64 num_over[5];   // The number of grid per number of overlap for Data
    uint64 num_weight[5]; // The number of grid per sum of weight for Data
    Float *dens;          // The density field, in a flattened grid
    Float *densFFT;       // The FFT of the density field, in a flattened grid.
    Float *work;          // Work space, in a flattened grid.
    Float *dens_tem;      // Temporary space, in a flattened grid.
    Float *dens_tem_2;      // Temporary space 2, in a flattened grid.
    Float *dens_s;        // The "smoothed" density field, in a flattened grid.
    Float *dens_now;      // The "current" estimate of density field, in a flattened grid.
    Float *dens_res;      // The "residual" density field, in a flattened grid.
    Float *dens_res_now;  // The "current" estimate of "residual" density field, in a flattened grid.
    Float *shift_x;       // Work space to obtain displacements (total) for each component, in a flattened grid.
    Float *shift_y;
    Float *shift_z;
    Float *shift_r_z;     // Work space to obtain displacements (real-space) for z-component, in a flattened grid.
    Float *shift_x_l;     // Last one
    Float *shift_y_l;
    Float *shift_z_l;
    Float *shift_r_z_l;
    Float *shift_xx;      // Work space to obtain derivatives of displacements for each component,
    Float *shift_xy;      //           in a flattened grid.
    Float *shift_xz;
    Float *shift_yx;
    Float *shift_yy;
    Float *shift_yz;
    Float *shift_zx;
    Float *shift_zy;
    Float *shift_zz;
    Float *shift_r_zx;
    Float *shift_r_zy;
    Float *shift_r_zz;
    Float *shift_t_x;     // Work space to obtain displacements (total) for each component, in a flattened grid.
    Float *shift_t_y;
    Float *shift_t_z;
    Float *C_O;           // Count the number of shift overlap

    uint64 cnt;                // The number of galaxies read in.
    Float Pshot;        // The sum of squares of the weights, which is the shot noise for P_0.

    uint64 zero_c;          // Count the number of "zero" grids

    Float sumsq_dif_x;      // Sum the square of difference between shifts
    Float sumsq_dif_y;
    Float sumsq_dif_z;

    // header: definition for structure
    struct_header header;
    // output: definition for structure
    struct_out_x3w *x_w;

    // outout filenames
    char* out_D;
    char* out_R;

    // Positions need to arrive in a coordinate system that has the observer at the origin

    ~Grid() {
        if (dens!=NULL) free(dens);
        if (densFFT!=NULL) free(densFFT);
        if (work!=NULL) free(work);
        if (dens_s!=NULL) free(dens_s);
        if (dens_now!=NULL) free(dens_now);
        if (dens_tem!=NULL) free(dens_tem);
        if (dens_tem_2!=NULL) free(dens_tem_2);
        if (dens_res!=NULL) free(dens_res);
        if (dens_res_now!=NULL) free(dens_res_now);
        if (shift_x!=NULL) free(shift_x);
        if (shift_y!=NULL) free(shift_y);
        if (shift_z!=NULL) free(shift_z);
        if (shift_r_z!=NULL) free(shift_r_z);
        if (shift_x_l!=NULL) free(shift_x_l);
        if (shift_y_l!=NULL) free(shift_y_l);
        if (shift_z_l!=NULL) free(shift_z_l);
        if (shift_r_z_l!=NULL) free(shift_r_z_l);
        if (shift_xx!=NULL) free(shift_xx);
        if (shift_xy!=NULL) free(shift_xy);
        if (shift_xz!=NULL) free(shift_xz);
        if (shift_yx!=NULL) free(shift_yx);
        if (shift_yy!=NULL) free(shift_yy);
        if (shift_yz!=NULL) free(shift_yz);
        if (shift_zx!=NULL) free(shift_zx);
        if (shift_zy!=NULL) free(shift_zy);
        if (shift_zz!=NULL) free(shift_zz);
        if (shift_r_zx!=NULL) free(shift_r_zx);
        if (shift_r_zy!=NULL) free(shift_r_zy);
        if (shift_r_zz!=NULL) free(shift_r_zz);
        if (shift_t_x!=NULL) free(shift_t_x);
        if (shift_t_y!=NULL) free(shift_t_y);
        if (shift_t_z!=NULL) free(shift_t_z);
        if (C_O!=NULL) free(C_O);
        if (x_w!=NULL) free(x_w);
        free(zcell);
        free(ycell);
        free(xcell);
        free(rnorm);
        free(cx_cell);
        free(cy_cell);
        free(cz_cell);
        free(knorm);
        free(k_znorm);
        free(kx_cell);
        free(ky_cell);
        free(kz_cell);
        free(CICwindow);
        // *densFFT and *work are freed in the reconstruct() routine.
    }

    Grid(const char filename[], int _ngrid[3], Float _cell, Float _sep, int qperiodic) {
        // This constructor is rather elaborate, but we're going to do most of the setup.
        // filename and filename2 are the input particles.
        // filename2==NULL will skip that one.
        // _sep is used here simply to adjust the box size if needed.
        // qperiodic flag will configure for periodic BC

        // Have to set these to null so that the initialization will work.
        dens = densFFT = work = dens_s = dens_now = dens_tem = dens_tem_2 = dens_res = dens_res_now = NULL;
        shift_x = shift_y = shift_z = shift_r_z
          = shift_x_l = shift_y_l = shift_z_l = shift_r_z_l
          = shift_xx = shift_xy = shift_xz
          = shift_yx = shift_yy = shift_yz = shift_zx = shift_zy = shift_zz
          = shift_r_zx = shift_r_zy = shift_r_zz
          = shift_t_x = shift_t_y = shift_t_z
          = C_O = NULL;
        x_w = NULL;
        rnorm = knorm = k_znorm = CICwindow = NULL;

        // Open a binary input file
        Setup.Start();
        FILE *fp = fopen(filename, "rb");
        assert(fp!=NULL);

        for (int j=0; j<3; j++) ngrid[j] = _ngrid[j];
        assert(ngrid[0]>0&&ngrid[0]<1e4);
        assert(ngrid[1]>0&&ngrid[1]<1e4);
        assert(ngrid[2]>0&&ngrid[2]<1e4);

        Float TOOBIG = 1e10;
        // This header is 64 bytes long.
        // Read posmin[3], posmax[3], max_sep, blank8;
        double tmp[4];
        int nread;
        nread=fread(tmp, sizeof(double), 3, fp); assert(nread==3);
        for (int j=0; j<3; j++) { posmin[j]=tmp[j]; assert(fabs(posmin[0])<TOOBIG); }
        nread=fread(tmp, sizeof(double), 3, fp); assert(nread==3);
        for (int j=0; j<3; j++) { posmax[j]=tmp[j]; assert(fabs(posmax[0])<TOOBIG); }
        nread=fread(tmp, sizeof(double), 1, fp); assert(nread==1);
        max_sep = tmp[0]; assert(max_sep>=0&&max_sep<TOOBIG);
        nread=fread(tmp, sizeof(double), 1, fp); assert(nread==1); // Not used, just for alignment
        fclose(fp);

        // If we're going to permute the axes, change here and in add_particles_to_grid().
        // The answers should be unchanged under permutation
        // std::swap(posmin[0], posmin[1]); std::swap(posmax[0], posmax[1]);
        // std::swap(posmin[2], posmin[1]); std::swap(posmax[2], posmax[1]);

        // If the user wants periodic BC, then we can ignore separation issues.
        if (qperiodic) max_sep = (posmax[0]-posmin[0])*100.0;

        // If the user asked for a larger separation than what was planned in the
        // input positions, then we can accomodate.  Add the extra padding to posrange;
        // don't change posmin, since that changes grid registration.
        Float extra_pad = 0.0;
        if (_sep>max_sep) {
            extra_pad = _sep-max_sep;
            max_sep = _sep;
        }
        sep = -1;   // Just as a test that setup() got run

        // Compute the box size required in each direction
        for (int j=0; j<3; j++) {
            posmax[j] += extra_pad;
            posrange[j]=posmax[j]-posmin[j];
            assert(posrange[j]>0.0);
        }

        if (qperiodic||_cell<=0) {
            // We need to compute the cell size
            // We've been given 3 ngrid and we have the bounding box.
            // Need to pick the most conservative choice
            // This is always required in the periodic case
            cell_size = std::max(posrange[0]/ngrid[0],
                            std::max(posrange[1]/ngrid[1], posrange[2]/ngrid[2]));
        } else {
            // We've been given a cell size and a grid.  Need to assure it is ok.
            cell_size = _cell;
            assert(cell_size*ngrid[0]>posrange[0]);
            assert(cell_size*ngrid[1]>posrange[1]);
            assert(cell_size*ngrid[2]>posrange[2]);
        }

        fprintf(stdout, "# Reading file %s.  max_sep=%f\n", filename, max_sep);
        fprintf(stdout, "# Adopting cell_size=%f for ngrid=%d, %d, %d\n",
                cell_size, ngrid[0], ngrid[1], ngrid[2]);
        fprintf(stdout, "# Adopted boxsize: %6.1f %6.1f %6.1f\n",
                cell_size*ngrid[0], cell_size*ngrid[1], cell_size*ngrid[2]);
        fprintf(stdout, "# Input pos range: %6.1f %6.1f %6.1f\n",
                posrange[0], posrange[1], posrange[2]);
        fprintf(stdout, "# Minimum ngrid=%d, %d, %d\n", int(ceil(posrange[0]/cell_size)),
                int(ceil(posrange[1]/cell_size)), int(ceil(posrange[2]/cell_size)));

        // ngrid2 pads out the array for the in-place FFT.
        // The default 3d FFTW format must have the following:
        ngrid2 = (ngrid[2]/2+1)*2;  // For the in-place FFT
        #ifdef FFTSLAB
            // That said, the rest of the code should work even extra space is used.
            // Some operations will blindly apply to the pad cells, but that's ok.
            // In particular, we might consider having ngrid2 be evenly divisible by
            // the critical alignment stride (32 bytes for AVX, but might be more for cache lines)
            // or even by a full PAGE for NUMA memory.  Doing this *will* force a more
            // complicated FFT, but at least for the NUMA case this is desired: we want
            // to force the 2D FFT to run on its socket, and only have the last 1D FFT
            // crossing sockets.  Re-using FFTW plans requires the consistent memory alignment.
            #define FFT_ALIGN 16
            // This is in units of Floats.  16 doubles is 1024 bits.
            ngrid2 = FFT_ALIGN*(ngrid2/FFT_ALIGN+1);
        #endif
        assert(ngrid2%2==0);
        fprintf(stdout, "# Using ngrid2=%d for FFT r2c padding\n", ngrid2);
        ngrid3 = (uint64)ngrid[0]*ngrid[1]*ngrid2;

        // Convert origin to grid units
        if (qperiodic) {
            // In this case, we'll place the observer centered in the grid, but
            // then displaced far away in the -x direction
            for (int j=0;j<3;j++) origin[j] = ngrid[j]/2.0;
            origin[2] -= ngrid[2]*1e4;        // Observer far away!
        } else {
            for (int j=0;j<3;j++) origin[j] = (0.0-posmin[j])/cell_size;
        }


        // Allocate xcell, ycell, zcell to [ngrid]
        int err;
        err=posix_memalign((void **) &xcell, PAGE, sizeof(Float)*ngrid[0]+PAGE); assert(err==0);
        err=posix_memalign((void **) &ycell, PAGE, sizeof(Float)*ngrid[1]+PAGE); assert(err==0);
        err=posix_memalign((void **) &zcell, PAGE, sizeof(Float)*ngrid[2]+PAGE); assert(err==0);
        assert(xcell!=NULL); assert(ycell!=NULL); assert(zcell!=NULL);
        // Now set up the cell centers relative to the origin, in grid units
        for (int j=0; j<ngrid[0]; j++) xcell[j] = 0.5+j-origin[0];
        for (int j=0; j<ngrid[1]; j++) ycell[j] = 0.5+j-origin[1];
        for (int j=0; j<ngrid[2]; j++) zcell[j] = 0.5+j-origin[2];
        Setup.Stop();

        // Allocate dens to [ngrid**2*ngrid2] and set it to zero
          initialize_matrix(dens, ngrid3, ngrid[0]);
          initialize_matrix(dens_now, ngrid3, ngrid[0]);

        #ifdef XCORR_SHIFT
          // Allocate shift_t_xyz to [ngrid**2*ngrid2] and set it to zero
          initialize_matrix(shift_t_x, ngrid3, ngrid[0]);
          initialize_matrix(shift_t_y, ngrid3, ngrid[0]);
          initialize_matrix(shift_t_z, ngrid3, ngrid[0]);
          // Allocate shift_xyz to [ngrid**2*ngrid2] and set it to zero
          initialize_matrix(shift_x, ngrid3, ngrid[0]);
          initialize_matrix(shift_y, ngrid3, ngrid[0]);
          initialize_matrix(shift_z, ngrid3, ngrid[0]);

          initialize_matrix(shift_xx, ngrid3, ngrid[0]);
          initialize_matrix(shift_xy, ngrid3, ngrid[0]);
          initialize_matrix(shift_xz, ngrid3, ngrid[0]);

          initialize_matrix(shift_yx, ngrid3, ngrid[0]);
          initialize_matrix(shift_yy, ngrid3, ngrid[0]);
          initialize_matrix(shift_yz, ngrid3, ngrid[0]);

          zero_c = 0;
          sumsq_dif_x = 0;
          sumsq_dif_y = 0;
          sumsq_dif_z = 0;
        #endif

        return;
    }


/* ------------------    Read galaxies (and output reconstructed ones)     ----------------------- */

    void read_galaxies(const char filename[], const char filename2[], const char filename3[], int output, int type) {
        // Read to the end of the file, bringing in x,y,z,w points.
        // Bin them onto the grid.
        // We're setting up a large buffer to read in the galaxies.
        // Will reset the buffer periodically, just to limit the size.
        double tmp[8];
        cnt = 0;
        uint64 index;
        Float totw = 0.0, totwsq = 0.0;
        // Set up a small buffer, just to reduce the calls to fread, which seem to be slow
        // on some machines.
        #define BUFFERSIZE 512
        double buffer[BUFFERSIZE], *b;
        #define MAXGAL 1000000
        std::vector<Galaxy> gal;
        gal.reserve(MAXGAL);    // Just to cut down on thrashing; it will expand as needed
        // if output=1, constructor x_w needs memory as same as gal.
        if(output){
            if (x_w==NULL) {
                int err=posix_memalign((void **) &x_w, PAGE, sizeof(Float)*4*8*MAXGAL+PAGE); assert(err==0);
            }
            assert(x_w!=NULL);
        }
        IO.Start();
        for (int file=0; file<3; file++) {
            char *fn, *fno;
            uint64 thiscnt=0;
            if (file==0) fn = (char *)filename;
            else if (file==1) fn = (char *)filename2;
            else fn = (char *)filename3;
            if (fn==NULL) continue;   // No file!
            // opening input file
            fprintf(stdout, "# Reading from file %d named %s\n", file, fn);
            FILE *fp = fopen(fn,  "rb");
            assert(fp!=NULL);
            // preparing output files
            if (file==0){
              string str = filename;
              string out_1 = str + "_rec";
              // doc. files: string >> char
              out_D = new char[out_1.size() + 1];
              strcpy(out_D,out_1.c_str());
              fno = (char *)out_D;
              if (fno==NULL) continue;   // No file!
            }
            else if(file==1){
              string str = filename2;
              string out_2 = str + "_rec";
              // doc. files: string >> char
              out_R = new char[out_2.size() + 1];
              strcpy(out_R,out_2.c_str());
              fno = (char *)out_R;
              if (fno==NULL) continue;   // No file!
            }
            // opening output file
            fprintf(stdout, "# Outputting to file %d named %s\n", file, fno);
            FILE *fpo = fopen(fno,  "wb");
            assert(fpo!=NULL);
            // header: reading posmin[3], posmax[3], max_sep, blank8
            for (int p=0; p<3; p++) {
                header.posmin[p] = posmin[p];
                header.posmax[p] = posmax[p];
            }
            header.maxcep = max_sep;
            header.blank8 = 123.456;  // 8byte blank
            fwrite(&header, sizeof(header), 1, fpo);
            // now, reading
            int nread=fread(tmp, sizeof(double), 8, fp); assert(nread==8); // Skip the header
            while ((nread=fread(&buffer, sizeof(double), BUFFERSIZE, fp))>0) {
                b=buffer;
                for (int j=0; j<nread; j+=4,b+=4) {
                    index=get_index(b);
                    gal.push_back(Galaxy(b,index));
                    thiscnt++; totw += b[3]; totwsq += b[3]*b[3];
                    if (gal.size()>=MAXGAL) {
                        IO.Stop();
                        galsize_out = gal.size();
                        Cloud_In_Cell(gal,1,output,type,file);
                        // if output=1, outputting structure x_w[] to file_D_rec
                        if(output){
                            Out.Start();
                            uint64 nout;
                            uint64 total_nout = 0;
                            uint64 num_ran = 0;
                            struct_out_x3w *px_w = x_w;
                            while (total_nout != galsize_out) {
                              nout=fwrite(px_w, sizeof(x_w[0]), num_ran, fpo);
                              total_nout += nout;
                              if((galsize_out - total_nout) >= BUFFERSIZE){
                                  num_ran = BUFFERSIZE;
                              }
                              else{
                                  num_ran = (galsize_out - total_nout);
                              }
                              px_w += nout;
                            }
                            Out.Stop();
                        }
                        IO.Start();
                    }
                }
                if (nread!=BUFFERSIZE) break;
            }
            cnt += thiscnt;
            fprintf(stdout, "# Found %lld galaxies in this file\n", thiscnt);
            fclose(fp);
            IO.Stop();
            // Add the remaining galaxies to the grid
            galsize_out = gal.size();
            Cloud_In_Cell(gal,1,output,type,file);
            // if output=1, outputting structure x_w[] to file_D_rec
            if(output){
                Out.Start();
                uint64 nout;
                uint64 total_nout = 0;
                uint64 num_ran = 0;
                struct_out_x3w *px_w = x_w;
                while (total_nout != galsize_out) {
                  nout=fwrite(px_w, sizeof(x_w[0]), num_ran, fpo);
                  total_nout += nout;
                  if((galsize_out - total_nout) >= BUFFERSIZE){
                      num_ran = BUFFERSIZE;
                  }
                  else{
                      num_ran = (galsize_out - total_nout);
                  }
                  px_w += nout;
                }
                Out.Stop();
            }
            fclose(fpo);
            if(!output) remove(fno);
            IO.Start();
        }
        IO.Stop();
        // only for input
        if(output==0){
            fprintf(stdout, "# Found %lld particles. Total weight %10.4e.\n", cnt, totw);
            Float totw2;
            if(type==9)       totw2 = sum_matrix(shift_t_x, ngrid3, ngrid[0]);
            else if(type==10) totw2 = sum_matrix(shift_x, ngrid3, ngrid[0]);
            else              totw2 = sum_matrix(dens, ngrid3, ngrid[0]);
            fprintf(stdout, "# Sum of grid is %10.4e (delta = %10.4e)\n", totw2, totw2-totw);
            Float sumsq_dens;
            if(type==9)       sumsq_dens = sumsq_matrix(shift_t_x, ngrid3, ngrid[0]);
            else if(type==10) sumsq_dens = sumsq_matrix(shift_x, ngrid3, ngrid[0]);
            else              sumsq_dens = sumsq_matrix(dens, ngrid3, ngrid[0]);
            fprintf(stdout, "# Sum of squares of density = %14.7e\n", sumsq_dens);
            Pshot = totwsq;
            fprintf(stdout, "# Sum of squares of weights (divide by I for Pshot) = %14.7e\n", Pshot);
            // When run with N=D-R, this divided by I would be the shot noise.

            // Meanwhile, an estimate of I when running with only R is
            // (sum of R^2)/Vcell - (11/20)**3*(sum_R w^2)/Vcell
            // The latter is correcting the estimate for shot noise
            // The 11/20 factor is for triangular cloud in cell.
            #ifndef NEAREST_CELL
            #ifdef WAVELET
                fprintf(stdout, "# Using D12 wavelet\n");
            #else
                totwsq *= 0.55*0.55*0.55;
                fprintf(stdout, "# Using triangular cloud-in-cell\n");
            #endif
            #else
                fprintf(stdout, "# Using nearest cell method\n");
            #endif
            Float Vcell = cell_size*cell_size*cell_size;
            fprintf(stdout, "# Estimate of I (denominator) = %14.7e - %14.7e = %14.7e\n",
                    sumsq_dens/Vcell, totwsq/Vcell, (sumsq_dens-totwsq)/Vcell);
        }
        return;
  }


/* ------------------------------------------------------------------- */

    void Cloud_In_Cell(std::vector<Galaxy> &gal, int ori_coor,
                                        int output, int type, int file) {
        // Given a set of Galaxies, add them to the grid and then reset the list
        CIC.Start();
        const int galsize = gal.size();
    #ifdef DEPRECATED
        // This works, but appears to be slower
        for (int j=0; j<galsize; j++) {
          if(output == 0) add_value_to_grid(j,gal[j],ori_coor,type,file);
          else if(output == 1) get_value_from_grid(j,gal[j],ori_coor,type,file);
          else {
            if(file == 0) sumsq_dif_x += return_value_from_grid(j,gal[j],ori_coor,type,file);
            else if(file == 1) sumsq_dif_y += return_value_from_grid(j,gal[j],ori_coor,type,file);
            else sumsq_dif_z += return_value_from_grid(j,gal[j],ori_coor,type,file);
          }
        }
    #else
        // If we're parallelizing this, then we need to keep the threads from stepping
        // on each other.  Do this in slabs, but with only every third slab active at
        // any time.

        // Let's sort the particles by x.
        // Need to supply an equal amount of temporary space to merge sort.
        // Do this by another vector.
        std::vector<Galaxy> tmp;
        tmp.reserve(galsize);
        mergesort_parallel_omp(gal.data(), galsize, tmp.data(), omp_get_max_threads());
        // This just falls back to std::sort if omp_get_max_threads==1

        // Now we need to find the starting point of each slab
        // Galaxies between N and N+1 should be in indices [first[N], first[N+1]).
        // That means that first[N] should be the index of the first galaxy to exceed N.
        // this gal[j].x is still original coordinate, so need to get grid coordinate !!
        int first[ngrid[0]+1], ptr=0;
        double gal_j_x;
        for (int j=0; j<galsize; j++){
            if(ori_coor) gal_j_x = (gal[j].x-posmin[0])/cell_size;
            else gal_j_x = gal[j].x;
          while (gal_j_x>=ptr)
                first[ptr++] = j;
        }
        for (;ptr<=ngrid[0];ptr++) first[ptr]=galsize;
        assert(first[0]==0);

        // Now, we'll loop, with each thread in charge of slab x.
        // Not bothering with NUMA issues.  a) Most of the time is spent waiting for
        // memory to respond, not actually piping between processors.  b) Adjacent
        // slabs may not be on the same memory bank anyways.  Keep it simple.
        int slabset = 3;
        while(ngrid[0]%slabset != 0) slabset ++;
        #ifdef WAVELET
            slabset = WCELLS;
        #endif
        for (int mod=0; mod<slabset; mod++) {
            Float sdx, sdy, sdz;
            sdx = sdy = sdz = 0.0;
            #pragma omp parallel for schedule(dynamic,1) reduction(+:sdx, sdy, sdz)
            for (int x=mod; x<ngrid[0]; x+=slabset) {
                // For each slab, insert these particles
                for (int j=first[x]; j<first[x+1]; j++) {
                  if(output == 0) add_value_to_grid(j,gal[j],ori_coor,type,file);
                  else if(output == 1) get_value_from_grid(j,gal[j],ori_coor,type,file);
                  else{
                    if(file == 0) sdx += return_value_from_grid(j,gal[j],ori_coor,type,file);
                    else if(file == 1) sdy += return_value_from_grid(j,gal[j],ori_coor,type,file);
                    else sdz += return_value_from_grid(j,gal[j],ori_coor,type,file);
                  }
                }
            }
            sumsq_dif_x += sdx;
            sumsq_dif_y += sdy;
            sumsq_dif_z += sdz;
        }
    #endif
        gal.clear();
        gal.reserve(MAXGAL);    // Just to check!
        CIC.Stop();
        return;
    }


/* ------------------------------------------------------------------- */

    inline uint64 get_index(Float tmp[4]) {
        // Given tmp[4] = x,y,z,w,
        // get the index "without" modifing to put them in box coordinates.
        // We'll have still use for the original coordinates!!!
        double tmp_d[3];
        tmp_d[0] = (tmp[0]-posmin[0])/cell_size;
        tmp_d[1] = (tmp[1]-posmin[1])/cell_size;
        tmp_d[2] = (tmp[2]-posmin[2])/cell_size;
        assert(tmp_d[0]>=0);
        assert(tmp_d[1]>=0);
        assert(tmp_d[2]>=0);

        uint64 ix = floor(tmp_d[0]);
        uint64 iy = floor(tmp_d[1]);
        uint64 iz = floor(tmp_d[2]);
        return (iz)+ngrid2*((iy)+(ix)*ngrid[1]);
    }

    void add_value_to_grid(int j, Galaxy g, int ori_coor, int type, int file) {
        // This is corresponding to output = 0
        // Add the value(g.w) to grid.
        // type =-1: work(sum of contributions from each grid (for grid)),
        //      = 0: dens,
        //      = 1: shift_t_X (for shift), shift_xX (for count)
        //      = 2: shift_X   (for shift), shift_yX (for count)

        // This does a 27-point triangular cloud-in-cell, unless one invokes NEAREST_CELL.
        Float ori_x[3],grid_x[3];
        if(ori_coor){
          // If g is still the original coordinates,
          // Copy the original coordinates to ori_x[].
          ori_x[0] = g.x;
          ori_x[1] = g.y;
          ori_x[2] = g.z;
          // Modify to put them in box coordinates.
          g.x = (g.x-posmin[0])/cell_size;
          g.y = (g.y-posmin[1])/cell_size;
          g.z = (g.z-posmin[2])/cell_size;
        }
        grid_x[0] = g.x;
        grid_x[1] = g.y;
        grid_x[2] = g.z;

        uint64 index;   // Trying not to assume that ngrid**3 won't spill 32-bits.
        uint64 ix = floor(grid_x[0]);
        uint64 iy = floor(grid_x[1]);
        uint64 iz = floor(grid_x[2]);
        index = (iz)+ngrid2*((iy)+(ix)*ngrid[1]);

        // If we're just doing nearest cell.
        #ifdef NEAREST_CELL
            if(type==0) dens[index] += g.w;
            else if(type < 0){
              C_O[index] += 1.0;
              work[index] += g.w;
            }
            else if(type==1){
                if(file==0)      shift_t_x[index] += g.w;
                else if(file==1) shift_t_y[index] += g.w;
                else             shift_t_z[index] += g.w;
            }
            else if(type==2){
                if(file==0)      shift_x[index] += g.w;
                else if(file==1) shift_y[index] += g.w;
                else             shift_z[index] += g.w;
            }
            return;
        #endif

        #ifdef WAVELET
            // In the wavelet version, we truncate to 1/WAVESAMPLE resolution in each
            // cell and use a lookup table.  Table is set up so that each sub-cell
            // resolution has the values for the various integral cell offsets contiguous
            // in memory.
            uint64 sx = floor((g.x-ix)*WAVESAMPLE);
            uint64 sy = floor((g.y-iy)*WAVESAMPLE);
            uint64 sz = floor((g.z-iz)*WAVESAMPLE);
            const Float *xwave = wave+sx*WCELLS;
            const Float *ywave = wave+sy*WCELLS;
            const Float *zwave = wave+sz*WCELLS;
            // This code does periodic wrapping
            const uint64 ng0 = ngrid[0];
            const uint64 ng1 = ngrid[1];
            const uint64 ng2 = ngrid[2];
            // Offset to the lower-most cell, taking care to handle unsigned int
            ix = (ix+ng0+WMIN)%ng0;
            iy = (iy+ng1+WMIN)%ng1;
            iz = (iz+ng2+WMIN)%ng2;
            Float *px = dens+ngrid2*ng1*ix;
            for (int ox=0; ox<WCELLS; ox++, px+=ngrid2*ng1) {
                if (ix+ox==ng0) px -= ng0*ng1*ngrid2;  // Periodic wrap in X
                Float Dx = xwave[ox]*g.w;
                Float *py = px + iy*ngrid2;
                for (int oy=0; oy<WCELLS; oy++, py+=ngrid2) {
                    if (iy+oy==ng1) py -= ng1*ngrid2;  // Periodic wrap in Y
                    Float *pz = py+iz;
                    Float Dxy = Dx*ywave[oy];
                    if (iz+WCELLS>ng2) {     // Z Wrap is needed
                        for (int oz=0; oz<WCELLS; oz++) {
                            if (iz+oz==ng2) pz -= ng2;  // Periodic wrap in Z
                            pz[oz] += zwave[oz]*Dxy;
                        }
                    } else {
                        for (int oz=0; oz<WCELLS; oz++) pz[oz] += zwave[oz]*Dxy;
                    }
                }
            }
            return;
        #endif

        // This is TSC
        if(type==0) TSC_to_grid(grid_x, g.w, dens, type);
        else if(type < 0){
          C_O[index] += 1.0;
          TSC_to_grid(grid_x, g.w, work, type);
        }
        else if(type==1){
            if(file==0){
                    TSC_to_grid(grid_x, g.w, shift_t_x, type);
                    TSC_to_grid(grid_x, 1.0, shift_xx, type);
            }
            else if(file==1){
                    TSC_to_grid(grid_x, g.w, shift_t_y, type);
                    TSC_to_grid(grid_x, 1.0, shift_xy, type);
            }
            else{
                    TSC_to_grid(grid_x, g.w, shift_t_z, type);
                    TSC_to_grid(grid_x, 1.0, shift_xz, type);
            }
        }
        else if(type==2){
            if(file==0){
                    TSC_to_grid(grid_x, g.w, shift_x, type);
                    TSC_to_grid(grid_x, 1.0, shift_yx, type);
            }
            else if(file==1){
                    TSC_to_grid(grid_x, g.w, shift_y, type);
                    TSC_to_grid(grid_x, 1.0, shift_yy, type);
            }
            else{
                    TSC_to_grid(grid_x, g.w, shift_z, type);
                    TSC_to_grid(grid_x, 1.0, shift_yz, type);
            }
        }
        return;
  } // add_value_to_grid

    void get_value_from_grid(int j, Galaxy g, int ori_coor, int type, int file) {
        // This is corresponding to output = 1
        // get the value corresponfing the position (g.x,y,and z) from grid.
        // type = 0: shift,
        //      = 1: dens_s
        //      = 2: dens_res
        // This does a 27-point triangular cloud-in-cell, unless one invokes NEAREST_CELL.
        Float ori_x[3],grid_x[3];
        if(ori_coor){
          // If g is still the original coordinates,
          // Copy the original coordinates to ori_x[].
          ori_x[0] = g.x;
          ori_x[1] = g.y;
          ori_x[2] = g.z;
          // Modify to put them in box coordinates.
          g.x = (g.x-posmin[0])/cell_size;
          g.y = (g.y-posmin[1])/cell_size;
          g.z = (g.z-posmin[2])/cell_size;
        }
        grid_x[0] = g.x;
        grid_x[1] = g.y;
        grid_x[2] = g.z;

        uint64 index;   // Trying not to assume that ngrid**3 won't spill 32-bits.
        uint64 ix = floor(grid_x[0]);
        uint64 iy = floor(grid_x[1]);
        uint64 iz = floor(grid_x[2]);
        index = (iz)+ngrid2*((iy)+(ix)*ngrid[1]);

        Float total_s_x, total_s_y, total_s_z,
        total_s_r_x, total_s_r_y, total_s_r_z,
        total_weight, total_dens;
        total_s_x = total_s_y = total_s_z
        = total_s_r_x = total_s_r_y = total_s_r_z
        = total_weight = total_dens = 0.0;

        // If we're just doing nearest cell.
        #ifdef NEAREST_CELL
            if(type==0){
              #ifdef ITERATION
                total_s_x = shift_xx[index];
                total_s_y = shift_yx[index];
                total_s_z = shift_zx[index];
                total_s_r_x = shift_r_zx[index];
                total_s_r_y = shift_r_zy[index];
                total_s_r_z = shift_r_zz[index];
              #else
                total_s_x = shift_x[index];
                total_s_y = shift_y[index];
                total_s_z = shift_z[index];
                total_s_r_z = shift_r_z[index];
              #endif
              total_weight = g.w;
            }
            else if(type==1){
              total_dens = dens_s[index];
              total_weight = g.w;
            }
            else if(type==2){
              total_dens = dens_res[index];
              total_weight = g.w;
            }
        // This is TSC
        #else
          if(type==0){
            #ifdef ITERATION
              total_s_x = TSC_from_grid(grid_x, 1.0, shift_xx, type);
              total_s_y = TSC_from_grid(grid_x, 1.0, shift_yx, type);
              total_s_z = TSC_from_grid(grid_x, 1.0, shift_zx, type);
              total_s_r_x = TSC_from_grid(grid_x, 1.0, shift_r_zx, type);
              total_s_r_y = TSC_from_grid(grid_x, 1.0, shift_r_zy, type);
              total_s_r_z = TSC_from_grid(grid_x, 1.0, shift_r_zz, type);
            #else
              total_s_x = TSC_from_grid(grid_x, 1.0, shift_x, type);
              total_s_y = TSC_from_grid(grid_x, 1.0, shift_y, type);
              total_s_z = TSC_from_grid(grid_x, 1.0, shift_z, type);
              total_s_r_z = TSC_from_grid(grid_x, 1.0, shift_r_z, type);
            #endif
            total_weight = g.w;
          }
          else if(type==1){
            total_dens = TSC_from_grid(grid_x, 1.0, dens_s, type);
            total_weight = g.w;
          }
          else if(type==2){
            total_dens = TSC_from_grid(grid_x, 1.0, dens_res, type);
            total_weight = g.w;
          }
        #endif

      /*    !!!!!  have not rewrited this part   !!!!!
        #ifdef WAVELET
            // In the wavelet version, we truncate to 1/WAVESAMPLE resolution in each
            // cell and use a lookup table.  Table is set up so that each sub-cell
            // resolution has the values for the various integral cell offsets contiguous
            // in memory.
            uint64 sx = floor((g.x-ix)*WAVESAMPLE);
            uint64 sy = floor((g.y-iy)*WAVESAMPLE);
            uint64 sz = floor((g.z-iz)*WAVESAMPLE);
            const Float *xwave = wave+sx*WCELLS;
            const Float *ywave = wave+sy*WCELLS;
            const Float *zwave = wave+sz*WCELLS;
            // This code does periodic wrapping
            const uint64 ng0 = ngrid[0];
            const uint64 ng1 = ngrid[1];
            const uint64 ng2 = ngrid[2];
            // Offset to the lower-most cell, taking care to handle unsigned int
            ix = (ix+ng0+WMIN)%ng0;
            iy = (iy+ng1+WMIN)%ng1;
            iz = (iz+ng2+WMIN)%ng2;
            Float *px = dens+ngrid2*ng1*ix;
            for (int ox=0; ox<WCELLS; ox++, px+=ngrid2*ng1) {
                if (ix+ox==ng0) px -= ng0*ng1*ngrid2;  // Periodic wrap in X
                Float Dx = xwave[ox]*g.w;
                Float *py = px + iy*ngrid2;
                for (int oy=0; oy<WCELLS; oy++, py+=ngrid2) {
                    if (iy+oy==ng1) py -= ng1*ngrid2;  // Periodic wrap in Y
                    Float *pz = py+iz;
                    Float Dxy = Dx*ywave[oy];
                    if (iz+WCELLS>ng2) {     // Z Wrap is needed
                        for (int oz=0; oz<WCELLS; oz++) {
                            if (iz+oz==ng2) pz -= ng2;  // Periodic wrap in Z
                            pz[oz] += zwave[oz]*Dxy;
                        }
                    } else {
                        for (int oz=0; oz<WCELLS; oz++) pz[oz] += zwave[oz]*Dxy;
                    }
                }
            }
            return;
        #endif
      */

        // getting shifted positions
        if(type==0){
              #ifdef RSD
                #ifndef ITERATION
                  if(file == 0){  // for Data
                    x_w[j].x[0] = ori_x[0] - total_s_x;
                    x_w[j].x[1] = ori_x[1] - total_s_y;
                    x_w[j].x[2] = ori_x[2] - (1.0 + lgr_f)*total_s_z;
                    x_w[j].w = total_weight;
                  }
                  else{    // for Random
                    x_w[j].x[0] = ori_x[0] - total_s_x;
                    x_w[j].x[1] = ori_x[1] - total_s_y;
                    x_w[j].x[2] = ori_x[2] - total_s_z;
                    x_w[j].w = total_weight;
                  }
                #else
                  if(file == 0){  // for Data
                    x_w[j].x[0] = ori_x[0] - total_s_x;
                    x_w[j].x[1] = ori_x[1] - total_s_y;
                    x_w[j].x[2] = ori_x[2] - total_s_z;
                    x_w[j].w = total_weight;
                  }
                  else{    // for Random
                    x_w[j].x[0] = ori_x[0] - total_s_r_x;
                    x_w[j].x[1] = ori_x[1] - total_s_r_y;
                    x_w[j].x[2] = ori_x[2] - total_s_r_z;
                    x_w[j].w = total_weight;
                  }
                #endif
              #else
                x_w[j].x[0] = ori_x[0] - total_s_x;
                x_w[j].x[1] = ori_x[1] - total_s_y;
                x_w[j].x[2] = ori_x[2] - total_s_z;
                x_w[j].w = total_weight;
              #endif
          if(x_w[j].x[0]<=posmin[0]) x_w[j].x[0] += posrange[0];
          if(x_w[j].x[0]>=posmax[0]) x_w[j].x[0] -= posrange[0];
          if(x_w[j].x[1]<=posmin[1]) x_w[j].x[1] += posrange[1];
          if(x_w[j].x[1]>=posmax[1]) x_w[j].x[1] -= posrange[1];
          if(x_w[j].x[2]<=posmin[2]) x_w[j].x[2] += posrange[2];
          if(x_w[j].x[2]>=posmax[2]) x_w[j].x[2] -= posrange[2];
        }
        else if(type==1 || type==2){
          work[(int)total_weight] += total_dens;
        }
        return;
  } // end get_value_from_grid

    Float return_value_from_grid(int j, Galaxy g, int ori_coor, int type, int file) {
        // This is corresponding to output = 2
        // return the value corresponfing the position (g.x,y,and z) from grid.
        // type = 0: shift_t_X
        // This does a 27-point triangular cloud-in-cell, unless one invokes NEAREST_CELL.
        Float ori_x[3],grid_x[3];
        if(ori_coor){
          // If g is still the original coordinates,
          // Copy the original coordinates to ori_x[].
          ori_x[0] = g.x;
          ori_x[1] = g.y;
          ori_x[2] = g.z;
          // Modify to put them in box coordinates.
          g.x = (g.x-posmin[0])/cell_size;
          g.y = (g.y-posmin[1])/cell_size;
          g.z = (g.z-posmin[2])/cell_size;
        }
        grid_x[0] = g.x;
        grid_x[1] = g.y;
        grid_x[2] = g.z;

        uint64 index;   // Trying not to assume that ngrid**3 won't spill 32-bits.
        uint64 ix = floor(grid_x[0]);
        uint64 iy = floor(grid_x[1]);
        uint64 iz = floor(grid_x[2]);
        index = (iz)+ngrid2*((iy)+(ix)*ngrid[1]);

        Float total_s_x, total_s_y, total_s_z,
        total_s_r_x, total_s_r_y, total_s_r_z,
        total_weight, total_dens, sumsq_dif;
        total_s_x = total_s_y = total_s_z
        = total_s_r_x = total_s_r_y = total_s_r_z
        = total_weight = total_dens = sumsq_dif = 0.0;

        // If we're just doing nearest cell.
        #ifdef NEAREST_CELL
            if(type==0){
              if(file==0){
                      total_s_x = shift_x[index];
                      total_weight = g.w;
              }
              else if(file==1){
                      total_s_y = shift_y[index];
                      total_weight = g.w;
              }
              else{
                      total_s_z = shift_z[index];
                      total_weight = g.w;
              }
            }
        // This is TSC
        #else
          if(type==0){
            if(file==0){
                    total_s_x = TSC_from_grid(grid_x, 1.0, shift_x, type);
                    total_weight = g.w;
            }
            else if(file==1){
                    total_s_y = TSC_from_grid(grid_x, 1.0, shift_y, type);
                    total_weight = g.w;
            }
            else{
                    total_s_z = TSC_from_grid(grid_x, 1.0, shift_z, type);
                    total_weight = g.w;
            }
          }
        #endif

        // getting shifted positions
        if(type==0){
          if(file==0){
                  sumsq_dif = (total_weight - total_s_x)*(total_weight - total_s_x);
          }
          else if(file==1){
                  sumsq_dif = (total_weight - total_s_y)*(total_weight - total_s_y);
          }
          else{
                  sumsq_dif = (total_weight - total_s_z)*(total_weight - total_s_z);
          }
        }
        return sumsq_dif;
  } // end return_value_from_grid

/* ------------------------------------------------------------------- */

    void TSC_to_grid(Float grid_x[3], Float grid_w, Float *T, int type) {

        uint64 index;   // Trying not to assume that ngrid**3 won't spill 32-bits.
        uint64 ix = floor(grid_x[0]);
        uint64 iy = floor(grid_x[1]);
        uint64 iz = floor(grid_x[2]);

        // Now to Cloud-in-Cell
        Float rx = grid_x[0]-ix;
        Float ry = grid_x[1]-iy;
        Float rz = grid_x[2]-iz;
        //
        Float xm = 0.5*(1-rx)*(1-rx);
        Float xp = 0.5*rx*rx;
        Float x0 = (0.5+rx-rx*rx);
        Float ym = 0.5*(1-ry)*(1-ry);
        Float yp = 0.5*ry*ry;
        Float y0 = 0.5+ry-ry*ry;
        Float zm = 0.5*(1-rz)*(1-rz);
        Float zp = 0.5*rz*rz;
        Float z0 = 0.5+rz-rz*rz;
        xm *=grid_w;
        xp *=grid_w;
        x0 *=grid_w;
        //
        if (ix==0||ix==ngrid[0]-1 || iy==0||iy==ngrid[1]-1 || iz==0||iz==ngrid[2]-1) {
            // This code does periodic wrapping
            const uint64 ng0 = ngrid[0];
            const uint64 ng1 = ngrid[1];
            const uint64 ng2 = ngrid[2];
            ix += ngrid[0];   // Just to put away any fears of negative mods
            iy += ngrid[1];
            iz += ngrid[2];
            const uint64 izm = (iz-1)%ng2;
            const uint64 iz0 = (iz  )%ng2;
            const uint64 izp = (iz+1)%ng2;

            //
            index = ngrid2*(((iy-1)%ng1)+((ix-1)%ng0)*ng1);
            T[index+izm] += xm*ym*zm;
            T[index+iz0] += xm*ym*z0;
            T[index+izp] += xm*ym*zp;
            index = ngrid2*(((iy  )%ng1)+((ix-1)%ng0)*ng1);
            T[index+izm] += xm*y0*zm;
            T[index+iz0] += xm*y0*z0;
            T[index+izp] += xm*y0*zp;
            index = ngrid2*(((iy+1)%ng1)+((ix-1)%ng0)*ng1);
            T[index+izm] += xm*yp*zm;
            T[index+iz0] += xm*yp*z0;
            T[index+izp] += xm*yp*zp;
            //
            index = ngrid2*(((iy-1)%ng1)+((ix  )%ng0)*ng1);
            T[index+izm] += x0*ym*zm;
            T[index+iz0] += x0*ym*z0;
            T[index+izp] += x0*ym*zp;
            index = ngrid2*(((iy  )%ng1)+((ix  )%ng0)*ng1);
            T[index+izm] += x0*y0*zm;
            T[index+iz0] += x0*y0*z0;
            T[index+izp] += x0*y0*zp;
            index = ngrid2*(((iy+1)%ng1)+((ix  )%ng0)*ng1);
            T[index+izm] += x0*yp*zm;
            T[index+iz0] += x0*yp*z0;
            T[index+izp] += x0*yp*zp;
            //
            index = ngrid2*(((iy-1)%ng1)+((ix+1)%ng0)*ng1);
            T[index+izm] += xp*ym*zm;
            T[index+iz0] += xp*ym*z0;
            T[index+izp] += xp*ym*zp;
            index = ngrid2*(((iy  )%ng1)+((ix+1)%ng0)*ng1);
            T[index+izm] += xp*y0*zm;
            T[index+iz0] += xp*y0*z0;
            T[index+izp] += xp*y0*zp;
            index = ngrid2*(((iy+1)%ng1)+((ix+1)%ng0)*ng1);
            T[index+izm] += xp*yp*zm;
            T[index+iz0] += xp*yp*z0;
            T[index+izp] += xp*yp*zp;

        } else {
            // This code is faster, but doesn't do periodic wrapping
            index = (iz-1)+ngrid2*((iy-1)+(ix-1)*ngrid[1]);
            T[index++] += xm*ym*zm;
            T[index++] += xm*ym*z0;
            T[index]   += xm*ym*zp;
            index += ngrid2-2;   // Step to the next row in y
            T[index++] += xm*y0*zm;
            T[index++] += xm*y0*z0;
            T[index]   += xm*y0*zp;
            index += ngrid2-2;   // Step to the next row in y
            T[index++] += xm*yp*zm;
            T[index++] += xm*yp*z0;
            T[index]   += xm*yp*zp;
            index = (iz-1)+ngrid2*((iy-1)+ix*ngrid[1]);
            T[index++] += x0*ym*zm;
            T[index++] += x0*ym*z0;
            T[index]   += x0*ym*zp;
            index += ngrid2-2;   // Step to the next row in y
            T[index++] += x0*y0*zm;
            T[index++] += x0*y0*z0;
            T[index]   += x0*y0*zp;
            index += ngrid2-2;   // Step to the next row in y
            T[index++] += x0*yp*zm;
            T[index++] += x0*yp*z0;
            T[index]   += x0*yp*zp;
            index = (iz-1)+ngrid2*((iy-1)+(ix+1)*ngrid[1]);
            T[index++] += xp*ym*zm;
            T[index++] += xp*ym*z0;
            T[index]   += xp*ym*zp;
            index += ngrid2-2;   // Step to the next row in y
            T[index++] += xp*y0*zm;
            T[index++] += xp*y0*z0;
            T[index]   += xp*y0*zp;
            index += ngrid2-2;   // Step to the next row in y
            T[index++] += xp*yp*zm;
            T[index++] += xp*yp*z0;
            T[index]   += xp*yp*zp;
        }
        return;
  } // end TSC_to_grid

    Float TSC_from_grid(Float grid_x[3], Float grid_w, Float *F, int type) {

        uint64 index;   // Trying not to assume that ngrid**3 won't spill 32-bits.
        uint64 ix = floor(grid_x[0]);
        uint64 iy = floor(grid_x[1]);
        uint64 iz = floor(grid_x[2]);
        index = (iz)+ngrid2*((iy)+(ix)*ngrid[1]);

        // Now to Cloud-in-Cell
        Float rx = grid_x[0]-ix;
        Float ry = grid_x[1]-iy;
        Float rz = grid_x[2]-iz;
        //
        Float xm = 0.5*(1-rx)*(1-rx);
        Float xp = 0.5*rx*rx;
        Float x0 = 0.5+rx-rx*rx;
        Float ym = 0.5*(1-ry)*(1-ry);
        Float yp = 0.5*ry*ry;
        Float y0 = 0.5+ry-ry*ry;
        Float zm = 0.5*(1-rz)*(1-rz);
        Float zp = 0.5*rz*rz;
        Float z0 = 0.5+rz-rz*rz;
        xm *=grid_w;
        xp *=grid_w;
        x0 *=grid_w;
        //
        Float to_value = 0.0;

        if (ix==0||ix==ngrid[0]-1 || iy==0||iy==ngrid[1]-1 || iz==0||iz==ngrid[2]-1) {
            // This code does periodic wrapping
            const uint64 ng0 = ngrid[0];
            const uint64 ng1 = ngrid[1];
            const uint64 ng2 = ngrid[2];
            ix += ngrid[0];   // Just to put away any fears of negative mods
            iy += ngrid[1];
            iz += ngrid[2];
            const uint64 izm = (iz-1)%ng2;
            const uint64 iz0 = (iz  )%ng2;
            const uint64 izp = (iz+1)%ng2;
            //
            index = ngrid2*(((iy-1)%ng1)+((ix-1)%ng0)*ng1);
            to_value += xm*ym*zm*F[index+izm];
            to_value += xm*ym*z0*F[index+iz0];
            to_value += xm*ym*zp*F[index+izp];

            index = ngrid2*(((iy  )%ng1)+((ix-1)%ng0)*ng1);
            to_value += xm*y0*zm*F[index+izm];
            to_value += xm*y0*z0*F[index+iz0];
            to_value += xm*y0*zp*F[index+izp];

            index = ngrid2*(((iy+1)%ng1)+((ix-1)%ng0)*ng1);
            to_value += xm*yp*zm*F[index+izm];
            to_value += xm*yp*z0*F[index+iz0];
            to_value += xm*yp*zp*F[index+izp];

            //
            index = ngrid2*(((iy-1)%ng1)+((ix  )%ng0)*ng1);
            to_value += x0*ym*zm*F[index+izm];
            to_value += x0*ym*z0*F[index+iz0];
            to_value += x0*ym*zp*F[index+izp];

            index = ngrid2*(((iy  )%ng1)+((ix  )%ng0)*ng1);
            to_value += x0*y0*zm*F[index+izm];
            to_value += x0*y0*z0*F[index+iz0];
            to_value += x0*y0*zp*F[index+izp];

            index = ngrid2*(((iy+1)%ng1)+((ix  )%ng0)*ng1);
            to_value += x0*yp*zm*F[index+izm];
            to_value += x0*yp*z0*F[index+iz0];
            to_value += x0*yp*zp*F[index+izp];

            //
            index = ngrid2*(((iy-1)%ng1)+((ix+1)%ng0)*ng1);
            to_value += xp*ym*zm*F[index+izm];
            to_value += xp*ym*z0*F[index+iz0];
            to_value += xp*ym*zp*F[index+izp];

            index = ngrid2*(((iy  )%ng1)+((ix+1)%ng0)*ng1);
            to_value += xp*y0*zm*F[index+izm];
            to_value += xp*y0*z0*F[index+iz0];
            to_value += xp*y0*zp*F[index+izp];

            index = ngrid2*(((iy+1)%ng1)+((ix+1)%ng0)*ng1);
            to_value += xp*yp*zm*F[index+izm];
            to_value += xp*yp*z0*F[index+iz0];
            to_value += xp*yp*zp*F[index+izp];

        } else {
            // This code is faster, but doesn't do periodic wrapping
            index = (iz-1)+ngrid2*((iy-1)+(ix-1)*ngrid[1]);
            to_value += xm*ym*zm*F[index++];
            to_value += xm*ym*z0*F[index++];
            to_value += xm*ym*zp*F[index];

            index += ngrid2-2;   // Step to the next row in y
            to_value += xm*y0*zm*F[index++];
            to_value += xm*y0*z0*F[index++];
            to_value += xm*y0*zp*F[index];

            index += ngrid2-2;   // Step to the next row in y
            to_value += xm*yp*zm*F[index++];
            to_value += xm*yp*z0*F[index++];
            to_value += xm*yp*zp*F[index];

            index = (iz-1)+ngrid2*((iy-1)+ix*ngrid[1]);
            to_value += x0*ym*zm*F[index++];
            to_value += x0*ym*z0*F[index++];
            to_value += x0*ym*zp*F[index];

            index += ngrid2-2;   // Step to the next row in y
            to_value += x0*y0*zm*F[index++];
            to_value += x0*y0*z0*F[index++];
            to_value += x0*y0*zp*F[index];

            index += ngrid2-2;   // Step to the next row in y
            to_value += x0*yp*zm*F[index++];
            to_value += x0*yp*z0*F[index++];
            to_value += x0*yp*zp*F[index];

            index = (iz-1)+ngrid2*((iy-1)+(ix+1)*ngrid[1]);
            to_value += xp*ym*zm*F[index++];
            to_value += xp*ym*z0*F[index++];
            to_value += xp*ym*zp*F[index];

            index += ngrid2-2;   // Step to the next row in y
            to_value += xp*y0*zm*F[index++];
            to_value += xp*y0*z0*F[index++];
            to_value += xp*y0*zp*F[index];

            index += ngrid2-2;   // Step to the next row in y
            to_value += xp*yp*zm*F[index++];
            to_value += xp*yp*z0*F[index++];
            to_value += xp*yp*zp*F[index];
        }
        return to_value;
  } // end TSC_from_grid

/* ------------------------------------------------------------------- */

    Float setup_corr(Float _sep, Float _kmax) {
        // Set up the sub-matrix information, assuming that we'll extract
        // -sep..+sep cells around zero-lag.
        // _sep<0 causes a default to the value in the file.
        Setup.Start();
        if (_sep<0) sep = max_sep;
            else sep = _sep;
        fprintf(stdout,"# Chosen separation %f vs max %f\n",sep, max_sep);
        assert(sep<=max_sep);

        int sep_cell = ceil(sep/cell_size);
        csize[0] = 2*sep_cell+1;
        csize[1] = csize[2] = csize[0];
        assert(csize[0]%2==1); assert(csize[1]%2==1); assert(csize[2]%2==1);
        csize3 = csize[0]*csize[1]*csize[2];
        // Allocate corr_cell to [csize] and rnorm to [csize**3]
        int err;
        err=posix_memalign((void **) &cx_cell, PAGE, sizeof(Float)*csize[0]+PAGE); assert(err==0);
        err=posix_memalign((void **) &cy_cell, PAGE, sizeof(Float)*csize[1]+PAGE); assert(err==0);
        err=posix_memalign((void **) &cz_cell, PAGE, sizeof(Float)*csize[2]+PAGE); assert(err==0);
        initialize_matrix(rnorm, csize3, csize[0]);

        for (int i=0; i<csize[0]; i++) cx_cell[i] = i-sep_cell;
        for (int i=0; i<csize[1]; i++) cy_cell[i] = i-sep_cell;
        for (int i=0; i<csize[2]; i++) cz_cell[i] = i-sep_cell;

        for (uint64 i=0; i<csize[0]; i++)
            for (int j=0; j<csize[1]; j++)
                for (int k=0; k<csize[2]; k++)
                    rnorm[k+csize[2]*(j+i*csize[1])] = cell_size*sqrt(
                             (i-sep_cell)*(i-sep_cell)
                            +(j-sep_cell)*(j-sep_cell)
                            +(k-sep_cell)*(k-sep_cell));
        fprintf(stdout, "# Done setting up the separation submatrix of size +-%d\n", sep_cell);

        // Our box has cubic-sized cells, so k_Nyquist is the same in all directions
        // The spacing of modes is therefore 2*k_Nyq/ngrid
        k_Nyq = M_PI/cell_size;
        kmax = _kmax;
        fprintf(stdout, "# Storing wavenumbers up to %6.4f, with k_Nyq = %6.4f\n", kmax, k_Nyq);
        for (int i=0; i<3; i++) ksize[i] = 2*ceil(kmax/(2.0*k_Nyq/ngrid[i]))+1;
        assert(ksize[0]%2==1); assert(ksize[1]%2==1); assert(ksize[2]%2==1);
        for (int i=0; i<3; i++) if (ksize[i]>ngrid[i]) {
            ksize[i] = 2*floor(ngrid[i]/2)+1;
            fprintf(stdout, "# WARNING: Requested wavenumber is too big.  Truncating ksize[%d] to %d\n", i, ksize[i]);
        }

        ksize3 = ksize[0]*ksize[1]*ksize[2];
        // Allocate kX_cell to [ksize] and knorm & k_znorm to [ksize**3]
        err=posix_memalign((void **) &kx_cell, PAGE, sizeof(Float)*ksize[0]+PAGE); assert(err==0);
        err=posix_memalign((void **) &ky_cell, PAGE, sizeof(Float)*ksize[1]+PAGE); assert(err==0);
        err=posix_memalign((void **) &kz_cell, PAGE, sizeof(Float)*ksize[2]+PAGE); assert(err==0);
        initialize_matrix(knorm, ksize3, ksize[0]);
        initialize_matrix(k_znorm, ksize3, ksize[0]);
        initialize_matrix(CICwindow, ksize3, ksize[0]);

        for (int i=0; i<ksize[0]; i++) kx_cell[i] = (i-ksize[0]/2)*2.0*k_Nyq/ngrid[0];
        for (int i=0; i<ksize[1]; i++) ky_cell[i] = (i-ksize[1]/2)*2.0*k_Nyq/ngrid[1];
        for (int i=0; i<ksize[2]; i++) kz_cell[i] = (i-ksize[2]/2)*2.0*k_Nyq/ngrid[2];

        for (uint64 i=0; i<ksize[0]; i++)
            for (int j=0; j<ksize[1]; j++)
                for (int k=0; k<ksize[2]; k++) {
                    knorm[k+ksize[2]*(j+i*ksize[1])] = sqrt( kx_cell[i]*kx_cell[i]
                            +ky_cell[j]*ky_cell[j] +kz_cell[k]*kz_cell[k]);
                    k_znorm[k+ksize[2]*(j+i*ksize[1])] = sqrt(kz_cell[k]*kz_cell[k]);
                    // For TSC, the square window is 1-sin^2(kL/2)+2/15*sin^4(kL/2)
                    Float sinkxL = sin(kx_cell[i]*cell_size/2.0);
                    Float sinkyL = sin(ky_cell[j]*cell_size/2.0);
                    Float sinkzL = sin(kz_cell[k]*cell_size/2.0);
                    sinkxL *= sinkxL;
                    sinkyL *= sinkyL;
                    sinkzL *= sinkzL;
                    Float Wx, Wy, Wz;
                    Wx = 1-sinkxL+2.0/15.0*sinkxL*sinkxL;
                    Wy = 1-sinkyL+2.0/15.0*sinkyL*sinkyL;
                    Wz = 1-sinkzL+2.0/15.0*sinkzL*sinkzL;
                    Float window = Wx*Wy*Wz;   // This is the square of the window
                    #ifdef NEAREST_CELL
                        // For this case, the window is unity
                        window = 1.0;
                    #endif
                    #ifdef WAVELET
                        // For this case, the window is unity
                        window = 1.0;
                    #endif
                    CICwindow[k+ksize[2]*(j+i*ksize[1])] = 1.0/window;
                    // We will divide the power spectrum by the square of the window
                }


        fprintf(stdout, "# Done setting up the wavevector submatrix of size +-%d, %d, %d\n",
                ksize[0]/2, ksize[1]/2, ksize[2]/2);

        Setup.Stop();
        return sep;
  }

    void print_submatrix(Float *m, int n, int p, FILE *fp, Float norm) {
        // Print the inner part of a matrix(n,n,n) for debugging
        int mid = n/2;
        assert(p<=mid);
        for (int i=-p; i<=p; i++)
            for (int j=-p; j<=p; j++) {
                fprintf(fp, "%2d %2d", i, j);
                for (int k=-p; k<=p; k++) {
                    // We want to print mid+i, mid+j, mid+k
                    fprintf(fp, " %12.8g", m[((mid+i)*n+(mid+j))*n+mid+k]*norm);
                }
                fprintf(fp, "\n");
            }
        return;
    }

      /* ------------------------- reconstruction -------------------------- */

      void reconstruct(Float power_k0, Float sig_sm, Float bias) {
          // Here's where most of the work occurs.
          // This computes the shift_x,y, and z for each grid,

          // Multiply final results by the FFTW normalization
          Float norm = 1.0/ngrid[0]/ngrid[1]/ngrid[2];

          // Allocate the work matrix
          initialize_matrix(work, ngrid3, ngrid[0]);

          // Allocate the shift_xyz matrix
          if(shift_x == NULL) initialize_matrix(shift_x, ngrid3, ngrid[0]);
          if(shift_y == NULL) initialize_matrix(shift_y, ngrid3, ngrid[0]);
          if(shift_z == NULL) initialize_matrix(shift_z, ngrid3, ngrid[0]);

          // need to multiply dens[] with a coefficient to get the estimator
          // for actual denstiy field
          Float coef = power_k0/(cell_size*cell_size*cell_size);

          // Multiply the density by the coef and load the work matrix with that
          Init.Start();
          copy_matrix(work, dens, coef, ngrid3, ngrid[0]);
          Init.Stop();

          /* Setup FFTW */
          fftw_plan fft, fftYZ, fftX, ifft, ifftYZ, ifftX;
          setup_FFTW(fft, fftYZ, fftX, ifft, ifftYZ, ifftX, ngrid, ngrid2, work);

          // FFTW might have destroyed the contents of work; need to restore work[]==dens[]*coef
          // So far, I haven't seen this happen.
          if (dens[1]*coef!=work[1] || dens[1+ngrid[2]]*coef!=work[1+ngrid[2]]
                               || dens[ngrid3-1]*coef!=work[ngrid3-1]) {
              fprintf(stdout, "Restoring work matrix\n");
              Init.Start();
              copy_matrix(work, dens, coef, ngrid3, ngrid[0]);
              Init.Stop();
          }

          Reconstruct.Start();    // Starting the main work
          // Now compute the FFT of the density field and conjugate it
          // FFT(work) in place and conjugate it, storing in densFFT
          fprintf(stdout,"# Computing the density FFT..."); fflush(NULL);
          FFT_Execute(fft, fftYZ, fftX, ngrid, ngrid2, work);

          Reconstruct.Stop();    // We're tracking initialization separately
          initialize_matrix_by_copy(densFFT, ngrid3, ngrid[0], work);
          fprintf(stdout,"Done!\n"); fflush(NULL);
          Reconstruct.Start();

          // Let's try a check as well -- convert with the 3D code and compare
          /* copy_matrix(work, dens, ngrid3, ngrid[0]);
          fftw_execute(fft);
          for (uint64 j=0; j<ngrid3; j++)
              if (densFFT[j]!=work[j]) {
                  int z = j%ngrid2;
                  int y = j/ngrid2; y=y%ngrid2;
                  int x = j/ngrid[1]/ngrid2;
                  printf("%d %d %d  %f  %f\n", x, y, z, densFFT[j], work[j]);
              }
          */

          /* ------------ computation of shift[] --------------- */
          fprintf(stdout,"# Computing shift[]...");
          /*   shift_x   */
          // load the work matrix with FFT[shift_x[]]
          densFFT_to_work((Complex *)work, (Complex *)densFFT, ngrid, ngrid2,
                                             cell_size, sig_sm, 0, bias);
          // iFFT the result, in place
          IFFT_Execute(ifft, ifftYZ, ifftX, ngrid, ngrid2, work);
          fprintf(stdout,"x...");
          // multipling with norm, copy work[] to shift_x[]
          Copy.Start();
          copy_matrix(shift_x, work, norm, ngrid3, ngrid[0]);
          Copy.Stop();

          /*   shift_y   */
          // load the work matrix with FFT[shift_y[]]
          densFFT_to_work((Complex *)work, (Complex *)densFFT, ngrid, ngrid2,
                                             cell_size, sig_sm, 1, bias);
          // iFFT the result, in place
          IFFT_Execute(ifft, ifftYZ, ifftX, ngrid, ngrid2, work);
          fprintf(stdout,"y...");
          // multipling with norm, copy work[] to shift_y[]
          Copy.Start();
          copy_matrix(shift_y, work, norm, ngrid3, ngrid[0]);
          Copy.Stop();

          /*   shift_z   */
          // load the work matrix with FFT[shift_z[]]
          densFFT_to_work((Complex *)work, (Complex *)densFFT, ngrid, ngrid2,
                                             cell_size, sig_sm, 2, bias);
          // iFFT the result, in place
          IFFT_Execute(ifft, ifftYZ, ifftX, ngrid, ngrid2, work);
          fprintf(stdout,"z...");
          // multipling with norm, copy work[] to shift_z[]
          Copy.Start();
          copy_matrix(shift_z, work, norm, ngrid3, ngrid[0]);
          Copy.Stop();

          fprintf(stdout,"Done!\n");

          /* ------------------- Clean up -------------------*/
          // Free densFFT and Ylm
          free_FFTW(fft, fftYZ, fftX, ifft, ifftYZ, ifftX);

          Reconstruct.Stop();
    } // end reconstruct

      void reconstruct_iterate(Float power_k0, Float sig_sm, Float divi_sm, Float last_sm, Float C_ani,
                                    int ite_times, int ite_switch, Float ite_weight_ini, Float ite_weight_2, Float bias) {
           // Here's where most of the work occurs.
           // This computes the shift_x,y, and z for each grid,

           // Multiply final results by the FFTW normalization
           Float norm = 1.0/ngrid[0]/ngrid[1]/ngrid[2];

           // Allocate the work matrix and dens_s
           initialize_matrix(work, ngrid3, ngrid[0]);
           initialize_matrix(dens_s, ngrid3, ngrid[0]);
           initialize_matrix(dens_tem, ngrid3, ngrid[0]);
           initialize_matrix(dens_tem_2, ngrid3, ngrid[0]);
           initialize_matrix(dens_res, ngrid3, ngrid[0]);
           initialize_matrix(dens_res_now, ngrid3, ngrid[0]);
           initialize_matrix(densFFT, ngrid3, ngrid[0]);

           // Allocate the shift_xyz matrix and the derivatives of them
           if(shift_x == NULL) initialize_matrix(shift_x, ngrid3, ngrid[0]);
           if(shift_y == NULL) initialize_matrix(shift_y, ngrid3, ngrid[0]);
           if(shift_z == NULL) initialize_matrix(shift_z, ngrid3, ngrid[0]);
           if(shift_xx == NULL) initialize_matrix(shift_xx, ngrid3, ngrid[0]);
           if(shift_xy == NULL) initialize_matrix(shift_xy, ngrid3, ngrid[0]);
           if(shift_xz == NULL) initialize_matrix(shift_xz, ngrid3, ngrid[0]);
           if(shift_yx == NULL) initialize_matrix(shift_yx, ngrid3, ngrid[0]);
           if(shift_yy == NULL) initialize_matrix(shift_yy, ngrid3, ngrid[0]);
           if(shift_yz == NULL) initialize_matrix(shift_yz, ngrid3, ngrid[0]);
           initialize_matrix(shift_zx, ngrid3, ngrid[0]);
           initialize_matrix(shift_zy, ngrid3, ngrid[0]);
           initialize_matrix(shift_zz, ngrid3, ngrid[0]);
           initialize_matrix(shift_x_l, ngrid3, ngrid[0]);
           initialize_matrix(shift_y_l, ngrid3, ngrid[0]);
           initialize_matrix(shift_z_l, ngrid3, ngrid[0]);

           // Allocate the shift_r_z matrices
           initialize_matrix(shift_r_z, ngrid3, ngrid[0]);
           initialize_matrix(shift_r_z_l, ngrid3, ngrid[0]);
           initialize_matrix(shift_r_zx, ngrid3, ngrid[0]);
           initialize_matrix(shift_r_zy, ngrid3, ngrid[0]);
           initialize_matrix(shift_r_zz, ngrid3, ngrid[0]);

           // Allocate the count of shift overlap
           initialize_matrix(C_O, ngrid3, ngrid[0]);

           // need to multiply dens[] with a coefficient to get the estimator
           // for actual denstiy field
           Float coef;
           coef = power_k0/(cell_size*cell_size*cell_size)/bias;

           // Multiply the density by the coef and load the dens_s matrix with that
           // In addition, load the work with the 1st guess for the linear density
           Init.Start();
           copy_matrix(dens_tem, dens, coef, ngrid3, ngrid[0]);
           copy_matrix(work, dens_tem, ngrid3, ngrid[0]);
           copy_matrix(dens_res, dens_tem, ngrid3, ngrid[0]);
           Init.Stop();
           /* Setup FFTW */
           fftw_plan fft, fftYZ, fftX, ifft, ifftYZ, ifftX;
           setup_FFTW(fft, fftYZ, fftX, ifft, ifftYZ, ifftX, ngrid, ngrid2, work);

           // FFTW might have destroyed the contents of work; need to restore work[]==dens[]*coef
           // So far, I haven't seen this happen.
           if (dens[1]*coef!=work[1]
                                || dens[1+ngrid[2]]*coef!=work[1+ngrid[2]]
                                || dens[ngrid3-1]*coef!=work[ngrid3-1]) {
               fprintf(stdout, "Restoring work matrix\n");
               Init.Start();
               copy_matrix(work, dens_tem, ngrid3, ngrid[0]);
               Init.Stop();
           }
           Init.Start();
           copy_matrix(dens_s, work, ngrid3, ngrid[0]);
           copy_matrix(dens_now, work, ngrid3, ngrid[0]);
           copy_matrix(dens_res_now, dens_res, ngrid3, ngrid[0]);
           fprintf(stdout,"Done!\n\n"); fflush(NULL);
           Init.Stop();

           /* ------------ Iteration --------------- */
           Float w1;  // iteration weight
           w1 = ite_weight_ini;

           Reconstruct.Start();
           ite_num = 0;
           while(ite_num < ite_times){
             if(ite_num == ite_switch) w1 = ite_weight_2;
             Copy.Start();
             set_matrix(dens, 0.0, ngrid3, ngrid[0]);
             set_matrix(dens_tem, 0.0, ngrid3, ngrid[0]);
             set_matrix(dens_tem_2, 0.0, ngrid3, ngrid[0]);
             Copy.Stop();
             fprintf(stdout,"# Now, the number of iteration is %d\n", ite_num); fflush(NULL);
             fprintf(stdout,"# Smoothing scale is %e\n", sig_sm); fflush(NULL);

             // Now compute the FFT of the density field and conjugate it
             // FFT(work) in place and conjugate it, storing in densFFT
             fprintf(stdout,"# Computing the density FFT..."); fflush(NULL);
             FFT_Execute(fft, fftYZ, fftX, ngrid, ngrid2, work);
             Copy.Start();
             copy_matrix(densFFT, work, ngrid3, ngrid[0]);
             fprintf(stdout,"Done!\n"); fflush(NULL);
             Copy.Stop();

             // Let's try a check as well -- convert with the 3D code and compare
             /* copy_matrix(work, dens, ngrid3, ngrid[0]);
             fftw_execute(fft);
             for (uint64 j=0; j<ngrid3; j++)
                 if (densFFT[j]!=work[j]) {
                     int z = j%ngrid2;
                     int y = j/ngrid2; y=y%ngrid2;
                     int x = j/ngrid[1]/ngrid2;
                     printf("%d %d %d  %f  %f\n", x, y, z, densFFT[j], work[j]);
                 }
             */

             /* ------------ computation of shift[] --------------- */
             fprintf(stdout,"# Computing shift[]...");
             /*   shift_x   */
             // load the work matrix with FFT[shift_x[]]
             densFFT_to_shift((Complex *)work, (Complex *)densFFT, ngrid, ngrid2,
                                                cell_size, sig_sm, 0, C_ani);
             // iFFT the result, in place
             IFFT_Execute(ifft, ifftYZ, ifftX, ngrid, ngrid2, work);
             fprintf(stdout,"x...");
             // multipling with norm, copy work[] to shift_x[]
             Copy.Start();
             copy_matrix(shift_x, work, norm, ngrid3, ngrid[0]);
             Copy.Stop();

             /*   shift_y   */
             // load the work matrix with FFT[shift_y[]]
             densFFT_to_shift((Complex *)work, (Complex *)densFFT, ngrid, ngrid2,
                                                cell_size, sig_sm, 1, C_ani);
             // iFFT the result, in place
             IFFT_Execute(ifft, ifftYZ, ifftX, ngrid, ngrid2, work);
             fprintf(stdout,"y...");
             // multipling with norm, copy work[] to shift_y[]
             Copy.Start();
             copy_matrix(shift_y, work, norm, ngrid3, ngrid[0]);
             Copy.Stop();

             /*   shift_z   */
             // load the work matrix with FFT[shift_z[]]
             densFFT_to_shift((Complex *)work, (Complex *)densFFT, ngrid, ngrid2,
                                                cell_size, sig_sm, 2, C_ani);
             // iFFT the result, in place
             IFFT_Execute(ifft, ifftYZ, ifftX, ngrid, ngrid2, work);
             fprintf(stdout,"z...");
             // multipling with norm, copy work[] to shift_z[]
             Copy.Start();
             copy_matrix(shift_z, work, norm*(1 + lgr_beta), ngrid3, ngrid[0]);
             if(ite_num == (ite_times-1)){
               copy_matrix(shift_r_z, work, norm, ngrid3, ngrid[0]);
             }
             Copy.Stop();

             fprintf(stdout,"Done!\n");

             /* ------------ computation of 1st derivative of shift(1)[]
                                            (2nd derivative of phi[]) --------------- */
             fprintf(stdout,"# Computing 1nd derivative of shift(1)[]...\n");
               /*   shift(1)_xx   */
               // load the work matrix with FFT[shift_xx[]]
               densFFT_to_shift_1d((Complex *)work, (Complex *)densFFT, ngrid, ngrid2,
                            cell_size, sig_sm, 0, 0, C_ani);
               // iFFT the result, in place
               IFFT_Execute(ifft, ifftYZ, ifftX, ngrid, ngrid2, work);
               fprintf(stdout,"# xx...");
               // multipling with norm, copy work[] to shift_xx[]
               Copy.Start();
               copy_matrix(shift_xx, work, norm, ngrid3, ngrid[0]);
               Copy.Stop();

               /*   shift(1)_xy   */
               // load the work matrix with FFT[shift_xy[]]
               densFFT_to_shift_1d((Complex *)work, (Complex *)densFFT, ngrid, ngrid2,
                            cell_size, sig_sm, 0, 1, C_ani);
               // iFFT the result, in place
               IFFT_Execute(ifft, ifftYZ, ifftX, ngrid, ngrid2, work);
               fprintf(stdout,"xy...");
               // multipling with norm, copy work[] to shift_xy[]
               Copy.Start();
               copy_matrix(shift_xy, work, norm, ngrid3, ngrid[0]);
               Copy.Stop();

               /*   shift(1)_xz   */
               // load the work matrix with FFT[shift_xz[]]
               densFFT_to_shift_1d((Complex *)work, (Complex *)densFFT, ngrid, ngrid2,
                            cell_size, sig_sm, 0, 2, C_ani);
               // iFFT the result, in place
               IFFT_Execute(ifft, ifftYZ, ifftX, ngrid, ngrid2, work);
               fprintf(stdout,"xz...\n");
               // multipling with norm, copy work[] to shift_xz[]
               Copy.Start();
               copy_matrix(shift_xz, work, norm, ngrid3, ngrid[0]);
               Copy.Stop();

               /*   shift(1)_yx   */
               // load the work matrix with FFT[shift_yx[]]
               densFFT_to_shift_1d((Complex *)work, (Complex *)densFFT, ngrid, ngrid2,
                            cell_size, sig_sm, 1, 0, C_ani);
               // iFFT the result, in place
               IFFT_Execute(ifft, ifftYZ, ifftX, ngrid, ngrid2, work);
               fprintf(stdout,"# yx...");
               // multipling with norm, copy work[] to shift_yx[]
               Copy.Start();
               copy_matrix(shift_yx, work, norm, ngrid3, ngrid[0]);
               Copy.Stop();

               /*   shift(1)_yy   */
               // load the work matrix with FFT[shift_yy[]]
               densFFT_to_shift_1d((Complex *)work, (Complex *)densFFT, ngrid, ngrid2,
                            cell_size, sig_sm, 1, 1, C_ani);
               // iFFT the result, in place
               IFFT_Execute(ifft, ifftYZ, ifftX, ngrid, ngrid2, work);
               fprintf(stdout,"yy...");
               // multipling with norm, copy work[] to shift_yy[]
               Copy.Start();
               copy_matrix(shift_yy, work, norm, ngrid3, ngrid[0]);
               Copy.Stop();

               /*   shift(1)_yz   */
               // load the work matrix with FFT[shift_yz[]]
               densFFT_to_shift_1d((Complex *)work, (Complex *)densFFT, ngrid, ngrid2,
                            cell_size, sig_sm, 1, 2, C_ani);
               // iFFT the result, in place
               IFFT_Execute(ifft, ifftYZ, ifftX, ngrid, ngrid2, work);
               fprintf(stdout,"yz...\n");
               // multipling with norm, copy work[] to shift_yz[]
               Copy.Start();
               copy_matrix(shift_yz, work, norm, ngrid3, ngrid[0]);
               Copy.Stop();

               /*   shift(1)_zx   */
               // load the work matrix with FFT[shift_zx[]]
               densFFT_to_shift_1d((Complex *)work, (Complex *)densFFT, ngrid, ngrid2,
                            cell_size, sig_sm, 2, 0, C_ani);
               // iFFT the result, in place
               IFFT_Execute(ifft, ifftYZ, ifftX, ngrid, ngrid2, work);
               fprintf(stdout,"# zx...");
               // multipling with norm, copy work[] to shift_zx[]
               Copy.Start();
               copy_matrix(shift_zx, work, norm, ngrid3, ngrid[0]);
               Copy.Stop();

               /*   shift(1)_zy   */
               // load the work matrix with FFT[shift_zy[]]
               densFFT_to_shift_1d((Complex *)work, (Complex *)densFFT, ngrid, ngrid2,
                            cell_size, sig_sm, 2, 1, C_ani);
               // iFFT the result, in place
               IFFT_Execute(ifft, ifftYZ, ifftX, ngrid, ngrid2, work);
               fprintf(stdout,"zy...");
               // multipling with norm, copy work[] to shift_zy[]
               Copy.Start();
               copy_matrix(shift_zy, work, norm, ngrid3, ngrid[0]);
               Copy.Stop();

               /*   shift(1)_zz   */
               // load the work matrix with FFT[shift_zz[]]
               densFFT_to_shift_1d((Complex *)work, (Complex *)densFFT, ngrid, ngrid2,
                            cell_size, sig_sm, 2, 2, C_ani);
               // iFFT the result, in place
               IFFT_Execute(ifft, ifftYZ, ifftX, ngrid, ngrid2, work);
               fprintf(stdout,"zz...");
               // multipling with norm, copy work[] to shift_zz[]
               Copy.Start();
               copy_matrix(shift_zz, work, norm, ngrid3, ngrid[0]);
               Copy.Stop();
               fprintf(stdout,"Done!\n");

               /* ------------ computation of mu_1 (1st)--------------- */
               fprintf(stdout,"# Computing mu_1 (1st)...");
               /*   mu_1: 1st   */
               fprintf(stdout,"mu_1...");
               // multipling with norm, copy work[] to mu_1[]
               mu.Start();
               mu_1_matrix(work, shift_xx, shift_yy, shift_zz,
                                ngrid3, ngrid[0]);
               mu.Stop();
               Copy.Start();
               add_matrix(dens, work, -1.0, ngrid3, ngrid[0]);
               Copy.Stop();
               fprintf(stdout,"Done!\n");

               /* ------------ from S(1) to S(s) --------------- */
               Float coef_rsd;
               coef_rsd = 1.0 + lgr_beta;
               Copy.Start();
               scale_matrix_const(shift_zx, coef_rsd, ngrid3, ngrid[0]);
               scale_matrix_const(shift_zy, coef_rsd, ngrid3, ngrid[0]);
               scale_matrix_const(shift_zz, coef_rsd, ngrid3, ngrid[0]);
               Copy.Stop();

           /* ------------ 2nd order: start --------------- */
           #ifdef SECOND
             /* ------------ computation of densFFT (2nd order) --------------- */
               Copy.Start();
               mu_2_matrix(work, shift_xx, shift_xy, shift_xz,
                                shift_yx, shift_yy, shift_yz, shift_zx, shift_zy, shift_zz,
                                ngrid3, ngrid[0]);
               Copy.Stop();
               // Now compute the FFT of the density field and conjugate it
               // FFT(work) in place and conjugate it, storing in densFFT
               fprintf(stdout,"# Computing the density FFT (2nd order)..."); fflush(NULL);
               FFT_Execute(fft, fftYZ, fftX, ngrid, ngrid2, work);
               Copy.Start();
               copy_matrix(densFFT, work, shift_2nd, ngrid3, ngrid[0]);
               fprintf(stdout,"Done!\n"); fflush(NULL);
               Copy.Stop();

             /* ------------ computation of shift[] (2nd order)--------------- */
             fprintf(stdout,"# Computing shift[] (2nd order)...");
               /*   shift_x   */
               // load the work matrix with FFT[shift_x[]]
               densFFT_to_shift((Complex *)work, (Complex *)densFFT, ngrid, ngrid2,
                                                  cell_size, sig_sm, 0, C_ani);
               // iFFT the result, in place
               IFFT_Execute(ifft, ifftYZ, ifftX, ngrid, ngrid2, work);
               fprintf(stdout,"x...");
               // multipling with norm, copy work[] to shift_x[]
               Copy.Start();
               add_matrix(shift_x, work, norm, ngrid3, ngrid[0]);
               Copy.Stop();

               /*   shift_y   */
               // load the work matrix with FFT[shift_y[]]
               densFFT_to_shift((Complex *)work, (Complex *)densFFT, ngrid, ngrid2,
                                                  cell_size, sig_sm, 1, C_ani);
               // iFFT the result, in place
               IFFT_Execute(ifft, ifftYZ, ifftX, ngrid, ngrid2, work);
               fprintf(stdout,"y...");
               // multipling with norm, copy work[] to shift_y[]
               Copy.Start();
               add_matrix(shift_y, work, norm, ngrid3, ngrid[0]);
               Copy.Stop();

               /*   shift_z   */
               // load the work matrix with FFT[shift_z[]]
               densFFT_to_shift((Complex *)work, (Complex *)densFFT, ngrid, ngrid2,
                                                  cell_size, sig_sm, 2, C_ani);
               // iFFT the result, in place
               IFFT_Execute(ifft, ifftYZ, ifftX, ngrid, ngrid2, work);
               fprintf(stdout,"z...");
               // multipling with norm, copy work[] to shift_z[]
               Copy.Start();
               add_matrix(shift_z, work, norm*(1 + 2.0*lgr_beta), ngrid3, ngrid[0]);
               if(ite_num == (ite_times-1)){
                 add_matrix(shift_r_z, work, norm, ngrid3, ngrid[0]);
               }
               Copy.Stop();
               fprintf(stdout,"Done!\n");

             /* ------------ computation of 1st derivative of shift(2)[] --------------- */
             fprintf(stdout,"# Computing 1nd derivative of shift(2)[]...\n");
               /*   shift(2)_xx   */
               // load the work matrix with FFT[shift_xx[]]
               densFFT_to_shift_1d((Complex *)work, (Complex *)densFFT, ngrid, ngrid2,
                            cell_size, sig_sm, 0, 0, C_ani);
               // iFFT the result, in place
               IFFT_Execute(ifft, ifftYZ, ifftX, ngrid, ngrid2, work);
               fprintf(stdout,"# xx...");
               // multipling with norm, copy work[] to shift_xx[]
               Copy.Start();
               add_matrix(shift_xx, work, norm, ngrid3, ngrid[0]);
               Copy.Stop();

               /*   shift(2)_xy   */
               // load the work matrix with FFT[shift_xy[]]
               densFFT_to_shift_1d((Complex *)work, (Complex *)densFFT, ngrid, ngrid2,
                            cell_size, sig_sm, 0, 1, C_ani);
               // iFFT the result, in place
               IFFT_Execute(ifft, ifftYZ, ifftX, ngrid, ngrid2, work);
               fprintf(stdout,"xy...");
               // multipling with norm, copy work[] to shift_xy[]
               Copy.Start();
               add_matrix(shift_xy, work, norm, ngrid3, ngrid[0]);
               Copy.Stop();

               /*   shift(2)_xz   */
               // load the work matrix with FFT[shift_xz[]]
               densFFT_to_shift_1d((Complex *)work, (Complex *)densFFT, ngrid, ngrid2,
                            cell_size, sig_sm, 0, 2, C_ani);
               // iFFT the result, in place
               IFFT_Execute(ifft, ifftYZ, ifftX, ngrid, ngrid2, work);
               fprintf(stdout,"xz...\n");
               // multipling with norm, copy work[] to shift_xz[]
               Copy.Start();
               add_matrix(shift_xz, work, norm, ngrid3, ngrid[0]);
               Copy.Stop();

               /*   shift(2)_yx   */
               // load the work matrix with FFT[shift_yx[]]
               densFFT_to_shift_1d((Complex *)work, (Complex *)densFFT, ngrid, ngrid2,
                            cell_size, sig_sm, 1, 0, C_ani);
               // iFFT the result, in place
               IFFT_Execute(ifft, ifftYZ, ifftX, ngrid, ngrid2, work);
               fprintf(stdout,"# yx...");
               // multipling with norm, copy work[] to shift_yx[]
               Copy.Start();
               add_matrix(shift_yx, work, norm, ngrid3, ngrid[0]);
               Copy.Stop();

               /*   shift(2)_yy   */
               // load the work matrix with FFT[shift_yy[]]
               densFFT_to_shift_1d((Complex *)work, (Complex *)densFFT, ngrid, ngrid2,
                            cell_size, sig_sm, 1, 1, C_ani);
               // iFFT the result, in place
               IFFT_Execute(ifft, ifftYZ, ifftX, ngrid, ngrid2, work);
               fprintf(stdout,"yy...");
               // multipling with norm, copy work[] to shift_yy[]
               Copy.Start();
               add_matrix(shift_yy, work, norm, ngrid3, ngrid[0]);
               Copy.Stop();

               /*   shift(2)_yz   */
               // load the work matrix with FFT[shift_yz[]]
               densFFT_to_shift_1d((Complex *)work, (Complex *)densFFT, ngrid, ngrid2,
                            cell_size, sig_sm, 1, 2, C_ani);
               // iFFT the result, in place
               IFFT_Execute(ifft, ifftYZ, ifftX, ngrid, ngrid2, work);
               fprintf(stdout,"yz...\n");
               // multipling with norm, copy work[] to shift_yz[]
               Copy.Start();
               add_matrix(shift_yz, work, norm, ngrid3, ngrid[0]);
               Copy.Stop();

               /*   shift(2)_zx   */
               // load the work matrix with FFT[shift_zx[]]
               densFFT_to_shift_1d((Complex *)work, (Complex *)densFFT, ngrid, ngrid2,
                            cell_size, sig_sm, 2, 0, C_ani);
               // iFFT the result, in place
               IFFT_Execute(ifft, ifftYZ, ifftX, ngrid, ngrid2, work);
               fprintf(stdout,"# zx...");
               // multipling with norm, copy work[] to shift_zx[]
               Copy.Start();
               add_matrix(shift_zx, work, norm*(1 + 2.0*lgr_beta), ngrid3, ngrid[0]);
               Copy.Stop();

               /*   shift(2)_zy   */
               // load the work matrix with FFT[shift_zy[]]
               densFFT_to_shift_1d((Complex *)work, (Complex *)densFFT, ngrid, ngrid2,
                            cell_size, sig_sm, 2, 1, C_ani);
               // iFFT the result, in place
               IFFT_Execute(ifft, ifftYZ, ifftX, ngrid, ngrid2, work);
               fprintf(stdout,"zy...");
               // multipling with norm, copy work[] to shift_zy[]
               Copy.Start();
               add_matrix(shift_zy, work, norm*(1 + 2.0*lgr_beta), ngrid3, ngrid[0]);
               Copy.Stop();

               /*   shift(2)_zz   */
               // load the work matrix with FFT[shift_zz[]]
               densFFT_to_shift_1d((Complex *)work, (Complex *)densFFT, ngrid, ngrid2,
                            cell_size, sig_sm, 2, 2, C_ani);
               // iFFT the result, in place
               IFFT_Execute(ifft, ifftYZ, ifftX, ngrid, ngrid2, work);
               fprintf(stdout,"zz...");
               // multipling with norm, copy work[] to shift_zz[]
               Copy.Start();
               add_matrix(shift_zz, work, norm*(1 + 2.0*lgr_beta), ngrid3, ngrid[0]);
               Copy.Stop();
               fprintf(stdout,"Done!\n");
           #endif
           /* ------------ 2nd order: end --------------- */

               // for normal
               /* ------------ computation of mu_1 (full)--------------- */
               fprintf(stdout,"# Computing mu_1 (full)...");
               /*   mu_1: full   */
               fprintf(stdout,"mu_1...");
               // multipling with norm, copy work[] to mu_1[]
               mu.Start();
               mu_1_matrix(work, shift_xx, shift_yy, shift_zz,
                                ngrid3, ngrid[0]);
               mu.Stop();
               Copy.Start();
               add_matrix(dens_tem, work, 1.0, ngrid3, ngrid[0]);
               Copy.Stop();
               fprintf(stdout,"Done!\n");

               /* ------------ computation of mu_2 (full)--------------- */
               fprintf(stdout,"# Computing mu_2 (full)...");
               /*   mu_2: full   */
               fprintf(stdout,"mu_2...");
               // multipling with norm, copy work[] to mu_2[]

               mu.Start();
               mu_2_matrix(work, shift_xx, shift_xy, shift_xz,
                                shift_yx, shift_yy, shift_yz, shift_zx, shift_zy, shift_zz,
                                ngrid3, ngrid[0]);
               mu.Stop();
               Copy.Start();
               add_matrix(dens_tem, work, 1.0, ngrid3, ngrid[0]);
               Copy.Stop();
               fprintf(stdout,"Done!\n");

               /* ------------ computation of mu_3 (full)--------------- */
               fprintf(stdout,"# Computing mu_3 (full)...");
               /*   mu_3: full   */
               fprintf(stdout,"mu_3...");
               // multipling with norm, copy work[] to mu_3[]
               mu.Start();
               mu_3_matrix(work, shift_xx, shift_xy, shift_xz,
                                shift_yx, shift_yy, shift_yz, shift_zx, shift_zy, shift_zz,
                                ngrid3, ngrid[0]);
               mu.Stop();
               Copy.Start();
               add_matrix(dens_tem, work, 1.0, ngrid3, ngrid[0]);
               Copy.Stop();
               fprintf(stdout,"Done!\n");


             /* ------------ computation of guess of the linear density --------------- */
             fprintf(stdout,"# Computing guess of dens_lin...");
             Float dmin = 0.0;
             Float dmax = 0.0;
             num_over[0] = num_over[1] = num_over[2] = num_over[3] = num_over[4] = 0.0;
             num_weight[0] = num_weight[1] = num_weight[2] = num_weight[3] = num_weight[4] = 0.0;
             /*   map: dens_s/(1 + dens_s)   */
             Copy.Start();
             set_matrix(C_O, 0.0, ngrid3, ngrid[0]);
             set_matrix(work, 0.0, ngrid3, ngrid[0]);
             set_matrix(shift_xx, 0.0, ngrid3, ngrid[0]);   //  shift_x
             set_matrix(shift_yx, 0.0, ngrid3, ngrid[0]);   //  shift_y
             set_matrix(shift_zx, 0.0, ngrid3, ngrid[0]);   //  shift_z
             Copy.Stop();
             fprintf(stdout,"map...");
             from_q_to_s(NULL, NULL, 0, -1);
             Copy.Start();
             num_over[0] = count_matrix(C_O, 0.0, ngrid3, ngrid[0]);
             num_over[1] = count_matrix(C_O, 1.0, ngrid3, ngrid[0]);
             num_over[2] = count_matrix(C_O, 2.0, ngrid3, ngrid[0]);
             num_over[3] = count_matrix(C_O, 3.0, ngrid3, ngrid[0]);
             num_over[4] = count_matrix(C_O, 4.0, ngrid3, ngrid[0]);
             num_weight[0] = count_matrix(work, 0.0, ngrid3, ngrid[0]);
             num_weight[1] = count_matrix(work, 1.0, ngrid3, ngrid[0]);
             num_weight[2] = count_matrix(work, 2.0, ngrid3, ngrid[0]);
             num_weight[3] = count_matrix(work, 3.0, ngrid3, ngrid[0]);
             num_weight[4] = count_matrix(work, 4.0, ngrid3, ngrid[0]);
             set_matrix(work, 0.0, ngrid3, ngrid[0]);
             if(ite_num == 0){
               copy_matrix(shift_x_l, shift_x, ngrid3, ngrid[0]);
               copy_matrix(shift_y_l, shift_y, ngrid3, ngrid[0]);
               copy_matrix(shift_z_l, shift_z, ngrid3, ngrid[0]);
             }
             #ifdef RSD
               Float shift_sum_sq_dif = sumsq_dif_matrix(shift_z, shift_z_l, ngrid3, ngrid[0]);
             #else
               Float shift_sum_sq_dif = sumsq_dif_matrix(shift_x, shift_x_l, ngrid3, ngrid[0]);
             #endif
             Copy.Stop();
             from_q_to_s(&dmin, &dmax, 1, 1);
             Copy.Start();

             copy_matrix(shift_x_l, shift_x, ngrid3, ngrid[0]);
             copy_matrix(shift_y_l, shift_y, ngrid3, ngrid[0]);
             copy_matrix(shift_z_l, shift_z, ngrid3, ngrid[0]);

             add_matrix_const(dens, -1.0, ngrid3, ngrid[0]);
             add_matrix_const(dens_tem, 1.0, ngrid3, ngrid[0]);

             add_matrix_const(work, 1.0, ngrid3, ngrid[0]);
             scale_matrix(work, dens_tem, ngrid3, ngrid[0]);
             add_matrix(dens, work, 1.0, ngrid3, ngrid[0]);

             //sub_matrix(dens_res, dens, 1.0, ngrid3, ngrid[0]);
             Float sum_sq_dif = sumsq_dif_matrix(dens, dens_now, ngrid3, ngrid[0]);
             Float sum_sq = sumsq_matrix(dens_s, ngrid3, ngrid[0]);

             // w1*(last[dens]) + (1-w1)*(previous[dens_now])
             //if(ite_num == (ite_times-1)) w1 = 1.0;
             scale_matrix_const(dens, w1, ngrid3, ngrid[0]);
             add_matrix(dens, dens_now, (1.0 - w1), ngrid3, ngrid[0]);

             if(ite_num == ite_times)
             {
               set_matrix(work, 0.0, ngrid3, ngrid[0]);
               from_q_to_s(&dmin, &dmax, 1, 2);
               //add_matrix(dens, work, 1.0, ngrid3, ngrid[0]);
               add_matrix(dens, dens_res, 1.0, ngrid3, ngrid[0]);
             }

             Copy.Stop();
             fprintf(stdout,"Done!\n");
             fprintf(stdout,"#### Minimum of dens_s is %f\n",dmin);
             fprintf(stdout,"#### Maximum of dens_s is %f\n",dmax);
             fprintf(stdout,"#### Number of grid per #overlap(1-5) is %lld, %lld, %lld, %lld, %lld\n",
             num_over[0], num_over[1], num_over[2], num_over[3], num_over[4]);
             fprintf(stdout,"#### Number of grid per #weight(1-5) is %lld, %lld, %lld, %lld, %lld\n",
             num_weight[0], num_weight[1], num_weight[2], num_weight[3], num_weight[4]);
             fprintf(stdout,"#### Mean square difference is %f\n",sqrt(sum_sq_dif/sum_sq));
             fprintf(stdout,"#### Mean square difference (shift) is %f\n\n",sqrt(shift_sum_sq_dif/ngrid3));

             /*   For next roop   */
             // Multiply the density by the coef and load the dens_s matrix with that
             // In addition, load the work with the 1st guess for the linear density
             Copy.Start();
             copy_matrix(dens_now, dens, ngrid3, ngrid[0]);
             copy_matrix(dens_res_now, dens_res, ngrid3, ngrid[0]);
             copy_matrix(work, dens, ngrid3, ngrid[0]);
             Copy.Stop();

             ite_num++;

             #ifdef CHANGE_SM
              sig_sm *= 1.0/divi_sm;
              if(sig_sm <= last_sm) sig_sm = last_sm;
             #endif
          }

          Float sum_dens = sum_matrix(dens_now, ngrid3, ngrid[0]);
          fprintf(stdout,"#### Mean of density field is %f\n\n",sum_dens/ngrid3);

          Float sum_dens_res = sum_matrix(dens_res_now, ngrid3, ngrid[0]);
          fprintf(stdout,"#### Mean of residual density field is %f\n\n",sum_dens_res/ngrid3);

          // to get consistent with the correlation of random (denominator)
          Copy.Start();
          copy_matrix(dens, work, ngrid3, ngrid[0]);
          scale_matrix_const(dens, (1/coef), ngrid3, ngrid[0]);
          Copy.Stop();

          /* ------------------- Clean up -------------------*/
           // Free densFFT and Ylm
           free_FFTW(fft, fftYZ, fftX, ifft, ifftYZ, ifftX);

           set_matrix(work, 0.0, ngrid3, ngrid[0]);
           set_matrix(shift_x_l, 0.0, ngrid3, ngrid[0]);
           set_matrix(shift_x_l, 0.0, ngrid3, ngrid[0]);
           set_matrix(shift_x_l, 0.0, ngrid3, ngrid[0]);
           set_matrix(shift_xx, 0.0, ngrid3, ngrid[0]);
           set_matrix(shift_xy, 0.0, ngrid3, ngrid[0]);
           set_matrix(shift_xz, 0.0, ngrid3, ngrid[0]);
           set_matrix(shift_yx, 0.0, ngrid3, ngrid[0]);
           set_matrix(shift_yy, 0.0, ngrid3, ngrid[0]);
           set_matrix(shift_yz, 0.0, ngrid3, ngrid[0]);
           set_matrix(shift_zx, 0.0, ngrid3, ngrid[0]);
           set_matrix(shift_zy, 0.0, ngrid3, ngrid[0]);
           set_matrix(shift_zz, 0.0, ngrid3, ngrid[0]);
           set_matrix(shift_r_zx, 0.0, ngrid3, ngrid[0]);
           set_matrix(shift_r_zy, 0.0, ngrid3, ngrid[0]);
           set_matrix(shift_r_zz, 0.0, ngrid3, ngrid[0]);

           Reconstruct.Stop();
    } // end reconstruct_iterate

      void from_q_to_s(Float *dmin, Float *dmax, int output, int type) {
        // Set up a small buffer, just to reduce the calls to fread, which seem to be slow
        // on some machines.
        fqts.Start();
        uint64 index_ori, index, nread;
        Float b[4];
        std::vector<Galaxy> grid;
        grid.reserve(MAXGAL);    // Just to cut down on thrashing; it will expand as needed
        index_ori = 0;
        nread = BUFFERSIZE;
        assert(ngrid3 > nread);
        while (index_ori<ngrid3){
          for (int j=0; j<nread; j++) {
              if(dmin != NULL) *dmin = fmin(*dmin,dens_s[index_ori]);
              if(dmax != NULL) *dmax = fmax(*dmax,dens_s[index_ori]);
              uint64 iz_ori = index_ori%ngrid2;
              uint64 iy_ori = ((index_ori - iz_ori)/ngrid2)%ngrid[1];
              uint64 ix_ori = (((index_ori - iz_ori)/ngrid2) - iy_ori)/ngrid[1];

              // b[0]
              b[0] = (Float)ix_ori + shift_x[index_ori]/cell_size;
              if(b[0] < 0.0){ while(b[0] < 0.0) b[0] += (Float)ngrid[0]; }
              if(b[0] >= (Float)ngrid[0]){ while(b[0] >= (Float)ngrid[0]) b[0] -= (Float)ngrid[0]; }

              // b[1]
              b[1] = (Float)iy_ori + shift_y[index_ori]/cell_size;
              if(b[1] < 0.0){ while(b[1] < 0.0) b[1] += (Float)ngrid[1]; }
              if(b[1] >= (Float)ngrid[1]){ while(b[1] >= (Float)ngrid[1]) b[1] -= (Float)ngrid[1]; }

              // b[2]
              if(output == 0 && type >= 5){
                b[2] = (Float)iz_ori + shift_r_z[index_ori]/cell_size; }
              else{
                b[2] = (Float)iz_ori + shift_z[index_ori]/cell_size;
              }
              if(b[2] < 0.0){ while(b[2] < 0.0) b[2] += (Float)ngrid[2]; }
              if(b[2] >= (Float)ngrid[2]){ while(b[2] >= (Float)ngrid[2]) b[2] -= (Float)ngrid[2]; }

              assert(b[0] >= 0.0 && b[0] < (Float)ngrid[0]);
              assert(b[1] >= 0.0 && b[1] < (Float)ngrid[1]);
              assert(b[2] >= 0.0 && b[2] < (Float)ngrid[2]);

              // b[3]
              if(output == 0 && type == -1) b[3] = 1.0;
              else b[3] = (Float)index_ori;    //　output == 1 && type >= 1

              uint64 ix = floor(b[0]);
              uint64 iy = floor(b[1]);
              uint64 iz = floor(b[2]);
              index=(iz)+ngrid2*((iy)+(ix)*ngrid[1]);

              if(iz_ori < ngrid[2]) grid.push_back(Galaxy(b,index));
              if (grid.size()>=MAXGAL) {
                  fqts.Stop();
                  Cloud_In_Cell(grid, 0, output, type, 0);
                  fqts.Start();
              }
              index_ori++;
          }
          if((ngrid3 - index_ori) >= BUFFERSIZE){
              nread = BUFFERSIZE;
          }
          else{
              nread = (ngrid3 - index_ori);
          }
        }
        fqts.Stop();
        Cloud_In_Cell(grid, 0, output, type, 0);
        fqts.Start();

        grid.clear();
        fqts.Stop();
    }

      void print_dens(FILE *fp, int axis, int shift_axis) {
          // Print out the results
          // axis == 0, 1, and 2 correponds to x, y, and z
          // shift_axis == 0, 1, and 2 correponds to shift_x, y, and z
          for (int j=0; j<ngrid[axis]; j++) {
              uint64 ind;
              fprintf(fp,"%1d ", axis);
              fprintf(fp,"%7.4f", (j)*cell_size);
              // dens
              if(axis == 0) ind = ngrid[2]/2 + ngrid2*(ngrid[1]/2 + j*ngrid[1]);
              else if(axis == 1) ind = ngrid[2]/2 + ngrid2*(j + ngrid[0]/2*ngrid[1]);
              else ind = j + ngrid2*(ngrid[1]/2 + ngrid[0]/2*ngrid[1]);
              fprintf(fp," %16.9e", dens_now[ind]);
              //fprintf(fp," %16.9e", dens_res_now[ind]);
              // shift
              if(shift_axis == 0) fprintf(fp," %16.9e", shift_x[ind]);
              //if(shift_axis == 0) fprintf(fp," %16.9e", shift_t_x[ind]);
              else if(shift_axis == 1) fprintf(fp," %16.9e", shift_y[ind]);
              else fprintf(fp," %16.9e", shift_z[ind]);
              fprintf(fp,"\n");
          }
    }

      /* ------------------------  correlation ---------------------------- */

      void correlate(int maxell, Histogram &h, Histogram &kh) {
          // Here's where most of the work occurs.
          // This computes the correlations for each ell, summing over m,
          // and then histograms the result.
          void makeYlm(Float *work, int ell, int m, int n[3], int n1,
                          Float *xcell, Float *ycell, Float *zcell, Float *dens);

          // Multiply total by 4*pi, to match SE15 normalization
          // Include the FFTW normalization
          Float norm = 4.0*M_PI/ngrid[0]/ngrid[1]/ngrid[2];
          Float Pnorm = 4.0*M_PI;
          assert(sep>0);    // This is a check that the submatrix got set up.

          // Allocate the work matrix and load it with the density
          // We do this here so that the array is touched before FFT planning
          initialize_matrix_by_copy(work, ngrid3, ngrid[0], dens);

          // Allocate total[csize**3] and corr[csize**3]
          Float *total=NULL;  initialize_matrix(total,  csize3, csize[0]);
          Float *corr=NULL;   initialize_matrix(corr,   csize3, csize[0]);
          Float *ktotal=NULL; initialize_matrix(ktotal, ksize3, ksize[0]);
          Float *kcorr=NULL;  initialize_matrix(kcorr,  ksize3, ksize[0]);

          /* Setup FFTW */
          fftw_plan fft, fftYZ, fftX, ifft, ifftYZ, ifftX;
          setup_FFTW(fft, fftYZ, fftX, ifft, ifftYZ, ifftX, ngrid, ngrid2, work);

          // FFTW might have destroyed the contents of work; need to restore work[]==dens[]
          // So far, I haven't seen this happen.
          if (dens[1]!=work[1] || dens[1+ngrid[2]]!=work[1+ngrid[2]]
                               || dens[ngrid3-1]!=work[ngrid3-1]) {
              fprintf(stdout, "Restoring work matrix\n");
              Init.Start();
              copy_matrix(work, dens, ngrid3, ngrid[0]);
              Init.Stop();
          }

          Correlate.Start();    // Starting the main work
          // Now compute the FFT of the density field and conjugate it
          // FFT(work) in place and conjugate it, storing in densFFT
          fprintf(stdout,"# Computing the density FFT..."); fflush(NULL);
          FFT_Execute(fft, fftYZ, fftX, ngrid, ngrid2, work);
          Correlate.Stop();    // We're tracking initialization separately
          initialize_matrix_by_copy(densFFT, ngrid3, ngrid[0], work);
          fprintf(stdout,"Done!\n"); fflush(NULL);
          Correlate.Start();

          // Let's try a check as well -- convert with the 3D code and compare
          /* copy_matrix(work, dens, ngrid3, ngrid[0]);
          fftw_execute(fft);
          for (uint64 j=0; j<ngrid3; j++)
              if (densFFT[j]!=work[j]) {
                  int z = j%ngrid2;
                  int y = j/ngrid2; y=y%ngrid2;
                  int x = j/ngrid[1]/ngrid2;
                  printf("%d %d %d  %f  %f\n", x, y, z, densFFT[j], work[j]);
              }
          */

          /* ------------ Loop over ell & m --------------- */
          // Loop over each ell to compute the anisotropic correlations
          for (int ell=0; ell<=maxell; ell+=2) {
              // Initialize the submatrix
              Extract.Start();
              set_matrix(total,0.0, csize3, csize[0]);
              set_matrix(ktotal,0.0, ksize3, ksize[0]);
              Extract.Stop();
              // Loop over m
              for (int m=-ell; m<=ell; m++) {
                  fprintf(stdout,"# Computing %d %2d...", ell, m);
                  // Create the Ylm matrix times dens
                  makeYlm(work, ell, m, ngrid, ngrid2, xcell, ycell, zcell, dens);
                  fprintf(stdout,"Ylm...");

                  // FFT in place
                  FFT_Execute(fft, fftYZ, fftX, ngrid, ngrid2, work);

                  // Multiply by conj(densFFT), as complex numbers
                  AtimesB.Start();
                  multiply_matrix_with_conjugation((Complex *)work,
                                  (Complex *)densFFT, ngrid3/2, ngrid[0]);
                  AtimesB.Stop();

                  // Extract the anisotropic power spectrum
                  // Load the Ylm's and include the CICwindow correction
                  makeYlm(kcorr, ell, m, ksize, ksize[2], kx_cell, ky_cell, kz_cell, CICwindow);
                  // Multiply these Ylm by the power result, and then add to total.
                  extract_submatrix_C2R(ktotal, kcorr, ksize, (Complex *)work, ngrid, ngrid2);

                  // iFFT the result, in place
                  IFFT_Execute(ifft, ifftYZ, ifftX, ngrid, ngrid2, work);
                  fprintf(stdout,"FFT...");

                  // Create Ylm for the submatrix that we'll extract for histogramming
                  // The extra multiplication by one here is of negligible cost, since
                  // this array is so much smaller than the FFT grid.
                  makeYlm(corr, ell, m, csize, csize[2], cx_cell, cy_cell, cz_cell, NULL);

                  // Multiply these Ylm by the correlation result, and then add to total.
                  extract_submatrix(total, corr, csize, work, ngrid, ngrid2);

                  fprintf(stdout,"Done!\n");
              }

              Extract.Start();
              scale_matrix_const(total, norm, csize3, csize[0]);
              scale_matrix_const(ktotal, Pnorm, ksize3, ksize[0]);
              Extract.Stop();
              // Histogram total by rnorm
              Hist.Start();
              h.histcorr(ell, csize3, rnorm, NULL, total);
              kh.histcorr(ell, ksize3, knorm, NULL, ktotal);
              Hist.Stop();

          }

          /* ------------------- Clean up -------------------*/
          // Free densFFT and Ylm
          free(corr);
          free(total);
          free(kcorr);
          free(ktotal);
          free_FFTW(fft, fftYZ, fftX, ifft, ifftYZ, ifftX);

          Correlate.Stop();
    } // end correlate

      /* ------------------------ Wiener filtering for shift ---------------------------- */

      void xcorr_shift(Histogram &kh_n, Histogram &kh_d, Histogram &kh_e, Float power_k0, Float bias) {
          // This computes the x-corr for shift
          // and then histograms the result.

          // Multiply final results by the FFTW normalization
          Float norm = 1.0/ngrid[0]/ngrid[1]/ngrid[2];
          assert(sep>0);    // This is a check that the submatrix got set up.

          // Allocate the work matrix
          if(work == NULL) initialize_matrix(work, ngrid3, ngrid[0]);

          // Allocate shift_xyz_l to [ngrid**2*ngrid2] and set it to zero
          if(shift_x_l == NULL) initialize_matrix(shift_x_l, ngrid3, ngrid[0]);
          if(shift_y_l == NULL) initialize_matrix(shift_y_l, ngrid3, ngrid[0]);
          if(shift_z_l == NULL) initialize_matrix(shift_z_l, ngrid3, ngrid[0]);

          if(shift_r_zx == NULL) initialize_matrix(shift_r_zx, ngrid3, ngrid[0]);
          if(shift_r_zy == NULL) initialize_matrix(shift_r_zy, ngrid3, ngrid[0]);
          if(shift_r_zz == NULL) initialize_matrix(shift_r_zz, ngrid3, ngrid[0]);

          /* Setup FFTW */
          fftw_plan fft, fftYZ, fftX, ifft, ifftYZ, ifftX;
          setup_FFTW(fft, fftYZ, fftX, ifft, ifftYZ, ifftX, ngrid, ngrid2, work);


          // Devide the shift_t_X & shift_X by the number of particles in each grid
          devide_matrix(shift_t_x, shift_xx, zero_c, ngrid3, ngrid[0]);
          fprintf(stdout,"#### Ratio of grids with no particle = %f\n", (double)zero_c/(double)ngrid3);
          devide_matrix(shift_t_y, shift_xy, zero_c, ngrid3, ngrid[0]);
          devide_matrix(shift_t_z, shift_xz, zero_c, ngrid3, ngrid[0]);

          // need to multiply shift_t_ [] with a coefficient to be consistent with the number of grids
          Float coef_s = 1.0;

          // Multiply the shift_t_ [] by the coef_s
          Init.Start();
          scale_matrix_const(shift_t_x, coef_s, ngrid3, ngrid[0]);
          scale_matrix_const(shift_t_y, coef_s, ngrid3, ngrid[0]);
          scale_matrix_const(shift_t_z, coef_s, ngrid3, ngrid[0]);
          Init.Stop();

        #ifdef FROM_DENSITY
          // need to multiply dens[] with a coefficient to get the estimator
          // for actual denstiy field
          Float coef_d = power_k0/(cell_size*cell_size*cell_size);

          // Multiply the density by the coef and load the work matrix with that
          Init.Start();
          copy_matrix(work, dens, coef_d, ngrid3, ngrid[0]);
          Init.Stop();

          //fprintf(stderr,"=%e\n",work[10000]);

          // Now compute the FFT of the density field and conjugate it
          // FFT(work) in place and conjugate it, storing in densFFT
          fprintf(stdout,"# Computing the density FFT..."); fflush(NULL);
          FFT_Execute(fft, fftYZ, fftX, ngrid, ngrid2, work);

          initialize_matrix_by_copy(densFFT, ngrid3, ngrid[0], work);
          fprintf(stdout,"Done!\n"); fflush(NULL);
        #endif

        #ifndef RECONST
          devide_matrix(shift_x, shift_yx, zero_c, ngrid3, ngrid[0]);
          devide_matrix(shift_y, shift_yy, zero_c, ngrid3, ngrid[0]);
          devide_matrix(shift_z, shift_yz, zero_c, ngrid3, ngrid[0]);

          // need to multiply shift_ [] with a coefficient to get to be consistent with shift_t_ []
          Float coef_ini = lgf;
          //Float coef_ini = 1.0;

          // Multiply the shift_ [] by the coef
          Init.Start();
          scale_matrix_const(shift_x, coef_ini, ngrid3, ngrid[0]);
          scale_matrix_const(shift_y, coef_ini, ngrid3, ngrid[0]);
          scale_matrix_const(shift_z, coef_ini*(1 + lgr_f), ngrid3, ngrid[0]);
          Init.Stop();
        #endif

        #if defined(RECONST) && !defined(ITERATION)
          // Multiply the shift_ [] by the coef
          Init.Start();
          scale_matrix_const(shift_z, (1 + lgr_f), ngrid3, ngrid[0]);
          Init.Stop();
        #endif

          // Allocate total[csize**3] and corr[csize**3]
          Float *kwork_n=NULL; initialize_matrix(kwork_n, ksize3, ksize[0]);
          Float *kwork_d=NULL; initialize_matrix(kwork_d, ksize3, ksize[0]);
          Float *kwork_e=NULL; initialize_matrix(kwork_e, ksize3, ksize[0]);
          Float *ktotal_n=NULL; initialize_matrix(ktotal_n, ksize3, ksize[0]);
          Float *ktotal_d=NULL; initialize_matrix(ktotal_d,  ksize3, ksize[0]);
          Float *ktotal_e=NULL; initialize_matrix(ktotal_e,  ksize3, ksize[0]);

          Correlate.Start();

          /* ------------ computation of the mean length of shifts --------------- */

          Float sum_nosm_x = sum_matrix(shift_x, ngrid3, ngrid[0]);
          Float sum_nosm_y = sum_matrix(shift_y, ngrid3, ngrid[0]);
          Float sum_nosm_z = sum_matrix(shift_z, ngrid3, ngrid[0]);

          Float sum_true_x = sum_matrix(shift_t_x, ngrid3, ngrid[0]);
          Float sum_true_y = sum_matrix(shift_t_y, ngrid3, ngrid[0]);
          Float sum_true_z = sum_matrix(shift_t_z, ngrid3, ngrid[0]);

          Float sum_sq_nosm_x = sumsq_matrix(shift_x, ngrid3, ngrid[0]);
          Float sum_sq_nosm_y = sumsq_matrix(shift_y, ngrid3, ngrid[0]);
          Float sum_sq_nosm_z = sumsq_matrix(shift_z, ngrid3, ngrid[0]);

          Float sum_sq_true_x = sumsq_matrix(shift_t_x, ngrid3, ngrid[0]);
          Float sum_sq_true_y = sumsq_matrix(shift_t_y, ngrid3, ngrid[0]);
          Float sum_sq_true_z = sumsq_matrix(shift_t_z, ngrid3, ngrid[0]);

          fprintf(stdout,"#### Mean of shift_x (true) = %f\n",sum_true_x/ngrid3);
          fprintf(stdout,"#### Mean of shift_y (true) = %f\n",sum_true_y/ngrid3);
          fprintf(stdout,"#### Mean of shift_z (true) = %f\n",sum_true_z/ngrid3);
          fprintf(stdout,"#### Mean of shift_x (no_sm) = %f\n",sum_nosm_x/ngrid3);
          fprintf(stdout,"#### Mean of shift_y (no_sm) = %f\n",sum_nosm_y/ngrid3);
          fprintf(stdout,"#### Mean of shift_z (no_sm) = %f\n",sum_nosm_z/ngrid3);

          fprintf(stdout,"#### Mean length of shift_x (true) = %f\n",
                                                            sqrt(sum_sq_true_x/ngrid3));
          fprintf(stdout,"#### Mean length of shift_y (true) = %f\n",
                                                            sqrt(sum_sq_true_y/ngrid3));
          fprintf(stdout,"#### Mean length of shift_z (true) = %f\n",
                                                            sqrt(sum_sq_true_z/ngrid3));
          fprintf(stdout,"#### Mean length of shift_x (no_sm) = %f\n",
                                                            sqrt(sum_sq_nosm_x/ngrid3));
          fprintf(stdout,"#### Mean length of shift_y (no_sm) = %f\n",
                                                            sqrt(sum_sq_nosm_y/ngrid3));
          fprintf(stdout,"#### Mean length of shift_z (no_sm) = %f\n",
                                                            sqrt(sum_sq_nosm_z/ngrid3));
          fprintf(stdout,"#### Ratio (true/no_sm) of the mean length of shift_x = %f\n",
                                                            sqrt(sum_sq_true_x/sum_sq_nosm_x));
          fprintf(stdout,"#### Ratio (true/no_sm) of the mean length of shift_y = %f\n",
                                                            sqrt(sum_sq_true_y/sum_sq_nosm_y));
          fprintf(stdout,"#### Ratio (true/no_sm) of the mean length of shift_z = %f\n",
                                                            sqrt(sum_sq_true_z/sum_sq_nosm_z));

        /*
          fprintf(stderr,"=%e\n",shift_x[10000]);
          fprintf(stderr,"=%e\n",shift_t_x[10000]);

          fprintf(stderr,"=%e\n",shift_y[10000]);
          fprintf(stderr,"=%e\n",shift_t_y[10000]);

          fprintf(stderr,"=%e\n",shift_z[10000]);
          fprintf(stderr,"=%e\n",shift_t_z[10000]);
        */

        #ifndef FROM_DENSITY
          /* ------------ computation of shift(no_smooth)[] --------------- */
          fprintf(stdout,"# Computing shift(no_smooth)[]...");
          /*   shift_x   */
          copy_matrix(work, shift_x, ngrid3, ngrid[0]);

          // FFT(work) in place
          FFT_Execute(fft, fftYZ, fftX, ngrid, ngrid2, work);
          fprintf(stdout,"FFT x...");
          // copy work[] to shift_x[]
          Copy.Start();
          copy_matrix(shift_x, work, norm, ngrid3, ngrid[0]);
          copy_matrix(shift_x_l, work, norm, ngrid3, ngrid[0]);
          Copy.Stop();

          /*   shift_y   */
          copy_matrix(work, shift_y, ngrid3, ngrid[0]);
          // FFT(work) in place
          FFT_Execute(fft, fftYZ, fftX, ngrid, ngrid2, work);
          fprintf(stdout,"y...");
          // copy work[] to shift_y[]
          Copy.Start();
          copy_matrix(shift_y, work, norm, ngrid3, ngrid[0]);
          copy_matrix(shift_y_l, work, norm, ngrid3, ngrid[0]);
          Copy.Stop();

          /*   shift_z   */
          copy_matrix(work, shift_z, ngrid3, ngrid[0]);
          // FFT(work) in place
          FFT_Execute(fft, fftYZ, fftX, ngrid, ngrid2, work);
          fprintf(stdout,"z...");
          // copy work[] to shift_z[]
          Copy.Start();
          copy_matrix(shift_z, work, norm, ngrid3, ngrid[0]);
          copy_matrix(shift_z_l, work, norm, ngrid3, ngrid[0]);
          Copy.Stop();

          fprintf(stdout,"Done!\n");

        #else
          /* ------------ computation of shift(no_smooth)[] --------------- */
          fprintf(stdout,"# Computing shift(no_smooth)[]...");
          /*   shift_x   */
          // load the work matrix with FFT[shift_x[]]
          densFFT_to_work((Complex *)work, (Complex *)densFFT, ngrid, ngrid2,
                                             cell_size, 0.0, 0, bias);
          fprintf(stdout,"x...");
          // multipling with norm, copy work[] to shift_x[]
          Copy.Start();
          copy_matrix(shift_x, work, norm, ngrid3, ngrid[0]);
          copy_matrix(shift_x_l, work, norm, ngrid3, ngrid[0]);
          Copy.Stop();

          /*   shift_y   */
          // load the work matrix with FFT[shift_y[]]
          densFFT_to_work((Complex *)work, (Complex *)densFFT, ngrid, ngrid2,
                                             cell_size, 0.0, 1, bias);
          fprintf(stdout,"y...");
          // multipling with norm, copy work[] to shift_y[]
          Copy.Start();
          copy_matrix(shift_y, work, norm, ngrid3, ngrid[0]);
          copy_matrix(shift_y_l, work, norm, ngrid3, ngrid[0]);
          Copy.Stop();

          /*   shift_z   */
          // load the work matrix with FFT[shift_z[]]
          densFFT_to_work((Complex *)work, (Complex *)densFFT, ngrid, ngrid2,
                                             cell_size, 0.0, 2, bias);
          fprintf(stdout,"z...");
          // multipling with norm, copy work[] to shift_z[]
          Copy.Start();
          copy_matrix(shift_z, work, norm*(1 + lgr_f), ngrid3, ngrid[0]);
          copy_matrix(shift_z_l, work, norm*(1 + lgr_f), ngrid3, ngrid[0]);
          Copy.Stop();

          fprintf(stdout,"Done!\n");
        #endif

          /* ------------ computation of shift(true)[] --------------- */
          fprintf(stdout,"# Computing shift(true)[]...");
          /*   shift_x   */
          copy_matrix(work, shift_t_x, ngrid3, ngrid[0]);
          // FFT(work) in place
          FFT_Execute(fft, fftYZ, fftX, ngrid, ngrid2, work);
          fprintf(stdout,"FFT x...");
          // copy work[] to shift_x[]
          Copy.Start();
          copy_matrix(shift_t_x, work, norm, ngrid3, ngrid[0]);
          copy_matrix(shift_r_zx, work, norm, ngrid3, ngrid[0]);
          Copy.Stop();

          /*   shift_y   */
          copy_matrix(work, shift_t_y, ngrid3, ngrid[0]);
          // FFT(work) in place
          FFT_Execute(fft, fftYZ, fftX, ngrid, ngrid2, work);
          fprintf(stdout,"y...");
          // copy work[] to shift_y[]
          Copy.Start();
          copy_matrix(shift_t_y, work, norm, ngrid3, ngrid[0]);
          copy_matrix(shift_r_zy, work, norm, ngrid3, ngrid[0]);
          Copy.Stop();

          /*   shift_z   */
          copy_matrix(work, shift_t_z, ngrid3, ngrid[0]);
          // FFT(work) in place
          FFT_Execute(fft, fftYZ, fftX, ngrid, ngrid2, work);
          fprintf(stdout,"z...");
          // copy work[] to shift_z[]
          Copy.Start();
          copy_matrix(shift_t_z, work, norm, ngrid3, ngrid[0]);
          copy_matrix(shift_r_zz, work, norm, ngrid3, ngrid[0]);
          Copy.Stop();

          fprintf(stdout,"Done!\n");



        /*
          fprintf(stderr,"=%e\n",shift_x[ngrid[0]*ngrid[1]*ngrid2-8*ngrid[1]*ngrid2]);
          fprintf(stderr,"=%e\n",shift_x[ngrid[0]*ngrid[1]*ngrid2-8*ngrid[1]*ngrid2+1]);
          fprintf(stderr,"=%e\n",shift_t_x[0]);
          fprintf(stderr,"=%e\n",shift_t_x[1]);
        */

          /* ------------ computation of the mean length of shifts --------------- */

          Float sum_sq_nosm_k_x = sumsq_matrix(shift_x, ngrid3, ngrid[0]);
          Float sum_sq_nosm_k_y = sumsq_matrix(shift_y, ngrid3, ngrid[0]);
          Float sum_sq_nosm_k_z = sumsq_matrix(shift_z, ngrid3, ngrid[0]);

          Float sum_sq_true_k_x = sumsq_matrix(shift_t_x, ngrid3, ngrid[0]);
          Float sum_sq_true_k_y = sumsq_matrix(shift_t_y, ngrid3, ngrid[0]);
          Float sum_sq_true_k_z = sumsq_matrix(shift_t_z, ngrid3, ngrid[0]);

          fprintf(stdout,"#### Mean length of shift_k_x (true) = %f\n",
                                                            sqrt(sum_sq_true_k_x/ngrid3));
          fprintf(stdout,"#### Mean length of shift_k_y (true) = %f\n",
                                                            sqrt(sum_sq_true_k_y/ngrid3));
          fprintf(stdout,"#### Mean length of shift_k_z (true) = %f\n",
                                                            sqrt(sum_sq_true_k_z/ngrid3));
          fprintf(stdout,"#### Mean length of shift_k_x (no_sm) = %f\n",
                                                            sqrt(sum_sq_nosm_k_x/ngrid3));
          fprintf(stdout,"#### Mean length of shift_k_y (no_sm) = %f\n",
                                                            sqrt(sum_sq_nosm_k_y/ngrid3));
          fprintf(stdout,"#### Mean length of shift_k_z (no_sm) = %f\n",
                                                            sqrt(sum_sq_nosm_k_z/ngrid3));
          fprintf(stdout,"#### Ratio (true/no_sm) of the mean length of shift_k_x = %f\n",
                                                            sqrt(sum_sq_true_k_x/sum_sq_nosm_k_x));
          fprintf(stdout,"#### Ratio (true/no_sm) of the mean length of shift_k_y = %f\n",
                                                            sqrt(sum_sq_true_k_y/sum_sq_nosm_k_y));
          fprintf(stdout,"#### Ratio (true/no_sm) of the mean length of shift_k_z = %f\n",
                                                            sqrt(sum_sq_true_k_z/sum_sq_nosm_k_z));

          /* ------------ computation of Wiener filtering --------------- */
          // Initialize the submatrix
          Extract.Start();
          set_matrix(kwork_n, 1.0, ksize3, ksize[0]);
          set_matrix(kwork_d, 1.0, ksize3, ksize[0]);
          set_matrix(kwork_e, 1.0, ksize3, ksize[0]);
          set_matrix(ktotal_n, 0.0, ksize3, ksize[0]);
          set_matrix(ktotal_d, 0.0, ksize3, ksize[0]);
          set_matrix(ktotal_e, 0.0, ksize3, ksize[0]);
          set_matrix(work, 0.0, ngrid3, ngrid[0]);
          Extract.Stop();

          fprintf(stdout,"# Computing numerator...");
          /*   x   */
          // Multiply shift_t_x(true) by conj[shift_x(no smoothed)], as complex numbers
          AtimesB.Start();
          multiply_matrix_with_conjugation((Complex *)shift_t_x,
                          (Complex *)shift_x_l, ngrid3/2, ngrid[0]);
          AtimesB.Stop();
          fprintf(stdout,"x...");
          add_matrix(work, shift_t_x, 1.0, ngrid3, ngrid[0]);

          /*   y   */
          // Multiply shift_t_y(true) by conj[shift_y(no smoothed)], as complex numbers
          AtimesB.Start();
          multiply_matrix_with_conjugation((Complex *)shift_t_y,
                          (Complex *)shift_y_l, ngrid3/2, ngrid[0]);
          AtimesB.Stop();
          fprintf(stdout,"y...");
          add_matrix(work, shift_t_y, 1.0, ngrid3, ngrid[0]);

          /*   z   */
          // Multiply shift_t_z(true) by conj[shift_z(no smoothed)], as complex numbers
          AtimesB.Start();
          multiply_matrix_with_conjugation((Complex *)shift_t_z,
                          (Complex *)shift_z_l, ngrid3/2, ngrid[0]);
          AtimesB.Stop();
          fprintf(stdout,"z...");
          add_matrix(work, shift_t_z, 1.0, ngrid3, ngrid[0]);

          // Extract the numerator, as real numbers
          // Multiply kwork_n (all = 1.0) by work, and then add to total.
          extract_submatrix_C2R(ktotal_n, kwork_n, ksize, (Complex *)work, ngrid, ngrid2);
          fprintf(stdout,"Done!\n");

          Extract.Start();
          set_matrix(work, 0.0, ngrid3, ngrid[0]);
          Extract.Stop();

          fprintf(stdout,"# Computing denominator (no_smooth)");
          /*   x   */
          // Multiply shift_x(no smoothed) by conj[shift_x(no smoothed)], as complex numbers
          AtimesB.Start();
          multiply_matrix_with_conjugation((Complex *)shift_x,
                          (Complex *)shift_x_l, ngrid3/2, ngrid[0]);
          AtimesB.Stop();
          fprintf(stdout,"x...");
          add_matrix(work, shift_x, 1.0, ngrid3, ngrid[0]);

          /*   y   */
          // Multiply shift_y(no smoothed) by conj[shift_y(no smoothed)], as complex numbers
          AtimesB.Start();
          multiply_matrix_with_conjugation((Complex *)shift_y,
                          (Complex *)shift_y_l, ngrid3/2, ngrid[0]);
          AtimesB.Stop();
          fprintf(stdout,"y...");
          add_matrix(work, shift_y, 1.0, ngrid3, ngrid[0]);

          /*   z   */
          // Multiply shift_z(no smoothed) by conj[shift_z(no smoothed)], as complex numbers
          AtimesB.Start();
          multiply_matrix_with_conjugation((Complex *)shift_z,
                          (Complex *)shift_z_l, ngrid3/2, ngrid[0]);
          AtimesB.Stop();
          fprintf(stdout,"z...");
          add_matrix(work, shift_z, 1.0, ngrid3, ngrid[0]);

          // Extract the numerator, as real numbers
          // Multiply kwork_d (all = 1.0) by work, and then add to total.
          extract_submatrix_C2R(ktotal_d, kwork_d, ksize, (Complex *)work, ngrid, ngrid2);
          fprintf(stdout,"Done!\n");

          Extract.Start();
          set_matrix(work, 0.0, ngrid3, ngrid[0]);
          Extract.Stop();

          fprintf(stdout,"# Computing denominator (true)");
          /*   x   */
          // Multiply shift_x(no smoothed) by conj[shift_x(no smoothed)], as complex numbers
          AtimesB.Start();
          multiply_matrix_with_conjugation((Complex *)shift_r_zx,
                          (Complex *)shift_r_zx, ngrid3/2, ngrid[0]);
          AtimesB.Stop();
          fprintf(stdout,"x...");
          add_matrix(work, shift_r_zx, 1.0, ngrid3, ngrid[0]);

          /*   y   */
          // Multiply shift_y(no smoothed) by conj[shift_y(no smoothed)], as complex numbers
          AtimesB.Start();
          multiply_matrix_with_conjugation((Complex *)shift_r_zy,
                          (Complex *)shift_r_zy, ngrid3/2, ngrid[0]);
          AtimesB.Stop();
          fprintf(stdout,"y...");
          add_matrix(work, shift_r_zy, 1.0, ngrid3, ngrid[0]);

          /*   z   */
          // Multiply shift_z(no smoothed) by conj[shift_z(no smoothed)], as complex numbers
          AtimesB.Start();
          multiply_matrix_with_conjugation((Complex *)shift_r_zz,
                          (Complex *)shift_r_zz, ngrid3/2, ngrid[0]);
          AtimesB.Stop();
          fprintf(stdout,"z...");
          add_matrix(work, shift_r_zz, 1.0, ngrid3, ngrid[0]);

          // Extract the numerator, as real numbers
          // Multiply kwork_e (all = 1.0) by work, and then add to total.
          extract_submatrix_C2R(ktotal_e, kwork_e, ksize, (Complex *)work, ngrid, ngrid2);
          fprintf(stdout,"Done!\n");

          // Histogram total by knorm
          Hist.Start();
          kh_n.histcorr(0, ksize3, knorm, k_znorm, ktotal_n);
          kh_d.histcorr(0, ksize3, knorm, k_znorm, ktotal_d);
          kh_e.histcorr(0, ksize3, knorm, k_znorm, ktotal_e);
          Hist.Stop();

          /* ------------------- Clean up -------------------*/
          // Free densFFT and Ylm
          free(kwork_n);
          free(kwork_d);
          free(kwork_e);
          free(ktotal_n);
          free(ktotal_d);
          free(ktotal_e);
          free_FFTW(fft, fftYZ, fftX, ifft, ifftYZ, ifftX);

          Correlate.Stop();
    } // end xcorr_shift

      /* ------------------------ X-correlation of density fields ---------------------------- */

      void xcorr_dens(Histogram &kh_n, Histogram &kh_d, Histogram &kh_e, Float power_k0, Float bias) {
          // This computes the x-corr for density fields
          // and then histograms the result.

          // Multiply final results by the FFTW normalization
          Float norm = 1.0/ngrid[0]/ngrid[1]/ngrid[2];
          assert(sep>0);    // This is a check that the submatrix got set up.

          // Allocate the work matrix
          if(work == NULL) initialize_matrix(work, ngrid3, ngrid[0]);
          if(dens_res == NULL) initialize_matrix(dens_res, ngrid3, ngrid[0]);
          if(dens_res_now == NULL) initialize_matrix(dens_res_now, ngrid3, ngrid[0]);

          /* Setup FFTW */
          fftw_plan fft, fftYZ, fftX, ifft, ifftYZ, ifftX;
          setup_FFTW(fft, fftYZ, fftX, ifft, ifftYZ, ifftX, ngrid, ngrid2, work);

          // need to multiply dens[] with a coefficient to get the estimator
          // for actual denstiy field
          Float coef_d = power_k0/(cell_size*cell_size*cell_size);

          // need to multiply dens_ [] with a coefficient to get to be consistent
          Float coef_ini = coef_d*lgf;
          // Multiply the density by the coef and load the work matrix with that
          Init.Start();
          copy_matrix(dens, dens, coef_ini, ngrid3, ngrid[0]);
          copy_matrix(dens_now, dens_now, coef_d, ngrid3, ngrid[0]);
          Init.Stop();

          // Allocate total[csize**3] and corr[csize**3]
          Float *kwork_n=NULL; initialize_matrix(kwork_n, ksize3, ksize[0]);
          Float *kwork_d=NULL; initialize_matrix(kwork_d, ksize3, ksize[0]);
          Float *kwork_e=NULL; initialize_matrix(kwork_e, ksize3, ksize[0]);
          Float *ktotal_n=NULL; initialize_matrix(ktotal_n, ksize3, ksize[0]);
          Float *ktotal_d=NULL; initialize_matrix(ktotal_d,  ksize3, ksize[0]);
          Float *ktotal_e=NULL; initialize_matrix(ktotal_e,  ksize3, ksize[0]);

          Correlate.Start();

          /* ------------ computation of dens_now(present)[] --------------- */
          fprintf(stdout,"# Computing dens_now(present)[]...");
          copy_matrix(work, dens_now, ngrid3, ngrid[0]);
          // FFT(work) in place
          FFT_Execute(fft, fftYZ, fftX, ngrid, ngrid2, work);
          fprintf(stdout,"FFT...");
          // copy work[] to dens_now[]
          Copy.Start();
          copy_matrix(dens_now, work, norm, ngrid3, ngrid[0]);
          copy_matrix(dens_res_now, work, norm, ngrid3, ngrid[0]);
          Copy.Stop();

          fprintf(stdout,"Done!\n");


          /* ------------ computation of dens(initial)[] --------------- */
          fprintf(stdout,"# Computing dens(initial)[]...");
          copy_matrix(work, dens, ngrid3, ngrid[0]);
          // FFT(work) in place
          FFT_Execute(fft, fftYZ, fftX, ngrid, ngrid2, work);
          fprintf(stdout,"FFT...");
          // copy work[] to dens[]
          Copy.Start();
          copy_matrix(dens, work, norm, ngrid3, ngrid[0]);
          copy_matrix(dens_res, work, norm, ngrid3, ngrid[0]);
          Copy.Stop();

          fprintf(stdout,"Done!\n");


          /* ------------ computation of Wiener filtering --------------- */
          // Initialize the submatrix
          Extract.Start();
          set_matrix(kwork_n, 1.0, ksize3, ksize[0]);
          set_matrix(kwork_d, 1.0, ksize3, ksize[0]);
          set_matrix(kwork_e, 1.0, ksize3, ksize[0]);
          set_matrix(ktotal_n, 0.0, ksize3, ksize[0]);
          set_matrix(ktotal_d, 0.0, ksize3, ksize[0]);
          set_matrix(ktotal_e, 0.0, ksize3, ksize[0]);
          set_matrix(work, 0.0, ngrid3, ngrid[0]);
          Extract.Stop();

          fprintf(stdout,"# Computing numerator...");
          // Multiply dens(initial) by conj[dens_now(present)], as complex numbers
          AtimesB.Start();
          multiply_matrix_with_conjugation((Complex *)dens,
                          (Complex *)dens_now, ngrid3/2, ngrid[0]);
          AtimesB.Stop();
          add_matrix(work, dens, 1.0, ngrid3, ngrid[0]);

          // Extract the numerator, as real numbers
          // Multiply kwork_n (all = 1.0) by work, and then add to total.
          extract_submatrix_C2R(ktotal_n, kwork_n, ksize, (Complex *)work, ngrid, ngrid2);
          fprintf(stdout,"Done!\n");

          Extract.Start();
          set_matrix(work, 0.0, ngrid3, ngrid[0]);
          Extract.Stop();

          fprintf(stdout,"# Computing denominator (present)");
          // Multiply dens_now(present) by conj[dens_now(present)], as complex numbers
          AtimesB.Start();
          multiply_matrix_with_conjugation((Complex *)dens_res_now,
                          (Complex *)dens_res_now, ngrid3/2, ngrid[0]);
          AtimesB.Stop();
          fprintf(stdout,"x...");
          add_matrix(work, dens_res_now, 1.0, ngrid3, ngrid[0]);

          // Extract the numerator, as real numbers
          // Multiply kwork_d (all = 1.0) by work, and then add to total.
          extract_submatrix_C2R(ktotal_d, kwork_d, ksize, (Complex *)work, ngrid, ngrid2);
          fprintf(stdout,"Done!\n");

          Extract.Start();
          set_matrix(work, 0.0, ngrid3, ngrid[0]);
          Extract.Stop();

          fprintf(stdout,"# Computing denominator (initial)");
          // Multiply dens(initial) by conj[dens(initial)], as complex numbers
          AtimesB.Start();
          multiply_matrix_with_conjugation((Complex *)dens_res,
                          (Complex *)dens_res, ngrid3/2, ngrid[0]);
          AtimesB.Stop();
          fprintf(stdout,"x...");
          add_matrix(work, dens_res, 1.0, ngrid3, ngrid[0]);

          // Extract the numerator, as real numbers
          // Multiply kwork_e (all = 1.0) by work, and then add to total.
          extract_submatrix_C2R(ktotal_e, kwork_e, ksize, (Complex *)work, ngrid, ngrid2);
          fprintf(stdout,"Done!\n");

          // Histogram total by knorm
          Hist.Start();
          kh_n.histcorr(0, ksize3, knorm, k_znorm, ktotal_n);
          kh_d.histcorr(0, ksize3, knorm, k_znorm, ktotal_d);
          kh_e.histcorr(0, ksize3, knorm, k_znorm, ktotal_e);
          Hist.Stop();

          /* ------------------- Clean up -------------------*/
          // Free densFFT and Ylm
          free(kwork_n);
          free(kwork_d);
          free(kwork_e);
          free(ktotal_n);
          free(ktotal_d);
          free(ktotal_e);
          free_FFTW(fft, fftYZ, fftX, ifft, ifftYZ, ifftX);

          Correlate.Stop();
    } // end xcorr

      void print_kcorr(FILE *fp, Histogram &kh_n, Histogram &kh_d, Histogram &kh_e, int norm) {

          // Print out the results
          // If norm==1, divide by counts
          Float denom;

          for (int j=0; j<kh_n.nbins; j++) {
              fprintf(fp,"%1d ", norm);
              // all-sky
              if (kh_n.sep>2)
                  fprintf(fp,"%6.2f %8.0f", (j+0.5)*kh_n.binsize, kh_n.cnt[j]);
              else
                  fprintf(fp,"%7.4f %8.0f", (j+0.5)*kh_n.binsize, kh_n.cnt[j]);
              if (kh_n.cnt[j]!=0&&norm) denom = kh_n.cnt[j]; else denom = 1.0;
              fprintf(fp," %16.9e", kh_n.hist[j]/denom);
              fprintf(fp," %16.9e", kh_d.hist[j]/denom);
              fprintf(fp," %16.9e", kh_e.hist[j]/denom);
              fprintf(fp," %16.9e", kh_n.hist[j]/kh_d.hist[j]);
              fprintf(fp," %16.9e", kh_n.hist[j]/kh_e.hist[j]);
              fprintf(fp," %16.9e", kh_n.hist[j]/(sqrt(kh_d.hist[j])*sqrt(kh_e.hist[j])));

              // wedge-1 (k_z/k > 2/3)
              fprintf(fp,"%8.0f", kh_n.cnt_w1[j]);
              if (kh_n.cnt_w1[j]!=0&&norm) denom = kh_n.cnt_w1[j]; else denom = 1.0;
              fprintf(fp," %16.9e", kh_n.hist_w1[j]/denom);
              fprintf(fp," %16.9e", kh_d.hist_w1[j]/denom);
              fprintf(fp," %16.9e", kh_e.hist_w1[j]/denom);
              fprintf(fp," %16.9e", kh_n.hist_w1[j]/kh_d.hist_w1[j]);
              fprintf(fp," %16.9e", kh_n.hist_w1[j]/kh_e.hist_w1[j]);

              // wedge-2 (2/3 >= k_z/k > 1/3)
              fprintf(fp,"%8.0f", kh_n.cnt_w2[j]);
              if (kh_n.cnt_w2[j]!=0&&norm) denom = kh_n.cnt_w2[j]; else denom = 1.0;
              fprintf(fp," %16.9e", kh_n.hist_w2[j]/denom);
              fprintf(fp," %16.9e", kh_d.hist_w2[j]/denom);
              fprintf(fp," %16.9e", kh_e.hist_w2[j]/denom);
              fprintf(fp," %16.9e", kh_n.hist_w2[j]/kh_d.hist_w2[j]);
              fprintf(fp," %16.9e", kh_n.hist_w2[j]/kh_e.hist_w2[j]);

              // wedge-3 (1/3 >= k_z/k)
              fprintf(fp,"%8.0f", kh_n.cnt_w3[j]);
              if (kh_n.cnt_w3[j]!=0&&norm) denom = kh_n.cnt_w3[j]; else denom = 1.0;
              fprintf(fp," %16.9e", kh_n.hist_w3[j]/denom);
              fprintf(fp," %16.9e", kh_d.hist_w3[j]/denom);
              fprintf(fp," %16.9e", kh_e.hist_w3[j]/denom);
              fprintf(fp," %16.9e", kh_n.hist_w3[j]/kh_d.hist_w3[j]);
              fprintf(fp," %16.9e", kh_n.hist_w3[j]/kh_e.hist_w3[j]);
              fprintf(fp,"\n");
          }
    }

};    // end Grid


/* =========================================================================== */

//  because Sorting & Mergin are done twice
Float IO_1, CIC_1, Sorting_1, Merging_1, FFTonly_1, FFTyz_1, FFTx_1,
              CIC_2, Sorting_2, Merging_2;

void ReportTimes(FILE *fp, uint64 nfft, uint64 ngrid3, uint64 cnt) {
    fflush(NULL);
    fprintf(fp, "#\n# Timing Report: \n");
    fprintf(fp, "# Setup time:       %8.4f s\n", Setup.Elapsed());
    fprintf(fp, "# I/O time:         %8.4f s, %6.3f Mparticles/sec, %6.2f MB/sec Read\n",
            IO_1, cnt/IO_1/1e6, cnt/IO_1*32.0/1e6);
    fprintf(fp, "# CIC Grid time:    %8.4f s, %6.3f Mparticles/sec, %6.2f GB/sec\n", CIC_1,
            cnt/CIC_1/1e6, 1e-9*cnt/CIC_1*27.0*2.0*sizeof(Float));
    fprintf(fp, "#        Sorting:   %8.4f s\n", Sorting_1);
    fprintf(fp, "#        Merging:   %8.4f s\n", Merging_1);
    fprintf(fp, "#        CIC:       %8.4f s\n", CIC_1-Merging_1-Sorting_1);
    fprintf(fp, "# FFTW Prep time:   %8.4f s\n", FFTW.Elapsed());
    fprintf(fp, "# Array Init time:  %8.4f s, %6.3f GB/s\n", Init.Elapsed(),
                    1e-9*ngrid3*sizeof(Float)*5/Init.Elapsed());

    fprintf(fp, "# Reconstruct time: %8.4f s\n", Reconstruct.Elapsed());
    // Expecting 6 Floats of load/store
    fprintf(fp, "#     FFT time:     %8.4f s, %6.3f Mcells/s, %6.3f GB/s, %6.3f GFLOPS/s\n",
        FFTonly.Elapsed(),
            nfft/1e6/FFTonly.Elapsed(),
            nfft/1e9/FFTonly.Elapsed()*6.0*sizeof(Float),
            nfft/1e6/FFTonly.Elapsed()*2.5*log(ngrid3)/log(2)/1e3);
#ifdef FFTSLAB
    fprintf(fp, "#        FFTyz time:%8.4f s, %6.3f Mcells/s, %6.3f GB/s, %6.3f GFLOPS/s\n",
        FFTyz.Elapsed(),
            nfft/1e6/FFTyz.Elapsed()*2.0/3.0,
            nfft/1e9/FFTyz.Elapsed()*6.0*sizeof(Float)*2.0/3.0,
            nfft/1e6/FFTyz.Elapsed()*2.5*log(ngrid3)/log(2)/1e3*2.0/3.0);
    fprintf(fp, "#        FFTx time: %8.4f s, %6.3f Mcells/s, %6.3f GB/s, %6.3f GFLOPS/s\n",
        FFTx.Elapsed(),
            nfft/1e6/FFTx.Elapsed()/3.0,
            nfft/1e9/FFTx.Elapsed()*6.0*sizeof(Float)/3.0,
            nfft/1e6/FFTx.Elapsed()*2.5*log(ngrid3)/log(2)/1e3/3.0);
#endif
    fprintf(fp, "#     Fourier Space time:   %8.4f s\n", FS.Elapsed());
    fprintf(fp, "#     Copy time:    %8.4f s\n", Copy.Elapsed());
    fprintf(fp, "#     map time:     %8.4f s, %6.3f Mparticles/sec, %6.2f MB/sec Read\n",
            fqts.Elapsed(), cnt/fqts.Elapsed()/1e6, cnt/fqts.Elapsed()*32.0/1e6);
    fprintf(fp, "#     CIC Grid time:%8.4f s, %6.3f Mparticles/sec, %6.2f GB/sec\n", CIC_2,
            cnt/CIC_2/1e6, 1e-9*cnt/CIC_2*27.0*2.0*sizeof(Float));
    fprintf(fp, "#        Sorting:   %8.4f s\n", Sorting_2);
    fprintf(fp, "#        Merging:   %8.4f s\n", Merging_2);
    fprintf(fp, "#        CIC:       %8.4f s\n", CIC_2-Merging_2-Sorting_2);
    fprintf(fp, "#     mu's time:    %8.4f s\n", mu.Elapsed());
#if defined (RECONST) && !defined (ITERATION)
    fprintf(fp, "# I/O for output time:  %8.4f s, %6.3f Mparticles/sec, %6.2f MB/sec Read\n",
            IO.Elapsed(), cnt/IO.Elapsed()/1e6, cnt/IO.Elapsed()*32.0/1e6);
    fprintf(fp, "# CIC Grid for shift time:    %8.4f s, %6.3f Mparticles/sec, %6.2f GB/sec\n",
            CIC.Elapsed(), cnt/CIC.Elapsed()/1e6, 1e-9*cnt/CIC.Elapsed()*27.0*2.0*sizeof(Float));
    fprintf(fp, "#        Sorting:   %8.4f s\n", Sorting.Elapsed());
    fprintf(fp, "#        Merging:   %8.4f s\n", Merging.Elapsed());
    fprintf(fp, "#        CIC:       %8.4f s\n", CIC.Elapsed()-Merging.Elapsed()-Sorting.Elapsed());
    fprintf(fp, "# Output time:      %8.4f s\n", Out.Elapsed());
#else
    fprintf(fp, "# Correlate time:   %8.4f s\n", Correlate.Elapsed());
    // Expecting 6 Floats of load/store
    fprintf(fp, "#      FFT time:    %8.4f s, %6.3f Mcells/s, %6.3f GB/s, %6.3f GFLOPS/s\n",
        FFTonly.Elapsed(),
            nfft/1e6/FFTonly.Elapsed(),
            nfft/1e9/FFTonly.Elapsed()*6.0*sizeof(Float),
            nfft/1e6/FFTonly.Elapsed()*2.5*log(ngrid3)/log(2)/1e3);
    #ifdef FFTSLAB
    fprintf(fp, "#        FFTyz time:%8.4f s, %6.3f Mcells/s, %6.3f GB/s, %6.3f GFLOPS/s\n",
        FFTyz.Elapsed(),
            nfft/1e6/FFTyz.Elapsed()*2.0/3.0,
            nfft/1e9/FFTyz.Elapsed()*6.0*sizeof(Float)*2.0/3.0,
            nfft/1e6/FFTyz.Elapsed()*2.5*log(ngrid3)/log(2)/1e3*2.0/3.0);
    fprintf(fp, "#        FFTx time: %8.4f s, %6.3f Mcells/s, %6.3f GB/s, %6.3f GFLOPS/s\n",
        FFTx.Elapsed(),
            nfft/1e6/FFTx.Elapsed()/3.0,
            nfft/1e9/FFTx.Elapsed()*6.0*sizeof(Float)/3.0,
            nfft/1e6/FFTx.Elapsed()*2.5*log(ngrid3)/log(2)/1e3/3.0);
    #endif
    // Approximating number of Ylm cells as FFT/2.
    // Each stores one float, but nearly all load one float too.
    fprintf(fp, "#     Ylm time:    %8.4f s, %6.3f GB/s\n", YlmTime.Elapsed(),
        (nfft-ngrid3)/2.0/1e9/YlmTime.Elapsed()*sizeof(Float)*2.0
            );
    fprintf(fp, "#     Hist time:   %8.4f s\n", Hist.Elapsed());
    fprintf(fp, "#     Extract time:%8.4f s\n", Extract.Elapsed());
    // We're doing two FFTs per loop and then one extra, so like 2*N+1
    // Hence N examples of A*Bt, each of which is 3 Floats of load/store
    fprintf(fp, "#     A*Bt time:   %8.4f s, %6.3f M/s of A=A*Bt, %6.3f GB/s\n", AtimesB.Elapsed(),
                    (nfft/2.0/ngrid3-0.5)*ngrid3/1e6/AtimesB.Elapsed(),
                    (nfft/2.0/ngrid3-0.5)*ngrid3/1e9/AtimesB.Elapsed()*3.0*sizeof(Float) );
#endif
    fprintf(fp, "# Total time:       %8.4f s\n", Total.Elapsed());
    if (Misc.Elapsed()>0.0) {
        fprintf(fp, "#\n# Misc time:          %8.4f s\n", Misc.Elapsed());
    }
    return;
}

class ThreadCount {
    int *cnt;
    int max;
  public:
    ThreadCount(int max_threads) {
        max = max_threads;
        int err=posix_memalign((void **) &cnt, PAGE, sizeof(int)*8*max); assert(err==0);
        for (int j=0; j<max*8; j++) cnt[j]=0;
        return;
    }
    ~ThreadCount() { free(cnt); return; }
    void add() {
        cnt[omp_get_thread_num()*8]++;
    }
    void print(FILE *fp) {
        for (int j=0; j<max; j++) if (cnt[j*8]>0)
            fprintf(fp, "# Thread %2d = %d\n", j, cnt[j*8]);
    }
};

#define MAX_THREADS 128
ThreadCount Ylm_count(MAX_THREADS);

void usage() {
    fprintf(stderr, "FFTCORR: Error in command-line \n");
    fprintf(stderr, "   -n <int> (or -ngrid): FFT linear grid size for a cubic box\n");
    fprintf(stderr, "   -n3 <int> <int> <int> (or -ngrid3): FFT linear grid sizes for rectangle\n");
    fprintf(stderr, "             -n3 will outrank -n\n");
    fprintf(stderr, "   -ell <int> (or -maxell): Multipole to compute.\n");
    fprintf(stderr, "   -b <float> (or -box): Bounding box size.  Must exceed value in input file.\n");
    fprintf(stderr, "             <0 will default to value in input file.\n");
    fprintf(stderr, "   -r <float> (or -sep): Max separation.  Cannot exceed value in input file.\n");
    fprintf(stderr, "             <0 will default to value in input file.\n");
    fprintf(stderr, "   -dr <float> (or -dsep): Binning of separation.\n");
    fprintf(stderr, "   -periodic (or -p): Configure for cubic periodic box.\n");
    fprintf(stderr, "   -in <filename>: Input file name\n");
    fprintf(stderr, "   -inini <filename>: Input file_ini name\n");
    fprintf(stderr, "   -inshifttruex <filename>: Input shift_true_x file name\n");
    fprintf(stderr, "   -inshiftinix <filename>: Input shift_ini_x file name\n");
    fprintf(stderr, "   -out <filename>: Output file name, default to stdout\n");
    fprintf(stderr, "   -outpower <filename>: Output file name for power spectrum (Data)\n");
    fprintf(stderr, "   -outpower2 <filename>: Output file name for power spectrum (Random)\n");
    fprintf(stderr, "   -outcorr <filename>: Output file name for correlations (Data)\n");
    fprintf(stderr, "   -outcorr2 <filename>: Output file name for correlations (Random)\n");
    fprintf(stderr, "   -outdens <filename>: Output file name for density contrast\n");
    fprintf(stderr, "   -outxcorrsh <filename>: Output file name for x-corr for shift\n");
    fprintf(stderr, "   -outxcorrde <filename>: Output file name for x-corr for density\n");
    fprintf(stderr, "\n");
    exit(1);
}


/* ==============================   Main   =============================== */

int main(int argc, char *argv[]) {
    // Need to get this information.
    Total.Start();
    // Here are some defaults
    int ngridCube = 256;
    int ngrid[3] = { -1, -1, -1};
    int maxell = 4;
    Float sep = -123.0;     // Default to max_sep from the file
    Float dsep = 10.0;
    Float kmax = 0.03;
    Float dk = 0.01;
    Float cell = -123.0;    // Default to what's implied by the file
    Float z_ave = 0.500;
    Float z_ini = 49.0;
    Float power_k0 = 10000.0;  // [Mpc/h]^3   along with DR12 [Ashley(2016)]
    Float Om_0 = 0.314153;     //   matter parameter, planck
    Float Ol_0 = 0.685847;     //   dark energy parameter, planck
    Float sig_sm = 10.0;       //  [Mpc/h]
    Float divi_sm = 1.2;       //  by which sig_sm is divided each time
    Float last_sm = 10.0;      //  to which the division is performed
    Float C_ani = 1.0;         //  the ratio of z-axis and x,y-axis
    Float f_eff = 1.0;         //  effective linear growth rate
    int ite_times = 9;         //  how many times do we iterate reconstruction
    int ite_switch = 0;        //  when switching to ite_weight_2
    Float ite_weight_ini = 0.5;//  [ini]   (next) = w1*(estimate) + (1-w1)*(previous)
    Float ite_weight_2 = 0.5;  //  [2nd]   (next) = w1*(estimate) + (1-w1)*(previous)
    Float bias = 1.00;         //  for matter field by default
    int qperiodic = 0;

    //const char default_fname[] = "/tmp/corrRR.dat";
                // for Daniel
    const char default_fname[] = "/mnt/store1/rhada/file_D";
                // for Ryuichiro
    char *infile = NULL;
    char *infile2 = NULL;
    char *infile_ini = NULL;
    char *infile_ini2 = NULL;
    char *shift_true_x = NULL;
    char *shift_true_y = NULL;
    char *shift_true_z = NULL;
    char *shift_ini_x = NULL;
    char *shift_ini_y = NULL;
    char *shift_ini_z = NULL;
    char *outfile = NULL;
    char *outfile_power = NULL;
    char *outfile_power2 = NULL;
    char *outfile_corr = NULL;
    char *outfile_corr2 = NULL;
    char *outfile_dens = NULL;
    char *outfile_xcorr_sh = NULL;
    char *outfile_xcorr_de = NULL;

    int i=1;
    while (i<argc) {
             if (!strcmp(argv[i],"-ngrid")||!strcmp(argv[i],"-n")) ngridCube = atoi(argv[++i]);
        else if (!strcmp(argv[i],"-ngrid3")||!strcmp(argv[i],"-n3")) {
                ngrid[0] = atoi(argv[++i]); ngrid[1] = atoi(argv[++i]); ngrid[2] = atoi(argv[++i]);
        }
        else if (!strcmp(argv[i],"-maxell")||!strcmp(argv[i],"-ell")) maxell = atoi(argv[++i]);
        else if (!strcmp(argv[i],"-sep")||!strcmp(argv[i],"-r")) sep = atof(argv[++i]);
        else if (!strcmp(argv[i],"-dsep")||!strcmp(argv[i],"-dr")) dsep = atof(argv[++i]);
        else if (!strcmp(argv[i],"-kmax")||!strcmp(argv[i],"-k")) kmax = atof(argv[++i]);
        else if (!strcmp(argv[i],"-dk")||!strcmp(argv[i],"-dk")) dk = atof(argv[++i]);
        else if (!strcmp(argv[i],"-cell")||!strcmp(argv[i],"-c")) cell = atof(argv[++i]);
        else if (!strcmp(argv[i],"-zave")||!strcmp(argv[i],"-z")) z_ave = atof(argv[++i]);
        else if (!strcmp(argv[i],"-zini")||!strcmp(argv[i],"-zi")) z_ini = atof(argv[++i]);
        else if (!strcmp(argv[i],"-powerk0")||!strcmp(argv[i],"-pk0")) power_k0 = atof(argv[++i]);
        else if (!strcmp(argv[i],"-omegam0")||!strcmp(argv[i],"-om0")) Om_0 = atof(argv[++i]);
        else if (!strcmp(argv[i],"-omegal0")||!strcmp(argv[i],"-ol0")) Ol_0 = atof(argv[++i]);
        else if (!strcmp(argv[i],"-sigmasm")||!strcmp(argv[i],"-ssm")) sig_sm = atof(argv[++i]);
        else if (!strcmp(argv[i],"-divism")||!strcmp(argv[i],"-dsm")) divi_sm = atof(argv[++i]);
        else if (!strcmp(argv[i],"-lastsm")||!strcmp(argv[i],"-lsm")) last_sm = atof(argv[++i]);
        else if (!strcmp(argv[i],"-cani")||!strcmp(argv[i],"-ca")) C_ani = atof(argv[++i]);
        else if (!strcmp(argv[i],"-feff")||!strcmp(argv[i],"-fe")) f_eff  = atof(argv[++i]);
        else if (!strcmp(argv[i],"-itetimes")||!strcmp(argv[i],"-itt")) ite_times  = atof(argv[++i]);
        else if (!strcmp(argv[i],"-iteswitch")||!strcmp(argv[i],"-its")) ite_switch  = atof(argv[++i]);
        else if (!strcmp(argv[i],"-itewini")||!strcmp(argv[i],"-itwi")) ite_weight_ini  = atof(argv[++i]);
        else if (!strcmp(argv[i],"-itew2")||!strcmp(argv[i],"-itw2")) ite_weight_2  = atof(argv[++i]);
        else if (!strcmp(argv[i],"-bias")||!strcmp(argv[i],"-bi")) bias  = atof(argv[++i]);
        else if (!strcmp(argv[i],"-in")||!strcmp(argv[i],"-i")) infile = argv[++i];
        else if (!strcmp(argv[i],"-in2")||!strcmp(argv[i],"-i2")) infile2 = argv[++i];
        else if (!strcmp(argv[i],"-inini")||!strcmp(argv[i],"-iini")) infile_ini = argv[++i];
        else if (!strcmp(argv[i],"-inini2")||!strcmp(argv[i],"-iini2")) infile_ini2 = argv[++i];
        else if (!strcmp(argv[i],"-inshifttruex")||!strcmp(argv[i],"-istx")) shift_true_x = argv[++i];
        else if (!strcmp(argv[i],"-inshifttruey")||!strcmp(argv[i],"-isty")) shift_true_y = argv[++i];
        else if (!strcmp(argv[i],"-inshifttruez")||!strcmp(argv[i],"-istz")) shift_true_z = argv[++i];
        else if (!strcmp(argv[i],"-inshiftinix")||!strcmp(argv[i],"-isix")) shift_ini_x = argv[++i];
        else if (!strcmp(argv[i],"-inshiftiniy")||!strcmp(argv[i],"-isiy")) shift_ini_y = argv[++i];
        else if (!strcmp(argv[i],"-inshiftiniz")||!strcmp(argv[i],"-isiz")) shift_ini_z = argv[++i];
        else if (!strcmp(argv[i],"-out")||!strcmp(argv[i],"-o")) outfile = argv[++i];
        else if (!strcmp(argv[i],"-outpower")||!strcmp(argv[i],"-op")) outfile_power = argv[++i];
        else if (!strcmp(argv[i],"-outpower2")||!strcmp(argv[i],"-op2")) outfile_power2 = argv[++i];
        else if (!strcmp(argv[i],"-outcorr")||!strcmp(argv[i],"-oc")) outfile_corr = argv[++i];
        else if (!strcmp(argv[i],"-outcorr2")||!strcmp(argv[i],"-oc2")) outfile_corr2 = argv[++i];
        else if (!strcmp(argv[i],"-outdens")||!strcmp(argv[i],"-od")) outfile_dens = argv[++i];
        else if (!strcmp(argv[i],"-outxcorrsh")||!strcmp(argv[i],"-oxcs")) outfile_xcorr_sh = argv[++i];
        else if (!strcmp(argv[i],"-outxcorrde")||!strcmp(argv[i],"-oxcd")) outfile_xcorr_de = argv[++i];
        else if (!strcmp(argv[i],"-periodic")||!strcmp(argv[i],"-p")) qperiodic = 1;
        else usage();
        i++;
    }

    assert(ngrid>0);
    assert(maxell>=0 && maxell%2==0);
    assert(sep!=0.0);
    assert(dsep>0.0);
    assert(kmax!=0.0);
    assert(dk>0.0);
    assert(qperiodic==0||sep>0);  // If qperiodic is set, user must supply a sep
    if (infile==NULL) infile = (char *)default_fname;
    if (outfile!=NULL) { FILE *discard=freopen(outfile,"w", stdout);
                    assert(discard!=NULL&&stdout!=NULL); }
    if (ngrid[0]<=0) ngrid[0] = ngrid[1] = ngrid[2] = ngridCube;
    assert(ngrid[0]>0);
    assert(ngrid[1]>0);
    assert(ngrid[2]>0);

    lgf = lg_factor(Om_0, Ol_0, z_ave)/lg_factor(Om_0, Ol_0, z_ini);
    #ifdef RSD
      lgr_f = lg_rate(Om_0, Ol_0, z_ave)*f_eff;
    #else
      lgr_f = 0.0;
    #endif
    lgr_beta = lgr_f/bias;
    shift_2nd = 3.0/7.0*pow(Omf(Om_0, Ol_0, z_ave), -(1.0/143.0));
    fprintf(stderr, "linear growth factor = %f\n", lgf);
    fprintf(stderr, "linear growth rate = %f\n", lgr_f);

    #ifdef OPENMP
        omp_set_num_threads(10);    // limit the number of threads for alan and gosling
        fprintf(stdout,"# Running with %d threads\n", omp_get_max_threads());
    #else
        fprintf(stdout,"# Running single threaded.\n");
    #endif

    setup_wavelet();

    //#if defined XCORR_SHIFT && !defined FROM_DENSITY && !defined RECONST
      //Grid g(shift_true_x, ngrid, cell, sep, qperiodic);
    //#else
      Grid g(infile, ngrid, cell, sep, qperiodic);
      // output = 0: add_value_to_grid
      // type = 0  : dens
      g.read_galaxies(infile, infile2, NULL, 0, 0);
    //#endif

    IO_1 = IO.Elapsed();
    IO.Clear();
    CIC_1 = CIC.Elapsed();
    CIC.Clear();
    Sorting_1 = Sorting.Elapsed();
    Sorting.Clear();
    Merging_1 = Merging.Elapsed();
    Merging.Clear();

    // The input grid is now in g.dens
    sep = g.setup_corr(sep, kmax);

    #ifdef RECONST
      if(infile2!=NULL || shift_true_x!=NULL){
        #ifndef ITERATION
          g.reconstruct(power_k0,sig_sm,bias);
        #else
          g.reconstruct_iterate(power_k0,sig_sm,divi_sm,last_sm,C_ani,
                                    ite_times,ite_switch,ite_weight_ini,ite_weight_2,bias);
        #endif
      }
    #endif

    CIC_2 = CIC.Elapsed();
    CIC.Clear();
    Sorting_2 = Sorting.Elapsed();
    Sorting.Clear();
    Merging_2 = Merging.Elapsed();
    Merging.Clear();

    #if defined (RECONST) && !defined (ITERATION)
      /* ------- STANDARD -------- */

      // output = 1: get_value_from_grid
      // type = 0  : shift
      g.read_galaxies(infile, infile2, NULL, 1, 0);

      // rename infile and infile2
      size_t len = strlen(infile) + strlen("_rec");
      size_t len2 = strlen(infile2) + strlen("_rec");;
      char *infile_rec = (char *)malloc(len + 1);
      char *infile2_rec = (char *)malloc(len2 + 1);
      infile_rec[len] = 0;
      strcpy(infile_rec, infile);
      strcat(infile_rec, "_rec");
      infile_rec[len2] = 0;
      strcpy(infile2_rec, infile2);
      strcat(infile2_rec, "_rec");
      infile = infile_rec;
      infile2 = infile2_rec;

      set_matrix(g.dens, 0.0, g.ngrid3, g.ngrid[0]);
      // output = 0: add_value_to_grid
      // type = 0  : dens
      g.read_galaxies(infile, infile2, NULL, 0, 0);
    #endif // STANDARD

    FFTonly_1 = FFTonly.Elapsed();
    FFTonly.Clear();
    FFTyz_1 = FFTyz.Elapsed();
    FFTyz.Clear();
    FFTx_1 = FFTx.Elapsed();
    FFTx.Clear();

    // density contrast
    if (outfile_dens!=NULL){
        FILE *o_d=freopen(outfile_dens,"w",stdout);
        assert(o_d!=NULL);
        #ifdef RSD
          g.print_dens(stdout,2, 2);
        #else
          g.print_dens(stdout,0, 0);
        #endif
        if (outfile!=NULL) { FILE *discard=freopen(outfile,"a", stdout);
                        assert(discard!=NULL&&stdout!=NULL); }
        else{ FILE *tem=freopen("/dev/tty", "a", stdout);
                        assert(tem!=NULL); }
    }

    /* ------- DENSE -------- */
    for(int i=0; i<2; i++){    // i = 0: Data,  1: Random
      Histogram h(maxell, sep, dsep);
      Histogram kh(maxell, kmax, dk);

      if(i==0){  // i = 0
      #ifndef INITIAL
        copy_matrix(g.dens, g.dens, 1/bias, g.ngrid3, g.ngrid[0]);
      #else
        copy_matrix(g.dens, g.dens, lgf, g.ngrid3, g.ngrid[0]);
      #endif
    }
      g.correlate(maxell, h, kh);

      // Output power spectrum and correlations
      Ylm_count.print(stdout);
      // power spectrum
      if (i==0) {
        fprintf(stdout, "# Anisotropic power spectrum (Data):\n");
      }
      else{
        outfile_power = outfile_power2;
        fprintf(stdout, "# Anisotropic power spectrum (Random):\n");
      }
      if (outfile_power!=NULL){
          fprintf(stdout, "# ... is in file: %s\n", outfile_power);
          FILE *o_p=freopen(outfile_power,"w",stdout);
          assert(o_p!=NULL);
      }
      kh.print(stdout,1);
      if (outfile!=NULL) { FILE *discard=freopen(outfile,"a", stdout);
                      assert(discard!=NULL&&stdout!=NULL); }
      else{ FILE *tem=freopen("/dev/tty", "a", stdout);
                      assert(tem!=NULL); }
      // correlations
      if (i==0) {
        fprintf(stdout, "# Anisotropic correlations (Data):\n");
      }
      else{
        outfile_corr = outfile_corr2;
        fprintf(stdout, "# Anisotropic correlations (Random):\n");
      }
      if (outfile_corr!=NULL){
          fprintf(stdout, "# ... are in file: %s\n", outfile_corr);
          FILE *o_c=freopen(outfile_corr,"w",stdout);
          assert(o_c!=NULL);
      }
      h.print(stdout,0);
      if (outfile!=NULL) { FILE *discard=freopen(outfile,"a", stdout);
                      assert(discard!=NULL&&stdout!=NULL); }
      else{ FILE *tem=freopen("/dev/tty", "a", stdout);
                      assert(tem!=NULL); }

      // We want to use the correlation at zero lag as the I normalization
      // factor in the FKP power spectrum.
      fprintf(stdout, "#\n# Zero-lag correlations are %14.7e\n", h.zerolag);
      // Integral of power spectrum needs a d^3k/(2 pi)^3, which is (1/L)^3 = (1/(cell_size*ngrid))^3
      fprintf(stdout, "#\n# Integral of power spectrum is %14.7e\n",
              kh.sum()/(g.cell_size*g.cell_size*g.cell_size*ngrid[0]*ngrid[1]*ngrid[2]));
      if(i==0){

        /* ------- XCORR_SHIFT -------- */
        #ifdef XCORR_SHIFT
          #ifndef DIFF_SHIFT
            /* ----- normal X-corr for shift fields ----- */
            // output = 0: add_value_to_grid
            // type = 1  : shift_t
            g.read_galaxies(shift_true_x, shift_true_y, shift_true_z, 0, 1);
            // output = 0: add_value_to_grid
            // type = 2  : shift
            g.read_galaxies(shift_ini_x, shift_ini_y, shift_ini_z, 0, 2);

            Histogram w_kh_n(0, kmax, dk);
            Histogram w_kh_d(0, kmax, dk);
            Histogram w_kh_e(0, kmax, dk);
            g.xcorr_shift(w_kh_n, w_kh_d, w_kh_e, power_k0,bias);

            // X-corr for shift
            if (outfile_xcorr_sh!=NULL){
                FILE *o_w=freopen(outfile_xcorr_sh,"w",stdout);
                assert(o_w!=NULL);
                g.print_kcorr(stdout, w_kh_n, w_kh_d, w_kh_e, 1);
                if (outfile!=NULL) { FILE *discard=freopen(outfile,"a", stdout);
                                assert(discard!=NULL&&stdout!=NULL); }
                else{ FILE *tem=freopen("/dev/tty", "a", stdout);
                                assert(tem!=NULL); }
            }
          #else
            /* ----- sum of the square of difference between shift_t_X and reconstructed shift ----- */
            // output = 2: return_value_from_grid
            // type = 0  : shift_t
            g.read_galaxies(shift_true_x, shift_true_y, shift_true_z, 2, 0);

            uint64 num_gal;
            num_gal = g.cnt/3;

            Float sumsq_dif_all;
            sumsq_dif_all = (g.sumsq_dif_x + g.sumsq_dif_x + g.sumsq_dif_x)/num_gal;

            // X-corr for shift
            if (outfile_xcorr_sh!=NULL){
                FILE *o_w=freopen(outfile_xcorr_sh,"w",stdout);
                assert(o_w!=NULL);

                fprintf(stdout,"\n#### Number of shift_t_X = %lld\n", num_gal);
                fprintf(stdout,"#### Average of the square of diferrence = %e\n", sumsq_dif_all);

                if (outfile!=NULL) { FILE *discard=freopen(outfile,"a", stdout);
                                assert(discard!=NULL&&stdout!=NULL); }
                else{ FILE *tem=freopen("/dev/tty", "a", stdout);
                                assert(tem!=NULL); }
            }
          #endif
        #endif // XCORR_SHIFT

        /* ------- XCORR_DENSITY -------- */
        #ifdef XCORR_DENSITY
          copy_matrix(g.dens_now, g.dens, g.ngrid3, g.ngrid[0]);
          set_matrix(g.dens, 0.0, g.ngrid3, g.ngrid[0]);
          // output = 0: add_value_to_grid
          // type = 0  : dens
          g.read_galaxies(infile_ini, infile_ini2, NULL, 0, 0);

          Histogram x_kh_n(0, kmax, dk);
          Histogram x_kh_d(0, kmax, dk);
          Histogram x_kh_e(0, kmax, dk);
          g.xcorr_dens(x_kh_n, x_kh_d, x_kh_e, power_k0,bias);

          // X-corr for density
          if (outfile_xcorr_de!=NULL){
              FILE *o_x=freopen(outfile_xcorr_de,"w",stdout);
              assert(o_x!=NULL);
              g.print_kcorr(stdout, x_kh_n, x_kh_d, x_kh_e, 1);
              if (outfile!=NULL) { FILE *discard=freopen(outfile,"a", stdout);
                              assert(discard!=NULL&&stdout!=NULL); }
              else{ FILE *tem=freopen("/dev/tty", "a", stdout);
                              assert(tem!=NULL); }
          }
        #endif // XCORR_DENSITY

        set_matrix(g.dens, 0.0, g.ngrid3, g.ngrid[0]);
        // output = 0: add_value_to_grid
        // type = 0  : dens
        g.read_galaxies(NULL, infile2, NULL, 0, 0);
    }
  }

    Total.Stop();
    uint64 nfft=1;
    for (int j=0; j<=maxell; j+=2) nfft+=2*(2*j+1);
    nfft*=g.ngrid3;
    fprintf(stdout,"#\n");
    ReportTimes(stdout, nfft, g.ngrid3, g.cnt);
    return 0;
}


/* ============== Spherical Harmonic routine ============== */

  void makeYlm(Float *Ylm, int ell, int m, int n[3], int n1, Float *xcell, Float *ycell, Float *z, Float *dens) {
    // We're not actually returning Ylm here.
    // m>0 will return Re(Y_lm)*sqrt(2)
    // m<0 will return Im(Y_l|m|)*sqrt(2)
    // m=0 will return Y_l0
    // These are not including common minus signs, since we're only
    // using them in matched squares.
    //
    // Input x[n[0]], y[n[1]], z[n[2]] are the x,y,z centers of this row of bins
    // n1 is supplied so that Ylm[n[0]][n[1]][n1] can be handled flexibly
    //
    // If dens!=NULL, then it should point to a [n[0]][n[1]][n1] vector that will be multiplied
    // element-wise onto the results.  This can save a store/load to main memory.
    YlmTime.Start();
    Float isqpi = sqrt(1.0/M_PI);
    if (m!=0) isqpi *= sqrt(2.0);    // Do this up-front, so we don't forget
    Float tiny = 1e-20;

    const uint64 nc3 = (uint64)n[0]*n[1]*n1;
    if (ell==0&&m==0) {
        // This case is so easy that we'll do it first and skip the rest of the set up
        if (dens==NULL) set_matrix(Ylm, 1.0/sqrt(4.0*M_PI), nc3, n[0]);
                  else copy_matrix(Ylm, dens, 1.0/sqrt(4.0*M_PI), nc3, n[0]);
        YlmTime.Stop();
        return;
    }

    const int cn2 = n[2];    // To help with loop vectorization
    Float *z2, *z3, *z4, *ones;
    int err=posix_memalign((void **) &z2, PAGE, sizeof(Float)*n[2]+PAGE); assert(err==0);
    err=posix_memalign((void **) &z3, PAGE, sizeof(Float)*n[2]+PAGE); assert(err==0);
    err=posix_memalign((void **) &z4, PAGE, sizeof(Float)*n[2]+PAGE); assert(err==0);
    err=posix_memalign((void **) &ones, PAGE, sizeof(Float)*n[2]+PAGE); assert(err==0);
    for (int k=0; k<cn2; k++) {
        z2[k] = z[k]*z[k];
        z3[k] = z2[k]*z[k];
        z4[k] = z3[k]*z[k];
        ones[k] = 1.0;
    }

    Ylm[0] = -123456.0;    // A sentinal value

    #pragma omp parallel for YLM_SCHEDULE
    for (uint64 i=0; i<n[0]; i++) {
            Ylm_count.add();
        Float *ir2;    // Need some internal workspace
        err=posix_memalign((void **) &ir2, PAGE, sizeof(Float)*n[2]+PAGE); assert(err==0);
        Float x = xcell[i], x2 = x*x;
        Float *Y = Ylm+i*n[1]*n1;
        Float *D = dens+i*n[1]*n1;
        for (int j=0; j<n[1]; j++, Y+=n1, D+=n1) {
            if (dens==NULL) D=ones;
            Float y = ycell[j], y2 = y*y, y3 = y2*y, y4 = y3*y;
            for (int k=0; k<cn2; k++) ir2[k] = 1.0/(x2+y2+z2[k]+tiny);
            if (ell==2) {
                if (m==2)
                    for (int k=0; k<cn2; k++)
                        Y[k] = D[k]*isqpi*sqrt(15./32.)*(x2-y2)*ir2[k];
                else if (m==1)
                    for (int k=0; k<cn2; k++)
                        Y[k] = D[k]*isqpi*sqrt(15./8.)*x*z[k]*ir2[k];
                else if (m==0)
                    for (int k=0; k<cn2; k++)
                        Y[k] = D[k]*isqpi*sqrt(5./16.)*(2.0*z2[k]-x2-y2)*ir2[k];
                else if (m==-1)
                    for (int k=0; k<cn2; k++)
                        Y[k] = D[k]*isqpi*sqrt(15./8.)*y*z[k]*ir2[k];
                else if (m==-2)
                    for (int k=0; k<cn2; k++)
                        Y[k] = D[k]*isqpi*sqrt(15./32.)*2.0*x*y*ir2[k];
            }
            else if (ell==4) {
                if (m==4)
                    for (int k=0; k<cn2; k++)
                    Y[k] = D[k]*isqpi*3.0/16.0*sqrt(35./2.)*(x2*x2-6.0*x2*y2+y4)*ir2[k]*ir2[k];
                else if (m==3)
                    for (int k=0; k<cn2; k++)
                    Y[k] = D[k]*isqpi*3.0/8.0*sqrt(35.)*(x2-3.0*y2)*z[k]*x*ir2[k]*ir2[k];
                else if (m==2)
                    for (int k=0; k<cn2; k++)
                    Y[k] = D[k]*isqpi*3.0/8.0*sqrt(5./2.)*(6.0*z2[k]*(x2-y2)-x2*x2+y4)*ir2[k]*ir2[k];
                else if (m==1)
                    for (int k=0; k<cn2; k++)
                    Y[k] = D[k]*isqpi*3.0/8.0*sqrt(5.)*3.0*(4.0/3.0*z2[k]-x2-y2)*x*z[k]*ir2[k]*ir2[k];
                else if (m==0)
                    for (int k=0; k<cn2; k++)
                    Y[k] = D[k]*isqpi*3.0/16.0*8.0*(z4[k]-3.0*z2[k]*(x2+y2)+3.0/8.0*(x2*x2+2.0*x2*y2+y4))*ir2[k]*ir2[k];
                else if (m==-1)
                    for (int k=0; k<cn2; k++)
                    Y[k] = D[k]*isqpi*3.0/8.0*sqrt(5.)*3.0*(4.0/3.0*z2[k]-x2-y2)*y*z[k]*ir2[k]*ir2[k];
                else if (m==-2)
                    for (int k=0; k<cn2; k++)
                    Y[k] = D[k]*isqpi*3.0/8.0*sqrt(5./2.)*(2.0*x*y*(6.0*z2[k]-x2-y2))*ir2[k]*ir2[k];
                else if (m==-3)
                    for (int k=0; k<cn2; k++)
                    Y[k] = D[k]*isqpi*3.0/8.0*sqrt(35.)*(3.0*x2*y-y3)*z[k]*ir2[k]*ir2[k];
                else if (m==-4)
                    for (int k=0; k<cn2; k++)
                    Y[k] = D[k]*isqpi*3.0/16.0*sqrt(35./2.)*(4.0*x*(x2*y-y3))*ir2[k]*ir2[k];
            }
        }
        free(ir2);
    }
    assert(Ylm[0]!=123456.0);  // This traps whether the user entered an illegal (ell,m)
    free(z2);
    free(z3);
    free(z4);
    YlmTime.Stop();
    return;
}
